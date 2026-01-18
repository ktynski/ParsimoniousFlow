"""
Curiosity / Active Learning Module
==================================

Theory-derived curiosity mechanisms for autonomous learning.

CORE INSIGHT:
    Curiosity is NOT a separate mechanism — it's the GRADIENT of existing
    computations applied in reverse:
    
    curiosity(query) = -∇_query [ grace_stability(retrieve(query)) ]
    
    The system descends toward queries where stability is lowest.
    This is already computable from existing operations!

THEORY DERIVATION:
    1. Grace stability σ(M) = witness_energy / total_energy
       - High σ → known, stable, certain
       - Low σ → unknown, unstable, uncertain
       
    2. Basin coverage measures how much of query space is "owned"
       - Edges of basins = boundaries of knowledge
       
    3. Curiosity = descend stability gradient + sample basin boundaries
       - Seeks information that would INCREASE overall stability
       
    4. Information gain = predicted stability increase from learning sample

IMPLEMENTATION:
    - curiosity_score(query) → how curious should we be about this?
    - estimate_information_gain(sample) → how much would learning this help?
    - sample_basin_boundary() → where does our knowledge end?
    - generate_curiosity_query() → what should we ask about?
    - active_learning_step() → which sample from pool should we learn?
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE
from holographic_v4.algebra import (
    build_clifford_basis,
    grace_operator,
    geometric_product,
    frobenius_similarity,
    decompose_to_coefficients,
)
from holographic_v4.quotient import grace_stability, extract_witness

Array = np.ndarray


# =============================================================================
# CURIOSITY SCORE
# =============================================================================

def curiosity_score(
    query: Array,
    model: 'TheoryTrueModel',
    dreaming: 'DreamingSystem',
) -> float:
    """
    Compute curiosity score for a query.
    
    THEORY:
        Curiosity = inverse of confidence/stability for this query.
        High curiosity means: "I don't know this well, I should learn it."
        
        Components:
        1. Low retrieval confidence → high curiosity
        2. Low grace stability → high curiosity
        3. Far from any basin center → high curiosity
        
    Args:
        query: [4, 4] Clifford matrix representing the query
        model: TheoryTrueModel
        dreaming: DreamingSystem with semantic memory
        
    Returns:
        Curiosity score [0, ∞) — higher means more curious
    """
    xp = model.xp
    basis = model.basis
    
    # 1. Grace stability of query itself
    query_stability = grace_stability(query, basis, xp)
    
    # 2. Distance to nearest prototype (if any)
    min_proto_sim = -10.0  # Default: very far
    if hasattr(dreaming, 'semantic_memory') and dreaming.semantic_memory is not None:
        # SemanticMemory has .levels which is a list of prototype lists
        if hasattr(dreaming.semantic_memory, 'levels'):
            for level in dreaming.semantic_memory.levels:
                for proto in level:
                    if hasattr(proto, 'centroid'):
                        sim = frobenius_similarity(query, proto.centroid, xp)
                        min_proto_sim = max(min_proto_sim, float(sim))
    
    # 3. Retrieval attempt (if model has attractors)
    retrieval_conf = 0.0
    if model.num_attractors > 0:
        # Try to find similar attractor
        query_witness = extract_witness(query, basis, xp)
        best_sim = -10.0
        n_sample = min(model.num_attractors, 100)
        for i in range(n_sample):
            attractor = model.attractor_matrices[i]
            att_witness = extract_witness(attractor, basis, xp)
            sim = query_witness[0] * att_witness[0] + query_witness[1] * att_witness[1]
            best_sim = max(best_sim, sim)
        retrieval_conf = max(0.0, (best_sim + 10) / 20)  # Normalize to ~[0,1]
    
    # Combine: curiosity is HIGH when stability/confidence/similarity are LOW
    # THEORY-TRUE: φ-power decay (NOT arbitrary 1/(1+x))
    # High stability → low curiosity via φ^(-stability)
    stability_term = PHI_INV ** query_stability  # φ⁻ˢ: high stability → low term
    proto_term = PHI_INV ** min_proto_sim        # φ⁻ˢⁱᵐ: high similarity → low term
    conf_term = 1.0 - retrieval_conf             # Linear: high confidence → low term
    
    # Weighted combination (φ-derived weights)
    curiosity = (
        PHI * stability_term +      # Stability most important
        PHI_INV * proto_term +      # Prototype distance
        PHI_INV_SQ * conf_term      # Retrieval confidence
    )
    
    return float(curiosity)


def curiosity_score_with_meta(
    query: Array,
    model: 'TheoryTrueModel',
    dreaming: 'DreamingSystem',
    meta_state: 'LearningState',
) -> float:
    """
    Curiosity score modulated by meta-learning state.
    
    When overall uncertainty is HIGH, be more cautious (lower curiosity).
    When overall uncertainty is LOW, explore more (higher curiosity).
    
    Args:
        query: [4, 4] Clifford matrix
        model: TheoryTrueModel
        dreaming: DreamingSystem
        meta_state: Current learning state
        
    Returns:
        Modulated curiosity score
    """
    base_curiosity = curiosity_score(query, model, dreaming)
    
    # Modulate by uncertainty
    # High uncertainty → be cautious → reduce curiosity
    # Low uncertainty → explore → increase curiosity
    uncertainty = meta_state.uncertainty
    
    # Modulation factor: low uncertainty → factor > 1, high → factor < 1
    modulation = 1.0 + (0.5 - uncertainty)  # Range: [0.5, 1.5]
    
    return base_curiosity * modulation


# =============================================================================
# INFORMATION GAIN
# =============================================================================

def estimate_information_gain(
    context: List[int],
    target: int,
    model: 'TheoryTrueModel',
    dreaming: 'DreamingSystem',
) -> float:
    """
    Estimate information gain from learning a sample.
    
    THEORY:
        Information gain = expected increase in overall stability.
        A sample is valuable if:
        1. It's in an uncertain region (high curiosity)
        2. It's not redundant with existing knowledge
        3. It connects to existing knowledge (not completely alien)
        
    Args:
        context: Token context
        target: Target token
        model: TheoryTrueModel
        dreaming: DreamingSystem
        
    Returns:
        Estimated information gain [0, ∞)
    """
    xp = model.xp
    basis = model.basis
    
    # Compose context
    ctx_matrix = model.compute_context(context)
    
    # 1. Curiosity about this region
    cur = curiosity_score(ctx_matrix, model, dreaming)
    
    # 2. Novelty: how different is target from what we'd predict?
    novelty = 1.0
    if model.num_attractors > 0:
        _, predicted = model.retrieve(context)  # Returns (matrix, target_idx)
        if predicted > 0 and predicted != target:
            # Different prediction → novel information
            pred_emb = model.get_embedding(predicted)
            actual_emb = model.get_embedding(target)
            diff = xp.linalg.norm(pred_emb - actual_emb)
            novelty = min(2.0, float(diff))
        elif predicted == target:
            # We already know this → less novel (φ⁻³ residual novelty)
            novelty = PHI_INV_CUBE  # 0.236
    
    # 3. Connectivity: does target connect to existing knowledge?
    # THEORY-TRUE: Default connectivity is φ⁻¹ (moderate connection assumed)
    connectivity = PHI_INV  # 0.618
    target_emb = model.get_embedding(target)
    target_witness = extract_witness(target_emb, basis, xp)
    
    if hasattr(dreaming, 'semantic_memory') and dreaming.semantic_memory is not None:
        # Check if target is near any prototype
        max_sim = -10.0
        if hasattr(dreaming.semantic_memory, 'levels'):
            for level in dreaming.semantic_memory.levels:
                for proto in level:
                    if hasattr(proto, 'centroid'):
                        proto_witness = extract_witness(proto.centroid, basis, xp)
                        sim = target_witness[0] * proto_witness[0] + target_witness[1] * proto_witness[1]
                        max_sim = max(max_sim, sim)
            if max_sim > -10.0:
                connectivity = max(0.0, min(1.0, (max_sim + 1) / 2))
    
    # THEORY-TRUE: Combine with φ-weighted modulation
    # High gain if curious AND novel AND connected
    gain = cur * novelty * (PHI_INV_SQ + connectivity)  # φ⁻² + connectivity
    
    return float(gain)


# =============================================================================
# BASIN BOUNDARY SAMPLING
# =============================================================================

def sample_basin_boundary(
    model: 'TheoryTrueModel',
    dreaming: 'DreamingSystem',
    n_candidates: int = 50,
    seed: int = None,
) -> Array:
    """
    Sample from the boundaries of knowledge basins.
    
    THEORY:
        Basin boundaries are where our knowledge ends.
        These are regions of INTERMEDIATE stability — not fully known,
        but not completely unknown either.
        
        Method: Generate random queries, keep ones with intermediate curiosity.
        
    Args:
        model: TheoryTrueModel
        dreaming: DreamingSystem
        n_candidates: Number of random candidates to consider
        seed: Random seed
        
    Returns:
        [4, 4] Clifford matrix from basin boundary
    """
    xp = model.xp
    basis = model.basis
    rng = np.random.default_rng(seed)
    
    best_candidate = None
    best_score = -float('inf')
    
    # Target intermediate curiosity (not too low, not too high)
    target_curiosity = PHI  # Golden ratio as target
    
    for _ in range(n_candidates):
        # Generate random query
        tokens = rng.integers(0, model.vocab_size, size=model.context_size).tolist()
        candidate = model.compute_context(tokens)
        
        # Score it
        cur = curiosity_score(candidate, model, dreaming)
        
        # Prefer intermediate curiosity (boundary)
        # Score is negative distance from target
        score = -abs(cur - target_curiosity)
        
        if score > best_score:
            best_score = score
            best_candidate = candidate.copy()
    
    # CRITICAL: No fallback - if we can't find a curiosity candidate, that's a failure
    if best_candidate is None:
        raise RuntimeError(
            f"curiosity_search failed: No valid candidate found after {n_candidates} attempts. "
            "This indicates a bug in curiosity scoring or candidate generation."
        )
    
    return best_candidate


# =============================================================================
# CURIOSITY QUERY GENERATION
# =============================================================================

def generate_curiosity_query(
    model: 'TheoryTrueModel',
    dreaming: 'DreamingSystem',
    n_candidates: int = 30,
    strategy: str = 'max_curiosity',
    seed: int = None,
) -> Array:
    """
    Generate a query that maximizes curiosity / information potential.
    
    THEORY:
        The system should ask about what it knows LEAST about.
        This is the query that would maximize information gain.
        
    Strategies:
        'max_curiosity': Maximize raw curiosity score
        'boundary': Sample from basin boundaries
        'gradient': Follow negative stability gradient
        
    Args:
        model: TheoryTrueModel
        dreaming: DreamingSystem
        n_candidates: Number of candidates to evaluate
        strategy: Selection strategy
        seed: Random seed
        
    Returns:
        [4, 4] Clifford matrix query
    """
    xp = model.xp
    basis = model.basis
    rng = np.random.default_rng(seed)
    
    if strategy == 'boundary':
        return sample_basin_boundary(model, dreaming, n_candidates, seed)
    
    elif strategy == 'gradient':
        # Start from a known point and move toward uncertainty
        if model.num_attractors > 0:
            # Start from random attractor
            start_idx = rng.integers(0, model.num_attractors)
            start = None
            for i, (k, v) in enumerate(model.attractor_map.items()):
                if i == start_idx:
                    start = v.copy()
                    break
            if start is None:
                tokens = rng.integers(0, model.vocab_size, size=model.context_size).tolist()
                start = model.compute_context(tokens)
        else:
            tokens = rng.integers(0, model.vocab_size, size=model.context_size).tolist()
            start = model.compute_context(tokens)
        
        # Add noise to move toward uncertainty
        current = start.copy()
        for _ in range(5):  # 5 gradient steps
            noise = rng.standard_normal((4, 4)) * 0.1
            perturbed = current + noise
            perturbed = grace_operator(perturbed, basis, xp)
            
            # Keep if more curious
            if curiosity_score(perturbed, model, dreaming) > curiosity_score(current, model, dreaming):
                current = perturbed
        
        return current
    
    else:  # max_curiosity
        best_query = None
        best_curiosity = -float('inf')
        
        for _ in range(n_candidates):
            tokens = rng.integers(0, model.vocab_size, size=model.context_size).tolist()
            candidate = model.compute_context(tokens)
            cur = curiosity_score(candidate, model, dreaming)
            
            if cur > best_curiosity:
                best_curiosity = cur
                best_query = candidate.copy()
        
        if best_query is None:
            tokens = rng.integers(0, model.vocab_size, size=model.context_size).tolist()
            best_query = model.compute_context(tokens)
        
        return best_query


# =============================================================================
# ACTIVE LEARNING
# =============================================================================

def active_learning_step(
    model: 'TheoryTrueModel',
    dreaming: 'DreamingSystem',
    sample_pool: List[Tuple[List[int], int]],
    n_evaluate: int = None,
) -> Tuple[List[int], int]:
    """
    Select the best sample to learn from a pool.
    
    THEORY:
        Optimal learning = select sample with highest information gain.
        This is the sample that would most reduce overall uncertainty.
        
    Args:
        model: TheoryTrueModel
        dreaming: DreamingSystem
        sample_pool: List of (context, target) pairs
        n_evaluate: Max samples to evaluate (None = all)
        
    Returns:
        (context, target) of selected sample
    """
    if not sample_pool:
        raise ValueError("Empty sample pool")
    
    n_evaluate = n_evaluate or len(sample_pool)
    n_evaluate = min(n_evaluate, len(sample_pool))
    
    best_sample = sample_pool[0]
    best_gain = -float('inf')
    
    # Evaluate samples
    for i in range(n_evaluate):
        ctx, target = sample_pool[i]
        gain = estimate_information_gain(ctx, target, model, dreaming)
        
        if gain > best_gain:
            best_gain = gain
            best_sample = (ctx, target)
    
    return best_sample


def rank_samples_by_curiosity(
    model: 'TheoryTrueModel',
    dreaming: 'DreamingSystem',
    sample_pool: List[Tuple[List[int], int]],
) -> List[Tuple[float, List[int], int]]:
    """
    Rank samples by information gain.
    
    Args:
        model: TheoryTrueModel
        dreaming: DreamingSystem
        sample_pool: List of (context, target) pairs
        
    Returns:
        List of (gain, context, target) sorted by gain descending
    """
    ranked = []
    for ctx, target in sample_pool:
        gain = estimate_information_gain(ctx, target, model, dreaming)
        ranked.append((gain, ctx, target))
    
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


# =============================================================================
# EXPLORATION STRATEGIES
# =============================================================================

def should_explore(
    model: 'TheoryTrueModel',
    dreaming: 'DreamingSystem',
    threshold: float = PHI,
) -> bool:
    """
    Should the system explore vs exploit?
    
    THEORY:
        Explore when overall curiosity is high (knowledge is sparse).
        Exploit when curiosity is low (knowledge is dense).
        
    Args:
        model: TheoryTrueModel
        dreaming: DreamingSystem
        threshold: Curiosity threshold for exploration
        
    Returns:
        True if should explore, False if should exploit
    """
    # Sample random queries and measure mean curiosity
    xp = model.xp
    rng = np.random.default_rng()
    
    curiosities = []
    for _ in range(10):
        tokens = rng.integers(0, model.vocab_size, size=model.context_size).tolist()
        query = model.compute_context(tokens)
        curiosities.append(curiosity_score(query, model, dreaming))
    
    mean_curiosity = np.mean(curiosities)
    return mean_curiosity > threshold


def exploration_rate(
    model: 'TheoryTrueModel',
    dreaming: 'DreamingSystem',
) -> float:
    """
    Compute recommended exploration rate ε for ε-greedy.
    
    THEORY:
        Rate scales with mean curiosity:
        - High curiosity → high ε (explore more)
        - Low curiosity → low ε (exploit more)
        
    Returns:
        Exploration rate in [0.1, 0.9]
    """
    xp = model.xp
    rng = np.random.default_rng()
    
    curiosities = []
    for _ in range(10):
        tokens = rng.integers(0, model.vocab_size, size=model.context_size).tolist()
        query = model.compute_context(tokens)
        curiosities.append(curiosity_score(query, model, dreaming))
    
    mean_curiosity = np.mean(curiosities)
    
    # THEORY-TRUE: Map curiosity to rate using φ-derived bounds
    # Min rate: φ⁻³ (0.236), Max rate: φ⁻¹ (0.618)
    # Range: φ⁻² (0.382)
    rate = PHI_INV_CUBE + PHI_INV_SQ * min(1.0, mean_curiosity / PHI)
    return float(rate)


# =============================================================================
# UTILITIES
# =============================================================================

def curiosity_map(
    model: 'TheoryTrueModel',
    dreaming: 'DreamingSystem',
    token_range: Tuple[int, int] = (0, 50),
    context_size: int = 3,
    n_samples: int = 100,
) -> Dict[int, float]:
    """
    Create a map of curiosity scores across token space.
    
    Useful for visualization and debugging.
    
    Args:
        model: TheoryTrueModel
        dreaming: DreamingSystem
        token_range: Range of tokens to sample
        context_size: Context length to use
        n_samples: Number of samples
        
    Returns:
        Dict mapping token_id to mean curiosity when that token appears
    """
    xp = model.xp
    rng = np.random.default_rng(42)
    
    token_curiosities = {i: [] for i in range(token_range[0], token_range[1])}
    
    for _ in range(n_samples):
        tokens = rng.integers(token_range[0], token_range[1], size=context_size).tolist()
        query = model.compute_context(tokens)
        cur = curiosity_score(query, model, dreaming)
        
        for t in tokens:
            if t in token_curiosities:
                token_curiosities[t].append(cur)
    
    # Average
    return {
        t: np.mean(scores) if scores else 0.0
        for t, scores in token_curiosities.items()
    }
