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
from typing import List, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from holographic_prod.dreaming import DreamingSystem
    from holographic_prod.cognitive.meta_learning import LearningState

from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE
from holographic_prod.core.algebra import grace_operator, frobenius_cosine
from holographic_prod.core.quotient import grace_stability, extract_witness

Array = np.ndarray


# =============================================================================
# CURIOSITY SCORE
# =============================================================================

def curiosity_score(
    query: Array,
    model,  # HolographicMemory or compatible
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
        model: HolographicMemory (or compatible with .xp, .basis, .n_patterns, .geometric_buckets)
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
                        # THEORY-TRUE: Use cosine similarity (normalized)
                        sim = frobenius_cosine(query, proto.centroid, xp)
                        min_proto_sim = max(min_proto_sim, float(sim))
    
    # 3. Retrieval attempt (if model has stored patterns)
    retrieval_conf = 0.0
    if hasattr(model, 'n_patterns') and model.n_patterns > 0:
        # Sample from geometric buckets to estimate retrieval confidence
        query_witness = extract_witness(query, basis, xp)
        best_sim = -10.0
        n_sampled = 0
        max_samples = 100
        
        # Sample from buckets (HolographicMemory stores in geometric_buckets)
        if hasattr(model, 'geometric_buckets'):
            for bucket in model.geometric_buckets.values():
                if n_sampled >= max_samples:
                    break
                for ctx_mat, _ in bucket[:min(10, len(bucket))]:  # Sample up to 10 per bucket
                    att_witness = extract_witness(ctx_mat, basis, xp)
                    sim = query_witness[0] * att_witness[0] + query_witness[1] * att_witness[1]
                    best_sim = max(best_sim, sim)
                    n_sampled += 1
        
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
    model,  # HolographicMemory or compatible
    dreaming: 'DreamingSystem',
    meta_state: 'LearningState',
) -> float:
    """
    Curiosity score modulated by meta-learning state.
    
    When overall uncertainty is HIGH, be more cautious (lower curiosity).
    When overall uncertainty is LOW, explore more (higher curiosity).
    
    Args:
        query: [4, 4] Clifford matrix
        model: HolographicMemory or compatible
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
    model,  # HolographicMemory or compatible
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
        model: HolographicMemory (or compatible with .embed_sequence(), .retrieve_deterministic())
        dreaming: DreamingSystem
        
    Returns:
        Estimated information gain [0, ∞)
    """
    xp = model.xp
    basis = model.basis
    
    # Compose context
    ctx_matrix = model.embed_sequence(context)
    
    # 1. Curiosity about this region
    cur = curiosity_score(ctx_matrix, model, dreaming)
    
    # 2. Novelty: how different is target from what we'd predict?
    novelty = 1.0
    if hasattr(model, 'n_patterns') and model.n_patterns > 0:
        predicted, _ = model.retrieve_deterministic(context)  # Returns (target, confidence)
        if predicted is not None and predicted != target:
            # Different prediction → novel information
            pred_emb = model.embed(predicted)
            actual_emb = model.embed(target)
            diff = xp.linalg.norm(pred_emb - actual_emb)
            novelty = min(2.0, float(diff))
        elif predicted == target:
            # We already know this → less novel (φ⁻³ residual novelty)
            novelty = PHI_INV_CUBE  # 0.236
    
    # 3. Connectivity: does target connect to existing knowledge?
    # THEORY-TRUE: Default connectivity is φ⁻¹ (moderate connection assumed)
    connectivity = PHI_INV  # 0.618
    target_emb = model.embed(target)
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
    model,  # HolographicMemory or compatible
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
        model: HolographicMemory (or compatible with .xp, .basis, .embed_sequence())
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
        context_len = getattr(model, 'context_size', 8)  # Default 8 if not set
        tokens = rng.integers(0, model.vocab_size, size=context_len).tolist()
        candidate = model.embed_sequence(tokens)
        
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
    model,  # HolographicMemory or compatible
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
        model: HolographicMemory (or compatible)
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
        if hasattr(model, 'n_patterns') and model.n_patterns > 0:
            # Start from random stored pattern
            if hasattr(model, 'geometric_buckets') and model.geometric_buckets:
                buckets = list(model.geometric_buckets.values())
                if buckets:
                    bucket = rng.choice(buckets)
                    if bucket:
                        start = bucket[0][0].copy()  # First context matrix in bucket
                    else:
                        tokens = rng.integers(0, model.vocab_size, size=8).tolist()
                        start = model.embed_sequence(tokens)
                else:
                    tokens = rng.integers(0, model.vocab_size, size=8).tolist()
                    start = model.embed_sequence(tokens)
            else:
                tokens = rng.integers(0, model.vocab_size, size=8).tolist()
                start = model.embed_sequence(tokens)
        else:
            tokens = rng.integers(0, model.vocab_size, size=8).tolist()
            start = model.embed_sequence(tokens)
        
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
            context_len = getattr(model, 'context_size', 8)  # Default 8 if not set
            tokens = rng.integers(0, model.vocab_size, size=context_len).tolist()
            candidate = model.embed_sequence(tokens)
            cur = curiosity_score(candidate, model, dreaming)
            
            if cur > best_curiosity:
                best_curiosity = cur
                best_query = candidate.copy()
        
        if best_query is None:
            context_len = getattr(model, 'context_size', 8)  # Default 8 if not set
            tokens = rng.integers(0, model.vocab_size, size=context_len).tolist()
            best_query = model.embed_sequence(tokens)
        
        return best_query


# =============================================================================
# ACTIVE LEARNING
# =============================================================================

def active_learning_step(
    model,  # HolographicMemory or compatible
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
        model: HolographicMemory or compatible
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
    model,  # HolographicMemory or compatible
    dreaming: 'DreamingSystem',
    sample_pool: List[Tuple[List[int], int]],
) -> List[Tuple[float, List[int], int]]:
    """
    Rank samples by information gain.
    
    Args:
        model: HolographicMemory or compatible
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
    model,  # HolographicMemory or compatible
    dreaming: 'DreamingSystem',
    threshold: float = PHI,
) -> bool:
    """
    Should the system explore vs exploit?
    
    THEORY:
        Explore when overall curiosity is high (knowledge is sparse).
        Exploit when curiosity is low (knowledge is dense).
        
    Args:
        model: HolographicMemory or compatible
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
        context_len = getattr(model, 'context_size', 8)  # Default 8 if not set
        tokens = rng.integers(0, model.vocab_size, size=context_len).tolist()
        query = model.embed_sequence(tokens)
        curiosities.append(curiosity_score(query, model, dreaming))
    
    mean_curiosity = np.mean(curiosities)
    return mean_curiosity > threshold


def exploration_rate(
    model,  # HolographicMemory or compatible
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
        context_len = getattr(model, 'context_size', 8)  # Default 8 if not set
        tokens = rng.integers(0, model.vocab_size, size=context_len).tolist()
        query = model.embed_sequence(tokens)
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
    model,  # HolographicMemory or compatible
    dreaming: 'DreamingSystem',
    token_range: Tuple[int, int] = (0, 50),
    context_size: int = 3,
    n_samples: int = 100,
) -> Dict[int, float]:
    """
    Create a map of curiosity scores across token space.
    
    Useful for visualization and debugging.
    
    Args:
        model: HolographicMemory or compatible
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
        query = model.embed_sequence(tokens)
        cur = curiosity_score(query, model, dreaming)
        
        for t in tokens:
            if t in token_curiosities:
                token_curiosities[t].append(cur)
    
    # Average
    return {
        t: np.mean(scores) if scores else 0.0
        for t, scores in token_curiosities.items()
    }
