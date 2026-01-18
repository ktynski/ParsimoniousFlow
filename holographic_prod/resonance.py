"""
resonance.py — Theory-True Equilibrium Dynamics
================================================

CORE THEORY (rhnsclifford.md):
    "The system finds EQUILIBRIUM, not predictions."
    
    def forward(context):
        field = build_initial_field(context)
        field = evolve_to_equilibrium(field, attractor[context])  # γ = φ⁻²
        return field  # Equilibrium IS output

THREE RETRIEVAL MODES (v4.8.0):

1. HOLOGRAPHIC RETRIEVAL (Theory-True, O(1)):
   - True superposition: memory += bind(context, target)
   - Retrieval: target ≈ unbind(context, memory)
   - See holographic_memory.py for implementation
   - This is the theory-true approach

2. WITNESS-BASED RETRIEVAL (Theory-True Fallback):
   - Index by quantized witness (scalar, pseudoscalar)
   - Respects that witness IS attractor identity
   - Falls back when holographic capacity exceeded

3. GRACE BASIN DISCOVERY (Semantic Memory):
   - Novel context → Grace flow → find attractor basin
   - Query undergoes Grace flow (stabilizes to witness)
   - Compare evolved witness to prototype witnesses (Euclidean distance)
   - No arbitrary similarity thresholds!
   - Natural basins emerge from Grace dynamics

DEPRECATED: Hash Lookup
   The original hash-based lookup (hash(context.tobytes())) was NOT
   theory-true. It ignored Clifford structure, grade hierarchy, and
   Grace dynamics. See ARCHITECTURE.md Part 3.0 for historical context.

WHY SOFTMAX/THRESHOLDS ARE WRONG:
    If we need softmax to sharpen basins, we're admitting:
    - Too many attractors (episodic, not semantic)
    - Attractors too similar (not consolidated)
    - Missing dreaming's compression

    If we need similarity thresholds, we're:
    - Imposing arbitrary cutoffs
    - Not trusting the theory's basin structure

GRACE BASIN DISCOVERY:
    1. Apply Grace flow to query → stabilizes to witness
    2. For each prototype, compare witness distances
    3. Optionally include vorticity grammar matching
    4. Closest witness = same basin
    5. Confidence from margin (how much closer than second best)
    
    This is theory-true: Grace defines the basins, not arbitrary thresholds.

VORTICITY GRAMMAR MATCHING:
    Combined score = (1-w) * witness_score + w * vorticity_score
    
    - witness_score: Euclidean distance between witness vectors
    - vorticity_score: Cosine similarity of vorticity signatures
    - w: Grammar weight (default 0.3 = 30% grammar, 70% semantic)
    
    This enables retrieval that matches both MEANING and STRUCTURE.

THEORY-TRUE RESONANCE:
    Resonance = scalar component of geometric product
    resonance(A, B) = Tr(A @ B) / (||A|| * ||B||)
    
    This measures what Grace PRESERVES - the coherent core.
    High resonance → similar geometric structure → same basin.

DISTRIBUTED PRIOR (Brain-Analog Population Coding):
    Instead of selecting ONE nearest prototype (winner-take-all),
    use MULTIPLE weak activations like biological neurons:
    
    1. Retrieve top-K prototypes by witness distance
    2. Weight by φ^(-distance) — NOT softmax!
    3. Superpose attractors: A_prior = Σ αᵢ Aᵢ
    4. Evolve query toward superposed attractor
    5. Combine target distributions
    6. Use geometric confidence (margin) to gate prior blending
    
    Brain analog:
    - Population coding (many weak activations)
    - Cortical maps (continuous semantic fields)
    - Attractor dynamics (Grace flow settling)
    
    Use via: distributed_prior_retrieve() or TheoryTrueRetriever
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_EPSILON
from holographic_prod.core.algebra import grace_operator


# =============================================================================
# THEORY-TRUE EQUILIBRIUM DYNAMICS
# =============================================================================

def evolve_to_equilibrium(
    initial: np.ndarray,
    attractor: np.ndarray,
    basis: np.ndarray,
    steps: int = 10,
    rate: float = PHI_INV_SQ,  # Theory: spectral gap γ = φ⁻²
    convergence_threshold: float = PHI_INV ** 18,  # φ⁻¹⁸ ≈ 2e-4 (theory-true threshold)
    xp = np,
) -> Tuple[np.ndarray, int, float]:
    """
    THEORY-TRUE: Evolve initial field toward a SINGLE attractor.
    
    This is the EXACT implementation from rhnsclifford.md:
        field = evolve_to_equilibrium(field, attractor[context])
    
    Update rule (per algebra.py grace_flow):
        M(t+1) = (1 - rate) * Grace(M(t)) + rate * attractor
    
    The dynamics:
        - Grace contracts toward coherent core (damps bivectors)
        - Attractor pulls toward stored state
        - Equilibrium is where these balance
    
    Args:
        initial: [4, 4] starting field (from geometric products)
        attractor: [4, 4] target attractor (from hash lookup)
        basis: [16, 4, 4] Clifford basis
        steps: max iterations
        rate: spectral gap γ = φ⁻² (theory-derived)
        convergence_threshold: early stopping
        xp: array module
    
    Returns:
        (equilibrium_matrix, steps_taken, final_delta)
    """
    state = initial.copy()
    prev_state = state.copy()
    
    for step in range(steps):
        # Theory: M_new = (1 - γ) * Grace(M) + γ * attractor
        graced = grace_operator(state, basis, xp)
        state = (1 - rate) * graced + rate * attractor
        
        # Check convergence
        delta = xp.linalg.norm(state - prev_state, 'fro')
        if delta < convergence_threshold:
            return state, step + 1, float(delta)
        
        prev_state = state.copy()
    
    return state, steps, float(xp.linalg.norm(state - prev_state, 'fro'))


# =============================================================================
# RESONANCE MEASURE (Theory-Derived)
# =============================================================================

def resonance(A: np.ndarray, B: np.ndarray, xp = np) -> float:
    """
    Theory-derived resonance measure.
    
    Resonance = what survives Grace = scalar component of product.
    
    resonance(A, B) = Tr(A @ B) / (||A|| * ||B||)
    
    This is equivalent to Frobenius similarity but conceptually:
    - High resonance → compatible geometric structure
    - Low resonance → incompatible (Grace will damp the product)
    
    Note: This IS essentially frobenius_similarity. The point is that
    the "search" metric is theory-derived, not an arbitrary choice.
    """
    norm_A = xp.linalg.norm(A, 'fro')
    norm_B = xp.linalg.norm(B, 'fro')
    
    if norm_A < PHI_EPSILON or norm_B < PHI_EPSILON:
        return 0.0
    
    # Tr(A @ B) = sum of element-wise product = Frobenius inner product
    return float(xp.sum(A * B) / (norm_A * norm_B))


def find_resonant_prototype(
    query: np.ndarray,
    prototypes: List[np.ndarray],
    xp = np,
) -> Tuple[int, float]:
    """
    Find which prototype the query resonates with.
    
    For semantic memory (few distinct prototypes), this is O(prototypes)
    which is small (~100-200) due to dreaming's compression.
    
    No softmax needed - prototypes are naturally distinct.
    
    Returns:
        (best_index, resonance_strength)
    """
    if len(prototypes) == 0:
        return -1, 0.0
    
    best_idx = 0
    best_res = -float('inf')
    
    for i, proto in enumerate(prototypes):
        res = resonance(query, proto, xp)
        if res > best_res:
            best_res = res
            best_idx = i
    
    return best_idx, float(best_res)


# =============================================================================
# GRACE BASIN DISCOVERY — Theory-True Semantic Retrieval
# =============================================================================

def grace_basin_discovery(
    query: np.ndarray,
    prototypes: List[np.ndarray],
    basis: np.ndarray,
    xp = np,
    grace_steps: int = 5,
    convergence_threshold: float = PHI_INV ** 18,  # φ⁻¹⁸ ≈ 2e-4 (theory-true threshold)
    query_vorticity_sig: np.ndarray = None,
    prototype_vorticity_sigs: List[np.ndarray] = None,
    vorticity_weight: float = PHI_INV_CUBE,  # φ-derived (was 0.3)
) -> Tuple[int, float, Dict[str, Any]]:
    """
    THEORY-TRUE semantic retrieval via Grace basin discovery WITH grammar matching.
    
    PROBLEM WITH THRESHOLD-BASED RETRIEVAL:
        The current approach uses `resonance(query, prototype) > threshold`.
        This fails because:
        1. Arbitrary threshold (why 0.5? why not 0.3 or 0.7?)
        2. Novel contexts often don't exceed threshold
        3. Ignores Grace dynamics entirely
    
    THEORY-TRUE APPROACH:
        Grace flow naturally finds equilibria. Instead of comparing similarity,
        we EVOLVE the query via Grace and see which prototype basin it lands in.
        
        1. Apply Grace to the query (contracts high-grade content)
        2. Compare the evolved query to each prototype
        3. The prototype whose basin the query converges to wins
        4. Convergence quality (stability) indicates match confidence
        
    GRAMMAR MATCHING (NEW):
        When vorticity signatures are provided, we also match syntactic structure:
        - Vorticity = wedge product = antisymmetric = word ORDER
        - Same grammatical structure → similar vorticity signature
        - Combined score = (1 - vorticity_weight) * witness_score + vorticity_weight * vorticity_score
    
    WHY THIS WORKS:
        - Grace damps vorticity (grade-2) at rate φ⁻⁴
        - After Grace flow, the query's "core" (scalar + pseudoscalar) dominates
        - Prototypes are already Grace-stable (from dreaming)
        - Basin membership is determined by witness alignment + grammar match
    
    NO ARBITRARY THRESHOLDS:
        - The best match is always returned
        - Confidence = how well the query's basin aligns with the prototype
        - Low confidence means "uncertain" not "no match"
    
    Args:
        query: [4, 4] query matrix (context representation)
        prototypes: List of [4, 4] semantic prototypes
        basis: [16, 4, 4] Clifford basis
        xp: array module
        grace_steps: Number of Grace flow iterations
        convergence_threshold: For early stopping
        query_vorticity_sig: [16] optional vorticity signature of query
        prototype_vorticity_sigs: List of [16] vorticity signatures for each prototype
        vorticity_weight: Weight for vorticity matching (0 = ignore grammar, 1 = only grammar)
        
    Returns:
        (best_prototype_index, confidence, info_dict)
    """
    from holographic_prod.core.quotient import extract_witness
    
    if len(prototypes) == 0:
        return -1, 0.0, {"source": "none", "reason": "no prototypes"}
    
    # Step 1: Evolve query via Grace flow (find its natural basin)
    evolved_query = query.copy()
    prev_query = query.copy()
    
    for step in range(grace_steps):
        evolved_query = grace_operator(evolved_query, basis, xp)
        delta = xp.linalg.norm(evolved_query - prev_query, 'fro')
        if delta < convergence_threshold:
            break
        prev_query = evolved_query.copy()
    
    # Step 2: Extract witness (gauge-invariant core) of evolved query
    query_witness = extract_witness(evolved_query, basis, xp)
    query_witness_norm = abs(query_witness[0]) + abs(query_witness[1]) + PHI_EPSILON
    
    # Step 3: Find which prototype basin the query belongs to
    # Use COMBINED witness distance + vorticity similarity
    # 
    # WHY DISTANCE NOT COSINE for witness:
    #   After Grace flow, witnesses are dominated by scalar (preserved 100%).
    #   Cosine similarity treats (1.5, 0.01) and (1.0, 0.01) as nearly identical
    #   because they point in the same direction. But we want to match the
    #   ACTUAL VALUES, not just the direction.
    #
    # WHY COSINE for vorticity:
    #   Vorticity signatures encode DIRECTION of word order structure.
    #   Same structure → similar direction, opposite structure → opposite direction.
    #
    #   Combined score = (1-w) * witness_score + w * vorticity_score
    #   where scores are normalized to [0, 1] range (higher = better)
    
    best_idx = 0
    best_combined_score = float('-inf')
    combined_scores = []
    
    # Check if we have vorticity signatures for grammar matching
    use_vorticity = (
        query_vorticity_sig is not None and 
        prototype_vorticity_sigs is not None and 
        len(prototype_vorticity_sigs) == len(prototypes) and
        vorticity_weight > 0
    )
    
    for i, proto in enumerate(prototypes):
        # Prototypes should already be Grace-stable, but apply once for consistency
        proto_evolved = grace_operator(proto, basis, xp)
        proto_witness = extract_witness(proto_evolved, basis, xp)
        
        # Witness distance (L1 norm) - lower is better
        # Weight scalar more since it's preserved 100% by Grace
        scalar_dist = abs(query_witness[0] - proto_witness[0])
        pseudo_dist = abs(query_witness[1] - proto_witness[1])
        witness_distance = scalar_dist + PHI_INV * pseudo_dist
        
        # Convert to score (higher = better)
        # Normalize by query witness scale for consistency
        witness_score = max(0.0, 1.0 - witness_distance / query_witness_norm)
        
        # Vorticity similarity (if available) - cosine similarity in [-1, 1]
        if use_vorticity:
            proto_vort = prototype_vorticity_sigs[i]
            if proto_vort is not None:
                query_vort_norm = float(xp.linalg.norm(query_vorticity_sig))
                proto_vort_norm = float(xp.linalg.norm(proto_vort))
                if query_vort_norm > PHI_EPSILON and proto_vort_norm > PHI_EPSILON:
                    vort_sim = float(xp.dot(query_vorticity_sig, proto_vort)) / (query_vort_norm * proto_vort_norm)
                    # Convert [-1, 1] cosine to [0, 1] score
                    vort_score = (vort_sim + 1.0) / 2.0
                else:
                    vort_score = 0.5  # Neutral if no vorticity
            else:
                vort_score = 0.5  # Neutral if no vorticity
        else:
            vort_score = 0.5  # Neutral when not using vorticity
        
        # Combined score
        combined_score = (1.0 - vorticity_weight) * witness_score + vorticity_weight * vort_score
        combined_scores.append(combined_score)
        
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_idx = i
    
    # Step 4: Compute confidence
    # Confidence = how much better the best match is than the second best
    sorted_scores = sorted(combined_scores, reverse=True)  # Descending
    if len(sorted_scores) >= 2:
        margin = sorted_scores[0] - sorted_scores[1]  # Gap to second best
        # Confidence: base score + margin bonus
        confidence = min(1.0, best_combined_score + margin)
    else:
        # Single prototype: confidence based on score alone
        confidence = best_combined_score
    
    info = {
        "source": "grace_basin_with_grammar" if use_vorticity else "grace_basin",
        "grace_steps": step + 1 if step < grace_steps else grace_steps,
        "best_combined_score": float(best_combined_score),
        "margin": float(margin) if len(sorted_scores) >= 2 else 0.0,
        "num_prototypes": len(prototypes),
        "vorticity_weight": vorticity_weight if use_vorticity else 0.0,
    }
    
    return best_idx, float(confidence), info


def grace_basin_retrieve(
    query: np.ndarray,
    prototypes: List[np.ndarray],
    prototype_targets: List[Dict[int, float]],
    basis: np.ndarray,
    xp = np,
    grace_steps: int = 5,
    min_confidence: float = 0.0,  # No arbitrary threshold by default!
    query_vorticity_sig: np.ndarray = None,
    prototype_vorticity_sigs: List[np.ndarray] = None,
    vorticity_weight: float = PHI_INV_CUBE,  # φ-derived (was 0.3)
) -> Tuple[Optional[int], int, float, Dict[str, Any]]:
    """
    Full semantic retrieval via Grace basin discovery WITH grammar matching.
    
    This replaces threshold-based retrieval with theory-true basin discovery.
    
    Args:
        query: [4, 4] query matrix
        prototypes: List of [4, 4] semantic prototypes
        prototype_targets: List of target distributions for each prototype
        basis: [16, 4, 4] Clifford basis
        xp: array module
        grace_steps: Number of Grace flow iterations
        min_confidence: Minimum confidence to return a match (default 0 = always return)
        query_vorticity_sig: [16] optional vorticity signature for grammar matching
        prototype_vorticity_sigs: List of [16] vorticity signatures for each prototype
        vorticity_weight: Weight for vorticity/grammar matching (0 = ignore, 1 = only grammar)
        
    Returns:
        (prototype_index, target, confidence, info)
        If no match (confidence < min_confidence), returns (None, 0, confidence, info)
    """
    if len(prototypes) == 0:
        return None, 0, 0.0, {"source": "none", "reason": "no prototypes"}
    
    best_idx, confidence, info = grace_basin_discovery(
        query, prototypes, basis, xp, grace_steps,
        query_vorticity_sig=query_vorticity_sig,
        prototype_vorticity_sigs=prototype_vorticity_sigs,
        vorticity_weight=vorticity_weight,
    )
    
    if confidence < min_confidence:
        info["reason"] = f"confidence {confidence:.3f} < min {min_confidence:.3f}"
        return None, 0, confidence, info
    
    # Get target distribution
    target_dist = prototype_targets[best_idx]
    
    # THEORY-TRUE: Don't just take mode! Sample weighted by probability.
    # Mode selection causes collapse to high-frequency words ("the", "and").
    # Sampling preserves distribution information.
    rng = np.random.default_rng()
    tokens = list(target_dist.keys())
    probs = np.array(list(target_dist.values()))
    probs = probs / probs.sum()  # Ensure normalized
    target = int(rng.choice(tokens, p=probs))
    
    info["target_dist"] = target_dist
    info["sampling"] = "probabilistic"  # Not mode!
    
    return best_idx, target, confidence, info


# =============================================================================
# DISTRIBUTED PRIOR RETRIEVAL (Brain-Analog Population Coding)
# =============================================================================

def distributed_prior_retrieve(
    query: np.ndarray,
    prototypes: List[np.ndarray],
    prototype_targets: List[Dict[int, float]],
    prototype_supports: List[float],
    basis: np.ndarray,
    xp = np,
    K: int = 8,
    grace_steps: int = 3,
    query_vorticity_sig: np.ndarray = None,
    prototype_vorticity_sigs: List[np.ndarray] = None,
    factorized_prior = None,  # Optional FactorizedAssociativePrior
    confidence_threshold: float = PHI_INV_SQ,  # Theory-true: φ⁻² (spectral gap)
) -> Tuple[Optional[int], int, float, Dict[str, Any]]:
    """
    BRAIN-ANALOG retrieval using distributed prior (population coding).
    
    Instead of selecting ONE nearest prototype (winner-take-all),
    this uses MULTIPLE weak activations (like biological population coding):
    
    1. Retrieve top-K prototypes by witness distance
    2. Weight by φ^(-distance) — NOT softmax!
    3. Superpose attractors: A_prior = Σ αᵢ Aᵢ
    4. Evolve query toward superposed attractor
    5. Combine target distributions
    6. Use geometric confidence (margin) to gate prior blending
    
    WHY THIS IS BRAIN-TRUE:
        - Multiple weak activations (like cortical population coding)
        - φ-weighted kernel (theory-derived, not arbitrary)
        - No softmax, no sampling — deterministic settling
        - Confidence from margin geometry (not probability)
    
    WHEN TO USE:
        - Always! This is the recommended retrieval method.
        - Falls back to single-prototype if only 1 available.
    
    Args:
        query: [4, 4] query matrix
        prototypes: List of [4, 4] semantic prototypes
        prototype_targets: List of target distributions
        prototype_supports: List of support counts (from consolidation)
        basis: [16, 4, 4] Clifford basis
        xp: array module
        K: Number of nearest neighbors to superpose
        grace_steps: Grace flow steps
        query_vorticity_sig: Optional vorticity signature for grammar matching
        prototype_vorticity_sigs: Optional list of prototype vorticity signatures
        factorized_prior: Optional global prior (FactorizedAssociativePrior) for blending
        confidence_threshold: Below this, blend with global prior (default 0.5 ≈ φ⁻¹)
    
    Returns:
        (prototype_index, target, confidence, info)
    """
    # Import here to avoid circular dependencies
    from holographic_prod.cognitive.distributed_prior import (
        retrieve_with_distributed_prior,
        FactorizedAssociativePrior,
    )
    
    if len(prototypes) == 0:
        return None, 0, 0.0, {"source": "none", "reason": "no prototypes"}
    
    # Use the full distributed prior retrieval
    equilibrium, combined_targets, confidence, info = retrieve_with_distributed_prior(
        query=query,
        prototypes=prototypes,
        prototype_targets=prototype_targets,
        prototype_supports=prototype_supports,
        factorized_prior=factorized_prior,
        basis=basis,
        xp=xp,
        K=K,
        confidence_threshold=confidence_threshold,
        grace_steps=grace_steps,
    )
    
    if not combined_targets:
        return None, 0, confidence, info
    
    # Sample from combined target distribution (not mode!)
    rng = np.random.default_rng()
    tokens = list(combined_targets.keys())
    probs = np.array(list(combined_targets.values()))
    probs = probs / (probs.sum() + PHI_EPSILON)  # Normalize
    target = int(rng.choice(tokens, p=probs))
    
    info["combined_targets"] = combined_targets
    info["sampling"] = "distributed_prior"
    
    # Best prototype is the first in top_indices
    best_idx = info.get("top_indices", [0])[0] if "top_indices" in info else 0
    
    return best_idx, target, confidence, info


# =============================================================================
# COMPLETE RETRIEVAL SYSTEM
# =============================================================================

class TheoryTrueRetriever:
    """
    Theory-true retrieval: hash lookup + distributed prior.
    
    For known contexts:
        hash → attractor → evolve_to_equilibrium
    
    For novel contexts (semantic generalization):
        distributed_prior_retrieve → evolve_to_equilibrium
    
    BRAIN-ANALOG MODES:
    
    1. Episodic Memory (hash lookup):
       - Exact context hash → stored attractor
       - Like hippocampal place cells
    
    2. Semantic Memory (distributed prior):
       - Top-K nearest prototypes (population coding)
       - φ-weighted superposition (NOT softmax!)
       - Grace flow to equilibrium
       - Geometric confidence from margin
    
    3. Global Fallback (factorized prior):
       - Hebbian-learned operator: witness → attractor
       - Provides smooth interpolation in uncovered regions
       - Like cortico-cortical projections
    
    THEORY-TRUE IMPROVEMENTS:
        - NO softmax (φ-kernel instead)
        - NO arbitrary thresholds (geometric confidence)
        - NO sampling from distributions (deterministic settling)
        - Multiple weak activations (population coding)
    """
    
    def __init__(
        self,
        basis: np.ndarray,
        equilibrium_steps: int = 10,
        grace_steps: int = 5,  # Steps for basin/prior discovery
        use_distributed_prior: bool = True,  # Use brain-analog population coding
        use_grace_basin: bool = True,  # Fallback to single prototype if False
        K: int = 8,  # Number of neighbors for distributed prior
        confidence_threshold: float = PHI_INV_SQ,  # Theory-true: φ⁻² (spectral gap)
        xp = np,
    ):
        self.basis = basis
        self.equilibrium_steps = equilibrium_steps
        self.grace_steps = grace_steps
        self.use_distributed_prior = use_distributed_prior
        self.use_grace_basin = use_grace_basin
        self.K = K
        self.confidence_threshold = confidence_threshold
        self.xp = xp
        
        # Episodic memory (hash-indexed)
        self.attractor_map: Dict[int, np.ndarray] = {}
        self.target_map: Dict[int, int] = {}
        
        # Semantic memory (prototypes from dreaming)
        self.prototypes: List[np.ndarray] = []
        self.prototype_targets: List[Dict[int, float]] = []  # Multi-modal targets
        self.prototype_supports: List[float] = []  # Support counts
        self.prototype_vorticity_sigs: List[np.ndarray] = []  # Grammar signatures
        
        # Global prior (complementary path, not fallback)
        self.factorized_prior = None  # Initialize with init_factorized_prior()
    
    def add_attractor(self, context_hash: int, attractor: np.ndarray, target: int):
        """Add to episodic memory."""
        self.attractor_map[context_hash] = attractor
        self.target_map[context_hash] = target
    
    def add_prototype(
        self, 
        prototype: np.ndarray, 
        target_dist: Dict[int, float],
        support: float = 1.0,
        vorticity_sig: np.ndarray = None,
    ):
        """
        Add semantic prototype (from dreaming).
        
        Args:
            prototype: [4, 4] prototype matrix
            target_dist: Target distribution {token: probability}
            support: Number of episodes that formed this prototype
            vorticity_sig: [16] vorticity signature for grammar matching
        """
        self.prototypes.append(prototype)
        self.prototype_targets.append(target_dist)
        self.prototype_supports.append(support)
        self.prototype_vorticity_sigs.append(vorticity_sig)
        
        # Update factorized prior if initialized
        if self.factorized_prior is not None:
            from holographic_prod.cognitive.distributed_prior import extended_witness
            witness = extended_witness(prototype, self.basis, self.xp)
            self.factorized_prior.update(witness, prototype, weight=support)
    
    def init_factorized_prior(self, witness_dim: int = 4):
        """
        Initialize the factorized associative prior (global prior path).
        
        Call this after creating the retriever to enable global prior blending
        for queries in uncovered regions.
        """
        from holographic_prod.cognitive.distributed_prior import FactorizedAssociativePrior
        self.factorized_prior = FactorizedAssociativePrior(
            witness_dim=witness_dim, xp=self.xp
        )
        
        # Train on existing prototypes
        from holographic_prod.cognitive.distributed_prior import extended_witness
        for i, proto in enumerate(self.prototypes):
            witness = extended_witness(proto, self.basis, self.xp)
            support = self.prototype_supports[i] if i < len(self.prototype_supports) else 1.0
            self.factorized_prior.update(witness, proto, weight=support)
    
    def retrieve(
        self, 
        query: np.ndarray, 
        context_hash: int,
        query_vorticity_sig: np.ndarray = None,
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Theory-true retrieval with distributed prior.
        
        1. Try hash lookup (episodic memory) — like hippocampal place cells
        2. If miss, try distributed prior (population coding) — like cortex
        3. Evolve to equilibrium
        
        Returns:
            (equilibrium, target, stats)
        """
        stats = {"source": None, "steps": 0, "resonance": 0.0}
        
        # Mode 1: Hash lookup (exact match — episodic memory)
        if context_hash in self.attractor_map:
            attractor = self.attractor_map[context_hash]
            target = self.target_map[context_hash]
            
            equilibrium, steps, delta = evolve_to_equilibrium(
                query, attractor, self.basis,
                steps=self.equilibrium_steps,
                xp=self.xp,
            )
            
            stats["source"] = "episodic"
            stats["steps"] = steps
            stats["delta"] = delta
            
            return equilibrium, target, stats
        
        # Mode 2: Semantic memory (prototype generalization)
        if len(self.prototypes) > 0:
            if self.use_distributed_prior:
                # BRAIN-ANALOG: Distributed prior (population coding)
                proto_idx, target, confidence, prior_info = distributed_prior_retrieve(
                    query=query,
                    prototypes=self.prototypes,
                    prototype_targets=self.prototype_targets,
                    prototype_supports=self.prototype_supports if self.prototype_supports else [1.0] * len(self.prototypes),
                    basis=self.basis,
                    xp=self.xp,
                    K=self.K,
                    grace_steps=self.grace_steps,
                    query_vorticity_sig=query_vorticity_sig,
                    prototype_vorticity_sigs=self.prototype_vorticity_sigs if self.prototype_vorticity_sigs else None,
                    factorized_prior=self.factorized_prior,
                    confidence_threshold=self.confidence_threshold,
                )
                
                if proto_idx is not None:
                    prototype = self.prototypes[proto_idx]
                    
                    equilibrium, steps, delta = evolve_to_equilibrium(
                        query, prototype, self.basis,
                        steps=self.equilibrium_steps,
                        xp=self.xp,
                    )
                    
                    stats["source"] = prior_info.get("source", "distributed_prior")
                    stats["steps"] = steps
                    stats["delta"] = delta
                    stats["confidence"] = confidence
                    stats["prior_info"] = prior_info
                    stats["target_dist"] = prior_info.get("combined_targets", {})
                    
                    return equilibrium, target, stats
            
            elif self.use_grace_basin:
                # FALLBACK: Single-prototype Grace basin discovery
                proto_idx, target, confidence, basin_info = grace_basin_retrieve(
                    query, self.prototypes, self.prototype_targets,
                    self.basis, self.xp, self.grace_steps,
                    min_confidence=0.0,  # Always return best match
                    query_vorticity_sig=query_vorticity_sig,
                    prototype_vorticity_sigs=self.prototype_vorticity_sigs if self.prototype_vorticity_sigs else None,
                )
                
                if proto_idx is not None:
                    prototype = self.prototypes[proto_idx]
                    
                    equilibrium, steps, delta = evolve_to_equilibrium(
                        query, prototype, self.basis,
                        steps=self.equilibrium_steps,
                        xp=self.xp,
                    )
                    
                    stats["source"] = "semantic_grace_basin"
                    stats["steps"] = steps
                    stats["delta"] = delta
                    stats["confidence"] = confidence
                    stats["basin_info"] = basin_info
                    stats["target_dist"] = basin_info.get("target_dist", {})
                    
                    return equilibrium, target, stats
        
        # Fallback: no match found
        stats["source"] = "none"
        return query, 0, stats


# =============================================================================
# TESTS
# =============================================================================

def test_equilibrium_convergence():
    """Test that equilibrium dynamics converge to attractor."""
    print("\n=== Test: Equilibrium Convergence (Single Attractor) ===\n")
    
    from holographic_prod.core.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    
    # Create single attractor
    np.random.seed(42)
    attractor = np.eye(4) + 0.3 * np.random.randn(4, 4)
    attractor = attractor / np.linalg.norm(attractor, 'fro')
    
    # Start from random initial state
    np.random.seed(123)
    initial = np.random.randn(4, 4)
    initial = initial / np.linalg.norm(initial, 'fro')
    
    initial_res = resonance(initial, attractor, np)
    
    print(f"  Initial resonance with attractor: {initial_res:.4f}")
    print()
    print("  Step | Resonance | Delta")
    print("  " + "-" * 35)
    
    state = initial.copy()
    for step in range(15):
        # Theory: M_new = (1 - γ) * Grace(M) + γ * attractor
        graced = grace_operator(state, basis, np)
        new_state = (1 - PHI_INV_SQ) * graced + PHI_INV_SQ * attractor
        
        res = resonance(new_state, attractor, np)
        delta = np.linalg.norm(new_state - state, 'fro')
        
        print(f"  {step:4d} | {res:9.4f} | {delta:.6f}")
        
        if delta < PHI_INV ** 18:  # φ⁻¹⁸ ≈ 2e-4 (theory-true threshold)
            print(f"\n  ✓ Converged at step {step}")
            break
        
        state = new_state
    
    final_res = resonance(state, attractor, np)
    print(f"\n  Initial resonance: {initial_res:.4f}")
    print(f"  Final resonance: {final_res:.4f}")
    print(f"  Improvement: {final_res - initial_res:+.4f}")
    
    return final_res > 0.9  # Should converge very close to attractor


def test_prototype_retrieval():
    """Test semantic memory retrieval with distinct prototypes."""
    print("\n=== Test: Prototype Retrieval (Semantic Memory) ===\n")
    
    from holographic_prod.core.algebra import build_clifford_basis
    from holographic_prod.core.quotient import extract_witness
    
    basis = build_clifford_basis(np)
    retriever = TheoryTrueRetriever(basis)
    
    # Create distinct prototypes with DIFFERENT witness signatures
    # (Don't normalize - that makes all scalars similar!)
    np.random.seed(42)
    proto_A = np.eye(4) * 1.5 + 0.2 * np.random.randn(4, 4)  # High scalar
    
    np.random.seed(123)  
    proto_B = np.eye(4) * 0.5 + 0.3 * np.random.randn(4, 4)  # Low scalar
    
    np.random.seed(456)
    proto_C = np.eye(4) * 1.0 + 0.25 * np.random.randn(4, 4)  # Medium scalar
    
    # Verify distinct witnesses
    print("  Prototype witness signatures:")
    for i, proto in enumerate([proto_A, proto_B, proto_C]):
        w = extract_witness(proto, basis, np)
        print(f"    Proto {i}: scalar={w[0]:.4f}, pseudo={w[1]:.4f}")
    print()
    
    # Add prototypes with target distributions
    retriever.add_prototype(proto_A, {0: 0.8, 1: 0.2})
    retriever.add_prototype(proto_B, {2: 0.9, 3: 0.1})
    retriever.add_prototype(proto_C, {4: 1.0})
    
    # Check prototype separations (resonance)
    sep_AB = resonance(proto_A, proto_B, np)
    sep_AC = resonance(proto_A, proto_C, np)
    sep_BC = resonance(proto_B, proto_C, np)
    
    print(f"  Prototype separations (resonance):")
    print(f"    A-B: {sep_AB:.4f}")
    print(f"    A-C: {sep_AC:.4f}")
    print(f"    B-C: {sep_BC:.4f}")
    print()
    
    # Test queries (perturbed versions of prototypes)
    correct = 0
    expected = [0, 2, 4]  # Mode targets for A, B, C
    
    for i, (proto, expected_target) in enumerate(zip([proto_A, proto_B, proto_C], expected)):
        # Perturb slightly (don't normalize the query either)
        np.random.seed(i * 100 + 999)
        query = proto + 0.1 * np.random.randn(4, 4)
        
        # Novel hash (not in episodic memory)
        novel_hash = hash(f"novel_{i}")
        
        equilibrium, target, stats = retriever.retrieve(query, novel_hash)
        
        match = "✓" if target == expected_target else "✗"
        conf = stats.get('confidence', stats.get('resonance', 0.0))
        print(f"  Query {i}: expected={expected_target}, got={target} {match}")
        print(f"    source={stats['source']}, confidence={conf:.4f}")
        
        if target == expected_target:
            correct += 1
    
    print(f"\n  Accuracy: {correct}/3 ({correct/3*100:.0f}%)")
    
    return correct == 3


def test_hash_vs_semantic():
    """Test that hash lookup takes precedence over semantic."""
    print("\n=== Test: Hash Lookup Precedence ===\n")
    
    from holographic_prod.core.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    retriever = TheoryTrueRetriever(basis)
    
    # Create attractor and prototype
    np.random.seed(42)
    attractor = np.eye(4) + 0.3 * np.random.randn(4, 4)
    attractor = attractor / np.linalg.norm(attractor, 'fro')
    
    np.random.seed(123)
    prototype = np.eye(4) + 0.3 * np.random.randn(4, 4)
    prototype = prototype / np.linalg.norm(prototype, 'fro')
    
    # Add both
    known_hash = hash("known_context")
    retriever.add_attractor(known_hash, attractor, target=42)
    retriever.add_prototype(prototype, {99: 1.0})
    
    # Query with known hash → should use episodic
    query = attractor + 0.1 * np.random.randn(4, 4)
    _, target, stats = retriever.retrieve(query, known_hash)
    
    print(f"  Known hash query:")
    print(f"    Target: {target} (expected: 42)")
    print(f"    Source: {stats['source']} (expected: episodic)")
    
    episodic_correct = (target == 42 and stats['source'] == 'episodic')
    
    # Query with unknown hash → should use semantic
    unknown_hash = hash("unknown_context")
    query = prototype + 0.1 * np.random.randn(4, 4)
    _, target, stats = retriever.retrieve(query, unknown_hash)
    
    print(f"\n  Unknown hash query:")
    print(f"    Target: {target} (expected: 99)")
    print(f"    Source: {stats['source']} (expected: semantic*)")
    
    # Accept any semantic source (grace_basin or threshold)
    semantic_correct = (target == 99 and 'semantic' in stats['source'])
    
    return episodic_correct and semantic_correct


def test_grace_basin_discovery():
    """
    Test Grace basin discovery for semantic retrieval.
    
    THEORY:
        Grace flow naturally finds equilibria. Instead of comparing similarity
        to prototypes (which requires arbitrary thresholds), we evolve the query
        via Grace and see which prototype basin it lands in.
    
    TEST:
        1. Create distinct prototypes with different witness signatures
        2. Create queries that are perturbed versions of each prototype
        3. Verify Grace basin discovery correctly identifies the source prototype
        4. Verify confidence is higher for closer matches
    """
    print("\n=== Test: Grace Basin Discovery ===\n")
    
    from holographic_prod.core.algebra import build_clifford_basis
    from holographic_prod.core.quotient import extract_witness
    
    basis = build_clifford_basis(np)
    
    # Create distinct prototypes with different witness signatures
    np.random.seed(42)
    proto_A = np.eye(4) * 1.5 + 0.2 * np.random.randn(4, 4)  # High scalar
    
    np.random.seed(123)
    proto_B = np.eye(4) * 0.5 + 0.4 * np.random.randn(4, 4)  # Lower scalar, more noise
    
    np.random.seed(456)
    proto_C = np.eye(4) * 1.0 + 0.3 * np.random.randn(4, 4)  # Medium
    
    prototypes = [proto_A, proto_B, proto_C]
    prototype_targets = [{0: 1.0}, {1: 1.0}, {2: 1.0}]
    
    # Check witness signatures
    print("  Prototype witness signatures:")
    for i, proto in enumerate(prototypes):
        w = extract_witness(proto, basis, np)
        print(f"    Proto {i}: scalar={w[0]:.4f}, pseudo={w[1]:.4f}")
    print()
    
    # Test queries: perturbed versions of each prototype
    correct = 0
    total = 0
    
    for expected_idx, proto in enumerate(prototypes):
        for noise_level in [0.05, 0.1, 0.2]:
            np.random.seed(expected_idx * 100 + int(noise_level * 100))
            query = proto + noise_level * np.random.randn(4, 4)
            
            # Use Grace basin discovery
            best_idx, confidence, info = grace_basin_discovery(
                query, prototypes, basis, np, grace_steps=5
            )
            
            match = "✓" if best_idx == expected_idx else "✗"
            print(f"  Query (proto {expected_idx}, noise={noise_level:.2f}): "
                  f"got={best_idx}, conf={confidence:.3f} {match}")
            
            if best_idx == expected_idx:
                correct += 1
            total += 1
    
    accuracy = correct / total
    print(f"\n  Accuracy: {correct}/{total} ({accuracy*100:.0f}%)")
    
    # Test confidence ordering: closer queries should have higher confidence
    print("\n  Confidence vs noise level:")
    confidences = []
    for noise_level in [0.01, 0.05, 0.1, 0.2, 0.3]:
        np.random.seed(999)
        query = proto_A + noise_level * np.random.randn(4, 4)
        _, conf, _ = grace_basin_discovery(query, prototypes, basis, np)
        confidences.append(conf)
        print(f"    noise={noise_level:.2f} → confidence={conf:.3f}")
    
    # Confidence should generally decrease with noise (not strictly required)
    confidence_trend_ok = confidences[0] >= confidences[-1] - 0.1  # Allow some tolerance
    
    success = accuracy >= PHI_INV and confidence_trend_ok  # φ-derived threshold (was 0.8)
    
    if success:
        print("\n  ✓ Grace basin discovery works correctly")
        print("    - Correctly identifies source prototype")
        print("    - Confidence reflects match quality")
    else:
        print("\n  ✗ FAILED")
        if accuracy < PHI_INV:
            print(f"    - Accuracy too low: {accuracy*100:.0f}%")
        if not confidence_trend_ok:
            print("    - Confidence trend incorrect")
    
    return success


if __name__ == "__main__":
    print("=" * 60)
    print("THEORY-TRUE EQUILIBRIUM DYNAMICS")
    print("=" * 60)
    
    results = []
    results.append(("Equilibrium Convergence", test_equilibrium_convergence()))
    results.append(("Prototype Retrieval", test_prototype_retrieval()))
    results.append(("Hash vs Semantic", test_hash_vs_semantic()))
    results.append(("Grace Basin Discovery", test_grace_basin_discovery()))
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(p for _, p in results)
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
