"""
Priority Computation — Brain-Inspired Parsimonies

Computes priority scores for episodes based on:
1. Emotional Salience (scalar + pseudoscalar) - what Grace PRESERVES
2. Novelty (distance from prototypes) - what memory DOESN'T KNOW
3. Prediction Error (Grace residual) - what was SURPRISING

All weights are φ-derived (theory-true).
"""

import numpy as np
from typing import Optional, Tuple

from holographic_prod.core.constants import (
    PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, DTYPE
)
from holographic_prod.core.quotient import consolidation_urgency


# =============================================================================
# SALIENCE COMPUTATION (Theory-Derived)
# =============================================================================

def compute_salience(M: np.ndarray, basis: np.ndarray, xp = np) -> float:
    """
    Compute emotional salience = what Grace PRESERVES.
    
    Theory (EMOTIONAL_TAGGING_THEORY.md):
        - Scalar (Grade 0): Intensity, preserved 100%
        - Pseudoscalar (Grade 4): Valence, preserved 61.8% (Fibonacci exception!)
        - Everything else decays faster
        
    High salience episodes:
        1. Are more stable (survive Grace better)
        2. Contain more "core" information
        3. Should be prioritized in consolidation
    
    Returns:
        salience: float in [0, 1] - higher = more important for memory
    
    INFORMATION PARSIMONY:
        σ = tr(M)/4 — scalar is just trace, no decomposition needed!
        p = <M, γ₅>/4 — pseudoscalar via projection onto γ₅
    """
    # Ensure both M and basis are on same device
    if xp is not np:
        import cupy as cp
        # Convert M to CuPy if needed
        if not isinstance(M, cp.ndarray):
            M = cp.asarray(M)
        # Convert basis to CuPy if needed
        if not isinstance(basis, cp.ndarray):
            basis = cp.asarray(basis)
    
    # PARSIMONY: σ = tr(M)/4 — scalar coefficient equals trace
    scalar = abs(float(xp.trace(M) / 4.0))
    
    # Pseudoscalar still needs projection onto γ₅ (basis[15])
    pseudoscalar = abs(float(xp.sum(basis[15] * M) / 4.0))
    
    # Salience = intensity + valence (weighted by survival rates)
    # Scalar survives at 1.0, pseudoscalar at PHI_INV ≈ 0.618
    salience = scalar + pseudoscalar * PHI_INV
    
    return salience


def compute_salience_batch(matrices: np.ndarray, basis: np.ndarray, xp = np) -> np.ndarray:
    """
    Batch salience computation for GPU efficiency.
    
    INFORMATION PARSIMONY:
        Only needs scalar (σ) and pseudoscalar (p) components.
        σ = tr(M)/4 — uses trace identity, no basis projection needed!
        p = <M, γ₅>/4 — still needs projection onto basis[15]
        
        Avoids full 16-coefficient decomposition for ~3× speedup.
    
    Args:
        matrices: [N, 4, 4] batch of matrices
        basis: [16, 4, 4] Clifford basis
        xp: array module (numpy or cupy)
    
    Returns:
        saliences: [N] array of salience scores
    """
    # PARSIMONY: σ = tr(M)/4 — scalar is just trace!
    scalars = xp.abs(xp.einsum('nii->n', matrices) / 4.0)
    
    # Pseudoscalar still needs projection onto γ₅ (basis[15])
    # But we use constant 4 instead of computing norm
    pseudoscalars = xp.abs(xp.einsum('nij,ij->n', matrices, basis[15]) / 4.0)
    
    # Salience = intensity + weighted valence
    saliences = scalars + pseudoscalars * PHI_INV
    
    return saliences


# =============================================================================
# NOVELTY COMPUTATION (Brain-Inspired Parsimony)
# =============================================================================

def compute_novelty(
    M: np.ndarray,
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
) -> Tuple[float, Optional['SemanticPrototype']]:
    """
    Compute novelty = distance from nearest semantic prototype.
    
    THEORY (Hippocampal Novelty Detection):
        - Brain encodes novel stimuli with priority
        - Novelty = what existing memory DOESN'T explain
        - High novelty → priority encoding
        
    IMPLEMENTATION:
        - If near a prototype: low novelty (already generalized)
        - If far from all prototypes: high novelty (new pattern)
        
    THEORY TIE-IN:
        Novelty complements emotional salience:
        - Salience = what Grace PRESERVES (scalar + pseudoscalar)
        - Novelty = what memory DOESN'T PREDICT
        
        Combined: total_priority = salience + novelty_bonus
        
    Returns:
        (novelty_score, nearest_prototype) where:
        - novelty_score: float in [0, 1] - higher = more novel
        - nearest_prototype: the closest prototype, or None if memory empty
    """
    if semantic_memory is None or semantic_memory.stats()['total_prototypes'] == 0:
        # No memory yet - everything is maximally novel
        return 1.0, None
    
    # Find nearest prototype
    matches = semantic_memory.retrieve(M, top_k=1)
    
    if not matches:
        return 1.0, None
    
    nearest_proto, similarity = matches[0]
    
    # Novelty = 1 - similarity (high similarity = low novelty)
    # Clamp to [0, 1]
    novelty = max(0.0, min(1.0, 1.0 - similarity))
    
    return novelty, nearest_proto


def compute_novelty_batch(
    matrices: np.ndarray,
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
) -> np.ndarray:
    """
    Batch novelty computation for efficiency.
    
    Args:
        matrices: [N, 4, 4] batch of episode matrices
        semantic_memory: SemanticMemory with existing prototypes
        basis: Clifford basis
        xp: array module
        
    Returns:
        novelties: [N] array of novelty scores in [0, 1]
    """
    N = matrices.shape[0]
    
    if semantic_memory is None or semantic_memory.stats()['total_prototypes'] == 0:
        # No memory - all maximally novel
        return xp.ones(N, dtype=DTYPE)
    
    # Get all prototype matrices
    all_protos = []
    for level in semantic_memory.levels:
        for proto in level:
            all_protos.append(proto.prototype_matrix)
    
    if not all_protos:
        return xp.ones(N, dtype=DTYPE)
    
    # GPU-NATIVE: Use xp.stack consistently
    proto_matrices = xp.stack([xp.asarray(m) for m in all_protos])  # [P, 4, 4]
    P = proto_matrices.shape[0]
    
    # FULLY VECTORIZED: Compute all N×P similarities at once
    # matrices: [N, 4, 4], proto_matrices: [P, 4, 4]
    # Result: [N, P] similarity matrix
    
    # Flatten for batch computation
    matrices_flat = matrices.reshape(N, -1)  # [N, 16]
    protos_flat = proto_matrices.reshape(P, -1)  # [P, 16]
    
    # Compute norms
    mat_norms = xp.linalg.norm(matrices_flat, axis=1, keepdims=True)  # [N, 1]
    proto_norms = xp.linalg.norm(protos_flat, axis=1, keepdims=True)  # [P, 1]
    
    # Compute all similarities: [N, P]
    # sim[i, j] = dot(matrices[i], protos[j]) / (norm_i * norm_j)
    all_sims = xp.dot(matrices_flat, protos_flat.T) / (mat_norms @ proto_norms.T + 1e-10)
    
    # Max similarity per episode
    max_sims = xp.max(all_sims, axis=1)  # [N]
    
    # Novelty = 1 - max_similarity
    # NOTE: Similarity is cosine similarity ∈ [-1, 1], so novelty ∈ [0, 2]
    # We don't clip - let the raw measure flow through. Values > 1 indicate
    # anti-correlation which is maximally novel.
    novelties = 1.0 - max_sims
    
    return novelties


# =============================================================================
# PREDICTION ERROR COMPUTATION
# =============================================================================

def compute_prediction_error(M: np.ndarray, basis: np.ndarray, xp = np) -> float:
    """
    Compute prediction error = what Grace REMOVES = TRANSIENT content.
    
    THEORY-TRUE MEASURE:
        Prediction error = 1 - grace_stability = consolidation_urgency
        
        This is NOT arbitrary sigmoid normalization - it's the fraction of
        coefficient energy in non-witness grades (grades 1, 2, 3).
        
        - Low error (≈0): Episode is mostly witness (stable, predictable)
        - High error (≈1): Episode is mostly transient (surprising, needs encoding)
    
    This replaces the old arbitrary sigmoid normalization with a theory-derived
    measure based on the spectral structure of the Grace operator.
    """
    # Use the theory-derived measure: 1 - grace_stability
    # This is consolidation_urgency
    return float(consolidation_urgency(M, basis, xp))


# =============================================================================
# COMBINED PRIORITY COMPUTATION
# =============================================================================

def compute_combined_priority(
    M: np.ndarray,
    basis: np.ndarray,
    semantic_memory: Optional['SemanticMemory'] = None,
    xp = np,
    salience_weight: float = PHI_INV,  # φ⁻¹ = 0.618 (primary)
    novelty_weight: float = PHI_INV_SQ,  # φ⁻² = 0.382 (secondary)
    prediction_error_weight: float = PHI_INV_CUBE,  # φ⁻³ = 0.236 (tertiary)
) -> float:
    """
    Compute combined priority score for an episode.
    
    THEORY-TRUE PRIORITIZATION (φ-derived weights):
        Priority = φ-weighted combination of:
        1. Emotional Salience (φ⁻¹) - what Grace PRESERVES (primary)
        2. Novelty (φ⁻²) - what memory DOESN'T KNOW (secondary)
        3. Prediction Error (φ⁻³) - what was SURPRISING (tertiary)
        
    High priority episodes should:
        - Be processed first during consolidation
        - Contribute more to prototype centroids
        - Be less likely to be pruned
        
    Args:
        M: Episode matrix
        basis: Clifford basis
        semantic_memory: Existing semantic memory (for novelty)
        xp: array module
        salience_weight: Weight for emotional salience (default 0.5)
        novelty_weight: Weight for novelty (default 0.3)
        prediction_error_weight: Weight for prediction error (default 0.2)
        
    Returns:
        Combined priority score (higher = more important)
    """
    # Compute components
    salience = compute_salience(M, basis, xp)
    pred_error = compute_prediction_error(M, basis, xp)
    
    if semantic_memory is not None:
        novelty, _ = compute_novelty(M, semantic_memory, basis, xp)
    else:
        novelty = 1.0  # No memory = maximally novel
    
    # Weighted combination
    priority = (
        salience_weight * salience +
        novelty_weight * novelty +
        prediction_error_weight * pred_error
    )
    
    return priority
