"""
Predictive Coding — Only Encode Unpredicted Content

Implements brain-inspired predictive coding:
- Hippocampus encodes PREDICTION ERRORS, not raw inputs
- Generate prediction from semantic memory (neocortex)
- Compute residual = observed - predicted
- If residual is SIGNIFICANT, encode it

THEORY (Predictive Coding in Hippocampus):
    This is PRINCIPLED delta compression:
    - Not just "similar to prototype" heuristically
    - But "what the model didn't predict" theoretically
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from holographic_prod.core.constants import PHI_INV_CUBE, DTYPE
from holographic_prod.core.algebra import decompose_to_coefficients

from .structures import EpisodicEntry, SemanticPrototype

if TYPE_CHECKING:
    from .semantic_memory import SemanticMemory


# =============================================================================
# PREDICTION FROM MEMORY
# =============================================================================

def predict_from_memory(
    query: np.ndarray,
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
    top_k: int = 1,
) -> Tuple[np.ndarray, float, Optional[SemanticPrototype]]:
    """
    Predict context from semantic memory.
    
    Returns:
        prediction: The predicted context matrix
        confidence: How confident the prediction is
        matched_proto: The matched prototype (if any)
    """
    if semantic_memory is None or semantic_memory.stats()['total_prototypes'] == 0:
        return xp.eye(4, dtype=DTYPE), 0.0, None
    
    results = semantic_memory.retrieve(query, top_k=top_k, use_pattern_completion=False)
    
    if not results:
        return xp.eye(4, dtype=DTYPE), 0.0, None
    
    proto, similarity = results[0]
    return proto.prototype_matrix, similarity, proto


# =============================================================================
# PREDICTION RESIDUAL
# =============================================================================

def compute_prediction_residual(
    observed: np.ndarray,
    predicted: np.ndarray,
    basis: np.ndarray,
    xp = np,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute the prediction residual (what wasn't predicted).
    
    THEORY (Prediction Error):
        Residual = Observed - Predicted
        This is the "surprising" content that the model didn't expect.
        
        In Clifford algebra:
        - Residual in scalar/pseudoscalar = intensity/valence difference
        - Residual in bivectors = unexpected relational structure
        - High residual = worth encoding; low residual = redundant
    
    Args:
        observed: [4, 4] actual input matrix
        predicted: [4, 4] predicted matrix from memory
        basis: [16, 4, 4] Clifford basis
        xp: numpy or cupy
        
    Returns:
        residual: [4, 4] prediction error matrix
        stats: Dict with residual statistics by grade
    """
    residual = observed - predicted
    
    # Decompose residual into grade components
    coeffs = decompose_to_coefficients(residual, basis, xp)
    
    # Grade-wise energy (squared coefficient sum)
    grade_0 = float(coeffs[0] ** 2)  # Scalar
    grade_1 = float(xp.sum(coeffs[1:5] ** 2))  # Vectors
    grade_2 = float(xp.sum(coeffs[5:11] ** 2))  # Bivectors
    grade_3 = float(xp.sum(coeffs[11:15] ** 2))  # Trivectors
    grade_4 = float(coeffs[15] ** 2)  # Pseudoscalar
    
    total_energy = float(xp.sum(coeffs ** 2))
    frobenius_norm = float(xp.linalg.norm(residual, 'fro'))
    
    stats = {
        'grade_0_energy': grade_0,
        'grade_1_energy': grade_1,
        'grade_2_energy': grade_2,
        'grade_3_energy': grade_3,
        'grade_4_energy': grade_4,
        'total_energy': total_energy,
        'frobenius_norm': frobenius_norm,
    }
    
    return residual, stats


# =============================================================================
# PREDICTIVE ENCODING
# =============================================================================

def predictive_encode(
    episode: EpisodicEntry,
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
    significance_threshold: float = PHI_INV_CUBE,  # Theory-true: φ⁻³
) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
    """
    Predictive encoding: only encode what the memory doesn't predict.
    
    THEORY (Predictive Coding in Hippocampus):
        The hippocampus encodes PREDICTION ERRORS, not raw inputs.
        
        1. Generate prediction from semantic memory (neocortex)
        2. Compute residual = observed - predicted
        3. If residual is SIGNIFICANT, encode it
        4. If residual is SMALL, skip (already known)
    
    Args:
        episode: EpisodicEntry to potentially encode
        semantic_memory: SemanticMemory for predictions
        basis: Clifford basis
        xp: numpy or cupy
        significance_threshold: Minimum residual norm to encode
        
    Returns:
        should_encode: bool - True if this episode is worth encoding
        residual: [4, 4] the prediction error
        info: Dict with encoding decision details
    """
    # Get prediction from memory
    prediction, confidence, matched_proto = predict_from_memory(
        episode.context_matrix, semantic_memory, basis, xp
    )
    
    # Compute residual
    residual, residual_stats = compute_prediction_residual(
        episode.context_matrix, prediction, basis, xp
    )
    
    # Decision: is the residual significant?
    residual_norm = residual_stats['frobenius_norm']
    is_significant = residual_norm >= significance_threshold
    
    info = {
        'confidence': confidence,
        'residual_norm': residual_norm,
        'threshold': significance_threshold,
        'is_significant': is_significant,
        'matched_proto_id': id(matched_proto) if matched_proto else None,
        'grade_energies': {
            0: residual_stats['grade_0_energy'],
            1: residual_stats['grade_1_energy'],
            2: residual_stats['grade_2_energy'],
            3: residual_stats['grade_3_energy'],
            4: residual_stats['grade_4_energy'],
        }
    }
    
    return is_significant, residual, info


def predictive_encode_batch(
    episodes: List[EpisodicEntry],
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
    significance_threshold: float = PHI_INV_CUBE,
) -> Tuple[List[EpisodicEntry], List[EpisodicEntry], Dict[str, Any]]:
    """
    Filter episodes by prediction error (batch).
    
    Only encode episodes that are SURPRISING (high prediction error).
    """
    if not episodes or semantic_memory is None or semantic_memory.stats()['total_prototypes'] == 0:
        return episodes, [], {'significant': len(episodes), 'redundant': 0, 'ratio': 1.0}
    
    # Get all prototype matrices
    all_protos = []
    for level in semantic_memory.levels:
        all_protos.extend(level)
    
    if not all_protos:
        return episodes, [], {'significant': len(episodes), 'redundant': 0, 'ratio': 1.0}
    
    n_protos = len(all_protos)
    n_episodes = len(episodes)
    
    # OPTIMIZED: Batch extract prototype matrices (single transfer)
    proto_matrices_np = np.empty((n_protos, 4, 4), dtype=np.float32)
    for i, p in enumerate(all_protos):
        mat = p.prototype_matrix
        proto_matrices_np[i] = mat.get() if hasattr(mat, 'get') else mat
    proto_matrices = xp.asarray(proto_matrices_np)
    proto_norms = xp.linalg.norm(proto_matrices.reshape(n_protos, -1), axis=1)
    protos_flat = proto_matrices.reshape(n_protos, -1)
    
    # OPTIMIZED: Batch extract episode matrices (single transfer)
    matrices_np = np.empty((n_episodes, 4, 4), dtype=np.float32)
    for i, ep in enumerate(episodes):
        mat = ep.context_matrix
        matrices_np[i] = mat.get() if hasattr(mat, 'get') else mat
    matrices = xp.asarray(matrices_np)
    mat_norms = xp.linalg.norm(matrices.reshape(n_episodes, -1), axis=1)
    matrices_flat = matrices.reshape(n_episodes, -1)
    
    # All similarities: [N, P] - fully vectorized on GPU
    all_sims = xp.dot(matrices_flat, protos_flat.T) / (mat_norms[:, None] @ proto_norms[None, :] + 1e-10)
    
    # Best similarity per episode
    best_sims = xp.max(all_sims, axis=1)
    
    # Prediction error = 1 - similarity
    prediction_errors = 1.0 - best_sims
    
    # OPTIMIZED: Transfer to CPU once, then filter (no per-element sync)
    errors_cpu = prediction_errors.get() if hasattr(prediction_errors, 'get') else prediction_errors
    significant_mask = errors_cpu >= significance_threshold
    
    significant = []
    redundant = []
    for i in range(n_episodes):
        ep = episodes[i]
        ep.prediction_error = float(errors_cpu[i])
        if significant_mask[i]:
            significant.append(ep)
        else:
            redundant.append(ep)
    
    stats = {
        'significant': len(significant),
        'redundant': len(redundant),
        'ratio': len(significant) / max(n_episodes, 1),
        'avg_prediction_error': float(np.mean(errors_cpu)),
    }
    
    return significant, redundant, stats
