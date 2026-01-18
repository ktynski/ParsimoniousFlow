"""
Episode Compression — Delta/Schema Compression

Implements delta compression for episodes relative to semantic prototypes.
This achieves 3-5x compression by storing only deviations from prototypes.

THEORY (Schema-Based Compression):
    - Most episodes are similar to existing prototypes
    - Only store DELTA (difference) from nearest prototype
    - Delta is sparse in Clifford basis (few non-zero coefficients)
"""

import numpy as np
from typing import List, Tuple, Dict

from holographic_prod.core.constants import PHI_INV_EIGHT, PHI_INV_SQ
from holographic_prod.core.algebra import (
    decompose_to_coefficients,
    frobenius_cosine,
)
from .structures import EpisodicEntry, CompressedEpisode, SemanticPrototype


def compress_episode(
    episode: EpisodicEntry,
    prototypes: List[SemanticPrototype],
    basis: np.ndarray,
    xp = np,
    sparsity_threshold: float = PHI_INV_EIGHT,  # φ⁻⁸ ≈ 0.021 for near-zero threshold
    min_similarity: float = PHI_INV_SQ,  # Theory-true: φ⁻² (spectral gap)
) -> CompressedEpisode:
    """
    Compress an episode using delta encoding relative to nearest prototype.
    
    THEORY:
        Schema compression stores DEVIATION from nearest prototype.
        If episode is similar to prototype, delta will be small and sparse.
        
    Args:
        episode: Full episodic entry to compress
        prototypes: List of semantic prototypes
        basis: Clifford basis
        xp: array module
        sparsity_threshold: Coefficients below this become zero
        min_similarity: Minimum similarity to prototype for compression
        
    Returns:
        CompressedEpisode (may be uncompressed if no good prototype match)
    """
    M = episode.context_matrix
    
    if not prototypes:
        # No prototypes: store full matrix as coefficients
        coeffs = decompose_to_coefficients(M, basis, xp)
        return CompressedEpisode(
            prototype_id=-1,
            delta_coeffs=coeffs,
            target_token=episode.target_token,
            count=episode.count,
            salience=episode.salience,
            sparsity=0.0,  # Not compressed
        )
    
    # Find nearest prototype using THEORY-TRUE cosine similarity
    best_proto_id = -1
    best_similarity = -float('inf')
    
    for i, proto in enumerate(prototypes):
        sim = frobenius_cosine(M, proto.prototype_matrix, xp)
        if sim > best_similarity:
            best_similarity = sim
            best_proto_id = i
    
    # Check if similar enough to compress (cosine in [-1, 1])
    if best_similarity < min_similarity or best_proto_id < 0:
        # Too different: store full matrix
        coeffs = decompose_to_coefficients(M, basis, xp)
        return CompressedEpisode(
            prototype_id=-1,
            delta_coeffs=coeffs,
            target_token=episode.target_token,
            count=episode.count,
            salience=episode.salience,
            sparsity=0.0,
        )
    
    # Compute delta
    proto_matrix = prototypes[best_proto_id].prototype_matrix
    delta_matrix = M - proto_matrix
    
    # Decompose delta into coefficients
    delta_coeffs = decompose_to_coefficients(delta_matrix, basis, xp)
    
    # Sparsify: set small coefficients to zero
    sparse_delta = delta_coeffs.copy()
    sparse_delta[xp.abs(sparse_delta) < sparsity_threshold] = 0.0
    
    # Compute sparsity (fraction of zeros)
    num_zeros = int(xp.sum(xp.abs(sparse_delta) < 1e-10))
    sparsity = num_zeros / 16.0
    
    return CompressedEpisode(
        prototype_id=best_proto_id,
        delta_coeffs=sparse_delta,
        target_token=episode.target_token,
        count=episode.count,
        salience=episode.salience,
        sparsity=sparsity,
    )


def compress_episodes_batch(
    episodes: List[EpisodicEntry],
    prototypes: List[SemanticPrototype],
    basis: np.ndarray,
    xp = np,
    sparsity_threshold: float = PHI_INV_EIGHT,  # φ⁻⁸ ≈ 0.021 for near-zero threshold
    min_similarity: float = PHI_INV_SQ,  # Theory-true: φ⁻² (spectral gap)
) -> Tuple[List[CompressedEpisode], Dict[str, float]]:
    """
    Batch compress episodes with statistics.
    
    Returns:
        (compressed_episodes, stats_dict)
    """
    compressed = []
    total_sparsity = 0.0
    num_actually_compressed = 0
    
    for ep in episodes:
        comp = compress_episode(ep, prototypes, basis, xp, sparsity_threshold, min_similarity)
        compressed.append(comp)
        total_sparsity += comp.sparsity
        if comp.prototype_id >= 0:
            num_actually_compressed += 1
    
    n = len(episodes)
    stats = {
        'total_episodes': n,
        'actually_compressed': num_actually_compressed,
        'compression_rate': num_actually_compressed / n if n > 0 else 0.0,
        'avg_sparsity': total_sparsity / n if n > 0 else 0.0,
        # Estimated memory savings: uncompressed uses 16 floats per episode
        # Compressed uses ~(1 - sparsity) * 16 floats
        'estimated_compression_ratio': 16.0 / (16.0 * (1.0 - total_sparsity / n) + 1) if n > 0 else 1.0,
    }
    
    return compressed, stats
