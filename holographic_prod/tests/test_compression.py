"""
Test-Driven Development for Episode Compression Module.

Tests compression functions before extraction.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from holographic_prod.core.constants import (
    PHI_INV_EIGHT, PHI_INV_SQ, PHI_INV_FOUR, DTYPE
)
from holographic_prod.core.algebra import build_clifford_basis
from holographic_prod.dreaming.structures import EpisodicEntry, CompressedEpisode, SemanticPrototype


# =============================================================================
# TESTS FOR COMPRESSION MODULE (to be extracted)
# =============================================================================

def test_compress_episode_no_prototypes():
    """Test compression when no prototypes exist."""
    basis = build_clifford_basis(np)
    
    # Create an episode
    ctx_matrix = np.eye(4, dtype=DTYPE) + PHI_INV_FOUR * np.random.randn(4, 4)
    episode = EpisodicEntry(context_matrix=ctx_matrix, target_token=42)
    
    # Import from compression module
    from holographic_prod.dreaming.compression import compress_episode
    
    # Compress with no prototypes
    compressed = compress_episode(episode, [], basis, np)
    
    assert compressed.prototype_id == -1  # Uncompressed
    assert compressed.target_token == 42
    assert compressed.sparsity == 0.0  # Not compressed
    assert compressed.delta_coeffs is not None


def test_compress_episode_with_similar_prototype():
    """Test compression when similar prototype exists."""
    basis = build_clifford_basis(np)
    
    # Create prototype
    proto_matrix = np.eye(4, dtype=DTYPE)
    proto = SemanticPrototype(
        prototype_matrix=proto_matrix,
        target_distribution={10: 1.0},
        radius=0.1,
        support=5,
    )
    
    # Create episode similar to prototype
    ctx_matrix = proto_matrix + PHI_INV_EIGHT * np.random.randn(4, 4)  # Very similar
    episode = EpisodicEntry(context_matrix=ctx_matrix, target_token=10)
    
    from holographic_prod.dreaming.compression import compress_episode
    
    # Compress
    compressed = compress_episode(episode, [proto], basis, np)
    
    assert compressed.prototype_id == 0  # Matched prototype
    assert compressed.target_token == 10
    assert compressed.sparsity > 0.0  # Should have sparsity (delta is small)
    assert compressed.delta_coeffs is not None


def test_compress_episode_with_dissimilar_prototype():
    """Test compression when prototype is too different."""
    basis = build_clifford_basis(np)
    
    # Create prototype
    proto_matrix = np.eye(4, dtype=DTYPE)
    proto = SemanticPrototype(
        prototype_matrix=proto_matrix,
        target_distribution={10: 1.0},
        radius=0.1,
        support=5,
    )
    
    # Create episode very different from prototype
    ctx_matrix = np.random.randn(4, 4)  # Very different
    episode = EpisodicEntry(context_matrix=ctx_matrix, target_token=99)
    
    from holographic_prod.dreaming.compression import compress_episode
    
    # Compress
    compressed = compress_episode(episode, [proto], basis, np)
    
    # Should not compress (too different)
    assert compressed.prototype_id == -1  # Uncompressed
    assert compressed.target_token == 99


def test_compress_episode_decompression():
    """Test that compression/decompression round-trip works."""
    basis = build_clifford_basis(np)
    
    # Create prototype
    proto_matrix = np.eye(4, dtype=DTYPE)
    proto = SemanticPrototype(
        prototype_matrix=proto_matrix,
        target_distribution={10: 1.0},
        radius=0.1,
        support=5,
    )
    
    # Create episode
    ctx_matrix = proto_matrix + PHI_INV_EIGHT * np.random.randn(4, 4)
    episode = EpisodicEntry(context_matrix=ctx_matrix, target_token=10)
    
    from holographic_prod.dreaming.compression import compress_episode
    
    # Compress
    compressed = compress_episode(episode, [proto], basis, np)
    
    # Decompress
    decompressed = compressed.decompress([proto], basis, np)
    
    # Should be close to original (within sparsity threshold)
    # Use cosine similarity for bounded [-1, 1] comparison
    from holographic_prod.core.algebra import frobenius_cosine
    similarity = frobenius_cosine(ctx_matrix, decompressed, np)
    assert similarity > 0.9  # Should be very similar (cosine ≈ 1)


def test_compress_episodes_batch():
    """Test batch compression."""
    basis = build_clifford_basis(np)
    
    # Create prototype
    proto_matrix = np.eye(4, dtype=DTYPE)
    proto = SemanticPrototype(
        prototype_matrix=proto_matrix,
        target_distribution={10: 1.0},
        radius=0.1,
        support=5,
    )
    
    # Create multiple episodes
    episodes = []
    for i in range(5):
        ctx_matrix = proto_matrix + PHI_INV_EIGHT * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(context_matrix=ctx_matrix, target_token=10))
    
    from holographic_prod.dreaming.compression import compress_episodes_batch
    
    # Batch compress
    compressed_list, stats = compress_episodes_batch(episodes, [proto], basis, np)
    
    assert len(compressed_list) == 5
    assert stats['total_episodes'] == 5
    assert stats['actually_compressed'] > 0
    assert stats['compression_rate'] > 0.0
    assert stats['avg_sparsity'] >= 0.0


def test_compress_episodes_batch_empty():
    """Test batch compression with empty list."""
    basis = build_clifford_basis(np)
    
    from holographic_prod.dreaming.compression import compress_episodes_batch
    
    compressed_list, stats = compress_episodes_batch([], [], basis, np)
    
    assert len(compressed_list) == 0
    assert stats['total_episodes'] == 0
    assert stats['compression_rate'] == 0.0
