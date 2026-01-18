"""
Test-Driven Development for Pattern Completion Module.

Tests pattern completion functions before extraction.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from holographic_prod.core.constants import DTYPE
from holographic_prod.core.algebra import build_clifford_basis


# =============================================================================
# TESTS FOR PATTERN COMPLETION MODULE (to be extracted)
# =============================================================================

def test_pattern_complete_identity():
    """Test pattern completion on identity matrix."""
    basis = build_clifford_basis(np)
    query = np.eye(4, dtype=DTYPE)
    
    from holographic_prod.dreaming.pattern_completion import pattern_complete
    
    completed, info = pattern_complete(query, basis, np, max_steps=5)
    
    assert completed.shape == (4, 4)
    assert info['steps_taken'] > 0
    assert 'converged' in info
    assert 'total_change' in info


def test_pattern_complete_noisy():
    """Test pattern completion on noisy input."""
    basis = build_clifford_basis(np)
    
    # Create noisy query (identity + noise)
    query = np.eye(4, dtype=DTYPE) + 0.1 * np.random.randn(4, 4)
    
    from holographic_prod.dreaming.pattern_completion import pattern_complete
    
    completed, info = pattern_complete(query, basis, np, max_steps=5)
    
    assert completed.shape == (4, 4)
    assert info['steps_taken'] > 0
    # Completed should be closer to stable pattern
    assert np.linalg.norm(completed, 'fro') > 0


def test_pattern_complete_convergence():
    """Test that pattern completion converges."""
    basis = build_clifford_basis(np)
    query = np.eye(4, dtype=DTYPE) + 0.1 * np.random.randn(4, 4)
    
    from holographic_prod.dreaming.pattern_completion import pattern_complete
    
    # With tight convergence threshold
    completed, info = pattern_complete(
        query, basis, np,
        max_steps=10,
        convergence_threshold=1e-6
    )
    
    assert info['steps_taken'] <= 10
    # If converged, change should be small
    if info['converged']:
        assert info['total_change'] < 1e-4


def test_pattern_complete_preserves_scale():
    """Test that pattern completion preserves original scale."""
    basis = build_clifford_basis(np)
    
    # Create query with specific norm
    query = 2.0 * np.eye(4, dtype=DTYPE)
    original_norm = np.linalg.norm(query, 'fro')
    
    from holographic_prod.dreaming.pattern_completion import pattern_complete
    
    completed, info = pattern_complete(query, basis, np, max_steps=5)
    completed_norm = np.linalg.norm(completed, 'fro')
    
    # Should preserve scale (within tolerance)
    assert abs(completed_norm - original_norm) < 0.1


def test_pattern_complete_batch():
    """Test batch pattern completion."""
    basis = build_clifford_basis(np)
    
    # Create batch of queries
    queries = np.array([
        np.eye(4, dtype=DTYPE) + 0.1 * np.random.randn(4, 4)
        for _ in range(3)
    ])
    
    from holographic_prod.dreaming.pattern_completion import pattern_complete_batch
    
    completed_batch, info = pattern_complete_batch(queries, basis, np, max_steps=5)
    
    assert completed_batch.shape == (3, 4, 4)
    assert info['steps_taken'] > 0
    assert info['batch_size'] == 3


def test_pattern_complete_batch_empty():
    """Test batch pattern completion with empty batch."""
    basis = build_clifford_basis(np)
    queries = np.zeros((0, 4, 4), dtype=DTYPE)
    
    from holographic_prod.dreaming.pattern_completion import pattern_complete_batch
    
    completed_batch, info = pattern_complete_batch(queries, basis, np, max_steps=5)
    
    assert completed_batch.shape == (0, 4, 4)
    assert info['steps_taken'] == 0
    assert info['converged'] is True


def test_pattern_complete_max_steps():
    """Test that pattern completion respects max_steps."""
    basis = build_clifford_basis(np)
    query = np.eye(4, dtype=DTYPE) + 0.5 * np.random.randn(4, 4)
    
    from holographic_prod.dreaming.pattern_completion import pattern_complete
    
    completed, info = pattern_complete(query, basis, np, max_steps=3)
    
    assert info['steps_taken'] <= 3
