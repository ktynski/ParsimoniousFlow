"""
Test-Driven Development for Pruning Module.

Tests theory-true pruning based on salience and support.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from holographic_prod.core.constants import (
    PHI_INV_CUBE, PHI_INV_FOUR, DTYPE
)
from holographic_prod.core.algebra import build_clifford_basis
from holographic_prod.dreaming.structures import SemanticPrototype
from holographic_prod.dreaming.priority import compute_salience


# =============================================================================
# TESTS FOR PRUNING MODULE (to be extracted)
# =============================================================================

def test_should_prune_prototype_low_salience_low_support():
    """Test pruning when both salience and support are low."""
    basis = build_clifford_basis(np)
    
    # Create low-salience prototype (near-zero matrix)
    low_salience_matrix = PHI_INV_CUBE * np.random.randn(4, 4)
    proto = SemanticPrototype(
        prototype_matrix=low_salience_matrix,
        target_distribution={1: 1.0},
        radius=0.1,
        support=1,  # Low support
    )
    
    from holographic_prod.dreaming.pruning import should_prune_prototype
    
    should_prune, reason = should_prune_prototype(proto, basis, np)
    
    assert should_prune is True
    assert 'low_salience' in reason.lower() or 'low_support' in reason.lower()


def test_should_prune_prototype_high_salience_high_support():
    """Test that high-salience, high-support prototypes are NOT pruned."""
    basis = build_clifford_basis(np)
    
    # Create high-salience prototype (identity matrix has high scalar component)
    high_salience_matrix = np.eye(4, dtype=DTYPE)
    proto = SemanticPrototype(
        prototype_matrix=high_salience_matrix,
        target_distribution={1: 1.0},
        radius=0.1,
        support=10,  # High support
    )
    
    from holographic_prod.dreaming.pruning import should_prune_prototype
    
    should_prune, reason = should_prune_prototype(proto, basis, np)
    
    assert should_prune is False
    assert reason == "healthy" or "healthy" in reason.lower()


def test_should_prune_prototype_conservative_strategy():
    """Test that pruning requires BOTH low salience AND low support."""
    basis = build_clifford_basis(np)
    
    # High salience but low support - should NOT prune
    high_salience_matrix = np.eye(4, dtype=DTYPE)
    proto1 = SemanticPrototype(
        prototype_matrix=high_salience_matrix,
        target_distribution={1: 1.0},
        radius=0.1,
        support=1,  # Low support but high salience
    )
    
    # Low salience but high support - should NOT prune
    low_salience_matrix = PHI_INV_CUBE * np.random.randn(4, 4)
    proto2 = SemanticPrototype(
        prototype_matrix=low_salience_matrix,
        target_distribution={1: 1.0},
        radius=0.1,
        support=10,  # High support but low salience
    )
    
    from holographic_prod.dreaming.pruning import should_prune_prototype
    
    should_prune1, _ = should_prune_prototype(proto1, basis, np)
    should_prune2, _ = should_prune_prototype(proto2, basis, np)
    
    # Conservative: need BOTH low
    assert should_prune1 is False  # High salience protects it
    assert should_prune2 is False  # High support protects it


def test_prune_semantic_memory():
    """Test pruning of semantic memory."""
    basis = build_clifford_basis(np)
    
    # Create semantic memory with mix of prototypes
    from holographic_prod.dreaming import SemanticMemory
    
    memory = SemanticMemory(basis, np)
    
    # Add some prototypes
    for i in range(5):
        matrix = np.eye(4, dtype=DTYPE) + PHI_INV_FOUR * np.random.randn(4, 4)
        proto = SemanticPrototype(
            prototype_matrix=matrix,
            target_distribution={i: 1.0},
            radius=0.1,
            support=5 if i < 3 else 1,  # First 3 have high support
        )
        memory.add_prototype(proto, level=0)
    
    initial_count = memory.stats()['total_prototypes']
    
    from holographic_prod.dreaming.pruning import prune_semantic_memory
    
    stats = prune_semantic_memory(memory, basis, np, verbose=False)
    
    assert stats['total_before'] == initial_count
    assert stats['total_after'] <= stats['total_before']
    assert stats['pruned'] >= 0


def test_prune_attractor_map():
    """Test pruning of attractor map."""
    basis = build_clifford_basis(np)
    
    # Create attractors with varying salience
    num_attractors = 5
    attractor_matrices = np.zeros((10, 4, 4), dtype=DTYPE)
    attractor_targets = np.zeros(10, dtype=np.int32)
    
    for i in range(num_attractors):
        if i < 2:
            # High salience (identity)
            attractor_matrices[i] = np.eye(4, dtype=DTYPE)
        else:
            # Low salience (near-zero)
            attractor_matrices[i] = PHI_INV_CUBE * np.random.randn(4, 4)
        attractor_targets[i] = i
    
    from holographic_prod.dreaming.pruning import prune_attractor_map
    
    new_matrices, new_targets, new_count, stats = prune_attractor_map(
        attractor_matrices, attractor_targets, num_attractors, basis, np, verbose=False
    )
    
    assert new_count <= num_attractors
    assert stats['pruned'] >= 0
    assert stats['total_before'] == num_attractors
    assert stats['total_after'] == new_count
