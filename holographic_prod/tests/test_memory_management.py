"""
Test-Driven Development for Memory Management Module.

Tests theory-true φ-decay forgetting and adaptive thresholds.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR, DTYPE
)
from holographic_prod.core.algebra import build_clifford_basis
from holographic_prod.dreaming.structures import EpisodicEntry, SemanticPrototype
from holographic_prod.dreaming.priority import compute_salience


# =============================================================================
# TESTS FOR MEMORY MANAGEMENT MODULE (to be extracted)
# =============================================================================

def test_phi_decay_forget_below_capacity():
    """Test that φ-decay doesn't forget when below capacity."""
    basis = build_clifford_basis(np)
    
    episodes = []
    for i in range(10):
        ctx_matrix = np.eye(4, dtype=DTYPE) + PHI_INV_FOUR * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(context_matrix=ctx_matrix, target_token=i))
    
    from holographic_prod.dreaming.memory_management import phi_decay_forget
    
    surviving, num_forgotten = phi_decay_forget(episodes, max_episodes=20, basis=basis, xp=np)
    
    assert len(surviving) == 10
    assert num_forgotten == 0


def test_phi_decay_forget_above_capacity():
    """Test φ-decay forgetting when above capacity."""
    basis = build_clifford_basis(np)
    
    episodes = []
    for i in range(100):
        ctx_matrix = np.eye(4, dtype=DTYPE) + PHI_INV_FOUR * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(context_matrix=ctx_matrix, target_token=i))
    
    from holographic_prod.dreaming.memory_management import phi_decay_forget
    
    max_episodes = 50
    surviving, num_forgotten = phi_decay_forget(episodes, max_episodes, basis=basis, xp=np)
    
    assert len(surviving) == max_episodes
    assert num_forgotten == 100 - max_episodes
    assert num_forgotten > 0


def test_phi_decay_forget_priority_based():
    """Test that φ-decay preserves high-priority episodes."""
    basis = build_clifford_basis(np)
    
    # Create episodes with varying salience
    episodes = []
    for i in range(20):
        if i < 5:
            # High salience (identity matrix)
            ctx_matrix = np.eye(4, dtype=DTYPE)
        else:
            # Low salience (near-zero)
            ctx_matrix = PHI_INV_CUBE * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(context_matrix=ctx_matrix, target_token=i))
    
    from holographic_prod.dreaming.memory_management import phi_decay_forget
    
    surviving, num_forgotten = phi_decay_forget(episodes, max_episodes=10, basis=basis, xp=np)
    
    # High-salience episodes should survive
    surviving_matrices = [ep.context_matrix for ep in surviving]
    high_salience_survived = sum(
        1 for m in surviving_matrices
        if compute_salience(m, basis, np) > 0.5
    )
    
    # At least some high-salience episodes should survive
    assert high_salience_survived >= 2


def test_compute_adaptive_threshold_low_diversity():
    """Test adaptive threshold when prototype diversity is low."""
    basis = build_clifford_basis(np)
    
    from holographic_prod.dreaming import SemanticMemory
    
    memory = SemanticMemory(basis, np)
    
    # Add few prototypes (low diversity)
    for i in range(5):
        matrix = np.eye(4, dtype=DTYPE) + PHI_INV_FOUR * np.random.randn(4, 4)
        proto = SemanticPrototype(
            prototype_matrix=matrix,
            target_distribution={i: 1.0},
            radius=0.1,
            support=5,
        )
        memory.add_prototype(proto, level=0)
    
    from holographic_prod.dreaming.memory_management import compute_adaptive_threshold
    
    threshold = compute_adaptive_threshold(memory)
    
    # Low diversity → should use lower threshold (encourage more clusters)
    assert threshold >= PHI_INV_CUBE  # Min threshold
    assert threshold <= PHI_INV  # Max threshold


def test_compute_adaptive_threshold_high_diversity():
    """Test adaptive threshold when prototype diversity is high."""
    basis = build_clifford_basis(np)
    
    from holographic_prod.dreaming import SemanticMemory
    
    memory = SemanticMemory(basis, np)
    
    # Add many prototypes (high diversity)
    for i in range(50):
        matrix = np.eye(4, dtype=DTYPE) + 0.3 * np.random.randn(4, 4)
        proto = SemanticPrototype(
            prototype_matrix=matrix,
            target_distribution={i: 1.0},
            radius=0.1,
            support=5,
        )
        memory.add_prototype(proto, level=0)
    
    from holographic_prod.dreaming.memory_management import compute_adaptive_threshold
    
    threshold = compute_adaptive_threshold(memory)
    
    # High diversity → threshold should be adjusted
    assert threshold >= PHI_INV_CUBE  # Min threshold
    assert threshold <= PHI_INV  # Max threshold


def test_compute_adaptive_threshold_bounds():
    """Test that adaptive threshold respects bounds."""
    basis = build_clifford_basis(np)
    
    from holographic_prod.dreaming import SemanticMemory, compute_adaptive_threshold
    
    memory = SemanticMemory(basis, np)
    
    # Add some prototypes
    for i in range(20):
        matrix = np.eye(4, dtype=DTYPE) + PHI_INV_FOUR * np.random.randn(4, 4)
        proto = SemanticPrototype(
            prototype_matrix=matrix,
            target_distribution={i: 1.0},
            radius=0.1,
            support=5,
        )
        memory.add_prototype(proto, level=0)
    
    min_threshold = PHI_INV_CUBE
    max_threshold = PHI_INV
    
    threshold = compute_adaptive_threshold(
        memory,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )
    
    assert min_threshold <= threshold <= max_threshold


def test_phi_decay_forget_theory_true():
    """Test that φ-decay uses theory-true survival probability."""
    basis = build_clifford_basis(np)
    
    # Create episodes with known priorities
    episodes = []
    for i in range(20):
        if i < 5:
            # High priority (identity = high salience)
            ctx_matrix = np.eye(4, dtype=DTYPE)
        else:
            # Low priority (near-zero = low salience)
            ctx_matrix = PHI_INV_CUBE * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(context_matrix=ctx_matrix, target_token=i))
    
    from holographic_prod.dreaming.memory_management import phi_decay_forget
    
    # Run multiple times to check statistical behavior
    survival_counts = np.zeros(20)
    n_runs = 10
    
    for _ in range(n_runs):
        surviving, _ = phi_decay_forget(episodes, max_episodes=10, basis=basis, xp=np)
        for ep in surviving:
            # Find index of surviving episode
            for idx, orig_ep in enumerate(episodes):
                if np.allclose(ep.context_matrix, orig_ep.context_matrix):
                    survival_counts[idx] += 1
                    break
    
    # High-priority episodes (first 5) should survive more often
    high_priority_survival_rate = survival_counts[:5].mean() / n_runs
    low_priority_survival_rate = survival_counts[5:].mean() / n_runs
    
    # High priority should survive more (theory-true)
    assert high_priority_survival_rate > low_priority_survival_rate
