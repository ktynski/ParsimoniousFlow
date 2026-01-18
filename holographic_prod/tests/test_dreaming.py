"""
Test-Driven Development for Dreaming System.

Tests the complete dreaming system including:
- Non-REM consolidation
- REM recombination
- Schema discovery
- Memory management

Run: pytest holographic_prod/tests/test_dreaming.py -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from holographic_prod.core.constants import (
    PHI_INV_FOUR, DTYPE
)
from holographic_prod.core.algebra import build_clifford_basis
from holographic_prod.dreaming import (
    DreamingSystem,
    EpisodicEntry,
)


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

def test_dreaming_system_creation():
    """Test that DreamingSystem can be created."""
    basis = build_clifford_basis(np)
    dreaming = DreamingSystem(basis=basis)
    
    assert dreaming is not None
    assert dreaming.basis is not None
    assert dreaming.semantic_memory is not None


def test_sleep_cycle_basic():
    """Test basic sleep cycle functionality."""
    basis = build_clifford_basis(np)
    dreaming = DreamingSystem(basis=basis)
    
    # Create some test episodes
    episodes = []
    for i in range(100):
        # Random context matrix (identity + φ⁻⁴ noise)
        ctx_matrix = np.eye(4) + PHI_INV_FOUR * np.random.randn(4, 4)
        target = i % 10  # 10 different targets
        episodes.append(EpisodicEntry(ctx_matrix, target))
    
    # Run sleep cycle
    stats = dreaming.sleep(episodes, rem_cycles=1, verbose=False)
    
    # Verify stats structure
    assert 'input_episodes' in stats
    assert 'prototypes_created' in stats
    assert 'schemas_discovered' in stats
    assert stats['input_episodes'] == 100
    
    # Verify system stats
    system_stats = dreaming.get_stats()
    assert 'sleep_cycles' in system_stats
    assert system_stats['sleep_cycles'] == 1


def test_episodic_entry_creation():
    """Test EpisodicEntry creation."""
    ctx_matrix = np.eye(4, dtype=DTYPE)
    target = 42
    
    entry = EpisodicEntry(
        context_matrix=ctx_matrix,
        target_token=target
    )
    
    assert entry.context_matrix is not None
    assert entry.target_token == target
    assert entry.count == 1
    assert entry.salience == 0.0
    assert entry.novelty == 1.0


def test_multiple_sleep_cycles():
    """Test that multiple sleep cycles work correctly."""
    basis = build_clifford_basis(np)
    dreaming = DreamingSystem(basis=basis)
    
    # First sleep cycle
    episodes1 = []
    for i in range(50):
        ctx_matrix = np.eye(4) + PHI_INV_FOUR * np.random.randn(4, 4)
        target = i % 5
        episodes1.append(EpisodicEntry(ctx_matrix, target))
    
    stats1 = dreaming.sleep(episodes1, rem_cycles=1, verbose=False)
    
    # Second sleep cycle
    episodes2 = []
    for i in range(50):
        ctx_matrix = np.eye(4) + PHI_INV_FOUR * np.random.randn(4, 4)
        target = (i + 5) % 10  # Different targets
        episodes2.append(EpisodicEntry(ctx_matrix, target))
    
    stats2 = dreaming.sleep(episodes2, rem_cycles=1, verbose=False)
    
    # Verify both cycles completed
    assert stats1['input_episodes'] == 50
    assert stats2['input_episodes'] == 50
    
    # Verify system stats
    system_stats = dreaming.get_stats()
    assert system_stats['sleep_cycles'] == 2


def test_retrieval_functionality():
    """Test retrieval from semantic memory."""
    basis = build_clifford_basis(np)
    dreaming = DreamingSystem(basis=basis)
    
    # Create and consolidate episodes
    episodes = []
    for i in range(50):
        ctx_matrix = np.eye(4) + PHI_INV_FOUR * np.random.randn(4, 4)
        target = i % 5
        episodes.append(EpisodicEntry(ctx_matrix, target))
    
    dreaming.sleep(episodes, rem_cycles=1, verbose=False)
    
    # Try to retrieve
    query = np.eye(4, dtype=DTYPE)
    proto, similarity, info = dreaming.retrieve(query, top_k=1)
    
    # Should return something (even if None)
    assert info is not None
    assert 'used_pattern_completion' in info


# =============================================================================
# INTEGRATION TEST (from original __main__ block)
# =============================================================================

def test_dreaming_system_integration():
    """Integration test matching original __main__ block."""
    print("Testing Dreaming System...")
    
    # Create basis
    basis = build_clifford_basis(np)
    
    # Create dreaming system
    dreaming = DreamingSystem(basis=basis)
    
    # Create some test episodes
    episodes = []
    for i in range(100):
        # Random context matrix (identity + φ⁻⁴ noise)
        ctx_matrix = np.eye(4) + PHI_INV_FOUR * np.random.randn(4, 4)
        target = i % 10  # 10 different targets
        episodes.append(EpisodicEntry(ctx_matrix, target))
    
    # Run sleep cycle
    stats = dreaming.sleep(episodes, rem_cycles=1, verbose=True)
    
    print()
    print("Final stats:", dreaming.get_stats())
    
    # Verify basic expectations
    assert stats['input_episodes'] == 100
    assert dreaming.get_stats()['sleep_cycles'] == 1
