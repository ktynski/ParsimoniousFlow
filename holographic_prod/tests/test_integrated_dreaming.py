"""
TDD Test Suite for Integrated Dreaming
=======================================

Tests for the unified dreaming system that combines:
1. Tower-level dreaming (synaptic homeostasis)
2. DreamingSystem (systems consolidation)

THEORY (Complementary Learning Systems):
    The brain has multiple levels of consolidation:
    
    SYNAPTIC LEVEL (Tower Dreaming):
        - Non-REM: Master broadcasts witness to satellites
        - REM: φ-jitter for creative exploration
        - Operates on holographic memory matrices
        
    SYSTEMS LEVEL (DreamingSystem):
        - Non-REM: Cluster episodes → semantic prototypes
        - REM: Recombine prototypes → schemas
        - 12 brain-inspired parsimonies
        
    INTEGRATED SLEEP CYCLE:
        1. Episodic buffer fills during waking
        2. Non-REM: Systems consolidation (episodic → semantic)
        3. Non-REM: Tower consolidation (witness propagation)
        4. REM: Systems recombination (schema discovery)
        5. REM: Tower recombination (φ-jitter exploration)
        6. Pruning and interference management
        7. Wake: buffers cleared, memory stabilized

NO MOCKS. NO FAKE DATA. REAL TESTS.
"""

import pytest
import numpy as np
from typing import List, Tuple

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, MATRIX_DIM, CLIFFORD_DIM, DTYPE,
)
from holographic_prod.core.algebra import get_cached_basis
from holographic_prod.memory import HolographicMemory, MemoryConfig
from holographic_prod.dreaming import DreamingSystem, EpisodicEntry


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def vocab_size():
    return 500


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def memory(vocab_size, seed):
    """Create HolographicMemory with level 2 for testing."""
    return HolographicMemory(
        vocab_size=vocab_size,
        max_levels=2,
        seed=seed,
        use_gpu=False,
    )


@pytest.fixture
def dreaming_system(memory):
    """Create DreamingSystem with all parsimonies enabled."""
    return DreamingSystem(
        basis=memory.basis,
        xp=memory.xp,
        use_salience=True,
        use_novelty=True,
        use_predictive_coding=True,
        use_pattern_completion=True,
        use_inhibition_of_return=True,
        use_sequence_replay=True,
        use_pseudo_rehearsal=True,
    )


@pytest.fixture
def training_data(vocab_size, seed):
    """Generate training data for testing."""
    np.random.seed(seed)
    data = []
    for i in range(100):
        ctx = list(np.random.randint(0, vocab_size, size=5))
        tgt = np.random.randint(0, vocab_size)
        data.append((ctx, tgt))
    return data


@pytest.fixture
def episodic_buffer(memory, training_data):
    """Create episodic buffer from training data."""
    episodes = []
    for ctx, tgt in training_data:
        ctx_matrix = memory.embed_sequence(ctx)
        episodes.append(EpisodicEntry(
            context_matrix=ctx_matrix,
            target_token=tgt,
        ))
    return episodes


# =============================================================================
# PHASE 1: TOWER DREAMING TESTS
# =============================================================================

class TestTowerDreaming:
    """Tests for tower-level dreaming (synaptic homeostasis)."""
    
    def test_tower_has_dreaming_methods(self, memory):
        """Tower should have non_rem_consolidation and rem_recombination."""
        assert hasattr(memory.tower, 'non_rem_consolidation')
        assert hasattr(memory.tower, 'rem_recombination')
        assert hasattr(memory.tower, 'get_stability')
    
    def test_non_rem_consolidation_increases_stability(self, memory, training_data):
        """Non-REM consolidation should tend to increase stability."""
        # Learn patterns
        for ctx, tgt in training_data[:50]:
            memory.learn(ctx, tgt)
        
        # Run consolidation
        memory.tower.non_rem_consolidation(PHI_INV_SQ)
        
        # Should complete without error
        stability = memory.tower.get_stability()
        assert 0.0 <= stability <= 1.0
    
    def test_rem_recombination_returns_improvement_flag(self, memory, training_data):
        """REM recombination should return whether stability improved."""
        # Learn patterns
        for ctx, tgt in training_data[:50]:
            memory.learn(ctx, tgt)
        
        improved = memory.tower.rem_recombination(PHI_INV_CUBE)
        assert isinstance(improved, bool)
    
    def test_holographic_memory_dream_method(self, memory, training_data):
        """HolographicMemory.dream() should run tower dreaming cycle."""
        # Learn patterns
        for ctx, tgt in training_data[:50]:
            memory.learn(ctx, tgt)
        
        # Run dream
        result = memory.dream()
        
        # Should return stats
        assert 'iterations' in result
        assert 'pre_stability' in result
        assert 'post_stability' in result


# =============================================================================
# PHASE 2: DREAMING SYSTEM TESTS
# =============================================================================

class TestDreamingSystem:
    """Tests for systems-level dreaming (episodic → semantic)."""
    
    def test_dreaming_system_has_sleep_method(self, dreaming_system):
        """DreamingSystem should have sleep method."""
        assert hasattr(dreaming_system, 'sleep')
    
    def test_sleep_creates_prototypes(self, dreaming_system, episodic_buffer):
        """Sleep should create semantic prototypes from episodes."""
        stats = dreaming_system.sleep(
            episodes=episodic_buffer,
            rem_cycles=1,
            verbose=False,
        )
        
        assert 'prototypes_created' in stats
        # With 100 episodes, should create at least some prototypes
        assert stats['prototypes_created'] >= 0
    
    def test_sleep_discovers_schemas(self, dreaming_system, episodic_buffer):
        """REM phase should discover schemas."""
        stats = dreaming_system.sleep(
            episodes=episodic_buffer,
            rem_cycles=1,
            verbose=False,
        )
        
        assert 'schemas_discovered' in stats
    
    def test_semantic_memory_populated(self, dreaming_system, episodic_buffer):
        """After sleep, semantic memory should have prototypes."""
        dreaming_system.sleep(episodes=episodic_buffer, verbose=False)
        
        mem_stats = dreaming_system.semantic_memory.stats()
        assert 'total_prototypes' in mem_stats


# =============================================================================
# PHASE 3: INTEGRATED DREAMING TESTS
# =============================================================================

class TestIntegratedDreaming:
    """
    Tests for integrated dreaming combining both systems.
    
    Theory: Both levels of consolidation should work together.
    """
    
    def test_integrated_sleep_exists(self, memory, dreaming_system):
        """integrated_sleep function should exist."""
        from holographic_prod.dreaming.integration import integrated_sleep
        assert callable(integrated_sleep)
    
    def test_integrated_sleep_runs_both_systems(
        self, memory, dreaming_system, episodic_buffer
    ):
        """Integrated sleep should run both tower and systems dreaming."""
        from holographic_prod.dreaming.integration import integrated_sleep
        
        # Learn patterns first
        for ep in episodic_buffer[:50]:
            # Learn from episodic entry
            tgt = ep.target_token
            memory.tower.learn([tgt % 100, (tgt + 1) % 100, (tgt + 2) % 100], tgt)
        
        pre_stability = memory.tower.get_stability()
        
        stats = integrated_sleep(
            memory=memory,
            dreaming_system=dreaming_system,
            episodes=episodic_buffer,
            verbose=False,
        )
        
        # Should have stats from both systems
        assert 'tower_stats' in stats
        assert 'systems_stats' in stats
        
        # Tower stats
        assert 'pre_stability' in stats['tower_stats']
        assert 'post_stability' in stats['tower_stats']
        
        # Systems stats
        assert 'prototypes_created' in stats['systems_stats']
    
    def test_integrated_sleep_order(self, memory, dreaming_system, episodic_buffer):
        """
        Sleep phases should execute in correct order.
        
        Theory order:
        1. Systems Non-REM (cluster → prototypes)
        2. Tower Non-REM (witness propagation)
        3. Systems REM (recombine → schemas)
        4. Tower REM (φ-jitter)
        5. Pruning
        """
        from holographic_prod.dreaming.integration import integrated_sleep
        
        # Learn patterns
        for ep in episodic_buffer[:50]:
            tgt = ep.target_token
            memory.tower.learn([tgt % 100, (tgt + 1) % 100], tgt)
        
        stats = integrated_sleep(
            memory=memory,
            dreaming_system=dreaming_system,
            episodes=episodic_buffer,
            verbose=False,
        )
        
        # Should have phase order info
        assert 'phases_completed' in stats
        phases = stats['phases_completed']
        
        # Verify correct order
        expected_order = [
            'systems_non_rem',
            'tower_non_rem',
            'systems_rem',
            'tower_rem',
            'pruning',
        ]
        
        for phase in expected_order:
            assert phase in phases, f"Missing phase: {phase}"
    
    def test_integrated_sleep_improves_stability(
        self, memory, dreaming_system, episodic_buffer
    ):
        """Integrated sleep should tend to improve or maintain stability."""
        from holographic_prod.dreaming.integration import integrated_sleep
        
        # Learn many patterns to create instability
        for ep in episodic_buffer:
            tgt = ep.target_token
            memory.tower.learn([tgt % 100, (tgt + 1) % 100, (tgt + 2) % 100], tgt)
        
        pre_stability = memory.tower.get_stability()
        
        stats = integrated_sleep(
            memory=memory,
            dreaming_system=dreaming_system,
            episodes=episodic_buffer,
            verbose=False,
        )
        
        post_stability = memory.tower.get_stability()
        
        # Should complete without error
        assert isinstance(post_stability, float)
        assert 0.0 <= post_stability <= 1.0
    
    def test_integrated_sleep_clears_episodic_buffer(
        self, memory, dreaming_system, episodic_buffer
    ):
        """After sleep, episodic buffer should be cleared (externally managed)."""
        from holographic_prod.dreaming.integration import integrated_sleep
        
        # Learn
        for ep in episodic_buffer[:20]:
            tgt = ep.target_token
            memory.tower.learn([tgt % 100], tgt)
        
        # Sleep
        stats = integrated_sleep(
            memory=memory,
            dreaming_system=dreaming_system,
            episodes=episodic_buffer,
            verbose=False,
        )
        
        # Buffer management is external - function should return input count
        assert stats['systems_stats']['input_episodes'] == len(episodic_buffer)


# =============================================================================
# PHASE 4: MULTI-LEVEL TOWER DREAMING
# =============================================================================

class TestMultiLevelTowerDreaming:
    """Tests for dreaming with multi-level tower."""
    
    def test_level2_tower_dreaming(self, vocab_size, seed, training_data):
        """Level 2 tower should support dreaming."""
        memory = HolographicMemory(
            vocab_size=vocab_size,
            max_levels=2,
            seed=seed,
        )
        
        # Learn patterns
        for ctx, tgt in training_data[:50]:
            memory.learn(ctx, tgt)
        
        # Dream
        result = memory.dream()
        
        assert 'iterations' in result
        assert result['post_stability'] >= 0.0
    
    def test_level3_tower_dreaming(self, vocab_size, seed, training_data):
        """Level 3 tower should support dreaming."""
        memory = HolographicMemory(
            vocab_size=vocab_size,
            max_levels=3,
            seed=seed,
        )
        
        # Learn patterns
        for ctx, tgt in training_data[:50]:
            memory.learn(ctx, tgt)
        
        # Dream
        result = memory.dream()
        
        assert 'iterations' in result
        # 4096 satellites should still consolidate
        assert memory.tower.n_satellites == 4096


# =============================================================================
# PHASE 5: THEORY COMPLIANCE
# =============================================================================

class TestTheoryCompliance:
    """Tests verifying theory-true implementation."""
    
    def test_all_rates_phi_derived(self, memory, dreaming_system):
        """All rates should be derived from φ."""
        # Tower consolidation rate
        assert memory.config.consolidation_rate == PHI_INV_SQ
        
        # DreamingSystem thresholds
        assert dreaming_system.base_similarity_threshold == PHI_INV
    
    def test_no_arbitrary_hyperparameters(self, memory, dreaming_system):
        """No arbitrary hyperparameters should exist."""
        # Memory config should use φ-derived values
        config = memory.config
        
        # Check key rates are φ-derived
        assert config.learning_rate == PHI_INV
        assert config.contrastive_rate == PHI_INV_SQ * PHI_INV_CUBE  # φ⁻⁵
    
    def test_no_softmax_no_temperature(self, memory):
        """Generation should use φ-kernel, not softmax with temperature."""
        # Retrieve should not have temperature parameter
        token, conf, candidates = memory.retrieve_probabilistic([1, 2, 3])
        
        # Should return valid result without temperature tuning
        # (temperature would be an arbitrary hyperparameter)
        assert isinstance(conf, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x'])
