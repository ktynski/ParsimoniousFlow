"""
Test suite for enhanced dreaming system.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from holographic_v4.constants import (
    PHI_INV, PHI_INV_SQ, CLIFFORD_DIM, GRADE_INDICES,
)
from holographic_v4.dreaming_enhanced import (
    EnhancedDreamingSystem,
    NonREMConsolidation,
    REMRecombination,
    ParadoxResolver,
    detect_contradiction,
    resolve_paradox,
)


class TestNonREMConsolidation:
    """Tests for Non-REM consolidation."""
    
    def test_consolidation_improves_coherence(self):
        """Verify consolidation improves satellite coherence."""
        np.random.seed(42)
        
        master = np.random.randn(CLIFFORD_DIM)
        master[GRADE_INDICES[0][0]] = 1.0
        
        satellites = np.random.randn(16, CLIFFORD_DIM) * 0.5
        
        nonrem = NonREMConsolidation()
        consolidated, stats = nonrem.consolidate(master, satellites)
        
        # Final coherence should be better than initial
        assert np.mean(stats['final_coherences']) >= np.mean(stats['initial_coherences'])
    
    def test_consolidation_preserves_shape(self):
        """Verify consolidation preserves satellite array shape."""
        np.random.seed(42)
        
        master = np.random.randn(CLIFFORD_DIM)
        satellites = np.random.randn(16, CLIFFORD_DIM)
        
        nonrem = NonREMConsolidation()
        consolidated, _ = nonrem.consolidate(master, satellites)
        
        assert consolidated.shape == satellites.shape


class TestREMRecombination:
    """Tests for REM recombination."""
    
    def test_recombination_returns_valid_result(self):
        """Verify REM returns valid multivector or None."""
        np.random.seed(42)
        satellites = np.random.randn(16, CLIFFORD_DIM)
        
        rem = REMRecombination(max_jitter_attempts=50)
        result, stats = rem.recombine(satellites, seed=42)
        
        if result is not None:
            assert result.shape == (CLIFFORD_DIM,)
            assert not np.isnan(result).any()
    
    def test_recombination_attempts_tracked(self):
        """Verify attempt count is tracked."""
        np.random.seed(42)
        satellites = np.random.randn(16, CLIFFORD_DIM)
        
        rem = REMRecombination(max_jitter_attempts=100)
        _, stats = rem.recombine(satellites, seed=42)
        
        assert 'attempts' in stats
        assert stats['attempts'] > 0
        assert stats['attempts'] <= 100


class TestParadoxResolution:
    """Tests for paradox detection and resolution."""
    
    def test_detect_contradiction(self):
        """Verify contradictory states are detected."""
        sat_a = np.zeros(CLIFFORD_DIM)
        sat_a[GRADE_INDICES[0][0]] = 1.0
        
        sat_b = np.zeros(CLIFFORD_DIM)
        sat_b[GRADE_INDICES[0][0]] = -1.0
        
        assert detect_contradiction(sat_a, sat_b)
    
    def test_no_false_positive_contradiction(self):
        """Verify similar states are not flagged as contradictions."""
        sat_a = np.zeros(CLIFFORD_DIM)
        sat_a[GRADE_INDICES[0][0]] = 1.0
        
        sat_b = np.zeros(CLIFFORD_DIM)
        sat_b[GRADE_INDICES[0][0]] = 0.9  # Similar, not contradictory
        
        assert not detect_contradiction(sat_a, sat_b)
    
    def test_resolve_paradox_changes_state(self):
        """Verify paradox resolution applies phase shift."""
        sat_a = np.zeros(CLIFFORD_DIM)
        sat_a[GRADE_INDICES[0][0]] = 1.0
        
        sat_b = np.zeros(CLIFFORD_DIM)
        sat_b[GRADE_INDICES[0][0]] = -1.0
        
        _, sat_b_resolved = resolve_paradox(sat_a, sat_b)
        
        # Should be different from original
        assert not np.allclose(sat_b, sat_b_resolved)
    
    def test_resolver_class(self):
        """Verify ParadoxResolver scans and resolves."""
        np.random.seed(42)
        
        # Create satellites with some contradictions
        satellites = np.random.randn(16, CLIFFORD_DIM) * 0.1
        satellites[0, GRADE_INDICES[0][0]] = 1.0
        satellites[1, GRADE_INDICES[0][0]] = -1.0  # Contradiction with 0
        
        resolver = ParadoxResolver()
        resolved, stats = resolver.scan_and_resolve(satellites)
        
        assert 'paradoxes_found' in stats


class TestEnhancedDreamingSystem:
    """Tests for complete dreaming system."""
    
    def test_sleep_cycle_completes(self):
        """Verify sleep cycle runs to completion."""
        np.random.seed(42)
        
        master = np.random.randn(CLIFFORD_DIM)
        satellites = np.random.randn(16, CLIFFORD_DIM)
        
        dreaming = EnhancedDreamingSystem(n_satellites=16)
        result = dreaming.sleep_cycle(master, satellites, seed=42)
        
        assert 'final_master' in result
        assert 'final_satellites' in result
        assert 'woke' in result
        assert 'stats' in result
    
    def test_sleep_cycle_returns_valid_states(self):
        """Verify sleep cycle returns valid multivectors."""
        np.random.seed(42)
        
        master = np.random.randn(CLIFFORD_DIM)
        satellites = np.random.randn(16, CLIFFORD_DIM)
        
        dreaming = EnhancedDreamingSystem(n_satellites=16)
        result = dreaming.sleep_cycle(master, satellites, seed=42)
        
        assert result['final_master'].shape == (CLIFFORD_DIM,)
        assert result['final_satellites'].shape == (16, CLIFFORD_DIM)
        assert not np.isnan(result['final_master']).any()
