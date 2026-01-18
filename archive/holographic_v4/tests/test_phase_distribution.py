"""
Test suite for φ-offset phase distribution.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from holographic_v4.constants import PI, PHI, PHI_INV
from holographic_v4.torus.phase_distribution import (
    PhaseDistribution,
    compute_satellite_positions,
    compute_frequency_stagger,
    compute_interference_coefficient,
)


class TestPhaseDistribution:
    """Tests for phase distribution system."""
    
    def test_satellite_positions_golden_spiral(self):
        """Verify 16 satellites are placed on golden spiral."""
        positions = compute_satellite_positions(16)
        assert len(positions) == 16
        assert all(0 <= p < 2 * PI for p in positions)
        
        # Check golden spacing
        for k in range(1, 16):
            expected = (2 * PI * k * PHI_INV) % (2 * PI)
            assert abs(positions[k] - expected) < 1e-10
    
    def test_frequency_stagger_phi_powers(self):
        """Verify frequencies use φ powers."""
        freqs = compute_frequency_stagger(16, omega_base=1.0)
        assert len(freqs) == 16
        
        for k in range(16):
            expected = PHI ** (k % 4)
            assert abs(freqs[k] - expected) < 1e-10
    
    def test_no_resonance_over_time(self):
        """Verify no sustained collisions (resonance) during evolution."""
        pd = PhaseDistribution(n_satellites=16)
        
        collision_counts = {}
        for _ in range(10000):
            pd.evolve(dt=0.01)
            collisions = pd.check_collisions()
            for pair in collisions:
                collision_counts[pair] = collision_counts.get(pair, 0) + 1
        
        # No pair should resonate (stay close > 5% of time)
        max_duration = max(collision_counts.values()) if collision_counts else 0
        assert max_duration < 500, f"Resonance detected: {max_duration} steps"
    
    def test_phi_optimal_interference(self):
        """Verify φ produces minimal interference coefficient."""
        phi_coeff = compute_interference_coefficient(PHI_INV)
        
        # Test against common rationals (should be infinite or higher)
        for ratio in [0.5, 1/3, 2/3]:
            rational_coeff = compute_interference_coefficient(ratio)
            assert phi_coeff < rational_coeff or rational_coeff == float('inf')
    
    def test_phase_evolution(self):
        """Verify phase evolves correctly."""
        pd = PhaseDistribution(n_satellites=16)
        initial_phases = pd.phases.copy()
        
        pd.evolve(dt=0.1)
        
        # Phases should have changed
        assert not np.allclose(pd.phases, initial_phases)
        
        # But should still be in [0, 2π)
        assert all(0 <= p < 2 * PI for p in pd.phases)
    
    def test_reset(self):
        """Verify reset restores initial state."""
        pd = PhaseDistribution(n_satellites=16)
        initial_phases = pd.phases.copy()
        
        pd.evolve(dt=1.0)
        pd.reset()
        
        assert np.allclose(pd.phases, initial_phases)
        assert pd.time == 0.0
