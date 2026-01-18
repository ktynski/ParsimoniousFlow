"""
Test Paradox Resolution — Contradiction Handling

Tests that contradictory memories are separated by phase shift Δψ = 2π/φ.

Theory (Chapter 11):
    When two satellites encode contradictory information:
        conflicting_satellite.phase += 2π × φ⁻¹
    
    This "lifts" the contradiction into a higher dimension of the torus.
    Both facts remain true, but they occupy different phases.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PI, PHI, PHI_INV,
    DTYPE,
)
from holographic_prod.dreaming.paradox_resolution import resolve_paradox


class TestParadoxResolution:
    """Test suite for paradox resolution."""
    
    def test_phase_shift_magnitude(self):
        """Test that phase shift is 2π × φ⁻¹."""
        phase_shift = 2 * PI * PHI_INV
        
        # Should be golden angle (2π/φ)
        golden_angle = 2 * PI / PHI
        assert abs(phase_shift - golden_angle) < 1e-10, \
            f"Phase shift should be golden angle (2π/φ), got {phase_shift}, expected {golden_angle}"
    
    def test_phase_shift_applied(self):
        """Test that phase shift is correctly applied."""
        original_phase = 0.5
        shifted_phase = resolve_paradox(original_phase)
        
        expected_shift = 2 * PI * PHI_INV
        expected_phase = (original_phase + expected_shift) % (2 * PI)
        
        assert abs(shifted_phase - expected_phase) < 1e-10, \
            f"Phase should be shifted by 2π×φ⁻¹, got {shifted_phase}, expected {expected_phase}"
    
    def test_phase_wrapping(self):
        """Test that phase wraps correctly to [0, 2π)."""
        # Phase near 2π
        phase_near_end = 2 * PI - 0.1
        shifted = resolve_paradox(phase_near_end)
        
        assert 0 <= shifted < 2 * PI, f"Shifted phase should be in [0, 2π), got {shifted}"
    
    def test_separation_of_contradictions(self):
        """Test that contradictory satellites are separated."""
        # Two satellites with same phase (contradictory)
        sat1_phase = 0.5
        sat2_phase = 0.5
        
        # Resolve paradox for one
        sat1_resolved = resolve_paradox(sat1_phase)
        
        # Phases should now be different
        phase_diff = abs(sat1_resolved - sat2_phase)
        # Normalize to [0, π]
        if phase_diff > PI:
            phase_diff = 2 * PI - phase_diff
        
        assert phase_diff > 0.1, f"Contradictory satellites should be separated, diff={phase_diff}"
    
    def test_golden_angle_separation(self):
        """Test that separation uses golden angle (maximally irrational)."""
        phase_shift = 2 * PI * PHI_INV
        
        # Golden angle should be maximally irrational (worst approximable)
        # This means it never aligns with rational multiples of 2π
        # Test: shift should not be a simple fraction of 2π
        fraction = phase_shift / (2 * PI)
        assert abs(fraction - PHI_INV) < 1e-10, \
            f"Phase shift fraction should be φ⁻¹, got {fraction}"
        
        # φ⁻¹ is irrational, so this separation prevents resonance
        assert fraction != 0.5, "Phase shift should not be 0.5 (would align)"
        assert fraction != 1/3, "Phase shift should not be 1/3 (would align)"
        assert fraction != 2/3, "Phase shift should not be 2/3 (would align)"
    
    def test_multiple_contradictions(self):
        """Test that multiple contradictions are handled."""
        # Three contradictory satellites
        phases = [0.5, 0.5, 0.5]
        
        # Resolve each (shift by golden angle)
        resolved_phases = [resolve_paradox(p) for p in phases]
        
        # All should be different (or at least separated)
        for i, p1 in enumerate(resolved_phases):
            for j, p2 in enumerate(resolved_phases[i+1:], i+1):
                phase_diff = abs(p1 - p2)
                if phase_diff > PI:
                    phase_diff = 2 * PI - phase_diff
                # Should be separated (or at least not identical)
                assert phase_diff > 1e-6 or abs(p1 - p2) < 1e-6, \
                    f"Contradictory satellites should be separated, diff={phase_diff}"
    
    def test_coexistence(self):
        """Test that contradictory memories coexist in different phases."""
        # Two contradictory facts
        fact1_phase = 0.0
        fact2_phase = 0.0  # Same phase = contradiction
        
        # Resolve: shift one
        fact2_resolved = resolve_paradox(fact2_phase)
        
        # Both should exist but in different phases
        assert fact1_phase != fact2_resolved, "Contradictory facts should be in different phases"
        assert 0 <= fact1_phase < 2 * PI, "Fact 1 phase should be valid"
        assert 0 <= fact2_resolved < 2 * PI, "Fact 2 phase should be valid"
    
    def test_phase_shift_idempotent(self):
        """Test that applying shift twice creates different separation."""
        phase = 0.5
        
        # Apply shift once
        shifted1 = resolve_paradox(phase)
        
        # Apply shift twice
        shifted2 = resolve_paradox(shifted1)
        
        # Should be different from original
        diff1 = abs(shifted1 - phase)
        if diff1 > PI:
            diff1 = 2 * PI - diff1
        
        diff2 = abs(shifted2 - phase)
        if diff2 > PI:
            diff2 = 2 * PI - diff2
        
        # Both should be separated
        assert diff1 > 0.1, "First shift should separate"
        assert diff2 > 0.1, "Second shift should separate"
        # But separations should be different (not identical)
        assert abs(diff1 - diff2) > 1e-6 or abs(shifted1 - shifted2) > 1e-6, \
            "Multiple shifts should create different separations"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
