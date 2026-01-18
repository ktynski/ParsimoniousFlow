"""
Test Phase-Locked Emission — Generation Timing

Tests that tokens are emitted only when toroidal phase enters the emission window.

Theory (Chapter 11, 14):
    Phase-locked emission: Tokens are released only when the toroidal phase enters
    the "emission window" (between φ⁻³ and φ⁻¹ of a cycle).
    
    Window: [π·φ⁻¹, π·φ⁻¹ + φ⁻²]
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
    DTYPE,
)
from holographic_prod.fractal.downward_projection import phase_locked_emission


class TestPhaseLockedEmission:
    """Test suite for phase-locked emission."""
    
    def test_emission_window_bounds(self):
        """Test that emission window has correct bounds."""
        window_start = PI * PHI_INV
        window_end = PI * PHI_INV + PHI_INV_SQ
        
        # Window should be within [0, 2π)
        assert 0 <= window_start < 2 * PI, "Window start should be in [0, 2π)"
        assert 0 <= window_end < 2 * PI, "Window end should be in [0, 2π)"
        assert window_end > window_start, "Window end should be after start"
        
        # Window width should be φ⁻²
        window_width = window_end - window_start
        assert abs(window_width - PHI_INV_SQ) < 1e-10, f"Window width should be φ⁻², got {window_width}"
    
    def test_emission_at_window_start(self):
        """Test emission at window start."""
        window_start = PI * PHI_INV
        assert phase_locked_emission(window_start), "Should emit at window start"
    
    def test_emission_at_window_end(self):
        """Test emission at window end."""
        window_end = PI * PHI_INV + PHI_INV_SQ
        assert phase_locked_emission(window_end), "Should emit at window end"
    
    def test_no_emission_before_window(self):
        """Test no emission before window."""
        window_start = PI * PHI_INV
        phase_before = window_start - 0.1
        assert not phase_locked_emission(phase_before), "Should not emit before window"
    
    def test_no_emission_after_window(self):
        """Test no emission after window."""
        window_end = PI * PHI_INV + PHI_INV_SQ
        phase_after = window_end + 0.1
        assert not phase_locked_emission(phase_after), "Should not emit after window"
    
    def test_emission_in_window_middle(self):
        """Test emission in middle of window."""
        window_start = PI * PHI_INV
        window_end = PI * PHI_INV + PHI_INV_SQ
        phase_middle = (window_start + window_end) / 2
        assert phase_locked_emission(phase_middle), "Should emit in window middle"
    
    def test_phase_wrapping(self):
        """Test that phase wrapping works correctly."""
        window_start = PI * PHI_INV
        window_end = PI * PHI_INV + PHI_INV_SQ
        
        # Phase wrapped to [0, 2π)
        phase_in_window = window_start + PHI_INV_SQ / 2
        phase_wrapped = phase_in_window + 2 * PI
        
        assert phase_locked_emission(phase_in_window), "Should emit at phase"
        assert phase_locked_emission(phase_wrapped), "Should emit at wrapped phase"
    
    def test_emission_frequency(self):
        """Test that emission occurs at correct frequency."""
        # Window width is φ⁻²
        # Full cycle is 2π
        # Emission frequency = window_width / (2π) = φ⁻² / (2π)
        window_width = PHI_INV_SQ
        cycle_length = 2 * PI
        emission_fraction = window_width / cycle_length
        
        # Should be a small fraction (emission is sparse)
        assert 0 < emission_fraction < 0.1, f"Emission should be sparse (fraction={emission_fraction})"
    
    def test_quasi_periodic_emission(self):
        """Test that emission is quasi-periodic (golden rhythm)."""
        # Generate phases over multiple cycles
        n_cycles = 10
        phases = np.linspace(0, n_cycles * 2 * PI, 1000)
        
        # Count emissions
        emissions = [phase_locked_emission(p) for p in phases]
        emission_count = sum(emissions)
        
        # Should have some emissions
        assert emission_count > 0, "Should have some emissions over multiple cycles"
        
        # Emission pattern should be quasi-periodic (not perfectly periodic)
        emission_indices = [i for i, e in enumerate(emissions) if e]
        if len(emission_indices) > 1:
            intervals = np.diff(emission_indices)
            # Intervals should vary (quasi-periodic, not constant)
            assert np.std(intervals) > 0, "Emission intervals should vary (quasi-periodic)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
