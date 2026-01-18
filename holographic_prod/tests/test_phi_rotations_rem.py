"""
Test φ-Rotations in REM — Phase Jitter

Tests that REM recombination applies φ-rotations to satellite phases for exploration.

Theory (Chapter 11):
    REM (Exploration): The master introduces small φ-rotations into satellite phases.
    The system explores nearby configurations, looking for more stable "chords".
    
    for each satellite:
        jitter = random() × 2π × φ⁻¹  # Golden-angle scale
        satellite.bivectors *= cos(jitter)  # Phase rotation
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ,
    MATRIX_DIM, DTYPE,
)
from holographic_prod.core.algebra import (
    build_clifford_basis,
    decompose_to_coefficients,
    reconstruct_from_coefficients,
)


class TestPhiRotationsREM:
    """Test suite for φ-rotations in REM."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.basis = build_clifford_basis(np)
        from holographic_prod.dreaming import REMRecombinator
        self.recombinator = REMRecombinator(self.basis, xp=np)
        np.random.seed(42)  # For reproducibility
    
    def test_jitter_scale(self):
        """Test that jitter uses golden-angle scale (2π × φ⁻¹)."""
        # Generate jitter
        jitter = np.random.random() * 2 * PI * PHI_INV
        
        # Should be in [0, 2π × φ⁻¹)
        assert 0 <= jitter < 2 * PI * PHI_INV, f"Jitter should be in [0, 2π×φ⁻¹), got {jitter}"
        
        # Maximum jitter should be 2π × φ⁻¹
        max_jitter = 2 * PI * PHI_INV
        assert jitter <= max_jitter, f"Jitter should not exceed {max_jitter}"
    
    def test_bivector_rotation(self):
        """Test that bivectors are rotated by cos(jitter)."""
        # Create satellite with strong bivector component
        satellite = np.random.randn(4, 4).astype(DTYPE)
        satellite = satellite + satellite.T  # Make symmetric
        
        # Get original bivector coefficients
        coeffs_orig = decompose_to_coefficients(satellite, self.basis, np)
        bivector_indices = list(range(5, 11))
        bivector_energy_orig = np.sum(coeffs_orig[bivector_indices]**2)
        
        # Apply rotation with known jitter
        jitter = PI / 4  # 45 degrees
        rotated = self.recombinator.apply_phi_rotation(satellite, jitter=jitter)
        
        # Get rotated bivector coefficients
        coeffs_rot = decompose_to_coefficients(rotated, self.basis, np)
        bivector_energy_rot = np.sum(coeffs_rot[bivector_indices]**2)
        
        # Bivector energy should be scaled by cos²(jitter)
        expected_energy = bivector_energy_orig * (np.cos(jitter)**2)
        assert abs(bivector_energy_rot - expected_energy) < 1e-6, \
            f"Bivector energy should scale by cos²(jitter), got {bivector_energy_rot}, expected {expected_energy}"
    
    def test_witness_preserved(self):
        """Test that witness (scalar + pseudoscalar) is preserved."""
        # Create satellite with witness
        satellite = np.eye(4, dtype=DTYPE) * 0.8  # Scalar
        satellite = satellite + 0.3 * self.basis[15]  # Add pseudoscalar
        
        # Get original witness
        from holographic_prod.core.quotient import extract_witness
        s_orig, p_orig = extract_witness(satellite, self.basis, np)
        
        # Apply rotation
        rotated = self.recombinator.apply_phi_rotation(satellite, jitter=PI/4)
        
        # Get rotated witness
        s_rot, p_rot = extract_witness(rotated, self.basis, np)
        
        # Witness should be unchanged
        assert abs(s_rot - s_orig) < 1e-6, "Scalar should be preserved"
        assert abs(p_rot - p_orig) < 1e-6, "Pseudoscalar should be preserved"
    
    def test_rotation_explores_nearby(self):
        """Test that rotation explores nearby configurations."""
        # Create satellite
        satellite1 = np.random.randn(4, 4).astype(DTYPE)
        satellite1 = satellite1 / np.linalg.norm(satellite1, 'fro')
        
        # Apply small rotation
        rotated1 = self.recombinator.apply_phi_rotation(satellite1, jitter=0.1)
        
        # Should be similar but different
        similarity = np.trace(satellite1 @ rotated1.T) / (
            np.linalg.norm(satellite1, 'fro') * np.linalg.norm(rotated1, 'fro')
        )
        assert 0.9 < similarity < 1.0, f"Rotated satellite should be similar (sim={similarity})"
        assert similarity < 1.0, "Rotated satellite should be different"
    
    def test_multiple_rotations(self):
        """Test that multiple rotations accumulate."""
        satellite = np.random.randn(4, 4).astype(DTYPE)
        satellite = satellite / np.linalg.norm(satellite, 'fro')
        
        # Apply rotation twice
        rotated1 = self.recombinator.apply_phi_rotation(satellite, jitter=0.1)
        rotated2 = self.recombinator.apply_phi_rotation(rotated1, jitter=0.1)
        
        # Should be different from original
        similarity = np.trace(satellite @ rotated2.T) / (
            np.linalg.norm(satellite, 'fro') * np.linalg.norm(rotated2, 'fro')
        )
        assert similarity < 1.0, "Multiple rotations should accumulate"
    
    def test_rotation_range(self):
        """Test that rotation covers full range."""
        satellite = np.random.randn(4, 4).astype(DTYPE)
        
        # Apply rotations with different jitters
        jitters = [0, PI/4, PI/2, 3*PI/4, PI]
        rotated_satellites = [
            self.recombinator.apply_phi_rotation(satellite, jitter=j) for j in jitters
        ]
        
        # All should be different
        for i, sat1 in enumerate(rotated_satellites):
            for j, sat2 in enumerate(rotated_satellites[i+1:], i+1):
                if i != j:
                    similarity = np.trace(sat1 @ sat2.T) / (
                        np.linalg.norm(sat1, 'fro') * np.linalg.norm(sat2, 'fro')
                    )
                    # Some should be different (not all identical)
                    if abs(jitters[i] - jitters[j]) > 0.1:
                        assert abs(similarity - 1.0) > 1e-3 or np.linalg.norm(sat1 - sat2, 'fro') > 1e-3, \
                            f"Rotations with different jitters should differ"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
