"""
Test Chirality Alternation — Satellite Handedness

Tests that even satellites are right-handed and odd satellites are left-handed.
This prevents interference between satellites in the fractal torus.

Theory (Chapter 11):
    Even satellites (k mod 2 == 0): Right-handed (standard orientation)
    Odd satellites (k mod 2 == 1): Left-handed (mirrored orientation)
    
    This creates topological friction that prevents destructive interference.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PHI, PHI_INV,
    CLIFFORD_DIM, MATRIX_DIM, DTYPE,
    GRADE_INDICES,
)


class TestChirality:
    """Test suite for chirality alternation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_satellites = 16
    
    def test_chirality_alternation(self):
        """Test that chirality alternates correctly."""
        from holographic_prod.torus.chirality import ChiralityFlip
        
        chirality = ChiralityFlip(n_satellites=self.n_satellites)
        
        # Even satellites should be right-handed
        assert chirality.is_right_handed(0), "Satellite 0 should be right-handed"
        assert chirality.is_right_handed(2), "Satellite 2 should be right-handed"
        assert chirality.is_right_handed(14), "Satellite 14 should be right-handed"
        
        # Odd satellites should be left-handed
        assert not chirality.is_right_handed(1), "Satellite 1 should be left-handed"
        assert not chirality.is_right_handed(3), "Satellite 3 should be left-handed"
        assert not chirality.is_right_handed(15), "Satellite 15 should be left-handed"
    
    def test_chirality_apply(self):
        """Test that chirality flip is applied correctly."""
        from holographic_prod.torus.chirality import ChiralityFlip
        
        chirality = ChiralityFlip(n_satellites=self.n_satellites)
        
        # Create a test multivector
        mv = np.random.randn(CLIFFORD_DIM).astype(DTYPE)
        
        # Apply chirality to even satellite (should preserve)
        mv_right = chirality.apply(mv, satellite_index=0)
        
        # Apply chirality to odd satellite (should flip)
        mv_left = chirality.apply(mv, satellite_index=1)
        
        # Right-handed should be similar to original
        similarity_right = np.dot(mv / np.linalg.norm(mv), mv_right / np.linalg.norm(mv_right))
        
        # Left-handed should be different (flipped)
        similarity_left = np.dot(mv / np.linalg.norm(mv), mv_left / np.linalg.norm(mv_left))
        
        # Right-handed should be close to 1 (preserved)
        assert similarity_right > 0.9, f"Right-handed should preserve, got {similarity_right}"
        
        # Left-handed should be different (not necessarily negative, but different)
        assert abs(similarity_left - similarity_right) > 0.1, \
            f"Left-handed should differ from right-handed, got {similarity_left} vs {similarity_right}"
    
    def test_chirality_preserves_witness(self):
        """Test that chirality preserves witness (scalar + pseudoscalar)."""
        from holographic_prod.torus.chirality import ChiralityFlip
        from holographic_prod.core.quotient import extract_witness
        
        chirality = ChiralityFlip(n_satellites=self.n_satellites)
        
        # Create multivector with strong witness
        mv = np.zeros(CLIFFORD_DIM, dtype=DTYPE)
        mv[0] = 0.8  # Scalar
        mv[15] = 0.6  # Pseudoscalar
        
        # Apply chirality
        mv_flipped = chirality.apply(mv, satellite_index=1)  # Odd = left-handed
        
        # Extract witness before and after
        from holographic_prod.core.algebra import build_clifford_basis
        basis = build_clifford_basis(np)
        
        s_before, p_before = extract_witness(
            np.sum([mv[i] * basis[i] for i in range(16)], axis=0),
            basis, np
        )
        s_after, p_after = extract_witness(
            np.sum([mv_flipped[i] * basis[i] for i in range(16)], axis=0),
            basis, np
        )
        
        # Witness should be preserved (chirality only affects bivectors/trivectors)
        assert abs(s_after - s_before) < 1e-6, "Scalar should be preserved"
        assert abs(p_after - p_before) < 1e-6, "Pseudoscalar should be preserved"
    
    def test_chirality_flips_bivectors(self):
        """Test that chirality flips bivector components for odd satellites."""
        from holographic_prod.torus.chirality import ChiralityFlip
        
        chirality = ChiralityFlip(n_satellites=self.n_satellites)
        
        # Create multivector with strong bivector component
        mv = np.zeros(CLIFFORD_DIM, dtype=DTYPE)
        bivector_indices = GRADE_INDICES[2]  # [5, 6, 7, 8, 9, 10]
        mv[bivector_indices] = 0.5
        
        # Apply to even satellite (right-handed, should preserve)
        mv_right = chirality.apply(mv, satellite_index=0)
        
        # Apply to odd satellite (left-handed, should flip)
        mv_left = chirality.apply(mv, satellite_index=1)
        
        # Bivector components should differ between right and left
        biv_right = mv_right[bivector_indices]
        biv_left = mv_left[bivector_indices]
        
        # Should be different (flipped)
        assert np.linalg.norm(biv_right - biv_left) > 1e-6, \
            "Bivectors should differ between right and left-handed"
    
    def test_chirality_all_satellites(self):
        """Test that all satellites have correct chirality."""
        from holographic_prod.torus.chirality import ChiralityFlip
        
        chirality = ChiralityFlip(n_satellites=self.n_satellites)
        
        # Check all satellites
        for k in range(self.n_satellites):
            is_right = chirality.is_right_handed(k)
            expected_right = (k % 2 == 0)
            
            assert is_right == expected_right, \
                f"Satellite {k} should be {'right' if expected_right else 'left'}-handed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
