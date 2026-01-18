"""
Test suite for GraceInverse inflation operator.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from holographic_v4.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
    CLIFFORD_DIM, GRADE_INDICES, GRACE_SCALES_FLAT,
)
from holographic_v4.torus.grace_inverse import (
    grace_inverse,
    GraceInverse,
    semantic_snap,
    GRACE_INVERSE_SCALES,
)


class TestGraceInverse:
    """Tests for GraceInverse operator."""
    
    def test_inverse_scales_correct(self):
        """Verify inverse scales are reciprocals of grace scales."""
        assert GRACE_INVERSE_SCALES[0] == 1.0
        assert abs(GRACE_INVERSE_SCALES[1] - PHI) < 1e-10
        assert abs(GRACE_INVERSE_SCALES[2] - PHI**2) < 1e-10
        assert abs(GRACE_INVERSE_SCALES[3] - PHI**3) < 1e-10
        assert abs(GRACE_INVERSE_SCALES[4] - PHI) < 1e-10  # Fibonacci exception
    
    def test_grace_inverse_reverses_grace(self):
        """Verify grace_inverse(grace(M)) = M."""
        np.random.seed(42)
        M = np.random.randn(CLIFFORD_DIM)
        
        def grace(M):
            return M * np.array(GRACE_SCALES_FLAT)
        
        M_graced = grace(M)
        M_recovered = grace_inverse(M_graced)
        
        assert np.allclose(M, M_recovered, rtol=1e-10)
    
    def test_bivector_inflation(self):
        """Verify bivectors are inflated by φ²."""
        np.random.seed(42)
        M = np.random.randn(CLIFFORD_DIM)
        
        orig_biv_norm = np.linalg.norm(M[GRADE_INDICES[2]])
        M_inflated = grace_inverse(M)
        inflated_biv_norm = np.linalg.norm(M_inflated[GRADE_INDICES[2]])
        
        ratio = inflated_biv_norm / orig_biv_norm
        assert abs(ratio - PHI**2) < 0.01
    
    def test_semantic_snap_activates_bivectors(self):
        """Verify semantic snap produces non-zero bivectors."""
        # Start with witness-only multivector
        M = np.zeros(CLIFFORD_DIM)
        M[GRADE_INDICES[0][0]] = 1.0  # Scalar
        M[GRADE_INDICES[4][0]] = 0.5  # Pseudoscalar
        
        # Apply grace (contract)
        M_graced = M * np.array(GRACE_SCALES_FLAT)
        
        # Semantic snap should restore structure
        M_snapped = semantic_snap(M_graced)
        
        biv_norm = np.linalg.norm(M_snapped[GRADE_INDICES[2]])
        assert biv_norm > 0
    
    def test_grace_inverse_class(self):
        """Verify GraceInverse class works like function."""
        np.random.seed(42)
        M = np.random.randn(CLIFFORD_DIM)
        
        gi = GraceInverse()
        M_class = gi.apply(M)
        M_func = grace_inverse(M)
        
        assert np.allclose(M_class, M_func)
    
    def test_selective_inflation(self):
        """Verify selective inflation only affects specified grades."""
        np.random.seed(42)
        M = np.random.randn(CLIFFORD_DIM)
        
        gi = GraceInverse()
        M_selective = gi.apply_selective(M, grades=[2])  # Only bivectors
        
        # Grade 0 should be unchanged
        assert np.allclose(M_selective[GRADE_INDICES[0]], M[GRADE_INDICES[0]])
        
        # Grade 2 should be inflated
        assert not np.allclose(M_selective[GRADE_INDICES[2]], M[GRADE_INDICES[2]])
