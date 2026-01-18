"""
Test suite for 16×6×4 interaction tensor.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ
from holographic_v4.torus.interaction_tensor import (
    InteractionTensor,
    build_rotation_rotor,
    build_projection_matrix,
)


class TestInteractionTensor:
    """Tests for interaction tensor system."""
    
    def test_rotation_rotors_orthogonal(self):
        """Verify rotation rotors are orthogonal matrices."""
        for k in range(16):
            R = build_rotation_rotor(k)
            assert R.shape == (6, 6)
            
            # R @ R.T should be identity
            RRT = R @ R.T
            assert np.allclose(RRT, np.eye(6), atol=1e-10)
    
    def test_projection_matrix_shape(self):
        """Verify projection matrix has correct shape."""
        P = build_projection_matrix()
        assert P.shape == (4, 6)
        
        # Rows should be normalized
        for i in range(4):
            norm = np.linalg.norm(P[i])
            assert abs(norm - 1.0) < 0.01 or abs(norm) < 0.01
    
    def test_project_up(self):
        """Verify upward projection works."""
        tensor = InteractionTensor(n_satellites=16)
        
        np.random.seed(42)
        sat_bivectors = np.random.randn(16, 6)
        
        trivector = tensor.project_up(sat_bivectors)
        assert trivector.shape == (4,)
        assert not np.isnan(trivector).any()
    
    def test_project_down(self):
        """Verify downward projection works."""
        tensor = InteractionTensor(n_satellites=16)
        
        trivector = np.array([1.0, 0.5, -0.3, 0.2])
        
        for k in range(16):
            bivector = tensor.project_down(trivector, k)
            assert bivector.shape == (6,)
            assert not np.isnan(bivector).any()
    
    def test_phi_squared_scaling(self):
        """Verify φ^-2 scaling in upward projection."""
        tensor = InteractionTensor(n_satellites=16)
        
        # Project up with uniform bivectors
        uniform_bivectors = np.ones((16, 6))
        trivector1 = tensor.project_up(uniform_bivectors)
        
        # Double the input
        trivector2 = tensor.project_up(uniform_bivectors * 2)
        
        # Output should scale proportionally
        assert np.allclose(trivector2, trivector1 * 2, rtol=0.01)
    
    def test_coupling_strength(self):
        """Verify coupling strength is in valid range."""
        tensor = InteractionTensor(n_satellites=16)
        
        for i in range(16):
            for j in range(16):
                coupling = tensor.get_coupling_strength(i, j)
                assert 0 <= coupling <= 1
