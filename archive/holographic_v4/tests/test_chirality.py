"""
Test suite for even/odd chirality system.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from holographic_v4.constants import CLIFFORD_DIM, GRADE_INDICES
from holographic_v4.torus.chirality import (
    ChiralityManager,
    get_chirality,
    apply_chirality_flip,
    build_chirality_operator,
)


class TestChirality:
    """Tests for chirality system."""
    
    def test_even_odd_assignment(self):
        """Verify even satellites are right-handed, odd are left-handed."""
        for k in range(16):
            expected = 'right' if k % 2 == 0 else 'left'
            assert get_chirality(k) == expected
    
    def test_chirality_operator_right(self):
        """Verify right-handed operator is identity."""
        op = build_chirality_operator('right')
        assert np.allclose(op, np.eye(CLIFFORD_DIM))
    
    def test_chirality_operator_left_flips_grades_3_4(self):
        """Verify left-handed operator flips grades 3 and 4."""
        op = build_chirality_operator('left')
        
        # Grades 0, 1, 2 should be unchanged
        for grade in [0, 1, 2]:
            for idx in GRADE_INDICES[grade]:
                assert op[idx, idx] == 1.0
        
        # Grades 3 and 4 should be flipped
        for grade in [3, 4]:
            for idx in GRADE_INDICES[grade]:
                assert op[idx, idx] == -1.0
    
    def test_chirality_flip_self_inverse(self):
        """Verify chirality flip is its own inverse."""
        np.random.seed(42)
        M = np.random.randn(CLIFFORD_DIM)
        
        M_flipped = apply_chirality_flip(M, 'right', 'left')
        M_back = apply_chirality_flip(M_flipped, 'left', 'right')
        
        assert np.allclose(M, M_back)
    
    def test_manager_to_from_master_frame(self):
        """Verify round-trip through master frame preserves states."""
        manager = ChiralityManager(n_satellites=16)
        
        np.random.seed(42)
        states = np.random.randn(16, CLIFFORD_DIM)
        
        master_frame = manager.to_master_frame(states)
        recovered = manager.from_master_frame(master_frame)
        
        assert np.allclose(states, recovered)
    
    def test_friction_non_negative(self):
        """Verify friction is non-negative."""
        manager = ChiralityManager(n_satellites=16)
        
        np.random.seed(42)
        states = np.random.randn(16, CLIFFORD_DIM)
        
        friction = manager.compute_friction(states)
        assert friction >= 0
