"""
Chirality — Even/Odd Handedness Flip for Topological Friction
=============================================================

Implements handedness alternation between satellites:
- Even k (0, 2, 4, ...): Right-handed Clifford basis
- Odd k (1, 3, 5, ...): Left-handed Clifford basis

This creates "topological friction" that prevents all satellites from
rotating in the same direction. The master torus sits at the center
of counter-rotating gears, and equilibrium only occurs when meaning
is consistent across all scales.

Theory:
    Right-handed: e1 ∧ e2 ∧ e3 = +e123
    Left-handed:  e1 ∧ e2 ∧ e3 = -e123
    
    The sign flip affects trivector and pseudoscalar components.

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import Literal
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import CLIFFORD_DIM, GRADE_INDICES


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_chirality(k: int) -> Literal['right', 'left']:
    """
    Get chirality (handedness) for satellite k.
    
    Simple rule: even = right, odd = left.
    
    Args:
        k: Satellite index
        
    Returns:
        'right' or 'left'
    """
    return 'right' if k % 2 == 0 else 'left'


def build_chirality_operator(chirality: Literal['right', 'left']) -> np.ndarray:
    """
    Build the chirality transformation operator.
    
    For left-handed basis, we flip the sign of:
    - Grade 3 (trivectors): e012, e013, e023, e123
    - Grade 4 (pseudoscalar): e0123
    
    These are the components that change sign under orientation reversal.
    
    Args:
        chirality: 'right' (identity) or 'left' (flip grades 3,4)
        
    Returns:
        Diagonal operator matrix [16, 16]
    """
    # Start with identity
    op = np.eye(CLIFFORD_DIM)
    
    if chirality == 'left':
        # Flip sign of grade 3 (trivectors)
        for idx in GRADE_INDICES[3]:
            op[idx, idx] = -1.0
        
        # Flip sign of grade 4 (pseudoscalar)
        for idx in GRADE_INDICES[4]:
            op[idx, idx] = -1.0
    
    return op


def apply_chirality_flip(
    M: np.ndarray,
    from_chirality: Literal['right', 'left'],
    to_chirality: Literal['right', 'left']
) -> np.ndarray:
    """
    Transform multivector between chiralities.
    
    If chiralities match, returns M unchanged.
    If different, flips sign of grades 3 and 4.
    
    Args:
        M: Multivector of shape [..., 16]
        from_chirality: Current chirality
        to_chirality: Target chirality
        
    Returns:
        Transformed multivector
    """
    if from_chirality == to_chirality:
        return M.copy()
    
    M_transformed = M.copy()
    
    # Flip grades 3 and 4
    for idx in GRADE_INDICES[3]:
        M_transformed[..., idx] *= -1.0
    for idx in GRADE_INDICES[4]:
        M_transformed[..., idx] *= -1.0
    
    return M_transformed


def compute_chirality_friction(
    satellites: np.ndarray,
    chiralities: np.ndarray
) -> float:
    """
    Compute total "friction" from chirality mixing.
    
    Friction is high when counter-rotating satellites have
    conflicting signals. This forces the system toward equilibrium.
    
    Args:
        satellites: Satellite states [n_satellites, 16]
        chiralities: Chirality for each satellite [n_satellites]
        
    Returns:
        Friction score (higher = more conflict)
    """
    n = len(satellites)
    friction = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            if chiralities[i] != chiralities[j]:
                # Counter-rotating pair
                # Compute overlap in grades 3 and 4
                g3_i = satellites[i, GRADE_INDICES[3]]
                g3_j = satellites[j, GRADE_INDICES[3]]
                g4_i = satellites[i, GRADE_INDICES[4]]
                g4_j = satellites[j, GRADE_INDICES[4]]
                
                # Friction is dot product (opposite signs mean conflict)
                friction += np.abs(np.dot(g3_i, g3_j))
                friction += np.abs(np.dot(g4_i, g4_j))
    
    # Normalize by number of pairs
    n_pairs = n * (n - 1) / 2
    return friction / (n_pairs + 1e-10)


# =============================================================================
# CHIRALITY MANAGER CLASS
# =============================================================================

@dataclass
class ChiralityManager:
    """
    Manages chirality for satellite system.
    
    Handles:
    - Chirality assignment (even/odd rule)
    - Transformation between chiralities
    - Friction computation
    - Master torus integration
    
    Attributes:
        n_satellites: Number of satellites
        chiralities: Chirality assignments
        operators: Pre-computed chirality operators
    """
    n_satellites: int = 16
    
    def __post_init__(self):
        """Initialize chirality assignments."""
        self.chiralities = [get_chirality(k) for k in range(self.n_satellites)]
        self.operators = {
            'right': build_chirality_operator('right'),
            'left': build_chirality_operator('left'),
        }
    
    def get_operator(self, k: int) -> np.ndarray:
        """Get chirality operator for satellite k."""
        chirality = self.chiralities[k]
        return self.operators[chirality]
    
    def apply(self, M: np.ndarray, k: int) -> np.ndarray:
        """
        Apply chirality transformation for satellite k.
        
        Args:
            M: Multivector [16]
            k: Satellite index
            
        Returns:
            Transformed multivector
        """
        return self.get_operator(k) @ M
    
    def apply_inverse(self, M: np.ndarray, k: int) -> np.ndarray:
        """
        Apply inverse chirality transformation.
        
        Since the operator is its own inverse (diagonal with ±1),
        this is the same as apply().
        
        Args:
            M: Multivector [16]
            k: Satellite index
            
        Returns:
            Transformed multivector
        """
        return self.apply(M, k)  # Self-inverse
    
    def to_master_frame(self, satellite_states: np.ndarray) -> np.ndarray:
        """
        Transform all satellites to master (right-handed) frame.
        
        Args:
            satellite_states: States [n_satellites, 16]
            
        Returns:
            Transformed states [n_satellites, 16]
        """
        transformed = np.zeros_like(satellite_states)
        for k in range(self.n_satellites):
            if self.chiralities[k] == 'right':
                transformed[k] = satellite_states[k]
            else:
                transformed[k] = apply_chirality_flip(
                    satellite_states[k], 'left', 'right'
                )
        return transformed
    
    def from_master_frame(self, master_states: np.ndarray) -> np.ndarray:
        """
        Transform from master frame back to native chirality.
        
        Args:
            master_states: States in master frame [n_satellites, 16]
            
        Returns:
            States in native chirality [n_satellites, 16]
        """
        # Same operation (self-inverse)
        return self.to_master_frame(master_states)
    
    def compute_friction(self, satellite_states: np.ndarray) -> float:
        """
        Compute friction from chirality mixing.
        
        Args:
            satellite_states: States [n_satellites, 16]
            
        Returns:
            Friction score
        """
        chirality_array = np.array([
            0 if c == 'right' else 1 for c in self.chiralities
        ])
        return compute_chirality_friction(satellite_states, chirality_array)


# =============================================================================
# TESTS
# =============================================================================

def _test_chirality():
    """Test chirality system."""
    print("Testing chirality...")
    
    # Test 1: Even/odd assignment
    for k in range(16):
        expected = 'right' if k % 2 == 0 else 'left'
        assert get_chirality(k) == expected, f"Satellite {k} chirality wrong"
    
    # Test 2: Operator structure
    right_op = build_chirality_operator('right')
    left_op = build_chirality_operator('left')
    
    assert np.allclose(right_op, np.eye(16)), "Right operator should be identity"
    assert not np.allclose(left_op, np.eye(16)), "Left operator should differ"
    
    # Check that left flips grades 3 and 4
    for idx in GRADE_INDICES[3]:
        assert left_op[idx, idx] == -1, f"Grade 3 index {idx} should flip"
    for idx in GRADE_INDICES[4]:
        assert left_op[idx, idx] == -1, f"Grade 4 index {idx} should flip"
    
    # Test 3: Self-inverse
    M = np.random.randn(16)
    M_left = apply_chirality_flip(M, 'right', 'left')
    M_back = apply_chirality_flip(M_left, 'left', 'right')
    assert np.allclose(M, M_back), "Chirality flip should be self-inverse"
    
    # Test 4: Manager
    manager = ChiralityManager(n_satellites=16)
    assert len(manager.chiralities) == 16
    
    # Test 5: To/from master frame
    states = np.random.randn(16, 16)
    master_states = manager.to_master_frame(states)
    recovered = manager.from_master_frame(master_states)
    assert np.allclose(states, recovered), "Round-trip should preserve states"
    
    # Test 6: Friction computation
    friction = manager.compute_friction(states)
    assert friction >= 0, "Friction should be non-negative"
    
    print("✓ Chirality tests passed!")
    return True


if __name__ == "__main__":
    _test_chirality()
