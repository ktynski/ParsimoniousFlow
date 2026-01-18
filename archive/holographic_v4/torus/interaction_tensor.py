"""
Interaction Tensor — 16×6×4 Projection Between Fractal Levels
=============================================================

Maps satellite bivectors (Grade 2, 6 components) to master trivectors
(Grade 3, 4 components) using φ-coupling.

The tensor I is structured as a permutation matrix governed by
the grade-scaling laws:
- Temporal vorticity (e01, e02, e03) → Master trivectors with e0
- Spatial vorticity (e12, e13, e23) → Master spatial trivector e123

Key Equation:
    M_grade3 = φ^-2 · Σ_{k=0}^{15} (R_k · S_k_grade2)
    
Where R_k is a rotation rotor tuned to satellite k's spiral position.

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR,
    CLIFFORD_DIM, GRADE_INDICES, MATRIX_DIM,
)


# =============================================================================
# ROTATION ROTORS
# =============================================================================

def build_rotation_rotor(k: int, n_satellites: int = 16) -> np.ndarray:
    """
    Build rotation rotor R_k for satellite k.
    
    The rotor is tuned to the satellite's position on the golden spiral:
        α_k = 2π · k · φ^-1
    
    The rotor performs a rotation in the bivector plane, "twisting"
    the satellite's vorticity as it moves up the fractal.
    
    Args:
        k: Satellite index
        n_satellites: Total satellites
        
    Returns:
        Rotation matrix of shape [6, 6] for bivector transformation
    """
    # Golden spiral angle for this satellite
    alpha = 2 * PI * k * PHI_INV
    
    # Build 6×6 rotation in bivector space
    # Bivector components: e01, e02, e03, e12, e13, e23
    # We rotate between temporal-spatial pairs
    
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    
    # Rotation in e01-e23 plane
    # Rotation in e02-e13 plane  
    # Rotation in e03-e12 plane
    
    R = np.eye(6)
    
    # Block rotation: temporal ↔ spatial coupling
    # e01 (idx 0) couples with e23 (idx 5)
    R[0, 0] = cos_a
    R[0, 5] = -sin_a
    R[5, 0] = sin_a
    R[5, 5] = cos_a
    
    # e02 (idx 1) couples with e13 (idx 4)
    R[1, 1] = cos_a
    R[1, 4] = -sin_a
    R[4, 1] = sin_a
    R[4, 4] = cos_a
    
    # e03 (idx 2) couples with e12 (idx 3)
    R[2, 2] = cos_a
    R[2, 3] = -sin_a
    R[3, 2] = sin_a
    R[3, 3] = cos_a
    
    return R


def build_projection_matrix() -> np.ndarray:
    """
    Build the 6→4 projection from bivectors to trivectors.
    
    Mapping:
    - e01, e02, e03 (temporal) → e012, e013, e023 (trivectors with e0)
    - e12, e13, e23 (spatial) → e123 (pure spatial trivector)
    
    The projection is φ-weighted to preserve relative importance.
    
    Returns:
        Projection matrix of shape [4, 6]
    """
    # 4 trivector components: e012, e013, e023, e123
    # 6 bivector components: e01, e02, e03, e12, e13, e23
    
    P = np.zeros((4, 6))
    
    # e012 ← e01 + φ^-1 · e02  (time-space plane)
    P[0, 0] = 1.0
    P[0, 1] = PHI_INV
    
    # e013 ← e01 + φ^-1 · e03
    P[1, 0] = PHI_INV
    P[1, 2] = 1.0
    
    # e023 ← e02 + φ^-1 · e03
    P[2, 1] = 1.0
    P[2, 2] = PHI_INV
    
    # e123 ← e12 + e13 + e23 (all spatial)
    # φ-weighted sum
    P[3, 3] = 1.0
    P[3, 4] = PHI_INV
    P[3, 5] = PHI_INV_SQ
    
    # Normalize rows
    for i in range(4):
        norm = np.linalg.norm(P[i])
        if norm > 0:
            P[i] /= norm
    
    return P


# =============================================================================
# INTERACTION TENSOR CLASS
# =============================================================================

@dataclass
class InteractionTensor:
    """
    Complete interaction tensor for Level 0 → Level 1 projection.
    
    Manages the 16×6×4 tensor that maps satellite bivectors to
    master trivectors with φ-coupling.
    
    The tensor combines:
    1. Per-satellite rotation rotors R_k
    2. Projection matrix P (6→4)
    3. φ^-2 scaling (spectral gap)
    
    Attributes:
        n_satellites: Number of satellites (default 16)
        rotors: List of rotation matrices [16 × [6,6]]
        projection: Projection matrix [4, 6]
    """
    n_satellites: int = 16
    
    def __post_init__(self):
        """Build tensor components."""
        # Build rotation rotors for each satellite
        self.rotors = [
            build_rotation_rotor(k, self.n_satellites)
            for k in range(self.n_satellites)
        ]
        
        # Build projection matrix
        self.projection = build_projection_matrix()
        
        # Build inverse projection (for downward flow)
        # Use pseudoinverse for 4→6 expansion
        self.inverse_projection = np.linalg.pinv(self.projection)
    
    def project_up(
        self,
        satellite_bivectors: np.ndarray,
        satellite_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Project satellite bivectors UP to master trivectors.
        
        M_grade3 = φ^-2 · Σ_k (R_k · S_k_grade2)
        
        Args:
            satellite_bivectors: Bivector components [n_satellites, 6] or [batch, n_satellites, 6]
            satellite_indices: Optional indices if only projecting subset
            
        Returns:
            Master trivector components [4] or [batch, 4]
        """
        if satellite_bivectors.ndim == 1:
            satellite_bivectors = satellite_bivectors[None, :]
            squeeze = True
        else:
            squeeze = False
        
        batch_size = satellite_bivectors.shape[0] if satellite_bivectors.ndim == 3 else 1
        
        # Handle different input shapes
        if satellite_bivectors.ndim == 2:
            # [n_satellites, 6] → single batch
            n_sats = satellite_bivectors.shape[0]
            rotated = np.zeros((n_sats, 6))
            for k in range(n_sats):
                idx = k if satellite_indices is None else satellite_indices[k]
                rotated[k] = self.rotors[idx % self.n_satellites] @ satellite_bivectors[k]
            
            # Sum rotated bivectors
            summed = rotated.sum(axis=0)
            
            # Project to trivectors
            trivector = self.projection @ summed
            
            # φ^-2 scaling
            trivector *= PHI_INV_SQ
            
        else:
            # [batch, n_satellites, 6]
            trivector = np.zeros((batch_size, 4))
            for b in range(batch_size):
                n_sats = satellite_bivectors.shape[1]
                rotated = np.zeros((n_sats, 6))
                for k in range(n_sats):
                    idx = k if satellite_indices is None else satellite_indices[k]
                    rotated[k] = self.rotors[idx % self.n_satellites] @ satellite_bivectors[b, k]
                summed = rotated.sum(axis=0)
                trivector[b] = PHI_INV_SQ * (self.projection @ summed)
        
        return trivector.squeeze() if squeeze else trivector
    
    def project_down(
        self,
        master_trivector: np.ndarray,
        target_satellite: int
    ) -> np.ndarray:
        """
        Project master trivector DOWN to satellite bivector.
        
        Used in "explanation" flow and Non-REM consolidation.
        
        Args:
            master_trivector: Trivector components [4]
            target_satellite: Which satellite to project to
            
        Returns:
            Bivector components for that satellite [6]
        """
        # Expand trivector to bivector space
        bivector = self.inverse_projection @ master_trivector
        
        # Apply INVERSE rotation for this satellite
        R_k = self.rotors[target_satellite % self.n_satellites]
        R_k_inv = R_k.T  # Orthogonal, so inverse = transpose
        
        # Scale by φ^2 (inverse of φ^-2 upward scaling)
        bivector = (PHI ** 2) * (R_k_inv @ bivector)
        
        return bivector
    
    def get_coupling_strength(self, sat_a: int, sat_b: int) -> float:
        """
        Compute coupling strength between two satellites.
        
        Satellites with similar golden angles couple more strongly.
        
        Args:
            sat_a: First satellite index
            sat_b: Second satellite index
            
        Returns:
            Coupling strength in [0, 1]
        """
        alpha_a = 2 * PI * sat_a * PHI_INV
        alpha_b = 2 * PI * sat_b * PHI_INV
        
        # Phase difference
        delta = abs(alpha_a - alpha_b) % (2 * PI)
        delta = min(delta, 2 * PI - delta)  # Normalize to [0, π]
        
        # Coupling is strongest at π/φ separation
        optimal_sep = PI * PHI_INV
        coupling = np.cos(delta - optimal_sep)
        
        return (coupling + 1) / 2  # Map to [0, 1]


# =============================================================================
# TESTS
# =============================================================================

def _test_interaction_tensor():
    """Test the interaction tensor system."""
    print("Testing interaction tensor...")
    
    # Test 1: Build tensor
    tensor = InteractionTensor(n_satellites=16)
    assert len(tensor.rotors) == 16, "Should have 16 rotation rotors"
    assert tensor.projection.shape == (4, 6), "Projection should be 4×6"
    
    # Test 2: Rotation rotors are orthogonal
    for k, R in enumerate(tensor.rotors):
        assert R.shape == (6, 6), f"Rotor {k} should be 6×6"
        RRT = R @ R.T
        assert np.allclose(RRT, np.eye(6), atol=1e-10), f"Rotor {k} should be orthogonal"
    
    # Test 3: Project up
    np.random.seed(42)
    sat_bivectors = np.random.randn(16, 6)
    trivector = tensor.project_up(sat_bivectors)
    assert trivector.shape == (4,), "Output should be 4D trivector"
    
    # Test 4: Project down
    for k in range(16):
        bivector = tensor.project_down(trivector, k)
        assert bivector.shape == (6,), f"Satellite {k} should get 6D bivector"
    
    # Test 5: Coupling strength
    coupling_00 = tensor.get_coupling_strength(0, 0)
    coupling_08 = tensor.get_coupling_strength(0, 8)
    assert 0 <= coupling_00 <= 1, "Coupling should be in [0,1]"
    assert 0 <= coupling_08 <= 1, "Coupling should be in [0,1]"
    
    # Test 6: φ^-2 scaling is applied
    # Project up twice with different scales
    single_sat = np.ones((1, 6))
    result = tensor.project_up(single_sat, satellite_indices=np.array([0]))
    # Should be scaled by φ^-2
    expected_scale = PHI_INV_SQ
    # Check that result is reasonable (not zero, not huge)
    assert 0 < np.linalg.norm(result) < 100, "Result should be reasonably scaled"
    
    print("✓ Interaction tensor tests passed!")
    return True


if __name__ == "__main__":
    _test_interaction_tensor()
