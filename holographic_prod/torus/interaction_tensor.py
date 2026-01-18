"""
Interaction Tensor — 16×6×4 Projection Between Fractal Levels

Maps satellite bivectors (Grade 2, 6 components) to master trivectors
(Grade 3, 4 components) using φ-coupling.

Theory (Chapter 11):
    M_grade3 = φ⁻² · Σ_{k=0}^{15} (R_k · S_k_grade2)
    
    Where:
    - S_k_grade2: 6 bivector components from satellite k
    - R_k: Rotation rotor for satellite k (φ-offset phase)
    - M_grade3: 4 trivector components in master
    
    Mapping:
    - Temporal bivectors (e01, e02, e03) → trivectors with e0
    - Spatial bivectors (e12, e13, e23) → pure spatial trivector e123

NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import Optional, Tuple, Any
from dataclasses import dataclass

from holographic_prod.core.constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ,
    CLIFFORD_DIM, DTYPE,
    GRADE_INDICES,
)


def build_rotation_rotor(k: int, n_satellites: int = 16) -> np.ndarray:
    """
    Build rotation rotor for satellite k based on φ-offset phase.
    
    Theory:
        Each satellite has a phase offset: θ_k = 2π × k × φ⁻¹
        This creates a rotation in bivector space.
    
    Args:
        k: Satellite index [0, n_satellites)
        n_satellites: Total number of satellites
        
    Returns:
        [6, 6] rotation matrix for bivectors
    """
    # Phase offset: golden angle spacing
    phase = 2 * PI * k * PHI_INV / n_satellites
    
    # Build rotation matrix for 6D bivector space
    # This is a simplified rotation - rotates bivectors by phase
    rotor = np.eye(6, dtype=DTYPE)
    
    # Apply phase rotation to bivector components
    # Temporal bivectors (indices 0, 1, 2) rotate together
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
    
    # Rotate temporal bivectors
    rotor[0, 0] = cos_phase
    rotor[0, 1] = -sin_phase * PHI_INV
    rotor[1, 0] = sin_phase * PHI_INV
    rotor[1, 1] = cos_phase
    
    # Spatial bivectors (indices 3, 4, 5) rotate with different phase
    spatial_phase = phase * PHI_INV  # φ-scaled phase
    cos_spatial = np.cos(spatial_phase)
    sin_spatial = np.sin(spatial_phase)
    
    rotor[3, 3] = cos_spatial
    rotor[3, 4] = -sin_spatial * PHI_INV_SQ
    rotor[4, 3] = sin_spatial * PHI_INV_SQ
    rotor[4, 4] = cos_spatial
    
    return rotor


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
    
    P = np.zeros((4, 6), dtype=DTYPE)
    
    # e012 ← e01 + φ⁻¹ · e02  (time-space plane)
    P[0, 0] = 1.0
    P[0, 1] = PHI_INV
    
    # e013 ← e01 + φ⁻¹ · e03
    P[1, 0] = PHI_INV
    P[1, 2] = 1.0
    
    # e023 ← e02 + φ⁻¹ · e03
    P[2, 1] = 1.0
    P[2, 2] = PHI_INV
    
    # e123 ← e12 + φ⁻¹·e13 + φ⁻²·e23 (all spatial, φ-weighted)
    P[3, 3] = 1.0
    P[3, 4] = PHI_INV
    P[3, 5] = PHI_INV_SQ
    
    # VECTORIZED: Normalize rows to preserve energy
    norms = np.linalg.norm(P, axis=1, keepdims=True)  # [4, 1]
    P = P / np.maximum(norms, 1e-8)
    
    return P


@dataclass
class InteractionTensor:
    """
    Interaction tensor for Level 0 → Level 1 projection.
    
    Maps 16 satellites × 6 bivectors → 4 trivectors with φ-coupling.
    """
    
    n_satellites: int = 16
    xp: Any = None  # Will be set in __post_init__
    
    def __post_init__(self):
        """Build tensor components."""
        # Set default xp if not provided
        if self.xp is None:
            self.xp = np
        
        # Build rotation rotors for each satellite
        rotors_list = [
            build_rotation_rotor(k, self.n_satellites)
            for k in range(self.n_satellites)
        ]
        # Convert to array (GPU-aware)
        self.rotors = self.xp.array(rotors_list)  # [16, 6, 6]
        
        # Build projection matrix
        self.projection = build_projection_matrix()
        # Convert to GPU if needed
        if self.xp != np:
            self.projection = self.xp.asarray(self.projection)
        
        # Build inverse projection (for downward flow)
        self.inverse_projection = self.xp.linalg.pinv(self.projection)
    
    def project_up(
        self,
        satellite_bivectors: np.ndarray,
        satellite_indices: Optional[np.ndarray] = None,
        squeeze: bool = True,
    ) -> np.ndarray:
        """
        Project satellite bivectors up to master trivector.
        
        Theory:
            M_grade3 = φ⁻² · Σ_{k=0}^{15} (R_k · S_k_grade2)
        
        Args:
            satellite_bivectors: [n_satellites, 6] or [batch, n_satellites, 6]
            satellite_indices: Optional indices (default: [0..n_satellites-1])
            squeeze: If True, remove single-dimension batches
            
        Returns:
            [4] or [batch, 4] master trivector
        """
        if satellite_bivectors.ndim == 2:
            # Single batch: [n_satellites, 6]
            n_sats = satellite_bivectors.shape[0]
            rotated = np.zeros((n_sats, 6), dtype=DTYPE)
            
            for k in range(n_sats):
                idx = k if satellite_indices is None else satellite_indices[k]
                rotor = self.rotors[idx % self.n_satellites]
                rotated[k] = rotor @ satellite_bivectors[k]
            
            # Sum rotated bivectors
            summed = rotated.sum(axis=0)
            
            # Project to trivectors
            trivector = self.projection @ summed
            
            # φ⁻² scaling
            trivector *= PHI_INV_SQ
            
            return trivector
        
        else:
            # Batch: [batch_size, n_satellites, 6] - FULLY VECTORIZED
            batch_size = satellite_bivectors.shape[0]
            n_sats = satellite_bivectors.shape[1]
            
            # Ensure satellite_bivectors is on correct device
            if self.xp != np and not hasattr(satellite_bivectors, 'get'):
                satellite_bivectors = self.xp.asarray(satellite_bivectors)
            elif self.xp == np and hasattr(satellite_bivectors, 'get'):
                satellite_bivectors = satellite_bivectors.get()
            
            # VECTORIZED: Process all batches at once
            if satellite_indices is None:
                rotor_indices = self.xp.arange(n_sats, dtype=self.xp.int32) % self.n_satellites
                rotor_indices = self.xp.broadcast_to(rotor_indices[None, :], (batch_size, n_sats))
            else:
                if satellite_indices.ndim == 1:
                    # Broadcast to all batches
                    rotor_indices = self.xp.broadcast_to(satellite_indices[None, :], (batch_size, n_sats))
                else:
                    rotor_indices = satellite_indices % self.n_satellites
            
            # Select rotors: [batch_size, n_sats, 6, 6]
            # Convert rotor_indices to numpy for indexing if needed
            if self.xp != np:
                rotor_indices_np = rotor_indices.get() if hasattr(rotor_indices, 'get') else rotor_indices
            else:
                rotor_indices_np = rotor_indices
            
            # Advanced indexing requires numpy arrays
            selected_rotors = self.rotors[rotor_indices_np]  # [batch_size, n_sats, 6, 6]
            # Convert back to xp array
            selected_rotors = self.xp.asarray(selected_rotors)
            
            # Batched matmul: [batch_size, n_sats, 6, 6] @ [batch_size, n_sats, 6] → [batch_size, n_sats, 6]
            rotated = self.xp.einsum('bnij,bnj->bni', selected_rotors, satellite_bivectors)
            
            # Sum over satellites: [batch_size, 6]
            summed = rotated.sum(axis=1)
            
            # Project to trivectors: [batch_size, 4]
            trivectors = self.xp.einsum('ij,bj->bi', self.projection, summed)
            
            # φ⁻² scaling
            trivectors *= PHI_INV_SQ
            
            if squeeze and batch_size == 1:
                return trivectors[0]
            return trivectors
    
    def project_down(
        self,
        master_trivector: np.ndarray,
        target_satellite: int,
    ) -> np.ndarray:
        """
        Project master trivector down to specific satellite bivector.
        
        Args:
            master_trivector: [4] master trivector
            target_satellite: Satellite index [0, n_satellites)
            
        Returns:
            [6] satellite bivector
        """
        # Expand trivector to bivector space
        bivector = self.inverse_projection @ master_trivector
        
        # Apply inverse rotation for this satellite
        rotor_inv = np.linalg.inv(self.rotors[target_satellite % self.n_satellites])
        satellite_bivector = rotor_inv @ bivector
        
        return satellite_bivector
