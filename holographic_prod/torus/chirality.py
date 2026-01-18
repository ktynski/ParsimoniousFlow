"""
Chirality Alternation — Satellite Handedness

Prevents interference between satellites by alternating handedness.

Theory (Chapter 11):
    Even satellites (k mod 2 == 0): Right-handed (standard orientation)
    Odd satellites (k mod 2 == 1): Left-handed (mirrored orientation)
    
    This creates topological friction that prevents destructive interference
    when multiple satellites encode similar information.

NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import Union
from dataclasses import dataclass

from holographic_prod.core.constants import (
    CLIFFORD_DIM, DTYPE,
    GRADE_INDICES,
)


@dataclass
class ChiralityFlip:
    """
    Chirality alternation for fractal torus satellites.
    
    Even satellites: Right-handed (preserve orientation)
    Odd satellites: Left-handed (flip bivectors/trivectors)
    """
    
    n_satellites: int = 16
    
    def is_right_handed(self, satellite_index: int) -> bool:
        """
        Check if satellite is right-handed.
        
        Args:
            satellite_index: Satellite index [0, n_satellites)
            
        Returns:
            True if right-handed (even), False if left-handed (odd)
        """
        return (satellite_index % 2) == 0
    
    def apply(
        self,
        multivector: np.ndarray,
        satellite_index: int,
    ) -> np.ndarray:
        """
        Apply chirality flip to multivector.
        
        For left-handed satellites (odd), flip bivector and trivector components.
        For right-handed satellites (even), preserve.
        
        Args:
            multivector: [16] Clifford coefficients or [4, 4] matrix
            satellite_index: Satellite index [0, n_satellites)
            
        Returns:
            [16] or [4, 4] flipped multivector
        """
        if multivector.ndim == 2:
            # Matrix form: convert to coefficients, flip, convert back
            from holographic_prod.core.algebra import (
                build_clifford_basis,
                decompose_to_coefficients,
                reconstruct_from_coefficients,
            )
            
            basis = build_clifford_basis(np)
            coeffs = decompose_to_coefficients(multivector, basis, np)
            coeffs_flipped = self._flip_coefficients(coeffs, satellite_index)
            return reconstruct_from_coefficients(coeffs_flipped, basis, np)
        
        else:
            # Coefficient form: flip directly
            return self._flip_coefficients(multivector.copy(), satellite_index)
    
    def _flip_coefficients(
        self,
        coeffs: np.ndarray,
        satellite_index: int,
    ) -> np.ndarray:
        """
        Flip bivector and trivector coefficients for left-handed satellites.
        
        Args:
            coeffs: [16] Clifford coefficients
            satellite_index: Satellite index
            
        Returns:
            [16] flipped coefficients
        """
        if self.is_right_handed(satellite_index):
            # Right-handed: preserve
            return coeffs
        
        # Left-handed: flip bivectors and trivectors
        flipped = coeffs.copy()
        
        # Flip bivectors (grade 2: indices 5-10)
        bivector_indices = GRADE_INDICES[2]
        flipped[bivector_indices] *= -1
        
        # Flip trivectors (grade 3: indices 11-14)
        trivector_indices = GRADE_INDICES[3]
        flipped[trivector_indices] *= -1
        
        return flipped
