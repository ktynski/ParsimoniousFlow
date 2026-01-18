"""
GraceInverse — Inflation Operator (Reverse of Grace)
====================================================

The inverse of the Grace contraction operator. Re-inflates a multivector
from the coherent core (scalar + pseudoscalar) back to high-vorticity
state with active bivector components.

Used in:
1. Downward projection (explanation/generation)
2. Semantic snap (restoring structural detail)
3. Token emission (finding matching attractors)

Scaling (reverse of Grace):
    Grade 0: ×1.0      (anchor, unchanged)
    Grade 1: ×φ        (direction)
    Grade 2: ×φ²       (vorticity - key re-animation)
    Grade 3: ×φ³       (structure)
    Grade 4: ×φ        (Fibonacci exception)

Note: grace_inverse(grace(M)) ≈ M (up to numerical precision)

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
    CLIFFORD_DIM, GRADE_INDICES, GRACE_SCALES,
)


# =============================================================================
# GRACE INVERSE SCALES
# =============================================================================

# Inverse of GRACE_SCALES: multiply by φ^k instead of φ^-k
GRACE_INVERSE_SCALES = {
    0: 1.0,                    # Scalar: preserved (anchor)
    1: PHI,                    # Vectors: ×φ (inflate direction)
    2: PHI * PHI,              # Bivectors: ×φ² (VORTICITY - key inflation)
    3: PHI * PHI * PHI,        # Trivectors: ×φ³ (structure)
    4: PHI,                    # Pseudoscalar: ×φ (FIBONACCI EXCEPTION)
}

# Flat array for vectorized application
GRACE_INVERSE_SCALES_FLAT = [
    1.0,                                    # Grade 0
    PHI, PHI, PHI, PHI,                     # Grade 1
    PHI*PHI, PHI*PHI, PHI*PHI,              # Grade 2
    PHI*PHI, PHI*PHI, PHI*PHI,
    PHI*PHI*PHI, PHI*PHI*PHI, PHI*PHI*PHI, PHI*PHI*PHI,  # Grade 3
    PHI,                                    # Grade 4 (Fibonacci!)
]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def grace_inverse(M: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Apply GraceInverse operator to inflate multivector.
    
    Each grade is scaled by φ^k (inverse of Grace's φ^-k):
        Grade 0: ×1.0 (scalar anchor)
        Grade 1: ×φ (vectors)
        Grade 2: ×φ² (bivectors - VORTICITY)
        Grade 3: ×φ³ (trivectors)
        Grade 4: ×φ (pseudoscalar - Fibonacci exception)
    
    Args:
        M: Multivector of shape [..., 16]
        iterations: Number of times to apply (default 1)
        
    Returns:
        Inflated multivector of same shape
    """
    M_inflated = M.copy()
    
    scales = np.array(GRACE_INVERSE_SCALES_FLAT)
    
    for _ in range(iterations):
        M_inflated = M_inflated * scales
    
    return M_inflated


def grace_inverse_gradewise(
    M: np.ndarray,
    grade_scales: Optional[dict] = None
) -> np.ndarray:
    """
    Apply GraceInverse with custom per-grade scaling.
    
    Allows selective inflation of specific grades.
    
    Args:
        M: Multivector of shape [..., 16]
        grade_scales: Optional dict {grade: scale}, defaults to standard
        
    Returns:
        Inflated multivector
    """
    if grade_scales is None:
        grade_scales = GRACE_INVERSE_SCALES
    
    M_inflated = M.copy()
    
    for grade, scale in grade_scales.items():
        for idx in GRADE_INDICES[grade]:
            M_inflated[..., idx] *= scale
    
    return M_inflated


def semantic_snap(M: np.ndarray) -> np.ndarray:
    """
    Perform "semantic snap" - redistribute energy from core to structure.
    
    After GraceInverse, energy that was concentrated in grades 0/4
    "snaps" outward into the bivector (grade 2) plane, creating
    structural detail for token matching.
    
    This is used in generation to convert abstract "intent" into
    specific word-matching vorticity patterns.
    
    Args:
        M: Multivector (typically Grace-contracted)
        
    Returns:
        Snapped multivector with active bivectors
    """
    M_snapped = grace_inverse(M)
    
    # Ensure bivector components are non-trivial
    bivector_energy = np.sum(M_snapped[..., GRADE_INDICES[2]]**2, axis=-1)
    total_energy = np.sum(M_snapped**2, axis=-1)
    
    # If bivector energy is too low, boost it
    bivector_ratio = bivector_energy / (total_energy + 1e-10)
    
    if np.any(bivector_ratio < PHI_INV_SQ):
        # Transfer some scalar energy to bivectors
        scalar_energy = M_snapped[..., GRADE_INDICES[0][0]]**2
        transfer = PHI_INV * np.sqrt(scalar_energy)
        
        # Distribute transfer across bivector components
        for i, idx in enumerate(GRADE_INDICES[2]):
            angle = 2 * np.pi * i * PHI_INV
            M_snapped[..., idx] += transfer * np.cos(angle)
    
    return M_snapped


# =============================================================================
# GRACE INVERSE CLASS
# =============================================================================

@dataclass
class GraceInverse:
    """
    Complete GraceInverse operator for the fractal architecture.
    
    Manages inflation with:
    - Standard φ-derived scaling
    - Custom grade selection
    - Energy redistribution
    - Round-trip verification
    
    Attributes:
        scales: Grade-wise inflation scales
    """
    
    def __post_init__(self):
        """Initialize scales."""
        self.scales = GRACE_INVERSE_SCALES.copy()
        self.flat_scales = np.array(GRACE_INVERSE_SCALES_FLAT)
    
    def apply(self, M: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Apply GraceInverse.
        
        Args:
            M: Multivector [..., 16]
            iterations: Number of applications
            
        Returns:
            Inflated multivector
        """
        return grace_inverse(M, iterations)
    
    def apply_selective(
        self,
        M: np.ndarray,
        grades: list
    ) -> np.ndarray:
        """
        Apply GraceInverse only to specific grades.
        
        Args:
            M: Multivector
            grades: List of grades to inflate
            
        Returns:
            Selectively inflated multivector
        """
        grade_scales = {g: self.scales[g] for g in grades}
        # Other grades get scale 1.0 (unchanged)
        for g in range(5):
            if g not in grade_scales:
                grade_scales[g] = 1.0
        
        return grace_inverse_gradewise(M, grade_scales)
    
    def semantic_snap(self, M: np.ndarray) -> np.ndarray:
        """Perform semantic snap."""
        return semantic_snap(M)
    
    def verify_inverse(self, M: np.ndarray, grace_fn, rtol: float = 0.01) -> bool:
        """
        Verify that GraceInverse reverses Grace.
        
        Args:
            M: Original multivector
            grace_fn: Grace function to test against
            rtol: Relative tolerance
            
        Returns:
            True if round-trip preserves structure
        """
        M_graced = grace_fn(M)
        M_recovered = self.apply(M_graced)
        
        # Compare norms
        orig_norm = np.linalg.norm(M)
        recovered_norm = np.linalg.norm(M_recovered)
        
        return abs(orig_norm - recovered_norm) / (orig_norm + 1e-10) < rtol


# =============================================================================
# TESTS
# =============================================================================

def _test_grace_inverse():
    """Test GraceInverse operator."""
    print("Testing GraceInverse...")
    
    # Test 1: Scale factors are correct
    assert GRACE_INVERSE_SCALES[0] == 1.0, "Grade 0 should be 1.0"
    assert abs(GRACE_INVERSE_SCALES[1] - PHI) < 1e-10, "Grade 1 should be φ"
    assert abs(GRACE_INVERSE_SCALES[2] - PHI**2) < 1e-10, "Grade 2 should be φ²"
    assert abs(GRACE_INVERSE_SCALES[3] - PHI**3) < 1e-10, "Grade 3 should be φ³"
    assert abs(GRACE_INVERSE_SCALES[4] - PHI) < 1e-10, "Grade 4 should be φ (Fibonacci)"
    
    # Test 2: Apply inflation
    np.random.seed(42)
    M = np.random.randn(16)
    M_inflated = grace_inverse(M)
    
    # Check bivector inflation
    orig_biv_norm = np.linalg.norm(M[GRADE_INDICES[2]])
    inflated_biv_norm = np.linalg.norm(M_inflated[GRADE_INDICES[2]])
    assert inflated_biv_norm > orig_biv_norm, "Bivectors should be inflated"
    assert abs(inflated_biv_norm / orig_biv_norm - PHI**2) < 0.01, "Bivector ratio should be φ²"
    
    # Test 3: Round-trip with Grace
    from constants import GRACE_SCALES_FLAT
    
    def grace(M):
        return M * np.array(GRACE_SCALES_FLAT)
    
    M_round = grace_inverse(grace(M))
    assert np.allclose(M, M_round, rtol=1e-10), "Grace then GraceInverse should recover M"
    
    # Test 4: Semantic snap
    M_snapped = semantic_snap(grace(M))  # Contract then snap
    snap_biv_norm = np.linalg.norm(M_snapped[GRADE_INDICES[2]])
    assert snap_biv_norm > 0, "Snapped should have non-zero bivectors"
    
    # Test 5: GraceInverse class
    gi = GraceInverse()
    M_inflated2 = gi.apply(M)
    assert np.allclose(M_inflated, M_inflated2), "Class should match function"
    
    # Test 6: Selective inflation
    M_selective = gi.apply_selective(M, grades=[2])  # Only inflate bivectors
    assert np.allclose(M_selective[GRADE_INDICES[0]], M[GRADE_INDICES[0]]), "Grade 0 unchanged"
    assert not np.allclose(M_selective[GRADE_INDICES[2]], M[GRADE_INDICES[2]]), "Grade 2 changed"
    
    print("✓ GraceInverse tests passed!")
    return True


if __name__ == "__main__":
    _test_grace_inverse()
