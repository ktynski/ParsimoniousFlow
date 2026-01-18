"""
Sacred Constants — Theory-derived, NOT tuned
============================================

These values are derived from the self-consistency equation Λ² = Λ + 1.
Changing them breaks the mathematical guarantees.

DO NOT MODIFY THESE VALUES.

SIGNATURE:
    Cl(3,1) with metric η = diag(+1,+1,+1,-1)
    - Three spacelike dimensions (e₁²=e₂²=e₃²=+1)
    - One timelike dimension (e₄²=-1)
    
    Isomorphism: Cl(3,1) ≅ M₄(ℝ) (4×4 real matrices)

GRACE AS VISCOSITY:
    The Grace operator acts as viscosity for bivectors (vorticity):
    
    - Grade 2 (bivectors) contains rotational/vorticity content
    - Grace scales grade-2 by φ⁻² per application
    - Enstrophy (||grade-2||²) decays at φ⁻⁴ per step
    
    This is the Clifford analogue of viscous damping in Navier-Stokes.
    
    Stability condition: coupling α < φ⁻² ensures bounded equilibrium
    (no blow-up, analogous to subcritical Reynolds number in fluids).
"""

import math

# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================

PI = 3.141592653589793

# Golden ratio — unique positive solution to Λ² = Λ + 1
PHI = 1.618033988749894848204586834365638118
PHI_INV = 0.618033988749894848204586834365638118      # 1/φ = φ - 1
PHI_INV_SQ = 0.381966011250105151795413165634361882   # 1/φ² = spectral gap γ
PHI_INV_CUBE = 0.236067977499789696409173668731276235  # 1/φ³
PHI_INV_FOUR = 0.145898033750315455386239496903085647  # 1/φ⁴
PHI_INV_FIVE = 0.090169943749474240590957503097190354  # 1/φ⁵
PHI_INV_SIX = 0.055728090000841214795281993805895293   # 1/φ⁶ ≈ 0.056 (fine resolution)
PHI_INV_SEVEN = 0.034441853748632925795675509291295061 # 1/φ⁷
PHI_INV_EIGHT = 0.021286236252208289100606484514600232 # 1/φ⁸ ≈ 0.021 (very fine resolution)

# Golden angle — for uniform distribution on circles/tori
GOLDEN_ANGLE = 2 * PI / (PHI * PHI)  # ≈ 2.399 rad ≈ 137.5°

# =============================================================================
# PRECISION CONFIGURATION
# =============================================================================
# H100 TENSOR CORE ACCELERATION:
#   - float64: No tensor cores, maximum precision
#   - float32: TF32 tensor cores (automatic on H100), 2× faster
#   - float16: Full tensor cores, 4× faster (may need gradient scaling)
#
# For Clifford algebra with 512-token contexts:
#   - Binary reduction has 9 levels (log₂512)
#   - FP32 precision ~10⁻⁷ → accumulated error ~10⁻⁶ (acceptable)
#   - Grace damping factors all > 0.1 → well within FP32 range

import numpy as np
DTYPE = np.float32  # TF32 tensor cores on H100
DTYPE_STR = 'float32'

# =============================================================================
# CLIFFORD ALGEBRA Cl(3,1)
# =============================================================================

CLIFFORD_DIM = 16   # 2^4 = 16 basis elements
MATRIX_DIM = 4      # 4×4 real matrices

# Grade dimensions
GRADE_DIMS = {0: 1, 1: 4, 2: 6, 3: 4, 4: 1}

# Grade indices in 16D representation
GRADE_INDICES = {
    0: [0],                    # 1 scalar
    1: [1, 2, 3, 4],           # 4 vectors
    2: [5, 6, 7, 8, 9, 10],    # 6 bivectors
    3: [11, 12, 13, 14],       # 4 trivectors
    4: [15],                   # 1 pseudoscalar
}

# =============================================================================
# GRACE OPERATOR SCALING
# =============================================================================
#
# Grace acts like viscosity in Navier-Stokes:
#   - Grade 2 (bivectors) = vorticity container
#   - Scaling by φ⁻² = viscous damping
#   - Enstrophy = ||grade-2||² decays at rate φ⁻⁴ per step
#
# The spectral gap γ = φ⁻² ensures exponential convergence to equilibrium.

GRACE_SCALES = {
    0: 1.0,           # Scalar: preserved (total "energy")
    1: PHI_INV,       # Vectors: φ⁻¹ (direction)
    2: PHI_INV_SQ,    # Bivectors: φ⁻² (VORTICITY - key damping)
    3: PHI_INV_CUBE,  # Trivectors: φ⁻³ (fine structure)
    4: PHI_INV,       # Pseudoscalar: φ⁻¹ (FIBONACCI EXCEPTION)
}

# Note: Grade 4 scales by φ⁻¹ NOT φ⁻⁴ due to Fibonacci anyon structure:
# The pseudoscalar represents anyon τ with quantum dimension d_τ = φ
# Scaling is 1/d_τ = φ⁻¹
#
# Enstrophy decay rate: (φ⁻²)² = φ⁻⁴ ≈ 0.1459 per Grace application
# This matches exactly what we observe empirically.

# Flat array for vectorized Grace application
GRACE_SCALES_FLAT = [
    1.0,                                    # Grade 0
    PHI_INV, PHI_INV, PHI_INV, PHI_INV,    # Grade 1
    PHI_INV_SQ, PHI_INV_SQ, PHI_INV_SQ,    # Grade 2
    PHI_INV_SQ, PHI_INV_SQ, PHI_INV_SQ,
    PHI_INV_CUBE, PHI_INV_CUBE, PHI_INV_CUBE, PHI_INV_CUBE,  # Grade 3
    PHI_INV,                                # Grade 4 (Fibonacci!)
]

# =============================================================================
# VERIFICATION
# =============================================================================

def verify_constants():
    """Verify all mathematical relationships hold."""
    errors = []
    
    # φ² = φ + 1
    if abs(PHI * PHI - (PHI + 1)) > 1e-15:
        errors.append("φ² ≠ φ + 1")
    
    # φ⁻¹ = φ - 1
    if abs(PHI_INV - (PHI - 1)) > 1e-15:
        errors.append("φ⁻¹ ≠ φ - 1")
    
    # φ × φ⁻¹ = 1
    if abs(PHI * PHI_INV - 1.0) > 1e-15:
        errors.append("φ × φ⁻¹ ≠ 1")
    
    # Spectral gap γ = 1 - φ⁻¹ = φ⁻²
    gamma = 1 - PHI_INV
    if abs(gamma - PHI_INV_SQ) > 1e-15:
        errors.append("γ = 1 - φ⁻¹ ≠ φ⁻²")
    
    # Grade dimensions sum to 16
    if sum(GRADE_DIMS.values()) != CLIFFORD_DIM:
        errors.append("Grade dimensions don't sum to 16")
    
    if errors:
        raise ValueError("Constant verification failed:\n  " + "\n  ".join(errors))
    
    return True


# Run verification at import time
verify_constants()
