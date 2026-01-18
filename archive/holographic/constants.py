"""
Sacred Constants — Theory-derived, NOT tuned
============================================

These values are derived from the self-consistency equation Λ² = Λ + 1.
Changing them breaks the mathematical guarantees.

DO NOT MODIFY THESE VALUES.

SIGNATURE NOTE:
    We use Cl(3,1) with metric η = diag(+1,+1,+1,-1)
    - Three spacelike dimensions (e₁²=e₂²=e₃²=+1)
    - One timelike dimension (e₄²=-1)
    
    This gives the isomorphism Cl(3,1) ≅ M₄(ℝ)
    - Multivectors ↔ 4×4 real matrices
    - Geometric product ↔ Matrix multiplication
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

# =============================================================================
# SCCMU-DERIVED CONSTANTS
# =============================================================================

# Inverse temperature — from coherence periodicity
BETA = 2 * PI * PHI  # ≈ 10.166

# Golden angle — for uniform distribution on circles/tori
GOLDEN_ANGLE = 2 * PI / (PHI * PHI)  # ≈ 2.399 rad ≈ 137.5°

# =============================================================================
# TORUS GEOMETRY
# =============================================================================

MAJOR_RADIUS = PHI  # R (distance from center to tube center)
MINOR_RADIUS = 1.0  # r (tube radius)

# =============================================================================
# CLIFFORD ALGEBRA Cl(3,1)
# =============================================================================

# Signature: (+,+,+,-) — three spacelike, one timelike
# Isomorphism: Cl(3,1) ≅ M₄(ℝ) (4×4 real matrices)

CLIFFORD_DIM = 16   # 2^4 = 16 basis elements
MATRIX_DIM = 4      # 4×4 real matrices

# Grade dimensions (binomial coefficients)
GRADE_DIMS = {
    0: 1,   # scalar
    1: 4,   # vectors
    2: 6,   # bivectors
    3: 4,   # trivectors
    4: 1,   # pseudoscalar
}

# Grade indices in 16D representation (for reference)
GRADE_0_IDX = [0]                    # 1 scalar
GRADE_1_IDX = [1, 2, 3, 4]           # 4 vectors
GRADE_2_IDX = [5, 6, 7, 8, 9, 10]    # 6 bivectors
GRADE_3_IDX = [11, 12, 13, 14]       # 4 trivectors
GRADE_4_IDX = [15]                   # 1 pseudoscalar

# Grade slices for 16D vector operations
GRADE_SLICES = {
    0: slice(0, 1),      # 1 scalar
    1: slice(1, 5),      # 4 vectors
    2: slice(5, 11),     # 6 bivectors
    3: slice(11, 15),    # 4 trivectors
    4: slice(15, 16),    # 1 pseudoscalar
}

# =============================================================================
# GRACE OPERATOR SCALING
# =============================================================================
# Per-grade scaling factors for Grace contraction

GRACE_SCALE = {
    0: 1.0,           # Scalar: preserved
    1: PHI_INV,       # Vectors: φ⁻¹
    2: PHI_INV_SQ,    # Bivectors: φ⁻²
    3: PHI_INV_CUBE,  # Trivectors: φ⁻³
    4: PHI_INV,       # Pseudoscalar: φ⁻¹ (FIBONACCI EXCEPTION)
}

# Note: Grade 4 scales by φ⁻¹ NOT φ⁻⁴ due to Fibonacci anyon structure:
# The pseudoscalar represents anyon τ with quantum dimension d_τ = φ
# Scaling is 1/d_τ = φ⁻¹

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
    
    # Grade indices cover all 16 components
    all_indices = set(GRADE_0_IDX + GRADE_1_IDX + GRADE_2_IDX + GRADE_3_IDX + GRADE_4_IDX)
    if all_indices != set(range(16)):
        errors.append("Grade indices don't cover all 16 components")
    
    if errors:
        raise ValueError("Constant verification failed:\n  " + "\n  ".join(errors))
    
    return True


# Run verification at import time
verify_constants()
