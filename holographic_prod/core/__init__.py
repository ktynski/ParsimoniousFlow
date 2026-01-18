"""
Core mathematical foundations for holographic computation.

- algebra: Clifford algebra operations (Cl(3,1))
- constants: φ-derived constants (no arbitrary values)
- binding: Attribute-object binding via geometric products
- quotient: Witness extraction and Grace dynamics
- quaternion: SO(4) ≅ (SU(2) × SU(2)) / Z₂ representation (2× memory reduction)
"""

from .constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR,
    GOLDEN_ANGLE, MATRIX_DIM, CLIFFORD_DIM, DTYPE,
    GRADE_INDICES, GRACE_SCALES, GRACE_SCALES_FLAT,
)

from .algebra import (
    build_gamma_matrices,
    build_clifford_basis,
    geometric_product,
    geometric_product_batch,
    wedge_product,
    inner_product,
    grace_operator,
    grace_operator_batch,
    competitive_grace_operator,
    clifford_inverse,
    frobenius_similarity,
    decompose_to_coefficients,
    coefficients_to_matrix,
    normalize_matrix,
    compute_vorticity,
    vorticity_magnitude,
    vorticity_signature,
    vorticity_similarity,
)

from .quotient import (
    extract_witness,
    witness_matrix,
    grace_stability,
    grace_stability_batch,
    compute_enstrophy,
    grade_energies,
    # Chirality (v5.27.0)
    extract_chirality,
    extract_chirality_batch,
    extract_chirality_strength,
    chirality_match_scores,
)

from .binding import (
    bind_attribute_to_object,
    unbind_attribute,
    binding_signature,
    binding_similarity,
)

from .quaternion import (
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_normalize,
    quaternion_pair_to_so4,
    so4_to_quaternion_pair,
    quaternion_geometric_product,
    create_quaternion_embeddings,
    batch_quaternion_to_so4,
    batch_so4_to_quaternion,
    left_quaternion_matrix,
    right_quaternion_matrix,
)

from .lensing import (
    PolarizedLens,
    PolarizedLensSet,
    create_lens_for_satellite,
    polarized_similarity,
)

from .fractal_position import (
    golden_angle,
    fractal_position_rotation,
    encode_position_fractal,
    encode_sequence_fractal,
    encode_sequence_fractal_vectorized,
    hierarchical_position_key,
    position_distance,
    position_correlation,
    analyze_position_coverage,
)

__all__ = [
    # Constants
    'PI', 'PHI', 'PHI_INV', 'PHI_INV_SQ', 'PHI_INV_CUBE', 'PHI_INV_FOUR',
    'GOLDEN_ANGLE', 'MATRIX_DIM', 'CLIFFORD_DIM', 'DTYPE',
    'GRADE_INDICES', 'GRACE_SCALES', 'GRACE_SCALES_FLAT',
    # Algebra
    'build_gamma_matrices', 'build_clifford_basis',
    'geometric_product', 'geometric_product_batch',
    'wedge_product', 'inner_product',
    'grace_operator', 'grace_operator_batch', 'competitive_grace_operator',
    'clifford_inverse', 'frobenius_similarity',
    'decompose_to_coefficients', 'coefficients_to_matrix',
    'normalize_matrix', 'compute_vorticity',
    'vorticity_magnitude', 'vorticity_signature', 'vorticity_similarity',
    # Quotient
    'extract_witness', 'witness_matrix',
    'grace_stability', 'grace_stability_batch',
    'compute_enstrophy', 'grade_energies',
    # Chirality (v5.27.0)
    'extract_chirality', 'extract_chirality_batch',
    'extract_chirality_strength', 'chirality_match_scores',
    # Binding
    'bind_attribute_to_object', 'unbind_attribute',
    'binding_signature', 'binding_similarity',
    # Quaternion (2× memory reduction)
    'quaternion_multiply', 'quaternion_conjugate', 'quaternion_normalize',
    'quaternion_pair_to_so4', 'so4_to_quaternion_pair',
    'quaternion_geometric_product', 'create_quaternion_embeddings',
    'batch_quaternion_to_so4', 'batch_so4_to_quaternion',
    'left_quaternion_matrix', 'right_quaternion_matrix',
    # Polarized Lensing (capacity expansion via holographic parallax)
    'PolarizedLens', 'PolarizedLensSet',
    'create_lens_for_satellite', 'polarized_similarity',
    # Fractal Position Encoding (φ-derived multi-scale position)
    'golden_angle', 'fractal_position_rotation',
    'encode_position_fractal', 'encode_sequence_fractal',
    'encode_sequence_fractal_vectorized', 'hierarchical_position_key',
    'position_distance', 'position_correlation', 'analyze_position_coverage',
]
