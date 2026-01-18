"""
Torus Module — Nested Fractal Torus Architecture
=================================================

Implements the toroidal state space for the 16^n scaling architecture.
Each level is a complete Cl(3,1) system nested on a torus with φ-offset
phase distribution to prevent resonance disaster.

Key Components:
- phase_distribution: φ-offset satellite placement (golden spiral)
- toroidal_coords: T² mapping (θ=vorticity, ψ=witness)
- interaction_tensor: 16×6×4 projection between levels
- chirality: Even/odd handedness flip for topological friction
- grace_inverse: Inflation operator (reverse of Grace)

Theory Reference:
    α_k = 2πkφ^-1  — Satellite position on master torus
    ω_k = ω_base · φ^(k mod 4)  — Frequency staggering
    
    The golden ratio φ prevents resonance because it is the "most irrational"
    number - signals never sync up, avoiding destructive interference.
"""

from .phase_distribution import (
    compute_satellite_positions,
    compute_frequency_stagger,
    PhaseDistribution,
    compute_interference_coefficient,
)
from .toroidal_coords import (
    multivector_to_torus,
    torus_to_multivector,
    ToroidalCoordinates,
)
from .interaction_tensor import (
    InteractionTensor,
    build_rotation_rotor,
)
from .chirality import (
    ChiralityManager,
    get_chirality,
    apply_chirality_flip,
)
from .grace_inverse import (
    grace_inverse,
    GraceInverse,
)

__all__ = [
    # Phase Distribution
    'compute_satellite_positions',
    'compute_frequency_stagger',
    'PhaseDistribution',
    'compute_interference_coefficient',
    # Toroidal Coordinates
    'multivector_to_torus',
    'torus_to_multivector',
    'ToroidalCoordinates',
    # Interaction Tensor
    'InteractionTensor',
    'build_rotation_rotor',
    # Chirality
    'ChiralityManager',
    'get_chirality',
    'apply_chirality_flip',
    # Grace Inverse
    'grace_inverse',
    'GraceInverse',
]
