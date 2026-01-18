"""
Fractal Module — Nested Torus Architecture
==========================================

Implements the 16^n fractal scaling system where each level
is a complete Cl(3,1) torus with satellites.

Key Components:
- nested_torus: MultiLevelTorus class integrating all components
- grand_equilibrium: Energy conservation across levels
- downward_projection: Generation flow from abstract to concrete

Hierarchy:
    Level 0: 16 components (word/token)
    Level 1: 16 Level-0 satellites = 256 components (phrase)
    Level 2: 16 Level-1 masters = 4096 components (concept)
    Level N: 16^N components
    
Tower depth 4 = 16^4 = 65,536 base units → ~1T effective capacity.

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

from .nested_torus import (
    NestedFractalTorus,
    TorusLevel,
    SatelliteState,
)
from .grand_equilibrium import (
    compute_grand_equilibrium,
    verify_energy_conservation,
    GrandEquilibrium,
)
from .downward_projection import (
    project_to_tokens,
    DownwardProjection,
)

__all__ = [
    # Nested Torus
    'NestedFractalTorus',
    'TorusLevel',
    'SatelliteState',
    # Grand Equilibrium
    'compute_grand_equilibrium',
    'verify_energy_conservation',
    'GrandEquilibrium',
    # Downward Projection
    'project_to_tokens',
    'DownwardProjection',
]
