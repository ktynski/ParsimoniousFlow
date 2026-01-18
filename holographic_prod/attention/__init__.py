"""
Structural attention via toroidal phase alignment.

NO LEARNED WEIGHTS. ALL φ-DERIVED.

- ToroidalAttention: Phase-coherent aggregation across 16 satellites
- SatellitePhase: Individual satellite state with golden spiral offset
"""

from .toroidal_attention import (
    ToroidalAttention,
    SatellitePhase,
)

__all__ = [
    'ToroidalAttention',
    'SatellitePhase',
]
