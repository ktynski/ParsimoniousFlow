"""
Fractal Torus Components — Hierarchical Memory Architecture

Implements the nested fractal torus architecture for scalable capacity (16^N).
"""

from .downward_projection import phase_locked_emission, DownwardProjection
from .grand_equilibrium import compute_grand_equilibrium, check_equilibrium

__all__ = [
    'phase_locked_emission',
    'DownwardProjection',
    'compute_grand_equilibrium',
    'check_equilibrium',
]
