"""
Torus Components — Fractal Torus Architecture

Implements the nested fractal torus for hierarchical memory.
"""

from .interaction_tensor import InteractionTensor, build_projection_matrix, build_rotation_rotor
from .chirality import ChiralityFlip

__all__ = [
    'InteractionTensor',
    'build_projection_matrix',
    'build_rotation_rotor',
    'ChiralityFlip',
]
