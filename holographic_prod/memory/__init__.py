"""
Memory systems for holographic language model.

UNIFIED ARCHITECTURE — Scalable hierarchical memory.

Use HolographicMemory and MemoryConfig directly.

For multi-level scaling (16^N capacity):
    - max_levels=1: 16 satellites (default, ~16K patterns)
    - max_levels=2: 256 satellites (~256K patterns)
    - max_levels=3: 4,096 satellites (~4M patterns, GPU-optimal)

Witness Entanglement (v5.27.0):
    WitnessIndex enables non-local updates across semantically equivalent
    memory locations. This is a quantum-inspired feature.
"""

from .holographic_memory_unified import (
    MemoryConfig,
    SatelliteMemory,
    TowerMemory,
    HolographicMemory,
    MultiTimescaleMemory,
)
from .multi_level_tower import MultiLevelTower
from .witness_index import (
    WitnessIndex,
    propagate_witness_update,
    batch_register_witnesses,
)

__all__ = [
    'MemoryConfig',
    'SatelliteMemory',
    'TowerMemory',
    'MultiLevelTower',
    'HolographicMemory',
    'MultiTimescaleMemory',
    # v5.27.0: Witness Entanglement
    'WitnessIndex',
    'propagate_witness_update',
    'batch_register_witnesses',
]
