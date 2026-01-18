"""
Witness Index — Quantum-Inspired Entanglement (v5.27.0)
=======================================================

Implements witness-based indexing for non-local memory updates.

QUANTUM THEORY:
    The witness (scalar + pseudoscalar) is gauge-invariant — it's the semantic
    "self-pointer" that survives infinite Grace iterations. All memory instances
    sharing the SAME witness are semantically identical (just in different
    structural contexts).
    
    Quantum analogy: Entangled particles share a quantum state. Measuring one
    instantly affects the other regardless of distance. Similarly, updating one
    witness should instantly update ALL memories with that witness.
    
BRAIN ANALOG:
    Semantic priming — activating one concept activates all related concepts.
    The witness index enables O(1) lookup of semantically equivalent memories.
    
INFORMATION PARSIMONY:
    Physical brains face decoherence — quantum states collapse before they can
    propagate. Our digital implementation maintains coherence indefinitely,
    enabling instant non-local learning that evolution hasn't achieved.

NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from holographic_prod.core.constants import (
    PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, DTYPE,
)
from holographic_prod.core.quotient import extract_witness, extract_witness_batch

Array = np.ndarray
ArrayModule = type(np)


@dataclass
class WitnessIndex:
    """
    Index mapping witness hashes to memory locations.
    
    QUANTUM THEORY (v5.27.0):
        Enables "witness entanglement" — when one memory location is updated,
        all locations sharing the same witness can be updated simultaneously.
        
        This is a quantum parsimony: if the brain were quantum and decoherence-free,
        it would exploit this for instant non-local learning.
        
    STRUCTURE:
        Key: Quantized (sigma, pseudo) tuple (bucket key)
        Value: List of (level_idx, satellite_idx) memory locations
        
    RESOLUTION:
        Default resolution=2 means witnesses are bucketed to 0.01 precision.
        This balances:
        - Too fine (resolution=4): Few collisions, less entanglement benefit
        - Too coarse (resolution=1): Many false collisions, wrong updates
        
        φ⁻² ≈ 0.38 is the spectral gap, so 0.01 resolution captures meaningful
        distinctions while allowing semantic equivalence.
    """
    
    # Index: hash(σ, p) → List[(level, satellite_idx)]
    index: Dict[Tuple[float, float], List[Tuple[int, int]]] = field(default_factory=dict)
    
    # Resolution for quantization (decimal places)
    resolution: int = 2
    
    # Statistics
    total_registrations: int = 0
    total_lookups: int = 0
    entanglement_propagations: int = 0
    
    def hash_witness(self, sigma: float, pseudo: float) -> Tuple[float, float]:
        """
        Quantize witness to bucket key.
        
        Args:
            sigma: Scalar coefficient
            pseudo: Pseudoscalar coefficient
            
        Returns:
            Quantized (sigma, pseudo) tuple
        """
        return (round(sigma, self.resolution), round(pseudo, self.resolution))
    
    def register(
        self,
        sigma: float,
        pseudo: float,
        level: int,
        satellite_idx: int,
    ) -> int:
        """
        Register a memory location with its witness.
        
        Args:
            sigma: Scalar coefficient of the memory
            pseudo: Pseudoscalar coefficient of the memory
            level: Level index in the tower
            satellite_idx: Satellite index at that level
            
        Returns:
            Number of entangled locations (including this one)
        """
        key = self.hash_witness(sigma, pseudo)
        location = (level, satellite_idx)
        
        if key not in self.index:
            self.index[key] = []
        
        # Avoid duplicate registrations
        if location not in self.index[key]:
            self.index[key].append(location)
            self.total_registrations += 1
        
        return len(self.index[key])
    
    def unregister(
        self,
        sigma: float,
        pseudo: float,
        level: int,
        satellite_idx: int,
    ) -> bool:
        """
        Remove a memory location from the index.
        
        Args:
            sigma: Scalar coefficient
            pseudo: Pseudoscalar coefficient
            level: Level index
            satellite_idx: Satellite index
            
        Returns:
            True if location was found and removed
        """
        key = self.hash_witness(sigma, pseudo)
        location = (level, satellite_idx)
        
        if key in self.index and location in self.index[key]:
            self.index[key].remove(location)
            if not self.index[key]:
                del self.index[key]
            return True
        return False
    
    def get_entangled(
        self,
        sigma: float,
        pseudo: float,
        exclude: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Find all memory locations with same witness (entangled peers).
        
        Args:
            sigma: Scalar coefficient
            pseudo: Pseudoscalar coefficient
            exclude: Optional location to exclude (e.g., the primary update location)
            
        Returns:
            List of (level, satellite_idx) pairs
        """
        self.total_lookups += 1
        key = self.hash_witness(sigma, pseudo)
        
        locations = self.index.get(key, [])
        
        if exclude is not None:
            locations = [loc for loc in locations if loc != exclude]
        
        return locations
    
    def get_entanglement_count(self, sigma: float, pseudo: float) -> int:
        """
        Get number of entangled locations for a witness.
        
        Args:
            sigma: Scalar coefficient
            pseudo: Pseudoscalar coefficient
            
        Returns:
            Number of entangled memory locations
        """
        key = self.hash_witness(sigma, pseudo)
        return len(self.index.get(key, []))
    
    def register_from_matrix(
        self,
        matrix: Array,
        basis: Array,
        level: int,
        satellite_idx: int,
        xp: ArrayModule = np,
    ) -> int:
        """
        Register a memory location by extracting witness from matrix.
        
        Convenience method that extracts witness and registers.
        
        Args:
            matrix: [4, 4] memory matrix
            basis: [16, 4, 4] Clifford basis
            level: Level index
            satellite_idx: Satellite index
            xp: Array module
            
        Returns:
            Number of entangled locations
        """
        sigma, pseudo = extract_witness(matrix, basis, xp)
        return self.register(sigma, pseudo, level, satellite_idx)
    
    def get_entangled_from_matrix(
        self,
        matrix: Array,
        basis: Array,
        xp: ArrayModule = np,
        exclude: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Find entangled locations by extracting witness from matrix.
        
        Args:
            matrix: [4, 4] memory matrix
            basis: [16, 4, 4] Clifford basis
            xp: Array module
            exclude: Optional location to exclude
            
        Returns:
            List of (level, satellite_idx) pairs
        """
        sigma, pseudo = extract_witness(matrix, basis, xp)
        return self.get_entangled(sigma, pseudo, exclude)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with statistics
        """
        n_buckets = len(self.index)
        n_locations = sum(len(locs) for locs in self.index.values())
        
        if n_buckets > 0:
            avg_entanglement = n_locations / n_buckets
            max_entanglement = max(len(locs) for locs in self.index.values())
        else:
            avg_entanglement = 0.0
            max_entanglement = 0
        
        return {
            'n_buckets': n_buckets,
            'n_locations': n_locations,
            'avg_entanglement': avg_entanglement,
            'max_entanglement': max_entanglement,
            'total_registrations': self.total_registrations,
            'total_lookups': self.total_lookups,
            'entanglement_propagations': self.entanglement_propagations,
        }
    
    def clear(self):
        """Clear the index."""
        self.index.clear()
        self.total_registrations = 0
        self.total_lookups = 0
        self.entanglement_propagations = 0


def propagate_witness_update(
    witness_index: WitnessIndex,
    tower: Any,  # MultiLevelTower or TowerMemory
    sigma: float,
    pseudo: float,
    delta_sigma: float,
    delta_pseudo: float,
    primary_location: Tuple[int, int],
    decay_factor: float = PHI_INV_SQ,
) -> int:
    """
    Propagate witness update to all entangled locations.
    
    QUANTUM THEORY (v5.27.0):
        When a witness is updated at one location, this function propagates
        the update to all semantically equivalent (entangled) locations.
        
        The update decays by φ⁻² per hop to prevent runaway propagation.
        
    BRAIN ANALOG:
        Semantic priming cascade — updating one concept's representation
        updates all related concepts, but with decreasing strength.
        
    Args:
        witness_index: WitnessIndex with registered locations
        tower: Memory tower with satellite memories
        sigma: Original scalar coefficient
        pseudo: Original pseudoscalar coefficient
        delta_sigma: Change in scalar coefficient
        delta_pseudo: Change in pseudoscalar coefficient
        primary_location: Location that was directly updated (to exclude)
        decay_factor: Decay per propagation (default: φ⁻²)
        
    Returns:
        Number of locations updated
    """
    # Find entangled locations (excluding primary)
    entangled = witness_index.get_entangled(sigma, pseudo, exclude=primary_location)
    
    if not entangled:
        return 0
    
    # Get basis for reconstruction
    basis = tower.basis
    xp = tower.xp if hasattr(tower, 'xp') else np
    
    # Propagate update to each entangled location
    updated_count = 0
    for level, sat_idx in entangled:
        try:
            # Apply decayed witness delta
            tower.update_satellite_witness(
                level=level,
                satellite_idx=sat_idx,
                delta_sigma=delta_sigma * decay_factor,
                delta_pseudo=delta_pseudo * decay_factor,
            )
            updated_count += 1
        except (IndexError, AttributeError):
            # Location may have been pruned or tower doesn't support this
            pass
    
    witness_index.entanglement_propagations += updated_count
    return updated_count


def batch_register_witnesses(
    witness_index: WitnessIndex,
    matrices: Array,
    basis: Array,
    level: int,
    start_satellite_idx: int,
    xp: ArrayModule = np,
) -> int:
    """
    Register multiple memory locations in batch.
    
    VECTORIZED for efficiency with large memory systems.
    
    Args:
        witness_index: WitnessIndex to update
        matrices: [N, 4, 4] memory matrices
        basis: [16, 4, 4] Clifford basis
        level: Level index for all matrices
        start_satellite_idx: Starting satellite index
        xp: Array module
        
    Returns:
        Total number of registrations
    """
    # Batch extract witnesses
    witnesses = extract_witness_batch(matrices, basis, xp)  # [N, 2]
    
    # Convert to numpy if needed
    if hasattr(witnesses, 'get'):
        witnesses = witnesses.get()
    
    # Register each
    n_registered = 0
    for i in range(witnesses.shape[0]):
        sigma = float(witnesses[i, 0])
        pseudo = float(witnesses[i, 1])
        satellite_idx = start_satellite_idx + i
        witness_index.register(sigma, pseudo, level, satellite_idx)
        n_registered += 1
    
    return n_registered


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'WitnessIndex',
    'propagate_witness_update',
    'batch_register_witnesses',
]
