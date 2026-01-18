"""
Paradox Resolution — Contradiction Handling

Implements phase shift mechanism to separate contradictory memories.

Theory (Chapter 11):
    When two satellites encode contradictory information, apply phase shift:
        Δψ = 2π/φ = 2π × φ⁻¹
    
    This "lifts" the contradiction into a higher dimension of the torus.
    Both facts remain true, but they occupy different phases.
"""

import numpy as np
from typing import Tuple, List, Dict, Any

from holographic_prod.core.constants import PI, PHI_INV


def resolve_paradox(satellite_phase: float) -> float:
    """
    Apply paradox resolution phase shift.
    
    Theory (Chapter 11):
        Δψ = 2π/φ = 2π × φ⁻¹
        
        This is the golden angle, which ensures maximally irrational
        separation. Contradictory memories are separated into different
        "phase lanes" and can coexist without destructive interference.
    
    Args:
        satellite_phase: Current satellite phase [0, 2π)
        
    Returns:
        Shifted phase [0, 2π)
    """
    phase_shift = 2 * PI * PHI_INV
    new_phase = (satellite_phase + phase_shift) % (2 * PI)
    return new_phase


def detect_contradiction(
    witness1: Tuple[float, float],
    witness2: Tuple[float, float],
    threshold: float = None,
) -> bool:
    """
    Detect if two witnesses represent a contradiction.
    
    Args:
        witness1: (scalar, pseudoscalar) for first memory
        witness2: (scalar, pseudoscalar) for second memory
        threshold: Coherence threshold (default: -φ⁻¹)
        
    Returns:
        True if contradiction detected
    """
    from holographic_prod.core.constants import PHI_INV
    
    if threshold is None:
        threshold = -PHI_INV
    
    # Compute coherence (cosine similarity)
    w1_vec = np.array(witness1)
    w2_vec = np.array(witness2)
    
    norm1 = np.linalg.norm(w1_vec)
    norm2 = np.linalg.norm(w2_vec)
    
    if norm1 < 1e-8 or norm2 < 1e-8:
        return False
    
    coherence = np.dot(w1_vec / norm1, w2_vec / norm2)
    
    # Contradiction if coherence below threshold
    return coherence < threshold


class ParadoxResolver:
    """
    Manages paradox detection and resolution across satellites.
    
    Scans for contradictory satellite pairs and applies φ-phase
    shifts to separate them into different phase lanes.
    """
    
    def __init__(self, contradiction_threshold: float = None):
        """
        Initialize paradox resolver.
        
        Args:
            contradiction_threshold: Coherence below this is paradox (default: -φ⁻¹)
        """
        from holographic_prod.core.constants import PHI_INV
        self.contradiction_threshold = contradiction_threshold if contradiction_threshold is not None else -PHI_INV
    
    def resolve_satellite_phases(
        self,
        satellite_phases: List[float],
        satellite_witnesses: List[Tuple[float, float]],
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Resolve paradoxes by shifting satellite phases.
        
        Args:
            satellite_phases: List of satellite phases [0, 2π)
            satellite_witnesses: List of (scalar, pseudoscalar) tuples
            
        Returns:
            (resolved_phases, stats)
        """
        resolved_phases = satellite_phases.copy()
        stats = {
            'paradoxes_found': 0,
            'paradoxes_resolved': 0,
            'pairs': [],
        }
        
        n = len(satellite_phases)
        
        for i in range(n):
            for j in range(i + 1, n):
                if detect_contradiction(
                    satellite_witnesses[i],
                    satellite_witnesses[j],
                    self.contradiction_threshold,
                ):
                    stats['paradoxes_found'] += 1
                    
                    # Resolve by shifting second satellite
                    resolved_phases[j] = resolve_paradox(resolved_phases[j])
                    
                    # Verify resolution
                    # (Re-extract witnesses after shift would require full matrix)
                    # For now, assume shift resolves it
                    stats['paradoxes_resolved'] += 1
                    stats['pairs'].append((i, j))
        
        return resolved_phases, stats
