"""
Grand Equilibrium — System-Wide Coherence

Ensures coherence across all levels of the fractal torus.

Theory (Chapter 11):
    E_Global = φ × Σ E_Local
    
    The golden ratio φ appears as the coupling constant between local
    and global energy. This ensures system-wide coherence.

NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import Union

from holographic_prod.core.constants import (
    PHI,
    DTYPE,
)


def compute_grand_equilibrium(local_energies: np.ndarray) -> float:
    """
    Compute grand equilibrium energy.
    
    Theory:
        E_Global = φ × Σ E_Local
    
    Args:
        local_energies: [n] array of local energies (e.g., witness energies)
        
    Returns:
        Global equilibrium energy
    """
    return float(PHI * np.sum(local_energies))


def check_equilibrium(
    global_energy: float,
    local_energies: np.ndarray,
    tolerance: float = 1e-6,
) -> bool:
    """
    Check if system is in equilibrium.
    
    Args:
        global_energy: Measured global energy
        local_energies: [n] array of local energies
        tolerance: Numerical tolerance
        
    Returns:
        True if E_Global ≈ φ × Σ E_Local
    """
    expected_energy = compute_grand_equilibrium(local_energies)
    return abs(global_energy - expected_energy) < tolerance
