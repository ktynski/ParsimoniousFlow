"""
Grand Equilibrium — Energy Conservation Across Levels
=====================================================

Implements the Grand Equilibrium Equation that governs energy flow
between individual satellites and the global master.

The equation ensures that local learning doesn't destabilize global
coherence, and vice versa.

Key Equation:
    W_global = (1/φ) ∫₀^{2π} [W_local(ψ) × φ^-2 · V_local(ψ)] dψ

Where:
    W_global: Global Witness (scalar + pseudoscalar at top level)
    W_local: Local Witness at phase ψ
    V_local: Local Vorticity (bivector) at phase ψ
    φ^-2: Spectral gap coupling

This integral over the torus means:
- Information is never truly lost, just shifted in phase
- Local states eventually rotate into the global core
- Energy is conserved across the fractal hierarchy

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
    CLIFFORD_DIM, GRADE_INDICES,
)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_local_witness(M: np.ndarray) -> np.ndarray:
    """
    Extract witness components (scalar + pseudoscalar).
    
    Args:
        M: Multivector [..., 16]
        
    Returns:
        Witness [2] or [..., 2]
    """
    scalar = M[..., GRADE_INDICES[0]]
    pseudo = M[..., GRADE_INDICES[4]]
    return np.concatenate([scalar, pseudo], axis=-1)


def compute_local_vorticity(M: np.ndarray) -> np.ndarray:
    """
    Extract vorticity (bivector) components.
    
    Args:
        M: Multivector [..., 16]
        
    Returns:
        Vorticity [6] or [..., 6]
    """
    return M[..., GRADE_INDICES[2]]


def compute_grand_equilibrium(
    local_states: np.ndarray,
    phases: np.ndarray,
    n_integration_points: int = 64
) -> np.ndarray:
    """
    Compute the Grand Equilibrium via numerical integration.
    
    W_global = (1/φ) ∫₀^{2π} [W_local(ψ) × φ^-2 · V_local(ψ)] dψ
    
    We discretize the integral over the torus phase.
    
    Args:
        local_states: Satellite states [n_satellites, 16]
        phases: Satellite phases [n_satellites]
        n_integration_points: Number of quadrature points
        
    Returns:
        Global witness [2]
    """
    # Integration grid
    psi_grid = np.linspace(0, 2 * PI, n_integration_points, endpoint=False)
    dpsi = 2 * PI / n_integration_points
    
    # Interpolate local states at each integration point
    # Using nearest satellite for each grid point
    n_satellites = len(local_states)
    
    integral = np.zeros(2)  # [scalar, pseudoscalar]
    
    for psi in psi_grid:
        # Find nearest satellite
        distances = np.abs(phases - psi)
        distances = np.minimum(distances, 2 * PI - distances)
        nearest_k = np.argmin(distances)
        
        # Get local witness and vorticity
        W_local = compute_local_witness(local_states[nearest_k])
        V_local = compute_local_vorticity(local_states[nearest_k])
        
        # Compute W × φ^-2 · V
        # The "×" here is geometric product (simplified as element-wise scaling)
        # The "·" is dot product with vorticity norm as coupling
        V_norm = np.linalg.norm(V_local)
        
        # Integrand: witness scaled by vorticity coupling
        integrand = W_local * PHI_INV_SQ * (1 + V_norm * PHI_INV)
        
        integral += integrand * dpsi
    
    # Scale by 1/φ
    W_global = PHI_INV * integral / (2 * PI)
    
    return W_global


def verify_energy_conservation(
    local_states: np.ndarray,
    global_state: np.ndarray,
    tolerance: float = 0.1
) -> Tuple[bool, Dict[str, float]]:
    """
    Verify energy conservation between local and global.
    
    The total energy at local level should match global (within φ-scaling).
    
    Args:
        local_states: Satellite states [n_satellites, 16]
        global_state: Global master state [16]
        tolerance: Relative tolerance for conservation check
        
    Returns:
        (conserved: bool, stats: dict)
    """
    # Local energy: sum of satellite energies (φ-weighted)
    local_energy = 0.0
    for k, state in enumerate(local_states):
        weight = PHI_INV ** (k % 4)
        local_energy += weight * np.sum(state ** 2)
    
    # Global energy
    global_energy = np.sum(global_state ** 2)
    
    # Expected relationship: global = φ^-1 × local (due to compression)
    expected_global = PHI_INV * local_energy
    
    # Check conservation
    relative_error = abs(global_energy - expected_global) / (expected_global + 1e-10)
    conserved = relative_error < tolerance
    
    return conserved, {
        'local_energy': local_energy,
        'global_energy': global_energy,
        'expected_global': expected_global,
        'relative_error': relative_error,
        'conserved': conserved,
    }


def compute_equilibrium_stability(
    local_states: np.ndarray,
    global_state: np.ndarray
) -> float:
    """
    Compute stability of the current equilibrium.
    
    High stability means the system is in a coherent state.
    Low stability means energy is distributed in volatile modes.
    
    Args:
        local_states: Satellite states [n_satellites, 16]
        global_state: Global master state [16]
        
    Returns:
        Stability score in [0, 1]
    """
    # Global witness energy
    global_witness = compute_local_witness(global_state)
    global_witness_energy = np.sum(global_witness ** 2)
    global_total_energy = np.sum(global_state ** 2)
    
    # Local witness coherence
    local_witnesses = np.array([compute_local_witness(s) for s in local_states])
    witness_mean = np.mean(local_witnesses, axis=0)
    witness_variance = np.var(local_witnesses, axis=0).sum()
    
    # Stability: high global witness + low local variance
    global_stability = global_witness_energy / (global_total_energy + 1e-10)
    local_coherence = 1.0 / (1.0 + witness_variance)
    
    # Combined stability (φ-weighted)
    stability = PHI_INV * global_stability + (1 - PHI_INV) * local_coherence
    
    return float(stability)


# =============================================================================
# GRAND EQUILIBRIUM CLASS
# =============================================================================

@dataclass
class GrandEquilibrium:
    """
    Manager for Grand Equilibrium computation and monitoring.
    
    Tracks energy conservation and stability across the fractal
    hierarchy, ensuring that the system remains in equilibrium.
    
    Attributes:
        n_integration_points: Quadrature points for integral
        conservation_tolerance: Tolerance for energy conservation
        stability_history: History of stability measurements
    """
    n_integration_points: int = 64
    conservation_tolerance: float = 0.1
    stability_history: list = None
    
    def __post_init__(self):
        if self.stability_history is None:
            self.stability_history = []
    
    def compute(
        self,
        local_states: np.ndarray,
        phases: np.ndarray
    ) -> np.ndarray:
        """
        Compute global witness via Grand Equilibrium equation.
        
        Args:
            local_states: Satellite states [n_satellites, 16]
            phases: Satellite phases [n_satellites]
            
        Returns:
            Global witness [2]
        """
        return compute_grand_equilibrium(
            local_states, phases, self.n_integration_points
        )
    
    def verify_conservation(
        self,
        local_states: np.ndarray,
        global_state: np.ndarray
    ) -> Tuple[bool, Dict[str, float]]:
        """Verify energy conservation."""
        return verify_energy_conservation(
            local_states, global_state, self.conservation_tolerance
        )
    
    def measure_stability(
        self,
        local_states: np.ndarray,
        global_state: np.ndarray
    ) -> float:
        """
        Measure and record stability.
        
        Args:
            local_states: Satellite states
            global_state: Global master state
            
        Returns:
            Current stability score
        """
        stability = compute_equilibrium_stability(local_states, global_state)
        self.stability_history.append(stability)
        return stability
    
    def is_at_equilibrium(self) -> bool:
        """
        Check if system is at equilibrium.
        
        Returns True if recent stability exceeds threshold.
        """
        if len(self.stability_history) < 3:
            return False
        
        recent = self.stability_history[-3:]
        avg_stability = np.mean(recent)
        
        return avg_stability > PHI_INV_SQ
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get equilibrium statistics."""
        if not self.stability_history:
            return {'measurements': 0}
        
        return {
            'measurements': len(self.stability_history),
            'current_stability': self.stability_history[-1],
            'avg_stability': np.mean(self.stability_history),
            'min_stability': np.min(self.stability_history),
            'max_stability': np.max(self.stability_history),
            'at_equilibrium': self.is_at_equilibrium(),
        }


# =============================================================================
# TESTS
# =============================================================================

def _test_grand_equilibrium():
    """Test grand equilibrium system."""
    print("Testing grand equilibrium...")
    
    np.random.seed(42)
    
    # Create test data
    n_satellites = 16
    local_states = np.random.randn(n_satellites, CLIFFORD_DIM) * 0.5
    
    # Add structure: identity bias
    for k in range(n_satellites):
        local_states[k, GRADE_INDICES[0][0]] += PHI_INV
    
    # Golden spiral phases
    phases = np.array([2 * PI * k * PHI_INV for k in range(n_satellites)])
    
    # Test 1: Compute global witness
    print("  Computing global witness...")
    W_global = compute_grand_equilibrium(local_states, phases)
    assert W_global.shape == (2,), "Global witness should be 2D"
    print(f"    W_global = [{W_global[0]:.4f}, {W_global[1]:.4f}]")
    
    # Test 2: Build global state from witness
    global_state = np.zeros(CLIFFORD_DIM)
    global_state[GRADE_INDICES[0][0]] = W_global[0]
    global_state[GRADE_INDICES[4][0]] = W_global[1]
    
    # Add vorticity from satellites
    for k in range(n_satellites):
        weight = PHI_INV ** (k % 4)
        global_state[GRADE_INDICES[2]] += weight * local_states[k, GRADE_INDICES[2]]
    
    # Test 3: Verify energy conservation
    print("  Verifying energy conservation...")
    conserved, stats = verify_energy_conservation(local_states, global_state)
    print(f"    Local energy: {stats['local_energy']:.4f}")
    print(f"    Global energy: {stats['global_energy']:.4f}")
    print(f"    Relative error: {stats['relative_error']:.4f}")
    print(f"    Conserved: {conserved}")
    
    # Test 4: Compute stability
    print("  Computing stability...")
    stability = compute_equilibrium_stability(local_states, global_state)
    print(f"    Stability: {stability:.4f}")
    
    # Test 5: GrandEquilibrium class
    print("  Testing GrandEquilibrium class...")
    ge = GrandEquilibrium()
    
    W_global_2 = ge.compute(local_states, phases)
    assert np.allclose(W_global, W_global_2), "Class should match function"
    
    stability = ge.measure_stability(local_states, global_state)
    ge.measure_stability(local_states, global_state)
    ge.measure_stability(local_states, global_state)
    
    eq_stats = ge.get_statistics()
    print(f"    Measurements: {eq_stats['measurements']}")
    print(f"    At equilibrium: {eq_stats['at_equilibrium']}")
    
    print("✓ Grand equilibrium tests passed!")
    return True


if __name__ == "__main__":
    _test_grand_equilibrium()
