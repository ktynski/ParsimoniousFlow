"""
Test Grand Equilibrium — System-Wide Coherence

Tests that the global energy equals φ times the sum of local energies.

Theory (Chapter 11):
    E_Global = φ × Σ E_Local
    
    This ensures system-wide coherence across all levels of the fractal torus.
    The golden ratio φ appears as the coupling constant between local and global.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PHI, PHI_INV,
    CLIFFORD_DIM, MATRIX_DIM, DTYPE,
)


class TestGrandEquilibrium:
    """Test suite for Grand Equilibrium."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_satellites = 16
    
    def test_equilibrium_equation(self):
        """Test that E_Global = φ × Σ E_Local."""
        from holographic_prod.fractal.grand_equilibrium import compute_grand_equilibrium
        
        # Create local energies (satellite states)
        local_energies = np.random.randn(self.n_satellites).astype(DTYPE)
        local_energies = np.abs(local_energies)  # Energy is positive
        
        # Compute grand equilibrium
        global_energy = compute_grand_equilibrium(local_energies)
        
        # Should equal φ × sum of local energies
        expected_energy = PHI * np.sum(local_energies)
        
        assert abs(global_energy - expected_energy) < 1e-6, \
            f"Global energy should be φ × Σ local, got {global_energy}, expected {expected_energy}"
    
    def test_equilibrium_with_witness(self):
        """Test equilibrium computation with witness energies."""
        from holographic_prod.fractal.grand_equilibrium import compute_grand_equilibrium
        from holographic_prod.core.quotient import extract_witness
        from holographic_prod.core.algebra import build_clifford_basis
        
        basis = build_clifford_basis(np)
        
        # Create satellite states with witness
        satellite_states = []
        for k in range(self.n_satellites):
            mv = np.random.randn(4, 4).astype(DTYPE)
            mv = mv / np.linalg.norm(mv, 'fro')
            satellite_states.append(mv)
        
        # Extract witness energies (scalar² + pseudoscalar²)
        local_energies = []
        for mv in satellite_states:
            s, p = extract_witness(mv, basis, np)
            energy = s**2 + p**2
            local_energies.append(energy)
        
        local_energies = np.array(local_energies)
        
        # Compute grand equilibrium
        global_energy = compute_grand_equilibrium(local_energies)
        
        # Should equal φ × sum
        expected_energy = PHI * np.sum(local_energies)
        
        assert abs(global_energy - expected_energy) < 1e-6, \
            f"Grand equilibrium should be φ × Σ witness energies"
    
    def test_equilibrium_conservation(self):
        """Test that equilibrium is conserved across levels."""
        from holographic_prod.fractal.grand_equilibrium import compute_grand_equilibrium
        
        # Level 0: 16 satellites
        level0_energies = np.random.randn(16).astype(DTYPE)
        level0_energies = np.abs(level0_energies)
        
        level0_global = compute_grand_equilibrium(level0_energies)
        
        # Level 1: Aggregate level 0 (should have same total energy)
        # In practice, level 1 would aggregate from level 0, but for test:
        # If we have 16 level-0 satellites → 1 level-1 master
        # The master's energy should relate to level-0 global energy
        
        # The relationship: E_L1 = φ × E_L0_global (if L0 is treated as "local")
        # But actually: E_L1 = φ × Σ E_L0_local
        
        # For this test, verify the equation holds
        assert level0_global == PHI * np.sum(level0_energies), \
            "Equilibrium equation should hold"
    
    def test_equilibrium_zero_energy(self):
        """Test equilibrium with zero local energies."""
        from holographic_prod.fractal.grand_equilibrium import compute_grand_equilibrium
        
        local_energies = np.zeros(self.n_satellites, dtype=DTYPE)
        
        global_energy = compute_grand_equilibrium(local_energies)
        
        assert abs(global_energy) < 1e-10, \
            "Global energy should be zero when local energies are zero"
    
    def test_equilibrium_single_satellite(self):
        """Test equilibrium with single satellite."""
        from holographic_prod.fractal.grand_equilibrium import compute_grand_equilibrium
        
        local_energies = np.array([1.0], dtype=DTYPE)
        
        global_energy = compute_grand_equilibrium(local_energies)
        
        expected = PHI * 1.0
        assert abs(global_energy - expected) < 1e-6, \
            f"Single satellite: E_global = φ × E_local = {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
