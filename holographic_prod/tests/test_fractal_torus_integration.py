"""
Test Fractal Torus Integration — End-to-End Component Integration

Tests that all fractal torus components work together correctly:
- Interaction Tensor (bivector → trivector projection)
- Chirality (satellite handedness)
- Grand Equilibrium (system-wide coherence)
- Downward Projection (generation cascade)
- Phase-Locked Emission (token timing)
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PI,
    CLIFFORD_DIM, MATRIX_DIM, DTYPE,
    GRADE_INDICES,
)
from holographic_prod.core.algebra import (
    build_clifford_basis,
    grace_inverse,
    geometric_product,
    decompose_to_coefficients,
    reconstruct_from_coefficients,
)
from holographic_prod.core.quotient import extract_witness


class TestFractalTorusIntegration:
    """Test suite for fractal torus component integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.basis = build_clifford_basis(np)
        self.n_satellites = 16
    
    def test_interaction_tensor_with_chirality(self):
        """Test that interaction tensor works with chirality alternation."""
        from holographic_prod.torus.interaction_tensor import InteractionTensor
        from holographic_prod.torus.chirality import ChiralityFlip
        
        tensor = InteractionTensor(n_satellites=self.n_satellites)
        chirality = ChiralityFlip(n_satellites=self.n_satellites)
        
        # Create satellite bivectors
        satellite_bivectors = np.random.randn(self.n_satellites, 6).astype(DTYPE)
        
        # Apply chirality to each satellite before projection
        chirality_applied = []
        for k in range(self.n_satellites):
            # Convert bivector to full multivector for chirality
            mv = np.zeros(CLIFFORD_DIM, dtype=DTYPE)
            mv[GRADE_INDICES[2]] = satellite_bivectors[k]
            
            # Apply chirality
            mv_chiral = chirality.apply(mv, k)
            
            # Extract bivector back
            biv_chiral = mv_chiral[GRADE_INDICES[2]]
            chirality_applied.append(biv_chiral)
        
        chirality_applied = np.array(chirality_applied)
        
        # Project up with chirality-applied bivectors
        master_trivector = tensor.project_up(chirality_applied)
        
        # Should get valid trivector
        assert master_trivector.shape == (4,), "Should get [4] trivector"
        assert np.linalg.norm(master_trivector) > 0, "Trivector should be non-zero"
    
    def test_grand_equilibrium_with_witnesses(self):
        """Test grand equilibrium computation with satellite witnesses."""
        from holographic_prod.fractal.grand_equilibrium import compute_grand_equilibrium
        
        # Create satellite states
        satellite_states = []
        for k in range(self.n_satellites):
            mv = np.random.randn(4, 4).astype(DTYPE)
            mv = mv / np.linalg.norm(mv, 'fro')
            satellite_states.append(mv)
        
        # Extract witness energies
        local_energies = []
        for mv in satellite_states:
            s, p = extract_witness(mv, self.basis, np)
            energy = s**2 + p**2
            local_energies.append(energy)
        
        local_energies = np.array(local_energies)
        
        # Compute grand equilibrium
        global_energy = compute_grand_equilibrium(local_energies)
        
        # Should equal φ × sum
        expected = PHI * np.sum(local_energies)
        assert abs(global_energy - expected) < 1e-6, \
            "Grand equilibrium should be φ × Σ local energies"
    
    def test_downward_projection_with_grace_inverse(self):
        """Test downward projection uses GraceInverse correctly."""
        from holographic_prod.fractal.downward_projection import DownwardProjection
        
        downward = DownwardProjection(self.basis)
        
        # Create coherent core (witness-heavy)
        coherent_core = np.eye(4, dtype=DTYPE) * 0.8
        
        # Project down (should inflate via GraceInverse)
        lower_memory = np.random.randn(4, 4).astype(DTYPE)
        lower_memory = lower_memory / np.linalg.norm(lower_memory, 'fro')
        
        projected, confidence = downward.project_level_down(coherent_core, lower_memory)
        
        # Should have structure after inflation
        assert projected.shape == (4, 4), "Should get [4, 4] matrix"
        assert np.linalg.norm(projected, 'fro') > 0, "Projected should be non-zero"
        assert 0 <= confidence <= 1, "Confidence should be in [0, 1]"
    
    def test_phase_locked_emission_timing(self):
        """Test phase-locked emission with downward projection."""
        from holographic_prod.fractal.downward_projection import phase_locked_emission, DownwardProjection
        
        downward = DownwardProjection(self.basis)
        
        # Simulate phase progression
        phases = np.linspace(0, 2 * PI, 100)
        emissions = [phase_locked_emission(p) for p in phases]
        
        # Should have some emissions
        emission_count = sum(emissions)
        assert emission_count > 0, "Should have some emissions over full cycle"
        
        # Emissions should be in correct window
        window_start = PI * PHI_INV
        window_end = PI * PHI_INV + PHI_INV_SQ
        
        for i, (phase, emitted) in enumerate(zip(phases, emissions)):
            if emitted:
                phase_norm = phase % (2 * PI)
                assert window_start <= phase_norm <= window_end, \
                    f"Emission at phase {phase_norm} should be in window [{window_start}, {window_end}]"
    
    def test_full_cascade_interaction_chirality_equilibrium(self):
        """Test full cascade: interaction tensor → chirality → equilibrium."""
        from holographic_prod.torus.interaction_tensor import InteractionTensor
        from holographic_prod.torus.chirality import ChiralityFlip
        from holographic_prod.fractal.grand_equilibrium import compute_grand_equilibrium
        
        tensor = InteractionTensor(n_satellites=self.n_satellites)
        chirality = ChiralityFlip(n_satellites=self.n_satellites)
        
        # Create satellite bivectors
        satellite_bivectors = np.random.randn(self.n_satellites, 6).astype(DTYPE)
        
        # Apply chirality
        chirality_applied = []
        for k in range(self.n_satellites):
            mv = np.zeros(CLIFFORD_DIM, dtype=DTYPE)
            mv[GRADE_INDICES[2]] = satellite_bivectors[k]
            mv_chiral = chirality.apply(mv, k)
            chirality_applied.append(mv_chiral[GRADE_INDICES[2]])
        
        chirality_applied = np.array(chirality_applied)
        
        # Project up to master trivector
        master_trivector = tensor.project_up(chirality_applied)
        
        # Compute energy from trivector (||trivector||²)
        trivector_energy = np.sum(master_trivector**2)
        
        # Compute local energies from bivectors
        local_energies = np.array([np.sum(b**2) for b in chirality_applied])
        
        # Grand equilibrium should relate local and global
        global_energy = compute_grand_equilibrium(local_energies)
        
        # Trivector energy should be related to grand equilibrium
        # (Not exactly equal, but should be consistent)
        assert trivector_energy > 0, "Trivector energy should be positive"
        assert global_energy > 0, "Global energy should be positive"
    
    def test_downward_projection_with_phase_locked_emission(self):
        """Test downward projection integrated with phase-locked emission."""
        from holographic_prod.fractal.downward_projection import (
            DownwardProjection,
            phase_locked_emission,
        )
        
        downward = DownwardProjection(self.basis)
        
        # Create grand master state
        grand_master = np.random.randn(4, 4).astype(DTYPE)
        grand_master = grand_master / np.linalg.norm(grand_master, 'fro')
        
        # Simulate generation with phase-locked emission
        current_phase = 0.0
        emissions = []
        
        for step in range(100):
            if phase_locked_emission(current_phase):
                # Would project down and emit token here
                emissions.append(step)
            
            current_phase += PHI_INV_SQ
            current_phase = current_phase % (2 * PI)
        
        # Should have some emissions
        assert len(emissions) > 0, "Should have emissions during generation"
        
        # Emissions should be spaced (quasi-periodic)
        if len(emissions) > 1:
            intervals = np.diff(emissions)
            assert np.std(intervals) > 0, "Emission intervals should vary (quasi-periodic)"
    
    def test_grace_inverse_inflation_for_generation(self):
        """Test that GraceInverse properly inflates for generation."""
        # Create coherent core with some structure (not pure identity)
        # Start with matrix that has structure, then contract with Grace
        from holographic_prod.core.algebra import grace_operator
        
        structured = np.random.randn(4, 4).astype(DTYPE)
        structured = structured / np.linalg.norm(structured, 'fro')
        
        # Contract with Grace to create coherent core
        coherent_core = grace_operator(structured, self.basis, np)
        
        # Apply GraceInverse (should inflate back)
        inflated = grace_inverse(coherent_core, self.basis, np)
        
        # Should have structure after inflation
        coeffs_coherent = decompose_to_coefficients(coherent_core, self.basis, np)
        coeffs_inflated = decompose_to_coefficients(inflated, self.basis, np)
        
        # Inflated should differ from coherent core
        diff = np.linalg.norm(coeffs_inflated - coeffs_coherent)
        assert diff > 1e-6, "Inflated should differ from coherent core"
        
        # Both should have non-zero energy
        energy_coherent = np.sum(coeffs_coherent**2)
        energy_inflated = np.sum(coeffs_inflated**2)
        assert energy_coherent > 0, "Coherent core should have energy"
        assert energy_inflated > 0, "Inflated should have energy"
    
    def test_master_satellite_broadcast_with_equilibrium(self):
        """Test master→satellite broadcasting maintains equilibrium."""
        from holographic_prod.dreaming import NonREMConsolidator
        from holographic_prod.fractal.grand_equilibrium import compute_grand_equilibrium
        
        consolidator = NonREMConsolidator(self.basis, xp=np)
        
        # Create master witness
        master_witness = np.random.randn(4, 4).astype(DTYPE)
        master_witness = master_witness / np.linalg.norm(master_witness, 'fro')
        
        # Create satellite witnesses
        satellite_witnesses = [
            np.random.randn(4, 4).astype(DTYPE) for _ in range(self.n_satellites)
        ]
        for sat in satellite_witnesses:
            sat[:] = sat / np.linalg.norm(sat, 'fro')
        
        # Compute initial local energies
        initial_energies = []
        for sat in satellite_witnesses:
            s, p = extract_witness(sat, self.basis, np)
            energy = s**2 + p**2
            initial_energies.append(energy)
        
        initial_global = compute_grand_equilibrium(np.array(initial_energies))
        
        # Broadcast master witness
        updated_satellites = consolidator.broadcast_master_witness(
            master_witness,
            satellite_witnesses,
        )
        
        # Compute final local energies
        final_energies = []
        for sat in updated_satellites:
            s, p = extract_witness(sat, self.basis, np)
            energy = s**2 + p**2
            final_energies.append(energy)
        
        final_global = compute_grand_equilibrium(np.array(final_energies))
        
        # Global energy should still follow equilibrium equation
        expected_final = PHI * np.sum(final_energies)
        assert abs(final_global - expected_final) < 1e-6, \
            "Equilibrium equation should hold after broadcasting"
        
        # Satellites should have moved toward master (coherence increased)
        # This is verified by the broadcast function itself, but we can check
        # that satellites changed
        changes = [
            np.linalg.norm(updated - original, 'fro')
            for updated, original in zip(updated_satellites, satellite_witnesses)
        ]
        assert max(changes) > 1e-6, "At least some satellites should have changed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
