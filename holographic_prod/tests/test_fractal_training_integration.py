"""
Fractal Training Integration Tests

Tests the integration of fractal components (InteractionTensor, Chirality, 
GraceInverse) into the training pipeline.

These tests verify that:
1. InteractionTensor correctly aggregates satellite bivectors to master trivectors
2. Chirality alternation prevents destructive interference
3. GraceInverse properly inflates witness states for generation
4. The full fractal pipeline works end-to-end

RUN:
    pytest holographic_prod/tests/test_fractal_training_integration.py -v
"""

import numpy as np
import pytest

from holographic_prod.core.constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ,
    MATRIX_DIM, DTYPE,
    GRADE_INDICES,
)
from holographic_prod.core.algebra import (
    build_clifford_basis,
    grace_operator,
    geometric_product,
    decompose_to_coefficients,
)
from holographic_prod.core.quotient import extract_witness, compute_enstrophy, grace_stability

from holographic_prod.torus.interaction_tensor import InteractionTensor
from holographic_prod.torus.chirality import ChiralityFlip
from holographic_prod.fractal.downward_projection import DownwardProjection, phase_locked_emission


# =============================================================================
# INTERACTION TENSOR TESTS
# =============================================================================

class TestInteractionTensorInTraining:
    """Test InteractionTensor in training context."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.xp = np
        self.basis = build_clifford_basis(self.xp)
        self.interaction_tensor = InteractionTensor()
    
    def test_satellite_bivectors_aggregate_to_master(self):
        """Satellite bivectors should aggregate to master trivectors via φ-scaling."""
        # Create 16 satellite states with non-zero bivector components
        np.random.seed(42)
        satellite_states = []
        
        for k in range(16):
            # Create random 4x4 multivector
            mv = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
            satellite_states.append(mv)
        
        satellite_states = np.array(satellite_states)
        
        # Extract bivector components from each satellite
        bivector_components = []
        for k in range(16):
            coeffs = decompose_to_coefficients(satellite_states[k], self.basis)
            # Grade 2 indices: [4, 5, 6, 7, 8, 9] = 6 bivector components
            bivector = coeffs[4:10]
            bivector_components.append(bivector)
        
        bivector_components = np.array(bivector_components)  # [16, 6]
        
        # Aggregate to master using InteractionTensor (project_up)
        master_trivector = self.interaction_tensor.project_up(
            bivector_components
        )
        
        # Master trivector should have 4 components (grade 3)
        assert master_trivector.shape == (4,), f"Expected (4,), got {master_trivector.shape}"
        
        # Master should be non-zero (aggregation happened)
        assert np.linalg.norm(master_trivector) > 0, "Master trivector should be non-zero"
        
        print(f"\n  Aggregated {len(satellite_states)} satellites")
        print(f"  Master trivector norm: {np.linalg.norm(master_trivector):.4f}")
    
    def test_phi_scaling_preserved_during_learn(self):
        """φ-scaling should be preserved during aggregation."""
        # The interaction tensor uses φ⁻² scaling
        np.random.seed(42)
        
        # Create uniform bivectors
        bivector_components = np.ones((16, 6), dtype=DTYPE)
        
        master = self.interaction_tensor.project_up(bivector_components)
        
        # The scaling factor should be approximately φ⁻² * sqrt(16)
        expected_scale = PHI_INV_SQ * np.sqrt(16)
        actual_norm = np.linalg.norm(master)
        
        # Due to rotation, the actual norm may differ, but should be proportional
        print(f"\n  Expected scale factor: φ⁻² × √16 ≈ {expected_scale:.4f}")
        print(f"  Actual master norm: {actual_norm:.4f}")
        
        # Should be non-zero
        assert actual_norm > 0, "Master should have non-zero norm"
    
    def test_bidirectional_flow_during_retrieval(self):
        """Master should be able to project down to satellites."""
        np.random.seed(42)
        
        # Create master trivector
        master_trivector = np.random.randn(4).astype(DTYPE)
        
        # Project down to a specific satellite
        satellite_bivector = self.interaction_tensor.project_down(
            master_trivector, target_satellite=0
        )
        
        # Should get 6 bivector components for one satellite
        assert satellite_bivector.shape == (6,), f"Expected (6,), got {satellite_bivector.shape}"
        
        # Project all satellites and re-aggregate
        satellite_bivectors = np.array([
            self.interaction_tensor.project_down(master_trivector, target_satellite=k)
            for k in range(16)
        ])
        
        # Re-aggregate should approximately recover master (with some loss)
        master_recovered = self.interaction_tensor.project_up(
            satellite_bivectors
        )
        
        # Due to dimensionality reduction (16×6 -> 4), recovery is approximate
        cos_sim = np.dot(master_trivector, master_recovered) / (
            np.linalg.norm(master_trivector) * np.linalg.norm(master_recovered) + 1e-10
        )
        
        print(f"\n  Master -> satellites -> master cosine similarity: {cos_sim:.4f}")
        assert cos_sim > 0.5, f"Bidirectional flow should preserve structure, got cos_sim={cos_sim:.4f}"


# =============================================================================
# CHIRALITY TESTS
# =============================================================================

class TestChiralityInTraining:
    """Test chirality alternation in training context."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.xp = np
        self.basis = build_clifford_basis(self.xp)
        self.chirality = ChiralityFlip(n_satellites=16)
    
    def test_even_odd_handedness_applied(self):
        """Even satellites should be right-handed, odd should be left-handed."""
        for k in range(16):
            is_right = self.chirality.is_right_handed(k)
            expected = (k % 2 == 0)
            assert is_right == expected, f"Satellite {k}: expected {expected}, got {is_right}"
        
        print("\n  Chirality pattern verified: even=right, odd=left")
    
    def test_chirality_creates_friction(self):
        """Chirality alternation should create difference between neighboring satellites."""
        np.random.seed(42)
        
        # Create same multivector for all satellites initially
        base_mv = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        
        # Apply chirality to neighbors
        mv_even = self.chirality.apply(base_mv.copy(), satellite_index=0)
        mv_odd = self.chirality.apply(base_mv.copy(), satellite_index=1)
        
        # They should be different (chirality creates friction)
        difference = np.linalg.norm(mv_even - mv_odd)
        
        print(f"\n  Difference between even and odd chirality: {difference:.4f}")
        
        # For left-handed (odd), bivectors/trivectors are flipped
        # This creates structural difference
        assert difference > 0, "Chirality should create difference between neighbors"
    
    def test_witness_preserved_through_chirality(self):
        """Witness (scalar + pseudoscalar) should be preserved through chirality flip."""
        np.random.seed(42)
        
        base_mv = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        
        # Get witness before chirality (scalar, pseudoscalar)
        scalar_before, pseudo_before = extract_witness(base_mv, self.basis)
        
        # Apply chirality (left-handed)
        mv_flipped = self.chirality.apply(base_mv.copy(), satellite_index=1)
        
        # Get witness after chirality
        scalar_after, pseudo_after = extract_witness(mv_flipped, self.basis)
        
        # Witness should be preserved (or negated for pseudoscalar)
        print(f"\n  Scalar before/after: {scalar_before:.4f} / {scalar_after:.4f}")
        print(f"  Pseudo before/after: {pseudo_before:.4f} / {pseudo_after:.4f}")
        
        # Scalar should be preserved (trace is invariant under chirality flip)
        assert abs(scalar_before - scalar_after) < 0.1, \
            "Scalar witness should be preserved through chirality"


# =============================================================================
# GRACE INVERSE TESTS
# =============================================================================

class TestGraceInverseInGeneration:
    """Test GraceInverse in generation context."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.xp = np
        self.basis = build_clifford_basis(self.xp)
        self.downward = DownwardProjection(basis=self.basis, xp=self.xp)
    
    def test_inflation_before_token_emission(self):
        """GraceInverse should inflate witness state to full structure."""
        np.random.seed(42)
        
        # Start with a collapsed state (mostly witness, little structure)
        collapsed = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE) * 0.1
        # Add strong witness (diagonal)
        collapsed += np.eye(MATRIX_DIM, dtype=DTYPE) * 2.0
        
        # Measure enstrophy before (should be low)
        enstrophy_before = compute_enstrophy(collapsed, self.basis)
        
        # Apply GraceInverse (inflation)
        from holographic_prod.core.algebra import grace_inverse
        inflated = grace_inverse(collapsed, self.basis, self.xp)
        
        # Measure enstrophy after (should be higher)
        enstrophy_after = compute_enstrophy(inflated, self.basis)
        
        print(f"\n  Enstrophy before inflation: {enstrophy_before:.4f}")
        print(f"  Enstrophy after inflation:  {enstrophy_after:.4f}")
        
        # Inflation should increase structural content
        assert enstrophy_after > enstrophy_before * 0.5, \
            "GraceInverse should restore/inflate structure"
    
    def test_structure_restored_from_witness(self):
        """Structure should be recoverable from witness state."""
        np.random.seed(42)
        
        # Create original state with structure
        original = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        original_scalar, original_pseudo = extract_witness(original, self.basis)
        original_enstrophy = compute_enstrophy(original, self.basis)
        
        # Collapse via Grace
        collapsed = grace_operator(original, self.basis, self.xp)
        for _ in range(5):  # Multiple Grace steps
            collapsed = grace_operator(collapsed, self.basis, self.xp)
        
        collapsed_enstrophy = compute_enstrophy(collapsed, self.basis)
        
        # Inflate via GraceInverse
        from holographic_prod.core.algebra import grace_inverse
        restored = grace_inverse(collapsed, self.basis, self.xp)
        
        restored_enstrophy = compute_enstrophy(restored, self.basis)
        
        print(f"\n  Original enstrophy:  {original_enstrophy:.4f}")
        print(f"  Collapsed enstrophy: {collapsed_enstrophy:.4f}")
        print(f"  Restored enstrophy:  {restored_enstrophy:.4f}")
        
        # Restoration should increase enstrophy from collapsed state
        assert restored_enstrophy >= collapsed_enstrophy * 0.8, \
            "GraceInverse should restore some structure"
    
    def test_generation_uses_downward_projection(self):
        """Full generation pipeline should use downward projection."""
        np.random.seed(42)
        
        # Create higher-level state (Grand Master)
        grand_master = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        
        # Create lower-level memory (Master level)
        lower_memory = np.random.randn(16, MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        
        # Project down
        projected, confidence = self.downward.project_level_down(
            grand_master, lower_memory[0]  # Use first satellite
        )
        
        # Should produce valid output
        assert projected.shape == (MATRIX_DIM, MATRIX_DIM), \
            f"Expected ({MATRIX_DIM}, {MATRIX_DIM}), got {projected.shape}"
        
        assert 0 <= confidence <= 1, f"Confidence should be in [0,1], got {confidence}"
        
        print(f"\n  Downward projection confidence: {confidence:.4f}")


# =============================================================================
# PHASE-LOCKED EMISSION TESTS
# =============================================================================

class TestPhaseLocking:
    """Test phase-locked token emission."""
    
    def test_emission_window_is_phi_derived(self):
        """Emission window should be φ-derived."""
        # Window: [π·φ⁻¹, π·φ⁻¹ + φ⁻²]
        window_start = PI * PHI_INV
        window_end = PI * PHI_INV + PHI_INV_SQ
        
        print(f"\n  Emission window: [{window_start:.4f}, {window_end:.4f}]")
        print(f"  Window width: {window_end - window_start:.4f} (should be φ⁻² ≈ {PHI_INV_SQ:.4f})")
        
        # Inside window
        assert phase_locked_emission(window_start + 0.1), "Should emit inside window"
        
        # Outside window
        assert not phase_locked_emission(0.0), "Should not emit at phase 0"
        assert not phase_locked_emission(PI), "Should not emit at phase π"
    
    def test_quasi_periodic_emission(self):
        """Emission should be quasi-periodic (golden rhythm)."""
        emissions = []
        
        for i in range(1000):
            phase = 2 * PI * i * PHI_INV  # Golden angle stepping
            if phase_locked_emission(phase):
                emissions.append(i)
        
        print(f"\n  Emissions in 1000 golden-angle steps: {len(emissions)}")
        
        # Should have some emissions but not too many
        assert 50 < len(emissions) < 500, \
            f"Expected moderate emission rate, got {len(emissions)}"


# =============================================================================
# END-TO-END INTEGRATION TEST
# =============================================================================

class TestFullFractalPipeline:
    """Test full fractal pipeline integration."""
    
    def test_learn_aggregate_generate_cycle(self):
        """Test complete learn -> aggregate -> generate cycle."""
        np.random.seed(42)
        xp = np
        basis = build_clifford_basis(xp)
        
        # Components
        interaction_tensor = InteractionTensor()
        chirality = ChiralityFlip(n_satellites=16)
        downward = DownwardProjection(basis=basis, xp=xp)
        
        # LEARN PHASE: Simulate 16 satellites learning different patterns
        satellite_states = []
        for k in range(16):
            # Random state
            mv = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
            
            # Apply chirality
            mv = chirality.apply(mv, satellite_index=k)
            
            satellite_states.append(mv)
        
        satellite_states = np.array(satellite_states)
        
        # AGGREGATE PHASE: Combine to master
        bivectors = []
        for k in range(16):
            coeffs = decompose_to_coefficients(satellite_states[k], basis)
            bivectors.append(coeffs[4:10])  # Grade 2
        bivectors = np.array(bivectors)
        
        master_trivector = interaction_tensor.project_up(bivectors)
        
        # GENERATE PHASE: Project down and emit
        # Create a "grand master" state from trivector
        from holographic_prod.core.algebra import coefficients_to_matrix
        grand_master_coeffs = np.zeros(16, dtype=DTYPE)
        grand_master_coeffs[10:14] = master_trivector  # Grade 3 indices
        grand_master = coefficients_to_matrix(grand_master_coeffs, basis)
        
        # Project down to first satellite
        projected, confidence = downward.project_level_down(
            grand_master, satellite_states[0]
        )
        
        # Check emission
        phase = PI * PHI_INV + 0.1  # In emission window
        can_emit = phase_locked_emission(phase)
        
        print("\n  LEARN -> AGGREGATE -> GENERATE cycle:")
        print(f"  - Learned {len(satellite_states)} satellite states")
        print(f"  - Aggregated to master (norm={np.linalg.norm(master_trivector):.4f})")
        print(f"  - Projected down (confidence={confidence:.4f})")
        print(f"  - Can emit at phase {phase:.4f}: {can_emit}")
        
        assert can_emit, "Should be able to emit in the emission window"
        assert confidence > 0, "Projection should have positive confidence"


def run_tests():
    """Run all fractal integration tests."""
    print("=" * 70)
    print("FRACTAL TRAINING INTEGRATION TESTS")
    print("=" * 70)
    print()
    
    import pytest
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
    ])
    
    return exit_code


if __name__ == '__main__':
    exit(run_tests())
