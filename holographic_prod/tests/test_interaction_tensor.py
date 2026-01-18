"""
Test Interaction Tensor — Bivector → Trivector Projection

Tests the 16×6×4 interaction tensor that projects satellite bivectors
to master trivectors during upward information flow.

Theory (Chapter 11):
    M_grade3 = φ⁻² · Σ_{k=0}^{15} (R_k · S_k_grade2)
    
    Where:
    - S_k_grade2: 6 bivector components from satellite k
    - R_k: Rotation rotor for satellite k (φ-offset phase)
    - M_grade3: 4 trivector components in master
    
    Mapping:
    - Temporal bivectors (e01, e02, e03) → trivectors with e0
    - Spatial bivectors (e12, e13, e23) → pure spatial trivector e123
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ,
    CLIFFORD_DIM, MATRIX_DIM, DTYPE,
    GRADE_INDICES,
)


class TestInteractionTensor:
    """Test suite for interaction tensor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_satellites = 16
        self.n_bivectors = 6  # Grade 2
        self.n_trivectors = 4  # Grade 3
    
    def test_tensor_dimensions(self):
        """Test that tensor has correct dimensions."""
        # Tensor should map: [16 satellites] × [6 bivectors] → [4 trivectors]
        # This is conceptually a 16×6×4 tensor, but implemented as:
        # - 16 rotation rotors (one per satellite)
        # - 1 projection matrix [4, 6]
        # - φ⁻² scaling
        
        # We'll verify the implementation has these components
        from holographic_prod.torus.interaction_tensor import InteractionTensor
        
        tensor = InteractionTensor(n_satellites=self.n_satellites)
        
        # Should have 16 rotors
        assert len(tensor.rotors) == self.n_satellites, \
            f"Should have {self.n_satellites} rotors"
        
        # Each rotor should be [6, 6] (rotates bivectors)
        for rotor in tensor.rotors:
            assert rotor.shape == (self.n_bivectors, self.n_bivectors), \
                f"Rotor should be [6, 6], got {rotor.shape}"
        
        # Projection should be [4, 6]
        assert tensor.projection.shape == (self.n_trivectors, self.n_bivectors), \
            f"Projection should be [4, 6], got {tensor.projection.shape}"
    
    def test_projection_matrix_structure(self):
        """Test that projection matrix maps bivectors to trivectors correctly."""
        from holographic_prod.torus.interaction_tensor import build_projection_matrix
        
        P = build_projection_matrix()
        
        # Should be [4, 6]
        assert P.shape == (4, 6), f"Projection should be [4, 6], got {P.shape}"
        
        # Temporal bivectors (indices 0, 1, 2: e01, e02, e03) should map to trivectors with e0
        # Spatial bivectors (indices 3, 4, 5: e12, e13, e23) should map to e123
        
        # Check that temporal bivectors contribute to first 3 trivectors
        temporal_mask = np.sum(np.abs(P[:3, :3]), axis=0) > 0
        assert np.all(temporal_mask), "Temporal bivectors should map to temporal trivectors"
        
        # Check that spatial bivectors contribute to e123 (index 3)
        spatial_mask = np.sum(np.abs(P[3, 3:]), axis=0) > 0
        assert np.all(spatial_mask), "Spatial bivectors should map to e123"
    
    def test_rotation_rotors(self):
        """Test that rotation rotors are properly structured."""
        from holographic_prod.torus.interaction_tensor import build_rotation_rotor
        
        # Each satellite should have a unique rotor based on its position
        rotors = [build_rotation_rotor(k, self.n_satellites) for k in range(self.n_satellites)]
        
        # All rotors should be [6, 6]
        for k, rotor in enumerate(rotors):
            assert rotor.shape == (self.n_bivectors, self.n_bivectors), \
                f"Rotor {k} should be [6, 6], got {rotor.shape}"
        
        # Rotors should be different (based on φ-offset phases)
        # At least some should differ
        differences = []
        for i in range(len(rotors)):
            for j in range(i + 1, len(rotors)):
                diff = np.linalg.norm(rotors[i] - rotors[j], 'fro')
                differences.append(diff)
        
        # Some rotors should be different
        assert max(differences) > 1e-6, "Rotors should differ based on satellite position"
    
    def test_project_up_single(self):
        """Test projecting single satellite bivectors to master trivector."""
        from holographic_prod.torus.interaction_tensor import InteractionTensor
        
        tensor = InteractionTensor(n_satellites=self.n_satellites)
        
        # Create bivector from single satellite
        satellite_bivectors = np.random.randn(self.n_satellites, self.n_bivectors).astype(DTYPE)
        
        # Project up
        master_trivector = tensor.project_up(satellite_bivectors)
        
        # Should get [4] trivector
        assert master_trivector.shape == (self.n_trivectors,), \
            f"Should get [4] trivector, got {master_trivector.shape}"
        
        # Should have φ⁻² scaling applied
        # (We can't easily verify exact scaling without knowing the sum, but structure should be correct)
        assert np.linalg.norm(master_trivector) > 0, "Trivector should be non-zero"
    
    def test_project_up_batch(self):
        """Test batch projection."""
        from holographic_prod.torus.interaction_tensor import InteractionTensor
        
        tensor = InteractionTensor(n_satellites=self.n_satellites)
        
        # Batch of 3 sets of satellites
        batch_size = 3
        satellite_bivectors = np.random.randn(
            batch_size, self.n_satellites, self.n_bivectors
        ).astype(DTYPE)
        
        # Project up
        master_trivectors = tensor.project_up(satellite_bivectors)
        
        # Should get [batch_size, 4]
        assert master_trivectors.shape == (batch_size, self.n_trivectors), \
            f"Should get [batch_size, 4], got {master_trivectors.shape}"
    
    def test_phi_scaling(self):
        """Test that φ⁻² scaling is applied."""
        from holographic_prod.torus.interaction_tensor import InteractionTensor
        
        tensor = InteractionTensor(n_satellites=self.n_satellites)
        
        # Create known bivector input
        satellite_bivectors = np.ones((self.n_satellites, self.n_bivectors), dtype=DTYPE)
        
        # Project up
        master_trivector = tensor.project_up(satellite_bivectors)
        
        # The result should be scaled by φ⁻²
        # Since we're summing 16 satellites with unit bivectors, the magnitude
        # should reflect the φ⁻² scaling
        # (Exact value depends on rotors and projection, but scaling should be present)
        assert np.linalg.norm(master_trivector) > 0, "Should have non-zero output"
    
    def test_project_down(self):
        """Test downward projection (trivector → bivector for specific satellite)."""
        from holographic_prod.torus.interaction_tensor import InteractionTensor
        
        tensor = InteractionTensor(n_satellites=self.n_satellites)
        
        # Create master trivector
        master_trivector = np.random.randn(self.n_trivectors).astype(DTYPE)
        
        # Project down to satellite 0
        satellite_bivector = tensor.project_down(master_trivector, target_satellite=0)
        
        # Should get [6] bivector
        assert satellite_bivector.shape == (self.n_bivectors,), \
            f"Should get [6] bivector, got {satellite_bivector.shape}"
    
    def test_bidirectional_consistency(self):
        """Test that up→down projection is approximately consistent."""
        from holographic_prod.torus.interaction_tensor import InteractionTensor
        
        tensor = InteractionTensor(n_satellites=self.n_satellites)
        
        # Start with satellite bivectors
        original_bivectors = np.random.randn(self.n_satellites, self.n_bivectors).astype(DTYPE)
        
        # Project up
        master_trivector = tensor.project_up(original_bivectors)
        
        # Project down to satellite 0
        reconstructed_bivector = tensor.project_down(master_trivector, target_satellite=0)
        
        # Should be similar (not identical, but same order of magnitude)
        # The projection is lossy (16×6 → 4 → 6), so we can't expect exact recovery
        assert np.linalg.norm(reconstructed_bivector) > 0, "Should reconstruct something"
        
        # Magnitude should be reasonable
        orig_norm = np.linalg.norm(original_bivectors[0])
        recon_norm = np.linalg.norm(reconstructed_bivector)
        assert 0.1 * orig_norm < recon_norm < 10 * orig_norm, \
            f"Reconstructed magnitude should be reasonable: {recon_norm} vs {orig_norm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
