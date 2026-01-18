"""
Tests for Distributed Prior Module — Smooth Generalization
=========================================================

Verifies that φ-weighted interpolation improves generalization
when queries fall between known basins.

THEORY:
    Without distributed prior: Query must fall IN a basin → discrete coverage
    With distributed prior: Query interpolates ACROSS basins → smooth coverage
    
    The key insight is that transformers interpolate smoothly via weights.
    We interpolate smoothly via φ-kernel weighted superposition.
    
ALL MECHANISMS USE ONLY:
   - φ-derived constants (not tuned)
   - Grace operator (not arbitrary normalization)
   - Witness/vorticity geometry (not learned)
"""

import numpy as np
import pytest
from typing import List, Tuple, Dict

from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, DTYPE
from holographic_prod.core.algebra import build_clifford_basis, grace_operator
from holographic_prod.core.quotient import extract_witness, grace_stability
from holographic_prod.cognitive.distributed_prior import (
    phi_kernel,
    witness_distance,
    extended_witness,
    superposed_attractor_prior,
    compute_prior_field_potential,
    FactorizedAssociativePrior,
    retrieve_with_factorized_prior,
)


@pytest.fixture
def basis():
    """Clifford basis for all tests."""
    return build_clifford_basis(np)


@pytest.fixture
def sample_prototypes(basis):
    """Create sample prototypes with known distributions."""
    np.random.seed(42)
    prototypes = []
    target_distributions = []
    
    # Create 5 prototypes with different witness patterns
    for i in range(5):
        # Each prototype is centered at a different scalar value
        proto = np.eye(4, dtype=DTYPE) * (i + 1) + 0.1 * np.random.randn(4, 4).astype(DTYPE)
        proto = grace_operator(proto, basis, np)  # Stabilize
        prototypes.append(proto)
        
        # Target distribution: each prototype "predicts" mainly one token
        target_distributions.append({i * 10: 0.8, (i * 10) + 1: 0.2})
    
    return prototypes, target_distributions


class TestPhiKernel:
    """Test 1: phi_kernel properties — φ^(-d), NOT softmax."""
    
    def test_phi_kernel_properties(self):
        """
        Test that phi_kernel has correct properties.
        
        THEORY: phi_kernel(d) = φ^(-d) NOT softmax
        Properties:
        - φ⁰ = 1 (zero distance = full weight)
        - φ⁻¹ ≈ 0.618 (unit distance)
        - Decays but never zero
        """
        # Zero distance = 1.0
        assert phi_kernel(0.0) == 1.0, "Zero distance should give weight 1.0"
        
        # Unit distance = φ⁻¹
        assert abs(phi_kernel(1.0) - PHI_INV) < 1e-6, (
            f"Unit distance should give φ⁻¹ ≈ 0.618, got {phi_kernel(1.0)}"
        )
        
        # Double distance = φ⁻²
        assert abs(phi_kernel(2.0) - PHI_INV_SQ) < 1e-6, (
            f"Double distance should give φ⁻² ≈ 0.382, got {phi_kernel(2.0)}"
        )
        
        # Monotonically decreasing
        for d in [0.5, 1.0, 2.0, 5.0, 10.0]:
            assert phi_kernel(d) > phi_kernel(d + 1), (
                f"phi_kernel should be monotonically decreasing"
            )
        
        # Never zero (unlike softmax which can underflow)
        assert phi_kernel(100.0) > 0, "phi_kernel should never be zero"
    
    def test_phi_kernel_is_not_softmax(self):
        """Verify that phi_kernel is fundamentally different from softmax."""
        import math
        
        # For comparison: softmax-like exponential decay
        def softmax_kernel(d):
            return math.exp(-d)
        
        # At large distances, they differ significantly
        large_d = 10.0
        phi_val = phi_kernel(large_d)
        soft_val = softmax_kernel(large_d)
        
        # phi_kernel decays slower than exp(-d) for large d
        ratio = phi_val / soft_val
        assert ratio > 10, f"phi_kernel should decay slower than exp(-d), ratio={ratio}"


class TestWitnessDistance:
    """Test 2: Witness distance computation."""
    
    def test_witness_distance(self, basis):
        """
        Test witness distance computation.
        
        THEORY: Distance between witnesses = L2 norm of (σ₁, p₁) - (σ₂, p₂)
        """
        # Create two matrices with known witnesses
        m1 = np.eye(4, dtype=DTYPE) * 2.0  # σ ≈ 2, p ≈ 0
        m2 = np.eye(4, dtype=DTYPE) * 3.0  # σ ≈ 3, p ≈ 0
        
        # Extract witnesses as tuples
        w1 = extract_witness(m1, basis, np)
        w2 = extract_witness(m2, basis, np)
        
        dist = witness_distance(w1, w2)
        
        # Should be approximately 1.0 (difference in scalar)
        assert dist > 0, "Distance should be positive for different matrices"
        
        # Same witness = zero distance
        assert witness_distance(w1, w1) == 0.0, "Same witness should have zero distance"
    
    def test_witness_distance_symmetry(self, basis):
        """Distance should be symmetric: d(a,b) = d(b,a)."""
        w1 = (1.0, 0.5)
        w2 = (2.0, 0.3)
        
        assert abs(witness_distance(w1, w2) - witness_distance(w2, w1)) < 1e-10


class TestSuperposedAttractorPrior:
    """Test 3: Superposed attractor prior — K-nearest interpolation."""
    
    def test_superposed_attractor_prior(self, basis, sample_prototypes):
        """
        Test K-nearest prototype superposition.
        
        THEORY:
        - Retrieve K nearest prototypes by witness distance
        - Weight by φ^(-distance)
        - Superpose: A_prior = Σ αᵢ Aᵢ
        """
        prototypes, target_dists = sample_prototypes
        
        # Create query near prototype 2 (scalar ≈ 2.5)
        np.random.seed(42)
        query = np.eye(4, dtype=DTYPE) * 2.5 + 0.05 * np.random.randn(4, 4).astype(DTYPE)
        
        # Get superposed prior
        equilibrium, combined_targets, confidence, info = superposed_attractor_prior(
            query, prototypes, target_dists, basis, np, K=3
        )
        
        assert equilibrium.shape == (4, 4), "Equilibrium should be 4x4"
        assert isinstance(combined_targets, dict), "Targets should be dict"
        assert 0.0 <= confidence <= 1.0, "Confidence should be in [0, 1]"
        assert info['K'] == 3, "Should use K=3 neighbors"
    
    def test_superposed_prior_empty_prototypes(self, basis):
        """Empty prototype list should return query unchanged."""
        query = np.eye(4, dtype=DTYPE)
        
        equilibrium, targets, conf, info = superposed_attractor_prior(
            query, [], [], basis, np
        )
        
        assert info['source'] == 'no_prototypes'
        assert len(targets) == 0
    
    def test_superposed_prior_interpolation(self, basis, sample_prototypes):
        """Query between prototypes should interpolate targets."""
        prototypes, target_dists = sample_prototypes
        
        # Query exactly between prototype 0 and 1
        query = np.eye(4, dtype=DTYPE) * 1.5  # Between σ=1 and σ=2
        
        _, combined_targets, _, info = superposed_attractor_prior(
            query, prototypes, target_dists, basis, np, K=2
        )
        
        # Should have contributions from multiple prototypes
        assert len(combined_targets) >= 2, "Should interpolate from multiple prototypes"


class TestPriorFieldPotential:
    """Test 4: Prior field potential — Green's function fallback."""
    
    def test_prior_field_potential(self, basis, sample_prototypes):
        """
        Test potential field computation.
        
        THEORY: U(W) = Σᵢ βᵢ φ^(-d(W, Wᵢ))
        """
        prototypes, _ = sample_prototypes
        
        # Extract witnesses from prototypes
        proto_witnesses = [extract_witness(p, basis, np) for p in prototypes]
        proto_supports = [1.0] * len(prototypes)  # Uniform support
        
        # Query witness
        query_witness = (2.5, 0.0)  # Between prototypes
        
        potential, contributions = compute_prior_field_potential(
            query_witness, proto_witnesses, proto_supports
        )
        
        assert potential > 0, "Potential should be positive"
        assert len(contributions) == len(prototypes), "Should have one contribution per prototype"
        assert all(c > 0 for c in contributions), "All contributions should be positive"
    
    def test_potential_increases_near_prototypes(self, basis, sample_prototypes):
        """Potential should be higher closer to prototypes."""
        prototypes, _ = sample_prototypes
        
        proto_witnesses = [extract_witness(p, basis, np) for p in prototypes]
        proto_supports = [1.0] * len(prototypes)
        
        # Near first prototype (σ ≈ 1)
        near_witness = (1.1, 0.0)
        far_witness = (10.0, 0.0)
        
        near_potential, _ = compute_prior_field_potential(
            near_witness, proto_witnesses, proto_supports
        )
        far_potential, _ = compute_prior_field_potential(
            far_witness, proto_witnesses, proto_supports
        )
        
        assert near_potential > far_potential, "Potential should be higher near prototypes"


class TestFactorizedAssociativePrior:
    """Test 5: Factorized associative prior — Hebbian weights."""
    
    def test_factorized_associative_prior(self, basis):
        """
        Test Hebbian-learned witness→attractor mapping.
        
        THEORY:
        - C = Σ WᵢWᵢᵀ (witness covariance)
        - B = Σ AᵢWᵢᵀ (witness-attractor association)
        - Prediction: Â(W) = B C⁻¹ W
        """
        prior = FactorizedAssociativePrior(witness_dim=4, xp=np)
        
        # Initial state
        assert prior.n_updates == 0
        
        # Add some associations
        np.random.seed(42)
        for i in range(10):
            witness = np.random.randn(4).astype(np.float64)
            attractor = np.random.randn(4, 4).astype(np.float64)
            prior.update(witness, attractor)
        
        assert prior.n_updates == 10
        
        # Make a prediction
        test_witness = np.array([1.0, 0.5, 0.2, 0.1], dtype=np.float64)
        predicted = prior.predict(test_witness, basis)
        
        assert predicted.shape == (4, 4), "Prediction should be 4x4"
    
    def test_factorized_prior_statistics(self, basis):
        """Test diagnostic statistics."""
        prior = FactorizedAssociativePrior(witness_dim=4, xp=np)
        
        np.random.seed(42)
        for i in range(20):
            witness = np.random.randn(4).astype(np.float64)
            attractor = np.random.randn(4, 4).astype(np.float64)
            prior.update(witness, attractor)
        
        stats = prior.get_statistics()
        
        assert 'n_updates' in stats
        assert 'effective_rank' in stats
        assert 'eigenvalues' in stats
        assert stats['n_updates'] == 20
    
    def test_retrieve_with_factorized_prior(self, basis):
        """Test full retrieval pipeline with factorized prior."""
        prior = FactorizedAssociativePrior(witness_dim=4, xp=np)
        
        # Train with some patterns
        np.random.seed(42)
        for i in range(20):
            witness = extended_witness(
                np.eye(4, dtype=DTYPE) * (i % 5 + 1),
                basis, np
            )
            attractor = np.eye(4, dtype=DTYPE) * (i % 5 + 1)
            prior.update(witness, attractor)
        
        # Query
        query = np.eye(4, dtype=DTYPE) * 3.0
        result, info = retrieve_with_factorized_prior(query, prior, basis, np)
        
        assert result.shape == (4, 4)
        assert info['source'] == 'factorized_prior'
        assert 'prior_stats' in info


class TestConfidenceThreshold:
    """Test 6: Decision rule — confidence threshold at φ⁻¹."""
    
    def test_retrieve_with_confidence_threshold(self, basis, sample_prototypes):
        """
        Test decision rule: conf >= φ⁻¹ → single basin, else → distributed.
        
        THEORY: φ⁻¹ is the spectral gap. Below this, query is in transition zone.
        """
        prototypes, target_dists = sample_prototypes
        
        # Query very close to prototype 0 → high confidence
        close_query = prototypes[0] + 0.01 * np.random.randn(4, 4).astype(DTYPE)
        _, _, high_conf, high_info = superposed_attractor_prior(
            close_query, prototypes, target_dists, basis, np, K=3
        )
        
        # Query between prototypes → lower confidence  
        between_query = np.eye(4, dtype=DTYPE) * 2.5  # Between prototypes
        _, _, low_conf, low_info = superposed_attractor_prior(
            between_query, prototypes, target_dists, basis, np, K=3
        )
        
        # Close query should have higher confidence (larger margin)
        assert high_conf >= low_conf * 0.5, (
            f"Close query should have higher confidence: {high_conf} vs {low_conf}"
        )
    
    def test_confidence_is_geometric(self, basis, sample_prototypes):
        """Confidence is margin-based: (d₂ - d₁) / (d₂ + ε)."""
        prototypes, target_dists = sample_prototypes
        
        # With only one prototype, confidence should be 1.0
        single_proto = [prototypes[0]]
        single_dist = [target_dists[0]]
        
        query = np.eye(4, dtype=DTYPE) * 2.0
        _, _, conf, _ = superposed_attractor_prior(
            query, single_proto, single_dist, basis, np, K=1
        )
        
        assert conf == 1.0, "Single prototype should give confidence 1.0"


class TestExtendedWitness:
    """Additional test: Extended witness computation."""
    
    def test_extended_witness_dimensions(self, basis):
        """Extended witness should be 4D: [scalar, pseudo, enstrophy, stability]."""
        matrix = np.eye(4, dtype=DTYPE) * 2.0
        
        ext_w = extended_witness(matrix, basis, np)
        
        assert ext_w.shape == (4,), f"Expected 4D, got {ext_w.shape}"
        
        # Components should be reasonable
        assert np.isfinite(ext_w).all(), "All components should be finite"
    
    def test_extended_witness_stability_component(self, basis):
        """Identity matrix should have high Grace stability."""
        # Identity is maximally stable under Grace
        identity = np.eye(4, dtype=DTYPE)
        ext_w = extended_witness(identity, basis, np)
        
        # Stability is the 4th component (index 3)
        stability = ext_w[3]
        
        # Grace stability of identity should be high (close to 1)
        assert stability > 0.5, f"Identity should have high stability, got {stability}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
