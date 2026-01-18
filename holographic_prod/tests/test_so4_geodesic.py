"""
Tests for SO(4) geodesic operations (theory-true contrastive learning).

These functions implement geodesic interpolation on the SO(4) manifold,
enabling contrastive learning that PRESERVES orthogonality.

CRITICAL INVARIANTS:
    1. exp(X) ∈ SO(4) when X ∈ so(4) (skew-symmetric)
    2. log(R) ∈ so(4) when R ∈ SO(4)
    3. geodesic_interpolation_so4(A, B, t) ∈ SO(4) for all t ∈ [0, 1]
    4. geodesic_interpolation_so4(A, B, 0) = A
    5. geodesic_interpolation_so4(A, B, 1) ≈ B
"""

import numpy as np
import pytest
from holographic_prod.core.constants import PHI_INV, DTYPE
from holographic_prod.core.algebra import (
    matrix_exp_skew4,
    matrix_log_so4,
    geodesic_interpolation_so4,
    contrastive_update_so4,
    verify_so4,
    initialize_embeddings_rotor,
)


def random_skew_symmetric(xp=np) -> np.ndarray:
    """Generate random skew-symmetric 4×4 matrix (in so(4) Lie algebra)."""
    A = xp.random.randn(4, 4).astype(DTYPE)
    return (A - A.T) / 2.0


def random_so4(xp=np, seed: int = None) -> np.ndarray:
    """Generate random SO(4) matrix via exponential map."""
    if seed is not None:
        xp.random.seed(seed)
    X = random_skew_symmetric(xp)
    return matrix_exp_skew4(X, xp)


class TestMatrixExpSkew4:
    """Tests for matrix_exp_skew4 (so(4) → SO(4))."""
    
    def test_identity_from_zero(self):
        """exp(0) = I."""
        X = np.zeros((4, 4), dtype=DTYPE)
        result = matrix_exp_skew4(X, np)
        
        assert np.allclose(result, np.eye(4), atol=1e-6), \
            "exp(0) should be identity"
    
    def test_output_is_so4(self):
        """exp(X) ∈ SO(4) for all X ∈ so(4)."""
        np.random.seed(42)
        for _ in range(10):
            X = random_skew_symmetric(np)
            R = matrix_exp_skew4(X, np)
            
            assert verify_so4(R, np, tol=1e-5), \
                f"exp(X) should be SO(4), got orthogonality error"
    
    def test_small_angle_accuracy(self):
        """Taylor expansion for small angles is accurate."""
        # Very small skew-symmetric matrix
        X = random_skew_symmetric(np) * 1e-7
        R = matrix_exp_skew4(X, np)
        
        assert verify_so4(R, np, tol=1e-4), \
            "Small angle approximation should still yield SO(4)"
        
        # Should be close to I + X
        expected = np.eye(4) + X
        assert np.allclose(R, expected, atol=1e-5), \
            "Small angle: exp(X) ≈ I + X"


class TestMatrixLogSO4:
    """Tests for matrix_log_so4 (SO(4) → so(4))."""
    
    def test_identity_gives_zero(self):
        """log(I) = 0."""
        I = np.eye(4, dtype=DTYPE)
        result = matrix_log_so4(I, np)
        
        assert np.allclose(result, np.zeros((4, 4)), atol=1e-6), \
            "log(I) should be zero"
    
    def test_output_is_skew_symmetric(self):
        """log(R) is skew-symmetric for R ∈ SO(4)."""
        np.random.seed(42)
        for _ in range(10):
            R = random_so4(np)
            X = matrix_log_so4(R, np)
            
            skew_check = np.max(np.abs(X + X.T))
            assert skew_check < 1e-5, \
                f"log(R) should be skew-symmetric, got error {skew_check}"
    
    def test_roundtrip_exp_log(self):
        """exp(log(R)) = R."""
        np.random.seed(42)
        for _ in range(10):
            R = random_so4(np)
            X = matrix_log_so4(R, np)
            R_recovered = matrix_exp_skew4(X, np)
            
            assert np.allclose(R, R_recovered, atol=1e-4), \
                "exp(log(R)) should equal R"
    
    def test_roundtrip_log_exp(self):
        """log(exp(X)) = X for small X."""
        np.random.seed(42)
        for _ in range(10):
            X = random_skew_symmetric(np) * 0.5  # Not too large
            R = matrix_exp_skew4(X, np)
            X_recovered = matrix_log_so4(R, np)
            
            assert np.allclose(X, X_recovered, atol=1e-4), \
                "log(exp(X)) should equal X for small X"


class TestGeodesicInterpolation:
    """Tests for geodesic_interpolation_so4."""
    
    def test_endpoints(self):
        """γ(0) = A and γ(1) = B."""
        np.random.seed(42)
        A = random_so4(np)
        B = random_so4(np)
        
        gamma_0 = geodesic_interpolation_so4(A, B, 0.0, np)
        gamma_1 = geodesic_interpolation_so4(A, B, 1.0, np)
        
        assert np.allclose(gamma_0, A, atol=1e-5), \
            "γ(0) should equal A"
        assert np.allclose(gamma_1, B, atol=1e-4), \
            "γ(1) should equal B"
    
    def test_preserves_so4(self):
        """γ(t) ∈ SO(4) for all t ∈ [0, 1]."""
        np.random.seed(42)
        A = random_so4(np)
        B = random_so4(np)
        
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            gamma_t = geodesic_interpolation_so4(A, B, t, np)
            assert verify_so4(gamma_t, np, tol=1e-4), \
                f"γ({t}) should be SO(4)"
    
    def test_midpoint_is_unique(self):
        """Midpoint γ(0.5) is equidistant from A and B."""
        np.random.seed(42)
        A = random_so4(np)
        B = random_so4(np)
        
        midpoint = geodesic_interpolation_so4(A, B, 0.5, np)
        
        # Distance on SO(4) is ||log(A.T @ B)||_F
        dist_A = np.linalg.norm(matrix_log_so4(A.T @ midpoint, np), 'fro')
        dist_B = np.linalg.norm(matrix_log_so4(midpoint.T @ B, np), 'fro')
        
        # Distances should be equal (geodesic is equidistant)
        assert abs(dist_A - dist_B) < 0.1, \
            f"Midpoint should be equidistant: {dist_A} vs {dist_B}"
    
    def test_symmetric_interpolation(self):
        """γ_AB(t) and γ_BA(1-t) should be the same."""
        np.random.seed(42)
        A = random_so4(np)
        B = random_so4(np)
        
        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            gamma_AB = geodesic_interpolation_so4(A, B, t, np)
            gamma_BA = geodesic_interpolation_so4(B, A, 1.0 - t, np)
            
            assert np.allclose(gamma_AB, gamma_BA, atol=1e-3), \
                f"γ_AB({t}) should equal γ_BA({1-t})"


class TestContrastiveUpdateSO4:
    """Tests for contrastive_update_so4."""
    
    def test_preserves_so4(self):
        """Updated embeddings remain in SO(4)."""
        np.random.seed(42)
        A = random_so4(np)
        B = random_so4(np)
        
        for rate in [0.01, 0.1, PHI_INV**5]:
            new_A, new_B = contrastive_update_so4(A, B, rate, np)
            
            assert verify_so4(new_A, np, tol=1e-4), \
                f"Updated A should be SO(4) at rate {rate}"
            assert verify_so4(new_B, np, tol=1e-4), \
                f"Updated B should be SO(4) at rate {rate}"
    
    def test_pulls_together(self):
        """Updated embeddings are closer than original."""
        np.random.seed(42)
        A = random_so4(np)
        B = random_so4(np)
        
        # Distance before
        dist_before = np.linalg.norm(matrix_log_so4(A.T @ B, np), 'fro')
        
        # Update
        rate = 0.1
        new_A, new_B = contrastive_update_so4(A, B, rate, np)
        
        # Distance after
        dist_after = np.linalg.norm(matrix_log_so4(new_A.T @ new_B, np), 'fro')
        
        assert dist_after < dist_before, \
            f"Distance should decrease: {dist_before} -> {dist_after}"
    
    def test_zero_rate_is_identity(self):
        """rate=0 should not change embeddings."""
        np.random.seed(42)
        A = random_so4(np)
        B = random_so4(np)
        
        new_A, new_B = contrastive_update_so4(A, B, 0.0, np)
        
        assert np.allclose(new_A, A, atol=1e-6), \
            "rate=0 should not change A"
        assert np.allclose(new_B, B, atol=1e-6), \
            "rate=0 should not change B"


class TestWithRealEmbeddings:
    """Integration tests with actual rotor embeddings.
    
    NOTE: Rotor embeddings are in Spin(3,1), not SO(4).
    They satisfy RR̃ = 1 in Clifford sense, not R.T @ R = I.
    The SO(4) geodesic functions work on orthogonal matrices.
    """
    
    def test_rotor_embeddings_have_stable_norm(self):
        """Rotor embeddings have consistent Frobenius norm."""
        embeddings = initialize_embeddings_rotor(100, seed=42)
        
        norms = [np.linalg.norm(embeddings[i], 'fro') for i in range(10)]
        
        # All rotor embeddings should have similar norm (not varying wildly)
        assert max(norms) / min(norms) < 2.0, \
            f"Rotor norms vary too much: {norms}"
    
    def test_contrastive_on_orthogonal_embeddings(self):
        """Contrastive update on orthogonalized embeddings preserves SO(4)."""
        embeddings = initialize_embeddings_rotor(100, seed=42)
        
        # Orthogonalize via SVD (polar decomposition)
        def orthogonalize(M):
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] = -U[:, -1]
                R = U @ Vt
            return R.astype(M.dtype)
        
        A = orthogonalize(embeddings[0])
        B = orthogonalize(embeddings[1])
        
        # Verify inputs are SO(4)
        assert verify_so4(A, np, tol=1e-4), "Input A should be SO(4)"
        assert verify_so4(B, np, tol=1e-4), "Input B should be SO(4)"
        
        new_A, new_B = contrastive_update_so4(A, B, PHI_INV**5, np)
        
        assert verify_so4(new_A, np, tol=1e-4), \
            "Updated A should remain SO(4)"
        assert verify_so4(new_B, np, tol=1e-4), \
            "Updated B should remain SO(4)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
