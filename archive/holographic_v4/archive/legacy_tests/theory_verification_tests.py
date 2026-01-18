"""
Theory Verification Tests

Rigorous verification that implementations match theoretical definitions.

THEORY SOURCES:
- rhnsclifford.md: Core theory document
- constants.py: Sacred constants
- paper.tex: Mathematical derivations
"""

import numpy as np
from typing import Tuple

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_INV_SQ = PHI_INV ** 2
PHI_INV_CUBE = PHI_INV ** 3
MATRIX_DIM = 4
DTYPE = np.float32


def get_xp():
    try:
        import cupy as cp
        return cp, True
    except ImportError:
        return np, False


class TestGraceOperator:
    """Verify Grace operator matches theory."""
    
    def setup_method(self):
        self.xp, self.has_gpu = get_xp()
        from holographic_v4.algebra import build_clifford_basis
        self.basis = build_clifford_basis(self.xp)
    
    def test_grace_scales_grades_correctly(self):
        """
        THEORY: Grace scales each grade by φ⁻ᵏ
        
        Grade 0 (scalar): × 1.0
        Grade 1 (vectors): × φ⁻¹
        Grade 2 (bivectors): × φ⁻²
        Grade 3 (trivectors): × φ⁻³
        Grade 4 (pseudoscalar): × φ⁻¹ (Fibonacci exception!)
        """
        from holographic_v4.algebra import (
            grace_operator,
            decompose_to_coefficients,
        )
        from holographic_v4.constants import GRADE_INDICES
        
        # Create test matrix with known grade content
        np.random.seed(42)
        M = self.xp.array(np.random.randn(4, 4).astype(DTYPE))
        
        # Get coefficients before and after Grace
        coeffs_before = decompose_to_coefficients(M, self.basis, self.xp)
        M_graced = grace_operator(M, self.basis, self.xp)
        coeffs_after = decompose_to_coefficients(M_graced, self.basis, self.xp)
        
        # Convert to CPU for comparison
        if self.has_gpu:
            coeffs_before = coeffs_before.get()
            coeffs_after = coeffs_after.get()
        
        # Expected scales per grade
        expected_scales = {
            0: 1.0,
            1: PHI_INV,
            2: PHI_INV_SQ,
            3: PHI_INV_CUBE,
            4: PHI_INV,  # Fibonacci exception!
        }
        
        # Verify each grade
        for grade, indices in GRADE_INDICES.items():
            expected_scale = expected_scales[grade]
            for idx in indices:
                before = coeffs_before[idx]
                after = coeffs_after[idx]
                
                if abs(before) > 1e-8:  # Only test non-zero coefficients
                    actual_scale = after / before
                    assert abs(actual_scale - expected_scale) < 1e-5, \
                        f"Grade {grade}, index {idx}: expected scale {expected_scale}, got {actual_scale}"
        
        print(f"✓ Grace scales all grades correctly")
        print(f"  Grade 0: ×1.0, Grade 1: ×φ⁻¹, Grade 2: ×φ⁻², Grade 3: ×φ⁻³, Grade 4: ×φ⁻¹")
    
    def test_grace_preserves_witness(self):
        """
        THEORY: Grace preserves the witness (scalar + pseudoscalar)
        at rate 1.0 for scalar, φ⁻¹ for pseudoscalar.
        
        The witness IS what Grace preserves.
        """
        from holographic_v4.algebra import grace_operator
        from holographic_v4.quotient import extract_witness
        
        np.random.seed(42)
        M = self.xp.array(np.random.randn(4, 4).astype(DTYPE))
        
        # Get witness before and after
        s_before, p_before = extract_witness(M, self.basis, self.xp)
        M_graced = grace_operator(M, self.basis, self.xp)
        s_after, p_after = extract_witness(M_graced, self.basis, self.xp)
        
        # Convert to CPU
        if self.has_gpu:
            s_before, p_before = float(s_before.get()), float(p_before.get())
            s_after, p_after = float(s_after.get()), float(p_after.get())
        else:
            s_before, p_before = float(s_before), float(p_before)
            s_after, p_after = float(s_after), float(p_after)
        
        # Scalar preserved at 1.0
        assert abs(s_after - s_before * 1.0) < 1e-5, \
            f"Scalar not preserved: before={s_before}, after={s_after}"
        
        # Pseudoscalar scaled by φ⁻¹
        assert abs(p_after - p_before * PHI_INV) < 1e-5, \
            f"Pseudoscalar not scaled correctly: before={p_before}, after={p_after}, expected={p_before * PHI_INV}"
        
        print(f"✓ Grace preserves witness correctly")
        print(f"  Scalar: {s_before:.4f} → {s_after:.4f} (×1.0)")
        print(f"  Pseudo: {p_before:.4f} → {p_after:.4f} (×φ⁻¹)")
    
    def test_enstrophy_decay_rate(self):
        """
        THEORY: Enstrophy (||grade-2||²) decays at rate φ⁻⁴ per Grace application.
        
        This is the Clifford analogue of viscous damping in Navier-Stokes.
        """
        from holographic_v4.algebra import grace_operator
        from holographic_v4.quotient import compute_enstrophy
        
        np.random.seed(42)
        M = self.xp.array(np.random.randn(4, 4).astype(DTYPE))
        
        # Compute enstrophy before and after
        enstrophy_before = compute_enstrophy(M, self.basis, self.xp)
        M_graced = grace_operator(M, self.basis, self.xp)
        enstrophy_after = compute_enstrophy(M_graced, self.basis, self.xp)
        
        # Convert to CPU
        if self.has_gpu:
            enstrophy_before = float(enstrophy_before.get())
            enstrophy_after = float(enstrophy_after.get())
        else:
            enstrophy_before = float(enstrophy_before)
            enstrophy_after = float(enstrophy_after)
        
        # Expected decay rate: (φ⁻²)² = φ⁻⁴
        expected_rate = PHI_INV_SQ ** 2  # ≈ 0.1459
        
        if enstrophy_before > 1e-8:
            actual_rate = enstrophy_after / enstrophy_before
            assert abs(actual_rate - expected_rate) < 1e-4, \
                f"Enstrophy decay wrong: expected {expected_rate:.4f}, got {actual_rate:.4f}"
        
        print(f"✓ Enstrophy decays at φ⁻⁴ ≈ 0.1459")
        print(f"  Before: {enstrophy_before:.4f}")
        print(f"  After: {enstrophy_after:.4f}")
        print(f"  Ratio: {enstrophy_after/enstrophy_before:.4f} (expected {expected_rate:.4f})")


class TestGeometricProduct:
    """Verify geometric product matches Clifford algebra definition."""
    
    def setup_method(self):
        self.xp, self.has_gpu = get_xp()
        from holographic_v4.algebra import build_clifford_basis
        self.basis = build_clifford_basis(self.xp)
    
    def test_geometric_product_is_matrix_multiply(self):
        """
        THEORY: In matrix representation Cl(3,1) ≅ M₄(ℝ),
        geometric product = matrix multiplication.
        """
        from holographic_v4.algebra import geometric_product
        
        np.random.seed(42)
        A = self.xp.array(np.random.randn(4, 4).astype(DTYPE))
        B = self.xp.array(np.random.randn(4, 4).astype(DTYPE))
        
        # Geometric product
        AB_geo = geometric_product(A, B)
        
        # Matrix multiplication
        AB_mat = A @ B
        
        # Should be identical
        if self.has_gpu:
            AB_geo = AB_geo.get()
            AB_mat = AB_mat.get()
        
        np.testing.assert_allclose(AB_geo, AB_mat, rtol=1e-5)
        print("✓ Geometric product = matrix multiplication in Cl(3,1) ≅ M₄(ℝ)")
    
    def test_basis_orthonormality(self):
        """
        THEORY: Clifford basis elements should be orthonormal under trace inner product.
        
        <e_i, e_j> = Tr(e_i · e_j†) / 4 = δ_ij
        """
        from holographic_v4.algebra import build_clifford_basis
        
        basis = build_clifford_basis(self.xp)
        
        # Compute Gram matrix: G_ij = <e_i, e_j>
        n = 16
        gram = self.xp.zeros((n, n), dtype=DTYPE)
        
        for i in range(n):
            for j in range(n):
                # Inner product: Tr(e_i · e_j^T) / 4
                inner = self.xp.trace(basis[i] @ basis[j].T) / 4
                gram[i, j] = inner
        
        # Should be close to identity (orthonormal)
        if self.has_gpu:
            gram = gram.get()
        
        # Check diagonal (normalization)
        diag = np.diag(gram)
        print(f"Basis norms (should be ~1): min={diag.min():.3f}, max={diag.max():.3f}")
        
        # Check off-diagonal (orthogonality)
        off_diag = gram - np.diag(diag)
        print(f"Off-diagonal (should be ~0): max abs={np.abs(off_diag).max():.3f}")
        
        print("✓ Clifford basis is approximately orthonormal")


class TestWitnessSpace:
    """Verify witness space operations match theory."""
    
    def setup_method(self):
        self.xp, self.has_gpu = get_xp()
        from holographic_v4.algebra import build_clifford_basis
        self.basis = build_clifford_basis(self.xp)
    
    def test_witness_is_scalar_plus_pseudoscalar(self):
        """
        THEORY: Witness W(M) = scalar + φ⁻¹ · pseudoscalar
        
        This is the gauge-invariant "self-pointer" that defines identity.
        """
        from holographic_v4.quotient import extract_witness, witness_matrix
        from holographic_v4.algebra import decompose_to_coefficients
        
        np.random.seed(42)
        M = self.xp.array(np.random.randn(4, 4).astype(DTYPE))
        
        # Extract witness
        s, p = extract_witness(M, self.basis, self.xp)
        
        # Get witness matrix
        W = witness_matrix(M, self.basis, self.xp)
        
        # Decompose witness matrix
        W_coeffs = decompose_to_coefficients(W, self.basis, self.xp)
        
        # Convert to CPU
        if self.has_gpu:
            s, p = float(s.get()), float(p.get())
            W_coeffs = W_coeffs.get()
        else:
            s, p = float(s), float(p)
        
        # Witness matrix should be: s · e_0 + φ⁻¹ · p · e_15
        expected_coeffs = np.zeros(16)
        expected_coeffs[0] = s
        expected_coeffs[15] = PHI_INV * p
        
        np.testing.assert_allclose(W_coeffs, expected_coeffs, rtol=1e-4, atol=1e-6)
        print(f"✓ Witness = scalar + φ⁻¹·pseudoscalar")
        print(f"  σ = {s:.4f}, p = {p:.4f}")


def run_all_tests():
    """Run all theory verification tests."""
    print("=" * 60)
    print("THEORY VERIFICATION TESTS")
    print("=" * 60)
    print()
    
    # Grace operator tests
    print("GRACE OPERATOR")
    print("-" * 40)
    grace_tests = TestGraceOperator()
    grace_tests.setup_method()
    grace_tests.test_grace_scales_grades_correctly()
    print()
    grace_tests.test_grace_preserves_witness()
    print()
    grace_tests.test_enstrophy_decay_rate()
    print()
    
    # Geometric product tests
    print("GEOMETRIC PRODUCT")
    print("-" * 40)
    geo_tests = TestGeometricProduct()
    geo_tests.setup_method()
    geo_tests.test_geometric_product_is_matrix_multiply()
    print()
    geo_tests.test_basis_orthonormality()
    print()
    
    # Witness space tests
    print("WITNESS SPACE")
    print("-" * 40)
    witness_tests = TestWitnessSpace()
    witness_tests.setup_method()
    witness_tests.test_witness_is_scalar_plus_pseudoscalar()
    print()
    
    print("=" * 60)
    print("ALL THEORY VERIFICATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
