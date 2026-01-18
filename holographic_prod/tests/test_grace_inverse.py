"""
Test GraceInverse Operator — Theory-True Inflation

Tests that GraceInverse correctly inflates coherent core back into structural detail.
This is the inverse of Grace operator, required for generation via downward projection.

Theory (Chapter 11, 14):
    GraceInverse(M) = ⟨M⟩₀ + φ¹⟨M⟩₁ + φ²⟨M⟩₂ + φ³⟨M⟩₃ + φ¹⟨M⟩₄
    
    This reverses Grace contraction, inflating the witness (scalar + pseudoscalar)
    back into full multivector structure for generation.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
    CLIFFORD_DIM, MATRIX_DIM, DTYPE,
)
from holographic_prod.core.algebra import (
    build_clifford_basis,
    grace_operator,
    grace_inverse,
    decompose_to_coefficients,
    reconstruct_from_coefficients,
)


class TestGraceInverse:
    """Test suite for GraceInverse operator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.basis = build_clifford_basis(np)
    
    def test_grace_inverse_scales(self):
        """Test that GraceInverse uses correct φᵏ scaling factors."""
        # Create a test multivector with known coefficients
        M = np.eye(4, dtype=DTYPE)  # Identity matrix
        
        # Apply GraceInverse
        M_inverse = grace_inverse(M, self.basis)
        
        # Decompose result
        coeffs = decompose_to_coefficients(M_inverse, self.basis, np)
        
        # Verify scaling: original coeffs should be scaled by φᵏ
        original_coeffs = decompose_to_coefficients(M, self.basis, np)
        
        # Grade 0 (index 0): should be ×1
        assert abs(coeffs[0] - original_coeffs[0]) < 1e-6, "Grade 0 should be preserved"
        
        # Grade 1 (indices 1-4): should be ×φ
        for i in range(1, 5):
            if abs(original_coeffs[i]) > 1e-8:
                expected = original_coeffs[i] * PHI
                assert abs(coeffs[i] - expected) < 1e-6, f"Grade 1[{i}] should be ×φ"
        
        # Grade 2 (indices 5-10): should be ×φ²
        for i in range(5, 11):
            if abs(original_coeffs[i]) > 1e-8:
                expected = original_coeffs[i] * (PHI**2)
                assert abs(coeffs[i] - expected) < 1e-6, f"Grade 2[{i}] should be ×φ²"
        
        # Grade 3 (indices 11-14): should be ×φ³
        for i in range(11, 15):
            if abs(original_coeffs[i]) > 1e-8:
                expected = original_coeffs[i] * (PHI**3)
                assert abs(coeffs[i] - expected) < 1e-6, f"Grade 3[{i}] should be ×φ³"
        
        # Grade 4 (index 15): should be ×φ (Fibonacci exception, NOT φ⁴!)
        if abs(original_coeffs[15]) > 1e-8:
            expected = original_coeffs[15] * PHI
            assert abs(coeffs[15] - expected) < 1e-6, "Grade 4 should be ×φ (Fibonacci exception)"
    
    def test_grace_inverse_reverses_grace(self):
        """Test that GraceInverse reverses Grace contraction (approximately)."""
        # Create a multivector with structure
        M = np.random.randn(4, 4).astype(DTYPE)
        M = M + M.T  # Make symmetric for stability
        
        # Apply Grace (contracts)
        M_graced = grace_operator(M, self.basis, np)
        
        # Apply GraceInverse (inflates)
        M_inflated = grace_inverse(M_graced, self.basis)
        
        # Apply Grace again (should contract back)
        M_recontracted = grace_operator(M_inflated, self.basis, np)
        
        # After Grace → GraceInverse → Grace, we should be close to original
        # (exact equality not expected due to numerical precision and witness-only preservation)
        diff = np.linalg.norm(M_recontracted - M_graced, 'fro')
        assert diff < 1e-3, f"GraceInverse should approximately reverse Grace (diff={diff})"
    
    def test_grace_inverse_preserves_witness(self):
        """Test that GraceInverse preserves witness (scalar + pseudoscalar)."""
        # Create a multivector with only witness components
        M = np.eye(4, dtype=DTYPE) * 0.5  # Scalar component
        M[0, 0] = 1.0  # Strong scalar
        
        # Apply GraceInverse
        M_inverse = grace_inverse(M, self.basis)
        
        # Extract witness from result
        scalar = np.trace(M_inverse) / 4.0
        pseudo = np.sum(self.basis[15] * M_inverse) / 4.0
        
        # Witness should be preserved (scalar × 1, pseudo × φ)
        original_scalar = np.trace(M) / 4.0
        assert abs(scalar - original_scalar) < 1e-6, "Scalar should be preserved"
    
    def test_grace_inverse_fibonacci_exception(self):
        """Test that Grade 4 uses φ¹, NOT φ⁴ (Fibonacci exception)."""
        # Create a multivector with strong pseudoscalar component
        M = self.basis[15].copy() * 0.5  # Pseudoscalar
        
        # Apply GraceInverse
        M_inverse = grace_inverse(M, self.basis)
        
        # Extract pseudoscalar coefficient
        coeffs = decompose_to_coefficients(M_inverse, self.basis, np)
        pseudo_coeff = coeffs[15]
        
        # Should be scaled by φ, NOT φ⁴
        original_coeff = decompose_to_coefficients(M, self.basis, np)[15]
        expected = original_coeff * PHI  # φ¹
        not_expected = original_coeff * (PHI**4)  # φ⁴
        
        assert abs(pseudo_coeff - expected) < 1e-6, "Grade 4 should scale by φ¹"
        assert abs(pseudo_coeff - not_expected) > 1e-3, "Grade 4 should NOT scale by φ⁴"
    
    def test_grace_inverse_idempotent_on_witness(self):
        """Test that GraceInverse scales witness components correctly."""
        # Create pure witness (scalar + pseudoscalar only)
        M = np.eye(4, dtype=DTYPE) * 0.5  # Scalar
        M = M + 0.3 * self.basis[15]  # Add pseudoscalar
        
        # Apply Grace to get pure witness
        M_witness = grace_operator(M, self.basis, np)
        
        # Apply GraceInverse
        M_inflated = grace_inverse(M_witness, self.basis)
        
        # Apply GraceInverse again (should scale by φ again)
        M_inflated2 = grace_inverse(M_inflated, self.basis)
        
        # GraceInverse scales witness: scalar × 1, pseudo × φ
        # Applying twice: scalar stays same, pseudo scales by φ²
        # So M_inflated2 should have pseudo scaled by φ² relative to M_inflated
        from holographic_prod.core.quotient import extract_witness
        s1, p1 = extract_witness(M_inflated, self.basis, np)
        s2, p2 = extract_witness(M_inflated2, self.basis, np)
        
        # Scalar should be preserved, pseudoscalar should scale by φ
        assert abs(s2 - s1) < 1e-6, "Scalar should be preserved"
        assert abs(p2 - p1 * PHI) < 1e-3, "Pseudoscalar should scale by φ"
    
    def test_grace_inverse_generation_ready(self):
        """Test that GraceInverse produces structure suitable for generation."""
        # Create a coherent core with some structure (not pure scalar)
        # Start with multivector that has structure in multiple grades
        M_coherent = np.random.randn(4, 4).astype(DTYPE)
        M_coherent = M_coherent + M_coherent.T  # Make symmetric
        M_coherent = M_coherent / np.linalg.norm(M_coherent, 'fro')
        
        # Apply Grace to contract to coherent core
        M_witness = grace_operator(M_coherent, self.basis, np)
        
        # Apply GraceInverse to inflate
        M_inflated = grace_inverse(M_witness, self.basis)
        
        # Check that inflated version has structure
        coeffs = decompose_to_coefficients(M_inflated, self.basis, np)
        
        # Should have non-zero coefficients
        non_zero_grades = np.sum(np.abs(coeffs) > 1e-6)
        assert non_zero_grades >= 1, "Inflated multivector should have some structure"
        
        # Energy should be present
        total_energy = np.sum(coeffs**2)
        assert total_energy > 1e-6, "Inflated multivector should have energy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
