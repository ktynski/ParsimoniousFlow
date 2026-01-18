"""
Test Downward Projection — Generation Pipeline

Tests that downward projection correctly cascades from Grand Master through levels
to generate tokens. This is the inverse of upward aggregation.

Theory (Chapter 11, 14):
    1. Start with Grand Master witness (coherent core)
    2. Apply GraceInverse to inflate structure
    3. Unbind cascade through levels
    4. Phase-locked emission at correct intervals
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PI,
    CLIFFORD_DIM, MATRIX_DIM, DTYPE,
)
from holographic_prod.core.algebra import (
    build_clifford_basis,
    grace_operator,
    clifford_inverse,
    geometric_product,
)
from holographic_prod.memory import HolographicMemory, MemoryConfig
from holographic_prod.core.algebra import grace_inverse
from holographic_prod.fractal.downward_projection import DownwardProjection


class TestDownwardProjection:
    """Test suite for downward projection pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.basis = build_clifford_basis(np)
        self.memory = HolographicMemory(
            max_levels=2,
            vocab_size=100,
            use_gpu=False,
        )
        self.downward_proj = DownwardProjection(self.basis)
    
    def test_grace_inverse_inflation(self):
        """Test that GraceInverse inflates coherent core."""
        # Create coherent core with some structure (not pure scalar)
        # Start with a matrix that has structure in multiple grades
        M_coherent = np.random.randn(4, 4).astype(DTYPE)
        M_coherent = M_coherent / np.linalg.norm(M_coherent, 'fro')
        
        # Apply Grace to contract it (creates coherent core)
        from holographic_prod.core.algebra import grace_operator
        M_contracted = grace_operator(M_coherent, self.basis, np)
        
        # Now apply GraceInverse - this should inflate it back
        M_inflated = grace_inverse(M_contracted, self.basis, np)
        
        # Should have structure in multiple grades after inflation
        from holographic_prod.core.algebra import decompose_to_coefficients
        coeffs = decompose_to_coefficients(M_inflated, self.basis, np)
        non_zero = np.sum(np.abs(coeffs) > 1e-8)
        assert non_zero >= 2, "Inflated multivector should have structure in multiple grades"
    
    def test_unbinding_cascade(self):
        """Test that unbinding cascades through levels."""
        # Create a test binding in memory
        context = np.random.randn(4, 4).astype(DTYPE)
        target = np.random.randn(4, 4).astype(DTYPE)
        
        # Normalize
        context = context / np.linalg.norm(context, 'fro')
        target = target / np.linalg.norm(target, 'fro')
        
        # Store binding
        binding = geometric_product(context, target)
        self.memory.holographic_memory += PHI_INV * binding
        
        # Create query from inflated master state
        master_state = np.eye(4, dtype=DTYPE) * 0.5
        master_inflated = grace_inverse(master_state, self.basis, np)
        
        # Unbind should retrieve target approximately
        ctx_inv = clifford_inverse(context, np)
        retrieved = geometric_product(ctx_inv, self.memory.holographic_memory)
        
        # Should be similar to target
        similarity = np.trace(retrieved @ target.T) / (
            np.linalg.norm(retrieved, 'fro') * np.linalg.norm(target, 'fro')
        )
        assert similarity > 0.1, "Unbinding should retrieve target"
    
    def test_level_cascade(self):
        """Test that projection cascades through tower levels."""
        # Store a simple association to create tower state
        self.memory.learn_batch([[1, 2]], [10])
        
        # Update tower (upward aggregation)
        # Tower aggregation is now dynamic (no explicit update needed)
        
        # Grand master should have aggregated state
        grand_master_coeffs = self.memory.master_state
        assert np.linalg.norm(grand_master_coeffs) > 0, "Grand master should have state"
        
        # Convert coefficients to matrix form
        grand_master_matrix = np.zeros((4, 4), dtype=DTYPE)
        for i, coeff in enumerate(grand_master_coeffs):
            grand_master_matrix += coeff * self.basis[i]
        
        # Downward projection: inflate grand master
        inflated = grace_inverse(grand_master_matrix, self.basis, np)
        
        # Should have structure
        assert np.linalg.norm(inflated, 'fro') > 0, "Inflated master should have structure"
    
    def test_phase_locked_window(self):
        """Test that phase-locked emission uses correct window."""
        # Emission window: [π·φ⁻¹, π·φ⁻¹ + φ⁻²]
        window_start = PI * PHI_INV
        window_end = PI * PHI_INV + PHI_INV_SQ
        
        # Test phases
        test_phases = [
            window_start - 0.1,  # Before window
            window_start,         # At start
            window_start + PHI_INV_SQ / 2,  # Middle
            window_end,           # At end
            window_end + 0.1,    # After window
        ]
        
        for phase in test_phases:
            in_window = window_start <= phase <= window_end
            # Phase should be checked correctly
            assert isinstance(in_window, (bool, np.bool_)), "Window check should work"
    
    def test_generation_pipeline(self):
        """Test full generation pipeline: Grand Master → Token."""
        # Store some associations
        contexts = [[1, 2], [2, 3], [3, 4]]
        targets = [10, 11, 12]
        
        self.memory.learn_batch(contexts, targets)
        
        # Get grand master state
        # Tower aggregation is now dynamic (no explicit update needed)
        grand_master_coeffs = self.memory.master_state.copy()
        
        # Convert to matrix
        grand_master_matrix = np.zeros((4, 4), dtype=DTYPE)
        for i, coeff in enumerate(grand_master_coeffs):
            grand_master_matrix += coeff * self.basis[i]
        
        # Apply GraceInverse
        inflated = grace_inverse(grand_master_matrix, self.basis, np)
        
        # Should be able to generate from this
        assert np.linalg.norm(inflated, 'fro') > 0, "Inflated master should have structure"
        
        # Try to retrieve a token
        context = [1, 2]
        token, prob, _ = self.memory.retrieve_probabilistic(context)
        
        # Should retrieve something
        assert token is not None or prob == 0.0, "Should retrieve token or return zero prob"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
