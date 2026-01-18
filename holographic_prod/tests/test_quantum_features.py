"""
Tests for Quantum-Inspired Features (v5.27.0)
=============================================

Tests for:
1. Chirality extraction and guidance
2. Witness entanglement and propagation

These features implement quantum parsimonies that physical brains cannot
exploit due to decoherence, but our digital implementation can maintain.
"""

import numpy as np
import pytest
from typing import List, Tuple

from holographic_prod.core.constants import (
    PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, DTYPE,
)
from holographic_prod.core.algebra import build_clifford_basis
from holographic_prod.core.quotient import (
    extract_witness,
    extract_chirality,
    extract_chirality_batch,
    extract_chirality_strength,
    chirality_match_scores,
)
from holographic_prod.memory.witness_index import (
    WitnessIndex,
    propagate_witness_update,
    batch_register_witnesses,
)


# =============================================================================
# CHIRALITY TESTS
# =============================================================================

class TestChiralityExtraction:
    """Test chirality extraction functions."""
    
    @pytest.fixture
    def basis(self):
        return build_clifford_basis(np)
    
    def test_extract_chirality_positive(self, basis):
        """Test that positive pseudoscalar gives +1 chirality."""
        # Create matrix with positive pseudoscalar
        M = np.eye(4, dtype=DTYPE)
        M += 0.5 * basis[15]  # Add positive pseudoscalar
        
        chirality = extract_chirality(M, basis, np)
        assert chirality == 1, "Positive pseudoscalar should give +1 chirality"
    
    def test_extract_chirality_negative(self, basis):
        """Test that negative pseudoscalar gives -1 chirality."""
        # Create matrix with negative pseudoscalar
        M = np.eye(4, dtype=DTYPE)
        M -= 0.5 * basis[15]  # Add negative pseudoscalar
        
        chirality = extract_chirality(M, basis, np)
        assert chirality == -1, "Negative pseudoscalar should give -1 chirality"
    
    def test_extract_chirality_batch(self, basis):
        """Test batch chirality extraction."""
        # Create batch with mixed chiralities
        batch = np.stack([
            np.eye(4) + 0.5 * basis[15],   # Positive
            np.eye(4) - 0.5 * basis[15],   # Negative
            np.eye(4) + 0.1 * basis[15],   # Weak positive
            np.eye(4) - 0.1 * basis[15],   # Weak negative
        ], axis=0).astype(DTYPE)
        
        chiralities = extract_chirality_batch(batch, basis, np)
        
        assert chiralities[0] == 1, "First should be positive"
        assert chiralities[1] == -1, "Second should be negative"
        assert chiralities[2] == 1, "Third should be positive"
        assert chiralities[3] == -1, "Fourth should be negative"
    
    def test_extract_chirality_strength(self, basis):
        """Test chirality strength extraction."""
        # Strong chirality
        M_strong = np.eye(4, dtype=DTYPE) + 0.8 * basis[15]
        strength_strong = extract_chirality_strength(M_strong, basis, np)
        
        # Weak chirality
        M_weak = np.eye(4, dtype=DTYPE) + 0.01 * basis[15]
        strength_weak = extract_chirality_strength(M_weak, basis, np)
        
        assert strength_strong > strength_weak, "Strong should have higher strength"
        assert strength_strong > PHI_INV_CUBE, "Strong should exceed threshold"
    
    def test_chirality_match_scores_matching(self, basis):
        """Test that matching chirality gives score 1.0."""
        # Create candidates with positive chirality
        candidates = np.stack([
            np.eye(4) + 0.5 * basis[15],
            np.eye(4) + 0.3 * basis[15],
        ], axis=0).astype(DTYPE)
        
        # Context with positive chirality
        ctx_chirality = 1
        ctx_strength = 0.5  # Strong enough to constrain
        
        scores = chirality_match_scores(
            ctx_chirality, ctx_strength, candidates, basis, np
        )
        
        # All should be 1.0 (matching)
        assert np.allclose(scores, 1.0), "Matching chirality should give 1.0"
    
    def test_chirality_match_scores_mismatching(self, basis):
        """Test that mismatching chirality gives suppressed score."""
        # Create candidates with negative chirality
        candidates = np.stack([
            np.eye(4) - 0.5 * basis[15],
            np.eye(4) - 0.3 * basis[15],
        ], axis=0).astype(DTYPE)
        
        # Context with positive chirality
        ctx_chirality = 1
        ctx_strength = 0.5  # Strong enough to constrain
        
        scores = chirality_match_scores(
            ctx_chirality, ctx_strength, candidates, basis, np
        )
        
        # All should be suppressed (< 1.0)
        assert np.all(scores < 1.0), "Mismatching chirality should be suppressed"
    
    def test_chirality_match_scores_weak_context(self, basis):
        """Test that weak context chirality doesn't constrain."""
        # Create candidates with mixed chirality
        candidates = np.stack([
            np.eye(4) + 0.5 * basis[15],  # Positive
            np.eye(4) - 0.5 * basis[15],  # Negative
        ], axis=0).astype(DTYPE)
        
        # Context with weak chirality (below threshold)
        ctx_chirality = 1
        ctx_strength = 0.001  # Too weak to constrain
        
        scores = chirality_match_scores(
            ctx_chirality, ctx_strength, candidates, basis, np
        )
        
        # All should be 1.0 (no constraint)
        assert np.allclose(scores, 1.0), "Weak context should not constrain"


# =============================================================================
# WITNESS ENTANGLEMENT TESTS
# =============================================================================

class TestWitnessIndex:
    """Test WitnessIndex class."""
    
    @pytest.fixture
    def basis(self):
        return build_clifford_basis(np)
    
    @pytest.fixture
    def witness_index(self):
        return WitnessIndex(resolution=2)
    
    def test_hash_witness(self, witness_index):
        """Test witness hashing."""
        key1 = witness_index.hash_witness(0.123, 0.456)
        key2 = witness_index.hash_witness(0.124, 0.457)  # Close
        key3 = witness_index.hash_witness(0.5, 0.8)      # Different
        
        # Close values should hash to same bucket (resolution=2)
        assert key1 == key2, "Close witnesses should hash to same bucket"
        assert key1 != key3, "Different witnesses should hash differently"
    
    def test_register_and_lookup(self, witness_index):
        """Test registering and looking up locations."""
        # Register some locations
        witness_index.register(0.5, 0.3, level=0, satellite_idx=5)
        witness_index.register(0.5, 0.3, level=0, satellite_idx=10)  # Same witness
        witness_index.register(0.8, 0.1, level=0, satellite_idx=15)  # Different
        
        # Lookup entangled locations
        entangled = witness_index.get_entangled(0.5, 0.3)
        
        assert len(entangled) == 2, "Should find 2 entangled locations"
        assert (0, 5) in entangled
        assert (0, 10) in entangled
        assert (0, 15) not in entangled
    
    def test_exclude_primary(self, witness_index):
        """Test excluding primary location from lookup."""
        witness_index.register(0.5, 0.3, level=0, satellite_idx=5)
        witness_index.register(0.5, 0.3, level=0, satellite_idx=10)
        
        # Lookup excluding primary
        entangled = witness_index.get_entangled(0.5, 0.3, exclude=(0, 5))
        
        assert len(entangled) == 1, "Should exclude primary"
        assert (0, 10) in entangled
        assert (0, 5) not in entangled
    
    def test_unregister(self, witness_index):
        """Test unregistering locations."""
        witness_index.register(0.5, 0.3, level=0, satellite_idx=5)
        witness_index.register(0.5, 0.3, level=0, satellite_idx=10)
        
        # Unregister one
        result = witness_index.unregister(0.5, 0.3, level=0, satellite_idx=5)
        
        assert result is True, "Should return True on successful unregister"
        
        entangled = witness_index.get_entangled(0.5, 0.3)
        assert len(entangled) == 1
        assert (0, 10) in entangled
    
    def test_register_from_matrix(self, witness_index, basis):
        """Test registering from matrix."""
        # Create matrix with known witness
        M = np.eye(4, dtype=DTYPE)
        M += 0.5 * basis[0]   # Add scalar
        M += 0.3 * basis[15]  # Add pseudoscalar
        
        n_entangled = witness_index.register_from_matrix(
            M, basis, level=0, satellite_idx=5, xp=np
        )
        
        assert n_entangled == 1, "First registration should have 1 entangled"
        
        # Register another with same witness
        n_entangled = witness_index.register_from_matrix(
            M, basis, level=0, satellite_idx=10, xp=np
        )
        
        assert n_entangled == 2, "Second registration should have 2 entangled"
    
    def test_statistics(self, witness_index):
        """Test statistics tracking."""
        witness_index.register(0.5, 0.3, level=0, satellite_idx=5)
        witness_index.register(0.5, 0.3, level=0, satellite_idx=10)
        witness_index.register(0.8, 0.1, level=0, satellite_idx=15)
        
        witness_index.get_entangled(0.5, 0.3)
        witness_index.get_entangled(0.8, 0.1)
        
        stats = witness_index.get_statistics()
        
        assert stats['n_buckets'] == 2, "Should have 2 buckets"
        assert stats['n_locations'] == 3, "Should have 3 locations"
        assert stats['total_registrations'] == 3
        assert stats['total_lookups'] == 2


class TestBatchRegisterWitnesses:
    """Test batch witness registration."""
    
    @pytest.fixture
    def basis(self):
        return build_clifford_basis(np)
    
    def test_batch_register(self, basis):
        """Test batch registration of witnesses."""
        witness_index = WitnessIndex(resolution=2)
        
        # Create batch of matrices with varying witnesses
        matrices = np.stack([
            np.eye(4) + 0.5 * basis[0] + 0.3 * basis[15],
            np.eye(4) + 0.5 * basis[0] + 0.3 * basis[15],  # Same witness
            np.eye(4) + 0.8 * basis[0] + 0.1 * basis[15],  # Different
        ], axis=0).astype(DTYPE)
        
        n_registered = batch_register_witnesses(
            witness_index, matrices, basis,
            level=0, start_satellite_idx=0, xp=np
        )
        
        assert n_registered == 3, "Should register all 3"
        
        stats = witness_index.get_statistics()
        assert stats['n_buckets'] == 2, "Should have 2 unique buckets"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestChiralityGuidedGeneration:
    """Integration tests for chirality-guided generation."""
    
    @pytest.fixture
    def basis(self):
        return build_clifford_basis(np)
    
    def test_chirality_preserved_through_composition(self, basis):
        """Test that chirality is preserved through geometric product."""
        # Create two matrices with same chirality
        M1 = np.eye(4, dtype=DTYPE) + 0.5 * basis[15]
        M2 = np.eye(4, dtype=DTYPE) + 0.3 * basis[15]
        
        # Compose them
        composed = M1 @ M2
        
        # Check chirality is preserved (both positive → positive)
        chi1 = extract_chirality(M1, basis, np)
        chi2 = extract_chirality(M2, basis, np)
        chi_composed = extract_chirality(composed, basis, np)
        
        # Note: chirality multiplication follows sign rules
        # (+) × (+) = (+), (+) × (-) = (-), etc.
        expected_chi = chi1 * chi2
        assert chi_composed == expected_chi, "Chirality should follow multiplication rules"


class TestWitnessEntanglementIntegration:
    """Integration tests for witness entanglement."""
    
    @pytest.fixture
    def basis(self):
        return build_clifford_basis(np)
    
    def test_entanglement_count_grows(self, basis):
        """Test that entanglement count grows with identical witnesses."""
        witness_index = WitnessIndex(resolution=2)
        
        # Create matrices with IDENTICAL witnesses (to ensure same bucket)
        base_witness = (0.5, 0.3)
        
        for i in range(10):
            # Use exact same witness values to guarantee same bucket
            witness_index.register(base_witness[0], base_witness[1], level=0, satellite_idx=i)
        
        # All should be entangled (same witness)
        entangled = witness_index.get_entangled(base_witness[0], base_witness[1])
        assert len(entangled) == 10, "All identical witnesses should be entangled"
    
    def test_entanglement_isolation(self, basis):
        """Test that different witnesses are isolated."""
        witness_index = WitnessIndex(resolution=2)
        
        # Register two distinct groups
        for i in range(5):
            witness_index.register(0.1, 0.1, level=0, satellite_idx=i)
        
        for i in range(5, 10):
            witness_index.register(0.9, 0.9, level=0, satellite_idx=i)
        
        # Groups should be isolated
        group1 = witness_index.get_entangled(0.1, 0.1)
        group2 = witness_index.get_entangled(0.9, 0.9)
        
        assert len(group1) == 5, "Group 1 should have 5 members"
        assert len(group2) == 5, "Group 2 should have 5 members"
        
        # No overlap
        group1_set = set(group1)
        group2_set = set(group2)
        assert len(group1_set & group2_set) == 0, "Groups should not overlap"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
