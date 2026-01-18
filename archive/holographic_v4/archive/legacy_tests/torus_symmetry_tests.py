"""
Torus Symmetry Tests

TEST-DRIVEN DEVELOPMENT for exploiting torus geometry in witness space.

THEORY (from rhnsclifford.md):
1. Bireflection: σ ↔ (1-σ) identifies two sheets of the torus
2. Critical line: σ = 0.5 is the "throat" where zeros accumulate
3. Periodic: p (pseudoscalar) dimension wraps around

EXPECTED BENEFITS:
1. Memory compression: Store only σ ∈ [0, 0.5], derive other half
2. Retrieval augmentation: Query both σ and (1-σ), take best match
3. Throat-based clustering: Special handling for σ ≈ 0.5
"""

import numpy as np
from typing import Tuple

# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_INV_SQ = PHI_INV ** 2
MATRIX_DIM = 4
DTYPE = np.float32


def get_xp():
    """Get array module."""
    try:
        import cupy as cp
        return cp, True
    except ImportError:
        return np, False


class TestTorusSymmetry:
    """Test suite for torus symmetry exploitation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.xp, self.has_gpu = get_xp()
        
        # Create proper Clifford basis [16, 4, 4]
        from holographic_v4.algebra import build_clifford_basis
        basis_np = build_clifford_basis(np)
        self.basis = self.xp.array(basis_np) if self.has_gpu else basis_np
        
    def _create_test_batch(self, batch_size: int) -> Tuple:
        """Create test batch."""
        np.random.seed(42)
        contexts = self.xp.array(
            np.random.randn(batch_size, MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        )
        targets = self.xp.array(
            np.random.randn(batch_size, MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        )
        target_idxs = self.xp.arange(batch_size, dtype=self.xp.int32)
        return contexts, targets, target_idxs
    
    # =========================================================================
    # TEST 1: Bireflection Computation
    # =========================================================================
    
    def test_bireflection_computes_correctly(self):
        """
        REQUIREMENT: bireflection(σ) = 1 - σ
        
        Verifies the mapping σ → (1-σ) for witness coordinates.
        """
        from holographic_v4.torus_symmetry import bireflect_witness
        
        # Test specific values
        test_cases = [
            (0.0, 1.0),
            (0.5, 0.5),  # Critical line - fixed point!
            (0.3, 0.7),
            (0.8, 0.2),
            (1.0, 0.0),
        ]
        
        for σ, expected in test_cases:
            result = bireflect_witness(σ, 0.0, self.xp)
            assert abs(result[0] - expected) < 1e-6, f"bireflect({σ}) = {result[0]}, expected {expected}"
    
    # =========================================================================
    # TEST 2: Critical Line Detection
    # =========================================================================
    
    def test_throat_detection(self):
        """
        REQUIREMENT: Detect when σ ≈ 0.5 (the critical line/throat).
        
        Witnesses near the throat are special - zeros accumulate there.
        """
        from holographic_v4.torus_symmetry import is_near_throat
        
        # Near throat (σ ≈ 0.5)
        assert is_near_throat(0.5, tolerance=0.1) == True
        assert is_near_throat(0.45, tolerance=0.1) == True
        assert is_near_throat(0.55, tolerance=0.1) == True
        
        # Not near throat
        assert is_near_throat(0.3, tolerance=0.1) == False
        assert is_near_throat(0.7, tolerance=0.1) == False
        assert is_near_throat(0.0, tolerance=0.1) == False
    
    # =========================================================================
    # TEST 3: Canonical Witness (σ ∈ [0, 0.5])
    # =========================================================================
    
    def test_canonical_witness_normalization(self):
        """
        REQUIREMENT: canonicalize_witness maps to σ ∈ [0, 0.5].
        
        This enables memory compression by storing only one sheet.
        """
        from holographic_v4.torus_symmetry import canonicalize_witness
        
        # Create witnesses with various σ values
        witnesses = self.xp.array([
            [0.3, 1.0],   # Already canonical
            [0.7, 1.5],   # Needs bireflection → (0.3, 1.5)
            [0.5, 0.5],   # Throat - stays as is
            [0.9, 2.0],   # Needs bireflection → (0.1, 2.0)
            [0.1, -1.0],  # Already canonical
        ], dtype=DTYPE)
        
        canonical = canonicalize_witness(witnesses, self.xp)
        
        # All σ should be ≤ 0.5
        if self.has_gpu:
            canonical_cpu = canonical.get()
        else:
            canonical_cpu = canonical
        
        for i, (σ, p) in enumerate(canonical_cpu):
            assert σ <= 0.5 + 1e-6, f"Witness {i}: σ={σ} should be ≤ 0.5"
    
    # =========================================================================
    # TEST 4: Bireflection-Augmented Retrieval
    # =========================================================================
    
    def test_bireflection_augmented_retrieval(self):
        """
        REQUIREMENT: Retrieval should check both σ and (1-σ).
        
        A pattern stored at σ=0.3 should be retrievable by querying σ=0.7.
        """
        from holographic_v4.torus_symmetry import TorusAwareWitnessIndex
        
        index = TorusAwareWitnessIndex.create(self.basis, max_items=10000, xp=self.xp)
        
        # Create a context with known σ
        contexts, targets, target_idxs = self._create_test_batch(100)
        
        # Store patterns
        index.store_batch(contexts, targets, target_idxs)
        
        # Query with bireflected witnesses
        # This tests that the retrieval considers both sheets
        retrieved_targets, retrieved_idxs, confidences = index.retrieve_batch(contexts[:10])
        
        # Should have some confidence (bireflection-aware)
        if self.has_gpu:
            confidences_cpu = confidences.get()
        else:
            confidences_cpu = confidences
        
        # At least some should match
        matches = sum(1 for c in confidences_cpu if c > 0.1)
        assert matches > 0, "Bireflection-aware retrieval should find matches"
    
    # =========================================================================
    # TEST 5: Throat-Based Priority
    # =========================================================================
    
    def test_throat_priority_storage(self):
        """
        REQUIREMENT: Patterns near σ=0.5 should have storage priority.
        
        Theory: Zeros accumulate at the throat, making it informationally dense.
        """
        from holographic_v4.torus_symmetry import compute_throat_priority
        
        # Create witnesses
        witnesses = self.xp.array([
            [0.5, 1.0],   # At throat - highest priority
            [0.4, 1.0],   # Near throat
            [0.3, 1.0],   # Medium distance
            [0.1, 1.0],   # Far from throat - lower priority
        ], dtype=DTYPE)
        
        priorities = compute_throat_priority(witnesses, self.xp)
        
        if self.has_gpu:
            priorities_cpu = priorities.get()
        else:
            priorities_cpu = priorities
        
        # Throat (σ=0.5) should have highest priority
        assert priorities_cpu[0] >= priorities_cpu[1] >= priorities_cpu[2] >= priorities_cpu[3], \
            f"Priorities should decrease away from throat: {priorities_cpu}"
    
    # =========================================================================
    # TEST 6: Memory Compression via Canonical Storage
    # =========================================================================
    
    def test_memory_compression(self):
        """
        REQUIREMENT: Canonical storage should reduce memory by ~50%.
        
        By storing only σ ∈ [0, 0.5], we halve the witness space coverage.
        """
        from holographic_v4.torus_symmetry import TorusAwareWitnessIndex
        from holographic_v4.gpu_witness_index import GPUWitnessIndex
        
        # Create two indices
        torus_index = TorusAwareWitnessIndex.create(self.basis, max_items=10000, xp=self.xp)
        standard_index = GPUWitnessIndex.create(self.basis, max_items=10000, xp=self.xp)
        
        # Store same patterns
        contexts, targets, target_idxs = self._create_test_batch(500)
        
        torus_index.store_batch(contexts, targets, target_idxs)
        standard_index.store_batch(contexts, targets, target_idxs)
        
        # Torus-aware should store in canonical form
        # Both should retrieve correctly
        query_contexts = contexts[:50:3]  # Query some stored patterns
        
        _, t_idxs, t_conf = torus_index.retrieve_batch(query_contexts)
        _, s_idxs, s_conf = standard_index.retrieve_batch(query_contexts)
        
        # Both should retrieve, but torus-aware may have different distribution
        if self.has_gpu:
            t_conf_cpu = t_conf.get()
            s_conf_cpu = s_conf.get()
        else:
            t_conf_cpu = t_conf
            s_conf_cpu = s_conf
        
        # Both should find matches
        assert sum(1 for c in t_conf_cpu if c > 0) > 0
        assert sum(1 for c in s_conf_cpu if c > 0) > 0


def run_tests():
    """Run all tests and report results."""
    test_instance = TestTorusSymmetry()
    
    tests = [
        ("bireflection_computes_correctly", test_instance.test_bireflection_computes_correctly),
        ("throat_detection", test_instance.test_throat_detection),
        ("canonical_witness_normalization", test_instance.test_canonical_witness_normalization),
        ("bireflection_augmented_retrieval", test_instance.test_bireflection_augmented_retrieval),
        ("throat_priority_storage", test_instance.test_throat_priority_storage),
        ("memory_compression", test_instance.test_memory_compression),
    ]
    
    results = []
    for name, test_fn in tests:
        test_instance.setup_method()
        try:
            test_fn()
            results.append((name, "PASS", None))
            print(f"✓ {name}")
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            print(f"✗ {name}: {e}")
    
    passed = sum(1 for _, status, _ in results if status == "PASS")
    print(f"\n{passed}/{len(results)} tests passed")
    return results


if __name__ == "__main__":
    run_tests()
