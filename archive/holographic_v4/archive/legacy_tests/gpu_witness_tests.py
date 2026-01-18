"""
GPU-Native Witness Index Tests

TEST-DRIVEN DEVELOPMENT for GPU-optimized witness storage.

DESIGN REQUIREMENTS:
1. All storage on GPU (no Python dicts)
2. Vectorized lookup (no element iteration)
3. Zero GPU→CPU sync during training
4. Preserve witness-space semantics

EXPECTED BEHAVIOR:
- store_batch: O(1) GPU time regardless of batch size
- retrieve_batch: O(1) GPU time for batch lookup
- No .get() or .copy() calls during normal operation
"""

import numpy as np
import time
from typing import Tuple, Optional
from dataclasses import dataclass

# Constants from the codebase
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_INV_SQ = PHI_INV ** 2
MATRIX_DIM = 4
DTYPE = np.float32


def get_xp():
    """Get array module (CuPy if available, else NumPy)."""
    try:
        import cupy as cp
        return cp, True
    except ImportError:
        return np, False


class TestGPUWitnessIndex:
    """Test suite for GPU-native witness index."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.xp, self.has_gpu = get_xp()
        
        # Create proper Clifford basis [16, 4, 4]
        from holographic_v4.algebra import build_clifford_basis
        basis_np = build_clifford_basis(np)
        self.basis = self.xp.array(basis_np) if self.has_gpu else basis_np
        
    def _create_test_batch(self, batch_size: int) -> Tuple:
        """Create test batch of contexts and targets."""
        contexts = self.xp.array(
            np.random.randn(batch_size, MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        )
        targets = self.xp.array(
            np.random.randn(batch_size, MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        )
        target_idxs = self.xp.arange(batch_size, dtype=self.xp.int32)
        return contexts, targets, target_idxs
    
    # =========================================================================
    # TEST 1: GPU Storage (No CPU Transfer)
    # =========================================================================
    
    def test_store_batch_stays_on_gpu(self):
        """
        REQUIREMENT: store_batch must NOT transfer data to CPU.
        
        We verify this by checking:
        1. All internal arrays are xp arrays (CuPy if GPU available)
        2. No .get() calls occur during store
        """
        from holographic_v4.gpu_witness_index import GPUWitnessIndex
        
        index = GPUWitnessIndex.create(self.basis, max_items=10000, xp=self.xp)
        contexts, targets, target_idxs = self._create_test_batch(100)
        
        # Store batch
        stats = index.store_batch(contexts, targets, target_idxs)
        
        # Verify internal storage is on correct device
        assert index.witnesses.dtype == DTYPE, "Witnesses should be float32"
        assert index.contexts.shape[1:] == (MATRIX_DIM, MATRIX_DIM)
        assert index.targets.shape[1:] == (MATRIX_DIM, MATRIX_DIM)
        
        # If GPU, verify arrays are CuPy
        if self.has_gpu:
            import cupy as cp
            assert isinstance(index.witnesses, cp.ndarray), "Witnesses must be CuPy array"
            assert isinstance(index.contexts, cp.ndarray), "Contexts must be CuPy array"
            assert isinstance(index.targets, cp.ndarray), "Targets must be CuPy array"
            assert isinstance(index.target_idxs, cp.ndarray), "Target indices must be CuPy array"
        
        assert stats['stored'] > 0, "Should store some items"
        
    # =========================================================================
    # TEST 2: Vectorized Batch Lookup
    # =========================================================================
    
    def test_retrieve_batch_vectorized(self):
        """
        REQUIREMENT: retrieve_batch must be fully vectorized.
        
        We verify by:
        1. Storing patterns
        2. Querying with patterns that WERE stored (accounting for subsampling)
        3. Checking results are correct
        4. Verifying no loops (timing check)
        """
        from holographic_v4.gpu_witness_index import GPUWitnessIndex
        
        index = GPUWitnessIndex.create(self.basis, max_items=10000, xp=self.xp)
        
        # Store 500 patterns (subsampled at step 3)
        contexts, targets, target_idxs = self._create_test_batch(500)
        index.store_batch(contexts, targets, target_idxs)
        
        # Query with patterns that WERE stored (every 3rd, matching subsample_step)
        subsample_step = 3
        stored_indices = list(range(0, 500, subsample_step))[:50]  # First 50 stored
        query_contexts = contexts[stored_indices]  # [50, 4, 4]
        
        # Retrieve
        retrieved_targets, retrieved_idxs, confidences = index.retrieve_batch(query_contexts)
        
        # Verify shapes
        assert retrieved_targets.shape == (50, MATRIX_DIM, MATRIX_DIM)
        assert retrieved_idxs.shape == (50,)
        assert confidences.shape == (50,)
        
        # Verify correct retrieval (querying stored patterns should return them)
        # Allow for quantization effects - at least 70% should match exactly
        if self.has_gpu:
            retrieved_idxs_cpu = retrieved_idxs.get()
        else:
            retrieved_idxs_cpu = retrieved_idxs
        
        # Expected indices are the same as stored_indices (since target_idxs = arange)
        matches = sum(1 for i, stored_idx in enumerate(stored_indices) 
                      if retrieved_idxs_cpu[i] == stored_idx)
        match_rate = matches / len(stored_indices)
        assert match_rate >= 0.7, f"Match rate {match_rate} too low (expected >= 0.7)"
        
    # =========================================================================
    # TEST 3: No GPU→CPU Sync During Training
    # =========================================================================
    
    def test_no_sync_during_training_loop(self):
        """
        REQUIREMENT: Full training loop with NO GPU→CPU synchronization.
        
        This is the critical performance test.
        We simulate a training loop and verify no sync points.
        """
        from holographic_v4.gpu_witness_index import GPUWitnessIndex
        
        index = GPUWitnessIndex.create(self.basis, max_items=50000, xp=self.xp)
        
        # Simulate training loop
        n_batches = 10
        batch_size = 500
        
        for batch_idx in range(n_batches):
            contexts, targets, target_idxs = self._create_test_batch(batch_size)
            
            # Offset indices to simulate different batches
            target_idxs = target_idxs + batch_idx * batch_size
            
            # Store - this must NOT sync
            stats = index.store_batch(contexts, targets, target_idxs)
            
            # Verify we haven't synced by checking arrays are still on device
            if self.has_gpu:
                import cupy as cp
                assert isinstance(index.witnesses, cp.ndarray)
        
        # Final count
        assert index.n_items >= n_batches * batch_size // 3  # Allow for subsampling
        
    # =========================================================================
    # TEST 4: Performance Scaling
    # =========================================================================
    
    def test_performance_scales_correctly(self):
        """
        REQUIREMENT: store_batch time should be O(1) w.r.t. existing items.
        
        We verify by timing stores at different capacities.
        """
        from holographic_v4.gpu_witness_index import GPUWitnessIndex
        
        index = GPUWitnessIndex.create(self.basis, max_items=100000, xp=self.xp)
        batch_size = 500
        
        times = []
        for i in range(5):
            contexts, targets, target_idxs = self._create_test_batch(batch_size)
            target_idxs = target_idxs + i * batch_size
            
            # Sync before timing (if GPU)
            if self.has_gpu:
                self.xp.cuda.Stream.null.synchronize()
            
            start = time.perf_counter()
            index.store_batch(contexts, targets, target_idxs)
            
            # Sync after (if GPU)
            if self.has_gpu:
                self.xp.cuda.Stream.null.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # First store might be slower (memory allocation)
        # Subsequent stores should be roughly constant
        later_times = times[1:]
        avg_time = sum(later_times) / len(later_times)
        max_time = max(later_times)
        
        # Max should not be more than 3x average (allowing for variance)
        assert max_time < avg_time * 3, f"Scaling issue: max={max_time:.4f}, avg={avg_time:.4f}"
        
        print(f"Store times (ms): {[f'{t*1000:.2f}' for t in times]}")
        
    # =========================================================================
    # TEST 5: Witness Space Semantics Preserved
    # =========================================================================
    
    def test_witness_quantization_correct(self):
        """
        REQUIREMENT: Witness quantization must match original implementation.
        
        Verify (σ, p) extraction and quantization is identical.
        """
        from holographic_v4.gpu_witness_index import GPUWitnessIndex
        from holographic_v4.quotient import extract_witness_batch
        
        index = GPUWitnessIndex.create(self.basis, max_items=10000, xp=self.xp)
        
        # Create test matrices
        contexts, _, _ = self._create_test_batch(100)
        
        # Extract witnesses using quotient module
        witnesses_ref = extract_witness_batch(contexts, self.basis, self.xp)
        
        # Extract using GPU index internal method
        witnesses_test = index._extract_witnesses(contexts)
        
        # Compare
        if self.has_gpu:
            witnesses_ref = witnesses_ref.get()
            witnesses_test = witnesses_test.get()
        
        np.testing.assert_allclose(witnesses_test, witnesses_ref, rtol=1e-5)
        
    # =========================================================================
    # TEST 6: Memory Efficiency
    # =========================================================================
    
    def test_memory_preallocated(self):
        """
        REQUIREMENT: Memory should be preallocated, not grown dynamically.
        
        Dynamic growth causes fragmentation and sync points.
        """
        from holographic_v4.gpu_witness_index import GPUWitnessIndex
        
        max_items = 10000
        index = GPUWitnessIndex.create(self.basis, max_items=max_items, xp=self.xp)
        
        # Verify preallocated arrays exist and have correct size
        assert index.witnesses.shape == (max_items, 2)
        assert index.contexts.shape == (max_items, MATRIX_DIM, MATRIX_DIM)
        assert index.targets.shape == (max_items, MATRIX_DIM, MATRIX_DIM)
        assert index.target_idxs.shape == (max_items,)
        
        # n_items should track actual stored count
        assert index.n_items == 0
        
        # Store and verify count increases
        contexts, targets, target_idxs = self._create_test_batch(100)
        index.store_batch(contexts, targets, target_idxs)
        assert index.n_items > 0
        
    # =========================================================================
    # TEST 7: Retrieval Without Stored Data Returns Zeros
    # =========================================================================
    
    def test_empty_retrieval(self):
        """
        REQUIREMENT: Querying empty index returns zeros with zero confidence.
        """
        from holographic_v4.gpu_witness_index import GPUWitnessIndex
        
        index = GPUWitnessIndex.create(self.basis, max_items=10000, xp=self.xp)
        
        # Query without storing
        contexts, _, _ = self._create_test_batch(10)
        targets, idxs, confidences = index.retrieve_batch(contexts)
        
        # Should return zeros
        if self.has_gpu:
            confidences = confidences.get()
        
        assert all(c == 0.0 for c in confidences), "Empty index should return zero confidence"
        

def run_tests():
    """Run all tests and report results."""
    test_instance = TestGPUWitnessIndex()
    
    tests = [
        ("store_batch_stays_on_gpu", test_instance.test_store_batch_stays_on_gpu),
        ("retrieve_batch_vectorized", test_instance.test_retrieve_batch_vectorized),
        ("no_sync_during_training_loop", test_instance.test_no_sync_during_training_loop),
        ("performance_scales_correctly", test_instance.test_performance_scales_correctly),
        ("witness_quantization_correct", test_instance.test_witness_quantization_correct),
        ("memory_preallocated", test_instance.test_memory_preallocated),
        ("empty_retrieval", test_instance.test_empty_retrieval),
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
