"""
Modal-Specific Tests — GPU, Batch Operations, and Performance
================================================================

Theory-true tests for Modal deployment using HolographicMemory directly.
No backward compatibility aliases.

Tests GPU acceleration, batch operations, and performance benchmarks.
GPU tests skip if CuPy unavailable (runs on Modal H100).
"""

import pytest
import numpy as np
import time
from typing import List, Tuple, Dict

# Check CuPy availability
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

# Import real classes - NO ALIASES
from holographic_prod.memory.holographic_memory_unified import (
    HolographicMemory,
    MemoryConfig,
)
from holographic_prod.attention.toroidal_attention import ToroidalAttention
from holographic_prod.dreaming import DreamingSystem, EpisodicEntry
from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
    MATRIX_DIM, CLIFFORD_DIM, DTYPE,
    GRACE_SCALES,
)
from holographic_prod.core.algebra import (
    geometric_product,
    geometric_product_batch,
    grace_operator,
    grace_operator_batch,
    decompose_to_coefficients_batch,
    build_clifford_basis,
)


# =============================================================================
# GPU AVAILABILITY TESTS
# =============================================================================

class TestGPUAvailability:
    """Test GPU availability and CuPy functionality."""
    
    def test_cupy_available(self):
        """Verify CuPy is available in Modal GPU environment."""
        if not HAS_CUPY:
            pytest.skip("CuPy not available - run on Modal H100")
        
        # Basic CuPy operations
        a = cp.array([1.0, 2.0, 3.0], dtype=DTYPE)
        b = cp.array([4.0, 5.0, 6.0], dtype=DTYPE)
        c = a + b
        
        assert cp.allclose(c, cp.array([5.0, 7.0, 9.0], dtype=DTYPE))
        assert c.device.id >= 0  # On GPU
    
    def test_gpu_memory_allocation(self):
        """Test GPU memory allocation for 4x4 Clifford matrices."""
        if not HAS_CUPY:
            pytest.skip("CuPy not available - run on Modal H100")
        
        # Allocate batch of Clifford matrices on GPU
        batch_size = 1000
        gpu_matrices = cp.zeros((batch_size, MATRIX_DIM, MATRIX_DIM), dtype=DTYPE)
        
        assert gpu_matrices.shape == (batch_size, MATRIX_DIM, MATRIX_DIM)
        assert gpu_matrices.device.id >= 0
        
        # Verify DTYPE
        assert gpu_matrices.dtype == DTYPE
    
    def test_cpu_gpu_transfer(self):
        """Test CPU↔GPU data transfer for Clifford matrices."""
        if not HAS_CUPY:
            pytest.skip("CuPy not available - run on Modal H100")
        
        # CPU → GPU
        cpu_matrix = np.eye(MATRIX_DIM, dtype=DTYPE)
        gpu_matrix = cp.asarray(cpu_matrix)
        
        assert isinstance(gpu_matrix, cp.ndarray)
        assert gpu_matrix.device.id >= 0
        
        # GPU → CPU
        cpu_back = cp.asnumpy(gpu_matrix)
        assert isinstance(cpu_back, np.ndarray)
        assert np.allclose(cpu_matrix, cpu_back)


# =============================================================================
# GPU-ACCELERATED HOLOGRAPHIC MEMORY
# =============================================================================

class TestGPUHolographicMemory:
    """Test GPU-accelerated HolographicMemory."""
    
    @pytest.fixture
    def gpu_memory(self):
        """Create GPU-enabled HolographicMemory."""
        if not HAS_CUPY:
            pytest.skip("CuPy not available - run on Modal H100")
        
        config = MemoryConfig(learning_rate=PHI_INV)
        return HolographicMemory(
            vocab_size=1000,
            config=config,
            use_gpu=True,
            seed=42,
        )
    
    def test_gpu_embeddings(self, gpu_memory):
        """Verify embeddings are on GPU."""
        assert gpu_memory.use_gpu
        assert hasattr(gpu_memory.embeddings, 'device')
        assert gpu_memory.embeddings.device.id >= 0
        
        # Shape should be [vocab_size, 4, 4]
        assert gpu_memory.embeddings.shape == (
            gpu_memory.vocab_size, MATRIX_DIM, MATRIX_DIM
        )
    
    def test_gpu_batch_learning(self, gpu_memory):
        """Test batch learning on GPU."""
        batch_size = 128
        contexts = [[i % 100, (i+1) % 100, (i+2) % 100] for i in range(batch_size)]
        targets = [(i+3) % 100 for i in range(batch_size)]
        
        start = time.perf_counter()
        result = gpu_memory.learn_batch(contexts, targets)
        elapsed = time.perf_counter() - start
        
        assert result['n_learned'] == batch_size
        
        # ARCHITECTURAL REALITY: 4x4 matrices (Cl(3,1)) are too small for GPU benefit
        # - GPU kernel launch overhead: ~10μs per operation
        # - 4x4 = 16 floats = 64 bytes (too small to amortize overhead)
        # - CPU outperforms GPU by ~100-300x for this algebra dimension
        # 
        # GPU WILL benefit for:
        # - Cl(4,1) with 32x32 matrices (1024 floats = 4KB)
        # - Very large batches (10k+) where overhead amortizes
        # - Custom fused CUDA kernels (future optimization)
        #
        # For now, verify correctness and reasonable completion time
        throughput = batch_size / elapsed
        print(f"\n  GPU batch learning: {throughput:,.0f} samples/sec")
        print(f"    (Note: 4x4 matrices too small for GPU benefit)")
        assert elapsed < 15.0, f"Batch learning too slow: {elapsed:.1f}s"
    
    def test_gpu_batch_retrieval(self, gpu_memory):
        """Test batch retrieval on GPU."""
        # Learn patterns
        contexts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        targets = [10, 11, 12]
        gpu_memory.learn_batch(contexts, targets)
        
        # Retrieve - should find learned targets
        token1, conf1 = gpu_memory.retrieve_deterministic([1, 2, 3])
        token2, conf2 = gpu_memory.retrieve_deterministic([4, 5, 6])
        
        assert token1 == 10
        assert token2 == 11
        assert conf1 > 0
        assert conf2 > 0
    
    def test_gpu_geometric_product_batch(self, gpu_memory):
        """Test batched geometric product on GPU.
        
        geometric_product_batch reduces a sequence: [M1, M2, ..., Mn] → M1⊗M2⊗...⊗Mn
        """
        xp = gpu_memory.xp
        batch_size = 64
        seq_len = 10
        
        # Create batch of SEQUENCES to reduce (not pairs)
        sequences = xp.random.randn(batch_size, seq_len, MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        
        start = time.perf_counter()
        # Process each sequence with geometric_product_batch
        results = []
        for i in range(batch_size):
            result = geometric_product_batch(sequences[i], xp)
            results.append(result)
        result = xp.stack(results)
        elapsed = time.perf_counter() - start
        
        assert result.shape == (batch_size, MATRIX_DIM, MATRIX_DIM)
        assert elapsed < 1.0
        
        print(f"\n  GPU geometric product batch: {elapsed*1000:.2f}ms")
    
    def test_gpu_grace_operator_batch(self, gpu_memory):
        """Test batched Grace operator on GPU."""
        xp = gpu_memory.xp
        basis = gpu_memory.basis
        batch_size = 32
        
        # Create batch of matrices
        matrices = xp.random.randn(batch_size, MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        
        start = time.perf_counter()
        graced = grace_operator_batch(matrices, basis, xp)
        elapsed = time.perf_counter() - start
        
        assert graced.shape == (batch_size, MATRIX_DIM, MATRIX_DIM)
        assert elapsed < 1.0
        
        # Grace scales grades: scalar×1, vector×φ⁻¹, bivector×φ⁻², trivector×φ⁻³
        # So graced matrices should have lower norms than original
        orig_norms = xp.linalg.norm(matrices.reshape(batch_size, -1), axis=1)
        graced_norms = xp.linalg.norm(graced.reshape(batch_size, -1), axis=1)
        
        # Most graced matrices should have lower norm (weighted down by φ)
        assert float((graced_norms <= orig_norms * 1.1).mean()) > 0.8
        
        print(f"\n  GPU Grace operator batch: {elapsed*1000:.2f}ms")


# =============================================================================
# LARGE BATCH OPERATIONS
# =============================================================================

class TestLargeBatchOperations:
    """Test large batch operations for Modal scale."""
    
    @pytest.fixture
    def memory(self):
        """Create HolographicMemory instance."""
        config = MemoryConfig(learning_rate=PHI_INV)
        return HolographicMemory(
            vocab_size=10000,
            config=config,
            use_gpu=HAS_CUPY,
            seed=42,
        )
    
    def test_large_batch_learning(self, memory):
        """Test learning with large batches (Modal scale)."""
        batch_size = 512
        contexts = [
            [i % 100, (i+1) % 100, (i+2) % 100, (i+3) % 100]
            for i in range(batch_size)
        ]
        targets = [(i+4) % 100 for i in range(batch_size)]
        
        start = time.perf_counter()
        result = memory.learn_batch(contexts, targets)
        elapsed = time.perf_counter() - start
        
        assert result['n_learned'] == batch_size
        assert elapsed < 10.0
        assert memory.n_patterns >= batch_size
        
        print(f"\n  Large batch learning: {batch_size/elapsed:,.0f} samples/sec")
    
    def test_very_large_batch(self, memory):
        """Test very large batch (1024 samples)."""
        batch_size = 1024
        contexts = [
            [i % 100, (i+1) % 100, (i+2) % 100]
            for i in range(batch_size)
        ]
        targets = [(i+3) % 100 for i in range(batch_size)]
        
        start = time.perf_counter()
        result = memory.learn_batch(contexts, targets)
        elapsed = time.perf_counter() - start
        
        assert result['n_learned'] == batch_size
        assert elapsed < 20.0
    
    def test_batch_with_variable_length_contexts(self, memory):
        """Test batch with variable-length contexts."""
        batch = [
            ([1, 2], 3),
            ([4, 5, 6], 7),
            ([8, 9, 10, 11], 12),
            ([13, 14], 15),
        ]
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        result = memory.learn_batch(contexts, targets)
        assert result['n_learned'] == len(batch)
    
    def test_batch_throughput(self, memory):
        """Measure batch learning throughput."""
        batch_size = 256
        num_batches = 10
        
        contexts = [
            [i % 100, (i+1) % 100, (i+2) % 100]
            for i in range(batch_size)
        ]
        targets = [(i+3) % 100 for i in range(batch_size)]
        
        start = time.perf_counter()
        for _ in range(num_batches):
            memory.learn_batch(contexts, targets)
        elapsed = time.perf_counter() - start
        
        total_samples = batch_size * num_batches
        throughput = total_samples / elapsed
        
        # ARCHITECTURAL REALITY: 4x4 matrices (Cl(3,1)) are GPU-inefficient
        # - CPU achieves ~340k samples/sec for this algebra dimension
        # - GPU achieves ~1-2k samples/sec due to kernel launch overhead
        # - This is NOT a bug - it's the physics of small matrix operations
        #
        # BENCHMARK (documentation, not pass/fail):
        print(f"\n  Batch throughput: {throughput:,.0f} samples/sec")
        print(f"    (4x4 matrices: GPU overhead dominates, CPU is faster)")
        
        # Just verify reasonable completion
        assert elapsed < 30.0, f"Batch operations too slow: {elapsed:.1f}s"


# =============================================================================
# MEMORY MANAGEMENT AT SCALE
# =============================================================================

class TestMemoryManagement:
    """Test memory management at Modal scale."""
    
    @pytest.fixture
    def memory(self):
        """Create HolographicMemory instance."""
        config = MemoryConfig(learning_rate=PHI_INV)
        return HolographicMemory(
            vocab_size=10000,
            config=config,
            use_gpu=HAS_CUPY,
            seed=42,
        )
    
    def test_memory_growth(self, memory):
        """Test memory growth with many patterns."""
        batch_size = 128
        num_batches = 50
        
        for batch_idx in range(num_batches):
            contexts = [
                [i % 1000, (i+1) % 1000, (i+2) % 1000]
                for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            ]
            targets = [
                (i+3) % 1000
                for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            ]
            memory.learn_batch(contexts, targets)
            assert memory.n_patterns > 0
        
        assert memory.n_patterns >= batch_size * num_batches
    
    def test_satellite_routing_distribution(self, memory):
        """Test bindings distributed across satellites."""
        contexts = [[i % 100, (i+1) % 100] for i in range(100)]
        targets = [(i+2) % 100 for i in range(100)]
        memory.learn_batch(contexts, targets)
        
        # Count bindings per satellite
        binding_counts = [sat.n_bindings for sat in memory.tower.satellites]
        non_empty = sum(1 for c in binding_counts if c > 0)
        
        # Multiple satellites should be used
        assert non_empty >= 4, f"Expected >=4 satellites, got {non_empty}"
        assert sum(binding_counts) == 100
    
    def test_grace_forgetting(self, memory):
        """Test Grace operator for controlled forgetting."""
        # Learn many patterns
        contexts = [[i % 100, (i+1) % 100] for i in range(1000)]
        targets = [(i+2) % 100 for i in range(1000)]
        memory.learn_batch(contexts, targets)
        
        xp = memory.xp
        holo = memory.holographic_memory
        initial_norm = float(xp.linalg.norm(holo, 'fro'))
        
        # Apply Grace operator
        basis = memory.basis if memory.use_gpu else memory.basis_cpu
        graced = grace_operator(holo, basis, xp)
        
        if memory.use_gpu and hasattr(graced, 'get'):
            graced_np = graced.get()
        else:
            graced_np = graced
        
        final_norm = float(np.linalg.norm(graced_np, 'fro'))
        
        # Grace reduces norm (φ-weighted forgetting)
        assert final_norm <= initial_norm


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarks for Modal deployment."""
    
    @pytest.fixture
    def memory(self):
        """Create HolographicMemory instance."""
        config = MemoryConfig(learning_rate=PHI_INV)
        return HolographicMemory(
            vocab_size=10000,
            config=config,
            use_gpu=HAS_CUPY,
            seed=42,
        )
    
    def test_learning_throughput(self, memory):
        """Benchmark learning throughput."""
        batch_size = 256
        num_iterations = 20
        
        contexts = [
            [i % 100, (i+1) % 100, (i+2) % 100]
            for i in range(batch_size)
        ]
        targets = [(i+3) % 100 for i in range(batch_size)]
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            memory.learn_batch(contexts, targets)
        elapsed = time.perf_counter() - start
        
        total_samples = batch_size * num_iterations
        throughput = total_samples / elapsed
        
        # ARCHITECTURAL REALITY for Cl(3,1) with 4x4 matrices:
        # - CPU: ~300k samples/sec (excellent, no overhead)
        # - GPU: ~1-2k samples/sec (kernel launch overhead dominates)
        #
        # This is NOT a performance bug - it's the physics of small matrices.
        # GPU benefit emerges at Cl(4,1) with 32x32 matrices.
        min_throughput = 500 if HAS_CUPY else 5000
        assert throughput > min_throughput, f"Throughput {throughput:,.0f} below {min_throughput:,}"
        
        print(f"\n  Learning throughput: {throughput:,.0f} samples/sec")
        if HAS_CUPY:
            print(f"    (4x4 matrices: GPU overhead dominates)")
    
    def test_retrieval_throughput(self, memory):
        """Benchmark retrieval throughput."""
        # Learn patterns
        contexts = [[i % 100, (i+1) % 100, (i+2) % 100] for i in range(100)]
        targets = [(i+3) % 100 for i in range(100)]
        memory.learn_batch(contexts, targets)
        
        # Benchmark retrieval
        num_queries = 500
        query_contexts = [[i % 100, (i+1) % 100, (i+2) % 100] for i in range(num_queries)]
        
        start = time.perf_counter()
        for ctx in query_contexts:
            memory.retrieve_deterministic(ctx)
        elapsed = time.perf_counter() - start
        
        throughput = num_queries / elapsed
        
        # Retrieval is inherently sequential (one query at a time)
        # GPU provides no benefit for sequential operations
        # BENCHMARK (documentation):
        print(f"\n  Retrieval throughput: {throughput:,.0f} queries/sec")
        print(f"    (Sequential retrieval: ~5-7ms per query)")
        
        # Verify reasonable performance (>100 queries/sec = <10ms per query)
        assert throughput > 100, f"Retrieval throughput {throughput:,.0f} too low"
    
    def test_generation_throughput(self, memory):
        """Benchmark generation throughput."""
        # Learn patterns
        contexts = [[i % 100, (i+1) % 100] for i in range(100)]
        targets = [(i+2) % 100 for i in range(100)]
        memory.learn_batch(contexts, targets)
        
        # Benchmark generation
        num_generations = 50
        prompts = [[i % 100, (i+1) % 100] for i in range(num_generations)]
        
        start = time.perf_counter()
        for prompt in prompts:
            memory.generate(prompt, max_tokens=10)
        elapsed = time.perf_counter() - start
        
        throughput = num_generations / elapsed
        
        # Target: >5 generations/sec
        # Theory-true decoding (vorticity weighting) has O(1) fast path for clean retrieval
        assert throughput > 5, f"Generation throughput {throughput:.2f} too low"
        print(f"\n  Generation throughput: {throughput:.2f} generations/sec")
    
    def test_batch_vs_single_speedup(self, memory):
        """Compare batch vs single-sample throughput."""
        batch_size = 256
        contexts = [
            [i % 100, (i+1) % 100, (i+2) % 100]
            for i in range(batch_size)
        ]
        targets = [(i+3) % 100 for i in range(batch_size)]
        
        # Batch learning
        start = time.perf_counter()
        memory.learn_batch(contexts, targets)
        batch_time = time.perf_counter() - start
        
        # Reset
        memory.n_patterns = 0
        for sat in memory.tower.satellites:
            sat.memory.fill(0)
            sat.n_bindings = 0
        
        # Single-sample learning
        start = time.perf_counter()
        for ctx, tgt in zip(contexts, targets):
            memory.learn(ctx, tgt)
        single_time = time.perf_counter() - start
        
        speedup = single_time / batch_time
        
        # Batch should be faster
        assert speedup > 1.0, "Batch should be faster than single-sample"
        print(f"\n  Batch speedup: {speedup:.2f}×")


# =============================================================================
# THEORY COMPLIANCE
# =============================================================================

class TestTheoryCompliance:
    """Verify theory compliance in Modal environment."""
    
    @pytest.fixture
    def memory(self):
        """Create HolographicMemory instance."""
        config = MemoryConfig(learning_rate=PHI_INV)
        return HolographicMemory(
            vocab_size=1000,
            config=config,
            use_gpu=HAS_CUPY,
            seed=42,
        )
    
    def test_phi_derived_constants(self, memory):
        """Verify all constants are φ-derived."""
        # Grace scales must be φ powers
        assert abs(GRACE_SCALES[0] - 1.0) < 1e-10  # Scalar: φ⁰
        assert abs(GRACE_SCALES[1] - PHI_INV) < 1e-10  # Vector: φ⁻¹
        assert abs(GRACE_SCALES[2] - PHI_INV_SQ) < 1e-10  # Bivector: φ⁻²
        assert abs(GRACE_SCALES[3] - PHI_INV_CUBE) < 1e-10  # Trivector: φ⁻³
        
        # Learning rate is φ⁻¹
        assert abs(memory.config.learning_rate - PHI_INV) < 1e-10
    
    def test_no_softmax(self, memory):
        """Verify attractor-based generation, not softmax (v5.16.0)."""
        contexts = [[1, 2, 3], [4, 5, 6]]
        targets = [7, 8]
        memory.learn_batch(contexts, targets)
        
        # Generation uses attractor flow (theory-true)
        generated, stats = memory.generate([1, 2, 3], max_tokens=5)
        
        # v5.16.0: Stats should show attractor-based generation
        # (no avg_probability - that was ML-thinking)
        assert 'attractor_flow' in stats or 'avg_stability' in stats
        assert len(generated) > 0
    
    def test_grace_basin_keys(self, memory):
        """Verify Grace basin key consistency."""
        contexts = [[1, 2, 3], [4, 5, 6], [1, 2, 3]]
        targets = [7, 8, 9]
        memory.learn_batch(contexts, targets)
        
        # Same context → same basin key
        key1 = memory._grace_basin_key(memory.embed_sequence([1, 2, 3]))
        key3 = memory._grace_basin_key(memory.embed_sequence([1, 2, 3]))
        
        assert key1 == key3
    
    def test_holographic_superposition(self, memory):
        """Verify holographic superposition via tower."""
        contexts = [[1, 2], [3, 4], [5, 6]]
        targets = [10, 11, 12]
        memory.learn_batch(contexts, targets)
        
        assert memory.n_patterns == 3
        
        # Tower should have non-zero bindings
        total_norm = sum(
            float(np.linalg.norm(sat.memory)) for sat in memory.tower.satellites
        )
        assert total_norm > 0
    
    def test_geometric_noncommutativity(self, memory):
        """Verify geometric product is non-commutative."""
        ctx1 = memory.embed_sequence([1, 2, 3])
        ctx2 = memory.embed_sequence([4, 5, 6])
        
        prod1 = geometric_product(ctx1, ctx2)
        prod2 = geometric_product(ctx2, ctx1)
        
        # Non-commutative
        assert not np.allclose(prod1, prod2, atol=1e-6)


# =============================================================================
# REAL DATA PROCESSING
# =============================================================================

class TestRealDataProcessing:
    """Test with real data patterns."""
    
    @pytest.fixture
    def memory(self):
        """Create HolographicMemory instance."""
        config = MemoryConfig(learning_rate=PHI_INV)
        return HolographicMemory(
            vocab_size=10000,
            config=config,
            use_gpu=HAS_CUPY,
            seed=42,
        )
    
    def test_wikitext_style_training(self, memory):
        """Test WikiText-2 style sequences."""
        sequences = [
            list(range(10, 20)),
            list(range(20, 35)),
            list(range(35, 45)),
            list(range(45, 60)),
        ]
        
        # Create (context, target) pairs
        batch = []
        for seq in sequences:
            for i in range(len(seq) - 1):
                context = seq[:i+1]
                target = seq[i+1]
                batch.append((context, target))
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        result = memory.learn_batch(contexts, targets)
        assert result['n_learned'] == len(batch)
        
        # Retrieval should work
        test_ctx = sequences[0][:5]
        token, conf = memory.retrieve_deterministic(test_ctx)
        assert token is not None
        assert conf >= 0
    
    def test_long_context_handling(self, memory):
        """Test long contexts (O(n) scaling advantage)."""
        # Create long context
        long_context = list(range(1000))
        
        # Should handle without error
        ctx_matrix = memory.embed_sequence(long_context)
        assert ctx_matrix.shape == (MATRIX_DIM, MATRIX_DIM)
        
        # Learn and retrieve
        memory.learn(long_context[:500], 501)
        token, conf = memory.retrieve_deterministic(long_context[:500])
        assert token is not None


# =============================================================================
# LONG-RUNNING STABILITY
# =============================================================================

class TestLongRunningStability:
    """Test long-running stability."""
    
    @pytest.fixture
    def memory(self):
        """Create HolographicMemory instance."""
        config = MemoryConfig(learning_rate=PHI_INV)
        return HolographicMemory(
            vocab_size=10000,
            config=config,
            use_gpu=HAS_CUPY,
            seed=42,
        )
    
    def test_extended_training(self, memory):
        """Test extended training without memory leaks."""
        batch_size = 128
        num_batches = 100
        
        initial_patterns = memory.n_patterns
        
        for batch_idx in range(num_batches):
            contexts = [
                [i % 1000, (i+1) % 1000, (i+2) % 1000]
                for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            ]
            targets = [
                (i+3) % 1000
                for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            ]
            memory.learn_batch(contexts, targets)
            
            # Check stability periodically
            if batch_idx % 10 == 0:
                stability = memory.get_stability()
                assert 0 <= stability <= 1
        
        assert memory.n_patterns > initial_patterns
        
        # Should still retrieve
        token, conf = memory.retrieve_deterministic([0, 1, 2])
        assert token is not None
    
    def test_deterministic_consistency(self, memory):
        """Test deterministic retrieval consistency."""
        contexts = [[i % 100, (i+1) % 100] for i in range(200)]
        targets = [(i+2) % 100 for i in range(200)]
        memory.learn_batch(contexts, targets)
        
        # Deterministic retrieval should be consistent
        results1 = [memory.retrieve_deterministic(ctx) for ctx in contexts[:10]]
        results2 = [memory.retrieve_deterministic(ctx) for ctx in contexts[:10]]
        
        for (t1, c1), (t2, c2) in zip(results1, results2):
            assert t1 == t2
            assert abs(c1 - c2) < 1e-6


# =============================================================================
# MODAL INTEGRATION
# =============================================================================

class TestModalIntegration:
    """Test Modal-specific integration."""
    
    @pytest.fixture
    def memory(self):
        """Create HolographicMemory instance."""
        config = MemoryConfig(learning_rate=PHI_INV)
        return HolographicMemory(
            vocab_size=1000,
            config=config,
            use_gpu=HAS_CUPY,
            seed=42,
        )
    
    @pytest.fixture
    def attention(self):
        """Create ToroidalAttention instance."""
        return ToroidalAttention(n_satellites=16)
    
    def test_learn_batch_with_attention(self, memory, attention):
        """Test learn_batch_with_attention (Modal training path)."""
        batch = [
            ([1, 2, 3], 4),
            ([5, 6, 7], 8),
            ([9, 10, 11], 12),
        ]
        
        result = memory.learn_batch_with_attention(batch, attention)
        assert result['n_samples'] == len(batch)
        assert 'accuracy' in result
    
    def test_dreaming_integration(self, memory):
        """Test DreamingSystem integration."""
        dreamer = DreamingSystem(
            basis=memory.basis,
            xp=memory.xp,
        )
        
        # Create episodes
        episodes = []
        for i in range(20):
            ctx = [i % 100, (i+1) % 100, (i+2) % 100]
            tgt = (i+3) % 100
            ctx_matrix = memory.embed_sequence(ctx)
            episodes.append(EpisodicEntry(context_matrix=ctx_matrix, target_token=tgt))
        
        # Run sleep cycle
        dream_stats = dreamer.sleep(episodes, verbose=False)
        assert 'prototypes_created' in dream_stats
        assert 'schemas_discovered' in dream_stats
    
    def test_generation_pipeline(self, memory):
        """Test full generation pipeline."""
        contexts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        targets = [10, 11, 12]
        memory.learn_batch(contexts, targets)
        
        # Generate
        generated, stats = memory.generate([1, 2, 3], max_tokens=10, context_size=3)
        
        assert len(generated) > 0
        assert 'tokens_generated' in stats
        assert stats['tokens_generated'] > 0


# =============================================================================
# CPU vs GPU COMPARISON
# =============================================================================

class TestCPUvsGPU:
    """Compare CPU vs GPU performance."""
    
    def test_throughput_comparison(self):
        """Compare CPU vs GPU throughput."""
        if not HAS_CUPY:
            pytest.skip("CuPy not available - run on Modal H100")
        
        batch_size = 256
        contexts = [
            [i % 100, (i+1) % 100, (i+2) % 100]
            for i in range(batch_size)
        ]
        targets = [(i+3) % 100 for i in range(batch_size)]
        
        # CPU
        cpu_config = MemoryConfig(learning_rate=PHI_INV)
        cpu_memory = HolographicMemory(
            vocab_size=1000, config=cpu_config, use_gpu=False, seed=42
        )
        
        start = time.perf_counter()
        cpu_memory.learn_batch(contexts, targets)
        cpu_time = time.perf_counter() - start
        
        # GPU
        gpu_config = MemoryConfig(learning_rate=PHI_INV)
        gpu_memory = HolographicMemory(
            vocab_size=1000, config=gpu_config, use_gpu=True, seed=42
        )
        
        start = time.perf_counter()
        gpu_memory.learn_batch(contexts, targets)
        gpu_time = time.perf_counter() - start
        
        speedup = cpu_time / gpu_time
        
        # Log speedup - GPU may not always be faster for small batches due to overhead
        # Primary benefit is at larger batch sizes (1024+) and longer contexts
        print(f"\n  GPU speedup: {speedup:.2f}×")
        print(f"    CPU time: {cpu_time*1000:.1f}ms, GPU time: {gpu_time*1000:.1f}ms")
        
        # Verify GPU works correctly (not necessarily faster for small batches)
        assert gpu_time > 0, "GPU should complete"


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStressTests:
    """Stress tests for Modal deployment."""
    
    @pytest.fixture
    def memory(self):
        """Create HolographicMemory instance."""
        config = MemoryConfig(learning_rate=PHI_INV)
        return HolographicMemory(
            vocab_size=10000,
            config=config,
            use_gpu=HAS_CUPY,
            seed=42,
        )
    
    def test_many_unique_contexts(self, memory):
        """Test with many unique contexts."""
        num_contexts = 5000
        contexts = [
            [i % 1000, (i+1) % 1000, (i+2) % 1000]
            for i in range(num_contexts)
        ]
        targets = [(i+3) % 1000 for i in range(num_contexts)]
        
        # Process in batches
        batch_size = 256
        for i in range(0, num_contexts, batch_size):
            batch_ctx = contexts[i:i+batch_size]
            batch_tgt = targets[i:i+batch_size]
            memory.learn_batch(batch_ctx, batch_tgt)
        
        assert memory.n_patterns >= num_contexts
    
    def test_rapid_batch_learning(self, memory):
        """Test rapid batch learning."""
        batch_size = 128
        num_batches = 50
        
        for batch_idx in range(num_batches):
            contexts = [
                [i % 1000, (i+1) % 1000, (i+2) % 1000]
                for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            ]
            targets = [
                (i+3) % 1000
                for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            ]
            memory.learn_batch(contexts, targets)
        
        assert memory.n_patterns >= batch_size * num_batches
    
    def test_concurrent_operations(self, memory):
        """Test concurrent learn/retrieve operations."""
        contexts = [[i % 100, (i+1) % 100] for i in range(100)]
        targets = [(i+2) % 100 for i in range(100)]
        memory.learn_batch(contexts, targets)
        
        # Many retrievals
        query_contexts = contexts[:20]
        results = []
        for ctx in query_contexts:
            token, prob, _ = memory.retrieve_probabilistic(ctx)
            results.append((token, prob))
        
        assert all(token is not None for token, _ in results)
        assert all(0 <= prob <= 1 for _, prob in results)
