"""
Modal Performance Benchmarks — Theory-True Performance Analysis
================================================================

Detailed performance benchmarks using HolographicMemory directly.
No backward compatibility aliases.

Measures:
- Throughput scaling with batch size
- Context length scaling (O(n) advantage)
- GPU memory usage
- Throughput consistency
"""

import pytest
import numpy as np
import time
from typing import Dict, List

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
from holographic_prod.core.constants import PHI_INV


class TestModalPerformance:
    """Detailed performance benchmarks."""
    
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
    
    def test_batch_size_scaling(self, memory):
        """Test throughput scaling with batch size."""
        batch_sizes = [32, 64, 128, 256, 512]
        results = {}
        
        for batch_size in batch_sizes:
            contexts = [
                [i % 100, (i+1) % 100, (i+2) % 100]
                for i in range(batch_size)
            ]
            targets = [(i+3) % 100 for i in range(batch_size)]
            
            start = time.perf_counter()
            memory.learn_batch(contexts, targets)
            elapsed = time.perf_counter() - start
            
            throughput = batch_size / elapsed
            results[batch_size] = throughput
            print(f"  Batch size {batch_size:3d}: {throughput:8,.0f} samples/sec")
        
        # Larger batches should have comparable throughput (amortized overhead)
        assert results[512] >= results[32] * 0.5
    
    def test_context_length_scaling(self, memory):
        """Test throughput scaling with context length.
        
        THEORY-TRUE: O(n) context scaling (NOT O(n²) like transformers).
        
        Long contexts are slower due to geometric product chain O(n),
        but the constant factor is low due to batched operations.
        """
        context_lengths = [3, 10, 32, 100, 512]
        batch_size = 128
        results = {}
        
        for ctx_len in context_lengths:
            contexts = [
                [i % 100 for _ in range(ctx_len)]
                for i in range(batch_size)
            ]
            targets = [i % 100 for i in range(batch_size)]
            
            start = time.perf_counter()
            memory.learn_batch(contexts, targets)
            elapsed = time.perf_counter() - start
            
            throughput = batch_size / elapsed
            results[ctx_len] = throughput
            print(f"  Context length {ctx_len:3d}: {throughput:8,.0f} samples/sec")
        
        # O(n) scaling: at 512 tokens (170× baseline), expect ~1/170 throughput
        # But batching gives ~4× boost, so threshold at 0.5%
        min_ratio = 0.005
        assert results[512] > results[3] * min_ratio, \
            f"512-token should be >{min_ratio*100:.1f}% of 3-token (O(n) scaling)"
    
    def test_gpu_memory_usage(self, memory):
        """Test GPU memory usage."""
        if not HAS_CUPY or not memory.use_gpu:
            pytest.skip("GPU not available - run on Modal H100")
        
        mempool = cp.get_default_memory_pool()
        baseline = mempool.used_bytes()
        
        # Allocate large batch
        batch_size = 512
        contexts = [
            [i % 100, (i+1) % 100, (i+2) % 100]
            for i in range(batch_size)
        ]
        targets = [(i+3) % 100 for i in range(batch_size)]
        
        memory.learn_batch(contexts, targets)
        
        used = mempool.used_bytes()
        print(f"  GPU memory used: {used / 1024**2:.1f} MB")
        
        # Should not exceed reasonable limits
        assert used < 1024**3  # < 1GB
    
    def test_throughput_consistency(self, memory):
        """Test throughput consistency over many batches."""
        batch_size = 256
        num_batches = 50
        throughputs = []
        
        for i in range(num_batches):
            contexts = [
                [j % 100, (j+1) % 100, (j+2) % 100]
                for j in range(i * batch_size, (i + 1) * batch_size)
            ]
            targets = [
                (j+3) % 100
                for j in range(i * batch_size, (i + 1) * batch_size)
            ]
            
            start = time.perf_counter()
            memory.learn_batch(contexts, targets)
            elapsed = time.perf_counter() - start
            
            throughput = batch_size / elapsed
            throughputs.append(throughput)
        
        mean_throughput = np.mean(throughputs)
        std_throughput = np.std(throughputs)
        cv = std_throughput / mean_throughput  # Coefficient of variation
        
        print(f"  Mean throughput: {mean_throughput:,.0f} samples/sec")
        print(f"  Std deviation: {std_throughput:,.0f} samples/sec")
        print(f"  Coefficient of variation: {cv:.3f}")
        
        # CV should be < 0.5 (relatively consistent)
        assert cv < 0.5, f"Throughput too variable: CV={cv:.3f}"
