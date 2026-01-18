"""
Benchmark: GPU-Native vs Dict-Based Witness Index

This benchmark measures the actual speedup achieved by eliminating
GPU→CPU synchronization in the witness index.

WHAT WE'RE MEASURING:
1. store_batch time comparison
2. Total sync overhead
3. Scaling behavior

EXPECTED RESULTS:
- GPU-native: O(1) CPU time per batch
- Dict-based: O(batch_size) CPU time per batch due to sync
"""

import numpy as np
import time
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


def create_test_batch(batch_size: int, xp) -> Tuple:
    """Create test batch."""
    np.random.seed(42)
    contexts = xp.array(np.random.randn(batch_size, MATRIX_DIM, MATRIX_DIM).astype(DTYPE))
    targets = xp.array(np.random.randn(batch_size, MATRIX_DIM, MATRIX_DIM).astype(DTYPE))
    target_idxs = xp.arange(batch_size, dtype=xp.int32)
    return contexts, targets, target_idxs


def benchmark_dict_based(n_batches: int, batch_size: int):
    """Benchmark original dict-based WitnessIndex."""
    from holographic_v4.holographic_memory import WitnessIndex
    from holographic_v4.algebra import build_clifford_basis
    
    xp, has_gpu = get_xp()
    basis = build_clifford_basis(xp)
    
    index = WitnessIndex.create(basis, xp=xp)
    
    times = []
    total_stored = 0
    
    for batch_idx in range(n_batches):
        contexts, targets, target_idxs = create_test_batch(batch_size, xp)
        target_idxs = target_idxs + batch_idx * batch_size
        
        # Sync before timing (if GPU)
        if has_gpu:
            xp.cuda.Stream.null.synchronize()
        
        start = time.perf_counter()
        stats = index.store_batch(contexts, targets, target_idxs)
        
        # Sync after (if GPU)
        if has_gpu:
            xp.cuda.Stream.null.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_stored += stats.get('stored', 0)
    
    return {
        'name': 'Dict-Based WitnessIndex',
        'times': times,
        'total_time': sum(times),
        'avg_time': sum(times) / len(times),
        'total_stored': total_stored,
    }


def benchmark_gpu_native(n_batches: int, batch_size: int):
    """Benchmark GPU-native WitnessIndex."""
    from holographic_v4.gpu_witness_index import GPUWitnessIndex
    from holographic_v4.algebra import build_clifford_basis
    
    xp, has_gpu = get_xp()
    basis = build_clifford_basis(xp)
    
    max_items = n_batches * batch_size  # Preallocate enough
    index = GPUWitnessIndex.create(basis, max_items=max_items, xp=xp)
    
    times = []
    total_stored = 0
    
    for batch_idx in range(n_batches):
        contexts, targets, target_idxs = create_test_batch(batch_size, xp)
        target_idxs = target_idxs + batch_idx * batch_size
        
        # Sync before timing (if GPU)
        if has_gpu:
            xp.cuda.Stream.null.synchronize()
        
        start = time.perf_counter()
        stats = index.store_batch(contexts, targets, target_idxs)
        
        # Sync after (if GPU)
        if has_gpu:
            xp.cuda.Stream.null.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_stored += stats.get('stored', 0)
    
    return {
        'name': 'GPU-Native WitnessIndex',
        'times': times,
        'total_time': sum(times),
        'avg_time': sum(times) / len(times),
        'total_stored': total_stored,
    }


def run_benchmark():
    """Run benchmark and print results."""
    xp, has_gpu = get_xp()
    print(f"Running on: {'GPU (CuPy)' if has_gpu else 'CPU (NumPy)'}")
    print("=" * 60)
    
    # Test parameters
    n_batches = 10
    batch_size = 500
    
    print(f"Parameters: {n_batches} batches × {batch_size} items = {n_batches * batch_size} total")
    print()
    
    # Warmup
    print("Warmup...")
    _ = benchmark_gpu_native(2, 100)
    _ = benchmark_dict_based(2, 100)
    
    # Actual benchmark
    print("Running dict-based benchmark...")
    dict_results = benchmark_dict_based(n_batches, batch_size)
    
    print("Running GPU-native benchmark...")
    gpu_results = benchmark_gpu_native(n_batches, batch_size)
    
    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for results in [dict_results, gpu_results]:
        print(f"\n{results['name']}:")
        print(f"  Total time: {results['total_time']*1000:.2f}ms")
        print(f"  Avg per batch: {results['avg_time']*1000:.2f}ms")
        print(f"  Items stored: {results['total_stored']}")
        print(f"  Per-batch times (ms): {[f'{t*1000:.2f}' for t in results['times'][:5]]}...")
    
    # Speedup
    speedup = dict_results['total_time'] / gpu_results['total_time']
    print()
    print(f"SPEEDUP: {speedup:.1f}×")
    print()
    
    if speedup > 10:
        print("✓ SIGNIFICANT SPEEDUP - GPU-native implementation is effective")
    elif speedup > 2:
        print("~ MODERATE SPEEDUP - Some benefit from GPU-native")
    else:
        print("⚠ MINIMAL SPEEDUP - May need further optimization")
        print("  (On CPU, both implementations are similar)")
    
    return dict_results, gpu_results


if __name__ == "__main__":
    run_benchmark()
