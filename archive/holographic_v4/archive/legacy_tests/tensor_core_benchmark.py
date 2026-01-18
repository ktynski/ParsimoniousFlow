"""
Tensor Core Benchmark

Compares coefficient-space (tensor-core-friendly) vs matrix-space pipelines.

HYPOTHESIS:
- Matrix space: 4×4 matmuls, cannot use tensor cores
- Coefficient space: 256×16 matmuls, CAN use tensor cores

The trade-off is:
- Coefficient space needs conversion (decompose/reconstruct)
- Matrix space is direct but can't leverage tensor cores

On GPU with tensor cores (V100/A100/H100), coefficient space should win.
On CPU, matrix space may be faster due to conversion overhead.
"""

import numpy as np
import time
from typing import Tuple

PHI = (1 + np.sqrt(5)) / 2
MATRIX_DIM = 4
DTYPE = np.float32


def get_xp():
    """Get array module."""
    try:
        import cupy as cp
        return cp, True
    except ImportError:
        return np, False


def benchmark_context_computation():
    """Benchmark the two context computation approaches."""
    from holographic_v4.algebra import (
        build_clifford_basis,
        geometric_product_batch_multi,
        geometric_product_batch_multi_coefficients,
        decompose_to_coefficients_batch,
        reconstruct_from_coefficients,
    )
    
    xp, has_gpu = get_xp()
    print(f"Running on: {'GPU' if has_gpu else 'CPU'}")
    print("=" * 60)
    
    basis = build_clifford_basis(xp)
    
    # Test parameters
    batch_size = 500
    seq_len = 8  # Context window
    n_iterations = 10
    
    print(f"Parameters: batch={batch_size}, seq_len={seq_len}, iterations={n_iterations}")
    print()
    
    # Create test data
    np.random.seed(42)
    # Matrix form: [BATCH, SEQ, 4, 4]
    batch_matrices = xp.array(
        np.random.randn(batch_size, seq_len, MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
    )
    
    # Coefficient form: [BATCH, SEQ, 16]
    batch_coefficients = decompose_to_coefficients_batch(
        batch_matrices.reshape(-1, MATRIX_DIM, MATRIX_DIM), 
        basis, xp
    ).reshape(batch_size, seq_len, 16)
    
    # Warmup
    _ = geometric_product_batch_multi(batch_matrices[:10], xp)
    _ = geometric_product_batch_multi_coefficients(batch_coefficients[:10], xp)
    
    if has_gpu:
        xp.cuda.Stream.null.synchronize()
    
    # Benchmark matrix-space
    print("Matrix-space pipeline (4×4 matmuls):")
    mat_times = []
    for i in range(n_iterations):
        if has_gpu:
            xp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        
        result_mat = geometric_product_batch_multi(batch_matrices, xp)
        
        if has_gpu:
            xp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        mat_times.append(elapsed)
    
    avg_mat = sum(mat_times[2:]) / len(mat_times[2:])  # Skip first 2 for warmup effects
    print(f"  Avg time: {avg_mat*1000:.2f}ms")
    print(f"  Throughput: {batch_size/avg_mat:.0f} contexts/sec")
    print(f"  Output shape: {result_mat.shape}")
    
    # Benchmark coefficient-space
    print()
    print("Coefficient-space pipeline (256×16 matmuls - tensor-core-friendly):")
    coeff_times = []
    for i in range(n_iterations):
        if has_gpu:
            xp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        
        result_coeff = geometric_product_batch_multi_coefficients(batch_coefficients, xp)
        
        if has_gpu:
            xp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        coeff_times.append(elapsed)
    
    avg_coeff = sum(coeff_times[2:]) / len(coeff_times[2:])
    print(f"  Avg time: {avg_coeff*1000:.2f}ms")
    print(f"  Throughput: {batch_size/avg_coeff:.0f} contexts/sec")
    print(f"  Output shape: {result_coeff.shape}")
    
    # Benchmark full pipeline with conversion
    print()
    print("Full coefficient pipeline (with decomposition + reconstruction):")
    full_times = []
    for i in range(n_iterations):
        if has_gpu:
            xp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        
        # Decompose to coefficients
        coeffs = decompose_to_coefficients_batch(
            batch_matrices.reshape(-1, MATRIX_DIM, MATRIX_DIM), basis, xp
        ).reshape(batch_size, seq_len, 16)
        
        # Compute in coefficient space
        result_coeff = geometric_product_batch_multi_coefficients(coeffs, xp)
        
        # Reconstruct to matrices - BATCHED (no Python loop!)
        # reconstruct_from_coefficients handles batch via einsum
        result_mat = reconstruct_from_coefficients(result_coeff, basis, xp)
        
        if has_gpu:
            xp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        full_times.append(elapsed)
    
    avg_full = sum(full_times[2:]) / len(full_times[2:])
    print(f"  Avg time: {avg_full*1000:.2f}ms")
    print(f"  Throughput: {batch_size/avg_full:.0f} contexts/sec")
    print(f"  Output shape: {result_mat.shape}")
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    coeff_speedup = avg_mat / avg_coeff
    full_speedup = avg_mat / avg_full
    
    print(f"Coefficient-only speedup: {coeff_speedup:.2f}×")
    print(f"Full pipeline speedup: {full_speedup:.2f}×")
    
    if coeff_speedup > 1.5:
        print("✓ Coefficient-space is significantly faster")
        print("  → Enable use_tensor_cores=True")
    elif coeff_speedup > 1.0:
        print("~ Coefficient-space is slightly faster")
    else:
        print("⚠ Matrix-space is faster (conversion overhead dominates)")
    
    if full_speedup < 1.0:
        print()
        print("NOTE: With conversion overhead, full pipeline is slower.")
        print("Consider keeping embeddings in coefficient space to avoid conversion.")
    
    return {
        'matrix_time': avg_mat,
        'coeff_time': avg_coeff,
        'full_time': avg_full,
        'coeff_speedup': coeff_speedup,
        'full_speedup': full_speedup,
    }


if __name__ == "__main__":
    benchmark_context_computation()
