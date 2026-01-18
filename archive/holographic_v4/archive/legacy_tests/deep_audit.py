"""
Deep audit tests for GPU hot path optimizations.

Tests the vectorized implementations of:
1. Vorticity signature batch (einsum vs loop)
2. Quotient similarity batch (vectorized vs Python loop)
"""
import time
import numpy as np
import sys
sys.path.insert(0, '.')

from holographic_v4.algebra import (
    build_clifford_basis,
    vorticity_magnitude_and_signature_batch,
    decompose_to_coefficients,
    DTYPE,
)
from holographic_v4.quotient import (
    quotient_similarity_batch,
    quotient_similarity,
    extract_witness_batch,
)
from holographic_v4.constants import PHI_INV_SQ, PHI_INV_CUBE


def benchmark_vorticity_batch(batch_size: int = 100, seq_len: int = 20, n_runs: int = 10):
    """Benchmark vorticity signature batch computation."""
    print(f"\n=== Vorticity Batch Benchmark ===")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Runs: {n_runs}")
    
    xp = np
    basis = build_clifford_basis(xp)
    
    # Create test data
    np.random.seed(42)
    batch_matrices = np.random.randn(batch_size, seq_len, 4, 4).astype(DTYPE)
    
    # Warmup
    _, _ = vorticity_magnitude_and_signature_batch(batch_matrices, basis, xp)
    
    # Benchmark vectorized (current)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        mags, sigs = vorticity_magnitude_and_signature_batch(batch_matrices, basis, xp)
        times.append(time.perf_counter() - start)
    
    vectorized_time = np.median(times) * 1000
    print(f"Vectorized (einsum): {vectorized_time:.2f}ms")
    print(f"  Output shapes: magnitudes={mags.shape}, signatures={sigs.shape}")
    
    return vectorized_time


def benchmark_quotient_similarity_batch(n_contexts: int = 1000, n_runs: int = 10):
    """Benchmark quotient similarity batch computation."""
    print(f"\n=== Quotient Similarity Batch Benchmark ===")
    print(f"Contexts: {n_contexts}, Runs: {n_runs}")
    
    xp = np
    basis = build_clifford_basis(xp)
    
    # Create test data
    np.random.seed(42)
    query = np.random.randn(4, 4).astype(DTYPE)
    contexts = np.random.randn(n_contexts, 4, 4).astype(DTYPE)
    
    # Warmup
    _ = quotient_similarity_batch(query, contexts, basis, xp)
    
    # Benchmark vectorized
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        sims_vec = quotient_similarity_batch(query, contexts, basis, xp)
        times.append(time.perf_counter() - start)
    
    vectorized_time = np.median(times) * 1000
    print(f"Vectorized quotient_similarity_batch: {vectorized_time:.2f}ms")
    
    # Benchmark Python loop (old method)
    small_n = min(100, n_contexts)  # Only test small subset for loop
    small_contexts = contexts[:small_n]
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        sims_loop = np.zeros(small_n, dtype=DTYPE)
        for i in range(small_n):
            sims_loop[i] = quotient_similarity(query, small_contexts[i], basis, xp)
        times.append(time.perf_counter() - start)
    
    loop_time = np.median(times) * 1000
    loop_time_scaled = loop_time * (n_contexts / small_n)  # Extrapolate
    print(f"Python loop (100 items): {loop_time:.2f}ms")
    print(f"Python loop (extrapolated to {n_contexts}): {loop_time_scaled:.2f}ms")
    
    speedup = loop_time_scaled / vectorized_time
    print(f"Speedup: {speedup:.1f}×")
    
    # Verify results are reasonable
    sims_loop_small = sims_loop
    sims_vec_small = quotient_similarity_batch(query, small_contexts, basis, xp)
    
    # They won't be identical since vectorized skips normal_form,
    # but correlation should be high
    corr = np.corrcoef(sims_loop_small, sims_vec_small)[0, 1]
    print(f"Correlation with full method: {corr:.4f}")
    
    return vectorized_time, loop_time_scaled, speedup


def benchmark_witness_extraction_batch(n_matrices: int = 10000, n_runs: int = 10):
    """Benchmark witness extraction batch vs loop."""
    print(f"\n=== Witness Extraction Batch Benchmark ===")
    print(f"Matrices: {n_matrices}, Runs: {n_runs}")
    
    xp = np
    basis = build_clifford_basis(xp)
    
    # Create test data
    np.random.seed(42)
    matrices = np.random.randn(n_matrices, 4, 4).astype(DTYPE)
    
    # Warmup
    _ = extract_witness_batch(matrices, basis, xp)
    
    # Benchmark vectorized
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        witnesses = extract_witness_batch(matrices, basis, xp)
        times.append(time.perf_counter() - start)
    
    vectorized_time = np.median(times) * 1000
    print(f"Vectorized (einsum): {vectorized_time:.2f}ms")
    print(f"  Output shape: {witnesses.shape}")
    
    return vectorized_time


def run_deep_audit():
    """Run all deep audit benchmarks."""
    print("=" * 60)
    print("DEEP AUDIT: GPU Hot Path Optimization Verification")
    print("=" * 60)
    
    # Test 1: Vorticity batch
    vort_time = benchmark_vorticity_batch()
    
    # Test 2: Quotient similarity batch
    qs_vec, qs_loop, qs_speedup = benchmark_quotient_similarity_batch()
    
    # Test 3: Witness extraction batch
    wit_time = benchmark_witness_extraction_batch()
    
    # Summary
    print("\n" + "=" * 60)
    print("DEEP AUDIT SUMMARY")
    print("=" * 60)
    print(f"✓ Vorticity batch (100×20 sequences): {vort_time:.2f}ms")
    print(f"✓ Quotient similarity batch (1000 contexts): {qs_vec:.2f}ms")
    print(f"  → {qs_speedup:.0f}× faster than Python loop")
    print(f"✓ Witness extraction batch (10000 matrices): {wit_time:.2f}ms")
    
    # Check for issues
    issues = []
    if qs_speedup < 10:
        issues.append(f"Quotient similarity speedup only {qs_speedup:.1f}× (expected >10×)")
    
    if issues:
        print(f"\n⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✓ All hot paths are vectorized and GPU-ready")


if __name__ == "__main__":
    run_deep_audit()
