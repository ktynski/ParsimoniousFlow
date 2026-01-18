"""
Hot Path GPU Audit

Systematically identifies GPU inefficiencies in the training loop.

AUDIT TARGETS:
1. Python list comprehensions that could be GPU operations
2. Imports inside functions (per-call overhead)
3. GPU→CPU synchronization points
4. Small kernel launches
5. Non-contiguous memory access patterns
"""

import numpy as np
import time

def get_xp():
    try:
        import cupy as cp
        return cp, True
    except ImportError:
        return np, False


def audit_batch_index_creation():
    """
    ISSUE: pipeline.py:966-969 uses Python list comprehension
    
    Current (slow):
        batch_indices = self.xp.array(
            [[t % self.vocab_size for t in tokens] for tokens in batch_tokens],
            dtype=self.xp.int64
        )
    
    This creates a Python list THEN converts to GPU array.
    
    SOLUTION: Pre-convert tokens to GPU array, then vectorized modulo
    """
    xp, has_gpu = get_xp()
    
    # Simulate batch_tokens
    batch_size = 2048
    context_size = 512
    vocab_size = 50257
    
    # Current approach: Python list comprehension
    np.random.seed(42)
    batch_tokens = [
        list(np.random.randint(0, vocab_size * 2, context_size))
        for _ in range(batch_size)
    ]
    
    # Time current approach
    start = time.perf_counter()
    for _ in range(10):
        batch_indices_slow = xp.array(
            [[t % vocab_size for t in tokens] for tokens in batch_tokens],
            dtype=xp.int64
        )
    slow_time = (time.perf_counter() - start) / 10
    
    # Optimized approach: Convert first, then vectorized modulo
    # Pre-convert to GPU array
    batch_tokens_array = xp.array(batch_tokens, dtype=xp.int64)
    
    start = time.perf_counter()
    for _ in range(10):
        batch_indices_fast = batch_tokens_array % vocab_size
    fast_time = (time.perf_counter() - start) / 10
    
    # Verify correctness
    if has_gpu:
        slow_cpu = batch_indices_slow.get()
        fast_cpu = batch_indices_fast.get()
    else:
        slow_cpu = batch_indices_slow
        fast_cpu = batch_indices_fast
    
    assert np.array_equal(slow_cpu, fast_cpu), "Results don't match!"
    
    speedup = slow_time / fast_time
    print(f"Batch index creation:")
    print(f"  Current (list comp): {slow_time*1000:.2f}ms")
    print(f"  Optimized (vectorized): {fast_time*1000:.2f}ms")
    print(f"  Speedup: {speedup:.1f}×")
    print()
    
    return speedup


def audit_import_overhead():
    """
    ISSUE: pipeline.py:949-953 imports inside compute_contexts_batch
    
    Imports inside functions add overhead per call.
    Should be module-level.
    """
    # Time import overhead
    n_iterations = 1000
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        from holographic_v4.algebra import geometric_product_batch_multi
    import_time = (time.perf_counter() - start) / n_iterations
    
    # Already imported - just function call
    from holographic_v4.algebra import geometric_product_batch_multi as gpbm
    
    print(f"Import overhead per call: {import_time*1000000:.2f}μs")
    print(f"  Over 2048 batches: {import_time * 2048 * 1000:.2f}ms")
    print()
    
    return import_time


def audit_grace_operator_efficiency():
    """
    ISSUE: grace_operator_batch reconstructs GRACE_SCALES_FLAT every call
    
    scales = xp.array(GRACE_SCALES_FLAT, dtype=DTYPE)
    
    This creates a new GPU array each call.
    Should be precomputed.
    """
    xp, has_gpu = get_xp()
    
    from holographic_v4.constants import GRACE_SCALES_FLAT, DTYPE
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(xp)
    batch = xp.array(np.random.randn(1000, 4, 4).astype(np.float32))
    
    # Current approach: create scales each call
    def grace_current(M, basis, xp):
        norm_sq = xp.sum(basis * basis, axis=(1, 2))
        coeffs = xp.einsum('cij,bij->bc', basis, M) / norm_sq
        scales = xp.array(GRACE_SCALES_FLAT, dtype=DTYPE)  # ALLOCATION!
        scaled_coeffs = coeffs * scales
        return xp.einsum('bc,cij->bij', scaled_coeffs, basis)
    
    # Optimized: precompute scales
    GRACE_SCALES_GPU = xp.array(GRACE_SCALES_FLAT, dtype=DTYPE)
    
    def grace_optimized(M, basis, xp, scales_gpu):
        norm_sq = xp.sum(basis * basis, axis=(1, 2))
        coeffs = xp.einsum('cij,bij->bc', basis, M) / norm_sq
        scaled_coeffs = coeffs * scales_gpu  # NO ALLOCATION
        return xp.einsum('bc,cij->bij', scaled_coeffs, basis)
    
    # Warmup
    _ = grace_current(batch[:10], basis, xp)
    _ = grace_optimized(batch[:10], basis, xp, GRACE_SCALES_GPU)
    
    if has_gpu:
        xp.cuda.Stream.null.synchronize()
    
    # Time
    n_iter = 100
    
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = grace_current(batch, basis, xp)
    if has_gpu:
        xp.cuda.Stream.null.synchronize()
    current_time = (time.perf_counter() - start) / n_iter
    
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = grace_optimized(batch, basis, xp, GRACE_SCALES_GPU)
    if has_gpu:
        xp.cuda.Stream.null.synchronize()
    opt_time = (time.perf_counter() - start) / n_iter
    
    speedup = current_time / opt_time
    print(f"Grace operator efficiency:")
    print(f"  Current (alloc per call): {current_time*1000:.3f}ms")
    print(f"  Optimized (precomputed scales): {opt_time*1000:.3f}ms")
    print(f"  Speedup: {speedup:.2f}×")
    print()
    
    return speedup


def audit_vorticity_computation():
    """
    Check if vorticity computation has redundant operations.
    """
    xp, has_gpu = get_xp()
    
    from holographic_v4.algebra import (
        build_clifford_basis,
        vorticity_magnitude_and_signature_batch
    )
    
    basis = build_clifford_basis(xp)
    batch = xp.array(np.random.randn(1000, 8, 4, 4).astype(np.float32))  # [batch, seq, 4, 4]
    
    # Time current implementation
    if has_gpu:
        xp.cuda.Stream.null.synchronize()
    
    start = time.perf_counter()
    for _ in range(50):
        mags, sigs = vorticity_magnitude_and_signature_batch(batch, basis, xp)
    if has_gpu:
        xp.cuda.Stream.null.synchronize()
    elapsed = (time.perf_counter() - start) / 50
    
    print(f"Vorticity computation:")
    print(f"  Time for batch=1000, seq=8: {elapsed*1000:.2f}ms")
    print(f"  Output shapes: mags={mags.shape}, sigs={sigs.shape}")
    print()
    
    return elapsed


def run_full_audit():
    """Run complete hot path audit."""
    xp, has_gpu = get_xp()
    print(f"Running GPU Hot Path Audit")
    print(f"Device: {'GPU (CuPy)' if has_gpu else 'CPU (NumPy)'}")
    print("=" * 60)
    print()
    
    results = {}
    
    print("1. BATCH INDEX CREATION")
    print("-" * 40)
    results['batch_index_speedup'] = audit_batch_index_creation()
    
    print("2. IMPORT OVERHEAD")
    print("-" * 40)
    results['import_overhead'] = audit_import_overhead()
    
    print("3. GRACE OPERATOR EFFICIENCY")
    print("-" * 40)
    results['grace_speedup'] = audit_grace_operator_efficiency()
    
    print("4. VORTICITY COMPUTATION")
    print("-" * 40)
    results['vorticity_time'] = audit_vorticity_computation()
    
    print("=" * 60)
    print("SUMMARY - RECOMMENDED FIXES")
    print("=" * 60)
    
    if results['batch_index_speedup'] > 1.5:
        print("✗ Fix batch index creation: Use pre-converted GPU arrays")
    else:
        print("✓ Batch index creation is efficient")
    
    if results['import_overhead'] > 0.00001:  # > 10μs
        print("✗ Move imports to module level (avoid per-call overhead)")
    else:
        print("✓ Import overhead is minimal")
    
    if results['grace_speedup'] > 1.1:
        print("✗ Precompute GRACE_SCALES as GPU array")
    else:
        print("✓ Grace operator is efficient")
    
    return results


if __name__ == "__main__":
    run_full_audit()
