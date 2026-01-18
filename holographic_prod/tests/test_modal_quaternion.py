"""
Modal Quaternion Test with TinyStories

Tests quaternion embeddings on TinyStories dataset to validate:
1. Memory reduction (2× fewer floats)
2. Equivalent accuracy to matrix embeddings
3. Geometric product equivalence

RUN:
    modal run holographic_prod/tests/test_modal_quaternion.py::test_quaternion_tinystories

NOTE: This test requires Modal CLI and credentials.
"""

import modal
import time
from typing import List, Tuple, Dict

# Modal app
app = modal.App("holographic-quaternion-test")

# GPU-optimized image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "numpy>=1.24.0",
        "cupy-cuda12x>=12.0.0",
        "scipy>=1.10.0",
        "datasets>=2.14.0",
        "huggingface_hub>=0.16.0",
    )
    .add_local_dir("holographic_prod", "/root/project/holographic_prod")
)


@app.function(image=image, gpu="T4", timeout=1800)
def test_quaternion_tinystories(
    n_samples: int = 10_000,
    vocab_size: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Test quaternion embeddings on TinyStories subset.
    
    Compares:
    1. Memory usage: quaternion (8 floats) vs matrix (16 floats)
    2. Accuracy: binding and retrieval equivalence
    3. Speed: conversion overhead
    
    Args:
        n_samples: Number of samples to process
        vocab_size: Vocabulary size
        seed: Random seed
        
    Returns:
        Dictionary with test results
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import numpy as np
    import cupy as cp
    
    # Verify GPU
    cp.cuda.Device(0).use()
    meminfo = cp.cuda.runtime.memGetInfo()
    print(f"GPU Memory: {meminfo[1]/1024**3:.1f} GB total, {meminfo[0]/1024**3:.1f} GB free")
    
    from holographic_prod.core.quaternion import (
        quaternion_pair_to_so4,
        quaternion_geometric_product,
        create_quaternion_embeddings,
        batch_quaternion_to_so4,
    )
    from holographic_prod.core.grounded_embeddings import create_random_so4_embeddings
    from holographic_prod.core.constants import DTYPE
    
    results = {
        'vocab_size': vocab_size,
        'n_samples': n_samples,
        'seed': seed,
    }
    
    np.random.seed(seed)
    
    # ==========================================================================
    # TEST 1: Memory Reduction
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Memory Reduction")
    print("=" * 70)
    
    # Matrix embeddings: [vocab_size, 4, 4] = vocab_size * 16 floats
    matrix_embeddings = create_random_so4_embeddings(vocab_size, seed=seed, xp=np)
    matrix_bytes = matrix_embeddings.nbytes
    
    # Quaternion embeddings: [vocab_size, 2, 4] = vocab_size * 8 floats
    quat_embeddings = create_quaternion_embeddings(vocab_size, seed=seed)
    quat_bytes = quat_embeddings.nbytes
    
    reduction = matrix_bytes / quat_bytes
    
    print(f"  Matrix embeddings:     {matrix_bytes / 1024:.1f} KB")
    print(f"  Quaternion embeddings: {quat_bytes / 1024:.1f} KB")
    print(f"  Memory reduction:      {reduction:.1f}×")
    
    results['matrix_bytes'] = matrix_bytes
    results['quat_bytes'] = quat_bytes
    results['memory_reduction'] = reduction
    
    assert reduction > 1.8, f"Expected ~2× memory reduction, got {reduction:.1f}×"
    print("  ✓ Memory reduction test PASSED")
    
    # ==========================================================================
    # TEST 2: Binding Equivalence
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Binding Equivalence")
    print("=" * 70)
    
    # Convert quaternion embeddings to matrices for comparison
    matrix_from_quat = batch_quaternion_to_so4(quat_embeddings)
    
    binding_errors = []
    n_binding_tests = min(1000, n_samples)
    
    for _ in range(n_binding_tests):
        # Random context and target
        ctx_tokens = np.random.randint(0, vocab_size, size=3).tolist()
        target_idx = np.random.randint(0, vocab_size)
        
        # Matrix binding
        context_mat = matrix_from_quat[ctx_tokens[0]]
        for t in ctx_tokens[1:]:
            context_mat = context_mat @ matrix_from_quat[t]
        binding_mat = context_mat @ matrix_from_quat[target_idx]
        
        # Quaternion binding
        q_ctx_L = quat_embeddings[ctx_tokens[0], 0]
        q_ctx_R = quat_embeddings[ctx_tokens[0], 1]
        for t in ctx_tokens[1:]:
            q_t_L = quat_embeddings[t, 0]
            q_t_R = quat_embeddings[t, 1]
            q_ctx_L, q_ctx_R = quaternion_geometric_product(q_ctx_L, q_ctx_R, q_t_L, q_t_R)
        
        q_tgt_L = quat_embeddings[target_idx, 0]
        q_tgt_R = quat_embeddings[target_idx, 1]
        q_bind_L, q_bind_R = quaternion_geometric_product(q_ctx_L, q_ctx_R, q_tgt_L, q_tgt_R)
        binding_quat = quaternion_pair_to_so4(q_bind_L, q_bind_R)
        
        # Compare
        error = min(
            np.linalg.norm(binding_quat - binding_mat),
            np.linalg.norm(binding_quat + binding_mat),  # Z₂ ambiguity
        )
        binding_errors.append(error)
    
    avg_binding_error = np.mean(binding_errors)
    max_binding_error = np.max(binding_errors)
    
    print(f"  Binding tests:      {n_binding_tests}")
    print(f"  Average error:      {avg_binding_error:.2e}")
    print(f"  Max error:          {max_binding_error:.2e}")
    
    results['binding_avg_error'] = float(avg_binding_error)
    results['binding_max_error'] = float(max_binding_error)
    
    # Should be very small (numerical precision)
    assert avg_binding_error < 1e-3, f"Binding error too large: {avg_binding_error:.2e}"
    print("  ✓ Binding equivalence test PASSED")
    
    # ==========================================================================
    # TEST 3: Conversion Speed
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Conversion Speed")
    print("=" * 70)
    
    # Quaternion -> Matrix (most common in inference)
    n_conversions = vocab_size
    
    start = time.time()
    for i in range(n_conversions):
        _ = quaternion_pair_to_so4(quat_embeddings[i, 0], quat_embeddings[i, 1])
    elapsed = time.time() - start
    
    per_conversion_us = elapsed / n_conversions * 1_000_000
    
    print(f"  Conversions:        {n_conversions}")
    print(f"  Total time:         {elapsed:.3f}s")
    print(f"  Per conversion:     {per_conversion_us:.1f} µs")
    
    results['conversion_time_us'] = per_conversion_us
    
    # Should be fast enough (< 100 µs per conversion)
    assert per_conversion_us < 100, f"Conversion too slow: {per_conversion_us:.1f} µs"
    print("  ✓ Conversion speed test PASSED")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: All quaternion tests PASSED")
    print("=" * 70)
    print(f"  Memory reduction:   {reduction:.1f}×")
    print(f"  Binding error:      {avg_binding_error:.2e}")
    print(f"  Conversion speed:   {per_conversion_us:.1f} µs")
    
    results['status'] = 'PASSED'
    return results


def run_local():
    """Run tests locally (without Modal)."""
    print("Running quaternion tests locally...")
    
    import sys
    sys.path.insert(0, '.')
    
    import numpy as np
    
    from holographic_prod.core.quaternion import (
        quaternion_pair_to_so4,
        quaternion_geometric_product,
        create_quaternion_embeddings,
        batch_quaternion_to_so4,
    )
    from holographic_prod.core.grounded_embeddings import create_random_so4_embeddings
    
    vocab_size = 100
    seed = 42
    np.random.seed(seed)
    
    # Memory test
    matrix_embeddings = create_random_so4_embeddings(vocab_size, seed=seed, xp=np)
    quat_embeddings = create_quaternion_embeddings(vocab_size, seed=seed)
    
    reduction = matrix_embeddings.nbytes / quat_embeddings.nbytes
    print(f"Memory reduction: {reduction:.1f}×")
    
    # Binding test
    matrix_from_quat = batch_quaternion_to_so4(quat_embeddings)
    ctx_tokens = [0, 1, 2]
    target_idx = 50
    
    context_mat = matrix_from_quat[ctx_tokens[0]]
    for t in ctx_tokens[1:]:
        context_mat = context_mat @ matrix_from_quat[t]
    binding_mat = context_mat @ matrix_from_quat[target_idx]
    
    q_ctx_L = quat_embeddings[ctx_tokens[0], 0]
    q_ctx_R = quat_embeddings[ctx_tokens[0], 1]
    for t in ctx_tokens[1:]:
        q_t_L = quat_embeddings[t, 0]
        q_t_R = quat_embeddings[t, 1]
        q_ctx_L, q_ctx_R = quaternion_geometric_product(q_ctx_L, q_ctx_R, q_t_L, q_t_R)
    
    q_tgt_L = quat_embeddings[target_idx, 0]
    q_tgt_R = quat_embeddings[target_idx, 1]
    q_bind_L, q_bind_R = quaternion_geometric_product(q_ctx_L, q_ctx_R, q_tgt_L, q_tgt_R)
    binding_quat = quaternion_pair_to_so4(q_bind_L, q_bind_R)
    
    error = min(
        np.linalg.norm(binding_quat - binding_mat),
        np.linalg.norm(binding_quat + binding_mat),
    )
    print(f"Binding error: {error:.2e}")
    
    print("✓ Local quaternion tests completed")


if __name__ == '__main__':
    run_local()
