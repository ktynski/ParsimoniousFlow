"""
Benchmark: Torus-Aware vs Standard Witness Index

Measures the information efficiency gains from torus symmetry exploitation.

WHAT WE'RE MEASURING:
1. Retrieval accuracy (does bireflection help?)
2. Storage efficiency (canonical form reduces duplication?)
3. Throat-based priority (do near-throat patterns help generalization?)
"""

import numpy as np
import time
from typing import Tuple

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
MATRIX_DIM = 4
DTYPE = np.float32


def get_xp():
    """Get array module."""
    try:
        import cupy as cp
        return cp, True
    except ImportError:
        return np, False


def create_test_batch(batch_size: int, xp, seed=42) -> Tuple:
    """Create test batch with reproducible seed."""
    np.random.seed(seed)
    contexts = xp.array(np.random.randn(batch_size, MATRIX_DIM, MATRIX_DIM).astype(DTYPE))
    targets = xp.array(np.random.randn(batch_size, MATRIX_DIM, MATRIX_DIM).astype(DTYPE))
    target_idxs = xp.arange(batch_size, dtype=xp.int32)
    return contexts, targets, target_idxs


def run_benchmark():
    """Run torus symmetry benchmark."""
    from holographic_v4.torus_symmetry import TorusAwareWitnessIndex
    from holographic_v4.gpu_witness_index import GPUWitnessIndex
    from holographic_v4.algebra import build_clifford_basis
    
    xp, has_gpu = get_xp()
    print(f"Running on: {'GPU' if has_gpu else 'CPU'}")
    print("=" * 60)
    
    # Create basis
    basis = build_clifford_basis(xp)
    
    # Test parameters
    n_store = 1000
    n_query = 200
    
    print(f"Storing {n_store} patterns, querying {n_query}")
    print()
    
    # Create indices
    torus_index = TorusAwareWitnessIndex.create(basis, max_items=10000, xp=xp)
    standard_index = GPUWitnessIndex.create(basis, max_items=10000, xp=xp)
    
    # Store same patterns in both
    contexts, targets, target_idxs = create_test_batch(n_store, xp, seed=42)
    
    torus_index.store_batch(contexts, targets, target_idxs)
    standard_index.store_batch(contexts, targets, target_idxs)
    
    # Create query set (mix of stored and novel patterns)
    # Use same seed for first half (stored patterns), different for second (novel)
    stored_queries, _, _ = create_test_batch(n_query // 2, xp, seed=42)
    novel_queries, _, _ = create_test_batch(n_query // 2, xp, seed=999)
    
    # Combine: first half are stored patterns, second half are novel
    query_contexts = xp.concatenate([stored_queries[:n_query//4 * 3:3], novel_queries], axis=0)
    actual_n_query = query_contexts.shape[0]
    
    print(f"Query set: {actual_n_query} patterns")
    
    # Benchmark retrieval
    print()
    print("Retrieval Performance:")
    print("-" * 40)
    
    # Torus-aware retrieval
    start = time.perf_counter()
    t_targets, t_idxs, t_conf = torus_index.retrieve_batch(query_contexts)
    t_time = time.perf_counter() - start
    
    # Standard retrieval  
    start = time.perf_counter()
    s_targets, s_idxs, s_conf = standard_index.retrieve_batch(query_contexts)
    s_time = time.perf_counter() - start
    
    # Convert to CPU for analysis
    if has_gpu:
        t_conf_cpu = t_conf.get()
        s_conf_cpu = s_conf.get()
        t_idxs_cpu = t_idxs.get()
        s_idxs_cpu = s_idxs.get()
    else:
        t_conf_cpu = t_conf
        s_conf_cpu = s_conf
        t_idxs_cpu = t_idxs
        s_idxs_cpu = s_idxs
    
    # Metrics
    t_matches = sum(1 for c in t_conf_cpu if c > 0.1)
    s_matches = sum(1 for c in s_conf_cpu if c > 0.1)
    
    t_high_conf = sum(1 for c in t_conf_cpu if c > 0.5)
    s_high_conf = sum(1 for c in s_conf_cpu if c > 0.5)
    
    print(f"Torus-Aware Index:")
    print(f"  Time: {t_time*1000:.2f}ms")
    print(f"  Matches (conf > 0.1): {t_matches}/{actual_n_query}")
    print(f"  High-confidence (> 0.5): {t_high_conf}/{actual_n_query}")
    print(f"  Mean confidence: {np.mean(t_conf_cpu):.4f}")
    
    print()
    print(f"Standard Index:")
    print(f"  Time: {s_time*1000:.2f}ms")
    print(f"  Matches (conf > 0.1): {s_matches}/{actual_n_query}")
    print(f"  High-confidence (> 0.5): {s_high_conf}/{actual_n_query}")
    print(f"  Mean confidence: {np.mean(s_conf_cpu):.4f}")
    
    # Index stats
    print()
    print("Index Statistics:")
    print("-" * 40)
    
    t_stats = torus_index.get_stats()
    s_stats = standard_index.get_stats()
    
    print(f"Torus-Aware:")
    print(f"  Items stored: {t_stats['n_items']}")
    if 'near_throat_fraction' in t_stats:
        print(f"  Near throat (σ ≈ 0.5): {t_stats['near_throat_fraction']*100:.1f}%")
        print(f"  Mean σ: {t_stats['mean_sigma']:.3f}")
        print(f"  Bireflected: {t_stats['bireflected_count']}")
    
    print()
    print(f"Standard:")
    print(f"  Items stored: {s_stats['n_items']}")
    
    # Comparison summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    match_improvement = (t_matches - s_matches) / max(s_matches, 1) * 100
    conf_improvement = (np.mean(t_conf_cpu) - np.mean(s_conf_cpu)) / max(np.mean(s_conf_cpu), 0.01) * 100
    
    print(f"Match improvement: {match_improvement:+.1f}%")
    print(f"Confidence improvement: {conf_improvement:+.1f}%")
    
    if t_matches > s_matches:
        print("✓ Torus symmetry improves retrieval coverage")
    elif t_matches == s_matches:
        print("~ Equal retrieval coverage")
    else:
        print("⚠ Standard index has more matches")
    
    if np.mean(t_conf_cpu) > np.mean(s_conf_cpu):
        print("✓ Torus symmetry improves confidence")
    else:
        print("~ Confidence similar or lower")


if __name__ == "__main__":
    run_benchmark()
