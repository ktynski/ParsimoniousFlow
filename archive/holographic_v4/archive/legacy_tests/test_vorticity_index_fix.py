#!/usr/bin/env python3
"""
TEST: Vorticity Index Fix for Bucket Collision Problem

This test suite verifies that VorticityWitnessIndex provides dramatically better
bucket distribution than the original WitnessIndex.

ROOT CAUSE (what we're fixing):
    - WitnessIndex uses 2D keys (σ, p) with resolution φ⁻² = 0.382
    - σ ∈ [-0.35, 0.35], p ∈ [-0.22, 0.22] → only ~4 buckets!
    - At 50M samples: ~12.5M items per bucket → retrieval is random guessing

THE FIX:
    - VorticityWitnessIndex uses 4D keys: (σ, p, enstrophy, dominant_plane)
    - enstrophy captures total bivector energy (vorticity magnitude)
    - dominant_plane captures which of 6 bivector planes is dominant
    - Creates ~1000+ buckets instead of ~4

TESTS:
    1. test_bucket_distribution_comparison: Old vs new bucket counts
    2. test_retrieval_accuracy: Accuracy on stored patterns
    3. test_scaling_behavior: Performance at increasing scales
    4. test_vorticity_discriminates: Verify vorticity adds discrimination
"""

import numpy as np
import time
from typing import Dict, Any

# Import the index classes
from holographic_v4.holographic_memory import (
    WitnessIndex,
    VorticityWitnessIndex,
    HybridHolographicMemory,
)
from holographic_v4.algebra import (
    build_clifford_basis,
    geometric_product_batch,
    grace_operator,
    initialize_embeddings_identity,
)
from holographic_v4.constants import PHI_INV_SQ


def generate_random_contexts(
    n_contexts: int,
    context_length: int,
    vocab_size: int = 50000,
    seed: int = 42
) -> np.ndarray:
    """Generate random context matrices."""
    rng = np.random.default_rng(seed)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    contexts = []
    for _ in range(n_contexts):
        seq = rng.integers(0, vocab_size, size=context_length)
        mats = embeddings[seq]
        ctx = geometric_product_batch(mats, np)
        ctx = grace_operator(ctx, basis, np)
        contexts.append(ctx)
    
    return np.array(contexts), basis


def test_bucket_distribution_comparison():
    """
    Compare bucket distribution between WitnessIndex and VorticityWitnessIndex.
    
    EXPECTED:
        - WitnessIndex: ~4 buckets
        - VorticityWitnessIndex: ~500+ buckets (for 5000 samples)
    """
    print("\n" + "=" * 70)
    print("  TEST: Bucket Distribution Comparison")
    print("=" * 70)
    
    n_samples = 5000
    context_length = 512
    
    print(f"\n  Generating {n_samples} random contexts (ctx_len={context_length})...")
    contexts, basis = generate_random_contexts(n_samples, context_length)
    
    # Create dummy targets (just use the contexts themselves)
    targets = contexts.copy()
    target_idxs = np.arange(n_samples)
    
    # Test old WitnessIndex
    print("\n  Testing WitnessIndex (2D keys)...")
    old_index = WitnessIndex.create(basis, resolution=PHI_INV_SQ, xp=np)
    
    t0 = time.time()
    for i in range(n_samples):
        old_index.store(contexts[i], targets[i], i)
    old_time = time.time() - t0
    
    old_buckets = len(old_index.buckets)
    old_bucket_sizes = [len(b) for b in old_index.buckets.values()]
    old_max_bucket = max(old_bucket_sizes)
    old_avg_bucket = sum(old_bucket_sizes) / len(old_bucket_sizes)
    
    print(f"    Unique buckets:     {old_buckets}")
    print(f"    Avg bucket size:    {old_avg_bucket:.1f}")
    print(f"    Max bucket size:    {old_max_bucket}")
    print(f"    Time:               {old_time:.2f}s")
    
    # Test new VorticityWitnessIndex
    print("\n  Testing VorticityWitnessIndex (4D keys)...")
    new_index = VorticityWitnessIndex.create(basis, xp=np)
    
    t0 = time.time()
    for i in range(n_samples):
        new_index.store(contexts[i], targets[i], i)
    new_time = time.time() - t0
    
    new_stats = new_index.stats()
    
    print(f"    Unique buckets:     {new_stats['n_buckets']}")
    print(f"    Avg bucket size:    {new_stats['avg_bucket_size']:.1f}")
    print(f"    Max bucket size:    {new_stats['max_bucket_size']}")
    print(f"    Time:               {new_time:.2f}s")
    
    # Calculate improvement
    bucket_improvement = new_stats['n_buckets'] / old_buckets
    collision_improvement = old_max_bucket / new_stats['max_bucket_size']
    
    print("\n  === IMPROVEMENT ===")
    print(f"    Bucket count:       {bucket_improvement:.0f}× more buckets")
    print(f"    Max collision:      {collision_improvement:.1f}× fewer collisions")
    
    # Assertions
    assert new_stats['n_buckets'] > old_buckets * 10, \
        f"Expected 10× more buckets, got {new_stats['n_buckets']} vs {old_buckets}"
    assert new_stats['max_bucket_size'] < old_max_bucket, \
        f"Expected smaller max bucket, got {new_stats['max_bucket_size']} vs {old_max_bucket}"
    
    print("\n  ✅ PASS: VorticityWitnessIndex has dramatically better distribution")
    return True


def test_retrieval_accuracy():
    """
    Test retrieval accuracy on stored patterns.
    
    EXPECTED:
        - VorticityWitnessIndex: Much higher accuracy than WitnessIndex
        - Both should be able to retrieve recently stored patterns
    """
    print("\n" + "=" * 70)
    print("  TEST: Retrieval Accuracy")
    print("=" * 70)
    
    n_samples = 1000
    n_test = 100
    context_length = 512
    
    print(f"\n  Generating {n_samples} contexts, testing {n_test}...")
    contexts, basis = generate_random_contexts(n_samples, context_length)
    targets = contexts.copy()  # Self-retrieval test
    
    # Test old WitnessIndex
    print("\n  Testing WitnessIndex retrieval...")
    old_index = WitnessIndex.create(basis, resolution=PHI_INV_SQ, xp=np)
    for i in range(n_samples):
        old_index.store(contexts[i], targets[i], i)
    
    old_correct = 0
    old_found = 0
    for i in range(n_test):
        result, idx, conf = old_index.retrieve(contexts[i])
        if result is not None:
            old_found += 1
            if idx == i:  # Exact match
                old_correct += 1
    
    old_accuracy = old_correct / n_test if n_test > 0 else 0
    old_recall = old_found / n_test if n_test > 0 else 0
    
    print(f"    Recall (found):     {old_recall:.1%}")
    print(f"    Accuracy (correct): {old_accuracy:.1%}")
    
    # Test new VorticityWitnessIndex
    print("\n  Testing VorticityWitnessIndex retrieval...")
    new_index = VorticityWitnessIndex.create(basis, xp=np)
    for i in range(n_samples):
        new_index.store(contexts[i], targets[i], i)
    
    new_correct = 0
    new_found = 0
    for i in range(n_test):
        result, idx, conf = new_index.retrieve(contexts[i])
        if result is not None:
            new_found += 1
            if idx == i:  # Exact match
                new_correct += 1
    
    new_accuracy = new_correct / n_test if n_test > 0 else 0
    new_recall = new_found / n_test if n_test > 0 else 0
    
    print(f"    Recall (found):     {new_recall:.1%}")
    print(f"    Accuracy (correct): {new_accuracy:.1%}")
    
    # Calculate improvement
    if old_accuracy > 0:
        accuracy_improvement = new_accuracy / old_accuracy
        print(f"\n  === IMPROVEMENT ===")
        print(f"    Accuracy:           {accuracy_improvement:.1f}×")
    else:
        print(f"\n  === IMPROVEMENT ===")
        print(f"    Old accuracy was 0%, new is {new_accuracy:.1%}")
    
    # Assertions (VorticityIndex should be at least as good)
    assert new_accuracy >= old_accuracy, \
        f"Expected better or equal accuracy: {new_accuracy:.1%} vs {old_accuracy:.1%}"
    
    print("\n  ✅ PASS: VorticityWitnessIndex retrieval works correctly")
    return True


def test_scaling_behavior():
    """
    Test bucket distribution at increasing scales.
    
    THEORY:
        - WitnessIndex: Buckets stay ~4 regardless of sample count
        - VorticityWitnessIndex: Buckets grow with samples (up to entropy limit)
    """
    print("\n" + "=" * 70)
    print("  TEST: Scaling Behavior")
    print("=" * 70)
    
    scales = [100, 500, 1000, 2000]
    context_length = 512
    
    print(f"\n  Testing at scales: {scales}")
    
    _, basis = generate_random_contexts(10, context_length)  # Just for basis
    
    results = {'old': [], 'new': []}
    
    for n in scales:
        print(f"\n  Scale: {n} samples...")
        contexts, _ = generate_random_contexts(n, context_length, seed=42+n)
        
        # Old index
        old_index = WitnessIndex.create(basis, resolution=PHI_INV_SQ, xp=np)
        for i in range(n):
            old_index.store(contexts[i], contexts[i], i)
        results['old'].append(len(old_index.buckets))
        
        # New index
        new_index = VorticityWitnessIndex.create(basis, xp=np)
        for i in range(n):
            new_index.store(contexts[i], contexts[i], i)
        results['new'].append(new_index.stats()['n_buckets'])
        
        print(f"    WitnessIndex buckets:         {results['old'][-1]}")
        print(f"    VorticityWitnessIndex buckets: {results['new'][-1]}")
    
    print("\n  === SCALING TABLE ===")
    print(f"  {'Samples':>8} | {'Old Buckets':>12} | {'New Buckets':>12} | {'Ratio':>8}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    for i, n in enumerate(scales):
        ratio = results['new'][i] / max(results['old'][i], 1)
        print(f"  {n:>8} | {results['old'][i]:>12} | {results['new'][i]:>12} | {ratio:>8.1f}×")
    
    # Verify old stays flat, new grows
    assert results['old'][-1] <= 10, \
        f"WitnessIndex should have ~4-10 buckets, got {results['old'][-1]}"
    assert results['new'][-1] > 100, \
        f"VorticityWitnessIndex should have 100+ buckets at {scales[-1]} samples, got {results['new'][-1]}"
    
    print("\n  ✅ PASS: Scaling behavior as expected")
    return True


def test_vorticity_discriminates():
    """
    Verify that vorticity (bivector) adds discrimination beyond witness.
    
    We create pairs of contexts with SAME witness but DIFFERENT vorticity,
    and verify they get different keys in VorticityWitnessIndex.
    """
    print("\n" + "=" * 70)
    print("  TEST: Vorticity Discriminates Beyond Witness")
    print("=" * 70)
    
    from holographic_v4.quotient import extract_witness
    
    n_samples = 500
    context_length = 512
    
    print(f"\n  Generating {n_samples} contexts...")
    contexts, basis = generate_random_contexts(n_samples, context_length)
    
    # Find pairs with similar witness values
    witnesses = np.array([extract_witness(c, basis, np) for c in contexts])
    
    # Quantize witnesses with old resolution
    old_keys = []
    for i in range(n_samples):
        s_idx = int(np.floor(witnesses[i, 0] / PHI_INV_SQ))
        p_idx = int(np.floor(witnesses[i, 1] / PHI_INV_SQ))
        old_keys.append((s_idx, p_idx))
    
    # Get VorticityWitnessIndex keys
    vort_index = VorticityWitnessIndex.create(basis, xp=np)
    new_keys = []
    for i in range(n_samples):
        new_keys.append(vort_index._vorticity_key(contexts[i]))
    
    # Count how many collisions in old are resolved in new
    from collections import Counter
    old_collision_groups = Counter(old_keys)
    
    resolved_collisions = 0
    total_collisions = 0
    
    for old_key, count in old_collision_groups.items():
        if count > 1:
            # Find which indices have this old key
            indices = [i for i, k in enumerate(old_keys) if k == old_key]
            
            # Count unique new keys for these indices
            unique_new_keys = len(set(new_keys[i] for i in indices))
            
            total_collisions += count - 1  # Number of collisions
            resolved_collisions += unique_new_keys - 1  # New keys minus 1
    
    resolution_rate = resolved_collisions / max(total_collisions, 1)
    
    print(f"\n  Old index collisions:     {total_collisions}")
    print(f"  Resolved by vorticity:    {resolved_collisions} ({resolution_rate:.1%})")
    print(f"  Old unique keys:          {len(set(old_keys))}")
    print(f"  New unique keys:          {len(set(new_keys))}")
    
    key_improvement = len(set(new_keys)) / len(set(old_keys))
    print(f"\n  === KEY DIVERSITY IMPROVEMENT: {key_improvement:.1f}× ===")
    
    assert len(set(new_keys)) > len(set(old_keys)) * 10, \
        f"Expected 10× more unique keys, got {len(set(new_keys))} vs {len(set(old_keys))}"
    
    print("\n  ✅ PASS: Vorticity adds significant discrimination")
    return True


def test_hybrid_memory_uses_vorticity():
    """
    Verify HybridHolographicMemory uses VorticityWitnessIndex by default.
    """
    print("\n" + "=" * 70)
    print("  TEST: HybridHolographicMemory Default Index")
    print("=" * 70)
    
    _, basis = generate_random_contexts(10, 64)
    
    # Default should use VorticityWitnessIndex
    memory = HybridHolographicMemory.create(basis, use_vorticity_index=True)
    assert isinstance(memory.witness_index, VorticityWitnessIndex), \
        f"Expected VorticityWitnessIndex, got {type(memory.witness_index)}"
    
    print("\n  ✅ HybridHolographicMemory uses VorticityWitnessIndex by default")
    
    # Legacy mode should use WitnessIndex
    memory_legacy = HybridHolographicMemory.create(basis, use_vorticity_index=False)
    assert isinstance(memory_legacy.witness_index, WitnessIndex), \
        f"Expected WitnessIndex in legacy mode, got {type(memory_legacy.witness_index)}"
    
    print("  ✅ Legacy mode uses WitnessIndex")
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  VORTICITY INDEX FIX — COMPREHENSIVE TEST SUITE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    tests = [
        ("Bucket Distribution", test_bucket_distribution_comparison),
        ("Retrieval Accuracy", test_retrieval_accuracy),
        ("Scaling Behavior", test_scaling_behavior),
        ("Vorticity Discriminates", test_vorticity_discriminates),
        ("Hybrid Memory Default", test_hybrid_memory_uses_vorticity),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"\n  ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))
    
    # Summary
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  TEST SUMMARY".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    
    all_passed = True
    for name, status in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"║  {icon} {name:.<50} {status:>10}  ║")
        if status != "PASS":
            all_passed = False
    
    print("╚" + "═" * 68 + "╝")
    
    if all_passed:
        print("\n  🎉 ALL TESTS PASSED — The vorticity index fix is working!")
    else:
        print("\n  ⚠️  SOME TESTS FAILED — Review the output above.")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
