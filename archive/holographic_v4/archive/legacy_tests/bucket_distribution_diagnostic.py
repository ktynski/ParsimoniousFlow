#!/usr/bin/env python3
"""
DIAGNOSTIC: Witness Index Bucket Distribution

ROOT CAUSE HYPOTHESIS:
    The witness values (σ, p) cluster around 0 with std ≈ 0.1-0.2.
    With resolution = φ⁻² ≈ 0.382, almost all samples hash to the SAME FEW BUCKETS.
    
    With 50M samples spread across ~10 buckets → ~5M samples per bucket!
    Retrieval within a 5M-item bucket is essentially random guessing.

TEST:
    Generate many random contexts, compute their witness keys, count bucket sizes.
"""

import numpy as np
from collections import Counter
from holographic_v4.algebra import (
    build_clifford_basis,
    geometric_product_batch,
    grace_operator,
    initialize_embeddings_identity,
)
from holographic_v4.quotient import extract_witness
from holographic_v4.constants import PHI_INV_SQ


def analyze_bucket_distribution(
    n_contexts: int = 10000,
    context_length: int = 512,
    vocab_size: int = 50000,
    resolution: float = PHI_INV_SQ,
    seed: int = 42
) -> dict:
    """Analyze witness key bucket distribution."""
    rng = np.random.default_rng(seed)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    print(f"\n  Generating {n_contexts} contexts of length {context_length}...")
    print(f"  Witness resolution: {resolution:.4f}")
    
    # Compute witness keys for many contexts
    keys = []
    witnesses = []
    
    for i in range(n_contexts):
        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i+1}/{n_contexts}...")
        
        seq = rng.integers(0, vocab_size, size=context_length)
        mats = embeddings[seq]
        ctx = geometric_product_batch(mats, np)
        ctx = grace_operator(ctx, basis, np)
        
        s, p = extract_witness(ctx, basis, np)
        s_idx = int(np.floor(s / resolution))
        p_idx = int(np.floor(p / resolution))
        
        keys.append((s_idx, p_idx))
        witnesses.append((s, p))
    
    witnesses = np.array(witnesses)
    
    # Count bucket sizes
    bucket_counts = Counter(keys)
    n_buckets = len(bucket_counts)
    
    # Statistics
    bucket_sizes = list(bucket_counts.values())
    avg_bucket_size = np.mean(bucket_sizes)
    max_bucket_size = max(bucket_sizes)
    min_bucket_size = min(bucket_sizes)
    
    # Witness statistics
    sigma_min, sigma_max = witnesses[:, 0].min(), witnesses[:, 0].max()
    pseudo_min, pseudo_max = witnesses[:, 1].min(), witnesses[:, 1].max()
    
    print()
    print(f"  === BUCKET DISTRIBUTION ===")
    print(f"  Total contexts: {n_contexts}")
    print(f"  Unique buckets: {n_buckets}")
    print(f"  Average bucket size: {avg_bucket_size:.1f}")
    print(f"  Max bucket size: {max_bucket_size} ({max_bucket_size/n_contexts*100:.1f}%)")
    print(f"  Min bucket size: {min_bucket_size}")
    print()
    print(f"  === WITNESS VALUE RANGES ===")
    print(f"  σ range: [{sigma_min:.4f}, {sigma_max:.4f}]")
    print(f"  p range: [{pseudo_min:.4f}, {pseudo_max:.4f}]")
    print(f"  σ span in units of resolution: {(sigma_max - sigma_min) / resolution:.1f}")
    print(f"  p span in units of resolution: {(pseudo_max - pseudo_min) / resolution:.1f}")
    print()
    print(f"  === TOP 10 BUCKETS ===")
    for key, count in bucket_counts.most_common(10):
        pct = count / n_contexts * 100
        print(f"    {key}: {count} ({pct:.1f}%)")
    
    # Estimate collision rate
    if n_contexts > 1:
        # Expected items per bucket at 50M samples
        projected_50M = {
            'items_per_bucket': 50_000_000 / n_buckets if n_buckets > 0 else float('inf'),
            'max_bucket_pct': max_bucket_size / n_contexts * 100,
            'estimated_50M_max_bucket': int(50_000_000 * (max_bucket_size / n_contexts)),
        }
        print()
        print(f"  === PROJECTION TO 50M SAMPLES ===")
        print(f"  If bucket distribution holds:")
        print(f"    Est. items per bucket: {projected_50M['items_per_bucket']:,.0f}")
        print(f"    Est. max bucket size:  {projected_50M['estimated_50M_max_bucket']:,.0f}")
        print()
        
        # Calculate expected retrieval accuracy
        # If we have M items in a bucket and randomly guess, P(correct) = 1/M
        avg_accuracy = 1 / avg_bucket_size if avg_bucket_size > 0 else 0
        worst_accuracy = 1 / max_bucket_size if max_bucket_size > 0 else 0
        print(f"  === RETRIEVAL ACCURACY (if random within bucket) ===")
        print(f"    Average bucket accuracy: {avg_accuracy:.4%}")
        print(f"    Worst bucket accuracy:   {worst_accuracy:.4%}")
        
        projected_avg_accuracy = n_buckets / 50_000_000
        print(f"    Projected accuracy @ 50M: {projected_avg_accuracy:.6%}")
    
    return {
        'n_contexts': n_contexts,
        'n_buckets': n_buckets,
        'avg_bucket_size': avg_bucket_size,
        'max_bucket_size': max_bucket_size,
        'bucket_counts': bucket_counts,
        'witnesses': witnesses,
    }


def test_with_finer_resolution():
    """Test with finer resolution to show the fix."""
    print()
    print("=" * 70)
    print("  TESTING FINER RESOLUTION")
    print("=" * 70)
    
    # Current resolution
    print("\n  Current (φ⁻² ≈ 0.382):")
    analyze_bucket_distribution(
        n_contexts=5000,
        context_length=512,
        resolution=PHI_INV_SQ,
    )
    
    # Finer resolution
    print("\n  Finer (0.05):")
    analyze_bucket_distribution(
        n_contexts=5000,
        context_length=512,
        resolution=0.05,
    )
    
    # Much finer
    print("\n  Much finer (0.01):")
    analyze_bucket_distribution(
        n_contexts=5000,
        context_length=512,
        resolution=0.01,
    )


def main():
    print("=" * 70)
    print("  WITNESS INDEX BUCKET DISTRIBUTION DIAGNOSTIC")
    print("=" * 70)
    
    # Test at different context lengths
    for ctx_len in [512, 1340, 3509]:
        print(f"\n{'='*70}")
        print(f"  CONTEXT LENGTH: {ctx_len}")
        print(f"{'='*70}")
        analyze_bucket_distribution(
            n_contexts=3000,
            context_length=ctx_len,
            resolution=PHI_INV_SQ,
        )
    
    # Show the fix
    test_with_finer_resolution()


if __name__ == "__main__":
    main()
