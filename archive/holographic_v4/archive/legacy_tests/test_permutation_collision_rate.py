#!/usr/bin/env python3
"""
TEST: Permutation Collision Rate

We found that witness is 100% blind to AB↔BA swaps.
But longer sequences accumulate more changes.

Question: How often do permuted sequences collide in the index?
"""

import numpy as np
from collections import defaultdict

from holographic_v4.algebra import (
    build_clifford_basis,
    geometric_product_batch,
    grace_operator,
    initialize_embeddings_identity,
)
from holographic_v4.holographic_memory import WitnessIndex, VorticityWitnessIndex
from holographic_v4.quotient import vorticity_index_key


def test_permutation_collision_rate():
    """
    For each sequence, create a permutation.
    Check if they end up in the same bucket.
    """
    print("\n" + "=" * 70)
    print("  TEST: Permutation Collision Rate")
    print("  How often do permuted sequences collide in each index?")
    print("=" * 70)
    
    n_tests = 1000
    context_lengths = [2, 4, 8, 16, 64, 256]
    vocab_size = 50000
    
    rng = np.random.default_rng(42)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    results = {}
    
    for ctx_len in context_lengths:
        witness_collisions = 0
        vorticity_collisions = 0
        
        for _ in range(n_tests):
            # Original sequence
            seq = rng.integers(0, vocab_size, size=ctx_len)
            
            # Random permutation (shuffle)
            perm = seq.copy()
            rng.shuffle(perm)
            
            # Skip if identical (can happen with short sequences)
            if np.array_equal(seq, perm):
                continue
            
            # Compute contexts
            mats1 = embeddings[seq]
            ctx1 = geometric_product_batch(mats1, np)
            ctx1 = grace_operator(ctx1, basis, np)
            
            mats2 = embeddings[perm]
            ctx2 = geometric_product_batch(mats2, np)
            ctx2 = grace_operator(ctx2, basis, np)
            
            # WitnessIndex key
            wi = WitnessIndex.create(basis, xp=np)
            wk1 = wi._witness_key(ctx1)
            wk2 = wi._witness_key(ctx2)
            
            # VorticityWitnessIndex key
            vk1 = vorticity_index_key(ctx1, basis, np)
            vk2 = vorticity_index_key(ctx2, basis, np)
            
            if wk1 == wk2:
                witness_collisions += 1
            if vk1 == vk2:
                vorticity_collisions += 1
        
        witness_rate = witness_collisions / n_tests * 100
        vorticity_rate = vorticity_collisions / n_tests * 100
        
        results[ctx_len] = {
            'witness': witness_rate,
            'vorticity': vorticity_rate,
        }
        
        print(f"\n  Context length {ctx_len}:")
        print(f"    WitnessIndex collisions:   {witness_collisions}/{n_tests} = {witness_rate:.1f}%")
        print(f"    VorticityIndex collisions: {vorticity_collisions}/{n_tests} = {vorticity_rate:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("  COLLISION RATE SUMMARY (lower is better)")
    print("=" * 70)
    print(f"  {'Ctx Len':>8} | {'Witness':>10} | {'Vorticity':>10} | {'Improvement':>12}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    
    for ctx_len in context_lengths:
        w = results[ctx_len]['witness']
        v = results[ctx_len]['vorticity']
        improvement = w / v if v > 0 else float('inf')
        print(f"  {ctx_len:>8} | {w:>9.1f}% | {v:>9.1f}% | {improvement:>11.1f}x")
    
    return results


def test_two_token_swap():
    """
    The most basic test: AB vs BA with just two tokens.
    This is where witness MUST be blind (we proved it mathematically).
    """
    print("\n" + "=" * 70)
    print("  TEST: Two-Token Swap (AB vs BA)")
    print("  The witness should be IDENTICAL for both orderings.")
    print("=" * 70)
    
    n_tests = 1000
    vocab_size = 50000
    
    rng = np.random.default_rng(42)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    witness_same = 0
    vorticity_same = 0
    
    for _ in range(n_tests):
        a = rng.integers(0, vocab_size)
        b = rng.integers(0, vocab_size)
        while b == a:
            b = rng.integers(0, vocab_size)
        
        # AB
        mats_ab = embeddings[[a, b]]
        ctx_ab = geometric_product_batch(mats_ab, np)
        ctx_ab = grace_operator(ctx_ab, basis, np)
        
        # BA
        mats_ba = embeddings[[b, a]]
        ctx_ba = geometric_product_batch(mats_ba, np)
        ctx_ba = grace_operator(ctx_ba, basis, np)
        
        # Check witnesses (extract sigma, pseudo)
        scalar_ab = np.sum(basis[0] * ctx_ab) / 4.0
        pseudo_ab = np.sum(basis[15] * ctx_ab) / 4.0
        
        scalar_ba = np.sum(basis[0] * ctx_ba) / 4.0
        pseudo_ba = np.sum(basis[15] * ctx_ba) / 4.0
        
        # Are witnesses identical?
        if abs(scalar_ab - scalar_ba) < 1e-10 and abs(pseudo_ab - pseudo_ba) < 1e-10:
            witness_same += 1
        
        # Check vorticity keys
        vk_ab = vorticity_index_key(ctx_ab, basis, np)
        vk_ba = vorticity_index_key(ctx_ba, basis, np)
        
        if vk_ab == vk_ba:
            vorticity_same += 1
    
    print(f"\n  Results for 2-token sequences (AB vs BA):")
    print(f"    Witness IDENTICAL:   {witness_same}/{n_tests} = {witness_same/n_tests*100:.1f}%")
    print(f"    Vorticity IDENTICAL: {vorticity_same}/{n_tests} = {vorticity_same/n_tests*100:.1f}%")
    
    print(f"\n  Theory says:")
    print(f"    - Witness = Tr(AB) and Tr(BA) are ALWAYS equal (trace is cyclic!)")
    print(f"    - So witness should be 100% identical for AB vs BA")
    print(f"    - Vorticity (bivectors) are NOT cyclic, so should differ")
    
    if witness_same == n_tests:
        print(f"\n  ✓ CONFIRMED: Witness is BLIND to 2-token ordering")
    else:
        print(f"\n  ✗ UNEXPECTED: Witness differs for some cases")


def run_all():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  PERMUTATION COLLISION RATE ANALYSIS".center(68) + "║")
    print("║" + "  Does adding vorticity actually help?".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    test_two_token_swap()
    test_permutation_collision_rate()
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  THEORY EXPLANATION".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + "".center(68) + "║")
    print("║  Witness (σ, p) = scalar + pseudoscalar components".center(68) + "║")
    print("║  These are computed via TRACE, which is CYCLIC:".center(68) + "║")
    print("║       Tr(AB) = Tr(BA) always!".center(68) + "║")
    print("║".center(68) + "║")
    print("║  So the witness CANNOT encode word order by construction.".center(68) + "║")
    print("║".center(68) + "║")
    print("║  Vorticity = bivector components = antisymmetric part".center(68) + "║")
    print("║       AB - BA ≠ 0 (captures the order difference)".center(68) + "║")
    print("╚" + "═" * 68 + "╝")


if __name__ == "__main__":
    run_all()
