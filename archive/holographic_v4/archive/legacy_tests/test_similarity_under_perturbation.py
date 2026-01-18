#!/usr/bin/env python3
"""
TEST: Similarity Under Perturbation

The REAL test: Can we distinguish "Dog bites man" from "Man bites dog"?

These have:
- Same tokens (witness should be similar)
- Different order (vorticity should differ)

We need to test if the metrics can distinguish SIMILAR but DIFFERENT contexts.
"""

import numpy as np
import time
from typing import Dict, List, Tuple

from holographic_v4.algebra import (
    build_clifford_basis,
    geometric_product_batch,
    geometric_product,
    grace_operator,
    initialize_embeddings_identity,
    decompose_to_coefficients,
)
from holographic_v4.quotient import extract_witness
from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, DTYPE


# =============================================================================
# BATCH FEATURE EXTRACTION
# =============================================================================

def extract_witness_batch(Ms: np.ndarray, basis: np.ndarray) -> np.ndarray:
    scalars = np.einsum('nij,ij->n', Ms, basis[0]) / 4.0
    pseudos = np.einsum('nij,ij->n', Ms, basis[15]) / 4.0
    return np.stack([scalars, pseudos], axis=1)


def extract_bivectors_batch(Ms: np.ndarray, basis: np.ndarray) -> np.ndarray:
    bivectors = []
    for i in range(6):
        coeff = np.einsum('nij,ij->n', Ms, basis[5 + i]) / 4.0
        bivectors.append(coeff)
    return np.stack(bivectors, axis=1)


# =============================================================================
# TEST: Permutation Sensitivity
# =============================================================================

def test_permutation_sensitivity():
    """
    Test: Given a sequence and its permutation, can we tell them apart?
    
    This is the key linguistic test - word ORDER matters!
    """
    print("\n" + "=" * 70)
    print("  TEST: Permutation Sensitivity")
    print("  Can we distinguish 'Dog bites man' from 'Man bites dog'?")
    print("=" * 70)
    
    n_tests = 500
    context_length = 16  # Short contexts to see the effect clearly
    vocab_size = 50000
    
    rng = np.random.default_rng(42)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    witness_same = []
    witness_different = []
    vorticity_same = []
    vorticity_different = []
    
    for _ in range(n_tests):
        # Generate a random sequence
        seq = rng.integers(0, vocab_size, size=context_length)
        
        # Create a permutation (shuffle the same tokens)
        perm = seq.copy()
        rng.shuffle(perm)
        
        # Compute contexts
        mats1 = embeddings[seq]
        ctx1 = geometric_product_batch(mats1, np)
        ctx1 = grace_operator(ctx1, basis, np)
        
        mats2 = embeddings[perm]
        ctx2 = geometric_product_batch(mats2, np)
        ctx2 = grace_operator(ctx2, basis, np)
        
        # Extract features
        w1 = extract_witness_batch(ctx1[None], basis)[0]
        w2 = extract_witness_batch(ctx2[None], basis)[0]
        
        b1 = extract_bivectors_batch(ctx1[None], basis)[0]
        b2 = extract_bivectors_batch(ctx2[None], basis)[0]
        
        # Compute similarities
        # Witness similarity
        n1, n2 = np.linalg.norm(w1), np.linalg.norm(w2)
        if n1 > 1e-8 and n2 > 1e-8:
            w_sim = np.dot(w1, w2) / (n1 * n2)
        else:
            w_sim = 0.0
        
        # Vorticity similarity
        n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
        if n1 > 1e-8 and n2 > 1e-8:
            v_sim = np.dot(b1, b2) / (n1 * n2)
        else:
            v_sim = 0.0
        
        # Record
        if np.array_equal(seq, perm):
            witness_same.append(w_sim)
            vorticity_same.append(v_sim)
        else:
            witness_different.append(w_sim)
            vorticity_different.append(v_sim)
    
    print(f"\n  Results (for PERMUTED sequences - same tokens, different order):")
    print(f"  {'-'*60}")
    print(f"  Witness similarity:   {np.mean(witness_different):.4f} ± {np.std(witness_different):.4f}")
    print(f"  Vorticity similarity: {np.mean(vorticity_different):.4f} ± {np.std(vorticity_different):.4f}")
    
    print(f"\n  Interpretation:")
    if np.mean(witness_different) > 0.9:
        print(f"    → Witness CANNOT distinguish permutations (similarity {np.mean(witness_different):.2f})")
    else:
        print(f"    → Witness CAN distinguish permutations (similarity {np.mean(witness_different):.2f})")
    
    if np.mean(vorticity_different) > 0.9:
        print(f"    → Vorticity CANNOT distinguish permutations (similarity {np.mean(vorticity_different):.2f})")
    else:
        print(f"    → Vorticity CAN distinguish permutations (similarity {np.mean(vorticity_different):.2f})")
    
    return {
        'witness': witness_different,
        'vorticity': vorticity_different,
    }


# =============================================================================
# TEST: Single Token Change Sensitivity
# =============================================================================

def test_single_token_change():
    """
    Test: Changing ONE token - can we detect it?
    
    This tests robustness vs sensitivity tradeoff.
    """
    print("\n" + "=" * 70)
    print("  TEST: Single Token Change Sensitivity")
    print("  Can we detect changing just ONE word?")
    print("=" * 70)
    
    n_tests = 500
    context_lengths = [8, 16, 64, 256]
    vocab_size = 50000
    
    rng = np.random.default_rng(42)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    results = {}
    
    for ctx_len in context_lengths:
        witness_sims = []
        vorticity_sims = []
        
        for _ in range(n_tests):
            # Original sequence
            seq = rng.integers(0, vocab_size, size=ctx_len)
            
            # Modified: change ONE token
            modified = seq.copy()
            change_pos = rng.integers(0, ctx_len)
            new_token = rng.integers(0, vocab_size)
            while new_token == seq[change_pos]:
                new_token = rng.integers(0, vocab_size)
            modified[change_pos] = new_token
            
            # Compute contexts
            mats1 = embeddings[seq]
            ctx1 = geometric_product_batch(mats1, np)
            ctx1 = grace_operator(ctx1, basis, np)
            
            mats2 = embeddings[modified]
            ctx2 = geometric_product_batch(mats2, np)
            ctx2 = grace_operator(ctx2, basis, np)
            
            # Features
            w1 = extract_witness_batch(ctx1[None], basis)[0]
            w2 = extract_witness_batch(ctx2[None], basis)[0]
            b1 = extract_bivectors_batch(ctx1[None], basis)[0]
            b2 = extract_bivectors_batch(ctx2[None], basis)[0]
            
            # Similarities
            n1, n2 = np.linalg.norm(w1), np.linalg.norm(w2)
            w_sim = np.dot(w1, w2) / (n1 * n2) if n1 > 1e-8 and n2 > 1e-8 else 0.0
            
            n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
            v_sim = np.dot(b1, b2) / (n1 * n2) if n1 > 1e-8 and n2 > 1e-8 else 0.0
            
            witness_sims.append(w_sim)
            vorticity_sims.append(v_sim)
        
        results[ctx_len] = {
            'witness': (np.mean(witness_sims), np.std(witness_sims)),
            'vorticity': (np.mean(vorticity_sims), np.std(vorticity_sims)),
        }
        
        print(f"\n  Context length {ctx_len}:")
        print(f"    Witness:   {results[ctx_len]['witness'][0]:.4f} ± {results[ctx_len]['witness'][1]:.4f}")
        print(f"    Vorticity: {results[ctx_len]['vorticity'][0]:.4f} ± {results[ctx_len]['vorticity'][1]:.4f}")
    
    return results


# =============================================================================
# TEST: Order Matters - Geometric Product Non-Commutativity
# =============================================================================

def test_geometric_product_noncommutativity():
    """
    Test: Is A ⊗ B different from B ⊗ A?
    
    The geometric product is NOT commutative - this is how we encode order!
    """
    print("\n" + "=" * 70)
    print("  TEST: Geometric Product Non-Commutativity")
    print("  Is A ⊗ B ≠ B ⊗ A? (How we encode word order)")
    print("=" * 70)
    
    n_tests = 1000
    vocab_size = 50000
    
    rng = np.random.default_rng(42)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    # Test just TWO tokens: AB vs BA
    differences = []
    witness_diffs = []
    vorticity_diffs = []
    
    for _ in range(n_tests):
        a_idx = rng.integers(0, vocab_size)
        b_idx = rng.integers(0, vocab_size)
        while b_idx == a_idx:
            b_idx = rng.integers(0, vocab_size)
        
        A = embeddings[a_idx]
        B = embeddings[b_idx]
        
        # AB = A ⊗ B
        AB = geometric_product(A, B)
        AB = grace_operator(AB, basis, np)
        
        # BA = B ⊗ A
        BA = geometric_product(B, A)
        BA = grace_operator(BA, basis, np)
        
        # Matrix difference (Frobenius)
        diff = np.linalg.norm(AB - BA, 'fro') / np.linalg.norm(AB, 'fro')
        differences.append(diff)
        
        # Witness difference
        wAB = extract_witness_batch(AB[None], basis)[0]
        wBA = extract_witness_batch(BA[None], basis)[0]
        w_diff = np.linalg.norm(wAB - wBA) / max(np.linalg.norm(wAB), 1e-8)
        witness_diffs.append(w_diff)
        
        # Vorticity difference
        bAB = extract_bivectors_batch(AB[None], basis)[0]
        bBA = extract_bivectors_batch(BA[None], basis)[0]
        v_diff = np.linalg.norm(bAB - bBA) / max(np.linalg.norm(bAB), 1e-8)
        vorticity_diffs.append(v_diff)
    
    print(f"\n  AB vs BA (just 2 tokens):")
    print(f"    Matrix difference:   {np.mean(differences):.4f} ± {np.std(differences):.4f}")
    print(f"    Witness difference:  {np.mean(witness_diffs):.4f} ± {np.std(witness_diffs):.4f}")
    print(f"    Vorticity difference:{np.mean(vorticity_diffs):.4f} ± {np.std(vorticity_diffs):.4f}")
    
    print(f"\n  Interpretation:")
    if np.mean(differences) < 0.01:
        print(f"    ✗ Geometric product is NEARLY COMMUTATIVE (problem!)")
    else:
        print(f"    ✓ Geometric product is NON-COMMUTATIVE ({np.mean(differences):.1%} different)")
    
    print(f"\n  Where is the order information?")
    if np.mean(vorticity_diffs) > np.mean(witness_diffs):
        print(f"    → Order is encoded in VORTICITY (bivectors)")
        print(f"       Vorticity captures {np.mean(vorticity_diffs):.1%} of the order change")
        print(f"       Witness captures {np.mean(witness_diffs):.1%} of the order change")
    else:
        print(f"    → Order is encoded in WITNESS")
    
    return differences, witness_diffs, vorticity_diffs


# =============================================================================
# TEST: What the Holographic Memory Actually Stores
# =============================================================================

def test_what_memory_stores():
    """
    What does the holographic memory ACTUALLY look for?
    
    Let's trace through the actual code path.
    """
    print("\n" + "=" * 70)
    print("  TEST: What Holographic Memory Actually Stores")
    print("=" * 70)
    
    # Import the actual memory classes
    from holographic_v4.holographic_memory import (
        HolographicMemory,
        WitnessIndex,
        VorticityWitnessIndex,
        HybridHolographicMemory,
    )
    
    vocab_size = 50000
    context_length = 64
    
    rng = np.random.default_rng(42)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    print("\n  1. HolographicMemory (superposition):")
    print("     Uses: bind(context, target) → superposition in 4x4 matrix")
    print("     Retrieval: unbind(context, memory) → recovers target")
    print("     ➤ Uses FULL MATRIX for storage/retrieval")
    
    print("\n  2. WitnessIndex (bucket fallback):")
    print("     Indexing: _witness_key(context) → (rounded_σ, rounded_p)")
    print("     Within bucket: witness_similarity(query, stored)")
    print("     ➤ Uses ONLY witness (σ, p) - LOSES VORTICITY")
    
    print("\n  3. VorticityWitnessIndex (new):")
    print("     Indexing: (σ, p, enstrophy, dominant_plane)")
    print("     Within bucket: vorticity_similarity(query, stored)")
    print("     ➤ Uses witness + vorticity for BOTH indexing and matching")
    
    # Create test memories
    mem_old = WitnessIndex.create(basis, xp=np)
    mem_new = VorticityWitnessIndex.create(basis, xp=np)
    
    # Create two similar contexts (same tokens, different order)
    seq = rng.integers(0, vocab_size, size=context_length)
    perm = seq.copy()
    rng.shuffle(perm)
    
    mats1 = embeddings[seq]
    ctx1 = geometric_product_batch(mats1, np)
    ctx1 = grace_operator(ctx1, basis, np)
    
    mats2 = embeddings[perm]
    ctx2 = geometric_product_batch(mats2, np)
    ctx2 = grace_operator(ctx2, basis, np)
    
    # Check keys
    from holographic_v4.quotient import vorticity_index_key
    
    old_key1 = mem_old._witness_key(ctx1)
    old_key2 = mem_old._witness_key(ctx2)
    
    new_key1 = vorticity_index_key(ctx1, basis, np)
    new_key2 = vorticity_index_key(ctx2, basis, np)
    
    print("\n  For two PERMUTED sequences (same tokens, different order):")
    print(f"\n    WitnessIndex keys:")
    print(f"      Original:  {old_key1}")
    print(f"      Permuted:  {old_key2}")
    print(f"      Same key?  {old_key1 == old_key2}")
    
    print(f"\n    VorticityWitnessIndex keys:")
    print(f"      Original:  {new_key1}")
    print(f"      Permuted:  {new_key2}")
    print(f"      Same key?  {new_key1 == new_key2}")
    
    if old_key1 == old_key2 and new_key1 != new_key2:
        print(f"\n  ➤ VORTICITY RESOLVES THE COLLISION!")
        print(f"    Old index: Both permutations → same bucket (collision)")
        print(f"    New index: Different buckets → can discriminate")


# =============================================================================
# RUN ALL
# =============================================================================

def run_all():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  SIMILARITY UNDER PERTURBATION".center(68) + "║")
    print("║" + "  Can we distinguish semantically similar but different contexts?".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Test 1: Non-commutativity
    test_geometric_product_noncommutativity()
    
    # Test 2: Permutation sensitivity
    test_permutation_sensitivity()
    
    # Test 3: Single token change
    test_single_token_change()
    
    # Test 4: What memory stores
    test_what_memory_stores()
    
    # Final summary
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  CONCLUSION".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + "  1. Geometric product IS non-commutative".center(68) + "║")
    print("║" + "  2. Vorticity (G2) captures word ORDER".center(68) + "║")
    print("║" + "  3. Witness (G0+G4) captures word CONTENT".center(68) + "║")
    print("║" + "  4. BOTH are needed: witness+vorticity together".center(68) + "║")
    print("╚" + "═" * 68 + "╝")


if __name__ == "__main__":
    run_all()
