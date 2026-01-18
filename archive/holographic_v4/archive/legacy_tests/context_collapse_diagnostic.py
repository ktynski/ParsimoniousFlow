#!/usr/bin/env python3
"""
DIAGNOSTIC: Witness Collapse at Long Context

HYPOTHESIS: At long context lengths (3000+ tokens), different input sequences
produce nearly identical witness (σ, p) values due to Central Limit Theorem
effects during geometric product composition.

This would explain:
- 4-8% accuracy at ctx=3509 (everything looks the same, wrong retrievals)
- 0% skip rate (novelty check doesn't recognize patterns)
- Constant witness stability 0.795 (all matrices have same structure)

THEORY INSIGHT:
    After n geometric products with normalization, the matrix distribution
    may converge to a fixed point regardless of input. This is **correct**
    Grace behavior (contracting to witness subspace) but becomes pathological
    when ALL witnesses converge to the SAME values.
    
TEST:
    Generate K random sequences of length N, compute context, measure:
    1. Pairwise witness similarity (should be LOW for discriminative power)
    2. Witness spread (std of σ and p across sequences)
    3. Full matrix similarity (are matrices identical or just witnesses?)
"""

import numpy as np
import time
from typing import Tuple, List
from holographic_v4.algebra import (
    build_clifford_basis,
    geometric_product_batch,
    grace_operator,
    initialize_embeddings_identity,
)
from holographic_v4.quotient import extract_witness
from holographic_v4.constants import DTYPE, MATRIX_DIM


def measure_context_diversity(
    context_lengths: List[int],
    vocab_size: int = 50000,
    n_sequences: int = 50,
    seed: int = 42
) -> dict:
    """
    Measure how diverse context representations are at different lengths.
    
    Returns dict mapping context_length -> diversity metrics
    """
    rng = np.random.default_rng(seed)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    results = {}
    
    for ctx_len in context_lengths:
        print(f"\n  Testing context length: {ctx_len}...")
        
        # Generate n_sequences random token sequences
        sequences = [rng.integers(0, vocab_size, size=ctx_len) for _ in range(n_sequences)]
        
        # Compute context for each
        contexts = []
        witnesses = []
        
        t0 = time.time()
        for seq in sequences:
            mats = embeddings[seq]  # [ctx_len, 4, 4]
            ctx = geometric_product_batch(mats, np)  # [4, 4]
            ctx = grace_operator(ctx, basis, np)  # Apply Grace
            contexts.append(ctx)
            
            w = extract_witness(ctx, basis, np)
            witnesses.append(w)
        
        elapsed = time.time() - t0
        
        contexts = np.array(contexts)  # [n_seq, 4, 4]
        witnesses = np.array(witnesses)  # [n_seq, 2]
        
        # Measure witness spread
        sigma_values = witnesses[:, 0]  # Scalar components
        pseudo_values = witnesses[:, 1]  # Pseudoscalar components
        
        sigma_mean = float(np.mean(sigma_values))
        sigma_std = float(np.std(sigma_values))
        pseudo_mean = float(np.mean(pseudo_values))
        pseudo_std = float(np.std(pseudo_values))
        
        # Measure pairwise witness similarity
        witness_sims = []
        for i in range(n_sequences):
            for j in range(i + 1, n_sequences):
                w1 = witnesses[i]
                w2 = witnesses[j]
                n1 = np.sqrt(np.dot(w1, w1) + 1e-12)
                n2 = np.sqrt(np.dot(w2, w2) + 1e-12)
                sim = np.dot(w1, w2) / (n1 * n2)
                witness_sims.append(sim)
        
        witness_sim_mean = float(np.mean(witness_sims))
        witness_sim_std = float(np.std(witness_sims))
        
        # Measure pairwise full matrix similarity (Frobenius)
        matrix_sims = []
        for i in range(n_sequences):
            for j in range(i + 1, n_sequences):
                m1 = contexts[i]
                m2 = contexts[j]
                n1 = np.linalg.norm(m1, 'fro')
                n2 = np.linalg.norm(m2, 'fro')
                sim = np.sum(m1 * m2) / (n1 * n2 + 1e-12)
                matrix_sims.append(sim)
        
        matrix_sim_mean = float(np.mean(matrix_sims))
        matrix_sim_std = float(np.std(matrix_sims))
        
        # Compute witness stability for each context
        stabilities = []
        for i in range(n_sequences):
            ctx = contexts[i]
            w = witnesses[i]
            witness_energy = w[0]**2 + w[1]**2
            total_energy = np.sum(ctx ** 2)
            stability = witness_energy / (total_energy + 1e-12)
            stabilities.append(stability)
        
        stability_mean = float(np.mean(stabilities))
        stability_std = float(np.std(stabilities))
        
        results[ctx_len] = {
            'sigma_mean': sigma_mean,
            'sigma_std': sigma_std,
            'pseudo_mean': pseudo_mean,
            'pseudo_std': pseudo_std,
            'witness_sim_mean': witness_sim_mean,
            'witness_sim_std': witness_sim_std,
            'matrix_sim_mean': matrix_sim_mean,
            'matrix_sim_std': matrix_sim_std,
            'stability_mean': stability_mean,
            'stability_std': stability_std,
            'time_per_ctx': elapsed / n_sequences,
        }
        
        # Print summary
        print(f"    σ: mean={sigma_mean:.4f}, std={sigma_std:.4f}")
        print(f"    p: mean={pseudo_mean:.4f}, std={pseudo_std:.4f}")
        print(f"    Witness pairwise sim: {witness_sim_mean:.4f} ± {witness_sim_std:.4f}")
        print(f"    Matrix pairwise sim:  {matrix_sim_mean:.4f} ± {matrix_sim_std:.4f}")
        print(f"    Witness stability:    {stability_mean:.4f} ± {stability_std:.4f}")
    
    return results


def diagnose_collapse():
    """Run the full diagnosis."""
    print("=" * 70)
    print("  WITNESS COLLAPSE DIAGNOSTIC")
    print("=" * 70)
    print()
    print("  Testing if context representations collapse at long sequences...")
    print()
    
    # Test at the context sizes used in training
    context_lengths = [16, 64, 256, 512, 1024, 2048, 3509]
    
    results = measure_context_diversity(
        context_lengths=context_lengths,
        vocab_size=50000,
        n_sequences=50,
        seed=42
    )
    
    print()
    print("=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print()
    print(f"{'Ctx Len':>8} | {'σ std':>8} | {'p std':>8} | {'W sim':>8} | {'M sim':>8} | {'Stability':>10}")
    print("-" * 70)
    
    for ctx_len in context_lengths:
        r = results[ctx_len]
        print(f"{ctx_len:>8} | {r['sigma_std']:>8.5f} | {r['pseudo_std']:>8.5f} | "
              f"{r['witness_sim_mean']:>8.4f} | {r['matrix_sim_mean']:>8.4f} | "
              f"{r['stability_mean']:>8.4f}±{r['stability_std']:.3f}")
    
    print()
    print("=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    print()
    
    # Check for collapse
    short = results[64]
    long = results[3509]
    
    sigma_collapse = short['sigma_std'] / (long['sigma_std'] + 1e-8)
    witness_collapse = long['witness_sim_mean'] - short['witness_sim_mean']
    
    if sigma_collapse > 10:
        print("  ❌ WITNESS COLLAPSE DETECTED!")
        print(f"     σ spread at ctx=64:   {short['sigma_std']:.5f}")
        print(f"     σ spread at ctx=3509: {long['sigma_std']:.5f}")
        print(f"     Collapse ratio: {sigma_collapse:.1f}× less diversity at long context")
        print()
        print("  ROOT CAUSE: Geometric product composition converges to fixed point.")
        print("  After 3509 products, information about input sequence is lost.")
        print()
        print("  IMPLICATIONS:")
        print("  1. All long contexts have similar witnesses → retrieval fails")
        print("  2. Novelty detection fails (everything looks the same)")
        print("  3. Schema discovery over-fires (random noise gets schematized)")
        print()
        print("  POTENTIAL FIXES:")
        print("  A. Use VORTICITY for long-context discrimination (it accumulates, doesn't collapse)")
        print("  B. Cap context size (stay below collapse threshold)")
        print("  C. Use hierarchical chunking with longer spans")
        print("  D. Add explicit positional encoding for long-range dependencies")
    elif long['witness_sim_mean'] > 0.95:
        print("  ⚠️  HIGH WITNESS SIMILARITY at long context!")
        print(f"     Mean pairwise witness sim at ctx=3509: {long['witness_sim_mean']:.4f}")
        print("     Different sequences produce nearly identical witnesses.")
    else:
        print("  ✓ No witness collapse detected.")
        print(f"    Witness diversity maintained across context lengths.")
    
    print()
    return results


if __name__ == "__main__":
    diagnose_collapse()
