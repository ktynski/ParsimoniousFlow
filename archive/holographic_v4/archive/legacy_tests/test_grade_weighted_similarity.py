#!/usr/bin/env python3
"""
TEST: Theory-True Grade-Weighted Similarity

From rhnsclifford.md:
    "The grades of Cl(1,3) form a phi-nested hierarchy"
    
    similarity = sum(GRACE_SCALE[k] * grade_sim(query, context, k) for k in range(5))

This test compares different similarity metrics:
1. Witness-only (σ, p) - current approach
2. Vorticity-only (bivectors) - what we thought was the fix
3. φ-weighted ALL grades - what theory says

The goal: Determine which approach provides the best discrimination
for retrieval while respecting theory.
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from collections import Counter

from holographic_v4.algebra import (
    build_clifford_basis,
    geometric_product_batch,
    grace_operator,
    initialize_embeddings_identity,
    decompose_to_coefficients,
)
from holographic_v4.quotient import extract_witness
from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, DTYPE


# =============================================================================
# SIMILARITY METRICS
# =============================================================================

def witness_similarity(M1: np.ndarray, M2: np.ndarray, basis: np.ndarray) -> float:
    """
    Similarity using ONLY witness (scalar + pseudoscalar).
    This is what the current system emphasizes.
    """
    s1, p1 = extract_witness(M1, basis, np)
    s2, p2 = extract_witness(M2, basis, np)
    
    w1 = np.array([s1, p1])
    w2 = np.array([s2, p2])
    
    n1 = np.linalg.norm(w1)
    n2 = np.linalg.norm(w2)
    
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    
    return float(np.dot(w1, w2) / (n1 * n2))


def vorticity_similarity(M1: np.ndarray, M2: np.ndarray, basis: np.ndarray) -> float:
    """
    Similarity using ONLY bivectors (grade 2 = vorticity).
    """
    # Extract bivector coefficients (indices 5-10 in Clifford basis)
    bv1 = np.array([float(np.sum(basis[5+i] * M1) / 4.0) for i in range(6)])
    bv2 = np.array([float(np.sum(basis[5+i] * M2) / 4.0) for i in range(6)])
    
    n1 = np.linalg.norm(bv1)
    n2 = np.linalg.norm(bv2)
    
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    
    return float(np.dot(bv1, bv2) / (n1 * n2))


def grade_weighted_similarity(M1: np.ndarray, M2: np.ndarray, basis: np.ndarray) -> float:
    """
    THEORY-TRUE: φ-weighted similarity across ALL grades.
    
    From rhnsclifford.md:
        similarity = sum(GRACE_SCALE[k] * grade_sim(query, context, k) for k in range(5))
        
    Grace scales:
        Grade 0: ×1.0
        Grade 1: ×φ⁻¹
        Grade 2: ×φ⁻²
        Grade 3: ×φ⁻³
        Grade 4: ×φ⁻¹ (Fibonacci exception)
    """
    # Grace scales for each grade
    GRACE_SCALE = {
        0: 1.0,       # Scalar
        1: PHI_INV,   # Vectors
        2: PHI_INV_SQ,  # Bivectors (vorticity)
        3: PHI_INV_CUBE,  # Trivectors
        4: PHI_INV,   # Pseudoscalar (Fibonacci exception: φ⁻¹ not φ⁻⁴)
    }
    
    # Grade indices in the 16-element Clifford decomposition
    # Grade 0: index 0 (scalar)
    # Grade 1: indices 1-4 (vectors)
    # Grade 2: indices 5-10 (bivectors)
    # Grade 3: indices 11-14 (trivectors)
    # Grade 4: index 15 (pseudoscalar)
    GRADE_INDICES = {
        0: [0],
        1: [1, 2, 3, 4],
        2: [5, 6, 7, 8, 9, 10],
        3: [11, 12, 13, 14],
        4: [15],
    }
    
    # Decompose both matrices
    c1 = decompose_to_coefficients(M1, basis, np)
    c2 = decompose_to_coefficients(M2, basis, np)
    
    total_sim = 0.0
    total_weight = 0.0
    
    for grade in range(5):
        indices = GRADE_INDICES[grade]
        weight = GRACE_SCALE[grade]
        
        # Extract grade components
        g1 = np.array([c1[i] for i in indices])
        g2 = np.array([c2[i] for i in indices])
        
        n1 = np.linalg.norm(g1)
        n2 = np.linalg.norm(g2)
        
        if n1 > 1e-8 and n2 > 1e-8:
            grade_sim = np.dot(g1, g2) / (n1 * n2)
            total_sim += weight * grade_sim
            total_weight += weight
    
    return total_sim / total_weight if total_weight > 0 else 0.0


def full_frobenius_similarity(M1: np.ndarray, M2: np.ndarray, basis: np.ndarray) -> float:
    """
    Simple Frobenius similarity (treats all 16 components equally).
    """
    n1 = np.linalg.norm(M1, 'fro')
    n2 = np.linalg.norm(M2, 'fro')
    
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    
    return float(np.sum(M1 * M2) / (n1 * n2))


# =============================================================================
# TEST: Discrimination Power
# =============================================================================

def test_discrimination_power():
    """
    Test how well each similarity metric discriminates between different contexts.
    
    A good metric should:
    1. Give HIGH similarity for same/similar contexts
    2. Give LOW similarity for different contexts
    3. Have a wide spread (not cluster around one value)
    """
    print("\n" + "=" * 70)
    print("  TEST: Discrimination Power of Similarity Metrics")
    print("=" * 70)
    
    n_contexts = 500
    context_length = 512
    vocab_size = 50000
    
    print(f"\n  Generating {n_contexts} random contexts...")
    rng = np.random.default_rng(42)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    contexts = []
    for _ in range(n_contexts):
        seq = rng.integers(0, vocab_size, size=context_length)
        mats = embeddings[seq]
        ctx = geometric_product_batch(mats, np)
        ctx = grace_operator(ctx, basis, np)
        contexts.append(ctx)
    
    contexts = np.array(contexts)
    
    # Test each similarity metric
    metrics = {
        'Witness (σ,p)': witness_similarity,
        'Vorticity (G2)': vorticity_similarity,
        'φ-Weighted (ALL)': grade_weighted_similarity,
        'Frobenius (flat)': full_frobenius_similarity,
    }
    
    results = {}
    
    for name, sim_fn in metrics.items():
        print(f"\n  Testing {name}...")
        
        # Self-similarity (should be 1.0)
        self_sims = []
        for i in range(min(100, n_contexts)):
            self_sims.append(sim_fn(contexts[i], contexts[i], basis))
        
        # Pairwise similarity (should have spread)
        pair_sims = []
        for i in range(min(100, n_contexts)):
            for j in range(i + 1, min(100, n_contexts)):
                pair_sims.append(sim_fn(contexts[i], contexts[j], basis))
        
        self_mean = np.mean(self_sims)
        self_std = np.std(self_sims)
        pair_mean = np.mean(pair_sims)
        pair_std = np.std(pair_sims)
        
        # Discrimination power = (self_mean - pair_mean) / pair_std
        # Higher is better (means self-similarity is many std devs above pair similarity)
        if pair_std > 1e-8:
            disc_power = (self_mean - pair_mean) / pair_std
        else:
            disc_power = 0.0
        
        results[name] = {
            'self_mean': self_mean,
            'self_std': self_std,
            'pair_mean': pair_mean,
            'pair_std': pair_std,
            'discrimination': disc_power,
        }
        
        print(f"    Self-similarity:  {self_mean:.4f} ± {self_std:.4f}")
        print(f"    Pair-similarity:  {pair_mean:.4f} ± {pair_std:.4f}")
        print(f"    Discrimination:   {disc_power:.2f}σ")
    
    # Summary table
    print("\n  === DISCRIMINATION POWER SUMMARY ===")
    print(f"  {'Metric':<20} | {'Self':>8} | {'Pair':>8} | {'Spread':>8} | {'Disc':>8}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    for name, r in sorted(results.items(), key=lambda x: -x[1]['discrimination']):
        print(f"  {name:<20} | {r['self_mean']:>8.4f} | {r['pair_mean']:>8.4f} | "
              f"{r['pair_std']:>8.4f} | {r['discrimination']:>8.2f}σ")
    
    # Best metric
    best = max(results.items(), key=lambda x: x[1]['discrimination'])
    print(f"\n  Best discriminator: {best[0]} ({best[1]['discrimination']:.2f}σ)")
    
    return results


# =============================================================================
# TEST: Retrieval Accuracy with Different Metrics
# =============================================================================

def test_retrieval_accuracy_by_metric():
    """
    Test retrieval accuracy using different similarity metrics.
    
    For each metric:
    1. Store N context-target pairs
    2. Query with each context
    3. Measure how often we retrieve the correct target
    """
    print("\n" + "=" * 70)
    print("  TEST: Retrieval Accuracy by Similarity Metric")
    print("=" * 70)
    
    n_patterns = 200
    n_queries = 50
    context_length = 512
    vocab_size = 50000
    
    print(f"\n  Storing {n_patterns} patterns, testing {n_queries} queries...")
    
    rng = np.random.default_rng(42)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    # Generate patterns
    stored = []
    for i in range(n_patterns):
        seq = rng.integers(0, vocab_size, size=context_length)
        mats = embeddings[seq]
        ctx = geometric_product_batch(mats, np)
        ctx = grace_operator(ctx, basis, np)
        
        target_idx = rng.integers(0, vocab_size)
        target = embeddings[target_idx]
        
        stored.append((ctx, target, target_idx))
    
    # Test each metric
    metrics = {
        'Witness (σ,p)': witness_similarity,
        'Vorticity (G2)': vorticity_similarity,
        'φ-Weighted (ALL)': grade_weighted_similarity,
        'Frobenius (flat)': full_frobenius_similarity,
    }
    
    results = {}
    
    for name, sim_fn in metrics.items():
        print(f"\n  Testing {name}...")
        
        correct = 0
        total = 0
        
        # Query first n_queries patterns
        for i in range(min(n_queries, n_patterns)):
            query_ctx, expected_target, expected_idx = stored[i]
            
            # Find best match
            best_sim = -float('inf')
            best_idx = -1
            
            for j, (ctx, target, idx) in enumerate(stored):
                sim = sim_fn(query_ctx, ctx, basis)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j
            
            if best_idx == i:  # Found the right pattern
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        results[name] = accuracy
        
        print(f"    Accuracy: {correct}/{total} = {accuracy:.1%}")
    
    # Summary
    print("\n  === RETRIEVAL ACCURACY SUMMARY ===")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {name:<20}: {bar} {acc:.1%}")
    
    best = max(results.items(), key=lambda x: x[1])
    print(f"\n  Best retrieval: {best[0]} ({best[1]:.1%})")
    
    return results


# =============================================================================
# TEST: Grade Energy Distribution
# =============================================================================

def test_grade_energy_distribution():
    """
    Measure how energy is distributed across grades in typical contexts.
    
    This tells us which grades carry the most information and should
    be weighted accordingly.
    """
    print("\n" + "=" * 70)
    print("  TEST: Grade Energy Distribution")
    print("=" * 70)
    
    n_contexts = 500
    context_length = 512
    vocab_size = 50000
    
    print(f"\n  Analyzing {n_contexts} contexts...")
    
    rng = np.random.default_rng(42)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    # Grade indices
    GRADE_INDICES = {
        0: [0],
        1: [1, 2, 3, 4],
        2: [5, 6, 7, 8, 9, 10],
        3: [11, 12, 13, 14],
        4: [15],
    }
    
    grade_energies = {g: [] for g in range(5)}
    
    for _ in range(n_contexts):
        seq = rng.integers(0, vocab_size, size=context_length)
        mats = embeddings[seq]
        ctx = geometric_product_batch(mats, np)
        ctx = grace_operator(ctx, basis, np)
        
        # Decompose
        coeffs = decompose_to_coefficients(ctx, basis, np)
        
        # Energy per grade
        total_energy = sum(c**2 for c in coeffs)
        for grade, indices in GRADE_INDICES.items():
            grade_energy = sum(coeffs[i]**2 for i in indices)
            if total_energy > 1e-12:
                grade_energies[grade].append(grade_energy / total_energy)
            else:
                grade_energies[grade].append(0.0)
    
    # Summary
    print("\n  === GRADE ENERGY DISTRIBUTION ===")
    print(f"  {'Grade':<12} | {'Mean':>8} | {'Std':>8} | {'Bar':<30}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*30}")
    
    grade_names = {
        0: 'G0 Scalar',
        1: 'G1 Vectors',
        2: 'G2 Bivectors',
        3: 'G3 Trivectors',
        4: 'G4 Pseudo',
    }
    
    for grade in range(5):
        mean = np.mean(grade_energies[grade])
        std = np.std(grade_energies[grade])
        bar = "█" * int(mean * 50) + "░" * (50 - int(mean * 50))
        print(f"  {grade_names[grade]:<12} | {mean:>8.4f} | {std:>8.4f} | {bar[:30]}")
    
    # Calculate theoretical weight vs actual energy
    print("\n  === THEORY vs REALITY ===")
    print(f"  {'Grade':<12} | {'Theory Weight':>14} | {'Actual Energy':>14} | {'Ratio':>8}")
    print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}")
    
    THEORY_WEIGHTS = {0: 1.0, 1: PHI_INV, 2: PHI_INV_SQ, 3: PHI_INV_CUBE, 4: PHI_INV}
    total_theory = sum(THEORY_WEIGHTS.values())
    
    for grade in range(5):
        theory_w = THEORY_WEIGHTS[grade] / total_theory
        actual_e = np.mean(grade_energies[grade])
        ratio = actual_e / theory_w if theory_w > 1e-8 else 0
        print(f"  {grade_names[grade]:<12} | {theory_w:>14.4f} | {actual_e:>14.4f} | {ratio:>8.2f}")
    
    return grade_energies


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all tests and summarize findings."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  THEORY-TRUE SIMILARITY TESTING".center(68) + "║")
    print("║" + "  What the theory says vs what works empirically".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Test 1: Energy distribution
    energy = test_grade_energy_distribution()
    
    # Test 2: Discrimination power
    discrimination = test_discrimination_power()
    
    # Test 3: Retrieval accuracy
    accuracy = test_retrieval_accuracy_by_metric()
    
    # Final summary
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  FINAL FINDINGS".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    
    # Best discriminator
    best_disc = max(discrimination.items(), key=lambda x: x[1]['discrimination'])
    print(f"║  Best discrimination:  {best_disc[0]:<30} ({best_disc[1]['discrimination']:.2f}σ)  ║")
    
    # Best retrieval
    best_ret = max(accuracy.items(), key=lambda x: x[1])
    print(f"║  Best retrieval:       {best_ret[0]:<30} ({best_ret[1]:.1%})    ║")
    
    print("╚" + "═" * 68 + "╝")
    
    return energy, discrimination, accuracy


if __name__ == "__main__":
    run_all_tests()
