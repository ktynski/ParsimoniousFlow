#!/usr/bin/env python3
"""
TEST: Similarity Metrics at Scale

At 200 patterns, all metrics got 100%. Let's test at 10K+ patterns
where discrimination actually matters.
"""

import numpy as np
import time
from typing import Dict, List, Tuple

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
# OPTIMIZED BATCH SIMILARITY METRICS
# =============================================================================

def extract_witness_batch(Ms: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Extract (scalar, pseudo) for batch of matrices."""
    # basis[0] = scalar, basis[15] = pseudoscalar
    scalars = np.einsum('nij,ij->n', Ms, basis[0]) / 4.0
    pseudos = np.einsum('nij,ij->n', Ms, basis[15]) / 4.0
    return np.stack([scalars, pseudos], axis=1)  # [N, 2]


def extract_bivectors_batch(Ms: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Extract 6 bivector coefficients for batch."""
    bivectors = []
    for i in range(6):
        coeff = np.einsum('nij,ij->n', Ms, basis[5 + i]) / 4.0
        bivectors.append(coeff)
    return np.stack(bivectors, axis=1)  # [N, 6]


def decompose_batch(Ms: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Decompose batch of matrices into 16 Clifford coefficients."""
    coeffs = []
    for i in range(16):
        coeff = np.einsum('nij,ij->n', Ms, basis[i]) / 4.0
        coeffs.append(coeff)
    return np.stack(coeffs, axis=1)  # [N, 16]


def batch_cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Cosine similarity between two feature vectors."""
    na = np.linalg.norm(A)
    nb = np.linalg.norm(B)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(A, B) / (na * nb))


# =============================================================================
# TEST: Retrieval at Scale
# =============================================================================

def test_retrieval_at_scale():
    """
    Test retrieval accuracy at different scales.
    """
    print("\n" + "=" * 70)
    print("  TEST: Retrieval Accuracy at Scale")
    print("=" * 70)
    
    scales = [100, 500, 1000, 2000, 5000]
    n_queries = 50
    context_length = 512
    vocab_size = 50000
    
    rng = np.random.default_rng(42)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    results = {scale: {} for scale in scales}
    
    for n_patterns in scales:
        print(f"\n  === Scale: {n_patterns} patterns ===")
        
        # Generate patterns
        print(f"  Generating contexts...", end=" ", flush=True)
        start = time.time()
        contexts = []
        for _ in range(n_patterns):
            seq = rng.integers(0, vocab_size, size=context_length)
            mats = embeddings[seq]
            ctx = geometric_product_batch(mats, np)
            ctx = grace_operator(ctx, basis, np)
            contexts.append(ctx)
        contexts = np.array(contexts)
        print(f"done ({time.time()-start:.1f}s)")
        
        # Precompute features
        print(f"  Extracting features...", end=" ", flush=True)
        start = time.time()
        witnesses = extract_witness_batch(contexts, basis)  # [N, 2]
        bivectors = extract_bivectors_batch(contexts, basis)  # [N, 6]
        all_coeffs = decompose_batch(contexts, basis)  # [N, 16]
        
        # Combined witness + bivector (theory-inspired)
        even_grades = np.concatenate([
            witnesses,  # scalar + pseudo (what survives Grace)
            bivectors,  # vorticity (structural information)
        ], axis=1)  # [N, 8]
        print(f"done ({time.time()-start:.1f}s)")
        
        # Test each metric
        metrics = {
            'Witness (σ,p)': witnesses,
            'Vorticity (G2)': bivectors,
            'Even (G0+G2+G4)': even_grades,
            'Full (all 16)': all_coeffs,
        }
        
        for name, features in metrics.items():
            correct = 0
            
            # Normalize features for comparison
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            features_norm = features / norms
            
            # Test queries
            for i in range(n_queries):
                query = features_norm[i]
                # Compute similarity to all patterns
                sims = np.dot(features_norm, query)
                best_match = np.argmax(sims)
                if best_match == i:
                    correct += 1
            
            accuracy = correct / n_queries
            results[n_patterns][name] = accuracy
            print(f"    {name:<20}: {accuracy:.1%}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("  ACCURACY vs SCALE")
    print("=" * 70)
    print(f"  {'Metric':<20}", end="")
    for scale in scales:
        print(f" | {scale:>6}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for _ in scales:
        print(f"-+-{'-'*6}", end="")
    print()
    
    for metric in ['Witness (σ,p)', 'Vorticity (G2)', 'Even (G0+G2+G4)', 'Full (all 16)']:
        print(f"  {metric:<20}", end="")
        for scale in scales:
            acc = results[scale].get(metric, 0)
            print(f" | {acc:>5.1%}", end="")
        print()
    
    return results


# =============================================================================
# TEST: Feature Dimensionality Analysis
# =============================================================================

def test_effective_dimensionality():
    """
    Analyze the effective dimensionality of each feature space.
    
    Use SVD to see how many dimensions actually carry variance.
    """
    print("\n" + "=" * 70)
    print("  TEST: Effective Dimensionality")
    print("=" * 70)
    
    n_contexts = 1000
    context_length = 512
    vocab_size = 50000
    
    print(f"\n  Analyzing {n_contexts} contexts...")
    
    rng = np.random.default_rng(42)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    # Generate contexts
    contexts = []
    for _ in range(n_contexts):
        seq = rng.integers(0, vocab_size, size=context_length)
        mats = embeddings[seq]
        ctx = geometric_product_batch(mats, np)
        ctx = grace_operator(ctx, basis, np)
        contexts.append(ctx)
    contexts = np.array(contexts)
    
    # Extract features
    witnesses = extract_witness_batch(contexts, basis)
    bivectors = extract_bivectors_batch(contexts, basis)
    all_coeffs = decompose_batch(contexts, basis)
    
    even_grades = np.concatenate([witnesses, bivectors], axis=1)
    
    # Analyze each
    features = {
        'Witness (σ,p)': witnesses,
        'Vorticity (G2)': bivectors,
        'Even (G0+G2+G4)': even_grades,
        'Full (all 16)': all_coeffs,
    }
    
    print("\n  === EFFECTIVE DIMENSIONALITY ===")
    print(f"  {'Metric':<20} | {'Dims':>5} | {'Eff':>5} | {'Var Explained':<30}")
    print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*5}-+-{'-'*30}")
    
    for name, feat in features.items():
        # Center
        feat_centered = feat - feat.mean(axis=0)
        
        # SVD
        U, S, Vt = np.linalg.svd(feat_centered, full_matrices=False)
        
        # Variance explained
        var = S ** 2
        total_var = var.sum()
        var_ratio = var / total_var if total_var > 1e-10 else var
        
        # Cumulative variance
        cum_var = np.cumsum(var_ratio)
        
        # Effective dimensionality (dimensions needed for 95% variance)
        eff_dim = np.searchsorted(cum_var, 0.95) + 1
        eff_dim = min(eff_dim, len(var_ratio))
        
        # Format variance bars
        bars = [f"{v:.0%}" for v in var_ratio[:5]]
        
        print(f"  {name:<20} | {feat.shape[1]:>5} | {eff_dim:>5} | {', '.join(bars)}")
    
    return features


# =============================================================================
# TEST: Theoretical Optimal Weighting
# =============================================================================

def test_optimal_weighting():
    """
    Find the optimal weighting between witness and vorticity empirically.
    
    Since we know:
    - Witness captures semantics (what survives Grace)
    - Vorticity captures syntax (structural information)
    - Actual energy: 35% scalar, 46% bivector, 18% pseudo
    
    Test different combinations.
    """
    print("\n" + "=" * 70)
    print("  TEST: Optimal Weighting (Witness vs Vorticity)")
    print("=" * 70)
    
    n_patterns = 2000
    n_queries = 100
    context_length = 512
    vocab_size = 50000
    
    print(f"\n  Testing with {n_patterns} patterns, {n_queries} queries...")
    
    rng = np.random.default_rng(42)
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(vocab_size, xp=np)
    
    # Generate contexts
    print("  Generating contexts...", end=" ", flush=True)
    contexts = []
    for _ in range(n_patterns):
        seq = rng.integers(0, vocab_size, size=context_length)
        mats = embeddings[seq]
        ctx = geometric_product_batch(mats, np)
        ctx = grace_operator(ctx, basis, np)
        contexts.append(ctx)
    contexts = np.array(contexts)
    print("done")
    
    # Extract features
    witnesses = extract_witness_batch(contexts, basis)  # [N, 2]
    bivectors = extract_bivectors_batch(contexts, basis)  # [N, 6]
    
    # Normalize each
    w_norm = np.linalg.norm(witnesses, axis=1, keepdims=True)
    w_norm = np.maximum(w_norm, 1e-8)
    witnesses_n = witnesses / w_norm
    
    b_norm = np.linalg.norm(bivectors, axis=1, keepdims=True)
    b_norm = np.maximum(b_norm, 1e-8)
    bivectors_n = bivectors / b_norm
    
    # Test different weights for vorticity
    weights = [0.0, 0.1, 0.2, 0.3, PHI_INV_SQ, 0.4, 0.5, PHI_INV, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    
    for vort_w in weights:
        wit_w = 1.0 - vort_w
        
        correct = 0
        for i in range(n_queries):
            # Combined similarity: wit_w * witness_sim + vort_w * vorticity_sim
            wit_sims = np.dot(witnesses_n, witnesses_n[i])
            biv_sims = np.dot(bivectors_n, bivectors_n[i])
            
            combined = wit_w * wit_sims + vort_w * biv_sims
            best_match = np.argmax(combined)
            
            if best_match == i:
                correct += 1
        
        accuracy = correct / n_queries
        results.append((vort_w, accuracy))
        
        # Mark special weights
        marker = ""
        if abs(vort_w - PHI_INV) < 0.01:
            marker = " ← φ⁻¹"
        elif abs(vort_w - PHI_INV_SQ) < 0.01:
            marker = " ← φ⁻²"
        
        print(f"    Witness {wit_w:.1%} + Vorticity {vort_w:.1%}: {accuracy:.1%}{marker}")
    
    # Find optimal
    best = max(results, key=lambda x: x[1])
    print(f"\n  OPTIMAL: Vorticity weight = {best[0]:.1%} → {best[1]:.1%} accuracy")
    
    return results


# =============================================================================
# RUN ALL
# =============================================================================

def run_all():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  SIMILARITY METRICS AT SCALE".center(68) + "║")
    print("║" + "  Finding the theory-true optimal approach".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    test_effective_dimensionality()
    test_optimal_weighting()
    test_retrieval_at_scale()


if __name__ == "__main__":
    run_all()
