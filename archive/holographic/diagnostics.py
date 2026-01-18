"""
Diagnostic Tools for Understanding What Level 1 Actually Learns
================================================================

KEY QUESTION: Why are metrics excellent but generation incoherent?

HYPOTHESIS: Level 1 learns statistical co-occurrence, not semantics.
    - Contexts with shared words → similar representations
    - But shared words ≠ shared meaning
    - So retrieval works (similar pattern) but semantics don't transfer

DIAGNOSTICS:
1. Semantic Coherence: Do contexts predicting the same word cluster?
2. Witness Stability: Is the self-pointer stable across contexts?
3. Grade Variance: Which grades differentiate, which stay stable?
4. Compositional Test: Does geometric product create meaningful structure?
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from .constants import PHI, PHI_INV, MATRIX_DIM
from .algebra import (
    build_clifford_basis, geometric_product_batch, 
    frobenius_similarity_batch, initialize_all_embeddings
)
from .quotient import (
    witness_pointer, compute_witness_stability, 
    compute_grade_variance, quotient_similarity
)

Array = np.ndarray
ArrayModule = type(np)


# =============================================================================
# SEMANTIC COHERENCE — Do contexts predicting the same word cluster?
# =============================================================================

def semantic_coherence_test(
    contexts: List[List[int]],  # List of context token sequences
    targets: List[int],         # Target word for each context
    embeddings: Array,          # [vocab, 4, 4] embeddings
    basis: Array,               # [16, 4, 4] basis
    xp: ArrayModule = np,
    max_pairs: int = 1000
) -> Dict[str, float]:
    """
    Test: Do contexts that predict the same target have similar representations?
    
    This is THE key test for semantic learning.
    
    Args:
        contexts: List of context sequences (each is [tok1, tok2, ...])
        targets: Target word for each context
        embeddings: Word embeddings
        basis: Clifford basis
        max_pairs: Max same-target pairs to compare
        
    Returns:
        Dict with same_target_sim, diff_target_sim, separation
    """
    # Group contexts by target
    target_to_contexts: Dict[int, List[int]] = defaultdict(list)
    for idx, target in enumerate(targets):
        target_to_contexts[target].append(idx)
    
    # Compute context representations
    def embed_context(ctx_tokens):
        if len(ctx_tokens) == 0:
            return xp.eye(MATRIX_DIM, dtype=xp.float64)
        if len(ctx_tokens) == 1:
            return embeddings[ctx_tokens[0] % len(embeddings)]
        token_mats = embeddings[xp.array(ctx_tokens) % len(embeddings)]
        return geometric_product_batch(token_mats, xp)
    
    context_reps = [embed_context(ctx) for ctx in contexts]
    
    # Sample same-target pairs
    same_sims = []
    diff_sims = []
    
    rng = np.random.default_rng(42)
    
    # Same-target pairs
    targets_with_multiple = [t for t, idxs in target_to_contexts.items() if len(idxs) >= 2]
    
    for _ in range(min(max_pairs, len(targets_with_multiple) * 10)):
        if not targets_with_multiple:
            break
        t = rng.choice(targets_with_multiple)
        idxs = target_to_contexts[t]
        if len(idxs) < 2:
            continue
        i, j = rng.choice(len(idxs), size=2, replace=False)
        rep_i = context_reps[idxs[i]]
        rep_j = context_reps[idxs[j]]
        sim = float(xp.sum(rep_i * rep_j))  # Frobenius-like
        same_sims.append(sim)
    
    # Different-target pairs
    all_targets = list(target_to_contexts.keys())
    for _ in range(min(max_pairs, len(contexts) * 2)):
        if len(all_targets) < 2:
            break
        t1, t2 = rng.choice(all_targets, size=2, replace=False)
        idxs1 = target_to_contexts[t1]
        idxs2 = target_to_contexts[t2]
        if not idxs1 or not idxs2:
            continue
        i = rng.choice(idxs1)
        j = rng.choice(idxs2)
        rep_i = context_reps[i]
        rep_j = context_reps[j]
        sim = float(xp.sum(rep_i * rep_j))
        diff_sims.append(sim)
    
    same_mean = float(np.mean(same_sims)) if same_sims else 0.0
    diff_mean = float(np.mean(diff_sims)) if diff_sims else 0.0
    
    return {
        'same_target_sim': same_mean,
        'diff_target_sim': diff_mean,
        'separation': same_mean - diff_mean,
        'num_same_pairs': len(same_sims),
        'num_diff_pairs': len(diff_sims),
    }


# =============================================================================
# WITNESS STABILITY ACROSS CONTEXTS
# =============================================================================

def witness_stability_analysis(
    contexts: List[List[int]],
    embeddings: Array,
    basis: Array,
    xp: ArrayModule = np,
    max_contexts: int = 500
) -> Dict[str, float]:
    """
    Analyze witness stability across contexts.
    
    If witness varies widely → no stable "self-pointer"
    If witness is stable → good anchor for learning
    
    Args:
        contexts: List of context sequences
        embeddings: Word embeddings
        basis: Clifford basis
        max_contexts: Max contexts to analyze
        
    Returns:
        Dict with witness stats
    """
    def embed_context(ctx_tokens):
        if len(ctx_tokens) == 0:
            return xp.eye(MATRIX_DIM, dtype=xp.float64)
        if len(ctx_tokens) == 1:
            return embeddings[ctx_tokens[0] % len(embeddings)]
        token_mats = embeddings[xp.array(ctx_tokens) % len(embeddings)]
        return geometric_product_batch(token_mats, xp)
    
    # Compute context representations
    n = min(len(contexts), max_contexts)
    context_reps = [embed_context(contexts[i]) for i in range(n)]
    
    # Compute witnesses
    witnesses = [witness_pointer(rep, basis, xp) for rep in context_reps]
    witnesses_arr = xp.stack(witnesses, axis=0)  # [n, 4, 4]
    
    # Witness similarity matrix
    sims = []
    for i in range(min(n, 100)):
        for j in range(i + 1, min(n, 100)):
            sim = float(xp.sum(witnesses_arr[i] * witnesses_arr[j]))
            sims.append(sim)
    
    if not sims:
        return {'witness_mean_sim': 0.0, 'witness_std': 0.0, 'num_pairs': 0}
    
    return {
        'witness_mean_sim': float(np.mean(sims)),
        'witness_std': float(np.std(sims)),
        'num_pairs': len(sims),
    }


# =============================================================================
# GRADE-WISE ANALYSIS
# =============================================================================

def grade_analysis(
    contexts: List[List[int]],
    targets: List[int],
    embeddings: Array,
    basis: Array,
    xp: ArrayModule = np,
    max_contexts: int = 200
) -> Dict[str, Dict[str, float]]:
    """
    Analyze which grades differentiate by target vs stay stable.
    
    Theory predicts:
    - Grade 0 (scalar): stable anchor
    - Grade 1-2: differentiate by syntax/semantics
    - Grade 4 (pseudoscalar): stable via φ⁻¹ scaling
    
    Args:
        contexts: Context sequences
        targets: Target words
        embeddings: Word embeddings
        basis: Clifford basis
        
    Returns:
        Dict mapping grade -> variance stats
    """
    from .quotient import GRADE_IDXS
    
    def embed_context(ctx_tokens):
        if len(ctx_tokens) == 0:
            return xp.eye(MATRIX_DIM, dtype=xp.float64)
        if len(ctx_tokens) == 1:
            return embeddings[ctx_tokens[0] % len(embeddings)]
        token_mats = embeddings[xp.array(ctx_tokens) % len(embeddings)]
        return geometric_product_batch(token_mats, xp)
    
    n = min(len(contexts), max_contexts)
    
    # Extract coefficients for each context
    all_coeffs = []
    for i in range(n):
        rep = embed_context(contexts[i])
        # Convert matrix to 16-component coefficients
        coeffs = extract_coeffs_from_matrix(rep, basis, xp)
        all_coeffs.append(coeffs)
    
    all_coeffs = xp.stack(all_coeffs, axis=0)  # [n, 16]
    
    # Compute variance by grade
    results = {}
    for grade, idxs in GRADE_IDXS.items():
        grade_coeffs = all_coeffs[:, idxs]  # [n, num_components_in_grade]
        variance = float(xp.var(grade_coeffs))
        mean_abs = float(xp.mean(xp.abs(grade_coeffs)))
        results[f'grade_{grade}'] = {
            'variance': variance,
            'mean_abs': mean_abs,
        }
    
    return results


def extract_coeffs_from_matrix(M: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    Extract 16-component coefficients from 4x4 matrix.
    
    Uses the trace trick: coeff_i = (1/4) * Tr(M @ basis[i])
    (Works because Tr(basis[i] @ basis[j]) = 4 * delta_ij for orthonormal basis)
    """
    coeffs = xp.zeros(16, dtype=xp.float64)
    for i in range(16):
        # Tr(M @ B_i) / 4
        coeffs[i] = xp.trace(M @ basis[i]) / 4.0
    return coeffs


# =============================================================================
# COMPOSITIONAL TEST — Does geometric product add meaningful structure?
# =============================================================================

def compositional_test(
    word_pairs: List[Tuple[int, int]],  # Pairs like (dog, cat), (run, walk)
    embeddings: Array,
    basis: Array,
    xp: ArrayModule = np
) -> Dict[str, float]:
    """
    Test: Does geometric product of related words create related patterns?
    
    If A~B (semantically similar) and C~D, does A*C ~ B*D?
    
    Args:
        word_pairs: List of semantically related word pairs
        embeddings: Word embeddings
        basis: Clifford basis
        
    Returns:
        Dict with compositional coherence metrics
    """
    if len(word_pairs) < 2:
        return {'compositional_sim': 0.0, 'random_sim': 0.0, 'num_pairs': 0}
    
    # Compute compositions
    compositions = []
    for w1, w2 in word_pairs:
        e1 = embeddings[w1 % len(embeddings)]
        e2 = embeddings[w2 % len(embeddings)]
        # Use matrix multiplication as geometric product
        comp = e1 @ e2
        compositions.append(comp)
    
    # Compare compositions across pairs
    sims = []
    for i in range(len(compositions)):
        for j in range(i + 1, len(compositions)):
            sim = float(xp.sum(compositions[i] * compositions[j]))
            sims.append(sim)
    
    # Compare with random compositions
    rng = np.random.default_rng(42)
    random_sims = []
    for _ in range(min(100, len(sims))):
        i, j = rng.integers(0, len(embeddings), size=2)
        k, l = rng.integers(0, len(embeddings), size=2)
        c1 = embeddings[i] @ embeddings[j]
        c2 = embeddings[k] @ embeddings[l]
        sim = float(xp.sum(c1 * c2))
        random_sims.append(sim)
    
    return {
        'compositional_sim': float(np.mean(sims)) if sims else 0.0,
        'random_sim': float(np.mean(random_sims)) if random_sims else 0.0,
        'num_pairs': len(sims),
    }


# =============================================================================
# FULL DIAGNOSTIC SUITE
# =============================================================================

def run_level1_diagnostics(
    contexts: List[List[int]],
    targets: List[int],
    embeddings: Array,
    basis: Array,
    xp: ArrayModule = np,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete Level 1 diagnostic suite.
    
    Args:
        contexts: Context sequences
        targets: Target words
        embeddings: Word embeddings
        basis: Clifford basis
        verbose: Print results
        
    Returns:
        Dict with all diagnostic results
    """
    results = {}
    
    if verbose:
        print("=" * 60)
        print("LEVEL 1 DIAGNOSTICS")
        print("=" * 60)
    
    # 1. Semantic coherence
    if verbose:
        print("\n1. Semantic Coherence Test")
    sc = semantic_coherence_test(contexts, targets, embeddings, basis, xp)
    results['semantic_coherence'] = sc
    if verbose:
        print(f"   Same-target similarity: {sc['same_target_sim']:.4f}")
        print(f"   Diff-target similarity: {sc['diff_target_sim']:.4f}")
        print(f"   Separation: {sc['separation']:.4f}")
    
    # 2. Witness stability
    if verbose:
        print("\n2. Witness Stability Analysis")
    ws = witness_stability_analysis(contexts, embeddings, basis, xp)
    results['witness_stability'] = ws
    if verbose:
        print(f"   Mean witness similarity: {ws['witness_mean_sim']:.4f}")
        print(f"   Witness std: {ws['witness_std']:.4f}")
    
    # 3. Grade analysis
    if verbose:
        print("\n3. Grade-wise Variance Analysis")
    ga = grade_analysis(contexts, targets, embeddings, basis, xp)
    results['grade_analysis'] = ga
    if verbose:
        for grade, stats in ga.items():
            print(f"   {grade}: var={stats['variance']:.4f}, mean_abs={stats['mean_abs']:.4f}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("DIAGNOSIS:")
        
        # Interpret results
        sep = sc['separation']
        wit_sim = ws['witness_mean_sim']
        
        if sep < 0.01:
            print("  ⚠️  LOW SEMANTIC SEPARATION")
            print("     Contexts predicting same word don't cluster.")
            print("     → Need semantic signal (contrastive loss or Level 2)")
        else:
            print(f"  ✓ Semantic separation: {sep:.4f}")
        
        if wit_sim > 0.95:
            print("  ✓ Witness highly stable (good anchor)")
        elif wit_sim > 0.8:
            print(f"  ⚠️ Witness moderately stable ({wit_sim:.2f})")
        else:
            print(f"  ✗ Witness unstable ({wit_sim:.2f}) - identity bias may help")
        
        print("=" * 60)
    
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'semantic_coherence_test',
    'witness_stability_analysis',
    'grade_analysis',
    'extract_coeffs_from_matrix',
    'compositional_test',
    'run_level1_diagnostics',
]
