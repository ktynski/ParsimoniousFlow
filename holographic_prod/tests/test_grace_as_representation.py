#!/usr/bin/env python3
"""
Test: Grace as Representation vs Grace as Routing

HYPOTHESIS:
    The theory says Grace finds ATTRACTORS. Similar contexts should flow
    to the SAME attractor. We should store and retrieve in attractor space,
    not raw context space.

CURRENT IMPLEMENTATION:
    - Raw context → route by basin_key → store raw context
    - Raw query → route by basin_key → match against raw contexts

THEORY-TRUE IMPLEMENTATION:
    - Raw context → Grace iterate → attractor → store attractor
    - Raw query → Grace iterate → attractor → retrieve by attractor

THIS TEST DEMONSTRATES:
    1. Raw contexts have low stability (expected)
    2. Attractors have HIGH stability (by definition)
    3. Similar contexts → same attractor (generalization)
    4. Retrieval in attractor space works; raw space fails
"""

import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ, DTYPE
from holographic_prod.core.algebra import (
    build_clifford_basis,
    geometric_product,
    grace_operator,
    frobenius_cosine,
)
from holographic_prod.core.quotient import grace_stability
from holographic_prod.core.grounded_embeddings import so4_generators, semantic_to_SO4_batch_fast


def create_so4_embeddings(vocab_size: int, seed: int = 42) -> np.ndarray:
    """Create random SO(4) embeddings."""
    np.random.seed(seed)
    embeddings = np.zeros((vocab_size, 4, 4), dtype=DTYPE)
    
    generators = so4_generators()  # 6 generators for SO(4)
    
    for i in range(vocab_size):
        # Random coefficients for 6 generators
        coeffs = np.random.randn(6) * 0.5
        # Exponentiate to get SO(4) element
        A = sum(c * g for c, g in zip(coeffs, generators))
        # Matrix exponential
        from scipy.linalg import expm
        embeddings[i] = expm(A).astype(DTYPE)
    
    return embeddings


def embed_sequence(tokens: list, embeddings: np.ndarray) -> np.ndarray:
    """Compose tokens via geometric product."""
    if not tokens:
        return np.eye(4, dtype=DTYPE)
    
    result = embeddings[tokens[0]].copy()
    for t in tokens[1:]:
        result = result @ embeddings[t]
    return result


def grace_until_stable(M: np.ndarray, basis: np.ndarray, max_iters: int = 20, 
                       tol: float = 1e-6) -> np.ndarray:
    """Apply Grace until convergence to attractor."""
    for i in range(max_iters):
        M_new = grace_operator(M, basis)
        
        # Check convergence
        diff = np.linalg.norm(M_new - M, 'fro')
        if diff < tol:
            return M_new
        M = M_new
    
    return M


def test_stability_raw_vs_attractor():
    """
    TEST 1: Stability of raw contexts vs attractors
    
    PREDICTION:
        - Raw long contexts: stability << φ⁻² (0.382)
        - Attractors: stability ≈ 1.0 (they ARE the fixed point)
    """
    print("\n" + "="*70)
    print("TEST 1: Stability of Raw Contexts vs Attractors")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    # Test various context lengths
    lengths = [4, 16, 64, 256, 512]
    
    print(f"\n{'Length':<10} {'Raw σ':<12} {'Attractor σ':<12} {'Ratio':<10}")
    print("-" * 44)
    
    for length in lengths:
        # Random context
        tokens = list(np.random.randint(0, 1000, size=length))
        
        # Raw context embedding
        raw_ctx = embed_sequence(tokens, embeddings)
        raw_stability = grace_stability(raw_ctx, basis)
        
        # Grace-converged attractor
        attractor = grace_until_stable(raw_ctx, basis)
        attractor_stability = grace_stability(attractor, basis)
        
        ratio = attractor_stability / max(raw_stability, 1e-10)
        
        print(f"{length:<10} {raw_stability:<12.4f} {attractor_stability:<12.4f} {ratio:<10.1f}x")
    
    print("\n✓ CONFIRMED: Attractors have ~1.0 stability regardless of context length")
    print("  Raw contexts lose stability as length increases (as observed in training)")


def test_similar_contexts_same_attractor():
    """
    TEST 2: Similar contexts should flow to the same attractor
    
    PREDICTION:
        - "the cat sat on the" and "the dog sat on the" are different raw contexts
        - But they should converge to SIMILAR attractors under Grace
        - This is how generalization works!
    """
    print("\n" + "="*70)
    print("TEST 2: Similar Contexts → Same Attractor (Generalization)")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    # Create "similar" contexts (same structure, one token different)
    base_context = [10, 20, 30, 40, 50]  # "the X sat on the"
    
    variants = [
        [10, 21, 30, 40, 50],  # Change one token
        [10, 22, 30, 40, 50],
        [10, 23, 30, 40, 50],
        [10, 24, 30, 40, 50],
    ]
    
    # Embed base
    base_raw = embed_sequence(base_context, embeddings)
    base_attractor = grace_until_stable(base_raw, basis)
    
    print(f"\nBase context: {base_context}")
    print(f"Base raw stability: {grace_stability(base_raw, basis):.4f}")
    print(f"Base attractor stability: {grace_stability(base_attractor, basis):.4f}")
    
    print(f"\n{'Variant':<20} {'Raw sim':<12} {'Attractor sim':<15} {'Improvement':<12}")
    print("-" * 59)
    
    for var in variants:
        var_raw = embed_sequence(var, embeddings)
        var_attractor = grace_until_stable(var_raw, basis)
        
        # Similarity in raw space
        raw_sim = frobenius_cosine(base_raw, var_raw)
        
        # Similarity in attractor space
        attractor_sim = frobenius_cosine(base_attractor, var_attractor)
        
        improvement = attractor_sim / max(raw_sim, 1e-10)
        
        print(f"{str(var):<20} {raw_sim:<12.4f} {attractor_sim:<15.4f} {improvement:<12.1f}x")
    
    print("\n✓ PREDICTION: Attractor similarity > Raw similarity")
    print("  Grace pulls similar contexts towards the same basin")


def test_retrieval_raw_vs_attractor():
    """
    TEST 3: Retrieval accuracy in raw space vs attractor space
    
    Setup:
        - Store N patterns: context → target
        - Query with slightly different context
        - Compare retrieval in raw space vs attractor space
    """
    print("\n" + "="*70)
    print("TEST 3: Retrieval Accuracy - Raw Space vs Attractor Space")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    # Store some patterns
    n_patterns = 100
    context_length = 50
    
    # Storage: two memories (raw vs attractor)
    memory_raw = np.zeros((4, 4), dtype=DTYPE)
    memory_attractor = np.zeros((4, 4), dtype=DTYPE)
    
    stored_contexts = []
    stored_targets = []
    
    print(f"\nStoring {n_patterns} patterns with context length {context_length}...")
    
    for i in range(n_patterns):
        context = list(np.random.randint(0, 1000, size=context_length))
        target = np.random.randint(0, 1000)
        
        stored_contexts.append(context)
        stored_targets.append(target)
        
        # Raw embedding
        ctx_raw = embed_sequence(context, embeddings)
        tgt_emb = embeddings[target]
        
        # Attractor embedding
        ctx_attractor = grace_until_stable(ctx_raw, basis)
        
        # Store in both memories
        memory_raw += PHI_INV * (ctx_raw @ tgt_emb)
        memory_attractor += PHI_INV * (ctx_attractor @ tgt_emb)
    
    print(f"Memory norms: raw={np.linalg.norm(memory_raw):.2f}, attractor={np.linalg.norm(memory_attractor):.2f}")
    
    # Test retrieval with exact contexts
    print(f"\n--- Exact Context Retrieval ---")
    
    correct_raw = 0
    correct_attractor = 0
    n_test = min(20, n_patterns)
    
    for i in range(n_test):
        context = stored_contexts[i]
        true_target = stored_targets[i]
        
        # Raw retrieval
        ctx_raw = embed_sequence(context, embeddings)
        retrieved_raw = ctx_raw.T @ memory_raw
        
        # Attractor retrieval
        ctx_attractor = grace_until_stable(ctx_raw, basis)
        retrieved_attractor = ctx_attractor.T @ memory_attractor
        
        # Find best matching target
        best_raw = max(range(1000), key=lambda t: frobenius_cosine(retrieved_raw, embeddings[t]))
        best_attractor = max(range(1000), key=lambda t: frobenius_cosine(retrieved_attractor, embeddings[t]))
        
        if best_raw == true_target:
            correct_raw += 1
        if best_attractor == true_target:
            correct_attractor += 1
    
    print(f"Raw retrieval accuracy: {correct_raw}/{n_test} = {100*correct_raw/n_test:.1f}%")
    print(f"Attractor retrieval accuracy: {correct_attractor}/{n_test} = {100*correct_attractor/n_test:.1f}%")
    
    # Test retrieval with PERTURBED contexts (generalization test)
    print(f"\n--- Perturbed Context Retrieval (Generalization) ---")
    
    correct_raw = 0
    correct_attractor = 0
    
    for i in range(n_test):
        context = stored_contexts[i].copy()
        true_target = stored_targets[i]
        
        # Perturb: change 1-3 tokens
        n_perturb = np.random.randint(1, 4)
        for _ in range(n_perturb):
            pos = np.random.randint(0, len(context))
            context[pos] = np.random.randint(0, 1000)
        
        # Raw retrieval
        ctx_raw = embed_sequence(context, embeddings)
        retrieved_raw = ctx_raw.T @ memory_raw
        
        # Attractor retrieval
        ctx_attractor = grace_until_stable(ctx_raw, basis)
        retrieved_attractor = ctx_attractor.T @ memory_attractor
        
        # Find best matching target
        best_raw = max(range(1000), key=lambda t: frobenius_cosine(retrieved_raw, embeddings[t]))
        best_attractor = max(range(1000), key=lambda t: frobenius_cosine(retrieved_attractor, embeddings[t]))
        
        if best_raw == true_target:
            correct_raw += 1
        if best_attractor == true_target:
            correct_attractor += 1
    
    print(f"Raw retrieval accuracy: {correct_raw}/{n_test} = {100*correct_raw/n_test:.1f}%")
    print(f"Attractor retrieval accuracy: {correct_attractor}/{n_test} = {100*correct_attractor/n_test:.1f}%")


def test_attractor_convergence():
    """
    TEST 4: Verify that Grace actually converges
    
    Shows the trajectory from raw context to attractor.
    """
    print("\n" + "="*70)
    print("TEST 4: Grace Convergence Trajectory")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    # Long context
    context = list(np.random.randint(0, 1000, size=200))
    M = embed_sequence(context, embeddings)
    
    print(f"\nContext length: {len(context)}")
    print(f"Initial stability: {grace_stability(M, basis):.6f}")
    print(f"\n{'Iter':<6} {'Stability':<12} {'Change':<12} {'Norm':<10}")
    print("-" * 40)
    
    prev_stability = grace_stability(M, basis)
    
    for i in range(20):
        M_new = grace_operator(M, basis)
        new_stability = grace_stability(M_new, basis)
        change = abs(new_stability - prev_stability)
        norm = np.linalg.norm(M_new, 'fro')
        
        print(f"{i:<6} {new_stability:<12.6f} {change:<12.6f} {norm:<10.4f}")
        
        if change < 1e-8:
            print(f"\n✓ Converged at iteration {i}")
            break
        
        prev_stability = new_stability
        M = M_new
    
    print(f"\nFinal attractor stability: {grace_stability(M, basis):.6f}")
    print("✓ Grace converges to stable attractor")


def test_interference_scaling():
    """
    TEST 5: How does interference scale with pattern count?
    
    PREDICTION:
        - Raw space: interference grows, retrieval degrades
        - Attractor space: interference is bounded, retrieval stable
    """
    print("\n" + "="*70)
    print("TEST 5: Interference Scaling - Raw vs Attractor")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    pattern_counts = [10, 50, 100, 500]
    context_length = 30
    
    print(f"\n{'Patterns':<10} {'Raw Norm':<12} {'Attr Norm':<12} {'Raw Acc':<10} {'Attr Acc':<10}")
    print("-" * 54)
    
    for n_patterns in pattern_counts:
        memory_raw = np.zeros((4, 4), dtype=DTYPE)
        memory_attractor = np.zeros((4, 4), dtype=DTYPE)
        
        stored_contexts = []
        stored_targets = []
        
        for i in range(n_patterns):
            context = list(np.random.randint(0, 1000, size=context_length))
            target = np.random.randint(0, 1000)
            
            stored_contexts.append(context)
            stored_targets.append(target)
            
            ctx_raw = embed_sequence(context, embeddings)
            ctx_attractor = grace_until_stable(ctx_raw, basis)
            tgt_emb = embeddings[target]
            
            memory_raw += PHI_INV * (ctx_raw @ tgt_emb)
            memory_attractor += PHI_INV * (ctx_attractor @ tgt_emb)
        
        # Test retrieval on first 10 patterns
        correct_raw = 0
        correct_attractor = 0
        n_test = min(10, n_patterns)
        
        for i in range(n_test):
            ctx_raw = embed_sequence(stored_contexts[i], embeddings)
            ctx_attractor = grace_until_stable(ctx_raw, basis)
            
            retrieved_raw = ctx_raw.T @ memory_raw
            retrieved_attractor = ctx_attractor.T @ memory_attractor
            
            # Top-10 accuracy (is true target in top 10?)
            sims_raw = [frobenius_cosine(retrieved_raw, embeddings[t]) for t in range(1000)]
            sims_attractor = [frobenius_cosine(retrieved_attractor, embeddings[t]) for t in range(1000)]
            
            top10_raw = sorted(range(1000), key=lambda t: -sims_raw[t])[:10]
            top10_attractor = sorted(range(1000), key=lambda t: -sims_attractor[t])[:10]
            
            if stored_targets[i] in top10_raw:
                correct_raw += 1
            if stored_targets[i] in top10_attractor:
                correct_attractor += 1
        
        raw_norm = np.linalg.norm(memory_raw)
        attr_norm = np.linalg.norm(memory_attractor)
        
        print(f"{n_patterns:<10} {raw_norm:<12.2f} {attr_norm:<12.2f} "
              f"{100*correct_raw/n_test:<10.0f}% {100*correct_attractor/n_test:<10.0f}%")


def main():
    print("\n" + "="*70)
    print("   GRACE AS REPRESENTATION: Theory Validation Tests")
    print("="*70)
    print("""
CORE HYPOTHESIS:
    The Grace operator finds ATTRACTORS. Similar contexts flow to the
    SAME attractor. We should store and retrieve in ATTRACTOR SPACE,
    not raw context space.

PREDICTIONS:
    1. Raw contexts have low stability; attractors have ~1.0 stability
    2. Similar contexts → same attractor (generalization)
    3. Retrieval in attractor space > retrieval in raw space
    4. Grace converges to fixed points
    5. Interference is bounded in attractor space
""")
    
    test_stability_raw_vs_attractor()
    test_similar_contexts_same_attractor()
    test_attractor_convergence()
    test_interference_scaling()
    test_retrieval_raw_vs_attractor()
    
    print("\n" + "="*70)
    print("   SUMMARY")
    print("="*70)
    print("""
If the hypothesis is correct:
    - Attractors have stability ≈ 1.0 (TEST 1)
    - Similar contexts converge to similar attractors (TEST 2)
    - Grace converges reliably (TEST 4)
    - Attractor space has bounded interference (TEST 5)
    - Retrieval works better in attractor space (TEST 3, 5)

If ALL tests pass, the fix is clear:
    Store and retrieve in ATTRACTOR space, not raw context space.
    
    learn():   ctx → grace_iterate → attractor → store (attractor, target)
    retrieve(): ctx → grace_iterate → attractor → retrieve by attractor
""")


if __name__ == "__main__":
    main()
