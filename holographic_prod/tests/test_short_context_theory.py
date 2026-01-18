#!/usr/bin/env python3
"""
Test: Short Context Theory Validation

HYPOTHESIS:
    The holographic architecture works for SHORT contexts (8-16 tokens)
    where:
    1. Witness content is preserved
    2. Attractors are diverse
    3. < 16 patterns per basin → interference is manageable
    
    Long sequences should be handled HIERARCHICALLY by the tower,
    not by single flat compositions.
    
    This test validates the theory at the scale it was designed for.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ, DTYPE
from holographic_prod.core.algebra import (
    build_clifford_basis,
    grace_operator,
    frobenius_cosine,
)
from holographic_prod.core.quotient import grace_stability
from holographic_prod.core.grounded_embeddings import so4_generators


def create_so4_embeddings(vocab_size: int, seed: int = 42) -> np.ndarray:
    """Create SO(4) embeddings."""
    np.random.seed(seed)
    embeddings = np.zeros((vocab_size, 4, 4), dtype=DTYPE)
    generators = so4_generators()
    
    from scipy.linalg import expm
    for i in range(vocab_size):
        coeffs = np.random.randn(6) * 0.5
        A = sum(c * g for c, g in zip(coeffs, generators))
        embeddings[i] = expm(A).astype(DTYPE)
    
    return embeddings


def embed_sequence(tokens: list, embeddings: np.ndarray) -> np.ndarray:
    """Compose tokens."""
    if not tokens:
        return np.eye(4, dtype=DTYPE)
    
    result = embeddings[tokens[0]].copy()
    for t in tokens[1:]:
        result = result @ embeddings[t]
    return result


def grace_until_stable(M: np.ndarray, basis: np.ndarray, max_iters: int = 30) -> np.ndarray:
    """Apply Grace until convergence."""
    for i in range(max_iters):
        M_new = grace_operator(M, basis)
        if np.linalg.norm(M_new - M, 'fro') < 1e-8:
            return M_new
        M = M_new
    return M


def test_short_context_retrieval():
    """Test retrieval with SHORT contexts (4-16 tokens)."""
    print("\n" + "="*70)
    print("TEST 1: Short Context Retrieval (Theory-True Scale)")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    context_lengths = [4, 8, 12, 16]
    
    for ctx_len in context_lengths:
        print(f"\n--- Context Length: {ctx_len} ---")
        
        # Store fewer patterns (theory: ~16 max per basin)
        n_patterns = 10
        
        memory = np.zeros((4, 4), dtype=DTYPE)
        stored_contexts = []
        stored_targets = []
        stored_raw = []
        stored_attractors = []
        
        for i in range(n_patterns):
            context = list(np.random.randint(0, 1000, size=ctx_len))
            target = np.random.randint(0, 1000)
            
            stored_contexts.append(context)
            stored_targets.append(target)
            
            raw = embed_sequence(context, embeddings)
            attractor = grace_until_stable(raw, basis)
            
            stored_raw.append(raw)
            stored_attractors.append(attractor)
            
            # Store using RAW context (current implementation)
            tgt_emb = embeddings[target]
            memory += PHI_INV * (raw @ tgt_emb)
        
        # Check diversity
        raw_sims = []
        attr_sims = []
        for i in range(n_patterns):
            for j in range(i + 1, n_patterns):
                raw_sims.append(abs(frobenius_cosine(stored_raw[i], stored_raw[j])))
                attr_sims.append(abs(frobenius_cosine(stored_attractors[i], stored_attractors[j])))
        
        print(f"Raw context diversity: mean|sim|={np.mean(raw_sims):.4f}")
        print(f"Attractor diversity:   mean|sim|={np.mean(attr_sims):.4f}")
        
        # Test retrieval with RAW contexts (current approach)
        print(f"\nRetrieval with RAW contexts:")
        correct_top1 = 0
        correct_top10 = 0
        
        for i in range(n_patterns):
            raw = stored_raw[i]
            retrieved = raw.T @ memory
            
            scores = [frobenius_cosine(retrieved, embeddings[t]) for t in range(1000)]
            ranking = sorted(range(1000), key=lambda t: -scores[t])
            
            if ranking[0] == stored_targets[i]:
                correct_top1 += 1
            if stored_targets[i] in ranking[:10]:
                correct_top10 += 1
        
        print(f"  Top-1:  {correct_top1}/{n_patterns} = {100*correct_top1/n_patterns:.0f}%")
        print(f"  Top-10: {correct_top10}/{n_patterns} = {100*correct_top10/n_patterns:.0f}%")
        
        # Test retrieval with ATTRACTOR (proposed approach)
        # Need to store with attractors too
        memory_attr = np.zeros((4, 4), dtype=DTYPE)
        for i in range(n_patterns):
            attractor = stored_attractors[i]
            tgt_emb = embeddings[stored_targets[i]]
            memory_attr += PHI_INV * (attractor @ tgt_emb)
        
        print(f"\nRetrieval with ATTRACTORS:")
        correct_top1 = 0
        correct_top10 = 0
        
        for i in range(n_patterns):
            attractor = stored_attractors[i]
            retrieved = attractor.T @ memory_attr
            
            scores = [frobenius_cosine(retrieved, embeddings[t]) for t in range(1000)]
            ranking = sorted(range(1000), key=lambda t: -scores[t])
            
            if ranking[0] == stored_targets[i]:
                correct_top1 += 1
            if stored_targets[i] in ranking[:10]:
                correct_top10 += 1
        
        print(f"  Top-1:  {correct_top1}/{n_patterns} = {100*correct_top1/n_patterns:.0f}%")
        print(f"  Top-10: {correct_top10}/{n_patterns} = {100*correct_top10/n_patterns:.0f}%")


def test_pattern_count_scaling():
    """Test how retrieval degrades with pattern count."""
    print("\n" + "="*70)
    print("TEST 2: Pattern Count Scaling (Finding the Limit)")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    context_length = 8  # Short context
    pattern_counts = [5, 10, 15, 20, 30, 50]
    
    print(f"\nContext length: {context_length}")
    print(f"\n{'Patterns':<10} {'Raw Top-10':<12} {'Attr Top-10':<12} {'Theory limit':<15}")
    print("-" * 49)
    
    for n_patterns in pattern_counts:
        memory_raw = np.zeros((4, 4), dtype=DTYPE)
        memory_attr = np.zeros((4, 4), dtype=DTYPE)
        
        stored_contexts = []
        stored_targets = []
        stored_raw = []
        stored_attractors = []
        
        for i in range(n_patterns):
            context = list(np.random.randint(0, 1000, size=context_length))
            target = np.random.randint(0, 1000)
            
            stored_contexts.append(context)
            stored_targets.append(target)
            
            raw = embed_sequence(context, embeddings)
            attractor = grace_until_stable(raw, basis)
            
            stored_raw.append(raw)
            stored_attractors.append(attractor)
            
            tgt_emb = embeddings[target]
            memory_raw += PHI_INV * (raw @ tgt_emb)
            memory_attr += PHI_INV * (attractor @ tgt_emb)
        
        # Test retrieval
        n_test = min(10, n_patterns)
        
        correct_raw = 0
        correct_attr = 0
        
        for i in range(n_test):
            # Raw
            retrieved = stored_raw[i].T @ memory_raw
            scores = [frobenius_cosine(retrieved, embeddings[t]) for t in range(1000)]
            ranking = sorted(range(1000), key=lambda t: -scores[t])
            if stored_targets[i] in ranking[:10]:
                correct_raw += 1
            
            # Attractor
            retrieved = stored_attractors[i].T @ memory_attr
            scores = [frobenius_cosine(retrieved, embeddings[t]) for t in range(1000)]
            ranking = sorted(range(1000), key=lambda t: -scores[t])
            if stored_targets[i] in ranking[:10]:
                correct_attr += 1
        
        raw_acc = 100 * correct_raw / n_test
        attr_acc = 100 * correct_attr / n_test
        
        # Theory: ~16 orthogonal patterns max in 16D space
        theory_limit = "✓ OK" if n_patterns <= 16 else "⚠ Beyond limit"
        
        print(f"{n_patterns:<10} {raw_acc:<12.0f}% {attr_acc:<12.0f}% {theory_limit:<15}")


def test_basin_routing_diversity():
    """Test if short contexts produce diverse basin keys."""
    print("\n" + "="*70)
    print("TEST 3: Basin Key Diversity (Short vs Long Contexts)")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    n_contexts = 50
    
    for ctx_len in [4, 8, 16, 64, 256]:
        # Generate contexts and compute basin keys
        attractors = []
        for _ in range(n_contexts):
            ctx = list(np.random.randint(0, 1000, size=ctx_len))
            raw = embed_sequence(ctx, embeddings)
            attractor = grace_until_stable(raw, basis)
            attractors.append(attractor)
        
        # Compute basin keys (simplified: sign of trace)
        # Real implementation uses quantized coefficients
        keys = []
        for attr in attractors:
            trace = np.trace(attr)
            # Simple 2-bin key based on sign
            key = 1 if trace > 0 else 0
            keys.append(key)
        
        unique_keys = len(set(keys))
        max_in_bin = max(keys.count(0), keys.count(1))
        
        # Better diversity metric: actual similarity clustering
        sims = []
        for i in range(n_contexts):
            for j in range(i + 1, n_contexts):
                sims.append(abs(frobenius_cosine(attractors[i], attractors[j])))
        
        print(f"\nContext length {ctx_len}:")
        print(f"  Unique basin keys (of 2): {unique_keys}")
        print(f"  Max patterns per bin: {max_in_bin}")
        print(f"  Mean |attractor similarity|: {np.mean(sims):.4f}")
        print(f"  Attractor norm mean: {np.mean([np.linalg.norm(a, 'fro') for a in attractors]):.4f}")


def test_theory_true_scale():
    """Test at the EXACT scale the theory was designed for."""
    print("\n" + "="*70)
    print("TEST 4: Theory-True Scale (Context=8, Patterns=10)")
    print("="*70)
    print("""
The theory claims:
1. Holographic superposition works for ~8-16 patterns
2. Grace basins provide generalization
3. SO(4) embeddings enable exact retrieval

Let's test at this EXACT scale.
""")
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(100)  # Small vocab for clean test
    
    context_length = 8
    n_patterns = 10
    
    # Create patterns with DISTINCT prefixes
    # This should maximize diversity
    stored_contexts = []
    stored_targets = []
    
    for i in range(n_patterns):
        # Each context starts with a unique token
        context = [i] + list(np.random.randint(0, 100, size=context_length-1))
        target = i  # Target = first token (for easy verification)
        stored_contexts.append(context)
        stored_targets.append(target)
    
    # Store using raw contexts
    memory = np.zeros((4, 4), dtype=DTYPE)
    stored_raw = []
    
    for i in range(n_patterns):
        raw = embed_sequence(stored_contexts[i], embeddings)
        stored_raw.append(raw)
        tgt_emb = embeddings[stored_targets[i]]
        memory += PHI_INV * (raw @ tgt_emb)
    
    print(f"Memory norm: {np.linalg.norm(memory):.4f}")
    
    # Check raw context diversity
    print(f"\nRaw context pairwise |similarities|:")
    sims = []
    for i in range(n_patterns):
        for j in range(i + 1, n_patterns):
            sim = abs(frobenius_cosine(stored_raw[i], stored_raw[j]))
            sims.append(sim)
    print(f"  Mean: {np.mean(sims):.4f}, Min: {np.min(sims):.4f}, Max: {np.max(sims):.4f}")
    
    # Test exact retrieval
    print(f"\nExact retrieval test:")
    for i in range(n_patterns):
        raw = stored_raw[i]
        retrieved = raw.T @ memory
        
        # Score all targets
        scores = [(t, frobenius_cosine(retrieved, embeddings[t])) for t in range(100)]
        scores.sort(key=lambda x: -x[1])
        
        true_target = stored_targets[i]
        true_rank = next(j for j, (t, s) in enumerate(scores) if t == true_target)
        top3 = [t for t, s in scores[:3]]
        
        status = "✓" if true_rank < 3 else "✗"
        print(f"  Pattern {i}: target={true_target}, rank={true_rank}, top3={top3} {status}")


def main():
    print("\n" + "="*70)
    print("   SHORT CONTEXT THEORY VALIDATION")
    print("="*70)
    print("""
CORE INSIGHT:
    The holographic architecture was designed for SHORT contexts (8-16 tokens)
    with FEW patterns per basin (~16 max).
    
    Long sequences should use the TOWER HIERARCHY, not flat composition.
    
    This test validates the theory at its designed scale.
""")
    
    test_short_context_retrieval()
    test_pattern_count_scaling()
    test_basin_routing_diversity()
    test_theory_true_scale()
    
    print("\n" + "="*70)
    print("   CONCLUSIONS")
    print("="*70)


if __name__ == "__main__":
    main()
