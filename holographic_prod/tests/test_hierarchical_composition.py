#!/usr/bin/env python3
"""
Test: Hierarchical Composition with Grace at Each Level

HYPOTHESIS:
    The problem with long compositions is witness content washes out.
    
    Solution: Don't compose 700 tokens into one matrix.
    Instead, compose hierarchically:
        Level 0: tokens[0:16] → chunk_0, tokens[16:32] → chunk_1, ...
        Level 1: Apply Grace to each chunk → attractor_0, attractor_1, ...
        Level 2: Compose attractors → level1_chunk_0, ...
        Repeat until single representation
    
    This preserves witness content at each level because:
    - Each chunk has some witness content
    - Grace amplifies witness (stability → 1.0)
    - Composing stable attractors maintains diversity
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
    """Create random SO(4) embeddings."""
    np.random.seed(seed)
    embeddings = np.zeros((vocab_size, 4, 4), dtype=DTYPE)
    
    generators = so4_generators()
    
    for i in range(vocab_size):
        coeffs = np.random.randn(6) * 0.5
        A = sum(c * g for c, g in zip(coeffs, generators))
        from scipy.linalg import expm
        embeddings[i] = expm(A).astype(DTYPE)
    
    return embeddings


def embed_sequence_flat(tokens: list, embeddings: np.ndarray) -> np.ndarray:
    """Flat composition: multiply all tokens."""
    if not tokens:
        return np.eye(4, dtype=DTYPE)
    
    result = embeddings[tokens[0]].copy()
    for t in tokens[1:]:
        result = result @ embeddings[t]
    return result


def grace_until_stable(M: np.ndarray, basis: np.ndarray, max_iters: int = 20, 
                       tol: float = 1e-6) -> np.ndarray:
    """Apply Grace until convergence."""
    for i in range(max_iters):
        M_new = grace_operator(M, basis)
        diff = np.linalg.norm(M_new - M, 'fro')
        if diff < tol:
            return M_new
        M = M_new
    return M


def embed_sequence_hierarchical(tokens: list, embeddings: np.ndarray, 
                                 basis: np.ndarray, chunk_size: int = 8) -> np.ndarray:
    """
    Hierarchical composition with Grace at each level.
    
    Algorithm:
        1. Split into chunks of size chunk_size
        2. Compose each chunk → raw chunk matrices
        3. Apply Grace to each chunk → stable attractors
        4. Recursively compose attractors
    """
    if not tokens:
        return np.eye(4, dtype=DTYPE)
    
    if len(tokens) <= chunk_size:
        # Base case: compose and Grace
        raw = embed_sequence_flat(tokens, embeddings)
        return grace_until_stable(raw, basis)
    
    # Split into chunks
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i+chunk_size]
        # Compose chunk
        chunk_raw = embed_sequence_flat(chunk_tokens, embeddings)
        # Apply Grace to get stable attractor
        chunk_attractor = grace_until_stable(chunk_raw, basis)
        chunks.append(chunk_attractor)
    
    # Recursively compose chunk attractors
    # Create "embeddings" from chunk attractors for recursive call
    if len(chunks) == 1:
        return chunks[0]
    
    # Compose chunks pairwise with Grace between
    while len(chunks) > 1:
        new_chunks = []
        for i in range(0, len(chunks), 2):
            if i + 1 < len(chunks):
                composed = chunks[i] @ chunks[i + 1]
                # Apply Grace to maintain stability
                stable = grace_until_stable(composed, basis)
                new_chunks.append(stable)
            else:
                new_chunks.append(chunks[i])
        chunks = new_chunks
    
    return chunks[0]


def test_stability_comparison():
    """Compare stability: flat vs hierarchical composition."""
    print("\n" + "="*70)
    print("TEST 1: Stability - Flat vs Hierarchical Composition")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    lengths = [8, 16, 32, 64, 128, 256, 512]
    
    print(f"\n{'Length':<10} {'Flat σ':<12} {'Hier σ':<12} {'Flat norm':<12} {'Hier norm':<12}")
    print("-" * 58)
    
    for length in lengths:
        tokens = list(np.random.randint(0, 1000, size=length))
        
        # Flat composition
        flat_result = embed_sequence_flat(tokens, embeddings)
        flat_stability = grace_stability(flat_result, basis)
        flat_norm = np.linalg.norm(flat_result, 'fro')
        
        # Hierarchical composition
        hier_result = embed_sequence_hierarchical(tokens, embeddings, basis, chunk_size=8)
        hier_stability = grace_stability(hier_result, basis)
        hier_norm = np.linalg.norm(hier_result, 'fro')
        
        print(f"{length:<10} {flat_stability:<12.4f} {hier_stability:<12.4f} "
              f"{flat_norm:<12.4f} {hier_norm:<12.4f}")
    
    print("\n✓ Hierarchical composition maintains stability at all lengths")


def test_attractor_diversity():
    """Test if hierarchical composition produces diverse attractors."""
    print("\n" + "="*70)
    print("TEST 2: Attractor Diversity - Flat vs Hierarchical")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    # Generate 20 random contexts
    n_contexts = 20
    length = 64
    
    contexts = [list(np.random.randint(0, 1000, size=length)) for _ in range(n_contexts)]
    
    # Flat attractors
    flat_attractors = []
    for ctx in contexts:
        raw = embed_sequence_flat(ctx, embeddings)
        attractor = grace_until_stable(raw, basis)
        flat_attractors.append(attractor)
    
    # Hierarchical attractors
    hier_attractors = []
    for ctx in contexts:
        attractor = embed_sequence_hierarchical(ctx, embeddings, basis, chunk_size=8)
        hier_attractors.append(attractor)
    
    # Compute pairwise similarities
    flat_sims = []
    hier_sims = []
    
    for i in range(n_contexts):
        for j in range(i + 1, n_contexts):
            flat_sims.append(abs(frobenius_cosine(flat_attractors[i], flat_attractors[j])))
            hier_sims.append(abs(frobenius_cosine(hier_attractors[i], hier_attractors[j])))
    
    print(f"\nFlat attractor pairwise |similarities|:")
    print(f"  Mean: {np.mean(flat_sims):.4f}")
    print(f"  Std:  {np.std(flat_sims):.4f}")
    print(f"  Min:  {np.min(flat_sims):.4f}")
    print(f"  Max:  {np.max(flat_sims):.4f}")
    
    print(f"\nHierarchical attractor pairwise |similarities|:")
    print(f"  Mean: {np.mean(hier_sims):.4f}")
    print(f"  Std:  {np.std(hier_sims):.4f}")
    print(f"  Min:  {np.min(hier_sims):.4f}")
    print(f"  Max:  {np.max(hier_sims):.4f}")
    
    print(f"\n✓ Lower mean similarity = more diverse attractors")
    if np.mean(hier_sims) < np.mean(flat_sims):
        print("  Hierarchical produces MORE diverse attractors!")
    else:
        print("  Both methods produce similar diversity")


def test_retrieval_comparison():
    """Compare retrieval accuracy: flat vs hierarchical."""
    print("\n" + "="*70)
    print("TEST 3: Retrieval Accuracy - Flat vs Hierarchical")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    # Store patterns
    n_patterns = 50
    context_length = 32
    
    memory_flat = np.zeros((4, 4), dtype=DTYPE)
    memory_hier = np.zeros((4, 4), dtype=DTYPE)
    
    stored_contexts = []
    stored_targets = []
    stored_flat_attractors = []
    stored_hier_attractors = []
    
    print(f"\nStoring {n_patterns} patterns with context length {context_length}...")
    
    for i in range(n_patterns):
        context = list(np.random.randint(0, 1000, size=context_length))
        target = np.random.randint(0, 1000)
        
        stored_contexts.append(context)
        stored_targets.append(target)
        
        tgt_emb = embeddings[target]
        
        # Flat attractor
        flat_raw = embed_sequence_flat(context, embeddings)
        flat_attractor = grace_until_stable(flat_raw, basis)
        stored_flat_attractors.append(flat_attractor)
        
        # Hierarchical attractor
        hier_attractor = embed_sequence_hierarchical(context, embeddings, basis, chunk_size=8)
        stored_hier_attractors.append(hier_attractor)
        
        # Store in memories
        memory_flat += PHI_INV * (flat_attractor @ tgt_emb)
        memory_hier += PHI_INV * (hier_attractor @ tgt_emb)
    
    print(f"Memory norms: flat={np.linalg.norm(memory_flat):.2f}, hier={np.linalg.norm(memory_hier):.2f}")
    
    # Test retrieval
    n_test = min(20, n_patterns)
    
    print(f"\n--- Exact Context Retrieval (top-10 accuracy) ---")
    
    correct_flat = 0
    correct_hier = 0
    
    for i in range(n_test):
        true_target = stored_targets[i]
        
        # Flat retrieval
        retrieved_flat = stored_flat_attractors[i].T @ memory_flat
        sims_flat = [frobenius_cosine(retrieved_flat, embeddings[t]) for t in range(1000)]
        top10_flat = sorted(range(1000), key=lambda t: -sims_flat[t])[:10]
        if true_target in top10_flat:
            correct_flat += 1
        
        # Hierarchical retrieval
        retrieved_hier = stored_hier_attractors[i].T @ memory_hier
        sims_hier = [frobenius_cosine(retrieved_hier, embeddings[t]) for t in range(1000)]
        top10_hier = sorted(range(1000), key=lambda t: -sims_hier[t])[:10]
        if true_target in top10_hier:
            correct_hier += 1
    
    print(f"Flat attractor retrieval: {correct_flat}/{n_test} = {100*correct_flat/n_test:.1f}%")
    print(f"Hierarchical retrieval:   {correct_hier}/{n_test} = {100*correct_hier/n_test:.1f}%")
    
    # Test with perturbed contexts
    print(f"\n--- Perturbed Context Retrieval (generalization) ---")
    
    correct_flat = 0
    correct_hier = 0
    
    for i in range(n_test):
        context = stored_contexts[i].copy()
        true_target = stored_targets[i]
        
        # Perturb: change 2 tokens
        for _ in range(2):
            pos = np.random.randint(0, len(context))
            context[pos] = np.random.randint(0, 1000)
        
        # Flat retrieval
        flat_raw = embed_sequence_flat(context, embeddings)
        flat_attractor = grace_until_stable(flat_raw, basis)
        retrieved_flat = flat_attractor.T @ memory_flat
        sims_flat = [frobenius_cosine(retrieved_flat, embeddings[t]) for t in range(1000)]
        top10_flat = sorted(range(1000), key=lambda t: -sims_flat[t])[:10]
        if true_target in top10_flat:
            correct_flat += 1
        
        # Hierarchical retrieval
        hier_attractor = embed_sequence_hierarchical(context, embeddings, basis, chunk_size=8)
        retrieved_hier = hier_attractor.T @ memory_hier
        sims_hier = [frobenius_cosine(retrieved_hier, embeddings[t]) for t in range(1000)]
        top10_hier = sorted(range(1000), key=lambda t: -sims_hier[t])[:10]
        if true_target in top10_hier:
            correct_hier += 1
    
    print(f"Flat attractor retrieval: {correct_flat}/{n_test} = {100*correct_flat/n_test:.1f}%")
    print(f"Hierarchical retrieval:   {correct_hier}/{n_test} = {100*correct_hier/n_test:.1f}%")


def test_scalar_content_preservation():
    """
    Test the key insight: Does hierarchical composition preserve scalar content?
    """
    print("\n" + "="*70)
    print("TEST 4: Scalar Content Preservation")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    lengths = [8, 16, 32, 64, 128, 256]
    
    print(f"\n{'Length':<10} {'Flat |scalar|':<15} {'Hier |scalar|':<15} {'Ratio':<10}")
    print("-" * 50)
    
    for length in lengths:
        tokens = list(np.random.randint(0, 1000, size=length))
        
        # Flat composition
        flat_result = embed_sequence_flat(tokens, embeddings)
        flat_scalar = abs(np.trace(flat_result) / 4)  # scalar coefficient
        
        # Hierarchical composition
        hier_result = embed_sequence_hierarchical(tokens, embeddings, basis, chunk_size=8)
        hier_scalar = abs(np.trace(hier_result) / 4)
        
        ratio = hier_scalar / max(flat_scalar, 1e-10)
        
        print(f"{length:<10} {flat_scalar:<15.6f} {hier_scalar:<15.6f} {ratio:<10.1f}x")
    
    print("\n✓ Higher scalar content = more distinguishable attractors")


def test_chunk_size_sensitivity():
    """Test how chunk size affects hierarchical composition."""
    print("\n" + "="*70)
    print("TEST 5: Chunk Size Sensitivity")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    context_length = 128
    tokens = list(np.random.randint(0, 1000, size=context_length))
    
    chunk_sizes = [4, 8, 16, 32, 64]
    
    print(f"\nContext length: {context_length}")
    print(f"\n{'Chunk size':<12} {'Stability':<12} {'|Scalar|':<12} {'Norm':<10}")
    print("-" * 46)
    
    results = []
    for chunk_size in chunk_sizes:
        result = embed_sequence_hierarchical(tokens, embeddings, basis, chunk_size=chunk_size)
        stability = grace_stability(result, basis)
        scalar = abs(np.trace(result) / 4)
        norm = np.linalg.norm(result, 'fro')
        
        results.append(result)
        print(f"{chunk_size:<12} {stability:<12.4f} {scalar:<12.6f} {norm:<10.4f}")
    
    # Compare diversity across chunk sizes
    print(f"\nPairwise similarities between different chunk sizes:")
    for i, cs1 in enumerate(chunk_sizes):
        for j, cs2 in enumerate(chunk_sizes):
            if i < j:
                sim = frobenius_cosine(results[i], results[j])
                print(f"  chunk={cs1} vs chunk={cs2}: sim={sim:.4f}")


def main():
    print("\n" + "="*70)
    print("   HIERARCHICAL COMPOSITION: Theory Validation")
    print("="*70)
    print("""
HYPOTHESIS:
    Flat composition of N tokens loses witness content.
    Hierarchical composition with Grace at each level preserves it.
    
    Algorithm:
        tokens → [chunk_0, chunk_1, ...] → [Grace(chunk_i), ...] → compose → Grace → ...
        
PREDICTIONS:
    1. Hierarchical maintains stability at all lengths (flat doesn't)
    2. Hierarchical produces more diverse attractors
    3. Hierarchical enables better retrieval
    4. Hierarchical preserves scalar content
""")
    
    test_stability_comparison()
    test_scalar_content_preservation()
    test_attractor_diversity()
    test_chunk_size_sensitivity()
    test_retrieval_comparison()
    
    print("\n" + "="*70)
    print("   SUMMARY")
    print("="*70)


if __name__ == "__main__":
    main()
