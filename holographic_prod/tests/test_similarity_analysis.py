"""
Deep dive into similarity analysis to understand why semantic retrieval rate is low.

Key questions:
1. What similarities do we get between queries and prototypes?
2. How does pattern completion affect similarity?
3. Is the threshold (φ⁻²) appropriate?
"""
import numpy as np
from typing import List

from holographic_prod.memory.holographic_memory_unified import HolographicMemory
from holographic_prod.dreaming.semantic_memory import SemanticMemory, SemanticPrototype
from holographic_prod.dreaming.structures import EpisodicEntry
from holographic_prod.dreaming.dreaming_system import DreamingSystem
from holographic_prod.core.algebra import build_clifford_basis, frobenius_cosine
from holographic_prod.dreaming.pattern_completion import pattern_complete
from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ, PHI_INV_CUBE


def create_memory(vocab_size=1000, levels=3, use_gpu=False):
    return HolographicMemory(vocab_size=vocab_size, max_levels=levels, use_gpu=use_gpu)


def create_episodic_entry(memory, ctx: List[int], target: int) -> EpisodicEntry:
    ctx_mat = memory.tower._embed_sequence(ctx)
    return EpisodicEntry(context_matrix=ctx_mat, target_token=target)


def test_similarity_between_related_contexts():
    """
    Test: What's the cosine similarity between RELATED contexts?
    
    Theory: Similar contexts should have similar SO(4) embeddings.
    """
    print("=" * 70)
    print("TEST 1: Similarity between related contexts")
    print("=" * 70)
    
    memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
    basis = build_clifford_basis(np)
    
    # Base context
    base = [1, 2, 3]
    base_mat = memory.tower._embed_sequence(base)
    
    # Variations
    variations = [
        ([1, 2, 3], "exact"),
        ([1, 2, 4], "last different"),
        ([1, 5, 3], "middle different"),
        ([10, 2, 3], "first different"),
        ([10, 20, 30], "all different"),
        ([3, 2, 1], "reversed"),
    ]
    
    print(f"Base context: {base}")
    print(f"Base embedding norm: {np.linalg.norm(base_mat):.4f}")
    print()
    
    for var, desc in variations:
        var_mat = memory.tower._embed_sequence(var)
        sim = frobenius_cosine(base_mat, var_mat, np)
        print(f"  {var} ({desc}): sim={sim:.4f}")
    
    # Also test pattern completion
    print("\n  With pattern completion (3 steps):")
    for var, desc in variations:
        var_mat = memory.tower._embed_sequence(var)
        completed, _ = pattern_complete(var_mat, basis, np, max_steps=3)
        sim_before = frobenius_cosine(base_mat, var_mat, np)
        sim_after = frobenius_cosine(base_mat, completed, np)
        print(f"  {var}: before={sim_before:.4f}, after={sim_after:.4f}, delta={sim_after-sim_before:+.4f}")


def test_prototype_similarity_distribution():
    """
    Test: After consolidation, what's the similarity distribution?
    """
    print("\n" + "=" * 70)
    print("TEST 2: Prototype similarity distribution after consolidation")
    print("=" * 70)
    
    memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
    basis = build_clifford_basis(np)
    dreaming = DreamingSystem(basis, xp=np)
    
    # Create patterns that SHOULD cluster together
    # Group A: All start with [1, 2, ...]
    # Group B: All start with [10, 20, ...]
    patterns_a = [[1, 2, i, 100] for i in range(3, 13)]  # 10 patterns
    patterns_b = [[10, 20, i, 200] for i in range(30, 40)]  # 10 patterns
    
    episodes = []
    for ctx in patterns_a + patterns_b:
        memory.learn(ctx[:-1], ctx[-1])
        entry = create_episodic_entry(memory, ctx[:-1], ctx[-1])
        episodes.append(entry)
    
    print(f"Created {len(episodes)} episodes")
    print(f"  Group A (target 100): {len(patterns_a)} patterns")
    print(f"  Group B (target 200): {len(patterns_b)} patterns")
    
    # Consolidate
    dreaming.sleep(episodes, verbose=False)
    
    stats = dreaming.semantic_memory.stats()
    print(f"\nAfter consolidation: {stats['total_prototypes']} prototypes")
    
    # Test queries
    test_queries = [
        [1, 2, 15],    # Should match Group A
        [10, 20, 45],  # Should match Group B  
        [1, 2, 3],     # Exact match from Group A
        [10, 20, 30],  # Exact match from Group B
        [50, 60, 70],  # No match expected
    ]
    
    print("\nQuery similarities:")
    print(f"  Threshold: φ⁻² = {PHI_INV_SQ:.4f}")
    print()
    
    for query in test_queries:
        query_mat = memory.tower._embed_sequence(query)
        results = dreaming.semantic_memory.retrieve(
            query_mat, top_k=3, use_pattern_completion=True, completion_steps=3
        )
        
        if results:
            print(f"  Query {query}:")
            for proto, sim in results[:3]:
                status = "✓" if sim > PHI_INV_SQ else "✗"
                print(f"    {status} sim={sim:.4f}, target={proto.mode_target()}, support={proto.support}")
        else:
            print(f"  Query {query}: NO PROTOTYPES")


def test_threshold_sensitivity():
    """
    Test: What % of queries would pass at different thresholds?
    """
    print("\n" + "=" * 70)
    print("TEST 3: Threshold sensitivity analysis")
    print("=" * 70)
    
    memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
    basis = build_clifford_basis(np)
    dreaming = DreamingSystem(basis, xp=np)
    
    # Create enough patterns to form prototypes
    np.random.seed(42)
    episodes = []
    for _ in range(50):
        # Create clustered patterns (same first 2 tokens)
        prefix = np.random.randint(1, 100, size=2)
        for i in range(5):  # 5 patterns per prefix
            ctx = list(prefix) + [np.random.randint(1, 500)]
            target = int(prefix[0])  # Target is first token (creates clusters)
            memory.learn(ctx, target)
            entry = create_episodic_entry(memory, ctx, target)
            episodes.append(entry)
    
    print(f"Created {len(episodes)} episodes (50 groups × 5)")
    
    # Consolidate
    dreaming.sleep(episodes, verbose=False)
    
    stats = dreaming.semantic_memory.stats()
    print(f"After consolidation: {stats['total_prototypes']} prototypes")
    
    # Generate random queries and check their similarity distribution
    np.random.seed(123)
    similarities = []
    for _ in range(200):
        query = list(np.random.randint(1, 500, size=3))
        query_mat = memory.tower._embed_sequence(query)
        results = dreaming.semantic_memory.retrieve(
            query_mat, top_k=1, use_pattern_completion=True, completion_steps=3
        )
        if results:
            _, sim = results[0]
            similarities.append(sim)
    
    if similarities:
        sims = np.array(similarities)
        print(f"\nSimilarity distribution for 200 random queries:")
        print(f"  Min: {sims.min():.4f}")
        print(f"  Max: {sims.max():.4f}")
        print(f"  Mean: {sims.mean():.4f}")
        print(f"  Median: {np.median(sims):.4f}")
        
        print(f"\nPass rate at different thresholds:")
        for threshold in [0.2, 0.3, PHI_INV_SQ, 0.4, 0.5, PHI_INV, 0.7]:
            pass_rate = (sims > threshold).mean() * 100
            marker = " ← φ⁻²" if abs(threshold - PHI_INV_SQ) < 0.01 else ""
            marker = " ← φ⁻¹" if abs(threshold - PHI_INV) < 0.01 else marker
            print(f"  {threshold:.3f}: {pass_rate:.1f}%{marker}")


def test_embedding_orthogonality():
    """
    Test: Are embeddings truly SO(4)?
    """
    print("\n" + "=" * 70)
    print("TEST 4: Embedding orthogonality check")
    print("=" * 70)
    
    memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
    
    # Check random embeddings
    orthogonality_errors = []
    for i in range(100):
        emb = memory.tower.embeddings[i]
        # For SO(4): E @ E.T should be identity
        product = emb @ emb.T
        identity = np.eye(4)
        error = np.linalg.norm(product - identity, 'fro')
        orthogonality_errors.append(error)
    
    errors = np.array(orthogonality_errors)
    print(f"E @ E.T - I (should be 0):")
    print(f"  Mean error: {errors.mean():.6f}")
    print(f"  Max error: {errors.max():.6f}")
    
    # Check context composition
    print("\nContext composition (A @ B should be SO(4)):")
    for _ in range(10):
        ctx = list(np.random.randint(1, 500, size=5))
        ctx_mat = memory.tower._embed_sequence(ctx)
        product = ctx_mat @ ctx_mat.T
        error = np.linalg.norm(product - np.eye(4), 'fro')
        print(f"  Context {ctx}: error = {error:.6f}")


if __name__ == "__main__":
    test_similarity_between_related_contexts()
    test_prototype_similarity_distribution()
    test_threshold_sensitivity()
    test_embedding_orthogonality()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
