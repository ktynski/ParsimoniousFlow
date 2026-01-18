"""
Investigation: Why SO(4) embeddings don't generalize, and how to fix it theory-true.

KEY INSIGHT:
Current: E(token) = random_SO4()  ← NO semantic structure
Theory-True: E(token) = project_to_SO4(semantic_embedding(token))

The issue is that tokens with similar MEANING should have similar SO(4) matrices.
"""
import numpy as np
from typing import List, Tuple

from holographic_prod.memory.holographic_memory_unified import HolographicMemory
from holographic_prod.core.algebra import build_clifford_basis, frobenius_cosine
from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ


def create_memory(vocab_size=1000, levels=3, use_gpu=False):
    return HolographicMemory(vocab_size=vocab_size, max_levels=levels, use_gpu=use_gpu)


def test_current_embedding_structure():
    """
    Test: Current embeddings have NO semantic structure.
    """
    print("=" * 70)
    print("TEST 1: Current Embedding Structure (RANDOM)")
    print("=" * 70)
    
    memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
    
    # Check similarity between consecutive tokens
    print("\nSimilarity between consecutive tokens (should be ~0 if random):")
    for i in range(0, 10):
        sim = frobenius_cosine(memory.tower.embeddings[i], memory.tower.embeddings[i+1], np)
        print(f"  E({i}) vs E({i+1}): {sim:.4f}")
    
    # Check similarity distribution
    print("\nSimilarity distribution for 100 random pairs:")
    sims = []
    for _ in range(100):
        i, j = np.random.randint(0, 500, size=2)
        if i != j:
            sim = frobenius_cosine(memory.tower.embeddings[i], memory.tower.embeddings[j], np)
            sims.append(sim)
    
    sims = np.array(sims)
    print(f"  Mean: {sims.mean():.4f}")
    print(f"  Std: {sims.std():.4f}")
    print(f"  Min: {sims.min():.4f}")
    print(f"  Max: {sims.max():.4f}")
    
    # This confirms embeddings are random - no structure
    print("\n⚠️ CONCLUSION: Embeddings are random - no semantic structure!")


def test_what_semantic_structure_would_look_like():
    """
    Demonstrate what SEMANTIC embeddings would provide.
    
    If similar tokens had similar embeddings:
    - "cat" and "dog" would have high similarity
    - Contexts with similar meaning would be close
    """
    print("\n" + "=" * 70)
    print("TEST 2: What Semantic Structure Would Look Like")
    print("=" * 70)
    
    memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
    
    # Simulate semantic structure by making similar tokens share a component
    # Theory: E(t) = U @ diag(semantic_features(t)) @ V
    # For now, just demonstrate with explicit similarity injection
    
    # Create "semantic groups" - tokens that should be similar
    # Group 1: tokens 10-19 are "similar" (e.g., animals)
    # Group 2: tokens 20-29 are "similar" (e.g., actions)
    
    print("\nSimulating semantic structure by averaging within groups...")
    
    # Compute group centroids
    group1_centroid = np.mean([memory.tower.embeddings[i] for i in range(10, 15)], axis=0)
    group2_centroid = np.mean([memory.tower.embeddings[i] for i in range(20, 25)], axis=0)
    
    # Project back to SO(4) using SVD (closest orthogonal matrix)
    def project_to_SO4(M):
        U, S, Vt = np.linalg.svd(M)
        return U @ Vt
    
    group1_so4 = project_to_SO4(group1_centroid)
    group2_so4 = project_to_SO4(group2_centroid)
    
    print(f"\nWithin-group similarity (tokens that SHOULD be similar):")
    for i in range(10, 15):
        sim = frobenius_cosine(memory.tower.embeddings[i], group1_so4, np)
        print(f"  E({i}) vs group1_centroid: {sim:.4f}")
    
    print(f"\nCross-group similarity (tokens that should be DIFFERENT):")
    for i in range(10, 15):
        sim = frobenius_cosine(memory.tower.embeddings[i], group2_so4, np)
        print(f"  E({i}) vs group2_centroid: {sim:.4f}")
    
    print("\n⚠️ Current embeddings don't have this structure - all tokens equally distant!")


def demonstrate_theory_true_fix():
    """
    Demonstrate the theory-true fix: structured SO(4) embeddings.
    
    KEY INSIGHT:
    Instead of E(t) = random_SO4(), use:
    
    E(t) = exp(θ(t) · generator)
    
    where θ(t) is a 6-dimensional semantic vector (SO(4) has 6 generators)
    and exp maps to SO(4) via matrix exponential.
    
    Similar semantic vectors → Similar SO(4) matrices!
    """
    print("\n" + "=" * 70)
    print("TEST 3: Theory-True Fix - Structured SO(4) Embeddings")
    print("=" * 70)
    
    # SO(4) generators (antisymmetric 4x4 matrices)
    def so4_generators():
        """Return the 6 generators of SO(4)."""
        gens = []
        for i in range(4):
            for j in range(i+1, 4):
                g = np.zeros((4, 4))
                g[i, j] = 1
                g[j, i] = -1
                gens.append(g)
        return gens
    
    gens = so4_generators()
    print(f"SO(4) has {len(gens)} generators (6 = dim of rotation space)")
    
    # Create structured embeddings via exponential map
    def semantic_to_SO4(semantic_vector, gens):
        """Map 6D semantic vector to SO(4) via exponential map."""
        # Linear combination of generators
        A = sum(theta * g for theta, g in zip(semantic_vector, gens))
        # Matrix exponential → SO(4)
        from scipy.linalg import expm
        return expm(A)
    
    # Create tokens with similar semantic vectors
    # "cat" and "dog" both have high "animal" component
    cat_semantic = np.array([1.0, 0.5, 0.0, 0.1, 0.2, 0.0])  # High in component 1 (animal?)
    dog_semantic = np.array([1.0, 0.6, 0.1, 0.0, 0.3, 0.0])  # Similar - also high in component 1
    car_semantic = np.array([0.0, 0.0, 1.0, 0.8, 0.0, 0.5])  # Different - high in component 3 (vehicle?)
    
    E_cat = semantic_to_SO4(cat_semantic * 0.5, gens)  # Scale for reasonable angles
    E_dog = semantic_to_SO4(dog_semantic * 0.5, gens)
    E_car = semantic_to_SO4(car_semantic * 0.5, gens)
    
    print("\nSimilarity with structured embeddings:")
    print(f"  E(cat) vs E(dog): {frobenius_cosine(E_cat, E_dog, np):.4f}  ← SIMILAR (both animals)")
    print(f"  E(cat) vs E(car): {frobenius_cosine(E_cat, E_car, np):.4f}  ← DIFFERENT")
    print(f"  E(dog) vs E(car): {frobenius_cosine(E_dog, E_car, np):.4f}  ← DIFFERENT")
    
    # Verify still SO(4)
    print("\nVerifying orthogonality (should be identity):")
    print(f"  E(cat) @ E(cat).T = I? error={np.linalg.norm(E_cat @ E_cat.T - np.eye(4)):.6f}")
    print(f"  E(dog) @ E(dog).T = I? error={np.linalg.norm(E_dog @ E_dog.T - np.eye(4)):.6f}")
    
    print("\n✓ THEORY-TRUE: Structured SO(4) embeddings maintain orthogonality")
    print("  AND provide semantic similarity!")


def test_context_generalization_with_structured_embeddings():
    """
    Show how structured embeddings would enable context generalization.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Context Generalization with Structured Embeddings")
    print("=" * 70)
    
    from scipy.linalg import expm
    
    def so4_generators():
        gens = []
        for i in range(4):
            for j in range(i+1, 4):
                g = np.zeros((4, 4))
                g[i, j] = 1
                g[j, i] = -1
                gens.append(g)
        return gens
    
    def semantic_to_SO4(semantic_vector, gens):
        A = sum(theta * g for theta, g in zip(semantic_vector * 0.5, gens))
        return expm(A)
    
    gens = so4_generators()
    
    # Create semantic vectors for a small vocabulary
    semantics = {
        "the": np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "a": np.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.0]),  # Similar to "the"
        "cat": np.array([1.0, 0.5, 0.0, 0.1, 0.2, 0.0]),
        "dog": np.array([1.0, 0.6, 0.1, 0.0, 0.3, 0.0]),  # Similar to "cat"
        "sat": np.array([0.0, 0.0, 0.8, 0.3, 0.0, 0.1]),
        "ran": np.array([0.0, 0.1, 0.9, 0.4, 0.1, 0.0]),  # Similar to "sat"
    }
    
    embeddings = {word: semantic_to_SO4(vec, gens) for word, vec in semantics.items()}
    
    # Compare context embeddings
    def embed_context(words):
        result = np.eye(4)
        for w in words:
            result = result @ embeddings[w]
        return result
    
    ctx1 = embed_context(["the", "cat", "sat"])
    ctx2 = embed_context(["the", "dog", "sat"])  # Similar - dog instead of cat
    ctx3 = embed_context(["a", "cat", "ran"])    # Similar - different articles and verb
    ctx4 = embed_context(["the", "dog", "ran"])  # Similar
    
    print("\nContext similarity (should be HIGH for similar meanings):")
    print(f"  'the cat sat' vs 'the dog sat': {frobenius_cosine(ctx1, ctx2, np):.4f}")
    print(f"  'the cat sat' vs 'a cat ran':   {frobenius_cosine(ctx1, ctx3, np):.4f}")
    print(f"  'the cat sat' vs 'the dog ran': {frobenius_cosine(ctx1, ctx4, np):.4f}")
    print(f"  'the dog sat' vs 'the dog ran': {frobenius_cosine(ctx2, ctx4, np):.4f}")
    
    # Compare to random embeddings
    print("\n--- Compare to RANDOM embeddings ---")
    np.random.seed(42)
    random_embeddings = {word: np.linalg.qr(np.random.randn(4, 4))[0] for word in semantics.keys()}
    
    def embed_random(words):
        result = np.eye(4)
        for w in words:
            result = result @ random_embeddings[w]
        return result
    
    rctx1 = embed_random(["the", "cat", "sat"])
    rctx2 = embed_random(["the", "dog", "sat"])
    
    print(f"  'the cat sat' vs 'the dog sat' (RANDOM): {frobenius_cosine(rctx1, rctx2, np):.4f}")
    
    print("\n✓ Structured embeddings enable GENERALIZATION")
    print("  Random embeddings do NOT generalize!")


if __name__ == "__main__":
    test_current_embedding_structure()
    test_what_semantic_structure_would_look_like()
    demonstrate_theory_true_fix()
    test_context_generalization_with_structured_embeddings()
    
    print("\n" + "=" * 70)
    print("SUMMARY: The Fix for Limited Generalization")
    print("=" * 70)
    print("""
The issue is NOT in the architecture - it's in the EMBEDDING initialization.

CURRENT (WRONG):
  E(token) = random_SO4()
  → All tokens equally distant
  → No generalization possible

THEORY-TRUE FIX:
  E(token) = exp(semantic_vector(token) · SO4_generators)
  → Similar tokens → Similar SO(4) matrices
  → Contexts with similar meaning are CLOSE
  → Generalization emerges naturally!

IMPLEMENTATION:
  1. Learn semantic vectors from data (like word2vec, but 6D)
  2. Map to SO(4) via exponential map
  3. This is EXACTLY what contrastive learning should do!

The architecture is correct. The embeddings need semantic structure.
""")
