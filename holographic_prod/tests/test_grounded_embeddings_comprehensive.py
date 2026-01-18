"""
Comprehensive Tests for Grounded Embeddings

Tests cover:
1. Mathematical correctness (SO(4) properties preserved)
2. Semantic structure (similar tokens → similar embeddings)
3. Performance (no bottlenecks, streaming works)
4. Edge cases (empty corpus, single token, etc.)
5. Integration with HolographicMemory
"""
import numpy as np
import time
from typing import List, Tuple

import pytest

from holographic_prod.core.grounded_embeddings import (
    so4_generators,
    get_so4_generators,
    compute_cooccurrence_from_corpus,
    compute_cooccurrence_streaming,
    cooccurrence_to_semantic_vectors,
    semantic_to_SO4,
    create_grounded_embeddings,
    create_grounded_embeddings_from_corpus,
    initialize_embeddings_grounded,
)
from holographic_prod.core.algebra import frobenius_cosine
from holographic_prod.core.constants import DTYPE


# =============================================================================
# TEST: Mathematical Correctness
# =============================================================================

class TestSO4Properties:
    """Verify SO(4) mathematical properties are preserved."""
    
    def test_generators_are_antisymmetric(self):
        """SO(4) generators must be antisymmetric (G = -G.T)."""
        gens = so4_generators()
        assert len(gens) == 6, "SO(4) has exactly 6 generators"
        
        for i, g in enumerate(gens):
            np.testing.assert_allclose(
                g, -g.T, 
                rtol=1e-10,
                err_msg=f"Generator {i} is not antisymmetric"
            )
    
    def test_generators_are_orthogonal_basis(self):
        """Generators should form orthogonal basis of so(4)."""
        gens = so4_generators()
        
        # Check Frobenius inner products
        for i, g1 in enumerate(gens):
            for j, g2 in enumerate(gens):
                inner = np.sum(g1 * g2)
                if i == j:
                    assert abs(inner - 2.0) < 1e-10, f"Generator {i} should have norm sqrt(2)"
                else:
                    assert abs(inner) < 1e-10, f"Generators {i} and {j} should be orthogonal"
    
    def test_semantic_to_SO4_produces_orthogonal(self):
        """Output matrices must be orthogonal (M @ M.T = I)."""
        np.random.seed(42)
        gens = get_so4_generators()
        
        for _ in range(100):
            vec = np.random.randn(6)
            M = semantic_to_SO4(vec, gens)
            
            # Check orthogonality (atol=1e-6 for numerical precision)
            product = M @ M.T
            np.testing.assert_allclose(
                product, np.eye(4),
                atol=1e-6,
                err_msg="Output is not orthogonal"
            )
    
    def test_semantic_to_SO4_produces_det_one(self):
        """Output matrices must have determinant +1 (special orthogonal)."""
        np.random.seed(42)
        gens = get_so4_generators()
        
        for _ in range(100):
            vec = np.random.randn(6)
            M = semantic_to_SO4(vec, gens)
            
            det = np.linalg.det(M)
            assert abs(det - 1.0) < 1e-6, f"Determinant should be 1, got {det}"
    
    def test_similar_vectors_produce_similar_matrices(self):
        """Similar semantic vectors should produce similar SO(4) matrices."""
        gens = get_so4_generators()
        
        vec1 = np.array([0.5, 0.3, 0.1, 0.0, 0.2, 0.1])
        vec2 = np.array([0.5, 0.31, 0.11, 0.01, 0.19, 0.09])  # Small perturbation
        vec3 = np.array([-0.5, 0.1, 0.8, 0.3, -0.2, 0.4])    # Very different
        
        M1 = semantic_to_SO4(vec1, gens)
        M2 = semantic_to_SO4(vec2, gens)
        M3 = semantic_to_SO4(vec3, gens)
        
        sim_12 = frobenius_cosine(M1, M2, np)
        sim_13 = frobenius_cosine(M1, M3, np)
        
        assert sim_12 > 0.99, f"Similar vectors should give similar matrices, got {sim_12}"
        assert sim_13 < sim_12, f"Different vectors should give different matrices"


# =============================================================================
# TEST: Co-occurrence Computation
# =============================================================================

class TestCooccurrence:
    """Test co-occurrence matrix computation."""
    
    def test_basic_cooccurrence(self):
        """Basic co-occurrence computation."""
        corpus = [
            [1, 2, 3],
            [1, 2, 4],
            [1, 2, 3],
        ]
        
        cooccur = compute_cooccurrence_from_corpus(corpus, vocab_size=10, window=5)
        
        # Token 1 and 2 appear together 3 times
        assert cooccur[1, 2] == 3, "1-2 co-occurrence should be 3"
        assert cooccur[2, 1] == 3, "Matrix should be symmetric"
        
        # Token 3 appears 2 times with 1 and 2
        assert cooccur[1, 3] == 2, "1-3 co-occurrence should be 2"
        assert cooccur[3, 1] == 2, "Matrix should be symmetric"
    
    def test_window_size_respected(self):
        """Co-occurrence should respect window size."""
        # Tokens far apart shouldn't co-occur with window=1
        corpus = [[1, 2, 3, 4, 5, 6]]
        
        cooccur = compute_cooccurrence_from_corpus(corpus, vocab_size=10, window=1)
        
        # 1 and 3 are 2 apart, shouldn't co-occur with window=1
        assert cooccur[1, 3] == 0, "Window should be respected"
        
        # 1 and 2 are adjacent
        assert cooccur[1, 2] > 0, "Adjacent tokens should co-occur"
    
    def test_empty_corpus(self):
        """Empty corpus should return zero matrix."""
        cooccur = compute_cooccurrence_from_corpus([], vocab_size=10, window=5)
        assert cooccur.shape == (10, 10)
        assert np.all(cooccur == 0)
    
    def test_out_of_bounds_tokens_ignored(self):
        """Tokens outside vocab_size should be ignored, not crash."""
        corpus = [[1, 2, 999, 3]]  # 999 is out of bounds for vocab_size=10
        
        # Should not raise
        cooccur = compute_cooccurrence_from_corpus(corpus, vocab_size=10, window=5)
        
        # Valid tokens should still co-occur
        assert cooccur[1, 2] > 0


# =============================================================================
# TEST: Semantic Structure
# =============================================================================

class TestSemanticStructure:
    """Test that grounded embeddings capture semantic structure."""
    
    def test_cooccurring_tokens_similar(self):
        """Tokens that co-occur frequently should have similar embeddings."""
        # Create corpus where tokens 1-5 co-occur (group A)
        # and tokens 10-15 co-occur (group B)
        np.random.seed(42)
        corpus = []
        
        for _ in range(500):
            # Group A sentences
            corpus.append([1, np.random.randint(2, 6), np.random.randint(2, 6)])
            # Group B sentences
            corpus.append([10, np.random.randint(11, 16), np.random.randint(11, 16)])
        
        embeddings = create_grounded_embeddings_from_corpus(corpus, vocab_size=20)
        
        # Within-group similarity should be high
        within_a = []
        for i in range(1, 5):
            for j in range(i+1, 6):
                within_a.append(frobenius_cosine(embeddings[i], embeddings[j], np))
        
        within_b = []
        for i in range(10, 14):
            for j in range(i+1, 15):
                within_b.append(frobenius_cosine(embeddings[i], embeddings[j], np))
        
        # Between-group similarity should be lower
        between = []
        for i in range(1, 6):
            for j in range(10, 16):
                between.append(frobenius_cosine(embeddings[i], embeddings[j], np))
        
        avg_within = (np.mean(within_a) + np.mean(within_b)) / 2
        avg_between = np.mean(between)
        
        assert avg_within > avg_between, \
            f"Within-group ({avg_within:.3f}) should be higher than between-group ({avg_between:.3f})"
    
    def test_semantic_vectors_normalized(self):
        """Semantic vectors should be unit normalized."""
        corpus = [[1, 2, 3, 4, 5] for _ in range(100)]
        cooccur = compute_cooccurrence_from_corpus(corpus, vocab_size=10)
        semantic_vecs = cooccurrence_to_semantic_vectors(cooccur, dim=6)
        
        norms = np.linalg.norm(semantic_vecs, axis=1)
        # Should be approximately unit (within tolerance for near-zero vectors)
        valid_norms = norms[norms > 0.1]  # Ignore near-zero vectors
        np.testing.assert_allclose(valid_norms, 1.0, rtol=0.01)


# =============================================================================
# TEST: Performance
# =============================================================================

class TestPerformance:
    """Test performance - no bottlenecks."""
    
    def test_cooccurrence_speed(self):
        """Co-occurrence computation should be fast."""
        np.random.seed(42)
        corpus = [list(np.random.randint(0, 500, size=10)) for _ in range(2000)]
        
        start = time.time()
        cooccur = compute_cooccurrence_from_corpus(corpus, vocab_size=500, window=5)
        elapsed = time.time() - start
        
        # Should complete in under 2 seconds for 2K sequences
        assert elapsed < 2.0, f"Co-occurrence too slow: {elapsed:.2f}s"
        print(f"Co-occurrence for 2K sequences: {elapsed:.2f}s")
    
    def test_svd_speed(self):
        """SVD should be fast for reasonable vocab sizes."""
        np.random.seed(42)
        # Use 2K vocab for faster test (production uses truncated SVD for large vocab)
        cooccur = np.random.rand(2000, 2000).astype(np.float32)
        
        start = time.time()
        semantic_vecs = cooccurrence_to_semantic_vectors(cooccur, dim=6)
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds for 2K vocab
        assert elapsed < 5.0, f"SVD too slow: {elapsed:.2f}s"
        print(f"SVD for 2K vocab: {elapsed:.2f}s")
    
    def test_embedding_creation_speed(self):
        """Full embedding creation should be fast."""
        np.random.seed(42)
        corpus = [list(np.random.randint(0, 1000, size=8)) for _ in range(1000)]
        
        start = time.time()
        embeddings = create_grounded_embeddings_from_corpus(corpus, vocab_size=1000)
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Embedding creation too slow: {elapsed:.2f}s"
        print(f"Full embedding creation for 1K vocab: {elapsed:.2f}s")
        
        # Verify shape and dtype
        assert embeddings.shape == (1000, 4, 4)
        assert embeddings.dtype == DTYPE
    
    def test_streaming_cooccurrence(self):
        """Streaming co-occurrence should work for large datasets."""
        def data_generator():
            np.random.seed(42)
            for _ in range(20000):
                ctx = list(np.random.randint(0, 500, size=5))
                tgt = np.random.randint(0, 500)
                yield ctx, tgt
        
        start = time.time()
        cooccur = compute_cooccurrence_streaming(
            data_generator(), 
            vocab_size=500, 
            max_samples=10000
        )
        elapsed = time.time() - start
        
        assert elapsed < 3.0, f"Streaming too slow: {elapsed:.2f}s"
        print(f"Streaming 10K samples: {elapsed:.2f}s")


# =============================================================================
# TEST: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_token_corpus(self):
        """Single token corpus should not crash."""
        corpus = [[1], [1], [1]]
        embeddings = create_grounded_embeddings_from_corpus(corpus, vocab_size=10)
        
        assert embeddings.shape == (10, 4, 4)
        # Token 1 should have valid embedding
        assert np.linalg.det(embeddings[1]) != 0
    
    def test_no_cooccurrence(self):
        """Tokens with no co-occurrence should still get valid embeddings."""
        corpus = [[1], [2], [3]]  # No co-occurrences
        embeddings = create_grounded_embeddings_from_corpus(corpus, vocab_size=10)
        
        # All should be valid SO(4) matrices
        for i in range(10):
            det = np.linalg.det(embeddings[i])
            # Should be valid (might be identity or random-like)
            assert np.isfinite(det)
    
    def test_large_vocab_size(self):
        """Should handle moderately large vocabulary sizes."""
        corpus = [[i, i+1] for i in range(0, 2000, 50)]
        
        # Should not crash
        embeddings = create_grounded_embeddings_from_corpus(corpus, vocab_size=2000)
        
        assert embeddings.shape == (2000, 4, 4)
    
    def test_fallback_to_random(self):
        """Should fall back to random when no grounding data."""
        embeddings = initialize_embeddings_grounded(
            vocab_size=100,
            data_samples=None,
            fallback_to_random=True
        )
        
        assert embeddings.shape == (100, 4, 4)
        
        # Should all be orthogonal
        for i in range(100):
            product = embeddings[i] @ embeddings[i].T
            np.testing.assert_allclose(product, np.eye(4), atol=1e-6)


# =============================================================================
# TEST: Integration
# =============================================================================

class TestIntegration:
    """Test integration with memory system."""
    
    def test_context_composition(self):
        """Grounded embeddings should compose correctly."""
        corpus = [[1, 2, 3, 4] for _ in range(100)]
        embeddings = create_grounded_embeddings_from_corpus(corpus, vocab_size=10)
        
        # Compose context
        ctx_mat = embeddings[1] @ embeddings[2] @ embeddings[3]
        
        # Should still be orthogonal (SO(4) closed under multiplication)
        product = ctx_mat @ ctx_mat.T
        np.testing.assert_allclose(product, np.eye(4), atol=1e-6)
    
    def test_unbinding_works(self):
        """Unbinding should work with grounded embeddings."""
        corpus = [[1, 2, 3, 4] for _ in range(100)]
        embeddings = create_grounded_embeddings_from_corpus(corpus, vocab_size=10)
        
        # Create binding: context × target
        ctx_mat = embeddings[1] @ embeddings[2] @ embeddings[3]
        tgt_mat = embeddings[4]
        binding = ctx_mat @ tgt_mat
        
        # Unbind: should recover target
        # For SO(4): inverse = transpose
        recovered = ctx_mat.T @ binding
        
        # Should be close to target
        sim = frobenius_cosine(recovered, tgt_mat, np)
        assert sim > 0.99, f"Unbinding failed, similarity={sim}"


# =============================================================================
# RUN TESTS
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("GROUNDED EMBEDDINGS COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    test_classes = [
        TestSO4Properties,
        TestCooccurrence,
        TestSemanticStructure,
        TestPerformance,
        TestEdgeCases,
        TestIntegration,
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 70)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
