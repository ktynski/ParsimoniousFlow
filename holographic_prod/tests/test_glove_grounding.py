"""
Test-Driven: GloVe-based Grounding for SO(4) Embeddings

THEORY:
    Human brains use PRE-EXISTING semantic structure from sensory grounding.
    Similarly, we can use PRE-TRAINED embeddings (GloVe) that already encode
    distributional semantics. This is:
    
    1. FAST: No co-occurrence computation (seconds vs minutes)
    2. HIGH-QUALITY: GloVe trained on billions of words
    3. THEORY-CONSISTENT: Still uses exponential map to SO(4)
    
    The key insight: GloVe vectors are just a different source of the
    semantic structure we'd extract from co-occurrence anyway.
"""

import numpy as np
import time
import pytest
from typing import Dict

# Import our grounding module
import sys
sys.path.insert(0, '/Users/fractlphoneroom1/Desktop/ParsimoniousFlow')
from holographic_prod.core.grounded_embeddings import (
    so4_generators,
    semantic_to_SO4,
    pretrained_to_SO4,
    DTYPE
)


class TestGloVeGrounding:
    """Test suite for GloVe-based SO(4) grounding."""
    
    def test_so4_generators_valid(self):
        """Generators must be antisymmetric 4x4 matrices."""
        gens = so4_generators()
        
        assert len(gens) == 6, "SO(4) has exactly 6 generators"
        
        for i, g in enumerate(gens):
            assert g.shape == (4, 4), f"Generator {i} not 4x4"
            # Antisymmetric: G^T = -G
            assert np.allclose(g, -g.T), f"Generator {i} not antisymmetric"
    
    def test_semantic_to_SO4_produces_valid_rotation(self):
        """Output of semantic_to_SO4 must be valid SO(4)."""
        # Test several random 6D vectors
        np.random.seed(42)
        for _ in range(100):
            vec = np.random.randn(6).astype(np.float32)
            R = semantic_to_SO4(vec)
            
            # Check shape
            assert R.shape == (4, 4), "Output not 4x4"
            
            # Check orthogonality: R^T R = I
            should_be_identity = R.T @ R
            assert np.allclose(should_be_identity, np.eye(4), atol=1e-5), \
                "R^T R != I (not orthogonal)"
            
            # Check determinant = +1 (not reflection)
            det = np.linalg.det(R)
            assert np.isclose(det, 1.0, atol=1e-5), f"det(R) = {det} != 1"
    
    def test_similar_vectors_similar_SO4(self):
        """Similar semantic vectors should produce similar SO(4) matrices."""
        base_vec = np.array([0.5, -0.3, 0.1, 0.4, -0.2, 0.6], dtype=np.float32)
        
        # Small perturbation
        small_perturb = base_vec + np.random.randn(6) * 0.01
        
        # Large perturbation  
        large_perturb = base_vec + np.random.randn(6) * 1.0
        
        R_base = semantic_to_SO4(base_vec)
        R_small = semantic_to_SO4(small_perturb)
        R_large = semantic_to_SO4(large_perturb)
        
        # Frobenius distance
        dist_small = np.linalg.norm(R_base - R_small, 'fro')
        dist_large = np.linalg.norm(R_base - R_large, 'fro')
        
        print(f"Small perturbation distance: {dist_small:.6f}")
        print(f"Large perturbation distance: {dist_large:.6f}")
        
        # Small perturbation should give smaller distance
        assert dist_small < dist_large, \
            f"Similar vectors should give similar SO(4): {dist_small} >= {dist_large}"
    
    def test_pretrained_to_SO4_speed(self):
        """PCA + exponential map should be fast."""
        vocab_size = 10_000
        embed_dim = 50
        
        # Simulate GloVe embeddings
        fake_glove = np.random.randn(vocab_size, embed_dim).astype(np.float32)
        
        start = time.time()
        so4_embeddings = pretrained_to_SO4(fake_glove)
        elapsed = time.time() - start
        
        print(f"\nPretrained → SO(4) for {vocab_size:,} vocab: {elapsed:.2f}s")
        
        assert elapsed < 10.0, f"Too slow: {elapsed:.2f}s > 10s"
        assert so4_embeddings.shape == (vocab_size, 4, 4)
        
        # Check first few are valid SO(4)
        for i in range(min(10, vocab_size)):
            R = so4_embeddings[i]
            should_be_identity = R.T @ R
            assert np.allclose(should_be_identity, np.eye(4), atol=1e-4), \
                f"Embedding {i} not orthogonal"
    
    def test_pretrained_to_SO4_preserves_similarity(self):
        """Semantic similarity in GloVe space should transfer to SO(4) space."""
        # Create embeddings where some are deliberately similar
        vocab_size = 100
        embed_dim = 50
        
        # Base random embeddings
        embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32)
        
        # Make indices 0, 1, 2 very similar (like "cat", "kitten", "feline")
        embeddings[1] = embeddings[0] + np.random.randn(embed_dim) * 0.1
        embeddings[2] = embeddings[0] + np.random.randn(embed_dim) * 0.1
        
        # Make indices 50, 51, 52 very similar but different cluster
        embeddings[50] = np.random.randn(embed_dim) * 2  # Different base
        embeddings[51] = embeddings[50] + np.random.randn(embed_dim) * 0.1
        embeddings[52] = embeddings[50] + np.random.randn(embed_dim) * 0.1
        
        # Convert to SO(4)
        so4 = pretrained_to_SO4(embeddings)
        
        # Check intra-cluster similarity > inter-cluster similarity
        def so4_sim(R1, R2):
            """Frobenius inner product normalized."""
            return np.sum(R1 * R2) / 16  # 4x4 = 16 elements
        
        # Cat cluster (0, 1, 2)
        cat_sim_01 = so4_sim(so4[0], so4[1])
        cat_sim_02 = so4_sim(so4[0], so4[2])
        cat_sim_12 = so4_sim(so4[1], so4[2])
        
        # Cross-cluster
        cross_sim_0_50 = so4_sim(so4[0], so4[50])
        
        print(f"\nCat cluster similarities: {cat_sim_01:.4f}, {cat_sim_02:.4f}, {cat_sim_12:.4f}")
        print(f"Cross-cluster similarity: {cross_sim_0_50:.4f}")
        
        # Intra should be higher than inter
        avg_intra = (cat_sim_01 + cat_sim_02 + cat_sim_12) / 3
        assert avg_intra > cross_sim_0_50, \
            f"Similarity not preserved: intra={avg_intra:.4f} <= inter={cross_sim_0_50:.4f}"


class TestGloVeIntegration:
    """Integration tests requiring actual GloVe loading."""
    
    def test_manual_glove_simulation(self):
        """Test the full pipeline with simulated GloVe-like embeddings."""
        # Simulate what GloVe would give us
        # In reality these would be loaded from the actual GloVe file
        vocab = {
            'cat': 0, 'kitten': 1, 'dog': 2, 'puppy': 3,
            'car': 4, 'vehicle': 5, 'truck': 6,
            'the': 7, 'a': 8, 'is': 9,
        }
        vocab_size = len(vocab)
        
        # Create embeddings with semantic structure
        # Animals are similar, vehicles are similar
        embeddings = np.random.randn(vocab_size, 50).astype(np.float32)
        
        # Animal cluster
        animal_base = np.random.randn(50).astype(np.float32)
        embeddings[0] = animal_base  # cat
        embeddings[1] = animal_base + np.random.randn(50) * 0.2  # kitten
        embeddings[2] = animal_base + np.random.randn(50) * 0.3  # dog  
        embeddings[3] = animal_base + np.random.randn(50) * 0.3  # puppy
        
        # Vehicle cluster
        vehicle_base = np.random.randn(50).astype(np.float32) * 2  # Different scale
        embeddings[4] = vehicle_base  # car
        embeddings[5] = vehicle_base + np.random.randn(50) * 0.2  # vehicle
        embeddings[6] = vehicle_base + np.random.randn(50) * 0.3  # truck
        
        # Convert to SO(4)
        so4 = pretrained_to_SO4(embeddings)
        
        # Verify structure preserved
        def so4_distance(R1, R2):
            return np.linalg.norm(R1 - R2, 'fro')
        
        # cat-kitten should be closer than cat-car
        d_cat_kitten = so4_distance(so4[0], so4[1])
        d_cat_car = so4_distance(so4[0], so4[4])
        
        print(f"\nDistance cat-kitten: {d_cat_kitten:.4f}")
        print(f"Distance cat-car: {d_cat_car:.4f}")
        
        assert d_cat_kitten < d_cat_car, \
            f"Semantic distance not preserved: cat-kitten ({d_cat_kitten}) >= cat-car ({d_cat_car})"
        
        # dog-puppy should be closer than dog-truck
        d_dog_puppy = so4_distance(so4[2], so4[3])
        d_dog_truck = so4_distance(so4[2], so4[6])
        
        print(f"Distance dog-puppy: {d_dog_puppy:.4f}")
        print(f"Distance dog-truck: {d_dog_truck:.4f}")
        
        assert d_dog_puppy < d_dog_truck, \
            f"Semantic distance not preserved: dog-puppy ({d_dog_puppy}) >= dog-truck ({d_dog_truck})"


class TestVectorizedImplementation:
    """Test vectorized (fast) implementation."""
    
    def test_vectorized_semantic_to_SO4(self):
        """Vectorized version should match loop version."""
        from scipy.linalg import expm
        
        gens = so4_generators()
        stacked_gens = np.stack(gens)  # [6, 4, 4]
        
        vocab_size = 100
        semantic_vecs = np.random.randn(vocab_size, 6).astype(np.float32)
        semantic_vecs = semantic_vecs / (np.linalg.norm(semantic_vecs, axis=1, keepdims=True) + 1e-10)
        scale = 0.3
        
        # Loop version (reference)
        loop_result = np.zeros((vocab_size, 4, 4), dtype=DTYPE)
        for i in range(vocab_size):
            loop_result[i] = semantic_to_SO4(semantic_vecs[i], gens, scale)
        
        # Vectorized version
        # A = sum(theta * g) -> einsum: semantic_vecs @ stacked_gens (reshaped)
        scaled = semantic_vecs * scale  # [vocab, 6]
        
        # Each A[i] = sum_j scaled[i,j] * gens[j]
        A_batch = np.einsum('vi,ijk->vjk', scaled, stacked_gens)  # [vocab, 4, 4]
        
        # Vectorized expm is tricky - scipy.linalg.expm doesn't vectorize
        # But we can parallelize with numpy for small angles
        vec_result = np.zeros((vocab_size, 4, 4), dtype=DTYPE)
        for i in range(vocab_size):
            vec_result[i] = expm(A_batch[i]).astype(DTYPE)
        
        # Check they match
        assert np.allclose(loop_result, vec_result, atol=1e-5), \
            "Vectorized doesn't match loop version"
        
        print("\n✓ Vectorized matches loop version")


if __name__ == '__main__':
    # Run tests
    import sys
    
    print("=" * 60)
    print("GloVe Grounding Tests")
    print("=" * 60)
    
    test = TestGloVeGrounding()
    
    print("\n1. Testing SO(4) generators validity...")
    test.test_so4_generators_valid()
    print("   ✓ Passed")
    
    print("\n2. Testing semantic_to_SO4 produces valid rotation...")
    test.test_semantic_to_SO4_produces_valid_rotation()
    print("   ✓ Passed")
    
    print("\n3. Testing similar vectors → similar SO(4)...")
    test.test_similar_vectors_similar_SO4()
    print("   ✓ Passed")
    
    print("\n4. Testing pretrained_to_SO4 speed...")
    test.test_pretrained_to_SO4_speed()
    print("   ✓ Passed")
    
    print("\n5. Testing similarity preservation...")
    test.test_pretrained_to_SO4_preserves_similarity()
    print("   ✓ Passed")
    
    print("\n6. Testing GloVe simulation...")
    test2 = TestGloVeIntegration()
    test2.test_manual_glove_simulation()
    print("   ✓ Passed")
    
    print("\n7. Testing vectorized implementation...")
    test3 = TestVectorizedImplementation()
    test3.test_vectorized_semantic_to_SO4()
    print("   ✓ Passed")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
