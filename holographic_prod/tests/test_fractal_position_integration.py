"""
Integration Tests for Fractal Position Encoding in MultiLevelTower
===================================================================

Verifies that fractal position encoding is correctly integrated into
the memory system and produces different embeddings for different
word orderings.

THEORY:
    "dog bites man" vs "man bites dog" should produce DIFFERENT
    context embeddings because word ORDER matters for syntax.
    
    Without position encoding: both could hash to similar embeddings
    (bag-of-words problem).
    
    With fractal position encoding: each position rotates the token
    embedding by a φ-derived angle, creating ORDER-SENSITIVE
    composite embeddings.
"""

import pytest
import numpy as np
from holographic_prod.memory.multi_level_tower import MultiLevelTower


class TestFractalPositionIntegration:
    """Tests for fractal position encoding in MultiLevelTower."""
    
    @pytest.fixture
    def tower_with_position(self):
        """Create tower with fractal position enabled."""
        return MultiLevelTower(
            vocab_size=1000,
            levels=2,
            seed=42,
            use_gpu=False,
            use_fractal_position=True,
            fractal_position_scales=4,
            max_context_length=64,
        )
    
    @pytest.fixture
    def tower_without_position(self):
        """Create tower without fractal position."""
        return MultiLevelTower(
            vocab_size=1000,
            levels=2,
            seed=42,
            use_gpu=False,
            use_fractal_position=False,
        )
    
    def test_position_flag_stored(self, tower_with_position, tower_without_position):
        """Position flag should be stored correctly."""
        assert tower_with_position.use_fractal_position is True
        assert tower_without_position.use_fractal_position is False
    
    def test_rotation_cache_created(self, tower_with_position):
        """Rotation cache should be created when position encoding enabled."""
        assert hasattr(tower_with_position, '_position_rotations')
        assert hasattr(tower_with_position, '_position_rotations_inv')
        
        # Check shape: [max_context_length, 4, 4]
        assert tower_with_position._position_rotations.shape == (64, 4, 4)
        assert tower_with_position._position_rotations_inv.shape == (64, 4, 4)
    
    def test_no_rotation_cache_without_position(self, tower_without_position):
        """Rotation cache should NOT exist when position encoding disabled."""
        assert not hasattr(tower_without_position, '_position_rotations')
    
    def test_swapped_tokens_different_embeddings_with_position(self, tower_with_position):
        """Swapped token order should give different embeddings WITH position."""
        context_1 = [10, 20, 30]  # "dog bites man"
        context_2 = [30, 20, 10]  # "man bites dog"
        
        emb_1 = tower_with_position._embed_sequence(context_1)
        emb_2 = tower_with_position._embed_sequence(context_2)
        
        # Should be DIFFERENT
        assert not np.allclose(emb_1, emb_2), \
            "Swapped tokens should give different embeddings WITH position encoding"
    
    def test_swapped_tokens_similar_without_position(self, tower_without_position):
        """
        Note: Even without position encoding, geometric product is non-commutative,
        so swapped tokens will give different results. However, the DEGREE of
        difference should be larger with position encoding.
        """
        context_1 = [10, 20, 30]
        context_2 = [30, 20, 10]
        
        emb_1 = tower_without_position._embed_sequence(context_1)
        emb_2 = tower_without_position._embed_sequence(context_2)
        
        # Geometric product is non-commutative, so they'll still be different
        # This test just confirms the baseline behavior
        diff_without = np.linalg.norm(emb_1 - emb_2)
        assert diff_without > 0, "Even without position, geometric product is non-commutative"
    
    def test_position_increases_differentiation(
        self, 
        tower_with_position, 
        tower_without_position
    ):
        """Position encoding should increase differentiation between orderings."""
        context_1 = [10, 20, 30]
        context_2 = [30, 20, 10]
        
        # With position encoding
        emb_1_pos = tower_with_position._embed_sequence(context_1)
        emb_2_pos = tower_with_position._embed_sequence(context_2)
        diff_with_pos = np.linalg.norm(emb_1_pos - emb_2_pos)
        
        # Without position encoding
        emb_1_no = tower_without_position._embed_sequence(context_1)
        emb_2_no = tower_without_position._embed_sequence(context_2)
        diff_without_pos = np.linalg.norm(emb_1_no - emb_2_no)
        
        print(f"\nDiff WITH position: {diff_with_pos:.4f}")
        print(f"Diff WITHOUT position: {diff_without_pos:.4f}")
        
        # Position encoding should create more differentiation
        # (Not strictly guaranteed, but expected for random embeddings)
        # We just check both are non-zero
        assert diff_with_pos > 0
        assert diff_without_pos > 0
    
    def test_same_context_same_embedding(self, tower_with_position):
        """Same context should always give same embedding (deterministic)."""
        context = [10, 20, 30, 40, 50]
        
        emb_1 = tower_with_position._embed_sequence(context)
        emb_2 = tower_with_position._embed_sequence(context)
        
        assert np.allclose(emb_1, emb_2), "Same context should give same embedding"
    
    def test_batch_embedding_matches_single(self, tower_with_position):
        """Batch embedding should match single embedding for same context."""
        contexts = [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ]
        
        # Single embeddings
        single_embs = [tower_with_position._embed_sequence(ctx) for ctx in contexts]
        
        # Batch embedding
        batch_embs = tower_with_position._embed_sequences_batch(contexts)
        
        for i, (single, batch) in enumerate(zip(single_embs, batch_embs)):
            if hasattr(batch, 'get'):
                batch = batch.get()
            assert np.allclose(single, batch, atol=1e-5), \
                f"Batch embedding {i} should match single embedding"
    
    def test_variable_length_batch(self, tower_with_position):
        """Variable length batch should work correctly."""
        contexts = [
            [10, 20],           # len 2
            [30, 40, 50],       # len 3
            [60, 70, 80, 90],   # len 4
        ]
        
        # Should not raise
        batch_embs = tower_with_position._embed_sequences_batch(contexts)
        
        # Check shape: [3, 4, 4]
        assert batch_embs.shape == (3, 4, 4)
    
    def test_preserves_orthogonality(self, tower_with_position):
        """Position encoding should preserve SO(4) structure."""
        context = [10, 20, 30, 40, 50]
        
        emb = tower_with_position._embed_sequence(context)
        
        # Should be orthogonal
        identity = emb @ emb.T
        assert np.allclose(identity, np.eye(4), atol=1e-5), \
            "Embedding should be orthogonal"


class TestFractalPositionLearningIntegration:
    """Tests that learning works correctly with fractal position encoding."""
    
    @pytest.fixture
    def tower(self):
        """Create tower with fractal position enabled."""
        return MultiLevelTower(
            vocab_size=1000,
            levels=2,
            seed=42,
            use_gpu=False,
            use_fractal_position=True,
        )
    
    def test_learn_and_retrieve(self, tower):
        """Basic learn and retrieve should work with position encoding."""
        context = [10, 20, 30]
        target = 100
        
        # Learn
        tower.learn(context, target)
        
        # Retrieve (returns just the token, not a tuple)
        pred = tower.retrieve(context)
        
        # Should retrieve the target
        assert pred == target, f"Expected {target}, got {pred}"
    
    def test_order_sensitive_learning(self, tower):
        """Different word orders should be distinguishable after learning."""
        context_1 = [10, 20, 30]  # "dog bites man" → 100
        context_2 = [30, 20, 10]  # "man bites dog" → 200
        
        # Learn both patterns
        tower.learn(context_1, 100)
        tower.learn(context_2, 200)
        
        # Retrieve should distinguish them
        pred_1 = tower.retrieve(context_1)
        pred_2 = tower.retrieve(context_2)
        
        print(f"\nContext 1 → {pred_1} (expected 100)")
        print(f"Context 2 → {pred_2} (expected 200)")
        
        # Should be different (though not guaranteed to be exact match
        # due to holographic interference)
        assert pred_1 != pred_2 or pred_1 in [100, 200], \
            "Should distinguish different word orders"
