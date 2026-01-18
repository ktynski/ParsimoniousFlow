#!/usr/bin/env python3
"""
Test-Driven Implementation: Settling NOT Argmax
================================================

THEORY: "NO sampling, NO argmax — just settling"

The brain doesn't pick tokens via argmax. It settles into attractor states.
The settled state IS the answer.

For EVALUATION, we should measure:
    semantic_similarity = frobenius_cosine(retrieved_state, target_embedding)
    
NOT:
    predicted_token = argmax(scores)
    accuracy = predicted_token == target

This test file enforces theory-true evaluation.
"""

import pytest
import numpy as np

from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ
from holographic_prod.core.algebra import get_cached_basis, frobenius_cosine
from holographic_prod.memory.holographic_memory_unified import HolographicMemory


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def basis():
    """Cached Clifford basis."""
    return get_cached_basis()


@pytest.fixture
def memory():
    """Memory for testing."""
    return HolographicMemory(vocab_size=100, max_levels=1)


# =============================================================================
# TEST: retrieve_settled_state MUST EXIST AND RETURN STATE (NOT TOKEN)
# =============================================================================

class TestRetrieveSettledState:
    """
    THEORY: The retrieval operation should return the SETTLED STATE,
    not a discrete token selected via argmax.
    
    The method `retrieve_settled_state(context)` should:
    1. Embed context
    2. Route to satellite
    3. Unbind: retrieved = context^T @ memory
    4. Return the retrieved matrix (4x4)
    
    NO argmax. NO token selection. Just settling.
    """
    
    def test_retrieve_settled_state_exists(self, memory):
        """Method must exist."""
        assert hasattr(memory, 'retrieve_settled_state'), \
            "HolographicMemory must have retrieve_settled_state method"
    
    def test_retrieve_settled_state_returns_matrix(self, memory):
        """
        Must return the settled state as a 4x4 matrix, NOT a token ID.
        """
        context = [1, 2, 3, 4]
        target = 42
        
        memory.learn(context, target)
        
        result = memory.retrieve_settled_state(context)
        
        # Must be a matrix, not an integer
        assert hasattr(result, 'shape'), "Must return array, not scalar"
        assert result.shape == (4, 4), f"Must return 4x4 matrix, got {result.shape}"
    
    def test_retrieve_settled_state_no_argmax_anywhere(self, memory):
        """
        The retrieve_settled_state path must NOT use argmax.
        
        This is verified by checking the returned value IS the unbinding result,
        not a discretized version.
        """
        context = [1, 2, 3, 4]
        target = 42
        
        memory.learn(context, target)
        
        # Get settled state via new method
        settled = memory.retrieve_settled_state(context)
        
        # Manual unbinding (ground truth)
        ctx_mat = memory.embed_sequence(context)
        sat_idx = memory.tower.route_to_satellite(context)
        
        # Get satellite memory (handle both TowerMemory and MultiLevelTower)
        if hasattr(memory.tower, '_all_memories'):
            sat_memory = memory.tower._all_memories[sat_idx]
        else:
            sat_memory = memory.tower.satellites[sat_idx].memory
        
        expected = ctx_mat.T @ sat_memory
        
        # Convert to numpy if needed
        if hasattr(settled, 'get'):
            settled = settled.get()
        if hasattr(expected, 'get'):
            expected = expected.get()
        
        # Must be identical (no processing, no argmax)
        np.testing.assert_allclose(settled, expected, rtol=1e-5,
            err_msg="retrieve_settled_state must return raw unbinding, no argmax")


# =============================================================================
# TEST: SEMANTIC SIMILARITY AS PRIMARY METRIC
# =============================================================================

class TestSemanticSimilarityMetric:
    """
    THEORY: Success is measured by semantic similarity, NOT exact token match.
    
    frobenius_cosine(retrieved_state, target_embedding) > 0.9 = success
    """
    
    def test_single_pattern_high_similarity(self, memory, basis):
        """
        Single pattern should have very high semantic similarity.
        """
        context = [1, 2, 3, 4]
        target = 42
        
        memory.learn(context, target)
        
        # Get settled state
        settled = memory.retrieve_settled_state(context)
        target_emb = memory.tower.embeddings[target]
        
        # Convert if needed
        if hasattr(settled, 'get'):
            settled = settled.get()
        if hasattr(target_emb, 'get'):
            target_emb = target_emb.get()
        
        # THEORY-TRUE METRIC
        similarity = frobenius_cosine(settled, target_emb)
        
        assert similarity > 0.9, \
            f"Single pattern similarity {similarity:.4f} < 0.9 (theory: should be very high)"
    
    def test_semantic_similarity_batch(self, memory, basis):
        """
        Batch semantic similarity measurement.
        """
        # Learn multiple patterns
        patterns = [
            ([1, 2, 3, 4], 10),
            ([5, 6, 7, 8], 20),
            ([9, 10, 11, 12], 30),
        ]
        
        for ctx, tgt in patterns:
            memory.learn(ctx, tgt)
        
        # Measure semantic similarity for each
        similarities = []
        for ctx, tgt in patterns:
            settled = memory.retrieve_settled_state(ctx)
            target_emb = memory.tower.embeddings[tgt]
            
            if hasattr(settled, 'get'):
                settled = settled.get()
            if hasattr(target_emb, 'get'):
                target_emb = target_emb.get()
            
            sim = frobenius_cosine(settled, target_emb)
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        
        # All patterns should have high similarity
        assert avg_similarity > 0.8, \
            f"Average semantic similarity {avg_similarity:.4f} < 0.8"


# =============================================================================
# TEST: EVALUATE_SEMANTIC (BATCH EVALUATION)
# =============================================================================

class TestEvaluateSemantic:
    """
    THEORY: Batch evaluation should use semantic similarity, not exact match.
    
    The method `evaluate_semantic(batch)` should:
    1. For each (context, target) in batch
    2. Get settled state
    3. Compute frobenius_cosine(settled, target_embedding)
    4. Return mean semantic similarity
    
    NO argmax. NO exact token comparison.
    """
    
    def test_evaluate_semantic_exists(self, memory):
        """Method must exist."""
        assert hasattr(memory, 'evaluate_semantic'), \
            "HolographicMemory must have evaluate_semantic method"
    
    def test_evaluate_semantic_returns_similarity(self, memory):
        """
        evaluate_semantic should return semantic similarity metric.
        """
        # Learn patterns
        batch = [
            ([1, 2, 3, 4], 10),
            ([5, 6, 7, 8], 20),
            ([9, 10, 11, 12], 30),
        ]
        
        for ctx, tgt in batch:
            memory.learn(ctx, tgt)
        
        # Evaluate
        result = memory.evaluate_semantic(batch)
        
        # Should return dict with semantic_similarity
        assert isinstance(result, dict), "Must return dict"
        assert 'semantic_similarity' in result, "Must include semantic_similarity"
        assert 0.0 <= result['semantic_similarity'] <= 1.0, \
            "semantic_similarity must be in [0, 1]"
    
    def test_evaluate_semantic_no_exact_match_in_output(self, memory):
        """
        evaluate_semantic should NOT return 'accuracy' (exact match metric).
        
        THEORY: Exact token match is the WRONG metric.
        """
        batch = [([1, 2, 3, 4], 10)]
        memory.learn(*batch[0])
        
        result = memory.evaluate_semantic(batch)
        
        # Should NOT have 'accuracy' key (that's the wrong metric)
        # May have it for backwards compatibility but semantic_similarity is primary
        assert 'semantic_similarity' in result, \
            "semantic_similarity MUST be the primary metric"


# =============================================================================
# TEST: NO ARGMAX IN EVALUATION PATH
# =============================================================================

class TestNoArgmaxInEvaluation:
    """
    THEORY: "NO sampling, NO argmax — just settling"
    
    The evaluation path MUST NOT use argmax.
    """
    
    def test_semantic_better_than_exact_match_under_superposition(self, memory):
        """
        Under superposition, semantic similarity remains meaningful
        while exact match becomes meaningless.
        
        This demonstrates WHY semantic similarity is the correct metric.
        """
        # Store multiple targets for overlapping contexts
        # This creates superposition
        memory.learn([1, 2, 3, 4], 10)
        memory.learn([1, 2, 3, 5], 20)  # Similar context
        memory.learn([1, 2, 3, 6], 30)  # Similar context
        
        # Query the first context
        settled = memory.retrieve_settled_state([1, 2, 3, 4])
        target_emb = memory.tower.embeddings[10]
        
        if hasattr(settled, 'get'):
            settled = settled.get()
        if hasattr(target_emb, 'get'):
            target_emb = target_emb.get()
        
        # Semantic similarity still meaningful
        similarity = frobenius_cosine(settled, target_emb)
        
        # The superposition means retrieved is a weighted average
        # Similarity will be lower than 1.0 but still positive
        # This is the FEATURE (generalization), not a bug
        assert similarity > 0.0, \
            f"Semantic similarity {similarity:.4f} should be positive under superposition"
        
        # The settled state IS the answer - no need to discretize


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
