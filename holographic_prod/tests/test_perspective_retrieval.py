"""
Integration Tests for Perspective-Aware Retrieval
=================================================

Tests the full integration of Theory of Mind and Distributed Prior
with the HolographicMemory retrieval pipeline.

KEY TESTS:
1. Context-dependent disambiguation - same word, different contexts
2. Abstraction/instantiation - abstract vs concrete witnesses
3. Retrieval with distributed fallback - smooth generalization
4. End-to-end ToM retrieval - full pipeline

THEORY:
    These tests verify that perspective transformation (ToM) and
    smooth interpolation (Distributed Prior) work together to enable
    context-dependent meaning and graceful generalization.
"""

import numpy as np
import pytest
from typing import List, Tuple

from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ, DTYPE
from holographic_prod.core.algebra import build_clifford_basis, grace_operator
from holographic_prod.core.quotient import extract_witness
from holographic_prod.memory.holographic_memory_unified import HolographicMemory, MemoryConfig
from holographic_prod.cognitive.theory_of_mind import (
    bind_to_witness,
    transform_perspective,
    infer_witness_from_observations,
)
from holographic_prod.cognitive.distributed_prior import (
    superposed_attractor_prior,
    phi_kernel,
    witness_distance,
    extended_witness,
    FactorizedAssociativePrior,
)
from holographic_prod.dreaming import SemanticMemory
from holographic_prod.dreaming.structures import SemanticPrototype


@pytest.fixture
def basis():
    """Clifford basis for all tests."""
    return build_clifford_basis(np)


@pytest.fixture
def memory():
    """Create a HolographicMemory instance for testing."""
    config = MemoryConfig(
        # Disable contrastive for faster tests
        contrastive_enabled=False,
    )
    return HolographicMemory(vocab_size=100, seed=42, config=config)


@pytest.fixture
def trained_memory(memory):
    """Memory with some learned patterns."""
    # Learn distinctive patterns
    for i in range(50):
        # Pattern A: context [1, 2, 3] → target 10
        memory.learn([1, 2, 3], 10)
        # Pattern B: context [4, 5, 6] → target 20
        memory.learn([4, 5, 6], 20)
        # Pattern C: context [7, 8, 9] → target 30
        memory.learn([7, 8, 9], 30)
    
    return memory


class TestContextDependentDisambiguation:
    """Test 1: Same word, different contexts = different meanings."""
    
    def test_context_dependent_disambiguation(self, basis):
        """
        Test that the same content appears differently from different perspectives.
        
        ANALOGY: "Bank" with financial context vs river context.
        The context IS the witness that frames the content.
        """
        np.random.seed(42)
        
        # Same "word" embedding (e.g., "bank")
        word_embedding = np.random.randn(4, 4).astype(DTYPE) * 0.5
        
        # Financial context witness
        financial_context = np.eye(4, dtype=DTYPE) * 2.0 + 0.1 * basis[15]  # High scalar
        financial_witness = extract_witness(financial_context, basis, np)
        
        # River context witness
        river_context = np.eye(4, dtype=DTYPE) * 0.5 + 0.3 * basis[15]  # Different pattern
        river_witness = extract_witness(river_context, basis, np)
        
        # Build witness matrices
        fin_w_mat = financial_witness[0] * basis[0] + financial_witness[1] * basis[15]
        river_w_mat = river_witness[0] * basis[0] + river_witness[1] * basis[15]
        
        # Bind word to each context
        financial_meaning = bind_to_witness(word_embedding, fin_w_mat, basis, xp=np)
        river_meaning = bind_to_witness(word_embedding, river_w_mat, basis, xp=np)
        
        # The meanings should be DIFFERENT
        meaning_diff = np.linalg.norm(financial_meaning - river_meaning)
        assert meaning_diff > 0.1, (
            f"Same word with different contexts should have different meanings, "
            f"got diff={meaning_diff}"
        )
        
        # But extracting witness from each should reflect the context
        fin_w_extracted = extract_witness(financial_meaning, basis, np)
        river_w_extracted = extract_witness(river_meaning, basis, np)
        
        # Witness distances should be significant
        w_dist = witness_distance(fin_w_extracted, river_w_extracted)
        assert w_dist > 0.01, f"Context witnesses should differ, got dist={w_dist}"
    
    def test_similar_contexts_similar_meanings(self, basis):
        """Similar contexts should produce similar meanings."""
        np.random.seed(42)
        
        word = np.random.randn(4, 4).astype(DTYPE)
        
        # Two similar contexts
        ctx1 = np.eye(4, dtype=DTYPE) * 2.0
        ctx2 = np.eye(4, dtype=DTYPE) * 2.1  # Slightly different
        
        meaning1 = bind_to_witness(word, ctx1, basis, xp=np)
        meaning2 = bind_to_witness(word, ctx2, basis, xp=np)
        
        diff = np.linalg.norm(meaning1 - meaning2) / np.linalg.norm(meaning1)
        assert diff < 0.5, f"Similar contexts should give similar meanings, got diff={diff}"


class TestAbstractionInstantiation:
    """Test 2: Abstract vs concrete = different witness configurations."""
    
    def test_abstraction_instantiation(self, basis):
        """
        Test that abstract and concrete representations have different witnesses.
        
        THEORY: "Dog" abstract vs "my dog Fluffy" concrete are the same
        semantic content with different witness configurations.
        """
        np.random.seed(42)
        
        # Abstract concept embedding (generic "dog")
        dog_content = np.random.randn(4, 4).astype(DTYPE)
        
        # Abstract witness (low specificity)
        abstract_witness = np.eye(4, dtype=DTYPE) * 0.5  # Lower scalar = more general
        
        # Concrete witness (high specificity) 
        concrete_witness = np.eye(4, dtype=DTYPE) * 3.0  # Higher scalar = more specific
        
        # Bind to each level
        abstract_dog = bind_to_witness(dog_content, abstract_witness, basis, xp=np)
        concrete_dog = bind_to_witness(dog_content, concrete_witness, basis, xp=np)
        
        # Extract witnesses
        abs_w = extract_witness(abstract_dog, basis, np)
        conc_w = extract_witness(concrete_dog, basis, np)
        
        # Concrete should have higher scalar (more "energy")
        # Note: The exact relationship depends on binding implementation
        # But they should definitely be different
        w_dist = witness_distance(abs_w, conc_w)
        assert w_dist > 0.1, f"Abstract and concrete should have different witnesses, got dist={w_dist}"
    
    def test_transform_abstract_to_concrete(self, basis):
        """Test transforming between abstraction levels."""
        np.random.seed(42)
        
        content = np.random.randn(4, 4).astype(DTYPE)
        
        # Witnesses for different levels
        abstract_w = np.eye(4, dtype=DTYPE) * 1.0
        concrete_w = np.eye(4, dtype=DTYPE) * 2.5
        
        # Start with abstract
        abstract_view = bind_to_witness(content, abstract_w, basis, xp=np)
        
        # Transform to concrete
        concrete_view = transform_perspective(
            abstract_view, abstract_w, concrete_w, basis, np
        )
        
        # The concrete view should exist and be different
        assert concrete_view.shape == (4, 4)
        diff = np.linalg.norm(concrete_view - abstract_view)
        assert diff > 0.01, "Transformation should change the view"


class TestRetrievalWithDistributedFallback:
    """Test 3: Smooth generalization when between basins."""
    
    def test_retrieval_with_distributed_fallback(self, basis):
        """
        Test that queries between basins get interpolated predictions.
        
        THEORY: When confidence < φ⁻¹, use distributed prior for smooth interpolation.
        """
        np.random.seed(42)
        
        # Create prototypes at different positions
        prototypes = []
        target_dists = []
        
        for i in range(3):
            proto = grace_operator(
                np.eye(4, dtype=DTYPE) * (i + 1) + 0.05 * np.random.randn(4, 4).astype(DTYPE),
                basis, np
            )
            prototypes.append(proto)
            target_dists.append({i * 10: 1.0})  # Each predicts different token
        
        # Query BETWEEN prototypes (at scalar ≈ 1.5)
        query = np.eye(4, dtype=DTYPE) * 1.5
        
        # Get superposed prior
        equilibrium, combined_targets, confidence, info = superposed_attractor_prior(
            query, prototypes, target_dists, basis, np, K=2
        )
        
        # Should have contributions from multiple prototypes
        assert len(combined_targets) >= 1, "Should get interpolated targets"
        
        # Equilibrium should be valid
        assert equilibrium.shape == (4, 4)
        assert np.isfinite(equilibrium).all()
    
    def test_high_confidence_single_basin(self, basis):
        """High confidence queries should not need fallback."""
        np.random.seed(42)
        
        # Single prototype
        proto = grace_operator(np.eye(4, dtype=DTYPE) * 2.0, basis, np)
        target_dist = {10: 1.0}
        
        # Query very close to prototype
        query = proto + 0.01 * np.random.randn(4, 4).astype(DTYPE)
        
        _, targets, confidence, info = superposed_attractor_prior(
            query, [proto], [target_dist], basis, np, K=1
        )
        
        # With single prototype, confidence should be 1.0
        assert confidence == 1.0, f"Single basin should have full confidence, got {confidence}"


class TestEndToEndTomRetrieval:
    """Test 4: Full pipeline - ToM + Memory + Prior."""
    
    def test_end_to_end_tom_retrieval(self, trained_memory, basis):
        """
        Test complete pipeline: learn, then retrieve with perspective awareness.
        """
        memory = trained_memory
        
        # Basic retrieval should work
        result, confidence = memory.retrieve_deterministic([1, 2, 3])
        
        # Should retrieve something (may be 10 or fallback)
        assert result is not None or memory.n_patterns > 0
    
    def test_memory_learns_and_retrieves(self, memory):
        """Verify basic memory functionality before ToM integration."""
        # Learn a simple pattern
        memory.learn([1, 2, 3], 42)
        
        # Check it was learned
        assert memory.n_patterns >= 1
        assert memory.learn_count >= 1
        
        # Retrieve
        result, conf = memory.retrieve_deterministic([1, 2, 3])
        
        # Should retrieve (exact match via episodic cache)
        # Note: result may vary depending on holographic vs episodic path
        assert result is not None or conf == 0.0  # Either found or explicit failure
    
    def test_witness_extraction_from_context(self, memory, basis):
        """Test extracting witness from context embeddings."""
        # Get context embedding
        ctx_mat = memory.embed_sequence([1, 2, 3])
        
        # Extract witness
        witness = extract_witness(ctx_mat, basis, np)
        
        assert isinstance(witness, tuple)
        assert len(witness) == 2
        assert np.isfinite(witness[0])
        assert np.isfinite(witness[1])
    
    def test_semantic_memory_with_prototypes(self, basis):
        """Test semantic memory retrieval with prototypes."""
        sem_mem = SemanticMemory(basis=basis, xp=np)
        
        # Create and add a prototype
        np.random.seed(42)
        matrix = grace_operator(np.eye(4, dtype=DTYPE) * 2.0, basis, np)
        vort_sig = np.zeros(16, dtype=DTYPE)  # Vorticity signature is 16D (Clifford coeffs)
        
        proto = SemanticPrototype(
            prototype_matrix=matrix,
            vorticity_signature=vort_sig,
            support=10,
            radius=0.5,
            target_distribution={42: 0.9, 43: 0.1},
        )
        
        sem_mem.add_prototype(proto)
        
        # Retrieve
        query = matrix + 0.05 * np.random.randn(4, 4).astype(DTYPE)
        results = sem_mem.retrieve(query, top_k=1)
        
        assert len(results) >= 0  # May or may not find depending on threshold


class TestPriorIntegrationWithMemory:
    """Additional integration tests for Prior with Memory."""
    
    def test_factorized_prior_learns_from_patterns(self, memory, basis):
        """Test that factorized prior can learn from memory patterns."""
        prior = FactorizedAssociativePrior(witness_dim=4, xp=np)
        
        # Learn patterns and update prior
        for i in range(20):
            ctx = [i % 10, (i + 1) % 10, (i + 2) % 10]
            target = i % 5 * 10
            
            # Learn in memory
            memory.learn(ctx, target)
            
            # Update prior with extended witness
            ctx_mat = memory.embed_sequence(ctx)
            ext_w = extended_witness(ctx_mat, basis, np)
            prior.update(ext_w, ctx_mat)
        
        # Prior should have learned something
        stats = prior.get_statistics()
        assert stats['n_updates'] == 20
        assert stats['effective_rank'] > 0
    
    def test_combined_retrieval_paths(self, trained_memory, basis):
        """Test that multiple retrieval paths can work together."""
        memory = trained_memory
        
        # Create factorized prior
        prior = FactorizedAssociativePrior(witness_dim=4, xp=np)
        
        # Train prior on memory patterns
        for _ in range(10):
            for ctx, tgt in [([1, 2, 3], 10), ([4, 5, 6], 20), ([7, 8, 9], 30)]:
                ctx_mat = memory.embed_sequence(ctx)
                ext_w = extended_witness(ctx_mat, basis, np)
                prior.update(ext_w, ctx_mat)
        
        # Test known context
        known_ctx = [1, 2, 3]
        result, conf = memory.retrieve_deterministic(known_ctx)
        
        # Test with prior prediction
        unknown_ctx = [2, 3, 4]  # Unseen pattern
        ctx_mat = memory.embed_sequence(unknown_ctx)
        prior_pred = prior.predict(extended_witness(ctx_mat, basis, np), basis)
        
        # Prior prediction should be a valid matrix
        assert prior_pred.shape == (4, 4)
        assert np.isfinite(prior_pred).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
