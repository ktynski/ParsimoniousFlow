"""
Tests for Theory of Mind Module — Perspective Transformation
============================================================

Verifies that the system can:
1. Infer witness from observations (understand another's perspective)
2. Bind content to specific witnesses (perspective transformation)
3. Transform content between perspectives (roundtrip)
4. Build agent models incrementally
5. Make ToM predictions
6. Disambiguate via different witnesses

THEORY:
    ToM = Binding(Content, OtherWitness) + GraceFlow(OtherBasins)
    This is a coordinate transformation in witness space.
    
    Critically: ToM is NOT just for multi-agent scenarios.
    It's core to context-dependent meaning, abstraction, and disambiguation.
"""

import numpy as np
import pytest
from typing import Tuple

from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, DTYPE
from holographic_prod.core.algebra import build_clifford_basis, grace_operator
from holographic_prod.core.quotient import extract_witness, witness_matrix
from holographic_prod.cognitive.theory_of_mind import (
    infer_witness_from_observations,
    infer_witness_from_observations_weighted,
    bind_to_witness,
    unbind_from_witness,
    transform_perspective,
    AgentModel,
    AgentModelBuilder,
    theory_of_mind,
)


@pytest.fixture
def basis():
    """Clifford basis for all tests."""
    return build_clifford_basis(np)


@pytest.fixture
def sample_observations(basis):
    """Create sample observations with known witness pattern.
    
    Each observation is a single [4, 4] matrix representing observed behavior.
    AgentModelBuilder.observe() accumulates these, and build() computes
    vorticity from the stacked sequence.
    """
    np.random.seed(42)
    # Observations centered around 2*I (scalar witness ≈ 2)
    observations = [
        np.eye(4, dtype=DTYPE) * 2.0 + 0.1 * np.random.randn(4, 4).astype(DTYPE)
        for _ in range(5)
    ]
    return observations


@pytest.fixture
def sample_single_matrices():
    """Alias for sample_observations for clarity in some tests."""
    np.random.seed(42)
    return [
        np.eye(4, dtype=DTYPE) * 2.0 + 0.1 * np.random.randn(4, 4).astype(DTYPE)
        for _ in range(5)
    ]


class TestWitnessInference:
    """Test 1: Inferring witness from observations."""
    
    def test_witness_inference_from_observations(self, basis, sample_single_matrices):
        """
        Test inferring another agent's witness from their behavior.
        
        THEORY: Witness = invariant across observations (stable core).
        By averaging witnesses, transient content cancels out.
        """
        # infer_witness_from_observations expects single 4x4 matrices
        scalar, pseudo = infer_witness_from_observations(sample_single_matrices, basis, np)
        
        # Should extract the stable pattern (identity * 2 has scalar ≈ 2)
        assert isinstance(scalar, float), "Scalar should be float"
        assert isinstance(pseudo, float), "Pseudoscalar should be float"
        
        # With identity-based observations, scalar should be significant
        assert abs(scalar) > 0.1, f"Scalar should be significant, got {scalar}"
    
    def test_empty_observations_returns_zero(self, basis):
        """Empty observations should return zero witness."""
        scalar, pseudo = infer_witness_from_observations([], basis, np)
        assert scalar == 0.0
        assert pseudo == 0.0
    
    def test_weighted_inference(self, basis, sample_single_matrices):
        """Weighted inference should emphasize high-weight observations."""
        # Equal weights should match unweighted
        uniform_result = infer_witness_from_observations(sample_single_matrices, basis, np)
        weighted_result = infer_witness_from_observations_weighted(
            sample_single_matrices, 
            weights=[1.0] * len(sample_single_matrices),
            basis=basis,
            xp=np,
        )
        
        # Results should be similar (may differ due to Grace stabilization)
        assert abs(uniform_result[0] - weighted_result[0]) < 1.0


class TestBindToWitness:
    """Test 2: Binding content to specific witnesses."""
    
    def test_bind_to_witness_transforms_content(self, basis):
        """
        Test binding content to a specific witness (perspective transformation).
        
        THEORY: bind_to_witness transforms "what's there" to "what THEY perceive"
        """
        np.random.seed(42)
        content = np.random.randn(4, 4).astype(DTYPE)
        witness = np.eye(4, dtype=DTYPE) * 1.5  # Specific witness perspective
        
        # Bind content to witness
        bound = bind_to_witness(content, witness, basis, lmbda=PHI_INV, xp=np)
        
        assert bound.shape == (4, 4), f"Expected (4, 4), got {bound.shape}"
        
        # Bound should be different from original content
        diff = np.linalg.norm(bound - content)
        assert diff > 0.01, "Binding should transform the content"
    
    def test_bind_with_zero_witness_returns_content(self, basis):
        """Binding with zero witness should return content unchanged."""
        np.random.seed(42)
        content = np.random.randn(4, 4).astype(DTYPE)
        zero_witness = np.zeros((4, 4), dtype=DTYPE)
        
        bound = bind_to_witness(content, zero_witness, basis, xp=np)
        
        # Should return content (approximately, since witness norm check)
        assert np.allclose(bound, content, atol=1e-5)


class TestTransformPerspective:
    """Test 3: Transform perspective roundtrip."""
    
    def test_transform_perspective_roundtrip(self, basis):
        """
        Test that transform_perspective is approximately invertible.
        
        THEORY: unbind(bind(content)) should approximately recover content.
        """
        np.random.seed(42)
        
        # Create content and two different witnesses
        content = np.eye(4, dtype=DTYPE) + 0.1 * np.random.randn(4, 4).astype(DTYPE)
        witness_a = np.eye(4, dtype=DTYPE) * 1.0
        witness_b = np.eye(4, dtype=DTYPE) * 2.0
        
        # Transform A → B
        transformed = transform_perspective(content, witness_a, witness_b, basis, np)
        
        # Transform B → A (should recover original)
        recovered = transform_perspective(transformed, witness_b, witness_a, basis, np)
        
        # Should approximately recover original
        # Note: Not exact due to numerical precision and witness extraction
        recovery_error = np.linalg.norm(recovered - content) / np.linalg.norm(content)
        assert recovery_error < 0.5, f"Roundtrip error too high: {recovery_error}"
    
    def test_transform_same_perspective_is_near_identity(self, basis):
        """Transforming to the same perspective should preserve structure.
        
        NOTE: Not exact identity due to the unbind/rebind process, but
        the witness components should be preserved.
        """
        np.random.seed(42)
        content = np.eye(4, dtype=DTYPE) * 2.0  # Clear witness structure
        witness = np.eye(4, dtype=DTYPE) * 1.5
        
        transformed = transform_perspective(content, witness, witness, basis, np)
        
        # Extract witnesses - should be similar
        orig_w = extract_witness(content, basis, np)
        trans_w = extract_witness(transformed, basis, np)
        
        # Witness should be approximately preserved
        witness_diff = abs(orig_w[0] - trans_w[0]) + abs(orig_w[1] - trans_w[1])
        assert witness_diff < 2.0, f"Witness should be preserved, got diff={witness_diff}"


class TestAgentModelBuilder:
    """Test 4: Incremental agent model construction."""
    
    def test_agent_model_builder(self, basis, sample_observations):
        """
        Test building an agent model from observations.
        
        THEORY: Observations accumulate to form a consistent model.
        
        NOTE: sample_observations are [n, 4, 4] sequences for vorticity.
        """
        builder = AgentModelBuilder(basis, np)
        
        # Add observations - each is a [n, 4, 4] sequence
        for obs in sample_observations:
            builder.observe(obs)
        
        # Build model
        model = builder.build()
        
        assert isinstance(model, AgentModel)
        assert model.observation_count == len(sample_observations)
        assert isinstance(model.witness, tuple)
        assert len(model.witness) == 2
        assert model.confidence > 0, "Confidence should increase with observations"
    
    def test_agent_model_similarity(self, basis):
        """Test AgentModel.is_similar_to() method."""
        model_a = AgentModel(witness=(1.0, 0.5), confidence=0.8)
        model_b = AgentModel(witness=(1.0, 0.5), confidence=0.9)  # Same witness
        model_c = AgentModel(witness=(5.0, 3.0), confidence=0.7)  # Different witness
        
        assert model_a.is_similar_to(model_b), "Same witness should be similar"
        assert not model_a.is_similar_to(model_c), "Different witness should not be similar"
    
    def test_builder_reset(self, basis, sample_observations):
        """Test that reset clears all observations."""
        builder = AgentModelBuilder(basis, np)
        
        for obs in sample_observations:
            builder.observe(obs)
        
        assert len(builder.observations) == len(sample_observations)
        
        builder.reset()
        
        assert len(builder.observations) == 0
        assert len(builder.contexts) == 0
        assert len(builder.targets) == 0


class TestTheoryOfMindPrediction:
    """Test 5: Full ToM prediction pipeline."""
    
    def test_theory_of_mind_prediction(self, basis, sample_observations):
        """
        Test full ToM pipeline: content + self_witness + other_model → prediction.
        
        THEORY: ToM transforms content to another's perspective and predicts
        what they would perceive.
        """
        # Build an agent model using sequence observations
        builder = AgentModelBuilder(basis, np)
        for obs in sample_observations:
            builder.observe(obs)
        other_model = builder.build()
        
        # Create content and self witness (single 4x4 matrix)
        np.random.seed(42)
        content = np.eye(4, dtype=DTYPE) + 0.1 * np.random.randn(4, 4).astype(DTYPE)
        self_witness = extract_witness(content, basis, np)
        
        # Run ToM
        perception, predicted_target, confidence = theory_of_mind(
            content, self_witness, other_model, basis, np
        )
        
        assert perception.shape == (4, 4), "Perception should be 4x4 matrix"
        assert isinstance(predicted_target, int), "Predicted target should be int"
        assert 0.0 <= confidence <= 1.0, "Confidence should be in [0, 1]"
    
    def test_tom_with_no_semantic_memory(self, basis):
        """ToM should work even without semantic memory (returns -1 for target)."""
        # Simple model without semantic memory
        other_model = AgentModel(witness=(1.0, 0.5), confidence=0.8)
        
        content = np.eye(4, dtype=DTYPE)
        self_witness = (0.5, 0.2)
        
        perception, predicted_target, confidence = theory_of_mind(
            content, self_witness, other_model, basis, np
        )
        
        assert perception.shape == (4, 4)
        assert predicted_target == -1, "Without semantic memory, target should be -1"


class TestDisambiguationViaWitness:
    """Test 6: Disambiguation - same content, different witnesses."""
    
    def test_tom_for_disambiguation(self, basis):
        """
        Test that the same content appears differently from different perspectives.
        
        THEORY: Context IS a witness. "Bank" with financial context vs river 
        context are different witnesses binding the same content.
        
        This is why ToM is core to language understanding, not just social cognition.
        """
        np.random.seed(42)
        
        # Same "content" (e.g., the word "bank")
        content = np.random.randn(4, 4).astype(DTYPE)
        
        # Two different witnesses (contexts)
        # Financial context: high scalar (concrete, specific)
        financial_witness = np.eye(4, dtype=DTYPE) * 2.0
        # River context: different pattern
        river_witness = np.eye(4, dtype=DTYPE) * 0.5 + 0.3 * basis[15]  # Add pseudoscalar
        
        # Bind to each witness
        financial_meaning = bind_to_witness(content, financial_witness, basis, xp=np)
        river_meaning = bind_to_witness(content, river_witness, basis, xp=np)
        
        # The two meanings should be DIFFERENT
        meaning_diff = np.linalg.norm(financial_meaning - river_meaning)
        assert meaning_diff > 0.1, (
            f"Same content with different witnesses should have different meanings, "
            f"got diff={meaning_diff}"
        )
        
        # But they should both have similar magnitude (they're the same "amount" of content)
        fin_norm = np.linalg.norm(financial_meaning)
        river_norm = np.linalg.norm(river_meaning)
        norm_ratio = max(fin_norm, river_norm) / min(fin_norm, river_norm)
        assert norm_ratio < 5.0, f"Norms should be comparable, got ratio={norm_ratio}"
    
    def test_witness_extraction_is_deterministic(self, basis):
        """Witness extraction should be deterministic."""
        content = np.eye(4, dtype=DTYPE) * 2.0
        
        w1 = extract_witness(content, basis, np)
        w2 = extract_witness(content, basis, np)
        
        assert w1 == w2, "Witness extraction should be deterministic"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
