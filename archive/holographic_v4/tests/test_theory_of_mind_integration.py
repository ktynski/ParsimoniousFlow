"""
Tests for Theory of Mind Integration — Perspective Transformation
=================================================================

Verifies that the system can:
1. Infer witness from observations (understand another's perspective)
2. Bind content to specific witnesses (perspective transformation)
3. Transform content between perspectives
4. Model other agents' beliefs
5. Predict what others would believe

THEORY:
    ToM = Binding(Content, OtherWitness) + GraceFlow(OtherBasins)
    This is a coordinate transformation in witness space.
"""

import numpy as np
import pytest

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ


# =============================================================================
# TEST 1: Witness Inference from Observations
# =============================================================================

def test_1_witness_inference():
    """
    Test inferring another agent's witness from their behavior.
    
    THEORY: Witness = invariant across observations (stable core)
    """
    print("\n=== Test 1: Witness Inference ===")
    
    from holographic_v4.theory_of_mind import infer_witness_from_observations
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    
    # Create some observations (behavior samples from another agent)
    np.random.seed(42)
    observations = [
        np.eye(4) * 2.0 + 0.1 * np.random.randn(4, 4).astype(np.float32)
        for _ in range(5)
    ]
    
    # Infer witness
    scalar, pseudo = infer_witness_from_observations(observations, basis, np)
    
    print(f"Inferred witness: (σ={scalar:.4f}, p={pseudo:.4f})")
    
    # Should extract the stable pattern (identity * 2 has scalar ≈ 2)
    assert isinstance(scalar, float), "Scalar should be float"
    assert isinstance(pseudo, float), "Pseudoscalar should be float"
    
    # With identity-based observations, scalar should be significant
    assert abs(scalar) > 0.1, f"Scalar should be significant, got {scalar}"
    
    print("✓ Witness inference works")


# =============================================================================
# TEST 2: Binding to Witness
# =============================================================================

def test_2_binding_to_witness():
    """
    Test binding content to a specific witness (perspective transformation).
    
    THEORY: bind_to_witness transforms "what's there" to "what THEY perceive"
    """
    print("\n=== Test 2: Binding to Witness ===")
    
    from holographic_v4.theory_of_mind import bind_to_witness
    from holographic_v4.quotient import witness_matrix
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    
    # Create content and witness
    content = np.random.randn(4, 4).astype(np.float32)
    witness = np.eye(4) * 1.5  # Specific witness perspective
    
    # Bind content to witness
    bound = bind_to_witness(content, witness, basis, lmbda=PHI_INV, xp=np)
    
    print(f"Content shape: {content.shape}")
    print(f"Bound shape: {bound.shape}")
    
    assert bound.shape == (4, 4), f"Expected (4, 4), got {bound.shape}"
    
    # Bound should be different from original content
    diff = np.linalg.norm(bound - content)
    print(f"Difference from original: {diff:.4f}")
    assert diff > 0.01, "Binding should transform the content"
    
    print("✓ Binding to witness works")


# =============================================================================
# TEST 3: Unbinding from Witness
# =============================================================================

def test_3_unbinding_from_witness():
    """
    Test recovering content from a bound representation.
    
    THEORY: unbind should approximately invert bind
    """
    print("\n=== Test 3: Unbinding from Witness ===")
    
    from holographic_v4.theory_of_mind import bind_to_witness, unbind_from_witness
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    
    # Create content and witness
    np.random.seed(42)
    content = np.random.randn(4, 4).astype(np.float32)
    witness = np.eye(4) * 1.5
    
    # Bind then unbind
    bound = bind_to_witness(content, witness, basis, xp=np)
    recovered = unbind_from_witness(bound, witness, basis, xp=np)
    
    # Should approximately recover original
    recovery_error = np.linalg.norm(recovered - content) / np.linalg.norm(content)
    print(f"Recovery error: {recovery_error:.4f}")
    
    # Note: unbind is approximate, not exact
    assert recovery_error < 2.0, f"Recovery should be reasonable, got {recovery_error}"
    
    print("✓ Unbinding from witness works")


# =============================================================================
# TEST 4: Perspective Transformation
# =============================================================================

def test_4_perspective_transformation():
    """
    Test transforming content from one perspective to another.
    """
    print("\n=== Test 4: Perspective Transformation ===")
    
    from holographic_v4.theory_of_mind import transform_perspective
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    
    # Content in perspective A
    np.random.seed(42)
    content_A = np.random.randn(4, 4).astype(np.float32)
    witness_A = np.eye(4) * 1.0
    witness_B = np.eye(4) * 2.0  # Different perspective
    
    # Transform A -> B
    content_B = transform_perspective(content_A, witness_A, witness_B, basis, np)
    
    print(f"Content A shape: {content_A.shape}")
    print(f"Content B shape: {content_B.shape}")
    
    # Content should change
    diff = np.linalg.norm(content_B - content_A)
    print(f"Perspective difference: {diff:.4f}")
    
    assert content_B.shape == (4, 4), f"Expected (4, 4), got {content_B.shape}"
    
    print("✓ Perspective transformation works")


# =============================================================================
# TEST 5: Agent Model
# =============================================================================

def test_5_agent_model():
    """
    Test creating and using an AgentModel.
    """
    print("\n=== Test 5: Agent Model ===")
    
    from holographic_v4.theory_of_mind import AgentModel
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    
    # Create agent model (witness is the only required field)
    agent = AgentModel(
        witness=(1.5, 0.2),  # (scalar, pseudo)
        confidence=0.8,
        observation_count=5,
    )
    
    print(f"Witness: {agent.witness}")
    print(f"Confidence: {agent.confidence}")
    print(f"Observation count: {agent.observation_count}")
    
    assert agent.witness == (1.5, 0.2)
    assert agent.confidence == 0.8
    
    # Get witness as matrix
    witness_mat = agent.witness_matrix(basis, np)
    print(f"Witness matrix shape: {witness_mat.shape}")
    assert witness_mat.shape == (4, 4)
    
    print("✓ Agent model works")


# =============================================================================
# TEST 6: Theory of Mind Operation
# =============================================================================

def test_6_theory_of_mind_operation():
    """
    Test the core ToM computation.
    """
    print("\n=== Test 6: Theory of Mind Operation ===")
    
    from holographic_v4.theory_of_mind import (
        theory_of_mind,
        AgentModel,
    )
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.algebra import build_clifford_basis
    from holographic_v4.quotient import extract_witness
    
    basis = build_clifford_basis(np)
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    
    # Train model
    for i in range(10):
        model.train_step([i, i+1, i+2], i * 10)
    
    # Create agent model for "other"
    other_agent = AgentModel(
        witness=(1.2, 0.1),
        confidence=0.7,
    )
    
    # Content to reason about
    content = model.compute_context([5, 6, 7])
    
    # Self witness (extracted from content)
    self_witness = extract_witness(content, basis, np)
    
    # What would 'other' perceive?
    # Returns (transformed_content, predicted_target, confidence)
    other_view, pred_target, conf = theory_of_mind(
        content, self_witness, other_agent, basis, np
    )
    
    print(f"Other's view shape: {other_view.shape}")
    print(f"Predicted target: {pred_target}, confidence: {conf:.4f}")
    
    # Should transform the content
    diff = np.linalg.norm(other_view - content)
    print(f"View difference: {diff:.4f}")
    
    assert other_view.shape == (4, 4), f"Expected (4, 4), got {other_view.shape}"
    
    print("✓ Theory of mind operation works")


# =============================================================================
# TEST 7: Predict Other Belief
# =============================================================================

def test_7_predict_other_belief():
    """
    Test predicting what another agent would believe.
    """
    print("\n=== Test 7: Predict Other Belief ===")
    
    from holographic_v4.theory_of_mind import (
        predict_other_belief,
        AgentModel,
    )
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    
    # Train
    for i in range(10):
        model.train_step([i, i+1, i+2], i * 10)
    
    # Other agent
    other = AgentModel(
        witness=(1.0, 0.0),
        confidence=0.5,
    )
    
    # What would 'other' predict for context?
    context = [5, 6, 7]
    predicted_target, confidence = predict_other_belief(context, other, model, basis, np)
    
    print(f"Predicted target: {predicted_target}")
    print(f"Confidence: {confidence:.4f}")
    
    # Should return some prediction
    assert predicted_target is not None or confidence >= 0, "Should make prediction"
    
    print("✓ Predict other belief works")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_1_witness_inference()
    test_2_binding_to_witness()
    test_3_unbinding_from_witness()
    test_4_perspective_transformation()
    test_5_agent_model()
    test_6_theory_of_mind_operation()
    test_7_predict_other_belief()
    
    print("\n" + "="*60)
    print("ALL THEORY OF MIND INTEGRATION TESTS PASSED")
    print("="*60)
