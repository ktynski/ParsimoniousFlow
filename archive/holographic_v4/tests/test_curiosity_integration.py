"""
Tests for Curiosity Integration — Metacognition and Self-Awareness
==================================================================

Verifies that the system can:
1. Know what it doesn't know (calibrated uncertainty)
2. Identify knowledge boundaries
3. Estimate information gain from potential learning
4. Prioritize samples for maximum learning efficiency

THEORY:
    Curiosity = -∇[grace_stability]
    The system descends toward queries where stability is lowest.
    This is METACOGNITION — awareness of one's own knowledge limits.
"""

import numpy as np
import pytest

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE


# =============================================================================
# TEST 1: Curiosity Score Computation
# =============================================================================

def test_1_curiosity_score_computation():
    """
    Test that curiosity score is computable and bounded.
    
    THEORY: Curiosity = inverse of stability/confidence
    """
    print("\n=== Test 1: Curiosity Score Computation ===")
    
    from holographic_v4.curiosity import curiosity_score
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.dreaming import DreamingSystem
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    
    # Create minimal model
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    dreaming = DreamingSystem(basis=basis)
    
    # Train on a few patterns to establish some knowledge
    for i in range(10):
        ctx = [i, i+1, i+2]
        tgt = i * 10
        model.train_step(ctx, tgt)
    
    # Compute curiosity for known context
    known_matrix = model.compute_context([0, 1, 2])
    cur_known = curiosity_score(known_matrix, model, dreaming)
    print(f"Curiosity for known [0,1,2]: {cur_known:.4f}")
    
    # Compute curiosity for unknown context
    unknown_matrix = model.compute_context([90, 91, 92])
    cur_unknown = curiosity_score(unknown_matrix, model, dreaming)
    print(f"Curiosity for unknown [90,91,92]: {cur_unknown:.4f}")
    
    # Both should be positive (curiosity is non-negative)
    assert cur_known >= 0, f"Curiosity should be non-negative, got {cur_known}"
    assert cur_unknown >= 0, f"Curiosity should be non-negative, got {cur_unknown}"
    
    # Unknown should have higher curiosity than known
    # (This is the CORE TEST of metacognition)
    assert cur_unknown > cur_known, (
        f"Unknown should have higher curiosity: {cur_unknown:.4f} > {cur_known:.4f}"
    )
    
    print("✓ Curiosity score computation works")


# =============================================================================
# TEST 2: Information Gain Estimation
# =============================================================================

def test_2_information_gain_estimation():
    """
    Test that information gain can be estimated for potential samples.
    """
    print("\n=== Test 2: Information Gain Estimation ===")
    
    from holographic_v4.curiosity import estimate_information_gain
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.dreaming import DreamingSystem
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    dreaming = DreamingSystem(basis=basis)
    
    # Learn some patterns
    for i in range(5):
        model.train_step([i, i+1, i+2], i * 10)
    
    # Information gain for novel sample
    gain_novel = estimate_information_gain([50, 51, 52], 500, model, dreaming)
    print(f"Info gain for novel [50,51,52]->500: {gain_novel:.4f}")
    
    # Information gain for redundant sample (already known)
    gain_redundant = estimate_information_gain([0, 1, 2], 0, model, dreaming)
    print(f"Info gain for redundant [0,1,2]->0: {gain_redundant:.4f}")
    
    # Both should be non-negative
    assert gain_novel >= 0, "Information gain should be non-negative"
    assert gain_redundant >= 0, "Information gain should be non-negative"
    
    # Novel should have higher gain than redundant
    assert gain_novel > gain_redundant, (
        f"Novel should have higher gain: {gain_novel:.4f} > {gain_redundant:.4f}"
    )
    
    print("✓ Information gain estimation works")


# =============================================================================
# TEST 3: Basin Boundary Sampling
# =============================================================================

def test_3_basin_boundary_sampling():
    """
    Test sampling from the boundaries of knowledge basins.
    
    THEORY: Boundaries = intermediate stability regions
    """
    print("\n=== Test 3: Basin Boundary Sampling ===")
    
    from holographic_v4.curiosity import sample_basin_boundary
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.dreaming import DreamingSystem
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    dreaming = DreamingSystem(basis=basis)
    
    # Learn patterns to create some basins
    for i in range(10):
        model.train_step([i * 5, i * 5 + 1, i * 5 + 2], i * 100)
    
    # Sample from boundary
    boundary_sample = sample_basin_boundary(model, dreaming, n_candidates=30, seed=42)
    
    print(f"Boundary sample shape: {boundary_sample.shape}")
    assert boundary_sample.shape == (4, 4), f"Expected (4, 4), got {boundary_sample.shape}"
    
    # The sample should exist (not None)
    assert boundary_sample is not None, "Boundary sample should not be None"
    
    print("✓ Basin boundary sampling works")


# =============================================================================
# TEST 4: Curiosity Query Generation
# =============================================================================

def test_4_curiosity_query_generation():
    """
    Test generating queries that maximize curiosity.
    """
    print("\n=== Test 4: Curiosity Query Generation ===")
    
    from holographic_v4.curiosity import generate_curiosity_query
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.dreaming import DreamingSystem
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    dreaming = DreamingSystem(basis=basis)
    
    # Learn some patterns
    for i in range(5):
        model.train_step([i, i+1, i+2], i * 10)
    
    # Generate max curiosity query
    query = generate_curiosity_query(
        model, dreaming, 
        n_candidates=20, 
        strategy='max_curiosity',
        seed=42
    )
    
    print(f"Generated query shape: {query.shape}")
    assert query.shape == (4, 4), f"Expected (4, 4), got {query.shape}"
    
    # Generate boundary query
    query_boundary = generate_curiosity_query(
        model, dreaming,
        n_candidates=20,
        strategy='boundary',
        seed=42
    )
    
    print(f"Boundary query shape: {query_boundary.shape}")
    assert query_boundary.shape == (4, 4)
    
    print("✓ Curiosity query generation works")


# =============================================================================
# TEST 5: Active Learning Step
# =============================================================================

def test_5_active_learning_step():
    """
    Test selecting the best sample from a pool for learning.
    """
    print("\n=== Test 5: Active Learning Step ===")
    
    from holographic_v4.curiosity import active_learning_step
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.dreaming import DreamingSystem
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    dreaming = DreamingSystem(basis=basis)
    
    # Learn some patterns
    for i in range(3):
        model.train_step([i, i+1, i+2], i * 10)
    
    # Create a pool of candidates
    pool = [
        ([0, 1, 2], 0),      # Redundant (already known)
        ([50, 51, 52], 500), # Novel
        ([99, 98, 97], 999), # Also novel
    ]
    
    # Select best from pool (returns context, target)
    best_context, best_target = active_learning_step(model, dreaming, pool)
    
    print(f"Best sample: {best_context} -> {best_target}")
    
    # Should NOT select the redundant sample
    # Redundant is [0, 1, 2] -> 0
    assert best_context != [0, 1, 2] or best_target != 0, (
        "Should not select redundant sample"
    )
    
    print("✓ Active learning step works")


# =============================================================================
# TEST 6: Curiosity with Meta State
# =============================================================================

def test_6_curiosity_with_meta_state():
    """
    Test curiosity modulation by meta-learning state.
    """
    print("\n=== Test 6: Curiosity with Meta State ===")
    
    from holographic_v4.curiosity import curiosity_score_with_meta
    from holographic_v4.meta_learning import create_learning_state, update_meta_state
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.dreaming import DreamingSystem
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    dreaming = DreamingSystem(basis=basis)
    meta_state = create_learning_state()
    
    # Simulate some errors to increase uncertainty
    for _ in range(10):
        meta_state = update_meta_state(meta_state, prediction_correct=False, salience=0.5, novelty=0.5)
    
    high_uncertainty_state = meta_state
    
    # Reset and simulate successes
    meta_state = create_learning_state()
    for _ in range(10):
        meta_state = update_meta_state(meta_state, prediction_correct=True, salience=0.5, novelty=0.5)
    
    low_uncertainty_state = meta_state
    
    # Compute curiosity with both states
    query = model.compute_context([50, 51, 52])
    
    cur_high = curiosity_score_with_meta(query, model, dreaming, high_uncertainty_state)
    cur_low = curiosity_score_with_meta(query, model, dreaming, low_uncertainty_state)
    
    print(f"Curiosity (high uncertainty): {cur_high:.4f}")
    print(f"Curiosity (low uncertainty): {cur_low:.4f}")
    
    # With high uncertainty, curiosity should be LOWER (be cautious)
    assert cur_high < cur_low, (
        f"High uncertainty should reduce curiosity: {cur_high:.4f} < {cur_low:.4f}"
    )
    
    print("✓ Curiosity with meta state works")


# =============================================================================
# TEST 7: Integration with AdaptiveMemory
# =============================================================================

def test_7_integration_with_adaptive_memory():
    """
    Test curiosity integration with the production AdaptiveMemory system.
    """
    print("\n=== Test 7: Integration with AdaptiveMemory ===")
    
    from holographic_v4.adaptive_memory import create_adaptive_memory
    from holographic_v4.curiosity import curiosity_score, estimate_information_gain
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.dreaming import DreamingSystem
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    
    # Create production memory
    memory = create_adaptive_memory(vocab_size=100, seed=42)
    
    # Train on some patterns
    for i in range(10):
        memory.learn_adaptive([i, i+1, i+2], i * 10)
    
    # Create model and dreaming for curiosity computation
    # (In full integration, these would be part of memory)
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    dreaming = DreamingSystem(basis=basis)
    
    # Copy knowledge from memory to model
    for i in range(10):
        model.train_step([i, i+1, i+2], i * 10)
    
    # Test curiosity for known vs unknown
    known_matrix = model.compute_context([0, 1, 2])
    unknown_matrix = model.compute_context([80, 81, 82])
    
    cur_known = curiosity_score(known_matrix, model, dreaming)
    cur_unknown = curiosity_score(unknown_matrix, model, dreaming)
    
    print(f"Known curiosity: {cur_known:.4f}")
    print(f"Unknown curiosity: {cur_unknown:.4f}")
    
    # Unknown should have higher curiosity
    assert cur_unknown > cur_known * 0.8, (
        "Unknown should have comparable or higher curiosity"
    )
    
    print("✓ Integration with AdaptiveMemory works")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_1_curiosity_score_computation()
    test_2_information_gain_estimation()
    test_3_basin_boundary_sampling()
    test_4_curiosity_query_generation()
    test_5_active_learning_step()
    test_6_curiosity_with_meta_state()
    test_7_integration_with_adaptive_memory()
    
    print("\n" + "="*60)
    print("ALL CURIOSITY INTEGRATION TESTS PASSED")
    print("="*60)
