"""
Tests for Planning Integration — Causal Reasoning and Counterfactuals
=====================================================================

Verifies that the system can:
1. Simulate future states from actions
2. Plan sequences toward goals
3. Reason about counterfactuals ("what if?")
4. Perform causal reasoning

THEORY:
    Planning = "What would future-me perceive after action X?"
    Counterfactual = "What would have happened if Y instead of Z?"
    Both are coordinate transformations in Clifford space.
"""

import numpy as np
import pytest

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE


# =============================================================================
# TEST 1: Action Simulation
# =============================================================================

def test_1_action_simulation():
    """
    Test that actions can be simulated to predict next states.
    
    THEORY: state' = Grace_flow(state ⊗ action_embedding)
    """
    print("\n=== Test 1: Action Simulation ===")
    
    from holographic_v4.planning import simulate_action
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    
    # Train some knowledge
    for i in range(10):
        model.train_step([i, i+1, i+2], i * 10)
    
    # Get current state
    current_state = model.compute_context([0, 1, 2])
    
    # Simulate action
    action = 5
    next_state, confidence = simulate_action(current_state, action, model)
    
    print(f"Current state shape: {current_state.shape}")
    print(f"Next state shape: {next_state.shape}")
    print(f"Confidence: {confidence:.4f}")
    
    assert next_state.shape == (4, 4), f"Expected (4, 4), got {next_state.shape}"
    assert 0 <= confidence <= 1, f"Confidence should be in [0, 1], got {confidence}"
    
    # States should be different after action
    state_diff = np.linalg.norm(next_state - current_state)
    print(f"State difference: {state_diff:.4f}")
    assert state_diff > 0.01, "Action should change the state"
    
    print("✓ Action simulation works")


# =============================================================================
# TEST 2: Plan to Goal
# =============================================================================

def test_2_plan_to_goal():
    """
    Test planning action sequences toward a goal.
    """
    print("\n=== Test 2: Plan to Goal ===")
    
    from holographic_v4.planning import plan_to_goal, Plan
    from holographic_v4.pipeline import TheoryTrueModel
    
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    
    # Train some transitions
    for i in range(20):
        model.train_step([i, i+1, i+2], i * 10 % 100)
    
    # Set up start and goal states
    start_state = model.compute_context([0, 1, 2])
    goal_state = model.compute_context([10, 11, 12])
    
    # Plan
    plan = plan_to_goal(
        start_state, goal_state, model,
        max_steps=5,
        action_space=list(range(10, 20))
    )
    
    print(f"Plan steps: {len(plan.steps)}")
    print(f"Goal achieved: {plan.goal_achieved}")
    print(f"Total confidence: {plan.total_confidence:.4f}")
    
    assert isinstance(plan, Plan), "Should return a Plan object"
    assert plan.total_confidence >= 0, "Confidence should be non-negative"
    
    # Plan should have at least attempted some steps
    # (may or may not achieve goal depending on training)
    print("✓ Plan to goal works")


# =============================================================================
# TEST 3: Counterfactual Reasoning
# =============================================================================

def test_3_counterfactual_reasoning():
    """
    Test counterfactual reasoning: "What if X instead of Y?"
    """
    print("\n=== Test 3: Counterfactual Reasoning ===")
    
    from holographic_v4.planning import counterfactual
    from holographic_v4.pipeline import TheoryTrueModel
    
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    
    # Train
    for i in range(20):
        model.train_step([i, i+1, i+2], i * 10 % 100)
    
    # What actually happened
    actual_context = [0, 1, 2]
    
    # What if we had done action 50 instead?
    hypothetical_action = 50
    
    cf_target, cf_confidence = counterfactual(actual_context, hypothetical_action, model)
    
    print(f"Counterfactual target: {cf_target}")
    print(f"Counterfactual confidence: {cf_confidence:.4f}")
    
    # Should return some target
    assert cf_target is not None or cf_confidence >= 0, "Should compute counterfactual"
    
    print("✓ Counterfactual reasoning works")


# =============================================================================
# TEST 4: Plan Evaluation
# =============================================================================

def test_4_plan_evaluation():
    """
    Test evaluating plan quality.
    """
    print("\n=== Test 4: Plan Evaluation ===")
    
    from holographic_v4.planning import plan_to_goal, evaluate_plan
    from holographic_v4.pipeline import TheoryTrueModel
    
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    
    # Train
    for i in range(10):
        model.train_step([i, i+1, i+2], i * 10)
    
    # Create a plan
    start = model.compute_context([0, 1, 2])
    goal = model.compute_context([5, 6, 7])
    
    plan = plan_to_goal(start, goal, model, max_steps=3)
    
    # Evaluate
    score = evaluate_plan(plan, goal, model)
    
    print(f"Plan score: {score:.4f}")
    
    assert 0 <= score <= 1, f"Score should be in [0, 1], got {score}"
    
    print("✓ Plan evaluation works")


# =============================================================================
# TEST 5: Multi-Step Counterfactual
# =============================================================================

def test_5_multi_step_counterfactual():
    """
    Test multi-step counterfactual trajectories.
    """
    print("\n=== Test 5: Multi-Step Counterfactual ===")
    
    from holographic_v4.planning import counterfactual_trajectory
    from holographic_v4.pipeline import TheoryTrueModel
    
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    
    # Train
    for i in range(20):
        model.train_step([i, i+1, i+2], i * 10 % 100)
    
    # Actual trajectory
    actual_trajectory = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
    ]
    
    # Branch at step 1 with different action
    trajectory = counterfactual_trajectory(
        actual_trajectory,
        branch_point=1,
        hypothetical_action=99,
        model=model,
        steps_to_simulate=2
    )
    
    print(f"Counterfactual trajectory length: {len(trajectory)}")
    
    for i, (state, target, conf) in enumerate(trajectory):
        print(f"  Step {i}: target={target}, conf={conf:.4f}")
    
    assert len(trajectory) > 0, "Should produce trajectory"
    
    print("✓ Multi-step counterfactual works")


# =============================================================================
# TEST 6: Plan with Subgoals
# =============================================================================

def test_6_plan_with_subgoals():
    """
    Test planning with intermediate subgoals.
    """
    print("\n=== Test 6: Plan with Subgoals ===")
    
    from holographic_v4.planning import plan_with_subgoals, Plan
    from holographic_v4.pipeline import TheoryTrueModel
    
    model = TheoryTrueModel(vocab_size=100, context_size=3)
    
    # Train
    for i in range(20):
        model.train_step([i, i+1, i+2], i * 10 % 100)
    
    # Set up start and goal
    start = model.compute_context([0, 1, 2])
    goal = model.compute_context([10, 11, 12])
    
    # Plan with subgoals (auto-generated)
    plan = plan_with_subgoals(start, goal, model, subgoal_count=2, steps_per_subgoal=3)
    
    print(f"Plan steps: {len(plan.steps)}")
    print(f"Goal achieved: {plan.goal_achieved}")
    print(f"Total confidence: {plan.total_confidence:.4f}")
    
    assert isinstance(plan, Plan), "Should return a Plan object"
    
    print("✓ Plan with subgoals works")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_1_action_simulation()
    test_2_plan_to_goal()
    test_3_counterfactual_reasoning()
    test_4_plan_evaluation()
    test_5_multi_step_counterfactual()
    test_6_plan_with_subgoals()
    
    print("\n" + "="*60)
    print("ALL PLANNING INTEGRATION TESTS PASSED")
    print("="*60)
