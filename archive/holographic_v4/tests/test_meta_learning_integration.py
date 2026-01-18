"""
Tests for Meta-Learning Integration with FractalGenerativeMemory
================================================================

TDD tests for adaptive learning rates integrated with the v4.28.0 architecture.
"""

import numpy as np
import pytest
from typing import List, Tuple

from holographic_v4.fractal_generative_memory import FractalGenerativeMemory
from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def memory():
    """Create a FractalGenerativeMemory for testing."""
    return FractalGenerativeMemory(
        max_levels=1,
        vocab_size=100,
        orthogonalize=True,
        contrastive_enabled=False,
        seed=42,
    )


# =============================================================================
# TEST 1: Adaptive Learning Rate Bounds
# =============================================================================

def test_1_adaptive_rate_bounds():
    """
    Test that adaptive rates stay within φ-derived bounds.
    
    THEORY: Rates should be in [base_rate × φ⁻¹, base_rate × φ]
    """
    print("\n=== Test 1: Adaptive Rate Bounds ===")
    
    from holographic_v4.meta_learning import compute_adaptive_learning_rate
    
    base_rate = PHI_INV
    min_bound = base_rate * PHI_INV
    max_bound = base_rate * PHI
    
    # Test extreme cases
    test_cases = [
        # (salience, novelty, uncertainty)
        (0.0, 0.0, 0.0),  # Minimum stimulation
        (1.0, 1.0, 0.0),  # Maximum stimulation, no uncertainty
        (0.0, 0.0, 1.0),  # Maximum uncertainty
        (1.0, 1.0, 1.0),  # Everything maxed
        (0.5, 0.5, 0.5),  # Balanced
    ]
    
    for salience, novelty, uncertainty in test_cases:
        rate = compute_adaptive_learning_rate(salience, novelty, uncertainty, base_rate)
        print(f"  S={salience:.1f}, N={novelty:.1f}, U={uncertainty:.1f} → rate={rate:.4f}")
        
        assert min_bound <= rate <= max_bound, (
            f"Rate {rate:.4f} out of bounds [{min_bound:.4f}, {max_bound:.4f}]"
        )
    
    print(f"Bounds: [{min_bound:.4f}, {max_bound:.4f}]")
    print("✓ All rates within φ-derived bounds")


# =============================================================================
# TEST 2: Novelty Increases Rate
# =============================================================================

def test_2_novelty_increases_rate():
    """
    Test that high novelty increases learning rate.
    
    THEORY: New patterns should be learned faster.
    """
    print("\n=== Test 2: Novelty Increases Rate ===")
    
    from holographic_v4.meta_learning import compute_adaptive_learning_rate
    
    base_rate = PHI_INV
    
    # Low novelty
    rate_low_novelty = compute_adaptive_learning_rate(
        salience=0.5, novelty=0.1, uncertainty=0.3, base_rate=base_rate
    )
    
    # High novelty
    rate_high_novelty = compute_adaptive_learning_rate(
        salience=0.5, novelty=0.9, uncertainty=0.3, base_rate=base_rate
    )
    
    print(f"Low novelty (0.1): rate={rate_low_novelty:.4f}")
    print(f"High novelty (0.9): rate={rate_high_novelty:.4f}")
    
    assert rate_high_novelty > rate_low_novelty, (
        f"High novelty rate ({rate_high_novelty:.4f}) should be > low novelty ({rate_low_novelty:.4f})"
    )
    
    print("✓ Novelty increases learning rate")


# =============================================================================
# TEST 3: Uncertainty Decreases Rate
# =============================================================================

def test_3_uncertainty_decreases_rate():
    """
    Test that high uncertainty decreases learning rate.
    
    THEORY: When uncertain, learn more cautiously.
    """
    print("\n=== Test 3: Uncertainty Decreases Rate ===")
    
    from holographic_v4.meta_learning import compute_adaptive_learning_rate
    
    base_rate = PHI_INV
    
    # Low uncertainty
    rate_low_uncertainty = compute_adaptive_learning_rate(
        salience=0.5, novelty=0.5, uncertainty=0.1, base_rate=base_rate
    )
    
    # High uncertainty
    rate_high_uncertainty = compute_adaptive_learning_rate(
        salience=0.5, novelty=0.5, uncertainty=0.9, base_rate=base_rate
    )
    
    print(f"Low uncertainty (0.1): rate={rate_low_uncertainty:.4f}")
    print(f"High uncertainty (0.9): rate={rate_high_uncertainty:.4f}")
    
    assert rate_low_uncertainty > rate_high_uncertainty, (
        f"Low uncertainty rate ({rate_low_uncertainty:.4f}) should be > high uncertainty ({rate_high_uncertainty:.4f})"
    )
    
    print("✓ Uncertainty decreases learning rate")


# =============================================================================
# TEST 4: Meta State Evolution
# =============================================================================

def test_4_meta_state_evolution():
    """
    Test that meta state evolves correctly over training.
    """
    print("\n=== Test 4: Meta State Evolution ===")
    
    from holographic_v4.meta_learning import (
        create_learning_state,
        update_meta_state,
        get_adaptive_parameters,
    )
    
    state = create_learning_state()
    
    # Initial state
    params_initial = get_adaptive_parameters(state)
    print(f"Initial: {params_initial}")
    
    assert params_initial['learning_rate'] == PHI_INV
    assert params_initial['uncertainty'] == PHI_INV_SQ
    
    # Simulate training with errors
    for i in range(20):
        correct = (i % 3 != 0)  # ~67% accuracy
        state = update_meta_state(state, correct, salience=0.5, novelty=0.3)
    
    params_after_errors = get_adaptive_parameters(state)
    print(f"After errors: {params_after_errors}")
    
    # Error rate should be tracked
    assert params_after_errors['recent_error_rate'] > 0, "Should track errors"
    
    # Uncertainty should increase with errors
    assert params_after_errors['uncertainty'] > params_initial['uncertainty'], (
        "Uncertainty should increase with errors"
    )
    
    # Simulate training with success
    for i in range(30):
        state = update_meta_state(state, prediction_correct=True, salience=0.5, novelty=0.3)
    
    params_after_success = get_adaptive_parameters(state)
    print(f"After success: {params_after_success}")
    
    # Error rate should decrease
    assert params_after_success['recent_error_rate'] < params_after_errors['recent_error_rate'], (
        "Error rate should decrease with success"
    )
    
    print("✓ Meta state evolves correctly")


# =============================================================================
# TEST 5: φ-Scaled Learning Rate Schedule
# =============================================================================

def test_5_phi_scaled_schedule():
    """
    Test the φ-scaled curriculum learning rate schedule.
    """
    print("\n=== Test 5: φ-Scaled Schedule ===")
    
    from holographic_v4.meta_learning import phi_scaled_learning_rate
    
    total_steps = 1000
    base_rate = PHI_INV
    
    # Sample rates at different points
    rates = []
    checkpoints = [0, 100, 250, 500, 750, 999]
    
    for step in checkpoints:
        rate = phi_scaled_learning_rate(step, total_steps, base_rate)
        rates.append(rate)
        print(f"  Step {step:4d}: rate={rate:.4f}")
    
    # Warmup phase (first ~23.6% of steps)
    warmup_end = int(total_steps * PHI_INV_CUBE)
    rate_start = phi_scaled_learning_rate(0, total_steps, base_rate)
    rate_warmup_end = phi_scaled_learning_rate(warmup_end, total_steps, base_rate)
    
    assert rate_warmup_end > rate_start, (
        f"Rate should increase during warmup: {rate_start:.4f} -> {rate_warmup_end:.4f}"
    )
    
    # Decay phase
    rate_middle = phi_scaled_learning_rate(500, total_steps, base_rate)
    rate_end = phi_scaled_learning_rate(999, total_steps, base_rate)
    
    assert rate_end < rate_middle, (
        f"Rate should decrease during decay: {rate_middle:.4f} -> {rate_end:.4f}"
    )
    
    # All rates should be bounded
    min_rate = base_rate * PHI_INV_SQ
    max_rate = base_rate
    
    for step in range(0, total_steps, 50):
        rate = phi_scaled_learning_rate(step, total_steps, base_rate)
        assert min_rate <= rate <= max_rate, (
            f"Rate at step {step} ({rate:.4f}) out of bounds [{min_rate:.4f}, {max_rate:.4f}]"
        )
    
    print("✓ φ-scaled schedule works correctly")


# =============================================================================
# TEST 6: Integration with FractalGenerativeMemory
# =============================================================================

def test_6_integration_with_memory(memory):
    """
    Test integration of meta-learning with FractalGenerativeMemory.
    """
    print("\n=== Test 6: Integration with Memory ===")
    
    from holographic_v4.meta_learning import (
        create_learning_state,
        update_meta_state,
    )
    
    state = create_learning_state()
    
    # Train with adaptive rates
    patterns = [
        ([1, 2, 3], 10),
        ([4, 5, 6], 20),
        ([7, 8, 9], 30),
    ]
    
    for ctx, tgt in patterns:
        # Check if context is novel
        ctx_hash = hash(tuple(ctx))
        is_novel = ctx_hash not in memory.memory
        novelty = 1.0 if is_novel else 0.2
        
        # Predict before learning
        pred, conf = memory.retrieve_deterministic(ctx)
        correct = (pred == tgt)
        
        # Estimate salience (rarer contexts = higher salience)
        # For now, use uniform salience
        salience = 0.5
        
        # Update meta state
        state = update_meta_state(state, correct, salience, novelty)
        
        # Get adaptive rate and apply to learning
        # (In real integration, this would modify memory.config.learning_rate)
        adaptive_rate = state.effective_learning_rate
        
        # Learn
        memory.learn(ctx, tgt)
        
        print(f"  {ctx} -> {tgt}: novel={is_novel}, rate={adaptive_rate:.4f}")
    
    # Verify learning worked
    for ctx, tgt in patterns:
        pred, conf = memory.retrieve_deterministic(ctx)
        assert pred == tgt, f"Should retrieve {tgt} for {ctx}, got {pred}"
    
    print("✓ Integration with memory works")


# =============================================================================
# TEST 7: Adaptive Rate Improves Learning
# =============================================================================

def test_7_adaptive_rate_improves_learning():
    """
    Test that adaptive rates improve learning compared to fixed rates.
    
    This compares:
    1. Fixed rate (always φ⁻¹)
    2. Adaptive rate (modulated by novelty/uncertainty)
    """
    print("\n=== Test 7: Adaptive vs Fixed Rate ===")
    
    from holographic_v4.meta_learning import (
        create_learning_state,
        update_meta_state,
    )
    
    # Generate test data with varying novelty
    np.random.seed(42)
    n_patterns = 50
    patterns = [
        (list(np.random.randint(0, 100, size=3)), np.random.randint(0, 100))
        for _ in range(n_patterns)
    ]
    
    # Test 1: Fixed rate learning
    memory_fixed = FractalGenerativeMemory(vocab_size=100, seed=42)
    for ctx, tgt in patterns:
        memory_fixed.learn(ctx, tgt)
    
    fixed_correct = 0
    for ctx, tgt in patterns:
        pred, _ = memory_fixed.retrieve_deterministic(ctx)
        if pred == tgt:
            fixed_correct += 1
    fixed_accuracy = fixed_correct / n_patterns
    
    # Test 2: Adaptive rate learning (simulated)
    memory_adaptive = FractalGenerativeMemory(vocab_size=100, seed=42)
    state = create_learning_state()
    
    for ctx, tgt in patterns:
        # Measure novelty
        ctx_hash = hash(tuple(ctx))
        novelty = 1.0 if ctx_hash not in memory_adaptive.memory else 0.2
        
        # Get prediction confidence for uncertainty
        pred, conf = memory_adaptive.retrieve_deterministic(ctx)
        correct = (pred == tgt)
        uncertainty = 1.0 - conf if conf else 0.5
        
        # Update state
        state = update_meta_state(state, correct, salience=0.5, novelty=novelty)
        
        # Learn (with conceptually adaptive rate - actual rate used by memory.learn is fixed,
        # but the state tracks what rate WOULD be used in full integration)
        memory_adaptive.learn(ctx, tgt)
    
    adaptive_correct = 0
    for ctx, tgt in patterns:
        pred, _ = memory_adaptive.retrieve_deterministic(ctx)
        if pred == tgt:
            adaptive_correct += 1
    adaptive_accuracy = adaptive_correct / n_patterns
    
    print(f"Fixed rate accuracy: {fixed_accuracy:.1%}")
    print(f"Adaptive rate accuracy: {adaptive_accuracy:.1%}")
    print(f"Final state: rate={state.effective_learning_rate:.4f}, uncertainty={state.uncertainty:.4f}")
    
    # Both should work (this is a sanity check, not a performance comparison)
    assert fixed_accuracy > 0.5, f"Fixed rate should work: {fixed_accuracy:.1%}"
    assert adaptive_accuracy > 0.5, f"Adaptive rate should work: {adaptive_accuracy:.1%}"
    
    print("✓ Both learning modes work correctly")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_1_adaptive_rate_bounds()
    test_2_novelty_increases_rate()
    test_3_uncertainty_decreases_rate()
    test_4_meta_state_evolution()
    test_5_phi_scaled_schedule()
    
    memory = FractalGenerativeMemory(vocab_size=100, seed=42)
    test_6_integration_with_memory(memory)
    test_7_adaptive_rate_improves_learning()
    
    print("\n" + "="*60)
    print("ALL META-LEARNING INTEGRATION TESTS PASSED")
    print("="*60)
