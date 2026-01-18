"""
Tests for Credit Assignment v2
==============================

Verifies theory-true credit assignment with FractalGenerativeMemory.
"""

import numpy as np
import pytest
from typing import List, Tuple

# Import the modules under test
from holographic_v4.fractal_generative_memory import FractalGenerativeMemory
from holographic_v4.credit_assignment import (
    CreditAssignmentTracker,
    ReconsolidationConfig,
    ErrorRecord,
    create_credit_assigned_learn,
    batch_credit_assignment,
)
from holographic_v4.constants import PHI_INV, PHI_INV_SQ, PHI_INV_CUBE


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
        contrastive_enabled=False,  # Disable to isolate credit assignment
        seed=42,
    )


@pytest.fixture
def tracker(memory):
    """Create a CreditAssignmentTracker."""
    return CreditAssignmentTracker(memory)


# =============================================================================
# TEST 1: Error Recording
# =============================================================================

def test_1_error_recording(tracker):
    """
    Test that errors are recorded correctly.
    """
    print("\n=== Test 1: Error Recording ===")
    
    # Record an error
    tracker.record_error(
        context=[1, 2, 3],
        predicted=5,
        actual=7,
        confidence=0.8,
    )
    
    assert len(tracker.errors) == 1
    assert tracker.total_errors == 1
    
    error = tracker.errors[0]
    assert error.predicted == 5
    assert error.actual == 7
    assert error.confidence == 0.8
    
    # Error magnitude should be φ-derived
    mag = error.error_magnitude
    print(f"Error magnitude: {mag:.4f}")
    assert PHI_INV_CUBE < mag < 1.0, f"Magnitude should be φ-bounded, got {mag}"
    
    # Recording correct prediction shouldn't add error
    tracker.record_error(
        context=[4, 5, 6],
        predicted=10,
        actual=10,  # Correct!
        confidence=0.9,
    )
    
    assert len(tracker.errors) == 1, "Correct predictions shouldn't be recorded"
    assert tracker.total_errors == 1
    
    print("✓ Error recording works correctly")


# =============================================================================
# TEST 2: Reconsolidation Improves Accuracy
# =============================================================================

def test_2_reconsolidation_improves_accuracy(memory):
    """
    Test that reconsolidation improves the ranking of correct targets.
    
    THEORY: Reconsolidation should boost correct and attenuate wrong,
    improving the rank of the correct target even if it doesn't
    immediately become the top prediction.
    """
    print("\n=== Test 2: Reconsolidation Improves Accuracy ===")
    
    # Train on a pattern with wrong target
    context = [1, 2, 3]
    wrong_target = 50
    correct_target = 75
    
    # Learn wrong target a few times
    for _ in range(3):
        memory.learn(context, wrong_target)
    
    # Get baseline ranking of correct target
    scores_before = memory._compute_target_scores(context)
    correct_rank_before = next(i for i, (t, s) in enumerate(scores_before) if t == correct_target)
    correct_score_before = next(s for t, s in scores_before if t == correct_target)
    
    print(f"Before reconsolidation:")
    print(f"  Correct target rank: {correct_rank_before}, score: {correct_score_before:.4f}")
    
    # Apply reconsolidation multiple times (simulating error correction over time)
    for _ in range(5):
        errors = [(context, wrong_target, correct_target, 0.8)]
        batch_credit_assignment(memory, errors)
    
    # Get ranking after reconsolidation
    scores_after = memory._compute_target_scores(context)
    correct_rank_after = next(i for i, (t, s) in enumerate(scores_after) if t == correct_target)
    correct_score_after = next(s for t, s in scores_after if t == correct_target)
    wrong_score_after = next(s for t, s in scores_after if t == wrong_target)
    
    print(f"After reconsolidation:")
    print(f"  Correct target rank: {correct_rank_after}, score: {correct_score_after:.4f}")
    print(f"  Wrong target score: {wrong_score_after:.4f}")
    
    # Correct target rank should improve (lower is better)
    assert correct_rank_after < correct_rank_before, (
        f"Correct target rank should improve: {correct_rank_before} -> {correct_rank_after}"
    )
    
    # Correct target score should increase
    assert correct_score_after > correct_score_before, (
        f"Correct target score should increase: {correct_score_before:.4f} -> {correct_score_after:.4f}"
    )
    
    print(f"Rank improvement: {correct_rank_before} -> {correct_rank_after}")
    print(f"Score improvement: {correct_score_before:.4f} -> {correct_score_after:.4f}")
    print("✓ Reconsolidation improves accuracy")


# =============================================================================
# TEST 3: Integrated Learning with Credit Assignment
# =============================================================================

def test_3_integrated_learning(memory):
    """
    Test the integrated learn function with automatic credit assignment.
    """
    print("\n=== Test 3: Integrated Learning ===")
    
    tracker, learn_ca = create_credit_assigned_learn(memory)
    
    # Train on some patterns
    patterns = [
        ([1, 2, 3], 10),
        ([4, 5, 6], 20),
        ([7, 8, 9], 30),
    ]
    
    # First pass: learn all patterns
    for ctx, tgt in patterns:
        stats = learn_ca(ctx, tgt)
        print(f"Learn {ctx} -> {tgt}: {stats['correct']}, recon={stats['reconsolidated']}")
    
    # Second pass: should be correct now
    n_correct = 0
    for ctx, tgt in patterns:
        stats = learn_ca(ctx, tgt)
        if stats['correct']:
            n_correct += 1
    
    accuracy = n_correct / len(patterns)
    print(f"Second pass accuracy: {accuracy:.1%}")
    
    # Should have high accuracy on known patterns
    assert accuracy >= 0.66, f"Should have >66% accuracy, got {accuracy:.1%}"
    
    print(f"Tracker stats: {tracker.get_statistics()}")
    print("✓ Integrated learning works correctly")


# =============================================================================
# TEST 4: Batch Credit Assignment
# =============================================================================

def test_4_batch_credit_assignment(memory):
    """
    Test batch processing of errors.
    """
    print("\n=== Test 4: Batch Credit Assignment ===")
    
    # Learn several patterns
    for i in range(10):
        context = [i, i+1, i+2]
        target = i * 10
        memory.learn(context, target)
    
    # Simulate errors: wrong predictions
    errors = [
        ([0, 1, 2], 0, 5, 0.7),   # predicted 0, actual 5
        ([1, 2, 3], 10, 15, 0.8),
        ([2, 3, 4], 20, 25, 0.6),
    ]
    
    stats = batch_credit_assignment(memory, errors)
    print(f"Batch stats: {stats}")
    
    assert stats['processed'] == 3, f"Should process 3 errors, got {stats['processed']}"
    assert stats['total_boost'] > 0, "Should have positive boost"
    assert stats['total_attenuate'] > 0, "Should have positive attenuation"
    
    # Verify boost > attenuate (asymmetry)
    assert stats['avg_boost'] > stats['avg_attenuate'], (
        f"Boost ({stats['avg_boost']:.4f}) should be > attenuate ({stats['avg_attenuate']:.4f})"
    )
    
    print("✓ Batch credit assignment works correctly")


# =============================================================================
# TEST 5: Rolling Window Bounds Memory
# =============================================================================

def test_5_rolling_window():
    """
    Test that error history is bounded by max_history.
    """
    print("\n=== Test 5: Rolling Window ===")
    
    memory = FractalGenerativeMemory(vocab_size=100, seed=42)
    config = ReconsolidationConfig(max_history=10)
    tracker = CreditAssignmentTracker(memory, config)
    
    # Record more errors than max_history
    for i in range(25):
        tracker.record_error(
            context=[i, i+1],
            predicted=i,
            actual=i + 50,
            confidence=0.5,
        )
    
    # Should be capped at max_history
    assert len(tracker.errors) == 10, f"Should have 10 errors, got {len(tracker.errors)}"
    assert tracker.total_errors == 25, f"Total should be 25, got {tracker.total_errors}"
    
    # Oldest errors should be dropped
    oldest_error = tracker.errors[0]
    assert oldest_error.timestamp == 15, f"Oldest should be step 15, got {oldest_error.timestamp}"
    
    print(f"Errors in window: {len(tracker.errors)}")
    print(f"Total errors recorded: {tracker.total_errors}")
    print("✓ Rolling window bounds memory correctly")


# =============================================================================
# TEST 6: φ-Derived Rates
# =============================================================================

def test_6_phi_derived_rates():
    """
    Test that all rates are φ-derived.
    """
    print("\n=== Test 6: φ-Derived Rates ===")
    
    config = ReconsolidationConfig()
    
    # Check boost rate
    expected_boost = PHI_INV_SQ  # φ⁻² ≈ 0.382
    assert abs(config.boost_rate - expected_boost) < 1e-6, (
        f"Boost rate should be φ⁻² ({expected_boost:.6f}), got {config.boost_rate:.6f}"
    )
    
    # Check attenuate rate
    expected_attenuate = PHI_INV_CUBE  # φ⁻³ ≈ 0.236
    assert abs(config.attenuate_rate - expected_attenuate) < 1e-6, (
        f"Attenuate rate should be φ⁻³ ({expected_attenuate:.6f}), got {config.attenuate_rate:.6f}"
    )
    
    # Verify asymmetry: boost > attenuate
    assert config.boost_rate > config.attenuate_rate, (
        "Boost rate should be > attenuate rate (asymmetric learning)"
    )
    
    # Check error magnitude formula
    error = ErrorRecord(
        ctx_hash=12345,
        context=(1, 2, 3),
        predicted=5,
        actual=10,
        confidence=1.0,  # Max confidence
    )
    
    mag = error.error_magnitude
    expected_max_mag = PHI_INV * (PHI_INV + 1.0 * (1 - PHI_INV))
    assert abs(mag - expected_max_mag) < 1e-6, (
        f"Max error magnitude should be {expected_max_mag:.6f}, got {mag:.6f}"
    )
    
    print(f"Boost rate: {config.boost_rate:.6f} (φ⁻²)")
    print(f"Attenuate rate: {config.attenuate_rate:.6f} (φ⁻³)")
    print(f"Max error magnitude: {mag:.6f}")
    print("✓ All rates are φ-derived")


# =============================================================================
# TEST 7: No Reconsolidation Without Errors
# =============================================================================

def test_7_no_reconsolidation_without_errors(memory):
    """
    Test that reconsolidation is a no-op when there are no errors.
    """
    print("\n=== Test 7: No Reconsolidation Without Errors ===")
    
    tracker = CreditAssignmentTracker(memory)
    
    # Try to reconsolidate with no errors
    stats = tracker.reconsolidate(force=True)
    
    assert stats['processed'] == 0
    assert 'skipped' in stats
    
    print(f"Stats: {stats}")
    print("✓ No reconsolidation without errors")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run all tests
    memory = FractalGenerativeMemory(vocab_size=100, seed=42)
    tracker = CreditAssignmentTracker(memory)
    
    test_1_error_recording(tracker)
    
    memory2 = FractalGenerativeMemory(vocab_size=100, seed=42)
    test_2_reconsolidation_improves_accuracy(memory2)
    
    memory3 = FractalGenerativeMemory(vocab_size=100, seed=42)
    test_3_integrated_learning(memory3)
    
    memory4 = FractalGenerativeMemory(vocab_size=100, seed=42)
    test_4_batch_credit_assignment(memory4)
    
    test_5_rolling_window()
    test_6_phi_derived_rates()
    
    memory5 = FractalGenerativeMemory(vocab_size=100, seed=42)
    test_7_no_reconsolidation_without_errors(memory5)
    
    print("\n" + "="*60)
    print("ALL CREDIT ASSIGNMENT v2 TESTS PASSED")
    print("="*60)
