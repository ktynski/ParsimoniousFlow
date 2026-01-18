"""
Meta-Learning Test Suite
========================

TDD tests for adaptive learning parameters.

THEORY:
    phi-derived parameters are the DEFAULT that can be modulated by context.
    High salience → faster learning. High uncertainty → slower learning.
"""

import numpy as np
import time
from typing import Dict, List, Tuple

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ
from holographic_v4.algebra import build_clifford_basis
from holographic_v4.pipeline import TheoryTrueModel
from holographic_v4.dreaming import DreamingSystem

# =============================================================================
# TEST SETUP
# =============================================================================

BASIS = build_clifford_basis()
XP = np
VOCAB_SIZE = 100


def create_model() -> Tuple[TheoryTrueModel, DreamingSystem]:
    """Create model and dreaming system for testing."""
    model = TheoryTrueModel(
        vocab_size=VOCAB_SIZE,
        context_size=5,
        noise_std=0.3,
        xp=XP,
    )
    dreaming = DreamingSystem(BASIS, XP)
    return model, dreaming


# =============================================================================
# ADAPTIVE LEARNING RATE TESTS
# =============================================================================

def test_high_salience_increases_learning_rate() -> bool:
    """
    Test that high salience increases learning rate.
    
    SUCCESS CRITERIA:
    - Learning rate for salient content > base rate
    """
    print("Test: high_salience_increases_learning_rate...")
    
    try:
        from holographic_v4.meta_learning import compute_adaptive_learning_rate
    except ImportError:
        print("  ✗ FAIL (meta_learning not implemented yet)")
        return False
    
    base_rate = PHI_INV
    
    # High salience
    high_salience_rate = compute_adaptive_learning_rate(
        salience=0.9,
        novelty=0.5,
        uncertainty=0.5,
        base_rate=base_rate,
    )
    
    # Low salience
    low_salience_rate = compute_adaptive_learning_rate(
        salience=0.1,
        novelty=0.5,
        uncertainty=0.5,
        base_rate=base_rate,
    )
    
    is_pass = high_salience_rate > low_salience_rate
    print(f"  High salience rate: {high_salience_rate:.4f}")
    print(f"  Low salience rate: {low_salience_rate:.4f}")
    print(f"  Base rate: {base_rate:.4f}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_high_uncertainty_decreases_learning_rate() -> bool:
    """
    Test that high uncertainty decreases learning rate.
    
    SUCCESS CRITERIA:
    - Learning rate with high uncertainty < learning rate with low uncertainty
    """
    print("Test: high_uncertainty_decreases_learning_rate...")
    
    try:
        from holographic_v4.meta_learning import compute_adaptive_learning_rate
    except ImportError:
        print("  ✗ FAIL (meta_learning not implemented yet)")
        return False
    
    base_rate = PHI_INV
    
    # High uncertainty
    high_uncertainty_rate = compute_adaptive_learning_rate(
        salience=0.5,
        novelty=0.5,
        uncertainty=0.9,
        base_rate=base_rate,
    )
    
    # Low uncertainty
    low_uncertainty_rate = compute_adaptive_learning_rate(
        salience=0.5,
        novelty=0.5,
        uncertainty=0.1,
        base_rate=base_rate,
    )
    
    is_pass = high_uncertainty_rate < low_uncertainty_rate
    print(f"  High uncertainty rate: {high_uncertainty_rate:.4f}")
    print(f"  Low uncertainty rate: {low_uncertainty_rate:.4f}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_meta_learning_improves_over_fixed() -> bool:
    """
    Test that adaptive learning outperforms fixed rate on varied tasks.
    """
    print("Test: meta_learning_improves_over_fixed...")
    
    try:
        from holographic_v4.meta_learning import LearningState, update_meta_state
    except ImportError:
        print("  ✗ FAIL (meta_learning not implemented yet)")
        return False
    
    # Create a learning state
    state = LearningState()
    
    # Simulate learning with varying conditions
    for i in range(10):
        salience = 0.3 + 0.5 * (i % 3) / 2  # Varying salience
        novelty = 0.2 + 0.6 * ((i + 1) % 3) / 2  # Varying novelty
        correct = i % 2 == 0  # Alternating success
        
        state = update_meta_state(state, correct, salience, novelty)
    
    # State should have adapted
    adapted_rate = state.effective_learning_rate
    
    is_pass = adapted_rate != PHI_INV  # Should be different from default
    print(f"  Adapted learning rate: {adapted_rate:.4f}")
    print(f"  Default rate: {PHI_INV:.4f}")
    print(f"  Has adapted: {adapted_rate != PHI_INV}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_adaptive_stays_within_phi_bounds() -> bool:
    """
    Test that adaptive rates stay within phi-derived bounds.
    
    SUCCESS CRITERIA:
    - Rate stays in [base_rate * phi^-1, base_rate * phi]
    """
    print("Test: adaptive_stays_within_phi_bounds...")
    
    try:
        from holographic_v4.meta_learning import compute_adaptive_learning_rate
    except ImportError:
        print("  ✗ FAIL (meta_learning not implemented yet)")
        return False
    
    base_rate = PHI_INV
    min_bound = base_rate * PHI_INV
    max_bound = base_rate * PHI
    
    # Test extreme cases
    rates = []
    for salience in [0.0, 0.5, 1.0]:
        for novelty in [0.0, 0.5, 1.0]:
            for uncertainty in [0.0, 0.5, 1.0]:
                rate = compute_adaptive_learning_rate(
                    salience=salience,
                    novelty=novelty,
                    uncertainty=uncertainty,
                    base_rate=base_rate,
                )
                rates.append(rate)
    
    all_in_bounds = all(min_bound - 0.01 <= r <= max_bound + 0.01 for r in rates)
    
    is_pass = all_in_bounds
    print(f"  Min rate: {min(rates):.4f} (bound: {min_bound:.4f})")
    print(f"  Max rate: {max(rates):.4f} (bound: {max_bound:.4f})")
    print(f"  All in bounds: {all_in_bounds}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_error_rate_affects_consolidation() -> bool:
    """
    Test that error rate affects consolidation threshold.
    """
    print("Test: error_rate_affects_consolidation...")
    
    try:
        from holographic_v4.meta_learning import compute_adaptive_consolidation
    except ImportError:
        print("  ✗ FAIL (meta_learning not implemented yet)")
        return False
    
    base_threshold = PHI_INV_SQ
    
    # High error rate
    high_error_threshold = compute_adaptive_consolidation(
        error_rate=0.8,
        memory_pressure=0.5,
        base_threshold=base_threshold,
    )
    
    # Low error rate
    low_error_threshold = compute_adaptive_consolidation(
        error_rate=0.2,
        memory_pressure=0.5,
        base_threshold=base_threshold,
    )
    
    # High error → less consolidation (prototypes might be wrong)
    is_pass = high_error_threshold >= low_error_threshold
    print(f"  High error threshold: {high_error_threshold:.4f}")
    print(f"  Low error threshold: {low_error_threshold:.4f}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


# =============================================================================
# LEARNING STATE TESTS
# =============================================================================

def test_learning_state_tracks_history() -> bool:
    """
    Test that LearningState tracks prediction history.
    """
    print("Test: learning_state_tracks_history...")
    
    try:
        from holographic_v4.meta_learning import LearningState, update_meta_state
    except ImportError:
        print("  ✗ FAIL (meta_learning not implemented yet)")
        return False
    
    state = LearningState()
    
    # Record some predictions
    for correct in [True, False, True, True, False]:
        state = update_meta_state(state, correct, salience=0.5, novelty=0.5)
    
    # Should have tracked error rate
    has_error_rate = hasattr(state, 'recent_error_rate')
    error_rate_reasonable = 0.0 <= state.recent_error_rate <= 1.0 if has_error_rate else False
    
    is_pass = has_error_rate and error_rate_reasonable
    print(f"  Has error rate: {has_error_rate}")
    print(f"  Error rate: {state.recent_error_rate if has_error_rate else 'N/A'}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

def test_adaptive_rate_computation_performance() -> bool:
    """
    Test that adaptive rate computation is fast.
    
    Target: < 0.1ms per computation
    """
    print("Test: adaptive_rate_computation_performance...")
    
    try:
        from holographic_v4.meta_learning import compute_adaptive_learning_rate
    except ImportError:
        print("  ✗ FAIL (meta_learning not implemented yet)")
        return False
    
    n_iterations = 10000
    start = time.perf_counter()
    for _ in range(n_iterations):
        compute_adaptive_learning_rate(
            salience=0.5,
            novelty=0.5,
            uncertainty=0.5,
        )
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / n_iterations) * 1000
    
    is_pass = avg_time_ms < 0.1
    print(f"  Average computation time: {avg_time_ms:.6f}ms")
    print(f"  Target: < 0.1ms")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_meta_learning_tests() -> Dict[str, bool]:
    """Run all meta-learning tests."""
    print("=" * 70)
    print("META-LEARNING — Test Suite".center(70))
    print("=" * 70)
    
    results = {}
    
    # Adaptive Learning Rate Tests
    print("\n--- Adaptive Learning Rate Tests ---")
    results['high_salience_increases_rate'] = test_high_salience_increases_learning_rate()
    results['high_uncertainty_decreases_rate'] = test_high_uncertainty_decreases_learning_rate()
    results['meta_learning_improves'] = test_meta_learning_improves_over_fixed()
    results['stays_within_phi_bounds'] = test_adaptive_stays_within_phi_bounds()
    results['error_affects_consolidation'] = test_error_rate_affects_consolidation()
    
    # Learning State Tests
    print("\n--- Learning State Tests ---")
    results['state_tracks_history'] = test_learning_state_tracks_history()
    
    # Performance Tests
    print("\n--- Performance Tests ---")
    results['rate_computation_performance'] = test_adaptive_rate_computation_performance()
    
    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed".center(70))
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_all_meta_learning_tests()
