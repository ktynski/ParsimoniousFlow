"""
META-COGNITIVE TRAINING LOOP TESTS

Tests for integrating Planning and Theory of Mind into training.
Each component is tested individually, then the full loop is tested.

Components:
1. Planning-based sample selection (curiosity-driven)
2. Predictive coding (predict before seeing target)
3. Theory of Mind (source attribution)
4. Residual-based learning (only learn surprises)
5. Adaptive consolidation (plan when to sleep)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

# Imports
from holographic_v4.pipeline import TheoryTrueModel
from holographic_v4.dreaming import DreamingSystem, compute_prediction_residual
from holographic_v4.algebra import (
    build_clifford_basis, geometric_product, grace_operator,
    frobenius_similarity
)
from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE
from holographic_v4.quotient import grace_stability, extract_witness
from holographic_v4.planning import simulate_action
from holographic_v4.theory_of_mind import AgentModel, AgentModelBuilder
# CuriosityTracker not needed for these tests


# =============================================================================
# TEST 1: PLANNING-BASED SAMPLE SELECTION
# =============================================================================

def test_uncertainty_based_selection() -> bool:
    """
    Test: Can we identify high-uncertainty regions for exploration?
    
    THEORY-TRUE: Uncertainty = "do I have a stored answer?"
    NOT Grace-stability (which measures matrix structure).
    
    Use RETRIEVAL CONFIDENCE as uncertainty metric:
    - Hash match found → low uncertainty (we know this)
    - No match → high uncertainty (novel context)
    """
    print("\n" + "="*60)
    print("TEST 1: UNCERTAINTY-BASED SAMPLE SELECTION")
    print("="*60)
    
    xp = np
    basis = build_clifford_basis(xp)
    model = TheoryTrueModel(vocab_size=100, context_size=4, xp=xp)
    
    # Train on a subset of patterns (tokens 0-49)
    print("\n  Phase 1: Training on tokens 0-49...")
    for i in range(50):
        context = [i % 50, (i+1) % 50, (i+2) % 50]
        target = (i+3) % 50
        model.train_step(context, target)
    
    print(f"    Stored {model.num_attractors} attractors")
    
    # Compute uncertainty based on RETRIEVAL CONFIDENCE
    print("\n  Phase 2: Computing uncertainty via retrieval confidence...")
    
    from holographic_v4.constants import PHI_INV_SQ
    
    known_uncertainties = []
    unknown_uncertainties = []
    
    for test_token in range(100):
        # Create a context with this token
        context = [test_token, (test_token + 1) % 100, (test_token + 2) % 100]
        
        # THEORY-TRUE: Check via holographic retrieval confidence
        ctx_rep = model.compute_context_representation(context)
        _, _, confidence, _ = model.holographic_memory.retrieve(ctx_rep)
        has_match = confidence >= PHI_INV_SQ
        
        # Uncertainty = 1 if no match, 0 if match
        uncertainty = 0.0 if has_match else 1.0
        
        # Check if this is in known region (tokens 0-49) or unknown (50-99)
        if test_token < 50:
            known_uncertainties.append(uncertainty)
        else:
            unknown_uncertainties.append(uncertainty)
    
    avg_known_uncertainty = np.mean(known_uncertainties)
    avg_unknown_uncertainty = np.mean(unknown_uncertainties)
    
    print(f"    Known region (0-49) avg uncertainty: {avg_known_uncertainty:.3f}")
    print(f"    Unknown region (50-99) avg uncertainty: {avg_unknown_uncertainty:.3f}")
    
    # Test: Can we select high-uncertainty samples?
    print("\n  Phase 3: Testing selection strategy...")
    
    # Simulate planning-based selection
    all_candidates = list(range(100))
    candidate_uncertainties = []
    
    for token in all_candidates:
        context = [token, (token + 1) % 100, (token + 2) % 100]
        ctx_rep = model.compute_context_representation(context)
        _, _, confidence, _ = model.holographic_memory.retrieve(ctx_rep)
        has_match = confidence >= PHI_INV_SQ
        candidate_uncertainties.append(0.0 if has_match else 1.0)
    
    # Select samples with uncertainty > 0.5 (no stored match)
    uncertain_samples = [i for i, u in enumerate(candidate_uncertainties) if u > 0.5]
    
    # How many are from unknown region?
    from_unknown = sum(1 for idx in uncertain_samples if idx >= 50)
    from_known = len(uncertain_samples) - from_unknown
    
    print(f"    Uncertain samples (no stored match): {len(uncertain_samples)}")
    print(f"    From unknown region (50-99): {from_unknown}")
    print(f"    From known region (0-49): {from_known}")
    
    # Success criteria: Most uncertain samples should be from unknown region
    # (We stored patterns for 0-49, so 50-99 should all be uncertain)
    passed = from_unknown >= 45 and avg_known_uncertainty < avg_unknown_uncertainty
    
    if passed:
        print("\n  ✓ PASSED: Retrieval-based uncertainty identifies unknown regions")
    else:
        print("\n  ✗ FAILED: Retrieval uncertainty not working as expected")
    
    return passed


# =============================================================================
# TEST 2: PREDICTIVE CODING
# =============================================================================

def test_predictive_coding() -> bool:
    """
    Test: Can we predict before seeing, and compute meaningful residuals?
    
    THEORY-TRUE: Prediction residual = "did we predict the right TOKEN?"
    NOT matrix norm (attractors ≠ embeddings).
    
    - Predicted token == actual → no surprise (residual = 0)
    - Predicted token ≠ actual → surprise (residual = 1)
    """
    print("\n" + "="*60)
    print("TEST 2: PREDICTIVE CODING")
    print("="*60)
    
    xp = np
    basis = build_clifford_basis(xp)
    model = TheoryTrueModel(vocab_size=100, context_size=4, xp=xp)
    
    # Train deterministic patterns: context [a, a+1, a+2] → target a+3
    print("\n  Phase 1: Training deterministic patterns...")
    for i in range(30):
        context = [i, i+1, i+2]
        target = i + 3
        model.train_step(context, target)
    
    print(f"    Trained {model.num_attractors} patterns")
    
    # Test prediction on known pattern
    print("\n  Phase 2: Testing prediction on KNOWN pattern...")
    test_context = [5, 6, 7]
    expected_target = 8
    
    # Predict
    predicted_attractor, predicted_target = model.retrieve(test_context)
    
    print(f"    Context: {test_context}")
    print(f"    Expected target: {expected_target}")
    print(f"    Predicted target: {predicted_target}")
    
    known_correct = (predicted_target == expected_target)
    print(f"    Prediction correct: {known_correct}")
    
    # THEORY-TRUE residual: binary (correct or not)
    known_residual = 0.0 if known_correct else 1.0
    print(f"    Prediction residual: {known_residual:.1f} (0=correct, 1=wrong)")
    
    # Test prediction on NOVEL pattern (unseen context)
    print("\n  Phase 3: Testing prediction on NOVEL pattern...")
    novel_context = [50, 51, 52]  # Never seen
    actual_target = 53
    
    predicted_attractor_novel, predicted_target_novel = model.retrieve(novel_context)
    
    print(f"    Novel context: {novel_context}")
    print(f"    Predicted target: {predicted_target_novel}")
    print(f"    Actual target: {actual_target}")
    
    novel_correct = (predicted_target_novel == actual_target)
    novel_residual = 0.0 if novel_correct else 1.0
    print(f"    Prediction correct: {novel_correct}")
    print(f"    Prediction residual: {novel_residual:.1f}")
    
    # Test: Known patterns should have low residual, novel should have high
    print("\n  Phase 4: Comparing residuals...")
    print(f"    Known pattern residual: {known_residual:.1f} (should be 0)")
    print(f"    Novel pattern residual: {novel_residual:.1f} (should be 1)")
    
    # The key insight: known = no surprise, novel = surprise
    known_is_surprise = known_residual > 0.5
    novel_is_surprise = novel_residual > 0.5
    
    print(f"\n    Known pattern is surprise: {known_is_surprise} (should be False)")
    print(f"    Novel pattern is surprise: {novel_is_surprise} (should be True)")
    
    passed = not known_is_surprise and novel_is_surprise
    
    if passed:
        print("\n  ✓ PASSED: Predictive coding distinguishes known from novel")
    else:
        print("\n  ✗ FAILED: Predictive coding not working as expected")
    
    return passed


# =============================================================================
# TEST 3: THEORY OF MIND - SOURCE ATTRIBUTION
# =============================================================================

def test_source_attribution() -> bool:
    """
    Test: Can we track different "speakers" and build separate models?
    
    Theory: Text comes from different sources (characters, narrator).
    Each source has different patterns. ToM should track this.
    """
    print("\n" + "="*60)
    print("TEST 3: THEORY OF MIND - SOURCE ATTRIBUTION")
    print("="*60)
    
    xp = np
    basis = build_clifford_basis(xp)
    
    # Create agent builders for two "speakers"
    print("\n  Phase 1: Creating agent models for two speakers...")
    
    alice_builder = AgentModelBuilder(basis=basis, xp=xp)
    bob_builder = AgentModelBuilder(basis=basis, xp=xp)
    
    # Simulate observations from Alice (tokens 0-20, pattern: +1)
    print("\n  Phase 2: Training on Alice's speech patterns...")
    alice_contexts = []
    for i in range(20):
        ctx = xp.random.randn(4, 4) * 0.1 + xp.eye(4)  # Near-identity with variation
        ctx = grace_operator(ctx, basis, xp)
        
        # Use observe() method with optional context and target
        alice_builder.observe(ctx, context=[i, i+1], target=i+2)
        alice_contexts.append(ctx)
    
    alice_model = alice_builder.build()
    print(f"    Alice model built:")
    print(f"      Witness: [{alice_model.witness[0]:.3f}, {alice_model.witness[1]:.3f}]")
    print(f"      Observation count: {alice_model.observation_count}")
    
    # Simulate observations from Bob (different patterns)
    print("\n  Phase 3: Training on Bob's speech patterns...")
    bob_contexts = []
    for i in range(20):
        # Bob has different base pattern
        ctx = xp.random.randn(4, 4) * 0.2 + xp.eye(4) * 0.5
        ctx = grace_operator(ctx, basis, xp)
        
        bob_builder.observe(ctx, context=[i+50, i+51], target=i+52)
        bob_contexts.append(ctx)
    
    bob_model = bob_builder.build()
    print(f"    Bob model built:")
    print(f"      Witness: [{bob_model.witness[0]:.3f}, {bob_model.witness[1]:.3f}]")
    print(f"      Observation count: {bob_model.observation_count}")
    
    # Test: Are the models different?
    print("\n  Phase 4: Testing model differentiation...")
    
    are_similar = alice_model.is_similar_to(bob_model)
    
    print(f"    Alice and Bob similar? {are_similar} (should be False)")
    
    # Test: Can we attribute new observations to the correct speaker?
    print("\n  Phase 5: Testing source attribution...")
    
    # Generate a new observation similar to Alice's pattern
    new_alice_ctx = alice_contexts[0].copy() + xp.random.randn(4, 4) * 0.05
    new_alice_ctx = grace_operator(new_alice_ctx, basis, xp)
    
    # Which model is it closer to?
    alice_witness = alice_model.witness_matrix(basis, xp)
    bob_witness = bob_model.witness_matrix(basis, xp)
    
    new_witness = extract_witness(new_alice_ctx, basis, xp)
    
    alice_dist = abs(new_witness[0] - alice_model.witness[0]) + abs(new_witness[1] - alice_model.witness[1])
    bob_dist = abs(new_witness[0] - bob_model.witness[0]) + abs(new_witness[1] - bob_model.witness[1])
    
    print(f"    New observation distance to Alice: {alice_dist:.4f}")
    print(f"    New observation distance to Bob: {bob_dist:.4f}")
    
    attributed_correctly = alice_dist < bob_dist
    print(f"    Attributed to Alice: {attributed_correctly} (should be True)")
    
    passed = not are_similar and attributed_correctly
    
    if passed:
        print("\n  ✓ PASSED: Theory of Mind can track and attribute sources")
    else:
        print("\n  ✗ FAILED: Source attribution not working")
    
    return passed


# =============================================================================
# TEST 4: RESIDUAL-BASED LEARNING
# =============================================================================

def test_residual_learning() -> bool:
    """
    Test: Does learning only on residuals (surprises) improve efficiency?
    
    THEORY-TRUE: Residual = "did I predict correctly?"
    - Correct prediction → skip (already know this)
    - Wrong prediction → store (new information)
    """
    print("\n" + "="*60)
    print("TEST 4: RESIDUAL-BASED LEARNING")
    print("="*60)
    
    xp = np
    basis = build_clifford_basis(xp)
    
    # Model 1: Standard learning (store everything)
    print("\n  Phase 1: Training with STANDARD learning (store all)...")
    model_standard = TheoryTrueModel(vocab_size=100, context_size=4, xp=xp)
    
    # Train on patterns, including repetitions
    patterns = []
    for i in range(30):
        patterns.append(([i % 20, (i+1) % 20, (i+2) % 20], (i+3) % 20))
    
    # Add repetitions (same patterns seen again)
    for i in range(30):
        patterns.append(([i % 20, (i+1) % 20, (i+2) % 20], (i+3) % 20))
    
    for ctx, tgt in patterns:
        model_standard.train_step(ctx, tgt)
    
    print(f"    Patterns presented: {len(patterns)}")
    print(f"    Attractors stored: {model_standard.num_attractors}")
    
    # Model 2: Residual-based learning (only store surprises)
    print("\n  Phase 2: Training with RESIDUAL learning (only surprises)...")
    model_residual = TheoryTrueModel(vocab_size=100, context_size=4, xp=xp)
    
    surprises_stored = 0
    redundant_skipped = 0
    
    for ctx, tgt in patterns:
        # Predict first
        _, predicted_target = model_residual.retrieve(ctx)
        
        # THEORY-TRUE: Is this surprising? (wrong prediction)
        is_surprise = (predicted_target != tgt)
        
        # Only store if surprising (wrong prediction)
        if is_surprise:
            model_residual.train_step(ctx, tgt)
            surprises_stored += 1
        else:
            redundant_skipped += 1
    
    print(f"    Patterns presented: {len(patterns)}")
    print(f"    Surprises stored: {surprises_stored}")
    print(f"    Redundant skipped: {redundant_skipped}")
    print(f"    Attractors stored: {model_residual.num_attractors}")
    
    # Test: Does residual learning have fewer redundant stores?
    print("\n  Phase 3: Comparing efficiency...")
    
    efficiency_gain = redundant_skipped / len(patterns) * 100
    print(f"    Efficiency gain: {efficiency_gain:.1f}% fewer stores")
    
    # Test accuracy on test set
    print("\n  Phase 4: Testing accuracy...")
    test_patterns = [([i, i+1, i+2], i+3) for i in range(20)]
    
    standard_correct = 0
    residual_correct = 0
    
    for ctx, expected in test_patterns:
        _, pred_std = model_standard.retrieve(ctx)
        _, pred_res = model_residual.retrieve(ctx)
        
        if pred_std == expected:
            standard_correct += 1
        if pred_res == expected:
            residual_correct += 1
    
    standard_acc = standard_correct / len(test_patterns) * 100
    residual_acc = residual_correct / len(test_patterns) * 100
    
    print(f"    Standard model accuracy: {standard_acc:.0f}%")
    print(f"    Residual model accuracy: {residual_acc:.0f}%")
    
    # Success: Efficiency gain (should skip ~50% since patterns repeat)
    # and accuracy maintained
    passed = efficiency_gain >= 40 and residual_acc >= standard_acc - 5
    
    if passed:
        print("\n  ✓ PASSED: Residual learning is more efficient without accuracy loss")
    else:
        print("\n  ✗ FAILED: Residual learning didn't improve efficiency or lost accuracy")
    
    return passed


# =============================================================================
# TEST 5: ADAPTIVE CONSOLIDATION
# =============================================================================

def test_adaptive_consolidation() -> bool:
    """
    Test: Can planning decide WHEN to consolidate (sleep)?
    
    Theory: Don't sleep on a fixed schedule. Sleep when:
    - Memory is getting full
    - Many recent novel patterns
    - Error rate is increasing
    """
    print("\n" + "="*60)
    print("TEST 5: ADAPTIVE CONSOLIDATION (WHEN TO SLEEP)")
    print("="*60)
    
    xp = np
    basis = build_clifford_basis(xp)
    
    # Metrics that should trigger sleep
    @dataclass
    class ConsolidationState:
        memory_pressure: float = 0.0  # How full is memory? [0, 1]
        novelty_rate: float = 0.0     # How many recent surprises? [0, 1]
        error_rate: float = 0.0       # Recent prediction errors? [0, 1]
        time_since_sleep: int = 0     # Samples since last sleep
        
    def should_sleep(state: ConsolidationState) -> Tuple[bool, str]:
        """
        Theory-true decision: Sleep when uncertainty is high.
        
        Uses φ-derived thresholds:
        - Memory pressure > φ⁻¹ → sleep (memory full)
        - Novelty rate > φ⁻² → sleep (lots to consolidate)
        - Error rate > φ⁻² → sleep (need to reorganize)
        - Time since sleep > φ × base_interval → forced sleep
        """
        reasons = []
        
        if state.memory_pressure > PHI_INV:
            reasons.append(f"memory_pressure={state.memory_pressure:.2f} > φ⁻¹")
        
        if state.novelty_rate > PHI_INV_SQ:
            reasons.append(f"novelty_rate={state.novelty_rate:.2f} > φ⁻²")
        
        if state.error_rate > PHI_INV_SQ:
            reasons.append(f"error_rate={state.error_rate:.2f} > φ⁻²")
        
        base_interval = 1000
        if state.time_since_sleep > PHI * base_interval:
            reasons.append(f"time={state.time_since_sleep} > φ×{base_interval}")
        
        should = len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else "no trigger"
        
        return should, reason_str
    
    print("\n  Testing consolidation triggers...")
    
    # Test 1: Low pressure - don't sleep
    state1 = ConsolidationState(
        memory_pressure=0.3,
        novelty_rate=0.2,
        error_rate=0.1,
        time_since_sleep=500
    )
    should1, reason1 = should_sleep(state1)
    print(f"\n  State 1 (low pressure):")
    print(f"    Memory: {state1.memory_pressure}, Novelty: {state1.novelty_rate}, Error: {state1.error_rate}")
    print(f"    Should sleep: {should1} (expected: False)")
    print(f"    Reason: {reason1}")
    
    # Test 2: High memory pressure - sleep
    state2 = ConsolidationState(
        memory_pressure=0.7,
        novelty_rate=0.2,
        error_rate=0.1,
        time_since_sleep=500
    )
    should2, reason2 = should_sleep(state2)
    print(f"\n  State 2 (high memory pressure):")
    print(f"    Memory: {state2.memory_pressure}, Novelty: {state2.novelty_rate}, Error: {state2.error_rate}")
    print(f"    Should sleep: {should2} (expected: True)")
    print(f"    Reason: {reason2}")
    
    # Test 3: High novelty rate - sleep
    state3 = ConsolidationState(
        memory_pressure=0.3,
        novelty_rate=0.5,
        error_rate=0.1,
        time_since_sleep=500
    )
    should3, reason3 = should_sleep(state3)
    print(f"\n  State 3 (high novelty):")
    print(f"    Memory: {state3.memory_pressure}, Novelty: {state3.novelty_rate}, Error: {state3.error_rate}")
    print(f"    Should sleep: {should3} (expected: True)")
    print(f"    Reason: {reason3}")
    
    # Test 4: High error rate - sleep
    state4 = ConsolidationState(
        memory_pressure=0.3,
        novelty_rate=0.2,
        error_rate=0.5,
        time_since_sleep=500
    )
    should4, reason4 = should_sleep(state4)
    print(f"\n  State 4 (high error):")
    print(f"    Memory: {state4.memory_pressure}, Novelty: {state4.novelty_rate}, Error: {state4.error_rate}")
    print(f"    Should sleep: {should4} (expected: True)")
    print(f"    Reason: {reason4}")
    
    # Test 5: Timeout - sleep
    state5 = ConsolidationState(
        memory_pressure=0.3,
        novelty_rate=0.2,
        error_rate=0.1,
        time_since_sleep=2000
    )
    should5, reason5 = should_sleep(state5)
    print(f"\n  State 5 (timeout):")
    print(f"    Memory: {state5.memory_pressure}, Novelty: {state5.novelty_rate}, Error: {state5.error_rate}")
    print(f"    Time since sleep: {state5.time_since_sleep}")
    print(f"    Should sleep: {should5} (expected: True)")
    print(f"    Reason: {reason5}")
    
    passed = (
        not should1 and  # Low pressure: don't sleep
        should2 and      # High memory: sleep
        should3 and      # High novelty: sleep
        should4 and      # High error: sleep
        should5          # Timeout: sleep
    )
    
    if passed:
        print("\n  ✓ PASSED: Adaptive consolidation triggers work correctly")
    else:
        print("\n  ✗ FAILED: Consolidation triggers not working as expected")
    
    return passed


# =============================================================================
# TEST 6: FULL META-COGNITIVE LOOP
# =============================================================================

def test_full_meta_cognitive_loop() -> bool:
    """
    Test: Does the full meta-cognitive loop work end-to-end?
    
    Components:
    1. Uncertainty-based sample selection
    2. Predictive coding
    3. Residual-based learning
    4. Adaptive consolidation
    """
    print("\n" + "="*60)
    print("TEST 6: FULL META-COGNITIVE TRAINING LOOP")
    print("="*60)
    
    xp = np
    basis = build_clifford_basis(xp)
    
    # Create model
    model = TheoryTrueModel(vocab_size=100, context_size=4, xp=xp)
    
    # Simulation parameters
    n_samples = 200
    
    # Tracking
    surprises = 0
    redundant = 0
    sleep_count = 0
    
    # State tracking
    memory_pressure = 0.0
    novelty_rate = 0.0
    error_rate = 0.0
    time_since_sleep = 0
    recent_surprises = []
    recent_errors = []
    
    print(f"\n  Running meta-cognitive loop for {n_samples} samples...")
    
    for step in range(n_samples):
        # 1. PLANNING: Select sample based on uncertainty
        # (In real training, we'd select from a dataset based on uncertainty)
        # Here we simulate by mixing known and unknown patterns
        if xp.random.rand() < 0.3:  # 30% chance of exploring unknown
            token = xp.random.randint(50, 100)  # Unknown region
        else:
            token = xp.random.randint(0, 50)  # Known region
        
        context = [token % 100, (token + 1) % 100, (token + 2) % 100]
        target = (token + 3) % 100
        
        # 2. PREDICTIVE CODING: Predict before seeing
        ctx_rep = model.compute_context(context)
        predicted_attractor, predicted_target = model.retrieve(context)
        
        # 3. COMPUTE RESIDUAL (theory-true: token mismatch)
        is_surprise = (predicted_target != target)
        is_error = predicted_target != target
        
        recent_surprises.append(1 if is_surprise else 0)
        recent_errors.append(1 if is_error else 0)
        
        # Keep only last 50
        if len(recent_surprises) > 50:
            recent_surprises.pop(0)
            recent_errors.pop(0)
        
        # 4. RESIDUAL-BASED LEARNING: Only store surprises
        if is_surprise:
            model.train_step(context, target)
            surprises += 1
        else:
            redundant += 1
        
        # 5. UPDATE STATE
        memory_pressure = model.num_attractors / 500  # Assume 500 max
        novelty_rate = sum(recent_surprises) / max(len(recent_surprises), 1)
        error_rate = sum(recent_errors) / max(len(recent_errors), 1)
        time_since_sleep += 1
        
        # 6. ADAPTIVE CONSOLIDATION: Check if should sleep
        should_sleep = (
            memory_pressure > PHI_INV or
            novelty_rate > PHI_INV_SQ or
            error_rate > PHI_INV_SQ or
            time_since_sleep > 100
        )
        
        if should_sleep and time_since_sleep >= 20:  # Minimum interval
            # Simulate sleep (just reset counters for this test)
            sleep_count += 1
            time_since_sleep = 0
            recent_surprises = []
            recent_errors = []
    
    # Results
    print(f"\n  Results:")
    print(f"    Total samples: {n_samples}")
    print(f"    Surprises stored: {surprises}")
    print(f"    Redundant skipped: {redundant}")
    print(f"    Sleep cycles: {sleep_count}")
    print(f"    Final attractors: {model.num_attractors}")
    print(f"    Efficiency: {redundant / n_samples * 100:.1f}% skipped")
    
    # Test accuracy
    print(f"\n  Testing accuracy on known patterns...")
    correct = 0
    for i in range(20):
        ctx = [i, i+1, i+2]
        expected = i + 3
        _, pred = model.retrieve(ctx)
        if pred == expected:
            correct += 1
    
    accuracy = correct / 20 * 100
    print(f"    Accuracy: {accuracy:.0f}%")
    
    # Success criteria
    passed = (
        redundant > 0 and           # Some redundant patterns skipped
        sleep_count > 0 and         # Adaptive sleeping occurred
        accuracy >= 50              # Reasonable accuracy maintained
    )
    
    if passed:
        print("\n  ✓ PASSED: Full meta-cognitive loop works")
    else:
        print("\n  ✗ FAILED: Meta-cognitive loop has issues")
    
    return passed


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_meta_cognitive_tests() -> Dict[str, bool]:
    """Run all meta-cognitive tests and report results."""
    
    print("\n" + "="*70)
    print("         META-COGNITIVE TRAINING LOOP TESTS")
    print("="*70)
    
    results = {}
    
    # Run each test
    tests = [
        ("Uncertainty-Based Selection", test_uncertainty_based_selection),
        ("Predictive Coding", test_predictive_coding),
        ("Theory of Mind", test_source_attribution),
        ("Residual-Based Learning", test_residual_learning),
        ("Adaptive Consolidation", test_adaptive_consolidation),
        ("Full Meta-Cognitive Loop", test_full_meta_cognitive_loop),
    ]
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  ✗ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("                        SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED - Meta-cognitive loop ready for integration!")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed - investigate before integrating")
    
    return results


if __name__ == "__main__":
    run_all_meta_cognitive_tests()
