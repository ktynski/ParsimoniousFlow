"""
Tests for Predictiveness-Based Semantic Extraction
===================================================

These tests verify that predictiveness correctly identifies semantic vs noise tokens
and achieves 100% accuracy on paraphrase generalization.
"""

import numpy as np
import pytest
from collections import defaultdict

from holographic_v4.predictiveness import (
    TokenStatistics,
    PredictivenessTracker,
    SemanticPrototypeBuilder,
    compute_semantic_context,
    semantic_retrieve,
    integrate_predictiveness,
    verify_semantic_extraction,
)
from holographic_v4.pipeline import TheoryTrueModel
from holographic_v4.algebra import build_clifford_basis


# =============================================================================
# TOKEN STATISTICS TESTS
# =============================================================================

def test_token_statistics_observe():
    """Test that TokenStatistics correctly tracks observations."""
    stats = TokenStatistics()
    
    stats.observe(target=100)
    stats.observe(target=100)
    stats.observe(target=101)
    
    assert stats.total_count == 3
    assert stats.target_counts[100] == 2
    assert stats.target_counts[101] == 1


def test_token_statistics_predictiveness_perfect():
    """Test predictiveness is ~1.0 when token always predicts same target."""
    stats = TokenStatistics()
    
    for _ in range(50):
        stats.observe(target=100)
    
    pred = stats.predictiveness(n_targets=5)
    assert pred >= 0.99, f"Perfect predictor should have predictiveness~1.0, got {pred}"


def test_token_statistics_predictiveness_random():
    """Test predictiveness is ~0 when token is random across targets."""
    stats = TokenStatistics()
    
    # Equal distribution across 5 targets
    for target in range(5):
        for _ in range(20):
            stats.observe(target=target)
    
    pred = stats.predictiveness(n_targets=5)
    assert pred < 0.1, f"Random token should have low predictiveness, got {pred}"


def test_token_statistics_dominant_target():
    """Test dominant_target returns the most frequent target."""
    stats = TokenStatistics()
    
    stats.observe(target=100)
    stats.observe(target=100)
    stats.observe(target=100)
    stats.observe(target=101)
    
    assert stats.dominant_target() == 100


# =============================================================================
# PREDICTIVENESS TRACKER TESTS
# =============================================================================

def test_tracker_observe():
    """Test that tracker correctly records observations."""
    tracker = PredictivenessTracker()
    
    context = [1, 2, 3]
    target = 100
    
    tracker.observe(context, target)
    
    assert tracker.total_observations == 1
    assert 100 in tracker.observed_targets
    assert tracker.token_stats[1].total_count == 1
    assert tracker.token_stats[2].total_count == 1
    assert tracker.token_stats[3].total_count == 1


def test_tracker_semantic_identification():
    """Test that tracker correctly identifies semantic vs noise tokens."""
    tracker = PredictivenessTracker()
    
    # Signature tokens: always appear with same target
    # Noise tokens: random across targets
    rng = np.random.default_rng(42)
    
    for _ in range(50):
        # Cluster 0: signature [0, 1, 2], target 500
        context = [
            200 + rng.integers(0, 50),  # noise
            0, 1,
            210 + rng.integers(0, 50),  # noise
            2,
            220 + rng.integers(0, 50),  # noise
        ]
        tracker.observe(context, target=500)
        
        # Cluster 1: signature [10, 11, 12], target 501
        context = [
            200 + rng.integers(0, 50),  # noise
            10, 11,
            210 + rng.integers(0, 50),  # noise
            12,
            220 + rng.integers(0, 50),  # noise
        ]
        tracker.observe(context, target=501)
    
    # Signature tokens should have high predictiveness
    assert tracker.predictiveness(0) > 0.9, "Signature token 0 should be predictive"
    assert tracker.predictiveness(1) > 0.9, "Signature token 1 should be predictive"
    assert tracker.predictiveness(10) > 0.9, "Signature token 10 should be predictive"
    
    # Check is_semantic
    assert tracker.is_semantic(0), "Token 0 should be classified as semantic"
    assert tracker.is_semantic(10), "Token 10 should be classified as semantic"


def test_tracker_extract_semantic():
    """Test that extract_semantic returns only predictive tokens."""
    tracker = PredictivenessTracker()
    rng = np.random.default_rng(42)
    
    # Train on structured data
    for _ in range(100):
        context = [200 + rng.integers(0, 50), 0, 1, 210 + rng.integers(0, 50), 2]
        tracker.observe(context, target=500)
    
    # Extract from a new context
    test_context = [205, 0, 1, 215, 2]
    semantic = tracker.extract_semantic(test_context)
    
    # Should extract only signature tokens
    assert 0 in semantic, "Token 0 should be extracted"
    assert 1 in semantic, "Token 1 should be extracted"
    assert 2 in semantic, "Token 2 should be extracted"
    # Note: noise tokens 205, 215 might or might not be extracted depending on their counts


def test_tracker_statistics():
    """Test that get_statistics returns correct summary."""
    tracker = PredictivenessTracker()
    
    tracker.observe([1, 2, 3], target=100)
    tracker.observe([1, 2, 4], target=100)
    tracker.observe([5, 6, 7], target=101)
    
    stats = tracker.get_statistics()
    
    assert stats['n_tokens'] == 7
    assert stats['n_targets'] == 2
    assert stats['total_observations'] == 3


# =============================================================================
# SEMANTIC CONTEXT COMPOSITION TESTS
# =============================================================================

def test_compute_semantic_context_filters():
    """Test that compute_semantic_context filters out noise tokens."""
    model = TheoryTrueModel(vocab_size=100, context_size=8)
    tracker = PredictivenessTracker()
    
    # Train tracker to identify semantic tokens
    for _ in range(50):
        tracker.observe([0, 1, 2], target=100)
        tracker.observe([10, 11, 12], target=101)
    
    # Full context with noise
    full_context = [50, 0, 1, 51, 2, 52]
    
    # Compute semantic context
    semantic_ctx = compute_semantic_context(full_context, tracker, model)
    
    # Should be a valid 4x4 matrix
    assert semantic_ctx.shape == (4, 4)
    assert np.isfinite(semantic_ctx).all()


def test_compute_semantic_context_cold_start():
    """Test cold start behavior when no semantic tokens found.
    
    v4.21.0: Removed fallback_to_full parameter. Now always uses full context
    when tracker hasn't identified semantic tokens yet (cold start).
    """
    model = TheoryTrueModel(vocab_size=100, context_size=8)
    tracker = PredictivenessTracker()
    
    # Empty tracker - no tokens are semantic yet
    context = [50, 51, 52]
    
    # Cold start: uses full context (NOT identity!)
    result = compute_semantic_context(context, tracker, model)
    assert result.shape == (4, 4)
    # Should NOT be identity matrix (full context has structure)
    # This is theory-true: identity has geometric meaning, returning it
    # causes decode_attractor to always return same token
    assert not np.allclose(result, np.eye(4))


# =============================================================================
# SEMANTIC PROTOTYPE BUILDER TESTS
# =============================================================================

def test_prototype_builder_creates_per_target():
    """Test that builder creates separate prototypes per target."""
    model = TheoryTrueModel(vocab_size=100, context_size=8)
    tracker = PredictivenessTracker()
    
    # Train tracker
    for _ in range(50):
        tracker.observe([0, 1, 2], target=500)
        tracker.observe([10, 11, 12], target=501)
    
    # Build prototypes
    builder = SemanticPrototypeBuilder(tracker, model)
    
    for _ in range(20):
        builder.add_episode([0, 1, 2], target=500)
        builder.add_episode([10, 11, 12], target=501)
    
    prototypes = builder.build_prototypes()
    
    assert 500 in prototypes, "Should have prototype for target 500"
    assert 501 in prototypes, "Should have prototype for target 501"
    assert prototypes[500]['support'] == 20
    assert prototypes[501]['support'] == 20


def test_prototype_builder_witnesses_differ():
    """Test that prototypes for different targets have different witnesses."""
    model = TheoryTrueModel(vocab_size=100, context_size=8)
    tracker = PredictivenessTracker()
    
    # Train tracker with well-separated tokens
    for _ in range(50):
        tracker.observe([0, 1, 2], target=500)
        tracker.observe([50, 51, 52], target=501)
    
    # Build prototypes
    builder = SemanticPrototypeBuilder(tracker, model)
    
    for _ in range(20):
        builder.add_episode([0, 1, 2], target=500)
        builder.add_episode([50, 51, 52], target=501)
    
    prototypes = builder.build_prototypes()
    
    # Witnesses should differ
    w1 = np.array(prototypes[500]['witness'])
    w2 = np.array(prototypes[501]['witness'])
    dist = np.linalg.norm(w1 - w2)
    
    # They should be different (not necessarily very different due to random embeddings)
    # Just check they're not identical
    assert dist > 1e-6, f"Witnesses should differ, got distance {dist}"


# =============================================================================
# SEMANTIC RETRIEVAL TESTS
# =============================================================================

def test_semantic_retrieve_correct_target():
    """Test that semantic_retrieve returns correct target."""
    model = TheoryTrueModel(vocab_size=100, context_size=8)
    tracker = PredictivenessTracker()
    
    # Train tracker
    for _ in range(50):
        tracker.observe([0, 1, 2], target=500)
        tracker.observe([50, 51, 52], target=501)
    
    # Build prototypes
    builder = SemanticPrototypeBuilder(tracker, model)
    
    for _ in range(20):
        builder.add_episode([0, 1, 2], target=500)
        builder.add_episode([50, 51, 52], target=501)
    
    prototypes = builder.build_prototypes()
    
    # Retrieve for known context
    target, conf, info = semantic_retrieve([0, 1, 2], prototypes, tracker, model)
    
    assert target == 500, f"Should retrieve target 500, got {target}"
    assert conf > 0, "Confidence should be positive"


def test_semantic_retrieve_paraphrase():
    """Test that semantic_retrieve works on paraphrases (same signature, different noise)."""
    model = TheoryTrueModel(vocab_size=600, context_size=8, noise_std=0.3)
    tracker = PredictivenessTracker()
    rng = np.random.default_rng(42)
    
    # Train with noisy contexts
    n_samples = 50
    for _ in range(n_samples):
        context = [
            200 + rng.integers(0, 50),
            0, 1,
            210 + rng.integers(0, 50),
            2,
            220 + rng.integers(0, 50),
        ]
        tracker.observe(context, target=500)
        model.train_step(context, 500)
    
    # Build prototype
    builder = SemanticPrototypeBuilder(tracker, model)
    for _ in range(20):
        context = [
            200 + rng.integers(0, 50),
            0, 1,
            210 + rng.integers(0, 50),
            2,
            220 + rng.integers(0, 50),
        ]
        builder.add_episode(context, target=500)
    
    prototypes = builder.build_prototypes()
    
    # Test on paraphrase (different noise tokens!)
    paraphrase = [
        300 + rng.integers(0, 50),  # Different noise range
        0, 1,
        310 + rng.integers(0, 50),
        2,
        320 + rng.integers(0, 50),
    ]
    
    target, conf, info = semantic_retrieve(paraphrase, prototypes, tracker, model)
    
    # Should still retrieve correct target despite different noise
    assert target == 500, f"Paraphrase should retrieve target 500, got {target}"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_integrate_predictiveness():
    """Test that integrate_predictiveness correctly wraps model."""
    model = TheoryTrueModel(vocab_size=100, context_size=4)
    model = integrate_predictiveness(model)
    
    assert hasattr(model, 'predictiveness_tracker')
    
    # Train step should now track
    model.train_step([1, 2, 3, 4], target=100)
    model.train_step([1, 2, 3, 4], target=100)
    
    assert model.predictiveness_tracker.total_observations == 2


def test_full_verification():
    """Run the full verification test."""
    passed = verify_semantic_extraction(verbose=False)
    assert passed, "Full semantic extraction verification should pass"


# =============================================================================
# PARAPHRASE GENERALIZATION TEST (THE KEY BENCHMARK)
# =============================================================================

def test_paraphrase_generalization_100_percent():
    """
    THE KEY TEST: Semantic extraction should achieve ~100% on paraphrases.
    
    This is the benchmark that motivated the entire investigation.
    Without semantic extraction: 24-42%
    With semantic extraction: 100%
    """
    model = TheoryTrueModel(vocab_size=600, context_size=8, noise_std=0.3)
    tracker = PredictivenessTracker()
    rng = np.random.default_rng(42)
    
    n_clusters = 5
    n_train_samples = 50
    
    # Training phase
    for cluster_id in range(n_clusters):
        base = cluster_id * 10
        signature = [base, base + 1, base + 2]
        target = 500 + cluster_id
        
        for _ in range(n_train_samples):
            context = [
                200 + rng.integers(0, 50),
                signature[0], signature[1],
                210 + rng.integers(0, 50),
                signature[2],
                220 + rng.integers(0, 50),
                230 + rng.integers(0, 50),
                240 + rng.integers(0, 50),
            ]
            tracker.observe(context, target)
            model.train_step(context, target)
    
    # Build prototypes
    builder = SemanticPrototypeBuilder(tracker, model)
    
    for cluster_id in range(n_clusters):
        base = cluster_id * 10
        signature = [base, base + 1, base + 2]
        target = 500 + cluster_id
        
        for _ in range(20):
            context = [
                200 + rng.integers(0, 50),
                signature[0], signature[1],
                210 + rng.integers(0, 50),
                signature[2],
                220 + rng.integers(0, 50),
                230 + rng.integers(0, 50),
                240 + rng.integers(0, 50),
            ]
            builder.add_episode(context, target)
    
    prototypes = builder.build_prototypes()
    
    # Test on PARAPHRASES (completely different noise tokens)
    correct = 0
    total = 0
    
    for cluster_id in range(n_clusters):
        base = cluster_id * 10
        signature = [base, base + 1, base + 2]
        expected = 500 + cluster_id
        
        for _ in range(10):
            # Use completely different noise token range!
            context = [
                300 + rng.integers(0, 50),
                signature[0], signature[1],
                310 + rng.integers(0, 50),
                signature[2],
                320 + rng.integers(0, 50),
                330 + rng.integers(0, 50),
                340 + rng.integers(0, 50),
            ]
            
            predicted, conf, info = semantic_retrieve(
                context, prototypes, tracker, model
            )
            
            if predicted == expected:
                correct += 1
            total += 1
    
    accuracy = correct / total
    
    # Should achieve very high accuracy (allowing small margin for edge cases)
    assert accuracy >= 0.95, f"Expected >=95% accuracy, got {accuracy:.1%}"


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_predictiveness_tests():
    """Run all predictiveness tests and report results."""
    print("=" * 70)
    print("PREDICTIVENESS MODULE TESTS")
    print("=" * 70)
    
    tests = [
        ("token_statistics_observe", test_token_statistics_observe),
        ("token_statistics_predictiveness_perfect", test_token_statistics_predictiveness_perfect),
        ("token_statistics_predictiveness_random", test_token_statistics_predictiveness_random),
        ("token_statistics_dominant_target", test_token_statistics_dominant_target),
        ("tracker_observe", test_tracker_observe),
        ("tracker_semantic_identification", test_tracker_semantic_identification),
        ("tracker_extract_semantic", test_tracker_extract_semantic),
        ("tracker_statistics", test_tracker_statistics),
        ("compute_semantic_context_filters", test_compute_semantic_context_filters),
        ("compute_semantic_context_cold_start", test_compute_semantic_context_cold_start),
        ("prototype_builder_creates_per_target", test_prototype_builder_creates_per_target),
        ("prototype_builder_witnesses_differ", test_prototype_builder_witnesses_differ),
        ("semantic_retrieve_correct_target", test_semantic_retrieve_correct_target),
        ("semantic_retrieve_paraphrase", test_semantic_retrieve_paraphrase),
        ("integrate_predictiveness", test_integrate_predictiveness),
        ("full_verification", test_full_verification),
        ("paraphrase_generalization_100_percent", test_paraphrase_generalization_100_percent),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    
    print()
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_predictiveness_tests()
    exit(0 if success else 1)
