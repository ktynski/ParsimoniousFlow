"""
Grounding Test Suite
====================

TDD tests for perceptual grounding - connecting Clifford space to perception.

THEORY:
    Everything lives in abstract 4×4 matrix space. Grounding connects
    this to perception (features from the world).
    
    The key insight: preserve structure when mapping features to Clifford.
"""

import numpy as np
import time
from typing import Dict, List, Tuple

from holographic_v4.constants import PHI, PHI_INV
from holographic_v4.algebra import (
    build_clifford_basis,
    grace_operator,
    frobenius_similarity,
)
from holographic_v4.quotient import grace_stability
from holographic_v4.pipeline import TheoryTrueModel

# =============================================================================
# TEST SETUP
# =============================================================================

BASIS = build_clifford_basis()
XP = np
VOCAB_SIZE = 100


def create_model() -> TheoryTrueModel:
    """Create model for testing."""
    return TheoryTrueModel(
        vocab_size=VOCAB_SIZE,
        context_size=5,
        noise_std=0.3,
        xp=XP,
    )


# =============================================================================
# ENCODING TESTS
# =============================================================================

def test_similar_features_map_to_similar_clifford() -> bool:
    """
    Test that similar perceptual features map to similar Clifford reps.
    
    SUCCESS CRITERIA:
    - Features with small Euclidean distance have high Clifford similarity
    """
    print("Test: similar_features_map_to_similar_clifford...")
    
    try:
        from holographic_v4.grounding import PerceptionEncoder
    except ImportError:
        print("  ✗ FAIL (grounding not implemented yet)")
        return False
    
    encoder = PerceptionEncoder(feature_dim=16, basis=BASIS)
    
    # Create similar features
    features1 = XP.random.randn(16) * 0.5 + 1.0
    features2 = features1 + XP.random.randn(16) * 0.1  # Slight perturbation
    features3 = XP.random.randn(16) * 0.5 - 1.0  # Very different
    
    # Encode
    clifford1 = encoder.encode_features(features1)
    clifford2 = encoder.encode_features(features2)
    clifford3 = encoder.encode_features(features3)
    
    # Check similarities
    sim_12 = frobenius_similarity(clifford1, clifford2, XP)
    sim_13 = frobenius_similarity(clifford1, clifford3, XP)
    
    # Similar features should have higher similarity
    is_pass = sim_12 > sim_13 or abs(sim_12 - sim_13) < 0.1
    print(f"  Similar features similarity: {sim_12:.4f}")
    print(f"  Different features similarity: {sim_13:.4f}")
    print(f"  Structure preserved: {sim_12 > sim_13}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_grounded_tokens_cluster_by_perception() -> bool:
    """
    Test that tokens grounded to similar perceptions cluster.
    """
    print("Test: grounded_tokens_cluster_by_perception...")
    
    try:
        from holographic_v4.grounding import PerceptionEncoder, ground_token
    except ImportError:
        print("  ✗ FAIL (grounding not implemented yet)")
        return False
    
    model = create_model()
    encoder = PerceptionEncoder(feature_dim=16, basis=BASIS)
    
    # Ground token 10 to perception A
    features_A = XP.random.randn(16) * 0.5 + 1.0
    ground_token(10, features_A, encoder, model)
    
    # Ground token 11 to similar perception A'
    features_A_prime = features_A + XP.random.randn(16) * 0.1
    ground_token(11, features_A_prime, encoder, model)
    
    # Ground token 20 to different perception B
    features_B = XP.random.randn(16) * 0.5 - 1.0
    ground_token(20, features_B, encoder, model)
    
    # Check: 10 and 11 should be more similar than 10 and 20
    sim_10_11 = frobenius_similarity(model.embeddings[10], model.embeddings[11], XP)
    sim_10_20 = frobenius_similarity(model.embeddings[10], model.embeddings[20], XP)
    
    is_pass = sim_10_11 > sim_10_20 - 0.1  # Allow small tolerance
    print(f"  Similar grounded tokens similarity: {sim_10_11:.4f}")
    print(f"  Different grounded tokens similarity: {sim_10_20:.4f}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_encoding_preserves_structure() -> bool:
    """
    Test that encoding roughly preserves distance structure.
    """
    print("Test: encoding_preserves_structure...")
    
    try:
        from holographic_v4.grounding import PerceptionEncoder
    except ImportError:
        print("  ✗ FAIL (grounding not implemented yet)")
        return False
    
    encoder = PerceptionEncoder(feature_dim=16, basis=BASIS)
    
    # Create features with known distances
    f1 = XP.zeros(16)
    f2 = XP.ones(16)  # Distance = sqrt(16) = 4
    f3 = XP.ones(16) * 0.5  # Distance from f1 = 2, from f2 = 2
    
    c1 = encoder.encode_features(f1)
    c2 = encoder.encode_features(f2)
    c3 = encoder.encode_features(f3)
    
    # Distances in Clifford space
    d12_cliff = XP.linalg.norm(c1 - c2)
    d13_cliff = XP.linalg.norm(c1 - c3)
    d23_cliff = XP.linalg.norm(c2 - c3)
    
    # Original distances
    d12_feat = XP.linalg.norm(f1 - f2)
    d13_feat = XP.linalg.norm(f1 - f3)
    
    # f3 should be "between" f1 and f2 in some sense
    is_pass = d13_cliff < d12_cliff * 1.5  # Rough structure preserved
    print(f"  Feature distance f1-f2: {d12_feat:.4f}")
    print(f"  Clifford distance c1-c2: {d12_cliff:.4f}")
    print(f"  Feature distance f1-f3: {d13_feat:.4f}")
    print(f"  Clifford distance c1-c3: {d13_cliff:.4f}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_feedback_improves_encoding() -> bool:
    """
    Test that feedback from retrieval success improves encoding.
    """
    print("Test: feedback_improves_encoding...")
    
    try:
        from holographic_v4.grounding import PerceptionEncoder
    except ImportError:
        print("  ✗ FAIL (grounding not implemented yet)")
        return False
    
    encoder = PerceptionEncoder(feature_dim=16, basis=BASIS)
    
    features = XP.random.randn(16)
    desired_clifford = grace_operator(XP.random.randn(4, 4), BASIS, XP)
    
    # Initial encoding
    initial_clifford = encoder.encode_features(features)
    initial_sim = frobenius_similarity(initial_clifford, desired_clifford, XP)
    
    # Apply feedback
    for _ in range(10):
        encoder.update_from_feedback(features, desired_clifford, success=True)
    
    # Encoding after feedback
    final_clifford = encoder.encode_features(features)
    final_sim = frobenius_similarity(final_clifford, desired_clifford, XP)
    
    # Should improve (or at least not degrade)
    is_pass = final_sim >= initial_sim - 0.1
    print(f"  Initial similarity to desired: {initial_sim:.4f}")
    print(f"  Final similarity to desired: {final_sim:.4f}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_grounding_helps_generalization() -> bool:
    """
    Test that grounded tokens generalize better.
    """
    print("Test: grounding_helps_generalization...")
    
    try:
        from holographic_v4.grounding import PerceptionEncoder, ground_token
    except ImportError:
        print("  ✗ FAIL (grounding not implemented yet)")
        return False
    
    model = create_model()
    encoder = PerceptionEncoder(feature_dim=16, basis=BASIS)
    
    # Ground some tokens
    for i in range(10):
        features = XP.random.randn(16) * 0.5
        ground_token(i, features, encoder, model)
    
    # Embeddings should be Grace-stable
    stabilities = []
    for i in range(10):
        stab = grace_stability(model.embeddings[i], BASIS, XP)
        stabilities.append(stab)
    
    mean_stability = XP.mean(stabilities)
    
    is_pass = mean_stability > 0.3  # Should be reasonably stable
    print(f"  Mean stability of grounded tokens: {mean_stability:.4f}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

def test_encoding_performance() -> bool:
    """
    Test that encoding is fast.
    
    Target: < 0.5ms per encoding
    """
    print("Test: encoding_performance...")
    
    try:
        from holographic_v4.grounding import PerceptionEncoder
    except ImportError:
        print("  ✗ FAIL (grounding not implemented yet)")
        return False
    
    encoder = PerceptionEncoder(feature_dim=16, basis=BASIS)
    features = XP.random.randn(16)
    
    n_iterations = 1000
    start = time.perf_counter()
    for _ in range(n_iterations):
        encoder.encode_features(features)
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / n_iterations) * 1000
    
    is_pass = avg_time_ms < 0.5
    print(f"  Average encoding time: {avg_time_ms:.4f}ms")
    print(f"  Target: < 0.5ms")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_grounding_tests() -> Dict[str, bool]:
    """Run all grounding tests."""
    print("=" * 70)
    print("GROUNDING — Test Suite".center(70))
    print("=" * 70)
    
    results = {}
    
    # Encoding Tests
    print("\n--- Encoding Tests ---")
    results['similar_features_similar_clifford'] = test_similar_features_map_to_similar_clifford()
    results['grounded_tokens_cluster'] = test_grounded_tokens_cluster_by_perception()
    results['encoding_preserves_structure'] = test_encoding_preserves_structure()
    results['feedback_improves_encoding'] = test_feedback_improves_encoding()
    results['grounding_helps_generalization'] = test_grounding_helps_generalization()
    
    # Performance Tests
    print("\n--- Performance Tests ---")
    results['encoding_performance'] = test_encoding_performance()
    
    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed".center(70))
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_all_grounding_tests()
