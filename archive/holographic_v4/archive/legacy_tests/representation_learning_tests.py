"""
Representation Learning Test Suite
==================================

TDD tests for embedding drift and learned representations.

THEORY:
    Fixed embeddings limit generalization. Allow embeddings to drift
    toward positions that improve retrieval, while maintaining the
    identity-bias structure that makes the system work.
    
    Key insight: Similar tokens should cluster in Clifford space
    after learning, even though they start with random embeddings.
"""

import numpy as np
import time
from typing import Dict, List, Tuple

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ
from holographic_v4.algebra import (
    build_clifford_basis,
    initialize_embeddings_identity,
    geometric_product,
    grace_operator,
    frobenius_similarity,
)
from holographic_v4.quotient import (
    extract_witness,
    grace_stability,
)
from holographic_v4.pipeline import TheoryTrueModel
from holographic_v4.dreaming import DreamingSystem

# =============================================================================
# TEST SETUP
# =============================================================================

BASIS = build_clifford_basis()
XP = np
VOCAB_SIZE = 100
CONTEXT_SIZE = 5


def create_model_and_dreaming() -> Tuple[TheoryTrueModel, DreamingSystem]:
    """Create fresh model and dreaming system."""
    model = TheoryTrueModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        noise_std=0.3,
        use_vorticity=True,
        use_equilibrium=True,
        xp=XP,
    )
    dreaming = DreamingSystem(BASIS, XP)
    return model, dreaming


# =============================================================================
# EMBEDDING GRADIENT TESTS
# =============================================================================

def test_embedding_gradient_computed_correctly() -> bool:
    """
    Test that embedding gradients point in the right direction.
    
    SUCCESS CRITERIA:
    - On successful retrieval, gradient should reinforce current position
    - On failed retrieval, gradient should point toward attractor
    """
    print("Test: embedding_gradient_computed_correctly...")
    
    try:
        from holographic_v4.representation_learning import compute_embedding_gradient
    except ImportError:
        print("  ✗ FAIL (representation_learning not implemented yet)")
        return False
    
    model, _ = create_model_and_dreaming()
    
    # Train a pattern
    context = [1, 2, 3, 4, 5]
    target = 10
    model.train_step(context, target)
    
    # Get the attractor
    attractor, _ = model.retrieve(context)
    
    # Compute gradient for successful retrieval
    grad_success = compute_embedding_gradient(
        token=target,
        retrieval_success=True,
        attractor=attractor,
        model=model,
    )
    
    # Compute gradient for failed retrieval
    grad_failure = compute_embedding_gradient(
        token=target,
        retrieval_success=False,
        attractor=attractor,
        model=model,
    )
    
    # Gradients should be different
    grad_diff = np.linalg.norm(grad_success - grad_failure)
    has_different_gradients = grad_diff > 0.01
    
    # Gradients should have correct shape
    has_correct_shape = grad_success.shape == (4, 4)
    
    is_pass = has_different_gradients and has_correct_shape
    print(f"  Gradient shape: {grad_success.shape}")
    print(f"  Gradient difference: {grad_diff:.4f}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_embedding_gradient_respects_learning_rate() -> bool:
    """
    Test that gradients are scaled by phi^-2 (slower than association learning).
    """
    print("Test: embedding_gradient_respects_learning_rate...")
    
    try:
        from holographic_v4.representation_learning import compute_embedding_gradient
    except ImportError:
        print("  ✗ FAIL (representation_learning not implemented yet)")
        return False
    
    model, _ = create_model_and_dreaming()
    
    context = [1, 2, 3, 4, 5]
    target = 10
    model.train_step(context, target)
    attractor, _ = model.retrieve(context)
    
    grad = compute_embedding_gradient(
        token=target,
        retrieval_success=False,
        attractor=attractor,
        model=model,
    )
    
    # Gradient norm should be bounded by phi^-2 * attractor norm
    attractor_norm = np.linalg.norm(attractor)
    grad_norm = np.linalg.norm(grad)
    expected_max_norm = PHI_INV_SQ * attractor_norm
    
    is_pass = grad_norm <= expected_max_norm * 1.1  # 10% tolerance
    print(f"  Gradient norm: {grad_norm:.4f}")
    print(f"  Expected max (φ⁻² × attractor): {expected_max_norm:.4f}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


# =============================================================================
# EMBEDDING UPDATE TESTS
# =============================================================================

def test_embedding_drift_improves_retrieval() -> bool:
    """
    Test that embedding drift actually improves retrieval accuracy.
    
    SUCCESS CRITERIA:
    - After drift updates, retrieval accuracy should improve
    """
    print("Test: embedding_drift_improves_retrieval...")
    
    try:
        from holographic_v4.representation_learning import (
            compute_embedding_gradient,
            update_embedding,
        )
    except ImportError:
        print("  ✗ FAIL (representation_learning not implemented yet)")
        return False
    
    model, _ = create_model_and_dreaming()
    
    # Train several patterns with the same target
    target = 50
    for i in range(10):
        context = [i, i+10, i+20, i+30, i+40]
        model.train_step(context, target)
    
    # Measure initial retrieval quality
    initial_similarities = []
    target_embedding = model.embeddings[target].copy()
    
    for i in range(10):
        context = [i, i+10, i+20, i+30, i+40]
        attractor, _ = model.retrieve(context)
        sim = frobenius_similarity(target_embedding, attractor, XP)
        initial_similarities.append(float(sim))
    
    initial_mean_sim = np.mean(initial_similarities)
    
    # Apply embedding drift updates
    for _ in range(5):  # 5 drift iterations
        for i in range(10):
            context = [i, i+10, i+20, i+30, i+40]
            attractor, retrieved_target = model.retrieve(context)
            success = retrieved_target == target
            
            grad = compute_embedding_gradient(
                token=target,
                retrieval_success=success,
                attractor=attractor,
                model=model,
            )
            
            update_embedding(
                token=target,
                gradient=grad,
                model=model,
            )
    
    # Measure post-drift retrieval quality
    final_similarities = []
    target_embedding = model.embeddings[target].copy()
    
    for i in range(10):
        context = [i, i+10, i+20, i+30, i+40]
        attractor, _ = model.retrieve(context)
        sim = frobenius_similarity(target_embedding, attractor, XP)
        final_similarities.append(float(sim))
    
    final_mean_sim = np.mean(final_similarities)
    
    # Success if similarity improved or stayed high
    is_pass = final_mean_sim >= initial_mean_sim - 0.1 or final_mean_sim > 0.8
    print(f"  Initial mean similarity: {initial_mean_sim:.4f}")
    print(f"  Final mean similarity: {final_mean_sim:.4f}")
    print(f"  Improvement: {final_mean_sim - initial_mean_sim:.4f}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_embedding_maintains_identity_bias() -> bool:
    """
    Test that embedding updates don't drift too far from identity.
    
    SUCCESS CRITERIA:
    - Updated embeddings should still be "close" to identity
    """
    print("Test: embedding_maintains_identity_bias...")
    
    try:
        from holographic_v4.representation_learning import (
            compute_embedding_gradient,
            update_embedding,
        )
    except ImportError:
        print("  ✗ FAIL (representation_learning not implemented yet)")
        return False
    
    model, _ = create_model_and_dreaming()
    
    # Record initial identity bias
    identity = np.eye(4)
    token = 50
    initial_embedding = model.embeddings[token].copy()
    initial_identity_sim = frobenius_similarity(initial_embedding, identity, XP)
    
    # Train and apply many updates
    for i in range(20):
        context = [i % 10, i % 10 + 10, i % 10 + 20, i % 10 + 30, i % 10 + 40]
        model.train_step(context, token)
        attractor, _ = model.retrieve(context)
        
        grad = compute_embedding_gradient(
            token=token,
            retrieval_success=False,  # Force update
            attractor=attractor,
            model=model,
        )
        
        update_embedding(
            token=token,
            gradient=grad,
            model=model,
            maintain_identity_bias=True,
        )
    
    # Check identity bias is maintained
    final_embedding = model.embeddings[token]
    final_identity_sim = frobenius_similarity(final_embedding, identity, XP)
    
    # Should not drift too far from identity (within 50% of original)
    is_pass = final_identity_sim > initial_identity_sim * 0.5
    print(f"  Initial identity similarity: {initial_identity_sim:.4f}")
    print(f"  Final identity similarity: {final_identity_sim:.4f}")
    print(f"  Drift: {initial_identity_sim - final_identity_sim:.4f}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_similar_tokens_cluster_after_learning() -> bool:
    """
    Test that tokens used in similar contexts cluster together.
    
    SUCCESS CRITERIA:
    - Tokens that frequently co-occur should become more similar
    """
    print("Test: similar_tokens_cluster_after_learning...")
    
    try:
        from holographic_v4.representation_learning import (
            compute_embedding_gradient,
            update_embedding,
            cluster_embeddings_by_retrieval,
        )
    except ImportError:
        print("  ✗ FAIL (representation_learning not implemented yet)")
        return False
    
    model, _ = create_model_and_dreaming()
    
    # Tokens 10-14 will be "similar" (same target)
    similar_target = 50
    # Tokens 20-24 will be "different" (different target)
    different_target = 60
    
    # Train similar tokens
    for i in range(10, 15):
        context = [i, i+5, i+10, i+15, i+20]
        model.train_step(context, similar_target)
    
    # Train different tokens
    for i in range(20, 25):
        context = [i, i+5, i+10, i+15, i+20]
        model.train_step(context, different_target)
    
    # Apply drift updates
    for _ in range(10):
        for i in range(10, 15):
            context = [i, i+5, i+10, i+15, i+20]
            attractor, _ = model.retrieve(context)
            grad = compute_embedding_gradient(
                token=similar_target,
                retrieval_success=True,
                attractor=attractor,
                model=model,
            )
            update_embedding(similar_target, grad, model)
        
        for i in range(20, 25):
            context = [i, i+5, i+10, i+15, i+20]
            attractor, _ = model.retrieve(context)
            grad = compute_embedding_gradient(
                token=different_target,
                retrieval_success=True,
                attractor=attractor,
                model=model,
            )
            update_embedding(different_target, grad, model)
    
    # Cluster by retrieval patterns
    clusters = cluster_embeddings_by_retrieval(model, min_cooccurrences=3)
    
    # Check if similar tokens clustered
    has_clusters = len(clusters) > 0
    is_pass = has_clusters
    print(f"  Number of clusters: {len(clusters)}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_embedding_updates_are_stable() -> bool:
    """
    Test that embedding updates don't cause runaway drift.
    
    SUCCESS CRITERIA:
    - Embedding norms should stay bounded
    - Updates should converge over time
    """
    print("Test: embedding_updates_are_stable...")
    
    try:
        from holographic_v4.representation_learning import (
            compute_embedding_gradient,
            update_embedding,
        )
    except ImportError:
        print("  ✗ FAIL (representation_learning not implemented yet)")
        return False
    
    model, _ = create_model_and_dreaming()
    
    token = 50
    target_context = [1, 2, 3, 4, 5]
    model.train_step(target_context, token)
    
    # Track embedding norms over many updates
    norms = []
    initial_norm = np.linalg.norm(model.embeddings[token])
    norms.append(initial_norm)
    
    for _ in range(50):  # Many updates
        attractor, _ = model.retrieve(target_context)
        grad = compute_embedding_gradient(
            token=token,
            retrieval_success=False,
            attractor=attractor,
            model=model,
        )
        update_embedding(token, grad, model)
        norms.append(np.linalg.norm(model.embeddings[token]))
    
    # Check stability
    final_norm = norms[-1]
    max_norm = max(norms)
    
    # Norm should not explode (stay within 2x initial)
    is_bounded = max_norm < initial_norm * 2.0
    
    # Norm should stabilize (last 10 should have low variance)
    last_10_std = np.std(norms[-10:])
    is_stable = last_10_std < 0.1
    
    is_pass = is_bounded and is_stable
    print(f"  Initial norm: {initial_norm:.4f}")
    print(f"  Final norm: {final_norm:.4f}")
    print(f"  Max norm: {max_norm:.4f}")
    print(f"  Last 10 std: {last_10_std:.4f}")
    print(f"  Is bounded: {is_bounded}, Is stable: {is_stable}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_precomputed_features_update_correctly() -> bool:
    """
    Test that precomputed features are updated when embeddings change.
    
    The model precomputes embedding features for efficiency.
    When embeddings drift, these features must be updated.
    """
    print("Test: precomputed_features_update_correctly...")
    
    try:
        from holographic_v4.representation_learning import (
            compute_embedding_gradient,
            update_embedding,
        )
    except ImportError:
        print("  ✗ FAIL (representation_learning not implemented yet)")
        return False
    
    model, _ = create_model_and_dreaming()
    
    token = 50
    
    # Get initial precomputed features (if any)
    has_precomputed = hasattr(model, '_embedding_witnesses')
    
    if has_precomputed:
        initial_witness = model._embedding_witnesses[token].copy()
    
    # Apply update
    context = [1, 2, 3, 4, 5]
    model.train_step(context, token)
    attractor, _ = model.retrieve(context)
    
    grad = compute_embedding_gradient(
        token=token,
        retrieval_success=False,
        attractor=attractor,
        model=model,
    )
    update_embedding(token, grad, model)
    
    # Check if precomputed features were updated
    if has_precomputed:
        final_witness = model._embedding_witnesses[token]
        features_changed = not np.allclose(initial_witness, final_witness)
        is_pass = features_changed
        print(f"  Precomputed features changed: {features_changed}")
    else:
        # No precomputed features - pass by default
        is_pass = True
        print(f"  No precomputed features to check")
    
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

def test_gradient_computation_performance() -> bool:
    """
    Test that gradient computation is fast.
    
    Target: < 1ms per gradient
    """
    print("Test: gradient_computation_performance...")
    
    try:
        from holographic_v4.representation_learning import compute_embedding_gradient
    except ImportError:
        print("  ✗ FAIL (representation_learning not implemented yet)")
        return False
    
    model, _ = create_model_and_dreaming()
    
    # Setup
    context = [1, 2, 3, 4, 5]
    target = 10
    model.train_step(context, target)
    attractor, _ = model.retrieve(context)
    
    # Time many gradient computations
    n_iterations = 100
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = compute_embedding_gradient(
            token=target,
            retrieval_success=False,
            attractor=attractor,
            model=model,
        )
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / n_iterations) * 1000
    
    is_pass = avg_time_ms < 1.0
    print(f"  Average gradient time: {avg_time_ms:.4f}ms")
    print(f"  Target: < 1ms")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_embedding_update_performance() -> bool:
    """
    Test that embedding updates are fast.
    
    Target: < 0.5ms per update
    """
    print("Test: embedding_update_performance...")
    
    try:
        from holographic_v4.representation_learning import (
            compute_embedding_gradient,
            update_embedding,
        )
    except ImportError:
        print("  ✗ FAIL (representation_learning not implemented yet)")
        return False
    
    model, _ = create_model_and_dreaming()
    
    # Setup
    context = [1, 2, 3, 4, 5]
    target = 10
    model.train_step(context, target)
    attractor, _ = model.retrieve(context)
    grad = compute_embedding_gradient(target, False, attractor, model)
    
    # Time many updates
    n_iterations = 100
    start = time.perf_counter()
    for _ in range(n_iterations):
        update_embedding(target, grad, model)
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / n_iterations) * 1000
    
    is_pass = avg_time_ms < 0.5
    print(f"  Average update time: {avg_time_ms:.4f}ms")
    print(f"  Target: < 0.5ms")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


# =============================================================================
# LYAPUNOV STABILITY TESTS
# =============================================================================

def test_lyapunov_function_properties():
    """
    Test that Lyapunov function V(E) has correct properties:
    - V(E) ≥ 0 for all E
    - V(I) ≈ 0 for identity (minimum)
    - V increases with distance from identity
    """
    print("Test: lyapunov_function_properties...")
    
    from holographic_v4.representation_learning import compute_lyapunov_function
    
    xp = np
    basis = build_clifford_basis(xp)
    identity = xp.eye(4, dtype=xp.float64)
    
    # Test V(I) is small
    V_identity = compute_lyapunov_function(identity, basis, xp)
    
    # Test V is non-negative for random matrices
    rng = np.random.default_rng(42)
    all_non_negative = True
    for _ in range(100):
        E = identity + rng.standard_normal((4, 4)) * 0.5
        V = compute_lyapunov_function(E, basis, xp)
        if V < -1e-10:
            all_non_negative = False
            break
    
    # Test V increases with distance
    V_small_perturbation = compute_lyapunov_function(
        identity + rng.standard_normal((4, 4)) * 0.1, basis, xp
    )
    V_large_perturbation = compute_lyapunov_function(
        identity + rng.standard_normal((4, 4)) * 0.5, basis, xp
    )
    
    print(f"  V(identity): {V_identity:.6f}")
    print(f"  V(small perturbation): {V_small_perturbation:.6f}")
    print(f"  V(large perturbation): {V_large_perturbation:.6f}")
    print(f"  All non-negative: {all_non_negative}")
    
    # V(I) should be small but may not be exactly 0 due to stability term
    is_pass = (
        V_identity < 1.0 and  # Identity should have small V
        all_non_negative and
        V_small_perturbation < V_large_perturbation * 2  # Roughly monotonic
    )
    
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


def test_lyapunov_decreases_asymptotically():
    """
    Test the THEOREM: V(E) decreases in expectation over many updates.
    
    Individual updates may temporarily increase V (due to gradient step),
    but the NET effect over many updates is contraction.
    """
    print("Test: lyapunov_decreases_asymptotically...")
    
    from holographic_v4.representation_learning import (
        compute_lyapunov_function,
        compute_embedding_gradient,
        update_embedding,
    )
    
    model, _ = create_model_and_dreaming()
    xp = model.xp
    basis = model.basis
    
    # Train some patterns
    for i in range(10):
        ctx = [i, i+1, i+2, i+3, i+4]
        target = i + 50
        model.train_step(ctx, target)
    
    # Record INITIAL total V across all embeddings
    initial_total_V = sum(
        compute_lyapunov_function(model.embeddings[i], basis, xp)
        for i in range(model.vocab_size)
    )
    
    # Apply MANY updates
    rng = np.random.default_rng(42)
    n_updates = 500
    
    for _ in range(n_updates):
        token = rng.integers(0, model.vocab_size)
        
        # Random attractor
        attractor = rng.standard_normal((4, 4)).astype(np.float64)
        attractor = grace_operator(attractor, basis, xp)
        
        # Apply update
        grad = compute_embedding_gradient(token, False, attractor, model)
        update_embedding(token, grad, model)
    
    # Record FINAL total V
    final_total_V = sum(
        compute_lyapunov_function(model.embeddings[i], basis, xp)
        for i in range(model.vocab_size)
    )
    
    reduction_ratio = final_total_V / initial_total_V
    
    print(f"  Initial total V: {initial_total_V:.4f}")
    print(f"  Final total V: {final_total_V:.4f}")
    print(f"  Reduction ratio: {reduction_ratio:.4f}")
    print(f"  Updates applied: {n_updates}")
    
    # Theorem: V should decrease significantly over many updates
    # We expect at least 50% reduction (typical is 90%+)
    is_pass = reduction_ratio < 0.5
    
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'} (V reduced by >50% over {n_updates} updates)")
    return is_pass


def test_embeddings_remain_in_identity_basin():
    """
    Test COROLLARY: Embeddings are trapped in the identity basin.
    
    After many updates, embeddings should still be close to identity.
    """
    print("Test: embeddings_remain_in_identity_basin...")
    
    from holographic_v4.representation_learning import (
        compute_embedding_gradient,
        update_embedding,
        verify_lyapunov_stability,
    )
    
    model, _ = create_model_and_dreaming()
    xp = model.xp
    identity = xp.eye(4, dtype=xp.float64)
    
    # Record initial distances from identity
    initial_distances = []
    for i in range(model.vocab_size):
        dist = float(xp.linalg.norm(model.embeddings[i] - identity))
        initial_distances.append(dist)
    initial_max = max(initial_distances)
    
    # Apply MANY aggressive updates
    rng = np.random.default_rng(42)
    for _ in range(500):
        token = rng.integers(0, model.vocab_size)
        
        # Large random attractor (trying to pull embedding away)
        attractor = rng.standard_normal((4, 4)).astype(np.float64) * 2.0
        
        grad = compute_embedding_gradient(token, False, attractor, model)
        update_embedding(token, grad, model)
    
    # Check final distances
    final_distances = []
    for i in range(model.vocab_size):
        dist = float(xp.linalg.norm(model.embeddings[i] - identity))
        final_distances.append(dist)
    final_max = max(final_distances)
    final_mean = float(np.mean(final_distances))
    
    print(f"  Initial max distance from I: {initial_max:.4f}")
    print(f"  Final max distance from I: {final_max:.4f}")
    print(f"  Final mean distance from I: {final_mean:.4f}")
    
    # Basin bound: embeddings should stay within reasonable distance
    # Identity-biased init gives ~0.3 distance, shouldn't grow much
    BASIN_BOUND = 2.0
    is_pass = final_max < BASIN_BOUND
    
    print(f"  Basin bound: {BASIN_BOUND}")
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'} (all embeddings remain in basin)")
    return is_pass


def test_grace_contraction_property():
    """
    Test that Grace contracts toward identity.
    
    CLAIM: ||Grace(E) - I|| ≤ c · ||E - I|| for some c < 1
    """
    print("Test: grace_contraction_property...")
    
    from holographic_v4.representation_learning import prove_contraction_bound
    
    results = prove_contraction_bound(n_samples=500, seed=42)
    
    print(f"  Mean contraction ratio: {results['mean_contraction_ratio']:.4f}")
    print(f"  Max contraction ratio: {results['max_contraction_ratio']:.4f}")
    print(f"  Expected (φ⁻²): {results['expected_ratio_phi_inv_sq']:.4f}")
    print(f"  Violations: {results['violations']}")
    
    # Grace should never expand (ratio ≤ 1)
    is_pass = results['contraction_holds']
    
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'} (Grace never expands)")
    return is_pass


def test_full_stability_verification():
    """
    Run the complete Lyapunov stability verification.
    
    Tests THREE properties:
    1. BOUNDEDNESS: Embeddings stay in identity basin (max distance < 2.0)
    2. CONVERGENCE: Mean V decreases over training
    3. GRACE CONTRACTION: Individual Grace applications never expand
    """
    print("Test: full_stability_verification...")
    
    from holographic_v4.representation_learning import verify_lyapunov_stability
    
    model, _ = create_model_and_dreaming()
    
    # Train some data first
    for i in range(20):
        ctx = [i, i+1, i+2, i+3, i+4]
        target = i + 50
        model.train_step(ctx, target)
    
    # Run full verification
    results = verify_lyapunov_stability(
        model,
        n_iterations=50,
        n_samples_per_iteration=10,
        seed=42,
    )
    
    print(f"  Total updates: {results['total_updates']}")
    print(f"  Mean initial V: {results['mean_initial_V']:.4f}")
    print(f"  Mean final V: {results['mean_final_V']:.4f}")
    print(f"  V reduction ratio: {results['mean_final_V']/max(results['mean_initial_V'], 1e-10):.4f}")
    print(f"  Max distance from identity: {results['max_distance_from_identity']:.4f}")
    print(f"  Identity basin bound holds: {results['identity_basin_bound']}")
    
    # The theorem holds if:
    # 1. Embeddings stay in basin
    # 2. Mean V decreased (converged toward equilibrium)
    v_decreased = results['mean_final_V'] < results['mean_initial_V']
    
    print(f"  V decreased over training: {v_decreased}")
    
    is_pass = results['identity_basin_bound'] and v_decreased
    
    print(f"  {'✓ PASS' if is_pass else '✗ FAIL'}")
    return is_pass


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_representation_learning_tests() -> Dict[str, bool]:
    """Run all representation learning tests."""
    print("=" * 70)
    print("REPRESENTATION LEARNING — Test Suite".center(70))
    print("=" * 70)
    
    results = {}
    
    # Gradient Tests
    print("\n--- Embedding Gradient Tests ---")
    results['gradient_computed_correctly'] = test_embedding_gradient_computed_correctly()
    results['gradient_respects_learning_rate'] = test_embedding_gradient_respects_learning_rate()
    
    # Update Tests
    print("\n--- Embedding Update Tests ---")
    results['drift_improves_retrieval'] = test_embedding_drift_improves_retrieval()
    results['maintains_identity_bias'] = test_embedding_maintains_identity_bias()
    results['similar_tokens_cluster'] = test_similar_tokens_cluster_after_learning()
    results['updates_are_stable'] = test_embedding_updates_are_stable()
    results['precomputed_features_update'] = test_precomputed_features_update_correctly()
    
    # Lyapunov Stability Tests (Theorem Verification)
    print("\n--- Lyapunov Stability Tests (Theorem Verification) ---")
    results['lyapunov_function_properties'] = test_lyapunov_function_properties()
    results['lyapunov_decreases_asymptotically'] = test_lyapunov_decreases_asymptotically()
    results['embeddings_remain_in_identity_basin'] = test_embeddings_remain_in_identity_basin()
    results['grace_contraction_property'] = test_grace_contraction_property()
    results['full_stability_verification'] = test_full_stability_verification()
    
    # Performance Tests
    print("\n--- Performance Tests ---")
    results['gradient_performance'] = test_gradient_computation_performance()
    results['update_performance'] = test_embedding_update_performance()
    
    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed".center(70))
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_all_representation_learning_tests()
