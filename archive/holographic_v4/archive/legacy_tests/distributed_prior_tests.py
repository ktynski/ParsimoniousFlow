"""
Distributed Prior Tests — Systematic Verification
==================================================

Tests for the theory-true distributed prior methods:
1. φ-kernel (NOT softmax!)
2. Superposed-attractor prior
3. Prior field (Green's function)
4. Factorized associative prior (Hebbian)
5. Combined retrieval with confidence-gated fallback
6. Basin coverage metrics

THEORY PREDICTIONS TO VERIFY:
- φ-kernel decays at rate φ^(-d), not softmax
- Superposition creates smooth interpolation between basins
- Factorized prior provides global fallback
- Geometric confidence identifies ambiguity
- Basin coverage is auditable
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/fractlphoneroom1/Desktop/ParsimoniousFlow')

from holographic_v4.algebra import build_clifford_basis, grace_operator
from holographic_v4.quotient import extract_witness, grace_stability
from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ
from holographic_v4.distributed_prior import (
    witness_distance,
    phi_kernel,
    extended_witness,
    superposed_attractor_prior,
    prior_field_retrieve,
    retrieve_with_factorized_prior,
    retrieve_with_distributed_prior,
    FactorizedAssociativePrior,
    compute_basin_coverage,
)


def test_phi_kernel():
    """
    TEST: φ-kernel decays at rate φ^(-d).
    
    THEORY:
        α = φ^(-d)
        This is NOT softmax — it's theory-derived from the golden ratio.
    """
    print("\n" + "="*70)
    print("TEST: φ-Kernel Decay Rate")
    print("="*70)
    
    # Test specific distances
    test_cases = [
        (0.0, 1.0),      # d=0 → α=1
        (1.0, PHI_INV),  # d=1 → α=φ⁻¹
        (2.0, PHI_INV_SQ),  # d=2 → α=φ⁻²
    ]
    
    all_passed = True
    for d, expected in test_cases:
        actual = phi_kernel(d)
        match = np.isclose(actual, expected, rtol=1e-10)
        status = "✓" if match else "✗"
        print(f"  d={d:.1f}: α={actual:.6f} (expected {expected:.6f}) {status}")
        all_passed = all_passed and match
    
    # Verify exponential decay
    distances = np.array([0, 1, 2, 3, 4, 5])
    values = np.array([phi_kernel(d) for d in distances])
    
    # Check ratio between consecutive values
    ratios = values[:-1] / values[1:]
    expected_ratio = PHI
    ratio_match = np.allclose(ratios, expected_ratio, rtol=1e-10)
    
    print(f"\n  Ratio between consecutive values: {ratios[0]:.6f}")
    print(f"  Expected (φ): {expected_ratio:.6f}")
    print(f"  Exponential decay verified: {ratio_match}")
    
    all_passed = all_passed and ratio_match
    
    if all_passed:
        print("\n  ✓ PASS: φ-kernel is theory-true (NOT softmax!)")
    else:
        print("\n  ✗ FAIL: φ-kernel incorrect")
    
    return all_passed


def test_witness_distance():
    """
    TEST: Witness distance is Euclidean in (scalar, pseudoscalar) space.
    """
    print("\n" + "="*70)
    print("TEST: Witness Distance")
    print("="*70)
    
    w1 = (1.0, 0.0)
    w2 = (0.0, 0.0)
    w3 = (1.0, 1.0)
    
    d12 = witness_distance(w1, w2)
    d13 = witness_distance(w1, w3)
    d23 = witness_distance(w2, w3)
    
    print(f"  d(w1, w2) = {d12:.6f} (expected: 1.0)")
    print(f"  d(w1, w3) = {d13:.6f} (expected: 1.0)")
    print(f"  d(w2, w3) = {d23:.6f} (expected: √2 ≈ 1.414)")
    
    test1 = np.isclose(d12, 1.0)
    test2 = np.isclose(d13, 1.0)
    test3 = np.isclose(d23, np.sqrt(2))
    
    all_passed = test1 and test2 and test3
    
    if all_passed:
        print("\n  ✓ PASS: Witness distance is Euclidean")
    else:
        print("\n  ✗ FAIL: Witness distance incorrect")
    
    return all_passed


def test_superposed_attractor_prior():
    """
    TEST: Superposed-attractor prior creates smooth interpolation.
    
    THEORY:
        When we superpose K prototypes with φ-weights:
        - Near one prototype → weights dominated by that one
        - Between prototypes → smooth blend
    """
    print("\n" + "="*70)
    print("TEST: Superposed-Attractor Prior")
    print("="*70)
    
    basis = build_clifford_basis(np)
    
    # Create two prototypes with distinct witnesses
    proto1 = np.eye(4) * 0.5
    proto1 = grace_operator(proto1, basis, np)
    
    proto2 = np.eye(4) * 2.0
    proto2 = grace_operator(proto2, basis, np)
    
    w1 = extract_witness(proto1, basis, np)
    w2 = extract_witness(proto2, basis, np)
    
    print(f"  Prototype 1 witness: ({w1[0]:.4f}, {w1[1]:.4f})")
    print(f"  Prototype 2 witness: ({w2[0]:.4f}, {w2[1]:.4f})")
    
    # Create target distributions
    targets = [{0: 1.0}, {1: 1.0}]
    
    # Test 1: Query close to prototype 1
    query1 = proto1 + 0.01 * np.random.randn(4, 4)
    query1 = grace_operator(query1, basis, np)
    
    eq1, tgt1, conf1, info1 = superposed_attractor_prior(
        query1, [proto1, proto2], targets, basis, np, K=2
    )
    
    print(f"\n  Query near proto1:")
    print(f"    Weights: {info1['weights']}")
    print(f"    Top indices: {info1['top_indices']}")
    print(f"    Confidence: {conf1:.4f}")
    # Proto1 (index 0) should be nearest to query1
    test1 = info1['top_indices'][0] == 0  # Proto1 is the nearest
    print(f"    Proto1 is nearest: {test1}")
    
    # Test 2: Query close to prototype 2 (NOT using random noise)
    query2 = proto2 * 1.001  # Slightly perturbed proto2
    query2 = grace_operator(query2, basis, np)
    
    eq2, tgt2, conf2, info2 = superposed_attractor_prior(
        query2, [proto1, proto2], targets, basis, np, K=2
    )
    
    print(f"\n  Query near proto2:")
    print(f"    Weights: {info2['weights']}")
    print(f"    Top indices: {info2['top_indices']}")
    print(f"    Confidence: {conf2:.4f}")
    # The first entry in weights corresponds to the nearest prototype (top_indices[0])
    # Proto2 (index 1) should be nearest to query2
    test2 = info2['top_indices'][0] == 1  # Proto2 is the nearest
    print(f"    Proto2 is nearest: {test2}")
    
    # Test 3: Query equidistant
    query3 = (proto1 + proto2) / 2
    query3 = grace_operator(query3, basis, np)
    
    eq3, tgt3, conf3, info3 = superposed_attractor_prior(
        query3, [proto1, proto2], targets, basis, np, K=2
    )
    
    print(f"\n  Query at midpoint:")
    print(f"    Weights: {info3['weights']}")
    print(f"    Confidence: {conf3:.4f}")
    weight_balance = abs(info3['weights'][0] - info3['weights'][1])
    test3 = weight_balance < 0.3
    print(f"    Roughly balanced: {test3}")
    
    all_passed = test1 and test2 and test3
    
    if all_passed:
        print("\n  ✓ PASS: Superposed prior creates smooth interpolation")
    else:
        print("\n  ✗ FAIL: Superposed prior incorrect")
    
    return all_passed


def test_factorized_prior():
    """
    TEST: Factorized associative prior learns witness→attractor mapping.
    
    THEORY:
        The factorized prior maintains:
        - C = Σ W⊗W (covariance)
        - B = Σ A⊗W (association)
        
        Prediction: Â(W) = B @ C^(-1) @ W
    """
    print("\n" + "="*70)
    print("TEST: Factorized Associative Prior")
    print("="*70)
    
    basis = build_clifford_basis(np)
    prior = FactorizedAssociativePrior(witness_dim=4)
    
    # Train on some prototype-witness pairs
    np.random.seed(42)
    
    for i in range(20):
        # Create matrix with varying scalar component
        M = np.eye(4) * (0.5 + i * 0.1)
        M = grace_operator(M, basis, np)
        
        # Compute extended witness
        W = extended_witness(M, basis, np)
        
        # Update prior
        prior.update(W, M)
    
    print(f"  Trained on 20 prototypes")
    print(f"  Updates: {prior.n_updates}")
    
    stats = prior.get_statistics()
    print(f"  C trace: {stats['C_trace']:.4f}")
    print(f"  B norm: {stats['B_norm']:.4f}")
    print(f"  Effective rank: {stats['effective_rank']:.4f}")
    
    # Test prediction
    test_M = np.eye(4) * 1.0
    test_M = grace_operator(test_M, basis, np)
    test_W = extended_witness(test_M, basis, np)
    
    predicted = prior.predict(test_W, basis)
    
    print(f"\n  Test witness: {test_W}")
    print(f"  Predicted attractor shape: {predicted.shape}")
    
    # Check prediction is valid (stable under Grace)
    stability = grace_stability(predicted, basis, np)
    print(f"  Predicted stability: {stability:.4f}")
    
    test_pass = stability > 0.3  # Should produce something reasonable
    
    if test_pass:
        print("\n  ✓ PASS: Factorized prior produces valid predictions")
    else:
        print("\n  ✗ FAIL: Factorized prior predictions invalid")
    
    return test_pass


def test_confidence_based_fallback():
    """
    TEST: Combined retrieval uses confidence to gate fallback.
    
    THEORY:
        - High confidence: trust local prototype
        - Low confidence: blend with global prior
    """
    print("\n" + "="*70)
    print("TEST: Confidence-Based Fallback")
    print("="*70)
    
    basis = build_clifford_basis(np)
    
    # Create prototypes
    np.random.seed(42)
    prototypes = []
    targets = []
    supports = []
    
    for i in range(5):
        M = np.eye(4) * (0.5 + i * 0.3)
        M = grace_operator(M, basis, np)
        prototypes.append(M)
        targets.append({i: 1.0})
        supports.append(10.0 + i * 5)
    
    # Create and train factorized prior
    prior = FactorizedAssociativePrior(witness_dim=4)
    for M in prototypes:
        W = extended_witness(M, basis, np)
        prior.update(W, M)
    
    print(f"  Created {len(prototypes)} prototypes")
    
    # Test 1: Query close to prototype (high confidence)
    query_close = prototypes[0] + 0.01 * np.random.randn(4, 4)
    query_close = grace_operator(query_close, basis, np)
    
    eq1, tgt1, conf1, info1 = retrieve_with_distributed_prior(
        query_close, prototypes, targets, supports, prior, basis, np,
        confidence_threshold=0.5
    )
    
    print(f"\n  Query close to proto0:")
    print(f"    Source: {info1['source']}")
    print(f"    Confidence: {conf1:.4f}")
    test1 = info1['source'] == 'superposed_prior'  # Should use local
    
    # Test 2: Query far from all (low confidence)
    query_far = np.eye(4) * 5.0  # Very different
    query_far = grace_operator(query_far, basis, np)
    
    eq2, tgt2, conf2, info2 = retrieve_with_distributed_prior(
        query_far, prototypes, targets, supports, prior, basis, np,
        confidence_threshold=0.5
    )
    
    print(f"\n  Query far from all:")
    print(f"    Source: {info2['source']}")
    print(f"    Confidence: {conf2:.4f}")
    test2 = info2['source'] == 'blended_prior'  # Should use blend
    
    all_passed = test1 and test2
    
    if all_passed:
        print("\n  ✓ PASS: Confidence-based fallback works correctly")
    else:
        print("\n  ✗ FAIL: Confidence-based fallback incorrect")
    
    return all_passed


def test_basin_coverage_metrics():
    """
    TEST: Basin coverage metrics audit what the system knows.
    """
    print("\n" + "="*70)
    print("TEST: Basin Coverage Metrics")
    print("="*70)
    
    basis = build_clifford_basis(np)
    
    # Create prototypes
    np.random.seed(42)
    prototypes = []
    for i in range(10):
        M = np.eye(4) * (0.5 + i * 0.2)
        M = grace_operator(M, basis, np)
        prototypes.append(M)
    
    # Create test queries
    test_queries = []
    for i in range(50):
        M = np.eye(4) * np.random.uniform(0.3, 3.0)
        M = grace_operator(M, basis, np)
        test_queries.append(M)
    
    metrics = compute_basin_coverage(test_queries, prototypes, basis, np)
    
    print(f"  Num prototypes: {metrics['num_prototypes']}")
    print(f"  Num queries: {metrics['num_queries']}")
    print(f"  Avg nearest distance: {metrics['avg_nearest_distance']:.4f}")
    print(f"  Coverage density: {metrics['coverage_density']:.4f}")
    print(f"  Boundary fraction: {metrics['boundary_fraction']:.4f}")
    print(f"  Basin entropy: {metrics['basin_entropy']:.4f}")
    print(f"  Normalized entropy: {metrics['normalized_entropy']:.4f}")
    
    # Verify metrics are reasonable
    test1 = metrics['num_prototypes'] == 10
    test2 = 0 <= metrics['coverage_density'] <= 1
    test3 = 0 <= metrics['boundary_fraction'] <= 1
    test4 = metrics['basin_entropy'] >= 0
    
    all_passed = test1 and test2 and test3 and test4
    
    if all_passed:
        print("\n  ✓ PASS: Basin coverage metrics computed correctly")
    else:
        print("\n  ✗ FAIL: Basin coverage metrics incorrect")
    
    return all_passed


def run_all_distributed_prior_tests():
    """Run all distributed prior tests."""
    print("="*70)
    print("DISTRIBUTED PRIOR TESTS")
    print("="*70)
    print("""
    Testing theory-true distributed prior methods:
    
    1. φ-kernel (NOT softmax!)
    2. Witness distance
    3. Superposed-attractor prior
    4. Factorized associative prior
    5. Confidence-based fallback
    6. Basin coverage metrics
    """)
    
    tests = [
        ("φ-Kernel", test_phi_kernel),
        ("Witness Distance", test_witness_distance),
        ("Superposed Prior", test_superposed_attractor_prior),
        ("Factorized Prior", test_factorized_prior),
        ("Confidence Fallback", test_confidence_based_fallback),
        ("Basin Coverage", test_basin_coverage_metrics),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  ✗ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    failed = total - passed
    if passed == total:
        print("\n  ✓ DISTRIBUTED PRIOR IS THEORY-TRUE!")
        print("    - φ-kernel replaces softmax")
        print("    - Superposition creates smooth interpolation")
        print("    - Factorized prior gives distributed 'weights'")
        print("    - Confidence gates local vs global")
        print("    - Basin coverage is auditable")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_distributed_prior_tests()
    exit(0 if failed == 0 else 1)
