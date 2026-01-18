"""
Systematic Investigation: Contrastive Survivability Ranking (CSR) & Invariant Consistency (IC)

GOAL: Determine if CSR/IC is theory-true and necessary for FSCTF, or if current
approach is sufficient. Test-driven investigation before implementation.

THEORY QUESTIONS TO ANSWER:
1. Does grace flow produce meaningful "survivability" scores?
2. Do similar tokens converge to similar attractors under grace flow?
3. Are invariants actually preserved under quotient transformations?
4. Does survivability correlate with semantic quality?
5. Would CSR/IC improve over current contrastive learning?

PRIMITIVES TO TEST:
- E(ψ): Echo duration (generations until collapse)
- A(ψ): Action/energy (lower = more stable)
- D(ψ): Drift/collapse rate
- F_k(ψ): k steps of grace flow
- I(ψ): Invariant extraction

TEST PLAN:
1. Test grace flow produces stable attractors
2. Test similar embeddings → similar attractors
3. Test invariants preserved under transformations
4. Test survivability predicts semantic quality
5. Compare CSR vs current contrastive approach
"""

import numpy as np
import sys
import time
from typing import Tuple, List, Dict

sys.path.insert(0, '/Users/fractlphoneroom1/Desktop/ParsimoniousFlow')

from holographic_prod.core.algebra import (
    grace_with_stability_batch,
    grace_n_iterations,
    grace_operator,
    frobenius_cosine,
    get_cached_basis,
)

# Alias for consistency
def build_basis(xp):
    return get_cached_basis(xp)
from holographic_prod.core.constants import PHI, PHI_INV, DTYPE
from scipy.linalg import expm

# Wrapper for grace iteration
def grace_iterate(psi, n_iters=10, return_stability=False):
    """Wrapper around grace_n_iterations."""
    basis = get_cached_basis(np)
    result = grace_n_iterations(psi, basis, n_iters, np)
    if return_stability:
        # Compute stability as deviation from orthogonality
        stability = 1.0 - np.linalg.norm(result.T @ result - np.eye(4), 'fro') / 4.0
        return result, stability
    return result


# =============================================================================
# PRIMITIVE IMPLEMENTATIONS (what CSR/IC would use)
# =============================================================================

def compute_action(psi: np.ndarray) -> float:
    """
    A(ψ): Action/energy of a state.
    
    Lower action = more stable/coherent.
    Uses Frobenius norm deviation from identity as proxy.
    """
    I = np.eye(4, dtype=DTYPE)
    return float(np.linalg.norm(psi - I, 'fro'))


def compute_echo_duration(psi: np.ndarray, max_iters: int = 100, threshold: float = 0.01) -> int:
    """
    E(ψ): Echo duration - how many grace iterations until collapse.
    
    Higher = more stable semantic identity.
    """
    current = psi.copy()
    prev_norm = np.linalg.norm(current, 'fro')
    
    for k in range(max_iters):
        # Apply one grace iteration
        graced, stability = grace_iterate(current, n_iters=1, return_stability=True)
        
        # Check for collapse (large change)
        current_norm = np.linalg.norm(graced, 'fro')
        change = abs(current_norm - prev_norm) / (prev_norm + 1e-10)
        
        if change < threshold:
            return k  # Converged
        
        current = graced
        prev_norm = current_norm
    
    return max_iters  # Didn't collapse


def compute_drift(psi: np.ndarray, n_steps: int = 10) -> float:
    """
    D(ψ): Drift rate under grace flow.
    
    Higher drift = less stable identity.
    """
    original = psi.copy()
    evolved = grace_iterate(psi, n_iters=n_steps)
    
    # Geodesic distance on SO(4)
    diff = original.T @ evolved
    # Use Frobenius norm of log as distance proxy
    return float(np.linalg.norm(diff - np.eye(4), 'fro'))


def extract_invariants(psi: np.ndarray) -> Dict[str, float]:
    """
    I(ψ): Extract invariants that should be preserved under transformations.
    
    These are what IC loss would compare across views.
    """
    # Determinant (should be ±1 for SO(4))
    det = float(np.linalg.det(psi))
    
    # Frobenius norm
    fro_norm = float(np.linalg.norm(psi, 'fro'))
    
    # Trace (character of rotation)
    trace = float(np.trace(psi))
    
    # Eigenvalue structure (moduli of eigenvalues)
    eigenvalues = np.linalg.eigvals(psi)
    eigen_moduli = np.sort(np.abs(eigenvalues))[::-1]
    
    # "Grade energy" - decompose into basis and measure grade contributions
    basis = build_basis(np)
    coeffs = np.array([np.sum(psi * basis[i]) / 16 for i in range(16)])
    
    grade_energies = {
        'grade_0': float(np.abs(coeffs[0])),      # scalar
        'grade_1': float(np.linalg.norm(coeffs[1:5])),   # vector
        'grade_2': float(np.linalg.norm(coeffs[5:11])),  # bivector
        'grade_3': float(np.linalg.norm(coeffs[11:15])), # trivector
        'grade_4': float(np.abs(coeffs[15])),     # pseudoscalar
    }
    
    return {
        'det': det,
        'fro_norm': fro_norm,
        'trace': trace,
        'eigen_moduli': eigen_moduli.tolist(),
        **grade_energies,
    }


def survivability_score(psi: np.ndarray, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.5) -> float:
    """
    S(ψ) = α·E(ψ) - β·A(ψ) - γ·D(ψ)
    
    Higher = more survivable.
    """
    E = compute_echo_duration(psi)
    A = compute_action(psi)
    D = compute_drift(psi)
    
    return alpha * E - beta * A - gamma * D


# =============================================================================
# TEST 1: Grace Flow Produces Stable Attractors
# =============================================================================

def test_grace_produces_attractors():
    """
    THEORY: Grace flow should converge to stable attractors.
    
    Test: Apply grace to random SO(4) matrices, verify convergence.
    """
    print("\n" + "="*70)
    print("TEST 1: Grace Flow Produces Stable Attractors")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 20
    n_steps = 50
    
    converged_count = 0
    convergence_steps = []
    
    for i in range(n_samples):
        # Create random SO(4)
        Q, _ = np.linalg.qr(np.random.randn(4, 4))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        Q = Q.astype(DTYPE)
        
        # Track convergence
        prev = Q.copy()
        for step in range(n_steps):
            current = grace_iterate(prev, n_iters=1)
            
            change = np.linalg.norm(current - prev, 'fro')
            if change < 0.001:
                converged_count += 1
                convergence_steps.append(step)
                break
            prev = current
        else:
            convergence_steps.append(n_steps)
    
    convergence_rate = converged_count / n_samples
    avg_steps = np.mean(convergence_steps)
    
    print(f"\nResults:")
    print(f"  Convergence rate: {convergence_rate:.1%} ({converged_count}/{n_samples})")
    print(f"  Average steps to converge: {avg_steps:.1f}")
    print(f"  Min steps: {min(convergence_steps)}, Max steps: {max(convergence_steps)}")
    
    # VERDICT
    if convergence_rate > 0.8:
        print("\n✅ VERDICT: Grace flow DOES produce stable attractors")
        return True
    else:
        print("\n❌ VERDICT: Grace flow does NOT reliably produce attractors")
        return False


# =============================================================================
# TEST 2: Similar Embeddings → Similar Attractors
# =============================================================================

def test_similar_embeddings_similar_attractors():
    """
    THEORY: Similar initial states should converge to similar attractors.
    
    This is crucial for CSR - if true, we can use attractor convergence
    as a measure of semantic similarity.
    """
    print("\n" + "="*70)
    print("TEST 2: Similar Embeddings → Similar Attractors")
    print("="*70)
    
    np.random.seed(42)
    n_steps = 30
    
    # Create base embedding
    Q_base, _ = np.linalg.qr(np.random.randn(4, 4))
    if np.linalg.det(Q_base) < 0:
        Q_base[:, 0] *= -1
    Q_base = Q_base.astype(DTYPE)
    
    # Create similar embedding (small perturbation)
    perturbation = np.random.randn(4, 4) * 0.05
    Q_similar = Q_base + perturbation
    # Re-orthogonalize
    U, _, Vt = np.linalg.svd(Q_similar)
    Q_similar = (U @ Vt).astype(DTYPE)
    
    # Create different embedding
    Q_different, _ = np.linalg.qr(np.random.randn(4, 4))
    if np.linalg.det(Q_different) < 0:
        Q_different[:, 0] *= -1
    Q_different = Q_different.astype(DTYPE)
    
    # Evolve all three
    attractor_base = grace_iterate(Q_base, n_iters=n_steps)
    attractor_similar = grace_iterate(Q_similar, n_iters=n_steps)
    attractor_different = grace_iterate(Q_different, n_iters=n_steps)
    
    # Measure distances
    dist_base_similar = np.linalg.norm(attractor_base - attractor_similar, 'fro')
    dist_base_different = np.linalg.norm(attractor_base - attractor_different, 'fro')
    
    # Initial distances for comparison
    init_dist_similar = np.linalg.norm(Q_base - Q_similar, 'fro')
    init_dist_different = np.linalg.norm(Q_base - Q_different, 'fro')
    
    print(f"\nInitial distances:")
    print(f"  Base-Similar: {init_dist_similar:.4f}")
    print(f"  Base-Different: {init_dist_different:.4f}")
    
    print(f"\nAttractor distances (after {n_steps} grace steps):")
    print(f"  Base-Similar: {dist_base_similar:.4f}")
    print(f"  Base-Different: {dist_base_different:.4f}")
    
    print(f"\nRatio (similar/different):")
    print(f"  Initial: {init_dist_similar/init_dist_different:.4f}")
    print(f"  Attractor: {dist_base_similar/dist_base_different:.4f}")
    
    # VERDICT: Similar should stay closer than different
    if dist_base_similar < dist_base_different:
        print("\n✅ VERDICT: Similar embeddings DO converge to similar attractors")
        return True
    else:
        print("\n❌ VERDICT: Attractor convergence does NOT preserve similarity")
        return False


# =============================================================================
# TEST 3: Invariants Preserved Under Transformations
# =============================================================================

def test_invariants_preserved():
    """
    THEORY: Certain quantities should be invariant under quotient transformations.
    
    This is what IC loss would verify.
    """
    print("\n" + "="*70)
    print("TEST 3: Invariants Preserved Under Transformations")
    print("="*70)
    
    np.random.seed(42)
    
    # Create base embedding
    Q, _ = np.linalg.qr(np.random.randn(4, 4))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    Q = Q.astype(DTYPE)
    
    # Extract invariants before transformation
    inv_before = extract_invariants(Q)
    
    # Apply various "quotient" transformations
    transformations = {}
    
    # Q1: Small noise (information dropout)
    noise = np.random.randn(4, 4) * 0.01
    Q_noisy = Q + noise
    U, _, Vt = np.linalg.svd(Q_noisy)
    Q_noisy = (U @ Vt).astype(DTYPE)
    transformations['noise'] = Q_noisy
    
    # Q2: Basis rotation (should preserve most invariants)
    R, _ = np.linalg.qr(np.random.randn(4, 4))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    Q_rotated = (R @ Q @ R.T).astype(DTYPE)
    transformations['rotation'] = Q_rotated
    
    # Q3: Grace flow (k steps)
    Q_graced = grace_iterate(Q, n_iters=10)
    transformations['grace'] = Q_graced
    
    # Compare invariants
    print("\nInvariants comparison:")
    print(f"{'Invariant':<15} {'Original':>10} {'Noise':>10} {'Rotation':>10} {'Grace':>10}")
    print("-" * 60)
    
    invariant_keys = ['det', 'fro_norm', 'trace', 'grade_0', 'grade_1', 'grade_2']
    
    preserved_count = 0
    total_checks = 0
    
    for key in invariant_keys:
        orig = inv_before[key]
        values = [orig]
        
        for name, Q_trans in transformations.items():
            inv_after = extract_invariants(Q_trans)
            values.append(inv_after[key])
        
        # Check if values are close
        if isinstance(orig, float):
            max_diff = max(abs(v - orig) for v in values[1:])
            preserved = max_diff < 0.5  # Tolerance
            
            print(f"{key:<15} {orig:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f} {values[3]:>10.4f} {'✓' if preserved else '✗'}")
            
            if preserved:
                preserved_count += 1
            total_checks += 1
    
    preservation_rate = preserved_count / total_checks
    
    print(f"\nInvariant preservation rate: {preservation_rate:.1%}")
    
    # VERDICT
    if preservation_rate > 0.5:
        print("\n✅ VERDICT: Some invariants ARE preserved (IC loss is feasible)")
        return True
    else:
        print("\n❌ VERDICT: Invariants NOT preserved (IC loss may not work)")
        return False


# =============================================================================
# TEST 4: Survivability Predicts Semantic Quality
# =============================================================================

def test_survivability_predicts_quality():
    """
    THEORY: Higher survivability should correlate with better semantic quality.
    
    Test: Compare survivability of "good" vs "bad" embeddings.
    """
    print("\n" + "="*70)
    print("TEST 4: Survivability Predicts Semantic Quality")
    print("="*70)
    
    np.random.seed(42)
    
    # "Good" embeddings: well-conditioned, close to identity
    good_embeddings = []
    for i in range(10):
        Q, _ = np.linalg.qr(np.random.randn(4, 4))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        good_embeddings.append(Q.astype(DTYPE))
    
    # "Bad" embeddings: ill-conditioned, far from orthogonal
    bad_embeddings = []
    for i in range(10):
        # Create a matrix that's not quite orthogonal
        M = np.random.randn(4, 4) * 2
        # Partially orthogonalize
        U, S, Vt = np.linalg.svd(M)
        # Use non-uniform singular values
        S_bad = np.array([2.0, 1.5, 0.8, 0.3])
        Q = (U @ np.diag(S_bad) @ Vt).astype(DTYPE)
        bad_embeddings.append(Q)
    
    # Compute survivability
    good_scores = [survivability_score(Q) for Q in good_embeddings]
    bad_scores = [survivability_score(Q) for Q in bad_embeddings]
    
    avg_good = np.mean(good_scores)
    avg_bad = np.mean(bad_scores)
    
    print(f"\nSurvivability scores:")
    print(f"  Good embeddings: {avg_good:.2f} ± {np.std(good_scores):.2f}")
    print(f"  Bad embeddings: {avg_bad:.2f} ± {np.std(bad_scores):.2f}")
    
    # VERDICT
    if avg_good > avg_bad:
        print("\n✅ VERDICT: Survivability DOES predict quality")
        return True
    else:
        print("\n❌ VERDICT: Survivability does NOT predict quality")
        return False


# =============================================================================
# TEST 5: CSR vs Current Contrastive Comparison
# =============================================================================

def test_csr_vs_current():
    """
    THEORY: CSR should be more theory-true than current contrastive.
    
    Current: Geodesic distance on SO(4) manifold
    CSR: Attractor convergence + survivability
    
    Test: Which better separates similar from different?
    """
    print("\n" + "="*70)
    print("TEST 5: CSR vs Current Contrastive Comparison")
    print("="*70)
    
    np.random.seed(42)
    n_pairs = 20
    n_steps = 20
    
    # Generate similar pairs and different pairs
    similar_distances_current = []
    similar_distances_csr = []
    different_distances_current = []
    different_distances_csr = []
    
    for i in range(n_pairs):
        # Create base
        Q_base, _ = np.linalg.qr(np.random.randn(4, 4))
        if np.linalg.det(Q_base) < 0:
            Q_base[:, 0] *= -1
        Q_base = Q_base.astype(DTYPE)
        
        # Similar (small perturbation)
        perturbation = np.random.randn(4, 4) * 0.05
        Q_similar = Q_base + perturbation
        U, _, Vt = np.linalg.svd(Q_similar)
        Q_similar = (U @ Vt).astype(DTYPE)
        
        # Different (random)
        Q_different, _ = np.linalg.qr(np.random.randn(4, 4))
        if np.linalg.det(Q_different) < 0:
            Q_different[:, 0] *= -1
        Q_different = Q_different.astype(DTYPE)
        
        # Current approach: direct distance
        similar_distances_current.append(
            np.linalg.norm(Q_base - Q_similar, 'fro')
        )
        different_distances_current.append(
            np.linalg.norm(Q_base - Q_different, 'fro')
        )
        
        # CSR approach: attractor distance + survivability
        attr_base = grace_iterate(Q_base, n_iters=n_steps)
        attr_similar = grace_iterate(Q_similar, n_iters=n_steps)
        attr_different = grace_iterate(Q_different, n_iters=n_steps)
        
        similar_distances_csr.append(
            np.linalg.norm(attr_base - attr_similar, 'fro')
        )
        different_distances_csr.append(
            np.linalg.norm(attr_base - attr_different, 'fro')
        )
    
    # Compute separation metrics
    def separation_ratio(similar, different):
        """Higher = better separation"""
        return np.mean(different) / (np.mean(similar) + 1e-10)
    
    current_separation = separation_ratio(similar_distances_current, different_distances_current)
    csr_separation = separation_ratio(similar_distances_csr, different_distances_csr)
    
    print(f"\nSeparation metrics (higher = better):")
    print(f"  Current (direct distance): {current_separation:.2f}")
    print(f"  CSR (attractor distance): {csr_separation:.2f}")
    
    print(f"\nMean distances:")
    print(f"  Current similar: {np.mean(similar_distances_current):.4f}")
    print(f"  Current different: {np.mean(different_distances_current):.4f}")
    print(f"  CSR similar: {np.mean(similar_distances_csr):.4f}")
    print(f"  CSR different: {np.mean(different_distances_csr):.4f}")
    
    # VERDICT
    if csr_separation > current_separation * 1.1:  # 10% improvement threshold
        print("\n✅ VERDICT: CSR provides BETTER separation than current approach")
        return True
    elif csr_separation >= current_separation * 0.9:
        print("\n⚠️ VERDICT: CSR is COMPARABLE to current approach")
        return None
    else:
        print("\n❌ VERDICT: CSR provides WORSE separation than current approach")
        return False


# =============================================================================
# MAIN: Run All Tests
# =============================================================================

def run_all_tests():
    """Run systematic investigation of CSR/IC."""
    
    print("="*70)
    print("SYSTEMATIC CSR/IC INVESTIGATION")
    print("="*70)
    print("""
PURPOSE: Determine if CSR/IC is necessary and beneficial for FSCTF.

THEORY BEHIND CSR/IC:
- CSR: "Correct views should survive better than incorrect views"
- IC: "Identity should be preserved across transformations"

CURRENT APPROACH:
- Geodesic interpolation on SO(4) manifold
- Contrastive learning based on co-occurrence

QUESTION: Does switching to CSR/IC provide meaningful benefit?
""")
    
    results = {}
    
    # Run tests
    results['attractors'] = test_grace_produces_attractors()
    results['similarity'] = test_similar_embeddings_similar_attractors()
    results['invariants'] = test_invariants_preserved()
    results['quality'] = test_survivability_predicts_quality()
    results['comparison'] = test_csr_vs_current()
    
    # Summary
    print("\n" + "="*70)
    print("INVESTIGATION SUMMARY")
    print("="*70)
    
    for test, result in results.items():
        status = "✅ PASS" if result else ("⚠️ PARTIAL" if result is None else "❌ FAIL")
        print(f"  {test}: {status}")
    
    # Overall verdict
    passes = sum(1 for r in results.values() if r is True)
    total = len(results)
    
    print(f"\nOverall: {passes}/{total} tests passed")
    
    if passes >= 4:
        print("""
CONCLUSION: CSR/IC is THEORY-TRUE and should be implemented.
The primitives work as expected, and CSR provides meaningful benefits.
""")
    elif passes >= 2:
        print("""
CONCLUSION: CSR/IC has MERIT but may not be strictly necessary.
Current approach may be sufficient with minor enhancements.
Consider implementing CSR as an optional enhancement.
""")
    else:
        print("""
CONCLUSION: CSR/IC may NOT be the right approach for this architecture.
The theoretical benefits don't materialize in practice.
Focus on improving current contrastive learning instead.
""")
    
    return results


if __name__ == '__main__':
    results = run_all_tests()
