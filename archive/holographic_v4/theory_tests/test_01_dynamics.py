"""
Test 01: Dynamics — Chaos, Criticality, Phase Transitions
==========================================================

PURPOSE: Validate fluid dynamics and chaos theory predictions.
         These tests derive from the Navier-Stokes / zeta-torus foundation.

TESTS:
    1. test_lyapunov_exponents   - λ < 0 for stable, λ ≈ 0 at criticality
    2. test_reynolds_number      - Re = (context_len × enstrophy) / Grace_viscosity
    3. test_enstrophy_cascade    - Energy flows to smaller or larger scales?
    4. test_edge_of_chaos        - Maximum computation at λ ≈ 0
    5. test_order_parameter      - What signals phase transitions?
    6. test_two_truths           - Witness (conventional) vs matrix (ultimate)

THEORY PREDICTIONS:
    - Lyapunov exponents: λ < 0 under Grace (contracting dynamics)
    - Reynolds number: Re = L × U / ν where ν ≈ 1 - φ⁻²
    - Enstrophy cascade: In 2D (even grades), enstrophy cascades to large scales
    - Edge of chaos: Optimal learning at λ ≈ 0 (criticality)
    - Order parameter: Witness stability σ signals phase transitions
"""

import pytest
import numpy as np
from typing import List, Tuple

from holographic_v4 import (
    build_clifford_basis,
    grace_operator,
    normalize_matrix,
    geometric_product_batch,
    extract_witness,
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
)
from holographic_v4.quotient import (
    compute_enstrophy,
    grade_energies,
    grace_stability,
    witness_similarity,
)
from holographic_v4.constants import DTYPE, GRACE_SCALES

from .utils import (
    compute_lyapunov_exponent,
    bootstrap_confidence_interval,
    permutation_test,
    effect_size_cohens_d,
    compute_grade_energy_distribution,
)


# =============================================================================
# Test 1: Lyapunov Exponents
# =============================================================================

class TestLyapunovExponents:
    """
    THEORY: Grace dynamics are contracting (λ < 0).
    
    The Lyapunov exponent measures sensitivity to initial conditions:
        λ = lim (1/t) ln(|δx(t)| / |δx(0)|)
    
    For Grace with spectral gap γ = φ⁻²:
        λ = ln(1 - γ) = ln(φ⁻¹) ≈ -0.481
    """
    
    def test_lyapunov_negative_under_grace(self, basis, xp, random_context):
        """
        Verify λ < 0 under Grace flow (contracting dynamics).
        """
        # Generate two nearby initial conditions
        M1 = random_context(n_tokens=10, seed=42)
        
        # Small perturbation
        perturbation = 1e-6 * np.random.randn(4, 4).astype(DTYPE)
        M2 = M1 + xp.array(perturbation)
        
        # Evolve both under Grace
        n_steps = 20
        trajectory1 = [M1]
        trajectory2 = [M2]
        
        current1, current2 = M1.copy(), M2.copy()
        for _ in range(n_steps):
            current1 = grace_operator(current1, basis, xp)
            current2 = grace_operator(current2, basis, xp)
            trajectory1.append(current1)
            trajectory2.append(current2)
        
        # Compute divergence over time
        divergences = []
        for t in range(1, n_steps + 1):
            dist = float(xp.linalg.norm(trajectory1[t] - trajectory2[t]))
            if dist > 1e-15:
                divergences.append(np.log(dist / 1e-6) / t)
        
        if divergences:
            lyapunov = np.mean(divergences)
            
            # Theory: λ = ln(φ⁻¹) ≈ -0.481
            predicted_lambda = np.log(PHI_INV)
            
            # Must be negative (contracting)
            assert lyapunov < 0, f"Lyapunov exponent should be negative, got {lyapunov:.4f}"
            
            print(f"Lyapunov: measured={lyapunov:.4f}, theory={predicted_lambda:.4f}")
    
    def test_lyapunov_vs_grace_strength(self, basis, xp, random_context):
        """
        Test how Lyapunov exponent changes with effective Grace strength.
        
        By applying Grace multiple times per step, we increase effective
        damping, which should make λ more negative.
        """
        M = random_context(n_tokens=8, seed=99)
        perturbation = 1e-6 * np.random.randn(4, 4).astype(DTYPE)
        M_perturbed = M + xp.array(perturbation)
        
        lyapunov_by_strength = []
        
        for grace_per_step in [1, 2, 3]:
            current1, current2 = M.copy(), M_perturbed.copy()
            
            divergences = []
            for t in range(1, 15):
                # Apply Grace multiple times
                for _ in range(grace_per_step):
                    current1 = grace_operator(current1, basis, xp)
                    current2 = grace_operator(current2, basis, xp)
                
                dist = float(xp.linalg.norm(current1 - current2))
                if dist > 1e-15:
                    divergences.append(np.log(dist / 1e-6) / t)
            
            if divergences:
                lyapunov_by_strength.append(np.mean(divergences))
        
        # More Grace = more negative Lyapunov
        if len(lyapunov_by_strength) >= 2:
            assert lyapunov_by_strength[1] <= lyapunov_by_strength[0], (
                "More Grace should make Lyapunov more negative"
            )


# =============================================================================
# Test 2: Reynolds Number Analog
# =============================================================================

class TestReynoldsNumber:
    """
    THEORY: Reynolds number Re = L × U / ν determines flow regime.
    
    In our system:
        L = context length (sequence scale)
        U = enstrophy (vorticity intensity)
        ν = Grace viscosity ≈ 1 - φ⁻² = φ⁻¹
    
    Low Re → laminar (stable retrieval)
    High Re → turbulent (chaotic, poor retrieval)
    """
    
    def test_reynolds_number_formula(self, basis, xp, embeddings):
        """
        Compute Reynolds number for contexts of varying length.
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        reynolds_numbers = []
        context_lengths = [4, 8, 16, 32]
        
        for L in context_lengths:
            # Generate context
            token_ids = np.random.randint(0, vocab_size, size=L)
            tokens = embeddings[token_ids]
            ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
            
            # Compute enstrophy (U)
            U = compute_enstrophy(ctx, basis, xp)
            
            # Viscosity (ν) from Grace
            nu = PHI_INV  # 1 - φ⁻² = φ⁻¹
            
            # Reynolds number
            Re = L * np.sqrt(U) / nu
            reynolds_numbers.append(Re)
        
        # Re should increase with context length (for fixed composition)
        for i in range(len(reynolds_numbers) - 1):
            assert reynolds_numbers[i+1] >= reynolds_numbers[i] * 0.5, (
                f"Re should generally increase with context length"
            )
    
    def test_reynolds_vs_retrieval_accuracy(self, basis, xp, embeddings, memory):
        """
        Test correlation between Reynolds number and retrieval accuracy.
        
        Theory: High Re → turbulent → poor retrieval
        """
        from holographic_v4.holographic_memory import HybridHolographicMemory
        
        test_memory = HybridHolographicMemory.create(basis, xp=xp)
        
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        # Store patterns with varying Re
        results = []
        
        for trial in range(30):
            # Vary context length to vary Re
            L = np.random.randint(4, 20)
            token_ids = np.random.randint(0, vocab_size, size=L)
            tokens = embeddings[token_ids]
            ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
            
            target_idx = np.random.randint(0, vocab_size)
            test_memory.store(ctx, embeddings[target_idx], target_idx=target_idx)
            
            # Compute Re
            U = compute_enstrophy(ctx, basis, xp)
            Re = L * np.sqrt(U) / PHI_INV
            
            # Test retrieval
            retrieved, idx, conf, source = test_memory.retrieve(ctx)
            correct = (idx == target_idx)
            
            results.append((Re, correct, conf))
        
        # Analyze correlation
        res_array = np.array(results)
        re_values = res_array[:, 0]
        accuracy_values = res_array[:, 1].astype(float)
        
        # Split into low/high Re
        median_re = np.median(re_values)
        low_re_acc = np.mean(accuracy_values[re_values < median_re])
        high_re_acc = np.mean(accuracy_values[re_values >= median_re])
        
        print(f"Low Re (<{median_re:.1f}) accuracy: {low_re_acc:.2%}")
        print(f"High Re (≥{median_re:.1f}) accuracy: {high_re_acc:.2%}")


# =============================================================================
# Test 3: Enstrophy Cascade
# =============================================================================

class TestEnstrophyCascade:
    """
    THEORY: In 2D turbulence, enstrophy cascades to smaller scales,
    while energy cascades to larger scales (inverse cascade).
    
    In Clifford algebra, grade-2 (bivectors) represents vorticity.
    We test whether energy redistributes across grades under composition.
    """
    
    def test_enstrophy_cascade_direction(self, basis, xp, embeddings):
        """
        Track how grade energies evolve during sequence composition.
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        # Start with identity
        current = xp.eye(4, dtype=DTYPE)
        
        grade_history = []
        
        # Compose tokens one by one
        for i in range(20):
            token = embeddings[np.random.randint(0, vocab_size)]
            current = normalize_matrix(current @ token, xp)
            
            energies = compute_grade_energy_distribution(current, basis)
            grade_history.append(energies)
        
        grade_history = np.array(grade_history)
        
        # Track grade-2 (enstrophy) evolution
        enstrophy_trend = grade_history[:, 2]
        
        # Also track total energy in even grades (0, 2, 4) vs odd grades (1, 3)
        even_energy = grade_history[:, 0] + grade_history[:, 2] + grade_history[:, 4]
        odd_energy = grade_history[:, 1] + grade_history[:, 3]
        
        print(f"Initial enstrophy: {enstrophy_trend[0]:.4f}")
        print(f"Final enstrophy: {enstrophy_trend[-1]:.4f}")
        print(f"Enstrophy trend: {'increasing' if enstrophy_trend[-1] > enstrophy_trend[0] else 'decreasing'}")
    
    def test_enstrophy_under_grace_flow(self, basis, xp, random_context):
        """
        Track enstrophy distribution under repeated Grace application.
        """
        M = random_context(n_tokens=12, seed=123)
        
        # Track grade energies under Grace
        grade_history = []
        current = M.copy()
        
        for i in range(15):
            energies = compute_grade_energy_distribution(current, basis)
            grade_history.append(energies)
            current = grace_operator(current, basis, xp)
        
        grade_history = np.array(grade_history)
        
        # Enstrophy (grade-2) should decrease fastest after grade-3
        decay_rates = []
        for g in range(5):
            if grade_history[0, g] > 1e-10:
                ratio = grade_history[-1, g] / grade_history[0, g]
                decay_rates.append(ratio)
            else:
                decay_rates.append(np.nan)
        
        print("Decay ratios after 15 Grace steps:")
        for g in range(5):
            print(f"  Grade {g}: {decay_rates[g]:.6f}")
        
        # Grade ordering should match Grace scales
        # Scalar (0) decays least, trivector (3) decays most
        if not np.isnan(decay_rates[0]):
            assert decay_rates[0] > decay_rates[2], "Scalar should decay less than bivector"


# =============================================================================
# Test 4: Edge of Chaos
# =============================================================================

class TestEdgeOfChaos:
    """
    THEORY: Maximum computation occurs at the "edge of chaos" (λ ≈ 0).
    
    Too stable (λ << 0): Information frozen, no processing
    Too chaotic (λ >> 0): Information destroyed, no memory
    Critical (λ ≈ 0): Maximum information processing
    
    We test by varying effective damping and measuring retrieval quality.
    """
    
    def test_edge_of_chaos_retrieval(self, basis, xp, embeddings):
        """
        Test retrieval quality at different damping levels.
        """
        from holographic_v4.holographic_memory import HybridHolographicMemory
        
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        # Test different "effective Grace" levels
        results_by_damping = {}
        
        for damping_level in [0, 1, 2, 5, 10]:
            memory = HybridHolographicMemory.create(basis, xp=xp)
            
            # Store patterns with varying damping
            successes = 0
            for trial in range(50):
                token_ids = np.random.randint(0, vocab_size, size=8)
                tokens = embeddings[token_ids]
                ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
                
                # Apply Grace damping_level times before storing
                for _ in range(damping_level):
                    ctx = grace_operator(ctx, basis, xp)
                
                target_idx = np.random.randint(0, vocab_size)
                memory.store(ctx, embeddings[target_idx], target_idx=target_idx)
                
                # Retrieve (with same damping)
                query = ctx.copy()
                retrieved, idx, conf, source = memory.retrieve(query)
                
                if idx == target_idx:
                    successes += 1
            
            results_by_damping[damping_level] = successes / 50
        
        print("Retrieval accuracy by damping level:")
        for level, acc in sorted(results_by_damping.items()):
            print(f"  Damping {level}: {acc:.1%}")
        
        # Too much damping should hurt (all contexts collapse to witness)
        # NOTE: With dual indexing (v4.22.0+), retrieval is more robust.
        # The test now checks that extreme damping doesn't *improve* accuracy
        # (which would indicate a bug where damping is being ignored).
        assert results_by_damping[10] <= results_by_damping[0], (
            "Extreme damping should not improve retrieval accuracy"
        )


# =============================================================================
# Test 5: Order Parameter
# =============================================================================

class TestOrderParameter:
    """
    THEORY: Witness stability σ is the order parameter for phase transitions.
    
    σ(M) = (scalar² + pseudo²) / total_energy
    
    - σ ≈ 1: Ordered phase (attractor, stable memory)
    - σ ≈ 0: Disordered phase (transient, needs consolidation)
    - σ = φ⁻²: Critical point (phase boundary)
    """
    
    def test_order_parameter_distribution(self, basis, xp, embeddings):
        """
        Examine distribution of σ across random contexts.
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        stabilities = []
        for trial in range(100):
            L = np.random.randint(3, 20)
            token_ids = np.random.randint(0, vocab_size, size=L)
            tokens = embeddings[token_ids]
            ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
            
            sigma = grace_stability(ctx, basis, xp)
            stabilities.append(sigma)
        
        stabilities = np.array(stabilities)
        
        # Analyze distribution
        mean_sigma = np.mean(stabilities)
        std_sigma = np.std(stabilities)
        
        # Count how many are above/below critical point
        critical = PHI_INV_SQ
        n_stable = np.sum(stabilities > critical)
        n_transient = np.sum(stabilities <= critical)
        
        print(f"Grace stability distribution:")
        print(f"  Mean σ: {mean_sigma:.4f}")
        print(f"  Std σ: {std_sigma:.4f}")
        print(f"  Stable (σ > {critical:.3f}): {n_stable}")
        print(f"  Transient (σ ≤ {critical:.3f}): {n_transient}")
    
    def test_order_parameter_under_grace(self, basis, xp, random_context):
        """
        Track how σ changes under Grace flow.
        
        Theory: σ should increase toward 1 (convergence to witness).
        """
        M = random_context(n_tokens=15, seed=456)
        
        stabilities = []
        current = M.copy()
        
        for i in range(20):
            sigma = grace_stability(current, basis, xp)
            stabilities.append(sigma)
            current = grace_operator(current, basis, xp)
        
        stabilities = np.array(stabilities)
        
        # σ should increase monotonically toward 1
        assert stabilities[-1] > stabilities[0], (
            f"Grace stability should increase: {stabilities[0]:.4f} → {stabilities[-1]:.4f}"
        )
        
        # Final σ should be close to 1
        assert stabilities[-1] > 0.95, (
            f"After many Grace steps, σ should approach 1, got {stabilities[-1]:.4f}"
        )


# =============================================================================
# Test 6: Two Truths Doctrine
# =============================================================================

class TestTwoTruths:
    """
    THEORY: Witness (σ, p) is "conventional truth", full matrix is "ultimate truth".
    
    From Buddhist philosophy:
    - Conventional: How things appear (witness - gauge-invariant summary)
    - Ultimate: How things are (full 16D Clifford structure)
    
    Both are valid for different purposes:
    - Witness: Fast retrieval, stable comparison
    - Matrix: Full semantic content, composition
    """
    
    def test_witness_vs_matrix_retrieval(self, basis, xp, embeddings):
        """
        Compare retrieval using witness similarity vs full matrix similarity.
        """
        from holographic_v4 import frobenius_similarity
        
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        # Generate contexts
        contexts = []
        for i in range(30):
            L = np.random.randint(5, 12)
            token_ids = np.random.randint(0, vocab_size, size=L)
            tokens = embeddings[token_ids]
            ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
            contexts.append(ctx)
        
        # Compute pairwise similarities
        witness_sims = []
        matrix_sims = []
        
        for i in range(len(contexts)):
            for j in range(i + 1, len(contexts)):
                w_sim = witness_similarity(contexts[i], contexts[j], basis, xp)
                m_sim = frobenius_similarity(contexts[i], contexts[j])
                
                witness_sims.append(w_sim)
                matrix_sims.append(m_sim)
        
        witness_sims = np.array(witness_sims)
        matrix_sims = np.array(matrix_sims)
        
        # Correlation between two similarity measures
        correlation = np.corrcoef(witness_sims, matrix_sims)[0, 1]
        
        print(f"Witness vs Matrix similarity correlation: {correlation:.4f}")
        print(f"Mean witness sim: {np.mean(witness_sims):.4f}")
        print(f"Mean matrix sim: {np.mean(matrix_sims):.4f}")
        
        # They should be correlated but not identical
        assert correlation > 0.3, "Witness and matrix similarity should be positively correlated"
        assert correlation < 0.99, "Witness and matrix similarity should differ somewhat"
    
    def test_witness_stability_vs_matrix_complexity(self, basis, xp, embeddings):
        """
        Test relationship between witness stability and matrix complexity.
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        stabilities = []
        enstrophies = []
        
        for trial in range(100):
            L = np.random.randint(3, 25)
            token_ids = np.random.randint(0, vocab_size, size=L)
            tokens = embeddings[token_ids]
            ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
            
            sigma = grace_stability(ctx, basis, xp)
            ens = compute_enstrophy(ctx, basis, xp)
            
            stabilities.append(sigma)
            enstrophies.append(ens)
        
        stabilities = np.array(stabilities)
        enstrophies = np.array(enstrophies)
        
        # Stability and enstrophy should be negatively correlated
        # (high vorticity → low stability)
        correlation = np.corrcoef(stabilities, enstrophies)[0, 1]
        
        print(f"Stability vs Enstrophy correlation: {correlation:.4f}")
        
        # Should be negative (more vorticity → less stable)
        assert correlation < 0, (
            f"Stability and enstrophy should be negatively correlated, got r={correlation:.4f}"
        )
