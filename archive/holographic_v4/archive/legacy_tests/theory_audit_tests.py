"""
Theory Audit Tests — Verify Implementation Against Theory
==========================================================

Each test validates ONE specific theoretical claim from rhnsclifford.md
by checking what the ACTUAL IMPLEMENTATION does.

THEORY CLAIMS TO TEST:
1. Storage: attractor[context] = embedding[target] (store TARGET, not context)
2. Learning rate: φ⁻¹ (not φ⁻²)
3. Learning rule: direct lerp, NOT Grace-then-mix
4. Grace: ONLY in forward/evolution, NOT in learning
5. No dynamic rate modulation (rates are fixed by theory)
6. Equilibrium dynamics: system converges, equilibrium IS output

Usage:
    python holographic_v4/theory_audit_tests.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import Dict, List, Tuple

# Constants
PHI = 1.618033988749894848204586834365638118
PHI_INV = 0.618033988749894848204586834365638118
PHI_INV_SQ = 0.381966011250105151795413165634361882


def test_1_storage_is_target():
    """
    THEORY (lines 23-24, 360-361):
        def learn(context, target):
            attractor[context] = embedding[target]  # Direct association
    
    Test: After train_step on NEW context, stored attractor should equal target_emb.
    """
    print("\n" + "="*60)
    print("TEST 1: Storage should be TARGET embedding")
    print("="*60)
    
    from holographic_v4 import TheoryTrueModel
    from holographic_v4.algebra import frobenius_similarity
    
    xp = np
    model = TheoryTrueModel(
        vocab_size=100,
        context_size=3,
        max_attractors=100,
        noise_std=0.3,
        use_binding=False,
        seed=42,
    )
    
    # Train on a new context
    context = [10, 20, 30]
    target = 50
    
    model.train_step(context, target)
    
    # Get what was stored via holographic retrieval
    ctx_rep = model.compute_context_representation(context)
    stored, _, _, _ = model.holographic_memory.retrieve(ctx_rep)
    
    # Get target embedding
    target_emb = model.get_embedding(target)
    
    # Get context representation (what OLD impl stored)
    ctx_rep = model.compute_context(context)
    
    # Check similarities
    sim_stored_to_target = frobenius_similarity(stored, target_emb, xp)
    sim_stored_to_context = frobenius_similarity(stored, ctx_rep, xp)
    
    print(f"\n  After train_step on NEW context:")
    print(f"    Similarity(stored, target_emb) = {sim_stored_to_target:.4f}")
    print(f"    Similarity(stored, context_rep) = {sim_stored_to_context:.4f}")
    
    # THEORY-TRUE: stored should be ~identical to target_emb
    correct = sim_stored_to_target > 0.99
    
    if correct:
        print(f"\n  ✓ THEORY-TRUE: Stores TARGET embedding")
    else:
        print(f"\n  ✗ DEVIATION: Stores context (sim to target = {sim_stored_to_target:.4f})")
    
    return correct


def test_2_learning_rate_is_phi_inv():
    """
    THEORY (line 111):
        attractor[context] = lerp(attractor[context], target_matrix, φ⁻¹)
    
    Test: After multiple updates to same context, convergence rate should be φ⁻¹.
    """
    print("\n" + "="*60)
    print("TEST 2: Learning rate should be φ⁻¹ (not φ⁻²)")
    print("="*60)
    
    from holographic_v4 import TheoryTrueModel
    from holographic_v4.algebra import frobenius_similarity
    
    xp = np
    model = TheoryTrueModel(
        vocab_size=100,
        context_size=3,
        max_attractors=100,
        noise_std=0.5,  # Higher noise for distinct embeddings
        use_binding=False,
        seed=42,
    )
    
    # Train same context with target_A several times
    context = [10, 20, 30]
    target_a = 50
    
    for _ in range(5):
        model.train_step(context, target_a)
    
    # Get current state via holographic retrieval
    ctx_rep = model.compute_context_representation(context)
    state_before, _, _, _ = model.holographic_memory.retrieve(ctx_rep)
    state_before = state_before.copy()
    
    # Now train with different target_B
    target_b = 60
    target_b_emb = model.get_embedding(target_b)
    
    model.train_step(context, target_b)
    
    state_after, _, _, _ = model.holographic_memory.retrieve(ctx_rep)
    state_after = state_after.copy()
    
    # Calculate actual mixing rate
    # state_after = (1 - rate) * state_before + rate * target_b
    # So: rate = (state_after - state_before) / (target_b - state_before)
    # We can estimate rate from the similarity change
    
    sim_before_to_b = frobenius_similarity(state_before, target_b_emb, xp)
    sim_after_to_b = frobenius_similarity(state_after, target_b_emb, xp)
    
    # If rate = φ⁻¹ ≈ 0.618, the similarity to target should increase by ~φ⁻¹ of the gap
    gap_before = 1.0 - sim_before_to_b
    improvement = sim_after_to_b - sim_before_to_b
    
    # Estimated rate
    if gap_before > 0.01:
        estimated_rate = improvement / gap_before
    else:
        estimated_rate = 0.0
    
    print(f"\n  After update with new target:")
    print(f"    Sim to new target before: {sim_before_to_b:.4f}")
    print(f"    Sim to new target after:  {sim_after_to_b:.4f}")
    print(f"    Improvement: {improvement:.4f}")
    print(f"    Estimated rate: {estimated_rate:.4f}")
    print(f"    Theory rate φ⁻¹: {PHI_INV:.4f}")
    print(f"    Old rate φ⁻²: {PHI_INV_SQ:.4f}")
    
    # Check if rate is closer to φ⁻¹ than to φ⁻²
    error_to_phi_inv = abs(estimated_rate - PHI_INV)
    error_to_phi_inv_sq = abs(estimated_rate - PHI_INV_SQ)
    
    correct = error_to_phi_inv < error_to_phi_inv_sq
    
    if correct:
        print(f"\n  ✓ THEORY-TRUE: Rate ≈ φ⁻¹ (error: {error_to_phi_inv:.4f})")
    else:
        print(f"\n  ✗ DEVIATION: Rate closer to φ⁻² (error: {error_to_phi_inv_sq:.4f})")
    
    return correct


def test_3_learning_is_direct_lerp():
    """
    THEORY (line 111):
        attractor[context] = lerp(attractor[context], target_matrix, φ⁻¹)
    
    This is DIRECT lerp: (1 - φ⁻¹) * attractor + φ⁻¹ * target
    NOT: (1 - rate) * Grace(attractor) + rate * target
    
    Test: After update, result should match direct lerp (not Grace-first).
    """
    print("\n" + "="*60)
    print("TEST 3: Learning should be DIRECT lerp (no Grace)")
    print("="*60)
    
    from holographic_v4 import TheoryTrueModel
    from holographic_v4.algebra import frobenius_similarity, grace_operator
    
    xp = np
    model = TheoryTrueModel(
        vocab_size=100,
        context_size=3,
        max_attractors=100,
        noise_std=0.5,
        use_binding=False,
        seed=42,
    )
    
    # Train once to establish attractor
    context = [10, 20, 30]
    target_a = 50
    model.train_step(context, target_a)
    
    # Get state before second update via holographic retrieval
    ctx_rep = model.compute_context_representation(context)
    state_before, _, _, _ = model.holographic_memory.retrieve(ctx_rep)
    state_before = state_before.copy()
    
    # Train with different target
    target_b = 60
    target_b_emb = model.get_embedding(target_b)
    model.train_step(context, target_b)
    
    state_after, _, _, _ = model.holographic_memory.retrieve(ctx_rep)
    state_after = state_after.copy()
    
    # What SHOULD happen (theory): direct lerp
    expected_direct = (1 - PHI_INV) * state_before + PHI_INV * target_b_emb
    
    # What WOULD happen with Grace-first (old impl):
    graced = grace_operator(state_before, model.basis, xp)
    expected_grace = (1 - PHI_INV_SQ) * graced + PHI_INV_SQ * target_b_emb
    
    # Check which is closer
    sim_to_direct = frobenius_similarity(state_after, expected_direct, xp)
    sim_to_grace = frobenius_similarity(state_after, expected_grace, xp)
    
    print(f"\n  After second update:")
    print(f"    Similarity to direct lerp result: {sim_to_direct:.4f}")
    print(f"    Similarity to Grace-first result: {sim_to_grace:.4f}")
    
    correct = sim_to_direct > sim_to_grace
    
    if correct:
        print(f"\n  ✓ THEORY-TRUE: Uses direct lerp (sim = {sim_to_direct:.4f})")
    else:
        print(f"\n  ✗ DEVIATION: Uses Grace-first (sim = {sim_to_grace:.4f})")
    
    return correct


def test_4_no_dynamic_rate_modulation():
    """
    THEORY: Grace scales are FIXED by self-consistency (Λ² = Λ + 1).
    
    Test: Learning rate should be constant across training, not modulated.
    We verify by measuring rate early vs late in training.
    """
    print("\n" + "="*60)
    print("TEST 4: Rates should be FIXED (no dynamic modulation)")
    print("="*60)
    
    from holographic_v4 import TheoryTrueModel
    from holographic_v4.algebra import frobenius_similarity
    
    xp = np
    
    def measure_rate(model, context, target_a, target_b):
        """Measure the effective learning rate for a context."""
        # First train with target_a
        model.train_step(context, target_a)
        
        ctx_rep = model.compute_context_representation(context)
        state_before, _, _, _ = model.holographic_memory.retrieve(ctx_rep)
        state_before = state_before.copy()
        
        # Then train with target_b
        target_b_emb = model.get_embedding(target_b)
        model.train_step(context, target_b)
        
        state_after, _, _, _ = model.holographic_memory.retrieve(ctx_rep)
        state_after = state_after.copy()
        
        # Calculate effective rate
        sim_before = frobenius_similarity(state_before, target_b_emb, xp)
        sim_after = frobenius_similarity(state_after, target_b_emb, xp)
        
        gap = 1.0 - sim_before
        improvement = sim_after - sim_before
        return improvement / max(gap, 0.01)
    
    # Test 1: Measure rate early in training
    model_early = TheoryTrueModel(
        vocab_size=100, context_size=3, max_attractors=1000,
        noise_std=0.5, use_binding=False, seed=42,
    )
    rate_early = measure_rate(model_early, [10, 20, 30], 50, 60)
    
    # Test 2: Measure rate after many training samples
    model_late = TheoryTrueModel(
        vocab_size=100, context_size=3, max_attractors=1000,
        noise_std=0.5, use_binding=False, seed=42,
    )
    # Train on many different contexts first
    for i in range(200):
        ctx = [i % 100, (i+1) % 100, (i+2) % 100]
        tgt = (i*7) % 100
        model_late.train_step(ctx, tgt)
    
    # Measure rate late
    rate_late = measure_rate(model_late, [77, 78, 79], 50, 60)
    
    print(f"\n  Measuring rate consistency:")
    print(f"    Rate early in training: {rate_early:.4f}")
    print(f"    Rate late in training:  {rate_late:.4f}")
    print(f"    Theory rate φ⁻¹: {PHI_INV:.4f}")
    print(f"    Vorticity (late): {model_late.total_vorticity / model_late.train_samples:.4f}")
    
    # Both rates should be close to φ⁻¹
    error_early = abs(rate_early - PHI_INV)
    error_late = abs(rate_late - PHI_INV)
    rate_consistent = abs(rate_early - rate_late) < 0.1
    
    correct = error_early < 0.1 and error_late < 0.1 and rate_consistent
    
    if correct:
        print(f"\n  ✓ THEORY-TRUE: Rate is fixed at φ⁻¹ (early err: {error_early:.4f}, late err: {error_late:.4f})")
    else:
        print(f"\n  ✗ DEVIATION: Rate varies (early: {rate_early:.4f}, late: {rate_late:.4f})")
    
    return correct


def test_5_grace_only_in_forward():
    """
    THEORY: Grace should only appear in forward pass (evolve_to_equilibrium),
    NOT in the learning update rule.
    
    Test: Verify by checking that stored attractors preserve all grades
    (Grace would damp higher grades).
    """
    print("\n" + "="*60)
    print("TEST 5: Grace belongs in FORWARD pass only")
    print("="*60)
    
    from holographic_v4 import TheoryTrueModel
    from holographic_v4.algebra import decompose_to_coefficients
    from holographic_v4.constants import GRADE_INDICES
    
    xp = np
    model = TheoryTrueModel(
        vocab_size=100,
        context_size=3,
        max_attractors=100,
        noise_std=0.5,
        use_binding=False,
        seed=42,
    )
    
    # Store a target
    context = [10, 20, 30]
    target = 50
    target_emb = model.get_embedding(target)
    
    # Get target's grade energies
    target_coeffs = decompose_to_coefficients(target_emb, model.basis, xp)
    target_grade2_energy = float(np.sum(target_coeffs[GRADE_INDICES[2]]**2))
    
    model.train_step(context, target)
    
    # Get stored attractor's grade energies via holographic retrieval
    ctx_rep = model.compute_context_representation(context)
    stored, _, _, _ = model.holographic_memory.retrieve(ctx_rep)
    
    stored_coeffs = decompose_to_coefficients(stored, model.basis, xp)
    stored_grade2_energy = float(np.sum(stored_coeffs[GRADE_INDICES[2]]**2))
    
    # If Grace was applied during learning, grade-2 would be damped
    # Grade-2 should be preserved if learning is direct (no Grace)
    
    print(f"\n  Grade-2 (bivector) energy:")
    print(f"    Target embedding: {target_grade2_energy:.4f}")
    print(f"    Stored attractor: {stored_grade2_energy:.4f}")
    
    preservation = stored_grade2_energy / max(target_grade2_energy, 0.001)
    
    # If Grace was NOT applied, preservation should be ~100%
    # If Grace WAS applied, preservation would be ~φ⁻⁴ ≈ 14.6%
    correct = preservation > 0.9  # At least 90% preserved
    
    if correct:
        print(f"\n  ✓ THEORY-TRUE: Grade-2 preserved ({preservation*100:.1f}%) - no Grace in learning")
    else:
        print(f"\n  ✗ DEVIATION: Grade-2 damped ({preservation*100:.1f}%) - Grace applied in learning")
    
    return correct


def test_6_equilibrium_convergence():
    """
    THEORY (lines 346-349):
        The system finds EQUILIBRIUM, not predictions.
        The equilibrium IS the output.
        
    Test: Grace flow should converge to a stable fixed point.
    """
    print("\n" + "="*60)
    print("TEST 6: Equilibrium convergence (Grace flow)")
    print("="*60)
    
    from holographic_v4.algebra import (
        build_clifford_basis,
        initialize_embeddings_identity,
        grace_flow,
        frobenius_similarity,
    )
    
    xp = np
    basis = build_clifford_basis(xp)
    embeddings = initialize_embeddings_identity(10, noise_std=0.5, seed=42, xp=xp)
    
    # Start from a context representation
    initial = embeddings[0].copy()
    attractor = embeddings[5].copy()
    
    # Evolve for different step counts
    results = []
    for steps in [1, 5, 10, 20, 50]:
        final = grace_flow(initial, attractor, basis, steps=steps, xp=xp)
        sim_to_attractor = frobenius_similarity(final, attractor, xp)
        results.append((steps, sim_to_attractor))
    
    print(f"\n  Grace flow convergence (initial → attractor):")
    for steps, sim in results:
        print(f"    {steps:2d} steps: similarity = {sim:.4f}")
    
    # Check convergence
    converging = results[-1][1] > results[0][1]
    stable = abs(results[-1][1] - results[-2][1]) < 0.01
    
    correct = converging and stable
    
    if correct:
        print(f"\n  ✓ THEORY-TRUE: Converges to stable equilibrium")
    else:
        print(f"\n  ✗ DEVIATION: Does not converge properly")
    
    return correct


def test_7_adaptive_similarity_decides():
    """
    THEORY-DERIVED: Similarity selection is parsimonious.
    
    The system decides between Frobenius and Quotient based on enstrophy:
    - Low enstrophy (near-identity): Frobenius is accurate (and fast)
    - High enstrophy (differentiated): Quotient structure matters
    
    This is theory-derived because:
    - Enstrophy measures bivector energy (grade-2 content)
    - Near-identity matrices have trivial quotient structure
    - The threshold φ⁻² × 0.05 connects to spectral gap
    """
    print("\n" + "="*60)
    print("TEST 7: Adaptive similarity (system decides)")
    print("="*60)
    
    from holographic_v4.quotient import (
        adaptive_similarity, compute_enstrophy, ENSTROPHY_THRESHOLD,
        quotient_similarity
    )
    from holographic_v4.algebra import (
        build_clifford_basis, initialize_embeddings_identity
    )
    
    xp = np
    basis = build_clifford_basis(xp)
    
    print(f"\n  Theory-derived threshold: φ⁻² × 0.05 = {ENSTROPHY_THRESHOLD:.6f}")
    
    # Test low enstrophy → should use Frobenius
    embs_low = initialize_embeddings_identity(10, noise_std=0.1, seed=42, xp=xp)
    ens_low = (compute_enstrophy(embs_low[0], basis, xp) + 
               compute_enstrophy(embs_low[5], basis, xp)) / 2
    
    frob_sim = float(xp.sum(embs_low[0] * embs_low[5]))
    adapt_sim_low = adaptive_similarity(embs_low[0], embs_low[5], basis, xp)
    
    low_correct = (ens_low < ENSTROPHY_THRESHOLD and 
                   abs(frob_sim - adapt_sim_low) < 0.001)
    
    print(f"\n  LOW enstrophy (noise=0.1):")
    print(f"    Enstrophy: {ens_low:.6f} (< threshold)")
    print(f"    Decision: Frobenius")
    print(f"    Match: {'✓' if low_correct else '✗'}")
    
    # Test high enstrophy → should use Quotient
    embs_high = initialize_embeddings_identity(10, noise_std=0.5, seed=42, xp=xp)
    ens_high = (compute_enstrophy(embs_high[0], basis, xp) + 
                compute_enstrophy(embs_high[5], basis, xp)) / 2
    
    quot_sim = quotient_similarity(embs_high[0], embs_high[5], basis, xp)
    adapt_sim_high = adaptive_similarity(embs_high[0], embs_high[5], basis, xp)
    
    high_correct = (ens_high > ENSTROPHY_THRESHOLD and 
                    abs(quot_sim - adapt_sim_high) < 0.001)
    
    print(f"\n  HIGH enstrophy (noise=0.5):")
    print(f"    Enstrophy: {ens_high:.6f} (> threshold)")
    print(f"    Decision: Quotient")
    print(f"    Match: {'✓' if high_correct else '✗'}")
    
    correct = low_correct and high_correct
    
    if correct:
        print(f"\n  ✓ THEORY-TRUE: System correctly decides based on enstrophy")
    else:
        print(f"\n  ✗ DEVIATION: Adaptive similarity not working as expected")
    
    return correct


def run_all_audit_tests():
    """Run all theory audit tests."""
    print("\n" + "="*70)
    print("THEORY AUDIT: Implementation vs rhnsclifford.md")
    print("="*70)
    
    tests = [
        ("Storage is TARGET", test_1_storage_is_target),
        ("Learning rate is φ⁻¹", test_2_learning_rate_is_phi_inv),
        ("Learning is direct lerp", test_3_learning_is_direct_lerp),
        ("No dynamic rate modulation", test_4_no_dynamic_rate_modulation),
        ("Grace only in forward pass", test_5_grace_only_in_forward),
        ("Equilibrium convergence", test_6_equilibrium_convergence),
        ("Adaptive similarity decides", test_7_adaptive_similarity_decides),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = passed
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ THEORY-TRUE" if passed else "✗ DEVIATION"
        print(f"  {status}: {name}")
    
    passed_count = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  {passed_count}/{total} tests match theory")
    
    if passed_count < total:
        print(f"\n  FIXES NEEDED:")
        for name, passed in results.items():
            if not passed:
                print(f"    - {name}")
    
    return results


if __name__ == "__main__":
    run_all_audit_tests()
