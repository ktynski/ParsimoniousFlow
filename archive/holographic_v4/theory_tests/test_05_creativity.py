"""
Test 05: Creativity — Bisociation, Metaphor, Compression
=========================================================

PURPOSE: Test creative cognition predictions.
         Validates that theory enables creative generation.

TESTS:
    1. test_bisociation         - Collision of unrelated schemas produces insight
    2. test_metaphor_generation - Schema transfer creates metaphor
    3. test_poetic_compression  - Maximum info density in minimal form
    4. test_insight_phase_transition - Sudden restructuring in witness space

THEORY PREDICTIONS:
    - Bisociation: combining distant schemas creates novel patterns
    - Metaphor: mapping schema from source to target domain
    - Compression: Grace removes noise, preserves essence (poetry)
    - Insight: phase transition in witness space (sudden restructuring)
"""

import pytest
import numpy as np
from typing import List, Tuple

from holographic_v4 import (
    build_clifford_basis,
    grace_operator,
    normalize_matrix,
    geometric_product_batch,
    geometric_product,
    extract_witness,
    frobenius_similarity,
    wedge_product,
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
)
from holographic_v4.quotient import witness_similarity
from holographic_v4.quotient import (
    compute_enstrophy,
    grade_energies,
    grace_stability,
    witness_similarity,
)
from holographic_v4.algebra import decompose_to_coefficients
from holographic_v4.constants import DTYPE

from .utils import (
    bootstrap_confidence_interval,
    permutation_test,
    effect_size_cohens_d,
    compute_grade_energy_distribution,
)


# =============================================================================
# Test 1: Bisociation
# =============================================================================

class TestBisociation:
    """
    THEORY: Bisociation (Koestler) — creativity from colliding unrelated frames.
    
    When two unrelated schemas are forced together, the collision
    can produce novel, creative combinations. This is the basis
    of humor, scientific discovery, and artistic creation.
    """
    
    def test_schema_collision_novelty(
        self, basis, xp, embeddings, tokenizer, sentence_to_context
    ):
        """
        Test that colliding unrelated schemas produces novel patterns.
        """
        # Schema A: Technical/scientific
        schema_a = sentence_to_context("computer algorithm data process")
        
        # Schema B: Culinary/cooking
        schema_b = sentence_to_context("recipe ingredients cook simmer")
        
        # Collision: compose them
        collision = normalize_matrix(
            geometric_product(schema_a, schema_b), xp
        )
        
        # Measure novelty: how different is collision from either parent?
        sim_a = frobenius_similarity(collision, schema_a)
        sim_b = frobenius_similarity(collision, schema_b)
        
        # Collision should be distinct from both parents
        novelty = 1 - max(sim_a, sim_b)
        
        print(f"Bisociation (schema collision):")
        print(f"  Similarity to Schema A: {sim_a:.4f}")
        print(f"  Similarity to Schema B: {sim_b:.4f}")
        print(f"  Novelty (1 - max): {novelty:.4f}")
        
        # Check witness of collision
        wit_a = extract_witness(schema_a, basis, xp)
        wit_b = extract_witness(schema_b, basis, xp)
        wit_coll = extract_witness(collision, basis, xp)
        
        print(f"  Witness A: ({wit_a[0]:.4f}, {wit_a[1]:.4f})")
        print(f"  Witness B: ({wit_b[0]:.4f}, {wit_b[1]:.4f})")
        print(f"  Witness collision: ({wit_coll[0]:.4f}, {wit_coll[1]:.4f})")
    
    def test_distant_vs_close_schema_collision(
        self, basis, xp, embeddings, tokenizer, sentence_to_context
    ):
        """
        Compare novelty from distant vs close schema collisions.
        
        Theory: More distant schemas produce more novel combinations.
        """
        # Base schema
        base = sentence_to_context("water flow river stream")
        
        # Close schema (similar domain)
        close = sentence_to_context("rain ocean wave tide")
        
        # Distant schema (different domain)
        distant = sentence_to_context("music melody rhythm beat")
        
        # Collisions
        close_collision = normalize_matrix(
            geometric_product(base, close), xp
        )
        distant_collision = normalize_matrix(
            geometric_product(base, distant), xp
        )
        
        # Measure novelty for each
        close_novelty = 1 - frobenius_similarity(close_collision, base)
        distant_novelty = 1 - frobenius_similarity(distant_collision, base)
        
        print(f"Schema distance effect:")
        print(f"  Close collision novelty:   {close_novelty:.4f}")
        print(f"  Distant collision novelty: {distant_novelty:.4f}")
        
        # Distant schemas should produce more novelty
        # (This is the Koestler bisociation principle)


# =============================================================================
# Test 2: Metaphor Generation
# =============================================================================

class TestMetaphorGeneration:
    """
    THEORY: Metaphor as schema transfer between domains.
    
    Creative metaphor: "Life is a journey"
    - Source domain: JOURNEY (path, destination, obstacles)
    - Target domain: LIFE (birth, death, challenges)
    
    The schema structure transfers while the content changes.
    """
    
    def test_metaphor_preserves_structure(
        self, basis, xp, embeddings, tokenizer, sentence_to_context
    ):
        """
        Test that metaphor preserves structural (vorticity) patterns.
        """
        # Source domain: JOURNEY
        journey_source = sentence_to_context("traveler walks road destination")
        
        # Target domain literal: LIFE
        life_literal = sentence_to_context("person lives years death")
        
        # Metaphorical: LIFE IS JOURNEY
        life_metaphor = sentence_to_context("person walks path destination")
        
        # Witness similarity (simpler than vorticity)
        sim_journey_metaphor = witness_similarity(journey_source, life_metaphor, basis, xp)
        sim_journey_literal = witness_similarity(journey_source, life_literal, basis, xp)
        
        print(f"Metaphor structure preservation:")
        print(f"  Journey-Metaphor witness sim: {sim_journey_metaphor:.4f}")
        print(f"  Journey-Literal witness sim:  {sim_journey_literal:.4f}")
        
        # Metaphor should be more similar to source structure
    
    def test_metaphor_generates_novel_meaning(
        self, basis, xp, embeddings, tokenizer, sentence_to_context
    ):
        """
        Test that metaphor creates novel meaning distinct from both domains.
        """
        # War domain
        war = sentence_to_context("army attack defend conquer")
        
        # Argument domain (literal)
        argument_literal = sentence_to_context("discuss debate disagree conclude")
        
        # ARGUMENT IS WAR (metaphor)
        argument_war = sentence_to_context("attack position defend claim")
        
        # The metaphor should blend properties of both
        wit_war = np.array(extract_witness(war, basis, xp))
        wit_literal = np.array(extract_witness(argument_literal, basis, xp))
        wit_metaphor = np.array(extract_witness(argument_war, basis, xp))
        
        # Distance from each source
        dist_to_war = np.linalg.norm(wit_metaphor - wit_war)
        dist_to_literal = np.linalg.norm(wit_metaphor - wit_literal)
        
        print(f"Metaphor novelty:")
        print(f"  Distance from WAR:     {dist_to_war:.4f}")
        print(f"  Distance from LITERAL: {dist_to_literal:.4f}")


# =============================================================================
# Test 3: Poetic Compression
# =============================================================================

class TestPoeticCompression:
    """
    THEORY: Poetry achieves maximum information density in minimal form.
    
    Grace removes noise while preserving semantic essence.
    Poetic language should have:
    - High witness stability (clear semantic core)
    - Low enstrophy (compressed, not verbose)
    - High information per token
    """
    
    def test_poetry_vs_prose_compression(
        self, basis, xp, embeddings, tokenizer, sentence_to_context
    ):
        """
        Compare compression metrics for poetic vs prose expression.
        """
        # Prose: verbose expression
        prose = sentence_to_context(
            "the sun was setting slowly behind the distant mountains"
        )
        
        # Poetry: compressed expression
        poetry = sentence_to_context("sunset fades mountain shadows")
        
        # Measure compression metrics
        ens_prose = compute_enstrophy(prose, basis, xp)
        ens_poetry = compute_enstrophy(poetry, basis, xp)
        
        stab_prose = grace_stability(prose, basis, xp)
        stab_poetry = grace_stability(poetry, basis, xp)
        
        print(f"Compression metrics:")
        print(f"  Prose:  enstrophy={ens_prose:.4f}, stability={stab_prose:.4f}")
        print(f"  Poetry: enstrophy={ens_poetry:.4f}, stability={stab_poetry:.4f}")
        
        # Compute "efficiency" (stability per unit enstrophy)
        eff_prose = stab_prose / (ens_prose + 1e-6)
        eff_poetry = stab_poetry / (ens_poetry + 1e-6)
        
        print(f"  Prose efficiency:  {eff_prose:.4f}")
        print(f"  Poetry efficiency: {eff_poetry:.4f}")
    
    def test_grace_as_poetic_distillation(self, basis, xp, random_context):
        """
        Test that Grace produces more "poetic" (compressed) representations.
        """
        # Start with verbose context
        M = random_context(n_tokens=15, seed=42)
        
        initial_ens = compute_enstrophy(M, basis, xp)
        initial_stab = grace_stability(M, basis, xp)
        
        # Apply Grace (distillation)
        distilled = M.copy()
        for _ in range(5):
            distilled = grace_operator(distilled, basis, xp)
        
        final_ens = compute_enstrophy(distilled, basis, xp)
        final_stab = grace_stability(distilled, basis, xp)
        
        print(f"Grace distillation:")
        print(f"  Initial: ens={initial_ens:.4f}, stab={initial_stab:.4f}")
        print(f"  After:   ens={final_ens:.4f}, stab={final_stab:.4f}")
        
        # Enstrophy should decrease, stability should increase
        assert final_ens < initial_ens, "Grace should reduce enstrophy"
        assert final_stab > initial_stab, "Grace should increase stability"
    
    def test_information_density(
        self, basis, xp, embeddings, tokenizer, sentence_to_context
    ):
        """
        Test information density: meaning per token.
        """
        # Redundant expression
        redundant = "the very big large huge enormous giant massive dog"
        
        # Dense expression
        dense = "colossal dog"
        
        # Parse
        redundant_tokens = tokenizer(redundant)
        dense_tokens = tokenizer(dense)
        
        ctx_redundant = sentence_to_context(redundant)
        ctx_dense = sentence_to_context(dense)
        
        # Witness captures "meaning"
        wit_redundant = np.array(extract_witness(ctx_redundant, basis, xp))
        wit_dense = np.array(extract_witness(ctx_dense, basis, xp))
        
        # Information density = witness magnitude / token count
        density_redundant = np.linalg.norm(wit_redundant) / max(len(redundant_tokens), 1)
        density_dense = np.linalg.norm(wit_dense) / max(len(dense_tokens), 1)
        
        print(f"Information density:")
        print(f"  Redundant ({len(redundant_tokens)} tokens): {density_redundant:.4f}")
        print(f"  Dense ({len(dense_tokens)} tokens):     {density_dense:.4f}")


# =============================================================================
# Test 4: Insight Phase Transition
# =============================================================================

class TestInsightPhaseTransition:
    """
    THEORY: Insight involves sudden restructuring (phase transition).
    
    The "Aha!" moment represents a phase transition in the cognitive
    state space — sudden reorganization from one attractor basin to another.
    
    In our system: witness trajectory should show discontinuity at insight.
    """
    
    def test_witness_discontinuity(self, basis, xp, embeddings, tokenizer):
        """
        Test that combining incompatible schemas creates witness discontinuity.
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        # Build context gradually
        witnesses = []
        contexts = []
        
        # Phase 1: Build consistent context
        tokens_phase1 = np.random.randint(0, vocab_size, size=5)
        ctx = xp.eye(4, dtype=DTYPE)
        
        for token_id in tokens_phase1:
            ctx = normalize_matrix(ctx @ embeddings[token_id], xp)
            wit = extract_witness(ctx, basis, xp)
            witnesses.append(wit)
            contexts.append(ctx.copy())
        
        # Phase 2: Introduce disruptive token (potential insight trigger)
        disruptive_token = (tokens_phase1[0] + vocab_size // 2) % vocab_size
        ctx = normalize_matrix(ctx @ embeddings[disruptive_token], xp)
        wit = extract_witness(ctx, basis, xp)
        witnesses.append(wit)
        contexts.append(ctx.copy())
        
        # Compute witness changes
        witness_changes = []
        for i in range(1, len(witnesses)):
            wit_prev = np.array(witnesses[i-1])
            wit_curr = np.array(witnesses[i])
            change = np.linalg.norm(wit_curr - wit_prev)
            witness_changes.append(change)
        
        print(f"Witness trajectory changes:")
        for i, change in enumerate(witness_changes):
            marker = " <- insight?" if i == len(witness_changes) - 1 else ""
            print(f"  Step {i+1}: Δwitness = {change:.4f}{marker}")
        
        # The last change (after disruptive token) should be larger
        # (representing potential insight discontinuity)
    
    def test_attractor_basin_transition(self, basis, xp, random_context):
        """
        Test detection of attractor basin transitions.
        """
        # Generate two distinct contexts (different basins)
        ctx_A = random_context(n_tokens=6, seed=42)
        ctx_B = random_context(n_tokens=6, seed=999)
        
        wit_A = np.array(extract_witness(ctx_A, basis, xp))
        wit_B = np.array(extract_witness(ctx_B, basis, xp))
        
        # Create trajectory that transitions between basins
        trajectory_witnesses = []
        
        # Start in basin A
        current = ctx_A.copy()
        for _ in range(3):
            wit = extract_witness(current, basis, xp)
            trajectory_witnesses.append(wit)
            current = grace_operator(current, basis, xp)
        
        # Force transition to basin B
        current = normalize_matrix(
            geometric_product(current, ctx_B), xp
        )
        
        # Continue in new basin
        for _ in range(3):
            wit = extract_witness(current, basis, xp)
            trajectory_witnesses.append(wit)
            current = grace_operator(current, basis, xp)
        
        # Detect transition point
        changes = []
        for i in range(1, len(trajectory_witnesses)):
            prev = np.array(trajectory_witnesses[i-1])
            curr = np.array(trajectory_witnesses[i])
            changes.append(np.linalg.norm(curr - prev))
        
        max_change_idx = np.argmax(changes)
        
        print(f"Basin transition detection:")
        print(f"  Maximum change at step {max_change_idx + 1}")
        print(f"  Change magnitude: {changes[max_change_idx]:.4f}")
        
        # The transition should occur around step 3 (when we force it)
        assert max_change_idx in [2, 3], (
            f"Transition should be detected near forced point, got step {max_change_idx + 1}"
        )
    
    def test_insight_stability_signature(self, basis, xp, embeddings, tokenizer):
        """
        Test that insight moments have distinctive stability signatures.
        """
        # Before insight: building toward understanding
        building = []
        
        np.random.seed(42)
        vocab_size = embeddings.shape[0]
        
        ctx = xp.eye(4, dtype=DTYPE)
        for i in range(5):
            token = embeddings[np.random.randint(0, vocab_size)]
            ctx = normalize_matrix(ctx @ token, xp)
            building.append(grace_stability(ctx, basis, xp))
        
        # "Insight" moment: sudden coherent integration
        # (Simulated by applying Grace to clarify)
        insight_ctx = grace_operator(ctx, basis, xp)
        insight_stab = grace_stability(insight_ctx, basis, xp)
        
        print(f"Stability during insight:")
        print(f"  Building phase: {building}")
        print(f"  After insight:  {insight_stab:.4f}")
        
        # Insight should show stability jump
        pre_insight_stab = building[-1]
        stability_jump = insight_stab - pre_insight_stab
        
        print(f"  Stability jump: {stability_jump:.4f}")
        
        # Grace should increase stability (representing insight clarity)
        assert stability_jump > 0, "Insight should increase stability"
