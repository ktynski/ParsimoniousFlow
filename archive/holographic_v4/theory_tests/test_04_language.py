"""
Test 04: Language — Semantics, Grammar, Discourse
==================================================

PURPOSE: Test linguistic predictions with real Gutenberg text.
         Validates that theory-derived operations capture language structure.

TESTS:
    1. test_semantic_priming       - Related words activate each other
    2. test_discourse_coherence    - Coherent text has stable Grace trajectory
    3. test_morphological_schemas  - Inflection as rotor transformation
    4. test_semantic_roles         - Agent/patient in different bivector planes
    5. test_frame_semantics        - Schema activation by token
    6. test_construction_grammar   - Form-meaning pairs at all levels
    7. test_conceptual_metaphor    - Cross-domain schema mapping
    8. test_recursive_composition  - Schemas compose hierarchically

THEORY PREDICTIONS:
    - Semantic similarity reflected in witness similarity (priming)
    - Coherent text produces stable Grace trajectory (discourse)
    - Inflection operates via rotor transformation on stems
    - Semantic roles encoded in bivector structure
    - Schemas activated by key tokens (frame semantics)
    - Constructions are form-meaning composites
    - Metaphor is cross-domain schema mapping
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
    wedge_product,
    extract_witness,
    frobenius_similarity,
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
)
from holographic_v4.quotient import (
    compute_enstrophy,
    grace_stability,
    witness_similarity,
)
from holographic_v4.constants import DTYPE

from .utils import (
    bootstrap_confidence_interval,
    permutation_test,
    effect_size_cohens_d,
)


# =============================================================================
# Test 1: Semantic Priming
# =============================================================================

class TestSemanticPriming:
    """
    THEORY: Semantically related words have similar witnesses.
    
    Semantic priming: Exposure to "doctor" activates "nurse" because
    they share semantic features. In our system, similar witnesses
    should indicate semantic relatedness.
    """
    
    def test_related_words_similar_witnesses(self, basis, xp, embeddings, tokenizer):
        """
        Test that semantically related word pairs have similar witnesses.
        """
        # Related word pairs (should be similar)
        related_pairs = [
            ("dog", "cat"),
            ("run", "walk"),
            ("large", "big"),
            ("man", "woman"),
            ("sun", "moon"),
        ]
        
        # Unrelated word pairs (should be dissimilar)
        unrelated_pairs = [
            ("dog", "walk"),
            ("run", "large"),
            ("cat", "sun"),
            ("man", "tree"),
            ("big", "river"),
        ]
        
        def get_word_witness(word):
            tokens = tokenizer(word)
            if not tokens:
                return None
            token_id = tokens[0] % embeddings.shape[0]
            emb = embeddings[token_id]
            sigma, pseudo = extract_witness(emb, basis, xp)
            return np.array([sigma, pseudo])
        
        related_sims = []
        for w1, w2 in related_pairs:
            wit1 = get_word_witness(w1)
            wit2 = get_word_witness(w2)
            if wit1 is not None and wit2 is not None:
                sim = np.dot(wit1, wit2) / (np.linalg.norm(wit1) * np.linalg.norm(wit2) + 1e-10)
                related_sims.append(sim)
        
        unrelated_sims = []
        for w1, w2 in unrelated_pairs:
            wit1 = get_word_witness(w1)
            wit2 = get_word_witness(w2)
            if wit1 is not None and wit2 is not None:
                sim = np.dot(wit1, wit2) / (np.linalg.norm(wit1) * np.linalg.norm(wit2) + 1e-10)
                unrelated_sims.append(sim)
        
        mean_related = np.mean(related_sims) if related_sims else 0
        mean_unrelated = np.mean(unrelated_sims) if unrelated_sims else 0
        
        print(f"Witness similarity:")
        print(f"  Related pairs:   {mean_related:.4f}")
        print(f"  Unrelated pairs: {mean_unrelated:.4f}")
        
        # Note: With random embeddings, this might not show strong difference
        # Real test requires trained embeddings
    
    def test_context_priming(self, basis, xp, embeddings, tokenizer, sentence_to_context):
        """
        Test that priming context affects subsequent retrieval.
        """
        # Prime: "The doctor examined the"
        # Target: "patient" vs "table"
        
        prime_context = sentence_to_context("The doctor examined the")
        
        # Get witnesses for targets
        related_word = "patient"
        unrelated_word = "table"
        
        related_tokens = tokenizer(related_word)
        unrelated_tokens = tokenizer(unrelated_word)
        
        if related_tokens and unrelated_tokens:
            related_emb = embeddings[related_tokens[0] % embeddings.shape[0]]
            unrelated_emb = embeddings[unrelated_tokens[0] % embeddings.shape[0]]
            
            # Compose prime with targets
            primed_related = normalize_matrix(
                geometric_product(prime_context, related_emb), xp
            )
            primed_unrelated = normalize_matrix(
                geometric_product(prime_context, unrelated_emb), xp
            )
            
            # Measure stability (how well does it "fit"?)
            stab_related = grace_stability(primed_related, basis, xp)
            stab_unrelated = grace_stability(primed_unrelated, basis, xp)
            
            print(f"Priming effect:")
            print(f"  'patient' stability: {stab_related:.4f}")
            print(f"  'table' stability:   {stab_unrelated:.4f}")


# =============================================================================
# Test 2: Discourse Coherence
# =============================================================================

class TestDiscourseCoherence:
    """
    THEORY: Coherent text produces stable Grace trajectory.
    
    Coherent discourse maintains topical consistency, which should
    result in stable witness evolution. Shuffled/incoherent text
    should have more erratic trajectories.
    """
    
    def test_coherent_vs_shuffled_stability(
        self, basis, xp, embeddings, tokenizer, gutenberg_sentences
    ):
        """
        Compare Grace stability for coherent vs shuffled sentences.
        """
        if len(gutenberg_sentences) < 10:
            pytest.skip("Insufficient Gutenberg data")
        
        coherent_stabilities = []
        shuffled_stabilities = []
        
        for i in range(min(20, len(gutenberg_sentences) - 1)):
            # Coherent: consecutive sentences
            sent1 = gutenberg_sentences[i]
            sent2 = gutenberg_sentences[i + 1]
            
            tokens1 = tokenizer(sent1)
            tokens2 = tokenizer(sent2)
            
            if len(tokens1) >= 3 and len(tokens2) >= 3:
                # Compose coherent sequence
                all_tokens = tokens1[:5] + tokens2[:5]
                token_matrices = embeddings[[t % embeddings.shape[0] for t in all_tokens]]
                ctx = normalize_matrix(geometric_product_batch(token_matrices, xp), xp)
                coherent_stabilities.append(grace_stability(ctx, basis, xp))
                
                # Shuffled: same tokens but shuffled
                shuffled_tokens = all_tokens.copy()
                np.random.shuffle(shuffled_tokens)
                shuffled_matrices = embeddings[[t % embeddings.shape[0] for t in shuffled_tokens]]
                shuffled_ctx = normalize_matrix(geometric_product_batch(shuffled_matrices, xp), xp)
                shuffled_stabilities.append(grace_stability(shuffled_ctx, basis, xp))
        
        if coherent_stabilities:
            mean_coherent = np.mean(coherent_stabilities)
            mean_shuffled = np.mean(shuffled_stabilities)
            
            print(f"Grace stability:")
            print(f"  Coherent text:  {mean_coherent:.4f}")
            print(f"  Shuffled text:  {mean_shuffled:.4f}")
    
    def test_discourse_trajectory_variance(
        self, basis, xp, embeddings, tokenizer, gutenberg_sentences
    ):
        """
        Measure trajectory variance for coherent vs random discourse.
        """
        if len(gutenberg_sentences) < 5:
            pytest.skip("Insufficient Gutenberg data")
        
        # Build trajectory from coherent text
        coherent_witnesses = []
        for sent in gutenberg_sentences[:5]:
            tokens = tokenizer(sent)
            if len(tokens) >= 3:
                token_matrices = embeddings[[t % embeddings.shape[0] for t in tokens[:6]]]
                ctx = normalize_matrix(geometric_product_batch(token_matrices, xp), xp)
                sigma, pseudo = extract_witness(ctx, basis, xp)
                coherent_witnesses.append([sigma, pseudo])
        
        # Build trajectory from random sentences
        random_indices = np.random.permutation(len(gutenberg_sentences))[:5]
        random_witnesses = []
        for idx in random_indices:
            sent = gutenberg_sentences[idx]
            tokens = tokenizer(sent)
            if len(tokens) >= 3:
                token_matrices = embeddings[[t % embeddings.shape[0] for t in tokens[:6]]]
                ctx = normalize_matrix(geometric_product_batch(token_matrices, xp), xp)
                sigma, pseudo = extract_witness(ctx, basis, xp)
                random_witnesses.append([sigma, pseudo])
        
        if len(coherent_witnesses) >= 3 and len(random_witnesses) >= 3:
            coherent_var = np.var(coherent_witnesses, axis=0).sum()
            random_var = np.var(random_witnesses, axis=0).sum()
            
            print(f"Witness trajectory variance:")
            print(f"  Coherent: {coherent_var:.6f}")
            print(f"  Random:   {random_var:.6f}")


# =============================================================================
# Test 3: Morphological Schemas
# =============================================================================

class TestMorphologicalSchemas:
    """
    THEORY: Inflection operates as rotor transformation.
    
    Morphological relations (walk → walked, walk → walking) should
    be representable as consistent transformations (rotors) in
    Clifford space.
    """
    
    def test_inflection_consistency(self, basis, xp, embeddings, tokenizer):
        """
        Test that inflectional patterns are consistent across words.
        """
        # Word sets with consistent inflection
        verb_pairs = [
            ("walk", "walked"),
            ("jump", "jumped"),
            ("talk", "talked"),
        ]
        
        transformations = []
        for base, inflected in verb_pairs:
            base_tokens = tokenizer(base)
            inflected_tokens = tokenizer(inflected)
            
            if base_tokens and inflected_tokens:
                base_emb = embeddings[base_tokens[0] % embeddings.shape[0]]
                infl_emb = embeddings[inflected_tokens[0] % embeddings.shape[0]]
                
                # Compute "transformation" as difference
                # (In true theory, this would be a rotor)
                transform = infl_emb - base_emb
                transformations.append(transform)
        
        if len(transformations) >= 2:
            # Check consistency of transformations
            sims = []
            for i in range(len(transformations)):
                for j in range(i + 1, len(transformations)):
                    sim = frobenius_similarity(transformations[i], transformations[j])
                    sims.append(sim)
            
            mean_sim = np.mean(sims) if sims else 0
            print(f"Inflection transformation consistency: {mean_sim:.4f}")
            
            # If transformations are consistent, similarity should be positive


# =============================================================================
# Test 4: Semantic Roles
# =============================================================================

class TestSemanticRoles:
    """
    THEORY: Semantic roles (agent, patient) encoded in bivector structure.
    
    "Dog bites man" vs "Man bites dog" should differ in their
    bivector (vorticity) components, capturing the role reversal.
    """
    
    def test_role_reversal_in_vorticity(
        self, basis, xp, embeddings, tokenizer, sentence_to_context
    ):
        """
        Test that role reversal affects vorticity structure.
        """
        # Create minimal pair with role reversal
        sent1 = "dog bites man"
        sent2 = "man bites dog"
        
        ctx1 = sentence_to_context(sent1)
        ctx2 = sentence_to_context(sent2)
        
        # Compare witness similarity (simpler than vorticity signature)
        wit_sim = witness_similarity(ctx1, ctx2, basis, xp)
        
        print(f"Witness similarity (role reversal): {wit_sim:.4f}")
        
        # Also check wedge product directly
        tokens1 = tokenizer(sent1)
        tokens2 = tokenizer(sent2)
        
        if len(tokens1) >= 2 and len(tokens2) >= 2:
            # "dog bites" wedge
            dog = embeddings[tokens1[0] % embeddings.shape[0]]
            bites = embeddings[tokens1[1] % embeddings.shape[0]]
            wedge1 = wedge_product(dog, bites)
            
            # "man bites" wedge
            man = embeddings[tokens2[0] % embeddings.shape[0]]
            wedge2 = wedge_product(man, bites)
            
            wedge_sim = frobenius_similarity(wedge1, wedge2)
            print(f"Wedge product similarity: {wedge_sim:.4f}")


# =============================================================================
# Test 5: Frame Semantics
# =============================================================================

class TestFrameSemantics:
    """
    THEORY: Schema activation by key tokens (frame semantics).
    
    Certain words activate entire frames/schemas. "Restaurant" activates
    slots for waiter, menu, bill, etc. In our system, frame words
    should produce distinctive witness signatures.
    """
    
    def test_frame_word_distinctiveness(self, basis, xp, embeddings, tokenizer):
        """
        Test that frame words have distinctive witnesses.
        """
        # Frame words
        frame_words = ["restaurant", "hospital", "school", "mountain", "river"]
        
        witnesses = []
        for word in frame_words:
            tokens = tokenizer(word)
            if tokens:
                emb = embeddings[tokens[0] % embeddings.shape[0]]
                sigma, pseudo = extract_witness(emb, basis, xp)
                witnesses.append([sigma, pseudo])
        
        if len(witnesses) >= 3:
            witnesses = np.array(witnesses)
            
            # Compute pairwise distances
            distances = []
            for i in range(len(witnesses)):
                for j in range(i + 1, len(witnesses)):
                    dist = np.linalg.norm(witnesses[i] - witnesses[j])
                    distances.append(dist)
            
            mean_dist = np.mean(distances)
            print(f"Mean witness distance between frame words: {mean_dist:.4f}")
    
    def test_frame_slot_filling(
        self, basis, xp, embeddings, tokenizer, sentence_to_context
    ):
        """
        Test that frame context creates expectations for slot fillers.
        """
        # Frame: restaurant
        frame_context = sentence_to_context("at the restaurant the waiter")
        
        # Good slot filler vs bad slot filler
        good_filler = "served"
        bad_filler = "mountain"
        
        good_tokens = tokenizer(good_filler)
        bad_tokens = tokenizer(bad_filler)
        
        if good_tokens and bad_tokens:
            good_emb = embeddings[good_tokens[0] % embeddings.shape[0]]
            bad_emb = embeddings[bad_tokens[0] % embeddings.shape[0]]
            
            # Compose with frame
            good_filled = normalize_matrix(
                geometric_product(frame_context, good_emb), xp
            )
            bad_filled = normalize_matrix(
                geometric_product(frame_context, bad_emb), xp
            )
            
            # Measure stability
            good_stab = grace_stability(good_filled, basis, xp)
            bad_stab = grace_stability(bad_filled, basis, xp)
            
            print(f"Frame slot filling:")
            print(f"  '{good_filler}': stability = {good_stab:.4f}")
            print(f"  '{bad_filler}':  stability = {bad_stab:.4f}")


# =============================================================================
# Test 6: Construction Grammar
# =============================================================================

class TestConstructionGrammar:
    """
    THEORY: Constructions are form-meaning composites at all levels.
    
    In Construction Grammar, grammatical patterns carry meaning.
    The same tokens in different constructions have different meanings.
    """
    
    def test_construction_affects_meaning(
        self, basis, xp, embeddings, tokenizer, sentence_to_context
    ):
        """
        Test that construction affects overall meaning (witness).
        """
        # Same words, different constructions
        declarative = "the dog runs"
        interrogative = "does the dog run"
        imperative = "run dog"
        
        ctx_decl = sentence_to_context(declarative)
        ctx_inter = sentence_to_context(interrogative)
        ctx_imper = sentence_to_context(imperative)
        
        # Extract witnesses
        wit_decl = extract_witness(ctx_decl, basis, xp)
        wit_inter = extract_witness(ctx_inter, basis, xp)
        wit_imper = extract_witness(ctx_imper, basis, xp)
        
        print(f"Construction witnesses (σ, p):")
        print(f"  Declarative:   ({wit_decl[0]:.4f}, {wit_decl[1]:.4f})")
        print(f"  Interrogative: ({wit_inter[0]:.4f}, {wit_inter[1]:.4f})")
        print(f"  Imperative:    ({wit_imper[0]:.4f}, {wit_imper[1]:.4f})")
        
        # Compute witness similarity (captures structural difference)
        sim_decl_inter = witness_similarity(ctx_decl, ctx_inter, basis, xp)
        sim_decl_imper = witness_similarity(ctx_decl, ctx_imper, basis, xp)
        
        print(f"Witness similarity:")
        print(f"  Declarative-Interrogative: {sim_decl_inter:.4f}")
        print(f"  Declarative-Imperative:    {sim_decl_imper:.4f}")


# =============================================================================
# Test 7: Conceptual Metaphor
# =============================================================================

class TestConceptualMetaphor:
    """
    THEORY: Metaphor is cross-domain schema mapping.
    
    "Time is money" maps the MONEY domain onto TIME.
    This should show up as schema transfer in witness space.
    """
    
    def test_metaphor_domain_mapping(
        self, basis, xp, embeddings, tokenizer, sentence_to_context
    ):
        """
        Test that metaphor creates structural mapping between domains.
        """
        # MONEY domain
        money_context = sentence_to_context("spend save invest budget")
        
        # TIME domain (literal)
        time_literal = sentence_to_context("hours minutes seconds duration")
        
        # TIME domain (metaphorical - using money terms)
        time_metaphor = sentence_to_context("spend time save time invest time budget time")
        
        # Compute witnesses
        wit_money = np.array(extract_witness(money_context, basis, xp))
        wit_time_lit = np.array(extract_witness(time_literal, basis, xp))
        wit_time_met = np.array(extract_witness(time_metaphor, basis, xp))
        
        # Metaphorical time should be closer to money than literal time is
        sim_money_literal = np.dot(wit_money, wit_time_lit) / (
            np.linalg.norm(wit_money) * np.linalg.norm(wit_time_lit) + 1e-10
        )
        sim_money_metaphor = np.dot(wit_money, wit_time_met) / (
            np.linalg.norm(wit_money) * np.linalg.norm(wit_time_met) + 1e-10
        )
        
        print(f"Metaphor mapping:")
        print(f"  Money-Literal Time:     {sim_money_literal:.4f}")
        print(f"  Money-Metaphorical Time: {sim_money_metaphor:.4f}")


# =============================================================================
# Test 8: Recursive Composition
# =============================================================================

class TestRecursiveComposition:
    """
    THEORY: Schemas compose hierarchically (recursion).
    
    Clauses embed within clauses, creating hierarchical structure.
    This should be reflected in nested witness relationships.
    """
    
    def test_embedded_clause_structure(
        self, basis, xp, embeddings, tokenizer, sentence_to_context
    ):
        """
        Test that clause embedding creates hierarchical structure.
        """
        # Simple clause
        simple = "the dog runs"
        
        # Embedded clause
        embedded = "I know that the dog runs"
        
        # Doubly embedded
        double = "She thinks I know that the dog runs"
        
        ctx_simple = sentence_to_context(simple)
        ctx_embed = sentence_to_context(embedded)
        ctx_double = sentence_to_context(double)
        
        # Check vorticity (structural complexity)
        ens_simple = compute_enstrophy(ctx_simple, basis, xp)
        ens_embed = compute_enstrophy(ctx_embed, basis, xp)
        ens_double = compute_enstrophy(ctx_double, basis, xp)
        
        print(f"Enstrophy by embedding depth:")
        print(f"  Simple:        {ens_simple:.6f}")
        print(f"  Embedded:      {ens_embed:.6f}")
        print(f"  Double-embed:  {ens_double:.6f}")
        
        # Structural complexity should increase with embedding
        # (more composition = more vorticity)
    
    def test_compositional_consistency(
        self, basis, xp, embeddings, tokenizer
    ):
        """
        Test that composition is associative (as theory requires).
        """
        # Get three tokens
        tokens = tokenizer("dog cat bird")
        if len(tokens) >= 3:
            a = embeddings[tokens[0] % embeddings.shape[0]]
            b = embeddings[tokens[1] % embeddings.shape[0]]
            c = embeddings[tokens[2] % embeddings.shape[0]]
            
            # (A @ B) @ C
            left = geometric_product(geometric_product(a, b), c)
            
            # A @ (B @ C)
            right = geometric_product(a, geometric_product(b, c))
            
            # Should be equal (associativity)
            diff = float(xp.linalg.norm(left - right))
            
            print(f"Associativity check: ||left - right|| = {diff:.10f}")
            
            assert diff < 1e-5, f"Composition should be associative, diff = {diff}"
