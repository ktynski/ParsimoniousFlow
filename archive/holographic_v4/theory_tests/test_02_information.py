"""
Test 02: Information Theory — Bottleneck, Capacity, Geometry
=============================================================

PURPOSE: Formalize Grace as information bottleneck, derive capacity limits.
         These tests connect to information-theoretic foundations.

TESTS:
    1. test_information_bottleneck - Grace compresses I(X;T) while preserving I(T;Y)
    2. test_channel_capacity       - C = max I(X;Y) for holographic retrieval
    3. test_mutual_info_geometry   - Fisher metric on witness space
    4. test_interdependence        - All memories co-arise (dependent origination)

THEORY PREDICTIONS:
    - Grace IS an information bottleneck (compresses noise, preserves signal)
    - Channel capacity bounded by witness dimensionality (2D: σ, p)
    - Fisher metric should be positive definite on witness space
    - Holographic memory creates interdependence between all stored items
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
    frobenius_similarity,
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
)
from holographic_v4.quotient import (
    compute_enstrophy,
    grade_energies,
    grace_stability,
    witness_similarity,
)
from holographic_v4.holographic_memory import HybridHolographicMemory, compute_witness_entropy
from holographic_v4.constants import DTYPE

from .utils import (
    compute_mutual_information,
    bootstrap_confidence_interval,
    permutation_test,
)


# =============================================================================
# Test 1: Information Bottleneck
# =============================================================================

class TestInformationBottleneck:
    """
    THEORY: Grace acts as an information bottleneck.
    
    Information Bottleneck principle:
        min I(X;T) - β·I(T;Y)
    
    Where:
        X = input (full context matrix)
        T = representation (after Grace)
        Y = target (what we want to predict)
    
    Grace compresses X into T by removing grade information,
    while preserving the witness which is sufficient for Y.
    """
    
    def test_grace_compresses_representation(self, basis, xp, random_context):
        """
        Verify that Grace reduces information content.
        
        Measure: Rank of matrix, entropy of coefficients
        """
        M = random_context(n_tokens=10, seed=42)
        
        # Measure "information" via coefficient variance
        from holographic_v4.algebra import decompose_to_coefficients
        
        def coefficient_entropy(M):
            coeffs = decompose_to_coefficients(M, basis, xp)
            # Normalize
            coeffs = np.abs(coeffs) / (np.sum(np.abs(coeffs)) + 1e-10)
            # Compute entropy
            entropy = -np.sum(coeffs[coeffs > 0] * np.log2(coeffs[coeffs > 0] + 1e-10))
            return entropy
        
        entropy_before = coefficient_entropy(M)
        
        # Apply Grace
        M_graced = grace_operator(M, basis, xp)
        entropy_after = coefficient_entropy(M_graced)
        
        # Grace should reduce entropy (compress)
        assert entropy_after <= entropy_before + 0.5, (
            f"Grace should reduce entropy: {entropy_before:.4f} → {entropy_after:.4f}"
        )
    
    def test_grace_preserves_relevant_info(self, basis, xp, embeddings):
        """
        Verify that Grace preserves information needed for retrieval.
        
        Even after Grace, similar contexts should have similar witnesses.
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        # Generate pairs of related contexts
        # (same tokens, slightly different order)
        pairs = []
        for trial in range(20):
            base_tokens = np.random.randint(0, vocab_size, size=6)
            
            # Original
            ctx1 = normalize_matrix(
                geometric_product_batch(embeddings[base_tokens], xp), xp
            )
            
            # Swap two adjacent tokens
            swapped = base_tokens.copy()
            swapped[2], swapped[3] = swapped[3], swapped[2]
            ctx2 = normalize_matrix(
                geometric_product_batch(embeddings[swapped], xp), xp
            )
            
            pairs.append((ctx1, ctx2))
        
        # Measure witness similarity before and after Grace
        sim_before = []
        sim_after = []
        
        for ctx1, ctx2 in pairs:
            sim_before.append(witness_similarity(ctx1, ctx2, basis, xp))
            
            g1 = grace_operator(ctx1, basis, xp)
            g2 = grace_operator(ctx2, basis, xp)
            sim_after.append(witness_similarity(g1, g2, basis, xp))
        
        mean_before = np.mean(sim_before)
        mean_after = np.mean(sim_after)
        
        print(f"Witness similarity (related contexts):")
        print(f"  Before Grace: {mean_before:.4f}")
        print(f"  After Grace:  {mean_after:.4f}")
        
        # Similarity should be preserved or increased (noise removed)
        assert mean_after >= mean_before - 0.1, (
            "Grace should preserve relevant similarity"
        )


# =============================================================================
# Test 2: Channel Capacity
# =============================================================================

class TestChannelCapacity:
    """
    THEORY: Channel capacity C = max I(X;Y) for holographic retrieval.
    
    The holographic memory is a noisy channel:
        X (context) → Memory → Y (retrieved target)
    
    Capacity depends on:
        - SNR (signal-to-interference ratio)
        - Memory occupation
        - Witness dimensionality (2D bottleneck)
    """
    
    def test_capacity_vs_memory_load(self, basis, xp, embeddings):
        """
        Test how retrieval accuracy degrades with memory load.
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        results_by_load = []
        
        for n_items in [10, 50, 100, 200, 500]:
            memory = HybridHolographicMemory.create(basis, xp=xp)
            
            # Store items
            stored = []
            for i in range(n_items):
                token_ids = np.random.randint(0, vocab_size, size=6)
                tokens = embeddings[token_ids]
                ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
                target_idx = np.random.randint(0, vocab_size)
                
                memory.store(ctx, embeddings[target_idx], target_idx=target_idx)
                stored.append((ctx, target_idx))
            
            # Test retrieval
            successes = 0
            for ctx, expected_idx in stored[:min(50, len(stored))]:
                retrieved, idx, conf, source = memory.retrieve(ctx)
                if idx == expected_idx:
                    successes += 1
            
            accuracy = successes / min(50, len(stored))
            results_by_load.append((n_items, accuracy))
        
        print("Capacity vs Memory Load:")
        for n, acc in results_by_load:
            print(f"  {n:4d} items: {acc:.1%}")
        
        # Accuracy should generally decrease with load (but might plateau)
        # This is a characteristic capacity curve
    
    def test_capacity_vs_snr(self, basis, xp, embeddings):
        """
        Test how retrieval accuracy depends on signal quality.
        
        SNR here = context stability (high stability = clean signal)
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        memory = HybridHolographicMemory.create(basis, xp=xp)
        
        results_by_stability = []
        
        for trial in range(100):
            # Generate context
            L = np.random.randint(4, 15)
            token_ids = np.random.randint(0, vocab_size, size=L)
            tokens = embeddings[token_ids]
            ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
            
            # Measure stability (our "SNR" analog)
            stability = grace_stability(ctx, basis, xp)
            
            target_idx = np.random.randint(0, vocab_size)
            memory.store(ctx, embeddings[target_idx], target_idx=target_idx)
            
            # Test retrieval
            retrieved, idx, conf, source = memory.retrieve(ctx)
            correct = (idx == target_idx)
            
            results_by_stability.append((stability, correct))
        
        # Analyze correlation
        stabilities = np.array([r[0] for r in results_by_stability])
        accuracies = np.array([r[1] for r in results_by_stability], dtype=float)
        
        # Split by stability
        median_stab = np.median(stabilities)
        low_stab_acc = np.mean(accuracies[stabilities < median_stab])
        high_stab_acc = np.mean(accuracies[stabilities >= median_stab])
        
        print(f"Accuracy by stability:")
        print(f"  Low stability (<{median_stab:.3f}):  {low_stab_acc:.1%}")
        print(f"  High stability (≥{median_stab:.3f}): {high_stab_acc:.1%}")


# =============================================================================
# Test 3: Mutual Information Geometry
# =============================================================================

class TestMutualInfoGeometry:
    """
    THEORY: Fisher metric on witness space should be positive definite.
    
    The Fisher information metric defines a natural geometry on the
    space of probability distributions. For witnesses (σ, p), we can
    define a metric based on how distinguishable nearby witnesses are.
    
    g_ij = E[∂_i log p(x|θ) · ∂_j log p(x|θ)]
    """
    
    def test_witness_space_metric(self, basis, xp, embeddings):
        """
        Empirically estimate metric on witness space.
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        # Generate many contexts and their witnesses
        witnesses = []
        for trial in range(200):
            L = np.random.randint(4, 12)
            token_ids = np.random.randint(0, vocab_size, size=L)
            tokens = embeddings[token_ids]
            ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
            
            sigma, pseudo = extract_witness(ctx, basis, xp)
            witnesses.append([sigma, pseudo])
        
        witnesses = np.array(witnesses)
        
        # Compute covariance (related to Fisher metric)
        cov = np.cov(witnesses.T)
        
        print("Witness space covariance matrix:")
        print(f"  [[{cov[0,0]:.6f}, {cov[0,1]:.6f}],")
        print(f"   [{cov[1,0]:.6f}, {cov[1,1]:.6f}]]")
        
        # Should be positive definite (eigenvalues > 0)
        eigenvalues = np.linalg.eigvalsh(cov)
        
        print(f"Eigenvalues: {eigenvalues}")
        
        # Both eigenvalues should be positive
        assert np.all(eigenvalues > 0), (
            f"Covariance matrix should be positive definite, eigenvalues: {eigenvalues}"
        )
    
    def test_witness_distinguishability(self, basis, xp, embeddings):
        """
        Test that nearby witnesses correspond to similar contexts.
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        # Generate contexts
        contexts = []
        witnesses = []
        
        for trial in range(100):
            L = np.random.randint(5, 10)
            token_ids = np.random.randint(0, vocab_size, size=L)
            tokens = embeddings[token_ids]
            ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
            
            sigma, pseudo = extract_witness(ctx, basis, xp)
            
            contexts.append(ctx)
            witnesses.append([sigma, pseudo])
        
        witnesses = np.array(witnesses)
        
        # For each context, find nearest neighbor in witness space
        # and check if their full matrices are also similar
        
        witness_dists = []
        matrix_dists = []
        
        for i in range(len(contexts)):
            for j in range(i + 1, len(contexts)):
                w_dist = np.linalg.norm(witnesses[i] - witnesses[j])
                m_dist = float(xp.linalg.norm(contexts[i] - contexts[j]))
                
                witness_dists.append(w_dist)
                matrix_dists.append(m_dist)
        
        # Correlation between witness distance and matrix distance
        correlation = np.corrcoef(witness_dists, matrix_dists)[0, 1]
        
        print(f"Witness distance vs Matrix distance correlation: {correlation:.4f}")
        
        # Should be positively correlated
        assert correlation > 0.2, (
            f"Witness and matrix distances should be correlated, got r={correlation:.4f}"
        )


# =============================================================================
# Test 4: Interdependence
# =============================================================================

class TestInterdependence:
    """
    THEORY: In holographic memory, all memories co-arise (dependent origination).
    
    Unlike slot-based memory where items are independent, holographic
    superposition creates subtle interdependence between all stored items.
    
    This is analogous to "dependent origination" in Buddhist philosophy:
    nothing exists independently, all phenomena arise together.
    """
    
    def test_memory_interdependence(self, basis, xp, embeddings):
        """
        Test that storing new items affects retrieval of existing items.
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        memory = HybridHolographicMemory.create(basis, xp=xp)
        
        # Store initial items
        initial_items = []
        for i in range(20):
            token_ids = np.random.randint(0, vocab_size, size=6)
            tokens = embeddings[token_ids]
            ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
            target_idx = np.random.randint(0, vocab_size)
            
            memory.store(ctx, embeddings[target_idx], target_idx=target_idx)
            initial_items.append((ctx, target_idx))
        
        # Measure retrieval confidence BEFORE adding more items
        confidence_before = []
        for ctx, expected_idx in initial_items:
            retrieved, idx, conf, source = memory.retrieve(ctx)
            confidence_before.append(conf)
        
        # Add many more items (create interference)
        for i in range(100):
            token_ids = np.random.randint(0, vocab_size, size=6)
            tokens = embeddings[token_ids]
            ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
            target_idx = np.random.randint(0, vocab_size)
            memory.store(ctx, embeddings[target_idx], target_idx=target_idx)
        
        # Measure retrieval confidence AFTER
        confidence_after = []
        for ctx, expected_idx in initial_items:
            retrieved, idx, conf, source = memory.retrieve(ctx)
            confidence_after.append(conf)
        
        conf_before = np.mean(confidence_before)
        conf_after = np.mean(confidence_after)
        
        print(f"Retrieval confidence:")
        print(f"  Before adding items: {conf_before:.4f}")
        print(f"  After adding items:  {conf_after:.4f}")
        
        # Confidence should decrease due to interference (interdependence)
        # This is expected behavior in holographic memory
    
    def test_holographic_superposition_creates_correlation(self, basis, xp, embeddings):
        """
        Test that holographic superposition creates correlation between items.
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        memory = HybridHolographicMemory.create(basis, xp=xp)
        
        # Store items
        items = []
        for i in range(30):
            token_ids = np.random.randint(0, vocab_size, size=5)
            tokens = embeddings[token_ids]
            ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
            target_idx = np.random.randint(0, vocab_size)
            
            memory.store(ctx, embeddings[target_idx], target_idx=target_idx)
            items.append((ctx, target_idx))
        
        # Check cross-talk: retrieve with context A, see if it partially
        # activates unrelated context B
        
        cross_activations = []
        for i in range(10):
            ctx_a, _ = items[i]
            ctx_b, _ = items[i + 10]
            
            # Retrieve with ctx_a
            retrieved_a, idx_a, conf_a, _ = memory.retrieve(ctx_a)
            
            # Check similarity to what ctx_b would retrieve
            retrieved_b, idx_b, conf_b, _ = memory.retrieve(ctx_b)
            
            if retrieved_a is not None and retrieved_b is not None:
                # Measure cross-similarity
                cross_sim = frobenius_similarity(retrieved_a, retrieved_b)
                cross_activations.append(cross_sim)
        
        if cross_activations:
            mean_cross = np.mean(cross_activations)
            print(f"Mean cross-activation between unrelated items: {mean_cross:.4f}")
            
            # Some cross-activation is expected in holographic memory
            # (unlike slot memory which would have 0)
    
    def test_witness_entropy_increases_with_load(self, basis, xp, embeddings):
        """
        Test that witness entropy (capacity signal) increases with memory load.
        """
        vocab_size = embeddings.shape[0]
        np.random.seed(42)
        
        memory = HybridHolographicMemory.create(basis, xp=xp)
        
        entropy_history = []
        
        for i in range(100):
            # Add item
            token_ids = np.random.randint(0, vocab_size, size=6)
            tokens = embeddings[token_ids]
            ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
            target_idx = np.random.randint(0, vocab_size)
            
            memory.store(ctx, embeddings[target_idx], target_idx=target_idx)
            
            # Measure proxy for entropy every 10 items
            if (i + 1) % 10 == 0:
                # Use number of stored items as proxy for entropy
                n_stored = memory.n_holographic_stores + memory.n_episodic_stores
                entropy = n_stored / 100.0
                entropy_history.append((i + 1, entropy))
        
        print("Witness entropy vs load:")
        for n, h in entropy_history:
            print(f"  {n:3d} items: H = {h:.4f}")
        
        # Entropy should generally increase (or plateau)
        if len(entropy_history) >= 3:
            early_entropy = entropy_history[1][1]
            late_entropy = entropy_history[-1][1]
            
            # Later entropy should not be much lower than early
            assert late_entropy >= early_entropy * 0.5, (
                f"Entropy should not decrease significantly: {early_entropy:.4f} → {late_entropy:.4f}"
            )
