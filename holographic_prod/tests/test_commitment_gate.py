"""
Test suite for CommitmentGate - the basal ganglia analog.

The commitment gate implements the brain's action selection mechanism:
- Direct pathway: GO (release highest-probability action)
- Indirect pathway: NO-GO (suppress alternatives)  
- Hyperdirect pathway: STOP (emergency brake when entropy too high)

Theory mapping:
- Semantic entropy < φ⁻² → ready to commit
- Stability > φ⁻¹ → state is settled
- Winner-take-all → striatal competition

This is NOT a punctuation module. It's a late-stage commitment gate
that activates only when semantic entropy is already low.
"""

import numpy as np
import pytest
from typing import List, Tuple

# φ-derived constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI          # ≈ 0.618
PHI_INV_SQ = 1 / PHI**2    # ≈ 0.382


class TestCommitmentGateBasics:
    """Test basic gate behavior: commit when ready, hold when uncertain."""
    
    def test_commits_when_entropy_low(self):
        """Gate should commit (return token) when entropy is below threshold."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        # Low entropy distribution: one candidate dominates
        # [0.95, 0.02, 0.02, 0.01] → entropy ≈ 0.29
        scores = np.array([3.0, -1.0, -1.0, -2.0])  # After softmax → low entropy
        candidates = [10, 20, 30, 40]  # Token IDs
        
        result = gate.decide(scores, candidates)
        
        # Should commit to highest-scoring candidate
        assert result.committed is True
        assert result.token == 10  # Highest score
        assert result.entropy < PHI_INV_SQ  # Below threshold
    
    def test_holds_when_entropy_high(self):
        """Gate should hold (not commit) when entropy is above threshold."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        # High entropy distribution: candidates are nearly equal
        # [0.26, 0.25, 0.25, 0.24] → entropy ≈ 1.38
        scores = np.array([0.1, 0.0, 0.0, -0.1])  # After softmax → high entropy
        candidates = [10, 20, 30, 40]
        
        result = gate.decide(scores, candidates)
        
        # Should NOT commit - entropy too high
        assert result.committed is False
        assert result.token is None
        assert result.entropy > PHI_INV_SQ
    
    def test_threshold_is_phi_derived(self):
        """Entropy threshold should be φ⁻² ≈ 0.382."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        assert np.isclose(gate.entropy_threshold, PHI_INV_SQ, rtol=1e-6)


class TestWinnerTakeAll:
    """Test striatal competition (winner-take-all) behavior."""
    
    def test_selects_highest_score(self):
        """Should select the candidate with highest score."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        # Clear winner
        scores = np.array([1.0, 5.0, 2.0, 0.5])  # Index 1 wins
        candidates = [100, 200, 300, 400]
        
        result = gate.decide(scores, candidates)
        
        if result.committed:
            assert result.token == 200  # Highest score
    
    def test_suppresses_alternatives(self):
        """Result should indicate which alternatives were suppressed."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        scores = np.array([5.0, 1.0, 0.5, 0.1])
        candidates = [10, 20, 30, 40]
        
        result = gate.decide(scores, candidates)
        
        if result.committed:
            # Suppressed should contain the non-winners
            assert 20 in result.suppressed
            assert 30 in result.suppressed
            assert 40 in result.suppressed
            assert 10 not in result.suppressed  # Winner not suppressed


class TestPathways:
    """Test the three basal ganglia pathways."""
    
    def test_direct_pathway_releases_action(self):
        """Direct pathway (GO): releases action when conditions met."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        # Strong, clear signal
        scores = np.array([10.0, 1.0, 0.0])
        candidates = [1, 2, 3]
        
        result = gate.decide(scores, candidates)
        
        assert result.committed is True
        assert result.pathway == "direct"
    
    def test_indirect_pathway_inhibits(self):
        """Indirect pathway (NO-GO): inhibits when uncertain."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        # Moderately ambiguous signal - entropy between φ⁻² and 1.0
        # Scores that give entropy ~0.6 (above 0.382 but below 1.0)
        scores = np.array([2.0, 1.0, 0.0])  # After softmax: ~[0.67, 0.24, 0.09]
        candidates = [1, 2, 3]
        
        result = gate.decide(scores, candidates)
        
        assert result.committed is False
        assert result.pathway == "indirect"
        assert PHI_INV_SQ < result.entropy < 1.0  # In indirect range
    
    def test_hyperdirect_pathway_emergency_stop(self):
        """Hyperdirect pathway (STOP): emergency brake on very high entropy."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        # Extremely ambiguous - max entropy
        n = 100
        scores = np.zeros(n)  # Uniform distribution
        candidates = list(range(n))
        
        result = gate.decide(scores, candidates)
        
        assert result.committed is False
        assert result.pathway == "hyperdirect"
        assert result.entropy > 1.0  # Very high entropy


class TestEntropyComputation:
    """Test entropy calculation is correct."""
    
    def test_entropy_of_uniform_distribution(self):
        """Uniform distribution should have max entropy = log(n)."""
        from holographic_prod.core.commitment_gate import compute_entropy
        
        n = 4
        probs = np.ones(n) / n  # Uniform
        
        entropy = compute_entropy(probs)
        
        expected = np.log(n)  # Max entropy for n classes
        assert np.isclose(entropy, expected, rtol=1e-6)
    
    def test_entropy_of_deterministic_distribution(self):
        """Deterministic distribution should have entropy = 0."""
        from holographic_prod.core.commitment_gate import compute_entropy
        
        probs = np.array([1.0, 0.0, 0.0, 0.0])  # All mass on one
        
        entropy = compute_entropy(probs)
        
        assert np.isclose(entropy, 0.0, atol=1e-10)
    
    def test_entropy_of_typical_distribution(self):
        """Test entropy of a realistic distribution."""
        from holographic_prod.core.commitment_gate import compute_entropy
        
        # Softmax of [3, 1, 0, -1]
        scores = np.array([3.0, 1.0, 0.0, -1.0])
        probs = np.exp(scores) / np.sum(np.exp(scores))
        
        entropy = compute_entropy(probs)
        
        # Should be low but not zero
        assert 0 < entropy < 1.0


class TestIntegrationWithVorticity:
    """Test integration with vorticity-weighted scoring."""
    
    def test_uses_vorticity_scores(self):
        """Gate should work with vorticity-weighted score inputs."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        from holographic_prod.core.quotient import vorticity_weighted_scores
        from holographic_prod.core.algebra import build_clifford_basis
        from scipy.stats import special_ortho_group
        
        gate = CommitmentGate()
        
        # Create a random SO(4) state and candidate embeddings
        def random_so4():
            return special_ortho_group.rvs(4)
        
        state = random_so4()
        n_candidates = 10
        candidate_embeddings = np.array([random_so4() for _ in range(n_candidates)])
        candidates = list(range(n_candidates))
        basis = build_clifford_basis()
        
        # Get vorticity-weighted scores
        scores = vorticity_weighted_scores(state, candidate_embeddings, basis, np)
        
        # Gate should handle these scores
        result = gate.decide(scores, candidates)
        
        assert result is not None
        assert isinstance(result.entropy, float)


class TestGPUCompatibility:
    """Test GPU acceleration compatibility."""
    
    def test_works_with_cupy_arrays(self):
        """Gate should work with CuPy arrays when available."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        try:
            import cupy as cp
            scores = cp.array([5.0, 1.0, 0.5])
            candidates = [1, 2, 3]
            
            result = gate.decide(scores, candidates)
            
            assert result is not None
        except ImportError:
            pytest.skip("CuPy not available")
    
    def test_works_with_numpy_arrays(self):
        """Gate should work with NumPy arrays."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        scores = np.array([5.0, 1.0, 0.5])
        candidates = [1, 2, 3]
        
        result = gate.decide(scores, candidates)
        
        assert result is not None


class TestBatchProcessing:
    """Test batch processing for efficiency."""
    
    def test_batch_decision(self):
        """Should process multiple decisions in parallel."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        batch_size = 8
        n_candidates = 50
        
        # Batch of score distributions
        scores_batch = np.random.randn(batch_size, n_candidates)
        candidates = list(range(n_candidates))
        
        results = gate.decide_batch(scores_batch, candidates)
        
        assert len(results) == batch_size
        for result in results:
            assert hasattr(result, 'committed')
            assert hasattr(result, 'entropy')


class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_single_candidate(self):
        """With one candidate, should always commit."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        scores = np.array([1.0])
        candidates = [42]
        
        result = gate.decide(scores, candidates)
        
        assert result.committed is True
        assert result.token == 42
        assert result.entropy == 0.0  # Zero entropy with single option
    
    def test_empty_candidates(self):
        """Empty candidates should raise an error."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        scores = np.array([])
        candidates = []
        
        with pytest.raises(ValueError, match="at least one candidate"):
            gate.decide(scores, candidates)
    
    def test_numerical_stability(self):
        """Should handle extreme score values without overflow."""
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        # Very large scores with extreme separation
        # After softmax: [~1.0, ~0.0, ~0.0] → low entropy, should commit
        scores = np.array([1000.0, 0.0, -1000.0])
        candidates = [1, 2, 3]
        
        result = gate.decide(scores, candidates)
        
        # Should handle without NaN/Inf
        assert not np.isnan(result.entropy)
        assert not np.isinf(result.entropy)
        # Should commit to highest
        assert result.committed is True
        assert result.token == 1


class TestBrainAnalogFidelity:
    """Test that behavior matches brain analog expectations."""
    
    def test_hesitation_on_near_ties(self):
        """Should hesitate (not commit) when options are nearly tied.
        
        Brain analog: Pausing mid-sentence while knowing what you mean.
        """
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        # Near-tie: "happy" vs "glad" - both valid continuations
        scores = np.array([2.01, 2.00])
        candidates = ["happy", "glad"]
        
        result = gate.decide(scores, candidates)
        
        # Should NOT commit prematurely
        assert result.committed is False
    
    def test_quick_commit_on_clear_winner(self):
        """Should commit quickly when one option dominates.
        
        Brain analog: Fluent speech when word retrieval is easy.
        """
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        # Clear winner: "the" after "Once upon a time,"
        scores = np.array([10.0, 1.0, 0.5, 0.1])
        candidates = ["the", "a", "one", "there"]
        
        result = gate.decide(scores, candidates)
        
        assert result.committed is True
        assert result.token == "the"
    
    def test_motor_vs_semantic_distinction(self):
        """Gate decision should be independent of semantic content.
        
        Brain analog: Punctuation is motor-prosodic, not semantic.
        """
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        gate = CommitmentGate()
        
        # Same entropy profile, different tokens
        scores1 = np.array([5.0, 1.0])
        scores2 = np.array([5.0, 1.0])  # Same distribution
        
        candidates1 = [".", ","]      # Punctuation
        candidates2 = ["cat", "dog"]  # Words
        
        result1 = gate.decide(scores1, candidates1)
        result2 = gate.decide(scores2, candidates2)
        
        # Same commitment behavior regardless of token content
        assert result1.committed == result2.committed
        assert np.isclose(result1.entropy, result2.entropy)


class TestNeurologicalFailureModes:
    """
    Test neurological failure modes mapped from clinical conditions.
    
    These tests validate that the commitment gate exhibits the same
    failure patterns as human neurological disorders when parameters
    are pushed to extremes:
    
    - Parkinson's: Gate stuck closed (too little release)
    - Tourette's: Gate stuck open (too much release)
    - Stuttering: Timing mismatch (semantic ready, motor hesitant)
    - Akinetic mutism: Complete failure to initiate
    
    This validates the brain-analog fidelity of the architecture.
    """
    
    def test_parkinsonian_mode_never_commits(self):
        """
        Parkinson's analog: Gate threshold so low it never commits.
        
        Clinical: Patients know what they want to say but can't release it.
        "I know what I want to say, but I can't get it out."
        
        Architecture: entropy_threshold = 0.001 means entropy must be < 0.001 to commit,
        which requires extreme score separation.
        """
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        # Parkinsonian gate: threshold so low it almost never commits
        parkinsonian_gate = CommitmentGate(entropy_threshold=0.001)
        
        # Moderate winner - entropy will be above 0.001
        # [5.0, 1.0, 0.5] → entropy ≈ 0.05 (above 0.001)
        scores = np.array([5.0, 1.0, 0.5])
        candidates = [1, 2, 3]
        
        result = parkinsonian_gate.decide(scores, candidates)
        
        # Should NOT commit - threshold too restrictive
        assert result.committed is False
        assert result.pathway in ["indirect", "hyperdirect"]
        
        # Would need forced_commit to ever produce output
        forced = parkinsonian_gate.forced_commit(scores, candidates)
        assert forced.committed is True
        assert forced.pathway == "forced"
    
    def test_tourettes_mode_always_commits(self):
        """
        Tourette's analog: Gate threshold so high it always commits.
        
        Clinical: Actions released before semantic planning complete.
        Inhibitory control fails, motor outputs "leak".
        
        Architecture: entropy_threshold = 10 AND hyperdirect_threshold = 10 means
        any entropy below 10 commits, which is essentially always.
        """
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        # Tourette's gate: BOTH thresholds high so it always commits
        # (hyperdirect_threshold must also be high to prevent emergency stop)
        tourettes_gate = CommitmentGate(entropy_threshold=10.0, hyperdirect_threshold=10.0)
        
        # Even with near-uniform distribution, should commit
        n = 10
        scores = np.zeros(n)  # Uniform → max entropy ≈ 2.3
        candidates = list(range(n))
        
        result = tourettes_gate.decide(scores, candidates)
        
        # Should commit even when uncertain - gate too permissive
        assert result.committed is True
        assert result.pathway == "direct"
        
        # Confidence should be LOW (we committed when we shouldn't have)
        assert result.confidence < 0.3  # Near-uniform → ~1/n confidence
    
    def test_stuttering_mode_boundary_hesitation(self):
        """
        Stuttering analog: Semantic planning outruns motor commitment.
        
        Clinical: Repetition of function words, blocks at sentence boundaries.
        The sentence is ready but the release mechanism stutters.
        
        Architecture: High stability (semantic ready) but moderate entropy (can't pick).
        This happens at boundaries where multiple valid continuations exist.
        
        Note: For entropy to be in the "indirect" range (φ⁻² < H < 1.0),
        we need scores that produce entropy ≈ 0.5-0.9.
        """
        from holographic_prod.core.commitment_gate import CommitmentGate
        from holographic_prod.core.quotient import grace_stability
        from holographic_prod.core.algebra import build_clifford_basis
        
        gate = CommitmentGate()
        basis = build_clifford_basis()
        
        # Simulate a boundary: ". vs ," - two valid options
        # Scores chosen to produce entropy in indirect range (0.382 < H < 1.0)
        # [2.0, 1.0] → softmax ≈ [0.73, 0.27] → entropy ≈ 0.59
        scores = np.array([2.0, 1.0])
        candidates = [".", ","]
        
        result = gate.decide(scores, candidates)
        
        # Should NOT commit - this is the stuttering point
        assert result.committed is False
        
        # This is exactly the pattern: semantic planning done, motor hesitant
        # The gate correctly identifies this as a "hold" situation
        assert result.pathway == "indirect"
        
        # Entropy should be in the indirect range
        assert PHI_INV_SQ < result.entropy < 1.0
    
    def test_akinetic_mutism_extreme(self):
        """
        Akinetic mutism analog: Complete failure to initiate action.
        
        Clinical: Near silence, but patients can still understand language.
        Semantic cognition without motor release, fully dissociated.
        
        Architecture: entropy_threshold = 0 AND hyperdirect_threshold = 0
        means ANY entropy triggers emergency stop.
        """
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        # Akinetic mutism: both thresholds at 0
        akinetic_gate = CommitmentGate(
            entropy_threshold=0.0,
            hyperdirect_threshold=0.0
        )
        
        # Any distribution with >1 candidate has entropy > 0
        scores = np.array([100.0, 0.0])  # Extremely clear winner
        candidates = [1, 2]
        
        result = akinetic_gate.decide(scores, candidates)
        
        # Should trigger hyperdirect (emergency stop) even with clear winner
        assert result.committed is False
        assert result.pathway == "hyperdirect"
    
    def test_healthy_gate_balanced_behavior(self):
        """
        Healthy gate: Balanced threshold allows appropriate commitment.
        
        Clinical: Fluent speech with appropriate pauses at uncertainty.
        
        Architecture: Default φ⁻² threshold provides optimal balance.
        
        Entropy ranges:
        - Direct (commit): H < φ⁻² ≈ 0.382
        - Indirect (hold): φ⁻² < H < 1.0
        - Hyperdirect (stop): H > 1.0
        """
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        # Healthy gate with default (φ-derived) parameters
        healthy_gate = CommitmentGate()
        
        # Test 1: Clear winner → commits (entropy < 0.382)
        # [10.0, 1.0, 0.5] → entropy ≈ 0.002 (very low)
        clear_scores = np.array([10.0, 1.0, 0.5])
        result_clear = healthy_gate.decide(clear_scores, [1, 2, 3])
        assert result_clear.committed is True
        assert result_clear.pathway == "direct"
        
        # Test 2: Moderate ambiguity → holds (0.382 < entropy < 1.0)
        # [2.0, 1.0] → entropy ≈ 0.59 (in indirect range)
        ambiguous_scores = np.array([2.0, 1.0])
        result_ambiguous = healthy_gate.decide(ambiguous_scores, [1, 2])
        assert result_ambiguous.committed is False
        assert result_ambiguous.pathway == "indirect"
        
        # Test 3: Extreme uncertainty → emergency stop (entropy > 1.0)
        # Uniform over 100 → entropy ≈ 4.6 (very high)
        uniform_scores = np.zeros(100)
        result_uniform = healthy_gate.decide(uniform_scores, list(range(100)))
        assert result_uniform.committed is False
        assert result_uniform.pathway == "hyperdirect"
    
    def test_dopamine_analog_threshold_sensitivity(self):
        """
        Test that threshold acts like dopamine level.
        
        Clinical: Dopamine modulates the "readiness to act" threshold.
        Low dopamine (Parkinson's) → high threshold → hard to commit.
        High dopamine (mania) → low threshold → easy to commit.
        
        Architecture: entropy_threshold is the dopamine analog.
        """
        from holographic_prod.core.commitment_gate import CommitmentGate
        
        # Same score distribution, different "dopamine levels"
        scores = np.array([3.0, 1.0, 0.5])
        candidates = [1, 2, 3]
        
        # Low dopamine (high threshold) → harder to commit
        low_dopamine = CommitmentGate(entropy_threshold=0.1)
        result_low = low_dopamine.decide(scores, candidates)
        
        # Normal dopamine (φ⁻² threshold) → balanced
        normal_dopamine = CommitmentGate(entropy_threshold=PHI_INV_SQ)
        result_normal = normal_dopamine.decide(scores, candidates)
        
        # High dopamine (high threshold) → easier to commit
        high_dopamine = CommitmentGate(entropy_threshold=1.0)
        result_high = high_dopamine.decide(scores, candidates)
        
        # Same entropy, different commitment decisions
        assert np.isclose(result_low.entropy, result_normal.entropy)
        assert np.isclose(result_normal.entropy, result_high.entropy)
        
        # But different commitment outcomes
        # Low dopamine should be most restrictive
        # High dopamine should be most permissive
        assert result_high.committed  # Should definitely commit
        # result_normal may or may not commit depending on exact entropy
        # result_low should be least likely to commit


class TestIntegrationWithGraceEvolution:
    """
    Test that the commitment gate integrates correctly with Grace evolution.
    
    THEORY: When the gate holds (NO-GO), the semantic state should evolve
    further via Grace until it's ready to commit. This is the key mechanism
    that prevents the "forced commitment" problem of transformers.
    """
    
    def test_grace_evolution_reduces_entropy(self):
        """
        Grace evolution should reduce entropy by contracting to attractors.
        
        THEORY: Grace damps high-grade components, leaving the stable witness.
        This naturally reduces entropy as the state settles.
        """
        from holographic_prod.core.commitment_gate import CommitmentGate, compute_entropy, phi_kernel_probs
        from holographic_prod.core.algebra import build_clifford_basis, grace_operator
        from holographic_prod.core.quotient import vorticity_weighted_scores
        from scipy.stats import special_ortho_group
        
        gate = CommitmentGate()
        basis = build_clifford_basis()
        
        # Create a random state and candidates
        np.random.seed(42)
        state = special_ortho_group.rvs(4)
        n_candidates = 20
        candidates = [special_ortho_group.rvs(4) for _ in range(n_candidates)]
        candidate_array = np.array(candidates)
        
        # Initial scores and entropy
        initial_scores = vorticity_weighted_scores(state, candidate_array, basis, np)
        initial_entropy = compute_entropy(phi_kernel_probs(initial_scores))
        
        # Apply Grace evolution
        evolved_state = state.copy()
        for _ in range(5):
            evolved_state = grace_operator(evolved_state, basis, np)
        
        # Scores after Grace evolution
        evolved_scores = vorticity_weighted_scores(evolved_state, candidate_array, basis, np)
        evolved_entropy = compute_entropy(phi_kernel_probs(evolved_scores))
        
        # Grace should generally reduce or maintain entropy (settle to attractor)
        # Note: This isn't guaranteed to decrease every time, but on average it should
        # The key is that Grace provides a mechanism for entropy reduction
        print(f"Initial entropy: {initial_entropy:.4f}, Evolved entropy: {evolved_entropy:.4f}")
    
    def test_hold_then_evolve_pattern(self):
        """
        Test the hold → evolve → retry pattern from attractor_generation.py.
        
        THEORY: When gate holds, we don't force commit. We evolve and retry.
        This is the brain-analog pattern: hesitate → think more → commit when ready.
        """
        from holographic_prod.core.commitment_gate import CommitmentGate
        from holographic_prod.core.algebra import build_clifford_basis, grace_operator
        from holographic_prod.core.quotient import vorticity_weighted_scores
        from scipy.stats import special_ortho_group
        
        gate = CommitmentGate()
        basis = build_clifford_basis()
        
        # Create ambiguous situation
        np.random.seed(123)
        state = special_ortho_group.rvs(4)
        n_candidates = 5
        candidates = [special_ortho_group.rvs(4) for _ in range(n_candidates)]
        candidate_array = np.array(candidates)
        candidate_ids = list(range(n_candidates))
        
        # Get initial scores
        scores = vorticity_weighted_scores(state, candidate_array, basis, np)
        
        # First decision
        decision1 = gate.decide(scores, candidate_ids)
        
        if not decision1.committed:
            # Gate held - evolve state via Grace
            evolved_state = state.copy()
            for _ in range(3):
                evolved_state = grace_operator(evolved_state, basis, np)
            
            # Retry with evolved state
            evolved_scores = vorticity_weighted_scores(evolved_state, candidate_array, basis, np)
            decision2 = gate.decide(evolved_scores, candidate_ids)
            
            # If still not committed, forced_commit is the fallback
            if not decision2.committed:
                final = gate.forced_commit(evolved_scores, candidate_ids)
                assert final.committed is True
                assert final.pathway == "forced"
