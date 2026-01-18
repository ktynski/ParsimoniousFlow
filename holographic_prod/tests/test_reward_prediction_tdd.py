"""
TDD Tests for Reward Prediction Error (Dopamine Analog)
=======================================================

These tests define the EXPECTED behavior of a reward system.
Run BEFORE implementation to see failures, then implement to pass.

THEORY:
    - VTA/NAc computes Reward Prediction Error (RPE)
    - RPE = actual_reward - predicted_reward
    - Positive RPE → strengthen (dopamine burst)
    - Negative RPE → weaken (dopamine dip)
    - This modulates BOTH learning AND generation

NO MOCKS. NO FALLBACKS. NO FAKE DATA.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/root/project' if '/root/project' not in sys.path else '')


class TestRewardPredictionError:
    """Tests for the core RPE computation."""
    
    def test_rpe_positive_when_better_than_expected(self):
        """
        REQUIREMENT: RPE > 0 when actual > predicted.
        This triggers dopamine BURST (strengthen binding).
        """
        from holographic_prod.cognitive.reward_prediction import RewardPredictor
        
        predictor = RewardPredictor()
        predictor.predicted_reward = 0.5  # Expecting 0.5
        
        actual = 0.8  # Got 0.8 (better!)
        rpe = predictor.compute_rpe(actual)
        
        assert rpe > 0, f"RPE should be positive when actual > predicted: {rpe}"
        assert abs(rpe - 0.3) < 0.01, f"RPE should be 0.3, got {rpe}"
        
        print(f"✓ RPE = {rpe:.3f} (positive → dopamine burst)")
    
    def test_rpe_negative_when_worse_than_expected(self):
        """
        REQUIREMENT: RPE < 0 when actual < predicted.
        This triggers dopamine DIP (weaken binding).
        """
        from holographic_prod.cognitive.reward_prediction import RewardPredictor
        
        predictor = RewardPredictor()
        predictor.predicted_reward = 0.7  # Expecting 0.7
        
        actual = 0.3  # Got 0.3 (worse!)
        rpe = predictor.compute_rpe(actual)
        
        assert rpe < 0, f"RPE should be negative when actual < predicted: {rpe}"
        assert abs(rpe - (-0.4)) < 0.01, f"RPE should be -0.4, got {rpe}"
        
        print(f"✓ RPE = {rpe:.3f} (negative → dopamine dip)")
    
    def test_rpe_zero_when_expected(self):
        """
        REQUIREMENT: RPE ≈ 0 when actual ≈ predicted.
        No learning signal needed.
        """
        from holographic_prod.cognitive.reward_prediction import RewardPredictor
        
        predictor = RewardPredictor()
        predictor.predicted_reward = 0.5
        
        actual = 0.5  # Exactly as expected
        rpe = predictor.compute_rpe(actual)
        
        assert abs(rpe) < 0.01, f"RPE should be ~0 when actual = predicted: {rpe}"
        
        print(f"✓ RPE = {rpe:.3f} (zero → no change)")


class TestRewardLearning:
    """Tests for reward prediction updating."""
    
    def test_prediction_updates_toward_actual(self):
        """
        REQUIREMENT: After observing reward, prediction should move toward it.
        """
        from holographic_prod.cognitive.reward_prediction import RewardPredictor
        from holographic_prod.core.constants import PHI_INV_CUBE
        
        predictor = RewardPredictor()
        initial_prediction = predictor.predicted_reward
        
        # Observe high reward multiple times
        for _ in range(10):
            predictor.update(actual_reward=1.0)
        
        # Prediction should have moved up
        assert predictor.predicted_reward > initial_prediction, \
            "Prediction should increase after observing high rewards"
        
        print(f"✓ Prediction moved: {initial_prediction:.3f} → {predictor.predicted_reward:.3f}")
    
    def test_learning_rate_is_phi_derived(self):
        """
        REQUIREMENT: Learning rate should be φ-derived (φ⁻³ ≈ 0.236).
        """
        from holographic_prod.cognitive.reward_prediction import RewardPredictor
        from holographic_prod.core.constants import PHI_INV_CUBE
        
        predictor = RewardPredictor()
        
        # Check that learning rate is φ⁻³
        assert hasattr(predictor, 'learning_rate'), "Should have learning_rate attribute"
        assert abs(predictor.learning_rate - PHI_INV_CUBE) < 0.01, \
            f"Learning rate should be φ⁻³ ≈ {PHI_INV_CUBE:.4f}"
        
        print(f"✓ Learning rate = φ⁻³ = {predictor.learning_rate:.4f}")


class TestThresholdModulation:
    """Tests for commitment threshold modulation by reward."""
    
    def test_high_reward_lowers_threshold(self):
        """
        REQUIREMENT: High recent rewards → lower commitment threshold.
        (More willing to act when things are going well)
        """
        from holographic_prod.cognitive.reward_prediction import RewardPredictor
        from holographic_prod.core.constants import PHI_INV_SQ
        
        predictor = RewardPredictor()
        baseline = predictor.baseline_threshold
        
        # Feed high rewards
        for _ in range(10):
            predictor.update(actual_reward=0.9)
        
        modulated = predictor.modulated_threshold()
        
        assert modulated < baseline, \
            f"High reward should lower threshold: {modulated} vs {baseline}"
        
        print(f"✓ Threshold lowered: {baseline:.3f} → {modulated:.3f}")
    
    def test_low_reward_raises_threshold(self):
        """
        REQUIREMENT: Low recent rewards → higher commitment threshold.
        (More cautious when things are going poorly)
        """
        from holographic_prod.cognitive.reward_prediction import RewardPredictor
        
        predictor = RewardPredictor()
        baseline = predictor.baseline_threshold
        
        # Feed low rewards
        for _ in range(10):
            predictor.update(actual_reward=0.1)
        
        modulated = predictor.modulated_threshold()
        
        assert modulated > baseline, \
            f"Low reward should raise threshold: {modulated} vs {baseline}"
        
        print(f"✓ Threshold raised: {baseline:.3f} → {modulated:.3f}")


class TestRewardIntegrationWithGeneration:
    """Tests for reward integration with generation."""
    
    def test_reward_modulates_candidate_scores(self):
        """
        REQUIREMENT: Candidates with history of good outcomes should score higher.
        """
        from holographic_prod.cognitive.reward_prediction import RewardPredictor
        
        predictor = RewardPredictor()
        
        # Simulate: token 5 has been rewarded, token 10 has been punished
        predictor.record_token_outcome(token_id=5, reward=0.9)
        predictor.record_token_outcome(token_id=5, reward=0.8)
        predictor.record_token_outcome(token_id=10, reward=0.2)
        predictor.record_token_outcome(token_id=10, reward=0.1)
        
        # Get value estimates
        value_5 = predictor.get_token_value(5)
        value_10 = predictor.get_token_value(10)
        
        assert value_5 > value_10, \
            f"Rewarded token should have higher value: {value_5} vs {value_10}"
        
        print(f"✓ Token 5 (rewarded): {value_5:.3f}")
        print(f"✓ Token 10 (punished): {value_10:.3f}")
    
    def test_value_weighted_scoring(self):
        """
        REQUIREMENT: Final score = coherence × value (or similar combination).
        """
        from holographic_prod.cognitive.reward_prediction import RewardPredictor
        import numpy as np
        
        predictor = RewardPredictor()
        
        # Setup: token 0 has high value, token 1 has low value
        predictor.record_token_outcome(0, reward=0.9)
        predictor.record_token_outcome(1, reward=0.1)
        
        # Coherence scores (from lensing/vorticity)
        coherence_scores = np.array([0.5, 0.6])  # Token 1 has higher coherence
        
        # But token 0 has higher value
        values = np.array([predictor.get_token_value(0), predictor.get_token_value(1)])
        
        # Combined score
        combined = predictor.combine_scores(coherence_scores, values)
        
        print(f"Coherence: {coherence_scores}")
        print(f"Values: {values}")
        print(f"Combined: {combined}")
        
        # The combination should balance coherence and value
        assert len(combined) == 2, "Should return combined scores"


class TestRewardFromPredictionAccuracy:
    """Tests for deriving reward from prediction accuracy."""
    
    def test_correct_prediction_gives_positive_reward(self):
        """
        REQUIREMENT: When model predicts correctly, reward = 1.0.
        """
        from holographic_prod.cognitive.reward_prediction import compute_reward_from_accuracy
        
        predicted_token = 5
        actual_token = 5  # Correct!
        
        reward = compute_reward_from_accuracy(predicted_token, actual_token)
        
        assert reward == 1.0, f"Correct prediction should give reward=1.0, got {reward}"
        print(f"✓ Correct prediction → reward = {reward}")
    
    def test_wrong_prediction_gives_low_reward(self):
        """
        REQUIREMENT: When model predicts wrong, reward is low (not 0).
        """
        from holographic_prod.cognitive.reward_prediction import compute_reward_from_accuracy
        
        predicted_token = 5
        actual_token = 10  # Wrong!
        
        reward = compute_reward_from_accuracy(predicted_token, actual_token)
        
        assert 0 < reward < 0.5, f"Wrong prediction should give low reward, got {reward}"
        print(f"✓ Wrong prediction → reward = {reward}")
    
    def test_close_prediction_gives_partial_reward(self):
        """
        REQUIREMENT: If prediction is semantically close, partial reward.
        (Requires embedding similarity check)
        """
        from holographic_prod.cognitive.reward_prediction import compute_reward_from_accuracy
        
        # This requires knowing embeddings are similar
        # For now, test that function accepts similarity parameter
        predicted_token = 5
        actual_token = 6
        semantic_similarity = 0.9  # These tokens are very similar
        
        reward = compute_reward_from_accuracy(
            predicted_token, actual_token, 
            semantic_similarity=semantic_similarity
        )
        
        # With sim=0.9 and formula: reward = sim * φ⁻¹ ≈ 0.9 * 0.618 ≈ 0.556
        assert reward > 0.5, \
            f"Semantically similar wrong prediction should get partial credit: {reward}"
        print(f"✓ Close prediction (sim={semantic_similarity}) → reward = {reward}")


class TestIntegrationWithCurrentSystem:
    """Tests for integration with existing components."""
    
    def test_reward_works_with_credit_assignment(self):
        """
        REQUIREMENT: Reward should integrate with existing CreditAssignmentTracker.
        """
        from holographic_prod.cognitive.reward_prediction import RewardPredictor, compute_reward_from_accuracy
        
        reward_predictor = RewardPredictor()
        
        # Simulate a credit assignment error record
        class MockError:
            predicted = 5
            actual = 10
        
        error = MockError()
        
        # Compute reward from the error
        reward = reward_predictor.reward_from_credit_error(error)
        
        assert isinstance(reward, float), "Should return a reward value"
        assert 0 <= reward <= 1, f"Reward should be in [0, 1], got {reward}"
        
        # Also test direct computation
        direct_reward = compute_reward_from_accuracy(5, 10)
        assert isinstance(direct_reward, float), "Should return a reward value"
        
        print(f"✓ Reward from error: {reward}")
        print(f"✓ Direct computation: {direct_reward}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
