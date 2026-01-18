"""
Reward Prediction Error (RPE) — VTA/NAc Dopamine Analog
========================================================

THEORY:
    The brain's reward system (VTA → NAc) computes:
        RPE = actual_reward - predicted_reward
    
    Positive RPE → dopamine burst → strengthen binding
    Negative RPE → dopamine dip → weaken binding
    Zero RPE → no change (expected outcome)

INTEGRATION:
    This module provides the QUALITY SIGNAL that IoR/φ-kernel lack.
    - IoR: Prevents repetition (but doesn't know what's good)
    - φ-kernel: Adds diversity (but doesn't know what's good)
    - Reward: Learns what outputs are GOOD (credit assignment)

ALL CONSTANTS ARE φ-DERIVED.

Version: v5.18.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from ..core.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE


@dataclass
class RewardPredictor:
    """
    VTA/NAc analog: Predicts reward, computes RPE, modulates behavior.
    
    BRAIN ANALOG:
        - VTA (Ventral Tegmental Area): Source of dopamine
        - NAc (Nucleus Accumbens): Reward prediction
        - Dopamine burst (positive RPE): "Better than expected!"
        - Dopamine dip (negative RPE): "Worse than expected!"
    
    THEORY-TRUE:
        - Learning rate = φ⁻³ ≈ 0.236 (tertiary rate)
        - Baseline threshold = φ⁻² ≈ 0.382 (spectral gap)
        - Modulation factor = φ⁻¹ ≈ 0.618 (primary rate)
    """
    
    # φ-derived constants
    learning_rate: float = PHI_INV_CUBE      # φ⁻³ ≈ 0.236
    baseline_threshold: float = PHI_INV_SQ   # φ⁻² ≈ 0.382
    modulation_factor: float = PHI_INV       # φ⁻¹ ≈ 0.618
    
    # State
    predicted_reward: float = 0.5            # Initial expectation (neutral)
    reward_history: List[float] = field(default_factory=list)
    
    # Per-token value estimates (learned from outcomes)
    token_values: Dict[int, float] = field(default_factory=dict)
    token_counts: Dict[int, int] = field(default_factory=dict)
    
    def compute_rpe(self, actual_reward: float) -> float:
        """
        Compute Reward Prediction Error.
        
        RPE = actual - predicted
        
        Positive RPE: Better than expected → dopamine burst
        Negative RPE: Worse than expected → dopamine dip
        Zero RPE: As expected → no change
        """
        return actual_reward - self.predicted_reward
    
    def update(self, actual_reward: float) -> float:
        """
        Update reward prediction based on observed outcome.
        
        Uses temporal difference learning:
            prediction += learning_rate * RPE
        
        Returns:
            RPE (for logging/debugging)
        """
        rpe = self.compute_rpe(actual_reward)
        
        # Update prediction toward actual (TD learning)
        self.predicted_reward += self.learning_rate * rpe
        
        # Clamp to [0, 1]
        self.predicted_reward = max(0.0, min(1.0, self.predicted_reward))
        
        # Track history
        self.reward_history.append(actual_reward)
        
        return rpe
    
    def modulated_threshold(self) -> float:
        """
        Compute commitment threshold modulated by reward history.
        
        BRAIN ANALOG:
            - High recent reward → lower threshold (more willing to act)
            - Low recent reward → higher threshold (more cautious)
        
        This integrates with CommitmentGate.
        """
        if len(self.reward_history) < 3:
            return self.baseline_threshold
        
        # Recent reward average (last 10)
        recent = np.mean(self.reward_history[-10:])
        
        # Modulate: high reward lowers threshold, low reward raises it
        # centered around 0.5 (neutral)
        modulation = self.modulation_factor * (recent - 0.5)
        
        # Threshold adjustment
        threshold = self.baseline_threshold * (1 - modulation)
        
        # Clamp to reasonable range
        return max(0.1, min(0.9, threshold))
    
    def record_token_outcome(self, token_id: int, reward: float) -> None:
        """
        Record the outcome of selecting a specific token.
        
        This builds per-token value estimates for generation.
        """
        if token_id not in self.token_values:
            self.token_values[token_id] = 0.5  # Prior: neutral
            self.token_counts[token_id] = 0
        
        # Incremental mean update
        self.token_counts[token_id] += 1
        n = self.token_counts[token_id]
        old_value = self.token_values[token_id]
        
        # Running average with decay toward recent
        alpha = max(self.learning_rate, 1.0 / n)  # At least φ⁻³
        self.token_values[token_id] = old_value + alpha * (reward - old_value)
    
    def get_token_value(self, token_id: int) -> float:
        """
        Get the learned value estimate for a token.
        
        Returns 0.5 (neutral) for unseen tokens.
        """
        return self.token_values.get(token_id, 0.5)
    
    def combine_scores(
        self, 
        coherence_scores: np.ndarray, 
        token_ids: Optional[np.ndarray] = None,
        value_weight: float = None
    ) -> np.ndarray:
        """
        Combine coherence scores with learned value estimates.
        
        Final score = coherence^(1-w) × value^w
        
        where w = value_weight (default: φ⁻¹ ≈ 0.382)
        
        This is a geometric mean that balances:
        - Coherence: How well does this fit the current context?
        - Value: How good has this token been historically?
        """
        if value_weight is None:
            value_weight = PHI_INV_SQ  # φ⁻² ≈ 0.382
        
        # Get value estimates for each candidate
        if token_ids is not None:
            values = np.array([self.get_token_value(int(t)) for t in token_ids])
        else:
            # Assume indices 0, 1, 2, ... correspond to token_ids
            values = np.array([
                self.get_token_value(i) for i in range(len(coherence_scores))
            ])
        
        # Ensure positive values for geometric mean
        coherence_safe = np.maximum(coherence_scores, 1e-10)
        values_safe = np.maximum(values, 1e-10)
        
        # Geometric combination
        combined = (coherence_safe ** (1 - value_weight)) * (values_safe ** value_weight)
        
        return combined
    
    def reward_from_credit_error(self, error: Any) -> float:
        """
        Convert a CreditAssignmentTracker error to a reward value.
        
        Correct prediction → reward = 1.0
        Wrong prediction → reward based on semantic similarity
        """
        if hasattr(error, 'predicted') and hasattr(error, 'actual'):
            if error.predicted == error.actual:
                return 1.0
            else:
                # Could incorporate semantic similarity here
                return 0.2  # Base reward for wrong prediction
        return 0.5  # Unknown → neutral


def compute_reward_from_accuracy(
    predicted_token: int,
    actual_token: int,
    semantic_similarity: float = 0.0
) -> float:
    """
    Compute reward from prediction accuracy.
    
    Args:
        predicted_token: The model's prediction
        actual_token: The ground truth
        semantic_similarity: Optional [0, 1] similarity between tokens
    
    Returns:
        Reward in [0, 1]:
            - 1.0 for exact match
            - semantic_similarity * PHI_INV for close match
            - PHI_INV_SQ for wrong prediction
    
    THEORY:
        Partial credit for semantically similar predictions
        encourages learning of semantic structure.
    """
    if predicted_token == actual_token:
        return 1.0
    
    if semantic_similarity > 0:
        # Partial credit based on semantic similarity
        # Scaled by φ⁻¹ to keep it below exact match
        return semantic_similarity * PHI_INV
    
    # Wrong prediction with no similarity info
    return PHI_INV_SQ  # ≈ 0.382 (not zero, to avoid harsh penalties)


# =============================================================================
# Integration with Generation
# =============================================================================

def create_reward_weighted_scorer(
    reward_predictor: RewardPredictor,
    coherence_weight: float = None
) -> callable:
    """
    Create a scoring function that combines coherence and value.
    
    Usage in generation:
        scorer = create_reward_weighted_scorer(reward_predictor)
        combined_scores = scorer(coherence_scores, candidate_ids)
    """
    if coherence_weight is None:
        coherence_weight = 1 - PHI_INV_SQ  # ≈ 0.618
    
    def scorer(coherence_scores: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
        return reward_predictor.combine_scores(
            coherence_scores, 
            candidate_ids,
            value_weight=1 - coherence_weight
        )
    
    return scorer
