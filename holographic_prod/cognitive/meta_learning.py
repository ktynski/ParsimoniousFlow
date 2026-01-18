"""
Meta-Learning Module — Adaptive Parameters
==========================================

This module makes phi-derived parameters ADAPTIVE based on context.

THEORY:
    The phi-based parameters (φ⁻¹, φ⁻², φ⁻³) are DEFAULTS that emerge
    from the self-consistency equation. But optimal learning depends on:
    
    - SALIENCE: Important content should be learned faster
    - NOVELTY: New patterns should be learned faster
    - UNCERTAINTY: When uncertain, learn more cautiously
    
    The adaptive rates modulate AROUND the phi defaults, never straying
    too far from the mathematically optimal values.

KEY INSIGHT:
    This is NOT hyperparameter tuning. It's context-sensitive modulation
    of fixed parameters within phi-derived bounds.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass, field

from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR, PHI_INV_SIX

Array = np.ndarray


# =============================================================================
# LEARNING STATE
# =============================================================================

@dataclass
class LearningState:
    """
    Tracks meta-learning state across training.
    
    This evolves over time based on prediction success/failure.
    """
    effective_learning_rate: float = PHI_INV
    effective_consolidation_threshold: float = PHI_INV_SQ
    uncertainty: float = PHI_INV_SQ  # φ²-derived initial epistemic uncertainty
    recent_error_rate: float = 0.0  # Moving average of errors
    
    # History tracking (for computing moving averages)
    _error_history: list = field(default_factory=list)
    _max_history: int = 100


# =============================================================================
# ADAPTIVE LEARNING RATE
# =============================================================================

def compute_adaptive_learning_rate(
    salience: float,
    novelty: float,
    uncertainty: float,
    base_rate: float = PHI_INV,
) -> float:
    """
    Modulate learning rate based on context.
    
    THEORY:
        - High salience → faster learning (important content!)
        - High novelty → faster learning (new pattern to capture!)
        - High uncertainty → slower learning (don't overwrite good stuff)
        
    The rate is bounded by [base_rate * φ⁻¹, base_rate * φ] to stay
    within the mathematically stable region.
    
    Args:
        salience: Importance of content [0, 1]
        novelty: How new the pattern is [0, 1]
        uncertainty: Epistemic uncertainty [0, 1]
        base_rate: Default rate (typically φ⁻¹)
        
    Returns:
        Modulated learning rate
    """
    # Clamp inputs to [0, 1]
    salience = max(0.0, min(1.0, salience))
    novelty = max(0.0, min(1.0, novelty))
    uncertainty = max(0.0, min(1.0, uncertainty))
    
    # Compute modulation factors (φ-weighted)
    # Salience (importance) weighted by φ⁻¹, novelty weighted by φ⁻²
    # This matches Grace eigenvalue structure: lower grades more important
    increase_factor = PHI_INV * salience + PHI_INV_SQ * novelty
    # Uncertainty decreases rate (weighted by its complement)
    decrease_factor = PHI_INV * uncertainty
    
    # Net modulation: centered at 0, range roughly [-1, 1]
    net_modulation = increase_factor - decrease_factor
    
    # Map to multiplicative factor in [φ⁻¹, φ]
    # net_modulation = -1 → factor = φ⁻¹
    # net_modulation = 0 → factor = 1
    # net_modulation = +1 → factor = φ
    if net_modulation >= 0:
        factor = 1.0 + net_modulation * (PHI - 1.0)
    else:
        factor = 1.0 + net_modulation * (1.0 - PHI_INV)
    
    # Apply to base rate
    rate = base_rate * factor
    
    # Clamp to bounds
    min_rate = base_rate * PHI_INV
    max_rate = base_rate * PHI
    
    return max(min_rate, min(max_rate, rate))


# =============================================================================
# ADAPTIVE CONSOLIDATION
# =============================================================================

def compute_adaptive_consolidation(
    error_rate: float,
    memory_pressure: float,
    base_threshold: float = PHI_INV_SQ,
) -> float:
    """
    Adjust consolidation threshold based on system state.
    
    THEORY:
        - High error rate → consolidate less (prototypes might be wrong)
        - High memory pressure → consolidate more (need to compress)
        
    Args:
        error_rate: Recent prediction error rate [0, 1]
        memory_pressure: How full is memory [0, 1]
        base_threshold: Default threshold (typically φ⁻²)
        
    Returns:
        Adjusted consolidation threshold
    """
    # Clamp inputs
    error_rate = max(0.0, min(1.0, error_rate))
    memory_pressure = max(0.0, min(1.0, memory_pressure))
    
    # Error rate increases threshold (less consolidation)
    # Memory pressure decreases threshold (more consolidation)
    # φ-derived weights: error_rate weighted by φ⁻¹, memory_pressure by φ⁻²
    adjustment = PHI_INV * error_rate - PHI_INV_SQ * memory_pressure
    
    # Apply adjustment
    # Positive adjustment → higher threshold → less consolidation
    # Negative adjustment → lower threshold → more consolidation
    threshold = base_threshold * (1.0 + adjustment)
    
    # Clamp to reasonable bounds
    min_threshold = base_threshold * PHI_INV  # More consolidation
    max_threshold = base_threshold * PHI  # Less consolidation
    
    return max(min_threshold, min(max_threshold, threshold))


# =============================================================================
# META STATE UPDATE
# =============================================================================

def update_meta_state(
    state: LearningState,
    prediction_correct: bool,
    salience: float,
    novelty: float,
) -> LearningState:
    """
    Update meta-learning state after each prediction.
    
    Args:
        state: Current learning state
        prediction_correct: Whether last prediction was correct
        salience: Salience of last sample
        novelty: Novelty of last sample
        
    Returns:
        Updated learning state
    """
    # Update error history
    error = 0.0 if prediction_correct else 1.0
    state._error_history.append(error)
    
    # Keep history bounded
    if len(state._error_history) > state._max_history:
        state._error_history = state._error_history[-state._max_history:]
    
    # Compute moving average error rate
    if state._error_history:
        state.recent_error_rate = sum(state._error_history) / len(state._error_history)
    
    # Update uncertainty based on recent errors
    # More errors → more uncertainty
    # φ-derived: base uncertainty φ⁻³, scales by φ⁻¹
    state.uncertainty = PHI_INV_CUBE + PHI_INV * state.recent_error_rate
    
    # Update effective learning rate
    state.effective_learning_rate = compute_adaptive_learning_rate(
        salience=salience,
        novelty=novelty,
        uncertainty=state.uncertainty,
    )
    
    # Update consolidation threshold
    memory_pressure = len(state._error_history) / state._max_history
    state.effective_consolidation_threshold = compute_adaptive_consolidation(
        error_rate=state.recent_error_rate,
        memory_pressure=memory_pressure,
    )
    
    return state


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_learning_state() -> LearningState:
    """Create a fresh learning state with defaults."""
    return LearningState()


def get_adaptive_parameters(state: LearningState) -> Dict[str, float]:
    """
    Get current adaptive parameters from state.
    
    Returns:
        Dict of parameter name to value
    """
    return {
        'learning_rate': state.effective_learning_rate,
        'consolidation_threshold': state.effective_consolidation_threshold,
        'uncertainty': state.uncertainty,
        'recent_error_rate': state.recent_error_rate,
    }


def estimate_optimal_rate(
    recent_performance: list,
    default_rate: float = PHI_INV,
) -> float:
    """
    Estimate optimal learning rate from recent performance.
    
    Simple heuristic: if performance is improving, maintain rate.
    If degrading, slow down.
    
    Args:
        recent_performance: List of recent accuracy values
        default_rate: Fallback rate
        
    Returns:
        Estimated optimal rate
    """
    if len(recent_performance) < 2:
        return default_rate
    
    # Check trend
    recent = np.array(recent_performance[-10:])
    if len(recent) < 2:
        return default_rate
    
    # Simple linear regression slope
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    
    if slope > 0:
        # Improving - maintain or φ⁻⁴ increase
        return default_rate * (1.0 + PHI_INV_FOUR * min(1.0, slope))
    else:
        # Degrading - φ⁻³ slowdown
        return default_rate * (1.0 - PHI_INV_CUBE * min(1.0, abs(slope)))


# =============================================================================
# φ-SCALED CURRICULUM LEARNING RATE
# =============================================================================

def phi_scaled_learning_rate(
    current_step: int,
    total_steps: int,
    base_rate: float = PHI_INV,
    warmup_fraction: float = PHI_INV_CUBE,  # φ⁻³ ≈ 0.236 (was 0.1)
) -> float:
    """
    Theory-true φ-scaled learning rate schedule.
    
    THEORY (from rhnsclifford.md):
        Information flows in φ-ratio: I(A:B)/I(B:C) = φ
        Learning rate should decay as φ^(-t/τ) for smooth convergence.
    
    SCHEDULE:
        1. Warmup: Linear increase from base_rate × φ⁻² to base_rate
        2. Decay: φ^(-progress) from base_rate to base_rate × φ⁻²
    
    Args:
        current_step: Current training step
        total_steps: Total expected training steps
        base_rate: Peak learning rate (default φ⁻¹)
        warmup_fraction: Fraction of steps for warmup (default 0.1)
    
    Returns:
        Learning rate for current step
    """
    if total_steps <= 0:
        return base_rate
    
    progress = current_step / total_steps
    warmup_end = warmup_fraction
    
    # Bounds
    min_rate = base_rate * PHI_INV_SQ  # φ⁻³ minimum
    max_rate = base_rate  # φ⁻¹ maximum
    
    if progress < warmup_end:
        # Warmup: linear increase
        warmup_progress = progress / warmup_end
        return min_rate + (max_rate - min_rate) * warmup_progress
    else:
        # Decay: φ-scaled
        decay_progress = (progress - warmup_end) / (1.0 - warmup_end)
        # φ^(-2 × progress) gives smooth decay from 1 to φ⁻²
        decay_factor = PHI ** (-2 * decay_progress)
        return min_rate + (max_rate - min_rate) * decay_factor


# =============================================================================
# VERIFIED RETRIEVAL
# =============================================================================

def verified_retrieve(
    memory,
    context: Array,
    basis: Array,
    xp = np,
    agreement_threshold: float = PHI_INV,  # φ-derived: φ⁻¹ ≈ 0.618 (was 0.8)
    noise_std: float = PHI_INV_SIX,  # φ⁻⁶ ≈ 0.056 perturbation
) -> tuple:
    """
    Retrieval with perturbation-based verification.
    
    THEORY (bireflection-inspired error correction):
        Query both original and perturbed context.
        If results agree → high confidence (redundancy validates)
        If results disagree → low confidence (interference detected)
    
    This catches holographic interference errors by checking consistency.
    
    Args:
        memory: HybridHolographicMemory or similar with .retrieve() method
        context: Query context matrix [4, 4]
        basis: Clifford basis
        xp: Array module (numpy/cupy)
        agreement_threshold: φ-derived threshold for "high agreement" (φ⁻¹ ≈ 0.618)
        noise_std: Standard deviation of perturbation noise (φ⁻⁶ ≈ 0.056)
    
    Returns:
        (result, target_idx, adjusted_confidence, status)
        status is "verified" or "uncertain"
    """
    # Primary retrieval
    result_A, idx_A, conf_A, source_A = memory.retrieve(context)
    
    # Perturbed retrieval for verification
    noise = xp.random.randn(4, 4).astype(context.dtype) * noise_std
    perturbed = context + noise
    # Normalize
    perturbed = perturbed / (xp.sqrt(xp.sum(perturbed * perturbed)) + 1e-8)
    
    result_B, idx_B, conf_B, source_B = memory.retrieve(perturbed)
    
    # Handle None results
    if result_A is None:
        if result_B is None:
            return None, -1, 0.0, "no_match"
        return result_B, idx_B, conf_B * PHI_INV_SQ, "uncertain"  # φ²-penalized uncertainty
    
    if result_B is None:
        return result_A, idx_A, conf_A * PHI_INV_SQ, "uncertain"  # φ²-penalized uncertainty
    
    # Compute agreement (cosine similarity)
    norm_A = xp.sqrt(xp.sum(result_A * result_A)) + 1e-8
    norm_B = xp.sqrt(xp.sum(result_B * result_B)) + 1e-8
    agreement = float(xp.sum(result_A * result_B) / (norm_A * norm_B))
    
    if agreement > agreement_threshold:
        # High agreement - boost confidence
        adjusted_conf = conf_A * (1.0 + agreement) / 2.0
        return result_A, idx_A, adjusted_conf, "verified"
    else:
        # Low agreement - penalize confidence
        adjusted_conf = conf_A * agreement
        return result_A, idx_A, adjusted_conf, "uncertain"
