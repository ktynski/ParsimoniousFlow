"""
Adaptive Memory — Integrated FractalGenerativeMemory with Meta-Learning
=======================================================================

Combines:
1. FractalGenerativeMemory — Hierarchical holographic storage
2. CreditAssignmentTracker — Error tracking and reconsolidation
3. Meta-learning state — Adaptive φ-derived rates

This is the PRODUCTION-READY memory system with all theory-true features.

THEORY:
    Learning rate is modulated by:
    - NOVELTY: New contexts → faster learning
    - UNCERTAINTY: Low confidence → slower learning
    - SALIENCE: Rare patterns → faster learning (rarity = salience)
    
    All modulations stay within φ-derived bounds:
    [base_rate × φ⁻¹, base_rate × φ]
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from holographic_v4.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
    MATRIX_DIM,
)
from holographic_v4.fractal_generative_memory import (
    FractalGenerativeMemory,
    FractalGenerativeConfig,
)
from holographic_v4.credit_assignment import (
    CreditAssignmentTracker,
    ReconsolidationConfig,
)
from holographic_v4.meta_learning import (
    LearningState,
    create_learning_state,
    update_meta_state,
    compute_adaptive_learning_rate,
    phi_scaled_learning_rate,
    get_adaptive_parameters,
)
from holographic_v4.algebra import geometric_product


# =============================================================================
# ADAPTIVE MEMORY CONFIG
# =============================================================================

@dataclass
class AdaptiveMemoryConfig:
    """
    Configuration for AdaptiveMemory.
    
    ALL VALUES ARE φ-DERIVED.
    """
    # Base config for underlying memory
    memory_config: FractalGenerativeConfig = field(default_factory=FractalGenerativeConfig)
    
    # Credit assignment config
    recon_config: ReconsolidationConfig = field(default_factory=ReconsolidationConfig)
    
    # Meta-learning
    use_adaptive_rates: bool = True
    use_curriculum_schedule: bool = True
    total_steps: int = 10000  # For curriculum schedule
    
    # Novelty detection
    novelty_threshold: float = PHI_INV_SQ  # Similarity below this = novel


# =============================================================================
# ADAPTIVE MEMORY
# =============================================================================

class AdaptiveMemory:
    """
    Production-ready memory with adaptive learning rates.
    
    Integrates:
    - FractalGenerativeMemory for storage
    - CreditAssignmentTracker for error correction
    - Meta-learning state for adaptive rates
    
    Usage:
        memory = AdaptiveMemory(vocab_size=10000)
        
        for context, target in data:
            stats = memory.learn_adaptive(context, target)
            # stats includes: predicted, correct, rate_used, etc.
        
        # Retrieve
        token, confidence = memory.retrieve(context)
        
        # Generate
        tokens, stats = memory.generate(prompt, max_tokens=20)
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        max_levels: int = 2,
        config: AdaptiveMemoryConfig = None,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.max_levels = max_levels
        self.config = config or AdaptiveMemoryConfig()
        self.seed = seed
        
        # Initialize underlying memory
        self.memory = FractalGenerativeMemory(
            max_levels=max_levels,
            vocab_size=vocab_size,
            orthogonalize=self.config.memory_config.orthogonalize,
            contrastive_enabled=True,
            seed=seed,
            config=self.config.memory_config,
        )
        
        # Initialize credit assignment
        self.credit_tracker = CreditAssignmentTracker(
            self.memory,
            self.config.recon_config,
        )
        
        # Initialize meta-learning state
        self.meta_state = create_learning_state()
        
        # Training state
        self.current_step = 0
        self.total_correct = 0
        self.total_predictions = 0
        
        # Context frequency for salience estimation
        self.context_counts: Dict[int, int] = {}
    
    # =========================================================================
    # LEARNING
    # =========================================================================
    
    def learn_adaptive(
        self,
        context: List[int],
        target: int,
    ) -> Dict[str, Any]:
        """
        Learn with adaptive rates based on novelty and uncertainty.
        
        This is the main learning interface. It:
        1. Measures novelty (is this context new?)
        2. Predicts to measure uncertainty
        3. Computes adaptive rate
        4. Learns with modulated rate
        5. Records errors for reconsolidation
        6. Updates meta state
        
        Args:
            context: List of token IDs
            target: Target token to predict
            
        Returns:
            Statistics dictionary
        """
        ctx_hash = hash(tuple(context))
        
        # 1. Measure novelty
        novelty = self._compute_novelty(context)
        
        # 2. Predict to measure uncertainty
        predicted, confidence = self.memory.retrieve_deterministic(context)
        uncertainty = self._compute_uncertainty(confidence)
        was_correct = (predicted == target)
        
        # 3. Compute salience (rarer patterns = more salient)
        salience = self._compute_salience(context)
        
        # 4. Compute adaptive rate
        if self.config.use_adaptive_rates:
            base_rate = self._get_base_rate()
            adaptive_rate = compute_adaptive_learning_rate(
                salience=salience,
                novelty=novelty,
                uncertainty=uncertainty,
                base_rate=base_rate,
            )
        else:
            adaptive_rate = PHI_INV  # Fixed default
        
        # 5. Learn with adaptive rate
        # Store original rate, apply adaptive, then restore
        original_rate = self.memory.config.learning_rate
        self.memory.config.learning_rate = adaptive_rate
        self.memory.learn(context, target)
        self.memory.config.learning_rate = original_rate
        
        # 6. Record error if wrong
        if not was_correct and predicted is not None:
            self.credit_tracker.record_error(
                context=context,
                predicted=predicted,
                actual=target,
                confidence=confidence,
            )
        
        # 7. Update meta state
        self.meta_state = update_meta_state(
            self.meta_state,
            prediction_correct=was_correct,
            salience=salience,
            novelty=novelty,
        )
        
        # 8. Update counts
        self.context_counts[ctx_hash] = self.context_counts.get(ctx_hash, 0) + 1
        self.current_step += 1
        self.total_predictions += 1
        if was_correct:
            self.total_correct += 1
        
        # 9. Periodic reconsolidation
        recon_stats = {}
        if self.credit_tracker.should_reconsolidate():
            recon_stats = self.credit_tracker.reconsolidate()
        
        return {
            'predicted': predicted,
            'actual': target,
            'correct': was_correct,
            'confidence': confidence,
            'novelty': novelty,
            'uncertainty': uncertainty,
            'salience': salience,
            'rate_used': adaptive_rate,
            'step': self.current_step,
            'reconsolidation': recon_stats,
        }
    
    def learn_batch(
        self,
        batch: List[Tuple[List[int], int]],
    ) -> Dict[str, Any]:
        """
        Learn a batch of (context, target) pairs.
        
        Args:
            batch: List of (context, target) tuples
            
        Returns:
            Aggregated statistics
        """
        stats = {
            'n_samples': len(batch),
            'n_correct': 0,
            'avg_rate': 0.0,
            'total_reconsolidated': 0,
        }
        
        for context, target in batch:
            result = self.learn_adaptive(context, target)
            if result['correct']:
                stats['n_correct'] += 1
            stats['avg_rate'] += result['rate_used']
            stats['total_reconsolidated'] += result['reconsolidation'].get('processed', 0)
        
        stats['avg_rate'] /= max(1, len(batch))
        stats['accuracy'] = stats['n_correct'] / max(1, len(batch))
        
        return stats
    
    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    
    def retrieve(self, context: List[int]) -> Tuple[Optional[int], float]:
        """
        Retrieve the most likely target for a context.
        
        Returns:
            (token_id, confidence) or (None, 0.0)
        """
        return self.memory.retrieve_deterministic(context)
    
    def retrieve_probabilistic(
        self,
        context: List[int],
        temperature: float = None,
        top_k: int = None,
    ) -> Tuple[Optional[int], float, List[Tuple[int, float]]]:
        """
        Sample a target probabilistically.
        
        Returns:
            (sampled_token, probability, top_k_with_probs)
        """
        return self.memory.retrieve_probabilistic(context, temperature, top_k)
    
    # =========================================================================
    # GENERATION
    # =========================================================================
    
    def generate(
        self,
        prompt: List[int],
        max_tokens: int = 20,
        temperature: float = None,
        context_size: int = 3,
    ) -> Tuple[List[int], Dict]:
        """
        Generate tokens autoregressively.
        
        Returns:
            (generated_tokens, stats)
        """
        return self.memory.generate(prompt, max_tokens, temperature, context_size)
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _compute_novelty(self, context: List[int]) -> float:
        """
        Compute novelty of a context.
        
        THEORY: Context is novel if not in memory or has low similarity
        to stored contexts.
        
        Returns:
            Novelty in [0, 1], where 1 = completely new
        """
        ctx_hash = hash(tuple(context))
        
        if ctx_hash not in self.memory.memory:
            return 1.0  # Completely new
        
        # Context exists - check how often seen
        count = self.context_counts.get(ctx_hash, 0)
        if count == 0:
            return 1.0
        
        # More occurrences = less novel
        # φ-derived decay: novelty = φ^(-count/10)
        novelty = PHI ** (-count / 10.0)
        return max(0.0, min(1.0, novelty))
    
    def _compute_uncertainty(self, confidence: float) -> float:
        """
        Compute uncertainty from retrieval confidence.
        
        THEORY: Low confidence = high uncertainty
        
        Returns:
            Uncertainty in [0, 1]
        """
        if confidence is None or confidence <= 0:
            return 1.0  # Maximum uncertainty
        
        # Confidence is typically in [0, ~2] for Frobenius similarity
        # Map to uncertainty: high confidence → low uncertainty
        uncertainty = 1.0 - min(1.0, confidence)
        return max(0.0, min(1.0, uncertainty))
    
    def _compute_salience(self, context: List[int]) -> float:
        """
        Compute salience (importance) of a context.
        
        THEORY: Rarer patterns are more salient.
        
        Returns:
            Salience in [0, 1], where 1 = very rare/important
        """
        ctx_hash = hash(tuple(context))
        count = self.context_counts.get(ctx_hash, 0)
        
        if count == 0:
            return 1.0  # Never seen = highly salient
        
        # Inverse frequency, capped
        # φ-derived: salience = φ^(-log(count + 1))
        salience = PHI ** (-np.log(count + 1))
        return max(0.0, min(1.0, salience))
    
    def _get_base_rate(self) -> float:
        """
        Get base learning rate, possibly with curriculum schedule.
        
        Returns:
            Base rate for this step
        """
        if self.config.use_curriculum_schedule:
            return phi_scaled_learning_rate(
                current_step=self.current_step,
                total_steps=self.config.total_steps,
                base_rate=PHI_INV,
            )
        return PHI_INV
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        meta_params = get_adaptive_parameters(self.meta_state)
        credit_stats = self.credit_tracker.get_statistics()
        memory_stats = self.memory.get_statistics() if hasattr(self.memory, 'get_statistics') else {}
        
        accuracy = self.total_correct / max(1, self.total_predictions)
        
        return {
            'current_step': self.current_step,
            'total_correct': self.total_correct,
            'total_predictions': self.total_predictions,
            'accuracy': accuracy,
            'meta_learning': meta_params,
            'credit_assignment': credit_stats,
            'memory': memory_stats,
            'unique_contexts': len(self.context_counts),
        }
    
    def reset_stats(self):
        """Reset statistics (but keep learned memory)."""
        self.current_step = 0
        self.total_correct = 0
        self.total_predictions = 0
        self.meta_state = create_learning_state()
        self.context_counts.clear()


# =============================================================================
# CONVENIENCE FACTORY
# =============================================================================

def create_adaptive_memory(
    vocab_size: int = 1000,
    max_levels: int = 2,
    use_adaptive_rates: bool = True,
    use_curriculum: bool = True,
    total_steps: int = 10000,
    seed: int = 42,
) -> AdaptiveMemory:
    """
    Factory function for creating AdaptiveMemory.
    
    Args:
        vocab_size: Vocabulary size
        max_levels: Fractal hierarchy levels
        use_adaptive_rates: Enable novelty/uncertainty modulation
        use_curriculum: Enable φ-scaled learning rate schedule
        total_steps: Total expected training steps (for curriculum)
        seed: Random seed
        
    Returns:
        Configured AdaptiveMemory instance
    """
    config = AdaptiveMemoryConfig(
        use_adaptive_rates=use_adaptive_rates,
        use_curriculum_schedule=use_curriculum,
        total_steps=total_steps,
    )
    
    return AdaptiveMemory(
        vocab_size=vocab_size,
        max_levels=max_levels,
        config=config,
        seed=seed,
    )
