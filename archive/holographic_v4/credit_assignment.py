"""
Credit Assignment v2 — Lean Integration with FractalGenerativeMemory
=====================================================================

Theory-true, compute-efficient credit assignment for holographic memory.

CORE INSIGHT:
    In accumulation-based memory, credit assignment is simple:
    
    1. Memory[ctx_hash] contains superposition of ALL targets seen for that context
    2. When prediction is wrong:
       - BOOST: Add more weight to correct target binding
       - ATTENUATE: Subtract weight from incorrect prediction binding
    
    This is Hebbian/anti-Hebbian learning with φ-scaled rates.

WHY THIS IS THEORY-TRUE:
    - No gradients needed (direct memory modification)
    - All rates are φ-derived
    - Operates on holographic superposition (not discrete entries)
    - Respects accumulation semantics

COMPUTATIONAL EFFICIENCY:
    - O(1) per error (direct hash lookup)
    - No iteration through attractors
    - Batch support for multiple errors
    - Memory-efficient (no trace storage needed)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque

from holographic_v4.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR,
    MATRIX_DIM,
)
from holographic_v4.algebra import (
    geometric_product,
    frobenius_similarity,
    grace_operator,
    build_clifford_basis,
)


# =============================================================================
# ERROR RECORD
# =============================================================================

@dataclass
class ErrorRecord:
    """
    Minimal record of a prediction error.
    
    Only stores what's needed for reconsolidation.
    No heavy matrices - just hashes and indices.
    """
    ctx_hash: int           # Hash of context (for memory lookup)
    context: Tuple[int, ...]  # Original context tokens
    predicted: int          # What was predicted
    actual: int            # What was correct
    confidence: float      # How confident the wrong prediction was
    timestamp: int = 0     # Step when error occurred
    
    @property
    def error_magnitude(self) -> float:
        """
        φ-derived error magnitude.
        
        Higher confidence in wrong answer = higher magnitude.
        """
        # Base error is φ⁻¹ for any wrong prediction
        # Scaled by confidence (more confident = worse)
        return PHI_INV * (PHI_INV + self.confidence * (1 - PHI_INV))


# =============================================================================
# RECONSOLIDATION CONFIG
# =============================================================================

@dataclass
class ReconsolidationConfig:
    """
    Configuration for credit assignment.
    
    ALL VALUES ARE φ-DERIVED.
    """
    # Boost rate: how much to reinforce correct target
    boost_rate: float = PHI_INV_SQ  # φ⁻² ≈ 0.382
    
    # Attenuate rate: how much to weaken wrong prediction
    attenuate_rate: float = PHI_INV_CUBE  # φ⁻³ ≈ 0.236
    
    # Minimum error magnitude to trigger reconsolidation
    min_error_threshold: float = PHI_INV_FOUR  # φ⁻⁴ ≈ 0.146
    
    # Maximum errors to track (rolling window)
    max_history: int = 100
    
    # Batch size for reconsolidation (process N errors at once)
    batch_size: int = 16


# =============================================================================
# CREDIT ASSIGNMENT TRACKER
# =============================================================================

class CreditAssignmentTracker:
    """
    Lean credit assignment for FractalGenerativeMemory.
    
    Design principles:
    1. O(1) per error (no searching)
    2. Batch reconsolidation for efficiency
    3. Rolling window to bound memory
    4. Theory-true φ-scaled rates
    """
    
    def __init__(
        self,
        memory: 'FractalGenerativeMemory',
        config: ReconsolidationConfig = None,
    ):
        """
        Initialize tracker.
        
        Args:
            memory: FractalGenerativeMemory instance
            config: Configuration (uses defaults if None)
        """
        self.memory = memory
        self.config = config or ReconsolidationConfig()
        
        # Rolling window of errors (bounded memory)
        self.errors: deque = deque(maxlen=self.config.max_history)
        
        # Statistics
        self.total_errors = 0
        self.total_reconsolidations = 0
        self.step = 0
    
    def record_error(
        self,
        context: List[int],
        predicted: int,
        actual: int,
        confidence: float = 0.5,
    ):
        """
        Record a prediction error.
        
        Args:
            context: The context tokens
            predicted: What was predicted (wrong)
            actual: What was correct
            confidence: Confidence of the wrong prediction
        """
        if predicted == actual:
            return  # Not an error
        
        error = ErrorRecord(
            ctx_hash=hash(tuple(context)),
            context=tuple(context),
            predicted=predicted,
            actual=actual,
            confidence=confidence,
            timestamp=self.step,
        )
        
        self.errors.append(error)
        self.total_errors += 1
        self.step += 1
    
    def should_reconsolidate(self) -> bool:
        """Check if we have enough errors to process."""
        return len(self.errors) >= self.config.batch_size
    
    def reconsolidate(self, force: bool = False) -> Dict[str, Any]:
        """
        Apply reconsolidation to correct errors.
        
        Args:
            force: Process even if below batch_size
        
        Returns:
            Statistics about the reconsolidation
        """
        if not force and not self.should_reconsolidate():
            return {'processed': 0, 'skipped': 'below batch size'}
        
        if not self.errors:
            return {'processed': 0, 'skipped': 'no errors'}
        
        # Process batch
        n_processed = 0
        n_boosted = 0
        n_attenuated = 0
        
        # Take up to batch_size errors
        batch = []
        while self.errors and len(batch) < self.config.batch_size:
            batch.append(self.errors.popleft())
        
        for error in batch:
            if error.error_magnitude < self.config.min_error_threshold:
                continue
            
            result = self._reconsolidate_single(error)
            if result['boosted']:
                n_boosted += 1
            if result['attenuated']:
                n_attenuated += 1
            n_processed += 1
        
        self.total_reconsolidations += n_processed
        
        return {
            'processed': n_processed,
            'boosted': n_boosted,
            'attenuated': n_attenuated,
            'remaining_errors': len(self.errors),
            'total_reconsolidations': self.total_reconsolidations,
        }
    
    def _reconsolidate_single(self, error: ErrorRecord) -> Dict[str, bool]:
        """
        Reconsolidate memory for a single error.
        
        THEORY:
            When memory[ctx] predicts A but correct answer is B:
            1. BOOST: Add bind(ctx, B) with rate φ⁻² × error_magnitude
            2. ATTENUATE: Subtract bind(ctx, A) with rate φ⁻³ × error_magnitude
            
            The asymmetry (boost > attenuate) ensures we don't catastrophically
            forget — we mostly reinforce correct, slightly weaken wrong.
        """
        ctx_hash = error.ctx_hash
        
        # Check if this context exists in memory
        if ctx_hash not in self.memory.memory:
            # Context not in memory — can't reconsolidate
            # (This shouldn't happen if error came from retrieval)
            return {'boosted': False, 'attenuated': False}
        
        # Get context matrix
        ctx_mat = self.memory.embed_sequence(list(error.context))
        
        # Compute correction bindings
        correct_binding = geometric_product(
            ctx_mat, 
            self.memory.embed(error.actual)
        )
        wrong_binding = geometric_product(
            ctx_mat,
            self.memory.embed(error.predicted)
        )
        
        # Compute rates (scaled by error magnitude)
        boost_rate = self.config.boost_rate * error.error_magnitude
        attenuate_rate = self.config.attenuate_rate * error.error_magnitude
        
        # Apply to memory
        self.memory.memory[ctx_hash] += boost_rate * correct_binding
        self.memory.memory[ctx_hash] -= attenuate_rate * wrong_binding
        
        # Also update frequency counts
        self.memory.context_target_counts[ctx_hash][error.actual] += 1
        # Don't decrement wrong count (could go negative)
        
        return {'boosted': True, 'attenuated': True}
    
    def reconsolidate_all(self) -> Dict[str, Any]:
        """Process all pending errors."""
        total_stats = {
            'processed': 0,
            'boosted': 0,
            'attenuated': 0,
        }
        
        while self.errors:
            stats = self.reconsolidate(force=True)
            total_stats['processed'] += stats.get('processed', 0)
            total_stats['boosted'] += stats.get('boosted', 0)
            total_stats['attenuated'] += stats.get('attenuated', 0)
        
        return total_stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            'total_errors': self.total_errors,
            'total_reconsolidations': self.total_reconsolidations,
            'pending_errors': len(self.errors),
            'reconsolidation_ratio': (
                self.total_reconsolidations / max(1, self.total_errors)
            ),
            'current_step': self.step,
        }


# =============================================================================
# INTEGRATED TRAINING STEP
# =============================================================================

def create_credit_assigned_learn(
    memory: 'FractalGenerativeMemory',
    config: ReconsolidationConfig = None,
) -> Tuple['CreditAssignmentTracker', callable]:
    """
    Create a learn function with integrated credit assignment.
    
    Returns:
        (tracker, learn_with_credit_assignment)
        
    Usage:
        tracker, learn_ca = create_credit_assigned_learn(memory)
        
        for context, target in data:
            stats = learn_ca(context, target)
            # stats contains: predicted, correct, reconsolidated, etc.
    """
    tracker = CreditAssignmentTracker(memory, config)
    
    def learn_with_credit_assignment(
        context: List[int],
        target: int,
        auto_reconsolidate: bool = True,
    ) -> Dict[str, Any]:
        """
        Learn with automatic credit assignment.
        
        1. Predict current (before learning)
        2. If wrong, record error
        3. Learn the association
        4. Optionally reconsolidate
        
        Returns:
            Statistics including prediction, correctness, reconsolidation
        """
        # Predict before learning
        predicted, confidence = memory.retrieve_deterministic(context)
        was_correct = (predicted == target)
        
        # Record error if wrong
        if not was_correct and predicted is not None:
            tracker.record_error(
                context=context,
                predicted=predicted,
                actual=target,
                confidence=confidence,
            )
        
        # Learn normally
        memory.learn(context, target)
        
        # Reconsolidate if batch is ready
        recon_stats = {}
        if auto_reconsolidate and tracker.should_reconsolidate():
            recon_stats = tracker.reconsolidate()
        
        return {
            'predicted': predicted,
            'actual': target,
            'correct': was_correct,
            'confidence': confidence,
            'reconsolidated': recon_stats.get('processed', 0),
            'tracker_stats': tracker.get_statistics(),
        }
    
    return tracker, learn_with_credit_assignment


# =============================================================================
# UTILITY: Batch Credit Assignment
# =============================================================================

def batch_credit_assignment(
    memory: 'FractalGenerativeMemory',
    errors: List[Tuple[List[int], int, int, float]],  # (context, predicted, actual, confidence)
    config: ReconsolidationConfig = None,
) -> Dict[str, Any]:
    """
    Apply credit assignment to a batch of errors at once.
    
    More efficient than one-by-one when processing accumulated errors.
    
    Args:
        memory: The memory to correct
        errors: List of (context, predicted, actual, confidence) tuples
        config: Configuration
    
    Returns:
        Statistics
    """
    config = config or ReconsolidationConfig()
    
    n_processed = 0
    total_boost = 0.0
    total_attenuate = 0.0
    
    for context, predicted, actual, confidence in errors:
        if predicted == actual:
            continue
        
        ctx_hash = hash(tuple(context))
        if ctx_hash not in memory.memory:
            continue
        
        # Compute error magnitude
        error_mag = PHI_INV * (PHI_INV + confidence * (1 - PHI_INV))
        
        if error_mag < config.min_error_threshold:
            continue
        
        # Get context matrix
        ctx_mat = memory.embed_sequence(context)
        
        # Compute and apply corrections
        correct_binding = geometric_product(ctx_mat, memory.embed(actual))
        wrong_binding = geometric_product(ctx_mat, memory.embed(predicted))
        
        boost_rate = config.boost_rate * error_mag
        attenuate_rate = config.attenuate_rate * error_mag
        
        memory.memory[ctx_hash] += boost_rate * correct_binding
        memory.memory[ctx_hash] -= attenuate_rate * wrong_binding
        
        total_boost += boost_rate
        total_attenuate += attenuate_rate
        n_processed += 1
    
    return {
        'processed': n_processed,
        'total_boost': total_boost,
        'total_attenuate': total_attenuate,
        'avg_boost': total_boost / max(1, n_processed),
        'avg_attenuate': total_attenuate / max(1, n_processed),
    }
