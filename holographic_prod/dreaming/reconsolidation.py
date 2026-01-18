"""
Reconsolidation — Retrieval-Induced Plasticity

Implements theory-true reconsolidation: when a memory is retrieved,
it becomes labile (modifiable) and can be updated based on feedback.

THEORY (Reconsolidation):
    Retrieving a memory makes it LABILE (modifiable).
    This allows:
    1. Strengthening: correct predictions → boost confidence
    2. Correction: wrong predictions → update toward actual
    3. Freshening: accessed memories → update recency
    
    In the brain:
    - Memory trace is reactivated
    - Protein synthesis required to re-stabilize
    - Window of plasticity during which memory can change
    
    In our framework:
    - Track recent retrievals
    - When feedback arrives, update the memory
    - φ⁻¹ rate for updates (same as initial learning)
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any

from holographic_prod.core.constants import PHI_INV
from .structures import RetrievalRecord, SemanticPrototype


class ReconsolidationTracker:
    """
    Track retrievals for reconsolidation.
    
    THEORY (Reconsolidation):
        Retrieving a memory makes it LABILE (modifiable).
        This allows:
        1. Strengthening: correct predictions → boost confidence
        2. Correction: wrong predictions → update toward actual
        3. Freshening: accessed memories → update recency
        
    In the brain:
        - Memory trace is reactivated
        - Protein synthesis required to re-stabilize
        - Window of plasticity during which memory can change
        
    In our framework:
        - Track recent retrievals
        - When feedback arrives, update the memory
        - φ⁻¹ rate for updates (same as initial learning)
    """
    
    def __init__(
        self,
        max_pending: int = 1000,
        reconsolidation_rate: float = PHI_INV,
    ):
        """
        Args:
            max_pending: Maximum pending retrievals to track
            reconsolidation_rate: Rate for memory updates (φ⁻¹ by theory)
        """
        self.pending_retrievals: Dict[int, RetrievalRecord] = {}
        self.max_pending = max_pending
        self.reconsolidation_rate = reconsolidation_rate
        
        # Statistics
        self.total_retrievals = 0
        self.total_feedback = 0
        self.correct_predictions = 0
        self.corrections_made = 0
    
    def record_retrieval(
        self,
        context_hash: int,
        predicted_target: int,
        source: str = "unknown",
    ) -> RetrievalRecord:
        """
        Record a retrieval event.
        
        Call this when retrieve() returns a prediction.
        """
        record = RetrievalRecord(
            context_hash=context_hash,
            predicted_target=predicted_target,
            timestamp=time.time(),
            source=source,
        )
        
        self.pending_retrievals[context_hash] = record
        self.total_retrievals += 1
        
        # Cleanup old records if needed
        if len(self.pending_retrievals) > self.max_pending:
            # Remove oldest (by hash, simple strategy)
            oldest_hash = min(self.pending_retrievals.keys())
            del self.pending_retrievals[oldest_hash]
        
        return record
    
    def provide_feedback(
        self,
        context_hash: int,
        actual_target: int,
    ) -> Optional[RetrievalRecord]:
        """
        Provide feedback for a previous retrieval.
        
        Call this when the actual target is known.
        
        Returns:
            The updated retrieval record, or None if not found
        """
        if context_hash not in self.pending_retrievals:
            return None
        
        record = self.pending_retrievals[context_hash]
        record.actual_target = actual_target
        record.was_correct = (record.predicted_target == actual_target)
        
        self.total_feedback += 1
        if record.was_correct:
            self.correct_predictions += 1
        
        return record
    
    def get_pending_for_reconsolidation(
        self,
        only_incorrect: bool = False,
    ) -> List[RetrievalRecord]:
        """
        Get retrievals that need reconsolidation.
        
        Args:
            only_incorrect: If True, only return records where prediction was wrong
            
        Returns:
            List of records needing reconsolidation
        """
        result = []
        for record in self.pending_retrievals.values():
            if record.actual_target is not None:  # Has feedback
                if not only_incorrect or not record.was_correct:
                    result.append(record)
        return result
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'total_retrievals': self.total_retrievals,
            'total_feedback': self.total_feedback,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.correct_predictions / max(1, self.total_feedback),
            'pending': len(self.pending_retrievals),
        }


def reconsolidate_attractor(
    attractor_matrix: np.ndarray,
    target_embedding: np.ndarray,
    was_correct: bool,
    rate: float = PHI_INV,
    xp = np,
) -> np.ndarray:
    """
    Reconsolidate (update) an attractor based on feedback.
    
    THEORY:
        When a memory is retrieved, it becomes labile.
        Feedback can update it:
        
        CORRECT: Strengthen (small update toward current target)
            attractor' = (1 - rate/2) * attractor + (rate/2) * target
            
        INCORRECT: Correct (larger update toward actual target)
            attractor' = (1 - rate) * attractor + rate * target
        
    The rate is φ⁻¹ by theory (same as initial learning).
    Correct predictions use rate/2 (gentler update).
    
    Args:
        attractor_matrix: Current attractor
        target_embedding: Correct target embedding
        was_correct: Whether the prediction was correct
        rate: Update rate (φ⁻¹ by theory)
        xp: array module
        
    Returns:
        Updated attractor matrix
    """
    if was_correct:
        # Strengthen: gentle update (half rate)
        effective_rate = rate * 0.5
    else:
        # Correct: full update
        effective_rate = rate
    
    # Lerp toward target
    updated = (1 - effective_rate) * attractor_matrix + effective_rate * target_embedding
    
    return updated


def reconsolidate_semantic_prototype(
    proto: SemanticPrototype,
    actual_target: int,
    was_correct: bool,
    rate: float = PHI_INV,
) -> SemanticPrototype:
    """
    Reconsolidate a semantic prototype based on feedback.
    
    THEORY:
        Semantic prototypes store target DISTRIBUTIONS.
        Reconsolidation updates the distribution:
        
        CORRECT: Boost probability of predicted target
        INCORRECT: Reduce predicted, boost actual
        
    Args:
        proto: Prototype to update
        actual_target: The actual target that occurred
        was_correct: Whether prediction matched actual
        rate: Update rate
        
    Returns:
        Updated prototype (new object)
    """
    new_dist = dict(proto.target_distribution)
    
    if was_correct:
        # Strengthen: slightly boost the actual target
        if actual_target in new_dist:
            boost = rate * 0.1
            new_dist[actual_target] = min(1.0, new_dist[actual_target] + boost)
    else:
        # Correct: add/boost actual target
        if actual_target not in new_dist:
            new_dist[actual_target] = rate * 0.2
        else:
            new_dist[actual_target] += rate * 0.2
    
    # Renormalize
    total = sum(new_dist.values())
    new_dist = {k: v / total for k, v in new_dist.items()}
    
    return SemanticPrototype(
        prototype_matrix=proto.prototype_matrix,  # Matrix unchanged
        target_distribution=new_dist,
        radius=proto.radius,
        support=proto.support + 1,  # Increase support (more evidence)
        level=proto.level,
    )
