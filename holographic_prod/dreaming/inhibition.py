"""
Inhibition of Return — Exploration Bonus for Memory Retrieval

Implements theory-true inhibition of return:
- Recently retrieved items are temporarily suppressed
- Encourages exploration and prevents fixation
- Penalty decays with time (φ⁻¹ per step)

THEORY (Inhibition of Return):
    The brain temporarily suppresses recently attended items.
    In memory retrieval:
    - Recently retrieved prototypes are penalized
    - Encourages diverse memory access
"""

import numpy as np
from typing import List, Tuple, Dict, Any

from holographic_prod.core.constants import (
    PHI_INV, PHI_INV_EIGHT, DTYPE
)
from holographic_prod.core.algebra import frobenius_cosine_batch


# =============================================================================
# RETRIEVAL HISTORY
# =============================================================================

class RetrievalHistory:
    """
    Tracks recently retrieved items for inhibition of return.
    
    THEORY (Inhibition of Return):
        The brain temporarily suppresses recently attended items.
        This encourages exploration and prevents fixation.
        
    In memory retrieval:
        - Recently retrieved prototypes are penalized
        - Penalty decays with time (φ⁻¹ per step)
        - Encourages diverse memory access
    """
    
    def __init__(
        self,
        decay_rate: float = PHI_INV,
        max_history: int = 100,
        xp = np,
    ):
        """
        Args:
            decay_rate: How fast inhibition decays (default: φ⁻¹ ≈ 0.618)
            max_history: Maximum number of retrievals to track
            xp: numpy or cupy
        """
        self.decay_rate = decay_rate
        self.max_history = max_history
        self.xp = xp
        
        # Map from prototype id to (timestamp, inhibition_strength)
        self.history: Dict[int, Tuple[float, float]] = {}
        self.current_time: float = 0.0
    
    def record_retrieval(self, proto_id: int, strength: float = 1.0):
        """
        Record that a prototype was retrieved.
        
        Args:
            proto_id: ID of the retrieved prototype
            strength: Initial inhibition strength (1.0 = full suppression)
        """
        self.history[proto_id] = (self.current_time, strength)
        
        # Prune old entries if too many
        if len(self.history) > self.max_history:
            # Remove oldest entries
            sorted_items = sorted(self.history.items(), key=lambda x: x[1][0])
            to_remove = len(self.history) - self.max_history
            for proto_id, _ in sorted_items[:to_remove]:
                del self.history[proto_id]
    
    def advance_time(self, steps: float = 1.0):
        """Advance time (allows inhibition to decay)."""
        self.current_time += steps
    
    def get_inhibition(self, proto_id: int) -> float:
        """
        Get current inhibition level for a prototype.
        
        Returns:
            inhibition: float in [0, 1] - 1.0 = fully suppressed, 0.0 = no suppression
        """
        if proto_id not in self.history:
            return 0.0
        
        timestamp, initial_strength = self.history[proto_id]
        time_elapsed = self.current_time - timestamp
        
        # Inhibition decays as φ⁻ᵗ
        decayed = initial_strength * (self.decay_rate ** time_elapsed)
        
        return decayed
    
    def get_all_inhibitions(self, proto_ids: List[int]) -> np.ndarray:
        """
        Get inhibition levels for multiple prototypes (vectorized).
        
        Args:
            proto_ids: List of prototype IDs
            
        Returns:
            inhibitions: [N] array of inhibition levels
        """
        xp = self.xp
        inhibitions = xp.zeros(len(proto_ids), dtype=DTYPE)
        
        for i, pid in enumerate(proto_ids):
            inhibitions[i] = self.get_inhibition(pid)
        
        return inhibitions
    
    def clear(self):
        """Clear all history."""
        self.history.clear()
        self.current_time = 0.0
    
    def stats(self) -> Dict[str, Any]:
        """Return history statistics."""
        if not self.history:
            return {
                'count': 0,
                'avg_inhibition': 0.0,
                'max_inhibition': 0.0,
            }
        
        inhibitions = [self.get_inhibition(pid) for pid in self.history.keys()]
        
        return {
            'count': len(self.history),
            'avg_inhibition': float(np.mean(inhibitions)),
            'max_inhibition': float(np.max(inhibitions)),
            'current_time': self.current_time,
        }


# =============================================================================
# INHIBITION FUNCTIONS
# =============================================================================

def apply_inhibition_of_return(
    similarities: np.ndarray,
    proto_ids: List[int],
    retrieval_history: RetrievalHistory,
    inhibition_weight: float = PHI_INV,  # Theory-true: φ⁻¹ (moderate inhibition)
    xp = np,
) -> np.ndarray:
    """
    Apply inhibition of return to similarity scores.
    
    THEORY:
        Adjusted_similarity = similarity - inhibition_weight * inhibition
        
        Recently retrieved items get their similarity reduced,
        making less recently retrieved items more competitive.
    
    Args:
        similarities: [N] array of similarity scores
        proto_ids: List of prototype IDs corresponding to similarities
        retrieval_history: RetrievalHistory tracking recent retrievals
        inhibition_weight: How much to penalize (default φ⁻¹ = moderate)
        xp: numpy or cupy
        
    Returns:
        adjusted_similarities: [N] array with inhibition applied
    """
    # Get inhibition levels for all prototypes
    inhibitions = retrieval_history.get_all_inhibitions(proto_ids)
    
    # Apply inhibition: reduce similarity for recently retrieved
    adjusted = similarities - inhibition_weight * inhibitions
    
    return adjusted


def retrieve_with_inhibition(
    query: np.ndarray,
    semantic_memory: 'SemanticMemory',
    retrieval_history: RetrievalHistory,
    top_k: int = 5,
    inhibition_weight: float = PHI_INV,  # φ-derived
    record_retrieval: bool = True,
    xp = np,
) -> Tuple[List[Tuple['SemanticPrototype', float]], Dict[str, Any]]:
    """
    Retrieve from semantic memory with inhibition of return.
    
    THEORY (Exploration Bonus):
        - Recently retrieved items are suppressed
        - Encourages exploring diverse memories
        - Prevents fixation on dominant patterns
        
    Process:
        1. Compute base similarities
        2. Apply inhibition penalty to recently retrieved
        3. Return top-k by adjusted similarity
        4. Record retrieval for future inhibition
    
    Args:
        query: [4, 4] query matrix
        semantic_memory: SemanticMemory to search
        retrieval_history: RetrievalHistory for inhibition
        top_k: Number of results
        inhibition_weight: Strength of inhibition penalty
        record_retrieval: Whether to record this retrieval
        xp: numpy or cupy
        
    Returns:
        results: List of (prototype, adjusted_similarity) pairs
        info: Retrieval statistics
    """
    # Get all prototypes
    all_protos = []
    for level in semantic_memory.levels:
        all_protos.extend(level)
    
    if not all_protos:
        return [], {'inhibited': 0}
    
    # Compute base similarities (cosine normalized to [-1, 1])
    # GPU-NATIVE: Use xp.stack consistently
    proto_matrices = xp.stack([xp.asarray(p.prototype_matrix) for p in all_protos])
    base_sims = frobenius_cosine_batch(xp.asarray(query), proto_matrices, xp)
    
    # Get prototype IDs
    proto_ids = [id(p) for p in all_protos]
    
    # Apply inhibition
    adjusted_sims = apply_inhibition_of_return(
        base_sims, proto_ids, retrieval_history,
        inhibition_weight=inhibition_weight, xp=xp
    )
    
    # Track how many were significantly inhibited (φ⁻⁸ threshold)
    inhibition_applied = base_sims - adjusted_sims
    n_inhibited = int(xp.sum(inhibition_applied > PHI_INV_EIGHT))
    
    # Sort by adjusted similarity
    sorted_indices = xp.argsort(-adjusted_sims)
    
    results = []
    for idx in sorted_indices[:top_k]:
        idx = int(idx)
        proto = all_protos[idx]
        adj_sim = float(adjusted_sims[idx])
        results.append((proto, adj_sim))
        
        # Record retrieval for future inhibition
        if record_retrieval and len(results) == 1:  # Only record top-1
            retrieval_history.record_retrieval(proto_ids[idx])
    
    info = {
        'inhibited': n_inhibited,
        'total_prototypes': len(all_protos),
        'avg_inhibition': float(xp.mean(inhibition_applied)),
    }
    
    return results, info
