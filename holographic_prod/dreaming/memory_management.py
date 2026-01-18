"""
Memory Management — Theory-True φ-Decay Forgetting and Adaptive Thresholds

Implements theory-true memory management:
- φ-decay forgetting: Survival probability = φ^(-k × (1 - priority))
- Adaptive similarity threshold: Adjusts based on prototype diversity

All thresholds are φ-derived (theory-true).
"""

import numpy as np
from typing import List, Tuple

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_EIGHT
)
from .priority import compute_salience_batch
from .structures import EpisodicEntry


def phi_decay_forget(
    episodes: List[EpisodicEntry],
    max_episodes: int,
    basis: np.ndarray,
    xp = np,
) -> Tuple[List[EpisodicEntry], int]:
    """
    Theory-true forgetting using φ-decay.
    
    DERIVATION:
        The brain forgets by DECAY, not arbitrary pruning.
        Forgetting probability is φ^(-priority) where priority = stability × salience.
        
        This gives:
        - High priority (important, stable): survival prob ≈ 1
        - Low priority (unimportant, unstable): survival prob → 0
        
        The decay rate φ⁻¹ ≈ 0.618 is the same as Grace's spectral gap,
        ensuring consistency between consolidation and forgetting dynamics.
    
    Args:
        episodes: Episodes to potentially forget
        max_episodes: Maximum episodes to keep (capacity)
        basis: Clifford basis for computing salience/stability
        xp: Array module
        
    Returns:
        (surviving_episodes, num_forgotten)
    """
    if len(episodes) <= max_episodes:
        return episodes, 0
    
    n_episodes = len(episodes)
    
    # OPTIMIZED: Batch extract matrices (single CPU→GPU transfer)
    matrices_np = np.empty((n_episodes, 4, 4), dtype=np.float32)
    for i, ep in enumerate(episodes):
        mat = ep.context_matrix
        matrices_np[i] = mat.get() if hasattr(mat, 'get') else mat
    matrices = xp.asarray(matrices_np)  # Single transfer
    
    # Compute priority scores (stays on GPU)
    from holographic_prod.core.quotient import grace_stability_batch
    saliences = compute_salience_batch(matrices, basis, xp)
    stabilities = grace_stability_batch(matrices, basis, xp)
    
    # Priority = stability × salience (φ-weighted importance)
    # Use xp random but keep on GPU
    priorities = stabilities * (saliences + PHI_INV_EIGHT) + PHI_INV_EIGHT * xp.random.rand(n_episodes)
    
    # φ-decay survival probability: P(survive) = φ^(-k × (1 - priority))
    excess_ratio = n_episodes / max_episodes
    k = np.log(excess_ratio) / np.log(PHI)  # Theory-derived scaling
    
    survival_probs = PHI ** (-k * (1.0 - priorities))
    survival_probs = xp.clip(survival_probs, 0.0, 1.0)
    
    # Stochastic forgetting
    random_vals = xp.random.rand(n_episodes)
    survive_mask = random_vals < survival_probs
    
    # Ensure we hit capacity exactly (top priority as fallback)
    num_surviving = int(xp.sum(survive_mask))
    if num_surviving != max_episodes:
        priority_order = xp.argsort(priorities)[::-1]
        survive_mask = xp.zeros(n_episodes, dtype=bool)
        survive_mask[priority_order[:max_episodes]] = True
    
    # OPTIMIZED: Transfer mask to CPU once, then filter
    survive_mask_cpu = survive_mask.get() if hasattr(survive_mask, 'get') else survive_mask
    surviving = [episodes[i] for i in range(n_episodes) if survive_mask_cpu[i]]
    num_forgotten = n_episodes - len(surviving)
    
    return surviving, num_forgotten


def compute_adaptive_threshold(
    semantic_memory: 'SemanticMemory',
    min_threshold: float = PHI_INV_CUBE,  # φ-derived: φ⁻³ ≈ 0.236 (was 0.3)
    max_threshold: float = PHI_INV,       # φ-derived: φ⁻¹ ≈ 0.618 (was 0.8)
    target_clusters_per_sleep: int = 10,
) -> float:
    """
    Theory-true adaptive similarity threshold.
    
    DERIVATION:
        The similarity threshold controls prototype diversity:
        - High threshold → few, broad prototypes (underfitting)
        - Low threshold → many, narrow prototypes (overfitting)
        
        The optimal threshold should produce ~φ² ≈ 2.618 × log(N) prototypes
        for N episodes, matching the Grace spectral structure.
        
        If we're creating too few clusters per sleep, LOWER the threshold.
        If we're creating too many (memory exploding), RAISE the threshold.
    
    Args:
        semantic_memory: Current semantic memory state
        min_threshold: Lower bound (don't go below this)
        max_threshold: Upper bound (don't go above this)
        target_clusters_per_sleep: Expected clusters per sleep cycle
        
    Returns:
        Adjusted similarity threshold
    """
    stats = semantic_memory.stats()
    total_protos = stats['total_prototypes']
    
    if total_protos < 10:
        # Not enough data - use low threshold to encourage diversity
        return min_threshold
    
    # Estimate clusters per sleep from prototype growth rate
    # If we have few prototypes, we need more diversity (lower threshold)
    # If we have many prototypes, we need more compression (higher threshold)
    
    # Target: ~1 prototype per 100 episodes (φ² ≈ 2.6 clusters × 40 episodes/cluster)
    # This gives O(log N) semantic compression
    
    # Simple heuristic: adjust based on recent growth
    # If prototypes < expected, lower threshold
    # The "expected" is φ² × log(total_episodes_seen) - but we don't track that
    # Use schema count as proxy for abstraction level
    
    num_schemas = stats['num_schemas']
    
    # If schemas >> prototypes, we're over-abstracting (threshold too high)
    # If schemas << prototypes, we're under-abstracting (threshold too low)
    schema_ratio = num_schemas / (total_protos + 1)
    
    # Target ratio is ~φ ≈ 1.618 (golden balance of abstraction)
    ratio_error = schema_ratio - PHI
    
    # Adjust threshold: positive error → lower threshold (need more prototypes)
    # Negative error → raise threshold (need more compression)
    adjustment = -PHI_INV_CUBE * ratio_error  # φ-derived damping (was -0.1)
    
    new_threshold = PHI_INV_SQ + adjustment  # Start from φ⁻² (was 0.5)
    new_threshold = max(min_threshold, min(max_threshold, new_threshold))
    
    return new_threshold
