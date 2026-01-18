"""
Pruning — Synaptic Pruning Analog

Implements theory-true pruning based on salience and support.
Only prunes when BOTH salience AND support are low (conservative strategy).

THEORY (Synaptic Pruning):
    Weak connections (low salience + low support) are removed.
    This prevents unbounded memory growth and reduces retrieval interference.
"""

import numpy as np
from typing import Tuple, Dict, Any

from holographic_prod.core.constants import PHI_INV_CUBE
from .priority import compute_salience, compute_salience_batch
from .structures import SemanticPrototype


def should_prune_prototype(
    proto: SemanticPrototype,
    basis: np.ndarray,
    xp = np,
    salience_threshold: float = PHI_INV_CUBE,  # Theory-true: φ⁻³ (tertiary threshold)
    support_threshold: int = 2,
    min_age: int = 0,  # Minimum number of sleep cycles before pruning
) -> Tuple[bool, str]:
    """
    Determine if a prototype should be pruned.
    
    THEORY (Synaptic Pruning):
        Brain removes weak synaptic connections. In our framework:
        - Low salience = not emotionally important (won't survive Grace)
        - Low support = rarely seen (weak evidence)
        - Both together = candidate for pruning
        
    The φ-connection:
        Salience threshold is φ⁻¹ × base_threshold ≈ 0.382
        This connects pruning to Grace dynamics (what survives vs decays)
        
    Args:
        proto: Prototype to evaluate
        basis: Clifford basis
        xp: array module
        salience_threshold: Below this is "low salience"
        support_threshold: Below this is "low support"
        min_age: Don't prune until this many cycles old
        
    Returns:
        (should_prune, reason)
    """
    # Compute salience of prototype
    salience = compute_salience(proto.prototype_matrix, basis, xp)
    
    reasons = []
    
    # Low salience check
    if salience < salience_threshold:
        reasons.append(f"low_salience({salience:.3f}<{salience_threshold})")
    
    # Low support check
    if proto.support < support_threshold:
        reasons.append(f"low_support({proto.support}<{support_threshold})")
    
    # Prune only if BOTH conditions met (conservative pruning)
    should_prune = salience < salience_threshold and proto.support < support_threshold
    
    reason = " AND ".join(reasons) if reasons else "healthy"
    return should_prune, reason


def prune_semantic_memory(
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
    salience_threshold: float = PHI_INV_CUBE,  # Theory-true: φ⁻³
    support_threshold: int = 2,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Prune low-value prototypes from semantic memory.
    
    THEORY (Synaptic Pruning):
        Weak connections (low salience + low support) are removed.
        This:
        1. Prevents unbounded memory growth
        2. Reduces retrieval interference
        3. Keeps only "strong" memories
        
    CONSERVATIVE STRATEGY:
        Only prune if BOTH salience AND support are low.
        Never prune schemas (they survived REM = highly stable).
        
    Args:
        semantic_memory: SemanticMemory to prune
        basis: Clifford basis
        xp: array module
        salience_threshold: Below this is "low salience"
        support_threshold: Below this is "low support"
        verbose: Print details
        
    Returns:
        Statistics about pruning
    """
    if verbose:
        print("  PRUNING (Synaptic Pruning)")
        print("  " + "-" * 40)
    
    stats = {
        'total_before': semantic_memory.stats()['total_prototypes'],
        'pruned': 0,
        'pruned_by_level': {},
        'reasons': [],
    }
    
    # Prune each level
    for level_idx, level in enumerate(semantic_memory.levels):
        pruned_this_level = 0
        kept = []
        
        for proto in level:
            should_prune, reason = should_prune_prototype(
                proto, basis, xp, salience_threshold, support_threshold
            )
            
            if should_prune:
                pruned_this_level += 1
                stats['reasons'].append(reason)
                if verbose:
                    print(f"    Pruned level-{level_idx} prototype: {reason}")
            else:
                kept.append(proto)
        
        # Update level with kept prototypes
        semantic_memory.levels[level_idx] = kept
        stats['pruned_by_level'][level_idx] = pruned_this_level
        stats['pruned'] += pruned_this_level
    
    stats['total_after'] = semantic_memory.stats()['total_prototypes']
    
    if verbose:
        print(f"    Total pruned: {stats['pruned']}")
        print(f"    Before: {stats['total_before']} → After: {stats['total_after']}")
    
    return stats


def prune_attractor_map(
    attractor_matrices: np.ndarray,
    attractor_targets: np.ndarray,
    num_attractors: int,
    basis: np.ndarray,
    xp = np,
    salience_threshold: float = PHI_INV_CUBE,  # Theory-true: φ⁻³
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    """
    Prune low-salience attractors from bookkeeping arrays.
    
    v4.13.0: SIMPLIFIED - Prune based on salience only (retention = salience)
    
    NOTE: This only prunes bookkeeping arrays. Holographic memory
    doesn't support selective removal (patterns are superposed).
    
    THEORY:
        Prune attractors with salience below threshold.
        Salience = witness stability = what Grace preserves.
        
    REMOVED (v4.13.0 - not theory-true):
        - access_counts parameter
        - access_threshold parameter
        Access counting doesn't work for holographic superposition.
    
    Args:
        attractor_matrices: [N, 4, 4] attractor storage
        attractor_targets: [N] target indices
        num_attractors: Current count
        basis: Clifford basis
        xp: array module
        salience_threshold: Below this is pruned
        verbose: Print details
        
    Returns:
        (new_matrices, new_targets, new_count, stats)
    """
    if verbose:
        print("  PRUNING ATTRACTORS (salience-based)")
        print("  " + "-" * 40)
    
    if num_attractors == 0:
        return attractor_matrices, attractor_targets, 0, {'pruned': 0}
    
    # Compute saliences for all attractors (VECTORIZED)
    matrices = attractor_matrices[:num_attractors]
    saliences = compute_salience_batch(matrices, basis, xp)
    
    # Determine which to keep (salience >= threshold)
    keep_mask = saliences >= salience_threshold
    keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
    pruned = num_attractors - len(keep_indices)
    
    # Compact the arrays
    new_count = len(keep_indices)
    new_matrices = xp.zeros_like(attractor_matrices)
    new_targets = xp.zeros_like(attractor_targets)
    
    for new_idx, old_idx in enumerate(keep_indices):
        new_matrices[new_idx] = attractor_matrices[old_idx]
        new_targets[new_idx] = attractor_targets[old_idx]
    
    stats = {
        'total_before': num_attractors,
        'pruned': pruned,
        'total_after': new_count,
        'keep_indices': keep_indices,  # For compacting tracking arrays
    }
    
    if verbose:
        print(f"    Total pruned: {pruned}")
        print(f"    Before: {num_attractors} → After: {new_count}")
    
    return new_matrices, new_targets, new_count, stats
