"""
Pattern Completion — Retrieval as Inference via Grace Flow

Completes partial/noisy patterns by applying Grace flow to converge
to stable attractors. This is "retrieval as inference" rather than lookup.

THEORY (Pattern Completion as Inference):
    The hippocampus completes partial inputs to stored patterns.
    In Clifford/Grace framework:
    
    1. Noisy query has high-grade noise (vectors, bivectors, trivectors)
    2. Grace contracts higher grades toward zero
    3. Iterative application converges to "stable core"
    4. Completed pattern is closer to stored attractors
"""

import numpy as np
from typing import Tuple, Dict, Any

from holographic_prod.core.algebra import grace_operator, grace_operator_batch


def pattern_complete(
    query: np.ndarray,
    basis: np.ndarray,
    xp = np,
    max_steps: int = 5,
    convergence_threshold: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Complete a partial/noisy pattern by applying Grace flow.
    
    THEORY (Pattern Completion as Inference):
        The hippocampus completes partial inputs to stored patterns.
        In Clifford/Grace framework:
        
        1. Noisy query has high-grade noise (vectors, bivectors, trivectors)
        2. Grace contracts higher grades toward zero
        3. Iterative application converges to "stable core"
        4. Completed pattern is closer to stored attractors
        
    This is "retrieval as inference" rather than lookup:
        - Query: partial/noisy input
        - Completion: What Grace flow converges to
        - Result: Cleaner signal for similarity search
        
    The number of steps controls completion depth:
        - Few steps: Light denoising (preserve detail)
        - Many steps: Strong completion (more abstract)
    
    Args:
        query: [4, 4] input matrix (possibly noisy/partial)
        basis: [16, 4, 4] Clifford basis
        xp: numpy or cupy
        max_steps: Maximum Grace iterations
        convergence_threshold: Stop if change < threshold
        
    Returns:
        completed: [4, 4] completed pattern
        info: Dict with completion stats
    """
    current = query.copy()
    initial_norm = float(xp.linalg.norm(current, 'fro'))
    
    steps_taken = 0
    total_change = 0.0
    converged = False
    
    for step in range(max_steps):
        previous = current.copy()
        current = grace_operator(current, basis, xp)
        
        # Measure change
        change = float(xp.linalg.norm(current - previous, 'fro'))
        total_change += change
        steps_taken += 1
        
        # Check convergence
        if change < convergence_threshold:
            converged = True
            break
    
    # Normalize to preserve original scale
    final_norm = float(xp.linalg.norm(current, 'fro'))
    if final_norm > 1e-8:
        current = current * (initial_norm / final_norm)
    
    info = {
        'steps_taken': steps_taken,
        'converged': converged,
        'total_change': total_change,
        'avg_change_per_step': total_change / max(steps_taken, 1),
    }
    
    return current, info


def pattern_complete_batch(
    queries: np.ndarray,
    basis: np.ndarray,
    xp = np,
    max_steps: int = 5,
    convergence_threshold: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    VECTORIZED pattern completion for multiple queries.
    
    Args:
        queries: [N, 4, 4] batch of query matrices
        basis: [16, 4, 4] Clifford basis
        xp: numpy or cupy
        max_steps: Maximum Grace iterations
        convergence_threshold: Stop if max change < threshold
        
    Returns:
        completed: [N, 4, 4] completed patterns
        info: Dict with batch completion stats
    """
    N = queries.shape[0]
    if N == 0:
        return queries, {'steps_taken': 0, 'converged': True}
    
    current = queries.copy()
    initial_norms = xp.linalg.norm(current.reshape(N, -1), axis=1, keepdims=True)
    
    steps_taken = 0
    converged = False
    
    for step in range(max_steps):
        previous = current.copy()
        current = grace_operator_batch(current, basis, xp)
        
        # Measure max change across batch
        changes = xp.linalg.norm((current - previous).reshape(N, -1), axis=1)
        max_change = float(xp.max(changes))
        steps_taken += 1
        
        if max_change < convergence_threshold:
            converged = True
            break
    
    # Restore original scales
    final_norms = xp.linalg.norm(current.reshape(N, -1), axis=1, keepdims=True)
    scale_factors = initial_norms / xp.maximum(final_norms, 1e-8)
    current = current * scale_factors.reshape(-1, 1, 1)
    
    info = {
        'steps_taken': steps_taken,
        'converged': converged,
        'batch_size': N,
    }
    
    return current, info
