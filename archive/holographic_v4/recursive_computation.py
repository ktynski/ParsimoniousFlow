"""
Recursive Computation Module — Iterative Retrieval and Geometric Search
========================================================================

This module adds computational depth to retrieval through:
1. ITERATIVE RETRIEVAL: Refine query through multiple Grace flow cycles
2. GEOMETRIC SEARCH: Beam search through attractor basins
3. RECURSIVE DECOMPOSITION: Split complex queries into simpler sub-queries

THEORY:
    One-shot retrieval is limited because:
    - Noisy queries may not land in the right basin
    - Multiple valid answers may exist
    - Complex queries encode multiple concepts
    
    The solution is iterative refinement:
    - Let Grace flow converge to equilibrium
    - If confidence is low, explore nearby basins
    - "Think longer on harder problems"

KEY INSIGHT:
    Grace stability serves as a CONFIDENCE MEASURE:
    - High stability = query is in a well-defined basin
    - Low stability = query is ambiguous, needs refinement
    
    This allows the system to adaptively allocate computation.

NO RECURSION IN THE PROGRAMMING SENSE:
    "Recursive" here means iterative refinement, not function recursion.
    The computation is bounded by max_iterations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import heapq

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ
from holographic_v4.algebra import (
    build_clifford_basis,
    frobenius_similarity,
    frobenius_similarity_batch,
    grace_operator,
    grace_flow,
    decompose_to_coefficients,
    reconstruct_from_coefficients,
)
from holographic_v4.quotient import (
    extract_witness,
    grace_stability,
    adaptive_similarity_batch,
)

Array = np.ndarray
ArrayModule = type(np)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IterationTrace:
    """Record of one iteration in iterative retrieval."""
    iteration: int
    query: Array
    best_attractor: Array
    best_target: int
    confidence: float
    stability: float


# =============================================================================
# ITERATIVE RETRIEVAL
# =============================================================================

def iterative_retrieval(
    query: Array,
    model: 'TheoryTrueModel',
    dreaming: Optional['DreamingSystem'] = None,
    max_iterations: int = 5,
    convergence_threshold: float = PHI_INV_SQ,
) -> Tuple[Array, int, List[IterationTrace]]:
    """
    Iteratively refine retrieval until convergence.
    
    THEORY:
        Each iteration:
        1. Find best matching attractor
        2. Evaluate confidence (Grace stability)
        3. If low confidence, blend query with attractor
        4. Repeat until stable or max iterations
        
    The blending rate is φ⁻² — slow refinement toward equilibrium.
    
    Args:
        query: Initial query matrix
        model: The model to retrieve from
        dreaming: Optional DreamingSystem for semantic memory
        max_iterations: Maximum refinement iterations
        convergence_threshold: Stop when stability change < threshold
        
    Returns:
        (final_attractor, predicted_target, iteration_traces)
    """
    xp = model.xp
    basis = model.basis
    
    traces = []
    current_query = query.copy()
    prev_stability = 0.0
    
    for i in range(max_iterations):
        # Find best matching attractor
        best_attractor, best_target, confidence = _find_best_match(
            current_query, model, dreaming
        )
        
        # Compute stability
        stability = grace_stability(current_query, basis, xp)
        
        # Record trace
        traces.append(IterationTrace(
            iteration=i,
            query=current_query.copy(),
            best_attractor=best_attractor.copy(),
            best_target=best_target,
            confidence=confidence,
            stability=stability,
        ))
        
        # Check convergence
        stability_change = abs(stability - prev_stability)
        if stability_change < convergence_threshold and i > 0:
            break
        
        prev_stability = stability
        
        # Refine query toward attractor
        # φ⁻¹ ≈ 0.618 is the threshold for "confident enough"
        if confidence < PHI_INV:
            current_query = (1 - PHI_INV_SQ) * current_query + PHI_INV_SQ * best_attractor
            current_query = grace_operator(current_query, basis, xp)
    
    # Final result
    final_attractor = traces[-1].best_attractor
    final_target = traces[-1].best_target
    
    return final_attractor, final_target, traces


def _find_best_match(
    query: Array,
    model: 'TheoryTrueModel',
    dreaming: Optional['DreamingSystem'] = None,
) -> Tuple[Array, int, float]:
    """Find the best matching attractor for a query."""
    xp = model.xp
    basis = model.basis
    
    if model.num_attractors == 0:
        return xp.eye(4), 0, 0.0
    
    # Compute similarities to all attractors
    attractors = model.attractor_matrices[:model.num_attractors]
    
    if model.use_adaptive_similarity:
        sims = adaptive_similarity_batch(query, attractors, basis, xp)
    else:
        sims = frobenius_similarity_batch(query, attractors, xp)
    
    # Find best match (nearest attractor - Grace dynamics already did competition)
    best_idx = int(xp.argmax(sims))
    best_attractor = attractors[best_idx].copy()
    best_target = int(model.attractor_targets[best_idx])
    confidence = float(sims[best_idx])
    
    # Normalize confidence to [0, 1]
    confidence = max(0.0, min(1.0, confidence))
    
    return best_attractor, best_target, confidence


# =============================================================================
# GEOMETRIC SEARCH
# =============================================================================

def geometric_search(
    query: Array,
    model: 'TheoryTrueModel',
    dreaming: Optional['DreamingSystem'] = None,
    beam_width: int = 3,
    max_depth: int = 5,
) -> List[Tuple[Array, int, float]]:
    """
    Beam search through attractor basins.
    
    THEORY:
        Instead of converging to one basin, explore multiple:
        1. Start with query
        2. Find top-k matches
        3. For each match, refine and find more matches
        4. Keep top-k overall at each step
        
    This finds multiple valid answers and ranks them.
    
    Args:
        query: Query matrix
        model: The model
        dreaming: Optional DreamingSystem
        beam_width: Number of candidates to keep at each step
        max_depth: Maximum search depth
        
    Returns:
        List of (attractor, target, score) sorted by score descending
    """
    xp = model.xp
    basis = model.basis
    
    if model.num_attractors == 0:
        return [(xp.eye(4), 0, 0.0)]
    
    # Use a counter to make heap items unique (avoid numpy array comparison)
    counter = [0]
    
    def make_beam_item(score, query_mat, depth):
        counter[0] += 1
        return (score, counter[0], query_mat, depth)
    
    # Initialize beam with query
    # Each beam item is (negative_score, counter, query, depth)
    beam = [make_beam_item(-1.0, query.copy(), 0)]
    
    # Track best results
    results = []  # (attractor, target, score)
    seen_targets = set()
    
    while beam and len(results) < beam_width * 2:
        # Pop best candidate
        neg_score, _, current_query, depth = heapq.heappop(beam)
        
        if depth >= max_depth:
            continue
        
        # Find matches for this query
        matches = _find_top_k_matches(current_query, model, beam_width)
        
        for attractor, target, score in matches:
            if target not in seen_targets:
                results.append((attractor, target, score))
                seen_targets.add(target)
            
            # Add refined query to beam for further exploration
            if depth + 1 < max_depth:
                refined = (1 - PHI_INV) * current_query + PHI_INV * attractor
                refined = grace_operator(refined, basis, xp)
                heapq.heappush(beam, make_beam_item(-score * PHI_INV, refined, depth + 1))  # φ-derived depth discount (was 0.9)
        
        # Keep beam bounded
        while len(beam) > beam_width * 2:
            heapq.heappop(beam)
    
    # Sort results by score (descending)
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results[:beam_width]


def _find_top_k_matches(
    query: Array,
    model: 'TheoryTrueModel',
    k: int,
) -> List[Tuple[Array, int, float]]:
    """Find top-k matching attractors."""
    xp = model.xp
    basis = model.basis
    
    if model.num_attractors == 0:
        return [(xp.eye(4), 0, 0.0)]
    
    attractors = model.attractor_matrices[:model.num_attractors]
    
    if model.use_adaptive_similarity:
        sims = adaptive_similarity_batch(query, attractors, basis, xp)
    else:
        sims = frobenius_similarity_batch(query, attractors, xp)
    
    # Get top-k indices
    top_indices = xp.argsort(sims)[-k:][::-1]
    
    results = []
    for idx in top_indices:
        idx = int(idx)
        attractor = attractors[idx].copy()
        target = int(model.attractor_targets[idx])
        score = float(sims[idx])
        results.append((attractor, target, score))
    
    return results


# =============================================================================
# RECURSIVE DECOMPOSITION
# =============================================================================

def recursive_decomposition(
    complex_query: Array,
    model: 'TheoryTrueModel',
    decomposition_threshold: float = PHI_INV,
) -> List[Array]:
    """
    Decompose complex query into simpler sub-queries.
    
    THEORY:
        If a query has low Grace stability, it may encode multiple
        concepts that should be processed separately.
        
        Decomposition uses grade-wise splitting:
        - Scalar + pseudoscalar = "core" query
        - Vectors = directional components
        - Bivectors = relational components
        
    This is like factoring a complex thought into simpler ones.
    
    Args:
        complex_query: Query to potentially decompose
        model: The model (for basis)
        decomposition_threshold: Stability below this triggers decomposition
        
    Returns:
        List of sub-queries (may be just [complex_query] if stable)
    """
    xp = model.xp
    basis = model.basis
    
    stability = grace_stability(complex_query, basis, xp)
    
    # If stable enough, no decomposition needed
    if stability >= decomposition_threshold:
        return [complex_query]
    
    # Decompose by grade
    coeffs = decompose_to_coefficients(complex_query, basis, xp)
    
    parts = []
    
    # Grade 0+4: Core (scalar + pseudoscalar)
    core_coeffs = xp.zeros(16)
    core_coeffs[0] = coeffs[0]  # Scalar
    core_coeffs[15] = coeffs[15]  # Pseudoscalar
    core = reconstruct_from_coefficients(core_coeffs, basis, xp)
    if xp.linalg.norm(core) > 1e-10:
        parts.append(grace_operator(core, basis, xp))
    
    # Grade 1: Vectors
    vector_coeffs = xp.zeros(16)
    vector_coeffs[1:5] = coeffs[1:5]
    vectors = reconstruct_from_coefficients(vector_coeffs, basis, xp)
    if xp.linalg.norm(vectors) > 1e-10:
        parts.append(grace_operator(vectors, basis, xp))
    
    # Grade 2: Bivectors
    bivector_coeffs = xp.zeros(16)
    bivector_coeffs[5:11] = coeffs[5:11]
    bivectors = reconstruct_from_coefficients(bivector_coeffs, basis, xp)
    if xp.linalg.norm(bivectors) > 1e-10:
        parts.append(grace_operator(bivectors, basis, xp))
    
    # Grade 3: Trivectors
    trivector_coeffs = xp.zeros(16)
    trivector_coeffs[11:15] = coeffs[11:15]
    trivectors = reconstruct_from_coefficients(trivector_coeffs, basis, xp)
    if xp.linalg.norm(trivectors) > 1e-10:
        parts.append(grace_operator(trivectors, basis, xp))
    
    # If decomposition produced no parts, return original
    if not parts:
        return [complex_query]
    
    return parts


# =============================================================================
# COMBINED OPERATIONS
# =============================================================================

def deep_retrieval(
    query: Array,
    model: 'TheoryTrueModel',
    dreaming: Optional['DreamingSystem'] = None,
    max_iterations: int = 5,
    beam_width: int = 3,
    use_decomposition: bool = True,
) -> List[Tuple[Array, int, float]]:
    """
    Full deep retrieval combining all techniques.
    
    Args:
        query: Query matrix
        model: The model
        dreaming: Optional DreamingSystem
        max_iterations: For iterative refinement
        beam_width: For geometric search
        use_decomposition: Whether to decompose complex queries
        
    Returns:
        List of (attractor, target, score) candidates
    """
    xp = model.xp
    basis = model.basis
    
    all_candidates = []
    
    # Optionally decompose query
    if use_decomposition:
        sub_queries = recursive_decomposition(query, model)
    else:
        sub_queries = [query]
    
    for sub_query in sub_queries:
        # Iterative refinement
        refined_query, iter_target, traces = iterative_retrieval(
            query=sub_query,
            model=model,
            dreaming=dreaming,
            max_iterations=max_iterations,
        )
        
        # Geometric search from refined query
        candidates = geometric_search(
            query=refined_query,
            model=model,
            dreaming=dreaming,
            beam_width=beam_width,
            max_depth=max_iterations,
        )
        
        all_candidates.extend(candidates)
    
    # Deduplicate by target, keeping highest score
    target_best = {}
    for attractor, target, score in all_candidates:
        if target not in target_best or score > target_best[target][2]:
            target_best[target] = (attractor, target, score)
    
    results = list(target_best.values())
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results[:beam_width]


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def measure_query_complexity(
    query: Array,
    model: 'TheoryTrueModel',
) -> Dict[str, float]:
    """
    Measure how "complex" a query is.
    
    Complex queries need more computation.
    
    Returns:
        Complexity metrics
    """
    xp = model.xp
    basis = model.basis
    
    stability = grace_stability(query, basis, xp)
    
    # Grade energy distribution
    coeffs = decompose_to_coefficients(query, basis, xp)
    
    grade_energies = {
        0: float(coeffs[0]**2),
        1: float(xp.sum(coeffs[1:5]**2)),
        2: float(xp.sum(coeffs[5:11]**2)),
        3: float(xp.sum(coeffs[11:15]**2)),
        4: float(coeffs[15]**2),
    }
    
    total_energy = sum(grade_energies.values())
    if total_energy > 0:
        grade_ratios = {k: v/total_energy for k, v in grade_energies.items()}
    else:
        grade_ratios = {k: 0.0 for k in grade_energies}
    
    # Complexity = 1 - stability (more complex = less stable)
    complexity = 1.0 - stability
    
    return {
        'complexity': complexity,
        'stability': stability,
        'total_energy': total_energy,
        **{f'grade_{k}_ratio': v for k, v in grade_ratios.items()},
    }


def estimate_iterations_needed(
    query: Array,
    model: 'TheoryTrueModel',
) -> int:
    """
    Estimate how many iterations a query needs.
    
    Based on complexity: more complex = more iterations.
    
    Returns:
        Estimated iterations (1 to 10)
    """
    metrics = measure_query_complexity(query, model)
    complexity = metrics['complexity']
    
    # Map complexity [0, 1] to iterations [1, 10]
    iterations = int(1 + complexity * 9)
    return max(1, min(10, iterations))
