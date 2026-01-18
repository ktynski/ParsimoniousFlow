"""
Interference Management — Merge Similar Prototypes

Manages interference by merging highly similar prototypes using theory-true
combined witness+vorticity similarity (not just Frobenius).

THEORY (Interference Management):
    - Similar memories compete during retrieval
    - High similarity → interference
    - But: aggressive merging destroys coverage → poor generalization!
    
    Conservative threshold (1 - φ⁻³ ≈ 0.764) preserves diversity.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict

from holographic_prod.core.constants import PHI_INV
from holographic_prod.core.algebra import (
    grace_operator,
    build_clifford_basis,
)
from holographic_prod.core.quotient import extract_witness, vorticity_similarity
from .structures import SemanticPrototype


def compute_prototype_similarity(
    proto1: SemanticPrototype,
    proto2: SemanticPrototype,
    xp = np,
    basis = None,
) -> float:
    """
    Compute THEORY-TRUE similarity between two prototypes.
    
    Uses vorticity_similarity (witness + vorticity, cosine normalized).
    Returns similarity in [-1, 1].
    """
    if basis is None:
        basis = build_clifford_basis(xp)
    return vorticity_similarity(
        proto1.prototype_matrix, proto2.prototype_matrix, basis, xp
    )


def merge_prototypes(
    proto1: SemanticPrototype,
    proto2: SemanticPrototype,
    basis: np.ndarray,
    xp = np,
) -> SemanticPrototype:
    """
    Merge two prototypes into one.
    
    THEORY:
        Merging reduces interference by combining similar memories.
        The merged prototype is STRONGER (higher support) and CLEARER
        (combined target distribution has more evidence).
        
    MERGE STRATEGY:
        - Matrix: Weighted average by support
        - Target distribution: Combined and normalized
        - Radius: Max of both (covers both regions)
        - Support: Sum of both (total evidence)
    
    Args:
        proto1, proto2: Prototypes to merge
        basis: Clifford basis
        xp: array module
        
    Returns:
        Merged prototype
    """
    total_support = proto1.support + proto2.support
    
    # Weighted average of matrices (by support)
    w1 = proto1.support / total_support
    w2 = proto2.support / total_support
    
    merged_matrix = w1 * proto1.prototype_matrix + w2 * proto2.prototype_matrix
    
    # THEORY-TRUE: Use Grace to stabilize, not arbitrary Frobenius normalization
    # Grace contracts higher grades, preserving the stable core
    merged_matrix = grace_operator(merged_matrix, basis, xp)
    
    # Combine target distributions
    merged_targets = defaultdict(float)
    
    for token, prob in proto1.target_distribution.items():
        merged_targets[token] += prob * w1
    
    for token, prob in proto2.target_distribution.items():
        merged_targets[token] += prob * w2
    
    # Normalize distribution
    total_prob = sum(merged_targets.values())
    merged_targets = {k: v / total_prob for k, v in merged_targets.items()}
    
    # Take max radius
    merged_radius = max(proto1.radius, proto2.radius)
    
    # Take lower level (more specific)
    merged_level = min(proto1.level, proto2.level)
    
    # Merge vorticity signatures (weighted average by support)
    merged_vort = None
    if proto1.vorticity_signature is not None and proto2.vorticity_signature is not None:
        merged_vort = w1 * proto1.vorticity_signature + w2 * proto2.vorticity_signature
    elif proto1.vorticity_signature is not None:
        merged_vort = proto1.vorticity_signature
    elif proto2.vorticity_signature is not None:
        merged_vort = proto2.vorticity_signature
    
    # FIXED: Always store prototypes as NumPy for CPU-based semantic retrieval
    merged_matrix_np = merged_matrix.get() if hasattr(merged_matrix, 'get') else merged_matrix
    merged_vort_np = merged_vort.get() if hasattr(merged_vort, 'get') else merged_vort
    
    return SemanticPrototype(
        prototype_matrix=merged_matrix_np,
        target_distribution=merged_targets,
        radius=merged_radius,
        support=total_support,
        level=merged_level,
        vorticity_signature=merged_vort_np,
    )


def find_similar_prototype_pairs(
    prototypes: List[SemanticPrototype],
    similarity_threshold: float = PHI_INV,  # φ-derived: ≈ 0.618 (now properly in [-1, 1])
    xp = np,
    basis: np.ndarray = None,
) -> List[Tuple[int, int, float]]:
    """
    Find pairs of prototypes that are highly similar.
    
    THEORY-TRUE SIMILARITY:
        Uses vorticity_similarity which combines:
        - Witness (σ, p) captures SEMANTIC content (cosine normalized)
        - Vorticity (bivectors) captures SYNTACTIC structure (cosine normalized)
        
        Result is in [-1, 1] where:
        - 1.0 = identical
        - 0.0 = orthogonal
        - -1.0 = opposite
        
        Default threshold φ⁻¹ ≈ 0.618 is conservative (only merge near-duplicates).
    
    Returns:
        List of (idx1, idx2, similarity) tuples, sorted by similarity descending
    """
    n = len(prototypes)
    if n < 2:
        return []
    
    # Need basis for proper similarity
    if basis is None:
        from holographic_prod.core.algebra import build_clifford_basis
        basis = build_clifford_basis(xp)
    
    # OPTIMIZED: Batch extract matrices (single CPU→GPU transfer)
    matrices_np = np.empty((n, 4, 4), dtype=np.float32)
    for i, p in enumerate(prototypes):
        mat = p.prototype_matrix
        matrices_np[i] = mat.get() if hasattr(mat, 'get') else mat
    matrices = xp.asarray(matrices_np)  # Single GPU transfer
    
    # VECTORIZED: Extract witnesses for all matrices at once
    # witness = (trace/4, pseudo-scalar component)
    traces = xp.trace(matrices, axis1=1, axis2=2) / 4.0  # [n]
    # Pseudo-scalar: e1234 coefficient (approximated as det for SO(4))
    pseudo_scalars = xp.linalg.det(matrices)  # [n]
    witnesses = xp.stack([traces, pseudo_scalars], axis=1)  # [n, 2]
    
    # VECTORIZED: Compute all pairwise similarities in batch
    # Flatten matrices for cosine similarity
    matrices_flat = matrices.reshape(n, -1)  # [n, 16]
    norms = xp.linalg.norm(matrices_flat, axis=1, keepdims=True)  # [n, 1]
    matrices_normed = matrices_flat / (norms + 1e-10)  # [n, 16]
    
    # Pairwise cosine similarity: [n, n]
    matrix_sims = xp.dot(matrices_normed, matrices_normed.T)
    
    # Witness similarity: cosine of witness vectors
    witness_norms = xp.linalg.norm(witnesses, axis=1, keepdims=True)  # [n, 1]
    witnesses_normed = witnesses / (witness_norms + 1e-10)  # [n, 2]
    witness_sims = xp.dot(witnesses_normed, witnesses_normed.T)  # [n, n]
    
    # Combined similarity: φ⁻¹ × matrix_sim + (1 - φ⁻¹) × witness_sim
    combined_sims = PHI_INV * matrix_sims + (1.0 - PHI_INV) * witness_sims
    
    # Transfer to CPU for pair extraction (single transfer)
    combined_sims_cpu = combined_sims.get() if hasattr(combined_sims, 'get') else combined_sims
    
    # Extract pairs above threshold (pure Python, no GPU syncs)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = combined_sims_cpu[i, j]
            if sim >= similarity_threshold:
                pairs.append((i, j, float(sim)))
    
    # Sort by similarity (highest first)
    pairs.sort(key=lambda x: -x[2])
    
    return pairs


def manage_interference(
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
    similarity_threshold: float = PHI_INV,  # ≈ 0.618 (conservative for cosine in [-1, 1])
    max_merges_per_cycle: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Manage interference by merging highly similar prototypes.
    
    THEORY (Interference Management):
        - Similar memories compete during retrieval
        - High similarity → interference
        - But: aggressive merging destroys coverage → poor generalization!
    
    THEORY-TRUE SIMILARITY:
        Uses vorticity_similarity which returns cosine in [-1, 1]:
        - 1.0 = identical
        - 0.0 = orthogonal
        - -1.0 = opposite
        
        Default threshold φ⁻¹ ≈ 0.618 is conservative (only merge near-duplicates).
        
    This is the Clifford algebra analog of:
        - Pattern separation (keeping things distinct)
        - Pattern completion (merging similar patterns)
        
    Args:
        semantic_memory: SemanticMemory to manage
        basis: Clifford basis
        xp: array module
        similarity_threshold: Above this, prototypes are "too similar" (in [-1, 1])
        max_merges_per_cycle: Limit merges per call (prevent over-merging)
        verbose: Print details
        
    Returns:
        Statistics about merging
    """
    if verbose:
        print("  INTERFERENCE MANAGEMENT")
        print("  " + "-" * 40)
    
    stats = {
        'total_before': semantic_memory.stats()['total_prototypes'],
        'merges': 0,
        'merge_details': [],
    }
    
    # Process each level
    for level_idx in range(semantic_memory.num_levels):
        level = semantic_memory.levels[level_idx]
        
        if len(level) < 2:
            continue
        
        merges_this_level = 0
        
        while merges_this_level < max_merges_per_cycle:
            # Find similar pairs using theory-true combined similarity
            pairs = find_similar_prototype_pairs(level, similarity_threshold, xp, basis)
            
            if not pairs:
                break  # No more similar pairs
            
            # Merge the most similar pair
            i, j, sim = pairs[0]
            
            merged = merge_prototypes(level[i], level[j], basis, xp)
            
            if verbose:
                print(f"    Level {level_idx}: Merged idx {i} and {j} (sim={sim:.3f})")
                print(f"      Support: {level[i].support} + {level[j].support} → {merged.support}")
            
            stats['merge_details'].append({
                'level': level_idx,
                'similarity': sim,
                'support_before': (level[i].support, level[j].support),
                'support_after': merged.support,
            })
            
            # Remove originals (higher index first to preserve lower index)
            del level[j]
            del level[i]
            
            # Add merged
            level.append(merged)
            
            merges_this_level += 1
            stats['merges'] += 1
    
    stats['total_after'] = semantic_memory.stats()['total_prototypes']
    
    if verbose:
        print(f"    Total merges: {stats['merges']}")
        print(f"    Before: {stats['total_before']} → After: {stats['total_after']}")
    
    return stats
