"""
Memory Scaling Analysis — Theoretical Bounds and Empirical Verification

MEMORY SCALING THEOREM
======================

QUESTION: How does prototype count grow with episode count?

CLAIM: Prototype count grows SUBLINEARLY in episode count.

ANALYSIS:
    Two mechanisms control prototype growth:
    
    1. WITNESS CLUSTERING:
       - Prototypes cover regions in witness space
       - Witness is 2D: (scalar, pseudoscalar)
       - Region volume ∝ φ⁻² (merge threshold)
       - Max prototypes ≈ witness_space_volume / region_volume
       
    2. φ-DECAY FORGETTING:
       - Low-salience prototypes decay over time
       - Survival probability = φ^(-k × (1 - priority))
       - Creates natural sparsity
       
THEORETICAL BOUND:
    Let N = episode count, P(N) = prototype count
    
    Without merging: P(N) = O(N)
    With merging: P(N) = O(min(N, V/r²))
    
    where V = witness space volume, r = merge radius ≈ φ⁻²
    
    Since witness is 2D:
        P(N) = O(min(N, 1/φ⁻⁴)) = O(min(N, φ⁴)) ≈ O(min(N, 7))
        
    In practice with real data: P(N) grows as O(√N) or O(log N)
    due to clustering in semantic space.
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from holographic_prod.core.constants import PHI_INV_FOUR, DTYPE
from holographic_prod.core.quotient import extract_witness

from .structures import EpisodicEntry

if TYPE_CHECKING:
    from .dreaming_system import DreamingSystem
    from .semantic_memory import SemanticMemory


def analyze_memory_scaling(
    dreaming: 'DreamingSystem',
    episode_counts: List[int] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Analyze how prototype count scales with episode count.
    
    Args:
        dreaming: DreamingSystem to test
        episode_counts: List of episode counts to test
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Scaling analysis results
    """
    from .semantic_memory import SemanticMemory
    
    if episode_counts is None:
        episode_counts = [100, 500, 1000, 2000, 5000]
    
    rng = np.random.default_rng(seed)
    xp = dreaming.xp
    basis = dreaming.basis
    
    results = {
        'episode_counts': [],
        'prototype_counts': [],
        'scaling_ratios': [],
    }
    
    previous_proto_count = 0
    previous_episode_count = 0
    
    for n_episodes in episode_counts:
        if verbose:
            print(f"  Testing with {n_episodes} episodes...")
        
        # Reset semantic memory
        dreaming.semantic_memory = SemanticMemory(basis, xp, num_levels=3)
        
        # Generate episodes (10 different semantic clusters)
        n_clusters = 10
        episodes = []
        
        for i in range(n_episodes):
            cluster = i % n_clusters
            
            # Each cluster has a distinct base matrix (φ⁻⁴ cluster separation)
            base = np.eye(4) + PHI_INV_FOUR * cluster * np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]
            ], dtype=DTYPE)
            
            # Add φ⁻⁶ noise
            ctx_matrix = base + PHI_INV_SIX * rng.standard_normal((4, 4))
            ctx_matrix = grace_operator(ctx_matrix, basis, xp)
            
            target = cluster * 10 + rng.integers(0, 5)
            
            episodes.append(EpisodicEntry(ctx_matrix, target))
        
        # Run consolidation
        dreaming.sleep(episodes, rem_cycles=3, verbose=False)
        
        # Count prototypes at all levels
        total_protos = sum(len(level) for level in dreaming.semantic_memory.levels)
        
        results['episode_counts'].append(n_episodes)
        results['prototype_counts'].append(total_protos)
        
        # Compute scaling ratio
        if previous_episode_count > 0:
            episode_growth = n_episodes / previous_episode_count
            proto_growth = (total_protos + 1) / (previous_proto_count + 1)
            ratio = proto_growth / episode_growth
            results['scaling_ratios'].append(ratio)
        
        previous_proto_count = total_protos
        previous_episode_count = n_episodes
        
        if verbose:
            print(f"    Episodes: {n_episodes}, Prototypes: {total_protos}")
    
    # Analyze scaling
    episodes = np.array(results['episode_counts'])
    protos = np.array(results['prototype_counts'])
    
    # Fit log-log slope (power law: P = c * N^α)
    if len(episodes) > 2:
        log_e = np.log(episodes + 1)
        log_p = np.log(protos + 1)
        
        A = np.column_stack([log_e, np.ones_like(log_e)])
        slope, intercept = np.linalg.lstsq(A, log_p, rcond=None)[0]
        
        results['scaling_exponent'] = float(slope)
        results['is_sublinear'] = slope < 1.0
    else:
        results['scaling_exponent'] = None
        results['is_sublinear'] = None
    
    if results['scaling_ratios']:
        results['mean_scaling_ratio'] = float(np.mean(results['scaling_ratios']))
    else:
        results['mean_scaling_ratio'] = None
    
    return results


def estimate_memory_capacity(
    dreaming: 'DreamingSystem',
) -> Dict[str, float]:
    """
    Estimate memory capacity based on witness space coverage.
    
    THEORY:
        - Witness is 2D: (scalar, pseudoscalar)
        - Effective range: [-1, 1] for normalized matrices
        - Merge radius: φ⁻² ≈ 0.38
        - Max prototypes ≈ area / radius²
        
    Returns:
        Capacity estimates
    """
    xp = dreaming.xp
    basis = dreaming.basis
    
    total_protos = sum(len(level) for level in dreaming.semantic_memory.levels)
    
    witnesses = []
    for level in dreaming.semantic_memory.levels:
        for proto in level:
            s, p = extract_witness(proto.prototype_matrix, basis, xp)
            witnesses.append((s, p))
    
    if not witnesses:
        return {
            'current_prototypes': 0,
            'estimated_capacity': float('inf'),
            'utilization': 0.0,
        }
    
    witnesses = np.array(witnesses)
    
    if len(witnesses) > 1:
        s_range = witnesses[:, 0].max() - witnesses[:, 0].min()
        p_range = witnesses[:, 1].max() - witnesses[:, 1].min()
        covered_area = (s_range + PHI_INV_CUBE) * (p_range + PHI_INV_CUBE)
    else:
        covered_area = PHI_INV_CUBE * PHI_INV_CUBE
    
    merge_radius = PHI_INV_SQ
    region_area = np.pi * merge_radius ** 2
    estimated_capacity = covered_area / region_area
    
    utilization = total_protos / max(estimated_capacity, 1)
    
    return {
        'current_prototypes': total_protos,
        'covered_witness_area': float(covered_area),
        'merge_region_area': float(region_area),
        'estimated_capacity': float(estimated_capacity),
        'utilization': float(utilization),
    }


def measure_prototype_entropy(
    dreaming: 'DreamingSystem',
) -> Dict[str, float]:
    """
    Measure entropy of prototype distribution in witness space.
    
    High entropy = good coverage, efficient memory use
    Low entropy = clustering, potential redundancy
    
    Returns:
        Entropy metrics
    """
    xp = dreaming.xp
    basis = dreaming.basis
    
    witnesses = []
    for level in dreaming.semantic_memory.levels:
        for proto in level:
            s, p = extract_witness(proto.prototype_matrix, basis, xp)
            witnesses.append((s, p))
    
    if len(witnesses) < 2:
        return {
            'witness_entropy': 0.0,
            'normalized_entropy': 0.0,
            'n_prototypes': len(witnesses),
            'n_bins_used': 0,
            'n_bins_total': 0,
        }
    
    witnesses = np.array(witnesses)
    
    n_bins = 10
    s_bins = np.linspace(witnesses[:, 0].min() - PHI_INV_EIGHT, witnesses[:, 0].max() + PHI_INV_EIGHT, n_bins + 1)
    p_bins = np.linspace(witnesses[:, 1].min() - PHI_INV_EIGHT, witnesses[:, 1].max() + PHI_INV_EIGHT, n_bins + 1)
    
    hist, _, _ = np.histogram2d(witnesses[:, 0], witnesses[:, 1], bins=[s_bins, p_bins])
    
    hist_flat = hist.flatten()
    hist_flat = hist_flat[hist_flat > 0]
    probs = hist_flat / hist_flat.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    max_entropy = np.log(n_bins * n_bins)
    normalized_entropy = entropy / max_entropy
    
    return {
        'witness_entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'n_prototypes': len(witnesses),
        'n_bins_used': len(hist_flat),
        'n_bins_total': n_bins * n_bins,
    }
