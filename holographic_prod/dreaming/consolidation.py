"""
Non-REM Consolidation — Episodic to Semantic Prototype Formation

Implements brain-inspired Non-REM consolidation:
- Grace-stability determines consolidation urgency
- Clustering by context similarity (not target!)
- Priority-weighted centroid computation
- Target distributions capture ambiguity

THEORY (Chapter 11):
    Master broadcasts witness DOWN to satellites during Non-REM.
    Dissonant satellites (coherence < φ⁻¹) receive accelerated Grace at φ⁻⁴ rate.

SELF-ORGANIZING PRINCIPLE:
    Grace-stability σ(M) determines consolidation:
    - σ = (scalar² + pseudo²) / total_energy
    - Priority = stability × salience
    - High-priority episodes seed clusters (better anchors)
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from holographic_prod.core.constants import (
    PHI_INV, PHI_INV_SQ, PHI_INV_FOUR, PHI_INV_EIGHT, PHI_EPSILON, DTYPE
)
from holographic_prod.core.algebra import (
    grace_operator, frobenius_cosine, grace_basin_keys_batch_direct
)

# Grace basin routing - same settings as tower for consistency
# v5.31.4: Fixed - must match tower settings for consistent clustering
GRACE_ROUTING_ITERS = 3  # φ⁻² contraction per iter (same as tower)
GRACE_ROUTING_RESOLUTION = PHI_INV ** 12  # φ⁻¹² for good satellite distribution
from holographic_prod.core.quotient import grace_stability_batch
from .structures import EpisodicEntry, SemanticPrototype
from .priority import compute_salience_batch


# =============================================================================
# DREAM GRACE (Stronger contraction for sleep)
# =============================================================================

def dream_grace_operator(M: np.ndarray, basis: np.ndarray, xp, rate: float = None) -> np.ndarray:
    """
    Apply Grace operator with stronger contraction for dreaming.
    
    During sleep, we use stronger φ⁻ᵏ scaling to strip away detail
    and force invariants to emerge.
    
    For simplicity, we apply the standard Grace operator multiple times
    to achieve stronger contraction.
    
    Args:
        M: 4x4 multivector matrix
        basis: Clifford basis matrices
        xp: numpy or cupy
        rate: Contraction rate (default: PHI_INV_SQ = stronger than waking)
    
    Returns:
        Contracted multivector
    """
    # Apply standard Grace operator (already uses theory-correct scaling)
    # Multiple applications achieve stronger contraction
    result = grace_operator(M, basis, xp)
    result = grace_operator(result, basis, xp)  # Double application for stronger damping
    return result


# =============================================================================
# NON-REM CONSOLIDATOR
# =============================================================================

class NonREMConsolidator:
    """
    Non-REM sleep phase: consolidation of episodic memories into semantic prototypes.
    
    This addresses:
        - Coverage problem: prototypes generalize beyond exact matches
        - Ambiguity: store target distributions, not single targets
        - Memory growth: compress many episodes into few prototypes
    
    BRAIN-INSPIRED PRIORITY SYSTEM:
        Episodes are prioritized by combined score:
        1. SALIENCE (scalar + pseudoscalar) - what Grace PRESERVES
        2. NOVELTY (distance from prototypes) - what memory DOESN'T KNOW
        3. PREDICTION ERROR (Grace residual) - what was SURPRISING
        
        High-priority episodes:
        - Seed clusters (they're more stable anchors)
        - Have higher weight in centroid computation
        - Produce more stable prototypes
    
    THEORY (Chapter 11):
        Master broadcasts witness DOWN to satellites during Non-REM.
        Dissonant satellites (coherence < φ⁻¹) receive accelerated Grace at φ⁻⁴ rate.
    """
    
    def __init__(
        self,
        basis: np.ndarray,
        xp = np,
        similarity_threshold: float = PHI_INV,  # φ-derived
        min_cluster_size: int = 3,
        grace_rate: float = PHI_INV_SQ,
        use_salience: bool = True,
        use_novelty: bool = True,
        semantic_memory: Optional['SemanticMemory'] = None,
    ):
        """
        Args:
            basis: Clifford algebra basis matrices
            xp: numpy or cupy
            similarity_threshold: Threshold for clustering
            min_cluster_size: Minimum episodes to form a prototype
            grace_rate: Contraction rate for canonicalization
            use_salience: If True, weight episodes by emotional salience
            use_novelty: If True, boost priority for novel episodes
            semantic_memory: Existing semantic memory for novelty computation
        """
        self.basis = basis
        self.xp = xp
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.grace_rate = grace_rate
        self.use_salience = use_salience
        self.use_novelty = use_novelty
        self.semantic_memory = semantic_memory
    
    def canonicalize(self, M: np.ndarray) -> np.ndarray:
        """
        Normalize for clustering (light canonicalization).
        
        We use LIGHT canonicalization to preserve discriminative structure
        while normalizing scale. Heavy Grace contraction is for survival testing,
        not clustering.
        """
        xp = self.xp
        norm = xp.linalg.norm(M, 'fro')
        if norm > PHI_EPSILON:
            return M / norm
        return M
    
    def canonicalize_batch(self, matrices: np.ndarray) -> np.ndarray:
        """Batch canonicalization - vectorized for GPU."""
        xp = self.xp
        norms = xp.linalg.norm(matrices.reshape(len(matrices), -1), axis=1, keepdims=True)
        norms = xp.maximum(norms, PHI_EPSILON)
        return matrices / norms.reshape(-1, 1, 1)
    
    def cluster_episodes(
        self, 
        episodes: List[EpisodicEntry]
    ) -> List[List[EpisodicEntry]]:
        """
        Cluster episodic entries by CONTEXT SIMILARITY (NOT by target!).
        
        THEORY-TRUE APPROACH (Brain-Analog):
            The brain clusters by SIMILARITY OF EXPERIENCE, not by outcome.
            Similar contexts get grouped even if they led to different targets.
            This creates prototypes with TARGET DISTRIBUTIONS, enabling:
            
            - Generalization: "I saw _" → {the: 0.4, a: 0.3, him: 0.2, ...}
            - Uncertainty encoding: Distribution entropy reflects ambiguity
            - Population coding: Multiple targets contribute to prediction
        """
        if len(episodes) == 0:
            return []
        
        return self._cluster_by_context_similarity(episodes)
    
    def _cluster_by_context_similarity(
        self,
        episodes: List[EpisodicEntry]
    ) -> List[List[EpisodicEntry]]:
        """Cluster episodes by context similarity, allowing mixed targets."""
        if len(episodes) <= self.min_cluster_size:
            return [episodes] if len(episodes) >= self.min_cluster_size else []
        
        return self._cluster_within_target(episodes)
    
    def _cluster_within_target(
        self,
        episodes: List[EpisodicEntry]
    ) -> List[List[EpisodicEntry]]:
        """
        Cluster episodes by Grace basin key — THEORY-TRUE O(n) approach.
        
        THEORY (Parsimonious):
            The Grace operator defines attractor basins in multivector space.
            Episodes with the same basin key are "similar" by definition.
            
            This is the SAME locality-sensitive hashing used by the tower
            for routing contexts to satellites. We reuse it here for
            consistent, theory-true clustering.
        
        COMPLEXITY: O(n) instead of O(n × k)
            - One batch basin key computation
            - One grouping pass
            - No pairwise comparisons
        """
        if len(episodes) <= self.min_cluster_size:
            return [episodes] if len(episodes) >= self.min_cluster_size else []
        
        xp = self.xp
        n_episodes = len(episodes)
        
        # 1. OPTIMIZED: Batch extract matrices (single CPU→GPU transfer)
        matrices_np = np.empty((n_episodes, 4, 4), dtype=DTYPE)
        for i, ep in enumerate(episodes):
            mat = ep.context_matrix
            matrices_np[i] = mat.get() if hasattr(mat, 'get') else mat
        matrices = xp.asarray(matrices_np)  # Single transfer
        
        # 2. Compute basin keys for ALL episodes at once (VECTORIZED GPU)
        # This uses the same Grace attractor dynamics as the tower routing
        basin_keys = grace_basin_keys_batch_direct(
            matrices, self.basis,
            n_iters=GRACE_ROUTING_ITERS,
            resolution=GRACE_ROUTING_RESOLUTION,
            xp=xp
        )
        
        # 3. Convert basin keys to hashable cluster IDs
        # Simple prime-based hash (same as tower routing for consistency)
        primes = xp.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53], dtype=xp.int64)
        cluster_ids = xp.sum(basin_keys * primes, axis=1)
        
        # Move to CPU for grouping (grouping is inherently sequential)
        if hasattr(cluster_ids, 'get'):
            cluster_ids = cluster_ids.get()
        
        # 4. Group episodes by cluster ID (O(n) single pass)
        clusters_dict = defaultdict(list)
        for i, cid in enumerate(cluster_ids):
            clusters_dict[int(cid)].append(episodes[i])
        
        # 5. Filter by min cluster size
        return [c for c in clusters_dict.values() if len(c) >= self.min_cluster_size]
    
    def create_prototype(
        self, 
        cluster: List[EpisodicEntry]
    ) -> Optional[SemanticPrototype]:
        """
        Create a semantic prototype from a cluster of episodes.
        
        Computes:
            - Prototype matrix (PRIORITY-WEIGHTED centroid)
            - Target distribution (weighted by counts × priority)
            - Radius (max distance to centroid)
            - Vorticity signature (averaged across cluster for grammar matching)
        """
        if len(cluster) < self.min_cluster_size:
            return None
        
        xp = self.xp
        
        # Ensure matrices are on device (GPU) - handle mixed CPU/GPU arrays
        raw_matrices = [ep.context_matrix for ep in cluster]
        if xp != np and hasattr(raw_matrices[0], 'get'):
            # Already on GPU
            matrices = xp.stack(raw_matrices)
        else:
            # CPU arrays - transfer to device
            matrices = xp.asarray(np.array(raw_matrices))
        
        if self.use_salience or self.use_novelty:
            # Priority-weighted centroid
            priorities = xp.array([max(PHI_INV_EIGHT, float(ep.priority)) for ep in cluster])
            priority_weights = priorities / (xp.sum(priorities) + PHI_EPSILON)
            centroid = xp.einsum('n,nij->ij', priority_weights, matrices)
        else:
            centroid = xp.mean(matrices, axis=0)
            priority_weights = xp.ones(len(cluster)) / len(cluster)
        
        # Apply canonicalization
        centroid = self.canonicalize(centroid)
        
        # Aggregate vorticity signatures
        vort_sigs = []
        for ep in cluster:
            if ep.vorticity_signature is not None:
                vort_sigs.append(ep.vorticity_signature)
        
        if vort_sigs:
            # GPU-NATIVE: Use xp.asarray for each vorticity signature
            vort_sigs_array = xp.stack([xp.asarray(v) for v in vort_sigs])
            valid_weights = priority_weights[:len(vort_sigs)]
            valid_weights = valid_weights / (xp.sum(valid_weights) + PHI_EPSILON)
            avg_vort_sig = xp.einsum('n,nj->j', valid_weights, vort_sigs_array)
        else:
            avg_vort_sig = xp.zeros(16, dtype=DTYPE)
        
        # Compute target distribution
        target_counts: Dict[int, float] = defaultdict(float)
        total_weight = 0.0
        
        for ep in cluster:
            weight = ep.count * (1.0 + ep.priority)
            target_counts[ep.target_token] += weight
            total_weight += weight
        
        target_dist = {
            token: count / total_weight 
            for token, count in target_counts.items()
        }
        
        # Compute radius using cosine distance (use already-transferred matrices)
        distances = [
            1.0 - frobenius_cosine(centroid, matrices[i], self.xp)
            for i in range(len(cluster))
        ]
        radius = max(distances) if distances else 0.0
        
        # FIXED: Always store prototypes as NumPy for CPU-based semantic retrieval
        centroid_np = centroid.get() if hasattr(centroid, 'get') else centroid
        avg_vort_sig_np = avg_vort_sig.get() if hasattr(avg_vort_sig, 'get') else avg_vort_sig
        
        return SemanticPrototype(
            prototype_matrix=centroid_np,
            target_distribution=target_dist,
            radius=radius,
            support=len(cluster),
            level=0,
            vorticity_signature=avg_vort_sig_np,
        )
    
    def consolidate(
        self, 
        episodes: List[EpisodicEntry],
        verbose: bool = False,
    ) -> List[SemanticPrototype]:
        """
        Main consolidation routine: episodes → prototypes.
        
        SELF-ORGANIZING PRINCIPLE (Theory-Derived):
            The Grace operator determines what survives.
            - σ < φ⁻² → consolidate (transient)
            - ≥3 same target → consolidate (redundant)
        """
        if verbose:
            print(f"  Non-REM: Consolidating {len(episodes)} episodes...")
        
        xp = self.xp
        
        if len(episodes) == 0:
            return []
        
        n_episodes = len(episodes)
        
        # OPTIMIZED: Extract matrices in bulk (single CPU→GPU transfer)
        # Pre-allocate numpy array, then transfer once to GPU
        matrices_np = np.empty((n_episodes, 4, 4), dtype=DTYPE)
        targets = np.empty(n_episodes, dtype=np.int32)
        for i, ep in enumerate(episodes):
            mat = ep.context_matrix
            matrices_np[i] = mat.get() if hasattr(mat, 'get') else mat
            targets[i] = ep.target_token
        matrices = xp.asarray(matrices_np)  # Single GPU transfer
        
        # Compute SALIENCE and STABILITY in batch (stays on GPU)
        saliences = compute_salience_batch(matrices, self.basis, xp)
        stabilities = grace_stability_batch(matrices, self.basis, xp)
        
        # Transfer results to CPU in bulk (single GPU→CPU transfer)
        saliences_cpu = saliences.get() if hasattr(saliences, 'get') else saliences
        stabilities_cpu = stabilities.get() if hasattr(stabilities, 'get') else stabilities
        
        # Compute priorities on CPU (numpy random, no GPU sync)
        priorities_cpu = saliences_cpu + 0.001 * np.random.rand(n_episodes).astype(DTYPE)
        
        # Detect redundancy by target (pure Python, fast for 10K)
        target_groups = defaultdict(list)
        for i in range(n_episodes):
            target_groups[targets[i]].append(i)
        
        redundant_mask = np.zeros(n_episodes, dtype=bool)
        for target, indices in target_groups.items():
            if len(indices) >= 3:
                for idx in indices:
                    redundant_mask[idx] = True
        
        # Build consolidation mask (vectorized)
        transient_mask = stabilities_cpu < PHI_INV_SQ
        consolidate_mask = transient_mask | redundant_mask
        
        n_transient = int(np.sum(transient_mask))
        n_redundant = int(np.sum(redundant_mask & ~transient_mask))
        
        # Annotate episodes that will be consolidated (only these need attributes)
        episodes_to_consolidate = []
        for i in range(n_episodes):
            if consolidate_mask[i]:
                ep = episodes[i]
                ep.stability = float(stabilities_cpu[i])
                ep.salience = float(saliences_cpu[i])
                ep.priority = float(priorities_cpu[i])
                episodes_to_consolidate.append(ep)
        
        if verbose:
            avg_stability = float(xp.mean(stabilities))
            n_unique_targets = len(target_groups)
            n_kept = len(episodes) - len(episodes_to_consolidate)
            print(f"    Grace-stability: avg={avg_stability:.3f}")
            print(f"    Unique targets: {n_unique_targets}")
            print(f"    Consolidation criteria:")
            print(f"      - Transient (σ < φ⁻²): {n_transient}")
            print(f"      - Redundant (≥3 same target): {n_redundant}")
            print(f"    Kept episodic: {n_kept}")
            print(f"    To consolidate: {len(episodes_to_consolidate)}")
        
        if not episodes_to_consolidate:
            if verbose:
                print(f"    No episodes meet consolidation criteria")
            return []
        
        # Sort by priority (highest first = seeds clusters)
        episodes_to_consolidate = sorted(
            episodes_to_consolidate, 
            key=lambda ep: ep.priority, 
            reverse=True
        )
        
        # Cluster
        clusters = self.cluster_episodes(episodes_to_consolidate)
        if verbose:
            print(f"    Found {len(clusters)} clusters")
        
        # Create prototypes
        prototypes = []
        for cluster in clusters:
            proto = self.create_prototype(cluster)
            if proto is not None:
                prototypes.append(proto)
        
        if verbose:
            print(f"    Created {len(prototypes)} prototypes")
            if prototypes:
                avg_entropy = np.mean([p.entropy() for p in prototypes])
                avg_support = np.mean([p.support for p in prototypes])
                print(f"    Avg entropy: {avg_entropy:.3f}")
                print(f"    Avg support: {avg_support:.1f}")
        
        return prototypes
    
    def broadcast_master_witness(
        self,
        master_witness: np.ndarray,
        satellite_witnesses: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Broadcast master witness down to satellites with accelerated Grace for dissonant ones.
        
        Theory (Chapter 11):
            for each satellite k:
                coherence = dot(master_witness, satellite_witness)
                if coherence < φ⁻¹:  # Dissonant
                    rate = φ⁻⁴       # Accelerated Grace
                else:
                    rate = φ⁻²       # Normal consolidation
                satellite.witness = (1 - rate) * satellite.witness + rate * master.witness
        """
        from holographic_prod.core.quotient import extract_witness
        
        updated_satellites = []
        
        master_s, master_p = extract_witness(master_witness, self.basis, self.xp)
        master_witness_vec = np.array([master_s, master_p])
        master_norm = np.linalg.norm(master_witness_vec)
        
        if master_norm < PHI_EPSILON:
            return satellite_witnesses
        
        master_witness_normalized = master_witness_vec / master_norm
        
        for sat_witness in satellite_witnesses:
            sat_s, sat_p = extract_witness(sat_witness, self.basis, self.xp)
            sat_witness_vec = np.array([sat_s, sat_p])
            sat_norm = np.linalg.norm(sat_witness_vec)
            
            if sat_norm < PHI_EPSILON:
                updated_satellites.append(master_witness.copy())
                continue
            
            sat_witness_normalized = sat_witness_vec / sat_norm
            coherence = float(np.dot(master_witness_normalized, sat_witness_normalized))
            
            if coherence < PHI_INV:  # Dissonant
                rate = PHI_INV_FOUR
                num_steps = 4
            else:  # Consonant
                rate = PHI_INV_SQ
                num_steps = 1
            
            updated_witness = sat_witness.copy()
            for step in range(num_steps):
                updated_witness = (1 - rate) * updated_witness + rate * master_witness
                updated_witness = grace_operator(updated_witness, self.basis, self.xp)
            
            updated_satellites.append(updated_witness)
        
        return updated_satellites
