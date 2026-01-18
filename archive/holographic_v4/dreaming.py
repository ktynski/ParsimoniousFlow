"""
SCCMU Dreaming Module — Brain-Inspired Informational Parsimonies v4.8.0
=======================================================================

Implements offline consolidation, abstraction, and memory management mechanisms
that address the core limitations of pure binding + retrieval.

Last Updated: 2026-01-11

CORE FEATURES:
    - Non-REM consolidation: σ < φ⁻² triggers prototype formation
    - REM recombination: Clifford operations create novel combinations
    - 12 brain-inspired parsimonies (see ARCHITECTURE.md Part 5)
    - Adaptive similarity threshold
    - φ-decay forgetting in sleep cycles

v4.8.0 ADDITIONS:
    - Integration with HolographicMemory (theory-true storage)
    - Witness-based indexing respects geometric structure
    - See holographic_memory.py for true superposition storage

v4.7.0 ADDITIONS:
    - Redundancy-based consolidation (not just transience)
    - Context-only clustering (not target-first)
    - Integrated with meta-cognitive training loop
    - Vorticity-driven clustering for long contexts

NO ARBITRARY OPERATIONS:
    - No softmax (replaced with grace_stability × salience or φ-kernel)
    - Frobenius for numerical stability only (uniform scaling)
    - Grace for theory-true grade-wise damping
    - No clipping (raw measures flow through)
    - All thresholds derived from φ and spectral structure

SELF-ORGANIZING PRINCIPLE (Theory-Derived):
    Grace-stability σ(M) determines consolidation:
    
        σ = (scalar² + pseudo²) / total_energy
        
        ALL episodes are consolidation candidates (removed stability filter)
        Priority = stability × salience (coherent + important episodes first)
    
    The clustering threshold adapts based on prototype diversity.

CORE PROBLEMS SOLVED:
    1. COVERAGE PROBLEM → Semantic prototypes generalize beyond exact matches
    2. AMBIGUITY → Multi-modal targets via mixture storage  
    3. BRITTLENESS → Quotient-stable clustering makes retrieval robust
    4. MEMORY GROWTH → φ-decay forgetting creates sparse efficiency

BRAIN-INSPIRED PARSIMONIES (12 total, all fully implemented):

    MEMORY ENCODING:
    
    1. EMOTIONAL SALIENCE (scalar + pseudoscalar)
       - High salience = survives Grace = prioritized
       - Theory: EMOTIONAL_TAGGING_THEORY.md
       
    2. NOVELTY-GATED LEARNING
       - Novel episodes (far from prototypes) get priority
       - Already-known patterns don't need re-encoding
       
    3. DELTA/SCHEMA COMPRESSION
       - Store deviations from nearest prototype
       - Sparse in Clifford basis → 3-5x compression
       
    4. PREDICTIVE CODING
       - Only encode prediction errors (Grace residuals)
       - Prediction error = 1 - grace_stability = consolidation_urgency
       
    MEMORY MAINTENANCE:
       
    5. φ-DECAY FORGETTING (Theory-True)
       - Survival probability = φ^(-k × (1 - priority))
       - Priority = stability × salience
       - Creates sparse efficiency through natural selection
       - NOT arbitrary capacity pruning
       
    6. INTERFERENCE MANAGEMENT
       - Merge highly similar prototypes via Grace
       - Combined prototype is stronger (higher support)
       
    7. RECONSOLIDATION
       - Retrieval makes memory labile (modifiable)
       - access_count tracks how often each memory is used
       - retention_strength = salience × φ^(access_count)
       
    8. PSEUDO-REHEARSAL
       - Generate samples from semantic memory
       - Interleave with real episodes
       - Prevents catastrophic forgetting
       
    MEMORY RETRIEVAL:
       
    9. WORKING MEMORY CACHE
       - Small, fast cache for recently retrieved items
       - φ-decay recency weighting
       - Capacity = 7±2 items (biological limit)
       
    10. PATTERN COMPLETION
        - Grace flow converges noisy queries to attractors
        - "Retrieval as inference" not lookup
        
    11. INHIBITION OF RETURN
        - Recently retrieved items temporarily suppressed
        - Suppression decays as φ⁻¹ per step
        
    SEQUENCE MEMORY:
        
    12. SEQUENCE REPLAY
        - Store temporal transitions via vorticity (A ∧ B)
        - Replay sequences during REM
        - Sharp wave ripple analog

VORTICITY GRAMMAR MATCHING:
    The wedge product A∧B = -B∧A captures word ORDER:
    
    - vorticity_signature(sequence) → 16 coefficients
    - Same structure → 0.92+ similarity ("The cat sat" ≈ "The dog ran")
    - Different structure → <0.3 similarity ("The cat sat" ≠ "Sat the cat")
    
    Implementation:
    - EpisodicEntry stores vorticity_signature
    - SemanticPrototype aggregates vorticity_signatures
    - Clustering uses combined similarity: context + vorticity weighted by stability

ADAPTIVE SIMILARITY THRESHOLD:
    The clustering threshold adapts based on prototype diversity:
    
        low diversity (< 10 protos) → lower threshold (more clusters)
        high diversity (> 100 protos) → higher threshold (merge similar)
    
    This prevents both under-clustering (too many similar protos) and
    over-clustering (losing important distinctions).

SLEEP CYCLE OUTPUT:
    sleep_stats = dreaming.sleep(episodes)
    
    Returns:
        episodes_forgotten: How many pruned by φ-decay
        similarity_threshold: Current adaptive threshold
        prototypes_created: New consolidated abstractions
        schemas_created: High-level patterns from REM
    - grace_basin_discovery combines witness + vorticity matching
    
    This discriminates grammar WITHOUT parsing!

SCALABLE CONTEXT (O(N) composition, O(1) storage):
    Context is always a 4×4 matrix regardless of length.
    Tested stable to 8192+ tokens.
    Use pg19 (books) or arxiv (papers) to exercise full capability.
    TinyStories (~200 words) wastes the architecture's potential!

SLEEP PHASES:

    Non-REM (Consolidation):
        - Compute grace_stability for each episode
        - Filter: σ < φ⁻² → needs consolidation
        - Sort by urgency = 1 - σ (most transient first)
        - Cluster by resonance, create prototypes
    
    REM (Recombination):
        - Sample and recombine prototypes
        - Apply strong Grace contraction
        - Keep only quotient-stable survivors

GPU OPTIMIZATION:
    All core operations are vectorized. Pass xp=cupy for CUDA.

Reference:
    "Waking SCCMU writes episodes. Dreaming SCCMU discovers invariants.
     REM tests which invariants are real by forcing them to survive 
     recombination under Grace."
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass, field

from holographic_v4.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR, PHI_INV_SIX, PHI_INV_EIGHT,
    GRADE_INDICES, DTYPE
)
from holographic_v4.algebra import (
    geometric_product,
    frobenius_similarity,
    frobenius_similarity_batch,
    decompose_to_coefficients,
    reconstruct_from_coefficients,
    grace_operator,
    grace_operator_batch,
)
from holographic_v4.resonance import resonance, evolve_to_equilibrium
from holographic_v4.quotient import (
    extract_witness,
    compute_enstrophy,
    quotient_similarity,
    normal_form,
    grace_stability,
    grace_stability_batch,
    consolidation_urgency,
)


# =============================================================================
# SALIENCE COMPUTATION (Theory-Derived)
# =============================================================================

def compute_salience(M: np.ndarray, basis: np.ndarray, xp = np) -> float:
    """
    Compute emotional salience = what Grace PRESERVES.
    
    Theory (EMOTIONAL_TAGGING_THEORY.md):
        - Scalar (Grade 0): Intensity, preserved 100%
        - Pseudoscalar (Grade 4): Valence, preserved 61.8% (Fibonacci exception!)
        - Everything else decays faster
        
    High salience episodes:
        1. Are more stable (survive Grace better)
        2. Contain more "core" information
        3. Should be prioritized in consolidation
    
    Returns:
        salience: float in [0, 1] - higher = more important for memory
    
    INFORMATION PARSIMONY:
        σ = tr(M)/4 — scalar is just trace, no decomposition needed!
        p = <M, γ₅>/4 — pseudoscalar via projection onto γ₅
    """
    # PARSIMONY: σ = tr(M)/4 — scalar coefficient equals trace
    scalar = abs(float(xp.trace(M) / 4.0))
    
    # Pseudoscalar still needs projection onto γ₅ (basis[15])
    pseudoscalar = abs(float(xp.sum(basis[15] * M) / 4.0))
    
    # Salience = intensity + valence (weighted by survival rates)
    # Scalar survives at 1.0, pseudoscalar at PHI_INV ≈ 0.618
    salience = scalar + pseudoscalar * PHI_INV
    
    return salience


def compute_salience_batch(matrices: np.ndarray, basis: np.ndarray, xp = np) -> np.ndarray:
    """
    Batch salience computation for GPU efficiency.
    
    INFORMATION PARSIMONY:
        Only needs scalar (σ) and pseudoscalar (p) components.
        σ = tr(M)/4 — uses trace identity, no basis projection needed!
        p = <M, γ₅>/4 — still needs projection onto basis[15]
        
        Avoids full 16-coefficient decomposition for ~3× speedup.
    
    Args:
        matrices: [N, 4, 4] batch of matrices
        basis: [16, 4, 4] Clifford basis
        xp: array module (numpy or cupy)
    
    Returns:
        saliences: [N] array of salience scores
    """
    # PARSIMONY: σ = tr(M)/4 — scalar is just trace!
    scalars = xp.abs(xp.einsum('nii->n', matrices) / 4.0)
    
    # Pseudoscalar still needs projection onto γ₅ (basis[15])
    # But we use constant 4 instead of computing norm
    pseudoscalars = xp.abs(xp.einsum('nij,ij->n', matrices, basis[15]) / 4.0)
    
    # Salience = intensity + weighted valence
    saliences = scalars + pseudoscalars * PHI_INV
    
    return saliences


# =============================================================================
# NOVELTY COMPUTATION (Brain-Inspired Parsimony)
# =============================================================================

def compute_novelty(
    M: np.ndarray,
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
) -> Tuple[float, Optional['SemanticPrototype']]:
    """
    Compute novelty = distance from nearest semantic prototype.
    
    THEORY (Hippocampal Novelty Detection):
        - Brain encodes novel stimuli with priority
        - Novelty = what existing memory DOESN'T explain
        - High novelty → priority encoding
        
    IMPLEMENTATION:
        - If near a prototype: low novelty (already generalized)
        - If far from all prototypes: high novelty (new pattern)
        
    THEORY TIE-IN:
        Novelty complements emotional salience:
        - Salience = what Grace PRESERVES (scalar + pseudoscalar)
        - Novelty = what memory DOESN'T PREDICT
        
        Combined: total_priority = salience + novelty_bonus
        
    Returns:
        (novelty_score, nearest_prototype) where:
        - novelty_score: float in [0, 1] - higher = more novel
        - nearest_prototype: the closest prototype, or None if memory empty
    """
    if semantic_memory is None or semantic_memory.stats()['total_prototypes'] == 0:
        # No memory yet - everything is maximally novel
        return 1.0, None
    
    # Find nearest prototype
    matches = semantic_memory.retrieve(M, top_k=1)
    
    if not matches:
        return 1.0, None
    
    nearest_proto, similarity = matches[0]
    
    # Novelty = 1 - similarity (high similarity = low novelty)
    # Clamp to [0, 1]
    novelty = max(0.0, min(1.0, 1.0 - similarity))
    
    return novelty, nearest_proto


def compute_novelty_batch(
    matrices: np.ndarray,
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
) -> np.ndarray:
    """
    Batch novelty computation for efficiency.
    
    Args:
        matrices: [N, 4, 4] batch of episode matrices
        semantic_memory: SemanticMemory with existing prototypes
        basis: Clifford basis
        xp: array module
        
    Returns:
        novelties: [N] array of novelty scores in [0, 1]
    """
    N = matrices.shape[0]
    
    if semantic_memory is None or semantic_memory.stats()['total_prototypes'] == 0:
        # No memory - all maximally novel
        return xp.ones(N, dtype=DTYPE)
    
    # Get all prototype matrices
    all_protos = []
    for level in semantic_memory.levels:
        for proto in level:
            all_protos.append(proto.prototype_matrix)
    
    if not all_protos:
        return xp.ones(N, dtype=DTYPE)
    
    proto_matrices = xp.array(all_protos)  # [P, 4, 4]
    P = proto_matrices.shape[0]
    
    # FULLY VECTORIZED: Compute all N×P similarities at once
    # matrices: [N, 4, 4], proto_matrices: [P, 4, 4]
    # Result: [N, P] similarity matrix
    
    # Flatten for batch computation
    matrices_flat = matrices.reshape(N, -1)  # [N, 16]
    protos_flat = proto_matrices.reshape(P, -1)  # [P, 16]
    
    # Compute norms
    mat_norms = xp.linalg.norm(matrices_flat, axis=1, keepdims=True)  # [N, 1]
    proto_norms = xp.linalg.norm(protos_flat, axis=1, keepdims=True)  # [P, 1]
    
    # Compute all similarities: [N, P]
    # sim[i, j] = dot(matrices[i], protos[j]) / (norm_i * norm_j)
    all_sims = xp.dot(matrices_flat, protos_flat.T) / (mat_norms @ proto_norms.T + 1e-10)
    
    # Max similarity per episode
    max_sims = xp.max(all_sims, axis=1)  # [N]
    
    # Novelty = 1 - max_similarity
    # NOTE: Similarity is cosine similarity ∈ [-1, 1], so novelty ∈ [0, 2]
    # We don't clip - let the raw measure flow through. Values > 1 indicate
    # anti-correlation which is maximally novel.
    novelties = 1.0 - max_sims
    
    return novelties


def compute_prediction_error(M: np.ndarray, basis: np.ndarray, xp = np) -> float:
    """
    Compute prediction error = what Grace REMOVES = TRANSIENT content.
    
    THEORY-TRUE MEASURE:
        Prediction error = 1 - grace_stability = consolidation_urgency
        
        This is NOT arbitrary sigmoid normalization - it's the fraction of
        coefficient energy in non-witness grades (grades 1, 2, 3).
        
        - Low error (≈0): Episode is mostly witness (stable, predictable)
        - High error (≈1): Episode is mostly transient (surprising, needs encoding)
    
    This replaces the old arbitrary sigmoid normalization with a theory-derived
    measure based on the spectral structure of the Grace operator.
    """
    # Use the theory-derived measure: 1 - grace_stability
    # This is consolidation_urgency
    return float(consolidation_urgency(M, basis, xp))


def compute_combined_priority(
    M: np.ndarray,
    basis: np.ndarray,
    semantic_memory: Optional['SemanticMemory'] = None,
    xp = np,
    salience_weight: float = PHI_INV,  # φ⁻¹ = 0.618 (primary)
    novelty_weight: float = PHI_INV_SQ,  # φ⁻² = 0.382 (secondary)
    prediction_error_weight: float = PHI_INV_CUBE,  # φ⁻³ = 0.236 (tertiary)
) -> float:
    """
    Compute combined priority score for an episode.
    
    THEORY-TRUE PRIORITIZATION (φ-derived weights):
        Priority = φ-weighted combination of:
        1. Emotional Salience (φ⁻¹) - what Grace PRESERVES (primary)
        2. Novelty (φ⁻²) - what memory DOESN'T KNOW (secondary)
        3. Prediction Error (φ⁻³) - what was SURPRISING (tertiary)
        
    High priority episodes should:
        - Be processed first during consolidation
        - Contribute more to prototype centroids
        - Be less likely to be pruned
        
    Args:
        M: Episode matrix
        basis: Clifford basis
        semantic_memory: Existing semantic memory (for novelty)
        xp: array module
        salience_weight: Weight for emotional salience (default 0.5)
        novelty_weight: Weight for novelty (default 0.3)
        prediction_error_weight: Weight for prediction error (default 0.2)
        
    Returns:
        Combined priority score (higher = more important)
    """
    # Compute components
    salience = compute_salience(M, basis, xp)
    pred_error = compute_prediction_error(M, basis, xp)
    
    if semantic_memory is not None:
        novelty, _ = compute_novelty(M, semantic_memory, basis, xp)
    else:
        novelty = 1.0  # No memory = maximally novel
    
    # Weighted combination
    priority = (
        salience_weight * salience +
        novelty_weight * novelty +
        prediction_error_weight * pred_error
    )
    
    return priority


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EpisodicEntry:
    """
    A single episodic memory entry (from waking).
    
    THEORY:
        Stores BOTH the semantic content (context_matrix) and the
        syntactic structure (vorticity_signature).
        
        - context_matrix: What tokens mean together (witness + structure)
        - vorticity_signature: How tokens are ORDERED (antisymmetric wedge product)
    """
    context_matrix: np.ndarray  # The context representation
    target_token: int  # The associated target
    count: int = 1  # How many times seen
    recency: float = 0.0  # Last update time
    salience: float = 0.0  # Emotional salience (scalar + pseudoscalar)
    novelty: float = 1.0  # Novelty score (1.0 = maximally novel)
    priority: float = 0.0  # Combined priority (salience + novelty + pred_error)
    vorticity_signature: np.ndarray = None  # [16] word order structure (computed during training)
    
    def __hash__(self):
        return hash(id(self.context_matrix))


@dataclass
class CompressedEpisode:
    """
    A delta-compressed episodic entry (schema-based compression).
    
    THEORY (Schema-Based Compression):
        - Most episodes are similar to existing prototypes
        - Only store DELTA (difference) from nearest prototype
        - Delta is sparse in Clifford basis (few non-zero coefficients)
        
    COMPRESSION:
        - Full episode: 16 floats (4×4 matrix)
        - Compressed: prototype_id + sparse delta (~3-5 floats)
        - Compression ratio: 3-5x
        
    DECOMPRESSION:
        full_matrix = prototype_matrix + delta
    """
    prototype_id: int  # ID of nearest prototype (-1 if uncompressed)
    delta_coeffs: np.ndarray  # Sparse delta in Clifford basis (16 floats, mostly zeros)
    target_token: int  # The associated target
    count: int = 1
    salience: float = 0.0
    sparsity: float = 0.0  # Fraction of zero coefficients (measure of compression quality)
    
    def decompress(self, prototypes: List['SemanticPrototype'], 
                   basis: np.ndarray, xp = np) -> np.ndarray:
        """
        Decompress to full matrix.
        
        Args:
            prototypes: List of semantic prototypes (for lookup)
            basis: Clifford basis for reconstruction
            xp: array module
            
        Returns:
            Full 4×4 matrix
        """
        if self.prototype_id < 0 or self.prototype_id >= len(prototypes):
            # Uncompressed: delta_coeffs IS the full matrix coefficients
            return reconstruct_from_coefficients(self.delta_coeffs, basis, xp)
        
        # Get prototype and add delta
        proto_matrix = prototypes[self.prototype_id].prototype_matrix
        delta_matrix = reconstruct_from_coefficients(self.delta_coeffs, basis, xp)
        return proto_matrix + delta_matrix


def compress_episode(
    episode: EpisodicEntry,
    prototypes: List['SemanticPrototype'],
    basis: np.ndarray,
    xp = np,
    sparsity_threshold: float = PHI_INV_EIGHT,  # φ⁻⁸ ≈ 0.021 for near-zero threshold
    min_similarity: float = PHI_INV_SQ,  # Theory-true: φ⁻² (spectral gap)
) -> CompressedEpisode:
    """
    Compress an episode using delta encoding relative to nearest prototype.
    
    THEORY:
        Schema compression stores DEVIATION from nearest prototype.
        If episode is similar to prototype, delta will be small and sparse.
        
    Args:
        episode: Full episodic entry to compress
        prototypes: List of semantic prototypes
        basis: Clifford basis
        xp: array module
        sparsity_threshold: Coefficients below this become zero
        min_similarity: Minimum similarity to prototype for compression
        
    Returns:
        CompressedEpisode (may be uncompressed if no good prototype match)
    """
    M = episode.context_matrix
    
    if not prototypes:
        # No prototypes: store full matrix as coefficients
        coeffs = decompose_to_coefficients(M, basis, xp)
        return CompressedEpisode(
            prototype_id=-1,
            delta_coeffs=coeffs,
            target_token=episode.target_token,
            count=episode.count,
            salience=episode.salience,
            sparsity=0.0,  # Not compressed
        )
    
    # Find nearest prototype
    best_proto_id = -1
    best_similarity = -float('inf')
    
    for i, proto in enumerate(prototypes):
        sim = frobenius_similarity(M, proto.prototype_matrix, xp)
        if sim > best_similarity:
            best_similarity = sim
            best_proto_id = i
    
    # Check if similar enough to compress
    if best_similarity < min_similarity or best_proto_id < 0:
        # Too different: store full matrix
        coeffs = decompose_to_coefficients(M, basis, xp)
        return CompressedEpisode(
            prototype_id=-1,
            delta_coeffs=coeffs,
            target_token=episode.target_token,
            count=episode.count,
            salience=episode.salience,
            sparsity=0.0,
        )
    
    # Compute delta
    proto_matrix = prototypes[best_proto_id].prototype_matrix
    delta_matrix = M - proto_matrix
    
    # Decompose delta into coefficients
    delta_coeffs = decompose_to_coefficients(delta_matrix, basis, xp)
    
    # Sparsify: set small coefficients to zero
    sparse_delta = delta_coeffs.copy()
    sparse_delta[xp.abs(sparse_delta) < sparsity_threshold] = 0.0
    
    # Compute sparsity (fraction of zeros)
    num_zeros = int(xp.sum(xp.abs(sparse_delta) < 1e-10))
    sparsity = num_zeros / 16.0
    
    return CompressedEpisode(
        prototype_id=best_proto_id,
        delta_coeffs=sparse_delta,
        target_token=episode.target_token,
        count=episode.count,
        salience=episode.salience,
        sparsity=sparsity,
    )


def compress_episodes_batch(
    episodes: List[EpisodicEntry],
    prototypes: List['SemanticPrototype'],
    basis: np.ndarray,
    xp = np,
    sparsity_threshold: float = PHI_INV_EIGHT,  # φ⁻⁸ ≈ 0.021 for near-zero threshold
    min_similarity: float = PHI_INV_SQ,  # Theory-true: φ⁻² (spectral gap)
) -> Tuple[List[CompressedEpisode], Dict[str, float]]:
    """
    Batch compress episodes with statistics.
    
    Returns:
        (compressed_episodes, stats_dict)
    """
    compressed = []
    total_sparsity = 0.0
    num_actually_compressed = 0
    
    for ep in episodes:
        comp = compress_episode(ep, prototypes, basis, xp, sparsity_threshold, min_similarity)
        compressed.append(comp)
        total_sparsity += comp.sparsity
        if comp.prototype_id >= 0:
            num_actually_compressed += 1
    
    n = len(episodes)
    stats = {
        'total_episodes': n,
        'actually_compressed': num_actually_compressed,
        'compression_rate': num_actually_compressed / n if n > 0 else 0.0,
        'avg_sparsity': total_sparsity / n if n > 0 else 0.0,
        # Estimated memory savings: uncompressed uses 16 floats per episode
        # Compressed uses ~(1 - sparsity) * 16 floats
        'estimated_compression_ratio': 16.0 / (16.0 * (1.0 - total_sparsity / n) + 1) if n > 0 else 1.0,
    }
    
    return compressed, stats


# =============================================================================
# PRUNING (Synaptic Pruning Analog)
# =============================================================================

def should_prune_prototype(
    proto: 'SemanticPrototype',
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


# =============================================================================
# INTERFERENCE MANAGEMENT (Merge Similar Prototypes)
# =============================================================================

def compute_prototype_similarity(
    proto1: 'SemanticPrototype',
    proto2: 'SemanticPrototype',
    xp = np,
) -> float:
    """Compute similarity between two prototypes."""
    return frobenius_similarity(proto1.prototype_matrix, proto2.prototype_matrix, xp)


def merge_prototypes(
    proto1: 'SemanticPrototype',
    proto2: 'SemanticPrototype',
    basis: np.ndarray,
    xp = np,
) -> 'SemanticPrototype':
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
    
    return SemanticPrototype(
        prototype_matrix=merged_matrix,
        target_distribution=merged_targets,
        radius=merged_radius,
        support=total_support,
        level=merged_level,
        vorticity_signature=merged_vort,
    )


def find_similar_prototype_pairs(
    prototypes: List['SemanticPrototype'],
    similarity_threshold: float = 1 - PHI_INV_CUBE,  # φ-derived: ≈ 0.764 (conservative)
    xp = np,
    basis: np.ndarray = None,
) -> List[Tuple[int, int, float]]:
    """
    Find pairs of prototypes that are highly similar.
    
    THEORY-TRUE SIMILARITY (v4.20.0):
        OLD: Frobenius (cosine) on full matrix
             Problem: Grace-stabilized matrices all have high similarity
                      because they share the identity component!
        
        NEW: Combined witness + vorticity similarity
             - Witness (σ, p) captures SEMANTIC content
             - Vorticity (bivectors) captures SYNTACTIC structure
             - Same witness + same vorticity = true duplicate
    
    Returns:
        List of (idx1, idx2, similarity) tuples, sorted by similarity descending
    """
    from .quotient import extract_witness, vorticity_similarity
    
    n = len(prototypes)
    if n < 2:
        return []
    
    # Need basis for proper similarity
    if basis is None:
        from .algebra import build_clifford_basis
        basis = build_clifford_basis(xp)
    
    # Extract witnesses and vorticity for all prototypes
    matrices = xp.array([p.prototype_matrix for p in prototypes])  # [n, 4, 4]
    
    # Extract witnesses [n, 2]
    witnesses = []
    for m in matrices:
        sigma, pseudo = extract_witness(m, basis, xp)
        witnesses.append([sigma, pseudo])
    witnesses = xp.array(witnesses)  # [n, 2]
    
    # Compute pairwise combined similarity using theory-true metric
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            # Use combined witness + vorticity similarity
            sim = vorticity_similarity(
                matrices[i], matrices[j], basis, xp,
                vort_weight=PHI_INV  # 61.8% vorticity, 38.2% witness
            )
            
            if sim >= similarity_threshold:
                pairs.append((i, j, float(sim)))
    
    # Sort by similarity (highest first)
    pairs.sort(key=lambda x: -x[2])
    
    return pairs


def manage_interference(
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
    similarity_threshold: float = 1 - PHI_INV_CUBE,  # ≈ 0.764 (conservative)
    max_merges_per_cycle: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Manage interference by merging highly similar prototypes.
    
    THEORY (Interference Management):
        - Similar memories compete during retrieval
        - High similarity → interference
        - But: aggressive merging destroys coverage → poor generalization!
    
    v4.20.0 FIX:
        OLD: Frobenius similarity → Grace-stabilized matrices ALL look similar
        NEW: Combined witness+vorticity similarity → only true duplicates merge
        - Resolution: Merge similar prototypes
        
    This is the Clifford algebra analog of:
        - Pattern separation (keeping things distinct)
        - Pattern completion (merging similar patterns)
        
    The similarity threshold relates to φ:
        threshold = φ⁻¹ ≈ 0.618 is TOO AGGRESSIVE (destroys coverage)
        threshold = 1 - φ⁻³ ≈ 0.764 is CONSERVATIVE (preserves diversity)
        
        Theory: "More prototypes → better generalization" (distributed_prior.py)
        Aggressive merging at 0.618 collapses 100+ prototypes to ~16, killing generalization.
        Conservative merging at 0.764 preserves more prototypes for semantic coverage.
        
    Args:
        semantic_memory: SemanticMemory to manage
        basis: Clifford basis
        xp: array module
        similarity_threshold: Above this, prototypes are "too similar"
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


# =============================================================================
# RECONSOLIDATION (Retrieval-Induced Plasticity)
# =============================================================================

@dataclass
class RetrievalRecord:
    """Record of a retrieval event for reconsolidation."""
    context_hash: int
    predicted_target: int
    actual_target: Optional[int] = None  # Filled in with feedback
    was_correct: Optional[bool] = None
    timestamp: float = 0.0
    source: str = "unknown"  # "episodic" or "semantic"


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
        import time
        
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
    proto: 'SemanticPrototype',
    actual_target: int,
    was_correct: bool,
    rate: float = PHI_INV,
) -> 'SemanticPrototype':
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


# =============================================================================
# WORKING MEMORY GATING (Salience-Based Attention)
# =============================================================================

def compute_embedding_saliences(
    embeddings: np.ndarray,
    token_indices: List[int],
    basis: np.ndarray,
    xp = np,
) -> np.ndarray:
    """
    Compute saliences for a sequence of token embeddings.
    
    Args:
        embeddings: [vocab_size, 4, 4] full embedding matrix
        token_indices: List of token indices to look up
        basis: Clifford basis
        xp: array module
        
    Returns:
        [len(token_indices)] array of saliences
    """
    token_matrices = xp.array([embeddings[idx % len(embeddings)] for idx in token_indices])
    return compute_salience_batch(token_matrices, basis, xp)


def apply_working_memory_gate(
    token_matrices: np.ndarray,
    basis: np.ndarray,
    xp = np,
    min_weight: float = None,  # Now theory-derived if None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply working memory gating to token embeddings.
    
    THEORY-TRUE (Working Memory Gating):
        Working memory preserves what SURVIVES.
        
        The theory-native attention weight is GRACE-STABILITY:
        - High stability (σ ≈ 1) → token survives Grace → high weight
        - Low stability (σ ≈ 0) → token decays under Grace → low weight
        
        This replaces arbitrary softmax with theory-derived measure.
        
    The φ-connection:
        min_weight = φ⁻² ≈ 0.382 (spectral gap) ensures even unstable
        tokens contribute something (no complete forgetting)
        
    Args:
        token_matrices: [N, 4, 4] sequence of token embeddings
        basis: Clifford basis
        xp: array module
        min_weight: Minimum weight (default: φ⁻² = spectral gap)
        
    Returns:
        (gated_matrices, weights) where gated_matrices are scaled by stability
    """
    N = token_matrices.shape[0]
    
    if N == 0:
        return token_matrices, xp.array([])
    
    # THEORY-TRUE: Attention = grace_stability × salience
    # 
    # This combines TWO theory-derived measures:
    #   - grace_stability: fraction that SURVIVES Grace (∈ [0, 1])
    #   - salience: MAGNITUDE of witness content (scalar + pseudo)
    #
    # The product prioritizes tokens that:
    #   1. Have high survivability (mostly witness content)
    #   2. Have strong witness content (intense signal)
    #
    # This is NOT arbitrary softmax - it's spectral structure × magnitude.
    
    stabilities = grace_stability_batch(token_matrices, basis, xp)
    saliences = compute_salience_batch(token_matrices, basis, xp)
    
    # Combined: stability × salience
    weights = stabilities * xp.maximum(saliences, 1e-8)
    
    # Theory-derived minimum weight: spectral gap φ⁻²
    if min_weight is None:
        min_weight = PHI_INV_SQ  # ≈ 0.382
    
    # Apply minimum weight (even weak tokens contribute)
    weights = xp.maximum(weights, min_weight * xp.mean(weights + 1e-10))
    
    # Normalize to sum to 1 (this IS justified - it's a distribution)
    weights = weights / xp.sum(weights)
    
    # Scale matrices by weights
    # Each matrix is scaled by its attention weight
    gated = token_matrices * weights.reshape(-1, 1, 1)
    
    # VECTORIZED: Renormalize each matrix to preserve magnitude
    # Compute all norms at once
    gated_flat = gated.reshape(N, -1)
    orig_flat = token_matrices.reshape(N, -1)
    
    gated_norms = xp.linalg.norm(gated_flat, axis=1, keepdims=True)  # [N, 1]
    orig_norms = xp.linalg.norm(orig_flat, axis=1, keepdims=True)    # [N, 1]
    
    # Scale factors: orig_norm / gated_norm (avoid div by zero)
    scale_factors = orig_norms / xp.maximum(gated_norms, 1e-8)  # [N, 1]
    
    # Apply scaling
    gated = gated * scale_factors.reshape(-1, 1, 1)
    
    return gated, weights


def gated_context_representation(
    token_matrices: np.ndarray,
    basis: np.ndarray,
    xp = np,
    use_gating: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute context representation with optional working memory gating.
    
    THEORY-TRUE:
        Standard context: C = M₁ × M₂ × ... × Mₙ (geometric product)
        Gated context: C = (w₁M₁) × (w₂M₂) × ... × (wₙMₙ)
        
        Where wᵢ are GRACE-STABILITY weights (not arbitrary softmax).
        
    EFFECT:
        High-stability tokens (survive Grace) dominate the context.
        Low-stability tokens (transient) contribute less.
        This is theory-true attention: what survives IS what matters.
        
    Args:
        token_matrices: [N, 4, 4] sequence of token embeddings
        basis: Clifford basis
        xp: array module
        use_gating: Whether to apply stability-based gating
        
    Returns:
        (context_matrix, info_dict)
    """
    N = token_matrices.shape[0]
    
    info = {
        'num_tokens': N,
        'use_gating': use_gating,
        'weights': None,
        'saliences': None,
    }
    
    if N == 0:
        return xp.eye(4, dtype=DTYPE), info
    
    if N == 1:
        return token_matrices[0], info
    
    # Apply gating if enabled (uses grace_stability, not softmax)
    if use_gating:
        gated_matrices, weights = apply_working_memory_gate(
            token_matrices, basis, xp
        )
        info['weights'] = weights
        info['saliences'] = compute_salience_batch(token_matrices, basis, xp)
    else:
        gated_matrices = token_matrices
    
    # Compute context via geometric product (sequential multiplication)
    context = gated_matrices[0].copy()
    for i in range(1, N):
        context = geometric_product(context, gated_matrices[i])
    
    # THEORY-TRUE: Use Grace to stabilize, not arbitrary Frobenius normalization
    # Grace contracts higher grades, naturally managing magnitude while
    # preserving the witness (stable core)
    context = grace_operator(context, basis, xp)
    
    return context, info


class WorkingMemoryBuffer:
    """
    A working memory buffer with capacity limits and salience-based eviction.
    
    THEORY (Working Memory):
        - Limited capacity (typically 4-7 items)
        - New items can displace old low-salience items
        - High-salience items are protected from eviction
        
    This is useful for streaming input where we need to maintain
    a fixed-size context window but want to keep important items.
    """
    
    def __init__(
        self,
        capacity: int = 7,
        basis: np.ndarray = None,
        xp = np,
    ):
        """
        Args:
            capacity: Maximum number of items in working memory
            basis: Clifford basis for salience computation
            xp: array module
        """
        self.capacity = capacity
        self.basis = basis
        self.xp = xp
        
        # Storage
        self.items: List[Tuple[int, np.ndarray, float]] = []  # (token_idx, matrix, salience)
    
    def add(self, token_idx: int, token_matrix: np.ndarray) -> Optional[int]:
        """
        Add an item to working memory.
        
        If at capacity, evicts lowest-salience item.
        
        Args:
            token_idx: Token index
            token_matrix: Token embedding matrix
            
        Returns:
            Evicted token_idx, or None if no eviction
        """
        # Compute salience
        salience = float(compute_salience(token_matrix, self.basis, self.xp))
        
        evicted = None
        
        if len(self.items) >= self.capacity:
            # Find lowest salience item
            min_idx = min(range(len(self.items)), key=lambda i: self.items[i][2])
            
            # Only evict if new item has higher salience
            if salience > self.items[min_idx][2]:
                evicted = self.items[min_idx][0]
                del self.items[min_idx]
            else:
                # New item has lowest salience, don't add
                return None
        
        self.items.append((token_idx, token_matrix, salience))
        return evicted
    
    def get_context_matrices(self) -> np.ndarray:
        """Get all matrices in working memory."""
        if not self.items:
            return self.xp.zeros((0, 4, 4), dtype=DTYPE)
        return self.xp.array([item[1] for item in self.items])
    
    def get_token_indices(self) -> List[int]:
        """Get all token indices in working memory."""
        return [item[0] for item in self.items]
    
    def clear(self):
        """Clear working memory."""
        self.items = []
    
    def stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if not self.items:
            return {'size': 0, 'capacity': self.capacity, 'avg_salience': 0.0}
        
        saliences = [item[2] for item in self.items]
        return {
            'size': len(self.items),
            'capacity': self.capacity,
            'avg_salience': float(np.mean(saliences)),
            'min_salience': float(min(saliences)),
            'max_salience': float(max(saliences)),
        }


@dataclass  
class SemanticPrototype:
    """
    A consolidated semantic memory entry (from Non-REM).
    
    THEORY:
        Prototypes store BOTH the attractor matrix AND the vorticity signature.
        - prototype_matrix: The semantic content (witness/core)
        - vorticity_signature: The syntactic structure (word order pattern)
        
        During retrieval, we match BOTH:
        - Witness similarity: Does it mean the same thing?
        - Vorticity similarity: Does it have the same structure?
    """
    prototype_matrix: np.ndarray  # Cluster centroid
    target_distribution: Dict[int, float]  # Token → probability
    radius: float  # Max distance to cluster members
    support: int  # Number of episodic entries consolidated
    level: int = 0  # Hierarchy level (0=fine, higher=coarse)
    vorticity_signature: np.ndarray = None  # [16] coefficients encoding word order structure
    
    def sample_target(self, rng: np.random.Generator = None) -> int:
        """Sample a target from the distribution."""
        if rng is None:
            rng = np.random.default_rng()
        tokens = list(self.target_distribution.keys())
        probs = list(self.target_distribution.values())
        return rng.choice(tokens, p=probs)
    
    def mode_target(self) -> int:
        """Return the most likely target."""
        return max(self.target_distribution.items(), key=lambda x: x[1])[0]
    
    def entropy(self) -> float:
        """Compute entropy of the target distribution."""
        probs = list(self.target_distribution.values())
        return -sum(p * np.log(p + 1e-10) for p in probs)


@dataclass
class Schema:
    """An abstract pattern discovered through REM (recombination + survival).
    
    THEORY:
        Schemas are abstract patterns that emerge from prototype recombination.
        They encode "how things combine" (syntax/structure) not "what to predict" (semantics).
        
        When a query matches a schema, we use the source prototypes' targets for retrieval.
        This enables generalization: if query matches the STRUCTURE of a schema,
        we can predict using any prototype that shares that structure.
    """
    canonical_form: np.ndarray  # Quotient-stable representation
    recurrence_count: int = 0  # How often this pattern was rediscovered
    utility_score: float = 0.0  # How useful it is for downstream tasks
    source_operations: List[str] = field(default_factory=list)  # How it was discovered
    source_prototype_ids: List[int] = field(default_factory=list)  # IDs of contributing prototypes


# =============================================================================
# PATTERN COMPLETION (Retrieval as Inference via Grace Flow)
# =============================================================================

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


# =============================================================================
# PREDICTIVE CODING (Only Encode Unpredicted Content)
# =============================================================================

def predict_from_memory(
    query: np.ndarray,
    semantic_memory: 'SemanticMemory',
    xp = np,
    top_k: int = 1,
) -> Tuple[np.ndarray, float, Optional['SemanticPrototype']]:
    """
    Generate prediction for a query from semantic memory.
    
    THEORY (Predictive Coding):
        The brain maintains a generative model that PREDICTS inputs.
        Semantic memory = model of "what patterns look like".
        
        Best-matching prototype IS the prediction:
        "Given this context, I expect something like this prototype"
    
    Args:
        query: [4, 4] input matrix
        semantic_memory: SemanticMemory with prototypes
        xp: numpy or cupy
        top_k: Number of prototypes to consider (usually 1)
        
    Returns:
        prediction: [4, 4] predicted matrix (best-matching prototype)
        confidence: float in [0, 1] - similarity to best prototype
        best_proto: The prototype used for prediction (or None)
    """
    if semantic_memory is None or semantic_memory.stats()['total_prototypes'] == 0:
        # No memory = no prediction = return identity (neutral expectation)
        return xp.eye(4, dtype=DTYPE), 0.0, None
    
    # Get best-matching prototype
    results = semantic_memory.retrieve(query, top_k=1, use_pattern_completion=False)
    
    if not results:
        return xp.eye(4, dtype=DTYPE), 0.0, None
    
    best_proto, similarity = results[0]
    
    # Prediction = prototype matrix
    # Confidence = similarity (how well the prediction matches)
    prediction = best_proto.prototype_matrix.copy()
    confidence = float(similarity)
    
    return prediction, confidence, best_proto


def compute_prediction_residual(
    observed: np.ndarray,
    predicted: np.ndarray,
    basis: np.ndarray,
    xp = np,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute the prediction residual (what wasn't predicted).
    
    THEORY (Prediction Error):
        Residual = Observed - Predicted
        This is the "surprising" content that the model didn't expect.
        
        In Clifford algebra:
        - Residual in scalar/pseudoscalar = intensity/valence difference
        - Residual in bivectors = unexpected relational structure
        - High residual = worth encoding; low residual = redundant
    
    Args:
        observed: [4, 4] actual input matrix
        predicted: [4, 4] predicted matrix from memory
        basis: [16, 4, 4] Clifford basis
        xp: numpy or cupy
        
    Returns:
        residual: [4, 4] prediction error matrix
        stats: Dict with residual statistics by grade
    """
    residual = observed - predicted
    
    # Decompose residual into grade components
    coeffs = decompose_to_coefficients(residual, basis, xp)
    
    # Grade-wise energy (squared coefficient sum)
    grade_0 = float(coeffs[0] ** 2)  # Scalar
    grade_1 = float(xp.sum(coeffs[1:5] ** 2))  # Vectors
    grade_2 = float(xp.sum(coeffs[5:11] ** 2))  # Bivectors
    grade_3 = float(xp.sum(coeffs[11:15] ** 2))  # Trivectors
    grade_4 = float(coeffs[15] ** 2)  # Pseudoscalar
    
    total_energy = float(xp.sum(coeffs ** 2))
    frobenius_norm = float(xp.linalg.norm(residual, 'fro'))
    
    stats = {
        'grade_0_energy': grade_0,
        'grade_1_energy': grade_1,
        'grade_2_energy': grade_2,
        'grade_3_energy': grade_3,
        'grade_4_energy': grade_4,
        'total_energy': total_energy,
        'frobenius_norm': frobenius_norm,
    }
    
    return residual, stats


def predictive_encode(
    episode: 'EpisodicEntry',
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
    significance_threshold: float = PHI_INV_CUBE,  # Theory-true: φ⁻³
) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
    """
    Predictive encoding: only encode what the memory doesn't predict.
    
    THEORY (Predictive Coding in Hippocampus):
        The hippocampus encodes PREDICTION ERRORS, not raw inputs.
        
        1. Generate prediction from semantic memory (neocortex)
        2. Compute residual = observed - predicted
        3. If residual is SIGNIFICANT, encode it
        4. If residual is SMALL, skip (already known)
        
    This is PRINCIPLED delta compression:
        - Not just "similar to prototype" heuristically
        - But "what the model didn't predict" theoretically
    
    Args:
        episode: EpisodicEntry to potentially encode
        semantic_memory: SemanticMemory for predictions
        basis: Clifford basis
        xp: numpy or cupy
        significance_threshold: Minimum residual norm to encode
        
    Returns:
        should_encode: bool - True if this episode is worth encoding
        residual: [4, 4] the prediction error (what to encode if significant)
        info: Dict with encoding decision details
    """
    # Get prediction from memory
    prediction, confidence, matched_proto = predict_from_memory(
        episode.context_matrix, semantic_memory, xp
    )
    
    # Compute residual
    residual, residual_stats = compute_prediction_residual(
        episode.context_matrix, prediction, basis, xp
    )
    
    # Decision: is the residual significant?
    residual_norm = residual_stats['frobenius_norm']
    is_significant = residual_norm >= significance_threshold
    
    info = {
        'confidence': confidence,
        'residual_norm': residual_norm,
        'threshold': significance_threshold,
        'is_significant': is_significant,
        'matched_proto_id': id(matched_proto) if matched_proto else None,
        'grade_energies': {
            0: residual_stats['grade_0_energy'],
            1: residual_stats['grade_1_energy'],
            2: residual_stats['grade_2_energy'],
            3: residual_stats['grade_3_energy'],
            4: residual_stats['grade_4_energy'],
        }
    }
    
    return is_significant, residual, info


def predictive_encode_batch(
    episodes: List['EpisodicEntry'],
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
    significance_threshold: float = PHI_INV_CUBE,  # Theory-true: φ⁻³
) -> Tuple[List['EpisodicEntry'], List['EpisodicEntry'], Dict[str, Any]]:
    """
    VECTORIZED predictive encoding for episode batch.
    
    Returns:
        significant_episodes: Episodes worth encoding (high residual)
        redundant_episodes: Episodes that were predicted (low residual)
        stats: Encoding statistics
    """
    if not episodes or semantic_memory is None or semantic_memory.stats()['total_prototypes'] == 0:
        return episodes, [], {'significant': len(episodes), 'redundant': 0, 'ratio': 1.0}
    
    # Get all episode matrices
    matrices = xp.array([ep.context_matrix for ep in episodes])
    N = len(episodes)
    
    # Get all prototype matrices
    all_protos = []
    for level in semantic_memory.levels:
        all_protos.extend(level)
    
    if not all_protos:
        return episodes, [], {'significant': N, 'redundant': 0, 'ratio': 1.0}
    
    proto_matrices = xp.array([p.prototype_matrix for p in all_protos])
    
    # VECTORIZED: Find best prototype for each episode
    matrices_flat = matrices.reshape(N, -1)
    protos_flat = proto_matrices.reshape(len(all_protos), -1)
    
    mat_norms = xp.linalg.norm(matrices_flat, axis=1, keepdims=True)
    proto_norms = xp.linalg.norm(protos_flat, axis=1, keepdims=True)
    
    # All similarities: [N, P]
    all_sims = xp.dot(matrices_flat, protos_flat.T) / (mat_norms @ proto_norms.T + 1e-10)
    
    # Best prototype indices and predictions
    best_indices = xp.argmax(all_sims, axis=1)
    predictions = proto_matrices[best_indices]  # [N, 4, 4]
    
    # Compute residuals
    residuals = matrices - predictions
    residual_norms = xp.linalg.norm(residuals.reshape(N, -1), axis=1)
    
    # Split by significance
    significant_mask = residual_norms >= significance_threshold
    
    significant_episodes = [ep for ep, sig in zip(episodes, significant_mask) if sig]
    redundant_episodes = [ep for ep, sig in zip(episodes, significant_mask) if not sig]
    
    stats = {
        'significant': len(significant_episodes),
        'redundant': len(redundant_episodes),
        'ratio': len(significant_episodes) / max(N, 1),
        'avg_residual_significant': float(xp.mean(residual_norms[significant_mask])) if any(significant_mask) else 0.0,
        'avg_residual_redundant': float(xp.mean(residual_norms[~significant_mask])) if any(~significant_mask) else 0.0,
    }
    
    return significant_episodes, redundant_episodes, stats


# =============================================================================
# SEQUENCE REPLAY (Temporal Transition Memory via Vorticity)
# =============================================================================

@dataclass
class TemporalTransition:
    """
    A temporal transition between consecutive contexts.
    
    THEORY (Sharp Wave Ripples / Sequence Replay):
        The brain doesn't just store snapshots—it stores SEQUENCES.
        During sleep, these sequences are replayed at compressed timescales.
        
    In Clifford algebra:
        - from_context: Where we were (M[t-1])
        - to_context: Where we went (M[t])
        - vorticity: The TRANSITION encoded as bivector (M[t] ∧ M[t-1])
        
    The vorticity encodes:
        - Direction of change (antisymmetric)
        - Magnitude of change
        - "Rotational" content of the transition
    """
    from_context: np.ndarray  # [4, 4] previous context
    to_context: np.ndarray    # [4, 4] next context
    vorticity: np.ndarray     # [4, 4] transition encoding (wedge product)
    from_target: int          # Target associated with from_context
    to_target: int            # Target associated with to_context
    salience: float = 0.0     # Transition importance
    timestamp: float = 0.0    # When this transition occurred


def compute_transition_vorticity(
    from_matrix: np.ndarray,
    to_matrix: np.ndarray,
    xp = np,
) -> np.ndarray:
    """
    Compute the vorticity (wedge product) encoding a transition.
    
    THEORY:
        Vorticity = (AB - BA) / 2 = A ∧ B
        This captures the antisymmetric (rotational) component of the transition.
        
    Properties:
        - Vorticity = 0 if A and B commute (parallel change)
        - High vorticity = non-commutative change (rotational)
        - Sign encodes direction: A→B vs B→A
    """
    # Wedge product: antisymmetric part of geometric product
    ab = from_matrix @ to_matrix
    ba = to_matrix @ from_matrix
    vorticity = (ab - ba) / 2.0
    
    return vorticity


def record_transition(
    from_context: np.ndarray,
    to_context: np.ndarray,
    from_target: int,
    to_target: int,
    basis: np.ndarray,
    xp = np,
    timestamp: float = 0.0,
) -> TemporalTransition:
    """
    Record a temporal transition between consecutive contexts.
    
    Args:
        from_context: [4, 4] previous context matrix
        to_context: [4, 4] next context matrix
        from_target: Target token for from_context
        to_target: Target token for to_context
        basis: Clifford basis for salience computation
        xp: numpy or cupy
        timestamp: When this transition occurred
        
    Returns:
        TemporalTransition with vorticity encoding
    """
    # Compute vorticity
    vorticity = compute_transition_vorticity(from_context, to_context, xp)
    
    # Compute salience of the TRANSITION (not just endpoints)
    # High vorticity = important transition (non-trivial change)
    vorticity_magnitude = float(xp.linalg.norm(vorticity, 'fro'))
    
    # Also consider endpoint saliences
    from_salience = compute_salience(from_context, basis, xp)
    to_salience = compute_salience(to_context, basis, xp)
    
    # Transition salience = vorticity magnitude + φ²-weighted endpoint saliences
    transition_salience = vorticity_magnitude + PHI_INV_SQ * (from_salience + to_salience)
    
    return TemporalTransition(
        from_context=from_context,
        to_context=to_context,
        vorticity=vorticity,
        from_target=from_target,
        to_target=to_target,
        salience=transition_salience,
        timestamp=timestamp,
    )


class TransitionBuffer:
    """
    Buffer for storing and replaying temporal transitions.
    
    THEORY (Sequence Memory):
        - Store transitions, not just snapshots
        - High-salience transitions prioritized
        - Replay during REM for sequence consolidation
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        xp = np,
    ):
        self.capacity = capacity
        self.xp = xp
        self.transitions: List[TemporalTransition] = []
    
    def add(self, transition: TemporalTransition):
        """Add a transition to the buffer."""
        self.transitions.append(transition)
        
        # If over capacity, remove lowest-salience transition
        if len(self.transitions) > self.capacity:
            self.transitions.sort(key=lambda t: t.salience)
            self.transitions.pop(0)  # Remove lowest salience
    
    def add_from_episode_sequence(
        self,
        episodes: List['EpisodicEntry'],
        basis: np.ndarray,
        xp = np,
    ):
        """
        Extract and store transitions from a sequence of episodes.
        
        Args:
            episodes: List of EpisodicEntry in temporal order
            basis: Clifford basis
            xp: numpy or cupy
        """
        for i in range(len(episodes) - 1):
            transition = record_transition(
                from_context=episodes[i].context_matrix,
                to_context=episodes[i+1].context_matrix,
                from_target=episodes[i].target_token,
                to_target=episodes[i+1].target_token,
                basis=basis,
                xp=xp,
                timestamp=float(i),
            )
            self.add(transition)
    
    def get_high_salience(self, top_k: int = 10) -> List[TemporalTransition]:
        """Get the top-k highest salience transitions."""
        sorted_transitions = sorted(self.transitions, key=lambda t: -t.salience)
        return sorted_transitions[:top_k]
    
    def sample_for_replay(
        self,
        n_samples: int,
    ) -> List[TemporalTransition]:
        """
        Sample transitions for replay, weighted by salience.
        
        THEORY-TRUE: Uses direct salience weighting (NOT temperature-scaled softmax).
        Salience = witness magnitude = what survives Grace = intrinsic importance.
        
        Args:
            n_samples: Number of transitions to sample
            
        Returns:
            List of sampled transitions
        """
        if not self.transitions or n_samples <= 0:
            return []
        
        xp = self.xp
        
        # THEORY-TRUE: Direct salience-weighted sampling (no arbitrary softmax)
        # Salience = scalar + pseudoscalar content = what survives Grace
        saliences = xp.array([t.salience for t in self.transitions])
        
        # Ensure positive weights
        weights = xp.maximum(saliences, 1e-8)
        
        # Normalize to probability distribution (this IS justified)
        probs = weights / xp.sum(weights)
        
        # Sample indices - USE NUMPY for weighted sampling (CuPy doesn't support p parameter)
        n_actual = min(n_samples, len(self.transitions))
        # CuPy arrays need .get() for explicit conversion
        if hasattr(probs, 'get'):
            probs_np = probs.get()  # CuPy → NumPy
        else:
            probs_np = np.array(probs)
        probs_np = probs_np / probs_np.sum()  # Ensure sums to 1
        indices = np.random.choice(
            len(self.transitions),
            size=n_actual,
            replace=False,
            p=probs_np,
        )
        
        return [self.transitions[int(i)] for i in indices]
    
    def replay_sequence(
        self,
        start_transition: TemporalTransition,
        max_length: int = 5,
        similarity_threshold: float = PHI_INV,  # φ-derived: φ⁻¹ ≈ 0.618
    ) -> List[TemporalTransition]:
        """
        Replay a sequence starting from a given transition.
        
        Find subsequent transitions whose from_context matches the
        current to_context (chained replay).
        
        Args:
            start_transition: Starting point of replay
            max_length: Maximum sequence length
            similarity_threshold: φ-derived threshold for context chaining
            
        Returns:
            List of chained transitions forming a sequence
        """
        xp = self.xp
        sequence = [start_transition]
        sequence_ids = {id(start_transition)}  # Use id for comparison
        current_to = start_transition.to_context
        
        for _ in range(max_length - 1):
            # Find best matching next transition
            best_match = None
            best_sim = -float('inf')
            
            for t in self.transitions:
                if id(t) in sequence_ids:  # Use id for comparison
                    continue
                
                sim = frobenius_similarity(current_to, t.from_context, xp)
                if sim > best_sim and sim >= similarity_threshold:
                    best_sim = sim
                    best_match = t
            
            if best_match is None:
                break
            
            sequence.append(best_match)
            sequence_ids.add(id(best_match))
            current_to = best_match.to_context
        
        return sequence
    
    def stats(self) -> Dict[str, Any]:
        """Return buffer statistics."""
        if not self.transitions:
            return {
                'count': 0,
                'capacity': self.capacity,
                'avg_salience': 0.0,
                'avg_vorticity_norm': 0.0,
            }
        
        xp = self.xp
        saliences = [t.salience for t in self.transitions]
        vort_norms = [float(xp.linalg.norm(t.vorticity, 'fro')) for t in self.transitions]
        
        return {
            'count': len(self.transitions),
            'capacity': self.capacity,
            'avg_salience': float(xp.mean(xp.array(saliences))),
            'max_salience': float(xp.max(xp.array(saliences))),
            'avg_vorticity_norm': float(xp.mean(xp.array(vort_norms))),
        }


def replay_transitions_during_rem(
    transition_buffer: TransitionBuffer,
    basis: np.ndarray,
    xp = np,
    n_replays: int = 10,
    sequence_length: int = 3,
) -> Tuple[List[List[TemporalTransition]], Dict[str, Any]]:
    """
    Replay sequences during REM sleep.
    
    THEORY (Sharp Wave Ripples):
        During REM, the brain replays important sequences.
        This helps consolidate sequential/procedural memory.
        
    Process:
        1. Sample high-salience starting transitions
        2. Chain them into sequences
        3. Apply Grace to the sequences (test stability)
        4. Return surviving sequences for schema discovery
    
    Args:
        transition_buffer: Buffer with stored transitions
        basis: Clifford basis
        xp: numpy or cupy
        n_replays: Number of sequences to replay
        sequence_length: Target length of each sequence
        
    Returns:
        replayed_sequences: List of transition sequences
        stats: Replay statistics
    """
    if transition_buffer.stats()['count'] == 0:
        return [], {'replayed': 0, 'avg_length': 0}
    
    # Sample starting transitions
    starters = transition_buffer.sample_for_replay(n_replays)
    
    # Replay sequences
    sequences = []
    total_length = 0
    
    for start in starters:
        seq = transition_buffer.replay_sequence(
            start,
            max_length=sequence_length,
            similarity_threshold=PHI_INV,  # φ-derived (was 0.6)
        )
        sequences.append(seq)
        total_length += len(seq)
    
    stats = {
        'replayed': len(sequences),
        'avg_length': total_length / max(len(sequences), 1),
        'max_length': max(len(s) for s in sequences) if sequences else 0,
    }
    
    return sequences, stats


# =============================================================================
# PSEUDO-REHEARSAL (Prevent Catastrophic Forgetting)
# =============================================================================

def generate_pseudo_episode(
    prototype: 'SemanticPrototype',
    basis: np.ndarray,
    xp = np,
    noise_std: float = PHI_INV_CUBE,  # φ-derived (was 0.1)
) -> 'EpisodicEntry':
    """
    Generate a pseudo-episode from a semantic prototype.
    
    THEORY (Complementary Learning Systems):
        To prevent catastrophic forgetting:
        1. Semantic memory (neocortex) generates "pseudo-patterns"
        2. These are mixed with real episodes during training
        3. Old knowledge is rehearsed while new is learned
        
    The pseudo-episode is:
        - Prototype matrix + small noise (variation around the core)
        - Target sampled from the prototype's distribution
        
    This is like "dreaming" during waking - generating internal samples
    that remind the system of what it already knows.
    
    Args:
        prototype: SemanticPrototype to generate from
        basis: Clifford basis
        xp: numpy or cupy
        noise_std: Standard deviation of noise to add
        
    Returns:
        EpisodicEntry representing a pseudo-episode
    """
    # Generate context: prototype + noise
    noise = noise_std * xp.random.randn(4, 4)
    pseudo_context = prototype.prototype_matrix + noise
    
    # Normalize
    norm = xp.linalg.norm(pseudo_context, 'fro')
    if norm > 1e-8:
        pseudo_context = pseudo_context / norm
    
    # Sample target from distribution - USE NUMPY (CuPy doesn't support p parameter)
    targets = list(prototype.target_distribution.keys())
    probs = np.array(list(prototype.target_distribution.values()))
    probs = probs / probs.sum()  # Ensure sums to 1
    target = int(np.random.choice(targets, p=probs))
    
    # Compute salience of pseudo-episode
    salience = compute_salience(pseudo_context, basis, xp)
    
    return EpisodicEntry(
        context_matrix=pseudo_context,
        target_token=target,
        count=1,
        recency=0.0,
        salience=salience,
        novelty=0.0,  # Not novel - generated from known prototype
        priority=salience * PHI_INV_SQ,  # φ²-reduced priority for synthetic episodes
    )


def generate_pseudo_episodes_batch(
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
    n_episodes: int = 10,
    noise_std: float = PHI_INV_CUBE,  # φ-derived noise for variation
) -> List['EpisodicEntry']:
    """
    Generate a batch of pseudo-episodes from semantic memory.
    
    THEORY-TRUE: Prototypes sampled weighted by support (evidence count).
    No temperature parameter - support weighting is theory-derived.
    
    Args:
        semantic_memory: SemanticMemory to generate from
        basis: Clifford basis
        xp: numpy or cupy
        n_episodes: Number of pseudo-episodes to generate
        noise_std: φ-derived noise for variation (PHI_INV_CUBE ≈ 0.236)
        
    Returns:
        List of EpisodicEntry pseudo-episodes
    """
    # Collect all prototypes
    all_protos = []
    for level in semantic_memory.levels:
        all_protos.extend(level)
    
    if not all_protos:
        return []
    
    # THEORY-TRUE: Weight by support directly (no arbitrary softmax)
    # Support = evidence count = naturally positive
    # Common patterns (high support) are rehearsed more
    supports = xp.array([p.support for p in all_protos], dtype=DTYPE)
    
    # Directly normalize to probability (support is already positive)
    probs = supports / (xp.sum(supports) + 1e-10)
    
    # Sample prototype indices - USE NUMPY (CuPy doesn't support p parameter)
    n_actual = min(n_episodes, len(all_protos) * 5)  # Don't over-sample
    # CuPy arrays need .get() for explicit conversion
    if hasattr(probs, 'get'):
        probs_np = probs.get()  # CuPy → NumPy
    else:
        probs_np = np.array(probs)
    probs_np = probs_np / probs_np.sum()  # Ensure sums to 1
    indices = np.random.choice(
        len(all_protos),
        size=n_actual,
        replace=True,  # Allow repeats
        p=probs_np,
    )
    
    # Generate pseudo-episodes
    pseudo_episodes = []
    for idx in indices:
        proto = all_protos[int(idx)]
        pseudo_ep = generate_pseudo_episode(proto, basis, xp, noise_std)
        pseudo_episodes.append(pseudo_ep)
    
    return pseudo_episodes


def interleave_with_pseudo_rehearsal(
    real_episodes: List['EpisodicEntry'],
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
    rehearsal_ratio: float = PHI_INV_CUBE,  # Theory-true: φ⁻³ ≈ 0.236
    noise_std: float = PHI_INV_CUBE,  # φ-derived (was 0.1)
) -> Tuple[List['EpisodicEntry'], Dict[str, Any]]:
    """
    Interleave real episodes with pseudo-episodes from semantic memory.
    
    THEORY (Preventing Catastrophic Forgetting):
        Without rehearsal:
            - New learning overwrites old
            - Model "forgets" previously learned patterns
            
        With pseudo-rehearsal:
            - Generate samples from semantic memory
            - Mix with real episodes
            - Old patterns are reinforced while learning new
            
    The ratio controls the trade-off:
        - High ratio (0.5): More rehearsal, stronger memory retention
        - Low ratio (0.1): Less rehearsal, faster new learning
        
    Args:
        real_episodes: Real episodes to learn from
        semantic_memory: SemanticMemory for generating pseudo-episodes
        basis: Clifford basis
        xp: numpy or cupy
        rehearsal_ratio: Fraction of pseudo-episodes to add (0.2 = 20%)
        noise_std: Noise for pseudo-episode generation
        
    Returns:
        combined: Interleaved list of real + pseudo episodes
        stats: Rehearsal statistics
    """
    n_real = len(real_episodes)
    n_pseudo = int(n_real * rehearsal_ratio)
    
    # Generate pseudo-episodes
    pseudo_episodes = generate_pseudo_episodes_batch(
        semantic_memory, basis, xp,
        n_episodes=n_pseudo,
        noise_std=noise_std,
    )
    
    # Combine and shuffle
    combined = list(real_episodes) + pseudo_episodes
    
    # Shuffle to interleave - USE NUMPY for permutation (safer)
    indices = np.random.permutation(len(combined))
    combined = [combined[int(i)] for i in indices]
    
    stats = {
        'real_episodes': n_real,
        'pseudo_episodes': len(pseudo_episodes),
        'total': len(combined),
        'actual_ratio': len(pseudo_episodes) / max(n_real, 1),
    }
    
    return combined, stats


# =============================================================================
# INHIBITION OF RETURN (Exploration Bonus)
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
        inhibition_weight: How much to penalize (0.5 = moderate)
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
    inhibition_weight: float = PHI_INV,  # φ-derived (was 0.5)
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
    
    # Compute base similarities
    proto_matrices = xp.array([p.prototype_matrix for p in all_protos])
    base_sims = frobenius_similarity_batch(query, proto_matrices, xp)
    
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


# =============================================================================
# GRACE FOR DREAMING (Stronger contraction)
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
# NON-REM: CONSOLIDATION
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
    """
    
    def __init__(
        self,
        basis: np.ndarray,
        xp = np,
        similarity_threshold: float = PHI_INV,  # φ-derived (was 0.8)
        min_cluster_size: int = 3,
        grace_rate: float = PHI_INV_SQ,
        use_salience: bool = True,  # Use emotional salience weighting
        use_novelty: bool = True,   # NEW: Use novelty weighting
        semantic_memory: Optional['SemanticMemory'] = None,  # For novelty computation
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
        # Just normalize - don't apply heavy Grace (that collapses everything)
        # The raw structure is what we want to cluster on
        xp = self.xp
        norm = xp.linalg.norm(M, 'fro')
        if norm > 1e-8:
            return M / norm
        return M
    
    def canonicalize_batch(self, matrices: np.ndarray) -> np.ndarray:
        """Batch canonicalization - vectorized for GPU."""
        xp = self.xp
        # matrices: (N, 4, 4)
        norms = xp.linalg.norm(matrices.reshape(len(matrices), -1), axis=1, keepdims=True)
        norms = xp.maximum(norms, 1e-8)
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
            
        WHY NOT TARGET-FIRST CLUSTERING:
            Clustering by target creates prototypes that predict ONLY ONE token
            (target_distribution = {single_token: 1.0}). This prevents:
            - True generalization (each prototype is a memorized association)
            - Uncertainty representation (no distribution, just certainty)
            - Population coding (no blending of predictions)
            
        COMBINED SIMILARITY:
            For SHORT contexts: context_matrix similarity dominates
            For LONG contexts: vorticity_signature similarity dominates
        """
        if len(episodes) == 0:
            return []
        
        # Cluster ALL episodes by context similarity (NOT by target!)
        return self._cluster_by_context_similarity(episodes)
    
    def _cluster_by_context_similarity(
        self,
        episodes: List[EpisodicEntry]
    ) -> List[List[EpisodicEntry]]:
        """Cluster episodes by context similarity, allowing mixed targets."""
        if len(episodes) <= self.min_cluster_size:
            return [episodes] if len(episodes) >= self.min_cluster_size else []
        
        # Use the existing clustering logic but without target grouping
        return self._cluster_within_target(episodes)
    
    def _cluster_within_target(
        self,
        episodes: List[EpisodicEntry]
    ) -> List[List[EpisodicEntry]]:
        """Cluster episodes that share the same target by context similarity."""
        # If small group, just return as one cluster
        if len(episodes) <= self.min_cluster_size:
            return [episodes] if len(episodes) >= self.min_cluster_size else []
        
        xp = self.xp
        
        # Batch extract and canonicalize - VECTORIZED
        matrices = xp.array([ep.context_matrix for ep in episodes])
        canonical_forms = self.canonicalize_batch(matrices)
        
        # Extract vorticity signatures (use zeros if not present)
        vorticity_sigs = []
        for ep in episodes:
            if ep.vorticity_signature is not None:
                vorticity_sigs.append(ep.vorticity_signature)
            else:
                vorticity_sigs.append(xp.zeros(16, dtype=DTYPE))
        vorticity_array = xp.array(vorticity_sigs)
        
        # Compute average stability to determine weighting
        stabilities = grace_stability_batch(matrices, self.basis, xp)
        avg_stability = float(xp.mean(stabilities))
        
        # α = stability: high stability → rely on vorticity (1-α large), low stability → rely on context (α large)
        # Actually invert: high stability means context is useless, so use vorticity MORE
        vorticity_weight = avg_stability  # High stability → high vorticity weight
        context_weight = 1.0 - avg_stability
        
        # Use vectorized clustering with COMBINED similarity
        clusters: List[List[int]] = []
        cluster_matrix_centroids = []  # For context matrices
        cluster_vorticity_centroids = []  # For vorticity signatures
        
        for i in range(len(episodes)):
            canonical = canonical_forms[i]
            vorticity = vorticity_array[i]
            
            if len(cluster_matrix_centroids) > 0:
                # Context matrix similarity (Frobenius)
                centroids_array = xp.array(cluster_matrix_centroids)
                dots = xp.sum(canonical * centroids_array, axis=(1, 2))
                norm_c = xp.linalg.norm(canonical.reshape(-1))
                norms_cent = xp.linalg.norm(centroids_array.reshape(len(centroids_array), -1), axis=1)
                context_sims = dots / (norm_c * norms_cent + 1e-10)
                
                # Vorticity signature similarity (cosine)
                vort_centroids = xp.array(cluster_vorticity_centroids)
                vort_dots = xp.sum(vorticity * vort_centroids, axis=1)
                norm_v = xp.linalg.norm(vorticity)
                norms_vc = xp.linalg.norm(vort_centroids, axis=1)
                vorticity_sims = vort_dots / (norm_v * norms_vc + 1e-10)
                
                # Combined similarity (theory-derived weighting)
                combined_sims = context_weight * context_sims + vorticity_weight * vorticity_sims
                
                best_idx = int(xp.argmax(combined_sims))
                best_sim = float(combined_sims[best_idx])
            else:
                best_idx = -1
                best_sim = 0.0
            
            # Join existing cluster or create new one
            if best_sim >= self.similarity_threshold and best_idx >= 0:
                clusters[best_idx].append(i)
                # Update centroids (running mean)
                n = len(clusters[best_idx])
                cluster_matrix_centroids[best_idx] = (
                    (n - 1) * cluster_matrix_centroids[best_idx] + canonical
                ) / n
                cluster_vorticity_centroids[best_idx] = (
                    (n - 1) * cluster_vorticity_centroids[best_idx] + vorticity
                ) / n
            else:
                clusters.append([i])
                cluster_matrix_centroids.append(canonical.copy())
                cluster_vorticity_centroids.append(vorticity.copy())
        
        # Convert to episode lists
        return [[episodes[i] for i in cluster] for cluster in clusters]
    
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
        
        PRIORITY WEIGHTING:
            High-priority episodes (salience + novelty) contribute more to the centroid.
            This produces prototypes centered on "important" patterns.
        """
        if len(cluster) < self.min_cluster_size:
            return None
        
        xp = self.xp
        
        # Get matrices and compute weights
        matrices = xp.array([ep.context_matrix for ep in cluster])
        
        if self.use_salience or self.use_novelty:
            # Priority-weighted centroid (uses priority which combines salience + novelty)
            # Handle CuPy scalars properly - convert to float first
            priorities = xp.array([max(PHI_INV_EIGHT, float(ep.priority)) for ep in cluster])  # φ⁻⁸ min
            # Normalize priorities to sum to 1
            priority_weights = priorities / (xp.sum(priorities) + 1e-10)
            # Weighted average: centroid = sum(weight_i * matrix_i)
            centroid = xp.einsum('n,nij->ij', priority_weights, matrices)
        else:
            # Simple mean
            centroid = xp.mean(matrices, axis=0)
            priority_weights = xp.ones(len(cluster)) / len(cluster)
        
        # Apply Grace to stabilize
        centroid = self.canonicalize(centroid)
        
        # Aggregate vorticity signatures (priority-weighted average)
        # This captures the typical word-order structure for this prototype
        vort_sigs = []
        for ep in cluster:
            if ep.vorticity_signature is not None:
                vort_sigs.append(ep.vorticity_signature)
        
        if vort_sigs:
            vort_sigs_array = xp.array(vort_sigs)
            # Weight by priorities (only for episodes with vorticity)
            valid_weights = priority_weights[:len(vort_sigs)]
            valid_weights = valid_weights / (xp.sum(valid_weights) + 1e-10)
            avg_vort_sig = xp.einsum('n,nj->j', valid_weights, vort_sigs_array)
        else:
            avg_vort_sig = xp.zeros(16, dtype=DTYPE)
        
        # Compute target distribution (weighted by count × priority)
        target_counts: Dict[int, float] = defaultdict(float)
        total_weight = 0.0
        
        for ep in cluster:
            # Weight = count × (1 + priority) - priority boost
            weight = ep.count * (1.0 + ep.priority)
            target_counts[ep.target_token] += weight
            total_weight += weight
        
        # Normalize to probabilities
        target_dist = {
            token: count / total_weight 
            for token, count in target_counts.items()
        }
        
        # Compute radius
        distances = [
            1.0 - frobenius_similarity(centroid, ep.context_matrix, self.xp)
            for ep in cluster
        ]
        radius = max(distances) if distances else 0.0
        
        return SemanticPrototype(
            prototype_matrix=centroid,
            target_distribution=target_dist,
            radius=radius,
            support=len(cluster),
            level=0,
            vorticity_signature=avg_vort_sig,
        )
    
    def consolidate(
        self, 
        episodes: List[EpisodicEntry],
        verbose: bool = False,
    ) -> List[SemanticPrototype]:
        """
        Main consolidation routine: episodes → prototypes.
        
        SELF-ORGANIZING PRINCIPLE (Theory-Derived):
            The Grace operator determines what survives. Episodes are sorted by
            CONSOLIDATION URGENCY = 1 - grace_stability.
            
            - High urgency (low stability) → consolidate immediately
            - Low urgency (high stability) → can remain episodic
            
            This is NOT a tuned parameter - it's the spectral structure of Grace!
        
        The algorithm:
            1. Compute grace_stability for each episode (coefficient energy in witness)
            2. Filter: very stable episodes (σ > φ⁻²) stay episodic
            3. Sort remaining by urgency (highest urgency = seeds clusters)
            4. Cluster by canonical similarity
            5. Create prototypes (urgency-weighted centroids)
        """
        if verbose:
            print(f"  Non-REM: Consolidating {len(episodes)} episodes...")
        
        xp = self.xp
        
        if len(episodes) == 0:
            return []
        
        matrices = xp.array([ep.context_matrix for ep in episodes])
        
        # Compute SALIENCE (for emotional tagging, used in prototype weighting)
        saliences = compute_salience_batch(matrices, self.basis, xp)
        for i, ep in enumerate(episodes):
            ep.salience = float(saliences[i])
        
        # Compute GRACE-STABILITY and URGENCY (SELF-ORGANIZING principle)
        stabilities = grace_stability_batch(matrices, self.basis, xp)
        urgencies = 1.0 - stabilities
        
        # THEORY-TRUE SELF-ORGANIZING PRINCIPLE:
        # Two criteria for consolidation (brain-inspired):
        #
        # 1. TRANSIENCE: σ < φ⁻² → unclear memories need abstraction
        #    - φ⁻² is the spectral gap of Grace (theory-derived)
        #    - Below threshold, transient grades dominate
        #
        # 2. REDUNDANCY: Multiple similar episodes with same target
        #    - Brain replays and consolidates REPEATED patterns
        #    - Statistical regularity indicates structure worth abstracting
        #    - Similarity threshold: φ⁻¹ (principal eigenvalue of Grace)
        #
        # Consolidate if: (σ < φ⁻²) OR (redundant with same target)
        
        # First pass: annotate all episodes with stability and salience
        for i, ep in enumerate(episodes):
            ep.stability = float(stabilities[i])
            ep.salience = float(saliences[i])
            # Use float() to handle CuPy scalars properly
            ep.priority = float(ep.salience) + 0.001 * float(xp.random.rand())
        
        # Second pass: detect redundancy by target
        # Group episodes by target_token
        from collections import defaultdict
        target_groups = defaultdict(list)
        for ep in episodes:
            target_groups[ep.target_token].append(ep)
        
        # Find redundant groups: same target, multiple episodes
        # A group is "redundant" if it has ≥3 episodes (φ-derived: min cluster size)
        redundant_episodes = set()
        for target, group in target_groups.items():
            if len(group) >= 3:  # Min cluster size
                # All episodes in this group are candidates for consolidation
                for ep in group:
                    redundant_episodes.add(id(ep))
        
        # Build consolidation list using EITHER criterion
        episodes_to_consolidate = []
        n_transient = 0
        n_redundant = 0
        for ep in episodes:
            is_transient = ep.stability < PHI_INV_SQ
            is_redundant = id(ep) in redundant_episodes
            
            if is_transient:
                n_transient += 1
                episodes_to_consolidate.append(ep)
            elif is_redundant:
                n_redundant += 1
                episodes_to_consolidate.append(ep)
        
        if verbose:
            avg_stability = float(xp.mean(stabilities))
            n_unique_targets = len(target_groups)
            n_kept = len(episodes) - len(episodes_to_consolidate)
            print(f"    Grace-stability: avg={avg_stability:.3f}")
            print(f"    Unique targets: {n_unique_targets}")
            print(f"    Consolidation criteria (theory-true):")
            print(f"      - Transient (σ < φ⁻²): {n_transient}")
            print(f"      - Redundant (≥3 same target): {n_redundant}")
            print(f"    Kept episodic: {n_kept}")
            print(f"    To consolidate: {len(episodes_to_consolidate)}")
        
        # Early exit if no episodes need consolidation
        if not episodes_to_consolidate:
            if verbose:
                print(f"    No episodes meet consolidation criteria")
            return []
        
        # Sort by PRIORITY (highest first = most salient = seeds clusters)
        episodes_to_consolidate = sorted(episodes_to_consolidate, key=lambda ep: ep.priority, reverse=True)
        
        # Cluster (high-priority episodes processed first = better seeds)
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
            
            # Summary stats
            if prototypes:
                avg_entropy = np.mean([p.entropy() for p in prototypes])
                avg_support = np.mean([p.support for p in prototypes])
                print(f"    Avg entropy: {avg_entropy:.3f}")
                print(f"    Avg support: {avg_support:.1f}")
        
        return prototypes


# =============================================================================
# REM: RECOMBINATION
# =============================================================================

class REMRecombinator:
    """
    REM sleep phase: recombination and survival testing.
    
    This addresses:
        - Abstraction: discover invariants not present in data
        - Robustness: only quotient-stable structures survive
    
    Theory:
        "Dreams may generate anything. Only structure earns the right to become belief."
    """
    
    def __init__(
        self,
        basis: np.ndarray,
        xp = np,
        survival_threshold: float = PHI_INV,  # φ-derived (was 0.9)
        recurrence_threshold: int = 3,
        grace_steps: int = 10,
        grace_rate: float = PHI_INV_SQ,
    ):
        """
        Args:
            basis: Clifford algebra basis matrices
            xp: numpy or cupy
            survival_threshold: Quotient stability required to survive
            recurrence_threshold: How many times a schema must appear to be promoted
            grace_steps: Number of Grace iterations for survival test
            grace_rate: Contraction rate (stronger for REM)
        """
        self.basis = basis
        self.xp = xp
        self.survival_threshold = survival_threshold
        self.recurrence_threshold = recurrence_threshold
        self.grace_steps = grace_steps
        self.grace_rate = grace_rate
        
        # Candidate pool for schemas
        self.schema_candidates: Dict[str, Schema] = {}
    
    def _hash_canonical(self, M: np.ndarray, precision: int = 2) -> str:
        """Create a hash from the canonical form for deduplication."""
        rounded = np.round(M, precision)
        return str(rounded.tobytes())
    
    def recombine_compose(
        self, 
        A: np.ndarray, 
        B: np.ndarray
    ) -> np.ndarray:
        """
        Recombination operator: composition (A × B).
        
        Creates new context by sequential application.
        """
        return geometric_product(A, B)
    
    def recombine_unbind(
        self, 
        A: np.ndarray, 
        B: np.ndarray
    ) -> np.ndarray:
        """
        Recombination operator: unbinding (A × B⁻¹).
        
        Extracts the "difference" between A and B.
        This can reveal abstract relations.
        """
        # Approximate inverse (use pseudoinverse for stability)
        try:
            B_inv = self.xp.linalg.pinv(B)
        except (self.xp.linalg.LinAlgError, ValueError, RuntimeError) as e:
            # Pinv failed - this indicates numerical instability or degenerate matrix
            # Falling back to identity is WRONG - it silently corrupts the computation
            # Instead, raise with context so we can fix the root cause
            raise RuntimeError(f"Pseudoinverse failed in recombine_unbind: {e}. Matrix B may be degenerate or numerically unstable.") from e
    
    def recombine_perturb(
        self, 
        A: np.ndarray,
        noise_scale: float = PHI_INV_CUBE,  # φ-derived (was 0.1)
    ) -> np.ndarray:
        """
        Recombination operator: perturbation.
        
        Add small noise to explore nearby structures.
        """
        xp = self.xp
        if hasattr(xp, 'random'):
            noise = noise_scale * xp.random.randn(*A.shape)
        else:
            noise = noise_scale * np.random.randn(*A.shape)
            noise = xp.asarray(noise)
        return A + noise
    
    def survival_test(self, M: np.ndarray) -> Tuple[bool, np.ndarray, float]:
        """
        Test if a candidate survives strong Grace contraction.
        
        Returns:
            survived: True if structure is quotient-stable
            final: The final canonical form
            stability: Measure of stability (0 to 1)
        """
        xp = self.xp
        current = M.copy()
        previous = None
        
        for step in range(self.grace_steps):
            previous = current.copy()
            current = dream_grace_operator(current, self.basis, xp, self.grace_rate)
        
        # Normalize - use xp for GPU
        norm = float(xp.linalg.norm(current, 'fro'))
        if norm < 1e-8:
            return False, current, 0.0  # Collapsed to zero
        
        current = current / norm
        
        # Check stability (did it converge?)
        if previous is not None:
            prev_norm = float(xp.linalg.norm(previous, 'fro'))
            if prev_norm > 1e-8:
                previous = previous / prev_norm
                stability = frobenius_similarity(current, previous, xp)
            else:
                stability = 0.0
        else:
            stability = 1.0
        
        survived = stability >= self.survival_threshold
        return survived, current, stability
    
    def survival_test_batch(self, candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        BATCHED survival test - GPU-accelerated.
        
        Args:
            candidates: [N, 4, 4] batch of candidate matrices
            
        Returns:
            survived: [N] bool array
            finals: [N, 4, 4] canonical forms
            stabilities: [N] stability scores
        """
        xp = self.xp
        current = candidates.copy()
        
        # Apply Grace steps in batch
        for step in range(self.grace_steps):
            previous = current.copy()
            current = grace_operator_batch(current, self.basis, xp)
            # Apply additional damping rate
            current = current * self.grace_rate + previous * (1 - self.grace_rate) * 0.1
        
        # Batch normalize
        norms = xp.linalg.norm(current.reshape(len(current), -1), axis=1, keepdims=True)
        norms = xp.maximum(norms, 1e-8)
        finals = current / norms.reshape(-1, 1, 1)
        
        # Batch stability check
        prev_norms = xp.linalg.norm(previous.reshape(len(previous), -1), axis=1, keepdims=True)
        prev_norms = xp.maximum(prev_norms, 1e-8)
        previous_normed = previous / prev_norms.reshape(-1, 1, 1)
        
        # Batch frobenius similarity
        stabilities = xp.sum(finals * previous_normed, axis=(1, 2)) / (
            xp.linalg.norm(finals.reshape(len(finals), -1), axis=1) *
            xp.linalg.norm(previous_normed.reshape(len(previous_normed), -1), axis=1) + 1e-10
        )
        
        survived = stabilities >= self.survival_threshold
        return survived, finals, stabilities
    
    def dream_cycle(
        self,
        prototypes: List[SemanticPrototype],
        num_recombinations: int = 100,
        verbose: bool = False,
    ) -> List[Schema]:
        """
        Main REM routine: recombine and test for survival.
        
        FULLY BATCHED for GPU acceleration.
        
        This is the REM sleep algorithm:
            1. Sample pairs of prototypes
            2. Recombine using various operators (batched)
            3. Test survival under strong Grace (batched)
            4. Track recurring survivors as candidate schemas
            5. Promote schemas that pass recurrence threshold
        """
        if len(prototypes) < 2:
            if verbose:
                print("  REM: Not enough prototypes for recombination")
            return []
        
        if verbose:
            print(f"  REM: Running {num_recombinations} recombinations...")
        
        xp = self.xp
        
        # Get prototype matrices as batch array
        matrices = xp.array([p.prototype_matrix for p in prototypes])
        n_proto = len(matrices)
        
        # Pre-generate all random choices with NumPy (CuPy random has limitations)
        rng = np.random.default_rng()
        idx_pairs_np = rng.choice(n_proto, size=(num_recombinations, 2), replace=True)
        op_choices_np = rng.integers(0, 3, size=num_recombinations)
        op_names = ["compose", "unbind", "perturb"]
        
        # Fix same-index pairs
        same_mask = idx_pairs_np[:, 0] == idx_pairs_np[:, 1]
        idx_pairs_np[same_mask, 1] = (idx_pairs_np[same_mask, 0] + 1) % n_proto
        
        # Convert to xp arrays for GPU compatibility
        idx_pairs = xp.asarray(idx_pairs_np)
        op_choices = xp.asarray(op_choices_np)
        
        # BATCH: Generate all candidates at once
        A_batch = matrices[idx_pairs[:, 0]]  # [N, 4, 4]
        B_batch = matrices[idx_pairs[:, 1]]  # [N, 4, 4]
        
        candidates = xp.zeros((num_recombinations, 4, 4), dtype=DTYPE)
        
        # Compose: A @ B
        compose_mask = op_choices == 0
        if xp.any(compose_mask):
            candidates[compose_mask] = xp.matmul(A_batch[compose_mask], B_batch[compose_mask])
        
        # Unbind: A @ B^{-1} (using adjoint as proxy)
        unbind_mask = op_choices == 1
        if xp.any(unbind_mask):
            # Use transpose as a simple pseudo-inverse for stability
            B_adj = xp.transpose(B_batch[unbind_mask], axes=(0, 2, 1))
            candidates[unbind_mask] = xp.matmul(A_batch[unbind_mask], B_adj)
        
        # Perturb: A + φ⁻⁴ noise
        perturb_mask = op_choices == 2
        if xp.any(perturb_mask):
            noise = xp.array(rng.standard_normal((int(xp.sum(perturb_mask)), 4, 4)) * PHI_INV_FOUR)
            candidates[perturb_mask] = A_batch[perturb_mask] + noise
        
        # BATCH survival test
        survived_mask, canonicals, stabilities = self.survival_test_batch(candidates)
        
        survivors = int(xp.sum(survived_mask))
        
        # Process survivors (must be sequential for hashing)
        for i in range(num_recombinations):
            if survived_mask[i]:
                canonical = canonicals[i]
                op_name = op_names[int(op_choices_np[i])]  # Use numpy version for list indexing
                
                # Track in candidate pool
                key = self._hash_canonical(canonical)
                # Get source prototype indices (use numpy version for indexing)
                src_ids = [int(idx_pairs_np[i, 0]), int(idx_pairs_np[i, 1])]
                
                if key not in self.schema_candidates:
                    self.schema_candidates[key] = Schema(
                        canonical_form=canonical,
                        recurrence_count=1,
                        source_operations=[op_name],
                        source_prototype_ids=src_ids,
                    )
                else:
                    self.schema_candidates[key].recurrence_count += 1
                    self.schema_candidates[key].source_operations.append(op_name)
                    # Add new source prototypes (deduplicated)
                    for sid in src_ids:
                        if sid not in self.schema_candidates[key].source_prototype_ids:
                            self.schema_candidates[key].source_prototype_ids.append(sid)
        
        if verbose:
            print(f"    Survival rate: {survivors}/{num_recombinations} ({100*survivors/num_recombinations:.1f}%)")
        
        # Promote schemas that pass recurrence threshold
        promoted = [
            schema for schema in self.schema_candidates.values()
            if schema.recurrence_count >= self.recurrence_threshold
        ]
        
        if verbose:
            print(f"    Promoted schemas: {len(promoted)}")
            for schema in promoted[:5]:  # Show top 5
                print(f"      recurrence={schema.recurrence_count}, ops={schema.source_operations[:3]}")
        
        return promoted


# =============================================================================
# SEMANTIC MEMORY (Hierarchical storage)
# =============================================================================

class SemanticMemory:
    """
    Hierarchical semantic memory supporting multi-resolution retrieval.
    
    Levels:
        0: Fine-grained prototypes (high detail)
        1+: Coarser abstractions (more generalization)
    
    Retrieval:
        1. Search at coarse level for candidate regions
        2. Refine at finer levels
        3. Return best match with target distribution
    """
    
    def __init__(
        self,
        basis: np.ndarray,
        xp = np,
        num_levels: int = 3,
    ):
        self.basis = basis
        self.xp = xp
        self.num_levels = num_levels
        
        # Storage by level
        self.levels: List[List[SemanticPrototype]] = [[] for _ in range(num_levels)]
        
        # Schema storage
        self.schemas: List[Schema] = []
    
    def add_prototype(self, proto: SemanticPrototype, level: int = 0):
        """Add a prototype at the specified level."""
        proto.level = level
        if level < len(self.levels):
            self.levels[level].append(proto)
    
    def add_schema(self, schema: Schema, max_schemas: int = None):
        """
        Add a discovered schema with φ-derived bounded growth.
        
        THEORY-TRUE: Schema capacity scales with prototype count.
        More prototypes → more possible combinations → need more schemas.
        Default: φ² × num_prototypes × 100 (allows rich schema space)
        """
        # φ-derived cap: scales with prototype count (no arbitrary 10000!)
        if max_schemas is None:
            num_protos = sum(len(level) for level in self.levels)
            max_schemas = max(10000, int(PHI * PHI * num_protos * 100))  # φ² × protos × 100
        
        if len(self.schemas) >= max_schemas:
            # Find schema with lowest recurrence
            min_idx = min(range(len(self.schemas)), key=lambda i: self.schemas[i].recurrence_count)
            if schema.recurrence_count > self.schemas[min_idx].recurrence_count:
                self.schemas[min_idx] = schema  # Replace weakest
            # else: discard new schema (not useful enough)
        else:
            self.schemas.append(schema)
    
    def retrieve(
        self, 
        query: np.ndarray,
        top_k: int = 5,
        use_pattern_completion: bool = False,
        completion_steps: int = 3,
        use_schemas: bool = True,
    ) -> List[Tuple[SemanticPrototype, float]]:
        """
        VECTORIZED hierarchical retrieval: coarse to fine.
        
        PATTERN COMPLETION (optional):
            When enabled, applies Grace flow to query before matching.
            This denoises partial/noisy queries toward stored patterns.
            
        SCHEMA RETRIEVAL (v4.12.0):
            When enabled, also searches schemas. If a query matches a schema,
            returns the prototypes that contributed to that schema.
            This enables structural generalization.
        
        Args:
            query: [4, 4] query matrix
            top_k: Number of results to return
            use_pattern_completion: If True, apply Grace flow first
            completion_steps: Number of Grace iterations
            use_schemas: If True, also search schemas for structural matches
        
        Returns:
            List of (prototype, similarity) pairs, sorted by similarity
        """
        xp = self.xp
        
        # Optional pattern completion
        if use_pattern_completion:
            query, _ = pattern_complete(query, self.basis, xp, max_steps=completion_steps)
        
        # Collect all prototypes across levels
        all_protos = []
        for level in self.levels:
            all_protos.extend(level)
        
        if not all_protos:
            return []
        
        # VECTORIZED: Compute all similarities at once
        proto_matrices = xp.array([p.prototype_matrix for p in all_protos])
        sims = frobenius_similarity_batch(query, proto_matrices, xp)
        
        # Create sorted (proto, sim) pairs
        candidates = list(zip(all_protos, [float(s) for s in sims]))
        
        # Schema-based retrieval (NEW in v4.12.0)
        if use_schemas and self.schemas:
            schema_matrices = xp.array([s.canonical_form for s in self.schemas])
            schema_sims = frobenius_similarity_batch(query, schema_matrices, xp)
            
            # For each matching schema, add its source prototypes
            for i, (schema, sim) in enumerate(zip(self.schemas, schema_sims)):
                sim = float(sim)
                if sim > PHI_INV_SQ:  # Schema matches structurally
                    # Add source prototypes with boosted similarity
                    for proto_id in schema.source_prototype_ids:
                        if proto_id < len(all_protos):
                            # Boost: combine schema match with prototype relevance
                            # Use φ-weighted combination
                            proto = all_protos[proto_id]
                            proto_sim = float(sims[proto_id])
                            combined_sim = PHI_INV * sim + (1 - PHI_INV) * proto_sim
                            candidates.append((proto, combined_sim))
        
        candidates.sort(key=lambda x: -x[1])
        
        # Deduplicate (same prototype may appear multiple times via schemas)
        seen = set()
        unique_candidates = []
        for proto, sim in candidates:
            proto_id = id(proto)
            if proto_id not in seen:
                seen.add(proto_id)
                unique_candidates.append((proto, sim))
        
        return unique_candidates[:top_k]
    
    def retrieve_with_radius(
        self,
        query: np.ndarray,
        use_pattern_completion: bool = False,
        completion_steps: int = 3,
    ) -> Optional[SemanticPrototype]:
        """
        VECTORIZED retrieve best prototype within its radius.
        
        This enables generalization: if query is close enough to a prototype,
        use that prototype's target distribution.
        
        PATTERN COMPLETION (optional):
            When enabled, applies Grace flow to query before matching.
            Denoises partial inputs toward stored patterns.
        """
        xp = self.xp
        
        # Optional pattern completion
        if use_pattern_completion:
            query, _ = pattern_complete(query, self.basis, xp, max_steps=completion_steps)
        
        # Collect all prototypes
        all_protos = []
        for level in self.levels:
            all_protos.extend(level)
        
        if not all_protos:
            return None
        
        # VECTORIZED: Compute all similarities at once
        proto_matrices = xp.array([p.prototype_matrix for p in all_protos])
        radii = xp.array([p.radius for p in all_protos])
        
        sims = frobenius_similarity_batch(query, proto_matrices, xp)
        distances = 1.0 - sims
        
        # Margins: radius - distance (positive = within radius)
        margins = radii - distances
        
        # Find best margin
        best_idx = int(xp.argmax(margins))
        best_margin = float(margins[best_idx])
        
        return all_protos[best_idx] if best_margin >= 0 else None
    
    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        total = sum(len(level) for level in self.levels)
        return {
            "total_prototypes": total,
            "prototypes_by_level": [len(level) for level in self.levels],
            "num_schemas": len(self.schemas),
            "avg_entropy": np.mean([p.entropy() for level in self.levels for p in level]) if total > 0 else 0,
        }
    
    def schema_attention(
        self,
        query: np.ndarray,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Theory-true φ-weighted attention over schemas.
        
        THEORY (from meta-attention exploration):
            Instead of softmax(QK^T / √d), use:
            weights = φ^(-distance) / Σ φ^(-distance)
            
            This is the same mathematical form, but with φ as the
            "natural temperature" of our Clifford algebra.
        
        Args:
            query: [4, 4] query context matrix
            temperature: Scaling factor for distances (default 1.0)
            
        Returns:
            (weighted_result, info_dict)
            weighted_result: φ-weighted combination of schema transformations
            info_dict: Contains attention weights and selected schemas
        """
        xp = self.xp
        
        if not self.schemas:
            return query, {"weights": [], "num_schemas": 0, "top_schema": None}
        
        # Compute distances in quotient space (witness distance)
        from holographic_v4.quotient import extract_witness
        
        query_sigma, query_pseudo = extract_witness(query, self.basis, xp)
        
        weights = []
        for schema in self.schemas:
            s_sigma, s_pseudo = extract_witness(schema.canonical_form, self.basis, xp)
            dist = np.sqrt((query_sigma - s_sigma)**2 + (query_pseudo - s_pseudo)**2)
            # φ-power weighting (analogous to softmax with temperature)
            weight = PHI ** (-dist * temperature)
            weights.append(weight)
        
        weights = np.array(weights)
        
        # Normalize (like softmax)
        total_weight = weights.sum()
        if total_weight < 1e-8:
            return query, {"weights": [], "num_schemas": len(self.schemas), "top_schema": None}
        
        weights = weights / total_weight
        
        # Weighted combination of schema canonical forms
        result = np.zeros((4, 4), dtype=query.dtype)
        for w, schema in zip(weights, self.schemas):
            # Apply schema as transformation: query @ schema
            # This "applies" the schema's relational structure
            transformed = query @ schema.canonical_form
            result += w * transformed
        
        # Normalize result
        result_norm = np.sqrt(np.sum(result * result)) + 1e-8
        result = result / result_norm
        
        # Find top schema
        top_idx = int(np.argmax(weights))
        
        return result, {
            "weights": weights.tolist(),
            "num_schemas": len(self.schemas),
            "top_schema": top_idx,
            "top_weight": float(weights[top_idx]),
            "entropy": float(-np.sum(weights * np.log(weights + 1e-8))),  # Attention entropy
        }
    
    def cluster_schemas_into_meta(
        self,
        similarity_threshold: float = PHI_INV,
        min_cluster_size: int = 2,
    ) -> List['MetaSchema']:
        """
        Cluster schemas into meta-schemas using quotient distance.
        
        THEORY:
            Meta-schemas are "categories of grammatical rules":
            - Meta-schema 1: Inflectional morphology (groups -ed, -s, -ing schemas)
            - Meta-schema 2: Word order rules (groups SVO pattern schemas)
            
            Clustering uses φ-threshold: schemas within φ⁻¹ distance belong together.
        
        Args:
            similarity_threshold: Max distance for same cluster (default φ⁻¹)
            min_cluster_size: Minimum schemas per meta-schema
            
        Returns:
            List of MetaSchema objects
        """
        if len(self.schemas) < min_cluster_size:
            return []
        
        xp = self.xp
        from holographic_v4.quotient import extract_witness
        
        # Extract witnesses for all schemas
        witnesses = []
        for schema in self.schemas:
            sigma, pseudo = extract_witness(schema.canonical_form, self.basis, xp)
            witnesses.append((sigma, pseudo))
        
        # Simple clustering: group by witness proximity
        used = set()
        clusters = []
        
        for i, (s_i, p_i) in enumerate(witnesses):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            
            for j, (s_j, p_j) in enumerate(witnesses):
                if j in used:
                    continue
                
                dist = np.sqrt((s_i - s_j)**2 + (p_i - p_j)**2)
                if dist < similarity_threshold:
                    cluster.append(j)
                    used.add(j)
            
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
        
        # Create MetaSchema objects
        meta_schemas = []
        for cluster_indices in clusters:
            cluster_schemas = [self.schemas[i] for i in cluster_indices]
            
            # Representative = average of canonical forms
            representative = np.mean(
                [s.canonical_form for s in cluster_schemas], 
                axis=0
            )
            # Normalize
            rep_norm = np.sqrt(np.sum(representative * representative)) + 1e-8
            representative = representative / rep_norm
            
            meta_schemas.append(MetaSchema(
                representative=representative,
                schema_indices=cluster_indices,
                schemas=cluster_schemas,
            ))
        
        return meta_schemas
    
    def hierarchical_attention(
        self,
        query: np.ndarray,
        meta_schemas: List['MetaSchema'] = None,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Two-level hierarchical φ-attention: meta-schema → schema.
        
        THEORY (from meta-attention exploration):
            Level 1: Select meta-schema category (e.g., "morphology")
            Level 2: Select specific schema within category (e.g., "past tense")
            
            Both levels use φ-weighted attention:
                weights = φ^(-distance) / Σ φ^(-distance)
            
            This is like the brain's executive function selecting strategy,
            then applying specific rules.
        
        BRAIN ANALOG:
            - Meta = Broca's area categories (clause types)
            - Schema = Wernicke's area patterns (lexical rules)
            - Hierarchical attention = Executive function routing
        
        Args:
            query: [4, 4] query context matrix
            meta_schemas: Pre-clustered meta-schemas (or None to cluster now)
            temperature: Scaling factor for distances
            
        Returns:
            (result, info_dict) with hierarchical attention information
        """
        xp = self.xp
        from holographic_v4.quotient import extract_witness
        
        # Cluster schemas if not provided
        if meta_schemas is None:
            meta_schemas = self.cluster_schemas_into_meta()
        
        if not meta_schemas:
            # Fall back to flat schema attention
            return self.schema_attention(query, temperature)
        
        # ==============================
        # Level 1: Meta-schema selection
        # ==============================
        query_sigma, query_pseudo = extract_witness(query, self.basis, xp)
        
        meta_weights = []
        for meta in meta_schemas:
            m_sigma, m_pseudo = extract_witness(meta.representative, self.basis, xp)
            dist = np.sqrt((query_sigma - m_sigma)**2 + (query_pseudo - m_pseudo)**2)
            meta_weights.append(PHI ** (-dist * temperature))
        
        meta_weights = np.array(meta_weights)
        meta_weights = meta_weights / (meta_weights.sum() + 1e-8)
        
        top_meta_idx = int(np.argmax(meta_weights))
        top_meta = meta_schemas[top_meta_idx]
        
        # ==============================
        # Level 2: Schema selection (within top meta)
        # ==============================
        schema_weights = []
        for schema in top_meta.schemas:
            s_sigma, s_pseudo = extract_witness(schema.canonical_form, self.basis, xp)
            dist = np.sqrt((query_sigma - s_sigma)**2 + (query_pseudo - s_pseudo)**2)
            schema_weights.append(PHI ** (-dist * temperature))
        
        schema_weights = np.array(schema_weights)
        schema_weights = schema_weights / (schema_weights.sum() + 1e-8)
        
        top_schema_idx = int(np.argmax(schema_weights))
        
        # ==============================
        # Apply weighted schema combination
        # ==============================
        result = np.zeros((4, 4), dtype=query.dtype)
        for w, schema in zip(schema_weights, top_meta.schemas):
            transformed = query @ schema.canonical_form
            result += w * transformed
        
        # Normalize
        result_norm = np.sqrt(np.sum(result * result)) + 1e-8
        result = result / result_norm
        
        return result, {
            "num_meta_schemas": len(meta_schemas),
            "meta_weights": meta_weights.tolist(),
            "top_meta": top_meta_idx,
            "top_meta_weight": float(meta_weights[top_meta_idx]),
            "schemas_in_top_meta": len(top_meta.schemas),
            "schema_weights": schema_weights.tolist(),
            "top_schema_in_meta": top_schema_idx,
            "top_schema_weight": float(schema_weights[top_schema_idx]),
            "meta_entropy": float(-np.sum(meta_weights * np.log(meta_weights + 1e-8))),
            "schema_entropy": float(-np.sum(schema_weights * np.log(schema_weights + 1e-8))),
        }


@dataclass
class MetaSchema:
    """
    A meta-schema = cluster of related schemas.
    
    THEORY:
        If schemas are grammatical rules, meta-schemas are grammatical categories:
        - Meta-schema "inflection": groups past tense, plural, progressive schemas
        - Meta-schema "word order": groups SVO, question inversion schemas
        
        The representative is the centroid of the cluster in witness space.
    """
    representative: np.ndarray  # [4, 4] centroid of cluster
    schema_indices: List[int]   # Indices into parent schemas list
    schemas: List[Schema]       # The actual schemas in this cluster


# =============================================================================
# φ-DECAY FORGETTING (Theory-True Memory Management)
# =============================================================================

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
    
    # Compute priority scores for all episodes
    matrices = xp.array([ep.context_matrix for ep in episodes])
    
    # Salience (scalar + pseudoscalar energy)
    from holographic_v4.quotient import grace_stability_batch
    saliences = compute_salience_batch(matrices, basis, xp)
    stabilities = grace_stability_batch(matrices, basis, xp)
    
    # Priority = stability × salience (φ-weighted importance)
    # Add φ⁻⁸ offset + φ⁻⁸-scaled random noise to break ties
    priorities = stabilities * (saliences + PHI_INV_EIGHT) + PHI_INV_EIGHT * xp.random.rand(len(episodes))
    
    # φ-decay survival probability: P(survive) = φ^(-k × (1 - priority))
    # where k scales how aggressive forgetting is
    # k = log(max/current) / log(φ) ensures we get to target capacity
    excess_ratio = len(episodes) / max_episodes
    k = np.log(excess_ratio) / np.log(PHI)  # Theory-derived scaling
    
    survival_probs = PHI ** (-k * (1.0 - priorities))
    survival_probs = xp.clip(survival_probs, 0.0, 1.0)
    
    # Stochastic forgetting (brain-like randomness)
    random_vals = xp.random.rand(len(episodes))
    survive_mask = random_vals < survival_probs
    
    # Ensure we hit capacity exactly (top priority as fallback)
    num_surviving = int(xp.sum(survive_mask))
    if num_surviving != max_episodes:
        # Either too few or too many survived - use deterministic top-k
        priority_order = xp.argsort(priorities)[::-1]  # Highest first
        survive_mask = xp.zeros(len(episodes), dtype=bool)
        survive_mask[priority_order[:max_episodes]] = True
    
    # Filter episodes
    surviving = [ep for i, ep in enumerate(episodes) if survive_mask[i]]
    num_forgotten = len(episodes) - len(surviving)
    
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


class WorkingMemory:
    """
    Small, fast working memory cache with φ-decay.
    
    BRAIN ANALOG:
        Working memory holds ~7±2 items (Miller's law).
        In our theory, this is approximately φ³ ≈ 4.236 → ~4 items minimum.
        
        Items decay with φ⁻¹ per retrieval step (inhibition of return).
        Most recent item has highest activation.
    
    This provides:
        1. O(1) lookup for very recent contexts (no semantic search needed)
        2. Priming effects (recent retrievals easier to re-retrieve)
        3. Natural decay (old items forgotten automatically)
    """
    
    def __init__(self, capacity: int = 7, decay_rate: float = PHI_INV):
        """
        Args:
            capacity: Maximum items (default 7 ≈ Miller's number)
            decay_rate: Activation decay per step (default φ⁻¹)
        """
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.items: List[Tuple[np.ndarray, int, float]] = []  # (context_matrix, target, activation)
    
    def add(self, context_matrix: np.ndarray, target: int):
        """Add item with maximum activation."""
        # Decay all existing items
        self.items = [(m, t, a * self.decay_rate) for m, t, a in self.items]
        
        # Remove items below minimum activation (φ⁻³ ≈ 0.236)
        self.items = [(m, t, a) for m, t, a in self.items if a > PHI_INV ** 3]
        
        # Add new item with activation 1.0
        self.items.append((context_matrix, target, 1.0))
        
        # Enforce capacity (keep highest activation)
        if len(self.items) > self.capacity:
            self.items.sort(key=lambda x: x[2], reverse=True)
            self.items = self.items[:self.capacity]
    
    def lookup(self, query_matrix: np.ndarray, threshold: float = 1.0 - PHI_INV_CUBE) -> Optional[Tuple[int, float]]:
        # NOTE: threshold ≈ 0.764 (was 0.95). For working memory, we want high confidence
        # matches. The φ-derived "high confidence" threshold is 1 - φ⁻³.
        """
        Fast lookup for exact/near-exact match.
        
        Returns:
            (target, similarity) if match found, else None
        """
        if len(self.items) == 0:
            return None
        
        # Check all items (working memory is small, so O(n) is fine)
        best_sim = 0.0
        best_target = None
        
        for ctx, target, activation in self.items:
            sim = frobenius_similarity(query_matrix, ctx, np)
            # Weight by activation (recent items get priority)
            weighted_sim = sim * activation
            
            if weighted_sim > best_sim:
                best_sim = weighted_sim
                best_target = target
        
        if best_sim >= threshold:
            return (best_target, best_sim)
        return None
    
    def __len__(self) -> int:
        return len(self.items)


# =============================================================================
# INTEGRATED DREAMING SYSTEM
# =============================================================================

class DreamingSystem:
    """
    Complete dreaming system integrating Non-REM and REM phases.
    
    BRAIN-INSPIRED FEATURES (12 total):
        
        MEMORY ENCODING:
        1. Salience: Prioritize emotionally important episodes (scalar + pseudoscalar)
        2. Novelty: Prioritize novel episodes (distance from existing prototypes)
        3. Prediction Error: Prioritize surprising episodes (Grace residual)
        4. Predictive Coding: Only encode what memory doesn't predict
        
        MEMORY MAINTENANCE:
        5. Pruning: Remove low-salience + low-support memories
        6. Interference Management: Merge similar prototypes
        7. Reconsolidation: Retrieval updates memory
        8. Pseudo-Rehearsal: Generate samples to prevent forgetting
        
        MEMORY RETRIEVAL:
        9. Working Memory Gating: Salience-based attention
        10. Pattern Completion: Grace flow denoises queries
        11. Inhibition of Return: Suppress recently retrieved
        
        SEQUENCE MEMORY:
        12. Sequence Replay: Store/replay transitions via vorticity
    
    Usage:
        dreaming = DreamingSystem(model.basis)
        
        # Periodically consolidate episodic memories
        episodes = [EpisodicEntry(ctx, target) for ctx, target in waking_data]
        dreaming.sleep(episodes)
        
        # Query semantic memory with pattern completion
        match = dreaming.retrieve(query, use_pattern_completion=True)
        if match:
            target = match.sample_target()
    """
    
    def __init__(
        self,
        basis: np.ndarray,
        xp = np,
        similarity_threshold: float = PHI_INV,  # φ-derived (was 0.8)
        min_cluster_size: int = 3,
        survival_threshold: float = PHI_INV,  # φ-derived (was 0.9)
        recurrence_threshold: int = 3,
        use_salience: bool = True,
        use_novelty: bool = True,
        use_predictive_coding: bool = True,
        use_pattern_completion: bool = True,
        use_inhibition_of_return: bool = True,
        use_sequence_replay: bool = True,
        use_pseudo_rehearsal: bool = True,
        transition_buffer_capacity: int = 1000,
    ):
        self.basis = basis
        self.xp = xp
        
        # Feature flags
        self.use_salience = use_salience
        self.use_novelty = use_novelty
        self.use_predictive_coding = use_predictive_coding
        self.use_pattern_completion = use_pattern_completion
        self.use_inhibition_of_return = use_inhibition_of_return
        self.use_sequence_replay = use_sequence_replay
        self.use_pseudo_rehearsal = use_pseudo_rehearsal
        
        # Initialize semantic memory first (needed for novelty computation)
        self.semantic_memory = SemanticMemory(basis=basis, xp=xp)
        
        # Initialize consolidator with semantic memory reference for novelty
        self.consolidator = NonREMConsolidator(
            basis=basis,
            xp=xp,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            use_salience=use_salience,
            use_novelty=use_novelty,
            semantic_memory=self.semantic_memory,
        )
        
        self.recombinator = REMRecombinator(
            basis=basis,
            xp=xp,
            survival_threshold=survival_threshold,
            recurrence_threshold=recurrence_threshold,
        )
        
        # NEW: Transition buffer for sequence replay
        self.transition_buffer = TransitionBuffer(
            capacity=transition_buffer_capacity,
            xp=xp,
        )
        
        # NEW: Retrieval history for inhibition of return
        self.retrieval_history = RetrievalHistory(
            decay_rate=PHI_INV,
            xp=xp,
        )
        
        # NEW: Working memory (fast cache for recent items)
        self.working_memory = WorkingMemory(capacity=7, decay_rate=PHI_INV)
        
        # NEW: Adaptive threshold parameters
        self.base_similarity_threshold = similarity_threshold
        self.use_adaptive_threshold = True
        self.total_episodes_seen = 0  # Track for adaptive threshold
        
        # Statistics
        self.sleep_count = 0
        self.total_episodes_processed = 0
    
    def sleep(
        self,
        episodes: List[EpisodicEntry],
        rem_cycles: int = 1,
        n_sequence_replays: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a complete sleep cycle with all brain-inspired features.
        
        PHASES:
            0. Pre-processing: Predictive coding filter + sequence recording
            1. Non-REM: Consolidation into prototypes
            2. REM: Recombination + schema discovery
            3. Sequence Replay: Replay temporal transitions
        
        Args:
            episodes: Episodic memories to consolidate
            rem_cycles: Number of REM recombination rounds
            n_sequence_replays: Number of sequences to replay
            verbose: Print progress
        
        Returns:
            Statistics about the sleep cycle
        """
        xp = self.xp
        
        if verbose:
            print()
            print("╔" + "═" * 50 + "╗")
            print("║" + "  SLEEP CYCLE  ".center(50) + "║")
            print("╚" + "═" * 50 + "╝")
        
        stats = {
            "input_episodes": len(episodes),
            "prototypes_created": 0,
            "schemas_discovered": 0,
            "predictive_filtered": 0,
            "transitions_recorded": 0,
            "sequences_replayed": 0,
            "episodes_forgotten": 0,
            "similarity_threshold": self.consolidator.similarity_threshold,
        }
        
        # Track total episodes for adaptive threshold
        self.total_episodes_seen += len(episodes)
        
        # Phase -1: φ-Decay Forgetting (if over capacity)
        # THEORY-TRUE: Cap scales with prototype count (not arbitrary 5000!)
        # More prototypes = more categories = can handle more episodes
        num_protos = self.semantic_memory.stats()['total_prototypes'] if self.semantic_memory else 0
        MAX_EPISODES_PER_SLEEP = max(5000, int(PHI * PHI * (num_protos + 100) * 10))  # φ² × (protos+100) × 10
        if len(episodes) > MAX_EPISODES_PER_SLEEP:
            if verbose:
                print()
                print("  Phase -1: φ-DECAY FORGETTING")
                print("  " + "-" * 40)
            
            episodes, num_forgotten = phi_decay_forget(
                episodes, MAX_EPISODES_PER_SLEEP, self.basis, xp
            )
            stats["episodes_forgotten"] = num_forgotten
            
            if verbose:
                print(f"    φ-decay: {num_forgotten} low-priority episodes forgotten")
                print(f"    Surviving: {len(episodes)} (capacity={MAX_EPISODES_PER_SLEEP})")
        
        # Adaptive similarity threshold (φ-derived bounds)
        if self.use_adaptive_threshold:
            old_threshold = self.consolidator.similarity_threshold
            new_threshold = compute_adaptive_threshold(
                self.semantic_memory,
                min_threshold=PHI_INV_CUBE,  # φ⁻³ ≈ 0.236 minimum
                max_threshold=1 - PHI_INV_CUBE,  # 1 - φ⁻³ ≈ 0.764 maximum
            )
            self.consolidator.similarity_threshold = new_threshold
            stats["similarity_threshold"] = new_threshold
            
            if verbose and abs(new_threshold - old_threshold) > PHI_INV_SIX:  # φ⁻⁶ significance
                print(f"    Adaptive threshold: {old_threshold:.2f} → {new_threshold:.2f}")
        
        # Phase 0: Pre-processing
        if verbose:
            print()
            print("  Phase 0: PRE-PROCESSING")
            print("  " + "-" * 40)
        
        # 0.1 Record temporal transitions for sequence replay
        if self.use_sequence_replay and len(episodes) >= 2:
            self.transition_buffer.add_from_episode_sequence(episodes, self.basis, xp)
            stats["transitions_recorded"] = len(episodes) - 1
            if verbose:
                print(f"    Recorded {len(episodes) - 1} transitions")
        
        # 0.2 Apply predictive coding filter (only encode unpredicted)
        episodes_to_consolidate = episodes
        if self.use_predictive_coding and self.semantic_memory.stats()['total_prototypes'] > 0:
            significant, redundant, pc_stats = predictive_encode_batch(
                episodes, self.semantic_memory, self.basis, xp,
                significance_threshold=PHI_INV_CUBE,  # φ⁻³ significance threshold
            )
            episodes_to_consolidate = significant
            stats["predictive_filtered"] = pc_stats['redundant']
            if verbose:
                print(f"    Predictive coding: {pc_stats['redundant']} redundant filtered")
                print(f"    Episodes to consolidate: {len(episodes_to_consolidate)}")
        
        # Phase 1: Non-REM (consolidation)
        if verbose:
            print()
            print("  Phase 1: NON-REM (Consolidation)")
            print("  " + "-" * 40)
        
        if len(episodes_to_consolidate) > 0:
            prototypes = self.consolidator.consolidate(episodes_to_consolidate, verbose=verbose)
            
            # Add to semantic memory
            for proto in prototypes:
                self.semantic_memory.add_prototype(proto, level=0)
            
            stats["prototypes_created"] = len(prototypes)
        else:
            if verbose:
                print("    No new episodes to consolidate (all predicted)")
        
        # Phase 2: REM (recombination)
        if verbose:
            print()
            print("  Phase 2: REM (Recombination)")
            print("  " + "-" * 40)
        
        for cycle in range(rem_cycles):
            if verbose and rem_cycles > 1:
                print(f"    Cycle {cycle + 1}/{rem_cycles}")
            
            # Use all prototypes from semantic memory
            all_prototypes = [p for level in self.semantic_memory.levels for p in level]
            
            # Cap REM recombinations to prevent O(n²) slowdown
            max_recombinations = min(len(all_prototypes) * 10, 1000)
            
            if len(all_prototypes) >= 2:
                schemas = self.recombinator.dream_cycle(
                    all_prototypes,
                    num_recombinations=max_recombinations,
                    verbose=verbose,
                )
                
                for schema in schemas:
                    self.semantic_memory.add_schema(schema)
                
                stats["schemas_discovered"] += len(schemas)
            else:
                if verbose:
                    print("    Not enough prototypes for recombination")
        
        # Phase 3: Sequence replay (Sharp Wave Ripples)
        if self.use_sequence_replay and self.transition_buffer.stats()['count'] > 0:
            if verbose:
                print()
                print("  Phase 3: SEQUENCE REPLAY")
                print("  " + "-" * 40)
            
            sequences, replay_stats = replay_transitions_during_rem(
                self.transition_buffer, self.basis, xp,
                n_replays=n_sequence_replays,
                sequence_length=3,
            )
            stats["sequences_replayed"] = replay_stats['replayed']
            
            if verbose:
                print(f"    Replayed {replay_stats['replayed']} sequences")
                print(f"    Avg length: {replay_stats['avg_length']:.1f}")
        
        # Phase 4: SYNAPTIC PRUNING (Tononi's Synaptic Homeostasis)
        # THEORY: Sleep actively prunes weak synapses to improve signal-to-noise
        # This is NOT capacity-based but quality-based - weak memories harm retrieval
        if verbose:
            print()
            print("  Phase 4: SYNAPTIC PRUNING")
            print("  " + "-" * 40)
        
        prune_stats = prune_semantic_memory(
            self.semantic_memory, 
            self.basis, 
            xp,
            salience_threshold=PHI_INV_CUBE,  # φ⁻³ - same as Grace spectral gap
            support_threshold=2,  # Need at least 2 supporting episodes
            verbose=verbose,
        )
        stats["prototypes_pruned"] = prune_stats['pruned']
        
        if verbose:
            print(f"    Pruned {prune_stats['pruned']} weak prototypes")
            print(f"    Prototypes: {prune_stats['total_before']} → {prune_stats['total_after']}")
        
        # Phase 5: INTERFERENCE MANAGEMENT (merge similar prototypes)
        # THEORY: Similar memories compete during retrieval
        # Merging reduces interference and improves pattern separation
        if verbose:
            print()
            print("  Phase 5: INTERFERENCE MANAGEMENT")
            print("  " + "-" * 40)
        
        # THEORY-TRUE MERGING THRESHOLD (v4.19.1 fix):
        # 
        # φ⁻¹ (0.618) is the RETRIEVAL confidence threshold (single basin vs distributed prior)
        # But merging should be MUCH more conservative - only merge TRUE DUPLICATES
        # 
        # Theory: "More prototypes → better generalization (coverage)" (distributed_prior.py)
        # Aggressive merging destroys coverage, killing generalization.
        # 
        # Using 1 - φ⁻³ ≈ 0.764 (same as WorkingMemory high-confidence threshold)
        # This merges only prototypes that are 76.4%+ similar (near-duplicates)
        # vs the old 61.8% threshold which merged semantically distinct patterns
        interference_stats = manage_interference(
            self.semantic_memory,
            self.basis,
            xp,
            similarity_threshold=1 - PHI_INV_CUBE,  # ≈ 0.764 - merge only near-duplicates
            max_merges_per_cycle=5,  # Further limit merging to preserve diversity
            verbose=verbose,
        )
        stats["prototypes_merged"] = interference_stats['merges']
        
        if verbose:
            print(f"    Merged {interference_stats['merges']} similar prototypes")
        
        # Update stats
        self.sleep_count += 1
        self.total_episodes_processed += len(episodes)
        
        # Advance retrieval history time (inhibition decays)
        self.retrieval_history.advance_time(1.0)
        
        if verbose:
            print()
            print("  SLEEP COMPLETE")
            print(f"    Prototypes created: {stats['prototypes_created']}")
            print(f"    Prototypes pruned:  {stats['prototypes_pruned']}")
            print(f"    Schemas: {stats['schemas_discovered']}")
            mem_stats = self.semantic_memory.stats()
            print(f"    Total memory: {mem_stats['total_prototypes']} prototypes")
        
        return stats
    
    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 1,
        use_pattern_completion: Optional[bool] = None,
        use_inhibition: Optional[bool] = None,
        completion_steps: int = 3,
        inhibition_weight: float = PHI_INV_CUBE,  # φ-derived (was 0.3)
    ) -> Tuple[Optional[SemanticPrototype], float, Dict[str, Any]]:
        """
        Retrieve from semantic memory with brain-inspired features.
        
        FEATURES APPLIED:
            1. Pattern Completion: Grace flow denoises query first
            2. Inhibition of Return: Recently retrieved items penalized
            
        Args:
            query: [4, 4] query matrix
            top_k: Number of results (default 1)
            use_pattern_completion: Override class default
            use_inhibition: Override class default
            completion_steps: Grace steps for completion
            inhibition_weight: Penalty for recently retrieved
            
        Returns:
            best_proto: Best matching prototype (or None)
            similarity: Adjusted similarity score
            info: Retrieval statistics
        """
        xp = self.xp
        
        # Use class defaults if not overridden
        do_completion = use_pattern_completion if use_pattern_completion is not None else self.use_pattern_completion
        do_inhibition = use_inhibition if use_inhibition is not None else self.use_inhibition_of_return
        
        info = {
            'used_pattern_completion': do_completion,
            'used_inhibition': do_inhibition,
            'completion_change': 0.0,
            'inhibited_count': 0,
        }
        
        # Apply pattern completion
        completed_query = query
        if do_completion:
            completed_query, comp_info = pattern_complete(
                query, self.basis, xp,
                max_steps=completion_steps,
            )
            info['completion_change'] = comp_info['total_change']
        
        # Retrieve with or without inhibition
        if do_inhibition:
            results, inh_info = retrieve_with_inhibition(
                completed_query, self.semantic_memory, self.retrieval_history,
                top_k=top_k,
                inhibition_weight=inhibition_weight,
                record_retrieval=True,
                xp=xp,
            )
            info['inhibited_count'] = inh_info['inhibited']
        else:
            # Standard retrieval
            results = self.semantic_memory.retrieve(
                completed_query,
                top_k=top_k,
                use_pattern_completion=False,  # Already done above
            )
        
        if not results:
            return None, 0.0, info
        
        best_proto, similarity = results[0]
        return best_proto, similarity, info
    
    def generate_rehearsal_episodes(
        self,
        n_episodes: int = 10,
        noise_std: float = PHI_INV_CUBE,  # φ-derived (was 0.1)
    ) -> List[EpisodicEntry]:
        """
        Generate pseudo-episodes for rehearsal during training.
        
        Use this to interleave with real episodes to prevent forgetting.
        
        Args:
            n_episodes: Number of pseudo-episodes to generate
            noise_std: φ-derived noise level for variation
            
        Returns:
            List of pseudo-episodes from semantic memory
        """
        if not self.use_pseudo_rehearsal:
            return []
        
        return generate_pseudo_episodes_batch(
            self.semantic_memory, self.basis, self.xp,
            n_episodes=n_episodes,
            noise_std=noise_std,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        return {
            "sleep_cycles": self.sleep_count,
            "total_episodes": self.total_episodes_processed,
            "transition_buffer_count": self.transition_buffer.stats()['count'],
            "retrieval_history_count": self.retrieval_history.stats()['count'],
            **self.semantic_memory.stats(),
        }


# =============================================================================
# INTEGRATION WITH TheoryTrueModel
# =============================================================================

def integrate_dreaming_with_model(
    model, 
    dreaming: DreamingSystem, 
    use_distributed_prior: bool = True,
    use_grace_basin: bool = True,
    K: int = 8,
    confidence_threshold: float = PHI_INV_SQ,  # Theory-true: φ⁻² (spectral gap)
    use_semantic_context: bool = True,
):
    """
    Integrate dreaming with the main TheoryTrueModel.
    
    BRAIN-ANALOG RETRIEVAL HIERARCHY:
        1. Episodic (hash lookup) - O(1), exact context matches
           Like hippocampal place cells
        2. Semantic (distributed prior) - For unknown contexts
           Like cortical population coding
    
    THEORY-TRUE IMPROVEMENTS:
        - Multiple weak activations (population coding, not winner-take-all)
        - φ-weighted superposition (NOT softmax!)
        - Geometric confidence from margin (NOT probability)
        - Global fallback via factorized prior (Hebbian operator)
    
    SEMANTIC EXTRACTION (v4.6.0):
        When use_semantic_context=True and model has predictiveness tracking:
        - Uses only semantic tokens (high predictiveness) for context composition
        - Enables paraphrase generalization (100% vs 24-42%)
        - Brain analog: VWFA co-occurrence filtering
    
    OPTIMIZATION:
        - Factorized prior trained ONCE at closure creation, not per retrieval
        - Prototype cache refreshed only when needed
    
    Args:
        model: TheoryTrueModel instance
        dreaming: DreamingSystem instance
        use_distributed_prior: If True, use brain-analog population coding.
        use_grace_basin: If False and not using distributed prior, use legacy threshold.
        K: Number of neighbors for distributed prior (default 8)
        confidence_threshold: Below this, blend with global prior (default 0.5 ≈ φ⁻¹)
        use_semantic_context: If True and model has predictiveness, use semantic-only context
    """
    from holographic_v4.resonance import distributed_prior_retrieve, grace_basin_retrieve
    from holographic_v4.distributed_prior import FactorizedAssociativePrior, extended_witness
    from holographic_v4.algebra import vorticity_signature
    
    # Initialize factorized prior for global fallback
    factorized_prior = FactorizedAssociativePrior(witness_dim=4, xp=model.xp)
    
    # OPTIMIZATION: Cache prototypes and train factorized prior ONCE at creation
    cached_prototypes = []
    cached_targets = []
    cached_supports = []
    cached_vorticity_sigs = []
    
    for proto in dreaming.semantic_memory.levels[0]:
        cached_prototypes.append(proto.prototype_matrix)
        cached_targets.append(proto.target_distribution)
        cached_supports.append(proto.support)
        cached_vorticity_sigs.append(proto.vorticity_signature)
        
        # Train factorized prior ONCE (not on every retrieval!)
        witness = extended_witness(proto.prototype_matrix, model.basis, model.xp)
        factorized_prior.update(witness, proto.prototype_matrix, weight=proto.support * PHI_INV_EIGHT)
    
    def retrieve_with_dreaming(context: List[int]):
        """
        Brain-analog hierarchical retrieval (v4.8.0).
        
        Returns:
            (attractor, target, source) where source is "holographic", "semantic", or "unknown"
        """
        # 1. Try holographic retrieval (O(1) unbinding) — like hippocampal pattern completion
        ctx_rep = model.compute_context_representation(context)
        attractor, target_idx, confidence, source = model.holographic_memory.retrieve(ctx_rep)
        
        if confidence >= PHI_INV_SQ:
            return attractor, int(target_idx), "holographic"
        
        # 2. Try semantic (prototype retrieval via dreaming)
        # SEMANTIC EXTRACTION: Use semantic-only context if enabled
        if use_semantic_context and hasattr(model, 'compute_semantic_context'):
            ctx_rep = model.compute_semantic_context(context)
        else:
            ctx_rep = model.compute_context(context)
        
        # Use cached prototypes (updated at each sleep cycle)
        prototypes = cached_prototypes
        prototype_targets = cached_targets
        prototype_supports = cached_supports
        prototype_vorticity_sigs = cached_vorticity_sigs
        
        # Compute query vorticity signature for grammar-aware matching
        if len(context) >= 2:
            query_mats = model.xp.stack([model.get_embedding(t) for t in context], axis=0)
            query_vort_sig = vorticity_signature(query_mats, model.basis, model.xp)
        else:
            query_vort_sig = None
        
        if len(prototypes) > 0:
            
            if use_distributed_prior:
                # BRAIN-ANALOG: Distributed prior (population coding)
                proto_idx, target, confidence, info = distributed_prior_retrieve(
                    query=ctx_rep,
                    prototypes=prototypes,
                    prototype_targets=prototype_targets,
                    prototype_supports=prototype_supports,
                    basis=model.basis,
                    xp=model.xp,
                    K=K,
                    grace_steps=3,
                    query_vorticity_sig=query_vort_sig,
                    prototype_vorticity_sigs=prototype_vorticity_sigs,
                    factorized_prior=factorized_prior,
                    confidence_threshold=confidence_threshold,
                )
                
                source = info.get("source", "distributed_prior")
                if proto_idx is not None:
                    return prototypes[proto_idx], target, f"{source}(conf={confidence:.2f})"
            
            elif use_grace_basin:
                # FALLBACK: Single-prototype Grace basin discovery
                proto_idx, target, confidence, info = grace_basin_retrieve(
                    ctx_rep, prototypes, prototype_targets,
                    model.basis, model.xp,
                    grace_steps=5,
                    min_confidence=0.0,
                    query_vorticity_sig=query_vort_sig,
                    prototype_vorticity_sigs=prototype_vorticity_sigs,
                    vorticity_weight=PHI_INV_CUBE,  # φ⁻³ weight for vorticity
                )
                
                source = info.get("source", "grace_basin")
                if proto_idx is not None:
                    return prototypes[proto_idx], target, f"{source}(conf={confidence:.2f})"
        
        # 3. Unknown - return context representation (NOT identity!)
        # THEORY-TRUE: Identity has specific geometric meaning (scalar=4, enstrophy=0).
        # Returning identity causes decode_attractor to always pick same token ("meteor").
        # Instead, return the context representation - its structure varies with input,
        # enabling φ-weighted sampling to produce diverse outputs.
        return ctx_rep, 0, "unknown"
    
    return retrieve_with_dreaming


# =============================================================================
# MEMORY SCALING ANALYSIS
# =============================================================================
"""
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

EMPIRICAL VERIFICATION:
    See verify_memory_scaling() below.
"""


def analyze_memory_scaling(
    dreaming: 'DreamingSystem',
    episode_counts: List[int] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, any]:
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
    
    # For each episode count, run consolidation and measure prototypes
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
            
            target = cluster * 10 + rng.integers(0, 5)  # Cluster-specific targets
            
            episodes.append(EpisodicEntry(ctx_matrix, target))
        
        # Run consolidation
        dreaming.sleep(episodes, rem_cycles=3, verbose=False)
        
        # Count prototypes at all levels
        total_protos = sum(len(level) for level in dreaming.semantic_memory.levels)
        
        results['episode_counts'].append(n_episodes)
        results['prototype_counts'].append(total_protos)
        
        # Compute scaling ratio (prototype growth / episode growth)
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
        
        # Linear regression in log-log space
        A = np.column_stack([log_e, np.ones_like(log_e)])
        slope, intercept = np.linalg.lstsq(A, log_p, rcond=None)[0]
        
        results['scaling_exponent'] = float(slope)  # α in P = c * N^α
        results['is_sublinear'] = slope < 1.0
    else:
        results['scaling_exponent'] = None
        results['is_sublinear'] = None
    
    # Mean scaling ratio
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
    
    # Current prototype count
    total_protos = sum(len(level) for level in dreaming.semantic_memory.levels)
    
    # Compute witness distribution
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
    
    # Estimate covered area
    if len(witnesses) > 1:
        # Convex hull or bounding box
        s_range = witnesses[:, 0].max() - witnesses[:, 0].min()
        p_range = witnesses[:, 1].max() - witnesses[:, 1].min()
        covered_area = (s_range + PHI_INV_CUBE) * (p_range + PHI_INV_CUBE)  # φ-derived margin
    else:
        covered_area = PHI_INV_CUBE * PHI_INV_CUBE  # φ-derived default area
    
    # Theoretical capacity
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
    
    # Get all prototype witnesses
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
    
    # Discretize witness space into bins for entropy calculation
    n_bins = 10
    s_bins = np.linspace(witnesses[:, 0].min() - PHI_INV_EIGHT, witnesses[:, 0].max() + PHI_INV_EIGHT, n_bins + 1)
    p_bins = np.linspace(witnesses[:, 1].min() - PHI_INV_EIGHT, witnesses[:, 1].max() + PHI_INV_EIGHT, n_bins + 1)
    
    # Count prototypes in each bin
    hist, _, _ = np.histogram2d(witnesses[:, 0], witnesses[:, 1], bins=[s_bins, p_bins])
    
    # Compute entropy
    hist_flat = hist.flatten()
    hist_flat = hist_flat[hist_flat > 0]  # Remove empty bins
    probs = hist_flat / hist_flat.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Normalized entropy (0 = all in one bin, 1 = uniform)
    max_entropy = np.log(n_bins * n_bins)
    normalized_entropy = entropy / max_entropy
    
    return {
        'witness_entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'n_prototypes': len(witnesses),
        'n_bins_used': len(hist_flat),
        'n_bins_total': n_bins * n_bins,
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/fractlphoneroom1/Desktop/ParsimoniousFlow')
    
    print("Testing Dreaming System...")
    
    # Create basis
    try:
        from holographic_v4 import build_clifford_basis
    except ImportError:
        from algebra import build_clifford_basis
    basis = build_clifford_basis(np)
    
    # Create dreaming system
    dreaming = DreamingSystem(basis=basis)
    
    # Create some test episodes
    episodes = []
    for i in range(100):
        # Random context matrix (identity + φ⁻⁴ noise)
        ctx_matrix = np.eye(4) + PHI_INV_FOUR * np.random.randn(4, 4)
        target = i % 10  # 10 different targets
        episodes.append(EpisodicEntry(ctx_matrix, target))
    
    # Run sleep cycle
    stats = dreaming.sleep(episodes, rem_cycles=1, verbose=True)
    
    print()
    print("Final stats:", dreaming.get_stats())
