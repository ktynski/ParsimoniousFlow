"""
Data Structures — Core Memory Representations

Defines all core data structures used throughout the dreaming system:
- EpisodicEntry: Single episodic memory entry
- CompressedEpisode: Delta-compressed episode
- SemanticPrototype: Consolidated semantic memory entry
- Schema: Abstract pattern from REM
- MetaSchema: Cluster of related schemas
- RetrievalRecord: Record of retrieval event
- TemporalTransition: Temporal transition between contexts
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# No constants needed - structures are pure data definitions


# =============================================================================
# EPISODIC MEMORY STRUCTURES
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
        from holographic_prod.core.algebra import reconstruct_from_coefficients
        
        if self.prototype_id < 0 or self.prototype_id >= len(prototypes):
            # Uncompressed: delta_coeffs IS the full matrix coefficients
            return reconstruct_from_coefficients(self.delta_coeffs, basis, xp)
        
        # Get prototype and add delta
        proto_matrix = prototypes[self.prototype_id].prototype_matrix
        delta_matrix = reconstruct_from_coefficients(self.delta_coeffs, basis, xp)
        return proto_matrix + delta_matrix


# =============================================================================
# SEMANTIC MEMORY STRUCTURES
# =============================================================================

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
# RETRIEVAL STRUCTURES
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


# =============================================================================
# SEQUENCE MEMORY STRUCTURES
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
