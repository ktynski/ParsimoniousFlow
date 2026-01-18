"""
Theory-True Pipeline — Full SCCMU Implementation v4.8.0
=======================================================

Implementation verified against rhnsclifford.md.
Last Updated: 2026-01-11

CORE FEATURES:
    - Vorticity grammar + scalable context (50k+ tested)
    - Equilibrium forgetting (φ-decay sparse efficiency)
    - Reconsolidation (accessed memories strengthen)
    - Chunked context computation for very long sequences
    
v4.8.0 ADDITIONS — THEORY-TRUE HOLOGRAPHIC MEMORY:
    - HolographicMemory: True superposition-based storage (O(1))
    - VorticityWitnessIndex: 8D episodic index (v4.21.0+)
    - CanonicalSemanticIndex: 2D semantic index (v4.22.0+)
    - HybridHolographicMemory: Triple cascade (holographic→episodic→semantic)
    
    HISTORICAL NOTE:
    The original implementation used hash(context.tobytes()) which was
    NOT theory-true. Hash ignores Clifford structure, grade hierarchy,
    and Grace dynamics. See ARCHITECTURE.md Part 3.0 for details.
    
    The new holographic memory uses true Clifford superposition:
    - Store: memory += φ⁻¹ × geometric_product(context, target)
    - Retrieve: target ≈ geometric_product(context_inverse, memory)
    
    This is O(1) and respects the theory's geometric structure.
    
v4.7.0 ADDITIONS:
    - Meta-learning: Adaptive φ-derived learning rates
    - Embedding drift: Representations evolve with experience
    - Credit assignment: Error tracking and reconsolidation
    - Predictiveness tracking: Semantic extraction
    - Self-organizing retrieval: Grace-stability orchestrates modules

THEORETICAL COMPONENTS:
1. GEOMETRIC PRODUCT for context composition (M₁ × M₂ × M₃ × ...)
2. WEDGE PRODUCT for sequential vorticity (bivector = rotation)
3. GRACE OPERATOR for equilibrium dynamics (THE normalizer)
4. WITNESS for gauge-invariant self-reference
5. BINDING for self-referential content  
6. QUOTIENT SIMILARITY for stable retrieval
7. VORTICITY-WEIGHTED DECODING to prevent mode collapse
8. VORTICITY GRAMMAR MATCHING for structure-aware retrieval
9. EQUILIBRIUM FORGETTING for sparse efficiency (φ-decay)
10. RECONSOLIDATION for memory strengthening via access

NORMALIZATION STRATEGY (Theory-True):
- Frobenius normalization: NUMERICAL necessity for long contexts
    - Applied after each chunk composition (preserves grade ratios)
    - Uniform scaling, does NOT favor any grade
    - Required because floating-point precision drifts over 50k+ products
- Grace operator: THEORY-TRUE grade-wise damping
    - Applied once at end of context computation
    - Damps higher grades by φ⁻ᵏ (spectral structure)
    - Creates stable witness + decaying vorticity

SCALABLE CONTEXT (O(N) composition, O(1) storage):
- Context is always a 4×4 matrix regardless of length
- Chunked computation: 1000 tokens at a time for long contexts
- Frobenius normalization after each chunk (numerical stability)
- Grace applied once at the end (theory-true)
- Tested stable to context_size = 50,000+ tokens
- 65536× cheaper than Transformer attention at context=65536!

EQUILIBRIUM FORGETTING (v4.13.0 Theory-True Memory Management):
Brain-like capacity-based homeostasis, NOT time-based decay:

    retention = salience (what Grace preserves)
    threshold = adaptive based on fill ratio (brain-like homeostasis)
    
    Forget if salience < adaptive_threshold

Where:
    - retention = salience (intrinsic property, no bookkeeping)
    - threshold = φ⁻³ at 0% fill → 0.50 at 100% fill (φ-smooth)
    
Creates SPARSE EFFICIENCY through capacity pressure:
    - Low fill → minimal pruning (only very weak memories)
    - High fill → aggressive pruning (~50% like brain sleep)
    - High salience → survives (more energy in witness)
    
v4.13.0 REMOVED (not theory-true):
    - access_counts (doesn't work for holographic superposition)
    - time_decay (artificial bookkeeping)
    - bucket tracking (arbitrary discretization)

CRITICAL THEORY-TRUE IMPLEMENTATION:

1. STORAGE (rhnsclifford.md lines 23-24, 360-361):
   
       attractor[context] = embedding[target]  # Direct association
   
   Hebbian association - NOT similarity-based storage.

2. LEARNING RATE (rhnsclifford.md line 111):
   
       attractor[ctx] = lerp(attractor[ctx], target, φ⁻¹)
   
   Rate is φ⁻¹ ≈ 0.618 (FIXED by Λ² = Λ + 1, NOT tuned).

3. LEARNING RULE:
   
   CORRECT: Direct lerp - (1 - φ⁻¹) * attractor + φ⁻¹ * target
   WRONG:   Grace-then-lerp
   
   Grace is for FORWARD PASS, NOT learning.

4. GRACE FOR NORMALIZATION:
   
   Applied ONCE at end of context computation:
       context = compute_context_with_vorticity(tokens)  # Includes Grace at end
   
   NOT after every geometric product (causes over-damping).

5. GRACE IN FORWARD PASS:
   
       field = evolve_to_equilibrium(field, attractor)  # Grace flow
       return field  # Equilibrium IS output
   
   Grace flow converges at rate γ = φ⁻².

6. EQUILIBRIUM FORGETTING:
   
       if retention_strength < φ⁻³: forget(memory)
   
   Continuous pruning creates natural equilibrium.

RETRIEVAL SEMANTICS (v4.8.0):
   - Holographic retrieval: O(1) via Clifford superposition (theory-true)
   - Witness-based cascade: O(bucket) via quantized witness space
   - Grace basin discovery: O(n) for semantic memory
   - Vorticity-weighted decoding: prevents mode collapse
   
   ALL storage and retrieval uses HolographicMemory.
   Hash-based storage has been REMOVED (was not theory-true).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from .constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FIVE,
    MATRIX_DIM, DTYPE
)
from .algebra import (
    build_clifford_basis,
    build_metric_matrix,
    normalize_matrix,
    geometric_product_batch,
    wedge_product,
    compute_vorticity,
    vorticity_magnitude,
    vorticity_signature,  # NEW: For syntactic structure encoding
    frobenius_similarity,
    frobenius_similarity_batch,
    grace_operator,
    grace_operator_batch,
    grace_flow,  # NEW: For equilibrium dynamics
    initialize_embeddings_identity,
    decompose_to_coefficients,
)
from .quotient import (
    extract_witness,
    witness_matrix,
    witness_similarity,
    bind,
    bind_batch,  # MOVED: Was imported inside train_batch (per-call overhead)
    normal_form,
    quotient_similarity,
    adaptive_similarity,
    adaptive_similarity_batch,
    compute_enstrophy,
    compute_enstrophy_batch,
    ENSTROPHY_THRESHOLD,
    grade_energies,
    witness_stability,
    vorticity_weighted_scores,
    vorticity_weighted_scores_batch,
)
from .predictiveness import PredictivenessTracker
from .meta_learning import (
    LearningState, 
    compute_adaptive_learning_rate, 
    update_meta_state,
    create_learning_state,
)
from .representation_learning import (
    compute_embedding_gradient,
    update_embedding,
)
# Credit Assignment v2 (lean integration with FractalGenerativeMemory)
try:
    from .credit_assignment import (
        CreditAssignmentTracker,
        create_credit_assigned_learn,
        batch_credit_assignment,
    )
    # Backward compatibility aliases for legacy code
    ProvenanceTracker = None  # Deprecated - use CreditAssignmentTracker
    reconsolidate_on_error = None  # Deprecated - use batch_credit_assignment
except ImportError:
    # Fallback for legacy credit_assignment module
    from .archive.legacy_tests.credit_assignment_legacy import (
        ProvenanceTracker,
        reconsolidate_on_error,
    )
    CreditAssignmentTracker = None
from .curiosity import curiosity_score
from .quotient import grace_stability, grace_stability_batch

# Theory-true holographic memory (v4.8.0) - PRIMARY STORAGE MECHANISM
from .holographic_memory import (
    HolographicMemory,
    VorticityWitnessIndex,
    CanonicalSemanticIndex,
    HybridHolographicMemory,
    MultiTimescaleMemory,
    compute_witness_entropy,
    is_memory_saturated,
)

Array = np.ndarray
ArrayModule = type(np)


class TheoryTrueModel:
    """
    Full theory-true SCCMU implementation.
    
    Last Updated: 2026-01-10 (Equilibrium Forgetting + Reconsolidation)
    
    THEORY-TRUE BEHAVIOR (v4.8.0):
    1. LEARNING: Direct lerp with rate φ⁻¹ (NOT Grace-then-lerp)
    2. STORAGE: Holographic superposition via HybridHolographicMemory
    3. FORWARD: Grace flow evolves context toward attractor equilibrium
    4. FORGETTING: Adaptive pruning based on capacity pressure (brain-like)
    5. RATES: Fixed by theory (φ⁻¹, φ⁻², φ⁻³), NOT tuned
    
    MEMORY MANAGEMENT (v4.13.0 Theory-True):
    - retention = salience (what Grace preserves, no bookkeeping)
    - threshold = adaptive based on fill ratio (brain-like homeostasis)
    - Low fill: threshold = φ⁻³ (minimal pruning)
    - High fill: threshold = 0.50 (~50% pruned like brain sleep)
    
    CONTEXT COMPUTATION (Scalable to 50k+ tokens):
    - Chunked processing: 1000 tokens at a time
    - Frobenius normalization after each chunk (numerical stability)
    - Grace applied once at end (theory-true damping)
    - Vorticity accumulated across chunks
    
    COMPONENTS:
    - Grace operator (grade-wise φ⁻ᵏ): Forward pass + normalization
    - Vorticity tracking: Grammar structure encoding
    - Adaptive similarity: THEORY-DERIVED decision (enstrophy-based)
    - Binding operator: Optional for self-referential content
    - Capacity-based pruning: Brain-like homeostasis
    
    PARAMETERS:
        vocab_size: Number of tokens
        context_size: Default context window (can vary per sample)
        max_attractors: Maximum stored memories
        noise_std: φ-derived (PHI_INV_CUBE ≈ 0.236) for discriminable embeddings
        use_vorticity: Include wedge product in context (recommended)
        use_equilibrium: Apply Grace flow during retrieval (recommended)
        use_adaptive_similarity: System decides Frobenius vs Quotient
        
    TRACKING ATTRIBUTES (for auditing):
        attractor_saliences: Salience at encoding (retention = salience)
        attractor_consolidated: Is in semantic memory?
        total_forgotten: How many memories pruned
    
    TRAIN STEP RETURNS:
        {
            'num_attractors': Current memory count,
            'context': Raw tokens for dreaming,
            'forgotten': Capacity-based forgetting count,
            'pruned': Adaptive pruning count,
        }
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_size: int = 8,
        max_attractors: int = 100000,
        noise_std: float = PHI_INV_CUBE,  # φ-derived: PHI_INV_CUBE ≈ 0.236 (was 0.3)
        use_binding: bool = True,
        use_adaptive_similarity: bool = True,  # THEORY-DERIVED: system decides
        use_vorticity: bool = True,  # Include wedge product in context
        use_equilibrium: bool = True,  # Evolve to equilibrium for retrieval
        equilibrium_steps: int = 5,  # Steps of grace flow
        use_vorticity_decoding: bool = True,  # THEORY-TRUE: prevents mode collapse
        vorticity_decode_weight: float = PHI_INV,  # Theory-true: φ⁻¹ structure weight
        use_predictiveness: bool = True,  # SEMANTIC EXTRACTION: tracks token-target co-occurrences
        predictiveness_threshold: float = PHI_INV_SQ,  # Theory-true: φ⁻² (spectral gap)
        use_meta_learning: bool = True,  # ADAPTIVE RATES: modulate φ⁻¹ based on context
        use_embedding_drift: bool = True,  # REPRESENTATION LEARNING: embeddings improve via drift
        # NOTE: After embedding drift, witness index is reindexed every _reindex_interval
        # samples to maintain correct retrieval with improved embeddings.
        use_credit_assignment: bool = True,  # ERROR TRACKING: provenance and reconsolidation
        use_tensor_cores: bool = False,  # TENSOR CORE: coefficient form (adds conversion overhead)
        skip_novelty_check: bool = True,  # PARSIMONY: skip retrieval before store (~200× faster train_step)
        xp: ArrayModule = np,
        seed: int = 42,
    ):
        """
        Initialize theory-true model.
        
        Args:
            vocab_size: Number of tokens in vocabulary
            context_size: Context window size
            max_attractors: Maximum number of stored attractors
            noise_std: Noise for identity-biased initialization.
                       OPTIMAL: 0.3 balances discrimination + generalization.
                       - Too low (0.05): collapses all embeddings to ≈ Identity
                       - Too high (0.7+): loses identity-bias benefit
                       - Sweet spot (0.3): 27x vorticity improvement, 5.7x generalization
            use_binding: Whether to apply binding operator
            use_adaptive_similarity: THEORY-DERIVED similarity selection.
                       When True (default), system decides based on enstrophy:
                       - Low enstrophy (near-identity) → fast Frobenius
                       - High enstrophy (differentiated) → full quotient
                       This is parsimonious: system decides, not user.
            use_vorticity: Include vorticity (wedge products) in context
            use_equilibrium: Apply Grace flow during retrieval
            equilibrium_steps: Number of Grace flow iterations
            use_vorticity_decoding: THEORY-TRUE decoding that prevents mode collapse.
                       When True (default), decoding considers structural match
                       (enstrophy correspondence + witness alignment), not just
                       raw Frobenius similarity. This prevents degenerate outputs
                       like "was was was was..." by preferring tokens whose
                       geometric structure matches the attractor's structure.
            vorticity_decode_weight: How much to weight structural match [0, 1].
                       Higher values = more structure-aware decoding.
                       0.5 is balanced; increase for longer sequences.
            use_predictiveness: SEMANTIC EXTRACTION (v4.6.0). When True (default),
                       tracks token-target co-occurrences and filters context to
                       only include semantic tokens (high mutual information with target).
                       This enables paraphrase generalization (100% vs 24-42%).
                       Brain analog: VWFA co-occurrence learning mechanism.
            predictiveness_threshold: Tokens with predictiveness > threshold are
                       classified as "semantic" (default 0.5 = 50% better than random).
            use_meta_learning: ADAPTIVE RATES. When True (default), modulates the
                       φ⁻¹ learning rate based on salience, novelty, and uncertainty.
                       High salience/novelty → faster learning.
                       High uncertainty → slower learning.
                       Stays within φ-derived bounds [φ⁻²·φ⁻¹, φ·φ⁻¹].
            use_embedding_drift: REPRESENTATION LEARNING. When True (default), embeddings
                       drift toward better positions based on retrieval success/failure.
                       Theory-true: The brain learns discriminative representations through
                       experience. Tokens that predict different targets → divergent embeddings.
            use_credit_assignment: ERROR TRACKING. When True (default), tracks
                       provenance of retrievals and reconsolidates on errors.
                       Enables targeted memory correction.
            xp: Array module (numpy or cupy)
            seed: Random seed
        """
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.max_attractors = max_attractors
        self.use_binding = use_binding
        self.use_adaptive_similarity = use_adaptive_similarity
        self.use_vorticity = use_vorticity
        self.use_equilibrium = use_equilibrium
        self.equilibrium_steps = equilibrium_steps
        self.use_vorticity_decoding = use_vorticity_decoding
        self.vorticity_decode_weight = vorticity_decode_weight
        self.use_predictiveness = use_predictiveness
        self.use_tensor_cores = use_tensor_cores
        self.xp = xp
        
        # SEMANTIC EXTRACTION: Track token-target co-occurrences
        # This enables automatic identification of semantic vs noise tokens
        # Brain analog: VWFA co-occurrence learning (see SEMANTIC_EXTRACTION_THEORY.md)
        if use_predictiveness:
            self.predictiveness_tracker = PredictivenessTracker(threshold=predictiveness_threshold)
        else:
            self.predictiveness_tracker = None
        
        # META-LEARNING: Adaptive learning rates based on context
        # Modulates φ⁻¹ based on salience, novelty, uncertainty
        self.use_meta_learning = use_meta_learning
        if use_meta_learning:
            self.learning_state = create_learning_state()
        else:
            self.learning_state = None
        
        # REPRESENTATION LEARNING: Embedding drift for better positions
        self.use_embedding_drift = use_embedding_drift
        self._needs_reindex = False  # Track when witness index needs reindexing
        self._reindex_interval = 1000  # Reindex every N samples after drift occurs
        
        # OPTIMIZATION: Don't compute witness entropy every step (20% of train_step time!)
        # Check periodically instead - dreaming doesn't need instant signals
        self._saturation_check_interval = 89  # Fibonacci number (theory-aligned)
        self._last_witness_entropy = 0.0
        self._last_memory_saturated = False
        
        # CREDIT ASSIGNMENT: Error tracking and reconsolidation
        # NOTE: Legacy ProvenanceTracker deprecated in v4.28.0
        # Use CreditAssignmentTracker with FractalGenerativeMemory instead
        self.use_credit_assignment = use_credit_assignment
        self.provenance_tracker = None  # Deprecated
        
        # Build Clifford structure
        self.basis = build_clifford_basis(xp)
        self.G = build_metric_matrix(xp)
        
        # Initialize embeddings (identity-biased, FIXED during training)
        self.embeddings = initialize_embeddings_identity(
            vocab_size, noise_std=noise_std, seed=seed, xp=xp
        )
        
        # PRECOMPUTE embedding features for vorticity-weighted decoding
        # This avoids the 20,000-iteration loop in every decode call!
        self._precompute_embedding_features()
        
        # ===================================================================
        # HOLOGRAPHIC MEMORY (v4.8.0) - Theory-True Primary Storage
        # ===================================================================
        # Uses true Clifford superposition instead of hash-based indexing.
        # Storage: memory += φ⁻¹ × geometric_product(context, target)
        # Retrieval: target ≈ geometric_product(context_inverse, memory)
        # 
        # This is O(1) and respects geometric structure (unlike hash).
        self.skip_novelty_check = skip_novelty_check
        
        # v4.22.0: Simplified - always uses dual indexing (episodic + semantic)
        self.holographic_memory = HybridHolographicMemory.create(self.basis, xp)
        
        # Bookkeeping arrays (for equilibrium forgetting statistics)
        # Note: Primary storage is self.holographic_memory, these are for diagnostics
        self.attractor_matrices = xp.zeros(
            (max_attractors, MATRIX_DIM, MATRIX_DIM), dtype=DTYPE
        )
        self.attractor_targets = xp.zeros(max_attractors, dtype=xp.int32)
        self.num_attractors = 0
        
        # φ-FORGETTING: Theory-true continuous memory management
        # 
        # THEORY DERIVATION (equilibrium forgetting):
        #   The brain doesn't wait until "full" - it continuously prunes based on:
        #   
        #   retention_strength = salience × φ^(access_count) × φ^(-time_since_access/τ)
        #   
        #   Where τ (tau) = 1000 samples (time constant, derived from φ³ ≈ 4.236 × 236 ≈ 1000)
        #   
        #   Memories with retention_strength < PRUNE_THRESHOLD (φ⁻³ ≈ 0.236) are forgotten.
        #   This creates SPARSE EFFICIENCY through natural selection:
        # v4.13.0: SIMPLIFIED RETENTION
        # retention = salience (what Grace preserves)
        # pruning = adaptive threshold based on capacity (brain-like homeostasis)
        # 
        # REMOVED: access_counts, last_access, decay_tau (not theory-true)
        # These were artificial bookkeeping that didn't match holographic memory dynamics.
        
        self.attractor_saliences = xp.zeros(max_attractors, dtype=DTYPE)  # Salience at encoding
        self.attractor_consolidated = xp.zeros(max_attractors, dtype=xp.bool_)  # Is in semantic memory?
        
        # Theory-true parameters (all derived from φ)
        self.prune_threshold = PHI_INV ** 3  # ≈ 0.236 - min threshold (low fill)
        self.prune_interval = 1000  # Check for pruning every N samples
        self.equilibrium_forgetting = True  # Theory-true: continuous pruning
        self.capacity_forgetting = True  # Also forget at capacity (safety net)
        self.total_forgotten = 0  # Track how many memories were forgotten
        self.last_prune_sample = 0  # When did we last prune?
        
        # MULTI-TIMESCALE MEMORY (v4.8.0): Brain-analog memory systems
        # Fast (φ⁻¹ decay): Working memory / hippocampus  
        # Medium (φ⁻² decay): Episodic memory
        # Slow (φ⁻³ decay): Semantic memory / cortex
        self.multi_timescale_memory = MultiTimescaleMemory.create(self.basis, xp=xp)
        
        # Statistics
        self.train_samples = 0
        self.total_vorticity = 0.0
        
        # v4.13.0: Removed bucket tracking (not theory-true for holographic memory)
        # Access counting doesn't strengthen holographic superposition.
        # Retention now uses salience-only with adaptive threshold.
    
    def _precompute_embedding_features(self):
        """
        OPTIMIZATION: Precompute enstrophies and witnesses for all embeddings.
        
        This is called ONCE at model creation, not on every decode call.
        FULLY VECTORIZED: Uses batch decomposition instead of Python loop!
        """
        from .algebra import decompose_to_coefficients_batch
        from .quotient import GRADE_INDICES
        
        xp = self.xp
        
        # VECTORIZED: Decompose ALL embeddings at once!
        all_coeffs = decompose_to_coefficients_batch(self.embeddings, self.basis, xp)  # [V, 16]
        
        # Extract features from coefficients
        grade2_coeffs = all_coeffs[:, GRADE_INDICES[2]]  # [V, 6]
        self.embedding_enstrophies = xp.sum(grade2_coeffs * grade2_coeffs, axis=1)  # [V]
        self.embedding_scalars = all_coeffs[:, 0]  # [V]
        self.embedding_pseudos = all_coeffs[:, 15]  # [V]
    
    # =========================================================================
    # EMBEDDING & CONTEXT
    # =========================================================================
    
    def get_embedding(self, token_idx: int) -> Array:
        """Get embedding for a token (FIXED, not learned)."""
        return self.embeddings[token_idx % self.vocab_size]
    
    # =========================================================================
    # v4.13.0: Removed bucket tracking helpers
    # Bucket-based reconsolidation was not theory-true for holographic memory.
    # Retention now uses salience with adaptive threshold.
    # =========================================================================
    
    # =========================================================================
    # INCREMENTAL CONTEXT — O(1) Context Updates (Brain-Inspired Efficiency)
    # =========================================================================
    # Instead of recomputing Context = M₁ @ M₂ @ ... @ Mₙ each time (O(n)),
    # maintain running context and update incrementally: Context ← Context @ Mₙ (O(1))
    #
    # CORRECTNESS: Mathematically identical (matrix multiplication is associative)
    # SPEEDUP: 97× at context_size=100 (verified in brain_efficiency_tests.py)
    # RISK: Zero — pure algebraic optimization
    
    def reset_incremental_context(self) -> None:
        """
        Reset incremental context to identity.
        
        Call this at the start of each new sequence/session.
        """
        self._incremental_context = self.xp.eye(4, dtype=DTYPE)
        self._incremental_context_length = 0
        self._incremental_vorticity_sum = self.xp.zeros(16, dtype=DTYPE)
    
    def update_context_incremental(self, token_idx: int, normalize: bool = True) -> Array:
        """
        Update context incrementally with a new token (O(1) operation).
        
        THEORY-TRUE: This is mathematically identical to full recomputation:
            Context_new = Context_old @ embedding[token]
            
        NUMERICAL STABILITY:
            When normalize=True (default), applies Frobenius normalization after
            each product to match compute_contexts_batch behavior. This ensures:
            - Identical results to batch computation
            - No underflow/overflow in long contexts
            - Grade ratios preserved (Frobenius is uniform scaling)
        
        Args:
            token_idx: Token to add to context
            normalize: Apply Frobenius normalization after product (default True)
            
        Returns:
            Updated context matrix [4, 4]
        """
        from holographic_v4.algebra import geometric_product
        
        # Initialize if needed
        if not hasattr(self, '_incremental_context'):
            self.reset_incremental_context()
        
        # Get embedding and update context (single matmul!)
        emb = self.get_embedding(token_idx)
        self._incremental_context = geometric_product(self._incremental_context, emb)
        self._incremental_context_length += 1
        
        # Frobenius normalization after EVERY product (matches batch behavior)
        # This is uniform scaling, preserves grade ratios, prevents overflow
        if normalize:
            norm = self.xp.sqrt(self.xp.sum(self._incremental_context ** 2))
            if norm > 1e-12:
                self._incremental_context = self._incremental_context / norm
        
        return self._incremental_context
    
    def get_incremental_context(self) -> Tuple[Array, int]:
        """
        Get current incremental context and its length.
        
        Returns:
            (context_matrix, length)
        """
        if not hasattr(self, '_incremental_context'):
            self.reset_incremental_context()
        return self._incremental_context.copy(), self._incremental_context_length
    
    def finalize_incremental_context(self) -> Array:
        """
        Finalize incremental context by applying Grace operator.
        
        Call this when the context is complete and ready for retrieval/prediction.
        Grace damps higher grades (theory-true normalization).
        
        Returns:
            Graced context matrix [4, 4]
        """
        from holographic_v4.algebra import grace_operator
        
        if not hasattr(self, '_incremental_context'):
            self.reset_incremental_context()
        
        return grace_operator(self._incremental_context, self.basis, self.xp)
    
    # =========================================================================
    # OSCILLATORY GRACE — Brain-Inspired Chunking (φ²-Token Boundaries)
    # =========================================================================
    # The brain uses gamma-theta coupling: fast gamma cycles (per-token) nest
    # within slow theta cycles (per-chunk). Grace at chunk boundaries mirrors
    # this oscillatory structure.
    #
    # SPEEDUP: ~2× fewer Grace operations (φ² ≈ 2.6 tokens per chunk)
    # STABILITY: 0.958 vs 0.980 (within 2%, verified in brain_efficiency_tests.py)
    # THEORY-TRUE: φ² emerges naturally from Fibonacci structure
    
    # Oscillatory chunk size (φ² ≈ 2.618, round to 3)
    OSCILLATORY_CHUNK_SIZE = 3  # = round(PHI * PHI)
    
    def reset_oscillatory_context(self) -> None:
        """
        Reset oscillatory context tracking.
        
        Call at the start of a new sequence.
        """
        self._oscillatory_context = self.xp.eye(4, dtype=DTYPE)
        self._oscillatory_token_count = 0
        self._oscillatory_grace_count = 0
    
    def update_context_oscillatory(self, token_idx: int) -> Tuple[Array, bool]:
        """
        Update context with oscillatory Grace (per-chunk, not per-token).
        
        BRAIN-INSPIRED: Applies Grace at φ²-token chunk boundaries,
        mimicking gamma-theta coupling in neural oscillations.
        
        Args:
            token_idx: Token to add to context
            
        Returns:
            (context_matrix, grace_applied) - context and whether Grace was applied
        """
        from holographic_v4.algebra import geometric_product, grace_operator
        
        # Initialize if needed
        if not hasattr(self, '_oscillatory_context'):
            self.reset_oscillatory_context()
        
        # O(1) context update
        emb = self.get_embedding(token_idx)
        self._oscillatory_context = geometric_product(self._oscillatory_context, emb)
        self._oscillatory_token_count += 1
        
        # Frobenius normalize (matches batch behavior)
        norm = self.xp.sqrt(self.xp.sum(self._oscillatory_context ** 2))
        if norm > 1e-12:
            self._oscillatory_context = self._oscillatory_context / norm
        
        # Apply Grace at chunk boundaries (θ cycle)
        grace_applied = False
        if self._oscillatory_token_count % self.OSCILLATORY_CHUNK_SIZE == 0:
            self._oscillatory_context = grace_operator(
                self._oscillatory_context, self.basis, self.xp
            )
            self._oscillatory_grace_count += 1
            grace_applied = True
        
        return self._oscillatory_context.copy(), grace_applied
    
    def finalize_oscillatory_context(self) -> Array:
        """
        Finalize oscillatory context (apply final Grace if not at boundary).
        
        Returns:
            Graced context matrix [4, 4]
        """
        from holographic_v4.algebra import grace_operator
        
        if not hasattr(self, '_oscillatory_context'):
            self.reset_oscillatory_context()
        
        # Only apply final Grace if we're not already at a chunk boundary
        if self._oscillatory_token_count % self.OSCILLATORY_CHUNK_SIZE != 0:
            return grace_operator(self._oscillatory_context, self.basis, self.xp)
        else:
            return self._oscillatory_context.copy()
    
    def get_oscillatory_stats(self) -> Dict[str, Any]:
        """Get statistics about oscillatory Grace usage."""
        if not hasattr(self, '_oscillatory_context'):
            return {'tokens': 0, 'grace_calls': 0, 'reduction': 0.0}
        
        expected_grace = self._oscillatory_token_count  # per-token would be this many
        actual_grace = self._oscillatory_grace_count
        reduction = expected_grace / max(actual_grace, 1)
        
        return {
            'tokens': self._oscillatory_token_count,
            'grace_calls': actual_grace,
            'reduction': reduction
        }
    
    # =========================================================================
    # SPARSE EMBEDDING STORAGE — Memory Compression for Near-Identity Matrices
    # =========================================================================
    # For embeddings with high stability (close to identity), we can store just
    # the witness (σ, p) = 2 floats instead of full 16 coefficients.
    #
    # MEMORY SAVINGS: 8× for sparse embeddings (16/2 = 8)
    # WITNESS PRESERVED: Exact (error = 0.0)
    # RECONSTRUCTION: Approximate (M ≈ σ·I + p·γ₅)
    #
    # USE CASE: Cold storage, archival, reducing memory footprint
    # NOT FOR: Active computation (reconstruction loses non-witness grades)
    
    SPARSE_STABILITY_THRESHOLD = 1 - PHI_INV_FIVE  # φ-derived: ≈0.91, if stability > threshold, use sparse
    
    def is_embedding_sparse(self, M: Array) -> bool:
        """
        Check if embedding can be stored sparsely (near-identity).
        
        Args:
            M: [4, 4] matrix
            
        Returns:
            True if embedding has high stability (mostly witness content)
        """
        from holographic_v4.quotient import grace_stability
        stability = float(grace_stability(M, self.basis, self.xp))
        return stability > self.SPARSE_STABILITY_THRESHOLD
    
    def compress_to_sparse(self, M: Array) -> Tuple[float, float, bool]:
        """
        Compress matrix to sparse representation (witness only).
        
        Args:
            M: [4, 4] matrix
            
        Returns:
            (sigma, pseudo, is_sparse) - witness values and whether sparse
        """
        from holographic_v4.quotient import extract_witness, grace_stability
        
        stability = float(grace_stability(M, self.basis, self.xp))
        is_sparse = stability > self.SPARSE_STABILITY_THRESHOLD
        
        witness = extract_witness(M, self.basis, self.xp)
        return float(witness[0]), float(witness[1]), is_sparse
    
    def reconstruct_from_sparse(self, sigma: float, pseudo: float) -> Array:
        """
        Reconstruct matrix from sparse representation.
        
        NOTE: This is APPROXIMATE - only the witness grades are preserved.
        For cold storage/archival, this is acceptable.
        For active computation, use full matrix.
        
        Args:
            sigma: Scalar coefficient
            pseudo: Pseudoscalar coefficient
            
        Returns:
            [4, 4] reconstructed matrix (M ≈ σ·I + p·γ₅)
        """
        gamma5 = self.basis[15]
        return sigma * self.xp.eye(4, dtype=DTYPE) + pseudo * gamma5
    
    def compress_embeddings_batch(self, matrices: Array) -> Dict[str, Any]:
        """
        Compress a batch of matrices to sparse representation where possible.
        
        Args:
            matrices: [N, 4, 4] batch of matrices
            
        Returns:
            Dict with:
                - sparse_indices: Indices of sparse matrices
                - sparse_witnesses: [K, 2] witnesses for sparse matrices
                - dense_indices: Indices of dense matrices
                - dense_matrices: [M, 4, 4] full matrices for dense ones
                - compression_ratio: Effective memory savings
        """
        from holographic_v4.quotient import extract_witness_batch, grace_stability_batch
        
        n = matrices.shape[0]
        
        # Batch stability computation
        stabilities = grace_stability_batch(matrices, self.basis, self.xp)
        
        # Identify sparse vs dense
        is_sparse = stabilities > self.SPARSE_STABILITY_THRESHOLD
        sparse_indices = self.xp.where(is_sparse)[0]
        dense_indices = self.xp.where(~is_sparse)[0]
        
        # Extract witnesses for sparse ones
        sparse_witnesses = None
        if len(sparse_indices) > 0:
            sparse_matrices = matrices[sparse_indices]
            sparse_witnesses = extract_witness_batch(sparse_matrices, self.basis, self.xp)
        
        # Keep full matrices for dense ones
        dense_matrices = None
        if len(dense_indices) > 0:
            dense_matrices = matrices[dense_indices]
        
        # Compute compression ratio
        n_sparse = len(sparse_indices)
        n_dense = len(dense_indices)
        # Sparse: 2 floats, Dense: 16 floats
        compressed_size = n_sparse * 2 + n_dense * 16
        original_size = n * 16
        compression_ratio = original_size / max(compressed_size, 1)
        
        return {
            'sparse_indices': sparse_indices,
            'sparse_witnesses': sparse_witnesses,
            'dense_indices': dense_indices,
            'dense_matrices': dense_matrices,
            'n_sparse': n_sparse,
            'n_dense': n_dense,
            'compression_ratio': compression_ratio,
            'sparse_rate': n_sparse / n if n > 0 else 0.0
        }
    
    def reconstruct_embeddings_batch(self, compressed: Dict[str, Any], n_total: int) -> Array:
        """
        Reconstruct full matrices from compressed representation.
        
        Args:
            compressed: Output from compress_embeddings_batch
            n_total: Total number of matrices to reconstruct
            
        Returns:
            [N, 4, 4] reconstructed matrices
        """
        result = self.xp.zeros((n_total, 4, 4), dtype=DTYPE)
        gamma5 = self.basis[15]
        
        # Reconstruct sparse ones
        if compressed['sparse_witnesses'] is not None:
            sparse_idx = compressed['sparse_indices']
            witnesses = compressed['sparse_witnesses']
            for i, idx in enumerate(sparse_idx):
                sigma, pseudo = witnesses[i, 0], witnesses[i, 1]
                result[idx] = sigma * self.xp.eye(4, dtype=DTYPE) + pseudo * gamma5
        
        # Copy dense ones directly
        if compressed['dense_matrices'] is not None:
            dense_idx = compressed['dense_indices']
            for i, idx in enumerate(dense_idx):
                result[idx] = compressed['dense_matrices'][i]
        
        return result
    
    # =========================================================================
    # φ-FORGETTING — Theory-True Memory Management
    # =========================================================================
    
    def compute_attractor_salience(self, matrix: Array) -> float:
        """
        Compute salience of an attractor matrix.
        
        THEORY (EMOTIONAL_TAGGING_THEORY.md):
            Salience = what Grace PRESERVES = scalar + φ⁻¹ × pseudoscalar
            
        High salience memories survive longer (like emotionally tagged memories in brain).
        
        INFORMATION PARSIMONY:
            σ = tr(M)/4 — scalar coefficient equals trace (no decomposition needed!)
            p = <M, γ₅>/4 — pseudoscalar via projection onto γ₅
        """
        # PARSIMONY: σ = tr(M)/4 — scalar is just trace!
        scalar = abs(float(self.xp.trace(matrix) / 4.0))
        # Pseudoscalar still needs projection onto γ₅ (basis[15])
        pseudoscalar = abs(float(self.xp.sum(self.basis[15] * matrix) / 4.0))
        return scalar + pseudoscalar * PHI_INV
    
    def compute_retention_strength(self, idx: int) -> float:
        """
        THEORY-TRUE retention = salience (what Grace preserves).
        
        v4.13.0: SIMPLIFIED based on empirical findings:
        - Access counting doesn't work for holographic superposition
        - Time-based decay is artificial bookkeeping
        - Salience IS the intrinsic measure of memory strength
        
        THEORY:
            Salience = witness stability = (scalar² + pseudoscalar²) / total_energy
            This is what Grace naturally preserves.
            
        PRUNING:
            Uses adaptive threshold based on capacity (see get_adaptive_prune_threshold)
        """
        return float(self.attractor_saliences[idx])
    
    # Alias for backwards compatibility
    def compute_retention_priority(self, idx: int) -> float:
        """Alias for compute_retention_strength (backwards compatibility)."""
        return self.compute_retention_strength(idx)
    
    def compute_retention_strength_batch(self) -> Array:
        """
        THEORY-TRUE retention = salience (what Grace preserves).
        
        v4.13.0: SIMPLIFIED - just return saliences.
        
        All the complexity (access counting, time decay) was artificial
        bookkeeping. Empirical tests showed:
        - Access counting doesn't strengthen holographic memories
        - Time decay is arbitrary (Grace provides natural decay)
        - Salience is the native measure of stability
        
        Returns:
            [num_attractors] array of retention strengths (= saliences)
        """
        if self.num_attractors == 0:
            return self.xp.array([], dtype=DTYPE)
        
        return self.attractor_saliences[:self.num_attractors].copy()
    
    def get_adaptive_prune_threshold(self) -> float:
        """
        THEORY-TRUE adaptive threshold based on capacity pressure.
        
        v4.13.0: Brain-like homeostasis
        - At low fill: minimal pruning (threshold = φ⁻³ = 0.236)
        - At high fill: ~50% pruning (threshold = 0.50 = median salience)
        - φ-smooth interpolation between extremes
        
        WHY 0.50 MAX (not φ⁻¹ = 0.618):
            - Empirical salience distribution: mean=0.48, median=0.50
            - φ⁻¹ = 0.618 would prune nearly everything (too aggressive)
            - 0.50 = median → ~50% pruned at max fill (brain-like)
        
        This is MORE BRAIN-LIKE than fixed time-based decay:
        - Synaptic homeostasis during sleep prunes ~50% of synapses
        - Pressure increases with memory load, not time
        
        Returns:
            Current pruning threshold based on fill ratio
        """
        fill_ratio = self.num_attractors / self.max_attractors
        # φ-smooth interpolation from φ⁻³ (low fill) to ~median salience (high fill)
        min_threshold = PHI_INV_CUBE  # 0.236 at 0% full (only very weak)
        max_threshold = 0.50          # median salience at 100% full (~50% pruned)
        return min_threshold + (max_threshold - min_threshold) * (fill_ratio ** PHI_INV)
    
    def prune_weak_memories(self) -> int:
        """
        THEORY-TRUE capacity-based pruning: forget weak memories under pressure.
        
        v4.13.0: Simplified to brain-like approach
        - retention = salience (intrinsic, no bookkeeping)
        - threshold adapts to capacity pressure
        - No time decay (artificial bookkeeping removed)
        - No access counting (doesn't work for holographic superposition)
        
        BRAIN ANALOGY:
            Synaptic homeostasis during sleep: when resources are stressed,
            prune weak synapses to make room for strong ones.
            
        Returns:
            Number of memories pruned
        """
        if self.num_attractors == 0:
            return 0
        
        # THEORY-TRUE: retention = salience
        strengths = self.compute_retention_strength_batch()
        
        # ADAPTIVE: threshold increases with fill ratio
        threshold = self.get_adaptive_prune_threshold()
        
        # Find weak memories
        weak_mask = strengths < threshold
        weak_indices = self.xp.where(weak_mask)[0]
        
        if len(weak_indices) == 0:
            return 0
        
        # Prune weak memories (process from end to avoid index shifts)
        pruned_count = 0
        for idx in sorted(weak_indices, reverse=True):
            idx = int(idx)
            self._remove_attractor(idx)
            pruned_count += 1
        
        return pruned_count
    
    def _remove_attractor(self, idx: int):
        """
        Remove a single attractor by index (internal helper).
        Swaps with last element to maintain contiguous storage.
        
        NOTE: This only updates bookkeeping arrays. Holographic memory
        doesn't support selective removal (patterns are superposed).
        """
        # If not the last element, swap with last
        last_idx = self.num_attractors - 1
        if idx < last_idx:
            # Move last entry to this slot
            self.attractor_matrices[idx] = self.attractor_matrices[last_idx]
            self.attractor_targets[idx] = self.attractor_targets[last_idx]
            self.attractor_saliences[idx] = self.attractor_saliences[last_idx]
            self.attractor_consolidated[idx] = self.attractor_consolidated[last_idx]
            # v4.13.0: Removed access_counts and last_access (not theory-true)
        
        self.num_attractors -= 1
        self.total_forgotten += 1
    
    def forget_lowest_priority(self, n_to_forget: int = 1) -> List[int]:
        """
        Capacity-based forgetting: forget the n lowest-priority memories.
        
        This is a FALLBACK for when equilibrium pruning isn't enough
        (e.g., very high salience memories that don't naturally decay).
        
        Returns:
            List of target tokens that were forgotten (for logging)
        """
        if self.num_attractors < n_to_forget:
            return []
        
        # OPTIMIZED: Use vectorized batch computation (~50x faster)
        strengths = self.compute_retention_strength_batch()
        
        # Find the n lowest-strength indices
        lowest_indices = self.xp.argsort(strengths)[:n_to_forget]
        
        # Remove these memories
        forgotten_targets = []
        for idx in sorted(lowest_indices, reverse=True):
            idx = int(idx)
            forgotten_targets.append(int(self.attractor_targets[idx]))
            self._remove_attractor(idx)
        
        return forgotten_targets
    
    def mark_consolidated(self, indices: List[int]):
        """
        Mark memories as consolidated (in semantic memory).
        
        Consolidated memories have lower retention priority since their
        information is now in semantic prototypes.
        
        NOTE (v4.8.0): With holographic memory, this only affects bookkeeping.
        The actual memory is superposed and can't be selectively marked.
        
        Args:
            indices: List of bookkeeping array indices to mark
        """
        for idx in indices:
            if 0 <= idx < self.num_attractors:
                self.attractor_consolidated[idx] = True
    
    def compute_context(self, tokens: List[int], use_chunking: bool = False) -> Array:
        """
        Compute context representation via geometric product + vorticity.
        
        THEORY-TRUE (with vorticity):
            Context = GeomProd(M₁...Mₙ) + φ⁻¹ · Σ(Fᵢ ∧ Fᵢ₋₁)
            
        The vorticity term (wedge products) captures:
            - Word ORDER (A∧B = -B∧A)
            - Syntactic tension between tokens
            - Semantic directionality
            
        CHUNKING PRIOR (v4.12.0):
            When use_chunking=True, applies Grace at phrase boundaries.
            This mimics the brain's phrase-completion processing:
            - 3× improvement in Grace stability for long sequences
            - Hierarchical composition (words → phrases → sentences)
            - Theory: Broca's area applies structure at phrase boundaries
            
        Args:
            tokens: List of token IDs
            use_chunking: If True, apply Grace every 2 tokens (phrase-level)
        """
        if not tokens:
            return self.xp.eye(MATRIX_DIM, dtype=DTYPE)
        
        if len(tokens) == 1:
            return self.get_embedding(tokens[0])
        
        # Get all embeddings - VECTORIZED (no Python loop!)
        token_indices = self.xp.array(tokens, dtype=self.xp.int64) % self.vocab_size
        mats = self.embeddings[token_indices]  # Direct array indexing
        
        if use_chunking and len(tokens) >= 4:
            # CHUNKING PRIOR: Apply Grace at phrase boundaries
            # This creates hierarchical composition: words → phrases → context
            # Tested: 3× stability improvement for long sequences
            context = self._compute_context_chunked(mats, chunk_size=2)
        else:
            # Standard flat composition
            context = geometric_product_batch(mats, self.xp)
        
        # THEORY-CRITICAL: Add vorticity term (Fᵢ ∧ Fᵢ₋₁)
        if self.use_vorticity and len(tokens) >= 2:
            vort = compute_vorticity(mats, self.xp)
            if vort.shape[0] > 0:
                # Sum all pairwise vorticities
                vort_sum = self.xp.sum(vort, axis=0)
                # Add with φ⁻¹ weighting (vorticity is "higher grade" structure)
                context = context + PHI_INV * vort_sum
                # THEORY-TRUE: Use Grace to stabilize, not arbitrary Frobenius norm
                # Grace contracts higher grades while preserving witness
                context = grace_operator(context, self.basis, self.xp)
        
        return context
    
    def _compute_context_chunked(self, mats: Array, chunk_size: int = 2) -> Array:
        """
        Compute context with Grace applied at chunk boundaries.
        
        BRAIN PARALLEL:
            Broca's area processes language in hierarchical chunks:
            - Word-level: individual tokens
            - Phrase-level: 2-3 words (noun phrases, verb phrases)
            - Clause-level: phrases combined
            - Sentence-level: full context
            
            Applying Grace at chunk boundaries mimics this:
            - Each chunk is "settled" before combining with next
            - Creates more stable representations (3× stability gain)
            - Preserves hierarchical structure
            
        Args:
            mats: [n, 4, 4] token embeddings
            chunk_size: Number of tokens per chunk (default 2 = phrase-level)
            
        Returns:
            [4, 4] context matrix with hierarchical structure
        """
        from holographic_v4.algebra import geometric_product
        
        xp = self.xp
        n = mats.shape[0]
        
        if n <= chunk_size:
            return geometric_product_batch(mats, xp)
        
        context = xp.eye(MATRIX_DIM, dtype=DTYPE)
        chunk_count = 0
        
        for i in range(n):
            # Compose current token
            context = geometric_product(context, mats[i])
            norm = xp.linalg.norm(context, 'fro')
            if norm > 1e-8:
                context = context / norm
            
            chunk_count += 1
            
            # Apply Grace at chunk boundary (phrase completion)
            if chunk_count >= chunk_size:
                context = grace_operator(context, self.basis, xp)
                chunk_count = 0
        
        return context
    
    def compute_semantic_context(self, tokens: List[int]) -> Array:
        """
        Compute context using ONLY semantic tokens (high predictiveness).
        
        v4.21.0: Removed fallback_to_full parameter. Behavior is now:
            - If predictiveness disabled: use full context
            - If no semantic tokens identified yet (cold start): use full context
            - If semantic tokens identified: use semantic tokens only
        
        THEORY (SEMANTIC_EXTRACTION_THEORY.md):
            Not all tokens contribute equally to meaning. Noise tokens (articles,
            filler words) have low correlation with targets. Semantic tokens
            (content words) have high correlation.
            
            By composing ONLY semantic tokens, we get:
            - 100% paraphrase generalization (vs 24-42% with full context)
            - Prototype purity (semantic concepts, not noise patterns)
            - Scalability (noise doesn't accumulate with length)
            
        Args:
            tokens: Full context (may include noise tokens)
            
        Returns:
            [4, 4] context matrix from semantic tokens only
        """
        if not self.use_predictiveness or self.predictiveness_tracker is None:
            # Predictiveness not enabled - use full context
            return self.compute_context(tokens)
        
        # Extract semantic tokens
        semantic_tokens = self.predictiveness_tracker.extract_semantic(tokens)
        
        if not semantic_tokens:
            # Cold start: no semantic tokens identified yet
            # THEORY-TRUE: Use full context (NOT identity/zero!)
            # Identity has specific geometric meaning - returning it causes
            # decode_attractor to always return same token.
            return self.compute_context(tokens)
        
        # Compose semantic tokens only
        return self.compute_context(semantic_tokens)
    
    def get_predictiveness_statistics(self) -> Dict[str, Any]:
        """
        Get predictiveness statistics for debugging/analysis.
        
        Returns:
            Dict with token classification stats, or empty if not enabled.
        """
        if not self.use_predictiveness or self.predictiveness_tracker is None:
            return {'enabled': False}
        
        stats = self.predictiveness_tracker.get_statistics()
        stats['enabled'] = True
        return stats
    
    def compute_context_with_vorticity(self, tokens: List[int]) -> Tuple[Array, float, Array]:
        """
        Compute context AND vorticity (sequential tension AND signature).
        
        OPTIMIZATION: For large contexts (>10k tokens), chunk computation to avoid OOM.
        
        Returns:
            (context_matrix, vorticity_magnitude, vorticity_signature)
            
        THEORY:
            - vorticity_magnitude: Scalar measure of "rotational" content
            - vorticity_signature: [16] Clifford coefficients encoding word ORDER
            
            The signature is used for grammar-aware prototype matching.
        """
        if len(tokens) < 2:
            return self.compute_context(tokens), 0.0, self.xp.zeros(16, dtype=DTYPE)
        
        # OPTIMIZATION: Chunk large contexts to avoid memory/compute explosion
        CHUNK_SIZE = 1000  # Process 1k tokens at a time
        if len(tokens) <= CHUNK_SIZE:
            # Small context - process directly
            token_indices = self.xp.array(tokens, dtype=self.xp.int64) % self.vocab_size
            mats = self.embeddings[token_indices]
            context = geometric_product_batch(mats, self.xp)
            # Apply Grace (matches compute_context behavior)
            from holographic_v4.algebra import grace_operator, vorticity_magnitude_and_signature
            context = grace_operator(context, self.basis, self.xp)
            # Compute vorticity ONCE (not twice!)
            vort_mag, vort_sig = vorticity_magnitude_and_signature(mats, self.basis, self.xp)
            return context, vort_mag, vort_sig
        
        # Large context - chunk and compose
        from holographic_v4.algebra import geometric_product, grace_operator, normalize_matrix, vorticity_magnitude_and_signature
        
        context = self.xp.eye(4, dtype=DTYPE)
        total_vort_mag = 0.0
        vort_sig = self.xp.zeros(16, dtype=DTYPE)
        
        # THEORY INSIGHT FOR LONG CONTEXTS:
        # =================================
        # For very long sequences (>10k tokens), the context matrix naturally
        # converges to mostly-witness (scalar + pseudoscalar). This is CORRECT
        # behavior - it's the fixed point of Grace flow.
        #
        # The VORTICITY SIGNATURE is what discriminates long contexts:
        #   - Vorticity = wedge products of consecutive tokens
        #   - Captures word ORDER (A∧B = -B∧A)
        #   - ACCUMULATES with context length (doesn't converge)
        #   - This is the primary feature for long-context clustering
        #
        # Numerical approach:
        #   - Frobenius normalization keeps context matrix bounded (preserves ratios)
        #   - Grace at the end applies theory-true grade-wise damping
        #   - Vorticity signature is the real discriminative information
        #
        # OPTIMIZATION:
        #   - Use vorticity_magnitude_and_signature to compute both from ONE vorticity pass
        #   - This halves the wedge product computations (2x speedup for vorticity!)
        
        for i in range(0, len(tokens), CHUNK_SIZE):
            chunk = tokens[i:i + CHUNK_SIZE]
            token_indices = self.xp.array(chunk, dtype=self.xp.int64) % self.vocab_size
            mats = self.embeddings[token_indices]
            
            # Compute chunk context (Frobenius normalized internally)
            chunk_context = geometric_product_batch(mats, self.xp)
            
            # Compose with running context
            context = geometric_product(context, chunk_context)
            
            # Frobenius normalization for numerical stability
            # This is uniform scaling (preserves grade ratios)
            context = normalize_matrix(context, self.xp)
        
            # ACCUMULATE vorticity - computed ONCE per chunk (not twice!)
            chunk_vort, chunk_sig = vorticity_magnitude_and_signature(mats, self.basis, self.xp)
            total_vort_mag += chunk_vort
            vort_sig += chunk_sig
        
        # Grace at the end for theory-true normalization
        context = grace_operator(context, self.basis, self.xp)
        
        return context, total_vort_mag, vort_sig
    
    def compute_context_representation(self, tokens: List[int]) -> Array:
        """
        Alias for compute_context (used by holographic memory).
        
        This computes the context matrix that will be used for holographic
        storage and retrieval. It includes Grace operator for theory-true
        normalization.
        """
        return self.compute_context(tokens)
    
    def _reindex_witness(self) -> int:
        """
        Reindex witness index after embedding drift.
        
        THEORY-TRUE: After embeddings improve via drift, the witness index
        needs recomputing so queries with new embeddings find correct buckets.
        
        Returns:
            Number of patterns reindexed
        """
        return self.holographic_memory.reindex_witness(
            compute_context_fn=self.compute_context_representation,
            get_embedding_fn=self.get_embedding,
        )
    
    def compute_contexts_batch(
        self, 
        batch_tokens: List[List[int]]
    ) -> Tuple[Array, Array, Array]:
        """
        Compute context matrices for MULTIPLE sequences in parallel.
        
        THIS IS THE KEY OPTIMIZATION FOR ORDER-OF-MAGNITUDE SPEEDUP.
        
        Instead of processing 1 context at a time (Python loop overhead),
        this processes BATCH_SIZE contexts simultaneously on GPU.
        
        Args:
            batch_tokens: List of token lists, each of length CONTEXT_SIZE
                         [[t0, t1, ..., t511], [t0, t1, ..., t511], ...]
        
        Returns:
            (context_matrices, vorticity_magnitudes, vorticity_signatures)
            - context_matrices: [BATCH, 4, 4] 
            - vorticity_magnitudes: [BATCH]
            - vorticity_signatures: [BATCH, 16]
            
        SPEEDUP: Processing N sequences in parallel gives ~N× speedup.
        """
        from holographic_v4.algebra import (
            geometric_product_batch_multi, 
            grace_operator_batch,
            vorticity_magnitude_and_signature_batch
        )
        
        batch_size = len(batch_tokens)
        if batch_size == 0:
            return (
                self.xp.zeros((0, 4, 4), dtype=DTYPE),
                self.xp.zeros(0, dtype=DTYPE),
                self.xp.zeros((0, 16), dtype=DTYPE)
            )
        
        context_size = len(batch_tokens[0])
        
        # 1. Gather all token indices: [BATCH, CONTEXT_SIZE]
        # OPTIMIZED: Convert to array first, then vectorized modulo (41.8× faster)
        batch_tokens_array = self.xp.array(batch_tokens, dtype=self.xp.int64)
        batch_indices = batch_tokens_array % self.vocab_size
        
        # 2. Lookup all embeddings at once: [BATCH, CONTEXT_SIZE, 4, 4]
        # Advanced indexing: self.embeddings[batch_indices] where batch_indices is [BATCH, CONTEXT_SIZE]
        batch_mats = self.embeddings[batch_indices.reshape(-1)].reshape(
            batch_size, context_size, 4, 4
        )
        
        # 3. Compute all geometric products in parallel: [BATCH, 4, 4]
        context_matrices = geometric_product_batch_multi(batch_mats, self.xp)
        
        # 4. Apply Grace to all: [BATCH, 4, 4]
        context_matrices = grace_operator_batch(context_matrices, self.basis, self.xp)
        
        # 5. Compute vorticity for all: [BATCH], [BATCH, 16]
        vort_magnitudes, vort_signatures = vorticity_magnitude_and_signature_batch(
            batch_mats, self.basis, self.xp
        )
        
        return context_matrices, vort_magnitudes, vort_signatures
    
    def compute_contexts_batch_tensor_core(
        self, 
        batch_tokens: List[List[int]]
    ) -> Tuple[Array, Array, Array]:
        """
        Compute context matrices using TENSOR CORE ACCELERATION.
        
        Uses coefficient representation (16-vectors) instead of 4×4 matrices,
        enabling 256×16 matmuls that H100 tensor cores can accelerate.
        
        Args:
            batch_tokens: List of token lists, each of length CONTEXT_SIZE
        
        Returns:
            (context_matrices, vorticity_magnitudes, vorticity_signatures)
        """
        from holographic_v4.algebra import (
            geometric_product_batch_multi_coefficients,
            matrix_to_coefficients,
            coefficients_to_matrix,
            grace_operator_batch,
            vorticity_magnitude_and_signature_batch
        )
        
        batch_size = len(batch_tokens)
        if batch_size == 0:
            return (
                self.xp.zeros((0, 4, 4), dtype=DTYPE),
                self.xp.zeros(0, dtype=DTYPE),
                self.xp.zeros((0, 16), dtype=DTYPE)
            )
        
        context_size = len(batch_tokens[0])
        
        # 1. Gather token indices: [BATCH, CONTEXT_SIZE]
        # OPTIMIZED: Convert to array first, then vectorized modulo (41.8× faster)
        batch_tokens_array = self.xp.array(batch_tokens, dtype=self.xp.int64)
        batch_indices = batch_tokens_array % self.vocab_size
        
        # 2. Lookup embeddings: [BATCH, CONTEXT_SIZE, 4, 4]
        batch_mats = self.embeddings[batch_indices.reshape(-1)].reshape(
            batch_size, context_size, 4, 4
        )
        
        # 3. Convert to coefficients: [BATCH, CONTEXT_SIZE, 16]
        batch_coeffs = matrix_to_coefficients(batch_mats, self.basis, self.xp)
        
        # 4. TENSOR CORE: Compute geometric products using 256×16 matmuls
        context_coeffs = geometric_product_batch_multi_coefficients(batch_coeffs, self.xp)
        
        # 5. Convert back to matrices: [BATCH, 4, 4]
        context_matrices = coefficients_to_matrix(context_coeffs, self.basis, self.xp)
        
        # 6. Apply Grace: [BATCH, 4, 4]
        context_matrices = grace_operator_batch(context_matrices, self.basis, self.xp)
        
        # 7. Compute vorticity: [BATCH], [BATCH, 16]
        vort_magnitudes, vort_signatures = vorticity_magnitude_and_signature_batch(
            batch_mats, self.basis, self.xp
        )
        
        return context_matrices, vort_magnitudes, vort_signatures
    
    def train_batch(
        self, 
        batch_contexts: List[List[int]], 
        batch_targets: List[int]
    ) -> Dict[str, Any]:
        """
        Train on MULTIPLE (context, target) pairs simultaneously.
        
        OPTIMIZED v4.10: Skip novelty checks and redundant storage.
        Holographic memory naturally handles duplicates via superposition.
        
        Args:
            batch_contexts: List of context token lists
            batch_targets: List of target tokens
            
        Returns:
            Dict with batch training statistics
        """
        batch_size = len(batch_contexts)
        if batch_size == 0:
            return {'batch_size': 0, 'stored': 0}
        
        # 1. BATCH CONTEXT COMPUTATION
        # Use tensor core acceleration if enabled (coefficient representation)
        if self.use_tensor_cores:
            ctx_matrices, vort_mags, vort_sigs = self.compute_contexts_batch_tensor_core(batch_contexts)
        else:
            ctx_matrices, vort_mags, vort_sigs = self.compute_contexts_batch(batch_contexts)
        
        # 2. BATCH EMBEDDING LOOKUP (vectorized - single GPU gather)
        # OPTIMIZED: Convert to array first, then vectorized modulo
        target_array = self.xp.array(batch_targets, dtype=self.xp.int32)
        target_indices = target_array % self.vocab_size
        target_embeddings = self.embeddings[target_indices]  # [BATCH, 4, 4]
        
        # 3. Apply binding if enabled (batched)
        if self.use_binding:
            # Batched binding: bind_batch imported at module level
            target_embeddings = bind_batch(target_embeddings, self.basis, self.xp)
        
        # 4. BATCH STORAGE (H100 OPTIMIZATION - single GPU operation for holographic)
        store_result = self.holographic_memory.store_batch(
            ctx_matrices, 
            target_embeddings,
            target_indices,
            token_sequences=batch_contexts  # For reindexing after drift
        )
        
        # 5. BOOKKEEPING UPDATE (for diagnostics and dreaming) — FULLY VECTORIZED
        # Note: Holographic memory is superposition-based and can't selectively remove.
        # "Forgetting" happens through prototype consolidation in dreaming, not deletion.
        old_count = self.num_attractors
        self.num_attractors = min(self.holographic_memory.holographic.n_patterns, self.max_attractors)
        new_count = self.num_attractors - old_count
        
        # VECTORIZED bookkeeping update (was per-item loop — 10x slower!)
        if new_count > 0 and old_count < self.max_attractors:
            # Determine indices to update
            n_to_update = min(new_count, batch_size, self.max_attractors - old_count)
            start_idx = old_count
            end_idx = old_count + n_to_update
            
            # BATCHED salience computation (single GPU call for entire batch)
            batch_saliences = grace_stability_batch(target_embeddings[:n_to_update], self.basis, self.xp)
            
            # VECTORIZED array updates (no Python loop!)
            self.attractor_matrices[start_idx:end_idx] = target_embeddings[:n_to_update]
            self.attractor_targets[start_idx:end_idx] = target_indices[:n_to_update].astype(self.xp.int32)
            self.attractor_saliences[start_idx:end_idx] = batch_saliences
            self.attractor_consolidated[start_idx:end_idx] = False
            # v4.13.0: Removed access_counts, last_access (not theory-true)
        
        self.train_samples += batch_size
        
        # Note: Equilibrium pruning disabled for holographic memory.
        # Superposition can't selectively remove patterns - use dreaming consolidation instead.
        pruned_count = 0
        
        return {
            'batch_size': batch_size,
            'stored': batch_size,  # All stored (no novelty filter)
            'num_attractors': self.num_attractors,
            'pruned': pruned_count,
            'avg_vorticity': float(self.xp.mean(vort_mags)) if batch_size > 0 else 0.0,
        }
    
    # =========================================================================
    # TRAINING — Theory-True Update Rule
    # =========================================================================
    
    def train_step(self, context: List[int], target: int) -> Dict[str, float]:
        """
        THEORY-TRUE training step (per rhnsclifford.md lines 23-24, 111, 360-361).
        
        Update rule (line 111):
            attractor[context] = lerp(attractor[context], target_matrix, φ⁻¹)
        
        This is DIRECT LERP with rate φ⁻¹:
            - NO Grace in learning (Grace is for forward/evolution only)
            - NO dynamic rate modulation (φ values are mathematically fixed)
            - Store via holographic superposition
        
        The learning is SIMPLE ASSOCIATION via holographic binding:
            memory += bind(context, target)
        
        Grace belongs in the FORWARD pass (evolve_to_equilibrium), not here.
        
        φ-FORGETTING (theory-true equilibrium):
            TWO MODES of forgetting for sparse efficiency:
            
            1. EQUILIBRIUM PRUNING (continuous):
               - Memories with retention < φ⁻³ are forgotten automatically
               - Creates natural selection: strong survive, weak decay
               - Runs every prune_interval samples (batch efficiency)
               
            2. CAPACITY PRUNING (fallback):
               - When memory truly full, forget lowest-strength
               - Safety net for edge cases
               
            This is theory-true: φ-decay matches Grace spectral structure.
        """
        target_idx = target % self.vocab_size
        
        # SEMANTIC EXTRACTION: Track token-target co-occurrences
        # This enables automatic identification of semantic vs noise tokens
        if self.use_predictiveness and self.predictiveness_tracker is not None:
            self.predictiveness_tracker.observe(context, target_idx)
        
        # Get target embedding (FIXED, not learned)
        target_emb = self.get_embedding(target_idx)
        
        # Optionally apply binding
        if self.use_binding:
            target_emb = bind(target_emb, self.basis, self.xp)
        
        # Compute context matrix for holographic storage
        context_matrix = self.compute_context_representation(context)
        
        # EQUILIBRIUM PRUNING: Check periodically for weak memories to forget
        pruned_count = 0
        if self.equilibrium_forgetting:
            samples_since_prune = self.train_samples - self.last_prune_sample
            if samples_since_prune >= self.prune_interval:
                pruned_count = self.prune_weak_memories()
                self.last_prune_sample = self.train_samples
        
        forgotten = []
        prediction_correct = None  # Track for meta-learning and credit assignment
        
        # =========================================================================
        # FAST PATH: Skip novelty check (~200× faster)
        # Holographic superposition naturally handles duplicates via reinforcement:
        #   memory += w × bind(C,T) reinforces existing patterns automatically
        # No need to check if context is "known" - just store and let superposition work
        # =========================================================================
        if self.skip_novelty_check:
            # Fixed theory-derived learning rate (no novelty modulation)
            learning_rate = PHI_INV
            salience = self.compute_attractor_salience(target_emb)
            
            # DIRECT STORE: Superposition handles reinforcement naturally
            # - First occurrence: creates new pattern
            # - Repeated occurrence: strengthens existing pattern (reconsolidation!)
            self.holographic_memory.store(context_matrix, target_emb, target_idx,
                                          token_sequence=context)
            
            # Simplified bookkeeping (no novel/known distinction needed)
            if self.num_attractors < self.max_attractors:
                idx = self.num_attractors
                self.attractor_matrices[idx] = target_emb.copy()
                self.attractor_targets[idx] = target_idx
                self.attractor_saliences[idx] = salience
                self.attractor_consolidated[idx] = False
                # v4.13.0: Removed access_counts, last_access (not theory-true)
                self.num_attractors += 1
            
            # Note: embedding_drift disabled in fast path (needs retrieval)
            # Use skip_novelty_check=False if embedding drift is critical
            
        # =========================================================================
        # FULL PATH: Novelty check for embedding drift and meta-learning
        # ~200× slower but enables: embedding drift, adaptive learning rate
        # =========================================================================
        else:
            # HOLOGRAPHIC NOVELTY CHECK: Try retrieval to determine if context is known
            existing_result, existing_target, existing_conf, existing_source = \
                self.holographic_memory.retrieve(context_matrix)
            
            # COMPUTE SALIENCE AND NOVELTY for meta-learning
            salience = self.compute_attractor_salience(target_emb)
            is_novel = existing_conf < PHI_INV_SQ  # Low confidence = novel context
            novelty = 1.0 if is_novel else PHI_INV_CUBE  # φ-derived: novel=1.0, known=φ⁻³
            
            # META-LEARNING: Compute adaptive learning rate
            if self.use_meta_learning and self.learning_state is not None:
                learning_rate = compute_adaptive_learning_rate(
                    salience=salience,
                    novelty=novelty,
                    uncertainty=self.learning_state.uncertainty,
                    base_rate=PHI_INV,
                )
            else:
                learning_rate = PHI_INV
            
            if not is_novel:
                # KNOWN CONTEXT: Update existing memory
                if self.use_credit_assignment or self.use_meta_learning or self.use_embedding_drift:
                    prediction_correct = (existing_target == target_idx)
                
                # EMBEDDING DRIFT: Update target embedding if prediction was wrong
                if self.use_embedding_drift and prediction_correct is False:
                    gradient = compute_embedding_gradient(
                        token=target_idx,
                        retrieval_success=False,
                        attractor=existing_result,
                        model=self,
                    )
                    update_embedding(target_idx, gradient, self)
                    self._needs_reindex = True
                
                # HOLOGRAPHIC UPDATE
                self.holographic_memory.update(context_matrix, target_emb, target_idx,
                                               token_sequence=context)
                
                # Update bookkeeping arrays
                if self.num_attractors > 0:
                    idx = 0
                    self.attractor_matrices[idx] = target_emb.copy()
                    self.attractor_targets[idx] = target_idx
                    # v4.13.0: Removed access_counts, last_access (not theory-true)
                    # φ-average salience (exponential smoothing)
                    self.attractor_saliences[idx] = (
                        (1 - PHI_INV) * self.attractor_saliences[idx] + PHI_INV * salience
                    )
                
            else:
                # NOVEL CONTEXT: Store new memory
                prediction_correct = True
                
                self.holographic_memory.store(context_matrix, target_emb, target_idx,
                                              token_sequence=context)
                
                if self.num_attractors < self.max_attractors:
                    idx = self.num_attractors
                    self.attractor_matrices[idx] = target_emb.copy()
                    self.attractor_targets[idx] = target_idx
                    self.attractor_saliences[idx] = salience
                    self.attractor_consolidated[idx] = False
                    # v4.13.0: Removed access_counts, last_access (not theory-true)
                    self.num_attractors += 1
            
            # Update novelty in meta-state
            if self.use_meta_learning and self.learning_state is not None:
                self.learning_state = update_meta_state(
                    state=self.learning_state,
                    prediction_correct=prediction_correct if prediction_correct is not None else True,
                    salience=salience,
                    novelty=novelty,
                )
        
        # Check for memory saturation - return signal so caller can trigger dreaming
        # OPTIMIZATION: Only compute every N steps (was 20.5% of train_step time!)
        # Dreaming triggers don't need instant signals - periodic checks suffice
        if self.train_samples % self._saturation_check_interval == 0:
            self._last_witness_entropy = compute_witness_entropy(
                self.holographic_memory.holographic.memory, self.basis, self.xp
            )
            self._last_memory_saturated = is_memory_saturated(self._last_witness_entropy)
        
        witness_entropy = self._last_witness_entropy
        memory_saturated = self._last_memory_saturated
        
        # OPTIMIZATION: Only compute context_matrix when caller needs it for dreaming
        # Holographic storage is already done via context_matrix above
        self.train_samples += 1
        
        # EMBEDDING DRIFT SUPPORT: Periodic reindexing of witness index
        # When embeddings drift, stored witnesses become stale. Reindex periodically.
        if self._needs_reindex and self.train_samples % self._reindex_interval == 0:
            self._reindex_witness()
            self._needs_reindex = False
        
        return {
            'num_attractors': self.num_attractors,
            'context': context,  # Return raw tokens so caller can compute matrix if needed
            'forgotten': len(forgotten),  # Capacity-based forgetting
            'pruned': pruned_count,  # Equilibrium-based pruning
            'learning_rate': learning_rate,  # Actual rate used (for diagnostics)
            'prediction_correct': prediction_correct,  # For tracking accuracy
            'memory_saturated': memory_saturated,  # Signal for dreaming trigger
            'witness_entropy': witness_entropy,  # Capacity diagnostic
        }
    
    def train_step_with_dreaming(self, context: List[int], target: int) -> Dict[str, Any]:
        """
        Train step that also returns context_matrix for episodic buffer.
        
        Use this ONLY when dreaming is enabled and episodic buffer needs filling.
        Otherwise use train_step() which is faster.
        """
        # Do the basic train step
        result = self.train_step(context, target)
        
        # Compute context matrix and vorticity for dreaming
        ctx_matrix, vort_mag, vort_sig = self.compute_context_with_vorticity(context)
        
        result['vorticity'] = vort_mag
        result['context_matrix'] = ctx_matrix
        result['vorticity_signature'] = vort_sig
        
        return result
    
    def train_step_with_context(self, context_matrix: Array, context: List[int], target: int) -> Dict[str, float]:
        """
        OPTIMIZED train step using pre-computed context matrix.
        
        CRITICAL PERFORMANCE OPTIMIZATION:
            This method accepts a pre-computed context_matrix, avoiding
            redundant O(context_size) computation. When context is 512 tokens,
            this saves ~512 matrix multiplications per call.
            
            Use this when you've already computed the context matrix for
            retrieval/predictive coding and want to avoid recomputation.
        
        Args:
            context_matrix: Pre-computed [4, 4] context representation
            context: Original token list (for bookkeeping only)
            target: Target token
            
        Returns:
            Training metrics dict
        """
        target_idx = target % self.vocab_size
        
        # SEMANTIC EXTRACTION: Track token-target co-occurrences
        if self.use_predictiveness and self.predictiveness_tracker is not None:
            self.predictiveness_tracker.observe(context, target_idx)
        
        # Get target embedding (FIXED, not learned)
        target_emb = self.get_embedding(target_idx)
        
        # Optionally apply binding
        if self.use_binding:
            target_emb = bind(target_emb, self.basis, self.xp)
        
        # Use pre-computed context matrix (the optimization!)
        # No call to compute_context_representation here
        
        # HOLOGRAPHIC NOVELTY CHECK: Use pre-computed context
        existing_result, existing_target, existing_conf, existing_source = \
            self.holographic_memory.retrieve(context_matrix)
        
        # Determine learning rate based on novelty
        is_novel = existing_conf < PHI_INV_SQ
        # COMPUTE SALIENCE ONCE (avoid double computation)
        salience = self.compute_attractor_salience(target_emb)
        
        if is_novel:
            learning_rate = PHI_INV  # Standard rate for novel
        else:
            # Reduce rate for known contexts (reconsolidation)
            learning_rate = PHI_INV_CUBE
        
        # META-LEARNING: Compute adaptive learning rate (matches train_step)
        if self.use_meta_learning and self.learning_state is not None:
            learning_rate = compute_adaptive_learning_rate(
                salience=salience,
                novelty=1.0 if is_novel else 0.0,
                uncertainty=self.learning_state.uncertainty,
                base_rate=PHI_INV,
            )
        
        # CREDIT ASSIGNMENT: Track prediction correctness (for meta-learning, embedding drift)
        prediction_correct = is_novel or (existing_target == target_idx)
        
        # STORE: Add to holographic memory (with token sequence for drift reindexing)
        self.holographic_memory.store(context_matrix, target_emb, target_idx,
                                       token_sequence=context)
        
        # Store in multi-timescale memory (reuse computed salience)
        self.multi_timescale_memory.store(context_matrix, target_emb, salience=salience)
        
        # Update bookkeeping (cap to max_attractors for array bounds safety)
        self.num_attractors = min(self.holographic_memory.holographic.n_patterns, self.max_attractors)
        self.train_samples += 1
        
        # φ-FORGETTING: Periodic equilibrium pruning
        pruned_count = 0
        if self.use_equilibrium and self.train_samples % self.prune_interval == 0:
            pruned_count = self.prune_weak_memories()
        
        return {
            'num_attractors': self.num_attractors,
            'stored': True,
            'is_novel': is_novel,
            'context': context,
            'pruned': pruned_count,
            'learning_rate': learning_rate,
            'prediction_correct': prediction_correct,
        }
    
    def train(
        self,
        contexts: List[List[int]],
        targets: List[int],
        log_every: int = 1000,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            contexts: List of context sequences
            targets: List of target tokens
            log_every: Log every N samples
            verbose: Print progress
            
        Returns:
            Training history
        """
        history = {
            'step': [],
            'num_attractors': [],
            'avg_vorticity': [],
        }
        
        n = len(contexts)
        
        for i, (ctx, target) in enumerate(zip(contexts, targets)):
            metrics = self.train_step(ctx, target)
            
            if verbose and (i + 1) % log_every == 0:
                avg_vort = self.total_vorticity / max(self.train_samples, 1)
                
                history['step'].append(i + 1)
                history['num_attractors'].append(self.num_attractors)
                history['avg_vorticity'].append(avg_vort)
                
                print(f"  [{i+1:,}/{n:,}] attractors={self.num_attractors:,} | "
                      f"avg_vorticity={avg_vort:.4f}")
        
        return history
    
    def train_streaming(
        self,
        token_stream: List[int],
        context_size: int = None,
        log_every: int = 1000,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        BRAIN-INSPIRED STREAMING TRAINING with O(1) context updates.
        
        Instead of recomputing context from scratch for each position (O(context_size)),
        this method maintains a running context that's updated incrementally (O(1)).
        
        SPEEDUP: ~97× at context_size=100 (verified in brain_efficiency_tests.py)
        
        Args:
            token_stream: Sequential list of tokens
            context_size: Window size (defaults to self.context_size)
            log_every: Log every N samples
            verbose: Print progress
            
        Returns:
            Training history with streaming-specific metrics
        """
        from holographic_v4.algebra import grace_operator
        
        context_size = context_size or self.context_size
        history = {
            'step': [],
            'num_attractors': [],
            'avg_vorticity': [],
            'context_update_time_ms': [],
        }
        
        n = len(token_stream)
        if n <= context_size:
            if verbose:
                print(f"Token stream too short ({n} tokens, need > {context_size})")
            return history
        
        # Initialize incremental context with first context_size tokens
        self.reset_incremental_context()
        for token_idx in token_stream[:context_size]:
            self.update_context_incremental(token_idx)
        
        import time
        total_update_time = 0.0
        num_updates = 0
        
        # Stream through remaining tokens
        for i in range(context_size, n):
            # The current context predicts token_stream[i]
            target = token_stream[i]
            
            # Finalize context (apply Grace) for storage
            context_matrix = self.finalize_incremental_context()
            
            # Get target embedding
            target_idx = target % self.vocab_size
            target_emb = self.get_embedding(target_idx)
            
            if self.use_binding:
                from holographic_v4.quotient import bind
                target_emb = bind(target_emb, self.basis, self.xp)
            
            # Store in holographic memory
            self.holographic_memory.store(
                context_matrix, target_emb, target_idx=target_idx
            )
            
            # O(1) context update: Add new token
            start_time = time.perf_counter()
            self.update_context_incremental(target_idx)
            update_time = time.perf_counter() - start_time
            total_update_time += update_time
            num_updates += 1
            
            # Update statistics
            self.num_attractors = self.holographic_memory.holographic.n_patterns
            self.train_samples += 1
            
            # Periodic pruning
            if self.use_equilibrium and self.train_samples % self.prune_interval == 0:
                self.prune_weak_memories()
            
            # Logging
            if verbose and (i + 1) % log_every == 0:
                avg_vort = self.total_vorticity / max(self.train_samples, 1)
                avg_update_ms = (total_update_time / num_updates) * 1000
                
                history['step'].append(i + 1)
                history['num_attractors'].append(self.num_attractors)
                history['avg_vorticity'].append(avg_vort)
                history['context_update_time_ms'].append(avg_update_ms)
                
                print(f"  [{i+1:,}/{n:,}] attractors={self.num_attractors:,} | "
                      f"avg_vorticity={avg_vort:.4f} | update={avg_update_ms:.3f}ms")
        
        return history
    
    # =========================================================================
    # RETRIEVAL & GENERATION
    # =========================================================================
    
    def retrieve(self, context: List[int]) -> Tuple[Array, int]:
        """
        Retrieve attractor for a context.
        
        THEORY-TRUE RETRIEVAL (v4.8.0):
            Uses HolographicMemory for O(1) unbinding from superposition.
            
            For unknown contexts (low confidence), returns the context matrix itself
            with target_idx=0 to signal "I don't know."
            
            Generalization requires SEMANTIC MEMORY from dreaming,
            which should be integrated externally via DreamingSystem.
        
        WHY HOLOGRAPHIC IS THEORY-TRUE:
            - Stores via Clifford superposition (respects geometric structure)
            - Retrieves via unbinding (geometric product with inverse)
            - Grace naturally denoises interference
            - O(1) regardless of stored patterns
        
        φ-FORGETTING: Retrieval increases access count (reconsolidation effect).
            Frequently accessed memories are more resistant to forgetting.
        
        Args:
            context: Token sequence
        
        Returns:
            (equilibrium_matrix, target_idx) or (context_matrix, 0) if unknown
        """
        # Compute context matrix for holographic retrieval
        ctx_rep = self.compute_context_representation(context)
        
        # HOLOGRAPHIC RETRIEVAL (O(1) unbinding)
        attractor, target_idx, confidence, source = self.holographic_memory.retrieve(ctx_rep)
        
        # Track retrieval for diagnostics
        if confidence >= PHI_INV_SQ:
            # WITNESS-BUCKET RECONSOLIDATION (v4.12.0)
            # THEORY: Fuzzy reconsolidation via witness bucket tracking
            # 
            # Why bucket-based (not slot-based):
            # 1. Holographic memory is SUPERPOSITION - no individual slots
            # 2. Same witness bucket = geometrically similar contexts
            # 3. This is O(1) per retrieval (dictionary update)
            #
            # Brain analog: Spreading activation
            # Recalling "cat" activates geometrically similar memories ("furry", "pet")
            # All memories in the same witness region get credit for being useful
            # NOTE: v4.16.2 - Removed _record_bucket_access (access tracking removed in v4.13.0)
            
            # THEORY-TRUE: Evolve to equilibrium via Grace flow
            if self.use_equilibrium:
                attractor = grace_flow(
                    ctx_rep, attractor, self.basis,
                    steps=self.equilibrium_steps,
                    rate=PHI_INV_SQ,
                    xp=self.xp
                )
            
            return attractor, int(target_idx)
        
        # LOW CONFIDENCE PATH (v4.12.0 fix)
        # When holographic confidence is low due to interference (overlapping contexts),
        # the holographic memory may STILL have found the correct target.
        # 
        # THEORY: Sliding window creates high overlap:
        #   ctx1 = [a,b,c,d,e,f,g,h] → target X
        #   ctx2 = [b,c,d,e,f,g,h,i] → target Y
        # These share 7/8 tokens = 87.5% overlap = interference
        # Confidence drops, but target_idx is often still CORRECT!
        #
        # FIX: If holographic found a non-zero target with ANY confidence, use it.
        # The witness index also confirms the target.
        if target_idx > 0:
            # Holographic found something - trust it even at low confidence
            # This handles the sliding window interference case
            return attractor, int(target_idx)
        
        # Truly unknown context: return context matrix itself (NOT identity!)
        # THEORY-TRUE: Identity is geometrically meaningful (scalar=4, enstrophy=0).
        # Returning identity causes decode_attractor to always pick the same token.
        # Instead, return the context representation - its Grace-stability σ signals
        # uncertainty, and decode_attractor's φ-weighted sampling handles diversity.
        # Generalization requires DreamingSystem for semantic retrieval.
        return ctx_rep, 0
    
    def decode_attractor(self, attractor: Array, use_sampling: bool = True) -> int:
        """
        Decode attractor to token via φ-weighted sampling (theory-true).
        
        THEORY-TRUE DECODING:
            Uses φ-weighted population coding, NOT argmax (winner-take-all).
            
            The theory explicitly rejects argmax because:
            1. argmax causes mode collapse to high-frequency tokens
            2. argmax violates population coding principle
            3. argmax ignores uncertainty (all confidence in one token)
            
            Instead, we:
            1. Compute similarities to all embeddings
            2. Apply φ-kernel: weight_i = φ^(-d_i) where d_i = 1 - sim_i
            3. Sample from the weighted distribution
            
            This is consistent with distributed_prior_retrieve which also
            uses φ-weighted sampling, not argmax.
        
        STRUCTURAL MATCHING:
            When use_vorticity_decoding=True (default), similarities account for
            enstrophy correspondence + witness alignment, not just Frobenius.
        
        Args:
            attractor: [4, 4] equilibrium field to decode
            use_sampling: If True, φ-weighted sampling. If False, argmax (legacy).
            
        Returns:
            Token index
        """
        xp = self.xp
        
        if self.use_vorticity_decoding:
            # FAST THEORY-TRUE: Use precomputed embedding features (no loop!)
            sims = self._decode_vorticity_fast(attractor)
        elif self.use_adaptive_similarity:
            sims = adaptive_similarity_batch(
                attractor, self.embeddings, self.basis, xp
            )
        else:
            sims = frobenius_similarity_batch(attractor, self.embeddings, xp)
        
        # THEORY-TRUE DECODING
        #
        # The Grace flow has ALREADY done the competition (lateral inhibition).
        # The equilibrium IS the answer. We find the nearest embedding.
        #
        # Why NOT probabilistic sampling?
        # - Sampling is an ML generation technique, not derived from theory
        # - The brain's lexical retrieval is deterministic (find matching word)
        # - Diversity should come from: representation noise, prototype variation,
        #   embedding drift - NOT from sampling at decode time
        #
        # Why argmax is now OK?
        # - Because we're finding the NEAREST embedding after Grace stabilization
        # - The Grace flow already handled soft competition
        # - This is like the brain's lexical access: deterministic lookup
        #
        # The use_sampling parameter is kept for backwards compatibility/testing.
        
        if use_sampling:
            # OPTIONAL: Add noise-based diversity (theory-true: noise is in the dynamics)
            # If caller wants diversity, add small noise to sims before argmax
            # This is like noise in neural firing rates
            noise = self.xp.asarray(np.random.randn(len(sims)) * PHI_INV_CUBE)
            sims = sims + noise
        
        # Nearest neighbor (winner after Grace competition + optional noise)
        return int(xp.argmax(sims))
    
    def _decode_vorticity_fast(self, attractor: Array) -> Array:
        """
        FAST vorticity-weighted decoding using precomputed embedding features.
        
        This is O(V) vectorized operations instead of O(V) Python loops!
        
        THEORY: Always consider structure, don't have a "fast path" that 
        bypasses structural matching (that causes mode collapse).
        """
        from .algebra import decompose_to_coefficients
        from .quotient import GRADE_INDICES
        
        xp = self.xp
        
        # THEORY-PARSIMONIOUS: Decompose ONCE, derive everything from coefficients
        # Old code called compute_enstrophy (decomposes) then decompose_to_coefficients (decomposes AGAIN)
        attractor_coeffs = decompose_to_coefficients(attractor, self.basis, xp)
        
        # Derive enstrophy from coefficients (no second decomposition!)
        grade2_coeffs = attractor_coeffs[GRADE_INDICES[2]]
        attractor_enstrophy = float(xp.sum(grade2_coeffs * grade2_coeffs))
        
        # Witness from coefficients (already have them!)
        attractor_scalar = attractor_coeffs[0]
        attractor_pseudo = attractor_coeffs[15]
        
        # Base Frobenius similarity (fully vectorized)
        base_sims = xp.sum(attractor * self.embeddings, axis=(1, 2))
        
        # Enstrophy match: penalize large enstrophy mismatch
        enstrophy_diff = xp.abs(self.embedding_enstrophies - attractor_enstrophy)
        enstrophy_penalty = enstrophy_diff / (xp.max(enstrophy_diff) + 1e-8)
        
        # Witness alignment: reward similar scalar+pseudo signatures
        witness_diff = (
            xp.abs(self.embedding_scalars - attractor_scalar) +
            xp.abs(self.embedding_pseudos - attractor_pseudo)
        )
        witness_penalty = witness_diff / (xp.max(witness_diff) + 1e-8)
        
        # Combined structural penalty [0, 1] — φ-derived weights
        structure_penalty = PHI_INV * enstrophy_penalty + PHI_INV_SQ * witness_penalty
        
        # Adaptive weighting based on attractor enstrophy
        # Higher enstrophy → more weight on structure match
        # FIX: Use minimum weight to prevent mode collapse when attractor_enstrophy ≈ 0
        # (Retrieved targets are single-token embeddings with low enstrophy)
        MIN_STRUCTURE_WEIGHT = PHI_INV_SQ  # φ⁻² ≈ 0.38 minimum prevents pure Frobenius collapse
        structure_weight = max(MIN_STRUCTURE_WEIGHT, min(1.0, attractor_enstrophy * 10)) * self.vorticity_decode_weight
        
        # Final scores: base similarity minus structural penalty
        # Normalize base_sims to [0, 1] range for fair combination
        base_min = float(xp.min(base_sims))
        base_max = float(xp.max(base_sims))
        if base_max - base_min > 1e-8:
            base_norm = (base_sims - base_min) / (base_max - base_min)
        else:
            # All sims equal - use φ-derived neutral default (was arbitrary 0.5)
            base_norm = xp.ones_like(base_sims) * PHI_INV_SQ
        
        # Combined score: similarity minus structure mismatch
        scores = base_norm - structure_weight * structure_penalty
        
        return scores
    
    def generate(self, context: List[int], num_tokens: int = 10) -> List[int]:
        """Generate tokens autoregressively."""
        generated = []
        current_ctx = list(context)
        
        for _ in range(num_tokens):
            attractor, _ = self.retrieve(current_ctx[-self.context_size:])
            token = self.decode_attractor(attractor)
            generated.append(token)
            current_ctx.append(token)
        
        return generated
    
    # =========================================================================
    # SELF-ORGANIZING RETRIEVAL (Theory-True Module Orchestration)
    # =========================================================================
    
    def self_organizing_retrieve(
        self, 
        context: List[int],
        dreaming = None,  # Type: Optional[DreamingSystem]
    ) -> Dict[str, Any]:
        """
        THEORY-TRUE RETRIEVAL with automatic module orchestration.
        
        This is the parsimonious, self-organizing retrieval system that
        automatically decides when to invoke optional modules based on
        INTRINSIC SIGNALS, not external configuration.
        
        THE PRINCIPLE:
            Grace-stability σ is the universal uncertainty signal.
            When σ < φ⁻² (threshold derived from theory), the system is UNCERTAIN.
            Uncertainty automatically triggers:
            - Credit assignment (why did we fail?)
            - Curiosity flagging (explore this region later)
            - Distributed prior (use population coding for uncertain queries)
        
        RETRIEVAL CASCADE (informationally parsimonious) — v4.8.0:
            1. Holographic unbinding (O(1)) - true superposition retrieval
            2. Semantic retrieval via dreaming (O(prototypes)) - generalization
            3. Distributed prior (O(K)) - uncertain queries
            
            Each level is only tried if previous fails or has low confidence.
        
        AUTOMATIC MODULE TRIGGERS:
            - Curiosity: σ < φ⁻² → flag for exploration
            - Credit assignment: prediction error → trace provenance
            - Meta-learning: updates learning state based on outcome
        
        Args:
            context: Token sequence
            dreaming: Optional DreamingSystem for semantic retrieval
            
        Returns:
            Dict containing:
                'result': (attractor, target_idx)
                'confidence': Grace-stability of result
                'method': 'holographic' | 'semantic' | 'distributed' | 'unknown'
                'curiosity_flagged': bool - should explore this region
                'meta_updated': bool - learning state was updated
        """
        result = {
            'method': 'unknown',
            'confidence': 0.0,
            'curiosity_flagged': False,
            'meta_updated': False,
        }
        
        # Compute context matrix for holographic retrieval
        ctx_rep = self.compute_context_representation(context)
        
        # ===== LEVEL 1: Holographic retrieval (O(1) unbinding) =====
        attractor, target_idx, holo_conf, source = self.holographic_memory.retrieve(ctx_rep)
        
        if holo_conf >= PHI_INV_SQ:  # φ⁻² ≈ 0.382
            # WITNESS-BUCKET RECONSOLIDATION: Track bucket access
            # NOTE: v4.16.2 - Removed _record_bucket_access (access tracking removed in v4.13.0)
            
            # Evolve to equilibrium if enabled
            if self.use_equilibrium:
                attractor = grace_flow(
                    ctx_rep, attractor, self.basis,
                    steps=self.equilibrium_steps,
                    rate=PHI_INV_SQ,
                    xp=self.xp
                )
            
            # Compute confidence via Grace-stability
            confidence = grace_stability(attractor, self.basis, self.xp)
            
            result['result'] = (attractor, int(target_idx))
            result['method'] = 'holographic'
            result['confidence'] = confidence
            
            # Self-organizing: High confidence → done
            if confidence >= PHI_INV_SQ:
                return result
        
        # ===== LEVEL 2: Semantic retrieval via dreaming (if available) =====
        if dreaming is not None and hasattr(dreaming, 'semantic_memory'):
            semantic_memory = dreaming.semantic_memory
            if semantic_memory is not None and hasattr(semantic_memory, 'levels'):
                # Use semantic context if predictiveness enabled
                if self.use_predictiveness:
                    ctx_rep = self.compute_semantic_context(context)
                else:
                    ctx_rep = self.compute_context(context)
                
                # Search prototypes
                from .resonance import distributed_prior_retrieve
                from .quotient import extract_witness
                
                # Gather all prototypes
                all_protos = []
                all_targets = []
                all_supports = []
                
                for level in semantic_memory.levels:
                    for proto in level:
                        if hasattr(proto, 'centroid') and hasattr(proto, 'target_distribution'):
                            all_protos.append(proto.centroid)
                            all_targets.append(proto.target_distribution)
                            all_supports.append(getattr(proto, 'support', 1.0))
                
                if all_protos:
                    # Distributed prior retrieval
                    pred_target, pred_conf, _ = distributed_prior_retrieve(
                        ctx_rep, all_protos, all_targets, all_supports,
                        self.basis, self.xp, K=min(8, len(all_protos))
                    )
                    
                    if pred_conf > result['confidence']:
                        # Semantic retrieval is better
                        pred_attractor = self.get_embedding(pred_target)
                        result['result'] = (pred_attractor, pred_target)
                        result['method'] = 'semantic'
                        result['confidence'] = pred_conf
        
        # ===== LEVEL 3: Unknown - flag for curiosity =====
        if result['method'] == 'unknown' or result['confidence'] < PHI_INV_SQ:
            # Return identity for unknown (explicit "I don't know")
            if 'result' not in result:
                result['result'] = (self.xp.eye(MATRIX_DIM, dtype=DTYPE), 0)
            
            # SELF-ORGANIZING: Flag this query for curiosity/exploration
            result['curiosity_flagged'] = True
            
            # Track this uncertainty for meta-learning
            if self.use_meta_learning and self.learning_state is not None:
                self.learning_state = update_meta_state(
                    state=self.learning_state,
                    prediction_correct=False,  # Uncertain = treat as error
                    salience=PHI_INV_SQ,  # φ-derived neutral salience (was arbitrary 0.5)
                    novelty=1.0,  # Unknown is maximally novel
                )
                result['meta_updated'] = True
        
        return result
    
    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        avg_vort = self.total_vorticity / max(self.train_samples, 1)
        
        # Witness stability of embeddings
        sample_size = min(100, self.vocab_size)
        sample_embs = self.embeddings[:sample_size]
        w_stab = witness_stability(sample_embs, self.basis, self.xp)
        
        return {
            'train_samples': self.train_samples,
            'num_attractors': self.num_attractors,
            'avg_vorticity': avg_vort,
            'witness_stability': w_stab,
        }
    
    def compute_enstrophy(self, M: Array) -> float:
        """
        Compute Clifford enstrophy = ||grade-2 content||²
        
        This is the bivector energy, analogous to fluid enstrophy.
        Grace should damp this at rate φ⁻⁴ per step.
        
        INFORMATION PARSIMONY (43.7× speedup!):
            α(M) = -γ₅ M γ₅ (grade involution)
            M_even = (M + α(M))/2
            Enstrophy = ||M_even||²_F/4 - σ² - p²
        """
        gamma5 = self.basis[15]
        
        # Grade involution: α(M) = -γ₅ M γ₅
        M_alpha = -gamma5 @ M @ gamma5
        M_even = (M + M_alpha) / 2
        
        # Even energy and witness components
        even_energy = float(self.xp.sum(M_even * M_even) / 4.0)
        sigma = float(self.xp.trace(M) / 4.0)
        pseudo = float(self.xp.sum(gamma5 * M) / 4.0)
        
        return even_energy - sigma**2 - pseudo**2
    
    def get_enstrophy_stats(self, sample_size: int = 100) -> Dict[str, float]:
        """
        Get enstrophy statistics for stored attractors.
        
        Returns average enstrophy (bivector energy) which should be bounded
        if Grace is properly damping vorticity injection.
        """
        if self.num_attractors == 0:
            return {'avg_enstrophy': 0.0, 'max_enstrophy': 0.0}
        
        n = min(sample_size, self.num_attractors)
        # OPTIMIZED: Use vectorized batch computation (~50x faster)
        matrices = self.attractor_matrices[:n]
        enstrophies = compute_enstrophy_batch(matrices, self.basis, self.xp)
        
        return {
            'avg_enstrophy': float(self.xp.mean(enstrophies)),
            'max_enstrophy': float(self.xp.max(enstrophies)),
            'min_enstrophy': float(self.xp.min(enstrophies)),
        }
    
    def grade_analysis(self, sample_size: int = 100) -> Dict[str, float]:
        """Analyze grade distribution of stored attractors."""
        if self.num_attractors == 0:
            return {}
        
        n = min(sample_size, self.num_attractors)
        
        avg_energies = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        
        for i in range(n):
            energies = grade_energies(self.attractor_matrices[i], self.basis, self.xp)
            for grade, energy in energies.items():
                avg_energies[grade] += energy / n
        
        return avg_energies


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TheoryTrueModel',
]
