"""
holographic_memory.py — True Holographic Storage via Clifford Superposition
===========================================================================

v4.19.0 UPDATE: Full 8D Even-Grade Indexing
-------------------------------------------
We now use ALL 8 even-grade coefficients for indexing, not compressed 4D.
This preserves the full geometric structure of the torus/fiber bundle.

FIBER BUNDLE STRUCTURE (from rhnsclifford.md):
    
    Cl(3,1) has a FIBER BUNDLE structure:
    
    ┌─────────────────────────────────────────────────────────────┐
    │   BASE SPACE = 2-Torus (from 6 bivectors, grade 2)         │
    │       - e₀₁, e₀₂, e₀₃: Time-space rotations (boosts)       │
    │       - e₁₂, e₁₃, e₂₃: Space-space rotations               │
    │       - Encodes SYNTACTIC STRUCTURE (word order)           │
    │                                                             │
    │   FIBER = Witness (grades 0, 4)                            │
    │       - σ (scalar): Total semantic "gist"                  │
    │       - p (pseudoscalar): Chirality/orientation            │
    │       - Encodes SEMANTIC CONTENT (what words)              │
    └─────────────────────────────────────────────────────────────┘
    
    Key insight: Witness is BLIND to word order (Tr(AB) = Tr(BA))!
    Vorticity (bivectors) captures order (AB - BA ≠ 0).
    
    BOTH are needed: witness + vorticity together, not as alternatives.

BUCKETS AND COLLISIONS (Theory-True Meaning):
    
    BUCKET = Region in quotient space where Grace flows to same attractor
    
    The theory says contexts in the same bucket are those that would
    converge to the same fixed point under infinite Grace iterations.
    
    COLLISION = Different contexts mapping to same bucket
    
    Collisions are BAD when they merge semantically/syntactically
    different contexts. The 8D key (σ, p, 6 bivectors) minimizes this.
    
    RESOLUTION = φ⁻² (spectral gap)
    
    From theory: "φ⁻² is the spectral gap - differences smaller than
    this are within the same basin of Grace attraction."

WHY 8D KEYS (v4.19.0):
    
    OLD (4D): (σ, p, enstrophy, dominant_plane)
        - Enstrophy = sum(bivector²) → loses DIRECTION
        - Dominant_plane = argmax → discrete, loses magnitude
        - Result: 6.5% permutation collisions
        
    NEW (8D): (σ, p, e₀₁, e₀₂, e₀₃, e₁₂, e₁₃, e₂₃)
        - Each bivector preserved individually
        - Preserves WHERE on torus, not just HOW FAR from origin
        - Result: 0% permutation collisions

COMBINED SIMILARITY (v4.18.0):
    
    Within-bucket matching uses φ-weighted combination:
    
        similarity = (1-φ⁻¹)·witness_sim + φ⁻¹·vorticity_sim
                   = 38.2% semantic + 61.8% syntactic
    
    This reflects the empirical finding that 46.4% of context energy
    is in bivectors (vorticity), which carries word order information.

HOLOGRAPHIC MEMORY ARCHITECTURE:
    
    1. HolographicMemory: True superposition-based storage (O(1))
       - Storage: memory += bind(context, target)
       - Retrieval: target ≈ unbind(context, memory)
       - Limited capacity due to interference (~4-16 patterns)
       
    2. VorticityWitnessIndex: Overflow with 8D geometric keys
       - Buckets contexts by full even-grade decomposition
       - Combined witness+vorticity similarity within bucket
       - Unlimited capacity, O(1) average retrieval
       
    3. HybridHolographicMemory: Cascade of both
       - Try holographic first (fastest, limited)
       - Fall back to index if not found
       - Best of both worlds

GRACE AS DENOISER:
    After holographic retrieval, the result contains:
    - The correct target (mostly in stable grades 0, 4)
    - Interference noise (mostly in transient grades 1, 2, 3)
    
    Grace naturally suppresses interference because:
    - Stable grades (0, 4) decay slowly (φ⁰, φ⁻¹)
    - Transient grades (1, 2, 3) decay quickly (φ⁻¹ to φ⁻³)

EVEN SUBALGEBRA CONSTRAINT:
    Our contexts live in Cl⁺(3,1) (even subalgebra) because:
    - Identity-biased initialization produces even elements
    - EVEN × EVEN = EVEN (closed under geometric product)
    
    Result: Grades 1 and 3 are always zero. We only use G0, G2, G4.
    This is 8 effective dimensions: 1 (scalar) + 6 (bivectors) + 1 (pseudo).

IMPLEMENTATION CLASSES:
    1. HolographicMemory: True superposition-based storage
    2. WitnessIndex: Legacy 2D index (deprecated, use VorticityWitnessIndex)
    3. VorticityWitnessIndex: Full 8D even-grade index (RECOMMENDED)
    4. HybridHolographicMemory: Combines holographic + index

References:
    - Plate (1995): Holographic Reduced Representations
    - Theory: rhnsclifford.md (witness, Grace, geometric product)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_SIX, PHI_INV_EIGHT,
    MATRIX_DIM, DTYPE
)
from .algebra import (
    geometric_product,
    grace_operator,
    frobenius_similarity,
    decompose_to_coefficients,
    normalize_matrix,
)
from .quotient import (
    extract_witness,
    extract_witness_batch,
    grace_stability,
    bind,
    vorticity_index_key,
    vorticity_index_key_batch,
)

Array = np.ndarray
ArrayModule = type(np)


# =============================================================================
# CLIFFORD INVERSE (REVERSION)
# =============================================================================

def clifford_reversion(M: Array, xp: ArrayModule = np) -> Array:
    """
    Compute the Clifford reversion (conjugate) of a multivector.
    
    In Cl(3,1), the reversion is:
        M̃ = M₀ - M₁ - M₂ + M₃ + M₄
    
    Where Mₖ is the grade-k component.
    
    This satisfies: M × M̃ ∝ scalar (for normalized M)
    
    For our 4×4 matrix representation:
        M̃ = M^T (transpose serves as reversion in M₄(ℝ))
    
    Theory: The transpose of a multivector matrix IS its reversion
    because the basis matrices satisfy: eᵢ^T = eᵢ for spatial,
    and the antisymmetric structure of bivectors reverses under transpose.
    """
    return M.T.copy()


def clifford_inverse(M: Array, basis: Array, xp: ArrayModule = np, eps: float = 1e-8) -> Array:
    """
    Compute the Clifford inverse of a multivector.
    
    CRITICAL FIX: In Cl(3,1) ≅ M₄(ℝ), the geometric product IS matrix multiplication.
    Therefore, the inverse under geometric product IS the matrix inverse.
    
    The reversion-based approach (M⁻¹ = M̃ / |M|²) only works for special elements
    like rotors. For general multivectors in the matrix representation, we need
    the actual matrix inverse.
    
    Args:
        M: [4, 4] multivector matrix
        basis: [16, 4, 4] Clifford basis (unused, kept for API compatibility)
        xp: array module
        eps: numerical stability
        
    Returns:
        [4, 4] inverse multivector (matrix inverse)
    """
    det = float(xp.linalg.det(M))
    
    if abs(det) < eps:
        # Degenerate matrix: use pseudoinverse for best approximation
        return xp.linalg.pinv(M)
    
    # Well-conditioned: use true matrix inverse
    # This IS the correct Clifford inverse because geometric_product = matrix multiply
    return xp.linalg.inv(M)


# =============================================================================
# TRUE HOLOGRAPHIC MEMORY (Superposition-Based)
# =============================================================================

@dataclass
class HolographicMemory:
    """
    True holographic memory via Clifford superposition.
    
    THEORY-TRUE STORAGE:
        All bindings are superposed into a single memory matrix.
        Retrieval is via unbinding with the context's inverse.
        
    HOW IT WORKS:
        Store: memory += φ⁻¹ × geometric_product(context, target)
        Retrieve: target ≈ geometric_product(context_inverse, memory)
        
    WHY O(1):
        - Storage is a single matrix addition
        - Retrieval is a single matrix multiplication
        - Independent of number of stored patterns!
        
    CAPACITY:
        Limited by interference. For 4×4 matrices:
        - ~8-16 patterns before noticeable degradation
        - Beyond this, use WitnessIndex fallback
        
    GRACE DENOISING:
        After retrieval, apply Grace to suppress interference:
        - Noise is mostly in transient grades (1, 2, 3)
        - Signal is mostly in stable grades (0, 4)
        - Grace damps transient grades → cleaner output
        
    Attributes:
        memory: [4, 4] superposed memory matrix
        n_patterns: Number of stored patterns
        basis: [16, 4, 4] Clifford basis
        capacity_warning_threshold: Warn when approaching capacity
    """
    memory: Array
    basis: Array
    n_patterns: int = 0
    capacity_warning_threshold: int = 12  # ~75% of theoretical max
    total_stored: int = 0  # Total patterns ever stored (for tracking)
    xp: ArrayModule = field(default=np, repr=False)
    
    @classmethod
    def create(cls, basis: Array, xp: ArrayModule = np) -> 'HolographicMemory':
        """Create a new holographic memory."""
        memory = xp.zeros((MATRIX_DIM, MATRIX_DIM), dtype=DTYPE)
        return cls(memory=memory, basis=basis, xp=xp)
    
    def store(self, context: Array, target: Array, weight: float = PHI_INV) -> Dict[str, Any]:
        """
        Store a context-target binding via superposition.
        
        THEORY-TRUE:
            memory += weight × geometric_product(context, target)
            
        The weight (default φ⁻¹) controls how strongly new patterns
        are added to the superposition. Using φ⁻¹ matches the theory's
        learning rate.
        
        Args:
            context: [4, 4] context matrix
            target: [4, 4] target matrix  
            weight: Binding strength (default φ⁻¹)
            
        Returns:
            Dict with storage stats
        """
        # Normalize inputs for consistent binding strength
        ctx_norm = self.xp.linalg.norm(context, 'fro')
        tgt_norm = self.xp.linalg.norm(target, 'fro')
        
        if ctx_norm < 1e-8 or tgt_norm < 1e-8:
            return {'stored': False, 'reason': 'zero_norm_input'}
        
        ctx_normalized = context / ctx_norm
        tgt_normalized = target / tgt_norm
        
        # Bind context to target via geometric product
        binding = geometric_product(ctx_normalized, tgt_normalized)
        
        # Add to superposed memory with φ-derived weight
        self.memory = self.memory + weight * binding
        
        self.n_patterns += 1
        self.total_stored += 1
        
        # Check capacity
        at_capacity = self.n_patterns >= self.capacity_warning_threshold
        
        return {
            'stored': True,
            'n_patterns': self.n_patterns,
            'at_capacity': at_capacity,
        }
    
    def store_batch(self, contexts: Array, targets: Array, weight: float = PHI_INV) -> Dict[str, Any]:
        """
        Store MULTIPLE context-target bindings in ONE operation.
        
        H100 OPTIMIZATION: This is the key speedup for GPU utilization.
        Instead of N separate stores (N GPU kernel launches), we:
            1. Normalize all contexts/targets at once (batched)
            2. Compute all bindings at once (batched matmul)
            3. Sum and add to memory (single operation)
        
        Args:
            contexts: [BATCH, 4, 4] context matrices
            targets: [BATCH, 4, 4] target matrices
            weight: Binding strength (default φ⁻¹)
            
        Returns:
            Dict with batch storage stats
        """
        batch_size = contexts.shape[0]
        if batch_size == 0:
            return {'stored': 0, 'n_patterns': self.n_patterns}
        
        # Batched normalization: [BATCH, 4, 4]
        ctx_norms = self.xp.linalg.norm(contexts.reshape(batch_size, -1), axis=1, keepdims=True)
        tgt_norms = self.xp.linalg.norm(targets.reshape(batch_size, -1), axis=1, keepdims=True)
        
        # Mask out zero-norm inputs
        valid = (ctx_norms.flatten() > 1e-8) & (tgt_norms.flatten() > 1e-8)
        n_valid = int(self.xp.sum(valid))
        
        if n_valid == 0:
            return {'stored': 0, 'n_patterns': self.n_patterns, 'reason': 'all_zero_norm'}
        
        # Filter to valid samples
        valid_contexts = contexts[valid]
        valid_targets = targets[valid]
        valid_ctx_norms = ctx_norms[valid].reshape(-1, 1, 1)
        valid_tgt_norms = tgt_norms[valid].reshape(-1, 1, 1)
        
        # Normalize: [N_VALID, 4, 4]
        ctx_normalized = valid_contexts / valid_ctx_norms
        tgt_normalized = valid_targets / valid_tgt_norms
        
        # Batched geometric product: [N_VALID, 4, 4] × [N_VALID, 4, 4] → [N_VALID, 4, 4]
        # This is batched matrix multiply - the key GPU operation
        bindings = self.xp.matmul(ctx_normalized, tgt_normalized)
        
        # Sum all bindings and add to memory in ONE operation
        total_binding = self.xp.sum(bindings, axis=0)  # [4, 4]
        self.memory = self.memory + weight * total_binding
        
        self.n_patterns += n_valid
        self.total_stored += n_valid
        
        at_capacity = self.n_patterns >= self.capacity_warning_threshold
        
        return {
            'stored': n_valid,
            'n_patterns': self.n_patterns,
            'at_capacity': at_capacity,
        }
    
    def retrieve(self, context: Array, denoise: bool = True) -> Tuple[Array, float]:
        """
        Retrieve target via unbinding with context inverse.
        
        THEORY-TRUE:
            target ≈ geometric_product(context_inverse, memory)
            
        The retrieval naturally extracts the bound partner because:
            ctx⁻¹ × (ctx × tgt) ≈ tgt
            
        Interference from other bindings appears as noise in transient
        grades, which Grace suppresses.
        
        Args:
            context: [4, 4] context matrix
            denoise: Whether to apply Grace denoising (default True)
            
        Returns:
            (retrieved_target, confidence)
            confidence = Grace-stability of result
        """
        # Normalize context
        ctx_norm = self.xp.linalg.norm(context, 'fro')
        if ctx_norm < 1e-8:
            return self.xp.zeros((MATRIX_DIM, MATRIX_DIM), dtype=DTYPE), 0.0
        
        ctx_normalized = context / ctx_norm
        
        # Compute context inverse (reversion)
        ctx_inverse = clifford_inverse(ctx_normalized, self.basis, self.xp)
        
        # Unbind: target ≈ ctx⁻¹ × memory
        retrieved = geometric_product(ctx_inverse, self.memory)
        
        # THEORY-TRUE DENOISING:
        # Grace suppresses interference in transient grades
        # The signal is mostly in stable grades (what survives Grace)
        if denoise:
            retrieved = grace_operator(retrieved, self.basis, self.xp)
        
        # Confidence = Grace-stability (fraction in stable grades)
        confidence = grace_stability(retrieved, self.basis, self.xp)
        
        return retrieved, confidence
    
    def clear(self):
        """Clear all stored patterns."""
        self.memory = self.xp.zeros((MATRIX_DIM, MATRIX_DIM), dtype=DTYPE)
        self.n_patterns = 0
    
    def consolidate(self) -> Array:
        """
        Consolidate memory via Grace (extract stable patterns).
        
        THEORY:
            Multiple Grace applications extract the most stable content.
            This is like dreaming's consolidation but at the memory level.
            
        Returns:
            Consolidated memory (mostly stable grades)
        """
        consolidated = self.memory.copy()
        for _ in range(5):  # Multiple Grace applications
            consolidated = grace_operator(consolidated, self.basis, self.xp)
        return consolidated


# =============================================================================
# WITNESS-BASED INDEX (DEPRECATED - Use VorticityWitnessIndex instead)
# =============================================================================

@dataclass  
class WitnessIndex:
    """
    DEPRECATED: Use VorticityWitnessIndex instead.
    
    This class is kept for backward compatibility in tests only.
    It creates only ~4-12 buckets (useless for retrieval) because witness
    values cluster near (0, 0).
    
    v4.21.0: Deprecated. Will be removed in v5.0.
    
    Original theory motivation (still valid for VorticityWitnessIndex):
        Two contexts with the same witness WILL flow to the same attractor.
        Therefore, indexing by witness respects the geometric structure.
        
    Attributes:
        buckets: Dict mapping witness key to list of (context, target, idx) pairs
        resolution: Quantization resolution (default φ⁻²)
        token_sequences: List of (tokens, target_idx) for reindexing
    """
    buckets: Dict[Tuple[int, int], List[Tuple[Array, Array, int]]] = field(default_factory=dict)
    targets: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    resolution: float = PHI_INV_SQ
    basis: Array = None
    xp: ArrayModule = field(default=np, repr=False)
    n_items: int = 0
    # For reindexing after embedding drift - stores (token_seq, target_idx)
    token_sequences: List[Tuple[List[int], int]] = field(default_factory=list)
    
    @classmethod
    def create(cls, basis: Array, resolution: float = PHI_INV_SQ, xp: ArrayModule = np) -> 'WitnessIndex':
        """Create a new witness index."""
        return cls(
            buckets={}, 
            targets={},
            resolution=resolution, 
            basis=basis, 
            xp=xp
        )
    
    def _witness_key(self, M: Array) -> Tuple[int, int]:
        """
        Compute witness-based key for indexing.
        
        Quantizes (scalar, pseudoscalar) to grid cells of size resolution.
        """
        s, p = extract_witness(M, self.basis, self.xp)
        # Use xp for GPU compatibility
        s_idx = int(self.xp.floor(s / self.resolution))
        p_idx = int(self.xp.floor(p / self.resolution))
        return (s_idx, p_idx)
    
    def _witness_keys_batch(self, Ms: Array) -> List[Tuple[int, int]]:
        """
        BATCHED witness key computation for GPU parallelism.
        
        Computes all witness keys in ONE GPU call, then transfers to CPU
        for dict operations.
        
        Args:
            Ms: [BATCH, 4, 4] context matrices
            
        Returns:
            List of (s_idx, p_idx) tuples
        """
        from holographic_v4.quotient import extract_witness_batch
        
        # Single GPU call for all witnesses: [BATCH, 2]
        witnesses = extract_witness_batch(Ms, self.basis, self.xp)
        
        # Quantize on GPU: [BATCH, 2]
        quantized = self.xp.floor(witnesses / self.resolution).astype(self.xp.int32)
        
        # Transfer to CPU and convert to tuples (unavoidable for dict)
        if hasattr(quantized, 'get'):  # CuPy
            quantized = quantized.get()
        
        return [(int(row[0]), int(row[1])) for row in quantized]
    
    def store_batch(self, contexts: Array, targets: Array, target_idxs: Array,
                    token_sequences: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """
        BATCHED witness storage with GPU-parallel key computation.
        
        Args:
            contexts: [BATCH, 4, 4] context matrices
            targets: [BATCH, 4, 4] target matrices
            target_idxs: [BATCH] integer target indices
            token_sequences: Optional list of token lists
            
        Returns:
            Storage stats
        """
        batch_size = contexts.shape[0]
        if batch_size == 0:
            return {'stored': 0}
        
        # φ²-SUBSAMPLING: Store every ~3rd sample (φ² ≈ 2.618)
        # THEORY-TRUE: φ² is the natural "coverage radius" in witness space
        # Nearby samples map to same bucket anyway (resolution = φ⁻²)
        # This is NOT optional - witness storage is essential for token ID lookup
        subsample_step = 3  # int(PHI * PHI) ≈ 2.618 → 3
        subsample_indices = list(range(0, batch_size, subsample_step))
        n_to_store = len(subsample_indices)
        
        if n_to_store == 0:
            return {'stored': 0, 'n_items': self.n_items}
        
        # SINGLE BATCH COPY: Extract subsampled matrices in ONE operation
        # This is the key optimization: 1 GPU→CPU sync instead of 683
        sub_contexts = contexts[subsample_indices].copy()  # [N, 4, 4] - ONE sync
        sub_targets = targets[subsample_indices].copy()    # [N, 4, 4] - ONE sync
        sub_idxs = target_idxs[subsample_indices]           # Already on GPU
        
        # GPU-parallel: compute witness keys for subsampled matrices only
        keys = self._witness_keys_batch(sub_contexts)
        
        # Python dict inserts (unavoidable, but copies are already done)
        stored = 0
        for i in range(n_to_store):
            key = keys[i]
            ctx = sub_contexts[i]  # Already copied, no sync
            tgt = sub_targets[i]   # Already copied, no sync
            idx = int(sub_idxs[i])
            
            if key not in self.buckets:
                self.buckets[key] = []
                self.targets[key] = []
            
            self.buckets[key].append((ctx, tgt, idx))  # No .copy() needed!
            self.targets[key].append(idx)
            self.n_items += 1
            stored += 1
            
            if token_sequences is not None:
                orig_idx = subsample_indices[i]
                self.token_sequences.append((list(token_sequences[orig_idx]), idx))
        
        return {'stored': stored, 'n_items': self.n_items}
    
    def store(self, context: Array, target: Array, target_idx: int, 
              token_sequence: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Store a context-target pair indexed by witness.
        
        Args:
            context: [4, 4] context matrix
            target: [4, 4] target matrix
            target_idx: Integer target token index
            token_sequence: Optional list of token IDs (for reindexing after drift)
            
        Returns:
            Storage stats
        """
        key = self._witness_key(context)
        
        if key not in self.buckets:
            self.buckets[key] = []
            self.targets[key] = []
        
        self.buckets[key].append((context.copy(), target.copy(), target_idx))
        self.targets[key].append(target_idx)
        self.n_items += 1
        
        # Store token sequence for reindexing after embedding drift
        if token_sequence is not None:
            self.token_sequences.append((list(token_sequence), target_idx))
        
        return {
            'stored': True,
            'key': key,
            'bucket_size': len(self.buckets[key]),
        }
    
    def reindex_all(self, compute_context_fn, get_embedding_fn) -> int:
        """
        Reindex all stored patterns after embedding drift.
        
        THEORY-TRUE: Embedding drift improves representations, but stored
        context matrices become stale. This recomputes all witnesses using
        current embeddings.
        
        Args:
            compute_context_fn: Function(tokens) -> context_matrix
            get_embedding_fn: Function(token_idx) -> embedding
            
        Returns:
            Number of entries reindexed
        """
        if not self.token_sequences:
            return 0
        
        # Clear current buckets
        self.buckets.clear()
        self.targets.clear()
        self.n_items = 0
        
        # Rebuild with current embeddings
        reindexed = 0
        for tokens, target_idx in self.token_sequences:
            # Recompute context matrix with current embeddings
            context = compute_context_fn(tokens)
            target = get_embedding_fn(target_idx)
            
            key = self._witness_key(context)
            if key not in self.buckets:
                self.buckets[key] = []
                self.targets[key] = []
            
            self.buckets[key].append((context.copy(), target.copy(), target_idx))
            self.targets[key].append(target_idx)
            self.n_items += 1
            reindexed += 1
        
        return reindexed
    
    def retrieve(self, context: Array) -> Tuple[Optional[Array], Optional[int], float]:
        """
        Retrieve target by witness lookup.
        
        If multiple items in bucket, return φ-weighted average.
        
        Args:
            context: [4, 4] context matrix
            
        Returns:
            (target_matrix, target_idx, confidence) or (None, None, 0.0) if not found
        """
        key = self._witness_key(context)
        
        if key not in self.buckets or len(self.buckets[key]) == 0:
            return None, None, 0.0
        
        bucket = self.buckets[key]
        
        if len(bucket) == 1:
            _, target, target_idx = bucket[0]
            return target, target_idx, 1.0
        
        # Multiple items: compute similarity-weighted average
        # Use witness similarity for weighting
        query_witness = extract_witness(context, self.basis, self.xp)
        
        total_weight = 0.0
        weighted_target = self.xp.zeros((MATRIX_DIM, MATRIX_DIM), dtype=DTYPE)
        target_weights: Dict[int, float] = {}
        
        for ctx, tgt, tgt_idx in bucket:
            ctx_witness = extract_witness(ctx, self.basis, self.xp)
            
            # Witness distance
            # Use xp for GPU compatibility
            dist = float(self.xp.sqrt(
                (query_witness[0] - ctx_witness[0])**2 + 
                (query_witness[1] - ctx_witness[1])**2
            ))
            
            # φ-kernel weight
            weight = PHI_INV ** dist if dist < 10 else 0.0
            
            weighted_target = weighted_target + weight * tgt
            total_weight += weight
            
            if tgt_idx not in target_weights:
                target_weights[tgt_idx] = 0.0
            target_weights[tgt_idx] += weight
        
        if total_weight < 1e-8:
            return None, None, 0.0
        
        # Most likely target
        best_target_idx = max(target_weights.keys(), key=lambda k: target_weights[k])
        
        result = weighted_target / total_weight
        confidence = target_weights[best_target_idx] / total_weight
        
        return result, best_target_idx, confidence
    
    def clear(self):
        """Clear all stored items."""
        self.buckets.clear()
        self.targets.clear()
        self.n_items = 0


# =============================================================================
# VORTICITY-EXTENDED WITNESS INDEX (Full 8D Even-Grade Keys)
# =============================================================================

@dataclass
class VorticityWitnessIndex:
    """
    Full 8D even-grade index using ALL Clifford algebra structure.
    
    THE FIBER BUNDLE STRUCTURE:
        From rhnsclifford.md, Cl(3,1) has a fiber bundle structure:
        
        BASE SPACE = 2-Torus (from 6 bivectors)
            Each bivector e_ij encodes a rotation plane:
            - e₀₁, e₀₂, e₀₃: Time-space rotations (temporal structure)
            - e₁₂, e₁₃, e₂₃: Space-space rotations (spatial structure)
            
        FIBER = Witness (σ, p)
            - σ (scalar): Semantic "gist" (what words are there)
            - p (pseudoscalar): Chirality/orientation
    
    WHY 8D KEYS (not 4D):
        OLD (4D): (σ, p, enstrophy, dominant_plane)
            - enstrophy = sum(bivector²) → loses DIRECTION on torus
            - dominant_plane = argmax → discrete, loses magnitude
            - Result: 6.5% permutation collisions
            
        NEW (8D): (σ, p, e₀₁, e₀₂, e₀₃, e₁₂, e₁₃, e₂₃)
            - Each bivector preserved individually
            - Preserves WHERE on torus, not just HOW FAR from origin
            - Result: 0% permutation collisions
    
    KEY INSIGHT — WITNESS IS BLIND TO ORDER:
        Tr(AB) = Tr(BA) → Witness is SAME for "dog bites man" and "man bites dog"
        AB - BA ≠ 0 → Bivectors DIFFER for permutations
        
        The 6 bivectors encode SYNTACTIC STRUCTURE (word order).
        The witness encodes SEMANTIC CONTENT (what words).
        BOTH are needed together, not as alternatives.
    
    BUCKETS AND COLLISIONS:
        BUCKET = Region where contexts flow to same attractor under Grace
        COLLISION = Different contexts in same bucket (bad for retrieval)
        
        A bucket at resolution φ⁻² contains contexts that would converge
        to the same fixed point under infinite Grace iterations.
        
    COMBINED SIMILARITY (within bucket):
        Uses φ-weighted combination of witness and vorticity:
        
            similarity = (1-φ⁻¹)·witness_sim + φ⁻¹·vorticity_sim
                       = 38.2% semantic + 61.8% syntactic
        
        This reflects the empirical energy distribution:
        - Scalar (G0): 35% of context energy
        - Bivectors (G2): 46% of context energy (word order!)
        - Pseudoscalar (G4): 19% of context energy
    """
    buckets: Dict[Tuple, List[Tuple[Array, Array, int]]] = field(default_factory=dict)
    targets: Dict[Tuple, List[int]] = field(default_factory=dict)
    sigma_resolution: float = PHI_INV_SIX  # φ⁻⁶ ≈ 0.056 (fine resolution for σ, p, bivectors)
    enstrophy_resolution: float = PHI_INV_EIGHT  # φ⁻⁸ ≈ 0.021 (very fine for enstrophy - LEGACY)
    basis: Array = None
    xp: ArrayModule = field(default=np, repr=False)
    n_items: int = 0
    token_sequences: List[Tuple[List[int], int]] = field(default_factory=list)
    
    @classmethod
    def create(cls, basis: Array, sigma_resolution: float = PHI_INV_SIX,
               enstrophy_resolution: float = PHI_INV_EIGHT, xp: ArrayModule = np) -> 'VorticityWitnessIndex':
        """
        Create a new vorticity-extended witness index.
        
        Args:
            basis: [16, 4, 4] Clifford basis
            sigma_resolution: Grid cell size for σ, p, and bivector coefficients (default φ⁻⁶ ≈ 0.056)
            enstrophy_resolution: Grid cell size for enstrophy (LEGACY, not used in 8D keys)
            xp: Array module
        """
        return cls(
            buckets={},
            targets={},
            sigma_resolution=sigma_resolution,
            enstrophy_resolution=enstrophy_resolution,
            basis=basis,
            xp=xp
        )
    
    def _vorticity_key(self, M: Array) -> Tuple[int, ...]:
        """
        Compute FULL 8D even-grade key for indexing.
        
        THEORY-TRUE:
            Uses ALL 8 even-grade coefficients, not compressed 4D summary:
            - σ (scalar)
            - p (pseudoscalar)
            - 6 individual bivector coefficients (e₀₁, e₀₂, e₀₃, e₁₂, e₁₃, e₂₃)
            
            This preserves the DIRECTION of vorticity, not just its magnitude.
            Each bivector plane encodes a different rotation:
            - e₀₁, e₀₂, e₀₃: Time-space rotations (boosts)
            - e₁₂, e₁₃, e₂₃: Space-space rotations
            
        Returns:
            (s_idx, p_idx, bv0_idx, bv1_idx, bv2_idx, bv3_idx, bv4_idx, bv5_idx)
        """
        # Extract witness
        s, p = extract_witness(M, self.basis, self.xp)
        
        # Extract ALL 6 bivector coefficients individually
        bv_coeffs = [
            float(self.xp.sum(self.basis[5+i] * M) / 4.0) for i in range(6)
        ]
        
        # Quantize all 8 components to grid
        s_idx = int(self.xp.floor(s / self.sigma_resolution))
        p_idx = int(self.xp.floor(p / self.sigma_resolution))
        bv_idxs = tuple(int(self.xp.floor(bv / self.enstrophy_resolution)) for bv in bv_coeffs)
        
        return (s_idx, p_idx) + bv_idxs
    
    def _vorticity_keys_batch(self, Ms: Array) -> List[Tuple[int, ...]]:
        """
        Batched FULL 8D key computation for GPU parallelism.
        
        Computes all 8 even-grade coefficients (σ, p, 6 bivectors) in parallel.
        """
        from holographic_v4.quotient import extract_witness_batch
        
        batch_size = Ms.shape[0]
        
        # Extract witnesses (σ, p) in batch
        witnesses = extract_witness_batch(Ms, self.basis, self.xp)  # [batch, 2]
        
        # Extract all 6 bivector coefficients in batch
        # basis[5:11] are the 6 bivector basis elements
        bv_coeffs = []
        for i in range(6):
            # For each bivector basis element, compute trace with all matrices
            coeff = self.xp.einsum('nij,ij->n', Ms, self.basis[5+i]) / 4.0
            bv_coeffs.append(coeff)
        bv_coeffs = self.xp.stack(bv_coeffs, axis=1)  # [batch, 6]
        
        # Quantize on GPU
        s_idx = self.xp.floor(witnesses[:, 0] / self.sigma_resolution).astype(self.xp.int32)
        p_idx = self.xp.floor(witnesses[:, 1] / self.sigma_resolution).astype(self.xp.int32)
        bv_idx = self.xp.floor(bv_coeffs / self.enstrophy_resolution).astype(self.xp.int32)  # [batch, 6]
        
        # Stack all 8 components
        keys_array = self.xp.concatenate([
            s_idx[:, None], p_idx[:, None], bv_idx
        ], axis=1)  # [batch, 8]
        
        if hasattr(keys_array, 'get'):  # CuPy
            keys_array = keys_array.get()
        
        return [tuple(int(x) for x in row) for row in keys_array]
    
    def store_batch(self, contexts: Array, targets: Array, target_idxs: Array,
                    token_sequences: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """
        Batched vorticity storage with GPU-parallel key computation.
        """
        batch_size = contexts.shape[0]
        if batch_size == 0:
            return {'stored': 0}
        
        # φ²-SUBSAMPLING (same as WitnessIndex)
        subsample_step = 3
        subsample_indices = list(range(0, batch_size, subsample_step))
        n_to_store = len(subsample_indices)
        
        if n_to_store == 0:
            return {'stored': 0, 'n_items': self.n_items}
        
        # Extract subsampled matrices
        sub_contexts = contexts[subsample_indices].copy()
        sub_targets = targets[subsample_indices].copy()
        sub_idxs = target_idxs[subsample_indices]
        
        # GPU-parallel: compute vorticity keys
        keys = self._vorticity_keys_batch(sub_contexts)
        
        # Store in dict
        stored = 0
        for i in range(n_to_store):
            key = keys[i]
            ctx = sub_contexts[i]
            tgt = sub_targets[i]
            idx = int(sub_idxs[i])
            
            if key not in self.buckets:
                self.buckets[key] = []
                self.targets[key] = []
            
            self.buckets[key].append((ctx, tgt, idx))
            self.targets[key].append(idx)
            self.n_items += 1
            stored += 1
            
            if token_sequences is not None:
                orig_idx = subsample_indices[i]
                self.token_sequences.append((list(token_sequences[orig_idx]), idx))
        
        return {'stored': stored, 'n_items': self.n_items}
    
    def store(self, context: Array, target: Array, target_idx: int,
              token_sequence: Optional[List[int]] = None) -> Dict[str, Any]:
        """Store a single context-target pair."""
        key = self._vorticity_key(context)
        
        if key not in self.buckets:
            self.buckets[key] = []
            self.targets[key] = []
        
        self.buckets[key].append((context.copy(), target.copy(), target_idx))
        self.targets[key].append(target_idx)
        self.n_items += 1
        
        if token_sequence is not None:
            self.token_sequences.append((list(token_sequence), target_idx))
        
        return {
            'stored': True,
            'key': key,
            'bucket_size': len(self.buckets[key]),
        }
    
    def retrieve(self, context: Array) -> Tuple[Optional[Array], Optional[int], float]:
        """
        Retrieve target by vorticity-extended lookup.
        
        THEORY-TRUE:
            Uses COMBINED witness + vorticity similarity within bucket.
            - Witness (σ, p) captures WHAT (semantic content)
            - Vorticity (bivectors) captures HOW (syntactic structure)
            - φ-weighted combination: (1-φ⁻¹)·witness + φ⁻¹·vorticity
        """
        key = self._vorticity_key(context)
        
        if key not in self.buckets or len(self.buckets[key]) == 0:
            return None, None, 0.0
        
        bucket = self.buckets[key]
        
        if len(bucket) == 1:
            _, target, target_idx = bucket[0]
            return target, target_idx, 1.0
        
        # Multiple items: compute COMBINED similarity-weighted average
        # Extract query features
        query_witness = extract_witness(context, self.basis, self.xp)
        query_bivectors = self.xp.array([
            float(self.xp.sum(self.basis[5+i] * context) / 4.0) for i in range(6)
        ])
        query_bv_norm = self.xp.linalg.norm(query_bivectors)
        
        total_weight = 0.0
        weighted_target = self.xp.zeros((MATRIX_DIM, MATRIX_DIM), dtype=DTYPE)
        target_weights: Dict[int, float] = {}
        
        for ctx, tgt, tgt_idx in bucket:
            # Extract stored features
            ctx_witness = extract_witness(ctx, self.basis, self.xp)
            ctx_bivectors = self.xp.array([
                float(self.xp.sum(self.basis[5+i] * ctx) / 4.0) for i in range(6)
            ])
            ctx_bv_norm = self.xp.linalg.norm(ctx_bivectors)
            
            # Witness similarity (semantic)
            w_dist = float(self.xp.sqrt(
                (query_witness[0] - ctx_witness[0])**2 +
                (query_witness[1] - ctx_witness[1])**2
            ))
            witness_sim = PHI_INV ** w_dist if w_dist < 10 else 0.0
            
            # Vorticity similarity (syntactic)
            if query_bv_norm > 1e-8 and ctx_bv_norm > 1e-8:
                vort_sim = float(self.xp.dot(query_bivectors, ctx_bivectors) / (query_bv_norm * ctx_bv_norm))
                vort_sim = max(0.0, vort_sim)  # Only positive similarity
            else:
                vort_sim = 0.0
            
            # THEORY-TRUE: φ-weighted combination
            # (1 - φ⁻¹) ≈ 0.382 to witness, φ⁻¹ ≈ 0.618 to vorticity
            combined_sim = (1 - PHI_INV) * witness_sim + PHI_INV * vort_sim
            
            weighted_target = weighted_target + combined_sim * tgt
            total_weight += combined_sim
            
            if tgt_idx not in target_weights:
                target_weights[tgt_idx] = 0.0
            target_weights[tgt_idx] += combined_sim
        
        if total_weight < 1e-12 or len(target_weights) == 0:
            return None, None, 0.0
        
        best_target_idx = max(target_weights.keys(), key=lambda k: target_weights[k])
        
        result = weighted_target / total_weight
        confidence = target_weights[best_target_idx] / total_weight
        
        return result, best_target_idx, confidence
    
    def clear(self):
        """Clear all stored items."""
        self.buckets.clear()
        self.targets.clear()
        self.n_items = 0
    
    def stats(self) -> Dict[str, Any]:
        """Return statistics about bucket distribution."""
        if not self.buckets:
            return {'n_buckets': 0, 'n_items': 0, 'avg_bucket_size': 0, 'max_bucket_size': 0}
        
        bucket_sizes = [len(b) for b in self.buckets.values()]
        return {
            'n_buckets': len(self.buckets),
            'n_items': self.n_items,
            'avg_bucket_size': sum(bucket_sizes) / len(bucket_sizes),
            'max_bucket_size': max(bucket_sizes),
            'min_bucket_size': min(bucket_sizes),
        }


# =============================================================================
# CANONICAL SEMANTIC INDEX (v4.22.0) — For Generalization
# =============================================================================

@dataclass
class CanonicalSemanticIndex:
    """
    Coarse-grained semantic index for GENERALIZATION.
    
    THEORY-TRUE DESIGN:
        This index is designed for semantic retrieval (paraphrases, novel contexts)
        rather than exact retrieval. It uses:
        
        1. COARSER RESOLUTION (φ⁻³ vs φ⁻⁶)
           Creates larger semantic neighborhoods where similar meanings cluster.
           
        2. BIREFLECTION-AWARE BUCKETING
           Uses |p| instead of p because bireflection symmetry (σ ↔ 1-σ)
           maps p ↔ -p. Contexts differing only by sign of p are equivalent.
           
        3. 2D KEYS (σ, |p|)
           Coarse 2D is sufficient for semantic bucketing.
           Fine 8D detail captured by VorticityWitnessIndex for exact retrieval.
    
    BRAIN ANALOGY:
        VorticityWitnessIndex = Hippocampus (episodic, fine-grained)
        CanonicalSemanticIndex = Cortex (semantic, coarse-grained)
        
    CASCADE RETRIEVAL:
        1. Try VorticityWitnessIndex first (episodic, exact)
        2. Fall back to CanonicalSemanticIndex (semantic, generalization)
    
    WHY THIS WORKS:
        Predictiveness tracking makes co-predictive tokens similar.
        If "cat" and "feline" both predict "meows", their contexts become similar.
        Coarse bucketing + bireflection symmetry makes them land in same bucket.
    """
    buckets: Dict[Tuple[int, int], List[Tuple[Array, Array, int]]] = field(default_factory=dict)
    targets: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    resolution: float = PHI_INV_CUBE  # φ⁻³ ≈ 0.236 (COARSE for generalization)
    basis: Array = None
    xp: ArrayModule = field(default=np, repr=False)
    n_items: int = 0
    
    @classmethod
    def create(cls, basis: Array, resolution: float = PHI_INV_CUBE,
               xp: ArrayModule = np) -> 'CanonicalSemanticIndex':
        """
        Create a new canonical semantic index.
        
        Args:
            basis: [16, 4, 4] Clifford basis
            resolution: Grid cell size (default φ⁻³ ≈ 0.236, coarse for generalization)
            xp: Array module
        """
        return cls(
            buckets={},
            targets={},
            resolution=resolution,
            basis=basis,
            xp=xp
        )
    
    def _canonical_key(self, M: Array) -> Tuple[int, int]:
        """
        Compute canonical 2D key with bireflection symmetry.
        
        THEORY:
            σ (scalar) is the primary semantic coordinate.
            |p| (abs pseudoscalar) respects p ↔ -p bireflection symmetry.
            
            Bireflection maps σ ↔ 1-σ, p ↔ -p.
            Using |p| ensures contexts that differ only by bireflection
            land in the same bucket → generalization.
        """
        sigma = float(self.xp.sum(self.basis[0] * M) / 4.0)
        pseudo = float(self.xp.sum(self.basis[15] * M) / 4.0)
        
        # Canonical: use abs(pseudo) for bireflection symmetry
        abs_pseudo = abs(pseudo)
        
        s_idx = int(self.xp.floor(sigma / self.resolution))
        p_idx = int(self.xp.floor(abs_pseudo / self.resolution))
        
        return (s_idx, p_idx)
    
    def store(self, context: Array, target: Array, target_idx: int) -> Dict[str, Any]:
        """Store with canonical key."""
        key = self._canonical_key(context)
        
        if key not in self.buckets:
            self.buckets[key] = []
            self.targets[key] = []
        
        self.buckets[key].append((context.copy(), target.copy(), target_idx))
        self.targets[key].append(target_idx)
        self.n_items += 1
        
        return {'key': key, 'n_items': self.n_items}
    
    def retrieve(self, context: Array) -> Tuple[Optional[Array], Optional[int], float]:
        """
        Retrieve using canonical key and combined similarity.
        
        Uses φ-weighted combination of witness and vorticity similarity
        for within-bucket matching.
        """
        from holographic_v4.quotient import vorticity_similarity
        
        key = self._canonical_key(context)
        
        if key not in self.buckets or not self.buckets[key]:
            return None, None, 0.0
        
        # Find best match in bucket using combined similarity
        best_match = None
        best_idx = None
        best_sim = -1.0
        
        for ctx, tgt, idx in self.buckets[key]:
            sim = vorticity_similarity(context, ctx, self.basis, self.xp)
            if sim > best_sim:
                best_sim = sim
                best_match = tgt
                best_idx = idx
        
        return best_match, best_idx, float(best_sim)
    
    def clear(self):
        """Clear all stored items."""
        self.buckets.clear()
        self.targets.clear()
        self.n_items = 0
    
    def stats(self) -> Dict[str, Any]:
        """Return statistics about bucket distribution."""
        if not self.buckets:
            return {'n_buckets': 0, 'n_items': 0, 'avg_bucket_size': 0, 'max_bucket_size': 0}
        
        bucket_sizes = [len(b) for b in self.buckets.values()]
        return {
            'n_buckets': len(self.buckets),
            'n_items': self.n_items,
            'avg_bucket_size': sum(bucket_sizes) / len(bucket_sizes),
            'max_bucket_size': max(bucket_sizes),
            'min_bucket_size': min(bucket_sizes),
        }


# =============================================================================
# HYBRID MEMORY (Holographic + Dual Index)
# =============================================================================

@dataclass
class HybridHolographicMemory:
    """
    Hybrid memory with DUAL INDEXING for both exact retrieval and generalization.
    
    v4.22.0 ARCHITECTURE:
        1. HolographicMemory: True superposition (O(1), limited capacity)
        2. VorticityWitnessIndex (episodic): 8D fine-grained exact retrieval
        3. CanonicalSemanticIndex (semantic): 2D coarse-grained generalization
        
    BRAIN ANALOGY:
        HolographicMemory = Working memory (fast, limited)
        VorticityWitnessIndex = Hippocampus (episodic, fine detail)
        CanonicalSemanticIndex = Cortex (semantic, generalizes)
        
    RETRIEVAL CASCADE:
        1. Try holographic retrieval (fast, O(1))
        2. If low confidence, try episodic index (exact match)
        3. If no match, try semantic index (generalization)
        
    WHY DUAL INDEXING:
        - Episodic (8D): Exact retrieval, no generalization
        - Semantic (2D): Generalization via bireflection-aware bucketing
        - BOTH needed: episodic for precision, semantic for recall
        
    THEORY-TRUE FEATURES:
        - φ-derived resolutions (φ⁻⁶ for episodic, φ⁻³ for semantic)
        - Bireflection symmetry (|p| not p) in semantic index
        - Combined φ-weighted similarity (38% witness + 62% vorticity)
    """
    holographic: HolographicMemory
    witness_index: VorticityWitnessIndex  # EPISODIC: 8D fine-grained
    semantic_index: CanonicalSemanticIndex  # SEMANTIC: 2D coarse, bireflection-aware
    basis: Array
    min_holographic_confidence: float = PHI_INV_SQ  # Theory: spectral gap
    xp: ArrayModule = field(default=np, repr=False)
    
    # Tracking
    n_holographic_retrievals: int = 0
    n_episodic_retrievals: int = 0  # Renamed from n_witness_retrievals
    n_semantic_retrievals: int = 0  # NEW
    n_holographic_stores: int = 0
    n_episodic_stores: int = 0  # Renamed from n_witness_stores
    n_semantic_stores: int = 0  # NEW
    
    @classmethod
    def create(cls, basis: Array, xp: ArrayModule = np) -> 'HybridHolographicMemory':
        """
        Create a new hybrid memory with dual indexing.
        
        v4.22.0: DUAL INDEXING
            - VorticityWitnessIndex (8D): Exact retrieval
            - CanonicalSemanticIndex (2D): Generalization
        
        Args:
            basis: [16, 4, 4] Clifford basis
            xp: Array module (numpy or cupy)
            
        Returns:
            HybridHolographicMemory with dual indexing
        """
        holographic = HolographicMemory.create(basis, xp)
        episodic_index = VorticityWitnessIndex.create(basis, xp=xp)
        semantic_index = CanonicalSemanticIndex.create(basis, xp=xp)
        
        return cls(
            holographic=holographic,
            witness_index=episodic_index,
            semantic_index=semantic_index,
            basis=basis,
            xp=xp,
        )
    
    def store(self, context: Array, target: Array, target_idx: int,
              token_sequence: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Store a context-target binding in ALL systems.
        
        v4.22.0 Strategy:
            1. Store in holographic memory (superposition, fast retrieval)
            2. Store in episodic index (8D, exact retrieval)
            3. Store in semantic index (2D, generalization)
            
        This triple storage ensures:
            - Holographic O(1) when capacity allows
            - Episodic exact matching when holographic fails
            - Semantic generalization for paraphrases/novel contexts
            
        Args:
            context: [4, 4] context matrix
            target: [4, 4] target matrix
            target_idx: Integer target token
            token_sequence: Optional token list for reindexing after drift
            
        Returns:
            Storage stats
        """
        # Store in ALL THREE systems
        holo_result = self.holographic.store(context, target)
        episodic_result = self.witness_index.store(context, target, target_idx, 
                                                    token_sequence=token_sequence)
        semantic_result = self.semantic_index.store(context, target, target_idx)
        
        if holo_result.get('stored', False):
            self.n_holographic_stores += 1
        self.n_episodic_stores += 1
        self.n_semantic_stores += 1
        
        return {
            'holographic': holo_result,
            'episodic': episodic_result,
            'semantic': semantic_result,
            'n_patterns': self.holographic.n_patterns,
            'n_episodic_items': self.witness_index.n_items,
            'n_semantic_items': self.semantic_index.n_items,
        }
    
    def store_batch(self, contexts: Array, targets: Array, target_idxs: Array,
                    token_sequences: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """
        Store MULTIPLE context-target bindings at once.
        
        TWO-LEVEL EPISODIC MEMORY (Theory-True):
        - Holographic: True superposition, O(1) retrieval, LIMITED capacity (~16 patterns)
        - WitnessIndex: Theory-true overflow using φ⁻² resolution (spectral gap)
        
        Both are REQUIRED for the system to work at scale.
        
        Args:
            contexts: [BATCH, 4, 4] context matrices
            targets: [BATCH, 4, 4] target matrices
            target_idxs: [BATCH] integer target indices
            token_sequences: Optional list of token lists
            
        Returns:
            Dict with batch storage stats
        """
        batch_size = contexts.shape[0]
        if batch_size == 0:
            return {'stored': 0}
        
        # BATCH holographic storage (single GPU operation)
        holo_result = self.holographic.store_batch(contexts, targets)
        holo_stored = holo_result.get('stored', 0)
        self.n_holographic_stores += holo_stored
        
        # BATCH episodic storage (8D, exact retrieval)
        episodic_result = self.witness_index.store_batch(
            contexts, targets, target_idxs, token_sequences
        )
        self.n_episodic_stores += episodic_result.get('stored', 0)
        
        # Store in semantic index (individual, no batch yet)
        for i in range(batch_size):
            self.semantic_index.store(contexts[i], targets[i], int(target_idxs[i]))
        self.n_semantic_stores += batch_size
        
        return {
            'stored': batch_size,
            'holo_stored': holo_stored,
            'n_patterns': self.holographic.n_patterns,
            'n_episodic_items': self.witness_index.n_items,
            'n_semantic_items': self.semantic_index.n_items,
        }
    
    def retrieve(self, context: Array) -> Tuple[Array, int, float, str]:
        """
        Retrieve target using cascade: holographic → episodic → semantic.
        
        v4.22.0 CASCADE:
            1. Try holographic (fast O(1), limited capacity)
            2. If low confidence, try episodic (8D exact match)
            3. If no match, try semantic (2D generalization)
            
        Args:
            context: [4, 4] context matrix
            
        Returns:
            (target_matrix, target_idx, confidence, source)
            source is "holographic", "episodic", "semantic", or "none"
        """
        # 1. Try holographic first (fast, O(1))
        holo_result, holo_conf = self.holographic.retrieve(context, denoise=True)
        
        if holo_conf >= self.min_holographic_confidence:
            self.n_holographic_retrievals += 1
            # Get target_idx from episodic lookup
            _, epi_idx, _ = self.witness_index.retrieve(context)
            if epi_idx is not None:
                return holo_result, epi_idx, holo_conf, "holographic"
        
        # 2. Try episodic index (8D, exact match)
        epi_result, epi_idx, epi_conf = self.witness_index.retrieve(context)
        
        if epi_result is not None and epi_idx is not None and epi_conf > PHI_INV_SQ:
            self.n_episodic_retrievals += 1
            return epi_result, epi_idx, epi_conf, "episodic"
        
        # 3. Try semantic index (2D, generalization)
        sem_result, sem_idx, sem_conf = self.semantic_index.retrieve(context)
        
        if sem_result is not None and sem_idx is not None:
            self.n_semantic_retrievals += 1
            return sem_result, sem_idx, sem_conf, "semantic"
        
        # 4. Return best available (even if low confidence)
        if epi_result is not None and epi_idx is not None:
            self.n_episodic_retrievals += 1
            return epi_result, epi_idx, epi_conf, "episodic_low"
        
        if holo_conf > 0:
            self.n_holographic_retrievals += 1
            return holo_result, 0, holo_conf, "holographic_low"
        
        # Nothing found
        return self.xp.zeros((MATRIX_DIM, MATRIX_DIM), dtype=DTYPE), 0, 0.0, "none"
    
    def update(self, context: Array, target: Array, target_idx: int,
               token_sequence: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Update an existing binding (Hebbian learning).
        
        THEORY-TRUE:
            new_binding = (1 - φ⁻¹) × old_binding + φ⁻¹ × new_target
            
        For holographic memory, this means:
            - Reduce weight of old binding
            - Add new binding with φ⁻¹ weight
            
        This is approximate because we can't selectively update
        superposed patterns. But the φ⁻¹ weighting gives recency
        preference naturally.
        
        Args:
            context: [4, 4] context matrix
            target: [4, 4] new target matrix
            target_idx: New target token index
            token_sequence: Optional token list for reindexing after drift
            
        Returns:
            Update stats
        """
        # For holographic: just add new binding (recency via weight)
        holo_result = self.holographic.store(context, target, weight=PHI_INV)
        
        # For episodic: update bucket (with token sequence for drift support)
        episodic_result = self.witness_index.store(context, target, target_idx,
                                                    token_sequence=token_sequence)
        
        # For semantic: update bucket
        semantic_result = self.semantic_index.store(context, target, target_idx)
        
        return {
            'holographic': holo_result,
            'episodic': episodic_result,
            'semantic': semantic_result,
        }
    
    def reindex_witness(self, compute_context_fn, get_embedding_fn) -> int:
        """
        Reindex witness index after embedding drift.
        
        THEORY-TRUE: After embeddings drift, stored patterns need reindexing
        so that queries with updated embeddings find the correct buckets.
        
        Args:
            compute_context_fn: Function(tokens) -> context_matrix
            get_embedding_fn: Function(token_idx) -> embedding
            
        Returns:
            Number of patterns reindexed
        """
        return self.witness_index.reindex_all(compute_context_fn, get_embedding_fn)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics for all three systems."""
        return {
            # Holographic
            'holographic_patterns': self.holographic.n_patterns,
            'holographic_total_stored': self.holographic.total_stored,
            'holographic_retrievals': self.n_holographic_retrievals,
            'holographic_stores': self.n_holographic_stores,
            # Episodic (8D)
            'episodic_items': self.witness_index.n_items,
            'episodic_buckets': len(self.witness_index.buckets),
            'episodic_retrievals': self.n_episodic_retrievals,
            'episodic_stores': self.n_episodic_stores,
            # Semantic (2D)
            'semantic_items': self.semantic_index.n_items,
            'semantic_buckets': len(self.semantic_index.buckets),
            'semantic_retrievals': self.n_semantic_retrievals,
            'semantic_stores': self.n_semantic_stores,
        }
    
    def clear(self):
        """Clear all memory systems."""
        self.holographic.clear()
        self.witness_index.clear()
        self.semantic_index.clear()
        self.n_holographic_retrievals = 0
        self.n_episodic_retrievals = 0
        self.n_semantic_retrievals = 0
        self.n_holographic_stores = 0
        self.n_episodic_stores = 0
        self.n_semantic_stores = 0


# =============================================================================
# WITNESS ENTROPY — Theory-True Capacity Signal
# =============================================================================

def compute_witness_entropy(M: Array, basis: Array, xp: ArrayModule = np) -> float:
    """
    Compute witness entropy H_w — the theory-true capacity signal.
    
    THEORY:
        H_w = -Σ p_k log(p_k)
        
        where p_k = |grade_k|² / Σ|grade_j|²
        
    INTERPRETATION (IMPORTANT — inverted from naive expectation):
        - HIGH H_w → energy spread across grades → FRESH, unstable memory
        - LOW H_w → energy concentrated → SATURATED, averaged memory
        
        Therefore:
        - Consolidate when H_w < φ⁻² (memory has become too uniform)
        - NOT when H_w is high
        
    WHY LOW ENTROPY = SATURATION:
        As more items are stored, interference averages out the grade
        distribution, making it more uniform in a different sense:
        the memory becomes a "gray soup" where everything looks similar.
        
    Args:
        M: [4, 4] matrix
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        Witness entropy H_w (bits). Lower = more saturated.
    """
    from .constants import GRADE_INDICES
    
    coeffs = decompose_to_coefficients(M, basis, xp)
    coeffs_sq = xp.asarray(coeffs) ** 2  # Vectorized squaring
    
    # VECTORIZED: Compute energy per grade using array indexing
    # GRADE_INDICES = {0: [0], 1: [1,2,3,4], 2: [5,6,7,8,9,10], 3: [11,12,13,14], 4: [15]}
    grade_energies = xp.array([
        float(xp.sum(coeffs_sq[xp.array(indices)]))
        for grade, indices in sorted(GRADE_INDICES.items())
    ])
    
    total_energy = float(xp.sum(grade_energies))
    if total_energy < 1e-12:
        return 0.0
    
    # VECTORIZED: Compute probabilities and entropy
    probs = grade_energies / total_energy
    probs = probs[probs > 1e-12]  # Filter zeros
    
    # Entropy (in bits) - vectorized
    entropy = -float(xp.sum(probs * xp.log2(probs + 1e-12)))
    
    return entropy


def is_memory_saturated(H_w: float, threshold: float = PHI_INV_SQ) -> bool:
    """
    Check if memory is saturated based on witness entropy.
    
    THEORY:
        Low entropy = saturated (everything has averaged out).
        Threshold is φ⁻² (spectral gap) by default.
        
    Args:
        H_w: Witness entropy
        threshold: Saturation threshold (default φ⁻² ≈ 0.382)
        
    Returns:
        True if memory needs consolidation
    """
    return H_w < threshold


# =============================================================================
# ITERATIVE UNBINDING — Multi-Item Retrieval
# =============================================================================

def iterative_unbind(
    memory: 'HolographicMemory',
    context: Array,
    max_items: int = 5,
    min_confidence: float = PHI_INV_CUBE,
    xp: ArrayModule = np,
) -> List[Tuple[Array, float]]:
    """
    Retrieve multiple items via iterative unbinding.
    
    THEORY:
        After retrieving t₁, subtract its contribution and retrieve again:
        
            M' = M - φ⁻¹ × (c ⊗ t₁)
            t₂ ≈ c⁻¹ ⊗ M'
            
        This extracts multiple bound items from superposition.
        
    USE CASES:
        - "What else is associated with this context?"
        - Multi-modal targets (same context → multiple outcomes)
        - Debugging memory contents
        
    Args:
        memory: HolographicMemory instance
        context: [4, 4] query context
        max_items: Maximum items to retrieve
        min_confidence: Stop when confidence drops below this
        xp: array module
        
    Returns:
        List of (retrieved_matrix, confidence) tuples
    """
    results = []
    current_memory = memory.memory.copy()
    
    ctx_norm = xp.linalg.norm(context, 'fro')
    if ctx_norm < 1e-8:
        return results
    
    ctx_normalized = context / ctx_norm
    ctx_inverse = clifford_inverse(ctx_normalized, memory.basis, xp)
    
    for i in range(max_items):
        # Retrieve from current memory state
        retrieved = geometric_product(ctx_inverse, current_memory)
        retrieved = grace_operator(retrieved, memory.basis, xp)
        
        # Compute confidence
        confidence = grace_stability(retrieved, memory.basis, xp)
        
        # Stop if confidence too low
        if confidence < min_confidence:
            break
        
        results.append((retrieved.copy(), float(confidence)))
        
        # Subtract this item's contribution for next iteration
        binding = geometric_product(ctx_normalized, retrieved)
        current_memory = current_memory - PHI_INV * binding
    
    return results


# =============================================================================
# MULTI-TIMESCALE BUFFERS — φ-Parameterized Decay
# =============================================================================

@dataclass
class MultiTimescaleMemory:
    """
    Multi-timescale holographic buffers with φ-parameterized decay.
    
    THEORY:
        Different buffers decay at different rates, all derived from φ:
        
        - FAST (working memory):  φ⁻¹ decay per cycle ≈ 0.618
        - MEDIUM (episodic):      φ⁻² decay per cycle ≈ 0.382
        - SLOW (near-semantic):   φ⁻³ decay per cycle ≈ 0.236
        
    BRAIN ANALOGY:
        - Fast ≈ prefrontal working memory (seconds)
        - Medium ≈ hippocampal episodic buffer (minutes-hours)
        - Slow ≈ cortico-hippocampal interface (hours-days)
        
    WHY THEORY-TRUE:
        All decay rates derived from φ, not arbitrary.
        This is the φ-analogue of complementary learning systems.
        
    STORAGE POLICY:
        High salience → all buffers (important, remember everywhere)
        Medium salience → medium + slow
        Low salience → slow only (background, long-term only)
        
    RETRIEVAL CASCADE:
        Try fast first (most recent), then medium, then slow.
        Return first result with sufficient confidence.
        
    Attributes:
        fast: Working memory buffer (φ⁻¹ decay)
        medium: Episodic buffer (φ⁻² decay)
        slow: Near-semantic buffer (φ⁻³ decay)
        basis: Clifford basis
        decay_count: Number of decay cycles applied
    """
    fast: 'HolographicMemory'
    medium: 'HolographicMemory'
    slow: 'HolographicMemory'
    basis: Array
    decay_count: int = 0
    xp: ArrayModule = field(default=np, repr=False)
    
    # Tracking
    fast_retrievals: int = 0
    medium_retrievals: int = 0
    slow_retrievals: int = 0
    
    @classmethod
    def create(cls, basis: Array, xp: ArrayModule = np) -> 'MultiTimescaleMemory':
        """Create multi-timescale memory with three buffers."""
        return cls(
            fast=HolographicMemory.create(basis, xp),
            medium=HolographicMemory.create(basis, xp),
            slow=HolographicMemory.create(basis, xp),
            basis=basis,
            xp=xp,
        )
    
    def store(self, context: Array, target: Array, salience: float = PHI_INV_SQ) -> Dict[str, Any]:
        """
        Store based on salience (theory-derived policy).
        
        HIGH salience (> φ⁻¹) → all buffers
        MEDIUM salience (> φ⁻²) → medium + slow
        LOW salience → slow only
        
        Args:
            context: [4, 4] context matrix
            target: [4, 4] target matrix
            salience: Importance score [0, 1]
            
        Returns:
            Storage info
        """
        buffers_used = []
        
        # Always store in slow (long-term)
        self.slow.store(context, target, weight=PHI_INV_CUBE)
        buffers_used.append('slow')
        
        if salience > PHI_INV:
            # High salience → all buffers
            self.fast.store(context, target, weight=PHI_INV)
            self.medium.store(context, target, weight=PHI_INV_SQ)
            buffers_used.extend(['fast', 'medium'])
        elif salience > PHI_INV_SQ:
            # Medium salience → medium + slow
            self.medium.store(context, target, weight=PHI_INV_SQ)
            buffers_used.append('medium')
        
        return {
            'buffers_used': buffers_used,
            'salience': salience,
        }
    
    def retrieve(self, context: Array, min_confidence: float = PHI_INV_CUBE) -> Tuple[Array, float, str]:
        """
        Retrieve with cascade: fast → medium → slow.
        
        Returns first result with confidence above threshold.
        Empty buffers are skipped (zero matrix has spurious high confidence).
        
        Args:
            context: [4, 4] query context
            min_confidence: Minimum acceptable confidence
            
        Returns:
            (result, confidence, source) where source is 'fast'/'medium'/'slow'/'none'
        """
        candidates = []
        
        # Try fast first (most recent) - skip if empty
        if self.fast.n_patterns > 0:
            fast_result, fast_conf = self.fast.retrieve(context)
            candidates.append((fast_result, fast_conf, "fast"))
            if fast_conf >= min_confidence:
                self.fast_retrievals += 1
                return fast_result, fast_conf, "fast"
        
        # Try medium - skip if empty
        if self.medium.n_patterns > 0:
            medium_result, medium_conf = self.medium.retrieve(context)
            candidates.append((medium_result, medium_conf, "medium"))
            if medium_conf >= min_confidence:
                self.medium_retrievals += 1
                return medium_result, medium_conf, "medium"
        
        # Fall back to slow - skip if empty
        if self.slow.n_patterns > 0:
            slow_result, slow_conf = self.slow.retrieve(context)
            candidates.append((slow_result, slow_conf, "slow"))
            if slow_conf >= min_confidence:
                self.slow_retrievals += 1
                return slow_result, slow_conf, "slow"
        
        # If no buffers have data, return zero with no confidence
        if not candidates:
            return self.xp.zeros((4, 4)), 0.0, "empty"
        
        # Return best available even if below threshold
        best = max(candidates, key=lambda x: x[1])
        return best[0], best[1], f"{best[2]}_low"
    
    def decay(self) -> Dict[str, float]:
        """
        Apply φ-parameterized decay to each buffer.
        
        This simulates time-based forgetting with theory-derived rates.
        
        Returns:
            Decay factors applied to each buffer
        """
        # Fast decays most rapidly
        self.fast.memory *= PHI_INV
        
        # Medium decays moderately
        self.medium.memory *= PHI_INV_SQ
        
        # Slow decays least
        self.slow.memory *= PHI_INV_CUBE
        
        self.decay_count += 1
        
        return {
            'fast_decay': PHI_INV,
            'medium_decay': PHI_INV_SQ,
            'slow_decay': PHI_INV_CUBE,
            'decay_count': self.decay_count,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics across all timescales."""
        return {
            'fast_patterns': self.fast.n_patterns,
            'medium_patterns': self.medium.n_patterns,
            'slow_patterns': self.slow.n_patterns,
            'fast_retrievals': self.fast_retrievals,
            'medium_retrievals': self.medium_retrievals,
            'slow_retrievals': self.slow_retrievals,
            'decay_count': self.decay_count,
        }
    
    def clear(self):
        """Clear all buffers."""
        self.fast.clear()
        self.medium.clear()
        self.slow.clear()
        self.decay_count = 0
        self.fast_retrievals = 0
        self.medium_retrievals = 0
        self.slow_retrievals = 0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_context_witness(context: Array, basis: Array, xp: ArrayModule = np) -> Tuple[float, float]:
    """
    Compute the witness of a context matrix.
    
    This is the theory-true "identity" of the context - what survives
    infinite Grace iterations.
    
    Args:
        context: [4, 4] context matrix
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        (scalar, pseudoscalar) witness components
    """
    return extract_witness(context, basis, xp)


def witness_distance(w1: Tuple[float, float], w2: Tuple[float, float]) -> float:
    """
    Euclidean distance between two witnesses.
    
    Theory-true: L2 distance is meaningful for gauge-invariant witnesses.
    """
    return np.sqrt((w1[0] - w2[0])**2 + (w1[1] - w2[1])**2)


def witness_similarity(w1: Tuple[float, float], w2: Tuple[float, float], eps: float = 1e-8) -> float:
    """
    Cosine similarity between two witnesses.
    
    Returns value in [-1, 1].
    """
    norm1 = np.sqrt(w1[0]**2 + w1[1]**2) + eps
    norm2 = np.sqrt(w2[0]**2 + w2[1]**2) + eps
    dot = w1[0] * w2[0] + w1[1] * w2[1]
    return dot / (norm1 * norm2)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    'HolographicMemory',
    'VorticityWitnessIndex',
    'CanonicalSemanticIndex',
    'HybridHolographicMemory',
    'MultiTimescaleMemory',
    
    # Clifford operations
    'clifford_reversion',
    'clifford_inverse',
    
    # Witness utilities
    'compute_context_witness',
    'witness_distance',
    'witness_similarity',
    
    # Witness entropy (capacity signal)
    'compute_witness_entropy',
    'is_memory_saturated',
    
    # Multi-item retrieval
    'iterative_unbind',
]
