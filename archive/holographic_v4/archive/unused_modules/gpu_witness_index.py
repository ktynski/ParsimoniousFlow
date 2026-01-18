"""
GPU-Native Witness Index

THEORY-TRUE IMPLEMENTATION:
The witness index maps context matrices to (σ, p) witness coordinates.
This GPU-native implementation eliminates ALL CPU synchronization during training.

DESIGN PRINCIPLES:
1. Preallocated arrays: No dynamic growth, no fragmentation
2. Vectorized operations: Batch everything, no Python loops
3. GPU-native storage: CuPy arrays instead of Python dicts
4. Quantized lookup: Hash witnesses to buckets using array indexing

PERFORMANCE:
- store_batch: O(batch_size) GPU time, O(1) CPU time
- retrieve_batch: O(batch_size * n_items) GPU time, O(1) CPU time
  (Can be optimized to O(batch_size * bucket_size) with spatial hashing)

MEMORY:
- Preallocated for max_items
- witnesses: [max_items, 2] float32
- contexts: [max_items, 4, 4] float32
- targets: [max_items, 4, 4] float32
- target_idxs: [max_items] int32
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any

# Import constants
try:
    from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, MATRIX_DIM, DTYPE
except ImportError:
    PHI = (1 + np.sqrt(5)) / 2
    PHI_INV = 1 / PHI
    PHI_INV_SQ = PHI_INV ** 2
    MATRIX_DIM = 4
    DTYPE = np.float32

# Type alias for array module
ArrayModule = type(np)
Array = np.ndarray


@dataclass
class GPUWitnessIndex:
    """
    GPU-native witness index with zero CPU synchronization.
    
    ARCHITECTURE:
        - Preallocated GPU arrays for all storage
        - Vectorized witness extraction and quantization
        - Batch retrieval using broadcasting
        
    WITNESS SPACE:
        The witness is a 2D coordinate (σ, p) where:
        - σ: Real part of trace (spectral invariant)
        - p: Norm projection (geometric invariant)
        
        Witnesses are quantized to resolution = φ⁻² for bucketing.
        
    STORAGE LAYOUT:
        witnesses: [max_items, 2] - (σ, p) coordinates
        contexts: [max_items, 4, 4] - context matrices
        targets: [max_items, 4, 4] - target matrices
        target_idxs: [max_items] - integer target indices
        n_items: int - number of stored items
        
    LOOKUP:
        1. Extract query witness
        2. Broadcast distance to all stored witnesses
        3. Find matches within resolution
        4. Return φ-weighted average of matching targets
    """
    basis: Array
    witnesses: Array  # [max_items, 2]
    contexts: Array   # [max_items, 4, 4]
    targets: Array    # [max_items, 4, 4]
    target_idxs: Array  # [max_items]
    valid_mask: Array  # [max_items] - which entries are valid
    n_items: int
    max_items: int
    resolution: float
    xp: ArrayModule = field(default=np, repr=False)
    
    @classmethod
    def create(cls, basis: Array, max_items: int = 100000, 
               resolution: float = PHI_INV_SQ, xp: ArrayModule = np) -> 'GPUWitnessIndex':
        """
        Create a new GPU witness index with preallocated storage.
        
        Args:
            basis: [16, 4, 4] Clifford basis for witness extraction
            max_items: Maximum number of items to store
            resolution: Quantization resolution for witness space
            xp: Array module (numpy or cupy)
            
        Returns:
            Initialized GPUWitnessIndex
        """
        # Validate basis shape
        if basis.shape != (16, MATRIX_DIM, MATRIX_DIM):
            raise ValueError(f"Basis must be [16, 4, 4], got {basis.shape}")
        
        # Preallocate all arrays on device
        witnesses = xp.zeros((max_items, 2), dtype=DTYPE)
        contexts = xp.zeros((max_items, MATRIX_DIM, MATRIX_DIM), dtype=DTYPE)
        targets = xp.zeros((max_items, MATRIX_DIM, MATRIX_DIM), dtype=DTYPE)
        target_idxs = xp.zeros(max_items, dtype=xp.int32)
        valid_mask = xp.zeros(max_items, dtype=xp.bool_)
        
        # Ensure basis is on same device
        if hasattr(basis, 'get') and not hasattr(xp, 'cuda'):
            # basis is CuPy but xp is NumPy
            basis = basis.get()
        elif not hasattr(basis, 'get') and hasattr(xp, 'cuda'):
            # basis is NumPy but xp is CuPy
            basis = xp.array(basis)
        
        return cls(
            basis=basis,
            witnesses=witnesses,
            contexts=contexts,
            targets=targets,
            target_idxs=target_idxs,
            valid_mask=valid_mask,
            n_items=0,
            max_items=max_items,
            resolution=resolution,
            xp=xp,
        )
    
    def _extract_witnesses(self, Ms: Array) -> Array:
        """
        Extract witness coordinates from matrices.
        
        GPU-PARALLEL: Single kernel for all matrices.
        
        THEORY (matching quotient.extract_witness_batch):
            σ = <M, e₀> / <e₀, e₀> - scalar projection
            p = <M, e₁₂₃₄> / <e₁₂₃₄, e₁₂₃₄> - pseudoscalar projection
            
        where e₀ = basis[0] (scalar) and e₁₂₃₄ = basis[15] (pseudoscalar)
            
        Args:
            Ms: [batch, 4, 4] matrices
            
        Returns:
            [batch, 2] witness coordinates (scalar, pseudoscalar)
        """
        # basis is [16, 4, 4] - extract scalar (index 0) and pseudoscalar (index 15)
        scalar_basis = self.basis[0]      # [4, 4]
        pseudo_basis = self.basis[15]     # [4, 4]
        
        # Precompute normalization constants (scalar)
        scalar_norm = self.xp.sum(scalar_basis * scalar_basis)
        pseudo_norm = self.xp.sum(pseudo_basis * pseudo_basis)
        
        # Batched projection: [BATCH, 4, 4] · [4, 4] → [BATCH]
        # Use einsum for efficient batched inner product
        scalars = self.xp.einsum('bij,ij->b', Ms, scalar_basis) / scalar_norm
        pseudos = self.xp.einsum('bij,ij->b', Ms, pseudo_basis) / pseudo_norm
        
        # Stack to [BATCH, 2]
        return self.xp.stack([scalars, pseudos], axis=1)
    
    def _quantize_witnesses(self, witnesses: Array) -> Array:
        """
        Quantize witnesses to bucket indices.
        
        Args:
            witnesses: [batch, 2] witness coordinates
            
        Returns:
            [batch, 2] quantized coordinates (int32)
        """
        return self.xp.floor(witnesses / self.resolution).astype(self.xp.int32)
    
    def store(self, context: Array, target: Array, target_idx: int,
              token_sequence=None) -> Dict[str, Any]:
        """
        Single-item storage (for compatibility).
        
        Note: For batch operations, use store_batch.
        
        Args:
            context: [4, 4] context matrix
            target: [4, 4] target matrix
            target_idx: Integer target token
            token_sequence: Optional (ignored for GPU index)
            
        Returns:
            Storage stats
        """
        # Expand to batch
        contexts = context[None, :, :]  # [1, 4, 4]
        targets = target[None, :, :]    # [1, 4, 4]
        target_idxs = self.xp.array([target_idx], dtype=self.xp.int32)
        
        return self.store_batch(contexts, targets, target_idxs)
    
    def store_batch(self, contexts: Array, targets: Array, 
                    target_idxs: Array,
                    token_sequences=None) -> Dict[str, Any]:
        """
        Store batch of context-target pairs.
        
        GPU-ONLY: No CPU synchronization.
        
        SUBSAMPLING: Store every ~3rd sample (φ² ≈ 2.618)
        This is theory-true: nearby samples map to same bucket.
        
        Args:
            contexts: [batch, 4, 4] context matrices
            targets: [batch, 4, 4] target matrices
            target_idxs: [batch] integer target indices
            token_sequences: Optional, ignored (for API compatibility)
            
        Returns:
            Storage statistics
        """
        # Note: token_sequences ignored - GPU index doesn't support reindexing
        # (would require GPU→CPU sync which defeats the purpose)
        batch_size = contexts.shape[0]
        if batch_size == 0:
            return {'stored': 0, 'n_items': self.n_items}
        
        # φ²-SUBSAMPLING: Store every ~3rd sample
        subsample_step = 3  # int(PHI * PHI) ≈ 2.618 → 3
        
        # Create subsampling indices on GPU
        subsample_indices = self.xp.arange(0, batch_size, subsample_step)
        n_to_store = len(subsample_indices)
        
        if n_to_store == 0:
            return {'stored': 0, 'n_items': self.n_items}
        
        # Check capacity
        if self.n_items + n_to_store > self.max_items:
            # Truncate to fit
            n_to_store = self.max_items - self.n_items
            if n_to_store <= 0:
                return {'stored': 0, 'n_items': self.n_items, 'full': True}
            subsample_indices = subsample_indices[:n_to_store]
        
        # Extract subsampled data - ALL GPU operations
        sub_contexts = contexts[subsample_indices]  # [n, 4, 4]
        sub_targets = targets[subsample_indices]    # [n, 4, 4]
        sub_idxs = target_idxs[subsample_indices]   # [n]
        
        # Extract witnesses - GPU vectorized
        sub_witnesses = self._extract_witnesses(sub_contexts)  # [n, 2]
        
        # Store in preallocated arrays - GPU slice assignment
        start_idx = self.n_items
        end_idx = start_idx + n_to_store
        
        self.witnesses[start_idx:end_idx] = sub_witnesses
        self.contexts[start_idx:end_idx] = sub_contexts
        self.targets[start_idx:end_idx] = sub_targets
        self.target_idxs[start_idx:end_idx] = sub_idxs
        self.valid_mask[start_idx:end_idx] = True
        
        self.n_items = end_idx
        
        return {
            'stored': n_to_store,
            'n_items': self.n_items,
            'full': self.n_items >= self.max_items,
        }
    
    def retrieve_batch(self, query_contexts: Array) -> Tuple[Array, Array, Array]:
        """
        Batch retrieval using vectorized witness matching.
        
        GPU-ONLY: No CPU synchronization.
        
        ALGORITHM:
            1. Extract query witnesses: [batch, 2]
            2. Broadcast distance to all stored: [batch, n_items]
            3. Find matches within resolution
            4. φ-weighted average of matching targets
            
        Args:
            query_contexts: [batch, 4, 4] query matrices
            
        Returns:
            (targets, target_idxs, confidences)
            - targets: [batch, 4, 4] retrieved targets
            - target_idxs: [batch] integer indices
            - confidences: [batch] confidence scores
        """
        batch_size = query_contexts.shape[0]
        
        # If empty, return zeros (allocate only on this path)
        if self.n_items == 0:
            return (
                self.xp.zeros((batch_size, MATRIX_DIM, MATRIX_DIM), dtype=DTYPE),
                self.xp.zeros(batch_size, dtype=self.xp.int32),
                self.xp.zeros(batch_size, dtype=DTYPE)
            )
        
        # Extract query witnesses: [batch, 2]
        query_witnesses = self._extract_witnesses(query_contexts)
        
        # Get valid stored witnesses: [n_items, 2]
        stored_witnesses = self.witnesses[:self.n_items]
        stored_targets = self.targets[:self.n_items]
        stored_idxs = self.target_idxs[:self.n_items]
        
        # Broadcast distance computation: [batch, n_items]
        # query: [batch, 1, 2], stored: [1, n_items, 2]
        diff = query_witnesses[:, None, :] - stored_witnesses[None, :, :]  # [batch, n_items, 2]
        distances = self.xp.sqrt(self.xp.sum(diff ** 2, axis=2))  # [batch, n_items]
        
        # Find matches within resolution
        matches = distances < self.resolution  # [batch, n_items] bool
        
        # THEORY-TRUE: φ-power weights (NOT exp)
        # φ⁻ᵈⁱˢᵗ gives smooth falloff with theory-derived decay rate
        weights = self.xp.where(
            matches,
            PHI_INV ** self.xp.clip(distances, 0, 10),  # φ⁻ᵈⁱˢᵗ
            self.xp.zeros_like(distances)
        )  # [batch, n_items]
        
        # Normalize weights per query
        weight_sums = self.xp.sum(weights, axis=1, keepdims=True)  # [batch, 1]
        weight_sums = self.xp.maximum(weight_sums, 1e-8)  # Avoid div by zero
        normalized_weights = weights / weight_sums  # [batch, n_items]
        
        # Weighted average of targets: [batch, 4, 4]
        # targets: [n_items, 4, 4], weights: [batch, n_items]
        result_targets = self.xp.einsum('bn,nij->bij', normalized_weights, stored_targets)
        
        # Best matching index per query (highest weight)
        best_match_indices = self.xp.argmax(weights, axis=1)  # [batch]
        result_idxs = stored_idxs[best_match_indices]  # [batch]
        
        # Confidence: max weight / sum (how concentrated the match is)
        max_weights = self.xp.max(weights, axis=1)  # [batch]
        result_confidences = max_weights / self.xp.squeeze(weight_sums)
        
        # Zero confidence for unmatched queries
        any_match = self.xp.any(matches, axis=1)  # [batch]
        result_confidences = self.xp.where(any_match, result_confidences, self.xp.zeros_like(result_confidences))
        
        return result_targets, result_idxs, result_confidences
    
    def retrieve(self, context: Array) -> Tuple[Optional[Array], Optional[int], float]:
        """
        Single-item retrieval (for compatibility).
        
        Note: For batch operations, use retrieve_batch.
        
        Args:
            context: [4, 4] query matrix
            
        Returns:
            (target, target_idx, confidence) or (None, None, 0.0)
        """
        # Expand to batch
        context_batch = context[None, :, :]  # [1, 4, 4]
        
        targets, idxs, confidences = self.retrieve_batch(context_batch)
        
        # Extract single result
        if self.xp == np:
            confidence = float(confidences[0])
            target_idx = int(idxs[0])
        else:
            confidence = float(confidences[0].get())
            target_idx = int(idxs[0].get())
        
        if confidence < 1e-8:
            return None, None, 0.0
        
        return targets[0], target_idx, confidence
    
    def clear(self):
        """Clear all stored items."""
        self.valid_mask[:] = False
        self.n_items = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'n_items': self.n_items,
            'max_items': self.max_items,
            'utilization': self.n_items / self.max_items,
            'memory_mb': (
                self.witnesses.nbytes + 
                self.contexts.nbytes + 
                self.targets.nbytes + 
                self.target_idxs.nbytes
            ) / (1024 * 1024),
        }


# =============================================================================
# INTEGRATION: Replace WitnessIndex in training loop
# =============================================================================

def create_gpu_witness_index(basis: Array, xp, max_items: int = 100000) -> GPUWitnessIndex:
    """
    Factory function for creating GPU witness index.
    
    Args:
        basis: [4, 4] basis matrix
        xp: Array module
        max_items: Maximum storage capacity
        
    Returns:
        GPUWitnessIndex instance
    """
    return GPUWitnessIndex.create(basis, max_items=max_items, xp=xp)
