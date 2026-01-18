"""
Working Memory — Salience-Based Attention and Short-Term Cache

Implements theory-true working memory:
- Salience-based gating (grace_stability × salience, not softmax)
- φ-decay for recency weighting
- Capacity limits matching Miller's law (~7±2 items)

THEORY (Working Memory Gating):
    Working memory preserves what SURVIVES.
    
    The theory-native attention weight is GRACE-STABILITY:
    - High stability (σ ≈ 1) → token survives Grace → high weight
    - Low stability (σ ≈ 0) → token decays under Grace → low weight
    
    This replaces arbitrary softmax with theory-derived measure.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any

from holographic_prod.core.constants import (
    PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, DTYPE
)
from holographic_prod.core.algebra import (
    frobenius_cosine,
    grace_operator,
    geometric_product,
)
from holographic_prod.core.quotient import grace_stability_batch
from .priority import compute_salience_batch


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
    # GPU-NATIVE: Use xp.stack consistently
    token_matrices = xp.stack([xp.asarray(embeddings[idx % len(embeddings)]) for idx in token_indices])
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
            basis: Clifford basis (required for salience computation)
            xp: array module
        """
        self.capacity = capacity
        self.basis = basis
        self.xp = xp
        
        # Storage: list of (matrix, target, salience, timestamp)
        self.items: List[Tuple[np.ndarray, int, float, int]] = []
        self.timestamp = 0
    
    def add(self, matrix: np.ndarray, target: int) -> Optional[Tuple[np.ndarray, int]]:
        """
        Add an item to working memory.
        
        If at capacity, evicts the lowest-salience item.
        
        Args:
            matrix: [4, 4] context matrix
            target: Associated target token
            
        Returns:
            Evicted item (matrix, target) if at capacity, else None
        """
        from .priority import compute_salience
        
        # Compute salience of new item
        if self.basis is not None:
            salience = compute_salience(matrix, self.basis, self.xp)
        else:
            salience = 1.0  # Default if no basis
        
        self.timestamp += 1
        
        evicted = None
        
        if len(self.items) >= self.capacity:
            # Find lowest-salience item
            min_idx = 0
            min_salience = self.items[0][2]
            
            for i, (_, _, s, _) in enumerate(self.items):
                if s < min_salience:
                    min_salience = s
                    min_idx = i
            
            # Evict if new item has higher salience
            if salience > min_salience:
                evicted = (self.items[min_idx][0], self.items[min_idx][1])
                self.items.pop(min_idx)
            else:
                # New item has lower salience - don't add it
                return None
        
        self.items.append((matrix, target, salience, self.timestamp))
        return evicted
    
    def get_context(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get combined context from all items in working memory.
        
        Returns:
            (context_matrix, info_dict)
        """
        if not self.items:
            return self.xp.eye(4, dtype=DTYPE), {'num_items': 0}
        
        # GPU-NATIVE: Use xp.stack consistently
        matrices = self.xp.stack([self.xp.asarray(item[0]) for item in self.items])
        
        if self.basis is not None:
            return gated_context_representation(matrices, self.basis, self.xp)
        else:
            # Without basis, just multiply
            context = matrices[0].copy()
            for i in range(1, len(matrices)):
                context = geometric_product(context, matrices[i])
            return context, {'num_items': len(self.items)}
    
    def clear(self):
        """Clear all items."""
        self.items = []
        self.timestamp = 0
    
    def __len__(self) -> int:
        return len(self.items)


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
            # THEORY-TRUE: Use cosine similarity (normalized to [-1, 1])
            sim = frobenius_cosine(query_matrix, ctx, np)
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
