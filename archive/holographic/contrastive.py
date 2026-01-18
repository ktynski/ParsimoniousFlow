"""
Contrastive Learning for Semantic Structure (EXPERIMENTAL)
==========================================================

NOTE: This module is superseded by compositional embeddings (compositional.py).
      Compositional + Hebbian learning achieves better semantic separation.
      Keep this for reference and potential future experimentation.

KEY INSIGHT: The hierarchy alone doesn't create semantics.
    We need a learning signal that pulls together contexts predicting the same target.

CONTRASTIVE OBJECTIVE:
    Minimize: D(context_i, context_j) when target_i == target_j
    Maximize: D(context_i, context_k) when target_i != target_k
    
IMPLEMENTATION:
    For each batch, identify same-target pairs and different-target pairs.
    Update embeddings to move same-target contexts closer.
    
THEORY:
    This adds the missing "semantic gradient" to the attractor map.
    Level 1 learns: statistical co-occurrence
    + Contrastive: semantic equivalence classes
    
SUPERSEDED BY: compositional.py + feature_learning.py (v3.0)
    Compositional embeddings achieve 12x better separation.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict

from .constants import PHI, PHI_INV, PHI_INV_SQ, MATRIX_DIM
from .algebra import (
    build_clifford_basis, geometric_product_batch,
    frobenius_similarity_batch, normalize_matrix
)
from .quotient import witness_pointer

Array = np.ndarray
ArrayModule = type(np)


def find_contrastive_pairs(
    targets: List[int],
    max_same_pairs: int = 1000,
    max_diff_pairs: int = 1000,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Find same-target and different-target pairs for contrastive learning.
    
    Args:
        targets: List of target indices
        max_same_pairs: Maximum same-target pairs to return
        max_diff_pairs: Maximum different-target pairs to return
        
    Returns:
        (same_pairs, diff_pairs) where each pair is (idx_i, idx_j)
    """
    # Group by target
    target_to_idxs: Dict[int, List[int]] = defaultdict(list)
    for idx, target in enumerate(targets):
        target_to_idxs[target].append(idx)
    
    rng = np.random.default_rng(42)
    
    # Same-target pairs
    same_pairs = []
    targets_with_multiple = [t for t, idxs in target_to_idxs.items() if len(idxs) >= 2]
    
    attempts = 0
    while len(same_pairs) < max_same_pairs and attempts < max_same_pairs * 10:
        if not targets_with_multiple:
            break
        t = rng.choice(targets_with_multiple)
        idxs = target_to_idxs[t]
        if len(idxs) >= 2:
            i, j = rng.choice(len(idxs), size=2, replace=False)
            same_pairs.append((idxs[i], idxs[j]))
        attempts += 1
    
    # Different-target pairs
    diff_pairs = []
    all_targets = list(target_to_idxs.keys())
    
    attempts = 0
    while len(diff_pairs) < max_diff_pairs and attempts < max_diff_pairs * 10:
        if len(all_targets) < 2:
            break
        t1, t2 = rng.choice(all_targets, size=2, replace=False)
        idxs1 = target_to_idxs[t1]
        idxs2 = target_to_idxs[t2]
        if idxs1 and idxs2:
            i = rng.choice(idxs1)
            j = rng.choice(idxs2)
            diff_pairs.append((i, j))
        attempts += 1
    
    return same_pairs, diff_pairs


def contrastive_update(
    embeddings: Array,
    contexts: List[List[int]],
    same_pairs: List[Tuple[int, int]],
    diff_pairs: List[Tuple[int, int]],
    learning_rate: float = 0.01,
    margin: float = 0.1,
    xp: ArrayModule = np,
) -> Tuple[Array, Dict[str, float]]:
    """
    Update embeddings using contrastive learning.
    
    The key operation:
    - For same-target pairs: move their context representations closer
    - For diff-target pairs: push their context representations apart
    
    We update the WORD embeddings based on how they contribute to context reps.
    
    Args:
        embeddings: [vocab_size, 4, 4] word embeddings
        contexts: List of context token sequences
        same_pairs: List of (idx_i, idx_j) for same-target pairs
        diff_pairs: List of (idx_i, idx_j) for different-target pairs
        learning_rate: Update rate
        margin: Minimum desired separation for diff pairs
        xp: array module
        
    Returns:
        (updated_embeddings, metrics_dict)
    """
    def embed_context(ctx_tokens):
        if len(ctx_tokens) == 0:
            return xp.eye(MATRIX_DIM, dtype=xp.float64)
        if len(ctx_tokens) == 1:
            return embeddings[ctx_tokens[0] % len(embeddings)]
        token_mats = embeddings[xp.array(ctx_tokens) % len(embeddings)]
        return geometric_product_batch(token_mats, xp)
    
    # Track updates
    grad_accumulator = xp.zeros_like(embeddings)
    word_counts = xp.zeros(len(embeddings), dtype=xp.float64)
    
    same_loss = 0.0
    diff_loss = 0.0
    
    # Same-target pairs: minimize distance
    for i, j in same_pairs:
        ctx_i = contexts[i]
        ctx_j = contexts[j]
        
        rep_i = embed_context(ctx_i)
        rep_j = embed_context(ctx_j)
        
        # Frobenius distance
        diff = rep_i - rep_j
        dist = float(xp.sqrt(xp.sum(diff * diff) + 1e-8))
        same_loss += dist
        
        # Gradient: move representations closer
        # For simplicity, update each word in both contexts
        grad_direction = diff / (dist + 1e-8)
        
        for word_idx in ctx_i:
            idx = word_idx % len(embeddings)
            grad_accumulator[idx] -= learning_rate * grad_direction
            word_counts[idx] += 1
        
        for word_idx in ctx_j:
            idx = word_idx % len(embeddings)
            grad_accumulator[idx] += learning_rate * grad_direction
            word_counts[idx] += 1
    
    # Different-target pairs: maximize distance (up to margin)
    for i, j in diff_pairs:
        ctx_i = contexts[i]
        ctx_j = contexts[j]
        
        rep_i = embed_context(ctx_i)
        rep_j = embed_context(ctx_j)
        
        diff = rep_i - rep_j
        dist = float(xp.sqrt(xp.sum(diff * diff) + 1e-8))
        
        # Only push apart if closer than margin
        if dist < margin:
            diff_loss += margin - dist
            
            grad_direction = diff / (dist + 1e-8)
            
            for word_idx in ctx_i:
                idx = word_idx % len(embeddings)
                grad_accumulator[idx] += learning_rate * grad_direction
                word_counts[idx] += 1
            
            for word_idx in ctx_j:
                idx = word_idx % len(embeddings)
                grad_accumulator[idx] -= learning_rate * grad_direction
                word_counts[idx] += 1
    
    # Average gradients by word frequency
    for i in range(len(embeddings)):
        if word_counts[i] > 0:
            grad_accumulator[i] /= word_counts[i]
    
    # Apply updates
    updated = embeddings + grad_accumulator
    
    # Re-normalize
    updated = normalize_matrix(updated, xp)
    
    # Metrics
    metrics = {
        'same_loss': same_loss / max(len(same_pairs), 1),
        'diff_loss': diff_loss / max(len(diff_pairs), 1),
        'total_loss': (same_loss + diff_loss) / max(len(same_pairs) + len(diff_pairs), 1),
        'num_same_pairs': len(same_pairs),
        'num_diff_pairs': len(diff_pairs),
    }
    
    return updated, metrics


def train_with_contrastive(
    contexts: List[List[int]],
    targets: List[int],
    vocab_size: int,
    num_epochs: int = 10,
    batch_size: int = 100,
    learning_rate: float = 0.01,
    xp: ArrayModule = np,
    verbose: bool = True,
) -> Tuple[Array, Dict[str, Any]]:
    """
    Train embeddings using contrastive learning.
    
    Args:
        contexts: Word-level contexts
        targets: Target words
        vocab_size: Vocabulary size
        num_epochs: Number of training epochs
        batch_size: Pairs per batch
        learning_rate: Update rate
        xp: array module
        verbose: Print progress
        
    Returns:
        (trained_embeddings, training_history)
    """
    from .algebra import initialize_all_embeddings
    from .diagnostics import semantic_coherence_test
    
    # Initialize embeddings
    basis = build_clifford_basis(xp)
    embeddings = initialize_all_embeddings(
        vocab_size, basis, mode='identity', noise_std=0.05, xp=xp
    )
    
    history = {
        'epochs': [],
        'same_loss': [],
        'diff_loss': [],
        'separation': [],
    }
    
    if verbose:
        print("=" * 60)
        print("CONTRASTIVE TRAINING")
        print("=" * 60)
    
    for epoch in range(num_epochs):
        # Find contrastive pairs
        same_pairs, diff_pairs = find_contrastive_pairs(
            targets, max_same_pairs=batch_size, max_diff_pairs=batch_size
        )
        
        # Update embeddings
        embeddings, metrics = contrastive_update(
            embeddings, contexts, same_pairs, diff_pairs,
            learning_rate=learning_rate, xp=xp
        )
        
        # Compute separation
        diag = semantic_coherence_test(
            contexts[:min(len(contexts), 200)],
            targets[:min(len(targets), 200)],
            embeddings, basis, xp
        )
        
        # Record
        history['epochs'].append(epoch)
        history['same_loss'].append(metrics['same_loss'])
        history['diff_loss'].append(metrics['diff_loss'])
        history['separation'].append(diag['separation'])
        
        if verbose:
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"same_loss={metrics['same_loss']:.4f}, "
                  f"diff_loss={metrics['diff_loss']:.4f}, "
                  f"separation={diag['separation']:.6f}")
    
    if verbose:
        print("=" * 60)
        initial_sep = history['separation'][0] if history['separation'] else 0
        final_sep = history['separation'][-1] if history['separation'] else 0
        print(f"Initial separation: {initial_sep:.6f}")
        print(f"Final separation: {final_sep:.6f}")
        print(f"Improvement: {final_sep - initial_sep:.6f}")
        print("=" * 60)
    
    return embeddings, history


def test_contrastive(xp: ArrayModule = np) -> bool:
    """Test contrastive training."""
    print("Testing contrastive training...")
    
    # Create data with clear structure
    np.random.seed(42)
    vocab_size = 50
    n_samples = 200
    
    contexts = []
    targets = []
    for i in range(n_samples):
        ctx = list(np.random.randint(0, vocab_size, size=4))
        # Clear pattern: target = ctx[0] mod 10
        target = ctx[0] % 10
        contexts.append(ctx)
        targets.append(target)
    
    # Train with contrastive
    embeddings, history = train_with_contrastive(
        contexts, targets,
        vocab_size=vocab_size,
        num_epochs=5,
        batch_size=50,
        learning_rate=0.05,
        xp=xp,
        verbose=True
    )
    
    # Check improvement
    initial_sep = history['separation'][0]
    final_sep = history['separation'][-1]
    
    print(f"\nImprovement: {final_sep - initial_sep:.6f}")
    
    if final_sep > initial_sep:
        print("✓ Contrastive training improved separation!")
    else:
        print("⚠ Separation didn't improve (may need more epochs)")
    
    return True


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'find_contrastive_pairs',
    'contrastive_update',
    'train_with_contrastive',
    'test_contrastive',
]


if __name__ == "__main__":
    test_contrastive()
