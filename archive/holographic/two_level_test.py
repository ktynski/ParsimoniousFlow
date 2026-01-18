"""
Two-Level Training Experiment
=============================

TEST: Does Level 2 improve semantic coherence?

HYPOTHESIS:
    Level 1 learns word co-occurrence (statistical, not semantic)
    Level 2 should learn phrase-level patterns, clustering similar word attractors
    
DESIGN:
    1. Train Level 1 on word prediction (same as before)
    2. Build codebook: top-k word attractors become Level 2 tokens
    3. Train Level 2 on phrase prediction (attractor sequences)
    4. Measure: Do phrase contexts cluster by meaning?
    
SUCCESS CRITERIA:
    - Level 2 separation > Level 1 separation
    - Generation at Level 2 shows coherent phrases
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from .constants import PHI, PHI_INV, MATRIX_DIM
from .algebra import (
    build_clifford_basis, geometric_product_batch,
    frobenius_similarity_batch, initialize_all_embeddings
)
from .hierarchy import HierarchyLevel, HierarchicalModel
from .diagnostics import semantic_coherence_test, run_level1_diagnostics

Array = np.ndarray
ArrayModule = type(np)


def create_phrase_dataset(
    texts: List[str],
    vocab: Dict[str, int],
    phrase_size: int = 3,
    max_phrases: int = 10000,
) -> Tuple[List[List[int]], List[int]]:
    """
    Create phrase-level training data.
    
    A "phrase" is a sequence of word-level attractors.
    Target: predict the next phrase's attractor index.
    
    Args:
        texts: Raw text strings
        vocab: Word to index mapping
        phrase_size: Words per phrase
        max_phrases: Maximum phrases to extract
        
    Returns:
        (phrase_contexts, phrase_targets)
        where phrase_contexts[i] is [attractor_idx1, attractor_idx2, ...]
    """
    contexts = []
    targets = []
    
    for text in texts:
        words = text.lower().split()
        word_ids = [vocab.get(w, 0) for w in words]
        
        # Group into phrases
        phrases = []
        for i in range(0, len(word_ids) - phrase_size + 1, phrase_size):
            phrase = tuple(word_ids[i:i + phrase_size])
            phrases.append(phrase)
        
        # Create context-target pairs at phrase level
        # Context: [phrase1, phrase2, phrase3], target: phrase4
        context_len = 3
        for i in range(len(phrases) - context_len):
            ctx = list(phrases[i:i + context_len])
            target = phrases[i + context_len]
            contexts.append(ctx)
            targets.append(target)
            
            if len(contexts) >= max_phrases:
                break
        
        if len(contexts) >= max_phrases:
            break
    
    return contexts, targets


def train_two_level(
    contexts_l1: List[List[int]],
    targets_l1: List[int],
    vocab_size: int,
    codebook_size: int = 500,
    max_l1_samples: int = 5000,
    max_l2_samples: int = 2000,
    xp: ArrayModule = np,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train a two-level hierarchical model.
    
    Args:
        contexts_l1: Word-level contexts
        targets_l1: Word-level targets
        vocab_size: Vocabulary size
        codebook_size: Number of L1 attractors to use as L2 tokens
        max_l1_samples: Max Level 1 training samples
        max_l2_samples: Max Level 2 training samples
        xp: array module
        verbose: Print progress
        
    Returns:
        Dict with training results and metrics
    """
    results = {}
    
    if verbose:
        print("=" * 60)
        print("TWO-LEVEL TRAINING EXPERIMENT")
        print("=" * 60)
    
    # Create model
    model = HierarchicalModel(
        vocab_size=vocab_size,
        num_levels=2,
        codebook_size=codebook_size,
        max_attractors_per_level=10000,
        xp=xp
    )
    
    if verbose:
        print(f"\nCreated 2-level model (vocab={vocab_size}, codebook={codebook_size})")
    
    # =========================================================================
    # LEVEL 1: Train word prediction
    # =========================================================================
    if verbose:
        print("\n--- LEVEL 1: Word Prediction ---")
    
    n_l1 = min(len(contexts_l1), max_l1_samples)
    for i in range(n_l1):
        ctx = contexts_l1[i]
        target_idx = targets_l1[i]
        target_emb = model.levels[0].embed_token(target_idx)
        model.levels[0].associate(ctx, target_emb)
        
        if verbose and (i + 1) % 1000 == 0:
            stats = model.levels[0].get_statistics()
            print(f"  [{i+1}/{n_l1}] attractors={stats['num_attractors']}")
    
    l1_stats = model.levels[0].get_statistics()
    results['level1'] = l1_stats
    
    if verbose:
        print(f"  Level 1 complete: {l1_stats['num_attractors']} attractors")
    
    # =========================================================================
    # LEVEL 1 DIAGNOSTICS
    # =========================================================================
    if verbose:
        print("\n--- LEVEL 1 DIAGNOSTICS ---")
    
    l1_diag = semantic_coherence_test(
        contexts_l1[:min(len(contexts_l1), 500)],
        targets_l1[:min(len(targets_l1), 500)],
        model.levels[0].embeddings,
        model.levels[0].basis,
        xp
    )
    results['level1_diagnostics'] = l1_diag
    
    if verbose:
        print(f"  Same-target sim: {l1_diag['same_target_sim']:.4f}")
        print(f"  Diff-target sim: {l1_diag['diff_target_sim']:.4f}")
        print(f"  Separation: {l1_diag['separation']:.4f}")
    
    # =========================================================================
    # BUILD CODEBOOK: L1 attractors → L2 tokens
    # =========================================================================
    if verbose:
        print("\n--- BUILDING CODEBOOK ---")
    
    model.update_codebook(1)
    
    if model.codebooks[0] is not None:
        cb_size = model.codebooks[0].matrices.shape[0]
        if verbose:
            print(f"  Codebook size: {cb_size}")
    else:
        if verbose:
            print("  Warning: Codebook is empty")
        return results
    
    # =========================================================================
    # LEVEL 2: Train phrase prediction
    # =========================================================================
    if verbose:
        print("\n--- LEVEL 2: Phrase Prediction ---")
    
    # Create phrase-level training data
    # Map L1 context patterns to codebook indices
    phrase_contexts = []
    phrase_targets = []
    
    # Group L1 contexts by their attractor (find which codebook entry they match)
    for i in range(min(n_l1, max_l2_samples * 4)):
        ctx = contexts_l1[i]
        target = targets_l1[i]
        
        # Get L1 context representation
        ctx_rep = model.levels[0].embed_sequence(ctx)
        
        # Find closest codebook entry
        cb_mats = model.codebooks[0].matrices
        sims = frobenius_similarity_batch(ctx_rep, cb_mats, xp)
        ctx_cb_idx = int(xp.argmax(sims))
        
        # Get target's closest codebook entry
        target_emb = model.levels[0].embed_token(target)
        target_sims = frobenius_similarity_batch(target_emb, cb_mats, xp)
        target_cb_idx = int(xp.argmax(target_sims))
        
        phrase_contexts.append([ctx_cb_idx])  # Single "phrase" context
        phrase_targets.append(target_cb_idx)
    
    # Train Level 2
    n_l2 = min(len(phrase_contexts), max_l2_samples)
    for i in range(n_l2):
        ctx = phrase_contexts[i]
        target_idx = phrase_targets[i]
        target_emb = model.levels[1].embed_token(target_idx)
        model.levels[1].associate(ctx, target_emb)
        
        if verbose and (i + 1) % 500 == 0:
            stats = model.levels[1].get_statistics()
            print(f"  [{i+1}/{n_l2}] attractors={stats['num_attractors']}")
    
    l2_stats = model.levels[1].get_statistics()
    results['level2'] = l2_stats
    
    if verbose:
        print(f"  Level 2 complete: {l2_stats['num_attractors']} attractors")
    
    # =========================================================================
    # LEVEL 2 DIAGNOSTICS
    # =========================================================================
    if verbose:
        print("\n--- LEVEL 2 DIAGNOSTICS ---")
    
    l2_diag = semantic_coherence_test(
        phrase_contexts[:min(len(phrase_contexts), 500)],
        phrase_targets[:min(len(phrase_targets), 500)],
        model.levels[1].embeddings,
        model.levels[1].basis,
        xp
    )
    results['level2_diagnostics'] = l2_diag
    
    if verbose:
        print(f"  Same-target sim: {l2_diag['same_target_sim']:.4f}")
        print(f"  Diff-target sim: {l2_diag['diff_target_sim']:.4f}")
        print(f"  Separation: {l2_diag['separation']:.4f}")
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("COMPARISON: Level 1 vs Level 2")
        print("=" * 60)
        print(f"  Level 1 separation: {l1_diag['separation']:.6f}")
        print(f"  Level 2 separation: {l2_diag['separation']:.6f}")
        
        if l2_diag['separation'] > l1_diag['separation']:
            print("  ✓ Level 2 has HIGHER separation (hierarchy helps)")
        else:
            print("  ✗ Level 2 has lower separation (need different approach)")
        
        print("=" * 60)
    
    return results


def test_two_level(xp: ArrayModule = np) -> bool:
    """Test the two-level training pipeline."""
    print("Testing two-level training...")
    
    # Create synthetic data
    np.random.seed(42)
    vocab_size = 100
    n_samples = 500
    
    contexts = []
    targets = []
    for i in range(n_samples):
        ctx = list(np.random.randint(0, vocab_size, size=5))
        # Pattern: target depends on first word modulo
        target = (ctx[0] * 2 + ctx[1]) % vocab_size
        contexts.append(ctx)
        targets.append(target)
    
    # Run two-level training
    results = train_two_level(
        contexts, targets,
        vocab_size=vocab_size,
        codebook_size=50,
        max_l1_samples=400,
        max_l2_samples=200,
        xp=xp,
        verbose=True
    )
    
    assert 'level1' in results
    assert 'level2' in results
    assert 'level1_diagnostics' in results
    assert 'level2_diagnostics' in results
    
    print("\n✓ Two-level test passed!")
    return True


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'create_phrase_dataset',
    'train_two_level',
    'test_two_level',
]


if __name__ == "__main__":
    test_two_level()
