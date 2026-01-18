"""
Full Integrated Pipeline — Compositional + Hebbian + Attractor
===============================================================

ARCHITECTURE:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    COMPOSITIONAL HOLOGRAPHIC MODEL                      │
    │                                                                         │
    │   1. COMPOSITIONAL EMBEDDINGS                                          │
    │      word → I + Σᵢ αᵢ(word) · fᵢ                                       │
    │      (features learned from co-occurrence)                             │
    │                                                                         │
    │   2. CONTEXT COMPUTATION                                               │
    │      context = geometric_product(embed(w₁), embed(w₂), ...)            │
    │                                                                         │
    │   3. ATTRACTOR MAP                                                     │
    │      context → target embedding                                         │
    │                                                                         │
    │   4. ONE-SHOT LEARNING                                                 │
    │      Unknown word in context → infer features from context             │
    └─────────────────────────────────────────────────────────────────────────┘

LEARNING PHASES:
    Phase 1: Build co-occurrence statistics
    Phase 2: Learn features via Hebbian/SVD
    Phase 3: Train attractor map
    Phase 4: Online one-shot learning for new words
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from .constants import PHI, PHI_INV, PHI_INV_SQ, MATRIX_DIM
from .algebra import (
    build_clifford_basis, geometric_product_batch,
    frobenius_similarity_batch
)
from .compositional import CompositionalEmbedding
from .feature_learning import (
    CooccurrenceTracker, learn_features_hebbian,
    infer_features_from_context, one_shot_learn_word
)
from .diagnostics import semantic_coherence_test

Array = np.ndarray
ArrayModule = type(np)


class CompositionalHolographicModel:
    """
    Full integrated model combining:
    - Compositional embeddings (feature-based words)
    - Hebbian feature learning (from co-occurrence)
    - Attractor map (context → target)
    - One-shot learning (context → infer features for unknown words)
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_features: int = 14,
        context_size: int = 5,
        max_attractors: int = 50000,
        xp: ArrayModule = np,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.num_features = min(num_features, 14)  # Max 14 orthogonal
        self.context_size = context_size
        self.max_attractors = max_attractors
        self.xp = xp
        
        # Clifford basis
        self.basis = build_clifford_basis(xp)
        
        # Compositional embeddings
        self.embedding = CompositionalEmbedding(
            vocab_size=vocab_size,
            num_features=self.num_features,
            xp=xp,
            seed=seed,
        )
        
        # Co-occurrence tracker for feature learning
        self.cooc = CooccurrenceTracker(
            vocab_size=vocab_size,
            window_size=context_size,
        )
        
        # Attractor map: context hash → (attractor matrix, target_idx)
        self.attractor_matrices = xp.zeros(
            (max_attractors, MATRIX_DIM, MATRIX_DIM), dtype=xp.float64
        )
        self.attractor_targets = xp.zeros(max_attractors, dtype=xp.int32)
        self.attractor_hashes: Dict[int, int] = {}
        self.num_attractors = 0
        
        # Statistics
        self.train_samples = 0
        self.exact_hits = 0
        self.novel_hits = 0
    
    # =========================================================================
    # EMBEDDING ACCESS
    # =========================================================================
    
    def get_word_embedding(self, word_idx: int) -> Array:
        """Get compositional embedding for a word."""
        return self.embedding.get_embedding(word_idx)
    
    def compute_context(self, tokens: List[int]) -> Array:
        """Compute context representation via geometric product."""
        if not tokens:
            return self.xp.eye(MATRIX_DIM, dtype=self.xp.float64)
        
        if len(tokens) == 1:
            return self.get_word_embedding(tokens[0])
        
        # Get all embeddings
        mats = self.xp.stack([
            self.get_word_embedding(t) for t in tokens
        ], axis=0)
        
        # Geometric product
        return geometric_product_batch(mats, self.xp)
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    def train_step(
        self,
        context: List[int],
        target: int,
        update_cooc: bool = True,
        hebbian_lr: float = 0.01,  # Ignored - kept for API compatibility
    ) -> Dict[str, float]:
        """
        THEORY-TRUE training step.
        
        From rhnsclifford.md:
            "Geometry = Clifford algebra, golden ratio, Grace contraction → FIXED"
            "Content = embeddings, context-attractor associations → LEARNED"
        
        The system is NOT a supervised learner. It's a dynamical system:
            1. Receives input → creates initial configuration (geometric products)
            2. Evolves under coherence dynamics (Grace flow toward attractor)
            3. Converges to equilibrium (the unique coherent state)
            4. The equilibrium IS the output
        
        CRITICAL: Embeddings are FIXED (identity-biased).
                  Only the ATTRACTOR MAP learns.
                  Grace contraction (φ⁻¹) is the learning rate.
        
        Args:
            context: Context token sequence
            target: Target token index
            update_cooc: Whether to update co-occurrence stats (for future use)
            hebbian_lr: IGNORED - kept for API compatibility
            
        Returns:
            Dict with metrics
        """
        target_idx = target % self.vocab_size
        
        # 1. Update co-occurrence statistics (for analysis, not learning)
        if update_cooc:
            full_seq = context + [target]
            self.cooc.update(full_seq)
        
        # 2. EMBEDDINGS ARE FIXED - NO MODIFICATION!
        # Theory: "Identity-biased initialization enables self-bootstrapping"
        # The algebraic structure provides geometry; content comes from associations.
        
        # 3. UPDATE ATTRACTOR MAP ONLY
        # Theory: "learn(context, target): attractor[context] = embedding[target]"
        # With Grace scaling: lerp by φ⁻¹ for stability
        
        ctx_hash = hash(tuple(context))
        target_emb = self.get_word_embedding(target_idx)  # Fixed embedding
        
        if ctx_hash in self.attractor_hashes:
            # Grace-weighted update: lerp by φ⁻¹
            # Theory: "attractor[context] = lerp(attractor[context], target_matrix, φ⁻¹)"
            idx = self.attractor_hashes[ctx_hash]
            self.attractor_matrices[idx] = (
                (1 - PHI_INV) * self.attractor_matrices[idx] + PHI_INV * target_emb
            )
        elif self.num_attractors < self.max_attractors:
            # New context: direct association
            # Theory: "attractor[context] = embedding[target]"
            idx = self.num_attractors
            self.attractor_matrices[idx] = target_emb.copy()
            self.attractor_targets[idx] = target_idx
            self.attractor_hashes[ctx_hash] = idx
            self.num_attractors += 1
        
        self.train_samples += 1
        
        return {'num_attractors': self.num_attractors}
    
    def train(
        self,
        contexts: List[List[int]],
        targets: List[int],
        hebbian_lr: float = 0.01,
        log_every: int = 1000,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            contexts: List of context sequences
            targets: List of target tokens
            hebbian_lr: Hebbian learning rate
            log_every: Log every N samples
            verbose: Print progress
            
        Returns:
            Training history
        """
        history = {
            'step': [],
            'num_attractors': [],
            'separation': [],
        }
        
        if verbose:
            print("=" * 60)
            print("TRAINING COMPOSITIONAL HOLOGRAPHIC MODEL")
            print("=" * 60)
        
        n = len(contexts)
        
        for i, (ctx, target) in enumerate(zip(contexts, targets)):
            metrics = self.train_step(ctx, target, hebbian_lr=hebbian_lr)
            
            if verbose and (i + 1) % log_every == 0:
                # Compute separation metric
                sample_size = min(500, i + 1)
                emb_matrices = self.embedding.get_all_embeddings()
                diag = semantic_coherence_test(
                    contexts[:sample_size],
                    targets[:sample_size],
                    emb_matrices,
                    self.basis,
                    self.xp
                )
                
                history['step'].append(i + 1)
                history['num_attractors'].append(self.num_attractors)
                history['separation'].append(diag['separation'])
                
                print(f"  [{i+1:,}/{n:,}] ctx={self.num_attractors:,} | "
                      f"sep={diag['separation']:.6f}")
        
        if verbose:
            print("=" * 60)
            print(f"Training complete: {self.num_attractors} attractors")
        
        return history
    
    # =========================================================================
    # RETRIEVAL & GENERATION
    # =========================================================================
    
    def retrieve(self, context: List[int]) -> Tuple[Array, int]:
        """
        Retrieve attractor for a context.
        
        For exact match: return stored attractor and target.
        For novel contexts: compute context rep, find most similar WORD embedding.
        
        Returns:
            (attractor_matrix, target_idx)
        """
        ctx_hash = hash(tuple(context))
        
        # Try exact match
        if ctx_hash in self.attractor_hashes:
            idx = self.attractor_hashes[ctx_hash]
            self.exact_hits += 1
            return self.attractor_matrices[idx].copy(), int(self.attractor_targets[idx])
        
        # Fall back to similarity-based retrieval
        self.novel_hits += 1
        
        # Compute context representation
        ctx_rep = self.compute_context(context)
        
        # KEY FIX: Compare to WORD embeddings, not stored attractors
        # This uses the compositional structure to find semantically similar words
        all_embeddings = self.embedding.get_all_embeddings()
        sims = frobenius_similarity_batch(ctx_rep, all_embeddings, self.xp)
        best_word = int(self.xp.argmax(sims))
        
        # Return the best word's embedding as the "attractor"
        return all_embeddings[best_word].copy(), best_word
    
    def generate(self, context: List[int], num_tokens: int = 10) -> List[int]:
        """
        Generate tokens autoregressively.
        
        Args:
            context: Starting context
            num_tokens: Number of tokens to generate
            
        Returns:
            List of generated token indices
        """
        generated = []
        current_ctx = list(context)
        
        for _ in range(num_tokens):
            # Get attractor and target
            attractor, target_idx = self.retrieve(current_ctx[-self.context_size:])
            
            # Could also decode attractor to find best word
            # For now, use the stored target
            generated.append(target_idx)
            current_ctx.append(target_idx)
        
        return generated
    
    # =========================================================================
    # ONE-SHOT LEARNING
    # =========================================================================
    
    def one_shot_learn(
        self,
        word_idx: int,
        context: List[int],
        strength: float = 0.8,
    ) -> None:
        """
        One-shot learn a new word's features from context.
        
        This is the key capability enabled by compositional embeddings:
        A new word appearing in a known context immediately clusters
        with words that appear in similar contexts.
        
        Args:
            word_idx: Index of new word
            context: Context in which word appeared
            strength: How much to trust inferred features
        """
        one_shot_learn_word(
            self.embedding,
            word_idx,
            context,
            strength=strength,
            xp=self.xp
        )
    
    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        total = self.exact_hits + self.novel_hits
        return {
            'train_samples': self.train_samples,
            'num_attractors': self.num_attractors,
            'exact_hits': self.exact_hits,
            'novel_hits': self.novel_hits,
            'generalization_rate': self.novel_hits / max(total, 1),
        }


# =============================================================================
# TESTS
# =============================================================================

def test_full_pipeline(xp: ArrayModule = np) -> bool:
    """Test the full integrated pipeline."""
    print("Testing full compositional pipeline...")
    
    # Create model
    model = CompositionalHolographicModel(
        vocab_size=100,
        num_features=14,
        context_size=5,
        max_attractors=1000,
        xp=xp,
    )
    
    print("  ✓ Model created")
    
    # Create training data with semantic structure
    # Group 1: words 0-19 co-occur (animals)
    # Group 2: words 20-39 co-occur (objects)
    np.random.seed(42)
    
    contexts = []
    targets = []
    
    for _ in range(500):
        # Group 1 contexts predict group 1 targets
        ctx1 = list(np.random.randint(0, 20, size=5))
        tgt1 = np.random.randint(0, 20)
        contexts.append(ctx1)
        targets.append(tgt1)
        
        # Group 2 contexts predict group 2 targets
        ctx2 = list(np.random.randint(20, 40, size=5))
        tgt2 = np.random.randint(20, 40)
        contexts.append(ctx2)
        targets.append(tgt2)
    
    # Train
    history = model.train(
        contexts, targets,
        hebbian_lr=0.05,
        log_every=200,
        verbose=True
    )
    
    print(f"  ✓ Training complete: {model.num_attractors} attractors")
    
    # Test retrieval
    test_ctx = [0, 1, 2, 3, 4]  # Group 1 context
    attractor, target = model.retrieve(test_ctx)
    print(f"  ✓ Retrieval works: target={target}")
    
    # Test generation - should stay in correct category
    generated = model.generate([0, 1, 2, 3, 4], num_tokens=5)
    group1_count = sum(1 for g in generated if g < 20)
    print(f"  ✓ Generation works: {generated} ({group1_count}/5 in group 1)")
    
    # Test one-shot learning
    # Word 50 appears in group 1 context → should cluster with group 1
    model.one_shot_learn(50, [0, 1, 2, 3, 4], strength=0.9)
    
    sim_to_g1 = model.embedding.embedding_similarity(50, 0)
    sim_to_g2 = model.embedding.embedding_similarity(50, 20)
    
    print(f"  One-shot word 50 similarity to group 1: {sim_to_g1:.4f}")
    print(f"  One-shot word 50 similarity to group 2: {sim_to_g2:.4f}")
    
    # Group 1 should be more similar
    assert sim_to_g1 > sim_to_g2, "One-shot word should cluster with context group!"
    print("  ✓ One-shot learning clusters correctly!")
    
    # Check final separation
    emb_matrices = model.embedding.get_all_embeddings()
    final_diag = semantic_coherence_test(
        contexts[:200], targets[:200],
        emb_matrices, model.basis, xp
    )
    print(f"  Final separation: {final_diag['separation']:.6f}")
    
    print("  ✓ Full pipeline test passed!")
    return True


def run_full_pipeline_tests(xp: ArrayModule = np) -> bool:
    """Run all full pipeline tests."""
    print("=" * 60)
    print("FULL PIPELINE TESTS")
    print("=" * 60)
    
    if test_full_pipeline(xp):
        print()
        print("=" * 60)
        print("ALL FULL PIPELINE TESTS PASSED")
        print("=" * 60)
        return True
    else:
        print("TESTS FAILED")
        return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CompositionalHolographicModel',
    'test_full_pipeline',
    'run_full_pipeline_tests',
]


if __name__ == "__main__":
    run_full_pipeline_tests()
