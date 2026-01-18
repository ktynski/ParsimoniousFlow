"""
Optimal Training Pipeline for Holographic Language Model (SUPERSEDED)
=====================================================================

NOTE: This module contains findings from early experiments with atomic embeddings.
      The key breakthrough (v3.0) was compositional embeddings + Hebbian learning.
      
      USE INSTEAD: full_pipeline.py with CompositionalHolographicModel
      
      Atomic embeddings achieved:  separation ≈ 0.017 (best case)
      Compositional embeddings:    separation ≈ 0.72 (12x improvement!)

---

HISTORICAL FINDINGS (atomic embeddings):

1. Identity-biased init with LOW noise (0.01-0.05):
   - Witness stable ✓
   - But representations collapse → no semantic separation
   
2. Identity-biased init with MODERATE noise (0.1-0.2):
   - Witness moderately stable
   - Better separation (0.007-0.017)
   - Still positive correlation with target
   
3. Random init:
   - No semantic structure
   - Negative separation (worse than chance)
   
4. Contrastive learning alone:
   - Doesn't help much when starting collapsed
   - Needs initial diversity to work

OLD RECIPE (atomic, superseded):
   1. Initialize with identity + noise_std=0.15 (moderate)
   2. Train attractor map (fast, establishes co-occurrence)
   3. Apply contrastive refinement (slower, improves semantics)
   4. Monitor separation metric as key success indicator
   
LEVEL 2 INSIGHT:
   Level 2 alone doesn't help without Level 1 semantics.
   → Fix Level 1 first, then Level 2 amplifies the structure.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

from .constants import PHI, PHI_INV, PHI_INV_SQ, MATRIX_DIM
from .algebra import (
    build_clifford_basis, geometric_product_batch,
    frobenius_similarity_batch, initialize_all_embeddings,
    normalize_matrix
)
from .diagnostics import semantic_coherence_test, witness_stability_analysis
from .contrastive import find_contrastive_pairs, contrastive_update

Array = np.ndarray
ArrayModule = type(np)


@dataclass
class TrainingConfig:
    """Configuration for optimal training."""
    vocab_size: int
    context_size: int = 5
    
    # Initialization
    init_mode: str = 'identity'
    init_noise: float = 0.15  # Sweet spot for separation
    
    # Attractor training
    attractor_lr: float = 0.382  # φ⁻¹
    max_attractors: int = 50000
    
    # Contrastive refinement
    contrastive_epochs: int = 5
    contrastive_lr: float = 0.05
    contrastive_batch: int = 200
    
    # Diagnostics
    log_every: int = 1000


class OptimalTrainer:
    """
    Optimal training pipeline combining:
    1. Identity-biased init with moderate noise
    2. Fast attractor map training
    3. Contrastive semantic refinement
    """
    
    def __init__(self, config: TrainingConfig, xp: ArrayModule = np):
        self.config = config
        self.xp = xp
        
        # Build Clifford structure
        self.basis = build_clifford_basis(xp)
        
        # Initialize embeddings with optimal noise
        self.embeddings = initialize_all_embeddings(
            config.vocab_size, self.basis,
            mode=config.init_mode,
            noise_std=config.init_noise,
            xp=xp
        )
        
        # Attractor storage
        self.attractor_matrices = xp.zeros(
            (config.max_attractors, MATRIX_DIM, MATRIX_DIM), dtype=xp.float64
        )
        self.attractor_contexts = []
        self.num_attractors = 0
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'step': [],
            'num_attractors': [],
            'separation': [],
            'same_target_sim': [],
            'diff_target_sim': [],
        }
    
    def embed_context(self, tokens: List[int]) -> Array:
        """Compute context matrix via geometric product."""
        if not tokens:
            return self.xp.eye(MATRIX_DIM, dtype=self.xp.float64)
        if len(tokens) == 1:
            return self.embeddings[tokens[0] % len(self.embeddings)]
        token_arr = self.xp.array(tokens, dtype=self.xp.int32)
        mats = self.embeddings[token_arr % len(self.embeddings)]
        return geometric_product_batch(mats, self.xp)
    
    def train_attractor_map(
        self,
        contexts: List[List[int]],
        targets: List[int],
        verbose: bool = True,
    ) -> None:
        """
        Phase 1: Train the attractor map (fast co-occurrence learning).
        """
        if verbose:
            print("=" * 60)
            print("PHASE 1: Attractor Map Training")
            print("=" * 60)
        
        n = len(contexts)
        lr = self.config.attractor_lr
        
        for i, (ctx, target) in enumerate(zip(contexts, targets)):
            target_emb = self.embeddings[target % len(self.embeddings)]
            
            # Check if context exists (update via EMA)
            ctx_hash = hash(tuple(ctx))
            found = False
            for j in range(self.num_attractors):
                if hash(tuple(self.attractor_contexts[j])) == ctx_hash:
                    self.attractor_matrices[j] = (
                        (1 - lr) * self.attractor_matrices[j] + lr * target_emb
                    )
                    found = True
                    break
            
            if not found and self.num_attractors < self.config.max_attractors:
                idx = self.num_attractors
                self.attractor_matrices[idx] = target_emb.copy()
                self.attractor_contexts.append(list(ctx))
                self.num_attractors += 1
            
            # Log progress
            if verbose and (i + 1) % self.config.log_every == 0:
                diag = semantic_coherence_test(
                    contexts[:min(i+1, 500)],
                    targets[:min(i+1, 500)],
                    self.embeddings, self.basis, self.xp
                )
                self.history['step'].append(i + 1)
                self.history['num_attractors'].append(self.num_attractors)
                self.history['separation'].append(diag['separation'])
                self.history['same_target_sim'].append(diag['same_target_sim'])
                self.history['diff_target_sim'].append(diag['diff_target_sim'])
                
                print(f"  [{i+1:,}/{n:,}] ctx={self.num_attractors:,} | "
                      f"sep={diag['separation']:.6f} | "
                      f"same={diag['same_target_sim']:.4f}")
        
        if verbose:
            print(f"  Complete: {self.num_attractors} attractors")
    
    def refine_with_contrastive(
        self,
        contexts: List[List[int]],
        targets: List[int],
        verbose: bool = True,
    ) -> None:
        """
        Phase 2: Contrastive refinement of embeddings.
        """
        if verbose:
            print("=" * 60)
            print("PHASE 2: Contrastive Refinement")
            print("=" * 60)
        
        for epoch in range(self.config.contrastive_epochs):
            # Find contrastive pairs
            same_pairs, diff_pairs = find_contrastive_pairs(
                targets,
                max_same_pairs=self.config.contrastive_batch,
                max_diff_pairs=self.config.contrastive_batch,
            )
            
            # Update embeddings
            self.embeddings, metrics = contrastive_update(
                self.embeddings, contexts, same_pairs, diff_pairs,
                learning_rate=self.config.contrastive_lr,
                xp=self.xp
            )
            
            # Measure separation
            diag = semantic_coherence_test(
                contexts[:min(len(contexts), 500)],
                targets[:min(len(targets), 500)],
                self.embeddings, self.basis, self.xp
            )
            
            if verbose:
                print(f"  Epoch {epoch+1}/{self.config.contrastive_epochs}: "
                      f"sep={diag['separation']:.6f} | "
                      f"same_loss={metrics['same_loss']:.4f}")
    
    def train(
        self,
        contexts: List[List[int]],
        targets: List[int],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training pipeline.
        """
        # Phase 1: Attractor map
        self.train_attractor_map(contexts, targets, verbose)
        
        # Phase 2: Contrastive refinement
        self.refine_with_contrastive(contexts, targets, verbose)
        
        # Final metrics
        final_diag = semantic_coherence_test(
            contexts[:min(len(contexts), 500)],
            targets[:min(len(targets), 500)],
            self.embeddings, self.basis, self.xp
        )
        
        if verbose:
            print("=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            print(f"  Final separation: {final_diag['separation']:.6f}")
            print(f"  Attractors: {self.num_attractors}")
        
        return {
            'final_separation': final_diag['separation'],
            'num_attractors': self.num_attractors,
            'history': self.history,
        }
    
    def generate(self, context: List[int], num_tokens: int = 10) -> List[int]:
        """Generate tokens using the trained model."""
        generated = []
        current_ctx = list(context)
        
        for _ in range(num_tokens):
            # Get context representation
            ctx_rep = self.embed_context(current_ctx[-self.config.context_size:])
            
            # Find closest attractor
            if self.num_attractors > 0:
                attractors = self.attractor_matrices[:self.num_attractors]
                sims = frobenius_similarity_batch(ctx_rep, attractors, self.xp)
                best_idx = int(self.xp.argmax(sims))
                attractor = self.attractor_matrices[best_idx]
                
                # Find word closest to attractor
                word_sims = frobenius_similarity_batch(attractor, self.embeddings, self.xp)
                next_token = int(self.xp.argmax(word_sims))
            else:
                next_token = 0
            
            generated.append(next_token)
            current_ctx.append(next_token)
        
        return generated


def test_optimal_training(xp: ArrayModule = np) -> bool:
    """Test the optimal training pipeline."""
    print("Testing optimal training pipeline...")
    
    # Create data with clear structure
    np.random.seed(42)
    vocab_size = 100
    n_samples = 2000
    
    contexts = []
    targets = []
    for i in range(n_samples):
        ctx = list(np.random.randint(0, vocab_size, size=5))
        # Clear pattern: target depends on first two words
        target = (ctx[0] + ctx[1]) % vocab_size
        contexts.append(ctx)
        targets.append(target)
    
    # Configure
    config = TrainingConfig(
        vocab_size=vocab_size,
        init_noise=0.15,
        attractor_lr=0.382,
        contrastive_epochs=3,
        contrastive_lr=0.03,
        contrastive_batch=100,
        log_every=500,
    )
    
    # Train
    trainer = OptimalTrainer(config, xp)
    results = trainer.train(contexts, targets, verbose=True)
    
    # Verify improvement
    print(f"\nFinal separation: {results['final_separation']:.6f}")
    
    # Test generation
    test_ctx = [10, 20, 30, 40, 50]
    generated = trainer.generate(test_ctx, num_tokens=5)
    print(f"Generated from {test_ctx}: {generated}")
    
    print("\n✓ Optimal training test passed!")
    return True


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TrainingConfig',
    'OptimalTrainer',
    'test_optimal_training',
]


if __name__ == "__main__":
    test_optimal_training()
