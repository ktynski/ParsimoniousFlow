"""
Delayed Scalar Collapse Hypothesis Test

THEORY:
    Genius maintains high-dimensional phase flow (bivector/trivector) longer
    before collapsing to scalar commitment.
    
    HYPOTHESIS: Delaying scalar commitment improves learning because:
    - More interference patterns are explored
    - Better disambiguation before commit
    - Richer associative structure

TEST DESIGN:
    A: Standard learning (immediate read after settle)
    B: Delayed commit (additional Grace iterations before read)
    C: Multi-attractor exploration (settle multiple times, aggregate)

PHYSICS FRAMING:
    Like a spinning top: premature damping loses angular momentum (information).
    Allowing longer rotation before settling preserves more structure.
"""

import modal
import numpy as np
import time
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Modal setup
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "numpy>=1.24.0",
        "cupy-cuda12x>=12.0.0",
        "datasets>=2.14.0",
        "tqdm",
        "scipy",
    )
    .add_local_dir("holographic_prod", "/root/project/holographic_prod")
)

app = modal.App("delayed-collapse-test")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_delayed_collapse():
    """
    Test whether delayed scalar collapse improves learning.
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    import numpy as np
    from collections import defaultdict
    from tqdm import tqdm
    
    from holographic_prod.memory.holographic_memory_unified import (
        HolographicMemory, GRACE_ROUTING_ITERS, GRACE_ROUTING_RESOLUTION
    )
    from holographic_prod.core.algebra import (
        grace_basin_keys_batch_direct, grace_operator, frobenius_cosine,
        decompose_to_coefficients
    )
    from holographic_prod.core.constants import PHI, PHI_INV
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast

    # Helper to handle both TowerMemory and MultiLevelTower
    def get_all_memories(tower):
        return getattr(tower, '_all_memories', None) or getattr(tower, '_all_memories', None)

    
    print("="*80)
    print("DELAYED SCALAR COLLAPSE HYPOTHESIS TEST")
    print("="*80)
    print("""
    HYPOTHESIS: Delaying scalar collapse preserves useful rotational structure.
    
    TEST:
        A. Standard (3 Grace iterations before read)
        B. Delayed (6 Grace iterations before read)
        C. Multi-settle (3 iterations × 3 different perturbations, aggregate)
    """)
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    print("\n  Loading data...")
    
    from datasets import load_dataset
    
    vocab_path = "/checkpoints/vocabulary.npz"
    vocab_data = np.load(vocab_path, allow_pickle=True)
    word_to_idx = vocab_data['word_to_idx'].item()
    print(f"  ✓ Vocabulary: {len(word_to_idx):,} words")
    
    # Create grounded embeddings
    print("  Creating grounded embeddings...")
    grounded_embs, coverage = create_grounded_embeddings_fast(word_to_idx, glove_dim=50, cache_dir="/checkpoints/glove")
    print(f"  ✓ GloVe coverage: {coverage*100:.1f}%")
    
    # Prepare samples
    context_size = 64
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    samples = []
    
    for item in tqdm(ds.take(10_000), total=10_000, desc="Tokenizing"):
        words = item['text'].lower().split()
        tokens = [word_to_idx.get(w, 1) for w in words]
        for i in range(len(tokens) - context_size):
            ctx = tokens[i:i+context_size]
            tgt = tokens[i+context_size]
            if tgt != 1:
                samples.append((ctx, tgt))
            if len(samples) >= 100_000:
                break
        if len(samples) >= 100_000:
            break
    
    print(f"  ✓ Prepared {len(samples):,} samples")
    
    # =========================================================================
    # THREE MODEL VARIANTS
    # =========================================================================
    print("\n  Initializing three model variants...")
    
    def create_model():
        model = HolographicMemory(
            vocab_size=len(word_to_idx),
            max_levels=4,
            seed=42,
            use_gpu=True,
            grounded_embeddings=cp.asarray(grounded_embs),
        )
        model.set_grounded_embeddings(cp.asarray(grounded_embs))
        return model
    
    model_standard = create_model()
    model_delayed = create_model()
    model_multi = create_model()
    
    print(f"  ✓ Models initialized: {model_standard.tower.n_satellites:,} satellites each")
    
    # =========================================================================
    # CUSTOM RETRIEVAL FUNCTIONS WITH DIFFERENT COLLAPSE TIMING
    # =========================================================================
    
    def retrieve_standard(model, context, grace_iters=3):
        """Standard: 3 Grace iterations."""
        xp = model.xp
        ctx_emb = model.tower._embed_sequence(context)
        
        # Apply Grace iterations
        settled = ctx_emb
        for _ in range(grace_iters):
            settled = grace_operator(settled, model.basis, xp)
        
        # Route and unbind
        basin_key = grace_basin_keys_batch_direct(
            settled[None], model.basis,
            n_iters=1, resolution=GRACE_ROUTING_RESOLUTION, xp=xp
        )[0]
        primes = xp.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53], dtype=xp.int64)
        sat_idx = int((xp.sum(basin_key * primes) % model.tower.n_satellites).get())
        
        sat_memory = get_all_memories(model.tower)[sat_idx]
        retrieved = settled.T @ sat_memory
        
        return retrieved
    
    def retrieve_delayed(model, context, grace_iters=6):
        """Delayed: 6 Grace iterations (2x standard)."""
        return retrieve_standard(model, context, grace_iters=grace_iters)
    
    def retrieve_multi_settle(model, context, n_settles=3, grace_iters=3):
        """Multi-settle: Average multiple perturbations."""
        xp = model.xp
        ctx_emb = model.tower._embed_sequence(context)
        
        retrieved_all = []
        
        for i in range(n_settles):
            # Add phi-scaled perturbation
            if i > 0:
                # Random SO(4) perturbation scaled by phi^-i
                noise = xp.random.randn(4, 4).astype(xp.float32) * PHI_INV**(i+1)
                # Orthogonalize to stay near SO(4)
                u, _, vh = xp.linalg.svd(ctx_emb + noise)
                perturbed = u @ vh
            else:
                perturbed = ctx_emb
            
            # Settle
            settled = perturbed
            for _ in range(grace_iters):
                settled = grace_operator(settled, model.basis, xp)
            
            # Route and unbind
            basin_key = grace_basin_keys_batch_direct(
                settled[None], model.basis,
                n_iters=1, resolution=GRACE_ROUTING_RESOLUTION, xp=xp
            )[0]
            primes = xp.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53], dtype=xp.int64)
            sat_idx = int((xp.sum(basin_key * primes) % model.tower.n_satellites).get())
            
            sat_memory = get_all_memories(model.tower)[sat_idx]
            retrieved = settled.T @ sat_memory
            retrieved_all.append(retrieved)
        
        # Average the retrievals
        avg_retrieved = sum(retrieved_all) / len(retrieved_all)
        return avg_retrieved
    
    def evaluate_retrieval(model, retrieve_fn, eval_samples, n_eval=100):
        """Evaluate semantic similarity using custom retrieval function."""
        xp = model.xp
        similarities = []
        
        for ctx, tgt in eval_samples[:n_eval]:
            # NO TRY/EXCEPT - retrieval MUST NOT fail
            retrieved = retrieve_fn(model, ctx)
            tgt_emb = model.tower.embeddings[tgt % model.vocab_size]
            sim = float(frobenius_cosine(retrieved, tgt_emb, xp).get())
            similarities.append(sim)
        
        if not similarities:
            raise RuntimeError("No similarities computed - all samples failed")
        return np.mean(similarities)
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPARISON")
    print("="*80)
    
    BATCH_SIZE = 2048
    N_BATCHES = 40
    EVAL_EVERY = 5
    
    metrics = {
        'standard': {'sim': [], 'time': []},
        'delayed': {'sim': [], 'time': []},
        'multi': {'sim': [], 'time': []},
    }
    
    eval_samples = samples[N_BATCHES * BATCH_SIZE:]
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        batch = samples[start_idx:start_idx + BATCH_SIZE]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # All models learn the same data (standard learning)
        model_standard.tower.learn_batch(contexts, targets)
        model_standard.n_patterns += len(contexts)
        
        model_delayed.tower.learn_batch(contexts, targets)
        model_delayed.n_patterns += len(contexts)
        
        model_multi.tower.learn_batch(contexts, targets)
        model_multi.n_patterns += len(contexts)
        
        # Evaluation
        if (batch_idx + 1) % EVAL_EVERY == 0:
            # Standard retrieval
            t0 = time.time()
            sim_standard = evaluate_retrieval(model_standard, retrieve_standard, eval_samples)
            time_standard = time.time() - t0
            metrics['standard']['sim'].append(sim_standard)
            metrics['standard']['time'].append(time_standard)
            
            # Delayed retrieval
            t0 = time.time()
            sim_delayed = evaluate_retrieval(model_delayed, retrieve_delayed, eval_samples)
            time_delayed = time.time() - t0
            metrics['delayed']['sim'].append(sim_delayed)
            metrics['delayed']['time'].append(time_delayed)
            
            # Multi-settle retrieval
            t0 = time.time()
            sim_multi = evaluate_retrieval(model_multi, retrieve_multi_settle, eval_samples)
            time_multi = time.time() - t0
            metrics['multi']['sim'].append(sim_multi)
            metrics['multi']['time'].append(time_multi)
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Standard (3 Grace): sim={sim_standard:.4f}")
            print(f"    Delayed (6 Grace):  sim={sim_delayed:.4f} ({(sim_delayed/sim_standard-1)*100:+.1f}%)")
            print(f"    Multi-settle (3×3): sim={sim_multi:.4f} ({(sim_multi/sim_standard-1)*100:+.1f}%)")
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("DELAYED COLLAPSE TEST RESULTS")
    print("="*80)
    
    avg_standard = np.mean(metrics['standard']['sim'])
    avg_delayed = np.mean(metrics['delayed']['sim'])
    avg_multi = np.mean(metrics['multi']['sim'])
    
    # Late-stage comparison (after learning stabilizes)
    late_standard = np.mean(metrics['standard']['sim'][2:])
    late_delayed = np.mean(metrics['delayed']['sim'][2:])
    late_multi = np.mean(metrics['multi']['sim'][2:])
    
    results = {
        'overall': {
            'standard': avg_standard,
            'delayed': avg_delayed,
            'multi': avg_multi,
            'delayed_improvement': (avg_delayed / avg_standard - 1) * 100,
            'multi_improvement': (avg_multi / avg_standard - 1) * 100,
        },
        'late_stage': {
            'standard': late_standard,
            'delayed': late_delayed,
            'multi': late_multi,
            'delayed_improvement': (late_delayed / late_standard - 1) * 100,
            'multi_improvement': (late_multi / late_standard - 1) * 100,
        },
        'curves': metrics,
    }
    
    print(f"""
    OVERALL AVERAGES:
      Standard (3 Grace):  {avg_standard:.4f}
      Delayed (6 Grace):   {avg_delayed:.4f} ({results['overall']['delayed_improvement']:+.2f}%)
      Multi-settle (3×3):  {avg_multi:.4f} ({results['overall']['multi_improvement']:+.2f}%)
      
    LATE-STAGE (after warmup):
      Standard:  {late_standard:.4f}
      Delayed:   {late_delayed:.4f} ({results['late_stage']['delayed_improvement']:+.2f}%)
      Multi:     {late_multi:.4f} ({results['late_stage']['multi_improvement']:+.2f}%)
      
    VERDICT:
      Delayed collapse helps:   {late_delayed > late_standard}
      Multi-settle helps:       {late_multi > late_standard}
    """)
    
    # Save results
    results_path = "/checkpoints/delayed_collapse_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results to {results_path}")
    
    return results


@app.local_entrypoint()
def main():
    print("Running Delayed Collapse Test on Modal H100...")
    result = test_delayed_collapse.remote()
    print(f"\nVerdict: Delayed helps = {result['late_stage']['delayed_improvement'] > 0}")
