"""
Parallel Proposal Generation Test (DMN Core)

THEORY:
    DMN generates multiple counterfactual paths, not just one attractor.
    This enables better selection via coherence comparison.
    
    Current system: settle to ONE attractor
    DMN-like: settle to MULTIPLE attractors, select best

BRAIN ANALOGY:
    DMN runs structured simulation, counterfactuals, recombinations.
    "What if X instead of Y?" generates alternative paths.

PHYSICS FRAMING:
    Like exploring multiple minima in an energy landscape.
    Single settling may find local minimum; multiple exploration finds global.
"""

import modal
import numpy as np
import time
import json
from typing import List, Dict, Any

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

app = modal.App("parallel-proposals-test")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_parallel_proposals():
    """
    Test whether parallel proposal generation (DMN-like) improves retrieval.
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
        grace_basin_keys_batch_direct, grace_operator, frobenius_cosine
    )
    from holographic_prod.core.constants import PHI, PHI_INV
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast

    # Helper to handle both TowerMemory and MultiLevelTower
    def get_all_memories(tower):
        return getattr(tower, '_all_memories', None) or getattr(tower, '_all_memories', None)

    
    print("="*80)
    print("PARALLEL PROPOSAL GENERATION TEST (DMN CORE)")
    print("="*80)
    print("""
    HYPOTHESIS: Multiple proposals enable better basin selection.
    
    TEST:
        A. Single settle (standard)
        B. 3 proposals, select by coherence
        C. 5 proposals, select by coherence
        D. 5 proposals, weighted average by coherence
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
    # MODEL INITIALIZATION
    # =========================================================================
    print("\n  Initializing model...")
    
    model = HolographicMemory(
        vocab_size=len(word_to_idx),
        max_levels=4,
        seed=42,
        use_gpu=True,
        grounded_embeddings=cp.asarray(grounded_embs),
    )
    model.set_grounded_embeddings(cp.asarray(grounded_embs))
    
    print(f"  ✓ Model: {model.tower.n_satellites:,} satellites")
    
    # =========================================================================
    # PROPOSAL GENERATION FUNCTIONS
    # =========================================================================
    
    def generate_single_proposal(model, context):
        """Standard: single attractor settling."""
        xp = model.xp
        ctx_emb = model.tower._embed_sequence(context)
        
        # Standard Grace settling
        settled = ctx_emb
        for _ in range(GRACE_ROUTING_ITERS):
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
        
        return retrieved, sat_idx
    
    def generate_multiple_proposals(model, context, k=5):
        """DMN-like: generate k proposals with phi-scaled perturbations."""
        xp = model.xp
        ctx_emb = model.tower._embed_sequence(context)
        
        proposals = []
        sat_indices = []
        
        for i in range(k):
            # Apply phi-scaled perturbation (0 for first, increasing for others)
            if i == 0:
                perturbed = ctx_emb
            else:
                # Random perturbation scaled by phi^(-i)
                noise_scale = PHI_INV ** i
                noise = xp.random.randn(4, 4).astype(xp.float32) * noise_scale
                # Project to near-SO(4) via SVD
                combined = ctx_emb + noise
                u, _, vh = xp.linalg.svd(combined)
                perturbed = u @ vh
            
            # Settle
            settled = perturbed
            for _ in range(GRACE_ROUTING_ITERS):
                settled = grace_operator(settled, model.basis, xp)
            
            # Route
            basin_key = grace_basin_keys_batch_direct(
                settled[None], model.basis,
                n_iters=1, resolution=GRACE_ROUTING_RESOLUTION, xp=xp
            )[0]
            primes = xp.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53], dtype=xp.int64)
            sat_idx = int((xp.sum(basin_key * primes) % model.tower.n_satellites).get())
            
            sat_memory = get_all_memories(model.tower)[sat_idx]
            retrieved = settled.T @ sat_memory
            
            proposals.append(retrieved)
            sat_indices.append(sat_idx)
        
        return proposals, sat_indices
    
    def select_by_coherence(proposals, model):
        """Select proposal with highest self-coherence (Frobenius norm)."""
        xp = model.xp
        coherences = []
        for p in proposals:
            # Self-coherence = how "clean" is the signal
            # Higher norm = more energy = more confidence
            coh = float(xp.linalg.norm(p).get())
            coherences.append(coh)
        
        best_idx = np.argmax(coherences)
        return proposals[best_idx], coherences
    
    def weighted_average_by_coherence(proposals, model):
        """Average proposals weighted by coherence."""
        xp = model.xp
        coherences = []
        for p in proposals:
            coh = float(xp.linalg.norm(p).get())
            coherences.append(coh)
        
        # Normalize to weights
        total = sum(coherences)
        weights = [c / (total + 1e-10) for c in coherences]
        
        # Weighted average
        result = sum(w * p for w, p in zip(weights, proposals))
        return result, coherences
    
    def evaluate_method(model, method_fn, eval_samples, n_eval=100):
        """Evaluate a retrieval method."""
        xp = model.xp
        similarities = []
        
        for ctx, tgt in eval_samples[:n_eval]:
            try:
                retrieved = method_fn(model, ctx)
                tgt_emb = model.tower.embeddings[tgt % model.vocab_size]
                sim = float(frobenius_cosine(retrieved, tgt_emb, xp).get())
                similarities.append(sim)
            except Exception as e:
                raise RuntimeError(f"Similarity computation failed: {e}") from e
        
        if not similarities:
            raise RuntimeError("No similarities computed - all samples failed")
        return np.mean(similarities)
    
    # Wrapper methods for evaluation
    def method_single(model, ctx):
        r, _ = generate_single_proposal(model, ctx)
        return r
    
    def method_3_select(model, ctx):
        proposals, _ = generate_multiple_proposals(model, ctx, k=3)
        r, _ = select_by_coherence(proposals, model)
        return r
    
    def method_5_select(model, ctx):
        proposals, _ = generate_multiple_proposals(model, ctx, k=5)
        r, _ = select_by_coherence(proposals, model)
        return r
    
    def method_5_weighted(model, ctx):
        proposals, _ = generate_multiple_proposals(model, ctx, k=5)
        r, _ = weighted_average_by_coherence(proposals, model)
        return r
    
    # =========================================================================
    # TRAINING + EVALUATION
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING AND EVALUATION")
    print("="*80)
    
    BATCH_SIZE = 2048
    N_BATCHES = 40
    EVAL_EVERY = 5
    
    metrics = {
        'single': [],
        '3_select': [],
        '5_select': [],
        '5_weighted': [],
    }
    
    eval_samples = samples[N_BATCHES * BATCH_SIZE:]
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        batch = samples[start_idx:start_idx + BATCH_SIZE]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # Learn
        model.tower.learn_batch(contexts, targets)
        model.n_patterns += len(contexts)
        
        # Evaluate
        if (batch_idx + 1) % EVAL_EVERY == 0:
            sim_single = evaluate_method(model, method_single, eval_samples, n_eval=50)
            sim_3_select = evaluate_method(model, method_3_select, eval_samples, n_eval=50)
            sim_5_select = evaluate_method(model, method_5_select, eval_samples, n_eval=50)
            sim_5_weighted = evaluate_method(model, method_5_weighted, eval_samples, n_eval=50)
            
            metrics['single'].append(sim_single)
            metrics['3_select'].append(sim_3_select)
            metrics['5_select'].append(sim_5_select)
            metrics['5_weighted'].append(sim_5_weighted)
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Single:     {sim_single:.4f}")
            print(f"    3-select:   {sim_3_select:.4f} ({(sim_3_select/sim_single-1)*100:+.1f}%)")
            print(f"    5-select:   {sim_5_select:.4f} ({(sim_5_select/sim_single-1)*100:+.1f}%)")
            print(f"    5-weighted: {sim_5_weighted:.4f} ({(sim_5_weighted/sim_single-1)*100:+.1f}%)")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("PARALLEL PROPOSALS RESULTS")
    print("="*80)
    
    results = {}
    for method, sims in metrics.items():
        results[method] = {
            'avg': float(np.mean(sims)),
            'late_avg': float(np.mean(sims[2:])) if len(sims) > 2 else float(np.mean(sims)),
        }
    
    base = results['single']['late_avg']
    
    print(f"""
    LATE-STAGE AVERAGES:
      Single proposal:      {results['single']['late_avg']:.4f} (baseline)
      3-proposal select:    {results['3_select']['late_avg']:.4f} ({(results['3_select']['late_avg']/base-1)*100:+.2f}%)
      5-proposal select:    {results['5_select']['late_avg']:.4f} ({(results['5_select']['late_avg']/base-1)*100:+.2f}%)
      5-proposal weighted:  {results['5_weighted']['late_avg']:.4f} ({(results['5_weighted']['late_avg']/base-1)*100:+.2f}%)
      
    VERDICT:
      Multi-proposal helps: {results['5_select']['late_avg'] > base or results['5_weighted']['late_avg'] > base}
      Best method: {max(results.items(), key=lambda x: x[1]['late_avg'])[0]}
    """)
    
    # Save
    results_path = "/checkpoints/parallel_proposals_results.json"
    with open(results_path, 'w') as f:
        json.dump({'results': results, 'curves': metrics}, f, indent=2)
    
    return results


@app.local_entrypoint()
def main():
    print("Running Parallel Proposals Test on Modal H100...")
    result = test_parallel_proposals.remote()
    best = max(result.items(), key=lambda x: x[1]['late_avg'])
    print(f"\nBest method: {best[0]} ({best[1]['late_avg']:.4f})")
