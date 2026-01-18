"""
Coherence-Based vs Novelty-Based Reward Test

THEORY:
    Genius is rewarded by compression/insight, not novelty alone.
    
    - Novelty reward: "This is NEW!" → spike on prediction error
    - Coherence reward: "This FITS!" → spike on coherence increase
    
    The profitable signal (from trading analogy) is NOT novelty,
    but deviation-from-expected followed by RESTORATION.

BRAIN ANALOGY:
    VTA dopamine responds to:
    - Insight (compression achieved)
    - Explanatory power (pattern fits many cases)
    - NOT just surprise/novelty

PHYSICS FRAMING:
    Like rewarding decrease in system entropy, not increase in fluctuations.
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

app = modal.App("coherence-reward-test")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_coherence_reward():
    """
    Test whether coherence-based reward shaping outperforms novelty-based.
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
    
    print("="*80)
    print("COHERENCE-BASED VS NOVELTY-BASED REWARD TEST")
    print("="*80)
    print("""
    HYPOTHESIS: Coherence reward (compression) beats novelty reward (surprise).
    
    TEST:
        A. Standard (uniform learning weight)
        B. Novelty-weighted (high prediction error → high weight)
        C. Coherence-weighted (high coherence gain → high weight)
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
    # THREE MODEL VARIANTS
    # =========================================================================
    print("\n  Initializing three models...")
    
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
    model_novelty = create_model()
    model_coherence = create_model()
    
    print(f"  ✓ Models: {model_standard.tower.n_satellites:,} satellites each")
    
    # =========================================================================
    # REWARD COMPUTATION FUNCTIONS
    # =========================================================================
    
    def compute_prediction_error(model, contexts, targets):
        """
        Compute prediction error for each sample (novelty signal).
        High error = surprising = novel.
        """
        xp = model.xp
        errors = []
        
        for ctx, tgt in zip(contexts, targets):
            try:
                ctx_emb = model.tower._embed_sequence(ctx)
                
                # Route
                basin_key = grace_basin_keys_batch_direct(
                    ctx_emb[None], model.basis,
                    n_iters=GRACE_ROUTING_ITERS,
                    resolution=GRACE_ROUTING_RESOLUTION,
                    xp=xp
                )[0]
                primes = xp.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53], dtype=xp.int64)
                sat_idx = int((xp.sum(basin_key * primes) % model.tower.n_satellites).get())
                
                sat_memory = model.tower._all_memories[sat_idx]
                retrieved = ctx_emb.T @ sat_memory
                
                tgt_emb = model.tower.embeddings[tgt % model.vocab_size]
                sim = float(frobenius_cosine(retrieved, tgt_emb, xp).get())
                
                # Error = 1 - similarity
                error = 1.0 - sim
                errors.append(error)
            except Exception as e:
                raise RuntimeError(f"Error computation failed: {e}") from e
        
        return np.array(errors, dtype=np.float32)
    
    def compute_coherence_gain(model, contexts, targets):
        """
        Compute coherence gain for each sample.
        
        Coherence gain = how much does learning this sample INCREASE
        the coherence of nearby patterns?
        
        Approximation: Samples that are similar to existing patterns
        but add new information have high coherence gain.
        """
        xp = model.xp
        gains = []
        
        for ctx, tgt in zip(contexts, targets):
            try:
                ctx_emb = model.tower._embed_sequence(ctx)
                
                # Route
                basin_key = grace_basin_keys_batch_direct(
                    ctx_emb[None], model.basis,
                    n_iters=GRACE_ROUTING_ITERS,
                    resolution=GRACE_ROUTING_RESOLUTION,
                    xp=xp
                )[0]
                primes = xp.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53], dtype=xp.int64)
                sat_idx = int((xp.sum(basin_key * primes) % model.tower.n_satellites).get())
                
                sat_memory = model.tower._all_memories[sat_idx]
                sat_norm = float(xp.linalg.norm(sat_memory).get())
                
                # Coherence gain heuristic:
                # - If satellite is sparse (low norm), high gain potential
                # - If satellite has some content, moderate gain
                # - If satellite is saturated (high norm), low gain
                
                # Binding we're about to add
                tgt_emb = model.tower.embeddings[tgt % model.vocab_size]
                binding = ctx_emb @ tgt_emb
                binding_norm = float(xp.linalg.norm(binding).get())
                
                # Coherence with existing content
                if sat_norm > 0.01:
                    coherence = float(frobenius_cosine(binding, sat_memory, xp).get())
                else:
                    coherence = 0.5  # Neutral for empty satellite
                
                # Gain = coherence × relative novelty
                # High coherence + moderate novelty = high gain (compression)
                # Low coherence = low gain (doesn't fit)
                # Perfect coherence = low gain (redundant)
                
                novelty = 1.0 - abs(coherence)
                gain = (0.5 + coherence) * (0.3 + novelty)  # Peaks at moderate coherence
                
                gains.append(gain)
            except Exception as e:
                raise RuntimeError(f"Gain computation failed: {e}") from e
        
        return np.array(gains, dtype=np.float32)
    
    def learn_batch_weighted(model, contexts, targets, weights):
        """Learn with per-sample weights."""
        xp = model.xp
        tower = model.tower
        
        if not contexts:
            return
        
        batch_size = len(contexts)
        
        # Embed contexts
        ctx_matrices = tower._embed_sequences_batch(contexts)
        
        # Route
        basin_keys = grace_basin_keys_batch_direct(
            ctx_matrices, model.basis,
            n_iters=GRACE_ROUTING_ITERS,
            resolution=GRACE_ROUTING_RESOLUTION,
            xp=xp
        )
        primes = xp.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53], dtype=xp.int64)
        satellite_indices = (xp.sum(basin_keys * primes, axis=1) % tower.n_satellites).astype(xp.int32)
        
        # Get target embeddings
        targets_np = np.array(targets, dtype=np.int32) % tower.vocab_size
        targets_gpu = xp.asarray(targets_np)
        tgt_matrices = tower.embeddings[targets_gpu]
        
        # Compute bindings
        bindings = xp.einsum('bij,bjk->bik', ctx_matrices, tgt_matrices)
        
        # Apply weights
        weights_gpu = xp.asarray(weights).reshape(-1, 1, 1)
        weighted_bindings = PHI_INV * weights_gpu * bindings
        
        # Scatter-add
        xp.add.at(tower._all_memories, satellite_indices, weighted_bindings)
        
        # Update counts
        sat_counts = xp.bincount(satellite_indices, minlength=tower.n_satellites)
        tower._satellite_n_bindings = tower._satellite_n_bindings + sat_counts.astype(xp.int64)
        
        model.n_patterns += batch_size
    
    def evaluate_model(model, eval_samples, n_eval=100):
        """Evaluate semantic similarity."""
        xp = model.xp
        similarities = []
        
        for ctx, tgt in eval_samples[:n_eval]:
            try:
                ctx_emb = model.tower._embed_sequence(ctx)
                
                basin_key = grace_basin_keys_batch_direct(
                    ctx_emb[None], model.basis,
                    n_iters=GRACE_ROUTING_ITERS,
                    resolution=GRACE_ROUTING_RESOLUTION,
                    xp=xp
                )[0]
                primes = xp.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53], dtype=xp.int64)
                sat_idx = int((xp.sum(basin_key * primes) % model.tower.n_satellites).get())
                
                sat_memory = model.tower._all_memories[sat_idx]
                settled = ctx_emb.T @ sat_memory
                
                tgt_emb = model.tower.embeddings[tgt % model.vocab_size]
                sim = float(frobenius_cosine(settled, tgt_emb, xp).get())
                similarities.append(sim)
            except Exception as e:
                raise RuntimeError(f"Similarity computation failed: {e}") from e
        
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
        'standard': [],
        'novelty': [],
        'coherence': [],
    }
    
    eval_samples = samples[N_BATCHES * BATCH_SIZE:]
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        batch = samples[start_idx:start_idx + BATCH_SIZE]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # Standard: uniform weights
        model_standard.tower.learn_batch(contexts, targets)
        model_standard.n_patterns += len(contexts)
        
        # Novelty-weighted: high error → high weight
        errors = compute_prediction_error(model_novelty, contexts, targets)
        novelty_weights = 1.0 + PHI_INV * errors  # 1.0 to 1.618
        novelty_weights = np.clip(novelty_weights, 1.0, 2.0)
        learn_batch_weighted(model_novelty, contexts, targets, novelty_weights)
        
        # Coherence-weighted: high coherence gain → high weight
        gains = compute_coherence_gain(model_coherence, contexts, targets)
        coherence_weights = 1.0 + PHI_INV * gains  # 1.0 to 1.618
        coherence_weights = np.clip(coherence_weights, 1.0, 2.0)
        learn_batch_weighted(model_coherence, contexts, targets, coherence_weights)
        
        # Evaluation
        if (batch_idx + 1) % EVAL_EVERY == 0:
            sim_standard = evaluate_model(model_standard, eval_samples)
            sim_novelty = evaluate_model(model_novelty, eval_samples)
            sim_coherence = evaluate_model(model_coherence, eval_samples)
            
            metrics['standard'].append(sim_standard)
            metrics['novelty'].append(sim_novelty)
            metrics['coherence'].append(sim_coherence)
            
            avg_novelty_weight = float(np.mean(novelty_weights))
            avg_coherence_weight = float(np.mean(coherence_weights))
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Standard:  sim={sim_standard:.4f}")
            print(f"    Novelty:   sim={sim_novelty:.4f} ({(sim_novelty/sim_standard-1)*100:+.1f}%), avg_w={avg_novelty_weight:.3f}")
            print(f"    Coherence: sim={sim_coherence:.4f} ({(sim_coherence/sim_standard-1)*100:+.1f}%), avg_w={avg_coherence_weight:.3f}")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("REWARD SHAPING RESULTS")
    print("="*80)
    
    late_standard = float(np.mean(metrics['standard'][2:])) if len(metrics['standard']) > 2 else float(np.mean(metrics['standard']))
    late_novelty = float(np.mean(metrics['novelty'][2:])) if len(metrics['novelty']) > 2 else float(np.mean(metrics['novelty']))
    late_coherence = float(np.mean(metrics['coherence'][2:])) if len(metrics['coherence']) > 2 else float(np.mean(metrics['coherence']))
    
    results = {
        'late_stage': {
            'standard': late_standard,
            'novelty': late_novelty,
            'coherence': late_coherence,
        },
        'improvement': {
            'novelty_vs_standard': (late_novelty / late_standard - 1) * 100,
            'coherence_vs_standard': (late_coherence / late_standard - 1) * 100,
            'coherence_vs_novelty': (late_coherence / late_novelty - 1) * 100,
        },
        'curves': metrics,
    }
    
    print(f"""
    LATE-STAGE SEMANTIC SIMILARITY:
      Standard (uniform):    {late_standard:.4f}
      Novelty-weighted:      {late_novelty:.4f} ({results['improvement']['novelty_vs_standard']:+.2f}%)
      Coherence-weighted:    {late_coherence:.4f} ({results['improvement']['coherence_vs_standard']:+.2f}%)
      
    THEORY VALIDATION:
      Coherence > Novelty: {late_coherence > late_novelty}
      Coherence > Standard: {late_coherence > late_standard}
      
    INTERPRETATION:
      If coherence > novelty: Compression reward is better than surprise reward
      This validates the "insight over novelty" genius hypothesis
      
    VERDICT: {'COHERENCE WINS' if late_coherence > max(late_standard, late_novelty) else 'COHERENCE DOES NOT HELP'}
    """)
    
    # Save
    results_path = "/checkpoints/coherence_reward_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


@app.local_entrypoint()
def main():
    print("Running Coherence Reward Test on Modal H100...")
    result = test_coherence_reward.remote()
    print(f"\nCoherence improvement: {result['improvement']['coherence_vs_standard']:+.2f}%")
