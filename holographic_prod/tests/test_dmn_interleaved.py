"""
DMN During Active Learning Test

THEORY:
    Genius minds are "always partially dreaming" - DMN runs un-suppressed during tasks.
    
    This test checks whether interleaving consolidation (mini-REM) with learning
    improves semantic compression faster.

BRAIN ANALOGY:
    - Typical: DMN suppressed during task → consolidation only in sleep
    - Genius: DMN runs in parallel → continuous consolidation

PHYSICS FRAMING:
    Like annealing with periodic temperature pulses.
    Consolidation = lower energy state; without it, system stays chaotic.
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

app = modal.App("dmn-interleaved-test")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=2700,  # 45 min
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_dmn_interleaved():
    """
    Test whether interleaving consolidation with learning improves semantic similarity.
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
        geometric_product
    )
    from holographic_prod.core.constants import PHI, PHI_INV
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast

    # Helper to handle both TowerMemory and MultiLevelTower
    def get_all_memories(tower):
        return getattr(tower, '_all_memories', None) or getattr(tower, '_all_memories', None)

    
    print("="*80)
    print("DMN DURING ACTIVE LEARNING TEST")
    print("="*80)
    print("""
    HYPOTHESIS: Interleaving consolidation with learning accelerates semantic compression.
    
    TEST:
        A. Standard (no consolidation during learning)
        B. Mini-REM every N batches (recombination + survival test)
        C. Continuous consolidation (φ-decay every batch)
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
    model_mini_rem = create_model()
    model_continuous = create_model()
    
    print(f"  ✓ Models: {model_standard.tower.n_satellites:,} satellites each")
    
    # =========================================================================
    # CONSOLIDATION FUNCTIONS
    # =========================================================================
    
    def mini_rem_consolidation(model, n_recombine=10, survival_iters=3):
        """
        Mini-REM: recombine satellite memories and survival test.
        
        Theory: Dreams recombine patterns, only stable structures survive.
        """
        xp = model.xp
        tower = model.tower
        
        # Get active satellites
        active_mask = tower._satellite_n_bindings > 0
        active_indices = xp.where(active_mask)[0]
        
        if len(active_indices) < 2:
            return 0
        
        # Random pairs for recombination
        survivors = 0
        
        for _ in range(n_recombine):
            # Pick two random active satellites
            pair_idx = np.random.choice(len(active_indices), size=2, replace=False)
            idx_a = int(active_indices[pair_idx[0]])
            idx_b = int(active_indices[pair_idx[1]])
            
            mem_a = get_all_memories(tower)[idx_a]
            mem_b = get_all_memories(tower)[idx_b]
            
            # Recombine: geometric product creates new pattern
            recombined = geometric_product(mem_a, mem_b)
            
            # Survival test: apply strong Grace contraction
            settled = recombined
            for _ in range(survival_iters):
                settled = grace_operator(settled, model.basis, xp)
            
            # Check stability: how much did it change?
            stability = float(frobenius_cosine(settled, recombined, xp).get())
            
            # If stable, add to a random empty-ish satellite
            if stability > PHI_INV:
                # Find satellite with low activity
                low_activity_mask = tower._satellite_n_bindings < 10
                low_indices = xp.where(low_activity_mask)[0]
                
                if len(low_indices) > 0:
                    target_idx = int(low_indices[np.random.randint(len(low_indices))])
                    # Add scaled recombined pattern
                    get_all_memories(tower)[target_idx] += PHI_INV * settled
                    survivors += 1
        
        return survivors
    
    def continuous_consolidation(model, decay_rate=0.001):
        """
        Continuous φ-decay: gentle forgetting of weak bindings.
        
        Theory: Low-energy patterns decay, high-energy preserved.
        """
        xp = model.xp
        tower = model.tower
        
        # Compute energy (Frobenius norm) per satellite
        energies = xp.linalg.norm(tower._all_memories, axis=(1, 2))
        
        # Apply φ-decay: weaker satellites decay more
        # decay = decay_rate × (1 - normalized_energy)
        max_energy = xp.max(energies) + 1e-10
        normalized = energies / max_energy
        
        # Decay factor: high energy = low decay
        decay_factors = 1.0 - decay_rate * (1.0 - normalized)[:, None, None]
        
        tower._all_memories *= decay_factors
        
        return float(xp.mean(decay_factors).get())
    
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
                
                sat_memory = get_all_memories(model.tower)[sat_idx]
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
    N_BATCHES = 50
    EVAL_EVERY = 5
    MINI_REM_EVERY = 5  # Mini-REM every 5 batches
    
    metrics = {
        'standard': {'sim': [], 'time': []},
        'mini_rem': {'sim': [], 'time': [], 'survivors': []},
        'continuous': {'sim': [], 'time': [], 'decay': []},
    }
    
    eval_samples = samples[N_BATCHES * BATCH_SIZE:]
    total_survivors = 0
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        batch = samples[start_idx:start_idx + BATCH_SIZE]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # Standard: just learn
        t0 = time.time()
        model_standard.tower.learn_batch(contexts, targets)
        model_standard.n_patterns += len(contexts)
        time_standard = time.time() - t0
        
        # Mini-REM: learn + periodic consolidation
        t0 = time.time()
        model_mini_rem.tower.learn_batch(contexts, targets)
        model_mini_rem.n_patterns += len(contexts)
        if (batch_idx + 1) % MINI_REM_EVERY == 0:
            survivors = mini_rem_consolidation(model_mini_rem, n_recombine=5)
            total_survivors += survivors
        time_mini_rem = time.time() - t0
        
        # Continuous: learn + decay every batch
        t0 = time.time()
        model_continuous.tower.learn_batch(contexts, targets)
        model_continuous.n_patterns += len(contexts)
        avg_decay = continuous_consolidation(model_continuous, decay_rate=0.001)
        time_continuous = time.time() - t0
        
        # Evaluation
        if (batch_idx + 1) % EVAL_EVERY == 0:
            sim_standard = evaluate_model(model_standard, eval_samples)
            sim_mini_rem = evaluate_model(model_mini_rem, eval_samples)
            sim_continuous = evaluate_model(model_continuous, eval_samples)
            
            metrics['standard']['sim'].append(sim_standard)
            metrics['standard']['time'].append(time_standard)
            
            metrics['mini_rem']['sim'].append(sim_mini_rem)
            metrics['mini_rem']['time'].append(time_mini_rem)
            metrics['mini_rem']['survivors'].append(total_survivors)
            
            metrics['continuous']['sim'].append(sim_continuous)
            metrics['continuous']['time'].append(time_continuous)
            metrics['continuous']['decay'].append(avg_decay)
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Standard:   sim={sim_standard:.4f}")
            print(f"    Mini-REM:   sim={sim_mini_rem:.4f} ({(sim_mini_rem/sim_standard-1)*100:+.1f}%), survivors={total_survivors}")
            print(f"    Continuous: sim={sim_continuous:.4f} ({(sim_continuous/sim_standard-1)*100:+.1f}%), decay={avg_decay:.4f}")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("DMN INTERLEAVED RESULTS")
    print("="*80)
    
    def get_late_avg(lst):
        return float(np.mean(lst[2:])) if len(lst) > 2 else float(np.mean(lst))
    
    late_standard = get_late_avg(metrics['standard']['sim'])
    late_mini_rem = get_late_avg(metrics['mini_rem']['sim'])
    late_continuous = get_late_avg(metrics['continuous']['sim'])
    
    avg_time_standard = np.mean(metrics['standard']['time'])
    avg_time_mini_rem = np.mean(metrics['mini_rem']['time'])
    avg_time_continuous = np.mean(metrics['continuous']['time'])
    
    results = {
        'late_stage': {
            'standard': late_standard,
            'mini_rem': late_mini_rem,
            'continuous': late_continuous,
        },
        'improvement': {
            'mini_rem_vs_standard': (late_mini_rem / late_standard - 1) * 100,
            'continuous_vs_standard': (late_continuous / late_standard - 1) * 100,
        },
        'throughput': {
            'mini_rem_ratio': avg_time_standard / avg_time_mini_rem,
            'continuous_ratio': avg_time_standard / avg_time_continuous,
        },
        'mini_rem_survivors': total_survivors,
        'curves': metrics,
    }
    
    print(f"""
    LATE-STAGE SEMANTIC SIMILARITY:
      Standard:   {late_standard:.4f} (baseline)
      Mini-REM:   {late_mini_rem:.4f} ({results['improvement']['mini_rem_vs_standard']:+.2f}%)
      Continuous: {late_continuous:.4f} ({results['improvement']['continuous_vs_standard']:+.2f}%)
      
    THROUGHPUT RATIO (vs standard):
      Mini-REM:   {results['throughput']['mini_rem_ratio']:.2f}x
      Continuous: {results['throughput']['continuous_ratio']:.2f}x
      
    MINI-REM SURVIVORS: {total_survivors}
      (Patterns that survived Grace survival test)
      
    THEORY VALIDATION:
      Interleaving helps: {late_mini_rem > late_standard or late_continuous > late_standard}
      Best method: {max(['standard', 'mini_rem', 'continuous'], key=lambda x: results['late_stage'][x])}
      
    VERDICT: {'DMN INTERLEAVING HELPS' if max(late_mini_rem, late_continuous) > late_standard else 'DMN INTERLEAVING DOES NOT HELP'}
    """)
    
    # Save
    results_path = "/checkpoints/dmn_interleaved_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


@app.local_entrypoint()
def main():
    print("Running DMN Interleaved Test on Modal H100...")
    result = test_dmn_interleaved.remote()
    best = max(['standard', 'mini_rem', 'continuous'], key=lambda x: result['late_stage'][x])
    print(f"\nBest method: {best} ({result['late_stage'][best]:.4f})")
