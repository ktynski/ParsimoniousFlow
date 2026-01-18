"""
Satellite-Level Holonomy Test

THEORY:
    Satellites = Semantic Regimes (Grace basin clusters)
    Holonomy = Regime Shift Detection (confidence deviation from baseline)
    
    From true_universal_signal.py (trading analogy):
        - Track regime stability over time
        - Detect when regime performance deviates from history
        - Exploit the deviation (boost learning for struggling regimes)
    
    The key insight: Random pattern noise cancels at satellite level,
    but systematic semantic novelty creates persistent confidence drops.

IMPLEMENTATION:
    1. Track per-satellite confidence EMA (integrated into learning)
    2. Compute holonomy = historical_conf / current_conf  
    3. Weight bindings by holonomy (boost struggling satellites)
    4. Zero Python loops in hot path - all vectorized on GPU

TEST:
    A/B comparison: Standard vs Holonomy-weighted learning
    Success criteria: Improved semantic similarity without throughput loss
"""

import modal
import numpy as np
import time
from typing import List, Tuple, Dict, Any

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

app = modal.App("satellite-holonomy-test")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_satellite_holonomy():
    """
    Test satellite-level holonomy with INTEGRATED tracking.
    
    Key optimization: Track confidence DURING learning, not separate pass.
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
    from holographic_prod.core.algebra import grace_basin_keys_batch_direct
    from holographic_prod.core.constants import PHI, PHI_INV
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    
    print("="*80)
    print("SATELLITE HOLONOMY TEST - INTEGRATED TRACKING")
    print("="*80)
    print("""
    THEORY:
        Satellites = Semantic Regimes (Grace basin clusters)
        Holonomy = Regime Shift Detection
        
        holonomy = historical_conf / current_conf
        - > 1: Satellite is struggling (novel/forgotten patterns)
        - = 1: Satellite is stable (well-learned)
        - < 1: Satellite is improving (consolidating)
        
        Weight = 1 + φ⁻¹ × max(0, holonomy - 1)
        → Boost learning for struggling satellites
    """)
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    print("\n  Loading data...")
    
    from datasets import load_dataset
    
    vocab_path = "/checkpoints/vocabulary.npz"
    vocab_size = 50_000
    
    try:
        vocab_data = np.load(vocab_path, allow_pickle=True)
        word_to_idx = vocab_data['word_to_idx'].item()
        idx_to_word = vocab_data['idx_to_word'].item()
        print(f"  ✓ Vocabulary: {len(word_to_idx):,} words")
    except FileNotFoundError:
        print("  No cached vocabulary - building from scratch...")
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        word_counts = defaultdict(int)
        for i, item in enumerate(tqdm(ds.take(50_000), total=50_000, desc="Counting")):
            for word in item['text'].lower().split():
                word_counts[word] += 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])[:vocab_size-4]
        word_to_idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        for i, (word, _) in enumerate(sorted_words):
            word_to_idx[word] = i + 4
        idx_to_word = {v: k for k, v in word_to_idx.items()}
        np.savez(vocab_path, word_to_idx=word_to_idx, idx_to_word=idx_to_word)
        print(f"  ✓ Built vocabulary: {len(word_to_idx):,} words")
    
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
            if tgt != 1:  # Skip <unk> targets
                samples.append((ctx, tgt))
            if len(samples) >= 100_000:
                break
        if len(samples) >= 100_000:
            break
    
    print(f"  ✓ Prepared {len(samples):,} samples")
    
    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    print("\n  Initializing models...")
    
    # Model A: Standard learning
    model_standard = HolographicMemory(
        vocab_size=len(word_to_idx),
        max_levels=4,
        seed=42,
        use_gpu=True,
        grounded_embeddings=cp.asarray(grounded_embs),
    )
    model_standard.set_grounded_embeddings(cp.asarray(grounded_embs))
    
    # Model B: Holonomy-weighted learning (same architecture)
    model_holonomy = HolographicMemory(
        vocab_size=len(word_to_idx),
        max_levels=4,
        seed=42,
        use_gpu=True,
        grounded_embeddings=cp.asarray(grounded_embs),
    )
    model_holonomy.set_grounded_embeddings(cp.asarray(grounded_embs))
    
    n_satellites = model_standard.tower.n_satellites
    print(f"  ✓ Models initialized: {n_satellites:,} satellites each")
    
    # =========================================================================
    # SATELLITE TRACKING (for holonomy model only)
    # =========================================================================
    
    # Per-satellite confidence EMA (starts at 0.5 = uncertain)
    sat_confidence_ema = cp.ones(n_satellites, dtype=cp.float32) * 0.5
    sat_pattern_count = cp.zeros(n_satellites, dtype=cp.int64)
    EMA_ALPHA = 0.1
    
    # =========================================================================
    # OPTIMIZED HOLONOMY LEARNING FUNCTION
    # =========================================================================
    
    def learn_batch_with_holonomy(
        model,
        contexts: List[List[int]],
        targets: List[int],
        sat_conf_ema: cp.ndarray,
        sat_count: cp.ndarray,
        alpha: float = PHI_INV,
        ema_alpha: float = 0.1,
    ) -> Tuple[cp.ndarray, cp.ndarray, Dict[str, float]]:
        """
        INTEGRATED holonomy-weighted learning.
        
        Computes confidence DURING learning (not separate pass).
        All operations vectorized on GPU.
        
        Returns:
            (updated_sat_conf_ema, updated_sat_count, stats_dict)
        """
        xp = model.xp
        tower = model.tower
        
        if not contexts:
            return sat_conf_ema, sat_count, {}
        
        batch_size = len(contexts)
        
        # 1. Embed contexts (same as standard)
        ctx_matrices = tower._embed_sequences_batch(contexts)
        
        # 2. Route to satellites (same as standard)
        basin_keys = grace_basin_keys_batch_direct(
            ctx_matrices, model.basis,
            n_iters=GRACE_ROUTING_ITERS,
            resolution=GRACE_ROUTING_RESOLUTION,
            xp=xp
        )
        primes = xp.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53], dtype=xp.int64)
        satellite_indices = (xp.sum(basin_keys * primes, axis=1) % tower.n_satellites).astype(xp.int32)
        
        # 3. Get target embeddings (same as standard)
        targets_np = np.array(targets, dtype=np.int32) % tower.vocab_size
        targets_gpu = xp.asarray(targets_np)
        tgt_matrices = tower.embeddings[targets_gpu]  # [batch, 4, 4]
        
        # 4. HOLONOMY: Estimate current confidence BEFORE learning
        # Get current satellite memories for these patterns
        sat_memories = tower._all_memories[satellite_indices]  # [batch, 4, 4]
        
        # Unbind: retrieved ≈ ctx^T @ memory
        ctx_inv = xp.swapaxes(ctx_matrices, -2, -1)  # [batch, 4, 4]
        retrieved = xp.einsum('bij,bjk->bik', ctx_inv, sat_memories)  # [batch, 4, 4]
        
        # Frobenius cosine similarity: current confidence
        retrieved_flat = retrieved.reshape(batch_size, -1)
        target_flat = tgt_matrices.reshape(batch_size, -1)
        
        dots = xp.sum(retrieved_flat * target_flat, axis=1)
        norms_r = xp.linalg.norm(retrieved_flat, axis=1)
        norms_t = xp.linalg.norm(target_flat, axis=1)
        
        current_conf = dots / (norms_r * norms_t + 1e-10)  # [batch]
        current_conf = xp.clip(current_conf, 0.01, 1.0)
        
        # 5. HOLONOMY: Compare to satellite history
        historical_conf = sat_conf_ema[satellite_indices]  # [batch]
        
        # holonomy = historical / current
        # > 1 means satellite is doing WORSE than history (novel/struggling)
        holonomy = historical_conf / (current_conf + 1e-10)  # [batch]
        
        # Weight = 1 + α × max(0, holonomy - 1)
        # Boost patterns in struggling satellites
        surprise_factor = xp.maximum(0, holonomy - 1.0)
        weights = 1.0 + alpha * surprise_factor  # [batch]
        weights = xp.clip(weights, 1.0, 2.0)
        
        # 6. WEIGHTED BINDING
        bindings = xp.einsum('bij,bjk->bik', ctx_matrices, tgt_matrices)  # [batch, 4, 4]
        
        # Apply per-pattern weights
        weighted_bindings = PHI_INV * weights.reshape(-1, 1, 1) * bindings
        
        # Scatter-add to satellites
        xp.add.at(tower._all_memories, satellite_indices, weighted_bindings)
        
        # Update binding counts
        sat_counts_batch = xp.bincount(satellite_indices, minlength=tower.n_satellites)
        tower._satellite_n_bindings = tower._satellite_n_bindings + sat_counts_batch.astype(xp.int64)
        
        # 7. UPDATE SATELLITE EMAS
        # Aggregate confidence by satellite (weighted mean)
        # Use scatter-add for sum and count
        conf_sum = xp.zeros(tower.n_satellites, dtype=xp.float32)
        conf_count = xp.zeros(tower.n_satellites, dtype=xp.float32)
        
        xp.add.at(conf_sum, satellite_indices, current_conf)
        xp.add.at(conf_count, satellite_indices, xp.ones(batch_size, dtype=xp.float32))
        
        # Update EMA where count > 0
        active_mask = conf_count > 0
        new_conf = conf_sum / (conf_count + 1e-10)
        
        # EMA update: new = α × current + (1-α) × old
        sat_conf_ema = xp.where(
            active_mask,
            ema_alpha * new_conf + (1 - ema_alpha) * sat_conf_ema,
            sat_conf_ema
        )
        sat_count = sat_count + sat_counts_batch.astype(xp.int64)
        
        # Update model state
        model.n_patterns += batch_size
        model.learn_count += batch_size
        model._global_memory_dirty = True
        
        # Compute stats (minimal GPU→CPU sync)
        stats = {
            'avg_weight': float(weights.mean().get()),
            'avg_holonomy': float(holonomy.mean().get()),
            'avg_confidence': float(current_conf.mean().get()),
        }
        
        return sat_conf_ema, sat_count, stats
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPARISON")
    print("="*80)
    
    BATCH_SIZE = 2048
    N_BATCHES = 40
    EVAL_EVERY = 5
    
    standard_sims = []
    holonomy_sims = []
    standard_times = []
    holonomy_times = []
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(samples))
        batch = samples[start_idx:end_idx]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # === MODEL A: Standard learning ===
        t0 = time.time()
        model_standard.tower.learn_batch(contexts, targets)
        model_standard.n_patterns += len(contexts)
        time_standard = time.time() - t0
        
        # === MODEL B: Holonomy-weighted learning ===
        t0 = time.time()
        sat_confidence_ema, sat_pattern_count, stats = learn_batch_with_holonomy(
            model_holonomy,
            contexts,
            targets,
            sat_confidence_ema,
            sat_pattern_count,
        )
        time_holonomy = time.time() - t0
        
        # === EVALUATION ===
        if (batch_idx + 1) % EVAL_EVERY == 0:
            eval_batch = samples[N_BATCHES * BATCH_SIZE:N_BATCHES * BATCH_SIZE + 200]
            
            def evaluate_model(model) -> float:
                """Compute average semantic similarity."""
                xp = model.xp
                similarities = []
                
                for ctx, tgt in eval_batch[:100]:
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
                        
                        dot = float(xp.sum(settled * tgt_emb).get())
                        norm_s = float(xp.linalg.norm(settled).get())
                        norm_t = float(xp.linalg.norm(tgt_emb).get())
                        
                        if norm_s > 1e-10 and norm_t > 1e-10:
                            sim = dot / (norm_s * norm_t)
                            similarities.append(sim)
                    except Exception as e:
                        raise RuntimeError(f"Similarity computation failed: {e}") from e
                
                if not similarities:
                    raise RuntimeError("No similarities computed - all samples failed")
                return np.mean(similarities)
            
            sim_standard = evaluate_model(model_standard)
            sim_holonomy = evaluate_model(model_holonomy)
            
            standard_sims.append(sim_standard)
            holonomy_sims.append(sim_holonomy)
            standard_times.append(time_standard)
            holonomy_times.append(time_holonomy)
            
            throughput_standard = len(batch) / time_standard if time_standard > 0 else 0
            throughput_holonomy = len(batch) / time_holonomy if time_holonomy > 0 else 0
            
            # Compute Zipf diagnostic
            sat_count_cpu = sat_pattern_count.get()
            active_mask = sat_count_cpu > 0
            n_active = int(np.sum(active_mask))
            
            if n_active > 10:
                sorted_counts = np.sort(sat_count_cpu[active_mask])[::-1]
                zipf_ratio = sorted_counts[0] / (sorted_counts[len(sorted_counts)//2] + 1)
                top10_share = np.sum(sorted_counts[:10]) / (np.sum(sorted_counts) + 1)
            else:
                zipf_ratio = 0
                top10_share = 0
            
            sat_conf_cpu = sat_confidence_ema.get()
            mean_conf = float(np.mean(sat_conf_cpu[active_mask])) if n_active > 0 else 0
            std_conf = float(np.std(sat_conf_cpu[active_mask])) if n_active > 0 else 0
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Standard:  sim={sim_standard:.4f}, rate={throughput_standard:,.0f}/s")
            print(f"    Holonomy:  sim={sim_holonomy:.4f}, rate={throughput_holonomy:,.0f}/s")
            print(f"    Stats:     avg_weight={stats['avg_weight']:.3f}, avg_conf={stats['avg_confidence']:.3f}")
            print(f"    Satellites: {n_active:,} active, conf={mean_conf:.3f}±{std_conf:.3f}")
            print(f"    Zipf:      ratio={zipf_ratio:.1f}, top10={top10_share*100:.1f}%")
            print(f"    Δ sim:     {sim_holonomy - sim_standard:+.4f} ({'✓' if sim_holonomy > sim_standard else '✗'})")
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    avg_sim_standard = np.mean(standard_sims)
    avg_sim_holonomy = np.mean(holonomy_sims)
    
    avg_time_standard = np.mean(standard_times)
    avg_time_holonomy = np.mean(holonomy_times)
    
    throughput_standard = BATCH_SIZE / avg_time_standard if avg_time_standard > 0 else 0
    throughput_holonomy = BATCH_SIZE / avg_time_holonomy if avg_time_holonomy > 0 else 0
    
    speed_ratio = throughput_holonomy / throughput_standard if throughput_standard > 0 else 0
    
    # Late-stage comparison (after history builds)
    late_sims_standard = standard_sims[2:]  # Skip first 2 evals
    late_sims_holonomy = holonomy_sims[2:]
    
    late_avg_standard = np.mean(late_sims_standard) if late_sims_standard else 0
    late_avg_holonomy = np.mean(late_sims_holonomy) if late_sims_holonomy else 0
    
    print(f"""
    OVERALL:
      Standard:     sim={avg_sim_standard:.4f}
      Holonomy:     sim={avg_sim_holonomy:.4f}
      Improvement:  {(avg_sim_holonomy/avg_sim_standard - 1)*100:+.2f}%
      
    LATE-STAGE (after history builds):
      Standard:     sim={late_avg_standard:.4f}
      Holonomy:     sim={late_avg_holonomy:.4f}
      Improvement:  {(late_avg_holonomy/late_avg_standard - 1)*100:+.2f}%
      
    THROUGHPUT:
      Standard:     {throughput_standard:,.0f}/s
      Holonomy:     {throughput_holonomy:,.0f}/s
      Speed ratio:  {speed_ratio:.2f}x
      
    SUCCESS CRITERIA:
      ✓ Late-stage improvement: {late_avg_holonomy > late_avg_standard}
      ✓ Speed > 0.8x:           {speed_ratio > 0.8}
    """)
    
    if late_avg_holonomy > late_avg_standard and speed_ratio > 0.8:
        print("    VERDICT: ✅ SATELLITE HOLONOMY HELPS")
    elif late_avg_holonomy > late_avg_standard:
        print("    VERDICT: ⚠️ HELPS BUT SLOW - needs optimization")
    else:
        print("    VERDICT: ❌ NO BENEFIT")
    
    print("\nTest complete!")
    return {
        'standard_sim': avg_sim_standard,
        'holonomy_sim': avg_sim_holonomy,
        'late_standard': late_avg_standard,
        'late_holonomy': late_avg_holonomy,
        'speed_ratio': speed_ratio,
    }


@app.local_entrypoint()
def main():
    print("Running Satellite Holonomy Test on Modal H100...")
    result = test_satellite_holonomy.remote()
    print(f"\nReturned: {result}")
