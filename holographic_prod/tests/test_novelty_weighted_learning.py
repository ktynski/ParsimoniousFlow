"""
Novelty-Weighted Learning Test (TDD)
====================================

THEORY:
    Cross-scale holonomy measures how much fine-scale (recent tokens)
    DIVERGES from coarse-scale (full context).
    
    In language, high holonomy = NOVELTY = more information added.
    
    Therefore: Weight learning by holonomy magnitude.
    - High holonomy patterns encoded MORE strongly (informative)
    - Low holonomy patterns encoded LESS strongly (redundant)

SUCCESS CRITERIA:
    1. Semantic similarity improves faster with novelty weighting
    2. No significant speed degradation (<10% slower)
    3. Memory usage unchanged

TEST PLAN:
    A. Train standard model for N batches
    B. Train novelty-weighted model for N batches  
    C. Compare semantic similarity at each checkpoint
    D. Compare throughput (samples/sec)

Run:
    modal run holographic_prod/tests/test_novelty_weighted_learning.py

Version: v1.0.0
"""

import modal
import numpy as np
import time
from typing import List, Tuple, Dict, Any
from collections import defaultdict

# Modal setup
app = modal.App("novelty-learning-test")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install([
        "numpy>=1.24",
        "cupy-cuda12x>=12.0",
        "scipy>=1.10",
        "datasets>=2.14",
    ])
    .run_commands(
        "apt-get update && apt-get install -y unzip",
        "mkdir -p /tmp/glove",
        'python -c "import urllib.request; urllib.request.urlretrieve(\'http://nlp.stanford.edu/data/glove.6B.zip\', \'/tmp/glove/glove.6B.zip\')"',
        "unzip -j /tmp/glove/glove.6B.zip glove.6B.50d.txt -d /tmp/glove",
        "rm /tmp/glove/glove.6B.zip",
    )
    .add_local_dir("holographic_prod", "/root/project/holographic_prod")
)


class SatelliteHolonomyTracker:
    """
    Track per-SATELLITE confidence for regime-level holonomy.
    
    THEORY (true_universal_signal analogy):
        Trading: Track regime stability, detect regime shifts
        Language: Track satellite stability, detect semantic shifts
        
        Satellites = semantic basins = "regimes"
        Patterns in same satellite = related concepts
        
        If a satellite's confidence drops → that semantic region is novel
        → boost learning for ALL patterns routed to that satellite
        
    This discriminates signal from noise:
        - Random misses on individual patterns → cancel out in satellite average
        - Systematic novelty in semantic region → consistent confidence drop → boost
    """
    
    def __init__(self, n_satellites: int, ema_alpha: float = 0.1):
        # Per-satellite confidence history (EMA)
        self.confidence_ema = np.ones(n_satellites, dtype=np.float32) * 0.5  # Start uncertain
        self.pattern_count = np.zeros(n_satellites, dtype=np.int64)
        self.ema_alpha = ema_alpha
        self.n_satellites = n_satellites
    
    def compute_weights_and_update(
        self,
        model,
        contexts: List[List[int]],
        targets: List[int],
        alpha: float = 0.618,
    ) -> np.ndarray:
        """
        Compute satellite-level holonomy weights.
        
        1. Route each pattern to its satellite
        2. Measure current confidence for each pattern
        3. Compare to satellite's historical confidence (coarse scale)
        4. Boost patterns in satellites that are "shifting" (holonomy > 1)
        5. Update satellite confidence histories
        
        Returns:
            [batch] array of weights
        """
        xp = model.xp
        batch_size = len(contexts)
        
        if batch_size == 0:
            return np.array([], dtype=np.float32)
        
        # Get satellite indices for each pattern
        from holographic_prod.core.algebra import grace_basin_keys_batch_direct
        from holographic_prod.memory.holographic_memory_unified import (
            GRACE_ROUTING_ITERS, GRACE_ROUTING_RESOLUTION
        )
        
        ctx_matrices = model.tower._embed_sequences_batch(contexts)
        basin_keys = grace_basin_keys_batch_direct(
            ctx_matrices, model.basis,
            n_iters=GRACE_ROUTING_ITERS,
            resolution=GRACE_ROUTING_RESOLUTION,
            xp=xp
        )
        satellite_indices = model.tower._route_to_satellites_batch(basin_keys)
        
        # Transfer to CPU for tracking
        if hasattr(satellite_indices, 'get'):
            sat_idx_cpu = satellite_indices.get().astype(np.int64)
        else:
            sat_idx_cpu = satellite_indices.astype(np.int64)
        
        # Measure CURRENT confidence for each pattern
        targets_arr = xp.asarray(targets, dtype=xp.int32) % model.tower.embeddings.shape[0]
        target_embs = model.tower.embeddings[targets_arr]  # [batch, 4, 4]
        
        # Get satellite memories and unbind
        sat_memories = model.tower._all_memories[satellite_indices]  # [batch, 4, 4]
        settled = xp.einsum('bij,bjk->bik', ctx_matrices.swapaxes(-2, -1), sat_memories)
        
        # Frobenius cosine similarity
        settled_flat = settled.reshape(batch_size, -1)
        target_flat = target_embs.reshape(batch_size, -1)
        
        dots = xp.sum(settled_flat * target_flat, axis=1)
        norms_s = xp.linalg.norm(settled_flat, axis=1)
        norms_t = xp.linalg.norm(target_flat, axis=1)
        
        current_conf = dots / (norms_s * norms_t + 1e-10)  # [batch]
        
        # Transfer to CPU
        if hasattr(current_conf, 'get'):
            current_conf = current_conf.get()
        
        # Compute per-pattern holonomy based on SATELLITE history
        # Holonomy = historical_conf / current_conf
        # > 1 means satellite is doing WORSE than usual → novel region
        historical_conf = self.confidence_ema[sat_idx_cpu]
        
        # Avoid division issues
        current_conf_safe = np.maximum(current_conf, 0.01)
        holonomy = historical_conf / current_conf_safe  # [batch]
        
        # Weight = 1 + α × max(0, holonomy - 1)
        # Boost when satellite is underperforming its history
        surprise_factor = np.maximum(0, holonomy - 1.0)
        weights = 1.0 + alpha * surprise_factor
        weights = np.clip(weights, 1.0, 2.0)
        
        # Update satellite confidence histories (aggregate by satellite)
        # Use scatter-add style update
        unique_sats = np.unique(sat_idx_cpu)
        for sat in unique_sats:
            mask = sat_idx_cpu == sat
            sat_conf = float(np.mean(current_conf[mask]))
            
            # EMA update
            self.confidence_ema[sat] = (
                self.ema_alpha * sat_conf + 
                (1 - self.ema_alpha) * self.confidence_ema[sat]
            )
            self.pattern_count[sat] += int(np.sum(mask))
        
        return weights.astype(np.float32)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics including Zipf diagnostic."""
        active_sats = self.pattern_count > 0
        active_counts = self.pattern_count[active_sats]
        
        # Zipf diagnostic: ratio of max to median count
        # Zipfian: ratio >> 10 (power law)
        # Uniform: ratio ≈ 1-2
        if len(active_counts) > 0:
            sorted_counts = np.sort(active_counts)[::-1]
            max_count = sorted_counts[0]
            median_count = sorted_counts[len(sorted_counts)//2]
            zipf_ratio = max_count / (median_count + 1)
            
            # Top-10 satellite share (should be high for Zipf)
            top10_share = np.sum(sorted_counts[:10]) / (np.sum(sorted_counts) + 1)
        else:
            zipf_ratio = 0.0
            top10_share = 0.0
        
        return {
            'n_active_satellites': int(np.sum(active_sats)),
            'mean_confidence': float(np.mean(self.confidence_ema[active_sats])) if np.any(active_sats) else 0.0,
            'conf_std': float(np.std(self.confidence_ema[active_sats])) if np.any(active_sats) else 0.0,
            'zipf_ratio': float(zipf_ratio),  # max/median count ratio
            'top10_share': float(top10_share),  # fraction of patterns in top 10 satellites
        }


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
)
def test_novelty_weighted_learning():
    """
    A/B test: Standard vs Novelty-Weighted Learning
    
    SUCCESS CRITERIA:
        1. Novelty-weighted achieves HIGHER semantic similarity
        2. Speed degradation < 10%
        3. Memory usage similar
    """
    import os
    import sys
    os.environ['HF_HOME'] = '/tmp/hf_cache'
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    from datasets import load_dataset
    
    print("="*80)
    print("NOVELTY-WEIGHTED LEARNING A/B TEST")
    print("="*80)
    
    print("""
    HYPOTHESIS: Weighting learning by cross-scale holonomy improves
    semantic similarity by encoding novel patterns more strongly.
    
    TEST:
        A. Standard learning (uniform weight)
        B. Novelty-weighted learning (weight = 1 + φ⁻¹ × holonomy)
    """)
    
    # =========================================================================
    # SETUP: Build vocabulary and prepare data
    # =========================================================================
    print("\n  Loading data...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    # Build vocabulary
    word_counts = defaultdict(int)
    doc_count = 0
    for item in ds:
        words = item['text'].lower().split()
        for w in words:
            word_counts[w] += 1
        doc_count += 1
        if doc_count >= 10000:
            break
    
    vocab_size = 50000
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    word_to_idx = {'<unk>': 0, '<pad>': 1}
    for word, _ in sorted_words[:vocab_size - 2]:
        idx = len(word_to_idx)
        word_to_idx[word] = idx
    
    print(f"  ✓ Vocabulary: {len(word_to_idx):,} words")
    
    # Load grounded embeddings
    from holographic_prod.core.grounded_embeddings import (
        load_glove_embeddings,
        pretrained_to_SO4
    )
    
    glove_embs, covered = load_glove_embeddings(word_to_idx, glove_dim=50, cache_dir="/tmp/glove")
    grounded_embs = pretrained_to_SO4(glove_embs)
    print(f"  ✓ GloVe coverage: {covered}/{len(word_to_idx)} ({covered/len(word_to_idx)*100:.1f}%)")
    
    # Prepare samples with fixed context length
    CONTEXT_LEN = 32
    
    def tokenize(text: str) -> List[int]:
        return [word_to_idx.get(w.lower(), 0) for w in text.split()]
    
    samples = []
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    for item in ds:
        tokens = tokenize(item['text'])
        if len(tokens) >= CONTEXT_LEN + 1:
            for i in range(0, len(tokens) - CONTEXT_LEN, 10):
                context = tokens[i:i + CONTEXT_LEN]
                target = tokens[i + CONTEXT_LEN]
                samples.append((context, target))
        if len(samples) >= 100000:  # 100K samples
            break
    
    print(f"  ✓ Prepared {len(samples):,} samples")
    
    # =========================================================================
    # INITIALIZE TWO MODELS
    # =========================================================================
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    
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
    
    # Model B: Will use same architecture, just weighted learning
    model_novelty = HolographicMemory(
        vocab_size=len(word_to_idx),
        max_levels=4,
        seed=42,  # Same seed for fair comparison
        use_gpu=True,
        grounded_embeddings=cp.asarray(grounded_embs),
    )
    model_novelty.set_grounded_embeddings(cp.asarray(grounded_embs))
    
    print(f"  ✓ Models initialized: {model_standard.tower.n_satellites:,} satellites each")
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPARISON")
    print("="*80)
    
    BATCH_SIZE = 2048
    N_BATCHES = 30
    EVAL_EVERY = 5
    
    # Metrics storage
    standard_metrics = {'semantic_sim': [], 'throughput': [], 'time': []}
    novelty_metrics = {'semantic_sim': [], 'throughput': [], 'time': []}
    
    # PHI constants
    PHI_INV = 0.6180339887
    
    # Satellite-level holonomy tracker (regime-based novelty)
    holonomy_tracker = SatelliteHolonomyTracker(
        n_satellites=model_novelty.tower.n_satellites,
        ema_alpha=0.1
    )
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(samples))
        batch = samples[start_idx:end_idx]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # =====================================================================
        # Model A: Standard Learning
        # =====================================================================
        t0 = time.time()
        model_standard.learn_batch(contexts, targets)
        time_standard = time.time() - t0
        
        # =====================================================================
        # Model B: Satellite-Level Holonomy Learning (regime-based novelty)
        # =====================================================================
        t0 = time.time()
        
        # Compute SATELLITE-LEVEL holonomy weights
        # Tracks per-satellite confidence history
        # Boosts patterns in satellites that are "shifting" (underperforming history)
        weights = holonomy_tracker.compute_weights_and_update(
            model_novelty,
            contexts,
            targets,
            alpha=PHI_INV
        )
        
        # Custom weighted learning
        # Instead of model_novelty.learn_batch, we do weighted version
        xp = model_novelty.xp
        
        # Get embeddings and route to satellites (same as standard)
        ctx_matrices = model_novelty.tower._embed_sequences_batch(contexts)
        
        from holographic_prod.core.algebra import grace_basin_keys_batch_direct
        from holographic_prod.memory.holographic_memory_unified import (
            GRACE_ROUTING_ITERS, GRACE_ROUTING_RESOLUTION
        )
        
        basin_keys = grace_basin_keys_batch_direct(
            ctx_matrices, model_novelty.basis,
            n_iters=GRACE_ROUTING_ITERS,
            resolution=GRACE_ROUTING_RESOLUTION,
            xp=xp
        )
        satellite_indices = model_novelty.tower._route_to_satellites_batch(basin_keys).astype(xp.int32)
        
        # Get target embeddings
        targets_np = np.array(targets, dtype=np.int32) % model_novelty.vocab_size
        targets_arr = xp.asarray(targets_np)
        tgt_matrices = model_novelty.tower.embeddings[targets_arr]
        
        # Compute bindings
        bindings = xp.einsum('bij,bjk->bik', ctx_matrices, tgt_matrices)
        
        # Apply novelty weights!
        weights_gpu = xp.asarray(weights).reshape(-1, 1, 1)  # [batch, 1, 1]
        weighted_bindings = PHI_INV * weights_gpu * bindings  # Scale by φ⁻¹ × weight
        
        # Scatter-add to satellites
        xp.add.at(model_novelty.tower._all_memories, satellite_indices, weighted_bindings)
        
        # Update counts
        ones = xp.ones(len(contexts), dtype=xp.uint64)
        xp.add.at(model_novelty.tower._satellite_n_bindings, satellite_indices.astype(xp.int64), ones)
        
        # Update model state
        model_novelty.n_patterns += len(contexts)
        model_novelty.learn_count += len(contexts)
        
        time_novelty = time.time() - t0
        
        # =====================================================================
        # EVALUATION
        # =====================================================================
        if (batch_idx + 1) % EVAL_EVERY == 0:
            # Evaluate on a held-out set
            eval_batch = samples[N_BATCHES * BATCH_SIZE:N_BATCHES * BATCH_SIZE + 100]
            
            def evaluate_model(model) -> float:
                """Compute average semantic similarity."""
                similarities = []
                for ctx, tgt in eval_batch:
                    try:
                        ctx_emb = model.tower._embed_sequence(ctx)
                        
                        # Route and retrieve
                        from holographic_prod.core.algebra import grace_basin_key_direct
                        basin_key = grace_basin_key_direct(
                            ctx_emb, model.basis,
                            n_iters=GRACE_ROUTING_ITERS,
                            resolution=GRACE_ROUTING_RESOLUTION,
                            xp=model.xp
                        )
                        sat_idx = model.tower._route_to_satellite(tuple(int(k) for k in basin_key))
                        sat_memory = model.tower._all_memories[sat_idx]
                        
                        # Unbind
                        settled = ctx_emb.T @ sat_memory
                        
                        # Compare to target
                        tgt_emb = model.tower.embeddings[tgt % model.vocab_size]
                        
                        # Frobenius cosine
                        dot = float(model.xp.sum(settled * tgt_emb))
                        norm_s = float(model.xp.linalg.norm(settled))
                        norm_t = float(model.xp.linalg.norm(tgt_emb))
                        
                        if norm_s > 1e-10 and norm_t > 1e-10:
                            sim = dot / (norm_s * norm_t)
                            similarities.append(sim)
                    except Exception as e:
                        raise RuntimeError(f"Similarity computation failed: {e}") from e
                
                if not similarities:
                    raise RuntimeError("No similarities computed - all samples failed")
                return np.mean(similarities)
            
            sim_standard = evaluate_model(model_standard)
            sim_novelty = evaluate_model(model_novelty)
            
            throughput_standard = len(batch) / time_standard if time_standard > 0 else 0
            throughput_novelty = len(batch) / time_novelty if time_novelty > 0 else 0
            
            standard_metrics['semantic_sim'].append(sim_standard)
            standard_metrics['throughput'].append(throughput_standard)
            standard_metrics['time'].append(time_standard)
            
            novelty_metrics['semantic_sim'].append(sim_novelty)
            novelty_metrics['throughput'].append(throughput_novelty)
            novelty_metrics['time'].append(time_novelty)
            
            # Compute average weight for this batch
            avg_weight = float(np.mean(weights))
            sat_stats = holonomy_tracker.get_stats()
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Standard:  sim={sim_standard:.4f}, rate={throughput_standard:.0f}/s")
            print(f"    Novelty:   sim={sim_novelty:.4f}, rate={throughput_novelty:.0f}/s, avg_weight={avg_weight:.3f}")
            print(f"    Satellites: {sat_stats['n_active_satellites']} active, conf={sat_stats['mean_confidence']:.3f}±{sat_stats['conf_std']:.3f}")
            print(f"    Zipf diagnostic: ratio={sat_stats['zipf_ratio']:.1f}, top10_share={sat_stats['top10_share']*100:.1f}%")
            print(f"    Δ sim:     {sim_novelty - sim_standard:+.4f} ({'✓' if sim_novelty > sim_standard else '✗'})")
    
    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    avg_sim_standard = np.mean(standard_metrics['semantic_sim'])
    avg_sim_novelty = np.mean(novelty_metrics['semantic_sim'])
    
    avg_throughput_standard = np.mean(standard_metrics['throughput'])
    avg_throughput_novelty = np.mean(novelty_metrics['throughput'])
    
    speed_ratio = avg_throughput_novelty / avg_throughput_standard if avg_throughput_standard > 0 else 0
    
    print(f"""
    SEMANTIC SIMILARITY:
      Standard:     {avg_sim_standard:.4f}
      Novelty:      {avg_sim_novelty:.4f}
      Improvement:  {(avg_sim_novelty - avg_sim_standard) / max(avg_sim_standard, 0.001) * 100:+.1f}%
      
    THROUGHPUT (samples/sec):
      Standard:     {avg_throughput_standard:.0f}
      Novelty:      {avg_throughput_novelty:.0f}
      Speed ratio:  {speed_ratio:.2f}x
      
    SUCCESS CRITERIA:
      ✓ Similarity improved: {avg_sim_novelty > avg_sim_standard}
      ✓ Speed degradation < 10%: {speed_ratio > 0.9}
      
    VERDICT: {"✅ NOVELTY WEIGHTING HELPS" if avg_sim_novelty > avg_sim_standard and speed_ratio > 0.9 else
              "⚠️ MIXED RESULTS" if avg_sim_novelty > avg_sim_standard or speed_ratio > 0.9 else
              "❌ NOVELTY WEIGHTING DOES NOT HELP"}
    """)
    
    return {
        'standard': standard_metrics,
        'novelty': novelty_metrics,
        'improvement': avg_sim_novelty - avg_sim_standard,
        'speed_ratio': speed_ratio,
    }


@app.local_entrypoint()
def main():
    print("Running Novelty-Weighted Learning A/B Test on Modal H100...")
    result = test_novelty_weighted_learning.remote()
    print("\nTest complete!")
    return result


if __name__ == "__main__":
    main()
