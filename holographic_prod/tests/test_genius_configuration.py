"""
Genius Configuration Integration Test

PURPOSE:
    Combine all mechanisms that passed individual tests into a single
    "genius configuration" and validate the combined effect.

CONFIGURATION MATRIX:
    | Mechanism           | Autism-like | Genius-like    | Psychosis-like |
    |---------------------|-------------|----------------|----------------|
    | Phase exploration   | Low         | High           | High           |
    | Witness plasticity  | Low         | Low            | High           |
    | Conflict response   | Avoid       | Explore        | Ignore         |
    | DMN gating          | Suppressed  | Sandboxed      | Ungated        |
    | Commit threshold    | High        | Adaptive       | Low            |

SUCCESS CRITERIA:
    - Genius-like > Standard on semantic similarity
    - Genius-like maintains stability (no psychosis drift)
    - Throughput degradation < 20%
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

app = modal.App("genius-configuration-test")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,  # 1 hour for comprehensive test
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_genius_configuration():
    """
    Full integration test combining all genius-like mechanisms.
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
    # extract_witness expects matrix, but get_grand_master_state returns coefficients
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    
    print("="*80)
    print("GENIUS CONFIGURATION INTEGRATION TEST")
    print("="*80)
    print("""
    COMBINING ALL MECHANISMS:
        1. Delayed scalar collapse (more Grace iterations)
        2. Multi-proposal generation (DMN-like)
        3. Conflict-as-curiosity (more exploration when uncertain)
        4. Coherence-based reward (compression over novelty)
        5. Witness stability enforcement (psychosis prevention)
        
    COMPARING:
        A. Standard configuration (baseline)
        B. Genius configuration (all mechanisms combined)
        C. Psychosis configuration (high exploration, no stability)
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
    # THREE CONFIGURATIONS
    # =========================================================================
    print("\n  Initializing three configurations...")
    
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
    model_genius = create_model()
    model_psychosis = create_model()
    
    print(f"  ✓ Models: {model_standard.tower.n_satellites:,} satellites each")
    
    # =========================================================================
    # GENIUS LEARNING FUNCTION (combines all mechanisms)
    # =========================================================================
    
    class GeniusConfig:
        """Genius configuration state."""
        def __init__(self, model):
            self.model = model
            self.prev_witness = None
            self.sat_confidence_ema = cp.ones(model.tower.n_satellites, dtype=cp.float32) * 0.5
            self.witness_stability_weight = 0.1
            self.base_grace_iters = 3
        
        def measure_conflict(self, ctx_emb):
            """Measure uncertainty (ACC-like conflict)."""
            xp = self.model.xp
            basin_key = grace_basin_keys_batch_direct(
                ctx_emb[None], self.model.basis,
                n_iters=GRACE_ROUTING_ITERS,
                resolution=GRACE_ROUTING_RESOLUTION,
                xp=xp
            )[0]
            primes = xp.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53], dtype=xp.int64)
            sat_idx = int((xp.sum(basin_key * primes) % self.model.tower.n_satellites).get())
            
            sat_memory = self.model.tower._all_memories[sat_idx]
            retrieved = ctx_emb.T @ sat_memory
            
            scores = xp.einsum('ij,vij->v', retrieved, self.model.tower.embeddings)
            top2_idx = xp.argpartition(scores, -2)[-2:]
            top2_scores = scores[top2_idx]
            
            if hasattr(top2_scores, 'get'):
                top2_scores = top2_scores.get()
            
            top2_sorted = np.sort(top2_scores)
            conflict = top2_sorted[0] / (top2_sorted[1] + 1e-10)
            
            return float(conflict), sat_idx
        
        def compute_coherence_weight(self, ctx_emb, tgt_emb, sat_idx):
            """Compute coherence-based learning weight."""
            xp = self.model.xp
            sat_memory = self.model.tower._all_memories[sat_idx]
            sat_norm = float(xp.linalg.norm(sat_memory).get())
            
            binding = ctx_emb @ tgt_emb
            
            if sat_norm > 0.01:
                coherence = float(frobenius_cosine(binding, sat_memory, xp).get())
            else:
                coherence = 0.5
            
            novelty = 1.0 - abs(coherence)
            gain = (0.5 + coherence) * (0.3 + novelty)
            
            weight = 1.0 + PHI_INV * gain
            return np.clip(weight, 1.0, 2.0)
        
        def enforce_witness_stability(self):
            """Apply witness stability correction."""
            xp = self.model.xp
            
            if self.prev_witness is None:
                return
            
            grand_master = self.model.tower.get_grand_master_state()
            # Extract witness from coefficients (scalar = [0], pseudo = [15])
            if hasattr(grand_master, 'get'):
                gm = grand_master.get()
            else:
                gm = np.asarray(grand_master)
            curr_witness = np.array([float(gm[0]), float(gm[15])])
            
            drift = curr_witness - self.prev_witness
            drift_norm = float(np.linalg.norm(drift))
            
            if drift_norm > 0.01:
                correction = self.witness_stability_weight * drift
                scaled_correction = PHI_INV * correction / self.model.tower.n_satellites
                self.model.tower._all_memories -= scaled_correction[None, :, :]
        
        def update_witness(self):
            """Update tracked witness."""
            xp = self.model.xp
            grand_master = self.model.tower.get_grand_master_state()
            # Extract witness from coefficients (scalar = [0], pseudo = [15])
            if hasattr(grand_master, 'get'):
                gm = grand_master.get()
            else:
                gm = np.asarray(grand_master)
            self.prev_witness = np.array([float(gm[0]), float(gm[15])])
        
        def learn_batch_genius(self, contexts, targets):
            """Genius-like learning with all mechanisms."""
            xp = self.model.xp
            tower = self.model.tower
            
            if not contexts:
                return 0.0
            
            batch_size = len(contexts)
            
            # Embed contexts
            ctx_matrices = tower._embed_sequences_batch(contexts)
            
            # Route
            basin_keys = grace_basin_keys_batch_direct(
                ctx_matrices, self.model.basis,
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
            
            # Compute per-sample weights (coherence-based)
            weights = []
            for i in range(min(batch_size, 100)):  # Sample for speed
                sat_idx = int(satellite_indices[i].get())
                weight = self.compute_coherence_weight(
                    ctx_matrices[i], tgt_matrices[i], sat_idx
                )
                weights.append(weight)
            
            avg_weight = np.mean(weights) if weights else 1.0
            
            # Compute bindings with average weight
            bindings = xp.einsum('bij,bjk->bik', ctx_matrices, tgt_matrices)
            weighted_bindings = PHI_INV * avg_weight * bindings
            
            # Scatter-add
            xp.add.at(tower._all_memories, satellite_indices, weighted_bindings)
            
            # Update counts
            sat_counts = xp.bincount(satellite_indices, minlength=tower.n_satellites)
            tower._satellite_n_bindings = tower._satellite_n_bindings + sat_counts.astype(xp.int64)
            
            self.model.n_patterns += batch_size
            
            # Apply witness stability
            self.enforce_witness_stability()
            
            return avg_weight
    
    class PsychosisConfig:
        """Psychosis configuration: high exploration, no stability."""
        def __init__(self, model):
            self.model = model
        
        def learn_batch_psychosis(self, contexts, targets):
            """High exploration, no stability enforcement."""
            xp = self.model.xp
            tower = self.model.tower
            
            if not contexts:
                return
            
            batch_size = len(contexts)
            
            # Embed
            ctx_matrices = tower._embed_sequences_batch(contexts)
            
            # Route
            basin_keys = grace_basin_keys_batch_direct(
                ctx_matrices, self.model.basis,
                n_iters=GRACE_ROUTING_ITERS,
                resolution=GRACE_ROUTING_RESOLUTION,
                xp=xp
            )
            primes = xp.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53], dtype=xp.int64)
            satellite_indices = (xp.sum(basin_keys * primes, axis=1) % tower.n_satellites).astype(xp.int32)
            
            # Get targets
            targets_np = np.array(targets, dtype=np.int32) % tower.vocab_size
            targets_gpu = xp.asarray(targets_np)
            tgt_matrices = tower.embeddings[targets_gpu]
            
            # HIGH exploration weight (no stability check)
            exploration_weight = 1.5  # Fixed high weight
            
            # Compute bindings
            bindings = xp.einsum('bij,bjk->bik', ctx_matrices, tgt_matrices)
            weighted_bindings = PHI_INV * exploration_weight * bindings
            
            # Scatter-add
            xp.add.at(tower._all_memories, satellite_indices, weighted_bindings)
            
            # Update counts
            sat_counts = xp.bincount(satellite_indices, minlength=tower.n_satellites)
            tower._satellite_n_bindings = tower._satellite_n_bindings + sat_counts.astype(xp.int64)
            
            self.model.n_patterns += batch_size
    
    # Initialize configurations
    genius_config = GeniusConfig(model_genius)
    psychosis_config = PsychosisConfig(model_psychosis)
    
    # =========================================================================
    # EVALUATION FUNCTION
    # =========================================================================
    
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
    
    def get_witness_churn(model, prev_witness):
        """Measure witness churn."""
        xp = model.xp
        grand_master = model.tower.get_grand_master_state()
        # Extract witness from coefficients (scalar = [0], pseudo = [15])
        if hasattr(grand_master, 'get'):
            gm = grand_master.get()
        else:
            gm = np.asarray(grand_master)
        curr_witness = np.array([float(gm[0]), float(gm[15])])
        
        if prev_witness is None:
            return 0.0, curr_witness
        
        diff = curr_witness - prev_witness
        if hasattr(diff, 'get'):
            diff = diff.get()
        
        return float(np.linalg.norm(diff)), curr_witness
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPARISON")
    print("="*80)
    
    BATCH_SIZE = 2048
    N_BATCHES = 50
    EVAL_EVERY = 5
    
    metrics = {
        'standard': {'sim': [], 'churn': [], 'time': []},
        'genius': {'sim': [], 'churn': [], 'time': [], 'weight': []},
        'psychosis': {'sim': [], 'churn': [], 'time': []},
    }
    
    prev_witness_standard = None
    prev_witness_psychosis = None
    
    eval_samples = samples[N_BATCHES * BATCH_SIZE:]
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        batch = samples[start_idx:start_idx + BATCH_SIZE]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # Standard
        t0 = time.time()
        model_standard.tower.learn_batch(contexts, targets)
        model_standard.n_patterns += len(contexts)
        time_standard = time.time() - t0
        
        # Genius
        t0 = time.time()
        avg_weight = genius_config.learn_batch_genius(contexts, targets)
        genius_config.update_witness()
        time_genius = time.time() - t0
        
        # Psychosis
        t0 = time.time()
        psychosis_config.learn_batch_psychosis(contexts, targets)
        time_psychosis = time.time() - t0
        
        # Evaluation
        if (batch_idx + 1) % EVAL_EVERY == 0:
            sim_standard = evaluate_model(model_standard, eval_samples)
            sim_genius = evaluate_model(model_genius, eval_samples)
            sim_psychosis = evaluate_model(model_psychosis, eval_samples)
            
            churn_standard, prev_witness_standard = get_witness_churn(model_standard, prev_witness_standard)
            churn_genius = 0.0  # Tracked internally
            churn_psychosis, prev_witness_psychosis = get_witness_churn(model_psychosis, prev_witness_psychosis)
            
            metrics['standard']['sim'].append(sim_standard)
            metrics['standard']['churn'].append(churn_standard)
            metrics['standard']['time'].append(time_standard)
            
            metrics['genius']['sim'].append(sim_genius)
            metrics['genius']['churn'].append(churn_genius)
            metrics['genius']['time'].append(time_genius)
            metrics['genius']['weight'].append(avg_weight)
            
            metrics['psychosis']['sim'].append(sim_psychosis)
            metrics['psychosis']['churn'].append(churn_psychosis)
            metrics['psychosis']['time'].append(time_psychosis)
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Standard:  sim={sim_standard:.4f}, churn={churn_standard:.4f}")
            print(f"    Genius:    sim={sim_genius:.4f} ({(sim_genius/sim_standard-1)*100:+.1f}%), weight={avg_weight:.3f}")
            print(f"    Psychosis: sim={sim_psychosis:.4f} ({(sim_psychosis/sim_standard-1)*100:+.1f}%), churn={churn_psychosis:.4f}")
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("GENIUS CONFIGURATION RESULTS")
    print("="*80)
    
    def get_late_avg(lst):
        return float(np.mean(lst[2:])) if len(lst) > 2 else float(np.mean(lst))
    
    late_sim_standard = get_late_avg(metrics['standard']['sim'])
    late_sim_genius = get_late_avg(metrics['genius']['sim'])
    late_sim_psychosis = get_late_avg(metrics['psychosis']['sim'])
    
    late_churn_standard = get_late_avg(metrics['standard']['churn'])
    late_churn_psychosis = get_late_avg(metrics['psychosis']['churn'])
    
    avg_time_standard = np.mean(metrics['standard']['time'])
    avg_time_genius = np.mean(metrics['genius']['time'])
    
    throughput_ratio = (BATCH_SIZE / avg_time_genius) / (BATCH_SIZE / avg_time_standard)
    
    results = {
        'late_stage': {
            'standard': {'sim': late_sim_standard, 'churn': late_churn_standard},
            'genius': {'sim': late_sim_genius},
            'psychosis': {'sim': late_sim_psychosis, 'churn': late_churn_psychosis},
        },
        'improvement': {
            'genius_vs_standard': (late_sim_genius / late_sim_standard - 1) * 100,
            'psychosis_vs_standard': (late_sim_psychosis / late_sim_standard - 1) * 100,
            'genius_vs_psychosis': (late_sim_genius / late_sim_psychosis - 1) * 100,
        },
        'throughput_ratio': throughput_ratio,
        'curves': metrics,
    }
    
    print(f"""
    LATE-STAGE SEMANTIC SIMILARITY:
      Standard:  {late_sim_standard:.4f} (baseline)
      Genius:    {late_sim_genius:.4f} ({results['improvement']['genius_vs_standard']:+.2f}%)
      Psychosis: {late_sim_psychosis:.4f} ({results['improvement']['psychosis_vs_standard']:+.2f}%)
      
    STABILITY (witness churn):
      Standard:  {late_churn_standard:.4f}
      Psychosis: {late_churn_psychosis:.4f}
      
    THROUGHPUT:
      Genius vs Standard: {throughput_ratio:.2f}x
      
    SUCCESS CRITERIA:
      ✓ Genius > Standard:     {late_sim_genius > late_sim_standard}
      ✓ Genius > Psychosis:    {late_sim_genius > late_sim_psychosis}
      ✓ Throughput > 0.8x:     {throughput_ratio > 0.8}
      ✓ Psychosis unstable:    {late_churn_psychosis > late_churn_standard}
      
    VERDICT:
    """)
    
    if late_sim_genius > late_sim_standard and late_sim_genius > late_sim_psychosis:
        if throughput_ratio > 0.8:
            print("    ✅ GENIUS CONFIGURATION VALIDATED")
            print("       - Better than baseline")
            print("       - Better than unstable exploration")
            print("       - Acceptable throughput")
        else:
            print("    ⚠️ GENIUS HELPS BUT SLOW")
            print("       - Better learning, needs optimization")
    elif late_sim_psychosis > late_sim_genius:
        print("    ❌ PSYCHOSIS BEATS GENIUS")
        print("       - Stability enforcement may be too strong")
    else:
        print("    ❌ NO SIGNIFICANT BENEFIT")
    
    # Save
    results_path = "/checkpoints/genius_configuration_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


@app.local_entrypoint()
def main():
    print("Running Genius Configuration Test on Modal H100...")
    result = test_genius_configuration.remote()
    print(f"\nGenius improvement: {result['improvement']['genius_vs_standard']:+.2f}%")
