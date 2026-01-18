"""
Witness Stability Under Exploration Test

THEORY:
    Genius = high exploration + STABLE witness
    Psychosis = high exploration + UNSTABLE witness
    
    The witness is the invariant structure preserved under Grace contraction.
    It represents "identity" or "self-model" in cognitive terms.

BRAIN ANALOGY:
    Stable DMN anchor allows aggressive peripheral plasticity.
    Without stable identity, exploration leads to fragmentation.

PHYSICS FRAMING:
    Like a gyroscope: can explore space while maintaining orientation.
    The axis of rotation (witness) must remain stable.

TEST:
    1. Measure witness churn rate during learning
    2. Correlate with learning quality
    3. Test if enforcing stability improves learning
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

app = modal.App("witness-stability-test")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_witness_stability():
    """
    Test witness stability during learning and its correlation with performance.
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
    from holographic_prod.core.quotient import grace_stability
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast

    # Helper to handle both TowerMemory and MultiLevelTower
    def get_all_memories(tower):
        return getattr(tower, '_all_memories', None) or getattr(tower, '_all_memories', None)

    
    print("="*80)
    print("WITNESS STABILITY UNDER EXPLORATION TEST")
    print("="*80)
    print("""
    HYPOTHESIS: Stable witness + high exploration = genius.
                Unstable witness + high exploration = psychosis.
    
    TEST:
        1. Track witness churn rate during learning
        2. Correlate churn with semantic similarity
        3. Test stability-enforced learning variant
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
    # TWO MODEL VARIANTS
    # =========================================================================
    print("\n  Initializing models...")
    
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
    model_stable = create_model()
    
    print(f"  ✓ Models: {model_standard.tower.n_satellites:,} satellites each")
    
    # =========================================================================
    # WITNESS TRACKING FUNCTIONS
    # =========================================================================
    
    def get_global_witness(model):
        """Extract witness from grand master matrix."""
        xp = model.xp
        grand_master = model.tower.get_grand_master_state()
        # Extract witness from coefficients (scalar = [0], pseudo = [15])
        if hasattr(grand_master, 'get'):
            gm = grand_master.get()
        else:
            gm = np.asarray(grand_master)
        witness = (float(gm[0]), float(gm[15]))
        return witness
    
    def witness_distance(w1, w2, xp):
        """Frobenius distance between two witnesses."""
        if w1 is None or w2 is None:
            return 0.0
        diff = w1 - w2
        if hasattr(diff, 'get'):
            diff = diff.get()
        return float(np.linalg.norm(diff))
    
    def learn_batch_stability_enforced(model, contexts, targets, prev_witness, stability_weight=0.1):
        """
        Learn with stability enforcement.
        
        After standard learning, apply a small correction to preserve witness.
        """
        xp = model.xp
        
        # Standard learning
        model.tower.learn_batch(contexts, targets)
        model.n_patterns += len(contexts)
        
        if prev_witness is None:
            return
        
        # Get current witness
        curr_witness = get_global_witness(model)
        
        # Compute witness drift
        drift = curr_witness - prev_witness
        drift_norm = float(xp.linalg.norm(drift).get())
        
        # If drift is significant, apply correction to grand master
        if drift_norm > 0.01:
            # Apply stabilizing correction (pull back toward previous witness direction)
            # This is a gentle regularization, not a hard constraint
            correction = stability_weight * drift
            
            # Apply correction to each satellite (distributed)
            # Scale by φ⁻¹ for theory-true correction
            scaled_correction = PHI_INV * correction / model.tower.n_satellites
            
            # Add negative correction to counteract drift
            model.tower._all_memories -= scaled_correction[None, :, :]
    
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
    # TRAINING WITH WITNESS TRACKING
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING WITH WITNESS TRACKING")
    print("="*80)
    
    BATCH_SIZE = 2048
    N_BATCHES = 50
    EVAL_EVERY = 5
    
    metrics = {
        'standard': {'sim': [], 'churn': []},
        'stable': {'sim': [], 'churn': []},
    }
    
    prev_witness_standard = None
    prev_witness_stable = None
    
    eval_samples = samples[N_BATCHES * BATCH_SIZE:]
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        batch = samples[start_idx:start_idx + BATCH_SIZE]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # Standard model: normal learning
        model_standard.tower.learn_batch(contexts, targets)
        model_standard.n_patterns += len(contexts)
        
        # Stability-enforced model
        learn_batch_stability_enforced(
            model_stable, contexts, targets,
            prev_witness_stable,
            stability_weight=0.1
        )
        
        # Evaluation
        if (batch_idx + 1) % EVAL_EVERY == 0:
            # Get current witnesses
            curr_witness_standard = get_global_witness(model_standard)
            curr_witness_stable = get_global_witness(model_stable)
            
            # Compute churn
            churn_standard = witness_distance(prev_witness_standard, curr_witness_standard, model_standard.xp)
            churn_stable = witness_distance(prev_witness_stable, curr_witness_stable, model_stable.xp)
            
            # Evaluate similarity
            sim_standard = evaluate_model(model_standard, eval_samples)
            sim_stable = evaluate_model(model_stable, eval_samples)
            
            metrics['standard']['sim'].append(sim_standard)
            metrics['standard']['churn'].append(churn_standard)
            metrics['stable']['sim'].append(sim_stable)
            metrics['stable']['churn'].append(churn_stable)
            
            # Update previous witnesses
            prev_witness_standard = curr_witness_standard
            prev_witness_stable = curr_witness_stable
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Standard: sim={sim_standard:.4f}, churn={churn_standard:.4f}")
            print(f"    Stable:   sim={sim_stable:.4f}, churn={churn_stable:.4f}")
            print(f"    Δ sim:    {sim_stable - sim_standard:+.4f}")
    
    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("WITNESS STABILITY ANALYSIS")
    print("="*80)
    
    # Compute correlation between churn and similarity
    from scipy.stats import pearsonr, spearmanr
    
    all_churns = metrics['standard']['churn'] + metrics['stable']['churn']
    all_sims = metrics['standard']['sim'] + metrics['stable']['sim']
    
    if len(all_churns) > 3:
        pearson_r, pearson_p = pearsonr(all_churns, all_sims)
        spearman_r, spearman_p = spearmanr(all_churns, all_sims)
    else:
        pearson_r, pearson_p = 0, 1
        spearman_r, spearman_p = 0, 1
    
    # Late-stage comparison
    late_sim_standard = np.mean(metrics['standard']['sim'][2:]) if len(metrics['standard']['sim']) > 2 else 0
    late_sim_stable = np.mean(metrics['stable']['sim'][2:]) if len(metrics['stable']['sim']) > 2 else 0
    
    late_churn_standard = np.mean(metrics['standard']['churn'][2:]) if len(metrics['standard']['churn']) > 2 else 0
    late_churn_stable = np.mean(metrics['stable']['churn'][2:]) if len(metrics['stable']['churn']) > 2 else 0
    
    results = {
        'correlation': {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
        },
        'standard': {
            'late_sim': late_sim_standard,
            'late_churn': late_churn_standard,
        },
        'stable': {
            'late_sim': late_sim_stable,
            'late_churn': late_churn_stable,
        },
        'curves': metrics,
    }
    
    print(f"""
    CORRELATION: Witness Churn vs Semantic Similarity
      Pearson r:  {pearson_r:.4f} (p={pearson_p:.4f})
      Spearman r: {spearman_r:.4f} (p={spearman_p:.4f})
      
      Interpretation:
        r < 0: Higher churn → LOWER similarity (instability hurts)
        r > 0: Higher churn → HIGHER similarity (instability helps)
        r ≈ 0: No relationship
    
    LATE-STAGE COMPARISON:
      Standard model:
        - Semantic sim: {late_sim_standard:.4f}
        - Witness churn: {late_churn_standard:.4f}
        
      Stability-enforced model:
        - Semantic sim: {late_sim_stable:.4f} ({(late_sim_stable/late_sim_standard-1)*100:+.2f}%)
        - Witness churn: {late_churn_stable:.4f} ({(late_churn_stable/late_churn_standard-1)*100:+.2f}%)
    
    VERDICT:
      Low churn correlates with better learning: {pearson_r < -0.2}
      Stability enforcement helps: {late_sim_stable > late_sim_standard}
    """)
    
    # Psychosis detection threshold
    avg_churn = np.mean(all_churns)
    std_churn = np.std(all_churns)
    psychosis_threshold = avg_churn + 2 * std_churn
    
    print(f"""
    PSYCHOSIS TRIPWIRE:
      Average churn: {avg_churn:.4f}
      Std churn:     {std_churn:.4f}
      Threshold:     {psychosis_threshold:.4f} (avg + 2σ)
      
      If churn > {psychosis_threshold:.4f}, trigger safety mode:
        - Raise commit threshold
        - Reduce learning rate
        - Freeze witness propagation
    """)
    
    results['psychosis_threshold'] = float(psychosis_threshold)
    
    # Save
    results_path = "/checkpoints/witness_stability_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


@app.local_entrypoint()
def main():
    print("Running Witness Stability Test on Modal H100...")
    result = test_witness_stability.remote()
    print(f"\nCorrelation: r={result['correlation']['pearson_r']:.4f}")
    print(f"Stability helps: {result['stable']['late_sim'] > result['standard']['late_sim']}")
