"""
Conflict Response Pattern Test: Curiosity vs Avoidance

THEORY:
    - Typical brain: conflict → anxiety → premature closure
    - Genius brain: conflict → curiosity → deeper exploration
    
    Conflict = uncertainty about correct answer (top candidates close in score)
    
    The genius response is to EXPLORE MORE when conflict is high,
    not to rush to a decision.

BRAIN ANALOGY:
    ACC (Anterior Cingulate Cortex) detects conflict.
    - Typical: ACC triggers anxiety, system rushes to resolve
    - Genius: ACC triggers curiosity, system explores deeper

PHYSICS FRAMING:
    Like a particle at a saddle point between two valleys.
    - Typical: Small perturbation chooses valley immediately
    - Genius: Explore landscape before committing to a valley
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

app = modal.App("conflict-curiosity-test")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_conflict_curiosity():
    """
    Test whether conflict-as-curiosity (genius) outperforms conflict-as-avoidance (typical).
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
    print("CONFLICT RESPONSE PATTERN TEST")
    print("="*80)
    print("""
    HYPOTHESIS: Conflict-as-curiosity (more exploration) beats conflict-as-avoidance.
    
    TEST:
        A. Standard (fixed Grace iterations regardless of conflict)
        B. Avoidance (fewer iterations when high conflict - rush to decide)
        C. Curiosity (more iterations when high conflict - explore deeper)
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
    # CONFLICT MEASUREMENT AND RESPONSE FUNCTIONS
    # =========================================================================
    
    def measure_conflict(model, context):
        """
        Measure conflict for a context (ACC-like).
        
        Conflict = ratio of second-best to best score.
        High conflict means top candidates are close (uncertain).
        """
        xp = model.xp
        ctx_emb = model.tower._embed_sequence(context)
        
        # Quick route to get satellite
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
        
        # Score against all embeddings
        scores = xp.einsum('ij,vij->v', retrieved, model.tower.embeddings)
        
        # Get top 2
        top2_idx = xp.argpartition(scores, -2)[-2:]
        top2_scores = scores[top2_idx]
        
        if hasattr(top2_scores, 'get'):
            top2_scores = top2_scores.get()
        
        top2_sorted = np.sort(top2_scores)
        second, first = top2_sorted[0], top2_sorted[1]
        
        # Conflict = second / first (1.0 = max conflict, 0 = no conflict)
        conflict = second / (first + 1e-10)
        
        return float(conflict), retrieved, sat_idx
    
    def retrieve_with_iterations(model, context, n_iters):
        """Retrieve with specified number of Grace iterations."""
        xp = model.xp
        ctx_emb = model.tower._embed_sequence(context)
        
        # Apply Grace iterations
        settled = ctx_emb
        for _ in range(n_iters):
            settled = grace_operator(settled, model.basis, xp)
        
        # Route
        basin_key = grace_basin_keys_batch_direct(
            settled[None], model.basis,
            n_iters=1, resolution=GRACE_ROUTING_RESOLUTION, xp=xp
        )[0]
        primes = xp.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53], dtype=xp.int64)
        sat_idx = int((xp.sum(basin_key * primes) % model.tower.n_satellites).get())
        
        sat_memory = model.tower._all_memories[sat_idx]
        retrieved = settled.T @ sat_memory
        
        return retrieved
    
    def standard_response(conflict, base_iters=3):
        """Standard: fixed iterations regardless of conflict."""
        return base_iters
    
    def avoidance_response(conflict, base_iters=3):
        """Avoidance: FEWER iterations when conflict is high (rush to decide)."""
        # High conflict → fewer iterations (min 1)
        reduction = int(conflict * 2)  # 0-2 reduction based on conflict
        return max(1, base_iters - reduction)
    
    def curiosity_response(conflict, base_iters=3):
        """Curiosity: MORE iterations when conflict is high (explore deeper)."""
        # High conflict → more iterations (up to 2x base)
        extra = int(conflict * base_iters)  # 0-3 extra based on conflict
        return base_iters + extra
    
    def evaluate_method(model, response_fn, eval_samples, n_eval=100):
        """Evaluate a conflict response method."""
        xp = model.xp
        similarities = []
        conflicts = []
        iterations_used = []
        
        for ctx, tgt in eval_samples[:n_eval]:
            try:
                # Measure conflict
                conflict, _, _ = measure_conflict(model, ctx)
                conflicts.append(conflict)
                
                # Determine iterations based on response function
                n_iters = response_fn(conflict)
                iterations_used.append(n_iters)
                
                # Retrieve
                retrieved = retrieve_with_iterations(model, ctx, n_iters)
                
                # Score
                tgt_emb = model.tower.embeddings[tgt % model.vocab_size]
                sim = float(frobenius_cosine(retrieved, tgt_emb, xp).get())
                similarities.append(sim)
            except Exception as e:
                raise RuntimeError(f"Similarity computation failed: {e}") from e
        
        if not similarities:
            raise RuntimeError("No similarities computed - all samples failed")
        
        return {
            'sim': float(np.mean(similarities)) if similarities else 0.0,
            'avg_conflict': float(np.mean(conflicts)) if conflicts else 0.0,
            'avg_iters': float(np.mean(iterations_used)) if iterations_used else 0.0,
        }
    
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
        'standard': [],
        'avoidance': [],
        'curiosity': [],
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
            result_standard = evaluate_method(model, standard_response, eval_samples, n_eval=50)
            result_avoidance = evaluate_method(model, avoidance_response, eval_samples, n_eval=50)
            result_curiosity = evaluate_method(model, curiosity_response, eval_samples, n_eval=50)
            
            metrics['standard'].append(result_standard)
            metrics['avoidance'].append(result_avoidance)
            metrics['curiosity'].append(result_curiosity)
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Avg conflict: {result_standard['avg_conflict']:.3f}")
            print(f"    Standard (3 iters):    sim={result_standard['sim']:.4f}")
            print(f"    Avoidance (↓ iters):   sim={result_avoidance['sim']:.4f}, avg_iters={result_avoidance['avg_iters']:.1f}")
            print(f"    Curiosity (↑ iters):   sim={result_curiosity['sim']:.4f}, avg_iters={result_curiosity['avg_iters']:.1f}")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("CONFLICT RESPONSE RESULTS")
    print("="*80)
    
    def get_late_avg(m_list):
        sims = [m['sim'] for m in m_list]
        return float(np.mean(sims[2:])) if len(sims) > 2 else float(np.mean(sims))
    
    late_standard = get_late_avg(metrics['standard'])
    late_avoidance = get_late_avg(metrics['avoidance'])
    late_curiosity = get_late_avg(metrics['curiosity'])
    
    results = {
        'late_stage': {
            'standard': late_standard,
            'avoidance': late_avoidance,
            'curiosity': late_curiosity,
        },
        'improvement': {
            'avoidance_vs_standard': (late_avoidance / late_standard - 1) * 100,
            'curiosity_vs_standard': (late_curiosity / late_standard - 1) * 100,
            'curiosity_vs_avoidance': (late_curiosity / late_avoidance - 1) * 100,
        },
        'curves': {k: [m['sim'] for m in v] for k, v in metrics.items()},
    }
    
    print(f"""
    LATE-STAGE SEMANTIC SIMILARITY:
      Standard (fixed 3 iters):      {late_standard:.4f}
      Avoidance (fewer when conflict): {late_avoidance:.4f} ({results['improvement']['avoidance_vs_standard']:+.2f}%)
      Curiosity (more when conflict):  {late_curiosity:.4f} ({results['improvement']['curiosity_vs_standard']:+.2f}%)
      
    THEORY VALIDATION:
      Curiosity > Avoidance: {late_curiosity > late_avoidance}
      Curiosity > Standard:  {late_curiosity > late_standard}
      
    INTERPRETATION:
      If curiosity > avoidance: Genius response (more exploration) is better
      If avoidance > curiosity: Typical response (quick decision) is better
      
    VERDICT: {'CURIOSITY WINS' if late_curiosity > max(late_standard, late_avoidance) else 'CURIOSITY DOES NOT HELP'}
    """)
    
    # Additional analysis: does curiosity help more when conflict is HIGH?
    print("  Analyzing conflict-dependent effects...")
    
    # Separate high-conflict from low-conflict samples
    high_conflict_results = []
    low_conflict_results = []
    
    for ctx, tgt in eval_samples[:100]:
        try:
            conflict, _, _ = measure_conflict(model, ctx)
            
            # Standard retrieval
            ret_std = retrieve_with_iterations(model, ctx, 3)
            # Curiosity retrieval
            ret_cur = retrieve_with_iterations(model, ctx, curiosity_response(conflict))
            
            tgt_emb = model.tower.embeddings[tgt % model.vocab_size]
            sim_std = float(frobenius_cosine(ret_std, tgt_emb, model.xp).get())
            sim_cur = float(frobenius_cosine(ret_cur, tgt_emb, model.xp).get())
            
            if conflict > 0.5:
                high_conflict_results.append({'std': sim_std, 'cur': sim_cur})
            else:
                low_conflict_results.append({'std': sim_std, 'cur': sim_cur})
        except Exception as e:
            raise RuntimeError(f"Conflict analysis failed: {e}") from e
    
    if high_conflict_results:
        high_std = np.mean([r['std'] for r in high_conflict_results])
        high_cur = np.mean([r['cur'] for r in high_conflict_results])
    else:
        high_std, high_cur = 0, 0
    
    if low_conflict_results:
        low_std = np.mean([r['std'] for r in low_conflict_results])
        low_cur = np.mean([r['cur'] for r in low_conflict_results])
    else:
        low_std, low_cur = 0, 0
    
    results['conflict_breakdown'] = {
        'high_conflict': {
            'n': len(high_conflict_results),
            'standard': high_std,
            'curiosity': high_cur,
            'improvement': (high_cur / high_std - 1) * 100 if high_std > 0 else 0,
        },
        'low_conflict': {
            'n': len(low_conflict_results),
            'standard': low_std,
            'curiosity': low_cur,
            'improvement': (low_cur / low_std - 1) * 100 if low_std > 0 else 0,
        },
    }
    
    print(f"""
    CONFLICT-DEPENDENT ANALYSIS:
      High conflict samples (n={len(high_conflict_results)}):
        Standard:  {high_std:.4f}
        Curiosity: {high_cur:.4f} ({results['conflict_breakdown']['high_conflict']['improvement']:+.2f}%)
        
      Low conflict samples (n={len(low_conflict_results)}):
        Standard:  {low_std:.4f}
        Curiosity: {low_cur:.4f} ({results['conflict_breakdown']['low_conflict']['improvement']:+.2f}%)
      
      KEY INSIGHT:
        Curiosity helps more in high-conflict: {results['conflict_breakdown']['high_conflict']['improvement'] > results['conflict_breakdown']['low_conflict']['improvement']}
    """)
    
    # Save
    results_path = "/checkpoints/conflict_curiosity_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


@app.local_entrypoint()
def main():
    print("Running Conflict-Curiosity Test on Modal H100...")
    result = test_conflict_curiosity.remote()
    print(f"\nCuriosity improvement: {result['improvement']['curiosity_vs_standard']:+.2f}%")
