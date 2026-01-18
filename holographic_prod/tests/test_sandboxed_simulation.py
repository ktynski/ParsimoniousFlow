"""
Sandboxed Simulation Test

THEORY:
    DMN proposals should NOT write to main memory until verified.
    This prevents "psychosis-like" drift from unverified associations.

BRAIN ANALOGY:
    Prefrontal cortex proposes, basal ganglia gates.
    Bad proposals don't corrupt long-term memory.

PHYSICS FRAMING:
    Like a test particle: sample the field without disturbing it.
    Only commit if the result is coherent with existing structure.
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

app = modal.App("sandboxed-simulation-test")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_sandboxed_simulation():
    """
    Test whether sandboxed learning (verify before commit) prevents drift.
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
    print("SANDBOXED SIMULATION TEST")
    print("="*80)
    print("""
    HYPOTHESIS: Sandboxed learning (verify coherence before commit) prevents drift.
    
    TEST:
        A. Standard (all bindings committed directly)
        B. Sandboxed (write to scratch, verify coherence, then commit)
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
    model_sandboxed = create_model()
    
    print(f"  ✓ Models: {model_standard.tower.n_satellites:,} satellites each")
    
    # =========================================================================
    # SANDBOXED LEARNING FUNCTION
    # =========================================================================
    
    class ScratchMemory:
        """Temporary scratch space for sandboxed proposals."""
        def __init__(self, n_satellites, xp):
            self.xp = xp
            self.scratch = xp.zeros((n_satellites, 4, 4), dtype=xp.float32)
            self.active = xp.zeros(n_satellites, dtype=xp.bool_)
        
        def clear(self):
            self.scratch.fill(0)
            self.active.fill(False)
        
        def propose(self, sat_idx, binding):
            """Write binding to scratch (not main memory)."""
            self.scratch[sat_idx] += binding
            self.active[sat_idx] = True
        
        def verify_and_commit(self, main_memory, coherence_threshold=0.3):
            """
            Commit only if binding is coherent with existing memory.
            
            Coherence = Frobenius cosine between binding and existing.
            """
            xp = self.xp
            committed_count = 0
            rejected_count = 0
            
            active_indices = xp.where(self.active)[0]
            
            for idx in active_indices:
                idx_int = int(idx)
                binding = self.scratch[idx_int]
                existing = main_memory[idx_int]
                
                existing_norm = xp.linalg.norm(existing)
                
                # If satellite is empty, always commit
                if existing_norm < 0.01:
                    main_memory[idx_int] += binding
                    committed_count += 1
                    continue
                
                # Compute coherence
                coherence = float(frobenius_cosine(binding, existing, xp).get())
                
                # Commit if coherent OR if binding is small (low-confidence proposal)
                binding_norm = float(xp.linalg.norm(binding).get())
                if coherence > coherence_threshold or binding_norm < 0.1:
                    main_memory[idx_int] += binding
                    committed_count += 1
                else:
                    rejected_count += 1
            
            return committed_count, rejected_count
    
    def learn_batch_sandboxed(model, contexts, targets, scratch, coherence_threshold=0.3):
        """Sandboxed learning: propose to scratch, verify, then commit."""
        xp = model.xp
        tower = model.tower
        
        if not contexts:
            return 0, 0
        
        batch_size = len(contexts)
        
        # Clear scratch
        scratch.clear()
        
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
        bindings = PHI_INV * xp.einsum('bij,bjk->bik', ctx_matrices, tgt_matrices)
        
        # Write to scratch (not main memory)
        for i in range(batch_size):
            sat_idx = int(satellite_indices[i].get())
            scratch.propose(sat_idx, bindings[i])
        
        # Verify and commit
        committed, rejected = scratch.verify_and_commit(
            tower._all_memories, 
            coherence_threshold
        )
        
        # Update counts for committed only
        model.n_patterns += committed
        
        return committed, rejected
    
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
    N_BATCHES = 40
    EVAL_EVERY = 5
    
    scratch = ScratchMemory(model_sandboxed.tower.n_satellites, model_sandboxed.xp)
    
    metrics = {
        'standard': {'sim': [], 'patterns': []},
        'sandboxed': {'sim': [], 'patterns': [], 'rejected': []},
    }
    
    eval_samples = samples[N_BATCHES * BATCH_SIZE:]
    
    total_committed = 0
    total_rejected = 0
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        batch = samples[start_idx:start_idx + BATCH_SIZE]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # Standard: all bindings committed
        model_standard.tower.learn_batch(contexts, targets)
        model_standard.n_patterns += len(contexts)
        
        # Sandboxed: verify before commit
        committed, rejected = learn_batch_sandboxed(
            model_sandboxed, contexts, targets, scratch,
            coherence_threshold=0.3
        )
        total_committed += committed
        total_rejected += rejected
        
        # Evaluation
        if (batch_idx + 1) % EVAL_EVERY == 0:
            sim_standard = evaluate_model(model_standard, eval_samples)
            sim_sandboxed = evaluate_model(model_sandboxed, eval_samples)
            
            metrics['standard']['sim'].append(sim_standard)
            metrics['standard']['patterns'].append(model_standard.n_patterns)
            
            metrics['sandboxed']['sim'].append(sim_sandboxed)
            metrics['sandboxed']['patterns'].append(model_sandboxed.n_patterns)
            metrics['sandboxed']['rejected'].append(total_rejected)
            
            reject_rate = total_rejected / (total_committed + total_rejected + 1) * 100
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Standard:  sim={sim_standard:.4f}, patterns={model_standard.n_patterns:,}")
            print(f"    Sandboxed: sim={sim_sandboxed:.4f} ({(sim_sandboxed/sim_standard-1)*100:+.1f}%), patterns={model_sandboxed.n_patterns:,}")
            print(f"    Rejected: {total_rejected:,} ({reject_rate:.1f}%)")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("SANDBOXED SIMULATION RESULTS")
    print("="*80)
    
    late_sim_standard = float(np.mean(metrics['standard']['sim'][2:])) if len(metrics['standard']['sim']) > 2 else float(np.mean(metrics['standard']['sim']))
    late_sim_sandboxed = float(np.mean(metrics['sandboxed']['sim'][2:])) if len(metrics['sandboxed']['sim']) > 2 else float(np.mean(metrics['sandboxed']['sim']))
    
    final_reject_rate = total_rejected / (total_committed + total_rejected + 1) * 100
    
    results = {
        'late_stage': {
            'standard': late_sim_standard,
            'sandboxed': late_sim_sandboxed,
        },
        'improvement': (late_sim_sandboxed / late_sim_standard - 1) * 100,
        'patterns': {
            'standard': model_standard.n_patterns,
            'sandboxed': model_sandboxed.n_patterns,
        },
        'rejection': {
            'total_rejected': total_rejected,
            'total_committed': total_committed,
            'rejection_rate': final_reject_rate,
        },
        'curves': metrics,
    }
    
    print(f"""
    LATE-STAGE SEMANTIC SIMILARITY:
      Standard:  {late_sim_standard:.4f}
      Sandboxed: {late_sim_sandboxed:.4f} ({results['improvement']:+.2f}%)
      
    PATTERNS LEARNED:
      Standard:  {model_standard.n_patterns:,}
      Sandboxed: {model_sandboxed.n_patterns:,} ({total_rejected:,} rejected, {final_reject_rate:.1f}%)
      
    THEORY VALIDATION:
      Sandboxing improves quality: {late_sim_sandboxed > late_sim_standard}
      Some proposals rejected: {total_rejected > 0}
      
    INTERPRETATION:
      If sandboxed > standard with rejections: Verification prevents drift
      If sandboxed < standard: Verification is too aggressive
      
    VERDICT: {'SANDBOXING HELPS' if late_sim_sandboxed > late_sim_standard else 'SANDBOXING DOES NOT HELP'}
    """)
    
    # Save
    results_path = "/checkpoints/sandboxed_simulation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


@app.local_entrypoint()
def main():
    print("Running Sandboxed Simulation Test on Modal H100...")
    result = test_sandboxed_simulation.remote()
    print(f"\nSandboxing improvement: {result['improvement']:+.2f}%")
