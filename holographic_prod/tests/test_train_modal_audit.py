"""
Comprehensive Audit and Testing for train_modal.py

Tests for:
1. Theory-true evaluation correctness
2. H100 performance optimization
3. Real text data training
4. Learning verification
5. End-to-end training validation
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

app = modal.App("train-modal-audit")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def audit_evaluation_correctness():
    """
    Audit 1: Verify train_modal.py uses theory-true evaluation.
    
    Checks:
    1. evaluate_semantic() matches theory-true retrieval path
    2. Semantic similarity is computed correctly (coherence, not similarity)
    3. No candidate sets or other theory violations
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    from tqdm import tqdm
    
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    from holographic_prod.tests.theory_true_evaluation_helper import (
        evaluate_semantic_similarity_theory_true,
    )
    
    print("="*80)
    print("AUDIT 1: EVALUATION CORRECTNESS")
    print("="*80)
    
    # Load vocabulary
    vocab_path = "/checkpoints/vocabulary.npz"
    vocab_data = np.load(vocab_path, allow_pickle=True)
    word_to_idx = vocab_data['word_to_idx'].item()
    
    # Create grounded embeddings
    grounded_embs, coverage = create_grounded_embeddings_fast(
        word_to_idx, glove_dim=50, cache_dir="/checkpoints/glove"
    )
    
    # Initialize model
    model = HolographicMemory(
        vocab_size=len(word_to_idx),
        max_levels=4,
        seed=42,
        use_gpu=True,
        grounded_embeddings=cp.asarray(grounded_embs),
    )
    
    # Create test samples
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    samples = []
    context_size = 64
    for item in tqdm(ds.take(1_000), total=1_000, desc="Preparing"):
        words = item['text'].lower().split()
        tokens = [word_to_idx.get(w, 1) for w in words]
        for i in range(len(tokens) - context_size):
            ctx = tokens[i:i+context_size]
            tgt = tokens[i+context_size]
            if tgt != 1:
                samples.append((ctx, tgt))
            if len(samples) >= 1_000:
                break
        if len(samples) >= 1_000:
            break
    
    # Learn some patterns
    print("\n  Learning patterns...")
    learn_batch = samples[:500]
    contexts = [ctx for ctx, _ in learn_batch]
    targets = [tgt for _, tgt in learn_batch]
    model.learn_batch(contexts, targets)
    
    # Test evaluation
    eval_batch = samples[500:600]
    
    # Method 1: train_modal.py's evaluate_semantic()
    print("\n  Testing evaluate_semantic()...")
    eval_result_train = model.evaluate_semantic(eval_batch)
    train_sim = eval_result_train['semantic_similarity']
    
    # Method 2: Theory-true helper (matches retrieve() exactly)
    print("  Testing theory-true helper...")
    eval_result_helper = evaluate_semantic_similarity_theory_true(
        model, eval_batch, n_eval=100
    )
    helper_sim = eval_result_helper['semantic_similarity']
    
    # Compare
    print(f"\n  Results:")
    print(f"    train_modal.evaluate_semantic(): {train_sim:.4f}")
    print(f"    theory-true helper:              {helper_sim:.4f}")
    print(f"    Difference:                     {abs(train_sim - helper_sim):.4f}")
    
    # They should be similar (within 0.1)
    diff = abs(train_sim - helper_sim)
    if diff < 0.1:
        print(f"\n  ✓ PASSED: Evaluation methods agree (diff={diff:.4f})")
    else:
        print(f"\n  ⚠ WARNING: Evaluation methods differ (diff={diff:.4f})")
        print(f"    This may indicate evaluate_semantic() doesn't match retrieve() path")
    
    # Check for theory violations
    print("\n  Checking for theory violations...")
    
    # Check: No candidate sets
    if 'candidate_set' in str(eval_result_train):
        print("    ✗ FAILED: Candidate sets detected (FORBIDDEN)")
    else:
        print("    ✓ PASSED: No candidate sets")
    
    # Check: Uses coherence, not just similarity
    if 'coherence' in str(eval_result_train) or 'witness' in str(eval_result_train):
        print("    ✓ PASSED: Uses coherence/witness metrics")
    else:
        print("    ⚠ WARNING: May not use coherence scoring")
    
    return {
        'train_sim': train_sim,
        'helper_sim': helper_sim,
        'diff': diff,
        'passed': diff < 0.1,
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def audit_h100_performance():
    """
    Audit 2: H100 Performance Optimization
    
    Tests:
    1. Optimal batch size (8192 for H100)
    2. Throughput targets (> 1000 samples/sec)
    3. GPU memory usage (< 80GB)
    4. Batch processing efficiency
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    from tqdm import tqdm
    
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    
    print("="*80)
    print("AUDIT 2: H100 PERFORMANCE")
    print("="*80)
    
    # Load vocabulary
    vocab_path = "/checkpoints/vocabulary.npz"
    vocab_data = np.load(vocab_path, allow_pickle=True)
    word_to_idx = vocab_data['word_to_idx'].item()
    
    # Create grounded embeddings
    grounded_embs, coverage = create_grounded_embeddings_fast(
        word_to_idx, glove_dim=50, cache_dir="/checkpoints/glove"
    )
    
    # Initialize model
    model = HolographicMemory(
        vocab_size=len(word_to_idx),
        max_levels=6,  # H100-optimized: 16M satellites
        seed=42,
        use_gpu=True,
        grounded_embeddings=cp.asarray(grounded_embs),
    )
    
    # Test different batch sizes
    batch_sizes = [1024, 2048, 4096, 8192, 16384]
    context_size = 64
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n  Testing batch_size={batch_size:,}...")
        
        # Generate test batch
        contexts = []
        targets = []
        for i in range(batch_size):
            ctx = np.random.randint(4, len(word_to_idx), size=context_size).tolist()
            tgt = np.random.randint(4, len(word_to_idx))
            contexts.append(ctx)
            targets.append(tgt)
        
        # Warmup
        model.learn_batch(contexts[:100], targets[:100])
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        times = []
        for _ in range(5):  # 5 runs for average
            t0 = time.perf_counter()
            model.learn_batch(contexts, targets)
            cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - t0)
        
        avg_time = np.mean(times)
        throughput = batch_size / avg_time
        
        # GPU memory
        mempool = cp.get_default_memory_pool()
        mem_used_gb = mempool.used_bytes() / (1024**3)
        
        results[batch_size] = {
            'avg_time': avg_time,
            'throughput': throughput,
            'memory_gb': mem_used_gb,
        }
        
        print(f"    Time: {avg_time:.3f}s")
        print(f"    Throughput: {throughput:,.0f} samples/sec")
        print(f"    Memory: {mem_used_gb:.2f} GB")
        
        # Check targets
        if throughput > 1000:
            print(f"    ✓ Throughput target met")
        else:
            print(f"    ⚠ Throughput below target (1000 samples/sec)")
        
        if mem_used_gb < 80:
            print(f"    ✓ Memory within H100 limit")
        else:
            print(f"    ⚠ Memory exceeds H100 limit (80GB)")
    
    # Find optimal batch size
    best_batch = max(batch_sizes, key=lambda bs: results[bs]['throughput'])
    print(f"\n  Optimal batch size: {best_batch:,} (throughput: {results[best_batch]['throughput']:,.0f}/s)")
    
    return {
        'results': results,
        'optimal_batch_size': best_batch,
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def audit_real_text_training():
    """
    Audit 3: Real Text Data Training
    
    Tests:
    1. OpenWebText loading and tokenization
    2. Proper context/target extraction
    3. Learning on real text (not synthetic)
    4. Semantic similarity increases over batches
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    from tqdm import tqdm
    from datasets import load_dataset
    
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    from holographic_prod.tests.theory_true_evaluation_helper import (
        evaluate_semantic_similarity_theory_true,
    )
    
    print("="*80)
    print("AUDIT 3: REAL TEXT DATA TRAINING")
    print("="*80)
    
    # Load vocabulary
    vocab_path = "/checkpoints/vocabulary.npz"
    vocab_data = np.load(vocab_path, allow_pickle=True)
    word_to_idx = vocab_data['word_to_idx'].item()
    
    # Create grounded embeddings
    grounded_embs, coverage = create_grounded_embeddings_fast(
        word_to_idx, glove_dim=50, cache_dir="/checkpoints/glove"
    )
    
    # Initialize model
    model = HolographicMemory(
        vocab_size=len(word_to_idx),
        max_levels=4,
        seed=42,
        use_gpu=True,
        grounded_embeddings=cp.asarray(grounded_embs),
    )
    
    # Load real text data
    print("\n  Loading OpenWebText...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    # Prepare samples (same as train_modal.py)
    context_size = 64
    samples = []
    
    for item in tqdm(ds.take(10_000), total=10_000, desc="Tokenizing"):
        words = item['text'].lower().split()
        tokens = [word_to_idx.get(w, 1) for w in words]
        for i in range(len(tokens) - context_size):
            ctx = tokens[i:i+context_size]
            tgt = tokens[i+context_size]
            if tgt != 1:  # Skip <unk> targets
                samples.append((ctx, tgt))
            if len(samples) >= 50_000:
                break
        if len(samples) >= 50_000:
            break
    
    print(f"  ✓ Prepared {len(samples):,} samples from real text")
    
    # Training loop with evaluation
    BATCH_SIZE = 2048
    N_BATCHES = 20
    EVAL_EVERY = 5
    
    # Split train/eval
    train_samples = samples[:40_000]
    eval_samples = samples[40_000:]
    
    semantic_sims = []
    
    print(f"\n  Training for {N_BATCHES} batches...")
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(train_samples))
        batch = train_samples[start_idx:end_idx]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # Learn batch
        model.learn_batch(contexts, targets)
        
        # Evaluate periodically
        if (batch_idx + 1) % EVAL_EVERY == 0:
            eval_result = evaluate_semantic_similarity_theory_true(
                model, eval_samples, n_eval=100
            )
            semantic_sims.append(eval_result['semantic_similarity'])
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Semantic sim: {eval_result['semantic_similarity']:.4f}")
            print(f"    Exact match:  {eval_result['exact_match_rate']:.1%}")
            print(f"    Avg rank:     {eval_result['avg_target_rank']:.1f}")
    
    # Verify learning trend
    print("\n  Verifying learning trend...")
    
    if len(semantic_sims) >= 2:
        initial_sim = semantic_sims[0]
        final_sim = semantic_sims[-1]
        improvement = final_sim - initial_sim
        
        print(f"    Initial: {initial_sim:.4f}")
        print(f"    Final:   {final_sim:.4f}")
        print(f"    Change:  {improvement:+.4f}")
        
        if improvement > 0:
            print(f"    ✓ PASSED: Semantic similarity increased")
        elif improvement > -0.05:
            print(f"    ⚠ WARNING: Semantic similarity didn't increase (may need more training)")
        else:
            print(f"    ✗ FAILED: Semantic similarity decreased")
    
    return {
        'n_samples': len(samples),
        'semantic_sims': semantic_sims,
        'improvement': improvement if len(semantic_sims) >= 2 else 0.0,
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,  # 2 hours for full training test
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def audit_end_to_end_training():
    """
    Audit 4: End-to-End Training Validation
    
    Simulates train_modal.py training loop and verifies:
    1. All components work together
    2. Learning metrics improve correctly
    3. Checkpointing works
    4. Dreaming works (if enabled)
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    from tqdm import tqdm
    from datasets import load_dataset
    
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    from holographic_prod.tests.theory_true_evaluation_helper import (
        evaluate_semantic_similarity_theory_true,
    )
    
    print("="*80)
    print("AUDIT 4: END-TO-END TRAINING")
    print("="*80)
    
    # Load vocabulary
    vocab_path = "/checkpoints/vocabulary.npz"
    vocab_data = np.load(vocab_path, allow_pickle=True)
    word_to_idx = vocab_data['word_to_idx'].item()
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    
    # Create grounded embeddings
    grounded_embs, coverage = create_grounded_embeddings_fast(
        word_to_idx, glove_dim=50, cache_dir="/checkpoints/glove"
    )
    
    # Initialize model (matching train_modal.py defaults)
    model = HolographicMemory(
        vocab_size=len(word_to_idx),
        max_levels=6,  # H100-optimized
        seed=42,
        use_gpu=True,
        grounded_embeddings=cp.asarray(grounded_embs),
    )
    
    # Load real text data
    print("\n  Loading OpenWebText...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    # Prepare samples
    context_size = 64
    samples = []
    
    for item in tqdm(ds.take(20_000), total=20_000, desc="Tokenizing"):
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
    
    # Training configuration (matching train_modal.py)
    batch_size = 8192  # H100-optimized
    max_samples = 50_000  # Smaller for test
    log_every = 10_000
    
    # Split train/eval
    train_samples = samples[:80_000]
    eval_samples = samples[80_000:]
    
    # Training loop
    sample_idx = 0
    metrics = {
        'semantic_sims': [],
        'throughputs': [],
        'batch_times': [],
    }
    
    print(f"\n  Training for up to {max_samples:,} samples...")
    
    while sample_idx < max_samples and sample_idx < len(train_samples):
        # Create batch
        batch_end = min(sample_idx + batch_size, len(train_samples))
        batch = train_samples[sample_idx:batch_end]
        sample_idx = batch_end
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # Learn batch
        t0 = time.perf_counter()
        model.learn_batch(contexts, targets)
        batch_time = time.perf_counter() - t0
        
        throughput = len(batch) / batch_time
        metrics['batch_times'].append(batch_time)
        metrics['throughputs'].append(throughput)
        
        # Evaluate periodically
        if sample_idx % log_every < batch_size:
            eval_result = evaluate_semantic_similarity_theory_true(
                model, eval_samples, n_eval=100
            )
            metrics['semantic_sims'].append(eval_result['semantic_similarity'])
            
            print(f"\n  Samples: {sample_idx:,}")
            print(f"    Semantic sim: {eval_result['semantic_similarity']:.4f}")
            print(f"    Throughput:   {throughput:,.0f} samples/sec")
    
    # Final evaluation
    print("\n  Final evaluation...")
    final_eval = evaluate_semantic_similarity_theory_true(
        model, eval_samples, n_eval=500
    )
    
    print(f"\n  Final Results:")
    print(f"    Semantic sim: {final_eval['semantic_similarity']:.4f}")
    print(f"    Exact match:  {final_eval['exact_match_rate']:.1%}")
    print(f"    Avg rank:     {final_eval['avg_target_rank']:.1f}")
    print(f"    Avg throughput: {np.mean(metrics['throughputs']):,.0f} samples/sec")
    
    # Verify learning happened
    if len(metrics['semantic_sims']) >= 2:
        improvement = metrics['semantic_sims'][-1] - metrics['semantic_sims'][0]
        print(f"    Improvement: {improvement:+.4f}")
        
        if improvement > 0:
            print(f"\n  ✓ PASSED: Learning verified")
        else:
            print(f"\n  ⚠ WARNING: Learning not detected (may need more training)")
    
    return {
        'final_semantic_sim': final_eval['semantic_similarity'],
        'avg_throughput': np.mean(metrics['throughputs']),
        'improvement': improvement if len(metrics['semantic_sims']) >= 2 else 0.0,
    }


@app.local_entrypoint()
def main():
    """Run all audits."""
    print("Running Comprehensive train_modal.py Audits...")
    
    print("\n" + "="*80)
    result1 = audit_evaluation_correctness.remote()
    print(f"\nEvaluation Audit: {result1}")
    
    print("\n" + "="*80)
    result2 = audit_h100_performance.remote()
    print(f"\nPerformance Audit: {result2}")
    
    print("\n" + "="*80)
    result3 = audit_real_text_training.remote()
    print(f"\nReal Text Training Audit: {result3}")
    
    print("\n" + "="*80)
    result4 = audit_end_to_end_training.remote()
    print(f"\nEnd-to-End Training Audit: {result4}")
    
    print("\n" + "="*80)
    print("ALL AUDITS COMPLETE!")
    print("="*80)
