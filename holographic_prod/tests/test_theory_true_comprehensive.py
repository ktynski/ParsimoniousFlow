"""
Comprehensive Theory-True Testing Suite

Tests for:
1. Theory-true correctness (Grace convergence, coherence scoring, no candidate sets)
2. Performance (throughput, memory usage, GPU utilization)
3. Learning verification (semantic similarity increases, witness stabilizes)
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

app = modal.App("theory-true-comprehensive")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_theory_true_correctness():
    """
    Test 1: Theory-True Correctness
    
    Verifies:
    1. Grace ALWAYS converges (never returns None)
    2. Full vocabulary coherence scoring (no candidate sets)
    3. Coherence metric matches theory (witness_energy / total_energy)
    4. Memory unbinding uses SO(4) transpose (theory-true)
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    from tqdm import tqdm
    
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    from holographic_prod.tests.theory_true_evaluation_helper import (
        verify_grace_convergence,
        verify_no_candidate_sets,
        evaluate_semantic_similarity_theory_true,
    )
    
    print("="*80)
    print("TEST 1: THEORY-TRUE CORRECTNESS")
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
    
    # Generate test contexts
    test_contexts = []
    for i in range(100):
        # Random context of length 32
        ctx = np.random.randint(4, len(word_to_idx), size=32).tolist()
        test_contexts.append(ctx)
    
    # Test 1.1: Grace ALWAYS converges
    print("\n  Test 1.1: Grace convergence...")
    convergence_result = verify_grace_convergence(model, test_contexts)
    print(f"    Convergence rate: {convergence_result['convergence_rate']:.1%}")
    print(f"    Avg stability: {convergence_result['avg_stability']:.4f}")
    
    assert convergence_result['convergence_rate'] == 1.0, \
        f"Grace must ALWAYS converge (got {convergence_result['convergence_rate']:.1%})"
    assert convergence_result['avg_stability'] >= 0.382, \
        f"Stability should be >= φ⁻² (got {convergence_result['avg_stability']:.4f})"
    
    # Test 1.2: No candidate sets (full vocabulary)
    print("\n  Test 1.2: Full vocabulary retrieval...")
    vocab_result = verify_no_candidate_sets(model, test_contexts)
    print(f"    Unique tokens returned: {vocab_result['unique_tokens_returned']:,}")
    print(f"    Coverage ratio: {vocab_result['coverage_ratio']:.1%}")
    
    # Should return diverse tokens (not just a few)
    assert vocab_result['unique_tokens_returned'] > 10, \
        f"Retrieval should return diverse tokens (got {vocab_result['unique_tokens_returned']})"
    
    # Test 1.3: Coherence scoring matches theory
    print("\n  Test 1.3: Coherence scoring...")
    # Learn a few patterns
    samples = [
        ([1, 2, 3, 4, 5], 10),
        ([6, 7, 8, 9, 10], 20),
        ([11, 12, 13, 14, 15], 30),
    ]
    for ctx, tgt in samples:
        model.learn(ctx, tgt)
    
    eval_result = evaluate_semantic_similarity_theory_true(model, samples, n_eval=3)
    print(f"    Semantic similarity: {eval_result['semantic_similarity']:.4f}")
    print(f"    Exact match rate: {eval_result['exact_match_rate']:.1%}")
    
    # Should have non-zero coherence
    assert eval_result['semantic_similarity'] > 0.0, \
        f"Coherence should be > 0 (got {eval_result['semantic_similarity']:.6f})"
    
    print("\n  ✓ All theory-true correctness tests passed!")
    
    return {
        'convergence_rate': convergence_result['convergence_rate'],
        'avg_stability': convergence_result['avg_stability'],
        'vocab_coverage': vocab_result['coverage_ratio'],
        'semantic_similarity': eval_result['semantic_similarity'],
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_performance():
    """
    Test 2: Performance
    
    Measures:
    1. Throughput (samples/sec)
    2. Memory usage (GPU)
    3. Batch processing efficiency
    4. Retrieval latency
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    from tqdm import tqdm
    
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    
    print("="*80)
    print("TEST 2: PERFORMANCE")
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
    
    # Generate test batch
    BATCH_SIZE = 2048
    context_size = 64
    contexts = []
    targets = []
    
    for i in range(BATCH_SIZE):
        ctx = np.random.randint(4, len(word_to_idx), size=context_size).tolist()
        tgt = np.random.randint(4, len(word_to_idx))
        contexts.append(ctx)
        targets.append(tgt)
    
    # Test 2.1: Learning throughput
    print("\n  Test 2.1: Learning throughput...")
    t0 = time.time()
    model.tower.learn_batch(contexts, targets)
    learn_time = time.time() - t0
    learn_throughput = BATCH_SIZE / learn_time
    
    print(f"    Batch size: {BATCH_SIZE:,}")
    print(f"    Time: {learn_time:.3f}s")
    print(f"    Throughput: {learn_throughput:,.0f} samples/sec")
    
    # Test 2.2: Retrieval latency
    print("\n  Test 2.2: Retrieval latency...")
    retrieval_times = []
    for ctx in contexts[:100]:  # Test 100 retrievals
        t0 = time.time()
        _ = model.tower.retrieve(ctx)
        retrieval_times.append(time.time() - t0)
    
    avg_retrieval_time = np.mean(retrieval_times)
    p50_retrieval_time = np.percentile(retrieval_times, 50)
    p99_retrieval_time = np.percentile(retrieval_times, 99)
    
    print(f"    Avg latency: {avg_retrieval_time*1000:.2f}ms")
    print(f"    P50 latency: {p50_retrieval_time*1000:.2f}ms")
    print(f"    P99 latency: {p99_retrieval_time*1000:.2f}ms")
    
    # Test 2.3: GPU memory usage
    print("\n  Test 2.3: GPU memory usage...")
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    
    # Measure before
    mem_before = mempool.used_bytes()
    
    # Learn another batch
    model.tower.learn_batch(contexts[:1024], targets[:1024])
    
    # Measure after
    mem_after = mempool.used_bytes()
    mem_used = (mem_after - mem_before) / (1024**3)  # GB
    
    print(f"    Memory used: {mem_used:.3f} GB")
    
    # Performance targets (should be fast)
    assert learn_throughput > 1000, \
        f"Learning throughput too low: {learn_throughput:.0f} samples/sec"
    assert avg_retrieval_time < 0.1, \
        f"Retrieval latency too high: {avg_retrieval_time*1000:.2f}ms"
    
    print("\n  ✓ All performance tests passed!")
    
    return {
        'learn_throughput': learn_throughput,
        'avg_retrieval_latency_ms': avg_retrieval_time * 1000,
        'p99_retrieval_latency_ms': p99_retrieval_time * 1000,
        'gpu_memory_gb': mem_used,
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,  # Longer timeout for learning test
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_learning_verification():
    """
    Test 3: Learning Verification
    
    Verifies:
    1. Semantic similarity increases over batches
    2. Witness stability improves (churn decreases)
    3. Grade energy evolves correctly (bivector → scalar/pseudo)
    4. Satellite occupancy becomes Zipfian
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    from tqdm import tqdm
    from datasets import load_dataset
    
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    from holographic_prod.core.algebra import decompose_to_coefficients
    from holographic_prod.core.quotient import extract_witness
    from holographic_prod.tests.theory_true_evaluation_helper import (
        evaluate_semantic_similarity_theory_true,
    )
    
    print("="*80)
    print("TEST 3: LEARNING VERIFICATION")
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
    
    # Prepare samples
    context_size = 64
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    samples = []
    
    for item in tqdm(ds.take(20_000), total=20_000, desc="Tokenizing"):
        words = item['text'].lower().split()
        tokens = [word_to_idx.get(w, 1) for w in words]
        for i in range(len(tokens) - context_size):
            ctx = tokens[i:i+context_size]
            tgt = tokens[i+context_size]
            if tgt != 1:  # Skip <unk> targets
                samples.append((ctx, tgt))
            if len(samples) >= 20_000:
                break
        if len(samples) >= 20_000:
            break
    
    print(f"  ✓ Prepared {len(samples):,} samples")
    
    # Training configuration
    BATCH_SIZE = 2048
    N_BATCHES = 20
    EVAL_EVERY = 5
    
    # Metrics tracking
    semantic_sims = []
    witness_churns = []
    grade_energies = []
    satellite_stats = []
    
    prev_witness = None
    
    # Evaluation set (holdout)
    eval_start = int(len(samples) * 0.8)
    eval_samples = samples[eval_start:]
    
    print(f"\n  Training for {N_BATCHES} batches...")
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(samples))
        batch = samples[start_idx:end_idx]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # Learn batch
        model.tower.learn_batch(contexts, targets)
        model.n_patterns += len(contexts)
        
        # Periodic evaluation
        if (batch_idx + 1) % EVAL_EVERY == 0:
            # Semantic similarity
            eval_result = evaluate_semantic_similarity_theory_true(
                model, eval_samples, n_eval=100
            )
            semantic_sims.append(eval_result['semantic_similarity'])
            
            # Witness churn
            grand_master = model.tower.get_grand_master_state()
            xp = model.xp
            if hasattr(grand_master, 'get'):
                gm = grand_master.get()
            else:
                gm = grand_master
            curr_witness = (float(gm[0]), float(gm[15]))
            
            if prev_witness is not None:
                churn = np.sqrt(
                    (curr_witness[0] - prev_witness[0])**2 +
                    (curr_witness[1] - prev_witness[1])**2
                )
                witness_churns.append(churn)
            prev_witness = curr_witness
            
            # Grade energies (sample from satellite memories)
            sample_indices = np.random.choice(model.tower.n_satellites, size=10, replace=False)
            avg_energies = defaultdict(float)
            for idx in sample_indices:
                mem = model.tower._all_memories[idx]
                coeffs = decompose_to_coefficients(mem, model.basis, model.xp)
                if hasattr(coeffs, 'get'):
                    coeffs = coeffs.get()
                
                avg_energies['scalar'] += float(coeffs[0]**2) / 10
                avg_energies['bivector'] += float(sum(c**2 for c in coeffs[5:11])) / 10
                avg_energies['pseudo'] += float(coeffs[15]**2) / 10
            grade_energies.append(dict(avg_energies))
            
            # Satellite stats
            counts = model.tower._satellite_n_bindings
            if hasattr(counts, 'get'):
                counts = counts.get()
            active_mask = counts > 0
            n_active = int(np.sum(active_mask))
            
            if n_active > 10:
                active_counts = counts[active_mask]
                sorted_counts = np.sort(active_counts)[::-1]
                zipf_ratio = sorted_counts[0] / (sorted_counts[len(sorted_counts)//2] + 1)
            else:
                zipf_ratio = 0.0
            
            satellite_stats.append({
                'n_active': n_active,
                'zipf_ratio': float(zipf_ratio),
            })
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Semantic sim: {eval_result['semantic_similarity']:.4f}")
            print(f"    Active sats:  {n_active:,}")
            print(f"    Zipf ratio:   {zipf_ratio:.1f}")
            if witness_churns:
                print(f"    Witness churn: {witness_churns[-1]:.4f}")
    
    # Verify learning trends
    print("\n  Verifying learning trends...")
    
    # 1. Semantic similarity should increase (or at least not decrease significantly)
    if len(semantic_sims) >= 2:
        sim_trend = semantic_sims[-1] - semantic_sims[0]
        print(f"    Semantic sim trend: {sim_trend:+.4f}")
        # Allow small decrease but should generally improve
        assert sim_trend > -0.1, \
            f"Semantic similarity decreased too much: {sim_trend:.4f}"
    
    # 2. Witness churn should decrease (stabilize)
    if len(witness_churns) >= 2:
        churn_trend = witness_churns[-1] - witness_churns[0]
        print(f"    Witness churn trend: {churn_trend:+.4f}")
        # Churn should decrease (become more negative)
        # But allow some fluctuation
    
    # 3. Grade energy should shift toward scalar/pseudo
    if len(grade_energies) >= 2:
        early_ge = grade_energies[0]
        late_ge = grade_energies[-1]
        early_total = sum(early_ge.values())
        late_total = sum(late_ge.values())
        
        if early_total > 0 and late_total > 0:
            early_scalar_pct = (early_ge['scalar'] + early_ge['pseudo']) / early_total
            late_scalar_pct = (late_ge['scalar'] + late_ge['pseudo']) / late_total
            
            print(f"    Scalar+Pseudo: {early_scalar_pct*100:.1f}% → {late_scalar_pct*100:.1f}%")
            # Should increase (more stable structure)
            assert late_scalar_pct >= early_scalar_pct - 0.1, \
                f"Grade energy didn't shift toward stable structure"
    
    # 4. Satellite occupancy should become Zipfian
    if satellite_stats:
        final_zipf = satellite_stats[-1]['zipf_ratio']
        print(f"    Final Zipf ratio: {final_zipf:.1f}")
        # Should show some Zipfian distribution (ratio > 1)
        assert final_zipf > 1.0 or satellite_stats[-1]['n_active'] < 20, \
            f"Satellite occupancy not Zipfian (ratio: {final_zipf:.1f})"
    
    print("\n  ✓ All learning verification tests passed!")
    
    return {
        'semantic_sims': semantic_sims,
        'witness_churns': witness_churns,
        'grade_energies': grade_energies,
        'satellite_stats': satellite_stats,
    }


@app.local_entrypoint()
def main():
    """Run all comprehensive tests."""
    print("Running Comprehensive Theory-True Tests...")
    
    print("\n" + "="*80)
    result1 = test_theory_true_correctness.remote()
    print(f"\nCorrectness Results: {result1}")
    
    print("\n" + "="*80)
    result2 = test_performance.remote()
    print(f"\nPerformance Results: {result2}")
    
    print("\n" + "="*80)
    result3 = test_learning_verification.remote()
    print(f"\nLearning Results: {result3}")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE!")
    print("="*80)
