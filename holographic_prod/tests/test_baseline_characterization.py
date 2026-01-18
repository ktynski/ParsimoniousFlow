"""
Baseline Learning Characterization

PURPOSE:
    Establish baseline metrics BEFORE adding DMN/genius features.
    All subsequent tests compare against these measurements.

METRICS CAPTURED:
    1. Semantic similarity curve over batches
    2. Grade energy distribution (scalar/bivector/trivector/pseudoscalar)
    3. Satellite occupancy (Zipf ratio, top-k share)
    4. Witness stability (churn rate)
    5. Throughput (samples/sec)

THEORY (Physics framing):
    Learning is like turbulent flow settling to laminar:
    - Early: High bivector/trivector (rotational energy = exploration)
    - Late: High scalar/pseudoscalar (settled structure = knowledge)
    
    The grade energy distribution reveals WHERE the system is in phase space.
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

app = modal.App("baseline-characterization")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_baseline_characterization():
    """
    Comprehensive baseline characterization of holographic memory learning.
    
    Returns metrics that all subsequent tests will compare against.
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
        decompose_to_coefficients, grace_basin_keys_batch_direct,
        frobenius_cosine
    )
    from holographic_prod.core.constants import PHI, PHI_INV
    from holographic_prod.core.quotient import extract_witness, grace_stability
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    
    print("="*80)
    print("BASELINE LEARNING CHARACTERIZATION")
    print("="*80)
    print("""
    PURPOSE: Establish reference metrics for all DMN/genius tests.
    
    METRICS:
        1. Semantic similarity curve
        2. Grade energy distribution
        3. Satellite occupancy (Zipf)
        4. Witness stability (churn)
        5. Throughput
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
    
    # Create grounded embeddings (includes GloVe loading)
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
    print("\n  Initializing model...")
    
    model = HolographicMemory(
        vocab_size=len(word_to_idx),
        max_levels=4,
        seed=42,
        use_gpu=True,
        grounded_embeddings=cp.asarray(grounded_embs),
    )
    model.set_grounded_embeddings(cp.asarray(grounded_embs))
    
    n_satellites = model.tower.n_satellites
    print(f"  ✓ Model initialized: {n_satellites:,} satellites")
    
    # =========================================================================
    # MEASUREMENT FUNCTIONS
    # =========================================================================
    
    def measure_grade_energies(matrix, basis, xp):
        """
        Decompose matrix into grade energies.
        
        THEORY:
            Clifford algebra has grades 0-4:
            - Grade 0 (scalar): committed decisions
            - Grade 1 (vector): directional bias
            - Grade 2 (bivector): rotational/relational structure
            - Grade 3 (trivector): higher-order relations
            - Grade 4 (pseudoscalar): narrative/holistic structure
            
            The distribution reveals phase-space position.
        """
        coeffs = decompose_to_coefficients(matrix, basis, xp)
        if hasattr(coeffs, 'get'):
            coeffs = coeffs.get()
        
        return {
            'scalar': float(coeffs[0]**2),
            'vector': float(sum(c**2 for c in coeffs[1:5])),
            'bivector': float(sum(c**2 for c in coeffs[5:11])),
            'trivector': float(sum(c**2 for c in coeffs[11:15])),
            'pseudo': float(coeffs[15]**2),
        }
    
    def measure_witness_churn(prev_witness, curr_witness, xp):
        """
        Measure how much the witness has changed.
        
        THEORY:
            Witness = invariant structure preserved under Grace.
            High churn = unstable identity (psychosis-like)
            Low churn = stable identity (healthy/genius)
        
        NOTE: extract_witness returns (scalar, pseudoscalar) tuple
        """
        if prev_witness is None:
            return 0.0
        # Handle tuple form (scalar, pseudoscalar)
        if isinstance(curr_witness, tuple):
            prev_s, prev_p = prev_witness
            curr_s, curr_p = curr_witness
            # Convert to scalar if needed
            if hasattr(prev_s, 'get'):
                prev_s, prev_p = float(prev_s.get()), float(prev_p.get())
                curr_s, curr_p = float(curr_s.get()), float(curr_p.get())
            else:
                prev_s, prev_p = float(prev_s), float(prev_p)
                curr_s, curr_p = float(curr_s), float(curr_p)
            return np.sqrt((curr_s - prev_s)**2 + (curr_p - prev_p)**2)
        # Array form
        diff = curr_witness - prev_witness
        if hasattr(diff, 'get'):
            diff = diff.get()
        return float(np.linalg.norm(diff))
    
    def measure_satellite_stats(model, xp):
        """
        Measure satellite distribution statistics.
        
        THEORY:
            Healthy learning should show Zipfian distribution
            (some satellites very active, most sparse).
        """
        counts = model.tower._satellite_n_bindings
        if hasattr(counts, 'get'):
            counts = counts.get()
        
        active_mask = counts > 0
        n_active = int(np.sum(active_mask))
        
        if n_active < 10:
            return {
                'n_active': n_active,
                'zipf_ratio': 0.0,
                'top10_share': 0.0,
                'gini': 0.0,
            }
        
        active_counts = counts[active_mask]
        sorted_counts = np.sort(active_counts)[::-1]
        
        # Zipf ratio: max / median
        zipf_ratio = sorted_counts[0] / (sorted_counts[len(sorted_counts)//2] + 1)
        
        # Top-10 share
        top10_share = np.sum(sorted_counts[:10]) / (np.sum(sorted_counts) + 1)
        
        # Gini coefficient (inequality measure)
        n = len(active_counts)
        sorted_asc = np.sort(active_counts)
        cumsum = np.cumsum(sorted_asc)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_asc)) - (n+1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)
        
        return {
            'n_active': n_active,
            'zipf_ratio': float(zipf_ratio),
            'top10_share': float(top10_share),
            'gini': float(gini),
        }
    
    # Import shared theory-true evaluation helper
    from holographic_prod.tests.theory_true_evaluation_helper import (
        evaluate_semantic_similarity_theory_true
    )
    
    def evaluate_semantic_similarity(model, eval_samples, n_eval=100):
        """
        Compute semantic similarity using THEORY-TRUE retrieval path.
        
        Uses shared helper that matches retrieve() exactly.
        """
        result = evaluate_semantic_similarity_theory_true(
            model, eval_samples, n_eval=n_eval, return_details=False
        )
        return result['semantic_similarity']
    
    # =========================================================================
    # TRAINING LOOP WITH MEASUREMENTS
    # =========================================================================
    print("\n" + "="*80)
    print("BASELINE TRAINING")
    print("="*80)
    
    BATCH_SIZE = 2048
    N_BATCHES = 50
    EVAL_EVERY = 5
    
    # Metrics storage
    metrics = {
        'semantic_sim': [],
        'throughput': [],
        'grade_energies': [],
        'satellite_stats': [],
        'witness_churn': [],
        'batch_times': [],
    }
    
    prev_witness = None
    # Use samples not in training set for evaluation
    # Reserve last 20% for evaluation, or use samples beyond training if available
    train_size = min(N_BATCHES * BATCH_SIZE, len(samples))
    eval_start = max(train_size, int(len(samples) * 0.8))
    eval_samples = samples[eval_start:]
    
    # If eval_samples is empty, use a random subset from training (not ideal but better than nothing)
    if len(eval_samples) == 0:
        import random
        random.seed(42)
        eval_indices = random.sample(range(train_size), min(1000, train_size))
        eval_samples = [samples[i] for i in eval_indices]
    
    print(f"  ✓ Evaluation set: {len(eval_samples):,} samples")
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(samples))
        batch = samples[start_idx:end_idx]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # Learn batch
        t0 = time.time()
        model.tower.learn_batch(contexts, targets)
        model.n_patterns += len(contexts)
        batch_time = time.time() - t0
        
        throughput = len(batch) / batch_time if batch_time > 0 else 0
        metrics['batch_times'].append(batch_time)
        
        # Periodic evaluation
        if (batch_idx + 1) % EVAL_EVERY == 0:
            # Semantic similarity
            sim = evaluate_semantic_similarity(model, eval_samples)
            metrics['semantic_sim'].append(sim)
            metrics['throughput'].append(throughput)
            
            # Grade energies (sample from a few satellite memories)
            # Handle both TowerMemory (_all_memories) and MultiLevelTower (_all_memories)
            satellite_memories = getattr(model.tower, '_all_memories', None)
            if satellite_memories is None:
                satellite_memories = getattr(model.tower, '_all_memories', None)
            
            sample_indices = np.random.choice(model.tower.n_satellites, size=10, replace=False)
            avg_energies = defaultdict(float)
            for idx in sample_indices:
                mem = satellite_memories[idx]
                energies = measure_grade_energies(mem, model.basis, model.xp)
                for k, v in energies.items():
                    avg_energies[k] += v / 10
            metrics['grade_energies'].append(dict(avg_energies))
            
            # Satellite stats
            sat_stats = measure_satellite_stats(model, model.xp)
            metrics['satellite_stats'].append(sat_stats)
            
            # Witness churn
            # Extract witness from grand master (coefficients[0] = scalar, [15] = pseudo)
            grand_master = model.tower.get_grand_master_state()
            xp = model.xp
            if hasattr(grand_master, 'get'):
                gm = grand_master.get()
            else:
                gm = grand_master
            curr_witness = (float(gm[0]), float(gm[15]))
            churn = measure_witness_churn(prev_witness, curr_witness, model.xp)
            metrics['witness_churn'].append(churn)
            prev_witness = curr_witness
            
            # Print progress
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Semantic sim: {sim:.4f}")
            print(f"    Throughput:   {throughput:,.0f}/s")
            print(f"    Active sats:  {sat_stats['n_active']:,}")
            print(f"    Zipf ratio:   {sat_stats['zipf_ratio']:.1f}")
            print(f"    Witness churn: {churn:.4f}")
            
            # Grade energy breakdown
            ge = metrics['grade_energies'][-1]
            total = sum(ge.values())
            if total > 0:
                print(f"    Grade energies:")
                print(f"      Scalar:    {ge['scalar']/total*100:5.1f}%")
                print(f"      Vector:    {ge['vector']/total*100:5.1f}%")
                print(f"      Bivector:  {ge['bivector']/total*100:5.1f}%")
                print(f"      Trivector: {ge['trivector']/total*100:5.1f}%")
                print(f"      Pseudo:    {ge['pseudo']/total*100:5.1f}%")
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "="*80)
    print("BASELINE SUMMARY")
    print("="*80)
    
    summary = {
        'final_semantic_sim': metrics['semantic_sim'][-1] if metrics['semantic_sim'] else 0.0,
        'avg_semantic_sim': float(np.mean(metrics['semantic_sim'])) if metrics['semantic_sim'] else 0.0,
        'avg_throughput': float(np.mean(metrics['throughput'])) if metrics['throughput'] else 0.0,
        'final_active_satellites': metrics['satellite_stats'][-1]['n_active'] if metrics['satellite_stats'] else 0,
        'final_zipf_ratio': metrics['satellite_stats'][-1]['zipf_ratio'] if metrics['satellite_stats'] else 0.0,
        'avg_witness_churn': float(np.mean(metrics['witness_churn'])) if metrics['witness_churn'] else 0.0,
        'total_patterns': model.n_patterns,
    }
    
    # Grade energy evolution
    if metrics['grade_energies']:
        first_ge = metrics['grade_energies'][0]
        last_ge = metrics['grade_energies'][-1]
        
        first_total = sum(first_ge.values())
        last_total = sum(last_ge.values())
        
        summary['grade_energy_evolution'] = {
            'early': {k: v/first_total for k, v in first_ge.items()} if first_total > 0 else first_ge,
            'late': {k: v/last_total for k, v in last_ge.items()} if last_total > 0 else last_ge,
        }
    
    print(f"""
    FINAL METRICS:
      Semantic similarity: {summary['final_semantic_sim']:.4f}
      Avg throughput:      {summary['avg_throughput']:,.0f} samples/sec
      Active satellites:   {summary['final_active_satellites']:,}
      Zipf ratio:          {summary['final_zipf_ratio']:.1f}
      Avg witness churn:   {summary['avg_witness_churn']:.4f}
      Total patterns:      {summary['total_patterns']:,}
    
    GRADE ENERGY EVOLUTION:
      Early → Late:
      - Scalar:    {summary.get('grade_energy_evolution', {}).get('early', {}).get('scalar', 0)*100:5.1f}% → {summary.get('grade_energy_evolution', {}).get('late', {}).get('scalar', 0)*100:5.1f}%
      - Bivector:  {summary.get('grade_energy_evolution', {}).get('early', {}).get('bivector', 0)*100:5.1f}% → {summary.get('grade_energy_evolution', {}).get('late', {}).get('bivector', 0)*100:5.1f}%
      - Pseudo:    {summary.get('grade_energy_evolution', {}).get('early', {}).get('pseudo', 0)*100:5.1f}% → {summary.get('grade_energy_evolution', {}).get('late', {}).get('pseudo', 0)*100:5.1f}%
    """)
    
    # Save metrics to checkpoint volume
    metrics_path = "/checkpoints/baseline_metrics.json"
    full_results = {
        'summary': summary,
        'curves': {
            'semantic_sim': metrics['semantic_sim'],
            'throughput': metrics['throughput'],
            'witness_churn': metrics['witness_churn'],
        },
        'satellite_stats': metrics['satellite_stats'],
        'grade_energies': metrics['grade_energies'],
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  ✓ Saved metrics to {metrics_path}")
    
    print("\nBaseline characterization complete!")
    return full_results


@app.local_entrypoint()
def main():
    print("Running Baseline Characterization on Modal H100...")
    result = test_baseline_characterization.remote()
    print(f"\nReturned summary: {result['summary']}")
