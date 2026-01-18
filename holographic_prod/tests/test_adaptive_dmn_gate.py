"""
Adaptive DMN Gate Test

THE BREAKTHROUGH:
    DMN interleaving is not a property of intelligence—it is a property of learning style.
    
    Genius-level cognition emerges when:
    - Consolidation pressure is present
    - But NOT strong enough to overwrite task signal
    - And NOT absent enough to let noise accumulate
    
    Fixed MINI_REM_EVERY is biologically wrong.
    Real brains gate DMN based on SYSTEM STATE.

GATING SIGNALS (what to measure):

    1. ENTROPY SIGNAL (ACC analog)
       - High entropy in satellite activations → need consolidation
       - Low entropy → already consolidated, don't disrupt
       
    2. CONFLICT SIGNAL (response uncertainty)
       - High conflict → confused, might benefit from recombination
       - Low conflict → clear task signal, don't inject noise
       
    3. WITNESS DRIFT SIGNAL (identity stability)
       - High drift → already destabilizing, suppress DMN
       - Low drift → stable base, can afford exploration

    4. COMPRESSION RATIO (learning efficiency)
       - Increasing patterns, flat similarity → noise accumulating
       - Both increasing → healthy learning, less DMN needed

GATE LOGIC:
    dmn_gate = f(entropy, conflict, drift, compression)
    
    OPEN DMN when:
        - High entropy (needs organization)
        - Moderate conflict (genuine uncertainty)
        - Low drift (stable identity)
        
    CLOSE DMN when:
        - Low entropy (already organized)
        - High conflict + high drift (psychosis risk)
        - During rapid learning (don't interrupt)

This is the "breakthrough" test: derive an adaptive gate, not a fixed schedule.
"""

import modal
import numpy as np
import time
import json
from typing import List, Dict, Any, Tuple

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

app = modal.App("adaptive-dmn-gate-test")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,  # 1 hour
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_adaptive_dmn_gate():
    """
    Test adaptive DMN gating vs fixed schedule.
    
    This is the key experiment: can we derive WHEN to consolidate
    from system state, rather than a fixed clock?
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
    # extract_witness expects matrix, but get_grand_master_state returns coefficients
    # So we extract witness directly: scalar = coeffs[0], pseudo = coeffs[15]
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    
    # Helper to handle both TowerMemory and MultiLevelTower
    def get_satellite_memories(tower):
        return getattr(tower, '_satellite_memories', None) or getattr(tower, '_all_memories', None)
    
    print("="*80)
    print("ADAPTIVE DMN GATE TEST")
    print("="*80)
    print("""
    HYPOTHESIS: Adaptive DMN gating based on system state beats fixed schedule.
    
    GATING SIGNALS:
        1. Entropy (satellite activation distribution)
        2. Conflict (response uncertainty)
        3. Witness drift (identity stability)
        4. Learning rate (sim improvement per batch)
        
    TEST:
        A. Standard (no DMN)
        B. Fixed DMN (every 5 batches)
        C. Adaptive DMN (state-dependent gating)
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
    model_fixed = create_model()
    model_adaptive = create_model()
    
    print(f"  ✓ Models: {model_standard.tower.n_satellites:,} satellites each")
    
    # =========================================================================
    # GATING SIGNAL FUNCTIONS
    # =========================================================================
    
    def compute_satellite_entropy(model):
        """
        Compute entropy of satellite activation distribution.
        
        High entropy = activations spread across many satellites (disorganized)
        Low entropy = concentrated in few satellites (organized)
        """
        xp = model.xp
        counts = model.tower._satellite_n_bindings
        
        if hasattr(counts, 'get'):
            counts = counts.get()
        
        # Normalize to probability distribution
        total = np.sum(counts) + 1e-10
        probs = counts / total
        
        # Entropy (clip to avoid log(0))
        probs = np.clip(probs, 1e-15, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        
        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log(len(counts))
        normalized_entropy = entropy / max_entropy
        
        return float(normalized_entropy)
    
    def compute_avg_conflict(model, sample_contexts, n_sample=20):
        """
        Compute average conflict (uncertainty) over sample contexts.
        
        High conflict = top candidates are close in score (uncertain)
        Low conflict = clear winner (certain)
        """
        xp = model.xp
        conflicts = []
        
        for ctx in sample_contexts[:n_sample]:
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
                
                sat_memory = get_satellite_memories(model.tower)[sat_idx]
                retrieved = ctx_emb.T @ sat_memory
                
                scores = xp.einsum('ij,vij->v', retrieved, model.tower.embeddings)
                
                # Top 2
                top2_idx = xp.argpartition(scores, -2)[-2:]
                top2_scores = scores[top2_idx]
                
                if hasattr(top2_scores, 'get'):
                    top2_scores = top2_scores.get()
                
                top2_sorted = np.sort(top2_scores)
                conflict = top2_sorted[0] / (top2_sorted[1] + 1e-10)
                conflicts.append(conflict)
            except Exception as e:
                raise RuntimeError(f"Conflict computation failed: {e}") from e
        
        if not conflicts:
            raise RuntimeError("No conflicts computed - all samples failed")
        return float(np.mean(conflicts))
    
    def compute_witness_drift(model, prev_witness):
        """
        Compute witness drift from previous state.
        
        High drift = identity unstable (psychosis risk)
        Low drift = identity stable (safe to explore)
        """
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
        
        drift = float(np.linalg.norm(diff))
        return drift, curr_witness
    
    def compute_learning_rate(sim_history, window=3):
        """
        Compute recent learning rate (improvement in similarity).
        
        High rate = actively learning, don't interrupt
        Low rate = stalled, might benefit from consolidation
        """
        if len(sim_history) < 2:
            return 0.0
        
        recent = sim_history[-window:] if len(sim_history) >= window else sim_history
        if len(recent) < 2:
            return 0.0
        
        # Slope of recent similarity
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        return float(slope)
    
    # =========================================================================
    # DMN GATE FUNCTION
    # =========================================================================
    
    def dmn_gate_decision(
        entropy: float,
        conflict: float,
        drift: float,
        learning_rate: float,
        thresholds: Dict[str, float] = None
    ) -> Tuple[bool, str]:
        """
        Adaptive DMN gate decision.
        
        Returns:
            (should_activate_dmn, reason)
        
        LOGIC:
            OPEN DMN when:
                - High entropy (needs organization) AND low drift (safe)
                - Low learning rate (stalled) AND moderate conflict
                
            CLOSE DMN when:
                - High drift (psychosis risk)
                - High learning rate (don't interrupt)
                - Low entropy AND low conflict (already stable)
        """
        if thresholds is None:
            thresholds = {
                'entropy_high': 0.7,     # Above this → needs organization
                'entropy_low': 0.4,      # Below this → already organized
                'conflict_high': 0.8,    # Above this → very uncertain
                'conflict_moderate': 0.5, # This range → genuine uncertainty
                'drift_high': 0.1,       # Above this → unstable
                'learning_high': 0.001,  # Above this → actively learning
            }
        
        # PSYCHOSIS GUARD: Never DMN during high drift
        if drift > thresholds['drift_high']:
            return False, "drift_too_high"
        
        # FOCUS GUARD: Don't interrupt rapid learning
        if learning_rate > thresholds['learning_high']:
            return False, "learning_rapidly"
        
        # CONSOLIDATION NEEDED: High entropy + safe conditions
        if entropy > thresholds['entropy_high']:
            if conflict < thresholds['conflict_high']:  # Not completely lost
                return True, "high_entropy_needs_organization"
        
        # STALLED LEARNING: Low learning rate + moderate conflict
        if learning_rate < -0.0001:  # Similarity actually decreasing
            if conflict > thresholds['conflict_moderate']:
                return True, "stalled_with_uncertainty"
        
        # LOW ENTROPY BUT HIGH CONFLICT: Might be stuck in local minimum
        if entropy < thresholds['entropy_low'] and conflict > thresholds['conflict_high']:
            return True, "stuck_in_local_minimum"
        
        # DEFAULT: No DMN needed
        return False, "stable_learning"
    
    # =========================================================================
    # CONSOLIDATION FUNCTION
    # =========================================================================
    
    def mini_rem_consolidation(model, n_recombine=5, survival_iters=3):
        """Mini-REM: recombine + survival test."""
        xp = model.xp
        tower = model.tower
        
        active_mask = tower._satellite_n_bindings > 0
        active_indices = xp.where(active_mask)[0]
        
        if len(active_indices) < 2:
            return 0
        
        survivors = 0
        
        for _ in range(n_recombine):
            pair_idx = np.random.choice(len(active_indices), size=2, replace=False)
            idx_a = int(active_indices[pair_idx[0]])
            idx_b = int(active_indices[pair_idx[1]])
            
            mem_a = get_satellite_memories(tower)[idx_a]
            mem_b = get_satellite_memories(tower)[idx_b]
            
            recombined = geometric_product(mem_a, mem_b)
            
            settled = recombined
            for _ in range(survival_iters):
                settled = grace_operator(settled, model.basis, xp)
            
            stability = float(frobenius_cosine(settled, recombined, xp).get())
            
            if stability > PHI_INV:
                low_activity_mask = tower._satellite_n_bindings < 10
                low_indices = xp.where(low_activity_mask)[0]
                
                if len(low_indices) > 0:
                    target_idx = int(low_indices[np.random.randint(len(low_indices))])
                    get_satellite_memories(tower)[target_idx] += PHI_INV * settled
                    survivors += 1
        
        return survivors
    
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
                
                sat_memory = get_satellite_memories(model.tower)[sat_idx]
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
    N_BATCHES = 60
    EVAL_EVERY = 3
    FIXED_DMN_EVERY = 5
    
    metrics = {
        'standard': {'sim': [], 'entropy': []},
        'fixed': {'sim': [], 'entropy': [], 'dmn_count': 0},
        'adaptive': {
            'sim': [], 'entropy': [], 'dmn_count': 0,
            'gate_decisions': [], 'reasons': []
        },
    }
    
    prev_witness_adaptive = None
    sim_history_adaptive = []
    
    eval_samples = samples[N_BATCHES * BATCH_SIZE:]
    sample_contexts = [ctx for ctx, _ in eval_samples[:50]]
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        batch = samples[start_idx:start_idx + BATCH_SIZE]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # Standard: just learn
        model_standard.tower.learn_batch(contexts, targets)
        model_standard.n_patterns += len(contexts)
        
        # Fixed: learn + DMN every N batches
        model_fixed.tower.learn_batch(contexts, targets)
        model_fixed.n_patterns += len(contexts)
        if (batch_idx + 1) % FIXED_DMN_EVERY == 0:
            mini_rem_consolidation(model_fixed, n_recombine=5)
            metrics['fixed']['dmn_count'] += 1
        
        # Adaptive: learn + state-dependent DMN
        model_adaptive.tower.learn_batch(contexts, targets)
        model_adaptive.n_patterns += len(contexts)
        
        # Compute gating signals
        entropy = compute_satellite_entropy(model_adaptive)
        conflict = compute_avg_conflict(model_adaptive, sample_contexts, n_sample=10)
        drift, prev_witness_adaptive = compute_witness_drift(model_adaptive, prev_witness_adaptive)
        learning_rate = compute_learning_rate(sim_history_adaptive)
        
        # Gate decision
        should_dmn, reason = dmn_gate_decision(entropy, conflict, drift, learning_rate)
        metrics['adaptive']['gate_decisions'].append(should_dmn)
        metrics['adaptive']['reasons'].append(reason)
        
        if should_dmn:
            mini_rem_consolidation(model_adaptive, n_recombine=5)
            metrics['adaptive']['dmn_count'] += 1
        
        # Evaluation
        if (batch_idx + 1) % EVAL_EVERY == 0:
            sim_standard = evaluate_model(model_standard, eval_samples)
            sim_fixed = evaluate_model(model_fixed, eval_samples)
            sim_adaptive = evaluate_model(model_adaptive, eval_samples)
            
            metrics['standard']['sim'].append(sim_standard)
            metrics['standard']['entropy'].append(compute_satellite_entropy(model_standard))
            
            metrics['fixed']['sim'].append(sim_fixed)
            metrics['fixed']['entropy'].append(compute_satellite_entropy(model_fixed))
            
            metrics['adaptive']['sim'].append(sim_adaptive)
            metrics['adaptive']['entropy'].append(entropy)
            
            sim_history_adaptive.append(sim_adaptive)
            
            recent_decisions = metrics['adaptive']['gate_decisions'][-EVAL_EVERY:]
            recent_reasons = metrics['adaptive']['reasons'][-EVAL_EVERY:]
            dmn_activations = sum(recent_decisions)
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Standard:   sim={sim_standard:.4f}")
            print(f"    Fixed DMN:  sim={sim_fixed:.4f} ({(sim_fixed/sim_standard-1)*100:+.1f}%), activations={metrics['fixed']['dmn_count']}")
            print(f"    Adaptive:   sim={sim_adaptive:.4f} ({(sim_adaptive/sim_standard-1)*100:+.1f}%), activations={metrics['adaptive']['dmn_count']}")
            print(f"    Gate signals: entropy={entropy:.3f}, conflict={conflict:.3f}, drift={drift:.4f}, lr={learning_rate:.5f}")
            print(f"    Recent DMN: {dmn_activations}/{len(recent_decisions)} ({[r[:10] for r in recent_reasons[-3:]]})")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("ADAPTIVE DMN GATE RESULTS")
    print("="*80)
    
    def get_late_avg(lst):
        return float(np.mean(lst[len(lst)//2:])) if lst else 0.0
    
    late_standard = get_late_avg(metrics['standard']['sim'])
    late_fixed = get_late_avg(metrics['fixed']['sim'])
    late_adaptive = get_late_avg(metrics['adaptive']['sim'])
    
    # Reason analysis
    reason_counts = defaultdict(int)
    for r in metrics['adaptive']['reasons']:
        reason_counts[r] += 1
    
    results = {
        'late_stage': {
            'standard': late_standard,
            'fixed': late_fixed,
            'adaptive': late_adaptive,
        },
        'improvement': {
            'fixed_vs_standard': (late_fixed / late_standard - 1) * 100,
            'adaptive_vs_standard': (late_adaptive / late_standard - 1) * 100,
            'adaptive_vs_fixed': (late_adaptive / late_fixed - 1) * 100,
        },
        'dmn_activations': {
            'fixed': metrics['fixed']['dmn_count'],
            'adaptive': metrics['adaptive']['dmn_count'],
            'efficiency': metrics['adaptive']['dmn_count'] / (metrics['fixed']['dmn_count'] + 1),
        },
        'gate_reasons': dict(reason_counts),
        'curves': {
            'sim': {k: v['sim'] for k, v in metrics.items()},
            'entropy': {k: v['entropy'] for k, v in metrics.items()},
        },
    }
    
    print(f"""
    LATE-STAGE SEMANTIC SIMILARITY:
      Standard (no DMN):    {late_standard:.4f}
      Fixed DMN (every 5):  {late_fixed:.4f} ({results['improvement']['fixed_vs_standard']:+.2f}%)
      Adaptive DMN:         {late_adaptive:.4f} ({results['improvement']['adaptive_vs_standard']:+.2f}%)
      
    DMN ACTIVATION COUNTS:
      Fixed:    {metrics['fixed']['dmn_count']} (every 5 batches)
      Adaptive: {metrics['adaptive']['dmn_count']} (state-dependent)
      Efficiency: {results['dmn_activations']['efficiency']:.2f}x (activations vs fixed)
      
    GATE DECISION BREAKDOWN:
    """)
    
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        pct = count / N_BATCHES * 100
        print(f"      {reason:30s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"""
    KEY METRICS:
      Adaptive > Fixed:     {late_adaptive > late_fixed}
      Fewer activations:    {metrics['adaptive']['dmn_count'] < metrics['fixed']['dmn_count']}
      Better efficiency:    {late_adaptive / metrics['adaptive']['dmn_count'] > late_fixed / metrics['fixed']['dmn_count'] if metrics['adaptive']['dmn_count'] > 0 else False}
      
    VERDICT:
    """)
    
    if late_adaptive > late_fixed and metrics['adaptive']['dmn_count'] <= metrics['fixed']['dmn_count']:
        print("    ✅ ADAPTIVE GATE IS THE BREAKTHROUGH")
        print("       - Better results with equal or fewer DMN activations")
        print("       - State-dependent gating outperforms fixed schedule")
    elif late_adaptive > late_fixed:
        print("    ⚠️ ADAPTIVE HELPS BUT USES MORE DMN")
        print("       - Better results, but not more efficient")
    elif late_adaptive < late_fixed:
        print("    ❌ FIXED SCHEDULE BEATS ADAPTIVE")
        print("       - Gating thresholds may need tuning")
    else:
        print("    🔄 NO SIGNIFICANT DIFFERENCE")
    
    # Save
    results_path = "/checkpoints/adaptive_dmn_gate_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


@app.local_entrypoint()
def main():
    print("Running Adaptive DMN Gate Test on Modal H100...")
    result = test_adaptive_dmn_gate.remote()
    print(f"\nAdaptive improvement: {result['improvement']['adaptive_vs_standard']:+.2f}%")
    print(f"Adaptive vs Fixed: {result['improvement']['adaptive_vs_fixed']:+.2f}%")
