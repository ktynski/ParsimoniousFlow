"""
Over-Dreaming Collapse Detection Test

THE PSYCHOSIS TRIPWIRE:
    Too much DMN without executive gating → collapse into incoherence.
    
    This test detects the signatures of "over-dreaming":
    - Witness churn accelerating
    - Semantic similarity plateauing then dropping
    - Entropy increasing despite consolidation
    - Cross-satellite coherence dropping
    
    The goal: derive automatic detection of "too much DMN"
    so the system can self-throttle before collapse.

COLLAPSE SIGNATURES:

    1. WITNESS CHURN ACCELERATION
       d²(witness)/dt² > 0 → accelerating instability
       
    2. SIMILARITY PLATEAU → DROP
       Similarity derivative goes: positive → zero → negative
       
    3. ENTROPY DESPITE CONSOLIDATION
       DMN active but entropy still rising → not helping
       
    4. CROSS-SATELLITE DECOHERENCE
       Satellites becoming less similar to each other
       (fragmentation instead of compression)

This is the safety mechanism. Without it, continuous DMN is psychosis.
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

app = modal.App("overdreaming-collapse-test")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=2700,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_overdreaming_collapse():
    """
    Test over-dreaming collapse detection and prevention.
    
    We INTENTIONALLY push a model toward collapse by aggressive DMN,
    then test if we can detect the collapse signatures in time.
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
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    
    print("="*80)
    print("OVER-DREAMING COLLAPSE DETECTION TEST")
    print("="*80)
    print("""
    GOAL: Detect collapse signatures before they become catastrophic.
    
    TEST:
        A. Standard (control)
        B. Moderate DMN (every 3 batches)
        C. Aggressive DMN (every batch) - INTENTIONALLY PUSHING TOWARD COLLAPSE
        D. Adaptive with collapse detection (throttles when danger detected)
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
    # FOUR MODEL VARIANTS
    # =========================================================================
    print("\n  Initializing four models...")
    
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
    model_moderate = create_model()
    model_aggressive = create_model()
    model_adaptive = create_model()
    
    print(f"  ✓ Models: {model_standard.tower.n_satellites:,} satellites each")
    
    # =========================================================================
    # COLLAPSE DETECTION FUNCTIONS
    # =========================================================================
    
    class CollapseDetector:
        """Detects signatures of over-dreaming collapse."""
        
        def __init__(self, window=5):
            self.window = window
            self.witness_history = []
            self.churn_history = []
            self.sim_history = []
            self.entropy_history = []
            self.coherence_history = []
        
        def update(self, witness_churn, sim, entropy, cross_coherence):
            """Update history with new measurements."""
            self.churn_history.append(witness_churn)
            self.sim_history.append(sim)
            self.entropy_history.append(entropy)
            self.coherence_history.append(cross_coherence)
        
        def _derivative(self, series):
            """Compute first derivative of series."""
            if len(series) < 2:
                return 0.0
            return series[-1] - series[-2]
        
        def _second_derivative(self, series):
            """Compute second derivative (acceleration)."""
            if len(series) < 3:
                return 0.0
            d1 = series[-1] - series[-2]
            d2 = series[-2] - series[-3]
            return d1 - d2
        
        def detect_collapse(self) -> Dict[str, Any]:
            """
            Detect collapse signatures.
            
            Returns dict with:
                - is_collapsing: bool
                - confidence: float [0, 1]
                - signatures: list of detected signatures
            """
            signatures = []
            danger_level = 0.0
            
            if len(self.churn_history) < 3:
                return {'is_collapsing': False, 'confidence': 0.0, 'signatures': []}
            
            # 1. WITNESS CHURN ACCELERATION
            churn_accel = self._second_derivative(self.churn_history)
            if churn_accel > 0.01:
                signatures.append(f"churn_accelerating ({churn_accel:.4f})")
                danger_level += 0.3
            
            # 2. SIMILARITY PLATEAU → DROP
            if len(self.sim_history) >= 3:
                recent_slope = self._derivative(self.sim_history[-3:])
                if recent_slope < -0.005:  # Similarity dropping
                    signatures.append(f"similarity_dropping ({recent_slope:.4f})")
                    danger_level += 0.3
            
            # 3. ENTROPY DESPITE CONSOLIDATION (rising entropy)
            if len(self.entropy_history) >= 3:
                entropy_slope = self._derivative(self.entropy_history[-3:])
                if entropy_slope > 0.01:  # Entropy rising
                    signatures.append(f"entropy_rising ({entropy_slope:.4f})")
                    danger_level += 0.2
            
            # 4. CROSS-SATELLITE DECOHERENCE
            if len(self.coherence_history) >= 3:
                coherence_slope = self._derivative(self.coherence_history[-3:])
                if coherence_slope < -0.01:  # Coherence dropping
                    signatures.append(f"decoherence ({coherence_slope:.4f})")
                    danger_level += 0.2
            
            # 5. COMBINED: High churn + dropping sim = critical
            if len(self.churn_history) >= 3 and len(self.sim_history) >= 3:
                avg_recent_churn = np.mean(self.churn_history[-3:])
                sim_slope = self._derivative(self.sim_history[-3:])
                if avg_recent_churn > 0.05 and sim_slope < 0:
                    signatures.append("CRITICAL: high_churn+dropping_sim")
                    danger_level += 0.5
            
            is_collapsing = danger_level > 0.5
            confidence = min(danger_level, 1.0)
            
            return {
                'is_collapsing': is_collapsing,
                'confidence': confidence,
                'signatures': signatures,
                'danger_level': danger_level,
            }
    
    def compute_satellite_entropy(model):
        """Compute entropy of satellite activation distribution."""
        xp = model.xp
        counts = model.tower._satellite_n_bindings
        if hasattr(counts, 'get'):
            counts = counts.get()
        
        total = np.sum(counts) + 1e-10
        probs = counts / total
        probs = np.clip(probs, 1e-15, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(counts))
        
        return float(entropy / max_entropy)
    
    def compute_witness_churn(model, prev_witness):
        """Compute witness drift."""
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
    
    def compute_cross_satellite_coherence(model, n_sample=20):
        """Compute average coherence between random satellite pairs."""
        xp = model.xp
        tower = model.tower
        
        active_mask = tower._satellite_n_bindings > 0
        active_indices = xp.where(active_mask)[0]
        
        if len(active_indices) < 2:
            return 0.0
        
        coherences = []
        for _ in range(n_sample):
            pair_idx = np.random.choice(len(active_indices), size=2, replace=False)
            idx_a = int(active_indices[pair_idx[0]])
            idx_b = int(active_indices[pair_idx[1]])
            
            mem_a = tower._all_memories[idx_a]
            mem_b = tower._all_memories[idx_b]
            
            coh = float(frobenius_cosine(mem_a, mem_b, xp).get())
            coherences.append(coh)
        
        return float(np.mean(coherences)) if coherences else 0.0
    
    def aggressive_dmn(model, n_recombine=10):
        """Aggressive DMN: more recombinations, less survival testing."""
        xp = model.xp
        tower = model.tower
        
        active_mask = tower._satellite_n_bindings > 0
        active_indices = xp.where(active_mask)[0]
        
        if len(active_indices) < 2:
            return
        
        for _ in range(n_recombine):
            pair_idx = np.random.choice(len(active_indices), size=2, replace=False)
            idx_a = int(active_indices[pair_idx[0]])
            idx_b = int(active_indices[pair_idx[1]])
            
            mem_a = tower._all_memories[idx_a]
            mem_b = tower._all_memories[idx_b]
            
            # Aggressive: less survival test, more injection
            recombined = geometric_product(mem_a, mem_b)
            settled = grace_operator(recombined, model.basis, xp)  # Only 1 iteration
            
            # Inject with higher weight
            target_idx = np.random.randint(tower.n_satellites)
            tower._all_memories[target_idx] += 0.5 * settled  # Higher weight
    
    def moderate_dmn(model, n_recombine=5):
        """Moderate DMN: standard recombination."""
        xp = model.xp
        tower = model.tower
        
        active_mask = tower._satellite_n_bindings > 0
        active_indices = xp.where(active_mask)[0]
        
        if len(active_indices) < 2:
            return
        
        for _ in range(n_recombine):
            pair_idx = np.random.choice(len(active_indices), size=2, replace=False)
            idx_a = int(active_indices[pair_idx[0]])
            idx_b = int(active_indices[pair_idx[1]])
            
            mem_a = tower._all_memories[idx_a]
            mem_b = tower._all_memories[idx_b]
            
            recombined = geometric_product(mem_a, mem_b)
            
            # Standard survival test
            settled = recombined
            for _ in range(3):
                settled = grace_operator(settled, model.basis, xp)
            
            stability = float(frobenius_cosine(settled, recombined, xp).get())
            
            if stability > PHI_INV:
                low_mask = tower._satellite_n_bindings < 10
                low_indices = xp.where(low_mask)[0]
                if len(low_indices) > 0:
                    target_idx = int(low_indices[np.random.randint(len(low_indices))])
                    tower._all_memories[target_idx] += PHI_INV * settled
    
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
                # FAIL LOUDLY - don't hide errors
                raise RuntimeError(f"Similarity computation failed: {e}") from e
        
        if not similarities:
            raise RuntimeError("No similarities computed - all samples failed")
        return np.mean(similarities)
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING WITH COLLAPSE MONITORING")
    print("="*80)
    
    BATCH_SIZE = 2048
    N_BATCHES = 50
    EVAL_EVERY = 2
    
    # Detectors for each model
    detector_aggressive = CollapseDetector()
    detector_adaptive = CollapseDetector()
    
    metrics = {
        'standard': {'sim': []},
        'moderate': {'sim': []},
        'aggressive': {'sim': [], 'collapse_detected': []},
        'adaptive': {'sim': [], 'dmn_throttled': 0, 'collapse_detected': []},
    }
    
    prev_witness_aggressive = None
    prev_witness_adaptive = None
    dmn_throttled = False
    
    eval_samples = samples[N_BATCHES * BATCH_SIZE:]
    
    for batch_idx in range(N_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        batch = samples[start_idx:start_idx + BATCH_SIZE]
        
        if not batch:
            break
        
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        
        # Standard
        model_standard.tower.learn_batch(contexts, targets)
        model_standard.n_patterns += len(contexts)
        
        # Moderate DMN (every 3)
        model_moderate.tower.learn_batch(contexts, targets)
        model_moderate.n_patterns += len(contexts)
        if (batch_idx + 1) % 3 == 0:
            moderate_dmn(model_moderate)
        
        # Aggressive DMN (EVERY batch)
        model_aggressive.tower.learn_batch(contexts, targets)
        model_aggressive.n_patterns += len(contexts)
        aggressive_dmn(model_aggressive)  # Every batch!
        
        # Adaptive with collapse detection
        model_adaptive.tower.learn_batch(contexts, targets)
        model_adaptive.n_patterns += len(contexts)
        
        # Evaluation and collapse detection
        if (batch_idx + 1) % EVAL_EVERY == 0:
            sim_standard = evaluate_model(model_standard, eval_samples)
            sim_moderate = evaluate_model(model_moderate, eval_samples)
            sim_aggressive = evaluate_model(model_aggressive, eval_samples)
            sim_adaptive = evaluate_model(model_adaptive, eval_samples)
            
            # Update aggressive detector
            churn_agg, prev_witness_aggressive = compute_witness_churn(model_aggressive, prev_witness_aggressive)
            entropy_agg = compute_satellite_entropy(model_aggressive)
            coherence_agg = compute_cross_satellite_coherence(model_aggressive, n_sample=10)
            detector_aggressive.update(churn_agg, sim_aggressive, entropy_agg, coherence_agg)
            collapse_agg = detector_aggressive.detect_collapse()
            
            # Update adaptive detector
            churn_ada, prev_witness_adaptive = compute_witness_churn(model_adaptive, prev_witness_adaptive)
            entropy_ada = compute_satellite_entropy(model_adaptive)
            coherence_ada = compute_cross_satellite_coherence(model_adaptive, n_sample=10)
            detector_adaptive.update(churn_ada, sim_adaptive, entropy_ada, coherence_ada)
            collapse_ada = detector_adaptive.detect_collapse()
            
            # Adaptive DMN with throttling
            if not collapse_ada['is_collapsing'] and not dmn_throttled:
                moderate_dmn(model_adaptive)
            elif collapse_ada['is_collapsing']:
                # THROTTLE: detected collapse risk, suppress DMN
                metrics['adaptive']['dmn_throttled'] += 1
                dmn_throttled = True
            else:
                # Recovery: resume DMN after danger passes
                dmn_throttled = False
            
            metrics['standard']['sim'].append(sim_standard)
            metrics['moderate']['sim'].append(sim_moderate)
            metrics['aggressive']['sim'].append(sim_aggressive)
            metrics['aggressive']['collapse_detected'].append(collapse_agg)
            metrics['adaptive']['sim'].append(sim_adaptive)
            metrics['adaptive']['collapse_detected'].append(collapse_ada)
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Standard:   sim={sim_standard:.4f}")
            print(f"    Moderate:   sim={sim_moderate:.4f}")
            print(f"    Aggressive: sim={sim_aggressive:.4f} {'⚠️ COLLAPSING' if collapse_agg['is_collapsing'] else ''}")
            print(f"    Adaptive:   sim={sim_adaptive:.4f} (throttled={metrics['adaptive']['dmn_throttled']})")
            
            if collapse_agg['signatures']:
                print(f"    ⚠️ Aggressive collapse signatures: {collapse_agg['signatures'][:3]}")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("OVER-DREAMING COLLAPSE RESULTS")
    print("="*80)
    
    def get_late_avg(lst):
        return float(np.mean(lst[-5:])) if len(lst) >= 5 else float(np.mean(lst))
    
    late_standard = get_late_avg(metrics['standard']['sim'])
    late_moderate = get_late_avg(metrics['moderate']['sim'])
    late_aggressive = get_late_avg(metrics['aggressive']['sim'])
    late_adaptive = get_late_avg(metrics['adaptive']['sim'])
    
    # Count collapse detections
    n_agg_collapse = sum(1 for c in metrics['aggressive']['collapse_detected'] if c['is_collapsing'])
    n_ada_collapse = sum(1 for c in metrics['adaptive']['collapse_detected'] if c['is_collapsing'])
    
    results = {
        'late_stage': {
            'standard': late_standard,
            'moderate': late_moderate,
            'aggressive': late_aggressive,
            'adaptive': late_adaptive,
        },
        'collapse_counts': {
            'aggressive': n_agg_collapse,
            'adaptive': n_ada_collapse,
        },
        'adaptive_throttle_count': metrics['adaptive']['dmn_throttled'],
        'curves': {
            'sim': {k: v['sim'] for k, v in metrics.items()},
        },
    }
    
    print(f"""
    LATE-STAGE SEMANTIC SIMILARITY:
      Standard:   {late_standard:.4f} (control)
      Moderate:   {late_moderate:.4f} ({(late_moderate/late_standard-1)*100:+.2f}%)
      Aggressive: {late_aggressive:.4f} ({(late_aggressive/late_standard-1)*100:+.2f}%)
      Adaptive:   {late_adaptive:.4f} ({(late_adaptive/late_standard-1)*100:+.2f}%)
      
    COLLAPSE DETECTION:
      Aggressive: {n_agg_collapse} collapse events detected
      Adaptive:   {n_ada_collapse} collapse events detected
      Throttle count: {metrics['adaptive']['dmn_throttled']}
      
    KEY FINDINGS:
      Aggressive collapsed: {late_aggressive < late_moderate}
      Adaptive avoided collapse: {late_adaptive >= late_moderate}
      Detection worked: {n_agg_collapse > n_ada_collapse or metrics['adaptive']['dmn_throttled'] > 0}
    """)
    
    if late_aggressive < late_moderate and late_adaptive >= late_moderate:
        print("    ✅ COLLAPSE DETECTION WORKS")
        print("       - Aggressive DMN caused collapse")
        print("       - Adaptive throttling prevented it")
    elif late_aggressive >= late_moderate:
        print("    🔄 NO COLLAPSE OBSERVED")
        print("       - May need more aggressive parameters")
    else:
        print("    ⚠️ COLLAPSE DETECTED BUT NOT PREVENTED")
        print("       - Detection thresholds may need tuning")
    
    # Save
    results_path = "/checkpoints/overdreaming_collapse_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


@app.local_entrypoint()
def main():
    print("Running Over-Dreaming Collapse Test on Modal H100...")
    result = test_overdreaming_collapse.remote()
    print(f"\nAdaptive vs Aggressive: {result['late_stage']['adaptive']:.4f} vs {result['late_stage']['aggressive']:.4f}")
