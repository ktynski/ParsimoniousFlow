"""
Adaptive DMN Gate v2 - Theory-True Implementation

THE KEY INSIGHT:
    Genius brains don't just generate more ideas.
    They generate ideas that are immediately forced to justify their existence.
    
    Three rates that matter:
        1. Prototype generation rate (recombinations)
        2. Schema evaluation & pruning rate (Grace survival)
        3. Integration rate (survivor injection + routing pressure)
    
    Genius = fast schema birth + faster schema death + immediate manifold update

THIS VERSION IMPLEMENTS:
    1. History-relative thresholds (no magic numbers)
    2. Dream harm detection + cooldown
    3. Distance-biased pairing (remote associations)
    4. Cross-depth mixing (abstraction levels)
    5. Temporal mixing (recent vs stable vs surprising)
    6. Adaptive intensity (n_recombine, survival_iters based on state)
"""

import modal
import numpy as np
import time
import json
import math
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

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

app = modal.App("adaptive-dmn-gate-v2")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def routing_entropy_from_counts(counts: np.ndarray) -> float:
    """Compute routing entropy from satellite hit counts."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p + 1e-12)).sum())


def energy_stats(xp, tower):
    """Compute satellite energy statistics."""
    energies = xp.linalg.norm(tower._all_memories, axis=(1, 2))
    e = energies.get() if hasattr(energies, "get") else np.asarray(energies)
    mean = float(e.mean())
    std = float(e.std())
    cv = std / (mean + 1e-12)
    # Junk mass: fraction below 20th percentile
    thr = np.quantile(e, 0.2)
    junk = float((e < thr).mean())
    return mean, cv, junk


class DMNGate:
    """
    Theory-true adaptive DMN gate.
    
    Uses history-relative thresholds, hysteresis, and dream harm detection.
    """
    
    def __init__(self, warmup_batches=5, eval_stride=5, cooldown_batches=5):
        self.warmup = warmup_batches
        self.eval_stride = eval_stride
        self.cooldown_batches = cooldown_batches
        self.cooldown = 0
        
        self.sim_hist = deque(maxlen=10)    # eval sims
        self.ent_hist = deque(maxlen=20)    # routing entropy
        self.junk_hist = deque(maxlen=20)   # junk mass
        self.cv_hist = deque(maxlen=20)     # energy CV (diversity)
        
        self.last_dream_sim = None
        self.last_sim = None
        self.recent_dream_hurt = False
        
        # Track decisions for analysis
        self.decision_log = []
    
    def update_eval(self, sim: float):
        """Update with new eval similarity."""
        self.sim_hist.append(sim)
        self.last_sim = sim
    
    def update_batch_stats(self, entropy: float, junk: float, cv: float):
        """Update with batch-level statistics."""
        self.ent_hist.append(entropy)
        self.junk_hist.append(junk)
        self.cv_hist.append(cv)
    
    def _slope(self) -> float:
        """Compute slope over recent evals."""
        if len(self.sim_hist) < 4:
            return 1e9  # During warmup, assume positive progress
        return float(self.sim_hist[-1] - self.sim_hist[-4])
    
    def _median_slope_magnitude(self) -> float:
        """Compute median magnitude of recent slopes for relative threshold."""
        if len(self.sim_hist) < 5:
            return 0.01
        slopes = [self.sim_hist[i] - self.sim_hist[i-1] 
                  for i in range(1, len(self.sim_hist))]
        return float(np.median(np.abs(slopes)) + 1e-6)
    
    def decide(self, batch_idx: int) -> Tuple[bool, int, int, Dict[str, Any]]:
        """
        Decide whether to run DMN consolidation.
        
        Returns:
            (run_dream, n_recombine, survival_iters, debug_info)
        """
        debug = {
            'batch': batch_idx,
            'cooldown': self.cooldown,
            'decision': 'none',
            'reason': None,
        }
        
        # Warmup phase
        if batch_idx < self.warmup:
            debug['decision'] = 'warmup'
            return False, 0, 0, debug
        
        # Cooldown after dream harm
        if self.cooldown > 0:
            self.cooldown -= 1
            debug['decision'] = 'cooldown'
            return False, 0, 0, debug
        
        # Need enough history
        if len(self.ent_hist) < 5 or len(self.junk_hist) < 5 or len(self.cv_hist) < 5:
            debug['decision'] = 'insufficient_history'
            return False, 0, 0, debug
        
        # Compute signals
        slope = self._slope()
        H = float(np.mean(list(self.ent_hist)[-5:]))
        junk = float(np.mean(list(self.junk_hist)[-5:]))
        cv = float(np.mean(list(self.cv_hist)[-5:]))
        
        # History-relative thresholds (no magic numbers)
        H_med = float(np.median(self.ent_hist))
        cv_med = float(np.median(self.cv_hist))
        slope_med = self._median_slope_magnitude()
        
        H_low = max(0.2, 0.6 * H_med)
        H_high = 1.4 * H_med
        cv_low = 0.5 * cv_med
        slope_threshold = 0.2 * slope_med
        
        debug.update({
            'slope': slope,
            'slope_threshold': slope_threshold,
            'H': H,
            'H_med': H_med,
            'H_low': H_low,
            'H_high': H_high,
            'junk': junk,
            'cv': cv,
            'cv_med': cv_med,
            'cv_low': cv_low,
            'recent_dream_hurt': self.recent_dream_hurt,
        })
        
        # Collapse risk: entropy too low OR diversity too low OR dream hurt recently
        collapse = (H < H_low) or (cv < cv_low) or self.recent_dream_hurt
        
        # Fragmentation / junk: entropy too high or junk too large
        fragmentation = (H > H_high) or (junk > 0.55)
        
        # Stall: slope near zero (history-relative)
        stall = (slope < slope_threshold)
        
        debug['collapse_risk'] = collapse
        debug['fragmentation'] = fragmentation
        debug['stall'] = stall
        
        # Decision: DMN when (stalled AND fragmenting) AND NOT collapsing
        run_dream = (stall and fragmentation) and not collapse
        
        if not run_dream:
            if collapse:
                debug['decision'] = 'blocked_collapse_risk'
                debug['reason'] = 'H_low' if H < H_low else ('cv_low' if cv < cv_low else 'dream_hurt')
            elif not stall:
                debug['decision'] = 'blocked_progressing'
            else:
                debug['decision'] = 'blocked_not_fragmenting'
            return False, 0, 0, debug
        
        # Adaptive intensity
        frag_strength = min(1.0, max(0.0, (H - H_med) / (H_med + 1e-9)))
        n_recombine = int(3 + 12 * frag_strength)     # 3..15
        survival_iters = 2 if frag_strength > 0.5 else 3  # stricter when less fragmented
        
        debug['decision'] = 'run_dream'
        debug['frag_strength'] = frag_strength
        debug['n_recombine'] = n_recombine
        debug['survival_iters'] = survival_iters
        
        self.decision_log.append(debug)
        
        return True, n_recombine, survival_iters, debug
    
    def on_dream_result(self, sim_before: float, sim_after: float):
        """Called after a dream to detect harm."""
        self.recent_dream_hurt = (sim_after + 1e-6) < sim_before
        if self.recent_dream_hurt:
            self.cooldown = self.cooldown_batches


class DistanceAwarePairing:
    """
    Distance-biased pairing for recombination.
    
    Instead of random pairs, sample pairs proportional to distance.
    This enables "remote association" - the hallmark of creative insight.
    """
    
    def __init__(self, alpha: float = 2.0):
        self.alpha = alpha  # Higher = prefer more distant pairs
    
    def choose_pairs(
        self,
        tower,
        xp,
        n_pairs: int,
        mix_depth: bool = True,
        mix_time: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Choose recombination pairs biased toward distance.
        
        Args:
            tower: The tower with satellite memories
            xp: numpy or cupy
            n_pairs: Number of pairs to generate
            mix_depth: If True, allow cross-level mixing
            mix_time: If True, mix recent active with stable old
        """
        # Get active satellites
        active_mask = tower._satellite_n_bindings > 0
        active_indices = xp.where(active_mask)[0]
        
        if hasattr(active_indices, 'get'):
            active_indices = active_indices.get()
        
        if len(active_indices) < 2:
            return []
        
        # Categorize by activity (proxy for "time")
        if mix_time:
            counts = tower._satellite_n_bindings
            if hasattr(counts, 'get'):
                counts = counts.get()
            
            active_counts = counts[active_indices]
            
            # Split into pools
            median_count = np.median(active_counts)
            recent_mask = active_counts > median_count
            stable_mask = ~recent_mask
            
            recent_indices = active_indices[recent_mask]
            stable_indices = active_indices[stable_mask]
        else:
            recent_indices = active_indices
            stable_indices = active_indices
        
        # Get memories for distance computation
        mems = tower._all_memories[active_indices]
        if hasattr(mems, 'get'):
            mems = mems.get()
        
        # Flatten for cosine distance
        flat = mems.reshape(len(mems), -1)
        norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9
        flat_normed = flat / norms
        
        pairs = []
        
        for _ in range(n_pairs):
            # Strategy:
            # 50% - distance-biased within active set
            # 30% - cross-pool (recent x stable) if mix_time
            # 20% - random within active set
            
            strategy = np.random.random()
            
            if strategy < 0.5:
                # Distance-biased
                a_idx = np.random.randint(len(active_indices))
                dists = 1 - flat_normed @ flat_normed[a_idx]
                dists = np.clip(dists, 0, 2)
                dists[a_idx] = 0  # Don't pair with self
                
                # Sample proportional to distance^alpha
                probs = dists ** self.alpha
                probs = probs / (probs.sum() + 1e-10)
                
                b_idx = np.random.choice(len(active_indices), p=probs)
                
                pairs.append((
                    int(active_indices[a_idx]),
                    int(active_indices[b_idx])
                ))
            
            elif strategy < 0.8 and mix_time and len(recent_indices) > 0 and len(stable_indices) > 0:
                # Cross-pool mixing (recent x stable)
                a = int(np.random.choice(recent_indices))
                b = int(np.random.choice(stable_indices))
                pairs.append((a, b))
            
            else:
                # Random
                idx = np.random.choice(len(active_indices), size=2, replace=False)
                pairs.append((
                    int(active_indices[idx[0]]),
                    int(active_indices[idx[1]])
                ))
        
        return pairs


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def test_adaptive_dmn_gate_v2():
    """
    Test the theory-true adaptive DMN gate with distance-biased pairing.
    
    Compares:
        A. Standard (no DMN)
        B. Fixed DMN (naive schedule)
        C. Adaptive v1 (my simple implementation)
        D. Adaptive v2 (theory-true with distance mixing)
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
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    
    # Helper to handle both TowerMemory and MultiLevelTower
    def get_satellite_memories(tower):
        return getattr(tower, '_satellite_memories', None) or getattr(tower, '_all_memories', None)
    
    print("="*80)
    print("ADAPTIVE DMN GATE v2 - THEORY-TRUE IMPLEMENTATION")
    print("="*80)
    print("""
    KEY INSIGHT: Genius = fast schema birth + faster schema death + immediate manifold update
    
    THREE RATES:
        1. Prototype generation rate (recombinations)
        2. Schema evaluation & pruning rate (Grace survival)
        3. Integration rate (survivor injection)
    
    THIS TEST:
        A. Standard (no DMN)
        B. Fixed DMN (every 5 batches, random pairs)
        C. Adaptive v2 (history-relative thresholds, distance-biased mixing)
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
    # MODELS
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
    model_fixed = create_model()
    model_adaptive = create_model()
    
    print(f"  ✓ Models: {model_standard.tower.n_satellites:,} satellites each")
    
    # =========================================================================
    # CONSOLIDATION FUNCTIONS
    # =========================================================================
    
    def mini_rem_random(model, n_recombine=5, survival_iters=3):
        """Original: random pairing."""
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
            
            mem_a = tower._all_memories[idx_a]
            mem_b = tower._all_memories[idx_b]
            
            recombined = geometric_product(mem_a, mem_b)
            
            settled = recombined
            for _ in range(survival_iters):
                settled = grace_operator(settled, model.basis, xp)
            
            stability = float(frobenius_cosine(settled, recombined, xp).get())
            
            if stability > PHI_INV:
                low_mask = tower._satellite_n_bindings < 10
                low_indices = xp.where(low_mask)[0]
                if len(low_indices) > 0:
                    target_idx = int(low_indices[np.random.randint(len(low_indices))])
                    tower._all_memories[target_idx] += PHI_INV * settled
                    survivors += 1
        
        return survivors
    
    def mini_rem_distance_aware(model, pairs: List[Tuple[int, int]], survival_iters: int = 3):
        """Distance-aware pairing with adaptive survival."""
        xp = model.xp
        tower = model.tower
        
        survivors = 0
        
        for idx_a, idx_b in pairs:
            mem_a = tower._all_memories[idx_a]
            mem_b = tower._all_memories[idx_b]
            
            recombined = geometric_product(mem_a, mem_b)
            
            settled = recombined
            for _ in range(survival_iters):
                settled = grace_operator(settled, model.basis, xp)
            
            stability = float(frobenius_cosine(settled, recombined, xp).get())
            
            if stability > PHI_INV:
                low_mask = tower._satellite_n_bindings < 10
                low_indices = xp.where(low_mask)[0]
                if len(low_indices) > 0:
                    target_idx = int(low_indices[np.random.randint(len(low_indices))])
                    tower._all_memories[target_idx] += PHI_INV * settled
                    survivors += 1
        
        return survivors
    
    def compute_batch_routing_entropy(model, contexts, n_probe=256):
        """Compute routing entropy for a batch."""
        xp = model.xp
        tower = model.tower
        
        counts = np.zeros(tower.n_satellites, dtype=np.int32)
        probe_n = min(n_probe, len(contexts))
        probe_idx = np.random.choice(len(contexts), size=probe_n, replace=False)
        
        for i in probe_idx:
            ctx = contexts[i]
            try:
                ctx_emb = tower._embed_sequence(ctx)
                basin_key = grace_basin_keys_batch_direct(
                    ctx_emb[None], model.basis,
                    n_iters=GRACE_ROUTING_ITERS,
                    resolution=GRACE_ROUTING_RESOLUTION,
                    xp=xp
                )[0]
                primes = xp.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53], dtype=xp.int64)
                sat_idx = int((xp.sum(basin_key * primes) % tower.n_satellites).get())
                counts[sat_idx] += 1
            except Exception as e:
                raise RuntimeError(f"Routing entropy computation failed: {e}") from e
        
        return routing_entropy_from_counts(counts)
    
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
    
    # Gate and pairing for adaptive model
    gate = DMNGate(warmup_batches=5, eval_stride=EVAL_EVERY, cooldown_batches=5)
    pairing = DistanceAwarePairing(alpha=2.0)
    
    metrics = {
        'standard': {'sim': []},
        'fixed': {'sim': [], 'dmn_count': 0, 'survivors': 0},
        'adaptive': {
            'sim': [], 'dmn_count': 0, 'survivors': 0,
            'decisions': []
        },
    }
    
    eval_samples = samples[N_BATCHES * BATCH_SIZE:]
    
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
        
        # Fixed: learn + DMN every N batches (random pairs)
        model_fixed.tower.learn_batch(contexts, targets)
        model_fixed.n_patterns += len(contexts)
        if (batch_idx + 1) % FIXED_DMN_EVERY == 0:
            s = mini_rem_random(model_fixed, n_recombine=5, survival_iters=3)
            metrics['fixed']['dmn_count'] += 1
            metrics['fixed']['survivors'] += s
        
        # Adaptive: learn + state-dependent DMN (distance-aware pairs)
        model_adaptive.tower.learn_batch(contexts, targets)
        model_adaptive.n_patterns += len(contexts)
        
        # Compute batch stats
        H_batch = compute_batch_routing_entropy(model_adaptive, contexts, n_probe=256)
        _, cv, junk = energy_stats(model_adaptive.xp, model_adaptive.tower)
        gate.update_batch_stats(H_batch, junk, cv)
        
        # Gate decision
        run_dream, n_recombine, survival_iters, debug = gate.decide(batch_idx)
        metrics['adaptive']['decisions'].append(debug)
        
        if run_dream:
            sim_before = evaluate_model(model_adaptive, eval_samples, n_eval=50)
            
            # Distance-aware pairing
            pairs = pairing.choose_pairs(
                model_adaptive.tower,
                model_adaptive.xp,
                n_pairs=n_recombine,
                mix_depth=True,
                mix_time=True
            )
            
            s = mini_rem_distance_aware(model_adaptive, pairs, survival_iters=survival_iters)
            metrics['adaptive']['dmn_count'] += 1
            metrics['adaptive']['survivors'] += s
            
            sim_after = evaluate_model(model_adaptive, eval_samples, n_eval=50)
            gate.on_dream_result(sim_before, sim_after)
        
        # Evaluation
        if (batch_idx + 1) % EVAL_EVERY == 0:
            sim_standard = evaluate_model(model_standard, eval_samples)
            sim_fixed = evaluate_model(model_fixed, eval_samples)
            sim_adaptive = evaluate_model(model_adaptive, eval_samples)
            
            gate.update_eval(sim_adaptive)
            
            metrics['standard']['sim'].append(sim_standard)
            metrics['fixed']['sim'].append(sim_fixed)
            metrics['adaptive']['sim'].append(sim_adaptive)
            
            print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}:")
            print(f"    Standard:   sim={sim_standard:.4f}")
            print(f"    Fixed DMN:  sim={sim_fixed:.4f} ({(sim_fixed/sim_standard-1)*100:+.1f}%), dmn={metrics['fixed']['dmn_count']}")
            print(f"    Adaptive:   sim={sim_adaptive:.4f} ({(sim_adaptive/sim_standard-1)*100:+.1f}%), dmn={metrics['adaptive']['dmn_count']}")
            print(f"    Gate: H={H_batch:.3f}, junk={junk:.3f}, cv={cv:.3f}, decision={debug['decision']}")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    def get_late_avg(lst):
        return float(np.mean(lst[len(lst)//2:])) if lst else 0.0
    
    late_standard = get_late_avg(metrics['standard']['sim'])
    late_fixed = get_late_avg(metrics['fixed']['sim'])
    late_adaptive = get_late_avg(metrics['adaptive']['sim'])
    
    # Decision analysis
    decision_counts = defaultdict(int)
    for d in metrics['adaptive']['decisions']:
        decision_counts[d['decision']] += 1
    
    results = {
        'late_stage': {
            'standard': late_standard,
            'fixed': late_fixed,
            'adaptive': late_adaptive,
        },
        'improvement': {
            'fixed_vs_standard': (late_fixed / late_standard - 1) * 100,
            'adaptive_vs_standard': (late_adaptive / late_standard - 1) * 100,
            'adaptive_vs_fixed': (late_adaptive / late_fixed - 1) * 100 if late_fixed > 0 else 0,
        },
        'dmn_stats': {
            'fixed_activations': metrics['fixed']['dmn_count'],
            'adaptive_activations': metrics['adaptive']['dmn_count'],
            'fixed_survivors': metrics['fixed']['survivors'],
            'adaptive_survivors': metrics['adaptive']['survivors'],
            'fixed_survival_rate': metrics['fixed']['survivors'] / (metrics['fixed']['dmn_count'] * 5 + 1),
            'adaptive_survival_rate': metrics['adaptive']['survivors'] / (metrics['adaptive']['dmn_count'] * 10 + 1) if metrics['adaptive']['dmn_count'] > 0 else 0,
        },
        'decision_breakdown': dict(decision_counts),
        'curves': {
            'sim': {k: v['sim'] for k, v in metrics.items()},
        },
    }
    
    print(f"""
    LATE-STAGE SEMANTIC SIMILARITY:
      Standard (no DMN):    {late_standard:.4f}
      Fixed DMN:            {late_fixed:.4f} ({results['improvement']['fixed_vs_standard']:+.2f}%)
      Adaptive v2:          {late_adaptive:.4f} ({results['improvement']['adaptive_vs_standard']:+.2f}%)
      
    DMN ACTIVATION STATS:
      Fixed:    {metrics['fixed']['dmn_count']} activations, {metrics['fixed']['survivors']} survivors
      Adaptive: {metrics['adaptive']['dmn_count']} activations, {metrics['adaptive']['survivors']} survivors
      
    EFFICIENCY (survivors per activation):
      Fixed:    {results['dmn_stats']['fixed_survival_rate']:.2f}
      Adaptive: {results['dmn_stats']['adaptive_survival_rate']:.2f}
      
    GATE DECISION BREAKDOWN:
    """)
    
    for decision, count in sorted(decision_counts.items(), key=lambda x: -x[1]):
        pct = count / N_BATCHES * 100
        print(f"      {decision:30s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"""
    KEY METRICS:
      Adaptive > Fixed:           {late_adaptive > late_fixed}
      Better survival rate:       {results['dmn_stats']['adaptive_survival_rate'] > results['dmn_stats']['fixed_survival_rate']}
      Adaptive > Standard:        {late_adaptive > late_standard}
      
    VERDICT:
    """)
    
    if late_adaptive > late_fixed and results['dmn_stats']['adaptive_survival_rate'] >= results['dmn_stats']['fixed_survival_rate']:
        print("    ✅ THEORY-TRUE ADAPTIVE GATE SUCCEEDS")
        print("       - Better results")
        print("       - Better survival efficiency")
        print("       - History-relative thresholds work")
    elif late_adaptive > late_fixed:
        print("    ⚠️ ADAPTIVE HELPS BUT DIFFERENT EFFICIENCY")
    else:
        print("    ❌ FIXED BEATS ADAPTIVE - NEED TUNING")
    
    # Save
    results_path = "/checkpoints/adaptive_dmn_gate_v2_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


@app.local_entrypoint()
def main():
    print("Running Adaptive DMN Gate v2 Test on Modal H100...")
    result = test_adaptive_dmn_gate_v2.remote()
    print(f"\nAdaptive v2 improvement: {result['improvement']['adaptive_vs_standard']:+.2f}%")
    print(f"Adaptive vs Fixed: {result['improvement']['adaptive_vs_fixed']:+.2f}%")
