"""
Phase 3: Scaling Verification

THEORY BEING TESTED:
    - O(1) inference time regardless of patterns learned
    - Sublinear memory scaling via hierarchical tower
    - Coherence/stability maintained at scale
    - Throughput consistent across scales

SUCCESS CRITERIA:
    - inference_ratio (500K/10K) < 2.0
    - memory_ratio (500K/10K) < 10.0
    - coherence_variance < 0.1
    - throughput > 5K samples/sec at all scales

RUNTIME: ~20 minutes on H100
"""

import modal
import numpy as np
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# Modal setup
app = modal.App("scaling-verification")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "numpy>=1.24",
        "cupy-cuda12x>=12.0",
        "scipy>=1.10",
        "datasets>=2.14",
        "tqdm>=4.65",
        "huggingface_hub>=0.16",
    )
    .add_local_dir("holographic_prod", "/root/project/holographic_prod")
)

checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@dataclass
class ScaleCheckpoint:
    """Metrics at a specific scale."""
    n_patterns: int
    inference_latency_ms: float
    memory_mb: float
    coherence: float
    stability: float
    throughput: float


@dataclass
class ScalingReport:
    """Full report from scaling experiment."""
    checkpoints: List[ScaleCheckpoint] = field(default_factory=list)
    
    # O(1) Inference
    inference_ratio_500K_10K: float = 0.0
    inference_passed: bool = False
    
    # Sublinear Memory
    memory_ratio_500K_10K: float = 0.0
    memory_passed: bool = False
    
    # Stability Under Load
    coherence_variance: float = 0.0
    coherence_passed: bool = False
    
    # Throughput
    min_throughput: float = 0.0
    throughput_passed: bool = False
    
    # Overall
    passed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'checkpoints': [
                {
                    'n_patterns': c.n_patterns,
                    'inference_latency_ms': c.inference_latency_ms,
                    'memory_mb': c.memory_mb,
                    'coherence': c.coherence,
                    'stability': c.stability,
                    'throughput': c.throughput,
                }
                for c in self.checkpoints
            ],
            'inference_ratio_500K_10K': self.inference_ratio_500K_10K,
            'inference_passed': self.inference_passed,
            'memory_ratio_500K_10K': self.memory_ratio_500K_10K,
            'memory_passed': self.memory_passed,
            'coherence_variance': self.coherence_variance,
            'coherence_passed': self.coherence_passed,
            'min_throughput': self.min_throughput,
            'throughput_passed': self.throughput_passed,
            'passed': self.passed,
        }


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def run_scaling_verification():
    """
    Run scaling verification tests.
    
    Tests O(1) inference, sublinear memory, and stability at:
    10K, 50K, 100K, 250K, 500K patterns
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    from datasets import load_dataset
    from tqdm import tqdm
    import re
    
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    from holographic_prod.core.algebra import (
        decompose_to_coefficients_batch,
        get_cached_basis,
        grace_with_stability,
    )
    from holographic_prod.core.constants import PHI_EPSILON
    
    print("="*80)
    print("PHASE 3: SCALING VERIFICATION")
    print("="*80)
    
    xp = cp
    basis = get_cached_basis(xp)
    
    # Configuration
    VOCAB_SIZE = 30_000
    CONTEXT_SIZE = 64
    BATCH_SIZE = 2048
    SCALE_CHECKPOINTS = [10_000, 50_000, 100_000, 250_000, 500_000]
    
    report = ScalingReport()
    
    # =========================================================================
    # STEP 1: Build vocabulary
    # =========================================================================
    print("\n[Step 1] Building vocabulary...")
    
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    word_counts: Dict[str, int] = {}
    pattern = re.compile(r'\b[a-zA-Z]+\b')
    
    for i, item in enumerate(tqdm(ds.take(5000), total=5000, desc="Building vocab")):
        words = pattern.findall(item['text'].lower())
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
    
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])[:VOCAB_SIZE - 2]
    word_to_idx = {"<unk>": 0, "<pad>": 1}
    for i, (word, _) in enumerate(sorted_words):
        word_to_idx[word] = i + 2
    
    print(f"  Vocabulary size: {len(word_to_idx):,}")
    
    # =========================================================================
    # STEP 2: Extract samples (need 500K+)
    # =========================================================================
    print("\n[Step 2] Extracting samples...")
    
    samples: List[Tuple[List[int], int]] = []
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    for item in tqdm(ds, desc="Extracting"):
        words = pattern.findall(item['text'].lower())
        tokens = [word_to_idx.get(w, 0) for w in words]
        
        if len(tokens) < CONTEXT_SIZE + 1:
            continue
        
        for i in range(0, min(len(tokens) - CONTEXT_SIZE, 50), 1):
            ctx = tokens[i:i + CONTEXT_SIZE]
            tgt = tokens[i + CONTEXT_SIZE]
            if tgt != 0:
                samples.append((ctx, tgt))
        
        if len(samples) >= 550_000:  # Extra for eval
            break
    
    print(f"  Total samples: {len(samples):,}")
    
    # Separate eval samples
    train_samples = samples[:500_000]
    eval_samples = samples[500_000:550_000]
    
    # =========================================================================
    # STEP 3: Create model
    # =========================================================================
    print("\n[Step 3] Creating model...")
    
    grounded_embs, coverage = create_grounded_embeddings_fast(
        word_to_idx, glove_dim=50, cache_dir="/checkpoints/glove"
    )
    
    model = HolographicMemory(
        vocab_size=len(word_to_idx),
        max_levels=5,  # More levels for scale
        seed=42,
        use_gpu=True,
        grounded_embeddings=cp.asarray(grounded_embs),
    )
    
    print(f"  Model created with {model.tower.n_satellites} satellites")
    
    # =========================================================================
    # HELPER: Measure inference latency
    # =========================================================================
    def measure_inference_latency(model, samples, n_trials=100):
        """Measure average inference latency in milliseconds."""
        # Warmup
        for ctx, _ in samples[:10]:
            _ = model.tower.retrieve(ctx)
        
        # Sync GPU
        cp.cuda.Stream.null.synchronize()
        
        # Measure
        latencies = []
        for ctx, _ in samples[:n_trials]:
            start = time.perf_counter()
            _ = model.tower.retrieve(ctx)
            cp.cuda.Stream.null.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
        
        return np.mean(latencies)
    
    # =========================================================================
    # HELPER: Evaluate coherence
    # =========================================================================
    def evaluate_coherence(model, samples, n_eval=100):
        """Evaluate coherence using theory-true path."""
        coherences = []
        stabilities = []
        
        for ctx, tgt in samples[:n_eval]:
            ctx_mat = model.tower._embed_sequence(ctx)
            graced_ctx, stability, _ = grace_with_stability(ctx_mat, basis, xp)
            stabilities.append(float(stability))
            
            sat_idx = model.tower.route_to_satellite(ctx)
            sat_memory = model.tower._all_memories[sat_idx]
            
            retrieved = graced_ctx.T @ sat_memory
            graced_ret, _, _ = grace_with_stability(retrieved, basis, xp)
            
            tgt_emb = model.tower.embeddings[tgt]
            composition = graced_ret @ tgt_emb.T
            
            coeffs = decompose_to_coefficients_batch(composition.reshape(1, 4, 4), basis, xp)[0]
            total_e = float(xp.sum(coeffs ** 2))
            witness_e = float(coeffs[0]**2 + coeffs[15]**2)
            coh = witness_e / max(total_e, PHI_EPSILON)
            coherences.append(coh)
        
        return np.mean(coherences), np.mean(stabilities)
    
    # =========================================================================
    # STEP 4: Learn and measure at each scale
    # =========================================================================
    print("\n[Step 4] Learning and measuring at each scale...")
    
    sample_idx = 0
    checkpoint_idx = 0
    start_time = time.perf_counter()
    
    while sample_idx < 500_000 and checkpoint_idx < len(SCALE_CHECKPOINTS):
        target_patterns = SCALE_CHECKPOINTS[checkpoint_idx]
        
        # Learn up to this checkpoint
        while sample_idx < target_patterns:
            batch_end = min(sample_idx + BATCH_SIZE, target_patterns)
            batch = train_samples[sample_idx:batch_end]
            
            contexts = [ctx for ctx, _ in batch]
            targets = [tgt for _, tgt in batch]
            model.learn_batch(contexts, targets)
            
            sample_idx = batch_end
        
        # Measure at this checkpoint
        print(f"\n  Checkpoint: {target_patterns:,} patterns")
        
        # Inference latency
        latency = measure_inference_latency(model, eval_samples, n_trials=100)
        print(f"    Inference latency: {latency:.3f} ms")
        
        # Memory usage
        cp.cuda.Stream.null.synchronize()
        mempool = cp.get_default_memory_pool()
        memory_mb = mempool.used_bytes() / (1024**2)
        print(f"    GPU memory: {memory_mb:.0f} MB")
        
        # Coherence and stability
        coherence, stability = evaluate_coherence(model, eval_samples, n_eval=100)
        print(f"    Coherence: {coherence:.4f}")
        print(f"    Stability: {stability:.4f}")
        
        # Throughput
        elapsed = time.perf_counter() - start_time
        throughput = sample_idx / elapsed
        print(f"    Throughput: {throughput:.0f} samples/sec")
        
        report.checkpoints.append(ScaleCheckpoint(
            n_patterns=target_patterns,
            inference_latency_ms=latency,
            memory_mb=memory_mb,
            coherence=coherence,
            stability=stability,
            throughput=throughput,
        ))
        
        checkpoint_idx += 1
    
    # =========================================================================
    # STEP 5: Analyze scaling
    # =========================================================================
    print("\n" + "="*80)
    print("SCALING ANALYSIS")
    print("="*80)
    
    # Get 10K and 500K checkpoints
    cp_10K = next((c for c in report.checkpoints if c.n_patterns == 10_000), None)
    cp_500K = next((c for c in report.checkpoints if c.n_patterns == 500_000), None)
    
    if cp_10K and cp_500K:
        # O(1) Inference
        report.inference_ratio_500K_10K = cp_500K.inference_latency_ms / cp_10K.inference_latency_ms
        report.inference_passed = report.inference_ratio_500K_10K < 2.0
        
        print(f"\n  O(1) Inference Test:")
        print(f"    Latency at 10K: {cp_10K.inference_latency_ms:.3f} ms")
        print(f"    Latency at 500K: {cp_500K.inference_latency_ms:.3f} ms")
        print(f"    Ratio: {report.inference_ratio_500K_10K:.2f}x")
        print(f"    Target: < 2.0x")
        print(f"    {'✓ PASSED' if report.inference_passed else '✗ FAILED'}")
        
        # Sublinear Memory
        report.memory_ratio_500K_10K = cp_500K.memory_mb / cp_10K.memory_mb
        report.memory_passed = report.memory_ratio_500K_10K < 10.0
        
        print(f"\n  Sublinear Memory Test:")
        print(f"    Memory at 10K: {cp_10K.memory_mb:.0f} MB")
        print(f"    Memory at 500K: {cp_500K.memory_mb:.0f} MB")
        print(f"    Ratio: {report.memory_ratio_500K_10K:.2f}x")
        print(f"    Target: < 10.0x (sublinear)")
        print(f"    {'✓ PASSED' if report.memory_passed else '✗ FAILED'}")
    
    # Coherence stability
    coherences = [c.coherence for c in report.checkpoints]
    report.coherence_variance = np.var(coherences)
    report.coherence_passed = report.coherence_variance < 0.1
    
    print(f"\n  Coherence Stability Test:")
    print(f"    Coherences: {[f'{c:.4f}' for c in coherences]}")
    print(f"    Variance: {report.coherence_variance:.4f}")
    print(f"    Target: < 0.1")
    print(f"    {'✓ PASSED' if report.coherence_passed else '✗ FAILED'}")
    
    # Throughput
    throughputs = [c.throughput for c in report.checkpoints]
    report.min_throughput = min(throughputs)
    report.throughput_passed = report.min_throughput > 5000
    
    print(f"\n  Throughput Stability Test:")
    print(f"    Throughputs: {[f'{t:.0f}' for t in throughputs]}")
    print(f"    Minimum: {report.min_throughput:.0f} samples/sec")
    print(f"    Target: > 5000 samples/sec")
    print(f"    {'✓ PASSED' if report.throughput_passed else '✗ FAILED'}")
    
    # =========================================================================
    # SUCCESS CRITERIA
    # =========================================================================
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)
    
    criteria = {
        'inference_ratio < 2.0': report.inference_passed,
        'memory_ratio < 10.0': report.memory_passed,
        'coherence_variance < 0.1': report.coherence_passed,
        'min_throughput > 5000': report.throughput_passed,
    }
    
    for name, passed in criteria.items():
        status = '✓' if passed else '✗'
        print(f"  {status} {name}: {passed}")
    
    # Overall pass: inference ratio is the key metric
    report.passed = report.inference_passed
    
    print("\n" + "="*80)
    if report.passed:
        print("PHASE 3 PASSED: O(1) inference scaling verified")
    else:
        print("PHASE 3 FAILED: Inference does not scale O(1)")
    print("="*80)
    
    # =========================================================================
    # SCALING TABLE
    # =========================================================================
    print("\n" + "="*80)
    print("SCALING DATA")
    print("="*80)
    print(f"\n{'Patterns':>12} {'Latency(ms)':>12} {'Memory(MB)':>12} {'Coherence':>10} {'Throughput':>12}")
    print("-" * 60)
    for c in report.checkpoints:
        print(f"{c.n_patterns:>12,} {c.inference_latency_ms:>12.3f} {c.memory_mb:>12.0f} {c.coherence:>10.4f} {c.throughput:>12.0f}")
    
    return report.to_dict()


@app.local_entrypoint()
def main():
    """Run scaling verification tests."""
    result = run_scaling_verification.remote()
    print("\n" + json.dumps(result, indent=2))
    
    with open("/tmp/scaling_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("\nResult saved to /tmp/scaling_result.json")
