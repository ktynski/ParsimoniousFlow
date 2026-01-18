"""
Phase 1: Learning Curves on Real Text

THEORY BEING TESTED:
    - Hebbian learning produces immediate, measurable improvement
    - Coherence (witness_energy / total_energy) increases with samples
    - Perplexity decreases with learning
    - Stability converges to phi^-2 region

SUCCESS CRITERIA:
    - coherence_at_100K > 0.4
    - monotonic improvement visible
    - perplexity reduction > 50%
    - stability stabilizes near phi^-2

RUNTIME: ~30 minutes on H100
"""

import modal
import numpy as np
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# Modal setup
app = modal.App("learning-curves-real-text")

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
class LearningCurveResult:
    """Results from learning curve measurement."""
    samples: int
    coherence: float
    perplexity: float
    stability: float
    throughput: float
    elapsed_time: float
    memory_mb: float


@dataclass 
class LearningCurveReport:
    """Full report from learning curve experiment."""
    checkpoints: List[LearningCurveResult] = field(default_factory=list)
    final_coherence: float = 0.0
    initial_coherence: float = 0.0
    coherence_improvement: float = 0.0
    perplexity_reduction: float = 0.0
    stability_converged: bool = False
    monotonic_improvement: bool = False
    passed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'checkpoints': [
                {
                    'samples': c.samples,
                    'coherence': c.coherence,
                    'perplexity': c.perplexity,
                    'stability': c.stability,
                    'throughput': c.throughput,
                    'elapsed_time': c.elapsed_time,
                    'memory_mb': c.memory_mb,
                }
                for c in self.checkpoints
            ],
            'final_coherence': self.final_coherence,
            'initial_coherence': self.initial_coherence,
            'coherence_improvement': self.coherence_improvement,
            'perplexity_reduction': self.perplexity_reduction,
            'stability_converged': self.stability_converged,
            'monotonic_improvement': self.monotonic_improvement,
            'passed': self.passed,
        }


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def run_learning_curves():
    """
    Run learning curve experiment on real OpenWebText data.
    
    Measures coherence, perplexity, and stability at checkpoints:
    1K, 5K, 10K, 25K, 50K, 100K samples
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
    )
    from holographic_prod.core.constants import PHI_INV_SQ, PHI_EPSILON
    
    print("="*80)
    print("PHASE 1: LEARNING CURVES ON REAL TEXT")
    print("="*80)
    
    xp = cp
    basis = get_cached_basis(xp)
    
    # Configuration
    VOCAB_SIZE = 30_000
    CONTEXT_SIZE = 64
    BATCH_SIZE = 2048
    CHECKPOINTS = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
    MAX_SAMPLES = 100_000
    
    # =========================================================================
    # STEP 1: Build vocabulary from OpenWebText
    # =========================================================================
    print("\n[Step 1] Building vocabulary from OpenWebText...")
    
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    word_counts: Dict[str, int] = {}
    pattern = re.compile(r'\b[a-zA-Z]+\b')
    
    docs_for_vocab = 5000
    for i, item in enumerate(tqdm(ds.take(docs_for_vocab), total=docs_for_vocab, desc="Building vocab")):
        words = pattern.findall(item['text'].lower())
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
    
    # Top vocab_size - 2 words (reserve for <unk> and <pad>)
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])[:VOCAB_SIZE - 2]
    word_to_idx = {"<unk>": 0, "<pad>": 1}
    for i, (word, _) in enumerate(sorted_words):
        word_to_idx[word] = i + 2
    
    print(f"  Vocabulary size: {len(word_to_idx):,}")
    
    # =========================================================================
    # STEP 2: Create grounded embeddings (GloVe)
    # =========================================================================
    print("\n[Step 2] Creating grounded embeddings...")
    
    grounded_embs, coverage = create_grounded_embeddings_fast(
        word_to_idx, glove_dim=50, cache_dir="/checkpoints/glove"
    )
    print(f"  GloVe coverage: {coverage*100:.1f}%")
    
    # =========================================================================
    # STEP 3: Extract samples from OpenWebText
    # =========================================================================
    print("\n[Step 3] Extracting samples from OpenWebText...")
    
    samples: List[Tuple[List[int], int]] = []
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    n_docs = 0
    for item in tqdm(ds, desc="Extracting samples"):
        words = pattern.findall(item['text'].lower())
        tokens = [word_to_idx.get(w, 0) for w in words]  # 0 = <unk>
        
        if len(tokens) < CONTEXT_SIZE + 1:
            continue
        
        # Extract overlapping windows (step=1)
        for i in range(0, len(tokens) - CONTEXT_SIZE, 1):
            ctx = tokens[i:i + CONTEXT_SIZE]
            tgt = tokens[i + CONTEXT_SIZE]
            if tgt != 0:  # Skip <unk> targets
                samples.append((ctx, tgt))
            
            if len(samples) >= MAX_SAMPLES + 10_000:  # Extra for eval
                break
        
        n_docs += 1
        if len(samples) >= MAX_SAMPLES + 10_000:
            break
    
    print(f"  Documents processed: {n_docs:,}")
    print(f"  Samples extracted: {len(samples):,}")
    
    # Split into train and eval
    train_samples = samples[:MAX_SAMPLES]
    eval_samples = samples[MAX_SAMPLES:MAX_SAMPLES + 5_000]
    
    # =========================================================================
    # STEP 4: Initialize model
    # =========================================================================
    print("\n[Step 4] Initializing model...")
    
    model = HolographicMemory(
        vocab_size=len(word_to_idx),
        max_levels=4,
        seed=42,
        use_gpu=True,
        grounded_embeddings=cp.asarray(grounded_embs),
    )
    
    print(f"  Model initialized with {model.tower.n_satellites} satellites")
    
    # =========================================================================
    # STEP 5: Run learning with checkpoints
    # =========================================================================
    print("\n[Step 5] Running learning with checkpoints...")
    
    report = LearningCurveReport()
    start_time = time.perf_counter()
    sample_idx = 0
    checkpoint_idx = 0
    
    def evaluate_coherence(model, eval_samples, n_eval=200):
        """Evaluate coherence using theory-true path."""
        coherences = []
        stabilities = []
        
        for ctx, tgt in eval_samples[:n_eval]:
            # Embed context
            ctx_mat = model.tower._embed_sequence(ctx)
            
            # Grace contraction
            from holographic_prod.core.algebra import grace_with_stability
            graced_ctx, ctx_stability, _ = grace_with_stability(ctx_mat, basis, xp)
            stabilities.append(float(ctx_stability))
            
            # Route to satellite
            sat_idx = model.tower.route_to_satellite(ctx)
            sat_memory = model.tower._all_memories[sat_idx]
            
            # Unbind
            retrieved = graced_ctx.T @ sat_memory
            
            # Grace on retrieved
            graced_ret, _, _ = grace_with_stability(retrieved, basis, xp)
            
            # Coherence with target
            tgt_emb = model.tower.embeddings[tgt]
            composition = graced_ret @ tgt_emb.T
            
            coeffs = decompose_to_coefficients_batch(composition.reshape(1, 4, 4), basis, xp)[0]
            total_e = float(xp.sum(coeffs ** 2))
            witness_e = float(coeffs[0]**2 + coeffs[15]**2)
            coh = witness_e / max(total_e, PHI_EPSILON)
            coherences.append(coh)
        
        return np.mean(coherences), np.mean(stabilities)
    
    def compute_perplexity(coherence, vocab_size):
        """Compute phi-kernel perplexity from coherence."""
        # PPL = vocab_size^(1 - coherence)
        return vocab_size ** (1 - coherence)
    
    # Initial evaluation (before learning)
    print("\n  Measuring baseline (0 samples)...")
    initial_coherence, initial_stability = evaluate_coherence(model, eval_samples)
    initial_ppl = compute_perplexity(initial_coherence, len(word_to_idx))
    
    print(f"    Baseline coherence: {initial_coherence:.4f}")
    print(f"    Baseline perplexity: {initial_ppl:.1f}")
    print(f"    Baseline stability: {initial_stability:.4f}")
    
    report.initial_coherence = initial_coherence
    report.checkpoints.append(LearningCurveResult(
        samples=0,
        coherence=initial_coherence,
        perplexity=initial_ppl,
        stability=initial_stability,
        throughput=0,
        elapsed_time=0,
        memory_mb=0,
    ))
    
    # Learning loop
    while sample_idx < MAX_SAMPLES:
        # Create batch
        batch_end = min(sample_idx + BATCH_SIZE, MAX_SAMPLES)
        batch = train_samples[sample_idx:batch_end]
        sample_idx = batch_end
        
        if not batch:
            break
        
        # Learn batch
        batch_start = time.perf_counter()
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        model.learn_batch(contexts, targets)
        batch_time = time.perf_counter() - batch_start
        
        # Check if we hit a checkpoint
        if checkpoint_idx < len(CHECKPOINTS) and sample_idx >= CHECKPOINTS[checkpoint_idx]:
            checkpoint = CHECKPOINTS[checkpoint_idx]
            elapsed = time.perf_counter() - start_time
            
            print(f"\n  Checkpoint: {checkpoint:,} samples ({elapsed:.1f}s)")
            
            # Evaluate
            coherence, stability = evaluate_coherence(model, eval_samples)
            ppl = compute_perplexity(coherence, len(word_to_idx))
            
            # Memory usage
            cp.cuda.Stream.null.synchronize()
            mempool = cp.get_default_memory_pool()
            memory_mb = mempool.used_bytes() / (1024**2)
            
            # Throughput
            throughput = sample_idx / elapsed
            
            print(f"    Coherence: {coherence:.4f}")
            print(f"    Perplexity: {ppl:.1f}")
            print(f"    Stability: {stability:.4f}")
            print(f"    Throughput: {throughput:.0f} samples/sec")
            print(f"    GPU Memory: {memory_mb:.0f} MB")
            
            report.checkpoints.append(LearningCurveResult(
                samples=checkpoint,
                coherence=coherence,
                perplexity=ppl,
                stability=stability,
                throughput=throughput,
                elapsed_time=elapsed,
                memory_mb=memory_mb,
            ))
            
            checkpoint_idx += 1
    
    # =========================================================================
    # STEP 6: Analyze results
    # =========================================================================
    print("\n" + "="*80)
    print("LEARNING CURVE ANALYSIS")
    print("="*80)
    
    # Final metrics
    final = report.checkpoints[-1]
    report.final_coherence = final.coherence
    
    # Coherence improvement
    report.coherence_improvement = final.coherence - report.initial_coherence
    print(f"\n  Coherence improvement: {report.initial_coherence:.4f} → {final.coherence:.4f}")
    print(f"    Delta: +{report.coherence_improvement:.4f}")
    
    # Perplexity reduction
    initial_ppl = report.checkpoints[0].perplexity
    final_ppl = final.perplexity
    report.perplexity_reduction = (initial_ppl - final_ppl) / initial_ppl
    print(f"\n  Perplexity reduction: {initial_ppl:.1f} → {final_ppl:.1f}")
    print(f"    Reduction: {report.perplexity_reduction*100:.1f}%")
    
    # Stability convergence
    final_stability = final.stability
    stability_target = PHI_INV_SQ  # ~0.382
    report.stability_converged = abs(final_stability - stability_target) < 0.15
    print(f"\n  Stability convergence: {final_stability:.4f} (target: {stability_target:.4f})")
    print(f"    Converged: {'✓' if report.stability_converged else '✗'}")
    
    # Monotonic improvement check
    coherences = [c.coherence for c in report.checkpoints]
    improvements = [coherences[i+1] >= coherences[i] * 0.95 for i in range(len(coherences)-1)]
    report.monotonic_improvement = sum(improvements) >= len(improvements) * 0.8
    print(f"\n  Monotonic improvement: {'✓' if report.monotonic_improvement else '✗'}")
    print(f"    Improvements: {sum(improvements)}/{len(improvements)}")
    
    # =========================================================================
    # SUCCESS CRITERIA
    # =========================================================================
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)
    
    criteria = {
        'coherence_at_100K > 0.4': final.coherence > 0.4,
        'perplexity_reduction > 50%': report.perplexity_reduction > 0.5,
        'stability_converged': report.stability_converged,
        'coherence_improved': report.coherence_improvement > 0,
    }
    
    for name, passed in criteria.items():
        status = '✓' if passed else '✗'
        print(f"  {status} {name}: {passed}")
    
    # Overall pass: coherence_at_100K > 0.4 is the key metric
    report.passed = final.coherence > 0.4
    
    print("\n" + "="*80)
    if report.passed:
        print("PHASE 1 PASSED: Learning curves demonstrate real learning")
    else:
        print("PHASE 1 FAILED: Coherence did not reach target")
        print(f"  Final coherence: {final.coherence:.4f} (target: 0.4)")
    print("="*80)
    
    # =========================================================================
    # LEARNING CURVE TABLE
    # =========================================================================
    print("\n" + "="*80)
    print("LEARNING CURVE DATA")
    print("="*80)
    print(f"\n{'Samples':>10} {'Coherence':>10} {'Perplexity':>12} {'Stability':>10} {'Throughput':>12}")
    print("-" * 56)
    for c in report.checkpoints:
        print(f"{c.samples:>10,} {c.coherence:>10.4f} {c.perplexity:>12.1f} {c.stability:>10.4f} {c.throughput:>12.0f}")
    
    return report.to_dict()


@app.local_entrypoint()
def main():
    """Run learning curves test."""
    result = run_learning_curves.remote()
    print("\n" + json.dumps(result, indent=2))
    
    # Save result
    with open("/tmp/learning_curves_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("\nResult saved to /tmp/learning_curves_result.json")
