"""
Run Few-Shot NLP Benchmarks on Modal
====================================

Tests general language learning via few-shot prompting on Modal GPU.

This is the TRUE test of whether the system is a general language learner:
- Model sees a few examples of a task
- Must generalize to new instances
- No task-specific training (just like GPT evaluation)

Usage:
    modal run holographic_prod/benchmarks/run_modal_benchmarks.py --benchmark all
    modal run holographic_prod/benchmarks/run_modal_benchmarks.py --benchmark glue --k-shot 5
"""

import modal
import time

# Modal app setup
app = modal.App("holographic-few-shot-benchmarks")

# Persistent volume for checkpoints (shared with training!)
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)

# GPU image with CUDA (same as training)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "numpy>=1.24.0",
        "cupy-cuda12x>=12.0.0",
        "scipy>=1.10.0",
        "datasets>=2.14.0",
        "tiktoken>=0.5.0",
        "transformers>=4.30.0",
        "huggingface_hub>=0.17.0",
    )
)


def get_tokenizer(vocab_size: int):
    """Get tiktoken tokenizer wrapper."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        
        class TiktokenWrapper:
            def __init__(self, enc, vocab_size):
                self.enc = enc
                self.vocab_size = vocab_size
            
            def encode(self, text: str):
                tokens = self.enc.encode(text)
                return [t % self.vocab_size for t in tokens]
            
            def decode(self, tokens):
                # Map back (approximate)
                return self.enc.decode([t for t in tokens])
        
        return TiktokenWrapper(enc, vocab_size)
    except ImportError:
        return None


def pretrain_on_data(memory, num_samples: int = 500000, verbose: bool = True):
    """
    Pre-train memory on real language data.
    
    Uses TinyStories for coherent, diverse text.
    """
    try:
        from datasets import load_dataset
        
        if verbose:
            print(f"  Loading TinyStories for pre-training...")
        
        # Load TinyStories (same as training run)
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split="train",
            trust_remote_code=True,
        )
        
        tokenizer = get_tokenizer(memory.vocab_size)
        if tokenizer is None:
            if verbose:
                print("  ⚠️ tiktoken not available, using simple tokenization")
            def tokenize(text):
                return [ord(c) % memory.vocab_size for c in text[:512]]
        else:
            def tokenize(text):
                return tokenizer.encode(text)[:512]
        
        samples_trained = 0
        context_len = 8
        
        for item in dataset:
            text = item.get("text", "")
            if len(text) < 50:
                continue
            
            tokens = tokenize(text)
            
            # Train on context-target pairs
            for i in range(context_len, len(tokens)):
                context = tokens[i-context_len:i]
                target = tokens[i]
                memory.learn(context, target)
                samples_trained += 1
                
                if samples_trained >= num_samples:
                    break
            
            if samples_trained >= num_samples:
                break
            
            if verbose and samples_trained % 100000 == 0:
                print(f"    Pre-trained: {samples_trained:,}/{num_samples:,}")
        
        if verbose:
            print(f"  ✓ Pre-trained on {samples_trained:,} samples")
        
        return samples_trained
        
    except Exception as e:
        if verbose:
            print(f"  ❌ Pre-training error: {e}")
        return 0


def load_checkpoint_into_memory(memory, checkpoint_path: str = "/checkpoints/holographic_checkpoint.npz"):
    """Load checkpoint from the shared volume into memory."""
    import numpy as np
    import os
    
    if not os.path.exists(checkpoint_path):
        print(f"  ❌ No checkpoint found at {checkpoint_path}")
        return False, 0
    
    try:
        checkpoint = np.load(checkpoint_path)
        
        # Restore satellite memories (sparse)
        active_indices = checkpoint['active_indices']
        active_memories = checkpoint['active_memories']
        active_bindings = checkpoint['active_bindings']
        samples = int(checkpoint['samples'])
        
        # Restore to memory
        xp = memory.xp
        for i, idx in enumerate(active_indices):
            if idx < memory.tower.n_satellites:
                memory.tower._all_memories[idx] = xp.asarray(active_memories[i])
                memory.tower._satellite_n_bindings[idx] = active_bindings[i]
        
        print(f"  ✓ Loaded checkpoint: {samples:,} samples")
        print(f"    Active satellites: {len(active_indices):,}")
        return True, samples
        
    except Exception as e:
        print(f"  ❌ Failed to load checkpoint: {e}")
        return False, 0


@app.function(
    image=image.add_local_dir(
        "/Users/fractlphoneroom1/Desktop/ParsimoniousFlow/holographic_prod",
        remote_path="/root/holographic_prod",
    ),
    gpu="A100",
    timeout=7200,  # 2 hours
    volumes={"/checkpoints": checkpoint_volume},  # Mount the shared checkpoint volume!
)
def run_few_shot_benchmarks(
    k_shot: int = 5,
    pretrain_samples: int = 1000000,
    max_benchmark_samples: int = 100,
    use_checkpoint: bool = True,  # Load from training run checkpoint!
):
    """
    Run few-shot benchmarks on pre-trained holographic memory.
    
    This tests GENERAL LANGUAGE LEARNING:
    - Load checkpoint from training run (if available)
    - OR pre-train on TinyStories
    - Test with few-shot prompting (like GPT)
    - No task-specific training
    """
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    
    print("\n" + "=" * 70)
    print(f"  FEW-SHOT NLP BENCHMARKS ({k_shot}-shot) on A100 GPU")
    print("  Testing GENERAL LANGUAGE LEARNING")
    print("=" * 70)
    
    # Initialize memory
    from holographic_prod import HolographicMemory
    from holographic_prod.benchmarks import (
        FewShotBenchmarkRunner,
    )
    
    print("\n  Initializing HolographicMemory (GPU)...")
    memory = HolographicMemory(
        vocab_size=50000,
        max_levels=5,
        use_gpu=True,
    )
    
    # Try to load checkpoint from training run
    checkpoint_loaded = False
    if use_checkpoint:
        print("\n  Looking for checkpoint from training run...")
        checkpoint_loaded, samples = load_checkpoint_into_memory(memory)
        if checkpoint_loaded:
            print(f"  🚀 Using {samples:,} sample checkpoint from training run!")
            pretrain_samples = 0  # Skip pre-training
    
    # Pre-train on diverse text (skip if checkpoint loaded)
    pretrain_time = 0
    if pretrain_samples > 0:
        print(f"\n  Pre-training on {pretrain_samples:,} samples...")
        start_pretrain = time.perf_counter()
        pretrain_on_data(memory, num_samples=pretrain_samples, verbose=True)
        pretrain_time = time.perf_counter() - start_pretrain
        print(f"  Pre-training time: {pretrain_time:.1f}s")
    else:
        print("\n  ✓ Skipping pre-training (using checkpoint)")
    
    # Get tokenizer for benchmarks
    tokenizer = get_tokenizer(memory.vocab_size)
    
    # Run few-shot benchmarks
    print(f"\n  Running {k_shot}-shot benchmarks...")
    runner = FewShotBenchmarkRunner(
        memory,
        tokenizer=tokenizer,
        k_shot=k_shot,
        context_length=512,
        verbose=True,
    )
    
    results = runner.run_all(
        benchmarks=["glue", "mmlu"],
        max_samples_per_benchmark=max_benchmark_samples,
        timeout_per_benchmark=1800,
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    
    summary = {
        "pretrain_samples": pretrain_samples,
        "pretrain_time_seconds": pretrain_time,
        "k_shot": k_shot,
        "results": {},
    }
    
    for name, result in results.items():
        summary["results"][name] = {
            "accuracy": result.accuracy,
            "correct": result.num_correct,
            "total": result.num_total,
            "per_task": result.per_task_scores,
        }
        print(f"  {name}: {result.accuracy:.1f}% ({result.num_correct}/{result.num_total})")
    
    print("\n  GPT-3.5 (5-shot) reference:")
    print("    GLUE (SST-2): ~95%")
    print("    MMLU: ~70%")
    print("=" * 70)
    
    return summary


@app.function(
    image=image.add_local_dir(
        "/Users/fractlphoneroom1/Desktop/ParsimoniousFlow/holographic_prod",
        remote_path="/root/holographic_prod",
    ),
    gpu="A100",
    timeout=3600,
    volumes={"/checkpoints": checkpoint_volume},
)
def run_few_shot_glue(
    k_shot: int = 5,
    pretrain_samples: int = 500000,
    max_samples: int = 200,
    use_checkpoint: bool = True,
):
    """Run few-shot GLUE benchmark only."""
    import sys
    sys.path.insert(0, "/root")
    
    print("\n" + "=" * 70)
    print(f"  FEW-SHOT GLUE ({k_shot}-shot) on A100 GPU")
    print("=" * 70)
    
    from holographic_prod import HolographicMemory
    from holographic_prod.benchmarks import FewShotGLUE, FewShotEvaluator
    
    memory = HolographicMemory(vocab_size=50000, max_levels=5, use_gpu=True)
    
    # Try to load checkpoint from training run
    if use_checkpoint:
        print("\n  Looking for checkpoint from training run...")
        loaded, samples = load_checkpoint_into_memory(memory)
        if loaded:
            print(f"  🚀 Using {samples:,} sample checkpoint!")
            pretrain_samples = 0
    
    if pretrain_samples > 0:
        print(f"\n  Pre-training on {pretrain_samples:,} samples...")
        pretrain_on_data(memory, num_samples=pretrain_samples, verbose=True)
    else:
        print("\n  ✓ Skipping pre-training (using checkpoint)")
    
    tokenizer = get_tokenizer(memory.vocab_size)
    evaluator = FewShotEvaluator(memory, tokenizer, context_length=512, verbose=True)
    
    glue = FewShotGLUE(evaluator, k_shot=k_shot, verbose=True)
    result = glue.run(
        tasks=["sst2"],
        max_samples_per_task=max_samples,
        timeout_seconds=1800,
    )
    
    print(f"\n  GLUE Result: {result.accuracy:.1f}%")
    print(f"  Correct: {result.num_correct}/{result.num_total}")
    
    return {
        "benchmark": "GLUE",
        "k_shot": k_shot,
        "accuracy": result.accuracy,
        "correct": result.num_correct,
        "total": result.num_total,
        "per_task": result.per_task_scores,
    }


@app.function(
    image=image.add_local_dir(
        "/Users/fractlphoneroom1/Desktop/ParsimoniousFlow/holographic_prod",
        remote_path="/root/holographic_prod",
    ),
    gpu="A100",
    timeout=3600,
    volumes={"/checkpoints": checkpoint_volume},
)
def run_few_shot_mmlu(
    k_shot: int = 5,
    pretrain_samples: int = 500000,
    max_samples: int = 50,
    use_checkpoint: bool = True,
):
    """Run few-shot MMLU benchmark only."""
    import sys
    sys.path.insert(0, "/root")
    
    print("\n" + "=" * 70)
    print(f"  FEW-SHOT MMLU ({k_shot}-shot) on A100 GPU")
    print("=" * 70)
    
    from holographic_prod import HolographicMemory
    from holographic_prod.benchmarks import FewShotMMLU, FewShotEvaluator
    
    memory = HolographicMemory(vocab_size=50000, max_levels=5, use_gpu=True)
    
    # Try to load checkpoint from training run
    if use_checkpoint:
        print("\n  Looking for checkpoint from training run...")
        loaded, samples = load_checkpoint_into_memory(memory)
        if loaded:
            print(f"  🚀 Using {samples:,} sample checkpoint!")
            pretrain_samples = 0
    
    if pretrain_samples > 0:
        print(f"\n  Pre-training on {pretrain_samples:,} samples...")
        pretrain_on_data(memory, num_samples=pretrain_samples, verbose=True)
    else:
        print("\n  ✓ Skipping pre-training (using checkpoint)")
    
    tokenizer = get_tokenizer(memory.vocab_size)
    evaluator = FewShotEvaluator(memory, tokenizer, context_length=512, verbose=True)
    
    mmlu = FewShotMMLU(evaluator, k_shot=k_shot, verbose=True)
    result = mmlu.run(
        subjects=["abstract_algebra", "anatomy", "high_school_mathematics"],
        max_samples_per_subject=max_samples,
        timeout_seconds=1800,
    )
    
    print(f"\n  MMLU Result: {result.accuracy:.1f}%")
    print(f"  Correct: {result.num_correct}/{result.num_total}")
    
    return {
        "benchmark": "MMLU",
        "k_shot": k_shot,
        "accuracy": result.accuracy,
        "correct": result.num_correct,
        "total": result.num_total,
        "per_subject": result.per_task_scores,
    }


@app.local_entrypoint()
def main(
    benchmark: str = "all",
    k_shot: int = 5,
    pretrain_samples: int = 500000,
    max_samples: int = 100,
):
    """
    Run few-shot NLP benchmarks on Modal.
    
    Args:
        benchmark: Which benchmark (all, glue, mmlu)
        k_shot: Number of few-shot examples
        pretrain_samples: Pre-training samples
        max_samples: Max samples per benchmark
    """
    print(f"\n🚀 Running {benchmark.upper()} ({k_shot}-shot) on Modal...")
    print(f"   Pre-training samples: {pretrain_samples:,}")
    print(f"   Benchmark samples: {max_samples}")
    
    benchmark = benchmark.lower()
    
    if benchmark == "all":
        result = run_few_shot_benchmarks.remote(
            k_shot=k_shot,
            pretrain_samples=pretrain_samples,
            max_benchmark_samples=max_samples,
        )
    elif benchmark == "glue":
        result = run_few_shot_glue.remote(
            k_shot=k_shot,
            pretrain_samples=pretrain_samples,
            max_samples=max_samples,
        )
    elif benchmark == "mmlu":
        result = run_few_shot_mmlu.remote(
            k_shot=k_shot,
            pretrain_samples=pretrain_samples,
            max_samples=max_samples,
        )
    else:
        print(f"❌ Unknown benchmark: {benchmark}")
        print("   Choose from: all, glue, mmlu")
        return
    
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    
    import json
    print(json.dumps(result, indent=2))
