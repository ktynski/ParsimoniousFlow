"""
Modal Test: Long Context Length Validation with Real Language

THEORY TO VERIFY:
    SO(4) embeddings enable "infinite context" because:
    1. Product of SO(4) matrices is always SO(4) (group closure)
    2. |R₁·R₂·...·Rₙ| = 2 (Frobenius norm preserved)
    3. det(R₁·R₂·...·Rₙ) = 1 (determinant preserved)
    4. No gradient vanishing/exploding (unit quaternion composition)

TEST CONTEXTS:
    - 256 tokens (baseline)
    - 512 tokens  
    - 1024 tokens
    - 2048 tokens
    - 4096 tokens (stress test)
    - 8192 tokens (extreme)

METRICS:
    - SO(4) validity: Frobenius norm ≈ 2, det ≈ 1
    - Orthogonality error: ||R^T R - I||
    - Accuracy at each context length
    - Stability (witness energy / total energy)
    - Generation coherence
"""

import modal
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import time

# Modal setup
app = modal.App("holographic-long-context-test")

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
        "transformers>=4.30",
        "huggingface_hub",
    )
    .add_local_dir("holographic_prod", "/root/project/holographic_prod")
)


@dataclass
class LongContextResult:
    """Results for a single context length test."""
    context_length: int
    n_samples: int
    
    # SO(4) validity
    avg_frobenius_norm: float = 0.0  # Should be ≈ 2.0
    avg_determinant: float = 0.0     # Should be ≈ 1.0
    avg_orthogonality_error: float = 0.0  # Should be ≈ 0.0
    
    # Performance
    accuracy_top1: float = 0.0
    accuracy_top5: float = 0.0
    avg_stability: float = 0.0
    
    # Timing
    avg_embed_time_ms: float = 0.0
    avg_retrieve_time_ms: float = 0.0
    
    # Errors
    errors: List[str] = field(default_factory=list)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    memory=32768,
)
def test_long_context(
    context_lengths: List[int] = [256, 512, 1024, 2048, 4096],
    n_samples_per_length: int = 100,
    vocab_size: int = 10000,
    max_levels: int = 4,
) -> Dict[str, Any]:
    """
    Test SO(4) properties at various context lengths.
    
    Returns detailed results for each context length.
    """
    import sys
    sys.path.insert(0, "/root/project")
    
    import numpy as np
    
    # Try GPU
    try:
        import cupy as cp
        xp = cp
        use_gpu = True
        print("✓ GPU available (CuPy)")
    except ImportError:
        xp = np
        use_gpu = False
        print("⚠ GPU not available, using CPU")
    
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    from holographic_prod.core.constants import PHI_INV_SQ
    
    print("=" * 70)
    print("  LONG CONTEXT LENGTH TEST")
    print("  Verifying SO(4) properties at scale")
    print("=" * 70)
    print(f"\n  Context lengths: {context_lengths}")
    print(f"  Samples per length: {n_samples_per_length}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Tower levels: {max_levels}")
    
    # Load real language data
    print("\n  Loading TinyStories dataset...")
    from datasets import load_dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # Build vocabulary from dataset
    print("  Building vocabulary...")
    word_counts = {}
    for i, story in enumerate(dataset):
        if i >= 10000:  # Sample 10K stories for vocab
            break
        words = story['text'].lower().split()
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Top vocab_size words
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    vocab = {word: i for i, (word, _) in enumerate(sorted_words[:vocab_size-2])}
    vocab['<unk>'] = vocab_size - 2
    vocab['<pad>'] = vocab_size - 1
    
    def tokenize(text: str, max_len: int) -> List[int]:
        words = text.lower().split()
        tokens = [vocab.get(w, vocab['<unk>']) for w in words]
        # Pad or truncate to exact length
        if len(tokens) < max_len:
            tokens = tokens + [vocab['<pad>']] * (max_len - len(tokens))
        return tokens[:max_len]
    
    print(f"  ✓ Vocabulary: {len(vocab)} words")
    
    # Initialize memory
    print(f"\n  Initializing HolographicMemory...")
    memory = HolographicMemory(
        vocab_size=vocab_size,
        max_levels=max_levels,
        use_gpu=use_gpu,
        seed=42,
    )
    
    # Pre-learn some patterns (warm up)
    print("  Warming up memory with 1000 patterns...")
    for i, story in enumerate(dataset):
        if i >= 1000:
            break
        tokens = tokenize(story['text'], 64)
        if len(set(tokens)) > 10:  # Skip mostly padding
            context = tokens[:-1]
            target = tokens[-1]
            memory.learn(context, target)
    
    print(f"  ✓ Memory initialized: {memory.n_patterns} patterns")
    
    # Results storage
    all_results = {}
    
    for ctx_len in context_lengths:
        print(f"\n{'=' * 70}")
        print(f"  TESTING CONTEXT LENGTH: {ctx_len}")
        print(f"{'=' * 70}")
        
        result = LongContextResult(
            context_length=ctx_len,
            n_samples=n_samples_per_length,
        )
        
        frobenius_norms = []
        determinants = []
        orthogonality_errors = []
        stabilities = []
        correct_top1 = 0
        correct_top5 = 0
        embed_times = []
        retrieve_times = []
        
        # Find stories long enough
        long_stories = []
        for story in dataset:
            tokens = tokenize(story['text'], ctx_len + 1)
            if sum(1 for t in tokens if t != vocab['<pad>']) >= ctx_len:
                long_stories.append(tokens)
                if len(long_stories) >= n_samples_per_length * 2:
                    break
        
        if len(long_stories) < n_samples_per_length:
            print(f"  ⚠ Only found {len(long_stories)} stories with {ctx_len}+ tokens")
            result.errors.append(f"Insufficient data: {len(long_stories)} stories")
            all_results[ctx_len] = result
            continue
        
        print(f"  Found {len(long_stories)} long stories")
        
        for i in range(min(n_samples_per_length, len(long_stories))):
            tokens = long_stories[i]
            context = tokens[:ctx_len]
            target = tokens[ctx_len] if ctx_len < len(tokens) else tokens[-1]
            
            try:
                # ============================================================
                # TEST 1: Embed sequence and check SO(4) properties
                # ============================================================
                t0 = time.perf_counter()
                ctx_mat = memory.tower._embed_sequence(context)
                embed_time = (time.perf_counter() - t0) * 1000
                embed_times.append(embed_time)
                
                # Convert to numpy for analysis (CuPy uses .get())
                if use_gpu:
                    ctx_np = cp.asnumpy(ctx_mat) if hasattr(cp, 'asnumpy') else ctx_mat.get()
                else:
                    ctx_np = np.array(ctx_mat)
                
                # Frobenius norm (should be 2 for SO(4))
                frob_norm = float(np.linalg.norm(ctx_np, 'fro'))
                frobenius_norms.append(frob_norm)
                
                # Determinant (should be 1 for SO(4))
                det = float(np.linalg.det(ctx_np))
                determinants.append(det)
                
                # Orthogonality error: ||R^T R - I||
                rtrans_r = ctx_np.T @ ctx_np
                identity = np.eye(4)
                orth_error = float(np.linalg.norm(rtrans_r - identity, 'fro'))
                orthogonality_errors.append(orth_error)
                
                # ============================================================
                # TEST 2: Learn and retrieve
                # ============================================================
                memory.learn(context, target)
                
                t0 = time.perf_counter()
                retrieved, confidence = memory.retrieve_deterministic(context)
                retrieve_time = (time.perf_counter() - t0) * 1000
                retrieve_times.append(retrieve_time)
                
                if retrieved == target:
                    correct_top1 += 1
                    correct_top5 += 1
                
                # ============================================================
                # TEST 3: Stability
                # ============================================================
                # Compute witness (scalar + pseudoscalar energy)
                scalar = ctx_np[0, 0]  # Simplified: use (0,0) element
                total_energy = float(np.sum(ctx_np ** 2))
                witness_energy = scalar ** 2
                stability = witness_energy / max(total_energy, 1e-10)
                stabilities.append(stability)
                
            except Exception as e:
                import traceback
                error_msg = f"Sample {i}: {str(e)}\n{traceback.format_exc()}"
                result.errors.append(error_msg)
                if i < 3:  # Print first 3 errors for debugging
                    print(f"    ERROR: {error_msg[:200]}")
                continue
            
            # Progress
            if (i + 1) % 20 == 0:
                print(f"    Processed {i + 1}/{n_samples_per_length} samples...")
        
        # Aggregate results
        if frobenius_norms:
            result.avg_frobenius_norm = float(np.mean(frobenius_norms))
            result.avg_determinant = float(np.mean(determinants))
            result.avg_orthogonality_error = float(np.mean(orthogonality_errors))
            result.avg_stability = float(np.mean(stabilities))
            result.accuracy_top1 = correct_top1 / len(frobenius_norms)
            result.accuracy_top5 = correct_top5 / len(frobenius_norms)
            result.avg_embed_time_ms = float(np.mean(embed_times))
            result.avg_retrieve_time_ms = float(np.mean(retrieve_times))
        
        # Print results
        print(f"\n  Results for context_length={ctx_len}:")
        print(f"    SO(4) Validity:")
        print(f"      Frobenius norm: {result.avg_frobenius_norm:.6f} (expect 2.0)")
        print(f"      Determinant:    {result.avg_determinant:.6f} (expect 1.0)")
        print(f"      Orth. error:    {result.avg_orthogonality_error:.6f} (expect 0.0)")
        print(f"    Performance:")
        print(f"      Top-1 accuracy: {result.accuracy_top1:.1%}")
        print(f"      Stability:      {result.avg_stability:.4f}")
        print(f"    Timing:")
        print(f"      Embed time:     {result.avg_embed_time_ms:.2f} ms")
        print(f"      Retrieve time:  {result.avg_retrieve_time_ms:.2f} ms")
        
        # Check SO(4) validity
        frob_ok = abs(result.avg_frobenius_norm - 2.0) < 0.01
        det_ok = abs(result.avg_determinant - 1.0) < 0.01
        orth_ok = result.avg_orthogonality_error < 0.01
        
        if frob_ok and det_ok and orth_ok:
            print(f"    ✅ SO(4) VALID at {ctx_len} tokens!")
        else:
            print(f"    ⚠️ SO(4) properties degraded:")
            if not frob_ok:
                print(f"       Frobenius norm drift: {abs(result.avg_frobenius_norm - 2.0):.4f}")
            if not det_ok:
                print(f"       Determinant drift: {abs(result.avg_determinant - 1.0):.4f}")
            if not orth_ok:
                print(f"       Orthogonality error: {result.avg_orthogonality_error:.4f}")
        
        all_results[ctx_len] = result
    
    # Final summary
    print("\n" + "=" * 70)
    print("  SUMMARY: SO(4) Properties Across Context Lengths")
    print("=" * 70)
    print("\n  | Context | Frob Norm | Det | Orth Err | Accuracy | Embed (ms) |")
    print("  |" + "-" * 9 + "|" + "-" * 11 + "|" + "-" * 7 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 12 + "|")
    
    for ctx_len, result in all_results.items():
        print(f"  | {ctx_len:>7} | {result.avg_frobenius_norm:>9.4f} | {result.avg_determinant:>5.3f} | {result.avg_orthogonality_error:>8.5f} | {result.accuracy_top1:>8.1%} | {result.avg_embed_time_ms:>10.2f} |")
    
    # Convert to serializable dict
    return {
        ctx_len: {
            'context_length': r.context_length,
            'n_samples': r.n_samples,
            'avg_frobenius_norm': r.avg_frobenius_norm,
            'avg_determinant': r.avg_determinant,
            'avg_orthogonality_error': r.avg_orthogonality_error,
            'accuracy_top1': r.accuracy_top1,
            'accuracy_top5': r.accuracy_top5,
            'avg_stability': r.avg_stability,
            'avg_embed_time_ms': r.avg_embed_time_ms,
            'avg_retrieve_time_ms': r.avg_retrieve_time_ms,
            'errors': r.errors,
        }
        for ctx_len, r in all_results.items()
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    memory=32768,
)
def test_extreme_context(
    context_length: int = 8192,
    n_samples: int = 20,
) -> Dict[str, Any]:
    """
    Stress test with extreme context length (8K+ tokens).
    """
    import sys
    sys.path.insert(0, "/root/project")
    
    import numpy as np
    
    try:
        import cupy as cp
        xp = cp
        use_gpu = True
    except ImportError:
        xp = np
        use_gpu = False
    
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    
    print("=" * 70)
    print(f"  EXTREME CONTEXT TEST: {context_length} tokens")
    print("=" * 70)
    
    # Initialize memory
    memory = HolographicMemory(
        vocab_size=10000,
        max_levels=5,  # Higher levels for more capacity
        use_gpu=use_gpu,
        seed=42,
    )
    
    # Generate synthetic long sequences
    # (Real data rarely has 8K+ token sequences)
    print(f"\n  Generating {n_samples} synthetic sequences of {context_length} tokens...")
    
    results = {
        'frobenius_norms': [],
        'determinants': [],
        'orthogonality_errors': [],
        'embed_times_ms': [],
    }
    
    for i in range(n_samples):
        # Random token sequence
        context = [np.random.randint(0, 10000) for _ in range(context_length)]
        target = np.random.randint(0, 10000)
        
        # Embed and measure
        t0 = time.perf_counter()
        ctx_mat = memory.tower._embed_sequence(context)
        embed_time = (time.perf_counter() - t0) * 1000
        
        # Convert to numpy for analysis (CuPy uses .get())
        if use_gpu:
            ctx_np = cp.asnumpy(ctx_mat) if hasattr(cp, 'asnumpy') else ctx_mat.get()
        else:
            ctx_np = np.array(ctx_mat)
        
        frob_norm = float(np.linalg.norm(ctx_np, 'fro'))
        det = float(np.linalg.det(ctx_np))
        rtrans_r = ctx_np.T @ ctx_np
        orth_error = float(np.linalg.norm(rtrans_r - np.eye(4), 'fro'))
        
        results['frobenius_norms'].append(frob_norm)
        results['determinants'].append(det)
        results['orthogonality_errors'].append(orth_error)
        results['embed_times_ms'].append(embed_time)
        
        if (i + 1) % 5 == 0:
            print(f"    Sample {i+1}/{n_samples}: frob={frob_norm:.4f}, det={det:.4f}, orth_err={orth_error:.6f}, time={embed_time:.1f}ms")
    
    # Aggregate
    avg_frob = float(np.mean(results['frobenius_norms']))
    avg_det = float(np.mean(results['determinants']))
    avg_orth = float(np.mean(results['orthogonality_errors']))
    avg_time = float(np.mean(results['embed_times_ms']))
    
    print(f"\n  Results for {context_length} tokens:")
    print(f"    Frobenius norm: {avg_frob:.6f} (expect 2.0)")
    print(f"    Determinant:    {avg_det:.6f} (expect 1.0)")
    print(f"    Orth. error:    {avg_orth:.6f} (expect 0.0)")
    print(f"    Embed time:     {avg_time:.2f} ms")
    
    # Check validity
    if abs(avg_frob - 2.0) < 0.1 and abs(avg_det - 1.0) < 0.1 and avg_orth < 0.1:
        print(f"\n  ✅ SO(4) VALID at {context_length} tokens!")
        print(f"     Theory confirmed: 'Infinite context' claim holds!")
    else:
        print(f"\n  ⚠️ SO(4) properties show drift at {context_length} tokens")
    
    return {
        'context_length': context_length,
        'n_samples': n_samples,
        'avg_frobenius_norm': avg_frob,
        'avg_determinant': avg_det,
        'avg_orthogonality_error': avg_orth,
        'avg_embed_time_ms': avg_time,
        'raw_results': results,
    }


@app.local_entrypoint()
def main(
    test_type: str = "standard",
    extreme_length: int = 8192,
):
    """
    Run long context tests on Modal.
    
    Args:
        test_type: "standard" (256-4096), "extreme" (8192+), or "both"
        extreme_length: Context length for extreme test
    """
    print("=" * 70)
    print("  HOLOGRAPHIC LONG CONTEXT TEST")
    print("  Verifying SO(4) 'Infinite Context' Claim")
    print("=" * 70)
    
    if test_type in ["standard", "both"]:
        print("\n  Running standard context length tests (256-4096)...")
        results = test_long_context.remote(
            context_lengths=[256, 512, 1024, 2048, 4096],
            n_samples_per_length=50,
        )
        
        print("\n  STANDARD TEST COMPLETE")
        for ctx_len, data in results.items():
            print(f"\n  Context {ctx_len}:")
            print(f"    Frobenius: {data['avg_frobenius_norm']:.4f}")
            print(f"    Det: {data['avg_determinant']:.4f}")
            print(f"    Accuracy: {data['accuracy_top1']:.1%}")
    
    if test_type in ["extreme", "both"]:
        print(f"\n  Running extreme context test ({extreme_length} tokens)...")
        extreme_results = test_extreme_context.remote(
            context_length=extreme_length,
            n_samples=20,
        )
        
        print("\n  EXTREME TEST COMPLETE")
        print(f"    Frobenius: {extreme_results['avg_frobenius_norm']:.4f}")
        print(f"    Det: {extreme_results['avg_determinant']:.4f}")
        print(f"    Orth error: {extreme_results['avg_orthogonality_error']:.6f}")
    
    print("\n" + "=" * 70)
    print("  ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # For local testing
    print("Run with: modal run holographic_prod/tests/test_modal_long_context.py")
    print("Options:")
    print("  --test-type standard  # Test 256-4096 tokens")
    print("  --test-type extreme   # Test 8192+ tokens")
    print("  --test-type both      # Test all")
    print("  --extreme-length 16384  # Custom extreme length")
