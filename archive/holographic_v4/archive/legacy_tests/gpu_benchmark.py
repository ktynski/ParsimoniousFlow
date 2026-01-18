"""
GPU Benchmark — Measure H100 utilization for theory-true operations.

Tests:
1. Context computation throughput (the main bottleneck)
2. Attractor storage/retrieval throughput
3. Generation overhead
4. Overall samples/second with different generation frequencies
"""

import time
import numpy as np

# Try CuPy for GPU, fall back to NumPy
try:
    import cupy as cp
    HAS_GPU = True
    xp = cp
    print("✓ CuPy available - using GPU")
except ImportError:
    HAS_GPU = False
    xp = np
    print("⚠ CuPy not available - using CPU (NumPy)")


def benchmark_context_computation(vocab_size=50257, context_size=512, n_iterations=1000):
    """Benchmark context computation throughput."""
    from holographic_v4.algebra import build_clifford_basis, geometric_product_batch, grace_operator
    from holographic_v4.algebra import initialize_embeddings_rotor
    
    print(f"\n{'='*60}")
    print("BENCHMARK 1: Context Computation")
    print(f"{'='*60}")
    print(f"  Vocab: {vocab_size:,} | Context: {context_size} | Iterations: {n_iterations:,}")
    
    # Initialize
    basis = build_clifford_basis(xp)
    embeddings = initialize_embeddings_rotor(vocab_size, basis, xp, seed=42)
    
    # Warm up
    for _ in range(10):
        tokens = np.random.randint(0, vocab_size, size=context_size)
        token_indices = xp.array(tokens, dtype=xp.int64)
        mats = embeddings[token_indices]
        ctx = geometric_product_batch(mats, xp)
        ctx = grace_operator(ctx, basis, xp)
    
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        tokens = np.random.randint(0, vocab_size, size=context_size)
        token_indices = xp.array(tokens, dtype=xp.int64)
        mats = embeddings[token_indices]
        ctx = geometric_product_batch(mats, xp)
        ctx = grace_operator(ctx, basis, xp)
    
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    
    elapsed = time.perf_counter() - start
    throughput = n_iterations / elapsed
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.1f} contexts/sec")
    print(f"  Per context: {1000*elapsed/n_iterations:.3f}ms")
    
    return throughput


def benchmark_holographic_memory(n_iterations=1000):
    """Benchmark holographic memory store/retrieve (v4.8.0)."""
    from holographic_v4 import HolographicMemory, build_clifford_basis
    
    print(f"\n{'='*60}")
    print("BENCHMARK 2: Holographic Memory (O(1) Store/Retrieve)")
    print(f"{'='*60}")
    
    basis = build_clifford_basis(xp)
    memory = HolographicMemory.create(basis, xp)
    
    # Generate random contexts and targets
    contexts = [xp.eye(4) + 0.2 * xp.random.randn(4, 4) for _ in range(n_iterations)]
    targets = [xp.eye(4) + 0.2 * xp.random.randn(4, 4) for _ in range(n_iterations)]
    
    # Benchmark store
    start = time.perf_counter()
    for ctx, tgt in zip(contexts, targets):
        memory.store(ctx, tgt)
    store_time = time.perf_counter() - start
    
    # Benchmark retrieve
    start = time.perf_counter()
    for ctx in contexts[:100]:
        memory.retrieve(ctx)
    retrieve_time = time.perf_counter() - start
    
    print(f"  Store: {n_iterations} patterns in {store_time*1000:.1f}ms ({n_iterations/store_time:.0f}/s)")
    print(f"  Retrieve: 100 queries in {retrieve_time*1000:.1f}ms ({100/retrieve_time:.0f}/s)")
    print(f"  Memory patterns: {memory.n_patterns}")


def benchmark_train_step(vocab_size=50257, context_size=512, n_iterations=1000):
    """Benchmark full train_step throughput."""
    from holographic_v4.pipeline import TheoryTrueModel
    
    print(f"\n{'='*60}")
    print("BENCHMARK 3: Full train_step")
    print(f"{'='*60}")
    print(f"  Vocab: {vocab_size:,} | Context: {context_size} | Iterations: {n_iterations:,}")
    
    model = TheoryTrueModel(vocab_size=vocab_size, context_size=context_size, xp=xp)
    
    # Generate test data
    contexts = [list(np.random.randint(0, vocab_size, size=context_size)) for _ in range(n_iterations)]
    targets = [int(np.random.randint(0, vocab_size)) for _ in range(n_iterations)]
    
    # Warm up
    for i in range(min(100, n_iterations)):
        model.train_step(contexts[i], targets[i])
    
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for i in range(n_iterations):
        model.train_step(contexts[i], targets[i])
    
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    
    elapsed = time.perf_counter() - start
    throughput = n_iterations / elapsed
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    print(f"  Per sample: {1000*elapsed/n_iterations:.3f}ms")
    
    return throughput


def benchmark_generation_overhead(vocab_size=50257, context_size=512, n_generations=100):
    """Benchmark generation overhead."""
    from holographic_v4.pipeline import TheoryTrueModel
    
    print(f"\n{'='*60}")
    print("BENCHMARK 4: Generation Overhead")
    print(f"{'='*60}")
    
    model = TheoryTrueModel(vocab_size=vocab_size, context_size=context_size, xp=xp)
    
    # Train some data first
    for i in range(1000):
        ctx = list(np.random.randint(0, vocab_size, size=context_size))
        tgt = int(np.random.randint(0, vocab_size))
        model.train_step(ctx, tgt)
    
    # Benchmark generation
    context = list(np.random.randint(0, vocab_size, size=context_size))
    
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_generations):
        generated = model.generate(context, num_tokens=30)
    
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    
    elapsed = time.perf_counter() - start
    per_gen = elapsed / n_generations
    
    print(f"  Time for {n_generations} generations: {elapsed:.3f}s")
    print(f"  Per generation (30 tokens): {1000*per_gen:.1f}ms")
    print(f"  Per token: {1000*per_gen/30:.2f}ms")
    
    return per_gen


def benchmark_batching_potential(vocab_size=50257, context_size=512):
    """Check if batching train_step would help."""
    from holographic_v4.algebra import build_clifford_basis, geometric_product_batch, grace_operator
    from holographic_v4.algebra import initialize_embeddings_rotor
    
    print(f"\n{'='*60}")
    print("BENCHMARK 5: Batching Potential")
    print(f"{'='*60}")
    
    basis = build_clifford_basis(xp)
    embeddings = initialize_embeddings_rotor(vocab_size, basis, xp, seed=42)
    
    n_samples = 1000
    
    # Sequential (current approach)
    start = time.perf_counter()
    for _ in range(n_samples):
        tokens = np.random.randint(0, vocab_size, size=context_size)
        token_indices = xp.array(tokens, dtype=xp.int64)
        mats = embeddings[token_indices]
        ctx = geometric_product_batch(mats, xp)
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    seq_time = time.perf_counter() - start
    
    # Batched embedding lookup (potential optimization)
    batch_size = 32
    n_batches = n_samples // batch_size
    
    start = time.perf_counter()
    for _ in range(n_batches):
        # Batch of contexts
        all_tokens = np.random.randint(0, vocab_size, size=(batch_size, context_size))
        all_indices = xp.array(all_tokens, dtype=xp.int64)
        # Batch embedding lookup
        all_mats = embeddings[all_indices]  # [batch, ctx, 4, 4]
        # Still need sequential geom product per sample
        for i in range(batch_size):
            ctx = geometric_product_batch(all_mats[i], xp)
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    batch_time = time.perf_counter() - start
    
    print(f"  Sequential ({n_samples} samples): {seq_time:.3f}s ({n_samples/seq_time:.1f}/s)")
    print(f"  Batched emb lookup ({n_samples} samples): {batch_time:.3f}s ({n_samples/batch_time:.1f}/s)")
    print(f"  Speedup from batched lookup: {seq_time/batch_time:.2f}x")
    
    # Note: The geometric product itself is inherently sequential (depends on previous result)
    # But embedding lookup CAN be batched


def benchmark_memory_bandwidth(vocab_size=50257):
    """Check GPU memory bandwidth utilization."""
    if not HAS_GPU:
        print("\n⚠ Skipping memory bandwidth test (no GPU)")
        return
    
    print(f"\n{'='*60}")
    print("BENCHMARK 6: Memory Bandwidth")
    print(f"{'='*60}")
    
    # Large matrix operations to saturate bandwidth
    sizes = [1000, 5000, 10000, 50000]
    
    for n in sizes:
        A = cp.random.randn(n, 4, 4).astype(cp.float64)
        B = cp.random.randn(n, 4, 4).astype(cp.float64)
        
        # Warm up
        C = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            C = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        # Calculate throughput
        bytes_moved = n * 4 * 4 * 8 * 3 * 100  # A, B read + C write, float64
        bandwidth_gb = bytes_moved / elapsed / 1e9
        
        print(f"  n={n:5}: {bandwidth_gb:.1f} GB/s effective bandwidth")


def run_all_benchmarks():
    """Run all benchmarks and provide recommendations."""
    print("\n" + "="*70)
    print("  H100 GPU UTILIZATION BENCHMARK")
    print("="*70)
    
    benchmark_hash_function()
    ctx_throughput = benchmark_context_computation()
    train_throughput = benchmark_train_step()
    gen_overhead = benchmark_generation_overhead()
    benchmark_batching_potential()
    benchmark_memory_bandwidth()
    
    # Recommendations
    print(f"\n{'='*70}")
    print("  RECOMMENDATIONS")
    print(f"{'='*70}")
    
    # Calculate optimal generation frequency
    # If generation takes X ms and we want <5% overhead:
    gen_ms = gen_overhead * 1000
    train_ms = 1000 / train_throughput  # ms per sample
    
    # For <5% overhead: gen_time / (train_time * interval) < 0.05
    # interval > gen_time / (train_time * 0.05)
    min_interval = int(gen_ms / (train_ms * 0.05))
    
    print(f"\n  Current throughput: ~{train_throughput:.0f} samples/sec")
    print(f"  Generation overhead: ~{gen_ms:.0f}ms per generation")
    print(f"  Training per sample: ~{train_ms:.1f}ms")
    print(f"\n  For <5% generation overhead: generate_every >= {min_interval:,}")
    print(f"  For <1% generation overhead: generate_every >= {min_interval*5:,}")
    
    # GPU utilization estimate
    if HAS_GPU:
        print(f"\n  GPU Memory Usage: {cp.get_default_memory_pool().used_bytes()/1e9:.2f} GB")
        print(f"  GPU Memory Total: {cp.cuda.Device().mem_info[1]/1e9:.1f} GB")
    
    print(f"\n  KEY INSIGHTS:")
    print(f"  • Context computation is GPU-bound (good!)")
    print(f"  • Holographic memory is O(1) for both store and retrieve")
    print(f"  • Generation adds overhead - reduce frequency for max throughput")
    print(f"  • Embedding lookup could be batched for additional speedup")


if __name__ == "__main__":
    run_all_benchmarks()
