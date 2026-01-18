"""
Speed Benchmarking for Holographic Memory
==========================================

Measures throughput, latency, and memory efficiency.

Key Metrics:
    1. Throughput - Tokens processed per second
    2. Latency - Time per operation (learn, retrieve)
    3. Memory Usage - VRAM/RAM utilization
    4. Scaling - How metrics change with batch size, vocab size, levels

Theory Predictions:
    - Learn: O(1) per token (hash + SO(4) multiply)
    - Retrieve: O(1) episodic, O(1) holographic
    - Memory: O(vocab_size) for embeddings, O(16^levels) for satellites
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ThroughputResult:
    """Result of throughput measurement."""
    operation: str  # "learn" or "retrieve"
    tokens_per_second: float
    batches_per_second: float
    total_tokens: int
    total_time_seconds: float
    batch_size: int


@dataclass
class LatencyResult:
    """Result of latency measurement."""
    operation: str
    mean_latency_us: float  # microseconds
    median_latency_us: float
    p50_latency_us: float
    p90_latency_us: float
    p99_latency_us: float
    min_latency_us: float
    max_latency_us: float
    num_operations: int


def measure_throughput(
    memory: Any,
    operation: str = "learn",
    num_tokens: int = 100000,
    batch_size: int = 1024,
    context_length: int = 8,
    warmup_batches: int = 10,
    timeout_seconds: float = 60.0,
) -> ThroughputResult:
    """
    Measure throughput for learn or retrieve operations.
    
    Args:
        memory: HolographicMemory instance
        operation: "learn" or "retrieve"
        num_tokens: Total tokens to process
        batch_size: Batch size for operations
        context_length: Context window size
        warmup_batches: Number of warmup batches before timing
        timeout_seconds: Maximum measurement time
    
    Returns:
        ThroughputResult with metrics
    """
    # Generate test data
    vocab_size = memory.vocab_size
    num_sequences = num_tokens // (context_length + 1)
    
    sequences = []
    for _ in range(num_sequences):
        seq = list(np.random.randint(0, vocab_size, size=context_length + 1))
        sequences.append(seq)
    
    # Warmup
    for i in range(min(warmup_batches, len(sequences))):
        seq = sequences[i]
        context = seq[:-1]
        target = seq[-1]
        if operation == "learn":
            memory.learn(context, target)
        else:
            # THEORY-TRUE (v5.15.0): Use retrieve_theory_true for benchmarks
            memory.retrieve_theory_true(context)
    
    # Timed run
    start_time = time.perf_counter()
    tokens_processed = 0
    batches_processed = 0
    
    for seq in sequences:
        elapsed = time.perf_counter() - start_time
        if elapsed > timeout_seconds:
            break
        
        context = seq[:-1]
        target = seq[-1]
        
        if operation == "learn":
            memory.learn(context, target)
        else:
            # THEORY-TRUE: retrieve_theory_true for consistent benchmarks
            memory.retrieve_theory_true(context)
        
        tokens_processed += len(seq)
        batches_processed += 1
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    return ThroughputResult(
        operation=operation,
        tokens_per_second=tokens_processed / total_time if total_time > 0 else 0,
        batches_per_second=batches_processed / total_time if total_time > 0 else 0,
        total_tokens=tokens_processed,
        total_time_seconds=total_time,
        batch_size=batch_size,
    )


def measure_latency(
    memory: Any,
    operation: str = "retrieve",
    num_operations: int = 10000,
    context_length: int = 8,
    warmup_operations: int = 100,
) -> LatencyResult:
    """
    Measure latency for individual operations.
    
    Args:
        memory: HolographicMemory instance
        operation: "learn" or "retrieve"
        num_operations: Number of operations to time
        context_length: Context window size
        warmup_operations: Number of warmup operations
    
    Returns:
        LatencyResult with percentile metrics
    """
    vocab_size = memory.vocab_size
    
    # Generate test data
    contexts = []
    targets = []
    for _ in range(num_operations + warmup_operations):
        context = list(np.random.randint(0, vocab_size, size=context_length))
        target = np.random.randint(0, vocab_size)
        contexts.append(context)
        targets.append(target)
    
    # Warmup
    for i in range(warmup_operations):
        if operation == "learn":
            memory.learn(contexts[i], targets[i])
        else:
            # THEORY-TRUE (v5.15.0): Use retrieve_theory_true
            memory.retrieve_theory_true(contexts[i])
    
    # Timed operations
    latencies = []
    
    for i in range(warmup_operations, warmup_operations + num_operations):
        start = time.perf_counter()
        
        if operation == "learn":
            memory.learn(contexts[i], targets[i])
        else:
            # THEORY-TRUE: retrieve_theory_true for latency benchmark
            memory.retrieve_theory_true(contexts[i])
        
        end = time.perf_counter()
        latency_us = (end - start) * 1e6  # microseconds
        latencies.append(latency_us)
    
    latencies = np.array(latencies)
    
    return LatencyResult(
        operation=operation,
        mean_latency_us=float(np.mean(latencies)),
        median_latency_us=float(np.median(latencies)),
        p50_latency_us=float(np.percentile(latencies, 50)),
        p90_latency_us=float(np.percentile(latencies, 90)),
        p99_latency_us=float(np.percentile(latencies, 99)),
        min_latency_us=float(np.min(latencies)),
        max_latency_us=float(np.max(latencies)),
        num_operations=num_operations,
    )


class SpeedBenchmark:
    """
    Comprehensive speed benchmarking for holographic memory.
    
    Usage:
        benchmark = SpeedBenchmark(memory)
        results = benchmark.run()
    """
    
    def __init__(
        self,
        memory: Any,
        verbose: bool = True,
    ):
        """
        Initialize speed benchmark.
        
        Args:
            memory: HolographicMemory instance
            verbose: Print progress
        """
        self.memory = memory
        self.verbose = verbose
    
    def run(
        self,
        num_tokens: int = 100000,
        num_latency_ops: int = 10000,
        timeout_seconds: float = 120.0,
    ) -> Dict[str, Any]:
        """
        Run comprehensive speed benchmark.
        
        Args:
            num_tokens: Tokens for throughput test
            num_latency_ops: Operations for latency test
            timeout_seconds: Maximum benchmark time
        
        Returns:
            Dictionary with all speed metrics
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("  SPEED BENCHMARK")
            print("=" * 60)
        
        results = {}
        
        # Throughput - Learn
        if self.verbose:
            print("\n  Measuring LEARN throughput...")
        results["learn_throughput"] = measure_throughput(
            self.memory,
            operation="learn",
            num_tokens=num_tokens,
            timeout_seconds=timeout_seconds / 4,
        )
        if self.verbose:
            print(f"    {results['learn_throughput'].tokens_per_second:,.0f} tokens/sec")
        
        # Throughput - Retrieve
        if self.verbose:
            print("\n  Measuring RETRIEVE throughput...")
        results["retrieve_throughput"] = measure_throughput(
            self.memory,
            operation="retrieve",
            num_tokens=num_tokens,
            timeout_seconds=timeout_seconds / 4,
        )
        if self.verbose:
            print(f"    {results['retrieve_throughput'].tokens_per_second:,.0f} tokens/sec")
        
        # Latency - Learn
        if self.verbose:
            print("\n  Measuring LEARN latency...")
        results["learn_latency"] = measure_latency(
            self.memory,
            operation="learn",
            num_operations=num_latency_ops,
        )
        if self.verbose:
            print(f"    p50: {results['learn_latency'].p50_latency_us:.1f} μs")
            print(f"    p99: {results['learn_latency'].p99_latency_us:.1f} μs")
        
        # Latency - Retrieve
        if self.verbose:
            print("\n  Measuring RETRIEVE latency...")
        results["retrieve_latency"] = measure_latency(
            self.memory,
            operation="retrieve",
            num_operations=num_latency_ops,
        )
        if self.verbose:
            print(f"    p50: {results['retrieve_latency'].p50_latency_us:.1f} μs")
            print(f"    p99: {results['retrieve_latency'].p99_latency_us:.1f} μs")
        
        # Memory usage (if available)
        try:
            import cupy as cp
            mem_info = cp.cuda.runtime.memGetInfo()
            results["gpu_memory"] = {
                "free_mb": mem_info[0] / 1024 / 1024,
                "total_mb": mem_info[1] / 1024 / 1024,
                "used_mb": (mem_info[1] - mem_info[0]) / 1024 / 1024,
            }
            if self.verbose:
                print(f"\n  GPU Memory: {results['gpu_memory']['used_mb']:.0f} MB / {results['gpu_memory']['total_mb']:.0f} MB")
        except ImportError:
            # CuPy not available - running on CPU
            results["gpu_memory"] = None
        
        if self.verbose:
            self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("  SPEED BENCHMARK SUMMARY")
        print("=" * 60)
        
        print("\n  THROUGHPUT:")
        print(f"    Learn:    {results['learn_throughput'].tokens_per_second:>12,.0f} tok/s")
        print(f"    Retrieve: {results['retrieve_throughput'].tokens_per_second:>12,.0f} tok/s")
        
        print("\n  LATENCY (microseconds):")
        print(f"              {'p50':>10} {'p90':>10} {'p99':>10}")
        print(f"    Learn:    {results['learn_latency'].p50_latency_us:>10.1f} "
              f"{results['learn_latency'].p90_latency_us:>10.1f} "
              f"{results['learn_latency'].p99_latency_us:>10.1f}")
        print(f"    Retrieve: {results['retrieve_latency'].p50_latency_us:>10.1f} "
              f"{results['retrieve_latency'].p90_latency_us:>10.1f} "
              f"{results['retrieve_latency'].p99_latency_us:>10.1f}")
        
        if results.get("gpu_memory"):
            print(f"\n  GPU MEMORY:")
            print(f"    Used: {results['gpu_memory']['used_mb']:.0f} MB")
            print(f"    Free: {results['gpu_memory']['free_mb']:.0f} MB")
        
        print("=" * 60)


def benchmark_scaling(
    memory_factory: callable,
    vocab_sizes: List[int] = [1000, 10000, 50000, 100000],
    level_counts: List[int] = [4, 5, 6],
    num_tokens: int = 50000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark how performance scales with configuration.
    
    Args:
        memory_factory: Callable(vocab_size, levels) -> HolographicMemory
        vocab_sizes: Vocabulary sizes to test
        level_counts: Level counts to test
        num_tokens: Tokens per throughput test
        verbose: Print progress
    
    Returns:
        Dictionary with scaling analysis
    """
    results = {}
    
    for vocab_size in vocab_sizes:
        for levels in level_counts:
            if verbose:
                print(f"\n  Testing vocab={vocab_size:,}, levels={levels}")
            
            try:
                memory = memory_factory(vocab_size, levels)
                
                # Quick throughput test
                throughput = measure_throughput(
                    memory,
                    operation="learn",
                    num_tokens=num_tokens,
                    timeout_seconds=30.0,
                )
                
                key = f"vocab{vocab_size}_level{levels}"
                results[key] = {
                    "vocab_size": vocab_size,
                    "levels": levels,
                    "satellites": 16 ** levels,
                    "throughput_tps": throughput.tokens_per_second,
                }
                
                if verbose:
                    print(f"    {throughput.tokens_per_second:,.0f} tok/s, "
                          f"{16**levels:,} satellites")
                
                # Clean up
                del memory
                
            except Exception as e:
                if verbose:
                    print(f"    ❌ Error: {e}")
                results[f"vocab{vocab_size}_level{levels}"] = {"error": str(e)}
    
    return results
