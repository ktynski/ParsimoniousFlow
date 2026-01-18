"""
Perplexity Benchmarking for Holographic Memory
===============================================

Theory-true perplexity computation for holographic memory systems.

Perplexity measures how well the model predicts a held-out test set:
    PPL = exp(H) where H = -1/N * Σ log P(token_i | context)

For holographic memory:
    - P(token | context) comes from retrieval confidence
    - Episodic retrieval: confidence = 1.0 if exact match, 0.0 otherwise
    - Holographic retrieval: confidence = frobenius_cosine(retrieved, target)
    - Combined: weighted average based on retrieval source
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


# Import constants from core
try:
    from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ
except ImportError:
    PHI_INV = 0.6180339887498949
    PHI_INV_SQ = 0.3819660112501051


@dataclass
class PerplexityResult:
    """Result of perplexity computation."""
    perplexity: float
    cross_entropy: float
    num_tokens: int
    num_correct: int
    accuracy: float
    episodic_hits: int
    holographic_hits: int
    misses: int
    avg_confidence: float
    computation_time_ms: float
    tokens_per_second: float


def compute_cross_entropy(
    log_probs: List[float],
    epsilon: float = 1e-15,
) -> float:
    """
    Compute cross-entropy from log probabilities.
    
    Args:
        log_probs: List of log probabilities (natural log)
        epsilon: Minimum probability to avoid log(0)
    
    Returns:
        Cross-entropy in nats
    """
    if not log_probs:
        return float('inf')
    
    # Clip to avoid numerical issues
    clipped = [max(lp, np.log(epsilon)) for lp in log_probs]
    return -np.mean(clipped)


def compute_perplexity(
    log_probs: List[float],
    epsilon: float = 1e-15,
) -> float:
    """
    Compute perplexity from log probabilities.
    
    PPL = exp(H) where H is cross-entropy.
    
    Args:
        log_probs: List of log probabilities (natural log)
        epsilon: Minimum probability to avoid log(0)
    
    Returns:
        Perplexity (lower is better, minimum is 1.0)
    """
    ce = compute_cross_entropy(log_probs, epsilon)
    return np.exp(ce)


class PerplexityBenchmark:
    """
    Comprehensive perplexity benchmarking for holographic memory.
    
    Usage:
        benchmark = PerplexityBenchmark(memory)
        result = benchmark.evaluate(test_sequences)
    """
    
    def __init__(
        self,
        memory: Any,
        context_length: int = 8,
        verbose: bool = True,
    ):
        """
        Initialize perplexity benchmark.
        
        Args:
            memory: HolographicMemory instance
            context_length: Number of preceding tokens for context
            verbose: Print progress
        """
        self.memory = memory
        self.context_length = context_length
        self.verbose = verbose
    
    def evaluate(
        self,
        test_sequences: List[List[int]],
        timeout_seconds: float = 300.0,
    ) -> PerplexityResult:
        """
        Evaluate perplexity on test sequences.
        
        Args:
            test_sequences: List of token sequences to evaluate
            timeout_seconds: Maximum evaluation time
        
        Returns:
            PerplexityResult with all metrics
        """
        start_time = time.perf_counter()
        
        log_probs = []
        num_correct = 0
        episodic_hits = 0
        holographic_hits = 0
        misses = 0
        confidences = []
        
        total_tokens = sum(len(seq) - self.context_length for seq in test_sequences if len(seq) > self.context_length)
        processed = 0
        
        for seq in test_sequences:
            if len(seq) <= self.context_length:
                continue
            
            for i in range(self.context_length, len(seq)):
                # Check timeout
                elapsed = time.perf_counter() - start_time
                if elapsed > timeout_seconds:
                    if self.verbose:
                        print(f"⚠️ Timeout after {elapsed:.1f}s, {processed}/{total_tokens} tokens")
                    break
                
                context = seq[i - self.context_length:i]
                target = seq[i]
                
                # THEORY-TRUE (v5.15.0): Use retrieve_theory_true_with_info
                # This NEVER returns None - Grace guarantees convergence
                predicted, confidence, info = self.memory.retrieve_theory_true_with_info(context)
                
                # Track retrieval source
                source = info.get('source', 'theory_true')
                if info.get('used_schema_fallback', False):
                    misses += 1  # Satellite empty, used context structure
                elif info.get('satellite_norm', 1.0) > 1e-6:
                    holographic_hits += 1  # Direct holographic retrieval
                else:
                    episodic_hits += 1  # Cached episodic (checked in retrieve_theory_true internally)
                
                # Compute probability
                if predicted == target:
                    num_correct += 1
                    prob = max(confidence, 0.01)  # At least 1% for correct predictions
                else:
                    # Incorrect prediction - use inverse vocabulary probability
                    vocab_size = self.memory.vocab_size
                    prob = 1.0 / vocab_size  # Random baseline
                
                log_probs.append(np.log(prob))
                confidences.append(confidence)
                processed += 1
                
                # Progress reporting
                if self.verbose and processed % 10000 == 0:
                    ppl_so_far = compute_perplexity(log_probs)
                    acc_so_far = num_correct / processed * 100
                    elapsed = time.perf_counter() - start_time
                    tps = processed / elapsed
                    print(f"  [{processed:,}/{total_tokens:,}] PPL={ppl_so_far:.2f}, Acc={acc_so_far:.1f}%, {tps:.0f} tok/s")
            
            # Check timeout at sequence level too
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                break
        
        # Compute final metrics
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        
        if not log_probs:
            return PerplexityResult(
                perplexity=float('inf'),
                cross_entropy=float('inf'),
                num_tokens=0,
                num_correct=0,
                accuracy=0.0,
                episodic_hits=0,
                holographic_hits=0,
                misses=0,
                avg_confidence=0.0,
                computation_time_ms=elapsed_ms,
                tokens_per_second=0.0,
            )
        
        perplexity = compute_perplexity(log_probs)
        cross_entropy = compute_cross_entropy(log_probs)
        accuracy = num_correct / len(log_probs) * 100
        avg_confidence = np.mean(confidences) if confidences else 0.0
        tokens_per_second = len(log_probs) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
        
        result = PerplexityResult(
            perplexity=perplexity,
            cross_entropy=cross_entropy,
            num_tokens=len(log_probs),
            num_correct=num_correct,
            accuracy=accuracy,
            episodic_hits=episodic_hits,
            holographic_hits=holographic_hits,
            misses=misses,
            avg_confidence=avg_confidence,
            computation_time_ms=elapsed_ms,
            tokens_per_second=tokens_per_second,
        )
        
        if self.verbose:
            self._print_result(result)
        
        return result
    
    def _print_result(self, result: PerplexityResult) -> None:
        """Print formatted result."""
        print("\n" + "=" * 60)
        print("  PERPLEXITY BENCHMARK RESULTS")
        print("=" * 60)
        print(f"  Perplexity:        {result.perplexity:.4f}")
        print(f"  Cross-Entropy:     {result.cross_entropy:.4f} nats")
        print(f"  Accuracy:          {result.accuracy:.2f}%")
        print(f"  Tokens Evaluated:  {result.num_tokens:,}")
        print(f"  Correct:           {result.num_correct:,}")
        print("-" * 60)
        print(f"  Episodic Hits:     {result.episodic_hits:,} ({result.episodic_hits/result.num_tokens*100:.1f}%)")
        print(f"  Holographic Hits:  {result.holographic_hits:,} ({result.holographic_hits/result.num_tokens*100:.1f}%)")
        print(f"  Misses:            {result.misses:,} ({result.misses/result.num_tokens*100:.1f}%)")
        print(f"  Avg Confidence:    {result.avg_confidence:.4f}")
        print("-" * 60)
        print(f"  Time:              {result.computation_time_ms:.1f} ms")
        print(f"  Throughput:        {result.tokens_per_second:.0f} tokens/sec")
        print("=" * 60)


def benchmark_perplexity_vs_transformer(
    memory: Any,
    test_sequences: List[List[int]],
    transformer_ppl: float,
    context_length: int = 8,
) -> Dict[str, Any]:
    """
    Compare holographic memory perplexity to a transformer baseline.
    
    Args:
        memory: HolographicMemory instance
        test_sequences: Test sequences
        transformer_ppl: Baseline transformer perplexity
        context_length: Context length for evaluation
    
    Returns:
        Comparison dictionary with metrics and analysis
    """
    benchmark = PerplexityBenchmark(memory, context_length=context_length, verbose=False)
    result = benchmark.evaluate(test_sequences)
    
    ppl_ratio = result.perplexity / transformer_ppl if transformer_ppl > 0 else float('inf')
    
    return {
        "holographic_ppl": result.perplexity,
        "transformer_ppl": transformer_ppl,
        "ratio": ppl_ratio,
        "holographic_better": result.perplexity < transformer_ppl,
        "improvement_percent": (1 - ppl_ratio) * 100 if ppl_ratio < 1 else -(ppl_ratio - 1) * 100,
        "holographic_accuracy": result.accuracy,
        "details": result,
    }
