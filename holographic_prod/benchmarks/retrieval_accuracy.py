"""
Retrieval Accuracy Benchmarking for Holographic Memory
======================================================

Measures accuracy of different retrieval pathways:
    1. Episodic (exact match) - O(1) hash lookup
    2. Holographic (generalization) - O(1) unbinding
    3. Combined (episodic + holographic) - Parallel with conflict detection

Theory Predictions:
    - Episodic: 100% accuracy on trained sequences, 0% on novel
    - Holographic: High accuracy when training approaches memory capacity
    - Combined: Should exceed either alone through synergy
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class RetrievalAccuracyResult:
    """Result of retrieval accuracy evaluation."""
    overall_accuracy: float
    episodic_accuracy: float
    holographic_accuracy: float
    episodic_hit_rate: float  # % of queries answered by episodic
    holographic_hit_rate: float  # % answered by holographic
    miss_rate: float  # % with no retrieval
    conflict_rate: float  # % where episodic != holographic
    avg_confidence: float
    num_queries: int
    evaluation_time_ms: float


def measure_retrieval_accuracy(
    memory: Any,
    test_sequences: List[List[int]],
    context_length: int = 8,
    timeout_seconds: float = 300.0,
    verbose: bool = True,
) -> RetrievalAccuracyResult:
    """
    Measure retrieval accuracy on test sequences.
    
    Args:
        memory: HolographicMemory instance
        test_sequences: List of token sequences for testing
        context_length: Context window size
        timeout_seconds: Maximum evaluation time
        verbose: Print progress
    
    Returns:
        RetrievalAccuracyResult with all metrics
    """
    start_time = time.perf_counter()
    
    correct_overall = 0
    correct_episodic = 0
    correct_holographic = 0
    episodic_hits = 0
    holographic_hits = 0
    misses = 0
    conflicts = 0
    confidences = []
    total_queries = 0
    
    for seq in test_sequences:
        if len(seq) <= context_length:
            continue
        
        for i in range(context_length, len(seq)):
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                if verbose:
                    print(f"⚠️ Timeout after {elapsed:.1f}s")
                break
            
            context = seq[i - context_length:i]
            target = seq[i]
            
            # THEORY-TRUE (v5.15.0): Use retrieve_theory_true_with_info
            # This NEVER returns None - Grace guarantees convergence
            predicted, confidence, info = memory.retrieve_theory_true_with_info(context)
            
            total_queries += 1
            confidences.append(confidence)
            
            # Track source (theory-true always has a source)
            source = info.get('source', 'theory_true')
            if info.get('used_schema_fallback', False):
                misses += 1  # Schema fallback (satellite empty)
            elif info.get('satellite_norm', 1.0) > 1e-6:
                holographic_hits += 1
                if predicted == target:
                    correct_holographic += 1
            else:
                episodic_hits += 1  # Default to episodic tracking
                if predicted == target:
                    correct_episodic += 1
            
            # Track coherence (theory-true metric)
            coherence = info.get('coherence', 0.0)
            # Low coherence = potential conflict
            if coherence < 0.382:  # φ⁻²
                conflicts += 1
            
            # Overall accuracy
            if predicted == target:
                correct_overall += 1
        
        # Check timeout at sequence level
        elapsed = time.perf_counter() - start_time
        if elapsed > timeout_seconds:
            break
        
        if verbose and total_queries > 0 and total_queries % 10000 == 0:
            acc = correct_overall / total_queries * 100
            print(f"  [{total_queries:,}] Accuracy: {acc:.1f}%")
    
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    
    if total_queries == 0:
        return RetrievalAccuracyResult(
            overall_accuracy=0.0,
            episodic_accuracy=0.0,
            holographic_accuracy=0.0,
            episodic_hit_rate=0.0,
            holographic_hit_rate=0.0,
            miss_rate=0.0,
            conflict_rate=0.0,
            avg_confidence=0.0,
            num_queries=0,
            evaluation_time_ms=elapsed_ms,
        )
    
    return RetrievalAccuracyResult(
        overall_accuracy=correct_overall / total_queries * 100,
        episodic_accuracy=correct_episodic / episodic_hits * 100 if episodic_hits > 0 else 0.0,
        holographic_accuracy=correct_holographic / holographic_hits * 100 if holographic_hits > 0 else 0.0,
        episodic_hit_rate=episodic_hits / total_queries * 100,
        holographic_hit_rate=holographic_hits / total_queries * 100,
        miss_rate=misses / total_queries * 100,
        conflict_rate=conflicts / total_queries * 100,
        avg_confidence=np.mean(confidences) if confidences else 0.0,
        num_queries=total_queries,
        evaluation_time_ms=elapsed_ms,
    )


class RetrievalAccuracyBenchmark:
    """
    Comprehensive retrieval accuracy benchmarking.
    
    Usage:
        benchmark = RetrievalAccuracyBenchmark(memory)
        result = benchmark.evaluate(test_sequences)
    """
    
    def __init__(
        self,
        memory: Any,
        context_length: int = 8,
        verbose: bool = True,
    ):
        """
        Initialize retrieval accuracy benchmark.
        
        Args:
            memory: HolographicMemory instance
            context_length: Context window size
            verbose: Print progress
        """
        self.memory = memory
        self.context_length = context_length
        self.verbose = verbose
    
    def evaluate(
        self,
        test_sequences: List[List[int]],
        timeout_seconds: float = 300.0,
    ) -> RetrievalAccuracyResult:
        """
        Evaluate retrieval accuracy on test sequences.
        
        Args:
            test_sequences: List of token sequences
            timeout_seconds: Maximum evaluation time
        
        Returns:
            RetrievalAccuracyResult with all metrics
        """
        result = measure_retrieval_accuracy(
            self.memory,
            test_sequences,
            context_length=self.context_length,
            timeout_seconds=timeout_seconds,
            verbose=self.verbose,
        )
        
        if self.verbose:
            self._print_result(result)
        
        return result
    
    def evaluate_by_pathway(
        self,
        test_sequences: List[List[int]],
        timeout_seconds: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Evaluate accuracy broken down by retrieval pathway.
        
        Returns detailed analysis of:
        - Episodic-only performance
        - Holographic-only performance
        - Combined performance
        - Conflict resolution accuracy
        """
        start_time = time.perf_counter()
        
        episodic_only_correct = 0
        episodic_only_total = 0
        holographic_only_correct = 0
        holographic_only_total = 0
        combined_correct = 0
        combined_total = 0
        conflict_resolved_correct = 0
        conflict_total = 0
        
        for seq in test_sequences:
            if len(seq) <= self.context_length:
                continue
            
            for i in range(self.context_length, len(seq)):
                elapsed = time.perf_counter() - start_time
                if elapsed > timeout_seconds:
                    break
                
                context = seq[i - self.context_length:i]
                target = seq[i]
                
                # THEORY-TRUE (v5.15.0): Use retrieve_theory_true
                predicted = self.memory.retrieve_theory_true(context)
                
                combined_total += 1
                if predicted == target:
                    combined_correct += 1
                
                # Track episodic-only
                episodic_target = info.get('episodic_target')
                if episodic_target is not None:
                    episodic_only_total += 1
                    if episodic_target == target:
                        episodic_only_correct += 1
                
                # Track holographic-only
                holographic_target = info.get('holographic_target')
                if holographic_target is not None:
                    holographic_only_total += 1
                    if holographic_target == target:
                        holographic_only_correct += 1
                
                # Track conflict resolution
                if info.get('conflict', 0) > 0:
                    conflict_total += 1
                    if predicted == target:
                        conflict_resolved_correct += 1
            
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                break
        
        return {
            "episodic_only": {
                "accuracy": episodic_only_correct / episodic_only_total * 100 if episodic_only_total > 0 else 0,
                "total": episodic_only_total,
                "correct": episodic_only_correct,
            },
            "holographic_only": {
                "accuracy": holographic_only_correct / holographic_only_total * 100 if holographic_only_total > 0 else 0,
                "total": holographic_only_total,
                "correct": holographic_only_correct,
            },
            "combined": {
                "accuracy": combined_correct / combined_total * 100 if combined_total > 0 else 0,
                "total": combined_total,
                "correct": combined_correct,
            },
            "conflict_resolution": {
                "accuracy": conflict_resolved_correct / conflict_total * 100 if conflict_total > 0 else 0,
                "total": conflict_total,
                "correct": conflict_resolved_correct,
            },
            "synergy": self._compute_synergy(
                episodic_only_correct / episodic_only_total if episodic_only_total > 0 else 0,
                holographic_only_correct / holographic_only_total if holographic_only_total > 0 else 0,
                combined_correct / combined_total if combined_total > 0 else 0,
            ),
        }
    
    def _compute_synergy(
        self,
        episodic_acc: float,
        holographic_acc: float,
        combined_acc: float,
    ) -> Dict[str, float]:
        """Compute synergy metrics between pathways."""
        max_individual = max(episodic_acc, holographic_acc)
        
        if max_individual > 0:
            synergy_ratio = combined_acc / max_individual
        else:
            synergy_ratio = 0
        
        return {
            "combined_vs_best_individual": synergy_ratio,
            "combined_exceeds_both": combined_acc > episodic_acc and combined_acc > holographic_acc,
            "improvement_over_episodic": (combined_acc - episodic_acc) * 100,
            "improvement_over_holographic": (combined_acc - holographic_acc) * 100,
        }
    
    def _print_result(self, result: RetrievalAccuracyResult) -> None:
        """Print formatted result."""
        print("\n" + "=" * 60)
        print("  RETRIEVAL ACCURACY BENCHMARK RESULTS")
        print("=" * 60)
        print(f"  Overall Accuracy:      {result.overall_accuracy:.2f}%")
        print("-" * 60)
        print(f"  Episodic Accuracy:     {result.episodic_accuracy:.2f}%")
        print(f"  Holographic Accuracy:  {result.holographic_accuracy:.2f}%")
        print("-" * 60)
        print(f"  Episodic Hit Rate:     {result.episodic_hit_rate:.1f}%")
        print(f"  Holographic Hit Rate:  {result.holographic_hit_rate:.1f}%")
        print(f"  Miss Rate:             {result.miss_rate:.1f}%")
        print(f"  Conflict Rate:         {result.conflict_rate:.1f}%")
        print("-" * 60)
        print(f"  Avg Confidence:        {result.avg_confidence:.4f}")
        print(f"  Total Queries:         {result.num_queries:,}")
        print(f"  Time:                  {result.evaluation_time_ms:.1f} ms")
        print("=" * 60)
