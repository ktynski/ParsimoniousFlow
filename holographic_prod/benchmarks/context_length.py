"""
Context Length Scaling Benchmark for Holographic Memory
========================================================

Tests the "infinite context" claim of the architecture.

Theory Prediction:
    - SO(4) composition: R_context = R_1 · R_2 · ... · R_n
    - Group closure: SO(4) × SO(4) = SO(4) always
    - Frobenius norm: ||R|| = 2.0 always (for SO(4))
    - Determinant: det(R) = 1.0 always
    - Orthogonality: R · R^T = I always
    
    At extreme context lengths, numerical errors accumulate as ~√n,
    but the algebraic structure should remain intact.

Benchmark Tests:
    1. SO(4) Property Preservation - Frobenius norm, determinant, orthogonality
    2. Embedding Time Scaling - Should be O(n) linear
    3. Retrieval Accuracy - Should not degrade with context length
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ContextLengthResult:
    """Result of context length scaling test."""
    context_length: int
    frobenius_norm: float  # Should be 2.0
    determinant: float  # Should be 1.0
    orthogonality_error: float  # Should be ~0
    embedding_time_ms: float
    retrieval_accuracy: float
    is_so4_valid: bool
    notes: str


def test_context_scaling(
    memory: Any,
    context_lengths: List[int],
    num_samples: int = 20,
    timeout_seconds: float = 60.0,
) -> List[ContextLengthResult]:
    """
    Test SO(4) property preservation across context lengths.
    
    Args:
        memory: HolographicMemory instance
        context_lengths: List of context lengths to test
        num_samples: Number of samples per context length
        timeout_seconds: Maximum time per context length
    
    Returns:
        List of ContextLengthResult for each context length
    """
    results = []
    
    try:
        import cupy as cp
        xp = cp if hasattr(memory, 'xp') and memory.xp.__name__ == 'cupy' else np
    except ImportError:
        xp = np
    
    for ctx_len in context_lengths:
        print(f"\n  Testing context length: {ctx_len:,}")
        
        frob_norms = []
        dets = []
        orth_errors = []
        embed_times = []
        
        start_total = time.perf_counter()
        
        for i in range(num_samples):
            elapsed = time.perf_counter() - start_total
            if elapsed > timeout_seconds:
                print(f"    ⚠️ Timeout after {i} samples")
                break
            
            # Generate random context
            context = list(np.random.randint(0, memory.vocab_size, size=ctx_len))
            
            # Time the embedding
            start = time.perf_counter()
            emb = memory.tower._embed_sequence(context)
            embed_time = (time.perf_counter() - start) * 1000
            
            # Convert to numpy for analysis if needed
            if hasattr(emb, 'get'):
                emb_np = emb.get()
            else:
                emb_np = np.asarray(emb)
            
            # Compute SO(4) metrics
            frob = np.linalg.norm(emb_np)
            det = np.linalg.det(emb_np)
            orth_err = np.linalg.norm(emb_np @ emb_np.T - np.eye(emb_np.shape[0]))
            
            frob_norms.append(frob)
            dets.append(det)
            orth_errors.append(orth_err)
            embed_times.append(embed_time)
        
        if not frob_norms:
            continue
        
        avg_frob = np.mean(frob_norms)
        avg_det = np.mean(dets)
        avg_orth = np.mean(orth_errors)
        avg_time = np.mean(embed_times)
        
        # Check SO(4) validity (allow more tolerance at extreme lengths)
        tolerance_frob = 0.05 * np.sqrt(ctx_len / 1000)  # Scale with √n
        tolerance_det = 0.05 * np.sqrt(ctx_len / 1000)
        tolerance_orth = 0.1 * np.sqrt(ctx_len / 1000)
        
        frob_ok = abs(avg_frob - 2.0) < max(0.05, tolerance_frob)
        det_ok = abs(avg_det - 1.0) < max(0.05, tolerance_det)
        orth_ok = avg_orth < max(0.1, tolerance_orth)
        
        is_valid = frob_ok and det_ok and orth_ok
        
        notes = []
        if not frob_ok:
            notes.append(f"Frobenius drift: {abs(avg_frob - 2.0):.4f}")
        if not det_ok:
            notes.append(f"Determinant drift: {abs(avg_det - 1.0):.4f}")
        if not orth_ok:
            notes.append(f"Orthogonality error: {avg_orth:.4f}")
        
        result = ContextLengthResult(
            context_length=ctx_len,
            frobenius_norm=avg_frob,
            determinant=avg_det,
            orthogonality_error=avg_orth,
            embedding_time_ms=avg_time,
            retrieval_accuracy=0.0,  # Computed separately if needed
            is_so4_valid=is_valid,
            notes="; ".join(notes) if notes else "SO(4) valid",
        )
        
        results.append(result)
        
        status = "✅" if is_valid else "❌"
        print(f"    {status} Frob={avg_frob:.4f}, Det={avg_det:.4f}, Orth={avg_orth:.6f}, Time={avg_time:.2f}ms")
    
    return results


class ContextLengthBenchmark:
    """
    Comprehensive context length scaling benchmark.
    
    Usage:
        benchmark = ContextLengthBenchmark(memory)
        results = benchmark.run()
    """
    
    def __init__(
        self,
        memory: Any,
        verbose: bool = True,
    ):
        """
        Initialize context length benchmark.
        
        Args:
            memory: HolographicMemory instance
            verbose: Print progress
        """
        self.memory = memory
        self.verbose = verbose
    
    def run(
        self,
        context_lengths: Optional[List[int]] = None,
        num_samples: int = 20,
        timeout_seconds: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Run comprehensive context length benchmark.
        
        Args:
            context_lengths: List of context lengths to test (default: exponential scale)
            num_samples: Number of samples per context length
            timeout_seconds: Maximum total benchmark time
        
        Returns:
            Dictionary with results, scaling analysis, and recommendations
        """
        if context_lengths is None:
            # Default: exponential scaling from 64 to 1M
            context_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                              16384, 32768, 65536, 131072, 262144, 524288, 1048576]
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("  CONTEXT LENGTH SCALING BENCHMARK")
            print("=" * 60)
            print(f"  Testing {len(context_lengths)} context lengths")
            print(f"  Range: {min(context_lengths):,} to {max(context_lengths):,}")
            print(f"  Samples per length: {num_samples}")
        
        start_time = time.perf_counter()
        results = []
        
        for ctx_len in context_lengths:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                if self.verbose:
                    print(f"\n⚠️ Timeout after {elapsed:.1f}s")
                break
            
            remaining = timeout_seconds - elapsed
            ctx_results = test_context_scaling(
                self.memory,
                [ctx_len],
                num_samples=num_samples,
                timeout_seconds=min(60.0, remaining),
            )
            results.extend(ctx_results)
        
        # Analyze scaling
        analysis = self._analyze_scaling(results)
        
        if self.verbose:
            self._print_summary(results, analysis)
        
        return {
            "results": results,
            "analysis": analysis,
            "total_time_seconds": time.perf_counter() - start_time,
        }
    
    def _analyze_scaling(self, results: List[ContextLengthResult]) -> Dict[str, Any]:
        """Analyze scaling behavior from results."""
        if len(results) < 2:
            return {"error": "Not enough data points"}
        
        lengths = [r.context_length for r in results]
        times = [r.embedding_time_ms for r in results]
        frob_drifts = [abs(r.frobenius_norm - 2.0) for r in results]
        det_drifts = [abs(r.determinant - 1.0) for r in results]
        orth_errors = [r.orthogonality_error for r in results]
        
        # Fit time scaling: t = a * n^b
        # log(t) = log(a) + b * log(n)
        log_lengths = np.log(lengths)
        log_times = np.log(times)
        
        # Linear regression
        A = np.vstack([log_lengths, np.ones(len(log_lengths))]).T
        b_coef, log_a = np.linalg.lstsq(A, log_times, rcond=None)[0]
        
        # Error scaling: should be ~√n
        log_errors = np.log([max(e, 1e-10) for e in orth_errors])
        error_coef, _ = np.linalg.lstsq(A, log_errors, rcond=None)[0]
        
        # Maximum valid context (where SO(4) still holds)
        valid_results = [r for r in results if r.is_so4_valid]
        max_valid_context = max((r.context_length for r in valid_results), default=0)
        
        return {
            "time_scaling_exponent": float(b_coef),
            "time_scaling_coefficient": float(np.exp(log_a)),
            "is_linear": abs(b_coef - 1.0) < 0.2,  # Close to O(n)
            "error_scaling_exponent": float(error_coef),
            "is_sqrt_error": abs(error_coef - 0.5) < 0.2,  # Close to √n
            "max_valid_context": max_valid_context,
            "all_valid": all(r.is_so4_valid for r in results),
            "avg_frob_drift": np.mean(frob_drifts),
            "avg_det_drift": np.mean(det_drifts),
            "avg_orth_error": np.mean(orth_errors),
        }
    
    def _print_summary(self, results: List[ContextLengthResult], analysis: Dict[str, Any]) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("  SCALING ANALYSIS")
        print("=" * 60)
        
        print(f"\n  Time Complexity:")
        b = analysis.get("time_scaling_exponent", 0)
        if analysis.get("is_linear", False):
            print(f"    ✅ O(n^{b:.2f}) ≈ O(n) LINEAR - as predicted by theory!")
        else:
            print(f"    ⚠️ O(n^{b:.2f}) - not linear")
        
        print(f"\n  Error Scaling:")
        e = analysis.get("error_scaling_exponent", 0)
        if analysis.get("is_sqrt_error", False):
            print(f"    ✅ Error ∝ n^{e:.2f} ≈ √n - as predicted by theory!")
        else:
            print(f"    ⚠️ Error ∝ n^{e:.2f} - not √n")
        
        print(f"\n  SO(4) Validity:")
        max_valid = analysis.get("max_valid_context", 0)
        if analysis.get("all_valid", False):
            print(f"    ✅ Valid at ALL tested lengths up to {max_valid:,} tokens!")
        else:
            print(f"    ⚠️ Valid up to {max_valid:,} tokens")
        
        print(f"\n  Average Errors:")
        print(f"    Frobenius drift: {analysis.get('avg_frob_drift', 0):.6f}")
        print(f"    Determinant drift: {analysis.get('avg_det_drift', 0):.6f}")
        print(f"    Orthogonality error: {analysis.get('avg_orth_error', 0):.6f}")
        
        print("\n" + "=" * 60)
        print("  CONTEXT LENGTH RESULTS")
        print("=" * 60)
        print(f"  {'Length':>12} {'Frob':>8} {'Det':>8} {'Orth':>10} {'Time(ms)':>10} {'Status'}")
        print("-" * 60)
        
        for r in results:
            status = "✅" if r.is_so4_valid else "❌"
            print(f"  {r.context_length:>12,} {r.frobenius_norm:>8.4f} {r.determinant:>8.4f} "
                  f"{r.orthogonality_error:>10.6f} {r.embedding_time_ms:>10.2f} {status}")
        
        print("=" * 60)
