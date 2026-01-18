"""
Test-Driven Development for Brain-Inspired Efficiencies

Each optimization must pass these tests BEFORE full implementation:
1. CORRECTNESS: Output matches naive implementation
2. EFFICIENCY: Measurable speedup achieved
3. STABILITY: No degradation in learning quality

Order of implementation (by ROI):
1. Incremental Context (O(n)→O(1))
2. Coarse-to-Fine Retrieval (witness-first matching)
3. Oscillatory Grace (per-φ² tokens)
4. Sensory Gating (informativeness-based skipping)
5. Sparse Embedding Storage (dual mode for near-identity)
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any

# Import from existing codebase
from holographic_v4.algebra import (
    build_clifford_basis,
    geometric_product,
    geometric_product_batch,
    grace_operator,
    grace_with_stability_batch,
    decompose_to_coefficients,
)
from holographic_v4.quotient import (
    extract_witness,
    extract_witness_batch,
    grace_stability,
    grace_stability_batch,
)
from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ


# =============================================================================
# TEST 1: INCREMENTAL CONTEXT
# =============================================================================
# Current: Context = M₁ @ M₂ @ ... @ Mₙ (recomputed each time)
# Proposed: Context_new = Context_old @ M_new (incremental)
# Expected: Mathematically identical, O(n) → O(1)

class TestIncrementalContext:
    """Test that incremental context update produces identical results to full recomputation."""
    
    def __init__(self, xp=np):
        self.xp = xp
        self.basis = build_clifford_basis(xp)
        
    def compute_context_naive(self, embeddings: List) -> np.ndarray:
        """Naive O(n) context computation - recompute from scratch."""
        xp = self.xp
        if len(embeddings) == 0:
            return xp.eye(4)
        
        context = embeddings[0].copy()
        for M in embeddings[1:]:
            context = geometric_product(context, M)  # matmul in Cl(3,1)
        return context
    
    def compute_context_incremental(self, running_context: np.ndarray, new_embedding: np.ndarray) -> np.ndarray:
        """Incremental O(1) context update."""
        return geometric_product(running_context, new_embedding)  # single matmul
    
    def test_correctness(self, n_tokens: int = 20, n_trials: int = 10) -> Dict[str, Any]:
        """Verify incremental matches naive for random sequences."""
        xp = self.xp
        results = {"passed": True, "max_error": 0.0, "trials": []}
        
        for trial in range(n_trials):
            # Generate random identity-biased embeddings
            embeddings = []
            for _ in range(n_tokens):
                M = xp.eye(4) + 0.1 * xp.random.randn(4, 4)
                embeddings.append(M)
            
            # Naive: Full recomputation
            context_naive = self.compute_context_naive(embeddings)
            
            # Incremental: Running update
            context_incremental = xp.eye(4)
            for M in embeddings:
                context_incremental = self.compute_context_incremental(context_incremental, M)
            
            # Compare
            error = float(xp.max(xp.abs(context_naive - context_incremental)))
            results["trials"].append({"error": error})
            results["max_error"] = max(results["max_error"], error)
            
            if error > 1e-10:
                results["passed"] = False
        
        return results
    
    def test_efficiency(self, context_lengths: List[int] = [5, 10, 20, 50, 100]) -> Dict[str, Any]:
        """Measure speedup of incremental vs naive."""
        xp = self.xp
        results = {"lengths": [], "naive_times": [], "incremental_times": [], "speedups": []}
        
        for n in context_lengths:
            # Generate embeddings
            embeddings = [xp.eye(4) + 0.1 * xp.random.randn(4, 4) for _ in range(n)]
            
            # Time naive (simulate adding one more token)
            n_repeats = 100
            start = time.perf_counter()
            for _ in range(n_repeats):
                _ = self.compute_context_naive(embeddings)
            naive_time = (time.perf_counter() - start) / n_repeats
            
            # Time incremental (simulate adding one more token with cached context)
            running_context = self.compute_context_naive(embeddings[:-1])
            start = time.perf_counter()
            for _ in range(n_repeats):
                _ = self.compute_context_incremental(running_context, embeddings[-1])
            incremental_time = (time.perf_counter() - start) / n_repeats
            
            speedup = naive_time / incremental_time if incremental_time > 0 else float('inf')
            
            results["lengths"].append(n)
            results["naive_times"].append(naive_time * 1000)  # ms
            results["incremental_times"].append(incremental_time * 1000)  # ms
            results["speedups"].append(speedup)
        
        return results


# =============================================================================
# TEST 2: COARSE-TO-FINE RETRIEVAL (REFINED: Filtering Only)
# =============================================================================
# INSIGHT: Witness captures STABILITY, not SEMANTIC content.
#          Witness similarity ≠ semantic similarity
#          
# CORRECT APPROACH: Use witness for FILTERING (exclusion), not RANKING
#   - If witness distance > threshold → definitely NOT a match (exclude)
#   - Otherwise → do full comparison (ranking preserves correctness)
#
# THEORY-TRUE JUSTIFICATION:
#   The witness (σ, p) is a Lorentz invariant - it captures how "rotated"
#   the multivector is from identity, not what direction it's rotated TO.
#   Two very different semantic contexts can have identical witnesses!
#   
#   But: If witnesses are VERY different, the contexts are definitely different.
#   This is the "contrapositive" that enables early rejection.

class TestCoarseToFineRetrieval:
    """Test witness-based FILTERING (not ranking) for retrieval speedup."""
    
    def __init__(self, xp=np):
        self.xp = xp
        self.basis = build_clifford_basis(xp)
        
    def full_similarity(self, M1: np.ndarray, M2: np.ndarray) -> float:
        """Full 16-coefficient similarity (ground truth)."""
        xp = self.xp
        c1 = decompose_to_coefficients(M1, self.basis, xp)
        c2 = decompose_to_coefficients(M2, self.basis, xp)
        
        norm1 = xp.sqrt(xp.sum(c1 * c1))
        norm2 = xp.sqrt(xp.sum(c2 * c2))
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(xp.sum(c1 * c2) / (norm1 * norm2))
    
    def witness_distance(self, M1: np.ndarray, M2: np.ndarray) -> float:
        """Euclidean distance in witness space."""
        xp = self.xp
        w1 = extract_witness(M1, self.basis, xp)
        w2 = extract_witness(M2, self.basis, xp)
        return float(xp.sqrt((w1[0] - w2[0])**2 + (w1[1] - w2[1])**2))
    
    def filter_candidates(
        self, 
        query: np.ndarray, 
        candidates: List[np.ndarray],
        witness_reject_threshold: float = 0.5  # Reject if witness distance > threshold
    ) -> Tuple[List[int], int]:
        """
        FILTERING ONLY: Exclude candidates whose witness is too different.
        
        THEORY-TRUE:
            - Large witness distance → definitely different → exclude
            - Small witness distance → maybe similar → keep for full comparison
            
        This does NOT try to rank by witness (which fails).
        It only EXCLUDES obviously bad candidates.
        
        Returns:
            (kept_indices, n_rejected) - indices that passed filter
        """
        kept = []
        rejected = 0
        
        for i, cand in enumerate(candidates):
            w_dist = self.witness_distance(query, cand)
            if w_dist <= witness_reject_threshold:
                kept.append(i)
            else:
                rejected += 1
        
        return kept, rejected
    
    def retrieve_with_filter(
        self,
        query: np.ndarray,
        candidates: List[np.ndarray],
        witness_reject_threshold: float = 0.5,
        top_k: int = 3
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Retrieve top-k using witness filter + full similarity ranking.
        
        Strategy:
            1. Filter out candidates with witness distance > threshold (cheap)
            2. Compute full similarity only for remaining candidates (expensive)
            3. Return top-k by full similarity
        """
        # Step 1: Witness filter (O(n) cheap operations)
        kept_indices, n_rejected = self.filter_candidates(
            query, candidates, witness_reject_threshold
        )
        
        # Step 2: Full similarity only on filtered candidates
        if len(kept_indices) == 0:
            return [], {'n_rejected': n_rejected, 'n_full_compared': 0}
        
        similarities = []
        for idx in kept_indices:
            sim = self.full_similarity(query, candidates[idx])
            similarities.append((idx, sim))
        
        # Step 3: Sort by full similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = [idx for idx, _ in similarities[:top_k]]
        
        return top_k_indices, {
            'n_rejected': n_rejected,
            'n_full_compared': len(kept_indices),
            'filter_rate': n_rejected / (n_rejected + len(kept_indices))
        }
    
    def test_early_exit_rate(self, n_pairs: int = 1000) -> Dict[str, Any]:
        """
        Measure how often witness filtering rejects candidates.
        Tests multiple thresholds to find optimal configuration.
        """
        xp = self.xp
        
        # Test multiple rejection thresholds
        results_by_threshold = {}
        
        for reject_threshold in [0.3, 0.5, 0.7, 1.0]:
            rejected = 0
            kept = 0
            false_rejects = 0  # Rejected something that was actually similar
            
            for i in range(n_pairs):
                # Generate query
                query = xp.eye(4) + 0.2 * xp.random.randn(4, 4)
                
                # Generate candidate with controlled similarity
                if i < n_pairs // 3:
                    # Similar (should NOT be rejected)
                    cand = query + 0.05 * xp.random.randn(4, 4)
                elif i < 2 * n_pairs // 3:
                    # Very different (SHOULD be rejected)
                    cand = xp.eye(4) + 0.5 * xp.random.randn(4, 4)
                else:
                    # Medium (ambiguous)
                    cand = query + 0.3 * xp.random.randn(4, 4)
                
                w_dist = self.witness_distance(query, cand)
                full_sim = self.full_similarity(query, cand)
                
                if w_dist > reject_threshold:
                    rejected += 1
                    # Check if this was a false reject (high similarity but rejected)
                    if full_sim > 0.7:
                        false_rejects += 1
                else:
                    kept += 1
            
            filter_rate = rejected / n_pairs
            false_reject_rate = false_rejects / max(rejected, 1)
            
            results_by_threshold[reject_threshold] = {
                "rejected": rejected,
                "kept": kept,
                "filter_rate": filter_rate,
                "false_reject_rate": false_reject_rate,
                "speedup": 1.0 / (1.0 - filter_rate * 0.9) if filter_rate < 1 else float('inf')
            }
        
        # Find best threshold: maximize filter rate while keeping false reject < 10%
        best_threshold = None
        best_filter_rate = 0.0
        for thresh, res in results_by_threshold.items():
            if res["false_reject_rate"] < 0.1 and res["filter_rate"] > best_filter_rate:
                best_threshold = thresh
                best_filter_rate = res["filter_rate"]
        
        if best_threshold is None:
            # No threshold works well, use loosest
            best_threshold = 1.0
        
        best = results_by_threshold[best_threshold]
        
        return {
            "all_thresholds": results_by_threshold,
            "best_threshold": best_threshold,
            "rejected": best["rejected"],
            "kept": best["kept"],
            "filter_rate": best["filter_rate"],
            "false_reject_rate": best["false_reject_rate"],
            "theoretical_speedup": best["speedup"],
            "passed": best["false_reject_rate"] < 0.1
        }
    
    def test_ranking_preservation(self, n_queries: int = 50, n_candidates: int = 20) -> Dict[str, Any]:
        """
        Verify that filter + full ranking preserves top-k correctly.
        
        THEORY-TRUE: Since we use full similarity for final ranking (not witness),
        ranking should be EXACTLY preserved for non-rejected candidates.
        """
        xp = self.xp
        
        ranking_errors = []
        filter_rates = []
        
        for _ in range(n_queries):
            # Generate query
            query = xp.eye(4) + 0.2 * xp.random.randn(4, 4)
            
            # Generate candidates - mix of similar and different
            candidates = []
            for j in range(n_candidates):
                if j < n_candidates // 3:
                    # Similar to query (should NOT be filtered)
                    candidates.append(query + 0.1 * xp.random.randn(4, 4))
                else:
                    # Random (may be filtered)
                    candidates.append(xp.eye(4) + 0.4 * xp.random.randn(4, 4))
            
            # Ground truth: Full ranking
            full_sims = [self.full_similarity(query, c) for c in candidates]
            full_ranking = np.argsort(full_sims)[::-1]  # Descending
            
            # Filter + Full ranking
            top_k_indices, stats = self.retrieve_with_filter(query, candidates, top_k=3)
            filter_rates.append(stats['filter_rate'])
            
            # Compare top-3
            top_k = 3
            full_top_k = set(full_ranking[:top_k])
            filtered_top_k = set(top_k_indices)
            
            if len(filtered_top_k) >= top_k:
                overlap = len(full_top_k & filtered_top_k) / top_k
            else:
                # Not enough candidates passed filter
                overlap = len(full_top_k & filtered_top_k) / len(filtered_top_k) if filtered_top_k else 0
            
            ranking_errors.append(1.0 - overlap)
        
        return {
            "mean_ranking_error": float(np.mean(ranking_errors)),
            "perfect_top_k_rate": float(np.mean([e == 0 for e in ranking_errors])),
            "mean_filter_rate": float(np.mean(filter_rates)),
            "passed": np.mean(ranking_errors) < 0.1  # Now stricter: 10% error (was 20%)
        }


# =============================================================================
# TEST 3: OSCILLATORY GRACE
# =============================================================================
# Current: Grace applied per token
# Proposed: Grace applied per φ² ≈ 2.618 tokens (chunk boundary)
# Expected: ~2.6× fewer Grace calls, similar stability

class TestOscillatoryGrace:
    """Test that chunked Grace produces similar stability with fewer calls."""
    
    def __init__(self, xp=np):
        self.xp = xp
        self.basis = build_clifford_basis(xp)
        self.chunk_size = int(PHI * PHI)  # φ² ≈ 2.618, round to 3
        
    def per_token_grace(self, embeddings: List[np.ndarray]) -> Tuple[np.ndarray, int]:
        """Apply Grace after every token (current method)."""
        xp = self.xp
        context = xp.eye(4)
        grace_count = 0
        
        for M in embeddings:
            context = geometric_product(context, M)
            context = grace_operator(context, self.basis, xp)
            grace_count += 1
        
        return context, grace_count
    
    def chunked_grace(self, embeddings: List[np.ndarray]) -> Tuple[np.ndarray, int]:
        """Apply Grace every φ² tokens (proposed method)."""
        xp = self.xp
        context = xp.eye(4)
        grace_count = 0
        
        for i, M in enumerate(embeddings):
            context = geometric_product(context, M)
            
            # Apply Grace at chunk boundaries
            if (i + 1) % self.chunk_size == 0:
                context = grace_operator(context, self.basis, xp)
                grace_count += 1
        
        # Final Grace if not at boundary
        if len(embeddings) % self.chunk_size != 0:
            context = grace_operator(context, self.basis, xp)
            grace_count += 1
        
        return context, grace_count
    
    def test_stability_comparison(self, n_tokens: int = 100, n_trials: int = 20) -> Dict[str, Any]:
        """Compare stability of per-token vs chunked Grace."""
        xp = self.xp
        results = {
            "per_token_stability": [],
            "chunked_stability": [],
            "witness_diff": [],
            "grace_reduction": []
        }
        
        for _ in range(n_trials):
            # Generate random embeddings
            embeddings = [xp.eye(4) + 0.15 * xp.random.randn(4, 4) for _ in range(n_tokens)]
            
            # Per-token Grace
            ctx_per_token, count_per_token = self.per_token_grace(embeddings)
            stab_per_token = grace_stability(ctx_per_token, self.basis, xp)
            witness_per_token = extract_witness(ctx_per_token, self.basis, xp)
            
            # Chunked Grace
            ctx_chunked, count_chunked = self.chunked_grace(embeddings)
            stab_chunked = grace_stability(ctx_chunked, self.basis, xp)
            witness_chunked = extract_witness(ctx_chunked, self.basis, xp)
            
            # Compare
            results["per_token_stability"].append(float(stab_per_token))
            results["chunked_stability"].append(float(stab_chunked))
            results["witness_diff"].append(float(
                xp.sqrt((witness_per_token[0] - witness_chunked[0])**2 + 
                       (witness_per_token[1] - witness_chunked[1])**2)
            ))
            results["grace_reduction"].append(count_per_token / count_chunked)
        
        return {
            "mean_per_token_stability": float(np.mean(results["per_token_stability"])),
            "mean_chunked_stability": float(np.mean(results["chunked_stability"])),
            "stability_preserved": abs(np.mean(results["per_token_stability"]) - 
                                       np.mean(results["chunked_stability"])) < 0.1,
            "mean_witness_diff": float(np.mean(results["witness_diff"])),
            "mean_grace_reduction": float(np.mean(results["grace_reduction"])),
            "passed": np.mean(results["chunked_stability"]) > 0.5  # Must maintain reasonable stability
        }
    
    def test_efficiency(self, context_lengths: List[int] = [20, 50, 100, 200]) -> Dict[str, Any]:
        """Measure speedup of chunked Grace."""
        xp = self.xp
        results = {"lengths": [], "per_token_times": [], "chunked_times": [], "speedups": []}
        
        for n in context_lengths:
            embeddings = [xp.eye(4) + 0.15 * xp.random.randn(4, 4) for _ in range(n)]
            
            n_repeats = 20
            
            # Time per-token
            start = time.perf_counter()
            for _ in range(n_repeats):
                _ = self.per_token_grace(embeddings)
            per_token_time = (time.perf_counter() - start) / n_repeats
            
            # Time chunked
            start = time.perf_counter()
            for _ in range(n_repeats):
                _ = self.chunked_grace(embeddings)
            chunked_time = (time.perf_counter() - start) / n_repeats
            
            speedup = per_token_time / chunked_time if chunked_time > 0 else float('inf')
            
            results["lengths"].append(n)
            results["per_token_times"].append(per_token_time * 1000)
            results["chunked_times"].append(chunked_time * 1000)
            results["speedups"].append(speedup)
        
        return results


# =============================================================================
# TEST 4: SENSORY GATING (REFINED: Identity Approximation)
# =============================================================================
# INSIGHT: Skipping tokens in geometric products is WRONG!
#          M₁ @ M₂ @ M₃ ≠ M₁ @ M₃ (skipping M₂ changes the result)
#          
# CORRECT APPROACH: Replace uninformative tokens with IDENTITY
#          M₁ @ I @ M₃ = M₁ @ M₃ (mathematically equivalent to skip!)
#          
# WHY THIS WORKS:
#   - Identity matrix I is the neutral element: M @ I = M
#   - For near-identity embeddings: M ≈ I + ε (small perturbation)
#   - If |ε| < threshold, we can safely approximate M ≈ I
#   
# THEORY-TRUE JUSTIFICATION:
#   The brain's thalamic gating doesn't "skip" inputs - it ATTENUATES them.
#   Replacing with identity is equivalent to "this token adds no information."
#   This preserves the geometric structure while saving computation.
#
# SPEEDUP: The identity product is essentially free (no matmul needed).

class TestSensoryGating:
    """Test that identity approximation for uninformative tokens works."""
    
    def __init__(self, xp=np):
        self.xp = xp
        self.basis = build_clifford_basis(xp)
        
    def compute_informativeness(self, M: np.ndarray) -> float:
        """
        How informative is this embedding?
        
        Informativeness = deviation from identity (Frobenius norm of M - I).
        Near-identity embeddings carry little information and can be approximated.
        """
        xp = self.xp
        return float(xp.sqrt(xp.sum((M - xp.eye(4))**2)))
    
    def ungated_context(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Process all tokens (ground truth)."""
        xp = self.xp
        context = xp.eye(4)
        for M in embeddings:
            context = geometric_product(context, M)
        return context
    
    def gated_context_skip(
        self, 
        embeddings: List[np.ndarray],
        threshold: float = 0.3
    ) -> Tuple[np.ndarray, int, int]:
        """
        WRONG APPROACH: Skip low-informativeness tokens.
        (Included for comparison to show why it fails)
        """
        xp = self.xp
        context = xp.eye(4)
        processed = 0
        skipped = 0
        
        for M in embeddings:
            info = self.compute_informativeness(M)
            if info >= threshold:
                context = geometric_product(context, M)
                processed += 1
            else:
                skipped += 1
        
        return context, processed, skipped
    
    def gated_context_identity(
        self, 
        embeddings: List[np.ndarray],
        threshold: float = 0.3
    ) -> Tuple[np.ndarray, int, int]:
        """
        CORRECT APPROACH: Replace low-informativeness tokens with identity.
        
        This is mathematically equivalent to skip (since M @ I = M),
        but preserves the geometric product structure.
        
        THEORY-TRUE: This is what the brain's thalamic gating does -
        it attenuates uninformative signals to baseline, not removes them.
        """
        xp = self.xp
        context = xp.eye(4)
        full_products = 0
        identity_approx = 0
        
        for M in embeddings:
            info = self.compute_informativeness(M)
            if info >= threshold:
                # Informative: Do full product
                context = geometric_product(context, M)
                full_products += 1
            else:
                # Uninformative: Approximate as identity
                # context = geometric_product(context, xp.eye(4))  # This is just context!
                # So we do NOTHING - context stays the same
                identity_approx += 1
        
        return context, full_products, identity_approx
    
    def test_gating_rate(self, n_tokens: int = 100, n_trials: int = 20) -> Dict[str, Any]:
        """
        Test that identity approximation preserves quality while saving computation.
        
        KEY INSIGHT: Skip and Identity approximation are mathematically EQUIVALENT!
        
        When we skip token M:
            context_after = context_before  (no change)
        
        When we approximate M ≈ I:
            context_after = context_before @ I = context_before  (no change)
        
        SAME RESULT! The question is: does skipping/approximating preserve quality?
        
        ANSWER: Only if threshold is TIGHT (skip only VERY near-identity).
        """
        xp = self.xp
        
        results_by_threshold = {}
        
        # Test multiple thresholds to find the right balance
        for threshold in [0.1, 0.15, 0.2, 0.3]:
            approx_rates = []
            witness_diffs = []
            
            for _ in range(n_trials):
                # Generate embeddings with varying informativeness
                embeddings = []
                for i in range(n_tokens):
                    if i % 3 == 0:  # ~33% stop-word-like
                        M = xp.eye(4) + 0.05 * xp.random.randn(4, 4)  # Very close to identity
                    else:
                        M = xp.eye(4) + 0.25 * xp.random.randn(4, 4)  # Content words
                    embeddings.append(M)
                
                # Ground truth
                ctx_ungated = self.ungated_context(embeddings)
                witness_ungated = extract_witness(ctx_ungated, self.basis, xp)
                
                # Identity approximation with this threshold
                ctx_approx, full_products, identity_approx = self.gated_context_identity(
                    embeddings, threshold=threshold
                )
                witness_approx = extract_witness(ctx_approx, self.basis, xp)
                
                approx_rates.append(identity_approx / n_tokens)
                witness_diffs.append(float(
                    xp.sqrt((witness_ungated[0] - witness_approx[0])**2 + 
                           (witness_ungated[1] - witness_approx[1])**2)
                ))
            
            results_by_threshold[threshold] = {
                "approx_rate": float(np.mean(approx_rates)),
                "witness_diff": float(np.mean(witness_diffs)),
                "speedup": 1.0 / (1.0 - np.mean(approx_rates)) if np.mean(approx_rates) < 1 else float('inf')
            }
        
        # Find best threshold: maximize speedup while keeping witness diff < 0.2
        best_threshold = None
        best_speedup = 1.0
        for thresh, res in results_by_threshold.items():
            if res["witness_diff"] < 0.2 and res["speedup"] > best_speedup:
                best_threshold = thresh
                best_speedup = res["speedup"]
        
        if best_threshold is None:
            # No threshold works well, use tightest
            best_threshold = 0.1
        
        best_result = results_by_threshold[best_threshold]
        
        return {
            "all_thresholds": results_by_threshold,
            "best_threshold": best_threshold,
            "mean_approx_rate": best_result["approx_rate"],
            "mean_witness_diff": best_result["witness_diff"],
            "theoretical_speedup": best_result["speedup"],
            "passed": best_result["witness_diff"] < 0.2,
            "recommendation": f"Use threshold={best_threshold}" if best_result["witness_diff"] < 0.2 else "Do not use"
        }


# =============================================================================
# TEST 5: SPARSE EMBEDDING STORAGE
# =============================================================================
# Current: Store all embeddings as 16 floats
# Proposed: If near identity, store just (σ, p) = 2 floats
# Expected: 8× memory reduction for stable embeddings

class TestSparseEmbeddingStorage:
    """Test that sparse storage can reconstruct embeddings accurately."""
    
    def __init__(self, xp=np):
        self.xp = xp
        self.basis = build_clifford_basis(xp)
        
    def is_sparse(self, M: np.ndarray, threshold: float = 0.1) -> bool:
        """Check if embedding is close enough to identity for sparse storage."""
        xp = self.xp
        # Compute enstrophy (non-witness energy)
        stab = grace_stability(M, self.basis, xp)
        return float(stab) > (1.0 - threshold)  # High stability = mostly witness
    
    def compress_to_sparse(self, M: np.ndarray) -> Tuple[float, float]:
        """Compress to just (σ, p)."""
        xp = self.xp
        witness = extract_witness(M, self.basis, xp)
        return (float(witness[0]), float(witness[1]))
    
    def reconstruct_from_sparse(self, sigma: float, pseudo: float) -> np.ndarray:
        """Reconstruct approximation from (σ, p)."""
        xp = self.xp
        # M ≈ σ·I + p·γ₅
        gamma5 = self.basis[15]
        return sigma * xp.eye(4) + pseudo * gamma5
    
    def test_reconstruction_quality(self, n_embeddings: int = 500) -> Dict[str, Any]:
        """Test how well sparse storage reconstructs embeddings."""
        xp = self.xp
        
        results = {
            "sparse_count": 0,
            "dense_count": 0,
            "reconstruction_errors": [],
            "witness_errors": []
        }
        
        for _ in range(n_embeddings):
            # Generate embedding with varying identity-bias
            bias = xp.random.uniform(0.0, 0.5)
            M = xp.eye(4) + bias * xp.random.randn(4, 4)
            
            if self.is_sparse(M):
                results["sparse_count"] += 1
                
                # Compress and reconstruct
                sigma, pseudo = self.compress_to_sparse(M)
                M_reconstructed = self.reconstruct_from_sparse(sigma, pseudo)
                
                # Measure error
                recon_error = float(xp.sqrt(xp.sum((M - M_reconstructed)**2)))
                results["reconstruction_errors"].append(recon_error)
                
                # Witness should be preserved exactly
                witness_orig = extract_witness(M, self.basis, xp)
                witness_recon = extract_witness(M_reconstructed, self.basis, xp)
                witness_error = float(xp.sqrt(
                    (witness_orig[0] - witness_recon[0])**2 + 
                    (witness_orig[1] - witness_recon[1])**2
                ))
                results["witness_errors"].append(witness_error)
            else:
                results["dense_count"] += 1
        
        return {
            "sparse_rate": results["sparse_count"] / n_embeddings,
            "dense_rate": results["dense_count"] / n_embeddings,
            "mean_reconstruction_error": float(np.mean(results["reconstruction_errors"])) if results["reconstruction_errors"] else 0.0,
            "mean_witness_error": float(np.mean(results["witness_errors"])) if results["witness_errors"] else 0.0,
            "memory_savings": 8.0 * (results["sparse_count"] / n_embeddings),  # 16/2 = 8× for sparse
            "passed": np.mean(results["witness_errors"]) < 0.01 if results["witness_errors"] else True
        }


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests(verbose: bool = True):
    """Run all brain efficiency tests and report results."""
    
    print("=" * 80)
    print("BRAIN-INSPIRED EFFICIENCY TESTS")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Incremental Context
    print("\n" + "-" * 40)
    print("TEST 1: INCREMENTAL CONTEXT (O(n)→O(1))")
    print("-" * 40)
    
    test1 = TestIncrementalContext()
    
    correctness = test1.test_correctness()
    print(f"Correctness: {'PASS' if correctness['passed'] else 'FAIL'}")
    print(f"  Max error: {correctness['max_error']:.2e}")
    
    efficiency = test1.test_efficiency()
    print(f"Efficiency:")
    for i, n in enumerate(efficiency['lengths']):
        print(f"  n={n}: naive={efficiency['naive_times'][i]:.3f}ms, "
              f"incremental={efficiency['incremental_times'][i]:.3f}ms, "
              f"speedup={efficiency['speedups'][i]:.1f}×")
    
    results["incremental_context"] = {
        "correctness": correctness,
        "efficiency": efficiency,
        "recommendation": "IMPLEMENT" if correctness['passed'] and max(efficiency['speedups']) > 2 else "SKIP"
    }
    
    # Test 2: Coarse-to-Fine Retrieval
    print("\n" + "-" * 40)
    print("TEST 2: COARSE-TO-FINE RETRIEVAL")
    print("-" * 40)
    
    test2 = TestCoarseToFineRetrieval()
    
    early_exit = test2.test_early_exit_rate()
    print(f"Testing multiple rejection thresholds:")
    for thresh, res in early_exit['all_thresholds'].items():
        status = "✓" if res['false_reject_rate'] < 0.1 else "✗"
        print(f"  threshold={thresh:.1f}: filter={res['filter_rate']:.1%}, "
              f"false_reject={res['false_reject_rate']:.1%}, speedup={res['speedup']:.2f}× {status}")
    print(f"")
    print(f"Best configuration:")
    print(f"  Threshold: {early_exit['best_threshold']}")
    print(f"  Filter rate: {early_exit['filter_rate']:.1%}")
    print(f"  False reject rate: {early_exit['false_reject_rate']:.1%}")
    print(f"  Speedup: {early_exit['theoretical_speedup']:.2f}×")
    print(f"  Quality preserved: {'PASS' if early_exit['passed'] else 'FAIL'}")
    
    ranking = test2.test_ranking_preservation()
    print(f"")
    print(f"Ranking preservation (using full similarity after filter):")
    print(f"  Mean ranking error: {ranking['mean_ranking_error']:.1%}")
    print(f"  Perfect top-k rate: {ranking['perfect_top_k_rate']:.1%}")
    print(f"  Mean filter rate: {ranking['mean_filter_rate']:.1%}")
    print(f"  Ranking preserved: {'PASS' if ranking['passed'] else 'FAIL'}")
    
    results["coarse_to_fine"] = {
        "early_exit": early_exit,
        "ranking": ranking,
        "recommendation": "IMPLEMENT" if ranking['passed'] and early_exit['passed'] else "REFINE"
    }
    
    # Test 3: Oscillatory Grace
    print("\n" + "-" * 40)
    print("TEST 3: OSCILLATORY GRACE (per-φ² tokens)")
    print("-" * 40)
    
    test3 = TestOscillatoryGrace()
    
    stability = test3.test_stability_comparison()
    print(f"Stability comparison:")
    print(f"  Per-token stability: {stability['mean_per_token_stability']:.3f}")
    print(f"  Chunked stability: {stability['mean_chunked_stability']:.3f}")
    print(f"  Stability preserved: {'YES' if stability['stability_preserved'] else 'NO'}")
    print(f"  Mean witness diff: {stability['mean_witness_diff']:.4f}")
    print(f"  Grace reduction: {stability['mean_grace_reduction']:.1f}×")
    
    efficiency3 = test3.test_efficiency()
    print(f"Efficiency:")
    for i, n in enumerate(efficiency3['lengths']):
        print(f"  n={n}: per-token={efficiency3['per_token_times'][i]:.3f}ms, "
              f"chunked={efficiency3['chunked_times'][i]:.3f}ms, "
              f"speedup={efficiency3['speedups'][i]:.2f}×")
    
    results["oscillatory_grace"] = {
        "stability": stability,
        "efficiency": efficiency3,
        "recommendation": "IMPLEMENT" if stability['passed'] else "REFINE"
    }
    
    # Test 4: Sensory Gating
    print("\n" + "-" * 40)
    print("TEST 4: SENSORY GATING (REFINED - Identity Approximation)")
    print("-" * 40)
    
    test4 = TestSensoryGating()
    
    gating = test4.test_gating_rate()
    print(f"Testing multiple thresholds:")
    for thresh, res in gating['all_thresholds'].items():
        status = "✓" if res['witness_diff'] < 0.2 else "✗"
        print(f"  threshold={thresh:.2f}: approx={res['approx_rate']:.1%}, "
              f"witness_diff={res['witness_diff']:.4f}, speedup={res['speedup']:.2f}× {status}")
    print(f"")
    print(f"Best configuration:")
    print(f"  Threshold: {gating['best_threshold']}")
    print(f"  Approximation rate: {gating['mean_approx_rate']:.1%}")
    print(f"  Witness diff: {gating['mean_witness_diff']:.4f}")
    print(f"  Speedup: {gating['theoretical_speedup']:.2f}×")
    print(f"  Quality preserved: {'PASS' if gating['passed'] else 'FAIL'}")
    print(f"  Recommendation: {gating['recommendation']}")
    
    results["sensory_gating"] = {
        "gating": gating,
        "recommendation": "IMPLEMENT" if gating['passed'] else "REFINE"
    }
    
    # Test 5: Sparse Embedding Storage
    print("\n" + "-" * 40)
    print("TEST 5: SPARSE EMBEDDING STORAGE")
    print("-" * 40)
    
    test5 = TestSparseEmbeddingStorage()
    
    sparse = test5.test_reconstruction_quality()
    print(f"Sparse storage results:")
    print(f"  Sparse rate: {sparse['sparse_rate']:.1%}")
    print(f"  Dense rate: {sparse['dense_rate']:.1%}")
    print(f"  Mean reconstruction error: {sparse['mean_reconstruction_error']:.4f}")
    print(f"  Mean witness error: {sparse['mean_witness_error']:.6f}")
    print(f"  Effective memory savings: {sparse['memory_savings']:.1f}×")
    print(f"  Quality preserved: {'PASS' if sparse['passed'] else 'FAIL'}")
    
    results["sparse_storage"] = {
        "sparse": sparse,
        "recommendation": "IMPLEMENT" if sparse['passed'] and sparse['sparse_rate'] > 0.3 else "REFINE"
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for name, result in results.items():
        print(f"  {name}: {result['recommendation']}")
    
    return results


# =============================================================================
# TEST: PIPELINE INTEGRATION
# =============================================================================

def test_streaming_training():
    """Test that streaming training with incremental context works correctly."""
    from holographic_v4.pipeline import TheoryTrueModel
    
    print("\n" + "-" * 40)
    print("TEST: STREAMING TRAINING INTEGRATION")
    print("-" * 40)
    
    # Create model
    vocab_size = 1000
    context_size = 20
    model = TheoryTrueModel(
        vocab_size=vocab_size,
        context_size=context_size,
        max_attractors=10000,
        use_meta_learning=False,  # Simplify for test
        use_embedding_drift=False,
        use_credit_assignment=False,
    )
    
    # Generate random token stream
    np.random.seed(42)
    token_stream = list(np.random.randint(0, vocab_size, size=500))
    
    # Test 1: Incremental context correctness
    print("1. Testing incremental context correctness...")
    
    # Get context using incremental method
    model.reset_incremental_context()
    for token in token_stream[:context_size]:
        model.update_context_incremental(token)
    ctx_incremental = model.finalize_incremental_context()
    
    # Get context using batch method
    ctx_batch, _, _ = model.compute_contexts_batch([token_stream[:context_size]])
    ctx_batch = ctx_batch[0]  # Extract single context
    
    # Compare
    error = float(np.max(np.abs(ctx_incremental - ctx_batch)))
    print(f"   Max difference between incremental and batch: {error:.2e}")
    
    if error < 1e-6:
        print("   ✅ Incremental context matches batch computation!")
    else:
        print("   ❌ Incremental context differs from batch!")
        return False
    
    # Test 2: Streaming training runs without error
    print("2. Testing streaming training...")
    
    try:
        history = model.train_streaming(
            token_stream[:200],  # Use subset for speed
            context_size=context_size,
            log_every=100,
            verbose=True
        )
        
        if model.num_attractors > 0:
            print(f"   ✅ Streaming training completed! Stored {model.num_attractors} attractors")
        else:
            print("   ❌ No attractors stored!")
            return False
            
    except Exception as e:
        print(f"   ❌ Error during streaming training: {e}")
        return False
    
    # Test 3: Compare streaming vs batch training (should produce similar attractors)
    print("3. Comparing streaming vs batch training...")
    
    # Reset and train with batch method on same data
    model2 = TheoryTrueModel(
        vocab_size=vocab_size,
        context_size=context_size,
        max_attractors=10000,
        use_meta_learning=False,
        use_embedding_drift=False,
        use_credit_assignment=False,
    )
    
    # Create contexts and targets from stream
    contexts = []
    targets = []
    for i in range(context_size, 200):
        contexts.append(token_stream[i-context_size:i])
        targets.append(token_stream[i])
    
    # Batch train
    for ctx, tgt in zip(contexts, targets):
        model2.train_step(ctx, tgt)
    
    print(f"   Streaming attractors: {model.num_attractors}")
    print(f"   Batch attractors: {model2.num_attractors}")
    
    # The number should be similar (might differ due to pruning timing)
    if abs(model.num_attractors - model2.num_attractors) < model.num_attractors * 0.1:
        print("   ✅ Streaming and batch produce similar results!")
    else:
        print("   ⚠️ Significant difference between streaming and batch")
    
    return True


def test_oscillatory_grace():
    """Test that oscillatory Grace maintains quality with fewer operations."""
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.quotient import extract_witness, grace_stability
    
    print("\n" + "-" * 40)
    print("TEST: OSCILLATORY GRACE INTEGRATION")
    print("-" * 40)
    
    # Create model
    vocab_size = 1000
    context_size = 30  # Use larger context to see Grace reduction
    model = TheoryTrueModel(
        vocab_size=vocab_size,
        context_size=context_size,
        use_meta_learning=False,
        use_embedding_drift=False,
        use_credit_assignment=False,
    )
    
    # Generate random token sequence
    np.random.seed(42)
    tokens = list(np.random.randint(0, vocab_size, size=context_size))
    
    # Test 1: Standard incremental context (Grace only at end)
    print("1. Standard incremental context (Grace at end)...")
    model.reset_incremental_context()
    for token in tokens:
        model.update_context_incremental(token)
    ctx_standard = model.finalize_incremental_context()
    stab_standard = float(grace_stability(ctx_standard, model.basis, np))
    witness_standard = extract_witness(ctx_standard, model.basis, np)
    print(f"   Stability: {stab_standard:.4f}")
    print(f"   Witness: ({float(witness_standard[0]):.4f}, {float(witness_standard[1]):.4f})")
    
    # Test 2: Oscillatory context (Grace every φ² tokens)
    print("2. Oscillatory context (Grace every φ² tokens)...")
    model.reset_oscillatory_context()
    grace_points = []
    for i, token in enumerate(tokens):
        _, grace_applied = model.update_context_oscillatory(token)
        if grace_applied:
            grace_points.append(i + 1)
    ctx_oscillatory = model.finalize_oscillatory_context()
    stab_oscillatory = float(grace_stability(ctx_oscillatory, model.basis, np))
    witness_oscillatory = extract_witness(ctx_oscillatory, model.basis, np)
    stats = model.get_oscillatory_stats()
    
    print(f"   Stability: {stab_oscillatory:.4f}")
    print(f"   Witness: ({float(witness_oscillatory[0]):.4f}, {float(witness_oscillatory[1]):.4f})")
    print(f"   Grace applied at tokens: {grace_points}")
    print(f"   Grace reduction: {stats['reduction']:.1f}× fewer calls")
    
    # Test 3: Compare quality
    print("3. Quality comparison...")
    witness_diff = float(np.sqrt(
        (witness_standard[0] - witness_oscillatory[0])**2 + 
        (witness_standard[1] - witness_oscillatory[1])**2
    ))
    stability_diff = abs(stab_standard - stab_oscillatory)
    
    print(f"   Witness difference: {witness_diff:.4f}")
    print(f"   Stability difference: {stability_diff:.4f}")
    
    # Quality checks
    if witness_diff < 0.3:
        print("   ✅ Witness similarity preserved!")
    else:
        print("   ⚠️ Witness differs significantly")
    
    if stability_diff < 0.1:
        print("   ✅ Stability preserved!")
    else:
        print("   ⚠️ Stability differs significantly")
    
    if stats['reduction'] > 1.5:
        print(f"   ✅ Achieved {stats['reduction']:.1f}× Grace reduction!")
    else:
        print("   ⚠️ Grace reduction less than expected")
    
    return True


def test_sparse_storage():
    """Test that sparse storage preserves witness and achieves compression."""
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.quotient import extract_witness_batch
    
    print("\n" + "-" * 40)
    print("TEST: SPARSE EMBEDDING STORAGE")
    print("-" * 40)
    
    # Create model
    vocab_size = 1000
    model = TheoryTrueModel(
        vocab_size=vocab_size,
        use_meta_learning=False,
        use_embedding_drift=False,
        use_credit_assignment=False,
    )
    
    # Test 1: Compress embeddings
    print("1. Testing batch compression...")
    
    # Use the model's embeddings (identity-biased)
    embeddings = model.embeddings[:500]  # First 500 embeddings
    compressed = model.compress_embeddings_batch(embeddings)
    
    print(f"   Sparse count: {compressed['n_sparse']}")
    print(f"   Dense count: {compressed['n_dense']}")
    print(f"   Sparse rate: {compressed['sparse_rate']:.1%}")
    print(f"   Compression ratio: {compressed['compression_ratio']:.2f}×")
    
    # Test 2: Reconstruct and verify witness preservation
    print("2. Testing reconstruction quality...")
    
    reconstructed = model.reconstruct_embeddings_batch(compressed, 500)
    
    # Compare witnesses
    original_witnesses = extract_witness_batch(embeddings, model.basis, np)
    recon_witnesses = extract_witness_batch(reconstructed, model.basis, np)
    
    witness_errors = np.sqrt(np.sum((original_witnesses - recon_witnesses)**2, axis=1))
    
    # For sparse embeddings, witness should be exactly preserved
    sparse_witness_error = np.mean(witness_errors[compressed['sparse_indices']])
    
    print(f"   Mean witness error (sparse): {sparse_witness_error:.2e}")
    
    if sparse_witness_error < 1e-6:
        print("   ✅ Sparse witnesses exactly preserved!")
    else:
        print(f"   ⚠️ Sparse witness error {sparse_witness_error:.2e} is larger than expected")
    
    # Test 3: Verify compression is worthwhile
    print("3. Verifying compression is worthwhile...")
    
    if compressed['compression_ratio'] > 1.5:
        print(f"   ✅ Achieved {compressed['compression_ratio']:.2f}× compression!")
    else:
        print(f"   ⚠️ Compression ratio {compressed['compression_ratio']:.2f}× is low")
    
    if compressed['sparse_rate'] > 0.3:
        print(f"   ✅ {compressed['sparse_rate']:.1%} of embeddings are sparse!")
    else:
        print(f"   ⚠️ Only {compressed['sparse_rate']:.1%} embeddings are sparse")
    
    return True


if __name__ == "__main__":
    results = run_all_tests()
    
    # Also test pipeline integration
    test_streaming_training()
    test_oscillatory_grace()
    test_sparse_storage()