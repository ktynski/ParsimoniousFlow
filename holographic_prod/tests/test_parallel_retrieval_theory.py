"""
Test Suite: Parallel Retrieval vs Sequential Retrieval

THEORY BEING TESTED:
    Per Complementary Learning Systems (McClelland & O'Reilly, 1995),
    hippocampus (episodic) and neocortex (semantic/holographic) process
    inputs IN PARALLEL, not sequentially.
    
    Hypothesis: Running both retrieval paths in parallel and comparing
    results will provide:
    1. Better confidence calibration (agreement → higher confidence)
    2. Conflict detection (ACC analog) for error/novelty signals
    3. Learning opportunities even on episodic cache hits
    
METRICS TO MEASURE:
    1. Agreement rate: How often episodic == holographic?
    2. Conflict prediction: Does disagreement predict actual errors?
    3. Confidence calibration: Is confidence correlated with correctness?
    4. Novel pattern detection: Can conflict identify unseen patterns?

EXPECTED OUTCOMES:
    If parallel retrieval IS beneficial:
        - High agreement (>80%) when episodic is correct
        - Low agreement (<50%) when episodic is wrong (corrupted/noise)
        - Conflict signals predict errors better than chance
        
    If parallel retrieval is NOT beneficial:
        - Random agreement (near base rate)
        - No correlation between conflict and errors
        - Added computation for no gain

Author: Testing the theory before implementation
Version: v5.12.0 (Parallel Retrieval Investigation)
"""

import numpy as np
import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE
from core.algebra import geometric_product, frobenius_cosine
from memory.multi_level_tower import MultiLevelTower


@dataclass
class ParallelRetrievalResult:
    """Result from parallel retrieval experiment."""
    context: List[int]
    actual_target: int
    
    # Episodic path
    episodic_target: Optional[int]
    episodic_confidence: float
    episodic_hit: bool
    
    # Holographic path (computed even when episodic hits)
    holographic_target: Optional[int]
    holographic_confidence: float
    
    # Computed metrics
    agreement: bool = False
    conflict_level: float = 0.0
    episodic_correct: bool = False
    holographic_correct: bool = False


@dataclass
class ExperimentResults:
    """Aggregated experiment results."""
    total_samples: int = 0
    
    # Agreement metrics
    agreement_count: int = 0
    disagreement_count: int = 0
    
    # Correctness metrics
    episodic_correct_count: int = 0
    holographic_correct_count: int = 0
    both_correct_count: int = 0
    both_wrong_count: int = 0
    
    # Conflict analysis
    conflict_when_episodic_correct: List[float] = field(default_factory=list)
    conflict_when_episodic_wrong: List[float] = field(default_factory=list)
    
    # Confidence analysis
    confidence_when_agree_correct: List[float] = field(default_factory=list)
    confidence_when_agree_wrong: List[float] = field(default_factory=list)
    confidence_when_disagree: List[float] = field(default_factory=list)
    
    # Timing
    episodic_only_time: float = 0.0
    parallel_time: float = 0.0


class ParallelRetrievalTester:
    """
    Test parallel vs sequential retrieval to validate theory.
    """
    
    def __init__(self, vocab_size: int = 1000, levels: int = 3, seed: int = 42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.tower = MultiLevelTower(vocab_size=vocab_size, levels=levels, seed=seed)
        
        # Internal episodic cache (mirrors what HolographicMemory does)
        self._episodic_cache: Dict[Tuple[int, ...], int] = {}
        
    def learn_patterns(self, n_patterns: int, context_length: int = 8) -> List[Tuple[List[int], int]]:
        """Learn patterns and return them for testing."""
        patterns = []
        
        for _ in range(n_patterns):
            context = [np.random.randint(0, self.vocab_size) for _ in range(context_length)]
            target = np.random.randint(0, self.vocab_size)
            
            # Learn in tower (holographic)
            self.tower.learn(context, target)
            
            # Store in episodic cache
            self._episodic_cache[tuple(context)] = target
            
            patterns.append((context, target))
        
        return patterns
    
    def corrupt_episodic(self, corruption_rate: float = 0.1):
        """
        Corrupt some episodic entries to simulate noise/errors.
        This tests whether holographic can detect episodic corruption.
        """
        keys = list(self._episodic_cache.keys())
        n_corrupt = int(len(keys) * corruption_rate)
        
        corrupted_keys = np.random.choice(len(keys), size=n_corrupt, replace=False)
        
        for idx in corrupted_keys:
            key = keys[idx]
            # Replace with random target
            self._episodic_cache[key] = np.random.randint(0, self.vocab_size)
        
        return n_corrupt
    
    def retrieve_parallel(self, context: List[int]) -> Tuple[Optional[int], float, Optional[int], float]:
        """
        Run BOTH episodic and holographic retrieval (the proposed parallel approach).
        
        Returns:
            (episodic_target, episodic_confidence, holographic_target, holographic_confidence)
        """
        ctx_tuple = tuple(context)
        
        # PATH 1: Episodic (O(1) hash lookup)
        episodic_target = self._episodic_cache.get(ctx_tuple)
        episodic_confidence = 1.0 if episodic_target is not None else 0.0
        
        # PATH 2: Holographic (always compute, even when episodic hits)
        holographic_target = self.tower.retrieve(context)
        
        # Compute holographic confidence
        if holographic_target is not None:
            # Get embeddings
            ctx_mat = self.tower._embed_sequence(context)
            target_emb = self.tower.embeddings[holographic_target]
            
            # Route to satellite
            sat_idx = self.tower.route_to_satellite(context)
            sat = self.tower.satellites[sat_idx]
            
            # Unbind: SO(4) inverse = transpose
            ctx_inv = ctx_mat.T
            retrieved = ctx_inv @ sat.memory
            
            # Confidence = cosine similarity
            retrieved_flat = retrieved.flatten()
            target_flat = target_emb.flatten()
            
            r_norm = np.linalg.norm(retrieved_flat)
            t_norm = np.linalg.norm(target_flat)
            
            if r_norm > 1e-10 and t_norm > 1e-10:
                holographic_confidence = float(np.dot(retrieved_flat, target_flat) / (r_norm * t_norm))
            else:
                holographic_confidence = 0.0
        else:
            holographic_confidence = 0.0
        
        return episodic_target, episodic_confidence, holographic_target, holographic_confidence
    
    def retrieve_sequential(self, context: List[int]) -> Tuple[Optional[int], float, str]:
        """
        Current sequential retrieval (episodic first, then holographic only on miss).
        
        Returns:
            (target, confidence, source)
        """
        ctx_tuple = tuple(context)
        
        # PATH 1: Episodic (short-circuit on hit)
        if ctx_tuple in self._episodic_cache:
            return self._episodic_cache[ctx_tuple], 1.0, "episodic"
        
        # PATH 2: Holographic (only on episodic miss)
        holographic_target = self.tower.retrieve(context)
        if holographic_target is not None:
            return holographic_target, PHI_INV, "holographic"  # Lower confidence for holographic
        
        return None, 0.0, "none"
    
    def run_experiment(
        self, 
        n_train: int = 500,
        n_test: int = 200,
        corruption_rate: float = 0.1,
        context_length: int = 8,
        verbose: bool = True
    ) -> ExperimentResults:
        """
        Run comprehensive experiment comparing parallel vs sequential retrieval.
        """
        results = ExperimentResults()
        
        if verbose:
            print("=" * 70)
            print("  PARALLEL RETRIEVAL THEORY TEST")
            print("=" * 70)
            print(f"  Training: {n_train} patterns")
            print(f"  Testing: {n_test} patterns")
            print(f"  Corruption rate: {corruption_rate:.1%}")
            print()
        
        # Phase 1: Learn patterns
        if verbose:
            print("  Phase 1: Learning patterns...")
        patterns = self.learn_patterns(n_train, context_length)
        
        # Store original targets before corruption
        original_targets = {tuple(ctx): tgt for ctx, tgt in patterns}
        
        # Phase 2: Corrupt some episodic entries
        if verbose:
            print(f"  Phase 2: Corrupting {corruption_rate:.1%} of episodic cache...")
        n_corrupted = self.corrupt_episodic(corruption_rate)
        if verbose:
            print(f"    Corrupted {n_corrupted} entries")
        
        # Phase 3: Test both retrieval methods
        if verbose:
            print("  Phase 3: Running retrieval experiments...")
        
        # Select test patterns (mix of seen and novel)
        test_patterns = patterns[:n_test]
        
        for context, _ in test_patterns:
            ctx_tuple = tuple(context)
            actual_target = original_targets[ctx_tuple]  # Ground truth
            
            # Run parallel retrieval
            t0 = time.perf_counter()
            ep_tgt, ep_conf, holo_tgt, holo_conf = self.retrieve_parallel(context)
            results.parallel_time += time.perf_counter() - t0
            
            # Run sequential retrieval (for timing comparison)
            t0 = time.perf_counter()
            seq_tgt, seq_conf, seq_src = self.retrieve_sequential(context)
            results.episodic_only_time += time.perf_counter() - t0
            
            # Analyze result
            result = ParallelRetrievalResult(
                context=context,
                actual_target=actual_target,
                episodic_target=ep_tgt,
                episodic_confidence=ep_conf,
                episodic_hit=(ep_tgt is not None),
                holographic_target=holo_tgt,
                holographic_confidence=holo_conf,
            )
            
            # Agreement
            result.agreement = (ep_tgt == holo_tgt) if (ep_tgt is not None and holo_tgt is not None) else False
            
            # Conflict level (how strongly holographic disagrees)
            if ep_tgt != holo_tgt and holo_tgt is not None:
                result.conflict_level = holo_conf
            
            # Correctness
            result.episodic_correct = (ep_tgt == actual_target)
            result.holographic_correct = (holo_tgt == actual_target)
            
            # Aggregate
            results.total_samples += 1
            
            if result.agreement:
                results.agreement_count += 1
            else:
                results.disagreement_count += 1
            
            if result.episodic_correct:
                results.episodic_correct_count += 1
            if result.holographic_correct:
                results.holographic_correct_count += 1
            if result.episodic_correct and result.holographic_correct:
                results.both_correct_count += 1
            if not result.episodic_correct and not result.holographic_correct:
                results.both_wrong_count += 1
            
            # Conflict analysis
            if result.episodic_correct:
                results.conflict_when_episodic_correct.append(result.conflict_level)
            else:
                results.conflict_when_episodic_wrong.append(result.conflict_level)
            
            # Confidence analysis
            if result.agreement:
                if result.episodic_correct:
                    results.confidence_when_agree_correct.append(ep_conf + PHI_INV * holo_conf)
                else:
                    results.confidence_when_agree_wrong.append(ep_conf + PHI_INV * holo_conf)
            else:
                results.confidence_when_disagree.append(ep_conf * (1 - PHI_INV * result.conflict_level))
        
        return results
    
    def print_results(self, results: ExperimentResults):
        """Print comprehensive analysis of results."""
        print()
        print("=" * 70)
        print("  RESULTS: PARALLEL RETRIEVAL THEORY TEST")
        print("=" * 70)
        
        # Basic metrics
        print("\n  AGREEMENT METRICS:")
        print(f"    Total samples: {results.total_samples}")
        agreement_rate = results.agreement_count / max(1, results.total_samples)
        print(f"    Agreement rate: {agreement_rate:.1%} ({results.agreement_count}/{results.total_samples})")
        print(f"    Disagreement rate: {1 - agreement_rate:.1%} ({results.disagreement_count}/{results.total_samples})")
        
        # Correctness
        print("\n  CORRECTNESS METRICS:")
        ep_acc = results.episodic_correct_count / max(1, results.total_samples)
        holo_acc = results.holographic_correct_count / max(1, results.total_samples)
        both_acc = results.both_correct_count / max(1, results.total_samples)
        print(f"    Episodic accuracy: {ep_acc:.1%}")
        print(f"    Holographic accuracy: {holo_acc:.1%}")
        print(f"    Both correct: {both_acc:.1%}")
        print(f"    Both wrong: {results.both_wrong_count / max(1, results.total_samples):.1%}")
        
        # KEY HYPOTHESIS TEST: Does conflict predict episodic errors?
        print("\n  KEY HYPOTHESIS: Does conflict predict episodic errors?")
        
        avg_conflict_when_correct = np.mean(results.conflict_when_episodic_correct) if results.conflict_when_episodic_correct else 0
        avg_conflict_when_wrong = np.mean(results.conflict_when_episodic_wrong) if results.conflict_when_episodic_wrong else 0
        
        print(f"    Avg conflict when episodic CORRECT: {avg_conflict_when_correct:.4f}")
        print(f"    Avg conflict when episodic WRONG: {avg_conflict_when_wrong:.4f}")
        
        if avg_conflict_when_wrong > avg_conflict_when_correct * 1.5:
            print(f"    ✅ SUPPORTED: Conflict is {avg_conflict_when_wrong / max(avg_conflict_when_correct, 0.001):.1f}x higher when episodic is wrong")
            print(f"       → Parallel retrieval CAN detect episodic errors")
        elif avg_conflict_when_wrong > avg_conflict_when_correct:
            print(f"    ⚠️ WEAK: Conflict is only {avg_conflict_when_wrong / max(avg_conflict_when_correct, 0.001):.1f}x higher when wrong")
            print(f"       → Some signal, but may not be worth the compute")
        else:
            print(f"    ❌ NOT SUPPORTED: Conflict does NOT predict errors")
            print(f"       → Parallel retrieval adds compute for no benefit")
        
        # Confidence calibration
        print("\n  CONFIDENCE CALIBRATION:")
        avg_conf_agree_correct = np.mean(results.confidence_when_agree_correct) if results.confidence_when_agree_correct else 0
        avg_conf_agree_wrong = np.mean(results.confidence_when_agree_wrong) if results.confidence_when_agree_wrong else 0
        avg_conf_disagree = np.mean(results.confidence_when_disagree) if results.confidence_when_disagree else 0
        
        print(f"    Confidence when agree & correct: {avg_conf_agree_correct:.4f}")
        print(f"    Confidence when agree & wrong: {avg_conf_agree_wrong:.4f}")
        print(f"    Confidence when disagree: {avg_conf_disagree:.4f}")
        
        if avg_conf_agree_correct > avg_conf_disagree > avg_conf_agree_wrong:
            print(f"    ✅ GOOD CALIBRATION: Confidence properly ranks outcomes")
        else:
            print(f"    ❌ POOR CALIBRATION: Confidence doesn't match correctness")
        
        # Timing
        print("\n  TIMING COMPARISON:")
        print(f"    Sequential (episodic-first): {results.episodic_only_time * 1000:.2f}ms total")
        print(f"    Parallel (both paths): {results.parallel_time * 1000:.2f}ms total")
        overhead = (results.parallel_time - results.episodic_only_time) / max(results.episodic_only_time, 0.001)
        print(f"    Overhead: {overhead:.1%}")
        
        # VERDICT
        print("\n" + "=" * 70)
        print("  VERDICT")
        print("=" * 70)
        
        # Scoring
        score = 0
        
        # Test 1: Conflict predicts errors
        if avg_conflict_when_wrong > avg_conflict_when_correct * 1.5:
            score += 2
            print("  ✅ Conflict detection: EFFECTIVE (+2)")
        elif avg_conflict_when_wrong > avg_conflict_when_correct:
            score += 1
            print("  ⚠️ Conflict detection: WEAK (+1)")
        else:
            print("  ❌ Conflict detection: INEFFECTIVE (+0)")
        
        # Test 2: Holographic can correct episodic
        holo_rescues = 0
        for i in range(min(len(results.conflict_when_episodic_wrong), results.holographic_correct_count)):
            holo_rescues += 1
        if holo_rescues > results.episodic_correct_count * 0.1:
            score += 2
            print(f"  ✅ Holographic rescue rate: HIGH ({holo_rescues} rescues) (+2)")
        elif holo_rescues > 0:
            score += 1
            print(f"  ⚠️ Holographic rescue rate: LOW ({holo_rescues} rescues) (+1)")
        else:
            print("  ❌ Holographic rescue rate: NONE (+0)")
        
        # Test 3: Overhead acceptable
        if overhead < 0.5:
            score += 1
            print(f"  ✅ Compute overhead: ACCEPTABLE ({overhead:.1%}) (+1)")
        else:
            print(f"  ❌ Compute overhead: TOO HIGH ({overhead:.1%}) (+0)")
        
        # Final verdict
        print()
        if score >= 4:
            print(f"  FINAL SCORE: {score}/5")
            print("  📈 IMPLEMENT PARALLEL RETRIEVAL")
            print("     Theory is validated. Benefits outweigh costs.")
            return True
        elif score >= 2:
            print(f"  FINAL SCORE: {score}/5")
            print("  ⚠️ CONSIDER CONDITIONAL PARALLEL RETRIEVAL")
            print("     Some benefit, but only use when confidence is low.")
            return False
        else:
            print(f"  FINAL SCORE: {score}/5")
            print("  ❌ DO NOT IMPLEMENT")
            print("     Sequential retrieval is sufficient.")
            return False


def test_parallel_retrieval_hypothesis():
    """
    Main test: Validate parallel retrieval theory.
    """
    print("\n" + "=" * 70)
    print("  TESTING PARALLEL RETRIEVAL HYPOTHESIS")
    print("  (Complementary Learning Systems Theory)")
    print("=" * 70)
    
    tester = ParallelRetrievalTester(vocab_size=1000, levels=3, seed=42)
    
    # Test with different corruption rates
    all_results = []
    
    for corruption_rate in [0.0, 0.05, 0.1, 0.2]:
        print(f"\n{'=' * 70}")
        print(f"  CORRUPTION RATE: {corruption_rate:.1%}")
        print(f"{'=' * 70}")
        
        # Reset for each test
        tester = ParallelRetrievalTester(vocab_size=1000, levels=3, seed=42)
        
        results = tester.run_experiment(
            n_train=500,
            n_test=200,
            corruption_rate=corruption_rate,
            verbose=True
        )
        
        tester.print_results(results)
        all_results.append((corruption_rate, results))
    
    # Summary across corruption rates
    print("\n" + "=" * 70)
    print("  SUMMARY ACROSS CORRUPTION RATES")
    print("=" * 70)
    
    print("\n  | Corruption | Agreement | Ep. Acc | Holo Acc | Conflict (correct) | Conflict (wrong) |")
    print("  |" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 20 + "|" + "-" * 18 + "|")
    
    for rate, results in all_results:
        agreement = results.agreement_count / max(1, results.total_samples)
        ep_acc = results.episodic_correct_count / max(1, results.total_samples)
        holo_acc = results.holographic_correct_count / max(1, results.total_samples)
        conf_correct = np.mean(results.conflict_when_episodic_correct) if results.conflict_when_episodic_correct else 0
        conf_wrong = np.mean(results.conflict_when_episodic_wrong) if results.conflict_when_episodic_wrong else 0
        
        print(f"  | {rate:>8.1%} | {agreement:>9.1%} | {ep_acc:>7.1%} | {holo_acc:>8.1%} | {conf_correct:>18.4f} | {conf_wrong:>16.4f} |")
    
    print("\n  KEY INSIGHT:")
    print("  If conflict increases with corruption rate, parallel retrieval is valuable.")
    print("  If conflict stays flat, holographic doesn't detect episodic errors.")


def test_conflict_as_acc_signal():
    """
    Test: Can conflict signal serve as ACC (Anterior Cingulate Cortex) analog?
    
    ACC triggers when there are competing high-confidence responses.
    We test if episodic-holographic disagreement provides this signal.
    """
    print("\n" + "=" * 70)
    print("  TESTING CONFLICT AS ACC (ANTERIOR CINGULATE CORTEX) SIGNAL")
    print("=" * 70)
    
    tester = ParallelRetrievalTester(vocab_size=500, levels=3, seed=123)
    
    # Learn with some ambiguous patterns (same context, different targets)
    print("\n  Creating ambiguous patterns (same context → multiple targets)...")
    
    # Learn same context with different targets (simulates ambiguity)
    base_context = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Learn target A 3 times
    for _ in range(3):
        tester.tower.learn(base_context, 100)
    tester._episodic_cache[tuple(base_context)] = 100
    
    # Learn target B 2 times (creates interference)
    for _ in range(2):
        tester.tower.learn(base_context, 200)
    
    # Now test: episodic says 100, but holographic might be uncertain
    ep_tgt, ep_conf, holo_tgt, holo_conf = tester.retrieve_parallel(base_context)
    
    print(f"\n  Ambiguous pattern test:")
    print(f"    Episodic target: {ep_tgt} (confidence: {ep_conf:.3f})")
    print(f"    Holographic target: {holo_tgt} (confidence: {holo_conf:.3f})")
    
    if ep_tgt != holo_tgt:
        conflict = holo_conf
        print(f"    CONFLICT DETECTED: {conflict:.3f}")
        print(f"    → This is the ACC signal! Competing high-confidence responses.")
        
        if conflict > PHI_INV:
            print(f"    → Conflict exceeds φ⁻¹ threshold: SLOW DOWN recommended")
        else:
            print(f"    → Conflict below threshold: proceed with episodic")
    else:
        print(f"    NO CONFLICT: Both systems agree on {ep_tgt}")
    
    # Test with unambiguous pattern
    print("\n  Unambiguous pattern test:")
    unambiguous_context = [10, 20, 30, 40, 50, 60, 70, 80]
    
    for _ in range(5):
        tester.tower.learn(unambiguous_context, 300)
    tester._episodic_cache[tuple(unambiguous_context)] = 300
    
    ep_tgt, ep_conf, holo_tgt, holo_conf = tester.retrieve_parallel(unambiguous_context)
    
    print(f"    Episodic target: {ep_tgt} (confidence: {ep_conf:.3f})")
    print(f"    Holographic target: {holo_tgt} (confidence: {holo_conf:.3f})")
    
    if ep_tgt == holo_tgt:
        print(f"    AGREEMENT: Both systems confident in {ep_tgt}")
        print(f"    → No ACC signal needed: high confidence, proceed fast")
    else:
        print(f"    UNEXPECTED CONFLICT on unambiguous pattern")


def test_learning_signal_on_cache_hit():
    """
    Test: Can parallel retrieval provide learning signals even on cache hits?
    
    Theory: Even when episodic cache hits, comparing with holographic gives:
    - Prediction error signal (holographic ≠ episodic)
    - Opportunity to strengthen holographic associations
    """
    print("\n" + "=" * 70)
    print("  TESTING LEARNING SIGNAL ON CACHE HIT")
    print("=" * 70)
    
    tester = ParallelRetrievalTester(vocab_size=500, levels=3, seed=456)
    
    # Phase 1: Learn patterns
    print("\n  Phase 1: Learning 100 patterns...")
    patterns = tester.learn_patterns(100, context_length=8)
    
    # Phase 2: Test retrieval and collect "would have been predicted" signals
    print("\n  Phase 2: Collecting holographic predictions on cache hits...")
    
    prediction_errors = []
    agreement_rate = []
    
    for context, target in patterns:
        ep_tgt, ep_conf, holo_tgt, holo_conf = tester.retrieve_parallel(context)
        
        # This is a cache hit (episodic returns correct answer)
        assert ep_tgt == target, "Episodic should return exact match"
        
        # But what did holographic predict?
        if holo_tgt is not None:
            agreement = (holo_tgt == target)
            agreement_rate.append(agreement)
            
            if not agreement:
                # This is a learning signal! Holographic got it wrong.
                prediction_errors.append({
                    'context': context,
                    'actual': target,
                    'holographic_predicted': holo_tgt,
                    'holographic_confidence': holo_conf,
                })
    
    print(f"\n  Results:")
    print(f"    Patterns tested: {len(patterns)}")
    print(f"    Holographic agreement rate: {np.mean(agreement_rate):.1%}")
    print(f"    Prediction errors (learning opportunities): {len(prediction_errors)}")
    
    if len(prediction_errors) > 0:
        print(f"\n  Sample prediction errors (learning signals):")
        for i, err in enumerate(prediction_errors[:5]):
            print(f"    {i+1}. Actual: {err['actual']}, Holographic: {err['holographic_predicted']} (conf: {err['holographic_confidence']:.3f})")
        
        print(f"\n  INSIGHT: These {len(prediction_errors)} cases are opportunities to strengthen")
        print(f"           holographic associations even though episodic got it right.")
        print(f"           This is the 'learning even on cache hit' benefit.")
    else:
        print(f"\n  INSIGHT: Holographic already agrees with episodic on all patterns.")
        print(f"           No additional learning signal needed.")


if __name__ == "__main__":
    # Run all tests
    test_parallel_retrieval_hypothesis()
    test_conflict_as_acc_signal()
    test_learning_signal_on_cache_hit()
    
    print("\n" + "=" * 70)
    print("  ALL TESTS COMPLETE")
    print("=" * 70)
