"""
SCCMU Limitation Tests
======================

This module tests the core limitations identified in the theoretical analysis:

1. COVERAGE PROBLEM: What happens with never-seen contexts?
2. KEY BRITTLENESS: How sensitive is retrieval to small changes?
3. CAPACITY/INTERFERENCE: How do collisions and blending affect accuracy?
4. AMBIGUITY PROBLEM: Can we handle multi-modal targets?
5. COMPRESSION GAP: Does memory scale with experience?

Each test is designed to MEASURE the limitation, not to pass/fail.
We need to understand the actual behavior before deciding if "Dreaming" is needed.

Theory Reference:
    SCCMU = binding + retrieval (perfect when key matches)
    Transformers = statistical generalization (decent when key is novel)
    
    The question: Is our current system hitting these walls?
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Import our implementation
import sys
sys.path.insert(0, '/Users/fractlphoneroom1/Desktop/ParsimoniousFlow')

from holographic_v4 import (
    TheoryTrueModel,
    build_clifford_basis,
    grace_operator,
    vorticity_magnitude,
)
from holographic_v4.algebra import (
    frobenius_similarity,
    geometric_product,
)
from holographic_v4.quotient import (
    compute_enstrophy,
    quotient_similarity,
)
from holographic_v4.constants import PHI, PHI_INV


def print_section(title: str):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =============================================================================
# LIMITATION 1: COVERAGE PROBLEM
# =============================================================================

def test_coverage_problem(verbose: bool = True) -> Dict[str, Any]:
    """
    Test: What happens when we query with a context we've never seen?
    
    Theory prediction:
        - Exact match accuracy = 100% for seen contexts
        - Novel context accuracy = depends on how similarity generalizes
        - If we just return nearest neighbor, it could be arbitrarily wrong
    
    What we measure:
        - Accuracy on EXACT seen contexts
        - Accuracy on SIMILAR but not identical contexts
        - Accuracy on COMPLETELY NOVEL contexts
        - Distribution of similarity scores for hits vs misses
    """
    if verbose:
        print_section("LIMITATION 1: COVERAGE PROBLEM")
        print()
        print("  Question: What happens when we query with a never-seen context?")
        print()
    
    np.random.seed(42)
    
    # Create model with moderate vocabulary
    model = TheoryTrueModel(
        vocab_size=500,
        context_size=4,
        max_attractors=10000,
        noise_std=0.3,
        use_adaptive_similarity=True,
    )
    
    # Create training data with clear structure
    # Pattern: [A, B, C, D] -> E (deterministic)
    training_contexts = []
    training_targets = []
    
    # Train on specific patterns
    for i in range(100):
        ctx = [i % 50, (i + 10) % 50, (i + 20) % 50, (i + 30) % 50]
        target = (i + 40) % 50  # Deterministic target
        training_contexts.append(ctx)
        training_targets.append(target)
        model.train_step(ctx, target)
    
    results = {
        "exact_match": {"correct": 0, "total": 0, "similarities": []},
        "similar_context": {"correct": 0, "total": 0, "similarities": []},
        "novel_context": {"correct": 0, "total": 0, "similarities": []},
    }
    
    # Test 1: EXACT matches (should be near 100%)
    if verbose:
        print("  Test 1: Exact match retrieval (seen contexts)")
    for ctx, expected in zip(training_contexts[:50], training_targets[:50]):
        attractor, predicted = model.retrieve(ctx)
        
        # Compute similarity to best match
        ctx_rep = model.compute_context(ctx)
        if model.num_attractors > 0:
            sims = [frobenius_similarity(ctx_rep, model.attractor_matrices[i], np) 
                    for i in range(model.num_attractors)]
            best_sim = max(sims)
        else:
            best_sim = 0.0
        
        results["exact_match"]["similarities"].append(best_sim)
        results["exact_match"]["total"] += 1
        if predicted == expected:
            results["exact_match"]["correct"] += 1
    
    exact_acc = results["exact_match"]["correct"] / results["exact_match"]["total"]
    avg_exact_sim = np.mean(results["exact_match"]["similarities"])
    if verbose:
        print(f"    Accuracy: {exact_acc:.1%}")
        print(f"    Avg similarity: {avg_exact_sim:.4f}")
    
    # Test 2: SIMILAR contexts (one token changed)
    if verbose:
        print()
        print("  Test 2: Similar context retrieval (one token different)")
    for ctx, expected in zip(training_contexts[:50], training_targets[:50]):
        # Change one token slightly
        similar_ctx = ctx.copy()
        similar_ctx[0] = (similar_ctx[0] + 1) % 50  # Small perturbation
        
        attractor, predicted = model.retrieve(similar_ctx)
        
        ctx_rep = model.compute_context(similar_ctx)
        if model.num_attractors > 0:
            sims = [frobenius_similarity(ctx_rep, model.attractor_matrices[i], np) 
                    for i in range(model.num_attractors)]
            best_sim = max(sims)
        else:
            best_sim = 0.0
        
        results["similar_context"]["similarities"].append(best_sim)
        results["similar_context"]["total"] += 1
        if predicted == expected:
            results["similar_context"]["correct"] += 1
    
    similar_acc = results["similar_context"]["correct"] / results["similar_context"]["total"]
    avg_similar_sim = np.mean(results["similar_context"]["similarities"])
    if verbose:
        print(f"    Accuracy: {similar_acc:.1%}")
        print(f"    Avg similarity: {avg_similar_sim:.4f}")
    
    # Test 3: COMPLETELY NOVEL contexts
    if verbose:
        print()
        print("  Test 3: Novel context retrieval (never seen)")
    for i in range(50):
        # Use tokens that were never in training
        novel_ctx = [200 + i, 210 + i, 220 + i, 230 + i]
        expected = 240 + i  # This pattern was never trained
        
        attractor, predicted = model.retrieve(novel_ctx)
        
        ctx_rep = model.compute_context(novel_ctx)
        if model.num_attractors > 0:
            sims = [frobenius_similarity(ctx_rep, model.attractor_matrices[i], np) 
                    for i in range(model.num_attractors)]
            best_sim = max(sims)
        else:
            best_sim = 0.0
        
        results["novel_context"]["similarities"].append(best_sim)
        results["novel_context"]["total"] += 1
        if predicted == expected:
            results["novel_context"]["correct"] += 1
    
    novel_acc = results["novel_context"]["correct"] / results["novel_context"]["total"]
    avg_novel_sim = np.mean(results["novel_context"]["similarities"])
    if verbose:
        print(f"    Accuracy: {novel_acc:.1%}")
        print(f"    Avg similarity: {avg_novel_sim:.4f}")
    
    # Analysis
    if verbose:
        print()
        print("  ANALYSIS:")
        print(f"    Coverage cliff detected: {exact_acc - novel_acc:.1%} drop")
        print(f"    Similarity gap: {avg_exact_sim - avg_novel_sim:.4f}")
        
        if exact_acc > 0.9 and novel_acc < 0.1:
            print("    → SEVERE: Near-perfect on seen, near-zero on novel")
            print("    → This is the 'binding vs generalization' limitation")
        elif exact_acc > novel_acc + 0.3:
            print("    → MODERATE: Significant coverage cliff")
        else:
            print("    → MILD: Some generalization is happening")
    
    results["coverage_cliff"] = exact_acc - novel_acc
    results["similarity_gap"] = avg_exact_sim - avg_novel_sim
    
    return results


# =============================================================================
# LIMITATION 2: KEY BRITTLENESS
# =============================================================================

def test_key_brittleness(verbose: bool = True) -> Dict[str, Any]:
    """
    Test: How sensitive is retrieval to small changes in context?
    
    Theory prediction:
        - Multiplicative composition can be sensitive
        - Single token change → changes entire product
        - Insertions/deletions change every subsequent multiplication
    
    What we measure:
        - Similarity between original and perturbed contexts
        - Accuracy under token substitution
        - Accuracy under token insertion/deletion
    """
    if verbose:
        print_section("LIMITATION 2: KEY BRITTLENESS")
        print()
        print("  Question: How sensitive is retrieval to small context changes?")
        print()
    
    np.random.seed(42)
    
    model = TheoryTrueModel(
        vocab_size=200,
        context_size=4,
        max_attractors=5000,
        noise_std=0.3,
    )
    
    # Train on patterns
    contexts = []
    targets = []
    for i in range(200):
        ctx = [i % 30, (i + 5) % 30, (i + 10) % 30, (i + 15) % 30]
        target = (i + 20) % 30
        contexts.append(ctx)
        targets.append(target)
        model.train_step(ctx, target)
    
    results = {
        "single_token_swap": {"similarities": [], "accuracy": 0, "total": 0},
        "synonym_substitution": {"similarities": [], "accuracy": 0, "total": 0},
        "reordering": {"similarities": [], "accuracy": 0, "total": 0},
    }
    
    # Test 1: Single token swap
    if verbose:
        print("  Test 1: Single token swap (first token)")
    for ctx, expected in zip(contexts[:50], targets[:50]):
        original_rep = model.compute_context(ctx)
        
        # Swap first token with similar token
        perturbed = ctx.copy()
        perturbed[0] = (perturbed[0] + 1) % 30
        perturbed_rep = model.compute_context(perturbed)
        
        sim = frobenius_similarity(original_rep, perturbed_rep, np)
        results["single_token_swap"]["similarities"].append(sim)
        
        _, predicted = model.retrieve(perturbed)
        results["single_token_swap"]["total"] += 1
        if predicted == expected:
            results["single_token_swap"]["accuracy"] += 1
    
    swap_acc = results["single_token_swap"]["accuracy"] / results["single_token_swap"]["total"]
    avg_swap_sim = np.mean(results["single_token_swap"]["similarities"])
    if verbose:
        print(f"    Context similarity after swap: {avg_swap_sim:.4f}")
        print(f"    Accuracy after swap: {swap_acc:.1%}")
    
    # Test 2: Synonym-like substitution (nearby token)
    if verbose:
        print()
        print("  Test 2: Synonym substitution (all tokens shifted by 1)")
    for ctx, expected in zip(contexts[:50], targets[:50]):
        original_rep = model.compute_context(ctx)
        
        # Shift all tokens by 1 (simulating synonyms)
        synonym_ctx = [(t + 1) % 30 for t in ctx]
        synonym_rep = model.compute_context(synonym_ctx)
        
        sim = frobenius_similarity(original_rep, synonym_rep, np)
        results["synonym_substitution"]["similarities"].append(sim)
        
        _, predicted = model.retrieve(synonym_ctx)
        results["synonym_substitution"]["total"] += 1
        if predicted == expected:
            results["synonym_substitution"]["accuracy"] += 1
    
    syn_acc = results["synonym_substitution"]["accuracy"] / results["synonym_substitution"]["total"]
    avg_syn_sim = np.mean(results["synonym_substitution"]["similarities"])
    if verbose:
        print(f"    Context similarity after synonym: {avg_syn_sim:.4f}")
        print(f"    Accuracy after synonym: {syn_acc:.1%}")
    
    # Test 3: Reordering (swap adjacent tokens)
    if verbose:
        print()
        print("  Test 3: Reordering (swap adjacent tokens)")
    for ctx, expected in zip(contexts[:50], targets[:50]):
        original_rep = model.compute_context(ctx)
        
        # Swap tokens 1 and 2
        reordered = [ctx[0], ctx[2], ctx[1], ctx[3]]
        reordered_rep = model.compute_context(reordered)
        
        sim = frobenius_similarity(original_rep, reordered_rep, np)
        results["reordering"]["similarities"].append(sim)
        
        _, predicted = model.retrieve(reordered)
        results["reordering"]["total"] += 1
        if predicted == expected:
            results["reordering"]["accuracy"] += 1
    
    reorder_acc = results["reordering"]["accuracy"] / results["reordering"]["total"]
    avg_reorder_sim = np.mean(results["reordering"]["similarities"])
    if verbose:
        print(f"    Context similarity after reorder: {avg_reorder_sim:.4f}")
        print(f"    Accuracy after reorder: {reorder_acc:.1%}")
    
    # Analysis
    if verbose:
        print()
        print("  ANALYSIS:")
        if avg_swap_sim < 0.5:
            print(f"    → HIGH BRITTLENESS: Single token change drops similarity to {avg_swap_sim:.2f}")
        elif avg_swap_sim < 0.8:
            print(f"    → MODERATE BRITTLENESS: Similarity degrades to {avg_swap_sim:.2f}")
        else:
            print(f"    → LOW BRITTLENESS: Similarity maintained at {avg_swap_sim:.2f}")
        
        if reorder_acc > 0.5:
            print("    → Reordering somewhat tolerated (vorticity helps?)")
        else:
            print("    → Reordering breaks retrieval (order encoding working)")
    
    return results


# =============================================================================
# LIMITATION 3: CAPACITY AND INTERFERENCE
# =============================================================================

def test_capacity_interference(verbose: bool = True) -> Dict[str, Any]:
    """
    Test: How do collisions and blending affect accuracy?
    
    Theory prediction:
        - More memories → denser space → accidental nearest neighbors
        - Hebbian blending literally blends targets
        - Need memory management policy at scale
    
    What we measure:
        - Accuracy vs number of attractors
        - Collision rate (same key, different target)
        - Blending effect (multiple updates to same key)
    """
    if verbose:
        print_section("LIMITATION 3: CAPACITY & INTERFERENCE")
        print()
        print("  Question: How do collisions and memory density affect accuracy?")
        print()
    
    np.random.seed(42)
    
    results = {
        "accuracy_vs_size": [],
        "collision_rate": 0,
        "blend_stability": [],
    }
    
    # Test 1: Accuracy vs memory size
    if verbose:
        print("  Test 1: Accuracy vs memory size")
    
    for max_attractors in [100, 500, 1000, 5000, 10000]:
        model = TheoryTrueModel(
            vocab_size=500,
            context_size=4,
            max_attractors=max_attractors,
            noise_std=0.3,
        )
        
        # Train
        test_pairs = []
        for i in range(min(max_attractors * 2, 5000)):
            ctx = [i % 100, (i + 10) % 100, (i + 20) % 100, (i + 30) % 100]
            target = (i + 40) % 100
            model.train_step(ctx, target)
            if i < 200:
                test_pairs.append((ctx, target))
        
        # Test
        correct = 0
        for ctx, expected in test_pairs:
            _, predicted = model.retrieve(ctx)
            if predicted == expected:
                correct += 1
        
        acc = correct / len(test_pairs)
        results["accuracy_vs_size"].append({
            "max_attractors": max_attractors,
            "actual_attractors": model.num_attractors,
            "accuracy": acc,
        })
        
        if verbose:
            print(f"    max={max_attractors:>5}, actual={model.num_attractors:>5}, acc={acc:.1%}")
    
    # Test 2: Collision detection
    if verbose:
        print()
        print("  Test 2: Collision rate (same context, different targets)")
    
    model = TheoryTrueModel(
        vocab_size=200,
        context_size=4,
        max_attractors=10000,
        noise_std=0.3,
    )
    
    collision_count = 0
    total_updates = 0
    
    # Train the same context with different targets
    base_ctx = [1, 2, 3, 4]
    targets_seen = set()
    
    from holographic_v4.constants import PHI_INV_SQ
    
    for i in range(100):
        target = i % 10  # Only 10 unique targets for this context
        
        # Check if this is a collision (same context, different target)
        ctx_rep = model.compute_context_representation(base_ctx)
        _, _, confidence, _ = model.holographic_memory.retrieve(ctx_rep)
        if confidence >= PHI_INV_SQ and target not in targets_seen:
            collision_count += 1
        targets_seen.add(target)
        
        model.train_step(base_ctx, target)
        total_updates += 1
    
    results["collision_rate"] = collision_count / total_updates
    if verbose:
        print(f"    Collision rate: {results['collision_rate']:.1%}")
        print(f"    (Same context trained with {len(targets_seen)} different targets)")
    
    # Test 3: Blending stability
    if verbose:
        print()
        print("  Test 3: Blending stability (repeated updates)")
    
    # What happens when we update the same context many times?
    model = TheoryTrueModel(
        vocab_size=200,
        context_size=4,
        max_attractors=1000,
        noise_std=0.3,
    )
    
    ctx = [5, 10, 15, 20]
    target_a = 100
    target_b = 101
    
    # Alternate between targets
    predictions = []
    for i in range(50):
        target = target_a if i % 2 == 0 else target_b
        model.train_step(ctx, target)
        
        _, predicted = model.retrieve(ctx)
        predictions.append(predicted)
    
    # Did it stabilize? Or oscillate?
    last_10 = predictions[-10:]
    unique_predictions = len(set(last_10))
    
    results["blend_stability"] = {
        "final_predictions": last_10,
        "unique_in_last_10": unique_predictions,
        "stabilized": unique_predictions == 1,
    }
    
    if verbose:
        print(f"    After 50 alternating updates:")
        print(f"    Last 10 predictions: {last_10}")
        print(f"    Unique predictions: {unique_predictions}")
        if unique_predictions == 1:
            print("    → Converged to single target (Hebbian blending working)")
        else:
            print("    → Unstable (oscillating between targets)")
    
    # Analysis
    if verbose:
        print()
        print("  ANALYSIS:")
        acc_drop = (results["accuracy_vs_size"][0]["accuracy"] - 
                   results["accuracy_vs_size"][-1]["accuracy"])
        if acc_drop > 0.2:
            print(f"    → SEVERE: Accuracy drops {acc_drop:.1%} as memory grows")
        elif acc_drop > 0.05:
            print(f"    → MODERATE: Some interference at scale")
        else:
            print(f"    → MILD: Accuracy relatively stable")
    
    return results


# =============================================================================
# LIMITATION 4: AMBIGUITY PROBLEM
# =============================================================================

def test_ambiguity_problem(verbose: bool = True) -> Dict[str, Any]:
    """
    Test: Can we handle contexts with multiple valid targets?
    
    Theory prediction:
        - Language is inherently multi-modal
        - Deterministic equilibrium picks one basin
        - Need multi-attractor or stochasticity
    
    What we measure:
        - Behavior when same context has multiple targets
        - Distribution of predictions
        - Does the system represent uncertainty?
    """
    if verbose:
        print_section("LIMITATION 4: AMBIGUITY PROBLEM")
        print()
        print("  Question: Can we handle multi-modal targets?")
        print()
    
    np.random.seed(42)
    
    model = TheoryTrueModel(
        vocab_size=200,
        context_size=4,
        max_attractors=5000,
        noise_std=0.3,
    )
    
    # Create ambiguous training data
    # Same context [1,2,3,4] can have targets: 100, 101, 102, 103, 104
    ambiguous_ctx = [1, 2, 3, 4]
    valid_targets = [100, 101, 102, 103, 104]
    
    # Train with all valid targets (uniformly)
    for _ in range(50):  # 10 times each
        for target in valid_targets:
            model.train_step(ambiguous_ctx, target)
    
    # Test: what does it predict?
    if verbose:
        print("  Test: Ambiguous context with 5 valid targets")
        print(f"    Context: {ambiguous_ctx}")
        print(f"    Valid targets: {valid_targets}")
        print()
    
    # Multiple retrievals
    predictions = []
    for _ in range(100):
        _, predicted = model.retrieve(ambiguous_ctx)
        predictions.append(predicted)
    
    # Analyze distribution
    from collections import Counter
    pred_counts = Counter(predictions)
    
    results = {
        "valid_targets": valid_targets,
        "predictions": dict(pred_counts),
        "entropy": 0,
        "covers_all": False,
    }
    
    if verbose:
        print("  Prediction distribution (100 retrievals):")
        for target, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
            marker = "✓" if target in valid_targets else "✗"
            print(f"    {target}: {count}x {marker}")
    
    # Compute entropy
    total = sum(pred_counts.values())
    probs = [count/total for count in pred_counts.values()]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    results["entropy"] = entropy
    
    # Does it cover all valid targets?
    covered = sum(1 for t in valid_targets if t in pred_counts)
    results["covers_all"] = covered == len(valid_targets)
    
    if verbose:
        print()
        print("  ANALYSIS:")
        print(f"    Entropy: {entropy:.3f} (max for 5 targets = {np.log(5):.3f})")
        print(f"    Covered targets: {covered}/{len(valid_targets)}")
        
        if len(pred_counts) == 1:
            print("    → DETERMINISTIC: Always returns same target")
            print("    → System cannot represent multi-modality")
        elif entropy < 0.5:
            print("    → LOW ENTROPY: Strong preference for one mode")
        elif results["covers_all"]:
            print("    → GOOD: Covers all valid targets (some multi-modality)")
        else:
            print("    → PARTIAL: Some modes but not all")
    
    return results


# =============================================================================
# LIMITATION 5: COMPRESSION GAP
# =============================================================================

def test_compression_gap(verbose: bool = True) -> Dict[str, Any]:
    """
    Test: Does memory grow linearly with experience?
    
    Theory prediction:
        - Transformers compress into fixed parameters
        - SCCMU stores explicitly → grows with experience
        - Need consolidation/abstraction
    
    What we measure:
        - Attractor count vs training samples
        - Redundancy in stored attractors
        - Potential for compression
    """
    if verbose:
        print_section("LIMITATION 5: COMPRESSION GAP")
        print()
        print("  Question: Does memory grow linearly with experience?")
        print()
    
    np.random.seed(42)
    
    model = TheoryTrueModel(
        vocab_size=500,
        context_size=4,
        max_attractors=50000,
        noise_std=0.3,
    )
    
    results = {
        "growth_curve": [],
        "redundancy_estimate": 0,
    }
    
    # Train and measure growth
    if verbose:
        print("  Training with growth tracking:")
    
    checkpoints = [1000, 5000, 10000, 20000, 50000]
    samples = 0
    
    for target_samples in checkpoints:
        while samples < target_samples:
            ctx = [
                samples % 100,
                (samples + 10) % 100,
                (samples + 20) % 100,
                (samples + 30) % 100,
            ]
            target = (samples + 40) % 100
            model.train_step(ctx, target)
            samples += 1
        
        results["growth_curve"].append({
            "samples": samples,
            "attractors": model.num_attractors,
            "ratio": model.num_attractors / samples,
        })
        
        if verbose:
            ratio = model.num_attractors / samples
            print(f"    {samples:>6} samples → {model.num_attractors:>6} attractors ({ratio:.1%})")
    
    # Estimate redundancy by checking similarity between random attractors
    if verbose:
        print()
        print("  Redundancy analysis:")
    
    if model.num_attractors >= 100:
        sample_indices = np.random.choice(model.num_attractors, min(100, model.num_attractors), replace=False)
        
        high_sim_pairs = 0
        total_pairs = 0
        
        for i, idx_i in enumerate(sample_indices):
            for idx_j in sample_indices[i+1:]:
                sim = frobenius_similarity(
                    model.attractor_matrices[idx_i],
                    model.attractor_matrices[idx_j],
                    np
                )
                if sim > 0.9:
                    high_sim_pairs += 1
                total_pairs += 1
        
        redundancy = high_sim_pairs / total_pairs if total_pairs > 0 else 0
        results["redundancy_estimate"] = redundancy
        
        if verbose:
            print(f"    High similarity pairs (>0.9): {high_sim_pairs}/{total_pairs} ({redundancy:.1%})")
    
    # Analysis
    if verbose:
        print()
        print("  ANALYSIS:")
        final_ratio = results["growth_curve"][-1]["ratio"]
        if final_ratio > 0.8:
            print(f"    → LINEAR GROWTH: {final_ratio:.1%} of samples become attractors")
            print("    → Memory will scale with experience (needs consolidation)")
        elif final_ratio > 0.5:
            print(f"    → SUB-LINEAR: {final_ratio:.1%} of samples stored")
            print("    → Some natural deduplication happening")
        else:
            print(f"    → GOOD COMPRESSION: Only {final_ratio:.1%} stored")
    
    return results


# =============================================================================
# MAIN: Run all limitation tests
# =============================================================================

def run_all_limitation_tests():
    """Run all limitation tests and summarize findings."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  SCCMU LIMITATION ASSESSMENT  ".center(68) + "║")
    print("║" + "  Testing whether 'Dreaming' mechanism is needed  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    all_results = {}
    
    # Run each test
    all_results["coverage"] = test_coverage_problem()
    all_results["brittleness"] = test_key_brittleness()
    all_results["capacity"] = test_capacity_interference()
    all_results["ambiguity"] = test_ambiguity_problem()
    all_results["compression"] = test_compression_gap()
    
    # Summary
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  SUMMARY: DO WE NEED DREAMING?  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    needs_dreaming = []
    
    # Coverage
    cliff = all_results["coverage"]["coverage_cliff"]
    if cliff > 0.5:
        needs_dreaming.append("COVERAGE: Severe cliff detected ({:.0%})".format(cliff))
    
    # Brittleness
    swap_sim = np.mean(all_results["brittleness"]["single_token_swap"]["similarities"])
    if swap_sim < 0.7:
        needs_dreaming.append("BRITTLENESS: Low robustness to perturbation ({:.2f})".format(swap_sim))
    
    # Ambiguity
    if all_results["ambiguity"]["entropy"] < 0.5:
        needs_dreaming.append("AMBIGUITY: Cannot represent multi-modality")
    
    # Compression
    final_ratio = all_results["compression"]["growth_curve"][-1]["ratio"]
    if final_ratio > 0.5:
        needs_dreaming.append("COMPRESSION: Linear memory growth ({:.0%})".format(final_ratio))
    
    if needs_dreaming:
        print("  LIMITATIONS DETECTED:")
        for issue in needs_dreaming:
            print(f"    • {issue}")
        print()
        print("  RECOMMENDATION: Implement 'Dreaming' mechanism")
        print("    - Non-REM: Consolidation (clustering, prototypes)")
        print("    - REM: Recombination + survival filtering")
    else:
        print("  No critical limitations detected.")
        print("  Current implementation may be sufficient.")
    
    print()
    return all_results


if __name__ == "__main__":
    results = run_all_limitation_tests()
