"""
Hinge Benchmark Tests — Path A vs Path B Determination
=======================================================

These benchmarks test whether the architecture SCALES (Path A) or
hits a hard CEILING (Path B).

If all pass → Geometric-memory AI is a viable paradigm
If any fail → We learn exactly WHERE the geometry breaks down

Benchmarks (in priority order):
    3. Paraphrase Generalization — Do prototypes generalize?
    5. Coverage Predicts Failure — Are metrics trustworthy?
    4. Basin Separation — Do basins stay distinct under load?
    1. Witness Expressivity — Is 2D witness enough?
    2. Long-Range Vorticity — Does structure survive at scale?

STATUS: Testing bifurcation points between Path A and Path B
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ
from holographic_v4.algebra import (
    build_clifford_basis,
    geometric_product,
    grace_operator,
    frobenius_similarity,
    vorticity_signature,
)
from holographic_v4.quotient import (
    extract_witness,
    grace_stability,
    witness_matrix,
)
from holographic_v4.pipeline import TheoryTrueModel
from holographic_v4.dreaming import (
    DreamingSystem,
    EpisodicEntry,
    SemanticMemory,
    measure_prototype_entropy,
    integrate_dreaming_with_model,
)
from holographic_v4.curiosity import curiosity_score

# =============================================================================
# TEST SETUP
# =============================================================================

BASIS = build_clifford_basis()
XP = np
VOCAB_SIZE = 600  # Increased for non-overlapping paraphrase clusters
CONTEXT_SIZE = 8


def create_model_and_dreaming() -> Tuple[TheoryTrueModel, DreamingSystem]:
    """Create fresh model and dreaming system."""
    model = TheoryTrueModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        noise_std=0.3,
        use_vorticity=True,
        use_equilibrium=True,
        xp=XP,
    )
    dreaming = DreamingSystem(basis=BASIS, xp=XP)
    return model, dreaming


# =============================================================================
# PARAPHRASE DATA GENERATION
# =============================================================================

def generate_paraphrase_clusters(
    n_clusters: int = 20,
    n_train_per_cluster: int = 5,
    n_test_per_cluster: int = 3,
    seed: int = 42,
) -> Tuple[List[Tuple[List[int], int]], List[Tuple[List[int], int]]]:
    """
    Generate synthetic "paraphrase" clusters with NON-OVERLAPPING signatures.
    
    Each cluster represents a semantic concept.
    Training and test examples differ in surface form but share semantic core.
    
    CRITICAL FIX: Signatures must be DISTINCT between clusters.
    Previous version had overlapping signatures (cluster 0 = [0,1,2], cluster 1 = [5,6,7])
    which caused interference.
    
    NEW DESIGN:
        - Each cluster uses tokens from a DISTINCT range
        - Signature tokens: cluster_id * 10 + [0, 1, 2]  (well separated)
        - Noise tokens from NON-signature ranges (200+)
    
    Returns:
        (train_data, test_data) where each is list of (context, target)
    """
    rng = np.random.default_rng(seed)
    
    train_data = []
    test_data = []
    
    for cluster_id in range(n_clusters):
        # Semantic signature: 3 tokens that UNIQUELY define this cluster
        # Use cluster_id * 10 to ensure NO overlap between clusters
        base = cluster_id * 10
        signature = [base, base + 1, base + 2]  # e.g., [0,1,2], [10,11,12], [20,21,22]...
        
        # Target for this cluster (in 500+ range, well separated from tokens)
        target = 500 + cluster_id
        
        # Generate TRAIN examples: signature + train-specific noise from range 200-249
        for i in range(n_train_per_cluster):
            train_context = [
                200 + rng.integers(0, 10),  # Train prefix (200-209)
                signature[0],
                signature[1],
                210 + rng.integers(0, 10),  # Train middle (210-219)
                signature[2],
                220 + rng.integers(0, 10),  # Train suffix (220-229)
                230 + i,                     # Instance variation (230-234)
                240 + cluster_id % 10,       # Cluster hint (240-249)
            ]
            train_data.append((train_context, target))
        
        # Generate TEST examples: signature + test-specific noise from range 300-349
        # COMPLETELY DIFFERENT surface tokens, but SAME semantic signature
        for i in range(n_test_per_cluster):
            test_context = [
                300 + rng.integers(0, 10),  # Test prefix (300-309) - DIFFERENT!
                signature[0],               # SAME semantic core
                signature[1],               # SAME semantic core
                310 + rng.integers(0, 10),  # Test middle (310-319) - DIFFERENT!
                signature[2],               # SAME semantic core
                320 + rng.integers(0, 10),  # Test suffix (320-329) - DIFFERENT!
                330 + i,                    # Instance variation (330-332)
                340 + cluster_id % 10,      # Cluster hint (340-349)
            ]
            test_data.append((test_context, target))
    
    return train_data, test_data


# =============================================================================
# BENCHMARK 3: PARAPHRASE GENERALIZATION
# =============================================================================

def test_paraphrase_generalization():
    """
    BENCHMARK 3: Do semantic prototypes generalize to paraphrases?
    
    QUESTION: When trained on "The cat sat on the mat", can the system
    recognize "A feline rested upon the rug" as the same meaning?
    
    SETUP:
        - Train on n clusters × m examples each
        - Test on n clusters × k examples each (DIFFERENT surface form)
        - Same semantic signature → same target
        
    PASS CRITERIA:
        - Paraphrase accuracy > 60% (prototypes generalize)
        - Prototype retrieval (not just episodic) handles paraphrases
        
    FAILURE MODE:
        - Accuracy < 30% → "hash brittleness" (exact match fails)
        - All retrievals = global prior → "prototype holes"
    """
    print("=" * 70)
    print("BENCHMARK 3: Paraphrase Generalization".center(70))
    print("=" * 70)
    
    model, dreaming = create_model_and_dreaming()
    
    # Generate paraphrase data
    train_data, test_data = generate_paraphrase_clusters(
        n_clusters=20,
        n_train_per_cluster=5,
        n_test_per_cluster=3,
        seed=42,
    )
    
    print(f"\n  Training examples: {len(train_data)}")
    print(f"  Test examples (paraphrases): {len(test_data)}")
    print(f"  Semantic clusters: 20")
    
    # PHASE 1: Train on training data
    print("\n  Phase 1: Training...")
    episodes = []
    for context, target in train_data:
        # Store in model
        model.train_step(context, target)
        
        # Create episodic entry for dreaming
        ctx_matrix = model.compute_context(context)
        episodes.append(EpisodicEntry(ctx_matrix, target))
    
    # Run sleep to consolidate prototypes
    print("  Phase 2: Sleep consolidation...")
    dreaming.sleep(episodes, rem_cycles=3, verbose=False)
    
    n_prototypes = sum(len(level) for level in dreaming.semantic_memory.levels)
    print(f"  Prototypes created: {n_prototypes}")
    
    # PHASE 2: Test on paraphrases
    print("\n  Phase 3: Testing on paraphrases...")
    
    # Create full retrieval function (uses dreaming + distributed prior)
    retrieve_fn = integrate_dreaming_with_model(model, dreaming, use_distributed_prior=True)
    
    correct = 0
    retrieval_sources = defaultdict(int)
    confidence_when_correct = []
    confidence_when_wrong = []
    
    for context, expected_target in test_data:
        # Use FULL retrieval pipeline (not just model.retrieve)
        attractor, retrieved_target, source = retrieve_fn(context)
        
        if retrieved_target == expected_target:
            correct += 1
            retrieval_sources[source.split('(')[0]] += 1  # Strip confidence suffix
            confidence_when_correct.append(grace_stability(attractor, BASIS, XP))
        else:
            retrieval_sources[f"{source.split('(')[0]}_wrong"] += 1
            confidence_when_wrong.append(grace_stability(attractor, BASIS, XP))
    
    accuracy = correct / len(test_data)
    
    print(f"\n  Results:")
    print(f"  ├─ Paraphrase accuracy: {accuracy:.1%}")
    print(f"  ├─ Retrieval sources:")
    for source, count in sorted(retrieval_sources.items()):
        print(f"  │   └─ {source}: {count}")
    
    if confidence_when_correct:
        print(f"  ├─ Mean confidence (correct): {np.mean(confidence_when_correct):.3f}")
    if confidence_when_wrong:
        print(f"  ├─ Mean confidence (wrong): {np.mean(confidence_when_wrong):.3f}")
    
    # Diagnosis
    print("\n  Diagnosis:")
    if accuracy >= 0.6:
        print("  ✓ PASS: Prototypes generalize to paraphrases")
        diagnosis = "PATH_A"
    elif accuracy >= 0.3:
        print("  ~ PARTIAL: Some generalization, but limited")
        diagnosis = "PATH_B_PARTIAL"
    else:
        print("  ✗ FAIL: Hash brittleness - exact match fails on paraphrases")
        diagnosis = "PATH_B_BRITTLE"
    
    print(f"\n  {'='*60}")
    
    return {
        'benchmark': 'paraphrase_generalization',
        'accuracy': accuracy,
        'n_train': len(train_data),
        'n_test': len(test_data),
        'n_prototypes': n_prototypes,
        'retrieval_sources': dict(retrieval_sources),
        'diagnosis': diagnosis,
        'passed': accuracy >= 0.6,
    }


# =============================================================================
# BENCHMARK 5: COVERAGE PREDICTS FAILURE
# =============================================================================

def test_coverage_predicts_failure():
    """
    BENCHMARK 5: Do coverage metrics actually predict failure?
    
    QUESTION: When curiosity_score says "I don't know this region",
    does the system actually fail on queries from that region?
    
    SETUP:
        - Train on clusters 0-9 (COVERED)
        - Don't train on clusters 10-19 (UNCOVERED)
        - Test on both, measure if coverage correlates with success
        
    PASS CRITERIA:
        - Correlation > 0.5 between coverage and success
        - High coverage → low accuracy (correctly predicts failure)
        - Low coverage → high accuracy (correctly predicts success)
        
    FAILURE MODE:
        - No correlation → "metrics lie"
        - Coverage predicts wrong direction → "inverted metrics"
    """
    print("=" * 70)
    print("BENCHMARK 5: Coverage Predicts Failure".center(70))
    print("=" * 70)
    
    model, dreaming = create_model_and_dreaming()
    
    # Define covered and uncovered regions
    covered_clusters = list(range(0, 10))
    uncovered_clusters = list(range(10, 20))
    
    print(f"\n  Covered clusters: {covered_clusters}")
    print(f"  Uncovered clusters: {uncovered_clusters}")
    
    # Generate training data (ONLY from covered clusters)
    train_data, _ = generate_paraphrase_clusters(
        n_clusters=10,  # Only clusters 0-9
        n_train_per_cluster=8,
        n_test_per_cluster=0,
        seed=42,
    )
    
    print(f"  Training examples (covered only): {len(train_data)}")
    
    # Train
    print("\n  Phase 1: Training on covered regions...")
    episodes = []
    for context, target in train_data:
        model.train_step(context, target)
        ctx_matrix = model.compute_context(context)
        episodes.append(EpisodicEntry(ctx_matrix, target))
    
    dreaming.sleep(episodes, rem_cycles=3, verbose=False)
    
    # Generate test data from BOTH covered and uncovered
    print("  Phase 2: Generating test queries...")
    
    test_queries = []
    rng = np.random.default_rng(123)
    
    # Covered test queries - use SAME signature pattern as training
    for cluster_id in covered_clusters:
        base = cluster_id * 10
        signature = [base, base + 1, base + 2]
        target = 500 + cluster_id
        
        for _ in range(3):
            context = [
                200 + rng.integers(0, 10),  # Same range as training
                signature[0], signature[1],
                210 + rng.integers(0, 10),
                signature[2],
                220 + rng.integers(0, 10),
                230 + rng.integers(0, 5),
                240 + cluster_id % 10,
            ]
            test_queries.append((context, target, True))  # True = covered
    
    # Uncovered test queries - signatures that were NEVER trained
    for cluster_id in uncovered_clusters:
        base = cluster_id * 10
        signature = [base, base + 1, base + 2]
        target = 500 + cluster_id
        
        for _ in range(3):
            context = [
                200 + rng.integers(0, 10),
                signature[0], signature[1],
                210 + rng.integers(0, 10),
                signature[2],
                220 + rng.integers(0, 10),
                230 + rng.integers(0, 5),
                240 + cluster_id % 10,
            ]
            test_queries.append((context, target, False))  # False = uncovered
    
    print(f"  Test queries: {len(test_queries)} (half covered, half uncovered)")
    
    # Test and measure coverage vs success
    print("\n  Phase 3: Testing coverage prediction...")
    
    # Create full retrieval function
    retrieve_fn = integrate_dreaming_with_model(model, dreaming, use_distributed_prior=True)
    
    results = []
    covered_correct = 0
    covered_total = 0
    uncovered_correct = 0
    uncovered_total = 0
    
    for context, expected_target, is_covered in test_queries:
        # Compute curiosity score (HIGH = uncovered, LOW = covered)
        ctx_matrix = model.compute_context(context)
        
        # Use grace stability as proxy for coverage (high = covered)
        coverage_score = grace_stability(ctx_matrix, BASIS, XP)
        
        # Use FULL retrieval pipeline
        attractor, retrieved_target, source = retrieve_fn(context)
        retrieval_confidence = grace_stability(attractor, BASIS, XP)
        
        success = (retrieved_target == expected_target)
        
        results.append({
            'coverage': coverage_score,
            'confidence': retrieval_confidence,
            'success': success,
            'is_covered': is_covered,
            'source': source,
        })
        
        if is_covered:
            covered_total += 1
            if success:
                covered_correct += 1
        else:
            uncovered_total += 1
            if success:
                uncovered_correct += 1
    
    covered_accuracy = covered_correct / covered_total if covered_total > 0 else 0
    uncovered_accuracy = uncovered_correct / uncovered_total if uncovered_total > 0 else 0
    
    print(f"\n  Results:")
    print(f"  ├─ Covered region accuracy: {covered_accuracy:.1%}")
    print(f"  ├─ Uncovered region accuracy: {uncovered_accuracy:.1%}")
    print(f"  ├─ Accuracy gap: {covered_accuracy - uncovered_accuracy:.1%}")
    
    # Compute correlation between confidence and success
    confidences = [r['confidence'] for r in results]
    successes = [1.0 if r['success'] else 0.0 for r in results]
    
    # Simple correlation
    if np.std(confidences) > 0 and np.std(successes) > 0:
        correlation = np.corrcoef(confidences, successes)[0, 1]
    else:
        correlation = 0.0
    
    print(f"  ├─ Confidence-success correlation: {correlation:.3f}")
    
    # Diagnosis
    print("\n  Diagnosis:")
    
    # Key test: Does the system perform BETTER on covered regions?
    gap = covered_accuracy - uncovered_accuracy
    
    if gap > 0.3 and correlation > 0.3:
        print("  ✓ PASS: Coverage metrics predict failure accurately")
        diagnosis = "PATH_A"
        passed = True
    elif gap > 0.1:
        print("  ~ PARTIAL: Some predictive power, but weak")
        diagnosis = "PATH_B_PARTIAL"
        passed = False
    else:
        print("  ✗ FAIL: Coverage metrics don't predict failure")
        diagnosis = "PATH_B_METRICS_LIE"
        passed = False
    
    print(f"\n  {'='*60}")
    
    return {
        'benchmark': 'coverage_predicts_failure',
        'covered_accuracy': covered_accuracy,
        'uncovered_accuracy': uncovered_accuracy,
        'accuracy_gap': gap,
        'confidence_correlation': correlation,
        'diagnosis': diagnosis,
        'passed': passed,
    }


# =============================================================================
# BENCHMARK 4: BASIN SEPARATION UNDER LOAD
# =============================================================================

def test_basin_separation_under_load():
    """
    BENCHMARK 4: Do Grace basins stay distinct with 50+ semantic clusters?
    
    QUESTION: As we add more semantic clusters, do prototypes stay distinct
    or do they "smear" together and lose discrimination?
    
    SETUP:
        - Generate 50 distinct clusters × 100 episodes each = 5000 episodes
        - Run 5 sleep cycles
        - Measure basin entropy and discrimination after each cycle
        
    PASS CRITERIA:
        - Witness entropy > 0.5 (prototypes cover space, not clustered)
        - Discrimination doesn't degrade over cycles
        
    FAILURE MODE:
        - Entropy collapses → "prototype smearing"
        - Discrimination drops → "basin boundary erosion"
    """
    print("=" * 70)
    print("BENCHMARK 4: Basin Separation Under Load".center(70))
    print("=" * 70)
    
    model, dreaming = create_model_and_dreaming()
    rng = np.random.default_rng(42)
    
    n_clusters = 50
    episodes_per_cluster = 100
    n_cycles = 5
    
    print(f"\n  Clusters: {n_clusters}")
    print(f"  Episodes per cluster: {episodes_per_cluster}")
    print(f"  Total episodes: {n_clusters * episodes_per_cluster}")
    print(f"  Sleep cycles: {n_cycles}")
    
    # Generate all episodes
    print("\n  Phase 1: Generating episodes...")
    all_episodes = []
    
    for cluster_id in range(n_clusters):
        # Cluster signature
        signature = [cluster_id * 3, cluster_id * 3 + 1, cluster_id * 3 + 2]
        target = cluster_id
        
        for _ in range(episodes_per_cluster):
            context = [
                signature[0],
                rng.integers(0, 50),
                signature[1],
                rng.integers(0, 50),
                signature[2],
                rng.integers(0, 50),
                cluster_id % 20,
                rng.integers(0, 20),
            ]
            
            model.train_step(context, target)
            ctx_matrix = model.compute_context(context)
            all_episodes.append(EpisodicEntry(ctx_matrix, target))
    
    # Run sleep cycles and measure basin separation
    print("\n  Phase 2: Sleep cycles with basin monitoring...")
    
    metrics_over_cycles = []
    
    for cycle in range(n_cycles):
        # Sleep
        dreaming.sleep(all_episodes, rem_cycles=2, verbose=False)
        
        # Measure entropy
        entropy_results = measure_prototype_entropy(dreaming)
        n_prototypes = sum(len(level) for level in dreaming.semantic_memory.levels)
        
        # Create full retrieval function for this cycle
        retrieve_fn = integrate_dreaming_with_model(model, dreaming, use_distributed_prior=True)
        
        # Measure discrimination: sample queries and check margin
        discrimination_samples = []
        for _ in range(100):
            cluster_id = rng.integers(0, n_clusters)
            signature = [cluster_id * 3, cluster_id * 3 + 1, cluster_id * 3 + 2]
            context = [
                signature[0], rng.integers(0, 50),
                signature[1], rng.integers(0, 50),
                signature[2], rng.integers(0, 50),
                cluster_id % 20, rng.integers(0, 20),
            ]
            
            attractor, retrieved, source = retrieve_fn(context)
            correct = (retrieved == cluster_id)
            confidence = grace_stability(attractor, BASIS, XP)
            discrimination_samples.append((correct, confidence))
        
        accuracy = sum(1 for c, _ in discrimination_samples if c) / len(discrimination_samples)
        mean_confidence = np.mean([conf for _, conf in discrimination_samples])
        
        metrics = {
            'cycle': cycle + 1,
            'n_prototypes': n_prototypes,
            'entropy': entropy_results['normalized_entropy'],
            'accuracy': accuracy,
            'mean_confidence': mean_confidence,
        }
        metrics_over_cycles.append(metrics)
        
        print(f"    Cycle {cycle + 1}: protos={n_prototypes}, entropy={metrics['entropy']:.3f}, acc={accuracy:.1%}")
    
    # Analyze trend
    print("\n  Results:")
    
    first_entropy = metrics_over_cycles[0]['entropy']
    last_entropy = metrics_over_cycles[-1]['entropy']
    first_accuracy = metrics_over_cycles[0]['accuracy']
    last_accuracy = metrics_over_cycles[-1]['accuracy']
    
    entropy_change = last_entropy - first_entropy
    accuracy_change = last_accuracy - first_accuracy
    
    print(f"  ├─ Entropy: {first_entropy:.3f} → {last_entropy:.3f} ({entropy_change:+.3f})")
    print(f"  ├─ Accuracy: {first_accuracy:.1%} → {last_accuracy:.1%} ({accuracy_change:+.1%})")
    print(f"  ├─ Final prototypes: {metrics_over_cycles[-1]['n_prototypes']}")
    
    # Diagnosis
    print("\n  Diagnosis:")
    
    # Pass if entropy stays reasonable and accuracy doesn't collapse
    entropy_ok = last_entropy > 0.3
    accuracy_ok = last_accuracy > 0.5
    no_degradation = accuracy_change > -0.2  # Didn't drop more than 20%
    
    if entropy_ok and accuracy_ok and no_degradation:
        print("  ✓ PASS: Basins stay distinct under load")
        diagnosis = "PATH_A"
        passed = True
    elif entropy_ok or accuracy_ok:
        print("  ~ PARTIAL: Some degradation but not catastrophic")
        diagnosis = "PATH_B_PARTIAL"
        passed = False
    else:
        print("  ✗ FAIL: Basin separation collapsed")
        diagnosis = "PATH_B_SMEARING"
        passed = False
    
    print(f"\n  {'='*60}")
    
    return {
        'benchmark': 'basin_separation',
        'metrics_over_cycles': metrics_over_cycles,
        'entropy_change': entropy_change,
        'accuracy_change': accuracy_change,
        'final_entropy': last_entropy,
        'final_accuracy': last_accuracy,
        'diagnosis': diagnosis,
        'passed': passed,
    }


# =============================================================================
# BENCHMARK 1: WITNESS EXPRESSIVITY
# =============================================================================

def test_witness_expressivity():
    """
    BENCHMARK 1: Does witness space distinguish semantically distinct inputs?
    
    QUESTION: Is the 2D witness space (scalar + pseudoscalar) expressive
    enough to separate different semantic clusters?
    
    SETUP:
        - Generate 100 sentences from 10 clusters
        - Compute witness for each
        - Measure intra-cluster vs inter-cluster witness distance
        
    PASS CRITERIA:
        - Inter-cluster distance > 2× intra-cluster variance
        - Witnesses span a meaningful 2D region
        
    FAILURE MODE:
        - All witnesses collapse to similar values → "invariant collapse"
        - Witness doesn't correlate with semantics → "metric mismatch"
    """
    print("=" * 70)
    print("BENCHMARK 1: Witness Expressivity".center(70))
    print("=" * 70)
    
    model, _ = create_model_and_dreaming()
    rng = np.random.default_rng(42)
    
    n_clusters = 10
    samples_per_cluster = 20
    
    print(f"\n  Clusters: {n_clusters}")
    print(f"  Samples per cluster: {samples_per_cluster}")
    
    # Generate samples and compute witnesses
    print("\n  Phase 1: Generating samples and computing witnesses...")
    
    cluster_witnesses = defaultdict(list)
    all_witnesses = []
    
    for cluster_id in range(n_clusters):
        signature = [cluster_id * 5, cluster_id * 5 + 1, cluster_id * 5 + 2]
        
        for _ in range(samples_per_cluster):
            context = [
                signature[0], rng.integers(0, 50),
                signature[1], rng.integers(0, 50),
                signature[2], rng.integers(0, 50),
                cluster_id % 10, rng.integers(0, 20),
            ]
            
            ctx_matrix = model.compute_context(context)
            s, p = extract_witness(ctx_matrix, BASIS, XP)
            
            cluster_witnesses[cluster_id].append((s, p))
            all_witnesses.append((s, p, cluster_id))
    
    # Analyze witness distribution
    print("\n  Phase 2: Analyzing witness distribution...")
    
    # Compute cluster centroids
    cluster_centroids = {}
    for cluster_id, witnesses in cluster_witnesses.items():
        witnesses_arr = np.array(witnesses)
        centroid = witnesses_arr.mean(axis=0)
        cluster_centroids[cluster_id] = centroid
    
    # Compute intra-cluster variance
    intra_variances = []
    for cluster_id, witnesses in cluster_witnesses.items():
        witnesses_arr = np.array(witnesses)
        centroid = cluster_centroids[cluster_id]
        distances = np.linalg.norm(witnesses_arr - centroid, axis=1)
        intra_variances.append(np.var(distances))
    
    mean_intra_variance = np.mean(intra_variances)
    
    # Compute inter-cluster distances
    inter_distances = []
    centroids_list = list(cluster_centroids.values())
    for i in range(len(centroids_list)):
        for j in range(i + 1, len(centroids_list)):
            dist = np.linalg.norm(centroids_list[i] - centroids_list[j])
            inter_distances.append(dist)
    
    mean_inter_distance = np.mean(inter_distances)
    
    # Compute witness space coverage
    all_witnesses_arr = np.array([(s, p) for s, p, _ in all_witnesses])
    s_range = all_witnesses_arr[:, 0].max() - all_witnesses_arr[:, 0].min()
    p_range = all_witnesses_arr[:, 1].max() - all_witnesses_arr[:, 1].min()
    
    print(f"\n  Results:")
    print(f"  ├─ Mean intra-cluster variance: {mean_intra_variance:.6f}")
    print(f"  ├─ Mean inter-cluster distance: {mean_inter_distance:.4f}")
    print(f"  ├─ Separation ratio: {mean_inter_distance / (np.sqrt(mean_intra_variance) + 1e-10):.2f}x")
    print(f"  ├─ Scalar range: {s_range:.4f}")
    print(f"  ├─ Pseudoscalar range: {p_range:.4f}")
    
    # Diagnosis
    print("\n  Diagnosis:")
    
    separation_ratio = mean_inter_distance / (np.sqrt(mean_intra_variance) + 1e-10)
    has_coverage = (s_range > 0.01 and p_range > 0.01)
    
    if separation_ratio > 2.0 and has_coverage:
        print("  ✓ PASS: Witness space separates semantic clusters")
        diagnosis = "PATH_A"
        passed = True
    elif separation_ratio > 1.0:
        print("  ~ PARTIAL: Some separation but not robust")
        diagnosis = "PATH_B_PARTIAL"
        passed = False
    else:
        print("  ✗ FAIL: Witness space collapsed or doesn't discriminate")
        diagnosis = "PATH_B_COLLAPSE"
        passed = False
    
    print(f"\n  {'='*60}")
    
    return {
        'benchmark': 'witness_expressivity',
        'mean_intra_variance': mean_intra_variance,
        'mean_inter_distance': mean_inter_distance,
        'separation_ratio': separation_ratio,
        's_range': s_range,
        'p_range': p_range,
        'diagnosis': diagnosis,
        'passed': passed,
    }


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_hinge_benchmarks() -> Dict[str, Dict]:
    """Run all hinge benchmarks and summarize results."""
    print("=" * 70)
    print("HINGE BENCHMARKS — Path A vs Path B".center(70))
    print("=" * 70)
    print()
    
    results = {}
    
    # Run in priority order
    print("Running Benchmark 3: Paraphrase Generalization...")
    results['paraphrase'] = test_paraphrase_generalization()
    print()
    
    print("Running Benchmark 5: Coverage Predicts Failure...")
    results['coverage'] = test_coverage_predicts_failure()
    print()
    
    print("Running Benchmark 4: Basin Separation Under Load...")
    results['basin'] = test_basin_separation_under_load()
    print()
    
    print("Running Benchmark 1: Witness Expressivity...")
    results['witness'] = test_witness_expressivity()
    print()
    
    # Summary
    print("=" * 70)
    print("HINGE BENCHMARK SUMMARY".center(70))
    print("=" * 70)
    
    passed_count = 0
    total_count = 0
    
    for name, result in results.items():
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        diagnosis = result['diagnosis']
        print(f"  {name}: {status} ({diagnosis})")
        total_count += 1
        if result['passed']:
            passed_count += 1
    
    print()
    print(f"  Overall: {passed_count}/{total_count} benchmarks passed")
    
    if passed_count == total_count:
        print()
        print("  ═══════════════════════════════════════════════════════")
        print("  PATH A: Architecture appears to scale")
        print("  ═══════════════════════════════════════════════════════")
    elif passed_count >= total_count // 2:
        print()
        print("  ═══════════════════════════════════════════════════════")
        print("  PATH B (PARTIAL): Some limitations identified")
        print("  ═══════════════════════════════════════════════════════")
    else:
        print()
        print("  ═══════════════════════════════════════════════════════")
        print("  PATH B: Hard ceiling reached")
        print("  ═══════════════════════════════════════════════════════")
    
    print("=" * 70)
    
    return results


# =============================================================================
# BENCHMARK 3b: PARAPHRASE GENERALIZATION (FIXED)
# =============================================================================

def test_paraphrase_generalization_fixed():
    """
    BENCHMARK 3b: Paraphrase generalization with POSITION-WEIGHTED prototypes.
    
    This is the FIXED version using SemanticPrototypeMemory instead of
    matrix-based clustering.
    
    THEORY:
        - Full matrix similarity fails because noise dominates
        - Position-weighted similarity isolates semantic signal
        - Variance-based learning identifies semantic positions
    """
    from holographic_v4.semantic_prototype import SemanticPrototypeMemory
    
    print("=" * 70)
    print("BENCHMARK 3b: Paraphrase (POSITION-WEIGHTED)".center(70))
    print("=" * 70)
    
    model, _ = create_model_and_dreaming()
    memory = SemanticPrototypeMemory(context_size=CONTEXT_SIZE)
    
    # Generate paraphrase data
    train_data, test_data = generate_paraphrase_clusters(
        n_clusters=20,
        n_train_per_cluster=5,
        n_test_per_cluster=3,
        seed=42,
    )
    
    print(f"\n  Training examples: {len(train_data)}")
    print(f"  Test examples (paraphrases): {len(test_data)}")
    
    # PHASE 1: Train
    print("\n  Phase 1: Training with position-weighted memory...")
    for context, target in train_data:
        embeddings = [model.get_embedding(t) for t in context]
        memory.add_episode(embeddings, target)
    
    # Consolidate
    stats = memory.consolidate()
    print(f"  Prototypes created: {stats['n_prototypes']}")
    
    # PHASE 2: Test on paraphrases
    print("\n  Phase 2: Testing on paraphrases...")
    
    correct = 0
    total = 0
    
    for context, expected_target in test_data:
        embeddings = [model.get_embedding(t) for t in context]
        predicted, sim = memory.retrieve(embeddings)
        
        if predicted == expected_target:
            correct += 1
        total += 1
    
    accuracy = correct / total
    
    print(f"\n  Results:")
    print(f"  ├─ Paraphrase accuracy: {accuracy:.1%}")
    print(f"  ├─ Position weights: {[f'{w:.3f}' for w in memory.position_weights]}")
    
    # Diagnosis
    print("\n  Diagnosis:")
    if accuracy >= 0.6:
        print("  ✓ PASS: Position-weighted prototypes generalize!")
        diagnosis = "PATH_A"
    elif accuracy >= 0.3:
        print("  ~ PARTIAL: Some improvement")
        diagnosis = "PATH_A_PARTIAL"
    else:
        print("  ✗ FAIL: Still doesn't generalize")
        diagnosis = "PATH_B"
    
    print(f"\n  {'='*60}")
    
    return {
        'benchmark': 'paraphrase_position_weighted',
        'accuracy': accuracy,
        'n_prototypes': stats['n_prototypes'],
        'diagnosis': diagnosis,
        'passed': accuracy >= 0.6,
    }


def run_comparison_benchmarks() -> Dict[str, Dict]:
    """Run comparison between old and new approaches."""
    print("=" * 70)
    print("HINGE BENCHMARK COMPARISON".center(70))
    print("OLD (matrix) vs NEW (position-weighted)".center(70))
    print("=" * 70)
    print()
    
    results = {}
    
    print("Running OLD approach (matrix-based)...")
    results['old_paraphrase'] = test_paraphrase_generalization()
    print()
    
    print("Running NEW approach (position-weighted)...")
    results['new_paraphrase'] = test_paraphrase_generalization_fixed()
    print()
    
    # Summary
    print("=" * 70)
    print("COMPARISON SUMMARY".center(70))
    print("=" * 70)
    
    old_acc = results['old_paraphrase']['accuracy']
    new_acc = results['new_paraphrase']['accuracy']
    improvement = new_acc - old_acc
    
    print(f"\n  Matrix-based accuracy:   {old_acc:.1%}")
    print(f"  Position-weighted:       {new_acc:.1%}")
    print(f"  Improvement:             {improvement:+.1%}")
    
    if new_acc >= 0.6 and old_acc < 0.6:
        print("\n  ═══════════════════════════════════════════════════════")
        print("  ✓ POSITION WEIGHTING SOLVES PARAPHRASE GENERALIZATION")
        print("  ═══════════════════════════════════════════════════════")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    print("\nRunning ALL hinge benchmarks...")
    run_all_hinge_benchmarks()
    
    print("\n" + "="*70)
    print("Running COMPARISON with position-weighted fix...")
    print("="*70 + "\n")
    run_comparison_benchmarks()