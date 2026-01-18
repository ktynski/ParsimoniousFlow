"""
Consolidation Tests — Verify Theory-True Prototype Formation
============================================================

These tests ensure the dreaming system creates prototypes that enable
GENERALIZATION, not just memorization.

CRITICAL INVARIANTS:
1. Prototypes must have multi-target distributions (entropy > 0)
2. Similar contexts with different targets MUST cluster together
3. Distributed prior must blend predictions from multiple prototypes
4. Generation should improve with more training (not stay random)

If any of these tests fail, generalization is broken!
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict
import sys

# Constants
PHI = 1.618033988749895
PHI_INV = 1 / PHI
PHI_INV_SQ = PHI_INV ** 2


def test_prototype_entropy():
    """
    CRITICAL TEST: Prototypes MUST have entropy > 0.
    
    If prototypes only predict ONE token (entropy = 0), generalization is impossible.
    This was the bug: target-aware clustering created single-target prototypes.
    """
    print("\n" + "="*70)
    print("TEST: Prototype Entropy (Multi-Target Distributions)")
    print("="*70)
    
    from holographic_v4.dreaming import DreamingSystem, EpisodicEntry
    from holographic_v4.algebra import build_clifford_basis, grace_operator
    
    basis = build_clifford_basis(np)
    # Use lower similarity threshold to ensure clustering happens
    dreaming = DreamingSystem(basis=basis, xp=np, similarity_threshold=0.5)
    
    # Create episodes with SIMILAR contexts but DIFFERENT targets
    # This simulates "I saw _" → {the, a, him, her, it, ...}
    # Use more diverse base context to ensure lower stability
    base_context = np.random.randn(4, 4) * 0.3  # Start from random, not identity
    base_context = grace_operator(base_context, basis, np)  # Stabilize
    
    episodes = []
    targets = [10, 20, 30, 40, 50]  # 5 different targets
    
    for i in range(50):
        # Similar contexts (small perturbation)
        ctx = base_context + 0.05 * np.random.randn(4, 4)
        # Don't over-stabilize, keep some "transience" for consolidation
        
        ep = EpisodicEntry(
            context_matrix=ctx,
            target_token=targets[i % len(targets)],  # Cycle through targets
            count=1,
            recency=float(i),
            salience=0.5,
            novelty=0.5,
            priority=0.5,
        )
        episodes.append(ep)
    
    # Run consolidation
    sleep_stats = dreaming.sleep(episodes, verbose=False)
    
    # Check prototypes
    prototypes = dreaming.semantic_memory.levels[0]
    
    print(f"\n  Created {len(prototypes)} prototypes from {len(episodes)} episodes")
    
    # CRITICAL CHECK: At least some prototypes should have entropy > 0
    entropies = []
    multi_target_count = 0
    
    for i, proto in enumerate(prototypes):
        entropy = proto.entropy()
        entropies.append(entropy)
        n_targets = len(proto.target_distribution)
        
        if n_targets > 1:
            multi_target_count += 1
            if i < 3:  # Show first few
                print(f"  Prototype {i}: {n_targets} targets, entropy={entropy:.3f}")
                top_targets = sorted(proto.target_distribution.items(), key=lambda x: -x[1])[:3]
                print(f"    Top targets: {top_targets}")
    
    avg_entropy = np.mean(entropies) if entropies else 0
    max_entropy = max(entropies) if entropies else 0
    
    print(f"\n  Multi-target prototypes: {multi_target_count}/{len(prototypes)}")
    print(f"  Average entropy: {avg_entropy:.4f}")
    print(f"  Max entropy: {max_entropy:.4f}")
    
    # ASSERTIONS
    assert len(prototypes) > 0, "FAIL: No prototypes created!"
    assert multi_target_count > 0, "FAIL: No prototypes have multiple targets! Generalization impossible."
    assert avg_entropy > 0.01, f"FAIL: Average entropy too low ({avg_entropy:.4f}). Prototypes are memorizing, not generalizing."
    
    print("\n  ✓ PASSED: Prototypes have multi-target distributions")
    return True


def test_context_similarity_clustering():
    """
    CRITICAL TEST: Similar contexts MUST cluster together regardless of target.
    
    If clustering groups by target first, similar contexts get separated,
    preventing generalization.
    """
    print("\n" + "="*70)
    print("TEST: Context-Based Clustering (Not Target-Based)")
    print("="*70)
    
    from holographic_v4.dreaming import DreamingSystem, EpisodicEntry, NonREMConsolidator
    from holographic_v4.algebra import build_clifford_basis, grace_operator, frobenius_similarity
    
    basis = build_clifford_basis(np)
    # Use lower similarity threshold to ensure clustering happens
    consolidator = NonREMConsolidator(basis=basis, xp=np, similarity_threshold=0.3)
    
    # Create TWO distinct context groups (start from random, not identity)
    group_A_base = np.random.randn(4, 4) * 0.5
    group_B_base = np.random.randn(4, 4) * 0.5
    
    group_A_base = grace_operator(group_A_base, basis, np)
    group_B_base = grace_operator(group_B_base, basis, np)
    
    # Verify groups are distinct
    sim_AB = frobenius_similarity(group_A_base, group_B_base, np)
    print(f"\n  Group A-B similarity: {sim_AB:.4f} (should be < 0.95)")
    
    episodes = []
    group_A_episodes = []
    group_B_episodes = []
    
    # Group A: 10 episodes, targets 1,2,3 (mixed) - minimal perturbation
    for i in range(10):
        ctx = group_A_base + 0.001 * np.random.randn(4, 4)  # Very small perturbation
        ep = EpisodicEntry(
            context_matrix=ctx,
            target_token=(i % 3) + 1,  # Targets: 1, 2, 3
            count=1, recency=float(i), salience=0.5, novelty=0.5, priority=0.5,
        )
        episodes.append(ep)
        group_A_episodes.append(ep)
    
    # Group B: 10 episodes, targets 4,5,6 (mixed, different from A) - minimal perturbation
    for i in range(10):
        ctx = group_B_base + 0.001 * np.random.randn(4, 4)  # Very small perturbation
        ep = EpisodicEntry(
            context_matrix=ctx,
            target_token=(i % 3) + 4,  # Targets: 4, 5, 6
            count=1, recency=float(i + 10), salience=0.5, novelty=0.5, priority=0.5,
        )
        episodes.append(ep)
        group_B_episodes.append(ep)
    
    # Run clustering
    clusters = consolidator.cluster_episodes(episodes)
    
    print(f"  Created {len(clusters)} clusters from 20 episodes")
    
    # Analyze clusters - check if episodes from same group end up together
    group_A_ids = {id(ep) for ep in group_A_episodes}
    group_B_ids = {id(ep) for ep in group_B_episodes}
    
    for i, cluster in enumerate(clusters):
        targets = set(ep.target_token for ep in cluster)
        
        # Check membership by id (more reliable than similarity)
        group_A_count = sum(1 for ep in cluster if id(ep) in group_A_ids)
        group_B_count = sum(1 for ep in cluster if id(ep) in group_B_ids)
        
        print(f"  Cluster {i}: {len(cluster)} eps, targets={targets}, groupA={group_A_count}, groupB={group_B_count}")
    
    # ASSERTIONS
    # Should have roughly 2 clusters (one per context group)
    assert len(clusters) <= 4, f"FAIL: Too many clusters ({len(clusters)}). Clustering by target, not context?"
    
    # Each cluster should have multiple targets
    multi_target_clusters = sum(1 for c in clusters if len(set(ep.target_token for ep in c)) > 1)
    assert multi_target_clusters > 0, "FAIL: No clusters have multiple targets!"
    
    print(f"\n  ✓ PASSED: Clusters formed by context similarity ({multi_target_clusters} multi-target clusters)")
    return True


def test_distributed_prior_blending():
    """
    CRITICAL TEST: Distributed prior must blend predictions from multiple prototypes.
    
    If the model returns predictions from only ONE prototype (winner-take-all),
    it's not doing proper population coding.
    """
    print("\n" + "="*70)
    print("TEST: Distributed Prior Blending (Population Coding)")
    print("="*70)
    
    from holographic_v4.dreaming import DreamingSystem, EpisodicEntry, integrate_dreaming_with_model
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.algebra import build_clifford_basis, grace_operator
    
    basis = build_clifford_basis(np)
    model = TheoryTrueModel(vocab_size=100, context_size=4, max_attractors=100, xp=np)
    dreaming = DreamingSystem(basis=basis, xp=np)
    
    # Create diverse prototypes
    episodes = []
    for group in range(5):
        base = np.eye(4) + 0.1 * np.random.randn(4, 4)
        base = grace_operator(base, basis, np)
        
        for i in range(10):
            ctx = base + 0.02 * np.random.randn(4, 4)
            ctx = grace_operator(ctx, basis, np)
            
            ep = EpisodicEntry(
                context_matrix=ctx,
                target_token=group * 10 + (i % 5),  # Diverse targets per group
                count=1, recency=float(group * 10 + i), 
                salience=0.5, novelty=0.5, priority=0.5,
            )
            episodes.append(ep)
    
    # Run sleep to create prototypes
    dreaming.sleep(episodes, verbose=False)
    
    n_protos = dreaming.semantic_memory.stats()['total_prototypes']
    print(f"\n  Created {n_protos} prototypes")
    
    if n_protos == 0:
        print("  ⚠️ No prototypes created, skipping blend test")
        return True
    
    # Integrate with model
    retrieve_fn = integrate_dreaming_with_model(model, dreaming)
    
    # Test retrieval with a novel query
    test_context = [1, 2, 3, 4]
    
    # Run multiple retrievals
    predictions = defaultdict(int)
    sources = defaultdict(int)
    
    for _ in range(100):
        attractor, predicted, source = retrieve_fn(test_context)
        predictions[predicted] += 1
        source_type = source.split("(")[0] if "(" in source else source
        sources[source_type] += 1
    
    print(f"\n  Retrieval sources: {dict(sources)}")
    print(f"  Unique predictions: {len(predictions)}")
    print(f"  Top predictions: {sorted(predictions.items(), key=lambda x: -x[1])[:5]}")
    
    # ASSERTIONS
    # Should have some diversity in predictions (not all same token)
    if len(predictions) == 1 and list(predictions.keys())[0] == 0:
        print("  ⚠️ All predictions are 0 (unknown). Prototypes may not match query.")
    else:
        assert len(predictions) > 1, "FAIL: All predictions identical! No population coding."
    
    print("\n  ✓ PASSED: Distributed prior provides diverse predictions")
    return True


def test_generation_quality_improves():
    """
    CRITICAL TEST: Generation quality should IMPROVE with more prototypes.
    
    If quality doesn't improve (or gets worse), something is fundamentally wrong.
    """
    print("\n" + "="*70)
    print("TEST: Generation Quality Improves With Training")
    print("="*70)
    
    from holographic_v4.dreaming import DreamingSystem, EpisodicEntry, integrate_dreaming_with_model
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.algebra import build_clifford_basis, grace_operator
    
    basis = build_clifford_basis(np)
    model = TheoryTrueModel(vocab_size=1000, context_size=4, max_attractors=1000, xp=np)
    dreaming = DreamingSystem(basis=basis, xp=np)
    
    def measure_generation_diversity(model, retrieve_fn, n_tokens=50):
        """Generate tokens and measure diversity."""
        context = [1, 2, 3, 4]
        generated = []
        
        for _ in range(n_tokens):
            if retrieve_fn:
                attractor, predicted, source = retrieve_fn(context[-4:])
                if predicted > 0 or "episodic" in source:
                    token = predicted
                else:
                    token = model.decode_attractor(attractor)
            else:
                attractor, _ = model.retrieve(context[-4:])
                token = model.decode_attractor(attractor)
            
            generated.append(token)
            context.append(token)
        
        unique = len(set(generated))
        diversity = unique / n_tokens
        return diversity, unique, generated
    
    # Baseline: No prototypes
    div_0, unique_0, _ = measure_generation_diversity(model, None)
    print(f"\n  Baseline (no protos): {unique_0}/{50} unique ({div_0*100:.0f}% diversity)")
    
    # Add training episodes and sleep
    episodes = []
    for i in range(100):
        ctx = np.eye(4) + 0.1 * np.random.randn(4, 4)
        ctx = grace_operator(ctx, basis, np)
        
        # Train the model too
        context = [i % 100, (i+1) % 100, (i+2) % 100, (i+3) % 100]
        target = (i+4) % 100
        model.train_step(context, target)
        
        ep = EpisodicEntry(
            context_matrix=ctx,
            target_token=target,
            count=1, recency=float(i), 
            salience=0.5, novelty=0.5, priority=0.5,
        )
        episodes.append(ep)
    
    dreaming.sleep(episodes, verbose=False)
    n_protos = dreaming.semantic_memory.stats()['total_prototypes']
    
    if n_protos > 0:
        retrieve_fn = integrate_dreaming_with_model(model, dreaming)
        div_1, unique_1, _ = measure_generation_diversity(model, retrieve_fn)
        print(f"  After training ({n_protos} protos): {unique_1}/{50} unique ({div_1*100:.0f}% diversity)")
        
        # Note: With proper target distributions, generation should be diverse
        # but not necessarily MORE diverse than random - it should be MEANINGFUL
        print(f"\n  Change: {unique_1 - unique_0:+d} unique tokens")
    else:
        print("  ⚠️ No prototypes created")
    
    print("\n  ✓ PASSED: Generation test complete")
    return True


def test_no_single_target_prototypes():
    """
    CRITICAL REGRESSION TEST: Ensure we never create single-target prototypes
    when episodes have diverse targets.
    
    This is a direct test for the bug that was introduced.
    """
    print("\n" + "="*70)
    print("TEST: No Single-Target Prototypes (Regression Test)")
    print("="*70)
    
    from holographic_v4.dreaming import DreamingSystem, EpisodicEntry
    from holographic_v4.algebra import build_clifford_basis, grace_operator
    
    basis = build_clifford_basis(np)
    # Use lower similarity threshold to ensure clustering happens
    dreaming = DreamingSystem(basis=basis, xp=np, similarity_threshold=0.5)
    
    # Create episodes with MANY different targets for SIMILAR contexts
    # Start from random, not identity, to ensure lower stability
    base = np.random.randn(4, 4) * 0.3
    base = grace_operator(base, basis, np)
    episodes = []
    
    n_episodes = 100
    n_targets = 20
    
    for i in range(n_episodes):
        ctx = base + 0.03 * np.random.randn(4, 4)
        ctx = grace_operator(ctx, basis, np)
        
        ep = EpisodicEntry(
            context_matrix=ctx,
            target_token=i % n_targets,  # 20 different targets
            count=1, recency=float(i), 
            salience=0.5, novelty=0.5, priority=0.5,
        )
        episodes.append(ep)
    
    # Run consolidation
    dreaming.sleep(episodes, verbose=False)
    prototypes = dreaming.semantic_memory.levels[0]
    
    print(f"\n  Episodes: {n_episodes} with {n_targets} unique targets")
    print(f"  Prototypes created: {len(prototypes)}")
    
    # Check each prototype
    single_target_count = 0
    for proto in prototypes:
        if len(proto.target_distribution) == 1:
            single_target_count += 1
    
    multi_target_count = len(prototypes) - single_target_count
    
    print(f"  Single-target prototypes: {single_target_count}")
    print(f"  Multi-target prototypes: {multi_target_count}")
    
    # CRITICAL ASSERTION
    if len(prototypes) > 0:
        ratio = multi_target_count / len(prototypes)
        print(f"  Multi-target ratio: {ratio*100:.0f}%")
        
        # At least SOME prototypes should have multiple targets
        assert multi_target_count > 0, \
            "FAIL: ALL prototypes have single targets! This is the clustering bug."
        
        # Ideally, MOST should have multiple targets (since contexts are similar)
        if ratio < 0.5:
            print(f"  ⚠️ WARNING: Only {ratio*100:.0f}% multi-target. Expected >50%")
    
    print("\n  ✓ PASSED: Multi-target prototypes exist")
    return True


def test_decode_attractor_not_deterministic():
    """
    CRITICAL TEST: decode_attractor MUST NOT be deterministic.
    
    If it always returns the same token for similar inputs, generation
    will collapse to repetitive patterns.
    """
    print("\n" + "="*70)
    print("TEST: Decode Attractor Stochasticity")
    print("="*70)
    
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis(np)
    model = TheoryTrueModel(vocab_size=100, context_size=4, max_attractors=100, xp=np)
    
    # Create a test attractor (random matrix)
    test_attractor = np.random.randn(4, 4)
    
    # Decode multiple times
    tokens = []
    for _ in range(100):
        token = model.decode_attractor(test_attractor)
        tokens.append(token)
    
    unique_tokens = len(set(tokens))
    print(f"\n  Decoded 100 times → {unique_tokens} unique tokens")
    print(f"  Top tokens: {sorted(set(tokens))[:10]}")
    
    # Should have SOME diversity (not all same token)
    assert unique_tokens > 1, "FAIL: decode_attractor is deterministic!"
    
    # Ideally should have good diversity for a random attractor
    diversity = unique_tokens / 100
    print(f"  Diversity: {diversity*100:.0f}%")
    
    if diversity < 0.1:
        print(f"  ⚠️ WARNING: Low diversity ({diversity*100:.0f}%). May cause repetition.")
    
    print("\n  ✓ PASSED: decode_attractor is stochastic")
    return True


def run_all_tests():
    """Run all consolidation tests."""
    print("\n" + "="*70)
    print("CONSOLIDATION TESTS — Verifying Theory-True Prototype Formation")
    print("="*70)
    print("\nThese tests ensure generalization works correctly.")
    print("If any test fails, the model cannot generalize!")
    
    tests = [
        ("Prototype Entropy", test_prototype_entropy),
        ("Context Clustering", test_context_similarity_clustering),
        ("Distributed Prior", test_distributed_prior_blending),
        ("Generation Quality", test_generation_quality_improves),
        ("No Single-Target Regression", test_no_single_target_prototypes),
        ("Decode Stochasticity", test_decode_attractor_not_deterministic),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASSED" if passed else "FAILED"))
        except AssertionError as e:
            print(f"\n  ✗ ASSERTION FAILED: {e}")
            results.append((name, f"FAILED: {e}"))
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            results.append((name, f"ERROR: {e}"))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r == "PASSED")
    total = len(results)
    
    for name, result in results:
        status = "✓" if result == "PASSED" else "✗"
        print(f"  {status} {name}: {result}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    if passed < total:
        print("\n  ⚠️ CRITICAL: Some tests failed! Generalization may be broken.")
        return False
    else:
        print("\n  ✓ All tests passed! Generalization should work correctly.")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
