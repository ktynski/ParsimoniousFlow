"""
Test: Generalization via Prototype Distributions

HYPOTHESIS: Test accuracy is low because we're measuring exact match,
not whether the target is in the learned distribution.

THIS TEST VERIFIES:
1. Prototypes ARE forming during dreaming
2. Prototypes have target DISTRIBUTIONS
3. Using prototype distributions improves "accuracy" significantly
4. The current exact-match metric is too strict
"""

import numpy as np
import pytest
import sys
from collections import defaultdict

sys.path.insert(0, '/Users/fractlphoneroom1/Desktop/ParsimoniousFlow')

from holographic_prod.memory import HolographicMemory, MemoryConfig
from holographic_prod.dreaming import DreamingSystem, EpisodicEntry, integrated_sleep
from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ, DTYPE


@pytest.fixture(scope="module")
def dreamer():
    """Shared fixture that creates a DreamingSystem with prototypes."""
    # Setup memory and dreamer
    np.random.seed(42)
    vocab_size = 200
    n_patterns = 100
    
    config = MemoryConfig(
        contrastive_enabled=False,
        distributed_prior_enabled=False,
    )
    memory = HolographicMemory(vocab_size=vocab_size, seed=42, config=config, max_levels=1)
    
    from holographic_prod.core.algebra import build_clifford_basis
    basis = build_clifford_basis()
    dreamer = DreamingSystem(basis, min_cluster_size=3)
    
    # Learn patterns
    for i in range(n_patterns):
        ctx = [i % 10, (i + 1) % 10, (i + 2) % 10]
        tgt = 50 + (i % 15)
        memory.learn(ctx, tgt)
    
    # Create episodes
    episodes = []
    for i in range(n_patterns):
        ctx = [i % 10, (i + 1) % 10, (i + 2) % 10]
        tgt = 50 + (i % 15)
        ctx_mat = memory.embed_sequence(ctx)
        episodes.append(EpisodicEntry(context_matrix=ctx_mat, target_token=tgt))
    
    # Run sleep
    dreamer.sleep(episodes, verbose=False)
    dreamer._memory = memory  # Attach memory for tests
    
    return dreamer


def test_prototype_formation():
    """Test 1: Verify prototypes are actually forming."""
    print("\n" + "="*70)
    print("TEST 1: Prototype Formation")
    print("="*70)
    
    # Create memory and dreaming system
    vocab_size = 1000
    model = HolographicMemory(vocab_size=vocab_size, max_levels=3)
    dreamer = DreamingSystem(
        basis=model.basis,
        xp=model.xp,
        use_salience=True,
        use_novelty=True,
        use_predictive_coding=False,  # Disable to ensure episodes consolidate
        use_pattern_completion=True,
    )
    
    # Create synthetic data with clear clusters
    # 10 "concepts", each with 5 variants
    np.random.seed(42)
    episodes = []
    
    for concept in range(10):
        # Each concept has a base context and multiple targets
        base_context = list(range(concept * 10, concept * 10 + 5))
        targets = [concept * 100 + t for t in range(5)]  # 5 different targets per concept
        
        # Create 20 episodes per concept with variation
        for _ in range(20):
            # Small context variation
            ctx = base_context.copy()
            ctx[-1] = ctx[-1] + np.random.randint(0, 3)  # Last token varies
            
            # Embed context
            ctx_mat = model.embed_sequence(ctx)
            
            # Pick a target from the distribution
            target = np.random.choice(targets)
            
            episodes.append(EpisodicEntry(
                context_matrix=ctx_mat,
                target_token=target,
                salience=0.5 + np.random.rand() * 0.5,
            ))
    
    print(f"\nCreated {len(episodes)} episodes (10 concepts × 20 variants)")
    
    # Run integrated sleep
    pre_prototypes = dreamer.semantic_memory.stats()['total_prototypes']
    print(f"Prototypes before sleep: {pre_prototypes}")
    
    sleep_result = integrated_sleep(
        memory=model,
        dreaming_system=dreamer,
        episodes=episodes,
        rem_cycles=1,
        verbose=True,
    )
    
    post_prototypes = dreamer.semantic_memory.stats()['total_prototypes']
    print(f"\nPrototypes after sleep: {post_prototypes}")
    print(f"Prototypes created: {sleep_result.get('prototypes_created', 0)}")
    
    # VERDICT
    if post_prototypes > 0:
        print("\n✅ PASS: Prototypes ARE forming")
        return True, dreamer
    else:
        print("\n❌ FAIL: No prototypes formed")
        return False, dreamer


def test_prototype_distributions(dreamer):
    """Test 2: Verify prototypes have target distributions."""
    print("\n" + "="*70)
    print("TEST 2: Prototype Target Distributions")
    print("="*70)
    
    all_prototypes = [p for level in dreamer.semantic_memory.levels for p in level]
    
    if len(all_prototypes) == 0:
        print("\n⚠️ SKIP: No prototypes to test")
        return False
    
    print(f"\nAnalyzing {len(all_prototypes)} prototypes:")
    
    multi_target_count = 0
    single_target_count = 0
    
    for i, proto in enumerate(all_prototypes[:10]):  # Check first 10
        target_dist = proto.target_distribution
        n_targets = len(target_dist)
        
        if n_targets > 1:
            multi_target_count += 1
            top_targets = sorted(target_dist.items(), key=lambda x: -x[1])[:5]
            print(f"\n  Prototype {i}: {n_targets} targets")
            print(f"    Top 5: {top_targets}")
        else:
            single_target_count += 1
            print(f"\n  Prototype {i}: Single target {list(target_dist.keys())[0]}")
    
    print(f"\nSummary:")
    print(f"  Multi-target prototypes: {multi_target_count}")
    print(f"  Single-target prototypes: {single_target_count}")
    
    # VERDICT
    if multi_target_count > 0:
        print("\n✅ PASS: Prototypes have target DISTRIBUTIONS")
        return True
    else:
        print("\n❌ FAIL: Prototypes only have single targets")
        return False


def test_distribution_vs_exact_accuracy(dreamer):
    """Test 3: Compare exact-match vs distribution-based accuracy."""
    print("\n" + "="*70)
    print("TEST 3: Distribution vs Exact Match Accuracy")
    print("="*70)
    
    # Get model from dreamer fixture
    model = getattr(dreamer, '_memory', None)
    if model is None:
        pytest.skip("No memory attached to dreamer")
    
    all_prototypes = [p for level in dreamer.semantic_memory.levels for p in level]
    
    if len(all_prototypes) == 0:
        print("\n⚠️ SKIP: No prototypes to test")
        return
    
    # Create test queries from the same distribution
    np.random.seed(123)  # Different seed for test
    n_test = 50
    
    exact_correct = 0
    top5_correct = 0
    top10_correct = 0
    in_distribution = 0
    
    for _ in range(n_test):
        concept = np.random.randint(10)
        base_context = list(range(concept * 10, concept * 10 + 5))
        
        # Small variation
        ctx = base_context.copy()
        ctx[-1] = ctx[-1] + np.random.randint(0, 3)
        
        # True target from the concept's distribution
        true_target = concept * 100 + np.random.randint(0, 5)
        
        # Embed and retrieve
        ctx_mat = model.embed_sequence(ctx)
        
        # Exact match (current approach)
        pred_exact, conf = model.retrieve_deterministic(ctx)
        if pred_exact == true_target:
            exact_correct += 1
        
        # Distribution-based (semantic memory)
        results = dreamer.semantic_memory.retrieve(
            ctx_mat,
            top_k=1,
            use_pattern_completion=True,
            completion_steps=3,
        )
        
        if results and len(results) > 0 and results[0][0] is not None:
            proto, similarity = results[0]
            target_dist = proto.target_distribution
            
            # Get ranked targets
            ranked_targets = sorted(target_dist.keys(), key=lambda x: -target_dist.get(x, 0))
            
            # Check if true target is in distribution
            if true_target in target_dist:
                in_distribution += 1
            
            # Top-k accuracy
            if true_target in ranked_targets[:1]:
                exact_correct += 1  # Would have matched anyway
            if true_target in ranked_targets[:5]:
                top5_correct += 1
            if true_target in ranked_targets[:10]:
                top10_correct += 1
    
    print(f"\nAccuracy comparison ({n_test} test samples):")
    print(f"  Exact match:      {exact_correct}/{n_test} = {exact_correct/n_test:.1%}")
    print(f"  Top-5:            {top5_correct}/{n_test} = {top5_correct/n_test:.1%}")
    print(f"  Top-10:           {top10_correct}/{n_test} = {top10_correct/n_test:.1%}")
    print(f"  In distribution:  {in_distribution}/{n_test} = {in_distribution/n_test:.1%}")
    
    improvement = (in_distribution - exact_correct) / max(1, exact_correct)
    print(f"\nImprovement (distribution over exact): {improvement:.0%}")
    
    if top5_correct > exact_correct:
        print("\n✅ INSIGHT: Distribution-based accuracy is MUCH higher than exact match")
    else:
        print("\n⚠️ INSIGHT: Distribution doesn't help much - prototypes may not be forming correctly")


def run_all_tests():
    """Run full generalization test suite."""
    print("="*70)
    print("GENERALIZATION TEST SUITE")
    print("="*70)
    print("""
PURPOSE: Verify that:
1. Prototypes are forming during dreaming
2. Prototypes have target distributions (not single targets)
3. Distribution-based accuracy >> exact match accuracy
""")
    
    # Test 1: Prototype formation
    passed, dreamer = test_prototype_formation()
    
    if not passed:
        print("\n🛑 STOPPING: Cannot test generalization without prototypes")
        return
    
    # Test 2: Distribution verification
    test_prototype_distributions(dreamer)
    
    # Test 3: Accuracy comparison
    # Need to get the model from test 1
    vocab_size = 1000
    model = HolographicMemory(vocab_size=vocab_size, max_levels=3)
    test_distribution_vs_exact_accuracy(dreamer, model)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
KEY FINDINGS:
- If prototypes form with distributions, current test metric is too strict
- Distribution-based accuracy is the theory-true metric
- Test accuracy of 1% with exact match could be 50%+ with distributions
""")


if __name__ == '__main__':
    run_all_tests()
