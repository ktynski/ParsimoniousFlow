"""
Test for v4.20.0 Generalization Fix — Prototype Coverage

HYPOTHESIS:
    With old threshold (φ⁻¹ ≈ 0.618): ~16 prototypes survive (too few)
    With new threshold (1-φ⁻³ ≈ 0.764): More prototypes survive (better coverage)
    
More prototypes → better semantic coverage → better generalization
"""

import numpy as np
from holographic_v4.constants import PHI_INV, PHI_INV_CUBE
from holographic_v4.algebra import build_clifford_basis, geometric_product, grace_operator
from holographic_v4.dreaming import (
    DreamingSystem,
    EpisodicEntry,
    manage_interference,
    SemanticMemory,
    SemanticPrototype,
)


def create_random_prototype(basis, support=5, seed=None):
    """Create a random prototype for testing."""
    rng = np.random.default_rng(seed)
    
    # Random matrix normalized and Grace-stabilized
    matrix = rng.normal(0, 0.3, (4, 4)).astype(np.float64)
    matrix = matrix + np.eye(4) * 0.5  # Bias toward identity
    matrix = grace_operator(matrix, basis, np)
    
    # Random target distribution
    n_targets = rng.integers(1, 5)
    targets = rng.integers(0, 1000, n_targets)
    probs = rng.random(n_targets)
    probs = probs / probs.sum()
    target_dist = {int(t): float(p) for t, p in zip(targets, probs)}
    
    return SemanticPrototype(
        prototype_matrix=matrix,
        target_distribution=target_dist,
        radius=0.5,
        support=support,
        level=0,
        vorticity_signature=rng.normal(0, 0.1, 16).astype(np.float64),
    )


def test_merging_threshold_effect():
    """
    Test: How many prototypes survive with different merging thresholds?
    
    EXPECTED:
        Old threshold (0.618): Aggressive merging, few survive
        New threshold (0.764): Conservative merging, more survive
    """
    print("\n" + "="*70)
    print("TEST: Merging Threshold Effect on Prototype Count")
    print("="*70)
    
    basis = build_clifford_basis(np)
    
    # Create 50 prototypes with varying similarity
    n_prototypes = 50
    prototypes = [create_random_prototype(basis, seed=i) for i in range(n_prototypes)]
    
    print(f"\n  Created {n_prototypes} initial prototypes")
    
    # Test both thresholds
    thresholds = [
        (PHI_INV, "OLD (φ⁻¹ ≈ 0.618)"),
        (1 - PHI_INV_CUBE, "NEW (1-φ⁻³ ≈ 0.764)"),
        (0.9, "REFERENCE (0.9)"),
    ]
    
    results = {}
    
    for threshold, label in thresholds:
        # Create fresh semantic memory for each test
        semantic_memory = SemanticMemory(basis, np, num_levels=1)
        
        # Add all prototypes
        for proto in prototypes:
            semantic_memory.add_prototype(proto, level=0)
        
        initial_count = semantic_memory.stats()['total_prototypes']
        
        # Run interference management multiple times (like multiple sleep cycles)
        for cycle in range(5):
            stats = manage_interference(
                semantic_memory,
                basis,
                np,
                similarity_threshold=threshold,
                max_merges_per_cycle=10,
                verbose=False,
            )
        
        final_count = semantic_memory.stats()['total_prototypes']
        merged = initial_count - final_count
        
        results[label] = {
            'initial': initial_count,
            'final': final_count,
            'merged': merged,
            'survival_rate': final_count / initial_count,
        }
        
        print(f"\n  {label}:")
        print(f"    Initial: {initial_count}")
        print(f"    Final:   {final_count}")
        print(f"    Merged:  {merged}")
        print(f"    Survival: {final_count / initial_count:.1%}")
    
    # Verify new threshold preserves more prototypes
    old_survival = results["OLD (φ⁻¹ ≈ 0.618)"]['final']
    new_survival = results["NEW (1-φ⁻³ ≈ 0.764)"]['final']
    
    improvement = new_survival - old_survival
    print(f"\n  IMPROVEMENT: +{improvement} prototypes with new threshold")
    print(f"  (New preserves {new_survival} vs Old preserves {old_survival})")
    
    passed = new_survival > old_survival
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: New threshold preserves more prototypes")
    
    return passed


def test_dreaming_system_prototype_growth():
    """
    Test: Does the full DreamingSystem grow more prototypes with new threshold?
    """
    print("\n" + "="*70)
    print("TEST: DreamingSystem Prototype Growth")
    print("="*70)
    
    basis = build_clifford_basis(np)
    
    # Create dreaming system
    dreaming = DreamingSystem(basis, np)
    
    # Generate random episodes
    rng = np.random.default_rng(42)
    
    def make_episode(seed):
        rng_ep = np.random.default_rng(seed)
        matrix = rng_ep.normal(0, 0.3, (4, 4)).astype(np.float64) + np.eye(4)
        matrix = grace_operator(matrix, basis, np)
        target = int(rng_ep.integers(0, 100))
        return EpisodicEntry(
            context_matrix=matrix,
            target_token=target,
            salience=float(rng_ep.random()),
            novelty=float(rng_ep.random()),
            priority=float(rng_ep.random()),
            vorticity_signature=rng_ep.normal(0, 0.1, 16).astype(np.float64),
        )
    
    # Run multiple sleep cycles
    prototype_counts = []
    for cycle in range(10):
        # Generate fresh episodes each cycle
        episodes = [make_episode(cycle * 100 + i) for i in range(100)]
        
        stats = dreaming.sleep(episodes, verbose=False)
        
        proto_count = dreaming.semantic_memory.stats()['total_prototypes']
        prototype_counts.append(proto_count)
        
        print(f"  Sleep {cycle + 1}: +{stats['prototypes_created']} created, "
              f"{stats['prototypes_merged']} merged → Total: {proto_count}")
    
    # Verify reasonable growth
    final_count = prototype_counts[-1]
    max_count = max(prototype_counts)
    
    print(f"\n  Final prototype count: {final_count}")
    print(f"  Max prototype count: {max_count}")
    print(f"  Expected (with old threshold): ~1 (collapsed!)")
    print(f"  Expected (with new threshold): >5 (diverse)")
    
    # With new threshold and combined similarity, prototypes should survive
    # The key improvement is from 1 → 10+ (10× better coverage)
    passed = final_count >= 5
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: Final count {final_count} >= 5 threshold")
    
    return passed


def test_prototype_diversity():
    """
    Test: Are preserved prototypes actually diverse (different target distributions)?
    """
    print("\n" + "="*70)
    print("TEST: Prototype Diversity (Target Distribution Entropy)")
    print("="*70)
    
    basis = build_clifford_basis(np)
    dreaming = DreamingSystem(basis, np)
    
    # Generate diverse episodes (different targets)
    rng = np.random.default_rng(42)
    
    def make_diverse_episode(target_token, seed):
        rng_ep = np.random.default_rng(seed)
        matrix = rng_ep.normal(0, 0.3, (4, 4)).astype(np.float64) + np.eye(4)
        matrix = grace_operator(matrix, basis, np)
        return EpisodicEntry(
            context_matrix=matrix,
            target_token=target_token,
            salience=0.8,
            novelty=0.9,
            priority=0.85,
            vorticity_signature=rng_ep.normal(0, 0.1, 16).astype(np.float64),
        )
    
    # Create episodes with 20 different targets (semantic diversity)
    episodes = []
    for target in range(20):
        for i in range(5):  # 5 episodes per target
            episodes.append(make_diverse_episode(target, target * 100 + i))
    
    print(f"  Created {len(episodes)} episodes with 20 distinct targets")
    
    # Run sleep
    stats = dreaming.sleep(episodes, verbose=False)
    
    # Count unique targets across all prototypes
    all_targets = set()
    for level in dreaming.semantic_memory.levels:
        for proto in level:
            all_targets.update(proto.target_distribution.keys())
    
    proto_count = dreaming.semantic_memory.stats()['total_prototypes']
    target_coverage = len(all_targets)
    
    print(f"\n  Prototypes created: {proto_count}")
    print(f"  Unique targets covered: {target_coverage} / 20")
    print(f"  Coverage ratio: {target_coverage / 20:.1%}")
    
    # Good system should cover most targets
    passed = target_coverage >= 10  # At least 50% coverage
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: Target coverage {target_coverage} >= 10")
    
    return passed


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERALIZATION FIX VERIFICATION (v4.20.0)")
    print("="*70)
    print(f"\n  OLD threshold (φ⁻¹):  {PHI_INV:.4f}")
    print(f"  NEW threshold (1-φ⁻³): {1 - PHI_INV_CUBE:.4f}")
    
    results = [
        ("Merging Threshold Effect", test_merging_threshold_effect()),
        ("Prototype Growth", test_dreaming_system_prototype_growth()),
        ("Prototype Diversity", test_prototype_diversity()),
    ]
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed
    
    print("\n" + "="*70)
    if all_passed:
        print("  ✅ ALL TESTS PASSED — Ready for training run")
    else:
        print("  ❌ SOME TESTS FAILED — Review results above")
    print("="*70)
