"""
Debug consolidation to understand why it's not creating prototypes.
"""
import numpy as np
from typing import List
from collections import defaultdict

from holographic_prod.memory.holographic_memory_unified import HolographicMemory
from holographic_prod.dreaming.structures import EpisodicEntry
from holographic_prod.dreaming.dreaming_system import DreamingSystem
from holographic_prod.dreaming.consolidation import NonREMConsolidator
from holographic_prod.core.algebra import build_clifford_basis, grace_with_stability_batch
from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ


def create_memory(vocab_size=1000, levels=3, use_gpu=False):
    return HolographicMemory(vocab_size=vocab_size, max_levels=levels, use_gpu=use_gpu)


def create_episodic_entry(memory, ctx: List[int], target: int) -> EpisodicEntry:
    ctx_mat = memory.tower._embed_sequence(ctx)
    return EpisodicEntry(context_matrix=ctx_mat, target_token=target)


def test_consolidation_step_by_step():
    """
    Step through consolidation to understand where it fails.
    """
    print("=" * 70)
    print("CONSOLIDATION DEBUG: Step by step")
    print("=" * 70)
    
    memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
    basis = build_clifford_basis(np)
    
    # Create realistic patterns - same context, same target (like real text)
    # "the cat sat on" -> "the" (many times)
    patterns = []
    for _ in range(10):
        patterns.append(([1, 2, 3, 4], 5))  # "the cat sat on" -> "the" x10
    for _ in range(10):
        patterns.append(([10, 20, 30, 40], 50))  # Different pattern
    for _ in range(10):
        patterns.append(([100, 200, 300, 400], 500))  # Another pattern
    
    print(f"Created {len(patterns)} patterns")
    
    episodes = []
    for ctx, target in patterns:
        entry = create_episodic_entry(memory, ctx, target)
        episodes.append(entry)
    
    # Check stability of episodes
    matrices = np.stack([ep.context_matrix for ep in episodes])
    _, stabilities, _ = grace_with_stability_batch(matrices, basis, np)
    
    print(f"\nEpisode stability:")
    print(f"  Min: {stabilities.min():.4f}")
    print(f"  Max: {stabilities.max():.4f}")
    print(f"  Mean: {stabilities.mean():.4f}")
    print(f"  Threshold (φ⁻²): {PHI_INV_SQ:.4f}")
    
    transient = (stabilities < PHI_INV_SQ).sum()
    print(f"  Transient episodes: {transient}/{len(episodes)}")
    
    # Check target redundancy
    target_counts = defaultdict(int)
    for ep in episodes:
        target_counts[ep.target_token] += 1
    
    print(f"\nTarget redundancy:")
    for target, count in sorted(target_counts.items()):
        status = "≥3" if count >= 3 else "<3"
        print(f"  Target {target}: {count} occurrences ({status})")
    
    redundant = sum(1 for ep in episodes if target_counts[ep.target_token] >= 3)
    print(f"  Redundant episodes: {redundant}/{len(episodes)}")
    
    # Run consolidation with verbose
    print("\n" + "=" * 40)
    print("RUNNING CONSOLIDATION:")
    print("=" * 40)
    
    consolidator = NonREMConsolidator(basis=basis, xp=np, min_cluster_size=3)
    prototypes = consolidator.consolidate(episodes, verbose=True)
    
    print(f"\n==> Created {len(prototypes)} prototypes")
    
    if prototypes:
        for i, proto in enumerate(prototypes):
            print(f"  Prototype {i}: target={proto.mode_target()}, "
                  f"support={proto.support}, radius={proto.radius:.4f}")


def test_clustering_behavior():
    """
    Test the clustering step specifically.
    """
    print("\n" + "=" * 70)
    print("CLUSTERING DEBUG: Understanding basin routing")
    print("=" * 70)
    
    memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
    basis = build_clifford_basis(np)
    
    # Create patterns with EXACT SAME context (should cluster together)
    ctx = [1, 2, 3]
    episodes = []
    for i in range(10):
        entry = create_episodic_entry(memory, ctx, target=100)
        episodes.append(entry)
    
    print(f"Created {len(episodes)} episodes with SAME context {ctx}")
    
    # Check what basin keys they get
    from holographic_prod.core.algebra import grace_basin_keys_batch_direct
    
    matrices = np.stack([ep.context_matrix for ep in episodes])
    basin_keys = grace_basin_keys_batch_direct(matrices, basis, n_iters=0, resolution=0.01, xp=np)
    
    print(f"\nBasin keys (should all be SAME):")
    unique_keys = set()
    for i, bk in enumerate(basin_keys):
        key_tuple = tuple(bk.tolist())
        unique_keys.add(key_tuple)
        print(f"  Episode {i}: {bk}")
    
    print(f"\nUnique basin keys: {len(unique_keys)}")
    
    # Now test with different contexts
    print("\n" + "-" * 40)
    print("Testing with DIFFERENT contexts:")
    
    contexts = [
        [1, 2, 3],
        [1, 2, 4],  # One token different
        [10, 20, 30],  # All different
    ]
    
    for ctx in contexts:
        mat = memory.tower._embed_sequence(ctx)
        bk = grace_basin_keys_batch_direct(mat.reshape(1, 4, 4), basis, n_iters=0, resolution=0.01, xp=np)[0]
        print(f"  Context {ctx}: basin_key = {bk}")


def test_min_cluster_size_impact():
    """
    Test impact of min_cluster_size on consolidation.
    """
    print("\n" + "=" * 70)
    print("MIN_CLUSTER_SIZE IMPACT")
    print("=" * 70)
    
    memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
    basis = build_clifford_basis(np)
    
    # Create diverse patterns that might not cluster well
    np.random.seed(42)
    episodes = []
    for _ in range(20):
        ctx = list(np.random.randint(1, 100, size=3))
        target = np.random.randint(1, 50)
        entry = create_episodic_entry(memory, ctx, target)
        episodes.append(entry)
    
    print(f"Testing with {len(episodes)} random episodes")
    
    for min_size in [1, 2, 3, 5]:
        consolidator = NonREMConsolidator(basis=basis, xp=np, min_cluster_size=min_size)
        prototypes = consolidator.consolidate(episodes, verbose=False)
        print(f"  min_cluster_size={min_size}: {len(prototypes)} prototypes")


if __name__ == "__main__":
    test_consolidation_step_by_step()
    test_clustering_behavior()
    test_min_cluster_size_impact()
    
    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)
