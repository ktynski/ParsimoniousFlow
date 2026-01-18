"""
Test embedding structure evolution through contrastive learning.

This test verifies:
1. Initial embeddings have no structure (random)
2. Contrastive learning pulls co-occurring tokens together
3. After N updates, similar tokens have higher similarity
"""
import numpy as np
from typing import List, Tuple

from holographic_prod.memory.holographic_memory_unified import HolographicMemory, MemoryConfig
from holographic_prod.core.algebra import build_clifford_basis, frobenius_cosine
from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ, PHI_INV_CUBE


def measure_embedding_structure(memory, groups: List[List[int]]) -> Tuple[float, float]:
    """
    Measure within-group vs between-group similarity.
    
    Returns: (avg_within, avg_between)
    """
    within_sims = []
    between_sims = []
    
    for group in groups:
        # Within-group similarity
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                sim = frobenius_cosine(
                    memory.tower.embeddings[group[i]],
                    memory.tower.embeddings[group[j]],
                    np
                )
                within_sims.append(sim)
    
    # Between-group similarity
    for g1 in range(len(groups)):
        for g2 in range(g1+1, len(groups)):
            for t1 in groups[g1]:
                for t2 in groups[g2]:
                    sim = frobenius_cosine(
                        memory.tower.embeddings[t1],
                        memory.tower.embeddings[t2],
                        np
                    )
                    between_sims.append(sim)
    
    return np.mean(within_sims), np.mean(between_sims)


def test_embedding_evolution_through_training():
    """
    Simulate training and measure embedding structure emergence.
    """
    print("=" * 70)
    print("TEST: Embedding Evolution Through Training")
    print("=" * 70)
    
    # Create memory with contrastive learning ENABLED
    config = MemoryConfig(contrastive_enabled=True)
    memory = HolographicMemory(vocab_size=1000, config=config, max_levels=3, use_gpu=False)
    
    # Define semantic groups (tokens that should become similar)
    # Simulating: Group 1 tokens appear as targets for similar contexts
    groups = [
        [10, 11, 12, 13, 14],  # "Animals" - these will co-occur as targets
        [20, 21, 22, 23, 24],  # "Actions" - these will co-occur as targets
        [30, 31, 32, 33, 34],  # "Places" - these will co-occur as targets
    ]
    
    # Measure initial structure
    w_init, b_init = measure_embedding_structure(memory, groups)
    print(f"\nInitial embedding structure:")
    print(f"  Within-group similarity: {w_init:.4f}")
    print(f"  Between-group similarity: {b_init:.4f}")
    print(f"  Ratio (higher = more structure): {w_init/b_init if b_init != 0 else 'N/A':.2f}")
    
    # Simulate training patterns that create co-occurrence
    # Context [1, 2, 3] -> targets from group 1 (animals)
    # Context [4, 5, 6] -> targets from group 2 (actions)
    # Context [7, 8, 9] -> targets from group 3 (places)
    
    np.random.seed(42)
    
    checkpoints = [100, 500, 1000, 5000]
    for checkpoint in checkpoints:
        # Train to checkpoint
        while memory.learn_count < checkpoint:
            # Pick a context and its corresponding target group
            group_idx = np.random.randint(0, 3)
            ctx = [group_idx * 3 + 1, group_idx * 3 + 2, group_idx * 3 + 3]
            target = groups[group_idx][np.random.randint(0, 5)]
            
            memory.learn(ctx, target)
        
        # Measure structure
        w, b = measure_embedding_structure(memory, groups)
        ratio = w / (abs(b) + 1e-6)
        contrast_updates = memory.contrastive_updates
        
        print(f"\nAfter {memory.learn_count} patterns ({contrast_updates} contrastive updates):")
        print(f"  Within-group similarity: {w:.4f}")
        print(f"  Between-group similarity: {b:.4f}")
        print(f"  Ratio: {ratio:.2f}")
    
    # Final analysis
    w_final, b_final = measure_embedding_structure(memory, groups)
    improvement = (w_final - w_init)
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print("=" * 40)
    print(f"Initial within-group: {w_init:.4f}")
    print(f"Final within-group: {w_final:.4f}")
    print(f"Improvement: {improvement:+.4f}")
    print(f"Total contrastive updates: {memory.contrastive_updates}")
    
    if improvement > 0.01:
        print("\n✓ Embeddings are learning structure!")
    else:
        print("\n⚠️ Structure emergence too slow - consider more training or parameter tuning")


def test_contrastive_rate_impact():
    """
    Test how contrastive rate affects structure emergence.
    """
    print("\n" + "=" * 70)
    print("TEST: Contrastive Rate Impact")
    print("=" * 70)
    
    # Define groups
    groups = [
        [10, 11, 12, 13, 14],
        [20, 21, 22, 23, 24],
    ]
    
    # Test different rates
    rates = [PHI_INV_CUBE * PHI_INV_SQ, PHI_INV_SQ, PHI_INV]  # φ⁻⁵, φ⁻², φ⁻¹
    rate_names = ["φ⁻⁵ (default)", "φ⁻²", "φ⁻¹"]
    
    for rate, name in zip(rates, rate_names):
        config = MemoryConfig(contrastive_enabled=True, contrastive_rate=rate)
        memory = HolographicMemory(vocab_size=1000, config=config, max_levels=3, use_gpu=False)
        
        w_init, _ = measure_embedding_structure(memory, groups)
        
        # Train 2000 patterns
        np.random.seed(42)
        for _ in range(2000):
            group_idx = np.random.randint(0, 2)
            ctx = [group_idx * 3 + 1, group_idx * 3 + 2, group_idx * 3 + 3]
            target = groups[group_idx][np.random.randint(0, 5)]
            memory.learn(ctx, target)
        
        w_final, b_final = measure_embedding_structure(memory, groups)
        improvement = w_final - w_init
        
        print(f"\nRate {name}:")
        print(f"  Contrastive updates: {memory.contrastive_updates}")
        print(f"  Within-group improvement: {improvement:+.4f}")
        print(f"  Final within/between ratio: {w_final/(abs(b_final)+1e-6):.2f}")


if __name__ == "__main__":
    test_embedding_evolution_through_training()
    test_contrastive_rate_impact()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
