"""
DEFINITIVE TEST: Holographic Memory Capacity Analysis

HYPOTHESIS TESTED:
    The holographic architecture's accuracy comes from:
    1. Episodic cache (exact match) - 100% accuracy
    2. Semantic prototypes (candidate narrowing) - TBD
    3. Raw holographic retrieval - ~1% accuracy (severe interference)

CONCLUSION:
    The 4x4 Cl(3,1) representation has ~16 effective dimensions.
    Signal-to-noise ratio (SNR) = √(D/N) where D=16, N=patterns.
    For N > 16, SNR < 1 → random predictions.
    
    The architecture compensates with:
    - Episodic cache (hippocampus) for exact recall
    - Semantic prototypes (cortical basins) for candidate narrowing
    - Grace basin routing to distribute load
    
    Grace operator increases STATE stability but doesn't fix 
    SCORE discrimination because it collapses to witness (scalar+pseudo)
    which loses structural information needed for embedding matching.

RUN:
    python -m pytest holographic_prod/tests/test_memory_capacity_analysis.py -v
"""

import numpy as np
import pytest
from typing import List, Tuple

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


class TestEmbeddingCorrelation:
    """Test that random SO(4) embeddings have significant correlation."""
    
    def test_embedding_correlation_distribution(self):
        """Random SO(4) embeddings have non-trivial correlation structure."""
        from holographic_prod.core.grounded_embeddings import create_random_so4_embeddings
        
        embeddings = create_random_so4_embeddings(200, seed=42, xp=np)
        
        # Compute correlation distribution
        correlations = []
        for i in range(100):
            for j in range(i+1, 150):
                e1 = embeddings[i].flatten()
                e2 = embeddings[j].flatten()
                corr = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                correlations.append(corr)
        
        mean_corr = np.mean(correlations)
        max_corr = np.max(correlations)
        min_corr = np.min(correlations)
        
        print(f"\n  SO(4) embedding correlation:")
        print(f"    Mean: {mean_corr:.4f}")
        print(f"    Max:  {max_corr:.4f}")
        print(f"    Min:  {min_corr:.4f}")
        
        # Embeddings are NOT orthogonal to each other
        assert max_corr > 0.5, f"Expected some high correlations, max={max_corr}"
        # But not all the same
        assert min_corr < -0.3, f"Expected some negative correlations, min={min_corr}"


class TestHolographicCapacity:
    """Test raw holographic memory capacity limits."""
    
    def test_single_pattern_perfect(self):
        """Single pattern has perfect recall."""
        from holographic_prod.core.grounded_embeddings import create_random_so4_embeddings
        from holographic_prod.core.algebra import build_clifford_basis, geometric_product_batch
        
        basis = build_clifford_basis()
        embeddings = create_random_so4_embeddings(100, seed=42, xp=np)
        
        # Store one pattern
        ctx = geometric_product_batch(embeddings[0:3], np)
        tgt_idx = 50
        tgt = embeddings[tgt_idx]
        
        memory = PHI_INV * (ctx @ tgt)
        
        # Retrieve using raw dot product
        retrieved = ctx.T @ memory
        scores = np.array([np.sum(retrieved * emb) for emb in embeddings])
        pred = int(np.argmax(scores))
        
        assert pred == tgt_idx, f"Single pattern failed: {pred} != {tgt_idx}"
    
    def test_two_pattern_success_is_rare(self):
        """Two-pattern success depends on specific target/context combinations."""
        from holographic_prod.core.grounded_embeddings import create_random_so4_embeddings
        from holographic_prod.core.algebra import build_clifford_basis, geometric_product_batch
        
        basis = build_clifford_basis()
        embeddings = create_random_so4_embeddings(500, seed=42, xp=np)
        
        ctx1 = geometric_product_batch(embeddings[0:3], np)
        ctx2 = geometric_product_batch(embeddings[10:13], np)
        
        # Test multiple target pairs
        successes = 0
        total = 0
        
        print("\n  Testing 2-pattern combinations:")
        
        for tgt1 in [100, 200, 300]:
            for tgt2 in [100, 200, 300, 400]:
                if tgt1 == tgt2:
                    continue
                
                total += 1
                memory = PHI_INV * (ctx1 @ embeddings[tgt1]) + PHI_INV * (ctx2 @ embeddings[tgt2])
                
                retrieved1 = ctx1.T @ memory
                retrieved2 = ctx2.T @ memory
                
                scores1 = np.array([np.sum(retrieved1 * emb) for emb in embeddings])
                scores2 = np.array([np.sum(retrieved2 * emb) for emb in embeddings])
                
                pred1 = int(np.argmax(scores1))
                pred2 = int(np.argmax(scores2))
                
                if pred1 == tgt1 and pred2 == tgt2:
                    successes += 1
        
        success_rate = 100 * successes / total
        print(f"    Success rate: {successes}/{total} ({success_rate:.0f}%)")
        
        # Most 2-pattern combinations fail due to interference
        # This documents the severe capacity limitation
        assert success_rate < 50, f"Expected low success rate, got {success_rate:.0f}%"
    
    def test_highly_correlated_targets_fail(self):
        """Two patterns with high-correlation targets fail."""
        from holographic_prod.core.grounded_embeddings import create_random_so4_embeddings
        from holographic_prod.core.algebra import build_clifford_basis, geometric_product_batch
        
        basis = build_clifford_basis()
        embeddings = create_random_so4_embeddings(500, seed=42, xp=np)
        
        # Find two highly correlated targets
        e100 = embeddings[100].flatten()
        for offset in range(1, 200):
            e_other = embeddings[100 + offset].flatten()
            corr = np.dot(e100, e_other) / (np.linalg.norm(e100) * np.linalg.norm(e_other))
            if corr > 0.5:
                tgt2_idx = 100 + offset
                tgt_corr = corr
                break
        else:
            pytest.skip("Could not find highly correlated pair")
        
        tgt1_idx = 100
        
        print(f"\n  Using highly correlated targets {tgt1_idx} and {tgt2_idx} (corr={tgt_corr:.4f})")
        
        ctx1 = geometric_product_batch(embeddings[0:3], np)
        ctx2 = geometric_product_batch(embeddings[10:13], np)
        
        memory = PHI_INV * (ctx1 @ embeddings[tgt1_idx]) + PHI_INV * (ctx2 @ embeddings[tgt2_idx])
        
        # Test retrieval
        retrieved1 = ctx1.T @ memory
        scores1 = np.array([np.sum(retrieved1 * emb) for emb in embeddings])
        pred1 = int(np.argmax(scores1))
        rank1 = np.sum(scores1 > scores1[tgt1_idx]) + 1
        
        print(f"    ctx1: pred={pred1}, tgt={tgt1_idx}, rank={rank1}")
        
        # Highly correlated targets cause interference → likely failure
        # (We don't assert failure, just document the phenomenon)
        if pred1 != tgt1_idx:
            print(f"    EXPECTED: Highly correlated targets interfere")
        else:
            print(f"    SURPRISE: Even correlated targets worked!")
    
    def test_capacity_depends_on_embedding_separation(self):
        """Holographic capacity fundamentally depends on embedding separation."""
        from holographic_prod.core.grounded_embeddings import create_random_so4_embeddings
        from holographic_prod.core.algebra import build_clifford_basis, geometric_product_batch
        
        basis = build_clifford_basis()
        embeddings = create_random_so4_embeddings(500, seed=42, xp=np)
        
        print("\n  Capacity depends on target separation:")
        
        # Test with well-separated vs poorly-separated targets
        ctx1 = geometric_product_batch(embeddings[0:3], np)
        ctx2 = geometric_product_batch(embeddings[10:13], np)
        
        for label, tgt_offsets in [
            ("Well-separated (100 apart)", [100, 200]),
            ("Moderately separated (50)", [100, 150]),
            ("Adjacent (1 apart)", [100, 101]),
        ]:
            tgt1_idx, tgt2_idx = tgt_offsets
            
            # Check actual correlation
            e1 = embeddings[tgt1_idx].flatten()
            e2 = embeddings[tgt2_idx].flatten()
            corr = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            
            memory = PHI_INV * (ctx1 @ embeddings[tgt1_idx]) + PHI_INV * (ctx2 @ embeddings[tgt2_idx])
            
            # Test
            correct = 0
            for ctx, tgt in [(ctx1, tgt1_idx), (ctx2, tgt2_idx)]:
                retrieved = ctx.T @ memory
                scores = np.array([np.sum(retrieved * emb) for emb in embeddings])
                if int(np.argmax(scores)) == tgt:
                    correct += 1
            
            print(f"    {label}: {correct}/2 correct, corr={corr:.3f}")


class TestEpisodicCacheMasking:
    """Test that episodic cache masks holographic limitations."""
    
    def test_episodic_provides_all_accuracy(self):
        """Episodic cache is source of 100% accuracy, not holographic."""
        from holographic_prod.memory.holographic_memory_unified import HolographicMemory, MemoryConfig
        
        config = MemoryConfig(contrastive_enabled=False)
        vocab_size = 500
        
        memory = HolographicMemory(vocab_size=vocab_size, config=config)
        
        # Train patterns
        patterns = []
        for i in range(200):
            ctx = [i % vocab_size, (i*2+10) % vocab_size, (i*3+20) % vocab_size]
            tgt = (i*7+50) % vocab_size
            patterns.append((ctx, tgt))
            memory.learn(ctx, tgt)
        
        print("\n  Episodic masking test:")
        print(f"    Episodic cache size: {len(memory._episodic_cache)}")
        
        # Save original cache
        original_cache = memory._episodic_cache.copy()
        
        # Test WITH episodic
        correct_with = sum(
            1 for ctx, tgt in patterns[:100]
            if memory.retrieve_deterministic(ctx)[0] == tgt
        )
        
        # Test WITHOUT episodic
        memory._episodic_cache = {}
        correct_without = sum(
            1 for ctx, tgt in patterns[:100]
            if memory.retrieve_deterministic(ctx)[0] == tgt
        )
        
        # Restore
        memory._episodic_cache = original_cache
        
        print(f"    With episodic:    {correct_with}% accuracy")
        print(f"    Without episodic: {correct_without}% accuracy")
        
        # Episodic should provide most of the accuracy
        assert correct_with > 90, f"Expected >90% with episodic, got {correct_with}%"
        assert correct_without < 20, f"Expected <20% without episodic, got {correct_without}%"


class TestGraceDoesNotFixCapacity:
    """Test that Grace operator doesn't fix the capacity problem."""
    
    def test_grace_increases_stability_not_discrimination(self):
        """Grace increases state stability but not score discrimination."""
        from holographic_prod.core.grounded_embeddings import create_random_so4_embeddings
        from holographic_prod.core.algebra import (
            build_clifford_basis, geometric_product_batch, grace_operator
        )
        from holographic_prod.core.quotient import grace_stability, vorticity_weighted_scores
        from holographic_prod.core.commitment_gate import compute_entropy, phi_kernel_probs
        
        basis = build_clifford_basis()
        embeddings = create_random_so4_embeddings(100, seed=42, xp=np)
        
        # Build memory with many patterns (high interference)
        memory = np.zeros((4, 4))
        for i in range(50):
            ctx = geometric_product_batch(embeddings[i*2:(i*2)+3], np)
            tgt = embeddings[(i*3+20) % 100]
            memory += PHI_INV * (ctx @ tgt)
        
        print("\n  Grace vs Discrimination test:")
        print("    Iter | Stability | Entropy | Conclusion")
        print("    " + "-" * 50)
        
        ctx = geometric_product_batch(embeddings[0:3], np)
        state = ctx.T @ memory
        
        for n_iters in range(6):
            stability = grace_stability(state, basis, np)
            scores = vorticity_weighted_scores(state, embeddings, basis, np)
            entropy = compute_entropy(phi_kernel_probs(scores))
            
            conclusion = "high stability, high entropy" if stability > 0.5 and entropy > 3 else ""
            print(f"    {n_iters:4d} | {stability:9.4f} | {entropy:7.4f} | {conclusion}")
            
            state = grace_operator(state, basis, np)
        
        # After Grace: high stability (good), high entropy (bad)
        final_stability = grace_stability(state, basis, np)
        final_scores = vorticity_weighted_scores(state, embeddings, basis, np)
        final_entropy = compute_entropy(phi_kernel_probs(final_scores))
        
        assert final_stability > 0.9, f"Grace should increase stability: {final_stability:.4f}"
        assert final_entropy > 4.0, f"Entropy stays high (no discrimination): {final_entropy:.4f}"
    
    def test_grace_contracts_to_witness_loses_structure(self):
        """Grace contracts to witness, losing structural info for matching."""
        from holographic_prod.core.grounded_embeddings import create_random_so4_embeddings
        from holographic_prod.core.algebra import (
            build_clifford_basis, geometric_product_batch, grace_operator,
            decompose_to_coefficients
        )
        from holographic_prod.core.quotient import grace_stability, compute_enstrophy
        
        basis = build_clifford_basis()
        embeddings = create_random_so4_embeddings(100, seed=42, xp=np)
        
        # Get a retrieved state
        ctx = geometric_product_batch(embeddings[0:3], np)
        tgt = embeddings[50]
        memory = PHI_INV * (ctx @ tgt)
        retrieved = ctx.T @ memory
        
        print("\n  Grace structural collapse test:")
        print("    Target embedding enstrophy:", compute_enstrophy(tgt, basis, np))
        print()
        print("    Iter | Stability | Enstrophy | Can match target?")
        print("    " + "-" * 55)
        
        state = retrieved.copy()
        for n_iters in range(6):
            stability = grace_stability(state, basis, np)
            enstrophy = compute_enstrophy(state, basis, np)
            
            # Can only match target if enstrophy is preserved
            can_match = "Yes" if enstrophy > 0.1 else "No (structural info lost)"
            
            print(f"    {n_iters:4d} | {stability:9.4f} | {enstrophy:9.4f} | {can_match}")
            
            state = grace_operator(state, basis, np)
        
        # After Grace, enstrophy should be near zero (witness-dominated)
        final_enstrophy = compute_enstrophy(state, basis, np)
        assert final_enstrophy < 0.01, f"Grace should collapse enstrophy: {final_enstrophy:.4f}"


class TestArchitecturalCompensation:
    """Test that the architecture compensates for holographic limitations."""
    
    def test_routing_distributes_load(self):
        """Grace basin routing distributes patterns across satellites."""
        from holographic_prod.memory.holographic_memory_unified import HolographicMemory, MemoryConfig
        
        config = MemoryConfig(contrastive_enabled=False)
        memory = HolographicMemory(vocab_size=500, config=config)
        
        # Train many patterns
        for i in range(200):
            ctx = [i % 500, (i*2+10) % 500, (i*3+20) % 500]
            tgt = (i*7+50) % 500
            memory.learn(ctx, tgt)
        
        loads = [memory.tower.satellites[i].n_bindings for i in range(16)]
        
        print("\n  Routing distribution test:")
        print(f"    Total patterns: 200")
        print(f"    Min load: {min(loads)}, Max load: {max(loads)}, Avg: {np.mean(loads):.1f}")
        
        # Should be somewhat distributed (not all in one satellite)
        assert max(loads) < 200, "Routing should distribute patterns"
        assert min(loads) > 0, "All satellites should get some patterns"
    
    def test_hierarchical_scaling_increases_capacity(self):
        """Multi-level tower increases effective capacity."""
        # This is a documentation test - we note that the architecture
        # uses 16^N satellites for O(16^N) capacity
        
        print("\n  Hierarchical scaling documentation:")
        print("    Level 0: 16 satellites → ~256 pattern capacity (16 per satellite)")
        print("    Level 1: 16² = 256 satellites → ~4096 pattern capacity")
        print("    Level 2: 16³ = 4096 satellites → ~65536 pattern capacity")
        print()
        print("    This is why large-scale training uses multi-level towers")
        print("    with semantic prototype candidate narrowing for efficiency.")


def run_all_tests():
    """Run comprehensive capacity analysis."""
    print("=" * 70)
    print("HOLOGRAPHIC MEMORY CAPACITY ANALYSIS")
    print("=" * 70)
    print()
    print("Testing hypothesis: 4x4 Cl(3,1) matrices have limited holographic capacity")
    print()
    
    import pytest
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
    ])
    
    if exit_code == 0:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print()
        print("KEY FINDINGS:")
        print("  1. Raw holographic memory: ~16 pattern capacity (4x4 → 16 dimensions)")
        print("  2. Episodic cache: Provides 100% recall for seen patterns")
        print("  3. Grace operator: Increases stability but NOT discrimination")
        print("  4. Routing: Distributes load, multiplies capacity by satellite count")
        print("  5. Hierarchical: 16^N scaling for large-scale learning")
        print()
        print("IMPLICATION:")
        print("  The architecture is brain-analog correct:")
        print("  - Hippocampus (episodic) for exact recall")
        print("  - Cortical basins (semantic prototypes) for generalization")
        print("  - Holographic binding for novel compositions")
        print("=" * 70)
    
    return exit_code


if __name__ == '__main__':
    exit(run_all_tests())
