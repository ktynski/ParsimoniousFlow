"""
Investigation Tests: Understanding Retrieval Pipeline Issues

Per theory, we need to understand:
1. Why semantic retrieval rate is low (0-12%)
2. Why holographic (tower) retrieval shows 0%
3. Why accuracy is declining over time

These tests are diagnostic - they reveal behavior, not just pass/fail.
"""
import numpy as np
import pytest
from typing import List, Tuple, Dict, Any

from holographic_prod.memory.holographic_memory_unified import HolographicMemory, MemoryConfig
from holographic_prod.dreaming.semantic_memory import SemanticMemory, SemanticPrototype
from holographic_prod.dreaming.dreaming_system import DreamingSystem
from holographic_prod.dreaming.structures import EpisodicEntry
from holographic_prod.core.algebra import build_clifford_basis
from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ

# Helper to create memory with simpler interface
def create_memory(vocab_size=1000, levels=3, use_gpu=False):
    return HolographicMemory(vocab_size=vocab_size, max_levels=levels, use_gpu=use_gpu)

def create_episodic_entry(memory, ctx: List[int], target: int) -> EpisodicEntry:
    """Helper to create episodic entries for testing."""
    ctx_mat = memory.tower._embed_sequence(ctx)
    return EpisodicEntry(context_matrix=ctx_mat, target_token=target)


class TestSemanticRetrievalRate:
    """
    ISSUE: Semantic retrieval is only 0-12%
    
    Per theory, semantic memory should generalize patterns.
    If it's rarely used, either:
    - Prototypes don't cover the query space well
    - Similarity threshold is too high
    - Pattern completion isn't working
    """
    
    def test_prototype_coverage_after_consolidation(self):
        """
        After consolidation, do prototypes cover the input space?
        
        Theory: Non-REM creates prototypes from episodic clusters.
        If prototypes are too specific, they won't match new queries.
        """
        # Create memory and dreaming system
        memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
        basis = build_clifford_basis(np)
        dreaming = DreamingSystem(basis, xp=np)
        
        # Learn diverse patterns
        patterns = [
            [1, 2, 3, 4],  # Pattern A
            [1, 2, 3, 5],  # Similar to A (different target)
            [1, 2, 3, 6],  # Similar to A
            [10, 20, 30, 40],  # Pattern B
            [10, 20, 30, 50],  # Similar to B
            [100, 200, 300, 400],  # Pattern C
        ]
        
        episodes = []
        for ctx in patterns:
            memory.learn(ctx[:-1], ctx[-1])
            entry = create_episodic_entry(memory, ctx[:-1], ctx[-1])
            episodes.append(entry)
        
        # Trigger consolidation with episodes
        dreaming.sleep(episodes, verbose=False)
        
        # Check: How many prototypes were created?
        stats = dreaming.semantic_memory.stats()
        print(f"\n[DIAGNOSTIC] Prototypes created: {stats['total_prototypes']}")
        print(f"[DIAGNOSTIC] By level: {stats.get('prototypes_by_level', [])}")
        
        # Theory check: With 3 distinct patterns, we should have ~3 prototypes
        assert stats['total_prototypes'] >= 1, "No prototypes created - consolidation failed"
        
        # Test retrieval for SIMILAR but not identical contexts
        test_queries = [
            [1, 2, 3],   # Should match Pattern A variants
            [10, 20, 30],  # Should match Pattern B variants
            [100, 200, 300],  # Should match Pattern C
        ]
        
        hits = 0
        for query in test_queries:
            ctx_mat = memory.tower._embed_sequence(query)
            results = dreaming.semantic_memory.retrieve(
                ctx_mat, top_k=1, use_pattern_completion=True, completion_steps=3
            )
            if results:
                proto, sim = results[0]
                print(f"[DIAGNOSTIC] Query {query} → similarity={sim:.3f}, target={proto.mode_target()}")
                if sim > PHI_INV_SQ:  # Theory-true threshold
                    hits += 1
            else:
                print(f"[DIAGNOSTIC] Query {query} → NO MATCH")
        
        hit_rate = hits / len(test_queries)
        print(f"\n[RESULT] Semantic hit rate: {hit_rate:.1%}")
        
        # This reveals if the threshold is the issue
        return hit_rate
    
    def test_similarity_distribution(self):
        """
        What's the distribution of similarities for semantic queries?
        
        If most similarities are just below threshold, the threshold is too high.
        """
        memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
        basis = build_clifford_basis(np)
        dreaming = DreamingSystem(basis, xp=np)
        
        # Learn many patterns
        np.random.seed(42)
        episodes = []
        for i in range(100):
            ctx = list(np.random.randint(1, 500, size=4))
            memory.learn(ctx[:-1], ctx[-1])
            entry = create_episodic_entry(memory, ctx[:-1], ctx[-1])
            episodes.append(entry)
        
        # Consolidate with episodes
        dreaming.sleep(episodes, verbose=False)
        
        stats = dreaming.semantic_memory.stats()
        print(f"\n[DIAGNOSTIC] After 100 patterns: {stats['total_prototypes']} prototypes")
        
        # Query with random contexts and collect similarities
        similarities = []
        for i in range(50):
            query = list(np.random.randint(1, 500, size=3))
            ctx_mat = memory.tower._embed_sequence(query)
            results = dreaming.semantic_memory.retrieve(
                ctx_mat, top_k=5, use_pattern_completion=True, completion_steps=3
            )
            for proto, sim in results:
                similarities.append(sim)
        
        if similarities:
            sims = np.array(similarities)
            print(f"\n[SIMILARITY DISTRIBUTION]")
            print(f"  Min: {sims.min():.3f}")
            print(f"  Max: {sims.max():.3f}")
            print(f"  Mean: {sims.mean():.3f}")
            print(f"  Median: {np.median(sims):.3f}")
            print(f"  Above φ⁻² ({PHI_INV_SQ:.3f}): {(sims > PHI_INV_SQ).sum()}/{len(sims)}")
            print(f"  Above φ⁻¹ ({PHI_INV:.3f}): {(sims > PHI_INV).sum()}/{len(sims)}")
        else:
            print("[WARNING] No similarities collected - semantic memory empty?")
        
        return similarities


class TestHolographicRetrievalRate:
    """
    ISSUE: Holographic retrieval shows 0%
    
    The calculation is: holographic_hits - episodic_hits
    
    This could mean:
    1. All "holographic" hits are actually episodic (cache hits)
    2. The tower isn't returning anything
    3. The tower is returning wrong tokens
    """
    
    def test_tower_retrieval_directly(self):
        """
        Does tower.retrieve() return tokens for learned patterns?
        """
        memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
        
        # Learn patterns
        patterns = [
            ([1, 2, 3], 100),
            ([4, 5, 6], 200),
            ([7, 8, 9], 300),
        ]
        
        for ctx, target in patterns:
            memory.learn(ctx, target)
        
        print(f"\n[DIAGNOSTIC] Learned {memory.n_patterns} patterns")
        active = getattr(memory.tower, '_n_active_satellites', 'N/A')
        print(f"[DIAGNOSTIC] Active satellites: {active}")
        
        # Test retrieval
        hits = 0
        for ctx, expected_target in patterns:
            token_id = memory.tower.retrieve(ctx)
            print(f"[DIAGNOSTIC] Query {ctx} → Retrieved: {token_id}, Expected: {expected_target}")
            if token_id == expected_target:
                hits += 1
        
        accuracy = hits / len(patterns)
        print(f"\n[RESULT] Tower retrieval accuracy: {accuracy:.1%}")
        
        # Theory check: For exact replay, should be 100%
        assert accuracy == 1.0, f"Tower retrieval failed: {accuracy:.1%}"
    
    def test_tower_retrieval_with_variation(self):
        """
        Does tower generalize to similar but not identical contexts?
        
        Theory: SO(4) geometric algebra should enable graceful degradation.
        """
        memory = create_memory(vocab_size=1000, levels=4, use_gpu=False)
        
        # Learn base pattern multiple times (reinforcement)
        base_pattern = [1, 2, 3]
        target = 100
        
        for _ in range(10):
            memory.learn(base_pattern, target)
        
        # Test retrieval with variations
        variations = [
            [1, 2, 3],    # Exact
            [1, 2, 4],    # One token different
            [1, 5, 3],    # Middle different
            [10, 2, 3],   # First different
            [10, 20, 30], # All different
        ]
        
        print(f"\n[DIAGNOSTIC] Testing generalization:")
        for var in variations:
            token_id = memory.tower.retrieve(var)
            pred, conf = memory.retrieve_deterministic(var)
            print(f"  Query {var} → tower={token_id}, det={pred}, conf={conf:.3f}")
    
    def test_episodic_vs_tower_separation(self):
        """
        Verify that episodic and tower paths are correctly separated.
        """
        memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
        
        # Learn a pattern
        ctx = [1, 2, 3]
        target = 100
        memory.learn(ctx, target)
        
        # Check episodic cache
        ctx_tuple = tuple(ctx)
        in_cache = ctx_tuple in memory._episodic_cache
        print(f"\n[DIAGNOSTIC] Context {ctx} in episodic cache: {in_cache}")
        
        # Get counters before retrieval
        hits_before = memory._episodic_hits
        
        # Retrieve
        pred, conf = memory.retrieve_deterministic(ctx)
        
        # Get counters after
        hits_after = memory._episodic_hits
        
        episodic_hit = (hits_after > hits_before)
        print(f"[DIAGNOSTIC] Episodic hit: {episodic_hit}")
        print(f"[DIAGNOSTIC] Retrieved: {pred}, Confidence: {conf}")
        
        # Now test with a context NOT in episodic cache
        new_ctx = [10, 20, 30]
        memory.learn(new_ctx, 200)
        
        # Clear episodic cache to force tower path
        memory._episodic_cache.clear()
        
        hits_before = memory._episodic_hits
        pred2, conf2 = memory.retrieve_deterministic(new_ctx)
        hits_after = memory._episodic_hits
        
        episodic_hit2 = (hits_after > hits_before)
        print(f"\n[DIAGNOSTIC] After clearing cache:")
        print(f"  Query {new_ctx} → pred={pred2}, conf={conf2:.3f}")
        print(f"  Episodic hit: {episodic_hit2}")


class TestAccuracyDecline:
    """
    ISSUE: Running accuracy declining (started ~100%, now ~13%)
    
    Possible causes:
    1. Catastrophic forgetting (new patterns overwrite old)
    2. Interference between similar patterns
    3. Tower saturation (too many patterns per satellite)
    """
    
    def test_catastrophic_forgetting(self):
        """
        Do early patterns get forgotten as new ones are learned?
        """
        memory = create_memory(vocab_size=1000, levels=4, use_gpu=False)
        
        # Learn patterns in phases
        phase1_patterns = [([i, i+1, i+2], i*100) for i in range(1, 11)]
        phase2_patterns = [([i, i+1, i+2], i*100) for i in range(100, 111)]
        
        # Phase 1
        for ctx, target in phase1_patterns:
            memory.learn(ctx, target)
        
        # Test Phase 1 accuracy
        phase1_acc = sum(
            memory.retrieve_deterministic(ctx)[0] == target 
            for ctx, target in phase1_patterns
        ) / len(phase1_patterns)
        print(f"\n[DIAGNOSTIC] Phase 1 accuracy before Phase 2: {phase1_acc:.1%}")
        
        # Phase 2
        for ctx, target in phase2_patterns:
            memory.learn(ctx, target)
        
        # Re-test Phase 1 accuracy
        phase1_acc_after = sum(
            memory.retrieve_deterministic(ctx)[0] == target 
            for ctx, target in phase1_patterns
        ) / len(phase1_patterns)
        print(f"[DIAGNOSTIC] Phase 1 accuracy after Phase 2: {phase1_acc_after:.1%}")
        
        # Test Phase 2 accuracy
        phase2_acc = sum(
            memory.retrieve_deterministic(ctx)[0] == target 
            for ctx, target in phase2_patterns
        ) / len(phase2_patterns)
        print(f"[DIAGNOSTIC] Phase 2 accuracy: {phase2_acc:.1%}")
        
        forgetting = phase1_acc - phase1_acc_after
        print(f"\n[RESULT] Catastrophic forgetting: {forgetting:.1%}")
        
        return forgetting
    
    def test_satellite_occupancy_impact(self):
        """
        Does accuracy degrade as satellites fill up?
        
        Theory: Superposition allows multiple patterns per satellite,
        but there's a capacity limit.
        """
        memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
        
        # Track accuracy as we add patterns
        accuracy_over_time = []
        all_patterns = []
        
        np.random.seed(42)
        for i in range(200):
            ctx = list(np.random.randint(1, 500, size=3))
            target = np.random.randint(1, 500)
            memory.learn(ctx, target)
            all_patterns.append((ctx, target))
            
            # Test accuracy every 20 patterns
            if (i + 1) % 20 == 0:
                correct = sum(
                    memory.retrieve_deterministic(c)[0] == t 
                    for c, t in all_patterns
                )
                acc = correct / len(all_patterns)
                active = getattr(memory.tower, '_n_active_satellites', memory.n_patterns)
                print(f"[DIAGNOSTIC] Patterns: {len(all_patterns)}, "
                      f"Satellites: {active}, Accuracy: {acc:.1%}")
                accuracy_over_time.append((len(all_patterns), active, acc))
        
        # Check for degradation
        if len(accuracy_over_time) >= 2:
            initial_acc = accuracy_over_time[0][2]
            final_acc = accuracy_over_time[-1][2]
            degradation = initial_acc - final_acc
            print(f"\n[RESULT] Accuracy degradation: {degradation:.1%}")
            print(f"  Initial: {initial_acc:.1%}")
            print(f"  Final: {final_acc:.1%}")
        
        return accuracy_over_time


class TestTheoryTrueMetrics:
    """
    Theory-true metrics should capture generalization, not just memorization.
    """
    
    def test_semantic_similarity_vs_exact_match(self):
        """
        Compare semantic similarity to exact match accuracy.
        
        Theory: A model that generalizes will have high semantic similarity
        but lower exact-match accuracy (it predicts synonyms/related words).
        """
        memory = create_memory(vocab_size=1000, levels=3, use_gpu=False)
        
        # Create semantic groups (synonyms)
        # Group 1: "happy" variants -> targets 100, 101, 102
        # Group 2: "sad" variants -> targets 200, 201, 202
        
        groups = [
            {"contexts": [[1, 2, 3], [1, 2, 4], [1, 2, 5]], "targets": [100, 101, 102]},
            {"contexts": [[10, 20, 30], [10, 20, 31], [10, 20, 32]], "targets": [200, 201, 202]},
        ]
        
        # Learn all patterns
        for group in groups:
            for ctx, target in zip(group["contexts"], group["targets"]):
                memory.learn(ctx, target)
        
        # Test
        exact_matches = 0
        semantic_matches = 0
        total = 0
        
        for group in groups:
            group_targets = set(group["targets"])
            for ctx, expected_target in zip(group["contexts"], group["targets"]):
                pred, conf = memory.retrieve_deterministic(ctx)
                total += 1
                
                if pred == expected_target:
                    exact_matches += 1
                    semantic_matches += 1
                elif pred in group_targets:
                    # Predicted a semantically related target
                    semantic_matches += 1
                    print(f"[DIAGNOSTIC] Semantic match: Query {ctx}, "
                          f"Expected {expected_target}, Got {pred} (same group)")
        
        print(f"\n[RESULT]")
        print(f"  Exact match accuracy: {exact_matches/total:.1%}")
        print(f"  Semantic match accuracy: {semantic_matches/total:.1%}")
        
        return exact_matches / total, semantic_matches / total


def run_all_investigations():
    """Run all investigation tests and summarize findings."""
    print("=" * 70)
    print("RETRIEVAL PIPELINE INVESTIGATION")
    print("=" * 70)
    
    # Semantic retrieval tests
    print("\n" + "=" * 70)
    print("1. SEMANTIC RETRIEVAL INVESTIGATION")
    print("=" * 70)
    
    test_sem = TestSemanticRetrievalRate()
    test_sem.test_prototype_coverage_after_consolidation()
    test_sem.test_similarity_distribution()
    
    # Holographic retrieval tests
    print("\n" + "=" * 70)
    print("2. HOLOGRAPHIC (TOWER) RETRIEVAL INVESTIGATION")
    print("=" * 70)
    
    test_holo = TestHolographicRetrievalRate()
    test_holo.test_tower_retrieval_directly()
    test_holo.test_tower_retrieval_with_variation()
    test_holo.test_episodic_vs_tower_separation()
    
    # Accuracy decline tests
    print("\n" + "=" * 70)
    print("3. ACCURACY DECLINE INVESTIGATION")
    print("=" * 70)
    
    test_acc = TestAccuracyDecline()
    test_acc.test_catastrophic_forgetting()
    test_acc.test_satellite_occupancy_impact()
    
    # Theory-true metrics
    print("\n" + "=" * 70)
    print("4. THEORY-TRUE METRICS INVESTIGATION")
    print("=" * 70)
    
    test_metrics = TestTheoryTrueMetrics()
    test_metrics.test_semantic_similarity_vs_exact_match()
    
    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_investigations()
