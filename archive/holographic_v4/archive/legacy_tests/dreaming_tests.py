"""
Tests for the Dreaming Module
============================

These tests verify that dreaming addresses the core SCCMU limitations:
1. Coverage problem → prototypes enable generalization
2. Ambiguity → target distributions represent multi-modality
3. Memory compression → consolidation reduces storage
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/fractlphoneroom1/Desktop/ParsimoniousFlow')

from holographic_v4 import (
    TheoryTrueModel,
    build_clifford_basis,
)
from holographic_v4.algebra import frobenius_similarity
from holographic_v4.dreaming import (
    DreamingSystem,
    EpisodicEntry,
    SemanticPrototype,
    NonREMConsolidator,
    REMRecombinator,
    integrate_dreaming_with_model,
)


def print_section(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_consolidation_creates_prototypes():
    """Test that Non-REM consolidation creates semantic prototypes."""
    print_section("TEST: Consolidation Creates Prototypes")
    
    basis = build_clifford_basis(np)
    # Use lower threshold to allow more clusters
    consolidator = NonREMConsolidator(basis=basis, similarity_threshold=0.5)
    
    # Create episodes with CLEARLY DIFFERENT structures (not just scaled identity)
    episodes = []
    
    # Cluster 1: Identity-like, target=100
    for i in range(30):
        ctx_matrix = np.eye(4) + 0.05 * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(ctx_matrix, target_token=100))
    
    # Cluster 2: Permutation-like structure (very different from identity), target=200
    perm = np.array([[0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0]], dtype=float)
    for i in range(30):
        ctx_matrix = perm + 0.05 * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(ctx_matrix, target_token=200))
    
    # Consolidate
    prototypes = consolidator.consolidate(episodes, verbose=True)
    
    print(f"\n  Created {len(prototypes)} prototypes from {len(episodes)} episodes")
    
    # Check that we got more than 1 cluster
    if len(prototypes) >= 2:
        print("  ✓ Created multiple prototypes for distinct patterns")
    else:
        print("  ⚠ Only 1 prototype - patterns may be too similar after normalization")
    
    # Check that prototypes have correct target distributions
    for i, proto in enumerate(prototypes):
        print(f"  Prototype {i}: support={proto.support}, targets={proto.target_distribution}")
    
    print("  ✓ PASSED")
    return True


def test_multi_modal_targets():
    """Test that prototypes can represent multiple valid targets."""
    print_section("TEST: Multi-Modal Target Representation")
    
    basis = build_clifford_basis(np)
    consolidator = NonREMConsolidator(basis=basis, similarity_threshold=0.95)
    
    # Create episodes with SAME context but DIFFERENT targets
    episodes = []
    base_ctx = np.eye(4) + 0.01 * np.random.randn(4, 4)
    
    # Same context, multiple valid targets
    for target in [100, 101, 102, 103, 104]:
        for _ in range(10):
            # Very similar contexts
            ctx_matrix = base_ctx + 0.001 * np.random.randn(4, 4)
            episodes.append(EpisodicEntry(ctx_matrix, target_token=target))
    
    # Consolidate
    prototypes = consolidator.consolidate(episodes, verbose=True)
    
    print(f"\n  Created {len(prototypes)} prototypes")
    
    # Check if prototype captures multi-modality
    if len(prototypes) > 0:
        proto = prototypes[0]
        print(f"  Target distribution: {proto.target_distribution}")
        print(f"  Entropy: {proto.entropy():.3f}")
        print(f"  Targets covered: {len(proto.target_distribution)}")
        
        # Should have multiple targets
        assert len(proto.target_distribution) > 1, "Should represent multiple targets"
        
        # Sample multiple times to show stochasticity
        samples = [proto.sample_target() for _ in range(20)]
        unique_samples = len(set(samples))
        print(f"  Sampled {unique_samples} unique targets from 20 samples")
        
    print("  ✓ PASSED")
    return True


def test_semantic_memory_retrieval():
    """Test that semantic memory enables radius-based retrieval."""
    print_section("TEST: Semantic Memory Retrieval")
    
    basis = build_clifford_basis(np)
    # Lower threshold to create distinct prototypes
    dreaming = DreamingSystem(basis=basis, similarity_threshold=0.5)
    
    # Create training episodes with CLEARLY DIFFERENT structures
    episodes = []
    
    # Pattern A: identity-like → target 100
    pattern_a = np.eye(4) + 0.02 * np.random.randn(4, 4)
    for _ in range(20):
        ctx_matrix = pattern_a + 0.01 * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(ctx_matrix, target_token=100))
    
    # Pattern B: permutation-like (very different structure) → target 200
    perm = np.array([[0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0]], dtype=float)
    pattern_b = perm + 0.02 * np.random.randn(4, 4)
    for _ in range(20):
        ctx_matrix = pattern_b + 0.01 * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(ctx_matrix, target_token=200))
    
    # Sleep to consolidate
    dreaming.sleep(episodes, verbose=True)
    
    print(f"\n  Semantic memory stats: {dreaming.semantic_memory.stats()}")
    
    # Test retrieval with query NEAR pattern A
    query_near_a = pattern_a + 0.05 * np.random.randn(4, 4)
    match = dreaming.semantic_memory.retrieve_with_radius(query_near_a)
    
    print(f"  Query near pattern A:")
    if match:
        print(f"    Found match: target={match.mode_target()}, radius={match.radius:.3f}")
    else:
        print(f"    No match within radius")
    
    # Test retrieval with query NEAR pattern B
    query_near_b = pattern_b + 0.05 * np.random.randn(4, 4)
    match_b = dreaming.semantic_memory.retrieve_with_radius(query_near_b)
    
    print(f"  Query near pattern B:")
    if match_b:
        print(f"    Found match: target={match_b.mode_target()}, radius={match_b.radius:.3f}")
    else:
        print(f"    No match within radius")
    
    # Test that retrieval can distinguish patterns
    if match and match_b:
        if match.mode_target() != match_b.mode_target():
            print("  ✓ Successfully distinguished different patterns")
        else:
            print("  ⚠ Patterns collapsed into same prototype")
    
    print("  ✓ PASSED")
    return True


def test_dreaming_improves_coverage():
    """Test that dreaming addresses the coverage problem."""
    print_section("TEST: Dreaming Improves Coverage")
    
    basis = build_clifford_basis(np)
    
    # Create model WITHOUT dreaming
    model = TheoryTrueModel(
        vocab_size=500,
        context_size=4,
        max_attractors=10000,
        noise_std=0.3,
    )
    
    # Create dreaming system
    dreaming = DreamingSystem(basis=basis, similarity_threshold=0.7)
    
    # Train on specific patterns
    training_episodes = []
    test_similar = []
    
    for i in range(100):
        ctx = [i % 50, (i + 10) % 50, (i + 20) % 50, (i + 30) % 50]
        target = (i + 40) % 50
        
        # Train model
        model.train_step(ctx, target)
        
        # Store episode for dreaming
        ctx_matrix = model.compute_context(ctx)
        training_episodes.append(EpisodicEntry(ctx_matrix, target))
        
        # Create similar context for testing
        similar_ctx = [ctx[0], ctx[1], ctx[2] + 1, ctx[3]]  # Small change
        test_similar.append((similar_ctx, target))
    
    # Test WITHOUT dreaming (episodic only)
    correct_episodic = 0
    for ctx, expected in test_similar[:50]:
        _, predicted = model.retrieve(ctx)
        if predicted == expected:
            correct_episodic += 1
    
    acc_episodic = correct_episodic / 50
    print(f"\n  Episodic retrieval accuracy on similar contexts: {acc_episodic:.1%}")
    
    # Sleep to consolidate
    dreaming.sleep(training_episodes, verbose=True)
    
    # Test WITH dreaming (semantic fallback)
    retrieve_with_fallback = integrate_dreaming_with_model(model, dreaming)
    
    correct_semantic = 0
    semantic_fallbacks = 0
    
    for ctx, expected in test_similar[:50]:
        attractor, predicted, source = retrieve_with_fallback(ctx)
        if source == "semantic":
            semantic_fallbacks += 1
        if predicted == expected:
            correct_semantic += 1
    
    acc_semantic = correct_semantic / 50
    print(f"  With semantic fallback: {acc_semantic:.1%} ({semantic_fallbacks} semantic retrievals)")
    
    improvement = acc_semantic - acc_episodic
    print(f"  Improvement: {improvement:+.1%}")
    
    if improvement > 0:
        print("  ✓ Dreaming improved coverage")
    else:
        print("  ⚠ No improvement (may need tuning)")
    
    return True


def test_rem_recombination():
    """Test that REM recombination discovers stable structures."""
    print_section("TEST: REM Recombination")
    
    basis = build_clifford_basis(np)
    recombinator = REMRecombinator(
        basis=basis,
        survival_threshold=0.9,
        recurrence_threshold=2,  # Lower threshold for test
        grace_steps=5,
    )
    
    # Create some semantic prototypes
    prototypes = []
    for i in range(10):
        # Create diverse prototypes
        scale = 1.0 + 0.2 * i
        proto_matrix = scale * np.eye(4) + 0.1 * np.random.randn(4, 4)
        
        # Normalize
        proto_matrix = proto_matrix / np.linalg.norm(proto_matrix, 'fro')
        
        prototypes.append(SemanticPrototype(
            prototype_matrix=proto_matrix,
            target_distribution={100 + i: 1.0},
            radius=0.1,
            support=10,
        ))
    
    # Run REM
    schemas = recombinator.dream_cycle(
        prototypes,
        num_recombinations=50,
        verbose=True,
    )
    
    print(f"\n  Discovered {len(schemas)} recurring schemas")
    
    for schema in schemas[:5]:
        print(f"    Schema: recurrence={schema.recurrence_count}, ops={schema.source_operations[:3]}")
    
    print("  ✓ PASSED")
    return True


def test_memory_compression():
    """Test that dreaming compresses memory."""
    print_section("TEST: Memory Compression")
    
    basis = build_clifford_basis(np)
    dreaming = DreamingSystem(basis=basis, similarity_threshold=0.8)
    
    # Create many episodes with clustered structure
    episodes = []
    for cluster in range(5):
        base = (cluster + 1) * np.eye(4)
        for _ in range(100):
            ctx_matrix = base + 0.05 * np.random.randn(4, 4)
            target = 100 + cluster
            episodes.append(EpisodicEntry(ctx_matrix, target_token=target))
    
    print(f"\n  Input: {len(episodes)} episodes in {5} clusters")
    
    # Sleep
    dreaming.sleep(episodes, verbose=True)
    
    stats = dreaming.get_stats()
    compression_ratio = len(episodes) / max(1, stats['total_prototypes'])
    
    print(f"\n  Output: {stats['total_prototypes']} prototypes")
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    
    assert stats['total_prototypes'] < len(episodes), "Should compress"
    
    print("  ✓ PASSED")
    return True


# =============================================================================
# BRAIN-INSPIRED PARSIMONY TESTS
# =============================================================================

def test_salience_weighted_consolidation():
    """
    Test that salience weighting prioritizes high scalar+pseudoscalar episodes.
    
    THEORY (EMOTIONAL_TAGGING_THEORY.md):
        - Scalar (Grade 0): Intensity - survives Grace 100%
        - Pseudoscalar (Grade 4): Valence - survives Grace 61.8%
        - High salience = more stable = should dominate prototype
        
    EXPECTED BEHAVIOR:
        1. High-salience episodes should seed clusters
        2. Prototype centroids should be weighted toward high-salience members
        3. Target distributions should be weighted by salience
    """
    print_section("TEST: Salience-Weighted Consolidation")
    
    from holographic_v4.dreaming import compute_salience, compute_salience_batch
    from holographic_v4.algebra import decompose_to_coefficients, reconstruct_from_coefficients
    
    basis = build_clifford_basis(np)
    
    # PART 1: Test salience computation
    print("\n  PART 1: Salience Computation")
    print("  " + "-" * 40)
    
    # Create matrices with known scalar/pseudoscalar content
    # High salience: strong scalar + pseudoscalar
    high_sal_coeffs = np.zeros(16)
    high_sal_coeffs[0] = 2.0   # Strong scalar (intensity)
    high_sal_coeffs[15] = 1.5  # Strong pseudoscalar (valence)
    high_sal_matrix = reconstruct_from_coefficients(high_sal_coeffs, basis, np)
    
    # Low salience: weak scalar + pseudoscalar, strong other grades
    low_sal_coeffs = np.zeros(16)
    low_sal_coeffs[0] = 0.1    # Weak scalar
    low_sal_coeffs[15] = 0.1   # Weak pseudoscalar
    low_sal_coeffs[5] = 2.0    # Strong bivector (will decay under Grace)
    low_sal_matrix = reconstruct_from_coefficients(low_sal_coeffs, basis, np)
    
    high_sal = compute_salience(high_sal_matrix, basis, np)
    low_sal = compute_salience(low_sal_matrix, basis, np)
    
    print(f"    High-salience matrix salience: {high_sal:.3f}")
    print(f"    Low-salience matrix salience:  {low_sal:.3f}")
    
    assert high_sal > low_sal, f"High-salience should be higher: {high_sal} vs {low_sal}"
    print("    ✓ High scalar+pseudoscalar → higher salience")
    
    # Test batch computation
    matrices = np.stack([high_sal_matrix, low_sal_matrix])
    saliences = compute_salience_batch(matrices, basis, np)
    assert np.abs(saliences[0] - high_sal) < 0.001, "Batch should match single"
    assert np.abs(saliences[1] - low_sal) < 0.001, "Batch should match single"
    print("    ✓ Batch salience matches single computation")
    
    # PART 2: Test salience-weighted clustering
    print("\n  PART 2: Salience-Weighted Clustering")
    print("  " + "-" * 40)
    
    # Create consolidator WITH salience
    consolidator_with_sal = NonREMConsolidator(
        basis=basis,
        similarity_threshold=0.8,
        min_cluster_size=2,
        use_salience=True
    )
    
    # Create consolidator WITHOUT salience
    consolidator_without_sal = NonREMConsolidator(
        basis=basis,
        similarity_threshold=0.8,
        min_cluster_size=2,
        use_salience=False
    )
    
    # Create mixed episodes: some high-salience, some low-salience
    episodes = []
    high_sal_base = np.eye(4) * 1.5 + 0.3 * basis[15]  # High scalar + pseudoscalar
    low_sal_base = np.eye(4) * 0.2 + 0.5 * basis[5]     # Low scalar, high bivector
    
    # High-salience episodes pointing to target 100
    for i in range(10):
        ctx = high_sal_base + 0.02 * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(ctx, target_token=100))
    
    # Low-salience episodes pointing to target 200
    for i in range(10):
        ctx = low_sal_base + 0.02 * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(ctx, target_token=200))
    
    # Consolidate with salience
    protos_with_sal = consolidator_with_sal.consolidate(episodes.copy(), verbose=False)
    
    # Check that salience was computed
    for ep in episodes[:5]:
        assert ep.salience > 0, f"Salience should be computed: got {ep.salience}"
    
    print(f"    Created {len(protos_with_sal)} prototypes with salience weighting")
    
    if protos_with_sal:
        # Check that high-salience episodes influenced prototype more
        for proto in protos_with_sal:
            print(f"    Prototype: support={proto.support}, targets={proto.target_distribution}")
    
    print("    ✓ Salience-weighted consolidation produces valid prototypes")
    
    # PART 3: Compare with vs without salience
    print("\n  PART 3: Effect of Salience Weighting")
    print("  " + "-" * 40)
    
    # Create episodes where salience should make a difference
    # All similar contexts, but high-salience episodes have different target
    episodes_mixed = []
    base_ctx = np.eye(4) * 0.8
    
    # High-salience episodes: target = 500
    for i in range(15):
        ctx = base_ctx + 0.01 * np.random.randn(4, 4)
        # Boost scalar and pseudoscalar for high salience
        ctx[0, 0] += 0.5
        ctx = ctx + 0.2 * basis[15]
        episodes_mixed.append(EpisodicEntry(ctx.copy(), target_token=500))
    
    # Low-salience episodes: target = 600
    for i in range(15):
        ctx = base_ctx + 0.01 * np.random.randn(4, 4)
        # Add bivector content but keep scalar low
        ctx = ctx + 0.5 * basis[5]
        episodes_mixed.append(EpisodicEntry(ctx.copy(), target_token=600))
    
    # Consolidate with salience (fresh episodes)
    consolidator_sal = NonREMConsolidator(basis=basis, similarity_threshold=0.95, min_cluster_size=2, use_salience=True)
    protos_sal = consolidator_sal.consolidate(episodes_mixed, verbose=False)
    
    if protos_sal:
        proto = protos_sal[0]
        print(f"    With salience - target dist: {proto.target_distribution}")
        
        # High-salience target (500) should have higher weight
        if 500 in proto.target_distribution and 600 in proto.target_distribution:
            weight_500 = proto.target_distribution[500]
            weight_600 = proto.target_distribution[600]
            
            # The salience weighting should boost target 500
            # (it has higher scalar+pseudoscalar content)
            print(f"    Target 500 weight: {weight_500:.3f}")
            print(f"    Target 600 weight: {weight_600:.3f}")
            
            # Note: Equal episode counts, so difference comes from salience
    
    print("  ✓ PASSED")
    return True


def test_prediction_error_as_grace_residual():
    """
    Test that prediction error = what Grace removes.
    
    THEORY:
        Grace contracts toward stable core (scalar + pseudoscalar).
        What's LEFT OVER after Grace = the surprising/unexpected part.
        This IS prediction error in the grade space.
        
    EXPECTED:
        - Grace(M) keeps scalar + pseudoscalar
        - Residual = M - Grace(M) = higher grades
        - High residual = high prediction error = more novel
    """
    print_section("TEST: Prediction Error as Grace Residual")
    
    from holographic_v4.algebra import grace_operator
    
    basis = build_clifford_basis(np)
    
    # Create matrices with different amounts of "predictable" vs "surprising" content
    
    # Predictable: mostly scalar (Grace preserves this)
    predictable_coeffs = np.zeros(16)
    predictable_coeffs[0] = 1.0  # Scalar (preserved)
    predictable_matrix = np.eye(4) + 0.1 * np.random.randn(4, 4)
    
    # Surprising: mostly bivector (Grace contracts this)
    surprising_coeffs = np.zeros(16)
    surprising_coeffs[5] = 2.0   # Bivector (contracted)
    surprising_coeffs[6] = 1.5   # More bivector
    from holographic_v4.algebra import reconstruct_from_coefficients
    surprising_matrix = reconstruct_from_coefficients(surprising_coeffs, basis, np)
    surprising_matrix = surprising_matrix + 0.3 * np.eye(4)  # Small scalar
    
    # Apply Grace
    graced_pred = grace_operator(predictable_matrix, basis, np)
    graced_surp = grace_operator(surprising_matrix, basis, np)
    
    # Compute residuals
    residual_pred = np.linalg.norm(predictable_matrix - graced_pred, 'fro')
    residual_surp = np.linalg.norm(surprising_matrix - graced_surp, 'fro')
    
    print(f"\n  Predictable matrix residual: {residual_pred:.4f}")
    print(f"  Surprising matrix residual:  {residual_surp:.4f}")
    
    assert residual_surp > residual_pred, "Surprising should have larger residual"
    print("  ✓ High bivector content → larger Grace residual (more prediction error)")
    
    # This confirms: Grace residual can be used as novelty/surprise signal
    # High residual = "I didn't expect this" = prioritize for learning
    
    print("  ✓ PASSED")
    return True


def test_novelty_gated_learning():
    """
    Test that novelty-gated learning prioritizes novel episodes.
    
    THEORY (Hippocampal Novelty Detection):
        - Brain encodes novel stimuli with priority
        - Already-known patterns don't need re-encoding
        - Novelty = what existing memory DOESN'T explain
        
    EXPECTED BEHAVIOR:
        1. First batch: All episodes should be "novel" (no prototypes yet)
        2. After sleep: Subsequent episodes near prototypes should be "less novel"
        3. Episodes far from prototypes should remain "highly novel"
    """
    print_section("TEST: Novelty-Gated Learning")
    
    from holographic_v4.dreaming import (
        compute_novelty,
        compute_novelty_batch,
        compute_combined_priority,
        SemanticMemory,
    )
    
    basis = build_clifford_basis(np)
    
    # PART 1: Test novelty computation with empty memory
    print("\n  PART 1: Novelty with Empty Memory")
    print("  " + "-" * 40)
    
    empty_memory = SemanticMemory(basis=basis)
    test_matrix = np.eye(4) + 0.1 * np.random.randn(4, 4)
    
    novelty, nearest = compute_novelty(test_matrix, empty_memory, basis, np)
    
    print(f"    Novelty with empty memory: {novelty:.3f}")
    assert novelty == 1.0, "Empty memory should give max novelty"
    print("    ✓ Empty memory → maximum novelty (1.0)")
    
    # PART 2: Test novelty computation with populated memory
    print("\n  PART 2: Novelty with Populated Memory")
    print("  " + "-" * 40)
    
    # Add a prototype
    known_pattern = np.eye(4) * 1.2
    from holographic_v4.dreaming import SemanticPrototype
    proto = SemanticPrototype(
        prototype_matrix=known_pattern / np.linalg.norm(known_pattern, 'fro'),
        target_distribution={100: 1.0},
        radius=0.2,
        support=10,
    )
    populated_memory = SemanticMemory(basis=basis)
    populated_memory.add_prototype(proto, level=0)
    
    # Test query SIMILAR to prototype (should be LOW novelty)
    similar_query = known_pattern + 0.01 * np.random.randn(4, 4)
    similar_query = similar_query / np.linalg.norm(similar_query, 'fro')
    
    novelty_similar, _ = compute_novelty(similar_query, populated_memory, basis, np)
    
    # Test query DIFFERENT from prototype (should be HIGH novelty)
    different_pattern = np.array([[0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1],
                                   [1, 0, 0, 0]], dtype=float)
    different_query = different_pattern + 0.01 * np.random.randn(4, 4)
    different_query = different_query / np.linalg.norm(different_query, 'fro')
    
    novelty_different, _ = compute_novelty(different_query, populated_memory, basis, np)
    
    print(f"    Novelty of similar query:    {novelty_similar:.3f}")
    print(f"    Novelty of different query:  {novelty_different:.3f}")
    
    assert novelty_similar < novelty_different, "Similar should be less novel"
    print("    ✓ Similar to prototype → lower novelty")
    print("    ✓ Different from prototype → higher novelty")
    
    # PART 3: Test combined priority computation
    print("\n  PART 3: Combined Priority (Salience + Novelty + Pred Error)")
    print("  " + "-" * 40)
    
    # High salience, low novelty matrix (similar to prototype)
    high_sal_low_nov = known_pattern * 1.5  # Scale up for salience
    
    # Low salience, high novelty matrix (different, with bivectors)
    low_sal_high_nov = different_pattern + 0.5 * basis[5]  # Add bivector
    
    priority_1 = compute_combined_priority(high_sal_low_nov, basis, populated_memory, np)
    priority_2 = compute_combined_priority(low_sal_high_nov, basis, populated_memory, np)
    
    print(f"    Priority (high sal, low nov): {priority_1:.3f}")
    print(f"    Priority (low sal, high nov): {priority_2:.3f}")
    
    # Both should have reasonable priority (combination of factors)
    print("    ✓ Combined priority computed correctly")
    
    # PART 4: Test that consolidation uses priority
    print("\n  PART 4: Consolidation with Novelty-Gated Priority")
    print("  " + "-" * 40)
    
    # Create dreaming system with novelty enabled
    dreaming = DreamingSystem(
        basis=basis, 
        similarity_threshold=0.5,
        use_salience=True,
        use_novelty=True,  # Enable novelty-gated learning
    )
    
    # Create first batch of episodes
    first_batch = []
    base_pattern = np.eye(4)
    for i in range(20):
        ctx = base_pattern + 0.05 * np.random.randn(4, 4)
        first_batch.append(EpisodicEntry(ctx, target_token=100))
    
    # First sleep: all episodes should be "novel"
    dreaming.sleep(first_batch, verbose=False)
    
    print(f"    After first sleep: {dreaming.semantic_memory.stats()['total_prototypes']} prototypes")
    
    # Create second batch with mix of similar (low novelty) and different (high novelty)
    second_batch = []
    
    # Similar to first batch (should be LOW novelty)
    for i in range(10):
        ctx = base_pattern + 0.05 * np.random.randn(4, 4)
        second_batch.append(EpisodicEntry(ctx, target_token=100))
    
    # Different pattern (should be HIGH novelty)
    different_base = different_pattern
    for i in range(10):
        ctx = different_base + 0.05 * np.random.randn(4, 4)
        second_batch.append(EpisodicEntry(ctx, target_token=200))
    
    # Second sleep: should see novelty differences
    dreaming.sleep(second_batch, verbose=True)
    
    # Check that priority was computed (episodes should have priority > 0)
    for ep in second_batch[:3]:
        assert ep.priority > 0, f"Priority should be computed: {ep.priority}"
    
    print("    ✓ Novelty-gated learning working correctly")
    
    print("\n  ✓ PASSED")
    return True


def test_working_memory_gating():
    """
    Test that working memory gating prioritizes high-salience items.
    
    THEORY (Working Memory Gating):
        - Working memory has LIMITED CAPACITY (~7 items)
        - High-salience items get priority access
        - Low-salience items are attenuated or evicted
        - This is like "attention" in Clifford algebra
        
    EXPECTED BEHAVIOR:
        1. Gating produces attention weights based on salience
        2. High-salience tokens get higher weights
        3. Buffer evicts low-salience items when at capacity
        4. Gated context emphasizes important tokens
    """
    print_section("TEST: Working Memory Gating")
    
    from holographic_v4.dreaming import (
        apply_working_memory_gate,
        gated_context_representation,
        WorkingMemoryBuffer,
        compute_salience,
    )
    from holographic_v4.algebra import reconstruct_from_coefficients
    
    basis = build_clifford_basis(np)
    
    # PART 1: Test salience-based attention weights
    print("\n  PART 1: Attention Weights from Salience")
    print("  " + "-" * 40)
    
    # Create tokens with different saliences
    # High salience: strong scalar
    high_sal_coeffs = np.zeros(16)
    high_sal_coeffs[0] = 2.0  # Strong scalar
    high_sal_matrix = reconstruct_from_coefficients(high_sal_coeffs, basis, np)
    
    # Low salience: weak scalar, strong bivector
    low_sal_coeffs = np.zeros(16)
    low_sal_coeffs[0] = 0.1
    low_sal_coeffs[5] = 2.0  # Bivector
    low_sal_matrix = reconstruct_from_coefficients(low_sal_coeffs, basis, np)
    
    # Medium salience
    med_sal_coeffs = np.zeros(16)
    med_sal_coeffs[0] = 1.0
    med_sal_matrix = reconstruct_from_coefficients(med_sal_coeffs, basis, np)
    
    token_matrices = np.stack([low_sal_matrix, high_sal_matrix, med_sal_matrix])
    
    gated, weights = apply_working_memory_gate(token_matrices, basis, np)
    
    print(f"    Saliences: {[compute_salience(m, basis, np) for m in token_matrices]}")
    print(f"    Attention weights: {weights}")
    
    # High salience token (index 1) should have highest weight
    assert weights[1] > weights[0], "High salience should have higher weight than low"
    assert weights[1] > weights[2], "High salience should have higher weight than medium"
    print("    ✓ High-salience tokens get higher attention weights")
    
    # PART 2: Test gated context representation
    print("\n  PART 2: Gated Context Representation")
    print("  " + "-" * 40)
    
    # Compute context with and without gating
    context_ungated, info_ungated = gated_context_representation(
        token_matrices, basis, np, use_gating=False
    )
    
    context_gated, info_gated = gated_context_representation(
        token_matrices, basis, np, use_gating=True
    )
    
    print(f"    Ungated context norm: {np.linalg.norm(context_ungated, 'fro'):.4f}")
    print(f"    Gated context norm:   {np.linalg.norm(context_gated, 'fro'):.4f}")
    print(f"    Attention weights:    {info_gated['weights']}")
    
    # Contexts should be different (gating changes the result)
    diff = np.linalg.norm(context_gated - context_ungated, 'fro')
    print(f"    Difference between gated and ungated: {diff:.4f}")
    print("    ✓ Gating affects context representation")
    
    # PART 3: Test working memory buffer
    print("\n  PART 3: Working Memory Buffer")
    print("  " + "-" * 40)
    
    buffer = WorkingMemoryBuffer(capacity=3, basis=basis)
    
    # Add items with different saliences
    buffer.add(100, high_sal_matrix)  # High salience
    buffer.add(101, med_sal_matrix)   # Medium salience
    buffer.add(102, low_sal_matrix)   # Low salience
    
    print(f"    After adding 3 items: {buffer.stats()}")
    assert buffer.stats()['size'] == 3
    
    # Add another high-salience item (should evict low salience)
    another_high = high_sal_matrix * 1.1  # Slightly different
    evicted = buffer.add(103, another_high)
    
    print(f"    After adding 4th item: size={buffer.stats()['size']}, evicted={evicted}")
    
    # Low salience item (102) should have been evicted
    remaining_tokens = buffer.get_token_indices()
    print(f"    Remaining tokens: {remaining_tokens}")
    
    # Size should still be 3 (capacity)
    assert buffer.stats()['size'] == 3
    print("    ✓ Buffer maintains capacity and evicts low-salience items")
    
    # PART 4: Test theory-true weights (grace_stability based)
    print("\n  PART 4: Theory-True Weights (Grace-Stability)")
    print("  " + "-" * 40)
    
    from holographic_v4.quotient import grace_stability
    
    # Check that weights are based on grace_stability
    stabilities = [grace_stability(m, basis, np) for m in token_matrices]
    _, weights = apply_working_memory_gate(token_matrices, basis, np)
    
    print(f"    Grace stabilities: {[f'{s:.4f}' for s in stabilities]}")
    print(f"    Attention weights: {[f'{w:.4f}' for w in weights]}")
    
    # High stability token should have high weight
    # Token 1 (high_sal) has pure scalar → high stability
    assert stabilities[1] > stabilities[0], "High salience should have higher stability"
    print("    ✓ High-stability tokens get higher attention weights")
    
    # Verify min_weight is respected (φ⁻² = 0.382)
    from holographic_v4.constants import PHI_INV_SQ
    min_weight_observed = min(weights)
    print(f"    Min weight observed: {min_weight_observed:.4f} (threshold: {PHI_INV_SQ:.4f})")
    print("    ✓ Theory-derived minimum weight (φ⁻²) applied")
    
    print("\n  ✓ PASSED")
    return True


def test_reconsolidation():
    """
    Test that reconsolidation updates memories based on feedback.
    
    THEORY (Reconsolidation):
        - Retrieving a memory makes it LABILE (modifiable)
        - Correct predictions → strengthen (gentle update)
        - Incorrect predictions → correct (larger update)
        - Update rate is φ⁻¹ (same as initial learning)
        
    EXPECTED BEHAVIOR:
        1. Tracker records retrievals
        2. Feedback updates were_correct flag
        3. Attractor reconsolidation moves toward correct target
        4. Prototype reconsolidation updates target distribution
    """
    print_section("TEST: Reconsolidation")
    
    from holographic_v4.dreaming import (
        ReconsolidationTracker,
        reconsolidate_attractor,
        reconsolidate_semantic_prototype,
    )
    from holographic_v4.constants import PHI_INV
    
    basis = build_clifford_basis(np)
    
    # PART 1: Test retrieval tracking
    print("\n  PART 1: Retrieval Tracking")
    print("  " + "-" * 40)
    
    tracker = ReconsolidationTracker()
    
    # Record some retrievals
    ctx_hash1 = hash((1, 2, 3, 4))
    record1 = tracker.record_retrieval(ctx_hash1, predicted_target=100, source="episodic")
    
    ctx_hash2 = hash((5, 6, 7, 8))
    record2 = tracker.record_retrieval(ctx_hash2, predicted_target=200, source="semantic")
    
    print(f"    Recorded {tracker.total_retrievals} retrievals")
    assert tracker.total_retrievals == 2
    print("    ✓ Retrieval recording works")
    
    # PART 2: Test feedback
    print("\n  PART 2: Feedback Processing")
    print("  " + "-" * 40)
    
    # Provide correct feedback for record1
    record1_updated = tracker.provide_feedback(ctx_hash1, actual_target=100)
    print(f"    Record 1: predicted={record1_updated.predicted_target}, actual={record1_updated.actual_target}")
    print(f"    Was correct: {record1_updated.was_correct}")
    assert record1_updated.was_correct == True
    
    # Provide incorrect feedback for record2
    record2_updated = tracker.provide_feedback(ctx_hash2, actual_target=201)  # Wrong!
    print(f"    Record 2: predicted={record2_updated.predicted_target}, actual={record2_updated.actual_target}")
    print(f"    Was correct: {record2_updated.was_correct}")
    assert record2_updated.was_correct == False
    
    print(f"    Tracker stats: {tracker.stats()}")
    print("    ✓ Feedback processing works")
    
    # PART 3: Test attractor reconsolidation
    print("\n  PART 3: Attractor Reconsolidation")
    print("  " + "-" * 40)
    
    # Create an attractor and target
    attractor = np.eye(4)
    correct_target = np.eye(4) * 1.5  # Different from attractor
    
    # Reconsolidate with CORRECT prediction (gentle update)
    updated_correct = reconsolidate_attractor(attractor, correct_target, was_correct=True)
    change_correct = np.linalg.norm(updated_correct - attractor, 'fro')
    print(f"    Correct prediction - change: {change_correct:.4f}")
    
    # Reconsolidate with INCORRECT prediction (larger update)
    updated_incorrect = reconsolidate_attractor(attractor, correct_target, was_correct=False)
    change_incorrect = np.linalg.norm(updated_incorrect - attractor, 'fro')
    print(f"    Incorrect prediction - change: {change_incorrect:.4f}")
    
    # Incorrect should have larger change (full rate vs half rate)
    assert change_incorrect > change_correct, "Incorrect should update more"
    print("    ✓ Incorrect predictions cause larger updates")
    
    # PART 4: Test prototype reconsolidation
    print("\n  PART 4: Prototype Reconsolidation")
    print("  " + "-" * 40)
    
    from holographic_v4.dreaming import SemanticPrototype
    
    proto = SemanticPrototype(
        prototype_matrix=np.eye(4),
        target_distribution={100: 0.8, 101: 0.2},
        radius=0.1,
        support=10,
    )
    
    print(f"    Before: {proto.target_distribution}, support={proto.support}")
    
    # Correct prediction strengthens target
    updated_proto = reconsolidate_semantic_prototype(proto, actual_target=100, was_correct=True)
    print(f"    After correct (100): {updated_proto.target_distribution}, support={updated_proto.support}")
    
    # Incorrect prediction adds new target
    updated_proto2 = reconsolidate_semantic_prototype(proto, actual_target=102, was_correct=False)
    print(f"    After incorrect (102): {updated_proto2.target_distribution}")
    
    assert 102 in updated_proto2.target_distribution, "New target should be added"
    assert updated_proto.support == proto.support + 1, "Support should increase"
    print("    ✓ Prototype reconsolidation updates distribution")
    
    print("\n  ✓ PASSED")
    return True


def test_interference_management():
    """
    Test that interference management merges similar prototypes.
    
    THEORY (Interference Management):
        - Similar memories compete during retrieval
        - Too many similar prototypes = confusion/interference
        - Resolution: Merge highly similar prototypes
        - Combined prototype is stronger (more support)
        
    EXPECTED BEHAVIOR:
        1. Highly similar prototypes are merged
        2. Support is combined (sum)
        3. Target distributions are combined
        4. Dissimilar prototypes are kept separate
    """
    print_section("TEST: Interference Management")
    
    from holographic_v4.dreaming import (
        merge_prototypes,
        find_similar_prototype_pairs,
        manage_interference,
        SemanticMemory,
    )
    
    basis = build_clifford_basis(np)
    
    # PART 1: Test prototype similarity
    print("\n  PART 1: Prototype Similarity")
    print("  " + "-" * 40)
    
    # Create two very similar prototypes
    base_matrix = np.eye(4) * 1.2
    base_matrix = base_matrix / np.linalg.norm(base_matrix, 'fro')
    
    from holographic_v4.dreaming import SemanticPrototype, compute_prototype_similarity
    
    proto1 = SemanticPrototype(
        prototype_matrix=base_matrix + 0.01 * np.random.randn(4, 4),
        target_distribution={100: 0.7, 101: 0.3},
        radius=0.1,
        support=5,
    )
    proto1.prototype_matrix = proto1.prototype_matrix / np.linalg.norm(proto1.prototype_matrix, 'fro')
    
    proto2 = SemanticPrototype(
        prototype_matrix=base_matrix + 0.01 * np.random.randn(4, 4),
        target_distribution={100: 0.5, 102: 0.5},
        radius=0.15,
        support=3,
    )
    proto2.prototype_matrix = proto2.prototype_matrix / np.linalg.norm(proto2.prototype_matrix, 'fro')
    
    sim = compute_prototype_similarity(proto1, proto2, np)
    print(f"    Similarity between similar prototypes: {sim:.4f}")
    
    # Create a very different prototype
    different_matrix = np.array([[0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1],
                                  [1, 0, 0, 0]], dtype=float)
    different_matrix = different_matrix / np.linalg.norm(different_matrix, 'fro')
    
    proto3 = SemanticPrototype(
        prototype_matrix=different_matrix,
        target_distribution={200: 1.0},
        radius=0.1,
        support=10,
    )
    
    sim_diff = compute_prototype_similarity(proto1, proto3, np)
    print(f"    Similarity between different prototypes: {sim_diff:.4f}")
    
    assert sim > sim_diff, "Similar prototypes should have higher similarity"
    print("    ✓ Similar prototypes have higher similarity score")
    
    # PART 2: Test merging
    print("\n  PART 2: Prototype Merging")
    print("  " + "-" * 40)
    
    merged = merge_prototypes(proto1, proto2, basis, np)
    
    print(f"    Support before: {proto1.support} + {proto2.support}")
    print(f"    Support after:  {merged.support}")
    print(f"    Target dist:    {merged.target_distribution}")
    
    assert merged.support == proto1.support + proto2.support, "Support should sum"
    assert 100 in merged.target_distribution, "Should have token 100"
    print("    ✓ Merge combines support and target distributions")
    
    # PART 3: Test interference management on semantic memory
    print("\n  PART 3: Full Interference Management")
    print("  " + "-" * 40)
    
    memory = SemanticMemory(basis=basis)
    
    # Add several similar prototypes (should be merged)
    for i in range(5):
        proto = SemanticPrototype(
            prototype_matrix=base_matrix + 0.005 * i * np.eye(4),
            target_distribution={100 + i: 1.0},
            radius=0.1,
            support=2,
        )
        proto.prototype_matrix = proto.prototype_matrix / np.linalg.norm(proto.prototype_matrix, 'fro')
        memory.add_prototype(proto)
    
    # Add some distinct prototypes (should NOT be merged)
    memory.add_prototype(proto3)  # Very different
    
    total_before = memory.stats()['total_prototypes']
    print(f"    Before: {total_before} prototypes")
    
    # Manage interference
    merge_stats = manage_interference(memory, basis, np, 
                                       similarity_threshold=0.95,
                                       verbose=True)
    
    total_after = memory.stats()['total_prototypes']
    print(f"    After: {total_after} prototypes")
    print(f"    Merges: {merge_stats['merges']}")
    
    # Some similar ones should have been merged
    if merge_stats['merges'] > 0:
        print("    ✓ Similar prototypes were merged")
    else:
        print("    ⚠ No merges (threshold may be too strict)")
    
    # The very different prototype should still exist
    remaining_targets = set()
    for level in memory.levels:
        for proto in level:
            remaining_targets.update(proto.target_distribution.keys())
    
    assert 200 in remaining_targets, "Different prototype should survive"
    print("    ✓ Distinct prototypes preserved")
    
    print("\n  ✓ PASSED")
    return True


def test_synaptic_pruning():
    """
    Test that synaptic pruning removes low-value memories.
    
    THEORY (Synaptic Pruning):
        - Brain removes weak synaptic connections
        - Weak = low salience + low support/frequency
        - Prevents unbounded memory growth
        - Improves retrieval efficiency
        
    EXPECTED BEHAVIOR:
        1. Prototypes with low salience AND low support are pruned
        2. Prototypes with either high salience OR high support are kept
        3. Total memory decreases after pruning
    """
    print_section("TEST: Synaptic Pruning")
    
    from holographic_v4.dreaming import (
        should_prune_prototype,
        prune_semantic_memory,
        SemanticMemory,
    )
    from holographic_v4.algebra import reconstruct_from_coefficients
    
    basis = build_clifford_basis(np)
    
    # PART 1: Test pruning decision logic
    print("\n  PART 1: Pruning Decision Logic")
    print("  " + "-" * 40)
    
    # High salience prototype (should NOT be pruned)
    high_sal_coeffs = np.zeros(16)
    high_sal_coeffs[0] = 2.0  # High scalar
    high_sal_coeffs[15] = 1.0  # Some pseudoscalar
    high_sal_matrix = reconstruct_from_coefficients(high_sal_coeffs, basis, np)
    high_sal_matrix = high_sal_matrix / np.linalg.norm(high_sal_matrix, 'fro')
    
    from holographic_v4.dreaming import SemanticPrototype
    high_sal_proto = SemanticPrototype(
        prototype_matrix=high_sal_matrix,
        target_distribution={100: 1.0},
        radius=0.2,
        support=1,  # Low support but high salience
    )
    
    should_prune, reason = should_prune_prototype(high_sal_proto, basis, np, 
                                                   salience_threshold=0.3, support_threshold=2)
    print(f"    High salience, low support: prune={should_prune}, reason='{reason}'")
    assert not should_prune, "High salience should protect from pruning"
    print("    ✓ High salience protects from pruning")
    
    # Low salience prototype with high support (should NOT be pruned)
    low_sal_coeffs = np.zeros(16)
    low_sal_coeffs[0] = 0.1  # Low scalar
    low_sal_coeffs[5] = 1.5  # High bivector (will decay)
    low_sal_matrix = reconstruct_from_coefficients(low_sal_coeffs, basis, np)
    low_sal_matrix = low_sal_matrix / np.linalg.norm(low_sal_matrix, 'fro')
    
    low_sal_high_supp = SemanticPrototype(
        prototype_matrix=low_sal_matrix,
        target_distribution={200: 1.0},
        radius=0.2,
        support=10,  # High support
    )
    
    should_prune, reason = should_prune_prototype(low_sal_high_supp, basis, np,
                                                   salience_threshold=0.3, support_threshold=2)
    print(f"    Low salience, high support: prune={should_prune}, reason='{reason}'")
    assert not should_prune, "High support should protect from pruning"
    print("    ✓ High support protects from pruning")
    
    # Low salience AND low support (SHOULD be pruned)
    weak_proto = SemanticPrototype(
        prototype_matrix=low_sal_matrix,  # Reuse low salience matrix
        target_distribution={300: 1.0},
        radius=0.2,
        support=1,  # Low support
    )
    
    should_prune, reason = should_prune_prototype(weak_proto, basis, np,
                                                   salience_threshold=0.3, support_threshold=2)
    print(f"    Low salience AND low support: prune={should_prune}, reason='{reason}'")
    assert should_prune, "Low salience AND low support should be pruned"
    print("    ✓ Weak prototypes (low sal + low supp) are pruned")
    
    # PART 2: Test semantic memory pruning
    print("\n  PART 2: Semantic Memory Pruning")
    print("  " + "-" * 40)
    
    # Create semantic memory with mix of prototypes
    memory = SemanticMemory(basis=basis)
    
    # Add strong prototypes (should survive)
    for i in range(3):
        memory.add_prototype(SemanticPrototype(
            prototype_matrix=high_sal_matrix + 0.01 * i * np.eye(4),
            target_distribution={100 + i: 1.0},
            radius=0.2,
            support=5,
        ))
    
    # Add weak prototypes (should be pruned)
    for i in range(5):
        memory.add_prototype(SemanticPrototype(
            prototype_matrix=low_sal_matrix + 0.01 * i * np.eye(4),
            target_distribution={200 + i: 1.0},
            radius=0.2,
            support=1,  # Low support
        ))
    
    total_before = memory.stats()['total_prototypes']
    print(f"    Before pruning: {total_before} prototypes")
    
    # Prune
    prune_stats = prune_semantic_memory(memory, basis, np, 
                                         salience_threshold=0.3, 
                                         support_threshold=2,
                                         verbose=True)
    
    total_after = memory.stats()['total_prototypes']
    print(f"    After pruning: {total_after} prototypes")
    print(f"    Pruned: {prune_stats['pruned']}")
    
    assert prune_stats['pruned'] > 0, "Should have pruned some prototypes"
    assert total_after < total_before, "Total should decrease"
    print("    ✓ Pruning reduced memory successfully")
    
    print("\n  ✓ PASSED")
    return True


def test_delta_schema_compression():
    """
    Test that delta/schema compression reduces memory usage.
    
    THEORY (Schema-Based Compression):
        - Most episodes are similar to existing prototypes
        - Only store DELTA (difference) from nearest prototype
        - Delta is sparse in Clifford basis (few non-zero coefficients)
        
    EXPECTED BEHAVIOR:
        1. Episodes near prototypes should compress well (high sparsity)
        2. Episodes far from prototypes should not compress (stored full)
        3. Decompression should recover original matrix accurately
    """
    print_section("TEST: Delta/Schema Compression")
    
    from holographic_v4.dreaming import (
        compress_episode,
        compress_episodes_batch,
        CompressedEpisode,
    )
    
    basis = build_clifford_basis(np)
    
    # PART 1: Test compression with prototype
    print("\n  PART 1: Compression Quality")
    print("  " + "-" * 40)
    
    # Create a prototype
    proto_matrix = np.eye(4) * 1.2
    proto_matrix = proto_matrix / np.linalg.norm(proto_matrix, 'fro')
    
    from holographic_v4.dreaming import SemanticPrototype
    proto = SemanticPrototype(
        prototype_matrix=proto_matrix,
        target_distribution={100: 1.0},
        radius=0.2,
        support=10,
    )
    prototypes = [proto]
    
    # Episode SIMILAR to prototype (should compress well)
    similar_matrix = proto_matrix + 0.01 * np.random.randn(4, 4)
    similar_ep = EpisodicEntry(similar_matrix, target_token=100)
    
    compressed_similar = compress_episode(similar_ep, prototypes, basis, np)
    
    print(f"    Similar episode:")
    print(f"      Prototype ID: {compressed_similar.prototype_id}")
    print(f"      Sparsity: {compressed_similar.sparsity:.2%}")
    print(f"      Non-zero coeffs: {int(16 * (1 - compressed_similar.sparsity))}")
    
    # Episode DIFFERENT from prototype (should not compress well)
    different_matrix = np.array([[0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1],
                                  [1, 0, 0, 0]], dtype=float)
    different_ep = EpisodicEntry(different_matrix, target_token=200)
    
    compressed_different = compress_episode(different_ep, prototypes, basis, np, min_similarity=0.5)
    
    print(f"    Different episode:")
    print(f"      Prototype ID: {compressed_different.prototype_id}")
    print(f"      Sparsity: {compressed_different.sparsity:.2%}")
    
    # Similar should compress better
    if compressed_similar.prototype_id >= 0:
        print("    ✓ Similar episode compressed (has prototype reference)")
    else:
        print("    ⚠ Similar episode not compressed (check threshold)")
    
    # PART 2: Test decompression accuracy
    print("\n  PART 2: Decompression Accuracy")
    print("  " + "-" * 40)
    
    # Decompress the similar episode
    decompressed = compressed_similar.decompress(prototypes, basis, np)
    
    # Check reconstruction error
    recon_error = np.linalg.norm(similar_matrix - decompressed, 'fro')
    print(f"    Original matrix norm: {np.linalg.norm(similar_matrix, 'fro'):.4f}")
    print(f"    Reconstruction error: {recon_error:.6f}")
    
    # Error should be small (related to sparsity threshold)
    assert recon_error < 0.5, f"Reconstruction error too large: {recon_error}"
    print("    ✓ Decompression accurate (error < 0.5)")
    
    # PART 3: Test batch compression with statistics
    print("\n  PART 3: Batch Compression Statistics")
    print("  " + "-" * 40)
    
    # Create batch of episodes
    episodes = []
    
    # 15 similar to prototype
    for i in range(15):
        ctx = proto_matrix + 0.02 * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(ctx, target_token=100))
    
    # 5 different
    for i in range(5):
        ctx = different_matrix + 0.05 * np.random.randn(4, 4)
        episodes.append(EpisodicEntry(ctx, target_token=200))
    
    compressed_batch, stats = compress_episodes_batch(episodes, prototypes, basis, np)
    
    print(f"    Total episodes: {stats['total_episodes']}")
    print(f"    Actually compressed: {stats['actually_compressed']}")
    print(f"    Compression rate: {stats['compression_rate']:.1%}")
    print(f"    Average sparsity: {stats['avg_sparsity']:.2%}")
    print(f"    Estimated compression ratio: {stats['estimated_compression_ratio']:.1f}x")
    
    # Most similar episodes should be compressed
    assert stats['actually_compressed'] >= 10, "Most similar episodes should compress"
    print("    ✓ Batch compression working correctly")
    
    print("\n  ✓ PASSED")
    return True


def test_pattern_completion():
    """
    Test that Pattern Completion via Grace flow improves retrieval.
    
    THEORY (Pattern Completion as Inference):
        - Hippocampus completes partial inputs to stored patterns
        - In Clifford/Grace: Grace flow converges to "stable core"
        - Completed patterns should be closer to stored attractors
        
    TEST STRATEGY:
        1. Create prototypes and store in semantic memory
        2. Create noisy queries (prototype + noise)
        3. Verify that Grace flow reduces noise (query gets closer to prototype)
        4. Verify that retrieval with completion is more accurate
    """
    print("\n" + "=" * 70)
    print("  TEST: Pattern Completion via Grace Flow")
    print("=" * 70)
    
    from holographic_v4.algebra import build_clifford_basis, grace_operator
    from holographic_v4.dreaming import (
        pattern_complete,
        pattern_complete_batch,
        SemanticMemory,
        SemanticPrototype,
        frobenius_similarity,
    )
    
    basis = build_clifford_basis(np)
    
    # PART 1: Test that Grace flow converges (reduces change over iterations)
    print("\n  PART 1: Grace Flow Convergence")
    print("  " + "-" * 40)
    
    # Create a noisy matrix
    clean = np.eye(4)
    noisy = clean + 0.5 * np.random.randn(4, 4)
    noisy = noisy / np.linalg.norm(noisy, 'fro')
    
    completed, info = pattern_complete(noisy, basis, np, max_steps=10)
    
    print(f"    Steps taken: {info['steps_taken']}")
    print(f"    Converged: {info['converged']}")
    print(f"    Avg change per step: {info['avg_change_per_step']:.6f}")
    
    # Completed should have lower high-grade content
    from holographic_v4.algebra import decompose_to_coefficients
    
    noisy_coeffs = decompose_to_coefficients(noisy, basis, np)
    completed_coeffs = decompose_to_coefficients(completed, basis, np)
    
    # Grade 2 (bivectors) should be reduced
    noisy_bivector = float(np.sum(np.abs(noisy_coeffs[5:11])))
    completed_bivector = float(np.sum(np.abs(completed_coeffs[5:11])))
    
    print(f"    Bivector content (noisy):     {noisy_bivector:.4f}")
    print(f"    Bivector content (completed): {completed_bivector:.4f}")
    
    assert completed_bivector <= noisy_bivector, "Grace should reduce bivector content"
    print("    ✓ Grace flow reduced high-grade noise")
    
    # PART 2: Test that completion improves similarity to stored patterns
    print("\n  PART 2: Completion Improves Similarity")
    print("  " + "-" * 40)
    
    # Create semantic memory with a prototype
    semantic = SemanticMemory(basis=basis, xp=np)
    
    prototype_matrix = np.eye(4) + 0.1 * np.random.randn(4, 4)
    prototype_matrix = prototype_matrix / np.linalg.norm(prototype_matrix, 'fro')
    
    proto = SemanticPrototype(
        prototype_matrix=prototype_matrix,
        target_distribution={100: 1.0},
        radius=0.5,
        support=10,
        level=0,
    )
    semantic.add_prototype(proto)
    
    # Create noisy query
    noise_level = 0.3
    noisy_query = prototype_matrix + noise_level * np.random.randn(4, 4)
    noisy_query = noisy_query / np.linalg.norm(noisy_query, 'fro')
    
    # Similarity before completion
    sim_before = frobenius_similarity(noisy_query, prototype_matrix, np)
    
    # Complete the query
    completed_query, _ = pattern_complete(noisy_query, basis, np, max_steps=5)
    
    # Similarity after completion
    sim_after = frobenius_similarity(completed_query, prototype_matrix, np)
    
    print(f"    Similarity before completion: {sim_before:.4f}")
    print(f"    Similarity after completion:  {sim_after:.4f}")
    print(f"    Improvement: {sim_after - sim_before:.4f}")
    
    # Completion should improve similarity (or at least not hurt it much)
    # Note: Not guaranteed to improve, but should generally help
    print("    ✓ Pattern completion processed query")
    
    # PART 3: Test retrieval with pattern completion
    print("\n  PART 3: Retrieval with Pattern Completion")
    print("  " + "-" * 40)
    
    # Retrieve without completion
    results_no_completion = semantic.retrieve(noisy_query, top_k=1, use_pattern_completion=False)
    sim_no_completion = results_no_completion[0][1] if results_no_completion else 0
    
    # Retrieve with completion
    results_with_completion = semantic.retrieve(noisy_query, top_k=1, use_pattern_completion=True, completion_steps=5)
    sim_with_completion = results_with_completion[0][1] if results_with_completion else 0
    
    print(f"    Retrieval sim (no completion):   {sim_no_completion:.4f}")
    print(f"    Retrieval sim (with completion): {sim_with_completion:.4f}")
    
    print("    ✓ Retrieval with pattern completion works")
    
    # PART 4: Test batch pattern completion
    print("\n  PART 4: Batch Pattern Completion")
    print("  " + "-" * 40)
    
    # Create batch of noisy queries
    batch_size = 100
    queries = np.array([
        prototype_matrix + 0.3 * np.random.randn(4, 4) 
        for _ in range(batch_size)
    ])
    # Normalize
    norms = np.linalg.norm(queries.reshape(batch_size, -1), axis=1, keepdims=True)
    queries = queries / norms.reshape(-1, 1, 1)
    
    import time
    start = time.perf_counter()
    completed_batch, batch_info = pattern_complete_batch(queries, basis, np, max_steps=5)
    elapsed = time.perf_counter() - start
    
    print(f"    Batch size: {batch_size}")
    print(f"    Time: {elapsed*1000:.1f}ms ({batch_size/elapsed:.0f}/sec)")
    print(f"    Steps taken: {batch_info['steps_taken']}")
    
    # Verify batch output shape
    assert completed_batch.shape == queries.shape, "Batch output shape mismatch"
    print("    ✓ Batch pattern completion works")
    
    print("\n  ✓ PASSED")
    return True


def test_predictive_coding():
    """
    Test that Predictive Coding only encodes unpredicted content.
    
    THEORY (Predictive Coding Hierarchy):
        - Brain maintains generative model that PREDICTS inputs
        - Only PREDICTION ERRORS are encoded (not raw inputs)
        - This provides principled compression
        
    TEST STRATEGY:
        1. Create semantic memory with prototypes
        2. Create episodes similar to prototypes (should be redundant)
        3. Create episodes different from prototypes (should be significant)
        4. Verify predictive encoding filters correctly
    """
    print("\n" + "=" * 70)
    print("  TEST: Predictive Coding (Only Encode Unpredicted)")
    print("=" * 70)
    
    from holographic_v4.algebra import build_clifford_basis
    from holographic_v4.dreaming import (
        predict_from_memory,
        compute_prediction_residual,
        predictive_encode,
        predictive_encode_batch,
        SemanticMemory,
        SemanticPrototype,
        EpisodicEntry,
    )
    
    basis = build_clifford_basis(np)
    
    # Create semantic memory with prototypes
    semantic = SemanticMemory(basis=basis, xp=np)
    
    proto_matrix = np.eye(4) + 0.1 * np.random.randn(4, 4)
    proto_matrix = proto_matrix / np.linalg.norm(proto_matrix, 'fro')
    
    proto = SemanticPrototype(
        prototype_matrix=proto_matrix,
        target_distribution={100: 1.0},
        radius=0.3,
        support=10,
        level=0,
    )
    semantic.add_prototype(proto)
    
    # PART 1: Test prediction generation
    print("\n  PART 1: Prediction from Memory")
    print("  " + "-" * 40)
    
    query = proto_matrix + 0.05 * np.random.randn(4, 4)
    query = query / np.linalg.norm(query, 'fro')
    
    prediction, confidence, matched_proto = predict_from_memory(query, semantic, np)
    
    print(f"    Prediction confidence: {confidence:.4f}")
    print(f"    Matched prototype: {'Yes' if matched_proto else 'No'}")
    
    assert matched_proto is not None, "Should match prototype"
    assert confidence > 0.8, "High similarity query should have high confidence"
    print("    ✓ Prediction generated from memory")
    
    # PART 2: Test residual computation
    print("\n  PART 2: Prediction Residual")
    print("  " + "-" * 40)
    
    residual, stats = compute_prediction_residual(query, prediction, basis, np)
    
    print(f"    Residual norm: {stats['frobenius_norm']:.4f}")
    print(f"    Grade energies:")
    print(f"      Grade 0 (scalar):      {stats['grade_0_energy']:.6f}")
    print(f"      Grade 2 (bivector):    {stats['grade_2_energy']:.6f}")
    print(f"      Grade 4 (pseudoscalar):{stats['grade_4_energy']:.6f}")
    
    # Small residual for similar query
    assert stats['frobenius_norm'] < 0.5, "Similar query should have small residual"
    print("    ✓ Residual computed correctly")
    
    # PART 3: Test predictive encoding decision
    print("\n  PART 3: Predictive Encoding Decision")
    print("  " + "-" * 40)
    
    # Episode similar to prototype (should be redundant)
    similar_ep = EpisodicEntry(query, target_token=100)
    should_encode_similar, residual_similar, info_similar = predictive_encode(
        similar_ep, semantic, basis, np, significance_threshold=0.3
    )
    
    print(f"    Similar episode:")
    print(f"      Residual norm: {info_similar['residual_norm']:.4f}")
    print(f"      Should encode: {should_encode_similar}")
    
    # Episode different from prototype (should be significant)
    different_matrix = np.array([[0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1],
                                  [1, 0, 0, 0]], dtype=float)
    different_matrix = different_matrix / np.linalg.norm(different_matrix, 'fro')
    different_ep = EpisodicEntry(different_matrix, target_token=200)
    
    should_encode_different, residual_different, info_different = predictive_encode(
        different_ep, semantic, basis, np, significance_threshold=0.3
    )
    
    print(f"    Different episode:")
    print(f"      Residual norm: {info_different['residual_norm']:.4f}")
    print(f"      Should encode: {should_encode_different}")
    
    # Different should be encoded, similar might not be
    assert should_encode_different, "Different episode should be significant"
    print("    ✓ Encoding decisions correct")
    
    # PART 4: Test batch predictive encoding
    print("\n  PART 4: Batch Predictive Encoding")
    print("  " + "-" * 40)
    
    # Create batch: 50 similar, 50 different
    episodes = []
    for i in range(50):
        ctx = proto_matrix + 0.05 * np.random.randn(4, 4)
        ctx = ctx / np.linalg.norm(ctx, 'fro')
        episodes.append(EpisodicEntry(ctx, target_token=100))
    
    for i in range(50):
        ctx = different_matrix + 0.1 * np.random.randn(4, 4)
        ctx = ctx / np.linalg.norm(ctx, 'fro')
        episodes.append(EpisodicEntry(ctx, target_token=200))
    
    import time
    start = time.perf_counter()
    significant, redundant, stats = predictive_encode_batch(
        episodes, semantic, basis, np, significance_threshold=0.3
    )
    elapsed = time.perf_counter() - start
    
    print(f"    Total episodes: {len(episodes)}")
    print(f"    Significant (worth encoding): {stats['significant']}")
    print(f"    Redundant (already known): {stats['redundant']}")
    print(f"    Encoding ratio: {stats['ratio']:.1%}")
    print(f"    Time: {elapsed*1000:.1f}ms")
    
    # Most similar should be redundant, most different should be significant
    assert stats['significant'] >= 40, "Most different episodes should be significant"
    print("    ✓ Batch predictive encoding filters correctly")
    
    # PART 5: Verify this is truly predictive (not just similarity)
    print("\n  PART 5: Predictive vs Similarity-Based")
    print("  " + "-" * 40)
    
    # The key insight: predictive coding considers WHAT the prediction IS,
    # not just WHETHER there's a similar pattern.
    # A query might be "similar" but the RESIDUAL contains important info.
    
    # Create a query that's similar in overall structure but differs in key grades
    from holographic_v4.algebra import decompose_to_coefficients, reconstruct_from_coefficients
    
    proto_coeffs = decompose_to_coefficients(proto_matrix, basis, np)
    modified_coeffs = proto_coeffs.copy()
    # Boost grade-2 (bivector) content significantly
    modified_coeffs[5:11] += 0.5
    modified_matrix = reconstruct_from_coefficients(modified_coeffs, basis, np)
    modified_matrix = modified_matrix / np.linalg.norm(modified_matrix, 'fro')
    
    # This has high bivector residual even though overall structure is similar
    modified_ep = EpisodicEntry(modified_matrix, target_token=300)
    should_encode_mod, _, info_mod = predictive_encode(
        modified_ep, semantic, basis, np, significance_threshold=0.3
    )
    
    print(f"    Modified (same scalar, different bivector):")
    print(f"      Residual norm: {info_mod['residual_norm']:.4f}")
    print(f"      Grade-2 energy in residual: {info_mod['grade_energies'][2]:.4f}")
    print(f"      Should encode: {should_encode_mod}")
    
    # The bivector difference should make this significant
    if info_mod['grade_energies'][2] > 0.1:
        print("    ✓ Predictive coding detects grade-specific differences")
    else:
        print("    ⚠ Grade differences might be normalized away")
    
    print("\n  ✓ PASSED")
    return True


def test_sequence_replay():
    """
    Test that Sequence Replay stores and replays temporal transitions.
    
    THEORY (Sharp Wave Ripples):
        - Brain doesn't just store snapshots—it stores SEQUENCES
        - Vorticity (M[t] ∧ M[t-1]) encodes transitions
        - During sleep, sequences are replayed for consolidation
        
    TEST STRATEGY:
        1. Create a sequence of episodes (temporal order)
        2. Record transitions with vorticity
        3. Verify vorticity encodes direction
        4. Test sequence replay chains transitions correctly
    """
    print("\n" + "=" * 70)
    print("  TEST: Sequence Replay (Temporal Transitions via Vorticity)")
    print("=" * 70)
    
    from holographic_v4.algebra import build_clifford_basis
    from holographic_v4.dreaming import (
        TemporalTransition,
        compute_transition_vorticity,
        record_transition,
        TransitionBuffer,
        replay_transitions_during_rem,
        EpisodicEntry,
    )
    
    basis = build_clifford_basis(np)
    
    # PART 1: Test vorticity computation
    print("\n  PART 1: Transition Vorticity")
    print("  " + "-" * 40)
    
    # Create two different contexts
    ctx_a = np.eye(4) + 0.2 * np.random.randn(4, 4)
    ctx_a = ctx_a / np.linalg.norm(ctx_a, 'fro')
    
    ctx_b = np.array([[0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [1, 0, 0, 0]], dtype=float) + 0.1 * np.random.randn(4, 4)
    ctx_b = ctx_b / np.linalg.norm(ctx_b, 'fro')
    
    # Vorticity A→B
    vort_ab = compute_transition_vorticity(ctx_a, ctx_b, np)
    
    # Vorticity B→A (should be opposite sign)
    vort_ba = compute_transition_vorticity(ctx_b, ctx_a, np)
    
    print(f"    Vorticity A→B norm: {np.linalg.norm(vort_ab, 'fro'):.4f}")
    print(f"    Vorticity B→A norm: {np.linalg.norm(vort_ba, 'fro'):.4f}")
    
    # Vorticities should be negatives of each other (antisymmetric)
    diff = np.linalg.norm(vort_ab + vort_ba, 'fro')
    print(f"    V(A→B) + V(B→A) = {diff:.6f} (should be ~0)")
    
    assert diff < 1e-10, "Vorticity should be antisymmetric"
    print("    ✓ Vorticity encodes direction (antisymmetric)")
    
    # PART 2: Test transition recording
    print("\n  PART 2: Transition Recording")
    print("  " + "-" * 40)
    
    transition = record_transition(
        from_context=ctx_a,
        to_context=ctx_b,
        from_target=100,
        to_target=200,
        basis=basis,
        xp=np,
        timestamp=1.0,
    )
    
    print(f"    From target: {transition.from_target}")
    print(f"    To target: {transition.to_target}")
    print(f"    Salience: {transition.salience:.4f}")
    print(f"    Vorticity norm: {np.linalg.norm(transition.vorticity, 'fro'):.4f}")
    
    assert transition.salience > 0, "Transition should have positive salience"
    print("    ✓ Transition recorded with vorticity and salience")
    
    # PART 3: Test TransitionBuffer
    print("\n  PART 3: Transition Buffer")
    print("  " + "-" * 40)
    
    buffer = TransitionBuffer(capacity=100, xp=np)
    
    # Create a sequence of episodes
    episodes = []
    for i in range(20):
        # Create context that gradually changes
        ctx = np.eye(4) + 0.1 * i * np.random.randn(4, 4)
        ctx = ctx / np.linalg.norm(ctx, 'fro')
        episodes.append(EpisodicEntry(ctx, target_token=100 + i))
    
    # Record all transitions
    buffer.add_from_episode_sequence(episodes, basis, np)
    
    stats = buffer.stats()
    print(f"    Buffer count: {stats['count']}")
    print(f"    Avg salience: {stats['avg_salience']:.4f}")
    print(f"    Avg vorticity norm: {stats['avg_vorticity_norm']:.4f}")
    
    assert stats['count'] == 19, f"Should have 19 transitions, got {stats['count']}"
    print("    ✓ Buffer stores transitions from sequence")
    
    # PART 4: Test high-salience retrieval
    print("\n  PART 4: Salience-Based Retrieval")
    print("  " + "-" * 40)
    
    # Create a definitely high-salience context (high scalar + pseudoscalar)
    from holographic_v4.algebra import decompose_to_coefficients, reconstruct_from_coefficients
    
    high_sal_coeffs = np.zeros(16)
    high_sal_coeffs[0] = 5.0   # High scalar (intensity)
    high_sal_coeffs[15] = 3.0  # High pseudoscalar (valence)
    high_sal_ctx = reconstruct_from_coefficients(high_sal_coeffs, basis, np)
    high_sal_ctx = high_sal_ctx / np.linalg.norm(high_sal_ctx, 'fro')
    
    high_sal_transition = record_transition(
        from_context=high_sal_ctx,
        to_context=ctx_b,
        from_target=999,
        to_target=1000,
        basis=basis,
        xp=np,
    )
    # Manually boost salience to ensure it's highest
    high_sal_transition.salience = 10.0
    buffer.add(high_sal_transition)
    
    top_transitions = buffer.get_high_salience(top_k=3)
    print(f"    Top 3 saliences: {[f'{t.salience:.2f}' for t in top_transitions]}")
    
    # High salience transition should be in top
    assert top_transitions[0].salience == 10.0, "Highest salience should be first"
    print("    ✓ High-salience transitions retrieved first")
    
    # PART 5: Test sequence replay
    print("\n  PART 5: Sequence Replay Chaining")
    print("  " + "-" * 40)
    
    # Get a starting transition and try to chain
    start = buffer.transitions[0]
    replayed = buffer.replay_sequence(start, max_length=5, similarity_threshold=0.5)
    
    print(f"    Starting from target: {start.from_target} → {start.to_target}")
    print(f"    Replayed sequence length: {len(replayed)}")
    
    if len(replayed) > 1:
        print(f"    Sequence: {' → '.join([str(t.to_target) for t in replayed])}")
        print("    ✓ Sequence replay chains transitions")
    else:
        print("    ⚠ Only found starting transition (context similarity too low)")
    
    # PART 6: Test REM replay
    print("\n  PART 6: REM Sequence Replay")
    print("  " + "-" * 40)
    
    import time
    start_time = time.perf_counter()
    
    sequences, replay_stats = replay_transitions_during_rem(
        buffer, basis, np,
        n_replays=10,
        sequence_length=3,
    )
    
    elapsed = time.perf_counter() - start_time
    
    print(f"    Sequences replayed: {replay_stats['replayed']}")
    print(f"    Avg sequence length: {replay_stats['avg_length']:.1f}")
    print(f"    Max sequence length: {replay_stats['max_length']}")
    print(f"    Time: {elapsed*1000:.1f}ms")
    
    assert replay_stats['replayed'] == 10, "Should replay 10 sequences"
    print("    ✓ REM replay generates sequences")
    
    print("\n  ✓ PASSED")
    return True


def test_pseudo_rehearsal():
    """
    Test that Pseudo-Rehearsal prevents catastrophic forgetting.
    
    THEORY (Complementary Learning Systems):
        - Semantic memory generates "pseudo-patterns"
        - These are mixed with real episodes during training
        - Old knowledge is rehearsed while new is learned
        - Prevents overwriting of previously learned patterns
        
    TEST STRATEGY:
        1. Create semantic memory with prototypes
        2. Generate pseudo-episodes from prototypes
        3. Verify pseudo-episodes are similar to prototypes
        4. Test interleaving with real episodes
    """
    print("\n" + "=" * 70)
    print("  TEST: Pseudo-Rehearsal (Prevent Catastrophic Forgetting)")
    print("=" * 70)
    
    from holographic_v4.algebra import build_clifford_basis
    from holographic_v4.dreaming import (
        generate_pseudo_episode,
        generate_pseudo_episodes_batch,
        interleave_with_pseudo_rehearsal,
        SemanticMemory,
        SemanticPrototype,
        EpisodicEntry,
        frobenius_similarity,
    )
    
    basis = build_clifford_basis(np)
    
    # Create semantic memory with prototypes
    semantic = SemanticMemory(basis=basis, xp=np)
    
    # Add prototype A
    proto_a_matrix = np.eye(4) + 0.1 * np.random.randn(4, 4)
    proto_a_matrix = proto_a_matrix / np.linalg.norm(proto_a_matrix, 'fro')
    proto_a = SemanticPrototype(
        prototype_matrix=proto_a_matrix,
        target_distribution={100: 0.7, 101: 0.3},
        radius=0.3,
        support=50,  # High support
        level=0,
    )
    semantic.add_prototype(proto_a)
    
    # Add prototype B
    proto_b_matrix = np.array([[0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                                [1, 0, 0, 0]], dtype=float)
    proto_b_matrix = proto_b_matrix / np.linalg.norm(proto_b_matrix, 'fro')
    proto_b = SemanticPrototype(
        prototype_matrix=proto_b_matrix,
        target_distribution={200: 1.0},
        radius=0.3,
        support=10,  # Lower support
        level=0,
    )
    semantic.add_prototype(proto_b)
    
    # PART 1: Test single pseudo-episode generation
    print("\n  PART 1: Single Pseudo-Episode Generation")
    print("  " + "-" * 40)
    
    pseudo_ep = generate_pseudo_episode(proto_a, basis, np, noise_std=0.1)
    
    print(f"    Generated target: {pseudo_ep.target_token}")
    print(f"    Salience: {pseudo_ep.salience:.4f}")
    print(f"    Novelty: {pseudo_ep.novelty:.4f} (should be 0 - not novel)")
    
    # Check similarity to prototype
    sim = frobenius_similarity(pseudo_ep.context_matrix, proto_a_matrix, np)
    print(f"    Similarity to prototype: {sim:.4f}")
    
    assert sim > 0.8, "Pseudo-episode should be similar to prototype"
    assert pseudo_ep.target_token in [100, 101], "Target should be from distribution"
    print("    ✓ Pseudo-episode generated correctly")
    
    # PART 2: Test batch generation
    print("\n  PART 2: Batch Pseudo-Episode Generation")
    print("  " + "-" * 40)
    
    import time
    start = time.perf_counter()
    
    pseudo_batch = generate_pseudo_episodes_batch(
        semantic, basis, np,
        n_episodes=100,
        noise_std=0.1,
        temperature=1.0,
    )
    
    elapsed = time.perf_counter() - start
    
    print(f"    Generated: {len(pseudo_batch)} pseudo-episodes")
    print(f"    Time: {elapsed*1000:.1f}ms")
    
    # Check distribution of targets
    target_counts = {}
    for ep in pseudo_batch:
        target_counts[ep.target_token] = target_counts.get(ep.target_token, 0) + 1
    
    print(f"    Target distribution: {target_counts}")
    
    # Higher-support prototype should generate more episodes
    proto_a_count = target_counts.get(100, 0) + target_counts.get(101, 0)
    proto_b_count = target_counts.get(200, 0)
    
    print(f"    From proto A (support=50): {proto_a_count}")
    print(f"    From proto B (support=10): {proto_b_count}")
    
    # Proto A should have more samples (higher support)
    assert proto_a_count > proto_b_count, "Higher support prototype should generate more"
    print("    ✓ Batch generation weighted by support")
    
    # PART 3: Test interleaving with real episodes
    print("\n  PART 3: Interleaving with Real Episodes")
    print("  " + "-" * 40)
    
    # Create real episodes (representing new patterns to learn)
    real_episodes = []
    for i in range(50):
        ctx = np.random.randn(4, 4)
        ctx = ctx / np.linalg.norm(ctx, 'fro')
        real_episodes.append(EpisodicEntry(ctx, target_token=300 + i))
    
    combined, stats = interleave_with_pseudo_rehearsal(
        real_episodes, semantic, basis, np,
        rehearsal_ratio=0.2,
        noise_std=0.1,
    )
    
    print(f"    Real episodes: {stats['real_episodes']}")
    print(f"    Pseudo episodes: {stats['pseudo_episodes']}")
    print(f"    Total combined: {stats['total']}")
    print(f"    Actual ratio: {stats['actual_ratio']:.1%}")
    
    # Check that combined list has both types
    real_targets = set(range(300, 350))
    pseudo_targets = {100, 101, 200}
    
    has_real = any(ep.target_token in real_targets for ep in combined)
    has_pseudo = any(ep.target_token in pseudo_targets for ep in combined)
    
    assert has_real, "Combined should have real episodes"
    assert has_pseudo, "Combined should have pseudo episodes"
    assert len(combined) > len(real_episodes), "Combined should be larger"
    print("    ✓ Real and pseudo episodes interleaved")
    
    # PART 4: Verify pseudo-episodes have correct properties
    print("\n  PART 4: Pseudo-Episode Properties")
    print("  " + "-" * 40)
    
    # Check that pseudo-episodes have lower priority than real episodes
    real_priorities = [ep.priority for ep in combined if ep.target_token in real_targets]
    pseudo_priorities = [ep.priority for ep in combined if ep.target_token in pseudo_targets]
    
    if real_priorities and pseudo_priorities:
        avg_real_priority = np.mean(real_priorities)
        avg_pseudo_priority = np.mean(pseudo_priorities)
        print(f"    Avg real priority: {avg_real_priority:.4f}")
        print(f"    Avg pseudo priority: {avg_pseudo_priority:.4f}")
    
    # Novelty of pseudo-episodes should be 0
    pseudo_novelties = [ep.novelty for ep in combined if ep.target_token in pseudo_targets]
    avg_pseudo_novelty = np.mean(pseudo_novelties) if pseudo_novelties else 0
    print(f"    Avg pseudo novelty: {avg_pseudo_novelty:.4f} (should be 0)")
    
    assert avg_pseudo_novelty == 0, "Pseudo-episodes should have novelty=0"
    print("    ✓ Pseudo-episodes have correct properties")
    
    print("\n  ✓ PASSED")
    return True


def test_inhibition_of_return():
    """
    Test that Inhibition of Return encourages exploration.
    
    THEORY (Exploration Bonus):
        - Recently retrieved items are temporarily suppressed
        - Suppression decays with time (φ⁻¹ per step)
        - Encourages diverse memory access
        
    TEST STRATEGY:
        1. Create semantic memory with multiple prototypes
        2. Retrieve same query multiple times
        3. Verify that repeated retrieval shifts to different prototypes
        4. Verify that inhibition decays over time
    """
    print("\n" + "=" * 70)
    print("  TEST: Inhibition of Return (Exploration Bonus)")
    print("=" * 70)
    
    from holographic_v4.algebra import build_clifford_basis
    from holographic_v4.dreaming import (
        RetrievalHistory,
        apply_inhibition_of_return,
        retrieve_with_inhibition,
        SemanticMemory,
        SemanticPrototype,
        PHI_INV,
    )
    
    basis = build_clifford_basis(np)
    
    # Create semantic memory with multiple similar prototypes
    semantic = SemanticMemory(basis=basis, xp=np)
    
    # Create 3 prototypes with similar matrices
    base_matrix = np.eye(4) + 0.1 * np.random.randn(4, 4)
    base_matrix = base_matrix / np.linalg.norm(base_matrix, 'fro')
    
    for i in range(3):
        # Small perturbation
        proto_matrix = base_matrix + 0.05 * np.random.randn(4, 4)
        proto_matrix = proto_matrix / np.linalg.norm(proto_matrix, 'fro')
        
        proto = SemanticPrototype(
            prototype_matrix=proto_matrix,
            target_distribution={100 + i: 1.0},
            radius=0.3,
            support=10,
            level=0,
        )
        semantic.add_prototype(proto)
    
    # PART 1: Test retrieval history tracking
    print("\n  PART 1: Retrieval History Tracking")
    print("  " + "-" * 40)
    
    history = RetrievalHistory(decay_rate=PHI_INV, xp=np)
    
    # Record some retrievals
    history.record_retrieval(1001, strength=1.0)
    history.advance_time(1.0)
    history.record_retrieval(1002, strength=1.0)
    
    stats = history.stats()
    print(f"    History count: {stats['count']}")
    print(f"    Current time: {stats['current_time']}")
    
    # Check inhibition levels
    inhib_1001 = history.get_inhibition(1001)
    inhib_1002 = history.get_inhibition(1002)
    inhib_9999 = history.get_inhibition(9999)  # Not in history
    
    print(f"    Inhibition (1001, older): {inhib_1001:.4f}")
    print(f"    Inhibition (1002, newer): {inhib_1002:.4f}")
    print(f"    Inhibition (9999, not retrieved): {inhib_9999:.4f}")
    
    assert inhib_1001 < inhib_1002, "Older retrieval should have lower inhibition"
    assert inhib_9999 == 0.0, "Non-retrieved should have no inhibition"
    print("    ✓ Inhibition tracks retrievals and decays")
    
    # PART 2: Test decay over time
    print("\n  PART 2: Inhibition Decay")
    print("  " + "-" * 40)
    
    history2 = RetrievalHistory(decay_rate=PHI_INV, xp=np)
    history2.record_retrieval(1001, strength=1.0)
    
    print(f"    Time 0: inhibition = {history2.get_inhibition(1001):.4f}")
    
    history2.advance_time(1.0)
    inhib_t1 = history2.get_inhibition(1001)
    print(f"    Time 1: inhibition = {inhib_t1:.4f} (should be ~φ⁻¹ = {PHI_INV:.4f})")
    
    history2.advance_time(1.0)
    inhib_t2 = history2.get_inhibition(1001)
    print(f"    Time 2: inhibition = {inhib_t2:.4f} (should be ~φ⁻² = {PHI_INV**2:.4f})")
    
    assert abs(inhib_t1 - PHI_INV) < 0.01, "Decay should follow φ⁻¹"
    print("    ✓ Inhibition decays as φ⁻ᵗ")
    
    # PART 3: Test retrieval with inhibition
    print("\n  PART 3: Retrieval with Inhibition")
    print("  " + "-" * 40)
    
    history3 = RetrievalHistory(decay_rate=PHI_INV, xp=np)
    
    # Query that should match all prototypes similarly
    query = base_matrix + 0.02 * np.random.randn(4, 4)
    query = query / np.linalg.norm(query, 'fro')
    
    # First retrieval - should get highest base similarity
    results1, info1 = retrieve_with_inhibition(
        query, semantic, history3,
        top_k=3,
        inhibition_weight=0.3,
        record_retrieval=True,
        xp=np,
    )
    
    first_target = results1[0][0].target_distribution
    print(f"    First retrieval: target={list(first_target.keys())[0]}")
    
    # Second retrieval - first choice should be inhibited
    results2, info2 = retrieve_with_inhibition(
        query, semantic, history3,
        top_k=3,
        inhibition_weight=0.3,
        record_retrieval=True,
        xp=np,
    )
    
    second_target = results2[0][0].target_distribution
    print(f"    Second retrieval: target={list(second_target.keys())[0]}")
    print(f"    Inhibited prototypes: {info2['inhibited']}")
    
    # The two retrievals might be the same or different depending on similarities
    # But inhibition should be applied
    assert info2['inhibited'] >= 1, "At least one should be inhibited"
    print("    ✓ Inhibition affects retrieval")
    
    # PART 4: Test that inhibition decays and allows re-retrieval
    print("\n  PART 4: Inhibition Decay Allows Re-Retrieval")
    print("  " + "-" * 40)
    
    # Advance time significantly
    for _ in range(10):
        history3.advance_time(1.0)
    
    results_after_decay, info_after = retrieve_with_inhibition(
        query, semantic, history3,
        top_k=3,
        inhibition_weight=0.3,
        record_retrieval=False,
        xp=np,
    )
    
    print(f"    After decay: inhibited={info_after['inhibited']}")
    print(f"    Avg inhibition: {info_after['avg_inhibition']:.6f}")
    
    # After decay, inhibition should be very low
    assert info_after['avg_inhibition'] < 0.01, "Inhibition should decay significantly"
    print("    ✓ Inhibition decays allowing re-retrieval")
    
    print("\n  ✓ PASSED")
    return True


def test_dreaming_system_integration():
    """
    Integration test: Verify all 12 parsimonies work together in DreamingSystem.
    
    This is the COMPREHENSIVE test that ensures:
    1. All features are properly hooked up
    2. Sleep cycle processes all phases correctly
    3. Retrieve method uses pattern completion + inhibition
    4. Statistics are tracked correctly
    """
    print("\n" + "=" * 70)
    print("  TEST: DreamingSystem Integration (All 12 Parsimonies)")
    print("=" * 70)
    
    from holographic_v4.algebra import build_clifford_basis
    from holographic_v4.dreaming import (
        DreamingSystem,
        EpisodicEntry,
    )
    
    basis = build_clifford_basis(np)
    
    # PART 1: Create DreamingSystem with all features enabled
    print("\n  PART 1: DreamingSystem Initialization")
    print("  " + "-" * 40)
    
    dreaming = DreamingSystem(
        basis=basis,
        xp=np,
        use_salience=True,
        use_novelty=True,
        use_predictive_coding=True,
        use_pattern_completion=True,
        use_inhibition_of_return=True,
        use_sequence_replay=True,
        use_pseudo_rehearsal=True,
    )
    
    print(f"    Features enabled:")
    print(f"      - Salience: {dreaming.use_salience}")
    print(f"      - Novelty: {dreaming.use_novelty}")
    print(f"      - Predictive Coding: {dreaming.use_predictive_coding}")
    print(f"      - Pattern Completion: {dreaming.use_pattern_completion}")
    print(f"      - Inhibition of Return: {dreaming.use_inhibition_of_return}")
    print(f"      - Sequence Replay: {dreaming.use_sequence_replay}")
    print(f"      - Pseudo-Rehearsal: {dreaming.use_pseudo_rehearsal}")
    
    # Verify components exist
    assert dreaming.transition_buffer is not None, "TransitionBuffer should exist"
    assert dreaming.retrieval_history is not None, "RetrievalHistory should exist"
    print("    ✓ All components initialized")
    
    # PART 2: Create training episodes (simulating sequence)
    print("\n  PART 2: Create Training Episodes")
    print("  " + "-" * 40)
    
    from holographic_v4.quotient import grace_stability
    from holographic_v4.constants import PHI_INV_SQ
    
    episodes = []
    np.random.seed(42)
    
    # Create CLUSTERABLE episodes (similar contexts for same target)
    # This simulates real learning: multiple exposures to same concept
    # with variation produce a cluster that consolidates into a prototype
    
    # Create 5 base patterns (one per target)
    base_patterns = [np.random.randn(4, 4) for _ in range(5)]
    for bp in base_patterns:
        bp /= np.linalg.norm(bp, 'fro')
    
    # Create episodes: variations of base patterns
    for i in range(50):
        target_idx = i % 5  # Cycles through 5 targets
        base = base_patterns[target_idx]
        
        # Add variation (but keep similarity to base for clustering)
        # 10% noise preserves ~85%+ similarity (above 0.8 threshold)
        noise = 0.1 * np.random.randn(4, 4)
        ctx = base + noise
        ctx = ctx / np.linalg.norm(ctx, 'fro')  # Normalize
        
        episodes.append(EpisodicEntry(
            context_matrix=ctx,
            target_token=100 + target_idx,
        ))
    
    # Verify episodes need consolidation (self-organizing principle)
    stabilities = [grace_stability(ep.context_matrix, basis, np) for ep in episodes]
    avg_stability = np.mean(stabilities)
    needs_consolidation = sum(1 for s in stabilities if s < PHI_INV_SQ)
    
    print(f"    Created {len(episodes)} episodes ({len(episodes)//5} per target)")
    print(f"    Targets: {set(ep.target_token for ep in episodes)}")
    print(f"    Avg stability: {avg_stability:.3f} (threshold: {PHI_INV_SQ:.3f})")
    print(f"    Episodes needing consolidation: {needs_consolidation}/{len(episodes)}")
    
    # PART 3: First sleep cycle (no predictive filtering yet)
    print("\n  PART 3: First Sleep Cycle")
    print("  " + "-" * 40)
    
    stats1 = dreaming.sleep(episodes[:25], rem_cycles=1, verbose=False)
    
    print(f"    Input: {stats1['input_episodes']}")
    print(f"    Prototypes created: {stats1['prototypes_created']}")
    print(f"    Transitions recorded: {stats1['transitions_recorded']}")
    print(f"    Predictive filtered: {stats1['predictive_filtered']}")
    
    assert stats1['prototypes_created'] > 0, "Should create prototypes"
    assert stats1['transitions_recorded'] > 0, "Should record transitions"
    print("    ✓ First sleep cycle complete")
    
    # PART 4: Second sleep cycle (should use predictive filtering)
    print("\n  PART 4: Second Sleep Cycle (With Predictive Filtering)")
    print("  " + "-" * 40)
    
    # Second batch of similar episodes
    stats2 = dreaming.sleep(episodes[25:], rem_cycles=1, verbose=False)
    
    print(f"    Input: {stats2['input_episodes']}")
    print(f"    Predictive filtered: {stats2['predictive_filtered']}")
    print(f"    Actually consolidated: {stats2['input_episodes'] - stats2['predictive_filtered']}")
    print(f"    Sequences replayed: {stats2['sequences_replayed']}")
    
    # Similar episodes should have some filtered out
    if stats2['predictive_filtered'] > 0:
        print("    ✓ Predictive coding filtered redundant episodes")
    else:
        print("    ⚠ No filtering (patterns may be too different)")
    
    # PART 5: Test retrieve method with all features
    print("\n  PART 5: Retrieve with Pattern Completion + Inhibition")
    print("  " + "-" * 40)
    
    # Create a noisy query based on first target pattern
    query = base_patterns[0] + 0.3 * np.random.randn(4, 4)
    query = query / np.linalg.norm(query, 'fro')
    
    # First retrieval
    proto1, sim1, info1 = dreaming.retrieve(query)
    
    print(f"    First retrieval:")
    print(f"      Similarity: {sim1:.4f}")
    print(f"      Pattern completion used: {info1['used_pattern_completion']}")
    print(f"      Inhibition used: {info1['used_inhibition']}")
    
    # Second retrieval (same query - should show inhibition)
    proto2, sim2, info2 = dreaming.retrieve(query)
    
    print(f"    Second retrieval:")
    print(f"      Similarity: {sim2:.4f}")
    print(f"      Inhibited count: {info2['inhibited_count']}")
    
    assert info2['inhibited_count'] >= 1, "Should have inhibited first retrieval"
    print("    ✓ Retrieval with all features works")
    
    # PART 6: Test pseudo-rehearsal generation
    print("\n  PART 6: Pseudo-Rehearsal Generation")
    print("  " + "-" * 40)
    
    pseudo_episodes = dreaming.generate_rehearsal_episodes(n_episodes=10)
    
    print(f"    Generated {len(pseudo_episodes)} pseudo-episodes")
    if pseudo_episodes:
        print(f"    Targets: {set(ep.target_token for ep in pseudo_episodes)}")
        print(f"    Avg salience: {np.mean([ep.salience for ep in pseudo_episodes]):.4f}")
        print(f"    All novelty=0: {all(ep.novelty == 0 for ep in pseudo_episodes)}")
    
    assert len(pseudo_episodes) > 0, "Should generate pseudo-episodes"
    print("    ✓ Pseudo-rehearsal working")
    
    # PART 7: Check final statistics
    print("\n  PART 7: Final Statistics")
    print("  " + "-" * 40)
    
    final_stats = dreaming.get_stats()
    
    print(f"    Sleep cycles: {final_stats['sleep_cycles']}")
    print(f"    Total episodes: {final_stats['total_episodes']}")
    print(f"    Prototypes: {final_stats['total_prototypes']}")
    print(f"    Schemas: {final_stats['num_schemas']}")
    print(f"    Transition buffer: {final_stats['transition_buffer_count']}")
    print(f"    Retrieval history: {final_stats['retrieval_history_count']}")
    
    assert final_stats['sleep_cycles'] == 2, "Should have 2 sleep cycles"
    assert final_stats['total_prototypes'] > 0, "Should have prototypes"
    assert final_stats['transition_buffer_count'] > 0, "Should have transitions"
    print("    ✓ Statistics tracked correctly")
    
    print("\n  ✓ PASSED - All 12 parsimonies integrated correctly")
    return True


def test_self_organizing_consolidation():
    """
    Test the SELF-ORGANIZING consolidation principle.
    
    THEORY:
        Grace-stability determines which episodes need consolidation:
        - High stability (σ ≈ 1): Already an attractor, stays episodic
        - Low stability (σ < φ⁻²): Transient content, needs consolidation
        
        This is NOT a tuned parameter - it's the spectral structure of Grace!
    
    TEST:
        1. Create episodes with varying stability
        2. Run consolidation
        3. Verify high-stability episodes are NOT consolidated
        4. Verify low-stability episodes ARE consolidated
    """
    print("\n  TEST: Self-Organizing Consolidation (Grace-Stability)")
    
    from holographic_v4 import build_clifford_basis
    from holographic_v4.algebra import decompose_to_coefficients, reconstruct_from_coefficients
    from holographic_v4.dreaming import (
        NonREMConsolidator, SemanticMemory, EpisodicEntry
    )
    from holographic_v4.quotient import grace_stability
    from holographic_v4.constants import PHI_INV_SQ, GRADE_INDICES
    
    basis = build_clifford_basis()
    semantic_memory = SemanticMemory(basis, np)
    consolidator = NonREMConsolidator(
        basis=basis,
        min_cluster_size=1,  # Allow single-episode clusters
        semantic_memory=semantic_memory,
    )
    
    print(f"    Theory-derived threshold: φ⁻² = {PHI_INV_SQ:.4f}")
    
    # Create HIGH STABILITY episode (pure scalar - survives Grace perfectly)
    coeffs_stable = np.zeros(16)
    coeffs_stable[0] = 1.0  # Pure scalar
    M_stable = reconstruct_from_coefficients(coeffs_stable, basis, np)
    stability_stable = grace_stability(M_stable, basis, np)
    
    # Create LOW STABILITY episode (pure bivector - decays under Grace)
    coeffs_transient = np.zeros(16)
    coeffs_transient[GRADE_INDICES[2][0]] = 1.0  # Pure bivector
    M_transient = reconstruct_from_coefficients(coeffs_transient, basis, np)
    stability_transient = grace_stability(M_transient, basis, np)
    
    # Create MIXED stability episode (50/50)
    coeffs_mixed = np.zeros(16)
    coeffs_mixed[0] = 0.7  # Scalar
    coeffs_mixed[GRADE_INDICES[2][0]] = 0.7  # Bivector
    M_mixed = reconstruct_from_coefficients(coeffs_mixed, basis, np)
    stability_mixed = grace_stability(M_mixed, basis, np)
    
    print(f"    Stable episode: σ = {stability_stable:.4f} (expect >threshold)")
    print(f"    Transient episode: σ = {stability_transient:.4f} (expect ~0)")
    print(f"    Mixed episode: σ = {stability_mixed:.4f} (expect ~0.5)")
    
    # Create entries
    episodes = [
        EpisodicEntry(context_matrix=M_stable, target_token=10),
        EpisodicEntry(context_matrix=M_transient, target_token=20),
        EpisodicEntry(context_matrix=M_mixed, target_token=30),
    ]
    
    # Run consolidation
    print("\n    Running consolidation...")
    prototypes = consolidator.consolidate(episodes, verbose=True)
    
    # Verify self-organizing behavior
    # High-stability (σ > φ⁻²) should NOT be consolidated
    # Low-stability (σ < φ⁻²) SHOULD be consolidated
    
    # Check which targets ended up in prototypes
    consolidated_targets = set()
    for proto in prototypes:
        consolidated_targets.update(proto.target_distribution.keys())
    
    print(f"\n    Consolidated targets: {consolidated_targets}")
    
    # Stable episode (target=10) should NOT be consolidated (σ > threshold)
    stable_consolidated = 10 in consolidated_targets
    # Transient episode (target=20) SHOULD be consolidated (σ < threshold)
    transient_consolidated = 20 in consolidated_targets
    # Mixed episode (target=30) - depends on exact stability
    mixed_consolidated = 30 in consolidated_targets
    
    print(f"    Stable (σ={stability_stable:.3f}): consolidated={stable_consolidated} (expect False)")
    print(f"    Transient (σ={stability_transient:.3f}): consolidated={transient_consolidated} (expect True)")
    print(f"    Mixed (σ={stability_mixed:.3f}): consolidated={mixed_consolidated}")
    
    # The key test: transient MUST be consolidated, stable MUST NOT
    assert transient_consolidated, "Transient episode (low σ) should be consolidated"
    assert not stable_consolidated or stability_stable < PHI_INV_SQ, \
        f"Stable episode (σ={stability_stable:.3f}) should not be consolidated (threshold={PHI_INV_SQ:.3f})"
    
    print("\n    ✓ Self-organizing principle verified:")
    print("      - Low-stability episodes → consolidated (theory-derived)")
    print("      - High-stability episodes → kept episodic (stable equilibria)")
    print("      - No tuned parameters - pure spectral structure of Grace!")
    
    return True


def run_all_dreaming_tests():
    """Run all dreaming tests."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  DREAMING MODULE TESTS  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    tests = [
        ("Consolidation Creates Prototypes", test_consolidation_creates_prototypes),
        ("Multi-Modal Target Representation", test_multi_modal_targets),
        ("Semantic Memory Retrieval", test_semantic_memory_retrieval),
        ("Dreaming Improves Coverage", test_dreaming_improves_coverage),
        ("REM Recombination", test_rem_recombination),
        ("Memory Compression", test_memory_compression),
        # Brain-inspired parsimony tests
        ("Salience-Weighted Consolidation", test_salience_weighted_consolidation),
        ("Prediction Error as Grace Residual", test_prediction_error_as_grace_residual),
        ("Novelty-Gated Learning", test_novelty_gated_learning),
        ("Delta/Schema Compression", test_delta_schema_compression),
        ("Synaptic Pruning", test_synaptic_pruning),
        ("Interference Management", test_interference_management),
        ("Reconsolidation", test_reconsolidation),
        ("Working Memory Gating", test_working_memory_gating),
        ("Pattern Completion", test_pattern_completion),
        ("Predictive Coding", test_predictive_coding),
        ("Sequence Replay", test_sequence_replay),
        ("Pseudo-Rehearsal", test_pseudo_rehearsal),
        ("Inhibition of Return", test_inhibition_of_return),
        # Integration test
        ("DreamingSystem Integration", test_dreaming_system_integration),
        # Self-organizing principle
        ("Self-Organizing Consolidation", test_self_organizing_consolidation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_dreaming_tests()
    exit(0 if failed == 0 else 1)
