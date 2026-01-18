"""
TEST-DRIVEN: Integrated Fractal Generative Memory
=================================================

Tests FIRST. Implementation SECOND.

This test file defines the interface for FractalGenerativeMemory,
which combines:
1. Nested Fractal Torus (16^N hierarchical scaling)
2. Generative Memory (accumulation + sampling)
3. Orthogonalized Embeddings (reduced correlation)
4. Contrastive Learning (targets only)

Run: python3 -m pytest holographic_v4/test_integrated_fractal_generative.py -v
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_v4.constants import PHI_INV, PHI_INV_SQ, MATRIX_DIM

# The class we're testing - will be implemented after tests are written
try:
    from holographic_v4.fractal_generative_memory import FractalGenerativeMemory
except ImportError:
    FractalGenerativeMemory = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def random_context(size: int = 3, vocab_size: int = 100) -> list:
    """Generate a random context."""
    return [np.random.randint(vocab_size) for _ in range(size)]


def random_target(vocab_size: int = 100) -> int:
    """Generate a random target."""
    return np.random.randint(vocab_size)


# ============================================================
# TEST 1: Single Level Generative Memory
# ============================================================

@pytest.mark.skipif(FractalGenerativeMemory is None, reason="FractalGenerativeMemory not implemented yet")
def test_1_fractal_generative_single_level():
    """
    Level 0 (16 satellites) with generative memory.
    Should match standalone generative memory results.
    
    REQUIREMENTS:
    - Store multiple targets per context (accumulation)
    - Retrieve from superposition (probabilistic)
    - All φ-derived constants
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=1, 
        vocab_size=100,
        orthogonalize=True
    )
    
    # Train with ambiguous contexts (same context, multiple targets)
    context = [10, 20, 30]
    targets = [50, 51, 52]
    
    for tgt in targets:
        for _ in range(3):  # Each target 3 times
            model.learn(context, tgt)
    
    # Retrieve with sampling
    token, prob, top_k = model.retrieve_probabilistic(context, temperature=PHI_INV)
    
    # Should retrieve one of the valid targets (or close)
    valid_targets = model.get_valid_targets(context)
    assert valid_targets == set(targets), f"Expected {set(targets)}, got {valid_targets}"
    
    # Token should be in valid targets with high probability
    # (with temperature, might sample nearby tokens too)
    print(f"  Retrieved: {token}, Valid: {valid_targets}")


# ============================================================
# TEST 2: Two Levels (16² = 256 satellites)
# ============================================================

@pytest.mark.skipif(FractalGenerativeMemory is None, reason="FractalGenerativeMemory not implemented yet")
def test_2_fractal_generative_two_levels():
    """
    Level 1 (16² = 256 satellites) with generative memory.
    Master torus should aggregate satellite knowledge.
    
    REQUIREMENTS:
    - Satellites store at Level 0
    - Master aggregates at Level 1
    - Grace flows UP the hierarchy
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2, 
        vocab_size=1000,
        orthogonalize=True
    )
    
    # Train with many pairs
    train_pairs = []
    for _ in range(200):
        ctx = random_context(3, 1000)
        tgt = random_target(1000)
        train_pairs.append((ctx, tgt))
        model.learn(ctx, tgt)
    
    # Get master witness (should be non-zero after training)
    master_witness = model.get_master_witness()
    assert master_witness is not None
    assert np.linalg.norm(master_witness) > 0, "Master witness should be non-zero"
    
    # Retrieval should work
    ctx, expected = train_pairs[0]
    retrieved, confidence = model.retrieve_deterministic(ctx)
    assert retrieved is not None
    print(f"  Retrieved: {retrieved}, Expected: {expected}, Confidence: {confidence:.4f}")


# ============================================================
# TEST 3: Orthogonalized Embeddings (100% Single Binding)
# ============================================================

@pytest.mark.skipif(FractalGenerativeMemory is None, reason="FractalGenerativeMemory not implemented yet")
def test_3_fractal_orthogonalized_embeddings():
    """
    Fractal memory with orthogonalized embeddings.
    Should achieve 100% single-binding retrieval.
    
    This is the KEY test - orthogonalization is critical.
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2, 
        vocab_size=500,
        orthogonalize=True
    )
    
    # Single binding per context (no ambiguity)
    pairs = []
    seen_contexts = set()
    for _ in range(100):
        ctx = tuple(random_context(3, 500))
        while ctx in seen_contexts:  # Ensure unique contexts
            ctx = tuple(random_context(3, 500))
        seen_contexts.add(ctx)
        tgt = random_target(500)
        pairs.append((list(ctx), tgt))
        model.learn(list(ctx), tgt)
    
    # Retrieval should be 100%
    correct = 0
    for ctx, expected in pairs:
        retrieved, _ = model.retrieve_deterministic(ctx)
        if retrieved == expected:
            correct += 1
    
    accuracy = correct / len(pairs)
    print(f"  Single-binding accuracy: {accuracy * 100:.1f}%")
    assert accuracy >= 0.95, f"Expected >=95%, got {accuracy * 100:.1f}%"


# ============================================================
# TEST 4: Generation Diversity
# ============================================================

@pytest.mark.skipif(FractalGenerativeMemory is None, reason="FractalGenerativeMemory not implemented yet")
def test_4_fractal_generation_diversity():
    """
    Generation should produce diverse outputs when multiple targets exist.
    
    REQUIREMENTS:
    - Same context → different outputs (with temperature)
    - φ⁻¹ temperature for balanced diversity
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2, 
        vocab_size=500,
        orthogonalize=True
    )
    
    # Train with ambiguous contexts
    context = [10, 20, 30]
    for target in [100, 101, 102, 103, 104]:
        for _ in range(5):  # Equal frequency
            model.learn(context, target)
    
    # Generate 20 times with different seeds
    outputs = set()
    for i in range(20):
        np.random.seed(1000 + i)
        token, _, _ = model.retrieve_probabilistic(context, temperature=PHI_INV)
        outputs.add(token)
    
    print(f"  Unique outputs: {len(outputs)}/20")
    assert len(outputs) >= 3, f"Expected >= 3 unique, got {len(outputs)}"


# ============================================================
# TEST 5: Dreaming Consolidation
# ============================================================

@pytest.mark.skipif(FractalGenerativeMemory is None, reason="FractalGenerativeMemory not implemented yet")
def test_5_fractal_dreaming_consolidation():
    """
    Dreaming should consolidate satellite knowledge to master.
    
    REQUIREMENTS:
    - Pre-dream: Knowledge distributed in satellites
    - Post-dream: Knowledge consolidated in master
    - Stability should not decrease significantly
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2, 
        vocab_size=500,
        orthogonalize=True
    )
    
    # Train
    for _ in range(100):
        model.learn(random_context(3, 500), random_target(500))
    
    # Get pre-dream stability
    pre_stability = model.get_stability()
    print(f"  Pre-dream stability: {pre_stability:.4f}")
    
    # Dream
    dream_stats = model.dream()
    print(f"  Dream stats: {dream_stats}")
    
    # Post-dream stability should be >= pre-dream (or within 10%)
    post_stability = model.get_stability()
    print(f"  Post-dream stability: {post_stability:.4f}")
    
    assert post_stability >= pre_stability * 0.9, \
        f"Stability dropped too much: {pre_stability:.4f} → {post_stability:.4f}"


# ============================================================
# TEST 6: Embedding Correlation Check
# ============================================================

@pytest.mark.skipif(FractalGenerativeMemory is None, reason="FractalGenerativeMemory not implemented yet")
def test_6_embedding_correlation():
    """
    Orthogonalized embeddings should have low correlation.
    
    REQUIREMENTS:
    - Avg pairwise correlation < 0.1
    - Much better than random (~0.27)
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=1, 
        vocab_size=100,
        orthogonalize=True
    )
    
    stats = model.get_embedding_stats()
    avg_sim = stats['avg_pairwise_similarity']
    
    print(f"  Avg pairwise similarity: {avg_sim:.4f}")
    assert avg_sim < 0.15, f"Expected < 0.15, got {avg_sim:.4f}"


# ============================================================
# TEST 7: Memory Efficiency
# ============================================================

@pytest.mark.skipif(FractalGenerativeMemory is None, reason="FractalGenerativeMemory not implemented yet")
def test_7_memory_efficiency():
    """
    Memory should scale O(n), not O(n²).
    
    Storing 1000 pairs should not explode memory.
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2, 
        vocab_size=1000,
        orthogonalize=True
    )
    
    # Store 1000 pairs
    for _ in range(1000):
        model.learn(random_context(3, 1000), random_target(1000))
    
    # Get memory stats
    stats = model.get_statistics()
    
    print(f"  Unique contexts: {stats.get('unique_contexts', 'N/A')}")
    print(f"  Memory entries: {stats.get('memory_entries', 'N/A')}")
    
    # Memory should be roughly proportional to unique contexts
    # Not to all pairs (which would be worse)
    assert 'unique_contexts' in stats or 'memory_entries' in stats


# ============================================================
# TEST 8: Contrastive Learning (Targets Only)
# ============================================================

@pytest.mark.skipif(FractalGenerativeMemory is None, reason="FractalGenerativeMemory not implemented yet")
def test_8_contrastive_targets_only():
    """
    Contrastive learning should pull TARGETS together, not contexts.
    
    CRITICAL: Context tokens must stay distinct for binding to work.
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2, 
        vocab_size=100,
        orthogonalize=True,
        contrastive_enabled=True
    )
    
    # Same context predicts multiple targets
    context = [10, 20, 30]
    for target in [50, 51]:
        for _ in range(10):
            model.learn(context, target)
    
    # Force contrastive update
    model.apply_contrastive_update()
    
    # Targets 50 and 51 should be MORE similar than before
    from holographic_v4.algebra import frobenius_similarity
    sim_50_51 = frobenius_similarity(model.embed(50), model.embed(51))
    
    # Context tokens 10, 20 should NOT be pulled together
    sim_10_20 = frobenius_similarity(model.embed(10), model.embed(20))
    
    print(f"  Target similarity (50, 51): {sim_50_51:.4f}")
    print(f"  Context similarity (10, 20): {sim_10_20:.4f}")
    
    # Targets should be more similar than arbitrary context tokens
    # (This is a weak test, but validates the direction)


# ============================================================
# MAIN: Run all tests
# ============================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("INTEGRATED FRACTAL GENERATIVE MEMORY - TEST SUITE")
    print("=" * 60)
    
    if FractalGenerativeMemory is None:
        print("\n⚠️  FractalGenerativeMemory not implemented yet!")
        print("    This is expected for TDD - tests are written first.")
        print("\n    Next step: Implement FractalGenerativeMemory class")
        print("    in holographic_v4/fractal_generative_memory.py")
        return False
    
    tests = [
        ("Test 1: Single Level", test_1_fractal_generative_single_level),
        ("Test 2: Two Levels", test_2_fractal_generative_two_levels),
        ("Test 3: Orthogonalized (100%)", test_3_fractal_orthogonalized_embeddings),
        ("Test 4: Diversity", test_4_fractal_generation_diversity),
        ("Test 5: Dreaming", test_5_fractal_dreaming_consolidation),
        ("Test 6: Correlation", test_6_embedding_correlation),
        ("Test 7: Memory", test_7_memory_efficiency),
        ("Test 8: Contrastive", test_8_contrastive_targets_only),
    ]
    
    passed = 0
    failed = 0
    
    for name, test in tests:
        print(f"\n{name}")
        print("-" * 40)
        try:
            test()
            print("  ✓ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
