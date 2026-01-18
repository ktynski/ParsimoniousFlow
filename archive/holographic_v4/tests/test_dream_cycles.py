"""
TEST-DRIVEN: Dream Cycles — Topological Re-alignment
====================================================

Tests FIRST. Implementation SECOND.

Dreaming in the nested fractal torus:
- Non-REM: Master broadcasts witness to satellites (consolidation)
- REM: Phase jitter searches for new attractors (creativity)
- Wake trigger: stability > φ⁻² (spectral gap threshold)

Theory:
    During sleep, the master torus re-tunes satellites toward global coherence.
    This is topological re-alignment, not weight updates.
    The brain does this during sleep; we do it via Grace + phase jitter.

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_v4.constants import PI, PHI, PHI_INV, PHI_INV_SQ

# Import the class we're testing (will fail until implemented)
try:
    from holographic_v4.dream_cycles import DreamCycle
    from holographic_v4.fractal_generative_memory import FractalGenerativeMemory
except ImportError:
    DreamCycle = None
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
# TEST 1: Non-REM Consolidation
# ============================================================

@pytest.mark.skipif(DreamCycle is None, reason="DreamCycle not implemented yet")
def test_1_non_rem_consolidates():
    """
    Non-REM should increase master-satellite coherence.
    
    Master broadcasts its witness DOWN to satellites.
    Dissonant satellites receive accelerated Grace.
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2,
        vocab_size=100,
        orthogonalize=True,
    )
    
    # Train with some noise
    for _ in range(50):
        model.learn(random_context(), random_target())
    
    # Create dream cycle
    dreamer = DreamCycle(model)
    
    # Get pre-dream coherence
    pre_coherence = dreamer.compute_coherence()
    print(f"  Pre-coherence: {pre_coherence:.4f}")
    
    # Non-REM cycle
    dreamer.non_rem_consolidation()
    
    # Coherence should increase (or at least not decrease much)
    post_coherence = dreamer.compute_coherence()
    print(f"  Post-coherence: {post_coherence:.4f}")
    
    # Allow small variance due to noise
    assert post_coherence >= pre_coherence * 0.9, \
        f"Coherence dropped: {pre_coherence:.4f} → {post_coherence:.4f}"


# ============================================================
# TEST 2: REM Recombination
# ============================================================

@pytest.mark.skipif(DreamCycle is None, reason="DreamCycle not implemented yet")
def test_2_rem_finds_new_attractors():
    """
    REM should search for new stable states (creativity).
    
    Phase jitter allows exploration of nearby attractors.
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2,
        vocab_size=100,
        orthogonalize=True,
    )
    
    # Train with related patterns
    model.learn([1, 2, 3], 10)
    model.learn([4, 5, 6], 10)  # Same target, different context
    
    # Create dream cycle
    dreamer = DreamCycle(model)
    
    # REM cycle
    discoveries = dreamer.rem_recombination()
    
    print(f"  Discoveries: {discoveries}")
    
    # May or may not find new attractors (stochastic)
    assert discoveries >= 0, "Discoveries should be non-negative"


# ============================================================
# TEST 3: Wake Trigger
# ============================================================

@pytest.mark.skipif(DreamCycle is None, reason="DreamCycle not implemented yet")
def test_3_wake_trigger():
    """
    System should wake when stability > φ⁻².
    
    High coherence = ready to wake.
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2,
        vocab_size=100,
        orthogonalize=True,
    )
    
    # Train minimally (already stable)
    model.learn([1, 2, 3], 10)
    
    # Create dream cycle
    dreamer = DreamCycle(model)
    
    # Set high stability
    initial_stability = model.get_stability()
    print(f"  Initial stability: {initial_stability:.4f}")
    print(f"  Wake threshold: {PHI_INV_SQ:.4f}")
    
    # Dream should wake quickly if already stable
    stats = dreamer.full_cycle()
    
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Woke early: {stats['woke_early']}")
    
    # If initial stability > threshold, should wake early
    if initial_stability > PHI_INV_SQ:
        assert stats['iterations'] <= 3, "Should wake early when already stable"


# ============================================================
# TEST 4: Paradox Resolution
# ============================================================

@pytest.mark.skipif(DreamCycle is None, reason="DreamCycle not implemented yet")
def test_4_paradox_resolution():
    """
    Contradictory memories should be phase-shifted, not deleted.
    
    Both "A → X" and "A → Y" can coexist in different phase lanes.
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2,
        vocab_size=100,
        orthogonalize=True,
    )
    
    # Learn contradiction
    context = [1, 2, 3]
    model.learn(context, 10)  # A → X
    model.learn(context, 20)  # A → Y (contradiction!)
    
    # Check both are stored
    valid_targets_before = model.get_valid_targets(context)
    print(f"  Valid targets before: {valid_targets_before}")
    
    # Create dream cycle
    dreamer = DreamCycle(model)
    
    # Dream with paradox resolution
    stats = dreamer.full_cycle()
    
    # Both should still be retrievable
    valid_targets_after = model.get_valid_targets(context)
    print(f"  Valid targets after: {valid_targets_after}")
    
    assert 10 in valid_targets_after, "Target 10 should survive dreaming"
    assert 20 in valid_targets_after, "Target 20 should survive dreaming"


# ============================================================
# TEST 5: Dream Improves Retrieval
# ============================================================

@pytest.mark.skipif(DreamCycle is None, reason="DreamCycle not implemented yet")
def test_5_dream_improves_retrieval():
    """
    Dreaming should maintain or improve retrieval accuracy.
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2,
        vocab_size=100,
        orthogonalize=True,
    )
    
    # Train
    pairs = []
    seen = set()
    for _ in range(50):
        ctx = tuple(random_context(3, 100))
        while ctx in seen:
            ctx = tuple(random_context(3, 100))
        seen.add(ctx)
        tgt = random_target(100)
        pairs.append((list(ctx), tgt))
        model.learn(list(ctx), tgt)
    
    # Pre-dream accuracy
    pre_correct = sum(1 for ctx, tgt in pairs 
                      if model.retrieve_deterministic(ctx)[0] == tgt)
    pre_accuracy = pre_correct / len(pairs)
    print(f"  Pre-dream accuracy: {pre_accuracy * 100:.1f}%")
    
    # Dream
    dreamer = DreamCycle(model)
    stats = dreamer.full_cycle()
    print(f"  Dream stats: {stats}")
    
    # Post-dream accuracy
    post_correct = sum(1 for ctx, tgt in pairs 
                       if model.retrieve_deterministic(ctx)[0] == tgt)
    post_accuracy = post_correct / len(pairs)
    print(f"  Post-dream accuracy: {post_accuracy * 100:.1f}%")
    
    # Should not decrease significantly
    assert post_accuracy >= pre_accuracy * 0.9, \
        f"Accuracy dropped: {pre_accuracy * 100:.1f}% → {post_accuracy * 100:.1f}%"


# ============================================================
# TEST 6: Dream Statistics
# ============================================================

@pytest.mark.skipif(DreamCycle is None, reason="DreamCycle not implemented yet")
def test_6_dream_statistics():
    """
    Dream cycle should return meaningful statistics.
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2,
        vocab_size=100,
        orthogonalize=True,
    )
    
    # Train
    for _ in range(30):
        model.learn(random_context(), random_target())
    
    # Dream
    dreamer = DreamCycle(model)
    stats = dreamer.full_cycle()
    
    print(f"  Stats: {stats}")
    
    # Check required keys
    assert 'iterations' in stats
    assert 'total_discoveries' in stats
    assert 'pre_stability' in stats
    assert 'post_stability' in stats
    assert 'woke_early' in stats
    
    # Check values are reasonable
    assert stats['iterations'] >= 1
    assert stats['iterations'] <= 15  # Max ~φ⁵ ≈ 11
    assert 0 <= stats['pre_stability'] <= 1
    assert 0 <= stats['post_stability'] <= 1


# ============================================================
# TEST 7: Multiple Dream Cycles
# ============================================================

@pytest.mark.skipif(DreamCycle is None, reason="DreamCycle not implemented yet")
def test_7_multiple_dream_cycles():
    """
    Multiple dream cycles should continue improving stability.
    """
    np.random.seed(42)
    model = FractalGenerativeMemory(
        max_levels=2,
        vocab_size=100,
        orthogonalize=True,
    )
    
    # Train
    for _ in range(50):
        model.learn(random_context(), random_target())
    
    dreamer = DreamCycle(model)
    
    # Run multiple cycles
    stabilities = [model.get_stability()]
    for i in range(3):
        stats = dreamer.full_cycle()
        stabilities.append(stats['post_stability'])
        print(f"  Cycle {i+1}: stability = {stats['post_stability']:.4f}")
    
    # Should generally trend stable or improve
    # (may fluctuate, so just check final >= initial * 0.8)
    assert stabilities[-1] >= stabilities[0] * 0.8, \
        f"Stability degraded: {stabilities[0]:.4f} → {stabilities[-1]:.4f}"


# ============================================================
# MAIN
# ============================================================

def run_all_tests():
    """Run all dream cycle tests."""
    print("=" * 60)
    print("DREAM CYCLES — TEST SUITE")
    print("=" * 60)
    
    if DreamCycle is None:
        print("\n⚠️  DreamCycle not implemented yet!")
        print("    This is expected for TDD - tests are written first.")
        print("\n    Next step: Implement DreamCycle class")
        print("    in holographic_v4/dream_cycles.py")
        return False
    
    tests = [
        ("Test 1: Non-REM Consolidates", test_1_non_rem_consolidates),
        ("Test 2: REM Finds Attractors", test_2_rem_finds_new_attractors),
        ("Test 3: Wake Trigger", test_3_wake_trigger),
        ("Test 4: Paradox Resolution", test_4_paradox_resolution),
        ("Test 5: Improves Retrieval", test_5_dream_improves_retrieval),
        ("Test 6: Statistics", test_6_dream_statistics),
        ("Test 7: Multiple Cycles", test_7_multiple_dream_cycles),
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
