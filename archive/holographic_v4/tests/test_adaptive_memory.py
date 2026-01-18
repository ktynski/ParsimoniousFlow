"""
Tests for AdaptiveMemory — Integrated Memory System
===================================================

Verifies the production-ready memory with:
- FractalGenerativeMemory
- CreditAssignmentTracker
- Meta-learning adaptive rates
"""

import numpy as np
import pytest
from typing import List, Tuple

from holographic_v4.adaptive_memory import (
    AdaptiveMemory,
    AdaptiveMemoryConfig,
    create_adaptive_memory,
)
from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def memory():
    """Create an AdaptiveMemory for testing."""
    return create_adaptive_memory(
        vocab_size=100,
        max_levels=1,
        use_adaptive_rates=True,
        use_curriculum=False,  # Disable for simpler testing
        seed=42,
    )


# =============================================================================
# TEST 1: Basic Learning and Retrieval
# =============================================================================

def test_1_basic_learning_retrieval(memory):
    """
    Test basic learn/retrieve functionality.
    """
    print("\n=== Test 1: Basic Learning and Retrieval ===")
    
    # Learn a pattern
    context = [1, 2, 3]
    target = 50
    
    stats = memory.learn_adaptive(context, target)
    print(f"Learn stats: {stats}")
    
    # Retrieve
    retrieved, confidence = memory.retrieve(context)
    print(f"Retrieved: {retrieved} (confidence: {confidence:.4f})")
    
    assert retrieved == target, f"Should retrieve {target}, got {retrieved}"
    assert confidence > 0, "Confidence should be positive"
    
    print("✓ Basic learning and retrieval works")


# =============================================================================
# TEST 2: Adaptive Rate Modulation
# =============================================================================

def test_2_adaptive_rate_modulation(memory):
    """
    Test that rates are modulated based on novelty/uncertainty.
    """
    print("\n=== Test 2: Adaptive Rate Modulation ===")
    
    rates_used = []
    novelties = []
    
    # First, learn some patterns to establish baseline
    for i in range(5):
        context = [i, i+1, i+2]
        target = i * 10
        memory.learn_adaptive(context, target)
    
    # Now learn with a mix of new and repeated patterns
    for i in range(10):
        if i < 5:
            # Repeat existing pattern (lower novelty)
            context = [i, i+1, i+2]
            target = i * 10
        else:
            # New pattern (high novelty)
            context = [i+100, i+101, i+102]
            target = (i + 100) * 10
        
        stats = memory.learn_adaptive(context, target)
        rates_used.append(stats['rate_used'])
        novelties.append(stats['novelty'])
        
        print(f"  Pattern {i}: novelty={stats['novelty']:.2f}, "
              f"uncertainty={stats['uncertainty']:.2f}, rate={stats['rate_used']:.4f}")
    
    # Novelties should vary (some repeated, some new)
    unique_novelties = set([f"{n:.2f}" for n in novelties])
    assert len(unique_novelties) > 1, (
        f"Novelties should vary: {unique_novelties}"
    )
    
    # Rates should be within bounds
    min_rate = PHI_INV * PHI_INV
    max_rate = PHI_INV * PHI
    
    for rate in rates_used:
        assert min_rate <= rate <= max_rate, (
            f"Rate {rate:.4f} out of bounds [{min_rate:.4f}, {max_rate:.4f}]"
        )
    
    # Repeated patterns should have lower novelty
    repeated_novelties = novelties[:5]
    new_novelties = novelties[5:]
    
    avg_repeated = np.mean(repeated_novelties)
    avg_new = np.mean(new_novelties)
    
    print(f"Avg novelty (repeated): {avg_repeated:.2f}")
    print(f"Avg novelty (new): {avg_new:.2f}")
    
    assert avg_new > avg_repeated, (
        f"New patterns should have higher novelty: {avg_new:.2f} vs {avg_repeated:.2f}"
    )
    
    print("✓ Adaptive rate modulation works")


# =============================================================================
# TEST 3: Novelty Detection
# =============================================================================

def test_3_novelty_detection(memory):
    """
    Test that novel contexts have higher novelty scores.
    """
    print("\n=== Test 3: Novelty Detection ===")
    
    # First occurrence should be novel
    context = [10, 20, 30]
    stats1 = memory.learn_adaptive(context, 100)
    novelty1 = stats1['novelty']
    print(f"First occurrence: novelty={novelty1:.4f}")
    
    assert novelty1 == 1.0, f"First occurrence should be fully novel, got {novelty1}"
    
    # Repeat same context - novelty should decrease
    for i in range(5):
        stats = memory.learn_adaptive(context, 100)
    
    novelty_after = stats['novelty']
    print(f"After 5 more occurrences: novelty={novelty_after:.4f}")
    
    assert novelty_after < novelty1, (
        f"Novelty should decrease: {novelty1:.4f} -> {novelty_after:.4f}"
    )
    
    # New context should be novel again
    new_context = [99, 98, 97]
    stats_new = memory.learn_adaptive(new_context, 200)
    print(f"New context: novelty={stats_new['novelty']:.4f}")
    
    assert stats_new['novelty'] == 1.0, "New context should be fully novel"
    
    print("✓ Novelty detection works")


# =============================================================================
# TEST 4: Credit Assignment Integration
# =============================================================================

def test_4_credit_assignment_integration(memory):
    """
    Test that credit assignment is integrated and working.
    """
    print("\n=== Test 4: Credit Assignment Integration ===")
    
    # Train on some patterns
    for i in range(5):
        context = [i, i+1]
        target = i * 10
        memory.learn_adaptive(context, target)
    
    # Now create errors by training with different targets
    errors_created = 0
    for i in range(5):
        context = [i, i+1]
        wrong_target = 99 - i  # Different from original
        stats = memory.learn_adaptive(context, wrong_target)
        
        if not stats['correct']:
            errors_created += 1
    
    # Check credit tracker
    credit_stats = memory.credit_tracker.get_statistics()
    print(f"Credit stats: {credit_stats}")
    
    assert credit_stats['total_errors'] > 0, "Should have recorded errors"
    print(f"Errors created: {errors_created}, recorded: {credit_stats['total_errors']}")
    
    # Force reconsolidation
    recon_stats = memory.credit_tracker.reconsolidate_all()
    print(f"Reconsolidation: {recon_stats}")
    
    print("✓ Credit assignment integration works")


# =============================================================================
# TEST 5: Statistics Tracking
# =============================================================================

def test_5_statistics_tracking(memory):
    """
    Test that statistics are properly tracked.
    """
    print("\n=== Test 5: Statistics Tracking ===")
    
    # Train on patterns
    np.random.seed(42)
    for i in range(20):
        context = list(np.random.randint(0, 100, size=3))
        target = np.random.randint(0, 100)
        memory.learn_adaptive(context, target)
    
    stats = memory.get_statistics()
    print(f"Statistics: {stats}")
    
    # Check required fields
    assert 'current_step' in stats
    assert 'accuracy' in stats
    assert 'meta_learning' in stats
    assert 'credit_assignment' in stats
    
    assert stats['current_step'] == 20
    assert 0 <= stats['accuracy'] <= 1
    
    print(f"Step: {stats['current_step']}")
    print(f"Accuracy: {stats['accuracy']:.1%}")
    print(f"Unique contexts: {stats['unique_contexts']}")
    
    print("✓ Statistics tracking works")


# =============================================================================
# TEST 6: Batch Learning
# =============================================================================

def test_6_batch_learning(memory):
    """
    Test batch learning functionality.
    """
    print("\n=== Test 6: Batch Learning ===")
    
    # Create batch
    batch = [
        ([1, 2, 3], 10),
        ([4, 5, 6], 20),
        ([7, 8, 9], 30),
        ([10, 11, 12], 40),
    ]
    
    stats = memory.learn_batch(batch)
    print(f"Batch stats: {stats}")
    
    assert stats['n_samples'] == 4
    assert 'accuracy' in stats
    assert 'avg_rate' in stats
    
    # Verify all patterns learned
    for ctx, tgt in batch:
        retrieved, _ = memory.retrieve(ctx)
        assert retrieved == tgt, f"Should retrieve {tgt} for {ctx}"
    
    print("✓ Batch learning works")


# =============================================================================
# TEST 7: Generation
# =============================================================================

def test_7_generation(memory):
    """
    Test text generation capability.
    """
    print("\n=== Test 7: Generation ===")
    
    # Train on some sequences
    sequences = [
        [1, 2, 3, 4, 5],
        [10, 11, 12, 13, 14],
        [20, 21, 22, 23, 24],
    ]
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            context = seq[max(0, i-2):i+1]
            target = seq[i+1]
            memory.learn_adaptive(context, target)
    
    # Generate from prompt
    prompt = [1, 2, 3]
    generated, gen_stats = memory.generate(prompt, max_tokens=5)
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    print(f"Stats: {gen_stats}")
    
    # Should generate some tokens
    assert len(generated) > 0, "Should generate at least one token"
    
    print("✓ Generation works")


# =============================================================================
# TEST 8: Factory Function
# =============================================================================

def test_8_factory_function():
    """
    Test the convenience factory function.
    """
    print("\n=== Test 8: Factory Function ===")
    
    # Test with different configurations
    mem1 = create_adaptive_memory(
        vocab_size=500,
        use_adaptive_rates=True,
        use_curriculum=True,
        total_steps=5000,
    )
    
    mem2 = create_adaptive_memory(
        vocab_size=500,
        use_adaptive_rates=False,
        use_curriculum=False,
    )
    
    # Both should work
    mem1.learn_adaptive([1, 2], 10)
    mem2.learn_adaptive([1, 2], 10)
    
    r1, _ = mem1.retrieve([1, 2])
    r2, _ = mem2.retrieve([1, 2])
    
    assert r1 == 10
    assert r2 == 10
    
    # mem1 should have adaptive rates, mem2 should have fixed
    stats1 = mem1.learn_adaptive([3, 4], 20)
    stats2 = mem2.learn_adaptive([3, 4], 20)
    
    print(f"Adaptive memory rate: {stats1['rate_used']:.4f}")
    print(f"Fixed memory rate: {stats2['rate_used']:.4f}")
    
    # Fixed rate should be exactly φ⁻¹
    assert stats2['rate_used'] == PHI_INV, (
        f"Fixed rate should be φ⁻¹ ({PHI_INV:.4f}), got {stats2['rate_used']:.4f}"
    )
    
    print("✓ Factory function works")


# =============================================================================
# TEST 9: Full Training Loop
# =============================================================================

def test_9_full_training_loop():
    """
    Test a realistic training loop scenario.
    """
    print("\n=== Test 9: Full Training Loop ===")
    
    np.random.seed(42)
    
    # Create memory with curriculum
    memory = create_adaptive_memory(
        vocab_size=100,
        use_adaptive_rates=True,
        use_curriculum=True,
        total_steps=100,
        seed=42,
    )
    
    # Generate training data
    data = []
    for i in range(100):
        context = list(np.random.randint(0, 100, size=3))
        target = (context[0] + context[1] + context[2]) % 100  # Deterministic pattern
        data.append((context, target))
    
    # Training loop
    accuracies = []
    for epoch in range(2):
        epoch_correct = 0
        for context, target in data:
            stats = memory.learn_adaptive(context, target)
            if stats['correct']:
                epoch_correct += 1
        
        epoch_accuracy = epoch_correct / len(data)
        accuracies.append(epoch_accuracy)
        print(f"Epoch {epoch + 1}: accuracy={epoch_accuracy:.1%}")
    
    # Final stats
    final_stats = memory.get_statistics()
    print(f"Final accuracy: {final_stats['accuracy']:.1%}")
    print(f"Meta state: {final_stats['meta_learning']}")
    
    # Second epoch should be better than first (learning works)
    assert accuracies[1] >= accuracies[0], (
        f"Learning should improve: {accuracies[0]:.1%} -> {accuracies[1]:.1%}"
    )
    
    print("✓ Full training loop works")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    memory = create_adaptive_memory(vocab_size=100, seed=42)
    
    test_1_basic_learning_retrieval(memory)
    
    memory2 = create_adaptive_memory(vocab_size=100, seed=42)
    test_2_adaptive_rate_modulation(memory2)
    
    memory3 = create_adaptive_memory(vocab_size=100, seed=42)
    test_3_novelty_detection(memory3)
    
    memory4 = create_adaptive_memory(vocab_size=100, seed=42)
    test_4_credit_assignment_integration(memory4)
    
    memory5 = create_adaptive_memory(vocab_size=100, seed=42)
    test_5_statistics_tracking(memory5)
    
    memory6 = create_adaptive_memory(vocab_size=100, seed=42)
    test_6_batch_learning(memory6)
    
    memory7 = create_adaptive_memory(vocab_size=100, seed=42)
    test_7_generation(memory7)
    
    test_8_factory_function()
    test_9_full_training_loop()
    
    print("\n" + "="*60)
    print("ALL ADAPTIVE MEMORY TESTS PASSED")
    print("="*60)
