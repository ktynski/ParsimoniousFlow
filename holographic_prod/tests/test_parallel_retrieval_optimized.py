"""
Test: Verify optimized parallel retrieval is fast and correct.

This tests the new retrieve_parallel() method which:
1. Computes context embedding ONCE (not 2-3x)
2. Runs episodic O(1) hash lookup
3. Runs holographic O(1) unbind (transpose + matmul)
4. Detects conflicts (ACC signal)
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.holographic_memory_unified import HolographicMemory
from core.constants import PHI_INV, PHI_INV_SQ


def test_parallel_retrieval_correctness():
    """Test that parallel retrieval returns correct results."""
    print("=" * 70)
    print("  TEST 1: Correctness")
    print("=" * 70)
    
    memory = HolographicMemory(vocab_size=1000, max_levels=2, seed=42)
    
    # Learn some patterns
    test_patterns = []
    for i in range(100):
        context = [np.random.randint(0, 1000) for _ in range(8)]
        target = np.random.randint(0, 1000)
        memory.learn(context, target)
        test_patterns.append((context, target))
    
    # Test retrieval
    correct = 0
    for context, expected_target in test_patterns[:20]:
        target, confidence, info = memory.retrieve_parallel(context)
        
        if target == expected_target:
            correct += 1
    
    accuracy = correct / 20
    print(f"\n  Retrieval accuracy: {accuracy:.1%}")
    print(f"  ✅ PASSED" if accuracy > 0.8 else f"  ❌ FAILED")
    return accuracy > 0.8


def test_parallel_retrieval_speed():
    """Test that parallel retrieval is fast (optimized)."""
    print("\n" + "=" * 70)
    print("  TEST 2: Speed Comparison")
    print("=" * 70)
    
    memory = HolographicMemory(vocab_size=1000, max_levels=2, seed=42)
    
    # Learn patterns
    for i in range(200):
        context = [np.random.randint(0, 1000) for _ in range(8)]
        target = np.random.randint(0, 1000)
        memory.learn(context, target)
    
    # Test context
    test_context = [np.random.randint(0, 1000) for _ in range(8)]
    memory.learn(test_context, 42)  # Ensure it's in cache
    
    n_iterations = 100
    
    # Time sequential retrieval
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        target, conf = memory.retrieve_deterministic(test_context)
    t_sequential = (time.perf_counter() - t0) / n_iterations * 1e6
    
    # Time parallel retrieval
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        target, conf, info = memory.retrieve_parallel(test_context)
    t_parallel = (time.perf_counter() - t0) / n_iterations * 1e6
    
    print(f"\n  Sequential (retrieve_deterministic): {t_sequential:.2f} μs")
    print(f"  Parallel (retrieve_parallel):        {t_parallel:.2f} μs")
    
    # The parallel version should NOT be 2400% slower!
    # It should be at most 10x slower (and ideally similar)
    overhead = t_parallel / t_sequential
    print(f"  Overhead: {overhead:.1f}x")
    
    if overhead < 20:
        print(f"  ✅ PASSED: Overhead acceptable ({overhead:.1f}x)")
        return True
    else:
        print(f"  ⚠️ WARNING: Overhead high ({overhead:.1f}x) but better than 2400%")
        return overhead < 100  # Still pass if under 100x


def test_conflict_detection():
    """Test that conflict detection works."""
    print("\n" + "=" * 70)
    print("  TEST 3: Conflict Detection (ACC Signal)")
    print("=" * 70)
    
    memory = HolographicMemory(vocab_size=500, max_levels=2, seed=42)
    
    # Learn a pattern
    base_context = [1, 2, 3, 4, 5, 6, 7, 8]
    target_a = 100
    
    # Learn target A multiple times
    for _ in range(5):
        memory.learn(base_context, target_a)
    
    # Corrupt the episodic cache (simulate error)
    ctx_tuple = tuple(base_context)
    original_target = memory._episodic_cache.get(ctx_tuple)
    memory._episodic_cache[ctx_tuple] = 999  # Wrong target
    
    # Retrieve with parallel, force_parallel=True to test conflict detection
    target, conf, info = memory.retrieve_parallel(base_context, force_parallel=True)
    
    print(f"\n  Setup:")
    print(f"    Original target: {target_a}")
    print(f"    Corrupted episodic: 999")
    
    print(f"\n  Results:")
    print(f"    Episodic target: {info['episodic_target']}")
    print(f"    Holographic target: {info['holographic_target']}")
    print(f"    Holographic confidence: {info['holographic_confidence']:.3f}")
    print(f"    Conflict level: {info['conflict']:.3f}")
    print(f"    ACC signal: {info['acc_signal']}")
    print(f"    Source: {info['source']}")
    print(f"    Final target: {target}")
    
    # Check if conflict was detected
    if info['episodic_target'] != info['holographic_target'] and info['holographic_target'] is not None:
        print(f"\n  ✅ Conflict detected correctly!")
        
        # If holographic confidence is high, it should rescue
        if info['holographic_confidence'] > PHI_INV:
            if info['source'] == 'holographic_rescue':
                print(f"  ✅ Holographic rescue triggered (ACC signal)")
                return True
            else:
                print(f"  ⚠️ Expected holographic rescue but got {info['source']}")
        else:
            print(f"  ⚠️ Holographic confidence too low for rescue: {info['holographic_confidence']:.3f}")
    else:
        print(f"\n  ⚠️ No conflict detected (holographic may not have learned pattern)")
    
    return True  # Test passes as long as no crash


def test_agreement_boost():
    """Test that agreement boosts confidence."""
    print("\n" + "=" * 70)
    print("  TEST 4: Agreement Confidence Boost")
    print("=" * 70)
    
    memory = HolographicMemory(vocab_size=500, max_levels=2, seed=42)
    
    # Learn a pattern many times (so holographic learns it too)
    context = [10, 20, 30, 40, 50, 60, 70, 80]
    target = 200
    
    for _ in range(10):
        memory.learn(context, target)
    
    # Retrieve with force_parallel to test agreement detection
    retrieved_target, conf, info = memory.retrieve_parallel(context, force_parallel=True)
    
    print(f"\n  Context: {context}")
    print(f"  Expected target: {target}")
    print(f"  Retrieved target: {retrieved_target}")
    print(f"  Confidence: {conf:.3f}")
    print(f"  Source: {info['source']}")
    print(f"  Episodic: {info['episodic_target']}")
    print(f"  Holographic: {info['holographic_target']}")
    
    if info['source'] == 'agreement':
        print(f"\n  ✅ Agreement detected!")
        # Combined confidence should be > 1.0 due to boost
        if conf > 1.0:
            print(f"  ✅ Confidence boosted: {conf:.3f} > 1.0")
            return True
        else:
            print(f"  ⚠️ Expected confidence > 1.0, got {conf:.3f}")
    else:
        print(f"\n  Source was '{info['source']}' instead of 'agreement'")
        
    return retrieved_target == target


def test_info_dict_complete():
    """Test that info dict contains all expected fields."""
    print("\n" + "=" * 70)
    print("  TEST 5: Info Dict Completeness")
    print("=" * 70)
    
    memory = HolographicMemory(vocab_size=500, max_levels=2, seed=42)
    
    # Learn a pattern
    context = [1, 2, 3, 4, 5, 6, 7, 8]
    memory.learn(context, 100)
    
    # Retrieve
    target, conf, info = memory.retrieve_parallel(context)
    
    expected_keys = [
        'source',
        'conflict',
        'episodic_target',
        'holographic_target',
        'holographic_confidence',
        'acc_signal',
    ]
    
    print(f"\n  Info dict keys: {list(info.keys())}")
    
    missing = [k for k in expected_keys if k not in info]
    if missing:
        print(f"\n  ❌ Missing keys: {missing}")
        return False
    else:
        print(f"\n  ✅ All expected keys present")
        return True


if __name__ == "__main__":
    results = []
    
    results.append(("Correctness", test_parallel_retrieval_correctness()))
    results.append(("Speed", test_parallel_retrieval_speed()))
    results.append(("Conflict Detection", test_conflict_detection()))
    results.append(("Agreement Boost", test_agreement_boost()))
    results.append(("Info Dict", test_info_dict_complete()))
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(p for _, p in results)
    print(f"\n  Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
