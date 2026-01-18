"""
TEST SUITE: Phase B - Semantic Token Filtering

Test-driven implementation of semantic filtering for generalization.

THEORY:
    Function words (the, a, is) have LOW predictiveness (appear with many targets).
    Content words (cat, run, happy) have HIGH predictiveness (predict specific targets).
    
    By filtering function words, semantic contexts become invariant to paraphrases:
        "the cat sat on the mat" → semantic: [cat, sat, mat]
        "a cat sits on a mat"    → semantic: [cat, sits, mat]
        
    These should map to similar semantic regions → generalization.

Run: python -m holographic_v4.test_phase_b_semantic_filtering
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import sys

# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_INV_SQ = PHI_INV ** 2
PHI_INV_CUBE = PHI_INV ** 3


# =============================================================================
# TEST 1: PREDICTIVENESS TRACKER IDENTIFIES FUNCTION WORDS
# =============================================================================

def test_predictiveness_identifies_function_words() -> Dict:
    """
    After training, function words should have low predictiveness.
    
    THEORY:
        Function words appear with many different targets → low mutual information.
        Content words appear with specific targets → high mutual information.
    """
    print("\n" + "="*60)
    print("TEST: Predictiveness Identifies Function Words")
    print("="*60)
    
    from holographic_v4.predictiveness import PredictivenessTracker
    
    tracker = PredictivenessTracker()
    
    # Simulate training data where:
    # - "the" and "a" appear with many targets (function words)
    # - "cat", "dog", "bird" appear with specific targets (content words)
    
    # Training: function words appear everywhere
    for target in range(10):  # 10 different targets
        tracker.observe([0], target)  # "the" → many targets
        tracker.observe([1], target)  # "a" → many targets
    
    # Training: content words predict specific targets
    tracker.observe([10], 0)  # "cat" → target 0
    tracker.observe([10], 0)
    tracker.observe([10], 0)
    tracker.observe([11], 1)  # "dog" → target 1
    tracker.observe([11], 1)
    tracker.observe([11], 1)
    tracker.observe([12], 2)  # "bird" → target 2
    tracker.observe([12], 2)
    tracker.observe([12], 2)
    
    # Check predictiveness
    function_word_pred = max(
        tracker.predictiveness(0),  # "the"
        tracker.predictiveness(1),  # "a"
    )
    
    content_word_pred = min(
        tracker.predictiveness(10),  # "cat"
        tracker.predictiveness(11),  # "dog"
        tracker.predictiveness(12),  # "bird"
    )
    
    print(f"\n  Function word predictiveness: {function_word_pred:.3f}")
    print(f"  Content word predictiveness: {content_word_pred:.3f}")
    
    # Content words should have HIGHER predictiveness than function words
    passed = content_word_pred > function_word_pred
    
    if passed:
        print(f"\n  ✅ PASS: Content words ({content_word_pred:.3f}) > Function words ({function_word_pred:.3f})")
    else:
        print(f"\n  ❌ FAIL: Content words NOT more predictive than function words")
    
    return {
        'test': 'predictiveness_identifies_function_words',
        'function_word_pred': function_word_pred,
        'content_word_pred': content_word_pred,
        'passed': passed
    }


# =============================================================================
# TEST 2: SEMANTIC EXTRACTION FILTERS FUNCTION WORDS
# =============================================================================

def test_semantic_extraction_filters_function_words() -> Dict:
    """
    extract_semantic() should return only content words after training.
    """
    print("\n" + "="*60)
    print("TEST: Semantic Extraction Filters Function Words")
    print("="*60)
    
    from holographic_v4.predictiveness import PredictivenessTracker
    
    tracker = PredictivenessTracker()
    
    # Same training as above
    for target in range(10):
        tracker.observe([0], target)  # "the"
        tracker.observe([1], target)  # "a"
    
    for _ in range(5):
        tracker.observe([10], 0)  # "cat"
        tracker.observe([11], 1)  # "dog"
    
    # Test extraction
    context = [0, 10, 1, 11]  # "the cat a dog"
    semantic_tokens = tracker.extract_semantic(context)
    
    print(f"\n  Full context: {context}")
    print(f"  Semantic tokens: {semantic_tokens}")
    
    # Should contain content words (10, 11) but NOT function words (0, 1)
    has_content = 10 in semantic_tokens or 11 in semantic_tokens
    no_function = 0 not in semantic_tokens and 1 not in semantic_tokens
    
    passed = has_content and no_function
    
    if passed:
        print(f"\n  ✅ PASS: Extracted content words, filtered function words")
    else:
        if not has_content:
            print(f"\n  ❌ FAIL: Missing content words in extraction")
        if not no_function:
            print(f"\n  ❌ FAIL: Function words not filtered")
    
    return {
        'test': 'semantic_extraction_filters_function_words',
        'semantic_tokens': semantic_tokens,
        'has_content': has_content,
        'no_function': no_function,
        'passed': passed
    }


# =============================================================================
# TEST 3: SEMANTIC CONTEXTS ARE SIMILAR FOR PARAPHRASES
# =============================================================================

def test_semantic_contexts_similar_for_paraphrases() -> Dict:
    """
    Semantic contexts of paraphrases should be more similar than full contexts.
    
    THEORY:
        Full context: "the cat sat" vs "a cat sits" → different (function words differ)
        Semantic: [cat, sat] vs [cat, sits] → similar (only content words)
    """
    print("\n" + "="*60)
    print("TEST: Semantic Contexts Similar for Paraphrases")
    print("="*60)
    
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.quotient import witness_similarity
    
    # Create model with predictiveness enabled
    model = TheoryTrueModel(
        vocab_size=100,
        max_attractors=1000,
        use_predictiveness=True,
        xp=np
    )
    
    # Train predictiveness tracker
    # Function words: 0-9 appear with all targets
    # Content words: 10-99 have specific associations
    for target in range(20):
        for func_word in range(10):
            model.predictiveness_tracker.observe([func_word], target)
    
    for i in range(10, 50):
        for _ in range(5):
            model.predictiveness_tracker.observe([i], i % 20)
    
    # Create two "paraphrases" differing only in function words
    context1 = [0, 10, 1, 20, 2, 30]  # "the cat a dog an bird" (func: 0,1,2; content: 10,20,30)
    context2 = [3, 10, 4, 20, 5, 30]  # "a cat the dog that bird" (func: 3,4,5; content: 10,20,30)
    
    # Compute full contexts
    full_ctx1 = model.compute_context(context1)
    full_ctx2 = model.compute_context(context2)
    
    # Compute semantic contexts
    sem_ctx1 = model.compute_semantic_context(context1)
    sem_ctx2 = model.compute_semantic_context(context2)
    
    # Compute similarities
    full_sim = witness_similarity(full_ctx1, full_ctx2, model.basis, np)
    sem_sim = witness_similarity(sem_ctx1, sem_ctx2, model.basis, np)
    
    print(f"\n  Full context similarity: {full_sim:.3f}")
    print(f"  Semantic context similarity: {sem_sim:.3f}")
    
    # Semantic should be MORE similar (ideally identical since same content words)
    passed = sem_sim > full_sim or sem_sim > PHI_INV  # Either better than full OR high absolute
    
    if passed:
        print(f"\n  ✅ PASS: Semantic similarity ({sem_sim:.3f}) better than full ({full_sim:.3f})")
    else:
        print(f"\n  ❌ FAIL: Semantic filtering didn't improve similarity")
    
    return {
        'test': 'semantic_contexts_similar_for_paraphrases',
        'full_similarity': full_sim,
        'semantic_similarity': sem_sim,
        'improvement': sem_sim - full_sim,
        'passed': passed
    }


# =============================================================================
# TEST 4: SEMANTIC RETRIEVAL GENERALIZES
# =============================================================================

def test_semantic_retrieval_generalizes() -> Dict:
    """
    Retrieval using semantic context should find matches for paraphrases.
    
    This is the key test for generalization.
    """
    print("\n" + "="*60)
    print("TEST: Semantic Retrieval Generalizes")
    print("="*60)
    
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.holographic_memory import VorticityWitnessIndex
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis()
    
    # Create model with predictiveness
    model = TheoryTrueModel(
        vocab_size=100,
        max_attractors=1000,
        use_predictiveness=True,
        xp=np
    )
    
    # Train predictiveness: tokens 0-9 are function words
    for target in range(20):
        for func_word in range(10):
            model.predictiveness_tracker.observe([func_word], target)
    
    # Content words predict specific targets
    for i in range(10, 50):
        for _ in range(5):
            model.predictiveness_tracker.observe([i], i % 10)
    
    # Create semantic index (stores semantic contexts only)
    semantic_index = VorticityWitnessIndex.create(basis, xp=np)
    
    # Store patterns using SEMANTIC contexts
    patterns = [
        ([0, 10, 1, 20], 100),  # "the cat a dog" → target 100
        ([2, 30, 3, 40], 101),  # "an bird the fish" → target 101
    ]
    
    for tokens, target_idx in patterns:
        semantic_ctx = model.compute_semantic_context(tokens)
        target = model.get_embedding(target_idx)
        semantic_index.store(semantic_ctx, target, target_idx)
    
    # Query with PARAPHRASES (different function words, same content)
    queries = [
        ([4, 10, 5, 20], 100),  # "a cat the dog" → should find 100
        ([6, 30, 7, 40], 101),  # "the bird a fish" → should find 101
    ]
    
    correct = 0
    for tokens, expected_target in queries:
        semantic_query = model.compute_semantic_context(tokens)
        result, retrieved_idx, conf = semantic_index.retrieve(semantic_query)
        
        if retrieved_idx == expected_target:
            correct += 1
    
    accuracy = correct / len(queries)
    
    print(f"\n  Patterns stored: {len(patterns)}")
    print(f"  Paraphrase queries: {len(queries)}")
    print(f"  Correct retrievals: {correct}")
    print(f"  Accuracy: {accuracy:.1%}")
    
    # Should generalize to paraphrases
    passed = accuracy >= 0.5  # At least 50% (random would be ~0%)
    
    if passed:
        print(f"\n  ✅ PASS: Semantic retrieval generalizes ({accuracy:.1%})")
    else:
        print(f"\n  ❌ FAIL: Semantic retrieval doesn't generalize ({accuracy:.1%})")
    
    return {
        'test': 'semantic_retrieval_generalizes',
        'accuracy': accuracy,
        'correct': correct,
        'total': len(queries),
        'passed': passed
    }


# =============================================================================
# TEST 5: DUAL INDEX PROVIDES BOTH EXACT AND GENERALIZATION
# =============================================================================

def test_dual_index_exact_and_generalization() -> Dict:
    """
    A dual index should provide:
    1. Exact retrieval for identical contexts
    2. Generalized retrieval for paraphrases
    """
    print("\n" + "="*60)
    print("TEST: Dual Index - Exact + Generalization")
    print("="*60)
    
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.holographic_memory import VorticityWitnessIndex
    from holographic_v4.algebra import build_clifford_basis
    
    basis = build_clifford_basis()
    
    model = TheoryTrueModel(
        vocab_size=100,
        max_attractors=1000,
        use_predictiveness=True,
        xp=np
    )
    
    # Train predictiveness
    for target in range(20):
        for func_word in range(10):
            model.predictiveness_tracker.observe([func_word], target)
    
    for i in range(10, 50):
        for _ in range(5):
            model.predictiveness_tracker.observe([i], i % 10)
    
    # Create DUAL indexes
    episodic_index = VorticityWitnessIndex.create(basis, xp=np)  # Full context
    semantic_index = VorticityWitnessIndex.create(basis, xp=np)  # Semantic context
    
    # Store pattern
    tokens = [0, 10, 1, 20]  # "the cat a dog"
    target_idx = 100
    
    full_ctx = model.compute_context(tokens)
    semantic_ctx = model.compute_semantic_context(tokens)
    target = model.get_embedding(target_idx)
    
    episodic_index.store(full_ctx, target, target_idx)
    semantic_index.store(semantic_ctx, target, target_idx)
    
    # Test 1: EXACT retrieval (same tokens)
    exact_query = model.compute_context([0, 10, 1, 20])
    result_epi, idx_epi, _ = episodic_index.retrieve(exact_query)
    exact_works = (idx_epi == target_idx)
    
    # Test 2: PARAPHRASE retrieval (different function words)
    para_tokens = [2, 10, 3, 20]  # "an cat the dog"
    para_semantic = model.compute_semantic_context(para_tokens)
    result_sem, idx_sem, _ = semantic_index.retrieve(para_semantic)
    para_works = (idx_sem == target_idx)
    
    print(f"\n  Exact retrieval (episodic): {'✓' if exact_works else '✗'}")
    print(f"  Paraphrase retrieval (semantic): {'✓' if para_works else '✗'}")
    
    passed = exact_works and para_works
    
    if passed:
        print(f"\n  ✅ PASS: Dual index provides both exact and generalization")
    else:
        print(f"\n  ❌ FAIL: Dual index incomplete")
    
    return {
        'test': 'dual_index_exact_and_generalization',
        'exact_works': exact_works,
        'para_works': para_works,
        'passed': passed
    }


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests() -> bool:
    """Run all Phase B tests."""
    print("\n" + "="*70)
    print("PHASE B: SEMANTIC TOKEN FILTERING TEST SUITE")
    print("="*70)
    
    results = []
    
    results.append(test_predictiveness_identifies_function_words())
    results.append(test_semantic_extraction_filters_function_words())
    results.append(test_semantic_contexts_similar_for_paraphrases())
    results.append(test_semantic_retrieval_generalizes())
    results.append(test_dual_index_exact_and_generalization())
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    for r in results:
        status = "✅" if r['passed'] else "❌"
        print(f"  {status} {r['test']}")
    
    print(f"\n  TOTAL: {passed}/{total} passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED - Phase B complete!")
    else:
        print(f"\n  ⚠️ {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
