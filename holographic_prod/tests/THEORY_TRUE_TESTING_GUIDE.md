# Theory-True Testing Guide

## Overview

This guide explains how to test the holographic memory system for theory-true correctness, optimal performance, and proper learning behavior.

## Core Principles

### 1. Theory-True Evaluation

All evaluation must match the exact `retrieve()` path:

```python
from holographic_prod.tests.theory_true_evaluation_helper import (
    evaluate_semantic_similarity_theory_true
)

# This matches MultiLevelTower.retrieve() exactly:
# 1. Grace contraction on context → graced_state
# 2. Route to satellite via Grace basin key
# 3. Memory unbinding: ctx_inv @ sat_memory → retrieved
# 4. Grace contraction on retrieved → retrieved_graced
# 5. Full vocabulary coherence scoring
# 6. Return coherence score of target token
```

### 2. Metrics

**Primary Metrics:**
- **Semantic Similarity**: Coherence score of target token (witness_energy / total_energy)
- **Exact Match Rate**: Fraction where `retrieve()` returns target token
- **Target Rank**: Average rank of target in coherence scores (lower is better)

**Learning Metrics:**
- **Semantic Similarity Curve**: Should increase over batches
- **Witness Churn**: Should decrease (stabilize) over time
- **Grade Energy Evolution**: Should shift from bivector → scalar/pseudoscalar
- **Satellite Occupancy**: Should become Zipfian (some very active, most sparse)

**Performance Metrics:**
- **Throughput**: Samples/sec for learning
- **Retrieval Latency**: P50, P99 latency for retrieval
- **GPU Memory**: Memory usage per batch

### 3. Theory-True Requirements

**MUST ALWAYS PASS:**
1. **Grace Convergence**: Grace ALWAYS converges (never returns None)
   - Test: `verify_grace_convergence()` → convergence_rate == 1.0
   - Stability should be >= φ⁻² (0.382)

2. **Full Vocabulary**: No candidate sets (FORBIDDEN per THEORY_TRUE_PARADIGM.md)
   - Test: `verify_no_candidate_sets()` → diverse tokens returned
   - Should return many different tokens, not just a few

3. **Coherence Scoring**: Uses witness_energy / total_energy (not similarity)
   - Test: `evaluate_semantic_similarity_theory_true()` → coherence > 0
   - Should match exact retrieve() implementation

## Test Suite

### Test 1: Theory-True Correctness
**File**: `test_theory_true_comprehensive.py::test_theory_true_correctness`

Tests:
- Grace convergence (100% success rate)
- Full vocabulary retrieval (no candidate sets)
- Coherence scoring matches theory

**Run**: `modal run holographic_prod/tests/test_theory_true_comprehensive.py::test_theory_true_correctness`

### Test 2: Performance
**File**: `test_theory_true_comprehensive.py::test_performance`

Measures:
- Learning throughput (should be > 1000 samples/sec)
- Retrieval latency (should be < 100ms)
- GPU memory usage

**Run**: `modal run holographic_prod/tests/test_theory_true_comprehensive.py::test_performance`

### Test 3: Learning Verification
**File**: `test_theory_true_comprehensive.py::test_learning_verification`

Verifies:
- Semantic similarity increases over batches
- Witness churn decreases (stabilizes)
- Grade energy shifts toward scalar/pseudoscalar
- Satellite occupancy becomes Zipfian

**Run**: `modal run holographic_prod/tests/test_theory_true_comprehensive.py::test_learning_verification`

### Baseline Characterization
**File**: `test_baseline_characterization.py`

Establishes baseline metrics for comparison. Uses shared evaluation helper.

**Run**: `modal run holographic_prod/tests/test_baseline_characterization.py`

## Usage Examples

### Quick Check: Theory-True Correctness
```python
from holographic_prod.tests.theory_true_evaluation_helper import (
    verify_grace_convergence,
    verify_no_candidate_sets,
)

# Test Grace convergence
result = verify_grace_convergence(model, test_contexts)
assert result['convergence_rate'] == 1.0

# Test full vocabulary
result = verify_no_candidate_sets(model, test_contexts)
assert result['unique_tokens_returned'] > 10
```

### Evaluate Learning Progress
```python
from holographic_prod.tests.theory_true_evaluation_helper import (
    evaluate_semantic_similarity_theory_true,
)

result = evaluate_semantic_similarity_theory_true(
    model, eval_samples, n_eval=100
)

print(f"Semantic similarity: {result['semantic_similarity']:.4f}")
print(f"Exact match rate: {result['exact_match_rate']:.1%}")
print(f"Avg target rank: {result['avg_target_rank']:.1f}")
```

### Check Retrieval Accuracy
```python
from holographic_prod.tests.theory_true_evaluation_helper import (
    evaluate_retrieval_accuracy,
)

result = evaluate_retrieval_accuracy(model, eval_samples)
print(f"Accuracy: {result['accuracy']:.1%}")
```

## Common Issues

### Issue: Semantic Similarity is 0.0

**Cause**: Evaluation samples are empty or evaluation path doesn't match retrieve()

**Fix**: 
1. Check `eval_samples` is not empty
2. Use `evaluate_semantic_similarity_theory_true()` helper (matches retrieve() exactly)
3. Verify samples are beyond training set

### Issue: Grace Doesn't Converge

**Cause**: Theory violation - Grace MUST always converge

**Fix**: Check Grace implementation, ensure stability >= φ⁻²

### Issue: Only Few Tokens Returned

**Cause**: Candidate sets being used (FORBIDDEN)

**Fix**: Verify `retrieve()` uses full vocabulary coherence scoring, not candidate sets

### Issue: Learning Not Improving

**Cause**: Evaluation doesn't match training, or samples overlap

**Fix**:
1. Use holdout evaluation set (not in training)
2. Verify semantic similarity increases over batches
3. Check witness churn decreases (stabilizes)

## Performance Targets

- **Learning Throughput**: > 1,000 samples/sec (H100 GPU)
- **Retrieval Latency**: < 100ms (P50), < 200ms (P99)
- **GPU Memory**: < 10GB per 1M patterns
- **Semantic Similarity**: Should increase from ~0.1 → ~0.5+ over 50 batches

## Theory References

- **THEORY_TRUE_PARADIGM.md**: Core theory-true principles
- **TESTING_PRINCIPLES.md**: Testing best practices
- **MultiLevelTower.retrieve()**: Reference implementation
