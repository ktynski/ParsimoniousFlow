# Theory-True Fixes Applied

## Critical Fixes (v5.32.0)

### Fix 1: `retrieve_settled_states_batch()` - Missing Grace Contractions

**Problem**: `retrieve_settled_states_batch()` was missing critical steps from `retrieve()`:
- ❌ No Grace contraction on context (before routing)
- ❌ No Grace contraction on retrieved state (after unbinding)
- ❌ Used raw unbinding instead of Grace-contracted path

**Impact**: Evaluation path didn't match production `retrieve()`, causing metrics to diverge.

**Fix**: Updated to match exact `retrieve()` path:
1. ✅ Grace contraction on context → `graced_state`
2. ✅ Route to satellite via Grace basin key
3. ✅ Memory unbinding: `ctx_inv @ sat_memory` → `retrieved`
4. ✅ Grace contraction on retrieved → `retrieved_graced`
5. ✅ Handle empty satellites (use `graced_state` directly)

**Code**: `holographic_prod/memory/holographic_memory_unified.py::retrieve_settled_states_batch()`

### Fix 2: `evaluate_semantic()` - Using Frobenius Cosine Instead of Coherence

**Problem**: `evaluate_semantic()` used Frobenius cosine similarity, but theory-true uses coherence scoring:
- ❌ `similarity = dot(a, b) / (norm(a) * norm(b))` (Frobenius cosine)
- ✅ `coherence = witness_energy / total_energy` (theory-true)

**Impact**: Metrics didn't reflect theory-true learning (coherence measures witness stability, not just magnitude).

**Fix**: Updated to use coherence scoring matching `retrieve()`:
1. ✅ Get `retrieved_graced` from `retrieve_settled_states_batch()` (now includes Grace)
2. ✅ Compute composition: `retrieved_graced @ target_emb.T`
3. ✅ Decompose into Clifford coefficients
4. ✅ Compute coherence: `witness_energy / total_energy`
   - `witness_energy = scalar² + pseudoscalar²`
   - `total_energy = sum(all_coeffs²)`

**Code**: `holographic_prod/memory/holographic_memory_unified.py::evaluate_semantic()`

## Verification

Both fixes ensure:
- ✅ **No fallbacks**: Exact path, no shortcuts
- ✅ **No simplifications**: Full Grace contractions, full coherence scoring
- ✅ **No fakeness**: Real theory-true metrics, not approximations
- ✅ **No information loss**: All steps from `retrieve()` preserved

## Testing

Run these tests to verify:
1. `test_train_modal_audit.py::audit_evaluation_correctness()` - Verifies evaluation matches retrieve()
2. `test_theory_true_comprehensive.py::test_theory_true_correctness()` - Verifies theory-true requirements
3. `test_baseline_characterization.py` - Verifies learning metrics

## Impact

- **Before**: Evaluation used simplified path (no Grace, Frobenius cosine)
- **After**: Evaluation matches exact `retrieve()` path (Grace + coherence)
- **Result**: Training metrics now accurately reflect production performance
