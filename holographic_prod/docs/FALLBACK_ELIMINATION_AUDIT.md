# Fallback Elimination Audit

## Date: 2026-01-18

---

## Summary

Ruthlessly hunted down and eliminated silent failure patterns and "fallback" terminology that was hiding bugs and causing confusion.

---

## Silent Failures Fixed (25+ instances)

### Pattern: `except: pass` → FAIL LOUDLY

| File | Line | Fix |
|------|------|-----|
| `train_modal.py` | 645 | Removed try/except entirely - xp MUST exist |
| `theory_true_evaluation_helper.py` | 134 | Raise RuntimeError with full traceback |
| `theory_true_evaluation_helper.py` | 190 | Remove try/except - retrieval MUST NOT fail |
| `theory_true_evaluation_helper.py` | 239 | Remove try/except - Grace MUST converge |
| `theory_true_evaluation_helper.py` | 270 | Remove try/except - retrieve MUST NOT fail |
| `test_complete_flow_granular.py` | 844 | Remove try/except |
| `test_baseline_characterization.py` | 111 | Specific FileNotFoundError |
| `test_overdreaming_collapse.py` | 406 | Raise RuntimeError |
| `test_witness_stability.py` | 239 | Raise RuntimeError |
| `test_genius_configuration.py` | 383 | Raise RuntimeError |
| `test_parallel_proposals.py` | 252 | Raise RuntimeError |
| `test_sandboxed_simulation.py` | 275 | Raise RuntimeError |
| `test_dmn_interleaved.py` | 253 | Raise RuntimeError |
| `test_adaptive_dmn_gate.py` | 257, 438 | Raise RuntimeError |
| `test_coherence_reward.py` | 181, 241, 314 | Raise RuntimeError |
| `test_satellite_holonomy.py` | 112, 396 | FileNotFoundError, RuntimeError |
| `test_conflict_curiosity.py` | 249, 377 | Raise RuntimeError |
| `test_adaptive_dmn_gate_v2.py` | 554, 583 | Raise RuntimeError |
| `test_delayed_collapse.py` | 236 | Remove try/except entirely |
| `test_novelty_weighted_learning.py` | 478 | Raise RuntimeError |
| `attractor_generation.py` | 379 | Remove try/except, use explicit type check |
| `quaternion.py` | 320 | Specific ValueError, LinAlgError |
| `speed_benchmarks.py` | 306 | Specific ImportError |
| `few_shot_benchmarks.py` | 579, 587 | Specific ValueError |

### Pattern: `return 0.0 if not X` → Raise Error

Changed all patterns like:
```python
# OLD (hides failures):
return np.mean(similarities) if similarities else 0.0

# NEW (fails loudly):
if not similarities:
    raise RuntimeError("No similarities computed - all samples failed")
return np.mean(similarities)
```

---

## "Fallback" Terminology Eliminated (35+ instances)

### Core Files

| File | Change |
|------|--------|
| `holographic_memory_unified.py` | "fallback" → "complementary path" |
| `holographic_memory_unified.py` | "used_schema_fallback" → "used_direct_state" |
| `holographic_memory_unified.py` | "_prior_fallback_count" → "_prior_usage_count" |
| `train_modal.py` | "prior_fallbacks" → "prior_usage" |
| `train_modal.py` | "no CPU fallback" → "GPU mandatory" |
| `resonance.py` | "gate fallback" → "gate prior blending" |
| `resonance.py` | "Global fallback" → "Global prior" |
| `distributed_prior.py` | "NO fallback" → "NO coverage" |
| `distributed_prior.py` | "global fallback" → "global prior path" |
| `grounded_embeddings.py` | "fallback_to_random" → "allow_random" |
| `integration.py` | Added WARNING about sequential pattern |

### Rationale

The term "fallback" implies:
1. Sequential execution (try A, if fails try B)
2. B is inferior to A
3. Failure is expected and acceptable

But our theory says:
1. PARALLEL execution (A and B run simultaneously)
2. Both paths contribute based on confidence
3. Agreement BOOSTS confidence (synergy)

---

## Remaining "fallback" Instances (28)

These are in tests/benchmarks and are lower priority:
- Most are in test descriptions or comments
- Some are in benchmark comparison code
- Can be cleaned up in a follow-up pass

---

## Impact

### Before
- Silent failures hid bugs for weeks
- "Fallback" terminology caused confusion about parallel vs sequential
- Tests passed even when core functionality was broken

### After
- Failures are LOUD and immediate
- Terminology matches theory (parallel, complementary)
- Tests will fail fast if something breaks

---

## Verification

Run any test - if it fails, you'll see:
```
RuntimeError: Theory violation: retrieve() returned None for context [1, 2, 3]...
```

Instead of silently returning 0.0 and appearing to pass.
