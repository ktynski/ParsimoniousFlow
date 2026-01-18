# Audit: Doc/Code Inconsistencies and Legacy Patterns

## Date: 2026-01-18

---

## Executive Summary

**Major inconsistency found**: The theory documentation claims "PARALLEL" retrieval but key implementations use "SEQUENTIAL FALLBACK" pattern.

This caused confusion and incorrect test design.

---

## Issue 1: PARALLEL vs SEQUENTIAL Mismatch (CRITICAL)

### What CRITICAL_PRINCIPLES.md Says (Line 311):
```
**CRITICAL:** All paths run IN PARALLEL, not sequentially. Winner by CONFIDENCE.

**WHY PARALLEL RETRIEVAL (v5.15.0):**
- The brain runs hippocampus + neocortex SIMULTANEOUSLY (not waterfall!)
- Complementary Learning Systems: fast + slow memory in parallel
- Conflict detection (ACC analog) signals when paths disagree
- Agreement BOOSTS confidence (synergy)
- NO sequential fallback — all paths contribute based on confidence
```

### What `integrate_dreaming_with_model()` Actually Does (integration.py line 330-336):
```python
# SEQUENTIAL with early exit!
if confidence >= PHI_INV_SQ:
    return attractor, int(target_idx), "holographic"  # EXIT HERE

# 2. Try semantic (prototype retrieval via dreaming)  # ONLY IF ABOVE FAILED
```

### What the Docstring Says (integration.py line 269):
```
BRAIN-ANALOG RETRIEVAL HIERARCHY:  # <-- "HIERARCHY" = waterfall!
    1. Episodic (hash lookup)
    2. Semantic (distributed prior) - For unknown contexts
```

### Impact
- Tests designed based on docs expected PARALLEL but got SEQUENTIAL
- The "benefit of dreaming" wasn't measurable because semantic path only ran when holographic failed
- Confusion about whether systems are "fallback" or "complementary"

### Resolution Required
Either:
1. UPDATE CODE to match docs (implement true parallel)
2. UPDATE DOCS to match code (document sequential pattern)

**Recommendation:** Update code to true parallel - the theory is correct.

---

## Issue 2: numpy/cupy Device Mismatch

### The Problem
```python
# DreamingSystem uses numpy (CPU):
dreamer = DreamingSystem(
    basis=basis_np,
    xp=np,  # CPU
)

# Model uses cupy (GPU):
model.xp = cupy
model.basis = cupy array

# integrate_dreaming_with_model() tries to mix them:
witness = extended_witness(proto.prototype_matrix, model.basis, model.xp)
#                          ^-- numpy             ^-- cupy      ^-- cupy
# TypeError: Unsupported type <class 'numpy.ndarray'>
```

### Files Affected
- `holographic_prod/dreaming/integration.py`
- All tests that create DreamingSystem with `xp=np` and HolographicMemory with `use_gpu=True`

### Resolution
Convert prototype matrices to GPU before integration, OR run integration on CPU.

---

## Issue 3: Legacy "fallback" Language

### Count
65 instances of "fallback" across 25 files.

### Examples
- `integration.py`: "Global fallback via factorized prior"
- `holographic_memory_unified.py`: "fallback to schema"
- `resonance.py`: 10 instances of "fallback"

### Impact
Creates mental model of "try A, if fails try B" instead of "A and B run together".

### Resolution
Audit each instance and replace with appropriate parallel terminology:
- "fallback" → "alternative pathway" or "complementary path"
- "if X fails, try Y" → "X and Y contribute based on confidence"

---

## Issue 4: Inconsistent Function Names

| Function | Pattern | Location |
|----------|---------|----------|
| `retrieve_parallel()` | PARALLEL | `holographic_memory_unified.py` |
| `integrate_dreaming_with_model()` | SEQUENTIAL | `integration.py` |
| `retrieve_with_dreaming()` (closure) | SEQUENTIAL | `integration.py` |

### Resolution
Rename or refactor to be consistent:
- `retrieve_parallel()` - keep (correct pattern)
- `integrate_dreaming_with_model()` - refactor to TRUE parallel

---

## Recommended Actions

### Priority 1: Fix `integrate_dreaming_with_model()`
```python
# OLD (sequential):
if confidence >= threshold:
    return holographic_result  # early exit

# NEW (parallel):
holographic_result = ...  # always compute
semantic_result = ...     # always compute
combined = synergy_combine(holographic_result, semantic_result)
return combined
```

### Priority 2: Fix Device Handling
Add device conversion in integration:
```python
# Convert prototypes to model device
if model.xp.__name__ == 'cupy' and proto.xp.__name__ == 'numpy':
    proto_matrix = model.xp.asarray(proto.prototype_matrix)
```

### Priority 3: Terminology Audit
Search and replace "fallback" with appropriate parallel terminology in:
- All docstrings
- All comments
- All documentation

---

## Tests to Verify Fix

1. `test_dreaming_effectiveness.py` - now uses direct parallel eval
2. New test needed: `test_true_parallel_retrieval.py` that verifies BOTH paths always run

---

## Root Cause Analysis

**Why did this happen?**

The codebase evolved over versions (v5.12.0 → v5.15.0 → v5.31.0) and:
1. Documentation was updated to reflect theory (PARALLEL)
2. Some implementations were updated (`retrieve_parallel()`)
3. Other implementations (`integrate_dreaming_with_model()`) kept old pattern
4. No comprehensive audit was done to ensure consistency

**Prevention:**
- Version tags in docstrings should match implementation
- CI tests should verify doc claims match code behavior
- Single source of truth for retrieval pattern
