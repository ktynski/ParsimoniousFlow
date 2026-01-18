# Test Codebase Cleanup Plan

**Date:** 2026-01-13  
**Version:** v4.29.0  
**Status:** ✅ COMPLETED  
**Before:** 108 Python files in root, 77 test files  
**After:** Organized test suite with 163 passing tests  
**Archived:** 66 legacy files to `archive/legacy_tests/`

---

## Current State Analysis

### Test Files by Category

#### ✅ ESSENTIAL (v4.29.0 Core) — IN PLACE
| File | Tests | Purpose |
|------|-------|---------|
| `test_toroidal_attention.py` | 7 | ToroidalAttention |
| `test_dream_cycles.py` | 7 | DreamCycle |
| `test_integrated_attention_dreaming.py` | 5 | Integration |
| `test_integrated_fractal_generative.py` | 8 | FractalGenerativeMemory |
| `test_nested_fractal_torus.py` | 11 | Nested Torus |
| `test_long_context.py` | 6 | Context scaling |
| `test_perplexity_benchmark.py` | 1 | PPL benchmark |
| `test_contrastive_generative.py` | 5 | Contrastive + Generative |
| `test_generative_memory.py` | 7 | Generative memory |

#### ✅ COGNITIVE CAPABILITIES (v4.28-4.29) — NEW
| File | Tests | Purpose |
|------|-------|---------|
| `test_credit_assignment.py` | 7 | Credit assignment |
| `test_meta_learning_integration.py` | 7 | Meta-learning |
| `test_adaptive_memory.py` | 9 | AdaptiveMemory production API |
| `test_distributed_prior_integration.py` | 8 | Distributed prior |
| `test_curiosity_integration.py` | 7 | Curiosity (metacognition) |
| `test_planning_integration.py` | 6 | Planning (causal reasoning) |
| `test_theory_of_mind_integration.py` | 7 | Theory of mind |

#### ✅ MODAL TESTS — KEEP
| File | Lines | Purpose |
|------|-------|---------|
| `test_modal_orthogonalized_scale.py` | 540 | WikiText-2 scale test |
| `test_modal_fractal_scale.py` | ~500 | Fractal scale test |
| `test_contrastive_modal.py` | ~300 | Modal contrastive |

#### ✅ COMPONENT TESTS (`tests/`) — KEEP
| File | Purpose |
|------|---------|
| `tests/test_chirality.py` | Chirality flip |
| `tests/test_dreaming_enhanced.py` | Enhanced dreaming |
| `tests/test_grace_inverse.py` | GraceInverse |
| `tests/test_interaction_tensor.py` | Interaction tensor |
| `tests/test_nested_torus.py` | Nested torus |
| `tests/test_phase_distribution.py` | Phase distribution |

#### ✅ THEORY TESTS (`theory_tests/`) — KEEP
All files in `theory_tests/` — comprehensive theory validation.

---

#### 🗄️ ARCHIVE CANDIDATES — Move to `archive/legacy_tests/`

These are exploratory/intermediate tests that are no longer essential:

| File | Lines | Reason |
|------|-------|--------|
| `binding_tests.py` | ~300 | Superseded by integration tests |
| `brain_efficiency_tests.py` | ~200 | Exploratory |
| `comprehensive_tests.py` | ~400 | Redundant |
| `consolidation_tests.py` | ~300 | Covered by dreaming tests |
| `credit_assignment_tests.py` | ~200 | Exploratory |
| `curiosity_tests.py` | ~200 | Exploratory |
| `distributed_prior_tests.py` | ~200 | Exploratory |
| `dreaming_tests.py` | ~300 | Superseded by test_dream_cycles.py |
| `gclm_engineering_tests.py` | ~400 | Legacy |
| `gpu_witness_tests.py` | ~300 | GPU-specific |
| `grounding_tests.py` | ~200 | Exploratory |
| `hinge_benchmark_tests.py` | ~300 | Benchmark |
| `holographic_exploration_tests.py` | ~300 | Exploratory |
| `holographic_memory_tests.py` | ~400 | Superseded |
| `integration_test.py` | ~200 | Superseded |
| `limitation_tests.py` | ~200 | Exploratory |
| `memory_scaling_tests.py` | ~300 | Superseded |
| `meta_cognitive_integration_test.py` | ~200 | Exploratory |
| `meta_cognitive_tests.py` | ~300 | Exploratory |
| `meta_learning_tests.py` | ~200 | Exploratory |
| `multiscale_resonance_tests.py` | ~300 | Exploratory |
| `planning_tests.py` | ~200 | Exploratory |
| `pre_modal_test.py` | ~200 | Superseded |
| `pre_training_tests.py` | ~200 | Superseded |
| `predictiveness_tests.py` | ~200 | May keep |
| `recursive_computation_tests.py` | ~200 | Exploratory |
| `representation_learning_tests.py` | ~300 | May keep |
| `semantic_prototype_tests.py` | ~200 | Exploratory |
| `test_bucket_debug.py` | ~100 | Debug script |
| `test_canonical_generalization.py` | ~200 | Intermediate |
| `test_cleanup_phase_a.py` | ~200 | Cleanup script |
| `test_contrastive_learning.py` | ~300 | Superseded |
| `test_final_generalization.py` | ~300 | Intermediate |
| `test_generalization_fix.py` | ~200 | Intermediate |
| `test_grade_similarity_at_scale.py` | ~200 | Intermediate |
| `test_grade_weighted_similarity.py` | ~200 | Intermediate |
| `test_iterated_grace_generalization.py` | ~200 | Intermediate |
| `test_learned_generalization.py` | ~200 | Intermediate |
| `test_permutation_collision_rate.py` | ~200 | Intermediate |
| `test_phase_b_semantic_filtering.py` | ~200 | Intermediate |
| `test_phase_c_target_clustering.py` | ~200 | Intermediate |
| `test_production_generalization.py` | ~300 | Intermediate |
| `test_real_text_generalization.py` | ~200 | Intermediate |
| `test_semantic_embeddings_generalization.py` | ~200 | Intermediate |
| `test_semantic_index_accuracy.py` | ~200 | Intermediate |
| `test_similarity_under_perturbation.py` | ~200 | Intermediate |
| `test_vorticity_index_fix.py` | ~200 | Intermediate |
| `theory_audit_tests.py` | ~300 | Superseded by theory_tests/ |
| `theory_of_mind_tests.py` | ~300 | Exploratory |
| `theory_verification_tests.py` | ~300 | Superseded |
| `torus_symmetry_tests.py` | ~200 | Exploratory |
| `vorticity_advanced_tests.py` | ~300 | Exploratory |
| `vorticity_features_tests.py` | ~300 | Exploratory |
| `vorticity_generalization_test.py` | ~200 | Intermediate |
| `vorticity_grammar_test.py` | ~200 | Intermediate |
| `vorticity_grammar_tests.py` | 583 | Exploratory |

---

## Proposed Structure

```
holographic_v4/
├── tests/                      # Organized test suite
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures
│   │
│   ├── core/                   # v4.27.0 Core Tests
│   │   ├── __init__.py
│   │   ├── test_attention.py   # ToroidalAttention
│   │   ├── test_dreaming.py    # DreamCycle
│   │   ├── test_memory.py      # FractalGenerativeMemory
│   │   └── test_integration.py # Full pipeline
│   │
│   ├── components/             # Component Tests
│   │   ├── __init__.py
│   │   ├── test_chirality.py
│   │   ├── test_grace_inverse.py
│   │   ├── test_interaction_tensor.py
│   │   ├── test_nested_torus.py
│   │   └── test_phase_distribution.py
│   │
│   ├── benchmarks/             # Performance Tests
│   │   ├── __init__.py
│   │   ├── test_perplexity.py
│   │   ├── test_long_context.py
│   │   └── test_generative.py
│   │
│   ├── modal/                  # Modal Scale Tests
│   │   ├── __init__.py
│   │   ├── test_orthogonalized_scale.py
│   │   └── test_fractal_scale.py
│   │
│   └── theory/                 # Theory Validation
│       └── (existing theory_tests/)
│
├── archive/                    # Archived Tests
│   └── legacy_tests/           # Old exploratory tests
│
└── (source files)
```

---

## Cleanup Commands

### Step 1: Create directories
```bash
mkdir -p holographic_v4/archive/legacy_tests
```

### Step 2: Move legacy tests to archive
```bash
# Move all *_tests.py files (old naming convention)
mv holographic_v4/*_tests.py holographic_v4/archive/legacy_tests/

# Move intermediate test_ files
mv holographic_v4/test_*generalization*.py holographic_v4/archive/legacy_tests/
mv holographic_v4/test_bucket_debug.py holographic_v4/archive/legacy_tests/
mv holographic_v4/test_cleanup_phase_a.py holographic_v4/archive/legacy_tests/
mv holographic_v4/test_contrastive_learning.py holographic_v4/archive/legacy_tests/
mv holographic_v4/test_grade*.py holographic_v4/archive/legacy_tests/
mv holographic_v4/test_permutation*.py holographic_v4/archive/legacy_tests/
mv holographic_v4/test_phase_*.py holographic_v4/archive/legacy_tests/
mv holographic_v4/test_semantic*.py holographic_v4/archive/legacy_tests/
mv holographic_v4/test_similarity*.py holographic_v4/archive/legacy_tests/
mv holographic_v4/test_vorticity*.py holographic_v4/archive/legacy_tests/
mv holographic_v4/integration_test.py holographic_v4/archive/legacy_tests/
mv holographic_v4/pre_modal_test.py holographic_v4/archive/legacy_tests/
```

### Step 3: Verify essential tests still work
```bash
python3 -m pytest holographic_v4/test_*.py -v --tb=short
```

---

## Summary

| Category | Files | Status |
|----------|-------|--------|
| Essential (v4.27.0) | 12 | ✅ Kept in root |
| Components (tests/) | 7 | ✅ Kept |
| Theory (theory_tests/) | 9 | ✅ Kept |
| **Archived** | **66** | ✅ Moved to `archive/legacy_tests/` |

**Result:** 108 → 42 Python files (61% reduction)

## Test Results After Cleanup

```
169 passed, 12 warnings in 33.47s
```

All tests pass:
- 12 root test files (v4.27.0 core + modal)
- 45 component tests (tests/)
- 64 theory tests (theory_tests/)
- 48 additional tests from integrated suites
