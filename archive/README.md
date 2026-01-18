# Archive — Legacy/Superseded Files

This folder contains files from earlier development phases that have been superseded
by the current implementation in `holographic/`.

**DO NOT USE THESE FILES** — they reflect outdated understanding.

## Why These Are Archived

| File | Issue | Current Location |
|------|-------|------------------|
| `sccmu_core.py` | Uses 16D vectors, not M₄(ℝ) matrices | `holographic/core.py` |
| `theory_verification.py` | Tests old implementation | `modal_sccmu.py::test` |
| `ARCHITECTURE.md` | Describes old 16D approach | `holographic/ARCHITECTURE.md` |
| `DEVELOPMENT_PHASES.md` | Phase 0-9 are superseded | Phase 10+ in holographic/ |
| `TOKENIZATION_*.md` | Old tokenization experiments | Word-level in train() |
| `THEORY_TRUE_*.md` | Early theoretical exploration | `holographic/FOUNDATIONS.md` |
| `CODEBASE_AUDIT.md` | Outdated audit | Current state is clean |

## Current Implementation

The canonical implementation is in:
- `holographic/` — Core library (Cl(3,1) ≅ M₄(ℝ))
- `modal_sccmu.py` — GPU training on Modal

See `holographic/ARCHITECTURE.md` and `holographic/FOUNDATIONS.md` for current theory.

## Key Changes Since Archive

1. **Matrix Representation**: Cl(3,1) ≅ M₄(ℝ) — geometric product = matmul
2. **Identity Bootstrap**: Identity is unique fixed point, enables self-bootstrapping
3. **Biologically Plausible**: Hebbian + Grace learning (no gradient descent)
4. **Active Inference**: EFE-based generation
