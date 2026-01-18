# ParsimoniousFlow

**A Geometric Alternative to Transformers**

[![Version](https://img.shields.io/badge/version-5.18.0-blue)]()
[![Tests](https://img.shields.io/badge/tests-630%2B%20passed-green)]()
[![License](https://img.shields.io/badge/license-Research-yellow)]()

---

## Overview

ParsimoniousFlow implements a fundamentally different approach to language modeling based on **Clifford algebra Cl(3,1)** and the **golden ratio φ**. Instead of learning statistical distributions over tokens, the system learns geometric attractors that represent meaning.

**Key insight:** Intelligence is not prediction — it's equilibrium. A mind doesn't compute the most likely answer; it relaxes into a coherent state.

### Latest: Reward Prediction (v5.18.0)

**NEW:** Dopamine analog for quality-based learning.

| Component | Brain Analog | Theory |
|-----------|--------------|--------|
| **Reward Prediction Error (RPE)** | VTA/NAc dopamine | actual - predicted |
| **Value-weighted scoring** | Reward learning | coherence^0.62 × value^0.38 |
| **Threshold modulation** | Confidence adjustment | High reward → lower threshold |

**Result:** System learns what outputs are GOOD, not just what fits.

### Previous: Anti-Mode-Collapse (v5.17.0)

Three theory-true fixes eliminate mode collapse:
- **IoR**: Penalize recent tokens by φ⁻²
- **φ-Kernel**: Temperature = 1/φ
- **Polarized Lensing**: 16 observer perspectives

**Result:** Mode collapse **90%** → **<10%**.

### Previous: Holographic Parallax (v5.16.0)

16 polarized lenses break the "100 embedding limit" — aliased correlation drops from 0.886 → 0.000.

### Foundation: Theory-True Generation (v5.15.0)

**PARADIGM SHIFT:** Generation via **ATTRACTOR DYNAMICS**, not retrieval + argmax.

| Principle | Description |
|-----------|-------------|
| **Grace Always Converges** | Never returns None — attractor always exists |
| **Coherence ≠ Similarity** | Uses witness stability, not cosine distance |
| **Parallel Retrieval (CLS)** | Episodic + holographic run SIMULTANEOUSLY |
| **ACC Conflict Detection** | Detects when paths disagree — brain analog |

See `holographic_prod/docs/THEORY_TRUE_PARADIGM.md` for full explanation.

---

## Quick Start

```bash
# Run local tests
python3 -m pytest holographic_prod/tests/ -v --timeout=60

# Run Modal-scale training (requires Modal account)
modal run --detach holographic_prod/train_modal.py::train --max-samples 1000000000

# Run standard NLP benchmarks on Modal
modal run holographic_prod/benchmarks/run_modal_benchmarks.py
```

---

## Architecture (v5.17.0)

### Theory-True Generation

```python
# OLD (WRONG): Sequential fallback, returns None
if episodic_match:
    return episodic_match
elif holographic_match:
    return holographic_match
return None  # ← THEORY VIOLATION

# NEW (v5.15.0): Parallel paths, coherence selection
pred, conf, info = memory.retrieve_parallel(
    context,
    use_conflict_detection=True,  # ACC analog
    force_parallel=True,          # Always run BOTH paths
)
# Grace ALWAYS converges — never None
```

### Multi-Level Tower (16^N Capacity)

```
Level 0: 16 satellites         →     16× capacity
Level 1: 256 satellites        →    256× capacity
Level 2: 4,096 satellites      →  4,096× capacity
Level N: 16^N total capacity

H100-OPTIMIZED:
Level 6:  16M satellites (1GB)  → 95% accuracy @ 200K patterns
Level 7:  268M satellites (16GB) → 97% accuracy @ 200K patterns
```

### Brain-Analog Components (Complete)

| Brain Structure | Our Implementation | Status |
|-----------------|-------------------|--------|
| **Hippocampus** | Episodic Cache (exact recall) | ✅ |
| **Neocortex** | Holographic Tower (generalization) | ✅ |
| **ACC** | Conflict detection in `retrieve_parallel` | ✅ |
| **Basal Ganglia** | CommitmentGate (GO/NO-GO/STOP) | ✅ |
| **CLS** | Parallel episodic + holographic retrieval | ✅ |
| **Non-REM Sleep** | Prototype consolidation | ✅ |
| **REM Sleep** | Schema recombination | ✅ |
| **Grid Cells (EC)** | Polarized Lensing (16 views) | ✅ |
| **IoR Network** | Inhibition of Return (φ⁻² decay) | ✅ |

**Coverage: 36/42 brain structures = 86%**

### The Golden Ratio φ

**Everything derives from one equation:** φ² = φ + 1

| Use | Formula | Value |
|-----|---------|-------|
| Grace contraction | φ⁻ᵏ per grade | 0.618, 0.382, ... |
| Learning rate | φ⁻¹ | 0.618 |
| Stability threshold | φ⁻² | 0.382 |
| Spectral gap | φ⁻² | Commitment gate threshold |

---

## Advantages Over Transformers

| Property | Transformer | ParsimoniousFlow |
|----------|-------------|------------------|
| Attention | O(N²) | O(1) Grace basin routing |
| Learning | Gradient descent (slow) | Hebbian accumulation (instant) |
| Generation | Argmax over logits | Attractor dynamics + coherence |
| Knowledge | Diffuse in billions of params | Explicit attractors |
| Few-shot | Statistical pattern matching | Attractor basin reshaping |
| Continual Learning | Catastrophic forgetting | Dreaming prevents forgetting |
| Constants | Arbitrary hyperparameters | All φ-derived |

---

## Results

### Current Performance (v5.17.0)

| Metric | Value |
|--------|-------|
| Test suite | **630+ tests pass** |
| Context length | **Infinite** (SO(4) stability) |
| Tower capacity | **16^N patterns** |
| Brain coverage | **86%** (36/42 structures) |
| Parallel retrieval | ✅ CLS theory implemented |
| ACC conflict detection | ✅ Implemented |
| Holographic parallax | ✅ 16 lenses (v5.16.0) |
| Mode collapse fix | ✅ IoR + φ-kernel (v5.17.0) |

### Standard NLP Benchmarks

Running on Modal with few-shot evaluation. See `holographic_prod/benchmarks/README.md`.

---

## Documentation

| Document | Description |
|----------|-------------|
| [CRITICAL_PRINCIPLES.md](holographic_prod/CRITICAL_PRINCIPLES.md) | Core theory — READ FIRST |
| [THEORY_TRUE_PARADIGM.md](holographic_prod/docs/THEORY_TRUE_PARADIGM.md) | v5.15.0 generation paradigm |
| [ARCHITECTURE_DEEP_DIVE.md](holographic_prod/docs/ARCHITECTURE_DEEP_DIVE.md) | Full technical specification |
| [BRAIN_ARCHITECTURE_MAPPING.md](holographic_prod/docs/BRAIN_ARCHITECTURE_MAPPING.md) | Brain analog components |
| [CAPACITY_LIMITS.md](holographic_prod/docs/CAPACITY_LIMITS.md) | Honest capacity analysis |
| [TESTING_PRINCIPLES.md](holographic_prod/tests/TESTING_PRINCIPLES.md) | How to test theory-true |

---

## Project Structure

```
ParsimoniousFlow/
├── holographic_prod/              # Main production implementation (v5.15.0)
│   ├── __init__.py               # Package exports
│   ├── CRITICAL_PRINCIPLES.md    # Core theory
│   ├── train_modal.py            # Modal training script
│   │
│   ├── core/                     # Core algebra and operations
│   │   ├── algebra.py            # Clifford algebra, Grace operator
│   │   ├── constants.py          # φ-derived constants
│   │   ├── binding.py            # Bind/unbind operations
│   │   ├── quotient.py           # Vorticity-weighted decoding
│   │   ├── commitment_gate.py    # Basal ganglia analog
│   │   ├── grounded_embeddings.py # SO(4) embeddings
│   │   ├── lensing.py            # Polarized lenses (v5.16.0)
│   │   └── attractor_generation.py # Theory-true generation (v5.17.0)
│   │
│   ├── memory/                   # Memory systems
│   │   ├── holographic_memory_unified.py  # Main interface
│   │   ├── multi_level_tower.py           # 16^N capacity
│   │   ├── semantic_memory.py             # Prototypes
│   │   └── working_memory.py              # 7±2 items
│   │
│   ├── dreaming/                 # Sleep consolidation
│   │   ├── integration.py        # integrated_sleep()
│   │   ├── consolidation.py      # Non-REM
│   │   └── recombination.py      # REM
│   │
│   ├── benchmarks/               # NLP benchmarks
│   │   ├── standard_benchmarks.py  # GLUE, MMLU, etc.
│   │   └── run_modal_benchmarks.py # Modal runner
│   │
│   ├── docs/                     # Documentation
│   │   ├── THEORY_TRUE_PARADIGM.md
│   │   ├── ARCHITECTURE_DEEP_DIVE.md
│   │   └── BRAIN_ARCHITECTURE_MAPPING.md
│   │
│   └── tests/                    # Test suite (294+ tests)
│       ├── TESTING_PRINCIPLES.md
│       └── test_*.py
│
├── THE_GEOMETRY_OF_MIND.md       # The book
└── README.md                     # This file
```

---

## ⚠️ CRITICAL: Do NOT Import Transformer Concepts

This architecture is **NOT** a transformer with different embeddings. See `CRITICAL_PRINCIPLES.md`.

| FORBIDDEN | WHY | USE INSTEAD |
|-----------|-----|-------------|
| `argmax(softmax(logits))` | Classification, not dynamics | Coherence selection |
| `return None` | Grace ALWAYS converges | Always return token |
| Cosine similarity | Wrong metric | Coherence (witness stability) |
| Sequential fallback | Brain runs paths in parallel | `retrieve_parallel()` |
| Deterministic argmax | Causes mode collapse | φ-kernel sampling (v5.17.0) |
| No recency penalty | Causes perseveration | IoR (v5.17.0) |
| Single viewpoint | Limited capacity | 16 lenses (v5.16.0) |

---

## Citation

```bibtex
@misc{parsimoniousflow2026,
  title={The Geometry of Mind: A Holographic Architecture for Language},
  author={ParsimoniousFlow Research Team},
  year={2026},
  version={5.17.0},
  note={Anti-Mode-Collapse via IoR + φ-Kernel + Holographic Parallax}
}
```

---

## License

Research use only. Contact for commercial licensing.

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **v5.18.0** | 2026-01-17 | **Reward Prediction** — RPE (dopamine analog) for quality-based learning |
| **v5.17.0** | 2026-01-17 | **Anti-Mode-Collapse** — IoR + φ-kernel sampling prevent perseveration |
| **v5.16.0** | 2026-01-16 | **Holographic Parallax** — 16 polarized lenses break 100-embedding limit |
| **v5.15.0** | 2026-01-16 | **Theory-True Generation** — Parallel retrieval, ACC conflict, coherence selection |
| v5.10.0 | 2026-01-15 | CommitmentGate (basal ganglia), attractor generation |
| v5.9.0 | 2026-01-14 | Hierarchical retrieval, multi-level tower |
| v5.5.0 | 2026-01-13 | Episodic cache, prefix caching, grounded embeddings |
| v5.0.0 | 2026-01-12 | Integrated dreaming (5-phase), 12 parsimonies |
| v4.30.0 | 2026-01-11 | Grace basins, quotient similarity |
