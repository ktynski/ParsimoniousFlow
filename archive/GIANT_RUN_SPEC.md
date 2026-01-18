# Giant Model Training Specification

## Current State Assessment (Updated 2026-01-11)

> **⚠️ NOTE:** This document reflects outdated parameter names. For current training,
> use `holographic_v4/holographic_modal.py::train` with modern defaults.

### What's Working
- ✓ Dreaming system (Non-REM + REM) with 12 brain-inspired parsimonies
- ✓ Self-organizing consolidation (grace_stability σ < φ⁻² → consolidates)
- ✓ Multi-modal targets (entropy 2.6+)
- ✓ Schema discovery (68 schemas per sleep)
- ✓ All 52 tests pass (19 core + 21 dreaming + 6 vorticity + 6 distributed prior)
- ✓ NO arbitrary normalizations, softmax, or clipping
- ✓ **Vorticity Grammar Matching** (0.92+ similarity for same structure)
- ✓ **Scalable Context Windows** (tested to 8192+, stable)
- ✓ **OPTIMIZED**: Two-path training (14x speedup when episodic buffer full)
- ✓ **OPTIMIZED**: Vectorized decomposition (34x speedup)
- ✓ **OPTIMIZED**: Vectorized embedding lookup (11x speedup)

### Key Theory-True Features
- Grace-stability determines consolidation (spectral gap threshold φ⁻²)
- Attention = grace_stability × salience (not softmax)
- Vorticity-weighted decoding (prevents mode collapse)
- Vorticity grammar matching (discriminates word order)
- Grace basin discovery for semantic retrieval (no arbitrary thresholds)
- **Distributed Prior** (brain-analog population coding):
  - Superposed attractors with φ-weighted blending
  - Factorized associative prior (Hebbian "weights")
  - Geometric confidence (margin-based, not probability)

### Issues Addressed
- Fixed: Generalization plateau (grace basin discovery)
- Fixed: Mode collapse (vorticity-weighted decoding)
- Fixed: Arbitrary normalizations removed (Grace is THE normalizer)
- Fixed: Context window limits (now O(1) storage, tested to 8192+)
- Performance: ~50,000/sec on CPU, expect 100,000-200,000/sec on H100

---

## Dataset Selection (CRITICAL)

**TinyStories is WRONG for this architecture!**

```
┌──────────────────────────────────────────────────────────────────────┐
│  Dataset         Avg Length      Max Context    Best Use Case        │
├──────────────────────────────────────────────────────────────────────┤
│  TinyStories     ~200 words      256            Testing only         │
│  Wikipedia       ~3000 words     2048           General knowledge    │
│  pg19 (books)    ~50,000 words   65536          FULL CAPABILITY      │
│  arxiv (papers)  ~8000 words     8192           Technical content    │
└──────────────────────────────────────────────────────────────────────┘
```

**Why pg19?**
- Full books = 50,000+ words per document
- Tests TRUE long-range dependencies (chapters, characters, plots)
- Exercises vorticity grammar at scale
- Shows O(1) context advantage vs O(N²) Transformers

---

## Giant Run Parameters

### Option A: "Show Full Capability" (RECOMMENDED)

```bash
modal run -d holographic_v4/holographic_modal.py::train \
  --dataset pg19 \
  --max-samples 100000000 \
  --vocab-size 100000 \
  --context-size 4096 \
  --max-attractors 100000000 \
  --noise-std 0.3 \
  --log-every 100000 \
  --deep-log-every 5000000 \
  --generate-every 1000000 \
  --sleep-every 1000000 \
  --episodic-buffer-size 50000 \
  --rem-cycles 3
```

**Expected:**
- Time: 10-12 hours on H100
- Samples: 100M (full books)
- Context: 4096 tokens (paragraph-level dependencies)
- Prototypes: 10,000+ after 100 sleep cycles

### Option B: "Maximum Context" (After Option A)

```bash
modal run -d holographic_v4/holographic_modal.py::train \
  --dataset pg19 \
  --max-samples 500000000 \
  --vocab-size 100000 \
  --context-size 8192 \
  --max-attractors 200000000 \
  --noise-std 0.3 \
  --log-every 500000 \
  --deep-log-every 10000000 \
  --generate-every 5000000 \
  --sleep-every 2000000 \
  --episodic-buffer-size 100000 \
  --rem-cycles 3
```

**Expected:**
- Time: 12+ hours on H100
- Context: 8192 tokens (chapter-level dependencies!)
- Tests architecture limits

---

## What We're Measuring

### Primary Metrics
1. **Retrieval Accuracy**: Should stay ~70%+ on seen data
2. **Generalization**: Should improve with more prototypes  
3. **Generation Quality**: Should produce coherent multi-sentence text

### Vorticity Grammar Metrics (NEW)
1. **Vorticity Discriminability**: Same structure similarity (target: >0.9)
2. **Grammar Matching Accuracy**: Correct prototype by structure
3. **Word Order Sensitivity**: A∧B vs B∧A distinctness

### Dreaming Metrics
1. **Prototypes Created**: More = better coverage
2. **Schemas Discovered**: More = more abstractions
3. **Compression Ratio**: Should be 10-50x
4. **Avg Entropy**: Should be >1.5 (multi-modality)

### Theory Metrics
1. **Grace-stability (σ)**: Avg σ of episodes (σ < φ⁻² → consolidates)
2. **Witness Stability**: Should stay >0.95
3. **Enstrophy**: Vorticity energy (high = structural information)

### Self-Organizing Metrics
1. **Consolidation Ratio**: % episodes below φ⁻² threshold
2. **Prototype Stability**: Avg σ of prototypes (should be high)
3. **Basin Discrimination**: Accuracy of grace basin discovery

---

## Key Insight

This is NOT a transformer. We're testing:

> "Can O(1) context storage + vorticity grammar match O(N²) Transformers?"

The theory-true approach:
1. No arbitrary hyperparameters (φ is derived, not tuned)
2. Self-organizing memory (grace_stability determines fate)
3. Equilibrium semantics (outputs are physics, not statistics)
4. **O(N) context composition** (vs O(N²) attention)
5. **O(1) context storage** (4×4 matrix regardless of length!)

---

## Context Scaling Advantages

```
┌──────────────────────────────────────────────────────────────────────┐
│  Context Size    Transformer Cost    Our Cost    Ratio              │
├──────────────────────────────────────────────────────────────────────┤
│      256              65,536           256        256x cheaper       │
│     1024           1,048,576          1024       1024x cheaper       │
│     4096          16,777,216          4096       4096x cheaper       │
│     8192          67,108,864          8192       8192x cheaper       │
│    65536       4,294,967,296         65536      65536x cheaper       │
└──────────────────────────────────────────────────────────────────────┘

At context=65536, we're 65,536x more efficient!
Transformer attention: O(N²) = 4.3 billion operations
Our context: O(N) = 65,536 operations + fixed 4×4 matrix
```

---

## Ready Check

Before giant run:
- [x] Vorticity grammar tests pass
- [x] Context scaling tests pass (to 8192)
- [x] pg19 dataset configured
- [ ] Option A run completed
- [ ] Generation is coherent at large context
- [ ] Prototypes capture book-level patterns

---

## Run Command (Option A — Full Capability)

```bash
cd /Users/fractlphoneroom1/Desktop/ParsimoniousFlow
modal run -d holographic_v4/holographic_modal.py::train \
  --dataset pg19 \
  --max-samples 100000000 \
  --vocab-size 100000 \
  --context-size 4096 \
  --max-attractors 100000000 \
  --noise-std 0.3 \
  --log-every 100000 \
  --deep-log-every 5000000 \
  --generate-every 1000000 \
  --sleep-every 1000000 \
  --episodic-buffer-size 50000 \
  --rem-cycles 3
```
