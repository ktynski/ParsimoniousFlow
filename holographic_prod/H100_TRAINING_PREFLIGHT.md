# H100 Training Pre-Flight Checklist

## Theory-True Verification ✓

### Critical Code Path Audit

- [x] **retrieve_theory_true()** uses coherence scoring
  - Grace contraction on context (line 2063)
  - Grace contraction on retrieved (line 2078)
  - Full vocabulary coherence: `witness_energy / total_energy` (lines 2117-2123)
  - NO candidate sets, NO argmax on similarity

- [x] **learn_batch()** uses Hebbian binding
  - `bindings = ctx @ tgt` (line 737)
  - Scaled by `PHI_INV` (line 747)
  - Superposition via `xp.add.at()` (lines 745-747)
  - NO gradient descent

- [x] **Grace operator** scales grades (doesn't zero)
  - Grade 0: 1.0 (preserved)
  - Grade 1: φ⁻¹ ≈ 0.618
  - Grade 2: φ⁻² ≈ 0.382 (vorticity damping)
  - Grade 3: φ⁻³ ≈ 0.236
  - Grade 4: φ⁻¹ ≈ 0.618 (Fibonacci exception)

- [x] **Tokenization** extracts ALL samples
  - Step size = 1 (overlapping windows)
  - `<unk>` correctly mapped to index 0
  - `<unk>` targets filtered out

- [x] **evaluate_semantic()** uses coherence
  - `retrieve_settled_states_batch()` includes Grace contractions
  - Returns `witness_energy / total_energy`, NOT Frobenius cosine

---

## Pre-Flight Commands

### 1. Run Verification Suite
```bash
# Quick sanity check (individual claims)
modal run holographic_prod/tests/TRANSFORMER_KILLER_VERIFICATION.py 3
modal run holographic_prod/tests/TRANSFORMER_KILLER_VERIFICATION.py 5

# Full verification
modal run holographic_prod/tests/TRANSFORMER_KILLER_VERIFICATION.py all
```

### 2. Run Information Flow Audit
```bash
modal run holographic_prod/tests/INFORMATION_FLOW_AUDIT.py
```

### 3. Run Granular Flow Tests
```bash
modal run holographic_prod/tests/test_complete_flow_granular.py
```

---

## Training Configuration (Theory-True)

### Recommended Settings
```python
# train_modal.py command
modal run holographic_prod/train_modal.py::train \
    --max-samples 1000000 \
    --batch-size 2048 \
    --dream-interval 100000
```

### Key Parameters (φ-derived, DO NOT CHANGE)
| Parameter | Value | Theory |
|-----------|-------|--------|
| `learning_rate` | φ⁻¹ ≈ 0.618 | Primary Hebbian rate |
| `context_size` | 64 (starts) | Curriculum grows via φ |
| `batch_size` | 2048 | H100 optimal |
| `n_satellites` | 16^N | Hierarchical tower |
| `GRACE_ROUTING_ITERS` | 3 | φ⁻⁶ total contraction |

---

## Expected Metrics

### During Training
| Metric | Expected Range | What It Means |
|--------|----------------|---------------|
| `semantic_similarity` | 0.1 → 0.6+ | Coherence improving |
| `perplexity` | High → Lower | Learning happening |
| `stability` | > 0.38 (φ⁻²) | Grace convergence |
| `throughput` | > 10K samples/sec | H100 performing |

### Red Flags (STOP TRAINING IF SEEN)
- `semantic_similarity = 0.0000` → Evaluation broken
- `None` returns from retrieve → Grace failed
- `throughput < 1000` → GPU sync bottleneck
- `perplexity = inf` → Numerical overflow

---

## Verification Results Required

Before large-scale training, ensure:

1. **Claim 1**: Inference time ratio < 3x (O(1) scaling)
2. **Claim 3**: Single-shot learning works (Hebbian)
3. **Claim 5**: 100% Grace convergence
4. **Audit**: All 8 information flow checks pass

---

## Quick Start

```bash
# Step 1: Verify theory-true (15 min)
modal run holographic_prod/tests/TRANSFORMER_KILLER_VERIFICATION.py 3
modal run holographic_prod/tests/TRANSFORMER_KILLER_VERIFICATION.py 5
modal run holographic_prod/tests/INFORMATION_FLOW_AUDIT.py

# Step 2: Small training test (30 min)
modal run holographic_prod/train_modal.py::train --max-samples 50000

# Step 3: Full training run (hours)
modal run holographic_prod/train_modal.py::train --max-samples 1000000
```

---

## Troubleshooting

### Low `semantic_similarity`
- Check: Is `evaluate_semantic()` using coherence?
- Check: Are Grace contractions applied?
- Check: Is vocabulary being filtered?

### Training Stuck
- Check: Is dreaming happening? (`dreams > 0`)
- Check: Is memory growing? (`n_patterns` increasing)
- Check: Is throughput stable? (no GPU sync issues)

### NaN/Inf Values
- Check: Are embeddings SO(4)? (`det ≈ 1`)
- Check: Is Grace scaling correct? (no zeros)
- Check: Is memory norm bounded? (not exploding)

---

## Summary

**This system is theory-true when:**

✓ Coherence scoring (not Frobenius cosine)  
✓ Hebbian learning (not gradient descent)  
✓ Grace ALWAYS converges (never None)  
✓ Full vocabulary accessible (no candidate sets)  
✓ O(1) inference scaling (not O(n²))  
✓ Information preserved (16 coefficients, SO(4))  

**Run verification before training. If all pass, you're ready for transformer-killing.**
