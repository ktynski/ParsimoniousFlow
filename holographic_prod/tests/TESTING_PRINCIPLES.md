# Testing Principles for Holographic Architecture

## CRITICAL: READ THIS BEFORE WRITING ANY TEST

This architecture is fundamentally different from transformers and traditional ML.
**If you think like a transformer engineer, you will write wrong tests.**

---

## The Paradigm Shift

| Traditional ML | Holographic Architecture |
|----------------|--------------------------|
| Argmax over logits | Grace equilibrium settling |
| Exact token match | Semantic similarity |
| Interference is bad | Superposition is the feature |
| Top-1 accuracy | Resonance and stability |
| Softmax probabilities | φ-weighted geometric scores |

---

## FORBIDDEN PATTERNS (DO NOT USE)

### 1. NEVER Use argmax for Evaluation

```python
# ❌ WRONG - This will give misleading results
scores = embeddings_flat @ retrieved_flat
predicted = np.argmax(scores)
if predicted == target:
    correct += 1
```

**Why it's wrong:** 
- `argmax(similarity)` causes mode collapse (ARCHITECTURE.md line 838)
- High-frequency tokens dominate due to scalar accumulation
- The theory says "NO argmax — just settling" (ARCHITECTURE.md line 1584)

**What to use instead:**
```python
# ✅ CORRECT - Use vorticity-weighted scores
from holographic_prod.core.quotient import vorticity_weighted_scores

scores = vorticity_weighted_scores(attractor, embeddings, basis)
```

### 2. NEVER Measure Exact Token Match as Primary Metric

```python
# ❌ WRONG - Exact match is not the goal
if retrieved_token == target_token:
    accuracy += 1
```

**Why it's wrong:**
- The architecture stores ALL valid continuations in superposition
- Multiple tokens can be "correct" for a given context
- Semantic similarity IS the success metric

**What to use instead:**
```python
# ✅ CORRECT - Use semantic similarity
from holographic_prod.core.algebra import frobenius_cosine

similarity = frobenius_cosine(retrieved_matrix, target_embedding)
# similarity > 0.9 is success
```

### 3. NEVER Call Superposition "Interference"

```python
# ❌ WRONG - This framing is backwards
interference = ctx1.T @ ctx2 @ tgt2
signal_to_interference_ratio = ...
```

**Why it's wrong:**
- Superposition IS the storage mechanism
- Multiple targets together is the DESIGN, not a bug
- Grace equilibrium selects from the superposition

**What to use instead:**
```python
# ✅ CORRECT - Frame as superposition retrieval
superposed_targets = ctx.T @ memory  # Contains all stored targets
equilibrium = evolve_to_equilibrium(superposed_targets, attractor, basis)
```

### 4. NEVER Write Your Own Decoding

```python
# ❌ WRONG - Reinventing (incorrectly)
similarities = embeddings @ retrieved.flatten()
return np.argmax(similarities)
```

**What to use instead:**
```python
# ✅ CORRECT - Use existing theory-true functions
from holographic_prod.core.quotient import vorticity_weighted_scores
from holographic_prod.resonance import evolve_to_equilibrium, find_resonant_prototype
```

---

## CORRECT METRICS FOR EVALUATION

### Primary Metrics (Theory-True)

1. **Semantic Similarity** (`semantic_sim`)
   - Frobenius cosine between retrieved and target
   - 0.96+ is excellent

2. **Average Rank** (`avg_rank`)
   - Where the correct token ranks in similarity scores
   - Lower is better; shows learning over time

3. **Stability** (`stability`)
   - Grace stability of the memory state
   - σ ≥ φ⁻² (0.382) indicates convergence

4. **Resonance**
   - How well retrieved state resonates with attractors
   - Theory-derived from Tr(A @ B)

### Secondary Metrics (Informational Only)

- Top-1 accuracy: Only for comparison, NOT the goal
- Top-5/Top-10: Better indicators than Top-1
- Perplexity: Must use φ-kernel, not softmax

---

## CORRECT TEST STRUCTURE

```python
"""
Test: [What aspect of theory you're testing]

THEORY: [Quote the specific theory being tested]

METRIC: [Which theory-true metric you're using]
"""

import numpy as np
from holographic_prod.core.algebra import frobenius_cosine, get_cached_basis
from holographic_prod.core.quotient import vorticity_weighted_scores
from holographic_prod.resonance import evolve_to_equilibrium


def test_example():
    """Test that retrieval produces semantically similar output."""
    # Setup
    model = HolographicMemory(vocab_size=1000)
    basis = get_cached_basis()
    
    # Store pattern
    context = [1, 2, 3, 4]
    target = 42
    model.learn(context, target)
    
    # Retrieve
    ctx_mat = model.embed_sequence(context)
    retrieved_mat = ctx_mat.T @ model.tower.satellites[...].memory
    target_emb = model.tower.embeddings[target]
    
    # THEORY-TRUE EVALUATION
    similarity = frobenius_cosine(retrieved_mat, target_emb)
    
    # Assert semantic similarity, NOT exact match
    assert similarity > 0.9, f"Semantic similarity {similarity} too low"
```

---

## CHECKLIST BEFORE COMMITTING A TEST

- [ ] Does NOT use `np.argmax` for accuracy
- [ ] Does NOT measure exact token match as primary metric
- [ ] Does NOT frame superposition as "interference"
- [ ] Uses existing theory-true functions (vorticity_weighted_scores, etc.)
- [ ] Cites which theory principle is being tested
- [ ] Uses semantic_sim or resonance as primary metric

---

## REFERENCE: Theory-True Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `vorticity_weighted_scores` | `core/quotient.py` | Theory-true decoding |
| `evolve_to_equilibrium` | `resonance.py` | Grace settling |
| `frobenius_cosine` | `core/algebra.py` | Semantic similarity |
| `grace_stability` | `core/quotient.py` | Stability metric |
| `find_resonant_prototype` | `resonance.py` | Semantic matching |

---

## WHY THIS MATTERS

The holographic architecture achieves O(1) storage and retrieval through geometric
operations in Clifford algebra. This is fundamentally different from:

- Transformers (O(n²) attention, learned weights)
- Traditional memory (hash tables, exact match)
- Classification (argmax over logits)

Tests that use transformer/ML patterns will:
1. Report "failure" when the architecture is working
2. Lead to wrong "fixes" that break the theory
3. Waste time debugging non-problems

**Trust the theory. Use theory-true evaluation.**
