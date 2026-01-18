# Holographic Training Plan — Transformer Killer Architecture

## WHY THIS BEATS TRANSFORMERS

### Architectural Advantages

| Aspect | Transformers | Holographic |
|--------|--------------|-------------|
| **Attention** | O(n²) softmax | O(1) Grace basin routing |
| **Learning** | Gradient descent (slow) | Hebbian accumulation (instant) |
| **Scaling** | Linear parameters | Sublinear (consolidation) |
| **Inference** | Full forward pass | Single unbind operation |
| **Temperature** | Arbitrary tuning | φ-kernel (theory-derived) |
| **Interpretability** | Black box | Inspectable attractors |
| **Continual Learning** | Catastrophic forgetting | Dreaming prevents forgetting |

### Information Parsimonies (12 Brain-Inspired)

**ENCODING (what to remember):**
1. **Salience**: scalar + pseudoscalar = "how important"
2. **Novelty**: distance from existing prototypes = "how new"
3. **Prediction Error**: Grace residual = "how surprising"
4. **Predictive Coding**: only encode what memory doesn't predict

**MAINTENANCE (how to organize):**
5. **φ-Decay Forgetting**: survival = φ^(-k × (1 - priority))
6. **Interference Management**: merge similar prototypes
7. **Reconsolidation**: retrieval makes memory labile → credit assignment
8. **Pseudo-Rehearsal**: generate samples to prevent forgetting

**RETRIEVAL (how to recall):**
9. **Working Memory**: 7±2 items, salience-gated
10. **Pattern Completion**: Grace flow denoises queries
11. **Inhibition of Return**: suppress recently retrieved

**TEMPORAL (sequences):**
12. **Sequence Replay**: vorticity encodes transitions

---

## GPU OPTIMIZATION STRATEGY

### What's GPU-Friendly
- **Matrix multiply**: Geometric product IS matrix multiply — perfect for GPU
- **Batch operations**: All core ops support batching
- **Einsum**: Grace operator uses einsum for tensor contraction
- **CuPy**: Direct CUDA 12.x acceleration on H100

### Key Batch Operations
```python
# Already implemented in algebra.py:
geometric_product_batch(A, B)     # [batch, 4, 4] × [batch, 4, 4]
grace_operator_batch(M, basis)    # Vectorized grade projection
decompose_to_coefficients_batch() # Extract all 16 components
grace_basin_keys_batch_direct()   # Batch witness extraction
```

### Performance Targets
- **Learning**: 50,000+ samples/sec (GPU) / 25,000+ (CPU)
- **Retrieval**: 5,000+ queries/sec
- **Generation**: 100+ tokens/sec

---

## TRAINING ARCHITECTURE

### Phase 1: Waking (Learning)
```
For each batch of (context, target) pairs:
    1. Embed context → [batch, 4, 4] via geometric product chain
    2. Route to Grace basin (witness extraction)
    3. Accumulate: memory[basin] += φ⁻¹ × bind(context, target)
    4. Track errors for credit assignment
    5. Collect episodes for dreaming
```

### Phase 2: Dreaming (Consolidation)
```
Triggered when:
    - stability < φ⁻² (memory needs consolidation)
    - error_rate > φ⁻¹ (credit assignment detected systematic errors)
    - max_samples exceeded (safety valve)

Non-REM (consolidation):
    - Cluster episodes by witness similarity
    - Form semantic prototypes (centroid + target distribution)
    - Grace-stable prototypes survive

REM (recombination):
    - Recombine prototypes via geometric product
    - Survival test under strong Grace (φ⁻⁴ rate)
    - Survivors become schemas (abstract patterns)
```

### Phase 3: Retrieval (Inference)
```
For query context:
    1. Embed query → [4, 4]
    2. Pattern completion via Grace flow (denoise)
    3. Find nearest prototype by witness distance
    4. Sample from target distribution via φ-kernel
```

---

## φ-CURRICULUM (Context Scaling)

Theory: Learning rate should match memory stability.
- Start small (fast learning, many attractors)
- Grow context as stability increases (slower, more consolidation)

```
Stage 0: ctx=64,     500k samples   — Fast acquisition
Stage 1: ctx=168,    809k samples   — Initial patterns
Stage 2: ctx=440,    1.31M samples  — Pattern composition
Stage 3: ctx=1152,   2.12M samples  — Abstract schemas
Stage 4: ctx=3017,   3.43M samples  — Long-range deps
Stage 5: ctx=7901+   (continue)     — Full context

context(stage) = 64 × φ^(2 × stage)
samples_per_stage = 500k × φ^stage
```

---

## METRICS (Theory-True)

### Primary: Perplexity
```
PPL = exp(-1/N × Σ log P(target|context))

Where P(target|context) uses φ-kernel:
    weight = φ^(-distance)  # distance = 1 - cosine_sim
    P = weight / Σ weights
```

### Secondary: Stability
```
stability = avg(grace_stability(prototype))
         = avg((scalar² + pseudo²) / total_energy)

Healthy: stability > φ⁻² ≈ 0.382
Crisis: stability < φ⁻³ ≈ 0.236 → trigger dreaming
```

### Diagnostic: Coverage
```
coverage = unique_basins / total_samples
         = how well prototypes cover input space

Good: coverage grows sublinearly (consolidation working)
Bad: coverage grows linearly (not consolidating)
```

---

## IMPLEMENTATION STRUCTURE

```python
# train_modal.py structure

# 1. IMPORTS (theory-true, no aliases)
from holographic_prod.memory import HolographicMemory, MemoryConfig
from holographic_prod.dreaming import DreamingSystem, EpisodicEntry
from holographic_prod.attention import ToroidalAttention
from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ, ...

# 2. CURRICULUM
def get_curriculum_stage(samples) -> (stage, context_size)

# 3. DREAMING TRIGGERS
def should_dream(stability, error_rate, samples_since_dream) -> bool

# 4. METRICS
class TrainingMetrics:
    - perplexity_history
    - stability_history
    - coverage_history
    - throughput_history

# 5. MAIN TRAINING LOOP
@app.function(gpu="H100")
def train(max_samples, vocab_size, ...):
    # Initialize
    model = HolographicMemory(vocab_size, use_gpu=True)
    dreamer = DreamingSystem(model.basis, model.xp)
    attention = ToroidalAttention(n_satellites=16)
    
    # Training loop
    for batch in data:
        # Learn
        model.learn_batch(batch)
        
        # Collect episodes
        episodes.extend(make_episodes(batch))
        
        # Adaptive dreaming
        if should_dream(...):
            dreamer.sleep(episodes)
            episodes.clear()
        
        # Log metrics
        if samples % log_every == 0:
            log_metrics(...)
    
    return final_stats

# 6. TEST FUNCTIONS
@app.function(gpu="H100")
def test(): ...

# 7. LOCAL ENTRY
@app.local_entrypoint()
def main(): ...
```

---

## UNIQUE VALUE PROPOSITIONS

### 1. No Attention Bottleneck
Transformers: O(n²) attention matrix
Holographic: O(1) Grace basin routing

**Result**: Linear scaling with context, not quadratic.

### 2. Instant Learning
Transformers: Thousands of gradient steps per fact
Holographic: One accumulation per fact

**Result**: 100-1000x faster learning per sample.

### 3. Interpretable Memory
Transformers: Weights are uninterpretable
Holographic: Each attractor = one concept cluster

**Result**: Can inspect, edit, debug what model "knows".

### 4. Natural Generalization
Transformers: Generalization from weight interpolation
Holographic: Generalization from Grace basin structure

**Result**: Novel queries route to appropriate basins geometrically.

### 5. Continual Learning
Transformers: Catastrophic forgetting
Holographic: Dreaming consolidates without forgetting

**Result**: Can keep learning indefinitely.

### 6. Theory-Derived Hyperparameters
Transformers: Temperature, learning rate, etc. are tuned
Holographic: All rates are φ-derived

**Result**: No hyperparameter search needed.

---

## NEXT STEP

Create `train_modal.py` implementing this plan with:
- Clean imports from actual module locations
- Full 12-parsimony DreamingSystem
- GPU-optimized batch operations
- φ-curriculum context scaling
- Theory-true metrics (no arbitrary thresholds)
- Comprehensive Modal H100 deployment
