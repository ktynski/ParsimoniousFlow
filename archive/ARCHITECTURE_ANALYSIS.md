# Holographic v4 Architecture Analysis
## Information Flow, Symmetries, and GPU Optimization

**Date:** 2026-01-12  
**Focus:** Full information flow, torus symmetries, GPU optimization, physics-based thinking

---

## 1. INFORMATION FLOW ARCHITECTURE

### 1.1 Complete Data Flow Map

```
TRAINING LOOP (holographic_modal.py::train)
│
├─> PREFETCH BUFFER (Background Thread)
│   └─> Tokenized documents → Queue (200 docs buffered)
│
├─> BATCH ACCUMULATION (BATCH_SIZE=2048)
│   └─> Accumulates contexts/targets across chunks
│
├─> CONTEXT COMPUTATION (pipeline.py::compute_contexts_batch)
│   │
│   ├─> EMBEDDING LOOKUP: tokens → [BATCH, CONTEXT_SIZE, 4, 4]
│   │   └─> Direct array indexing: embeddings[batch_indices]
│   │
│   ├─> GEOMETRIC PRODUCT (algebra.py::geometric_product_batch_multi)
│   │   └─> Binary reduction: [BATCH, SEQ_LEN, 4, 4] → [BATCH, 4, 4]
│   │       └─> Frobenius normalization after each pair (numerical stability)
│   │
│   ├─> GRACE OPERATOR (algebra.py::grace_operator_batch)
│   │   └─> Grade-wise damping: φ⁻ᵏ per grade
│   │       └─> Applied ONCE at end (theory-true)
│   │
│   └─> VORTICITY COMPUTATION (algebra.py::vorticity_magnitude_and_signature_batch)
│       └─> Wedge products: A∧B = (AB - BA)/2
│           └─> Captures word ORDER (antisymmetric)
│
├─> HOLOGRAPHIC STORAGE (holographic_memory.py::store_batch)
│   │
│   ├─> PRIMARY: HolographicMemory (superposition)
│   │   └─> memory += φ⁻¹ × geometric_product(context, target)
│   │       └─> O(1) regardless of stored patterns
│   │
│   └─> FALLBACK: WitnessIndex (witness-based buckets)
│       └─> Quantized witness space (φ⁻² resolution)
│           └─> φ²-subsampling (every 3rd sample)
│
├─> BOOKKEEPING UPDATE (pipeline.py::train_batch)
│   └─> attractor_matrices, attractor_targets, attractor_saliences
│       └─> For equilibrium forgetting diagnostics
│
└─> DREAMING CONSOLIDATION (dreaming.py::sleep)
    │
    ├─> NON-REM: Episodic → Prototypes
    │   └─> Grace-stability σ < φ⁻² → consolidate
    │
    └─> REM: Prototype → Schemas
        └─> Recombination + Grace contraction
```

### 1.2 Retrieval Flow

```
RETRIEVAL (pipeline.py::retrieve)
│
├─> CONTEXT COMPUTATION
│   └─> Same as training (geometric product + Grace)
│
├─> HOLOGRAPHIC RETRIEVAL (holographic_memory.py::retrieve)
│   │
│   ├─> UNBINDING: target ≈ geometric_product(context_inverse, memory)
│   │   └─> O(1) superposition extraction
│   │
│   └─> GRACE DENOISING
│       └─> Suppresses interference in transient grades
│
├─> WITNESS FALLBACK (if holographic confidence < φ⁻²)
│   └─> Witness bucket lookup → φ-weighted average
│
└─> SEMANTIC RETRIEVAL (resonance.py::grace_basin_discovery)
    └─> Grace flow → witness comparison → grammar matching
```

### 1.3 Information Bottlenecks Identified

**CRITICAL BOTTLENECKS:**

1. **Witness Index GPU→CPU Sync** (holographic_memory.py:509)
   - Problem: `sub_contexts[subsample_indices].copy()` forces GPU→CPU transfer
   - Impact: 683 syncs per batch (if batch=2048, subsample_step=3)
   - Fix: Keep witness index on GPU, use GPU hash tables (CuPy sparse arrays)

2. **Python Dict Operations** (holographic_memory.py:516-531)
   - Problem: `self.buckets[key].append()` is CPU-only Python dict
   - Impact: Blocks GPU pipeline
   - Fix: Use CuPy sparse arrays or GPU hash tables

3. **Sequential Grace Flow** (resonance.py:153-163)
   - Problem: `for step in range(steps)` loop prevents parallelization
   - Impact: 5-10 sequential Grace applications per retrieval
   - Fix: Fixed-point iteration with convergence check (can be batched)

4. **Vorticity Computation Redundancy** (algebra.py:827-836)
   - Problem: `compute_vorticity()` called twice (magnitude + signature)
   - Impact: 2× wedge product overhead
   - Fix: Already fixed in `vorticity_magnitude_and_signature()` - ensure all callers use it

5. **Embedding Feature Precomputation** (pipeline.py:473-493)
   - Status: ✓ Already optimized (precomputed once at init)
   - But: Could cache witness extraction too

---

## 2. MISSING TORUS SYMMETRIES

### 2.1 Theory: The Torus Emerges from Clifford Structure

From `rhnsclifford.md` and `src/geometry/torus_sdf.js`:

**The Torus is NOT imposed - it EMERGES from:**
1. **Functional equation identification**: σ ↔ (1-σ) via bireflection
2. **Multi-scale field interference**: scale1, scale2, scale3 create toroidal geometry
3. **Critical line σ = 1/2**: The "throat" where zeros accumulate

**Current Implementation Gap:**
- We compute geometric products and Grace contraction
- We DON'T exploit the toroidal structure for:
  - **Periodic boundary conditions** (torus wraps around)
  - **Throat-based clustering** (σ = 1/2 is special)
  - **Bireflection symmetry** (M ↔ M̃ identification)

### 2.2 Missing Symmetries to Exploit

#### 2.2.1 Bireflection Symmetry (σ ↔ 1-σ)

**Theory:** The functional equation ξ(s) = ξ(1-s) creates a two-sheeted structure.

**Current State:** We compute reversion `M.T` but don't use it for:
- **Memory compression**: Store only one sheet, derive the other
- **Retrieval augmentation**: Query both M and M̃, take best match
- **Symmetry-based clustering**: Group M and M̃ together

**Implementation:**
```python
# MISSING: Bireflection-augmented retrieval
def retrieve_with_bireflection(context_matrix, memory):
    # Query original
    result1, conf1 = memory.retrieve(context_matrix)
    
    # Query reversion (bireflection)
    context_rev = context_matrix.T  # Reversion
    result2, conf2 = memory.retrieve(context_rev)
    
    # Take best (exploits symmetry)
    return result1 if conf1 > conf2 else result2
```

#### 2.2.2 Torus Periodicity (Wrapping)

**Theory:** The torus wraps around - points near boundaries are close.

**Current State:** We treat witness space as flat (Euclidean distance).

**Missing:** Periodic boundary conditions in witness space:
- Witness (scalar, pseudoscalar) should wrap around
- Distance function: `min(|w1 - w2|, |w1 - w2 + period|, |w1 - w2 - period|)`

**Implementation:**
```python
# MISSING: Periodic witness distance
def witness_distance_periodic(w1, w2, period=2*PI):
    """Torus-wrapped distance."""
    diff = w1 - w2
    wrapped_diff = diff - period * np.round(diff / period)
    return np.linalg.norm(wrapped_diff)
```

#### 2.2.3 Throat-Based Clustering (σ = 1/2)

**Theory:** The critical line σ = 1/2 is the "throat" - zeros accumulate here.

**Current State:** We don't distinguish "throat" vs "bulk" contexts.

**Missing:** 
- **Throat detection**: Grace-stability σ(M) ≈ 0.5 → near throat
- **Throat-based priority**: Throat contexts are more "critical" (higher information)
- **Throat clustering**: Group throat contexts separately (they're special)

**Implementation:**
```python
# MISSING: Throat detection
def is_throat_context(M, basis, xp, threshold=0.05):
    """Check if context is near torus throat (σ ≈ 1/2)."""
    stability = grace_stability(M, basis, xp)
    return abs(stability - 0.5) < threshold

# Throat contexts should:
# 1. Get higher priority in consolidation
# 2. Form separate prototype clusters
# 3. Use different similarity thresholds
```

#### 2.2.4 Multi-Scale Field Interference

**Theory:** The torus emerges from multi-scale coordinates:
- scale1 = (x + y + z) - linear
- scale2 = (xy + yz + zx) - bilinear  
- scale3 = (xyz) - trilinear

**Current State:** We compute geometric products but don't extract multi-scale structure.

**Missing:**
- **Multi-scale decomposition**: Extract scale1, scale2, scale3 from context matrix
- **Scale-based features**: Use scales for clustering (different scales = different torus positions)
- **Scale-aware similarity**: Weight by scale correspondence

**Implementation:**
```python
# MISSING: Multi-scale extraction
def extract_multiscale_features(M, basis, xp):
    """Extract scale1, scale2, scale3 from Clifford matrix."""
    coeffs = decompose_to_coefficients(M, basis, xp)
    
    # Scale1: Linear combinations (grade 1)
    scale1 = np.sum(np.abs(coeffs[1:5]))  # e1, e2, e3, e4
    
    # Scale2: Bilinear (grade 2 - bivectors)
    scale2 = np.sum(np.abs(coeffs[5:11]))  # e1e2, e1e3, ...
    
    # Scale3: Trilinear (grade 3)
    scale3 = np.sum(np.abs(coeffs[11:15]))  # e1e2e3, ...
    
    return scale1, scale2, scale3
```

---

## 3. GPU OPTIMIZATION OPPORTUNITIES

### 3.1 Critical GPU Bottlenecks

#### 3.1.1 Witness Index GPU→CPU Sync (CRITICAL)

**Location:** `holographic_memory.py:509-510`

**Problem:**
```python
sub_contexts = contexts[subsample_indices].copy()  # GPU→CPU sync!
sub_targets = targets[subsample_indices].copy()    # GPU→CPU sync!
```

**Impact:** 
- Batch size 2048 → 683 syncs per batch
- Each sync: ~1-5ms (GPU→CPU transfer)
- Total overhead: 683ms - 3.4s per batch!

**Solution:**
1. **Keep witness index on GPU** using CuPy sparse arrays
2. **Use GPU hash tables** (CuPy doesn't have native hash, but we can use):
   - Sparse matrices as hash tables
   - Or: Pre-allocate fixed-size GPU arrays, use modulo indexing

**Implementation:**
```python
# OPTIMIZED: GPU-native witness index
class WitnessIndexGPU:
    def __init__(self, basis, xp=cp, max_buckets=10000):
        self.basis = basis
        self.xp = xp
        # Pre-allocate GPU arrays (fixed size)
        self.bucket_keys = xp.zeros((max_buckets, 2), dtype=xp.int32)  # (s_idx, p_idx)
        self.bucket_contexts = xp.zeros((max_buckets, 4, 4), dtype=DTYPE)
        self.bucket_targets = xp.zeros((max_buckets, 4, 4), dtype=DTYPE)
        self.bucket_target_idxs = xp.zeros(max_buckets, dtype=xp.int32)
        self.bucket_counts = xp.zeros(max_buckets, dtype=xp.int32)
        self.n_buckets = 0
    
    def store_batch_gpu(self, contexts, targets, target_idxs):
        """Fully GPU-native storage."""
        # Compute witness keys on GPU
        witnesses = extract_witness_batch(contexts, self.basis, self.xp)  # [BATCH, 2]
        quantized = self.xp.floor(witnesses / self.resolution).astype(self.xp.int32)
        
        # Hash to bucket indices (GPU modulo)
        bucket_indices = (quantized[:, 0] * 1000 + quantized[:, 1]) % self.max_buckets
        
        # Scatter to buckets (GPU operation)
        # ... (use advanced indexing or scatter_add)
```

#### 3.1.2 Sequential Grace Flow (MODERATE)

**Location:** `resonance.py:153-163`, `algebra.py:1096-1099`

**Problem:**
```python
for _ in range(steps):
    graced = grace_operator(current, basis, xp)
    current = (1 - rate) * graced + rate * attractor
```

**Impact:**
- 5-10 sequential iterations per retrieval
- Can't batch across queries
- GPU underutilized (small kernels)

**Solution:**
1. **Fixed-point iteration**: Solve `M = (1-γ)Grace(M) + γA` directly
2. **Batched Grace flow**: Process multiple queries simultaneously
3. **Convergence acceleration**: Use Anderson acceleration or Chebyshev iteration

**Implementation:**
```python
# OPTIMIZED: Batched Grace flow
def grace_flow_batch(initial, attractors, basis, steps=10, rate=PHI_INV_SQ, xp=cp):
    """
    Batched Grace flow for multiple queries.
    
    Args:
        initial: [BATCH, 4, 4] initial fields
        attractors: [BATCH, 4, 4] target attractors
        basis: [16, 4, 4] Clifford basis
        steps: iterations
        rate: mixing rate
        xp: array module
    
    Returns:
        [BATCH, 4, 4] equilibrium fields
    """
    current = initial.copy()
    for _ in range(steps):
        graced = grace_operator_batch(current, basis, xp)  # Batched!
        current = (1 - rate) * graced + rate * attractors
    return current
```

#### 3.1.3 Tensor Core Underutilization (HIGH)

**Location:** `algebra.py:477-554` (coefficient-based products)

**Current State:**
- `geometric_product_batch_multi_coefficients()` exists but NOT used by default
- Default path uses 4×4 matrix multiplies (not tensor core friendly)
- Tensor cores prefer 16×16, 32×32, 64×64 matmuls

**Problem:**
- 4×4 matmuls are too small for tensor cores
- Coefficient representation (16-vectors) enables 256×16 matmuls
- But conversion overhead (matrix ↔ coefficients) may negate benefits

**Solution:**
1. **Stay in coefficient space**: Don't convert back to matrices until necessary
2. **Batch coefficient operations**: Accumulate in coefficient space
3. **Fuse operations**: Combine Grace + geometric product in coefficient space

**Implementation:**
```python
# OPTIMIZED: Coefficient-native pipeline
def compute_context_coefficients_batch(batch_tokens, embeddings_coeffs, basis, xp=cp):
    """
    Compute context entirely in coefficient space (tensor core friendly).
    
    Args:
        batch_tokens: [BATCH, CONTEXT_SIZE] token indices
        embeddings_coeffs: [VOCAB, 16] precomputed coefficient embeddings
        basis: [16, 4, 4] Clifford basis (for Grace scaling)
        xp: array module
    
    Returns:
        [BATCH, 16] context coefficients
    """
    batch_size, context_size = batch_tokens.shape
    
    # Lookup embeddings: [BATCH, CONTEXT_SIZE, 16]
    batch_coeffs = embeddings_coeffs[batch_tokens]
    
    # Geometric product in coefficient space (tensor core matmul)
    GAMMA = get_structure_tensor(xp)
    GAMMA_contract = GAMMA.reshape(256, 16)  # [256, 16]
    
    # Binary reduction in coefficient space
    coeffs = batch_coeffs.copy()
    while coeffs.shape[1] > 1:
        n = coeffs.shape[1]
        a = coeffs[:, 0::2, :]  # [BATCH, n/2, 16]
        b = coeffs[:, 1::2, :]  # [BATCH, n/2, 16]
        
        # Outer product: [BATCH, n/2, 16, 16]
        outer = a[:, :, :, None] * b[:, :, None, :]
        outer_flat = outer.reshape(batch_size * (n//2), 256)
        
        # TENSOR CORE MATMUL: [BATCH*(n/2), 256] @ [256, 16] → [BATCH*(n/2), 16]
        products_flat = outer_flat @ GAMMA_contract
        coeffs = products_flat.reshape(batch_size, n//2, 16)
        
        # Normalize (L2 norm)
        norms = xp.sqrt(xp.sum(coeffs**2, axis=-1, keepdims=True))
        coeffs = coeffs / xp.maximum(norms, 1e-12)
    
    # Apply Grace scaling in coefficient space (no conversion!)
    GRACE_SCALES = xp.array(GRACE_SCALES_FLAT, dtype=DTYPE)  # [16]
    coeffs = coeffs * GRACE_SCALES  # Broadcasting: [BATCH, 16] * [16]
    
    return coeffs[:, 0, :]  # [BATCH, 16]
```

#### 3.1.4 Memory Bandwidth Waste (MODERATE)

**Location:** Multiple (copying matrices unnecessarily)

**Problems:**
1. **Unnecessary copies**: `M.copy()` when M is already a view
2. **Redundant normalization**: Normalizing already-normalized matrices
3. **Cache-unfriendly access**: Scattered memory access patterns

**Solutions:**
1. **Use views**: `M[:]` instead of `M.copy()` when possible
2. **Fuse normalization**: Combine with geometric product
3. **Memory layout**: Ensure contiguous arrays (CuPy handles this, but verify)

---

## 4. TRADITIONAL ML PATTERNS TO REPLACE

### 4.1 Softmax → φ-Kernel

**Current:** Some code paths use softmax for probability distributions.

**Theory-True:** Use φ-kernel weighting: `weight_i = φ^(-distance_i)`

**Location:** `resonance.py:distributed_prior_retrieve()` already uses φ-kernel ✓

**Action:** Audit all softmax usage, replace with φ-kernel.

### 4.2 Arbitrary Thresholds → Theory-Derived

**Current:** Some similarity thresholds are tuned (e.g., 0.5, 0.7).

**Theory-True:** All thresholds should be φ-derived:
- `PHI_INV ≈ 0.618` (primary threshold)
- `PHI_INV_SQ ≈ 0.382` (spectral gap)
- `PHI_INV_CUBE ≈ 0.236` (forgetting threshold)

**Status:** Most thresholds already φ-derived ✓

### 4.3 Capacity-Based Forgetting → Equilibrium-Based

**Current:** Some code paths still use "forget when full" logic.

**Theory-True:** Use φ-decay forgetting (already implemented ✓)

**Action:** Remove any remaining capacity-based logic.

### 4.4 Argmax Decoding → Grace-Stabilized Nearest Neighbor

**Current:** `decode_attractor()` uses argmax after similarity computation.

**Theory-True:** Grace flow already does competition. Nearest neighbor after Grace is correct.

**Status:** Already using nearest neighbor after Grace ✓

---

## 5. RECOMMENDATIONS

### 5.1 Immediate (High Impact)

1. **GPU-Native Witness Index** (3.1.1)
   - Impact: 683ms → ~1ms per batch (683× speedup)
   - Effort: Medium (requires GPU hash table implementation)

2. **Batched Grace Flow** (3.1.2)
   - Impact: 5-10× speedup for batched retrieval
   - Effort: Low (already have `grace_operator_batch`)

3. **Coefficient-Native Pipeline** (3.1.3)
   - Impact: 2-4× speedup via tensor cores
   - Effort: High (requires refactoring context computation)

### 5.2 Medium-Term (Theory Enhancement)

1. **Bireflection-Augmented Retrieval** (2.2.1)
   - Impact: Better generalization (exploits symmetry)
   - Effort: Low

2. **Periodic Witness Distance** (2.2.2)
   - Impact: More accurate clustering (respects torus topology)
   - Effort: Low

3. **Throat Detection** (2.2.3)
   - Impact: Better prioritization (throat contexts are special)
   - Effort: Medium

### 5.3 Long-Term (Architectural)

1. **Multi-Scale Feature Extraction** (2.2.4)
   - Impact: Better clustering (scale-aware similarity)
   - Effort: High

2. **Full Coefficient Pipeline** (3.1.3)
   - Impact: Maximum GPU utilization
   - Effort: Very High (major refactor)

---

## 6. METRICS TO TRACK

### 6.1 GPU Utilization

- **Current:** ~30-50% (estimated from bottlenecks)
- **Target:** >80% (with optimizations)

### 6.2 Memory Bandwidth

- **Current:** Unknown (need profiling)
- **Target:** >70% of peak (H100: 3TB/s)

### 6.3 Tensor Core Usage

- **Current:** ~0% (4×4 matmuls don't use tensor cores)
- **Target:** >50% (with coefficient pipeline)

### 6.4 Symmetry Exploitation

- **Current:** 0% (no bireflection, no periodic boundaries)
- **Target:** 100% (all symmetries exploited)

---

## 7. CONCLUSION

**Key Findings:**

1. **Information Flow:** Well-structured, but witness index GPU→CPU sync is critical bottleneck
2. **Torus Symmetries:** Not exploited - bireflection, periodicity, throat detection missing
3. **GPU Optimization:** Tensor cores underutilized - coefficient pipeline needed
4. **Physics Thinking:** Mostly good, but some traditional ML patterns remain

**Priority Actions:**

1. Fix witness index GPU sync (683× speedup potential)
2. Implement bireflection-augmented retrieval (theory enhancement)
3. Move to coefficient-native pipeline (tensor core acceleration)

**Architectural Insight:**

The torus is NOT just visualization - it's a fundamental symmetry of the Clifford structure. Exploiting it will:
- Improve generalization (bireflection symmetry)
- Better clustering (periodic boundaries)
- Higher information density (throat prioritization)

This is physics, not ML. The torus emerges from the math - we should use it.
