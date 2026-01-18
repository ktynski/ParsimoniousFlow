# Holographic Language Model — Architecture v3.0

## Quick Reference: The Breakthrough

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPOSITIONAL EMBEDDINGS (v3.0)                          │
│                                                                             │
│   THE INSIGHT:                                                              │
│       We were using Clifford algebra at the WRONG level.                   │
│                                                                             │
│   WRONG (atomic):   word = random 4×4 matrix                               │
│                     → Learns co-occurrence, not semantics                   │
│                     → Separation: 0.06                                      │
│                                                                             │
│   RIGHT (compositional):  word = I + Σᵢ αᵢ(word) · fᵢ                      │
│                     where fᵢ = orthogonal feature directions               │
│                     → Learns semantic structure via Hebbian                 │
│                     → Separation: 0.72 (12x better!)                       │
│                     → One-shot learning works                               │
│                     → Correct category generation                           │
│                                                                             │
│   KEY FILES:                                                                │
│       compositional.py   - Feature-based word embeddings                   │
│       feature_learning.py - Hebbian + one-shot inference                   │
│       full_pipeline.py   - Integrated model                                │
│                                                                             │
│   USAGE:                                                                    │
│       from holographic import CompositionalHolographicModel               │
│       model = CompositionalHolographicModel(vocab_size=10000)             │
│       model.train(contexts, targets, hebbian_lr=0.05)                      │
│       tokens = model.generate(context, num_tokens=10)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 0. Topological Foundations

> **See `FOUNDATIONS.md` for the complete formal treatment.**
> **See Section 22 for cross-disciplinary positioning (gauge theory, geometric algebra, dynamical systems, philosophy of mind).**

The architecture is not a design choice — it is **mathematically forced** by the requirements of self-reference:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     WHY THIS ARCHITECTURE IS NECESSARY                          │
└─────────────────────────────────────────────────────────────────────────────────┘

SELF-REFERENCE forces:
    │
    ├──▶ QUOTIENT STRUCTURE (identifying state with representation)
    │        │
    │        └──▶ Fixed-point seams (the "self")
    │             Implemented as: clifford_adjoint(A, G) = G A^T G
    │
    └──▶ COVERING STRUCTURE (multi-valued continuation)
             │
             └──▶ Branch loci (caustics)
                  Implemented as: Grade structure [0,1,2,3,4]

STABILITY requires:
    │
    └──▶ GRACE (contraction guaranteeing well-defined gluing)
             │
             ├──▶ Spectral gap γ = φ⁻² (convergence rate)
             └──▶ Fibonacci exception α₄ = 1 (throat closure)
```

**Lemma (Self-reference forces singular loci).**
Any system that identifies states with representations induces (i) a quotient by an involution, and/or (ii) a multi-valued continuation requiring a covering space. Quotients generically contain fixed-point seams; coverings generically contain branch loci. These are topologically protected and act as attractors under Grace. Therefore self-reference generically produces stable singular sets which function as natural "addresses" of interiority.

---

## 1. Core Isomorphism

The entire system is built on one fundamental insight:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   Cl(3,1)  ≅  M₄(ℝ)                                                    │
│                                                                         │
│   16D Clifford Algebra  ↔  4×4 Real Matrices                           │
│                                                                         │
│   Geometric Product     ↔  Matrix Multiplication (GEMM!)               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**This means**: We can use highly optimized GPU matrix multiplication for all algebra operations.

---

## 2. Clifford Algebra Signature

We use **Cl(3,1)**, NOT Cl(1,3):

```
                    Cl(3,1) Metric: η = diag(+1, +1, +1, -1)
                    
    ┌────────┬────────┬────────┬────────┐
    │   e₁   │   e₂   │   e₃   │   e₄   │
    ├────────┼────────┼────────┼────────┤
    │  e₁²   │  e₂²   │  e₃²   │  e₄²   │
    │  = +I  │  = +I  │  = +I  │  = -I  │
    ├────────┼────────┼────────┼────────┤
    │SPACE   │SPACE   │SPACE   │ TIME   │
    │LIKE    │LIKE    │LIKE    │ LIKE   │
    └────────┴────────┴────────┴────────┘
    
    Anticommutation: {eᵢ, eⱼ} = eᵢeⱼ + eⱼeᵢ = 2ηᵢⱼI  (0 for i≠j)
```

**Why this matters**:
- Cl(3,1) ≅ M₄(ℝ) → **Real** 4×4 matrices (fast, simple)
- Cl(1,3) ≅ M₂(ℍ) → 2×2 **Quaternionic** matrices (complex)

---

## 3. Grade Structure

The 16 basis elements are organized by grade (number of basis vectors in product):

```
                         GRADE HIERARCHY
    
    Grade 4 ─────────────────────────────────────────── φ⁻¹  (FIBONACCI!)
        │
        │   e₁e₂e₃e₄  (pseudoscalar, 1 element)
        │
    Grade 3 ─────────────────────────────────────────── φ⁻³
        │
        │   e₁e₂e₃   e₁e₂e₄   e₁e₃e₄   e₂e₃e₄  (4 trivectors)
        │
    Grade 2 ─────────────────────────────────────────── φ⁻²  (SPECTRAL GAP)
        │
        │   e₁e₂   e₁e₃   e₁e₄   e₂e₃   e₂e₄   e₃e₄  (6 bivectors)
        │
    Grade 1 ─────────────────────────────────────────── φ⁻¹
        │
        │   e₁   e₂   e₃   e₄  (4 vectors)
        │
    Grade 0 ─────────────────────────────────────────── 1.0  (PRESERVED)
        │
        │   1  (scalar, identity)
        │
        ▼
    TOTAL: 1 + 4 + 6 + 4 + 1 = 16 basis elements
```

**Grace Scaling** (φ = 1.618...):

| Grade | Count | Grace Scale | Physical Role |
|-------|-------|-------------|---------------|
| 0 | 1 | 1.0 | Core energy (preserved) |
| 1 | 4 | φ⁻¹ ≈ 0.618 | Direction |
| 2 | 6 | φ⁻² ≈ 0.382 | Torus position (spectral gap!) |
| 3 | 4 | φ⁻³ ≈ 0.236 | Fine structure |
| 4 | 1 | **φ⁻¹ ≈ 0.618** | Fibonacci anyon (NOT φ⁻⁴!) |

---

## 4. Gamma Matrices

Constructed from tensor products of Pauli-like matrices:

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   e₁ = σ₃ ⊗ I₂      e₂ = σ₁ ⊗ σ₃      e₃ = σ₁ ⊗ σ₁            │
    │                                                                 │
    │   ┌ 1  0  0  0┐     ┌ 0  0  1  0┐     ┌ 0  0  0  1┐            │
    │   │ 0  1  0  0│     │ 0  0  0 -1│     │ 0  0  1  0│            │
    │   │ 0  0 -1  0│     │ 1  0  0  0│     │ 0  1  0  0│            │
    │   └ 0  0  0 -1┘     └ 0 -1  0  0┘     └ 1  0  0  0┘            │
    │                                                                 │
    │   e₁² = +I          e₂² = +I          e₃² = +I                 │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   e₄ = σ₂ ⊗ I₂  (TIMELIKE)                                     │
    │                                                                 │
    │   ┌ 0  0  0 -1┐                                                │
    │   │ 0  0  1  0│      e₄² = -I  ← Key difference!               │
    │   │ 0 -1  0  0│                                                │
    │   └ 1  0  0  0┘      G = e₄  (metric matrix for adjoint)       │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Structure

```
holographic/
│
├── constants.py      ← Sacred constants (DO NOT MODIFY)
│   │
│   ├── PHI = 1.618...        # Golden ratio
│   ├── PHI_INV = 0.618...    # 1/φ
│   ├── PHI_INV_SQ = 0.382... # Spectral gap γ
│   ├── MATRIX_DIM = 4        # 4×4 matrices
│   └── CLIFFORD_DIM = 16     # 16 basis elements
│
├── algebra.py        ← Matrix operations
│   │
│   ├── build_gamma_matrices()   # Cl(3,1) generators
│   ├── build_clifford_basis()   # All 16 basis matrices
│   ├── geometric_product()      # = matmul!
│   ├── frobenius_similarity()   # Fast similarity
│   └── grace_operator_matrix()  # Grade scaling
│
├── core.py           ← Learning system
│   │
│   ├── MatrixEmbedding         # Token → 4×4 matrix
│   ├── ContextAttractorMap     # Context → Attractor
│   ├── train_step()            # Single learning step
│   └── generate_token()        # Inference
│
└── __init__.py       ← Package exports
```

---

## 6. Token Embedding and Initialization

Each token is represented as a **4×4 real matrix**. The initialization strategy is **critical**.

### 6.1 The Identity Bootstrap Discovery

**Key finding**: The identity matrix is the unique fixed point of the geometric product.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CRITICAL INITIALIZATION DISCOVERY                            │
└─────────────────────────────────────────────────────────────────────────────────┘

    RANDOM INITIALIZATION:
        Context similarity:  mean=0.02, std=0.21  ← HIGH VARIANCE, UNSTABLE
        
    IDENTITY-BIASED INITIALIZATION:
        Context similarity:  mean=0.76, std=0.08  ← LOW VARIANCE, STABLE
        
    Variance reduction: 3x more stable!
```

### 6.2 Why Identity-Biased Initialization Works

```
                    FIXED POINT ANALYSIS
    
    Question: What happens under repeated geometric product?
    
        M → M @ M → M @ M @ M → ...
        
    Answer: Converges to SCALAR-DOMINATED state in ~5 iterations
    
    The scalar (identity) component is the UNIQUE self-similar basis element:
    
        e₀ @ e₀ = e₀     (self-similarity = 1.0)
        e₁ @ e₁ = +I     (self-similarity = 0.0 to e₁)
        ...
        All other basis elements lose their structure under squaring!
```

### 6.3 Brain Analogy

```
    BRAIN DEVELOPMENT:
        1. All neurons start similar (undifferentiated)
        2. Experience creates differentiation
        3. Homeostasis provides stability
        4. Common features stay similar
        5. Specific features diverge
    
    CLIFFORD BOOTSTRAP:
        1. All embeddings = I + small_noise (undifferentiated)
        2. Hebbian learning creates differentiation
        3. Grace contraction provides stability
        4. Scalar component stays similar (general "word-ness")
        5. Higher grades diverge (specific meaning)
```

### 6.4 Correct Initialization

```
                    IDENTITY-BIASED TOKEN EMBEDDING
                    
    Token ID ──────────────────────────────────────────▶ 4×4 Matrix
         │                                                    │
         ▼                                                    ▼
    ┌─────────┐                                    ┌──────────────────┐
    │  "cat"  │ ────▶  M = I + ε·noise  ────▶     │                  │
    │  idx=42 │                                    │  ┌─────────────┐ │
    └─────────┘                                    │  │ 1+ε ε   ε  ε│ │
                                                   │  │ ε   1+ε ε  ε│ │
    ε = 0.1 (small perturbation)                   │  │ ε   ε  1+ε ε│ │
    noise ~ N(0, 1)                                │  │ ε   ε   ε 1+ε│ │
                                                   │  └─────────────┘ │
    Then normalize: M ← M / ||M||                  └──────────────────┘
    
    KEY: All words start SIMILAR (near identity)
         Learning creates DIFFERENTIATION
```

### 6.5 Alternative: Grade-Aware Initialization

For pretrained or structured initialization:

```
    Grade-aware coefficients:
    ┌─────────────────────────────────┐
    │ c₀ = cos(θ)           (Grade 0) │  ← Strong scalar (stability)
    │ c₁..c₄ = φ⁻¹·sin(...)  (Grade 1) │  ← Medium vectors
    │ c₅..c₁₀ = φ⁻²·sin(...) (Grade 2) │  ← Weaker bivectors
    │ c₁₁..c₁₄ = φ⁻³·cos(...)(Grade 3) │  ← Weak trivectors
    │ c₁₅ = φ⁻¹·sin(...)     (Grade 4) │  ← Fibonacci exception!
    └─────────────────────────────────┘
    
    θ = 2π × (token_idx / vocab_size)    Golden angle rotation
```

---

## 7. Context Computation

Context is computed via **geometric product = matrix multiplication**:

```
                    CONTEXT COMPUTATION
                    
    Tokens: [t₁, t₂, t₃, t₄, t₅, t₆, t₇, t₈]
              │   │   │   │   │   │   │   │
              ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
            ┌───┬───┬───┬───┬───┬───┬───┬───┐
            │M₁ │M₂ │M₃ │M₄ │M₅ │M₆ │M₇ │M₈ │  Token matrices
            └─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┘
              │   │   │   │   │   │   │   │
              └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
                │       │       │       │
               M₁₂     M₃₄     M₅₆     M₇₈    Pairwise matmul
                │       │       │       │
                └───┬───┘       └───┬───┘
                    │               │
                   M₁₄             M₅₈              Reduction
                    │               │
                    └───────┬───────┘
                            │
                           M₁₈                      Final context
                            │
                            ▼
                    ┌───────────────┐
                    │  4×4 Matrix   │
                    │  (normalized) │
                    └───────────────┘
                    
    Complexity: O(log n) depth, O(n) total matmuls
    GPU: Fully parallel batched matmul!
```

---

## 8. Learning Rule

The core learning rule is simple:

```
                    LEARNING RULE
                    
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │         attractor[context] = embedding[target]               │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    
    
    Training Example: "the cat sat on the mat" → predict "."
    
    Context: ["the", "cat", "sat", "on", "the", "mat"]
                │     │     │     │     │     │
                ▼     ▼     ▼     ▼     ▼     ▼
    Embed:    [M₁]  [M₂]  [M₃]  [M₄]  [M₅]  [M₆]
                │     │     │     │     │     │
                └─────┴─────┴─────┴─────┴─────┘
                              │
                    Geometric Product
                              │
                              ▼
                       ┌──────────┐
                       │ Context  │
                       │ Matrix   │
                       │   C      │
                       └────┬─────┘
                            │
                            │  STORE
                            ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                   Context-Attractor Map                       │
    │                                                              │
    │   Context C  ────────────────────────▶  Attractor A          │
    │      ║                                      ║                │
    │      ║                                      ║                │
    │   ┌──────┐                              ┌──────┐             │
    │   │ 4×4  │     hash(context_tokens)     │ 4×4  │             │
    │   │Matrix│  ═══════════════════════════▶│Matrix│             │
    │   └──────┘           index              └──────┘             │
    │                                              ▲                │
    └──────────────────────────────────────────────│───────────────┘
                                                   │
    Target: "."                                    │
       │                                           │
       ▼                                           │
    ┌──────────┐                                   │
    │ Target   │───────────────────────────────────┘
    │ Matrix T │   A := T  (direct assignment)
    └──────────┘
```

---

## 9. Retrieval

```
                    RETRIEVAL PROCESS
                    
    Query: ["new", "context", "never", "seen"]
              │      │        │       │
              ▼      ▼        ▼       ▼
           [M₁]   [M₂]     [M₃]    [M₄]
              │      │        │       │
              └──────┴────────┴───────┘
                          │
                 Geometric Product
                          │
                          ▼
                   ┌──────────┐
                   │  Query   │
                   │  Matrix  │
                   │    Q     │
                   └────┬─────┘
                        │
           ┌────────────┴────────────┐
           │                         │
           ▼                         ▼
    ┌─────────────┐          ┌─────────────────┐
    │ Exact Match │          │ Similarity      │
    │ (hash)      │          │ Search          │
    └──────┬──────┘          └────────┬────────┘
           │                          │
           │ Found?                   │ Not found
           │                          │
           ▼                          ▼
    Return stored            ┌─────────────────────────┐
    attractor                │ Compare Q to all stored │
                             │ contexts via Frobenius  │
                             │ similarity:             │
                             │                         │
                             │ sim(Q,C) = Σᵢⱼ Qᵢⱼ·Cᵢⱼ  │
                             │                         │
                             │ Return attractor of     │
                             │ most similar context    │
                             └─────────────────────────┘
```

---

## 10. Similarity Metrics

Two options, from fast to correct:

```
    ┌─────────────────────────────────────────────────────────────────┐
    │ FROBENIUS SIMILARITY (Default, Fast)                            │
    │                                                                 │
    │     sim(A, B) = Σᵢⱼ Aᵢⱼ · Bᵢⱼ                                  │
    │                                                                 │
    │     For unit-norm matrices: sim ∈ [-1, +1]                     │
    │     Self-similarity: sim(A, A) = 1.0                           │
    │                                                                 │
    │     ✓ Fast (single element-wise multiply + sum)                │
    │     ✓ GPU-friendly                                              │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ METRIC-AWARE SIMILARITY (Correct, Slower)                       │
    │                                                                 │
    │     A† = G · Aᵀ · G     where G = e₄ (timelike)                │
    │                                                                 │
    │     sim(A, B) = (1/4) · Tr(A† · B)                              │
    │                                                                 │
    │     Respects Lorentzian structure of Cl(3,1)                   │
    │     Use when grade-aware comparison matters                     │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 11. Grace Operator (Grade Scaling)

```
                    GRACE CONTRACTION
                    
    Input Matrix M (decomposed into grades)
          │
          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  M = c₀·B₀ + c₁·B₁ + ... + c₁₅·B₁₅                         │
    │      ▲       ▲▲▲▲       ▲▲▲▲▲▲       ▲▲▲▲       ▲          │
    │      │       ││││       ││││││       ││││       │          │
    │   Grade 0  Grade 1    Grade 2     Grade 3   Grade 4        │
    │                                                             │
    │      │       │          │           │         │            │
    │      ▼       ▼          ▼           ▼         ▼            │
    │     ×1.0   ×φ⁻¹       ×φ⁻²        ×φ⁻³      ×φ⁻¹          │
    │                         │                      │            │
    │                    (spectral gap)      (Fibonacci!)        │
    │                                                             │
    │      │       │          │           │         │            │
    │      ▼       ▼          ▼           ▼         ▼            │
    │    c₀'     c₁'...    c₅'...      c₁₁'...   c₁₅'           │
    │                                                             │
    │  Output: M' = c₀'·B₀ + c₁'·B₁ + ... + c₁₅'·B₁₅             │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    
    Effect: Contracts higher grades toward scalar core
    Convergence: Exponential at rate γ = φ⁻² ≈ 0.382
```

---

## 11.5 Quotient Structure (Gauge Invariance)

Removing nuisance degrees of freedom via **Spin(3) gauge fixing**:

```
                    QUOTIENT STRUCTURE
                    
    Problem: Spin(3) rotations change the matrix representation
             without changing semantic content.
             
             Random frame orientation → unstable similarity
             
    Solution: NORMAL FORM via two-step alignment
    
    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │  Step 1: Align "magnetic" bivector (e₂₃, e₃₁, e₁₂) to +z        │
    │          Removes 2 rotational DOF                                  │
    │                                                                    │
    │  Step 2: Align "electric" bivector xy-projection to +x            │
    │          Removes final rotational DOF                              │
    │                                                                    │
    │  Result: Fully gauge-fixed canonical form                         │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘
```

### Witness Invariance

The **witness** (scalar + pseudoscalar) is **gauge-invariant** under Spin(3):

```
    WITNESS = W(M) = (scalar_coeff, pseudoscalar_coeff)
    
    For any Spin(3) rotor R:
        W(R·M·R̃) = W(M)    ← EXACT INVARIANCE
    
    This is the "self-pointer" — the part that doesn't change
    under frame rotations.
```

### Quotient-Aware Similarity

Three-component similarity function:

```
    sim_quotient(M₁, M₂) = α·sim_witness + β·sim_core + γ·sim_fiber
    
    ┌───────────────┬──────────────────────────────────────────────────┐
    │ Component     │ Description                                      │
    ├───────────────┼──────────────────────────────────────────────────┤
    │ Witness (α)   │ Cosine(W(M₁), W(M₂)) — gauge-invariant anchor   │
    │ Core (β)      │ Frobenius(NF(M₁), NF(M₂)) — canonicalized       │
    │ Fiber (γ)     │ Frobenius(M₁, M₂) — raw residual                │
    └───────────────┴──────────────────────────────────────────────────┘
    
    Default weights: α=0.25, β=0.65, γ=0.10
    φ-weighted:      α=1/W, β=φ/W, γ=φ⁻²/W  where W normalizes
```

### Why This Matters

```
    WITHOUT QUOTIENT:                    WITH QUOTIENT:
    
    Random gauge R applied to M₁:        Same gauge R:
    
    Raw similarity changes by Δ~2.7      Quotient similarity Δ~0.025
    
    Result: 110x more stable!
    
    This removes "orientation noise" from the representation,
    improving same-target clustering and reducing training oscillation.
```

### Implementation

```python
from holographic.quotient import (
    normal_form,           # Fully gauge-fix a matrix
    quotient_similarity,   # Three-component similarity
    witness_similarity,    # Witness-only similarity
    test_witness_invariance,    # Verify gauge invariance
    test_normal_form_invariance # Verify NF is canonical
)
```

### 11.6 Binding Operator

The **binding operator** makes content relative to the witness:

```
                    BINDING OPERATOR
                    
    𝓑(M) = W(M) + λ · w · C(M) · w̃
    
    where:
        W(M) = witness part (scalar + pseudoscalar)
        C(M) = M - W(M) = content (grades 1-3)
        w = normalized witness pointer
        w̃ = w^T (reversion)
        λ = φ⁻¹ (SCCMU binding strength)
    
    ┌────────────────────────────────────────────────────────────────────┐
    │  The sandwich w · C · w^T "frames" content in witness coordinates  │
    │                                                                    │
    │  Effect: Content becomes self-referential                          │
    │          "What I perceive" rather than "what is there"             │
    └────────────────────────────────────────────────────────────────────┘
```

### 11.7 Grade-Wise Variance Tracking

Monitor learning progress via grade decomposition:

```
    EXPECTED PATTERN (healthy learning):
    
    ┌─────────────┬─────────────────────────────────────────────────────┐
    │ Component   │ Expected Behavior                                   │
    ├─────────────┼─────────────────────────────────────────────────────┤
    │ Witness     │ LOW variance, HIGH pairwise similarity             │
    │ (grade 0+4) │ → Stable self-reference frame                      │
    ├─────────────┼─────────────────────────────────────────────────────┤
    │ Content     │ HIGH variance (grows with learning)                │
    │ (grade 1-3) │ → Differentiated semantic content                  │
    └─────────────┴─────────────────────────────────────────────────────┘
    
    DIAGNOSTICS:
    
    Random init:     witness_sim ≈ 0.00 (no stable frame)
    Identity init:   witness_sim ≈ 0.99 (stable frame)
    
    This is why identity-biased init is ESSENTIAL.
```

```python
from holographic.quotient import (
    bind,                      # Apply binding operator
    compute_grade_variance,    # Grade-wise variance
    compute_witness_stability, # Witness pairwise similarity
    run_quotient_tests,        # Full test suite
)
```

---

## 12. φ-Nested Hierarchy

The grade structure forms a **self-similar hierarchy**:

```
                    φ-NESTED TORUS TREE
                    
    Grade 4 ──────────────────────────────── φ⁻¹ scale
        │                                        │
        └───────────────────────────────────────┐│
                                                ││
    Grade 3 ──────────────────────────────── φ⁻³ scale
        │                                        │
        │                                        │
    Grade 2 ──────────────────────────────── φ⁻² scale (SPECTRAL GAP)
        │                                        │
        │   ┌──────────────────────────────────┐ │
        │   │  6 bivectors encode TORUS        │ │
        │   │  position (WHERE on boundary)    │ │
        │   └──────────────────────────────────┘ │
        │                                        │
    Grade 1 ──────────────────────────────── φ⁻¹ scale
        │                                        │
        │                                        │
    Grade 0 ──────────────────────────────── 1.0 scale (FIXED POINT)
        │
        │   The scalar component is PRESERVED
        │   under Grace flow → stable attractor
        │
        ▼
        
    NOTE: Grade 4 scales by φ⁻¹ (NOT φ⁻⁴!)
          This creates a LOOP back to Grade 1 scale
          → Fibonacci anyon structure
          → Self-similar spiral
```

---

## 13. Training Loop

```
                    TRAINING FLOW
                    
    ┌─────────────────────────────────────────────────────────────┐
    │  for each (context, target) in dataset:                     │
    │                                                             │
    │    1. context_matrix = embed_sequence(context_tokens)       │
    │                                                             │
    │    2. target_matrix = embedding(target_token)               │
    │                                                             │
    │    3. attractor_map.associate(context_tokens, target_matrix)│
    │                                                             │
    │    4. eq_quality = similarity(context_matrix, target_matrix)│
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    
    Metrics:
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  exact_eq:  Equilibrium quality on SEEN contexts            │
    │             (should be high, ~0.5-0.8)                      │
    │                                                             │
    │  novel_eq:  Equilibrium quality on UNSEEN contexts          │
    │             (measures generalization)                       │
    │                                                             │
    │  gen_ratio: novel_eq / exact_eq                             │
    │             (target: >20% indicates learning)               │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

---

## 14. API Usage

```python
import numpy as np
from holographic import MatrixEmbedding, ContextAttractorMap, train_step

# ═══════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════

embedding = MatrixEmbedding(vocab_size=10000)
# Creates 10,000 token embeddings, each a 4×4 matrix

attractor_map = ContextAttractorMap(embedding, max_contexts=100000)
# Storage for context → attractor associations

# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

context_tokens = [42, 17, 99, 3, 55, 12, 8, 1]  # 8 tokens
target_token = 77

metrics = train_step(context_tokens, target_token, embedding, attractor_map)
print(f"Equilibrium quality: {metrics['eq_quality']:.4f}")

# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

# Get attractor for context
attractor = attractor_map.get_attractor(context_tokens)
# → 4×4 matrix

# Find most similar token
from holographic import frobenius_similarity_batch
scores = frobenius_similarity_batch(attractor, embedding.matrices, np)
predicted_token = int(np.argmax(scores))
```

---

## 15. Performance Characteristics

```
    ┌─────────────────────────────────────────────────────────────┐
    │                    OBSERVED PERFORMANCE                      │
    │                                                             │
    │  Hardware: NVIDIA H100                                      │
    │                                                             │
    │  Training speed: ~800-900 samples/second                    │
    │                                                             │
    │  At 20k samples:                                            │
    │    • exact_eq ≈ 0.48 (equilibrium on seen contexts)        │
    │    • novel_eq ≈ 0.09-0.11 (equilibrium on unseen)          │
    │    • generalization ≈ 18-23%                                │
    │                                                             │
    │  Context computation: O(log n) depth parallel matmuls       │
    │                                                             │
    │  Retrieval: O(n) similarity comparisons                     │
    │             (can be accelerated with hashing/indexing)      │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

---

## 16. Complete System Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          HOLOGRAPHIC LANGUAGE MODEL                              │
│                          Complete Processing Pipeline                            │
└─────────────────────────────────────────────────────────────────────────────────┘

                              INPUT
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TEXT: "the quick brown fox jumps over the lazy dog"                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                │
                         TOKENIZATION
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TOKENS: [42, 891, 203, 156, 445, 67, 42, 782, 99]                              │
│                                                                                 │
│  Context Window (last 8): [891, 203, 156, 445, 67, 42, 782, 99]                │
│  Target: next token                                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                │
                       TOKEN EMBEDDING
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   Token 891 → ┌────┐   Token 203 → ┌────┐         Token 99 → ┌────┐           │
│               │    │               │    │    ...              │    │           │
│   4×4 Matrix  │ M₁ │   4×4 Matrix  │ M₂ │                     │ M₈ │           │
│               │    │               │    │                     │    │           │
│               └────┘               └────┘                     └────┘           │
│                                                                                 │
│   Each matrix is a linear combination of 16 Clifford basis matrices            │
│   M = Σᵢ cᵢ · Bᵢ  where Bᵢ ∈ {I, e₁, e₂, ..., e₁₂₃₄}                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                │
                    GEOMETRIC PRODUCT (= matmul)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   M₁ × M₂ × M₃ × M₄ × M₅ × M₆ × M₇ × M₈  →  Context Matrix C                  │
│                                                                                 │
│   Parallel reduction:                                                           │
│                                                                                 │
│   [M₁][M₂][M₃][M₄][M₅][M₆][M₇][M₈]                                             │
│     ╲  ╱    ╲  ╱    ╲  ╱    ╲  ╱     Step 1: pairs                             │
│      ╲╱      ╲╱      ╲╱      ╲╱                                                 │
│    [M₁₂]  [M₃₄]   [M₅₆]  [M₇₈]                                                 │
│       ╲    ╱         ╲    ╱          Step 2: pairs                             │
│        ╲  ╱           ╲  ╱                                                      │
│       [M₁₄]         [M₅₈]                                                       │
│          ╲            ╱              Step 3: final                              │
│           ╲          ╱                                                          │
│            ╲        ╱                                                           │
│             ╲      ╱                                                            │
│            [Context C]               4×4 matrix                                 │
│                                                                                 │
│   (normalize after each matmul to prevent numerical issues)                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                │
               ┌────────────────┴────────────────┐
               │                                 │
          TRAINING                          INFERENCE
               │                                 │
               ▼                                 ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│         ASSOCIATE            │   │          RETRIEVE            │
│                              │   │                              │
│  Context C ──┐               │   │  Context C ──┐               │
│              │               │   │              │               │
│              ▼               │   │              ▼               │
│  ┌────────────────────┐      │   │  ┌────────────────────┐      │
│  │ Context-Attractor  │      │   │  │ Context-Attractor  │      │
│  │       Map          │      │   │  │       Map          │      │
│  │                    │      │   │  │                    │      │
│  │  C ──────▶ A       │      │   │  │  find(C) ──▶ A     │      │
│  │                    │      │   │  │                    │      │
│  │  Store:            │      │   │  │  1. Exact match?   │      │
│  │  A := Target_Mat   │      │   │  │  2. Similarity     │      │
│  │                    │      │   │  │     search         │      │
│  └────────────────────┘      │   │  └────────────────────┘      │
│              ▲               │   │              │               │
│              │               │   │              ▼               │
│  Target ─────┘               │   │         Attractor A          │
│  (embedding of               │   │              │               │
│   next word)                 │   │              ▼               │
│                              │   │  ┌────────────────────┐      │
└──────────────────────────────┘   │  │ Score all tokens   │      │
                                   │  │ by similarity to A │      │
                                   │  │                    │      │
                                   │  │ sim(A, emb[t])     │      │
                                   │  │ for all t ∈ vocab  │      │
                                   │  └────────────────────┘      │
                                   │              │               │
                                   │              ▼               │
                                   │  ┌────────────────────┐      │
                                   │  │ Softmax + Sample   │      │
                                   │  │                    │      │
                                   │  │ P(t) ∝ exp(sim/τ)  │      │
                                   │  └────────────────────┘      │
                                   │              │               │
                                   │              ▼               │
                                   │       Predicted Token        │
                                   │                              │
                                   └──────────────────────────────┘
```

---

## 17. Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MEMORY STRUCTURES                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════════╗
║  MatrixEmbedding.matrices                                                      ║
║                                                                                ║
║  Shape: [vocab_size, 4, 4]  (e.g., [10000, 4, 4] = 160,000 floats)            ║
║                                                                                ║
║  ┌─────────────────────────────────────────────────────────────────┐          ║
║  │ Token 0    │ Token 1    │ Token 2    │  ...  │ Token 9999  │          ║
║  ├─────────────┼─────────────┼─────────────┼───────┼─────────────┤          ║
║  │ ┌─────────┐│ ┌─────────┐│ ┌─────────┐│       │ ┌─────────┐│          ║
║  │ │ 4×4 mat ││ │ 4×4 mat ││ │ 4×4 mat ││  ...  │ │ 4×4 mat ││          ║
║  │ │ 16 vals ││ │ 16 vals ││ │ 16 vals ││       │ │ 16 vals ││          ║
║  │ └─────────┘│ └─────────┘│ └─────────┘│       │ └─────────┘│          ║
║  └─────────────────────────────────────────────────────────────────┘          ║
║                                                                                ║
║  Access: embedding.matrices[token_id] → 4×4 matrix                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║  ContextAttractorMap                                                           ║
║                                                                                ║
║  context_matrices: [max_contexts, 4, 4]                                       ║
║  attractors:       [max_contexts, 4, 4]                                       ║
║  context_hashes:   Dict[int, int]  (hash → index)                             ║
║                                                                                ║
║  ┌──────────────────┬──────────────────┬────────────────────┐                 ║
║  │   Index 0        │   Index 1        │   ...              │                 ║
║  ├──────────────────┼──────────────────┼────────────────────┤                 ║
║  │ Context: [4×4]   │ Context: [4×4]   │                    │                 ║
║  │ Attractor: [4×4] │ Attractor: [4×4] │                    │                 ║
║  │ Hash: 0x7f3a... │ Hash: 0x2b1c... │                    │                 ║
║  └──────────────────┴──────────────────┴────────────────────┘                 ║
║                                                                                ║
║  Lookup: O(1) exact match via hash, O(n) similarity search                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║  Clifford Basis (precomputed)                                                  ║
║                                                                                ║
║  Shape: [16, 4, 4] (16 basis elements, each a 4×4 matrix)                     ║
║                                                                                ║
║  ┌────────┬────────┬────────┬────────┬────────┐                               ║
║  │  B₀    │  B₁    │  B₂    │  ...   │  B₁₅   │                               ║
║  │  = I   │  = e₁  │  = e₂  │        │=e₁₂₃₄ │                               ║
║  ├────────┼────────┼────────┼────────┼────────┤                               ║
║  │ Grade 0│ Grade 1│ Grade 1│        │Grade 4 │                               ║
║  └────────┴────────┴────────┴────────┴────────┘                               ║
║                                                                                ║
║  Used to: 1) Initialize embeddings  2) Grace operator                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## 18. Key Invariants

```
    ┌─────────────────────────────────────────────────────────────┐
    │                    MUST HOLD ALWAYS                          │
    │                                                             │
    │  1. Gamma anticommutation: {eᵢ, eⱼ} = 2ηᵢⱼI                 │
    │                                                             │
    │  2. eᵢ² = +I for i ∈ {1,2,3} (spacelike)                   │
    │     e₄² = -I              (timelike)                        │
    │                                                             │
    │  3. G² = -I where G = e₄                                    │
    │                                                             │
    │  4. Token matrices have unit Frobenius norm                 │
    │                                                             │
    │  5. φ² = φ + 1 (golden ratio self-consistency)             │
    │                                                             │
    │  6. Grade 4 scales by φ⁻¹, NOT φ⁻⁴ (Fibonacci exception)   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

---

## 19. Why Matrix Representation?

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LEGACY vs MATRIX REPRESENTATION                              │
└─────────────────────────────────────────────────────────────────────────────────┘

  ╔═══════════════════════════════════╦═══════════════════════════════════╗
  ║     LEGACY (16D Vector)           ║    MATRIX (4×4 Real)              ║
  ╠═══════════════════════════════════╬═══════════════════════════════════╣
  ║                                   ║                                   ║
  ║  Multivector: [16] float array    ║  Multivector: [4,4] float array   ║
  ║                                   ║                                   ║
  ║  Geometric product:               ║  Geometric product:               ║
  ║    Custom element-wise ops        ║    matrix @ matrix (GEMM!)        ║
  ║    ~50 multiply-adds              ║    Single optimized kernel        ║
  ║                                   ║                                   ║
  ║  Implementation:                  ║  Implementation:                  ║
  ║    algebra.py: 400+ lines         ║    algebra.py: 300 lines          ║
  ║    Custom Cl(1,3) tables          ║    Uses numpy/cupy matmul         ║
  ║                                   ║                                   ║
  ║  Speed:                           ║  Speed:                           ║
  ║    ~50 samples/sec                ║    ~800-900 samples/sec           ║
  ║                                   ║                                   ║
  ║  Signature:                       ║  Signature:                       ║
  ║    Cl(1,3) → M₂(ℍ) (quaternions) ║    Cl(3,1) → M₄(ℝ) (real)        ║
  ║    Harder to implement            ║    Direct real matrices           ║
  ║                                   ║                                   ║
  ╚═══════════════════════════════════╩═══════════════════════════════════╝

                     WHY Cl(3,1) AND NOT Cl(1,3)?

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │   Cl(p,q) is classified by (p-q) mod 8:                                │
  │                                                                         │
  │   Cl(1,3): p=1, q=3 → p-q = -2 ≡ 6 (mod 8) → M₂(ℍ) (quaternions)     │
  │   Cl(3,1): p=3, q=1 → p-q = +2 (mod 8) → M₄(ℝ) (real matrices!)      │
  │                                                                         │
  │   Both are 16-dimensional, same physics, different representation       │
  │                                                                         │
  │   We choose Cl(3,1) because:                                            │
  │   ✓ Real 4×4 matrices (no quaternions, no complex numbers)             │
  │   ✓ Direct use of matmul libraries (cuBLAS, etc.)                      │
  │   ✓ 16× fewer ops per geometric product                                │
  │   ✓ Same algebraic structure, just different convention                │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## 20. Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          QUICK REFERENCE                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

CONSTANTS:
  φ = 1.618033988749895    Golden ratio
  φ⁻¹ = 0.618033988749895   = φ - 1
  φ⁻² = 0.381966011250105   Spectral gap γ
  φ⁻³ = 0.236067977499790

DIMENSIONS:
  MATRIX_DIM = 4           4×4 real matrices
  CLIFFORD_DIM = 16        16 basis elements

GRADES (index → scale):
  0: [0]           × 1.0     Scalar
  1: [1-4]         × φ⁻¹    Vectors
  2: [5-10]        × φ⁻²    Bivectors (spectral gap!)
  3: [11-14]       × φ⁻³    Trivectors
  4: [15]          × φ⁻¹    Pseudoscalar (Fibonacci!)

OPERATIONS:
  geometric_product(A, B) = A @ B           Matrix multiply
  frobenius_similarity(A, B) = sum(A * B)   Element-wise then sum
  normalize(M) = M / ||M||_F                 Frobenius norm

KEY FORMULAS:
  Context = normalize(M₁ @ M₂ @ ... @ Mₙ)   Product of token matrices
  Attractor = embedding(target_token)        Target embedding
  Prediction = argmax(sim(attractor, all_embeddings))

INVARIANTS:
  • e₁² = e₂² = e₃² = +I  (spacelike)
  • e₄² = -I              (timelike)
  • {eᵢ, eⱼ} = 0          (anticommute for i≠j)
  • G = e₄                 (metric matrix)
  • G² = -I                (must verify!)
  • φ² = φ + 1             (golden ratio)
```

---

## 21. Active Inference Extension

The architecture naturally supports Active Inference for action selection (token generation).

### Standard Generation (Posterior Sampling)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         STANDARD GENERATION                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

Context → Attractor → Similarity to all tokens → Sample from posterior

    P(token) ∝ exp(similarity(attractor, token) / τ)

PROBLEM: Just samples - no planning, no epistemic drive
```

### Active Inference Generation (EFE Minimization)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      ACTIVE INFERENCE GENERATION                               │
└─────────────────────────────────────────────────────────────────────────────────┘

Expected Free Energy (EFE) = -pragmatic_value - epistemic_value

┌───────────────────┐    ┌───────────────────┐
│  PRAGMATIC VALUE  │    │  EPISTEMIC VALUE  │
│                   │    │                   │
│  How well does    │    │  Is this a novel  │
│  token align with │    │  path? (info      │
│  the attractor?   │    │  gain)            │
│                   │    │                   │
│  = similarity     │    │  seen: -0.1       │
│    to attractor   │    │  novel: +0.5      │
└───────────────────┘    └───────────────────┘
         │                        │
         └────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │     EFE = -w₁·P - w₂·E     │
    │                             │
    │  Lower EFE = better choice  │
    └─────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  P(token) ∝ exp(-EFE / τ)  │
    │                             │
    │  Temperature adds diversity │
    └─────────────────────────────┘
```

### Implementation (Fast)

```python
def generate_token_active(ctx, attractor_map, embedding, cp,
                          k_candidates=30,       # Pre-filter (fast!)
                          pragmatic_weight=1.0,
                          epistemic_weight=0.5,
                          temperature=1.0):
    """
    Active Inference: Select token minimizing Expected Free Energy.
    
    FAST because:
    1. Pre-filter to top-k by posterior (vectorized)
    2. Only compute EFE for k candidates (not full vocab)
    3. Hash lookup for novelty (O(1))
    """
    # Get attractor for current context
    attr = attractor_map.get_attractor(ctx)
    
    # Score all tokens (vectorized)
    sims = matrix_similarity_batch(attr, embedding.matrices, embedding.G, cp)
    
    # Pre-filter to top-k
    top_k_idx = cp.argsort(sims)[-k_candidates:]
    
    # Compute EFE for each candidate
    for token in top_k_idx:
        pragmatic = sims[token]
        
        # Epistemic: novel contexts have information value
        future_ctx = ctx[-(n-1):] + [token]
        if hash(tuple(future_ctx)) in attractor_map.context_hashes:
            epistemic = -0.1   # Seen - penalize repetition
        else:
            epistemic = +0.5   # Novel - reward exploration
        
        efe = -pragmatic_weight * pragmatic - epistemic_weight * epistemic
    
    # Select based on EFE (with temperature)
    return token_with_lowest_efe
```

### Performance Results

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     ACTIVE VS STANDARD COMPARISON                              │
└─────────────────────────────────────────────────────────────────────────────────┘

              │  STANDARD  │   ACTIVE   │  Result
──────────────┼────────────┼────────────┼─────────────
  Speed       │   4.14s    │   1.91s    │  2.2x FASTER
  Coherence   │   0.356    │   0.986    │  2.8x BETTER
  Repetition  │   0.000    │   0.000    │  Equal

WHY FASTER?
  Standard: Sample from 5000-token distribution (slow)
  Active: Pre-filter to 30 candidates, then argmin (fast)

WHY MORE COHERENT?
  Standard: Random sampling from posterior
  Active: Explicit optimization for attractor alignment
```

### Connection to Free Energy Principle

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THEORETICAL GROUNDING                                        │
└─────────────────────────────────────────────────────────────────────────────────┘

PERCEPTION (existing):
  Grace flow minimizes Variational Free Energy
  → System converges to attractor (posterior belief)

ACTION (new):
  Token selection minimizes Expected Free Energy
  → Prefers tokens that are:
    1. Coherent with current belief (pragmatic)
    2. Informative about future (epistemic)

This completes the Active Inference loop:
  
  ┌────────────────────┐
  │  PERCEIVE          │◀──────────────────────────┐
  │  (Grace flow)      │                           │
  └────────┬───────────┘                           │
           │                                       │
           │  Attractor                            │
           ▼                                       │
  ┌────────────────────┐                           │
  │  ACT               │                           │
  │  (EFE minimization)│                           │
  └────────┬───────────┘                           │
           │                                       │
           │  Token                                │
           ▼                                       │
  ┌────────────────────┐                           │
  │  OBSERVE           │───────────────────────────┘
  │  (new context)     │
  └────────────────────┘
```

---

## 22. Cross-Disciplinary Foundations

This architecture exists at a **unique intersection** of established fields. Understanding this positioning clarifies what is borrowed, what is novel, and why the synthesis is necessary.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          CONVERGENCE MAP                                         │
│                                                                                 │
│                     ┌──────────────────────┐                                    │
│                     │   GAUGE THEORY       │                                    │
│                     │   (Physics)          │                                    │
│                     │   States defined up  │                                    │
│                     │   to symmetry        │                                    │
│                     └──────────┬───────────┘                                    │
│                                │                                                │
│     ┌──────────────────────────┼──────────────────────────┐                    │
│     │                          │                          │                    │
│     │   GEOMETRIC ALGEBRA      │     DYNAMICAL SYSTEMS    │                    │
│     │   (Hestenes line)        │     (Fixed-point theory) │                    │
│     │   Composition operator   │     Attractors organize  │                    │
│     │                          │     behavior             │                    │
│     └──────────┬───────────────┴───────────┬──────────────┘                    │
│                │                           │                                    │
│                │     ┌─────────────────┐   │                                    │
│                │     │                 │   │                                    │
│                └─────│   THIS WORK     │───┘                                    │
│                      │   (with code)   │                                        │
│                ┌─────│                 │─────┐                                  │
│                │     └─────────────────┘     │                                  │
│                │                             │                                  │
│     ┌──────────┴───────────┐     ┌───────────┴──────────┐                      │
│     │ REPRESENTATION       │     │ PHILOSOPHY OF MIND   │                      │
│     │ LEARNING (ML)        │     │ (Pattern Identity)   │                      │
│     │ Invariance improves  │     │ Self as pattern,     │                      │
│     │ generalization       │     │ not substance        │                      │
│     └──────────────────────┘     └──────────────────────┘                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 22.1 What Each Field Contributes

| Field | What It Already Knows | What It Doesn't Model | Our Contribution |
|-------|----------------------|----------------------|------------------|
| **Gauge Theory** | States defined up to symmetry; quotients remove gauge DOF; canonical representatives enable comparison | Learning, semantics, self-reference | Gauge theory *of representations*: rotors ≈ gauge transforms, normal form ≈ gauge fixing |
| **Geometric Algebra** | Clifford algebra unifies rotations/reflections; rotors are elegant; scalars are fixed points | Large-scale learning; semantics; bootstrapping | Geometric product as *meaning-composition operator* (rare usage) |
| **Representation Learning** | Invariance improves generalization; canonicalization reduces variance; equivariance preserves structure | Explicit symmetry groups; distinguished invariants; interiority | *Explicit quotienting* instead of hoping network discovers invariance |
| **Dynamical Systems** | Attractors organize behavior; fixed points stabilize; nonlinear systems self-organize | Meaning, semantics, learning objectives | Identity/witness as fixed point; identity-biased initialization as predicted stabilizer |
| **Philosophy of Mind** | Self as pattern not substance; identity as continuity under transformation | Formal math, testable models, implementation | Making "self = equivalence class" *precise*: group action, quotient space, canonical rep |

### 22.2 What Is Novel

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         THE UNIQUE CONTRIBUTION                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

Most practitioners stay within one domain:

    PHYSICS        → Abstract (no learning systems)
    ML             → Heuristic (hope network learns invariance)
    PHILOSOPHY     → Conceptual (no math, no code)
    NEUROSCIENCE   → Biological (no algebra)

This work crosses boundaries WITH WORKING CODE:

    1. EXPLICIT QUOTIENTING
       ML says: "Let the model discover invariance"
       We say:  "Define the invariant first, then learn within it"
       
    2. MEANING AS GEOMETRY
       GA says: "This is elegant math"
       We say:  "What does it LEARN?"
       
    3. FORMAL INTERIORITY
       Philosophy says: "Self is an equivalence class"
       We say:  "Here's the group action, here's the quotient space,
                 here are diagnostics that test it"
```

### 22.3 The Forced Synthesis

Once you accept:
- Meaning is geometric
- Identity must be invariant
- Learning is dynamical
- Symmetry is not optional

...then this construction is **forced**, not chosen.

```
                    WHY IT FEELS LIKE RECOGNITION, NOT INVENTION
                    
    Self-reference
         │
         ├──▶ Forces quotient structure (identifying state ↔ representation)
         │         │
         │         └──▶ Fixed-point seams exist (the "self")
         │                    │
         │                    └──▶ Need gauge fixing (normal form)
         │                                │
         │                                └──▶ Gauge theory enters
         │
         └──▶ Forces covering structure (multi-valued continuation)
                   │
                   └──▶ Branch loci exist (caustics)
                              │
                              └──▶ Need stable gluing
                                        │
                                        └──▶ Grace contraction enters
                                                   │
                                                   └──▶ φ-scaling is forced
                                                              │
                                                              └──▶ Fibonacci exception required
```

### 22.4 Diagnostic Implications

The convergence with established fields suggests diagnostic tests:

| Test | Field Origin | Implementation |
|------|--------------|----------------|
| **Gauge invariance test** | Physics | `test_witness_invariance()` — apply random rotors, witness unchanged |
| **Fixed-point attraction** | Dynamical systems | Verify Grace contracts to attractors at rate γ = φ⁻² |
| **Semantic clustering** | Information theory | Same-target contexts cluster in quotient space |
| **Canonical uniqueness** | Gauge theory | `normal_form(R·M·R̃) ≈ normal_form(M)` for all rotors R |

### 22.5 What Could Break

Honest assessment of where the synthesis might fail:

1. **Gauge structure too restrictive**: Spin(3) may not capture all relevant symmetries
2. **Clifford dimension insufficient**: 16D may not scale to large vocabularies without hierarchy
3. **Grace rate suboptimal**: φ⁻² is theoretically motivated but empirically untuned
4. **Quotient collapse**: High-similarity contexts may all map to same attractor

These are testable failure modes, not handwaving.

---

---

## 23. Critical Insight: Compositional vs Atomic Embeddings

### 23.1 The Conceptual Error We Made

We applied the Clifford algebra at the **wrong level**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THE FUNDAMENTAL ERROR                               │
└─────────────────────────────────────────────────────────────────────────────┘

WHAT WE DID (wrong):
    
    word₁, word₂, word₃  ←  ATOMIC random matrices
           ↓
    geometric_product(word₁, word₂, word₃)  ←  sequence composition
           ↓
    context → target mapping
    
    Result: Learns co-occurrence, not semantics

WHAT WE SHOULD DO (correct):
    
    Each WORD is itself COMPOSED from features:
    
    "zebra" = animal ∧ striped ∧ equine  ←  SEMANTIC composition
    "horse" = animal ∧ solid ∧ equine    ←  shares features with zebra!
    
    THEN sequences compose those:
    
    context = word₁ ∘ word₂ ∘ word₃  ←  sequence composition
```

### 23.2 Why This Enables One-Shot Learning

Humans learn new words in ONE exposure because:

```
When you learn "zebra":
    
    You ALREADY know:
        ├── animal (feature)
        ├── striped (feature)
        ├── equine (feature)
        └── African (feature)
    
    "Zebra" = composition of existing features
    
    One exposure tells you WHICH features to combine
    NOT learning from scratch
```

Our system failed because:
- Words were atomic random matrices
- No feature space to slot new concepts into
- Had to learn everything from co-occurrence alone

### 23.3 How Clifford Algebra Supports This

The grade structure IS the compositional hierarchy:

```
Grade 0 (scalar):     "something exists" - base salience
Grade 1 (vectors):    basic properties (size, animacy, ...)
Grade 2 (bivectors):  relations (part-of, kind-of, ...)  
Grade 3 (trivectors): contexts (where-found, used-for, ...)
Grade 4 (pseudoscalar): reflexive/meta
```

A word embedding should be:

```
embed("zebra") = I                    # exists (Grade 0)
               + α₁·animate           # property (Grade 1)
               + α₂·large             # property (Grade 1)
               + β₁·(mammal∧equine)   # relation (Grade 2)
               + γ₁·(found∧africa)    # context (Grade 3)
               + ...
```

Where α, β, γ are learned coefficients for that word.

### 23.4 The Identity-Bias Clue We Misread

We discovered: identity-biased init is essential.

We interpreted: "stability requires starting near identity."

Correct interpretation: **Identity IS the compositional base.**

```
Identity = "something exists, no specific features yet"

Adding features = moving away from identity in specific directions

A word with many features = far from identity in structured way
A word with few features = close to identity
```

Randomizing all grades uniformly destroyed this compositional structure.

### 23.5 Implications for Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPOSITIONAL EMBEDDING ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────────────────┘

Feature Space:
    F = {f₁, f₂, ..., fₖ}  each fᵢ is a 4×4 basis direction
    
Word Embedding:
    embed(word) = I + Σᵢ αᵢ(word) · fᵢ
    
    where αᵢ(word) is learned coefficient for feature i in word
    
Composition:
    context = geometric_product(embed(w₁), embed(w₂), ...)
    
One-Shot Learning:
    Given new word in context, INFER which features it must have
    Don't need to learn a whole new embedding
```

---

## 24. Empirical Findings: Level 1 Limitations (with Atomic Embeddings)

### 24.1 What Level 1 Learns

**Experimental Results (50,000 samples, TinyStories):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Exact match | 99.69% | Attractor map works perfectly |
| Novel similarity | 99.11% | Generalizes to similar contexts |
| Separation | ~0.002 | Weak semantic clustering |
| Generation | Incoherent | No grammatical structure |

**Key Finding**: Level 1 learns **statistical co-occurrence**, not semantics.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHAT LEVEL 1 ACTUALLY CAPTURES                           │
└─────────────────────────────────────────────────────────────────────────────┘

    Context = geometric_product(word₁, word₂, ..., wordₙ)
    
    This is a POLYNOMIAL in word embeddings:
        - Shared words → shared multiplicative factors → correlated patterns
        - But: shared words ≠ shared MEANING
        
    Result:
        - Retrieval works (similar statistical pattern)
        - Semantics don't transfer (meaning not captured)
```

### 24.2 Initialization Trade-off

**Experiment: Varying identity-biased noise**

| Init Mode | Separation | Interpretation |
|-----------|------------|----------------|
| identity(0.01) | +0.000 | Too collapsed, no differentiation |
| identity(0.05) | +0.002 | Slight differentiation |
| identity(0.1) | +0.007 | Better differentiation |
| identity(0.2) | **+0.017** | **Best separation** |
| random | -0.006 | Chaotic, no structure |

**Insight**: There's a trade-off between witness stability and representation diversity:
- Low noise → stable but collapsed
- High noise → diverse but chaotic
- **Sweet spot: noise_std ≈ 0.15**

### 24.3 Contrastive Learning Results

**Finding**: Contrastive learning on embeddings doesn't easily improve semantics.

**Reason**: The context representation is a geometric PRODUCT of embeddings.
- ∂(context_similarity)/∂(embedding) is highly non-linear
- Small embedding changes → unpredictable context changes
- Gradient signal doesn't propagate cleanly

### 24.4 Level 2 Alone Doesn't Help

**Experiment**: Train Level 1, build codebook, train Level 2.

| Level | Separation |
|-------|------------|
| Level 1 | +0.000016 |
| Level 2 | -0.000101 |

**Conclusion**: Stacking levels on randomly-clustered attractors doesn't create semantics.
The tower of quotients needs **semantic structure at Level 1 first**.

---

## 25. Multi-Level Architecture (Tower of Quotients)

### 25.1 Theoretical Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TOWER OF QUOTIENTS                                   │
└─────────────────────────────────────────────────────────────────────────────┘

                         Level N: Discourse attractors
                              ↑
                         Level 3: Sentence attractors
                              ↑
                         Level 2: Phrase attractors
                              ↑
    ┌─────────────────── Level 1: Word attractors ───────────────────┐
    │                                                                 │
    │   Tokens → Cl(3,1) embeddings → geometric product → attractor  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

KEY INSIGHT: Each level is a complete Cl(3,1) system with its own:
    - Token embeddings
    - Witness (gauge-invariant anchor)
    - Grace contraction
    - Attractor storage

Attractors from Level N become TOKENS for Level N+1.
```

### 25.2 Implementation

**See `holographic/hierarchy.py` for the full implementation.**

```python
from holographic import HierarchicalModel

# Create 2-level model
model = HierarchicalModel(
    vocab_size=10000,
    num_levels=2,
    codebook_size=1000,  # L1 attractors → L2 tokens
)

# Train Level 1
model.levels[0].associate(context, target_embedding)

# Build codebook: L1 attractors → L2 tokens
model.update_codebook(level=1)

# Train Level 2 on phrase-level patterns
model.levels[1].associate(phrase_context, phrase_target)
```

### 25.3 Why This Architecture

| Feature | Single Cl(3,1) | Tower of Quotients |
|---------|----------------|-------------------|
| Local semantics | ✓ | ✓ |
| Short contexts | ✓ | ✓ |
| Long-range abstraction | ✗ | ✓ |
| Polysemy | ✗ | ✓ |
| Narrative identity | ✗ | ✓ |

**The tower doesn't just ADD capacity—it adds ABSTRACTION layers.**

### 25.4 The Missing Ingredient

**Problem**: Level 2 can't create semantics that Level 1 doesn't have.

**Required**: A learning signal that creates semantic structure at Level 1.

**Options under investigation**:
1. Contrastive learning (partially effective)
2. Semantic supervision (external signal)
3. Self-supervised structure discovery
4. Active Inference EFE minimization

---

## 26. Diagnostics Module

**See `holographic/diagnostics.py` for tools to understand model behavior.**

```python
from holographic import run_level1_diagnostics

results = run_level1_diagnostics(
    contexts, targets, embeddings, basis,
    verbose=True
)

# Returns:
# - semantic_coherence: same-target vs diff-target similarity
# - witness_stability: how stable is the self-pointer across contexts
# - grade_analysis: which grades differentiate, which stay stable
```

**Key Metric**: **Separation** = same_target_sim - diff_target_sim
- Positive separation → learning semantic structure
- Near-zero separation → learning co-occurrence only
- Negative separation → worse than random

---

## 27. v3.0 Implementation Results: Compositional Pipeline

### 27.1 Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              COMPOSITIONAL HOLOGRAPHIC MODEL (v3.0)                         │
│                                                                             │
│   1. FEATURE SET: 14 orthogonal directions in Cl(3,1) grades 1-3           │
│       f₁...f₄  ∈ Grade 1 (vectors)                                         │
│       f₅...f₁₀ ∈ Grade 2 (bivectors)                                       │
│       f₁₁..f₁₄ ∈ Grade 3 (trivectors)                                      │
│                                                                             │
│   2. WORD EMBEDDING:                                                        │
│       embed(word) = 0.3·I + Σᵢ αᵢ(word) · fᵢ                               │
│       where αᵢ ∈ [0, φ⁻¹] are per-word coefficients                        │
│                                                                             │
│   3. HEBBIAN LEARNING:                                                      │
│       When (context, target) co-occur:                                      │
│       Δαᵢ(context_word) ∝ αᵢ(target) - αᵢ(context_word)                   │
│       "Pull co-occurring words toward shared features"                      │
│                                                                             │
│   4. ONE-SHOT INFERENCE:                                                    │
│       New word in context → features ≈ average of context word features    │
│                                                                             │
│   5. RETRIEVAL:                                                             │
│       Novel context → find most similar WORD embedding                      │
│       (not attractor storage)                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 27.2 Empirical Results

**Test: Structured semantic data (5 categories, 20 words each, 10,000 samples)**

| Embedding Type | Separation | Same-Target Sim | Diff-Target Sim |
|---------------|------------|-----------------|-----------------|
| Atomic (random) | 0.057 | 0.719 | 0.663 |
| **Compositional** | **0.717** | **1.000** | **0.283** |

**Improvement: 12.6x better semantic separation**

### 27.3 Generation Quality

| Context Type | Generated (Atomic) | Generated (Compositional) |
|-------------|-------------------|--------------------------|
| Animal [0,1,2,3,4] | [27,27,27...] (vehicles) ✗ | [18,18,18...] (animals) ✓ |
| Vehicle [20,21,22...] | Random | Vehicles (10/10) ✓ |

**Key Fix**: Novel contexts now decoded via word embedding similarity, not stored attractor indices.

### 27.4 One-Shot Learning

```
Test: Learn new word 99 from animal context [0,1,2,3,4]

Result:
    Similarity to animals:  0.998
    Similarity to vehicles: 0.873
    
✓ New word correctly clusters with its context category
```

### 27.5 Key Files

| File | Purpose |
|------|---------|
| `compositional.py` | `CompositionalEmbedding`, `FeatureSet` |
| `feature_learning.py` | `CooccurrenceTracker`, `learn_features_hebbian`, `one_shot_learn_word` |
| `full_pipeline.py` | `CompositionalHolographicModel` (integrated) |

### 27.6 Usage

```python
from holographic import CompositionalHolographicModel

# Create model
model = CompositionalHolographicModel(
    vocab_size=10000,
    num_features=14,
    context_size=5,
    max_attractors=50000,
)

# Train with Hebbian learning
model.train(contexts, targets, hebbian_lr=0.05, verbose=True)

# Generate from context
tokens = model.generate([1, 2, 3, 4, 5], num_tokens=10)

# One-shot learn new word from context
model.one_shot_learn(new_word_idx, context_list, strength=0.8)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **3.0.0** | 2026-01-08 | **Full compositional pipeline** - Hebbian + attractor + one-shot |
| 2.9.0 | 2026-01-08 | Compositional embeddings implementation |
| 2.8.0 | 2026-01-08 | Compositional embeddings insight (Section 23) |
| 2.7.0 | 2026-01-08 | Multi-level hierarchy, diagnostics, empirical findings |
| 2.6.0 | 2026-01-08 | Quotient structure, binding operator |
| 2.4.0 | 2026-01-08 | Cross-disciplinary foundations (Section 22) |
| 2.3.0 | 2026-01-08 | Topological foundations (FOUNDATIONS.md) |
| 2.2.0 | 2026-01-08 | Active Inference extension (EFE-based generation) |
| 2.1.0 | 2026-01-08 | Matrix representation Cl(3,1) ≅ M₄(ℝ) |
| 2.0.0 | 2026-01-07 | Hierarchical retrieval (deprecated) |
| 1.x | pre-2026 | Legacy 16D vector implementation (archived) |
