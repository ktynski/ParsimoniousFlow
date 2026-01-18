# Holographic Language Model Architecture

**Version**: 1.0 (Updated 2026-01-08)  
**Status**: Phase 10 Complete — Theory-True Learned Embeddings  
**Goal**: Theory-true generative model via holographic projection with learned semantic content

---

## ⚠️ CRITICAL ARCHITECTURAL INSIGHT (2026-01-08)

**The geometry is fixed. The content must be LEARNED.**

From `rhnsclifford.md`:
```python
def learn(context, target):
    attractor[context] = embedding[target]  # Direct association
```

The original implementation (Phases 0-9) used **fixed** character encodings via `char_to_clifford()`. This fundamentally violated the theory requirement that semantic content comes from learning, not structure.

**Phase 10** introduced:
- `LearnedCliffordEmbedding` — Trainable token embeddings
- `ContextAttractorMap` — Learned context → attractor associations
- Interference-based entanglement (not character similarity)
- Singularity-based caustics (not averages)

| Gap | Fixed (Phases 0-9) | Learned (Phase 10) | Improvement |
|-----|-------------------|-------------------|-------------|
| Caustic Discrimination | 0.9996 | 0.8392 | **-16%** |
| Composition | 0.5118 | 1.0000 | **+95%** |
| Semantic Grounding | None | Context-specific attractors | ✓ |

---

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HOLOGRAPHIC LANGUAGE MODEL                        │
│                    (Theory-True Learned Version)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT: "the cat"                                                    │
│      │                                                               │
│      ▼                                                               │
│  ┌─────────────────────────────────────────────┐                     │
│  │ STEP 1: LEARNED BOUNDARY ENCODING           │                     │
│  │                                             │                     │
│  │ Embed characters using LEARNED embeddings   │                     │
│  │ - Position: golden-angle spiral (fixed)     │                     │
│  │ - Value: LEARNED Clifford multivector       │                     │
│  │ - Result: LearnedBoundaryState              │                     │
│  └──────────────────┬──────────────────────────┘                     │
│                     │                                                │
│                     ▼                                                │
│  ┌─────────────────────────────────────────────┐                     │
│  │ STEP 2: CONTEXT-ATTRACTOR MAPPING           │                     │
│  │                                             │                     │
│  │ Map context to its learned attractor        │                     │
│  │ - attractor[context] = embedding[target]    │                     │
│  │ - Different contexts → different attractors │                     │
│  │ - THIS IS WHERE SEMANTICS LIVE              │                     │
│  └──────────────────┬──────────────────────────┘                     │
│                     │                                                │
│                     ▼                                                │
│  ┌─────────────────────────────────────────────┐                     │
│  │ STEP 3: EQUILIBRIUM EVOLUTION               │                     │
│  │                                             │                     │
│  │ Evolve field toward context-specific        │                     │
│  │ attractor at rate γ = φ⁻²                   │                     │
│  │ - field = field + γ × (attractor - field)   │                     │
│  │ - Apply Grace contraction                   │                     │
│  │ - THE EQUILIBRIUM IS THE OUTPUT             │                     │
│  └──────────────────┬──────────────────────────┘                     │
│                     │                                                │
│                     ▼                                                │
│  ┌─────────────────────────────────────────────┐                     │
│  │ STEP 4: SURVIVABILITY-BASED GENERATION      │                     │
│  │                                             │                     │
│  │ For each candidate continuation:            │                     │
│  │ - Compute survivability score               │                     │
│  │ - Score = coherence + caustic + attractor   │                     │
│  │ - Select highest survivability              │                     │
│  └──────────────────┬──────────────────────────┘                     │
│                     │                                                │
│                     ▼                                                │
│  OUTPUT: next character (highest survivability)                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## THEORY FOUNDATION

### From rhnsclifford.md:

```
2+1D E8 Fibonacci CFT  →  3+1D Einstein Gravity + Standard Model
     (Boundary)                    (Bulk - our universe)
```

| Theory Concept | Implementation |
|----------------|----------------|
| 2+1D boundary CFT | Torus surface encoding |
| Holographic projection | Boundary → Bulk map |
| Entanglement → geometry | Interference-based entanglement |
| Caustic (singular locus) | Singularity detection in field |
| Grace contraction | Scale hierarchy (φ⁻ᵏ) |
| **LEARNING** | **Context → Attractor associations** |

### The Key Theoretical Insight

The theory explicitly states (rhnsclifford.md):

```python
# THEORY-TRUE: Coherence dynamics
def forward(context):
    field = build_initial_field(context)  # Geometric products
    field = evolve_to_equilibrium(field, attractor[context])  # γ = φ⁻²
    return field  # Equilibrium IS output

def learn(context, target):
    attractor[context] = embedding[target]  # Direct association
```

**The Clifford algebra provides GEOMETRY.**
**The CONTENT must be LEARNED.**

---

## COMPONENT SPECIFICATIONS

### Component 1: Learned Clifford Embedding (Phase 10)

```python
class LearnedCliffordEmbedding:
    """
    LEARNED token embeddings in Clifford space.
    
    REPLACES fixed char_to_clifford() with trainable embeddings.
    
    Theory basis:
    - The geometry (Clifford algebra, Grace, golden ratio) is FIXED
    - The CONTENT (embeddings) must be LEARNED from data
    - Learning = finding embeddings that make coherent contexts similar
    """
    
    def __init__(self, vocab_size: int = 256):
        # Initialize with golden-ratio structure, then allow learning
        self.embeddings = initialize_with_phi_structure(vocab_size)
    
    def __call__(self, token: int) -> np.ndarray:
        return self.embeddings[token]
    
    def update(self, token: int, gradient: np.ndarray):
        # Update preserves Clifford structure
        self.embeddings[token] -= lr * gradient
        self.embeddings[token] = grace_normalize(self.embeddings[token])
```

### Component 2: Context-Attractor Map (Phase 10)

```python
class ContextAttractorMap:
    """
    Maps contexts to their learned attractors.
    
    Theory basis (rhnsclifford.md):
        "def learn(context, target): attractor[context] = embedding[target]"
    
    The attractor is what the system evolves toward for a given context.
    Learning = associating contexts with target embeddings.
    """
    
    def associate(self, context: str, target: str):
        # attractor[context] = embedding[target]
        key = hash(context)
        self.attractors[key] = self.embedding_fn(ord(target[0]))
    
    def get_attractor(self, context: str) -> np.ndarray:
        # Return learned attractor, or interpolate from similar contexts
        ...
    
    def evolve_to_equilibrium(self, field, attractor, rate=PHI_INV_SQ):
        # Evolve field toward attractor at rate γ = φ⁻²
        for _ in range(max_steps):
            field = field + rate * (attractor - field)
            field = grace_operator(field)
        return field
```

### Component 3: Interference-Based Entanglement

```python
def compute_interference_entanglement(boundary: LearnedBoundaryState) -> np.ndarray:
    """
    Compute entanglement from INTERFERENCE PATTERNS, not character similarity.
    
    E[i,j] = phase_coherence × locality × interaction
    
    Where:
    - phase_coherence = cos(θ_i - θ_j)  # Golden angle phases
    - locality = φ⁻^(|i-j|/4)           # Distance decay
    - interaction = geom_product(f_i, f_j)[0]  # Clifford interaction
    """
```

### Component 4: Singularity-Based Caustic

```python
class LearnedCausticState:
    """
    Caustic computed from LEARNED boundary, capturing SINGULARITIES.
    
    Theory: Caustic = "singular locus where information focuses"
    
    Key components:
    - extrema_positions: WHERE are the field maxima?
    - eigenspectrum: WHICH eigenvalues dominate entanglement?
    - phase_winding: HOW does phase wind with position?
    """
```

---

## LEGACY CODE (Phases 0-9)

The following components are **DEPRECATED** for semantic tasks but retained for compatibility:

| Component | Status | Use Instead |
|-----------|--------|-------------|
| `char_to_clifford()` | DEPRECATED | `LearnedCliffordEmbedding` |
| `BoundaryState` | DEPRECATED | `LearnedBoundaryState` |
| `compute_entanglement_matrix()` | DEPRECATED | `compute_interference_entanglement()` |
| `CausticState` | DEPRECATED | `LearnedCausticState` |
| `survivability_score()` | DEPRECATED | `learned_survivability_score()` |

**The fixed encoding (Phases 0-9) is retained for:**
- Geometric primitives (torus, Clifford algebra)
- Testing invariants
- Understanding the architecture

**For actual language modeling, use the Phase 10 learned components.**

---

## IMPLEMENTATION PHASES

### Phases 0-9: Geometric Foundation ✓ COMPLETE
- Torus geometry, boundary encoding, holographic projection
- Transitions, memory, generation
- Caustic computation, survivability
- **LIMITATION**: Fixed encoding prevents semantic discrimination

### Phase 10: Theory-True Learned Embeddings ✓ COMPLETE
- [x] `LearnedCliffordEmbedding` — trainable embeddings
- [x] `ContextAttractorMap` — context → attractor learning
- [x] `LearnedBoundaryState` — boundary with learned fields
- [x] Interference-based entanglement
- [x] Singularity-based caustic
- [x] `learned_survivability_score()` — uses learned components
- [x] `train_embeddings_step()` — gradient-free training

### Phase 11: Large-Scale Training ⏳ PENDING
- [ ] Train on TinyStories with learned embeddings
- [ ] Scale to 100k+ context-attractor associations
- [ ] Benchmark vs fixed encoding approach

---

## SACRED CONSTANTS (From Theory — FIXED)

```python
# Golden ratio — from Λ² = Λ + 1
PHI = 1.618033988749894848204586834365638118
PHI_INV = 0.618033988749894848204586834365638118
PHI_INV_SQ = 0.381966011250105151795413165634361882

# Temperature — from coherence periodicity
BETA = 2 * PI * PHI  # ≈ 10.166

# Convergence rate — from spectral gap
GAMMA = PHI_INV_SQ  # ≈ 0.382

# Golden angle — for uniform torus coverage
GOLDEN_ANGLE = 2 * PI / (PHI * PHI)  # ≈ 2.399 rad ≈ 137.5°

# Clifford algebra
CLIFFORD_DIM = 16  # Cl(1,3)
```

---

## FILES

```
holographic/
├── ARCHITECTURE.md          ← This document
├── core.py                  ← All implementation (Phases 0-10)
└── __init__.py             ← Package marker

Supporting:
├── theory_verification.py   ← Verifies learned vs fixed improvement
└── THEORY_TRUE_ANALYSIS.md  ← Documents the architectural insight
```

---

## KEY INSIGHT (REMEMBER THIS)

> **The Clifford algebra provides the GEOMETRY.**
> **The CONTENT must be LEARNED.**
> 
> Fixed encoding gives structure without meaning.
> Learned encoding gives structure WITH meaning.
>
> This is what the theory always said:
> `attractor[context] = embedding[target]`

---

*Document Version: 1.0*  
*Status: Phase 10 complete, Phase 11 pending*
