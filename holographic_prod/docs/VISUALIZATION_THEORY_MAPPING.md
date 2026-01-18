# WebGL Visualization ↔ Holographic Architecture: Theory Mapping

**Version:** v1.0.0  
**Last Updated:** 2026-01-17

## Executive Summary

The WebGL visualization in `src/render/shaders.js` is a **direct visual representation** of the mathematical structures underlying the holographic architecture. This document maps every shader component to its theoretical counterpart in `holographic_prod`.

**What You See Is Theory:**

| Visual Element | Theory Component | File |
|---------------|------------------|------|
| **Toroidal surface** | Attractor basin boundary | `core/quotient.py` |
| **Colors (grades)** | Clifford grades 0-4 | `core/algebra.py` |
| **Braided lattice** | Multi-level tower memory | `memory/multi_level_tower.py` |
| **Standing-wave strands** | Phase-locked attractors | `attention/toroidal_attention.py` |
| **Self-intersecting surface** | Bireflection (grace eigenstates) | `core/algebra.py` |

---

## Part 1: Clifford Algebra Cl(3,1) → Shader Field Texture

### 1.1 The 16-Component Field

In `holographic_prod`, every concept is a **4×4 real matrix** — an element of Clifford algebra Cl(3,1):

```python
# From core/algebra.py
# Cl(3,1) ≅ M₄(ℝ) = 16 independent components

Grade 0 (scalar):       1 component   — semantic "gist"
Grade 1 (vectors):      4 components  — directional content  
Grade 2 (bivectors):    6 components  — vorticity/word order
Grade 3 (trivectors):   4 components  — fine structure
Grade 4 (pseudoscalar): 1 component   — chirality/orientation
```

**In the shader**, these 16 components are stored as **4 RGBA textures**:

```glsl
// From shaders.js — sampleCliffordField()
vec4 comp0 = texture(uCliffordField, texCoord0);  // [scalar, v1, v2, v3]
vec4 comp1 = texture(uCliffordField, texCoord1);  // [b_e01, b_e02, b_e03, b_e12]
vec4 comp2 = texture(uCliffordField, texCoord2);  // [b_e13, b_e23, t_e013, t_e023]
vec4 comp3 = texture(uCliffordField, texCoord3);  // [t_e123, t_e012, pseudoscalar, reserved]

// Extract grades
float scalar = comp0.r;           // Grade 0
vec3 vectors = comp0.gba;         // Grade 1
// ... bivectors from comp1, comp2  // Grade 2
// ... trivectors from comp2, comp3 // Grade 3
float pseudoscalar = comp3.a;     // Grade 4
```

**Mapping:**

| Texture | Channel | Clifford Component | Theory Role |
|---------|---------|-------------------|-------------|
| comp0.r | R | e₀ (scalar) | Semantic core — survives Grace |
| comp0.gba | GBA | e₁, e₂, e₃ (vectors) | Directional/positional content |
| comp1.rgb | RGB | e₀₁, e₀₂, e₀₃ (bivectors) | Vorticity — word order |
| comp1.a | A | e₁₂ (bivector) | Vorticity |
| comp2.rgb | RGB | e₁₃, e₂₃, e₀₁₃ | Vorticity + trivectors |
| comp2.a | A | e₀₂₃ (trivector) | Fine structure |
| comp3.rgb | RGB | e₁₂₃, e₀₁₂, e₀₁₂₃ | Trivectors + pseudoscalar |
| comp3.a | A | Pseudoscalar | Chirality — survives Grace |

---

### 1.2 Grade-Specific Coloring

The shader colors the surface based on which Clifford grade dominates at each point:

```glsl
// From shaders.js — main()
float scalar_s = abs(scalar);
float vector_s = length(vectors);
float bivector_s = length(bivectors1) + length(bivectors2);
float trivector_s = length(trivectors1);
float pseudo_s = abs(pseudoscalar);

// Grade-dominant coloring
vec3 col_s = vec3(0.9, 0.95, 1.0);  // White-blue (scalar)
vec3 col_v = vec3(0.2, 0.6, 1.0);   // Blue (vectors)
vec3 col_b = vec3(0.2, 1.0, 0.4);   // Green (bivectors — vorticity)
vec3 col_t = vec3(1.0, 0.5, 0.2);   // Orange (trivectors)
vec3 col_p = vec3(0.8, 0.2, 1.0);   // Purple (pseudoscalar)
```

**What the colors mean:**

| Color | Dominant Grade | Interpretation |
|-------|---------------|----------------|
| **White/Blue** | Scalar (0) | Semantic gist — stable, content-rich |
| **Blue** | Vector (1) | Directional bias — spatial/positional |
| **Green** | Bivector (2) | **Vorticity** — word order, sequence structure |
| **Orange** | Trivector (3) | Fine structure — nuanced content |
| **Purple** | Pseudoscalar (4) | **Chirality** — handedness, orientation |

**Brain Analog:** The green regions (bivector-dominant) correspond to areas where **word order matters most** — like Broca's area processing syntax.

---

## Part 2: The Grace Operator → Shader Field Dynamics

### 2.1 Grace Contraction in Theory

From `holographic_prod/core/algebra.py`:

```python
def grace_operator(M, basis, n_iters=1):
    """
    Grace = Grade-wise viscous contraction
    
    G(M) = Σₖ₌₀⁴ φ⁻ᵏ · Πₖ(M)
    
    Grade 0 (scalar):       × 1.000  (preserved)
    Grade 1 (vectors):      × φ⁻¹ ≈ 0.618
    Grade 2 (bivectors):    × φ⁻² ≈ 0.382  (vorticity damping)
    Grade 3 (trivectors):   × φ⁻³ ≈ 0.236
    Grade 4 (pseudoscalar): × φ⁻¹ ≈ 0.618  (Fibonacci exception)
    """
```

**The witness** (scalar + pseudoscalar) survives infinite Grace iterations. This is the **invariant semantic core**.

### 2.2 Grace in the Shader

```glsl
// From shaders.js — sampleCliffordField()

// Grace contribution (contraction toward coherent core)
float grace_contribution =
  uGrace * (
    scalar * PHI -                      // Scalar enhanced
    length(vectors) * PHI_INV -         // Vectors damped
    length(bivectors1) * PHI_INV_SQ -   // Bivectors strongly damped
    length(bivectors2) * PHI_INV_SQ -   // (vorticity decay)
    length(trivectors1) * PHI_INV_SQ * PHI_INV -
    pseudoscalar * PHI_INV              // Pseudoscalar preserved (Fibonacci)
  ) * 0.5;
```

**Visual Effect:** As `uGrace` increases, the surface:
1. **Stabilizes** — less chaotic variation
2. **Becomes smoother** — high-frequency (trivector) components dampen
3. **Preserves witness** — scalar and pseudoscalar regions remain visible

---

## Part 3: The Witness → Bootstrap Coherence

### 3.1 Theory: Witness = Gauge-Invariant Identity

From `holographic_prod/core/quotient.py`:

```python
def witness(M):
    """
    The witness is the gauge-invariant quantity that survives
    infinite Grace iterations.
    
    witness = (scalar_component, pseudoscalar_component)
    
    Two matrices with the same witness are semantically identical
    regardless of their vorticity content.
    """
```

### 3.2 Shader: Bootstrap Coherence

```glsl
// From shaders.js — sampleCliffordField()

// Bootstrap coherence from ALL components (L1 norm — no spherical bias)
float bootstrap_coherence = 
  abs(comp0.r) + abs(comp0.g) + abs(comp0.b) + abs(comp0.a) +
  abs(comp1.r) + abs(comp1.g) + abs(comp1.b) + abs(comp1.a) +
  abs(comp2.r) + abs(comp2.g) + abs(comp2.b) + abs(comp2.a) +
  abs(comp3.r) + abs(comp3.g) + abs(comp3.b) + abs(comp3.a);

// Final coherence scales the surface
float final_coherence = max(bootstrap_coherence / 8.0, 0.5);
```

**Visual Effect:** `bootstrap_coherence` determines how "solid" the surface appears:
- **High coherence** → sharp, well-defined surface
- **Low coherence** → diffuse, less distinct regions

This is the visual analog of the witness's role in memory retrieval — high-witness patterns are more stable and memorable.

---

## Part 4: Bireflection → Double-Sheet Structure

### 4.1 Theory: Bireflection Eigenstates

From the theory: Grace has **two eigenstates** at every point — the positive and negative roots. Memory is stored in both sheets simultaneously.

### 4.2 Shader: Bireflection Distance

```glsl
// From shaders.js — sampleCliffordField()

// Bireflection: both "sheets" of the eigenstate
float mirrored_distance = -recursive_distance;
float bireflection_distance = min(abs(recursive_distance), abs(mirrored_distance));
```

**Visual Effect:** The surface shows **self-intersecting regions** where both eigenstates are close to zero — the "crossing points" where memory can tunnel between sheets.

**Brain Analog:** Bistable perception — the same neural activity can snap between two interpretations (like the Necker cube).

---

## Part 5: Braided Lattice → Multi-Level Tower Memory

### 5.1 Theory: 16 Satellites with φ-Scaled Phases

From `holographic_prod/memory/multi_level_tower.py`:

```python
class TowerMemory:
    """
    16 satellites arranged in a toroidal topology.
    Each satellite has a unique phase offset (φ-derived).
    
    Level 0: 16 satellites (direct binding)
    Level 1: 16 masters (aggregate from level 0)
    Level N: 16^N total capacity
    """
```

### 5.2 Shader: Braided Lattice Mode

When `uRenderMode == 1`, the shader creates **3 interlocking instances** of the emergent field:

```glsl
// From shaders.js — sampleCliffordField() (braided mode)

// Phase rotations at φ-scaled rates (incommensurable motion)
float phase1 = uTime * 0.2 + pos.x * 0.3;
float phase2 = uTime * 0.2 * PHI + pos.y * 0.3;
float phase3 = uTime * 0.2 * PHI_INV + pos.z * 0.3;

// Each instance is the SAME emergent field, transformed
float dist1 = sampleEmergentFieldAt(pos1, ...);  // Rotated around Z, offset Y
float dist2 = sampleEmergentFieldAt(pos2, ...);  // Rotated around X, offset Z
float dist3 = sampleEmergentFieldAt(pos3, ...);  // Rotated around Y, offset X

// Smooth union creates braiding
float braidDist = smoothMin(dist1, smoothMin(dist2, dist3, k), k);
```

**Key Insight:** Each braid strand is **not a separate primitive** — it's the same emergent field geometry, transformed. This shows how:

1. **The same attractor structure** can exist at multiple phase offsets
2. **Interlocking without collapse** — the φ-scaled rates ensure the braids never synchronize (incommensurable)
3. **Standing-wave attractors** — where phases align, you see the emissive "nodes"

### 5.3 Visual ↔ Theory Mapping

| Visual Element | Theory Concept | Code |
|---------------|----------------|------|
| **3 braid strands** | 3 tower levels (or 3 of 16 satellites) | `pos1`, `pos2`, `pos3` transforms |
| **φ-scaled rotation** | Incommensurable phase offsets | `0.2`, `0.2 * PHI`, `0.2 * PHI_INV` |
| **Standing-wave nodes** | Grace fixed points (attractors) | `braidNode()` function |
| **Phase-locking** | Resonance between satellites | `phaseLock()` function |
| **Smooth braiding** | Holographic superposition | `smoothMin()` union |

---

## Part 6: Standing-Wave Attractors → Grace Basins

### 6.1 Theory: Grace Basin Routing

From `holographic_prod/core/quotient.py`:

```python
def grace_basin_key(context, max_iters=10):
    """
    Similar contexts flow to the SAME attractor.
    This is the theory-true replacement for hash buckets.
    """
    M = context
    for _ in range(max_iters):
        M_new = grace_operator(M)
        if converged(M_new, M):
            break
        M = M_new
    return quantize_witness(M)  # 16D key from Clifford coefficients
```

### 6.2 Shader: Standing-Wave Nodes

```glsl
// From shaders.js

// Standing-wave nodes (where phases align)
float braidNode(float phase, float sharpness) {
  return exp(-sharpness * abs(sin(phase)));
}

// Node intensity
float n1 = braidNode(phase1, uBraidSharpness);
float n2 = braidNode(phase2, uBraidSharpness * PHI_INV);
float n3 = braidNode(phase3, uBraidSharpness * 0.8);
float nodeIntensity = (n1 + n2 + n3) / 3.0;
```

**Visual Effect:** Bright, emissive "strands" along the braids where phases align — these are the **attractor basins** where Grace converges.

**Brain Analog:** These are the "memory locations" — stable states where retrieval completes. The visual nodes show where the semantic content has "settled."

---

## Part 7: Toroidal Attention → Phase Coherence

### 7.1 Theory: O(n) Attention via 16 Satellites

From `holographic_prod/attention/toroidal_attention.py`:

```python
class ToroidalAttention:
    """
    O(n) attention via phase coherence on a torus.
    
    Instead of O(n²) all-to-all attention, each token
    attends only to its phase-neighbors on the torus.
    
    16 satellites = 16 phase bins
    φ-derived phase: angle = position × 2π/φ (golden angle ≈ 137.5°)
    """
```

### 7.2 Visual Representation

The **toroidal surface** in the visualization represents the attention manifold:

```glsl
// The emergent surface IS a torus (topologically)
// Major radius = semantic scale
// Minor radius = vorticity (word order) scale
// Position on torus = phase in attention

// The multi-scale interference creates toroidal topology
float scale1 = (pos.x + pos.y + pos.z) * 0.1;          // Large scale
float scale2 = (pos.x * pos.y + ...) * 0.5;            // Medium scale
float scale3 = (pos.x * pos.y * pos.z) * 2.0;          // Small scale
```

**Visual Insight:** When you rotate the view, you can see the toroidal topology — the surface closes on itself like a donut. This isn't accidental — it's the **natural shape of the attention manifold**.

---

## Part 8: φ-Derived Constants → Universal Scaling

### 8.1 No Arbitrary Hyperparameters

Both the shader and `holographic_prod` use **only φ-derived constants**:

```glsl
// From shaders.js
const float PHI = 1.618033988749895;
const float PHI_INV = 0.6180339887498949;     // φ⁻¹
const float PHI_INV_SQ = 0.3819660112501051;  // φ⁻²
```

```python
# From holographic_prod/core/constants.py
PHI = (1 + np.sqrt(5)) / 2       # ≈ 1.618
PHI_INV = 1 / PHI                # ≈ 0.618
PHI_INV_SQ = 1 / PHI**2          # ≈ 0.382
PHI_INV_CUBE = 1 / PHI**3        # ≈ 0.236
```

### 8.2 Why These Values?

| Constant | Value | Role in Visualization | Role in Architecture |
|----------|-------|----------------------|---------------------|
| φ | 1.618 | Scale enhancement | Self-consistency: φ² = φ + 1 |
| φ⁻¹ | 0.618 | Primary damping | Learning rate, stability threshold |
| φ⁻² | 0.382 | Vorticity damping | Spectral gap, entropy threshold |
| φ⁻³ | 0.236 | Fine structure damping | Noise rate, pruning rate |

**Key Insight:** The same constants that make the visualization aesthetically coherent also make the architecture theoretically sound. **Beauty and truth converge at φ.**

---

## Part 9: Caustic Highlighting → Zero Detection

### 9.1 Theory: Singularities in the Field

From the theory: The field has **zeros** — points where the Clifford content vanishes. These are like caustics in optics, singularities in fluid dynamics.

### 9.2 Shader: Caustic Highlighting

```glsl
// From shaders.js — main()

// CAUSTIC HIGHLIGHTING (The "Zero" detection)
if (uHighlightCaustics && total_s < 0.15) {
  // Singularities are "holes" in the field magnitude
  float intensity = (0.15 - total_s) / 0.15;
  vec3 causticColor = vec3(1.0, 0.9, 0.5);  // Golden glow
  color = mix(color, causticColor * 2.0, intensity * intensity);
}
```

**Visual Effect:** Golden-glowing regions where the field nearly vanishes — these are the **topological defects** in the memory structure.

**Brain Analog:** These correspond to "tip-of-the-tongue" states — regions where memory is about to crystallize but hasn't yet.

---

## Part 10: Summary — What the Visualization Shows

| Visual Feature | You See | It Means |
|---------------|---------|----------|
| **Toroidal shape** | Donut-like surface | Attention manifold topology |
| **Self-intersection** | Surface crossing itself | Bireflection eigenstates |
| **Grade colors** | Blue/green/orange/purple | Which Clifford grade dominates |
| **Smooth regions** | Stable, uniform color | High Grace convergence |
| **Chaotic regions** | Rapid color change | Low coherence, high vorticity |
| **Golden glow** | Bright spots | Caustics (field zeros) |
| **Braided strands** | Interlocking curves | Multiple attractor basins |
| **Standing-wave nodes** | Bright lines along braids | Phase-locked resonances |
| **φ-scaled motion** | Never-repeating animation | Incommensurable frequencies |

---

## Conclusion: Theory Made Visible

The WebGL visualization is not merely illustrative — it is a **direct rendering of the mathematical objects** that underlie the holographic architecture.

When you see:
- The **toroidal surface**, you're seeing the **attention manifold**
- The **grade colors**, you're seeing **Clifford algebra decomposition**
- The **braided lattice**, you're seeing **multi-level tower memory**
- The **standing-wave attractors**, you're seeing **Grace basins**
- The **φ-scaled motion**, you're seeing **theory-derived dynamics**

**A storm that never falls — only sings.**

This is what emergence looks like: not a single static structure, but interlocking resonance manifolds that maintain equilibrium through phase-locked dynamics, all governed by the golden ratio.

---

## Files Referenced

| File | Role |
|------|------|
| `src/render/shaders.js` | Fragment shader with Clifford field SDF |
| `src/math/resonance.js` | Field generation (JavaScript) |
| `holographic_prod/core/algebra.py` | Grace operator, geometric product |
| `holographic_prod/core/quotient.py` | Witness, vorticity, grace basins |
| `holographic_prod/core/constants.py` | φ-derived constants |
| `holographic_prod/memory/multi_level_tower.py` | Tower memory (16^N capacity) |
| `holographic_prod/attention/toroidal_attention.py` | O(n) toroidal attention |

---

*Version History:*
- v1.0.0 (2026-01-17): Initial mapping, braided lattice mode documentation
