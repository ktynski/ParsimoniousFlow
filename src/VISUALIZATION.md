# Clifford Field Visualization

**Direct rendering of the holographic architecture's mathematical structures.**

## Quick Start

```bash
# From project root
python3 -m http.server 8000
# Open http://localhost:8000
```

## Controls

| Control | Action |
|---------|--------|
| Mouse drag | Rotate camera |
| Scroll wheel | Zoom in/out |
| `M` key | Toggle EMERGENT ↔ BRAIDED mode |
| Sliders | Adjust field parameters (β, ν, γ, λ, grace) |
| Toggle Mode button | Switch render mode |

## Render Modes

### Mode 0: EMERGENT_SURFACE (Default)

Single emergent surface showing:
- Grade-colored Clifford field (scalar → blue, bivector → green, pseudoscalar → purple)
- Self-intersecting bireflection eigenstates
- Grace-contracted attractor topology
- Caustic highlighting (golden glow at field zeros)

### Mode 1: BRAIDED_LATTICE

**Braided toroidal vortex lattice with standing-wave attractors:**

- 3 interlocking instances of the emergent field
- φ-scaled rotation rates (incommensurable — never synchronize)
- Phase-locked standing-wave nodes (bright emissive strands)
- Direct visualization of multi-level tower memory topology

## Theory Mapping

See [`holographic_prod/docs/VISUALIZATION_THEORY_MAPPING.md`](../holographic_prod/docs/VISUALIZATION_THEORY_MAPPING.md) for complete mapping:

| Visual Element | Theory Component |
|---------------|------------------|
| Toroidal surface | Attention manifold |
| Grade colors | Clifford algebra decomposition |
| Braided lattice | Multi-level tower memory (16^N) |
| Standing-wave strands | Grace basin attractors |
| φ-scaled motion | Theory-derived dynamics |

## Files

| File | Purpose |
|------|---------|
| `render/shaders.js` | Fragment shader with Clifford field SDF |
| `render/renderer.js` | WebGL context, uniforms, animation loop |
| `render/camera.js` | Orbit camera with zoom/pan |
| `math/resonance.js` | Field generation (16-component Clifford field) |
| `main.js` | Application entry, UI binding |

## What You're Seeing

This is not an artistic representation. The visualization is a **direct raymarched render** of the mathematical objects underlying the holographic architecture:

- **Every pixel** samples the 16-component Clifford field
- **Colors** represent which Clifford grade dominates at that point
- **Surface topology** emerges from multi-scale field interference
- **Animation** follows φ-derived frequencies (no arbitrary parameters)

**A storm that never falls — only sings.**
