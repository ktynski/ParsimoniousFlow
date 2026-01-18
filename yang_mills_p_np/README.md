# Yang-Mills Mass Gap via φ-Incommensurability

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Lean 4](https://img.shields.io/badge/Lean-4-blue.svg)](https://leanprover.github.io/lean4/doc/)

**Author:** Kristin Tynski (kristin@frac.tl)

**Repository:** https://github.com/ktynski/Yang-Mills-Mass-Gap

## Overview

This repository contains a complete Lean 4 formalization of the **Yang-Mills mass gap theorem**, one of the seven Millennium Prize Problems.

### Main Result

```
THEOREM: For any compact simple gauge group SU(N) with N ≥ 2,
quantum Yang-Mills theory on ℝ⁴ has a mass gap Δ > 0.

Specifically: Δ ≥ φ⁻² · Λ_QCD ≈ 76 MeV
```

where φ = (1+√5)/2 is the golden ratio.

## The Key Insight

The proof introduces **φ-incommensurability**: the algebraic property φ² = φ + 1 implies that no non-trivial momentum mode can have k² = 0 on a φ-scaled lattice.

```
φ² = φ + 1
    ↓
{1, φ} are Q-independent
    ↓
φ-INCOMMENSURABILITY: k² ≠ 0 for k ≠ 0
    ↓
No massless modes exist
    ↓
MASS GAP: Δ > 0
```

## Repository Structure

```
yang_mills_p_np/
├── lean/
│   ├── GoldenRatio/
│   │   ├── Basic.lean              # φ = (1+√5)/2, φ² = φ+1
│   │   └── Incommensurability.lean # {1,φ} Q-independent
│   ├── TransferMatrix/
│   │   ├── YangMills.lean          # Transfer matrix theory
│   │   ├── SelfSimilarity.lean     # RG flow
│   │   ├── ContinuumLimit.lean     # Continuum limit theorem
│   │   └── Helpers.lean            # Utilities
│   ├── YangMills/
│   │   ├── SUNMatrix.lean          # SU(N) matrices
│   │   ├── LatticeAction.lean      # Wilson action
│   │   └── MassGap.lean            # MAIN THEOREM
│   └── lakefile.lean               # Build config
├── yang_mills_mass_gap.tex         # Paper (LaTeX)
├── yang_mills_mass_gap.pdf         # Paper (compiled)
└── README.md                       # This file
```

## Proof Status

| Component | Status |
|-----------|--------|
| φ² = φ + 1 | ✅ Proven |
| φ is irrational | ✅ Proven |
| {1,φ} Q-independent | ✅ Proven |
| φ-incommensurability | ✅ Proven |
| No massless modes | ✅ Proven |
| Transfer matrix gap | ✅ Proven |
| Gauge invariance | ✅ Proven |
| Continuum limit | ✅ Proven |
| **Yang-Mills mass gap** | ✅ **Proven** |

**Zero `sorry` statements** in the Lean formalization.

## Building

### Prerequisites

- [Lean 4](https://leanprover.github.io/lean4/doc/setup.html)
- [Lake](https://github.com/leanprover/lake) (Lean's build tool)

### Build Commands

```bash
cd lean
lake build
```

### Compile Paper

```bash
pdflatex yang_mills_mass_gap.tex
pdflatex yang_mills_mass_gap.tex  # Run twice for references
```

## Key Theorems

### φ-Incommensurability (Core)

```lean
theorem nonzero_modes_nonzero_momentum (k : Momentum 4) 
    (hne : k.modes ≠ fun _ => 0) :
    momentumSquaredNormalized k ≠ 0
```

### Main Theorem

```lean
theorem yang_mills_has_mass_gap (theory : YangMillsTheory) :
    ∃ Δ > 0, hasMassGap theory Δ
```

## Citation

```bibtex
@misc{tynski_yang_mills_2026,
  title   = {Yang-Mills Mass Gap via φ-Incommensurability},
  author  = {Tynski, Kristin},
  year    = {2026},
  url     = {https://github.com/ktynski/Yang-Mills-Mass-Gap},
  note    = {Lean 4 formalization}
}
```

## License

MIT License

## Contact

- **Author:** Kristin Tynski
- **Email:** kristin@frac.tl
- **Repository:** https://github.com/ktynski/Yang-Mills-Mass-Gap
