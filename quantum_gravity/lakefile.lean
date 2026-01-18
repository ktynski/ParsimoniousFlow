import Lake
open Lake DSL

package «quantum_gravity» where
  moreLeanArgs := #["-Dpp.unicode.fun=true", "-DautoImplicit=false"]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

-- All modules as a single library
@[default_target]
lean_lib «QuantumGravity» where
  globs := #[.submodules `GoldenRatio, .submodules `CliffordAlgebra, 
             .submodules `CoherenceField, .submodules `InformationGeometry,
             .submodules `Holography, .submodules `Caustics, .submodules `MainTheorem]
