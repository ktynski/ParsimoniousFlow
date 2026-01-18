import Lake
open Lake DSL

package «phi_proofs» where
  -- Settings
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «GoldenRatio» where
  srcDir := "GoldenRatio"

lean_lib «CliffordAlgebra» where
  srcDir := "CliffordAlgebra"

lean_lib «TransferMatrix» where
  srcDir := "TransferMatrix"

lean_lib «Complexity» where
  srcDir := "Complexity"

lean_lib «YangMills» where
  srcDir := "YangMills"
