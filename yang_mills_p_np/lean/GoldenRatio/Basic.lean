/-
  Golden Ratio: Basic Properties
  
  This file establishes the fundamental properties of φ = (1 + √5)/2
  that underpin both the Yang-Mills mass gap and P vs NP analyses.
  
  KEY THEOREM: φ satisfies the self-consistency equation φ² = φ + 1
  This is the same equation that appears in:
  - SCCMU Theory (coherence maximization)
  - Geometry of Mind (Grace operator)
  - Fibonacci anyons (quantum dimension)
-/

import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic

namespace GoldenRatio

/-! ## Definition -/

/-- The golden ratio φ = (1 + √5) / 2 -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The conjugate golden ratio ψ = (1 - √5) / 2 -/
noncomputable def ψ : ℝ := (1 - Real.sqrt 5) / 2

/-! ## Basic Bounds -/

theorem sqrt5_pos : Real.sqrt 5 > 0 := Real.sqrt_pos.mpr (by norm_num)

theorem sqrt5_gt_two : Real.sqrt 5 > 2 := by
  rw [show (2 : ℝ) = Real.sqrt 4 by rw [Real.sqrt_eq_iff_sq_eq]; ring_nf; norm_num; norm_num]
  exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)

theorem sqrt5_lt_three : Real.sqrt 5 < 3 := by
  rw [show (3 : ℝ) = Real.sqrt 9 by rw [Real.sqrt_eq_iff_sq_eq]; ring_nf; norm_num; norm_num]
  exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)

/-- φ > 1 -/
theorem phi_gt_one : φ > 1 := by
  unfold φ
  have h : Real.sqrt 5 > 1 := by
    calc Real.sqrt 5 > 2 := sqrt5_gt_two
      _ > 1 := by norm_num
  linarith

/-- φ > 0 -/
theorem phi_pos : φ > 0 := by linarith [phi_gt_one]

/-- φ ≠ 0 -/
theorem phi_ne_zero : φ ≠ 0 := ne_of_gt phi_pos

/-- ψ < 0 -/
theorem psi_neg : ψ < 0 := by
  unfold ψ
  have h : Real.sqrt 5 > 1 := by linarith [sqrt5_gt_two]
  linarith

/-- |ψ| < 1 -/
theorem psi_abs_lt_one : |ψ| < 1 := by
  unfold ψ
  have h1 : Real.sqrt 5 > 2 := sqrt5_gt_two
  have h2 : Real.sqrt 5 < 3 := sqrt5_lt_three
  rw [abs_of_neg (by linarith : (1 - Real.sqrt 5) / 2 < 0)]
  linarith

/-! ## The Fundamental Identity -/

/-- 
  THE CORE THEOREM: φ² = φ + 1
  
  This self-referential equation is the source of all φ-structure.
  It says: "the square of the whole equals the whole plus unity"
  
  This same equation appears in:
  1. SCCMU: Λ² = Λ + 1 (coherence self-consistency)
  2. Fibonacci: F(n+2) = F(n+1) + F(n) limiting ratio
  3. Geometry of Mind: Grace operator eigenvalue
-/
theorem phi_squared : φ ^ 2 = φ + 1 := by
  unfold φ
  have h5 : (Real.sqrt 5) ^ 2 = 5 := Real.sq_sqrt (by norm_num : (5:ℝ) ≥ 0)
  field_simp
  ring_nf
  rw [h5]
  ring

/-- ψ also satisfies the golden equation -/
theorem psi_squared : ψ ^ 2 = ψ + 1 := by
  unfold ψ
  have h5 : (Real.sqrt 5) ^ 2 = 5 := Real.sq_sqrt (by norm_num : (5:ℝ) ≥ 0)
  field_simp
  ring_nf
  rw [h5]
  ring

/-! ## Vieta's Formulas -/

/-- φ and ψ are roots of x² - x - 1 = 0 -/
theorem phi_root : φ ^ 2 - φ - 1 = 0 := by linarith [phi_squared]
theorem psi_root : ψ ^ 2 - ψ - 1 = 0 := by linarith [psi_squared]

/-- Sum of roots: φ + ψ = 1 -/
theorem phi_plus_psi : φ + ψ = 1 := by
  unfold φ ψ
  ring

/-- Product of roots: φ · ψ = -1 -/
theorem phi_times_psi : φ * ψ = -1 := by
  unfold φ ψ
  have h5 : (Real.sqrt 5) ^ 2 = 5 := Real.sq_sqrt (by norm_num : (5:ℝ) ≥ 0)
  field_simp
  ring_nf
  rw [h5]
  ring

/-! ## Inverse Relations -/

/-- 1/φ = φ - 1 (the inverse equals φ minus unity) -/
theorem phi_inv : φ⁻¹ = φ - 1 := by
  have h := phi_squared
  field_simp [phi_ne_zero] at h ⊢
  linarith

/-- φ · (φ - 1) = 1 (equivalent form) -/
theorem phi_times_inv : φ * (φ - 1) = 1 := by
  have h := phi_squared
  linarith

/-! ## Powers of φ (Fibonacci Connection) -/

/-- Binet's formula: φⁿ = Fₙφ + Fₙ₋₁ where Fₙ is nth Fibonacci -/

/-- φ³ = 2φ + 1 -/
theorem phi_cubed : φ ^ 3 = 2 * φ + 1 := by
  calc φ ^ 3 = φ ^ 2 * φ := by ring
    _ = (φ + 1) * φ := by rw [phi_squared]
    _ = φ ^ 2 + φ := by ring
    _ = (φ + 1) + φ := by rw [phi_squared]
    _ = 2 * φ + 1 := by ring

/-- φ⁴ = 3φ + 2 -/
theorem phi_fourth : φ ^ 4 = 3 * φ + 2 := by
  calc φ ^ 4 = φ ^ 3 * φ := by ring
    _ = (2 * φ + 1) * φ := by rw [phi_cubed]
    _ = 2 * φ ^ 2 + φ := by ring
    _ = 2 * (φ + 1) + φ := by rw [phi_squared]
    _ = 3 * φ + 2 := by ring

/-- φ⁵ = 5φ + 3 -/
theorem phi_fifth : φ ^ 5 = 5 * φ + 3 := by
  calc φ ^ 5 = φ ^ 4 * φ := by ring
    _ = (3 * φ + 2) * φ := by rw [phi_fourth]
    _ = 3 * φ ^ 2 + 2 * φ := by ring
    _ = 3 * (φ + 1) + 2 * φ := by rw [phi_squared]
    _ = 5 * φ + 3 := by ring

/-! ## The Fibonacci Recurrence -/

/-- 
  General form: φⁿ⁺² = φⁿ⁺¹ + φⁿ
  This is the Fibonacci recurrence in exponential form!
-/
theorem phi_fibonacci_recurrence (n : ℕ) : φ ^ (n + 2) = φ ^ (n + 1) + φ ^ n := by
  have h := phi_squared
  calc φ ^ (n + 2) = φ ^ n * φ ^ 2 := by ring
    _ = φ ^ n * (φ + 1) := by rw [h]
    _ = φ ^ n * φ + φ ^ n := by ring
    _ = φ ^ (n + 1) + φ ^ n := by ring

/-! ## Approximation Value -/

/-- φ ≈ 1.618 (within bounds) -/
theorem phi_bounds : 1.618 < φ ∧ φ < 1.619 := by
  constructor
  · unfold φ
    have h1 : Real.sqrt 5 > 2.236 := by
      rw [Real.lt_sqrt (by norm_num) (by norm_num)]
      norm_num
    linarith
  · unfold φ
    have h2 : Real.sqrt 5 < 2.237 := by
      rw [Real.sqrt_lt' (by norm_num)]
      constructor
      · exact sqrt5_pos
      · norm_num
    linarith

end GoldenRatio
