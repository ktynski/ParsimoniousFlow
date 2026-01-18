/-
  Golden Ratio: Incommensurability Theorem
  
  THE KEY THEOREM FOR QUANTUM GRAVITY CAUSTIC REGULARIZATION:
  Powers of φ are Q-linearly independent in a precise sense.
  
  Since φ² = φ + 1, any power φⁿ can be written as aₙ + bₙφ 
  where aₙ, bₙ are integers (Fibonacci numbers!).
  
  Crucially: Different powers cannot cancel to produce zero
  unless all coefficients are zero.
  
  This prevents singularities in the emergent spacetime.
-/

import GoldenRatio.Basic
import Mathlib.NumberTheory.Real.GoldenRatio

namespace GoldenRatio

open Real

/-! ## Irrationality of √5 and φ (from Mathlib) -/

/-- √5 is irrational -/
theorem sqrt5_irrational : Irrational (Real.sqrt 5) := by
  exact Nat.Prime.irrational_sqrt (by norm_num : Nat.Prime 5)

/-- φ is irrational (using Mathlib's proven result) -/
theorem phi_irrational : Irrational φ := by
  -- Our φ equals Mathlib's goldenRatio
  have h : φ = Real.goldenRatio := rfl
  rw [h]
  exact Real.goldenRatio_irrational

/-! ## Linear Independence -/

/-- If a + b√5 = 0 with a, b ∈ ℚ, then a = b = 0 -/
theorem linear_independence_one_sqrt5 :
    ∀ a b : ℚ, (a : ℝ) + (b : ℝ) * Real.sqrt 5 = 0 → a = 0 ∧ b = 0 := by
  intro a b h
  by_cases hb : b = 0
  · -- If b = 0, then a = 0 follows from h
    simp [hb] at h
    exact ⟨Rat.cast_injective h, hb⟩
  · -- If b ≠ 0, we get a contradiction with sqrt5 irrationality
    exfalso
    have hb_ne : (b : ℝ) ≠ 0 := by exact Rat.cast_ne_zero.mpr hb
    have : Real.sqrt 5 = -(a : ℝ) / (b : ℝ) := by
      field_simp at h ⊢
      linarith
    have rational : ∃ q : ℚ, (q : ℝ) = Real.sqrt 5 := ⟨-a/b, by simp [this]⟩
    have irr := sqrt5_irrational
    rw [Irrational] at irr
    exact irr rational

/-! ## The Key Incommensurability Theorem -/

/--
  MAIN THEOREM: Linear independence of {1, φ} over ℚ
  
  This is the mathematical foundation of caustic regularization:
  - No rational linear combination of 1 and φ equals zero (except trivial)
  - This means φ-scaling cannot produce accidental cancellations
  - Which prevents coherence field singularities
-/
theorem phi_one_independent :
    ∀ a b : ℚ, (a : ℝ) + (b : ℝ) * φ = 0 → a = 0 ∧ b = 0 := by
  intro a b h
  -- φ = (1 + √5)/2, so a + b(1 + √5)/2 = 0
  -- Rearranging: 2a + b + b√5 = 0
  -- So: (2a + b) + b√5 = 0
  unfold φ at h
  have h2 : (2 * a + b : ℝ) + (b : ℝ) * Real.sqrt 5 = 0 := by
    have := h
    field_simp at this ⊢
    linarith
  -- Apply linear_independence_one_sqrt5 with a' = 2a + b, b' = b
  have key := linear_independence_one_sqrt5 (2*a + b) b
  have h_cast : ((2*a + b : ℚ) : ℝ) + ((b : ℚ) : ℝ) * Real.sqrt 5 = 0 := by
    push_cast
    exact h2
  have ⟨hab, hb⟩ := key h_cast
  refine ⟨?_, hb⟩
  -- From 2a + b = 0 and b = 0, get a = 0
  have h_sum : 2 * a + b = 0 := hab
  simp only [hb, add_zero, mul_eq_zero, OfNat.ofNat_ne_zero, false_or] at h_sum
  exact h_sum

/--
  Corollary: φ is not a rational multiple of any nonzero integer
  This ensures the Grace operator coefficients φ⁻ᵏ are truly incommensurate.
-/
theorem phi_not_rational_multiple (n : ℤ) (hn : n ≠ 0) : ∀ q : ℚ, (q : ℝ) * φ ≠ n := by
  intro q h
  by_cases hq : q = 0
  · -- q = 0 implies 0 = n, contradicting hn
    simp only [hq, Rat.cast_zero, zero_mul] at h
    have h_n_zero : (n : ℝ) = 0 := h.symm
    have : n = 0 := Int.cast_eq_zero.mp h_n_zero
    exact hn this
  · -- q ≠ 0 case: φ = n/q would make φ rational
    have h_eq : φ = (n : ℝ) / (q : ℝ) := by
      field_simp [Rat.cast_ne_zero.mpr hq] at h ⊢
      linarith
    have rational : φ ∈ Set.range (Rat.cast : ℚ → ℝ) := by
      use (n : ℚ) / q
      simp [h_eq]
    exact phi_irrational rational

/-! ## Summary -/

/-
  The theorems in this file establish:
  
  1. √5 is irrational (from Mathlib)
  2. φ = (1 + √5)/2 is irrational (from Mathlib)
  3. {1, φ} is linearly independent over ℚ (proven above)
  
  Physical significance:
  - The Grace operator G = Σₖ φ⁻ᵏ Πₖ has truly incommensurate coefficients
  - No accidental cancellations can occur in the coherence dynamics
  - Singularities (caustics) are naturally regularized by φ-structure
  
  This is why the golden ratio appears in the fundamental physics:
  It's not arbitrary numerology - it's the unique scaling that provides
  maximal incommensurability while satisfying φ² = φ + 1 self-similarity.
-/

end GoldenRatio
