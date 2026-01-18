/-
  Coherence Field Dynamics: The Grace Operator
  =============================================
  
  This file defines the evolution of the coherence field through the
  Grace operator - the fundamental dynamical law of FSCTF.
  
  KEY INSIGHT: The Grace operator is a CONTRACTION toward scalar equilibrium.
  
  G(Ψ) = Σₖ₌₀⁴ φ⁻ᵏ · Πₖ(Ψ)
  
  This scales each grade by decreasing powers of 1/φ:
  - Grade 0: × 1.000 (preserved)
  - Grade 1: × 0.618 (contracted)
  - Grade 2: × 0.382 (contracted more)
  - Grade 3: × 0.236 
  - Grade 4: × 0.146 (contracted most)
  
  The result: information flows toward the scalar "gist".
-/

import CoherenceField.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace CoherenceField.Dynamics

open GoldenRatio
open Cl31
open CoherenceField

/-! ## Grace Operator Properties -/

/-- φ⁻ᵏ coefficients are positive for all k -/
theorem grace_coeff_pos (k : ℕ) : φ^(-(k : ℤ)) > 0 := phi_inv_pow_pos k

/-- φ⁻ᵏ coefficients are at most 1 (contraction property) -/
theorem grace_coeff_le_one (k : ℕ) : φ^(-(k : ℤ)) ≤ 1 := phi_inv_pow_le_one k

/-- φ⁻⁰ = 1: Grade 0 is preserved exactly -/
theorem grace_coeff_zero : φ^(-(0 : ℤ)) = 1 := phi_inv_zero

/-- φ⁻¹ = φ - 1 ≈ 0.618 -/
theorem grace_coeff_one : φ^(-(1 : ℤ)) = φ - 1 := phi_inv_one

/-! ## Equilibrium States -/

/--
  DEFINITION: Equilibrium State
  
  A state x is at equilibrium if G(x) = x.
  The only such states are scalars (grade 0 only).
-/
def isEquilibrium (x : Cl31) : Prop :=
  graceOperator x = x

/--
  THEOREM: Equilibrium iff Pure Scalar
  
  A state is at equilibrium under Grace iff it is a pure scalar.
  
  Mathematical proof:
  - (→) If G(x) = x and k > 0: Πₖ(G(x)) = φ⁻ᵏ Πₖ(x) = Πₖ(x) implies (φ⁻ᵏ - 1)Πₖ(x) = 0.
        Since φ⁻ᵏ ≠ 1 for k > 0, we have Πₖ(x) = 0.
  - (←) If Πₖ(x) = 0 for all k > 0: G(x) = Σⱼ φ⁻ʲ Πⱼ(x) = φ⁻⁰ Π₀(x) = Π₀(x) = x.
-/
theorem equilibrium_iff_scalar (x : Cl31) :
    isEquilibrium x ↔ (∀ k > 0, gradeProject k x = 0) := by
  constructor
  · -- (→) If G(x) = x, then all higher grades are zero
    intro heq k hk
    unfold isEquilibrium at heq
    -- Apply grade projection to both sides of G(x) = x
    have hgk : gradeProject k (graceOperator x) = gradeProject k x := by rw [heq]
    -- By grace_grade_scaling: Πₖ(G(x)) = φ⁻ᵏ · Πₖ(x)
    have hscale : gradeProject k (graceOperator x) = φ^(-(k : ℤ)) • gradeProject k x := by
      by_cases hk4 : k ≤ 4
      · exact grace_grade_scaling k hk4 x
      · -- k > 4 means Πₖ(x) = 0 anyway
        rw [gradeProject_high k (by omega : k > 4) (graceOperator x)]
        rw [gradeProject_high k (by omega : k > 4) x]
        simp
    -- So φ⁻ᵏ · Πₖ(x) = Πₖ(x)
    rw [hscale] at hgk
    -- This means (φ⁻ᵏ - 1) · Πₖ(x) = 0
    -- For k > 0, φ⁻ᵏ < 1, so φ⁻ᵏ - 1 ≠ 0
    have hphi_ne : φ^(-(k : ℤ)) ≠ 1 := by
      have hphi_lt : φ^(-(k : ℤ)) < 1 := by
        rw [zpow_neg, zpow_natCast]
        have hpow_gt : φ^k > 1 := one_lt_pow₀ phi_gt_one (Nat.pos_iff_ne_zero.mp hk)
        exact inv_lt_one_of_one_lt₀ hpow_gt
      linarith
    -- From hgk: φ⁻ᵏ • Πₖ(x) = Πₖ(x)
    -- Rearranging: (φ⁻ᵏ - 1) • Πₖ(x) = 0
    -- Since φ⁻ᵏ - 1 ≠ 0, we have Πₖ(x) = 0
    have hdiff : (φ^(-(k : ℤ)) - 1) • gradeProject k x = 0 := by
      calc (φ^(-(k : ℤ)) - 1) • gradeProject k x 
        = φ^(-(k : ℤ)) • gradeProject k x - 1 • gradeProject k x := sub_smul _ _ _
        _ = φ^(-(k : ℤ)) • gradeProject k x - gradeProject k x := by rw [one_smul]
        _ = gradeProject k x - gradeProject k x := by rw [hgk]
        _ = 0 := sub_self _
    have hcoeff_ne : φ^(-(k : ℤ)) - 1 ≠ 0 := sub_ne_zero.mpr hphi_ne
    exact (smul_eq_zero.mp hdiff).resolve_left hcoeff_ne
  · -- (←) If all higher grades are zero, then G(x) = x
    intro hzero
    unfold isEquilibrium
    -- Strategy: Show G(x) = x by showing both equal Π₀(x)
    -- 1. x = Π₀(x) since all higher grades are zero
    -- 2. G(x) = φ⁰·Π₀(x) + Σₖ>0 φ⁻ᵏ·Πₖ(x) = Π₀(x) + Σₖ>0 φ⁻ᵏ·0 = Π₀(x)
    have h1 : gradeProject 1 x = 0 := hzero 1 (by norm_num)
    have h2 : gradeProject 2 x = 0 := hzero 2 (by norm_num)
    have h3 : gradeProject 3 x = 0 := hzero 3 (by norm_num)
    have h4 : gradeProject 4 x = 0 := hzero 4 (by norm_num)
    -- G(x) = Σⱼ φ⁻ʲ · Πⱼ(x) where only j=0 term survives
    simp only [graceOperator, LinearMap.sum_apply, LinearMap.smul_apply]
    rw [Finset.sum_eq_single 0]
    · -- j = 0 term: φ⁰ · Π₀(x)
      simp only [Nat.cast_zero, neg_zero, zpow_zero, one_smul]
      -- Need: Π₀(x) = x
      conv_rhs => rw [grade_decomposition x]
      simp only [Finset.sum_range_succ, Finset.range_zero, Finset.sum_empty, zero_add,
                 h1, h2, h3, h4, add_zero]
    · -- j ≠ 0 terms are zero
      intro j _ hj0
      have hj_pos : j > 0 := Nat.pos_of_ne_zero hj0
      by_cases hj_le : j ≤ 4
      · rw [hzero j hj_pos, smul_zero]
      · rw [gradeProject_high j (by omega : j > 4) x, smul_zero]
    · -- 0 ∈ range 5
      intro h_absurd
      simp at h_absurd

/-! ## Spectral Gap -/

/--
  DEFINITION: Spectral Gap
  
  The gap between the largest and second-largest Grace eigenvalue.
  This is 1 - φ⁻¹ = 1 - (φ-1) = 2 - φ ≈ 0.382
  
  The spectral gap controls the rate of convergence to equilibrium.
-/
noncomputable def spectralGap : ℝ := 1 - φ^(-(1 : ℤ))

theorem spectralGap_value : spectralGap = 2 - φ := by
  unfold spectralGap
  rw [phi_inv_one]
  ring

theorem spectralGap_positive : spectralGap > 0 := by
  rw [spectralGap_value]
  have h := phi_bounds
  linarith

/-! ## Conservation Laws -/

/--
  THEOREM: Scalar Conservation
  
  The Grace operator preserves the scalar (grade 0) component exactly.
  This is the "conserved charge" of coherence dynamics.
  
  Proof: By grace_grade_scaling with k=0: Π₀(G(x)) = φ⁻⁰ Π₀(x) = 1 · Π₀(x) = Π₀(x)
-/
theorem scalar_conservation (x : Cl31) :
    gradeProject 0 (graceOperator x) = gradeProject 0 x := by
  have h := grace_grade_scaling 0 (by norm_num : 0 ≤ 4) x
  -- h: gradeProject 0 (graceOperator x) = φ^(-(0:ℤ)) • gradeProject 0 x
  calc gradeProject 0 (graceOperator x) 
    = φ^(-(0:ℤ)) • gradeProject 0 x := h
    _ = 1 • gradeProject 0 x := by norm_num
    _ = gradeProject 0 x := one_smul _ _

/-! ## Summary -/

/-
  The dynamics established here show:
  
  1. CONTRACTION: Higher grades decay toward zero
  2. PRESERVATION: Scalar component is conserved  
  3. SPECTRAL GAP: Convergence rate is controlled by φ
  
  Physical significance:
  - Coherence field naturally evolves toward stable scalar states
  - Information crystallizes into invariant "meaning"
  - Fluctuations are damped without energy dissipation
  
  This is NOT standard quantum mechanics or thermodynamics.
  It's a new type of dynamics based on self-consistency.
-/

end CoherenceField.Dynamics
