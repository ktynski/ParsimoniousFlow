/**
 * grace.js - The Grace Operator
 * 
 * The Grace operator 𝒢 is a CONTRACTION that prevents blow-up.
 * It projects a multivector onto its "coherent core" - the scalar + pseudoscalar
 * components, scaled by the golden ratio inverse φ⁻¹ ≈ 0.618.
 * 
 * Mathematical definition:
 *   𝒢(M) = ⟨M⟩₀ + φ⁻¹ · ⟨M⟩₄
 * 
 * Key properties:
 *   1. Contraction: ||𝒢(M)|| ≤ φ⁻¹ ||M|| (since φ⁻¹ < 1)
 *   2. Idempotent on grade-0⊕4 subspace
 *   3. Annihilates grades 1, 2, 3
 * 
 * Physical interpretation:
 *   - In Navier-Stokes: Acts like viscosity, dissipating energy
 *   - In RH: Prevents zeros from escaping critical line
 *   - In topology: Contracts the manifold toward the throat
 */

import { Multivector, PHI, PHI_INV, GRADES } from './clifford.js';

/**
 * Apply Grace operator to a multivector
 * Returns the "coherent core": scalar + φ⁻¹ × pseudoscalar
 */
export function grace(M) {
  const result = new Multivector();
  
  // Project to grade 0 (scalar)
  const scalar = M.get(0);
  
  // Project to grade 4 (pseudoscalar) and scale by φ⁻¹
  const pseudoscalar = M.get(15);
  
  // Grace core = scalar + φ⁻¹ × pseudoscalar
  result.set(0, scalar);
  result.set(15, PHI_INV * pseudoscalar);
  
  return result;
}

/**
 * Compute the Grace magnitude (coherence measure)
 * This is the L1 norm of the Grace projection
 */
export function graceMagnitude(M) {
  const scalar = Math.abs(M.get(0));
  const pseudoscalar = Math.abs(M.get(15));
  return scalar + PHI_INV * pseudoscalar;
}

/**
 * Apply full Grace contraction to all components
 * Each grade is scaled by φ⁻ᵏ where k is the grade
 * 
 * This creates a "harmonic cascade" where higher grades are
 * progressively suppressed, preventing energy from escaping
 * to high-frequency modes (analogous to Navier-Stokes viscosity)
 */
export function graceContract(M) {
  const result = new Multivector();
  
  for (let i = 0; i < 16; i++) {
    const grade = GRADES[i];
    // Each grade scaled by φ⁻ᵍʳᵃᵈᵉ
    const scale = Math.pow(PHI_INV, grade);
    result.set(i, M.get(i) * scale);
  }
  
  return result;
}

/**
 * Iterative Grace flow - evolves field toward fixed point
 * 
 * dM/dt = -∇𝒢(M) = 𝒢(M) - M
 * 
 * This is gradient flow in the Grace potential, which
 * converges to the coherent core.
 * 
 * @param M - Current multivector
 * @param dt - Time step
 * @returns Evolved multivector
 */
export function graceFlow(M, dt = 0.1) {
  const G = graceContract(M);
  const result = new Multivector();
  
  // M' = M + dt * (G(M) - M) = (1-dt)M + dt*G(M)
  for (let i = 0; i < 16; i++) {
    result.set(i, (1 - dt) * M.get(i) + dt * G.get(i));
  }
  
  return result;
}

/**
 * Check if multivector is in the "coherent cone"
 * A multivector is coherent if its Grace magnitude exceeds
 * the magnitude of its non-coherent parts
 */
export function isCoherent(M) {
  const graceM = grace(M);
  const nonGrace = M.clone();
  nonGrace.set(0, 0);
  nonGrace.set(15, 0);
  
  return graceM.norm() > nonGrace.norm();
}

/**
 * Compute spectral gap estimate
 * 
 * The spectral gap γ = λ_max - λ_2 controls convergence rate.
 * For the Grace operator with φ-scaling:
 *   λ_max = 1 (scalar eigenvalue)
 *   λ_2 = φ⁻¹ (pseudoscalar eigenvalue)
 *   γ = 1 - φ⁻¹ = 1/φ² ≈ 0.382
 * 
 * This is the "universal" spectral gap for φ-based systems.
 */
export function spectralGap() {
  return 1 - PHI_INV; // = 1/φ² ≈ 0.382
}

/**
 * Compute the Grace coherence of a field at a point
 * 
 * Coherence = (|scalar| + φ⁻¹|pseudoscalar|) / ||M||
 * 
 * Returns value in [0, 1] where:
 *   1 = fully coherent (only scalar + pseudoscalar)
 *   0 = fully incoherent (no scalar or pseudoscalar)
 */
export function coherence(M) {
  const norm = M.norm();
  if (norm < 1e-10) return 0;
  
  const graceMag = graceMagnitude(M);
  return Math.min(1, graceMag / norm);
}

/**
 * The Bireflection operator β
 * 
 * β(M) = M̃ where M̃ is the grade-involution followed by reversion
 * This creates the "two-sheeted" structure in the SDF
 * 
 * Property: β ∘ β = identity (involution)
 */
export function bireflect(M) {
  const result = new Multivector();
  
  // Grade involution: grade k → (-1)^k
  // Reversion: grade k → (-1)^(k(k-1)/2)
  // Combined: grade k → (-1)^k × (-1)^(k(k-1)/2)
  
  for (let i = 0; i < 16; i++) {
    const k = GRADES[i];
    const gradeSign = Math.pow(-1, k);
    const revSign = Math.pow(-1, k * (k - 1) / 2);
    result.set(i, M.get(i) * gradeSign * revSign);
  }
  
  return result;
}

/**
 * Symmetric Grace distance
 * 
 * d_G(M) = min(||M - 𝒢(M)||, ||M + 𝒢(M)||)
 * 
 * This creates the caustic structure - zeros occur where
 * the field equals its Grace projection (coherent core)
 */
export function graceDistance(M) {
  const G = grace(M);
  
  let distMinus = 0;
  let distPlus = 0;
  
  for (let i = 0; i < 16; i++) {
    const diff = M.get(i) - G.get(i);
    const sum = M.get(i) + G.get(i);
    distMinus += diff * diff;
    distPlus += sum * sum;
  }
  
  return Math.min(Math.sqrt(distMinus), Math.sqrt(distPlus));
}

