/**
 * resonance.js - Theory-Compliant Field Generation
 * 
 * Matches the original FIRM-Core resonance_driven_engine.js
 * 
 * Key parameters:
 *   - β (beta): Temperature - controls scalar/vector thermal fluctuations
 *   - ν (nu): Diffusion - controls bivector rotation/mixing (TOROIDAL PATTERNS)
 *   - γ (gamma): Spectral gap - controls trivector complexity
 *   - λ_max: Eigenvalue - controls pseudoscalar geometric scale
 *   - emergenceRate: Modulates oscillation amplitudes
 *   - bootstrapEnergy: Base field amplitude
 *   - graceScale: Grace operator contraction factor
 */

import { PHI, PHI_INV, Multivector } from './clifford.js';

// Default control parameters (match FIRM-Core slider defaults)
export const DEFAULT_PARAMS = {
  beta: 0.5,              // Temperature
  nu: 0.5,                // Diffusion
  spectralGap: 0.381966,  // 1 - φ⁻¹ = 1/φ²
  lambdaMax: PHI,         // Dominant eigenvalue
  emergenceRate: 0.8,     // Oscillation strength
  bootstrapEnergy: 0.5,   // Base field amplitude
  graceScale: PHI,        // Grace operator scaling
  resonanceFrequency: PHI, // Time evolution speed
  defectPos: null,        // [x, y, z] for forced defect injection

  // Render controls (used by shader via uniforms; kept here so params are centralized)
  renderMode: 0,          // 0 = emergent field surface (default), 1 = braided lattice emphasis
  braidStrength: 0.35,    // How strongly phase-locked braid nodes influence SDF + emissive coloring
  braidSharpness: 10.0,   // Higher = thinner, crisper braid strands (standing-wave nodes)
  braidScale: 1.0         // Spatial scaling for braid phase construction
};

/**
 * Compute φ-structured resonance at a point
 * 
 * Three incommensurable modes create quasi-periodic behavior:
 *   - φ mode (wavelength φ)
 *   - φ² mode (wavelength φ²)  
 *   - unit mode (wavelength 1)
 */
export function computeResonance(x, y, z) {
  // Mode 1: φ-wavelength
  const mode_phi = Math.cos(x / PHI) * Math.cos(y / PHI) * Math.cos(z / PHI);
  
  // Mode 2: φ²-wavelength
  const mode_phiSq = Math.cos(x / (PHI * PHI)) * 
                     Math.cos(y / (PHI * PHI)) * 
                     Math.cos(z / (PHI * PHI));
  
  // Mode 3: unit wavelength
  const mode_unit = Math.cos(x) * Math.cos(y) * Math.cos(z);
  
  // φ-duality weighted combination
  const coherence = PHI_INV * (1 + mode_phi) +
                    PHI_INV * (1 + mode_phiSq) / 2 +
                    PHI_INV * (1 + mode_unit);
  
  return coherence;
}

/**
 * Generate the 16-component Clifford field
 * 
 * This mirrors the logic in resonance_driven_engine.js:mapToCliffordField()
 * 
 * @param time - Current time for evolution
 * @param params - Control parameters (beta, nu, gamma, etc.)
 */
export function generateCliffordField(x, y, z, time, params = {}) {
  // Merge with defaults
  const p = { ...DEFAULT_PARAMS, ...params };
  
  const components = new Float32Array(16);
  
  // ========================================================================
  // 1. RESONANCE BASE: φ-structured resonance for all 16 components
  // ========================================================================
  const spatialScale = Math.sqrt(p.lambdaMax);
  const spatialFreqMod = Math.sqrt(p.resonanceFrequency / PHI);
  const timePhase = time * p.resonanceFrequency * 0.3;
  
  for (let i = 0; i < 16; i++) {
    const ix = (i % 4) * 0.5 * spatialScale * spatialFreqMod;
    const iy = Math.floor(i / 4) * 0.5 * spatialScale * spatialFreqMod;
    const iz = timePhase * 0.1;
    
    const resonance = computeResonance(ix, iy, iz);
    components[i] = resonance * Math.sqrt(p.bootstrapEnergy) * 0.4;
  }
  
  // ========================================================================
  // INJECT DEFECT (Counter-Factual Test)
  // Force a zero at a specific location to test stability response
  // ========================================================================
  if (p.defectPos) {
    // Current point in 3D space (this function generates field at origin for now)
    // To properly test this, we'd need spatial field generation.
    // For the visualization, we simulate the *global effect* of a local defect:
    // It creates high-frequency noise (stress) in the bivector/trivector terms.
    
    // Distance of "defect" from "critical line" (center)
    const defectDist = Math.sqrt(p.defectPos[0]**2 + p.defectPos[1]**2);
    
    // Stress increases exponentially with distance from center
    const stress = Math.exp(defectDist * 2.0) * 0.1;
    
    // Add stress to higher grades (turbulence)
    for (let i = 4; i < 16; i++) {
      components[i] += (Math.random() - 0.5) * stress;
    }
  }

  // ========================================================================
  // 2. GRACE PROJECTION: scalar + φ⁻¹ × pseudoscalar
  // ========================================================================
  const scalar = components[0];
  const pseudoscalar = components[15];
  const graceCore = scalar + PHI_INV * pseudoscalar;
  
  const graceAmplitude = Math.sqrt(p.bootstrapEnergy) * 0.5;
  components[0] = graceCore * graceAmplitude;
  components[15] = graceCore * PHI_INV * graceAmplitude;
  
  // ========================================================================
  // 3. TEMPERATURE (β): Scalar/vector thermal fluctuations
  // ========================================================================
  const scalarAmp = 0.3 * p.beta;
  components[0] += Math.sin(timePhase) * scalarAmp;
  components[1] += Math.cos(timePhase) * scalarAmp * 0.8;
  components[2] += Math.sin(timePhase * PHI) * scalarAmp * 0.8;
  components[3] += Math.cos(timePhase * PHI) * scalarAmp * 0.8;
  
  // ========================================================================
  // 4. DIFFUSION (ν): Bivector rotation/mixing - CREATES TOROIDAL PATTERNS
  // ========================================================================
  const bivectorAmp = 0.4 * p.nu * 8.0 * Math.sqrt(p.emergenceRate);
  
  // Lissajous patterns create toroidal geometry
  const lissajousX = Math.sin(timePhase * 0.7);
  const lissajousY = Math.cos(timePhase * 0.9);
  const lissajousZ = Math.sin(timePhase * 1.1);
  
  components[4] += lissajousX * bivectorAmp;
  components[5] += lissajousY * bivectorAmp;
  components[6] += lissajousZ * bivectorAmp;
  components[7] += (lissajousX * lissajousY) * bivectorAmp * 0.8; // Toroidal coupling
  
  // ========================================================================
  // 5. SPECTRAL GAP (γ): Trivector complexity
  // ========================================================================
  const trivectorAmp = 0.4 * p.spectralGap * 10.0 * Math.sqrt(p.emergenceRate);
  
  components[8] += Math.sin(timePhase * 1.4) * trivectorAmp;
  components[9] += Math.cos(timePhase * 1.6) * trivectorAmp;
  components[10] += Math.sin(timePhase * 1.8) * trivectorAmp;
  components[11] += Math.cos(timePhase * 2.0) * trivectorAmp;
  components[12] += Math.sin(timePhase * 2.2) * trivectorAmp * 0.8;
  components[13] += Math.cos(timePhase * 2.4) * trivectorAmp * 0.8;
  components[14] += Math.sin(timePhase * 2.6) * trivectorAmp * 0.8;
  
  // ========================================================================
  // 6. EIGENVALUE (λ_max): Pseudoscalar geometric structure
  // ========================================================================
  const pseudoAmp = 0.3 * p.lambdaMax;
  components[15] += Math.sin(timePhase * 2.8) * pseudoAmp * 0.6 + 
                    Math.cos(timePhase * 3.0) * pseudoAmp * 0.6;
  
  // ========================================================================
  // 7. GRACE OPERATOR: Apply φ⁻¹ scaling (contraction)
  // ========================================================================
  const graceScaleFactor = Math.sqrt(p.graceScale / PHI);
  const graceOperator = PHI_INV * graceScaleFactor;
  
  for (let i = 0; i < 16; i++) {
    components[i] *= graceOperator;
  }
  
  // ========================================================================
  // 8. CLAMP to [-1, 1] for texture encoding
  // ========================================================================
  for (let i = 0; i < 16; i++) {
    components[i] = Math.max(-1.0, Math.min(1.0, components[i]));
  }
  
  return new Multivector(components);
}

/**
 * Spectral gap constant
 */
export const SPECTRAL_GAP = 1 - PHI_INV; // = 1/φ² ≈ 0.382

/**
 * Dominant eigenvalue
 */
export const LAMBDA_MAX = PHI;

/**
 * Compute the resonance gradient (for flow direction)
 */
export function computeResonanceGradient(x, y, z) {
  const h = 0.001;
  
  const dx = (computeResonance(x + h, y, z) - computeResonance(x - h, y, z)) / (2 * h);
  const dy = (computeResonance(x, y + h, z) - computeResonance(x, y - h, z)) / (2 * h);
  const dz = (computeResonance(x, y, z + h) - computeResonance(x, y, z - h)) / (2 * h);
  
  return [dx, dy, dz];
}

/**
 * Time-evolved resonance (for flow visualization)
 */
export function evolveResonance(x, y, z, t, evolutionRate = 1.0) {
  const omega1 = PHI * evolutionRate;
  const omega2 = PHI * PHI * evolutionRate;
  const omega3 = 1.0 * evolutionRate;
  
  const mode_phi = Math.cos(x / PHI + omega1 * t) *
                   Math.cos(y / PHI + omega1 * t * PHI_INV) *
                   Math.cos(z / PHI + omega1 * t * PHI_INV * PHI_INV);
  
  const mode_phiSq = Math.cos(x / (PHI * PHI) + omega2 * t) *
                     Math.cos(y / (PHI * PHI) + omega2 * t * PHI_INV) *
                     Math.cos(z / (PHI * PHI) + omega2 * t * PHI_INV * PHI_INV);
  
  const mode_unit = Math.cos(x + omega3 * t) *
                    Math.cos(y + omega3 * t * PHI_INV) *
                    Math.cos(z + omega3 * t * PHI_INV * PHI_INV);
  
  return PHI_INV * (1 + mode_phi) +
         PHI_INV * (1 + mode_phiSq) / 2 +
         PHI_INV * (1 + mode_unit);
}
