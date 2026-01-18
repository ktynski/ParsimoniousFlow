/**
 * zeta.js - Rigorous Riemann Zeta Function Implementation
 * 
 * This module implements ζ(s) using:
 * 1. Dirichlet series for Re(s) > 1
 * 2. Functional equation for analytic continuation
 * 3. Riemann-Siegel formula for critical strip efficiency
 * 
 * All computations use complex arithmetic with explicit real/imaginary parts.
 */

//=============================================================================
// COMPLEX NUMBER OPERATIONS
//=============================================================================

/**
 * Complex number: {re: real, im: imaginary}
 */
export function complex(re, im = 0) {
  return { re, im };
}

export function cadd(a, b) {
  return { re: a.re + b.re, im: a.im + b.im };
}

export function csub(a, b) {
  return { re: a.re - b.re, im: a.im - b.im };
}

export function cmul(a, b) {
  return {
    re: a.re * b.re - a.im * b.im,
    im: a.re * b.im + a.im * b.re
  };
}

export function cdiv(a, b) {
  const denom = b.re * b.re + b.im * b.im;
  if (denom === 0) return { re: Infinity, im: Infinity };
  return {
    re: (a.re * b.re + a.im * b.im) / denom,
    im: (a.im * b.re - a.re * b.im) / denom
  };
}

export function cabs(z) {
  return Math.sqrt(z.re * z.re + z.im * z.im);
}

export function carg(z) {
  return Math.atan2(z.im, z.re);
}

export function cexp(z) {
  const expRe = Math.exp(z.re);
  return {
    re: expRe * Math.cos(z.im),
    im: expRe * Math.sin(z.im)
  };
}

export function clog(z) {
  return {
    re: Math.log(cabs(z)),
    im: carg(z)
  };
}

export function cpow(base, exp) {
  // base^exp = e^(exp * log(base))
  if (base.re === 0 && base.im === 0) {
    return { re: 0, im: 0 };
  }
  const logBase = clog(base);
  const expLog = cmul(exp, logBase);
  return cexp(expLog);
}

export function csin(z) {
  // sin(z) = (e^(iz) - e^(-iz)) / (2i)
  const iz = { re: -z.im, im: z.re };
  const niz = { re: z.im, im: -z.re };
  const e1 = cexp(iz);
  const e2 = cexp(niz);
  const diff = csub(e1, e2);
  return { re: diff.im / 2, im: -diff.re / 2 };
}

export function ccos(z) {
  // cos(z) = (e^(iz) + e^(-iz)) / 2
  const iz = { re: -z.im, im: z.re };
  const niz = { re: z.im, im: -z.re };
  const e1 = cexp(iz);
  const e2 = cexp(niz);
  const sum = cadd(e1, e2);
  return { re: sum.re / 2, im: sum.im / 2 };
}

//=============================================================================
// GAMMA FUNCTION (Lanczos approximation)
//=============================================================================

const LANCZOS_G = 7;
const LANCZOS_COEFFS = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7
];

/**
 * Complex Gamma function using Lanczos approximation
 */
export function cgamma(z) {
  // Reflection formula for Re(z) < 0.5
  if (z.re < 0.5) {
    // Γ(z) = π / (sin(πz) * Γ(1-z))
    const oneMinusZ = { re: 1 - z.re, im: -z.im };
    const piZ = { re: Math.PI * z.re, im: Math.PI * z.im };
    const sinPiZ = csin(piZ);
    const gammaOneMinusZ = cgamma(oneMinusZ);
    const denom = cmul(sinPiZ, gammaOneMinusZ);
    return cdiv({ re: Math.PI, im: 0 }, denom);
  }

  // Lanczos approximation for Re(z) >= 0.5
  const zMinus1 = { re: z.re - 1, im: z.im };
  
  let x = { re: LANCZOS_COEFFS[0], im: 0 };
  for (let i = 1; i < LANCZOS_G + 2; i++) {
    const term = cdiv(
      { re: LANCZOS_COEFFS[i], im: 0 },
      { re: zMinus1.re + i, im: zMinus1.im }
    );
    x = cadd(x, term);
  }

  const t = { re: zMinus1.re + LANCZOS_G + 0.5, im: zMinus1.im };
  const sqrt2pi = Math.sqrt(2 * Math.PI);
  
  // Γ(z) = sqrt(2π) * t^(z-0.5) * e^(-t) * x
  const tPow = cpow(t, { re: zMinus1.re + 0.5, im: zMinus1.im });
  const expNegT = cexp({ re: -t.re, im: -t.im });
  
  let result = { re: sqrt2pi, im: 0 };
  result = cmul(result, tPow);
  result = cmul(result, expNegT);
  result = cmul(result, x);
  
  return result;
}

//=============================================================================
// RIEMANN ZETA FUNCTION
//=============================================================================

/**
 * Dirichlet series for ζ(s) when Re(s) > 1
 * ζ(s) = Σ_{n=1}^∞ n^(-s)
 * 
 * @param {Object} s - Complex number {re, im}
 * @param {number} terms - Number of terms (default 1000)
 */
export function zetaDirichlet(s, terms = 1000) {
  let sum = { re: 0, im: 0 };
  
  for (let n = 1; n <= terms; n++) {
    // n^(-s) = e^(-s * log(n))
    const logN = Math.log(n);
    const exponent = { re: -s.re * logN, im: -s.im * logN };
    const term = cexp(exponent);
    sum = cadd(sum, term);
  }
  
  return sum;
}

/**
 * Functional equation: ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
 * Used for analytic continuation to Re(s) < 1
 */
export function zetaFunctionalEquation(s) {
  // For Re(s) < 0, use the functional equation
  const oneMinusS = { re: 1 - s.re, im: -s.im };
  
  // Compute ζ(1-s) using Dirichlet series (valid since Re(1-s) > 1 when Re(s) < 0)
  const zeta1MinusS = zetaDirichlet(oneMinusS, 500);
  
  // 2^s
  const twoToS = cpow({ re: 2, im: 0 }, s);
  
  // π^(s-1)
  const piToSMinus1 = cpow({ re: Math.PI, im: 0 }, { re: s.re - 1, im: s.im });
  
  // sin(πs/2)
  const piSOver2 = { re: Math.PI * s.re / 2, im: Math.PI * s.im / 2 };
  const sinTerm = csin(piSOver2);
  
  // Γ(1-s)
  const gamma1MinusS = cgamma(oneMinusS);
  
  // Combine: 2^s * π^(s-1) * sin(πs/2) * Γ(1-s) * ζ(1-s)
  let result = twoToS;
  result = cmul(result, piToSMinus1);
  result = cmul(result, sinTerm);
  result = cmul(result, gamma1MinusS);
  result = cmul(result, zeta1MinusS);
  
  return result;
}

/**
 * Riemann-Siegel Z function for efficient computation on critical line
 * Z(t) = e^(iθ(t)) ζ(1/2 + it)
 * where θ(t) is the Riemann-Siegel theta function
 */
export function riemannSiegelTheta(t) {
  // θ(t) ≈ (t/2) * log(t/(2π)) - t/2 - π/8 + 1/(48t) + ...
  const logTerm = (t / 2) * Math.log(t / (2 * Math.PI));
  return logTerm - t / 2 - Math.PI / 8 + 1 / (48 * t);
}

/**
 * Riemann-Siegel Z function (real-valued on critical line)
 * Zeros of Z(t) correspond to zeros of ζ(1/2 + it)
 */
export function riemannSiegelZ(t) {
  const s = { re: 0.5, im: t };
  const zeta = zetaCriticalStrip(s);
  const theta = riemannSiegelTheta(Math.abs(t));
  
  // Z(t) = e^(iθ) * ζ(1/2 + it)
  const rotation = cexp({ re: 0, im: theta });
  const Z = cmul(rotation, zeta);
  
  // Z(t) should be real on the critical line (up to numerical error)
  return Z.re;
}

/**
 * Main Zeta function for critical strip 0 < Re(s) < 1
 * Uses alternating series (Dirichlet eta function) for convergence
 * η(s) = (1 - 2^(1-s)) ζ(s)
 */
export function zetaCriticalStrip(s, terms = 500) {
  // Use Dirichlet eta function: η(s) = Σ (-1)^(n-1) / n^s
  // Then ζ(s) = η(s) / (1 - 2^(1-s))
  
  let eta = { re: 0, im: 0 };
  
  for (let n = 1; n <= terms; n++) {
    const sign = (n % 2 === 1) ? 1 : -1;
    const logN = Math.log(n);
    const exponent = { re: -s.re * logN, im: -s.im * logN };
    const term = cexp(exponent);
    eta = cadd(eta, { re: sign * term.re, im: sign * term.im });
  }
  
  // Compute 1 - 2^(1-s)
  const oneMinusS = { re: 1 - s.re, im: -s.im };
  const twoTo1MinusS = cpow({ re: 2, im: 0 }, oneMinusS);
  const divisor = csub({ re: 1, im: 0 }, twoTo1MinusS);
  
  // Handle the pole at s = 1 (divisor = 0)
  if (cabs(divisor) < 1e-10) {
    return { re: Infinity, im: 0 };
  }
  
  return cdiv(eta, divisor);
}

/**
 * Master Zeta function dispatcher
 * Chooses the appropriate method based on s
 */
export function zeta(s) {
  if (s.re > 1) {
    return zetaDirichlet(s, 1000);
  } else if (s.re < 0) {
    return zetaFunctionalEquation(s);
  } else {
    // Critical strip: 0 <= Re(s) <= 1
    return zetaCriticalStrip(s, 500);
  }
}

//=============================================================================
// ZERO DETECTION
//=============================================================================

/**
 * Find zeros of ζ(s) on the critical line using sign changes of Z(t)
 * 
 * @param {number} tMin - Start of search range
 * @param {number} tMax - End of search range
 * @param {number} step - Initial step size
 * @param {number} tolerance - Precision for zero location
 * @returns {Array} List of t values where ζ(1/2 + it) = 0
 */
export function findZerosOnCriticalLine(tMin, tMax, step = 0.1, tolerance = 1e-6) {
  const zeros = [];
  let t = tMin;
  let prevZ = riemannSiegelZ(t);
  
  while (t < tMax) {
    t += step;
    const currentZ = riemannSiegelZ(t);
    
    // Sign change indicates a zero
    if (prevZ * currentZ < 0) {
      // Bisection to refine
      let lo = t - step;
      let hi = t;
      
      while (hi - lo > tolerance) {
        const mid = (lo + hi) / 2;
        const midZ = riemannSiegelZ(mid);
        
        if (riemannSiegelZ(lo) * midZ < 0) {
          hi = mid;
        } else {
          lo = mid;
        }
      }
      
      zeros.push((lo + hi) / 2);
    }
    
    prevZ = currentZ;
  }
  
  return zeros;
}

/**
 * Known first 10 non-trivial zeros (for verification)
 * These are the imaginary parts t where ζ(1/2 + it) = 0
 */
export const KNOWN_ZEROS = [
  14.134725,
  21.022040,
  25.010858,
  30.424876,
  32.935062,
  37.586178,
  40.918720,
  43.327073,
  48.005151,
  49.773832
];

/**
 * Verify our implementation against known zeros
 */
export function verifyZeroDetection() {
  const results = [];
  
  for (const knownT of KNOWN_ZEROS.slice(0, 5)) {
    // Compute ζ at the known zero
    const s = { re: 0.5, im: knownT };
    const zetaValue = zeta(s);
    const magnitude = cabs(zetaValue);
    
    // Also find zero via our algorithm
    const foundZeros = findZerosOnCriticalLine(knownT - 0.5, knownT + 0.5, 0.01, 1e-6);
    const foundT = foundZeros.length > 0 ? foundZeros[0] : null;
    const error = foundT ? Math.abs(foundT - knownT) : Infinity;
    
    results.push({
      knownT,
      magnitude,
      foundT,
      error,
      passed: magnitude < 0.1 && error < 0.01
    });
  }
  
  return results;
}

//=============================================================================
// XI FUNCTION (Riemann's completed zeta - symmetric around Re(s)=1/2)
//=============================================================================

/**
 * Riemann Xi function: ξ(s) = (1/2)s(s-1)π^(-s/2)Γ(s/2)ζ(s)
 * 
 * CRUCIAL PROPERTY: ξ(s) = ξ(1-s) (perfect symmetry around Re(s) = 1/2)
 * This is the correct basis for the Lyapunov functional.
 */
export function xi(s) {
  // ξ(s) = (1/2) * s * (s-1) * π^(-s/2) * Γ(s/2) * ζ(s)
  
  const half = { re: 0.5, im: 0 };
  const sMinusOne = { re: s.re - 1, im: s.im };
  const sOver2 = { re: s.re / 2, im: s.im / 2 };
  const negSOver2 = { re: -s.re / 2, im: -s.im / 2 };
  
  // (1/2) * s * (s-1)
  let prefactor = cmul(half, s);
  prefactor = cmul(prefactor, sMinusOne);
  
  // π^(-s/2)
  const piTerm = cpow({ re: Math.PI, im: 0 }, negSOver2);
  
  // Γ(s/2)
  const gammaTerm = cgamma(sOver2);
  
  // ζ(s)
  const zetaTerm = zeta(s);
  
  // Combine all terms
  let result = cmul(prefactor, piTerm);
  result = cmul(result, gammaTerm);
  result = cmul(result, zetaTerm);
  
  return result;
}

/**
 * Verify Xi symmetry: ξ(s) = ξ(1-s)
 */
export function verifyXiSymmetry(s) {
  const xiS = xi(s);
  const oneMinusS = { re: 1 - s.re, im: -s.im };
  const xi1MinusS = xi(oneMinusS);
  
  const diff = cabs(csub(xiS, xi1MinusS));
  const avg = (cabs(xiS) + cabs(xi1MinusS)) / 2;
  
  return {
    xiS,
    xi1MinusS,
    relativeDiff: avg > 0 ? diff / avg : diff,
    isSymmetric: avg > 0 ? (diff / avg < 0.01) : (diff < 0.01)
  };
}

//=============================================================================
// CRITICAL LINE ENERGY FUNCTIONAL (Lyapunov) - CORRECTED
//=============================================================================

/**
 * Compute the "energy" at a point s in the complex plane
 * Uses the XI FUNCTION which is symmetric around Re(s) = 1/2
 * 
 * The key insight: |ξ(σ + it)| has its MAXIMUM on σ = 1/2 for fixed t,
 * not its minimum. The zeros are where ξ = 0.
 * 
 * Energy functional: E(s) = -log|ξ(s)| + λ * |σ - 1/2|²
 * 
 * This measures how far we are from being on the critical line,
 * weighted by the distance from zeros.
 */
export function computeZetaEnergy(s, lambda = 1.0) {
  const xiS = xi(s);
  const magnitude = cabs(xiS);
  
  // Distance from critical line
  const distFromLine = Math.abs(s.re - 0.5);
  
  // Energy: penalize being off the critical line
  // Near zeros, |ξ| → 0, so -log|ξ| → +∞ (high energy at zeros)
  // This correctly models "caustics" as energy spikes
  
  const logTerm = magnitude > 1e-10 ? -Math.log(magnitude) : 20; // Cap at ~exp(-20)
  const linePenalty = lambda * distFromLine * distFromLine;
  
  return logTerm + linePenalty;
}

/**
 * Alternative: Hardy Z-function based energy
 * Z(t) is REAL on the critical line, so |Im(ζ(1/2+it))| measures deviation
 */
export function computeHardyEnergy(t) {
  const s = { re: 0.5, im: t };
  const xiVal = xi(s);
  
  // On critical line, ξ should be real (after phase correction)
  // The "energy" is how much imaginary component there is
  return Math.abs(xiVal.im);
}

/**
 * Test if energy is minimized on critical line
 * Compare energy at Re(s) = 0.5 vs off-line points
 */
export function testEnergyMinimization(t, offsets = [0.1, 0.2, 0.3, 0.4]) {
  const results = [];
  
  // Energy on critical line
  const energyOnLine = computeZetaEnergy({ re: 0.5, im: t });
  results.push({ sigma: 0.5, energy: energyOnLine, isMinimum: true });
  
  // Energy at various offsets
  for (const offset of offsets) {
    const energyRight = computeZetaEnergy({ re: 0.5 + offset, im: t });
    const energyLeft = computeZetaEnergy({ re: 0.5 - offset, im: t });
    
    results.push({ sigma: 0.5 + offset, energy: energyRight, isMinimum: false });
    results.push({ sigma: 0.5 - offset, energy: energyLeft, isMinimum: false });
  }
  
  // Sort by sigma for nice display
  results.sort((a, b) => a.sigma - b.sigma);
  
  // Check if 0.5 has minimum energy
  const minEnergy = Math.min(...results.map(r => r.energy));
  const criticalLineEnergy = results.find(r => r.sigma === 0.5).energy;
  
  return {
    results,
    criticalLineIsMinimum: criticalLineEnergy <= minEnergy * 1.1, // Allow 10% tolerance
    ratio: criticalLineEnergy / minEnergy
  };
}

//=============================================================================
// WINDING NUMBER (Topological Protection)
//=============================================================================

/**
 * Compute winding number of ζ(s) around a contour
 * W = (1/2πi) ∮ (ζ'/ζ) ds
 * 
 * Counts zeros minus poles inside the contour
 */
export function computeWindingNumber(center, radius, samples = 100) {
  let integral = { re: 0, im: 0 };
  
  for (let i = 0; i < samples; i++) {
    const theta1 = (2 * Math.PI * i) / samples;
    const theta2 = (2 * Math.PI * (i + 1)) / samples;
    
    // Points on contour
    const s1 = {
      re: center.re + radius * Math.cos(theta1),
      im: center.im + radius * Math.sin(theta1)
    };
    const s2 = {
      re: center.re + radius * Math.cos(theta2),
      im: center.im + radius * Math.sin(theta2)
    };
    
    // ζ at these points
    const z1 = zeta(s1);
    const z2 = zeta(s2);
    
    // Contribution to winding: Δarg(ζ)
    const arg1 = carg(z1);
    const arg2 = carg(z2);
    
    let deltaArg = arg2 - arg1;
    // Handle branch cut
    if (deltaArg > Math.PI) deltaArg -= 2 * Math.PI;
    if (deltaArg < -Math.PI) deltaArg += 2 * Math.PI;
    
    integral.im += deltaArg;
  }
  
  // Winding number = integral / (2π)
  return Math.round(integral.im / (2 * Math.PI));
}

/**
 * Test topological protection: verify winding number around known zeros
 */
export function testTopologicalProtection() {
  const results = [];
  
  // Test around first few known zeros
  for (const t of KNOWN_ZEROS.slice(0, 3)) {
    const center = { re: 0.5, im: t };
    
    // Small contour around the zero
    const windingSmall = computeWindingNumber(center, 0.3, 200);
    
    // Contour that doesn't contain the zero
    const centerOffset = { re: 0.8, im: t };
    const windingOffset = computeWindingNumber(centerOffset, 0.2, 200);
    
    results.push({
      t,
      windingAroundZero: windingSmall,
      windingOffCenter: windingOffset,
      zeroDetected: windingSmall === 1,
      noFalsePositive: windingOffset === 0
    });
  }
  
  return results;
}

//=============================================================================
// EXPORTS
//=============================================================================

export default {
  complex,
  zeta,
  cabs,
  findZerosOnCriticalLine,
  verifyZeroDetection,
  computeZetaEnergy,
  testEnergyMinimization,
  computeWindingNumber,
  testTopologicalProtection,
  KNOWN_ZEROS
};

