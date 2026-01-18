/**
 * clifford.js - Cl(1,3) Clifford Algebra Implementation
 * 
 * The Clifford algebra Cl(1,3) is the algebra of spacetime.
 * Signature (1,3) means: e₀² = +1 (timelike), e₁² = e₂² = e₃² = -1 (spacelike)
 * 
 * This gives 2⁴ = 16 basis elements organized by grade.
 */

// Golden ratio constants
export const PHI = 1.618033988749894848;
export const PHI_INV = 0.618033988749894848;

// Basis element indices
export const BASIS = {
  // Grade 0: Scalar (1 element)
  SCALAR: 0,
  
  // Grade 1: Vectors (4 elements)
  E0: 1, E1: 2, E2: 3, E3: 4,
  
  // Grade 2: Bivectors (6 elements) - represent rotations/boosts
  E01: 5, E02: 6, E03: 7,  // Boost generators
  E12: 8, E13: 9, E23: 10, // Rotation generators
  
  // Grade 3: Trivectors (4 elements) - oriented volumes
  E012: 11, E013: 12, E023: 13, E123: 14,
  
  // Grade 4: Pseudoscalar (1 element) - oriented 4-volume
  E0123: 15
};

// Grade of each basis element
export const GRADES = [
  0,                    // scalar
  1, 1, 1, 1,          // vectors
  2, 2, 2, 2, 2, 2,    // bivectors
  3, 3, 3, 3,          // trivectors
  4                     // pseudoscalar
];

/**
 * Multivector in Cl(1,3)
 * Represented as a 16-element Float32Array
 */
export class Multivector {
  constructor(components = null) {
    if (components instanceof Float32Array && components.length === 16) {
      this.data = components;
    } else if (Array.isArray(components) && components.length === 16) {
      this.data = new Float32Array(components);
    } else {
      this.data = new Float32Array(16);
    }
  }
  
  // Create identity (scalar = 1)
  static identity() {
    const m = new Multivector();
    m.data[BASIS.SCALAR] = 1.0;
    return m;
  }
  
  // Create zero multivector
  static zero() {
    return new Multivector();
  }
  
  // Create from scalar
  static scalar(value) {
    const m = new Multivector();
    m.data[BASIS.SCALAR] = value;
    return m;
  }
  
  // Create from vector components (e₀, e₁, e₂, e₃)
  static vector(t, x, y, z) {
    const m = new Multivector();
    m.data[BASIS.E0] = t;
    m.data[BASIS.E1] = x;
    m.data[BASIS.E2] = y;
    m.data[BASIS.E3] = z;
    return m;
  }
  
  // Create from bivector (rotation plane)
  static bivector(components) {
    const m = new Multivector();
    m.data[BASIS.E01] = components.e01 || 0;
    m.data[BASIS.E02] = components.e02 || 0;
    m.data[BASIS.E03] = components.e03 || 0;
    m.data[BASIS.E12] = components.e12 || 0;
    m.data[BASIS.E13] = components.e13 || 0;
    m.data[BASIS.E23] = components.e23 || 0;
    return m;
  }
  
  // Create pseudoscalar
  static pseudoscalar(value) {
    const m = new Multivector();
    m.data[BASIS.E0123] = value;
    return m;
  }
  
  // Clone
  clone() {
    return new Multivector(new Float32Array(this.data));
  }
  
  // Get component
  get(index) {
    return this.data[index];
  }
  
  // Set component
  set(index, value) {
    this.data[index] = value;
  }
  
  // Get scalar part
  getScalar() {
    return this.data[BASIS.SCALAR];
  }
  
  // Get vector part as [t, x, y, z]
  getVector() {
    return [
      this.data[BASIS.E0],
      this.data[BASIS.E1],
      this.data[BASIS.E2],
      this.data[BASIS.E3]
    ];
  }
  
  // Get bivector parts
  getBivector() {
    return {
      // Boost generators (timelike-spacelike)
      e01: this.data[BASIS.E01],
      e02: this.data[BASIS.E02],
      e03: this.data[BASIS.E03],
      // Rotation generators (spacelike-spacelike)
      e12: this.data[BASIS.E12],
      e13: this.data[BASIS.E13],
      e23: this.data[BASIS.E23]
    };
  }
  
  // Get pseudoscalar part
  getPseudoscalar() {
    return this.data[BASIS.E0123];
  }
  
  // Project to grade k
  gradeProject(k) {
    const m = new Multivector();
    for (let i = 0; i < 16; i++) {
      if (GRADES[i] === k) {
        m.data[i] = this.data[i];
      }
    }
    return m;
  }
  
  // Addition
  add(other) {
    const result = new Multivector();
    for (let i = 0; i < 16; i++) {
      result.data[i] = this.data[i] + other.data[i];
    }
    return result;
  }
  
  // Scalar multiplication
  scale(s) {
    const result = new Multivector();
    for (let i = 0; i < 16; i++) {
      result.data[i] = this.data[i] * s;
    }
    return result;
  }
  
  // Magnitude squared (sum of squares of all components)
  normSquared() {
    let sum = 0;
    for (let i = 0; i < 16; i++) {
      sum += this.data[i] * this.data[i];
    }
    return sum;
  }
  
  // Magnitude
  norm() {
    return Math.sqrt(this.normSquared());
  }
  
  // Normalize to unit magnitude
  normalize() {
    const n = this.norm();
    if (n < 1e-10) return Multivector.zero();
    return this.scale(1.0 / n);
  }
  
  // L1 norm (sum of absolute values) - used for field activity
  l1Norm() {
    let sum = 0;
    for (let i = 0; i < 16; i++) {
      sum += Math.abs(this.data[i]);
    }
    return sum;
  }
  
  // Grade magnitudes
  gradeMagnitudes() {
    const mags = [0, 0, 0, 0, 0];
    for (let i = 0; i < 16; i++) {
      mags[GRADES[i]] += this.data[i] * this.data[i];
    }
    return mags.map(Math.sqrt);
  }
  
  /**
   * Geometric product (Clifford product)
   * This is the fundamental operation in Clifford algebra
   * 
   * For basis elements: eᵢeⱼ = -eⱼeᵢ (anticommute), eᵢ² = ±1 depending on signature
   */
  geometricProduct(other) {
    // Full geometric product is complex - we implement key cases
    // For visualization, we primarily use scalar/bivector/pseudoscalar
    
    const result = new Multivector();
    const a = this.data;
    const b = other.data;
    
    // Scalar × Scalar → Scalar
    result.data[0] += a[0] * b[0];
    
    // Scalar × Vector → Vector
    // Vector × Scalar → Vector
    for (let i = 1; i <= 4; i++) {
      result.data[i] += a[0] * b[i] + a[i] * b[0];
    }
    
    // Scalar × Bivector → Bivector (and vice versa)
    for (let i = 5; i <= 10; i++) {
      result.data[i] += a[0] * b[i] + a[i] * b[0];
    }
    
    // Scalar × Trivector → Trivector (and vice versa)
    for (let i = 11; i <= 14; i++) {
      result.data[i] += a[0] * b[i] + a[i] * b[0];
    }
    
    // Scalar × Pseudoscalar → Pseudoscalar (and vice versa)
    result.data[15] += a[0] * b[15] + a[15] * b[0];
    
    // Bivector × Bivector → Scalar + Bivector (simplified)
    // This is where rotation composition happens
    // e₁₂ × e₁₂ = e₁e₂e₁e₂ = -e₁e₁e₂e₂ = -(-1)(-1) = -1
    result.data[0] -= a[8] * b[8];  // e₁₂ × e₁₂
    result.data[0] -= a[9] * b[9];  // e₁₃ × e₁₃
    result.data[0] -= a[10] * b[10]; // e₂₃ × e₂₃
    
    // Pseudoscalar × Pseudoscalar → Scalar
    // e₀₁₂₃ × e₀₁₂₃ = (-1)³ × (+1) = -1 for signature (1,3)
    result.data[0] -= a[15] * b[15];
    
    return result;
  }
  
  /**
   * Reversion (reverses order of basis vectors in each term)
   * Grade k picks up factor (-1)^(k(k-1)/2)
   */
  reverse() {
    const result = this.clone();
    // Grade 0: +1
    // Grade 1: +1
    // Grade 2: -1
    for (let i = 5; i <= 10; i++) {
      result.data[i] = -result.data[i];
    }
    // Grade 3: -1
    for (let i = 11; i <= 14; i++) {
      result.data[i] = -result.data[i];
    }
    // Grade 4: +1
    return result;
  }
  
  /**
   * Convert to Float32Array for GPU upload
   */
  toFloat32Array() {
    return this.data;
  }
  
  /**
   * Convert to byte array for texture upload (maps [-1,1] to [0,255])
   */
  toUint8Array() {
    const bytes = new Uint8Array(16);
    for (let i = 0; i < 16; i++) {
      // Clamp to [-1, 1] then map to [0, 255]
      const clamped = Math.max(-1, Math.min(1, this.data[i]));
      bytes[i] = Math.floor((clamped + 1.0) * 127.5);
    }
    return bytes;
  }
  
  /**
   * Pretty print for debugging
   */
  toString() {
    const parts = [];
    if (Math.abs(this.data[0]) > 1e-6) parts.push(`${this.data[0].toFixed(3)}`);
    if (Math.abs(this.data[1]) > 1e-6) parts.push(`${this.data[1].toFixed(3)}e₀`);
    if (Math.abs(this.data[2]) > 1e-6) parts.push(`${this.data[2].toFixed(3)}e₁`);
    if (Math.abs(this.data[3]) > 1e-6) parts.push(`${this.data[3].toFixed(3)}e₂`);
    if (Math.abs(this.data[4]) > 1e-6) parts.push(`${this.data[4].toFixed(3)}e₃`);
    if (Math.abs(this.data[8]) > 1e-6) parts.push(`${this.data[8].toFixed(3)}e₁₂`);
    if (Math.abs(this.data[9]) > 1e-6) parts.push(`${this.data[9].toFixed(3)}e₁₃`);
    if (Math.abs(this.data[10]) > 1e-6) parts.push(`${this.data[10].toFixed(3)}e₂₃`);
    if (Math.abs(this.data[15]) > 1e-6) parts.push(`${this.data[15].toFixed(3)}e₀₁₂₃`);
    return parts.length > 0 ? parts.join(' + ') : '0';
  }
}

/**
 * Create a rotor (rotation element) from angle and bivector plane
 * 
 * Rotor R = cos(θ/2) + sin(θ/2)B where B is a unit bivector
 */
export function createRotor(angle, plane) {
  const halfAngle = angle / 2;
  const c = Math.cos(halfAngle);
  const s = Math.sin(halfAngle);
  
  const m = new Multivector();
  m.data[BASIS.SCALAR] = c;
  
  // Normalize plane and scale by sin
  const norm = Math.sqrt(plane.e12*plane.e12 + plane.e13*plane.e13 + plane.e23*plane.e23);
  if (norm > 1e-10) {
    m.data[BASIS.E12] = s * plane.e12 / norm;
    m.data[BASIS.E13] = s * plane.e13 / norm;
    m.data[BASIS.E23] = s * plane.e23 / norm;
  }
  
  return m;
}

