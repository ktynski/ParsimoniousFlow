/**
 * torus_sdf.js - EMERGENT Geometry from Clifford Field
 * 
 * CRITICAL: NO imposed torus geometry. The toroidal structure EMERGES
 * from multi-scale field interference patterns:
 * 
 *   scale1 = (x + y + z)          -- Linear combination
 *   scale2 = (xy + yz + zx)       -- Bilinear (creates cross-terms)
 *   scale3 = (xyz)                -- Trilinear (creates triple products)
 * 
 * These naturally produce quasi-toroidal geometry when combined with
 * the bivector rotation patterns in the Clifford field.
 */

import { PHI, PHI_INV } from '../math/clifford.js';

/**
 * Compute the multi-scale coordinates used for field interference
 * These Cartesian products create emergent geometry WITHOUT imposing shapes
 */
export function computeScales(x, y, z) {
  return {
    scale1: (x + y + z) * 0.1,              // Linear
    scale2: (x*y + y*z + z*x) * 0.5,        // Bilinear
    scale3: x * y * z * 2.0                  // Trilinear
  };
}

/**
 * Sample the emergent SDF from field components
 * This is a JavaScript mirror of the shader logic for CPU-side analysis
 * 
 * @param pos - [x, y, z] position
 * @param components - 16-component Float32Array from Multivector
 */
export function sampleEmergentSDF(pos, components) {
  const [x, y, z] = pos;
  const { scale1, scale2, scale3 } = computeScales(x, y, z);
  
  // Unpack components by grade
  const scalar = components[0];
  const vectors = [components[1], components[2], components[3]];
  const bivectors1 = [components[4], components[5], components[6]]; // e01, e02, e03
  const bivector_e12 = components[7];
  const bivectors2 = [components[8], components[9], components[10]]; // e13, e23, e012
  const trivector_e013 = components[11];
  const trivectors1 = [components[12], components[13], components[14]];
  const pseudoscalar = components[15];
  
  // Bootstrap coherence (L1 norm - no spherical bias)
  let bootstrap_coherence = 0;
  for (let i = 0; i < 16; i++) {
    bootstrap_coherence += Math.abs(components[i]);
  }
  
  // === MULTI-SCALE FIELD LAYERS ===
  
  // Layer 1: Scalar + Vector + Bivector
  const dot_vectors_pos = vectors[0]*x + vectors[1]*y + vectors[2]*z;
  let field_layer1 = scalar * Math.cos(scale1) +
                     dot_vectors_pos * Math.sin(scale1) +
                     (bivectors1[0]*x*y + bivectors1[1]*y*z + bivectors1[2]*z*x) * 
                     Math.cos(scale1 * PHI);
  
  // Layer 2: Bivector + Trivector
  const dot_biv2_pos = bivectors2[0]*x + bivectors2[1]*y + bivectors2[2]*z;
  let field_layer2 = dot_biv2_pos * Math.sin(scale2) +
                     trivector_e013 * Math.cos(x * scale2) +
                     trivectors1[0] * Math.sin(y * scale2) +
                     trivectors1[1] * Math.cos(z * scale2);
  
  // Layer 3: Pseudoscalar + Trivector + Bivector cross
  let field_layer3 = pseudoscalar * Math.sin(scale3) +
                     trivectors1[2] * Math.cos(scale3 * PHI_INV) +
                     bivector_e12 * Math.sin(x * y * scale3);
  
  // === RECURSIVE INTERFERENCE ===
  const interference1 = field_layer1 * field_layer2;
  const interference2 = field_layer2 * field_layer3;
  const interference3 = field_layer3 * field_layer1;
  
  // === CUMULATIVE DISTANCE ===
  const coherence_factor = bootstrap_coherence / 16.0;
  
  let pure_field_distance = 
    field_layer1 * (0.3 + coherence_factor * 0.2) +
    field_layer2 * (0.2 + coherence_factor * 0.3) +
    field_layer3 * (0.1 + coherence_factor * 0.4) +
    interference1 * (0.05 + coherence_factor * 0.1) +
    interference2 * (0.03 + coherence_factor * 0.15) +
    interference3 * (0.02 + coherence_factor * 0.2);
  
  // === COMPONENT COMPLEXITY ===
  const component_complexity =
    Math.abs(scalar) * 0.1 +
    (Math.abs(vectors[0]) + Math.abs(vectors[1]) + Math.abs(vectors[2])) * 0.08 +
    (Math.abs(bivectors1[0]) + Math.abs(bivectors1[1]) + Math.abs(bivectors1[2])) * 0.06 +
    Math.abs(bivector_e12) * 0.05 +
    (Math.abs(bivectors2[0]) + Math.abs(bivectors2[1]) + Math.abs(bivectors2[2])) * 0.04 +
    Math.abs(trivector_e013) * 0.03 +
    (Math.abs(trivectors1[0]) + Math.abs(trivectors1[1]) + Math.abs(trivectors1[2])) * 0.025 +
    Math.abs(pseudoscalar) * 0.01;
  
  // === SELF-REFERENCE PATTERNS ===
  let self_reference =
    scalar * dot_vectors_pos * 0.01 +
    (vectors[0]*bivectors1[0] + vectors[1]*bivectors1[1] + vectors[2]*bivectors1[2]) * Math.sin(scale2) * 0.02 +
    trivector_e013 * trivectors1[0] * Math.cos(scale3) * 0.01 +
    bivectors1[0] * y * z * Math.sin(scale1) * 0.03 +
    bivectors1[1] * z * x * Math.cos(scale2) * 0.03 +
    bivectors1[2] * x * y * Math.sin(scale3) * 0.03 +
    vectors[0] * bivectors2[1] * z * 0.02 +
    vectors[1] * bivectors2[2] * x * 0.02 +
    vectors[2] * bivectors2[0] * y * 0.02 +
    trivector_e013 * bivectors2[0] * Math.sin(scale1) * 0.015 +
    trivectors1[0] * trivectors1[1] * Math.cos(scale2) * 0.015 +
    trivectors1[2] * pseudoscalar * Math.sin(scale3) * 0.01;
  
  // === GRACE OPERATOR (additive) ===
  const grace_core = Math.abs(scalar) + PHI_INV * Math.abs(pseudoscalar);
  const grace_contribution = grace_core * PHI_INV * 0.1;
  
  let recursive_distance = pure_field_distance + grace_contribution;
  
  // === CONTINUOUS COMPLEXITY EMERGENCE ===
  let cumulative_self_reference = self_reference;
  const complexityField = bootstrap_coherence / 16.0;
  
  const secondOrderWeight = Math.sin(complexityField * PHI * Math.PI) * 0.1;
  cumulative_self_reference += self_reference * self_reference * secondOrderWeight;
  
  const thirdOrderWeight = Math.cos(complexityField * PHI * PHI * Math.PI) * 0.01;
  cumulative_self_reference += Math.pow(self_reference, 3) * thirdOrderWeight;
  
  const nonlinearWeight = Math.sin(complexityField * PHI * PHI * PHI * Math.PI) * 0.05;
  cumulative_self_reference += Math.sin(self_reference * component_complexity) * nonlinearWeight;
  
  recursive_distance += cumulative_self_reference + component_complexity * Math.sin(bootstrap_coherence);
  
  // === BIREFLECTION ===
  const mirrored_distance = -recursive_distance;
  const bireflection_distance = Math.min(Math.abs(recursive_distance), Math.abs(mirrored_distance));
  
  // === DIRECTIONAL ASYMMETRY ===
  const directional_asymmetry = 
    Math.sin(x * 0.5) * Math.cos(y * 0.3) * Math.sin(z * 0.7) * component_complexity * 0.1;
  
  // === FINAL DISTANCE ===
  const final_coherence = Math.max(bootstrap_coherence / 8.0, 0.5);
  const raw_dist = bireflection_distance * final_coherence + directional_asymmetry;
  
  return raw_dist * 0.8;
}

/**
 * Compute surface normal via gradient
 */
export function computeNormal(pos, components, epsilon = 0.01) {
  const dx = sampleEmergentSDF([pos[0] + epsilon, pos[1], pos[2]], components) -
             sampleEmergentSDF([pos[0] - epsilon, pos[1], pos[2]], components);
  const dy = sampleEmergentSDF([pos[0], pos[1] + epsilon, pos[2]], components) -
             sampleEmergentSDF([pos[0], pos[1] - epsilon, pos[2]], components);
  const dz = sampleEmergentSDF([pos[0], pos[1], pos[2] + epsilon], components) -
             sampleEmergentSDF([pos[0], pos[1], pos[2] - epsilon], components);
  
  const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
  if (len < 1e-10) return [0, 0, 1];
  
  return [dx/len, dy/len, dz/len];
}

/**
 * Analyze the emergent geometry at a point
 * Returns information about local field structure
 */
export function analyzeEmergentGeometry(pos, components) {
  const distance = sampleEmergentSDF(pos, components);
  const normal = computeNormal(pos, components);
  
  // Compute local curvature estimate via Laplacian
  const epsilon = 0.05;
  const d0 = sampleEmergentSDF(pos, components);
  const dxp = sampleEmergentSDF([pos[0] + epsilon, pos[1], pos[2]], components);
  const dxm = sampleEmergentSDF([pos[0] - epsilon, pos[1], pos[2]], components);
  const dyp = sampleEmergentSDF([pos[0], pos[1] + epsilon, pos[2]], components);
  const dym = sampleEmergentSDF([pos[0], pos[1] - epsilon, pos[2]], components);
  const dzp = sampleEmergentSDF([pos[0], pos[1], pos[2] + epsilon], components);
  const dzm = sampleEmergentSDF([pos[0], pos[1], pos[2] - epsilon], components);
  
  const laplacian = (dxp + dxm + dyp + dym + dzp + dzm - 6*d0) / (epsilon * epsilon);
  
  // Compute grade dominance
  const grades = [
    Math.abs(components[0]), // scalar
    Math.abs(components[1]) + Math.abs(components[2]) + Math.abs(components[3]), // vectors
    Math.abs(components[4]) + Math.abs(components[5]) + Math.abs(components[6]) + 
    Math.abs(components[7]) + Math.abs(components[8]) + Math.abs(components[9]) + 
    Math.abs(components[10]), // bivectors (7 components in this layout)
    Math.abs(components[11]) + Math.abs(components[12]) + Math.abs(components[13]) + 
    Math.abs(components[14]), // trivectors
    Math.abs(components[15]) // pseudoscalar
  ];
  
  const dominantGrade = grades.indexOf(Math.max(...grades));
  
  return {
    distance,
    normal,
    curvature: laplacian,
    gradeMagnitudes: grades,
    dominantGrade,
    isNearSurface: Math.abs(distance) < 0.1
  };
}
