/**
 * shaders.js - GLSL Shaders for EMERGENT Clifford Field Visualization
 * 
 * CRITICAL: The geometry EMERGES from the field - NO imposed torus geometry.
 * The torus-like structure appears naturally from multi-scale field interference.
 */

// Vertex shader - fullscreen quad for raymarching
export const vertexShaderSource = `#version 300 es
precision highp float;

in vec2 aPosition;

out vec2 vUv;
out vec3 vRayDir;

uniform mat4 uInverseViewMatrix;
uniform mat4 uInverseProjectionMatrix;
uniform vec3 uCameraPosition;

void main() {
  vUv = aPosition * 0.5 + 0.5;
  
  // Compute ray direction in world space
  vec4 ndc = vec4(aPosition, 1.0, 1.0);
  vec4 viewPos = uInverseProjectionMatrix * ndc;
  viewPos /= viewPos.w;
  vec4 worldDir = uInverseViewMatrix * vec4(viewPos.xyz, 0.0);
  vRayDir = normalize(worldDir.xyz);
  
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// Fragment shader - PURE FIELD EMERGENCE (no imposed geometry)
export const fragmentShaderSource = `#version 300 es
precision highp float;

in vec2 vUv;
in vec3 vRayDir;

out vec4 fragColor;

uniform vec3 uCameraPosition;
uniform float uTime;
uniform float uMinDistance;
uniform float uMaxDistance;
uniform sampler2D uCliffordField;
uniform bool uHighlightCaustics; // New uniform
uniform int uRenderMode;         // 0 = emergent surface, 1 = braided lattice emphasis
uniform float uBraidStrength;
uniform float uBraidSharpness;
uniform float uBraidScale;

// Golden ratio constants
const float PHI = 1.618033988749;
const float PHI_INV = 0.618033988749;

#define MAX_STEPS 128
#define NORMAL_EPSILON 0.005 // Smaller epsilon for smoother normals

//=============================================================================
// PURE EMERGENT CLIFFORD FIELD SDF - NO IMPOSED GEOMETRY
//=============================================================================

float braidNode(float phase, float sharpness) {
  // Standing-wave node intensity: highest near sin(phase)=0
  // Using exp(-k*|sin|) yields thin "wires" as sharpness grows.
  return exp(-sharpness * abs(sin(phase)));
}

float phaseLock(float a, float b) {
  // 1 when phases align; 0 when maximally misaligned
  return 1.0 - abs(sin(a - b));
}

//=============================================================================
// Sample emergent field at a transformed position
// This is a helper to compute the full emergent field SDF at any position
//=============================================================================
float sampleEmergentFieldAt(vec3 pos, float scalar, vec3 vectors, vec3 bivectors1, float bivector_e12, 
                             vec3 bivectors2, float trivector_e013, vec3 trivectors1, float pseudoscalar,
                             float bootstrap_coherence) {
  // Multi-scale coordinates
  float scale1 = (pos.x + pos.y + pos.z) * 0.1;
  float scale2 = (pos.x * pos.y + pos.y * pos.z + pos.z * pos.x) * 0.5;
  float scale3 = (pos.x * pos.y * pos.z) * 2.0;
  
  // Field layers
  float field_layer1 = scalar * cos(scale1) +
                      dot(vectors, pos) * sin(scale1) +
                      (bivectors1.x * pos.x * pos.y + 
                       bivectors1.y * pos.y * pos.z + 
                       bivectors1.z * pos.z * pos.x) * cos(scale1 * PHI);
  
  float field_layer2 = dot(bivectors2, pos) * sin(scale2) +
                      trivector_e013 * cos(pos.x * scale2) +
                      trivectors1.x * sin(pos.y * scale2) +
                      trivectors1.y * cos(pos.z * scale2);
  
  float field_layer3 = pseudoscalar * sin(scale3) +
                      trivectors1.z * cos(scale3 * PHI_INV) +
                      bivector_e12 * sin(pos.x * pos.y * scale3);
  
  // Interference
  float interference1 = field_layer1 * field_layer2;
  float interference2 = field_layer2 * field_layer3;
  float interference3 = field_layer3 * field_layer1;
  
  // Cumulative distance
  float coherence_factor = bootstrap_coherence / 16.0;
  float pure_field_distance = 
    field_layer1 * (0.3 + coherence_factor * 0.2) +
    field_layer2 * (0.2 + coherence_factor * 0.3) +
    field_layer3 * (0.1 + coherence_factor * 0.4) +
    interference1 * (0.05 + coherence_factor * 0.1) +
    interference2 * (0.03 + coherence_factor * 0.15) +
    interference3 * (0.02 + coherence_factor * 0.2);
  
  // Component complexity
  float component_complexity =
    abs(scalar) * 0.1 +
    (abs(vectors.x) + abs(vectors.y) + abs(vectors.z)) * 0.08 +
    (abs(bivectors1.x) + abs(bivectors1.y) + abs(bivectors1.z)) * 0.06 +
    abs(bivector_e12) * 0.05 +
    (abs(bivectors2.x) + abs(bivectors2.y) + abs(bivectors2.z)) * 0.04 +
    abs(trivector_e013) * 0.03 +
    (abs(trivectors1.x) + abs(trivectors1.y) + abs(trivectors1.z)) * 0.025 +
    abs(pseudoscalar) * 0.01;
  
  // Self-reference
  float self_reference =
    scalar * dot(vectors, pos) * 0.01 +
    dot(vectors, bivectors1) * sin(scale2) * 0.02 +
    trivector_e013 * trivectors1.x * cos(scale3) * 0.01 +
    bivectors1.x * pos.y * pos.z * sin(scale1) * 0.03 +
    bivectors1.y * pos.z * pos.x * cos(scale2) * 0.03 +
    bivectors1.z * pos.x * pos.y * sin(scale3) * 0.03 +
    vectors.x * bivectors2.y * pos.z * 0.02 +
    vectors.y * bivectors2.z * pos.x * 0.02 +
    vectors.z * bivectors2.x * pos.y * 0.02 +
    trivector_e013 * bivectors2.x * sin(scale1) * 0.015 +
    trivectors1.x * trivectors1.y * cos(scale2) * 0.015 +
    trivectors1.z * pseudoscalar * sin(scale3) * 0.01;
  
  // Grace
  float grace_core = abs(scalar) + PHI_INV * abs(pseudoscalar);
  float grace_contribution = grace_core * PHI_INV * 0.1;
  
  float recursive_distance = pure_field_distance + grace_contribution;
  
  // Complexity emergence
  float complexityField = bootstrap_coherence / 16.0;
  float secondOrderWeight = sin(complexityField * PHI * 3.14159) * 0.1;
  float cumulative_self_reference = self_reference + self_reference * self_reference * secondOrderWeight;
  
  float thirdOrderWeight = cos(complexityField * PHI * PHI * 3.14159) * 0.01;
  cumulative_self_reference += pow(self_reference, 3.0) * thirdOrderWeight;
  
  float nonlinearWeight = sin(complexityField * PHI * PHI * PHI * 3.14159) * 0.05;
  cumulative_self_reference += sin(self_reference * component_complexity) * nonlinearWeight;
  
  recursive_distance += cumulative_self_reference + component_complexity * sin(bootstrap_coherence);
  
  // Bireflection
  float mirrored_distance = -recursive_distance;
  float bireflection_distance = min(abs(recursive_distance), abs(mirrored_distance));
  
  // Directional asymmetry
  float directional_asymmetry = 
    sin(pos.x * 0.5) * cos(pos.y * 0.3) * sin(pos.z * 0.7) * component_complexity * 0.1;
  
  float final_coherence = max(bootstrap_coherence / 8.0, 0.5);
  return (bireflection_distance * final_coherence + directional_asymmetry) * 0.8;
}

float sampleCliffordField(vec3 pos) {
  // Sample ALL 16 components from texture
  vec4 raw0 = texture(uCliffordField, vec2(0.0625, 0.5));
  vec4 raw1 = texture(uCliffordField, vec2(0.1875, 0.5));
  vec4 raw2 = texture(uCliffordField, vec2(0.3125, 0.5));
  vec4 raw3 = texture(uCliffordField, vec2(0.4375, 0.5));
  
  // Map [0,1] texture back to [-1,1]
  vec4 comp0 = (raw0 - 0.5) * 2.0;
  vec4 comp1 = (raw1 - 0.5) * 2.0;
  vec4 comp2 = (raw2 - 0.5) * 2.0;
  vec4 comp3 = (raw3 - 0.5) * 2.0;
  
  // Bootstrap coherence from ALL components (L1 norm - no spherical bias)
  float bootstrap_coherence = 
    abs(comp0.r) + abs(comp0.g) + abs(comp0.b) + abs(comp0.a) +
    abs(comp1.r) + abs(comp1.g) + abs(comp1.b) + abs(comp1.a) +
    abs(comp2.r) + abs(comp2.g) + abs(comp2.b) + abs(comp2.a) +
    abs(comp3.r) + abs(comp3.g) + abs(comp3.b) + abs(comp3.a);
  
  // Extract mathematical structure from ALL grades
  float scalar = comp0.r;                        // Grade-0
  vec3 vectors = comp0.gba;                      // Grade-1
  vec3 bivectors1 = comp1.rgb;                   // Grade-2 (e01, e02, e03)
  float bivector_e12 = comp1.a;                  // Grade-2
  vec3 bivectors2 = comp2.rgb;                   // Grade-2 (e13, e23, e012)
  float trivector_e013 = comp2.a;                // Grade-3
  vec3 trivectors1 = comp3.rgb;                  // Grade-3
  float pseudoscalar = comp3.a;                  // Grade-4
  
  //=========================================================================
  // MULTI-SCALE FIELD INTERFERENCE (NO SPHERICAL BIAS)
  // These Cartesian combinations create EMERGENT toroidal structure
  //=========================================================================
  float scale1 = (pos.x + pos.y + pos.z) * 0.1;              // Linear
  float scale2 = (pos.x * pos.y + pos.y * pos.z + pos.z * pos.x) * 0.5;  // Bilinear
  float scale3 = (pos.x * pos.y * pos.z) * 2.0;              // Trilinear
  
  //=========================================================================
  // COMPLEX FIELD SUPERPOSITION - Includes ALL grades
  //=========================================================================
  
  // Layer 1: Scalar + Vector + Bivector interactions
  float field_layer1 = scalar * cos(scale1) +
                      dot(vectors, pos) * sin(scale1) +
                      (bivectors1.x * pos.x * pos.y + 
                       bivectors1.y * pos.y * pos.z + 
                       bivectors1.z * pos.z * pos.x) * cos(scale1 * PHI);
  
  // Layer 2: Bivector + Trivector interactions
  float field_layer2 = dot(bivectors2, pos) * sin(scale2) +
                      trivector_e013 * cos(pos.x * scale2) +
                      trivectors1.x * sin(pos.y * scale2) +
                      trivectors1.y * cos(pos.z * scale2);
  
  // Layer 3: Pseudoscalar + Trivector + Bivector cross-terms
  float field_layer3 = pseudoscalar * sin(scale3) +
                      trivectors1.z * cos(scale3 * PHI_INV) +
                      bivector_e12 * sin(pos.x * pos.y * scale3);
  
  //=========================================================================
  // RECURSIVE INTERFERENCE PATTERNS
  //=========================================================================
  float interference1 = field_layer1 * field_layer2;
  float interference2 = field_layer2 * field_layer3;
  float interference3 = field_layer3 * field_layer1;
  
  //=========================================================================
  // CUMULATIVE MATHEMATICAL DISTANCE
  //=========================================================================
  float coherence_factor = bootstrap_coherence / 16.0;
  
  float pure_field_distance = 
    field_layer1 * (0.3 + coherence_factor * 0.2) +
    field_layer2 * (0.2 + coherence_factor * 0.3) +
    field_layer3 * (0.1 + coherence_factor * 0.4) +
    interference1 * (0.05 + coherence_factor * 0.1) +
    interference2 * (0.03 + coherence_factor * 0.15) +
    interference3 * (0.02 + coherence_factor * 0.2);
  
  //=========================================================================
  // SELF-REFERENCE PATTERNS (Ψ ≅ Hom(Ψ,Ψ))
  //=========================================================================
  float component_complexity =
    abs(scalar) * 0.1 +
    (abs(vectors.x) + abs(vectors.y) + abs(vectors.z)) * 0.08 +
    (abs(bivectors1.x) + abs(bivectors1.y) + abs(bivectors1.z)) * 0.06 +
    abs(bivector_e12) * 0.05 +
    (abs(bivectors2.x) + abs(bivectors2.y) + abs(bivectors2.z)) * 0.04 +
    abs(trivector_e013) * 0.03 +
    (abs(trivectors1.x) + abs(trivectors1.y) + abs(trivectors1.z)) * 0.025 +
    abs(pseudoscalar) * 0.01;
  
  // Asymmetric self-reference (breaks all symmetries)
  float self_reference =
    scalar * dot(vectors, pos) * 0.01 +
    dot(vectors, bivectors1) * sin(scale2) * 0.02 +
    trivector_e013 * trivectors1.x * cos(scale3) * 0.01 +
    // Asymmetric field breaking terms
    bivectors1.x * pos.y * pos.z * sin(scale1) * 0.03 +
    bivectors1.y * pos.z * pos.x * cos(scale2) * 0.03 +
    bivectors1.z * pos.x * pos.y * sin(scale3) * 0.03 +
    // Component cross-coupling
    vectors.x * bivectors2.y * pos.z * 0.02 +
    vectors.y * bivectors2.z * pos.x * 0.02 +
    vectors.z * bivectors2.x * pos.y * 0.02 +
    // Trivector cross-coupling
    trivector_e013 * bivectors2.x * sin(scale1) * 0.015 +
    trivectors1.x * trivectors1.y * cos(scale2) * 0.015 +
    trivectors1.z * pseudoscalar * sin(scale3) * 0.01;
  
  //=========================================================================
  // GRACE OPERATOR (additive, not multiplicative)
  //=========================================================================
  float grace_core = abs(scalar) + PHI_INV * abs(pseudoscalar);
  float grace_contribution = grace_core * PHI_INV * 0.1;
  
  float recursive_distance = pure_field_distance + grace_contribution;
  
  //=========================================================================
  // CONTINUOUS COMPLEXITY EMERGENCE (φ-modulated)
  //=========================================================================
  float cumulative_self_reference = self_reference;
  float complexityField = bootstrap_coherence / 16.0;
  
  // Second-order emergence
  float secondOrderWeight = sin(complexityField * PHI * 3.14159) * 0.1;
  cumulative_self_reference += self_reference * self_reference * secondOrderWeight;
  
  // Third-order emergence
  float thirdOrderWeight = cos(complexityField * PHI * PHI * 3.14159) * 0.01;
  cumulative_self_reference += pow(self_reference, 3.0) * thirdOrderWeight;
  
  // Nonlinear coupling
  float nonlinearWeight = sin(complexityField * PHI * PHI * PHI * 3.14159) * 0.05;
  cumulative_self_reference += sin(self_reference * component_complexity) * nonlinearWeight;
  
  recursive_distance += cumulative_self_reference + component_complexity * sin(bootstrap_coherence);
  
  //=========================================================================
  // BIREFLECTION: β∘β = 1_A (creates double-sheet caustic structure)
  //=========================================================================
  float mirrored_distance = -recursive_distance;
  float bireflection_distance = min(abs(recursive_distance), abs(mirrored_distance));
  
  //=========================================================================
  // DIRECTIONAL ASYMMETRY (breaks remaining spherical bias)
  //=========================================================================
  float directional_asymmetry = 
    sin(pos.x * 0.5) * cos(pos.y * 0.3) * sin(pos.z * 0.7) * component_complexity * 0.1;
  
  // Scale by bootstrap coherence
  float final_coherence = max(bootstrap_coherence / 8.0, 0.5);
  float raw_dist = bireflection_distance * final_coherence + directional_asymmetry;
  
  //=========================================================================
  // BRAIDED TOROIDAL VORTEX LATTICE (phase-coupled standing-wave attractors)
  //
  // In mode 1, we create a BRAIDED LATTICE by sampling the EMERGENT field
  // at multiple transformed positions/orientations, then combining them.
  // This shows how the same emergent geometry can form interlocking structures.
  // The braids are MADE OF the emergent field, not separate primitives.
  //=========================================================================
  if (uRenderMode == 1) {
    // Phase rotations for braiding (φ-scaled rates create incommensurable motion)
    float phase1 = uTime * 0.2 + pos.x * 0.3;
    float phase2 = uTime * 0.2 * PHI + pos.y * 0.3;
    float phase3 = uTime * 0.2 * PHI_INV + pos.z * 0.3;
    
    // Create 3 transformed instances of the emergent field
    // Each represents a different "attractor basin" or "memory tower level"
    
    // Instance 1: Rotated around Z-axis, offset along Y
    float c1 = cos(phase1);
    float s1 = sin(phase1);
    vec3 pos1 = vec3(
      pos.x * c1 - pos.y * s1,
      pos.x * s1 + pos.y * c1 + 0.4,  // Offset creates interlocking
      pos.z
    );
    
    // Instance 2: Rotated around X-axis, offset along Z
    float c2 = cos(phase2);
    float s2 = sin(phase2);
    vec3 pos2 = vec3(
      pos.x,
      pos.y * c2 - pos.z * s2,
      pos.y * s2 + pos.z * c2 + 0.4  // Offset creates interlocking
    );
    
    // Instance 3: Rotated around Y-axis, offset along X
    float c3 = cos(phase3);
    float s3 = sin(phase3);
    vec3 pos3 = vec3(
      pos.x * c3 + pos.z * s3 + 0.4,  // Offset creates interlocking
      pos.y,
      -pos.x * s3 + pos.z * c3
    );
    
    // Sample the FULL EMERGENT field at each transformed position
    // This creates multiple "copies" of the emergent geometry, braided together
    float dist1 = sampleEmergentFieldAt(pos1, scalar, vectors, bivectors1, bivector_e12,
                                        bivectors2, trivector_e013, trivectors1, pseudoscalar,
                                        bootstrap_coherence);
    float dist2 = sampleEmergentFieldAt(pos2, scalar, vectors, bivectors1, bivector_e12,
                                        bivectors2, trivector_e013, trivectors1, pseudoscalar,
                                        bootstrap_coherence);
    float dist3 = sampleEmergentFieldAt(pos3, scalar, vectors, bivectors1, bivector_e12,
                                        bivectors2, trivector_e013, trivectors1, pseudoscalar,
                                        bootstrap_coherence);
    
    // Phase-lock the instances (they resonate together)
    float lock12 = phaseLock(phase1, phase2);
    float lock23 = phaseLock(phase2, phase3);
    float lock31 = phaseLock(phase3, phase1);
    float lock = lock12 * lock23 * lock31;
    
    // Standing-wave nodes (where phases align) - these create visible "strands"
    float n1 = braidNode(phase1, uBraidSharpness);
    float n2 = braidNode(phase2, uBraidSharpness * PHI_INV);
    float n3 = braidNode(phase3, uBraidSharpness * 0.8);
    float nodeIntensity = (n1 + n2 + n3) / 3.0;
    
    // Union of the three emergent field instances (smooth minimum for braiding)
    float k = 0.35; // Smoothness parameter - controls how distinct the braids are
    float h12 = clamp(0.5 + 0.5 * (dist2 - dist1) / k, 0.0, 1.0);
    float dist12 = mix(dist2, dist1, h12) - k * h12 * (1.0 - h12);
    
    float h123 = clamp(0.5 + 0.5 * (dist3 - dist12) / k, 0.0, 1.0);
    float braidDist = mix(dist3, dist12, h123) - k * h123 * (1.0 - h123);
    
    // Enhance nodes (standing-wave attractors create visible "strands" along the braids)
    braidDist -= nodeIntensity * lock * uBraidStrength * 0.15;
    
    return braidDist;
  }

  // Conservative stepping factor
  return raw_dist * 0.8;
}

//=============================================================================
// RAYMARCHING
//=============================================================================

void main() {
  vec3 rayPos = uCameraPosition;
  vec3 rayDir = normalize(vRayDir);
  float totalDist = 0.0;
  
  for (int i = 0; i < MAX_STEPS; i++) {
    float dist = sampleCliffordField(rayPos);
    
    if (dist < uMinDistance) {
      // HIT SURFACE - Color based on field composition
      float depth = totalDist / uMaxDistance;
      
      // Sample field for coloring
      vec4 raw0 = texture(uCliffordField, vec2(0.0625, 0.5));
      vec4 raw1 = texture(uCliffordField, vec2(0.1875, 0.5));
      vec4 raw2 = texture(uCliffordField, vec2(0.3125, 0.5));
      vec4 raw3 = texture(uCliffordField, vec2(0.4375, 0.5));
      
      vec4 comp0 = (raw0 - 0.5) * 2.0;
      vec4 comp1 = (raw1 - 0.5) * 2.0;
      vec4 comp2 = (raw2 - 0.5) * 2.0;
      vec4 comp3 = (raw3 - 0.5) * 2.0;
      
      // Grade strengths
      float scalar_s = abs(comp0.r);
      float vector_s = abs(comp0.g) + abs(comp0.b) + abs(comp0.a);
      float bivector_s = abs(comp1.r) + abs(comp1.g) + abs(comp1.b) + abs(comp1.a) +
                        abs(comp2.r) + abs(comp2.g) + abs(comp2.b);
      float trivector_s = abs(comp2.a) + abs(comp3.r) + abs(comp3.g) + abs(comp3.b);
      float pseudo_s = abs(comp3.a);
      
      float total_s = scalar_s + vector_s + bivector_s + trivector_s + pseudo_s;
      
      vec3 color;
      if (total_s > 0.01) {
        // Continuous color from grade ratios
        float s = scalar_s / total_s;
        float v = vector_s / total_s;
        float b = bivector_s / total_s;
        float t = trivector_s / total_s;
        float p = pseudo_s / total_s;
        
        // Grade colors
        vec3 col_s = vec3(1.0, 0.1, 0.1);   // Scalar: Red
        vec3 col_v = vec3(1.0, 0.6, 0.0);   // Vector: Orange
        vec3 col_b = vec3(0.0, 1.0, 0.2);   // Bivector: Green
        vec3 col_t = vec3(0.0, 0.8, 1.0);   // Trivector: Cyan
        vec3 col_p = vec3(0.8, 0.0, 1.0);   // Pseudoscalar: Magenta
        
        color = s * col_s + v * col_v + b * col_b + t * col_t + p * col_p;
        color = pow(color, vec3(0.8));
        
        // Trivector emphasis
        if (trivector_s > bivector_s * 0.5) {
          color.r = min(1.0, color.r + trivector_s * 0.3);
          color.b = min(1.0, color.b + trivector_s * 0.4);
          color.g = max(0.0, color.g - trivector_s * 0.2);
        }
        
        // Pseudoscalar emphasis
        if (pseudo_s > scalar_s * 0.3) {
          color += vec3(pseudo_s * 0.5, pseudo_s * 0.5, pseudo_s * 0.3);
        }
      } else {
        color = vec3(0.1, 0.1, 0.2);
      }
      
      // Grace color modulation
      float grace_activation = abs(scalar_s - vector_s) / max(bivector_s, 0.01);
      float grace_factor = 1.0 + grace_activation * PHI_INV;
      color *= vec3(grace_factor, 1.0, 1.0 / grace_factor);
      
      // CAUSTIC HIGHLIGHTING (The "Zero" detection)
      // If enabled, highlight regions where the field is vanishingly small (singularities)
      if (uHighlightCaustics && total_s < 0.15) {
        // Singularities are "holes" in the field magnitude
        float intensity = (0.15 - total_s) / 0.15;
        vec3 causticColor = vec3(1.0, 0.9, 0.5); // Golden glow
        color = mix(color, causticColor * 2.0, intensity * intensity);
      }
      
      // Surface normal for lighting
      vec3 normal = normalize(vec3(
        sampleCliffordField(rayPos + vec3(NORMAL_EPSILON, 0.0, 0.0)) - sampleCliffordField(rayPos - vec3(NORMAL_EPSILON, 0.0, 0.0)),
        sampleCliffordField(rayPos + vec3(0.0, NORMAL_EPSILON, 0.0)) - sampleCliffordField(rayPos - vec3(0.0, NORMAL_EPSILON, 0.0)),
        sampleCliffordField(rayPos + vec3(0.0, 0.0, NORMAL_EPSILON)) - sampleCliffordField(rayPos - vec3(0.0, 0.0, NORMAL_EPSILON))
      ));
      
      vec3 lightDir = normalize(vec3(0.5, 0.7, 1.0));
      float diffuse = max(0.3, dot(normal, lightDir));
      
      color *= diffuse * (1.0 - depth * 0.3);
      
      // Normal-based color variation
      color.r += normal.x * 0.2; // Reduced slightly
      color.g += normal.y * 0.2;
      color.b += normal.z * 0.2;
      
      // Time-varying color shifts
      float colorTime = totalDist * 0.3 + (rayPos.x + rayPos.y + rayPos.z) * 0.2 + uTime;
      color.r += 0.2 * sin(colorTime * 2.0);
      color.g += 0.2 * cos(colorTime * 2.2);
      color.b += 0.2 * sin(colorTime * 1.8);
      
      // Braided lattice coloring (only in lattice mode)
      if (uRenderMode == 1) {
        // Determine which transformed emergent field instance we're closest to
        float phase1 = uTime * 0.2 + rayPos.x * 0.3;
        float phase2 = uTime * 0.2 * PHI + rayPos.y * 0.3;
        float phase3 = uTime * 0.2 * PHI_INV + rayPos.z * 0.3;
        
        float c1 = cos(phase1);
        float s1 = sin(phase1);
        vec3 pos1 = vec3(
          rayPos.x * c1 - rayPos.y * s1,
          rayPos.x * s1 + rayPos.y * c1 + 0.4,
          rayPos.z
        );
        
        float c2 = cos(phase2);
        float s2 = sin(phase2);
        vec3 pos2 = vec3(
          rayPos.x,
          rayPos.y * c2 - rayPos.z * s2,
          rayPos.y * s2 + rayPos.z * c2 + 0.4
        );
        
        float c3 = cos(phase3);
        float s3 = sin(phase3);
        vec3 pos3 = vec3(
          rayPos.x * c3 + rayPos.z * s3 + 0.4,
          rayPos.y,
          -rayPos.x * s3 + rayPos.z * c3
        );
        
        // Sample emergent field distances at each transformed position
        float scalar = comp0.r;
        vec3 vectors = comp0.gba;
        vec3 bivectors1 = comp1.rgb;
        float bivector_e12 = comp1.a;
        vec3 bivectors2 = comp2.rgb;
        float trivector_e013 = comp2.a;
        vec3 trivectors1 = comp3.rgb;
        float pseudoscalar = comp3.a;
        
        // Compute bootstrap_coherence (needed for sampleEmergentFieldAt)
        float bootstrap_coherence = 
          abs(comp0.r) + abs(comp0.g) + abs(comp0.b) + abs(comp0.a) +
          abs(comp1.r) + abs(comp1.g) + abs(comp1.b) + abs(comp1.a) +
          abs(comp2.r) + abs(comp2.g) + abs(comp2.b) + abs(comp2.a) +
          abs(comp3.r) + abs(comp3.g) + abs(comp3.b) + abs(comp3.a);
        
        float d1 = sampleEmergentFieldAt(pos1, scalar, vectors, bivectors1, bivector_e12,
                                         bivectors2, trivector_e013, trivectors1, pseudoscalar,
                                         bootstrap_coherence);
        float d2 = sampleEmergentFieldAt(pos2, scalar, vectors, bivectors1, bivector_e12,
                                         bivectors2, trivector_e013, trivectors1, pseudoscalar,
                                         bootstrap_coherence);
        float d3 = sampleEmergentFieldAt(pos3, scalar, vectors, bivectors1, bivector_e12,
                                         bivectors2, trivector_e013, trivectors1, pseudoscalar,
                                         bootstrap_coherence);
        
        // Color each braid strand distinctly (based on which instance dominates)
        vec3 braid1Color = vec3(1.0, 0.3, 0.5);  // Pink-red
        vec3 braid2Color = vec3(0.2, 0.8, 1.0);  // Cyan-blue
        vec3 braid3Color = vec3(0.8, 1.0, 0.3);  // Yellow-green
        
        // Weight colors by proximity to each emergent field instance
        float w1 = exp(-abs(d1) * 4.0);
        float w2 = exp(-abs(d2) * 4.0);
        float w3 = exp(-abs(d3) * 4.0);
        float totalW = w1 + w2 + w3 + 0.001;
        
        vec3 braidColor = (w1 * braid1Color + w2 * braid2Color + w3 * braid3Color) / totalW;
        
        // Standing-wave nodes create bright emissive strands along the braids
        float scale1 = (rayPos.x + rayPos.y + rayPos.z) * 0.1;
        float scale2 = (rayPos.x * rayPos.y + rayPos.y * rayPos.z + rayPos.z * rayPos.x) * 0.5;
        float scale3 = (rayPos.x * rayPos.y * rayPos.z) * 2.0;
        
        float p1 = (scale1 * uBraidScale) + scalar * PHI;
        float p2 = (scale2 * uBraidScale) + pseudoscalar * PHI_INV;
        float p3 = (scale3 * uBraidScale) + (bivector_e12 + trivector_e013) * 0.7;
        
        float n1 = braidNode(p1, uBraidSharpness);
        float n2 = braidNode(p2, uBraidSharpness * PHI_INV);
        float n3 = braidNode(p3, uBraidSharpness * 0.8);
        float nodeIntensity = (n1 + n2 + n3) / 3.0;
        
        // Phase-lock intensity
        float lock12 = phaseLock(phase1, phase2);
        float lock23 = phaseLock(phase2, phase3);
        float lock31 = phaseLock(phase3, phase1);
        float lock = lock12 * lock23 * lock31;
        
        // Emissive glow at nodes (standing-wave attractors)
        color = mix(color, braidColor, 0.5);
        color += braidColor * nodeIntensity * lock * uBraidStrength * 1.2;
      }

      // Distance-based pulse
      float pulse = 0.15 * sin(totalDist * 5.0);
      color += vec3(pulse);
      
      fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
      return;
    }
    
    // Step forward
    float stepDist = max(abs(dist) * 0.9, uMinDistance);
    stepDist = min(stepDist, 2.0);
    
    rayPos += rayDir * stepDist;
    totalDist += stepDist;
    
    if (totalDist > uMaxDistance) break;
  }
  
  // Background
  fragColor = vec4(0.0, 0.0, 0.1, 1.0);
}
`;

/**
 * Create and compile a shader program
 */
export function createShaderProgram(gl) {
  const vertexShader = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertexShader, vertexShaderSource);
  gl.compileShader(vertexShader);
  
  if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
    console.error('Vertex shader error:', gl.getShaderInfoLog(vertexShader));
    throw new Error('Vertex shader compilation failed');
  }
  
  const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragmentShader, fragmentShaderSource);
  gl.compileShader(fragmentShader);
  
  if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
    console.error('Fragment shader error:', gl.getShaderInfoLog(fragmentShader));
    throw new Error('Fragment shader compilation failed');
  }
  
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(program));
    throw new Error('Shader program linking failed');
  }
  
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  
  return program;
}

/**
 * Get uniform locations
 */
export function getUniformLocations(gl, program) {
  return {
    uCliffordField: gl.getUniformLocation(program, 'uCliffordField'),
    uTime: gl.getUniformLocation(program, 'uTime'),
    uMinDistance: gl.getUniformLocation(program, 'uMinDistance'),
    uMaxDistance: gl.getUniformLocation(program, 'uMaxDistance'),
    uHighlightCaustics: gl.getUniformLocation(program, 'uHighlightCaustics'),
    uRenderMode: gl.getUniformLocation(program, 'uRenderMode'),
    uBraidStrength: gl.getUniformLocation(program, 'uBraidStrength'),
    uBraidSharpness: gl.getUniformLocation(program, 'uBraidSharpness'),
    uBraidScale: gl.getUniformLocation(program, 'uBraidScale'),
    uInverseViewMatrix: gl.getUniformLocation(program, 'uInverseViewMatrix'),
    uInverseProjectionMatrix: gl.getUniformLocation(program, 'uInverseProjectionMatrix'),
    uCameraPosition: gl.getUniformLocation(program, 'uCameraPosition')
  };
}

/**
 * Get attribute locations
 */
export function getAttributeLocations(gl, program) {
  return {
    aPosition: gl.getAttribLocation(program, 'aPosition')
  };
}
