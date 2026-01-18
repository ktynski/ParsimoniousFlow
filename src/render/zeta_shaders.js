/**
 * zeta_shaders.js - WebGL Shaders for Riemann Zeta Visualization
 * 
 * Raymarching visualization of the Zeta function in the critical strip,
 * showing zeros as caustic singularities and the critical line as
 * the emergent attractor.
 */

export const ZETA_VERTEX_SHADER = `#version 300 es
precision highp float;

in vec2 aPosition;
out vec2 vUv;

void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

export const ZETA_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 vUv;
out vec4 fragColor;

uniform float uTime;
uniform vec2 uResolution;
uniform vec3 uCameraPos;
uniform vec3 uCameraTarget;
uniform float uZoom;
uniform float uTOffset;      // Scroll through imaginary axis
uniform int uShowWinding;    // Show winding number contours
uniform int uShowSymmetry;   // Show Xi symmetry

// Constants
const float PI = 3.14159265359;
const float PHI = 1.618033988749;
const float PHI_INV = 0.618033988749;
const int MAX_STEPS = 100;
const float MIN_DIST = 0.001;
const float MAX_DIST = 50.0;

//=============================================================================
// COMPLEX ARITHMETIC
//=============================================================================

vec2 cmul(vec2 a, vec2 b) {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 cdiv(vec2 a, vec2 b) {
  float denom = dot(b, b);
  return vec2(dot(a, b), a.y * b.x - a.x * b.y) / denom;
}

vec2 cexp(vec2 z) {
  return exp(z.x) * vec2(cos(z.y), sin(z.y));
}

vec2 clog(vec2 z) {
  return vec2(log(length(z)), atan(z.y, z.x));
}

vec2 cpow(vec2 base, vec2 exp) {
  if (length(base) < 0.0001) return vec2(0.0);
  return cexp(cmul(exp, clog(base)));
}

vec2 csin(vec2 z) {
  return vec2(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}

//=============================================================================
// RIEMANN ZETA APPROXIMATION (Dirichlet eta for critical strip)
//=============================================================================

vec2 zeta(vec2 s) {
  // Use Dirichlet eta function: η(s) = Σ (-1)^(n-1) / n^s
  // Then ζ(s) = η(s) / (1 - 2^(1-s))
  
  vec2 eta = vec2(0.0);
  const int TERMS = 50;
  
  for (int n = 1; n <= TERMS; n++) {
    float sign = mod(float(n), 2.0) == 1.0 ? 1.0 : -1.0;
    float logN = log(float(n));
    vec2 exponent = vec2(-s.x * logN, -s.y * logN);
    vec2 term = cexp(exponent);
    eta += sign * term;
  }
  
  // Compute 1 - 2^(1-s)
  vec2 oneMinusS = vec2(1.0 - s.x, -s.y);
  vec2 twoTo1MinusS = cpow(vec2(2.0, 0.0), oneMinusS);
  vec2 divisor = vec2(1.0, 0.0) - twoTo1MinusS;
  
  if (length(divisor) < 0.001) {
    return vec2(100.0, 0.0); // Near pole
  }
  
  return cdiv(eta, divisor);
}

//=============================================================================
// SIGNED DISTANCE FUNCTION - ZETA MAGNITUDE FIELD
//=============================================================================

float zetaSDF(vec3 pos) {
  // Map 3D position to complex plane
  // x = σ (real part, 0 to 1)
  // y = t (imaginary part)
  // z = radial distance from critical strip
  
  float sigma = pos.x * 0.5 + 0.5;  // Map to [0, 1]
  float t = pos.y * 10.0 + uTOffset; // Scroll through t values
  
  // Compute |ζ(σ + it)|
  vec2 s = vec2(sigma, t);
  vec2 zetaVal = zeta(s);
  float mag = length(zetaVal);
  
  // Create surface where |ζ| = threshold
  // Zeros appear as "holes" where mag → 0
  float threshold = 0.5 + 0.3 * sin(uTime * 0.5);
  
  // Distance to the |ζ| = threshold surface
  float zetaDist = (mag - threshold) * 0.1;
  
  // Add toroidal structure centered on critical line (σ = 0.5)
  float distFromCritical = abs(sigma - 0.5);
  float torusR = 1.5;  // Major radius
  float torusr = 0.3;  // Minor radius
  
  // Torus centered on critical line
  float torusX = pos.x;  // Distance from σ = 0.5 in x
  float torusY = pos.z;  // Radial distance
  float torusDist = length(vec2(length(vec2(torusX, torusY)) - torusR, 0.0)) - torusr;
  
  // Blend zeta field with torus - zeros create caustics in the torus
  float causticStrength = 1.0 / (mag + 0.01);
  float blendedDist = torusDist - causticStrength * 0.05;
  
  // Add ripples from zeros
  float ripple = sin(mag * 20.0 + uTime * 2.0) * 0.02 * exp(-mag * 2.0);
  
  return blendedDist + ripple;
}

//=============================================================================
// RAYMARCHING
//=============================================================================

float raymarch(vec3 ro, vec3 rd) {
  float t = 0.0;
  
  for (int i = 0; i < MAX_STEPS; i++) {
    vec3 pos = ro + rd * t;
    float d = zetaSDF(pos);
    
    if (d < MIN_DIST) return t;
    if (t > MAX_DIST) break;
    
    t += d * 0.8;
  }
  
  return -1.0;
}

vec3 getNormal(vec3 pos) {
  const float eps = 0.001;
  return normalize(vec3(
    zetaSDF(pos + vec3(eps, 0, 0)) - zetaSDF(pos - vec3(eps, 0, 0)),
    zetaSDF(pos + vec3(0, eps, 0)) - zetaSDF(pos - vec3(0, eps, 0)),
    zetaSDF(pos + vec3(0, 0, eps)) - zetaSDF(pos - vec3(0, 0, eps))
  ));
}

//=============================================================================
// COLORING BASED ON ZETA PROPERTIES
//=============================================================================

vec3 getZetaColor(vec3 pos) {
  float sigma = pos.x * 0.5 + 0.5;
  float t = pos.y * 10.0 + uTOffset;
  
  vec2 s = vec2(sigma, t);
  vec2 zetaVal = zeta(s);
  float mag = length(zetaVal);
  float phase = atan(zetaVal.y, zetaVal.x);
  
  // Base color from phase (argument coloring)
  vec3 phaseColor = vec3(
    0.5 + 0.5 * cos(phase),
    0.5 + 0.5 * cos(phase + 2.094),
    0.5 + 0.5 * cos(phase + 4.189)
  );
  
  // Critical line highlight (σ = 0.5)
  float criticalGlow = exp(-pow(sigma - 0.5, 2.0) * 50.0);
  vec3 criticalColor = vec3(1.0, 0.84, 0.0); // Gold
  
  // Zero highlight (|ζ| → 0)
  float zeroGlow = exp(-mag * 5.0);
  vec3 zeroColor = vec3(1.0, 0.3, 0.1); // Bright orange/red
  
  // Symmetry visualization (|ζ(σ)| ≈ |ζ(1-σ)|)
  vec3 symmetryColor = vec3(0.0);
  if (uShowSymmetry == 1) {
    vec2 sReflected = vec2(1.0 - sigma, t);
    vec2 zetaReflected = zeta(sReflected);
    float magReflected = length(zetaReflected);
    float symmetryMatch = exp(-abs(mag - magReflected) * 10.0);
    symmetryColor = vec3(0.5, 0.0, 1.0) * symmetryMatch * 0.3;
  }
  
  // Combine colors
  vec3 color = phaseColor * 0.4;
  color = mix(color, criticalColor, criticalGlow * 0.6);
  color = mix(color, zeroColor, zeroGlow * 0.8);
  color += symmetryColor;
  
  return color;
}

//=============================================================================
// WINDING NUMBER VISUALIZATION
//=============================================================================

float getWindingContour(vec3 pos) {
  if (uShowWinding == 0) return 0.0;
  
  float sigma = pos.x * 0.5 + 0.5;
  float t = pos.y * 10.0 + uTOffset;
  
  // Draw contours around known zero locations
  float contour = 0.0;
  
  // First few zeros (approximate t values)
  float zeros[5] = float[5](14.13, 21.02, 25.01, 30.42, 32.94);
  
  for (int i = 0; i < 5; i++) {
    float zeroT = zeros[i];
    float dist = length(vec2(sigma - 0.5, t - zeroT));
    
    // Draw ring at radius ~0.3
    float ring = abs(dist - 0.3) < 0.05 ? 1.0 : 0.0;
    contour = max(contour, ring * exp(-abs(t - zeroT) * 0.1));
  }
  
  return contour;
}

//=============================================================================
// MAIN
//=============================================================================

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;
  
  // Camera setup
  vec3 ro = uCameraPos;
  vec3 target = uCameraTarget;
  
  vec3 forward = normalize(target - ro);
  vec3 right = normalize(cross(vec3(0, 1, 0), forward));
  vec3 up = cross(forward, right);
  
  vec3 rd = normalize(forward * uZoom + right * uv.x + up * uv.y);
  
  // Raymarch
  float t = raymarch(ro, rd);
  
  vec3 color = vec3(0.02, 0.02, 0.05); // Dark background
  
  if (t > 0.0) {
    vec3 pos = ro + rd * t;
    vec3 normal = getNormal(pos);
    
    // Lighting
    vec3 lightDir = normalize(vec3(1.0, 2.0, 3.0));
    float diff = max(dot(normal, lightDir), 0.0);
    float spec = pow(max(dot(reflect(-lightDir, normal), -rd), 0.0), 32.0);
    float fresnel = pow(1.0 - max(dot(normal, -rd), 0.0), 3.0);
    
    // Get zeta-based color
    vec3 surfaceColor = getZetaColor(pos);
    
    // Combine lighting
    color = surfaceColor * (0.2 + diff * 0.6);
    color += vec3(1.0, 0.9, 0.8) * spec * 0.3;
    color += surfaceColor * fresnel * 0.4;
    
    // Winding contours
    float winding = getWindingContour(pos);
    color = mix(color, vec3(0.0, 1.0, 0.8), winding * 0.7);
    
    // Fog based on distance
    float fog = 1.0 - exp(-t * 0.05);
    color = mix(color, vec3(0.02, 0.02, 0.05), fog);
  } else {
    // Background - show grid lines for critical strip
    float gridX = smoothstep(0.02, 0.0, abs(fract(uv.x * 2.0 + 0.5) - 0.5));
    float gridY = smoothstep(0.02, 0.0, abs(fract(uv.y * 5.0) - 0.5));
    color += vec3(0.1, 0.1, 0.15) * max(gridX, gridY) * 0.3;
    
    // Critical line marker
    float criticalLine = smoothstep(0.01, 0.0, abs(uv.x));
    color += vec3(1.0, 0.84, 0.0) * criticalLine * 0.2;
  }
  
  // Vignette
  float vignette = 1.0 - length(vUv - 0.5) * 0.8;
  color *= vignette;
  
  // Gamma correction
  color = pow(color, vec3(0.8));
  
  fragColor = vec4(color, 1.0);
}
`;

export default { ZETA_VERTEX_SHADER, ZETA_FRAGMENT_SHADER };

