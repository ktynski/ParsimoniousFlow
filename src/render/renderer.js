/**
 * renderer.js - WebGL Renderer for EMERGENT Clifford Field
 * 
 * The geometry emerges from the field - no imposed torus.
 * Matches the original FIRM-Core renderer.js approach.
 */

import { createShaderProgram, getUniformLocations, getAttributeLocations } from './shaders.js';
import { OrbitCamera } from './camera.js';
import { Multivector, PHI, PHI_INV } from '../math/clifford.js';
import { generateCliffordField, DEFAULT_PARAMS, SPECTRAL_GAP } from '../math/resonance.js';
import { coherence } from '../math/grace.js';

export class CliffordTorusRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = null;
    this.program = null;
    this.uniforms = null;
    this.attributes = null;
    
    this.camera = new OrbitCamera();
    
    // Field state
    this.fieldTexture = null;
    this.currentField = new Multivector();
    
    // Animation state
    this.time = 0;
    this.isRunning = false;
    this.frameCount = 0;
    
    // Control parameters (match FIRM-Core defaults)
    this.params = {
      ...DEFAULT_PARAMS,
      minDistance: 0.001,
      maxDistance: 150.0, // Increased for braided lattice (multiple interlocking structures)
      highlightCaustics: true // Enable by default
    };
  }
  
  /**
   * Initialize WebGL context and resources
   */
  async initialize() {
    this.gl = this.canvas.getContext('webgl2', {
      alpha: false,
      depth: false,
      antialias: false,
      powerPreference: 'high-performance'
    });
    
    if (!this.gl) {
      throw new Error('WebGL2 not supported');
    }
    
    const gl = this.gl;
    console.log('🎮 WebGL2 context created');
    
    try {
      this.program = createShaderProgram(gl);
      console.log('✅ Shader program compiled');
    } catch (error) {
      console.error('❌ Shader compilation failed:', error);
      throw error;
    }
    
    this.uniforms = getUniformLocations(gl, this.program);
    this.attributes = getAttributeLocations(gl, this.program);
    
    this.createQuad();
    this.createFieldTexture();
    
    this.camera.attachControls(this.canvas);
    
    this.handleResize();
    window.addEventListener('resize', () => this.handleResize());
    
    console.log('✅ Renderer initialized');
    return true;
  }
  
  /**
   * Create fullscreen quad geometry
   */
  createQuad() {
    const gl = this.gl;
    
    const vertices = new Float32Array([
      -1, -1,
       1, -1,
       1,  1,
      -1, -1,
       1,  1,
      -1,  1
    ]);
    
    this.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  }
  
  /**
   * Create texture for Clifford field
   * 
   * Layout: 4 pixels × 1 row = 16 components (RGBA × 4)
   * Matches the shader sampling at texture coords 0.0625, 0.1875, 0.3125, 0.4375
   */
  createFieldTexture() {
    const gl = this.gl;
    
    this.fieldTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.fieldTexture);
    
    // Initialize with neutral field (0.5 = 0 in [-1,1] → [0,1] mapping)
    const initialData = new Uint8Array(16).fill(128);
    
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      4, 1, // 4 pixels × 1 row = 16 components
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      initialData
    );
    
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }
  
  /**
   * Update field texture from Multivector
   * 
   * Encodes [-1, 1] field values to [0, 255] bytes
   */
  updateFieldTexture(field) {
    const gl = this.gl;
    const bytes = field.toUint8Array();
    
    gl.bindTexture(gl.TEXTURE_2D, this.fieldTexture);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0, 0,
      4, 1,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      bytes
    );
  }
  
  /**
   * Handle canvas resize
   */
  handleResize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    this.camera.setAspectRatio(rect.width / rect.height);
  }
  
  /**
   * Render a single frame
   */
  renderFrame() {
    const gl = this.gl;
    
    gl.clearColor(0.0, 0.0, 0.1, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    
    gl.useProgram(this.program);
    
    // Bind quad
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.enableVertexAttribArray(this.attributes.aPosition);
    gl.vertexAttribPointer(this.attributes.aPosition, 2, gl.FLOAT, false, 0, 0);
    
    // Set uniforms
    gl.uniform1f(this.uniforms.uTime, this.time);
    gl.uniform1f(this.uniforms.uMinDistance, this.params.minDistance);
    gl.uniform1f(this.uniforms.uMaxDistance, this.params.maxDistance);
    gl.uniform1i(this.uniforms.uHighlightCaustics, this.params.highlightCaustics ? 1 : 0);
    gl.uniform1i(this.uniforms.uRenderMode, this.params.renderMode);
    gl.uniform1f(this.uniforms.uBraidStrength, this.params.braidStrength);
    gl.uniform1f(this.uniforms.uBraidSharpness, this.params.braidSharpness);
    gl.uniform1f(this.uniforms.uBraidScale, this.params.braidScale);
    
    // Camera uniforms
    gl.uniform3fv(this.uniforms.uCameraPosition, this.camera.position);
    gl.uniformMatrix4fv(this.uniforms.uInverseViewMatrix, false, this.camera.inverseViewMatrix);
    gl.uniformMatrix4fv(this.uniforms.uInverseProjectionMatrix, false, this.camera.inverseProjectionMatrix);
    
    // Bind field texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fieldTexture);
    gl.uniform1i(this.uniforms.uCliffordField, 0);
    
    // Draw
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    
    this.frameCount++;
  }
  
  /**
   * Evolution step - generate field and render
   */
  evolve(dt = 0.016) {
    this.time += dt;
    
    // Generate Clifford field using theory-compliant parameters
    const newField = generateCliffordField(0, 0, 0, this.time, this.params);
    
    this.currentField = newField;
    this.updateFieldTexture(newField);
    
    // Log occasionally
    if (this.frameCount % 300 === 0) {
      const coh = coherence(newField);
      const mags = newField.gradeMagnitudes();
      console.log(`🌀 Frame ${this.frameCount}: coherence=${coh.toFixed(3)}, ` +
                  `grades=[${mags.map(m => m.toFixed(2)).join(', ')}]`);
    }
    
    this.renderFrame();
  }
  
  /**
   * Update control parameters
   */
  updateParams(newParams) {
    this.params = { ...this.params, ...newParams };
  }
  
  /**
   * Start animation loop
   */
  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    
    const loop = () => {
      if (!this.isRunning) return;
      this.evolve(0.016);
      requestAnimationFrame(loop);
    };
    
    requestAnimationFrame(loop);
    console.log('🎬 Animation started');
  }
  
  /**
   * Stop animation loop
   */
  stop() {
    this.isRunning = false;
    console.log('⏹️ Animation stopped');
  }
  
  /**
   * Reset to initial state
   */
  reset() {
    this.time = 0;
    this.frameCount = 0;
    this.currentField = new Multivector();
    this.camera.reset();
    console.log('🔄 Renderer reset');
  }
  
  /**
   * Dispose of resources
   */
  dispose() {
    this.stop();
    
    const gl = this.gl;
    if (gl) {
      gl.deleteProgram(this.program);
      gl.deleteBuffer(this.quadBuffer);
      gl.deleteTexture(this.fieldTexture);
    }
  }
}

/**
 * Create and initialize renderer
 */
export async function createRenderer(canvas) {
  const renderer = new CliffordTorusRenderer(canvas);
  await renderer.initialize();
  return renderer;
}
