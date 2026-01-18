/**
 * zeta_renderer.js - WebGL Renderer for Zeta Function Visualization
 * 
 * Creates a raymarched visualization of the Riemann Zeta function
 * showing zeros as caustic singularities in a toroidal geometry.
 */

import { ZETA_VERTEX_SHADER, ZETA_FRAGMENT_SHADER } from './zeta_shaders.js';

export class ZetaRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true
    });
    
    if (!this.gl) {
      throw new Error('WebGL2 not supported');
    }
    
    this.program = null;
    this.uniformLocations = {};
    this.time = 0;
    this.tOffset = 14.0; // Start near first zero
    
    // Camera
    this.cameraPos = [0, 0, 5];
    this.cameraTarget = [0, 0, 0];
    this.zoom = 1.5;
    
    // Rotation
    this.rotationX = 0;
    this.rotationY = 0;
    this.autoRotate = true;
    
    // Options
    this.showWinding = true;
    this.showSymmetry = false;
    
    this.init();
  }
  
  init() {
    const gl = this.gl;
    
    // Create shaders
    const vertexShader = this.createShader(gl.VERTEX_SHADER, ZETA_VERTEX_SHADER);
    const fragmentShader = this.createShader(gl.FRAGMENT_SHADER, ZETA_FRAGMENT_SHADER);
    
    // Create program
    this.program = gl.createProgram();
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);
    
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(this.program));
      return;
    }
    
    // Create fullscreen quad
    const positions = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1
    ]);
    
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    
    const positionLoc = gl.getAttribLocation(this.program, 'aPosition');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
    
    // Get uniform locations
    this.uniformLocations = {
      uTime: gl.getUniformLocation(this.program, 'uTime'),
      uResolution: gl.getUniformLocation(this.program, 'uResolution'),
      uCameraPos: gl.getUniformLocation(this.program, 'uCameraPos'),
      uCameraTarget: gl.getUniformLocation(this.program, 'uCameraTarget'),
      uZoom: gl.getUniformLocation(this.program, 'uZoom'),
      uTOffset: gl.getUniformLocation(this.program, 'uTOffset'),
      uShowWinding: gl.getUniformLocation(this.program, 'uShowWinding'),
      uShowSymmetry: gl.getUniformLocation(this.program, 'uShowSymmetry')
    };
    
    // Setup mouse interaction
    this.setupInteraction();
  }
  
  createShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(shader));
      return null;
    }
    
    return shader;
  }
  
  setupInteraction() {
    let isDragging = false;
    let lastX = 0, lastY = 0;
    
    this.canvas.addEventListener('mousedown', (e) => {
      isDragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      this.autoRotate = false;
    });
    
    this.canvas.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      
      this.rotationY += dx * 0.01;
      this.rotationX += dy * 0.01;
      
      lastX = e.clientX;
      lastY = e.clientY;
    });
    
    this.canvas.addEventListener('mouseup', () => {
      isDragging = false;
    });
    
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoom *= e.deltaY > 0 ? 1.1 : 0.9;
      this.zoom = Math.max(0.5, Math.min(5, this.zoom));
    });
  }
  
  updateCamera() {
    if (this.autoRotate) {
      this.rotationY += 0.005;
    }
    
    const dist = 5;
    this.cameraPos = [
      Math.sin(this.rotationY) * Math.cos(this.rotationX) * dist,
      Math.sin(this.rotationX) * dist,
      Math.cos(this.rotationY) * Math.cos(this.rotationX) * dist
    ];
  }
  
  render(deltaTime) {
    const gl = this.gl;
    
    this.time += deltaTime * 0.001;
    this.updateCamera();
    
    // Resize if needed
    if (this.canvas.width !== this.canvas.clientWidth || 
        this.canvas.height !== this.canvas.clientHeight) {
      this.canvas.width = this.canvas.clientWidth;
      this.canvas.height = this.canvas.clientHeight;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }
    
    gl.useProgram(this.program);
    
    // Set uniforms
    gl.uniform1f(this.uniformLocations.uTime, this.time);
    gl.uniform2f(this.uniformLocations.uResolution, this.canvas.width, this.canvas.height);
    gl.uniform3fv(this.uniformLocations.uCameraPos, this.cameraPos);
    gl.uniform3fv(this.uniformLocations.uCameraTarget, this.cameraTarget);
    gl.uniform1f(this.uniformLocations.uZoom, this.zoom);
    gl.uniform1f(this.uniformLocations.uTOffset, this.tOffset);
    gl.uniform1i(this.uniformLocations.uShowWinding, this.showWinding ? 1 : 0);
    gl.uniform1i(this.uniformLocations.uShowSymmetry, this.showSymmetry ? 1 : 0);
    
    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  
  setTOffset(t) {
    this.tOffset = t;
  }
  
  setShowWinding(show) {
    this.showWinding = show;
  }
  
  setShowSymmetry(show) {
    this.showSymmetry = show;
  }
  
  resetCamera() {
    this.rotationX = 0;
    this.rotationY = 0;
    this.zoom = 1.5;
    this.autoRotate = true;
  }
}

export default ZetaRenderer;

