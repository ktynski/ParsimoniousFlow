/**
 * camera.js - Orbital Camera for Torus Visualization
 * 
 * The camera orbits around the torus, allowing observation of:
 *   - The outer surface (general structure)
 *   - The throat (where caustics form)
 *   - The flow patterns
 */

import { PHI } from '../math/clifford.js';

export class OrbitCamera {
  constructor() {
    // Spherical coordinates
    this.theta = 0;           // Azimuthal angle (around z-axis)
    this.phi = Math.PI / 4;   // Polar angle (from z-axis)
    this.distance = 12;       // Distance from origin (increased for better view of braided structure)
    
    // Target point (center of torus)
    this.target = [0, 0, 0];
    
    // Camera parameters
    this.fov = 60;            // Field of view in degrees
    this.aspectRatio = 16/9;  // Will be updated on resize
    this.near = 0.1;
    this.far = 100;
    
    // Computed position
    this.position = [0, 0, 8];
    this.up = [0, 1, 0];
    
    // Interaction state
    this.isDragging = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;
    
    // Precomputed matrices
    this.viewMatrix = new Float32Array(16);
    this.projectionMatrix = new Float32Array(16);
    this.inverseViewMatrix = new Float32Array(16);
    this.inverseProjectionMatrix = new Float32Array(16);
    
    this.updatePosition();
  }
  
  /**
   * Update camera position from spherical coordinates
   */
  updatePosition() {
    // Convert spherical to Cartesian
    const sinPhi = Math.sin(this.phi);
    const cosPhi = Math.cos(this.phi);
    const sinTheta = Math.sin(this.theta);
    const cosTheta = Math.cos(this.theta);
    
    this.position = [
      this.target[0] + this.distance * sinPhi * cosTheta,
      this.target[1] + this.distance * cosPhi,
      this.target[2] + this.distance * sinPhi * sinTheta
    ];
    
    // Update matrices
    this.computeViewMatrix();
    this.computeProjectionMatrix();
  }
  
  /**
   * Compute view matrix (world → camera)
   */
  computeViewMatrix() {
    const [px, py, pz] = this.position;
    const [tx, ty, tz] = this.target;
    const [ux, uy, uz] = this.up;
    
    // Forward vector
    let fx = tx - px, fy = ty - py, fz = tz - pz;
    const flen = Math.sqrt(fx*fx + fy*fy + fz*fz);
    fx /= flen; fy /= flen; fz /= flen;
    
    // Right vector (forward × up)
    let rx = fy*uz - fz*uy;
    let ry = fz*ux - fx*uz;
    let rz = fx*uy - fy*ux;
    const rlen = Math.sqrt(rx*rx + ry*ry + rz*rz);
    rx /= rlen; ry /= rlen; rz /= rlen;
    
    // True up vector (right × forward)
    const upx = ry*fz - rz*fy;
    const upy = rz*fx - rx*fz;
    const upz = rx*fy - ry*fx;
    
    // View matrix (column-major)
    this.viewMatrix[0] = rx;
    this.viewMatrix[1] = upx;
    this.viewMatrix[2] = -fx;
    this.viewMatrix[3] = 0;
    
    this.viewMatrix[4] = ry;
    this.viewMatrix[5] = upy;
    this.viewMatrix[6] = -fy;
    this.viewMatrix[7] = 0;
    
    this.viewMatrix[8] = rz;
    this.viewMatrix[9] = upz;
    this.viewMatrix[10] = -fz;
    this.viewMatrix[11] = 0;
    
    this.viewMatrix[12] = -(rx*px + ry*py + rz*pz);
    this.viewMatrix[13] = -(upx*px + upy*py + upz*pz);
    this.viewMatrix[14] = fx*px + fy*py + fz*pz;
    this.viewMatrix[15] = 1;
    
    // Compute inverse
    this.invertMatrix4(this.viewMatrix, this.inverseViewMatrix);
  }
  
  /**
   * Compute projection matrix (perspective)
   */
  computeProjectionMatrix() {
    const fovRad = this.fov * Math.PI / 180;
    const f = 1 / Math.tan(fovRad / 2);
    const nf = 1 / (this.near - this.far);
    
    this.projectionMatrix[0] = f / this.aspectRatio;
    this.projectionMatrix[1] = 0;
    this.projectionMatrix[2] = 0;
    this.projectionMatrix[3] = 0;
    
    this.projectionMatrix[4] = 0;
    this.projectionMatrix[5] = f;
    this.projectionMatrix[6] = 0;
    this.projectionMatrix[7] = 0;
    
    this.projectionMatrix[8] = 0;
    this.projectionMatrix[9] = 0;
    this.projectionMatrix[10] = (this.far + this.near) * nf;
    this.projectionMatrix[11] = -1;
    
    this.projectionMatrix[12] = 0;
    this.projectionMatrix[13] = 0;
    this.projectionMatrix[14] = 2 * this.far * this.near * nf;
    this.projectionMatrix[15] = 0;
    
    // Compute inverse
    this.invertMatrix4(this.projectionMatrix, this.inverseProjectionMatrix);
  }
  
  /**
   * Invert a 4x4 matrix
   */
  invertMatrix4(m, out) {
    const m00 = m[0], m01 = m[1], m02 = m[2], m03 = m[3];
    const m10 = m[4], m11 = m[5], m12 = m[6], m13 = m[7];
    const m20 = m[8], m21 = m[9], m22 = m[10], m23 = m[11];
    const m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];
    
    const b00 = m00 * m11 - m01 * m10;
    const b01 = m00 * m12 - m02 * m10;
    const b02 = m00 * m13 - m03 * m10;
    const b03 = m01 * m12 - m02 * m11;
    const b04 = m01 * m13 - m03 * m11;
    const b05 = m02 * m13 - m03 * m12;
    const b06 = m20 * m31 - m21 * m30;
    const b07 = m20 * m32 - m22 * m30;
    const b08 = m20 * m33 - m23 * m30;
    const b09 = m21 * m32 - m22 * m31;
    const b10 = m21 * m33 - m23 * m31;
    const b11 = m22 * m33 - m23 * m32;
    
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    
    if (Math.abs(det) < 1e-10) {
      // Return identity if singular
      for (let i = 0; i < 16; i++) out[i] = i % 5 === 0 ? 1 : 0;
      return;
    }
    
    det = 1 / det;
    
    out[0] = (m11 * b11 - m12 * b10 + m13 * b09) * det;
    out[1] = (m02 * b10 - m01 * b11 - m03 * b09) * det;
    out[2] = (m31 * b05 - m32 * b04 + m33 * b03) * det;
    out[3] = (m22 * b04 - m21 * b05 - m23 * b03) * det;
    out[4] = (m12 * b08 - m10 * b11 - m13 * b07) * det;
    out[5] = (m00 * b11 - m02 * b08 + m03 * b07) * det;
    out[6] = (m32 * b02 - m30 * b05 - m33 * b01) * det;
    out[7] = (m20 * b05 - m22 * b02 + m23 * b01) * det;
    out[8] = (m10 * b10 - m11 * b08 + m13 * b06) * det;
    out[9] = (m01 * b08 - m00 * b10 - m03 * b06) * det;
    out[10] = (m30 * b04 - m31 * b02 + m33 * b00) * det;
    out[11] = (m21 * b02 - m20 * b04 - m23 * b00) * det;
    out[12] = (m11 * b07 - m10 * b09 - m12 * b06) * det;
    out[13] = (m00 * b09 - m01 * b07 + m02 * b06) * det;
    out[14] = (m31 * b01 - m30 * b03 - m32 * b00) * det;
    out[15] = (m20 * b03 - m21 * b01 + m22 * b00) * det;
  }
  
  /**
   * Rotate camera by delta angles
   */
  rotate(deltaTheta, deltaPhi) {
    this.theta += deltaTheta;
    this.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.phi + deltaPhi));
    this.updatePosition();
  }
  
  /**
   * Zoom camera (change distance)
   */
  zoom(delta) {
    this.distance = Math.max(3, Math.min(100, this.distance + delta)); // Increased max to 100 for braided lattice view
    this.updatePosition();
  }
  
  /**
   * Set aspect ratio (call on resize)
   */
  setAspectRatio(aspect) {
    this.aspectRatio = aspect;
    this.computeProjectionMatrix();
  }
  
  /**
   * Focus on the throat (inner edge of torus)
   */
  focusOnThroat() {
    const R = 2.0;
    const r = R / PHI;
    const throatRadius = R - r;
    
    // Move camera to look at the throat from above
    this.phi = Math.PI / 2.5;
    this.distance = 6;
    this.target = [throatRadius, 0, 0];
    this.updatePosition();
  }
  
  /**
   * Reset to default view
   */
  reset() {
    this.theta = 0;
    this.phi = Math.PI / 4;
    this.distance = 12; // Increased for better view of braided structure
    this.target = [0, 0, 0];
    this.updatePosition();
  }
  
  /**
   * Attach mouse/touch handlers to canvas
   */
  attachControls(canvas) {
    // Mouse drag for rotation
    canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
    });
    
    canvas.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      
      const dx = e.clientX - this.lastMouseX;
      const dy = e.clientY - this.lastMouseY;
      
      this.rotate(-dx * 0.01, dy * 0.01);
      
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
    });
    
    canvas.addEventListener('mouseup', () => {
      this.isDragging = false;
    });
    
    canvas.addEventListener('mouseleave', () => {
      this.isDragging = false;
    });
    
    // Scroll for zoom
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoom(e.deltaY * 0.01);
    }, { passive: false });
    
    // Touch support
    let lastTouchDist = 0;
    
    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this.isDragging = true;
        this.lastMouseX = e.touches[0].clientX;
        this.lastMouseY = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        lastTouchDist = Math.sqrt(dx*dx + dy*dy);
      }
    });
    
    canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      
      if (e.touches.length === 1 && this.isDragging) {
        const dx = e.touches[0].clientX - this.lastMouseX;
        const dy = e.touches[0].clientY - this.lastMouseY;
        
        this.rotate(-dx * 0.01, dy * 0.01);
        
        this.lastMouseX = e.touches[0].clientX;
        this.lastMouseY = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.sqrt(dx*dx + dy*dy);
        
        this.zoom((lastTouchDist - dist) * 0.02);
        lastTouchDist = dist;
      }
    }, { passive: false });
    
    canvas.addEventListener('touchend', () => {
      this.isDragging = false;
    });
    
    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      switch (e.key) {
        case 'r':
        case 'R':
          this.reset();
          console.log('📷 Camera reset');
          break;
        case 't':
        case 'T':
          this.focusOnThroat();
          console.log('📷 Focused on throat');
          break;
      }
    });
  }
}

