/**
 * main.js - Application Entry Point
 * 
 * Initializes the Clifford Torus Flow visualization:
 *   - Creates canvas and renderer
 *   - Sets up UI controls
 *   - Starts animation loop
 */

import { createRenderer } from './render/renderer.js';
import { PHI } from './math/clifford.js';
import { SPECTRAL_GAP, LAMBDA_MAX } from './math/resonance.js';

class Application {
  constructor() {
    this.renderer = null;
    this.infoPanel = null;
    this.startTime = performance.now();
  }
  
  /**
   * Initialize the application
   */
  async initialize() {
    console.log('╔════════════════════════════════════════════╗');
    console.log('║    CLIFFORD TORUS FLOW VISUALIZATION       ║');
    console.log('║    Caustic Formation in the Throat         ║');
    console.log('╠════════════════════════════════════════════╣');
    console.log(`║  φ = ${PHI.toFixed(6)} (golden ratio)            ║`);
    console.log(`║  γ = ${SPECTRAL_GAP.toFixed(6)} (spectral gap)           ║`);
    console.log(`║  λ_max = ${LAMBDA_MAX.toFixed(6)}                        ║`);
    console.log('╚════════════════════════════════════════════╝');
    
    // Get canvas
    const canvas = document.getElementById('canvas');
    if (!canvas) {
      throw new Error('Canvas element not found');
    }
    
    // Create renderer
    try {
      this.renderer = await createRenderer(canvas);
      console.log('✅ Renderer ready');
    } catch (error) {
      console.error('❌ Renderer initialization failed:', error);
      this.showError(error.message);
      return;
    }
    
    // Setup info panel
    this.infoPanel = document.getElementById('info');
    this.setupInfoUpdates();
    
    // Setup keyboard controls
    this.setupControls();
    
    // Setup slider controls
    this.setupSliders();
    
    // Start animation
    this.renderer.start();
    
    console.log('✅ Application initialized');
  }
  
  /**
   * Setup periodic info panel updates
   */
  setupInfoUpdates() {
    if (!this.infoPanel) return;
    
    setInterval(() => {
      if (!this.renderer || !this.renderer.currentField) return;
      
      const field = this.renderer.currentField;
      const mags = field.gradeMagnitudes();
      const elapsed = ((performance.now() - this.startTime) / 1000).toFixed(1);
      
      this.infoPanel.innerHTML = `
        <div class="info-row">
          <span class="label">Time:</span>
          <span class="value">${elapsed}s</span>
        </div>
        <div class="info-row">
          <span class="label">Frame:</span>
          <span class="value">${this.renderer.frameCount}</span>
        </div>
        <div class="info-row separator">
          <span class="label">Grade Magnitudes:</span>
        </div>
        <div class="info-row grade grade-0">
          <span class="label">G0 (Scalar):</span>
          <span class="value">${mags[0].toFixed(3)}</span>
          <div class="bar" style="width: ${Math.min(100, mags[0] * 100)}%"></div>
        </div>
        <div class="info-row grade grade-1">
          <span class="label">G1 (Vector):</span>
          <span class="value">${mags[1].toFixed(3)}</span>
          <div class="bar" style="width: ${Math.min(100, mags[1] * 100)}%"></div>
        </div>
        <div class="info-row grade grade-2">
          <span class="label">G2 (Bivector):</span>
          <span class="value">${mags[2].toFixed(3)}</span>
          <div class="bar" style="width: ${Math.min(100, mags[2] * 100)}%"></div>
        </div>
        <div class="info-row grade grade-3">
          <span class="label">G3 (Trivector):</span>
          <span class="value">${mags[3].toFixed(3)}</span>
          <div class="bar" style="width: ${Math.min(100, mags[3] * 100)}%"></div>
        </div>
        <div class="info-row grade grade-4">
          <span class="label">G4 (Pseudo):</span>
          <span class="value">${mags[4].toFixed(3)}</span>
          <div class="bar" style="width: ${Math.min(100, mags[4] * 100)}%"></div>
        </div>
        <div class="info-row separator">
          <span class="label">Constants:</span>
        </div>
        <div class="info-row">
          <span class="label">φ:</span>
          <span class="value">${PHI.toFixed(6)}</span>
        </div>
        <div class="info-row">
          <span class="label">γ (gap):</span>
          <span class="value">${SPECTRAL_GAP.toFixed(6)}</span>
        </div>
      `;
    }, 100);
  }
  
  /**
   * Setup keyboard controls
   */
  setupControls() {
    window.addEventListener('keydown', (e) => {
      switch (e.key) {
        case ' ':
          e.preventDefault();
          if (this.renderer.isRunning) {
            this.renderer.stop();
          } else {
            this.renderer.start();
          }
          break;
        case 'm':
        case 'M': {
          // Render mode toggle (no UI dependency)
          this.toggleRenderMode();
          break;
        }
      }
    });

    // Optional UI button toggle (if present)
    const btn = document.getElementById('btn-toggle-mode');
    if (btn) {
      btn.addEventListener('click', () => this.toggleRenderMode());
    }
  }

  toggleRenderMode() {
    const nextMode = (this.renderer.params.renderMode + 1) % 2;
    this.renderer.updateParams({ renderMode: nextMode });
    this.updateModeIndicator(nextMode);
    console.log(`🎛️ Render mode: ${nextMode === 0 ? 'EMERGENT_SURFACE' : 'BRAIDED_LATTICE'}`);
  }

  updateModeIndicator(mode) {
    const el = document.getElementById('mode-indicator');
    if (!el) return;
    el.textContent = mode === 0 ? 'EMERGENT' : 'BRAIDED';
  }
  
  /**
   * Setup parameter sliders
   */
  setupSliders() {
    const bindSlider = (id, paramName, displayScale = 1) => {
      const slider = document.getElementById(id);
      const display = document.getElementById(id.replace('param-', 'val-'));
      
      if (!slider || !display) return;
      
      // Initial value from renderer defaults
      slider.value = this.renderer.params[paramName];
      display.textContent = (slider.value * displayScale).toFixed(2);
      
      slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        display.textContent = (val * displayScale).toFixed(2);
        
        // Update renderer
        this.renderer.updateParams({ [paramName]: val });
      });
    };
    
    bindSlider('param-beta', 'beta');
    bindSlider('param-nu', 'nu');
    bindSlider('param-gamma', 'spectralGap');
    bindSlider('param-lambda', 'lambdaMax');
    bindSlider('param-grace', 'graceScale');

    // Caustic highlight checkbox
    const causticCheck = document.getElementById('param-caustics');
    if (causticCheck) {
      causticCheck.checked = this.renderer.params.highlightCaustics;
      causticCheck.addEventListener('change', (e) => {
        this.renderer.updateParams({ highlightCaustics: e.target.checked });
      });
    }
    
    // Defect injection button
    const defectBtn = document.getElementById('btn-inject-defect');
    if (defectBtn) {
      defectBtn.addEventListener('mousedown', () => {
        // Inject a defect at a random position off-center
        const angle = Math.random() * Math.PI * 2;
        const r = 2.0; // Outside the throat
        const x = Math.cos(angle) * r;
        const y = Math.sin(angle) * r;
        this.renderer.updateParams({ defectPos: [x, y, 0] });
        defectBtn.style.background = 'rgba(255, 0, 0, 0.6)';
        defectBtn.innerText = '⚠️ DEFECT ACTIVE';
      });
      
      defectBtn.addEventListener('mouseup', () => {
        // Remove defect
        this.renderer.updateParams({ defectPos: null });
        defectBtn.style.background = '';
        defectBtn.innerText = 'Inject Off-Line Zero';
      });
      
      defectBtn.addEventListener('mouseleave', () => {
        if (this.renderer.params.defectPos) {
          this.renderer.updateParams({ defectPos: null });
          defectBtn.style.background = '';
          defectBtn.innerText = 'Inject Off-Line Zero';
        }
      });
    }

    // Initialize mode indicator if present
    this.updateModeIndicator(this.renderer.params.renderMode);
  }
  
  /**
   * Show error message
   */
  showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.innerHTML = `
      <h2>⚠️ Error</h2>
      <p>${message}</p>
      <p>Please ensure your browser supports WebGL2.</p>
    `;
    document.body.appendChild(errorDiv);
  }
}

// Start application when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
  const app = new Application();
  await app.initialize();
});
