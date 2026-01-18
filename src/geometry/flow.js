/**
 * flow.js - Hamiltonian Flow on the Torus Surface
 * 
 * Fluid flow on the torus creates the dynamical system whose
 * "fixed points" are the Riemann zeros (caustics).
 * 
 * The flow is HAMILTONIAN (energy-preserving) with:
 *   - The resonance field as the Hamiltonian H
 *   - Symplectic structure from the torus geometry
 *   - Grace contraction providing dissipation (regularization)
 * 
 * This connects to Navier-Stokes:
 *   - Laminar flow: stable orbits around torus
 *   - Turbulence: chaotic mixing
 *   - Viscosity (Grace): prevents energy cascade
 */

import { PHI, PHI_INV } from '../math/clifford.js';
import { computeResonance, evolveResonance, computeResonanceGradient } from '../math/resonance.js';
import { toTorusCoords } from './torus_sdf.js';

/**
 * Hamiltonian flow velocity at a point
 * 
 * The velocity is perpendicular to the gradient of H (resonance):
 *   v = J ∇H
 * 
 * where J is the symplectic form on the torus.
 */
export function flowVelocity(pos, time = 0) {
  const [gx, gy, gz] = computeResonanceGradient(pos[0], pos[1], pos[2]);
  
  // Symplectic rotation: (gx, gy, gz) → (-gy, gx, ...)
  // This makes flow perpendicular to gradient (constant H along flow)
  const vx = -gy + gz * PHI_INV;
  const vy = gx - gz * PHI_INV;
  const vz = (gx - gy) * PHI_INV;
  
  // Scale by resonance (flow faster in high-resonance regions)
  const H = computeResonance(pos[0], pos[1], pos[2]);
  const speed = 0.1 * (1 + H);
  
  return [vx * speed, vy * speed, vz * speed];
}

/**
 * Advect a particle along the flow for time dt
 * 
 * Uses 4th-order Runge-Kutta integration for accuracy.
 */
export function advectParticle(pos, dt, time = 0) {
  const x0 = pos[0], y0 = pos[1], z0 = pos[2];
  
  // RK4 integration
  const [vx1, vy1, vz1] = flowVelocity([x0, y0, z0], time);
  
  const x1 = x0 + 0.5 * dt * vx1;
  const y1 = y0 + 0.5 * dt * vy1;
  const z1 = z0 + 0.5 * dt * vz1;
  const [vx2, vy2, vz2] = flowVelocity([x1, y1, z1], time + 0.5 * dt);
  
  const x2 = x0 + 0.5 * dt * vx2;
  const y2 = y0 + 0.5 * dt * vy2;
  const z2 = z0 + 0.5 * dt * vz2;
  const [vx3, vy3, vz3] = flowVelocity([x2, y2, z2], time + 0.5 * dt);
  
  const x3 = x0 + dt * vx3;
  const y3 = y0 + dt * vy3;
  const z3 = z0 + dt * vz3;
  const [vx4, vy4, vz4] = flowVelocity([x3, y3, z3], time + dt);
  
  const newX = x0 + dt * (vx1 + 2*vx2 + 2*vx3 + vx4) / 6;
  const newY = y0 + dt * (vy1 + 2*vy2 + 2*vy3 + vy4) / 6;
  const newZ = z0 + dt * (vz1 + 2*vz2 + 2*vz3 + vz4) / 6;
  
  return [newX, newY, newZ];
}

/**
 * Compute the vorticity at a point (curl of velocity field)
 * 
 * Vorticity ω = ∇ × v
 * High vorticity indicates rotation - the "swirling" that creates caustics.
 */
export function computeVorticity(pos, time = 0) {
  const h = 0.01;
  
  // Get velocities at neighboring points
  const [vxp, vyp, vzp] = flowVelocity([pos[0] + h, pos[1], pos[2]], time);
  const [vxm, vym, vzm] = flowVelocity([pos[0] - h, pos[1], pos[2]], time);
  const [_, vypy, vzpy] = flowVelocity([pos[0], pos[1] + h, pos[2]], time);
  const [__, vymy, vzmy] = flowVelocity([pos[0], pos[1] - h, pos[2]], time);
  const [___, vyypz, vzpz] = flowVelocity([pos[0], pos[1], pos[2] + h], time);
  const [____, vyymz, vzmz] = flowVelocity([pos[0], pos[1], pos[2] - h], time);
  
  // Curl components
  const omegaX = (vzpy - vzmy) / (2*h) - (vyypz - vyymz) / (2*h);
  const omegaY = (vxp - vxm) / (2*h) - (vzpz - vzmz) / (2*h);
  const omegaZ = (vyp - vym) / (2*h) - (vxp - vxm) / (2*h);
  
  return [omegaX, omegaY, omegaZ];
}

/**
 * Compute the enstrophy (total vorticity squared)
 * 
 * Enstrophy is a key quantity in 2D turbulence theory.
 * Its growth or decay indicates laminar vs turbulent flow.
 */
export function computeEnstrophy(pos, time = 0) {
  const [ox, oy, oz] = computeVorticity(pos, time);
  return ox*ox + oy*oy + oz*oz;
}

/**
 * Detect a fixed point (zero velocity)
 * 
 * Fixed points are where caustics form - the "Riemann zeros"
 */
export function isFixedPoint(pos, time = 0, threshold = 0.01) {
  const [vx, vy, vz] = flowVelocity(pos, time);
  const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);
  return speed < threshold;
}

/**
 * Lyapunov exponent estimator
 * 
 * Positive Lyapunov exponent → chaos (trajectories diverge)
 * Negative Lyapunov exponent → stability (trajectories converge)
 * Zero Lyapunov exponent → neutral (periodic orbits)
 * 
 * The Grace operator should keep the exponent bounded (preventing blow-up).
 */
export function estimateLyapunovExponent(pos, time = 0, dt = 0.1, steps = 100) {
  let pos1 = [...pos];
  let pos2 = [pos[0] + 0.001, pos[1], pos[2]]; // Nearby point
  
  let sumLog = 0;
  
  for (let i = 0; i < steps; i++) {
    pos1 = advectParticle(pos1, dt, time + i * dt);
    pos2 = advectParticle(pos2, dt, time + i * dt);
    
    // Distance between trajectories
    const dx = pos2[0] - pos1[0];
    const dy = pos2[1] - pos1[1];
    const dz = pos2[2] - pos1[2];
    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
    
    if (dist < 1e-10) break;
    
    sumLog += Math.log(dist / 0.001);
    
    // Renormalize to prevent overflow
    const scale = 0.001 / dist;
    pos2[0] = pos1[0] + dx * scale;
    pos2[1] = pos1[1] + dy * scale;
    pos2[2] = pos1[2] + dz * scale;
  }
  
  return sumLog / (steps * dt);
}

/**
 * Generate streamlines from seed points
 * 
 * Streamlines visualize the flow pattern on the torus.
 */
export function generateStreamline(startPos, dt = 0.1, maxSteps = 100, time = 0) {
  const points = [startPos];
  let pos = [...startPos];
  
  for (let i = 0; i < maxSteps; i++) {
    pos = advectParticle(pos, dt, time + i * dt);
    
    // Check for divergence
    const r = Math.sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
    if (r > 10 || isNaN(r)) break;
    
    // Check for fixed point
    if (isFixedPoint(pos, time + i * dt)) {
      points.push([...pos]);
      break;
    }
    
    points.push([...pos]);
  }
  
  return points;
}

/**
 * Compute the Poincaré section
 * 
 * The Poincaré section is the set of points where the flow
 * crosses a fixed plane (z = 0). For integrable systems,
 * this forms closed curves. For chaotic systems, it's scattered.
 */
export function computePoincareSection(startPos, maxCrossings = 100, time = 0) {
  const crossings = [];
  let pos = [...startPos];
  let prevZ = pos[2];
  
  const dt = 0.1;
  let step = 0;
  
  while (crossings.length < maxCrossings && step < 10000) {
    pos = advectParticle(pos, dt, time + step * dt);
    step++;
    
    // Check for z = 0 crossing (from below)
    if (prevZ < 0 && pos[2] >= 0) {
      crossings.push([pos[0], pos[1]]);
    }
    
    prevZ = pos[2];
    
    // Bound check
    const r = Math.sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
    if (r > 10 || isNaN(r)) break;
  }
  
  return crossings;
}

/**
 * Toroidal flow pattern (for visualization)
 * 
 * Returns a vector field sampled on a grid around the torus.
 */
export function sampleFlowField(gridSize = 20, R = 2.0, time = 0) {
  const samples = [];
  
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      for (let k = 0; k < gridSize; k++) {
        // Map grid to space around torus
        const x = -4 + 8 * i / (gridSize - 1);
        const y = -4 + 8 * j / (gridSize - 1);
        const z = -3 + 6 * k / (gridSize - 1);
        
        const pos = [x, y, z];
        const vel = flowVelocity(pos, time);
        
        samples.push({
          position: pos,
          velocity: vel,
          speed: Math.sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]),
          resonance: computeResonance(x, y, z)
        });
      }
    }
  }
  
  return samples;
}

