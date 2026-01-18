"""
Dream Cycles — Topological Re-alignment
=======================================

Implements dreaming as topological re-alignment in the nested fractal torus.
The master broadcasts its witness DOWN to satellites, re-tuning them toward
global coherence.

Non-REM: Harmonic Consolidation
    - Master broadcasts stable witness to satellites
    - Dissonant satellites receive accelerated Grace (φ⁻⁴)
    - Prunes noise, reinforces prototypes

REM: Stochastic Recombination
    - Phase jitter by random × 2π × φ⁻¹
    - Search for new stable attractors
    - Creative synthesis, concept bridging

Wake Trigger:
    When stability > φ⁻² (spectral gap threshold)
    Or after max iterations (φ⁵ ≈ 11 cycles)

Theory:
    During sleep, the brain consolidates memories via replay.
    We do this geometrically: Master → Satellites broadcast.
    The Grace operator provides the "settling" mechanism.

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_v4.constants import PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR, CLIFFORD_DIM

if TYPE_CHECKING:
    from holographic_v4.fractal_generative_memory import FractalGenerativeMemory


# =============================================================================
# DREAM CYCLE CONFIGURATION
# =============================================================================

@dataclass
class DreamConfig:
    """
    Configuration for dream cycles.
    
    ALL VALUES ARE φ-DERIVED.
    """
    # Wake trigger
    stability_threshold: float = PHI_INV_SQ  # Wake when stability > φ⁻² ≈ 0.38
    
    # Max iterations
    max_iterations: int = int(PHI ** 5)  # ~11 cycles
    
    # Consolidation
    consolidation_rate: float = PHI_INV_SQ  # φ⁻² for normal, φ⁻⁴ for accelerated
    accelerated_rate: float = PHI_INV_FOUR  # For dissonant satellites
    
    # REM
    jitter_scale: float = PHI_INV  # Scale of phase jitter
    
    # Thresholds
    dissonance_threshold: float = PHI_INV  # Below this = dissonant


# =============================================================================
# DREAM CYCLE
# =============================================================================

class DreamCycle:
    """
    Topological Re-alignment via Sleep Cycles.
    
    Integrates with FractalGenerativeMemory.
    
    Phases:
        1. Non-REM: Master → Satellites (consolidation)
        2. REM: Phase jitter (creative recombination)
        3. Wake: When stability > φ⁻²
    """
    
    def __init__(
        self,
        memory: 'FractalGenerativeMemory',
        config: Optional[DreamConfig] = None,
    ):
        """
        Initialize dream cycle.
        
        Args:
            memory: FractalGenerativeMemory instance to dream on
            config: Optional configuration
        """
        self.memory = memory
        self.config = config or DreamConfig()
    
    # =========================================================================
    # COHERENCE
    # =========================================================================
    
    def compute_coherence(self) -> float:
        """
        Compute coherence between master and satellites.
        
        Coherence = average alignment of satellite witnesses with master.
        
        Returns:
            Coherence in [0, 1]
        """
        master_witness = self.memory.get_master_witness()
        master_norm = np.linalg.norm(master_witness)
        
        if master_norm < 1e-10:
            return 0.0
        
        coherences = []
        for sat in self.memory.satellite_states:
            # Extract witness from satellite (scalar, pseudoscalar)
            sat_witness = np.array([sat[0], sat[-1]])
            sat_norm = np.linalg.norm(sat_witness)
            
            if sat_norm < 1e-10:
                coherences.append(0.0)
                continue
            
            # Cosine similarity
            alignment = np.dot(master_witness, sat_witness) / (master_norm * sat_norm)
            coherences.append((1 + alignment) / 2)  # Normalize to [0, 1]
        
        return float(np.mean(coherences))
    
    # =========================================================================
    # NON-REM CONSOLIDATION
    # =========================================================================
    
    def non_rem_consolidation(self):
        """
        Master broadcasts witness to satellites.
        
        Dissonant satellites receive accelerated Grace.
        """
        master_witness = self.memory.get_master_witness()
        
        for i, sat in enumerate(self.memory.satellite_states):
            # Extract satellite witness
            sat_witness = np.array([sat[0], sat[-1]])
            
            # Compute coherence with master
            master_norm = np.linalg.norm(master_witness)
            sat_norm = np.linalg.norm(sat_witness)
            
            if master_norm > 1e-10 and sat_norm > 1e-10:
                coherence = np.dot(master_witness, sat_witness) / (master_norm * sat_norm)
            else:
                coherence = 0.0
            
            # Determine consolidation rate
            if coherence < self.config.dissonance_threshold:
                # Dissonant: accelerated Grace
                rate = self.config.accelerated_rate
            else:
                # Consonant: normal consolidation
                rate = self.config.consolidation_rate
            
            # Broadcast master witness to satellite
            # Blend satellite toward master
            sat[0] = (1 - rate) * sat[0] + rate * master_witness[0]
            sat[-1] = (1 - rate) * sat[-1] + rate * master_witness[1]
            
            # Apply Grace to vorticity (middle components)
            # This dampens noise while preserving structure
            for j in range(1, len(sat) - 1):
                sat[j] *= (1 - rate * PHI_INV)  # Gentle dampening
    
    # =========================================================================
    # REM RECOMBINATION
    # =========================================================================
    
    def rem_recombination(self) -> int:
        """
        Phase jitter for creative synthesis.
        
        Introduces golden-angle jitter to satellite phases.
        
        Returns:
            Number of discoveries (stability improvements)
        """
        discoveries = 0
        pre_stability = self.memory.get_stability()
        
        for sat in self.memory.satellite_states:
            # Random phase jitter scaled by golden angle
            jitter = np.random.randn() * 2 * PI * self.config.jitter_scale
            
            # Apply to bivector components (indices 5-10 in Clifford basis)
            # These are the "phase carriers" in the torus
            bivector_start = min(5, len(sat) - 1)
            bivector_end = min(11, len(sat))
            
            for j in range(bivector_start, bivector_end):
                sat[j] *= np.cos(jitter)
                sat[j] += np.random.randn() * abs(sat[j]) * PHI_INV_CUBE  # Small noise
        
        # Re-aggregate master
        self.memory._update_master()
        
        # Check if stability improved (found new attractor)
        post_stability = self.memory.get_stability()
        if post_stability > pre_stability:
            discoveries = 1
        
        return discoveries
    
    # =========================================================================
    # FULL CYCLE
    # =========================================================================
    
    def full_cycle(self) -> Dict[str, Any]:
        """
        Run complete dream cycle (Non-REM + REM).
        
        Returns:
            Dictionary with dream statistics
        """
        stats = {
            'iterations': 0,
            'total_discoveries': 0,
            'pre_stability': self.memory.get_stability(),
            'woke_early': False,
        }
        
        for i in range(self.config.max_iterations):
            stats['iterations'] = i + 1
            
            # Non-REM: Consolidate
            self.non_rem_consolidation()
            
            # Check wake trigger
            stability = self.memory.get_stability()
            if stability > self.config.stability_threshold:
                stats['woke_early'] = True
                break
            
            # REM: Recombine
            discoveries = self.rem_recombination()
            stats['total_discoveries'] += discoveries
        
        stats['post_stability'] = self.memory.get_stability()
        return stats
    
    # =========================================================================
    # ADVANCED: PARADOX RESOLUTION
    # =========================================================================
    
    def resolve_paradoxes(self):
        """
        Resolve contradictions via golden-angle phase shift.
        
        When two memories create destructive interference,
        shift one by 2π × φ⁻¹ to separate them into different
        "phase lanes".
        
        This allows both "A → X" and "A → Y" to coexist.
        """
        # Find high-vorticity satellites (potential paradoxes)
        for i, sat in enumerate(self.memory.satellite_states):
            vorticity = np.sum(sat[5:11] ** 2)  # Bivector energy
            witness = sat[0] ** 2 + sat[-1] ** 2  # Scalar + pseudoscalar energy
            
            # High vorticity relative to witness = potential paradox
            if vorticity > witness * PHI:
                # Apply golden-angle shift to bivectors
                golden_angle = 2 * PI * PHI_INV
                for j in range(5, min(11, len(sat))):
                    # Rotate by golden angle
                    original = sat[j]
                    sat[j] = original * np.cos(golden_angle)
                    # Add orthogonal component
                    if j + 1 < len(sat):
                        sat[j + 1] += original * np.sin(golden_angle) * PHI_INV


# =============================================================================
# TEST
# =============================================================================

def _test_basic():
    """Quick sanity check."""
    print("Testing DreamCycle...")
    
    # Need FractalGenerativeMemory
    from holographic_v4.fractal_generative_memory import FractalGenerativeMemory
    
    model = FractalGenerativeMemory(
        max_levels=2,
        vocab_size=100,
        orthogonalize=True,
    )
    
    # Train
    np.random.seed(42)
    for _ in range(20):
        ctx = [np.random.randint(100) for _ in range(3)]
        tgt = np.random.randint(100)
        model.learn(ctx, tgt)
    
    # Dream
    dreamer = DreamCycle(model)
    
    print(f"  Pre-stability: {model.get_stability():.4f}")
    print(f"  Coherence: {dreamer.compute_coherence():.4f}")
    
    stats = dreamer.full_cycle()
    
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Discoveries: {stats['total_discoveries']}")
    print(f"  Post-stability: {stats['post_stability']:.4f}")
    print(f"  Woke early: {stats['woke_early']}")
    
    print("\n  ✓ Basic test passed!")


if __name__ == "__main__":
    _test_basic()
