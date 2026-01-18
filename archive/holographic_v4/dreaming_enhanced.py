"""
Enhanced Dreaming — Topological Re-alignment
============================================

Implements dreaming as topological re-alignment in the nested fractal
torus architecture. The Master broadcasts its witness DOWN to satellites,
re-tuning them toward global coherence.

Non-REM: Harmonic Consolidation
    - Master trivectors project back to satellite bivectors
    - Dissonant satellites get accelerated Grace (φ^-4)
    - Noise pruning, prototype formation

REM: Stochastic Recombination (φ-Jitter)
    - Master jitters satellite phases by random * 2π * φ^-1
    - Search for new stable attractors
    - Creative synthesis, concept bridging

Paradox Resolution:
    - When contradictions create destructive interference
    - Shift one satellite by 2π * φ^-1 (golden angle)
    - Both states coexist in different phase lanes

Theory:
    Wake trigger when:
        (scalar² + pseudoscalar²) / total_energy > φ^-2
    
    This is the stability threshold from the spectral gap.

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR,
    CLIFFORD_DIM, GRADE_INDICES, GRACE_SCALES_FLAT,
)
from torus.phase_distribution import PhaseDistribution
from torus.interaction_tensor import InteractionTensor
from torus.chirality import ChiralityManager
from torus.grace_inverse import grace_inverse, GraceInverse


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def grace(M: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """
    Apply Grace operator with optional rate multiplier.
    
    Args:
        M: Multivector [..., 16]
        rate: Multiplier for scaling (e.g., φ^-4 for accelerated)
        
    Returns:
        Contracted multivector
    """
    scales = np.array(GRACE_SCALES_FLAT)
    if rate != 1.0:
        # Accelerate by applying extra power
        scales = scales ** (rate / PHI_INV_SQ)
    return M * scales


def compute_coherence(M1: np.ndarray, M2: np.ndarray) -> float:
    """
    Compute coherence between two multivectors.
    
    Uses normalized Frobenius inner product.
    
    Args:
        M1, M2: Multivectors to compare
        
    Returns:
        Coherence in [-1, 1]
    """
    norm1 = np.linalg.norm(M1)
    norm2 = np.linalg.norm(M2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return np.dot(M1.flatten(), M2.flatten()) / (norm1 * norm2)


def compute_witness_stability(M: np.ndarray) -> float:
    """
    Compute witness stability of multivector.
    
    Stability = (scalar² + pseudoscalar²) / total_energy
    
    Args:
        M: Multivector [..., 16]
        
    Returns:
        Stability in [0, 1]
    """
    scalar = M[..., GRADE_INDICES[0]]
    pseudo = M[..., GRADE_INDICES[4]]
    
    witness_energy = np.sum(scalar**2) + np.sum(pseudo**2)
    total_energy = np.sum(M**2)
    
    return witness_energy / (total_energy + 1e-10)


def extract_bivector(M: np.ndarray) -> np.ndarray:
    """Extract bivector (grade 2) components."""
    return M[..., GRADE_INDICES[2]]


def extract_trivector(M: np.ndarray) -> np.ndarray:
    """Extract trivector (grade 3) components."""
    return M[..., GRADE_INDICES[3]]


# =============================================================================
# NON-REM: HARMONIC CONSOLIDATION
# =============================================================================

@dataclass
class NonREMConsolidation:
    """
    Non-REM sleep phase: Master re-tunes satellites to global coherence.
    
    The Master broadcasts its stable witness DOWN through the inverse
    interaction tensor, creating "corrections" for each satellite.
    Satellites that are dissonant (low coherence) get accelerated Grace.
    
    Attributes:
        tensor: Interaction tensor for projection
        chirality: Chirality manager
        coherence_threshold: Minimum coherence for stability (φ^-2)
        accelerated_grace_rate: Grace rate for dissonant satellites (φ^-4)
    """
    tensor: InteractionTensor = field(default_factory=InteractionTensor)
    chirality: ChiralityManager = field(default_factory=ChiralityManager)
    coherence_threshold: float = PHI_INV_SQ
    accelerated_grace_rate: float = PHI_INV_FOUR
    
    def consolidate(
        self,
        master_state: np.ndarray,
        satellite_states: np.ndarray,
        max_iterations: int = 10
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform Non-REM consolidation.
        
        Args:
            master_state: Master torus state [16]
            satellite_states: Satellite states [n_satellites, 16]
            max_iterations: Maximum consolidation iterations
            
        Returns:
            (consolidated_satellites, stats_dict)
        """
        n_satellites = len(satellite_states)
        consolidated = satellite_states.copy()
        
        # Extract master trivector
        master_trivector = extract_trivector(master_state)
        
        stats = {
            'iterations': 0,
            'satellites_corrected': 0,
            'initial_coherences': [],
            'final_coherences': [],
        }
        
        for iteration in range(max_iterations):
            corrections_made = 0
            
            for k in range(n_satellites):
                # Project master trivector down to this satellite's bivector
                correction_bivector = self.tensor.project_down(master_trivector, k)
                
                # Build correction multivector
                correction = np.zeros(CLIFFORD_DIM)
                correction[GRADE_INDICES[2]] = correction_bivector
                
                # Apply chirality
                correction = self.chirality.apply(correction, k)
                
                # Measure coherence between satellite and correction
                sat_bivector = extract_bivector(consolidated[k])
                coherence = compute_coherence(sat_bivector, correction_bivector)
                
                if iteration == 0:
                    stats['initial_coherences'].append(coherence)
                
                # If dissonant, apply accelerated Grace
                if coherence < self.coherence_threshold:
                    # Apply accelerated Grace
                    consolidated[k] = grace(consolidated[k], rate=self.accelerated_grace_rate)
                    
                    # Blend toward correction
                    blend_factor = PHI_INV  # φ^-1 blending
                    consolidated[k] = (1 - blend_factor) * consolidated[k] + blend_factor * correction
                    corrections_made += 1
            
            stats['iterations'] = iteration + 1
            
            # Check for convergence
            if corrections_made == 0:
                break
        
        # Final coherence measurements
        for k in range(n_satellites):
            correction_bivector = self.tensor.project_down(master_trivector, k)
            sat_bivector = extract_bivector(consolidated[k])
            coherence = compute_coherence(sat_bivector, correction_bivector)
            stats['final_coherences'].append(coherence)
        
        stats['satellites_corrected'] = sum(
            1 for i, f in zip(stats['initial_coherences'], stats['final_coherences'])
            if f > i
        )
        
        return consolidated, stats


# =============================================================================
# REM: STOCHASTIC RECOMBINATION (φ-JITTER)
# =============================================================================

@dataclass
class REMRecombination:
    """
    REM sleep phase: Search for new stable attractors via φ-jitter.
    
    The Master jitters satellite phases by random multiples of the
    golden angle (2π * φ^-1). This creates novel combinations that
    are tested for stability.
    
    Attributes:
        phase_dist: Phase distribution manager
        stability_threshold: Minimum stability for new attractor (φ^-2)
        max_jitter_attempts: Maximum search iterations
    """
    phase_dist: PhaseDistribution = field(default_factory=PhaseDistribution)
    stability_threshold: float = PHI_INV_SQ
    max_jitter_attempts: int = 100
    
    def recombine(
        self,
        satellite_states: np.ndarray,
        seed: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Perform REM recombination to discover new attractors.
        
        Args:
            satellite_states: Current satellite states [n_satellites, 16]
            seed: Random seed for reproducibility
            
        Returns:
            (new_attractor or None, stats_dict)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_satellites = len(satellite_states)
        stats = {
            'attempts': 0,
            'stable_found': False,
            'best_stability': 0.0,
            'discoveries': [],
        }
        
        for attempt in range(self.max_jitter_attempts):
            # Jitter phases by random * golden angle
            jittered = satellite_states.copy()
            
            for k in range(n_satellites):
                # Random φ-derived phase shift
                jitter = np.random.random() * 2 * PI * PHI_INV
                
                # Apply as rotation in witness plane
                scalar = jittered[k, GRADE_INDICES[0][0]]
                pseudo = jittered[k, GRADE_INDICES[4][0]]
                
                # Rotate witness
                cos_j = np.cos(jitter)
                sin_j = np.sin(jitter)
                new_scalar = cos_j * scalar - sin_j * pseudo
                new_pseudo = sin_j * scalar + cos_j * pseudo
                
                jittered[k, GRADE_INDICES[0][0]] = new_scalar
                jittered[k, GRADE_INDICES[4][0]] = new_pseudo
            
            # Compose jittered satellites
            composed = self._compose_satellites(jittered)
            
            # Test stability
            stability = compute_witness_stability(composed)
            
            if stability > stats['best_stability']:
                stats['best_stability'] = stability
            
            if stability > self.stability_threshold:
                stats['stable_found'] = True
                stats['attempts'] = attempt + 1
                stats['discoveries'].append({
                    'attempt': attempt,
                    'stability': stability,
                })
                return composed, stats
        
        stats['attempts'] = self.max_jitter_attempts
        return None, stats
    
    def _compose_satellites(self, satellite_states: np.ndarray) -> np.ndarray:
        """
        Compose satellite states into single multivector.
        
        Uses φ-weighted sum respecting chirality.
        
        Args:
            satellite_states: States [n_satellites, 16]
            
        Returns:
            Composed multivector [16]
        """
        n = len(satellite_states)
        composed = np.zeros(CLIFFORD_DIM)
        
        # φ-weighted sum
        total_weight = 0.0
        for k in range(n):
            weight = PHI_INV ** (k % 4)  # Same frequency staggering as evolution
            composed += weight * satellite_states[k]
            total_weight += weight
        
        composed /= total_weight
        return composed


# =============================================================================
# PARADOX RESOLUTION
# =============================================================================

def detect_contradiction(
    sat_a: np.ndarray,
    sat_b: np.ndarray,
    threshold: float = -PHI_INV
) -> bool:
    """
    Detect if two satellites are contradictory.
    
    Contradiction is when coherence is negative (anti-correlated)
    in the witness components.
    
    Args:
        sat_a: First satellite [16]
        sat_b: Second satellite [16]
        threshold: Coherence below this is contradiction
        
    Returns:
        True if contradictory
    """
    # Extract witness components
    witness_a = np.concatenate([
        sat_a[GRADE_INDICES[0]],
        sat_a[GRADE_INDICES[4]]
    ])
    witness_b = np.concatenate([
        sat_b[GRADE_INDICES[0]],
        sat_b[GRADE_INDICES[4]]
    ])
    
    coherence = compute_coherence(witness_a, witness_b)
    return coherence < threshold


def resolve_paradox(
    sat_a: np.ndarray,
    sat_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve paradox by phase-shifting one satellite.
    
    Shifts sat_b by 2π * φ^-1 (golden angle) so both states
    can coexist in different "lanes" of the torus.
    
    Args:
        sat_a: First satellite (unchanged)
        sat_b: Second satellite (will be shifted)
        
    Returns:
        (sat_a_unchanged, sat_b_shifted)
    """
    sat_b_shifted = sat_b.copy()
    
    # Golden angle phase shift
    delta_psi = 2 * PI * PHI_INV
    
    # Apply rotation to witness components
    scalar = sat_b_shifted[GRADE_INDICES[0][0]]
    pseudo = sat_b_shifted[GRADE_INDICES[4][0]]
    
    cos_d = np.cos(delta_psi)
    sin_d = np.sin(delta_psi)
    
    sat_b_shifted[GRADE_INDICES[0][0]] = cos_d * scalar - sin_d * pseudo
    sat_b_shifted[GRADE_INDICES[4][0]] = sin_d * scalar + cos_d * pseudo
    
    return sat_a, sat_b_shifted


@dataclass
class ParadoxResolver:
    """
    Manages paradox detection and resolution across satellites.
    
    Scans for contradictory satellite pairs and applies φ-phase
    shifts to separate them into different phase lanes.
    
    Attributes:
        contradiction_threshold: Coherence below this is paradox
    """
    contradiction_threshold: float = -PHI_INV
    
    def scan_and_resolve(
        self,
        satellite_states: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Scan all satellite pairs and resolve paradoxes.
        
        Args:
            satellite_states: States [n_satellites, 16]
            
        Returns:
            (resolved_states, stats_dict)
        """
        n = len(satellite_states)
        resolved = satellite_states.copy()
        
        stats = {
            'paradoxes_found': 0,
            'paradoxes_resolved': 0,
            'pairs': [],
        }
        
        for i in range(n):
            for j in range(i + 1, n):
                if detect_contradiction(
                    resolved[i], resolved[j], 
                    self.contradiction_threshold
                ):
                    stats['paradoxes_found'] += 1
                    
                    # Resolve by shifting the second satellite
                    _, resolved[j] = resolve_paradox(resolved[i], resolved[j])
                    
                    # Verify resolution
                    if not detect_contradiction(
                        resolved[i], resolved[j],
                        self.contradiction_threshold
                    ):
                        stats['paradoxes_resolved'] += 1
                        stats['pairs'].append((i, j))
        
        return resolved, stats


# =============================================================================
# COMPLETE DREAMING SYSTEM
# =============================================================================

@dataclass
class EnhancedDreamingSystem:
    """
    Complete enhanced dreaming system for nested fractal torus.
    
    Integrates:
    - Non-REM consolidation (noise pruning, prototype formation)
    - REM recombination (creative synthesis, concept discovery)
    - Paradox resolution (phase-shift handling)
    - Wake trigger (stability threshold)
    
    Attributes:
        n_satellites: Number of satellites (default 16)
        nonrem: Non-REM consolidation system
        rem: REM recombination system
        resolver: Paradox resolver
        wake_threshold: Stability for wake trigger (φ^-2)
    """
    n_satellites: int = 16
    nonrem: NonREMConsolidation = field(default_factory=NonREMConsolidation)
    rem: REMRecombination = field(default_factory=REMRecombination)
    resolver: ParadoxResolver = field(default_factory=ParadoxResolver)
    wake_threshold: float = PHI_INV_SQ
    
    def sleep_cycle(
        self,
        master_state: np.ndarray,
        satellite_states: np.ndarray,
        max_nonrem_iterations: int = 10,
        max_rem_attempts: int = 50,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform complete sleep cycle.
        
        Sequence:
        1. Non-REM: Consolidate satellites toward master coherence
        2. Paradox Resolution: Separate contradictions
        3. REM: Search for new stable attractors
        4. Wake Trigger: Check if stable state achieved
        
        Args:
            master_state: Master torus state [16]
            satellite_states: Satellite states [n_satellites, 16]
            max_nonrem_iterations: Max Non-REM iterations
            max_rem_attempts: Max REM jitter attempts
            seed: Random seed
            
        Returns:
            Dict with:
                'final_master': Updated master state
                'final_satellites': Updated satellite states
                'discoveries': New attractors found
                'woke': Whether wake trigger fired
                'stats': Detailed statistics
        """
        result = {
            'final_master': master_state.copy(),
            'final_satellites': satellite_states.copy(),
            'discoveries': [],
            'woke': False,
            'stats': {
                'nonrem': {},
                'paradox': {},
                'rem': {},
            }
        }
        
        # Phase 1: Non-REM Consolidation
        consolidated, nonrem_stats = self.nonrem.consolidate(
            master_state,
            satellite_states,
            max_iterations=max_nonrem_iterations
        )
        result['final_satellites'] = consolidated
        result['stats']['nonrem'] = nonrem_stats
        
        # Phase 2: Paradox Resolution
        resolved, paradox_stats = self.resolver.scan_and_resolve(consolidated)
        result['final_satellites'] = resolved
        result['stats']['paradox'] = paradox_stats
        
        # Phase 3: REM Recombination
        new_attractor, rem_stats = self.rem.recombine(resolved, seed=seed)
        result['stats']['rem'] = rem_stats
        
        if new_attractor is not None:
            result['discoveries'].append(new_attractor)
            # Update master toward new attractor
            blend = PHI_INV  # φ^-1 blend
            result['final_master'] = (1 - blend) * master_state + blend * new_attractor
        
        # Phase 4: Wake Trigger
        final_stability = compute_witness_stability(result['final_master'])
        result['woke'] = final_stability > self.wake_threshold
        result['stats']['final_stability'] = final_stability
        
        return result


# =============================================================================
# TESTS
# =============================================================================

def _test_enhanced_dreaming():
    """Test enhanced dreaming system."""
    print("Testing enhanced dreaming...")
    
    np.random.seed(42)
    
    # Create test data
    master = np.random.randn(CLIFFORD_DIM)
    satellites = np.random.randn(16, CLIFFORD_DIM)
    
    # Test 1: Non-REM Consolidation
    print("  Testing Non-REM consolidation...")
    nonrem = NonREMConsolidation()
    consolidated, stats = nonrem.consolidate(master, satellites)
    
    assert consolidated.shape == satellites.shape, "Shape should be preserved"
    assert stats['iterations'] > 0, "Should run at least one iteration"
    print(f"    Iterations: {stats['iterations']}, Corrected: {stats['satellites_corrected']}")
    
    # Test 2: REM Recombination
    print("  Testing REM recombination...")
    rem = REMRecombination()
    new_attractor, stats = rem.recombine(satellites, seed=42)
    
    print(f"    Attempts: {stats['attempts']}, Found: {stats['stable_found']}")
    print(f"    Best stability: {stats['best_stability']:.4f}")
    
    # Test 3: Paradox Detection
    print("  Testing paradox detection...")
    # Create contradictory satellites
    sat_a = np.zeros(CLIFFORD_DIM)
    sat_a[GRADE_INDICES[0][0]] = 1.0  # Positive scalar
    
    sat_b = np.zeros(CLIFFORD_DIM)
    sat_b[GRADE_INDICES[0][0]] = -1.0  # Negative scalar (contradiction)
    
    is_paradox = detect_contradiction(sat_a, sat_b)
    assert is_paradox, "Should detect contradiction"
    
    # Test 4: Paradox Resolution
    print("  Testing paradox resolution...")
    _, sat_b_resolved = resolve_paradox(sat_a, sat_b)
    
    is_still_paradox = detect_contradiction(sat_a, sat_b_resolved, threshold=-0.9)
    print(f"    Resolved: {not is_still_paradox}")
    
    # Test 5: Complete Sleep Cycle
    print("  Testing complete sleep cycle...")
    dreaming = EnhancedDreamingSystem(n_satellites=16)
    result = dreaming.sleep_cycle(master, satellites, seed=42)
    
    assert 'final_master' in result, "Should have final master"
    assert 'final_satellites' in result, "Should have final satellites"
    assert 'woke' in result, "Should have wake status"
    
    print(f"    Woke: {result['woke']}")
    print(f"    Discoveries: {len(result['discoveries'])}")
    print(f"    Final stability: {result['stats']['final_stability']:.4f}")
    
    print("✓ Enhanced dreaming tests passed!")
    return True


if __name__ == "__main__":
    _test_enhanced_dreaming()
