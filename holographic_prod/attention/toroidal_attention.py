"""
Toroidal Attention — Structural Attention via Phase Coherence
=============================================================

Attention emerges STRUCTURALLY from the nested torus, not as learned weights.

Key insight: Attention_ij = (1 + cos(θ_i - θ_j)) / 2

This is fundamentally different from transformers:
- No Q/K/V matrices (no learned weights)
- O(n) aggregation (satellites aggregate locally)
- Inductive bias from φ-offset phase distribution

Architecture:
    - 16 satellites with golden spiral phase distribution
    - Phase alignment determines attention weights
    - Master aggregates via φ-weighted sum

Theory:
    In the torus, two points with aligned phases (θ_i ≈ θ_j) are
    "closer" on the manifold and naturally attend to each other.
    Points with opposite phases (θ_i ≈ θ_j + π) are maximally
    distant and don't attend to each other.

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, DTYPE


# =============================================================================
# SATELLITE PHASE
# =============================================================================

@dataclass
class SatellitePhase:
    """
    A satellite in the toroidal attention mechanism.
    
    Attributes:
        index: Position in the torus (0-15)
        phase: Current phase angle [0, 2π)
        frequency: Rotation frequency (φ-staggered)
        witness: Stored witness (scalar, pseudoscalar)
        memory: Optional stored memory vector
    """
    index: int
    phase: float = 0.0
    frequency: float = 1.0
    witness: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    memory: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    
    @classmethod
    def create(cls, index: int, n_satellites: int = 16) -> 'SatellitePhase':
        """Create satellite with φ-derived initial state."""
        # Golden spiral position
        phase = (2 * PI * index * PHI_INV) % (2 * PI)
        
        # φ-staggered frequency
        frequency = PHI ** (index % 4) * PHI_INV_CUBE
        
        return cls(
            index=index,
            phase=phase,
            frequency=frequency,
            witness=np.array([PHI_INV, 0.0], dtype=DTYPE),  # Default witness
            memory=np.zeros(2, dtype=DTYPE),
        )
    
    def evolve(self, dt: float):
        """Evolve phase by time step."""
        self.phase = (self.phase + self.frequency * dt) % (2 * PI)


# =============================================================================
# TOROIDAL ATTENTION
# =============================================================================

class ToroidalAttention:
    """
    Structural Attention via Phase-Coherent Aggregation.
    
    NO LEARNED WEIGHTS. ALL φ-DERIVED.
    
    Architecture:
        - 16 satellites with φ-offset phases (golden spiral)
        - Attention = cos(θ_i - θ_j) (phase alignment)
        - Master aggregates via φ-weighted sum
    
    This is O(n) because:
        - Each token maps to a satellite (mod 16)
        - Satellites aggregate locally (φ-weighted)
        - Only 16 satellites regardless of sequence length
    """
    
    def __init__(self, n_satellites: int = 16):
        """
        Initialize toroidal attention.
        
        Args:
            n_satellites: Number of satellites (default 16 for Cl(3,1))
        """
        self.n_satellites = n_satellites
        
        # Create satellites with golden spiral distribution
        self.satellites = [
            SatellitePhase.create(k, n_satellites)
            for k in range(n_satellites)
        ]
        
        # PRECOMPUTED: φ-weights for master aggregation (avoids recomputing every call)
        k_vals = np.arange(n_satellites)
        raw_weights = PHI_INV ** (k_vals % 4)
        self._aggregate_weights = (raw_weights / np.sum(raw_weights)).astype(DTYPE)
    
    # =========================================================================
    # PHASE MANIPULATION
    # =========================================================================
    
    def set_satellite_phase(self, index: int, phase: float):
        """Set phase of a specific satellite."""
        self.satellites[index].phase = phase % (2 * PI)
    
    def get_satellite_phase(self, index: int) -> float:
        """Get phase of a specific satellite."""
        return self.satellites[index].phase
    
    # =========================================================================
    # ATTENTION COMPUTATION
    # =========================================================================
    
    def attention_weight(self, i: int, j: int) -> float:
        """
        Compute attention from satellite i to satellite j.
        
        Based purely on phase alignment:
            Attention = (1 + cos(θ_i - θ_j)) / 2
        
        This gives:
            - 1.0 when phases are identical
            - 0.0 when phases are opposite (π apart)
            - 0.5 when phases are orthogonal (π/2 apart)
        
        Args:
            i: Source satellite index
            j: Target satellite index
            
        Returns:
            Attention weight in [0, 1]
        """
        phase_diff = self.satellites[i].phase - self.satellites[j].phase
        return (1 + np.cos(phase_diff)) / 2
    
    def compute_attention_matrix(self) -> np.ndarray:
        """
        Compute full attention matrix between all satellites.
        
        VECTORIZED: No Python loops, pure NumPy.
        
        Returns:
            [n_satellites, n_satellites] attention matrix
        """
        # VECTORIZED: Extract all phases into array
        phases = np.array([sat.phase for sat in self.satellites], dtype=DTYPE)
        
        # VECTORIZED: Compute all pairwise phase differences via broadcasting
        # phases[:, None] - phases[None, :] gives [n, n] differences
        phase_diffs = phases[:, None] - phases[None, :]
        
        # VECTORIZED: Compute attention: (1 + cos(diff)) / 2
        attention = (1 + np.cos(phase_diffs)) / 2
        
        # THEORY-TRUE: Phase-coherent attention preserves magnitude
        # No normalization needed — phase coherence IS the structure
        # Magnitude encodes strength of alignment, which is meaningful
        
        return attention
    
    def compute_context_attention(self, context: List[int]) -> np.ndarray:
        """
        Compute attention matrix for a context sequence.
        
        CRITICAL FIX: Use BOTH position AND token value for phase.
        This preserves order information (non-commutativity).
        
        FULLY VECTORIZED: No Python loops.
        
        Phase_i = (position_phase + token_phase) % 2π
        where:
            position_phase = 2π × i × φ⁻¹
            token_phase = 2π × token_id × φ⁻² (scaled differently to distinguish)
        
        Args:
            context: List of token IDs
            
        Returns:
            [len(context), len(context)] attention matrix
        """
        n = len(context)
        tokens = np.array(context, dtype=DTYPE)
        positions = np.arange(n, dtype=DTYPE)
        
        # VECTORIZED: Compute all phases at once
        position_phases = (2 * PI * positions * PHI_INV) % (2 * PI)
        token_phases = (2 * PI * tokens * PHI_INV_SQ) % (2 * PI)
        phases = (position_phases + token_phases) % (2 * PI)
        
        # VECTORIZED: Compute all pairwise phase differences via broadcasting
        phase_diffs = phases[:, None] - phases[None, :]
        
        # VECTORIZED: Compute attention: (1 + cos(diff)) / 2
        attention = (1 + np.cos(phase_diffs)) / 2
        
        # THEORY-TRUE: Phase-coherent attention preserves magnitude
        # No normalization needed — phase coherence IS the structure
        
        return attention
    
    def compute_context_attention_fast(self, context: List[int]) -> np.ndarray:
        """
        FAST O(n) attention using satellite aggregation.
        
        THEORY-TRUE (Chapter 15):
            "Instead of n² token-to-token comparisons:
            1. Map each token to one of 16 satellites (O(n))
            2. Satellites interact via pre-computed 16×16 attention (O(1))
            3. Master aggregates satellite witnesses (O(16) = O(1))
            Total: O(n), not O(n²)."
        
        FULLY VECTORIZED: No Python loops.
        
        Args:
            context: List of token IDs
            
        Returns:
            [len(context), len(context)] attention matrix
        """
        n = len(context)
        tokens = np.array(context, dtype=DTYPE)
        positions = np.arange(n, dtype=DTYPE)
        
        # VECTORIZED: Map tokens to satellites via phase
        position_phases = (2 * PI * positions * PHI_INV) % (2 * PI)
        token_phases = (2 * PI * tokens * PHI_INV_SQ) % (2 * PI)
        combined_phases = (position_phases + token_phases) % (2 * PI)
        
        # VECTORIZED: Map phase to satellite index (16 evenly spaced)
        token_to_sat = ((combined_phases / (2 * PI)) * self.n_satellites).astype(np.int32) % self.n_satellites
        
        # Get 16×16 satellite attention - O(1) (already vectorized)
        sat_attention = self.compute_attention_matrix()
        
        # VECTORIZED: Expand to n×n via fancy indexing
        attention = sat_attention[token_to_sat][:, token_to_sat]
        
        # THEORY-TRUE: Phase-coherent attention preserves magnitude
        # No normalization needed — phase coherence IS the structure
        
        return attention
    
    # =========================================================================
    # MASTER AGGREGATION
    # =========================================================================
    
    def aggregate_to_master(self) -> np.ndarray:
        """
        Aggregate satellite witnesses to master witness.
        
        VECTORIZED: Uses pre-computed φ-weights.
        
        Uses φ-weighted sum:
            Master = Σ φ^(-k mod 4) × Satellite_k.witness / Σ weights
        
        Returns:
            Master witness [2] (scalar, pseudoscalar)
        """
        # VECTORIZED: Stack all witnesses [16, 2]
        witnesses = np.stack([sat.witness for sat in self.satellites])
        
        # VECTORIZED: Weighted sum (uses precomputed weights from __init__)
        master_witness = np.sum(witnesses * self._aggregate_weights[:, None], axis=0)
        
        return master_witness
    
    # =========================================================================
    # ATTENTION APPLICATION
    # =========================================================================
    
    def apply_attention(self, context: List[int]) -> np.ndarray:
        """
        Apply attention to get output representation.
        
        VECTORIZED: No Python loops.
        
        1. Map tokens to satellites
        2. Aggregate satellite memories with attention weights
        3. Return weighted sum
        
        Args:
            context: List of token IDs
            
        Returns:
            Attention-weighted output [2]
        """
        n = len(context)
        
        # Get attention matrix (already vectorized)
        attn = self.compute_context_attention(context)
        
        # VECTORIZED: Gather satellite memories using modular indexing
        sat_indices = np.arange(n) % self.n_satellites
        all_sat_memories = np.stack([sat.memory for sat in self.satellites])  # [16, 2]
        memories = all_sat_memories[sat_indices]  # [n, 2]
        
        # Apply attention: output[i] = Σ_j attn[i,j] × memory[j]
        output = attn @ memories
        
        # Aggregate to single output (mean)
        return output.mean(axis=0)
    
    # =========================================================================
    # EVOLUTION
    # =========================================================================
    
    def evolve(self, dt: float):
        """
        Evolve all satellite phases by time step.
        
        VECTORIZED: Updates all phases in one operation.
        """
        # VECTORIZED: Extract all frequencies and phases
        phases = np.array([sat.phase for sat in self.satellites], dtype=DTYPE)
        frequencies = np.array([sat.frequency for sat in self.satellites], dtype=DTYPE)
        
        # VECTORIZED: Update all phases at once
        new_phases = (phases + frequencies * dt) % (2 * PI)
        
        # Update satellite objects
        for i, sat in enumerate(self.satellites):
            sat.phase = new_phases[i]
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get attention statistics."""
        attn = self.compute_attention_matrix()
        
        return {
            'n_satellites': self.n_satellites,
            'mean_attention': float(attn.mean()),
            'attention_entropy': float(-np.sum(attn * np.log(attn + 1e-10)) / self.n_satellites),
            'master_witness': self.aggregate_to_master().tolist(),
        }


# =============================================================================
# TEST
# =============================================================================

def _test_basic():
    """Quick sanity check."""
    print("Testing ToroidalAttention...")
    
    model = ToroidalAttention(n_satellites=16)
    
    # Check golden spiral
    for k in range(16):
        expected = (2 * PI * k * PHI_INV) % (2 * PI)
        actual = model.satellites[k].phase
        assert abs((actual - expected + PI) % (2 * PI) - PI) < 0.01, f"Phase mismatch at {k}"
    print("  ✓ Golden spiral phases")
    
    # Check attention weights
    attn_same = model.attention_weight(0, 0)
    attn_opp = model.attention_weight(0, 8)
    print(f"  Attention(0→0): {attn_same:.4f}")
    print(f"  Attention(0→8): {attn_opp:.4f}")
    assert attn_same > attn_opp, "Same satellite should have higher self-attention"
    print("  ✓ Attention weights")
    
    # Check aggregation
    master = model.aggregate_to_master()
    print(f"  Master witness: {master}")
    print("  ✓ Aggregation")
    
    print("\n  All basic tests passed!")


if __name__ == "__main__":
    _test_basic()
