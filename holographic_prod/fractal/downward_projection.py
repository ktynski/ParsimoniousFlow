"""
Downward Projection — Generation Pipeline

Implements generation flow from Grand Master through levels to token emission.

Theory (Chapter 11, 14):
    1. Start with Grand Master witness (coherent core)
    2. Apply GraceInverse to inflate structure
    3. Unbind cascade through levels
    4. Phase-locked emission at correct intervals

Chirality Cascade (v5.27.0 — Quantum-inspired):
    5. Parent chirality constrains child output handedness
    6. Top-down propagation of processing mode (analytic vs holistic)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from holographic_prod.core.constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_EPSILON,
    MATRIX_DIM, DTYPE,
)
from holographic_prod.core.algebra import (
    build_clifford_basis,
    grace_inverse,
    clifford_inverse,
    geometric_product,
)
from holographic_prod.core.quotient import (
    extract_chirality,
    extract_chirality_strength,
    chirality_match_scores,
)


def phase_locked_emission(phase: float) -> bool:
    """
    Check if phase is in emission window.
    
    Theory (Chapter 11, 14):
        Emission window: [π·φ⁻¹, π·φ⁻¹ + φ⁻²]
        
        Tokens are released only when the toroidal phase enters the
        "emission window". This creates quasi-periodic output, paced
        by the golden rhythm.
    
    Args:
        phase: Current toroidal phase [0, 2π)
        
    Returns:
        True if phase is in emission window
    """
    window_start = PI * PHI_INV
    window_end = PI * PHI_INV + PHI_INV_SQ
    
    # Normalize phase to [0, 2π)
    phase = phase % (2 * PI)
    
    return window_start <= phase <= window_end


@dataclass
class DownwardProjection:
    """
    Downward projection pipeline for generation.
    
    Cascades from Grand Master down through levels to emit tokens.
    
    CHIRALITY CASCADE (v5.27.0):
        Parent chirality propagates to constrain child outputs.
        This is a quantum-inspired feature that provides top-down
        constraint on processing mode (analytic vs holistic).
    """
    
    def __init__(self, basis: np.ndarray = None, xp = np):
        """Initialize downward projection."""
        self.xp = xp
        if basis is not None:
            self.basis = basis
        else:
            self.basis = build_clifford_basis(xp)
        
        # v5.27.0: Track inherited chirality through cascade
        self._inherited_chirality: int = 0  # 0 = neutral, +1 = right, -1 = left
        self._inherited_strength: float = 0.0
    
    def project_level_down(
        self,
        higher_mv: np.ndarray,
        lower_level_memory: np.ndarray,
        propagate_chirality: bool = True,
    ) -> Tuple[np.ndarray, float, int, float]:
        """
        Project multivector from higher level down to lower level.
        
        Process:
            1. Apply GraceInverse to inflate coherent core
            2. Unbind with lower-level memory to get candidates
            3. (v5.27.0) Extract and propagate chirality for cascade
        
        Args:
            higher_mv: [4, 4] multivector from higher level
            lower_level_memory: [4, 4] memory from lower level
            propagate_chirality: Whether to extract chirality for cascade
            
        Returns:
            (projected_mv, confidence, chirality_sign, chirality_strength)
        """
        # Step 1: Inflate coherent core (GPU-aware)
        # Ensure inputs are on correct device
        if self.xp is not np:
            import cupy as cp
            if not isinstance(higher_mv, cp.ndarray):
                higher_mv = cp.asarray(higher_mv)
            if not isinstance(lower_level_memory, cp.ndarray):
                lower_level_memory = cp.asarray(lower_level_memory)
        
        inflated = grace_inverse(higher_mv, self.basis, self.xp)
        
        # Step 2: Unbind with lower-level memory (GPU-aware)
        memory_inv = clifford_inverse(lower_level_memory, self.xp)
        # geometric_product uses the array module of its inputs automatically
        projected = geometric_product(inflated, memory_inv)
        
        # Confidence based on similarity (using cosine for bounded [-1, 1] output)
        from holographic_prod.core.algebra import frobenius_cosine
        similarity = frobenius_cosine(projected, inflated, self.xp)
        # Cosine similarity is in [-1, 1], map to [0, 1] for confidence
        confidence = float((similarity + 1.0) / 2.0)
        
        # Step 3 (v5.27.0): Extract chirality for cascade
        chirality_sign = 0
        chirality_strength = 0.0
        if propagate_chirality:
            chirality_sign = extract_chirality(higher_mv, self.basis, self.xp)
            chirality_strength = extract_chirality_strength(higher_mv, self.basis, self.xp)
            
            # Update inherited chirality (φ-weighted combination with parent)
            if chirality_strength > PHI_INV_CUBE:  # Only if strong enough
                self._inherited_chirality = chirality_sign
                self._inherited_strength = chirality_strength
        
        return projected, confidence, chirality_sign, chirality_strength
    
    def set_parent_chirality(self, chirality: int, strength: float):
        """
        Set inherited chirality from parent level (for explicit cascade control).
        
        Args:
            chirality: +1 (right-handed) or -1 (left-handed)
            strength: Chirality strength [0, ∞)
        """
        self._inherited_chirality = chirality
        self._inherited_strength = strength
    
    def get_chirality_multipliers(
        self,
        candidate_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Get chirality match scores for candidate selection.
        
        Uses inherited chirality from parent level to constrain candidates.
        
        Args:
            candidate_embeddings: [N, 4, 4] candidate matrices
            
        Returns:
            [N] multipliers (1.0 for match, φ⁻³ for mismatch)
        """
        if self._inherited_strength < PHI_INV_CUBE:
            # Weak chirality: no constraint
            return self.xp.ones(candidate_embeddings.shape[0], dtype=DTYPE)
        
        return chirality_match_scores(
            self._inherited_chirality,
            self._inherited_strength,
            candidate_embeddings,
            self.basis,
            self.xp,
        )
    
    def generate_sequence(
        self,
        grand_master_mv: np.ndarray,
        memory_system,
        max_tokens: int = 20,
        use_chirality_cascade: bool = True,
        # v5.31.1: Anti-mode-collapse (was missing!)
        inhibition_window: int = 3,
        inhibition_factor: float = None,
        use_phi_kernel: bool = True,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        THEORY-TRUE: Generate via attractor flow from grand master state.
        
        BRAIN ANALOG:
            Grand master = global context state
            State flows through learned attractors (not discrete lookups)
            Phase-locked emission gates output timing
            
        CHIRALITY CASCADE (v5.27.0):
            Grand master chirality constrains candidate selection.
            This is a quantum-inspired top-down constraint.
            
        Args:
            grand_master_mv: [4, 4] grand master multivector (global context)
            memory_system: Memory system with tower
            max_tokens: Maximum tokens to generate
            use_chirality_cascade: Apply chirality constraints (default: True)
            
        Returns:
            (generated_tokens, stats)
        """
        from ..core.algebra import grace_operator
        from ..core.quotient import vorticity_weighted_scores
        
        xp = np  # Use numpy for now (could be upgraded to cupy)
        
        # v5.31.1: Anti-mode-collapse defaults
        if inhibition_factor is None:
            inhibition_factor = PHI_INV_SQ  # φ⁻² ≈ 0.382
        
        # Get tower embeddings and basis
        tower = memory_system.tower
        basis = tower.basis
        embeddings = tower.embeddings
        
        # (v5.27.0) Extract grand master chirality for cascade
        if use_chirality_cascade:
            gm_chirality = extract_chirality(grand_master_mv, basis, xp)
            gm_strength = extract_chirality_strength(grand_master_mv, basis, xp)
            self.set_parent_chirality(gm_chirality, gm_strength)
        
        # =====================================================================
        # THEORY-TRUE (v5.31.0): Use FULL VOCABULARY, not just learned targets
        # =====================================================================
        # "Candidate sets" are FORBIDDEN per THEORY_TRUE_PARADIGM.md
        # Grace ALWAYS converges - score ALL tokens by COHERENCE
        # =====================================================================
        vocab_size = len(embeddings)
        
        # (v5.27.0) Pre-compute chirality multipliers for full vocabulary
        chirality_multipliers = None
        if use_chirality_cascade:
            chirality_multipliers = self.get_chirality_multipliers(embeddings)
        
        # Get aggregated memory (φ-weighted)
        active_mask = tower._satellite_n_bindings > 0
        active_indices = xp.where(active_mask)[0][:1000]
        
        grand_memory = xp.zeros((4, 4), dtype=np.float32)
        for i, sat_i in enumerate(active_indices):
            weight = PHI_INV ** (i % 16)
            grand_memory += weight * tower._all_memories[int(sat_i)]
        
        generated = []
        stabilities = []
        current_phase = 0.0
        state = grand_master_mv.copy()  # Start from grand master
        recent_tokens = []  # v5.31.1: IoR tracking
        
        stats = {
            'tokens_generated': 0,
            'emissions_attempted': 0,
            'emissions_successful': 0,
            'chirality_used': use_chirality_cascade,
            'chirality_sign': int(self._inherited_chirality) if use_chirality_cascade else 0,
            'chirality_strength': float(self._inherited_strength) if use_chirality_cascade else 0.0,
        }
        
        # Precompute for coherence scoring
        norm_sq = xp.sum(basis * basis, axis=(1, 2))
        embed_T = xp.swapaxes(embeddings, -2, -1)
        
        for step in range(max_tokens):
            # Check if phase is in emission window
            if phase_locked_emission(current_phase):
                stats['emissions_attempted'] += 1
                
                # THEORY-TRUE: Unbind from current state
                state_inv = state.T
                retrieved = state_inv @ grand_memory
                
                # Apply Grace dynamics
                for _ in range(3):  # φ-derived iterations
                    retrieved = grace_operator(retrieved, basis, xp)
                
                # =========================================================
                # THEORY-TRUE (v5.31.0): Full vocabulary coherence scoring
                # =========================================================
                # Compute compositions: retrieved @ embed[t].T for all t
                compositions = xp.einsum('ij,vjk->vik', retrieved, embed_T)
                
                # Decompose into Clifford coefficients
                coeffs_all = xp.einsum('cij,vij->vc', basis, compositions) / norm_sq
                
                # Compute coherence: witness_energy / total_energy
                energies = xp.sum(coeffs_all ** 2, axis=1)
                witness_energies = coeffs_all[:, 0]**2 + coeffs_all[:, 15]**2
                coherences = witness_energies / xp.maximum(energies, PHI_EPSILON)
                
                # (v5.27.0) Apply chirality cascade constraint
                if use_chirality_cascade and chirality_multipliers is not None:
                    coherences = coherences * chirality_multipliers
                
                # =========================================================
                # v5.31.1: IoR + φ-kernel (NOT argmax!)
                # =========================================================
                scores = np.asarray(coherences).copy()
                
                # IoR: Penalize recently used tokens
                for recent_idx in recent_tokens[-inhibition_window:]:
                    if 0 <= recent_idx < len(scores):
                        scores[recent_idx] *= inhibition_factor
                
                # φ-kernel sampling: P(token) ∝ score^φ (POWER LAW, not softmax!)
                if use_phi_kernel:
                    scores_positive = np.maximum(scores, 1e-10)
                    logits = np.log(scores_positive) / PHI_INV  # = φ * log(scores)
                    logits = logits - np.max(logits)  # Numerical stability
                    probs = np.exp(logits)  # = scores^φ (power law!)
                    probs = probs / np.sum(probs)
                    token = int(np.random.choice(len(probs), p=probs))
                else:
                    token = int(np.argmax(scores))
                
                generated.append(token)
                recent_tokens.append(token)  # Track for IoR
                stats['emissions_successful'] += 1
                stats['tokens_generated'] += 1
                
                # Evolve state (continuous flow)
                token_emb = embeddings[token]
                state = retrieved @ token_emb
                state = state / (xp.linalg.norm(state) + 1e-10) * 2.0
            
            # Advance phase (toroidal dynamics)
            current_phase += PHI_INV_SQ
            current_phase = current_phase % (2 * PI)
        
        stats['attractor_flow'] = True
        stats['stabilities'] = stabilities
        
        return generated, stats
