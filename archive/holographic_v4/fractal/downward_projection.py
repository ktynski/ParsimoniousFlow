"""
Downward Projection — Generation Flow from Grand Master to Tokens
================================================================

Implements the "explanation" flow where abstract high-level intent
is projected down through the fractal hierarchy to emit specific tokens.

The flow is:
    Grand Master → Master → Satellite → Token

At each step:
1. Unfold: Trivector → Bivector (via inverse interaction tensor)
2. Inflate: GraceInverse to restore vorticity
3. Unbind: Find matching token in vocabulary

Phase-Locked Emission:
    Tokens are released only in the "emission window" defined by:
        ψ ∈ [π·φ^-1, π·φ^-1 + φ^-2]
    
    This creates quasi-periodic rhythm in generation.

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
    CLIFFORD_DIM, GRADE_INDICES,
)
from torus.interaction_tensor import InteractionTensor
from torus.grace_inverse import grace_inverse, semantic_snap, GraceInverse
from torus.chirality import ChiralityManager


# =============================================================================
# EMISSION WINDOW
# =============================================================================

# Phase-locked emission window
EMISSION_WINDOW_START = PI * PHI_INV
EMISSION_WINDOW_END = PI * PHI_INV + PHI_INV_SQ


def in_emission_window(phase: float) -> bool:
    """
    Check if phase is in the emission window.
    
    Args:
        phase: Current phase in [0, 2π)
        
    Returns:
        True if ready to emit
    """
    return EMISSION_WINDOW_START <= phase < EMISSION_WINDOW_END


def time_to_emission(phase: float) -> float:
    """
    Compute time until next emission window.
    
    Args:
        phase: Current phase
        
    Returns:
        Phase delta until emission (0 if already in window)
    """
    if in_emission_window(phase):
        return 0.0
    
    if phase < EMISSION_WINDOW_START:
        return EMISSION_WINDOW_START - phase
    else:
        return (2 * PI - phase) + EMISSION_WINDOW_START


# =============================================================================
# DOWNWARD PROJECTION FUNCTIONS
# =============================================================================

def project_level_down(
    higher_state: np.ndarray,
    tensor: InteractionTensor,
    target_satellite: int = 0
) -> np.ndarray:
    """
    Project from higher level down to satellite.
    
    Extracts trivector from higher state and projects to bivector
    for the target satellite.
    
    Args:
        higher_state: State from higher level [16]
        tensor: Interaction tensor
        target_satellite: Which satellite to project to
        
    Returns:
        Projected state for satellite [16]
    """
    # Extract trivector
    trivector = higher_state[GRADE_INDICES[3]]
    
    # Project down to bivector
    bivector = tensor.project_down(trivector, target_satellite)
    
    # Build satellite state
    satellite_state = np.zeros(CLIFFORD_DIM)
    
    # Copy witness from higher level
    satellite_state[GRADE_INDICES[0]] = higher_state[GRADE_INDICES[0]]
    satellite_state[GRADE_INDICES[4]] = higher_state[GRADE_INDICES[4]]
    
    # Set projected bivector
    satellite_state[GRADE_INDICES[2]] = bivector
    
    # Copy vectors (scaled down)
    satellite_state[GRADE_INDICES[1]] = higher_state[GRADE_INDICES[1]] * PHI_INV
    
    return satellite_state


def inflate_for_emission(state: np.ndarray) -> np.ndarray:
    """
    Inflate state for token emission.
    
    Uses semantic snap to restore vorticity for unbinding.
    
    Args:
        state: Contracted state [16]
        
    Returns:
        Inflated state ready for unbinding [16]
    """
    return semantic_snap(state)


def unbind_token(
    query_state: np.ndarray,
    embeddings: np.ndarray
) -> Tuple[int, float]:
    """
    Find token whose embedding best matches query state.
    
    Uses witness-weighted similarity for robust matching.
    
    Args:
        query_state: Query multivector [16]
        embeddings: Token embeddings [vocab_size, 16]
        
    Returns:
        (token_id, confidence)
    """
    vocab_size = len(embeddings)
    
    # Normalize query
    query_norm = np.linalg.norm(query_state)
    if query_norm < 1e-10:
        return 0, 0.0
    query_normalized = query_state / query_norm
    
    # Compute similarities
    # Weight by φ-derived grade importance
    weights = np.ones(CLIFFORD_DIM)
    weights[GRADE_INDICES[0]] = 1.0  # Scalar most important
    weights[GRADE_INDICES[4]] = 1.0  # Pseudoscalar most important
    weights[GRADE_INDICES[2]] = PHI_INV_SQ  # Vorticity important for syntax
    weights[GRADE_INDICES[1]] = PHI_INV_CUBE
    weights[GRADE_INDICES[3]] = PHI_INV_CUBE
    
    # Weighted similarity
    query_weighted = query_normalized * weights
    emb_norms = np.linalg.norm(embeddings * weights, axis=1, keepdims=True)
    emb_weighted = (embeddings * weights) / (emb_norms + 1e-10)
    
    similarities = emb_weighted @ query_weighted
    
    best_idx = np.argmax(similarities)
    confidence = similarities[best_idx]
    
    return int(best_idx), float(confidence)


# =============================================================================
# DOWNWARD PROJECTION CLASS
# =============================================================================

@dataclass
class DownwardProjection:
    """
    Complete downward projection system for generation.
    
    Manages the flow from Grand Master intent to token emission.
    
    Attributes:
        tensor: Interaction tensor
        chirality: Chirality manager
        grace_inverse: Inflation operator
        embeddings: Token embeddings [vocab_size, 16]
        current_phase: Current emission phase
    """
    tensor: InteractionTensor = None
    chirality: ChiralityManager = None
    grace_inverse: GraceInverse = None
    embeddings: np.ndarray = None
    current_phase: float = 0.0
    
    def __post_init__(self):
        if self.tensor is None:
            self.tensor = InteractionTensor()
        if self.chirality is None:
            self.chirality = ChiralityManager()
        if self.grace_inverse is None:
            self.grace_inverse = GraceInverse()
    
    def set_embeddings(self, embeddings: np.ndarray):
        """Set token embeddings."""
        self.embeddings = embeddings
    
    def project_and_emit(
        self,
        grand_master: np.ndarray,
        context: Optional[np.ndarray] = None,
        wait_for_window: bool = True
    ) -> Tuple[Optional[int], float, Dict[str, Any]]:
        """
        Project grand master down and emit token.
        
        Args:
            grand_master: Grand master state [16]
            context: Optional context for modulation
            wait_for_window: Whether to wait for emission window
            
        Returns:
            (token_id or None, confidence, stats)
        """
        stats = {
            'phase': self.current_phase,
            'in_window': in_emission_window(self.current_phase),
            'time_to_emit': time_to_emission(self.current_phase),
        }
        
        # Check emission window
        if wait_for_window and not in_emission_window(self.current_phase):
            return None, 0.0, stats
        
        if self.embeddings is None:
            raise ValueError("Embeddings not set")
        
        # Step 1: Project down through levels
        # Grand Master → Master (satellite 0 for now)
        master_state = project_level_down(grand_master, self.tensor, target_satellite=0)
        
        # Step 2: Inflate for emission
        inflated = self.grace_inverse.semantic_snap(master_state)
        
        # Step 3: Apply context modulation if provided
        if context is not None:
            # Blend with context
            inflated = (1 - PHI_INV) * inflated + PHI_INV * context
        
        # Step 4: Unbind to find token
        token_id, confidence = unbind_token(inflated, self.embeddings)
        
        stats['token_id'] = token_id
        stats['confidence'] = confidence
        
        # Advance phase
        self.advance_phase(dt=PHI_INV_SQ)
        
        return token_id, confidence, stats
    
    def advance_phase(self, dt: float):
        """Advance emission phase."""
        self.current_phase = (self.current_phase + dt) % (2 * PI)
    
    def reset_phase(self):
        """Reset to start of emission cycle."""
        self.current_phase = 0.0


def project_to_tokens(
    grand_master: np.ndarray,
    embeddings: np.ndarray,
    max_tokens: int = 50,
    confidence_threshold: float = PHI_INV_CUBE
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Generate token sequence from grand master.
    
    Emits tokens until confidence drops below threshold or
    max_tokens is reached.
    
    Args:
        grand_master: Grand master state [16]
        embeddings: Token embeddings [vocab_size, 16]
        max_tokens: Maximum tokens to generate
        confidence_threshold: Minimum confidence for emission
        
    Returns:
        (token_ids, stats)
    """
    projector = DownwardProjection()
    projector.set_embeddings(embeddings)
    
    tokens = []
    stats = {
        'tokens_emitted': 0,
        'avg_confidence': 0.0,
        'stopped_reason': None,
    }
    
    current_state = grand_master.copy()
    total_confidence = 0.0
    
    for i in range(max_tokens):
        # Wait for emission window
        while not in_emission_window(projector.current_phase):
            projector.advance_phase(dt=PHI_INV_SQ)
        
        # Project and emit
        token_id, confidence, emit_stats = projector.project_and_emit(
            current_state,
            wait_for_window=False  # We already waited
        )
        
        if confidence < confidence_threshold:
            stats['stopped_reason'] = 'low_confidence'
            break
        
        tokens.append(token_id)
        total_confidence += confidence
        
        # Modulate current state based on emitted token
        # (feedback from emission)
        token_emb = embeddings[token_id]
        current_state = (1 - PHI_INV_SQ) * current_state + PHI_INV_SQ * token_emb
    
    stats['tokens_emitted'] = len(tokens)
    stats['avg_confidence'] = total_confidence / (len(tokens) + 1e-10)
    
    if stats['stopped_reason'] is None:
        stats['stopped_reason'] = 'max_tokens'
    
    return tokens, stats


# =============================================================================
# TESTS
# =============================================================================

def _test_downward_projection():
    """Test downward projection system."""
    print("Testing downward projection...")
    
    np.random.seed(42)
    
    # Create test embeddings
    vocab_size = 100
    embeddings = np.random.randn(vocab_size, CLIFFORD_DIM) * 0.1
    embeddings[:, GRADE_INDICES[0][0]] += PHI_INV  # Identity bias
    
    # Create grand master state
    grand_master = np.zeros(CLIFFORD_DIM)
    grand_master[GRADE_INDICES[0][0]] = 1.0  # Strong scalar
    grand_master[GRADE_INDICES[4][0]] = 0.5  # Some pseudoscalar
    grand_master[GRADE_INDICES[2]] = np.random.randn(6) * 0.3  # Vorticity
    grand_master[GRADE_INDICES[3]] = np.random.randn(4) * 0.2  # Trivector
    
    # Test 1: Emission window
    print("  Testing emission window...")
    assert in_emission_window(PI * PHI_INV), "Should be in window at start"
    assert not in_emission_window(0), "Should not be in window at 0"
    
    tte = time_to_emission(0)
    assert tte > 0, "Should have positive time to emission"
    print(f"    Time to emission from 0: {tte:.4f}")
    
    # Test 2: Project level down
    print("  Testing level projection...")
    tensor = InteractionTensor()
    satellite = project_level_down(grand_master, tensor, target_satellite=0)
    assert satellite.shape == (CLIFFORD_DIM,), "Should be 16D"
    print(f"    Satellite norm: {np.linalg.norm(satellite):.4f}")
    
    # Test 3: Inflate for emission
    print("  Testing inflation...")
    inflated = inflate_for_emission(satellite)
    assert np.linalg.norm(inflated[GRADE_INDICES[2]]) > 0, "Should have vorticity"
    print(f"    Inflated bivector norm: {np.linalg.norm(inflated[GRADE_INDICES[2]]):.4f}")
    
    # Test 4: Unbind token
    print("  Testing unbinding...")
    token_id, confidence = unbind_token(inflated, embeddings)
    assert 0 <= token_id < vocab_size, "Token should be valid"
    print(f"    Token: {token_id}, Confidence: {confidence:.4f}")
    
    # Test 5: DownwardProjection class
    print("  Testing DownwardProjection class...")
    projector = DownwardProjection()
    projector.set_embeddings(embeddings)
    
    # Set phase to emission window
    projector.current_phase = EMISSION_WINDOW_START
    
    token_id, confidence, stats = projector.project_and_emit(grand_master)
    assert token_id is not None, "Should emit token"
    print(f"    Emitted: {token_id}, Confidence: {confidence:.4f}")
    
    # Test 6: Generate sequence
    print("  Testing sequence generation...")
    tokens, gen_stats = project_to_tokens(
        grand_master, embeddings,
        max_tokens=10,
        confidence_threshold=PHI_INV_CUBE
    )
    
    print(f"    Tokens: {tokens}")
    print(f"    Stats: {gen_stats}")
    
    print("✓ Downward projection tests passed!")
    return True


if __name__ == "__main__":
    _test_downward_projection()
