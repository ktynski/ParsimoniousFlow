"""
Rotor Dynamics — Theory-True Implementation
============================================

Uses actual Clifford algebra rotor computation:
- R = exp(θ/2 × B) where B is bivector
- Sandwich product M' = R × M × R†
- Grace operator for regularization

Trade the self-consistency of the flow, not indicators.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, DTYPE, GRACE_SCALES_FLAT
)
from holographic_prod.core.algebra import (
    build_clifford_basis, build_gamma_matrices,
    geometric_product, wedge_product,
    decompose_to_coefficients, reconstruct_from_coefficients,
    grace_operator, get_cached_basis,
    frobenius_cosine
)

try:
    from .state_encoder import CliffordState
except ImportError:
    from state_encoder import CliffordState


def extract_bivector_from_matrix(M: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Extract grade-2 (bivector) component from Clifford matrix.
    
    Returns [6] array of bivector coefficients.
    """
    coeffs = decompose_to_coefficients(M, basis, np)
    return coeffs[5:11]  # Indices 5-10 are bivector


def bivector_to_matrix(b: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Convert 6D bivector coefficients to 4×4 Clifford matrix.
    """
    coeffs = np.zeros(16, dtype=DTYPE)
    coeffs[5:11] = b
    return reconstruct_from_coefficients(coeffs, basis, np)


def compute_rotor_from_transition(
    M_prev: np.ndarray,
    M_curr: np.ndarray,
    basis: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Compute rotor R such that M_curr ≈ R × M_prev × R†
    
    Theory: The transition M_prev → M_curr can be decomposed as a rotation
    (rotor) plus a non-rotational part. We extract the rotational part.
    
    Method:
    1. Compute log(M_curr × M_prev⁻¹) to get the "velocity" 
    2. The bivector part of this velocity is the rotation generator
    3. R = exp(θ/2 × B) where B is the normalized bivector
    
    For numerical stability, we use the commutator approximation:
    [M_curr, M_prev] / 2 captures the rotational component.
    
    Returns:
        rotor: [4,4] rotor matrix R
        angle: rotation angle θ
        plane: [6] unit bivector coefficients
    """
    # Commutator captures the antisymmetric (rotational) transition
    # [A, B] = AB - BA = 2 × wedge(A, B) for vectors
    comm = M_curr @ M_prev - M_prev @ M_curr
    
    # Extract bivector component
    comm_coeffs = decompose_to_coefficients(comm, basis, np)
    bivector_coeffs = comm_coeffs[5:11]
    
    # Angle is magnitude of bivector
    angle = np.linalg.norm(bivector_coeffs)
    
    if angle < 1e-10:
        # No rotation
        return np.eye(4, dtype=DTYPE), 0.0, np.zeros(6, dtype=DTYPE)
    
    # Unit bivector (rotation plane)
    unit_bivector = bivector_coeffs / angle
    
    # Build bivector matrix
    B = bivector_to_matrix(unit_bivector, basis)
    
    # Verify B² ≈ -I for unit bivector
    B_sq = B @ B
    B_sq_trace = np.trace(B_sq)
    # For unit bivector in Cl(3,1): B² = -|B|² I = -I
    # Normalize if needed
    if abs(B_sq_trace + 4) > 0.5:  # Should be trace = -4
        # Not a pure bivector, renormalize
        norm_factor = np.sqrt(abs(B_sq_trace) / 4)
        if norm_factor > 1e-10:
            B = B / norm_factor
            angle = angle * norm_factor
    
    # Build rotor: R = exp(θ/2 × B) = cos(θ/2)I + sin(θ/2)B
    I = np.eye(4, dtype=DTYPE)
    half_angle = angle / 2
    rotor = np.cos(half_angle) * I + np.sin(half_angle) * B
    
    return rotor, angle, unit_bivector


def apply_rotor(R: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Apply rotor via sandwich product: M' = R × M × R†
    
    For rotors in Cl(3,1), R† = R^T (transpose).
    """
    return R @ M @ R.T


def rotor_prediction_error(
    R: np.ndarray,
    M_prev: np.ndarray,
    M_curr: np.ndarray
) -> float:
    """
    Compute how well the rotor predicts the transition.
    
    Error = ||R × M_prev × R† - M_curr||_F / ||M_curr||_F
    """
    predicted = apply_rotor(R, M_prev)
    diff = predicted - M_curr
    
    error = np.linalg.norm(diff, 'fro') / (np.linalg.norm(M_curr, 'fro') + 1e-10)
    return error


@dataclass
class RotorState:
    """Encapsulates rotor dynamics state."""
    rotor: np.ndarray         # [4,4] rotor matrix
    angle: float              # Rotation angle θ
    plane: np.ndarray         # [6] unit bivector (rotation plane)
    prediction_error: float   # How well rotor predicted transition
    coherence: float          # Stability of rotor over recent history
    chirality_sign: int       # Sign of pseudoscalar (-1, 0, +1)


class RotorPredictor:
    """
    Tracks rotor dynamics and predicts next state.
    
    Theory-true implementation using actual Clifford rotors.
    """
    
    def __init__(self, history_length: int = 5):
        self.history_length = history_length
        self.basis = get_cached_basis(np)
        
        self._state_history: List[np.ndarray] = []
        self._rotor_history: List[Tuple[np.ndarray, float, np.ndarray]] = []
    
    def reset(self):
        """Reset predictor state."""
        self._state_history = []
        self._rotor_history = []
    
    def update(self, state: CliffordState) -> Optional[RotorState]:
        """
        Update with new state and compute rotor dynamics.
        """
        M = state.matrix
        self._state_history.append(M)
        
        if len(self._state_history) > self.history_length + 2:
            self._state_history = self._state_history[-(self.history_length + 2):]
        
        if len(self._state_history) < 2:
            return None
        
        M_prev = self._state_history[-2]
        M_curr = self._state_history[-1]
        
        # Compute rotor from transition
        rotor, angle, plane = compute_rotor_from_transition(M_prev, M_curr, self.basis)
        
        self._rotor_history.append((rotor, angle, plane))
        if len(self._rotor_history) > self.history_length:
            self._rotor_history = self._rotor_history[-self.history_length:]
        
        # Prediction error: apply previous rotor to M_{t-1}, compare to M_t
        if len(self._rotor_history) >= 2:
            prev_rotor, _, _ = self._rotor_history[-2]
            prediction_error = rotor_prediction_error(prev_rotor, M_prev, M_curr)
        else:
            prediction_error = 1.0
        
        # Coherence: how stable is the rotor?
        coherence = self._compute_coherence()
        
        # Chirality sign from pseudoscalar
        chirality_sign = int(np.sign(state.pseudoscalar)) if abs(state.pseudoscalar) > 1e-10 else 0
        
        return RotorState(
            rotor=rotor,
            angle=angle,
            plane=plane,
            prediction_error=prediction_error,
            coherence=coherence,
            chirality_sign=chirality_sign
        )
    
    def _compute_coherence(self) -> float:
        """
        Compute coherence of rotor sequence.
        
        High coherence = stable rotation = geometry predicts itself.
        """
        if len(self._rotor_history) < 2:
            return 0.0
        
        # Angle coherence: low variance relative to mean
        angles = np.array([r[1] for r in self._rotor_history])
        angle_mean = np.mean(np.abs(angles)) + 1e-10
        angle_std = np.std(angles)
        angle_coherence = 1.0 - min(angle_std / angle_mean, 1.0)
        
        # Plane coherence: cosine similarity between consecutive planes
        planes = [r[2] for r in self._rotor_history]
        plane_coherence = 0.0
        count = 0
        for i in range(len(planes) - 1):
            n1 = np.linalg.norm(planes[i])
            n2 = np.linalg.norm(planes[i+1])
            if n1 > 1e-10 and n2 > 1e-10:
                cos_sim = np.dot(planes[i], planes[i+1]) / (n1 * n2)
                plane_coherence += abs(cos_sim)
                count += 1
        if count > 0:
            plane_coherence /= count
        
        # Combined coherence
        return (angle_coherence + plane_coherence) / 2
    
    def predict_next(self) -> Optional[np.ndarray]:
        """
        Predict next multivector using current rotor.
        """
        if len(self._state_history) < 1 or len(self._rotor_history) < 1:
            return None
        
        M_curr = self._state_history[-1]
        rotor, _, _ = self._rotor_history[-1]
        
        return apply_rotor(rotor, M_curr)


class BetTrigger:
    """
    Determines when to bet based on rotor self-consistency.
    
    Theory: Trade the self-consistency of the flow.
    
    Bet when:
    1. Low prediction error (geometry predicts itself)
    2. High coherence (rotor is stable)
    3. Consistent chirality or trend plane
    """
    
    def __init__(
        self,
        error_threshold: float = 0.3,
        coherence_threshold: float = 0.6,
        chirality_persistence: int = 3,
    ):
        self.error_threshold = error_threshold
        self.coherence_threshold = coherence_threshold
        self.chirality_persistence = chirality_persistence
        
        self._chirality_history: List[int] = []
        self._signal_history: List[float] = []
    
    def reset(self):
        """Reset trigger state."""
        self._chirality_history = []
        self._signal_history = []
    
    def evaluate(self, rotor_state: RotorState, clifford_state: CliffordState) -> dict:
        """
        Evaluate whether to trigger a bet.
        """
        self._chirality_history.append(rotor_state.chirality_sign)
        if len(self._chirality_history) > self.chirality_persistence + 2:
            self._chirality_history = self._chirality_history[-(self.chirality_persistence + 2):]
        
        # Check conditions
        error_ok = rotor_state.prediction_error < self.error_threshold
        coherence_ok = rotor_state.coherence > self.coherence_threshold
        
        # Check chirality persistence
        chirality_persistent = False
        chirality_direction = 0
        if len(self._chirality_history) >= self.chirality_persistence:
            recent = self._chirality_history[-self.chirality_persistence:]
            if all(s == recent[0] and s != 0 for s in recent):
                chirality_persistent = True
                chirality_direction = recent[0]
        
        # Bivector magnitude (trend strength)
        bivector_mag = np.linalg.norm(clifford_state.bivector)
        bivector_strong = bivector_mag > 0.1
        
        # Trigger logic
        trigger = error_ok and coherence_ok and (chirality_persistent or bivector_strong)
        
        # Direction
        if chirality_persistent:
            direction = chirality_direction
        elif bivector_strong:
            direction = int(np.sign(clifford_state.bivector[0]))
        else:
            direction = 0
        
        # Confidence
        confidence = rotor_state.coherence * (1 - rotor_state.prediction_error)
        
        # Build reason
        reasons = []
        if error_ok:
            reasons.append(f"low_error({rotor_state.prediction_error:.2f})")
        if coherence_ok:
            reasons.append(f"high_coherence({rotor_state.coherence:.2f})")
        if chirality_persistent:
            reasons.append(f"chirality_persist({chirality_direction})")
        if bivector_strong:
            reasons.append(f"trend_plane({bivector_mag:.2f})")
        
        # Signal
        signal = direction * confidence if trigger else 0.0
        self._signal_history.append(signal)
        if len(self._signal_history) > 100:
            self._signal_history = self._signal_history[-100:]
        
        return {
            'trigger': trigger,
            'direction': direction,
            'confidence': confidence,
            'reason': ' & '.join(reasons) if reasons else 'no_trigger',
            'signal': signal,
            'diagnostics': {
                'prediction_error': rotor_state.prediction_error,
                'coherence': rotor_state.coherence,
                'chirality_sign': rotor_state.chirality_sign,
                'bivector_magnitude': bivector_mag,
                'angle': rotor_state.angle,
            }
        }
    
    def get_cumulative_signal(self, window: int = 10) -> float:
        """Get smoothed signal over recent history."""
        if len(self._signal_history) < window:
            return 0.0
        return np.mean(self._signal_history[-window:])
