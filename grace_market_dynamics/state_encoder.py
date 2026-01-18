"""
Observable-Agnostic State Encoder — Theory-True Implementation
==============================================================

Uses actual Cl(3,1) Clifford algebra (4×4 real matrices) from holographic_prod.

The encoding pipeline:
1. Gram matrix G_t from increment covariance
2. Eigendecompose: G = UΛU^T → canonical 4D frame
3. Project increment into canonical frame: v_t ∈ ℝ⁴
4. Embed v_t as grade-1 vector in Cl(3,1)
5. Build full multivector via geometric products of lagged vectors
6. Apply Grace operator for regularization

All operations use the actual Clifford geometric product (matrix multiplication),
not linear approximations.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import sys
import os

# Add holographic_prod to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, DTYPE, GRACE_SCALES_FLAT,
    GRADE_INDICES
)
from holographic_prod.core.algebra import (
    build_clifford_basis, build_gamma_matrices,
    geometric_product, wedge_product,
    decompose_to_coefficients, reconstruct_from_coefficients,
    grace_operator, get_cached_basis
)


@dataclass
class CliffordState:
    """
    Full 16D Clifford multivector state as 4×4 matrix.
    
    Theory: M ∈ Cl(3,1) ≅ M₄(ℝ)
    """
    matrix: np.ndarray  # [4, 4] - the actual Clifford element
    coefficients: np.ndarray  # [16] - decomposition into basis
    
    @property
    def scalar(self) -> float:
        """Grade 0 component."""
        return float(self.coefficients[0])
    
    @property
    def vector(self) -> np.ndarray:
        """Grade 1 components (4D)."""
        return self.coefficients[1:5]
    
    @property
    def bivector(self) -> np.ndarray:
        """Grade 2 components (6D) - vorticity/rotation."""
        return self.coefficients[5:11]
    
    @property
    def trivector(self) -> np.ndarray:
        """Grade 3 components (4D)."""
        return self.coefficients[11:15]
    
    @property
    def pseudoscalar(self) -> float:
        """Grade 4 component - chirality."""
        return float(self.coefficients[15])
    
    def grade_magnitudes(self) -> dict:
        """Magnitude of each grade."""
        return {
            'G0': abs(self.scalar),
            'G1': np.linalg.norm(self.vector),
            'G2': np.linalg.norm(self.bivector),
            'G3': np.linalg.norm(self.trivector),
            'G4': abs(self.pseudoscalar)
        }
    
    def witness(self) -> Tuple[float, float]:
        """
        Extract witness (scalar, pseudoscalar) - the stable core under Grace.
        
        Theory: Grace contracts all grades toward the witness (G0, G4).
        """
        return (self.scalar, self.pseudoscalar)
    
    @property
    def coeffs(self) -> np.ndarray:
        """Alias for coefficients (for compatibility)."""
        return self.coefficients
    
    @classmethod
    def from_increments(cls, increments: np.ndarray, basis: np.ndarray) -> 'CliffordState':
        """
        Create CliffordState directly from 4D increments.
        
        This constructor embeds increments as a normalized Clifford multivector
        with meaningful structure at all grades.
        
        Theory-true encoding:
        - Grade 0: energy (norm)
        - Grade 1: direction (normalized vector)
        - Grade 2: rotation signature (from sign patterns)
        - Grade 3: higher-order structure
        - Grade 4: chirality (overall orientation)
        
        Args:
            increments: [4] vector of log-return increments
            basis: [16, 4, 4] Clifford basis
            
        Returns:
            CliffordState with full 16D representation
        """
        # Ensure we have exactly 4 elements as 1D array
        inc = np.asarray(increments, dtype=np.float64).flatten()
        
        if len(inc) > 4:
            inc = inc[:4]
        elif len(inc) < 4:
            inc = np.pad(inc, (0, 4 - len(inc)), mode='constant')
        
        # Ensure correct shape
        inc = inc.reshape(4)
        
        # Energy and direction
        alpha = np.linalg.norm(inc)
        if alpha < 1e-15:
            alpha = 1e-15
            
        # Normalized direction
        v_hat = inc / alpha
        
        # Build coefficients with meaningful structure
        coeffs = np.zeros(16, dtype=DTYPE)
        
        # Grade 0: scalar (log energy to handle scale)
        coeffs[0] = np.sign(alpha) * np.log1p(abs(alpha) * 100) / np.log1p(100)  # Normalized to ~[0,1]
        
        # Grade 1: normalized direction components
        coeffs[1:5] = v_hat
        
        # Grade 2: bivector components from direction relationships
        # These capture "rotation planes" between axes
        # e12 = relationship between axes 1 and 2
        coeffs[5] = (v_hat[0] - v_hat[1]) * PHI_INV     # e12: x vs y comparison
        coeffs[6] = (v_hat[0] - v_hat[2]) * PHI_INV     # e13: x vs z comparison  
        coeffs[7] = (v_hat[0] - v_hat[3]) * PHI_INV     # e14: x vs t comparison
        coeffs[8] = (v_hat[1] - v_hat[2]) * PHI_INV     # e23: y vs z comparison
        coeffs[9] = (v_hat[1] - v_hat[3]) * PHI_INV     # e24: y vs t comparison
        coeffs[10] = (v_hat[2] - v_hat[3]) * PHI_INV    # e34: z vs t comparison
        
        # Sign pattern: use np.sign but treat 0 as +1 to avoid zeroing out
        # (Small positive increments are "up", small negative are "down")
        eps = 1e-15
        sign_pattern = np.where(np.abs(inc) < eps, 1.0, np.sign(inc))
        
        # Grade 3: trivector from three-way sign combinations
        # These capture "momentum" patterns (all-up, all-down, mixed)
        coeffs[11] = sign_pattern[0] * sign_pattern[1] * sign_pattern[2] * PHI_INV_SQ       # e123
        coeffs[12] = sign_pattern[0] * sign_pattern[1] * sign_pattern[3] * PHI_INV_SQ       # e124
        coeffs[13] = sign_pattern[0] * sign_pattern[2] * sign_pattern[3] * PHI_INV_SQ       # e134
        coeffs[14] = sign_pattern[1] * sign_pattern[2] * sign_pattern[3] * PHI_INV_SQ       # e234
        
        # Grade 4: pseudoscalar (chirality) = overall sign pattern
        # This is the "handed-ness" of the market movement
        chirality = sign_pattern[0] * sign_pattern[1] * sign_pattern[2] * sign_pattern[3]
        
        # Also weight by directional consistency (how aligned the moves are)
        consistency = np.mean(sign_pattern)  # -1 to +1: all down to all up
        coeffs[15] = chirality * (0.5 + 0.5 * abs(consistency)) * PHI_INV
        
        # Reconstruct matrix from coefficients
        M = reconstruct_from_coefficients(coeffs.astype(DTYPE), basis, np)
        
        return cls(matrix=M, coefficients=coeffs.astype(DTYPE))


def vector_to_clifford(v: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Embed 4-vector as grade-1 Clifford element.
    
    Theory: v = v^μ e_μ → V = v^μ γ_μ (4×4 matrix)
    
    Args:
        v: [4] vector components
        gamma: [4, 4, 4] gamma matrices
        
    Returns:
        [4, 4] Clifford grade-1 element
    """
    return np.sum(v[:, None, None] * gamma, axis=0)


def clifford_exp_bivector(B: np.ndarray, theta: float) -> np.ndarray:
    """
    Compute rotor R = exp(θ/2 × B) where B is unit bivector.
    
    Theory: For unit bivector B² = -1, we have:
        exp(θ/2 × B) = cos(θ/2) × I + sin(θ/2) × B
        
    This is the half-angle formula. R rotates vectors by angle θ.
    
    Args:
        B: [4, 4] unit bivector (B² = -1)
        theta: rotation angle (full angle, not half)
        
    Returns:
        [4, 4] rotor matrix
    """
    I = np.eye(4, dtype=DTYPE)
    half_theta = theta / 2
    return np.cos(half_theta) * I + np.sin(half_theta) * B


def sandwich_product(R: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Apply rotor via sandwich product: M' = R × M × R†
    
    For rotors, R† = R^T (transpose, since R is orthogonal).
    
    This is THE theory-true way to rotate multivectors.
    """
    return R @ M @ R.T


class ObservableAgnosticEncoder:
    """
    Encodes time series into Cl(3,1) multivector without
    assuming any specific observable structure.
    
    Theory-true implementation using actual Clifford algebra.
    """
    
    def __init__(
        self,
        gram_window: int = 20,
        continuity_threshold: float = 0.5,
        use_log_returns: bool = True,
        apply_grace: bool = True
    ):
        """
        Args:
            gram_window: Window size for Gram matrix computation
            continuity_threshold: Min cosine similarity for eigenvector sign continuity
            use_log_returns: Whether to take log of input before differencing
            apply_grace: Whether to apply Grace regularization
        """
        self.gram_window = gram_window
        self.continuity_threshold = continuity_threshold
        self.use_log_returns = use_log_returns
        self.apply_grace = apply_grace
        
        # Clifford algebra infrastructure
        self.gamma = build_gamma_matrices(np)
        self.basis = get_cached_basis(np)
        
        # State tracking
        self._prev_eigvecs: Optional[np.ndarray] = None
        self._history: List[np.ndarray] = []
        self._canonical_history: List[np.ndarray] = []  # 4-vectors in canonical frame
        self._clifford_history: List[np.ndarray] = []   # Clifford matrices
    
    def reset(self):
        """Reset encoder state."""
        self._prev_eigvecs = None
        self._history = []
        self._canonical_history = []
        self._clifford_history = []
    
    def _compute_gram_matrix(self, increments: np.ndarray) -> np.ndarray:
        """
        Compute Gram matrix from increments.
        
        G = Σ Δy_i Δy_i^T
        
        Key property: if you apply unknown linear mixing y → Ay,
        then G → AGA^T. The eigenspectrum is intrinsic.
        """
        return increments.T @ increments
    
    def _compute_canonical_frame(self, gram: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract canonical 4D frame from Gram matrix via eigendecomposition.
        
        Returns:
            eigvecs: [d, 4] top 4 eigenvectors (columns)
            eigvals: [4] corresponding eigenvalues
        """
        eigvals, eigvecs = np.linalg.eigh(gram)
        
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take top 4 (or pad if rank < 4)
        d = eigvecs.shape[1]
        if d < 4:
            eigvecs = np.pad(eigvecs, ((0, 0), (0, 4-d)), mode='constant')
            eigvals = np.pad(eigvals, (0, 4-d), mode='constant')
        else:
            eigvecs = eigvecs[:, :4]
            eigvals = eigvals[:4]
        
        # Enforce sign continuity with previous frame
        if self._prev_eigvecs is not None:
            for i in range(min(4, eigvecs.shape[0], self._prev_eigvecs.shape[0])):
                if eigvecs.shape[0] == self._prev_eigvecs.shape[0]:
                    dot = np.dot(eigvecs[:, i], self._prev_eigvecs[:, i])
                    if dot < -self.continuity_threshold:
                        eigvecs[:, i] *= -1
        
        self._prev_eigvecs = eigvecs.copy()
        return eigvecs, eigvals
    
    def _project_to_canonical(self, y: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Project observation into canonical 4D frame.
        
        v = U^(4)^T Δy
        
        This is feature-order agnostic and linear-mixing agnostic.
        """
        if len(y) < frame.shape[0]:
            y = np.pad(y, (0, frame.shape[0] - len(y)), mode='constant')
        elif len(y) > frame.shape[0]:
            y = y[:frame.shape[0]]
        
        return frame.T @ y
    
    def _build_multivector(
        self,
        v0: np.ndarray,  # Current canonical vector
        v1: np.ndarray,  # t-1
        v2: np.ndarray,  # t-2
        v3: np.ndarray   # t-3
    ) -> np.ndarray:
        """
        Build full Clifford multivector from lagged canonical vectors.
        
        Theory: Use geometric products to fill all grades.
        
        M = α×I + V₀ + V₀∧V₁ + V₀∧V₁∧V₂ + V₀∧V₁∧V₂∧V₃
        
        Where:
        - α = |v₀| (scalar = energy)
        - V₀ = v₀ embedded as Clifford vector
        - V₀∧V₁ = bivector (turning plane)
        - V₀∧V₁∧V₂ = trivector (3-way interaction)
        - V₀∧V₁∧V₂∧V₃ = pseudoscalar (chirality)
        """
        I = np.eye(4, dtype=DTYPE)
        
        # Embed vectors as Clifford elements
        V0 = vector_to_clifford(v0.astype(DTYPE), self.gamma)
        V1 = vector_to_clifford(v1.astype(DTYPE), self.gamma)
        V2 = vector_to_clifford(v2.astype(DTYPE), self.gamma)
        V3 = vector_to_clifford(v3.astype(DTYPE), self.gamma)
        
        # Scalar: energy (Frobenius norm scaled)
        alpha = np.linalg.norm(v0)
        
        # Build multivector using actual Clifford operations
        # Grade 0: scalar × identity
        M = alpha * I
        
        # Grade 1: current vector (normalized)
        if alpha > 1e-10:
            M = M + V0 / alpha
        
        # Grade 2: bivector from wedge product
        B01 = wedge_product(V0, V1, np)
        B01_norm = np.linalg.norm(B01, 'fro')
        if B01_norm > 1e-10:
            M = M + B01 / B01_norm * PHI_INV  # Scale by φ⁻¹
        
        # Grade 3: trivector from triple wedge
        # V₀∧V₁∧V₂ = (V₀∧V₁)∧V₂ = V₀∧(V₁∧V₂) 
        B12 = wedge_product(V1, V2, np)
        T012 = wedge_product(V0, B12, np)
        T_norm = np.linalg.norm(T012, 'fro')
        if T_norm > 1e-10:
            M = M + T012 / T_norm * PHI_INV_SQ  # Scale by φ⁻²
        
        # Grade 4: pseudoscalar from quadruple wedge
        # V₀∧V₁∧V₂∧V₃ - proportional to 4×4 determinant
        B23 = wedge_product(V2, V3, np)
        Q0123 = wedge_product(B01, B23, np)
        Q_norm = np.linalg.norm(Q0123, 'fro')
        if Q_norm > 1e-10:
            M = M + Q0123 / Q_norm * PHI_INV  # φ⁻¹ for pseudoscalar (Fibonacci)
        
        return M
    
    def update(self, observation: np.ndarray) -> Optional[CliffordState]:
        """
        Process new observation and return current Clifford state.
        
        Args:
            observation: Raw observation vector (any dimension)
            
        Returns:
            CliffordState if enough history accumulated, else None
        """
        # Apply log if requested
        obs = observation.copy().astype(np.float64)
        if self.use_log_returns and np.all(obs > 0):
            obs = np.log(obs)
        
        self._history.append(obs)
        
        # Need at least gram_window + 4 observations
        if len(self._history) < self.gram_window + 4:
            return None
        
        # Keep history bounded
        if len(self._history) > self.gram_window + 10:
            self._history = self._history[-(self.gram_window + 10):]
        
        # Compute increments
        history_arr = np.array(self._history)
        increments = np.diff(history_arr, axis=0)
        
        # Use last gram_window increments
        window_increments = increments[-self.gram_window:]
        
        # Compute Gram matrix
        gram = self._compute_gram_matrix(window_increments)
        
        # Get canonical frame
        frame, eigvals = self._compute_canonical_frame(gram)
        
        # Project current increment into canonical frame
        current_increment = increments[-1]
        v_t = self._project_to_canonical(current_increment, frame)
        
        # Store in canonical history
        self._canonical_history.append(v_t)
        if len(self._canonical_history) > 10:
            self._canonical_history = self._canonical_history[-10:]
        
        # Need at least 4 canonical vectors
        if len(self._canonical_history) < 4:
            return None
        
        # Get lagged vectors
        v0 = self._canonical_history[-1]
        v1 = self._canonical_history[-2]
        v2 = self._canonical_history[-3]
        v3 = self._canonical_history[-4]
        
        # Build full Clifford multivector
        M = self._build_multivector(v0, v1, v2, v3)
        
        # Apply Grace operator for regularization
        if self.apply_grace:
            M = grace_operator(M, self.basis, np)
        
        # Store Clifford matrix
        self._clifford_history.append(M)
        if len(self._clifford_history) > 10:
            self._clifford_history = self._clifford_history[-10:]
        
        # Decompose into coefficients
        coeffs = decompose_to_coefficients(M, self.basis, np)
        
        return CliffordState(matrix=M, coefficients=coeffs)
    
    def get_invariants(self) -> dict:
        """
        Extract gauge-invariant quantities from current state.
        """
        if len(self._history) < self.gram_window + 1:
            return {}
        
        history_arr = np.array(self._history)
        increments = np.diff(history_arr, axis=0)[-self.gram_window:]
        gram = self._compute_gram_matrix(increments)
        
        eigvals = np.linalg.eigvalsh(gram)[::-1]
        
        # Scale: trace/Frobenius
        scale = np.trace(gram)
        
        # Anisotropy: ratio of eigenvalues
        total = eigvals.sum() + 1e-10
        anisotropy = eigvals[:4] / total if len(eigvals) >= 4 else eigvals / total
        
        # Effective dimension (entropy)
        aniso_nonzero = anisotropy[anisotropy > 1e-10]
        eff_dim = np.exp(-np.sum(aniso_nonzero * np.log(aniso_nonzero))) if len(aniso_nonzero) > 0 else 1.0
        
        return {
            'scale': scale,
            'energy': np.sqrt(scale),
            'anisotropy': anisotropy,
            'effective_dimension': eff_dim
        }


def delay_embed(prices: np.ndarray, delays: int = 4) -> np.ndarray:
    """
    Embed scalar price series into higher dimension via delays.
    
    y_t = [Δlog(p_t), Δlog(p_{t-1}), ..., Δlog(p_{t-delays+1})]
    """
    if prices.ndim > 1:
        prices = prices.flatten()
    
    log_returns = np.diff(np.log(prices))
    
    n = len(log_returns) - delays + 1
    if n < 1:
        raise ValueError(f"Need at least {delays+1} prices, got {len(prices)}")
    
    embedded = np.zeros((n, delays))
    for i in range(n):
        embedded[i] = log_returns[i:i+delays][::-1]
    
    return embedded
