"""
CommitmentGate - Basal Ganglia Analog for Action Selection

This implements the brain's late-stage commitment mechanism that collapses
a coherent semantic state into discrete, timed actions.

Brain Mapping:
- Direct pathway (GO): Release highest-probability action when entropy low
- Indirect pathway (NO-GO): Suppress and hold when uncertain
- Hyperdirect pathway (STOP): Emergency brake on very high entropy

Theory:
- This is NOT a punctuation module
- It's a ψ → action collapse gate
- Activates only when semantic entropy is already low
- Implements winner-take-all striatal competition

φ-Derived Thresholds:
- entropy_threshold = φ⁻² ≈ 0.382 (commit when below this)
- confidence_boost = φ⁻¹ ≈ 0.618 (stability requirement)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union, Sequence

# Try CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# φ-derived constants (theory-mandated)
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI           # ≈ 0.618034
PHI_INV_SQ = 1 / PHI**2     # ≈ 0.381966
PHI_INV_CUBE = 1 / PHI**3   # ≈ 0.236068
PHI_EPSILON = PHI_INV ** 20 # ≈ 6.7×10⁻⁹ (theory-true numerical stability)


@dataclass
class GateDecision:
    """Result of commitment gate decision."""
    committed: bool              # Whether to commit to an action
    token: Optional[any]         # The selected token (if committed)
    entropy: float               # Entropy of the distribution
    pathway: str                 # Which pathway was activated
    suppressed: List[any]        # Tokens that were suppressed
    confidence: float            # Confidence in the decision


def compute_entropy(probs: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    
    H = -Σ p_i * log(p_i)
    
    Handles zeros gracefully (0 * log(0) = 0).
    """
    # Handle array module (NumPy or CuPy)
    xp = cp if HAS_CUPY and hasattr(probs, 'device') else np
    
    # Convert to numpy for computation if needed
    if HAS_CUPY and isinstance(probs, cp.ndarray):
        probs = cp.asnumpy(probs)
    
    # Ensure valid probability distribution
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, PHI_EPSILON, 1.0)  # Avoid log(0)
    probs = probs / probs.sum()  # Normalize
    
    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs))
    
    return float(entropy)


def phi_kernel_probs(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores to probabilities via φ-kernel (THEORY-TRUE).
    
    P(i) ∝ score_i^φ — POWER LAW, not softmax!
    
    WHY THIS IS DIFFERENT FROM SOFTMAX:
        - Softmax: P ∝ exp(score/T) — exponential (transformer convention)
        - φ-kernel: P ∝ score^φ — power law (theory-derived)
        
    The φ exponent emerges from self-consistency Λ² = Λ + 1.
    """
    # Handle array module
    if HAS_CUPY and isinstance(scores, cp.ndarray):
        scores_np = cp.asnumpy(scores)
    else:
        scores_np = np.asarray(scores)
    
    # φ-kernel: P(i) ∝ score_i^φ
    scores_positive = np.maximum(scores_np, PHI_EPSILON)
    
    # log(P) ∝ φ * log(score), so logits = φ * log(score)
    # Using log for numerical stability
    logits = np.log(scores_positive) / PHI_INV  # = φ * log(scores)
    logits = logits - np.max(logits)  # Numerical stability
    
    # P = exp(logits) = score^φ (power law!)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)
    
    return probs


# NOTE: softmax() was removed (v5.32.0) - use phi_kernel_probs() instead
# Softmax was transformer convention, NOT theory-true.
# phi_kernel_probs() uses φ-derived kernel per theory.


class CommitmentGate:
    """
    Basal ganglia analog for action selection.
    
    Implements the late-stage commitment mechanism that decides
    WHEN to collapse a semantic state into a discrete action.
    
    This is theory-true to neuroscience:
    - Striatum: competing action representations
    - Direct pathway: GO signal for selected action
    - Indirect pathway: NO-GO signal for alternatives
    - Hyperdirect pathway: STOP signal on high uncertainty
    """
    
    def __init__(
        self,
        entropy_threshold: float = PHI_INV_SQ,  # φ⁻² ≈ 0.382
        hyperdirect_threshold: float = 1.0,      # Emergency brake
    ):
        """
        Initialize commitment gate.
        
        Args:
            entropy_threshold: Commit when entropy below this (φ⁻² by theory)
            hyperdirect_threshold: Emergency stop when entropy above this
            
        Note: Uses φ-kernel (power law) for score → probability, NOT softmax.
        """
        self.entropy_threshold = entropy_threshold
        self.hyperdirect_threshold = hyperdirect_threshold
    
    def decide(
        self,
        scores: Union[np.ndarray, 'cp.ndarray'],
        candidates: Sequence[any],
    ) -> GateDecision:
        """
        Decide whether to commit to an action.
        
        Args:
            scores: Score for each candidate (higher = more likely)
            candidates: Token IDs or values corresponding to scores
            
        Returns:
            GateDecision with commitment status and selected token
        """
        # Validate inputs
        if len(candidates) == 0:
            raise ValueError("Must have at least one candidate")
        
        # Convert to numpy if needed
        if HAS_CUPY and isinstance(scores, cp.ndarray):
            scores_np = cp.asnumpy(scores)
        else:
            scores_np = np.asarray(scores, dtype=np.float64)
        
        candidates = list(candidates)
        
        # Handle single candidate case
        if len(candidates) == 1:
            return GateDecision(
                committed=True,
                token=candidates[0],
                entropy=0.0,
                pathway="direct",
                suppressed=[],
                confidence=1.0,
            )
        
        # Compute probability distribution (φ-kernel, NOT softmax!)
        probs = phi_kernel_probs(scores_np)
        
        # Compute entropy
        entropy = compute_entropy(probs)
        
        # Find winner via φ-kernel sampling (NOT argmax!)
        # THEORY: P ∝ score^φ, then sample from distribution
        winner_idx = int(np.random.choice(len(candidates), p=probs))
        winner_token = candidates[winner_idx]
        winner_prob = probs[winner_idx]
        
        # Determine pathway and commitment
        if entropy > self.hyperdirect_threshold:
            # HYPERDIRECT pathway: Emergency stop, too uncertain
            return GateDecision(
                committed=False,
                token=None,
                entropy=entropy,
                pathway="hyperdirect",
                suppressed=[],
                confidence=winner_prob,
            )
        elif entropy > self.entropy_threshold:
            # INDIRECT pathway: Hold, not confident enough
            return GateDecision(
                committed=False,
                token=None,
                entropy=entropy,
                pathway="indirect",
                suppressed=[],
                confidence=winner_prob,
            )
        else:
            # DIRECT pathway: GO, commit to winner
            suppressed = [c for i, c in enumerate(candidates) if i != winner_idx]
            return GateDecision(
                committed=True,
                token=winner_token,
                entropy=entropy,
                pathway="direct",
                suppressed=suppressed,
                confidence=winner_prob,
            )
    
    def decide_batch(
        self,
        scores_batch: Union[np.ndarray, 'cp.ndarray'],
        candidates: Sequence[any],
    ) -> List[GateDecision]:
        """
        Process multiple decisions in parallel.
        
        Args:
            scores_batch: Shape (batch_size, n_candidates)
            candidates: Token IDs (same for all in batch)
            
        Returns:
            List of GateDecision, one per batch item
        """
        # Convert to numpy if needed
        if HAS_CUPY and isinstance(scores_batch, cp.ndarray):
            scores_np = cp.asnumpy(scores_batch)
        else:
            scores_np = np.asarray(scores_batch, dtype=np.float64)
        
        batch_size = scores_np.shape[0]
        results = []
        
        for i in range(batch_size):
            result = self.decide(scores_np[i], candidates)
            results.append(result)
        
        return results
    
    def forced_commit(
        self,
        scores: Union[np.ndarray, 'cp.ndarray'],
        candidates: Sequence[any],
    ) -> GateDecision:
        """
        Force commitment regardless of entropy (for generation endpoints).
        
        Use sparingly - this bypasses the gate's protective function.
        """
        if HAS_CUPY and isinstance(scores, cp.ndarray):
            scores_np = cp.asnumpy(scores)
        else:
            scores_np = np.asarray(scores, dtype=np.float64)
        
        candidates = list(candidates)
        
        if len(candidates) == 0:
            raise ValueError("Must have at least one candidate")
        
        probs = phi_kernel_probs(scores_np)
        entropy = compute_entropy(probs)
        
        # Even forced commit uses φ-kernel sampling (NOT argmax!)
        winner_idx = int(np.random.choice(len(candidates), p=probs))
        winner_token = candidates[winner_idx]
        suppressed = [c for i, c in enumerate(candidates) if i != winner_idx]
        
        return GateDecision(
            committed=True,
            token=winner_token,
            entropy=entropy,
            pathway="forced",
            suppressed=suppressed,
            confidence=probs[winner_idx],
        )


# =============================================================================
# Integration with Generation Flow
# =============================================================================

def gate_assisted_decode(
    state: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidates: List[int],
    gate: Optional[CommitmentGate] = None,
    vorticity_fn: Optional[callable] = None,
    max_holds: int = 3,
) -> tuple:
    """
    Decode with commitment gating.
    
    This integrates the gate into the generation flow:
    1. Score candidates using vorticity-weighted matching
    2. Pass to gate for commitment decision
    3. If gate holds, allow Grace evolution before retry
    
    Args:
        state: Current semantic state (4x4 matrix)
        candidate_embeddings: Embeddings for candidates (N x 4 x 4)
        candidates: Token IDs
        gate: CommitmentGate instance (default creates one)
        vorticity_fn: Function to compute scores (default uses algebra.vorticity_weighted_scores)
        max_holds: Maximum consecutive holds before forced commit
        
    Returns:
        (token, decision): Selected token and full decision info
    """
    if gate is None:
        gate = CommitmentGate()
    
    if vorticity_fn is None:
        from holographic_prod.core.algebra import vorticity_weighted_scores
        vorticity_fn = vorticity_weighted_scores
    
    # Get scores
    scores = vorticity_fn(state, candidate_embeddings)
    
    # Gate decision
    decision = gate.decide(scores, candidates)
    
    if decision.committed:
        return decision.token, decision
    else:
        # Gate held - force commit after max_holds
        # In practice, caller should evolve state via Grace before retry
        return None, decision


# =============================================================================
# Diagnostic Utilities
# =============================================================================

def diagnose_commitment_failure(
    decision: GateDecision,
    candidates: List[any],
    top_k: int = 5,
) -> str:
    """
    Generate diagnostic message for commitment failures.
    
    Useful for understanding why the gate held.
    """
    if decision.committed:
        return f"Committed to '{decision.token}' via {decision.pathway} pathway"
    
    lines = [
        f"Gate HELD via {decision.pathway} pathway",
        f"  Entropy: {decision.entropy:.4f} (threshold: {PHI_INV_SQ:.4f})",
        f"  Top confidence: {decision.confidence:.4f}",
    ]
    
    if decision.pathway == "hyperdirect":
        lines.append("  Diagnosis: Extremely high uncertainty - semantic state not settled")
    elif decision.pathway == "indirect":
        lines.append("  Diagnosis: Moderate uncertainty - need more Grace evolution")
    
    return "\n".join(lines)
