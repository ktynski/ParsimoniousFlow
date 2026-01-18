"""
Sequence Replay — Temporal Transition Memory via Vorticity

Implements theory-true sequence memory:
- Store transitions, not just snapshots
- High-salience transitions prioritized
- Replay during REM for sequence consolidation

THEORY (Sharp Wave Ripples):
    During REM, the brain replays important sequences.
    This helps consolidate sequential/procedural memory.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ
from holographic_prod.core.algebra import frobenius_cosine, frobenius_cosine_batch
from .structures import EpisodicEntry, TemporalTransition
from .priority import compute_salience


# =============================================================================
# TRANSITION COMPUTATION
# =============================================================================

def compute_transition_vorticity(
    from_matrix: np.ndarray,
    to_matrix: np.ndarray,
    xp = np,
) -> np.ndarray:
    """
    Compute the vorticity (wedge product) encoding a transition.
    
    THEORY:
        Vorticity = (AB - BA) / 2 = A ∧ B
        This captures the antisymmetric (rotational) component of the transition.
        
    Properties:
        - Vorticity = 0 if A and B commute (parallel change)
        - High vorticity = non-commutative change (rotational)
        - Sign encodes direction: A→B vs B→A
    """
    # Wedge product: antisymmetric part of geometric product
    ab = from_matrix @ to_matrix
    ba = to_matrix @ from_matrix
    vorticity = (ab - ba) / 2.0
    
    return vorticity


def record_transition(
    from_context: np.ndarray,
    to_context: np.ndarray,
    from_target: int,
    to_target: int,
    basis: np.ndarray,
    xp = np,
    timestamp: float = 0.0,
) -> TemporalTransition:
    """
    Record a temporal transition between consecutive contexts.
    
    Args:
        from_context: [4, 4] previous context matrix
        to_context: [4, 4] next context matrix
        from_target: Target token for from_context
        to_target: Target token for to_context
        basis: Clifford basis for salience computation
        xp: numpy or cupy
        timestamp: When this transition occurred
        
    Returns:
        TemporalTransition with vorticity encoding
    """
    # Compute vorticity
    vorticity = compute_transition_vorticity(from_context, to_context, xp)
    
    # Compute salience of the TRANSITION (not just endpoints)
    # High vorticity = important transition (non-trivial change)
    vorticity_magnitude = float(xp.linalg.norm(vorticity, 'fro'))
    
    # Also consider endpoint saliences
    from_salience = compute_salience(from_context, basis, xp)
    to_salience = compute_salience(to_context, basis, xp)
    
    # Transition salience = vorticity magnitude + φ²-weighted endpoint saliences
    transition_salience = vorticity_magnitude + PHI_INV_SQ * (from_salience + to_salience)
    
    return TemporalTransition(
        from_context=from_context,
        to_context=to_context,
        vorticity=vorticity,
        from_target=from_target,
        to_target=to_target,
        salience=transition_salience,
        timestamp=timestamp,
    )


# =============================================================================
# TRANSITION BUFFER
# =============================================================================

class TransitionBuffer:
    """
    Buffer for storing and replaying temporal transitions.
    
    THEORY (Sequence Memory):
        - Store transitions, not just snapshots
        - High-salience transitions prioritized
        - Replay during REM for sequence consolidation
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        xp = np,
    ):
        self.capacity = capacity
        self.xp = xp
        self.transitions: List[TemporalTransition] = []
    
    def add(self, transition: TemporalTransition):
        """Add a transition to the buffer."""
        self.transitions.append(transition)
        
        # If over capacity, remove lowest-salience transition
        if len(self.transitions) > self.capacity:
            self.transitions.sort(key=lambda t: t.salience)
            self.transitions.pop(0)  # Remove lowest salience
    
    def add_from_episode_sequence(
        self,
        episodes: List[EpisodicEntry],
        basis: np.ndarray,
        xp = np,
    ):
        """
        Extract and store transitions from a sequence of episodes.
        
        OPTIMIZED: Batch all computations to minimize GPU syncs.
        
        Args:
            episodes: List of EpisodicEntry in temporal order
            basis: Clifford basis
            xp: numpy or cupy
        """
        n_episodes = len(episodes)
        if n_episodes < 2:
            return
        
        n_transitions = n_episodes - 1
        
        # OPTIMIZED: Batch extract matrices (single CPU→GPU transfer)
        matrices_np = np.empty((n_episodes, 4, 4), dtype=np.float32)
        targets = np.empty(n_episodes, dtype=np.int32)
        for i, ep in enumerate(episodes):
            mat = ep.context_matrix
            matrices_np[i] = mat.get() if hasattr(mat, 'get') else mat
            targets[i] = ep.target_token
        matrices = xp.asarray(matrices_np)  # Single GPU transfer
        
        # VECTORIZED: Compute all vorticities in batch
        from_mats = matrices[:-1]  # [n_transitions, 4, 4]
        to_mats = matrices[1:]     # [n_transitions, 4, 4]
        
        # Batch vorticity: (AB - BA) / 2 for all pairs
        ab = xp.einsum('nij,njk->nik', from_mats, to_mats)
        ba = xp.einsum('nij,njk->nik', to_mats, from_mats)
        vorticities = (ab - ba) / 2.0  # [n_transitions, 4, 4]
        
        # Batch norms
        vort_mags = xp.linalg.norm(vorticities.reshape(n_transitions, -1), axis=1)
        
        # Batch saliences for all endpoints
        from holographic_prod.dreaming.priority import compute_salience_batch
        all_saliences = compute_salience_batch(matrices, basis, xp)  # [n_episodes]
        from_saliences = all_saliences[:-1]  # [n_transitions]
        to_saliences = all_saliences[1:]     # [n_transitions]
        
        # Batch transition saliences
        transition_saliences = vort_mags + PHI_INV_SQ * (from_saliences + to_saliences)
        
        # SINGLE bulk transfer to CPU
        vorticities_cpu = vorticities.get() if hasattr(vorticities, 'get') else vorticities
        saliences_cpu = transition_saliences.get() if hasattr(transition_saliences, 'get') else transition_saliences
        
        # Create transitions (pure Python, no GPU syncs)
        for i in range(n_transitions):
            transition = TemporalTransition(
                from_context=matrices_np[i],
                to_context=matrices_np[i+1],
                vorticity=vorticities_cpu[i],
                from_target=int(targets[i]),
                to_target=int(targets[i+1]),
                salience=float(saliences_cpu[i]),
                timestamp=float(i),
            )
            self.add(transition)
    
    def get_high_salience(self, top_k: int = 10) -> List[TemporalTransition]:
        """Get the top-k highest salience transitions."""
        sorted_transitions = sorted(self.transitions, key=lambda t: -t.salience)
        return sorted_transitions[:top_k]
    
    def sample_for_replay(
        self,
        n_samples: int,
    ) -> List[TemporalTransition]:
        """
        Sample transitions for replay, weighted by salience.
        
        THEORY-TRUE: Uses direct salience weighting (NOT temperature-scaled softmax).
        Salience = witness magnitude = what survives Grace = intrinsic importance.
        
        Args:
            n_samples: Number of transitions to sample
            
        Returns:
            List of sampled transitions
        """
        if not self.transitions or n_samples <= 0:
            return []
        
        xp = self.xp
        
        # THEORY-TRUE: Direct salience-weighted sampling (no arbitrary softmax)
        # Salience = scalar + pseudoscalar content = what survives Grace
        saliences = xp.array([t.salience for t in self.transitions])
        
        # Ensure positive weights
        weights = xp.maximum(saliences, 1e-8)
        
        # Normalize to probability distribution (this IS justified)
        probs = weights / xp.sum(weights)
        
        # Sample indices - USE NUMPY for weighted sampling (CuPy doesn't support p parameter)
        n_actual = min(n_samples, len(self.transitions))
        # CuPy arrays need .get() for explicit conversion
        if hasattr(probs, 'get'):
            probs_np = probs.get()  # CuPy → NumPy
        else:
            probs_np = np.array(probs)
        probs_np = probs_np / probs_np.sum()  # Ensure sums to 1
        indices = np.random.choice(
            len(self.transitions),
            size=n_actual,
            replace=False,
            p=probs_np,
        )
        
        return [self.transitions[int(i)] for i in indices]
    
    def replay_sequence(
        self,
        start_transition: TemporalTransition,
        max_length: int = 5,
        similarity_threshold: float = PHI_INV,  # φ-derived: φ⁻¹ ≈ 0.618
    ) -> List[TemporalTransition]:
        """
        Replay a sequence starting from a given transition.
        
        Find subsequent transitions whose from_context matches the
        current to_context (chained replay).
        
        v5.31.2: VECTORIZED - batch similarity computation (was O(max_length × N) GPU ops)
        Now O(max_length) GPU ops with batched frobenius_cosine_batch.
        
        Args:
            start_transition: Starting point of replay
            max_length: Maximum sequence length
            similarity_threshold: φ-derived threshold for context chaining
            
        Returns:
            List of chained transitions forming a sequence
        """
        xp = self.xp
        n_transitions = len(self.transitions)
        
        if n_transitions == 0:
            return [start_transition]
        
        # Pre-stack all from_contexts ONCE (avoid repeated access)
        all_from_contexts = xp.stack([t.from_context for t in self.transitions])  # [N, 4, 4]
        
        sequence = [start_transition]
        used_mask = xp.zeros(n_transitions, dtype=bool)  # Track used indices
        
        # Mark starting transition as used
        start_idx = next((i for i, t in enumerate(self.transitions) if id(t) == id(start_transition)), -1)
        if start_idx >= 0:
            used_mask[start_idx] = True
        
        current_to = xp.asarray(start_transition.to_context)  # Ensure on GPU/CPU
        
        for _ in range(max_length - 1):
            # VECTORIZED: Batch similarity computation - SINGLE GPU op
            sims = frobenius_cosine_batch(current_to, all_from_contexts, xp)  # [N]
            
            # Mask out already-used transitions
            sims = xp.where(used_mask, xp.array(-float('inf')), sims)
            
            # Find best
            best_idx = int(xp.argmax(sims))
            best_sim = float(sims[best_idx])
            
            if best_sim < similarity_threshold:
                break
            
            best_match = self.transitions[best_idx]
            sequence.append(best_match)
            used_mask[best_idx] = True
            current_to = xp.asarray(best_match.to_context)
        
        return sequence
    
    def stats(self) -> Dict[str, Any]:
        """Return buffer statistics."""
        if not self.transitions:
            return {
                'count': 0,
                'capacity': self.capacity,
                'avg_salience': 0.0,
                'avg_vorticity_norm': 0.0,
            }
        
        xp = self.xp
        saliences = [t.salience for t in self.transitions]
        vort_norms = [float(xp.linalg.norm(t.vorticity, 'fro')) for t in self.transitions]
        
        return {
            'count': len(self.transitions),
            'capacity': self.capacity,
            'avg_salience': float(xp.mean(xp.array(saliences))),
            'max_salience': float(xp.max(xp.array(saliences))),
            'avg_vorticity_norm': float(xp.mean(xp.array(vort_norms))),
        }


# =============================================================================
# REPLAY DURING REM
# =============================================================================

def replay_transitions_during_rem(
    transition_buffer: TransitionBuffer,
    basis: np.ndarray,
    xp = np,
    n_replays: int = 10,
    sequence_length: int = 3,
) -> Tuple[List[List[TemporalTransition]], Dict[str, Any]]:
    """
    Replay sequences during REM sleep.
    
    THEORY (Sharp Wave Ripples):
        During REM, the brain replays important sequences.
        This helps consolidate sequential/procedural memory.
        
    Process:
        1. Sample high-salience starting transitions
        2. Chain them into sequences
        3. Apply Grace to the sequences (test stability)
        4. Return surviving sequences for schema discovery
    
    Args:
        transition_buffer: Buffer with stored transitions
        basis: Clifford basis
        xp: numpy or cupy
        n_replays: Number of sequences to replay
        sequence_length: Target length of each sequence
        
    Returns:
        replayed_sequences: List of transition sequences
        stats: Replay statistics
    """
    if transition_buffer.stats()['count'] == 0:
        return [], {'replayed': 0, 'avg_length': 0}
    
    # Sample starting transitions
    starters = transition_buffer.sample_for_replay(n_replays)
    
    # Replay sequences
    sequences = []
    total_length = 0
    
    for start in starters:
        seq = transition_buffer.replay_sequence(
            start,
            max_length=sequence_length,
            similarity_threshold=PHI_INV,  # φ-derived
        )
        sequences.append(seq)
        total_length += len(seq)
    
    stats = {
        'replayed': len(sequences),
        'avg_length': total_length / max(len(sequences), 1),
        'max_length': max(len(s) for s in sequences) if sequences else 0,
    }
    
    return sequences, stats
