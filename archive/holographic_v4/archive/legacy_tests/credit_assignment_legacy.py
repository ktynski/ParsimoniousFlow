"""
Credit Assignment Module — Provenance Tracing and Targeted Reconsolidation
==========================================================================

This module solves the CREDIT ASSIGNMENT PROBLEM for the holographic memory system:
When a prediction is wrong, WHICH memories caused the error?

THEORY:
    Unlike gradient descent (where errors flow backward through computation),
    our associative memory needs explicit provenance tracking:
    
    1. TRACE: Record which memories were retrieved/considered
    2. ATTRIBUTE: When wrong, identify culprit memories
    3. RECONSOLIDATE: Update specific memories, not everything
    
    This is backpropagation THROUGH MEMORY, not through computation.

KEY INSIGHT:
    Vorticity signatures act as "fingerprints" — similar structures activate
    similar memories. When we trace, we record both:
    - WHICH memories were considered (indices)
    - HOW strongly they contributed (confidence × similarity)
    
    Blame is proportional to: contribution × error_magnitude

NO GRADIENT REQUIRED:
    We don't compute gradients. Instead:
    - High blame → lerp toward correct answer (rate φ⁻¹)
    - Low blame → small correction (rate φ⁻² × blame)
    - Zero blame → no change

MEMORY SAFETY:
    Reconsolidation respects the same φ-decay equilibrium as normal learning.
    It's NOT override — it's adjustment within the existing framework.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import OrderedDict
import time

# TYPE_CHECKING imports to avoid circular dependencies
if TYPE_CHECKING:
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.dreaming import DreamingSystem

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE
from holographic_v4.algebra import (
    build_clifford_basis,
    geometric_product,
    frobenius_similarity,
    frobenius_similarity_batch,
    decompose_to_coefficients,
    grace_operator,
)
from holographic_v4.quotient import (
    extract_witness,
    witness_matrix,
    grace_stability,
    quotient_similarity,
    adaptive_similarity,
    adaptive_similarity_batch,
    vorticity_weighted_scores,
)

Array = np.ndarray
ArrayModule = type(np)


# =============================================================================
# PROVENANCE TRACE — Record of memory contributions
# =============================================================================

@dataclass
class ProvenanceTrace:
    """
    Complete record of which memories contributed to a prediction.
    
    This is the "paper trail" for credit assignment — when a prediction
    is wrong, we can trace back to see who's responsible.
    
    Attributes:
        query_hash: Hash of the query matrix (for lookup)
        query_matrix: The actual query matrix
        retrieved_indices: Indices of attractors that were retrieved/considered
        prototype_ids: IDs of semantic prototypes that matched
        confidence_scores: How confident each contributor was
        similarity_scores: Raw similarity of each contributor
        vorticity_signature: The query's structural fingerprint
        timestamp: When this trace was recorded
        predicted_target: What was predicted
        attractor_matrices: The actual attractor matrices (for detailed analysis)
    """
    query_hash: int
    query_matrix: Array
    retrieved_indices: List[int]
    prototype_ids: List[str]
    confidence_scores: List[float]
    similarity_scores: List[float]
    vorticity_signature: Array
    timestamp: float
    predicted_target: Optional[int] = None
    attractor_matrices: Optional[List[Array]] = None
    
    def total_contribution(self, idx: int) -> float:
        """Get total contribution weight for a given index."""
        try:
            pos = self.retrieved_indices.index(idx)
            return self.confidence_scores[pos] * self.similarity_scores[pos]
        except ValueError:
            return 0.0


# =============================================================================
# PROVENANCE TRACKER — Maintains history for credit assignment
# =============================================================================

class ProvenanceTracker:
    """
    Tracks provenance across multiple retrievals.
    
    Maintains a bounded history of traces for post-hoc credit assignment.
    Uses LRU eviction when history exceeds max_history.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._traces: OrderedDict[int, ProvenanceTrace] = OrderedDict()
        self._basis = build_clifford_basis()
    
    def record_retrieval(
        self,
        query_matrix: Array,
        model: 'TheoryTrueModel',
        dreaming: Optional['DreamingSystem'] = None,
    ) -> ProvenanceTrace:
        """
        Record a complete retrieval trace.
        
        Args:
            query_matrix: The query being retrieved
            model: The model being queried
            dreaming: Optional dreaming system for semantic memory
            
        Returns:
            ProvenanceTrace capturing all contributors
        """
        trace = trace_retrieval(query_matrix, model, dreaming, self._basis)
        
        # Add to history with LRU eviction
        self._traces[trace.query_hash] = trace
        self._traces.move_to_end(trace.query_hash)
        
        while len(self._traces) > self.max_history:
            self._traces.popitem(last=False)
        
        return trace
    
    def get_trace(self, query_hash: int) -> Optional[ProvenanceTrace]:
        """Look up a trace by query hash."""
        trace = self._traces.get(query_hash)
        if trace is not None:
            self._traces.move_to_end(query_hash)  # LRU update
        return trace
    
    def get_history(self) -> List[ProvenanceTrace]:
        """Get all traces in order (oldest first)."""
        return list(self._traces.values())
    
    def clear(self):
        """Clear all traces."""
        self._traces.clear()
    
    def get_contributors(self, query_hash: int) -> List[int]:
        """Get the indices of memories that contributed to a prediction."""
        trace = self.get_trace(query_hash)
        if trace is None:
            return []
        return trace.retrieved_indices
    
    def assign_credit(
        self,
        error_signal: float,
        trace: ProvenanceTrace,
    ) -> Dict[int, float]:
        """
        Assign credit/blame based on a trace.
        
        Args:
            error_signal: Magnitude of error (can be negative for "good")
            trace: The provenance trace
            
        Returns:
            Dict mapping memory index to blame score
        """
        attribution = {}
        total_contribution = sum(
            c * s for c, s in zip(trace.confidence_scores, trace.similarity_scores)
        )
        
        if total_contribution < 1e-10:
            return attribution
        
        for i, idx in enumerate(trace.retrieved_indices):
            contribution = trace.confidence_scores[i] * trace.similarity_scores[i]
            blame = error_signal * (contribution / total_contribution)
            attribution[idx] = blame
        
        return attribution


# =============================================================================
# TRACE RETRIEVAL — Capture what contributed to a prediction
# =============================================================================

def trace_retrieval(
    query_matrix: Array,
    model: 'TheoryTrueModel',
    dreaming: Optional['DreamingSystem'] = None,
    basis: Optional[Array] = None,
    top_k: int = 10,
) -> ProvenanceTrace:
    """
    Create a complete provenance trace for a retrieval.
    
    This records:
    1. Which attractors in episodic memory were similar
    2. Which prototypes in semantic memory matched
    3. The confidence and similarity of each
    
    Args:
        query_matrix: The query being retrieved
        model: The model to trace
        dreaming: Optional DreamingSystem for semantic memory
        basis: Clifford basis (built if not provided)
        top_k: How many top matches to record
        
    Returns:
        ProvenanceTrace with full attribution information
    """
    if basis is None:
        basis = build_clifford_basis()
    
    xp = model.xp
    
    # Compute query hash and structural signature
    # For a single matrix, we use Clifford coefficients as the "signature"
    # (vorticity_signature requires a sequence of matrices)
    query_hash = hash(tuple(query_matrix.flatten().tolist()))
    vort_sig = decompose_to_coefficients(query_matrix, basis, xp)
    
    # Track contributions
    retrieved_indices = []
    prototype_ids = []
    confidence_scores = []
    similarity_scores = []
    attractor_matrices = []
    
    # 1. Check episodic memory (attractors in model)
    if model.num_attractors > 0:
        # Compute similarities to all attractors
        if model.use_adaptive_similarity:
            sims = adaptive_similarity_batch(
                query_matrix, model.attractor_matrices[:model.num_attractors], 
                basis, xp
            )
        else:
            sims = frobenius_similarity_batch(
                query_matrix, model.attractor_matrices[:model.num_attractors], xp
            )
        
        # Get top-k matches
        top_indices = xp.argsort(sims)[-top_k:][::-1]
        
        for idx in top_indices:
            idx = int(idx)
            sim = float(sims[idx])
            if sim > PHI_INV_CUBE:  # φ⁻³ ≈ 0.236 minimum relevance threshold
                retrieved_indices.append(idx)
                similarity_scores.append(sim)
                
                # Confidence based on grace stability of the attractor
                attractor = model.attractor_matrices[idx]
                stab = grace_stability(attractor, basis, xp)
                confidence_scores.append(float(stab))
                attractor_matrices.append(attractor.copy())
    
    # 2. Check semantic memory (prototypes in dreaming)
    if dreaming is not None:
        # Check if semantic memory is initialized before querying
        if hasattr(dreaming, 'semantic_memory') and dreaming.semantic_memory is not None:
            try:
                sem_stats = dreaming.semantic_memory.stats()
                if sem_stats.get('total_prototypes', 0) > 0:
                    # Query semantic memory
                    matches = dreaming.semantic_memory.retrieve(query_matrix, top_k=top_k)
                    for proto, sim in matches:
                        prototype_ids.append(proto.prototype_id if hasattr(proto, 'prototype_id') else str(id(proto)))
            except (AttributeError, KeyError) as e:
                # Semantic memory exists but stats() or retrieve() failed - this is a bug
                # CRITICAL: Don't silently ignore bugs - raise error
                raise RuntimeError(
                    f"Semantic memory query failed in trace_provenance: {e}. "
                    "This indicates a bug in semantic memory interface."
                ) from e
            except Exception as e:
                # Unexpected error - should not be silently ignored
                raise RuntimeError(f"Unexpected error querying semantic memory: {e}") from e
    
    # Determine predicted target
    predicted_target = None
    if retrieved_indices and model.num_attractors > 0:
        best_idx = retrieved_indices[0]
        predicted_target = int(model.attractor_targets[best_idx])
    
    return ProvenanceTrace(
        query_hash=query_hash,
        query_matrix=query_matrix.copy(),
        retrieved_indices=retrieved_indices,
        prototype_ids=prototype_ids,
        confidence_scores=confidence_scores,
        similarity_scores=similarity_scores,
        vorticity_signature=vort_sig,
        timestamp=time.time(),
        predicted_target=predicted_target,
        attractor_matrices=attractor_matrices if attractor_matrices else None,
    )


# =============================================================================
# ERROR ATTRIBUTION — Identify which memories caused the error
# =============================================================================

def compute_error_attribution(
    predicted: int,
    actual: int,
    trace: ProvenanceTrace,
    model: 'TheoryTrueModel',
) -> Dict[int, float]:
    """
    Attribute prediction error to contributing memories.
    
    THEORY:
        Blame is proportional to:
        - How much a memory contributed (similarity × confidence)
        - How wrong the prediction was (distance between predicted and actual)
        
    A memory that was highly influential in making the wrong prediction
    gets high blame. Uninvolved memories get zero blame.
    
    Args:
        predicted: The predicted target token
        actual: The correct target token
        trace: The provenance trace from retrieval
        model: The model (for embedding distances)
        
    Returns:
        Dict mapping attractor index to blame score in [-1, 1]
        Positive blame = memory led to wrong answer
        Negative blame = memory would have helped (rare)
    """
    attribution = {}
    
    if predicted == actual:
        # No error - no blame to assign
        return attribution
    
    if not trace.retrieved_indices:
        # No memories were involved - can't assign blame
        return attribution
    
    xp = model.xp
    basis = model.basis
    
    # Compute total contribution for normalization
    total_contribution = 0.0
    contributions = []
    for i, idx in enumerate(trace.retrieved_indices):
        conf = trace.confidence_scores[i] if i < len(trace.confidence_scores) else 0.0
        sim = trace.similarity_scores[i] if i < len(trace.similarity_scores) else 0.0
        contrib = conf * sim
        contributions.append(contrib)
        total_contribution += contrib
    
    if total_contribution < 1e-10:
        return attribution
    
    # Compute error magnitude
    # Simple approach: if tokens differ, there IS an error
    # Scale by embedding distance for magnitude
    predicted_emb = model.embeddings[predicted % model.vocab_size]
    actual_emb = model.embeddings[actual % model.vocab_size]
    sim = frobenius_similarity(predicted_emb, actual_emb, xp)
    
    # THEORY-TRUE: Error magnitude uses φ-derived weights
    # Base error is φ⁻¹ for wrong prediction, scaled by embedding distance
    from .constants import PHI_INV, PHI_INV_SQ, PHI_INV_CUBE
    base_error = PHI_INV if predicted != actual else 0.0  # 0.618 for wrong
    embedding_distance = max(0.0, 1.0 - float(sim))
    error_magnitude = base_error + embedding_distance * PHI_INV_SQ  # Scale by φ⁻²
    error_magnitude = max(PHI_INV_CUBE, min(1.0, error_magnitude))  # Min φ⁻³ if wrong
    
    # Assign blame proportional to contribution
    for i, idx in enumerate(trace.retrieved_indices):
        contribution_frac = contributions[i] / total_contribution
        
        # Check if this memory pointed to the wrong answer
        if idx < model.num_attractors:
            target = int(model.attractor_targets[idx])
            if target == predicted:
                # This memory directly caused the wrong prediction
                blame = contribution_frac * error_magnitude
            elif target == actual:
                # This memory would have helped (was overridden) - credit (negative blame)
                blame = -contribution_frac * error_magnitude * PHI_INV  # Credit = φ⁻¹ × error
            else:
                # This memory pointed somewhere else - partial blame
                blame = contribution_frac * error_magnitude * PHI_INV_SQ  # φ⁻² × error
        else:
            blame = contribution_frac * error_magnitude
        
        attribution[idx] = blame
    
    return attribution


# =============================================================================
# RECONSOLIDATION — Update memories based on error attribution
# =============================================================================

def reconsolidate_on_error(
    error_attribution: Dict[int, float],
    correct_target: int,
    model: 'TheoryTrueModel',
    dreaming: Optional['DreamingSystem'] = None,
    min_blame_threshold: float = PHI_INV_CUBE * PHI_INV_CUBE,  # φ⁻⁶ ≈ 0.056 (was 0.05)
) -> int:
    """
    Selectively update memories that caused an error.
    
    THEORY:
        Instead of blindly overwriting, we:
        1. Identify high-blame memories
        2. Lerp them toward the correct answer
        3. Amount of lerp proportional to blame
        
    This is TARGETED RECONSOLIDATION:
        - Fixes specific errors
        - Doesn't damage unrelated memories
        - Respects φ-derived learning rates
        
    Args:
        error_attribution: Dict from compute_error_attribution
        correct_target: The correct target token
        model: The model to update
        dreaming: Optional DreamingSystem to update prototypes
        min_blame_threshold: Minimum blame to trigger update
        
    Returns:
        Number of memories updated
    """
    if not error_attribution:
        return 0
    
    xp = model.xp
    basis = model.basis
    
    # Get correct target embedding
    correct_embedding = model.embeddings[correct_target % model.vocab_size]
    
    n_updated = 0
    
    for idx, blame in error_attribution.items():
        if abs(blame) < min_blame_threshold:
            continue
        
        if idx < 0 or idx >= model.num_attractors:
            continue
        
        # Compute learning rate based on blame
        # High blame → more update (up to φ⁻¹)
        # Low blame → less update (down to φ⁻³)
        base_rate = PHI_INV
        blame_factor = min(1.0, abs(blame) / 1.0)
        learning_rate = base_rate * blame_factor + PHI_INV_CUBE * (1 - blame_factor)
        
        # Get current attractor
        current_attractor = model.attractor_matrices[idx].copy()
        
        if blame > 0:
            # Memory led to wrong answer — move toward correct
            new_attractor = (1 - learning_rate) * current_attractor + learning_rate * correct_embedding
        else:
            # Memory would have helped — slightly reinforce it
            # (This is rare and the update is smaller)
            reinforce_rate = abs(learning_rate) * 0.1
            new_attractor = (1 - reinforce_rate) * current_attractor + reinforce_rate * correct_embedding
        
        # Apply Grace to maintain stability
        new_attractor = grace_operator(new_attractor, basis, xp)
        
        # Update the model
        model.attractor_matrices[idx] = new_attractor
        model.attractor_targets[idx] = correct_target
        # v4.13.0: Removed access_counts, last_access tracking (not theory-true)
        
        n_updated += 1
    
    # Optionally update semantic memory prototypes
    if dreaming is not None and hasattr(dreaming, 'semantic_memory'):
        # Mark affected prototypes for reconsolidation during next sleep
        for proto_id in error_attribution.get('prototype_ids', []):
            if hasattr(dreaming.semantic_memory, 'mark_for_reconsolidation'):
                dreaming.semantic_memory.mark_for_reconsolidation(proto_id)
    
    return n_updated


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_blame_distribution(
    traces: List[ProvenanceTrace],
    errors: List[Tuple[int, int]],  # (predicted, actual) pairs
    model: 'TheoryTrueModel',
) -> Dict[int, float]:
    """
    Compute cumulative blame across multiple errors.
    
    Useful for identifying persistently problematic memories.
    
    Args:
        traces: List of provenance traces
        errors: Corresponding (predicted, actual) pairs
        model: The model
        
    Returns:
        Dict mapping attractor index to cumulative blame
    """
    cumulative_blame = {}
    
    for trace, (predicted, actual) in zip(traces, errors):
        attribution = compute_error_attribution(predicted, actual, trace, model)
        for idx, blame in attribution.items():
            cumulative_blame[idx] = cumulative_blame.get(idx, 0.0) + blame
    
    return cumulative_blame


def identify_problematic_memories(
    model: 'TheoryTrueModel',
    cumulative_blame: Dict[int, float],
    threshold: float = 1.0,
) -> List[int]:
    """
    Identify memories that are persistently problematic.
    
    Args:
        model: The model
        cumulative_blame: From compute_blame_distribution
        threshold: Cumulative blame threshold
        
    Returns:
        List of indices of problematic memories
    """
    problematic = [
        idx for idx, blame in cumulative_blame.items()
        if blame > threshold and idx < model.num_attractors
    ]
    return sorted(problematic, key=lambda x: cumulative_blame[x], reverse=True)


def selective_forgetting(
    model: 'TheoryTrueModel',
    indices_to_forget: List[int],
) -> int:
    """
    Selectively forget specific memories.
    
    Use this for persistently problematic memories that
    reconsolidation can't fix.
    
    Args:
        model: The model
        indices_to_forget: Indices to remove
        
    Returns:
        Number of memories forgotten
    """
    if not indices_to_forget:
        return 0
    
    # Mark for removal by setting salience to 0
    # The next equilibrium forgetting pass will clean them up
    n_forgotten = 0
    for idx in indices_to_forget:
        if 0 <= idx < model.num_attractors:
            if hasattr(model, 'attractor_saliences'):
                model.attractor_saliences[idx] = 0.0
            n_forgotten += 1
    
    return n_forgotten


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_error_correction_step(
    model: 'TheoryTrueModel',
    dreaming: Optional['DreamingSystem'] = None,
    tracker: Optional[ProvenanceTracker] = None,
) -> callable:
    """
    Create a wrapped train_step that does error correction.
    
    Returns a function that:
    1. Predicts using current model
    2. If wrong, traces and reconsolidates
    3. Then trains normally
    
    Args:
        model: The model
        dreaming: Optional DreamingSystem
        tracker: Optional ProvenanceTracker (created if not provided)
        
    Returns:
        Callable that takes (context, target) and returns train info
    """
    if tracker is None:
        tracker = ProvenanceTracker()
    
    def error_correcting_train_step(context: List[int], target: int) -> Dict[str, Any]:
        # First, predict
        predicted = model.predict(context)
        
        # If wrong, do credit assignment
        n_reconsolidated = 0
        if predicted != target:
            query_matrix = model.compute_context(context)
            trace = tracker.record_retrieval(query_matrix, model, dreaming)
            attribution = compute_error_attribution(predicted, target, trace, model)
            n_reconsolidated = reconsolidate_on_error(attribution, target, model, dreaming)
        
        # Then train normally
        train_info = model.train_step(context, target)
        train_info['prediction_before'] = predicted
        train_info['was_correct'] = predicted == target
        train_info['n_reconsolidated'] = n_reconsolidated
        
        return train_info
    
    return error_correcting_train_step
