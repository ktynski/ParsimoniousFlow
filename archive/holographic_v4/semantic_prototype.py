"""
Position-Weighted Semantic Prototypes
=====================================

THEORY:
    The composed context matrix mixes signal (semantic tokens) and noise
    (surface tokens). Full matrix similarity fails because noise dominates.
    
    SOLUTION: Compare contexts by position-wise embedding similarity,
    weighting positions by their semantic importance.
    
INSIGHT:
    - Semantic positions have CONSISTENT embeddings for same target
    - Noise positions have VARYING embeddings for same target
    - Weight by inverse variance → automatically detect semantic positions

BRAIN ANALOG:
    Like hippocampal pattern separation: identify which features are
    diagnostic for distinguishing concepts, weight those higher.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_CUBE
from holographic_v4.algebra import build_clifford_basis


Array = np.ndarray


@dataclass
class PositionWisePrototype:
    """
    Prototype that stores per-position embedding averages.
    
    Instead of one composed 4x4 matrix, stores a list of
    context_size 4x4 matrices (one per position).
    """
    position_embeddings: List[Array]  # [context_size] of [4, 4] matrices
    target_distribution: Dict[int, float]
    support: int
    position_variances: Optional[Array] = None  # [context_size] variance per position
    
    def mode_target(self) -> int:
        """Return most likely target."""
        if not self.target_distribution:
            return -1
        return max(self.target_distribution.items(), key=lambda x: x[1])[0]


def embedding_cosine_similarity(a: Array, b: Array) -> float:
    """Cosine similarity between two 4x4 matrices (as flattened vectors)."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-10))


def position_weighted_similarity(
    query_embeddings: List[Array],
    proto_embeddings: List[Array],
    weights: Optional[Array] = None,
) -> float:
    """
    Compute weighted sum of per-position similarities.
    
    Args:
        query_embeddings: List of [4, 4] embeddings for query
        proto_embeddings: List of [4, 4] prototype embeddings
        weights: Optional [context_size] weights (default: uniform)
        
    Returns:
        Weighted similarity score
    """
    if len(query_embeddings) != len(proto_embeddings):
        raise ValueError("Query and prototype must have same context size")
    
    n_positions = len(query_embeddings)
    if weights is None:
        weights = np.ones(n_positions)
    
    weights = weights / (weights.sum() + 1e-10)
    
    total_sim = 0.0
    for i, (q, p) in enumerate(zip(query_embeddings, proto_embeddings)):
        sim = embedding_cosine_similarity(q, p)
        total_sim += weights[i] * sim
    
    return total_sim


def compute_position_weights_from_variance(
    position_variances: Array,
) -> Array:
    """
    Compute position weights from variance: low variance → high weight.
    
    THEORY:
        Positions with low variance across same-target samples are
        semantic (consistent). High variance = noise (varying).
        
    THEORY-TRUE: Uses φ-kernel (NOT softmax!)
        weight[i] = φ^(1 - normalized_variance[i])
        
        Why φ-kernel instead of softmax:
        - φ is THE theory-derived scaling constant
        - φ-kernel is φ^(-distance), matching Grace operator's spectral decay
        - Softmax uses e^x for convexity reasons (gradient descent optimization)
        - We don't use gradient descent, so softmax's convexity is irrelevant
        
    Args:
        position_variances: [context_size] variance per position
        
    Returns:
        [context_size] normalized weights
        
    NOTE: The temperature parameter was REMOVED as it's not theory-derived.
    The φ-kernel naturally gives φ:1 ratio, which is sufficient discrimination.
    """
    from .constants import PHI
    
    # Normalize variance to [0, 1] for stability
    var_max = position_variances.max()
    if var_max > 1e-10:
        normalized_var = position_variances / var_max
    else:
        # All variances near zero - uniform weights
        return np.ones_like(position_variances) / len(position_variances)
    
    # THEORY-TRUE φ-kernel: low variance → high weight
    # weight = φ^(1 - normalized_var)
    # When var=0: weight = φ (high)
    # When var=max: weight = 1 (low)
    # Ratio is exactly φ:1 - the fundamental ratio
    weights = PHI ** (1.0 - normalized_var)
    
    # Normalize to sum to 1
    weights = weights / (weights.sum() + 1e-10)
    return weights


class SemanticPrototypeMemory:
    """
    Memory that stores position-wise prototypes with learned weights.
    
    OPERATIONS:
        1. add_episode(embeddings, target) - Store episode
        2. consolidate() - Create prototypes from episodes by target
        3. retrieve(query_embeddings) - Find best matching prototype
        4. update_weights_from_error(query, predicted, actual) - Learn weights
    """
    
    def __init__(
        self,
        context_size: int = 8,
        initial_weights: Optional[Array] = None,
        learning_rate: float = PHI_INV_CUBE,  # φ-derived (was 0.1)
    ):
        self.context_size = context_size
        
        # Position weights (learnable)
        if initial_weights is None:
            self.position_weights = np.ones(context_size)
        else:
            self.position_weights = initial_weights.copy()
        
        self.learning_rate = learning_rate
        
        # Episode buffer
        self.episodes: List[Tuple[List[Array], int]] = []  # (embeddings, target)
        
        # Prototypes (after consolidation)
        self.prototypes: Dict[int, PositionWisePrototype] = {}
        
        # Statistics
        self.total_retrievals = 0
        self.correct_retrievals = 0
    
    def add_episode(self, embeddings: List[Array], target: int) -> None:
        """Add an episode to the buffer."""
        if len(embeddings) != self.context_size:
            raise ValueError(f"Expected {self.context_size} embeddings, got {len(embeddings)}")
        self.episodes.append((embeddings, target))
    
    def consolidate(self) -> Dict[str, Any]:
        """
        Create prototypes from episodes grouped by target.
        
        THEORY:
            - Group episodes by target (supervised signal)
            - Average embeddings per position
            - Compute variance per position (for weight learning)
            
        Returns:
            Statistics about consolidation
        """
        # Group by target
        target_groups = defaultdict(list)
        for embeddings, target in self.episodes:
            target_groups[target].append(embeddings)
        
        self.prototypes = {}
        
        for target, embs_list in target_groups.items():
            n_episodes = len(embs_list)
            
            # Compute per-position mean and variance
            position_embeddings = []
            position_variances = []
            
            for pos in range(self.context_size):
                # Stack all embeddings at this position: [n_episodes, 4, 4]
                pos_embs = np.stack([embs[pos] for embs in embs_list])
                
                # Mean embedding at this position
                mean_emb = pos_embs.mean(axis=0)
                position_embeddings.append(mean_emb)
                
                # Variance at this position (scalar: mean squared distance from mean)
                diffs = pos_embs - mean_emb
                variance = np.mean(diffs ** 2)
                position_variances.append(variance)
            
            self.prototypes[target] = PositionWisePrototype(
                position_embeddings=position_embeddings,
                target_distribution={target: 1.0},
                support=n_episodes,
                position_variances=np.array(position_variances),
            )
        
        # Update position weights based on average variance across prototypes
        if self.prototypes:
            all_variances = np.stack([p.position_variances for p in self.prototypes.values()])
            avg_variances = all_variances.mean(axis=0)
            learned_weights = compute_position_weights_from_variance(avg_variances)
            
            # Blend with current weights
            self.position_weights = (
                (1 - self.learning_rate) * self.position_weights +
                self.learning_rate * learned_weights
            )
            self.position_weights = self.position_weights / self.position_weights.sum()
        
        return {
            'n_prototypes': len(self.prototypes),
            'n_episodes': len(self.episodes),
            'position_weights': self.position_weights.tolist(),
        }
    
    def retrieve(
        self,
        query_embeddings: List[Array],
        use_learned_weights: bool = True,
    ) -> Tuple[int, float]:
        """
        Retrieve best matching prototype.
        
        Args:
            query_embeddings: [context_size] list of [4, 4] embeddings
            use_learned_weights: If True, use position weights
            
        Returns:
            (predicted_target, similarity_score)
        """
        if not self.prototypes:
            return -1, 0.0
        
        weights = self.position_weights if use_learned_weights else None
        
        best_target = -1
        best_sim = -2.0
        
        for target, proto in self.prototypes.items():
            sim = position_weighted_similarity(
                query_embeddings,
                proto.position_embeddings,
                weights,
            )
            if sim > best_sim:
                best_sim = sim
                best_target = target
        
        self.total_retrievals += 1
        return best_target, best_sim
    
    def update_from_feedback(
        self,
        query_embeddings: List[Array],
        predicted_target: int,
        actual_target: int,
    ) -> None:
        """
        Update position weights based on prediction error.
        
        THEORY:
            If prediction was wrong:
            - Positions where query matches ACTUAL prototype → increase weight
            - Positions where query matches PREDICTED prototype → decrease weight
        """
        if predicted_target == actual_target:
            self.correct_retrievals += 1
            return
        
        if actual_target not in self.prototypes or predicted_target not in self.prototypes:
            return
        
        actual_proto = self.prototypes[actual_target]
        predicted_proto = self.prototypes[predicted_target]
        
        # Compute per-position similarities
        actual_sims = np.array([
            embedding_cosine_similarity(q, p)
            for q, p in zip(query_embeddings, actual_proto.position_embeddings)
        ])
        predicted_sims = np.array([
            embedding_cosine_similarity(q, p)
            for q, p in zip(query_embeddings, predicted_proto.position_embeddings)
        ])
        
        # Gradient: increase weight where actual is more similar than predicted
        gradient = actual_sims - predicted_sims
        
        # Update weights
        self.position_weights += self.learning_rate * gradient
        
        # Keep positive and normalize
        self.position_weights = np.maximum(self.position_weights, 0.01)
        self.position_weights = self.position_weights / self.position_weights.sum()
    
    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        accuracy = (
            self.correct_retrievals / self.total_retrievals
            if self.total_retrievals > 0
            else 0.0
        )
        return {
            'n_prototypes': len(self.prototypes),
            'n_episodes': len(self.episodes),
            'total_retrievals': self.total_retrievals,
            'accuracy': accuracy,
            'position_weights': self.position_weights.tolist(),
        }


