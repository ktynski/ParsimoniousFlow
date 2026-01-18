"""
Grounding Module — Perception to Clifford Mapping
=================================================

This module connects the abstract Clifford space to perceptual features.

THEORY:
    The Clifford algebra operates on 4×4 matrices that represent meaning.
    But meaning must be grounded in perception to be useful.
    
    Grounding = learn a mapping from feature vectors to Clifford matrices
    that preserves relevant structure (similar features → similar matrices).

KEY INSIGHT:
    This is the ONE place where we might need gradient-based learning,
    OR we use structure-preserving projection:
    
    features ∈ ℝ^n → projection → 4×4 matrix → Grace normalization
"""

import numpy as np
from typing import Dict, Optional

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ
from holographic_v4.algebra import (
    build_clifford_basis,
    grace_operator,
    frobenius_similarity,
    decompose_to_coefficients,
    reconstruct_from_coefficients,
)
from holographic_v4.quotient import grace_stability

Array = np.ndarray
ArrayModule = type(np)


class PerceptionEncoder:
    """
    Maps perceptual features to Clifford representations.
    
    THEORY:
        Uses a learned projection from feature space to the 16-dimensional
        Clifford coefficient space, then reconstructs as a 4×4 matrix.
        
        The projection is initialized to preserve rough distance structure
        and can be updated via feedback.
    """
    
    def __init__(
        self,
        feature_dim: int = 16,
        basis: Optional[Array] = None,
        xp: ArrayModule = np,
        seed: int = 42,
    ):
        """
        Initialize the encoder.
        
        Args:
            feature_dim: Dimension of input feature vectors
            basis: Clifford basis (built if not provided)
            xp: Array module
            seed: Random seed
        """
        self.feature_dim = feature_dim
        self.basis = basis if basis is not None else build_clifford_basis(xp)
        self.xp = xp
        
        # Initialize projection matrix (feature_dim → 16)
        rng = np.random.default_rng(seed)
        
        # Random orthogonal-ish projection
        raw = rng.standard_normal((16, feature_dim))
        # Normalize rows to unit norm for stability
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self.projection = raw / (norms + 1e-10)
        
        # Bias toward scalar component (stability)
        self.projection[0] *= 2.0
        
        # Learning rate for feedback updates
        self.learning_rate = PHI_INV_SQ
    
    def encode_features(self, features: Array) -> Array:
        """
        Map feature vector to 4×4 Clifford matrix.
        
        Args:
            features: [feature_dim] feature vector
            
        Returns:
            [4, 4] Clifford matrix
        """
        xp = self.xp
        
        # Ensure features are the right shape
        features = xp.asarray(features).flatten()
        if len(features) != self.feature_dim:
            # Pad or truncate
            if len(features) < self.feature_dim:
                features = xp.concatenate([features, xp.zeros(self.feature_dim - len(features))])
            else:
                features = features[:self.feature_dim]
        
        # Project to 16D Clifford coefficient space
        coeffs = xp.dot(self.projection, features)
        
        # Reconstruct as 4×4 matrix
        matrix = reconstruct_from_coefficients(coeffs, self.basis, xp)
        
        # Grace-normalize for stability
        matrix = grace_operator(matrix, self.basis, xp)
        
        return matrix
    
    def update_from_feedback(
        self,
        features: Array,
        desired_clifford: Array,
        success: bool,
    ) -> None:
        """
        Adjust encoding based on downstream success.
        
        THEORY:
            If retrieval succeeded, strengthen the current mapping.
            If retrieval failed, adjust toward the desired Clifford representation.
            
        Args:
            features: The input features
            desired_clifford: What we wanted to get
            success: Whether retrieval succeeded
        """
        xp = self.xp
        
        features = xp.asarray(features).flatten()
        if len(features) != self.feature_dim:
            if len(features) < self.feature_dim:
                features = xp.concatenate([features, xp.zeros(self.feature_dim - len(features))])
            else:
                features = features[:self.feature_dim]
        
        # Get current encoding
        current_clifford = self.encode_features(features)
        
        # Get desired coefficients
        desired_coeffs = decompose_to_coefficients(desired_clifford, self.basis, xp)
        current_coeffs = decompose_to_coefficients(current_clifford, self.basis, xp)
        
        # Compute coefficient error
        error = desired_coeffs - current_coeffs
        
        # Update projection to reduce error
        if success:
            # Small reinforcement
            rate = self.learning_rate * 0.1
        else:
            # Larger correction
            rate = self.learning_rate
        
        # Gradient: error ⊗ features
        # Each row of projection is updated by: rate * error[i] * features
        for i in range(16):
            self.projection[i] += rate * error[i] * features


def ground_token(
    token_id: int,
    perceptual_features: Array,
    encoder: PerceptionEncoder,
    model: 'TheoryTrueModel',
) -> None:
    """
    Associate token with perceptual grounding.
    
    Updates the token's embedding to incorporate perceptual structure.
    
    Args:
        token_id: Token to ground
        perceptual_features: Features describing the token's perceptual meaning
        encoder: PerceptionEncoder to use
        model: Model whose embeddings to update
    """
    xp = model.xp
    basis = model.basis
    
    # Encode features to Clifford matrix
    perceptual_clifford = encoder.encode_features(perceptual_features)
    
    # Get current embedding
    current_embedding = model.embeddings[token_id % model.vocab_size]
    
    # Blend: move embedding toward perceptual grounding
    blend_rate = PHI_INV  # Moderate blending
    new_embedding = (1 - blend_rate) * current_embedding + blend_rate * perceptual_clifford
    
    # Grace-normalize
    new_embedding = grace_operator(new_embedding, basis, xp)
    
    # Preserve norm
    old_norm = xp.linalg.norm(current_embedding)
    new_norm = xp.linalg.norm(new_embedding)
    if new_norm > 1e-10:
        new_embedding = new_embedding * (old_norm / new_norm)
    
    # Update model
    model.embeddings[token_id % model.vocab_size] = new_embedding


def perceptual_similarity(
    features1: Array,
    features2: Array,
    encoder: PerceptionEncoder,
) -> float:
    """
    Compute similarity in Clifford space after encoding.
    
    Should correlate with perceptual similarity.
    
    Args:
        features1: First feature vector
        features2: Second feature vector
        encoder: PerceptionEncoder
        
    Returns:
        Similarity score
    """
    clifford1 = encoder.encode_features(features1)
    clifford2 = encoder.encode_features(features2)
    
    return float(frobenius_similarity(clifford1, clifford2, encoder.xp))


def batch_encode(
    features_batch: Array,
    encoder: PerceptionEncoder,
) -> Array:
    """
    Encode a batch of feature vectors.
    
    Args:
        features_batch: [N, feature_dim] batch of features
        encoder: PerceptionEncoder
        
    Returns:
        [N, 4, 4] batch of Clifford matrices
    """
    xp = encoder.xp
    N = features_batch.shape[0]
    
    result = xp.zeros((N, 4, 4), dtype=xp.float64)
    for i in range(N):
        result[i] = encoder.encode_features(features_batch[i])
    
    return result
