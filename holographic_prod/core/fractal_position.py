"""
Fractal Positional Encoding — φ-Derived Multi-Scale Position
=============================================================

THEORY:
    Language has hierarchical structure at multiple scales:
        Word → Phrase → Clause → Sentence → Paragraph
    
    Each scale needs positional information, and they should be
    SELF-SIMILAR (same pattern at each scale).
    
    The golden ratio φ is the UNIQUE solution to self-consistency:
        φ² = φ + 1
    
    This creates a fractal structure where:
        - Scale 0 (finest): phase = position × 2π/φ⁰
        - Scale 1: phase = position × 2π/φ¹
        - Scale 2: phase = position × 2π/φ²
        - etc.
    
    The phases are INCOMMENSURATE (never repeat exactly), creating
    a unique "fingerprint" for every position at every scale.

BRAIN ANALOG:
    - Grid cells fire at multiple scales (like our multi-scale phases)
    - Theta oscillations (7-12 Hz) nest within gamma (30-100 Hz)
    - Broca's processes syntax at multiple granularities

THEORY-TRUE:
    - Uses ONLY φ-derived constants
    - No learned positional embeddings
    - Self-similar structure at all scales
    - Composable via geometric product

Version: v5.18.0
"""

import numpy as np
from typing import Optional, Tuple
from scipy.stats import ortho_group

from .constants import PHI, PHI_INV, PHI_INV_SQ


def golden_angle(scale: int = 0) -> float:
    """
    Compute the golden angle at a given scale.
    
    Scale 0: 2π/φ⁰ = 2π ≈ 360° (full rotation per position)
    Scale 1: 2π/φ¹ ≈ 222.5° (golden angle)
    Scale 2: 2π/φ² ≈ 137.5° (classic golden angle)
    Scale 3: 2π/φ³ ≈ 85.0°
    etc.
    
    The golden angle (137.5°) is famous in phyllotaxis (leaf arrangement)
    because it creates optimal packing with NO repeating patterns.
    """
    return 2 * np.pi / (PHI ** scale)


def create_so4_rotation_from_angles(
    theta1: float, 
    theta2: float,
    xp = np
) -> np.ndarray:
    """
    Create an SO(4) rotation matrix from two angles.
    
    SO(4) has 6 degrees of freedom, but we use 2 angles
    corresponding to the two independent rotation planes
    in 4D (like Clifford torus structure).
    
    This is a double rotation: one in the (x,y) plane,
    one in the (z,w) plane.
    """
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    
    # Block diagonal SO(4) rotation
    rotation = xp.array([
        [c1, -s1, 0, 0],
        [s1, c1, 0, 0],
        [0, 0, c2, -s2],
        [0, 0, s2, c2]
    ], dtype=np.float32)
    
    return rotation


def fractal_position_rotation(
    position: int,
    n_scales: int = 4,
    xp = np
) -> np.ndarray:
    """
    Compute the fractal positional rotation matrix.
    
    Composes rotations from multiple scales:
        R_total = R_0 @ R_1 @ R_2 @ ... @ R_{n-1}
    
    where R_k uses golden angle at scale k.
    
    Args:
        position: Position in sequence (0, 1, 2, ...)
        n_scales: Number of scales to compose (default: 4)
        xp: Array module (numpy or cupy)
    
    Returns:
        4x4 SO(4) rotation matrix encoding position fractally
    """
    # Start with identity
    R_total = xp.eye(4, dtype=np.float32)
    
    for scale in range(n_scales):
        # Golden angle at this scale
        angle = position * golden_angle(scale)
        
        # Use different rotation planes for different scales
        # This ensures scales don't interfere
        if scale % 2 == 0:
            theta1, theta2 = angle, angle * PHI_INV
        else:
            theta1, theta2 = angle * PHI_INV, angle
        
        R_scale = create_so4_rotation_from_angles(theta1, theta2, xp)
        R_total = R_total @ R_scale
    
    return R_total


def encode_position_fractal(
    token_embedding: np.ndarray,
    position: int,
    n_scales: int = 4,
    xp = np
) -> np.ndarray:
    """
    Apply fractal positional encoding to a token embedding.
    
    The embedding is ROTATED by the fractal position rotation.
    This is theory-true because:
        - SO(4) rotation preserves the Clifford structure
        - Different positions create different orientations
        - The encoding is deterministic (no learned params)
    
    Args:
        token_embedding: 4x4 SO(4) token embedding
        position: Position in sequence
        n_scales: Number of fractal scales
        xp: Array module
    
    Returns:
        Position-encoded embedding (still 4x4 SO(4))
    """
    R = fractal_position_rotation(position, n_scales, xp)
    R_inv = R.T  # SO(4): inverse = transpose
    
    # Conjugate the embedding by the rotation
    return R @ token_embedding @ R_inv


def encode_sequence_fractal(
    token_embeddings: np.ndarray,
    n_scales: int = 4,
    xp = np
) -> np.ndarray:
    """
    Apply fractal positional encoding to a sequence of embeddings.
    
    Args:
        token_embeddings: [seq_len, 4, 4] array of embeddings
        n_scales: Number of fractal scales
        xp: Array module
    
    Returns:
        Position-encoded embeddings [seq_len, 4, 4]
    """
    seq_len = token_embeddings.shape[0]
    encoded = xp.zeros_like(token_embeddings)
    
    for i in range(seq_len):
        encoded[i] = encode_position_fractal(
            token_embeddings[i], i, n_scales, xp
        )
    
    return encoded


def encode_sequence_fractal_vectorized(
    token_embeddings: np.ndarray,
    n_scales: int = 4,
    xp = np
) -> np.ndarray:
    """
    VECTORIZED fractal positional encoding.
    
    Precomputes all rotation matrices for efficiency.
    
    Args:
        token_embeddings: [seq_len, 4, 4] array of embeddings
        n_scales: Number of fractal scales
        xp: Array module
    
    Returns:
        Position-encoded embeddings [seq_len, 4, 4]
    """
    seq_len = token_embeddings.shape[0]
    
    # Precompute all rotations
    rotations = xp.stack([
        fractal_position_rotation(i, n_scales, xp) 
        for i in range(seq_len)
    ])  # [seq_len, 4, 4]
    
    rotations_inv = xp.swapaxes(rotations, -2, -1)  # Transpose each
    
    # Batched conjugation: R @ emb @ R^T
    # Using einsum for efficiency
    encoded = xp.einsum(
        'nij,njk,nkl->nil',
        rotations, token_embeddings, rotations_inv
    )
    
    return encoded


# =============================================================================
# HIERARCHICAL POSITION (for nested structures)
# =============================================================================

def hierarchical_position_key(
    word_pos: int,
    phrase_pos: int = 0,
    clause_pos: int = 0,
    sentence_pos: int = 0,
    n_scales: int = 4,
    xp = np
) -> np.ndarray:
    """
    Create a hierarchical position encoding.
    
    Instead of just word position, encodes position at multiple
    linguistic levels:
        - word_pos: Position within phrase
        - phrase_pos: Position within clause
        - clause_pos: Position within sentence
        - sentence_pos: Position within document
    
    Each level uses a different scale of the golden angle.
    
    Example:
        "The dog that bit the man ran away"
        
        word_pos:     0   1    2    3   4   5   6   7
        phrase_pos:   0   0    1    1   1   1   2   2
        clause_pos:   0   0    1    1   1   1   0   0
        sentence_pos: 0   0    0    0   0   0   0   0
    
    Args:
        word_pos: Position within phrase (0-indexed)
        phrase_pos: Which phrase (0-indexed)
        clause_pos: Which clause (0-indexed)
        sentence_pos: Which sentence (0-indexed)
        n_scales: Number of scales per level
        xp: Array module
    
    Returns:
        4x4 SO(4) hierarchical position rotation
    """
    R_total = xp.eye(4, dtype=np.float32)
    
    # Each level contributes at its own scale
    levels = [word_pos, phrase_pos, clause_pos, sentence_pos]
    
    for level_idx, pos in enumerate(levels):
        # Base scale for this level: φ^level_idx
        base_scale = level_idx
        
        for sub_scale in range(n_scales):
            angle = pos * golden_angle(base_scale + sub_scale)
            
            # Alternate rotation planes
            if (level_idx + sub_scale) % 2 == 0:
                theta1, theta2 = angle, angle * PHI_INV
            else:
                theta1, theta2 = angle * PHI_INV, angle
            
            R_scale = create_so4_rotation_from_angles(theta1, theta2, xp)
            R_total = R_total @ R_scale
    
    return R_total


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def position_distance(pos1: int, pos2: int, n_scales: int = 4) -> float:
    """
    Compute the distance between two positions in fractal space.
    
    Uses Frobenius norm of the difference between rotation matrices.
    """
    R1 = fractal_position_rotation(pos1, n_scales)
    R2 = fractal_position_rotation(pos2, n_scales)
    
    return float(np.linalg.norm(R1 - R2))


def position_correlation(pos1: int, pos2: int, n_scales: int = 4) -> float:
    """
    Compute correlation between two position encodings.
    
    Low correlation = well separated = good.
    """
    R1 = fractal_position_rotation(pos1, n_scales)
    R2 = fractal_position_rotation(pos2, n_scales)
    
    # Frobenius inner product normalized
    return float(np.sum(R1 * R2)) / (np.linalg.norm(R1) * np.linalg.norm(R2))


def analyze_position_coverage(max_pos: int = 64, n_scales: int = 4) -> dict:
    """
    Analyze how well fractal positions cover the space.
    
    Returns statistics about position separation.
    """
    # Compute all pairwise distances
    distances = []
    correlations = []
    
    for i in range(max_pos):
        for j in range(i + 1, max_pos):
            distances.append(position_distance(i, j, n_scales))
            correlations.append(position_correlation(i, j, n_scales))
    
    return {
        'min_distance': min(distances),
        'max_distance': max(distances),
        'mean_distance': np.mean(distances),
        'min_correlation': min(correlations),
        'max_correlation': max(correlations),
        'mean_correlation': np.mean(correlations),
        'n_positions': max_pos,
        'n_scales': n_scales,
    }
