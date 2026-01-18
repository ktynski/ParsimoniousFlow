"""
Pseudo-Rehearsal — Prevent Catastrophic Forgetting

Implements theory-true pseudo-rehearsal:
- Semantic memory (neocortex) generates "pseudo-patterns"
- These are mixed with real episodes during training
- Old knowledge is rehearsed while new is learned

THEORY (Complementary Learning Systems):
    Without rehearsal:
        - New learning overwrites old
        - Model "forgets" previously learned patterns
        
    With pseudo-rehearsal:
        - Generate samples from semantic memory
        - Mix with real episodes
        - Old patterns are reinforced while learning new
"""

import numpy as np
from typing import List, Tuple, Dict, Any, TYPE_CHECKING

from holographic_prod.core.constants import PHI_INV_SQ, PHI_INV_CUBE, DTYPE
from .structures import EpisodicEntry, SemanticPrototype
from .priority import compute_salience

if TYPE_CHECKING:
    from .semantic_memory import SemanticMemory


# =============================================================================
# PSEUDO-EPISODE GENERATION
# =============================================================================

def generate_pseudo_episode(
    prototype: SemanticPrototype,
    basis: np.ndarray,
    xp = np,
    noise_std: float = PHI_INV_CUBE,  # φ-derived (was 0.1)
) -> EpisodicEntry:
    """
    Generate a pseudo-episode from a semantic prototype.
    
    THEORY (Complementary Learning Systems):
        To prevent catastrophic forgetting:
        1. Semantic memory (neocortex) generates "pseudo-patterns"
        2. These are mixed with real episodes during training
        3. Old knowledge is rehearsed while new is learned
        
    The pseudo-episode is:
        - Prototype matrix + small noise (variation around the core)
        - Target sampled from the prototype's distribution
        
    This is like "dreaming" during waking - generating internal samples
    that remind the system of what it already knows.
    
    Args:
        prototype: SemanticPrototype to generate from
        basis: Clifford basis
        xp: numpy or cupy
        noise_std: Standard deviation of noise to add
        
    Returns:
        EpisodicEntry representing a pseudo-episode
    """
    # Generate context: prototype + noise
    noise = noise_std * xp.random.randn(4, 4)
    pseudo_context = prototype.prototype_matrix + noise
    
    # Normalize
    norm = xp.linalg.norm(pseudo_context, 'fro')
    if norm > 1e-8:
        pseudo_context = pseudo_context / norm
    
    # Sample target from distribution - USE NUMPY (CuPy doesn't support p parameter)
    targets = list(prototype.target_distribution.keys())
    probs = np.array(list(prototype.target_distribution.values()))
    probs = probs / probs.sum()  # Ensure sums to 1
    target = int(np.random.choice(targets, p=probs))
    
    # Compute salience of pseudo-episode
    salience = compute_salience(pseudo_context, basis, xp)
    
    return EpisodicEntry(
        context_matrix=pseudo_context,
        target_token=target,
        count=1,
        recency=0.0,
        salience=salience,
        novelty=0.0,  # Not novel - generated from known prototype
        priority=salience * PHI_INV_SQ,  # φ²-reduced priority for synthetic episodes
    )


def generate_pseudo_episodes_batch(
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
    n_episodes: int = 10,
    noise_std: float = PHI_INV_CUBE,  # φ-derived noise for variation
) -> List[EpisodicEntry]:
    """
    Generate a batch of pseudo-episodes from semantic memory.
    
    THEORY-TRUE: Prototypes sampled weighted by support (evidence count).
    No temperature parameter - support weighting is theory-derived.
    
    Args:
        semantic_memory: SemanticMemory to generate from
        basis: Clifford basis
        xp: numpy or cupy
        n_episodes: Number of pseudo-episodes to generate
        noise_std: φ-derived noise for variation (PHI_INV_CUBE ≈ 0.236)
        
    Returns:
        List of EpisodicEntry pseudo-episodes
    """
    # Collect all prototypes
    all_protos = []
    for level in semantic_memory.levels:
        all_protos.extend(level)
    
    if not all_protos:
        return []
    
    # THEORY-TRUE: Weight by support directly (no arbitrary softmax)
    # Support = evidence count = naturally positive
    # Common patterns (high support) are rehearsed more
    supports = xp.array([p.support for p in all_protos], dtype=DTYPE)
    
    # Directly normalize to probability (support is already positive)
    probs = supports / (xp.sum(supports) + 1e-10)
    
    # Sample prototype indices - USE NUMPY (CuPy doesn't support p parameter)
    n_actual = min(n_episodes, len(all_protos) * 5)  # Don't over-sample
    # CuPy arrays need .get() for explicit conversion
    if hasattr(probs, 'get'):
        probs_np = probs.get()  # CuPy → NumPy
    else:
        probs_np = np.array(probs)
    probs_np = probs_np / probs_np.sum()  # Ensure sums to 1
    indices = np.random.choice(
        len(all_protos),
        size=n_actual,
        replace=True,  # Allow repeats
        p=probs_np,
    )
    
    # Generate pseudo-episodes
    pseudo_episodes = []
    for idx in indices:
        proto = all_protos[int(idx)]
        pseudo_ep = generate_pseudo_episode(proto, basis, xp, noise_std)
        pseudo_episodes.append(pseudo_ep)
    
    return pseudo_episodes


def interleave_with_pseudo_rehearsal(
    real_episodes: List[EpisodicEntry],
    semantic_memory: 'SemanticMemory',
    basis: np.ndarray,
    xp = np,
    rehearsal_ratio: float = PHI_INV_CUBE,  # Theory-true: φ⁻³ ≈ 0.236
    noise_std: float = PHI_INV_CUBE,  # φ-derived (was 0.1)
) -> Tuple[List[EpisodicEntry], Dict[str, Any]]:
    """
    Interleave real episodes with pseudo-episodes from semantic memory.
    
    THEORY (Preventing Catastrophic Forgetting):
        Without rehearsal:
            - New learning overwrites old
            - Model "forgets" previously learned patterns
            
        With pseudo-rehearsal:
            - Generate samples from semantic memory
            - Mix with real episodes
            - Old patterns are reinforced while learning new
            
    The ratio controls the trade-off:
        - High ratio (0.5): More rehearsal, stronger memory retention
        - Low ratio (0.1): Less rehearsal, faster new learning
        
    Args:
        real_episodes: Real episodes to learn from
        semantic_memory: SemanticMemory for generating pseudo-episodes
        basis: Clifford basis
        xp: numpy or cupy
        rehearsal_ratio: Fraction of pseudo-episodes to add (0.2 = 20%)
        noise_std: Noise for pseudo-episode generation
        
    Returns:
        combined: Interleaved list of real + pseudo episodes
        stats: Rehearsal statistics
    """
    n_real = len(real_episodes)
    n_pseudo = int(n_real * rehearsal_ratio)
    
    # Generate pseudo-episodes
    pseudo_episodes = generate_pseudo_episodes_batch(
        semantic_memory, basis, xp,
        n_episodes=n_pseudo,
        noise_std=noise_std,
    )
    
    # Combine and shuffle
    combined = list(real_episodes) + pseudo_episodes
    
    # Shuffle to interleave - USE NUMPY for permutation (safer)
    indices = np.random.permutation(len(combined))
    combined = [combined[int(i)] for i in indices]
    
    stats = {
        'real_episodes': n_real,
        'pseudo_episodes': len(pseudo_episodes),
        'total': len(combined),
        'actual_ratio': len(pseudo_episodes) / max(n_real, 1),
    }
    
    return combined, stats
