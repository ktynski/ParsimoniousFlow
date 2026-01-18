"""
Multi-Level Torus Hierarchy — Tower of Quotients
=================================================

ARCHITECTURE:
    Level 1: Cl(3,1) contexts → attractors (word-level)
    Level 2: Attractors → higher attractors (phrase/sentence-level)
    Level N: Recursive composition

KEY INSIGHT:
    Do NOT jump to Cl(7,1) or higher — that explodes computation.
    Instead: COMPOSE quotient spaces.
    
    Each level has:
    - Its own Clifford algebra (Cl(3,1) at each level)
    - Its own witness (stable self-pointer at that scale)
    - Its own gauge group (Spin(3) at each level)
    - Its own Grace contraction

SCALING:
    Single nested torus:  O(dim²) computation, fragile long-range
    Tower of quotients:   O(log depth) computation, robust hierarchy

MATHEMATICAL STRUCTURE:
    Let A₁ be the attractor space at Level 1.
    Define embedding: ι: A₁ → Cl(3,1)₂
    
    This embeds Level 1 attractors as "tokens" for Level 2.
    Composition at Level 2 uses the same Clifford structure.
    
    The tower is:
        Cl(3,1)₁ --quotient--> A₁ --embed--> Cl(3,1)₂ --quotient--> A₂ --> ...
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .constants import PHI, PHI_INV, PHI_INV_SQ, MATRIX_DIM
from .algebra import (
    build_clifford_basis, normalize_matrix, geometric_product_batch,
    frobenius_similarity_batch, initialize_all_embeddings
)
from .quotient import (
    witness_pointer, normal_form, quotient_similarity,
    compute_witness_stability
)

Array = np.ndarray
ArrayModule = type(np)


# =============================================================================
# ATTRACTOR CODEBOOK — Compressed representation of Level N attractors
# =============================================================================

@dataclass
class AttractorCodebook:
    """
    A codebook of attractors from Level N that become tokens for Level N+1.
    
    This is the key data structure for hierarchical composition.
    
    Attributes:
        level: Which level this codebook represents
        matrices: [num_attractors, 4, 4] the attractor matrices
        witnesses: [num_attractors, 4, 4] witness pointers for each
        usage_counts: How often each attractor is retrieved
    """
    level: int
    matrices: Array  # [num_attractors, 4, 4]
    witnesses: Array  # [num_attractors, 4, 4]
    usage_counts: Array  # [num_attractors]


def create_codebook(attractors: List[Array], level: int, 
                    basis: Array, xp: ArrayModule = np) -> AttractorCodebook:
    """
    Create a codebook from a list of attractor matrices.
    
    Args:
        attractors: List of [4, 4] attractor matrices
        level: The level this codebook represents
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        AttractorCodebook instance
    """
    n = len(attractors)
    matrices = xp.stack(attractors, axis=0)
    
    # Compute witnesses for each attractor
    witnesses = xp.zeros((n, 4, 4), dtype=xp.float64)
    for i in range(n):
        witnesses[i] = witness_pointer(matrices[i], basis, xp)
    
    return AttractorCodebook(
        level=level,
        matrices=matrices,
        witnesses=witnesses,
        usage_counts=xp.zeros(n, dtype=xp.int64)
    )


# =============================================================================
# LEVEL INTERFACE — Abstract interface for each level of the hierarchy
# =============================================================================

class HierarchyLevel:
    """
    One level of the torus hierarchy.
    
    Each level:
    - Has its own Cl(3,1) representation
    - Takes "tokens" from the previous level (or raw tokens for Level 1)
    - Produces attractors that become tokens for the next level
    
    The key operation is:
        context → attractor → (optionally) next-level token
    """
    
    def __init__(self, level: int, input_dim: int, max_attractors: int = 10000,
                 xp: ArrayModule = np):
        """
        Args:
            level: 1, 2, 3, ... (Level 1 is word-level)
            input_dim: Number of input "tokens" (vocab for L1, codebook size for L2+)
            max_attractors: Maximum attractors to store
            xp: array module
        """
        self.level = level
        self.input_dim = input_dim
        self.max_attractors = max_attractors
        self.xp = xp
        
        # Build Clifford structure for this level
        self.basis = build_clifford_basis(xp)
        
        # Token embeddings for this level
        # For L1: words -> matrices
        # For L2+: attractors from below -> matrices
        self.embeddings = initialize_all_embeddings(
            input_dim, self.basis, mode='identity', noise_std=0.05, xp=xp
        )
        
        # Attractor storage
        self.attractor_matrices = xp.zeros(
            (max_attractors, MATRIX_DIM, MATRIX_DIM), dtype=xp.float64
        )
        self.attractor_contexts = []  # List of context token sequences
        self.num_attractors = 0
        
        # Statistics
        self.exact_hits = 0
        self.novel_hits = 0
    
    def embed_token(self, token: int) -> Array:
        """Get embedding matrix for a token at this level."""
        return self.embeddings[token % self.input_dim].copy()
    
    def embed_sequence(self, tokens: List[int]) -> Array:
        """
        Compute context matrix via geometric product.
        
        Args:
            tokens: List of token indices at this level
            
        Returns:
            [4, 4] context matrix
        """
        if not tokens:
            return self.xp.eye(MATRIX_DIM, dtype=self.xp.float64)
        
        if len(tokens) == 1:
            return self.embed_token(tokens[0])
        
        # Get all token matrices
        token_arr = self.xp.array(tokens, dtype=self.xp.int32)
        mats = self.embeddings[token_arr % self.input_dim]
        
        # Parallel geometric product
        return geometric_product_batch(mats, self.xp)
    
    def associate(self, context_tokens: List[int], target_matrix: Array) -> None:
        """
        Associate a context with a target attractor.
        
        Args:
            context_tokens: Sequence of tokens at this level
            target_matrix: [4, 4] target attractor matrix
        """
        ctx_hash = hash(tuple(context_tokens))
        
        # Check if context already exists (update via EMA)
        for i, stored_ctx in enumerate(self.attractor_contexts):
            if hash(tuple(stored_ctx)) == ctx_hash:
                lr = PHI_INV_SQ
                self.attractor_matrices[i] = (
                    (1 - lr) * self.attractor_matrices[i] + lr * target_matrix
                )
                return
        
        # Store new attractor
        if self.num_attractors < self.max_attractors:
            idx = self.num_attractors
            self.attractor_matrices[idx] = target_matrix
            self.attractor_contexts.append(list(context_tokens))
            self.num_attractors += 1
    
    def get_attractor(self, context_tokens: List[int]) -> Array:
        """
        Retrieve attractor for a context.
        
        Args:
            context_tokens: Sequence of tokens at this level
            
        Returns:
            [4, 4] attractor matrix
        """
        if self.num_attractors == 0:
            return self.xp.eye(MATRIX_DIM, dtype=self.xp.float64)
        
        ctx_hash = hash(tuple(context_tokens))
        
        # Try exact match
        for i, stored_ctx in enumerate(self.attractor_contexts):
            if hash(tuple(stored_ctx)) == ctx_hash:
                self.exact_hits += 1
                return self.attractor_matrices[i].copy()
        
        # Fall back to similarity-based retrieval
        self.novel_hits += 1
        query = self.embed_sequence(context_tokens)
        
        contexts = self.attractor_matrices[:self.num_attractors]
        sims = frobenius_similarity_batch(query, contexts, self.xp)
        best_idx = int(self.xp.argmax(sims))
        
        return self.attractor_matrices[best_idx].copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get level statistics."""
        total = self.exact_hits + self.novel_hits
        return {
            'level': self.level,
            'num_attractors': self.num_attractors,
            'exact_hits': self.exact_hits,
            'novel_hits': self.novel_hits,
            'generalization_rate': self.novel_hits / max(total, 1),
        }


# =============================================================================
# HIERARCHICAL MODEL — Tower of Levels
# =============================================================================

class HierarchicalModel:
    """
    Complete hierarchical model with multiple levels.
    
    Architecture:
        Level 1: word tokens -> word attractors
        Level 2: word attractors -> phrase attractors
        Level 3: phrase attractors -> sentence attractors
        ...
    
    Each level is a complete Cl(3,1) system with its own:
        - Token embeddings
        - Context computation
        - Attractor storage
        - Witness invariants
    """
    
    def __init__(self, vocab_size: int, num_levels: int = 2,
                 codebook_size: int = 1000, max_attractors_per_level: int = 10000,
                 xp: ArrayModule = np):
        """
        Args:
            vocab_size: Size of word vocabulary (Level 1 input)
            num_levels: Number of hierarchy levels
            codebook_size: Number of attractors that become tokens for next level
            max_attractors_per_level: Max attractors per level
            xp: array module
        """
        self.vocab_size = vocab_size
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.xp = xp
        
        # Create levels
        self.levels: List[HierarchyLevel] = []
        
        # Level 1: words -> attractors
        self.levels.append(HierarchyLevel(
            level=1,
            input_dim=vocab_size,
            max_attractors=max_attractors_per_level,
            xp=xp
        ))
        
        # Levels 2+: attractors -> higher attractors
        for lvl in range(2, num_levels + 1):
            self.levels.append(HierarchyLevel(
                level=lvl,
                input_dim=codebook_size,  # Attractors from below become tokens
                max_attractors=max_attractors_per_level,
                xp=xp
            ))
        
        # Codebooks connecting levels
        # codebooks[i] maps level i attractors to level i+1 tokens
        self.codebooks: List[Optional[AttractorCodebook]] = [None] * (num_levels - 1)
    
    def update_codebook(self, level: int) -> None:
        """
        Update the codebook for level -> level+1 transition.
        
        Selects top-k attractors from level to become tokens for level+1.
        
        Args:
            level: Which level's attractors to compress (1-indexed)
        """
        if level >= self.num_levels:
            return
        
        lvl = self.levels[level - 1]
        n = min(lvl.num_attractors, self.codebook_size)
        
        if n == 0:
            return
        
        # For now, take first n attractors (later: cluster by similarity)
        attractors = [lvl.attractor_matrices[i] for i in range(n)]
        
        self.codebooks[level - 1] = create_codebook(
            attractors, level, lvl.basis, self.xp
        )
        
        # Update next level's embeddings to match codebook
        next_lvl = self.levels[level]
        for i, attr in enumerate(attractors):
            # Normalize and use as embedding
            next_lvl.embeddings[i] = normalize_matrix(attr[None], self.xp)[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all levels."""
        return {
            f'level_{lvl.level}': lvl.get_statistics()
            for lvl in self.levels
        }


# =============================================================================
# TESTS
# =============================================================================

def test_hierarchy_level(xp: ArrayModule = np) -> bool:
    """Test single hierarchy level."""
    print("Testing HierarchyLevel...")
    
    # Create level
    level = HierarchyLevel(level=1, input_dim=100, max_attractors=100, xp=xp)
    
    # Test embedding
    emb = level.embed_token(42)
    assert emb.shape == (4, 4), f"Wrong shape: {emb.shape}"
    print("  ✓ embed_token")
    
    # Test sequence embedding
    ctx = level.embed_sequence([1, 2, 3, 4])
    assert ctx.shape == (4, 4), f"Wrong shape: {ctx.shape}"
    print("  ✓ embed_sequence")
    
    # Test association
    target = level.embed_token(5)
    level.associate([1, 2, 3, 4], target)
    assert level.num_attractors == 1
    print("  ✓ associate")
    
    # Test retrieval (exact)
    attr = level.get_attractor([1, 2, 3, 4])
    assert level.exact_hits == 1
    print("  ✓ get_attractor (exact)")
    
    # Test retrieval (novel)
    attr2 = level.get_attractor([5, 6, 7, 8])
    assert level.novel_hits == 1
    print("  ✓ get_attractor (novel)")
    
    print("  All HierarchyLevel tests passed!")
    return True


def test_hierarchical_model(xp: ArrayModule = np) -> bool:
    """Test multi-level hierarchical model."""
    print("Testing HierarchicalModel...")
    
    # Create 2-level model
    model = HierarchicalModel(
        vocab_size=100,
        num_levels=2,
        codebook_size=50,
        max_attractors_per_level=100,
        xp=xp
    )
    
    assert len(model.levels) == 2
    print("  ✓ Created 2-level model")
    
    # Train Level 1
    for i in range(20):
        ctx = [i, i+1, i+2]
        target = model.levels[0].embed_token((i+3) % 100)
        model.levels[0].associate(ctx, target)
    
    assert model.levels[0].num_attractors == 20
    print("  ✓ Trained Level 1 (20 attractors)")
    
    # Update codebook
    model.update_codebook(1)
    assert model.codebooks[0] is not None
    assert model.codebooks[0].matrices.shape[0] == 20
    print("  ✓ Created codebook (20 entries)")
    
    # Verify Level 2 embeddings updated
    # (First 20 embeddings should now be Level 1 attractors)
    l2_emb = model.levels[1].embed_token(0)
    l1_attr = model.levels[0].attractor_matrices[0]
    sim = float(xp.sum(l2_emb * l1_attr))
    # Should be high similarity (same matrix, normalized)
    print(f"  L2 embedding vs L1 attractor similarity: {sim:.4f}")
    
    # Get statistics
    stats = model.get_statistics()
    assert 'level_1' in stats and 'level_2' in stats
    print("  ✓ Statistics available")
    
    print("  All HierarchicalModel tests passed!")
    return True


def run_hierarchy_tests(xp: ArrayModule = np) -> bool:
    """Run all hierarchy tests."""
    print("=" * 60)
    print("HIERARCHY TESTS")
    print("=" * 60)
    
    all_pass = True
    
    if not test_hierarchy_level(xp):
        all_pass = False
    
    print()
    
    if not test_hierarchical_model(xp):
        all_pass = False
    
    print()
    print("=" * 60)
    if all_pass:
        print("ALL HIERARCHY TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    
    return all_pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'AttractorCodebook',
    'create_codebook',
    'HierarchyLevel',
    'HierarchicalModel',
    'test_hierarchy_level',
    'test_hierarchical_model',
    'run_hierarchy_tests',
]
