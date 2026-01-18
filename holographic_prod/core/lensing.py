"""
Polarized Satellite Lensing — Observer-Oriented Holographic Parallax
=====================================================================

THEORY:
    The holographic architecture faces a fundamental capacity limit:
    4D SO(4) space can only hold ~100 distinguishable embeddings before
    aliasing occurs (ghosting). This module solves the aliasing problem
    through "Holographic Parallax" — each satellite observes embeddings
    from a unique orientation, filtering via polarization (ReLU).

THE PROBLEM (Semantic Aliasing):
    - SO(4) has limited "slots" for unique rotations (~100 at ρ < 0.9)
    - 50,000 vocabulary tokens → ~500 tokens per slot → GHOSTING
    - Pure conjugation (L @ M @ L^T) preserves Frobenius correlation
    - All satellites see the SAME aliasing → no disambiguation

THE SOLUTION (Polarized Lensing):
    - Each satellite has a unique, fixed SO(4) "observer orientation"
    - Embeddings are POLARIZED (ReLU) in the observer's frame
    - ReLU destroys metric invariance (irreversible, asymmetric)
    - Different observers see different "faces" of each concept
    - Ghosts (symmetric confusion) don't survive fragmentation

PHYSICAL ANALOGY:
    Polarizing filters in optics:
    - Unpolarized light contains all orientations (ambiguous)
    - Polarizing filter keeps only aligned components (asymmetric)
    - Multiple filters at different angles distinguish sources
    - Same math: ReLU keeps only "positive-facing" components

BRAIN ANALOG (Grid Cells):
    In the entorhinal cortex, grid cells exhibit:
    - Individual aliasing: Each cell fires at multiple locations
    - Population uniqueness: Combined pattern is unique to each location
    - Phase diversity: Different cells have different phase offsets
    
    Our lenses are the "phase offsets" that make each satellite see
    a unique perspective, enabling population-level uniqueness.

MATHEMATICAL JUSTIFICATION:
    1. Frobenius = Scalar Grade of Geometric Product (theory-true)
       ⟨A, B⟩_F = Tr(A^T B) = grade₀(A † B)
       This is the "ruler" of the algebra, not an external metric.
       
    2. Pure Conjugation preserves Frobenius (problem):
       ⟨L@A@L^T, L@B@L^T⟩_F = ⟨A, B⟩_F  (metric invariant)
       
    3. Polarization (ReLU) breaks invariance (solution):
       ReLU(L@A@L^T) ≠ L @ ReLU(A) @ L^T  (asymmetric, irreversible)
       Different observers see different zeros.

CAPACITY INCREASE:
    - Single view: ~100 distinguishable embeddings
    - 16 polarized views (chord): ~100^16 = effectively unlimited
    - Collision requires matching across ALL views simultaneously

ALL φ-DERIVED. NO ARBITRARY HYPERPARAMETERS.
NO FALLBACKS. NO FAKE DATA. NO ML-THINKING.
"""

import numpy as np
from typing import List, Tuple, Optional, Any
from scipy.stats import ortho_group

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_EPSILON, MATRIX_DIM, DTYPE,
)

# GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

ArrayModule = Any  # numpy or cupy


# =============================================================================
# POLARIZED LENS
# =============================================================================

class PolarizedLens:
    """
    A satellite's unique observer orientation with polarizing filter.
    
    THEORY:
        Each satellite is an "observer" facing a unique direction in SO(4).
        The lens:
        1. Rotates embeddings into the observer's reference frame
        2. Applies ReLU polarization (keeps only "positive-facing" components)
        3. Returns a sparse, asymmetric view of the embedding
        
        This breaks the metric invariance that causes aliasing.
    
    BRAIN ANALOG:
        Like a grid cell with a specific phase offset. Individual cells
        are aliased (fire at multiple locations), but the population
        code is unique.
    
    GPU SUPPORT:
        Uses CuPy when available. Lens matrices are kept on the
        appropriate device.
    """
    
    __slots__ = ('lens', 'lens_inv', 'seed', 'xp')
    
    def __init__(self, seed: int, xp: ArrayModule = np):
        """
        Create a polarized lens with deterministic orientation.
        
        Args:
            seed: Random seed for reproducible lens generation
            xp: Array module (numpy or cupy)
        """
        self.seed = seed
        self.xp = xp
        
        # Generate deterministic SO(4) rotation
        # Uses scipy on CPU, then transfers if needed
        M = ortho_group.rvs(MATRIX_DIM, random_state=seed).astype(DTYPE)
        
        # Ensure SO(4) not O(4) (determinant +1)
        if np.linalg.det(M) < 0:
            M[:, 0] *= -1
        
        # Store on appropriate device
        if xp is np:
            self.lens = M
            self.lens_inv = M.T  # SO(4): inverse = transpose
        else:
            self.lens = xp.asarray(M)
            self.lens_inv = xp.asarray(M.T)
    
    def polarize(self, matrix: Any) -> Any:
        """
        Apply polarized projection (rotation + ReLU).
        
        THEORY:
            1. Rotate into observer's frame: L @ M @ L^T
            2. Polarize (ReLU): keep only positive components
            
            This is irreversible and breaks metric invariance.
        
        Args:
            matrix: [4, 4] embedding matrix
            
        Returns:
            [4, 4] polarized view (sparse, non-orthogonal)
        """
        xp = self.xp
        rotated = self.lens @ matrix @ self.lens_inv
        return xp.maximum(0, rotated)
    
    def polarize_batch(self, matrices: Any) -> Any:
        """
        Apply polarized projection to batch of matrices.
        
        Args:
            matrices: [batch, 4, 4] embedding matrices
            
        Returns:
            [batch, 4, 4] polarized views
        """
        xp = self.xp
        # Batch rotation: L @ M[i] @ L^T for each i
        rotated = xp.einsum('ij,bjk,kl->bil', self.lens, matrices, self.lens_inv)
        return xp.maximum(0, rotated)
    
    def restore(self, polarized: Any) -> Any:
        """
        Restore polarized view to global frame (for chord aggregation).
        
        Note: This is NOT the inverse of polarize() - information was
        destroyed by ReLU. This just rotates back to global coordinates.
        
        Args:
            polarized: [4, 4] polarized matrix in observer's frame
            
        Returns:
            [4, 4] matrix in global frame (still sparse from ReLU)
        """
        return self.lens_inv @ polarized @ self.lens
    
    def restore_batch(self, polarized: Any) -> Any:
        """
        Restore batch of polarized views to global frame.
        
        Args:
            polarized: [batch, 4, 4] polarized matrices
            
        Returns:
            [batch, 4, 4] matrices in global frame
        """
        xp = self.xp
        return xp.einsum('ij,bjk,kl->bil', self.lens_inv, polarized, self.lens)


# =============================================================================
# LENS SET (For Multi-Satellite Systems)
# =============================================================================

class PolarizedLensSet:
    """
    A collection of polarized lenses for satellite arrays.
    
    THEORY:
        Each satellite in a tower gets a unique lens. The lenses are
        deterministically generated from a master seed to ensure
        reproducibility across restarts.
        
        The lenses should be maximally diverse (spread across SO(4))
        to maximize disambiguation power.
    
    OPTIMIZATION (v5.16.1):
        Pre-computes and caches stacked lens tensors for vectorized operations.
        This avoids recomputing xp.stack() on every scoring call.
    
    Usage:
        lens_set = PolarizedLensSet(n_lenses=16, seed=42)
        
        # Store with polarization
        for i, lens in enumerate(lens_set):
            satellites[i].store(lens.polarize(embedding))
        
        # Retrieve with chord aggregation (FAST, vectorized)
        scores = lens_set.score_all_lenses_vectorized(retrieved, candidates)
    """
    
    __slots__ = ('lenses', 'n_lenses', 'seed', 'xp', '_stacked_L', '_stacked_L_inv')
    
    def __init__(self, n_lenses: int, seed: int = 42, xp: ArrayModule = np):
        """
        Create a set of polarized lenses.
        
        Args:
            n_lenses: Number of lenses (typically 16 for L0 satellites)
            seed: Master seed for reproducibility
            xp: Array module (numpy or cupy)
        """
        self.n_lenses = n_lenses
        self.seed = seed
        self.xp = xp
        
        # Generate lenses with deterministic, diverse seeds
        # Using prime multiplier (137) for good distribution
        self.lenses = [
            PolarizedLens(seed + i * 137, xp=xp)
            for i in range(n_lenses)
        ]
        
        # PRE-COMPUTE stacked tensors for vectorized operations (v5.16.1)
        # This is the key optimization - avoid xp.stack() on every scoring call
        self._stacked_L = xp.stack([lens.lens for lens in self.lenses], axis=0)  # [n_lenses, 4, 4]
        self._stacked_L_inv = xp.stack([lens.lens_inv for lens in self.lenses], axis=0)  # [n_lenses, 4, 4]
    
    def __iter__(self):
        return iter(self.lenses)
    
    def __getitem__(self, idx: int) -> PolarizedLens:
        return self.lenses[idx]
    
    def __len__(self) -> int:
        return self.n_lenses
    
    def polarize_all(self, matrix: Any) -> List[Any]:
        """
        Apply all lenses to a single embedding.
        
        Args:
            matrix: [4, 4] embedding matrix
            
        Returns:
            List of [4, 4] polarized views, one per lens
        """
        return [lens.polarize(matrix) for lens in self.lenses]
    
    def compute_chord(self, polarized_views: List[Any]) -> Any:
        """
        Compute chord representation by averaging restored views.
        
        THEORY:
            The chord is the interference pattern of all polarized views.
            True signal reinforces across views; ghosts do not.
            
        Note: This is one aggregation method. For retrieval, you may
        want to use min-distance or voting instead.
        
        Args:
            polarized_views: List of [4, 4] polarized matrices from satellites
            
        Returns:
            [4, 4] chord representation (averaged restored views)
        """
        xp = self.xp
        
        # Restore each view to global frame and sum
        restored_sum = xp.zeros((MATRIX_DIM, MATRIX_DIM), dtype=DTYPE)
        for lens, polarized in zip(self.lenses, polarized_views):
            restored_sum += lens.restore(polarized)
        
        # Average (could also use φ-weighted sum)
        return restored_sum / self.n_lenses
    
    def min_correlation_across_views(
        self, 
        polarized_a: List[Any], 
        polarized_b: List[Any]
    ) -> float:
        """
        Compute minimum Frobenius correlation across all views.
        
        THEORY (Aliasing Criterion):
            Two concepts are "aliased" only if ALL views see them as similar.
            If even ONE view distinguishes them, they're different.
            
            This is the strongest disambiguation criterion.
        
        Args:
            polarized_a: List of polarized views for embedding A
            polarized_b: List of polarized views for embedding B
            
        Returns:
            Minimum correlation across views (0 = distinguishable in some view)
        """
        xp = self.xp
        
        min_corr = 1.0
        for pa, pb in zip(polarized_a, polarized_b):
            norm_a = float(xp.linalg.norm(pa))
            norm_b = float(xp.linalg.norm(pb))
            
            if norm_a < PHI_EPSILON or norm_b < PHI_EPSILON:
                # At least one is zero in this view → distinguishable
                return 0.0
            
            corr = abs(float(xp.sum(pa * pb)) / (norm_a * norm_b))
            min_corr = min(min_corr, corr)
        
        return min_corr
    
    def to_device(self, xp: ArrayModule) -> 'PolarizedLensSet':
        """
        Move lens set to a different device (CPU ↔ GPU).
        
        Args:
            xp: Target array module (numpy or cupy)
            
        Returns:
            New PolarizedLensSet on target device (with cached tensors)
        """
        if xp is self.xp:
            return self
        # Create new lens set on target device (this also pre-computes cached tensors)
        return PolarizedLensSet(self.n_lenses, self.seed, xp=xp)
    
    def get_stacked_lenses(self) -> Any:
        """
        Get all lens matrices stacked into a single tensor.
        
        OPTIMIZATION (v5.16.1): Returns PRE-COMPUTED cached tensor.
        
        Returns:
            [n_lenses, 4, 4] tensor of lens matrices
        """
        return self._stacked_L
    
    def get_stacked_lens_invs(self) -> Any:
        """
        Get all lens inverse matrices stacked into a single tensor.
        
        OPTIMIZATION (v5.16.1): Returns PRE-COMPUTED cached tensor.
        
        Returns:
            [n_lenses, 4, 4] tensor of lens inverses
        """
        return self._stacked_L_inv
    
    def polarize_all_vectorized(self, matrix: Any) -> Any:
        """
        Apply ALL lenses to a single embedding in one vectorized operation.
        
        OPTIMIZATION: Instead of looping over lenses, use einsum for
        batch rotation and apply ReLU once.
        
        Args:
            matrix: [4, 4] embedding matrix
            
        Returns:
            [n_lenses, 4, 4] polarized views, one per lens
        """
        xp = self.xp
        
        # Get stacked lenses [n_lenses, 4, 4]
        L = self.get_stacked_lenses()
        L_inv = self.get_stacked_lens_invs()
        
        # Batch rotation: L[i] @ M @ L_inv[i] for all i
        # Using einsum: 'lij,jk,lkm->lim' where l=lens, i,j,k,m=matrix dims
        rotated = xp.einsum('lij,jk,lkm->lim', L, matrix, L_inv)
        
        # ReLU polarization (vectorized)
        return xp.maximum(0, rotated)
    
    def polarize_candidates_all_lenses(self, candidates: Any) -> Any:
        """
        Apply ALL lenses to ALL candidates in one vectorized operation.
        
        OPTIMIZATION: This is the HOT PATH for scoring. Instead of:
            for lens in lenses:
                for candidate in candidates:
                    polarize(candidate)
        
        We do a single einsum operation.
        
        Args:
            candidates: [n_candidates, 4, 4] candidate embeddings
            
        Returns:
            [n_lenses, n_candidates, 4, 4] polarized views
        """
        xp = self.xp
        
        # Get stacked lenses [n_lenses, 4, 4]
        L = self.get_stacked_lenses()
        L_inv = self.get_stacked_lens_invs()
        
        # Batch rotation: L[l] @ C[c] @ L_inv[l] for all l, c
        # Input: L[l,i,j], C[c,j,k], L_inv[l,k,m]
        # Output: [l,c,i,m]
        rotated = xp.einsum('lij,cjk,lkm->lcim', L, candidates, L_inv)
        
        # ReLU polarization (vectorized)
        return xp.maximum(0, rotated)
    
    def score_all_lenses_vectorized(
        self, 
        retrieved: Any, 
        candidates: Any,
    ) -> Any:
        """
        Score candidates across ALL lenses in one vectorized operation.
        
        OPTIMIZATION (CRITICAL FOR PERFORMANCE v5.16.1):
            Old code: 16 iterations × vorticity_weighted_scores = SLOW
            New code: Single batch Frobenius computation = FAST
            
        Uses Frobenius similarity (which IS theory-true: Scalar Grade).
        
        Args:
            retrieved: [4, 4] retrieved state from unbinding
            candidates: [n_candidates, 4, 4] candidate embeddings
            
        Returns:
            [n_candidates] averaged scores across all lenses
        """
        xp = self.xp
        n_candidates = candidates.shape[0]
        n_lenses = self.n_lenses
        
        # Get cached stacked lenses [16, 4, 4]
        L = self._stacked_L
        L_inv = self._stacked_L_inv
        
        # =====================================================================
        # STEP 1: Polarize retrieved through all lenses [16, 4, 4]
        # =====================================================================
        # Compute L[l] @ retrieved @ L_inv[l] for all l
        # More efficient as two matmuls than einsum
        temp = xp.tensordot(L, retrieved, axes=([2], [0]))  # [16, 4, 4]
        retrieved_rotated = xp.einsum('lij,ljk->lik', temp, L_inv)  # [16, 4, 4]
        retrieved_polarized = xp.maximum(0, retrieved_rotated)  # ReLU
        
        # =====================================================================
        # STEP 2: Polarize all candidates through all lenses [16, n_cand, 4, 4]
        # =====================================================================
        # Compute L[l] @ C[c] @ L_inv[l] for all l, c
        # Using two einsum operations is faster than one with 4 indices
        temp_cand = xp.einsum('lij,cjk->lcik', L, candidates)  # [16, n_cand, 4, 4]
        candidates_rotated = xp.einsum('lcij,ljk->lcik', temp_cand, L_inv)  # [16, n_cand, 4, 4]
        candidates_polarized = xp.maximum(0, candidates_rotated)  # ReLU
        
        # =====================================================================
        # STEP 3: Compute Frobenius similarity for all (lens, candidate) pairs
        # =====================================================================
        # Frobenius similarity = sum(A * B) / (||A|| * ||B||)
        # This IS theory-true: Frobenius = Scalar Grade of Geometric Product
        
        # Flatten last two dims
        r_flat = retrieved_polarized.reshape(n_lenses, -1)  # [16, 16]
        c_flat = candidates_polarized.reshape(n_lenses, n_candidates, -1)  # [16, n_cand, 16]
        
        # Compute norms (squared, then sqrt at end for efficiency)
        r_norm_sq = xp.sum(r_flat ** 2, axis=1, keepdims=True)  # [16, 1]
        c_norm_sq = xp.sum(c_flat ** 2, axis=2)  # [16, n_cand]
        
        # Dot products: r_flat[l] · c_flat[l, c]
        # Using matmul: [16, 1, 16] @ [16, 16, n_cand] -> [16, 1, n_cand] -> [16, n_cand]
        dots = xp.einsum('li,lci->lc', r_flat, c_flat)  # [16, n_cand]
        
        # Normalize
        denom = xp.sqrt(r_norm_sq * c_norm_sq) + PHI_EPSILON  # [16, n_cand]
        scores_per_lens = dots / denom  # [16, n_cand]
        
        # =====================================================================
        # STEP 4: Aggregate across lenses (average = chord)
        # =====================================================================
        return xp.mean(scores_per_lens, axis=0)  # [n_cand]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_lens_for_satellite(satellite_index: int, tower_seed: int = 42, xp: ArrayModule = np) -> PolarizedLens:
    """
    Create a deterministic lens for a specific satellite.
    
    Args:
        satellite_index: Index of the satellite (0-15 for L0)
        tower_seed: Tower's master seed
        xp: Array module
        
    Returns:
        PolarizedLens for this satellite
    """
    # Deterministic seed combining tower and satellite index
    lens_seed = tower_seed + satellite_index * 137
    return PolarizedLens(lens_seed, xp=xp)


def polarized_similarity(
    embedding_a: Any,
    embedding_b: Any,
    lens_set: PolarizedLensSet,
) -> Tuple[float, float, float]:
    """
    Compute similarity metrics using polarized lensing.
    
    Returns three metrics:
    1. min_corr: Minimum correlation across views (strictest)
    2. max_corr: Maximum correlation across views (loosest)
    3. mean_corr: Average correlation across views
    
    Args:
        embedding_a: [4, 4] first embedding
        embedding_b: [4, 4] second embedding
        lens_set: Set of polarized lenses
        
    Returns:
        (min_corr, max_corr, mean_corr)
    """
    xp = lens_set.xp
    
    correlations = []
    for lens in lens_set:
        pa = lens.polarize(embedding_a)
        pb = lens.polarize(embedding_b)
        
        norm_a = float(xp.linalg.norm(pa))
        norm_b = float(xp.linalg.norm(pb))
        
        if norm_a < PHI_EPSILON or norm_b < PHI_EPSILON:
            correlations.append(0.0)
        else:
            corr = abs(float(xp.sum(pa * pb)) / (norm_a * norm_b))
            correlations.append(corr)
    
    return min(correlations), max(correlations), sum(correlations) / len(correlations)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PolarizedLens',
    'PolarizedLensSet',
    'create_lens_for_satellite',
    'polarized_similarity',
]
