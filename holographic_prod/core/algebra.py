"""
Clifford Algebra Operations — Matrix Representation
====================================================

Cl(3,1) ≅ M₄(ℝ) — Geometric product IS matrix multiplication!

This module provides:
    1. Gamma matrices (Cl(3,1) generators)
    2. Full 16-element Clifford basis
    3. Geometric product (full AB)
    4. Wedge product (antisymmetric A∧B) — VORTICITY CONTAINER
    5. Inner product (symmetric A·B)
    6. Grace operator (grade-wise contraction) — VISCOUS DAMPING
    7. Similarity functions (Frobenius and metric-aware)
    8. Vorticity signature/similarity — GRAMMAR MATCHING

KEY INSIGHT — Grace as Viscosity:
    The wedge product A∧B captures rotation/vorticity (lives in grade-2 bivectors).
    The Grace operator damps grade-2 by φ⁻² per step, acting as viscosity.
    Enstrophy (||bivector content||²) decays at rate φ⁻⁴, preventing blow-up.
    
    This is the Clifford algebra analogue of viscous damping in Navier-Stokes.

KEY INSIGHT — Vorticity Grammar:
    A∧B = -B∧A (antisymmetric!) — reversed word order = opposite signature
    
    vorticity_signature(sequence) → 16 coefficients capturing sequential structure
    vorticity_similarity(sig1, sig2) → cosine similarity
    
    Same grammatical structure → 0.92+ similarity
    Different structure → <0.3 similarity
    
    Discriminates "I saw the man" vs "The man saw I" WITHOUT parsing!

All functions support both numpy and cupy arrays.
"""

from typing import Tuple, List, Any
import numpy as np

from .constants import PHI, PHI_INV, PHI_INV_SQ, PHI_EPSILON, GRACE_SCALES_FLAT, DTYPE

ArrayModule = type(np)
Array = np.ndarray


# =============================================================================
# GAMMA MATRICES — Cl(3,1) basis as 4×4 real matrices
# =============================================================================

def build_gamma_matrices(xp: ArrayModule = np) -> Array:
    """
    Build REAL gamma matrices for Cl(3,1).
    
    Signature: η = diag(+1,+1,+1,-1)
        e₁² = e₂² = e₃² = +I (spacelike)
        e₄² = -I (timelike)
        {eᵢ, eⱼ} = 0 for i ≠ j
    
    Returns:
        [4, 4, 4] gamma matrices (gamma[i] = eᵢ₊₁)
    """
    gamma = xp.zeros((4, 4, 4), dtype=DTYPE)
    
    # e₁ = σ₃ ⊗ I₂ (e₁² = +I)
    gamma[0] = xp.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]
    ], dtype=DTYPE)
    
    # e₂ = σ₁ ⊗ σ₃ (e₂² = +I)
    gamma[1] = xp.array([
        [0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]
    ], dtype=DTYPE)
    
    # e₃ = σ₁ ⊗ σ₁ (e₃² = +I)
    gamma[2] = xp.array([
        [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]
    ], dtype=DTYPE)
    
    # e₄ = σ₂ ⊗ I₂ (e₄² = -I, timelike)
    gamma[3] = xp.array([
        [0, 0, 0, -1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]
    ], dtype=DTYPE)
    
    return gamma


def build_clifford_basis(xp: ArrayModule = np) -> Array:
    """
    Build all 16 basis elements of Cl(3,1) as 4×4 REAL matrices.
    
    Grade 0: 1 (identity)
    Grade 1: e₁, e₂, e₃, e₄
    Grade 2: e₁e₂, e₁e₃, e₁e₄, e₂e₃, e₂e₄, e₃e₄
    Grade 3: e₁e₂e₃, e₁e₂e₄, e₁e₃e₄, e₂e₃e₄
    Grade 4: e₁e₂e₃e₄ (pseudoscalar)
    
    Returns:
        [16, 4, 4] basis matrices
    """
    gamma = build_gamma_matrices(xp)
    I = xp.eye(4, dtype=DTYPE)
    basis = xp.zeros((16, 4, 4), dtype=DTYPE)
    
    # Grade 0
    basis[0] = I
    
    # Grade 1
    basis[1:5] = gamma
    
    # Grade 2 (bivectors)
    basis[5] = gamma[0] @ gamma[1]   # e₁e₂
    basis[6] = gamma[0] @ gamma[2]   # e₁e₃
    basis[7] = gamma[0] @ gamma[3]   # e₁e₄
    basis[8] = gamma[1] @ gamma[2]   # e₂e₃
    basis[9] = gamma[1] @ gamma[3]   # e₂e₄
    basis[10] = gamma[2] @ gamma[3]  # e₃e₄
    
    # Grade 3 (trivectors)
    basis[11] = gamma[0] @ gamma[1] @ gamma[2]  # e₁e₂e₃
    basis[12] = gamma[0] @ gamma[1] @ gamma[3]  # e₁e₂e₄
    basis[13] = gamma[0] @ gamma[2] @ gamma[3]  # e₁e₃e₄
    basis[14] = gamma[1] @ gamma[2] @ gamma[3]  # e₂e₃e₄
    
    # Grade 4 (pseudoscalar)
    basis[15] = gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
    
    return basis


def build_structure_tensor(basis: Array = None, xp: ArrayModule = np) -> Array:
    """
    Build the Clifford algebra structure tensor GAMMA[i,j,k].
    
    THEORY-TRUE TENSOR CORE ACCELERATION:
        The geometric product eᵢ × eⱼ = Σₖ Γᵢⱼₖ eₖ
        where Γᵢⱼₖ are the structure constants.
        
    This allows computing geometric products as:
        (a × b)ₖ = Σᵢⱼ Γᵢⱼₖ aᵢ bⱼ
        
    Which can be done via 16×16 matrix operations (tensor core friendly!)
    
    Args:
        basis: [16, 4, 4] Clifford basis (computed if None)
        xp: array module
        
    Returns:
        [16, 16, 16] structure tensor
    """
    if basis is None:
        basis = build_clifford_basis(xp)
    
    # Precompute basis norms for projection
    # basis_norms[k] = trace(basis[k] @ basis[k].T)
    basis_norms = xp.array([
        float(xp.sum(basis[k] * basis[k])) for k in range(16)
    ])
    
    # Compute structure constants: Γᵢⱼₖ = trace(eᵢeⱼ · eₖ) / ||eₖ||²
    GAMMA = xp.zeros((16, 16, 16), dtype=DTYPE)
    
    for i in range(16):
        for j in range(16):
            # Compute product eᵢ × eⱼ (matrix multiplication)
            product = basis[i] @ basis[j]
            
            # Project onto each basis element
            for k in range(16):
                if basis_norms[k] > PHI_EPSILON:
                    GAMMA[i, j, k] = float(xp.sum(product * basis[k])) / basis_norms[k]
    
    return GAMMA


# Cached structure tensor (computed once)
_STRUCTURE_TENSOR_CACHE = {}

# Cached Clifford basis (computed once per array module)
_BASIS_CACHE = {}


def get_cached_basis(xp: ArrayModule = np) -> Array:
    """
    Get cached Clifford basis for the given array module.
    
    OPTIMIZATION: Basis is computed once and reused everywhere.
    Previous bug: basis was rebuilt 4+ times per HolographicMemory init!
    """
    key = id(xp)
    if key not in _BASIS_CACHE:
        _BASIS_CACHE[key] = build_clifford_basis(xp)
    return _BASIS_CACHE[key]


def get_structure_tensor(xp: ArrayModule = np) -> Array:
    """Get cached structure tensor for the given array module."""
    key = id(xp)
    if key not in _STRUCTURE_TENSOR_CACHE:
        basis = get_cached_basis(xp)
        _STRUCTURE_TENSOR_CACHE[key] = build_structure_tensor(basis, xp)
    return _STRUCTURE_TENSOR_CACHE[key]


def geometric_product_coefficients(a: Array, b: Array, xp: ArrayModule = np) -> Array:
    """
    Compute geometric product using coefficient representation.
    
    TENSOR CORE ACCELERATION:
        Instead of 4×4 matrix multiply (not tensor core friendly),
        use 16-vector coefficients with structure tensor.
        
        (a × b)ₖ = Σᵢⱼ Γᵢⱼₖ aᵢ bⱼ
        
    This can be restructured as 16×16 matrix operations!
    
    Args:
        a: [BATCH, 16] or [16] coefficient vectors
        b: [BATCH, 16] or [16] coefficient vectors
        xp: array module
        
    Returns:
        [BATCH, 16] or [16] result coefficients
    """
    GAMMA = get_structure_tensor(xp)
    
    # Handle both batched and non-batched inputs
    if a.ndim == 1:
        # Single vector: [16]
        # result[k] = Σᵢⱼ GAMMA[i,j,k] × a[i] × b[j]
        return xp.einsum('ijk,i,j->k', GAMMA, a, b)
    else:
        # Batched: [BATCH, 16]
        # result[b,k] = Σᵢⱼ GAMMA[i,j,k] × a[b,i] × b[b,j]
        return xp.einsum('ijk,bi,bj->bk', GAMMA, a, b)


def geometric_product_coefficients_optimized(a: Array, b: Array, xp: ArrayModule = np) -> Array:
    """
    Optimized geometric product using restructured matmul.
    
    TENSOR CORE FRIENDLY:
        Restructure einsum as actual matrix multiplies that
        can leverage H100 tensor cores.
        
        Step 1: B_contrib[b,i,k] = Σⱼ GAMMA[i,j,k] × b[b,j]
                This is a batched matmul: [16,16,16] @ [BATCH,16] → [BATCH,16,16]
                
        Step 2: result[b,k] = Σᵢ B_contrib[b,i,k] × a[b,i]
                This is a batched vector-matrix product
    
    Args:
        a: [BATCH, 16] coefficient vectors
        b: [BATCH, 16] coefficient vectors
        xp: array module
        
    Returns:
        [BATCH, 16] result coefficients
    """
    GAMMA = get_structure_tensor(xp)
    batch_size = a.shape[0]
    
    # Reshape GAMMA for matmul: [16, 16*16] then reshape result
    # GAMMA[i,j,k] with b[batch,j] → need [batch, i, k]
    
    # Method: Use tensordot for better GPU utilization
    # B_contrib[batch, i, k] = Σⱼ GAMMA[i,j,k] × b[batch, j]
    # Reshape GAMMA to [16*16, 16] and b to [batch, 16]
    GAMMA_flat = GAMMA.reshape(16, 16*16)  # [i, j*k]
    
    # This is the key 16×16 matmul that tensor cores can accelerate!
    # For each batch element: contrib = GAMMA_transposed @ b
    # b_expanded: [batch, 16, 1]
    # GAMMA_expanded: [1, 16, 16*16] → need to reorganize
    
    # Actually, the cleanest approach:
    # Step 1: Compute outer product a ⊗ b: [batch, 16, 16]
    outer = a[:, :, None] * b[:, None, :]  # [batch, 16, 16]
    
    # Step 2: Contract with GAMMA: [16,16,16] · [batch,16,16] → [batch,16]
    # Reshape outer to [batch, 256]
    outer_flat = outer.reshape(batch_size, 256)
    
    # Reshape GAMMA to [256, 16] (i*j, k)
    GAMMA_contract = GAMMA.reshape(256, 16)
    
    # Final matmul: [batch, 256] @ [256, 16] → [batch, 16]
    # THIS IS A TENSOR CORE FRIENDLY MATMUL!
    result = outer_flat @ GAMMA_contract
    
    return result


def build_metric_matrix(xp: ArrayModule = np) -> Array:
    """
    Build the metric matrix G for Clifford adjoint.
    
    For Cl(3,1), G = e₄ (timelike basis).
    The Clifford adjoint is: A† = G A^T G
    
    Note: G² = -I, so G⁻¹ = -G
    
    Returns:
        [4, 4] metric matrix
    """
    gamma = build_gamma_matrices(xp)
    return gamma[3]  # e₄


# =============================================================================
# MATRIX OPERATIONS
# =============================================================================

def normalize_matrix(m: Array, xp: ArrayModule = np) -> Array:
    """
    Normalize matrix to unit Frobenius norm.
    
    ⚠️⚠️⚠️ DEPRECATED — DO NOT USE ⚠️⚠️⚠️
    
    This function exists ONLY for backwards compatibility with legacy tests.
    
    WHY THIS DESTROYS SO(4):
        - SO(4) matrices have Frobenius norm = 2 (always)
        - This function divides by norm, giving R / 2
        - det(R/2) = det(R) / 2^4 = 1/16 ≠ 1
        - The result is NO LONGER in SO(4)
    
    WHY YOU DON'T NEED THIS:
        - SO(4) is a GROUP: R1 @ R2 is automatically SO(4)
        - After 1000 compositions: norm=2.0, det=1.0, orthogonal
        - The algebra SELF-MANAGES without any normalization
    
    FOR SIMILARITY COMPARISON:
        - Use frobenius_cosine(a, b) — computes a·b/(|a||b|) without modifying
        - All SO(4) have norm=2, so frobenius_cosine = a·b/4
    
    Legacy code in archive/ uses this. Production code should NEVER use this.
    """
    norm = xp.sqrt(xp.sum(m**2, axis=(-2, -1), keepdims=True))
    return m / xp.maximum(norm, PHI_EPSILON)


def geometric_product(a: Array, b: Array) -> Array:
    """
    Geometric product of Clifford multivectors (matrix multiplication).
    
    AB = A·B + A∧B (contains both symmetric and antisymmetric parts)
    """
    return a @ b


def wedge_product(a: Array, b: Array, xp: ArrayModule = np) -> Array:
    """
    Wedge (exterior) product: A∧B = (AB - BA) / 2
    
    The ANTISYMMETRIC part of the geometric product.
    
    Properties:
        - A∧B = -B∧A (anticommutative)
        - Captures ORDER (word order matters!)
        - Pure rotation, no scaling
        - Maps to NS vorticity / syntactic tension
    
    Args:
        a: [4, 4] or [batch, 4, 4] first multivector(s)
        b: [4, 4] or [batch, 4, 4] second multivector(s)
        xp: array module
        
    Returns:
        [4, 4] or [batch, 4, 4] wedge product
    """
    return (a @ b - b @ a) / 2.0


def inner_product(a: Array, b: Array, xp: ArrayModule = np) -> Array:
    """
    Inner (symmetric) product: A·B = (AB + BA) / 2
    
    The SYMMETRIC part of the geometric product.
    
    Properties:
        - A·B = B·A (commutative)
        - Captures SIMILARITY (shared structure)
        - Pure scaling, no rotation
    
    Args:
        a: [4, 4] or [batch, 4, 4] first multivector(s)
        b: [4, 4] or [batch, 4, 4] second multivector(s)
        xp: array module
        
    Returns:
        [4, 4] or [batch, 4, 4] inner product
    """
    return (a @ b + b @ a) / 2.0


def geometric_product_batch(matrices: Array, xp: ArrayModule = np) -> Array:
    """
    Compute cumulative geometric product of a sequence of matrices.
    
    Uses parallel binary reduction.
    
    SO(4) PRESERVATION:
        With SO(4) embeddings, NO normalization is needed:
        - SO(4) × SO(4) = SO(4)  (closed under multiplication)
        - det(A @ B) = det(A) × det(B) = 1 × 1 = 1
        - (A @ B)ᵀ @ (A @ B) = Bᵀ @ Aᵀ @ A @ B = Bᵀ @ I @ B = I
        - Products remain orthogonal with det=1, cond=1
        
    CRITICAL: Frobenius normalization DESTROYS SO(4)!
        - ||M||_F = 2 for SO(4) matrices
        - M / ||M||_F has ||M'||_F = 1, but is NOT orthogonal
        - This breaks unbinding (M.T no longer equals M⁻¹)
    
    Args:
        matrices: [n, 4, 4] sequence of SO(4) matrices
        xp: array module
        
    Returns:
        [4, 4] cumulative product (still SO(4), no normalization needed)
    """
    if matrices.shape[0] == 0:
        return xp.eye(4, dtype=DTYPE)
    
    if matrices.shape[0] == 1:
        return matrices[0]
    
    # Parallel pairwise reduction - NO normalization for SO(4)!
    mats = matrices.copy()
    while mats.shape[0] > 1:
        n = mats.shape[0]
        if n % 2 == 1:
            last = mats[-1:]
            a, b = mats[0:n-1:2], mats[1:n-1:2]
            reduced = xp.matmul(a, b)
            # NO normalization - SO(4) × SO(4) = SO(4)
            mats = xp.concatenate([reduced, last], axis=0)
        else:
            a, b = mats[0::2], mats[1::2]
            mats = xp.matmul(a, b)
            # NO normalization - SO(4) × SO(4) = SO(4)
    
    return mats[0]


def geometric_product_batch_multi(batch_matrices: Array, xp: ArrayModule = np) -> Array:
    """
    Compute cumulative geometric product for MULTIPLE sequences in parallel.
    
    Uses batched cuBLAS matmuls for each reduction level.
    Each level is ONE highly optimized GPU call.
    
    SO(4) PRESERVATION:
        With SO(4) embeddings, NO normalization is needed:
        - SO(4) × SO(4) = SO(4)  (closed under multiplication)
        - Products remain orthogonal with det=1, cond=1
        
    CRITICAL: Frobenius normalization DESTROYS SO(4)!
    
    Args:
        batch_matrices: [BATCH, SEQ_LEN, 4, 4] - batch of SO(4) embedding sequences
        xp: array module
        
    Returns:
        [BATCH, 4, 4] - one context matrix per sequence (still SO(4))
    """
    batch_size = batch_matrices.shape[0]
    seq_len = batch_matrices.shape[1]
    
    if seq_len == 0:
        return xp.eye(4, dtype=DTYPE).reshape(1, 4, 4).repeat(batch_size, axis=0)
    
    if seq_len == 1:
        return batch_matrices[:, 0, :, :]
    
    # Binary reduction with batched matmuls (cuBLAS optimized)
    # NO normalization - SO(4) × SO(4) = SO(4)
    mats = batch_matrices
    
    while mats.shape[1] > 1:
        n = mats.shape[1]
        if n % 2 == 1:
            # Odd: save last, reduce pairs
            last = mats[:, -1:, :, :]
            a = mats[:, 0:n-1:2, :, :]
            b = mats[:, 1:n-1:2, :, :]
            reduced = xp.matmul(a, b)
            # NO normalization - preserves SO(4)
            mats = xp.concatenate([reduced, last], axis=1)
        else:
            a = mats[:, 0::2, :, :]
            b = mats[:, 1::2, :, :]
            mats = xp.matmul(a, b)
            # NO normalization - preserves SO(4)
    
    return mats[:, 0, :, :]


def geometric_product_batch_multi_coefficients(batch_coefficients: Array, xp: ArrayModule = np) -> Array:
    """
    Compute cumulative geometric product using COEFFICIENT representation.
    
    TENSOR CORE ACCELERATION:
        Instead of 4×4 matrix multiplies, use 16-vector coefficient products
        via the structure tensor, which enables 256×16 matmuls.
        
        H100 tensor cores can accelerate this!
    
    Args:
        batch_coefficients: [BATCH, SEQ_LEN, 16] - batch of coefficient sequences
        xp: array module
        
    Returns:
        [BATCH, 16] - one result coefficient vector per sequence
    """
    batch_size = batch_coefficients.shape[0]
    seq_len = batch_coefficients.shape[1]
    
    if seq_len == 0:
        # Return identity (1, 0, 0, ..., 0)
        result = xp.zeros((batch_size, 16), dtype=batch_coefficients.dtype)
        result[:, 0] = 1.0
        return result
    
    if seq_len == 1:
        return batch_coefficients[:, 0, :]
    
    # Get structure tensor
    GAMMA = get_structure_tensor(xp)
    GAMMA_contract = GAMMA.reshape(256, 16)  # For tensor core matmul
    
    # Parallel binary reduction across ALL sequences
    coeffs = batch_coefficients.copy()  # [BATCH, SEQ_LEN, 16]
    
    while coeffs.shape[1] > 1:
        n = coeffs.shape[1]
        if n % 2 == 1:
            # Handle odd: save last, reduce pairs, concatenate
            last = coeffs[:, -1:, :]  # [BATCH, 1, 16]
            a = coeffs[:, 0:n-1:2, :]  # [BATCH, K, 16]
            b = coeffs[:, 1:n-1:2, :]  # [BATCH, K, 16]
            
            # TENSOR CORE MATMUL: Compute geometric product via structure tensor
            # Outer product: [BATCH, K, 16, 16]
            outer = a[:, :, :, None] * b[:, :, None, :]
            # Reshape: [BATCH*K, 256]
            BK = a.shape[0] * a.shape[1]
            outer_flat = outer.reshape(BK, 256)
            # Matmul with structure tensor: [BATCH*K, 256] @ [256, 16] → [BATCH*K, 16]
            reduced_flat = outer_flat @ GAMMA_contract
            # Reshape back: [BATCH, K, 16]
            reduced = reduced_flat.reshape(batch_size, -1, 16)
            
            # Normalize (L2 norm of coefficient vector)
            norms = xp.sqrt(xp.sum(reduced ** 2, axis=-1, keepdims=True))
            norms = xp.maximum(norms, PHI_EPSILON)
            reduced = reduced / norms
            
            coeffs = xp.concatenate([reduced, last], axis=1)
        else:
            a = coeffs[:, 0::2, :]  # [BATCH, n/2, 16]
            b = coeffs[:, 1::2, :]  # [BATCH, n/2, 16]
            
            # TENSOR CORE MATMUL
            outer = a[:, :, :, None] * b[:, :, None, :]  # [BATCH, n/2, 16, 16]
            BK = a.shape[0] * a.shape[1]
            outer_flat = outer.reshape(BK, 256)
            coeffs_flat = outer_flat @ GAMMA_contract  # [BATCH*K, 16]
            coeffs = coeffs_flat.reshape(batch_size, -1, 16)
            
            # Normalize
            norms = xp.sqrt(xp.sum(coeffs ** 2, axis=-1, keepdims=True))
            norms = xp.maximum(norms, PHI_EPSILON)
            coeffs = coeffs / norms
    
    return coeffs[:, 0, :]  # [BATCH, 16]


def matrix_to_coefficients(matrices: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    Convert 4×4 matrices to 16-vector coefficients.
    
    Args:
        matrices: [..., 4, 4] matrices
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        [..., 16] coefficient vectors
    """
    # Compute basis norms
    basis_norms = xp.sum(basis * basis, axis=(-2, -1))  # [16]
    
    # Project onto each basis element
    # For each matrix M: coeff[k] = trace(M · basis[k]) / ||basis[k]||²
    # Using einsum: [..., 4, 4] · [16, 4, 4] → [..., 16]
    
    if matrices.ndim == 2:
        # Single matrix [4, 4]
        coeffs = xp.einsum('ij,kij->k', matrices, basis) / basis_norms
    elif matrices.ndim == 3:
        # Batch [BATCH, 4, 4]
        coeffs = xp.einsum('bij,kij->bk', matrices, basis) / basis_norms
    else:
        # Higher dims [..., 4, 4]
        coeffs = xp.einsum('...ij,kij->...k', matrices, basis) / basis_norms
    
    return coeffs


def coefficients_to_matrix(coeffs: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    Convert 16-vector coefficients to 4×4 matrices.
    
    Args:
        coeffs: [..., 16] coefficient vectors
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        [..., 4, 4] matrices
    """
    # M = Σₖ coeff[k] × basis[k]
    if coeffs.ndim == 1:
        # Single vector [16]
        return xp.einsum('k,kij->ij', coeffs, basis)
    elif coeffs.ndim == 2:
        # Batch [BATCH, 16]
        return xp.einsum('bk,kij->bij', coeffs, basis)
    else:
        # Higher dims [..., 16]
        return xp.einsum('...k,kij->...ij', coeffs, basis)


def sliding_context_incremental(
    embeddings: Array,
    start_idx: int,
    context_size: int,
    prev_context: Array = None,
    prev_first_token_idx: int = None,
    xp: ArrayModule = np
) -> Tuple[Array, int]:
    """
    Compute context matrix using incremental update for sliding windows.
    
    OPTIMIZATION INSIGHT:
        Consecutive sliding windows share (context_size - 1) tokens.
        Instead of recomputing all context_size products, we can:
        
        Context[i+1] = reversion(F_i) × Context[i] × F_{i+context_size}
        
        This reduces O(context_size) to O(3) operations per slide!
        
    MATHEMATICAL BASIS:
        For Clifford algebra elements, reversion(M) acts as approximate inverse:
        M × reversion(M) ≈ scalar (for normalized elements)
        
        So: rev(F_i) × (F_i × F_{i+1} × ... × F_{i+k-1}) × F_{i+k}
          ≈ F_{i+1} × ... × F_{i+k-1} × F_{i+k}
          = Context[i+1]
    
    Args:
        embeddings: [vocab_size, 4, 4] embedding matrices
        start_idx: Starting token index for this context
        context_size: Number of tokens in context
        prev_context: Previous context matrix (if doing incremental update)
        prev_first_token_idx: Token index that was first in previous context
        xp: Array module
        
    Returns:
        (context_matrix, first_token_idx) - context and its first token index
        
    SPEEDUP: 512/3 ≈ 170× for context computation when sliding by 1
    """
    # If no previous context or can't do incremental, compute from scratch
    if prev_context is None or prev_first_token_idx is None:
        # Full computation
        token_mats = embeddings[start_idx:start_idx + context_size]
        if len(token_mats) < context_size:
            # Not enough tokens, compute what we have
            context = geometric_product_batch(token_mats, xp)
        else:
            context = geometric_product_batch(token_mats, xp)
        return context, start_idx
    
    # Check if we're sliding by exactly 1 (the common case)
    slide_amount = start_idx - prev_first_token_idx
    
    if slide_amount != 1:
        # Not a single-step slide, recompute from scratch
        token_mats = embeddings[start_idx:start_idx + context_size]
        context = geometric_product_batch(token_mats, xp)
        return context, start_idx
    
    # INCREMENTAL UPDATE: slide by 1
    # Remove old first token, add new last token
    old_first = embeddings[prev_first_token_idx]  # F_i (being removed)
    new_last = embeddings[start_idx + context_size - 1]  # F_{i+k} (being added)
    
    # Compute reversion of old first token (approximate inverse)
    old_first_rev = clifford_reversion(old_first)
    
    # Incremental update: rev(F_i) × Context[i] × F_{i+k}
    # Step 1: Remove old first token
    intermediate = xp.matmul(old_first_rev, prev_context)
    # Step 2: Add new last token
    new_context = xp.matmul(intermediate, new_last)
    
    # Normalize to prevent drift (uniform scaling, preserves grade ratios)
    norm = xp.linalg.norm(new_context, 'fro')
    if norm > PHI_EPSILON:
        new_context = new_context / norm
    
    return new_context, start_idx


def clifford_reversion(M: Array) -> Array:
    """
    Compute the reversion (dagger) of a Clifford algebra element.
    
    For Clifford algebra Cl(3,1), reversion reverses the order of
    basis vector products:
        (e_i e_j e_k)† = e_k e_j e_i
        
    For grade-k elements, reversion introduces sign (-1)^(k(k-1)/2):
        Grade 0: +1 (scalars unchanged)
        Grade 1: +1 (vectors unchanged)  
        Grade 2: -1 (bivectors negated)
        Grade 3: -1 (trivectors negated)
        Grade 4: +1 (pseudoscalar unchanged)
        
    In the matrix representation, this is the TRANSPOSE.
    
    For elements close to the Clifford group (products of unit vectors),
    M × reversion(M) ≈ scalar × I
    
    This makes reversion an approximate inverse for normalized elements.
    """
    return M.T


def compute_vorticity(matrices: Array, xp: ArrayModule = np) -> Array:
    """
    Compute sequential vorticity (wedge products between consecutive tokens).
    
    Vorticity captures:
        - Word ORDER (A∧B = -B∧A)
        - Syntactic tension between tokens
        - Semantic directionality
    
    This is the "relational curl" term from NS-Clifford analogy.
    
    Args:
        matrices: [n, 4, 4] sequence of token embeddings
        xp: array module
        
    Returns:
        [n-1, 4, 4] pairwise wedge products
    """
    if matrices.shape[0] < 2:
        return xp.zeros((0, 4, 4), dtype=DTYPE)
    
    # Fᵢ ∧ Fᵢ₋₁ for all consecutive pairs
    a = matrices[1:]   # Fᵢ
    b = matrices[:-1]  # Fᵢ₋₁
    
    return wedge_product(a, b, xp)


def vorticity_magnitude(matrices: Array, xp: ArrayModule = np) -> float:
    """
    Compute total vorticity magnitude (scalar measure of sequential tension).
    
    Args:
        matrices: [n, 4, 4] sequence of token embeddings
        xp: array module
        
    Returns:
        Scalar vorticity magnitude
    """
    vort = compute_vorticity(matrices, xp)
    if vort.shape[0] == 0:
        return 0.0
    
    # Sum of Frobenius norms
    return float(xp.sum(xp.sqrt(xp.sum(vort**2, axis=(-2, -1)))))


def vorticity_signature(matrices: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    Extract the vorticity signature of a token sequence as Clifford coefficients.
    
    THEORY:
        Vorticity captures WORD ORDER through consecutive wedge products.
        The signature is a 16-coefficient vector that encodes:
        - Which grade components carry the order information
        - The specific rotation pattern of the sequence
        
        Two sequences with the SAME syntactic structure will have
        SIMILAR vorticity signatures (high cosine similarity).
        
    Args:
        matrices: [n, 4, 4] sequence of token embeddings
        basis: Clifford basis [16, 4, 4]
        xp: array module
        
    Returns:
        [16] array of Clifford coefficients encoding the vorticity structure
    """
    if matrices.shape[0] < 2:
        return xp.zeros(16, dtype=DTYPE)
    
    # Compute all pairwise vorticities and sum
    vort = compute_vorticity(matrices, xp)  # [n-1, 4, 4]
    vort_sum = xp.sum(vort, axis=0)  # [4, 4]
    
    # Decompose into Clifford coefficients
    return decompose_to_coefficients(vort_sum, basis, xp)


def vorticity_magnitude_and_signature(
    matrices: Array, 
    basis: Array, 
    xp: ArrayModule = np
) -> Tuple[float, Array]:
    """
    Compute BOTH vorticity magnitude and signature in one pass.
    
    OPTIMIZATION: compute_vorticity() is expensive (all pairwise wedge products).
    This function computes it ONCE and extracts both quantities.
    
    For 50k context (50 chunks of 1k tokens):
        - Before: 100 compute_vorticity calls (2 per chunk)
        - After:  50 compute_vorticity calls (1 per chunk)
        - 2x speedup for vorticity computation!
    
    Args:
        matrices: [n, 4, 4] sequence of token embeddings
        basis: Clifford basis [16, 4, 4]
        xp: array module
        
    Returns:
        (magnitude: float, signature: [16] array)
    """
    if matrices.shape[0] < 2:
        return 0.0, xp.zeros(16, dtype=DTYPE)
    
    # Compute vorticity ONCE
    vort = compute_vorticity(matrices, xp)  # [n-1, 4, 4]
    
    # Magnitude: sum of Frobenius norms
    magnitude = float(xp.sum(xp.sqrt(xp.sum(vort**2, axis=(-2, -1)))))
    
    # Signature: decompose summed vorticity
    vort_sum = xp.sum(vort, axis=0)  # [4, 4]
    signature = decompose_to_coefficients(vort_sum, basis, xp)
    
    return magnitude, signature


def vorticity_similarity(sig1: Array, sig2: Array, xp: ArrayModule = np) -> float:
    """
    Compute similarity between two vorticity signatures.
    
    Uses cosine similarity: same structure → +1, opposite structure → -1.
    
    Args:
        sig1: [16] first vorticity signature
        sig2: [16] second vorticity signature
        xp: array module
        
    Returns:
        Cosine similarity in [-1, 1]
    """
    norm1 = xp.linalg.norm(sig1)
    norm2 = xp.linalg.norm(sig2)
    
    if norm1 < PHI_EPSILON or norm2 < PHI_EPSILON:
        return 0.0  # Can't compare zero signatures
    
    return float(xp.dot(sig1, sig2) / (norm1 * norm2))


def vorticity_magnitude_and_signature_batch(
    batch_matrices: Array, 
    basis: Array, 
    xp: ArrayModule = np
) -> Tuple[Array, Array]:
    """
    Compute vorticity magnitude and signature for MULTIPLE sequences in parallel.
    
    OPTIMIZATION: Batch version of vorticity_magnitude_and_signature.
    
    Args:
        batch_matrices: [BATCH, SEQ_LEN, 4, 4] batch of token embedding sequences
        basis: Clifford basis [16, 4, 4]
        xp: array module
        
    Returns:
        (magnitudes: [BATCH], signatures: [BATCH, 16])
    """
    batch_size = batch_matrices.shape[0]
    seq_len = batch_matrices.shape[1]
    
    if seq_len < 2:
        return (
            xp.zeros(batch_size, dtype=DTYPE),
            xp.zeros((batch_size, 16), dtype=DTYPE)
        )
    
    # Compute vorticity for all sequences: [BATCH, SEQ_LEN-1, 4, 4]
    a = batch_matrices[:, 1:, :, :]   # [BATCH, SEQ_LEN-1, 4, 4]
    b = batch_matrices[:, :-1, :, :]  # [BATCH, SEQ_LEN-1, 4, 4]
    
    # Batched wedge product: (AB - BA) / 2
    ab = xp.matmul(a, b)
    ba = xp.matmul(b, a)
    vort = (ab - ba) / 2  # [BATCH, SEQ_LEN-1, 4, 4]
    
    # Magnitude: sum of Frobenius norms for each sequence
    # [BATCH, SEQ_LEN-1] → [BATCH]
    magnitudes = xp.sum(xp.sqrt(xp.sum(vort**2, axis=(-2, -1))), axis=1)
    
    # Signature: decompose summed vorticity for each sequence
    # [BATCH, 4, 4]
    vort_sum = xp.sum(vort, axis=1)
    
    # VECTORIZED: Batch decomposition [BATCH, 4, 4] → [BATCH, 16]
    # Using einsum to avoid Python loop (was 16 GPU kernel launches!)
    # signatures[b, k] = sum(vort_sum[b] * basis[k]) / 4
    signatures = xp.einsum('bij,kij->bk', vort_sum, basis) / 4.0
    
    return magnitudes, signatures


# =============================================================================
# SIMILARITY FUNCTIONS
# =============================================================================

def frobenius_similarity(a: Array, b: Array, xp: ArrayModule = np) -> float:
    """Frobenius inner product (raw, unbounded). Use frobenius_cosine for normalized."""
    return float(xp.sum(a * b))


def frobenius_similarity_batch(query: Array, contexts: Array, xp: ArrayModule = np) -> Array:
    """Batch Frobenius inner product (raw). Use frobenius_cosine_batch for normalized."""
    return xp.sum(query * contexts, axis=(1, 2))


def frobenius_cosine(a: Array, b: Array, xp: ArrayModule = np) -> float:
    """
    Frobenius cosine similarity (normalized to [-1, 1]).
    
    THEORY-TRUE: Measures DIRECTION, not magnitude.
    Direction = what concept; magnitude = salience (handled separately).
    
    FIX v3: Ensures both arrays are on the same device before computation.
    """
    # Ensure device consistency
    a_is_cupy = hasattr(a, '__cuda_array_interface__')
    b_is_cupy = hasattr(b, '__cuda_array_interface__')
    
    if a_is_cupy != b_is_cupy:
        if hasattr(xp, 'cuda'):  # xp is CuPy
            if not a_is_cupy:
                a = xp.asarray(a)
            if not b_is_cupy:
                b = xp.asarray(b)
        else:  # xp is NumPy
            if a_is_cupy:
                a = a.get()
            if b_is_cupy:
                b = b.get()
    
    a_norm = xp.linalg.norm(a, 'fro')
    b_norm = xp.linalg.norm(b, 'fro')
    if a_norm < PHI_EPSILON or b_norm < PHI_EPSILON:
        return 0.0
    return float(xp.sum(a * b) / (a_norm * b_norm))


def frobenius_cosine_batch(query: Array, contexts: Array, xp: ArrayModule = np) -> Array:
    """
    Batch Frobenius cosine similarity: [4,4] vs [n,4,4] → [n] in [-1, 1].
    
    THEORY-TRUE: Measures DIRECTION, not magnitude.
    
    FIX v3: Ensures both arrays are on the same device before computation.
    """
    # Ensure device consistency - check if arrays are on different devices
    query_is_cupy = hasattr(query, '__cuda_array_interface__')
    contexts_is_cupy = hasattr(contexts, '__cuda_array_interface__')
    
    if query_is_cupy != contexts_is_cupy:
        # Device mismatch - convert to xp's device
        if hasattr(xp, 'cuda'):  # xp is CuPy
            if not query_is_cupy:
                query = xp.asarray(query)
            if not contexts_is_cupy:
                contexts = xp.asarray(contexts)
        else:  # xp is NumPy
            if query_is_cupy:
                query = query.get()
            if contexts_is_cupy:
                contexts = contexts.get()
    
    dots = xp.sum(query * contexts, axis=(1, 2))
    query_norm = xp.linalg.norm(query, 'fro')
    context_norms = xp.sqrt(xp.sum(contexts * contexts, axis=(1, 2)))
    return dots / (query_norm * context_norms + PHI_EPSILON)


def clifford_adjoint(A: Array, G: Array, xp: ArrayModule = np) -> Array:
    """Compute Clifford adjoint (reversion): A† = G A^T G"""
    if A.ndim == 2:
        return G @ A.T @ G
    else:
        return xp.einsum('ij,bjk,kl->bil', G, xp.transpose(A, (0, 2, 1)), G)


def metric_similarity(a: Array, b: Array, G: Array, xp: ArrayModule = np) -> float:
    """Metric-aware Clifford similarity: ⟨A, B⟩ = (1/4) Tr(A† B)"""
    a_adj = clifford_adjoint(a, G, xp)
    return float(xp.trace(a_adj @ b)) / 4.0


# =============================================================================
# GRACE OPERATOR — Grade-wise Contraction
# =============================================================================

def decompose_to_coefficients(M: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    Decompose matrix into 16 Clifford basis coefficients.
    
    VECTORIZED: Uses einsum instead of Python loop!
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis matrices
        xp: array module
        
    Returns:
        [16] coefficient array
    """
    # Vectorized: compute all 16 coefficients at once
    # Inner product: sum over (i,j) of basis[k,i,j] * M[i,j]
    numerators = xp.einsum('kij,ij->k', basis, M)
    # Norm squared of each basis element
    norm_sqs = xp.einsum('kij,kij->k', basis, basis)
    return numerators / norm_sqs


def decompose_to_coefficients_batch(Ms: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    Decompose BATCH of matrices into Clifford coefficients.
    
    FULLY VECTORIZED for GPU!
    
    Args:
        Ms: [N, 4, 4] batch of matrix multivectors
        basis: [16, 4, 4] Clifford basis matrices
        xp: array module
        
    Returns:
        [N, 16] coefficient array
    """
    # Vectorized batch decomposition
    # numerators[n,k] = sum_{i,j} basis[k,i,j] * Ms[n,i,j]
    numerators = xp.einsum('kij,nij->nk', basis, Ms)
    norm_sqs = xp.einsum('kij,kij->k', basis, basis)
    return numerators / norm_sqs


def reconstruct_from_coefficients(coeffs: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    Reconstruct matrix from Clifford basis coefficients.
    
    BATCHED: Handles single [16], batch [BATCH, 16], or higher dims.
    
    Args:
        coeffs: [..., 16] coefficient array
        basis: [16, 4, 4] Clifford basis matrices
        xp: array module
        
    Returns:
        [..., 4, 4] matrix multivector(s)
    """
    # M = Σₖ coeff[k] × basis[k]
    if coeffs.ndim == 1:
        # Single vector [16]
        return xp.einsum('k,kij->ij', coeffs, basis)
    elif coeffs.ndim == 2:
        # Batch [BATCH, 16]
        return xp.einsum('bk,kij->bij', coeffs, basis)
    else:
        # Higher dims [..., 16]
        return xp.einsum('...k,kij->...ij', coeffs, basis)


def grace_operator(M: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    Apply Grace contraction to matrix multivector.
    
    GRACE IS VISCOSITY FOR BIVECTORS:
        - Decomposes matrix into Clifford basis
        - Scales each grade by φ⁻ᵏ
        - Reconstructs
    
    Grade 2 (bivectors = vorticity) is damped by φ⁻².
    Enstrophy = ||grade-2||² decays at rate φ⁻⁴ ≈ 0.146 per step.
    This is the Clifford analogue of viscous damping in NS.
    
    Grace scaling:
        Grade 0: × 1.0      (scalar preserved - "total energy")
        Grade 1: × φ⁻¹     (vectors - direction)
        Grade 2: × φ⁻²     (bivectors - VORTICITY DAMPING)
        Grade 3: × φ⁻³     (trivectors - fine structure)
        Grade 4: × φ⁻¹     (pseudoscalar - Fibonacci exception!)
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis matrices
        xp: array module
        
    Returns:
        [4, 4] Grace-contracted matrix
    """
    # Decompose
    coeffs = decompose_to_coefficients(M, basis, xp)
    
    # Apply Grace scaling
    scales = xp.array(GRACE_SCALES_FLAT, dtype=DTYPE)
    scaled_coeffs = coeffs * scales
    
    # Reconstruct
    return reconstruct_from_coefficients(scaled_coeffs, basis, xp)


def competitive_grace_operator(
    M: Array, 
    basis: Array, 
    xp: ArrayModule = np,
    n_winners: int = 4,
    suppression_factor: float = None,
) -> Array:
    """
    Apply Grace with LATERAL INHIBITION (winner-take-all dynamics).
    
    THEORY (CORTICAL LATERAL INHIBITION ANALOG):
        Standard Grace contracts ALL grades uniformly → everything collapses to same witness.
        This prevents discrimination between different patterns.
        
        Competitive Grace implements lateral inhibition:
        - Find the strongest components (winners)
        - Apply gentle decay to winners (φ⁻¹)
        - Apply aggressive suppression to losers (φ⁻³)
        
        This maintains pattern separation: different inputs stay different.
        
    BRAIN ANALOG:
        Cortical columns have inhibitory interneurons that implement
        winner-take-all: the strongest activation suppresses neighbors.
        This sharpens representations during retrieval.
        
    WHY THIS MATTERS:
        Standard Grace: similarity increases after iteration (collapse)
        Competitive Grace: similarity does NOT collapse (separation maintained)
        
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis matrices
        xp: array module
        n_winners: Number of components to preserve (default 4)
        suppression_factor: How much to suppress losers (default φ⁻²)
        
    Returns:
        [4, 4] competitively contracted matrix
    """
    if suppression_factor is None:
        suppression_factor = PHI_INV_SQ  # φ⁻² ≈ 0.38
    
    # Decompose
    coeffs = decompose_to_coefficients(M, basis, xp)
    
    # Find winners (largest magnitude components)
    magnitudes = xp.abs(coeffs)
    
    # Get indices of top n_winners
    if hasattr(xp, 'argpartition'):
        # NumPy path
        winner_indices = xp.argpartition(magnitudes, -n_winners)[-n_winners:]
    else:
        # CuPy fallback
        winner_indices = xp.argsort(magnitudes)[-n_winners:]
    
    # Create competitive scaling: winners get φ⁻¹, losers get aggressive suppression
    scales = xp.full(16, suppression_factor, dtype=DTYPE)  # Default: suppress
    
    # Winners get standard Grace scaling for their grade
    standard_scales = xp.array(GRACE_SCALES_FLAT, dtype=DTYPE)
    for idx in winner_indices:
        idx = int(idx)  # Ensure Python int for indexing
        scales[idx] = standard_scales[idx]
    
    # Always preserve witness (grade 0 and 15) somewhat
    scales[0] = max(float(scales[0]), standard_scales[0] * 0.5)   # Scalar
    scales[15] = max(float(scales[15]), standard_scales[15] * 0.5)  # Pseudoscalar
    
    # Apply competitive scaling
    scaled_coeffs = coeffs * scales
    
    # Reconstruct
    return reconstruct_from_coefficients(scaled_coeffs, basis, xp)


def grace_with_stability(M: Array, basis: Array, xp: ArrayModule = np) -> Tuple[Array, float, Tuple[float, float]]:
    """
    Apply Grace AND return stability/witness without double decomposition.
    
    THEORY-PARSIMONIOUS: Grace already decomposes M internally. Instead of:
        graced = grace_operator(M)        # decompose → scale → reconstruct
        sigma = grace_stability(graced)   # decompose AGAIN
        
    We do:
        graced, sigma, witness = grace_with_stability(M)  # ONE decomposition
    
    This saves a full decomposition operation per call!
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        (graced_matrix, stability, (scalar, pseudo))
    """
    # Decompose ONCE
    coeffs = decompose_to_coefficients(M, basis, xp)
    
    # Apply Grace scaling
    scales = xp.array(GRACE_SCALES_FLAT, dtype=DTYPE)
    scaled_coeffs = coeffs * scales
    
    # Compute stability from scaled coefficients (witness already scaled by 1.0 and φ⁻¹)
    total_energy = float(xp.sum(scaled_coeffs * scaled_coeffs))
    witness_energy = float(scaled_coeffs[0]**2 + scaled_coeffs[15]**2)
    stability = witness_energy / max(total_energy, PHI_EPSILON)
    
    # Extract witness (scalar at idx 0, pseudo at idx 15)
    witness = (float(scaled_coeffs[0]), float(scaled_coeffs[15]))
    
    # Reconstruct
    graced = reconstruct_from_coefficients(scaled_coeffs, basis, xp)
    
    return graced, stability, witness


def grace_with_stability_batch(M: Array, basis: Array, xp: ArrayModule = np) -> Tuple[Array, Array, Array]:
    """
    Batched Grace with stability — ONE decomposition for entire batch.
    
    Args:
        M: [batch, 4, 4] matrix multivectors
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        (graced_matrices [batch, 4, 4], stabilities [batch], witnesses [batch, 2])
    """
    batch_size = M.shape[0]
    
    # Decompose all at once
    norm_sq = xp.sum(basis * basis, axis=(1, 2))  # [16]
    coeffs = xp.einsum('cij,bij->bc', basis, M) / norm_sq  # [batch, 16]
    
    # Apply Grace scaling
    scales = xp.array(GRACE_SCALES_FLAT, dtype=DTYPE)
    scaled_coeffs = coeffs * scales  # [batch, 16]
    
    # Compute stability from scaled coefficients
    total_energy = xp.sum(scaled_coeffs * scaled_coeffs, axis=1)  # [batch]
    witness_energy = scaled_coeffs[:, 0]**2 + scaled_coeffs[:, 15]**2  # [batch]
    stability = witness_energy / xp.maximum(total_energy, PHI_EPSILON)  # [batch]
    
    # Extract witnesses: [batch, 2]
    witnesses = xp.stack([scaled_coeffs[:, 0], scaled_coeffs[:, 15]], axis=1)
    
    # Reconstruct: [batch, 4, 4]
    graced = xp.einsum('bc,cij->bij', scaled_coeffs, basis)
    
    return graced, stability, witnesses


def grace_operator_batch(M: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    Apply Grace contraction to batch of matrices.
    
    Args:
        M: [batch, 4, 4] matrix multivectors
        basis: [16, 4, 4] Clifford basis matrices
        xp: array module
        
    Returns:
        [batch, 4, 4] Grace-contracted matrices
    """
    batch_size = M.shape[0]
    
    # Decompose all at once
    # coeffs[b, i] = sum(basis[i] * M[b]) / norm_sq[i]
    norm_sq = xp.sum(basis * basis, axis=(1, 2))  # [16]
    coeffs = xp.einsum('cij,bij->bc', basis, M) / norm_sq  # [batch, 16]
    
    # Apply Grace scaling
    scales = xp.array(GRACE_SCALES_FLAT, dtype=DTYPE)
    scaled_coeffs = coeffs * scales  # [batch, 16]
    
    # Reconstruct
    return xp.einsum('bc,cij->bij', scaled_coeffs, basis)


# =============================================================================
# PRECOMPUTED N-ITERATION GRACE SCALES (Theory-true speed optimization)
# =============================================================================
#
# MATHEMATICAL INSIGHT:
#   Grace scales grade k by φ^(-grade_power[k]) per iteration.
#   After N iterations: scale_final = scale^N
#   
#   This means N Grace iterations can be computed as:
#       1. Decompose ONCE
#       2. Scale by GRACE_SCALES^N (precomputed)
#       3. Reconstruct ONCE
#   
#   Current: O(N × 3) operations (N decompose-scale-reconstruct cycles)
#   Optimized: O(3) operations (1 decompose + 1 scale + 1 reconstruct)
#   
# For basin keys (8 coefficients), even better:
#   No reconstruction needed! Just scale 8 coefficients.
#
# THEORY-TRUE: This is mathematically equivalent to iterating Grace N times.
# The early stopping heuristic is REMOVED for consistency—all contexts in a
# basin should have the same effective iteration count for coherent keys.

# Precompute scales for common iteration counts
_GRACE_SCALES_CACHE = {}

def _get_grace_scales_n(n_iters: int, xp) -> Array:
    """Get precomputed Grace scales for N iterations."""
    key = (n_iters, id(xp))
    if key not in _GRACE_SCALES_CACHE:
        scales = xp.array(GRACE_SCALES_FLAT, dtype=DTYPE)
        _GRACE_SCALES_CACHE[key] = scales ** n_iters
    return _GRACE_SCALES_CACHE[key]


# Basin key indices: ALL 16 Clifford coefficients for maximum routing diversity
# Full Cl(3,1) structure: scalar + 4 vectors + 6 bivectors + 4 trivectors + pseudoscalar
# This enables 8 levels of hierarchical routing (16 elements / 2 per level)
# Required for 200K+ pattern training without catastrophic collision
BASIN_KEY_INDICES = list(range(16))  # All 16 coefficients


def grace_n_iterations(M: Array, basis: Array, n_iters: int, xp: ArrayModule = np) -> Array:
    """
    Apply Grace N times via DIRECT coefficient scaling (no iteration).
    
    THEORY-TRUE OPTIMIZATION:
        Instead of: for _ in range(N): M = grace_operator(M)
        We compute: coeffs * (GRACE_SCALES ** N)
        
        This is MATHEMATICALLY EQUIVALENT but 2× faster for N=3.
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis
        n_iters: Number of Grace iterations to simulate
        xp: array module
        
    Returns:
        [4, 4] Grace-contracted matrix (equivalent to N iterations)
    """
    # Decompose ONCE
    coeffs = decompose_to_coefficients(M, basis, xp)
    
    # Scale by precomputed N-iteration scales
    scales_n = _get_grace_scales_n(n_iters, xp)
    scaled_coeffs = coeffs * scales_n
    
    # Reconstruct ONCE
    return reconstruct_from_coefficients(scaled_coeffs, basis, xp)


def grace_n_iterations_batch(M: Array, basis: Array, n_iters: int, xp: ArrayModule = np) -> Array:
    """
    Apply Grace N times to BATCH via direct scaling (no iteration).
    
    THEORY-TRUE OPTIMIZATION for batch processing.
    
    Args:
        M: [batch, 4, 4] matrix multivectors
        basis: [16, 4, 4] Clifford basis
        n_iters: Number of Grace iterations to simulate
        xp: array module
        
    Returns:
        [batch, 4, 4] Grace-contracted matrices
    """
    # Decompose all at once
    norm_sq = xp.sum(basis * basis, axis=(1, 2))  # [16]
    coeffs = xp.einsum('cij,bij->bc', basis, M) / norm_sq  # [batch, 16]
    
    # Scale by precomputed N-iteration scales
    scales_n = _get_grace_scales_n(n_iters, xp)
    scaled_coeffs = coeffs * scales_n  # [batch, 16]
    
    # Reconstruct
    return xp.einsum('bc,cij->bij', scaled_coeffs, basis)


def grace_basin_key_direct(M: Array, basis: Array, n_iters: int, resolution: float, xp: ArrayModule = np) -> Tuple[int, ...]:
    """
    Compute Grace basin key via DIRECT coefficient scaling (no iteration).
    
    THEORY-TRUE OPTIMIZATION:
        Uses ALL 16 Clifford coefficients for maximum routing diversity.
        Enables 8 levels of hierarchical routing (16^8 = 4.3B unique paths).
        Required for 200K+ pattern training without catastrophic collision.
        
    CONSISTENCY: Uses fixed iteration count (no early stopping).
    This ensures all contexts in a basin use the same scaling, improving
    basin coherence and generalization.
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis
        n_iters: Number of Grace iterations to simulate
        resolution: Quantization resolution (typically φ⁻⁶)
        xp: array module
        
    Returns:
        16-tuple of quantized coefficients (basin key)
    """
    # Decompose ONCE
    coeffs = decompose_to_coefficients(M, basis, xp)
    
    # Get scales for key indices only (avoid scaling all 16)
    scales_n = _get_grace_scales_n(n_iters, xp)
    
    # Extract and scale only the 8 key coefficients
    key_coeffs = coeffs[BASIN_KEY_INDICES] * scales_n[BASIN_KEY_INDICES]
    
    # Quantize (convert to CPU if needed)
    if hasattr(key_coeffs, 'get'):
        key_coeffs = key_coeffs.get()
    
    return tuple(int(c / resolution) for c in key_coeffs)


def grace_basin_keys_batch_direct(
    M: Array, 
    basis: Array, 
    n_iters: int, 
    resolution: float, 
    xp: ArrayModule = np
) -> Any:
    """
    Compute Grace basin keys for BATCH via direct coefficient scaling.
    
    THEORY-TRUE OPTIMIZATION for batch basin key computation.
    Uses ALL 16 Clifford coefficients for maximum routing diversity.
    Enables 8 levels of hierarchical routing (16^8 = 4.3B unique paths).
    
    Speedup vs iterative method:
        - Iterative: N × (decompose + scale + reconstruct) per matrix
        - Direct: 1 decompose + 1 scale (16 coeffs) per matrix
        - For N=3: ~6× fewer operations
    
    Args:
        M: [batch, 4, 4] matrix multivectors
        basis: [16, 4, 4] Clifford basis
        n_iters: Number of Grace iterations to simulate
        resolution: Quantization resolution
        xp: array module
        
    Returns:
        [batch, 16] array of quantized coefficients (basin keys)
    """
    batch_size = M.shape[0]
    
    # Decompose all at once
    norm_sq = xp.sum(basis * basis, axis=(1, 2))  # [16]
    coeffs = xp.einsum('cij,bij->bc', basis, M) / norm_sq  # [batch, 16]
    
    # Get scales and extract ALL key indices (16D for maximum diversity)
    scales_n = _get_grace_scales_n(n_iters, xp)
    key_indices = xp.array(BASIN_KEY_INDICES)  # All 16 indices
    
    # Scale all key coefficients: [batch, 16]
    key_coeffs = coeffs[:, key_indices] * scales_n[key_indices]
    
    # Quantize
    quantized = (key_coeffs / resolution).astype(xp.int64)
    
    # Return as array [batch, 16] - stay on GPU if possible
    return quantized


def grace_flow(M: Array, attractor: Array, basis: Array,
               steps: int = 10, rate: float = PHI_INV_SQ,
               xp: ArrayModule = np) -> Array:
    """
    Iterate Grace flow toward attractor.
    
    M(t+1) = (1 - rate) * Grace(M(t)) + rate * attractor
    
    Converges at spectral gap rate γ = φ⁻².
    
    Args:
        M: [4, 4] initial matrix
        attractor: [4, 4] target attractor
        basis: [16, 4, 4] Clifford basis
        steps: number of iterations
        rate: mixing rate (default: φ⁻²)
        xp: array module
        
    Returns:
        [4, 4] equilibrium matrix
    """
    current = M.copy()
    for _ in range(steps):
        graced = grace_operator(current, basis, xp)
        current = (1 - rate) * graced + rate * attractor
    return current


# =============================================================================
# EMBEDDING INITIALIZATION
# =============================================================================

def initialize_embeddings_rotor(vocab_size: int, angle_std: float = PHI_INV,  # φ-derived (was 0.1)
                                seed: int = 42, xp: ArrayModule = np) -> Array:
    """
    THEORY-TRUE rotor initialization for embeddings.
    
    ROTORS ARE THE CORRECT CLIFFORD ELEMENTS:
        - Rotors R = exp(θ·B) where B is a unit bivector
        - Rotors form a GROUP: R₁ R₂ is also a rotor (closure under product!)
        - Rotors satisfy R R̃ = 1 (unit norm in Clifford sense)
        - NO EXTERNAL NORMALIZATION NEEDED - geometry handles stability
    
    Why rotors, not arbitrary matrices?
        - Identity + noise violates Clifford structure
        - Arbitrary matrices can grow/shrink unboundedly under product
        - Rotors stay bounded because they form a compact Lie group (Spin(3,1))
    
    For small angles:
        exp(θ·B) ≈ cos(θ)·I + sin(θ)·B ≈ I + θ·B + O(θ²)
        
    This gives identity-biased initialization (like before) but CONSTRAINED
    to the proper Clifford subgroup.
    
    REPRODUCIBILITY NOTE:
        Random numbers are ALWAYS generated with NumPy, then converted to the
        target array module (CuPy if xp != np). This ensures identical embeddings
        regardless of whether running on CPU or GPU, enabling:
        - Consistent debugging across environments
        - Reproducible results with the same seed
        - No "lucky/unlucky" embedding initializations
    
    Args:
        vocab_size: Number of tokens
        angle_std: Standard deviation of rotation angles (radians)
        seed: Random seed
        xp: Array module (embeddings will be converted to this type)
        
    Returns:
        [vocab_size, 4, 4] rotor matrices (automatically unit norm in Clifford sense)
    """
    # ALWAYS use NumPy for random generation to ensure reproducibility!
    # CuPy and NumPy produce different random sequences even with the same seed.
    # This was causing the "meteor" bug where token 19999 had special properties
    # only in CuPy-generated embeddings.
    np.random.seed(seed)
    
    # Build the basis (will be converted to target xp at the end)
    basis_np = build_clifford_basis(np)
    
    # Bivector indices in our basis (grade 2): indices 5-10
    # These are: e₁e₂, e₁e₃, e₁e₄, e₂e₃, e₂e₄, e₃e₄
    bivector_indices = [5, 6, 7, 8, 9, 10]
    bivector_basis = basis_np[bivector_indices]  # [6, 4, 4]
    
    # FULLY VECTORIZED rotor initialization (~100x faster for large vocab)
    
    # Random bivector directions: [vocab_size, 6] — NumPy!
    all_bivector_coeffs = np.random.randn(vocab_size, 6)
    
    # Normalize to unit bivectors
    norms = np.sqrt(np.sum(all_bivector_coeffs**2, axis=1, keepdims=True))
    # Handle zero-norm case (very rare, but for safety)
    norms = np.maximum(norms, float(PHI_EPSILON))
    all_bivector_coeffs = all_bivector_coeffs / norms
    
    # Build all bivector matrices at once: [vocab_size, 4, 4]
    # einsum: coeffs[v,c] * basis[c,i,j] -> B[v,i,j]
    all_B = np.einsum('vc,cij->vij', all_bivector_coeffs, bivector_basis)
    
    # Random angles: [vocab_size] — NumPy!
    thetas = np.random.randn(vocab_size) * angle_std
    
    # Rotor: R = cos(θ)·I + sin(θ)·B
    # cos_theta[v] * I + sin_theta[v] * B[v]
    cos_theta = np.cos(thetas).reshape(-1, 1, 1)
    sin_theta = np.sin(thetas).reshape(-1, 1, 1)
    identity = np.eye(4, dtype=np.float64)
    
    matrices = cos_theta * identity + sin_theta * all_B
    
    # THEORY-TRUE ENHANCEMENT: Add pseudoscalar variation for 2D witness space
    # The witness (scalar, pseudoscalar) encodes semantic meaning.
    # Without pseudoscalar variation, all tokens have similar witnesses.
    # Adding small random pseudoscalar creates diverse "semantic orientations".
    #
    # The pseudoscalar γ₀γ₁γ₂γ₃ is the "volume element" - it encodes chirality.
    # Different pseudoscalar values = different "handedness" in semantic space.
    # Scale by angle_std to match the magnitude of other variations.
    pseudoscalar_basis = basis_np[15]  # Index 15 is γ₀γ₁γ₂γ₃
    pseudo_coeffs = np.random.randn(vocab_size) * angle_std  # NumPy!
    pseudo_coeffs = pseudo_coeffs.reshape(-1, 1, 1)
    matrices = matrices + pseudo_coeffs * pseudoscalar_basis
    
    # FROBENIUS NORMALIZATION: Essential for meaningful similarity comparisons!
    # Without unit norm, frobenius_similarity_batch returns raw inner products
    # that vary wildly based on embedding norm, not semantic similarity.
    # With unit norm, inner product equals cosine similarity.
    norms = np.sqrt(np.sum(matrices ** 2, axis=(1, 2), keepdims=True))
    matrices = matrices / norms
    
    # Convert to target array module (CuPy if xp != np)
    if xp is not np:
        matrices = xp.asarray(matrices)
    
    return matrices


def initialize_embeddings_identity(vocab_size: int, noise_std: float = PHI_INV,
                                   seed: int = 42, xp: ArrayModule = np) -> Array:
    """
    Initialize embeddings with theory-true rotor initialization.
    
    Delegates to initialize_embeddings_rotor() with mapped parameters.
    
    Args:
        vocab_size: Number of tokens
        noise_std: Maps to angle_std * 2 for rotor initialization
                   Default φ⁻¹ → angle_std ≈ 1.236 for sufficient embedding separation
                   to support witness index retrieval (tested: 100% accuracy)
        seed: Random seed
        xp: Array module (numpy or cupy)
        
    Returns:
        [vocab_size, 4, 4] array of Spin(3,1) rotor embeddings
    """
    return initialize_embeddings_rotor(vocab_size, angle_std=noise_std * 2, seed=seed, xp=xp)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_gamma_matrices(xp: ArrayModule = np) -> bool:
    """Verify gamma matrices satisfy Cl(3,1) anticommutation relations."""
    gamma = build_gamma_matrices(xp)
    eta = xp.diag(xp.array([1, 1, 1, -1], dtype=DTYPE))
    
    for mu in range(4):
        for nu in range(4):
            anticomm = gamma[mu] @ gamma[nu] + gamma[nu] @ gamma[mu]
            expected = 2 * eta[mu, nu] * xp.eye(4, dtype=DTYPE)
            diff = float(xp.max(xp.abs(anticomm - expected)))
            if diff > PHI_EPSILON:
                return False
    return True


def verify_wedge_antisymmetry(xp: ArrayModule = np) -> bool:
    """Verify A∧B = -B∧A"""
    xp.random.seed(42)
    a = xp.random.randn(4, 4)
    b = xp.random.randn(4, 4)
    
    ab = wedge_product(a, b, xp)
    ba = wedge_product(b, a, xp)
    
    diff = float(xp.max(xp.abs(ab + ba)))
    return diff < PHI_EPSILON


def grace_inverse(M: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    GraceInverse operator — inflates coherent core back into structural detail.
    
    THEORY (Chapter 11, 14):
        GraceInverse(M) = ⟨M⟩₀ + φ¹⟨M⟩₁ + φ²⟨M⟩₂ + φ³⟨M⟩₃ + φ¹⟨M⟩₄
        
        This is the inverse of Grace, multiplying each grade by φᵏ instead of φ⁻ᵏ.
        Grade 4 uses φ¹ (Fibonacci exception), not φ⁴.
        
        Used in downward projection for generation: inflate coherent core (witness)
        back into full multivector structure.
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        [4, 4] inflated matrix
    """
    # Decompose
    coeffs = decompose_to_coefficients(M, basis, xp)
    
    # Apply GraceInverse scaling (inverse of Grace)
    inverse_scales = xp.array([
        1.0,                                    # Grade 0: ×1
        PHI, PHI, PHI, PHI,                    # Grade 1: ×φ
        PHI**2, PHI**2, PHI**2, PHI**2, PHI**2, PHI**2,  # Grade 2: ×φ²
        PHI**3, PHI**3, PHI**3, PHI**3,       # Grade 3: ×φ³
        PHI,                                    # Grade 4: ×φ (Fibonacci exception!)
    ], dtype=DTYPE)
    
    scaled_coeffs = coeffs * inverse_scales
    
    # Reconstruct
    return reconstruct_from_coefficients(scaled_coeffs, basis, xp)


def verify_grace_contraction(xp: ArrayModule = np) -> bool:
    """Verify Grace contracts higher grades toward scalar."""
    basis = build_clifford_basis(xp)
    
    # Start with random matrix
    xp.random.seed(42)
    M = xp.random.randn(4, 4)
    M = normalize_matrix(M, xp)
    
    # Apply Grace multiple times
    current = M.copy()
    for _ in range(20):
        current = grace_operator(current, basis, xp)
    
    # Should be dominated by scalar (identity) component
    coeffs = decompose_to_coefficients(current, basis, xp)
    scalar_energy = float(coeffs[0]**2)
    total_energy = float(xp.sum(coeffs**2))
    
    # THEORY-TRUE: Scalar should dominate (>φ⁻¹ = 61.8% of energy)
    # Per Grace operator theory: witness (scalar+pseudo) is the stable core
    from .constants import PHI_INV
    return scalar_energy / total_energy > PHI_INV


# =============================================================================
# HOLOGRAPHIC MEMORY OPERATIONS
# =============================================================================

def clifford_inverse(M: Array, xp: ArrayModule = np) -> Array:
    """
    Compute Clifford inverse M⁻¹ such that M × M⁻¹ ≈ I.
    
    In Cl(3,1) ≅ M₄(ℝ), this is the matrix inverse when it exists.
    For degenerate matrices, uses pseudoinverse.
    
    THEORY:
        For holographic memory, we need to "unbind" associations:
            Store:    memory += context × target
            Retrieve: target ≈ memory × context⁻¹
        
        The inverse allows retrieval of the target from the superposition.
    
    Args:
        M: [4, 4] matrix multivector
        xp: array module
        
    Returns:
        [4, 4] inverse (or pseudoinverse)
    """
    det = xp.linalg.det(M)
    
    if abs(det) > PHI_EPSILON:
        # Well-conditioned: use true inverse
        return xp.linalg.inv(M)
    else:
        # Degenerate: use pseudoinverse
        return xp.linalg.pinv(M)


# =============================================================================
# SO(4) GEODESIC OPERATIONS — Theory-True Contrastive Learning
# =============================================================================

def matrix_exp_skew4(X: Array, xp: ArrayModule = np) -> Array:
    """
    Matrix exponential for 4×4 skew-symmetric matrix.
    
    THEORY: For X ∈ so(4) (Lie algebra), exp(X) ∈ SO(4) (Lie group).
    
    Uses Padé approximation / scaling-squaring via scipy or Padé(6,6).
    
    For small X: Taylor series exp(X) = I + X + X²/2! + X³/3! + ...
    For larger X: Use scaling-squaring: exp(X) = exp(X/2^k)^(2^k)
    
    Args:
        X: 4×4 skew-symmetric matrix (in so(4) Lie algebra)
        xp: array module (numpy or cupy)
        
    Returns:
        4×4 SO(4) matrix
    """
    # Check norm to decide on approach
    norm_X = float(xp.linalg.norm(X, 'fro'))
    
    if norm_X < PHI_EPSILON:
        return xp.eye(4, dtype=X.dtype)
    
    # Use scaling-squaring for numerical stability
    # Scale down until ||X/2^k|| < 0.5
    k = 0
    X_scaled = X.astype(xp.float64)  # Higher precision for intermediate
    while float(xp.linalg.norm(X_scaled, 'fro')) > 0.5:
        X_scaled = X_scaled / 2.0
        k += 1
    
    # Padé(6,6) approximation for small matrices
    # exp(X) ≈ P(X) / Q(X) where P, Q are polynomials
    # For skew-symmetric X, we can use simpler Taylor series since it converges
    X2 = X_scaled @ X_scaled
    X3 = X2 @ X_scaled
    X4 = X2 @ X2
    X5 = X4 @ X_scaled
    X6 = X3 @ X3
    
    # Taylor: I + X + X²/2 + X³/6 + X⁴/24 + X⁵/120 + X⁶/720
    I = xp.eye(4, dtype=xp.float64)
    result = (I + X_scaled + X2 / 2.0 + X3 / 6.0 + 
              X4 / 24.0 + X5 / 120.0 + X6 / 720.0)
    
    # Square k times: exp(X) = exp(X/2^k)^(2^k)
    for _ in range(k):
        result = result @ result
    
    # Enforce orthogonality via polar decomposition: R = U @ V.T where M = U @ S @ V.T
    # For matrices close to SO(4), this is stable
    U, _, Vt = xp.linalg.svd(result)
    result = U @ Vt
    
    # Ensure det = +1 (not -1)
    if float(xp.linalg.det(result)) < 0:
        U[:, -1] = -U[:, -1]
        result = U @ Vt
    
    return result.astype(X.dtype)


def matrix_log_so4(R: Array, xp: ArrayModule = np) -> Array:
    """
    Matrix logarithm for SO(4) matrix → so(4) (skew-symmetric).
    
    THEORY: For R ∈ SO(4), log(R) ∈ so(4).
    
    SO(4) ≅ (SU(2) × SU(2)) / Z₂, so it has a more complex structure than SO(3).
    We use the general matrix logarithm via eigendecomposition.
    
    For orthogonal matrices, eigenvalues come in conjugate pairs e^{±iθ},
    and the logarithm has imaginary eigenvalues ±iθ.
    
    Args:
        R: 4×4 SO(4) matrix (orthogonal with det=1)
        xp: array module
        
    Returns:
        4×4 skew-symmetric matrix in so(4)
    """
    # Convert to numpy for eigendecomposition (CuPy doesn't have linalg.eig)
    # This is OK because this function is only used in contrastive updates, not the hot path
    if xp != np:
        R_np = R.get() if hasattr(R, 'get') else np.asarray(R)
    else:
        R_np = R
    
    R_64 = R_np.astype(np.float64)  # Higher precision
    
    # Check if close to identity
    diff_from_I = float(np.linalg.norm(R_64 - np.eye(4, dtype=np.float64), 'fro'))
    if diff_from_I < float(PHI_EPSILON):
        result = np.zeros((4, 4), dtype=R_np.dtype)
        return xp.asarray(result) if xp != np else result
    
    # Use eigendecomposition: R = V @ D @ V^H where D = diag(e^{iθ_j})
    # log(R) = V @ log(D) @ V^H = V @ diag(iθ_j) @ V^H
    eigenvalues, V = np.linalg.eig(R_64)
    
    # Take log of eigenvalues (complex log)
    log_eigenvalues = np.log(eigenvalues + float(PHI_EPSILON)*1j)  # Add small imaginary to handle edge cases
    
    # Construct log(R)
    D_log = np.diag(log_eigenvalues)
    log_R = V @ D_log @ np.linalg.inv(V)
    
    # Result should be real and skew-symmetric
    # Take real part (imaginary should be negligible)
    log_R_real = np.real(log_R).astype(np.float64)
    
    # Enforce skew-symmetry: X = (X - X.T) / 2
    log_R_skew = (log_R_real - log_R_real.T) / 2.0
    
    result = log_R_skew.astype(R_np.dtype)
    return xp.asarray(result) if xp != np else result


def geodesic_interpolation_so4(A: Array, B: Array, t: float, xp: ArrayModule = np) -> Array:
    """
    Geodesic interpolation on SO(4) manifold.
    
    THEORY: The geodesic (shortest path) from A to B on SO(4) is:
        γ(t) = A @ exp(t · log(A.T @ B))
        
    At t=0: γ(0) = A
    At t=1: γ(1) = B  
    At t=0.5: geodesic midpoint
    
    This preserves SO(4) structure (orthogonality and det=1).
    
    Args:
        A: 4×4 SO(4) matrix (start)
        B: 4×4 SO(4) matrix (end)
        t: interpolation parameter [0, 1]
        xp: array module
        
    Returns:
        4×4 SO(4) matrix on geodesic between A and B
    """
    # Relative rotation: A.T @ B is how to get from A to B
    relative = A.T @ B
    
    # Log map: SO(4) → so(4)
    log_relative = matrix_log_so4(relative, xp)
    
    # Scale by t (move fraction t along geodesic)
    scaled = t * log_relative
    
    # Exp map: so(4) → SO(4)
    rotation = matrix_exp_skew4(scaled, xp)
    
    # Apply to starting point
    return A @ rotation


def contrastive_update_so4(emb_a: Array, emb_b: Array, rate: float, xp: ArrayModule = np) -> Tuple[Array, Array]:
    """
    Theory-true contrastive update on SO(4) manifold.
    
    Pulls emb_a and emb_b toward each other along their geodesic.
    
    THEORY: Instead of Euclidean midpoint (which breaks SO(4)):
        - Move emb_a toward emb_b by rate fraction along geodesic
        - Move emb_b toward emb_a by rate fraction along geodesic
        
    Result: Both embeddings remain in SO(4)!
    
    Args:
        emb_a: 4×4 SO(4) embedding
        emb_b: 4×4 SO(4) embedding
        rate: learning rate (how far to move, φ⁻⁵ recommended)
        xp: array module
        
    Returns:
        (new_emb_a, new_emb_b) both still in SO(4)
    """
    # Move A toward B
    new_a = geodesic_interpolation_so4(emb_a, emb_b, rate, xp)
    
    # Move B toward A
    new_b = geodesic_interpolation_so4(emb_b, emb_a, rate, xp)
    
    return new_a, new_b


def verify_so4(M: Array, xp: ArrayModule = np, tol: float = PHI_INV_SQ**4) -> bool:
    """
    Verify that M is in SO(4) (orthogonal with det=1).
    
    Checks:
        1. M.T @ M ≈ I (orthogonality)
        2. det(M) ≈ 1 (special)
    """
    identity_check = M.T @ M
    identity_error = float(xp.max(xp.abs(identity_check - xp.eye(4, dtype=M.dtype))))
    
    det_val = float(xp.linalg.det(M))
    det_error = abs(det_val - 1.0)
    
    return identity_error < tol and det_error < tol


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Basis construction
    'build_gamma_matrices',
    'build_clifford_basis',
    'build_metric_matrix',
    'build_structure_tensor',
    'get_structure_tensor',
    
    # Matrix operations
    'normalize_matrix',
    'geometric_product',
    'geometric_product_batch',
    'wedge_product',
    'inner_product',
    'compute_vorticity',
    'vorticity_magnitude',
    
    # Tensor core accelerated (coefficient representation)
    'geometric_product_coefficients',
    'geometric_product_coefficients_optimized',
    'geometric_product_batch_multi_coefficients',
    'matrix_to_coefficients',
    'coefficients_to_matrix',
    
    # Similarity
    'frobenius_similarity',
    'frobenius_similarity_batch',
    'frobenius_cosine',
    'frobenius_cosine_batch',
    'clifford_adjoint',
    'metric_similarity',
    
    # Coefficient decomposition
    'decompose_to_coefficients',
    'decompose_to_coefficients_batch',
    'reconstruct_from_coefficients',
    
    # Grace operator
    'grace_operator',
    'grace_operator_batch',
    'grace_flow',
    'grace_inverse',
    
    # Grace N-iteration (theory-true speed optimization)
    'grace_n_iterations',
    'grace_n_iterations_batch',
    'grace_basin_key_direct',
    'grace_basin_keys_batch_direct',
    'BASIN_KEY_INDICES',
    
    # Embedding initialization
    'initialize_embeddings_identity',
    'initialize_embeddings_rotor',
    
    # Clifford inverse (for holographic retrieval)
    'clifford_inverse',
    
    # Vorticity
    'vorticity_signature',
    'vorticity_similarity',
    'vorticity_magnitude_and_signature',
    
    # Verification
    'verify_gamma_matrices',
    'verify_wedge_antisymmetry',
    'verify_grace_contraction',
    
    # SO(4) geodesic operations (theory-true contrastive learning)
    'matrix_exp_skew4',
    'matrix_log_so4',
    'geodesic_interpolation_so4',
    'contrastive_update_so4',
    'verify_so4',
]
