"""
Quotient Structure for SCCMU — Gauge Invariance and Normal Forms
================================================================

This module implements the quotient space structure that removes nuisance
degrees of freedom (frame orientation) while preserving semantic content.

KEY INSIGHT:
    The witness subspace (scalar + pseudoscalar) is INVARIANT under Spin(3)
    spatial rotations. By quotienting out these rotations, we:
    - Remove 3 degrees of freedom (orientation)
    - Stabilize same-target similarity
    - Reduce epoch-to-epoch oscillation

MATHEMATICAL STRUCTURE:
    - Gauge group: G_W = Spin(3) ⊂ Spin(3,1) (spatial rotations only)
    - Action: M → R M R̃  (sandwich conjugation)
    - Invariant: W(M) = scalar + φ⁻¹ · pseudoscalar (witness)
    - Normal form: Align "magnetic bivector" (e₂₃, e₃₁, e₁₂) to +z axis

SIGNATURE NOTE:
    We use Cl(3,1) with (+,+,+,-): e₁²=e₂²=e₃²=+1, e₄²=-1
    Spatial bivectors: e₁₂ (idx 5), e₁₃ (idx 6), e₂₃ (idx 8)
    In matrix form, these are 4×4 real matrices.
"""

import numpy as np
from typing import Tuple, List, Dict, Callable, Optional
from .constants import PHI, PHI_INV, PHI_INV_SQ

# Type aliases
Array = np.ndarray
ArrayModule = type(np)


# =============================================================================
# GRADE STRUCTURE (for 16-component multivector extraction)
# =============================================================================

# Grade indices in 16D coefficient representation
GRADE_IDXS = {
    0: [0],                     # scalar
    1: [1, 2, 3, 4],            # vectors
    2: [5, 6, 7, 8, 9, 10],     # bivectors
    3: [11, 12, 13, 14],        # trivectors
    4: [15],                    # pseudoscalar
}

# Witness indices: scalar + pseudoscalar
IDX_WITNESS = [0, 15]

# Core content: bivectors + trivectors (most of the semantic structure)
IDX_CORE = list(range(5, 15))  # indices 5-14

# Fiber: vectors only (low weight)
IDX_FIBER = [1, 2, 3, 4]

# Spatial bivector indices in our Cl(3,1) basis ordering
# e₁₂ (idx 5), e₁₃ (idx 6), e₂₃ (idx 8)
IDX_SPATIAL_BV = [5, 6, 8]  # corresponds to "magnetic" bivector


# =============================================================================
# WITNESS EXTRACTION
# =============================================================================

def extract_witness_coeffs(coeffs: Array) -> Array:
    """
    Extract witness (scalar + pseudoscalar) from 16D coefficient vector.
    
    The witness is the gauge-invariant "self-pointer" subspace.
    
    Args:
        coeffs: [16] Clifford coefficients
        
    Returns:
        [2] array: (scalar_coeff, pseudoscalar_coeff)
    """
    return np.array([coeffs[0], coeffs[15]], dtype=np.float64)


def extract_witness_matrix(M: Array, basis: Array, xp: ArrayModule = np) -> Tuple[float, float]:
    """
    Extract witness (scalar + pseudoscalar) from 4×4 matrix representation.
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis matrices
        xp: array module
        
    Returns:
        (scalar_coeff, pseudoscalar_coeff)
    """
    # Project onto scalar (identity)
    scalar = float(xp.sum(basis[0] * M) / xp.sum(basis[0] * basis[0]))
    # Project onto pseudoscalar
    pseudo = float(xp.sum(basis[15] * M) / xp.sum(basis[15] * basis[15]))
    return scalar, pseudo


def witness_similarity(M1: Array, M2: Array, basis: Array, 
                       xp: ArrayModule = np, eps: float = 1e-12) -> float:
    """
    Compute similarity of witness components only.
    
    This is gauge-invariant — identical before and after Spin(3) rotation.
    
    Args:
        M1, M2: [4, 4] matrix multivectors
        basis: [16, 4, 4] Clifford basis
        xp: array module
        eps: numerical stability
        
    Returns:
        Cosine similarity of witness vectors
    """
    w1 = np.array(extract_witness_matrix(M1, basis, xp))
    w2 = np.array(extract_witness_matrix(M2, basis, xp))
    
    n1 = np.sqrt(np.dot(w1, w1) + eps)
    n2 = np.sqrt(np.dot(w2, w2) + eps)
    
    return float(np.dot(w1, w2) / (n1 * n2))


# =============================================================================
# SPIN(3) ROTORS — Spatial Rotations That Fix Witness
# =============================================================================

def spin3_rotor_matrix(axis: Array, theta: float, basis: Array, 
                       xp: ArrayModule = np) -> Array:
    """
    Build Spin(3) rotor as 4×4 matrix via exponential map.
    
    R = exp(-θ/2 · B) where B = n·(e₂₃, e₃₁, e₁₂)
    
    For spatial bivectors, B² = -|n|² (negative), so:
    R = cos(θ/2) - B·sin(θ/2)
    
    Args:
        axis: [3] unit rotation axis (nx, ny, nz)
        theta: rotation angle in radians
        basis: [16, 4, 4] Clifford basis matrices
        xp: array module
        
    Returns:
        [4, 4] rotor matrix
    """
    axis = xp.asarray(axis, dtype=xp.float64)
    axis = axis / (xp.sqrt(xp.sum(axis**2)) + 1e-12)
    nx, ny, nz = float(axis[0]), float(axis[1]), float(axis[2])
    
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    
    # Spatial bivector matrices: e₁₂ (idx 5), e₁₃ (idx 6), e₂₃ (idx 8)
    # Note: e₃₁ = -e₁₃, so we use nx*e₂₃ + ny*e₃₁ + nz*e₁₂
    # In our convention: e₃₁ = -basis[6] (e₁₃)
    
    # B = nx * e₂₃ + ny * e₃₁ + nz * e₁₂
    #   = nx * basis[8] - ny * basis[6] + nz * basis[5]
    B = nx * basis[8] - ny * basis[6] + nz * basis[5]
    
    # R = cos(θ/2) · I - sin(θ/2) · B
    R = c * basis[0] - s * B
    
    return R


def random_spin3_rotor(basis: Array, rng: np.random.Generator,
                       xp: ArrayModule = np) -> Array:
    """
    Sample a random Spin(3) rotor.
    
    Args:
        basis: [16, 4, 4] Clifford basis
        rng: numpy random generator
        xp: array module
        
    Returns:
        [4, 4] random rotor matrix
    """
    # Random unit axis
    v = rng.normal(size=3)
    v = v / (np.linalg.norm(v) + 1e-12)
    
    # Random angle
    theta = float(rng.uniform(0, 2 * np.pi))
    
    return spin3_rotor_matrix(v, theta, basis, xp)


def sandwich(R: Array, M: Array, xp: ArrayModule = np) -> Array:
    """
    Apply rotor sandwich: M' = R M R̃
    
    For real matrices, reversion of a rotor R built from bivectors is R^T.
    But actually, for our Spin(3) rotors: R̃ = R^T (transpose works).
    
    More precisely: R R̃ = I for unit rotors, and R̃ = R⁻¹.
    For orthogonal rotors, R⁻¹ = R^T.
    
    Args:
        R: [4, 4] rotor matrix
        M: [4, 4] multivector matrix
        xp: array module
        
    Returns:
        [4, 4] rotated multivector
    """
    # For Spin(3) rotors built as above, R is orthogonal, so R̃ = R^T = R⁻¹
    return R @ M @ R.T


# =============================================================================
# NORMAL FORM — Canonicalize by Aligning Spatial Bivector to +z
# =============================================================================

def extract_spatial_bivector(M: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    Extract the "magnetic" spatial bivector components (e₂₃, e₃₁, e₁₂).
    
    This is the part that rotates under Spin(3) gauge.
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        [3] array: (coeff_e23, coeff_e31, coeff_e12)
    """
    # Project onto each spatial bivector
    # e₂₃ is basis[8]
    c_e23 = float(xp.sum(basis[8] * M) / xp.sum(basis[8] * basis[8]))
    # e₃₁ = -e₁₃, so e₃₁ coeff = -e₁₃ coeff; basis[6] is e₁₃
    c_e31 = -float(xp.sum(basis[6] * M) / xp.sum(basis[6] * basis[6]))
    # e₁₂ is basis[5]
    c_e12 = float(xp.sum(basis[5] * M) / xp.sum(basis[5] * basis[5]))
    
    return np.array([c_e23, c_e31, c_e12], dtype=np.float64)


def align_rotor_to_z(b_vec: Array, basis: Array, xp: ArrayModule = np, 
                     eps: float = 1e-9) -> Array:
    """
    Compute rotor that aligns spatial bivector b_vec to +z direction.
    
    Args:
        b_vec: [3] spatial bivector components (e23, e31, e12)
        basis: [16, 4, 4] Clifford basis
        xp: array module
        eps: numerical tolerance
        
    Returns:
        [4, 4] alignment rotor
    """
    nb = np.linalg.norm(b_vec)
    if nb < eps:
        # Near zero — return identity
        return basis[0].copy()
    
    b = b_vec / nb
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    dot = float(np.clip(np.dot(b, z), -1.0, 1.0))
    theta = float(np.arccos(dot))
    
    axis = np.cross(b, z)
    na = np.linalg.norm(axis)
    
    if na < eps:
        # Already aligned or exactly opposite
        if dot > 0:
            return basis[0].copy()  # Identity
        else:
            # 180° rotation around x-axis
            return spin3_rotor_matrix([1, 0, 0], np.pi, basis, xp)
    
    axis = axis / na
    return spin3_rotor_matrix(axis, theta, basis, xp)


def extract_electric_bivector(M: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    Extract the "electric" timelike bivector components (e₀₁, e₀₂, e₀₃).
    
    In our Cl(3,1), these are e₁₄, e₂₄, e₃₄ (involving the timelike e₄).
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        [3] array: (coeff_e14, coeff_e24, coeff_e34)
    """
    # Timelike bivectors: e₁e₄ (idx 7), e₂e₄ (idx 9), e₃e₄ (idx 10)
    c_e14 = float(xp.sum(basis[7] * M) / xp.sum(basis[7] * basis[7]))
    c_e24 = float(xp.sum(basis[9] * M) / xp.sum(basis[9] * basis[9]))
    c_e34 = float(xp.sum(basis[10] * M) / xp.sum(basis[10] * basis[10]))
    
    return np.array([c_e14, c_e24, c_e34], dtype=np.float64)


def align_rotor_around_z(e_vec: Array, basis: Array, xp: ArrayModule = np,
                         eps: float = 1e-9) -> Array:
    """
    Compute rotor that rotates around z-axis to align e_vec's xy-projection to +x.
    
    Args:
        e_vec: [3] vector (only xy components used)
        basis: [16, 4, 4] Clifford basis
        xp: array module
        eps: numerical tolerance
        
    Returns:
        [4, 4] z-axis rotation rotor
    """
    # Project to xy plane
    ex, ey = e_vec[0], e_vec[1]
    r = np.sqrt(ex**2 + ey**2)
    
    if r < eps:
        return basis[0].copy()  # Identity
    
    # Angle from +x axis
    theta = np.arctan2(ey, ex)
    
    # Rotate by -theta around z (using e12)
    return spin3_rotor_matrix([0, 0, 1], -theta, basis, xp)


def normal_form(M: Array, basis: Array, xp: ArrayModule = np) -> Tuple[Array, Array]:
    """
    Compute normal form by fully gauge-fixing Spin(3).
    
    Two-step alignment:
    1. Align "magnetic" bivector (e₂₃, e₃₁, e₁₂) to +z direction
    2. Align "electric" bivector's xy-projection to +x direction
    
    This removes all 3 rotational degrees of freedom, making
    similarity comparison fully gauge-invariant.
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        (NF_M, R): normalized form and the total alignment rotor used
    """
    # Step 1: Align magnetic bivector to +z
    b_vec = extract_spatial_bivector(M, basis, xp)
    R1 = align_rotor_to_z(b_vec, basis, xp)
    M1 = sandwich(R1, M, xp)
    
    # Step 2: Align electric bivector's xy-projection to +x
    e_vec = extract_electric_bivector(M1, basis, xp)
    R2 = align_rotor_around_z(e_vec, basis, xp)
    M2 = sandwich(R2, M1, xp)
    
    # Total rotor is R2 @ R1 (applied in order: first R1, then R2)
    R_total = R2 @ R1
    
    return M2, R_total


# =============================================================================
# QUOTIENT-AWARE SIMILARITY
# =============================================================================

def frobenius_sim(A: Array, B: Array, xp: ArrayModule = np, eps: float = 1e-12) -> float:
    """Simple Frobenius similarity (cosine on flattened matrices)."""
    a = A.flatten()
    b = B.flatten()
    na = np.sqrt(np.dot(a, a) + eps)
    nb = np.sqrt(np.dot(b, b) + eps)
    return float(np.dot(a, b) / (na * nb))


def quotient_similarity(M1: Array, M2: Array, basis: Array,
                        alpha: float = 0.25,  # witness weight
                        beta: float = 0.65,   # core (NF) weight
                        gamma: float = 0.10,  # fiber weight
                        xp: ArrayModule = np,
                        eps: float = 1e-12) -> float:
    """
    Quotient-aware similarity with three components:
    
    1. Witness (invariant anchor): scalar + pseudoscalar
    2. Core (canonicalized): normal form of full matrix
    3. Fiber (raw residual): original matrix (low weight)
    
    Args:
        M1, M2: [4, 4] matrix multivectors
        basis: [16, 4, 4] Clifford basis
        alpha, beta, gamma: weights for witness, core, fiber
        xp: array module
        eps: numerical stability
        
    Returns:
        Combined similarity in [-1, 1]
    """
    # Witness similarity (invariant)
    s_witness = witness_similarity(M1, M2, basis, xp, eps)
    
    # Core similarity (on normal forms)
    NF1, _ = normal_form(M1, basis, xp)
    NF2, _ = normal_form(M2, basis, xp)
    s_core = frobenius_sim(NF1, NF2, xp, eps)
    
    # Fiber similarity (raw, low weight)
    s_fiber = frobenius_sim(M1, M2, xp, eps)
    
    # Combine
    total = alpha + beta + gamma
    return float((alpha * s_witness + beta * s_core + gamma * s_fiber) / total)


def quotient_similarity_phi(M1: Array, M2: Array, basis: Array,
                            xp: ArrayModule = np,
                            eps: float = 1e-12) -> float:
    """
    φ-weighted quotient similarity.
    
    Weights higher grades more heavily (they carry more semantic content).
    
    Args:
        M1, M2: [4, 4] matrix multivectors
        basis: [16, 4, 4] Clifford basis
        xp: array module
        eps: numerical stability
        
    Returns:
        Combined similarity
    """
    # φ-based weights
    w_witness = 1.0
    w_core = PHI           # grade-2/3 content
    w_fiber = PHI_INV_SQ   # grade-1 (lower importance)
    
    return quotient_similarity(M1, M2, basis,
                               alpha=w_witness,
                               beta=w_core,
                               gamma=w_fiber,
                               xp=xp, eps=eps)


# =============================================================================
# DIAGNOSTICS: SEPARATION METRIC
# =============================================================================

def compute_separation(context_reps: List[Array],
                       targets: List[int],
                       sim_fn: Callable[[Array, Array], float],
                       max_diff_pairs: int = 50) -> Dict[str, float]:
    """
    Compute same-target vs diff-target separation.
    
    This is THE canonical metric for semantic learning:
    - Same-target similarity > Diff-target similarity ⟺ learned semantic partition
    
    Args:
        context_reps: List of [4, 4] context matrices
        targets: List of target token IDs
        sim_fn: Similarity function (M1, M2) -> float
        max_diff_pairs: Max diff-target pairs to sample per context (for speed)
        
    Returns:
        Dict with same_mean, diff_mean, separation, counts
    """
    from collections import defaultdict
    from itertools import combinations
    
    N = len(context_reps)
    assert N == len(targets), "Mismatch between contexts and targets"
    
    # Group by target
    by_target = defaultdict(list)
    for i, t in enumerate(targets):
        by_target[t].append(i)
    
    same_sims = []
    diff_sims = []
    
    # Same-target pairs
    for t, idxs in by_target.items():
        if len(idxs) < 2:
            continue
        for i, j in combinations(idxs, 2):
            same_sims.append(sim_fn(context_reps[i], context_reps[j]))
    
    # Diff-target pairs (sampled for speed)
    for i in range(N):
        ti = targets[i]
        cnt = 0
        for j in range(N):
            if targets[j] == ti or j == i:
                continue
            diff_sims.append(sim_fn(context_reps[i], context_reps[j]))
            cnt += 1
            if cnt >= max_diff_pairs:
                break
    
    same = float(np.mean(same_sims)) if same_sims else float("nan")
    diff = float(np.mean(diff_sims)) if diff_sims else float("nan")
    sep = same - diff if not (np.isnan(same) or np.isnan(diff)) else float("nan")
    
    return {
        "same_mean": same,
        "diff_mean": diff,
        "separation": sep,
        "same_n": len(same_sims),
        "diff_n": len(diff_sims),
        "targets_n": len(by_target),
    }


# =============================================================================
# INVARIANCE TESTS
# =============================================================================

def test_witness_invariance(basis: Array, xp: ArrayModule = np, 
                            n_tests: int = 100, seed: int = 42) -> bool:
    """
    Test that witness is invariant under Spin(3) gauge.
    
    Args:
        basis: [16, 4, 4] Clifford basis
        xp: array module
        n_tests: number of random tests
        seed: random seed
        
    Returns:
        True if all tests pass
    """
    rng = np.random.default_rng(seed)
    
    # Random multivector
    M = rng.normal(size=(4, 4))
    w0 = np.array(extract_witness_matrix(M, basis, xp))
    
    for _ in range(n_tests):
        R = random_spin3_rotor(basis, rng, xp)
        M_rot = sandwich(R, M, xp)
        w1 = np.array(extract_witness_matrix(M_rot, basis, xp))
        
        if np.max(np.abs(w1 - w0)) > 1e-5:
            print(f"Witness changed: {w0} → {w1}")
            return False
    
    return True


def test_normal_form_invariance(basis: Array, xp: ArrayModule = np,
                                n_tests: int = 100, seed: int = 42) -> bool:
    """
    Test that NF(R·M·R̃) ≈ NF(M) for any gauge R.
    
    Args:
        basis: [16, 4, 4] Clifford basis
        xp: array module
        n_tests: number of tests
        seed: random seed
        
    Returns:
        True if all tests pass
    """
    rng = np.random.default_rng(seed)
    
    # Random multivector
    M = rng.normal(size=(4, 4))
    NF0, _ = normal_form(M, basis, xp)
    
    for _ in range(n_tests):
        R = random_spin3_rotor(basis, rng, xp)
        M_rot = sandwich(R, M, xp)
        NF1, _ = normal_form(M_rot, basis, xp)
        
        sim = frobenius_sim(NF0, NF1, xp)
        if sim < 0.99:
            print(f"NF not invariant: sim={sim}")
            return False
    
    return True


# =============================================================================
# BINDING OPERATOR — Content Relative to Witness
# =============================================================================

def witness_pointer(M: Array, basis: Array, xp: ArrayModule = np, 
                    eps: float = 1e-12) -> Array:
    """
    Compute normalized witness pointer as a matrix.
    
    The witness pointer is the unit-norm projection onto the witness subspace
    (scalar + φ⁻¹ · pseudoscalar), re-embedded as a matrix.
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis
        xp: array module
        eps: numerical stability
        
    Returns:
        [4, 4] unit-norm witness pointer matrix
    """
    s, p = extract_witness_matrix(M, basis, xp)
    
    # Witness = scalar + φ⁻¹ · pseudoscalar (SCCMU convention)
    W = s * basis[0] + PHI_INV * p * basis[15]
    
    # Normalize
    norm = xp.sqrt(xp.sum(W**2) + eps)
    return W / norm


def extract_content(M: Array, basis: Array, xp: ArrayModule = np) -> Array:
    """
    Extract content (non-witness) part of multivector.
    
    Content = M - projection_onto_witness(M)
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        [4, 4] content matrix
    """
    s, p = extract_witness_matrix(M, basis, xp)
    W = s * basis[0] + p * basis[15]
    return M - W


def bind(M: Array, basis: Array, xp: ArrayModule = np,
         lmbda: float = PHI_INV) -> Array:
    """
    Apply binding operator: make content relative to witness.
    
    𝓑(M) = W(M) + λ · w · C(M) · w̃
    
    where:
        W(M) = witness part (scalar + pseudoscalar)
        C(M) = content part (grades 1-3)
        w = normalized witness pointer
        w̃ = reversion of w (for real matrices, w̃ = w^T)
        λ = binding strength (default: φ⁻¹)
    
    The binding operation "frames" the content relative to the witness,
    making the representation self-referential.
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis
        xp: array module
        lmbda: binding strength (default: φ⁻¹ for SCCMU consistency)
        
    Returns:
        [4, 4] bound matrix
    """
    # Extract witness and content
    s, p = extract_witness_matrix(M, basis, xp)
    W = s * basis[0] + p * basis[15]
    C = M - W
    
    # Witness pointer (normalized)
    w = witness_pointer(M, basis, xp)
    
    # Sandwich: w · C · w^T
    # For our real symmetric witness, w^T ≈ w̃
    bound_content = w @ C @ w.T
    
    # Combine: witness + λ · bound_content
    return W + lmbda * bound_content


# =============================================================================
# GRADE-WISE VARIANCE TRACKING
# =============================================================================

def project_to_grade(M: Array, grade: int, basis: Array, 
                     xp: ArrayModule = np) -> Array:
    """
    Project matrix onto specific grade.
    
    Args:
        M: [4, 4] matrix multivector
        grade: 0, 1, 2, 3, or 4
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        [4, 4] grade-k projection
    """
    idxs = GRADE_IDXS[grade]
    result = xp.zeros((4, 4), dtype=xp.float64)
    
    for i in idxs:
        # Project onto basis[i]
        coeff = xp.sum(basis[i] * M) / xp.sum(basis[i] * basis[i])
        result += float(coeff) * basis[i]
    
    return result


def grade_energies(M: Array, basis: Array, xp: ArrayModule = np) -> Dict[int, float]:
    """
    Compute energy (squared Frobenius norm) in each grade.
    
    Args:
        M: [4, 4] matrix multivector
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        Dict mapping grade → energy
    """
    energies = {}
    for grade in range(5):
        proj = project_to_grade(M, grade, basis, xp)
        energies[grade] = float(xp.sum(proj**2))
    return energies


def compute_grade_variance(reps: List[Array], basis: Array,
                           xp: ArrayModule = np) -> Dict[str, float]:
    """
    Compute variance of representations decomposed by grade.
    
    Key diagnostic: witness variance should be LOW, content variance should be HIGH.
    This indicates stable self-reference with differentiated content.
    
    Args:
        reps: List of [4, 4] matrix representations
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        Dict with variance metrics per grade and summary stats
    """
    n = len(reps)
    if n < 2:
        return {"error": "need at least 2 reps"}
    
    # Compute grade energies for each rep
    all_energies = {g: [] for g in range(5)}
    for M in reps:
        energies = grade_energies(M, basis, xp)
        for g, e in energies.items():
            all_energies[g].append(e)
    
    # Compute variance per grade
    result = {}
    for g in range(5):
        arr = np.array(all_energies[g])
        result[f"grade_{g}_mean"] = float(np.mean(arr))
        result[f"grade_{g}_std"] = float(np.std(arr))
        result[f"grade_{g}_var"] = float(np.var(arr))
    
    # Summary: witness (0+4) vs content (1+2+3)
    witness_energies = np.array(all_energies[0]) + np.array(all_energies[4])
    content_energies = np.array(all_energies[1]) + np.array(all_energies[2]) + np.array(all_energies[3])
    
    result["witness_mean"] = float(np.mean(witness_energies))
    result["witness_std"] = float(np.std(witness_energies))
    result["content_mean"] = float(np.mean(content_energies))
    result["content_std"] = float(np.std(content_energies))
    
    # Ratio: want content_std >> witness_std
    if result["witness_std"] > 1e-10:
        result["differentiation_ratio"] = result["content_std"] / result["witness_std"]
    else:
        result["differentiation_ratio"] = float("inf")
    
    return result


def compute_witness_stability(reps: List[Array], basis: Array,
                              xp: ArrayModule = np) -> Dict[str, float]:
    """
    Measure how stable the witness pointer is across representations.
    
    High stability = consistent self-reference frame
    
    Args:
        reps: List of [4, 4] matrix representations
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        Dict with witness stability metrics
    """
    n = len(reps)
    if n < 2:
        return {"error": "need at least 2 reps"}
    
    # Extract witness pointers
    witnesses = [witness_pointer(M, basis, xp) for M in reps]
    
    # Pairwise similarities
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sim = frobenius_sim(witnesses[i], witnesses[j], xp)
            sims.append(sim)
    
    sims = np.array(sims)
    
    return {
        "mean_witness_sim": float(np.mean(sims)),
        "std_witness_sim": float(np.std(sims)),
        "min_witness_sim": float(np.min(sims)),
        "max_witness_sim": float(np.max(sims)),
        "n_pairs": len(sims),
    }


# =============================================================================
# COMPREHENSIVE QUOTIENT TEST
# =============================================================================

def test_binding_properties(basis: Array, xp: ArrayModule = np, 
                            seed: int = 42) -> Dict[str, bool]:
    """
    Test that binding operator has expected properties.
    
    1. Binding preserves witness
    2. Binding is idempotent on witness-only states
    3. Binding frames content relative to witness
    
    Args:
        basis: [16, 4, 4] Clifford basis
        xp: array module
        seed: random seed
        
    Returns:
        Dict of test results
    """
    rng = np.random.default_rng(seed)
    results = {}
    
    # Test 1: Binding preserves witness
    M = rng.normal(size=(4, 4))
    w_before = extract_witness_matrix(M, basis, xp)
    M_bound = bind(M, basis, xp=xp)
    w_after = extract_witness_matrix(M_bound, basis, xp)
    
    results["witness_preserved"] = (
        abs(w_before[0] - w_after[0]) < 1e-6 and
        abs(w_before[1] - w_after[1]) < 1e-6
    )
    
    # Test 2: Pure witness is fixed point (no content to bind)
    s, p = rng.normal(), rng.normal()
    W_pure = s * basis[0] + p * basis[15]
    W_bound = bind(W_pure, basis, xp=xp)
    
    # Should be unchanged (content is zero)
    results["witness_fixed_point"] = frobenius_sim(W_pure, W_bound, xp) > 0.99
    
    # Test 3: Binding reduces content-witness angle
    # (Content becomes more "aligned" with witness frame)
    M = rng.normal(size=(4, 4))
    C_before = extract_content(M, basis, xp)
    M_bound = bind(M, basis, xp=xp)
    C_after = extract_content(M_bound, basis, xp)
    
    # Content energy should change (it's being framed)
    energy_before = float(xp.sum(C_before**2))
    energy_after = float(xp.sum(C_after**2))
    results["content_reframed"] = abs(energy_before - energy_after) > 1e-6
    
    return results


def run_quotient_tests(basis: Array, xp: ArrayModule = np, 
                       verbose: bool = True) -> bool:
    """
    Run all quotient structure tests.
    
    Args:
        basis: [16, 4, 4] Clifford basis
        xp: array module
        verbose: print results
        
    Returns:
        True if all tests pass
    """
    all_pass = True
    
    if verbose:
        print("=" * 60)
        print("QUOTIENT STRUCTURE COMPREHENSIVE TESTS")
        print("=" * 60)
    
    # 1. Witness invariance
    if verbose:
        print("\n1. Witness Invariance under Spin(3)...")
    if test_witness_invariance(basis, xp, n_tests=100):
        if verbose:
            print("   ✓ PASSED")
    else:
        if verbose:
            print("   ✗ FAILED")
        all_pass = False
    
    # 2. Normal form invariance
    if verbose:
        print("\n2. Normal Form Invariance...")
    if test_normal_form_invariance(basis, xp, n_tests=100):
        if verbose:
            print("   ✓ PASSED")
    else:
        if verbose:
            print("   ✗ FAILED")
        all_pass = False
    
    # 3. Binding properties
    if verbose:
        print("\n3. Binding Operator Properties...")
    binding_results = test_binding_properties(basis, xp)
    for name, passed in binding_results.items():
        if verbose:
            status = "✓" if passed else "✗"
            print(f"   {status} {name}")
        if not passed:
            all_pass = False
    
    # 4. Grade variance on random data
    if verbose:
        print("\n4. Grade Variance (random data)...")
    rng = np.random.default_rng(42)
    reps = [rng.normal(size=(4, 4)) for _ in range(50)]
    variance = compute_grade_variance(reps, basis, xp)
    if verbose:
        print(f"   Witness std: {variance['witness_std']:.4f}")
        print(f"   Content std: {variance['content_std']:.4f}")
        print(f"   Differentiation ratio: {variance['differentiation_ratio']:.2f}")
    
    # 5. Witness stability on random data
    if verbose:
        print("\n5. Witness Stability (random data)...")
    stability = compute_witness_stability(reps, basis, xp)
    if verbose:
        print(f"   Mean witness similarity: {stability['mean_witness_sim']:.4f}")
        print(f"   Std witness similarity: {stability['std_witness_sim']:.4f}")
    
    if verbose:
        print("\n" + "=" * 60)
        if all_pass:
            print("ALL QUOTIENT TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
        print("=" * 60)
    
    return all_pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'GRADE_IDXS', 'IDX_WITNESS', 'IDX_CORE', 'IDX_FIBER', 'IDX_SPATIAL_BV',
    
    # Witness
    'extract_witness_coeffs',
    'extract_witness_matrix',
    'witness_similarity',
    'witness_pointer',
    
    # Content & Binding
    'extract_content',
    'bind',
    
    # Spin(3) rotors
    'spin3_rotor_matrix',
    'random_spin3_rotor',
    'sandwich',
    
    # Normal form
    'extract_spatial_bivector',
    'extract_electric_bivector',
    'align_rotor_to_z',
    'align_rotor_around_z',
    'normal_form',
    
    # Grade analysis
    'project_to_grade',
    'grade_energies',
    'compute_grade_variance',
    'compute_witness_stability',
    
    # Similarity
    'frobenius_sim',
    'quotient_similarity',
    'quotient_similarity_phi',
    
    # Diagnostics
    'compute_separation',
    
    # Tests
    'test_witness_invariance',
    'test_normal_form_invariance',
    'test_binding_properties',
    'run_quotient_tests',
]
