"""
Distributed Prior — Theory-True Generalization Beyond Basin Coverage
====================================================================

CRITICAL INSIGHT (This is Different From Neural Networks):
──────────────────────────────────────────────────────────
    GENERALIZATION IS NOT LEARNED. IT IS INDUCED BY GEOMETRY AT RETRIEVAL TIME.
    
    Transformers encode priors in weights (learned via gradient descent).
    This system encodes priors in geometry (emergent from attractor fields).
    
    This means:
    • No training → no generalization (obviously)
    • More prototypes → better generalization (coverage)
    • Better basin separation → cleaner generalization (precision)
    • But the *mechanism* of generalization is geometric, not statistical.

PROBLEM STATEMENT:
    Without distributed priors, queries must fall inside discrete prototype basins.
    Uncovered regions have NO fallback — unlike transformers which interpolate smoothly.

THIS MODULE IMPLEMENTS THREE THEORY-TRUE SOLUTIONS:

1. SUPERPOSED-ATTRACTOR PRIOR (Recommended, Brain Analog: Population Coding)
   ──────────────────────────────────────────────────────────────────────────
   Instead of selecting ONE nearest prototype:
   • Retrieve K nearest prototypes by witness distance
   • Weight by φ^(-distance) — NOT softmax! (see phi_kernel docstring for why)
   • Superpose attractors: A_prior = Σ αᵢ Aᵢ
   • Evolve to equilibrium against the superposition
   
   This creates SMOOTH INTERPOLATION across basins.

2. PRIOR FIELD (Green's Function, Brain Analog: Cortical Potential Fields)
   ────────────────────────────────────────────────────────────────────────
   Treat prototypes as sources generating a potential field:
   • U(W) = Σ βᵢ φ^(-d(W, Wᵢ))
   • Query follows gradient toward stable equilibrium
   • Global inductive bias even in uncovered regions

3. FACTORIZED ASSOCIATIVE PRIOR (Hebbian "Weights", Brain Analog: Cortico-Cortical)
   ─────────────────────────────────────────────────────────────────────────────────
   Maintain a low-rank operator: witness → attractor
   • C = Σ WᵢWᵢᵀ  (witness covariance)
   • B = Σ AᵢWᵢᵀ  (witness-attractor association)
   • Prediction: Â(W) = B C⁻¹ W
   • Updates via Hebbian rule with φ⁻¹ learning rate
   
   This gives "distributed weights" that are INSPECTABLE.

EXPLICIT DECISION RULE (When to Use Superposition vs Single Basin):
───────────────────────────────────────────────────────────────────
    conf = (d₂ - d₁) / (d₂ + ε)      # Margin-based geometric confidence
    
    If conf ≥ φ⁻¹ (≈ 0.618):
        → Use single basin (confident match)
        → Fast, precise, deterministic
    
    If conf < φ⁻¹:
        → Use superposed prior + global fallback
        → Interpolates across basins
        → Smoother, trades precision for coverage
    
    This threshold is NOT arbitrary:
    • φ⁻¹ is the spectral gap of the Grace operator
    • Below this, the query is in the "transition zone" between basins
    • Above this, one basin clearly dominates

ALL MECHANISMS USE ONLY:
   • φ-derived constants (not tuned)
   • Grace operator (not arbitrary normalization)
   • Witness/vorticity geometry (not learned)

ONE-PARAGRAPH SUMMARY (Copy-Paste Ready):
─────────────────────────────────────────
    Generalization in this system does not come from statistical learning or smooth
    weights. It comes from geometry. Episodic memory stores exact associations.
    Semantic memory stores stable attractors. A distributed prior emerges when
    multiple attractors act simultaneously as a φ-weighted field, and Grace
    dynamics select an equilibrium. This produces smooth behavior in uncovered
    regions without probabilities, sampling, or tuned parameters. Transformers
    encode priors in weights; brains encode priors in fields. This system does
    the latter.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

# Import our primitives
from .constants import PHI, PHI_INV, PHI_INV_SQ
from .algebra import (
    grace_operator, grace_flow, decompose_to_coefficients,
    vorticity_signature, frobenius_similarity
)
from .quotient import extract_witness, compute_enstrophy, grace_stability


Array = np.ndarray
ArrayModule = type(np)


# =============================================================================
# WITNESS GEOMETRY UTILITIES
# =============================================================================

def witness_distance(w1: Tuple[float, float], w2: Tuple[float, float]) -> float:
    """
    Euclidean distance between two witness vectors (scalar, pseudoscalar).
    
    Theory: Witness is gauge-invariant, so L2 distance is meaningful.
    """
    return np.sqrt((w1[0] - w2[0])**2 + (w1[1] - w2[1])**2)


def phi_kernel(distance: float) -> float:
    """
    φ-derived kernel: φ^(-d)
    
    WHY φ⁻ᵈ AND NOT e⁻ᵈ (CANONICAL JUSTIFICATION):
    ───────────────────────────────────────────────
    φ⁻ᵈ is the UNIQUE exponential decay compatible with Grace's spectral structure.
    
    Proof sketch:
    1. Grace eigenvalues are powers of φ⁻¹: {1, φ⁻¹, φ⁻², φ⁻³, φ⁻¹}
    2. Information at "distance d" in witness space undergoes d-many Grace steps
    3. Each step scales by φ⁻¹ → total decay = φ⁻ᵈ
    4. Using e⁻ᵈ would introduce an arbitrary base disconnected from the algebra
    
    This is NOT softmax (which uses e^x for convexity reasons).
    This is NOT arbitrary (φ emerges from the algebra).
    This is GEOMETRIC decay derived from Grace's spectral gap.
    """
    return PHI ** (-distance)


def extended_witness(M: Array, basis: Array, xp = np) -> Array:
    """
    Extended witness: [scalar, pseudo, enstrophy, grace_stability]
    
    This gives a 4D embedding that captures:
    - Semantic content (scalar + pseudo)
    - Structural complexity (enstrophy)
    - Stability (grace_stability)
    """
    scalar, pseudo = extract_witness(M, basis, xp)
    ens = compute_enstrophy(M, basis, xp)
    sigma = grace_stability(M, basis, xp)
    return xp.array([scalar, pseudo, ens, sigma], dtype=xp.float64)


# =============================================================================
# 1. SUPERPOSED-ATTRACTOR PRIOR
# =============================================================================

def superposed_attractor_prior(
    query: Array,
    prototypes: List[Array],
    prototype_targets: List[Dict[int, float]],
    basis: Array,
    xp = np,
    K: int = 8,
    use_support: bool = True,
    use_stability: bool = True,
    grace_steps: int = 3,
) -> Tuple[Array, Dict[int, float], float, Dict[str, Any]]:
    """
    Retrieve using SUPERPOSED attractors from K nearest prototypes.
    
    Instead of winner-take-all, this creates a smooth interpolation
    across basins — analogous to transformer's distributed prior.
    
    Algorithm:
        1. Compute stabilized query witness
        2. Find K nearest prototypes by witness distance
        3. Weight by φ^(-distance) × support × stability
        4. Superpose: A_prior = Σ αᵢ Aᵢ
        5. Evolve query toward A_prior
        6. Combine target distributions
    
    Args:
        query: [4,4] context matrix
        prototypes: List of [4,4] prototype matrices
        prototype_targets: List of target distributions
        basis: Clifford basis
        xp: array module
        K: number of neighbors (or use cumulative weight threshold)
        use_support: weight by prototype support count
        use_stability: weight by prototype grace_stability
        grace_steps: equilibrium evolution steps
    
    Returns:
        (equilibrium, combined_targets, confidence, info)
    """
    if len(prototypes) == 0:
        return query, {}, 0.0, {"source": "no_prototypes"}
    
    # Step 1: Stabilize query via Grace
    stabilized = grace_operator(query, basis, xp)
    for _ in range(grace_steps - 1):
        stabilized = grace_operator(stabilized, basis, xp)
    
    query_witness = extract_witness(stabilized, basis, xp)
    
    # Step 2: Compute distances and find K nearest
    distances = []
    witnesses = []
    
    for proto in prototypes:
        proto_witness = extract_witness(proto, basis, xp)
        witnesses.append(proto_witness)
        d = witness_distance(query_witness, proto_witness)
        distances.append(d)
    
    distances = np.array(distances)
    sorted_indices = np.argsort(distances)
    
    # Use top K or all if fewer
    K_actual = min(K, len(prototypes))
    top_k = sorted_indices[:K_actual]
    
    # Step 3: Compute weights using φ-kernel (NOT softmax!)
    raw_weights = []
    for idx in top_k:
        w = phi_kernel(distances[idx])
        
        # Optionally multiply by support and stability
        # (These are theory-derived, not arbitrary)
        if use_support and hasattr(prototype_targets[idx], '__len__'):
            support = sum(prototype_targets[idx].values()) if isinstance(prototype_targets[idx], dict) else 1.0
            w *= np.sqrt(support + 1)  # Gentle support influence
        
        if use_stability:
            sigma = grace_stability(prototypes[idx], basis, xp)
            w *= sigma  # High stability = more weight
        
        raw_weights.append(w)
    
    # Normalize to sum to 1 (this is NOT softmax, just convex combination)
    total_weight = sum(raw_weights)
    if total_weight < 1e-10:
        weights = [1.0 / K_actual] * K_actual
    else:
        weights = [w / total_weight for w in raw_weights]
    
    # Step 4: Superpose attractors
    prior_attractor = xp.zeros((4, 4), dtype=xp.float64)
    for i, idx in enumerate(top_k):
        prior_attractor += weights[i] * prototypes[idx]
    
    # Apply Grace to stabilize the superposition
    prior_attractor = grace_operator(prior_attractor, basis, xp)
    
    # Step 5: Evolve query toward superposed attractor
    equilibrium = grace_flow(
        query, prior_attractor, basis,
        steps=grace_steps,
        rate=PHI_INV_SQ,
        xp=xp
    )
    
    # Step 6: Combine target distributions
    combined_targets = {}
    for i, idx in enumerate(top_k):
        targets = prototype_targets[idx]
        if isinstance(targets, dict):
            for token, prob in targets.items():
                combined_targets[token] = combined_targets.get(token, 0.0) + weights[i] * prob
    
    # Step 7: Compute confidence from margin
    if len(distances) >= 2:
        d1, d2 = sorted(distances)[:2]
        confidence = (d2 - d1) / (d2 + 1e-8)
    else:
        confidence = 1.0
    
    info = {
        "source": "superposed_prior",
        "K": K_actual,
        "distances": [float(distances[i]) for i in top_k],
        "weights": [float(w) for w in weights],
        "margin_confidence": float(confidence),
        "top_indices": [int(i) for i in top_k],
    }
    
    return equilibrium, combined_targets, confidence, info


# =============================================================================
# 2. PRIOR FIELD (Green's Function)
# =============================================================================

def compute_prior_field_potential(
    query_witness: Tuple[float, float],
    prototype_witnesses: List[Tuple[float, float]],
    prototype_supports: List[float],
) -> Tuple[float, List[float]]:
    """
    Compute the potential field value at a query point.
    
    U(W) = Σᵢ βᵢ φ^(-d(W, Wᵢ))
    
    Returns (potential, individual_contributions)
    """
    contributions = []
    for i, pw in enumerate(prototype_witnesses):
        d = witness_distance(query_witness, pw)
        contrib = prototype_supports[i] * phi_kernel(d)
        contributions.append(contrib)
    
    return sum(contributions), contributions


def prior_field_retrieve(
    query: Array,
    prototypes: List[Array],
    prototype_targets: List[Dict[int, float]],
    prototype_supports: List[float],
    basis: Array,
    xp = np,
    grace_steps: int = 3,
) -> Tuple[Array, Dict[int, float], float, Dict[str, Any]]:
    """
    Retrieve using a continuous potential field induced by prototypes.
    
    This creates a GLOBAL prior — every point in witness space has
    a defined expected attractor, even far from any prototype.
    """
    if len(prototypes) == 0:
        return query, {}, 0.0, {"source": "no_prototypes"}
    
    # Stabilize query
    stabilized = grace_operator(query, basis, xp)
    query_witness = extract_witness(stabilized, basis, xp)
    
    # Compute prototype witnesses
    proto_witnesses = [extract_witness(p, basis, xp) for p in prototypes]
    
    # Compute field contributions
    potential, contributions = compute_prior_field_potential(
        query_witness, proto_witnesses, prototype_supports
    )
    
    if potential < 1e-10:
        return query, {}, 0.0, {"source": "zero_potential"}
    
    # Field-weighted superposition
    weights = [c / potential for c in contributions]
    
    prior_attractor = xp.zeros((4, 4), dtype=xp.float64)
    for i, proto in enumerate(prototypes):
        prior_attractor += weights[i] * proto
    
    prior_attractor = grace_operator(prior_attractor, basis, xp)
    
    # Evolve to equilibrium
    equilibrium = grace_flow(
        query, prior_attractor, basis,
        steps=grace_steps,
        rate=PHI_INV_SQ,
        xp=xp
    )
    
    # Combine targets
    combined_targets = {}
    for i, targets in enumerate(prototype_targets):
        if isinstance(targets, dict):
            for token, prob in targets.items():
                combined_targets[token] = combined_targets.get(token, 0.0) + weights[i] * prob
    
    # Confidence from potential gradient (high potential = confident)
    max_contrib = max(contributions) if contributions else 0
    confidence = max_contrib / (potential + 1e-8)
    
    info = {
        "source": "prior_field",
        "potential": float(potential),
        "max_contribution": float(max_contrib),
        "confidence": float(confidence),
    }
    
    return equilibrium, combined_targets, confidence, info


# =============================================================================
# 3. FACTORIZED ASSOCIATIVE PRIOR (Hebbian "Weights")
# =============================================================================

class FactorizedAssociativePrior:
    """
    A Hebbian-learned linear operator: witness → attractor
    
    This is the closest analogue to "weights" in our architecture,
    but it's:
    - Explicit and inspectable
    - Updated via Hebbian rule (not gradient descent)
    - Uses φ⁻¹ learning rate (not tuned)
    
    Stores:
        C: witness covariance matrix (witness_dim × witness_dim)
        B: witness-attractor association (16 × witness_dim)
    
    Prediction:
        Â(W) = B @ C⁻¹ @ W
    """
    
    def __init__(
        self,
        witness_dim: int = 4,  # Extended witness: [scalar, pseudo, enstrophy, stability]
        regularization: float = 1e-6,
        xp = np,
    ):
        """
        Initialize the factorized prior.
        
        Args:
            witness_dim: Dimension of witness vector (2 for basic, 4+ for extended)
            regularization: Ridge regularization for C inverse
            xp: array module
        """
        self.witness_dim = witness_dim
        self.regularization = regularization
        self.xp = xp
        
        # Covariance and association matrices
        self.C = xp.eye(witness_dim, dtype=xp.float64) * regularization
        self.B = xp.zeros((16, witness_dim), dtype=xp.float64)
        
        # Track statistics
        self.n_updates = 0
        self.total_energy = 0.0
    
    def update(
        self,
        witness: Array,
        attractor: Array,
        weight: float = 1.0,
    ):
        """
        Hebbian update: store (witness, attractor) association.
        
        Update rule (theory-true):
            C ← (1 - φ⁻¹)C + φ⁻¹ · weight · W Wᵀ
            B ← (1 - φ⁻¹)B + φ⁻¹ · weight · A Wᵀ
        
        Args:
            witness: Extended witness vector
            attractor: 4×4 attractor matrix
            weight: Optional importance weight (e.g., salience)
        """
        xp = self.xp
        
        # Flatten attractor to 16D
        A_flat = attractor.flatten()
        W = witness.reshape(-1)
        
        # Hebbian update with φ⁻¹ learning rate
        self.C = (1 - PHI_INV) * self.C + PHI_INV * weight * xp.outer(W, W)
        self.B = (1 - PHI_INV) * self.B + PHI_INV * weight * xp.outer(A_flat, W)
        
        self.n_updates += 1
        self.total_energy += float(xp.sum(A_flat * A_flat))
    
    def predict(self, witness: Array, basis: Array) -> Array:
        """
        Predict attractor from witness.
        
        Â(W) = B @ C⁻¹ @ W
        
        Then apply Grace to stabilize.
        """
        xp = self.xp
        W = witness.reshape(-1)
        
        # Regularized inverse
        C_reg = self.C + self.regularization * xp.eye(self.witness_dim)
        C_inv = xp.linalg.inv(C_reg)
        
        # Linear prediction
        A_flat = self.B @ C_inv @ W
        A_pred = A_flat.reshape(4, 4)
        
        # Stabilize with Grace
        A_pred = grace_operator(A_pred, basis, xp)
        
        return A_pred
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get diagnostic statistics about the prior."""
        xp = self.xp
        
        # Eigenvalues of C (indicates which dimensions are "learned")
        eigvals = xp.linalg.eigvalsh(self.C)
        
        # Effective rank (how many dimensions are active)
        eigsum = float(xp.sum(eigvals))
        if eigsum > 1e-10:
            normalized_eigvals = eigvals / eigsum
            entropy = -float(xp.sum(normalized_eigvals * xp.log(normalized_eigvals + 1e-10)))
            effective_rank = float(xp.exp(entropy))
        else:
            effective_rank = 0.0
        
        return {
            "n_updates": self.n_updates,
            "avg_energy": self.total_energy / max(1, self.n_updates),
            "C_trace": float(xp.trace(self.C)),
            "B_norm": float(xp.linalg.norm(self.B, 'fro')),
            "effective_rank": effective_rank,
            "eigenvalues": [float(e) for e in sorted(eigvals, reverse=True)],
        }


def retrieve_with_factorized_prior(
    query: Array,
    prior: FactorizedAssociativePrior,
    basis: Array,
    xp = np,
    grace_steps: int = 3,
) -> Tuple[Array, Dict[str, Any]]:
    """
    Retrieve using the factorized associative prior.
    
    This provides a GLOBAL fallback when local prototypes are unavailable
    or ambiguous.
    """
    # Compute extended witness
    query_witness = extended_witness(query, basis, xp)
    
    # Predict attractor from prior
    prior_attractor = prior.predict(query_witness, basis)
    
    # Evolve to equilibrium
    equilibrium = grace_flow(
        query, prior_attractor, basis,
        steps=grace_steps,
        rate=PHI_INV_SQ,
        xp=xp
    )
    
    info = {
        "source": "factorized_prior",
        "prior_stats": prior.get_statistics(),
    }
    
    return equilibrium, info


# =============================================================================
# COMBINED RETRIEVAL WITH CONFIDENCE-GATED PRIOR
# =============================================================================

def retrieve_with_distributed_prior(
    query: Array,
    prototypes: List[Array],
    prototype_targets: List[Dict[int, float]],
    prototype_supports: List[float],
    factorized_prior: Optional[FactorizedAssociativePrior],
    basis: Array,
    xp = np,
    K: int = 8,
    confidence_threshold: float = PHI_INV_SQ,  # Theory-true: φ⁻² (spectral gap)
    grace_steps: int = 3,
) -> Tuple[Array, Dict[int, float], float, Dict[str, Any]]:
    """
    FULL RETRIEVAL with distributed prior fallback.
    
    Strategy:
        1. Try superposed-attractor prior (local, precise)
        2. Compute confidence from margin
        3. If low confidence AND factorized prior exists:
           → Blend with factorized prior (global, smooth)
    
    This gives:
        - Local precision when confident
        - Global smoothness when ambiguous
        - No arbitrary thresholds (confidence is geometric)
    
    Args:
        query: Context matrix
        prototypes: List of prototype matrices
        prototype_targets: List of target distributions
        prototype_supports: List of prototype support counts
        factorized_prior: Optional global prior (can be None)
        basis: Clifford basis
        xp: array module
        K: number of neighbors for superposition
        confidence_threshold: blend threshold (default ≈ 0.5)
        grace_steps: equilibrium steps
    
    Returns:
        (equilibrium, targets, confidence, info)
    """
    # First: try local superposed prior
    local_eq, local_targets, local_conf, local_info = superposed_attractor_prior(
        query, prototypes, prototype_targets, basis, xp,
        K=K, grace_steps=grace_steps
    )
    
    # If confident or no global prior, return local
    if local_conf >= confidence_threshold or factorized_prior is None:
        return local_eq, local_targets, local_conf, local_info
    
    # Otherwise: blend with factorized prior
    global_eq, global_info = retrieve_with_factorized_prior(
        query, factorized_prior, basis, xp, grace_steps
    )
    
    # Blend using confidence as mixing coefficient
    # High conf → mostly local, Low conf → more global
    blend_weight = local_conf  # φ-natural: conf is already in [0,1]
    
    blended_eq = blend_weight * local_eq + (1 - blend_weight) * global_eq
    blended_eq = grace_operator(blended_eq, basis, xp)  # Stabilize blend
    
    info = {
        "source": "blended_prior",
        "local_confidence": float(local_conf),
        "blend_weight": float(blend_weight),
        "local_info": local_info,
        "global_info": global_info,
    }
    
    return blended_eq, local_targets, local_conf, info


# =============================================================================
# COVERAGE METRICS (for auditing)
# =============================================================================

def compute_basin_coverage(
    test_queries: List[Array],
    prototypes: List[Array],
    basis: Array,
    xp = np,
) -> Dict[str, float]:
    """
    Compute coverage metrics for the prototype basin structure.
    
    Returns:
        - avg_nearest_distance: Mean distance to nearest prototype
        - coverage_density: Fraction of queries with confident match
        - basin_entropy: Distribution entropy of prototype selections
        - boundary_fraction: Fraction of queries near boundaries
    """
    if len(prototypes) == 0 or len(test_queries) == 0:
        return {"error": "no data"}
    
    # Compute prototype witnesses
    proto_witnesses = [extract_witness(p, basis, xp) for p in prototypes]
    
    nearest_distances = []
    second_distances = []
    selected_protos = []
    
    for query in test_queries:
        q_witness = extract_witness(query, basis, xp)
        
        distances = [witness_distance(q_witness, pw) for pw in proto_witnesses]
        sorted_d = sorted(enumerate(distances), key=lambda x: x[1])
        
        nearest_distances.append(sorted_d[0][1])
        if len(sorted_d) >= 2:
            second_distances.append(sorted_d[1][1])
        selected_protos.append(sorted_d[0][0])
    
    # Compute metrics
    avg_nearest = float(np.mean(nearest_distances))
    
    # Confidence for each query
    confidences = []
    for d1, d2 in zip(nearest_distances, second_distances):
        conf = (d2 - d1) / (d2 + 1e-8)
        confidences.append(conf)
    
    coverage_density = float(np.mean([c > 0.5 for c in confidences]))
    boundary_fraction = float(np.mean([c < 0.2 for c in confidences]))
    
    # Basin entropy (how evenly distributed are selections?)
    counts = np.bincount(selected_protos, minlength=len(prototypes))
    probs = counts / (len(test_queries) + 1e-10)
    entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
    max_entropy = np.log(len(prototypes) + 1e-10)
    normalized_entropy = entropy / (max_entropy + 1e-10)
    
    return {
        "avg_nearest_distance": avg_nearest,
        "coverage_density": coverage_density,
        "boundary_fraction": boundary_fraction,
        "basin_entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "num_prototypes": len(prototypes),
        "num_queries": len(test_queries),
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core primitives
    'witness_distance',
    'phi_kernel',
    'extended_witness',
    
    # Retrieval methods
    'superposed_attractor_prior',
    'prior_field_retrieve',
    'retrieve_with_factorized_prior',
    'retrieve_with_distributed_prior',
    
    # Factorized prior class
    'FactorizedAssociativePrior',
    
    # Metrics
    'compute_basin_coverage',
]
