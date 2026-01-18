"""
Attribute Binding Module — Clifford Grade-Based Binding
========================================================

This module solves the binding problem using Clifford algebra structure:
How do we represent "red ball" differently from "blue ball" and "ball red"?

THEORY:
    The wedge product A ∧ B is antisymmetric: A ∧ B = -B ∧ A
    This naturally encodes ORDER: "red ball" ≠ "ball red"
    
    Grade structure gives us binding semantics:
    - Grade 0 (scalar): Intensity/salience
    - Grade 1 (vectors): Objects/entities
    - Grade 2 (bivectors): Relations/bindings (A ∧ B)
    - Grade 3 (trivectors): Contexts/situations
    - Grade 4 (pseudoscalar): Valence/perspective

KEY INSIGHT:
    Binding = wedge product + geometric product
    
    bind(attribute, object) = attribute ∧ object + λ · attribute · object
    
    The wedge creates the unique "relation" (grade 2)
    The geometric product preserves invertibility
"""

import numpy as np
from typing import Dict, Tuple, Optional

from .constants import PHI_INV, PHI_EPSILON, GRADE_INDICES
from .algebra import (
    wedge_product,
    geometric_product,
    grace_operator,
    decompose_to_coefficients,
    reconstruct_from_coefficients,
    frobenius_cosine,
)

Array = np.ndarray
ArrayModule = type(np)


# =============================================================================
# CORE BINDING OPERATIONS
# =============================================================================

def bind_attribute_to_object(
    attribute: Array,
    obj: Array,
    basis: Array,
    xp: ArrayModule = np,
    binding_strength: float = PHI_INV,
) -> Array:
    """
    Create bound representation: attribute-object.
    
    THEORY:
        The binding combines:
        1. Wedge product: Creates unique grade-2 "relation" component
        2. Geometric product: Preserves information for recovery
        
        bound = wedge(attr, obj) + λ · geom(attr, obj)
        
        This ensures:
        - Order matters (wedge is antisymmetric)
        - Different attributes give different results
        - We can (approximately) recover the object
        
    Args:
        attribute: Attribute matrix (e.g., "red")
        obj: Object matrix (e.g., "ball")
        basis: Clifford basis
        xp: Array module
        binding_strength: Blend factor (default φ⁻¹)
        
    Returns:
        Bound representation matrix
    """
    # Wedge product creates the unique "relation"
    relation = wedge_product(attribute, obj, xp)
    
    # Geometric product preserves invertibility
    composition = geometric_product(attribute, obj)
    
    # Combine with binding strength
    bound = relation + binding_strength * composition
    
    # Grace-normalize for stability
    bound = grace_operator(bound, basis, xp)
    
    return bound


def extract_object_from_bound(
    bound: Array,
    attribute: Array,
    basis: Array,
    xp: ArrayModule = np,
) -> Array:
    """
    Extract object from attribute-object binding.
    
    THEORY:
        Uses the geometric product's invertibility:
        If bound ≈ attr ∧ obj + λ · attr · obj
        Then obj ≈ attr⁻¹ · (bound - attr ∧ obj) / λ
        
        This is approximate because Grace normalization loses some info.
        
    Args:
        bound: The bound representation
        attribute: The known attribute
        basis: Clifford basis
        xp: Array module
        
    Returns:
        Recovered object (approximate)
    """
    # Compute attribute inverse (use transpose for orthogonal-ish matrices)
    # This is approximate for non-orthogonal matrices
    attr_norm = xp.linalg.norm(attribute)
    if attr_norm < PHI_EPSILON:
        return bound  # Can't extract without attribute
    
    attr_inv = attribute.T / (attr_norm ** 2)
    
    # Remove the wedge component (approximately)
    # We approximate by computing what the wedge would have been
    # and subtracting it
    estimated_wedge = wedge_product(attribute, bound, xp)
    reduced = bound - PHI_INV * estimated_wedge  # φ-derived partial removal (was 0.5)
    
    # Apply inverse attribute
    recovered = geometric_product(attr_inv, reduced)
    
    # Grace-normalize
    recovered = grace_operator(recovered, basis, xp)
    
    return recovered


def compare_bindings(
    binding1: Array,
    binding2: Array,
    basis: Array,
    xp: ArrayModule = np,
) -> Dict[str, float]:
    """
    Compare two bindings at each grade level.
    
    THEORY:
        Bindings with the same attribute should share grade-2 structure.
        Bindings with the same object should share grade-1 structure.
        This allows us to identify what's shared vs different.
        
    Args:
        binding1: First binding (e.g., "red ball")
        binding2: Second binding (e.g., "blue ball")
        basis: Clifford basis
        xp: Array module
        
    Returns:
        Dict mapping grade to similarity score
    """
    # Decompose both bindings
    coeffs1 = decompose_to_coefficients(binding1, basis, xp)
    coeffs2 = decompose_to_coefficients(binding2, basis, xp)
    
    result = {}
    
    # Compare each grade
    for grade, indices in GRADE_INDICES.items():
        c1_grade = coeffs1[indices]
        c2_grade = coeffs2[indices]
        
        norm1 = xp.linalg.norm(c1_grade)
        norm2 = xp.linalg.norm(c2_grade)
        
        if norm1 > PHI_EPSILON and norm2 > PHI_EPSILON:
            similarity = float(xp.dot(c1_grade, c2_grade) / (norm1 * norm2))
        else:
            similarity = 0.0 if norm1 != norm2 else 1.0
        
        result[f'grade_{grade}'] = similarity
    
    # Overall similarity (using cosine for bounded [-1, 1] output)
    result['overall'] = float(frobenius_cosine(binding1, binding2, xp))
    
    return result


# =============================================================================
# ADVANCED BINDING OPERATIONS
# =============================================================================

def bind_multiple_attributes(
    attributes: list,
    obj: Array,
    basis: Array,
    xp: ArrayModule = np,
) -> Array:
    """
    Bind multiple attributes to an object: "big red ball".
    
    Applies attributes in sequence (order matters).
    
    Args:
        attributes: List of attribute matrices [big, red, ...]
        obj: Object matrix
        basis: Clifford basis
        xp: Array module
        
    Returns:
        Fully bound representation
    """
    if not attributes:
        return obj
    
    result = obj
    for attr in reversed(attributes):  # Apply inner-most first
        result = bind_attribute_to_object(attr, result, basis, xp)
    
    return result


def unbind_attribute(
    bound: Array,
    attribute: Array,
    basis: Array,
    xp: ArrayModule = np,
) -> Array:
    """
    Remove an attribute from a binding.
    
    Given "big red ball" and "big", returns approximately "red ball".
    
    Args:
        bound: The bound representation
        attribute: Attribute to remove
        basis: Clifford basis
        xp: Array module
        
    Returns:
        Binding with attribute removed
    """
    return extract_object_from_bound(bound, attribute, basis, xp)


def query_binding(
    query_attr: Array,
    bindings: list,
    basis: Array,
    xp: ArrayModule = np,
) -> Tuple[int, float]:
    """
    Find which binding best matches a query attribute.
    
    "Which of these has attribute X?"
    
    Args:
        query_attr: Attribute to search for
        bindings: List of bound representations
        basis: Clifford basis
        xp: Array module
        
    Returns:
        (best_index, confidence)
    """
    if not bindings:
        return -1, 0.0
    
    best_idx = 0
    best_score = -float('inf')
    
    for i, bound in enumerate(bindings):
        # Compute how much the binding "contains" the query attribute
        # Use grade-2 similarity (where bindings live)
        comparison = compare_bindings(
            bind_attribute_to_object(query_attr, xp.eye(4), basis, xp),
            bound,
            basis,
            xp,
        )
        
        score = comparison.get('grade_2', 0.0)
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    confidence = max(0.0, min(1.0, best_score))
    return best_idx, confidence


# =============================================================================
# BINDING ANALYSIS
# =============================================================================

def decompose_binding(
    bound: Array,
    basis: Array,
    xp: ArrayModule = np,
) -> Dict[str, Array]:
    """
    Decompose a binding into its grade components.
    
    Returns:
        Dict with 'scalar', 'vector', 'bivector', 'trivector', 'pseudoscalar'
    """
    coeffs = decompose_to_coefficients(bound, basis, xp)
    
    result = {}
    
    for grade, indices in GRADE_INDICES.items():
        grade_coeffs = xp.zeros(16)
        grade_coeffs[indices] = coeffs[indices]
        result[f'grade_{grade}'] = reconstruct_from_coefficients(grade_coeffs, basis, xp)
    
    return result


def binding_signature(
    bound: Array,
    basis: Array,
    xp: ArrayModule = np,
) -> Array:
    """
    Extract a compact signature of a binding.
    
    The signature captures the grade distribution without full reconstruction.
    
    Returns:
        [5] array of grade energies
    """
    coeffs = decompose_to_coefficients(bound, basis, xp)
    
    signature = xp.zeros(5)
    for grade, indices in GRADE_INDICES.items():
        signature[grade] = xp.sum(coeffs[indices] ** 2)
    
    return signature


def binding_similarity(
    binding1: Array,
    binding2: Array,
    basis: Array,
    xp: ArrayModule = np,
    grade_weights: Optional[Dict[int, float]] = None,
) -> float:
    """
    Compute weighted similarity between bindings.
    
    Args:
        binding1: First binding
        binding2: Second binding
        basis: Clifford basis
        xp: Array module
        grade_weights: Optional weights for each grade
        
    Returns:
        Similarity score in [-1, 1]
    """
    if grade_weights is None:
        # THEORY-TRUE: Use Grace scaling factors (φ-derived)
        # These are the natural weights from Grace operator structure
        from holographic_prod.core.constants import GRACE_SCALES
        grade_weights = GRACE_SCALES.copy()
        # Normalize to sum to 1.0 for weighted average
        total = sum(grade_weights.values())
        grade_weights = {k: v / total for k, v in grade_weights.items()}
    
    comparison = compare_bindings(binding1, binding2, basis, xp)
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for grade, weight in grade_weights.items():
        key = f'grade_{grade}'
        if key in comparison:
            weighted_sum += weight * comparison[key]
            total_weight += weight
    
    if total_weight > 0:
        return weighted_sum / total_weight
    return comparison.get('overall', 0.0)
