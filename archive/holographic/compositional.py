"""
Compositional Embeddings — Feature-Based Word Representations
==============================================================

CRITICAL INSIGHT:
    We were using Clifford algebra at the WRONG level.
    
    WRONG: Words are atomic random matrices, compose into contexts
    RIGHT: Words are COMPOSED from features, then contexts compose words

WHY ONE-SHOT LEARNING WORKS:
    Human learns "zebra" in one shot because:
    - Already knows: animal, striped, equine, African
    - "zebra" = composition of existing features
    - One exposure tells WHICH features to combine

ARCHITECTURE:
    Feature Space: F = {f₁, f₂, ..., fₖ} where each fᵢ is a 4×4 direction
    Word Embedding: embed(w) = I + Σᵢ αᵢ(w) · fᵢ
    Context: geometric_product of word embeddings

GRADE STRUCTURE AS FEATURE TYPES:
    Grade 0 (scalar):      existence/salience (the identity base)
    Grade 1 (4 vectors):   basic properties
    Grade 2 (6 bivectors): relations
    Grade 3 (4 trivectors): contexts
    Grade 4 (pseudoscalar): reflexive/meta

KEY CHANGE:
    - Features are LEARNED basis directions in Cl(3,1)
    - Words have LEARNED coefficients for each feature
    - Sharing features → similar embeddings → clustering
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .constants import PHI, PHI_INV, MATRIX_DIM, CLIFFORD_DIM
from .algebra import (
    build_clifford_basis, geometric_product_batch,
    frobenius_similarity_batch, normalize_matrix
)

Array = np.ndarray
ArrayModule = type(np)


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

@dataclass
class FeatureSet:
    """
    A set of semantic features, each represented as a direction in Cl(3,1).
    
    Features are organized by grade:
    - Grade 1: basic properties (animate, concrete, positive, large, ...)
    - Grade 2: relations (part-of, kind-of, causes, ...)
    - Grade 3: contexts (location, time, use, ...)
    
    Each feature is a unit-norm 4×4 matrix.
    """
    num_features: int
    feature_matrices: Array  # [num_features, 4, 4]
    feature_names: List[str]  # Human-readable names
    feature_grades: List[int]  # Which grade each feature lives in
    
    
def create_feature_set(
    num_features: int,
    basis: Array,
    xp: ArrayModule = np,
    seed: int = 42,
) -> FeatureSet:
    """
    Create a set of ORTHOGONAL feature directions.
    
    CRITICAL FIX: Features must be orthogonal to enable proper compositional structure.
    Otherwise, shared features in coefficient space don't map to similar embeddings.
    
    Strategy: Use ALL grades (1-3) for each feature, ensuring orthogonality.
    This gives us 14 dimensions (4+6+4) to work with.
    
    Args:
        num_features: Number of features to create (max ~14 for full orthogonality)
        basis: [16, 4, 4] Clifford basis
        xp: array module
        seed: random seed
        
    Returns:
        FeatureSet with orthogonal features
    """
    rng = np.random.default_rng(seed)
    
    # Use grades 1, 2, 3 (indices 1-14, excluding scalar and pseudoscalar)
    # This gives us 14 dimensions for features
    all_indices = list(range(1, 15))  # [1, 2, ..., 14]
    dim = len(all_indices)  # 14
    
    # Limit features to available dimensions
    actual_features = min(num_features, dim)
    
    # Generate orthogonal random vectors via QR decomposition
    random_matrix = rng.normal(size=(dim, actual_features))
    Q, _ = np.linalg.qr(random_matrix)  # Q is [dim, actual_features] with orthonormal columns
    
    feature_matrices = xp.zeros((actual_features, MATRIX_DIM, MATRIX_DIM), dtype=xp.float64)
    feature_grades = []
    
    for i in range(actual_features):
        coeffs = Q[:, i]  # Orthonormal column
        
        for j, idx in enumerate(all_indices):
            feature_matrices[i] += coeffs[j] * basis[idx]
        
        # Determine dominant grade for naming
        g1_energy = np.sum(coeffs[:4] ** 2)
        g2_energy = np.sum(coeffs[4:10] ** 2)
        g3_energy = np.sum(coeffs[10:] ** 2)
        
        if g1_energy >= g2_energy and g1_energy >= g3_energy:
            feature_grades.append(1)
        elif g2_energy >= g3_energy:
            feature_grades.append(2)
        else:
            feature_grades.append(3)
        
        # Normalize the matrix (should already be ~unit norm from QR)
        norm = xp.sqrt(xp.sum(feature_matrices[i] ** 2))
        feature_matrices[i] = feature_matrices[i] / (norm + 1e-8)
    
    # Generate names
    feature_names = [f"f{i}_g{feature_grades[i]}" for i in range(actual_features)]
    
    return FeatureSet(
        num_features=actual_features,
        feature_matrices=feature_matrices,
        feature_names=feature_names,
        feature_grades=feature_grades,
    )


# =============================================================================
# COMPOSITIONAL EMBEDDING
# =============================================================================

class CompositionalEmbedding:
    """
    Word embeddings as compositions of semantic features.
    
    Each word is represented as:
        embed(w) = I + Σᵢ αᵢ(w) · fᵢ
    
    Where:
        - I is the identity (existence/salience base)
        - fᵢ are feature directions (learned)
        - αᵢ(w) are per-word feature coefficients (learned)
    
    KEY PROPERTY:
        Words sharing features → similar embeddings → natural clustering
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_features: int = 64,
        xp: ArrayModule = np,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.num_features = num_features
        self.xp = xp
        
        # Build Clifford basis
        self.basis = build_clifford_basis(xp)
        
        # Identity matrix (compositional base)
        self.identity = xp.eye(MATRIX_DIM, dtype=xp.float64)
        
        # Create feature set (will be capped at 14 orthogonal features)
        self.features = create_feature_set(num_features, self.basis, xp, seed)
        self.num_features = self.features.num_features  # May be less than requested
        
        # Per-word feature coefficients: [vocab_size, actual_num_features]
        # Initialize sparse: most words have few features active
        rng = np.random.default_rng(seed + 1)
        self.coefficients = xp.zeros((vocab_size, self.num_features), dtype=xp.float64)
        
        # Sparse initialization: each word gets ~3-6 active features
        for w in range(vocab_size):
            num_active = min(rng.integers(3, 7), self.num_features)
            active_idx = rng.choice(self.num_features, size=num_active, replace=False)
            self.coefficients[w, active_idx] = rng.normal(scale=0.5, size=num_active)
        
        # Cache for computed embeddings
        self._embedding_cache: Dict[int, Array] = {}
    
    def get_embedding(self, word_idx: int) -> Array:
        """
        Get the 4×4 embedding matrix for a word.
        
        embed(w) = α₀·I + Σᵢ αᵢ(w) · fᵢ
        
        Where α₀ = 0.3 (small identity component) so features dominate.
        """
        if word_idx in self._embedding_cache:
            return self._embedding_cache[word_idx]
        
        word_idx = word_idx % self.vocab_size
        
        # Start with SMALL identity component (not dominating)
        # This allows features to differentiate words
        identity_weight = 0.3
        emb = identity_weight * self.identity.copy()
        
        # Add weighted features (these should dominate)
        coeffs = self.coefficients[word_idx]  # [num_features]
        for i, alpha in enumerate(coeffs):
            if abs(alpha) > 1e-8:
                emb = emb + alpha * self.features.feature_matrices[i]
        
        # Normalize (keep unit Frobenius norm)
        norm = self.xp.sqrt(self.xp.sum(emb ** 2))
        emb = emb / (norm + 1e-8)
        
        self._embedding_cache[word_idx] = emb
        return emb
    
    def get_all_embeddings(self) -> Array:
        """Get all word embeddings as [vocab_size, 4, 4] array."""
        embeddings = self.xp.zeros(
            (self.vocab_size, MATRIX_DIM, MATRIX_DIM), dtype=self.xp.float64
        )
        for w in range(self.vocab_size):
            embeddings[w] = self.get_embedding(w)
        return embeddings
    
    def clear_cache(self):
        """Clear embedding cache (call after updating coefficients)."""
        self._embedding_cache.clear()
    
    def feature_similarity(self, w1: int, w2: int) -> float:
        """
        Compute feature-space similarity between two words.
        
        This is cosine similarity in coefficient space.
        Words with shared features will have high similarity.
        """
        c1 = self.coefficients[w1 % self.vocab_size]
        c2 = self.coefficients[w2 % self.vocab_size]
        
        dot = float(self.xp.sum(c1 * c2))
        norm1 = float(self.xp.sqrt(self.xp.sum(c1 ** 2)) + 1e-8)
        norm2 = float(self.xp.sqrt(self.xp.sum(c2 ** 2)) + 1e-8)
        
        return dot / (norm1 * norm2)
    
    def embedding_similarity(self, w1: int, w2: int) -> float:
        """
        Compute embedding-space similarity (Frobenius).
        """
        e1 = self.get_embedding(w1)
        e2 = self.get_embedding(w2)
        return float(self.xp.sum(e1 * e2))
    
    def set_word_features(self, word_idx: int, feature_indices: List[int], 
                          coefficients: List[float]) -> None:
        """
        Explicitly set which features a word has.
        
        This is how one-shot learning works:
        Given context, infer features and set them directly.
        """
        word_idx = word_idx % self.vocab_size
        
        # Clear existing
        self.coefficients[word_idx] = 0.0
        
        # Set specified features
        for idx, coeff in zip(feature_indices, coefficients):
            if 0 <= idx < self.num_features:
                self.coefficients[word_idx, idx] = coeff
        
        # Clear cache for this word
        if word_idx in self._embedding_cache:
            del self._embedding_cache[word_idx]


# =============================================================================
# TESTS
# =============================================================================

def test_feature_set(xp: ArrayModule = np) -> bool:
    """Test feature set creation."""
    print("Testing FeatureSet creation...")
    
    basis = build_clifford_basis(xp)
    features = create_feature_set(32, basis, xp)  # Will be capped at 14
    
    # Feature count is capped at 14 (dimension of grades 1-3)
    assert features.num_features <= 14, f"Expected ≤14 features, got {features.num_features}"
    assert features.feature_matrices.shape == (features.num_features, 4, 4)
    assert len(features.feature_names) == features.num_features
    assert len(features.feature_grades) == features.num_features
    
    print(f"  Created {features.num_features} orthogonal features")
    
    # Check grade distribution
    n_g1 = sum(1 for g in features.feature_grades if g == 1)
    n_g2 = sum(1 for g in features.feature_grades if g == 2)
    n_g3 = sum(1 for g in features.feature_grades if g == 3)
    print(f"  Grade distribution: G1={n_g1}, G2={n_g2}, G3={n_g3}")
    
    # Check normalization
    for i in range(features.num_features):
        norm = float(xp.sqrt(xp.sum(features.feature_matrices[i] ** 2)))
        assert abs(norm - 1.0) < 0.01, f"Feature {i} not normalized: {norm}"
    
    # Check orthogonality
    max_off_diag = 0.0
    for i in range(features.num_features):
        for j in range(i + 1, features.num_features):
            sim = float(xp.sum(features.feature_matrices[i] * features.feature_matrices[j]))
            max_off_diag = max(max_off_diag, abs(sim))
    print(f"  Max off-diagonal similarity: {max_off_diag:.6f} (should be ~0)")
    assert max_off_diag < 0.01, f"Features not orthogonal! Max sim: {max_off_diag}"
    
    print("  ✓ FeatureSet tests passed!")
    return True


def test_compositional_embedding(xp: ArrayModule = np) -> bool:
    """Test compositional embedding."""
    print("Testing CompositionalEmbedding...")
    
    emb = CompositionalEmbedding(vocab_size=100, num_features=14, xp=xp)  # Max 14
    
    # Test single embedding
    e0 = emb.get_embedding(0)
    assert e0.shape == (4, 4)
    print("  ✓ get_embedding")
    
    # Test all embeddings
    all_emb = emb.get_all_embeddings()
    assert all_emb.shape == (100, 4, 4)
    print("  ✓ get_all_embeddings")
    
    # Test similarity
    sim = emb.embedding_similarity(0, 1)
    assert -1.5 <= sim <= 1.5  # Frobenius sim can exceed 1
    print(f"  ✓ embedding_similarity: {sim:.4f}")
    
    # Test feature similarity
    fsim = emb.feature_similarity(0, 1)
    assert -1.0 <= fsim <= 1.0
    print(f"  ✓ feature_similarity: {fsim:.4f}")
    
    print("  ✓ CompositionalEmbedding tests passed!")
    return True


def test_shared_features_cluster(xp: ArrayModule = np) -> bool:
    """
    KEY TEST: Words with shared features should have similar embeddings.
    """
    print("Testing: shared features → similar embeddings...")
    
    emb = CompositionalEmbedding(vocab_size=100, num_features=14, xp=xp)
    
    # Create two words with SAME features
    shared_features = [0, 5, 10, 15]
    shared_coeffs = [0.5, 0.3, 0.4, 0.2]
    
    emb.set_word_features(50, shared_features, shared_coeffs)
    emb.set_word_features(51, shared_features, shared_coeffs)
    
    # Create a word with DIFFERENT features
    diff_features = [1, 6, 11, 16]
    diff_coeffs = [0.5, 0.3, 0.4, 0.2]
    emb.set_word_features(52, diff_features, diff_coeffs)
    
    # Similarity between same-feature words should be HIGH
    sim_same = emb.embedding_similarity(50, 51)
    
    # Similarity between different-feature words should be LOWER
    sim_diff_1 = emb.embedding_similarity(50, 52)
    sim_diff_2 = emb.embedding_similarity(51, 52)
    
    print(f"  Same features (50,51): {sim_same:.4f}")
    print(f"  Diff features (50,52): {sim_diff_1:.4f}")
    print(f"  Diff features (51,52): {sim_diff_2:.4f}")
    
    # KEY ASSERTION: Same features should cluster
    assert sim_same > sim_diff_1, "Same-feature words should be more similar!"
    assert sim_same > sim_diff_2, "Same-feature words should be more similar!"
    
    print("  ✓ Shared features → higher similarity!")
    return True


def test_one_shot_learning(xp: ArrayModule = np) -> bool:
    """
    Test one-shot learning: set features for new word, verify it clusters correctly.
    """
    print("Testing one-shot learning...")
    
    emb = CompositionalEmbedding(vocab_size=100, num_features=14, xp=xp)
    
    # Define "animal" features
    animal_features = [0, 1, 2]  # animate, concrete, living
    animal_coeffs = [0.5, 0.3, 0.4]
    
    # Set up "dog" and "cat" with animal features + unique features
    emb.set_word_features(10, animal_features + [5], animal_coeffs + [0.3])  # dog
    emb.set_word_features(11, animal_features + [6], animal_coeffs + [0.3])  # cat
    
    # Set up "rock" with different features
    emb.set_word_features(20, [7, 8, 9], [0.5, 0.3, 0.4])  # rock: inanimate
    
    # Now "one-shot learn" a new animal "zebra"
    # We infer it has animal features + unique zebra feature
    emb.set_word_features(30, animal_features + [10], animal_coeffs + [0.3])  # zebra
    
    # Zebra should be more similar to dog/cat than to rock
    sim_zebra_dog = emb.embedding_similarity(30, 10)
    sim_zebra_cat = emb.embedding_similarity(30, 11)
    sim_zebra_rock = emb.embedding_similarity(30, 20)
    
    print(f"  zebra-dog: {sim_zebra_dog:.4f}")
    print(f"  zebra-cat: {sim_zebra_cat:.4f}")
    print(f"  zebra-rock: {sim_zebra_rock:.4f}")
    
    assert sim_zebra_dog > sim_zebra_rock, "Zebra should be closer to dog than rock!"
    assert sim_zebra_cat > sim_zebra_rock, "Zebra should be closer to cat than rock!"
    
    print("  ✓ One-shot learning works: zebra clusters with animals!")
    return True


def run_compositional_tests(xp: ArrayModule = np) -> bool:
    """Run all compositional embedding tests."""
    print("=" * 60)
    print("COMPOSITIONAL EMBEDDING TESTS")
    print("=" * 60)
    
    all_pass = True
    
    if not test_feature_set(xp):
        all_pass = False
    print()
    
    if not test_compositional_embedding(xp):
        all_pass = False
    print()
    
    if not test_shared_features_cluster(xp):
        all_pass = False
    print()
    
    if not test_one_shot_learning(xp):
        all_pass = False
    print()
    
    print("=" * 60)
    if all_pass:
        print("ALL COMPOSITIONAL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    
    return all_pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'FeatureSet',
    'create_feature_set',
    'CompositionalEmbedding',
    'test_feature_set',
    'test_compositional_embedding',
    'test_shared_features_cluster',
    'test_one_shot_learning',
    'run_compositional_tests',
]


if __name__ == "__main__":
    run_compositional_tests()
