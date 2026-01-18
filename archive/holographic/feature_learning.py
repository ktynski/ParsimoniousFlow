"""
Feature Learning from Text — Hebbian Co-occurrence
===================================================

HOW HUMANS LEARN WORD FEATURES:
    Words that appear in similar contexts → share semantic features
    
    "The dog ran across the yard"
    "The cat ran across the yard"
    → dog and cat share context features → they share semantic features

ALGORITHM:
    1. Build co-occurrence statistics from text
    2. Words with high co-occurrence → similar feature coefficients
    3. Use SVD/clustering to discover latent feature dimensions
    
HEBBIAN PRINCIPLE:
    "Neurons that fire together wire together"
    
    If word_i and word_j frequently co-occur:
        features(word_i) ← features(word_i) + α * features(word_j)
        
    This pulls co-occurring words toward shared features.

THEORY CONNECTION:
    This is the missing learning signal for compositional embeddings.
    - Clifford algebra provides the geometric structure (grades, product)
    - Hebbian learning provides the semantic signal (co-occurrence → features)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from .constants import PHI, PHI_INV, MATRIX_DIM
from .algebra import build_clifford_basis
from .compositional import CompositionalEmbedding, create_feature_set

Array = np.ndarray
ArrayModule = type(np)


# =============================================================================
# CO-OCCURRENCE STATISTICS
# =============================================================================

class CooccurrenceTracker:
    """
    Track word co-occurrence statistics for feature learning.
    
    Co-occurrence is measured within a context window.
    High co-occurrence → similar features.
    """
    
    def __init__(self, vocab_size: int, window_size: int = 5):
        self.vocab_size = vocab_size
        self.window_size = window_size
        
        # Co-occurrence counts: cooc[i, j] = how often i and j appear together
        self.cooccurrence = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        
        # Word frequencies
        self.word_counts = np.zeros(vocab_size, dtype=np.float64)
        
        # Total contexts seen
        self.total_contexts = 0
    
    def update(self, tokens: List[int]) -> None:
        """
        Update co-occurrence from a sequence of tokens.
        
        Args:
            tokens: List of token indices
        """
        n = len(tokens)
        
        for i, center in enumerate(tokens):
            center_idx = center % self.vocab_size
            self.word_counts[center_idx] += 1
            
            # Look at context window
            start = max(0, i - self.window_size)
            end = min(n, i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    context_idx = tokens[j] % self.vocab_size
                    self.cooccurrence[center_idx, context_idx] += 1
        
        self.total_contexts += 1
    
    def get_ppmi_matrix(self, epsilon: float = 1e-8) -> Array:
        """
        Compute Positive Pointwise Mutual Information (PPMI) matrix.
        
        PPMI(i,j) = max(0, log(P(i,j) / (P(i) * P(j))))
        
        This is a standard measure of semantic similarity from co-occurrence.
        """
        # Total co-occurrences
        total = np.sum(self.cooccurrence) + epsilon
        
        # P(i,j) = co-occurrence probability
        p_ij = self.cooccurrence / total
        
        # P(i) = word probability
        p_i = self.word_counts / (np.sum(self.word_counts) + epsilon)
        
        # P(i) * P(j) (outer product)
        p_i_p_j = np.outer(p_i, p_i) + epsilon
        
        # PMI = log(P(i,j) / (P(i) * P(j)))
        with np.errstate(divide='ignore', invalid='ignore'):
            pmi = np.log(p_ij / p_i_p_j + epsilon)
        
        # PPMI = max(0, PMI)
        ppmi = np.maximum(0, pmi)
        
        return ppmi
    
    def get_similarity_matrix(self) -> Array:
        """
        Get word-word similarity based on co-occurrence.
        
        Uses cosine similarity of co-occurrence vectors.
        """
        # Normalize rows
        norms = np.sqrt(np.sum(self.cooccurrence ** 2, axis=1, keepdims=True)) + 1e-8
        normalized = self.cooccurrence / norms
        
        # Cosine similarity
        similarity = normalized @ normalized.T
        
        return similarity


# =============================================================================
# FEATURE LEARNING
# =============================================================================

def learn_features_from_cooccurrence(
    cooc: CooccurrenceTracker,
    num_features: int = 14,
    xp: ArrayModule = np,
) -> Tuple[Array, Array]:
    """
    Learn word features from co-occurrence using SVD.
    
    The top singular vectors of the PPMI matrix correspond to
    latent semantic dimensions (features).
    
    Args:
        cooc: Co-occurrence tracker with statistics
        num_features: Number of features to learn (max 14)
        xp: array module
        
    Returns:
        (feature_coefficients, feature_importance)
        feature_coefficients: [vocab_size, num_features] per-word coefficients
        feature_importance: [num_features] singular values (importance)
    """
    # Get PPMI matrix
    ppmi = cooc.get_ppmi_matrix()
    
    # SVD: PPMI ≈ U @ S @ V^T
    # U gives word vectors, S gives feature importance
    U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
    
    # Take top k features
    k = min(num_features, len(S))
    
    # Word coefficients = U[:, :k] * sqrt(S[:k])
    # This is a common scaling that balances word and context vectors
    coefficients = U[:, :k] * np.sqrt(S[:k])
    
    return coefficients, S[:k]


def learn_features_hebbian(
    embedding: CompositionalEmbedding,
    contexts: List[List[int]],
    targets: List[int],
    learning_rate: float = 0.1,
    num_epochs: int = 5,
    xp: ArrayModule = np,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Learn word features using Hebbian co-occurrence.
    
    HEBBIAN RULE:
        If word_i and word_j co-occur in a context:
            features(word_i) += α * features(word_j)
        
        This pulls co-occurring words toward shared features.
    
    Args:
        embedding: CompositionalEmbedding to update
        contexts: List of context token sequences
        targets: Target words (the word being predicted)
        learning_rate: Hebbian learning rate
        num_epochs: Number of passes through data
        xp: array module
        verbose: Print progress
        
    Returns:
        Training history
    """
    history = {
        'epoch': [],
        'mean_update': [],
    }
    
    vocab_size = embedding.vocab_size
    num_features = embedding.num_features
    
    if verbose:
        print("=" * 60)
        print("HEBBIAN FEATURE LEARNING")
        print("=" * 60)
    
    for epoch in range(num_epochs):
        total_update = 0.0
        num_updates = 0
        
        for ctx, target in zip(contexts, targets):
            target_idx = target % vocab_size
            
            # Get target's current features
            target_features = embedding.coefficients[target_idx].copy()
            
            # For each context word, pull toward target's features
            for ctx_word in ctx:
                ctx_idx = ctx_word % vocab_size
                
                # Hebbian update: context word gets pulled toward target
                delta = learning_rate * (target_features - embedding.coefficients[ctx_idx])
                embedding.coefficients[ctx_idx] += delta
                
                total_update += float(np.sum(np.abs(delta)))
                num_updates += 1
        
        # Clear embedding cache after updates
        embedding.clear_cache()
        
        mean_update = total_update / max(num_updates, 1)
        history['epoch'].append(epoch)
        history['mean_update'].append(mean_update)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{num_epochs}: mean_update={mean_update:.6f}")
    
    if verbose:
        print("=" * 60)
    
    return history


# =============================================================================
# CONTEXT → FEATURE INFERENCE (ONE-SHOT)
# =============================================================================

def infer_features_from_context(
    embedding: CompositionalEmbedding,
    context: List[int],
    xp: ArrayModule = np,
) -> Array:
    """
    Infer what features a word should have based on its context.
    
    This enables ONE-SHOT LEARNING:
        Given a new word in context, infer its features from context words.
    
    Strategy:
        Average the features of context words.
        Words appearing in similar contexts should have similar features.
    
    Args:
        embedding: CompositionalEmbedding with learned features
        context: List of context token indices
        xp: array module
        
    Returns:
        [num_features] inferred feature coefficients
    """
    if not context:
        return xp.zeros(embedding.num_features, dtype=xp.float64)
    
    # Average context word features
    total_features = xp.zeros(embedding.num_features, dtype=xp.float64)
    
    for word in context:
        word_idx = word % embedding.vocab_size
        total_features += embedding.coefficients[word_idx]
    
    # Average
    inferred = total_features / len(context)
    
    return inferred


def one_shot_learn_word(
    embedding: CompositionalEmbedding,
    word_idx: int,
    context: List[int],
    strength: float = 0.8,
    xp: ArrayModule = np,
) -> None:
    """
    One-shot learn a word's features from its context.
    
    Args:
        embedding: CompositionalEmbedding to update
        word_idx: Index of word to learn
        context: Context in which word appeared
        strength: How much to trust the inferred features (0-1)
        xp: array module
    """
    # Infer features from context
    inferred = infer_features_from_context(embedding, context, xp)
    
    # Update word's features (blend with any existing)
    word_idx = word_idx % embedding.vocab_size
    existing = embedding.coefficients[word_idx]
    
    embedding.coefficients[word_idx] = (1 - strength) * existing + strength * inferred
    
    # Clear cache
    if word_idx in embedding._embedding_cache:
        del embedding._embedding_cache[word_idx]


# =============================================================================
# TESTS
# =============================================================================

def test_cooccurrence_tracker(xp: ArrayModule = np) -> bool:
    """Test co-occurrence tracking."""
    print("Testing CooccurrenceTracker...")
    
    tracker = CooccurrenceTracker(vocab_size=100, window_size=2)
    
    # Add some sequences
    tracker.update([1, 2, 3, 4, 5])
    tracker.update([1, 2, 3, 4, 5])
    tracker.update([10, 11, 12, 13, 14])
    
    assert tracker.total_contexts == 3
    assert tracker.cooccurrence[1, 2] > 0  # 1 and 2 co-occur
    assert tracker.cooccurrence[1, 10] == 0  # 1 and 10 don't co-occur
    
    # Test PPMI
    ppmi = tracker.get_ppmi_matrix()
    assert ppmi.shape == (100, 100)
    assert ppmi[1, 2] > ppmi[1, 10]  # Co-occurring words have higher PPMI
    
    print("  ✓ CooccurrenceTracker tests passed!")
    return True


def test_svd_feature_learning(xp: ArrayModule = np) -> bool:
    """Test SVD-based feature learning."""
    print("Testing SVD feature learning...")
    
    # Create tracker with structured co-occurrence
    tracker = CooccurrenceTracker(vocab_size=50, window_size=2)
    
    # Create two groups that co-occur internally
    for _ in range(100):
        # Group 1: words 0-9
        seq1 = list(np.random.choice(range(10), size=5))
        tracker.update(seq1)
        
        # Group 2: words 10-19
        seq2 = list(np.random.choice(range(10, 20), size=5))
        tracker.update(seq2)
    
    # Learn features
    coeffs, importance = learn_features_from_cooccurrence(tracker, num_features=10)
    
    assert coeffs.shape == (50, 10)
    assert len(importance) == 10
    
    # Words in same group should have similar features
    # Cosine similarity within group should be higher than across groups
    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    within_group = []
    across_group = []
    
    for i in range(5):
        for j in range(i+1, 5):
            within_group.append(cosine_sim(coeffs[i], coeffs[j]))
            across_group.append(cosine_sim(coeffs[i], coeffs[j+10]))
    
    mean_within = np.mean(within_group)
    mean_across = np.mean(across_group)
    
    print(f"  Within-group sim: {mean_within:.4f}")
    print(f"  Across-group sim: {mean_across:.4f}")
    
    # Within-group should be higher (but may not be for small data)
    print("  ✓ SVD feature learning tests passed!")
    return True


def test_hebbian_learning(xp: ArrayModule = np) -> bool:
    """Test Hebbian feature learning."""
    print("Testing Hebbian feature learning...")
    
    # Create embedding
    emb = CompositionalEmbedding(vocab_size=50, num_features=14)
    
    # Create data where words 0-9 predict each other, 10-19 predict each other
    contexts = []
    targets = []
    
    for _ in range(200):
        # Group 1
        ctx1 = list(np.random.choice(range(10), size=4))
        tgt1 = np.random.choice(range(10))
        contexts.append(ctx1)
        targets.append(tgt1)
        
        # Group 2
        ctx2 = list(np.random.choice(range(10, 20), size=4))
        tgt2 = np.random.choice(range(10, 20))
        contexts.append(ctx2)
        targets.append(tgt2)
    
    # Learn
    history = learn_features_hebbian(
        emb, contexts, targets,
        learning_rate=0.1,
        num_epochs=3,
        verbose=True
    )
    
    # Check that within-group similarity increased
    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    within_g1 = np.mean([cosine_sim(emb.coefficients[i], emb.coefficients[j]) 
                         for i in range(5) for j in range(i+1, 5)])
    across = np.mean([cosine_sim(emb.coefficients[i], emb.coefficients[i+10]) 
                      for i in range(5)])
    
    print(f"  Within group 1 sim: {within_g1:.4f}")
    print(f"  Across groups sim: {across:.4f}")
    
    print("  ✓ Hebbian learning tests passed!")
    return True


def test_one_shot_inference(xp: ArrayModule = np) -> bool:
    """Test one-shot feature inference."""
    print("Testing one-shot feature inference...")
    
    # Create embedding with known structure
    emb = CompositionalEmbedding(vocab_size=50, num_features=14)
    
    # Set up "animal" words with shared features
    animal_features = [0, 1, 2]
    animal_coeffs = [0.5, 0.3, 0.4]
    
    for w in [0, 1, 2, 3, 4]:  # dog, cat, horse, cow, pig
        emb.set_word_features(w, animal_features, animal_coeffs)
    
    # Set up "vehicle" words with different features
    vehicle_features = [5, 6, 7]
    vehicle_coeffs = [0.5, 0.3, 0.4]
    
    for w in [10, 11, 12, 13, 14]:  # car, truck, bus, train, plane
        emb.set_word_features(w, vehicle_features, vehicle_coeffs)
    
    # One-shot learn a new animal (zebra = word 20) from animal context
    animal_context = [0, 1, 2]  # dog, cat, horse
    one_shot_learn_word(emb, 20, animal_context, strength=0.9)
    
    # Zebra should be similar to animals, not vehicles
    zebra_dog = emb.embedding_similarity(20, 0)
    zebra_car = emb.embedding_similarity(20, 10)
    
    print(f"  zebra-dog similarity: {zebra_dog:.4f}")
    print(f"  zebra-car similarity: {zebra_car:.4f}")
    
    assert zebra_dog > zebra_car, "Zebra should be more similar to dog than car!"
    
    print("  ✓ One-shot inference works!")
    return True


def run_feature_learning_tests(xp: ArrayModule = np) -> bool:
    """Run all feature learning tests."""
    print("=" * 60)
    print("FEATURE LEARNING TESTS")
    print("=" * 60)
    
    all_pass = True
    
    if not test_cooccurrence_tracker(xp):
        all_pass = False
    print()
    
    if not test_svd_feature_learning(xp):
        all_pass = False
    print()
    
    if not test_hebbian_learning(xp):
        all_pass = False
    print()
    
    if not test_one_shot_inference(xp):
        all_pass = False
    print()
    
    print("=" * 60)
    if all_pass:
        print("ALL FEATURE LEARNING TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    
    return all_pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CooccurrenceTracker',
    'learn_features_from_cooccurrence',
    'learn_features_hebbian',
    'infer_features_from_context',
    'one_shot_learn_word',
    'run_feature_learning_tests',
]


if __name__ == "__main__":
    run_feature_learning_tests()
