"""
Shared Fixtures for Theory Validation Tests
============================================

This module provides pytest fixtures shared across all test modules:
    - basis: Clifford basis (built once per session)
    - embeddings: Identity-biased embeddings (vocab_size=1000)
    - gutenberg_sentences: Real text iterator from Gutenberg corpus
    - trained_model: Pre-trained model (built for language tests)
    - xp: Array module (numpy, or cupy if available)

PRINCIPLE: Build expensive fixtures once, reuse everywhere.
"""

import pytest
import numpy as np
from typing import Iterator, List, Optional
import os

# Import holographic_v4 components
from holographic_v4 import (
    build_clifford_basis,
    initialize_embeddings_identity,
    geometric_product_batch,
    normalize_matrix,
    grace_operator,
    PHI, PHI_INV, PHI_INV_SQ,
)
from holographic_v4.holographic_memory import HybridHolographicMemory
from holographic_v4.dreaming import DreamingSystem
from holographic_v4.constants import DTYPE


# =============================================================================
# SESSION-SCOPED FIXTURES (Built once per test session)
# =============================================================================

@pytest.fixture(scope="session")
def xp():
    """
    Array module — numpy by default, cupy if available.
    """
    try:
        import cupy as cp
        # Test that CUDA is actually available
        cp.cuda.runtime.getDeviceCount()
        return cp
    except (ImportError, RuntimeError):
        return np


@pytest.fixture(scope="session")
def basis(xp):
    """
    Clifford Cl(3,1) basis — 16 matrices of shape [4, 4].
    Built once per session.
    """
    return build_clifford_basis(xp)


@pytest.fixture(scope="session")
def vocab_size():
    """Standard vocabulary size for tests."""
    return 1000


@pytest.fixture(scope="session")
def embeddings(vocab_size, xp):
    """
    Identity-biased embeddings for all tokens.
    Shape: [vocab_size, 4, 4]
    """
    return initialize_embeddings_identity(
        vocab_size=vocab_size,
        noise_std=PHI_INV,  # Theory-derived
        xp=xp,
        seed=42
    )


@pytest.fixture(scope="session")
def memory(basis, xp):
    """
    Hybrid holographic memory for testing.
    """
    return HybridHolographicMemory.create(
        basis=basis,
        xp=xp
    )


@pytest.fixture(scope="session")
def dreaming_system(basis, xp):
    """
    Dreaming system for schema/prototype tests.
    """
    return DreamingSystem(basis=basis, xp=xp)


# =============================================================================
# TEXT DATA FIXTURES
# =============================================================================

def _load_gutenberg_sentences(max_sentences: int = 10000) -> List[str]:
    """
    Load sentences from Gutenberg corpus.
    Uses HuggingFace datasets if available, falls back to synthetic.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "nikolina-p/gutenberg_clean_tokenized_en",
            split="train",
            streaming=True
        )
        
        sentences = []
        for item in ds:
            text = item.get('text', '')
            # Split into sentences
            for sent in text.split('.'):
                sent = sent.strip()
                if len(sent) > 20 and len(sent.split()) >= 4:
                    sentences.append(sent)
                    if len(sentences) >= max_sentences:
                        break
            if len(sentences) >= max_sentences:
                break
        
        return sentences
    
    except Exception as e:
        print(f"Warning: Could not load Gutenberg data: {e}")
        print("Using synthetic sentences for testing")
        return _generate_synthetic_sentences(max_sentences)


def _generate_synthetic_sentences(n: int = 10000) -> List[str]:
    """
    Generate synthetic but grammatically varied sentences.
    For testing when Gutenberg is unavailable.
    """
    np.random.seed(42)
    
    # Templates with varied grammatical structures
    templates = [
        "The {adj} {noun} {verb} the {adj2} {noun2}",
        "{noun} {verb} {adv}",
        "While the {noun} {verb}, the {noun2} {verb2}",
        "The {adj} {noun} that {verb} the {noun2} {verb2} {adv}",
        "{noun} and {noun2} {verb} together",
        "When {noun} {verb}, {noun2} always {verb2}",
        "The {noun} {verb} because the {noun2} {verb2}",
    ]
    
    nouns = ["dog", "cat", "man", "woman", "child", "bird", "tree", "river", "mountain", "house"]
    verbs = ["runs", "jumps", "sleeps", "eats", "watches", "follows", "loves", "fears", "finds", "sees"]
    adjs = ["large", "small", "quick", "slow", "bright", "dark", "old", "young", "quiet", "loud"]
    advs = ["quickly", "slowly", "carefully", "happily", "sadly", "loudly", "quietly", "suddenly"]
    
    sentences = []
    for _ in range(n):
        template = templates[np.random.randint(len(templates))]
        sent = template.format(
            noun=np.random.choice(nouns),
            noun2=np.random.choice(nouns),
            verb=np.random.choice(verbs),
            verb2=np.random.choice(verbs),
            adj=np.random.choice(adjs),
            adj2=np.random.choice(adjs),
            adv=np.random.choice(advs),
        )
        sentences.append(sent)
    
    return sentences


@pytest.fixture(scope="session")
def gutenberg_sentences() -> List[str]:
    """
    Real sentences from Gutenberg corpus.
    Loaded once per session.
    """
    return _load_gutenberg_sentences(max_sentences=10000)


@pytest.fixture(scope="module")
def tokenizer():
    """
    Simple whitespace tokenizer.
    Returns function: str -> List[int]
    """
    # Build vocabulary from common words
    vocab = {}
    idx = 0
    
    common_words = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "and", "or", "but", "if", "then", "else", "when", "where", "who",
        "what", "which", "how", "why", "i", "you", "he", "she", "it", "we",
        "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
        "our", "their", "this", "that", "these", "those", "here", "there",
        "to", "from", "in", "out", "on", "off", "up", "down", "over", "under",
        "of", "for", "with", "at", "by", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "among", "while",
        "dog", "cat", "man", "woman", "child", "bird", "tree", "house", "river",
        "mountain", "sun", "moon", "star", "sky", "earth", "water", "fire",
        "large", "small", "big", "little", "old", "young", "new", "good", "bad",
        "run", "walk", "jump", "eat", "drink", "sleep", "see", "hear", "speak",
        "quickly", "slowly", "carefully", "happily", "sadly", "suddenly",
    ]
    
    for word in common_words:
        vocab[word.lower()] = idx
        idx += 1
    
    def tokenize(text: str, max_tokens: int = 50) -> List[int]:
        words = text.lower().split()
        tokens = []
        for word in words[:max_tokens]:
            # Strip punctuation
            word = ''.join(c for c in word if c.isalnum())
            if word in vocab:
                tokens.append(vocab[word])
            else:
                # Hash unknown words to vocabulary range
                tokens.append(hash(word) % len(vocab))
        return tokens
    
    return tokenize


# =============================================================================
# HELPER FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def random_context(embeddings, basis, xp):
    """
    Generate a random context matrix.
    Returns function: (n_tokens) -> [4, 4] matrix
    """
    def _make_context(n_tokens: int = 5, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        
        vocab_size = embeddings.shape[0]
        token_ids = np.random.randint(0, vocab_size, size=n_tokens)
        tokens = embeddings[token_ids]
        
        # Compose via geometric product
        ctx = normalize_matrix(geometric_product_batch(tokens, xp), xp)
        return ctx
    
    return _make_context


@pytest.fixture(scope="function")
def sentence_to_context(embeddings, tokenizer, basis, xp):
    """
    Convert a sentence to its context matrix.
    Returns function: (str) -> [4, 4] matrix
    """
    def _convert(sentence: str) -> np.ndarray:
        tokens = tokenizer(sentence)
        if not tokens:
            return np.eye(4, dtype=DTYPE)
        
        vocab_size = embeddings.shape[0]
        token_ids = [t % vocab_size for t in tokens]
        token_matrices = embeddings[token_ids]
        
        ctx = normalize_matrix(geometric_product_batch(token_matrices, xp), xp)
        return ctx
    
    return _convert


@pytest.fixture(scope="session")
def grace_trajectory(basis, xp):
    """
    Returns function to compute Grace trajectory from initial matrix.
    """
    def _trajectory(M: np.ndarray, n_steps: int = 10) -> np.ndarray:
        trajectory = [M.copy()]
        current = M.copy()
        
        for _ in range(n_steps):
            current = grace_operator(current, basis, xp)
            trajectory.append(current.copy())
        
        return np.array(trajectory)
    
    return _trajectory


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session")
def n_tests_in_module():
    """
    Number of tests per module for Bonferroni correction.
    Adjust per module as needed.
    """
    return {
        'foundations': 5,
        'dynamics': 6,
        'information': 4,
        'memory': 6,
        'language': 8,
        'creativity': 4,
    }


@pytest.fixture(scope="session")
def significance_alpha():
    """Significance level before Bonferroni correction."""
    return 0.05


# =============================================================================
# SKIP CONDITIONS
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests requiring CUDA GPU"
    )
    config.addinivalue_line(
        "markers", "requires_network: marks tests requiring network access"
    )


@pytest.fixture(scope="session")
def skip_if_no_gpu(xp):
    """Skip test if GPU not available."""
    if xp.__name__ != 'cupy':
        pytest.skip("Test requires CUDA GPU")


@pytest.fixture(scope="session")
def skip_if_no_network():
    """Skip test if network not available."""
    try:
        import socket
        socket.create_connection(("huggingface.co", 443), timeout=5)
    except OSError:
        pytest.skip("Test requires network access")
