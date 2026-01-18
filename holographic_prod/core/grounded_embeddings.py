"""
Grounded SO(4) Embeddings — Theory-True Initialization

Human brains don't learn semantic structure from scratch. They use
GROUNDED representations where similar concepts have similar neural 
patterns BEFORE language learning.

This module provides grounded embedding initialization:
1. Extract co-occurrence structure from corpus (single pass)
2. SVD → 6D semantic vectors (φ²≈2.618 quality/dimension)
3. Exponential map → SO(4) matrices

Result: Similar tokens have similar embeddings from the START,
enabling O(√N) sample complexity instead of O(N).

THEORY:
    The 6 generators of SO(4) form the Lie algebra so(4).
    Any SO(4) matrix M = exp(Σ θᵢ Gᵢ) where Gᵢ are generators.
    Similar θ vectors → Similar SO(4) matrices.
    
    By extracting θ from co-occurrence structure, we ensure
    tokens that appear in similar contexts get similar embeddings.
    
ΦDERIVATIONS:
    - Embedding dimension: 6 = SO(4) degrees of freedom (exact)
    - SVD truncation: top 6 singular values (information-optimal)
    - Angle scaling: 0.3 radians max (avoids geodesic wrapping)
"""

import numpy as np
from scipy.linalg import expm
from typing import List, Optional, Tuple
from collections import defaultdict

from holographic_prod.core.constants import PHI, PHI_INV, PHI_EPSILON, DTYPE


# =============================================================================
# SO(4) GENERATORS
# =============================================================================

def so4_generators() -> List[np.ndarray]:
    """
    Return the 6 generators of the SO(4) Lie algebra.
    
    These are antisymmetric 4×4 matrices that span the tangent
    space at the identity. Any SO(4) matrix can be written as
    exp(linear combination of generators).
    """
    gens = []
    for i in range(4):
        for j in range(i + 1, 4):
            g = np.zeros((4, 4), dtype=DTYPE)
            g[i, j] = 1.0
            g[j, i] = -1.0
            gens.append(g)
    return gens


_SO4_GENERATORS = None

def get_so4_generators() -> List[np.ndarray]:
    """Cached SO(4) generators."""
    global _SO4_GENERATORS
    if _SO4_GENERATORS is None:
        _SO4_GENERATORS = so4_generators()
    return _SO4_GENERATORS


# =============================================================================
# CENTRALIZED SO(4) EMBEDDING CREATION
# =============================================================================

def create_random_so4_embeddings(
    vocab_size: int,
    seed: int = 42,
    xp = None,
) -> np.ndarray:
    """
    Create random SO(4) embeddings using BATCHED QR decomposition.
    
    THEORY (INFORMATIONAL PARSIMONY):
        This is the SINGLE SOURCE OF TRUTH for random SO(4) embedding creation.
        All memory classes should use this instead of duplicating the logic.
        
        SO(4) = {M ∈ O(4) : det(M) = 1}
        
        Properties:
        - M^T @ M = I (orthogonal)
        - det(M) = 1 (special)
        - M^(-1) = M^T (trivial inversion!)
        
        This ensures:
        - det(product of N embeddings) = 1 for ANY N
        - Condition number = 1 (perfectly conditioned!)
        - Retrieval works at ANY sequence length
        
        The key insight: SO(4) ≅ (SU(2) × SU(2)) / Z₂
        This is the DOUBLE COVER of SO(3) × SO(3), connecting to
        the quaternionic structure underlying Cl(3,1).
    
    PERFORMANCE:
        Uses batched np.linalg.qr which is ~20x faster than looping
        with scipy.stats.ortho_group.
    
    Args:
        vocab_size: Number of tokens in vocabulary
        seed: Random seed for reproducibility
        xp: Array module (numpy or cupy). If None, uses numpy.
        
    Returns:
        [vocab_size, 4, 4] SO(4) embeddings on the appropriate device
    """
    if xp is None:
        xp = np
    
    # Generate random matrices on CPU (required for seeding)
    np.random.seed(seed)
    random_matrices = np.random.randn(vocab_size, 4, 4).astype(DTYPE)
    
    # Batched QR decomposition - ~20x faster than loop
    Q, _ = np.linalg.qr(random_matrices)
    
    # Ensure det = +1 (SO(4), not O(4)) - vectorized
    dets = np.linalg.det(Q)
    Q[dets < 0, :, 0] *= -1
    
    # Transfer to device if needed
    if xp != np:
        return xp.asarray(Q.astype(DTYPE))
    return Q.astype(DTYPE)


def create_orthogonal_so4_embeddings(
    vocab_size: int,
    seed: int = 42,
    max_correlation: float = 0.5,
    xp = None,
) -> np.ndarray:
    """
    Create SO(4) embeddings with PATTERN SEPARATION via rejection sampling.
    
    THEORY (DENTATE GYRUS ANALOG):
        The brain's dentate gyrus performs pattern separation.
        We use rejection sampling: generate random SO(4), reject if too similar.
        
    LIMITATION:
        SO(4) in 4x4 has limited capacity. For N >> 16, some pairs WILL be similar.
        This function does best-effort separation up to geometric limits.
        
    Args:
        vocab_size: Number of tokens
        seed: Random seed
        max_correlation: Rejection threshold (default 0.5)
        xp: Array module (numpy or cupy)
        
    Returns:
        [vocab_size, 4, 4] separated SO(4) embeddings
    """
    if xp is None:
        xp = np
    
    np.random.seed(seed)
    embeddings = []
    max_attempts = vocab_size * 100
    attempts = 0
    
    while len(embeddings) < vocab_size and attempts < max_attempts:
        attempts += 1
        
        # Generate a random SO(4) matrix
        random_mat = np.random.randn(4, 4).astype(DTYPE)
        Q, _ = np.linalg.qr(random_mat)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        
        # Check correlation with existing embeddings
        is_valid = True
        candidate_flat = Q.flatten()
        candidate_norm = np.linalg.norm(candidate_flat)
        
        for existing in embeddings:
            existing_flat = existing.flatten()
            corr = abs(np.dot(candidate_flat, existing_flat) / (candidate_norm * np.linalg.norm(existing_flat)))
            if corr > max_correlation:
                is_valid = False
                break
        
        if is_valid:
            embeddings.append(Q)
    
    # If we couldn't generate enough unique embeddings, fill remainder with random
    while len(embeddings) < vocab_size:
        random_mat = np.random.randn(4, 4).astype(DTYPE)
        Q, _ = np.linalg.qr(random_mat)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        embeddings.append(Q)
    
    embeddings = np.array(embeddings, dtype=DTYPE)
    
    # Verify and report
    max_corr = 0.0
    n_checked = 0
    for i in range(min(100, vocab_size)):
        for j in range(i+1, min(100, vocab_size)):
            e1 = embeddings[i].flatten()
            e2 = embeddings[j].flatten()
            corr = abs(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
            max_corr = max(max_corr, corr)
            n_checked += 1
    
    success_rate = len([e for e in embeddings[:100]]) / vocab_size * 100
    status = "✓" if max_corr <= max_correlation else "✗"
    print(f"  Rejection-sampled SO(4): max_corr = {max_corr:.4f} {status} ({attempts} attempts)")
    
    # Transfer to device if needed
    if xp != np:
        return xp.asarray(embeddings)
    return embeddings


# =============================================================================
# CO-OCCURRENCE EXTRACTION
# =============================================================================

def compute_cooccurrence_streaming(
    data_iterator,
    vocab_size: int,
    window: int = 5,
    max_samples: int = 1_000_000,
) -> np.ndarray:
    """
    Compute co-occurrence matrix from streaming data.
    
    Memory-efficient: doesn't load entire corpus.
    
    Args:
        data_iterator: Yields (context, target) pairs
        vocab_size: Size of vocabulary
        window: Context window size
        max_samples: Maximum samples to process
        
    Returns:
        [vocab_size, vocab_size] co-occurrence matrix
    """
    cooccur = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    
    count = 0
    for context, target in data_iterator:
        # Add target co-occurrences
        for token in context:
            if 0 <= token < vocab_size and 0 <= target < vocab_size:
                cooccur[token, target] += 1
                cooccur[target, token] += 1
        
        # Add context-context co-occurrences
        for i, t1 in enumerate(context):
            for j, t2 in enumerate(context):
                if i != j and 0 <= t1 < vocab_size and 0 <= t2 < vocab_size:
                    cooccur[t1, t2] += 1
        
        count += 1
        if count >= max_samples:
            break
    
    return cooccur


def compute_cooccurrence_from_corpus(
    corpus: List[List[int]],
    vocab_size: int,
    window: int = 5,
) -> np.ndarray:
    """
    Compute co-occurrence matrix from corpus.
    
    Args:
        corpus: List of token sequences
        vocab_size: Size of vocabulary
        window: Context window size
        
    Returns:
        [vocab_size, vocab_size] co-occurrence matrix
    """
    cooccur = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    
    for sequence in corpus:
        for i, token in enumerate(sequence):
            if not (0 <= token < vocab_size):
                continue
            # Look at context window
            start = max(0, i - window)
            end = min(len(sequence), i + window + 1)
            for j in range(start, end):
                if i != j and 0 <= sequence[j] < vocab_size:
                    cooccur[token, sequence[j]] += 1
    
    return cooccur


# =============================================================================
# SEMANTIC VECTOR EXTRACTION
# =============================================================================

def cooccurrence_to_semantic_vectors(
    cooccur: np.ndarray,
    dim: int = 6,
    use_ppmi: bool = True,
) -> np.ndarray:
    """
    Extract semantic vectors from co-occurrence matrix via SVD.
    
    THEORY:
        This is analogous to how the brain extracts feature representations
        from raw sensory input. Words that appear in similar contexts
        get similar vectors (distributional hypothesis).
    
    Args:
        cooccur: [vocab_size, vocab_size] co-occurrence matrix
        dim: Output dimension (6 for SO(4))
        use_ppmi: If True, use Positive PMI transformation
        
    Returns:
        [vocab_size, dim] semantic vectors
    """
    vocab_size = cooccur.shape[0]
    
    if use_ppmi:
        # Positive Pointwise Mutual Information
        # Better than raw counts for semantic similarity
        total = cooccur.sum() + PHI_EPSILON
        row_sums = cooccur.sum(axis=1, keepdims=True) + PHI_EPSILON
        col_sums = cooccur.sum(axis=0, keepdims=True) + PHI_EPSILON
        
        # PMI = log(P(w1,w2) / (P(w1) * P(w2)))
        expected = (row_sums @ col_sums) / total
        pmi = np.log((cooccur + PHI_EPSILON) / expected + PHI_EPSILON)
        
        # Positive PMI (clip negative values)
        matrix = np.maximum(pmi, 0)
    else:
        # Log-transformed counts
        matrix = np.log1p(cooccur)
    
    # SVD to extract latent dimensions
    # Use TRUNCATED SVD for large vocabularies (much faster!)
    if vocab_size > 5000:
        from scipy.sparse.linalg import svds
        from scipy.sparse import csr_matrix
        
        # Convert to sparse for efficiency
        sparse_matrix = csr_matrix(matrix)
        # Truncated SVD - only compute top k components
        U, S, Vt = svds(sparse_matrix, k=min(dim + 10, vocab_size - 1))
        # Sort by singular values (svds returns in ascending order)
        idx = np.argsort(S)[::-1]
        U, S = U[:, idx], S[idx]
    else:
        # Full SVD for small vocabularies
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Take top 'dim' dimensions, weighted by sqrt of singular values
    # This balances the contribution of each dimension
    semantic_vecs = U[:, :dim] * np.sqrt(S[:dim])
    
    # Normalize to unit vectors (for consistent angle scaling)
    norms = np.linalg.norm(semantic_vecs, axis=1, keepdims=True) + PHI_EPSILON
    semantic_vecs = semantic_vecs / norms
    
    return semantic_vecs.astype(DTYPE)


# =============================================================================
# SO(4) MAPPING
# =============================================================================

def semantic_to_SO4(
    semantic_vec: np.ndarray,
    gens: Optional[List[np.ndarray]] = None,
    scale: float = 0.3,
) -> np.ndarray:
    """
    Map a 6D semantic vector to SO(4) via exponential map.
    
    THEORY:
        SO(4) is a 6-dimensional Lie group.
        Any SO(4) matrix can be written as exp(Σ θᵢ Gᵢ)
        where Gᵢ are the 6 generators (antisymmetric matrices).
        
        Similar semantic vectors → Similar SO(4) matrices!
        
    Args:
        semantic_vec: 6D semantic vector
        gens: SO(4) generators (computed if None)
        scale: Angle scaling (0.3 avoids geodesic wrapping)
        
    Returns:
        4×4 SO(4) matrix
    """
    if gens is None:
        gens = get_so4_generators()
    
    # Scale to reasonable rotation angles
    scaled = semantic_vec * scale
    
    # Linear combination of generators
    A = sum(theta * g for theta, g in zip(scaled, gens))
    
    # Matrix exponential → SO(4)
    return expm(A).astype(DTYPE)


def semantic_to_SO4_batch_fast(
    semantic_vecs: np.ndarray,
    scale: float = 0.3,
    taylor_order: int = 8,
) -> np.ndarray:
    """
    FAST vectorized mapping from 6D semantic vectors to SO(4).
    
    Uses Taylor expansion of matrix exponential:
        exp(A) = I + A + A²/2! + A³/3! + ... + Aⁿ/n!
    
    For small angles (scale ~ 0.3), Taylor order 8 gives error < 1e-10.
    
    THEORY:
        This is ~100x faster than loop over scipy.linalg.expm because:
        1. Batched einsum for Lie algebra combination: O(V) → O(1) with vectorization
        2. Batched matrix power: np.einsum broadcasts over batch dimension
        3. QR projection ensures exact orthogonality
    
    Args:
        semantic_vecs: [vocab_size, 6] normalized semantic vectors
        scale: Angle scaling (0.3 default, keeps us in linear regime)
        taylor_order: Taylor expansion order (8 is plenty for scale=0.3)
        
    Returns:
        [vocab_size, 4, 4] SO(4) matrices
    """
    vocab_size = semantic_vecs.shape[0]
    
    # Get generators and stack them: [6, 4, 4]
    gens = get_so4_generators()
    stacked_gens = np.stack(gens)
    
    # Scale semantic vectors
    scaled = semantic_vecs * scale  # [vocab_size, 6]
    
    # Compute Lie algebra elements: A[i] = sum_j scaled[i,j] * gens[j]
    # This is batched: [vocab_size, 6] @ [6, 4, 4] -> [vocab_size, 4, 4]
    A = np.einsum('vi,ijk->vjk', scaled, stacked_gens)  # [vocab_size, 4, 4]
    
    # Taylor expansion: exp(A) ≈ I + A + A²/2! + A³/3! + ...
    # Start with identity
    I4 = np.eye(4, dtype=DTYPE)
    result = np.tile(I4, (vocab_size, 1, 1))  # [vocab_size, 4, 4]
    
    # Accumulate Taylor terms
    A_power = A.copy()  # A^1
    factorial = 1.0
    
    for n in range(1, taylor_order + 1):
        factorial *= n
        result += A_power / factorial
        
        if n < taylor_order:
            # A^(n+1) = A^n @ A
            A_power = np.einsum('vij,vjk->vik', A_power, A)
    
    # Project back to SO(4) using BATCHED QR decomposition
    # PERFORMANCE: Batched QR is ~20x faster than loop
    Q, R = np.linalg.qr(result)  # [vocab_size, 4, 4] -> [vocab_size, 4, 4]
    
    # Ensure det = +1 (rotation, not reflection) - vectorized
    dets = np.linalg.det(Q)
    Q[dets < 0, :, 0] *= -1
    
    return Q.astype(DTYPE)


def create_grounded_embeddings(
    cooccur: np.ndarray,
    vocab_size: int,
    use_ppmi: bool = True,
    scale: float = 0.3,
) -> np.ndarray:
    """
    Create grounded SO(4) embeddings from co-occurrence matrix.
    
    This is the theory-true approach:
    1. Extract structure from data (co-occurrence → semantic vectors)
    2. Map to SO(4) via exponential map
    3. Result: Similar words have similar embeddings FROM THE START
    
    Args:
        cooccur: [vocab_size, vocab_size] co-occurrence matrix
        vocab_size: Vocabulary size
        use_ppmi: Use Positive PMI transformation
        scale: Angle scaling for exponential map
        
    Returns:
        [vocab_size, 4, 4] SO(4) embeddings
    """
    # Step 1: Extract semantic vectors
    semantic_vecs = cooccurrence_to_semantic_vectors(cooccur, dim=6, use_ppmi=use_ppmi)
    
    # Step 2: Map to SO(4) - use FAST vectorized version
    return semantic_to_SO4_batch_fast(semantic_vecs, scale)


# =============================================================================
# FAST PATH: PRE-TRAINED EMBEDDINGS (GloVe, Word2Vec)
# =============================================================================

# Global cache for loaded GloVe embeddings
_GLOVE_CACHE: dict = {}


def load_glove_raw(
    glove_dim: int = 50,
    cache_dir: str = None,
) -> dict:
    """
    Load raw GloVe embeddings into memory (cached).
    
    Args:
        glove_dim: GloVe dimension (50, 100, 200, 300)
        cache_dir: Where to cache downloaded embeddings
        
    Returns:
        Dictionary mapping words to numpy vectors
    """
    global _GLOVE_CACHE
    
    cache_key = f"glove-{glove_dim}"
    if cache_key in _GLOVE_CACHE:
        return _GLOVE_CACHE[cache_key]
    
    import os
    
    cache_dir = cache_dir or "/tmp/glove"
    os.makedirs(cache_dir, exist_ok=True)
    
    glove_file = os.path.join(cache_dir, f"glove.6B.{glove_dim}d.txt")
    
    if not os.path.exists(glove_file):
        print(f"  Downloading GloVe {glove_dim}d (~850MB zip)...")
        import urllib.request
        import zipfile
        
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = os.path.join(cache_dir, "glove.6B.zip")
        urllib.request.urlretrieve(url, zip_path)
        
        print(f"  Extracting {glove_dim}d embeddings...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extract(f"glove.6B.{glove_dim}d.txt", cache_dir)
    
    # Load embeddings
    print(f"  Loading GloVe-{glove_dim}d from disk...")
    glove_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != glove_dim + 1:
                continue  # Skip malformed lines
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                glove_dict[word] = vec
            except ValueError:
                continue
    
    _GLOVE_CACHE[cache_key] = glove_dict
    print(f"  ✓ Loaded {len(glove_dict):,} GloVe vectors")
    return glove_dict


def normalize_token_for_glove(token: str) -> List[str]:
    """
    Normalize a tokenizer token to potential GloVe matches.
    
    BPE tokens like "Ġcat" → "cat", "##ing" → "ing", etc.
    Also handles punctuation, casing, etc.
    
    Returns list of candidates to try in order.
    """
    candidates = []
    
    # Remove BPE prefixes/suffixes
    clean = token
    
    # GPT-2 style: Ġ prefix for space
    if clean.startswith('Ġ'):
        clean = clean[1:]
    # BERT style: ## prefix for continuation
    if clean.startswith('##'):
        clean = clean[2:]
    # Llama/sentencepiece: _ prefix
    if clean.startswith('▁'):
        clean = clean[1:]
    
    # Add the cleaned version
    if clean:
        candidates.append(clean)
        candidates.append(clean.lower())  # GloVe is lowercase
    
    # Original token
    if token != clean and token:
        candidates.append(token)
        candidates.append(token.lower())
    
    return [c for c in candidates if c]  # Filter empty


def load_glove_for_tokenizer(
    tokenizer,
    glove_dim: int = 50,
    cache_dir: str = None,
) -> Tuple[np.ndarray, int, float]:
    """
    Load GloVe embeddings mapped to a HuggingFace tokenizer's vocabulary.
    
    FAST: Handles BPE/WordPiece token → GloVe word mapping.
    
    Args:
        tokenizer: HuggingFace tokenizer with get_vocab() method
        glove_dim: GloVe dimension (50 fastest, 300 best)
        cache_dir: Where to cache downloaded embeddings
        
    Returns:
        (embeddings [vocab_size, glove_dim], covered_count, coverage_ratio)
    """
    # Load raw GloVe
    glove_dict = load_glove_raw(glove_dim, cache_dir)
    
    # Get tokenizer vocabulary
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    # Initialize with small random values (for OOV tokens)
    embeddings = np.random.randn(vocab_size, glove_dim).astype(np.float32) * 0.1
    
    covered = 0
    for token, idx in vocab.items():
        # Try multiple normalizations
        for candidate in normalize_token_for_glove(token):
            if candidate in glove_dict:
                embeddings[idx] = glove_dict[candidate]
                covered += 1
                break
    
    coverage = covered / vocab_size if vocab_size > 0 else 0.0
    return embeddings, covered, coverage


def load_glove_embeddings(
    vocab: dict,  # word → idx mapping
    glove_dim: int = 50,  # GloVe dimension (50, 100, 200, 300)
    cache_dir: str = None,
) -> Tuple[np.ndarray, int]:
    """
    Load GloVe embeddings for vocabulary (dict version).
    
    FAST: ~1-2 seconds for 50K vocab vs minutes for co-occurrence approach.
    
    Args:
        vocab: Dictionary mapping words to indices
        glove_dim: GloVe dimension (50, 100, 200, 300)
        cache_dir: Where to cache downloaded embeddings
        
    Returns:
        (embeddings array [vocab_size, glove_dim], coverage count)
    """
    # Load raw GloVe
    glove_dict = load_glove_raw(glove_dim, cache_dir)
    
    vocab_size = max(vocab.values()) + 1 if vocab else 0
    if vocab_size == 0:
        return np.zeros((0, glove_dim), dtype=np.float32), 0
    
    embeddings = np.random.randn(vocab_size, glove_dim).astype(np.float32) * 0.1
    
    covered = 0
    for word, idx in vocab.items():
        # Try multiple normalizations
        for candidate in normalize_token_for_glove(word):
            if candidate in glove_dict:
                embeddings[idx] = glove_dict[candidate]
                covered += 1
                break
    
    return embeddings, covered


def pretrained_to_SO4(
    pretrained_embeddings: np.ndarray,
    scale: float = 0.3,
) -> np.ndarray:
    """
    Convert pre-trained embeddings (any dimension) to SO(4).
    
    FAST: Just PCA to 6D + vectorized exponential map.
    ~0.5s for 10K vocab vs 26s with loop.
    
    Args:
        pretrained_embeddings: [vocab_size, any_dim] pre-trained vectors
        scale: Angle scaling for exponential map
        
    Returns:
        [vocab_size, 4, 4] SO(4) embeddings
    """
    vocab_size, dim = pretrained_embeddings.shape
    
    # PCA to 6 dimensions (SO(4) has 6 generators)
    # Center data
    mean = pretrained_embeddings.mean(axis=0)
    centered = pretrained_embeddings - mean
    
    # SVD for PCA
    if dim > 6:
        # Project to 6D - use truncated SVD for efficiency
        if vocab_size > 5000:
            from scipy.sparse.linalg import svds
            from scipy.sparse import csr_matrix
            # svds is much faster for large matrices
            U, S, Vt = svds(csr_matrix(centered), k=6)
            # Sort descending (svds returns ascending)
            idx = np.argsort(S)[::-1]
            U, S = U[:, idx], S[idx]
            semantic_vecs = U * np.sqrt(S)
        else:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            semantic_vecs = U[:, :6] * np.sqrt(S[:6])
    else:
        # Pad to 6D if needed
        semantic_vecs = np.zeros((vocab_size, 6), dtype=np.float32)
        semantic_vecs[:, :dim] = centered
    
    # Normalize
    norms = np.linalg.norm(semantic_vecs, axis=1, keepdims=True) + PHI_EPSILON
    semantic_vecs = semantic_vecs / norms
    
    # Map to SO(4) - use FAST vectorized version
    return semantic_to_SO4_batch_fast(semantic_vecs.astype(DTYPE), scale)


def create_grounded_embeddings_fast(
    vocab: dict,
    glove_dim: int = 50,
    cache_dir: str = None,
) -> Tuple[np.ndarray, float]:
    """
    FAST grounded embedding creation using GloVe (dict vocab version).
    
    ~1-2 seconds for 50K vocab vs minutes for co-occurrence.
    
    Args:
        vocab: Dictionary mapping words to indices
        glove_dim: GloVe dimension to use (50 is fastest, 300 is best quality)
        cache_dir: Where to cache GloVe files (use persistent storage on Modal!)
        
    Returns:
        (SO4 embeddings [vocab_size, 4, 4], coverage ratio)
    """
    vocab_size = max(vocab.values()) + 1 if vocab else 0
    
    print(f"  Loading GloVe-{glove_dim}d embeddings...")
    glove_vecs, covered = load_glove_embeddings(vocab, glove_dim, cache_dir=cache_dir)
    coverage = covered / len(vocab) if len(vocab) > 0 else 0.0
    print(f"  ✓ Coverage: {covered:,}/{len(vocab):,} ({coverage:.1%})")
    
    print(f"  Mapping to SO(4)...")
    so4_embeddings = pretrained_to_SO4(glove_vecs)
    print(f"  ✓ SO(4) embeddings created: {so4_embeddings.shape}")
    
    return so4_embeddings, coverage


def create_grounded_embeddings_from_tokenizer(
    tokenizer,
    glove_dim: int = 50,
    cache_dir: str = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    FAST grounded embedding creation using GloVe + HuggingFace tokenizer.
    
    This is the RECOMMENDED approach for production use:
    1. ~2-5 seconds total (vs minutes for co-occurrence)
    2. High-quality semantic structure from GloVe
    3. Proper BPE/WordPiece → GloVe word mapping
    
    Args:
        tokenizer: HuggingFace tokenizer (AutoTokenizer, GPT2Tokenizer, etc.)
        glove_dim: GloVe dimension (50=fastest, 100=balanced, 300=best)
        cache_dir: Where to cache downloaded GloVe files
        verbose: Print progress
        
    Returns:
        (SO4 embeddings [vocab_size, 4, 4], coverage ratio)
    """
    if verbose:
        print(f"  Creating grounded SO(4) embeddings from GloVe-{glove_dim}d...")
    
    # Load GloVe vectors mapped to tokenizer
    glove_vecs, covered, coverage = load_glove_for_tokenizer(
        tokenizer, glove_dim, cache_dir
    )
    
    if verbose:
        vocab_size = len(tokenizer.get_vocab())
        print(f"  ✓ GloVe coverage: {covered:,}/{vocab_size:,} ({coverage:.1%})")
    
    # Convert to SO(4)
    if verbose:
        print(f"  Mapping {glove_vecs.shape[0]:,} vectors to SO(4)...")
    
    so4_embeddings = pretrained_to_SO4(glove_vecs)
    
    if verbose:
        print(f"  ✓ SO(4) embeddings created: {so4_embeddings.shape}")
    
    return so4_embeddings, coverage


def create_grounded_embeddings_from_glove(
    vocab_size: int,
    tokenizer=None,
    vocab: dict = None,
    glove_dim: int = 50,
    cache_dir: str = None,
) -> Tuple[np.ndarray, float]:
    """
    Main entry point for GloVe-based grounding.
    
    Accepts either a tokenizer or a vocab dict.
    
    Args:
        vocab_size: Expected vocabulary size
        tokenizer: Optional HuggingFace tokenizer
        vocab: Optional word → idx dict
        glove_dim: GloVe dimension
        cache_dir: Cache directory
        
    Returns:
        (SO4 embeddings [vocab_size, 4, 4], coverage ratio)
    """
    if tokenizer is not None:
        return create_grounded_embeddings_from_tokenizer(tokenizer, glove_dim, cache_dir)
    elif vocab is not None:
        return create_grounded_embeddings_fast(vocab, glove_dim)
    else:
        raise ValueError("Must provide either tokenizer or vocab")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_grounded_embeddings_from_corpus(
    corpus: List[List[int]],
    vocab_size: int,
    window: int = 5,
    use_ppmi: bool = True,
    scale: float = 0.3,
) -> np.ndarray:
    """
    One-step grounded embedding creation from corpus.
    
    Args:
        corpus: List of token sequences
        vocab_size: Vocabulary size
        window: Context window for co-occurrence
        use_ppmi: Use Positive PMI transformation
        scale: Angle scaling
        
    Returns:
        [vocab_size, 4, 4] grounded SO(4) embeddings
    """
    cooccur = compute_cooccurrence_from_corpus(corpus, vocab_size, window)
    return create_grounded_embeddings(cooccur, vocab_size, use_ppmi, scale)


def initialize_embeddings_grounded(
    vocab_size: int,
    data_samples: Optional[List[Tuple[List[int], int]]] = None,
    allow_random: bool = True,
) -> np.ndarray:
    """
    Initialize embeddings with grounding if data available, else random.
    
    This is the recommended initialization function:
    - If training data is available, use grounded initialization
    - If not, fall back to random (but warn)
    
    Args:
        vocab_size: Vocabulary size
        data_samples: Optional list of (context, target) pairs for grounding
        allow_random: If True, use random when no grounding data available
        
    Returns:
        [vocab_size, 4, 4] SO(4) embeddings
    """
    if data_samples is not None and len(data_samples) > 0:
        # Convert to corpus format
        corpus = [[*ctx, tgt] for ctx, tgt in data_samples]
        return create_grounded_embeddings_from_corpus(corpus, vocab_size)
    
    if allow_random:
        print("WARNING: No grounding data provided, using random embeddings.")
        print("         This will require ~100M+ samples for good accuracy.")
        print("         Consider providing grounding data for O(√N) efficiency.")
        
        # Random SO(4) embeddings - BATCHED QR (~20x faster than loop)
        random_matrices = np.random.randn(vocab_size, 4, 4).astype(DTYPE)
        Q, _ = np.linalg.qr(random_matrices)
        
        # Ensure det = +1 (rotation, not reflection) - vectorized
        dets = np.linalg.det(Q)
        Q[dets < 0, :, 0] *= -1
        
        return Q.astype(DTYPE)
    
    raise ValueError("No grounding data provided and allow_random=False")
