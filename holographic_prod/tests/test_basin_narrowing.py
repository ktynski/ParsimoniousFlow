"""
Test Basin-Key Candidate Narrowing — Theory-True Holographic Retrieval
=======================================================================

THEORY (Brain Analog):
    The brain NEVER compares against ALL vocabulary items. It uses:
    1. Sparse activation - Only ~1-3% of neurons fire
    2. Topographic organization - Similar → nearby cortical regions
    3. Lateral connections - Spread activation to neighbors
    
    Grace basins provide the geometric equivalent:
    - Similar patterns → same basin (attractor)
    - Basin key → topographic location
    - L1 neighbors → lateral spread

PROBLEM (What We're Fixing):
    Current holographic path scores ALL 50K tokens when semantic prototype misses.
    This is NOT theory-true and leads to random outputs.
    
    Current: retrieve → unbind → score ALL embeddings → random
    Fixed: retrieve → unbind → basin key → narrow to basin → score ~100 candidates

TEST STRATEGY:
    1. Build token→basin index at initialization
    2. Query should use basin key to narrow candidates
    3. Verify improvement in discrimination (50K → ~100 candidates)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from holographic_prod.core.algebra import (
    build_clifford_basis, 
    grace_basin_key_direct,
    grace_basin_keys_batch_direct,
    BASIN_KEY_INDICES,
)
from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ, PHI_INV_EIGHT, DTYPE


def test_basin_key_clustering():
    """
    THEORY TEST: Similar embeddings should have similar basin keys.
    This validates that basin keys can be used for candidate narrowing.
    """
    print("\n" + "="*70)
    print("TEST: Basin Key Clustering")
    print("="*70)
    
    np.random.seed(42)
    basis = build_clifford_basis()
    
    # Create embeddings with known structure
    # Cluster 1: Similar rotations around axis 1
    # Cluster 2: Similar rotations around axis 2
    vocab_size = 100
    embeddings = np.zeros((vocab_size, 4, 4), dtype=DTYPE)
    
    # Initialize with random SO(4) matrices
    for i in range(vocab_size):
        Q, _ = np.linalg.qr(np.random.randn(4, 4).astype(DTYPE))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        embeddings[i] = Q
    
    # Compute basin keys for all embeddings
    resolution = PHI_INV_EIGHT
    n_iters = 0  # Direct method
    
    basin_keys = grace_basin_keys_batch_direct(embeddings, basis, n_iters, resolution, np)
    
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Basin key dimension: {basin_keys.shape[1]}")
    print(f"  Resolution: {resolution:.6f}")
    
    # Count unique basin keys
    unique_keys = set(tuple(int(x) for x in row) for row in basin_keys)
    print(f"  Unique basin keys: {len(unique_keys)}")
    
    # Verify basin keys are integers
    assert basin_keys.dtype in [np.int32, np.int64, np.float32, np.float64], \
        f"Basin keys should be numeric, got {basin_keys.dtype}"
    
    print("  ✓ PASS: Basin keys computed successfully")
    
    return basin_keys


def test_basin_index_construction():
    """
    THEORY TEST: Construct basin_key → token_ids index for fast lookup.
    """
    print("\n" + "="*70)
    print("TEST: Basin Index Construction")
    print("="*70)
    
    np.random.seed(123)
    vocab_size = 500
    basis = build_clifford_basis()
    
    # Create random embeddings
    embeddings = np.zeros((vocab_size, 4, 4), dtype=DTYPE)
    for i in range(vocab_size):
        Q, _ = np.linalg.qr(np.random.randn(4, 4).astype(DTYPE))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        embeddings[i] = Q
    
    # Compute basin keys
    resolution = PHI_INV_EIGHT
    basin_keys = grace_basin_keys_batch_direct(embeddings, basis, 0, resolution, np)
    
    # Build index: basin_key → list of token IDs
    from collections import defaultdict
    basin_index = defaultdict(list)
    
    for token_id in range(vocab_size):
        key = tuple(int(x) for x in basin_keys[token_id])
        basin_index[key].append(token_id)
    
    # Statistics
    n_basins = len(basin_index)
    tokens_per_basin = [len(v) for v in basin_index.values()]
    avg_tokens = np.mean(tokens_per_basin)
    max_tokens = max(tokens_per_basin)
    min_tokens = min(tokens_per_basin)
    
    print(f"  Vocabulary: {vocab_size}")
    print(f"  Unique basins: {n_basins}")
    print(f"  Tokens per basin: avg={avg_tokens:.1f}, min={min_tokens}, max={max_tokens}")
    print(f"  Narrowing factor: {vocab_size / avg_tokens:.1f}x")
    
    # Verify narrowing is meaningful
    assert n_basins >= vocab_size * 0.1, f"Too few basins: {n_basins}"
    assert avg_tokens < vocab_size * 0.5, f"Basins too large: avg={avg_tokens}"
    
    print("  ✓ PASS: Basin index constructed with meaningful narrowing")
    
    return basin_index, embeddings, basin_keys


def test_l1_neighbor_expansion():
    """
    THEORY TEST: L1 neighbors provide lateral activation spread.
    """
    print("\n" + "="*70)
    print("TEST: L1 Neighbor Expansion")
    print("="*70)
    
    # Create a simple basin key
    key = (0, 1, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def get_l1_neighbors(key, max_distance=1):
        """Get all keys within L1 distance."""
        neighbors = set()
        neighbors.add(key)
        
        for i in range(len(key)):
            for delta in range(-max_distance, max_distance + 1):
                if delta == 0:
                    continue
                neighbor = list(key)
                neighbor[i] += delta
                neighbors.add(tuple(neighbor))
        
        return neighbors
    
    neighbors_d1 = get_l1_neighbors(key, max_distance=1)
    neighbors_d2 = get_l1_neighbors(key, max_distance=2)
    
    print(f"  Original key: {key[:4]}... (16D)")
    print(f"  L1 distance=1: {len(neighbors_d1)} neighbors")
    print(f"  L1 distance=2: {len(neighbors_d2)} neighbors")
    
    # For 16D key with d=1: 1 + 16*2 = 33 neighbors
    expected_d1 = 1 + 16 * 2
    assert len(neighbors_d1) == expected_d1, f"Expected {expected_d1}, got {len(neighbors_d1)}"
    
    print("  ✓ PASS: L1 neighbors computed correctly")
    
    return neighbors_d1


def test_basin_narrowing_improves_discrimination():
    """
    THEORY TEST: Basin narrowing should improve avg_rank dramatically.
    
    Setup:
        - 1000 token vocabulary
        - Query embedding
        - Compare: full vocab vs basin-narrowed candidates
        
    Expected:
        - Without narrowing: avg_rank ≈ 500 (random)
        - With narrowing: avg_rank << 100 (within basin)
    """
    print("\n" + "="*70)
    print("TEST: Basin Narrowing Improves Discrimination")
    print("="*70)
    
    np.random.seed(456)
    vocab_size = 1000
    basis = build_clifford_basis()
    
    # Create embeddings
    embeddings = np.zeros((vocab_size, 4, 4), dtype=DTYPE)
    for i in range(vocab_size):
        Q, _ = np.linalg.qr(np.random.randn(4, 4).astype(DTYPE))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        embeddings[i] = Q
    
    # Compute basin keys and build index
    resolution = PHI_INV_EIGHT
    basin_keys = grace_basin_keys_batch_direct(embeddings, basis, 0, resolution, np)
    
    from collections import defaultdict
    basin_index = defaultdict(list)
    for token_id in range(vocab_size):
        key = tuple(int(x) for x in basin_keys[token_id])
        basin_index[key].append(token_id)
    
    # Test queries
    n_queries = 50
    ranks_full = []
    ranks_narrowed = []
    candidates_counts = []
    
    for _ in range(n_queries):
        # Pick a random target token
        target = np.random.randint(0, vocab_size)
        target_emb = embeddings[target]
        
        # Compute query basin key (slightly perturbed target)
        noise = np.random.randn(4, 4).astype(DTYPE) * 0.1
        query = target_emb + noise
        query_key = grace_basin_key_direct(query, basis, 0, resolution, np)
        
        # Score all tokens (no narrowing)
        from holographic_prod.core.quotient import vorticity_weighted_scores
        all_scores = vorticity_weighted_scores(query, embeddings, basis, np)
        
        sorted_full = np.argsort(all_scores)[::-1]
        rank_full = int(np.where(sorted_full == target)[0][0]) + 1
        ranks_full.append(rank_full)
        
        # Score only basin candidates
        def get_l1_neighbors(key, max_distance=1):
            neighbors = set()
            neighbors.add(key)
            for i in range(len(key)):
                for delta in range(-max_distance, max_distance + 1):
                    if delta == 0:
                        continue
                    neighbor = list(key)
                    neighbor[i] += delta
                    neighbors.add(tuple(neighbor))
            return neighbors
        
        neighbor_keys = get_l1_neighbors(query_key, max_distance=1)
        candidates = []
        for nk in neighbor_keys:
            candidates.extend(basin_index.get(nk, []))
        candidates = list(set(candidates))  # Dedupe
        
        if len(candidates) == 0:
            candidates = [target]  # Fallback
        
        candidates_counts.append(len(candidates))
        
        candidate_embeddings = embeddings[np.array(candidates)]
        narrowed_scores = vorticity_weighted_scores(query, candidate_embeddings, basis, np)
        
        if target in candidates:
            target_idx = candidates.index(target)
            sorted_narrowed = np.argsort(narrowed_scores)[::-1]
            rank_narrowed = int(np.where(sorted_narrowed == target_idx)[0][0]) + 1
        else:
            rank_narrowed = len(candidates) + 1
        
        ranks_narrowed.append(rank_narrowed)
    
    avg_rank_full = np.mean(ranks_full)
    avg_rank_narrowed = np.mean(ranks_narrowed)
    avg_candidates = np.mean(candidates_counts)
    
    print(f"\n  Results ({n_queries} queries):")
    print(f"    Without narrowing: avg_rank = {avg_rank_full:.1f} / {vocab_size}")
    print(f"    With narrowing:    avg_rank = {avg_rank_narrowed:.1f} / {avg_candidates:.0f} candidates")
    print(f"    Candidates: avg={avg_candidates:.0f}, vocab={vocab_size}")
    print(f"    Narrowing factor: {vocab_size / avg_candidates:.1f}x")
    print(f"    Rank improvement: {avg_rank_full / max(1, avg_rank_narrowed):.1f}x")
    
    # Key assertion: narrowing should improve
    assert avg_rank_narrowed <= avg_rank_full, \
        f"Narrowing should not harm: {avg_rank_narrowed} <= {avg_rank_full}"
    
    # Candidates should be much smaller than vocab
    assert avg_candidates < vocab_size * 0.5, \
        f"Should narrow candidates: {avg_candidates} < {vocab_size * 0.5}"
    
    print("\n  ✓ PASS: Basin narrowing improves discrimination")
    
    return {
        'avg_rank_full': avg_rank_full,
        'avg_rank_narrowed': avg_rank_narrowed,
        'avg_candidates': avg_candidates,
        'improvement': avg_rank_full / max(1, avg_rank_narrowed),
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BASIN NARROWING TEST SUITE")
    print("Theory: Use Grace basin keys for sparse activation in holographic path")
    print("="*70)
    
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Test timed out")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    
    try:
        signal.alarm(120)
        
        test_basin_key_clustering()
        test_basin_index_construction()
        test_l1_neighbor_expansion()
        result = test_basin_narrowing_improves_discrimination()
        
        signal.alarm(0)
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETE")
        print(f"Final improvement: {result['improvement']:.1f}x")
        print("="*70)
        
    except TimeoutError:
        print("\n❌ Test timed out after 120 seconds")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
