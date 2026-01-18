#!/usr/bin/env python3
"""
Test: Holographic Memory WITH Basin Routing

HYPOTHESIS:
    Our previous tests failed because we stored all patterns in ONE matrix.
    The architecture works via ROUTING:
    1. Compute basin key for context
    2. Route to appropriate satellite
    3. Each satellite has few patterns → interference manageable
    
    This test validates retrieval WITH routing.
"""

import numpy as np
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ, DTYPE
from holographic_prod.core.algebra import (
    build_clifford_basis,
    grace_basin_key_direct,
    frobenius_cosine,
)
from holographic_prod.core.grounded_embeddings import so4_generators


def create_so4_embeddings(vocab_size: int, seed: int = 42) -> np.ndarray:
    """Create SO(4) embeddings."""
    np.random.seed(seed)
    embeddings = np.zeros((vocab_size, 4, 4), dtype=DTYPE)
    generators = so4_generators()
    
    from scipy.linalg import expm
    for i in range(vocab_size):
        coeffs = np.random.randn(6) * 0.5
        A = sum(c * g for c, g in zip(coeffs, generators))
        embeddings[i] = expm(A).astype(DTYPE)
    
    return embeddings


def embed_sequence(tokens: list, embeddings: np.ndarray) -> np.ndarray:
    """Compose tokens."""
    if not tokens:
        return np.eye(4, dtype=DTYPE)
    
    result = embeddings[tokens[0]].copy()
    for t in tokens[1:]:
        result = result @ embeddings[t]
    return result


class SimpleRoutedMemory:
    """Simple implementation of routed holographic memory."""
    
    def __init__(self, basis, resolution=0.1):
        self.basis = basis
        self.resolution = resolution
        self.satellites = defaultdict(lambda: np.zeros((4, 4), dtype=DTYPE))
        self.satellite_counts = defaultdict(int)
    
    def get_basin_key(self, ctx_mat):
        """Compute basin key using raw coefficients (n_iters=0)."""
        return grace_basin_key_direct(ctx_mat, self.basis, n_iters=0, 
                                       resolution=self.resolution)
    
    def store(self, ctx_mat, tgt_mat):
        """Store pattern in appropriate satellite."""
        key = self.get_basin_key(ctx_mat)
        self.satellites[key] += PHI_INV * (ctx_mat @ tgt_mat)
        self.satellite_counts[key] += 1
    
    def retrieve(self, ctx_mat, embeddings):
        """Retrieve from appropriate satellite."""
        key = self.get_basin_key(ctx_mat)
        memory = self.satellites.get(key)
        
        if memory is None or np.allclose(memory, 0):
            return None, 0.0
        
        retrieved = ctx_mat.T @ memory
        
        # Find best matching target
        best_score = -float('inf')
        best_idx = -1
        for i in range(len(embeddings)):
            score = frobenius_cosine(retrieved, embeddings[i])
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx, best_score


def test_routed_retrieval():
    """Test retrieval with basin routing."""
    print("\n" + "="*70)
    print("TEST 1: Retrieval WITH Basin Routing")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    n_patterns = 100
    context_length = 8
    
    memory = SimpleRoutedMemory(basis, resolution=0.1)
    
    stored_contexts = []
    stored_targets = []
    stored_raw = []
    
    print(f"\nStoring {n_patterns} patterns with context length {context_length}...")
    
    for i in range(n_patterns):
        context = list(np.random.randint(0, 1000, size=context_length))
        target = np.random.randint(0, 1000)
        
        stored_contexts.append(context)
        stored_targets.append(target)
        
        raw = embed_sequence(context, embeddings)
        stored_raw.append(raw)
        
        memory.store(raw, embeddings[target])
    
    # Statistics
    n_satellites = len(memory.satellites)
    counts = list(memory.satellite_counts.values())
    print(f"\nRouting statistics:")
    print(f"  Unique satellites used: {n_satellites}")
    print(f"  Patterns per satellite: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}")
    
    # Test exact retrieval
    print(f"\n--- Exact Context Retrieval ---")
    
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    not_found = 0
    n_test = min(50, n_patterns)
    
    for i in range(n_test):
        raw = stored_raw[i]
        true_target = stored_targets[i]
        
        pred_idx, score = memory.retrieve(raw, embeddings)
        
        if pred_idx is None:
            not_found += 1
            continue
        
        # Get rankings
        key = memory.get_basin_key(raw)
        mem = memory.satellites[key]
        retrieved = raw.T @ mem
        
        scores = [(t, frobenius_cosine(retrieved, embeddings[t])) for t in range(1000)]
        scores.sort(key=lambda x: -x[1])
        
        ranking = [t for t, s in scores]
        true_rank = ranking.index(true_target) if true_target in ranking else 1000
        
        if true_rank == 0:
            correct_top1 += 1
        if true_rank < 5:
            correct_top5 += 1
        if true_rank < 10:
            correct_top10 += 1
    
    valid_tests = n_test - not_found
    print(f"  Not found in satellite: {not_found}/{n_test}")
    if valid_tests > 0:
        print(f"  Top-1:  {correct_top1}/{valid_tests} = {100*correct_top1/valid_tests:.0f}%")
        print(f"  Top-5:  {correct_top5}/{valid_tests} = {100*correct_top5/valid_tests:.0f}%")
        print(f"  Top-10: {correct_top10}/{valid_tests} = {100*correct_top10/valid_tests:.0f}%")
    
    # Distribution of patterns per satellite
    print(f"\n--- Satellite Distribution ---")
    count_dist = defaultdict(int)
    for c in counts:
        if c == 1:
            count_dist["1"] += 1
        elif c <= 5:
            count_dist["2-5"] += 1
        elif c <= 10:
            count_dist["6-10"] += 1
        elif c <= 16:
            count_dist["11-16"] += 1
        else:
            count_dist[">16"] += 1
    
    print(f"  1 pattern:   {count_dist.get('1', 0)} satellites")
    print(f"  2-5:         {count_dist.get('2-5', 0)} satellites")
    print(f"  6-10:        {count_dist.get('6-10', 0)} satellites")
    print(f"  11-16:       {count_dist.get('11-16', 0)} satellites")
    print(f"  >16 (limit): {count_dist.get('>16', 0)} satellites")


def test_resolution_sensitivity():
    """Test how resolution affects routing and retrieval."""
    print("\n" + "="*70)
    print("TEST 2: Resolution Sensitivity")
    print("="*70)
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    n_patterns = 100
    context_length = 8
    
    resolutions = [1.0, 0.5, 0.1, 0.05, 0.01]
    
    print(f"\n{'Resolution':<12} {'Satellites':<12} {'Max/satellite':<15} {'Top-10 acc':<12}")
    print("-" * 51)
    
    for resolution in resolutions:
        memory = SimpleRoutedMemory(basis, resolution=resolution)
        
        stored_raw = []
        stored_targets = []
        
        for i in range(n_patterns):
            context = list(np.random.randint(0, 1000, size=context_length))
            target = np.random.randint(0, 1000)
            
            raw = embed_sequence(context, embeddings)
            stored_raw.append(raw)
            stored_targets.append(target)
            
            memory.store(raw, embeddings[target])
        
        n_satellites = len(memory.satellites)
        max_count = max(memory.satellite_counts.values())
        
        # Test retrieval
        correct = 0
        n_test = min(20, n_patterns)
        
        for i in range(n_test):
            raw = stored_raw[i]
            true_target = stored_targets[i]
            
            key = memory.get_basin_key(raw)
            mem = memory.satellites.get(key)
            
            if mem is None:
                continue
            
            retrieved = raw.T @ mem
            scores = [(t, frobenius_cosine(retrieved, embeddings[t])) for t in range(1000)]
            scores.sort(key=lambda x: -x[1])
            ranking = [t for t, s in scores]
            
            if true_target in ranking[:10]:
                correct += 1
        
        acc = 100 * correct / n_test
        print(f"{resolution:<12} {n_satellites:<12} {max_count:<15} {acc:<12.0f}%")


def test_few_patterns_per_satellite():
    """Test retrieval when we ENSURE few patterns per satellite."""
    print("\n" + "="*70)
    print("TEST 3: Controlled Satellite Occupancy")
    print("="*70)
    print("""
Theory says holographic superposition works for ~8-16 patterns.
Let's test with GUARANTEED few patterns per satellite.
""")
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(1000)
    
    # Create patterns with UNIQUE first tokens to ensure different basins
    context_length = 8
    n_patterns = 16  # Theory limit
    
    stored_raw = []
    stored_targets = []
    
    memory = SimpleRoutedMemory(basis, resolution=0.01)  # Fine resolution
    
    # Ensure each pattern goes to different satellite by using unique prefixes
    for i in range(n_patterns):
        # Unique first token
        context = [i * 50] + list(np.random.randint(0, 1000, size=context_length-1))
        target = np.random.randint(0, 1000)
        
        raw = embed_sequence(context, embeddings)
        stored_raw.append(raw)
        stored_targets.append(target)
        
        memory.store(raw, embeddings[target])
    
    print(f"Satellites used: {len(memory.satellites)}")
    print(f"Patterns per satellite: {list(memory.satellite_counts.values())}")
    
    # Test retrieval
    print(f"\n--- Retrieval with ~1 pattern/satellite ---")
    
    correct_top1 = 0
    correct_top10 = 0
    
    for i in range(n_patterns):
        raw = stored_raw[i]
        true_target = stored_targets[i]
        
        key = memory.get_basin_key(raw)
        mem = memory.satellites[key]
        
        retrieved = raw.T @ mem
        
        scores = [(t, frobenius_cosine(retrieved, embeddings[t])) for t in range(1000)]
        scores.sort(key=lambda x: -x[1])
        ranking = [t for t, s in scores]
        
        true_rank = ranking.index(true_target) if true_target in ranking else 1000
        
        if true_rank == 0:
            correct_top1 += 1
        if true_rank < 10:
            correct_top10 += 1
        
        if i < 5:
            print(f"  Pattern {i}: target={true_target}, rank={true_rank}, "
                  f"top3={ranking[:3]}, n_in_sat={memory.satellite_counts[key]}")
    
    print(f"\nTop-1:  {correct_top1}/{n_patterns} = {100*correct_top1/n_patterns:.0f}%")
    print(f"Top-10: {correct_top10}/{n_patterns} = {100*correct_top10/n_patterns:.0f}%")


def test_single_pattern_retrieval():
    """Test the simplest case: one pattern per satellite."""
    print("\n" + "="*70)
    print("TEST 4: Single Pattern Retrieval (Sanity Check)")
    print("="*70)
    print("""
If SO(4) retrieval works, ctx.T @ (ctx @ tgt) = tgt should be exact.
This is the FUNDAMENTAL claim of the architecture.
""")
    
    basis = build_clifford_basis()
    embeddings = create_so4_embeddings(100)  # Small vocab
    
    # Test single pattern retrieval
    context_length = 8
    n_tests = 20
    
    exact_matches = 0
    
    for i in range(n_tests):
        context = list(np.random.randint(0, 100, size=context_length))
        target = np.random.randint(0, 100)
        
        ctx = embed_sequence(context, embeddings)
        tgt = embeddings[target]
        
        # Single pattern memory
        memory = PHI_INV * (ctx @ tgt)
        
        # Retrieve
        retrieved = ctx.T @ memory
        
        # Should be exactly φ⁻¹ * tgt (scaled)
        expected = PHI_INV * tgt
        
        # Check if retrieved matches target
        match_score = frobenius_cosine(retrieved, tgt)
        expected_score = frobenius_cosine(expected, tgt)
        
        # Find ranking
        scores = [(t, frobenius_cosine(retrieved, embeddings[t])) for t in range(100)]
        scores.sort(key=lambda x: -x[1])
        ranking = [t for t, s in scores]
        true_rank = ranking.index(target)
        
        if true_rank == 0:
            exact_matches += 1
        
        if i < 5:
            print(f"  Test {i}: target={target}, rank={true_rank}, "
                  f"match_score={match_score:.4f}, expected_score={expected_score:.4f}")
    
    print(f"\nExact matches (rank=0): {exact_matches}/{n_tests} = {100*exact_matches/n_tests:.0f}%")
    
    if exact_matches == n_tests:
        print("✓ Single pattern retrieval works perfectly!")
    else:
        print("✗ Single pattern retrieval fails - this is a fundamental issue!")


def main():
    print("\n" + "="*70)
    print("   HOLOGRAPHIC MEMORY WITH ROUTING")
    print("="*70)
    
    test_single_pattern_retrieval()
    test_few_patterns_per_satellite()
    test_routed_retrieval()
    test_resolution_sensitivity()
    
    print("\n" + "="*70)
    print("   SUMMARY")
    print("="*70)


if __name__ == "__main__":
    main()
