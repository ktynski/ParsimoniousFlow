"""Profile each operation in retrieve_parallel to find remaining bottlenecks.

v5.31.0: Updated to profile theory-true full vocabulary coherence scoring.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.holographic_memory_unified import HolographicMemory


def profile_parallel_retrieval():
    """Profile each step of parallel retrieval."""
    print("=" * 70)
    print("  PROFILING retrieve_parallel() COMPONENTS (v5.31.0)")
    print("=" * 70)
    
    memory = HolographicMemory(vocab_size=1000, max_levels=2, seed=42)
    
    # Learn patterns
    for i in range(200):
        context = [np.random.randint(0, 1000) for _ in range(8)]
        target = np.random.randint(0, 1000)
        memory.learn(context, target)
    
    # Test context (ensure it's in cache)
    test_context = [np.random.randint(0, 1000) for _ in range(8)]
    memory.learn(test_context, 42)
    
    ctx_tuple = tuple(test_context)
    n = 100
    
    # Profile each component
    print(f"\n  n_iterations = {n}")
    
    # 1. Tuple conversion
    t0 = time.perf_counter()
    for _ in range(n):
        _ = tuple(test_context)
    t_tuple = (time.perf_counter() - t0) / n * 1e6
    print(f"\n  tuple(context):              {t_tuple:.2f} μs")
    
    # 2. Episodic lookup
    t0 = time.perf_counter()
    for _ in range(n):
        _ = memory._episodic_cache.get(ctx_tuple)
    t_episodic = (time.perf_counter() - t0) / n * 1e6
    print(f"  _episodic_cache.get():       {t_episodic:.2f} μs")
    
    # 3. _embed_sequence
    t0 = time.perf_counter()
    for _ in range(n):
        ctx_mat = memory.tower._embed_sequence(test_context)
    t_embed = (time.perf_counter() - t0) / n * 1e6
    print(f"  _embed_sequence():           {t_embed:.2f} μs")
    
    # 4. route_to_satellite
    t0 = time.perf_counter()
    for _ in range(n):
        sat_idx = memory.tower.route_to_satellite(test_context)
    t_route = (time.perf_counter() - t0) / n * 1e6
    print(f"  route_to_satellite():        {t_route:.2f} μs")
    
    # 5. Memory access
    sat_idx = memory.tower.route_to_satellite(test_context)
    ctx_mat = memory.tower._embed_sequence(test_context)
    t0 = time.perf_counter()
    for _ in range(n):
        sat_memory = memory.tower._all_memories[sat_idx]
    t_mem_access = (time.perf_counter() - t0) / n * 1e6
    print(f"  _all_memories[idx]:          {t_mem_access:.2f} μs")
    
    # 6. Transpose + matmul (unbind)
    sat_memory = memory.tower._all_memories[sat_idx]
    t0 = time.perf_counter()
    for _ in range(n):
        ctx_inv = ctx_mat.T
        retrieved = ctx_inv @ sat_memory
    t_unbind = (time.perf_counter() - t0) / n * 1e6
    print(f"  unbind (transpose+matmul):   {t_unbind:.2f} μs")
    
    # 7. THEORY-TRUE: Full vocabulary coherence scoring
    xp = np
    embeddings = memory.tower.embeddings
    basis = memory.tower.basis
    vocab_size = len(embeddings)
    retrieved = ctx_mat.T @ sat_memory
    
    # Precompute for coherence scoring
    norm_sq = xp.sum(basis * basis, axis=(1, 2))
    embed_T = xp.swapaxes(embeddings, -2, -1)
    
    t0 = time.perf_counter()
    for _ in range(n):
        # Compute compositions: retrieved @ embed[t].T for all t
        compositions = xp.einsum('ij,vjk->vik', retrieved, embed_T)
        # Decompose into Clifford coefficients
        coeffs_all = xp.einsum('cij,vij->vc', basis, compositions) / norm_sq
        # Compute coherence
        energies = xp.sum(coeffs_all ** 2, axis=1)
        witness_energies = coeffs_all[:, 0]**2 + coeffs_all[:, 15]**2
        coherences = witness_energies / xp.maximum(energies, 1e-10)
        best_token = int(xp.argmax(coherences))
    t_coherence = (time.perf_counter() - t0) / n * 1e6
    print(f"  full vocab coherence score:  {t_coherence:.2f} μs (vocab={vocab_size})")
    
    # Total holographic path
    total_holo = t_embed + t_route + t_mem_access + t_unbind + t_coherence
    
    print(f"\n  ─────────────────────────────────────")
    print(f"  EPISODIC PATH (O(1)):        {t_tuple + t_episodic:.2f} μs")
    print(f"  HOLOGRAPHIC PATH:            {total_holo:.2f} μs")
    print(f"    - embed:                   {t_embed:.2f} μs")
    print(f"    - route:                   {t_route:.2f} μs")
    print(f"    - memory access:           {t_mem_access:.2f} μs")
    print(f"    - unbind:                  {t_unbind:.2f} μs")
    print(f"    - coherence scoring:       {t_coherence:.2f} μs")
    
    print(f"\n  THEORY-TRUE INSIGHT:")
    print(f"    Full vocabulary coherence scoring is O(vocab_size)")
    print(f"    For vocab_size={vocab_size}: {t_coherence:.2f} μs")
    print(f"    Episodic path is O(1): {t_episodic:.2f} μs")
    print(f"    Holographic / Episodic = {total_holo / max(t_episodic, 0.01):.0f}x")
    print(f"\n  SOLUTION:")
    print(f"    Episodic cache provides O(1) exact match")
    print(f"    Holographic path provides generalization")
    print(f"    Both paths run in PARALLEL per CLS theory")


if __name__ == "__main__":
    profile_parallel_retrieval()
