"""
Test: Semantic Index Retrieval Accuracy
========================================

GOAL: Verify that semantic 2D index can achieve high GENERALIZATION accuracy
by combining:
1. Coarse 2D bucketing (converged witness)
2. Within-bucket vorticity similarity matching

THEORY:
    - Coarse buckets group semantically similar contexts
    - Vorticity (bivector direction) discriminates within bucket
    - Combined: generalization + discrimination
"""

import numpy as np
from holographic_v4.algebra import (
    build_clifford_basis, 
    grace_operator,
    geometric_product_batch,
    initialize_embeddings_identity,
)
from holographic_v4.quotient import extract_witness, vorticity_similarity
from holographic_v4.constants import PHI_INV, PHI_INV_SQ


class SemanticWitnessIndex:
    """
    Semantic index using CONVERGED witness for generalization.
    
    KEY DIFFERENCES from VorticityWitnessIndex:
    1. Uses 2D keys (witness only), not 8D
    2. Applies N Grace iterations before extracting witness
    3. Coarser resolution for larger basins
    4. Uses vorticity_similarity for within-bucket matching
    """
    
    def __init__(self, basis, xp=np, n_grace=5, resolution=PHI_INV):
        self.basis = basis
        self.xp = xp
        self.n_grace = n_grace
        self.resolution = resolution
        self.buckets = {}  # (s_idx, p_idx) -> [(context, target, target_idx)]
        self.n_items = 0
    
    @classmethod
    def create(cls, basis, xp=np, n_grace=5, resolution=PHI_INV):
        return cls(basis, xp, n_grace, resolution)
    
    def _converged_witness(self, M):
        """Apply N Grace iterations and extract witness."""
        ctx = M.copy()
        for _ in range(self.n_grace):
            ctx = grace_operator(ctx, self.basis, self.xp)
        return extract_witness(ctx, self.basis, self.xp)
    
    def _semantic_key(self, M):
        """Compute 2D semantic key from converged witness."""
        s, p = self._converged_witness(M)
        s_idx = int(self.xp.floor(s / self.resolution))
        p_idx = int(self.xp.floor(p / self.resolution))
        return (s_idx, p_idx)
    
    def store(self, context, target, target_idx):
        """Store pattern with semantic key."""
        key = self._semantic_key(context)
        if key not in self.buckets:
            self.buckets[key] = []
        self.buckets[key].append((context.copy(), target.copy(), target_idx))
        self.n_items += 1
    
    def retrieve(self, context):
        """
        Retrieve using semantic key + vorticity similarity.
        
        Returns: (target, target_idx, confidence)
        """
        key = self._semantic_key(context)
        
        if key not in self.buckets or len(self.buckets[key]) == 0:
            return None, None, 0.0
        
        bucket = self.buckets[key]
        
        if len(bucket) == 1:
            _, target, target_idx = bucket[0]
            return target, target_idx, 1.0
        
        # Multiple items: use vorticity similarity
        best_sim = -float('inf')
        best_target = None
        best_idx = None
        
        for ctx, tgt, idx in bucket:
            sim = vorticity_similarity(context, ctx, self.basis, self.xp)
            if sim > best_sim:
                best_sim = sim
                best_target = tgt
                best_idx = idx
        
        confidence = max(0.0, best_sim)
        return best_target, best_idx, confidence


def test_semantic_index_generalization():
    """Test generalization accuracy with semantic index."""
    print("\n" + "="*70)
    print("TEST: Semantic Index Generalization Accuracy")
    print("="*70)
    
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(50000, xp=np)
    rng = np.random.default_rng(42)
    
    # Test different configurations
    configs = [
        {"n_grace": 1, "resolution": PHI_INV},
        {"n_grace": 3, "resolution": PHI_INV},
        {"n_grace": 5, "resolution": PHI_INV},
        {"n_grace": 5, "resolution": PHI_INV * 2},  # Coarser
        {"n_grace": 10, "resolution": PHI_INV * 2},
    ]
    
    n_train = 200
    n_test = 100
    ctx_len = 512
    
    # Generate training data
    train_data = []
    for i in range(n_train):
        seq = rng.integers(0, 50000, size=ctx_len)
        mats = embeddings[seq]
        ctx = geometric_product_batch(mats, np)
        ctx = grace_operator(ctx, basis, np)
        target = embeddings[rng.integers(0, 50000)]
        train_data.append((seq, ctx, target, i))
    
    print(f"\n  Training patterns: {n_train}")
    print(f"  Test queries: {n_test}")
    print(f"  Context length: {ctx_len}")
    print(f"  Perturbation: Change token at position 0")
    
    print(f"\n  {'Config':<30} {'Buckets':<10} {'Exact':<10} {'Perturbed':<10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    
    for config in configs:
        # Create semantic index
        index = SemanticWitnessIndex.create(
            basis, xp=np, 
            n_grace=config["n_grace"],
            resolution=config["resolution"]
        )
        
        # Store all training patterns
        for seq, ctx, target, idx in train_data:
            index.store(ctx, target, idx)
        
        # Test exact retrieval
        exact_correct = 0
        for seq, ctx, target, expected_idx in train_data[:n_test]:
            _, retrieved_idx, conf = index.retrieve(ctx)
            if retrieved_idx == expected_idx:
                exact_correct += 1
        
        # Test perturbed retrieval
        perturbed_correct = 0
        for seq, ctx, target, expected_idx in train_data[:n_test]:
            # Create perturbed context
            seq_p = seq.copy()
            seq_p[0] = (seq_p[0] + 1) % 50000
            mats_p = embeddings[seq_p]
            ctx_p = geometric_product_batch(mats_p, np)
            ctx_p = grace_operator(ctx_p, basis, np)
            
            _, retrieved_idx, conf = index.retrieve(ctx_p)
            if retrieved_idx == expected_idx:
                perturbed_correct += 1
        
        config_str = f"n={config['n_grace']}, res={config['resolution']:.3f}"
        exact_acc = exact_correct / n_test
        pert_acc = perturbed_correct / n_test
        
        print(f"  {config_str:<30} {len(index.buckets):<10} {exact_acc:<10.1%} {pert_acc:<10.1%}")
    
    return True


def test_position_sensitivity():
    """Test if perturbation position affects generalization."""
    print("\n" + "="*70)
    print("TEST: Perturbation Position Sensitivity")
    print("="*70)
    
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(50000, xp=np)
    rng = np.random.default_rng(42)
    
    n_train = 200
    n_test = 50
    ctx_len = 512
    
    # Create semantic index
    index = SemanticWitnessIndex.create(basis, xp=np, n_grace=5, resolution=PHI_INV)
    
    # Generate and store training data
    train_data = []
    for i in range(n_train):
        seq = rng.integers(0, 50000, size=ctx_len)
        mats = embeddings[seq]
        ctx = geometric_product_batch(mats, np)
        ctx = grace_operator(ctx, basis, np)
        target = embeddings[rng.integers(0, 50000)]
        index.store(ctx, target, i)
        train_data.append((seq, ctx, target, i))
    
    # Test different perturbation positions
    positions = [0, ctx_len//4, ctx_len//2, 3*ctx_len//4, ctx_len-1]
    
    print(f"\n  Perturbation Position -> Accuracy")
    print(f"  {'-'*40}")
    
    results = {}
    for pos in positions:
        correct = 0
        for seq, ctx, target, expected_idx in train_data[:n_test]:
            # Perturb at specific position
            seq_p = seq.copy()
            seq_p[pos] = (seq_p[pos] + 1) % 50000
            mats_p = embeddings[seq_p]
            ctx_p = geometric_product_batch(mats_p, np)
            ctx_p = grace_operator(ctx_p, basis, np)
            
            _, retrieved_idx, conf = index.retrieve(ctx_p)
            if retrieved_idx == expected_idx:
                correct += 1
        
        acc = correct / n_test
        results[pos] = acc
        print(f"  Position {pos:>4} ({pos/ctx_len:.1%} through): {acc:.1%}")
    
    # Analysis
    print(f"\n  ANALYSIS:")
    print(f"  - Position 0 (first token): {results[0]:.1%}")
    print(f"  - Position {ctx_len-1} (last token): {results[ctx_len-1]:.1%}")
    
    if results[ctx_len-1] > results[0]:
        print(f"  ✓ Later positions more robust (expected: earlier tokens affect more)")
    else:
        print(f"  ~ Position doesn't matter much for semantic index")
    
    return True


def test_multiple_token_perturbation():
    """Test generalization with multiple token changes."""
    print("\n" + "="*70)
    print("TEST: Multiple Token Perturbation")
    print("="*70)
    
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(50000, xp=np)
    rng = np.random.default_rng(42)
    
    n_train = 200
    n_test = 50
    ctx_len = 512
    
    # Create semantic index
    index = SemanticWitnessIndex.create(basis, xp=np, n_grace=5, resolution=PHI_INV)
    
    # Generate and store training data
    train_data = []
    for i in range(n_train):
        seq = rng.integers(0, 50000, size=ctx_len)
        mats = embeddings[seq]
        ctx = geometric_product_batch(mats, np)
        ctx = grace_operator(ctx, basis, np)
        target = embeddings[rng.integers(0, 50000)]
        index.store(ctx, target, i)
        train_data.append((seq, ctx, target, i))
    
    # Test different numbers of perturbed tokens
    n_perturb_list = [1, 2, 5, 10, 20, 50]
    
    print(f"\n  # Perturbed Tokens -> Accuracy")
    print(f"  {'-'*40}")
    
    for n_perturb in n_perturb_list:
        correct = 0
        for seq, ctx, target, expected_idx in train_data[:n_test]:
            # Perturb multiple random positions
            seq_p = seq.copy()
            perturb_positions = rng.choice(ctx_len, size=min(n_perturb, ctx_len), replace=False)
            for pos in perturb_positions:
                seq_p[pos] = (seq_p[pos] + 1) % 50000
            
            mats_p = embeddings[seq_p]
            ctx_p = geometric_product_batch(mats_p, np)
            ctx_p = grace_operator(ctx_p, basis, np)
            
            _, retrieved_idx, conf = index.retrieve(ctx_p)
            if retrieved_idx == expected_idx:
                correct += 1
        
        acc = correct / n_test
        pct = n_perturb / ctx_len * 100
        print(f"  {n_perturb:>3} tokens ({pct:>5.1f}% changed): {acc:.1%}")
    
    return True


if __name__ == "__main__":
    print("="*70)
    print("SEMANTIC INDEX ACCURACY TESTS")
    print("="*70)
    print("\nGOAL: Find configuration that maximizes generalization accuracy")
    print("while maintaining reasonable exact-match accuracy.")
    
    import time
    start = time.time()
    
    test_semantic_index_generalization()
    test_position_sensitivity()
    test_multiple_token_perturbation()
    
    elapsed = time.time() - start
    
    print("\n" + "="*70)
    print(f"Total time: {elapsed:.1f}s")
    print("="*70)
