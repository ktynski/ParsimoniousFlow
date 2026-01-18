"""
Test: Iterated Grace for Generalization
========================================

THEORY HYPOTHESIS:
    Multiple Grace iterations converge similar contexts to the same witness.
    A single token change should NOT dramatically change the converged witness.

EXPERIMENT:
    1. Create two contexts: original and perturbed (one token changed)
    2. Apply N Grace iterations to each
    3. Measure witness similarity at each iteration count
    4. Verify convergence → similar contexts should have similar witnesses

EXPECTED RESULT:
    - At N=1: Witnesses may differ significantly
    - At N=5+: Witnesses should converge (high similarity)
"""

import numpy as np
from holographic_v4.algebra import (
    build_clifford_basis, 
    grace_operator,
    geometric_product_batch,
    initialize_embeddings_identity,
)
from holographic_v4.quotient import extract_witness, witness_similarity
from holographic_v4.constants import PHI_INV


def test_iterated_grace_convergence():
    """Test that iterated Grace converges similar contexts to same witness."""
    print("\n" + "="*70)
    print("TEST: Iterated Grace Convergence")
    print("="*70)
    
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(50000, xp=np)
    rng = np.random.default_rng(42)
    
    # Generate test cases
    n_tests = 20
    context_lengths = [64, 256, 512, 1024]
    
    results = {}
    
    for ctx_len in context_lengths:
        print(f"\n  Context length: {ctx_len}")
        print(f"  {'N Grace':<10} {'Witness Sim':<15} {'Status':<20}")
        print(f"  {'-'*10} {'-'*15} {'-'*20}")
        
        witness_sims_by_n = {n: [] for n in [1, 2, 3, 5, 10]}
        
        for _ in range(n_tests):
            # Original context
            seq = rng.integers(0, 50000, size=ctx_len)
            mats = embeddings[seq]
            ctx_original = geometric_product_batch(mats, np)
            
            # Perturbed context (change first token)
            seq_perturbed = seq.copy()
            seq_perturbed[0] = (seq_perturbed[0] + 1) % 50000
            mats_perturbed = embeddings[seq_perturbed]
            ctx_perturbed = geometric_product_batch(mats_perturbed, np)
            
            # Test different Grace iteration counts
            for n_grace in [1, 2, 3, 5, 10]:
                # Apply N Grace iterations
                ctx_o = ctx_original.copy()
                ctx_p = ctx_perturbed.copy()
                
                for _ in range(n_grace):
                    ctx_o = grace_operator(ctx_o, basis, np)
                    ctx_p = grace_operator(ctx_p, basis, np)
                
                # Extract witnesses
                w_o = extract_witness(ctx_o, basis, np)
                w_p = extract_witness(ctx_p, basis, np)
                
                # Compute similarity
                w_o_arr = np.array(w_o)
                w_p_arr = np.array(w_p)
                
                norm_o = np.sqrt(np.dot(w_o_arr, w_o_arr) + 1e-12)
                norm_p = np.sqrt(np.dot(w_p_arr, w_p_arr) + 1e-12)
                sim = np.dot(w_o_arr, w_p_arr) / (norm_o * norm_p)
                
                witness_sims_by_n[n_grace].append(sim)
        
        # Print results for this context length
        results[ctx_len] = {}
        for n_grace in [1, 2, 3, 5, 10]:
            mean_sim = np.mean(witness_sims_by_n[n_grace])
            std_sim = np.std(witness_sims_by_n[n_grace])
            results[ctx_len][n_grace] = (mean_sim, std_sim)
            
            status = "✓ CONVERGED" if mean_sim > 0.95 else "~ converging" if mean_sim > 0.8 else "✗ divergent"
            print(f"  {n_grace:<10} {mean_sim:.4f} ± {std_sim:.4f}  {status}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Witness Similarity vs Grace Iterations")
    print("="*70)
    print("\n  Theory prediction: More Grace iterations → higher witness similarity")
    print("  (Because Grace damps bivector differences, leaving only semantic core)")
    
    # Check if convergence improves with more iterations
    improvements = []
    for ctx_len in context_lengths:
        sim_1 = results[ctx_len][1][0]
        sim_10 = results[ctx_len][10][0]
        improvement = sim_10 - sim_1
        improvements.append(improvement)
        print(f"\n  ctx_len={ctx_len}: N=1 sim={sim_1:.4f}, N=10 sim={sim_10:.4f}, Δ={improvement:+.4f}")
    
    avg_improvement = np.mean(improvements)
    print(f"\n  Average improvement: {avg_improvement:+.4f}")
    
    # CRITICAL CHECK: Does iterated Grace help?
    if avg_improvement > 0.05:
        print("\n  ✓ CONFIRMED: Iterated Grace improves witness convergence")
        print("    → This supports using converged witness for generalization index")
        return True
    else:
        print("\n  ✗ UNEXPECTED: Iterated Grace doesn't help much")
        print("    → Need to investigate further")
        return False


def test_semantic_vs_full_context_keys():
    """
    Compare bucket collision rates:
    1. Full context 8D keys (current)
    2. Converged witness 2D keys (proposed for generalization)
    """
    print("\n" + "="*70)
    print("TEST: Full Context vs Converged Witness Keys")
    print("="*70)
    
    from holographic_v4.holographic_memory import VorticityWitnessIndex
    
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(50000, xp=np)
    rng = np.random.default_rng(42)
    
    n_patterns = 100
    ctx_len = 512
    
    # Generate original and perturbed contexts
    originals = []
    perturbed = []
    
    for _ in range(n_patterns):
        seq = rng.integers(0, 50000, size=ctx_len)
        mats = embeddings[seq]
        ctx = geometric_product_batch(mats, np)
        ctx = grace_operator(ctx, basis, np)
        originals.append(ctx)
        
        # Perturb
        seq_p = seq.copy()
        seq_p[0] = (seq_p[0] + 1) % 50000
        mats_p = embeddings[seq_p]
        ctx_p = geometric_product_batch(mats_p, np)
        ctx_p = grace_operator(ctx_p, basis, np)
        perturbed.append(ctx_p)
    
    # Method 1: Full 8D keys (current VorticityWitnessIndex)
    index_8d = VorticityWitnessIndex.create(basis, xp=np)
    matches_8d = 0
    
    for orig, pert in zip(originals, perturbed):
        key_o = index_8d._vorticity_key(orig)
        key_p = index_8d._vorticity_key(pert)
        if key_o == key_p:
            matches_8d += 1
    
    match_rate_8d = matches_8d / n_patterns
    print(f"\n  Full 8D keys (1 Grace): {match_rate_8d:.1%} match rate")
    
    # Method 2: Converged witness keys (5 Grace iterations, 2D witness only)
    def converged_witness_key(M, basis, n_grace=5, resolution=PHI_INV):
        """Extract witness from converged context."""
        ctx = M.copy()
        for _ in range(n_grace):
            ctx = grace_operator(ctx, basis, np)
        w = extract_witness(ctx, basis, np)
        s_idx = int(np.floor(w[0] / resolution))
        p_idx = int(np.floor(w[1] / resolution))
        return (s_idx, p_idx)
    
    matches_2d = 0
    for orig, pert in zip(originals, perturbed):
        key_o = converged_witness_key(orig, basis)
        key_p = converged_witness_key(pert, basis)
        if key_o == key_p:
            matches_2d += 1
    
    match_rate_2d = matches_2d / n_patterns
    print(f"  Converged 2D keys (5 Grace): {match_rate_2d:.1%} match rate")
    
    # Method 3: Converged 8D keys
    def converged_8d_key(M, basis, n_grace=5, index=None):
        """Extract full 8D from converged context."""
        ctx = M.copy()
        for _ in range(n_grace):
            ctx = grace_operator(ctx, basis, np)
        return index._vorticity_key(ctx)
    
    matches_8d_converged = 0
    for orig, pert in zip(originals, perturbed):
        key_o = converged_8d_key(orig, basis, index=index_8d)
        key_p = converged_8d_key(pert, basis, index=index_8d)
        if key_o == key_p:
            matches_8d_converged += 1
    
    match_rate_8d_conv = matches_8d_converged / n_patterns
    print(f"  Converged 8D keys (5 Grace): {match_rate_8d_conv:.1%} match rate")
    
    # Summary
    print("\n  INTERPRETATION:")
    print(f"    Full 8D (current): {match_rate_8d:.1%} - Perturbed contexts rarely match")
    print(f"    Converged 2D:      {match_rate_2d:.1%} - Better for generalization")
    print(f"    Converged 8D:      {match_rate_8d_conv:.1%} - Best of both?")
    
    improvement = match_rate_2d - match_rate_8d
    print(f"\n    Improvement from converged 2D: {improvement:+.1%}")
    
    if match_rate_2d > 0.5:
        print("\n  ✓ CONVERGED WITNESS KEYS enable generalization")
        return True
    else:
        print("\n  ✗ Need more investigation")
        return False


def test_generalization_with_dual_index():
    """
    Test proposed dual-index architecture:
    1. Episodic index: 8D keys for exact match
    2. Semantic index: Converged 2D keys for generalization
    """
    print("\n" + "="*70)
    print("TEST: Dual Index Architecture (Episodic + Semantic)")
    print("="*70)
    
    from holographic_v4.holographic_memory import VorticityWitnessIndex
    
    basis = build_clifford_basis(np)
    embeddings = initialize_embeddings_identity(50000, xp=np)
    rng = np.random.default_rng(42)
    
    # Create dual indices
    episodic_index = VorticityWitnessIndex.create(basis, xp=np)  # 8D keys
    
    # Semantic index using converged witness (we'll simulate with dict)
    semantic_index = {}  # (s_idx, p_idx) -> [(context, target, target_idx)]
    
    n_patterns = 200
    ctx_len = 512
    
    def converged_witness_key(M, n_grace=5, resolution=PHI_INV):
        ctx = M.copy()
        for _ in range(n_grace):
            ctx = grace_operator(ctx, basis, np)
        w = extract_witness(ctx, basis, np)
        s_idx = int(np.floor(w[0] / resolution))
        p_idx = int(np.floor(w[1] / resolution))
        return (s_idx, p_idx)
    
    # Store patterns
    stored_contexts = []
    stored_targets = []
    
    for i in range(n_patterns):
        seq = rng.integers(0, 50000, size=ctx_len)
        mats = embeddings[seq]
        ctx = geometric_product_batch(mats, np)
        ctx = grace_operator(ctx, basis, np)
        target = embeddings[rng.integers(0, 50000)]
        
        # Store in episodic (8D)
        episodic_index.store(ctx, target, i)
        
        # Store in semantic (2D converged)
        sem_key = converged_witness_key(ctx)
        if sem_key not in semantic_index:
            semantic_index[sem_key] = []
        semantic_index[sem_key].append((ctx, target, i))
        
        stored_contexts.append(ctx)
        stored_targets.append(i)
    
    print(f"\n  Stored {n_patterns} patterns")
    print(f"  Episodic buckets: {len(episodic_index.buckets)}")
    print(f"  Semantic buckets: {len(semantic_index)}")
    
    # Test retrieval with exact and perturbed contexts
    exact_correct = 0
    perturbed_correct_episodic = 0
    perturbed_correct_semantic = 0
    perturbed_correct_dual = 0
    
    n_test = 50
    for i in range(min(n_test, n_patterns)):
        original_ctx = stored_contexts[i]
        expected_target = stored_targets[i]
        
        # Exact retrieval (episodic)
        result, idx, conf = episodic_index.retrieve(original_ctx)
        if idx == expected_target:
            exact_correct += 1
        
        # Create perturbed context
        seq = rng.integers(0, 50000, size=ctx_len)
        seq[0] = (seq[0] + 1) % 50000  # Change first token
        mats = embeddings[seq]
        ctx_perturbed = geometric_product_batch(mats, np)
        ctx_perturbed = grace_operator(ctx_perturbed, basis, np)
        
        # Perturbed retrieval - episodic
        result_ep, idx_ep, conf_ep = episodic_index.retrieve(ctx_perturbed)
        if idx_ep == expected_target:
            perturbed_correct_episodic += 1
        
        # Perturbed retrieval - semantic
        sem_key = converged_witness_key(ctx_perturbed)
        if sem_key in semantic_index and len(semantic_index[sem_key]) > 0:
            # Use first match (simplified)
            _, _, idx_sem = semantic_index[sem_key][0]
            if idx_sem == expected_target:
                perturbed_correct_semantic += 1
        
        # Dual retrieval: try episodic first, fall back to semantic
        if idx_ep == expected_target:
            perturbed_correct_dual += 1
        elif sem_key in semantic_index and len(semantic_index[sem_key]) > 0:
            _, _, idx_sem = semantic_index[sem_key][0]
            if idx_sem == expected_target:
                perturbed_correct_dual += 1
    
    # Note: This test has a flaw - we're testing with random perturbed contexts
    # that don't necessarily match stored patterns. Let me fix this.
    
    print(f"\n  Exact retrieval accuracy: {exact_correct/n_test:.1%}")
    print(f"  (Note: Perturbed test uses random contexts, not actual perturbations of stored)")
    
    print("\n  Fixing test to use actual perturbations of stored contexts...")
    
    # Fixed test: perturb stored contexts
    perturbed_correct_ep = 0
    perturbed_correct_sem = 0
    
    for i in range(min(n_test, n_patterns)):
        # Get original sequence (we need to regenerate)
        rng_test = np.random.default_rng(42 + i)
        seq = rng_test.integers(0, 50000, size=ctx_len)
        
        # Original
        mats = embeddings[seq]
        ctx_orig = geometric_product_batch(mats, np)
        ctx_orig = grace_operator(ctx_orig, basis, np)
        
        # Perturbed (change token 0)
        seq_p = seq.copy()
        seq_p[0] = (seq_p[0] + 1) % 50000
        mats_p = embeddings[seq_p]
        ctx_pert = geometric_product_batch(mats_p, np)
        ctx_pert = grace_operator(ctx_pert, basis, np)
        
        # Check if perturbed matches original's bucket
        key_orig_ep = episodic_index._vorticity_key(ctx_orig)
        key_pert_ep = episodic_index._vorticity_key(ctx_pert)
        
        if key_orig_ep == key_pert_ep:
            perturbed_correct_ep += 1
        
        # Check semantic
        key_orig_sem = converged_witness_key(ctx_orig)
        key_pert_sem = converged_witness_key(ctx_pert)
        
        if key_orig_sem == key_pert_sem:
            perturbed_correct_sem += 1
    
    print(f"\n  BUCKET MATCH RATE (original vs perturbed):")
    print(f"    Episodic (8D, 1 Grace): {perturbed_correct_ep/n_test:.1%}")
    print(f"    Semantic (2D, 5 Grace): {perturbed_correct_sem/n_test:.1%}")
    
    if perturbed_correct_sem > perturbed_correct_ep:
        print(f"\n  ✓ Semantic index better for generalization: +{(perturbed_correct_sem-perturbed_correct_ep)/n_test:.1%}")
        return True
    else:
        print(f"\n  ✗ Unexpected: Semantic not better")
        return False


if __name__ == "__main__":
    print("="*70)
    print("ITERATED GRACE GENERALIZATION TESTS")
    print("="*70)
    print("\nTHEORY:")
    print("  Grace damps higher grades by φ⁻ᵏ per iteration.")
    print("  After N iterations, bivector differences decay as (0.382)^N.")
    print("  This should make similar contexts converge to the same witness.")
    print("  The converged witness is the theory-true 'semantic core'.")
    
    import time
    start = time.time()
    
    test1 = test_iterated_grace_convergence()
    test2 = test_semantic_vs_full_context_keys()
    test3 = test_generalization_with_dual_index()
    
    elapsed = time.time() - start
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n  Test 1 (Grace Convergence):     {'PASS' if test1 else 'FAIL'}")
    print(f"  Test 2 (Key Match Rates):        {'PASS' if test2 else 'FAIL'}")
    print(f"  Test 3 (Dual Index):             {'PASS' if test3 else 'FAIL'}")
    print(f"\n  Total time: {elapsed:.1f}s")
    
    if test1 and test2 and test3:
        print("\n  ✓ ALL TESTS PASS: Iterated Grace + Dual Index is theory-true")
        print("    → Implement: SemanticWitnessIndex with converged 2D keys")
        print("    → Use dual retrieval: episodic (exact) → semantic (general)")
    else:
        print("\n  ⚠ SOME TESTS FAILED: Need further investigation")
