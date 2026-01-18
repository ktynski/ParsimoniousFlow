"""
Pre-Modal Comprehensive Test Suite
==================================
Run this before expensive Modal runs to validate the implementation.
"""

import numpy as np
import time


def run_modal_simulation():
    """Simulate Modal-scale training locally."""
    from .pipeline import TheoryTrueModel
    
    print("=" * 70)
    print("MODAL SIMULATION TEST")
    print("=" * 70)
    
    # Reduced parameters for faster testing
    VOCAB_SIZE = 1000
    CONTEXT_SIZE = 4
    MAX_ATTRACTORS = 10000
    MAX_SAMPLES = 10000
    NOISE_STD = 0.5
    
    print(f"Config: vocab={VOCAB_SIZE}, ctx={CONTEXT_SIZE}, attractors={MAX_ATTRACTORS}")
    print()
    
    model = TheoryTrueModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        max_attractors=MAX_ATTRACTORS,
        noise_std=NOISE_STD,
        use_vorticity=True,
        use_equilibrium=False,  # Faster without equilibrium
        seed=42
    )
    
    # Training with semantic groups
    num_groups = 20
    group_size = VOCAB_SIZE // num_groups
    
    print("Training...")
    np.random.seed(42)
    start = time.time()
    
    for i in range(MAX_SAMPLES):
        if np.random.random() < 0.8:
            group = np.random.randint(0, num_groups)
            ctx = [np.random.randint(group * group_size, (group + 1) * group_size) 
                   for _ in range(CONTEXT_SIZE)]
            target = group * group_size + np.random.randint(0, 10)
        else:
            ctx = [np.random.randint(0, VOCAB_SIZE) for _ in range(CONTEXT_SIZE)]
            target = np.random.randint(0, VOCAB_SIZE)
        
        model.train_step(ctx, target)
    
    elapsed = time.time() - start
    rate = MAX_SAMPLES / elapsed
    avg_vort = model.total_vorticity / model.train_samples
    
    print(f"  {MAX_SAMPLES:,} samples in {elapsed:.1f}s ({rate:.0f}/s)")
    print(f"  Attractors: {model.num_attractors:,}")
    print(f"  Avg vorticity: {avg_vort:.4f}")
    print()
    
    # Evaluation
    print("Evaluation...")
    
    # In-group generalization
    np.random.seed(456)
    in_group = 0
    for _ in range(200):
        group = np.random.randint(0, num_groups)
        ctx = [np.random.randint(group * group_size, (group + 1) * group_size) 
               for _ in range(CONTEXT_SIZE)]
        _, pred = model.retrieve(ctx)
        if pred // group_size == group:
            in_group += 1
    
    in_group_pct = in_group / 2
    random_baseline = 100 / num_groups
    print(f"  In-group accuracy: {in_group_pct:.1f}% (random={random_baseline:.1f}%)")
    
    # Generation test
    print("\nGeneration test:")
    
    def generate(m, ctx, length=15):
        tokens = list(ctx)
        for _ in range(length):
            c = tokens[-CONTEXT_SIZE:]
            _, pred = m.retrieve(c)
            tokens.append(pred)
        return tokens
    
    degen_count = 0
    for trial in range(5):
        start_ctx = [np.random.randint(0, VOCAB_SIZE) for _ in range(CONTEXT_SIZE)]
        gen = generate(model, start_ctx, 15)
        last = gen[-8:]
        unique = len(set(last))
        is_degen = unique <= 2
        if is_degen:
            degen_count += 1
        print(f"  Trial {trial+1}: unique={unique}/8 {'DEGEN' if is_degen else 'OK'}")
    
    print(f"\nDegenerate: {degen_count}/5")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    results = {
        "Speed > 5k/s": rate > 5000,
        "Vorticity > 0.1": avg_vort > 0.1,
        "In-group > random*2": in_group_pct > random_baseline * 2,
        "Non-degenerate": degen_count <= 2,
    }
    
    for k, v in results.items():
        print(f"  {k}: {'✓ PASS' if v else '✗ FAIL'}")
    
    passed = all(results.values())
    if passed:
        print("\n✅ ALL CHECKS PASSED — Safe to run Modal!")
    else:
        print("\n❌ ISSUES FOUND — Review before Modal run")
    
    return passed, results


def run_all_pre_modal_tests():
    """Run all pre-modal validation tests."""
    from .tests import run_all_tests
    
    print("=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)
    print()
    
    # Run unit tests
    test_results = run_all_tests()
    unit_passed = all(test_results.values())
    
    print()
    print("=" * 70)
    print("RUNNING SIMULATION")
    print("=" * 70)
    print()
    
    # Run simulation
    sim_passed, sim_results = run_modal_simulation()
    
    # Overall summary
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"  Unit tests: {sum(test_results.values())}/{len(test_results)}")
    print(f"  Simulation: {sum(sim_results.values())}/{len(sim_results)}")
    
    if unit_passed and sim_passed:
        print("\n✅ ALL PRE-MODAL CHECKS PASSED!")
        return True
    else:
        print("\n❌ SOME CHECKS FAILED — Fix before Modal run")
        return False


if __name__ == "__main__":
    run_all_pre_modal_tests()
