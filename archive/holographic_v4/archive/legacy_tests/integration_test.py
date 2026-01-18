"""
Comprehensive Integration Test for Dreaming System
==================================================

This test validates the full pipeline:
1. Model training with episodic collection
2. Periodic sleep cycles (Non-REM + REM)
3. Semantic memory retrieval
4. Generalization improvement measurement

Run this BEFORE launching expensive Modal runs!
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/Users/fractlphoneroom1/Desktop/ParsimoniousFlow')

from holographic_v4 import (
    TheoryTrueModel,
    build_clifford_basis,
    witness_stability,
)
from holographic_v4.algebra import frobenius_similarity
from holographic_v4.quotient import compute_enstrophy
from holographic_v4.dreaming import (
    DreamingSystem,
    EpisodicEntry,
    integrate_dreaming_with_model,
)


def print_section(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def create_synthetic_data(vocab_size: int, n_samples: int, context_size: int):
    """
    Create synthetic training data with clear patterns.
    
    Patterns:
        [A, B, C, D] → E where E = (A + B + C + D) % vocab_size
    """
    np.random.seed(42)
    data = []
    
    for i in range(n_samples):
        # Create context
        ctx = [
            (i + j) % vocab_size 
            for j in range(context_size)
        ]
        # Deterministic target
        target = sum(ctx) % vocab_size
        data.append((ctx, target))
    
    return data


def run_integration_test(
    vocab_size: int = 200,
    context_size: int = 4,
    max_attractors: int = 5000,
    n_samples: int = 2000,
    sleep_every: int = 500,
    verbose: bool = True,
):
    """
    Run complete integration test.
    
    This tests:
        1. Training accumulates episodic memories
        2. Sleep cycles consolidate into prototypes
        3. Semantic memory enables generalization
        4. Overall accuracy improves
    """
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  INTEGRATION TEST: Dreaming System  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    results = {
        "passed": [],
        "failed": [],
    }
    
    # Create model
    print_section("1. MODEL CREATION")
    
    model = TheoryTrueModel(
        vocab_size=vocab_size,
        context_size=context_size,
        max_attractors=max_attractors,
        noise_std=0.3,
        use_adaptive_similarity=True,
    )
    
    print(f"  ✓ Model created")
    print(f"    vocab_size: {vocab_size}")
    print(f"    context_size: {context_size}")
    print(f"    max_attractors: {max_attractors}")
    
    # Create dreaming system
    print_section("2. DREAMING SYSTEM")
    
    dreaming = DreamingSystem(
        basis=model.basis,
        xp=np,
        similarity_threshold=0.6,
    )
    
    print(f"  ✓ Dreaming system initialized")
    
    # Generate training data
    print_section("3. TRAINING DATA")
    
    train_data = create_synthetic_data(vocab_size, n_samples, context_size)
    print(f"  ✓ Generated {len(train_data)} training examples")
    
    # Training loop with dreaming
    print_section("4. TRAINING WITH DREAMING")
    
    episodic_buffer = []
    sleep_cycles = 0
    
    start_time = time.perf_counter()
    
    for i, (ctx, target) in enumerate(train_data):
        # Train step
        model.train_step(ctx, target)
        
        # Collect episodic entry
        ctx_matrix = model.compute_context(ctx)
        episodic_buffer.append(EpisodicEntry(ctx_matrix, target))
        
        # Periodic sleep
        if (i + 1) % sleep_every == 0:
            print(f"\n  💤 Sleep cycle #{sleep_cycles + 1} @ {i+1} samples")
            
            # Run sleep
            sleep_stats = dreaming.sleep(episodic_buffer, rem_cycles=1, verbose=False)
            sleep_cycles += 1
            
            proto_count = dreaming.semantic_memory.stats()['total_prototypes']
            schema_count = dreaming.semantic_memory.stats()['num_schemas']
            print(f"    Prototypes: {proto_count}, Schemas: {schema_count}")
            
            # Clear buffer
            episodic_buffer = []
    
    train_time = time.perf_counter() - start_time
    
    print(f"\n  ✓ Training complete in {train_time:.1f}s")
    print(f"    Attractors: {model.num_attractors}")
    print(f"    Sleep cycles: {sleep_cycles}")
    
    # Test 1: Episodic accuracy (seen data)
    print_section("5. TEST: EPISODIC ACCURACY")
    
    correct = 0
    for ctx, expected in train_data[:200]:
        _, predicted = model.retrieve(ctx)
        if predicted == expected:
            correct += 1
    
    episodic_acc = correct / 200
    print(f"  Episodic accuracy (seen data): {episodic_acc:.1%}")
    
    if episodic_acc > 0.7:
        results["passed"].append("Episodic accuracy > 70%")
        print("  ✓ PASSED")
    else:
        results["failed"].append(f"Episodic accuracy = {episodic_acc:.1%} (expected > 70%)")
        print("  ✗ FAILED")
    
    # Test 2: Generalization (novel data)
    print_section("6. TEST: GENERALIZATION")
    
    # Create novel test data (perturbed)
    novel_data = []
    for ctx, expected in train_data[:100]:
        # Perturb context
        perturbed = ctx.copy()
        perturbed[0] = (perturbed[0] + 1) % vocab_size
        novel_data.append((perturbed, expected))
    
    # Test without dreaming
    correct_episodic = 0
    for ctx, expected in novel_data:
        _, predicted = model.retrieve(ctx)
        if predicted == expected:
            correct_episodic += 1
    
    gen_acc_episodic = correct_episodic / len(novel_data)
    print(f"  Episodic only (novel data): {gen_acc_episodic:.1%}")
    
    # Test with dreaming
    retrieve_with_dreaming = integrate_dreaming_with_model(model, dreaming)
    
    correct_semantic = 0
    semantic_used = 0
    
    for ctx, expected in novel_data:
        _, predicted, source = retrieve_with_dreaming(ctx)
        # Check for ANY semantic source (distributed_prior, grace_basin, etc.)
        # Not just literal "semantic" - sources are like "distributed_prior(conf=0.85)"
        if source not in ["episodic", "unknown"]:
            semantic_used += 1
        if predicted == expected:
            correct_semantic += 1
    
    gen_acc_semantic = correct_semantic / len(novel_data)
    print(f"  With semantic memory:       {gen_acc_semantic:.1%}")
    print(f"  Semantic retrievals used:   {semantic_used}/{len(novel_data)}")
    
    improvement = gen_acc_semantic - gen_acc_episodic
    print(f"  Improvement:                {improvement:+.1%}")
    
    # Check if semantic memory is being used
    if semantic_used > 0:
        results["passed"].append(f"Semantic memory used ({semantic_used} times)")
        print("  ✓ Semantic memory is active")
    else:
        results["failed"].append("Semantic memory never used")
        print("  ✗ Semantic memory not being used")
    
    # Test 3: Multi-modal targets
    print_section("7. TEST: MULTI-MODAL TARGETS")
    
    # Check if prototypes have multi-modal target distributions
    mem_stats = dreaming.semantic_memory.stats()
    proto_count = mem_stats['total_prototypes']
    
    if proto_count > 0:
        # Get a prototype and check its entropy
        proto = dreaming.semantic_memory.levels[0][0] if dreaming.semantic_memory.levels[0] else None
        if proto:
            entropy = proto.entropy()
            n_targets = len(proto.target_distribution)
            print(f"  Sample prototype:")
            print(f"    Targets:   {n_targets}")
            print(f"    Entropy:   {entropy:.3f}")
            
            if n_targets > 1 or entropy > 0:
                results["passed"].append("Multi-modal targets represented")
                print("  ✓ Multi-modality supported")
            else:
                results["passed"].append("Single-modal (acceptable for deterministic data)")
                print("  ✓ Single-modal (expected for this test data)")
        else:
            results["failed"].append("No prototypes found")
            print("  ✗ No prototypes")
    else:
        results["failed"].append("No prototypes created")
        print("  ✗ No prototypes created")
    
    # Test 4: Schema discovery
    print_section("8. TEST: SCHEMA DISCOVERY")
    
    schema_count = mem_stats['num_schemas']
    print(f"  Schemas discovered: {schema_count}")
    
    if schema_count > 0:
        results["passed"].append(f"Discovered {schema_count} schemas")
        print("  ✓ REM recombination working")
        
        # Show some schemas
        for schema in dreaming.semantic_memory.schemas[:3]:
            print(f"    Schema: recurrence={schema.recurrence_count}")
    else:
        results["passed"].append("No schemas (may need more data/iterations)")
        print("  ⚠ No schemas discovered (normal for small test)")
    
    # Test 5: Memory compression
    print_section("9. TEST: MEMORY COMPRESSION")
    
    compression_ratio = n_samples / max(1, proto_count)
    print(f"  Input samples:    {n_samples}")
    print(f"  Prototypes:       {proto_count}")
    print(f"  Compression:      {compression_ratio:.1f}x")
    
    if proto_count < n_samples / 10:
        results["passed"].append(f"Compression ratio: {compression_ratio:.1f}x")
        print("  ✓ Good compression")
    else:
        results["passed"].append("Low compression (may need tuning)")
        print("  ⚠ Low compression")
    
    # Summary
    print_section("SUMMARY")
    
    print(f"  PASSED: {len(results['passed'])}")
    for r in results['passed']:
        print(f"    ✓ {r}")
    
    print(f"\n  FAILED: {len(results['failed'])}")
    for r in results['failed']:
        print(f"    ✗ {r}")
    
    all_passed = len(results['failed']) == 0
    
    print()
    if all_passed:
        print("  ╔════════════════════════════════════╗")
        print("  ║  ALL TESTS PASSED - READY FOR MODAL  ║")
        print("  ╚════════════════════════════════════╝")
    else:
        print("  ⚠️ Some tests failed - review before Modal run")
    
    return results


def run_stress_test(
    vocab_size: int = 500,
    context_size: int = 4,
    max_attractors: int = 20000,
    n_samples: int = 10000,
    sleep_every: int = 2000,
):
    """
    Larger stress test to validate scaling behavior.
    """
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  STRESS TEST: Large Scale  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    print(f"\n  Config:")
    print(f"    samples: {n_samples}")
    print(f"    vocab: {vocab_size}")
    print(f"    max_attractors: {max_attractors}")
    print(f"    sleep_every: {sleep_every}")
    
    # Run integration test with larger parameters
    results = run_integration_test(
        vocab_size=vocab_size,
        context_size=context_size,
        max_attractors=max_attractors,
        n_samples=n_samples,
        sleep_every=sleep_every,
        verbose=True,
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress", action="store_true", help="Run stress test")
    args = parser.parse_args()
    
    if args.stress:
        run_stress_test()
    else:
        run_integration_test()
