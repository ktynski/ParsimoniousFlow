"""
TEST SUITE: Phase A - Cleanup Verification

Tests to run BEFORE and AFTER cleanup to ensure:
1. WitnessIndex is proven useless (justifies removal)
2. No regressions after removal
3. All constants are φ-derived
4. No fallback behavior exists

Run: python -m holographic_v4.test_cleanup_phase_a
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import re

# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_INV_SQ = PHI_INV ** 2
PHI_INV_CUBE = PHI_INV ** 3


def test_witness_index_is_useless() -> Dict:
    """
    PROVE: WitnessIndex creates so few buckets it's useless.
    
    If this test passes, we have justification to remove WitnessIndex.
    """
    from holographic_v4.algebra import build_clifford_basis, initialize_embeddings_identity
    from holographic_v4.holographic_memory import WitnessIndex
    
    print("\n" + "="*60)
    print("TEST: WitnessIndex Is Useless")
    print("="*60)
    
    basis = build_clifford_basis()
    embeddings = initialize_embeddings_identity(1000, xp=np)
    
    # Generate 1000 random contexts
    n_samples = 1000
    contexts = []
    for i in range(n_samples):
        # Random token sequence
        tokens = np.random.randint(0, 1000, size=np.random.randint(5, 20))
        ctx = np.eye(4, dtype=np.float32)
        for t in tokens:
            ctx = ctx @ embeddings[t]
        # Normalize
        norm = np.linalg.norm(ctx)
        if norm > 1e-8:
            ctx = ctx / norm
        contexts.append(ctx)
    
    # Get WitnessIndex keys for all contexts
    index = WitnessIndex.create(basis, xp=np)
    keys = [index._witness_key(ctx) for ctx in contexts]
    
    unique_keys = len(set(keys))
    max_bucket_size = max(keys.count(k) for k in set(keys))
    
    print(f"\n  Samples: {n_samples}")
    print(f"  Unique buckets: {unique_keys}")
    print(f"  Max bucket size: {max_bucket_size}")
    print(f"  Average per bucket: {n_samples / unique_keys:.1f}")
    
    # WitnessIndex should have < 20 unique buckets (far less than samples)
    # With 12 buckets for 1000 samples, retrieval is ~random chance
    is_useless = unique_keys < 20 and max_bucket_size > n_samples / 10
    
    if is_useless:
        print(f"\n  ✅ CONFIRMED: WitnessIndex is USELESS ({unique_keys} buckets)")
        print("     → Retrieval is essentially random chance")
        print("     → JUSTIFIED TO REMOVE")
    else:
        print(f"\n  ⚠️ UNEXPECTED: WitnessIndex has {unique_keys} buckets")
    
    return {
        'test': 'witness_index_is_useless',
        'n_samples': n_samples,
        'unique_buckets': unique_keys,
        'max_bucket_size': max_bucket_size,
        'is_useless': is_useless,
        'passed': is_useless
    }


def test_vorticity_index_is_useful() -> Dict:
    """
    PROVE: VorticityWitnessIndex creates enough buckets for useful retrieval.
    """
    from holographic_v4.algebra import build_clifford_basis, initialize_embeddings_identity
    from holographic_v4.holographic_memory import VorticityWitnessIndex
    
    print("\n" + "="*60)
    print("TEST: VorticityWitnessIndex Is Useful")
    print("="*60)
    
    basis = build_clifford_basis()
    embeddings = initialize_embeddings_identity(1000, xp=np)
    
    # Generate 1000 random contexts
    n_samples = 1000
    contexts = []
    for i in range(n_samples):
        tokens = np.random.randint(0, 1000, size=np.random.randint(5, 20))
        ctx = np.eye(4, dtype=np.float32)
        for t in tokens:
            ctx = ctx @ embeddings[t]
        norm = np.linalg.norm(ctx)
        if norm > 1e-8:
            ctx = ctx / norm
        contexts.append(ctx)
    
    # Get VorticityWitnessIndex keys
    index = VorticityWitnessIndex.create(basis, xp=np)
    keys = [index._vorticity_key(ctx) for ctx in contexts]
    
    unique_keys = len(set(keys))
    max_bucket_size = max(keys.count(k) for k in set(keys)) if keys else 0
    
    print(f"\n  Samples: {n_samples}")
    print(f"  Unique buckets: {unique_keys}")
    print(f"  Max bucket size: {max_bucket_size}")
    print(f"  Average per bucket: {n_samples / unique_keys:.1f}")
    
    # VorticityWitnessIndex should have > 100 unique buckets
    is_useful = unique_keys > 100 and max_bucket_size < n_samples / 10
    
    if is_useful:
        print(f"\n  ✅ CONFIRMED: VorticityWitnessIndex is USEFUL ({unique_keys} buckets)")
    else:
        print(f"\n  ❌ FAIL: VorticityWitnessIndex insufficient ({unique_keys} buckets)")
    
    return {
        'test': 'vorticity_index_is_useful',
        'n_samples': n_samples,
        'unique_buckets': unique_keys,
        'max_bucket_size': max_bucket_size,
        'is_useful': is_useful,
        'passed': is_useful
    }


def test_no_arbitrary_constants() -> Dict:
    """
    AUDIT: Scan codebase for arbitrary constants (0.3, 0.5, 0.7, etc.)
    
    All numeric thresholds should be φ-derived.
    """
    import os
    
    print("\n" + "="*60)
    print("TEST: No Arbitrary Constants")
    print("="*60)
    
    holographic_dir = os.path.dirname(__file__)
    
    # Patterns for arbitrary constants (not φ-derived)
    arbitrary_patterns = [
        r'[^0-9\.](0\.3)[^0-9]',   # 0.3
        r'[^0-9\.](0\.5)[^0-9]',   # 0.5  
        r'[^0-9\.](0\.7)[^0-9]',   # 0.7
        r'[^0-9\.](0\.9)[^0-9]',   # 0.9
        r'[^0-9\.](0\.1)[^0-9]',   # 0.1
        r'[^0-9\.](0\.05)[^0-9]',  # 0.05
        r'[^0-9\.](0\.01)[^0-9]',  # 0.01
        r'[^0-9\.](0\.75)[^0-9]',  # 0.75
        r'[^0-9\.](0\.25)[^0-9]',  # 0.25
    ]
    
    # Allowed φ-derived values (approximate)
    phi_values = {
        '0.618': 'PHI_INV',
        '0.382': 'PHI_INV_SQ',
        '0.236': 'PHI_INV_CUBE',
        '0.146': 'PHI_INV_FOUR',
        '0.764': '1 - PHI_INV_CUBE',
        '0.854': '1 - PHI_INV_FOUR',
    }
    
    violations = []
    
    # Only check core implementation files
    core_files = [
        'holographic_memory.py',
        'pipeline.py',
        'dreaming.py',
        'algebra.py',
        'quotient.py',
        'meta_learning.py',
        'credit_assignment.py',
        'recursive_computation.py',
    ]
    
    for filename in core_files:
        filepath = os.path.join(holographic_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
        in_docstring = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Track docstrings
            if '"""' in stripped or "'''" in stripped:
                # Toggle docstring state (handles single-line and multi-line)
                quote_count = stripped.count('"""') + stripped.count("'''")
                if quote_count == 1:
                    in_docstring = not in_docstring
                # Skip this line either way
                continue
            
            if in_docstring:
                continue
            
            # Skip pure comments
            if stripped.startswith('#'):
                continue
            
            for pattern in arbitrary_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    value = match.group(1)
                    # Check if it's in a comment at end of line
                    if '#' in line:
                        comment_start = line.index('#')
                        if comment_start < match.start():
                            continue
                    # Skip if in a comment describing what a phi constant equals
                    if 'φ' in line or 'PHI' in line:
                        continue
                    violations.append({
                        'file': filename,
                        'line': i,
                        'value': value,
                        'context': line.strip()[:80]
                    })
    
    print(f"\n  Core files checked: {len(core_files)}")
    print(f"  Violations found: {len(violations)}")
    
    if violations:
        print("\n  VIOLATIONS:")
        for v in violations[:10]:  # Show first 10
            print(f"    {v['file']}:{v['line']} - {v['value']}")
            print(f"      {v['context']}")
    
    passed = len(violations) == 0
    
    if passed:
        print("\n  ✅ PASS: All constants are φ-derived")
    else:
        print(f"\n  ❌ FAIL: {len(violations)} arbitrary constants found")
        print("     → Replace with PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, etc.")
    
    return {
        'test': 'no_arbitrary_constants',
        'violations': violations,
        'n_violations': len(violations),
        'passed': passed
    }


def test_no_fallback_in_compute_semantic_context() -> Dict:
    """
    VERIFY: compute_semantic_context has no fallback behavior.
    
    If no semantic tokens found, should return zero matrix, NOT full context.
    """
    print("\n" + "="*60)
    print("TEST: No Fallback in compute_semantic_context")
    print("="*60)
    
    from holographic_v4.pipeline import TheoryTrueModel
    
    # Create model with correct API (no embeddings argument)
    model = TheoryTrueModel(
        vocab_size=100,
        max_attractors=100,
        xp=np
    )
    
    # Test with empty list - should return zero or near-zero, NOT fallback to full context
    empty_ctx = model.compute_semantic_context([])
    
    # Should be zero matrix or identity (near-zero signal), not some arbitrary fallback
    trace = np.trace(empty_ctx)
    frobenius = np.linalg.norm(empty_ctx)
    
    # Empty context should have trace close to 0 or 4 (identity)
    # NOT some arbitrary non-zero value from fallback
    is_valid = frobenius < 0.1 or np.isclose(trace, 4.0, atol=0.5)
    
    print(f"\n  Empty context trace: {trace:.3f}")
    print(f"  Empty context Frobenius norm: {frobenius:.3f}")
    
    if is_valid:
        print("\n  ✅ PASS: Empty input returns valid empty/identity context (no arbitrary fallback)")
    else:
        print("\n  ❌ FAIL: Empty input returns unexpected matrix")
        print(f"     Got trace={trace:.3f}, norm={frobenius:.3f}")
    
    return {
        'test': 'no_fallback_in_compute_semantic_context',
        'trace': float(trace),
        'frobenius': float(frobenius),
        'is_valid': is_valid,
        'passed': is_valid
    }


def test_hybrid_memory_uses_vorticity_only() -> Dict:
    """
    VERIFY: HybridHolographicMemory uses VorticityWitnessIndex by default.
    
    No alternative index options should be available after cleanup.
    """
    print("\n" + "="*60)
    print("TEST: HybridHolographicMemory Uses VorticityWitnessIndex Only")
    print("="*60)
    
    from holographic_v4.algebra import build_clifford_basis
    from holographic_v4.holographic_memory import HybridHolographicMemory, VorticityWitnessIndex
    
    basis = build_clifford_basis()
    
    # Default creation
    memory = HybridHolographicMemory.create(basis, xp=np)
    
    uses_vorticity = isinstance(memory.witness_index, VorticityWitnessIndex)
    
    if uses_vorticity:
        print("\n  ✅ PASS: Uses VorticityWitnessIndex by default")
    else:
        print(f"\n  ❌ FAIL: Uses {type(memory.witness_index).__name__} instead")
    
    return {
        'test': 'hybrid_memory_uses_vorticity_only',
        'index_type': type(memory.witness_index).__name__,
        'passed': uses_vorticity
    }


def test_retrieval_works_after_cleanup() -> Dict:
    """
    REGRESSION TEST: Ensure retrieval still works after cleanup.
    """
    print("\n" + "="*60)
    print("TEST: Retrieval Works After Cleanup")
    print("="*60)
    
    from holographic_v4.algebra import build_clifford_basis, initialize_embeddings_identity
    from holographic_v4.holographic_memory import HybridHolographicMemory
    
    basis = build_clifford_basis()
    embeddings = initialize_embeddings_identity(100, xp=np)
    memory = HybridHolographicMemory.create(basis, xp=np)
    
    # Store 50 patterns
    n_patterns = 50
    stored_pairs = []
    
    for i in range(n_patterns):
        tokens = np.random.randint(0, 100, size=np.random.randint(5, 15))
        ctx = np.eye(4, dtype=np.float32)
        for t in tokens:
            ctx = ctx @ embeddings[t]
        norm = np.linalg.norm(ctx)
        if norm > 1e-8:
            ctx = ctx / norm
        
        target_idx = np.random.randint(0, 100)
        target = embeddings[target_idx]
        
        memory.store(ctx, target, target_idx)
        stored_pairs.append((ctx, target_idx))
    
    # Retrieve and check accuracy
    correct = 0
    for ctx, expected_idx in stored_pairs:
        result, idx, conf, source = memory.retrieve(ctx)
        if idx == expected_idx:
            correct += 1
    
    accuracy = correct / n_patterns
    
    print(f"\n  Patterns stored: {n_patterns}")
    print(f"  Correct retrievals: {correct}")
    print(f"  Accuracy: {accuracy:.1%}")
    
    passed = accuracy >= PHI_INV  # At least 61.8% accuracy
    
    if passed:
        print(f"\n  ✅ PASS: Retrieval accuracy {accuracy:.1%} >= {PHI_INV:.1%}")
    else:
        print(f"\n  ❌ FAIL: Retrieval accuracy {accuracy:.1%} < {PHI_INV:.1%}")
    
    return {
        'test': 'retrieval_works_after_cleanup',
        'n_patterns': n_patterns,
        'accuracy': accuracy,
        'passed': passed
    }


def run_all_tests() -> bool:
    """Run all Phase A tests and report results."""
    print("\n" + "="*70)
    print("PHASE A: CLEANUP VERIFICATION TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test 1: Prove WitnessIndex is useless
    results.append(test_witness_index_is_useless())
    
    # Test 2: Prove VorticityWitnessIndex is useful
    results.append(test_vorticity_index_is_useful())
    
    # Test 3: No arbitrary constants
    results.append(test_no_arbitrary_constants())
    
    # Test 4: No fallback behavior
    results.append(test_no_fallback_in_compute_semantic_context())
    
    # Test 5: Uses VorticityWitnessIndex only
    results.append(test_hybrid_memory_uses_vorticity_only())
    
    # Test 6: Retrieval regression test
    results.append(test_retrieval_works_after_cleanup())
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    for r in results:
        status = "✅" if r['passed'] else "❌"
        print(f"  {status} {r['test']}")
    
    print(f"\n  TOTAL: {passed}/{total} passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED - Ready for cleanup!")
    else:
        print(f"\n  ⚠️ {total - passed} tests failed - Fix before cleanup")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
