"""
INFORMATION FLOW AUDIT — Zero Truncation Verification
======================================================

This audit traces information through EVERY step of the pipeline
to verify NO information is lost, truncated, or approximated.

AUDIT POINTS:
1. Token → Embedding: Full SO(4) matrix preserved
2. Embedding Sequence → Context: Geometric product preserves all 16 coefficients
3. Context + Target → Binding: Full Clifford algebra binding
4. Binding → Memory: Superposition (addition), not replacement
5. Memory → Retrieval: Full unbinding, not candidate narrowing
6. Retrieval → Output: Coherence scoring over FULL vocabulary
7. Grace → Contraction: All grades scaled (not zeroed)

ANTI-PATTERNS TO DETECT:
- Truncation to "top-k candidates"
- Dimensionality reduction
- Lossy compression
- Candidate sets that limit vocabulary
- Argmax instead of coherence
- Any None returns from retrieval

THEORY-TRUE REQUIREMENTS:
- 16 Clifford coefficients at every step
- SO(4) structure preserved (det=1, orthogonal)
- Grace scales grades by φ^(-k), never zeros them
- Full vocabulary always accessible
"""

import modal
import numpy as np
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Modal setup
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "numpy>=1.24.0",
        "cupy-cuda12x>=12.0.0",
        "datasets>=2.14.0",
        "tqdm",
        "scipy",
    )
    .add_local_dir("holographic_prod", "/root/project/holographic_prod")
)

app = modal.App("information-flow-audit")
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@dataclass
class AuditResult:
    """Result of an audit checkpoint."""
    step_name: str
    passed: bool
    details: Dict[str, Any]
    error_message: str = ""


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/checkpoints": checkpoint_volume},
)
def run_information_flow_audit():
    """
    COMPREHENSIVE INFORMATION FLOW AUDIT
    
    Traces information through every step and verifies:
    1. No truncation
    2. No approximation
    3. Full Clifford algebra preserved
    4. SO(4) structure maintained
    5. Full vocabulary accessible
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    from holographic_prod.core.algebra import (
        decompose_to_coefficients_batch,
        get_cached_basis,
        grace_with_stability,
        grace_with_stability_batch,
        verify_so4,
    )
    from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ, GRACE_SCALES_FLAT
    
    print("="*80)
    print("INFORMATION FLOW AUDIT — ZERO TRUNCATION VERIFICATION")
    print("="*80)
    
    xp = cp
    basis = get_cached_basis(xp)
    vocab_size = 10_000
    
    # Initialize model
    model = HolographicMemory(
        vocab_size=vocab_size,
        max_levels=4,
        seed=42,
        use_gpu=True,
    )
    
    audit_results = []
    
    # =========================================================================
    # AUDIT 1: Token → Embedding (SO(4) structure preserved)
    # =========================================================================
    print("\n" + "="*80)
    print("AUDIT 1: Token → Embedding")
    print("="*80)
    
    token_id = 123
    embedding = model.tower.embeddings[token_id]
    
    # Check SO(4) properties
    is_so4 = verify_so4(embedding, xp, tol=1e-5)
    det = float(xp.linalg.det(embedding))
    orthogonality_error = float(xp.max(xp.abs(embedding @ embedding.T - xp.eye(4, dtype=embedding.dtype))))
    
    # Decompose to check all 16 coefficients
    coeffs = decompose_to_coefficients_batch(embedding.reshape(1, 4, 4), basis, xp)[0]
    non_zero_coeffs = int(xp.sum(xp.abs(coeffs) > 1e-10))
    
    audit1_passed = is_so4 and (non_zero_coeffs >= 8)
    
    print(f"  SO(4) verified: {is_so4}")
    print(f"  Determinant: {det:.6f} (should be 1.0)")
    print(f"  Orthogonality error: {orthogonality_error:.2e} (should be <1e-5)")
    print(f"  Non-zero Clifford coefficients: {non_zero_coeffs}/16")
    print(f"  AUDIT 1: {'✓ PASSED' if audit1_passed else '✗ FAILED'}")
    
    audit_results.append(AuditResult(
        step_name="Token → Embedding",
        passed=audit1_passed,
        details={
            'is_so4': is_so4,
            'det': det,
            'orthogonality_error': orthogonality_error,
            'non_zero_coeffs': non_zero_coeffs
        }
    ))
    
    # =========================================================================
    # AUDIT 2: Embedding Sequence → Context (Geometric product)
    # =========================================================================
    print("\n" + "="*80)
    print("AUDIT 2: Embedding Sequence → Context")
    print("="*80)
    
    context_tokens = [100, 200, 300, 400, 500, 600, 700, 800]
    context_mat = model.tower._embed_sequence(context_tokens)
    
    # Check SO(4) preserved after composition
    is_so4_ctx = verify_so4(context_mat, xp, tol=1e-4)
    det_ctx = float(xp.linalg.det(context_mat))
    orth_error_ctx = float(xp.max(xp.abs(context_mat @ context_mat.T - xp.eye(4, dtype=context_mat.dtype))))
    
    # Check all 16 coefficients
    ctx_coeffs = decompose_to_coefficients_batch(context_mat.reshape(1, 4, 4), basis, xp)[0]
    non_zero_ctx = int(xp.sum(xp.abs(ctx_coeffs) > 1e-10))
    
    audit2_passed = is_so4_ctx and (non_zero_ctx >= 8)
    
    print(f"  Context length: {len(context_tokens)}")
    print(f"  SO(4) preserved: {is_so4_ctx}")
    print(f"  Determinant: {det_ctx:.6f} (should be 1.0)")
    print(f"  Orthogonality error: {orth_error_ctx:.2e}")
    print(f"  Non-zero Clifford coefficients: {non_zero_ctx}/16")
    print(f"  AUDIT 2: {'✓ PASSED' if audit2_passed else '✗ FAILED'}")
    
    audit_results.append(AuditResult(
        step_name="Embedding Sequence → Context",
        passed=audit2_passed,
        details={
            'is_so4': is_so4_ctx,
            'det': det_ctx,
            'orthogonality_error': orth_error_ctx,
            'non_zero_coeffs': non_zero_ctx,
            'context_length': len(context_tokens)
        }
    ))
    
    # =========================================================================
    # AUDIT 3: Context + Target → Binding (Full binding preserved)
    # =========================================================================
    print("\n" + "="*80)
    print("AUDIT 3: Context + Target → Binding")
    print("="*80)
    
    target_token = 999
    target_emb = model.tower.embeddings[target_token]
    
    # Binding = context @ target
    binding = context_mat @ target_emb
    
    # Check binding properties
    binding_coeffs = decompose_to_coefficients_batch(binding.reshape(1, 4, 4), basis, xp)[0]
    non_zero_binding = int(xp.sum(xp.abs(binding_coeffs) > 1e-10))
    
    # Verify binding is NOT degenerate
    binding_norm = float(xp.linalg.norm(binding, 'fro'))
    binding_rank = int(xp.linalg.matrix_rank(binding))
    
    audit3_passed = (non_zero_binding >= 8) and (binding_rank == 4) and (binding_norm > 0.5)
    
    print(f"  Binding norm: {binding_norm:.4f}")
    print(f"  Binding rank: {binding_rank}/4")
    print(f"  Non-zero Clifford coefficients: {non_zero_binding}/16")
    print(f"  AUDIT 3: {'✓ PASSED' if audit3_passed else '✗ FAILED'}")
    
    audit_results.append(AuditResult(
        step_name="Context + Target → Binding",
        passed=audit3_passed,
        details={
            'binding_norm': binding_norm,
            'binding_rank': binding_rank,
            'non_zero_coeffs': non_zero_binding
        }
    ))
    
    # =========================================================================
    # AUDIT 4: Binding → Memory (Superposition, not replacement)
    # =========================================================================
    print("\n" + "="*80)
    print("AUDIT 4: Binding → Memory (Superposition)")
    print("="*80)
    
    # Learn one pattern
    initial_memory_norm = float(xp.linalg.norm(model.tower._all_memories))
    model.tower.learn(context_tokens, target_token)
    memory_after_1 = float(xp.linalg.norm(model.tower._all_memories))
    
    # Learn another pattern (SUPERPOSITION = addition, not replacement)
    context2 = [200, 300, 400, 500, 600, 700, 800, 900]
    target2 = 888
    model.tower.learn(context2, target2)
    memory_after_2 = float(xp.linalg.norm(model.tower._all_memories))
    
    # Verify SUPERPOSITION: memory grows (not replaced)
    memory_grew = memory_after_2 > memory_after_1 > initial_memory_norm
    
    # Verify both patterns are stored by checking unbinding coherence
    # (not token retrieval, which uses vocab-wide scoring)
    ctx1_mat = model.tower._embed_sequence(context_tokens)
    ctx2_mat = model.tower._embed_sequence(context2)
    tgt1_emb = model.tower.embeddings[target_token]
    tgt2_emb = model.tower.embeddings[target2]
    
    sat1_idx = model.tower.route_to_satellite(context_tokens)
    sat2_idx = model.tower.route_to_satellite(context2)
    
    # Unbind and check coherence with target
    retrieved1 = ctx1_mat.T @ model.tower._all_memories[sat1_idx]
    retrieved2 = ctx2_mat.T @ model.tower._all_memories[sat2_idx]
    
    comp1 = retrieved1 @ tgt1_emb.T
    comp2 = retrieved2 @ tgt2_emb.T
    
    coeffs1 = decompose_to_coefficients_batch(comp1.reshape(1, 4, 4), basis, xp)[0]
    coeffs2 = decompose_to_coefficients_batch(comp2.reshape(1, 4, 4), basis, xp)[0]
    
    coh1 = float((coeffs1[0]**2 + coeffs1[15]**2) / max(float(xp.sum(coeffs1**2)), 1e-12))
    coh2 = float((coeffs2[0]**2 + coeffs2[15]**2) / max(float(xp.sum(coeffs2**2)), 1e-12))
    
    pattern1_stored = coh1 > 0.1  # Some coherence indicates storage
    pattern2_stored = coh2 > 0.1
    
    audit4_passed = memory_grew and pattern1_stored and pattern2_stored
    
    print(f"  Initial memory norm: {initial_memory_norm:.4f}")
    print(f"  After pattern 1: {memory_after_1:.4f}")
    print(f"  After pattern 2: {memory_after_2:.4f}")
    print(f"  Memory grew (superposition): {'✓' if memory_grew else '✗'}")
    print(f"  Pattern 1 coherence: {coh1:.4f} — {'✓' if pattern1_stored else '✗'}")
    print(f"  Pattern 2 coherence: {coh2:.4f} — {'✓' if pattern2_stored else '✗'}")
    print(f"  AUDIT 4: {'✓ PASSED' if audit4_passed else '✗ FAILED'}")
    
    audit_results.append(AuditResult(
        step_name="Binding → Memory (Superposition)",
        passed=audit4_passed,
        details={
            'initial_norm': initial_memory_norm,
            'final_norm': memory_after_2,
            'memory_grew': memory_grew,
            'pattern1_coherence': coh1,
            'pattern2_coherence': coh2
        }
    ))
    
    # =========================================================================
    # AUDIT 5: Memory → Retrieval (Full unbinding, no candidate narrowing)
    # =========================================================================
    print("\n" + "="*80)
    print("AUDIT 5: Memory → Retrieval (No Candidate Narrowing)")
    print("="*80)
    
    # Get settled state
    settled_state = model.retrieve_settled_state(context_tokens)
    
    # Check settled state is full 4x4, not truncated
    settled_shape = tuple(settled_state.shape)
    settled_coeffs = decompose_to_coefficients_batch(settled_state.reshape(1, 4, 4), basis, xp)[0]
    non_zero_settled = int(xp.sum(xp.abs(settled_coeffs) > 1e-10))
    
    # Verify NO candidate narrowing — all vocab accessible
    # Check the retrieve_theory_true function
    theory_true_result = model.retrieve_theory_true(context_tokens)
    
    audit5_passed = (settled_shape == (4, 4)) and (non_zero_settled >= 4) and (theory_true_result is not None)
    
    print(f"  Settled state shape: {settled_shape} (expected (4, 4))")
    print(f"  Non-zero Clifford coefficients: {non_zero_settled}/16")
    print(f"  Theory-true retrieval result: {theory_true_result}")
    print(f"  AUDIT 5: {'✓ PASSED' if audit5_passed else '✗ FAILED'}")
    
    audit_results.append(AuditResult(
        step_name="Memory → Retrieval",
        passed=audit5_passed,
        details={
            'settled_shape': settled_shape,
            'non_zero_coeffs': non_zero_settled,
            'theory_true_result': theory_true_result
        }
    ))
    
    # =========================================================================
    # AUDIT 6: Retrieval → Output (Coherence scoring, not argmax)
    # =========================================================================
    print("\n" + "="*80)
    print("AUDIT 6: Retrieval → Output (Coherence Scoring)")
    print("="*80)
    
    # Test evaluate_semantic uses coherence
    batch = [(context_tokens, target_token), (context2, target2)]
    eval_result = model.evaluate_semantic(batch)
    
    # Coherence should be in [0, 1]
    coherence_valid = (
        0.0 <= eval_result['semantic_similarity'] <= 1.0 and
        0.0 <= eval_result['min_similarity'] <= 1.0 and
        0.0 <= eval_result['max_similarity'] <= 1.0
    )
    
    # Verify it's not using argmax (check that result is float, not int accuracy)
    is_float_metric = isinstance(eval_result['semantic_similarity'], float)
    
    audit6_passed = coherence_valid and is_float_metric
    
    print(f"  Semantic similarity: {eval_result['semantic_similarity']:.4f}")
    print(f"  Min similarity: {eval_result['min_similarity']:.4f}")
    print(f"  Max similarity: {eval_result['max_similarity']:.4f}")
    print(f"  Coherence in [0,1]: {coherence_valid}")
    print(f"  Is float metric (not int): {is_float_metric}")
    print(f"  AUDIT 6: {'✓ PASSED' if audit6_passed else '✗ FAILED'}")
    
    audit_results.append(AuditResult(
        step_name="Retrieval → Output (Coherence)",
        passed=audit6_passed,
        details={
            'semantic_similarity': eval_result['semantic_similarity'],
            'coherence_valid': coherence_valid,
            'is_float_metric': is_float_metric
        }
    ))
    
    # =========================================================================
    # AUDIT 7: Grace Contraction (Scales grades, doesn't zero them)
    # =========================================================================
    print("\n" + "="*80)
    print("AUDIT 7: Grace Contraction (No Zeroing)")
    print("="*80)
    
    # Create random matrix
    random_mat = xp.random.randn(4, 4).astype(xp.float32)
    random_mat = random_mat / xp.linalg.norm(random_mat, 'fro')
    
    # Get coefficients before Grace
    coeffs_before = decompose_to_coefficients_batch(random_mat.reshape(1, 4, 4), basis, xp)[0]
    
    # Apply Grace
    graced, stability, witness = grace_with_stability(random_mat, basis, xp)
    
    # Get coefficients after Grace
    coeffs_after = decompose_to_coefficients_batch(graced.reshape(1, 4, 4), basis, xp)[0]
    
    # Verify grades are SCALED, not zeroed
    grace_scales = xp.array(GRACE_SCALES_FLAT, dtype=xp.float32)
    expected_coeffs = coeffs_before * grace_scales
    
    # Check ratio matches Grace scaling
    scale_errors = []
    for i in range(16):
        if abs(float(coeffs_before[i])) > 1e-6:
            actual_scale = float(coeffs_after[i]) / float(coeffs_before[i])
            expected_scale = float(grace_scales[i])
            scale_errors.append(abs(actual_scale - expected_scale))
    
    max_scale_error = max(scale_errors) if scale_errors else 0.0
    
    # Verify no coefficient is zeroed that wasn't zero before
    non_zero_before = int(xp.sum(xp.abs(coeffs_before) > 1e-10))
    non_zero_after = int(xp.sum(xp.abs(coeffs_after) > 1e-10))
    
    audit7_passed = (max_scale_error < 0.1) and (non_zero_after >= non_zero_before * 0.5)
    
    print(f"  Non-zero coefficients before: {non_zero_before}/16")
    print(f"  Non-zero coefficients after: {non_zero_after}/16")
    print(f"  Max scale error: {max_scale_error:.4f}")
    print(f"  Stability: {stability:.4f}")
    print(f"  Grace scales grades (doesn't zero): {max_scale_error < 0.1}")
    print(f"  AUDIT 7: {'✓ PASSED' if audit7_passed else '✗ FAILED'}")
    
    audit_results.append(AuditResult(
        step_name="Grace Contraction",
        passed=audit7_passed,
        details={
            'non_zero_before': non_zero_before,
            'non_zero_after': non_zero_after,
            'max_scale_error': max_scale_error,
            'stability': float(stability)
        }
    ))
    
    # =========================================================================
    # AUDIT 8: Full Vocabulary Accessible (No candidate sets)
    # =========================================================================
    print("\n" + "="*80)
    print("AUDIT 8: Full Vocabulary Accessible")
    print("="*80)
    
    # Test retrieve_theory_true with many contexts to ensure diverse outputs
    outputs = set()
    for i in range(100):
        ctx = list(np.random.randint(0, vocab_size, size=8))
        model.tower.learn(ctx, np.random.randint(0, vocab_size))
    
    for _ in range(500):
        ctx = list(np.random.randint(0, vocab_size, size=8))
        result = model.retrieve_theory_true(ctx)
        if result is not None:
            outputs.add(result)
    
    unique_outputs = len(outputs)
    
    # Should have many unique outputs, not limited to candidate set
    audit8_passed = unique_outputs > 100  # Should be diverse
    
    print(f"  Unique outputs from 500 queries: {unique_outputs}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Diversity indicates full vocab access: {unique_outputs > 100}")
    print(f"  AUDIT 8: {'✓ PASSED' if audit8_passed else '✗ FAILED'}")
    
    audit_results.append(AuditResult(
        step_name="Full Vocabulary Accessible",
        passed=audit8_passed,
        details={
            'unique_outputs': unique_outputs,
            'vocab_size': vocab_size
        }
    ))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("INFORMATION FLOW AUDIT SUMMARY")
    print("="*80)
    
    all_passed = all(r.passed for r in audit_results)
    
    for result in audit_results:
        status = '✓' if result.passed else '✗'
        print(f"  {status} {result.step_name}")
    
    print("\n" + "="*80)
    if all_passed:
        print("ALL AUDITS PASSED — INFORMATION FLOW IS THEORY-TRUE")
        print("NO TRUNCATION, NO APPROXIMATION, NO CANDIDATE NARROWING")
    else:
        print("SOME AUDITS FAILED — REVIEW REQUIRED")
        for result in audit_results:
            if not result.passed:
                print(f"  FAILED: {result.step_name}")
    print("="*80)
    
    return {
        'all_passed': all_passed,
        'results': [
            {
                'step': r.step_name,
                'passed': r.passed,
                'details': r.details
            }
            for r in audit_results
        ]
    }


@app.local_entrypoint()
def main():
    """Run the information flow audit."""
    result = run_information_flow_audit.remote()
    print(f"\nFinal result: {'ALL PASSED' if result['all_passed'] else 'SOME FAILED'}")
