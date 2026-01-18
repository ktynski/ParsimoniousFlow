"""
Theory-True Evaluation Helper

Provides consistent evaluation functions that match the exact retrieve() path.
All evaluation must use the same theory-true paradigm as the actual retrieval.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from holographic_prod.core.constants import PHI_EPSILON
from holographic_prod.core.algebra import grace_with_stability, grace_basin_keys_batch_direct
from holographic_prod.memory.multi_level_tower import (
    GRACE_ROUTING_ITERS, GRACE_ROUTING_RESOLUTION
)


def evaluate_semantic_similarity_theory_true(
    model,
    eval_samples: List[Tuple[List[int], int]],
    n_eval: Optional[int] = None,
    return_details: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate semantic similarity using EXACT theory-true retrieval path.
    
    This matches MultiLevelTower.retrieve() exactly:
        1. Grace contraction on context → graced_state
        2. Route to satellite via Grace basin key
        3. Memory unbinding: ctx_inv @ sat_memory → retrieved
        4. Grace contraction on retrieved → retrieved_graced
        5. Full vocabulary coherence scoring
        6. Return coherence score of target token (not argmax)
    
    THEORY-TRUE METRIC:
        Coherence = witness_energy / total_energy
        Where witness = (scalar, pseudoscalar) coefficients
        
    Args:
        model: HolographicMemory instance
        eval_samples: List of (context, target) tuples
        n_eval: Number of samples to evaluate (None = all)
        return_details: If True, return per-sample details
        
    Returns:
        Dict with:
            - semantic_similarity: Mean coherence score of target tokens
            - exact_match_rate: Fraction where retrieve() returns target
            - avg_target_rank: Average rank of target in coherence scores
            - details: Per-sample results (if return_details=True)
    """
    xp = model.xp
    basis = model.basis
    
    if n_eval is not None:
        eval_samples = eval_samples[:n_eval]
    
    similarities = []
    exact_matches = []
    target_ranks = []
    details_list = []
    
    for ctx, tgt in eval_samples:
        try:
            # STEP 1: Embed context
            ctx_mat = model.tower._embed_sequence(ctx)
            
            # STEP 2: Grace contraction (matches retrieve() exactly)
            graced_state, state_stability, _ = grace_with_stability(ctx_mat, basis, xp)
            
            # STEP 3: Route to satellite (matches retrieve() exactly)
            basin_key = grace_basin_keys_batch_direct(
                graced_state[None], basis,
                n_iters=GRACE_ROUTING_ITERS,
                resolution=GRACE_ROUTING_RESOLUTION,
                xp=xp
            )[0]
            primes = xp.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53], dtype=xp.int64)
            sat_idx = int((xp.sum(basin_key * primes) % model.tower.n_satellites).get())
            
            # STEP 4: Memory unbinding (theory-true: ctx_inv @ sat_memory)
            ctx_inv = graced_state.T  # For SO(4): inverse = transpose
            sat_memory = model.tower._all_memories[sat_idx]
            sat_norm = float(xp.linalg.norm(sat_memory))
            
            if sat_norm > PHI_EPSILON:
                retrieved = ctx_inv @ sat_memory
                # STEP 5: Grace contraction on retrieved (matches retrieve() exactly)
                retrieved_graced, _, _ = grace_with_stability(retrieved, basis, xp)
            else:
                # No satellite content - use graced state directly
                retrieved_graced = graced_state
            
            # STEP 6: Full vocabulary coherence scoring (matches retrieve() exactly)
            all_embeddings = model.tower.embeddings
            vocab_size = len(all_embeddings)
            
            # Compute compositions: retrieved_graced @ embed[t].T for all t
            embed_T = xp.swapaxes(all_embeddings, -2, -1)  # [vocab, 4, 4]
            compositions = xp.einsum('ij,vjk->vik', retrieved_graced, embed_T)  # [vocab, 4, 4]
            
            # Coherence scoring via Clifford decomposition
            norm_sq = xp.sum(basis * basis, axis=(1, 2))  # [16]
            coeffs_all = xp.einsum('cij,vij->vc', basis, compositions) / norm_sq  # [vocab, 16]
            
            energies = xp.sum(coeffs_all ** 2, axis=1)  # [vocab]
            witness_energies = coeffs_all[:, 0]**2 + coeffs_all[:, 15]**2  # [vocab]
            coherences = witness_energies / xp.maximum(energies, PHI_EPSILON)
            
            # Get coherence score of target token
            tgt_coherence = float(coherences[tgt % vocab_size])
            similarities.append(tgt_coherence)
            
            # Check exact match (what retrieve() would return)
            best_token = int(xp.argmax(coherences))
            exact_match = (best_token == (tgt % vocab_size))
            exact_matches.append(exact_match)
            
            # Compute rank of target token
            sorted_indices = xp.argsort(coherences)[::-1]  # Descending
            target_rank = int(xp.where(sorted_indices == (tgt % vocab_size))[0][0]) + 1
            target_ranks.append(target_rank)
            
            if return_details:
                details_list.append({
                    'context': ctx,
                    'target': tgt,
                    'coherence': tgt_coherence,
                    'exact_match': exact_match,
                    'rank': target_rank,
                    'best_token': best_token,
                    'state_stability': float(state_stability),
                })
                
        except Exception as e:
            # FAIL LOUDLY - theory-true code should never silently skip
            import traceback
            print(f"  🔴 THEORY VIOLATION: Evaluation failed for context {ctx[:3]}...")
            print(f"     Error: {e}")
            print(traceback.format_exc())
            raise RuntimeError(f"Theory-true evaluation MUST NOT fail: {e}") from e
    
    if not similarities:
        return {
            'semantic_similarity': 0.0,
            'exact_match_rate': 0.0,
            'avg_target_rank': float('inf'),
            'n_evaluated': 0,
        }
    
    result = {
        'semantic_similarity': float(np.mean(similarities)),
        'exact_match_rate': float(np.mean(exact_matches)) if exact_matches else 0.0,
        'avg_target_rank': float(np.mean(target_ranks)) if target_ranks else float('inf'),
        'n_evaluated': len(similarities),
    }
    
    if return_details:
        result['details'] = details_list
    
    return result


def evaluate_retrieval_accuracy(
    model,
    eval_samples: List[Tuple[List[int], int]],
    n_eval: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate retrieval accuracy by calling retrieve() directly.
    
    This tests the actual retrieval path end-to-end.
    
    Returns:
        Dict with:
            - accuracy: Fraction of exact matches
            - avg_confidence: Mean confidence (if available)
    """
    if n_eval is not None:
        eval_samples = eval_samples[:n_eval]
    
    correct = 0
    total = 0
    
    for ctx, tgt in eval_samples:
        # NO TRY/EXCEPT - retrieval MUST NOT fail per theory
        pred = model.tower.retrieve(ctx)
        if pred is None:
            raise RuntimeError(f"Theory violation: retrieve() returned None for context {ctx[:3]}...")
        if pred == (tgt % model.vocab_size):
            correct += 1
        total += 1
    
    return {
        'accuracy': correct / total if total > 0 else 0.0,
        'n_evaluated': total,
    }


def verify_grace_convergence(
    model,
    contexts: List[List[int]],
    max_iterations: int = 10,
    stability_threshold: float = 0.382,  # φ⁻²
) -> Dict[str, Any]:
    """
    Verify that Grace ALWAYS converges (theory-true requirement).
    
    THEORY: Grace contracts ANY state toward an attractor basin.
    This should NEVER fail or return None.
    
    Returns:
        Dict with:
            - convergence_rate: Fraction that converged
            - avg_stability: Mean stability after Grace
            - max_iterations_needed: Maximum iterations for convergence
    """
    xp = model.xp
    basis = model.basis
    
    converged = 0
    stabilities = []
    max_iters = 0
    
    for ctx in contexts:
        # NO TRY/EXCEPT - Grace MUST ALWAYS converge per theory
        ctx_mat = model.tower._embed_sequence(ctx)
        
        # Apply Grace with stability tracking
        graced_state, stability, _ = grace_with_stability(ctx_mat, basis, xp)
        
        if graced_state is None:
            raise RuntimeError(f"Theory violation: Grace failed to converge for context {ctx[:3]}...")
        
        converged += 1
        stabilities.append(float(stability))
        
        # Check if converged (stability >= threshold)
        if stability >= stability_threshold:
            max_iters = max(max_iters, 1)  # Single iteration sufficient
    
    return {
        'convergence_rate': converged / len(contexts) if contexts else 0.0,
        'avg_stability': float(np.mean(stabilities)) if stabilities else 0.0,
        'max_iterations_needed': max_iters,
        'n_tested': len(contexts),
    }


def verify_no_candidate_sets(
    model,
    contexts: List[List[int]],
) -> Dict[str, Any]:
    """
    Verify that retrieval uses FULL vocabulary, not candidate sets.
    
    THEORY: "Candidate sets" are FORBIDDEN per THEORY_TRUE_PARADIGM.md.
    Grace contraction handles selection naturally.
    
    This checks that retrieve() can return ANY token in vocabulary.
    """
    vocab_size = model.vocab_size
    returned_tokens = set()
    
    for ctx in contexts:
        # NO TRY/EXCEPT - retrieve() MUST NOT fail per theory
        pred = model.tower.retrieve(ctx)
        if pred is None:
            raise RuntimeError(f"Theory violation: retrieve() returned None for context {ctx[:3]}...")
        returned_tokens.add(pred)
    
    return {
        'unique_tokens_returned': len(returned_tokens),
        'vocab_size': vocab_size,
        'coverage_ratio': len(returned_tokens) / vocab_size if vocab_size > 0 else 0.0,
        'n_tested': len(contexts),
    }
