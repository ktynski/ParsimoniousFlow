"""
holographic_exploration_tests.py — TDD Exploration of Proposed Enhancements
============================================================================

This module tests PROPOSED enhancements against three criteria:
1. Theory-true: Derivable from φ/Clifford/Grace
2. Informationally parsimonious: No arbitrary parameters
3. Transformer-killer: Improves O(1), generalization, or capacity

EVALUATION RESULTS FROM ANALYSIS:

ALREADY IMPLEMENTED (verify working):
- Cleanup memory → Grace basin discovery
- Role/filler binding → Attribute binding module  
- Novelty-gated writes → Meta-cognitive loop v4.7.0
- Dream mixing → REM recombination in dreaming.py

THEORY-TRUE & WORTH EXPLORING:
1. Witness-Based Banks (sparse addressing via witness quantization)
2. Multi-Timescale Buffers (φ^k decay rates)
3. Witness Entropy H_w (capacity signal)
4. Iterative Unbinding (multi-item retrieval)

OFF-THEORY (do not implement):
- Context whitening (arbitrary covariance)
- Random phase/sign (arbitrary randomness)
- Hopfield temperature (softmax forbidden)
- ECC contexts (not geometric)

Run with: python -m holographic_v4.holographic_exploration_tests
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, MATRIX_DIM, GRADE_INDICES
from holographic_v4.algebra import (
    build_clifford_basis,
    geometric_product,
    grace_operator,
    decompose_to_coefficients,
    initialize_embeddings_identity,
)
from holographic_v4.quotient import extract_witness, grace_stability
from holographic_v4.holographic_memory import (
    HolographicMemory,
    WitnessIndex,
    HybridHolographicMemory,
    clifford_inverse,
)


# =============================================================================
# WITNESS ENTROPY — Theory-True Capacity Signal
# =============================================================================

def compute_witness_entropy(M: np.ndarray, basis: np.ndarray) -> float:
    """
    Compute witness entropy H_w — the theory-true capacity overload signal.
    
    THEORY:
        H_w = -Σ p_k log(p_k)
        
        where p_k = |grade_k|² / Σ|grade_j|²
        
    INTERPRETATION:
        - Low H_w → one or two grades dominate → clean recall
        - High H_w → energy smeared across grades → interference overload
        
    WHY THIS IS THEORY-TRUE:
        - Uses grade structure (not arbitrary)
        - Measures what Grace cares about (grade distribution)
        - No arbitrary parameters
        
    Args:
        M: [4, 4] matrix
        basis: [16, 4, 4] Clifford basis
        
    Returns:
        Witness entropy H_w (bits)
    """
    coeffs = decompose_to_coefficients(M, basis, np)
    
    # Compute energy per grade
    grade_energies = {}
    for grade, indices in GRADE_INDICES.items():
        energy = sum(coeffs[i]**2 for i in indices)
        grade_energies[grade] = energy
    
    total_energy = sum(grade_energies.values())
    if total_energy < 1e-12:
        return 0.0
    
    # Compute probabilities
    probs = [e / total_energy for e in grade_energies.values() if e > 1e-12]
    
    # Entropy
    entropy = -sum(p * np.log2(p + 1e-12) for p in probs)
    
    return entropy


def test_witness_entropy_as_capacity_signal():
    """
    Test that witness entropy increases with interference (more stored items).
    
    HYPOTHESIS:
        H_w should increase as holographic memory fills up, providing
        a theory-derived "memory full" signal.
    """
    print("\n=== Test: Witness Entropy as Capacity Signal ===\n")
    
    basis = build_clifford_basis(np)
    
    pattern_counts = [1, 2, 4, 8, 16, 32]
    entropies = []
    
    for n_patterns in pattern_counts:
        memory = HolographicMemory.create(basis)
        
        # Store patterns
        for i in range(n_patterns):
            np.random.seed(i * 100)
            ctx = np.eye(4) + 0.3 * np.random.randn(4, 4)
            tgt = np.eye(4) + 0.3 * np.random.randn(4, 4)
            memory.store(ctx, tgt)
        
        # Retrieve a pattern and measure entropy
        np.random.seed(0)
        query = np.eye(4) + 0.3 * np.random.randn(4, 4)
        retrieved, conf = memory.retrieve(query)
        
        H_w = compute_witness_entropy(retrieved, basis)
        entropies.append(H_w)
        
        print(f"  n={n_patterns:2d}: H_w={H_w:.4f}, conf={conf:.4f}")
    
    # Entropy should generally increase with more patterns
    trend_positive = entropies[-1] > entropies[0]
    
    print(f"\n  Entropy trend: {entropies[0]:.4f} → {entropies[-1]:.4f}")
    print(f"  Trend positive: {trend_positive}")
    
    success = True  # We're exploring, not asserting
    print(f"\n  EXPLORATION RESULT: H_w {'INCREASES' if trend_positive else 'DOES NOT INCREASE'} with capacity")
    
    return trend_positive, entropies


# =============================================================================
# WITNESS-BASED BANKS — Theory-True Sparse Addressing
# =============================================================================

@dataclass
class WitnessBankedMemory:
    """
    Theory-true sparse addressing via witness-based bank selection.
    
    THEORY:
        Instead of hash-based bank selection (off-theory), select banks
        based on WITNESS REGION. This respects that witness IS attractor
        identity.
        
    WHY DIFFERENT FROM SINGLE WITNESS INDEX:
        Each bank is an INDEPENDENT holographic memory, not just a bucket.
        This gives true sparse superposition (K-of-B addressing).
        
    CAPACITY SCALING:
        With B banks and K selected per write:
        - Effective capacity ≈ B × single_bank_capacity
        - Interference per bank ≈ N/B instead of N
    """
    banks: List[HolographicMemory]
    basis: np.ndarray
    n_banks: int
    k_select: int  # How many banks to write to
    bank_grid_size: float = PHI_INV  # Grid size for bank selection
    
    @classmethod
    def create(cls, basis: np.ndarray, n_banks: int = 4, k_select: int = 2) -> 'WitnessBankedMemory':
        """Create banked memory with n_banks independent holographic stores."""
        banks = [HolographicMemory.create(basis) for _ in range(n_banks)]
        return cls(banks=banks, basis=basis, n_banks=n_banks, k_select=k_select)
    
    def _select_banks(self, M: np.ndarray) -> List[int]:
        """
        Select K banks based on witness.
        
        THEORY-TRUE: Uses witness (what survives Grace) for addressing.
        """
        s, p = extract_witness(M, self.basis, np)
        
        # Map witness to bank indices via φ-derived grid
        # Use both scalar and pseudo to spread across banks
        s_idx = int(np.floor(s / self.bank_grid_size)) % self.n_banks
        p_idx = int(np.floor(p / self.bank_grid_size)) % self.n_banks
        
        # Select K unique banks
        selected = set()
        selected.add(s_idx)
        selected.add(p_idx)
        
        # If need more, add neighbors
        while len(selected) < self.k_select:
            selected.add((s_idx + len(selected)) % self.n_banks)
        
        return list(selected)[:self.k_select]
    
    def store(self, context: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """Store in K selected banks."""
        bank_indices = self._select_banks(context)
        
        for idx in bank_indices:
            self.banks[idx].store(context, target)
        
        return {'banks_used': bank_indices}
    
    def retrieve(self, context: np.ndarray) -> Tuple[np.ndarray, float]:
        """Retrieve from selected banks and combine."""
        bank_indices = self._select_banks(context)
        
        results = []
        confidences = []
        
        for idx in bank_indices:
            result, conf = self.banks[idx].retrieve(context)
            results.append(result)
            confidences.append(conf)
        
        # Combine results (φ-weighted by confidence)
        total_weight = 0.0
        combined = np.zeros((MATRIX_DIM, MATRIX_DIM))
        
        for result, conf in zip(results, confidences):
            weight = PHI ** conf  # φ-weighted
            combined += weight * result
            total_weight += weight
        
        if total_weight > 0:
            combined /= total_weight
        
        avg_conf = np.mean(confidences)
        return combined, avg_conf


def test_witness_banked_capacity():
    """
    Test if witness-based banks improve capacity over single memory.
    
    HYPOTHESIS:
        With B banks and K-of-B selection, effective capacity should
        scale better than single memory because interference is distributed.
    """
    print("\n=== Test: Witness-Based Banks vs Single Memory ===\n")
    
    basis = build_clifford_basis(np)
    
    n_patterns = 20
    
    # Single memory
    single = HolographicMemory.create(basis)
    
    # Banked memory (4 banks, select 2)
    banked = WitnessBankedMemory.create(basis, n_banks=4, k_select=2)
    
    # Store same patterns in both
    contexts = []
    targets = []
    
    for i in range(n_patterns):
        np.random.seed(i * 100)
        ctx = np.eye(4) + 0.4 * np.random.randn(4, 4)
        tgt = np.eye(4) + 0.4 * np.random.randn(4, 4)
        contexts.append(ctx)
        targets.append(tgt)
        
        single.store(ctx, tgt)
        banked.store(ctx, tgt)
    
    # Compare retrieval quality
    single_correct = 0
    banked_correct = 0
    
    for i in range(n_patterns):
        # Single memory
        single_result, _ = single.retrieve(contexts[i])
        single_sims = [np.sum(single_result * t) / (np.linalg.norm(single_result) * np.linalg.norm(t) + 1e-8) 
                       for t in targets]
        if np.argmax(single_sims) == i:
            single_correct += 1
        
        # Banked memory
        banked_result, _ = banked.retrieve(contexts[i])
        banked_sims = [np.sum(banked_result * t) / (np.linalg.norm(banked_result) * np.linalg.norm(t) + 1e-8) 
                       for t in targets]
        if np.argmax(banked_sims) == i:
            banked_correct += 1
    
    single_acc = single_correct / n_patterns
    banked_acc = banked_correct / n_patterns
    
    print(f"  Single memory accuracy: {single_correct}/{n_patterns} ({single_acc*100:.0f}%)")
    print(f"  Banked memory accuracy: {banked_correct}/{n_patterns} ({banked_acc*100:.0f}%)")
    print(f"  Improvement: {(banked_acc - single_acc)*100:+.1f}%")
    
    improvement = banked_acc > single_acc
    print(f"\n  EXPLORATION RESULT: Banks {'HELP' if improvement else 'DO NOT HELP'}")
    
    return improvement, single_acc, banked_acc


# =============================================================================
# MULTI-TIMESCALE BUFFERS — φ-Parameterized Decay
# =============================================================================

@dataclass
class MultiTimescaleMemory:
    """
    Multi-timescale holographic buffers with φ-parameterized decay.
    
    THEORY:
        Different buffers decay at different rates:
        - Fast: φ⁻¹ decay per access (working memory)
        - Medium: φ⁻² decay (episodic buffer)
        - Slow: φ⁻³ decay (near-semantic)
        
    WHY THEORY-TRUE:
        All decay rates derived from φ, not arbitrary.
    """
    fast: HolographicMemory    # φ⁻¹ decay
    medium: HolographicMemory  # φ⁻² decay
    slow: HolographicMemory    # φ⁻³ decay
    basis: np.ndarray
    access_count: int = 0
    
    @classmethod
    def create(cls, basis: np.ndarray) -> 'MultiTimescaleMemory':
        return cls(
            fast=HolographicMemory.create(basis),
            medium=HolographicMemory.create(basis),
            slow=HolographicMemory.create(basis),
            basis=basis,
        )
    
    def store(self, context: np.ndarray, target: np.ndarray, salience: float = 0.5):
        """
        Store based on salience.
        
        High salience → all buffers
        Medium salience → medium + slow
        Low salience → slow only
        """
        # Always store in slow (long-term)
        self.slow.store(context, target, weight=PHI_INV_CUBE)
        
        if salience > PHI_INV_SQ:
            # High salience → all buffers
            self.fast.store(context, target, weight=PHI_INV)
            self.medium.store(context, target, weight=PHI_INV_SQ)
        elif salience > PHI_INV_CUBE:
            # Medium salience → medium + slow
            self.medium.store(context, target, weight=PHI_INV_SQ)
    
    def retrieve(self, context: np.ndarray) -> Tuple[np.ndarray, float, str]:
        """
        Retrieve with cascade: fast → medium → slow.
        """
        # Try fast first
        fast_result, fast_conf = self.fast.retrieve(context)
        if fast_conf > PHI_INV_SQ:
            return fast_result, fast_conf, "fast"
        
        # Try medium
        medium_result, medium_conf = self.medium.retrieve(context)
        if medium_conf > PHI_INV_CUBE:
            return medium_result, medium_conf, "medium"
        
        # Fall back to slow
        slow_result, slow_conf = self.slow.retrieve(context)
        return slow_result, slow_conf, "slow"
    
    def decay(self):
        """
        Apply φ-parameterized decay to each buffer.
        
        This simulates forgetting over time.
        """
        # Fast decays most
        self.fast.memory *= PHI_INV
        
        # Medium decays moderately
        self.medium.memory *= PHI_INV_SQ
        
        # Slow decays least
        self.slow.memory *= PHI_INV_CUBE
        
        self.access_count += 1


def test_multi_timescale_retention():
    """
    Test that multi-timescale buffers have different retention properties.
    
    HYPOTHESIS:
        - Fast buffer: rapid forgetting, recent items only
        - Slow buffer: long retention, older items preserved
    """
    print("\n=== Test: Multi-Timescale Retention ===\n")
    
    basis = build_clifford_basis(np)
    memory = MultiTimescaleMemory.create(basis)
    
    # Store patterns with different saliences
    np.random.seed(42)
    high_salience_ctx = np.eye(4) + 0.3 * np.random.randn(4, 4)
    high_salience_tgt = np.eye(4) + 0.3 * np.random.randn(4, 4)
    
    np.random.seed(123)
    low_salience_ctx = np.eye(4) + 0.3 * np.random.randn(4, 4)
    low_salience_tgt = np.eye(4) + 0.3 * np.random.randn(4, 4)
    
    memory.store(high_salience_ctx, high_salience_tgt, salience=0.8)
    memory.store(low_salience_ctx, low_salience_tgt, salience=0.2)
    
    print("  Before decay:")
    _, high_conf_before, high_source = memory.retrieve(high_salience_ctx)
    _, low_conf_before, low_source = memory.retrieve(low_salience_ctx)
    print(f"    High salience: conf={high_conf_before:.4f}, source={high_source}")
    print(f"    Low salience: conf={low_conf_before:.4f}, source={low_source}")
    
    # Apply several decay cycles
    for _ in range(10):
        memory.decay()
    
    print("\n  After 10 decay cycles:")
    _, high_conf_after, high_source = memory.retrieve(high_salience_ctx)
    _, low_conf_after, low_source = memory.retrieve(low_salience_ctx)
    print(f"    High salience: conf={high_conf_after:.4f}, source={high_source}")
    print(f"    Low salience: conf={low_conf_after:.4f}, source={low_source}")
    
    # High salience should retain better (in slow buffer)
    high_retained = high_conf_after > 0.1
    
    print(f"\n  EXPLORATION RESULT: Multi-timescale {'SHOWS DIFFERENTIAL RETENTION' if high_retained else 'NO CLEAR BENEFIT'}")
    
    return high_retained


# =============================================================================
# ITERATIVE UNBINDING — Multi-Item Retrieval
# =============================================================================

def iterative_unbind(memory: HolographicMemory, context: np.ndarray, 
                     max_items: int = 3) -> List[Tuple[np.ndarray, float]]:
    """
    Retrieve multiple items via iterative unbinding.
    
    THEORY:
        After retrieving t₁, subtract its contribution and retrieve again:
        M' = M - (c ⊗ t₁)
        t₂ ≈ c⁻¹ ⊗ M'
        
    This extracts multiple bound items from superposition.
    
    Args:
        memory: HolographicMemory instance
        context: Query context
        max_items: Maximum items to retrieve
        
    Returns:
        List of (retrieved_matrix, confidence) tuples
    """
    results = []
    current_memory = memory.memory.copy()
    
    ctx_norm = np.linalg.norm(context, 'fro')
    if ctx_norm < 1e-8:
        return results
    
    ctx_normalized = context / ctx_norm
    ctx_inverse = clifford_inverse(ctx_normalized, memory.basis, np)
    
    for i in range(max_items):
        # Retrieve from current memory state
        retrieved = geometric_product(ctx_inverse, current_memory)
        retrieved = grace_operator(retrieved, memory.basis, np)
        
        confidence = grace_stability(retrieved, memory.basis, np)
        
        # Stop if confidence too low
        if confidence < PHI_INV_CUBE:
            break
        
        results.append((retrieved, confidence))
        
        # Subtract this item's contribution
        binding = geometric_product(ctx_normalized, retrieved)
        current_memory = current_memory - PHI_INV * binding
    
    return results


def test_iterative_unbinding():
    """
    Test if iterative unbinding can retrieve multiple items.
    
    HYPOTHESIS:
        Storing multiple targets with same context, iterative unbinding
        should retrieve them in sequence.
    """
    print("\n=== Test: Iterative Unbinding for Multi-Item Retrieval ===\n")
    
    basis = build_clifford_basis(np)
    memory = HolographicMemory.create(basis)
    
    # Store multiple targets with similar contexts
    np.random.seed(42)
    base_ctx = np.eye(4) + 0.2 * np.random.randn(4, 4)
    
    targets = []
    for i in range(3):
        np.random.seed(100 + i)
        tgt = np.eye(4) + 0.3 * np.random.randn(4, 4)
        targets.append(tgt)
        
        # Slightly perturb context for each
        np.random.seed(200 + i)
        ctx_perturb = base_ctx + 0.05 * np.random.randn(4, 4)
        memory.store(ctx_perturb, tgt)
    
    print(f"  Stored {len(targets)} items with similar contexts")
    
    # Retrieve iteratively
    results = iterative_unbind(memory, base_ctx, max_items=5)
    
    print(f"  Retrieved {len(results)} items:")
    for i, (retrieved, conf) in enumerate(results):
        # Find best matching target
        sims = [np.sum(retrieved * t) / (np.linalg.norm(retrieved) * np.linalg.norm(t) + 1e-8) 
                for t in targets]
        best_match = np.argmax(sims)
        print(f"    Item {i}: conf={conf:.4f}, best_match=target_{best_match}, sim={sims[best_match]:.4f}")
    
    multi_retrieved = len(results) > 1
    print(f"\n  EXPLORATION RESULT: Iterative unbinding {'RETRIEVES MULTIPLE' if multi_retrieved else 'SINGLE ONLY'}")
    
    return multi_retrieved


# =============================================================================
# VERIFY ALREADY IMPLEMENTED FEATURES
# =============================================================================

def test_verify_novelty_gating():
    """
    Verify that meta-cognitive novelty gating is already implemented.
    """
    print("\n=== Verify: Novelty Gating (Meta-Cognitive Loop) ===\n")
    
    try:
        from holographic_v4.pipeline import TheoryTrueModel
        
        model = TheoryTrueModel(vocab_size=100)
        
        # Check for meta-learning
        has_meta = hasattr(model, 'use_meta_learning') and model.use_meta_learning
        has_predictor = hasattr(model, 'learning_state')
        
        print(f"  use_meta_learning: {has_meta}")
        print(f"  has learning_state: {has_predictor}")
        
        if has_meta:
            print("\n  ✓ VERIFIED: Novelty gating already implemented via meta-cognitive loop")
            return True
        else:
            print("\n  ⚠️ NOT FOUND: May need to verify implementation")
            return False
            
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_verify_grace_basin_cleanup():
    """
    Verify that Grace basin discovery serves as cleanup memory.
    """
    print("\n=== Verify: Grace Basin Discovery (Cleanup Memory) ===\n")
    
    try:
        from holographic_v4.resonance import grace_basin_discovery
        
        basis = build_clifford_basis(np)
        
        # Create distinct prototypes
        np.random.seed(42)
        proto_a = np.eye(4) * 1.5 + 0.2 * np.random.randn(4, 4)
        np.random.seed(123)
        proto_b = np.eye(4) * 0.5 + 0.3 * np.random.randn(4, 4)
        
        prototypes = [proto_a, proto_b]
        
        # Noisy query (should "clean up" to nearest prototype)
        np.random.seed(456)
        noisy_query = proto_a + 0.3 * np.random.randn(4, 4)
        
        best_idx, confidence, info = grace_basin_discovery(
            noisy_query, prototypes, basis, np, grace_steps=5
        )
        
        print(f"  Noisy query cleaned up to prototype {best_idx}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Info: {info}")
        
        cleanup_works = best_idx == 0  # Should clean to proto_a
        
        if cleanup_works:
            print("\n  ✓ VERIFIED: Grace basin discovery IS the cleanup mechanism")
        else:
            print("\n  ⚠️ Cleanup to wrong prototype")
        
        return cleanup_works
        
    except Exception as e:
        print(f"  Error: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def run_all_explorations():
    """Run all exploration tests."""
    print("=" * 70)
    print("HOLOGRAPHIC MEMORY — TDD EXPLORATION")
    print("Evaluating Proposed Enhancements for Theory-Truth & Utility")
    print("=" * 70)
    
    results = {}
    
    # Theory-True Proposals
    print("\n" + "=" * 70)
    print("THEORY-TRUE PROPOSALS (Worth Exploring)")
    print("=" * 70)
    
    results['witness_entropy'] = test_witness_entropy_as_capacity_signal()
    results['witness_banks'] = test_witness_banked_capacity()
    results['multi_timescale'] = test_multi_timescale_retention()
    results['iterative_unbind'] = test_iterative_unbinding()
    
    # Verify Already Implemented
    print("\n" + "=" * 70)
    print("VERIFY ALREADY IMPLEMENTED")
    print("=" * 70)
    
    results['novelty_gating'] = test_verify_novelty_gating()
    results['grace_basin_cleanup'] = test_verify_grace_basin_cleanup()
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPLORATION SUMMARY")
    print("=" * 70)
    
    print("\n  THEORY-TRUE PROPOSALS:")
    print(f"    Witness Entropy (H_w):      {'✓ USEFUL' if results['witness_entropy'][0] else '? UNCLEAR'}")
    print(f"    Witness-Based Banks:        {'✓ USEFUL' if results['witness_banks'][0] else '? UNCLEAR'}")
    print(f"    Multi-Timescale Buffers:    {'✓ USEFUL' if results['multi_timescale'] else '? UNCLEAR'}")
    print(f"    Iterative Unbinding:        {'✓ USEFUL' if results['iterative_unbind'] else '? UNCLEAR'}")
    
    print("\n  ALREADY IMPLEMENTED:")
    print(f"    Novelty Gating:             {'✓ VERIFIED' if results['novelty_gating'] else '⚠️ CHECK'}")
    print(f"    Grace Basin Cleanup:        {'✓ VERIFIED' if results['grace_basin_cleanup'] else '⚠️ CHECK'}")
    
    print("\n  OFF-THEORY (DO NOT IMPLEMENT):")
    print("    - Context Whitening (arbitrary covariance)")
    print("    - Random Phase/Sign (arbitrary randomness)")
    print("    - Hopfield Temperature (softmax forbidden)")
    print("    - ECC Contexts (not geometric)")
    
    print("\n  RECOMMENDATION:")
    useful_count = sum([
        results['witness_entropy'][0],
        results['witness_banks'][0],
        results['multi_timescale'],
        results['iterative_unbind'],
    ])
    
    if useful_count >= 3:
        print("    → Multiple theory-true enhancements show promise")
        print("    → Recommend implementing: H_w, witness banks, multi-timescale")
    elif useful_count >= 1:
        print("    → Some enhancements useful, others need more testing")
    else:
        print("    → Current implementation may be sufficient")
    
    return results


if __name__ == "__main__":
    results = run_all_explorations()
