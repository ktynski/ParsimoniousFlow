"""
Generation Fix Proposal — Comprehensive Diagnosis and Solution
==============================================================

DIAGNOSIS CONFIRMED:
    1. Training retrieval uses: retrieve_parallel() → tower._score_with_polarized_lensing() ✅
    2. Sample generation uses: generate_attractor_flow() → vorticity_weighted_scores() ❌
    
    The mode collapse ("hallway hallway hallway") occurs because generation
    bypasses all the polarized lensing we implemented for training.

HOWEVER:
    Tests show that even with polarized lensing, collapse is only reduced from
    90% to 80%. This indicates a deeper issue with Grace attractor dynamics:
    
    - Grace operator contracts states to stable subspaces
    - These subspaces have ~100 stable attractors (regardless of lensing)
    - argmax selection always picks the same attractor
    
SOLUTION (Multi-pronged):
    1. ADD POLARIZED LENSING to generation scoring (10-20% improvement)
    2. ADD INHIBITION OF RETURN (prevent exact repeats)
    3. CONSIDER φ-KERNEL SAMPLING (stochastic, not deterministic argmax)

This test file validates each component of the fix.

NO MOCKS. NO FALLBACKS. NO FAKE DATA.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/root/project' if '/root/project' not in sys.path else '')


class TestProposedFix:
    """Test the proposed fix components."""
    
    def test_generation_with_full_fix(self):
        """
        Test generation with ALL fixes applied:
        1. Polarized lensing for scoring
        2. Inhibition of return (penalize recent tokens)
        3. φ-kernel probabilistic sampling
        """
        from holographic_prod.core.lensing import PolarizedLensSet
        from holographic_prod.core.algebra import build_clifford_basis, grace_operator
        from holographic_prod.core.constants import PHI_INV
        from scipy.stats import ortho_group
        
        np.random.seed(42)
        basis = build_clifford_basis()
        
        # Same vocabulary
        vocab_size = 50
        embeddings = np.array([
            ortho_group.rvs(4, random_state=i).astype(np.float32)
            for i in range(vocab_size)
        ])
        
        # Create lens set
        lens_set = PolarizedLensSet(n_lenses=16, seed=42)
        
        # Same starting state
        state = ortho_group.rvs(4, random_state=1000).astype(np.float32)
        
        generated = []
        recent_tokens = []  # For inhibition of return
        inhibition_window = 3  # Penalize last N tokens
        inhibition_factor = 0.5  # Reduce score by this factor
        
        print(f"\n=== GENERATION WITH FULL FIX ===")
        print(f"  Polarized lensing: ✓")
        print(f"  Inhibition of return: ✓ (window={inhibition_window}, factor={inhibition_factor})")
        print(f"  φ-kernel sampling: ✓")
        
        for step in range(10):
            # Apply Grace
            for _ in range(3):
                state = grace_operator(state, basis, np)
            
            # Score with polarized lensing
            scores = lens_set.score_all_lenses_vectorized(state, embeddings)
            scores = np.array(scores, dtype=np.float64)  # Ensure float for manipulation
            
            # Apply inhibition of return (penalize recent tokens)
            for recent_idx in recent_tokens[-inhibition_window:]:
                scores[recent_idx] *= inhibition_factor
            
            # φ-kernel probabilistic sampling (not argmax)
            # P(token) ∝ score^(1/φ) — theory-derived temperature
            phi_inv = PHI_INV
            scores_positive = np.maximum(scores, 1e-10)
            logits = np.log(scores_positive) / phi_inv
            logits = logits - np.max(logits)  # Numerical stability
            probs = np.exp(logits)
            probs = probs / np.sum(probs)
            
            # Sample (not argmax!)
            best_idx = np.random.choice(vocab_size, p=probs)
            generated.append(best_idx)
            recent_tokens.append(best_idx)
            
            # Evolve state
            state = state @ embeddings[best_idx]
            state = state / (np.linalg.norm(state) + 1e-10) * 2.0
            
            print(f"Step {step+1}: token {best_idx}, top_prob={probs[best_idx]:.3f}")
        
        # Check collapse rate
        unique_tokens = len(set(generated))
        repetition_rate = 1 - unique_tokens / len(generated)
        
        print(f"\nGenerated: {generated}")
        print(f"Unique tokens: {unique_tokens}/{len(generated)}")
        print(f"Repetition rate: {repetition_rate*100:.1f}%")
        
        # With full fix, we expect much lower collapse
        if repetition_rate < 0.3:
            print("✓ FULL FIX eliminates mode collapse!")
        elif repetition_rate < 0.5:
            print("✓ FULL FIX significantly reduces mode collapse")
        else:
            print(f"⚠️ Still some collapse at {repetition_rate*100:.1f}%")
        
        # The key: should be much better than 80-90%
        assert repetition_rate < 0.6, f"Full fix should reduce collapse: {repetition_rate*100:.1f}%"


class TestInhibitionOfReturnAlone:
    """Test inhibition of return (IoR) as a standalone fix."""
    
    def test_ior_prevents_immediate_repeats(self):
        """
        Inhibition of Return: Penalize recently used tokens.
        
        BRAIN ANALOG:
            - IoR is a real cognitive phenomenon
            - Prevents perseveration (stuck on same response)
            - Theory: φ⁻² decay rate for inhibition
        """
        from holographic_prod.core.quotient import vorticity_weighted_scores
        from holographic_prod.core.algebra import build_clifford_basis, grace_operator
        from scipy.stats import ortho_group
        
        np.random.seed(42)
        basis = build_clifford_basis()
        vocab_size = 50
        embeddings = np.array([
            ortho_group.rvs(4, random_state=i).astype(np.float32)
            for i in range(vocab_size)
        ])
        
        state = ortho_group.rvs(4, random_state=1000).astype(np.float32)
        
        generated = []
        recent_tokens = []
        inhibition_window = 3
        
        print(f"\n=== INHIBITION OF RETURN TEST (No Lensing) ===")
        
        for step in range(10):
            for _ in range(3):
                state = grace_operator(state, basis, np)
            
            # Raw scoring (no lensing)
            scores = vorticity_weighted_scores(state, embeddings, basis, np)
            scores = np.array(scores, dtype=np.float64)
            
            # Apply IoR
            for recent_idx in recent_tokens[-inhibition_window:]:
                scores[recent_idx] *= 0.1  # Strong inhibition
            
            best_idx = np.argmax(scores)
            generated.append(best_idx)
            recent_tokens.append(best_idx)
            
            state = state @ embeddings[best_idx]
            state = state / (np.linalg.norm(state) + 1e-10) * 2.0
            
            print(f"Step {step+1}: token {best_idx}")
        
        unique_tokens = len(set(generated))
        repetition_rate = 1 - unique_tokens / len(generated)
        
        print(f"\nGenerated: {generated}")
        print(f"Unique tokens: {unique_tokens}/{len(generated)}")
        print(f"Repetition rate: {repetition_rate*100:.1f}%")
        
        # Check for no immediate repeats (due to IoR)
        immediate_repeats = sum(1 for i in range(1, len(generated)) 
                               if generated[i] == generated[i-1])
        
        print(f"Immediate repeats: {immediate_repeats}")
        assert immediate_repeats == 0, "IoR should prevent immediate repeats"
        print("✓ IoR prevents immediate repeats")


class TestPhiKernelSamplingAlone:
    """Test φ-kernel probabilistic sampling."""
    
    def test_phi_kernel_adds_diversity(self):
        """
        φ-kernel sampling: P(token) ∝ score^(1/φ)
        
        THEORY:
            - φ = (1+√5)/2 ≈ 1.618
            - 1/φ ≈ 0.618 — optimal "temperature"
            - Not too hot (random), not too cold (deterministic)
        """
        from holographic_prod.core.quotient import vorticity_weighted_scores
        from holographic_prod.core.algebra import build_clifford_basis, grace_operator
        from holographic_prod.core.constants import PHI_INV
        from scipy.stats import ortho_group
        
        np.random.seed(42)
        basis = build_clifford_basis()
        vocab_size = 50
        embeddings = np.array([
            ortho_group.rvs(4, random_state=i).astype(np.float32)
            for i in range(vocab_size)
        ])
        
        state = ortho_group.rvs(4, random_state=1000).astype(np.float32)
        
        # Run 5 times with same state to see diversity
        all_generated = []
        print(f"\n=== φ-KERNEL SAMPLING TEST ===")
        print(f"  Temperature = 1/φ ≈ {PHI_INV:.4f}")
        
        for trial in range(5):
            np.random.seed(42 + trial * 100)
            trial_state = state.copy()
            generated = []
            
            for step in range(5):
                for _ in range(3):
                    trial_state = grace_operator(trial_state, basis, np)
                
                scores = vorticity_weighted_scores(trial_state, embeddings, basis, np)
                scores = np.array(scores, dtype=np.float64)
                
                # φ-kernel sampling
                scores_positive = np.maximum(scores, 1e-10)
                logits = np.log(scores_positive) / PHI_INV
                logits = logits - np.max(logits)
                probs = np.exp(logits)
                probs = probs / np.sum(probs)
                
                best_idx = np.random.choice(vocab_size, p=probs)
                generated.append(best_idx)
                
                trial_state = trial_state @ embeddings[best_idx]
                trial_state = trial_state / (np.linalg.norm(trial_state) + 1e-10) * 2.0
            
            all_generated.append(generated)
            print(f"  Trial {trial+1}: {generated}")
        
        # Check diversity across trials
        first_tokens = [g[0] for g in all_generated]
        unique_first = len(set(first_tokens))
        
        print(f"\nFirst token across 5 trials: {first_tokens}")
        print(f"Unique first tokens: {unique_first}/5")
        
        # φ-kernel should give some variation
        assert unique_first >= 2, "φ-kernel should add diversity"
        print("✓ φ-kernel sampling adds diversity")


class TestComparisonTable:
    """Generate comparison table of different approaches."""
    
    def test_generate_comparison(self):
        """
        Compare collapse rates for different configurations:
        1. Baseline (raw scores, argmax)
        2. + Polarized lensing
        3. + Inhibition of return
        4. + φ-kernel sampling
        5. ALL combined
        """
        from holographic_prod.core.lensing import PolarizedLensSet
        from holographic_prod.core.quotient import vorticity_weighted_scores
        from holographic_prod.core.algebra import build_clifford_basis, grace_operator
        from holographic_prod.core.constants import PHI_INV
        from scipy.stats import ortho_group
        
        np.random.seed(42)
        basis = build_clifford_basis()
        vocab_size = 50
        embeddings = np.array([
            ortho_group.rvs(4, random_state=i).astype(np.float32)
            for i in range(vocab_size)
        ])
        lens_set = PolarizedLensSet(n_lenses=16, seed=42)
        
        def run_generation(use_lensing, use_ior, use_phi_kernel, n_steps=10):
            """Run generation with specified configuration."""
            state = ortho_group.rvs(4, random_state=1000).astype(np.float32)
            generated = []
            recent_tokens = []
            
            for step in range(n_steps):
                for _ in range(3):
                    state = grace_operator(state, basis, np)
                
                # Scoring
                if use_lensing:
                    scores = lens_set.score_all_lenses_vectorized(state, embeddings)
                else:
                    scores = vorticity_weighted_scores(state, embeddings, basis, np)
                scores = np.array(scores, dtype=np.float64)
                
                # IoR
                if use_ior:
                    for recent_idx in recent_tokens[-3:]:
                        scores[recent_idx] *= 0.1
                
                # Selection
                if use_phi_kernel:
                    scores_positive = np.maximum(scores, 1e-10)
                    logits = np.log(scores_positive) / PHI_INV
                    logits = logits - np.max(logits)
                    probs = np.exp(logits)
                    probs = probs / np.sum(probs)
                    best_idx = np.random.choice(vocab_size, p=probs)
                else:
                    best_idx = np.argmax(scores)
                
                generated.append(best_idx)
                recent_tokens.append(best_idx)
                
                state = state @ embeddings[best_idx]
                state = state / (np.linalg.norm(state) + 1e-10) * 2.0
            
            return generated
        
        # Run all configurations
        configs = [
            ("Baseline (current)", False, False, False),
            ("+ Polarized Lensing", True, False, False),
            ("+ IoR only", False, True, False),
            ("+ φ-kernel only", False, False, True),
            ("+ Lensing + IoR", True, True, False),
            ("+ Lensing + φ-kernel", True, False, True),
            ("+ IoR + φ-kernel", False, True, True),
            ("FULL FIX (all)", True, True, True),
        ]
        
        print(f"\n{'='*70}")
        print("CONFIGURATION COMPARISON — MODE COLLAPSE RATES")
        print(f"{'='*70}")
        print(f"{'Configuration':<25} | {'Unique':>8} | {'Collapse':>10} | Sample")
        print("-" * 70)
        
        np.random.seed(42)  # Reset for consistency
        
        for name, lensing, ior, phi in configs:
            generated = run_generation(lensing, ior, phi)
            unique = len(set(generated))
            collapse = 1 - unique / len(generated)
            sample_str = ",".join(str(g) for g in generated[:6]) + "..."
            print(f"{name:<25} | {unique:>8} | {collapse*100:>9.1f}% | {sample_str}")
        
        print(f"{'='*70}")
        print("\nRECOMMENDATION: Implement FULL FIX (Lensing + IoR + φ-kernel)")
        print(f"{'='*70}")
        
        assert True  # Diagnostic test


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
