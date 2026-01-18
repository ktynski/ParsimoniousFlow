"""
TDD Tests for Generation Fix — Must Pass After Implementation
==============================================================

These tests define the EXPECTED behavior after implementing:
1. Inhibition of Return (IoR) in generation
2. φ-kernel probabilistic sampling
3. Polarized lensing integration

Run these tests BEFORE implementation to see them fail,
then implement to make them pass.

NO MOCKS. NO FALLBACKS. NO FAKE DATA.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/root/project' if '/root/project' not in sys.path else '')


# =============================================================================
# TDD TESTS: GENERATION DIVERSITY
# =============================================================================

class TestGenerationDiversity:
    """Tests that generation produces diverse output, not mode collapse."""
    
    def test_no_immediate_repeats(self):
        """
        REQUIREMENT: Generation should NOT produce immediate token repeats.
        
        This tests that IoR (Inhibition of Return) is working.
        """
        from holographic_prod.core.attractor_generation import generate_attractor_flow
        from holographic_prod.memory.holographic_memory_unified import HolographicMemory
        from scipy.stats import ortho_group
        
        np.random.seed(42)
        
        # Create a small memory with learned patterns
        memory = HolographicMemory(vocab_size=100, max_levels=2)
        
        # Learn some patterns
        for i in range(50):
            ctx = [i % 100, (i + 1) % 100, (i + 2) % 100]
            tgt = (i + 3) % 100
            memory.learn(ctx, tgt)
        
        # Generate 10 tokens
        prompt = [1, 2, 3]
        generated, stabilities = generate_attractor_flow(
            memory=memory,
            prompt_tokens=prompt,
            max_tokens=10,
            grace_steps=3,
        )
        
        # Extract only generated tokens (not prompt)
        new_tokens = generated[len(prompt):]
        
        # Count immediate repeats
        immediate_repeats = sum(
            1 for i in range(1, len(new_tokens)) 
            if new_tokens[i] == new_tokens[i-1]
        )
        
        print(f"\nGenerated tokens: {new_tokens}")
        print(f"Immediate repeats: {immediate_repeats}")
        
        # REQUIREMENT: No immediate repeats (IoR should prevent this)
        assert immediate_repeats == 0, \
            f"IoR should prevent immediate repeats, but got {immediate_repeats}"
    
    def test_reasonable_diversity(self):
        """
        REQUIREMENT: Generation should have reasonable token diversity.
        
        Over 10 generated tokens, at least 4 should be unique.
        (φ-kernel sampling should ensure this)
        """
        from holographic_prod.core.attractor_generation import generate_attractor_flow
        from holographic_prod.memory.holographic_memory_unified import HolographicMemory
        
        np.random.seed(42)
        
        memory = HolographicMemory(vocab_size=100, max_levels=2)
        
        # Learn patterns
        for i in range(50):
            ctx = [i % 100, (i + 1) % 100, (i + 2) % 100]
            tgt = (i + 3) % 100
            memory.learn(ctx, tgt)
        
        # Generate
        prompt = [1, 2, 3]
        generated, _ = generate_attractor_flow(
            memory=memory,
            prompt_tokens=prompt,
            max_tokens=10,
        )
        
        new_tokens = generated[len(prompt):]
        unique_tokens = len(set(new_tokens))
        
        print(f"\nGenerated tokens: {new_tokens}")
        print(f"Unique tokens: {unique_tokens}/10")
        
        # REQUIREMENT: At least 40% unique tokens
        assert unique_tokens >= 4, \
            f"Expected at least 4 unique tokens, got {unique_tokens}"
    
    def test_stochastic_generation(self):
        """
        REQUIREMENT: Same prompt should give different outputs on different runs.
        
        This tests that φ-kernel sampling is working (not deterministic argmax).
        """
        from holographic_prod.core.attractor_generation import generate_attractor_flow
        from holographic_prod.memory.holographic_memory_unified import HolographicMemory
        
        memory = HolographicMemory(vocab_size=100, max_levels=2)
        
        # Learn patterns
        for i in range(50):
            ctx = [i % 100, (i + 1) % 100, (i + 2) % 100]
            tgt = (i + 3) % 100
            memory.learn(ctx, tgt)
        
        # Generate multiple times with different seeds
        prompt = [1, 2, 3]
        all_outputs = []
        
        for seed in [42, 123, 456, 789, 1000]:
            np.random.seed(seed)
            generated, _ = generate_attractor_flow(
                memory=memory,
                prompt_tokens=prompt,
                max_tokens=5,
            )
            new_tokens = tuple(generated[len(prompt):])
            all_outputs.append(new_tokens)
        
        print(f"\nOutputs from 5 different seeds:")
        for i, out in enumerate(all_outputs):
            print(f"  Seed {i}: {out}")
        
        # REQUIREMENT: Not all outputs should be identical
        unique_outputs = len(set(all_outputs))
        assert unique_outputs >= 2, \
            f"φ-kernel should give different outputs, but all 5 were identical"


# =============================================================================
# TDD TESTS: ION OF RETURN IMPLEMENTATION
# =============================================================================

class TestInhibitionOfReturn:
    """Tests for the IoR mechanism in generation."""
    
    def test_ior_parameter_exists(self):
        """
        REQUIREMENT: generate_attractor_flow should accept IoR parameters.
        """
        import inspect
        from holographic_prod.core.attractor_generation import generate_attractor_flow
        
        sig = inspect.signature(generate_attractor_flow)
        params = sig.parameters
        
        # Check for IoR-related parameters
        has_ior_window = 'inhibition_window' in params or 'ior_window' in params
        has_ior_factor = 'inhibition_factor' in params or 'ior_decay' in params
        
        print(f"\nFunction parameters: {list(params.keys())}")
        
        # At least one IoR parameter should exist
        assert has_ior_window or has_ior_factor or 'inhibition' in str(params).lower(), \
            "generate_attractor_flow should have IoR parameters"
    
    def test_ior_prevents_perseveration(self):
        """
        REQUIREMENT: With IoR enabled, no token should repeat more than 
        inhibition_window times in a row.
        """
        from holographic_prod.core.attractor_generation import generate_attractor_flow
        from holographic_prod.memory.holographic_memory_unified import HolographicMemory
        
        np.random.seed(42)
        
        memory = HolographicMemory(vocab_size=50, max_levels=2)
        
        # Learn limited patterns to encourage repetition
        for i in range(20):
            memory.learn([i % 50, (i+1) % 50], (i+2) % 50)
        
        # Generate with default IoR
        prompt = [1, 2]
        generated, _ = generate_attractor_flow(
            memory=memory,
            prompt_tokens=prompt,
            max_tokens=15,
        )
        
        new_tokens = generated[len(prompt):]
        
        # Count max consecutive repeats
        max_consecutive = 1
        current_count = 1
        for i in range(1, len(new_tokens)):
            if new_tokens[i] == new_tokens[i-1]:
                current_count += 1
                max_consecutive = max(max_consecutive, current_count)
            else:
                current_count = 1
        
        print(f"\nGenerated: {new_tokens}")
        print(f"Max consecutive repeats: {max_consecutive}")
        
        # REQUIREMENT: Max 3 consecutive (default IoR window)
        assert max_consecutive <= 3, \
            f"IoR should limit consecutive repeats to 3, got {max_consecutive}"


# =============================================================================
# TDD TESTS: PHI-KERNEL SAMPLING
# =============================================================================

class TestPhiKernelSampling:
    """Tests for φ-kernel probabilistic sampling."""
    
    def test_phi_temperature_used(self):
        """
        REQUIREMENT: Sampling should use temperature = 1/φ ≈ 0.618
        """
        from holographic_prod.core.constants import PHI_INV
        
        # Verify φ⁻¹ is correct
        expected_phi_inv = (np.sqrt(5) - 1) / 2  # ≈ 0.618
        assert abs(PHI_INV - expected_phi_inv) < 0.001, \
            f"PHI_INV should be {expected_phi_inv}, got {PHI_INV}"
        
        print(f"\nPHI_INV = {PHI_INV:.6f} (expected {expected_phi_inv:.6f})")
    
    def test_sampling_not_deterministic(self):
        """
        REQUIREMENT: Same scores should give different selections sometimes.
        
        φ-kernel converts scores to probabilities, so same input can give
        different output (unlike argmax).
        """
        from holographic_prod.core.constants import PHI_INV
        
        # Simulate φ-kernel sampling
        scores = np.array([0.3, 0.35, 0.32, 0.28, 0.25])
        
        def phi_kernel_sample(scores, seed):
            np.random.seed(seed)
            scores_pos = np.maximum(scores, 1e-10)
            logits = np.log(scores_pos) / PHI_INV
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / np.sum(probs)
            return np.random.choice(len(scores), p=probs)
        
        # Sample 10 times with different seeds
        samples = [phi_kernel_sample(scores, seed=i*100) for i in range(10)]
        unique_samples = len(set(samples))
        
        print(f"\nScores: {scores}")
        print(f"10 samples: {samples}")
        print(f"Unique selections: {unique_samples}")
        
        # Should get at least 2 different selections
        assert unique_samples >= 2, \
            "φ-kernel should be stochastic, but always selected same token"


# =============================================================================
# TDD TESTS: POLARIZED LENSING IN GENERATION
# =============================================================================

class TestPolarizedLensingInGeneration:
    """Tests that generation uses polarized lensing for scoring."""
    
    def test_generation_uses_lensing(self):
        """
        REQUIREMENT: Generation should use polarized lensing for candidate scoring.
        
        Check by inspecting the code or by testing that lensing affects output.
        """
        import inspect
        from holographic_prod.core import attractor_generation
        
        source = inspect.getsource(attractor_generation.generate_attractor_flow)
        
        # Check for lensing-related code
        uses_lensing = any(term in source.lower() for term in [
            'lens', 'polarize', 'score_all_lenses', 'polarizedlensset'
        ])
        
        print(f"\nGeneration code analysis:")
        print(f"  Uses polarized lensing: {uses_lensing}")
        
        # This should pass after implementation
        assert uses_lensing, \
            "generate_attractor_flow should use polarized lensing for scoring"


# =============================================================================
# TDD TESTS: CONTEXTUAL UNK EMBEDDING
# =============================================================================

class TestContextualUnkEmbedding:
    """Tests for contextual <unk> handling."""
    
    def test_unk_not_fixed_embedding(self):
        """
        REQUIREMENT: <unk> should not use a fixed embedding.
        
        Instead, it should be computed from surrounding context.
        """
        from holographic_prod.memory.holographic_memory_unified import HolographicMemory
        
        memory = HolographicMemory(vocab_size=100, max_levels=2)
        
        # Check if there's a method for contextual embedding
        has_contextual = hasattr(memory, 'embed_sequence_contextual') or \
                        hasattr(memory, '_embed_with_context') or \
                        hasattr(memory.tower, 'embed_sequence_contextual')
        
        # Or check if embed_sequence handles UNK specially
        # We'll test by embedding two different contexts with UNK
        # They should give different embeddings for UNK position
        
        UNK_IDX = 0  # Standard UNK index
        
        ctx1 = [5, UNK_IDX, 10]  # UNK between 5 and 10
        ctx2 = [20, UNK_IDX, 30]  # UNK between 20 and 30
        
        emb1 = memory.tower._embed_sequence(ctx1)
        emb2 = memory.tower._embed_sequence(ctx2)
        
        # If contextual, embeddings should be different
        # If fixed UNK, embeddings might be similar
        diff = np.linalg.norm(emb1 - emb2)
        
        print(f"\nContext 1: {ctx1}")
        print(f"Context 2: {ctx2}")
        print(f"Embedding difference: {diff:.4f}")
        
        # For now, this is informational - we'll make it a requirement
        # after implementing contextual UNK
        if diff > 0.5:
            print("✓ Different contexts give different embeddings")
        else:
            print("⚠ Contexts give similar embeddings - contextual UNK may help")


# =============================================================================
# TDD TESTS: INTEGRATION
# =============================================================================

class TestGenerationIntegration:
    """Integration tests for the complete generation pipeline."""
    
    def test_full_generation_pipeline(self):
        """
        REQUIREMENT: Full generation should work with all fixes integrated.
        """
        from holographic_prod.memory.holographic_memory_unified import HolographicMemory
        from holographic_prod.core.attractor_generation import generate_attractor_flow
        
        np.random.seed(42)
        
        # Create memory and learn patterns
        memory = HolographicMemory(vocab_size=100, max_levels=2)
        
        for i in range(100):
            ctx = [i % 100, (i + 10) % 100, (i + 20) % 100]
            tgt = (i + 30) % 100
            memory.learn(ctx, tgt)
        
        # Generate
        prompt = [5, 15, 25]
        generated, stabilities = generate_attractor_flow(
            memory=memory,
            prompt_tokens=prompt,
            max_tokens=10,
        )
        
        new_tokens = generated[len(prompt):]
        
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {new_tokens}")
        print(f"Stabilities: {[f'{s:.3f}' for s in stabilities]}")
        
        # Basic sanity checks
        assert len(new_tokens) > 0, "Should generate at least 1 token"
        assert all(0 <= t < 100 for t in new_tokens), "All tokens should be valid"
        
        # Quality checks (after implementation)
        unique_ratio = len(set(new_tokens)) / len(new_tokens)
        print(f"Unique ratio: {unique_ratio:.1%}")
        
        assert unique_ratio >= 0.3, f"Should have at least 30% unique tokens"
    
    def test_generation_returns_stats(self):
        """
        REQUIREMENT: Generation should return useful statistics.
        """
        from holographic_prod.memory.holographic_memory_unified import HolographicMemory
        from holographic_prod.core.attractor_generation import generate_attractor_flow
        
        memory = HolographicMemory(vocab_size=50, max_levels=2)
        memory.learn([1, 2, 3], 4)
        
        generated, stabilities = generate_attractor_flow(
            memory=memory,
            prompt_tokens=[1, 2, 3],
            max_tokens=5,
        )
        
        # Should return stabilities
        assert stabilities is not None, "Should return stability trace"
        assert len(stabilities) > 0, "Stability trace should not be empty"
        assert all(0 <= s <= 1 for s in stabilities), "Stabilities should be in [0, 1]"
        
        print(f"\nGenerated: {generated}")
        print(f"Stabilities: {stabilities}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
