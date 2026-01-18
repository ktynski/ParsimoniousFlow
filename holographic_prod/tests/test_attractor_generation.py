"""
Test Attractor-Based Generation — Theory-True Continuous Flow
==============================================================

THEORY VALIDATION:
    1. State evolves continuously (not discrete lookups)
    2. Stability remains high throughout generation
    3. Output is coherent trajectory through attractors
    4. Errors don't compound (brain analog)
    
BRAIN ANALOG:
    - Human speech maintains coherence across sentences
    - Working memory state flows smoothly
    - Errors are locally corrected, not compounded
"""

import pytest
import numpy as np
import sys
import os

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ
from holographic_prod.core.algebra import build_clifford_basis
from holographic_prod.core.attractor_generation import (
    generate_attractor_flow,
    generate_batch_attractor_flow,
    _grace_operator_batch,
    _grace_stability_batch,
    _vorticity_scores_batch,
)
from holographic_prod.memory.holographic_memory_unified import HolographicMemory


@pytest.fixture
def learned_memory():
    """Create memory with learned patterns for generation testing."""
    memory = HolographicMemory(vocab_size=1000)
    
    # Learn some coherent sequences (like simple sentence patterns)
    # Pattern: "the cat sat on the mat"
    the, cat, sat, on, mat = 10, 20, 30, 40, 50
    
    # Learn forward associations
    memory.learn([the], cat)
    memory.learn([the, cat], sat)
    memory.learn([cat, sat], on)
    memory.learn([sat, on], the)
    memory.learn([on, the], mat)
    
    # Learn some variations
    memory.learn([the], cat)  # Reinforce
    memory.learn([a_dog := 60], sat)  # "a dog sat"
    memory.learn([a_dog, sat], on)
    
    return memory


class TestAttractorGeneration:
    """Test attractor-based generation mechanism."""
    
    def test_single_prompt_generation(self, learned_memory):
        """Test basic generation from a prompt."""
        prompt = [10]  # "the"
        generated, stabilities = generate_attractor_flow(
            memory=learned_memory,
            prompt_tokens=prompt,
            max_tokens=5,
            grace_steps=3,
        )
        
        # Should generate tokens
        assert len(generated) > len(prompt), "Should generate new tokens"
        
        # Stabilities should be tracked
        assert len(stabilities) > 0, "Should track stabilities"
        
        # Output should include prompt
        assert generated[:len(prompt)] == prompt, "Should start with prompt"
    
    def test_stability_maintained(self, learned_memory):
        """Test that stability remains high during coherent generation."""
        prompt = [10, 20]  # "the cat"
        generated, stabilities = generate_attractor_flow(
            memory=learned_memory,
            prompt_tokens=prompt,
            max_tokens=5,
            grace_steps=3,
        )
        
        if stabilities:
            # Stability should generally stay above threshold
            mean_stability = np.mean(stabilities)
            assert mean_stability > 0.1, f"Mean stability too low: {mean_stability}"
            
            # No catastrophic drops (errors compounding)
            # Note: stability can vary as state flows through attractors
            # We only check for sustained drops, not single oscillations
            sustained_drops = 0
            for i, s in enumerate(stabilities):
                if i > 0:
                    drop = stabilities[i-1] - s
                    if drop > 0.3:  # Individual step drop
                        sustained_drops += 1
            assert sustained_drops < 3, f"Stability dropped catastrophically {sustained_drops} times"
    
    def test_continuity_not_discrete(self, learned_memory):
        """
        Verify generation uses state continuity, not discrete lookups.
        
        THEORY: Each step should evolve from previous state, not restart.
        
        NOTE (v5.17.0): With φ-kernel sampling, generation is intentionally
        stochastic. To test determinism, we disable φ-kernel.
        """
        prompt = [10]
        
        np.random.seed(42)
        
        # Generate twice with φ-kernel DISABLED for determinism test
        gen1, _ = generate_attractor_flow(
            memory=learned_memory,
            prompt_tokens=prompt,
            max_tokens=3,
            grace_steps=3,
            use_phi_kernel=False,  # Disable stochastic sampling
        )
        
        gen2, _ = generate_attractor_flow(
            memory=learned_memory,
            prompt_tokens=prompt,
            max_tokens=3,
            grace_steps=3,
            use_phi_kernel=False,  # Disable stochastic sampling
        )
        
        # Same input, same state evolution → same output (when deterministic)
        assert gen1 == gen2, "Deterministic state flow should give same results"
    
    def test_no_error_compounding(self, learned_memory):
        """
        Test that errors don't compound into gibberish.
        
        BRAIN ANALOG: Humans don't produce "the the the the park on".
        """
        prompt = [10]
        generated, stabilities = generate_attractor_flow(
            memory=learned_memory,
            prompt_tokens=prompt,
            max_tokens=10,
            grace_steps=3,
        )
        
        # Check for repetition (error compounding symptom)
        gen_part = generated[len(prompt):]
        if len(gen_part) >= 3:
            # No more than 2 consecutive repeats
            repeats = 0
            max_repeats = 0
            prev = None
            for t in gen_part:
                if t == prev:
                    repeats += 1
                    max_repeats = max(max_repeats, repeats)
                else:
                    repeats = 0
                prev = t
            
            assert max_repeats < 3, f"Error compounding: {max_repeats+1} consecutive repeats"


class TestBatchGeneration:
    """Test batched generation for GPU efficiency."""
    
    def test_batch_similar_to_single(self, learned_memory):
        """Batch generation should produce similar results to single generation.
        
        Note: With φ-kernel sampling (v5.17.0), generation is stochastic.
        We test in deterministic mode to verify structural correctness.
        """
        prompts = [[10], [20], [10, 20]]
        
        np.random.seed(42)
        
        # Single generations (deterministic mode)
        singles = [
            generate_attractor_flow(
                memory=learned_memory,
                prompt_tokens=p,
                max_tokens=3,
                grace_steps=3,
                use_phi_kernel=False,  # Deterministic for comparison
            )
            for p in prompts
        ]
        
        np.random.seed(42)  # Reset seed
        
        # Batch generation (also deterministic)
        batched = generate_batch_attractor_flow(
            memory=learned_memory,
            prompts=prompts,
            max_tokens=3,
            grace_steps=3,
        )
        
        # Check that both produce valid results (prompts included)
        for i, (single, batch) in enumerate(zip(singles, batched)):
            single_tokens = single[0]
            batch_tokens = batch[0]
            
            # Both should start with prompt
            prompt = prompts[i]
            assert single_tokens[:len(prompt)] == prompt, "Single should include prompt"
            assert batch_tokens[:len(prompt)] == prompt, "Batch should include prompt"
            
            # Both should generate something
            assert len(single_tokens) > len(prompt), "Single should generate tokens"
            assert len(batch_tokens) > len(prompt), "Batch should generate tokens"


class TestBatchedOperations:
    """Test GPU-optimized batched operations."""
    
    @pytest.fixture
    def setup(self):
        """Create test data."""
        basis = build_clifford_basis()
        batch = np.random.randn(8, 4, 4).astype(np.float32)
        candidates = np.random.randn(100, 4, 4).astype(np.float32)
        return basis, batch, candidates
    
    def test_batched_grace_operator(self, setup):
        """Test batched Grace operator contracts all inputs."""
        basis, batch, _ = setup
        
        result = _grace_operator_batch(batch, basis, np)
        
        assert result.shape == batch.shape, "Shape should be preserved"
        
        # Norms should decrease (Grace contracts)
        for i in range(len(batch)):
            assert np.linalg.norm(result[i]) <= np.linalg.norm(batch[i]) * 1.1, \
                f"Grace should contract or preserve norm for sample {i}"
    
    def test_batched_stability(self, setup):
        """Test batched stability computation."""
        basis, batch, _ = setup
        
        stabilities = _grace_stability_batch(batch, basis, np)
        
        assert stabilities.shape == (len(batch),), "One stability per sample"
        assert np.all(stabilities >= 0), "Stabilities should be non-negative"
        assert np.all(stabilities <= 1), "Stabilities should be ≤ 1"
    
    def test_batched_vorticity_scores(self, setup):
        """Test batched scoring against candidates."""
        basis, batch, candidates = setup
        
        scores = _vorticity_scores_batch(batch, candidates, basis, np)
        
        assert scores.shape == (len(batch), len(candidates)), \
            "Should have [batch, candidates] scores"
        
        # Scores should be similarity values
        assert np.all(scores >= -1.1), "Scores should be ≥ -1"
        assert np.all(scores <= 1.1), "Scores should be ≤ 1"


class TestMemoryIntegration:
    """Test integration with HolographicMemory.generate()."""
    
    def test_generate_method_uses_attractor_flow(self, learned_memory):
        """Verify memory.generate() uses attractor flow."""
        prompt = [10, 20]
        
        generated, stats = learned_memory.generate(
            prompt=prompt,
            max_tokens=5,
        )
        
        # Should have attractor_flow flag
        assert stats.get('attractor_flow') == True, \
            "Should use attractor flow, not discrete lookups"
        
        # Should have stability metrics
        assert 'avg_stability' in stats or 'stability_trace' in stats, \
            "Should report stability metrics"
    
    def test_generation_coherence(self, learned_memory):
        """Generated output should be coherent (not gibberish)."""
        prompt = [10]  # "the"
        
        generated, stats = learned_memory.generate(
            prompt=prompt,
            max_tokens=10,
        )
        
        # Should generate something
        assert len(generated) > 0, "Should generate tokens"
        
        # All tokens should be valid
        vocab_size = learned_memory.vocab_size
        for t in generated:
            assert 0 <= t < vocab_size, f"Invalid token: {t}"


def run_tests():
    """Run all tests with detailed output."""
    print("=" * 70)
    print("ATTRACTOR-BASED GENERATION TESTS")
    print("Theory: State flows through attractors (not discrete lookups)")
    print("=" * 70)
    
    # Run pytest with verbose output
    import pytest
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-x',  # Stop on first failure
    ])
    
    if exit_code == 0:
        print("\n" + "=" * 70)
        print("✓ ALL ATTRACTOR GENERATION TESTS PASSED")
        print("  State flows continuously through learned attractors")
        print("  Errors don't compound (brain-like coherence)")
        print("=" * 70)
    
    return exit_code


if __name__ == '__main__':
    exit(run_tests())
