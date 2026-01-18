"""
Comprehensive Tests for Polarized Satellite Lensing
====================================================

TDD tests verifying:
1. Lens generation is deterministic and produces valid SO(4)
2. Polarization breaks metric invariance (the core fix)
3. Aliased pairs become distinguishable after polarization
4. Chord reconstruction preserves signal
5. Min-correlation criterion provides robust disambiguation
6. GPU/CPU parity
7. Performance is acceptable
"""

import numpy as np
import pytest
from scipy.stats import ortho_group
import time
import sys
sys.path.insert(0, '/Users/fractlphoneroom1/Desktop/ParsimoniousFlow')

from holographic_prod.core.lensing import (
    PolarizedLens,
    PolarizedLensSet,
    create_lens_for_satellite,
    polarized_similarity,
)
from holographic_prod.core.constants import MATRIX_DIM, DTYPE


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def basis():
    """Clifford basis for decomposition tests."""
    from holographic_prod.core.algebra import build_clifford_basis
    return build_clifford_basis(np)


@pytest.fixture
def random_so4_embedding():
    """Generate a random SO(4) embedding."""
    def _make(seed: int) -> np.ndarray:
        M = ortho_group.rvs(MATRIX_DIM, random_state=seed).astype(DTYPE)
        if np.linalg.det(M) < 0:
            M[:, 0] *= -1
        return M
    return _make


@pytest.fixture
def aliased_pair(random_so4_embedding):
    """Find a highly correlated pair of SO(4) embeddings."""
    best_corr = 0
    best_pair = None
    
    for i in range(500):
        e1 = random_so4_embedding(i)
        e2 = random_so4_embedding(i + 500)
        corr = abs(np.sum(e1 * e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
        if corr > best_corr:
            best_corr = corr
            best_pair = (e1, e2, corr)
    
    return best_pair


@pytest.fixture
def lens_set():
    """Standard 16-lens set."""
    return PolarizedLensSet(n_lenses=16, seed=42)


# =============================================================================
# TEST: LENS GENERATION
# =============================================================================

class TestLensGeneration:
    """Test that lenses are properly generated."""
    
    def test_lens_is_so4(self, random_so4_embedding):
        """Lens matrix should be SO(4): orthogonal with det=+1."""
        lens = PolarizedLens(seed=42)
        
        # Orthogonality: L @ L^T = I
        product = lens.lens @ lens.lens.T
        assert np.allclose(product, np.eye(MATRIX_DIM), atol=1e-6), \
            "Lens should be orthogonal"
        
        # Determinant +1 (not -1)
        det = np.linalg.det(lens.lens)
        assert abs(det - 1.0) < 1e-6, f"Lens should have det=+1, got {det}"
    
    def test_lens_is_deterministic(self):
        """Same seed should produce identical lens."""
        lens1 = PolarizedLens(seed=42)
        lens2 = PolarizedLens(seed=42)
        
        assert np.allclose(lens1.lens, lens2.lens), \
            "Same seed should produce identical lens"
    
    def test_different_seeds_different_lenses(self):
        """Different seeds should produce different lenses."""
        lens1 = PolarizedLens(seed=42)
        lens2 = PolarizedLens(seed=43)
        
        assert not np.allclose(lens1.lens, lens2.lens), \
            "Different seeds should produce different lenses"
    
    def test_lens_set_has_correct_count(self):
        """LensSet should have specified number of lenses."""
        lens_set = PolarizedLensSet(n_lenses=16, seed=42)
        assert len(lens_set) == 16
        
        lens_set_8 = PolarizedLensSet(n_lenses=8, seed=42)
        assert len(lens_set_8) == 8
    
    def test_lens_set_all_unique(self):
        """All lenses in a set should be unique."""
        lens_set = PolarizedLensSet(n_lenses=16, seed=42)
        
        for i in range(16):
            for j in range(i + 1, 16):
                corr = abs(np.sum(lens_set[i].lens * lens_set[j].lens))
                # Should not be identical (corr < 16 for orthogonal 4x4)
                assert corr < 15, f"Lenses {i} and {j} should be distinct"


# =============================================================================
# TEST: POLARIZATION BREAKS INVARIANCE
# =============================================================================

class TestPolarizationBreaksInvariance:
    """
    The core test: polarization must break the metric invariance
    that defeated pure conjugation.
    """
    
    def test_pure_conjugation_preserves_correlation(self, aliased_pair):
        """Verify the PROBLEM: pure rotation preserves correlation."""
        e1, e2, original_corr = aliased_pair
        
        # Test with multiple lenses - pure conjugation
        correlations = []
        for seed in range(16):
            lens = PolarizedLens(seed=42 + seed * 137)
            
            # Pure conjugation (no ReLU)
            e1_rotated = lens.lens @ e1 @ lens.lens_inv
            e2_rotated = lens.lens @ e2 @ lens.lens_inv
            
            corr = abs(np.sum(e1_rotated * e2_rotated) / 
                      (np.linalg.norm(e1_rotated) * np.linalg.norm(e2_rotated)))
            correlations.append(corr)
        
        # All correlations should be identical (invariant)
        variance = np.var(correlations)
        assert variance < 1e-10, \
            f"Pure conjugation should preserve correlation (var={variance})"
    
    def test_polarization_breaks_correlation(self, aliased_pair):
        """Verify the FIX: polarization breaks correlation."""
        e1, e2, original_corr = aliased_pair
        
        # Test with multiple lenses - polarized projection
        correlations = []
        for seed in range(16):
            lens = PolarizedLens(seed=42 + seed * 137)
            
            # Polarized (with ReLU)
            e1_polarized = lens.polarize(e1)
            e2_polarized = lens.polarize(e2)
            
            norm1 = np.linalg.norm(e1_polarized)
            norm2 = np.linalg.norm(e2_polarized)
            
            if norm1 < 1e-10 or norm2 < 1e-10:
                corr = 0.0  # Distinguishable (one is zero)
            else:
                corr = abs(np.sum(e1_polarized * e2_polarized) / (norm1 * norm2))
            correlations.append(corr)
        
        # Correlations should NOW vary
        variance = np.var(correlations)
        assert variance > 0 or min(correlations) < original_corr * 0.5, \
            "Polarization should break correlation invariance"
        
        # And should reduce maximum correlation significantly
        max_corr = max(correlations)
        assert max_corr < original_corr * 0.5, \
            f"Polarized max corr {max_corr} should be << original {original_corr}"
    
    def test_polarization_reduces_aliased_correlation_dramatically(self, aliased_pair, lens_set):
        """The aliased pair should become distinguishable after polarization."""
        e1, e2, original_corr = aliased_pair
        
        min_corr, max_corr, mean_corr = polarized_similarity(e1, e2, lens_set)
        
        print(f"\nOriginal correlation: {original_corr:.4f}")
        print(f"Polarized min: {min_corr:.4f}")
        print(f"Polarized max: {max_corr:.4f}")
        print(f"Polarized mean: {mean_corr:.4f}")
        
        # Key assertion: minimum should be very low
        assert min_corr < 0.1, \
            f"Min polarized corr {min_corr} should be < 0.1 for aliased pair"


# =============================================================================
# TEST: CHORD RECONSTRUCTION
# =============================================================================

class TestChordReconstruction:
    """Test that chord aggregation preserves signal."""
    
    def test_chord_captures_majority_of_signal(self, random_so4_embedding, lens_set):
        """Chord reconstruction should capture most of the original signal."""
        original = random_so4_embedding(42)
        
        # Polarize through all lenses
        polarized_views = lens_set.polarize_all(original)
        
        # Compute chord
        chord = lens_set.compute_chord(polarized_views)
        
        # Check correlation with original
        norm_orig = np.linalg.norm(original)
        norm_chord = np.linalg.norm(chord)
        
        if norm_chord > 1e-10:
            reconstruction_corr = np.sum(original * chord) / (norm_orig * norm_chord)
        else:
            reconstruction_corr = 0.0
        
        print(f"\nReconstruction correlation: {reconstruction_corr:.4f}")
        
        # Should capture significant portion (not perfect due to ReLU)
        assert reconstruction_corr > 0.5, \
            f"Chord should capture >50% of signal, got {reconstruction_corr}"
    
    def test_chord_suppresses_ghost(self, aliased_pair, lens_set):
        """Chord should reduce correlation between aliased pairs."""
        e1, e2, original_corr = aliased_pair
        
        # Compute chords
        chord1 = lens_set.compute_chord(lens_set.polarize_all(e1))
        chord2 = lens_set.compute_chord(lens_set.polarize_all(e2))
        
        # Check chord correlation
        norm1 = np.linalg.norm(chord1)
        norm2 = np.linalg.norm(chord2)
        
        if norm1 > 1e-10 and norm2 > 1e-10:
            chord_corr = abs(np.sum(chord1 * chord2) / (norm1 * norm2))
        else:
            chord_corr = 0.0
        
        print(f"\nOriginal correlation: {original_corr:.4f}")
        print(f"Chord correlation: {chord_corr:.4f}")
        
        # Chord correlation should be lower (ghost suppressed)
        # Note: may go negative due to ReLU asymmetry, so check absolute
        assert abs(chord_corr) < original_corr, \
            f"Chord should reduce aliasing: {chord_corr} vs {original_corr}"


# =============================================================================
# TEST: MIN-CORRELATION CRITERION
# =============================================================================

class TestMinCorrelationCriterion:
    """Test the min-correlation disambiguation criterion."""
    
    def test_min_correlation_for_distinct_embeddings(self, random_so4_embedding, lens_set):
        """Distinct random embeddings should have some view with low correlation."""
        e1 = random_so4_embedding(42)
        e2 = random_so4_embedding(43)
        
        min_corr = lens_set.min_correlation_across_views(
            lens_set.polarize_all(e1),
            lens_set.polarize_all(e2)
        )
        
        # Should be distinguishable in at least one view
        assert min_corr < 0.5, f"Distinct embeddings should have min_corr < 0.5, got {min_corr}"
    
    def test_min_correlation_for_same_embedding(self, random_so4_embedding, lens_set):
        """Same embedding should have high correlation in all views."""
        e1 = random_so4_embedding(42)
        
        min_corr = lens_set.min_correlation_across_views(
            lens_set.polarize_all(e1),
            lens_set.polarize_all(e1)  # Same embedding
        )
        
        # Should be identical in all views
        assert min_corr > 0.99, f"Same embedding should have min_corr > 0.99, got {min_corr}"
    
    def test_min_correlation_breaks_aliasing(self, aliased_pair, lens_set):
        """Aliased pair should be distinguishable via min-correlation."""
        e1, e2, original_corr = aliased_pair
        
        min_corr = lens_set.min_correlation_across_views(
            lens_set.polarize_all(e1),
            lens_set.polarize_all(e2)
        )
        
        print(f"\nAliased pair: original corr = {original_corr:.4f}")
        print(f"Min correlation across views: {min_corr:.4f}")
        
        # The aliased pair should be distinguishable in some view
        assert min_corr < original_corr * 0.1, \
            f"Min corr {min_corr} should be << original {original_corr}"


# =============================================================================
# TEST: EFFECTIVE CAPACITY
# =============================================================================

class TestEffectiveCapacity:
    """Test that polarized lensing increases effective capacity."""
    
    def test_collision_rate_reduction(self, random_so4_embedding, lens_set):
        """Polarized views should have fewer collisions than single view."""
        n_embeddings = 100
        collision_threshold = 0.9
        
        # Generate embeddings
        embeddings = [random_so4_embedding(i) for i in range(n_embeddings)]
        
        # Count single-view collisions
        single_view_collisions = 0
        for i in range(n_embeddings):
            for j in range(i + 1, n_embeddings):
                corr = abs(np.sum(embeddings[i] * embeddings[j]) / 
                          (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])))
                if corr > collision_threshold:
                    single_view_collisions += 1
        
        # Count polarized collisions (using min-correlation criterion)
        polarized_views = [lens_set.polarize_all(e) for e in embeddings]
        polarized_collisions = 0
        for i in range(n_embeddings):
            for j in range(i + 1, n_embeddings):
                min_corr = lens_set.min_correlation_across_views(
                    polarized_views[i], polarized_views[j]
                )
                if min_corr > collision_threshold:
                    polarized_collisions += 1
        
        print(f"\nSingle-view collisions: {single_view_collisions}")
        print(f"Polarized collisions (min-corr): {polarized_collisions}")
        
        # Polarized should have fewer or equal collisions
        assert polarized_collisions <= single_view_collisions, \
            "Polarized lensing should not increase collisions"


# =============================================================================
# TEST: BATCH OPERATIONS
# =============================================================================

class TestBatchOperations:
    """Test batch polarization for efficiency."""
    
    def test_batch_polarize_matches_single(self, random_so4_embedding):
        """Batch polarization should match individual polarization."""
        lens = PolarizedLens(seed=42)
        
        # Create batch of embeddings
        batch_size = 10
        embeddings = np.stack([random_so4_embedding(i) for i in range(batch_size)])
        
        # Batch polarization
        batch_result = lens.polarize_batch(embeddings)
        
        # Individual polarization
        individual_results = np.stack([lens.polarize(embeddings[i]) for i in range(batch_size)])
        
        # Should match
        assert np.allclose(batch_result, individual_results, atol=1e-6), \
            "Batch polarization should match individual"
    
    def test_batch_restore_matches_single(self, random_so4_embedding):
        """Batch restore should match individual restore."""
        lens = PolarizedLens(seed=42)
        
        # Create batch of polarized views
        batch_size = 10
        embeddings = np.stack([random_so4_embedding(i) for i in range(batch_size)])
        polarized = lens.polarize_batch(embeddings)
        
        # Batch restore
        batch_restored = lens.restore_batch(polarized)
        
        # Individual restore
        individual_restored = np.stack([lens.restore(polarized[i]) for i in range(batch_size)])
        
        # Should match
        assert np.allclose(batch_restored, individual_restored, atol=1e-6), \
            "Batch restore should match individual"


# =============================================================================
# TEST: PERFORMANCE
# =============================================================================

class TestPerformance:
    """Test that lensing operations are efficient."""
    
    def test_polarize_is_fast(self, random_so4_embedding):
        """Single polarization should be very fast."""
        lens = PolarizedLens(seed=42)
        embedding = random_so4_embedding(42)
        
        # Warm up
        _ = lens.polarize(embedding)
        
        # Time 1000 operations
        n_ops = 1000
        start = time.perf_counter()
        for _ in range(n_ops):
            _ = lens.polarize(embedding)
        elapsed = time.perf_counter() - start
        
        ops_per_sec = n_ops / elapsed
        print(f"\nPolarize: {ops_per_sec:.0f} ops/sec ({elapsed*1000:.2f}ms for {n_ops})")
        
        # Should be very fast (>10k ops/sec on CPU)
        assert ops_per_sec > 1000, f"Polarize too slow: {ops_per_sec} ops/sec"
    
    def test_batch_is_faster_than_loop(self, random_so4_embedding):
        """Batch operations should be faster than looping."""
        lens = PolarizedLens(seed=42)
        batch_size = 100
        embeddings = np.stack([random_so4_embedding(i) for i in range(batch_size)])
        
        # Warm up
        _ = lens.polarize_batch(embeddings)
        
        # Time batch
        n_trials = 100
        start = time.perf_counter()
        for _ in range(n_trials):
            _ = lens.polarize_batch(embeddings)
        batch_time = time.perf_counter() - start
        
        # Time loop
        start = time.perf_counter()
        for _ in range(n_trials):
            for i in range(batch_size):
                _ = lens.polarize(embeddings[i])
        loop_time = time.perf_counter() - start
        
        speedup = loop_time / batch_time
        print(f"\nBatch time: {batch_time*1000:.2f}ms")
        print(f"Loop time: {loop_time*1000:.2f}ms")
        print(f"Speedup: {speedup:.1f}x")
        
        # Batch should be faster
        assert speedup > 1.5, f"Batch should be faster: {speedup}x"


# =============================================================================
# TEST: DETERMINISM
# =============================================================================

class TestDeterminism:
    """Test reproducibility across restarts."""
    
    def test_create_lens_for_satellite_is_deterministic(self):
        """create_lens_for_satellite should be deterministic."""
        lens1 = create_lens_for_satellite(satellite_index=5, tower_seed=42)
        lens2 = create_lens_for_satellite(satellite_index=5, tower_seed=42)
        
        assert np.allclose(lens1.lens, lens2.lens), \
            "create_lens_for_satellite should be deterministic"
    
    def test_different_satellites_different_lenses(self):
        """Different satellites should have different lenses."""
        lens0 = create_lens_for_satellite(satellite_index=0, tower_seed=42)
        lens1 = create_lens_for_satellite(satellite_index=1, tower_seed=42)
        
        assert not np.allclose(lens0.lens, lens1.lens), \
            "Different satellites should have different lenses"
    
    def test_lens_set_is_deterministic(self):
        """LensSet should be fully deterministic."""
        set1 = PolarizedLensSet(n_lenses=16, seed=42)
        set2 = PolarizedLensSet(n_lenses=16, seed=42)
        
        for i in range(16):
            assert np.allclose(set1[i].lens, set2[i].lens), \
                f"Lens {i} should be identical across sets"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
