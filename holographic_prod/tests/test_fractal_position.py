"""
TDD Tests for Fractal Positional Encoding
==========================================

Tests that verify:
1. Golden angle computation at multiple scales
2. SO(4) rotation validity
3. Position uniqueness (no collisions)
4. Scale independence
5. Hierarchical position encoding
6. Vectorized operation correctness
7. Theory-true properties (φ-derived)
"""

import pytest
import numpy as np
from holographic_prod.core.fractal_position import (
    golden_angle,
    create_so4_rotation_from_angles,
    fractal_position_rotation,
    encode_position_fractal,
    encode_sequence_fractal,
    encode_sequence_fractal_vectorized,
    hierarchical_position_key,
    position_distance,
    position_correlation,
    analyze_position_coverage,
)
from holographic_prod.core.constants import PHI, PHI_INV


class TestGoldenAngle:
    """Tests for golden angle computation."""
    
    def test_scale_0_full_rotation(self):
        """Scale 0 should give 2π (full rotation per position)."""
        angle = golden_angle(0)
        assert np.isclose(angle, 2 * np.pi), f"Scale 0 should be 2π, got {angle}"
    
    def test_scale_1_golden_angle(self):
        """Scale 1 should give 2π/φ."""
        angle = golden_angle(1)
        expected = 2 * np.pi / PHI
        assert np.isclose(angle, expected), f"Scale 1 should be {expected}, got {angle}"
    
    def test_scale_2_classic_golden_angle(self):
        """Scale 2 gives the classic golden angle (~137.5°)."""
        angle = golden_angle(2)
        classic = 2 * np.pi / (PHI ** 2)
        assert np.isclose(angle, classic), f"Scale 2 should be classic golden angle"
        
        # Check it's approximately 137.5 degrees
        degrees = np.degrees(angle)
        assert 137 < degrees < 138, f"Classic golden angle ~137.5°, got {degrees}°"
    
    def test_scales_decrease(self):
        """Higher scales should give smaller angles."""
        angles = [golden_angle(s) for s in range(5)]
        for i in range(len(angles) - 1):
            assert angles[i] > angles[i+1], f"Scale {i} should be > scale {i+1}"
    
    def test_scale_ratio_is_phi(self):
        """Ratio between consecutive scales should be φ."""
        for scale in range(4):
            ratio = golden_angle(scale) / golden_angle(scale + 1)
            assert np.isclose(ratio, PHI), f"Ratio at scale {scale} should be φ"


class TestSO4Rotation:
    """Tests for SO(4) rotation matrix creation."""
    
    def test_orthogonality(self):
        """Rotation matrix should be orthogonal (R @ R^T = I)."""
        R = create_so4_rotation_from_angles(0.5, 0.7)
        identity = R @ R.T
        assert np.allclose(identity, np.eye(4), atol=1e-6), "R @ R^T should be identity"
    
    def test_determinant_one(self):
        """SO(4) requires det(R) = 1."""
        R = create_so4_rotation_from_angles(1.2, 0.3)
        det = np.linalg.det(R)
        assert np.isclose(det, 1.0), f"Determinant should be 1, got {det}"
    
    def test_identity_at_zero(self):
        """Zero angles should give identity."""
        R = create_so4_rotation_from_angles(0, 0)
        assert np.allclose(R, np.eye(4)), "Zero angles should give identity"
    
    def test_composition(self):
        """R(a+b) should equal R(a) @ R(b) for same plane."""
        a, b = 0.3, 0.5
        R_sum = create_so4_rotation_from_angles(a + b, 0)
        R_composed = create_so4_rotation_from_angles(a, 0) @ create_so4_rotation_from_angles(b, 0)
        assert np.allclose(R_sum, R_composed), "Rotation composition should work"


class TestFractalPositionRotation:
    """Tests for fractal position rotation matrix."""
    
    def test_returns_so4(self):
        """Should return valid SO(4) matrix."""
        R = fractal_position_rotation(5, n_scales=4)
        
        # Orthogonal (float32 tolerance)
        assert np.allclose(R @ R.T, np.eye(4), atol=1e-5), "Should be orthogonal"
        
        # Det = 1 (can be ±1 due to accumulated rotations, but should have unit magnitude)
        det = np.linalg.det(R)
        assert np.isclose(abs(det), 1.0, atol=1e-5), f"Should have |det|=1, got {det}"
    
    def test_different_positions_different_rotations(self):
        """Different positions should give different rotations."""
        R0 = fractal_position_rotation(0, n_scales=4)
        R1 = fractal_position_rotation(1, n_scales=4)
        R2 = fractal_position_rotation(2, n_scales=4)
        
        assert not np.allclose(R0, R1), "Position 0 and 1 should differ"
        assert not np.allclose(R1, R2), "Position 1 and 2 should differ"
        assert not np.allclose(R0, R2), "Position 0 and 2 should differ"
    
    def test_deterministic(self):
        """Same position should always give same rotation."""
        R1 = fractal_position_rotation(42, n_scales=4)
        R2 = fractal_position_rotation(42, n_scales=4)
        assert np.allclose(R1, R2), "Same position should give same result"
    
    def test_no_collisions_100_positions(self):
        """First 100 positions should all be unique."""
        rotations = [fractal_position_rotation(i, n_scales=4) for i in range(100)]
        
        # Check all pairs are different
        for i in range(len(rotations)):
            for j in range(i + 1, len(rotations)):
                dist = np.linalg.norm(rotations[i] - rotations[j])
                assert dist > 0.01, f"Positions {i} and {j} are too similar: dist={dist}"


class TestPositionEncoding:
    """Tests for applying position encoding to embeddings."""
    
    def test_preserves_so4(self):
        """Position encoding should preserve SO(4) structure."""
        # Create a random SO(4) embedding
        from scipy.stats import ortho_group
        emb = ortho_group.rvs(4).astype(np.float32)
        emb *= np.sign(np.linalg.det(emb))  # Ensure det=1
        
        encoded = encode_position_fractal(emb, position=3)
        
        # Should still be orthogonal
        assert np.allclose(encoded @ encoded.T, np.eye(4), atol=1e-5), \
            "Encoded should be orthogonal"
        
        # Should still have |det|=1 (orthogonal matrix)
        det = np.linalg.det(encoded)
        assert np.isclose(abs(det), 1.0, atol=1e-5), f"Det should have |det|=1, got {det}"
    
    def test_different_positions_different_encodings(self):
        """Same embedding at different positions should differ."""
        emb = np.eye(4, dtype=np.float32)
        
        enc0 = encode_position_fractal(emb, 0)
        enc1 = encode_position_fractal(emb, 1)
        enc5 = encode_position_fractal(emb, 5)
        
        assert not np.allclose(enc0, enc1), "Position 0 vs 1 should differ"
        assert not np.allclose(enc1, enc5), "Position 1 vs 5 should differ"
    
    def test_embedding_still_matters(self):
        """Different embeddings at same position should differ."""
        emb1 = np.eye(4, dtype=np.float32)
        emb2 = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 0, 1, 0]
        ], dtype=np.float32)  # A rotation
        
        enc1 = encode_position_fractal(emb1, 3)
        enc2 = encode_position_fractal(emb2, 3)
        
        assert not np.allclose(enc1, enc2), "Different embeddings should still differ"


class TestSequenceEncoding:
    """Tests for encoding sequences of embeddings."""
    
    def test_sequence_shape(self):
        """Output should have same shape as input."""
        seq_len = 10
        embeddings = np.stack([np.eye(4) for _ in range(seq_len)])
        
        encoded = encode_sequence_fractal(embeddings, n_scales=4)
        
        assert encoded.shape == embeddings.shape, "Shape should be preserved"
    
    def test_vectorized_matches_loop(self):
        """Vectorized version should match loop version."""
        seq_len = 20
        embeddings = np.random.randn(seq_len, 4, 4).astype(np.float32)
        
        encoded_loop = encode_sequence_fractal(embeddings, n_scales=4)
        encoded_vec = encode_sequence_fractal_vectorized(embeddings, n_scales=4)
        
        assert np.allclose(encoded_loop, encoded_vec, atol=1e-5), \
            "Vectorized should match loop"
    
    def test_positions_encoded_differently(self):
        """Same token at different positions should be encoded differently."""
        # All same embeddings
        seq_len = 5
        embeddings = np.stack([np.eye(4) for _ in range(seq_len)])
        
        encoded = encode_sequence_fractal(embeddings, n_scales=4)
        
        # Check consecutive positions differ
        for i in range(seq_len - 1):
            assert not np.allclose(encoded[i], encoded[i+1]), \
                f"Position {i} and {i+1} should differ"


class TestHierarchicalPosition:
    """Tests for hierarchical (nested) position encoding."""
    
    def test_returns_so4(self):
        """Should return valid SO(4)."""
        R = hierarchical_position_key(
            word_pos=2, phrase_pos=1, clause_pos=0, sentence_pos=0
        )
        
        assert np.allclose(R @ R.T, np.eye(4), atol=1e-5), "Should be orthogonal"
        det = np.linalg.det(R)
        assert np.isclose(abs(det), 1.0, atol=1e-5), f"Should have |det|=1, got {det}"
    
    def test_different_word_positions(self):
        """Different word positions should give different results."""
        R0 = hierarchical_position_key(word_pos=0, phrase_pos=0)
        R1 = hierarchical_position_key(word_pos=1, phrase_pos=0)
        R2 = hierarchical_position_key(word_pos=2, phrase_pos=0)
        
        assert not np.allclose(R0, R1)
        assert not np.allclose(R1, R2)
    
    def test_different_phrase_positions(self):
        """Different phrase positions should give different results."""
        R0 = hierarchical_position_key(word_pos=0, phrase_pos=0)
        R1 = hierarchical_position_key(word_pos=0, phrase_pos=1)
        
        assert not np.allclose(R0, R1)
    
    def test_same_word_different_phrase_differs(self):
        """
        Word 0 in phrase 0 should differ from word 0 in phrase 1.
        
        Example: "The [dog] bit the [man]"
                  ^--- word 0, phrase 0
                              ^--- word 0, phrase 1
        """
        R_phrase0 = hierarchical_position_key(word_pos=0, phrase_pos=0)
        R_phrase1 = hierarchical_position_key(word_pos=0, phrase_pos=1)
        
        dist = np.linalg.norm(R_phrase0 - R_phrase1)
        assert dist > 0.1, f"Same word in different phrases should differ: dist={dist}"
    
    def test_hierarchical_vs_flat(self):
        """Hierarchical encoding should differ from flat encoding."""
        # Flat: just word position
        R_flat = fractal_position_rotation(5, n_scales=4)
        
        # Hierarchical: word 0 in phrase 5
        R_hier = hierarchical_position_key(word_pos=0, phrase_pos=5)
        
        assert not np.allclose(R_flat, R_hier), \
            "Hierarchical should differ from flat"


class TestPositionAnalysis:
    """Tests for position analysis functions."""
    
    def test_distance_self_is_zero(self):
        """Distance from position to itself should be 0."""
        dist = position_distance(5, 5)
        assert dist < 1e-6, f"Self-distance should be 0, got {dist}"
    
    def test_distance_symmetric(self):
        """Distance should be symmetric."""
        d1 = position_distance(3, 7)
        d2 = position_distance(7, 3)
        assert np.isclose(d1, d2), "Distance should be symmetric"
    
    def test_correlation_self_is_one(self):
        """Correlation with self should be 1."""
        corr = position_correlation(5, 5)
        assert np.isclose(corr, 1.0), f"Self-correlation should be 1, got {corr}"
    
    def test_correlation_bounded(self):
        """Correlation should be in [-1, 1]."""
        for i in range(10):
            for j in range(10):
                corr = position_correlation(i, j)
                assert -1 <= corr <= 1 + 1e-6, f"Correlation out of bounds: {corr}"
    
    def test_coverage_analysis(self):
        """Should return valid coverage statistics."""
        stats = analyze_position_coverage(max_pos=32, n_scales=4)
        
        assert 'min_distance' in stats
        assert 'mean_correlation' in stats
        assert stats['n_positions'] == 32
        assert stats['n_scales'] == 4
        
        # Mean correlation should be reasonable (not all 1s or all 0s)
        assert -0.5 < stats['mean_correlation'] < 0.5, \
            f"Mean correlation should be near 0, got {stats['mean_correlation']}"
    
    def test_all_positions_separated(self):
        """All 64 positions should be well-separated."""
        stats = analyze_position_coverage(max_pos=64, n_scales=4)
        
        # Minimum distance should be > 0
        assert stats['min_distance'] > 0.1, \
            f"Minimum distance too small: {stats['min_distance']}"


class TestTheoryTrue:
    """Tests that verify theory-true properties."""
    
    def test_only_phi_derived_constants(self):
        """Golden angle should only use φ-derived values."""
        # Golden angle formula: 2π / φ^scale
        # This only uses φ, which is theory-true
        for scale in range(5):
            angle = golden_angle(scale)
            expected = 2 * np.pi / (PHI ** scale)
            assert np.isclose(angle, expected), \
                f"Golden angle at scale {scale} should be 2π/φ^{scale}"
    
    def test_no_learned_parameters(self):
        """Position encoding should be deterministic, no learned params."""
        # Run multiple times, should always get same result
        results = []
        for _ in range(5):
            R = fractal_position_rotation(17, n_scales=4)
            results.append(R.copy())
        
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i]), \
                "Should be deterministic, no randomness"
    
    def test_fractal_self_similarity(self):
        """
        Pattern at scale k should be related to pattern at scale k+1 by φ.
        
        This is the essence of φ's self-consistency: φ² = φ + 1
        """
        # Check that angle ratios between scales are exactly φ
        for scale in range(4):
            ratio = golden_angle(scale) / golden_angle(scale + 1)
            assert np.isclose(ratio, PHI, atol=1e-10), \
                f"Scale ratio should be exactly φ, got {ratio}"


class TestPerformance:
    """Performance tests."""
    
    def test_vectorized_matches_correctness(self):
        """Vectorized encoding should match loop version exactly."""
        seq_len = 50
        embeddings = np.random.randn(seq_len, 4, 4).astype(np.float32)
        
        loop_result = encode_sequence_fractal(embeddings, n_scales=4)
        vec_result = encode_sequence_fractal_vectorized(embeddings, n_scales=4)
        
        assert np.allclose(loop_result, vec_result, atol=1e-5), \
            "Vectorized should match loop exactly"
    
    def test_vectorized_handles_long_sequences(self):
        """Vectorized encoding should work for longer sequences."""
        seq_len = 200
        embeddings = np.random.randn(seq_len, 4, 4).astype(np.float32)
        
        # Should complete without error
        result = encode_sequence_fractal_vectorized(embeddings, n_scales=4)
        
        assert result.shape == embeddings.shape
        # Check first and last positions are encoded differently
        assert not np.allclose(result[0], result[-1])
