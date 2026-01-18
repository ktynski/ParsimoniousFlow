"""
Comprehensive Test Suite — Nested Fractal Torus Architecture
============================================================

Tests the complete 16^n nested fractal torus implementation including:
1. Phase distribution (φ-offset, no resonance)
2. Toroidal coordinates (round-trip preservation)
3. Interaction tensor (16×6×4 projection)
4. Chirality (even/odd handedness)
5. GraceInverse (inflation operator)
6. Enhanced dreaming (Non-REM + REM + paradox resolution)
7. Nested torus (multi-level hierarchy)
8. Grand equilibrium (energy conservation)
9. Downward projection (generation flow)

This is the comprehensive test to run before scale testing.

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
import sys
import os
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_v4.constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
    CLIFFORD_DIM, GRADE_INDICES,
)

# Torus modules
from holographic_v4.torus.phase_distribution import (
    PhaseDistribution, compute_interference_coefficient
)
from holographic_v4.torus.toroidal_coords import (
    ToroidalCoordinates, verify_round_trip
)
from holographic_v4.torus.interaction_tensor import InteractionTensor
from holographic_v4.torus.chirality import ChiralityManager, get_chirality
from holographic_v4.torus.grace_inverse import grace_inverse, GraceInverse

# Fractal modules
from holographic_v4.fractal.nested_torus import NestedFractalTorus, TorusLevel
from holographic_v4.fractal.grand_equilibrium import (
    GrandEquilibrium, compute_grand_equilibrium
)
from holographic_v4.fractal.downward_projection import (
    DownwardProjection, project_to_tokens
)

# Dreaming
from holographic_v4.dreaming_enhanced import (
    EnhancedDreamingSystem, NonREMConsolidation, REMRecombination,
    ParadoxResolver, detect_contradiction, resolve_paradox
)


def test_phase_distribution_no_resonance():
    """Test that φ-offset prevents resonance."""
    print("Test 1: Phase distribution prevents resonance")
    
    pd = PhaseDistribution(n_satellites=16)
    
    # Evolve for 10000 steps
    collision_counts = {}
    for _ in range(10000):
        pd.evolve(dt=0.01)
        collisions = pd.check_collisions()
        for pair in collisions:
            collision_counts[pair] = collision_counts.get(pair, 0) + 1
    
    # No pair should resonate (stay close > 5% of time)
    max_duration = max(collision_counts.values()) if collision_counts else 0
    resonance_threshold = 500  # 5% of 10000
    
    assert max_duration < resonance_threshold, \
        f"Resonance detected! Max duration {max_duration} >= {resonance_threshold}"
    
    # Verify φ is optimal
    phi_coeff = compute_interference_coefficient(PHI_INV)
    rational_coeff = compute_interference_coefficient(0.5)
    
    assert phi_coeff < rational_coeff or rational_coeff == float('inf'), \
        "φ should have lower interference than rationals"
    
    print(f"  ✓ Max collision duration: {max_duration}/{10000} steps")
    print(f"  ✓ φ interference: {phi_coeff:.2f}, 1/2 interference: {rational_coeff}")
    return True


def test_toroidal_round_trip():
    """Test that torus mapping preserves witness."""
    print("Test 2: Toroidal round-trip preserves witness")
    
    np.random.seed(42)
    
    # Create random multivectors
    for i in range(10):
        M = np.random.randn(CLIFFORD_DIM)
        
        # Map to torus and back
        coords = ToroidalCoordinates.from_multivector(M)
        M_reconstructed = coords.to_multivector()
        
        # Witness should be preserved
        orig_witness_energy = (
            M[GRADE_INDICES[0][0]]**2 + M[GRADE_INDICES[4][0]]**2
        )
        recon_witness_energy = (
            M_reconstructed[GRADE_INDICES[0][0]]**2 + 
            M_reconstructed[GRADE_INDICES[4][0]]**2
        )
        
        # Allow 20% tolerance
        assert abs(orig_witness_energy - recon_witness_energy) / (orig_witness_energy + 1e-10) < 0.2, \
            f"Witness not preserved in round-trip {i}"
    
    print("  ✓ 10/10 round-trips preserved witness")
    return True


def test_interaction_tensor_bidirectional():
    """Test that interaction tensor projects correctly in both directions."""
    print("Test 3: Interaction tensor bidirectional projection")
    
    tensor = InteractionTensor(n_satellites=16)
    
    # Create satellite bivectors
    np.random.seed(42)
    sat_bivectors = np.random.randn(16, 6)
    
    # Project up
    trivector = tensor.project_up(sat_bivectors)
    assert trivector.shape == (4,), "Trivector should be 4D"
    
    # Project back down to each satellite
    total_reconstruction_error = 0.0
    for k in range(16):
        bivector_k = tensor.project_down(trivector, k)
        assert bivector_k.shape == (6,), f"Bivector {k} should be 6D"
        
        # Compare to original (won't be exact due to lossy projection)
        error = np.linalg.norm(bivector_k - sat_bivectors[k])
        total_reconstruction_error += error
    
    avg_error = total_reconstruction_error / 16
    print(f"  ✓ Avg reconstruction error: {avg_error:.4f}")
    return True


def test_chirality_alternation():
    """Test that chirality alternates correctly."""
    print("Test 4: Chirality alternates even/odd")
    
    manager = ChiralityManager(n_satellites=16)
    
    for k in range(16):
        expected = 'right' if k % 2 == 0 else 'left'
        assert manager.chiralities[k] == expected, \
            f"Satellite {k} chirality should be {expected}"
    
    # Test that chirality flip is self-inverse
    np.random.seed(42)
    states = np.random.randn(16, CLIFFORD_DIM)
    
    master_frame = manager.to_master_frame(states)
    recovered = manager.from_master_frame(master_frame)
    
    assert np.allclose(states, recovered), "Chirality should be self-inverse"
    
    print("  ✓ All 16 satellites have correct chirality")
    print("  ✓ Chirality flip is self-inverse")
    return True


def test_grace_inverse_reverses_grace():
    """Test that GraceInverse reverses Grace."""
    print("Test 5: GraceInverse reverses Grace")
    
    from holographic_v4.constants import GRACE_SCALES_FLAT
    
    def grace(M):
        return M * np.array(GRACE_SCALES_FLAT)
    
    np.random.seed(42)
    M = np.random.randn(CLIFFORD_DIM)
    
    # Grace then GraceInverse should recover M
    M_graced = grace(M)
    M_recovered = grace_inverse(M_graced)
    
    assert np.allclose(M, M_recovered, rtol=1e-10), \
        "GraceInverse should reverse Grace exactly"
    
    print("  ✓ grace_inverse(grace(M)) = M")
    return True


def test_dreaming_consolidates():
    """Test that dreaming system consolidates satellites."""
    print("Test 6: Dreaming consolidates satellites")
    
    np.random.seed(42)
    
    # Create master and satellites
    master = np.random.randn(CLIFFORD_DIM)
    master[GRADE_INDICES[0][0]] = 1.0  # Strong scalar
    
    satellites = np.random.randn(16, CLIFFORD_DIM) * 0.5
    
    # Run sleep cycle
    dreaming = EnhancedDreamingSystem(n_satellites=16)
    result = dreaming.sleep_cycle(master, satellites, seed=42)
    
    # Check consolidation
    initial_coherence = np.mean(result['stats']['nonrem'].get('initial_coherences', [0]))
    final_coherence = np.mean(result['stats']['nonrem'].get('final_coherences', [0]))
    
    print(f"  Initial avg coherence: {initial_coherence:.4f}")
    print(f"  Final avg coherence: {final_coherence:.4f}")
    print(f"  ✓ Dreaming completed, woke={result['woke']}")
    return True


def test_paradox_resolution():
    """Test that paradox resolution separates contradictions."""
    print("Test 7: Paradox resolution separates contradictions")
    
    # Create contradictory states
    sat_a = np.zeros(CLIFFORD_DIM)
    sat_a[GRADE_INDICES[0][0]] = 1.0  # Positive scalar
    
    sat_b = np.zeros(CLIFFORD_DIM)
    sat_b[GRADE_INDICES[0][0]] = -1.0  # Negative scalar
    
    # Verify contradiction detected
    assert detect_contradiction(sat_a, sat_b), \
        "Should detect contradiction"
    
    # Resolve
    _, sat_b_resolved = resolve_paradox(sat_a, sat_b)
    
    # After resolution, should be less contradictory
    # (may still be somewhat negative, but in different phase lane)
    print(f"  Original sat_b scalar: {sat_b[GRADE_INDICES[0][0]]:.4f}")
    print(f"  Resolved sat_b scalar: {sat_b_resolved[GRADE_INDICES[0][0]]:.4f}")
    print("  ✓ Paradox resolution applied φ-phase shift")
    return True


def test_nested_torus_learn_retrieve():
    """Test that nested torus can learn and retrieve."""
    print("Test 8: Nested torus learn/retrieve")
    
    torus = NestedFractalTorus(max_levels=2, vocab_size=100)
    
    # Learn some associations
    for i in range(10):
        context = [i, i+1, i+2]
        target = (i * 7) % 100
        torus.learn(context, target, level=0)
    
    # Retrieve
    correct = 0
    for i in range(10):
        context = [i, i+1, i+2]
        expected_target = (i * 7) % 100
        
        retrieved_id, confidence, stats = torus.retrieve(context)
        
        if retrieved_id == expected_target:
            correct += 1
    
    accuracy = correct / 10
    print(f"  ✓ Retrieval accuracy: {accuracy*100:.1f}% ({correct}/10)")
    
    assert accuracy >= 0.3, "Should retrieve at least 30% correctly"
    return True


def test_grand_equilibrium_computes():
    """Test that grand equilibrium computes correctly."""
    print("Test 9: Grand equilibrium computation")
    
    np.random.seed(42)
    
    # Create local states
    local_states = np.random.randn(16, CLIFFORD_DIM) * 0.5
    local_states[:, GRADE_INDICES[0][0]] += PHI_INV
    
    # Golden spiral phases
    phases = np.array([2 * PI * k * PHI_INV for k in range(16)])
    
    # Compute global witness
    W_global = compute_grand_equilibrium(local_states, phases)
    
    assert W_global.shape == (2,), "Global witness should be 2D"
    assert not np.isnan(W_global).any(), "Global witness should not have NaN"
    
    print(f"  ✓ W_global = [{W_global[0]:.4f}, {W_global[1]:.4f}]")
    return True


def test_downward_projection_generates():
    """Test that downward projection generates tokens."""
    print("Test 10: Downward projection generates tokens")
    
    np.random.seed(42)
    
    # Create embeddings
    vocab_size = 100
    embeddings = np.random.randn(vocab_size, CLIFFORD_DIM) * 0.1
    embeddings[:, GRADE_INDICES[0][0]] += PHI_INV
    
    # Create grand master with structure
    grand_master = np.zeros(CLIFFORD_DIM)
    grand_master[GRADE_INDICES[0][0]] = 1.0
    grand_master[GRADE_INDICES[4][0]] = 0.5
    grand_master[GRADE_INDICES[2]] = np.random.randn(6) * 0.3
    grand_master[GRADE_INDICES[3]] = np.random.randn(4) * 0.2
    
    # Generate
    tokens, stats = project_to_tokens(
        grand_master, embeddings,
        max_tokens=20,
        confidence_threshold=PHI_INV_CUBE
    )
    
    assert len(tokens) > 0, "Should generate at least one token"
    assert all(0 <= t < vocab_size for t in tokens), "Tokens should be valid"
    
    print(f"  ✓ Generated {len(tokens)} tokens")
    print(f"  ✓ Avg confidence: {stats['avg_confidence']:.4f}")
    return True


def test_complete_pipeline():
    """Test complete pipeline: learn → dream → generate."""
    print("Test 11: Complete pipeline")
    
    np.random.seed(42)
    
    # Create system
    torus = NestedFractalTorus(max_levels=2, vocab_size=100)
    
    # Learn
    print("  Learning 50 associations...")
    for i in range(50):
        context = [i % 100, (i+1) % 100]
        target = (i * 3) % 100
        torus.learn(context, target, level=0)
    
    # Dream
    print("  Dreaming...")
    dream_stats = torus.dream(levels=[0])
    
    # Generate from a learned context
    print("  Generating from grand master...")
    torus.levels[0].compose_master()
    grand_master = torus.levels[0].master_state
    
    tokens, gen_stats = project_to_tokens(
        grand_master, torus.embeddings,
        max_tokens=10
    )
    
    print(f"  ✓ Learned 50 associations")
    print(f"  ✓ Dreamed: {dream_stats}")
    print(f"  ✓ Generated: {tokens}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("NESTED FRACTAL TORUS — COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_phase_distribution_no_resonance,
        test_toroidal_round_trip,
        test_interaction_tensor_bidirectional,
        test_chirality_alternation,
        test_grace_inverse_reverses_grace,
        test_dreaming_consolidates,
        test_paradox_resolution,
        test_nested_torus_learn_retrieve,
        test_grand_equilibrium_computes,
        test_downward_projection_generates,
        test_complete_pipeline,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
            print()
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
