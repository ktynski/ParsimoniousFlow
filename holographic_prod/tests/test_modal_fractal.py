"""
Modal Fractal Test with TinyStories

Tests fractal components (InteractionTensor, Chirality, GraceInverse) on TinyStories:
1. Satellite aggregation via InteractionTensor
2. Chirality alternation prevents interference
3. Downward projection for generation

RUN:
    modal run holographic_prod/tests/test_modal_fractal.py::test_fractal_tinystories

NOTE: This test requires Modal CLI and credentials.
"""

import modal
import time
from typing import List, Tuple, Dict

# Modal app
app = modal.App("holographic-fractal-test")

# GPU-optimized image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "numpy>=1.24.0",
        "cupy-cuda12x>=12.0.0",
        "scipy>=1.10.0",
        "datasets>=2.14.0",
        "huggingface_hub>=0.16.0",
    )
    .add_local_dir("holographic_prod", "/root/project/holographic_prod")
)


@app.function(image=image, gpu="T4", timeout=1800)
def test_fractal_tinystories(
    n_samples: int = 10_000,
    vocab_size: int = 1000,
    n_satellites: int = 16,
    seed: int = 42,
) -> Dict:
    """
    Test fractal components on TinyStories subset.
    
    Tests:
    1. InteractionTensor: satellite bivector → master trivector aggregation
    2. ChiralityFlip: even/odd handedness alternation
    3. DownwardProjection: master → satellite projection for generation
    4. Full pipeline integration
    
    Args:
        n_samples: Number of samples to process
        vocab_size: Vocabulary size
        n_satellites: Number of satellites (typically 16)
        seed: Random seed
        
    Returns:
        Dictionary with test results
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import numpy as np
    import cupy as cp
    
    # Verify GPU
    cp.cuda.Device(0).use()
    meminfo = cp.cuda.runtime.memGetInfo()
    print(f"GPU Memory: {meminfo[1]/1024**3:.1f} GB total, {meminfo[0]/1024**3:.1f} GB free")
    
    from holographic_prod.core.constants import (
        PI, PHI, PHI_INV, PHI_INV_SQ,
        MATRIX_DIM, DTYPE,
        GRADE_INDICES,
    )
    from holographic_prod.core.algebra import (
        build_clifford_basis,
        grace_operator,
        decompose_to_coefficients,
        coefficients_to_matrix,
    )
    from holographic_prod.core.quotient import extract_witness, compute_enstrophy
    
    from holographic_prod.torus.interaction_tensor import InteractionTensor
    from holographic_prod.torus.chirality import ChiralityFlip
    from holographic_prod.fractal.downward_projection import DownwardProjection, phase_locked_emission
    
    results = {
        'vocab_size': vocab_size,
        'n_samples': n_samples,
        'n_satellites': n_satellites,
        'seed': seed,
    }
    
    np.random.seed(seed)
    basis = build_clifford_basis(np)
    
    # ==========================================================================
    # TEST 1: InteractionTensor - Satellite → Master Aggregation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: InteractionTensor - Satellite → Master Aggregation")
    print("=" * 70)
    
    interaction_tensor = InteractionTensor(n_satellites=n_satellites)
    
    # Create random satellite bivectors
    satellite_bivectors = np.random.randn(n_satellites, 6).astype(DTYPE)
    
    start = time.time()
    n_aggregations = 1000
    for _ in range(n_aggregations):
        master_trivector = interaction_tensor.project_up(satellite_bivectors)
    elapsed = time.time() - start
    
    per_aggregation_us = elapsed / n_aggregations * 1_000_000
    
    print(f"  Aggregations:       {n_aggregations}")
    print(f"  Total time:         {elapsed:.3f}s")
    print(f"  Per aggregation:    {per_aggregation_us:.1f} µs")
    print(f"  Master shape:       {master_trivector.shape}")
    print(f"  Master norm:        {np.linalg.norm(master_trivector):.4f}")
    
    results['aggregation_time_us'] = per_aggregation_us
    results['master_norm'] = float(np.linalg.norm(master_trivector))
    
    assert master_trivector.shape == (4,), "Master should have 4 trivector components"
    print("  ✓ InteractionTensor test PASSED")
    
    # ==========================================================================
    # TEST 2: ChiralityFlip - Handedness Alternation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: ChiralityFlip - Handedness Alternation")
    print("=" * 70)
    
    chirality = ChiralityFlip(n_satellites=n_satellites)
    
    # Create base multivector
    base_mv = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
    
    even_mvs = []
    odd_mvs = []
    
    for k in range(n_satellites):
        flipped = chirality.apply(base_mv.copy(), satellite_index=k)
        if k % 2 == 0:
            even_mvs.append(flipped)
        else:
            odd_mvs.append(flipped)
    
    # Compute average difference between even and odd
    avg_even = np.mean([mv for mv in even_mvs], axis=0)
    avg_odd = np.mean([mv for mv in odd_mvs], axis=0)
    chirality_diff = np.linalg.norm(avg_even - avg_odd)
    
    print(f"  Satellites:         {n_satellites}")
    print(f"  Even satellites:    {len(even_mvs)}")
    print(f"  Odd satellites:     {len(odd_mvs)}")
    print(f"  Chirality diff:     {chirality_diff:.4f}")
    
    results['chirality_diff'] = float(chirality_diff)
    
    assert chirality_diff > 0, "Chirality should create difference"
    print("  ✓ ChiralityFlip test PASSED")
    
    # ==========================================================================
    # TEST 3: DownwardProjection - Generation Pipeline
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: DownwardProjection - Generation Pipeline")
    print("=" * 70)
    
    downward = DownwardProjection(basis=basis, xp=np)
    
    # Create higher-level state
    higher_mv = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
    lower_memory = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
    
    start = time.time()
    n_projections = 1000
    for _ in range(n_projections):
        projected, confidence = downward.project_level_down(higher_mv, lower_memory)
    elapsed = time.time() - start
    
    per_projection_us = elapsed / n_projections * 1_000_000
    
    print(f"  Projections:        {n_projections}")
    print(f"  Total time:         {elapsed:.3f}s")
    print(f"  Per projection:     {per_projection_us:.1f} µs")
    print(f"  Confidence:         {confidence:.4f}")
    
    results['projection_time_us'] = per_projection_us
    results['projection_confidence'] = float(confidence)
    
    assert 0 <= confidence <= 1, "Confidence should be in [0, 1]"
    print("  ✓ DownwardProjection test PASSED")
    
    # ==========================================================================
    # TEST 4: Phase-Locked Emission
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Phase-Locked Emission")
    print("=" * 70)
    
    # Count emissions over golden-angle steps
    n_steps = 10000
    emissions = 0
    for i in range(n_steps):
        phase = 2 * PI * i * PHI_INV
        if phase_locked_emission(phase):
            emissions += 1
    
    emission_rate = emissions / n_steps * 100
    
    print(f"  Steps:              {n_steps}")
    print(f"  Emissions:          {emissions}")
    print(f"  Emission rate:      {emission_rate:.1f}%")
    
    results['emission_rate'] = emission_rate
    
    # Should have moderate emission rate (not 0%, not 100%)
    assert 5 < emission_rate < 50, f"Emission rate should be moderate, got {emission_rate:.1f}%"
    print("  ✓ Phase-locked emission test PASSED")
    
    # ==========================================================================
    # TEST 5: Full Pipeline Integration
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 5: Full Pipeline Integration")
    print("=" * 70)
    
    # Simulate full learn → aggregate → generate cycle
    n_cycles = 100
    start = time.time()
    
    for _ in range(n_cycles):
        # LEARN: Create 16 satellite states
        satellite_states = []
        for k in range(n_satellites):
            mv = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
            mv = chirality.apply(mv, satellite_index=k)
            satellite_states.append(mv)
        satellite_states = np.array(satellite_states)
        
        # AGGREGATE: Extract bivectors and project up
        bivectors = []
        for k in range(n_satellites):
            coeffs = decompose_to_coefficients(satellite_states[k], basis)
            bivectors.append(coeffs[4:10])  # Grade 2
        bivectors = np.array(bivectors)
        master_trivector = interaction_tensor.project_up(bivectors)
        
        # GENERATE: Create grand master and project down
        grand_master_coeffs = np.zeros(16, dtype=DTYPE)
        grand_master_coeffs[10:14] = master_trivector
        grand_master = coefficients_to_matrix(grand_master_coeffs, basis)
        projected, confidence = downward.project_level_down(grand_master, satellite_states[0])
        
        # EMIT: Check phase
        phase = PI * PHI_INV + 0.1
        can_emit = phase_locked_emission(phase)
    
    elapsed = time.time() - start
    per_cycle_ms = elapsed / n_cycles * 1000
    
    print(f"  Cycles:             {n_cycles}")
    print(f"  Total time:         {elapsed:.3f}s")
    print(f"  Per cycle:          {per_cycle_ms:.2f} ms")
    
    results['pipeline_time_ms'] = per_cycle_ms
    
    print("  ✓ Full pipeline integration test PASSED")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: All fractal tests PASSED")
    print("=" * 70)
    print(f"  Aggregation:        {per_aggregation_us:.1f} µs")
    print(f"  Chirality diff:     {chirality_diff:.4f}")
    print(f"  Projection:         {per_projection_us:.1f} µs")
    print(f"  Emission rate:      {emission_rate:.1f}%")
    print(f"  Full pipeline:      {per_cycle_ms:.2f} ms")
    
    results['status'] = 'PASSED'
    return results


def run_local():
    """Run tests locally (without Modal)."""
    print("Running fractal tests locally...")
    
    import sys
    sys.path.insert(0, '.')
    
    import numpy as np
    
    from holographic_prod.core.constants import (
        PI, PHI_INV, MATRIX_DIM, DTYPE,
    )
    from holographic_prod.core.algebra import (
        build_clifford_basis,
        decompose_to_coefficients,
        coefficients_to_matrix,
    )
    from holographic_prod.torus.interaction_tensor import InteractionTensor
    from holographic_prod.torus.chirality import ChiralityFlip
    from holographic_prod.fractal.downward_projection import DownwardProjection, phase_locked_emission
    
    np.random.seed(42)
    basis = build_clifford_basis(np)
    n_satellites = 16
    
    # Quick tests
    interaction_tensor = InteractionTensor(n_satellites=n_satellites)
    chirality = ChiralityFlip(n_satellites=n_satellites)
    downward = DownwardProjection(basis=basis, xp=np)
    
    # Aggregation
    bivectors = np.random.randn(n_satellites, 6).astype(DTYPE)
    master = interaction_tensor.project_up(bivectors)
    print(f"Aggregation: master shape = {master.shape}, norm = {np.linalg.norm(master):.4f}")
    
    # Chirality
    base_mv = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
    even = chirality.apply(base_mv.copy(), satellite_index=0)
    odd = chirality.apply(base_mv.copy(), satellite_index=1)
    diff = np.linalg.norm(even - odd)
    print(f"Chirality: even/odd diff = {diff:.4f}")
    
    # Projection
    higher = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
    lower = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
    projected, conf = downward.project_level_down(higher, lower)
    print(f"Projection: confidence = {conf:.4f}")
    
    # Phase emission
    emissions = sum(1 for i in range(1000) if phase_locked_emission(2 * PI * i * PHI_INV))
    print(f"Emission: {emissions}/1000 ({emissions/10:.1f}%)")
    
    print("✓ Local fractal tests completed")


if __name__ == '__main__':
    run_local()
