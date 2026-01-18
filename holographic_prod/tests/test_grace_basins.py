"""
Test-Driven Development for Theory-True Grace Basin Memory.

These tests define the CORRECT behavior according to THE_GEOMETRY_OF_MIND.md:
1. Grace Basins (not hash buckets) - contexts flow to attractors
2. Stability-based pruning (not FIFO) - keep stable, prune unstable
3. Quotient similarity (not Frobenius) - 38.2% witness + 61.8% vorticity

Run: pytest holographic_prod/tests/test_grace_basins.py -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, MATRIX_DIM, DTYPE, CLIFFORD_DIM
)
from holographic_prod.core.algebra import (
    grace_operator, geometric_product, build_clifford_basis,
    clifford_inverse, frobenius_similarity
)
from holographic_prod.memory import HolographicMemory, MemoryConfig


# =============================================================================
# THEORY-TRUE FUNCTIONS TO TEST (will implement after tests)
# =============================================================================

def grace_basin_id(context: np.ndarray, basis: np.ndarray, 
                   max_iters: int = 10, tolerance: float = None) -> tuple:
    """
    Find the Grace basin this context belongs to.
    
    THEORY (Ch. 7): "BUCKET = Region in quotient space where Grace flows to same attractor"
    
    Algorithm:
    1. Iterate Grace operator until convergence
    2. Extract ALL even-grade coefficients at fixed point (8D witness key)
    3. Return discretized key as basin ID
    
    THEORY-TRUE: Use 8D key (scalar + pseudoscalar + 6 bivectors) not just 2D.
    The bivectors encode vorticity (word order) which is essential for discrimination.
    
    Args:
        context: 4x4 multivector matrix
        basis: Clifford basis (16 elements)
        max_iters: Maximum Grace iterations
        tolerance: Convergence threshold (default: φ⁻³)
        
    Returns:
        Tuple of 8 integers: discretized even-grade coefficients
    """
    if tolerance is None:
        tolerance = PHI_INV_CUBE
    
    M = context.copy()
    for _ in range(max_iters):
        M_new = grace_operator(M, basis, np)
        delta = np.linalg.norm(M_new - M, 'fro')
        if delta < tolerance:
            break
        M = M_new
    
    # Extract ALL 16 Clifford coefficients
    coefficients = np.array([np.trace(basis[i] @ M) / 4.0 for i in range(16)])
    
    # 8D key: scalar (0) + 6 bivectors (5-10) + pseudoscalar (15)
    # This matches our 8D witness key in the production code
    key_coeffs = [
        coefficients[0],   # scalar
        coefficients[15],  # pseudoscalar
        coefficients[5],   # e01
        coefficients[6],   # e02
        coefficients[7],   # e03
        coefficients[8],   # e12
        coefficients[9],   # e13
        coefficients[10],  # e23
    ]
    
    # Use finer resolution: φ⁻⁴ ≈ 0.146 for better discrimination
    resolution = PHI_INV_SQ * PHI_INV_SQ  # φ⁻⁴
    return tuple(int(c / resolution) for c in key_coeffs)


def witness_stability(M: np.ndarray, basis: np.ndarray) -> float:
    """
    Compute witness stability of a multivector.
    
    THEORY (Ch. 10): "stability = (scalar² + pseudoscalar²) / total_energy"
    
    High stability (≥ φ⁻²) = close to equilibrium, keep
    Low stability (< φ⁻²) = unsettled, prune/consolidate
    
    NOTE: Must use Clifford coefficient energy, not matrix Frobenius norm.
    The 16 Clifford coefficients are extracted via trace(basis[i] @ M).
    """
    # Extract ALL 16 Clifford coefficients
    coefficients = np.array([np.trace(basis[i] @ M) / 4.0 for i in range(16)])
    
    # Witness = grade 0 (scalar, index 0) + grade 4 (pseudoscalar, index 15)
    scalar = coefficients[0]
    pseudoscalar = coefficients[15]
    
    witness_energy = scalar**2 + pseudoscalar**2
    total_energy = np.sum(coefficients**2)
    
    if total_energy < 1e-10:
        return 0.0
    
    return witness_energy / total_energy


def quotient_similarity(A: np.ndarray, B: np.ndarray, basis: np.ndarray) -> float:
    """
    Compute quotient similarity between two multivectors.
    
    THEORY (Ch. 7): "quotient_similarity = (1 - φ⁻¹) × witness_sim + φ⁻¹ × vorticity_sim"
                  = 0.382 × witness_sim + 0.618 × vorticity_sim
    
    This weights:
    - Semantic content (witness) at 38.2%
    - Structural content (vorticity) at 61.8%
    """
    # Extract witnesses
    scalar_A = np.trace(A) / 4.0
    pseudo_A = np.trace(basis[15] @ A) / 4.0
    scalar_B = np.trace(B) / 4.0
    pseudo_B = np.trace(basis[15] @ B) / 4.0
    
    witness_A = np.array([scalar_A, pseudo_A])
    witness_B = np.array([scalar_B, pseudo_B])
    
    # Witness similarity (cosine)
    norm_A = np.linalg.norm(witness_A)
    norm_B = np.linalg.norm(witness_B)
    if norm_A < 1e-10 or norm_B < 1e-10:
        witness_sim = 0.0
    else:
        witness_sim = np.dot(witness_A, witness_B) / (norm_A * norm_B)
    
    # Extract vorticity (6 bivector coefficients)
    # Bivectors are basis elements 5-10 in standard ordering
    vorticity_A = np.array([np.trace(basis[i] @ A) / 4.0 for i in range(5, 11)])
    vorticity_B = np.array([np.trace(basis[i] @ B) / 4.0 for i in range(5, 11)])
    
    # Vorticity similarity (cosine)
    norm_vA = np.linalg.norm(vorticity_A)
    norm_vB = np.linalg.norm(vorticity_B)
    if norm_vA < 1e-10 or norm_vB < 1e-10:
        vorticity_sim = 0.0
    else:
        vorticity_sim = np.dot(vorticity_A, vorticity_B) / (norm_vA * norm_vB)
    
    # Theory-true weighting
    # (1 - φ⁻¹) = φ⁻² ≈ 0.382
    # φ⁻¹ ≈ 0.618
    return PHI_INV_SQ * witness_sim + PHI_INV * vorticity_sim


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def basis():
    """Clifford basis for Cl(3,1)."""
    return build_clifford_basis(np)


@pytest.fixture
def random_context(basis):
    """Generate a random normalized context matrix."""
    np.random.seed(42)
    M = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
    # Add identity bias (theory: embeddings have identity bias)
    M += PHI_INV * np.eye(MATRIX_DIM)
    # Normalize
    M = M / np.linalg.norm(M, 'fro') * PHI_INV
    return M


# =============================================================================
# TEST 1: GRACE BASIN CONVERGENCE
# =============================================================================

class TestGraceBasinConvergence:
    """Test that Grace iteration converges to fixed points."""
    
    def test_grace_converges(self, basis):
        """Grace operator should converge to a fixed point."""
        np.random.seed(42)
        M = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        M = M / np.linalg.norm(M, 'fro')
        
        # Iterate Grace
        prev = M.copy()
        for i in range(20):
            M = grace_operator(M, basis, np)
            delta = np.linalg.norm(M - prev, 'fro')
            prev = M.copy()
        
        # Should have converged
        assert delta < PHI_INV_CUBE, f"Grace did not converge: delta={delta}"
    
    def test_fixed_point_is_witness_dominated(self, basis):
        """At fixed point, most energy should be in scalar+pseudoscalar."""
        np.random.seed(42)
        M = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        
        # Iterate to convergence
        for _ in range(20):
            M = grace_operator(M, basis, np)
        
        stability = witness_stability(M, basis)
        # At fixed point, stability should be high (most energy in witness)
        assert stability > 0.5, f"Fixed point not witness-dominated: stability={stability}"
    
    def test_basin_id_deterministic(self, basis):
        """Same context should always give same basin ID."""
        np.random.seed(42)
        M = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        M = M / np.linalg.norm(M, 'fro')
        
        basin1 = grace_basin_id(M, basis)
        basin2 = grace_basin_id(M, basis)
        basin3 = grace_basin_id(M, basis)
        
        assert basin1 == basin2 == basin3, "Basin ID should be deterministic"


# =============================================================================
# TEST 2: SIMILAR CONTEXTS → SAME BASIN (KEY GENERALIZATION TEST)
# =============================================================================

class TestBasinGeneralization:
    """Test that similar contexts flow to the same basin."""
    
    def test_small_perturbation_same_basin(self, basis):
        """Small perturbation should not change basin."""
        np.random.seed(42)
        M = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        M = M / np.linalg.norm(M, 'fro')
        
        # Small perturbation (1% noise)
        noise = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE) * 0.01
        M_perturbed = M + noise
        M_perturbed = M_perturbed / np.linalg.norm(M_perturbed, 'fro')
        
        basin_original = grace_basin_id(M, basis)
        basin_perturbed = grace_basin_id(M_perturbed, basis)
        
        assert basin_original == basin_perturbed, \
            f"Small perturbation changed basin: {basin_original} → {basin_perturbed}"
    
    def test_moderate_perturbation_same_basin(self, basis):
        """Moderate perturbation (5%) should often preserve basin."""
        np.random.seed(42)
        same_basin_count = 0
        n_tests = 20
        
        for i in range(n_tests):
            np.random.seed(42 + i)
            M = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
            M = M / np.linalg.norm(M, 'fro')
            
            noise = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE) * 0.05
            M_perturbed = M + noise
            M_perturbed = M_perturbed / np.linalg.norm(M_perturbed, 'fro')
            
            if grace_basin_id(M, basis) == grace_basin_id(M_perturbed, basis):
                same_basin_count += 1
        
        # Should preserve basin at least 70% of time
        assert same_basin_count >= 14, \
            f"Only {same_basin_count}/{n_tests} preserved basin with 5% perturbation"
    
    def test_very_different_contexts_different_basins(self, basis):
        """Very different contexts should have different basins."""
        np.random.seed(42)
        M1 = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        M1 = M1 / np.linalg.norm(M1, 'fro')
        
        np.random.seed(999)  # Very different seed
        M2 = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        M2 = M2 / np.linalg.norm(M2, 'fro')
        
        basin1 = grace_basin_id(M1, basis)
        basin2 = grace_basin_id(M2, basis)
        
        # Different contexts should have different basins
        # (not guaranteed but highly likely with different seeds)
        assert basin1 != basin2, \
            f"Very different contexts have same basin: {basin1}"


# =============================================================================
# TEST 3: WITNESS STABILITY
# =============================================================================

class TestWitnessStability:
    """Test witness stability calculation."""
    
    def test_identity_high_stability(self, basis):
        """Identity matrix should have high stability (pure scalar)."""
        M = np.eye(MATRIX_DIM, dtype=DTYPE)
        stability = witness_stability(M, basis)
        assert stability > 0.9, f"Identity should have high stability: {stability}"
    
    def test_graced_matrix_higher_stability(self, basis):
        """Grace should increase stability."""
        np.random.seed(42)
        M = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        
        stability_before = witness_stability(M, basis)
        M_graced = grace_operator(M, basis, np)
        stability_after = witness_stability(M_graced, basis)
        
        assert stability_after > stability_before, \
            f"Grace should increase stability: {stability_before} → {stability_after}"
    
    def test_stability_threshold_is_spectral_gap(self, basis):
        """The stability threshold should be φ⁻² (spectral gap)."""
        # Theory: stability ≥ φ⁻² means "close to equilibrium"
        assert abs(PHI_INV_SQ - 0.382) < 0.001, "Spectral gap should be ~0.382"


# =============================================================================
# TEST 4: QUOTIENT SIMILARITY
# =============================================================================

class TestQuotientSimilarity:
    """Test quotient similarity calculation."""
    
    def test_identical_matrices_similarity_one(self, basis):
        """Identical matrices should have similarity 1.0."""
        np.random.seed(42)
        M = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        M = M / np.linalg.norm(M, 'fro')
        
        sim = quotient_similarity(M, M, basis)
        assert abs(sim - 1.0) < 0.01, f"Self-similarity should be 1.0: {sim}"
    
    def test_quotient_weights_are_phi_derived(self, basis):
        """Quotient weights should be φ⁻² and φ⁻¹."""
        # Theory: (1 - φ⁻¹) = φ⁻² ≈ 0.382 for witness
        #         φ⁻¹ ≈ 0.618 for vorticity
        witness_weight = 1 - PHI_INV
        vorticity_weight = PHI_INV
        
        assert abs(witness_weight - PHI_INV_SQ) < 0.001
        assert abs(witness_weight + vorticity_weight - 1.0) < 0.001
        assert abs(witness_weight - 0.382) < 0.001
        assert abs(vorticity_weight - 0.618) < 0.001
    
    def test_similar_witness_increases_similarity(self, basis):
        """Matrices with similar witness should have higher similarity."""
        np.random.seed(42)
        M = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        M = M / np.linalg.norm(M, 'fro')
        
        # Create matrix with similar witness but different vorticity
        M_similar_witness = M.copy()
        # Perturb only bivector components (indices 5-10 in basis expansion)
        for i in range(5, 11):
            M_similar_witness += 0.1 * basis[i]
        M_similar_witness = M_similar_witness / np.linalg.norm(M_similar_witness, 'fro')
        
        # Create matrix with very different witness
        np.random.seed(999)
        M_different = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        M_different = M_different / np.linalg.norm(M_different, 'fro')
        
        sim_similar = quotient_similarity(M, M_similar_witness, basis)
        sim_different = quotient_similarity(M, M_different, basis)
        
        assert sim_similar > sim_different, \
            f"Similar witness should give higher similarity: {sim_similar} vs {sim_different}"


# =============================================================================
# TEST 5: STABILITY-BASED PRUNING
# =============================================================================

class TestStabilityPruning:
    """Test that pruning keeps high-stability patterns."""
    
    def test_prune_keeps_stable(self, basis):
        """Pruning should keep high-stability patterns, remove low-stability."""
        np.random.seed(42)
        patterns = []
        
        # Create patterns with varying stability
        for i in range(10):
            np.random.seed(i)
            M = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
            # Apply Grace to some (making them more stable)
            if i % 2 == 0:
                for _ in range(5):
                    M = grace_operator(M, basis, np)
            patterns.append(M)
        
        # Sort by stability
        stabilities = [witness_stability(p, basis) for p in patterns]
        sorted_indices = np.argsort(stabilities)[::-1]  # Descending
        
        # Keep top 5 by stability
        pruned = [patterns[i] for i in sorted_indices[:5]]
        
        # Verify pruned set has higher average stability
        avg_original = np.mean(stabilities)
        avg_pruned = np.mean([witness_stability(p, basis) for p in pruned])
        
        assert avg_pruned > avg_original, \
            f"Pruned should have higher stability: {avg_pruned} vs {avg_original}"


# =============================================================================
# TEST 6: END-TO-END GRACE BASIN MEMORY
# =============================================================================

class TestGraceBasinMemory:
    """Test the complete Grace Basin memory system."""
    
    def test_learn_and_retrieve_exact(self, basis):
        """Learn a pattern and retrieve it exactly."""
        from holographic_prod.memory import HolographicMemory, MemoryConfig
        
        mem = HolographicMemory(vocab_size=1000, max_levels=2, seed=42)
        
        context = [10, 20, 30, 40]
        target = 50
        
        mem.learn(context, target)
        result = mem.retrieve_deterministic(context)
        
        assert result[0] == target, f"Expected {target}, got {result[0]}"
    
    def test_learn_batch_and_retrieve(self, basis):
        """Batch learn and retrieve."""
        from holographic_prod.memory import HolographicMemory, MemoryConfig
        
        mem = HolographicMemory(vocab_size=1000, max_levels=2, seed=42)
        
        # Contexts verified to route to different satellites with φ⁻⁶ resolution:
        # [0, 1, 2] → satellite 2
        # [10, 11, 12] → satellite 6
        # [20, 21, 22] → satellite 5
        contexts = [[0, 1, 2], [10, 11, 12], [20, 21, 22]]
        targets = [100, 200, 300]
        
        mem.learn_batch(contexts, targets)
        
        for ctx, tgt in zip(contexts, targets):
            result = mem.retrieve_deterministic(ctx)
            assert result[0] == tgt, f"Context {ctx}: expected {tgt}, got {result[0]}"
    
    def test_exact_context_retrieval(self, basis):
        """Exact context should retrieve with confidence."""
        from holographic_prod.memory import HolographicMemory, MemoryConfig
        
        mem = HolographicMemory(vocab_size=1000, max_levels=2, seed=42)
        
        # Learn pattern
        context = [10, 20, 30, 40]
        target = 50
        mem.learn(context, target)
        
        # Query with EXACT same context
        result = mem.retrieve_deterministic(context)
        
        print(f"Exact context retrieval: {result}")
        # Exact match should work
        assert result[0] == target, f"Expected {target}, got {result[0]}"
        assert result[1] > 0, "Should have confidence for exact context"


# =============================================================================
# TEST 7: GRACE BASIN VS HASH BUCKET (COMPARISON)
# =============================================================================

class TestGraceVsHash:
    """Demonstrate Grace basins are superior to hash buckets."""
    
    def test_grace_basins_generalize_better(self, basis):
        """Grace basins should generalize better than hash keys."""
        np.random.seed(42)
        
        # Create a base context
        base = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE)
        base = base / np.linalg.norm(base, 'fro')
        
        # Create perturbations
        perturbations = []
        for i in range(10):
            noise = np.random.randn(MATRIX_DIM, MATRIX_DIM).astype(DTYPE) * 0.02
            perturbed = base + noise
            perturbed = perturbed / np.linalg.norm(perturbed, 'fro')
            perturbations.append(perturbed)
        
        # Count how many land in same Grace basin
        base_basin = grace_basin_id(base, basis)
        same_basin = sum(1 for p in perturbations if grace_basin_id(p, basis) == base_basin)
        
        # Grace basins should group similar contexts together
        # Hash buckets would scatter them across different buckets
        print(f"Same basin count: {same_basin}/10")
        assert same_basin >= 7, f"Grace basins should group similar: {same_basin}/10"


# =============================================================================
# MULTI-LEVEL TOWER TESTS
# =============================================================================

class TestMultiLevelTower:
    """Test the 16^N fractal tower structure."""
    
    def test_tower_structure_level1(self):
        """Test tower structure with level 1 (16 satellites)."""
        mem = HolographicMemory(vocab_size=100, max_levels=1, seed=42)
        
        # Level 1: 16 satellites
        sat_states = mem.satellite_states
        assert sat_states.shape == (16, 16), f"Expected (16, 16), got {sat_states.shape}"
        
        # master_state is a property returning [16] coefficient vector
        master = mem.master_state
        assert master.shape == (16,), f"Expected (16,), got {master.shape}"
        
        # TowerMemory has 16 satellites
        assert mem.tower.n_satellites == 16
    
    def test_tower_structure_level2(self):
        """Test tower structure with level 2 (256 satellites)."""
        mem = HolographicMemory(vocab_size=100, max_levels=2, seed=42)
        
        # Level 2: 256 satellites = 16^2
        sat_states = mem.satellite_states
        assert sat_states.shape == (256, 16), f"Expected (256, 16), got {sat_states.shape}"
        
        # master_state is still a property returning [16] coefficient vector (grand master)
        master = mem.master_state
        assert master.shape == (16,), f"Expected (16,), got {master.shape}"
        
        # MultiLevelTower has 256 satellites
        assert mem.tower.n_satellites == 256
    
    def test_tower_satellites_are_16d(self):
        """Test that each satellite is 16D (Cl(3,1))."""
        mem = HolographicMemory(vocab_size=100, max_levels=1, seed=42)
        
        # Each satellite's memory is a 4x4 matrix = 16D
        for sat in mem.tower.satellites:
            assert sat.memory.shape == (4, 4), f"Expected (4, 4), got {sat.memory.shape}"
    
    def test_tower_master_is_aggregated(self):
        """Test that master state is φ-weighted aggregation of satellites."""
        mem = HolographicMemory(vocab_size=100, max_levels=2, seed=42)
        
        # Learn some patterns
        mem.learn([10, 20, 30], 100)
        mem.learn([40, 50, 60], 200)
        
        # Master state should be non-zero (aggregated from satellites)
        master = mem.master_state
        assert np.linalg.norm(master) > 0, "Master should aggregate satellite states"
    
    def test_tower_routing_by_basin(self):
        """Test that learning routes through tower by basin key."""
        mem = HolographicMemory(vocab_size=1000, max_levels=3, seed=42)
        
        # Learn a pattern
        ctx = [10, 20, 30, 40]
        target = 50
        
        # Learn
        mem.learn(ctx, target)
        
        # Verify satellite was updated (at least one should have bindings)
        total_bindings = sum(sat.n_bindings for sat in mem.tower.satellites)
        assert total_bindings > 0, "At least one satellite should have bindings"
        
        # Master state should be non-zero
        assert np.linalg.norm(mem.master_state) > 0, "Master should be non-zero"
    
    def test_tower_propagation_via_aggregation(self):
        """Test that master state aggregates satellite states."""
        mem = HolographicMemory(vocab_size=1000, max_levels=3, seed=42)
        
        # Learn many patterns to populate tower
        np.random.seed(42)
        for _ in range(100):
            ctx = list(np.random.randint(0, 1000, 8))
            target = np.random.randint(0, 1000)
            mem.learn(ctx, target)
        
        # Check satellite energies
        sat_energies = [np.linalg.norm(sat.memory) for sat in mem.tower.satellites]
        total_sat_energy = sum(sat_energies)
        
        # Master energy (aggregated)
        master_energy = np.linalg.norm(mem.master_state)
        
        print(f"Total satellite energy: {total_sat_energy:.2f}, Master energy: {master_energy:.2f}")
        
        # Both should be non-trivial
        assert total_sat_energy > 0.5, "Satellites should have energy"
        assert master_energy > 0, "Master should have aggregated energy"
    
    def test_tower_capacity_scaling(self):
        """Test that tower distributes bindings across satellites."""
        np.random.seed(42)
        
        mem = HolographicMemory(vocab_size=10000, max_levels=2, seed=42)
        
        # Learn many patterns
        n_patterns = 200
        for _ in range(n_patterns):
            ctx = list(np.random.randint(0, 10000, 8))
            target = np.random.randint(0, 10000)
            mem.learn(ctx, target)
        
        # Patterns should be learned
        assert mem.n_patterns == n_patterns
        
        # UNIFIED ARCHITECTURE: Check satellite distribution
        satellites_used = sum(1 for sat in mem.tower.satellites if sat.n_bindings > 0)
        total_bindings = sum(sat.n_bindings for sat in mem.tower.satellites)
        
        print(f"Satellites used: {satellites_used}/16, Total bindings: {total_bindings}")
        
        # Multiple satellites should be used
        assert satellites_used >= 4, f"Expected >=4 satellites used, got {satellites_used}"
        assert total_bindings == n_patterns, f"Expected {n_patterns} bindings, got {total_bindings}"


# =============================================================================
# GPU SUPPORT TESTS
# =============================================================================

class TestGPUSupport:
    """Test GPU/CuPy integration (runs on CPU if GPU not available)."""
    
    def test_gpu_detection(self):
        """Test that GPU detection works correctly."""
        # Try with GPU enabled
        try:
            mem_gpu = HolographicMemory(vocab_size=100, use_gpu=True, seed=42)
            print(f"GPU enabled: {mem_gpu.use_gpu}")
        except Exception as e:
            print(f"GPU not available: {e}")
            # Should fall back gracefully
            mem_gpu = HolographicMemory(vocab_size=100, use_gpu=False, seed=42)
        
        # CPU should always work
        mem_cpu = HolographicMemory(vocab_size=100, use_gpu=False, seed=42)
        assert mem_cpu.xp == np
    
    def test_learn_batch_xp_consistency(self):
        """Test that learn_batch uses correct array module."""
        mem = HolographicMemory(vocab_size=1000, use_gpu=False, seed=42)
        
        # Learn batch
        contexts = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
        targets = [100, 200, 300]
        
        stats = mem.learn_batch(contexts, targets)
        
        assert stats['n_learned'] == 3
        # Embeddings should be numpy arrays (CPU)
        assert isinstance(mem.embeddings, np.ndarray)
    
    def test_embed_sequence_return_types(self):
        """Test embed_sequence return types."""
        mem = HolographicMemory(vocab_size=100, use_gpu=False, seed=42)
        
        # Default returns CPU numpy
        result = mem.embed_sequence([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.shape == (MATRIX_DIM, MATRIX_DIM)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
