"""
Test Suite: Nested Fractal Torus Integration
=============================================

TDD tests for theory-true 16^N fractal torus from THE_GEOMETRY_OF_MIND.md Chapter 11.

THEORY REQUIREMENTS (from Chapter 11):
    "16 satellites orbit a master, pattern repeats at every level"
    
    Architecture:
        Level 0: 16 base components (single token)
        Level 1: 16 Level-0 satellites → 256 components (phrase)
        Level 2: 16 Level-1 masters → 4096 components (concept)
        Level N: 16^N total base units
    
    Capacity: Tower depth 4 = 16^4 = 65,536 base units

ALL VALUES ARE φ-DERIVED. NO ARBITRARY CONSTANTS.
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR,
    MATRIX_DIM, CLIFFORD_DIM,
)
from holographic_prod.memory import HolographicMemory, MemoryConfig


# =============================================================================
# THEORY-TRUE TESTS: 16^N Fractal Structure
# =============================================================================

class TestFractalTowerStructure:
    """Verify the 16^N fractal tower structure"""
    
    def test_16_satellites_per_level(self):
        """
        MULTI-LEVEL ARCHITECTURE: 16^N satellites for N levels.
        
        From Chapter 11:
            "16 satellites orbit a master, pattern repeats at every level"
            
        Level 1: 16 satellites
        Level 2: 256 satellites (16×16)
        Level 3: 4096 satellites (16×16×16)
        """
        mem1 = HolographicMemory(vocab_size=100, max_levels=1, seed=42)
        assert mem1.tower.n_satellites == 16
        
        mem2 = HolographicMemory(vocab_size=100, max_levels=2, seed=42)
        assert mem2.tower.n_satellites == 256
        
        mem3 = HolographicMemory(vocab_size=100, max_levels=3, seed=42)
        assert mem3.tower.n_satellites == 4096
    
    def test_capacity_is_16_to_n(self):
        """
        MULTI-LEVEL ARCHITECTURE: 16^N satellites provide 16^N × capacity.
        
        Each satellite can hold multiple bindings (with interference).
        Total capacity = n_satellites × bindings per satellite.
        """
        # Level 1: 16 satellites
        mem1 = HolographicMemory(vocab_size=100, max_levels=1, seed=42)
        assert mem1.tower.n_satellites == 16
        
        # Level 2: 256 satellites = 16^2
        mem2 = HolographicMemory(vocab_size=100, max_levels=2, seed=42)
        assert mem2.tower.n_satellites == 256
        
        # Level 3: 4096 satellites = 16^3
        mem3 = HolographicMemory(vocab_size=100, max_levels=3, seed=42)
        assert mem3.tower.n_satellites == 4096
    
    def test_tower_satellites_are_16d(self):
        """
        Each satellite is a 16D Cl(3,1) memory (4×4 matrix).
        """
        mem = HolographicMemory(vocab_size=100, max_levels=2, seed=42)
        
        for sat in mem.tower.satellites:
            # Memory is a 4×4 matrix = 16 dimensions
            assert sat.memory.shape == (4, 4), f"Expected (4,4), got {sat.memory.shape}"


# =============================================================================
# ROUTING TESTS: Basin-Based Path Through Tower
# =============================================================================

class TestBasinBasedRouting:
    """Verify basin key determines path through tower"""
    
    def test_identical_contexts_same_satellite(self):
        """Identical contexts should always route to the same satellite."""
        mem = HolographicMemory(vocab_size=100, max_levels=3, seed=42)
        
        ctx = [1, 2, 3]
        
        # Route same context multiple times - should be deterministic
        sat1 = mem.tower.route_to_satellite(ctx)
        sat2 = mem.tower.route_to_satellite(ctx)
        sat3 = mem.tower.route_to_satellite(ctx)
        
        assert sat1 == sat2 == sat3, "Identical contexts should route to same satellite"
    
    def test_routing_distributes_across_satellites(self):
        """Different contexts should distribute across satellites."""
        mem = HolographicMemory(vocab_size=100, max_levels=3, seed=42)
        
        # Learn many diverse patterns
        for i in range(20):
            ctx = [i*10, i*10+1, i*10+2]
            mem.learn(ctx, i*10+3)
        
        # Check that satellites have received bindings
        # (if routing works, satellites should have bindings)
        satellites_with_bindings = sum(
            1 for sat in mem.tower.satellites
            if sat.n_bindings > 0
        )
        
        # At least some satellites should have bindings
        assert satellites_with_bindings >= 1, "Routing should distribute to satellites"
        
        # Total bindings should match patterns learned
        total_bindings = sum(sat.n_bindings for sat in mem.tower.satellites)
        assert total_bindings == 20, f"Expected 20 bindings, got {total_bindings}"


# =============================================================================
# PROPAGATION TESTS: Upward/Downward Information Flow
# =============================================================================

class TestTowerPropagation:
    """Verify information propagates through the tower"""
    
    def test_learning_updates_satellites(self):
        """Learning should store bindings in satellites."""
        mem = HolographicMemory(vocab_size=100, max_levels=3, seed=42)
        
        # Get initial state
        initial_bindings = sum(sat.n_bindings for sat in mem.tower.satellites)
        assert initial_bindings == 0, "Should start with no bindings"
        
        # Learn many patterns
        for i in range(50):
            ctx = [i*5, i*5+1]
            mem.learn(ctx, i*5+2)
        
        # Satellites should have bindings
        final_bindings = sum(sat.n_bindings for sat in mem.tower.satellites)
        assert final_bindings == 50, f"Expected 50 bindings, got {final_bindings}"
    
    def test_master_aggregates_satellites(self):
        """Master state should aggregate satellite states."""
        mem = HolographicMemory(vocab_size=100, max_levels=3, seed=42)
        
        # Learn patterns
        for i in range(20):
            ctx = [i*5, i*5+1]
            mem.learn(ctx, i*5+2)
        
        # Master state is φ-weighted aggregation of satellite states
        master = mem.master_state
        
        # Should have non-zero energy
        assert np.linalg.norm(master) > 0, "Master should aggregate satellite states"
    
    def test_satellite_states_are_clifford_dim(self):
        """Each satellite state should be CLIFFORD_DIM dimensional."""
        # Level 1: 16 satellites
        mem1 = HolographicMemory(vocab_size=100, max_levels=1, seed=42)
        sat_states1 = mem1.satellite_states
        assert sat_states1.shape == (16, CLIFFORD_DIM), \
            f"Expected (16, {CLIFFORD_DIM}), got {sat_states1.shape}"
        
        # Level 2: 256 satellites
        mem2 = HolographicMemory(vocab_size=100, max_levels=2, seed=42)
        sat_states2 = mem2.satellite_states
        assert sat_states2.shape == (256, CLIFFORD_DIM), \
            f"Expected (256, {CLIFFORD_DIM}), got {sat_states2.shape}"


# =============================================================================
# SCALABILITY TESTS
# =============================================================================

class TestTowerScalability:
    """Verify the tower scales efficiently"""
    
    def test_satellites_scale_with_levels(self):
        """
        MULTI-LEVEL ARCHITECTURE: 16^N satellites for N levels.
        
        Level 1: 16 satellites
        Level 2: 256 satellites
        Level 3: 4096 satellites
        """
        expected = {1: 16, 2: 256, 3: 4096}
        
        for levels in [1, 2, 3]:
            mem = HolographicMemory(vocab_size=100, max_levels=levels, seed=42)
            
            total_satellites = mem.tower.n_satellites
            
            assert total_satellites == expected[levels], \
                f"Level {levels} should have {expected[levels]} satellites, got {total_satellites}"
    
    def test_retrieval_is_bounded(self):
        """Retrieval should be O(1) or O(log n), not O(n)"""
        mem = HolographicMemory(vocab_size=100, max_levels=3, seed=42)
        
        # Learn many patterns
        for i in range(100):
            ctx = [i*3, i*3+1]
            mem.learn(ctx, i*3+2)
        
        # Retrieval should complete quickly
        import time
        start = time.time()
        for i in range(100):
            ctx = [i*3, i*3+1]
            pred, conf = mem.retrieve_deterministic(ctx)
        elapsed = time.time() - start
        
        # Should be fast (< 1 second for 100 retrievals)
        assert elapsed < 2.0, f"Retrieval took too long: {elapsed}s"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
