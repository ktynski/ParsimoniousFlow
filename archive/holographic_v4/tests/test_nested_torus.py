"""
Test suite for nested fractal torus architecture.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from holographic_v4.constants import (
    PHI_INV, CLIFFORD_DIM, GRADE_INDICES,
)
from holographic_v4.fractal.nested_torus import (
    NestedFractalTorus,
    TorusLevel,
    SatelliteState,
)


class TestSatelliteState:
    """Tests for satellite state."""
    
    def test_create_satellite(self):
        """Verify satellite creation with φ-derived structure."""
        sat = SatelliteState.create(index=5, n_satellites=16)
        
        assert sat.index == 5
        assert sat.chirality == 'left'  # Odd index
        assert sat.multivector.shape == (CLIFFORD_DIM,)
    
    def test_satellite_evolution(self):
        """Verify satellite phase evolves."""
        sat = SatelliteState.create(index=0, n_satellites=16)
        initial_phase = sat.phase
        
        sat.evolve(dt=0.1)
        
        assert sat.phase != initial_phase


class TestTorusLevel:
    """Tests for torus level."""
    
    def test_level_initialization(self):
        """Verify level creates 16 satellites."""
        level = TorusLevel(level=0)
        
        assert len(level.satellites) == 16
        assert level.level == 0
    
    def test_compose_master(self):
        """Verify master composition produces valid state."""
        level = TorusLevel(level=0)
        master = level.compose_master()
        
        assert master.shape == (CLIFFORD_DIM,)
        assert not np.isnan(master).any()
    
    def test_store_retrieve_attractor(self):
        """Verify attractor storage and retrieval."""
        level = TorusLevel(level=0)
        
        np.random.seed(42)
        attractor = np.random.randn(CLIFFORD_DIM)
        context_hash = 12345
        
        level.store_attractor(context_hash, attractor)
        retrieved = level.retrieve_attractor(context_hash)
        
        assert np.allclose(attractor, retrieved)


class TestNestedFractalTorus:
    """Tests for nested fractal torus."""
    
    def test_initialization(self):
        """Verify initialization creates correct structure."""
        torus = NestedFractalTorus(max_levels=2, vocab_size=100)
        
        assert len(torus.levels) == 2
        assert torus.embeddings.shape == (100, CLIFFORD_DIM)
    
    def test_embed_token(self):
        """Verify token embedding."""
        torus = NestedFractalTorus(max_levels=2, vocab_size=100)
        
        emb = torus.embed_token(42)
        
        assert emb.shape == (CLIFFORD_DIM,)
        assert emb[GRADE_INDICES[0][0]] > 0  # Identity bias
    
    def test_embed_sequence(self):
        """Verify sequence embedding."""
        torus = NestedFractalTorus(max_levels=2, vocab_size=100)
        
        context = torus.embed_sequence([1, 2, 3])
        
        assert context.shape == (CLIFFORD_DIM,)
        assert not np.isnan(context).any()
    
    def test_learn_and_retrieve_exact(self):
        """Verify exact retrieval works."""
        torus = NestedFractalTorus(max_levels=2, vocab_size=100)
        
        # Learn
        context = [1, 2, 3]
        target = 42
        torus.learn(context, target, level=0)
        
        # Retrieve
        retrieved, confidence, stats = torus.retrieve(context)
        
        assert retrieved == target
        assert confidence == 1.0
        assert stats['retrieval_type'] == 'exact_hash'
    
    def test_learn_multiple_associations(self):
        """Verify multiple associations can be stored."""
        torus = NestedFractalTorus(max_levels=2, vocab_size=100)
        
        # Learn multiple
        for i in range(10):
            context = [i, i+1, i+2]
            target = (i * 7) % 100
            torus.learn(context, target, level=0)
        
        # Verify all retrievable
        correct = 0
        for i in range(10):
            context = [i, i+1, i+2]
            expected = (i * 7) % 100
            retrieved, _, _ = torus.retrieve(context)
            if retrieved == expected:
                correct += 1
        
        assert correct == 10
    
    def test_statistics(self):
        """Verify statistics are computed."""
        torus = NestedFractalTorus(max_levels=2, vocab_size=100)
        
        torus.learn([1, 2, 3], 42, level=0)
        
        stats = torus.get_statistics()
        
        assert stats['max_levels'] == 2
        assert stats['vocab_size'] == 100
        assert stats['total_attractors'] == 1
        assert stats['total_capacity'] == 256  # 16²
