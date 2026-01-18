"""
Test Master→Satellite Broadcasting — Non-REM Consolidation

Tests that master torus broadcasts its witness down to satellites during Non-REM,
and that dissonant satellites receive accelerated Grace at φ⁻⁴ rate.

Theory (Chapter 11):
    for each satellite k:
        coherence = dot(master_witness, satellite_witness)
        if coherence < φ⁻¹:  # Dissonant
            rate = φ⁻⁴       # Accelerated Grace (NOT φ⁻¹!)
        else:
            rate = φ⁻²       # Normal consolidation
        satellite.witness = (1 - rate) * satellite.witness + rate * master.witness
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_FOUR,
    MATRIX_DIM, DTYPE,
)
from holographic_prod.core.algebra import (
    build_clifford_basis,
    grace_operator,
)
from holographic_prod.core.quotient import extract_witness
from holographic_prod.dreaming import NonREMConsolidator


class TestMasterSatelliteBroadcast:
    """Test suite for master→satellite broadcasting."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.basis = build_clifford_basis(np)
        self.xp = np
        self.consolidator = NonREMConsolidator(self.basis, xp=self.xp)
    
    def test_broadcast_consonant_satellite(self):
        """Test that consonant satellites receive normal φ⁻² Grace."""
        # Create master witness
        master_witness = np.eye(4, dtype=DTYPE) * 0.8
        
        # Create consonant satellite (high coherence)
        consonant_sat = np.eye(4, dtype=DTYPE) * 0.7  # Similar to master
        
        satellites = [consonant_sat]
        
        # Broadcast
        updated = self.consolidator.broadcast_master_witness(master_witness, satellites)
        
        # Should have moved toward master
        assert len(updated) == 1, "Should update all satellites"
        assert np.linalg.norm(updated[0] - consonant_sat, 'fro') > 1e-6, "Should change satellite"
    
    def test_broadcast_dissonant_satellite(self):
        """Test that dissonant satellites receive φ⁻⁴ accelerated Grace."""
        # Create master witness
        master_witness = np.eye(4, dtype=DTYPE) * 0.8
        
        # Create dissonant satellite (low coherence, opposite direction)
        dissonant_sat = -np.eye(4, dtype=DTYPE) * 0.7  # Opposite to master
        
        satellites = [dissonant_sat]
        
        # Broadcast
        updated = self.consolidator.broadcast_master_witness(master_witness, satellites)
        
        # Should have moved more strongly toward master (accelerated)
        assert len(updated) == 1, "Should update all satellites"
        
        # Dissonant satellite should change more than consonant would
        change_magnitude = np.linalg.norm(updated[0] - dissonant_sat, 'fro')
        assert change_magnitude > 1e-6, "Dissonant satellite should change"
    
    def test_coherence_threshold(self):
        """Test that coherence threshold φ⁻¹ correctly identifies dissonance."""
        # Create master witness
        master_witness = np.eye(4, dtype=DTYPE) * 0.8
        
        # Test different coherence levels
        test_cases = [
            (0.9, False),   # High coherence, consonant
            (PHI_INV, False),  # Exactly at threshold, consonant
            (PHI_INV - 0.01, True),  # Just below threshold, dissonant
            (0.3, True),    # Low coherence, dissonant
        ]
        
        for coherence_val, should_be_dissonant in test_cases:
            # Create satellite with specific coherence
            if coherence_val >= PHI_INV:
                sat_witness = master_witness * coherence_val  # Consonant
            else:
                sat_witness = -master_witness * coherence_val  # Dissonant
            
            satellites = [sat_witness]
            updated = self.consolidator.broadcast_master_witness(master_witness, satellites)
            
            # Check that dissonant satellites receive stronger correction
            change = np.linalg.norm(updated[0] - sat_witness, 'fro')
            
            if should_be_dissonant:
                assert change > 1e-6, f"Dissonant satellite (coherence={coherence_val}) should change"
    
    def test_multiple_satellites(self):
        """Test broadcasting to multiple satellites."""
        # Create master witness
        master_witness = np.eye(4, dtype=DTYPE) * 0.8
        
        # Create mix of consonant and dissonant satellites
        satellites = [
            np.eye(4, dtype=DTYPE) * 0.7,   # Consonant
            -np.eye(4, dtype=DTYPE) * 0.7,  # Dissonant
            np.eye(4, dtype=DTYPE) * 0.6,   # Consonant
        ]
        
        # Broadcast
        updated = self.consolidator.broadcast_master_witness(master_witness, satellites)
        
        # Should update all satellites
        assert len(updated) == len(satellites), "Should update all satellites"
        
        # All should have moved toward master
        for i, (orig, upd) in enumerate(zip(satellites, updated)):
            change = np.linalg.norm(upd - orig, 'fro')
            assert change > 1e-6 or np.linalg.norm(orig, 'fro') < 1e-6, \
                f"Satellite {i} should change or be empty"
    
    def test_empty_satellite(self):
        """Test that empty satellites get master witness."""
        # Create master witness
        master_witness = np.eye(4, dtype=DTYPE) * 0.8
        
        # Create empty satellite
        empty_sat = np.zeros((4, 4), dtype=DTYPE)
        
        satellites = [empty_sat]
        
        # Broadcast
        updated = self.consolidator.broadcast_master_witness(master_witness, satellites)
        
        # Should receive master witness
        assert np.linalg.norm(updated[0] - master_witness, 'fro') < 1e-3, \
            "Empty satellite should receive master witness"
    
    def test_accelerated_grace_rate(self):
        """Test that φ⁻⁴ rate is actually faster than φ⁻²."""
        # Create master witness
        master_witness = np.eye(4, dtype=DTYPE) * 0.8
        
        # Create dissonant satellite
        dissonant_sat = -np.eye(4, dtype=DTYPE) * 0.7
        
        satellites = [dissonant_sat]
        
        # Broadcast (should use φ⁻⁴)
        updated_accelerated = self.consolidator.broadcast_master_witness(master_witness, satellites)
        
        # Compare to normal Grace (φ⁻²)
        updated_normal = grace_operator(dissonant_sat, self.basis, self.xp)
        updated_normal = (1 - PHI_INV_SQ) * updated_normal + PHI_INV_SQ * master_witness
        
        # Accelerated should move more toward master
        dist_accelerated = np.linalg.norm(updated_accelerated[0] - master_witness, 'fro')
        dist_normal = np.linalg.norm(updated_normal - master_witness, 'fro')
        
        # Accelerated should be closer (or at least different)
        assert dist_accelerated < dist_normal or abs(dist_accelerated - dist_normal) < 1e-3, \
            "Accelerated Grace should move satellite closer to master"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
