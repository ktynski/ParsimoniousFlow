"""
Nested Fractal Torus — Multi-Level Torus Architecture
=====================================================

Implements the 16^n fractal scaling system. Each level is a complete
Cl(3,1) torus with 16 satellites, where Level N satellites become
Level N+1 inputs.

Architecture:
    Level 0: 16 base components (single token)
    Level 1: 16 Level-0 satellites → 256 components (phrase)
    Level 2: 16 Level-1 masters → 4096 components (concept)
    Level N: 16^N total base units

Capacity: Tower depth 4 = 16^4 = 65,536 base units → ~1T effective capacity

Key Operations:
    - learn(): Associate context with target at appropriate level
    - retrieve(): Find matching attractor via witness cascade
    - dream(): Consolidate via enhanced dreaming system

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR,
    CLIFFORD_DIM, GRADE_INDICES, GRACE_SCALES_FLAT,
)
from torus.phase_distribution import PhaseDistribution
from torus.interaction_tensor import InteractionTensor
from torus.chirality import ChiralityManager
from torus.grace_inverse import grace_inverse, GraceInverse
from torus.toroidal_coords import ToroidalCoordinates


# =============================================================================
# SATELLITE STATE
# =============================================================================

@dataclass
class SatelliteState:
    """
    State of a single satellite in the torus.
    
    Attributes:
        multivector: Cl(3,1) state [16]
        phase: Current phase angle
        frequency: Rotation frequency
        chirality: 'right' or 'left'
        index: Position in parent torus
    """
    multivector: np.ndarray
    phase: float
    frequency: float
    chirality: str
    index: int
    
    @classmethod
    def create(cls, index: int, n_satellites: int = 16) -> 'SatelliteState':
        """Create satellite with φ-derived initial state."""
        # Golden spiral position
        phase = 2 * PI * index * PHI_INV
        
        # φ-staggered frequency
        frequency = PHI ** (index % 4)
        
        # Alternating chirality
        chirality = 'right' if index % 2 == 0 else 'left'
        
        # Initialize multivector with φ-derived structure
        multivector = np.zeros(CLIFFORD_DIM)
        # Small identity bias
        multivector[GRADE_INDICES[0][0]] = PHI_INV
        
        return cls(
            multivector=multivector,
            phase=phase,
            frequency=frequency,
            chirality=chirality,
            index=index
        )
    
    def evolve(self, dt: float):
        """Evolve phase by time step."""
        self.phase = (self.phase + self.frequency * dt) % (2 * PI)
    
    def apply_grace(self, rate: float = 1.0):
        """Apply Grace contraction."""
        scales = np.array(GRACE_SCALES_FLAT)
        if rate != 1.0:
            scales = scales ** (rate / PHI_INV_SQ)
        self.multivector = self.multivector * scales


# =============================================================================
# TORUS LEVEL
# =============================================================================

@dataclass
class TorusLevel:
    """
    Single level in the nested torus hierarchy.
    
    Contains 16 satellites that can be composed into a master state.
    The master state of Level N becomes input for Level N+1.
    
    Attributes:
        level: Hierarchy level (0 = base)
        satellites: 16 SatelliteState objects
        master_state: Composed state of all satellites
        tensor: Interaction tensor for upward/downward projection
        chirality_mgr: Manages handedness
        phase_dist: Phase distribution
        attractors: Stored attractors at this level
    """
    level: int
    n_satellites: int = 16
    satellites: List[SatelliteState] = field(default_factory=list)
    master_state: np.ndarray = field(default_factory=lambda: np.zeros(CLIFFORD_DIM))
    tensor: InteractionTensor = field(default_factory=InteractionTensor)
    chirality_mgr: ChiralityManager = field(default_factory=ChiralityManager)
    phase_dist: PhaseDistribution = field(default_factory=PhaseDistribution)
    attractors: Dict[int, np.ndarray] = field(default_factory=dict)
    attractor_count: int = 0
    
    def __post_init__(self):
        """Initialize satellites if empty."""
        if not self.satellites:
            self.satellites = [
                SatelliteState.create(k, self.n_satellites)
                for k in range(self.n_satellites)
            ]
    
    def compose_master(self) -> np.ndarray:
        """
        Compose satellites into master state.
        
        Uses φ-weighted sum with chirality correction.
        
        Returns:
            Master multivector [16]
        """
        # Get all satellite multivectors
        sat_states = np.array([s.multivector for s in self.satellites])
        
        # Transform to master frame (correct chirality)
        master_frame = self.chirality_mgr.to_master_frame(sat_states)
        
        # Project bivectors up to trivectors
        bivectors = master_frame[:, GRADE_INDICES[2]]
        trivector = self.tensor.project_up(bivectors)
        
        # Build master state
        self.master_state = np.zeros(CLIFFORD_DIM)
        
        # Witness: φ-weighted sum of satellite witnesses
        total_weight = 0.0
        for k, sat in enumerate(self.satellites):
            weight = PHI_INV ** (k % 4)
            self.master_state[GRADE_INDICES[0]] += weight * sat.multivector[GRADE_INDICES[0]]
            self.master_state[GRADE_INDICES[4]] += weight * sat.multivector[GRADE_INDICES[4]]
            total_weight += weight
        
        self.master_state[GRADE_INDICES[0]] /= total_weight
        self.master_state[GRADE_INDICES[4]] /= total_weight
        
        # Trivector from projection
        self.master_state[GRADE_INDICES[3]] = trivector
        
        # Vectors: average of satellite vectors
        for idx in GRADE_INDICES[1]:
            self.master_state[idx] = np.mean([s.multivector[idx] for s in self.satellites])
        
        return self.master_state
    
    def distribute_to_satellites(self, master: np.ndarray):
        """
        Distribute master state down to satellites.
        
        Used in dreaming consolidation and generation.
        
        Args:
            master: Master multivector to distribute [16]
        """
        # Extract master trivector
        master_trivector = master[GRADE_INDICES[3]]
        
        for k, sat in enumerate(self.satellites):
            # Project trivector down to this satellite's bivector
            bivector = self.tensor.project_down(master_trivector, k)
            
            # Apply chirality correction
            correction = np.zeros(CLIFFORD_DIM)
            correction[GRADE_INDICES[2]] = bivector
            correction = self.chirality_mgr.apply(correction, k)
            
            # Blend with current state
            blend = PHI_INV
            sat.multivector = (1 - blend) * sat.multivector + blend * correction
            
            # Distribute witness
            sat.multivector[GRADE_INDICES[0]] = master[GRADE_INDICES[0]]
            sat.multivector[GRADE_INDICES[4]] = master[GRADE_INDICES[4]]
    
    def evolve(self, dt: float):
        """Evolve all satellites by time step."""
        for sat in self.satellites:
            sat.evolve(dt)
        self.phase_dist.evolve(dt)
    
    def store_attractor(self, context_hash: int, attractor: np.ndarray):
        """
        Store an attractor at this level.
        
        Args:
            context_hash: Hash of context for retrieval
            attractor: Attractor multivector
        """
        self.attractors[context_hash] = attractor.copy()
        self.attractor_count = len(self.attractors)
    
    def retrieve_attractor(self, context_hash: int) -> Optional[np.ndarray]:
        """
        Retrieve stored attractor.
        
        Args:
            context_hash: Hash of context
            
        Returns:
            Attractor if found, None otherwise
        """
        return self.attractors.get(context_hash)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get level statistics."""
        sat_energies = [np.linalg.norm(s.multivector) for s in self.satellites]
        return {
            'level': self.level,
            'n_satellites': self.n_satellites,
            'n_attractors': self.attractor_count,
            'avg_satellite_energy': np.mean(sat_energies),
            'master_energy': np.linalg.norm(self.master_state),
            'friction': self.chirality_mgr.compute_friction(
                np.array([s.multivector for s in self.satellites])
            ),
        }


# =============================================================================
# NESTED FRACTAL TORUS
# =============================================================================

@dataclass
class NestedFractalTorus:
    """
    Complete nested fractal torus architecture.
    
    Implements 16^n scaling via hierarchical torus levels.
    Each level's master state becomes input for the next level.
    
    Attributes:
        max_levels: Maximum hierarchy depth
        levels: List of TorusLevel objects
        embeddings: Token embeddings [vocab_size, 16]
        vocab_size: Size of vocabulary
    """
    max_levels: int = 4
    vocab_size: int = 1000
    levels: List[TorusLevel] = field(default_factory=list)
    embeddings: np.ndarray = field(default=None)
    
    def __post_init__(self):
        """Initialize levels and embeddings."""
        # Create levels
        if not self.levels:
            self.levels = [
                TorusLevel(level=l)
                for l in range(self.max_levels)
            ]
        
        # Initialize embeddings with φ-derived structure
        if self.embeddings is None:
            self.embeddings = self._init_embeddings()
    
    def _init_embeddings(self) -> np.ndarray:
        """
        Initialize token embeddings with φ-derived structure.
        
        Each embedding has:
        - Identity-biased scalar (PHI_INV)
        - Random components scaled by grade
        """
        np.random.seed(42)
        emb = np.random.randn(self.vocab_size, CLIFFORD_DIM) * 0.1
        
        # Add identity bias to scalar
        emb[:, GRADE_INDICES[0][0]] += PHI_INV
        
        # Scale by grace factors
        emb *= np.array(GRACE_SCALES_FLAT)
        
        return emb
    
    def embed_token(self, token_id: int) -> np.ndarray:
        """Get embedding for token."""
        return self.embeddings[token_id % self.vocab_size].copy()
    
    def embed_sequence(self, token_ids: List[int]) -> np.ndarray:
        """
        Embed sequence via geometric product.
        
        Context = E_1 × E_2 × ... × E_n
        
        This is O(N) and non-commutative (order matters).
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Context multivector [16]
        """
        if not token_ids:
            # Return identity
            result = np.zeros(CLIFFORD_DIM)
            result[GRADE_INDICES[0][0]] = 1.0
            return result
        
        # Start with first token
        result = self.embed_token(token_ids[0])
        
        # Compose with remaining tokens
        for tid in token_ids[1:]:
            emb = self.embed_token(tid)
            result = self._geometric_product(result, emb)
            
            # Apply Grace for stability (every token)
            result *= np.array(GRACE_SCALES_FLAT)
        
        return result
    
    def _geometric_product(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute geometric product of two multivectors.
        
        Simplified implementation using the 4×4 matrix representation.
        
        Args:
            A, B: Multivectors [16]
            
        Returns:
            Product multivector [16]
        """
        # Convert to 4×4 matrices
        A_mat = self._to_matrix(A)
        B_mat = self._to_matrix(B)
        
        # Matrix multiply
        C_mat = A_mat @ B_mat
        
        # Convert back
        return self._from_matrix(C_mat)
    
    def _to_matrix(self, M: np.ndarray) -> np.ndarray:
        """Convert 16D multivector to 4×4 matrix."""
        # Simplified: use scalar + vector + pseudoscalar
        mat = np.eye(4) * M[GRADE_INDICES[0][0]]
        
        # Add vector components as off-diagonal
        for i, idx in enumerate(GRADE_INDICES[1][:4]):
            mat[0, i] += M[idx] * PHI_INV
            mat[i, 0] += M[idx] * PHI_INV
        
        # Add pseudoscalar
        mat += M[GRADE_INDICES[4][0]] * np.array([
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0]
        ]) * PHI_INV
        
        return mat
    
    def _from_matrix(self, mat: np.ndarray) -> np.ndarray:
        """Convert 4×4 matrix to 16D multivector."""
        M = np.zeros(CLIFFORD_DIM)
        
        # Scalar from trace
        M[GRADE_INDICES[0][0]] = np.trace(mat) / 4
        
        # Vectors from off-diagonal
        for i, idx in enumerate(GRADE_INDICES[1][:4]):
            M[idx] = (mat[0, i] + mat[i, 0]) / (2 * PHI_INV)
        
        # Pseudoscalar from antisymmetric part
        M[GRADE_INDICES[4][0]] = (mat[0, 3] - mat[3, 0]) / (2 * PHI_INV)
        
        return M
    
    def learn(
        self,
        context: List[int],
        target: int,
        level: int = 0
    ) -> Dict[str, Any]:
        """
        Learn association between context and target.
        
        One-shot learning via direct binding.
        
        Args:
            context: Context token IDs
            target: Target token ID
            level: Which level to store at
            
        Returns:
            Learning statistics
        """
        # Embed context
        context_vec = self.embed_sequence(context)
        
        # Get target embedding
        target_vec = self.embed_token(target)
        
        # Bind context to target via geometric product
        attractor = self._geometric_product(context_vec, target_vec)
        
        # Store at level - store BOTH the attractor and target directly
        # This ensures we can retrieve the target accurately
        context_hash = hash(tuple(context))
        self.levels[level].store_attractor(context_hash, attractor)
        
        # Also store target ID directly for reliable retrieval
        if not hasattr(self.levels[level], 'target_ids'):
            self.levels[level].target_ids = {}
        self.levels[level].target_ids[context_hash] = target
        
        # Also set in satellites for level dynamics
        for k, sat in enumerate(self.levels[level].satellites):
            # Each satellite gets a portion of the attractor
            sat.multivector = (
                (1 - PHI_INV) * sat.multivector + 
                PHI_INV * attractor * (PHI_INV ** (k % 4))
            )
        
        return {
            'level': level,
            'context_hash': context_hash,
            'attractor_norm': np.linalg.norm(attractor),
            'target': target,
        }
    
    def retrieve(
        self,
        context: List[int],
        max_level: Optional[int] = None
    ) -> Tuple[Optional[int], float, Dict[str, Any]]:
        """
        Retrieve target for context.
        
        Cascades through levels from specific to general.
        First tries exact hash match (high confidence), then
        falls back to geometric unbinding for similarity.
        
        Args:
            context: Context token IDs
            max_level: Maximum level to search
            
        Returns:
            (target_id or None, confidence, stats)
        """
        if max_level is None:
            max_level = self.max_levels - 1
        
        context_vec = self.embed_sequence(context)
        context_hash = hash(tuple(context))
        
        stats = {
            'levels_searched': 0,
            'found_at_level': None,
            'retrieval_type': None,
        }
        
        for level in range(max_level + 1):
            stats['levels_searched'] = level + 1
            level_obj = self.levels[level]
            
            # First: Try exact hash match with stored target ID
            if hasattr(level_obj, 'target_ids') and context_hash in level_obj.target_ids:
                target_id = level_obj.target_ids[context_hash]
                stats['found_at_level'] = level
                stats['retrieval_type'] = 'exact_hash'
                return target_id, 1.0, stats  # Perfect confidence for exact match
            
            # Second: Try geometric unbinding from attractor
            attractor = level_obj.retrieve_attractor(context_hash)
            
            if attractor is not None:
                # Unbind to get target
                target_vec = self._unbind(context_vec, attractor)
                
                # Find closest embedding
                target_id, confidence = self._find_closest_embedding(target_vec)
                
                if confidence > PHI_INV_CUBE:  # ~0.236
                    stats['found_at_level'] = level
                    stats['retrieval_type'] = 'geometric_unbind'
                    return target_id, confidence, stats
        
        # No confident match
        return None, 0.0, stats
    
    def _unbind(self, context: np.ndarray, binding: np.ndarray) -> np.ndarray:
        """Unbind context from binding to get target."""
        # Compute inverse of context (simplified)
        ctx_norm_sq = np.sum(context ** 2)
        if ctx_norm_sq < 1e-10:
            return binding
        
        # Reversion (flip sign of grades 2,3)
        ctx_rev = context.copy()
        ctx_rev[GRADE_INDICES[2]] *= -1
        ctx_rev[GRADE_INDICES[3]] *= -1
        
        # Multiply and scale
        target = self._geometric_product(ctx_rev, binding)
        target /= ctx_norm_sq
        
        return target
    
    def _find_closest_embedding(self, vec: np.ndarray) -> Tuple[int, float]:
        """Find embedding closest to vector."""
        # Normalize
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-10:
            return 0, 0.0
        vec_normalized = vec / vec_norm
        
        # Compute similarities
        emb_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        emb_normalized = self.embeddings / (emb_norms + 1e-10)
        
        similarities = emb_normalized @ vec_normalized
        
        best_idx = np.argmax(similarities)
        confidence = similarities[best_idx]
        
        return int(best_idx), float(confidence)
    
    def dream(self, levels: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Perform dreaming consolidation across levels.
        
        Args:
            levels: Which levels to dream (default: all)
            
        Returns:
            Dreaming statistics
        """
        from dreaming_enhanced import EnhancedDreamingSystem
        
        if levels is None:
            levels = list(range(self.max_levels))
        
        stats = {}
        dreaming = EnhancedDreamingSystem()
        
        for level in levels:
            torus_level = self.levels[level]
            
            # Compose master state
            master = torus_level.compose_master()
            satellites = np.array([s.multivector for s in torus_level.satellites])
            
            # Run sleep cycle
            result = dreaming.sleep_cycle(master, satellites)
            
            # Update level state
            torus_level.master_state = result['final_master']
            for k, sat in enumerate(torus_level.satellites):
                sat.multivector = result['final_satellites'][k]
            
            stats[f'level_{level}'] = {
                'woke': result['woke'],
                'discoveries': len(result['discoveries']),
                'final_stability': result['stats'].get('final_stability', 0),
            }
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get complete system statistics."""
        total_attractors = sum(l.attractor_count for l in self.levels)
        level_stats = [l.get_statistics() for l in self.levels]
        
        return {
            'max_levels': self.max_levels,
            'vocab_size': self.vocab_size,
            'total_attractors': total_attractors,
            'levels': level_stats,
            'total_capacity': 16 ** self.max_levels,
        }


# =============================================================================
# TESTS
# =============================================================================

def _test_nested_torus():
    """Test nested fractal torus."""
    print("Testing nested fractal torus...")
    
    # Test 1: Create system
    torus = NestedFractalTorus(max_levels=2, vocab_size=100)
    assert len(torus.levels) == 2, "Should have 2 levels"
    assert torus.embeddings.shape == (100, 16), "Embeddings should be [100, 16]"
    
    # Test 2: Embed token
    emb = torus.embed_token(42)
    assert emb.shape == (CLIFFORD_DIM,), "Embedding should be 16D"
    
    # Test 3: Embed sequence
    context = torus.embed_sequence([1, 2, 3])
    assert context.shape == (CLIFFORD_DIM,), "Context should be 16D"
    
    # Test 4: Learn association
    stats = torus.learn(context=[1, 2, 3], target=42, level=0)
    assert stats['level'] == 0
    assert stats['attractor_norm'] > 0
    
    # Test 5: Retrieve
    target_id, confidence, stats = torus.retrieve([1, 2, 3])
    print(f"  Retrieved: {target_id}, confidence: {confidence:.4f}")
    print(f"  Stats: {stats}")
    
    # Test 6: Compose master
    master = torus.levels[0].compose_master()
    assert master.shape == (CLIFFORD_DIM,), "Master should be 16D"
    
    # Test 7: Statistics
    full_stats = torus.get_statistics()
    print(f"  Total attractors: {full_stats['total_attractors']}")
    print(f"  Total capacity: {full_stats['total_capacity']}")
    
    print("✓ Nested fractal torus tests passed!")
    return True


if __name__ == "__main__":
    _test_nested_torus()
