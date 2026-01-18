# Nested Fractal Torus Architecture Specification

**Version:** v4.24.0  
**Date:** 2026-01-13  
**Status:** IMPLEMENTED AND TESTED

---

## Overview

The Nested Fractal Torus is a hierarchical memory architecture based on Clifford algebra Cl(3,1), where 16 "satellite" units orbit a "master" torus, and this structure repeats fractally at higher levels. The architecture achieves **16^N scalable capacity** with **O(N) computation**.

---

## Core Principles

### 1. The Golden Ratio φ

Everything in this architecture derives from a single equation:

```
φ² = φ + 1
φ = (1 + √5) / 2 ≈ 1.618033988749895
```

| Constant | Value | Use |
|----------|-------|-----|
| φ | 1.618 | Inflation scaling |
| φ⁻¹ | 0.618 | Contraction (Grace) |
| φ⁻² | 0.382 | Spectral gap threshold |
| 2π·φ⁻¹ | 3.883 | Golden angle (radians) |

### 2. The Torus T²

State space is the surface of a 2-torus with coordinates:
- **θ (poloidal)**: Represents **vorticity** (Grade 2 bivectors) — syntactic structure
- **ψ (toroidal)**: Represents **witness** (Grades 0,4) — semantic essence

### 3. Fractal Nesting

Each level is a complete Cl(3,1) system:
```
Level 0: 16 satellites (base units)
Level 1: 1 master (aggregates Level 0)
Level 2: 1 grand master (aggregates Level 1s)
...
Level N: 16^N total capacity
```

---

## Module Specifications

### `torus/phase_distribution.py`

**Purpose:** Distribute 16 satellites around master torus using golden ratio to prevent resonance.

**Key Functions:**

```python
def compute_satellite_positions(n_satellites: int = 16) -> np.ndarray:
    """
    Place satellites using golden angle for maximal irrationality.
    
    α_k = 2π · k · φ⁻¹ (mod 2π)
    
    Returns: [n_satellites] array of angles in [0, 2π)
    """

def compute_frequency_stagger(n_satellites: int, omega_base: float) -> np.ndarray:
    """
    Stagger frequencies using φ powers for non-resonance.
    
    ω_k = ω_base · φ^(k mod 4)
    
    Returns: [n_satellites] array of frequencies
    """

class PhaseDistribution:
    """Manages satellite phases and frequencies over time."""
    
    def evolve(self, dt: float) -> None:
        """Advance all satellite phases by dt."""
        
    def check_collisions(self, threshold: float = 0.1) -> List[Tuple[int, int]]:
        """Return pairs of satellites within threshold of each other."""
```

**Theory:** The golden angle ensures satellites never phase-lock because φ is the "most irrational" number (worst approximable by rationals).

---

### `torus/toroidal_coords.py`

**Purpose:** Map between Clifford multivector space and T² toroidal coordinates.

**Key Functions:**

```python
class ToroidalCoords:
    """Maps Cl(3,1) ↔ T² angles."""
    
    def multivector_to_angles(self, mv: np.ndarray) -> Tuple[float, float]:
        """
        Extract (θ, ψ) from multivector.
        
        θ = atan2(e01, e02)  # From bivector components
        ψ = atan2(scalar, pseudoscalar)  # From witness
        
        Returns: (theta, psi) in [-π, π]
        """
    
    def angles_to_multivector(self, theta: float, psi: float) -> np.ndarray:
        """
        Reconstruct multivector from (θ, ψ).
        
        Distributes energy into scalar, pseudoscalar, and bivectors
        based on angle values.
        """
```

---

### `torus/interaction_tensor.py`

**Purpose:** Define geometric projection from Level 0 satellite bivectors to Level 1 master trivectors.

**Key Structure:**

```python
class InteractionTensor:
    """
    16×6×4 tensor for Level 0 → Level 1 projection.
    
    - 16 satellites
    - 6 bivector components (grade 2)
    - 4 trivector components (grade 3)
    """
    
    def __init__(self):
        # Build rotation rotor for each satellite
        for k in range(16):
            R_k = build_rotation_rotor(k)  # 6×6 orthogonal
            P = build_projection_matrix()   # 4×6
            self.tensor[k] = P @ R_k * PHI_INV_SQ
    
    def project_up(self, satellite_bivectors: np.ndarray) -> np.ndarray:
        """
        Project all satellite bivectors UP to master trivectors.
        
        T_master = φ⁻² Σ_k R_k · B_k
        
        Args: [16, 6] satellite bivector array
        Returns: [4] trivector array
        """
    
    def project_down(self, master_trivectors: np.ndarray, k: int) -> np.ndarray:
        """
        Project master trivectors DOWN to satellite k's bivectors.
        
        B_k = φ⁻³ R_k^T · T_master
        
        Returns: [6] bivector array
        """
```

**Theory:** The interaction tensor encodes how local syntactic structure (bivectors) aggregates into higher-level relationships (trivectors).

---

### `torus/chirality.py`

**Purpose:** Implement even/odd handedness flip for topological friction.

**Key Functions:**

```python
def get_chirality(k: int) -> str:
    """
    Even satellites are right-handed, odd are left-handed.
    
    Returns: 'right' or 'left'
    """
    return 'right' if k % 2 == 0 else 'left'

def build_chirality_operator(handedness: str) -> np.ndarray:
    """
    Build 16×16 diagonal operator.
    
    Right: Identity (no change)
    Left: Flip signs of grades 3 and 4
    """

class ChiralityManager:
    """Manages chirality transformations for satellite array."""
    
    def to_master_frame(self, states: np.ndarray) -> np.ndarray:
        """Transform all satellites to master's reference frame."""
        
    def from_master_frame(self, states: np.ndarray) -> np.ndarray:
        """Transform back to individual chiralities."""
        
    def compute_friction(self, states: np.ndarray) -> float:
        """Compute topological friction from chirality mixing."""
```

**Theory:** Alternating chirality prevents satellites from "agreeing too much" — creates healthy tension in the system.

---

### `torus/grace_inverse.py`

**Purpose:** Implement inverse of Grace operator for generation (inflation from coherent core).

**Key Functions:**

```python
# Grace inverse scales (reciprocal of Grace)
GRACE_INVERSE_SCALES = {
    0: 1.0,      # Scalar stays anchored
    1: PHI,      # Vectors inflated by φ
    2: PHI_SQ,   # Bivectors inflated by φ² (most expansion)
    3: PHI_CUBE, # Trivectors inflated by φ³
    4: PHI,      # Pseudoscalar (Fibonacci exception)
}

def grace_inverse(M: np.ndarray) -> np.ndarray:
    """
    Inflate multivector from coherent core back to high-vorticity state.
    
    Each grade k is multiplied by φ^k (except pseudoscalar: φ^1).
    
    This is the REVERSE of Grace contraction, used for generation.
    """

class GraceInverse:
    """Class wrapper with selective inflation options."""
    
    def apply(self, M: np.ndarray) -> np.ndarray:
        """Apply full inverse Grace."""
        
    def apply_selective(self, M: np.ndarray, grades: List[int]) -> np.ndarray:
        """Apply inverse Grace only to specified grades."""
```

**Theory:** Grace contracts toward the coherent core (witness). GraceInverse re-inflates structure for fluent generation.

---

### `dreaming_enhanced.py`

**Purpose:** Implement enhanced dreaming for the nested fractal torus.

**Key Classes:**

```python
class NonREMConsolidation:
    """
    Harmonic Consolidation: Master broadcasts stable Witness DOWN.
    
    - Identify dissonant satellites (low stability or contradictory witness)
    - Apply accelerated Grace (φ⁻⁴) to force alignment
    - Preserve high-stability satellites unchanged
    """
    
    def consolidate(self, master: np.ndarray, satellites: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Consolidate satellites toward master's witness.
        
        Returns: (consolidated_satellites, stats)
        """

class REMRecombination:
    """
    Stochastic Recombination: Find new stable "chords".
    
    - Apply φ-jitter rotations to satellite phases
    - Test resulting configurations for stability
    - Keep discoveries that improve overall Grace-stability
    """
    
    def recombine(self, satellites: np.ndarray, seed: int = None) -> Tuple[np.ndarray, dict]:
        """
        Attempt to discover new stable configurations.
        
        Returns: (new_configuration or None, stats)
        """

class ParadoxResolver:
    """
    Handle contradictions via Golden Ratio Phase Shift.
    
    When two satellites have contradictory witnesses (opposite scalars),
    apply phase shift of 2π·φ⁻¹ to one of them.
    """
    
    def scan_and_resolve(self, satellites: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Find and resolve all paradoxes.
        
        Returns: (resolved_satellites, stats)
        """

class EnhancedDreamingSystem:
    """Complete dreaming system integrating all components."""
    
    def sleep_cycle(self, master: np.ndarray, satellites: np.ndarray, 
                    max_iterations: int = 10) -> dict:
        """
        Run complete sleep cycle:
        1. Non-REM consolidation
        2. REM recombination
        3. Paradox resolution
        4. Repeat until stable or max iterations
        
        Returns: {
            'final_master': np.ndarray,
            'final_satellites': np.ndarray,
            'woke': bool,
            'stats': dict
        }
        """
```

---

### `fractal/nested_torus.py`

**Purpose:** Integrate all components into a single hierarchical memory system.

**Key Class:**

```python
class NestedFractalTorus:
    """
    Complete nested fractal torus memory system.
    
    Integrates:
    - Phase distribution for satellite management
    - Interaction tensor for level-to-level projection
    - Chirality for topological friction
    - Enhanced dreaming for consolidation
    - Downward projection for generation
    """
    
    def __init__(self, max_levels: int = 2, vocab_size: int = 1000):
        self.levels = [TorusLevel(l) for l in range(max_levels)]
        self.embeddings = initialize_embeddings(vocab_size)
    
    def embed_token(self, token_id: int) -> np.ndarray:
        """Get multivector embedding for token."""
        
    def embed_sequence(self, token_ids: List[int]) -> np.ndarray:
        """Compose sequence into context multivector."""
        
    def learn(self, context: List[int], target: int, level: int = 0):
        """
        Learn association at specified level.
        
        1. Embed context and target
        2. Bind via geometric product
        3. Store in level's memory
        4. Propagate to satellites
        """
        
    def retrieve(self, context: List[int]) -> Tuple[int, float, dict]:
        """
        Retrieve target for context.
        
        1. Try exact hash lookup
        2. Try holographic unbinding
        3. Try compositional retrieval from satellites
        
        Returns: (target_id, confidence, stats)
        """
        
    def dream(self) -> dict:
        """Run dreaming cycle on all levels."""
        
    def get_statistics(self) -> dict:
        """Return system statistics."""
```

---

### `fractal/grand_equilibrium.py`

**Purpose:** Compute global witness and verify energy conservation.

**Key Class:**

```python
class GrandEquilibrium:
    """
    Computes and monitors the Grand Equilibrium state.
    
    The Grand Equilibrium Equation:
        W_global = (1/φ) ∫ [W_local × φ⁻² · V_local] dψ
        
    Energy Conservation:
        E_Global ≈ φ × Σ E_Local
    """
    
    def update(self, level_mvs: List[np.ndarray]):
        """Update with current state of all levels."""
        
    def get_global_witness(self) -> np.ndarray:
        """Compute aggregated global witness."""
        
    def verify_energy_conservation(self) -> Tuple[bool, float]:
        """
        Check if energy is conserved across levels.
        
        Returns: (is_conserved, relative_error)
        """
        
    def get_stability(self) -> float:
        """Overall system stability (0-1)."""
        
    def is_at_equilibrium(self) -> bool:
        """Check if system has reached stable Grand Equilibrium."""
```

---

### `fractal/downward_projection.py`

**Purpose:** Handle generation flow from Grand Master down to token emission.

**Key Class:**

```python
class DownwardProjection:
    """
    Generation pipeline: Grand Master → Token.
    
    Flow:
    1. Start with Grand Master witness (what to express)
    2. Apply GraceInverse to inflate structure
    3. Unbind through each level to find matching patterns
    4. Emit token at phase-locked intervals
    """
    
    def __init__(self, vocab_size: int):
        self.grace_inverse = GraceInverse()
        self.token_embeddings = initialize_embeddings(vocab_size)
        
        # Phase-locked emission window
        self.emission_window_start = PI * PHI_INV
        self.emission_window_end = PI * PHI_INV + PHI_INV_SQ
    
    def project_level_down(self, higher_mv: np.ndarray, 
                           lower_memory: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Project from higher level to lower.
        
        1. Apply GraceInverse to inflate
        2. Unbind from lower level's memory
        3. Apply Grace to find stable attractor
        
        Returns: (projected_mv, confidence)
        """
    
    def unbind_to_token(self, projected_mv: np.ndarray) -> Tuple[int, float]:
        """
        Find closest token to projected multivector.
        
        Returns: (token_id, similarity)
        """
    
    def phase_locked_emission(self, master_phase: float, 
                               projected_mv: np.ndarray) -> Tuple[int, float]:
        """
        Emit token only within phase window.
        
        Emission window: [π·φ⁻¹, π·φ⁻¹ + φ⁻²]
        
        Returns: (token_id or None, confidence)
        """
    
    def generate_sequence(self, grand_master_mv: np.ndarray,
                          lower_memory: np.ndarray,
                          max_tokens: int = 20) -> Tuple[List[int], dict]:
        """
        Generate sequence from grand master state.
        
        Returns: (token_ids, stats)
        """
```

---

## Test Suite

### Unit Tests (`holographic_v4/tests/`)

| Test File | Coverage |
|-----------|----------|
| `test_phase_distribution.py` | Golden spiral, frequency stagger, no resonance |
| `test_interaction_tensor.py` | Orthogonal rotors, bidirectional projection |
| `test_chirality.py` | Even/odd assignment, self-inverse flip |
| `test_grace_inverse.py` | Reverses Grace, correct inflation scales |
| `test_dreaming_enhanced.py` | Non-REM, REM, paradox resolution |
| `test_nested_torus.py` | Full integration, learn/retrieve |

### Integration Test (`test_nested_fractal_torus.py`)

11 comprehensive tests covering:
1. Phase distribution prevents resonance
2. Toroidal round-trip preserves witness
3. Interaction tensor bidirectional projection
4. Chirality alternates even/odd
5. GraceInverse reverses Grace
6. Dreaming consolidates satellites
7. Paradox resolution separates contradictions
8. Nested torus learn/retrieve
9. Grand equilibrium computation
10. Downward projection generates tokens
11. Complete pipeline integration

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Learn (single association) | O(1) | Holographic binding |
| Retrieve (exact) | O(1) | Hash lookup |
| Retrieve (compositional) | O(N) | Aggregate from satellites |
| Dreaming (per level) | O(16) | Fixed satellite count |
| Generation (per token) | O(L) | L = number of levels |

---

## Future Work

1. **Modal-scale testing**: WikiText-2 with 16³ system
2. **GPU optimization**: Batch operations for `torus/` modules
3. **Dynamic level allocation**: Add levels as capacity fills
4. **Cross-level attention**: Direct satellite-to-grand-master paths
