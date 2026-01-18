"""
REM Recombination — Creative Discovery of Schemas

Implements brain-inspired REM sleep recombination:
- Recombine prototypes using Clifford operations
- Survival test under strong Grace contraction
- Only quotient-stable structures survive as schemas

THEORY:
    "Dreams may generate anything. Only structure earns the right to become belief."

φ-ROTATIONS:
    REM introduces small φ-rotations into satellite phases for exploration.
    This explores nearby configurations, looking for more stable "chords".
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import replace

from holographic_prod.core.constants import (
    PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR, DTYPE
)
from holographic_prod.core.algebra import (
    geometric_product, grace_operator_batch, frobenius_cosine,
    decompose_to_coefficients, reconstruct_from_coefficients
)
from .structures import SemanticPrototype, Schema
from .consolidation import dream_grace_operator


# =============================================================================
# REM RECOMBINATOR
# =============================================================================

class REMRecombinator:
    """
    REM sleep phase: recombination and survival testing.
    
    This addresses:
        - Abstraction: discover invariants not present in data
        - Robustness: only quotient-stable structures survive
    
    Theory:
        "Dreams may generate anything. Only structure earns the right to become belief."
    """
    
    def __init__(
        self,
        basis: np.ndarray,
        xp = np,
        survival_threshold: float = PHI_INV,  # φ-derived
        recurrence_threshold: int = 3,
        grace_steps: int = 10,
        grace_rate: float = PHI_INV_SQ,
    ):
        """
        Args:
            basis: Clifford algebra basis matrices
            xp: numpy or cupy
            survival_threshold: Quotient stability required to survive
            recurrence_threshold: How many times a schema must appear to be promoted
            grace_steps: Number of Grace iterations for survival test
            grace_rate: Contraction rate (stronger for REM)
        """
        self.basis = basis
        self.xp = xp
        self.survival_threshold = survival_threshold
        self.recurrence_threshold = recurrence_threshold
        self.grace_steps = grace_steps
        self.grace_rate = grace_rate
        
        # Candidate pool for schemas
        self.schema_candidates: Dict[str, Schema] = {}
    
    def _hash_canonical(self, M: np.ndarray, precision: int = 2) -> str:
        """Create a hash from the canonical form for deduplication."""
        rounded = np.round(M, precision)
        return str(rounded.tobytes())
    
    def recombine_compose(
        self, 
        A: np.ndarray, 
        B: np.ndarray
    ) -> np.ndarray:
        """
        Recombination operator: composition (A × B).
        Creates new context by sequential application.
        """
        return geometric_product(A, B)
    
    def recombine_unbind(
        self, 
        A: np.ndarray, 
        B: np.ndarray
    ) -> np.ndarray:
        """
        Recombination operator: unbinding (A × B⁻¹).
        Extracts the "difference" between A and B.
        This can reveal abstract relations.
        """
        try:
            B_inv = self.xp.linalg.pinv(B)
        except (self.xp.linalg.LinAlgError, ValueError, RuntimeError) as e:
            raise RuntimeError(
                f"Pseudoinverse failed in recombine_unbind: {e}. "
                "Matrix B may be degenerate or numerically unstable."
            ) from e
    
    def recombine_perturb(
        self, 
        A: np.ndarray,
        noise_scale: float = PHI_INV_CUBE,  # φ-derived
    ) -> np.ndarray:
        """
        Recombination operator: perturbation.
        Add small noise to explore nearby structures.
        """
        xp = self.xp
        if hasattr(xp, 'random'):
            noise = noise_scale * xp.random.randn(*A.shape)
        else:
            noise = noise_scale * np.random.randn(*A.shape)
            noise = xp.asarray(noise)
        return A + noise
    
    def apply_phi_rotation(
        self,
        satellite_matrix: np.ndarray,
        jitter: float = None,
    ) -> np.ndarray:
        """
        Apply φ-rotation (phase jitter) to satellite bivectors.
        
        Theory (Chapter 11):
            REM (Exploration): The master introduces small φ-rotations into satellite phases.
            jitter = random() × 2π × φ⁻¹  # Golden-angle scale
            satellite.bivectors *= cos(jitter)  # Phase rotation
        """
        if jitter is None:
            if hasattr(self.xp, 'random'):
                jitter = float(self.xp.random.random() * 2 * np.pi * PHI_INV)
            else:
                jitter = float(np.random.random() * 2 * np.pi * PHI_INV)
        
        # Decompose to get bivector coefficients
        coeffs = decompose_to_coefficients(satellite_matrix, self.basis, self.xp)
        
        # Grade 2 (bivectors) are indices 5-10
        bivector_indices = list(range(5, 11))
        
        # Apply rotation: bivectors *= cos(jitter)
        rotation_factor = float(np.cos(jitter))
        for idx in bivector_indices:
            coeffs[idx] *= rotation_factor
        
        # Reconstruct
        return reconstruct_from_coefficients(coeffs, self.basis, self.xp)
    
    def apply_phi_rotation_to_prototypes(
        self,
        prototypes: List[SemanticPrototype],
    ) -> List[SemanticPrototype]:
        """
        Apply φ-rotations to prototype bivectors during REM.
        
        Theory (Chapter 11):
            REM introduces small φ-rotations into satellite phases for exploration.
            This explores nearby configurations, looking for more stable "chords".
        """
        rotated_prototypes = []
        
        for proto in prototypes:
            rotated_matrix = self.apply_phi_rotation(proto.prototype_matrix)
            rotated_proto = replace(proto, prototype_matrix=rotated_matrix)
            rotated_prototypes.append(rotated_proto)
        
        return rotated_prototypes
    
    def survival_test(self, M: np.ndarray) -> Tuple[bool, np.ndarray, float]:
        """
        Test if a candidate survives strong Grace contraction.
        
        Returns:
            survived: True if structure is quotient-stable
            final: The final canonical form
            stability: Measure of stability (0 to 1)
        """
        xp = self.xp
        current = M.copy()
        previous = None
        
        for step in range(self.grace_steps):
            previous = current.copy()
            current = dream_grace_operator(current, self.basis, xp, self.grace_rate)
        
        norm = float(xp.linalg.norm(current, 'fro'))
        if norm < 1e-8:
            return False, current, 0.0
        
        current = current / norm
        
        if previous is not None:
            # Use cosine similarity for bounded [-1, 1] stability measure
            stability = frobenius_cosine(current, previous, xp)
        else:
            stability = 1.0
        
        survived = stability >= self.survival_threshold
        return survived, current, stability
    
    def survival_test_batch(
        self, candidates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        BATCHED survival test - GPU-accelerated.
        
        Args:
            candidates: [N, 4, 4] batch of candidate matrices
            
        Returns:
            survived: [N] bool array
            finals: [N, 4, 4] canonical forms
            stabilities: [N] stability scores
        """
        xp = self.xp
        current = candidates.copy()
        
        # Apply Grace steps in batch
        for step in range(self.grace_steps):
            previous = current.copy()
            current = grace_operator_batch(current, self.basis, xp)
            current = current * self.grace_rate + previous * (1 - self.grace_rate) * 0.1
        
        # Batch normalize
        norms = xp.linalg.norm(current.reshape(len(current), -1), axis=1, keepdims=True)
        norms = xp.maximum(norms, 1e-8)
        finals = current / norms.reshape(-1, 1, 1)
        
        # Batch stability check
        prev_norms = xp.linalg.norm(previous.reshape(len(previous), -1), axis=1, keepdims=True)
        prev_norms = xp.maximum(prev_norms, 1e-8)
        previous_normed = previous / prev_norms.reshape(-1, 1, 1)
        
        # Batch frobenius similarity
        stabilities = xp.sum(finals * previous_normed, axis=(1, 2)) / (
            xp.linalg.norm(finals.reshape(len(finals), -1), axis=1) *
            xp.linalg.norm(previous_normed.reshape(len(previous_normed), -1), axis=1) + 1e-10
        )
        
        survived = stabilities >= self.survival_threshold
        return survived, finals, stabilities
    
    def dream_cycle(
        self,
        prototypes: List[SemanticPrototype],
        num_recombinations: int = 100,
        verbose: bool = False,
    ) -> List[Schema]:
        """
        Main REM routine: recombine and test for survival.
        
        FULLY BATCHED for GPU acceleration.
        
        This is the REM sleep algorithm:
            1. Sample pairs of prototypes
            2. Recombine using various operators (batched)
            3. Test survival under strong Grace (batched)
            4. Track recurring survivors as candidate schemas
            5. Promote schemas that pass recurrence threshold
        """
        if len(prototypes) < 2:
            if verbose:
                print("  REM: Not enough prototypes for recombination")
            return []
        
        if verbose:
            print(f"  REM: Running {num_recombinations} recombinations...")
        
        xp = self.xp
        
        # OPTIMIZED: Batch extract matrices (single CPU→GPU transfer)
        n_proto = len(prototypes)
        matrices_np = np.empty((n_proto, 4, 4), dtype=np.float32)
        for i, p in enumerate(prototypes):
            mat = p.prototype_matrix
            matrices_np[i] = mat.get() if hasattr(mat, 'get') else mat
        matrices = xp.asarray(matrices_np)  # Single GPU transfer
        
        # Pre-generate all random choices with NumPy
        rng = np.random.default_rng()
        idx_pairs_np = rng.choice(n_proto, size=(num_recombinations, 2), replace=True)
        op_choices_np = rng.integers(0, 3, size=num_recombinations)
        op_names = ["compose", "unbind", "perturb"]
        
        # Fix same-index pairs
        same_mask = idx_pairs_np[:, 0] == idx_pairs_np[:, 1]
        idx_pairs_np[same_mask, 1] = (idx_pairs_np[same_mask, 0] + 1) % n_proto
        
        # Convert to xp arrays for GPU compatibility
        idx_pairs = xp.asarray(idx_pairs_np)
        op_choices = xp.asarray(op_choices_np)
        
        # BATCH: Generate all candidates at once
        A_batch = matrices[idx_pairs[:, 0]]
        B_batch = matrices[idx_pairs[:, 1]]
        
        candidates = xp.zeros((num_recombinations, 4, 4), dtype=DTYPE)
        
        # Compose: A @ B
        compose_mask = op_choices == 0
        if xp.any(compose_mask):
            candidates[compose_mask] = xp.matmul(A_batch[compose_mask], B_batch[compose_mask])
        
        # Unbind: A @ B^{-1} (using adjoint as proxy)
        unbind_mask = op_choices == 1
        if xp.any(unbind_mask):
            B_adj = xp.transpose(B_batch[unbind_mask], axes=(0, 2, 1))
            candidates[unbind_mask] = xp.matmul(A_batch[unbind_mask], B_adj)
        
        # Perturb: A + φ⁻⁴ noise
        perturb_mask = op_choices == 2
        if xp.any(perturb_mask):
            noise = xp.array(rng.standard_normal((int(xp.sum(perturb_mask)), 4, 4)) * PHI_INV_FOUR)
            candidates[perturb_mask] = A_batch[perturb_mask] + noise
        
        # BATCH survival test
        survived_mask, canonicals, stabilities = self.survival_test_batch(candidates)
        
        survivors = int(xp.sum(survived_mask))
        
        # OPTIMIZED: Transfer all results to CPU once (single GPU→CPU transfer)
        survived_mask_cpu = survived_mask.get() if hasattr(survived_mask, 'get') else survived_mask
        canonicals_cpu = canonicals.get() if hasattr(canonicals, 'get') else canonicals
        
        # Process survivors (pure Python, no GPU syncs)
        for i in range(num_recombinations):
            if survived_mask_cpu[i]:
                canonical = canonicals_cpu[i]
                op_name = op_names[int(op_choices_np[i])]
                
                key = self._hash_canonical(canonical)
                src_ids = [int(idx_pairs_np[i, 0]), int(idx_pairs_np[i, 1])]
                
                if key not in self.schema_candidates:
                    self.schema_candidates[key] = Schema(
                        canonical_form=canonical,
                        recurrence_count=1,
                        source_operations=[op_name],
                        source_prototype_ids=src_ids,
                    )
                else:
                    self.schema_candidates[key].recurrence_count += 1
                    self.schema_candidates[key].source_operations.append(op_name)
                    for sid in src_ids:
                        if sid not in self.schema_candidates[key].source_prototype_ids:
                            self.schema_candidates[key].source_prototype_ids.append(sid)
        
        if verbose:
            print(f"    Survival rate: {survivors}/{num_recombinations} ({100*survivors/num_recombinations:.1f}%)")
        
        # Promote schemas that pass recurrence threshold
        promoted = [
            schema for schema in self.schema_candidates.values()
            if schema.recurrence_count >= self.recurrence_threshold
        ]
        
        if verbose:
            print(f"    Promoted schemas: {len(promoted)}")
            for schema in promoted[:5]:
                print(f"      recurrence={schema.recurrence_count}, ops={schema.source_operations[:3]}")
        
        return promoted
