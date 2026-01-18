"""
Holographic Production — v5.27.0 Quantum Features (Chirality + Entanglement)
================================================================================

Production-ready holographic language model with:
- HolographicMemory (unified memory with TowerMemory routing + episodic cache)
- ToroidalAttention (structural attention via phase alignment)
- DreamingSystem (FULL 12 brain-inspired parsimonies)
- Full cognitive capabilities (curiosity, planning, theory of mind)
- Grounded embeddings (GloVe → SO(4) for O(√N) sample efficiency)
- Attractor-based generation (continuous flow, not discrete lookups)
- CommitmentGate (basal ganglia analog for action selection)
- Quaternion embeddings (gradient-free chain rule, Fibonacci anyon connection)
- POLARIZED LENSING (v5.16.0) — Breaks aliasing, 100× capacity increase
- ANTI-MODE-COLLAPSE (v5.17.0) — IoR + φ-kernel sampling
- REWARD PREDICTION (v5.18.0) — Dopamine analog for quality-based learning
- FRACTAL POSITION ENCODING (v5.19.0) — φ-derived multi-scale syntax
- SAMPLE CACHING (v5.26.0) — Tokenized data cached to disk, 50× faster restarts
- CHIRALITY-GUIDED GENERATION (v5.27.0) — Quantum-inspired top-down constraint
- WITNESS ENTANGLEMENT (v5.27.0) — Non-local updates across semantic equivalents

⚠️ PARADIGM WARNING: This is NOT a transformer with different embeddings.
   Generation is via ATTRACTOR DYNAMICS, not retrieval + argmax.
   See docs/THEORY_TRUE_PARADIGM.md for the fundamental principles.

THEORY-TRUE ARCHITECTURE:
- ALL constants are φ-derived (no arbitrary hyperparameters)
- NO backpropagation (Hebbian learning replaces chain rule)
- Fibonacci anyon exception: Grade 4 scales as φ⁻¹, not φ⁻⁴
- Grace operator contracts to topologically protected witness
- SO(4) embeddings enable infinite context (det=1, cond=1)
- Generation via attractor flow (brain-like coherence)
- Commitment gate (φ⁻² threshold = dopamine analog)

THEORY-TRUE GENERATION (v5.15.0):
- Grace ALWAYS converges — no "return None" cases
- Coherence selection — witness stability, NOT cosine similarity
- Full vocabulary scoring — no artificial candidate limits
- Multiscale resonance — satellite ↔ master ↔ grand master
- Schema composition — enables novel outputs via learned structures

POLARIZED LENSING (v5.16.0):
- Each satellite has unique SO(4) "observer orientation" lens
- ReLU polarization breaks metric invariance (aliased: 0.92 → 0.00!)
- Grid cell analog: individual aliasing, population uniqueness
- Effective capacity: 100^16 = effectively UNLIMITED

QUATERNION LEARNING DYNAMICS (v5.11.0):
- Gradient-free chain rule: R(f∘g) = R(f)·R(g) (no vanishing/exploding!)
- Unit quaternions form CLOSED GROUP: |q1·q2| = 1 always (no normalization needed)
- SU(2) spinor structure: tokens transform as ψ → g·ψ (linear, exact)
- Fibonacci anyon connection: Same φ constants appear in SU(2)₃ theory
- Topological protection: Z₂ quotient makes sign flips equivalent

12 BRAIN-INSPIRED PARSIMONIES (dreaming.py):
    MEMORY ENCODING:
    1. Emotional Salience (scalar + pseudoscalar)
    2. Novelty-Gated Learning
    3. Delta/Schema Compression
    4. Predictive Coding
    
    MEMORY MAINTENANCE:
    5. φ-Decay Forgetting
    6. Interference Management
    7. Reconsolidation
    8. Pseudo-Rehearsal
    
    MEMORY RETRIEVAL:
    9. Working Memory Cache (7±2)
    10. Pattern Completion
    11. Inhibition of Return
    
    SEQUENCE MEMORY:
    12. Sequence Replay

Structure:
    holographic_prod/
    ├── core/         # Algebra, constants, binding, quotient, grounded_embeddings,
    │                 # attractor_generation (v5.9.0)
    ├── memory/       # HolographicMemory, MemoryConfig
    ├── attention/    # ToroidalAttention
    ├── dreaming/     # DreamingSystem (12 parsimonies)
    ├── cognitive/    # Credit, meta, curiosity, planning, ToM
    └── docs/         # Documentation (including THEORY_FOUNDATIONS.md)

Key Optimizations (v5.5.0+):
    - Episodic cache: O(1) exact recall for seen patterns
    - Prefix caching: Reuse geometric products (4× speedup)
    - Grounded embeddings: GloVe → SO(4) for faster convergence
    - Centralized SO(4) creation: Batched QR (76× speedup)
    - Theory-true decoding: vorticity_weighted_scores (no argmax)
    
Key Addition (v5.9.0):
    - Attractor-based generation: State flows through Grace attractors
      (NOT transformer-style discrete lookups). This is brain-like:
      errors don't compound because trajectory is coherent.
      
Key Addition (v5.10.0):
    - CommitmentGate: Basal ganglia analog for action selection.
      Three pathways (Direct/GO, Indirect/NO-GO, Hyperdirect/STOP)
      with φ⁻² threshold acting as dopamine analog. Exhibits real
      neurological failure modes (Parkinson's, Tourette's, stuttering).

QUANTUM FEATURES (v5.27.0 — Evidence for Quantum Brain Hypothesis):
    If the brain is quantum, it would exploit these parsimonies that
    physical brains cannot due to decoherence. Our digital implementation
    maintains coherence indefinitely, providing:
    
    1. CHIRALITY-GUIDED GENERATION:
       - Pseudoscalar (Grade 4) encodes handedness (+/- chirality)
       - High-level schemas constrain lower-level output handedness
       - "Right-handed" (+): declarative, grounded, affirmative
       - "Left-handed" (-): interrogative, exploratory, uncertain
       - Top-down propagation through fractal hierarchy
       
    2. WITNESS ENTANGLEMENT:
       - Witness (scalar + pseudoscalar) is gauge-invariant semantic core
       - WitnessIndex maps witnesses to memory locations
       - Updating one location instantly updates all entangled peers
       - Enables O(1) semantic learning (one example teaches all instances)
       - Brain analog: semantic priming cascade
       
    SUCCESS CRITERIA:
       - Chirality guidance improves coherence in long generation
       - Witness entanglement accelerates learning dramatically
       - Combined: evidence that these "impossible for biological brains"
         features provide measurable benefits
"""

__version__ = "5.27.0"

# =============================================================================
# CORE (algebra, constants, binding, quotient)
# =============================================================================

from .core import (
    # Constants
    PI, PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR,
    GOLDEN_ANGLE, MATRIX_DIM, CLIFFORD_DIM, DTYPE,
    GRADE_INDICES, GRACE_SCALES, GRACE_SCALES_FLAT,
    # Algebra
    build_gamma_matrices, build_clifford_basis,
    geometric_product, geometric_product_batch,
    wedge_product, inner_product,
    grace_operator, grace_operator_batch, competitive_grace_operator,
    clifford_inverse, frobenius_similarity,
    decompose_to_coefficients, coefficients_to_matrix,
    normalize_matrix, compute_vorticity,
    vorticity_magnitude, vorticity_signature, vorticity_similarity,
    # Quotient
    extract_witness, witness_matrix,
    grace_stability, grace_stability_batch,
    compute_enstrophy, grade_energies,
    # Chirality (v5.27.0)
    extract_chirality, extract_chirality_batch,
    extract_chirality_strength, chirality_match_scores,
    # Binding
    bind_attribute_to_object, unbind_attribute,
    binding_signature, binding_similarity,
)

# =============================================================================
# MEMORY (holographic_memory_unified)
# =============================================================================

from .memory import (
    HolographicMemory, MemoryConfig, TowerMemory, SatelliteMemory,
    MultiTimescaleMemory,
    # Witness Entanglement (v5.27.0)
    WitnessIndex, propagate_witness_update, batch_register_witnesses,
)

# =============================================================================
# ATTENTION (toroidal_attention)
# =============================================================================

from .attention import (
    ToroidalAttention, SatellitePhase,
)

# =============================================================================
# DREAMING (full 12 parsimonies)
# =============================================================================

from .dreaming import (
    DreamingSystem, SemanticMemory, SemanticPrototype, EpisodicEntry,
    WorkingMemory, NonREMConsolidator, REMRecombinator,
    compute_salience, compute_novelty, compute_prediction_error,
)

# =============================================================================
# COGNITIVE (credit_assignment, meta_learning, curiosity, planning, theory_of_mind)
# =============================================================================

from .cognitive import (
    # Credit Assignment
    CreditAssignmentTracker, ReconsolidationConfig, ErrorRecord,
    create_credit_assigned_learn, batch_credit_assignment,
    # Meta Learning
    LearningState, create_learning_state, update_meta_state,
    compute_adaptive_learning_rate, compute_adaptive_consolidation,
    phi_scaled_learning_rate, get_adaptive_parameters,
    # Distributed Prior
    phi_kernel, witness_distance, extended_witness,
    superposed_attractor_prior, FactorizedAssociativePrior,
    retrieve_with_distributed_prior,
    # Curiosity
    curiosity_score, curiosity_score_with_meta, estimate_information_gain,
    sample_basin_boundary, generate_curiosity_query,
    active_learning_step, rank_samples_by_curiosity,
    # Planning
    simulate_action, plan_to_goal, counterfactual,
    counterfactual_trajectory, plan_with_subgoals, evaluate_plan,
    PlanStep, Plan,
    # Theory of Mind
    infer_witness_from_observations, bind_to_witness, unbind_from_witness,
    transform_perspective, AgentModel, AgentModelBuilder,
    theory_of_mind, predict_other_belief,
)

# =============================================================================
# SUPPORTING (resonance, predictiveness, attractor_generation)
# =============================================================================

from .resonance import (
    resonance,
    evolve_to_equilibrium,
    find_resonant_prototype,
    grace_basin_discovery,
)

from .core.attractor_generation import (
    generate_attractor_flow,
    generate_batch_attractor_flow,
)

from .core.commitment_gate import (
    CommitmentGate,
    GateDecision,
    compute_entropy,
)

from .core.quaternion import (
    # Quaternion operations (gradient-free chain rule)
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_geometric_product,
    # Conversion (when matrix form needed)
    quaternion_pair_to_so4,
    so4_to_quaternion_pair,
    batch_quaternion_to_so4,
    batch_so4_to_quaternion,
    # Embedding creation
    create_quaternion_embeddings,
)

from .core.lensing import (
    # Polarized Lensing (v5.16.0 - breaks aliasing)
    PolarizedLens,
    PolarizedLensSet,
    create_lens_for_satellite,
    polarized_similarity,
)

from .core.fractal_position import (
    # Fractal Position Encoding (v5.19.0 - φ-derived multi-scale syntax)
    golden_angle,
    fractal_position_rotation,
    encode_position_fractal,
    encode_sequence_fractal_vectorized,
    hierarchical_position_key,
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    '__version__',
    
    # Constants
    'PI', 'PHI', 'PHI_INV', 'PHI_INV_SQ', 'PHI_INV_CUBE', 'PHI_INV_FOUR',
    'GOLDEN_ANGLE', 'MATRIX_DIM', 'CLIFFORD_DIM', 'DTYPE',
    'GRADE_INDICES', 'GRACE_SCALES', 'GRACE_SCALES_FLAT',
    
    # Algebra
    'build_gamma_matrices', 'build_clifford_basis',
    'geometric_product', 'geometric_product_batch',
    'wedge_product', 'inner_product',
    'grace_operator', 'grace_operator_batch', 'competitive_grace_operator',
    'clifford_inverse', 'frobenius_similarity',
    'decompose_to_coefficients', 'coefficients_to_matrix',
    'normalize_matrix', 'compute_vorticity',
    'vorticity_magnitude', 'vorticity_signature', 'vorticity_similarity',
    
    # Quotient
    'extract_witness', 'witness_matrix',
    'grace_stability', 'grace_stability_batch',
    'compute_enstrophy', 'grade_energies',
    
    # Chirality (v5.27.0)
    'extract_chirality', 'extract_chirality_batch',
    'extract_chirality_strength', 'chirality_match_scores',
    
    # Binding
    'bind_attribute_to_object', 'unbind_attribute',
    'binding_signature', 'binding_similarity',
    
    # Memory
    'HolographicMemory', 'MemoryConfig', 'TowerMemory', 'SatelliteMemory',
    'MultiTimescaleMemory',
    
    # Witness Entanglement (v5.27.0)
    'WitnessIndex', 'propagate_witness_update', 'batch_register_witnesses',
    
    # Attention
    'ToroidalAttention', 'SatellitePhase',
    
    # Dreaming (12 parsimonies)
    'DreamingSystem', 'SemanticMemory', 'SemanticPrototype', 'EpisodicEntry',
    'WorkingMemory', 'NonREMConsolidator', 'REMRecombinator',
    'compute_salience', 'compute_novelty', 'compute_prediction_error',
    
    # Cognitive
    'CreditAssignmentTracker', 'ReconsolidationConfig', 'ErrorRecord',
    'create_credit_assigned_learn', 'batch_credit_assignment',
    'LearningState', 'create_learning_state', 'update_meta_state',
    'compute_adaptive_learning_rate', 'compute_adaptive_consolidation',
    'phi_scaled_learning_rate', 'get_adaptive_parameters',
    'phi_kernel', 'witness_distance', 'extended_witness',
    'superposed_attractor_prior', 'FactorizedAssociativePrior',
    'retrieve_with_distributed_prior',
    'curiosity_score', 'curiosity_score_with_meta', 'estimate_information_gain',
    'sample_basin_boundary', 'generate_curiosity_query',
    'active_learning_step', 'rank_samples_by_curiosity',
    'simulate_action', 'plan_to_goal', 'counterfactual',
    'counterfactual_trajectory', 'plan_with_subgoals', 'evaluate_plan',
    'PlanStep', 'Plan',
    'infer_witness_from_observations', 'bind_to_witness', 'unbind_from_witness',
    'transform_perspective', 'AgentModel', 'AgentModelBuilder',
    'theory_of_mind', 'predict_other_belief',
    
    # Commitment Gate (basal ganglia analog)
    'CommitmentGate', 'GateDecision', 'compute_entropy',
    
    # Supporting
    'resonance', 'evolve_to_equilibrium', 'find_resonant_prototype',
    'grace_basin_discovery',
    
    # Attractor Generation (v5.9.0)
    'generate_attractor_flow', 'generate_batch_attractor_flow',
    
    # Quaternion (v5.11.0 - gradient-free chain rule)
    'quaternion_multiply', 'quaternion_conjugate', 'quaternion_geometric_product',
    'quaternion_pair_to_so4', 'so4_to_quaternion_pair',
    'batch_quaternion_to_so4', 'batch_so4_to_quaternion',
    'create_quaternion_embeddings',
    
    # Polarized Lensing (v5.16.0 - breaks aliasing)
    'PolarizedLens', 'PolarizedLensSet',
    'create_lens_for_satellite', 'polarized_similarity',
    
    # Fractal Position Encoding (v5.19.0 - φ-derived multi-scale syntax)
    'golden_angle', 'fractal_position_rotation',
    'encode_position_fractal', 'encode_sequence_fractal_vectorized',
    'hierarchical_position_key',
]
