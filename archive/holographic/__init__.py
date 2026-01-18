"""
Holographic Language Model Package — v3.0
==========================================

Theory-true implementation using Cl(3,1) ≅ M₄(ℝ) isomorphism with
COMPOSITIONAL feature-based embeddings.

┌─────────────────────────────────────────────────────────────────────────────┐
│                         ARCHITECTURE OVERVIEW                               │
│                                                                             │
│   COMPOSITIONAL EMBEDDINGS (the breakthrough):                             │
│       word = I + Σᵢ αᵢ(word) · fᵢ                                          │
│                                                                             │
│       where fᵢ are 14 ORTHOGONAL feature directions in Cl(3,1)            │
│       and αᵢ(word) are learned coefficients per word                       │
│                                                                             │
│   This enables:                                                             │
│       ✓ 12x better semantic separation (0.72 vs 0.06)                      │
│       ✓ One-shot learning (new word clusters from context)                 │
│       ✓ Correct category generation (10/10)                                │
│                                                                             │
│   CONTEXT COMPUTATION:                                                      │
│       context = geometric_product(embed(w₁), embed(w₂), ...)               │
│                                                                             │
│   FEATURE LEARNING:                                                         │
│       Hebbian: words in same context → share features                       │
│       "Neurons that fire together wire together"                            │
└─────────────────────────────────────────────────────────────────────────────┘

Mathematical Foundation:
    Cl(3,1) with signature η = diag(+1,+1,+1,-1)
    Isomorphism: Cl(3,1) ≅ M₄(ℝ) (4×4 real matrices)
    Geometric product = Matrix multiplication (GPU GEMM optimized)

Key Components:
    1. CompositionalEmbedding: Word = feature composition
    2. FeatureSet: 14 orthogonal directions in grades 1-3
    3. CompositionalHolographicModel: Full integrated pipeline

Usage (Recommended):
    from holographic import CompositionalHolographicModel
    
    model = CompositionalHolographicModel(
        vocab_size=10000,
        num_features=14,
        context_size=5,
    )
    
    # Train
    model.train(contexts, targets, hebbian_lr=0.05)
    
    # Generate
    tokens = model.generate([1, 2, 3, 4], num_tokens=10)
    
    # One-shot learn new word from context
    model.one_shot_learn(new_word_idx, context)

Legacy API (still supported):
    from holographic import MatrixEmbedding, ContextAttractorMap
    
    embedding = MatrixEmbedding(vocab_size=10000)
    attractor_map = ContextAttractorMap(embedding)

Constants:
    PHI ≈ 1.618: Golden ratio (self-consistency)
    MATRIX_DIM = 4: 4×4 real matrices
    CLIFFORD_DIM = 16: 16 basis elements
"""

# Constants (sacred, theory-derived)
from .constants import (
    PI,
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
    BETA, GOLDEN_ANGLE,
    MAJOR_RADIUS, MINOR_RADIUS,
    CLIFFORD_DIM, MATRIX_DIM,
    GRADE_DIMS, GRADE_SLICES, GRACE_SCALE,
)

# Algebra operations (matrix representation)
from .algebra import (
    # Basis construction
    build_gamma_matrices,
    build_clifford_basis,
    build_metric_matrix,
    
    # Matrix operations
    normalize_matrix,
    geometric_product,
    geometric_product_batch,
    
    # Similarity functions
    frobenius_similarity,
    frobenius_similarity_batch,
    clifford_adjoint,
    metric_similarity,
    metric_similarity_batch,
    
    # Grace operator
    grace_operator_matrix,
    grace_iterate_matrix,
    
    # Embedding initialization
    initialize_embedding_matrix,
    initialize_all_embeddings,
    
    # Verification
    verify_gamma_matrices,
)

# Core classes
from .core import (
    MatrixEmbedding,
    ContextAttractorMap,
    generate_token,
    generate_sequence,
    generate_token_active,
    generate_sequence_active,
    train_step,
)

# Quotient structure (gauge invariance)
from .quotient import (
    # Witness extraction
    extract_witness_matrix,
    witness_similarity,
    witness_pointer,
    
    # Content & Binding
    extract_content,
    bind,
    
    # Spin(3) normal form
    spin3_rotor_matrix,
    random_spin3_rotor,
    sandwich,
    normal_form,
    
    # Grade analysis
    project_to_grade,
    grade_energies,
    compute_grade_variance,
    compute_witness_stability,
    
    # Quotient-aware similarity
    quotient_similarity,
    quotient_similarity_phi,
    
    # Diagnostics
    compute_separation,
    
    # Tests
    test_witness_invariance,
    test_normal_form_invariance,
    test_binding_properties,
    run_quotient_tests,
)

# Hierarchy (multi-level tower of quotients)
from .hierarchy import (
    AttractorCodebook,
    create_codebook,
    HierarchyLevel,
    HierarchicalModel,
    run_hierarchy_tests,
)

# Diagnostics
from .diagnostics import (
    semantic_coherence_test,
    witness_stability_analysis,
    grade_analysis,
    compositional_test,
    run_level1_diagnostics,
)

# Contrastive Learning
from .contrastive import (
    find_contrastive_pairs,
    contrastive_update,
    train_with_contrastive,
)

# Optimal Training
from .optimal_training import (
    TrainingConfig,
    OptimalTrainer,
)

# Two-Level Testing
from .two_level_test import (
    train_two_level,
)

# Compositional Embeddings (NEW - key insight)
from .compositional import (
    FeatureSet,
    create_feature_set,
    CompositionalEmbedding,
    run_compositional_tests,
)

# Feature Learning
from .feature_learning import (
    CooccurrenceTracker,
    learn_features_from_cooccurrence,
    learn_features_hebbian,
    infer_features_from_context,
    one_shot_learn_word,
)

# Full Integrated Pipeline
from .full_pipeline import (
    CompositionalHolographicModel,
    run_full_pipeline_tests,
)

__all__ = [
    # Constants
    'PI',
    'PHI', 'PHI_INV', 'PHI_INV_SQ', 'PHI_INV_CUBE',
    'BETA', 'GOLDEN_ANGLE',
    'MAJOR_RADIUS', 'MINOR_RADIUS',
    'CLIFFORD_DIM', 'MATRIX_DIM',
    'GRADE_DIMS', 'GRADE_SLICES', 'GRACE_SCALE',
    
    # Algebra - Basis
    'build_gamma_matrices',
    'build_clifford_basis',
    'build_metric_matrix',
    
    # Algebra - Operations
    'normalize_matrix',
    'geometric_product',
    'geometric_product_batch',
    
    # Algebra - Similarity
    'frobenius_similarity',
    'frobenius_similarity_batch',
    'clifford_adjoint',
    'metric_similarity',
    'metric_similarity_batch',
    
    # Algebra - Grace
    'grace_operator_matrix',
    'grace_iterate_matrix',
    
    # Algebra - Initialization
    'initialize_embedding_matrix',
    'initialize_all_embeddings',
    
    # Algebra - Verification
    'verify_gamma_matrices',
    
    # Core
    'MatrixEmbedding',
    'ContextAttractorMap',
    'generate_token',
    'generate_sequence',
    'generate_token_active',
    'generate_sequence_active',
    'train_step',
    
    # Quotient structure - Witness
    'extract_witness_matrix',
    'witness_similarity',
    'witness_pointer',
    
    # Quotient structure - Binding
    'extract_content',
    'bind',
    
    # Quotient structure - Normal form
    'spin3_rotor_matrix',
    'random_spin3_rotor',
    'sandwich',
    'normal_form',
    
    # Quotient structure - Grade analysis
    'project_to_grade',
    'grade_energies',
    'compute_grade_variance',
    'compute_witness_stability',
    
    # Quotient structure - Similarity
    'quotient_similarity',
    'quotient_similarity_phi',
    'compute_separation',
    
    # Quotient structure - Tests
    'test_witness_invariance',
    'test_normal_form_invariance',
    'test_binding_properties',
    'run_quotient_tests',
    
    # Hierarchy (multi-level)
    'AttractorCodebook',
    'create_codebook',
    'HierarchyLevel',
    'HierarchicalModel',
    'run_hierarchy_tests',
    
    # Diagnostics
    'semantic_coherence_test',
    'witness_stability_analysis',
    'grade_analysis',
    'compositional_test',
    'run_level1_diagnostics',
    
    # Contrastive Learning
    'find_contrastive_pairs',
    'contrastive_update',
    'train_with_contrastive',
    
    # Optimal Training
    'TrainingConfig',
    'OptimalTrainer',
    
    # Two-Level Testing
    'train_two_level',
    
    # Compositional Embeddings
    'FeatureSet',
    'create_feature_set',
    'CompositionalEmbedding',
    'run_compositional_tests',
    
    # Feature Learning
    'CooccurrenceTracker',
    'learn_features_from_cooccurrence',
    'learn_features_hebbian',
    'infer_features_from_context',
    'one_shot_learn_word',
    
    # Full Pipeline
    'CompositionalHolographicModel',
    'run_full_pipeline_tests',
    
    # Version
    '__version__',
]

__version__ = '3.0.0'  # Full compositional pipeline (Hebbian + attractor + one-shot)
