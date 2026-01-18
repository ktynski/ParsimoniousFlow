"""
Holographic Language Model v4.23.0 — Theory-True Cognitive Architecture
=======================================================================

A theory-true cognitive architecture implementing all core learning capabilities
using Clifford algebra (Cl(3,1)), Grace flow dynamics, and φ-derived parameters.
NO arbitrary normalizations, clipping, or softmax — only theory-derived operations.

Last Updated: 2026-01-13

v4.23.0 — CONTRASTIVE EMBEDDING LEARNING
========================================

MAJOR ACHIEVEMENT: 100% paraphrase generalization through contrastive learning

NEW FEATURES:
    ✅ contrastive_embedding_update() - Pull co-predictive embeddings together
    ✅ auto_contrastive_update() - Automatically find pairs from predictiveness
    ✅ co_occurrence_count scaling - Brain-like LTP (more co-predictions = faster learning)
    ✅ φ-derived iteration theory - Learning rate φ⁻⁵, max_similarity 1-φ⁻⁴
    ✅ Validated on WikiText-2 via Modal

HOW ITERATIONS ARE DETERMINED (BRAIN-LIKE):
    The brain doesn't do fixed iteration counts. Instead:
    - STDP: Each co-firing strengthens by a small amount
    - LTP: ~100 Hz for ~1 second induces potentiation  
    - Sleep cycles: 4-6 consolidation passes per night
    - Reconsolidation: Each retrieval is another "iteration"
    
    Our system mirrors this:
    - effective_rate = φ⁻⁵ × log(1 + co_occurrence_count)  # Scales with co-predictions
    - max_similarity = 1 - φ⁻⁴ ≈ 0.854                     # Above identity-bias
    - Convergence stops when similarity > max_similarity    # Prevents collapse

v4.22.0 — DUAL MEMORY INDEXING
==============================

    ✅ Episodic Index: VorticityWitnessIndex (8D even-grade keys, exact matches)
    ✅ Semantic Index: CanonicalSemanticIndex (2D, abs(p), bireflection-aware)
    ✅ Triple cascade: holographic → episodic → semantic

v4.21.0 — CLEANUP: ALL φ-DERIVED, NO LEGACY, NO FALLBACKS
=========================================================

CLEANUP COMPLETED:
    ✅ 54+ arbitrary constants (0.3, 0.5, 0.7, etc.) replaced with φ-derived values
    ✅ WitnessIndex removed (only ~12 buckets for 1000 samples = useless)
    ✅ VorticityWitnessIndex is now the ONLY episodic index
    ✅ Removed use_gpu_witness, use_torus_symmetry, use_vorticity_index flags
    ✅ Removed fallback_to_full from compute_semantic_context
    ✅ HybridHolographicMemory.create simplified to always use VorticityWitnessIndex

PHILOSOPHY:
    - NO arbitrary constants — all values are φ-derived
    - NO fallbacks — if something doesn't work, it fails explicitly
    - NO legacy code — deprecated code is removed

v4.20.0 — GENERALIZATION FIX (Prototype Coverage)
=================================================

ROOT CAUSE IDENTIFIED: Generalization failed (1%) because interference management
was TOO AGGRESSIVE, merging 100+ prototypes down to just 16.

THEORY:
    From distributed_prior.py: "More prototypes → better generalization (coverage)"
    But sleep() used similarity_threshold=φ⁻¹ (0.618) for merging.
    This merged prototypes that were 62%+ similar — semantically DISTINCT patterns!
    
FIX:
    Changed similarity_threshold from φ⁻¹ (0.618) to 1-φ⁻³ (0.764)
    Now only merges true NEAR-DUPLICATES (76%+ similar)
    Expected result: More prototypes survive → better semantic coverage → better generalization

KEY DISTINCTION (theory-true):
    φ⁻¹ = RETRIEVAL confidence threshold (single basin vs distributed prior)
    1-φ⁻³ = MERGING threshold (only merge near-duplicates)
    
    These are DIFFERENT operations with DIFFERENT appropriate thresholds.

v4.19.0 — FULL 8D EVEN-GRADE INDEXING
=====================================

KEY INSIGHT: Witness is BLIND to word order. Vorticity captures order.

    Witness (σ, p):  Tr(AB) = Tr(BA) → SAME witness for permutations
    Vorticity (G2):  AB - BA ≠ 0    → DIFFERENT for permutations
    
    BOTH are needed together, not as alternatives or fallbacks.

FIBER BUNDLE STRUCTURE (from rhnsclifford.md):

    Cl(3,1) = BASE SPACE (2-torus from bivectors) × FIBER (witness)
    
    | Grade | Components | What It Encodes |
    |-------|------------|-----------------|
    | G0    | 1 scalar   | Semantic "gist" |
    | G2    | 6 bivectors| Syntactic structure (word ORDER) |
    | G4    | 1 pseudo   | Chirality/orientation |
    
    The 6 bivectors each encode a different rotation plane:
    - e₀₁, e₀₂, e₀₃: Time-space (temporal relationships)
    - e₁₂, e₁₃, e₂₃: Space-space (spatial/structural)

BUCKETS AND COLLISIONS:

    BUCKET = Region where contexts flow to same attractor under Grace
    COLLISION = Different contexts in same bucket (bad!)
    
    OLD (4D key): 6.5% permutation collisions
    NEW (8D key): 0% permutation collisions
    
    Resolution = φ⁻² (spectral gap = "indistinguishable under Grace")

COMBINED SIMILARITY (φ-weighted):

    similarity = (1-φ⁻¹)·witness + φ⁻¹·vorticity = 38% semantic + 62% syntactic

THEORY VALIDATION COMPLETE (v4.16.0):
    64 tests pass across 6 domains — see THEORY_VALIDATION_RESULTS.md
    - Foundations: Spectral gap γ=φ⁻², enstrophy decay φ⁻⁴
    - Dynamics: Lyapunov λ<0, Navier-Stokes analogy real
    - Information: Grace IS information bottleneck
    - Memory: Testing effect, encoding specificity emerge
    - Language: Semantic roles in bivectors, composition associative
    - Creativity: Bisociation, metaphor, poetic compression work

CORE ARCHITECTURE:
    1. GEOMETRIC PRODUCT for context composition (AB)
    2. WEDGE PRODUCT for sequential vorticity (A∧B = (AB-BA)/2)
    3. GRACE OPERATOR for grade-wise contraction (φ⁻ᵏ per grade)
       - THE normalizer: replaces arbitrary Frobenius/softmax
    4. WITNESS for gauge-invariant self-reference (scalar + pseudoscalar)
    5. BINDING for self-referential content
    6. QUOTIENT SIMILARITY for stable retrieval
    7. HOLOGRAPHIC MEMORY via Clifford superposition (theory-true O(1) storage)
    8. EQUILIBRIUM FORGETTING for sparse efficiency (φ-decay pruning)
    9. φ-CURRICULUM for theory-true context scaling
    10. PREDICTIVENESS — automatic semantic vs noise identification

v4.10.0 ADDITIONS — PARSIMONY OPTIMIZATIONS:
    Removed redundant/expensive operations that violated theory or wasted compute:
    
    1. NOVELTY CHECK REMOVAL (21× speedup on train_step):
       - Holographic superposition naturally handles duplicates via reinforcement
       - memory += bind(C,T) twice = 2 × bind(C,T) (pattern REINFORCED!)
       - No need to check "is this novel?" — just store
       - Flag: skip_novelty_check=True (default)
    
    2. PERIODIC SATURATION CHECK (31% speedup):
       - compute_witness_entropy was 20% of train_step time
       - Dreaming triggers don't need instant signals
       - Now checked every 89 steps (Fibonacci - theory-aligned)
       - Interval: _saturation_check_interval=89
    
    3. ARBITRARY CONSTANTS → φ-DERIVED:
       - 0.5 → PHI_INV_SQ (0.382) in vorticity_decode_scores
       - 0.5 → PHI_INV_SQ (0.382) in self_organizing_retrieve
    
    4. CODEBASE AUDIT — NO TRANSFORMER VESTIGES:
       ✓ No softmax (uses φ-kernel)
       ✓ No temperature parameters
       ✓ No learning rate schedules (fixed φ⁻¹)
       ✓ No dropout/batch norm
       ✓ No optimizer state (Adam, momentum)

v4.8.0 ADDITIONS — TRUE HOLOGRAPHIC MEMORY:
    The architecture now uses TRUE holographic superposition instead of hash lookup:
    
    STORAGE: memory += bind(context, target)    [superposition]
    RETRIEVAL: target ≈ unbind(context, memory) [O(1) constant time]
    
    NEW FEATURES:
    - HolographicMemory: True superposition-based storage
    - MultiTimescaleMemory: φ-parameterized decay (fast/medium/slow)
    - iterative_unbind(): Multi-item retrieval from superposition
    - compute_witness_entropy(): Capacity saturation signal
    - is_memory_saturated(): Consolidation trigger
    
    WHY THIS MATTERS:
    - Hash lookup was computationally convenient but OFF-THEORY
    - True holographic memory respects geometric structure
    - Grace naturally denoises interference (built-in cleanup)
    - Graceful degradation instead of brittle hash collisions

v4.7.0 ADDITIONS — META-COGNITIVE TRAINING LOOP:
    The brain doesn't re-learn what it already knows. We now do the same:
    
    For each (context, target):
        1. PREDICT: What do I expect? (retrieve(context))
        2. COMPARE: Was I surprised? (predicted ≠ actual)
        3. LEARN: Only if surprised! (skip redundant patterns)
    
    RESULT: 60-70% efficiency gain, same accuracy.
    
    ADAPTIVE SLEEP:
        Sleep triggers when φ-derived thresholds are exceeded:
        - Memory pressure > φ⁻¹ (≈0.618)
        - Novelty rate > φ⁻² (≈0.382)
        - Error rate > φ⁻² (≈0.382)

ADVANCED CAPABILITIES:
    11. THEORY OF MIND — perspective transformation via witness binding
    12. CREDIT ASSIGNMENT — provenance tracking and targeted reconsolidation
    13. REPRESENTATION LEARNING — embedding drift with identity-bias constraint
    14. RECURSIVE COMPUTATION — iterative retrieval and geometric search
    15. PLANNING — action simulation, goal-seeking, counterfactual reasoning
    16. ATTRIBUTE BINDING — object-attribute composition via Clifford grades
    17. GROUNDING — perception-to-Clifford mapping
    18. META-LEARNING — adaptive φ-derived parameters based on context
    19. CURIOSITY — active learning via stability gradient descent
    20. META-COGNITIVE LOOP — predictive coding + adaptive sleep (v4.7.0)

SELF-ORGANIZING PRINCIPLE (Theory-Derived):
    Grace-stability determines memory fate:
    
        σ(M) = (scalar² + pseudo²) / total_coefficient_energy
        
        if σ ≥ φ⁻² (0.382): Episode is STABLE → stays episodic
        if σ < φ⁻² (0.382): Episode is TRANSIENT → consolidates
    
    This is NOT a tuned parameter — it's the SPECTRAL GAP of Grace!

KEY THEORETICAL INSIGHTS:
    1. Grace is viscosity for bivectors
       - Vorticity (grade-2) = rotational energy
       - Grace damps at rate φ⁻² per step
       - Bounded equilibrium (Navier-Stokes analogy)
    
    2. Grace is the ONLY normalizer
       - Frobenius used only for numerical stability (uniform scaling)
       - Grace does theory-true grade-wise damping
       - No softmax (replaced with φ-kernel)
    
    3. Witness = Stable Core
       - Scalar + pseudoscalar survive Grace
       - All other grades decay
       - Witness similarity for retrieval

VORTICITY GRAMMAR GENERALIZATION (v4.7.0):
    The wedge product A∧B = (AB-BA)/2 captures word ORDER geometrically:
    
    VERIFIED BY TEST (6/6 pass):
    • ||A ∧ B + B ∧ A|| = 0.0 (perfect anti-symmetry)
    • "john loves mary" ↔ "mary loves john" = -1.0 (perfect anti-correlation!)
    • Same structure clusters: +0.22 similarity (vs -0.19 for different)
    • Novel words in familiar structures work via GEOMETRIC matching
    
    BRAIN-LIKE COHERENCE (replaces failed FFT approach):
    • FFT on magnitudes failed — coherence is in DIRECTION (phase), not amplitude
    • Brains use phase-based binding, not amplitude-based frequency
    • Predictability is strongest discriminator (14.7% difference)
    
    NEW FEATURES:
    • compute_vorticity_coherence() — brain-like metrics
    • compute_loop_circulation() — 35% lower for paraphrases
    • VorticityTracker — generation monitoring
    • check_semantic_invariance() — 10x higher for paraphrases

SCALABLE CONTEXT WINDOWS:
    Unlike Transformers (O(N²) attention), this architecture:
    • O(N) in context length for composition (chunked for large contexts)
    • O(1) storage for context matrix (always 4×4!)
    • Tested stable to context_size = 50,000+ with Frobenius stabilization
    
    Use long-sequence datasets (pg19, gutenberg) for full capability.
    TinyStories (~200 words) wastes the architecture's potential!

BRAIN-INSPIRED PARSIMONIES (12 total):
    Memory Encoding: Salience, Novelty, Delta Compression, Predictive Coding
    Memory Maintenance: Pruning, Interference Management, Reconsolidation, Pseudo-Rehearsal
    Memory Retrieval: Pattern Completion, Sequence Replay, Inhibition of Return, Grace Basin

EQUILIBRIUM FORGETTING (Theory-True Sparse Efficiency):
    The brain doesn't wait until "full" to forget — it continuously prunes:
    
        retention_strength = salience × φ^(access_count) × φ^(-Δt/τ)
        
        Forget if retention < φ⁻³ (≈ 0.236, Grace spectral structure)
    
    This creates SPARSE EFFICIENCY through natural selection:
    - Frequently accessed → survives (reconsolidation)
    - High salience → survives (emotional tagging)
    - Consolidated → episodic copy can go (safe in semantic memory)
    - Low salience + not accessed + old → naturally forgotten
    
    NOT capacity-based (arbitrary) but equilibrium-based (theory-true).
    Typical result: 99% sparse efficiency (494k pruned → 5k surviving).

φ-CURRICULUM (Theory-Derived Context Scaling):
    Context length grows by φ² (≈ 2.618) per stage:
    
        context(stage) = BASE_CONTEXT × φ^(2 × stage)
        samples_per_stage = BASE_SAMPLES × φ^stage
    
    Stages (BASE_CONTEXT=512, BASE_SAMPLES=500k):
        Stage 0: 512 ctx / 500k samples
        Stage 1: 1,340 ctx / 809k samples
        Stage 2: 3,509 ctx / 1.31M samples
        Stage 3: 9,187 ctx / 2.12M samples
        Stage 4: 24,053 ctx / 3.43M samples
        Stage 5: 50,000 ctx (capped)
    
    This ensures sufficient prototype diversity at each scale.

DISTRIBUTED PRIOR (Brain-Analog Smooth Generalization):
    Transformers bake their prior into WEIGHTS. Brains bake their prior into GEOMETRY.
    This system does the latter using theory-true mechanisms:
    
    1. SUPERPOSED ATTRACTORS (Population Coding)
       - Top-K nearest prototypes weighted by φ^(-distance)
       - NOT softmax! φ-kernel is theory-derived
       - A_prior = Σ φ^(-d_i) × support_i × stability_i × A_i
       
    2. FACTORIZED ASSOCIATIVE PRIOR (Cortico-Cortical Projections)
       - Hebbian-learned operator: witness → attractor
       - Â(W) = B @ C⁻¹ @ W  (explicit, inspectable "weights")
       - Global fallback for uncovered regions
       
    3. GEOMETRIC CONFIDENCE (NOT probability!)
       - conf = (d₂ - d₁) / (d₂ + ε)  (margin geometry)
       - High margin → trust local basin
       - Low margin → blend with global prior
       
    Brain analog mapping:
        Cortical maps → Witness space
        Population coding → Superposed attractors
        Attractor networks → Grace basin discovery
        Cortico-cortical proj. → Factorized prior
        Schema cells → Semantic prototypes
        Neuromodulators → salience × grace_stability

Mathematical Foundation:
    Cl(3,1) ≅ M₄(ℝ) with signature η = diag(+1,+1,+1,-1)
    
    Grade structure & Grace scaling:
        Grade 0: 1 scalar       - scaled by 1.0   (SURVIVES)
        Grade 1: 4 vectors      - scaled by φ⁻¹
        Grade 2: 6 bivectors    - scaled by φ⁻²  (VORTICITY)
        Grade 3: 4 trivectors   - scaled by φ⁻³
        Grade 4: 1 pseudoscalar - scaled by φ⁻¹  (SURVIVES, Fibonacci)

Usage (Basic):
    from holographic_v4 import TheoryTrueModel, DreamingSystem
    
    # Create model with equilibrium forgetting
    model = TheoryTrueModel(
        vocab_size=30000,
        context_size=512,  # Will grow via curriculum
        max_attractors=100000,
        noise_std=0.3,  # OPTIMAL
        use_vorticity=True,
    )
    
    # Create dreaming system (self-organizing consolidation)
    dreaming = DreamingSystem(basis=model.basis)
    
    # Train with periodic sleep cycles
    for ctx, target in data:
        result = model.train_step(ctx, target)
    
    # Sleep consolidates unstable episodes into semantic prototypes
    dreaming.sleep(episodes)

Usage (Advanced Capabilities):
    # Theory of Mind - model another agent's beliefs
    from holographic_v4 import AgentModelBuilder, predict_other_belief
    
    other_agent = AgentModelBuilder(dreaming).add_observations(obs).build()
    belief = predict_other_belief(context, other_agent, dreaming)
    
    # Credit Assignment - trace and fix errors
    from holographic_v4 import trace_retrieval, compute_error_attribution, reconsolidate_on_error
    
    trace = trace_retrieval(query, model, dreaming)
    if predicted != actual:
        attribution = compute_error_attribution(predicted, actual, trace, model)
        reconsolidate_on_error(actual, trace, model)
    
    # Curiosity-Driven Active Learning
    from holographic_v4 import curiosity_score, generate_curiosity_query, active_learning_step
    
    curious_query = generate_curiosity_query(model, dreaming)
    best_sample = active_learning_step(model, dreaming, sample_pool)
    
    # Planning and Counterfactual Reasoning
    from holographic_v4 import simulate_action, plan_to_goal, counterfactual
    
    next_state, conf = simulate_action(current_state, action, model)
    plan = plan_to_goal(start_state, goal_state, model, max_depth=10)
    alt_outcome = counterfactual(state, actual_action, hypothetical_action, model)
    
    # Attribute-Object Binding
    from holographic_v4 import bind_attribute_to_object, compare_bindings
    
    red_ball = bind_attribute_to_object(red_emb, ball_emb, model.basis)
    
    # Meta-Learning - adaptive parameters
    from holographic_v4 import LearningState, update_meta_state, compute_adaptive_learning_rate
    
    state = LearningState()
    state = update_meta_state(state, prediction_correct=True, salience=0.8, novelty=0.5)
    rate = compute_adaptive_learning_rate(salience=0.9, novelty=0.7, uncertainty=state.uncertainty)

For GPU acceleration:
    import cupy as cp
    model = TheoryTrueModel(vocab_size=30000, xp=cp)

Test Suite (170 tests total):
    from holographic_v4 import run_all_tests, run_all_tom_tests
    from holographic_v4.curiosity_tests import run_all_curiosity_tests
    # ... (credit_assignment_tests, planning_tests, binding_tests, etc.)
"""

# Constants (φ-derived)
from .constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE,
    PHI_INV_FOUR, PHI_INV_FIVE, PHI_INV_SIX, PHI_INV_SEVEN, PHI_INV_EIGHT,
    GOLDEN_ANGLE,
    CLIFFORD_DIM, MATRIX_DIM,
    GRADE_DIMS, GRADE_INDICES,
    GRACE_SCALES, GRACE_SCALES_FLAT,
)

# Algebra operations
from .algebra import (
    # Basis
    build_gamma_matrices,
    build_clifford_basis,
    build_metric_matrix,
    
    # Operations
    normalize_matrix,
    geometric_product,
    geometric_product_batch,
    wedge_product,
    inner_product,
    compute_vorticity,
    vorticity_magnitude,
    vorticity_signature,
    vorticity_similarity,
    
    # Similarity
    frobenius_similarity,
    frobenius_similarity_batch,
    clifford_adjoint,
    metric_similarity,
    
    # Coefficients
    decompose_to_coefficients,
    reconstruct_from_coefficients,
    
    # Grace
    grace_operator,
    grace_operator_batch,
    grace_flow,
    
    # Initialization
    initialize_embeddings_identity,
    
    # Verification
    verify_gamma_matrices,
    verify_wedge_antisymmetry,
    verify_grace_contraction,
)

# Quotient structure
from .quotient import (
    # Witness
    extract_witness,
    witness_matrix,
    witness_matrix_batch,  # VECTORIZED
    witness_similarity,
    
    # Content & Binding
    extract_content,
    bind,
    bind_batch,  # VECTORIZED
    bind_to_witness,
    unbind_from_witness,
    
    # Spin(3) rotors
    spin3_rotor,
    random_spin3_rotor,
    sandwich,
    
    # Normal form
    normal_form,
    
    # Quotient similarity
    quotient_similarity,
    metric_aware_similarity,
    
    # Adaptive similarity (theory-derived decision)
    adaptive_similarity,
    adaptive_similarity_batch,
    compute_enstrophy,
    
    # Analysis
    grade_energies,
    witness_stability,
    
    # Grace-stability (SELF-ORGANIZING principle)
    grace_stability,
    grace_stability_batch,
    should_consolidate,
    consolidation_urgency,
    
    # Vorticity-weighted decoding (prevents mode collapse)
    vorticity_weighted_scores,
    vorticity_weighted_scores_batch,
    
    # Verification
    verify_witness_invariance,
    verify_normal_form_uniqueness,
)

# Pipeline
from .pipeline import (
    TheoryTrueModel,
)

# Dreaming (generalization via semantic memory + brain-inspired parsimonies)
from .dreaming import (
    # Core system
    DreamingSystem,
    EpisodicEntry,
    SemanticPrototype,
    integrate_dreaming_with_model,
    
    # Brain-inspired parsimonies (original 7)
    compute_salience,
    compute_salience_batch,
    compute_novelty,
    compute_novelty_batch,
    compute_prediction_error,
    compute_combined_priority,
    
    # Compression
    CompressedEpisode,
    compress_episode,
    compress_episodes_batch,
    
    # Pruning
    prune_semantic_memory,
    prune_attractor_map,
    
    # Interference management
    merge_prototypes,
    manage_interference,
    
    # Reconsolidation
    ReconsolidationTracker,
    reconsolidate_attractor,
    reconsolidate_semantic_prototype,
    
    # Working memory
    apply_working_memory_gate,
    gated_context_representation,
    WorkingMemoryBuffer,
    
    # Pattern Completion (new)
    pattern_complete,
    pattern_complete_batch,
    
    # Predictive Coding (new)
    predict_from_memory,
    compute_prediction_residual,
    predictive_encode,
    predictive_encode_batch,
    
    # Sequence Replay (new)
    TemporalTransition,
    compute_transition_vorticity,
    record_transition,
    TransitionBuffer,
    replay_transitions_during_rem,
    
    # Pseudo-Rehearsal (new)
    generate_pseudo_episode,
    generate_pseudo_episodes_batch,
    interleave_with_pseudo_rehearsal,
    
    # Inhibition of Return (new)
    RetrievalHistory,
    apply_inhibition_of_return,
    retrieve_with_inhibition,
)

# Resonance (equilibrium dynamics)
from .resonance import (
    TheoryTrueRetriever,
    evolve_to_equilibrium,
    resonance,
    find_resonant_prototype,
    grace_basin_discovery,
    grace_basin_retrieve,
    distributed_prior_retrieve,  # Brain-analog population coding
)

# Holographic Memory (Theory-True Storage via Clifford Superposition)
# HISTORICAL NOTE: The original implementation used hash(context.tobytes())
# which was NOT theory-true. True holographic memory uses superposition.
from .holographic_memory import (
    # Core classes
    HolographicMemory,         # True superposition-based storage
    VorticityWitnessIndex,     # 8D episodic indexing (~1000+ buckets) [v4.17.0]
    CanonicalSemanticIndex,    # 2D semantic indexing (generalization) [v4.22.0]
    HybridHolographicMemory,   # Dual indexing: episodic + semantic [v4.22.0]
    MultiTimescaleMemory,      # φ-parameterized decay (fast/medium/slow) [v4.8.0]
    
    # Clifford operations
    clifford_reversion,      # M̃ = conjugate
    clifford_inverse,        # M⁻¹ = M̃ / (M × M̃)
    
    # Witness utilities
    compute_context_witness,
    witness_distance as holographic_witness_distance,  # Avoid name collision
    witness_similarity as holographic_witness_similarity,
    
    # Capacity signals (v4.8.0)
    compute_witness_entropy,  # H_w = capacity saturation signal
    is_memory_saturated,      # Consolidation trigger
    
    # Multi-item retrieval (v4.8.0)
    iterative_unbind,         # Retrieve multiple items from superposition
)

# Distributed Prior (Theory-True Smooth Generalization)
from .distributed_prior import (
    # Core primitives
    witness_distance,
    phi_kernel,  # NOT softmax!
    extended_witness,
    
    # Retrieval methods
    superposed_attractor_prior,  # Smooth interpolation
    prior_field_retrieve,  # Green's function
    retrieve_with_factorized_prior,
    retrieve_with_distributed_prior,
    
    # Factorized prior class (Hebbian "weights")
    FactorizedAssociativePrior,
    
    # Coverage metrics (auditable!)
    compute_basin_coverage,
)

# Theory of Mind (Perspective transformation via witness-relative binding)
from .theory_of_mind import (
    # Witness inference
    infer_witness_from_observations,
    infer_witness_from_observations_weighted,
    
    # Agent modeling
    AgentModel,
    AgentModelBuilder,
    
    # Core ToM operations
    theory_of_mind,
    predict_other_belief,
    explain_other_action,
    
    # Recursive ToM
    recursive_tom,
    
    # Utilities
    merge_agent_models,
    # Note: witness_distance already imported from distributed_prior
)

# Credit Assignment v2 (Lean integration with FractalGenerativeMemory)
from .credit_assignment import (
    ErrorRecord,
    ReconsolidationConfig,
    CreditAssignmentTracker,
    create_credit_assigned_learn,
    batch_credit_assignment,
)

# Representation Learning (Embedding drift)
from .representation_learning import (
    compute_embedding_gradient,
    update_embedding,
    cluster_embeddings_by_retrieval,
    compute_drift_step,
    # v4.23.0: Contrastive learning
    contrastive_embedding_update,
    auto_contrastive_update,
)

# Recursive Computation (Iterative retrieval and geometric search)
from .recursive_computation import (
    IterationTrace,
    iterative_retrieval,
    geometric_search,
    recursive_decomposition,
    deep_retrieval,
)

# Planning (Causal reasoning and counterfactual simulation)
from .planning import (
    PlanStep,
    Plan,
    simulate_action,
    plan_to_goal,
    counterfactual,
    evaluate_plan,
)

# Attribute Binding (Clifford grade-based binding)
from .binding import (
    bind_attribute_to_object,
    extract_object_from_bound,
    compare_bindings,
    bind_multiple_attributes,
    binding_signature,
    binding_similarity,
)

# Grounding (Perception to Clifford mapping)
from .grounding import (
    PerceptionEncoder,
    ground_token,
    perceptual_similarity,
    batch_encode,
)

# Meta-Learning (Adaptive parameters)
from .meta_learning import (
    LearningState,
    compute_adaptive_learning_rate,
    compute_adaptive_consolidation,
    update_meta_state,
    create_learning_state,
    phi_scaled_learning_rate,
    verified_retrieve,
    get_adaptive_parameters,
)

# Adaptive Memory (v4.29.0: Integrated production-ready system)
from .adaptive_memory import (
    AdaptiveMemory,
    AdaptiveMemoryConfig,
    create_adaptive_memory,
)

# Curiosity / Active Learning (Theory-derived exploration)
from .curiosity import (
    curiosity_score,
    curiosity_score_with_meta,
    estimate_information_gain,
    sample_basin_boundary,
    generate_curiosity_query,
    active_learning_step,
    rank_samples_by_curiosity,
    should_explore,
    exploration_rate,
)

# Semantic Prototype Memory (position-weighted approach)
from .semantic_prototype import (
    SemanticPrototypeMemory,
    PositionWisePrototype,
    position_weighted_similarity,
    compute_position_weights_from_variance,
    embedding_cosine_similarity,
)

# Predictiveness-Based Semantic Extraction (theory-true solution)
from .predictiveness import (
    PredictivenessTracker,
    TokenStatistics,
    SemanticPrototypeBuilder,
    compute_semantic_context,
    semantic_retrieve,
    integrate_predictiveness,
    verify_semantic_extraction,
)

# Stability Theorems (v4.5.0)
from .representation_learning import (
    compute_lyapunov_function,
    verify_lyapunov_decrease,
    verify_lyapunov_stability,
    prove_contraction_bound,
    EmbeddingLearner,
)

from .planning import (
    verify_error_accumulation,
    compute_planning_error_bound,
)

from .dreaming import (
    analyze_memory_scaling,
    estimate_memory_capacity,
    measure_prototype_entropy,
)

# Multi-Scale Resonance (φ-structured context analysis)
from .multiscale_resonance import (
    # Constants
    FIBONACCI_SCALES,
    OPTIMAL_SCALES,
    
    # Core functions
    compute_context_at_scale,
    compute_multiscale_stability,
    compute_enstrophy_trend,
    compute_phi_ratio,
    compute_phi_ratio_batch,
    
    # Diagnostics
    diagnose_phi_structure,
    should_use_multiscale,
)

# Vorticity Features (theory-validated)
from .vorticity_features import (
    CirculationResult,
    compute_loop_circulation,
    is_paraphrase_loop,
    VorticityTrace,
    VorticityTracker,
    GenerationQualityMetrics,
    compute_generation_quality,
    check_semantic_invariance,
    VorticityHealth,
    diagnose_vorticity_health,
    # Brain-like coherence (replaces failed FFT)
    VorticityCoherenceMetrics,
    compute_plv,
    compute_directional_stability,
    compute_vorticity_predictability,
    compute_vorticity_autocorrelation,
    compute_vorticity_coherence,
)

# Tests - import dynamically to avoid circular import issues
# from .tests import run_all_tests  # Removed - use pytest instead
try:
    from .theory_of_mind_tests import run_all_tom_tests
except ImportError:
    run_all_tom_tests = None  # Optional test module

__version__ = "4.29.0"  # 2026-01-13: ADAPTIVE MEMORY (META-LEARNING INTEGRATED)
# v4.22.0: Dual indexing (episodic 8D + semantic 2D)
# v4.23.0: Contrastive embedding learning (Hebbian, theory-true)
# v4.24.0: Nested Fractal Torus (16^N scaling, φ-offset, GraceInverse)
# v4.25.0: Generative Memory (accumulation + sampling, orthogonalized embeddings)
# v4.26.0: FractalGenerativeMemory (8/8 tests), PPL 470, 15+ token context
# v4.27.0: ToroidalAttention + DreamCycle (structural attention + topological dreaming)
# v4.28.0: Credit Assignment v2 (lean, O(1), φ-scaled reconsolidation)
# v4.29.0: AdaptiveMemory (novelty/uncertainty modulation, φ-scaled curriculum)
# v4.27.0: ToroidalAttention (7/7), DreamCycle (7/7), Integration (5/5) — READY FOR SCALE
# v4.24.0: Nested Fractal Torus Architecture
# - torus/: φ-offset phase distribution, T² coordinates, interaction tensor
# - fractal/: nested_torus, grand_equilibrium, downward_projection
# - dreaming_enhanced.py: Non-REM consolidation, REM recombination, paradox resolution
# - GraceInverse: Inflation operator for generation
# - 16^N scalable capacity, 45 pytest tests + 11 integration tests pass
# - Deprecated WitnessIndex (only ~12 buckets = useless)
# - VorticityWitnessIndex is now the ONLY index
# - Removed legacy index flags (use_gpu_witness, use_torus_symmetry, use_vorticity_index)
# - Removed fallback_to_full from compute_semantic_context
# - Simplified HybridHolographicMemory.create

__all__ = [
    # Constants (φ-derived)
    'PHI', 'PHI_INV', 'PHI_INV_SQ', 'PHI_INV_CUBE',
    'PHI_INV_FOUR', 'PHI_INV_FIVE', 'PHI_INV_SIX', 'PHI_INV_SEVEN', 'PHI_INV_EIGHT',
    'GOLDEN_ANGLE',
    'CLIFFORD_DIM', 'MATRIX_DIM',
    'GRADE_DIMS', 'GRADE_INDICES',
    'GRACE_SCALES', 'GRACE_SCALES_FLAT',
    
    # Algebra
    'build_gamma_matrices',
    'build_clifford_basis',
    'build_metric_matrix',
    'normalize_matrix',
    'geometric_product',
    'geometric_product_batch',
    'wedge_product',
    'inner_product',
    'compute_vorticity',
    'vorticity_magnitude',
    'vorticity_signature',
    'vorticity_similarity',
    'frobenius_similarity',
    'frobenius_similarity_batch',
    'clifford_adjoint',
    'metric_similarity',
    'decompose_to_coefficients',
    'reconstruct_from_coefficients',
    'grace_operator',
    'grace_operator_batch',
    'grace_flow',
    'initialize_embeddings_identity',
    'verify_gamma_matrices',
    'verify_wedge_antisymmetry',
    'verify_grace_contraction',
    
    # Quotient
    'extract_witness',
    'witness_matrix',
    'witness_similarity',
    'extract_content',
    'bind',
    'bind_batch',
    'witness_matrix_batch',
    'spin3_rotor',
    'random_spin3_rotor',
    'sandwich',
    'normal_form',
    'quotient_similarity',
    'metric_aware_similarity',
    'adaptive_similarity',
    'adaptive_similarity_batch',
    'compute_enstrophy',
    'grade_energies',
    'witness_stability',
    # Grace-stability (self-organizing)
    'grace_stability',
    'grace_stability_batch',
    'should_consolidate',
    'consolidation_urgency',
    # Vorticity-weighted decoding
    'vorticity_weighted_scores',
    'vorticity_weighted_scores_batch',
    'verify_witness_invariance',
    'verify_normal_form_uniqueness',
    
    # Pipeline
    'TheoryTrueModel',
    
    # Dreaming + Brain-Inspired Parsimonies (12 total)
    'DreamingSystem',
    'EpisodicEntry',
    'SemanticPrototype',
    'integrate_dreaming_with_model',
    # Salience & Novelty
    'compute_salience',
    'compute_salience_batch',
    'compute_novelty',
    'compute_novelty_batch',
    'compute_prediction_error',
    'compute_combined_priority',
    # Compression
    'CompressedEpisode',
    'compress_episode',
    'compress_episodes_batch',
    # Pruning
    'prune_semantic_memory',
    'prune_attractor_map',
    # Interference
    'merge_prototypes',
    'manage_interference',
    # Reconsolidation
    'ReconsolidationTracker',
    'reconsolidate_attractor',
    'reconsolidate_semantic_prototype',
    # Working Memory
    'apply_working_memory_gate',
    'gated_context_representation',
    'WorkingMemoryBuffer',
    # Pattern Completion
    'pattern_complete',
    'pattern_complete_batch',
    # Predictive Coding
    'predict_from_memory',
    'compute_prediction_residual',
    'predictive_encode',
    'predictive_encode_batch',
    # Sequence Replay
    'TemporalTransition',
    'compute_transition_vorticity',
    'record_transition',
    'TransitionBuffer',
    'replay_transitions_during_rem',
    # Pseudo-Rehearsal
    'generate_pseudo_episode',
    'generate_pseudo_episodes_batch',
    'interleave_with_pseudo_rehearsal',
    # Inhibition of Return
    'RetrievalHistory',
    'apply_inhibition_of_return',
    'retrieve_with_inhibition',
    
    # Resonance
    'TheoryTrueRetriever',
    'evolve_to_equilibrium',
    'grace_basin_discovery',
    'grace_basin_retrieve',
    'distributed_prior_retrieve',
    
    # Holographic Memory (Theory-True Storage via Clifford Superposition)
    'HolographicMemory',
    'VorticityWitnessIndex',     # v4.17.0: 8D episodic (exact retrieval)
    'CanonicalSemanticIndex',    # v4.22.0: 2D semantic (generalization)
    'HybridHolographicMemory',   # v4.22.0: Dual indexing
    'MultiTimescaleMemory',      # v4.8.0: φ-decay working/episodic/semantic
    'clifford_reversion',
    'clifford_inverse',
    'compute_context_witness',
    'holographic_witness_distance',
    'holographic_witness_similarity',
    'compute_witness_entropy',   # v4.8.0: Capacity saturation signal
    'is_memory_saturated',       # v4.8.0: Consolidation trigger
    'iterative_unbind',          # v4.8.0: Multi-item retrieval
    
    # Distributed Prior (Theory-True Smooth Generalization)
    'witness_distance',
    'phi_kernel',
    'extended_witness',
    'superposed_attractor_prior',
    'prior_field_retrieve',
    'retrieve_with_factorized_prior',
    'retrieve_with_distributed_prior',
    'FactorizedAssociativePrior',
    'compute_basin_coverage',
    
    # Theory of Mind
    'infer_witness_from_observations',
    'infer_witness_from_observations_weighted',
    'AgentModel',
    'AgentModelBuilder',
    'theory_of_mind',
    'predict_other_belief',
    'explain_other_action',
    'recursive_tom',
    'merge_agent_models',
    'bind_to_witness',
    'unbind_from_witness',
    
    # Credit Assignment
    'ProvenanceTrace',
    'ProvenanceTracker',
    'trace_retrieval',
    'compute_error_attribution',
    'reconsolidate_on_error',
    
    # Representation Learning
    'compute_embedding_gradient',
    'update_embedding',
    'cluster_embeddings_by_retrieval',
    'compute_drift_step',
    'contrastive_embedding_update',  # v4.23.0
    'auto_contrastive_update',       # v4.23.0
    
    # Recursive Computation
    'IterationTrace',
    'iterative_retrieval',
    'geometric_search',
    'recursive_decomposition',
    'deep_retrieval',
    
    # Planning
    'PlanStep',
    'Plan',
    'simulate_action',
    'plan_to_goal',
    'counterfactual',
    'evaluate_plan',
    
    # Attribute Binding
    'bind_attribute_to_object',
    'extract_object_from_bound',
    'compare_bindings',
    'bind_multiple_attributes',
    'binding_signature',
    'binding_similarity',
    
    # Grounding
    'PerceptionEncoder',
    'ground_token',
    'perceptual_similarity',
    'batch_encode',
    
    # Meta-Learning
    'LearningState',
    'compute_adaptive_learning_rate',
    'compute_adaptive_consolidation',
    'update_meta_state',
    'create_learning_state',
    'get_adaptive_parameters',
    
    # Curiosity / Active Learning
    'curiosity_score',
    'curiosity_score_with_meta',
    'estimate_information_gain',
    'sample_basin_boundary',
    'generate_curiosity_query',
    'active_learning_step',
    'rank_samples_by_curiosity',
    'should_explore',
    'exploration_rate',
    
    # Semantic Prototype Memory (position-weighted approach)
    'SemanticPrototypeMemory',
    'PositionWisePrototype',
    'position_weighted_similarity',
    'compute_position_weights_from_variance',
    'embedding_cosine_similarity',
    
    # Predictiveness-Based Semantic Extraction (theory-true solution)
    'PredictivenessTracker',
    'TokenStatistics',
    'SemanticPrototypeBuilder',
    'compute_semantic_context',
    'semantic_retrieve',
    'integrate_predictiveness',
    'verify_semantic_extraction',
    
    # Stability Theorems (v4.5.0)
    # Lyapunov Stability (Representation Learning)
    'compute_lyapunov_function',
    'verify_lyapunov_decrease',
    'verify_lyapunov_stability',
    'prove_contraction_bound',
    'EmbeddingLearner',
    
    # Error Accumulation (Planning)
    'verify_error_accumulation',
    'compute_planning_error_bound',
    
    # Memory Scaling
    'analyze_memory_scaling',
    'estimate_memory_capacity',
    'measure_prototype_entropy',
    
    # Vorticity Features (v4.7.0 - theory-validated)
    'CirculationResult',
    'compute_loop_circulation',
    'is_paraphrase_loop',
    'VorticityTrace',
    'VorticityTracker',
    'GenerationQualityMetrics',
    'compute_generation_quality',
    'check_semantic_invariance',
    'VorticityHealth',
    'diagnose_vorticity_health',
    
    # Brain-like Coherence (v4.7.0 - replaces FFT)
    'VorticityCoherenceMetrics',
    'compute_plv',
    'compute_directional_stability',
    'compute_vorticity_predictability',
    'compute_vorticity_autocorrelation',
    'compute_vorticity_coherence',
    
    # Multi-Scale Resonance (φ-structured context analysis)
    'FIBONACCI_SCALES',
    'OPTIMAL_SCALES',
    'compute_context_at_scale',
    'compute_multiscale_stability',
    'compute_enstrophy_trend',
    'compute_phi_ratio',
    'compute_phi_ratio_batch',
    'diagnose_phi_structure',
    'should_use_multiscale',
    
    # Tests
    'run_all_tests',
    'run_all_tom_tests',
]
