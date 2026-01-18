"""
Higher-level cognitive capabilities.

ALL THEORY-TRUE. NO BACKPROP.

MODULES:
- credit_assignment: Provenance tracing and targeted reconsolidation ✅ INTEGRATED
- meta_learning: Adaptive φ-derived learning rates ✅ INTEGRATED
- distributed_prior: Geometric generalization via φ-kernels ✅ INTEGRATED

STANDALONE (work with any model providing .xp, .basis, .embed()):
- theory_of_mind: Perspective transformation and agent modeling ✅ WORKS
  - infer_witness_from_observations, bind_to_witness, transform_perspective

NEEDS PORTING (written for TheoryTrueModel, need HolographicMemory adapter):
- curiosity: Metacognition and information gain estimation ⚠️ PARTIAL
  - Uses model.num_attractors, model.attractor_matrices (TheoryTrueModel API)
- planning: Causal simulation and counterfactual reasoning ⚠️ PARTIAL
  - Uses model.num_attractors, model.retrieve() (TheoryTrueModel API)

NOTE: The theory in curiosity.py and planning.py is correct. They just need
an adapter layer to work with HolographicMemory's different storage model.
"""

from .credit_assignment import (
    CreditAssignmentTracker,
    ReconsolidationConfig,
    ErrorRecord,
    create_credit_assigned_learn,
    batch_credit_assignment,
)

from .meta_learning import (
    LearningState,
    create_learning_state,
    update_meta_state,
    compute_adaptive_learning_rate,
    compute_adaptive_consolidation,
    phi_scaled_learning_rate,
    get_adaptive_parameters,
)

from .distributed_prior import (
    phi_kernel,
    witness_distance,
    extended_witness,
    superposed_attractor_prior,
    FactorizedAssociativePrior,
    retrieve_with_distributed_prior,
)

from .curiosity import (
    curiosity_score,
    curiosity_score_with_meta,
    estimate_information_gain,
    sample_basin_boundary,
    generate_curiosity_query,
    active_learning_step,
    rank_samples_by_curiosity,
)

from .planning import (
    simulate_action,
    plan_to_goal,
    counterfactual,
    counterfactual_trajectory,
    plan_with_subgoals,
    evaluate_plan,
    PlanStep,
    Plan,
)

from .theory_of_mind import (
    infer_witness_from_observations,
    bind_to_witness,
    unbind_from_witness,
    transform_perspective,
    AgentModel,
    AgentModelBuilder,
    theory_of_mind,
    predict_other_belief,
)

__all__ = [
    # Credit Assignment
    'CreditAssignmentTracker', 'ReconsolidationConfig', 'ErrorRecord',
    'create_credit_assigned_learn', 'batch_credit_assignment',
    # Meta Learning
    'LearningState', 'create_learning_state', 'update_meta_state',
    'compute_adaptive_learning_rate', 'compute_adaptive_consolidation',
    'phi_scaled_learning_rate', 'get_adaptive_parameters',
    # Distributed Prior
    'phi_kernel', 'witness_distance', 'extended_witness',
    'superposed_attractor_prior', 'FactorizedAssociativePrior',
    'retrieve_with_distributed_prior',
    # Curiosity
    'curiosity_score', 'curiosity_score_with_meta', 'estimate_information_gain',
    'sample_basin_boundary', 'generate_curiosity_query',
    'active_learning_step', 'rank_samples_by_curiosity',
    # Planning
    'simulate_action', 'plan_to_goal', 'counterfactual',
    'counterfactual_trajectory', 'plan_with_subgoals', 'evaluate_plan',
    'PlanStep', 'Plan',
    # Theory of Mind
    'infer_witness_from_observations', 'bind_to_witness', 'unbind_from_witness',
    'transform_perspective', 'AgentModel', 'AgentModelBuilder',
    'theory_of_mind', 'predict_other_belief',
]
