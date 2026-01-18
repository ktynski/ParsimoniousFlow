"""
Theory of Mind Module — Geometric Perspective Transformation
============================================================

Implements Theory of Mind (ToM) as a natural extension of the witness+binding
machinery. ToM is the ability to model other agents' mental states.

CORE INSIGHT:
    ToM is NOT a special module - it's applying witness-relative binding
    to models of other agents. Just as binding transforms "what is there"
    into "what I perceive," ToM binding transforms "what is there" into
    "what THEY perceive."

THEORY DERIVATION:
    1. Witness = gauge-invariant self-model (scalar + pseudoscalar)
    2. Binding = framing content relative to a witness perspective
    3. ToM = Binding(Content, OtherWitness) + GraceFlow(OtherBasins)

    This is geometrically elegant: ToM is a coordinate transformation
    in witness space.

BRAIN ANALOGS:
    - Witness inference → Mirror neuron system (inferring others' states)
    - AgentModel → Temporoparietal junction (maintaining other models)
    - ToM operation → Medial prefrontal cortex (mentalizing)
    - Belief prediction → Right TPJ (belief attribution)

DEVELOPMENTAL PROGRESSION:
    1. No ToM (infant): Only self-witness, no models of others
    2. Proto-ToM: Recognize others HAVE witnesses
    3. Basic ToM: Maintain simple other-witness models
    4. Advanced ToM: Full semantic memory simulation
    5. Recursive ToM: Nested models ("I know that you know")

COMPONENTS:
    - infer_witness_from_observations(): Extract other's witness from behavior
    - bind_to_witness(): Transform content to a specific perspective
    - unbind_from_witness(): Inverse transformation
    - AgentModel: Data structure for other-agent models
    - AgentModelBuilder: Incrementally build models from observations
    - theory_of_mind(): Core ToM computation
    - predict_other_belief(): Predict what another would believe

NO ARBITRARY OPERATIONS:
    - All operations use φ-derived parameters
    - Grace dynamics for stabilization
    - No softmax, no tuned thresholds
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

# TYPE_CHECKING import to avoid circular dependency
if TYPE_CHECKING:
    from holographic_v4.pipeline import TheoryTrueModel

from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE
from holographic_v4.algebra import (
    build_clifford_basis,
    normalize_matrix,
    grace_operator,
    decompose_to_coefficients,
    vorticity_signature,
    vorticity_similarity,
)
from holographic_v4.quotient import (
    extract_witness,
    witness_matrix,
    extract_content,
    bind,
    grace_stability,
)
from holographic_v4.resonance import evolve_to_equilibrium, grace_basin_discovery
from holographic_v4.dreaming import SemanticMemory

Array = np.ndarray
ArrayModule = type(np)


# =============================================================================
# WITNESS INFERENCE — Extracting Other's Perspective from Observations
# =============================================================================

def infer_witness_from_observations(
    observations: List[Array],
    basis: Array,
    xp: ArrayModule = np,
) -> Tuple[float, float]:
    """
    Infer another agent's witness configuration from their behavior.
    
    THEORY:
        The witness is what's INVARIANT across an agent's expressions.
        By averaging witnesses across multiple observations, transient
        content cancels out and the stable core (witness) emerges.
        
        This is analogous to how we infer others' personalities from
        observing regularities in their behavior.
    
    IMPLEMENTATION:
        1. Extract witness from each observation
        2. Average the witnesses (stable component accumulates)
        
        NOTE: We do NOT apply Grace after averaging because witness
        components (scalar + pseudoscalar) are already the Grace-stable
        part. Applying Grace would scale them unnecessarily.
        
    Args:
        observations: List of [4, 4] matrices representing observed behavior
        basis: [16, 4, 4] Clifford basis
        xp: Array module (numpy or cupy)
        
    Returns:
        (scalar, pseudoscalar) - the inferred witness
    """
    if not observations:
        return (0.0, 0.0)
    
    # Extract witness from each observation
    witnesses = []
    for obs in observations:
        w = extract_witness(obs, basis, xp)
        witnesses.append(w)
    
    # Average the witnesses - this IS the stable pattern
    # No Grace needed because witness IS what survives Grace
    avg_scalar = sum(w[0] for w in witnesses) / len(witnesses)
    avg_pseudo = sum(w[1] for w in witnesses) / len(witnesses)
    
    return (avg_scalar, avg_pseudo)


def infer_witness_from_observations_weighted(
    observations: List[Array],
    weights: Optional[List[float]],
    basis: Array,
    xp: ArrayModule = np,
) -> Tuple[float, float]:
    """
    Infer witness with weighted observations.
    
    More recent or salient observations can be weighted higher.
    
    Args:
        observations: List of [4, 4] matrices
        weights: Optional weights for each observation (None = uniform)
        basis: Clifford basis
        xp: Array module
        
    Returns:
        (scalar, pseudoscalar) - weighted inferred witness
    """
    if not observations:
        return (0.0, 0.0)
    
    if weights is None:
        weights = [1.0] * len(observations)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight < 1e-12:
        return (0.0, 0.0)
    weights = [w / total_weight for w in weights]
    
    # Weighted average of witnesses
    avg_scalar = 0.0
    avg_pseudo = 0.0
    for obs, weight in zip(observations, weights):
        w = extract_witness(obs, basis, xp)
        avg_scalar += weight * w[0]
        avg_pseudo += weight * w[1]
    
    # Grace stabilize
    W_matrix = avg_scalar * basis[0] + avg_pseudo * basis[15]
    W_stable = grace_operator(W_matrix, basis, xp)
    
    return extract_witness(W_stable, basis, xp)


# =============================================================================
# PERSPECTIVE BINDING — Transforming Content Between Perspectives
# =============================================================================

def bind_to_witness(
    content: Array,
    witness: Array,
    basis: Array,
    lmbda: float = PHI_INV,
    xp: ArrayModule = np,
) -> Array:
    """
    Bind content relative to a SPECIFIC witness.
    
    THEORY:
        Standard bind() extracts the witness FROM the content.
        This function uses an EXTERNAL witness, enabling perspective
        transformation: "How would content appear to someone with
        this witness configuration?"
        
        𝓑_W(C) = W + λ · w · C · w̃
        
        where w = normalized witness pointer.
    
    DIFFERENCE FROM bind():
        bind(M) uses W(M) - the content's OWN witness
        bind_to_witness(C, W) uses W - an EXTERNAL witness
        
    This is the key operation for Theory of Mind:
        - Self perception: bind(content, basis) [uses own witness]
        - Other perception: bind_to_witness(content, other_witness, basis)
        
    Args:
        content: [4, 4] content matrix (grades 1-3, or full matrix)
        witness: [4, 4] witness matrix to bind relative to
        basis: [16, 4, 4] Clifford basis
        lmbda: Binding strength (default: φ⁻¹)
        xp: Array module
        
    Returns:
        [4, 4] bound matrix from the perspective of the given witness
    """
    # Normalize witness for sandwich operation
    w_norm = xp.sqrt(xp.sum(witness * witness))
    if w_norm < 1e-12:
        return content.copy()
    
    w = witness / w_norm
    
    # Extract content if not already (remove any witness component from input)
    # This ensures we're binding pure content
    input_witness = witness_matrix(content, basis, xp)
    pure_content = content - input_witness
    
    # Sandwich: w · C · w^T
    bound_content = w @ pure_content @ w.T
    
    # Result = witness + λ * bound_content
    result = witness + lmbda * bound_content
    
    return result


def unbind_from_witness(
    bound: Array,
    witness: Array,
    basis: Array,
    lmbda: float = PHI_INV,
    xp: ArrayModule = np,
) -> Array:
    """
    Approximate inverse of bind_to_witness.
    
    THEORY:
        Unbinding attempts to recover the original content from a
        bound representation. 
        
        Since bind does: result = W + λ * (w @ C @ w.T)
        We invert: C ≈ w.T @ ((result - W) / λ) @ w
        
        But since w is orthogonal (w @ w.T = I for normalized witness),
        the inverse sandwich is exact.
        
    This enables:
        - Extracting "objective" content from a perspective-bound view
        - Transforming between perspectives: unbind(other) → rebind(self)
        
    Args:
        bound: [4, 4] bound matrix
        witness: [4, 4] witness matrix that was used for binding
        basis: [16, 4, 4] Clifford basis
        lmbda: Binding strength used (default: φ⁻¹)
        xp: Array module
        
    Returns:
        [4, 4] approximately recovered content
    """
    # Normalize witness
    w_norm = xp.sqrt(xp.sum(witness * witness))
    if w_norm < 1e-12:
        return bound.copy()
    
    w = witness / w_norm
    
    # Remove witness contribution
    bound_content = (bound - witness) / lmbda
    
    # Inverse sandwich: w.T @ (w @ C @ w.T) @ w = C
    # Because w.T @ w = I (normalized)
    recovered_content = w.T @ bound_content @ w
    
    # Add back the original witness from the bound content
    # The full recovery is: recovered_content + original_witness
    # But we don't know original witness, so return content + extracted witness
    input_witness = witness_matrix(bound, basis, xp)
    recovered = recovered_content + input_witness
    
    return recovered


def transform_perspective(
    content: Array,
    from_witness: Array,
    to_witness: Array,
    basis: Array,
    xp: ArrayModule = np,
) -> Array:
    """
    Transform content from one perspective to another.
    
    THEORY:
        This is the core perspective-taking operation:
        1. Unbind from source perspective
        2. Rebind to target perspective
        
        Useful for: "What does X look like from Y's perspective?"
        
    Args:
        content: [4, 4] content in from_witness perspective
        from_witness: [4, 4] source perspective
        to_witness: [4, 4] target perspective
        basis: Clifford basis
        xp: Array module
        
    Returns:
        [4, 4] content in to_witness perspective
    """
    # Unbind from source
    objective = unbind_from_witness(content, from_witness, basis, xp=xp)
    
    # Rebind to target
    transformed = bind_to_witness(objective, to_witness, basis, xp=xp)
    
    return transformed


# =============================================================================
# AGENT MODEL — Data Structure for Modeling Other Agents
# =============================================================================

@dataclass
class AgentModel:
    """
    Model of another agent's cognitive state.
    
    THEORY:
        An agent is characterized by:
        1. Witness: Their perspective (how they frame content)
        2. Semantic Memory: Their belief structure (attractor basins)
        3. Vorticity Pattern: Their cognitive "grammar" (thought flow)
        
        This is sufficient to predict their behavior because:
        - Witness determines HOW they see things
        - Semantic memory determines WHAT they believe
        - Vorticity pattern determines HOW they reason
        
    BRAIN ANALOG:
        This corresponds to our mental model of another person,
        maintained in the temporoparietal junction (TPJ) and
        medial prefrontal cortex (mPFC).
        
    Attributes:
        witness: (scalar, pseudo) - their perspective
        semantic_memory: Their belief structure (optional)
        vorticity_pattern: Their characteristic vorticity signature (optional)
        confidence: How certain we are about this model [0, 1]
        observation_count: Number of observations used to build this model
    """
    witness: Tuple[float, float]
    semantic_memory: Optional[SemanticMemory] = None
    vorticity_pattern: Optional[Array] = None
    confidence: float = 0.0
    observation_count: int = 0
    
    def witness_matrix(self, basis: Array, xp: ArrayModule = np) -> Array:
        """Get witness as a matrix."""
        return self.witness[0] * basis[0] + self.witness[1] * basis[15]
    
    def is_similar_to(self, other: 'AgentModel', threshold: float = PHI_INV_CUBE) -> bool:
        """Check if two agent models are similar (same person?). φ-derived threshold."""
        witness_diff = (
            abs(self.witness[0] - other.witness[0]) +
            abs(self.witness[1] - other.witness[1])
        )
        return witness_diff < threshold


class AgentModelBuilder:
    """
    Incrementally builds an AgentModel from observations.
    
    THEORY:
        We infer another agent's cognitive state by observing:
        1. Their expressions (matrices) → witness pattern
        2. Their context-target pairs → semantic beliefs
        3. Their sequential patterns → vorticity signature
        
        Each observation refines our estimate of who they are.
        
    USAGE:
        builder = AgentModelBuilder(basis)
        for obs in observations:
            builder.observe(obs)
        model = builder.build()
        
    BRAIN ANALOG:
        This is like forming an impression of someone over
        multiple interactions. Early impressions are uncertain;
        repeated interactions build confidence.
    """
    
    def __init__(self, basis: Array, xp: ArrayModule = np):
        """
        Initialize builder.
        
        Args:
            basis: [16, 4, 4] Clifford basis
            xp: Array module
        """
        self.basis = basis
        self.xp = xp
        
        # Accumulated observations
        self.observations: List[Array] = []
        self.contexts: List[List[int]] = []
        self.targets: List[int] = []
        self.vorticity_signatures: List[Array] = []
        
    def observe(
        self,
        observation: Array,
        context: Optional[List[int]] = None,
        target: Optional[int] = None,
    ) -> None:
        """
        Add an observation of the other agent's behavior.
        
        Args:
            observation: [4, 4] matrix representing their output/behavior
            context: Optional context tokens they were responding to
            target: Optional target they predicted
        """
        self.observations.append(observation.copy())
        
        if context is not None:
            self.contexts.append(context)
        if target is not None:
            self.targets.append(target)
            
        # Compute vorticity signature if possible
        # If this fails, it's a bug - don't silently ignore
        try:
            vort_sig = vorticity_signature(observation, self.basis, self.xp)
            self.vorticity_signatures.append(vort_sig)
        except (ValueError, TypeError, AttributeError) as e:
            # Vorticity computation failed - this indicates invalid input or missing dependencies
            # CRITICAL: Don't silently ignore - raise error to catch bugs early
            raise RuntimeError(
                f"Vorticity signature computation failed in AgentModelBuilder.observe: {e}. "
                "This indicates invalid input or missing dependencies."
            ) from e
        except Exception as e:
            # Unexpected error - should not be silently ignored
            raise RuntimeError(f"Unexpected error computing vorticity signature: {e}") from e
    
    def build(self) -> AgentModel:
        """
        Build current best estimate of the agent model.
        
        Returns:
            AgentModel with inferred characteristics
        """
        xp = self.xp
        
        # Infer witness
        if self.observations:
            witness = infer_witness_from_observations(
                self.observations, self.basis, xp
            )
        else:
            witness = (0.0, 0.0)
        
        # Compute average vorticity pattern
        vorticity_pattern = None
        if self.vorticity_signatures:
            # Stack and average
            vort_stack = xp.stack(self.vorticity_signatures)
            vorticity_pattern = xp.mean(vort_stack, axis=0)
        
        # Confidence based on observation count
        # More observations → higher confidence (saturates at ~50)
        n_obs = len(self.observations)
        confidence = min(1.0, n_obs / 50.0)
        
        # Build semantic memory from context-target pairs if available
        semantic_memory = None
        if self.contexts and self.targets and len(self.contexts) == len(self.targets):
            semantic_memory = SemanticMemory(basis=self.basis, xp=xp)
            # Note: Full semantic memory building would require more infrastructure
            # For now, we just create an empty one that can be populated
        
        return AgentModel(
            witness=witness,
            semantic_memory=semantic_memory,
            vorticity_pattern=vorticity_pattern,
            confidence=confidence,
            observation_count=n_obs,
        )
    
    def reset(self) -> None:
        """Clear all observations."""
        self.observations = []
        self.contexts = []
        self.targets = []
        self.vorticity_signatures = []


# =============================================================================
# CORE TOM OPERATIONS
# =============================================================================

def theory_of_mind(
    content: Array,
    self_witness: Tuple[float, float],
    other_model: AgentModel,
    basis: Array,
    xp: ArrayModule = np,
) -> Tuple[Array, int, float]:
    """
    Model what another agent would perceive about content.
    
    THEORY:
        Theory of Mind is perspective transformation:
        1. Extract "objective" content (remove self-perspective)
        2. Bind to other's perspective
        3. Evolve in other's basin structure (their beliefs)
        4. The result is "what they would perceive"
        
    MATHEMATICAL FORMULATION:
        ToM(content, self, other) = GraceFlow_other(Bind_other(Unbind_self(content)))
        
    BRAIN ANALOG:
        This is mentalizing - simulating another's mental state.
        Involves TPJ (perspective transformation) and mPFC (simulation).
        
    Args:
        content: [4, 4] content matrix (in self's perspective)
        self_witness: (scalar, pseudo) self's witness
        other_model: AgentModel representing the other
        basis: [16, 4, 4] Clifford basis
        xp: Array module
        
    Returns:
        Tuple of:
        - perception: [4, 4] what other would perceive
        - predicted_target: int token other would predict (-1 if unknown)
        - confidence: float confidence in prediction
    """
    # Step 1: Construct witness matrices
    W_self = self_witness[0] * basis[0] + self_witness[1] * basis[15]
    W_other = other_model.witness_matrix(basis, xp)
    
    # Step 2: Transform perspective (self → other)
    # This is the core ToM operation
    other_perception = transform_perspective(
        content, W_self, W_other, basis, xp
    )
    
    # Step 3: Evolve in other's basin structure
    # If other has semantic memory, use it for basin discovery
    predicted_target = -1
    confidence = other_model.confidence
    
    if other_model.semantic_memory is not None:
        # Use Grace basin discovery in other's memory
        matches = other_model.semantic_memory.retrieve(other_perception, top_k=1)
        if matches:
            proto, similarity = matches[0]
            # Get target from prototype's most common target
            if proto.target_distribution:
                predicted_target = max(
                    proto.target_distribution.items(),
                    key=lambda x: x[1]
                )[0]
                confidence *= similarity
    
    # Step 4: Apply Grace to stabilize (ensure equilibrium)
    other_perception = grace_operator(other_perception, basis, xp)
    
    return other_perception, predicted_target, confidence


def predict_other_belief(
    context: List[int],
    other_model: AgentModel,
    model: 'TheoryTrueModel',
    basis: Array,
    xp: ArrayModule = np,
) -> Tuple[int, float]:
    """
    Predict what target the other agent would predict for this context.
    
    THEORY:
        Given a context, what would the other agent predict?
        This requires:
        1. Computing the context matrix
        2. Transforming to other's perspective
        3. Retrieving from other's belief structure
        
    USE CASE:
        - Sally-Anne test: "Where does Sally think the ball is?"
        - Predicting another's answers/choices
        
    Args:
        context: List of token IDs
        other_model: AgentModel of the other
        model: TheoryTrueModel for computing context matrices
        basis: Clifford basis
        xp: Array module
        
    Returns:
        (predicted_target, confidence)
    """
    # Compute context matrix using model's embeddings
    context_matrix = model.compute_context(context)
    
    # Get self witness
    self_witness = extract_witness(context_matrix, basis, xp)
    
    # Use ToM
    _, predicted_target, confidence = theory_of_mind(
        context_matrix, self_witness, other_model, basis, xp
    )
    
    # If ToM didn't give a target, try direct retrieval through other's perspective
    if predicted_target < 0:
        # Transform context to other's perspective
        W_self = self_witness[0] * basis[0] + self_witness[1] * basis[15]
        W_other = other_model.witness_matrix(basis, xp)
        
        other_context = transform_perspective(
            context_matrix, W_self, W_other, basis, xp
        )
        
        # Try to decode using model's vocabulary
        # This is approximate - ideally we'd have other's decoder
        predicted_target = model.decode_attractor(other_context)
        confidence = other_model.confidence * PHI_INV_SQ  # φ²-penalized confidence for fallback
    
    return predicted_target, max(0.0, min(1.0, confidence))


def explain_other_action(
    action: Array,
    context: Array,
    other_model: AgentModel,
    basis: Array,
    xp: ArrayModule = np,
) -> Dict[str, Any]:
    """
    Explain why another agent took a particular action.
    
    THEORY:
        Given an action and context, infer WHY the other did it:
        1. What did they perceive? (their perspective on context)
        2. What were they trying to achieve? (their goal/target)
        3. How certain are we? (based on model confidence)
        
    USE CASE:
        Understanding others' motivations, attributing intentions.
        
    Args:
        action: [4, 4] the action they took (as matrix)
        context: [4, 4] the context they were in
        other_model: AgentModel of the other
        basis: Clifford basis
        xp: Array module
        
    Returns:
        Dict with explanation components
    """
    # What they perceived
    self_witness = extract_witness(context, basis, xp)
    their_perception, _, _ = theory_of_mind(
        context, self_witness, other_model, basis, xp
    )
    
    # How similar is action to their perception?
    # High similarity = action follows from perception
    from holographic_v4.algebra import frobenius_similarity
    action_perception_sim = frobenius_similarity(action, their_perception, xp)
    
    # Grace stability of their action (were they "decided"?)
    action_stability = grace_stability(action, basis, xp)
    
    # Vorticity match (does action match their cognitive style?)
    vort_match = 0.0
    if other_model.vorticity_pattern is not None:
        action_vort = vorticity_signature(action, basis, xp)
        vort_match = vorticity_similarity(
            action_vort, other_model.vorticity_pattern, xp
        )
    
    return {
        'their_perception': their_perception,
        'action_perception_similarity': float(action_perception_sim),
        'action_stability': float(action_stability),
        'vorticity_match': float(vort_match),
        'confidence': other_model.confidence,
        'explanation': (
            f"Action similarity to perception: {action_perception_sim:.2f}, "
            f"stability: {action_stability:.2f}, "
            f"style match: {vort_match:.2f}"
        ),
    }


# =============================================================================
# RECURSIVE TOM — "I know that you know that I know"
# =============================================================================

def recursive_tom(
    content: Array,
    self_witness: Tuple[float, float],
    agent_chain: List[AgentModel],
    basis: Array,
    xp: ArrayModule = np,
) -> Tuple[Array, float]:
    """
    Recursive Theory of Mind - nested perspective taking.
    
    THEORY:
        First-order ToM: What does A think?
        Second-order ToM: What does A think B thinks?
        Third-order ToM: What does A think B thinks C thinks?
        
        Each level is a perspective transformation through
        the chain of agents.
        
    COGNITIVE LIMITS:
        Humans typically max out at 4-5 levels of recursion.
        Confidence decreases with each level.
        
    Args:
        content: [4, 4] initial content
        self_witness: Starting perspective
        agent_chain: List of AgentModels to transform through
        basis: Clifford basis
        xp: Array module
        
    Returns:
        (final_perception, confidence)
    """
    current = content.copy()
    current_witness = self_witness
    confidence = 1.0
    
    for agent in agent_chain:
        # Transform through this agent's perspective
        W_current = current_witness[0] * basis[0] + current_witness[1] * basis[15]
        W_agent = agent.witness_matrix(basis, xp)
        
        current = transform_perspective(current, W_current, W_agent, basis, xp)
        current_witness = agent.witness
        
        # Confidence decreases with each level
        confidence *= agent.confidence * PHI_INV  # Decay by φ⁻¹ per level
    
    # Final Grace stabilization
    current = grace_operator(current, basis, xp)
    
    return current, confidence


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# witness_distance is imported from distributed_prior, not duplicated here


def merge_agent_models(
    models: List[AgentModel],
    basis: Array,
    xp: ArrayModule = np,
) -> AgentModel:
    """
    Merge multiple agent models into one (e.g., for group modeling).
    
    Weighted average based on confidence and observation count.
    """
    if not models:
        return AgentModel(witness=(0.0, 0.0), confidence=0.0, observation_count=0)
    
    if len(models) == 1:
        return models[0]
    
    # Compute weights
    weights = [m.confidence * m.observation_count for m in models]
    total_weight = sum(weights)
    if total_weight < 1e-12:
        weights = [1.0] * len(models)
        total_weight = len(models)
    weights = [w / total_weight for w in weights]
    
    # Weighted average witness
    avg_scalar = sum(w * m.witness[0] for w, m in zip(weights, models))
    avg_pseudo = sum(w * m.witness[1] for w, m in zip(weights, models))
    
    # Combined confidence and observation count
    total_obs = sum(m.observation_count for m in models)
    avg_conf = sum(w * m.confidence for w, m in zip(weights, models))
    
    return AgentModel(
        witness=(avg_scalar, avg_pseudo),
        semantic_memory=None,  # Can't easily merge semantic memories
        vorticity_pattern=None,
        confidence=avg_conf,
        observation_count=total_obs,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Witness inference
    'infer_witness_from_observations',
    'infer_witness_from_observations_weighted',
    
    # Perspective binding
    'bind_to_witness',
    'unbind_from_witness',
    'transform_perspective',
    
    # Agent model
    'AgentModel',
    'AgentModelBuilder',
    
    # Core ToM operations
    'theory_of_mind',
    'predict_other_belief',
    'explain_other_action',
    
    # Recursive ToM
    'recursive_tom',
    
    # Utilities
    # witness_distance is imported from distributed_prior
    'merge_agent_models',
]
