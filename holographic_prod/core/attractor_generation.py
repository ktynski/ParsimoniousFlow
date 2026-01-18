"""
Attractor-Based Generation — Theory-True Continuous Flow
=========================================================

THEORY (Brain Analog):
    Humans don't generate token-by-token with independent lookups.
    The brain maintains a STATE that FLOWS through attractor basins.
    Each state naturally leads to the next - continuity, not jumps.

CURRENT (WRONG - Transformer-style):
    for step in range(max_tokens):
        pred = retrieve(context)  # Independent lookup
        tokens.append(pred)       # Discrete jump

THEORY-TRUE (Attractor flow):
    state = embed(context)
    for step in range(max_tokens):
        state = evolve_state(state, memory)  # Continuous flow
        token = decode(state)                 # Read from trajectory

KEY DIFFERENCES:
    1. STATE CONTINUITY: State evolves, not discrete lookups
    2. GRACE DYNAMICS: Grace operator guides flow through attractors
    3. MEMORY INTERACTION: State @ memory gives next direction
    4. TRAJECTORY COHERENCE: Output follows geometric path

MATHEMATICAL FORMULATION:
    Let M(t) be the state at time t.
    Memory is a superposition of learned bindings.
    
    Evolution:
        retrieved = M(t)^T @ memory  # Unbind from current state
        M(t+1) = Grace(retrieved)    # Flow to attractor
        
    This ensures:
        - Each step uses the FULL history (via state)
        - Grace contracts to coherent subspace
        - Output is trajectory through learned attractors
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from .constants import (
    PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_EPSILON,
    MATRIX_DIM, CLIFFORD_DIM, DTYPE
)
from .algebra import (
    grace_operator, 
    build_clifford_basis,
    decompose_to_coefficients,
    geometric_product_batch,
)
from .quotient import (
    vorticity_weighted_scores, grace_stability,
    extract_chirality, extract_chirality_strength, chirality_match_scores,
)
from .commitment_gate import CommitmentGate, GateDecision

Array = np.ndarray
ArrayModule = type(np)


def generate_attractor_flow(
    memory: 'HolographicMemory',
    prompt_tokens: List[int],
    max_tokens: int = 10,
    grace_steps: int = 3,
    stability_threshold: float = None,
    xp: ArrayModule = np,
    # v5.17.0: Anti-mode-collapse parameters (theory-true)
    inhibition_window: int = 3,      # IoR: penalize last N tokens
    inhibition_factor: float = None,  # IoR: penalty factor (default: φ⁻²)
    use_phi_kernel: bool = True,      # Use φ-kernel probabilistic sampling
    use_polarized_lensing: bool = True,  # Use 16-lens chord for scoring
    # v5.27.0: Chirality-guided generation (quantum-inspired)
    use_chirality_guidance: bool = True,  # Constrain output by context handedness
) -> Tuple[List[int], List[float]]:
    """
    Generate tokens via attractor flow (theory-true, v5.17.0).
    
    THEORY:
        State flows through learned attractor landscape.
        Each step: state → unbind → Grace → decode → continue
        
    BRAIN ANALOG:
        - State = working memory representation
        - Memory = long-term learned associations
        - Grace = attractor dynamics (hippocampus → cortex)
        - Flow = coherent thought trajectory
        
    ANTI-MODE-COLLAPSE (v5.17.0):
        - Inhibition of Return (IoR): Prevents perseveration (repeating tokens)
        - φ-kernel sampling: Probabilistic selection with temperature = 1/φ
        - Polarized lensing: 16-view chord scoring for disambiguation
        
    CHIRALITY GUIDANCE (v5.27.0 — Quantum-inspired):
        - Context chirality (pseudoscalar sign) constrains output handedness
        - "Right-handed" (+) contexts bias toward declarative outputs
        - "Left-handed" (-) contexts bias toward exploratory outputs
        - Strength determines constraint intensity (weak chirality = no constraint)
        
    Args:
        memory: HolographicMemory with learned patterns
        prompt_tokens: Initial context tokens
        max_tokens: Number of tokens to generate
        grace_steps: Grace iterations per step (stability)
        stability_threshold: Stop if stability drops (theory: φ⁻²)
        xp: Array module (numpy or cupy)
        inhibition_window: How many recent tokens to inhibit (default: 3)
        inhibition_factor: Inhibition strength (default: φ⁻² ≈ 0.382)
        use_phi_kernel: Use φ-kernel probabilistic sampling (default: True)
        use_polarized_lensing: Use 16-lens chord scoring (default: True)
        use_chirality_guidance: Constrain output by context chirality (default: True)
        
    Returns:
        (generated_tokens, stability_trace) for analysis
    """
    from .lensing import PolarizedLensSet
    
    if stability_threshold is None:
        stability_threshold = PHI_INV_SQ * 0.5  # Half spectral gap
    
    if inhibition_factor is None:
        inhibition_factor = PHI_INV_SQ  # φ⁻² ≈ 0.382 (theory-derived)
    
    basis = memory.tower.basis
    embeddings = memory.tower.embeddings
    vocab_size = memory.vocab_size
    
    # Initialize polarized lens set for scoring (v5.17.0)
    lens_set = None
    if use_polarized_lensing:
        lens_set = PolarizedLensSet(n_lenses=16, seed=42, xp=xp)
    
    # Initialize state from prompt embedding
    state = memory.tower._embed_sequence(prompt_tokens)
    
    generated = list(prompt_tokens)
    stabilities = []
    recent_tokens = []  # For Inhibition of Return (v5.17.0)
    
    for step in range(max_tokens):
        # =====================================================================
        # STEP 1: Unbind from hierarchical memory
        # =====================================================================
        grand_memory = _get_aggregated_memory(memory.tower, xp)
        state_inv = xp.swapaxes(state, -2, -1) if state.ndim > 2 else state.T
        retrieved = state_inv @ grand_memory
        
        # =====================================================================
        # STEP 2: Apply Grace dynamics (attractor flow)
        # =====================================================================
        for _ in range(grace_steps):
            retrieved = grace_operator(retrieved, basis, xp)
        
        stability = grace_stability(retrieved, basis, xp)
        stabilities.append(float(stability))
        
        if stability < stability_threshold and step > 0:
            break
        
        # =====================================================================
        # STEP 3: FULL VOCABULARY COHERENCE SCORING (Theory-True v5.31.0)
        # =====================================================================
        # "Candidate sets" are FORBIDDEN per THEORY_TRUE_PARADIGM.md
        # Grace ALWAYS converges - score ALL tokens by COHERENCE
        # Coherence = witness_energy / total_energy (Clifford structure)
        # =====================================================================
        
        # Compute compositions: retrieved @ embed[t].T for all t
        # For SO(4): embed† = embed.T (orthogonal inverse)
        embed_T = xp.swapaxes(embeddings, -2, -1)  # [vocab, 4, 4]
        compositions = xp.einsum('ij,vjk->vik', retrieved, embed_T)  # [vocab, 4, 4]
        
        # Coherence scoring via Clifford decomposition
        norm_sq = xp.sum(basis * basis, axis=(1, 2))  # [16]
        coeffs_all = xp.einsum('cij,vij->vc', basis, compositions) / norm_sq  # [vocab, 16]
        
        energies = xp.sum(coeffs_all ** 2, axis=1)  # [vocab]
        witness_energies = coeffs_all[:, 0]**2 + coeffs_all[:, 15]**2  # [vocab]
        coherences = witness_energies / xp.maximum(energies, PHI_EPSILON)
        
        # Convert to numpy for processing
        if hasattr(coherences, 'get'):
            scores_np = np.array(coherences.get(), dtype=np.float64)
        else:
            scores_np = np.array(coherences, dtype=np.float64)
        
        # =====================================================================
        # STEP 3.5: Apply Chirality Guidance (v5.27.0 — Quantum-inspired)
        # =====================================================================
        # THEORY: If the brain is quantum, chirality would propagate top-down.
        # High-level schemas carry handedness that constrains lower-level output.
        if use_chirality_guidance:
            # Extract context chirality and strength from current state
            ctx_chirality = extract_chirality(state, basis, xp)
            ctx_strength = extract_chirality_strength(state, basis, xp)
            
            # Get chirality multipliers for all tokens (full vocab)
            chirality_mults = chirality_match_scores(
                ctx_chirality, ctx_strength, embeddings, basis, xp
            )
            if hasattr(chirality_mults, 'get'):
                chirality_mults = chirality_mults.get()
            
            # Apply chirality constraint
            scores_np = scores_np * chirality_mults
        
        # =====================================================================
        # STEP 4: Apply Inhibition of Return (v5.17.0)
        # =====================================================================
        # BRAIN ANALOG: IoR prevents perseveration (getting stuck)
        # Penalize recently used tokens to encourage diversity
        for recent_idx in recent_tokens[-inhibition_window:]:
            if 0 <= recent_idx < len(scores_np):
                scores_np[recent_idx] *= inhibition_factor
        
        # =====================================================================
        # STEP 5: Select token via φ-kernel sampling (v5.17.0)
        # =====================================================================
        if use_phi_kernel:
            # THEORY: P(token) ∝ score^(1/φ) — φ-derived temperature
            # This is theory-true: φ emerges from self-consistency Λ² = Λ + 1
            scores_positive = np.maximum(scores_np, float(PHI_EPSILON))
            logits = np.log(scores_positive) / PHI_INV  # Temperature = 1/φ ≈ 0.618
            logits = logits - np.max(logits)  # Numerical stability
            probs = np.exp(logits)
            probs = probs / np.sum(probs)
            
            # Sample from distribution (not argmax!)
            token = int(np.random.choice(vocab_size, p=probs))
        else:
            # Deterministic mode: select highest-scoring token
            # Use when reproducibility is needed (e.g., evaluation)
            token = int(np.argmax(scores_np))
        
        generated.append(token)
        recent_tokens.append(token)  # Track for IoR
        
        # =====================================================================
        # STEP 6: Update state for next iteration
        # =====================================================================
        token_emb = embeddings[token]
        state = retrieved @ token_emb
        state = state / (xp.linalg.norm(state) + PHI_EPSILON) * 2.0
    
    return generated, stabilities


def generate_batch_attractor_flow(
    memory: 'HolographicMemory',
    prompts: List[List[int]],
    max_tokens: int = 10,
    grace_steps: int = 3,
    xp: ArrayModule = np,
    # v5.31.1: Anti-mode-collapse parameters (was missing!)
    inhibition_window: int = 3,      # IoR: penalize last N tokens
    inhibition_factor: float = None,  # IoR: penalty (default: φ⁻²)
    use_phi_kernel: bool = True,      # Probabilistic selection with T=1/φ
) -> List[Tuple[List[int], List[float]]]:
    """
    THEORY-TRUE VECTORIZED batch generation via attractor flow (v5.31.0).
    
    THEORY-TRUE PARADIGM:
        1. Grace ALWAYS converges — generation works even with empty memory
        2. Score FULL vocabulary — no "candidate sets" (FORBIDDEN)
        3. Use COHERENCE metric — not similarity
        4. Output EMERGES from attractor dynamics
        
    GPU OPTIMIZATION:
        - All prompts processed in parallel
        - Single memory access per step
        - Batched Grace operations
        - Full vocabulary coherence scoring
        
    Args:
        memory: HolographicMemory with learned patterns
        prompts: List of prompt token sequences
        max_tokens: Tokens to generate per prompt
        grace_steps: Grace iterations per step
        xp: Array module
        
    Returns:
        List of (generated_tokens, stability_trace) per prompt
    """
    batch_size = len(prompts)
    basis = memory.tower.basis
    embeddings = memory.tower.embeddings
    vocab_size = len(embeddings)
    
    # v5.31.1: Anti-mode-collapse defaults
    if inhibition_factor is None:
        inhibition_factor = PHI_INV_SQ  # φ⁻² ≈ 0.382
    
    # Initialize states from prompts (batched)
    states = xp.stack([
        memory.tower._embed_sequence(prompt) 
        for prompt in prompts
    ])  # [batch, 4, 4]
    
    # Track generated tokens and stabilities
    all_generated = [list(p) for p in prompts]
    all_stabilities = [[] for _ in range(batch_size)]
    active_mask = xp.ones(batch_size, dtype=bool)
    recent_tokens = [[] for _ in range(batch_size)]  # v5.31.1: IoR tracking
    
    # Get aggregated memory once (for unbinding)
    grand_memory = _get_aggregated_memory(memory.tower, xp)
    
    # =========================================================================
    # THEORY-TRUE (v5.31.0): Score FULL VOCABULARY, not just learned targets
    # =========================================================================
    # "Candidate sets" are FORBIDDEN per THEORY_TRUE_PARADIGM.md
    # Grace ALWAYS converges - even with zero training, embeddings form attractors
    # Schemas enable compositional generation of novel outputs
    # =========================================================================
    
    # Precompute for coherence scoring
    norm_sq = xp.sum(basis * basis, axis=(1, 2))  # [16]
    embed_T = xp.swapaxes(embeddings, -2, -1)  # [vocab, 4, 4]
    
    for step in range(max_tokens):
        if not xp.any(active_mask):
            break
            
        # =====================================================================
        # BATCHED: Unbind from memory
        # =====================================================================
        # states: [batch, 4, 4], grand_memory: [4, 4]
        states_inv = xp.swapaxes(states, -2, -1)  # [batch, 4, 4]
        retrieved = xp.einsum('bij,jk->bik', states_inv, grand_memory)  # [batch, 4, 4]
        
        # =====================================================================
        # BATCHED: Grace dynamics
        # =====================================================================
        for _ in range(grace_steps):
            retrieved = _grace_operator_batch(retrieved, basis, xp)
        
        # =====================================================================
        # BATCHED: Compute stability
        # =====================================================================
        stabilities = _grace_stability_batch(retrieved, basis, xp)  # [batch]
        
        for i in range(batch_size):
            if active_mask[i]:
                all_stabilities[i].append(float(stabilities[i]))
                if stabilities[i] < PHI_INV_SQ * 0.5 and step > 0:
                    active_mask[i] = False
        
        # =====================================================================
        # BATCHED: FULL VOCABULARY COHERENCE SCORING (Theory-True!)
        # =====================================================================
        # Score ALL tokens by COHERENCE, not similarity.
        # Coherence = witness_energy / total_energy of the composition.
        #
        # For each batch item b and token t:
        #   composition[b,t] = retrieved[b] @ embed[t].T
        #   coherence[b,t] = witness_energy / total_energy
        # =====================================================================
        
        # Compute all compositions: [batch, vocab, 4, 4]
        compositions = xp.einsum('bij,vjk->bvik', retrieved, embed_T)
        
        # Decompose into Clifford coefficients: [batch, vocab, 16]
        coeffs_all = xp.einsum('cij,bvij->bvc', basis, compositions) / norm_sq
        
        # Compute coherence: witness_energy / total_energy
        energies = xp.sum(coeffs_all ** 2, axis=2)  # [batch, vocab]
        witness_energies = coeffs_all[:, :, 0]**2 + coeffs_all[:, :, 15]**2  # [batch, vocab]
        coherences = witness_energies / xp.maximum(energies, PHI_EPSILON)  # [batch, vocab]
        
        # =====================================================================
        # v5.31.1: Apply Inhibition of Return + φ-kernel sampling
        # =====================================================================
        # Move to numpy for per-batch IoR and sampling (small overhead vs mode collapse)
        # Convert to numpy for per-batch IoR and sampling
        if hasattr(coherences, 'get'):
            coherences_np = coherences.get()  # cupy → numpy
        else:
            coherences_np = np.asarray(coherences)  # ensure numpy array
        
        selected_tokens = []
        for i in range(batch_size):
            if not active_mask[i]:
                selected_tokens.append(0)  # Dummy
                continue
                
            scores = coherences_np[i].copy()
            
            # IoR: Penalize recently used tokens
            for recent_idx in recent_tokens[i][-inhibition_window:]:
                if 0 <= recent_idx < len(scores):
                    scores[recent_idx] *= inhibition_factor
            
            # φ-kernel sampling (if enabled)
            if use_phi_kernel:
                # THEORY: P(token) ∝ score^φ — φ-derived power law
                # NOT softmax! φ emerges from self-consistency Λ² = Λ + 1
                scores_positive = np.maximum(scores, float(PHI_EPSILON))
                logits = np.log(scores_positive) / PHI_INV  # = φ * log(scores)
                logits = logits - np.max(logits)  # Numerical stability
                probs = np.exp(logits)  # = scores^φ (power law!)
                probs = probs / np.sum(probs)
                
                # Sample from distribution
                token = int(np.random.choice(len(probs), p=probs))
            else:
                token = int(np.argmax(scores))
            
            selected_tokens.append(token)
        
        # =====================================================================
        # Update states and collect tokens
        # =====================================================================
        for i in range(batch_size):
            if active_mask[i]:
                token = selected_tokens[i]
                all_generated[i].append(token)
                recent_tokens[i].append(token)  # Track for IoR
                
                # Evolve state
                token_emb = embeddings[token]
                new_state = retrieved[i] @ token_emb
                norm = xp.linalg.norm(new_state)
                states[i] = new_state / (norm + PHI_EPSILON) * 2.0
    
    return [(g, s) for g, s in zip(all_generated, all_stabilities)]


# =============================================================================
# HELPER FUNCTIONS (GPU-optimized)
# =============================================================================

def _get_aggregated_memory(tower, xp) -> Array:
    """
    Get φ-weighted aggregation of active satellite memories.
    
    THEORY: Grand memory is superposition of all learned patterns.
    
    Handles both:
        - TowerMemory: uses _satellite_memories
        - MultiLevelTower: uses _all_memories
    """
    # Get the satellite memories array (different attribute names)
    if hasattr(tower, '_satellite_memories'):
        memories = tower._satellite_memories
    elif hasattr(tower, '_all_memories'):
        memories = tower._all_memories
    else:
        return xp.eye(MATRIX_DIM, dtype=DTYPE)
    
    # Get binding counts
    if not hasattr(tower, '_satellite_n_bindings'):
        return xp.eye(MATRIX_DIM, dtype=DTYPE)
    
    active_mask = tower._satellite_n_bindings > 0
    active_indices = xp.where(active_mask)[0]
    
    if len(active_indices) == 0:
        return xp.eye(MATRIX_DIM, dtype=DTYPE)
    
    # Cap at 1000 for performance (φ-weighted, so early ones matter most)
    active_indices = active_indices[:1000]
    
    grand_memory = xp.zeros((MATRIX_DIM, MATRIX_DIM), dtype=DTYPE)
    for i, sat_i in enumerate(active_indices):
        sat_i = int(sat_i)
        weight = PHI_INV ** (i % 16)
        grand_memory += weight * memories[sat_i]
    
    return grand_memory

def _grace_operator_batch(M: Array, basis: Array, xp: ArrayModule) -> Array:
    """
    Batched Grace operator for GPU efficiency.
    
    Args:
        M: [batch, 4, 4] matrices
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        [batch, 4, 4] Grace-contracted matrices
    """
    from .algebra import GRACE_SCALES_FLAT as GRACE_SCALES
    
    batch_size = M.shape[0]
    
    # Decompose batch
    norm_sq = xp.sum(basis * basis, axis=(1, 2))  # [16]
    coeffs = xp.einsum('cij,bij->bc', basis, M) / norm_sq  # [batch, 16]
    
    # Apply Grace scales
    scales = xp.asarray(GRACE_SCALES) if xp != np else GRACE_SCALES
    scaled_coeffs = coeffs * scales  # [batch, 16]
    
    # Reconstruct
    result = xp.einsum('bc,cij->bij', scaled_coeffs, basis)  # [batch, 4, 4]
    
    return result


def _grace_stability_batch(M: Array, basis: Array, xp: ArrayModule) -> Array:
    """
    Batched Grace stability computation.
    
    Args:
        M: [batch, 4, 4] matrices
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        [batch] stability values
    """
    # Decompose
    norm_sq = xp.sum(basis * basis, axis=(1, 2))  # [16]
    coeffs = xp.einsum('cij,bij->bc', basis, M) / norm_sq  # [batch, 16]
    
    # Witness energy (scalar + pseudoscalar)
    witness_energy = coeffs[:, 0] ** 2 + coeffs[:, 15] ** 2  # [batch]
    
    # Total energy
    total_energy = xp.sum(coeffs ** 2, axis=1) + PHI_EPSILON  # [batch]
    
    return witness_energy / total_energy


def _vorticity_scores_batch(
    queries: Array, 
    embeddings: Array, 
    basis: Array, 
    xp: ArrayModule
) -> Array:
    """
    Batched vorticity-weighted scoring.
    
    Args:
        queries: [batch, 4, 4] query matrices
        embeddings: [n_candidates, 4, 4] candidate embeddings
        basis: [16, 4, 4] Clifford basis
        xp: array module
        
    Returns:
        [batch, n_candidates] scores
    """
    batch_size = queries.shape[0]
    n_candidates = embeddings.shape[0]
    
    # Decompose queries and embeddings
    norm_sq = xp.sum(basis * basis, axis=(1, 2))  # [16]
    
    query_coeffs = xp.einsum('cij,bij->bc', basis, queries) / norm_sq  # [batch, 16]
    embed_coeffs = xp.einsum('cij,nij->nc', basis, embeddings) / norm_sq  # [n_candidates, 16]
    
    # Cosine similarity in coefficient space
    query_norms = xp.sqrt(xp.sum(query_coeffs ** 2, axis=1, keepdims=True)) + PHI_EPSILON  # [batch, 1]
    embed_norms = xp.sqrt(xp.sum(embed_coeffs ** 2, axis=1, keepdims=True)) + PHI_EPSILON  # [n_candidates, 1]
    
    query_normed = query_coeffs / query_norms  # [batch, 16]
    embed_normed = embed_coeffs / embed_norms  # [n_candidates, 16]
    
    # Similarity matrix
    scores = xp.einsum('bc,nc->bn', query_normed, embed_normed)  # [batch, n_candidates]
    
    return scores
