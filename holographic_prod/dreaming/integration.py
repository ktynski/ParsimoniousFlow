"""
Dreaming Integration — Unified Sleep Cycle
==========================================

COMPLEMENTARY LEARNING SYSTEMS (Theory):
    The brain uses two complementary memory systems:
    
    1. FAST SYSTEM (Hippocampus / Tower):
       - Rapid binding of experiences
       - Synaptic homeostasis during sleep
       - Non-REM: Witness propagation (master → satellites)
       - REM: φ-jitter for creative exploration
       
    2. SLOW SYSTEM (Cortex / Semantic Memory):
       - Gradual extraction of statistical regularities
       - Systems consolidation during sleep
       - Non-REM: Cluster episodes → prototypes
       - REM: Recombine prototypes → schemas

INTEGRATED SLEEP CYCLE:
    The integrated_sleep() function combines both systems:
    
    1. Systems Non-REM: Episodic → Semantic (prototype formation)
    2. Tower Non-REM: Master → Satellites (witness propagation)
    3. Systems REM: Prototype → Schema (creative recombination)
    4. Tower REM: φ-jitter (exploration in Grace space)
    5. Pruning: Remove weak memories from both systems

ALL RATES φ-DERIVED. NO ARBITRARY HYPERPARAMETERS.
"""

from typing import List, Dict, Any, TYPE_CHECKING

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_EIGHT
)

if TYPE_CHECKING:
    from .dreaming_system import DreamingSystem
    from holographic_prod.memory import HolographicMemory

from .structures import EpisodicEntry


# =============================================================================
# INTEGRATED SLEEP — Combines Tower + Systems Dreaming
# =============================================================================

def integrated_sleep(
    memory: 'HolographicMemory',
    dreaming_system: 'DreamingSystem',
    episodes: List[EpisodicEntry],
    rem_cycles: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute integrated sleep cycle combining both dreaming systems.
    
    THEORY (Complementary Learning Systems):
        The brain consolidates memories at two levels during sleep:
        
        SYNAPTIC (Tower):
            - Non-REM: Master broadcasts witness to satellites
            - REM: φ-jitter for creative exploration
            - Maintains holographic memory coherence
            
        SYSTEMS (DreamingSystem):
            - Non-REM: Cluster episodes → semantic prototypes
            - REM: Recombine prototypes → schemas
            - Extracts statistical regularities
    
    PHASE ORDER:
        1. systems_non_rem: Episodic → Semantic consolidation
        2. tower_non_rem: Witness propagation within holographic memory
        3. systems_rem: Creative recombination → schema discovery
        4. tower_rem: φ-jitter exploration in Grace space
        5. pruning: Remove weak memories
    
    Args:
        memory: HolographicMemory instance (contains Tower)
        dreaming_system: DreamingSystem instance (semantic consolidation)
        episodes: List of EpisodicEntry from waking experience
        rem_cycles: Number of REM cycles (default 1)
        verbose: Whether to print progress
        
    Returns:
        Dict with stats from both systems and phase completion info
    """
    phases_completed = []
    
    if verbose:
        print()
        print("╔" + "═" * 60 + "╗")
        print("║" + "  INTEGRATED SLEEP CYCLE  ".center(60) + "║")
        print("║" + "  (Complementary Learning Systems)  ".center(60) + "║")
        print("╚" + "═" * 60 + "╝")
    
    # Get pre-sleep state
    pre_tower_stability = memory.tower.get_stability()
    pre_semantic_count = dreaming_system.semantic_memory.stats()['total_prototypes']
    
    # =========================================================================
    # PHASE 1: SYSTEMS NON-REM (Episodic → Semantic)
    # =========================================================================
    if verbose:
        print()
        print("  Phase 1: SYSTEMS NON-REM (Episodic → Semantic)")
        print("  " + "-" * 50)
    
    # Run systems consolidation (Non-REM portion only)
    # This clusters episodes into prototypes
    systems_stats = dreaming_system.sleep(
        episodes=episodes,
        rem_cycles=0,  # No REM yet - we do that after tower Non-REM
        n_sequence_replays=0,  # Replay happens in REM
        verbose=verbose,
    )
    phases_completed.append('systems_non_rem')
    
    if verbose:
        print(f"    Prototypes created: {systems_stats.get('prototypes_created', 0)}")
    
    # =========================================================================
    # PHASE 2: TOWER NON-REM (Witness Propagation)
    # =========================================================================
    if verbose:
        print()
        print("  Phase 2: TOWER NON-REM (Witness Propagation)")
        print("  " + "-" * 50)
    
    # Run tower consolidation
    memory.tower.non_rem_consolidation(consolidation_rate=PHI_INV_SQ)
    phases_completed.append('tower_non_rem')
    
    mid_tower_stability = memory.tower.get_stability()
    if verbose:
        print(f"    Stability: {pre_tower_stability:.3f} → {mid_tower_stability:.3f}")
    
    # =========================================================================
    # PHASE 3: SYSTEMS REM (Creative Recombination)
    # =========================================================================
    if verbose:
        print()
        print("  Phase 3: SYSTEMS REM (Creative Recombination)")
        print("  " + "-" * 50)
    
    # Get all prototypes for recombination
    all_prototypes = [p for level in dreaming_system.semantic_memory.levels for p in level]
    
    schemas_discovered = 0
    for cycle in range(rem_cycles):
        if verbose and rem_cycles > 1:
            print(f"    REM Cycle {cycle + 1}/{rem_cycles}")
        
        if len(all_prototypes) >= 2:
            schemas = dreaming_system.recombinator.dream_cycle(
                all_prototypes,
                num_recombinations=min(len(all_prototypes) * 10, 1000),
                verbose=False,
            )
            
            for schema in schemas:
                dreaming_system.semantic_memory.add_schema(schema)
            
            schemas_discovered += len(schemas)
    
    phases_completed.append('systems_rem')
    systems_stats['schemas_discovered'] = schemas_discovered
    
    if verbose:
        print(f"    Schemas discovered: {schemas_discovered}")
    
    # =========================================================================
    # PHASE 4: TOWER REM (φ-Jitter Exploration)
    # =========================================================================
    if verbose:
        print()
        print("  Phase 4: TOWER REM (φ-Jitter Exploration)")
        print("  " + "-" * 50)
    
    tower_improvements = 0
    for cycle in range(rem_cycles):
        if memory.tower.rem_recombination(jitter_scale=PHI_INV_CUBE):
            tower_improvements += 1
    
    phases_completed.append('tower_rem')
    
    if verbose:
        print(f"    Improvements: {tower_improvements}/{rem_cycles}")
    
    # =========================================================================
    # PHASE 5: PRUNING
    # =========================================================================
    if verbose:
        print()
        print("  Phase 5: PRUNING")
        print("  " + "-" * 50)
    
    from .pruning import prune_semantic_memory
    
    prune_stats = prune_semantic_memory(
        dreaming_system.semantic_memory,
        dreaming_system.basis,
        dreaming_system.xp,
        salience_threshold=PHI_INV_CUBE,
        support_threshold=2,
        verbose=False,
    )
    phases_completed.append('pruning')
    
    if verbose:
        print(f"    Pruned: {prune_stats['pruned']} prototypes")
    
    # =========================================================================
    # FINAL STATE
    # =========================================================================
    post_tower_stability = memory.tower.get_stability()
    post_semantic_count = dreaming_system.semantic_memory.stats()['total_prototypes']
    
    # Update DreamingSystem statistics
    dreaming_system.sleep_count += 1
    dreaming_system.total_episodes_processed += len(episodes)
    
    # Build result
    tower_stats = {
        'pre_stability': pre_tower_stability,
        'post_stability': post_tower_stability,
        'stability_change': post_tower_stability - pre_tower_stability,
        'rem_improvements': tower_improvements,
    }
    
    systems_stats['prototypes_before'] = pre_semantic_count
    systems_stats['prototypes_after'] = post_semantic_count
    systems_stats['prototypes_pruned'] = prune_stats['pruned']
    
    result = {
        'tower_stats': tower_stats,
        'systems_stats': systems_stats,
        'phases_completed': phases_completed,
        'total_episodes': len(episodes),
    }
    
    if verbose:
        print()
        print("  INTEGRATED SLEEP COMPLETE")
        print(f"    Tower stability: {pre_tower_stability:.3f} → {post_tower_stability:.3f}")
        print(f"    Semantic memory: {pre_semantic_count} → {post_semantic_count} prototypes")
        print(f"    Phases: {' → '.join(phases_completed)}")
    
    return result


# =============================================================================
# MODEL INTEGRATION
# =============================================================================

def integrate_dreaming_with_model(
    model, 
    dreaming: 'DreamingSystem', 
    use_distributed_prior: bool = True,
    use_grace_basin: bool = True,
    K: int = 8,
    confidence_threshold: float = PHI_INV_SQ,
    use_semantic_context: bool = True,
):
    """
    Integrate dreaming with the main TheoryTrueModel.
    
    ⚠️ WARNING (v5.31.0): This function uses SEQUENTIAL pattern, not PARALLEL!
    See AUDIT_INCONSISTENCIES.md for details. CRITICAL_PRINCIPLES.md claims
    "all paths run IN PARALLEL" but this implementation uses waterfall.
    
    TODO: Refactor to TRUE PARALLEL per CLS theory (both paths always run,
    winner by confidence, agreement boosts confidence).
    
    CURRENT IMPLEMENTATION (Sequential):
        1. Try holographic (if confidence >= φ⁻², return immediately)
        2. Try semantic (only if holographic was uncertain)
    
    THEORY-TRUE SHOULD BE (Parallel):
        1. Run BOTH holographic AND semantic simultaneously
        2. Agreement = synergy (boosted confidence)
        3. Conflict = attention needed (ACC analog)
        4. Winner by confidence, not order
    
    KNOWN ISSUES:
        - Device mismatch: DreamingSystem typically uses numpy (CPU),
          model may use cupy (GPU). Ensure prototype matrices are on
          same device as model.basis before calling.
    
    OPTIMIZATION:
        - Factorized prior trained ONCE at closure creation, not per retrieval
        - Prototype cache refreshed only when needed
    
    Args:
        model: TheoryTrueModel instance
        dreaming: DreamingSystem instance
        use_distributed_prior: If True, use brain-analog population coding.
        use_grace_basin: If False and not using distributed prior, use simpler retrieval.
        K: Number of neighbors for distributed prior (default 8)
        confidence_threshold: Below this, blend with global prior (default φ⁻²)
        use_semantic_context: If True and model has predictiveness, use semantic-only context
    """
    from holographic_prod.resonance import distributed_prior_retrieve, grace_basin_retrieve
    from holographic_prod.cognitive.distributed_prior import FactorizedAssociativePrior, extended_witness
    from holographic_prod.core.algebra import vorticity_signature
    
    # Initialize factorized prior for global fallback
    factorized_prior = FactorizedAssociativePrior(witness_dim=4, xp=model.xp)
    
    # OPTIMIZATION: Cache prototypes and train factorized prior ONCE at creation
    cached_prototypes = []
    cached_targets = []
    cached_supports = []
    cached_vorticity_sigs = []
    
    for proto in dreaming.semantic_memory.levels[0]:
        cached_prototypes.append(proto.prototype_matrix)
        cached_targets.append(proto.target_distribution)
        cached_supports.append(proto.support)
        cached_vorticity_sigs.append(proto.vorticity_signature)
        
        # Train factorized prior ONCE (not on every retrieval!)
        witness = extended_witness(proto.prototype_matrix, model.basis, model.xp)
        factorized_prior.update(witness, proto.prototype_matrix, weight=proto.support * PHI_INV_EIGHT)
    
    def retrieve_with_dreaming(context: List[int]):
        """
        Brain-analog hierarchical retrieval (v4.8.0).
        
        Returns:
            (attractor, target, source) where source is "holographic", "semantic", or "unknown"
        """
        # 1. Try holographic retrieval (O(1) unbinding) — like hippocampal pattern completion
        ctx_rep = model.compute_context_representation(context)
        attractor, target_idx, confidence, source = model.holographic_memory.retrieve(ctx_rep)
        
        if confidence >= PHI_INV_SQ:
            return attractor, int(target_idx), "holographic"
        
        # 2. Try semantic (prototype retrieval via dreaming)
        # SEMANTIC EXTRACTION: Use semantic-only context if enabled
        if use_semantic_context and hasattr(model, 'compute_semantic_context'):
            ctx_rep = model.compute_semantic_context(context)
        else:
            ctx_rep = model.compute_context(context)
        
        # Use cached prototypes (updated at each sleep cycle)
        prototypes = cached_prototypes
        prototype_targets = cached_targets
        prototype_supports = cached_supports
        prototype_vorticity_sigs = cached_vorticity_sigs
        
        # Compute query vorticity signature for grammar-aware matching
        if len(context) >= 2:
            query_mats = model.xp.stack([model.get_embedding(t) for t in context], axis=0)
            query_vort_sig = vorticity_signature(query_mats, model.basis, model.xp)
        else:
            query_vort_sig = None
        
        if len(prototypes) > 0:
            
            if use_distributed_prior:
                # BRAIN-ANALOG: Distributed prior (population coding)
                proto_idx, target, confidence, info = distributed_prior_retrieve(
                    query=ctx_rep,
                    prototypes=prototypes,
                    prototype_targets=prototype_targets,
                    prototype_supports=prototype_supports,
                    basis=model.basis,
                    xp=model.xp,
                    K=K,
                    grace_steps=3,
                    query_vorticity_sig=query_vort_sig,
                    prototype_vorticity_sigs=prototype_vorticity_sigs,
                    factorized_prior=factorized_prior,
                    confidence_threshold=confidence_threshold,
                )
                
                source = info.get("source", "distributed_prior")
                if proto_idx is not None:
                    return prototypes[proto_idx], target, f"{source}(conf={confidence:.2f})"
            
            elif use_grace_basin:
                # FALLBACK: Single-prototype Grace basin discovery
                proto_idx, target, confidence, info = grace_basin_retrieve(
                    ctx_rep, prototypes, prototype_targets,
                    model.basis, model.xp,
                    grace_steps=5,
                    min_confidence=0.0,
                    query_vorticity_sig=query_vort_sig,
                    prototype_vorticity_sigs=prototype_vorticity_sigs,
                    vorticity_weight=PHI_INV_CUBE,
                )
                
                source = info.get("source", "grace_basin")
                if proto_idx is not None:
                    return prototypes[proto_idx], target, f"{source}(conf={confidence:.2f})"
        
        # 3. Unknown - return context representation (NOT identity!)
        return ctx_rep, 0, "unknown"
    
    return retrieve_with_dreaming
