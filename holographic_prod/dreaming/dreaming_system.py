"""
DreamingSystem — Integrated Sleep Cycle with All 12 Brain-Inspired Parsimonies

BRAIN-INSPIRED FEATURES (12 total):
    
    MEMORY ENCODING:
    1. Salience: Prioritize emotionally important episodes (scalar + pseudoscalar)
    2. Novelty: Prioritize novel episodes (distance from existing prototypes)
    3. Prediction Error: Prioritize surprising episodes (Grace residual)
    4. Predictive Coding: Only encode what memory doesn't predict
    
    MEMORY MAINTENANCE:
    5. Pruning: Remove low-salience + low-support memories
    6. Interference Management: Merge similar prototypes
    7. Reconsolidation: Retrieval updates memory
    8. Pseudo-Rehearsal: Generate samples to prevent forgetting
    
    MEMORY RETRIEVAL:
    9. Working Memory Gating: Salience-based attention
    10. Pattern Completion: Grace flow denoises queries
    11. Inhibition of Return: Suppress recently retrieved
    
    SEQUENCE MEMORY:
    12. Sequence Replay: Store/replay transitions via vorticity

Usage:
    dreaming = DreamingSystem(model.basis)
    
    # Periodically consolidate episodic memories
    episodes = [EpisodicEntry(ctx, target) for ctx, target in waking_data]
    dreaming.sleep(episodes)
    
    # Query semantic memory with pattern completion
    match = dreaming.retrieve(query, use_pattern_completion=True)
    if match:
        target = match.sample_target()
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_CUBE, PHI_INV_SIX
)

from .structures import EpisodicEntry, SemanticPrototype
from .consolidation import NonREMConsolidator
from .recombination import REMRecombinator
from .semantic_memory import SemanticMemory
from .sequence_replay import TransitionBuffer, replay_transitions_during_rem
from .inhibition import RetrievalHistory, retrieve_with_inhibition
from .working_memory import WorkingMemory
from .pattern_completion import pattern_complete
from .memory_management import phi_decay_forget, compute_adaptive_threshold
from .pruning import prune_semantic_memory
from .interference import manage_interference
from .pseudo_rehearsal import generate_pseudo_episodes_batch


# Import from predictive_coding module (no duplication)
from .predictive_coding import predictive_encode_batch


# =============================================================================
# DREAMING SYSTEM
# =============================================================================

class DreamingSystem:
    """
    Complete dreaming system integrating Non-REM and REM phases.
    """
    
    def __init__(
        self,
        basis: np.ndarray,
        xp = np,
        similarity_threshold: float = PHI_INV,
        min_cluster_size: int = 3,
        survival_threshold: float = PHI_INV,
        recurrence_threshold: int = 3,
        use_salience: bool = True,
        use_novelty: bool = True,
        use_predictive_coding: bool = True,
        use_pattern_completion: bool = True,
        use_inhibition_of_return: bool = True,
        use_sequence_replay: bool = True,
        use_pseudo_rehearsal: bool = True,
        transition_buffer_capacity: int = 1000,
    ):
        self.basis = basis
        self.xp = xp
        
        # Feature flags
        self.use_salience = use_salience
        self.use_novelty = use_novelty
        self.use_predictive_coding = use_predictive_coding
        self.use_pattern_completion = use_pattern_completion
        self.use_inhibition_of_return = use_inhibition_of_return
        self.use_sequence_replay = use_sequence_replay
        self.use_pseudo_rehearsal = use_pseudo_rehearsal
        
        # Initialize semantic memory first
        self.semantic_memory = SemanticMemory(basis=basis, xp=xp)
        
        # Initialize consolidator
        self.consolidator = NonREMConsolidator(
            basis=basis,
            xp=xp,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            use_salience=use_salience,
            use_novelty=use_novelty,
            semantic_memory=self.semantic_memory,
        )
        
        self.recombinator = REMRecombinator(
            basis=basis,
            xp=xp,
            survival_threshold=survival_threshold,
            recurrence_threshold=recurrence_threshold,
        )
        
        # Transition buffer for sequence replay
        self.transition_buffer = TransitionBuffer(
            capacity=transition_buffer_capacity,
            xp=xp,
        )
        
        # Retrieval history for inhibition of return
        self.retrieval_history = RetrievalHistory(
            decay_rate=PHI_INV,
            xp=xp,
        )
        
        # Working memory (fast cache for recent items)
        self.working_memory = WorkingMemory(capacity=7, decay_rate=PHI_INV)
        
        # Adaptive threshold parameters
        self.base_similarity_threshold = similarity_threshold
        self.use_adaptive_threshold = True
        self.total_episodes_seen = 0
        
        # Statistics
        self.sleep_count = 0
        self.total_episodes_processed = 0
    
    def sleep(
        self,
        episodes: List[EpisodicEntry],
        rem_cycles: int = 1,
        n_sequence_replays: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a complete sleep cycle with all brain-inspired features.
        
        PHASES:
            0. Pre-processing: Predictive coding filter + sequence recording
            1. Non-REM: Consolidation into prototypes
            2. REM: Recombination + schema discovery
            3. Sequence Replay: Replay temporal transitions
            4. Pruning: Remove weak memories
            5. Interference Management: Merge similar prototypes
        """
        xp = self.xp
        
        if verbose:
            print()
            print("╔" + "═" * 50 + "╗")
            print("║" + "  SLEEP CYCLE  ".center(50) + "║")
            print("╚" + "═" * 50 + "╝")
        
        stats = {
            "input_episodes": len(episodes),
            "prototypes_created": 0,
            "schemas_discovered": 0,
            "predictive_filtered": 0,
            "transitions_recorded": 0,
            "sequences_replayed": 0,
            "episodes_forgotten": 0,
            "similarity_threshold": self.consolidator.similarity_threshold,
        }
        
        self.total_episodes_seen += len(episodes)
        
        # Phase -1: φ-Decay Forgetting
        num_protos = self.semantic_memory.stats()['total_prototypes'] if self.semantic_memory else 0
        MAX_EPISODES_PER_SLEEP = max(5000, int(PHI * PHI * (num_protos + 100) * 10))
        if len(episodes) > MAX_EPISODES_PER_SLEEP:
            if verbose:
                print()
                print("  Phase -1: φ-DECAY FORGETTING")
                print("  " + "-" * 40)
            
            episodes, num_forgotten = phi_decay_forget(
                episodes, MAX_EPISODES_PER_SLEEP, self.basis, xp
            )
            stats["episodes_forgotten"] = num_forgotten
            
            if verbose:
                print(f"    φ-decay: {num_forgotten} low-priority episodes forgotten")
                print(f"    Surviving: {len(episodes)} (capacity={MAX_EPISODES_PER_SLEEP})")
        
        # Adaptive similarity threshold
        if self.use_adaptive_threshold:
            old_threshold = self.consolidator.similarity_threshold
            new_threshold = compute_adaptive_threshold(
                self.semantic_memory,
                min_threshold=PHI_INV_CUBE,
                max_threshold=1 - PHI_INV_CUBE,
            )
            self.consolidator.similarity_threshold = new_threshold
            stats["similarity_threshold"] = new_threshold
            
            if verbose and abs(new_threshold - old_threshold) > PHI_INV_SIX:
                print(f"    Adaptive threshold: {old_threshold:.2f} → {new_threshold:.2f}")
        
        # Phase 0: Pre-processing
        if verbose:
            print()
            print("  Phase 0: PRE-PROCESSING")
            print("  " + "-" * 40)
        
        # Record temporal transitions
        if self.use_sequence_replay and len(episodes) >= 2:
            self.transition_buffer.add_from_episode_sequence(episodes, self.basis, xp)
            stats["transitions_recorded"] = len(episodes) - 1
            if verbose:
                print(f"    Recorded {len(episodes) - 1} transitions")
        
        # Apply predictive coding filter
        episodes_to_consolidate = episodes
        if self.use_predictive_coding and self.semantic_memory.stats()['total_prototypes'] > 0:
            significant, redundant, pc_stats = predictive_encode_batch(
                episodes, self.semantic_memory, self.basis, xp,
                significance_threshold=PHI_INV_CUBE,
            )
            episodes_to_consolidate = significant
            stats["predictive_filtered"] = pc_stats['redundant']
            if verbose:
                print(f"    Predictive coding: {pc_stats['redundant']} redundant filtered")
                print(f"    Episodes to consolidate: {len(episodes_to_consolidate)}")
        
        # Phase 1: Non-REM
        if verbose:
            print()
            print("  Phase 1: NON-REM (Consolidation)")
            print("  " + "-" * 40)
        
        if len(episodes_to_consolidate) > 0:
            prototypes = self.consolidator.consolidate(episodes_to_consolidate, verbose=verbose)
            
            for proto in prototypes:
                self.semantic_memory.add_prototype(proto, level=0)
            
            stats["prototypes_created"] = len(prototypes)
        else:
            if verbose:
                print("    No new episodes to consolidate")
        
        # Phase 2: REM
        if verbose:
            print()
            print("  Phase 2: REM (Recombination)")
            print("  " + "-" * 40)
        
        for cycle in range(rem_cycles):
            if verbose and rem_cycles > 1:
                print(f"    Cycle {cycle + 1}/{rem_cycles}")
            
            all_prototypes = [p for level in self.semantic_memory.levels for p in level]
            max_recombinations = min(len(all_prototypes) * 10, 1000)
            
            if len(all_prototypes) >= 2:
                schemas = self.recombinator.dream_cycle(
                    all_prototypes,
                    num_recombinations=max_recombinations,
                    verbose=verbose,
                )
                
                for schema in schemas:
                    self.semantic_memory.add_schema(schema)
                
                stats["schemas_discovered"] += len(schemas)
            else:
                if verbose:
                    print("    Not enough prototypes for recombination")
        
        # Phase 3: Sequence replay
        if self.use_sequence_replay and self.transition_buffer.stats()['count'] > 0:
            if verbose:
                print()
                print("  Phase 3: SEQUENCE REPLAY")
                print("  " + "-" * 40)
            
            sequences, replay_stats = replay_transitions_during_rem(
                self.transition_buffer, self.basis, xp,
                n_replays=n_sequence_replays,
                sequence_length=3,
            )
            stats["sequences_replayed"] = replay_stats['replayed']
            
            if verbose:
                print(f"    Replayed {replay_stats['replayed']} sequences")
                print(f"    Avg length: {replay_stats['avg_length']:.1f}")
        
        # Phase 4: Pruning
        if verbose:
            print()
            print("  Phase 4: SYNAPTIC PRUNING")
            print("  " + "-" * 40)
        
        prune_stats = prune_semantic_memory(
            self.semantic_memory, 
            self.basis, 
            xp,
            salience_threshold=PHI_INV_CUBE,
            support_threshold=2,
            verbose=verbose,
        )
        stats["prototypes_pruned"] = prune_stats['pruned']
        
        if verbose:
            print(f"    Pruned {prune_stats['pruned']} weak prototypes")
            print(f"    Prototypes: {prune_stats['total_before']} → {prune_stats['total_after']}")
        
        # Phase 5: Interference Management
        if verbose:
            print()
            print("  Phase 5: INTERFERENCE MANAGEMENT")
            print("  " + "-" * 40)
        
        interference_stats = manage_interference(
            self.semantic_memory,
            self.basis,
            xp,
            similarity_threshold=PHI_INV,
            max_merges_per_cycle=5,
            verbose=verbose,
        )
        stats["prototypes_merged"] = interference_stats['merges']
        
        if verbose:
            print(f"    Merged {interference_stats['merges']} similar prototypes")
        
        # Update stats
        self.sleep_count += 1
        self.total_episodes_processed += len(episodes)
        self.retrieval_history.advance_time(1.0)
        
        if verbose:
            print()
            print("  SLEEP COMPLETE")
            print(f"    Prototypes created: {stats['prototypes_created']}")
            print(f"    Prototypes pruned:  {stats['prototypes_pruned']}")
            print(f"    Schemas: {stats['schemas_discovered']}")
            mem_stats = self.semantic_memory.stats()
            print(f"    Total memory: {mem_stats['total_prototypes']} prototypes")
        
        return stats
    
    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 1,
        use_pattern_completion: Optional[bool] = None,
        use_inhibition: Optional[bool] = None,
        completion_steps: int = 3,
        inhibition_weight: float = PHI_INV_CUBE,
    ) -> Tuple[Optional[SemanticPrototype], float, Dict[str, Any]]:
        """
        Retrieve from semantic memory with brain-inspired features.
        """
        xp = self.xp
        
        do_completion = use_pattern_completion if use_pattern_completion is not None else self.use_pattern_completion
        do_inhibition = use_inhibition if use_inhibition is not None else self.use_inhibition_of_return
        
        info = {
            'used_pattern_completion': do_completion,
            'used_inhibition': do_inhibition,
            'completion_change': 0.0,
            'inhibited_count': 0,
        }
        
        # Apply pattern completion
        completed_query = query
        if do_completion:
            completed_query, comp_info = pattern_complete(
                query, self.basis, xp,
                max_steps=completion_steps,
            )
            info['completion_change'] = comp_info['total_change']
        
        # Retrieve with or without inhibition
        if do_inhibition:
            results, inh_info = retrieve_with_inhibition(
                completed_query, self.semantic_memory, self.retrieval_history,
                top_k=top_k,
                inhibition_weight=inhibition_weight,
                record_retrieval=True,
                xp=xp,
            )
            info['inhibited_count'] = inh_info['inhibited']
        else:
            results = self.semantic_memory.retrieve(
                completed_query,
                top_k=top_k,
                use_pattern_completion=False,
            )
        
        if not results:
            return None, 0.0, info
        
        best_proto, similarity = results[0]
        return best_proto, similarity, info
    
    def generate_rehearsal_episodes(
        self,
        n_episodes: int = 10,
        noise_std: float = PHI_INV_CUBE,
    ) -> List[EpisodicEntry]:
        """
        Generate pseudo-episodes for rehearsal during training.
        """
        if not self.use_pseudo_rehearsal:
            return []
        
        return generate_pseudo_episodes_batch(
            self.semantic_memory, self.basis, self.xp,
            n_episodes=n_episodes,
            noise_std=noise_std,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        return {
            "sleep_cycles": self.sleep_count,
            "total_episodes": self.total_episodes_processed,
            "transition_buffer_count": self.transition_buffer.stats()['count'],
            "retrieval_history_count": self.retrieval_history.stats()['count'],
            **self.semantic_memory.stats(),
        }
