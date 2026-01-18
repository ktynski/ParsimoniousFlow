"""
Dreaming Module — Brain-Inspired Memory Consolidation (v5.0.0)
==============================================================

Implements offline consolidation, abstraction, and memory management
combining TWO COMPLEMENTARY LEARNING SYSTEMS:

INTEGRATED SLEEP (v5.0.0):
    Use `integrated_sleep(memory, dreaming_system, episodes)` for unified:
    
    TOWER DREAMING (Synaptic Level):
        - Non-REM: Witness propagation (master → satellites)
        - REM: φ-jitter exploration in Grace space
        - Operates on holographic memory matrices
        
    SYSTEMS DREAMING (Cortical Level):
        - Non-REM: Cluster episodes → semantic prototypes
        - REM: Recombine prototypes → schemas
        - 12 brain-inspired parsimonies

CORE COMPONENTS:
    - integrated_sleep(): Unified 5-phase sleep cycle
    - DreamingSystem: Systems consolidation with 12 parsimonies
    - SemanticMemory: Hierarchical prototype storage
    - WorkingMemory: 7±2 fast cache with φ-decay
    - NonREMConsolidator: Clustering and prototype formation
    - REMRecombinator: Creative recombination and schema discovery

5-PHASE INTEGRATED SLEEP CYCLE:
    1. Systems Non-REM: Episodic → Prototypes
    2. Tower Non-REM: Witness propagation
    3. Systems REM: Prototype → Schema recombination
    4. Tower REM: φ-jitter exploration
    5. Pruning: Remove weak memories

12 BRAIN-INSPIRED PARSIMONIES:
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
    9. Working Memory Cache
    10. Pattern Completion
    11. Inhibition of Return
    
    SEQUENCE MEMORY:
    12. Sequence Replay

NO FALLBACKS. NO ARBITRARY CONSTANTS. ALL φ-DERIVED.
"""

# Core system - import from new modular files
from .dreaming_system import DreamingSystem
from .semantic_memory import SemanticMemory
from .consolidation import NonREMConsolidator, dream_grace_operator
from .recombination import REMRecombinator

# Data structures
from .structures import (
    EpisodicEntry,
    CompressedEpisode,
    SemanticPrototype,
    Schema,
    MetaSchema,
    RetrievalRecord,
    TemporalTransition,
)

# Sequence replay
from .sequence_replay import (
    compute_transition_vorticity,
    record_transition,
    TransitionBuffer,
    replay_transitions_during_rem,
)

# Inhibition of return
from .inhibition import (
    RetrievalHistory,
    apply_inhibition_of_return,
    retrieve_with_inhibition,
)

# Pseudo-rehearsal
from .pseudo_rehearsal import (
    generate_pseudo_episode,
    generate_pseudo_episodes_batch,
    interleave_with_pseudo_rehearsal,
)

# Priority computation
from .priority import (
    compute_salience,
    compute_salience_batch,
    compute_novelty,
    compute_novelty_batch,
    compute_prediction_error,
    compute_combined_priority,
)

# Pruning
from .pruning import (
    should_prune_prototype,
    prune_semantic_memory,
    prune_attractor_map,
)

# Interference management
from .interference import (
    compute_prototype_similarity,
    merge_prototypes,
    find_similar_prototype_pairs,
    manage_interference,
)

# Memory management
from .memory_management import (
    phi_decay_forget,
    compute_adaptive_threshold,
)

# Reconsolidation
from .reconsolidation import (
    ReconsolidationTracker,
    reconsolidate_attractor,
    reconsolidate_semantic_prototype,
)

# Working memory
from .working_memory import (
    compute_embedding_saliences,
    apply_working_memory_gate,
    gated_context_representation,
    WorkingMemoryBuffer,
    WorkingMemory,
)

# Pattern completion
from .pattern_completion import (
    pattern_complete,
    pattern_complete_batch,
)

# Compression
from .compression import (
    compress_episode,
    compress_episodes_batch,
)

# Predictive coding
from .predictive_coding import (
    predict_from_memory,
    compute_prediction_residual,
    predictive_encode,
    predictive_encode_batch,
)

# Analysis utilities
from .analysis import (
    analyze_memory_scaling,
    estimate_memory_capacity,
    measure_prototype_entropy,
)

# Model integration
from .integration import integrate_dreaming_with_model, integrated_sleep

__all__ = [
    # Core
    'DreamingSystem',
    
    # Memory structures
    'SemanticMemory',
    'SemanticPrototype',
    'EpisodicEntry',
    'CompressedEpisode',
    'Schema',
    'MetaSchema',
    'RetrievalRecord',
    'TemporalTransition',
    'WorkingMemory',
    'WorkingMemoryBuffer',
    
    # Consolidation
    'NonREMConsolidator',
    'REMRecombinator',
    'dream_grace_operator',
    
    # Priority computation
    'compute_salience',
    'compute_salience_batch',
    'compute_novelty',
    'compute_novelty_batch',
    'compute_prediction_error',
    'compute_combined_priority',
    
    # Pruning
    'should_prune_prototype',
    'prune_semantic_memory',
    'prune_attractor_map',
    
    # Interference management
    'compute_prototype_similarity',
    'merge_prototypes',
    'find_similar_prototype_pairs',
    'manage_interference',
    
    # Memory management
    'phi_decay_forget',
    'compute_adaptive_threshold',
    
    # Reconsolidation
    'ReconsolidationTracker',
    'reconsolidate_attractor',
    'reconsolidate_semantic_prototype',
    
    # Working memory
    'compute_embedding_saliences',
    'apply_working_memory_gate',
    'gated_context_representation',
    
    # Sequence replay
    'compute_transition_vorticity',
    'record_transition',
    'TransitionBuffer',
    'replay_transitions_during_rem',
    
    # Inhibition of return
    'RetrievalHistory',
    'apply_inhibition_of_return',
    'retrieve_with_inhibition',
    
    # Pseudo-rehearsal
    'generate_pseudo_episode',
    'generate_pseudo_episodes_batch',
    'interleave_with_pseudo_rehearsal',
    
    # Pattern completion
    'pattern_complete',
    'pattern_complete_batch',
    
    # Compression
    'compress_episode',
    'compress_episodes_batch',
    
    # Predictive coding
    'predict_from_memory',
    'compute_prediction_residual',
    'predictive_encode',
    'predictive_encode_batch',
    
    # Analysis utilities
    'analyze_memory_scaling',
    'estimate_memory_capacity',
    'measure_prototype_entropy',
    
    # Model integration
    'integrate_dreaming_with_model',
    'integrated_sleep',
]
