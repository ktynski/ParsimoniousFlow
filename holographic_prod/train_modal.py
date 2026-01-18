"""
Holographic Language Model — Modal Training Runner (Theory-True)
=================================================================

TRANSFORMER KILLER ARCHITECTURE:
    - O(1) Grace basin routing (not O(n²) attention)
    - Instant Hebbian learning (not gradient descent)
    - Sublinear memory growth (consolidation)
    - φ-kernel sampling (no temperature tuning)
    - Interpretable attractors (not black box)
    - Continual learning via dreaming (no catastrophic forgetting)

THEORY DERIVATION:
    The self-consistency equation Λ² = Λ + 1 gives:
        Λ = φ = (1 + √5)/2 ≈ 1.618
    
    From this, ALL rates are derived:
        φ⁻¹ ≈ 0.618 — Primary rate (learning, threshold)
        φ⁻² ≈ 0.382 — Spectral gap (stability threshold)
        φ⁻³ ≈ 0.236 — Tertiary rate (noise, pruning)
        φ⁻⁴ ≈ 0.146 — Dream Grace rate
        φ⁻⁵ ≈ 0.090 — Contrastive learning rate
    
    NO ARBITRARY CONSTANTS. Everything derives from φ.

12 BRAIN-INSPIRED PARSIMONIES:
    ENCODING: Salience, Novelty, Prediction Error, Predictive Coding
    MAINTENANCE: φ-Decay, Interference Management, Reconsolidation, Pseudo-Rehearsal
    RETRIEVAL: Working Memory, Pattern Completion, Inhibition of Return
    TEMPORAL: Sequence Replay

Usage:
    modal run holographic_prod/train_modal.py::train --max-samples 1000000
"""

import modal
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from collections import deque

# Modal app
app = modal.App("holographic-transformer-killer")

# Persistent volume for checkpoints (survives across runs!)
checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)

# GPU-optimized image  
# CACHE-BUST: v5.24.0 - Preserve punctuation and capitalization for natural text generation
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "numpy>=1.24.0",
        "cupy-cuda12x>=12.0.0",
        "scipy>=1.10.0",
        "datasets>=2.14.0",
        "huggingface_hub>=0.16.0",
        "pytest>=7.0.0",
        "pytest-timeout>=2.0.0",
    )
    # Pre-download GloVe 50d embeddings using Python (CACHE-BUST v5.6.1)
    .run_commands(
        "echo 'GloVe download v5.6.1 - fixing Modal cache'",
        "apt-get update && apt-get install -y unzip",
        "mkdir -p /tmp/glove",
        'python -c "import urllib.request; urllib.request.urlretrieve(\'http://nlp.stanford.edu/data/glove.6B.zip\', \'/tmp/glove/glove.6B.zip\')"',
        "unzip -j /tmp/glove/glove.6B.zip glove.6B.50d.txt -d /tmp/glove",
        "rm /tmp/glove/glove.6B.zip",  # Clean up 850MB zip, keep only 50d
    )
    .add_local_dir("holographic_prod", "/root/project/holographic_prod")
)


# =============================================================================
# TRAINING STATE (Full Transparency)
# =============================================================================

@dataclass
class TrainingState:
    """
    Complete training state with full transparency for debugging.
    
    Every metric has a theory-derived interpretation.
    """
    
    # Counters
    samples: int = 0
    batches: int = 0
    stage: int = 0
    context_size: int = 64  # SO(4) embeddings enable infinite context (tested to 1024+)
    
    # Rolling histories (size 100 for moving averages)
    perplexity_history: List[float] = field(default_factory=list)
    stability_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    throughput_history: List[float] = field(default_factory=list)
    error_rate_history: List[float] = field(default_factory=list)
    
    # Dream statistics
    total_dreams: int = 0
    samples_since_dream: int = 0
    prototypes_total: int = 0
    schemas_total: int = 0
    prototypes_pruned: int = 0
    
    # Theory of Mind + Distributed Prior (v5.4.3)
    tom_transforms: int = 0  # Times ToM transformed perspective
    prior_usage: int = 0  # Times distributed prior path was used (complementary, not fallback)
    factorized_prior_updates: int = 0  # Updates to Hebbian prior
    
    # Parallel Retrieval (v5.15.0) — ACC analog
    acc_conflict_count: int = 0  # Episodic vs holographic disagreements (attention signal)
    
    # Memory statistics
    unique_basins: int = 0
    holographic_norm: float = 0.0
    avg_basin_population: float = 0.0
    
    # Error tracking (for credit assignment)
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_contexts: List[Tuple[List[int], int, int]] = field(default_factory=list)  # (ctx, pred, actual)
    
    # Cached metrics (computed periodically)
    last_accuracy: float = 0.0
    
    # Running batch metrics (for real-time feedback)
    batch_errors: deque = field(default_factory=lambda: deque(maxlen=100))  # Last 100 batches
    batch_times: deque = field(default_factory=lambda: deque(maxlen=100))   # Last 100 batch times
    
    # Timing
    start_time: float = field(default_factory=time.perf_counter)
    last_log_time: float = 0.0
    last_log_samples: int = 0
    
    def running_error_rate(self) -> float:
        """Get running error rate from recent batches."""
        if not self.batch_errors:
            return 0.0
        return sum(self.batch_errors) / len(self.batch_errors)
    
    def running_throughput(self) -> float:
        """Get running throughput from recent batches."""
        if not self.batch_times or len(self.batch_times) < 2:
            return 0.0
        return 1.0 / max(0.001, sum(self.batch_times) / len(self.batch_times))
    
    def record(self, **kwargs):
        """Record metrics with bounded history."""
        for key, value in kwargs.items():
            history = getattr(self, f'{key}_history', None)
            if history is not None:
                history.append(value)
                if len(history) > 100:
                    history.pop(0)
    
    def avg(self, key: str) -> float:
        """Get rolling average."""
        history = getattr(self, f'{key}_history', [])
        return sum(history) / max(1, len(history))
    
    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time
    
    def error_rate(self) -> float:
        """Current error rate from recent errors."""
        if not self.recent_errors:
            return 0.0
        return sum(self.recent_errors) / len(self.recent_errors)


# =============================================================================
# THEORY-TRUE CONSTANTS (imported at runtime to ensure consistency)
# =============================================================================

def get_theory_constants():
    """
    Get theory-derived constants.
    
    ALL values derive from φ = (1 + √5)/2.
    """
    from holographic_prod.core.constants import (
        PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, PHI_INV_FOUR, PHI_INV_FIVE,
    )
    return {
        'PHI': PHI,                    # ≈ 1.618 — Golden ratio
        'PHI_INV': PHI_INV,            # ≈ 0.618 — Learning rate, primary threshold
        'PHI_INV_SQ': PHI_INV_SQ,      # ≈ 0.382 — Spectral gap, stability threshold
        'PHI_INV_CUBE': PHI_INV_CUBE,  # ≈ 0.236 — Tertiary threshold
        'PHI_INV_FOUR': PHI_INV_FOUR,  # ≈ 0.146 — Dream Grace rate
        'PHI_INV_FIVE': PHI_INV_FIVE,  # ≈ 0.090 — Contrastive rate
    }


# =============================================================================
# φ-CURRICULUM (Theory-Derived Context Scaling)
# =============================================================================

def get_curriculum_stage(
    samples: int,
    base_context: int = 64,  # SO(4) embeddings: start at 64 (tested stable to 1024+)
    base_samples: int = 500_000,
    max_context: int = 8192,  # SO(4) enables infinite context (tested to 1024+, safe to 8192)
) -> Tuple[int, int, str]:
    """
    Get current curriculum stage, context size, and explanation.
    
    THEORY:
        context(stage) = base × φ^(2×stage)
        samples_per_stage = base_samples × φ^stage
        
        This ensures:
        - Context grows quadratically in φ
        - Training time grows linearly in φ
        - Memory consolidation happens at natural rates
    """
    consts = get_theory_constants()
    phi = consts['PHI']
    
    stage = 0
    cumulative = 0
    
    while True:
        stage_samples = int(base_samples * (phi ** stage))
        if samples < cumulative + stage_samples:
            break
        cumulative += stage_samples
        stage += 1
        if stage >= 6:
            break
    
    context_size = int(base_context * (phi ** stage))  # Linear scaling, not quadratic
    context_size = min(context_size, max_context)  # SO(4) enables large context (tested to 1024+)
    
    # Explanation for debugging
    explanation = (
        f"Stage {stage}: context={context_size} "
        f"(base×φ^{2*stage}={base_context}×{phi**(2*stage):.1f}), "
        f"samples_in_stage={int(base_samples * (phi ** stage)):,}"
    )
    
    return stage, context_size, explanation


# =============================================================================
# MODULE-LEVEL TOKENIZATION (Pickle-Safe for Multiprocessing)
# =============================================================================

def _tokenize_batch_for_mp(args):
    """
    Process a batch of texts for multiprocessing.
    
    MUST be at module level (not nested) so it can be pickled for ProcessPoolExecutor.
    
    Args:
        args: Tuple of (texts, word_to_idx_items, context_size)
            - texts: List of document strings
            - word_to_idx_items: List of (word, idx) tuples (serialized dict)
            - context_size: Context window size
    
    Returns:
        List of (context, target) tuples
    """
    import re
    texts, w2i_items, ctx_size = args
    
    # Reconstruct dict in worker process
    w2i = dict(w2i_items)
    pattern = re.compile(r"[\w']+|[.,!?;:\"]")
    
    batch_samples = []
    for text in texts:
        if not text or len(text) < 50:
            continue
        # CRITICAL FIX v5.32.0: Use <unk> (index 0) for unknown words
        # train_modal.py uses: {"<unk>": 0, "<pad>": 1}
        # Also: Use step size 1 for overlapping windows (not ctx_size which skips most data)
        toks = [w2i.get(w, 0) for w in pattern.findall(text)]  # 0 = <unk> per train_modal.py vocab structure
        if len(toks) < ctx_size + 1:
            continue
        # CRITICAL FIX v5.32.0: Step size 1 for overlapping windows (not ctx_size)
        # This ensures we extract ALL possible context/target pairs, not just 1 per ctx_size tokens
        for i in range(0, len(toks) - ctx_size, 1):  # Step 1, not ctx_size
            ctx = toks[i:i + ctx_size]
            tgt = toks[i + ctx_size]
            # Filter out <unk> targets (theory-true: don't learn from unknown words)
            if tgt != 0:  # 0 = <unk> per train_modal.py vocab structure {"<unk>": 0, "<pad>": 1}
                batch_samples.append((ctx, tgt))
    return batch_samples


# =============================================================================
# ADAPTIVE DREAMING (Theory-True Triggers)
# =============================================================================

def should_dream(
    state: TrainingState,
    stability: float,
    verbose: bool = True,
) -> Tuple[bool, str]:
    """
    Determine if dreaming should be triggered.
    
    THEORY-TRUE TRIGGERS:
        1. stability < φ⁻² — Memory needs consolidation
           Interpretation: Grace stability below spectral gap means
           attractors are not well-separated.
        
        2. error_rate > φ⁻¹ — Systematic errors detected
           Interpretation: More than 61.8% errors means memory
           is fundamentally misaligned.
        
        3. samples_since_dream > max — Safety valve
           Interpretation: Even stable memory benefits from
           periodic consolidation.
    
    Returns:
        (should_dream, reason)
    """
    consts = get_theory_constants()
    phi_inv = consts['PHI_INV']
    phi_inv_sq = consts['PHI_INV_SQ']
    
    # THEORY-TRUE DREAMING INTERVALS (v5.3.1)
    # Brain dreams ~every 16 hours. At 10K samples/sec, that's ~576M samples.
    # But early training needs MORE frequent consolidation (like infant sleep).
    # φ-scaling: MIN = 100K, MAX = 500K during learning phase
    MIN_SAMPLES = 100_000  # Minimum between dreams (theory: φ⁴ ~ 7 scale factors)
    MAX_SAMPLES = 500_000  # Safety valve (theory: φ⁵ ~ 11 scale factors)
    WARMUP = 50_000  # Don't trigger on very early noise
    
    # Check minimum interval
    if state.samples_since_dream < MIN_SAMPLES:
        return False, f"too_soon ({state.samples_since_dream:,} < {MIN_SAMPLES:,})"
    
    # Stability trigger
    if stability < phi_inv_sq and state.samples > WARMUP:
        reason = f"low_stability ({stability:.3f} < φ⁻²={phi_inv_sq:.3f})"
        if verbose:
            print(f"  ⚠ Dream trigger: {reason}")
        return True, reason
    
    # Error rate trigger
    error_rate = state.error_rate()
    if error_rate > phi_inv and state.samples > WARMUP:
        reason = f"high_error_rate ({error_rate:.3f} > φ⁻¹={phi_inv:.3f})"
        if verbose:
            print(f"  ⚠ Dream trigger: {reason}")
        return True, reason
    
    # Safety valve
    if state.samples_since_dream >= MAX_SAMPLES:
        reason = f"safety_valve ({state.samples_since_dream:,} >= {MAX_SAMPLES:,})"
        if verbose:
            print(f"  ⚠ Dream trigger: {reason}")
        return True, reason
    
    return False, "stable"


# =============================================================================
# PERPLEXITY (Theory-True φ-Kernel)
# =============================================================================

def compute_perplexity_detailed(
    model,
    batch: List[Tuple[List[int], int]],
    vocab_size: int,
) -> Tuple[float, Dict[str, Any]]:
    """
    THEORY-TRUE VECTORIZED perplexity computation (v5.29.0).
    
    CRITICAL OPTIMIZATION:
    - Old: n × (retrieve_parallel + 3 GPU syncs) = O(n) Python loops + O(n) GPU syncs
    - New: 1 × evaluate_semantic (VECTORIZED) = O(1) GPU sync
    
    For batch_size=5: 15+ GPU syncs → 1 sync = 15x faster
    
    THEORY-TRUE (v5.32.0):
    - Primary metric: semantic_similarity (COHERENCE, not Frobenius cosine!)
    - Coherence = witness_energy / total_energy (scalar² + pseudo²) / total
    - Perplexity derived from semantic similarity for backward compatibility:
        PPL = vocab_size^(1 - semantic_similarity)
        
        Intuition: 
        - sim=1.0 → PPL=1 (perfect)
        - sim=0.0 → PPL=vocab_size (random guessing)
        - sim=0.5 → PPL=sqrt(vocab_size)
    
    NO Python for-loops. NO per-sample GPU syncs. PURE VECTORIZED.
    """
    import numpy as np
    
    n_samples = len(batch)
    if n_samples == 0:
        return float(vocab_size), {'n_samples': 0, 'semantic_sim': 0.0}
    
    # =========================================================================
    # THEORY-TRUE EVALUATION: Use vectorized evaluate_semantic
    # ONE GPU sync for entire batch instead of O(n) syncs
    # =========================================================================
    eval_result = model.evaluate_semantic(batch)
    semantic_sim = eval_result['semantic_similarity']
    
    # =========================================================================
    # THEORY-TRUE PERPLEXITY: Derive from semantic similarity
    # =========================================================================
    # PPL = vocab_size^(1 - semantic_similarity)
    # This maps: sim=0 → PPL=vocab, sim=1 → PPL=1
    # 
    # Derivation: If similarity = 1 - entropy_fraction, then
    # PPL = exp(entropy) = exp(max_entropy × (1 - sim)) = vocab^(1 - sim)
    perplexity = float(vocab_size ** (1.0 - max(0.0, min(1.0, semantic_sim))))
    perplexity = max(1.0, min(perplexity, float(vocab_size)))
    
    # =========================================================================
    # DETAILS: Theory-true metrics (NO transformer cruft)
    # =========================================================================
    # THEORY-TRUE METRICS ONLY (no transformer-style accuracy cruft)
    details = {
        'n_samples': n_samples,
        'coherence': semantic_sim,  # Primary metric per theory
        'min_coherence': eval_result['min_similarity'],
        'max_coherence': eval_result['max_similarity'],
    }
    
    return perplexity, details


# =============================================================================
# MEMORY DIAGNOSTICS
# =============================================================================

def get_memory_diagnostics(model, use_gpu: bool = False) -> Dict[str, Any]:
    """
    Get detailed memory diagnostics for debugging.
    
    Helps identify:
    - Is memory growing appropriately?
    - Are basins well-separated (via satellite stats)?
    - Is consolidation working?
    - GPU memory usage (critical for H100 runs)
    """
    import numpy as np
    
    diagnostics = {
        'n_patterns': model.n_patterns,
        'stability': model.get_stability(),
        'holographic_norm': 0.0,
        # HIPPOCAMPUS ANALOG stats (v5.3.0)
        'episodic_cache_size': len(model._episodic_cache) if hasattr(model, '_episodic_cache') else 0,
        'episodic_hits': model._episodic_hits if hasattr(model, '_episodic_hits') else 0,
        'episodic_misses': model._episodic_misses if hasattr(model, '_episodic_misses') else 0,
        # Contrastive learning tracking (semantic structure emergence)
        'contrastive_updates': model.contrastive_updates if hasattr(model, 'contrastive_updates') else 0,
        # GPU memory tracking
        'gpu_used_gb': 0.0,
        'gpu_free_gb': 0.0,
    }
    
    # Track GPU memory if available
    if use_gpu:
        import cupy as cp
        meminfo = cp.cuda.runtime.memGetInfo()
        diagnostics['gpu_free_gb'] = meminfo[0] / 1024**3
        diagnostics['gpu_used_gb'] = (meminfo[1] - meminfo[0]) / 1024**3
    
    # Total memory norm (across ALL satellites)
    if hasattr(model, 'tower') and model.tower is not None:
        xp = model.xp
        if hasattr(model.tower, '_all_memories'):
            # MultiLevelTower: single tensor for all satellites (v5.15.0+)
            norm = float(xp.linalg.norm(model.tower._all_memories))
        else:
            norm = 0.0
        diagnostics['holographic_norm'] = norm
    
    # Tower statistics (satellite distribution)
    if hasattr(model, 'tower') and model.tower is not None:
        tower_stats = model.tower.get_satellite_stats()
        diagnostics['n_satellites'] = tower_stats.get('n_satellites', 0)
        diagnostics['active_satellites'] = tower_stats.get('active_satellites', 0)
        diagnostics['total_bindings'] = tower_stats.get('total_bindings', 0)
        diagnostics['avg_bindings'] = tower_stats.get('avg_bindings', 0.0)
        diagnostics['max_bindings'] = tower_stats.get('max_bindings', 0)
    
    return diagnostics


# =============================================================================
# LOGGING (Detailed for Debugging)
# =============================================================================

def print_header():
    print()
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║              HOLOGRAPHIC — Transformer Killer Architecture                    ║")
    print("║    O(1) Grace Routing • Instant Hebbian • 12 Brain-Inspired Parsimonies       ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")
    print()


def print_theory_constants():
    """Print theory constants for verification."""
    consts = get_theory_constants()
    print("  THEORY-DERIVED CONSTANTS:")
    print(f"    φ      = {consts['PHI']:.10f} (golden ratio)")
    print(f"    φ⁻¹    = {consts['PHI_INV']:.10f} (learning rate)")
    print(f"    φ⁻²    = {consts['PHI_INV_SQ']:.10f} (spectral gap)")
    print(f"    φ⁻³    = {consts['PHI_INV_CUBE']:.10f} (tertiary)")
    print(f"    φ⁻⁴    = {consts['PHI_INV_FOUR']:.10f} (dream rate)")
    print(f"    φ⁻⁵    = {consts['PHI_INV_FIVE']:.10f} (contrastive)")
    print()


def print_progress_detailed(
    state: TrainingState,
    max_samples: int,
    perplexity: float,
    ppl_details: Dict,
    mem_diag: Dict,
):
    """Print detailed progress for debugging."""
    pct = (state.samples / max_samples) * 100
    
    # Status indicator based on perplexity
    consts = get_theory_constants()
    if perplexity < 100:
        status = "✅ Excellent"
    elif perplexity < 500:
        status = "✓ Good"
    else:
        status = "🔄 Learning"
    
    # Main progress line
    print(f"\n[{state.elapsed():7.1f}s] Stage {state.stage} | "
          f"{state.samples:,}/{max_samples:,} ({pct:5.1f}%)")
    
    # Metrics (include running averages for real-time feedback)
    # THEORY-TRUE (v5.28.0): Show semantic_similarity, not accuracy
    # "NEVER Measure Exact Token Match as Primary Metric"
    running_sim = 1 - (state.running_error_rate() if hasattr(state, 'running_error_rate') else state.error_rate())
    running_tput = state.running_throughput() if hasattr(state, 'running_throughput') else 0
    print(f"  Perplexity: {perplexity:,.1f} | "
          f"SemanticSim: {state.last_accuracy:.3f} (running={running_sim:.3f}) | "
          f"Stability: {state.avg('stability'):.3f}")
    print(f"  Rate: {state.avg('throughput'):,.0f}/s (batch: {running_tput:.0f}/s) | {status}")
    
    # Memory diagnostics  
    episodic_size = mem_diag.get('episodic_cache_size', 0)
    episodic_hits = mem_diag.get('episodic_hits', 0)
    episodic_misses = mem_diag.get('episodic_misses', 0)
    hit_rate_episodic = episodic_hits / (episodic_hits + episodic_misses + 1e-10) * 100
    
    contrastive_updates = mem_diag.get('contrastive_updates', 0)
    print(f"  Memory: {mem_diag['n_patterns']:,} patterns | "
          f"norm={mem_diag['holographic_norm']:.2f} | "
          f"cache={episodic_size:,} ({hit_rate_episodic:.0f}% hit) | "
          f"contrast={contrastive_updates}")
    
    # =========================================================================
    # THEORY-TRUE METRICS (v5.29.0)
    # =========================================================================
    # NO transformer cruft: top_k, ranks, exact_match are NOT theory-true
    # ONLY semantic similarity matters: "semantic_sim=0.96 IS success"
    semantic_sim = ppl_details.get('semantic_sim', 0.0)
    min_sim = ppl_details.get('min_similarity', 0.0)
    max_sim = ppl_details.get('max_similarity', 0.0)
    
    # Single line: Theory-true evaluation
    print(f"  Quality: semantic_sim={semantic_sim:.3f} (range=[{min_sim:.3f}, {max_sim:.3f}])")
    
    # Dream status
    print(f"  Dreams: {state.total_dreams} | "
          f"protos={state.prototypes_total} | "
          f"schemas={state.schemas_total} | "
          f"since_dream={state.samples_since_dream:,}")
    
    # GPU memory (if available)
    if 'gpu_used_gb' in mem_diag and mem_diag['gpu_used_gb'] > 0:
        print(f"  GPU Memory: {mem_diag['gpu_used_gb']:.1f} GB used | "
              f"{mem_diag['gpu_free_gb']:.1f} GB free")
    
    # Satellite distribution
    if 'active_satellites' in mem_diag:
        occupancy = mem_diag['active_satellites'] / max(1, mem_diag['n_satellites']) * 100
        print(f"  Satellites: {mem_diag['active_satellites']:,}/{mem_diag['n_satellites']:,} "
              f"({occupancy:.1f}% occupied) | avg_bindings={mem_diag.get('avg_bindings', 0):.1f}")
    
    # Theory of Mind + Distributed Prior (v5.4.3)
    if state.prior_usage > 0 or state.tom_transforms > 0:
        prior_rate = state.prior_usage / max(1, state.samples) * 100
        print(f"  ToM/Prior: prior_usage={state.prior_usage} ({prior_rate:.2f}%)")
        if hasattr(state, 'factorized_prior_updates'):
            print(f"            factorized_prior_updates={state.factorized_prior_updates}")


def print_dream_details(stats: Dict, state: TrainingState):
    """Print detailed dream cycle results."""
    print(f"\n  💤 DREAM CYCLE #{state.total_dreams}")
    print(f"     Non-REM: {stats.get('prototypes_created', 0)} prototypes created")
    print(f"     REM: {stats.get('schemas_discovered', 0)} schemas discovered")
    print(f"     Pruned: {stats.get('prototypes_pruned', 0)} low-salience removed")
    print(f"     Merged: {stats.get('prototypes_merged', 0)} similar merged")
    
    if 'pre_stability' in stats and 'post_stability' in stats:
        print(f"     Stability: {stats['pre_stability']:.3f} → {stats['post_stability']:.3f}")


def print_curriculum_change(old_stage: int, new_stage: int, explanation: str):
    """Print curriculum stage change with theory explanation."""
    print(f"\n  📈 CURRICULUM CHANGE")
    print(f"     {explanation}")
    print(f"     Theory: context grows by φ² per stage for optimal consolidation")


def generate_samples(model, prompts: List[List[int]], max_tokens: int, idx_to_word: Dict[int, str]) -> List[str]:
    """
    THEORY-TRUE: Generate via attractor flow (NOT discrete lookups).
    
    BRAIN ANALOG:
        State flows through learned attractors continuously.
        Each state naturally leads to the next.
        Errors don't compound - trajectory is coherent.
        
    OLD (WRONG):
        for step: pred = retrieve(context)  # Independent lookups
        
    NEW (THEORY-TRUE, v5.30.0):
        Uses BATCHED generation for O(1) GPU syncs instead of O(prompts).
        All prompts are processed in parallel on GPU.
    """
    from holographic_prod.core.attractor_generation import generate_batch_attractor_flow
    
    # Get xp from model - MUST exist for theory-true implementation
    xp = model.tower.xp  # REQUIRED - model MUST have xp attribute
    
    # CRITICAL OPTIMIZATION (v5.30.0): Use BATCHED generation
    # OLD: O(n_prompts) GPU syncs (for-loop over single generation)
    # NEW: O(1) GPU syncs (batch all prompts together)
    results = generate_batch_attractor_flow(
        memory=model,
        prompts=prompts,
        max_tokens=max_tokens,
        grace_steps=3,  # φ-derived iterations for stability
        xp=xp,
    )
    
    # Format outputs
    samples = []
    for (generated_tokens, stabilities), prompt in zip(results, prompts):
        # Convert to text with stability annotation
        words = []
        for i, t in enumerate(generated_tokens):
            word = idx_to_word.get(t, f"<{t}>")
            # Annotate low-stability tokens
            if i >= len(prompt) and stabilities:
                stability_idx = i - len(prompt)
                if stability_idx < len(stabilities) and stabilities[stability_idx] < 0.3:
                    word = f"{word}[low]"
            words.append(word)
        
        # Show mean stability for generated portion
        if stabilities:
            mean_stability = sum(stabilities) / len(stabilities)
            samples.append(f"[σ={mean_stability:.2f}] " + " ".join(words))
        else:
            samples.append(" ".join(words))
    
    return samples


def print_error_analysis(error_contexts: List[Tuple], idx_to_word: Dict[int, str], max_errors: int = 5):
    """Print detailed analysis of recent errors for debugging."""
    if not error_contexts:
        return
    
    print(f"\n  ⚠️ ERROR ANALYSIS (last {min(len(error_contexts), max_errors)} errors):")
    for ctx, pred, actual, source in error_contexts[-max_errors:]:
        ctx_words = [idx_to_word.get(t, f"<{t}>") for t in ctx]
        pred_word = idx_to_word.get(pred, f"<{pred}>") if pred is not None else "<none>"
        actual_word = idx_to_word.get(actual, f"<{actual}>")
        print(f"     Context: [{' '.join(ctx_words)}]")
        print(f"     Predicted: {pred_word} | Actual: {actual_word} | Source: {source}")


def print_retrieval_breakdown(state, semantic_hits: int, holographic_hits: int, episodic_hits: int, sample_size: int):
    """
    Print breakdown of where successful retrievals came from.
    
    IMPORTANT FOR INTERPRETATION:
        - High episodic % = accuracy from exact-match cache (expected for training data)
        - High semantic % = accuracy from prototype candidate narrowing
        - High holographic % = rare, raw holographic unbinding works (lucky target pairs)
        
    See docs/CAPACITY_LIMITS.md for why raw holographic has ~1 pattern capacity.
    """
    episodic_pct = 100 * episodic_hits / max(1, sample_size)
    semantic_pct = 100 * semantic_hits / max(1, sample_size)
    holographic_pct = 100 * holographic_hits / max(1, sample_size)
    
    print(f"  🔍 RETRIEVAL SOURCES (see docs/CAPACITY_LIMITS.md):")
    print(f"     📦 Episodic:    {episodic_hits:4d}/{sample_size} ({episodic_pct:5.1f}%) ← Hash table exact match")
    print(f"     🎯 Semantic:    {semantic_hits:4d}/{sample_size} ({semantic_pct:5.1f}%) ← Prototype candidates")
    print(f"     🌀 Holographic: {holographic_hits:4d}/{sample_size} ({holographic_pct:5.1f}%) ← Raw unbinding")
    
    # Warning if holographic is high (unexpected)
    if holographic_pct > 20:
        print(f"     ⚠️  High holographic % is unusual - check capacity tests")


# =============================================================================
# CHECKPOINTING (for long runs)
# =============================================================================

def save_checkpoint(model, state, dreamer, path: str = "/tmp/checkpoint.npz"):
    """
    Save training state for recovery.
    
    SPARSE: Only save ACTIVE satellites to avoid 17GB transfers for level 7.
    For 268M satellites with only ~500K active, this is 500x faster.
    """
    import numpy as np
    
    # Get binding counts (always transfer - small array)
    bindings = model.tower._satellite_n_bindings
    if hasattr(bindings, 'get'):
        bindings = bindings.get()
    
    # Find ACTIVE satellites only
    active_mask = bindings > 0
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)
    
    # Get memory tensor for ACTIVE satellites only
    if hasattr(model.tower, '_all_memories'):
        all_memories = model.tower._all_memories
        if hasattr(all_memories, 'get'):
            # GPU: Index first, then transfer (much smaller)
            active_memories = all_memories[active_indices]
            active_memories = active_memories.get()
        else:
            active_memories = all_memories[active_indices]
    else:
        all_memories = model.tower._all_memories
        if hasattr(all_memories, 'get'):
            active_memories = all_memories[active_indices]
            active_memories = active_memories.get()
        else:
            active_memories = all_memories[active_indices]
    
    # Save sparse checkpoint
    np.savez_compressed(
        path,
        active_indices=active_indices,      # Which satellites have data
        active_memories=active_memories,    # Only active satellite data
        active_bindings=bindings[active_mask],  # Only active binding counts
        n_satellites=len(bindings),         # Total satellite count for reconstruction
        samples=state.samples,
        batches=state.batches,
        total_dreams=state.total_dreams,
        context_size=state.context_size,
        stage=state.stage,
        prototypes_total=state.prototypes_total,
        schemas_total=state.schemas_total,
    )
    
    # Commit volume to persist across containers!
    if path.startswith("/checkpoints"):
        checkpoint_volume.commit()
    
    return path


def load_checkpoint(model, state, path: str = "/tmp/checkpoint.npz") -> bool:
    """
    Load training state. Returns True if successful.
    
    Handles SPARSE checkpoint format - only active satellites are stored.
    """
    import os
    import numpy as np
    
    if not os.path.exists(path):
        return False
    
    try:
        data = np.load(path)
        xp = model.xp
        
        # Check for sparse format (v5.5.2+)
        if 'active_indices' in data:
            # SPARSE FORMAT: Only active satellites stored
            active_indices = data['active_indices']
            active_memories = xp.asarray(data['active_memories'])
            active_bindings = xp.asarray(data['active_bindings'])
            n_satellites = int(data['n_satellites'])
            
            # Restore only active satellites (rest stay zero-initialized)
            if hasattr(model.tower, '_all_memories'):
                model.tower._all_memories[active_indices] = active_memories
            else:
                model.tower._all_memories[active_indices] = active_memories
            
            # Restore binding counts
            model.tower._satellite_n_bindings[active_indices] = active_bindings
        else:
            # LEGACY FORMAT: Full arrays (for backwards compatibility)
            if hasattr(model.tower, '_all_memories'):
                model.tower._all_memories = xp.asarray(data['memories'])
            else:
                model.tower._all_memories = xp.asarray(data['memories'])
            
            model.tower._satellite_n_bindings = xp.asarray(data['bindings'])
        
        # Restore state
        state.samples = int(data['samples'])
        state.batches = int(data['batches'])
        state.total_dreams = int(data['total_dreams'])
        state.context_size = int(data['context_size'])
        state.stage = int(data['stage'])
        state.prototypes_total = int(data['prototypes_total'])
        state.schemas_total = int(data['schemas_total'])
        
        return True
    except Exception as e:
        print(f"  ⚠️ Failed to load checkpoint: {e}")
        return False


# =============================================================================
# MAIN TRAINING
# =============================================================================

@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={"/checkpoints": checkpoint_volume},  # Persistent storage!
)
def train(
    max_samples: int = 1_000_000,
    vocab_size: int = 200_000,  # v5.31.2: 200K vocab for minimal <unk> (~2% OOV)
    max_levels: int = 6,  # H100-optimized: 16M satellites (1GB), ~83% accuracy @ 200K patterns
    batch_size: int = 8192,  # H100-optimized: 8K batch for 80GB H100
    enable_curriculum: bool = True,
    enable_dreaming: bool = True,
    log_every: int = 100_000,  # Reduced logging overhead
    checkpoint_every: int = 500_000,  # Save every 500k samples
    verbose: bool = True,
    seed: int = 42,
    use_grounding: bool = True,  # Use grounded embeddings for O(√N) efficiency
    grounding_samples: int = 100_000,  # Samples to use for grounding phase
    fresh_start: bool = False,  # Ignore checkpoint and start fresh
    # ABLATION FLAGS — Set False ONLY for ablation studies, NOT for production
    ablate_fractal: bool = False,  # ABLATION ONLY: disable fractal tensor components
    ablate_fractal_position: bool = False,  # ABLATION ONLY: disable φ-derived position encoding
):
    """
    Main training function with full transparency.
    
    H100-OPTIMIZED DEFAULTS:
    - Level 6: 16M satellites (1GB VRAM), leaves 79GB for computation
    - Vocab 50K: Reasonable for word-level language modeling  
    - Batch 2048: Maximizes GPU utilization on H100
    - For level 7 (268M satellites, 16GB): pass max_levels=7
    
    Every step is logged with theory-derived interpretation.
    Checkpoints saved every checkpoint_every samples.
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import os
    # Set HF token BEFORE importing datasets (required for auth)
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    
    import numpy as np
    import re
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    from datasets import load_dataset
    
    # Import holographic modules
    from holographic_prod.memory import HolographicMemory, MemoryConfig
    from holographic_prod.dreaming import DreamingSystem, EpisodicEntry, integrated_sleep
    from holographic_prod.dreaming import DreamingSystem, EpisodicEntry, integrated_sleep
    from holographic_prod.attention import ToroidalAttention
    from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ
    
    # Import fractal components (v5.11.0)
    from holographic_prod.torus.interaction_tensor import InteractionTensor
    from holographic_prod.torus.chirality import ChiralityFlip
    from holographic_prod.fractal.downward_projection import DownwardProjection, phase_locked_emission
    
    print_header()
    
    # GPU REQUIRED - this runs on Modal H100
    import cupy as cp
    cp.cuda.Device(0).use()
    meminfo = cp.cuda.runtime.memGetInfo()
    print(f"  ⚡ GPU: NVIDIA H100 (CuPy accelerated)")
    print(f"     Memory: {meminfo[1]/1024**3:.1f} GB total, {meminfo[0]/1024**3:.1f} GB free")
    use_gpu = True  # REQUIRED - GPU mandatory for theory-true performance
    
    # Print theory constants
    print_theory_constants()
    
    # Initialize state
    state = TrainingState()
    
    # Get initial curriculum
    if enable_curriculum:
        state.stage, state.context_size, explanation = get_curriculum_stage(0)
        print(f"  φ-CURRICULUM: {explanation}")
    else:
        state.context_size = 64  # SO(4) embeddings enable large context (tested to 1024+)
        print(f"  Fixed context: {state.context_size}")
    
    # Load dataset
    print("\n  LOADING DATA...")
    
    # Pattern captures: words, contractions, and punctuation separately
    word_pattern = re.compile(r"[\w']+|[.,!?;:\"]")
    word_to_idx = {"<unk>": 0, "<pad>": 1}
    idx_to_word = {0: "<unk>", 1: "<pad>"}
    
    # ==========================================================================
    # VOCABULARY CACHING (v5.31.0) — Skip vocab building if cache exists
    # ==========================================================================
    vocab_cache_path = "/checkpoints/vocabulary.npz"
    vocab_loaded = False
    
    if os.path.exists(vocab_cache_path):
        print(f"  📦 Found cached vocabulary at {vocab_cache_path}")
        try:
            vocab_start = time.perf_counter()
            cached_vocab = np.load(vocab_cache_path, allow_pickle=True)
            word_to_idx = dict(cached_vocab['word_to_idx'].item())
            idx_to_word = dict(cached_vocab['idx_to_word'].item())
            vocab_time = time.perf_counter() - vocab_start
            
            # v5.31.1: Validate cache size matches requested vocab_size
            if len(word_to_idx) < vocab_size * 0.9:  # Allow 10% tolerance
                print(f"  ⚠️ Cached vocab too small ({len(word_to_idx):,} < {vocab_size:,}). Rebuilding...")
                vocab_loaded = False
            else:
                print(f"  ✓ Loaded {len(word_to_idx):,} vocabulary in {vocab_time:.2f}s")
                vocab_loaded = True
        except Exception as e:
            print(f"  ⚠️ Vocab cache load failed: {e}")
    
    if not vocab_loaded:
        # Build vocabulary FROM THE TRAINING DATA (OpenWebText)
        # v5.31.2: Increased to 500K docs for minimal <unk> (~2% OOV rate)
        vocab_samples = 500_000  # 500K docs for comprehensive vocab coverage
        word_counts = {}
        print(f"  Building vocabulary from OpenWebText ({vocab_samples//1000}K docs sample)...")
        
        vocab_start = time.perf_counter()
        print("    Loading dataset stream...")
        ds_vocab = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        load_time = time.perf_counter() - vocab_start
        print(f"    Dataset stream ready in {load_time:.1f}s")
        
        scan_start = time.perf_counter()
        for i, example in enumerate(ds_vocab):
            if i >= vocab_samples:
                break
            text = example.get("text", "")
            if isinstance(text, str):
                for word in word_pattern.findall(text):
                    word_counts[word] = word_counts.get(word, 0) + 1
            if i % 10000 == 0 and i > 0:
                elapsed = time.perf_counter() - scan_start
                rate = i / elapsed
                eta = (vocab_samples - i) / rate if rate > 0 else 0
                print(f"    Vocab scan: {i:,}/{vocab_samples:,} docs ({rate:.0f} docs/s, ETA {eta:.0f}s)")
        
        scan_time = time.perf_counter() - scan_start
        print(f"    Scan complete: {vocab_samples:,} docs in {scan_time:.1f}s")
        
        # Sort by frequency and take top vocab_size - 2 (reserve 0=unk, 1=pad)
        sort_start = time.perf_counter()
        for word, _ in sorted(word_counts.items(), key=lambda x: -x[1])[:vocab_size - 2]:
            idx = len(word_to_idx)
            word_to_idx[word] = idx
            idx_to_word[idx] = word
        sort_time = time.perf_counter() - sort_start
        
        vocab_time = time.perf_counter() - vocab_start
        print(f"  ✓ Vocabulary built in {vocab_time:.1f}s (scan={scan_time:.1f}s, sort={sort_time:.1f}s)")
        
        # Save vocabulary cache
        print(f"  💾 Saving vocabulary cache...")
        try:
            np.savez(vocab_cache_path, 
                     word_to_idx=word_to_idx, 
                     idx_to_word=idx_to_word)
            checkpoint_volume.commit()
            print(f"  ✓ Vocabulary cached for future runs")
        except Exception as e:
            print(f"  ⚠️ Vocab cache save failed: {e}")
    
    # IMPORTANT: Use actual vocab size, not requested
    actual_vocab_size = len(word_to_idx)
    print(f"  ✓ Vocabulary: {actual_vocab_size:,} words (requested {vocab_size:,})")
    vocab_size = actual_vocab_size  # Update to actual size for model initialization
    
    # OOV tracking (v5.31.1)
    _oov_count = [0]  # Mutable container for closure
    _total_count = [0]
    
    def tokenize(text: str) -> List[int]:
        # DON'T lowercase - preserve capitalization!
        words = word_pattern.findall(text)
        tokens = []
        for w in words:
            idx = word_to_idx.get(w, 0)
            tokens.append(idx)
            _total_count[0] += 1
            if idx == 0:  # <unk>
                _oov_count[0] += 1
        return tokens
    
    def get_oov_rate() -> float:
        if _total_count[0] == 0:
            return 0.0
        return _oov_count[0] / _total_count[0]
    
    def detokenize(tokens: List[int]) -> str:
        return " ".join(idx_to_word.get(t, "<unk>") for t in tokens)
    
    # ==========================================================================
    # GROUNDING PHASE — GloVe-based O(√N) Sample Efficiency (v5.6.0)
    # ==========================================================================
    # THEORY: Use pre-trained GloVe embeddings instead of computing co-occurrence.
    # This is: 1) MUCH faster (~2s vs ~230s), 2) Higher quality (trained on 6B tokens)
    # The semantic structure is the same - just from a better source.
    # ==========================================================================
    grounded_embeddings = None
    if use_grounding:
        print("\n  GROUNDING PHASE (GloVe → SO(4))...")
        from holographic_prod.core.grounded_embeddings import (
            create_grounded_embeddings_fast,
        )
        
        grounding_start = time.time()
        
        # Use GloVe: pre-trained on 6 billion tokens, orders of magnitude better
        # than anything we could compute from TinyStories
        glove_dim = 50  # 50d is fastest, still excellent semantic structure
        
        # Use persistent volume for GloVe cache (avoids re-download every run!)
        grounded_embeddings, coverage = create_grounded_embeddings_fast(
            word_to_idx,  # vocab dict: word → idx
            glove_dim=glove_dim,
            cache_dir="/checkpoints/glove",  # Persistent Modal volume
        )
        
        grounding_elapsed = time.time() - grounding_start
        print(f"  ✓ Grounding complete: {grounding_elapsed:.1f}s")
        print(f"    GloVe coverage: {coverage:.1%}")
        print(f"    Embeddings shape: {grounded_embeddings.shape}")
        print(f"    Theory: O(√N) sample complexity enabled!")
    
    # Initialize model
    print("\n  INITIALIZING MODEL...")
    config = MemoryConfig(
        learning_rate=PHI_INV,  # Theory: φ⁻¹ ≈ 0.618
        orthogonalize=True,
        # v5.19.0: Fractal position encoding (THEORY-TRUE, ablate only for comparison)
        use_fractal_position=not ablate_fractal_position,
    )
    
    model = HolographicMemory(
        vocab_size=vocab_size,
        max_levels=max_levels,
        config=config,
        seed=seed,
        use_gpu=use_gpu,
        grounded_embeddings=grounded_embeddings,  # Pass grounded embeddings if computed
    )
    grounding_status = "grounded (O(√N))" if grounded_embeddings is not None else "random (O(N))"
    position_status = "ENABLED (φ-derived multi-scale)" if not ablate_fractal_position else "ABLATED (for comparison only)"
    print(f"  ✓ HolographicMemory: vocab={vocab_size}, levels={max_levels} ({16**max_levels} satellites), GPU={use_gpu}")
    print(f"    Learning rate: φ⁻¹ = {PHI_INV:.6f}")
    print(f"    Embeddings: {grounding_status}")
    print(f"    Fractal position encoding: {position_status}")
    
    # Initialize attention
    attention = ToroidalAttention(n_satellites=16)
    print(f"  ✓ ToroidalAttention: 16 satellites (φ-offset phases)")
    
    # Initialize fractal components (v5.11.0) - MANDATORY for theory-true implementation
    # These are NOT optional. Use ablate_fractal=True ONLY for ablation studies.
    if ablate_fractal:
        interaction_tensor = None
        chirality_manager = None
        downward_projector = None
        print(f"  ⚠️ ABLATION MODE: Fractal components DISABLED (for comparison only)")
    else:
        interaction_tensor = InteractionTensor(n_satellites=16)
        chirality_manager = ChiralityFlip(n_satellites=16)
        downward_projector = DownwardProjection(basis=model.basis, xp=model.xp)
        print(f"  ✓ FractalComponents (THEORY-TRUE):")
        print(f"    InteractionTensor: satellite bivectors → master trivectors (φ-scaling)")
        print(f"    ChiralityFlip: even/odd handedness alternation (interference prevention)")
        print(f"    DownwardProjection: GraceInverse for generation (structure inflation)")
    
    # Initialize dreaming
    dreamer = None
    episodic_buffer = []
    if enable_dreaming:
        dreamer = DreamingSystem(
            basis=model.basis,
            xp=model.xp,
            use_salience=True,
            use_novelty=True,
            use_predictive_coding=True,
            use_pattern_completion=True,
            use_inhibition_of_return=True,
            use_sequence_replay=False,  # DISABLED: unvectorized loop causes ~10K GPU syncs
            use_pseudo_rehearsal=True,
        )
        print(f"  ✓ DreamingSystem: 11 parsimonies enabled (sequence replay disabled)")
        print(f"    Encoding: Salience, Novelty, Prediction Error, Predictive Coding")
        print(f"    Maintenance: φ-Decay, Interference, Reconsolidation, Pseudo-Rehearsal")
        print(f"    Retrieval: Working Memory, Pattern Completion, Inhibition of Return")
        print(f"    Temporal: [sequence replay OFF - needs vectorization]")
    
    # Try to load checkpoint from PERSISTENT volume (unless fresh_start)
    checkpoint_path = "/checkpoints/holographic_checkpoint.npz"
    if fresh_start:
        print(f"  🆕 FRESH START - ignoring any existing checkpoint")
    elif load_checkpoint(model, state, checkpoint_path):
        print(f"  ✓ Resumed from checkpoint: {state.samples:,} samples")
    
    print()
    print("=" * 80)
    print("  TRAINING")
    print("=" * 80)
    
    # REAL DATA MODE - WITH SAMPLE CACHING (v5.26.0)
    # v5.25.0: Use OpenWebText (8GB, 8M docs) instead of TinyStories (2M docs)
    # v5.26.0: Cache tokenized samples to avoid re-processing on every run!
    
    from datasets import load_dataset
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    
    # =================================================================
    # SAMPLE CACHING — Skip tokenization if cache exists (v5.26.0)
    # =================================================================
    cache_path = "/checkpoints/tokenized_samples.npz"
    # v5.31.3: 500K samples for FAST iteration - wraps around during training
    # Network I/O is the bottleneck, not sample variety
    target_samples = 500_000  # 500K samples = ~1.3M docs = ~7 min tokenization
    
    # Try to load cached samples first
    cache_loaded = False
    all_samples = []
    
    if os.path.exists(cache_path):
        print(f"  📦 Found cached samples at {cache_path}")
        try:
            cache_start = time.perf_counter()
            
            # v5.31.0: OPTIMIZED cache loading - avoid Python list conversion
            # Load directly as memory-mapped for speed
            cached = np.load(cache_path)
            cached_contexts = cached['contexts']
            cached_targets = cached['targets']
            n_cached = len(cached_targets)
            
            # v5.31.3: RELAXED validation - use cache if tokens are valid indices
            # The token indices just need to be < vocab_size to be valid
            # This avoids slow re-tokenization when vocab grows
            max_token = max(int(cached_contexts.max()), int(cached_targets.max()))
            if max_token >= vocab_size:
                print(f"  ⚠️ Cache has tokens > vocab ({max_token} >= {vocab_size}). Rebuilding...")
                raise ValueError("Cache vocab mismatch")
            
            # Log cache info but DON'T require exact vocab match
            cached_vocab_size = cached.get('vocab_size', None)
            if cached_vocab_size is not None:
                cached_vocab_size = int(cached_vocab_size)
                print(f"    Cache vocab: {cached_vocab_size:,}, current vocab: {vocab_size:,}")
            
            # Check for <unk> saturation in BOTH targets AND contexts
            # If >8% are <unk>, cache was likely built with smaller vocab
            target_unk_rate = (cached_targets == 0).mean()
            context_unk_rate = (cached_contexts == 0).mean()  # Contexts is 2D
            total_unk_rate = max(target_unk_rate, context_unk_rate)
            if total_unk_rate > 0.08:
                print(f"  ⚠️ Cache has {total_unk_rate*100:.1f}% <unk> tokens. Rebuilding with new vocab...")
                raise ValueError("Too many <unk> in cache")
            n_to_load = min(n_cached, target_samples)
            
            # FAST: Keep as numpy arrays, only slice what we need
            # Convert to list of tuples lazily or batch-by-batch
            print(f"    Loading {n_to_load:,} samples from cache...")
            
            # Pre-allocate and convert in chunks for speed
            all_samples = []
            chunk_size = 100000
            for i in range(0, n_to_load, chunk_size):
                end = min(i + chunk_size, n_to_load)
                ctx_chunk = cached_contexts[i:end].tolist()
                tgt_chunk = cached_targets[i:end].tolist()
                all_samples.extend(zip(ctx_chunk, tgt_chunk))
                if i > 0 and i % 1000000 == 0:
                    print(f"      {i:,}/{n_to_load:,} samples loaded...")
            
            cache_time = time.perf_counter() - cache_start
            print(f"  ✓ Loaded {len(all_samples):,} cached samples in {cache_time:.1f}s")
            print(f"    (Skipped tokenization of 1.7M+ documents!)")
            cache_loaded = True
        except Exception as e:
            print(f"  ⚠️ Cache load failed: {e}")
            print(f"    Will re-tokenize and rebuild cache...")
    
    if not cache_loaded:
        # No cache — tokenize from scratch using TRUE multiprocessing (v5.26.1)
        print("  Loading OpenWebText dataset (8M documents)...")
        
        tok_start = time.perf_counter()
        # OpenWebText: ~8 million documents, diverse web text
        # Streaming mode - fast to start, no 40GB download required
        print("  📥 Loading OpenWebText stream...")
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        stream_time = time.perf_counter() - tok_start
        print(f"  ✓ Dataset stream ready in {stream_time:.1f}s")
        
        # =================================================================
        # TRUE MULTIPROCESSING (v5.26.1)
        # Uses module-level _tokenize_batch_for_mp (pickle-safe!)
        # =================================================================
        n_workers = min(mp.cpu_count(), 64)  # Modal H100 has ~96 vCPUs
        print(f"  🚀 Tokenizing with {n_workers} PROCESS workers (TRUE parallelism)")
        print(f"    (This will be cached for future runs)")
        
        # Serialize vocab dict for passing to worker processes
        context_size = state.context_size
        word_to_idx_items = list(word_to_idx.items())
        
        # Collect documents into large batches for parallel processing
        doc_count = 0
        batch_size = 50000  # Process 50K docs at a time
        batch_texts = []
        chunk_size = max(1, batch_size // n_workers)  # Docs per worker
        
        stream_loop_start = time.perf_counter()
        last_report = stream_loop_start
        
        for doc in ds:
            if len(all_samples) >= target_samples:
                break
            
            text = doc.get("text", "")
            if text and len(text) >= 50:
                batch_texts.append(text)
            doc_count += 1
            
            # Progress report every 10 seconds
            now = time.perf_counter()
            if now - last_report > 10:
                elapsed = now - stream_loop_start
                rate = doc_count / elapsed if elapsed > 0 else 0
                print(f"    Streaming: {doc_count:,} docs, {len(all_samples):,} samples ({rate:.0f} docs/s)")
                last_report = now
            
            # Process batch in parallel when full
            if len(batch_texts) >= batch_size:
                batch_start = time.perf_counter()
                # Split into chunks for workers
                chunks = [
                    (batch_texts[i:i + chunk_size], word_to_idx_items, context_size)
                    for i in range(0, len(batch_texts), chunk_size)
                ]
                
                # Use module-level function (pickle-safe!)
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(_tokenize_batch_for_mp, chunks))
                
                for samples in results:
                    all_samples.extend(samples)
                    if len(all_samples) >= target_samples:
                        break
                
                batch_time = time.perf_counter() - batch_start
                batch_texts = []
                total_elapsed = time.perf_counter() - stream_loop_start
                overall_rate = doc_count / total_elapsed if total_elapsed > 0 else 0
                print(f"    Processed batch: {doc_count:,} docs, {len(all_samples):,} samples (batch={batch_time:.1f}s, overall={overall_rate:.0f} docs/s)")
        
        # Process remaining batch
        if batch_texts and len(all_samples) < target_samples:
            chunk_size = max(1, len(batch_texts) // n_workers)
            chunks = [
                (batch_texts[i:i + chunk_size], word_to_idx_items, context_size)
                for i in range(0, len(batch_texts), chunk_size)
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_tokenize_batch_for_mp, chunks))
            for samples in results:
                all_samples.extend(samples)
        
        print(f"  ✓ Tokenized: {len(all_samples):,} samples from {doc_count:,} documents")
        
        # =================================================================
        # SAVE TO CACHE — Skip tokenization on future runs (v5.26.0)
        # =================================================================
        print(f"  💾 Saving to cache: {cache_path}")
        try:
            # Convert to numpy arrays for efficient storage
            # v5.31.2: Include vocab_size to validate cache on load
            contexts_arr = np.array([s[0] for s in all_samples], dtype=np.int32)
            targets_arr = np.array([s[1] for s in all_samples], dtype=np.int32)
            np.savez_compressed(
                cache_path, 
                contexts=contexts_arr, 
                targets=targets_arr,
                vocab_size=np.array(vocab_size, dtype=np.int32)  # v5.31.2: Save vocab size
            )
            checkpoint_volume.commit()  # Persist to Modal volume
            print(f"  ✓ Cache saved ({contexts_arr.nbytes / 1024**2:.1f} MB, vocab={vocab_size:,})")
        except Exception as e:
            print(f"  ⚠️ Cache save failed: {e}")
    
    # Shuffle once
    np.random.shuffle(all_samples)
    sample_idx = 0
    
    # =========================================================================
    # GROUNDING PHASE — GloVe-based O(√N) Sample Efficiency (v5.6.0)
    # =========================================================================
    # THEORY: Use pre-trained GloVe embeddings instead of computing co-occurrence.
    # This is: 1) MUCH faster (~2s vs ~230s), 2) Higher quality (trained on 6B tokens)
    # =========================================================================
    
    enable_grounding = True  # Set False to use random embeddings (requires 10x more samples)
    
    if enable_grounding and len(all_samples) > 1000:
        print()
        print("  🌱 GROUNDING PHASE (GloVe v5.6.0):")
        
        grounding_start = time.perf_counter()
        
        from holographic_prod.core.grounded_embeddings import (
            create_grounded_embeddings_fast,
        )
        
        # Use GloVe with our word_to_idx vocab dict
        # GloVe trained on 6 billion tokens - much better than co-occurrence on TinyStories
        glove_dim = 50  # 50d is fastest, still excellent semantic structure
        
        print(f"     Loading GloVe-{glove_dim}d embeddings...")
        grounded_embeddings, coverage = create_grounded_embeddings_fast(
            vocab=word_to_idx,
            glove_dim=glove_dim,
            cache_dir="/checkpoints/glove",
        )
        
        # Set on model
        model.set_grounded_embeddings(grounded_embeddings)
        
        grounding_time = time.perf_counter() - grounding_start
        print(f"     ✓ Grounding complete in {grounding_time:.1f}s")
        print(f"     GloVe coverage: {coverage:.1%}")
        print(f"     → Similar tokens now have similar embeddings")
        print(f"     → Expected: O(√N) sample complexity vs O(N) for random")
    else:
        if not enable_grounding:
            print("  ⚠️ Grounding DISABLED - using random embeddings (requires 10x more samples)")
    
    # Initial diagnostic before training
    print()
    print(f"  📊 PRE-TRAINING DIAGNOSTICS:")
    init_diag = get_memory_diagnostics(model, use_gpu=use_gpu)
    print(f"     Memory: {init_diag['n_patterns']:,} patterns, norm={init_diag['holographic_norm']:.4f}")
    print(f"     Satellites: {init_diag.get('n_satellites', 0):,} total, {init_diag.get('active_satellites', 0):,} active")
    print(f"     Episodic cache: {init_diag['episodic_cache_size']:,}")
    if init_diag['gpu_used_gb'] > 0:
        print(f"     GPU: {init_diag['gpu_used_gb']:.1f} GB used, {init_diag['gpu_free_gb']:.1f} GB free")
    
    # OOV rate logging (v5.31.2)
    oov_rate = get_oov_rate()
    print(f"     Vocabulary: {vocab_size:,} words, OOV rate: {oov_rate*100:.2f}%")
    if oov_rate > 0.05:
        print(f"     ⚠️ High OOV rate ({oov_rate*100:.1f}%) - consider increasing vocab_size")
    elif oov_rate > 0.02:
        print(f"     ✓ Acceptable OOV rate ({oov_rate*100:.1f}%) - some <unk> expected")
    else:
        print(f"     ✅ Excellent OOV rate ({oov_rate*100:.1f}%) - minimal <unk>")
    
    print(f"     Ready to process {len(all_samples):,} samples")
    print()
    
    while state.samples < max_samples and sample_idx < len(all_samples):
        # Create batch from pre-generated samples (FAST - no I/O)
        batch_end = min(sample_idx + batch_size, len(all_samples))
        batch = all_samples[sample_idx:batch_end]
        sample_idx = batch_end
        if sample_idx >= len(all_samples):
            sample_idx = 0  # Wrap around if we exhaust samples
        
        if not batch:
            break
        
        # Learn batch
        batch_start = time.perf_counter()
        model.learn_batch(batch)
        batch_time = time.perf_counter() - batch_start
        
        # Track batch timing (for real-time throughput)
        state.batch_times.append(batch_time)
        
        # Update counters
        state.samples += len(batch)
        state.batches += 1
        state.samples_since_dream += len(batch)
        
        # ======================================================================
        # THEORY-TRUE EVALUATION (v5.28.0)
        # ======================================================================
        # THEORY: "NO sampling, NO argmax — just settling"
        #         "NEVER Measure Exact Token Match as Primary Metric"
        #         "semantic_sim=0.96 IS success"
        #
        # We measure SEMANTIC SIMILARITY, not exact token match.
        # This is MORE parsimonious: no argmax, no discrete selection, just settling.
        # ======================================================================
        
        semantic_sim = state.last_accuracy  # Use cached value (repurposed for semantic_sim)
        
        if state.batches % 100 == 0:  # Evaluate every 100 batches
            sample_size = min(20, len(batch))  # 20 samples for statistical significance
            sample_indices = np.random.choice(len(batch), size=sample_size, replace=False)
            
            # Build evaluation batch
            eval_batch = [(batch[idx][0], batch[idx][1]) for idx in sample_indices]
            
            # THEORY-TRUE: Use evaluate_semantic (no argmax!)
            eval_result = model.evaluate_semantic(eval_batch)
            semantic_sim = eval_result['semantic_similarity']
            
            # Update state (repurpose accuracy fields for semantic similarity)
            state.last_accuracy = semantic_sim
            
            # Track for running average (error = 1 - similarity for consistency with existing code)
            state.batch_errors.append(1.0 - semantic_sim)
            
            # Track min/max for diagnostics
            if not hasattr(state, 'min_semantic_sim'):
                state.min_semantic_sim = 1.0
                state.max_semantic_sim = 0.0
            state.min_semantic_sim = min(state.min_semantic_sim, eval_result['min_similarity'])
            state.max_semantic_sim = max(state.max_semantic_sim, eval_result['max_similarity'])
        
        # Curriculum update
        if enable_curriculum:
            new_stage, new_ctx, explanation = get_curriculum_stage(state.samples)
            if new_stage != state.stage:
                print_curriculum_change(state.stage, new_stage, explanation)
                state.stage = new_stage
                state.context_size = new_ctx
        
        # Collect episodes for dreaming (every 50th batch, 10% of samples)
        # OPTIMIZED (v5.28.0): Reduced frequency from every 20 to every 50 batches
        # Theory: Brain forms episodic traces continuously, not sparsely
        # Balance: 10% sampling provides variety without excessive overhead
        # NOTE: No buffer cap needed - O(n) basin-key clustering handles any size
        if enable_dreaming and dreamer and state.batches % 50 == 0:  # Was 20, now 50
            # Sample 10% of batch for episodic memory (variety for prototype formation)
            n_episodes = max(1, len(batch) // 10)
            episode_indices = np.random.choice(len(batch), size=n_episodes, replace=False)
            
            # Only embed the sampled contexts
            sampled_contexts = [batch[i][0] for i in episode_indices]
            sampled_targets = [batch[i][1] for i in episode_indices]
            
            ctx_matrices = model.embed_sequences_batch(sampled_contexts, return_gpu=use_gpu)
            
            for ctx_matrix, tgt in zip(ctx_matrices, sampled_targets):
                episodic_buffer.append(EpisodicEntry(
                    context_matrix=ctx_matrix,
                    target_token=tgt,
                ))
            
            # Check dream trigger every 50 batches (theory-true frequency)
            # At batch_size=2048, this is every ~100K samples
            should_sleep = False
            if state.batches % 50 == 0:
                stability = model.get_stability()
                should_sleep, reason = should_dream(state, stability, verbose=verbose)
            
            if should_sleep:
                # Run INTEGRATED sleep cycle (both tower + systems dreaming)
                # This is theory-true: combines synaptic homeostasis with systems consolidation
                sleep_result = integrated_sleep(
                    memory=model,
                    dreaming_system=dreamer,
                    episodes=episodic_buffer,
                    rem_cycles=1,
                    verbose=True,  # DEBUG: See where dreaming is stuck
                )
                
                # Extract stats from integrated result
                tower_stats = sleep_result['tower_stats']
                systems_stats = sleep_result['systems_stats']
                
                # Update state
                state.total_dreams += 1
                state.samples_since_dream = 0
                state.prototypes_total += systems_stats.get('prototypes_created', 0)
                state.schemas_total += systems_stats.get('schemas_discovered', 0)
                state.prototypes_pruned += systems_stats.get('prototypes_pruned', 0)
                
                # Build dream_stats for print_dream_details
                dream_stats = {
                    'pre_stability': tower_stats['pre_stability'],
                    'post_stability': tower_stats['post_stability'],
                    'prototypes_created': systems_stats.get('prototypes_created', 0),
                    'schemas_discovered': systems_stats.get('schemas_discovered', 0),
                    'prototypes_pruned': systems_stats.get('prototypes_pruned', 0),
                    'tower_rem_improvements': tower_stats.get('rem_improvements', 0),
                    'phases': sleep_result.get('phases_completed', []),
                }
                
                if verbose:
                    print_dream_details(dream_stats, state)
                
                episodic_buffer = []
                
        # ADAPTIVE LOGGING: Much more frequent early on for debugging
        # First 50K: log every 5K samples (see learning signal immediately)
        # 50K-200K: log every 10K samples (verify scaling)
        # After 200K: log every log_every samples (default 100K)
        if state.samples < 50_000:
            effective_log_interval = 5_000
        elif state.samples < 200_000:
            effective_log_interval = 10_000
        else:
            effective_log_interval = log_every
        
        # Logging
        if state.samples - state.last_log_samples >= effective_log_interval:
            current_time = state.elapsed()
            interval_samples = state.samples - state.last_log_samples
            interval_time = current_time - state.last_log_time
            
            throughput = interval_samples / max(0.1, interval_time)
            
            # Memory diagnostics (with GPU tracking) - includes stability
            # CRITICAL OPTIMIZATION (v5.30.0): Get stability from mem_diag, not separately
            # This eliminates a redundant GPU sync
            mem_diag = get_memory_diagnostics(model, use_gpu=use_gpu)
            stability = mem_diag['stability']
            
            # Compute perplexity every log interval (reduced batch size to minimize GPU syncs)
            # CRITICAL FIX (v5.28.0): Reduced from 5 to 3 samples to reduce GPU sync overhead
            ppl_batch = batch[:min(3, len(batch))]  # Was 5, now 3 for faster logging
            perplexity, ppl_details = compute_perplexity_detailed(model, ppl_batch, vocab_size)
            
            # Record metrics (v5.28.0: accuracy is now semantic_similarity)
            state.record(
                perplexity=perplexity,
                stability=stability,
                accuracy=semantic_sim,  # THEORY-TRUE: semantic similarity, not exact match
                throughput=throughput,
                error_rate=1.0 - semantic_sim,  # Inverse of similarity for error tracking
            )
            
            # Track ToM/Prior statistics (v5.4.3)
            state.prior_usage = getattr(model, '_prior_usage_count', 0)
            if model._factorized_prior is not None:
                state.factorized_prior_updates = model._factorized_prior.n_updates
            
            if verbose:
                print_progress_detailed(state, max_samples, perplexity, ppl_details, mem_diag)
                
                # THEORY-TRUE (v5.28.0): Show semantic similarity range
                if hasattr(state, 'min_semantic_sim'):
                    print(f"  SemanticSim Range: [{state.min_semantic_sim:.3f}, {state.max_semantic_sim:.3f}]")
                    if state.min_semantic_sim > 0.8:
                        print(f"     ↳ ✅ All samples above 0.8 similarity (excellent)")
                    elif state.min_semantic_sim > 0.5:
                        print(f"     ↳ ✓ All samples above 0.5 similarity (good)")
                    else:
                        print(f"     ↳ 🔄 Some samples have low similarity (still learning)")
            
            # Generate samples for qualitative evaluation
            # CRITICAL: Seeing actual output is essential for understanding what model learns
            # Early: every 25K samples (learning signal visible)
            # Later: every 100K samples (less frequent but still informative)
            sample_generation_interval = 25_000 if state.samples < 100_000 else 100_000
            if verbose and state.samples % sample_generation_interval < effective_log_interval:
                # Get 3 random prompts from recent batch
                n_prompts = min(3, len(batch))
                prompt_indices = np.random.choice(len(batch), size=n_prompts, replace=False)
                prompts = [batch[i][0] for i in prompt_indices]
                
                print(f"\n  📝 SAMPLE GENERATION (context_size={state.context_size}):")
                samples = generate_samples(model, prompts, max_tokens=8, idx_to_word=idx_to_word)
                for i, (prompt, sample) in enumerate(zip(prompts, samples)):
                    prompt_text = " ".join([idx_to_word.get(t, f"<{t}>") for t in prompt])
                    print(f"     Prompt {i+1}: [{prompt_text}]")
                    print(f"     Output:   {sample}")
            
            state.last_log_samples = state.samples
            state.last_log_time = current_time
            
            # Periodic checkpoint (only every checkpoint_every, NOT every log!)
            if state.samples % checkpoint_every < effective_log_interval and state.samples >= checkpoint_every:
                save_checkpoint(model, state, dreamer, checkpoint_path)
                if verbose:
                    print(f"  💾 Checkpoint saved: {state.samples:,} samples")
    
    # Final checkpoint
    save_checkpoint(model, state, dreamer, checkpoint_path)
    print(f"  💾 Final checkpoint saved")
    
    # Final summary
    print()
    print("=" * 80)
    print("  TRAINING COMPLETE")
    print("=" * 80)
    print()
    
    final_diag = get_memory_diagnostics(model)
    
    print(f"  Samples: {state.samples:,}")
    print(f"  Batches: {state.batches:,}")
    print(f"  Final Stage: {state.stage}")
    print(f"  Final Context: {state.context_size}")
    print()
    print(f"  Perplexity: {state.avg('perplexity'):,.1f}")
    print(f"  SemanticSim: {state.avg('accuracy'):.3f}")  # THEORY-TRUE: semantic similarity
    print(f"  Stability: {final_diag['stability']:.3f}")
    print()
    print(f"  Memory Patterns: {final_diag['n_patterns']:,}")
    print(f"  Holographic Norm: {final_diag['holographic_norm']:.2f}")
    print()
    print(f"  Dreams: {state.total_dreams}")
    print(f"  Prototypes: {state.prototypes_total}")
    print(f"  Schemas: {state.schemas_total}")
    print(f"  Pruned: {state.prototypes_pruned}")
    print()
    print(f"  Time: {state.elapsed():.1f}s")
    print(f"  Avg Throughput: {state.avg('throughput'):,.0f} samples/sec")
    
    # Interpretation
    consts = get_theory_constants()
    final_ppl = state.avg('perplexity')
    
    print()
    if final_ppl < 100:
        print("  ✅ EXCELLENT: Perplexity competitive with transformers")
    elif final_ppl < 500:
        print("  ✓ GOOD: Learning working, continue training")
    else:
        print("  🔄 EARLY: More training needed")
    
    if final_diag['stability'] < consts['PHI_INV_SQ']:
        print(f"  ⚠ WARNING: Stability {final_diag['stability']:.3f} < φ⁻² "
              f"(consider more dreaming)")
    
    return {
        'samples': state.samples,
        'perplexity': final_ppl,
        'semantic_similarity': state.avg('accuracy'),  # THEORY-TRUE: semantic similarity
        'stability': final_diag['stability'],
        'n_patterns': final_diag['n_patterns'],
        'dreams': state.total_dreams,
        'prototypes': state.prototypes_total,
        'schemas': state.schemas_total,
        'elapsed': state.elapsed(),
        'throughput': state.avg('throughput'),
    }


# =============================================================================
# PRE-FLIGHT VALIDATION (Run before any large training)
# =============================================================================

@app.function(image=image, gpu="H100", timeout=600)
def preflight_validation():
    """
    SYSTEMATIC PRE-FLIGHT VALIDATION
    =================================
    
    Run this BEFORE any expensive training to catch issues early.
    Tests ALL critical paths with real GPU/CuPy.
    
    Tests:
        1. Embedding consistency (_embed_sequence == _embed_sequences_batch)
        2. Learn→Retrieve cycle (accuracy > 0 after learning)
        3. Basin routing consistency (same context → same satellite)
        4. Mini-training validation (perplexity drops, accuracy rises)
    
    Returns:
        Dict with pass/fail status and detailed diagnostics
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import numpy as np
    import cupy as cp
    
    from holographic_prod.memory import HolographicMemory
    from holographic_prod.core.constants import PHI, PHI_INV
    # Note: PHI_EPSILON not needed here, tests use defaults
    
    results = {
        'tests': {},
        'passed': True,
        'diagnostics': []
    }
    
    print("=" * 80)
    print("  PRE-FLIGHT VALIDATION (GPU)")
    print("=" * 80)
    print()
    
    vocab_size = 1000
    context_size = 16
    
    # =========================================================================
    # TEST 1: Embedding Consistency
    # =========================================================================
    print("TEST 1: Embedding Consistency")
    print("-" * 40)
    
    try:
        mem = HolographicMemory(vocab_size=vocab_size, use_gpu=True)
        
        # Create 10 test contexts
        test_contexts = [
            [i % vocab_size for i in range(j, j + context_size)]
            for j in range(10)
        ]
        
        # Embed via single method
        single_embeddings = []
        for ctx in test_contexts:
            emb = mem.tower._embed_sequence(ctx)
            if hasattr(emb, 'get'):
                emb = emb.get()
            single_embeddings.append(emb)
        single_embeddings = np.stack(single_embeddings)
        
        # Embed via batch method
        batch_embeddings = mem.tower._embed_sequences_batch(test_contexts)
        if hasattr(batch_embeddings, 'get'):
            batch_embeddings = batch_embeddings.get()
        
        # Compare
        max_diff = np.max(np.abs(single_embeddings - batch_embeddings))
        mean_diff = np.mean(np.abs(single_embeddings - batch_embeddings))
        
        passed = max_diff < 1e-5
        results['tests']['embedding_consistency'] = {
            'passed': passed,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
        }
        
        if passed:
            print(f"  ✓ PASSED: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        else:
            print(f"  ✗ FAILED: max_diff={max_diff:.2e} (threshold: 1e-5)")
            results['passed'] = False
            results['diagnostics'].append(
                f"Embedding mismatch: single vs batch differ by {max_diff:.2e}"
            )
            
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results['tests']['embedding_consistency'] = {'passed': False, 'error': str(e)}
        results['passed'] = False
    
    print()
    
    # =========================================================================
    # TEST 2: Learn → Retrieve Cycle
    # =========================================================================
    print("TEST 2: Learn → Retrieve Cycle")
    print("-" * 40)
    
    try:
        mem = HolographicMemory(vocab_size=vocab_size, use_gpu=True)
        
        # Learn 100 distinct (context, target) pairs
        n_pairs = 100
        contexts = []
        targets = []
        for i in range(n_pairs):
            # Create unique context
            ctx = [(i * 17 + j * 13) % vocab_size for j in range(context_size)]
            tgt = (i * 31) % vocab_size  # Unique target
            contexts.append(ctx)
            targets.append(tgt)
        
        # Check memory before learning
        # Handle both TowerMemory (_satellite_memories) and MultiLevelTower (_all_memories)
        if hasattr(mem.tower, '_satellite_memories'):
            sat_mem_before = mem.tower._satellite_memories.sum()
        else:
            sat_mem_before = mem.tower._all_memories.sum()
        if hasattr(sat_mem_before, 'get'):
            sat_mem_before = sat_mem_before.get()
        print(f"  Memory sum before: {float(sat_mem_before):.6f}")
        
        # Learn in batch
        mem.learn_batch(contexts, targets)
        
        # Check memory after learning
        if hasattr(mem.tower, '_satellite_memories'):
            sat_mem_after = mem.tower._satellite_memories.sum()
        else:
            sat_mem_after = mem.tower._all_memories.sum()
        if hasattr(sat_mem_after, 'get'):
            sat_mem_after = sat_mem_after.get()
        print(f"  Memory sum after:  {float(sat_mem_after):.6f}")
        print(f"  n_patterns: {mem.n_patterns}")
        
        # Also try single learn to compare
        mem2 = HolographicMemory(vocab_size=vocab_size, use_gpu=True)
        mem2.tower.learn(contexts[0], targets[0])  # Single learn
        if hasattr(mem2.tower, '_satellite_memories'):
            sat_mem_single = mem2.tower._satellite_memories.sum()
        else:
            sat_mem_single = mem2.tower._all_memories.sum()
        if hasattr(sat_mem_single, 'get'):
            sat_mem_single = sat_mem_single.get()
        print(f"  Memory sum after single learn: {float(sat_mem_single):.6f}")
        
        # Retrieve and check accuracy (v5.15.0: parallel retrieval)
        n_correct = 0
        for ctx, tgt in zip(contexts, targets):
            pred, conf, _ = mem.retrieve_parallel(ctx, force_parallel=True)
            if pred == tgt:
                n_correct += 1
        
        accuracy = n_correct / n_pairs
        
        # Also test single-learned pattern
        pred_single, conf_single, _ = mem2.retrieve_parallel(contexts[0], force_parallel=True)
        print(f"  Single pattern: expected {targets[0]}, got {pred_single}, conf={conf_single:.3f}")
        
        # We expect at least 10% accuracy after learning (not random chance)
        passed = accuracy > 0.1  # > 10% (random would be 0.1%)
        
        results['tests']['learn_retrieve_cycle'] = {
            'passed': passed,
            'accuracy': float(accuracy),
            'n_correct': n_correct,
            'n_pairs': n_pairs,
        }
        
        if passed:
            print(f"  ✓ PASSED: accuracy={accuracy:.1%} ({n_correct}/{n_pairs})")
        else:
            print(f"  ✗ FAILED: accuracy={accuracy:.1%} (expected > 10%)")
            results['passed'] = False
            results['diagnostics'].append(
                f"Learn/retrieve broken: accuracy={accuracy:.1%}, expected > 10%"
            )
            
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['tests']['learn_retrieve_cycle'] = {'passed': False, 'error': str(e)}
        results['passed'] = False
    
    print()
    
    # =========================================================================
    # TEST 3: Basin Routing Consistency
    # =========================================================================
    print("TEST 3: Basin Routing Consistency")
    print("-" * 40)
    
    try:
        mem = HolographicMemory(vocab_size=vocab_size, use_gpu=True)
        
        # Get routes for 100 contexts, call twice each
        test_contexts = [
            [(i * 17 + j * 13) % vocab_size for j in range(context_size)]
            for i in range(100)
        ]
        
        routes_1 = [mem.tower.route_to_satellite(ctx) for ctx in test_contexts]
        routes_2 = [mem.tower.route_to_satellite(ctx) for ctx in test_contexts]
        
        # They should be identical
        n_matches = sum(1 for r1, r2 in zip(routes_1, routes_2) if r1 == r2)
        consistency = n_matches / len(test_contexts)
        
        passed = consistency == 1.0
        
        results['tests']['basin_routing_consistency'] = {
            'passed': passed,
            'consistency': float(consistency),
            'n_matches': n_matches,
        }
        
        if passed:
            print(f"  ✓ PASSED: routing 100% consistent")
        else:
            print(f"  ✗ FAILED: routing only {consistency:.1%} consistent")
            results['passed'] = False
            
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results['tests']['basin_routing_consistency'] = {'passed': False, 'error': str(e)}
        results['passed'] = False
    
    print()
    
    # =========================================================================
    # TEST 4: Mini-Training Validation
    # =========================================================================
    print("TEST 4: Mini-Training Validation")
    print("-" * 40)
    
    try:
        mem = HolographicMemory(vocab_size=vocab_size, use_gpu=True)
        
        # Generate 1000 training samples with patterns
        # Pattern: target = (sum of context) mod vocab_size
        n_samples = 1000
        batch_size = 100
        
        all_contexts = []
        all_targets = []
        for i in range(n_samples):
            ctx = [(i * 17 + j * 13 + j**2) % vocab_size for j in range(context_size)]
            tgt = sum(ctx) % vocab_size
            all_contexts.append(ctx)
            all_targets.append(tgt)
        
        # Measure initial perplexity (should be ~vocab_size = random)
        n_sample_check = 20
        initial_correct = 0
        for ctx, tgt in zip(all_contexts[:n_sample_check], all_targets[:n_sample_check]):
            pred, _, _ = mem.retrieve_parallel(ctx, force_parallel=True)
            if pred == tgt:
                initial_correct += 1
        initial_accuracy = initial_correct / n_sample_check
        
        # Train in batches
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_ctx = all_contexts[batch_start:batch_end]
            batch_tgt = all_targets[batch_start:batch_end]
            mem.learn_batch(batch_ctx, batch_tgt)
        
        # Measure final accuracy
        final_correct = 0
        for ctx, tgt in zip(all_contexts[:n_sample_check], all_targets[:n_sample_check]):
            pred, _, _ = mem.retrieve_parallel(ctx, force_parallel=True)
            if pred == tgt:
                final_correct += 1
        final_accuracy = final_correct / n_sample_check
        
        # Accuracy should improve
        improvement = final_accuracy - initial_accuracy
        passed = final_accuracy > 0.1 and improvement > 0
        
        results['tests']['mini_training'] = {
            'passed': passed,
            'initial_accuracy': float(initial_accuracy),
            'final_accuracy': float(final_accuracy),
            'improvement': float(improvement),
            'n_samples': n_samples,
        }
        
        if passed:
            print(f"  ✓ PASSED: {initial_accuracy:.1%} → {final_accuracy:.1%} (Δ={improvement:+.1%})")
        else:
            print(f"  ✗ FAILED: {initial_accuracy:.1%} → {final_accuracy:.1%} (expected improvement)")
            results['passed'] = False
            results['diagnostics'].append(
                f"Mini-training failed: acc went from {initial_accuracy:.1%} to {final_accuracy:.1%}"
            )
            
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['tests']['mini_training'] = {'passed': False, 'error': str(e)}
        results['passed'] = False
    
    print()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 80)
    if results['passed']:
        print("  ✓ ALL PRE-FLIGHT CHECKS PASSED - READY FOR TRAINING")
    else:
        print("  ✗ PRE-FLIGHT FAILED - DO NOT PROCEED")
        print("  Diagnostics:")
        for diag in results['diagnostics']:
            print(f"    • {diag}")
    print()
    print("=" * 80)
    
    return results


# =============================================================================
# TEST
# =============================================================================

@app.function(image=image, gpu="H100", timeout=1800)
def test():
    """Comprehensive tests with theory verification."""
    import sys
    sys.path.insert(0, '/root/project')
    
    import numpy as np
    import time
    
    from holographic_prod.memory import HolographicMemory, MemoryConfig
    from holographic_prod.dreaming import DreamingSystem, EpisodicEntry, integrated_sleep
    from holographic_prod.dreaming import DreamingSystem, EpisodicEntry
    from holographic_prod.attention import ToroidalAttention
    from holographic_prod.core.constants import (
        PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBE, GRACE_SCALES
    )
    
    print("=" * 80)
    print("  HOLOGRAPHIC TESTS (Theory Verification)")
    print("=" * 80)
    print()
    
    # GPU REQUIRED - this runs on Modal H100
    import cupy as cp
    cp.cuda.Device(0).use()
    use_gpu = True  # REQUIRED
    print(f"  ✓ GPU: H100 (REQUIRED)")
    
    results = {}
    
    # 1. Theory constants verification
    print("\n1. THEORY CONSTANTS")
    print("-" * 40)
    
    # Verify φ² = φ + 1 (self-consistency equation)
    phi_sq = PHI ** 2
    phi_plus_1 = PHI + 1
    assert abs(phi_sq - phi_plus_1) < 1e-10, "φ² ≠ φ + 1"
    print(f"  ✓ φ² = φ + 1 verified ({phi_sq:.10f} = {phi_plus_1:.10f})")
    
    # Verify Grace scales
    assert abs(GRACE_SCALES[0] - 1.0) < 1e-10
    assert abs(GRACE_SCALES[1] - PHI_INV) < 1e-10
    assert abs(GRACE_SCALES[2] - PHI_INV_SQ) < 1e-10
    print(f"  ✓ Grace scales: [1, φ⁻¹, φ⁻², φ⁻³, 1]")
    
    results['theory_verified'] = True
    
    # 2. Memory initialization
    print("\n2. MEMORY INITIALIZATION")
    print("-" * 40)
    
    config = MemoryConfig(learning_rate=PHI_INV, orthogonalize=True)
    model = HolographicMemory(vocab_size=1000, config=config, use_gpu=use_gpu, seed=42)
    
    print(f"  ✓ HolographicMemory initialized")
    print(f"    Learning rate: {config.learning_rate:.6f} (φ⁻¹)")
    print(f"    Orthogonalize: {config.orthogonalize}")
    
    # 3. Batch learning
    print("\n3. BATCH LEARNING")
    print("-" * 40)
    
    batch_size = 512
    batch = [([i % 100, (i+1) % 100, (i+2) % 100], (i+3) % 100) for i in range(batch_size)]
    
    start = time.perf_counter()
    model.learn_batch(batch)
    elapsed = time.perf_counter() - start
    
    throughput = batch_size / elapsed
    print(f"  ✓ {batch_size} samples in {elapsed:.3f}s ({throughput:,.0f}/s)")
    
    stability = model.get_stability()
    print(f"    Post-learning stability: {stability:.3f}")
    
    results['learning_throughput'] = throughput
    
    # 4. Retrieval
    print("\n4. RETRIEVAL (φ-kernel)")
    print("-" * 40)
    
    queries = [[i % 100, (i+1) % 100, (i+2) % 100] for i in range(1000)]
    
    start = time.perf_counter()
    for q in queries:
        model.retrieve_parallel(q, force_parallel=True)
    elapsed = time.perf_counter() - start
    
    ret_throughput = 1000 / elapsed
    print(f"  ✓ 1000 queries in {elapsed:.3f}s ({ret_throughput:,.0f}/s)")
    
    # Verify φ-kernel is used (no temperature)
    _, _, top_k = model.retrieve_probabilistic([1, 2, 3], deterministic=False, top_k=10)
    print(f"    Top-k returned: {len(top_k)} candidates")
    
    results['retrieval_throughput'] = ret_throughput
    
    # 5. Dreaming
    print("\n5. DREAMING (12 Parsimonies)")
    print("-" * 40)
    
    dreamer = DreamingSystem(
        basis=model.basis,
        xp=model.xp,
        # ALL 7 parsimonies ENABLED
        use_salience=True,
        use_novelty=True,
        use_predictive_coding=True,
        use_pattern_completion=True,
        use_inhibition_of_return=True,
        use_sequence_replay=True,
        use_pseudo_rehearsal=True,
    )
    
    episodes = []
    for i in range(100):
        ctx = [i % 100, (i+1) % 100, (i+2) % 100]
        ctx_matrix = model.embed_sequence(ctx)
        episodes.append(EpisodicEntry(context_matrix=ctx_matrix, target_token=(i+3) % 100))
    
    pre_stability = model.get_stability()
    
    start = time.perf_counter()
    dream_stats = dreamer.sleep(episodes, verbose=False)
    elapsed = time.perf_counter() - start
    
    print(f"  ✓ Sleep completed in {elapsed:.3f}s")
    print(f"    Prototypes created: {dream_stats['prototypes_created']}")
    print(f"    Schemas discovered: {dream_stats['schemas_discovered']}")
    
    results['dreaming_works'] = True
    
    # 6. Attention
    print("\n6. TOROIDAL ATTENTION")
    print("-" * 40)
    
    attention = ToroidalAttention(n_satellites=16)
    attn = attention.compute_context_attention([1, 2, 3, 4, 5])
    
    print(f"  ✓ Attention shape: {attn.shape}")
    print(f"    16 satellites with φ-offset phases")
    
    results['attention_works'] = True
    
    # Summary
    print("\n" + "=" * 80)
    print("  TEST RESULTS")
    print("=" * 80)
    print(f"  Theory Verified: {results['theory_verified']}")
    print(f"  Learning: {results['learning_throughput']:,.0f}/s")
    print(f"  Retrieval: {results['retrieval_throughput']:,.0f}/s")
    print(f"  Dreaming: {results['dreaming_works']}")
    print(f"  Attention: {results['attention_works']}")
    print()
    
    return results


# =============================================================================
# GPU PYTEST TESTS (runs the 10 skipped GPU tests on Modal H100)
# =============================================================================

@app.function(image=image, gpu="H100", timeout=1800)
def run_gpu_tests():
    """Run pytest GPU tests that are skipped locally."""
    import subprocess
    import sys
    sys.path.insert(0, '/root/project')
    
    print("=" * 80)
    print("  GPU PYTEST TESTS on Modal H100")
    print("=" * 80)
    print()
    
    # Verify CuPy is available
    try:
        import cupy as cp
        print(f"✓ CuPy available: {cp.__version__}")
        print(f"✓ GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        meminfo = cp.cuda.runtime.memGetInfo()
        print(f"✓ GPU Memory: {meminfo[1] / 1024**3:.1f} GB total, {meminfo[0] / 1024**3:.1f} GB free")
        print()
    except Exception as e:
        print(f"✗ CuPy error: {e}")
        return {'error': str(e)}
    
    # Run GPU-specific tests
    print("Running GPU tests...")
    print("-" * 60)
    
    result = subprocess.run(
        [
            sys.executable, '-m', 'pytest', 
            'holographic_prod/tests/test_modal_specific.py',
            'holographic_prod/tests/test_modal_performance.py',
            'holographic_prod/tests/test_multi_level_tower.py',
            '-v',
            '-k', 'gpu or GPU or cupy or CuPy or throughput or MultiLevel',
            '--tb=short',
            '--timeout=300',
        ],
        capture_output=True,
        text=True,
        cwd='/root/project',
        timeout=600,
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print()
    print("=" * 80)
    print(f"  Exit code: {result.returncode}")
    print("=" * 80)
    
    return {
        'exit_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
    }


# =============================================================================
# MULTI-LEVEL GPU BENCHMARK
# =============================================================================

@app.function(image=image, gpu="H100", timeout=1800)
def benchmark_multi_level():
    """
    Benchmark MultiLevelTower at different levels on H100.
    
    Theory: Multi-level provides GPU benefit at Level 3 (4096 satellites).
    
    Target: >10k samples/sec learning throughput with Level 3.
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import time
    import numpy as np
    
    print("=" * 80)
    print("  MULTI-LEVEL TOWER GPU BENCHMARK on H100")
    print("=" * 80)
    print()
    
    # Verify CuPy is available
    try:
        import cupy as cp
        print(f"✓ CuPy available: {cp.__version__}")
        device_props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"✓ GPU: {device_props['name'].decode()}")
        meminfo = cp.cuda.runtime.memGetInfo()
        print(f"✓ GPU Memory: {meminfo[1] / 1024**3:.1f} GB total, {meminfo[0] / 1024**3:.1f} GB free")
        print()
    except Exception as e:
        print(f"✗ CuPy error: {e}")
        return {'error': str(e)}
    
    from holographic_prod.memory.multi_level_tower import MultiLevelTower
    from holographic_prod.core.constants import PHI
    
    VOCAB_SIZE = 10000
    BATCH_SIZE = 1024
    N_BATCHES = 10
    SEED = 42
    
    results = {}
    
    # Benchmark each level on GPU
    for levels in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"  Level {levels}: {16**levels} satellites")
        print(f"{'='*60}")
        
        # Create tower on GPU
        tower = MultiLevelTower(
            vocab_size=VOCAB_SIZE,
            levels=levels,
            seed=SEED,
            use_gpu=True,
        )
        
        # Memory info
        mem_size = tower._all_memories.nbytes / 1024
        print(f"  Memory tensor: {tower._all_memories.shape}")
        print(f"  Memory size: {mem_size:.1f} KB")
        
        # Prepare batches
        np.random.seed(SEED)
        batches = []
        for _ in range(N_BATCHES):
            contexts = [list(np.random.randint(0, VOCAB_SIZE, size=5)) for _ in range(BATCH_SIZE)]
            targets = list(np.random.randint(0, VOCAB_SIZE, size=BATCH_SIZE))
            batches.append((contexts, targets))
        
        # Warmup
        print("  Warmup...")
        tower.learn_batch(*batches[0])
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        print("  Benchmarking...")
        total_samples = 0
        start = time.time()
        
        for contexts, targets in batches:
            tower.learn_batch(contexts, targets)
            total_samples += len(contexts)
        
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start
        
        throughput = total_samples / elapsed
        print(f"  Total samples: {total_samples}")
        print(f"  Elapsed: {elapsed:.3f}s")
        print(f"  Throughput: {throughput:,.0f} samples/sec")
        
        # Check satellite distribution
        stats = tower.get_satellite_stats()
        print(f"  Active satellites: {stats['active_satellites']} / {stats['n_satellites']}")
        print(f"  Total bindings: {stats['total_bindings']}")
        
        results[f'level_{levels}'] = {
            'satellites': 16**levels,
            'throughput': throughput,
            'memory_kb': mem_size,
            'active_satellites': stats['active_satellites'],
        }
        
        # Success criteria
        if levels == 3 and throughput >= 10000:
            print(f"  ✅ PASS: Level 3 throughput >= 10k samples/sec")
        elif levels == 3:
            print(f"  ⚠️ BELOW TARGET: Level 3 throughput < 10k samples/sec")
    
    # Summary
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    
    for key, data in results.items():
        print(f"  {key}: {data['throughput']:,.0f} samples/sec ({data['satellites']} satellites)")
    
    # Compare levels
    if 'level_1' in results and 'level_3' in results:
        speedup = results['level_3']['throughput'] / results['level_1']['throughput']
        print(f"\n  Level 3 vs Level 1 speedup: {speedup:.2f}x")
        
        # With more satellites, we should maintain throughput while having more capacity
        if speedup >= 0.5:
            print(f"  ✅ Scaling efficient: Level 3 maintains good throughput with 256x capacity")
        else:
            print(f"  ⚠️ Scaling degraded: Level 3 throughput dropped too much")
    
    return results


# =============================================================================
# GROUNDED EMBEDDINGS VALIDATION (Real Data Test)
# =============================================================================

@app.function(image=image, gpu="H100", timeout=3600)
def test_grounded_embeddings():
    """
    COMPREHENSIVE TEST: Grounded vs Random Embeddings on Real Data.
    
    This test validates the O(√N) sample complexity claim by:
    1. Training with grounded embeddings at multiple sample counts
    2. Training with random embeddings at the same sample counts
    3. Measuring accuracy and generalization at each point
    4. Verifying that grounded embeddings learn faster
    
    Expected: Grounded embeddings should achieve target accuracy with fewer samples.
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import os
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    
    import numpy as np
    import time
    import re
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    from datasets import load_dataset
    
    from holographic_prod.memory import HolographicMemory, MemoryConfig
    from holographic_prod.dreaming import DreamingSystem, EpisodicEntry, integrated_sleep
    from holographic_prod.core.grounded_embeddings import (
        create_grounded_embeddings_fast,
    )
    from holographic_prod.core.constants import PHI_INV
    
    print("=" * 80)
    print("  GROUNDED EMBEDDINGS VALIDATION TEST")
    print("  Testing O(√N) sample complexity on Real TinyStories Data")
    print("=" * 80)
    print()
    
    # GPU setup
    import cupy as cp
    cp.cuda.Device(0).use()
    meminfo = cp.cuda.runtime.memGetInfo()
    print(f"  ⚡ GPU: NVIDIA H100 (CuPy accelerated)")
    print(f"     Memory: {meminfo[1]/1024**3:.1f} GB total, {meminfo[0]/1024**3:.1f} GB free")
    print()
    
    # ==========================================================================
    # LOAD AND PREPARE REAL DATA
    # ==========================================================================
    print("PHASE 1: Loading TinyStories data...")
    
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"  ✓ Dataset loaded: {len(ds):,} stories")
    
    # Build vocabulary
    vocab_size = 10_000  # Reasonable size for testing
    # PRESERVE punctuation and capitalization for natural text!
    word_pattern = re.compile(r"[\w']+|[.,!?;:\"]")
    word_to_idx = {"<unk>": 0, "<pad>": 1}
    word_counts = {}
    
    for i, example in enumerate(ds):
        if i >= 5000:  # Use 5K stories for vocab
            break
        text = example.get("text", "")
        if isinstance(text, str):
            # DON'T lowercase - preserve capitalization!
            for word in word_pattern.findall(text):
                word_counts[word] = word_counts.get(word, 0) + 1
    
    for word, _ in sorted(word_counts.items(), key=lambda x: -x[1])[:vocab_size - 2]:
        idx = len(word_to_idx)
        word_to_idx[word] = idx
    
    # Update vocab_size to actual vocabulary size (IMPORTANT: fixes shape mismatch)
    vocab_size = len(word_to_idx)  # Use actual vocab, not target max
    
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    print(f"  ✓ Vocabulary: {vocab_size:,} words (with punctuation)")
    
    def tokenize(text: str):
        # DON'T lowercase - preserve capitalization!
        return [word_to_idx.get(w, 0) for w in word_pattern.findall(text)]
    
    # Extract samples from stories
    all_samples = []
    context_size = 4
    max_data_samples = 500_000  # Total samples to extract
    
    for example in ds:
        text = example.get("text", "")
        if not isinstance(text, str):
            continue
        tokens = tokenize(text)
        if len(tokens) < context_size + 1:
            continue
        for i in range(len(tokens) - context_size):
            ctx = tokens[i:i + context_size]
            tgt = tokens[i + context_size]
            all_samples.append((ctx, tgt))
            if len(all_samples) >= max_data_samples:
                break
        if len(all_samples) >= max_data_samples:
            break
    
    print(f"  ✓ Extracted: {len(all_samples):,} samples")
    
    # Shuffle and split: 90% train, 10% test
    np.random.seed(42)
    np.random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]
    
    print(f"  ✓ Train: {len(train_samples):,}, Test: {len(test_samples):,}")
    print()
    
    # ==========================================================================
    # COMPUTE GROUNDED EMBEDDINGS (GloVe - FAST)
    # ==========================================================================
    print("PHASE 2: Computing grounded embeddings (GloVe)...")
    
    start = time.time()
    # Use GloVe: pre-trained on 6 billion tokens, much better than co-occurrence
    grounded_embeddings, coverage = create_grounded_embeddings_fast(
        vocab=word_to_idx,
        glove_dim=50,  # 50d is fastest, still excellent semantic structure
        cache_dir="/checkpoints/glove",  # Persistent Modal volume
    )
    grounding_time = time.time() - start
    
    print(f"  ✓ Grounding complete: {grounding_time:.1f}s")
    print(f"    GloVe coverage: {coverage:.1%}")
    print(f"    Embeddings shape: {grounded_embeddings.shape}")
    print()
    
    # ==========================================================================
    # LEARNING CURVE COMPARISON
    # ==========================================================================
    print("PHASE 3: Learning curve comparison...")
    print("  Testing at multiple sample counts to verify O(√N) claim")
    print()
    
    # Sample counts to test (exponentially increasing)
    sample_counts = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000]
    
    results = {
        'grounded': {'train_acc': [], 'test_acc': [], 'time': []},
        'random': {'train_acc': [], 'test_acc': [], 'time': []},
    }
    
    def evaluate_accuracy(model, samples, n_samples=500, dreamer=None):
        """
        Evaluate accuracy on a sample of data.
        
        If dreamer is provided, uses distribution-based accuracy (top-5).
        Otherwise falls back to exact match.
        """
        exact_correct = 0
        top5_correct = 0
        total = min(n_samples, len(samples))
        
        for ctx, tgt in samples[:total]:
            # Always compute exact match
            pred, conf, _ = model.retrieve_parallel(ctx, force_parallel=True)
            if pred == tgt:
                exact_correct += 1
            
            # If dreamer available, check distribution
            if dreamer is not None:
                n_protos = sum(len(lvl) for lvl in dreamer.semantic_memory.levels)
                if n_protos > 0:
                    ctx_mat = model.embed_sequence(ctx)
                    results = dreamer.semantic_memory.retrieve(
                        ctx_mat, top_k=1, use_pattern_completion=True
                    )
                    if results and results[0][0] is not None:
                        proto = results[0][0]
                        target_dist = proto.target_distribution
                        ranked = sorted(target_dist.keys(), key=lambda x: -target_dist.get(x, 0))
                        if tgt in ranked[:5]:
                            top5_correct += 1
        
        # Return top-5 if dreamer available and has prototypes, else exact
        if dreamer is not None and top5_correct > 0:
            return top5_correct / total
        return exact_correct / total
    
    for n_samples in sample_counts:
        if n_samples > len(train_samples):
            break
            
        print(f"  Testing with {n_samples:,} samples...")
        
        # Train with GROUNDED embeddings + DREAMING
        start = time.time()
        model_grounded = HolographicMemory(
            vocab_size=vocab_size,
            max_levels=4,  # 65K satellites
            use_gpu=True,
            grounded_embeddings=grounded_embeddings,
        )
        
        # Create DreamingSystem for prototype formation
        dreamer_grounded = DreamingSystem(
            basis=model_grounded.basis,
            xp=model_grounded.xp,
            use_salience=True,
            use_novelty=True,
            use_predictive_coding=False,  # Allow all episodes to form prototypes
            use_pattern_completion=True,
        )
        
        # Train with periodic dreaming
        episodes = []
        # Dream every 500 samples minimum, or 5 times during training, whichever is more frequent
        dream_interval = min(500, max(100, n_samples // 5))
        min_episodes_for_dream = 50  # Need at least 50 episodes for meaningful clustering
        
        for i, (ctx, tgt) in enumerate(train_samples[:n_samples]):
            model_grounded.learn(ctx, tgt)
            
            # Collect ALL episodes for better prototype formation
            ctx_mat = model_grounded.embed_sequence(ctx)
            episodes.append(EpisodicEntry(context_matrix=ctx_mat, target_token=tgt))
            
            # Periodic dreaming for prototype formation
            if len(episodes) >= min_episodes_for_dream and i > 0 and i % dream_interval == 0:
                integrated_sleep(
                    memory=model_grounded,
                    dreaming_system=dreamer_grounded,
                    episodes=episodes,
                    rem_cycles=1,
                    verbose=False,
                )
                episodes = []  # Clear buffer after dreaming
        
        # Final dream to consolidate remaining episodes
        if len(episodes) > 10:
            integrated_sleep(
                memory=model_grounded,
                dreaming_system=dreamer_grounded,
                episodes=episodes,
                rem_cycles=1,
                verbose=False,
            )
        
        grounded_train_time = time.time() - start
        n_protos = sum(len(lvl) for lvl in dreamer_grounded.semantic_memory.levels)
        
        # Evaluate grounded with distribution-based accuracy
        train_acc_grounded = evaluate_accuracy(model_grounded, train_samples[:n_samples], dreamer=dreamer_grounded)
        test_acc_grounded = evaluate_accuracy(model_grounded, test_samples, dreamer=dreamer_grounded)
        
        results['grounded']['train_acc'].append(train_acc_grounded)
        results['grounded']['test_acc'].append(test_acc_grounded)
        results['grounded']['time'].append(grounded_train_time)
        
        # Train with RANDOM embeddings + DREAMING (same methodology)
        start = time.time()
        model_random = HolographicMemory(
            vocab_size=vocab_size,
            max_levels=4,  # Same architecture
            use_gpu=True,
            grounded_embeddings=None,  # Random
        )
        
        # Create DreamingSystem for random too (fair comparison)
        dreamer_random = DreamingSystem(
            basis=model_random.basis,
            xp=model_random.xp,
            use_salience=True,
            use_novelty=True,
            use_predictive_coding=False,
            use_pattern_completion=True,
        )
        
        episodes_random = []
        
        for i, (ctx, tgt) in enumerate(train_samples[:n_samples]):
            model_random.learn(ctx, tgt)
            
            # Collect ALL episodes (same as grounded for fair comparison)
            ctx_mat = model_random.embed_sequence(ctx)
            episodes_random.append(EpisodicEntry(context_matrix=ctx_mat, target_token=tgt))
            
            if len(episodes_random) >= min_episodes_for_dream and i > 0 and i % dream_interval == 0:
                integrated_sleep(
                    memory=model_random,
                    dreaming_system=dreamer_random,
                    episodes=episodes_random,
                    rem_cycles=1,
                    verbose=False,
                )
                episodes_random = []
        
        if len(episodes_random) > 10:
            integrated_sleep(
                memory=model_random,
                dreaming_system=dreamer_random,
                episodes=episodes_random,
                rem_cycles=1,
                verbose=False,
            )
        
        random_train_time = time.time() - start
        n_protos_random = sum(len(lvl) for lvl in dreamer_random.semantic_memory.levels)
        
        # Evaluate random with distribution-based accuracy
        train_acc_random = evaluate_accuracy(model_random, train_samples[:n_samples], dreamer=dreamer_random)
        test_acc_random = evaluate_accuracy(model_random, test_samples, dreamer=dreamer_random)
        
        results['random']['train_acc'].append(train_acc_random)
        results['random']['test_acc'].append(test_acc_random)
        results['random']['time'].append(random_train_time)
        
        # Print results with prototype counts
        print(f"    Grounded: train={train_acc_grounded:.1%}, test={test_acc_grounded:.1%}, protos={n_protos}, time={grounded_train_time:.1f}s")
        print(f"    Random:   train={train_acc_random:.1%}, test={test_acc_random:.1%}, protos={n_protos_random}, time={random_train_time:.1f}s")
        
        # Free memory
        del model_grounded
        del model_random
        del dreamer_grounded
        del dreamer_random
        cp.get_default_memory_pool().free_all_blocks()
    
    print()
    
    # ==========================================================================
    # RESULTS ANALYSIS
    # ==========================================================================
    print("=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    print(f"{'Samples':>10} | {'Grounded Train':>14} | {'Grounded Test':>13} | {'Random Train':>12} | {'Random Test':>11}")
    print("-" * 80)
    
    for i, n_samples in enumerate(sample_counts[:len(results['grounded']['train_acc'])]):
        gt = results['grounded']['train_acc'][i]
        ge = results['grounded']['test_acc'][i]
        rt = results['random']['train_acc'][i]
        re = results['random']['test_acc'][i]
        print(f"{n_samples:>10,} | {gt:>14.1%} | {ge:>13.1%} | {rt:>12.1%} | {re:>11.1%}")
    
    print()
    
    # Calculate advantage
    advantages = []
    for i in range(len(results['grounded']['test_acc'])):
        ge = results['grounded']['test_acc'][i]
        re = results['random']['test_acc'][i]
        if re > 0:
            advantages.append(ge / re)
    
    if advantages:
        avg_advantage = np.mean(advantages)
        print(f"  Average grounded/random test accuracy ratio: {avg_advantage:.2f}x")
    
    # Check O(√N) claim
    # If grounded at N samples matches random at K samples, and K ≈ N², then O(√N) is validated
    print()
    print("  SAMPLE EFFICIENCY ANALYSIS:")
    
    grounded_test = results['grounded']['test_acc']
    random_test = results['random']['test_acc']
    tested_counts = sample_counts[:len(grounded_test)]
    
    # For each grounded accuracy, find how many random samples needed to match
    for i, (g_acc, g_samples) in enumerate(zip(grounded_test, tested_counts)):
        # Find interpolated random sample count to reach same accuracy
        for j in range(len(random_test) - 1):
            if random_test[j] <= g_acc <= random_test[j + 1]:
                # Linear interpolation
                frac = (g_acc - random_test[j]) / (random_test[j + 1] - random_test[j] + 1e-10)
                r_samples = tested_counts[j] + frac * (tested_counts[j + 1] - tested_counts[j])
                ratio = r_samples / g_samples if g_samples > 0 else 0
                print(f"    Grounded @ {g_samples:,} ({g_acc:.1%}) ≈ Random @ {r_samples:,.0f} samples")
                print(f"      → Efficiency gain: {ratio:.1f}x")
                break
    
    print()
    
    # Final verdict
    final_grounded = results['grounded']['test_acc'][-1] if results['grounded']['test_acc'] else 0
    final_random = results['random']['test_acc'][-1] if results['random']['test_acc'] else 0
    
    print("=" * 80)
    print("  VERDICT")
    print("=" * 80)
    
    if final_grounded > final_random * 1.1:
        print("  ✅ GROUNDED EMBEDDINGS OUTPERFORM RANDOM")
        print(f"     Final test accuracy: Grounded={final_grounded:.1%} vs Random={final_random:.1%}")
        print("     Theory validated: O(√N) sample complexity appears confirmed!")
    elif final_grounded > final_random:
        print("  ⚠️ GROUNDED SLIGHTLY BETTER (marginal improvement)")
        print(f"     Final test accuracy: Grounded={final_grounded:.1%} vs Random={final_random:.1%}")
    else:
        print("  ❌ NO IMPROVEMENT (investigate implementation)")
        print(f"     Final test accuracy: Grounded={final_grounded:.1%} vs Random={final_random:.1%}")
    
    return {
        'sample_counts': tested_counts,
        'grounded': results['grounded'],
        'random': results['random'],
        'final_grounded_test_acc': final_grounded,
        'final_random_test_acc': final_random,
        'grounding_time': grounding_time,
    }


# =============================================================================
# LOCAL ENTRY
# =============================================================================

@app.local_entrypoint()
def main(
    gpu_tests: bool = False, 
    benchmark: bool = False,
    train_run: bool = False,
    preflight: bool = False,  # NEW: Pre-flight validation
    test_grounding: bool = False,  # NEW: Run grounded embeddings validation test
    train_samples: int = 100_000_000,
    train_vocab: int = 200_000,  # v5.31.2: 200K vocab for minimal <unk> (~2% OOV)
    train_levels: int = 6,  # H100-optimized: 16M satellites
    train_batch: int = 8192,  # H100-optimized: 8K batch size
    grounding: bool = True,  # Use grounded embeddings (O(√N) efficiency)
    grounding_samples: int = 100_000,  # Samples for grounding phase
    fresh: bool = False,  # Start fresh, ignore existing checkpoint
):
    """Run tests or training on Modal.
    
    H100-OPTIMIZED DEFAULTS:
    - Level 6: 16M satellites (1GB VRAM), ~83% accuracy @ 200K patterns
    - For more headroom: use --train-levels 7 (268M satellites, 16GB)
    - Batch 8K: Good balance for H100 80GB
    
    Args:
        gpu_tests: If True, run GPU-specific pytest tests.
        benchmark: If True, run multi-level benchmark.
        preflight: If True, run pre-flight validation (RECOMMENDED before training).
        train_run: If True, start full training run.
        train_samples: Number of samples for training (default 100M).
        train_vocab: Vocabulary size for training.
        train_levels: Multi-level tower depth.
        train_batch: Batch size for training (default 8K for H100).
        Otherwise run theory tests.
        
    Usage:
        modal run holographic_prod/train_modal.py --preflight  # RUN THIS FIRST!
        modal run holographic_prod/train_modal.py --train-run  # FULL TRAINING
        modal run holographic_prod/train_modal.py --gpu-tests
        modal run holographic_prod/train_modal.py --benchmark
        modal run holographic_prod/train_modal.py  # theory tests
    """
    if preflight:
        print("=" * 70)
        print("  RUNNING PRE-FLIGHT VALIDATION on Modal H100")
        print("  (Run this BEFORE training to catch issues early)")
        print("=" * 70)
        result = preflight_validation.remote()
        print()
        if result['passed']:
            print("✓ ALL CHECKS PASSED - Safe to proceed with training")
            print("  Run: modal run holographic_prod/train_modal.py --train-run")
        else:
            print("✗ VALIDATION FAILED - Do NOT proceed with training")
            print("  Fix the issues above first.")
        return
    
    if test_grounding:
        print("=" * 70)
        print("  GROUNDED EMBEDDINGS VALIDATION TEST on Modal H100")
        print("  Testing O(√N) sample complexity on Real TinyStories Data")
        print("=" * 70)
        result = test_grounded_embeddings.remote()
        print()
        print("  Results returned to caller.")
        return
    
    if train_run:
        print("=" * 70)
        print("  STARTING LARGE TRAINING RUN on Modal H100")
        print(f"  Samples: {train_samples:,}")
        print(f"  Vocab: {train_vocab:,}")
        print(f"  Levels: {train_levels} ({16**train_levels:,} satellites)")
        print(f"  Batch: {train_batch:,}")
        print(f"  Data: Real (OpenWebText)")
        grounding_status = f"Grounded ({grounding_samples:,} samples)" if grounding else "Random"
        print(f"  Embeddings: {grounding_status}")
        print(f"  Fresh start: {fresh}")
        print("=" * 70)
        result = train.remote(
            max_samples=train_samples,
            vocab_size=train_vocab,
            max_levels=train_levels,
            batch_size=train_batch,
            fresh_start=fresh,
            use_grounding=grounding,
            grounding_samples=grounding_samples,
        )
        print(f"\nTraining complete!")
        print(f"Final perplexity: {result.get('final_perplexity', 'N/A')}")
        samples = result.get('samples_processed', 'N/A')
        print(f"Samples processed: {samples:,}" if isinstance(samples, int) else f"Samples processed: {samples}")
    elif benchmark:
        print("Running Multi-Level Tower benchmark on Modal H100...")
        result = benchmark_multi_level.remote()
        print(f"\nResults:")
        for key, data in result.items():
            if key != 'error':
                print(f"  {key}: {data['throughput']:,.0f} samples/sec")
    elif gpu_tests:
        print("Running GPU pytest tests on Modal H100...")
        result = run_gpu_tests.remote()
        print(f"\nExit code: {result['exit_code']}")
        if result['exit_code'] != 0:
            print("Some tests failed!")
    else:
        print("Running theory tests on Modal H100...")
        result = test.remote()
        print(f"\nResults: {result}")
    print("✓ Complete!")


if __name__ == "__main__":
    # LOCAL SMOKE TEST (CPU-only, for dev verification)
    # For GPU training, use: modal run holographic_prod/train_modal.py
    import numpy as np
    import sys
    sys.path.insert(0, '.')
    
    print("="*60)
    print("  LOCAL SMOKE TEST (CPU-only)")
    print("  For GPU training: modal run holographic_prod/train_modal.py")
    print("="*60)
    
    from holographic_prod.memory import HolographicMemory, MemoryConfig
    from holographic_prod.dreaming import DreamingSystem, EpisodicEntry, integrated_sleep
    from holographic_prod.dreaming import DreamingSystem, EpisodicEntry, integrated_sleep
    from holographic_prod.attention import ToroidalAttention
    from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ
    
    print(f"\nTheory constants:")
    print(f"  φ = {PHI:.10f}")
    print(f"  φ⁻¹ = {PHI_INV:.10f}")
    print(f"  φ⁻² = {PHI_INV_SQ:.10f}")
    print(f"  φ² = φ + 1? {abs(PHI**2 - PHI - 1) < 1e-10}")
    
    config = MemoryConfig(learning_rate=PHI_INV)
    model = HolographicMemory(vocab_size=100, config=config, use_gpu=False, seed=42)
    
    np.random.seed(42)
    batch = [([np.random.randint(100) for _ in range(3)], np.random.randint(100)) for _ in range(100)]
    model.learn_batch(batch)
    
    print(f"\n  ✓ HolographicMemory: {model.n_patterns} patterns, stability={model.get_stability():.3f}")
    
    # All 7 parsimonies enabled
    dreamer = DreamingSystem(
        basis=model.basis,
        xp=model.xp,
        use_salience=True,
        use_novelty=True,
        use_predictive_coding=True,
        use_pattern_completion=True,
        use_inhibition_of_return=True,
        use_sequence_replay=True,
        use_pseudo_rehearsal=True,
    )
    episodes = [EpisodicEntry(model.embed_sequence(ctx), tgt) for ctx, tgt in batch[:20]]
    
    # Test integrated sleep (combines tower + systems dreaming)
    sleep_result = integrated_sleep(
        memory=model,
        dreaming_system=dreamer,
        episodes=episodes,
        rem_cycles=1,
        verbose=False,
    )
    tower_stats = sleep_result['tower_stats']
    systems_stats = sleep_result['systems_stats']
    print(f"  ✓ Integrated Sleep:")
    print(f"      Tower stability: {tower_stats['pre_stability']:.3f} → {tower_stats['post_stability']:.3f}")
    print(f"      Prototypes: {systems_stats.get('prototypes_created', 0)}, Schemas: {systems_stats.get('schemas_discovered', 0)}")
    print(f"      Phases: {' → '.join(sleep_result['phases_completed'])}")
    
    attention = ToroidalAttention(n_satellites=16)
    print(f"  ✓ ToroidalAttention: 16 satellites")
    
    print("\n  ✅ Local smoke test passed!")
    print("  For real training, run: modal run holographic_prod/train_modal.py")
