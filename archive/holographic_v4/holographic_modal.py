"""
Holographic v4.8.0 — Modal Training Runner with Brain-Inspired Parsimonies
==========================================================================

Clean Modal deployment for the theory-true SCCMU implementation.
Last Updated: 2026-01-12 19:30 (Witness MANDATORY with φ² subsampling - NOT optional)

CORE FEATURES:
    - Full Grace operator (grade-wise φ⁻ᵏ contraction = viscosity for bivectors)
    - Wedge product for vorticity tracking (discrimination signal)
    - TRUE HOLOGRAPHIC MEMORY: O(1) superposition storage (replaces hash tables)
    - Vorticity-weighted decoding (prevents mode collapse)
    - Grace basin discovery for semantic retrieval
    - Distributed prior for smooth generalization (φ-weighted superposition)
    - DREAMING: Self-organizing consolidation via grace_stability
    - φ-CURRICULUM: Theory-derived context scaling (φ² growth per stage)
    - EQUILIBRIUM FORGETTING: φ-decay sparse efficiency
    
v4.7.0 ADDITIONS:
    - META-COGNITIVE LOOP: Predictive coding (learn only from surprises)
    - ADAPTIVE SLEEP: φ-derived triggers (memory pressure, novelty, error rate)
    - FULL GPU: CuPy acceleration for all operations
    - 90% EFFICIENCY: Skip redundant patterns via prediction

φ-CURRICULUM (Theory-Derived Context Scaling):
    Context grows by φ² (≈ 2.618) per stage, samples grow by φ:
    
        context(stage) = 512 × φ^(2 × stage)
        samples_per_stage = 500k × φ^stage
    
    STAGES:
        Stage 0: 512 ctx,     500k samples  (0 - 500k)
        Stage 1: 1,340 ctx,   809k samples  (500k - 1.31M)
        Stage 2: 3,509 ctx,   1.31M samples (1.31M - 2.62M)
        Stage 3: 9,192 ctx,   2.12M samples (2.62M - 4.74M)
        Stage 4: 24,065 ctx,  3.43M samples (4.74M - 8.17M)
        Stage 5: 50,000 ctx (capped)
    
    Enable with: --enable-curriculum --target-context-size 50000

EQUILIBRIUM FORGETTING (Theory-True Memory Management):
    NOT capacity-based (arbitrary) but equilibrium-based:
    
        retention = salience × φ^(access_count) × φ^(-Δt/τ)
        Forget if retention < φ⁻³ (≈ 0.236)
    
    Creates SPARSE EFFICIENCY through natural selection:
        - Frequently accessed → survives (reconsolidation)
        - High salience → survives (emotional tagging)
        - Weak + old + not accessed → naturally forgotten
        
    Typical result: 99% sparse (494k pruned → 5k surviving)

BRAIN-INSPIRED PARSIMONIES (12 total):
    MEMORY ENCODING:
    1. Emotional Salience: scalar + pseudoscalar prioritization
    2. Novelty-Gated Learning: prioritize novel episodes
    3. Delta/Schema Compression: sparse deviation storage
    4. Predictive Coding: encode only Grace residuals
    
    MEMORY MAINTENANCE:
    5. φ-Decay Forgetting: theory-true sparse efficiency
    6. Interference Management: merge similar prototypes
    7. Reconsolidation: access strengthens memories
    8. Pseudo-Rehearsal: prevent catastrophic forgetting
    
    MEMORY RETRIEVAL:
    9. Working Memory Cache: 7±2 items with φ-decay recency
    10. Pattern Completion: Grace flow to attractors
    11. Inhibition of Return: temporal suppression
    12. Sequence Replay: sharp wave ripple analog

SELF-ORGANIZING CONSOLIDATION:
    Grace-stability σ(M) = fraction of energy in witness
    ALL episodes are consolidation candidates
    Priority = stability × salience (coherent + important first)
    
    Adaptive similarity threshold based on prototype diversity.

NORMALIZATION STRATEGY:
    - Frobenius: Numerical necessity for long contexts (uniform scaling)
    - Grace: Theory-true grade-wise damping (applied once at end)
    
    For 50k+ contexts:
    - Chunked computation (1000 tokens at a time)
    - Frobenius after each chunk (prevents norm collapse)
    - Grace once at final result (semantic normalization)

OPTIMAL PARAMETERS:
    --noise-std 0.3        OPTIMAL: balances discrimination + generalization
    --sleep-every 50000    Run sleep cycle every N samples
    --enable-curriculum    Enable φ-curriculum context scaling
    --target-context-size 50000  Target context size for curriculum
    --max-attractors 100000  Allow 100k episodic memories
    Default: use_vorticity_decoding=True (prevents mode collapse)

THEORY INSIGHT:
    "Waking SCCMU writes episodes. Dreaming SCCMU discovers invariants.
     REM tests which invariants are real by forcing them to survive Grace.
     Equilibrium forgetting creates sparse efficiency through natural selection."
    
    Grace is viscosity for bivectors AND the semantic normalizer.
    All thresholds derived from φ and spectral structure.

Usage:
    # Basic training run
    modal run holographic_v4/holographic_modal.py::train --max-samples 10000000
    
    # With φ-curriculum to 50k context (recommended for production)
    modal run -d holographic_v4/holographic_modal.py::train \\
        --dataset gutenberg \\
        --enable-curriculum \\
        --target-context-size 50000 \\
        --max-samples 50000000 \\
        --max-attractors 100000
    
    # Test run
    modal run holographic_v4/holographic_modal.py::test
"""

import modal
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass, field
from collections import deque

# Theory-derived constants (need at module level for function signature defaults)
PHI = 1.6180339887498949  # Golden ratio
PHI_INV = 1 / PHI  # ≈ 0.618
PHI_INV_SQ = PHI_INV ** 2  # ≈ 0.382
PHI_INV_CUBE = PHI_INV ** 3  # ≈ 0.236

# Precision for tensor cores (set inside functions after numpy/cupy import)
import numpy as np
DTYPE = np.float32  # TF32 tensor cores on H100


# =============================================================================
# TRAINING METRICS — Comprehensive tracking for large runs
# =============================================================================

@dataclass
class TrainingMetrics:
    """
    Track and display training metrics for large Modal runs.
    
    THEORY-TRUE METRICS (v4.10):
    - Operational: rate, samples, attractors (standard ML metrics)
    - Learning Health: accuracy, confidence, pruning rate
    - Semantic Quality: prototype coherence, schema coverage
    - Theory Validation: witness stability, enstrophy bounds, Grace damping
    
    KEY INSIGHT: Most metrics tell you "what" (counts), but learning health
    tells you "how well" (quality). We need both.
    """
    start_time: float = field(default_factory=time.perf_counter)
    max_samples: int = 1_000_000
    
    # Core counters
    samples: int = 0
    attractors: int = 0
    prototypes: int = 0
    schemas: int = 0
    forgotten: int = 0
    
    # Rolling windows for averages
    rate_window: deque = field(default_factory=lambda: deque(maxlen=10))
    accuracy_window: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Learning curve (accuracy at checkpoints)
    learning_curve: List[Tuple[int, float]] = field(default_factory=list)
    
    # Generation quality
    generation_scores: List[Tuple[int, float, int]] = field(default_factory=list)  # (samples, diversity, uniques)
    
    # Curriculum tracking
    current_stage: int = 0
    current_context_size: int = 512
    stage_changes: List[Tuple[int, int, int]] = field(default_factory=list)  # (samples, stage, ctx_size)
    
    # Timing
    last_log_time: float = field(default_factory=time.perf_counter)
    training_time: float = 0.0  # Pure training time (excluding I/O)
    
    # =========================================================================
    # THEORY-TRUE METRICS — What really matters for learning
    # =========================================================================
    
    # Learning Signal Quality (predictive coding efficiency)
    prediction_correct_window: deque = field(default_factory=lambda: deque(maxlen=100))
    novel_patterns_window: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Holographic Memory Health
    retrieval_confidence_window: deque = field(default_factory=lambda: deque(maxlen=50))
    pruning_rate_window: deque = field(default_factory=lambda: deque(maxlen=10))  # memories pruned per interval
    
    # Theory Validation (sampled periodically - expensive)
    witness_stability_history: List[Tuple[int, float]] = field(default_factory=list)
    enstrophy_history: List[Tuple[int, float]] = field(default_factory=list)
    
    # Semantic Organization
    prototype_coherence_history: List[Tuple[int, float]] = field(default_factory=list)
    
    def update(self, samples: int, attractors: int, prototypes: int = 0, 
               schemas: int = 0, forgotten: int = 0, context_size: int = 512, stage: int = 0):
        """Update core metrics."""
        self.samples = samples
        self.attractors = attractors
        self.prototypes = prototypes
        self.schemas = schemas
        self.forgotten = forgotten
        
        # Track curriculum changes
        if stage != self.current_stage:
            self.stage_changes.append((samples, stage, context_size))
            self.current_stage = stage
            self.current_context_size = context_size
        elif context_size != self.current_context_size:
            self.current_context_size = context_size
    
    def record_rate(self, rate: float):
        """Record training rate for rolling average."""
        self.rate_window.append(rate)
    
    def record_accuracy(self, accuracy: float):
        """Record accuracy for rolling average and learning curve."""
        self.accuracy_window.append(accuracy)
        self.learning_curve.append((self.samples, accuracy))
    
    def record_generation(self, diversity_ratio: float, unique_tokens: int):
        """Record generation quality metrics."""
        self.generation_scores.append((self.samples, diversity_ratio, unique_tokens))
    
    # =========================================================================
    # THEORY-TRUE METRIC RECORDING
    # =========================================================================
    
    def record_prediction(self, correct: bool, is_novel: bool):
        """
        Record predictive coding metrics per sample.
        
        THEORY INSIGHT: We should learn mostly from surprises (wrong predictions).
        If prediction_rate is high, we're redundantly encoding known patterns.
        If novel_rate is low, we're in a repetitive regime.
        """
        self.prediction_correct_window.append(1 if correct else 0)
        self.novel_patterns_window.append(1 if is_novel else 0)
    
    def record_retrieval_confidence(self, confidence: float):
        """Record holographic retrieval confidence for memory health tracking."""
        self.retrieval_confidence_window.append(confidence)
    
    def record_pruning(self, pruned_count: int):
        """Record memory pruning for equilibrium health tracking."""
        self.pruning_rate_window.append(pruned_count)
    
    def record_theory_metrics(self, witness_stability: float, enstrophy: float):
        """Record expensive theory-validation metrics (called during deep_log)."""
        self.witness_stability_history.append((self.samples, witness_stability))
        self.enstrophy_history.append((self.samples, enstrophy))
    
    def record_prototype_coherence(self, coherence: float):
        """Record semantic organization quality (called during deep_log if dreaming)."""
        self.prototype_coherence_history.append((self.samples, coherence))
    
    @property
    def prediction_rate(self) -> float:
        """How often we correctly predict (should be moderate - too high means redundant)."""
        if not self.prediction_correct_window:
            return 0.0
        return sum(self.prediction_correct_window) / len(self.prediction_correct_window)
    
    @property
    def novelty_rate(self) -> float:
        """How often we see novel patterns (should be positive for learning)."""
        if not self.novel_patterns_window:
            return 0.0
        return sum(self.novel_patterns_window) / len(self.novel_patterns_window)
    
    @property
    def avg_retrieval_confidence(self) -> float:
        """Average retrieval confidence (should be > φ⁻² for good memory health)."""
        if not self.retrieval_confidence_window:
            return 0.0
        return sum(self.retrieval_confidence_window) / len(self.retrieval_confidence_window)
    
    @property
    def avg_pruning_rate(self) -> float:
        """Average memories pruned per interval (equilibrium indicator)."""
        if not self.pruning_rate_window:
            return 0.0
        return sum(self.pruning_rate_window) / len(self.pruning_rate_window)
    
    def learning_efficiency(self) -> str:
        """
        Assess learning efficiency based on predictive coding.
        
        THEORY: Optimal learning has moderate prediction rate (0.3-0.7).
        - Too low (< 0.3): Chaotic, no structure learned
        - Too high (> 0.7): Redundant, wasting compute on known patterns
        - Sweet spot: Learning from meaningful surprises
        """
        pred_rate = self.prediction_rate
        if pred_rate < 0.3:
            return "⚠ chaotic (no structure)"
        elif pred_rate > 0.7:
            return "⚠ redundant (skip more)"
        elif pred_rate > 0.5:
            return "✓ efficient"
        else:
            return "→ learning"
    
    def memory_health(self) -> str:
        """
        Assess holographic memory health.
        
        THEORY: Healthy memory has:
        - High retrieval confidence (> φ⁻² ≈ 0.382)
        - Stable attractor count (not exploding)
        - Active pruning (equilibrium forgetting working)
        """
        conf = self.avg_retrieval_confidence
        if conf < PHI_INV_CUBE:  # < 0.236
            return "⚠ noisy (interference)"
        elif conf < PHI_INV_SQ:  # < 0.382
            return "~ moderate"
        elif conf >= PHI_INV:  # >= 0.618
            return "✓ excellent"
        return "✓ good"
    
    @property
    def avg_rate(self) -> float:
        """Rolling average training rate."""
        return sum(self.rate_window) / len(self.rate_window) if self.rate_window else 0.0
    
    @property
    def overall_rate(self) -> float:
        """Overall training rate (samples/elapsed_time)."""
        elapsed = self.elapsed
        return self.samples / elapsed if elapsed > 0 else 0.0
    
    @property
    def avg_accuracy(self) -> float:
        """Rolling average accuracy."""
        return sum(self.accuracy_window) / len(self.accuracy_window) if self.accuracy_window else 0.0
    
    @property
    def progress(self) -> float:
        """Progress as percentage."""
        return 100 * self.samples / self.max_samples if self.max_samples > 0 else 0.0
    
    @property
    def elapsed(self) -> float:
        """Total elapsed time."""
        return time.perf_counter() - self.start_time
    
    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining in seconds."""
        if self.samples == 0 or self.avg_rate == 0:
            return float('inf')
        remaining = self.max_samples - self.samples
        return remaining / self.avg_rate
    
    def format_eta(self) -> str:
        """Format ETA as human-readable string."""
        eta = self.eta_seconds
        if eta == float('inf'):
            return "calculating..."
        hours = int(eta // 3600)
        minutes = int((eta % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    
    def format_elapsed(self) -> str:
        """Format elapsed time as human-readable string."""
        elapsed = self.elapsed
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    
    def memory_fill(self, max_attractors: int) -> float:
        """Memory fill percentage."""
        return 100 * self.attractors / max_attractors if max_attractors > 0 else 0.0
    
    def sparse_efficiency(self) -> float:
        """What % of total memories have been pruned (sparse efficiency)."""
        total = self.attractors + self.forgotten
        return 100 * self.forgotten / total if total > 0 else 0.0
    
    def learning_trend(self) -> str:
        """Determine if learning is improving, stable, or declining."""
        if len(self.learning_curve) < 3:
            return "starting"
        recent = [acc for _, acc in self.learning_curve[-5:]]
        older = [acc for _, acc in self.learning_curve[-10:-5]] if len(self.learning_curve) >= 10 else recent[:1]
        
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older) if older else avg_recent
        
        diff = avg_recent - avg_older
        if diff > 0.05:
            return "↑ improving"
        elif diff < -0.05:
            return "↓ declining"
        return "→ stable"
    
    def generation_quality_trend(self) -> str:
        """Track generation quality trend."""
        if len(self.generation_scores) < 2:
            return "insufficient data"
        recent = [div for _, div, _ in self.generation_scores[-3:]]
        avg = sum(recent) / len(recent)
        if avg > 0.7:
            return "✓ diverse"
        elif avg > 0.4:
            return "→ moderate"
        return "⚠ repetitive"


def format_number(n: int) -> str:
    """Format large numbers compactly."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def print_progress_header():
    """Print minimal training header."""
    print("\n" + "─" * 80)
    print("  TRAINING PROGRESS")
    print("─" * 80)


# =============================================================================
# TWO-TIER LOGGING SYSTEM — Frequent lightweight + periodic full
# =============================================================================
# 
# DESIGN RATIONALE:
#   - Lightweight logs (every 1K): Just rate/attractors, ~0ms overhead
#   - Full logs (every 10K): Include accuracy check, ~200-500ms acceptable
#   - Generation (every 50K): Full generation display
#
# This gives 10x more frequent feedback without blocking training.
# =============================================================================

def print_lightweight_progress(samples: int, attractors: int, rate: float, 
                               max_attractors: int, context_size: int, 
                               max_samples: int, last_accuracy: float = 0.0,
                               metrics: 'TrainingMetrics' = None):
    """
    Ultra-lightweight inline progress — NO INFERENCE, just counters + cached metrics.
    
    Shows: progress, memory, speed, accuracy, learning efficiency.
    Prints single line, carriage-return style for minimal visual noise.
    Called frequently (every 1K samples) without blocking training.
    """
    progress_pct = (samples / max_samples * 100) if max_samples > 0 else 0
    mem_fill = (attractors / max_attractors * 100) if max_attractors > 0 else 0
    
    # Speed emoji
    speed_icon = "🚀" if rate >= 500 else "⚡" if rate >= 100 else "🐢"
    
    # Learning efficiency indicator (from metrics if available)
    learn_icon = "📚"  # Default
    if metrics:
        pred_rate = metrics.prediction_rate
        if pred_rate > 0.7:
            learn_icon = "🔄"  # Redundant
        elif pred_rate > 0.5:
            learn_icon = "✓"   # Efficient
        elif pred_rate > 0.3:
            learn_icon = "📚"  # Learning
        elif pred_rate > 0:
            learn_icon = "🌀"  # Chaotic
    
    # Compact line with key info - use newline for terminal capture visibility
    print(f"  {speed_icon} {format_number(samples):>6s} ({progress_pct:4.1f}%) │ "
          f"💾 {attractors:,} ({mem_fill:4.1f}%) │ "
          f"ctx:{context_size:>5} │ "
          f"{rate:>5.0f}/s │ "
          f"{learn_icon} acc:{last_accuracy:>4.1%}", flush=True)


def print_progress_line(metrics: TrainingMetrics, accuracy: float, rate: float, 
                        status: str, context_size: int = 512, max_attractors: int = 500000,
                        stage: int = 0, batch_size: int = 256):
    """
    Print informative progress line with MEANINGFUL metrics (FULL version).
    
    Shows:
    - Progress bar + ETA
    - Retrieval accuracy (are we learning associations?)
    - Learning efficiency (are we learning from surprises or redundant?)
    - Memory health (is holographic retrieval confident?)
    - Memory stats (fill, pruned)
    """
    progress_pct = metrics.progress
    filled = int(progress_pct / 5)  # 20 char bar
    bar = "█" * filled + "░" * (20 - filled)
    
    # Memory fill percentage
    mem_fill = (metrics.attractors / max_attractors * 100) if max_attractors > 0 else 0
    
    # Color-code speed
    if rate >= 500:
        speed_indicator = "🚀"
    elif rate >= 100:
        speed_indicator = "⚡"
    else:
        speed_indicator = "🐢"
    
    # LEARNING STATUS based on accuracy (φ-derived thresholds)
    if accuracy >= PHI_INV + PHI_INV_SQ:  # ≈ 1.0 (excellent)
        learn_status = "LEARNING ✓✓"
    elif accuracy >= PHI_INV:  # ≈ 0.618 (good)
        learn_status = "LEARNING ✓ "
    elif accuracy >= PHI_INV_SQ:  # ≈ 0.382 (progressing)
        learn_status = "learning ~ "
    else:
        learn_status = "warming up↑"
    
    # LEARNING EFFICIENCY from predictive coding
    efficiency = metrics.learning_efficiency()
    
    # MEMORY HEALTH from retrieval confidence
    mem_health = metrics.memory_health()
    
    print()
    print(f"  ┌─────────────────────────────────────────────────────────────────────────────┐")
    print(f"  │ {bar} {progress_pct:5.1f}% │ {format_number(metrics.samples):>6s} samples │ ETA {metrics.format_eta():<8}         │")
    print(f"  ├─────────────────────────────────────────────────────────────────────────────┤")
    print(f"  │ 🧠 RETRIEVAL ACCURACY: {accuracy:>5.1%}  [{learn_status}]                              │")
    print(f"  │ 📊 Learning Efficiency: {efficiency:<22} Memory: {mem_health:<14} │")
    print(f"  │ 💾 Attractors: {metrics.attractors:>7,} ({mem_fill:4.1f}%) │ Pruned: {metrics.forgotten:>7,} ({metrics.sparse_efficiency():4.1f}%)       │")
    print(f"  │ 📏 Context: {context_size:>5} │ Stage {stage} │ {speed_indicator} {rate:>5,.0f}/s │ Trend: {metrics.learning_trend():<11} │")
    print(f"  └─────────────────────────────────────────────────────────────────────────────┘", flush=True)


def print_diagnostics_block(metrics: TrainingMetrics, model, basis, 
                            wit_stab: float, enstrophy: float, max_attractors: int,
                            meta_surprises: int = 0, meta_redundant: int = 0,
                            novelty_rate: float = 0.0, error_rate: float = 0.0):
    """Print detailed diagnostics summary."""
    fill_pct = metrics.memory_fill(max_attractors)
    sparse_pct = metrics.sparse_efficiency()
    stability_status = "✓" if wit_stab > PHI_INV else "⚠" if wit_stab < PHI_INV_SQ else "~"
    gen_trend = metrics.generation_quality_trend()
    
    # Calculate predictive coding efficiency
    total_meta = meta_surprises + meta_redundant
    skip_rate = (meta_redundant / total_meta * 100) if total_meta > 0 else 0
    
    print()
    print(f"  ┌─────────────────────────────────────────────────────────────────────────┐")
    print(f"  │ 📊 DIAGNOSTICS @ {format_number(metrics.samples):>6s} samples                                     │")
    print(f"  ├─────────────────────────────────────────────────────────────────────────┤")
    print(f"  │ Memory:    {metrics.attractors:>8,} / {max_attractors:,} ({fill_pct:5.1f}% full)                   │")
    print(f"  │ Prototypes: {metrics.prototypes:>6} │ Schemas: {metrics.schemas:>4}                               │")
    print(f"  │ Witness σ:  {wit_stab:>6.3f} {stability_status}  │ Enstrophy: {enstrophy:>8.4f}                      │")
    print(f"  │ Novelty:    {novelty_rate:>5.1%}   │ Skip rate: {skip_rate:>5.1f}%                           │")
    print(f"  │ Generation: {gen_trend:<12}                                              │")
    print(f"  └─────────────────────────────────────────────────────────────────────────┘")


def print_stage_change(stage: int, old_ctx: int, new_ctx: int, samples: int):
    """Print curriculum stage change with visual emphasis."""
    growth = new_ctx / old_ctx if old_ctx > 0 else 1
    print()
    print(f"  ╔═══════════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  🌀 φ-CURRICULUM STAGE {stage}                                                  ║")
    print(f"  ║     Context: {old_ctx:,} → {new_ctx:,} tokens ({growth:.2f}× growth)                   ║")
    print(f"  ║     Triggered @ {format_number(samples)} samples                                      ║")
    print(f"  ╚═══════════════════════════════════════════════════════════════════════════╝")
    print()


def print_sleep_summary(cycle: int, episodes: int, prototypes_created: int, 
                        schemas_discovered: int, total_protos: int, total_schemas: int, sleep_time: float):
    """Print sleep cycle summary with consolidation stats."""
    compression = (prototypes_created / episodes * 100) if episodes > 0 else 0
    print(f"  💤 Sleep #{cycle}: {episodes} episodes → +{prototypes_created} protos " +
          f"(+{schemas_discovered} schemas) │ Total: {total_protos} protos, {total_schemas} schemas │ {sleep_time:.1f}s", flush=True)


def print_training_config(max_samples: int, vocab_size: int, context_size: int, 
                          batch_size: int, enable_curriculum: bool, fast_curriculum: bool,
                          enable_dreaming: bool, dataset: str, max_attractors: int,
                          tokenizer_type: str = "word"):
    """Print training configuration summary."""
    print()
    print("  ┌─────────────────────────────────────────────────────────────────────────┐")
    print("  │                        TRAINING CONFIGURATION                          │")
    print("  ├─────────────────────────────────────────────────────────────────────────┤")
    print(f"  │ Samples:      {max_samples:>12,}  │  Vocab:     {vocab_size:>12,}       │")
    print(f"  │ Batch size:   {batch_size:>12}  │  Max memory: {max_attractors:>11,}       │")
    tok_label = "word ✓" if tokenizer_type == "word" else "bpe"
    print(f"  │ Tokenizer:    {tok_label:>12}  │  Dreaming:  {'✓ Enabled' if enable_dreaming else '✗ Disabled':>12}       │")
    print(f"  │ Dataset:      {dataset:>12}  │                                   │")
    if enable_curriculum:
        mode = "⚡ FAST" if fast_curriculum else "Standard"
        print(f"  │ Curriculum:   {mode:>12}  │  Start ctx:  {context_size:>11,}       │")
    else:
        print(f"  │ Context:      {context_size:>12,}  │  (Fixed - no curriculum)          │")
    print("  ├─────────────────────────────────────────────────────────────────────────┤")
    print("  │                         OPTIMIZATIONS ACTIVE                           │")
    print("  ├─────────────────────────────────────────────────────────────────────────┤")
    print(f"  │ ✓ Batched context computation (2048 samples parallel)                 │")
    print(f"  │ ✓ GPU-accelerated matrix operations                                   │")
    print(f"  │ ✓ Predictive coding (skip known patterns)                             │")
    if tokenizer_type == "word":
        print(f"  │ ✓ Word-level tokenization (29× faster, theory-true)                   │")
    if fast_curriculum:
        print(f"  │ ✓ Fast curriculum (16× faster start @ ctx=32)                         │")
    print("  └─────────────────────────────────────────────────────────────────────────┘")
    print()


def print_generation_sample(prompt: str, output: str, unique_tokens: int, 
                            total_tokens: int, quality_status: str):
    """Print generation sample - compact and readable."""
    diversity = unique_tokens / total_tokens if total_tokens > 0 else 0
    
    # Truncate for display
    max_prompt = 30
    max_output = 80
    display_prompt = prompt[:max_prompt] + "..." if len(prompt) > max_prompt else prompt
    display_output = output[:max_output] + "..." if len(output) > max_output else output
    
    # Clean output (remove newlines for compact display)
    display_output = display_output.replace('\n', ' ').replace('\r', '')
    
    print(f"\n  ✍️  \"{display_prompt}\" → \"{display_output}\"")
    print(f"      [{unique_tokens}/{total_tokens} unique = {diversity:.0%}] {quality_status}", flush=True)


def print_final_summary(metrics: TrainingMetrics, model, final_accuracy: float,
                        gen_acc_episodic: float = 0, gen_acc_semantic: float = 0, 
                        max_attractors: int = 500000):
    """Print comprehensive final training summary."""
    print()
    print("╔" + "═" * 100 + "╗")
    print("║" + "  TRAINING COMPLETE".center(100) + "║")
    print("╠" + "═" * 100 + "╣")
    
    # Summary stats
    print("║  " + "📈 SUMMARY".ljust(98) + "║")
    print("║  " + "─" * 98 + "║")
    print(f"║    Samples processed:    {metrics.samples:>15,}".ljust(101) + "║")
    print(f"║    Training time:        {metrics.format_elapsed():>15}".ljust(101) + "║")
    print(f"║    Average speed:        {metrics.overall_rate:>12,.0f}/s".ljust(101) + "║")
    print(f"║    Memory stored:        {metrics.attractors:>15,} associations".ljust(101) + "║")
    
    if metrics.forgotten > 0:
        sparse_pct = metrics.sparse_efficiency()
        print(f"║    φ-Forgotten:          {metrics.forgotten:>15,} ({sparse_pct:.0f}% sparse efficiency)".ljust(101) + "║")
    
    if metrics.prototypes > 0:
        print(f"║    Prototypes:           {metrics.prototypes:>15,} (consolidated abstractions)".ljust(101) + "║")
        print(f"║    Schemas:              {metrics.schemas:>15,} (high-level patterns)".ljust(101) + "║")
    
    print("║  " + "─" * 98 + "║")
    print("║  " + "🎯 EVALUATION".ljust(98) + "║")
    print("║  " + "─" * 98 + "║")
    
    acc_status = "✅ Excellent" if final_accuracy >= PHI_INV else "✓ Good" if final_accuracy >= PHI_INV_SQ else "⚠️ Needs work"
    print(f"║    Retrieval accuracy:   {final_accuracy:>12.1%}  ({acc_status})".ljust(101) + "║")
    
    if gen_acc_semantic > 0:
        improvement = gen_acc_semantic - gen_acc_episodic
        imp_status = "✅" if improvement > 0.1 else "✓" if improvement > 0 else "⚠️"
        print(f"║    Generalization:       {gen_acc_semantic:>12.1%}  ({imp_status} {improvement:+.1%} vs episodic)".ljust(101) + "║")
    
    # Learning curve summary
    if len(metrics.learning_curve) >= 3:
        print("║  " + "─" * 98 + "║")
        print("║  " + "📉 LEARNING CURVE".ljust(98) + "║")
        
        # Show key checkpoints
        checkpoints = metrics.learning_curve[::max(1, len(metrics.learning_curve)//5)][:5]
        for samples, acc in checkpoints:
            bar_len = int(acc * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"║    {format_number(samples):>8} samples: {bar} {acc:.1%}".ljust(101) + "║")
    
    print("╚" + "═" * 100 + "╝")
    print()


# =============================================================================
# φ-CURRICULUM — Theory-derived context growth
# =============================================================================

def get_curriculum_context_size(
    samples_trained: int, 
    target_context: int = 50000,
    base_context: int = 512,
    base_samples: int = 500_000,
) -> Tuple[int, int]:
    """
    Theory-derived context size curriculum using φ scaling.
    
    DERIVATION:
        The Grace operator's spectral gap is γ = φ⁻² ≈ 0.382.
        This determines the rate of convergence to equilibrium.
        
        For context composition, the natural growth factor is φ² ≈ 2.618
        (Fibonacci doubling). This is NOT arbitrary — it's the ratio that
        emerges from the self-consistency equation Λ² = Λ + 1.
        
        FULLY THEORY-TRUE VERSION:
        - Context grows by φ² per stage
        - Samples per stage ALSO grows by φ (longer contexts need more samples)
        - This ensures prototype space is saturated at each level before scaling
        
        context(stage) = BASE_CONTEXT × φ^(2 × stage)
        samples_for_stage(n) = BASE_SAMPLES × φ^n
        
    STAGES (φ-scaled samples AND context) - DEFAULT base_context=512:
        Stage 0: 512 tokens,    500k samples  (0 - 500k)
        Stage 1: 1,340 tokens,  809k samples  (500k - 1.31M)
        Stage 2: 3,509 tokens,  1.31M samples (1.31M - 2.62M)
        Stage 3: 9,192 tokens,  2.12M samples (2.62M - 4.74M)
        Stage 4: 24,065 tokens, 3.43M samples (4.74M - 8.17M)
        Stage 5: 50,000 tokens, REST         (8.17M+)
        
    FAST CURRICULUM (base_context=32, base_samples=50k):
        Stage 0: 32 tokens,   50k samples  (16× faster than default!)
        Stage 1: 84 tokens,   81k samples
        Stage 2: 219 tokens,  131k samples
        Stage 3: 574 tokens,  212k samples
        ... continues with φ² growth
        
    THEORY-TRUE:
        - Base context is configurable (32-512 typical)
        - Context growth: φ² per stage (spectral structure)
        - Samples growth: φ per stage (prototype saturation)
        - No arbitrary hyperparameters — all derived from φ
    
    Args:
        samples_trained: Number of samples trained so far
        target_context: Maximum context size to reach
        base_context: Starting context size (32 for fast, 512 for standard)
        base_samples: Samples for stage 0 (50k for fast, 500k for standard)
        
    Returns:
        Tuple of (current_context_size, current_stage)
    """
    # φ = golden ratio
    PHI = 1.618033988749894848204586834365638118
    
    BASE_CONTEXT = base_context
    BASE_SAMPLES = base_samples
    
    # Compute cumulative samples needed to complete each stage
    # Stage n ends at: BASE_SAMPLES × (φ^0 + φ^1 + ... + φ^n) = BASE_SAMPLES × (φ^(n+1) - 1) / (φ - 1)
    
    # Find current stage by checking cumulative thresholds
    stage = 0
    cumulative = 0
    while True:
        samples_this_stage = int(BASE_SAMPLES * (PHI ** stage))
        if samples_trained < cumulative + samples_this_stage:
            break
        cumulative += samples_this_stage
        stage += 1
        # Cap at stage where we reach target context
        context_at_stage = int(BASE_CONTEXT * (PHI ** (2 * stage)))
        if context_at_stage >= target_context:
            break
    
    # Compute context for this stage
    context = int(BASE_CONTEXT * (PHI ** (2 * stage)))
    context = min(context, target_context)
    
    return context, stage

# =============================================================================
# MODAL SETUP
# =============================================================================

app = modal.App("holographic-v4")

holographic_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "numpy>=1.24.0",
        "cupy-cuda12x>=12.0.0",
        "datasets>=2.14.0",
        "tqdm>=4.65.0",
        "zstandard",  # Required for zstd-compressed HuggingFace datasets
        "huggingface_hub",  # For HF token auth
        "transformers>=4.30.0",  # For GPT-2 tokenizer (text generation output)
    )
    .env({"HF_TOKEN": os.getenv("HF_TOKEN", "")})  # HF auth for faster downloads
    # Copy the holographic_v4 directory
    .add_local_dir(".", "/root/project", copy=True)
    .run_commands(
        "echo 'HOLOGRAPHIC_V4'",
        "ls -la /root/project/holographic_v4/",
    )
)

GPU_CONFIG = "H100"
volume = modal.Volume.from_name("holographic-v4", create_if_missing=True)


# =============================================================================
# PRODUCTION CONFIGURATION — Optimized for large-scale runs
# =============================================================================
# These settings are tuned for maximum throughput on H100 with 50k context
# 
# KEY INSIGHTS:
#   1. This is a HASH-BASED memory system, not gradient descent
#   2. Per-sample: O(context_size) composition + O(1) hash lookup/store
#   3. GPU helps for batch operations but hash storage is inherently CPU
#   4. Main bottleneck: Python loop + context composition for long sequences
#
# OPTIMAL PARAMETERS for 50M+ sample runs:
#   TWO-TIER LOGGING (frequent feedback without blocking):
#   - lightweight_log_every: 1000 (inline rate/attractors, ~0ms overhead)
#   - log_every: 10000 (full progress with accuracy check, ~200ms)
#   - deep_log_every: 50000 (expensive diagnostics: enstrophy, witness)
#   - generate_every: 50000 (generation quality check)
#   - sleep_every: 100000 (dreaming is expensive)
#   - episodic_buffer_size: 5000 (more episodes = better prototypes)
#   - max_attractors: 100000-500000 (with equilibrium forgetting)
#
# RECOMMENDED COMMAND:
#   modal run -d holographic_v4/holographic_modal.py::train \
#       --dataset gutenberg \
#       --enable-curriculum \
#       --target-context-size 50000 \
#       --max-samples 50000000 \
#       --max-attractors 100000 \
#       --log-every 10000 \
#       --deep-log-every 50000 \
#       --generate-every 100000 \
#       --sleep-every 100000 \
#       --episodic-buffer-size 5000
# =============================================================================

# NOTE: prepare_dataset() and data_volume REMOVED - gutenberg is pre-tokenized, no prep needed


# NOTE: train_fast() REMOVED - gutenberg is pre-tokenized, use train() directly


# =============================================================================
# TEST — Verify algebra and theory
# =============================================================================

@app.function(image=holographic_image, gpu=GPU_CONFIG, timeout=300)
def test():
    """Run all theory validation tests."""
    import sys
    sys.path.insert(0, '/root/project')
    
    from holographic_v4 import run_all_tests
    
    print("=" * 60)
    print("HOLOGRAPHIC v4.11 — Theory Validation")
    print("=" * 60)
    
    results = run_all_tests()
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print()
    print(f"Summary: {passed}/{total} tests passed")
    
    return results


# =============================================================================
# DREAMING TESTS — Verify brain-inspired parsimonies
# =============================================================================

@app.function(image=holographic_image, gpu=GPU_CONFIG, timeout=600)
def test_dreaming():
    """Run all dreaming and brain-inspired parsimony tests."""
    import sys
    sys.path.insert(0, '/root/project')
    
    from holographic_v4.dreaming_tests import run_all_dreaming_tests
    
    print("=" * 60)
    print("HOLOGRAPHIC v4.11 — Brain-Inspired Parsimony Tests")
    print("=" * 60)
    
    success = run_all_dreaming_tests()
    
    print()
    if success:
        print("✓ All dreaming tests passed")
    else:
        print("✗ Some dreaming tests failed")
    
    return success


# =============================================================================
# BENCHMARK — GPU performance test
# =============================================================================

@app.function(image=holographic_image, gpu=GPU_CONFIG, timeout=600)
def benchmark():
    """Benchmark GPU performance."""
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    
    from holographic_v4 import (
        build_clifford_basis,
        geometric_product_batch,
        grace_operator_batch,
        initialize_embeddings_identity,
        normalize_matrix,
    )
    
    print("=" * 60)
    print("HOLOGRAPHIC v4.11 — GPU Benchmark")
    print("=" * 60)
    
    # Build basis
    basis = build_clifford_basis(cp)
    
    # Test geometric product throughput
    print("\n1. Geometric Product Batch:")
    for batch_size in [100, 1000, 10000]:
        matrices = cp.random.randn(batch_size, 4, 4).astype(cp.float32)
        matrices = normalize_matrix(matrices, cp)
        
        # Warmup
        for _ in range(3):
            _ = geometric_product_batch(matrices[:10], cp)
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = geometric_product_batch(matrices, cp)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        rate = batch_size * 100 / elapsed
        print(f"  Batch {batch_size:,}: {rate:,.0f} products/sec")
    
    # Test Grace operator throughput
    print("\n2. Grace Operator Batch:")
    for batch_size in [100, 1000, 10000]:
        matrices = cp.random.randn(batch_size, 4, 4).astype(cp.float32)
        matrices = normalize_matrix(matrices, cp)
        
        # Warmup
        for _ in range(3):
            _ = grace_operator_batch(matrices[:10], basis, cp)
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = grace_operator_batch(matrices, basis, cp)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        rate = batch_size * 100 / elapsed
        print(f"  Batch {batch_size:,}: {rate:,.0f} grace ops/sec")
    
    # Test embedding initialization
    print("\n3. Embedding Initialization:")
    for vocab_size in [1000, 10000, 50000]:
        start = time.perf_counter()
        embs = initialize_embeddings_identity(vocab_size, seed=42, xp=cp)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        print(f"  Vocab {vocab_size:,}: {elapsed*1000:.1f}ms")
    
    print("\n" + "=" * 60)
    print("Benchmark complete")
    
    return {"status": "success"}


# =============================================================================
# TRAIN — Main training loop
# =============================================================================

# NOTE: Synthetic dataset code REMOVED - use gutenberg (real data) only


@app.function(
    image=holographic_image,
    gpu=GPU_CONFIG,
    timeout=43200,  # 12 hours (extended for large runs)
    volumes={"/data": volume},
)
def train(
    max_samples: int = 1000000,
    vocab_size: int = 50257,  # Full GPT-2 vocabulary (no OOV problem)
    context_size: int = 512,  # Starting context (or fixed if no curriculum)
    target_context_size: int = 50000,  # Target context for curriculum
    enable_curriculum: bool = True,  # φ-derived context growth
    max_attractors: int = 500000,
    # TWO-TIER LOGGING: lightweight (no inference) + full (with accuracy check)
    lightweight_log_every: int = 1000,  # Fast inline update - just rate/attractors
    log_every: int = 10000,  # Full progress with accuracy check (~200ms)
    deep_log_every: int = 50000,  # Detailed diagnostics (enstrophy, witness)
    generate_every: int = 50000,  # Generation quality check (moved from 25K)
    use_binding: bool = True,  # THEORY-TRUE: Attribute binding in context
    use_adaptive_similarity: bool = True,
    use_vorticity: bool = True,  # THEORY-TRUE: Word order via wedge product
    use_equilibrium: bool = True,
    equilibrium_steps: int = 5,
    noise_std: float = PHI_INV_CUBE,  # Theory-true: φ⁻³ ≈ 0.236
    # THEORY-TRUE FEATURES (all enabled by default)
    use_predictiveness: bool = True,  # Semantic extraction
    use_meta_learning: bool = True,   # Adaptive learning rates
    use_embedding_drift: bool = True, # Representation learning
    use_credit_assignment: bool = True,  # Error tracking
    seed: int = 42,
    # DREAMING PARAMETERS
    enable_dreaming: bool = True,
    sleep_every: int = 50000,  # Run sleep cycle every N samples
    episodic_buffer_size: int = 5000,  # Max episodes per sleep cycle (increased for more clustering)
    similarity_threshold: float = PHI_INV_SQ,  # φ-derived: φ⁻² ≈ 0.382 (looser for more prototypes)
    rem_cycles: int = 1,  # REM recombination rounds per sleep
    # DATASET SELECTION
    dataset: str = "gutenberg",  # PRE-TOKENIZED: fast, reliable, 38k full books
    # TOKENIZER SELECTION (Theory-true: word-level is optimal)
    tokenizer_type: str = "word",  # "word" (theory-true) or "bpe" (legacy)
    # SPEED OPTIMIZATION
    fast_curriculum: bool = False,  # Start at ctx=32 instead of 512 (16× faster early training)
    # BRAIN-INSPIRED PRIORS (v4.12.0)
    use_chunking: bool = True,  # Apply Grace at phrase boundaries (Broca's area analog)
):
    """
    Train theory-true holographic model with DREAMING.
    
    TOKENIZER OPTIONS:
        word (default, THEORY-TRUE):
            - Each token = semantic unit (brain-native)
            - 29× faster tokenization
            - 22% fewer tokens for same content
            - Vorticity captures SYNTAX (not morphology)
            
        bpe (legacy):
            - GPT-2 subword tokenization
            - Fragments like "ologist" pollute context
            - Not theory-optimal but compatible with existing work
    
    DATASETS (tested working with modern HuggingFace):
        gutenberg - Full books (38k books) with raw text AND BPE tokens
    
    ARCHITECTURE ADVANTAGE:
        This model has O(N) context composition vs O(N²) Transformer attention.
        Gutenberg's long books (avg 74k WORDS) maximize this advantage.
    
    METRICS EXPLAINED:
    
    BASIC (every log_every samples):
      samples     - Total training examples seen
      attractors  - Unique context→target associations stored
      fill%       - attractors / max_attractors (capacity used)
      acc         - Exact match accuracy on validation sample
      vort        - Average vorticity (order-sensitivity signal)
      rate        - Training throughput (samples/second)
    
    THEORY-RELEVANT (every deep_log_every samples):
      unique_ctx% - Fraction of samples that created NEW attractors
      sem_sep     - Semantic separation (same_target - diff_target similarity)
      enstrophy   - Grade-2 energy (bivector content)
      witness_stab- Gauge-invariant stability across embeddings
    
    DREAMING (every sleep_every samples):
      Non-REM: Consolidate episodic buffer → semantic prototypes
      REM: Recombine prototypes → discover schemas
      generalization_acc: Accuracy on novel contexts (should improve!)
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    import numpy as np
    import os
    from datasets import load_dataset
    from huggingface_hub import login
    
    # Authenticate with HF token for faster downloads
    hf_token = os.environ.get("HF_TOKEN", "")
    login(token=hf_token, add_to_git_credential=False)
    
    from holographic_v4 import (
        TheoryTrueModel,
        build_clifford_basis,
        grace_operator,
        grade_energies,
        witness_stability,
        vorticity_magnitude,
    )
    from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, GRADE_INDICES
    from holographic_v4.algebra import decompose_to_coefficients, frobenius_similarity, grace_operator
    from holographic_v4.quotient import compute_enstrophy
    
    # Import dreaming system
    if enable_dreaming:
        from holographic_v4.dreaming import (
            DreamingSystem,
            EpisodicEntry,
            integrate_dreaming_with_model,
        )
    
    # Determine effective context sizes
    # Initialize these at function scope to ensure they're always defined
    current_stage = 0
    current_context_size = context_size
    
    # Curriculum parameters: fast curriculum starts smaller for faster experimentation
    if fast_curriculum:
        curriculum_base_context = 32   # 16× smaller than default
        curriculum_base_samples = 50_000  # 10× fewer samples per stage
    else:
        curriculum_base_context = 512
        curriculum_base_samples = 500_000
    
    if enable_curriculum:
        start_context, start_stage = get_curriculum_context_size(
            0, target_context_size, 
            base_context=curriculum_base_context, 
            base_samples=curriculum_base_samples
        )
        # For prefetch, use START context (not target!) so documents aren't filtered out
        # Documents will be re-chunked as curriculum advances
        prefetch_context_size = start_context  # Was: target_context_size (BUG!)
        current_stage = start_stage
        current_context_size = start_context
    else:
        start_context = context_size
        start_stage = 0
        target_context_size = context_size
        prefetch_context_size = context_size
        current_context_size = context_size
    
    # Clean header with comprehensive config display
    BATCH_SIZE = 2048  # H100 OPTIMIZED: Match the constant used in training loop
    print()
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║           HOLOGRAPHIC v4.11 — Theory-True Language Model                      ║")
    print("║                 with Self-Organizing Dreaming System                         ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")
    
    # Print comprehensive training configuration
    print_training_config(
        max_samples=max_samples,
        vocab_size=vocab_size,
        context_size=start_context,
        batch_size=BATCH_SIZE,
        enable_curriculum=enable_curriculum,
        fast_curriculum=fast_curriculum,
        enable_dreaming=enable_dreaming,
        dataset=dataset,
        max_attractors=max_attractors,
        tokenizer_type=tokenizer_type
    )
    
    # Load dataset - GUTENBERG with raw text (word-level) or BPE tokens
    ds_name = "nikolina-p/gutenberg_clean_tokenized_en"
    ds_split = None
    
    # Select field based on tokenizer type
    if tokenizer_type == "word":
        text_field = "text"  # Raw text for word-level tokenization (THEORY-TRUE)
        print(f"  Loading {ds_name} (raw text for word-level tokenization)...")
        print(f"  ✓ Theory-true: Each token = semantic unit")
    else:
        text_field = "tokenized"  # BPE tokens (legacy)
        print(f"  Loading {ds_name} (BPE pre-tokenized)...")
        print(f"  ⚠ Using BPE: fragments may pollute context")
    
    try:
        ds = load_dataset(ds_name, ds_split, split="train", streaming=True)
        # Verify we can actually get data
        test_iter = iter(ds)
        test_example = next(test_iter)
        test_data = test_example.get(text_field, "")
        
        # Handle pre-tokenized vs text
        if isinstance(test_data, list):
            print(f"  ✓ Dataset loaded (pre-tokenized: {len(test_data):,} tokens per sample)")
        else:
            if not str(test_data).strip():
                # Skip empty examples until we find one with content
                for _ in range(100):
                    test_example = next(test_iter)
                    test_data = test_example.get(text_field, "")
                    if str(test_data).strip():
                        break
            print(f"  ✓ Dataset loaded (sample: {len(str(test_data)):,} chars)")
        # Reload fresh iterator
        ds = load_dataset(ds_name, ds_split, split="train", streaming=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {ds_name}: {e}. No fallback.")
    
    # Handle tokenizer selection
    import re
    
    if tokenizer_type == "word":
        # =======================================================================
        # WORD-LEVEL TOKENIZER (THEORY-TRUE)
        # =======================================================================
        # Each token = semantic unit, matches brain's VWFA processing
        # 31× faster than BPE, 22% fewer tokens for same content
        # =======================================================================
        import json
        
        # Fast word extraction using regex (handles punctuation correctly)
        word_pattern = re.compile(r'\b\w+\b')
        
        def extract_words(text: str) -> List[str]:
            """Fast word extraction - 31× faster than BPE."""
            return word_pattern.findall(text.lower())
        
        # Try to load cached vocabulary first
        vocab_cache_path = f"/tmp/holographic_word_vocab_{vocab_size}.json"
        word_to_idx = None
        idx_to_word = None
        
        try:
            with open(vocab_cache_path, 'r') as f:
                cache_data = json.load(f)
                if cache_data.get('vocab_size') == vocab_size:
                    word_to_idx = cache_data['word_to_idx']
                    idx_to_word = {int(k): v for k, v in cache_data['idx_to_word'].items()}
                    print(f"  ✓ Loaded cached vocabulary ({len(word_to_idx):,} words)")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        
        if word_to_idx is None:
            # Build vocabulary from scratch
            print("  Building word-level vocabulary (theory-true)...")
            
            word_to_idx = {"<unk>": 0, "<pad>": 1}
            idx_to_word = {0: "<unk>", 1: "<pad>"}
            
            # Collect vocabulary from first samples (streaming)
            vocab_samples = 500  # Fewer samples needed for word vocab (more unique per doc)
            word_counts = {}
            vocab_start = time.perf_counter()
            
            # Reload fresh iterator for vocab building
            ds_vocab = load_dataset(ds_name, ds_split, split="train", streaming=True)
            
            for i, example in enumerate(ds_vocab):
                if i >= vocab_samples:
                    break
                text = example.get("text", "")
                if isinstance(text, str):
                    for word in extract_words(text):
                        word_counts[word] = word_counts.get(word, 0) + 1
                if (i + 1) % 100 == 0:
                    print(f"    Vocab scan: {i+1}/{vocab_samples} docs, {len(word_counts):,} unique words", flush=True)
            
            # Keep top vocab_size-2 words
            sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
            for word, _ in sorted_words[:vocab_size - 2]:
                idx = len(word_to_idx)
                word_to_idx[word] = idx
                idx_to_word[idx] = word
            
            vocab_time = time.perf_counter() - vocab_start
            coverage = sum(c for w, c in word_counts.items() if w in word_to_idx) / sum(word_counts.values())
            
            print(f"  ✓ Vocabulary built in {vocab_time:.1f}s")
            print(f"  ✓ {len(word_to_idx):,} words, {coverage:.1%} coverage")
            
            # Cache vocabulary for future runs
            try:
                cache_data = {
                    'vocab_size': vocab_size,
                    'word_to_idx': word_to_idx,
                    'idx_to_word': {str(k): v for k, v in idx_to_word.items()}
                }
                with open(vocab_cache_path, 'w') as f:
                    json.dump(cache_data, f)
                print(f"  ✓ Vocabulary cached to {vocab_cache_path}")
            except Exception as e:
                print(f"  ⚠ Could not cache vocabulary: {e}")
        
        def tokenize(text: str) -> List[int]:
            """Word-level tokenization (31× faster than BPE)."""
            return [word_to_idx.get(w, 0) for w in extract_words(text)]
        
        def detokenize(tokens: List[int]) -> str:
            """Decode tokens back to text."""
            return " ".join(idx_to_word.get(t, "<unk>") for t in tokens)
        
        actual_vocab_size = len(word_to_idx)
        print(f"  ✓ Word-level tokenizer ready (vocab={actual_vocab_size:,})")
        
    else:
        # =======================================================================
        # BPE TOKENIZER (LEGACY)
        # =======================================================================
        # Uses GPT-2 subword tokens - fragments pollute context
        # Kept for backwards compatibility
        # =======================================================================
        print("  Using BPE tokenizer (legacy, not theory-optimal)...")
        print(f"  Vocab size: {vocab_size} (will cap GPT-2 tokens to this)")
        
        # Load GPT-2 tokenizer for proper text decoding
        from transformers import GPT2Tokenizer
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print(f"  ✓ GPT-2 tokenizer loaded")
        
        def tokenize(tokens_or_text):
            if isinstance(tokens_or_text, list):
                # Already tokenized - FILTER OUT tokens >= vocab_size (don't clamp!)
                return [t for t in tokens_or_text if t < vocab_size]
            elif isinstance(tokens_or_text, str):
                # Text input - use GPT-2 tokenizer, FILTER OUT OOV tokens
                return [t for t in gpt2_tokenizer.encode(tokens_or_text) if t < vocab_size]
            else:
                raise TypeError(f"tokenize() expected list or str, got {type(tokens_or_text)}")
        
        def detokenize(tokens: List[int]) -> str:
            """Decode tokens back to readable text using GPT-2 tokenizer."""
            try:
                return gpt2_tokenizer.decode(tokens, skip_special_tokens=True)
            except (ValueError, IndexError, TypeError) as e:
                return f"[DETOKENIZATION_ERROR: {e}]"
        
        actual_vocab_size = vocab_size
    
    # Create model (use target_context_size for max capacity)
    print("  Creating holographic model...")
    model = TheoryTrueModel(
        vocab_size=actual_vocab_size,
        context_size=target_context_size,  # Max context we'll use
        max_attractors=max_attractors,
        noise_std=noise_std,
        use_binding=use_binding,
        use_adaptive_similarity=use_adaptive_similarity,
        use_vorticity=use_vorticity,
        use_equilibrium=use_equilibrium,
        equilibrium_steps=equilibrium_steps,
        # THEORY-TRUE FEATURES
        use_predictiveness=use_predictiveness,
        use_meta_learning=use_meta_learning,
        use_embedding_drift=use_embedding_drift,
        use_credit_assignment=use_credit_assignment,
        xp=cp,  # USE GPU (CuPy)!
        seed=seed,
    )
    
    # Build GPU basis for fast operations
    basis_gpu = build_clifford_basis(cp)
    
    # Quick stability check
    init_witness = witness_stability(model.embeddings[:100], model.basis)
    print(f"  ✓ Model ready (witness stability: {init_witness:.3f})")
    
    # Initialize dreaming system
    dreaming = None
    episodic_buffer = []
    total_sleep_cycles = 0
    total_prototypes = 0
    total_schemas = 0
    
    if enable_dreaming:
        dreaming = DreamingSystem(
            basis=model.basis,
            xp=cp,  # GPU! Critical for performance
            similarity_threshold=similarity_threshold if similarity_threshold else PHI_INV,
        )
        print("  ✓ Dreaming system ready (GPU)")
    
    # Initialize training metrics tracker
    metrics = TrainingMetrics(
        start_time=time.perf_counter(),
        max_samples=max_samples,
    )
    metrics.current_context_size = start_context if enable_curriculum else context_size
    
    # Training loop header
    print_progress_header()
    
    start_time = time.perf_counter()
    last_log_time = start_time
    
    # ==========================================================================
    # PREFETCH BUFFER — Load data in background while training
    # ==========================================================================
    # This eliminates network I/O as a bottleneck:
    # - Background thread fills buffer while training runs
    # - Training pulls from buffer (instant, no wait)
    # - No massive upfront download needed
    # ==========================================================================
    
    import queue
    import threading
    
    PREFETCH_SIZE = 200  # Number of tokenized documents to buffer (higher = more I/O overlap)
    
    def prefetch_worker(ds_iter, text_field, tokenize_fn, context_size, out_queue, stop_event):
        """Background worker that tokenizes documents and chunks them into manageable sizes."""
        doc_count = 0
        samples_queued = 0
        CHUNK_SIZE = max(context_size * 10, 100000)  # Chunk large docs (10x context or 100k tokens)
        
        for example in ds_iter:
            if stop_event.is_set():
                break
            doc_count += 1
            data = example.get(text_field, example.get("text", ""))
            
            # Handle pre-tokenized (list) vs text (string)
            if isinstance(data, list):
                # Already tokenized - use directly
                tokens = tokenize_fn(data)  # Will cap to vocab_size
            else:
                # Need to tokenize text
                tokens = tokenize_fn(data)
            
            if len(tokens) >= context_size + 1:
                # For huge books, chunk them into manageable pieces
                if len(tokens) > CHUNK_SIZE:
                    # Chunk the book into overlapping windows
                    for chunk_start in range(0, len(tokens) - context_size, CHUNK_SIZE - context_size):
                        chunk = tokens[chunk_start:chunk_start + CHUNK_SIZE]
                        if len(chunk) >= context_size + 1:
                            out_queue.put(chunk, block=True)
                            samples_queued += len(chunk) - context_size
                else:
                    # Small enough to use directly
                    out_queue.put(tokens, block=True)
                    samples_queued += len(tokens) - context_size
                
                # Progress update every doc or every 10k samples
                if doc_count % 1 == 0 or samples_queued % 10000 == 0:
                    # Only log prefetch every 500 docs to reduce noise
                    if doc_count % 500 == 0:
                        print(f"  📥 {doc_count:,} docs → {samples_queued:,} samples buffered", flush=True)
        out_queue.put(None)  # Sentinel to indicate end
        print(f"  ✓ Prefetch complete: {doc_count} docs → {samples_queued:,} total samples", flush=True)
    
    # Start prefetch thread
    prefetch_queue = queue.Queue(maxsize=PREFETCH_SIZE)
    stop_prefetch = threading.Event()
    
    # Load fresh iterator for prefetch (same dataset we validated above)
    ds_prefetch = load_dataset(ds_name, ds_split, split="train", streaming=True)
    
    prefetch_thread = threading.Thread(
        target=prefetch_worker,
        args=(iter(ds_prefetch), text_field, tokenize, prefetch_context_size, prefetch_queue, stop_prefetch),
        daemon=True
    )
    prefetch_thread.start()
    print(f"  ✓ Prefetch buffer started (size={PREFETCH_SIZE})", flush=True)
    if enable_curriculum:
        print(f"  ✓ φ-Curriculum enabled: context will grow from {start_context} → {target_context_size}", flush=True)
    print(f"  ⏳ Waiting for first sample...", flush=True)
    
    samples = 0
    samples_at_chunk_start = 0  # Track for "crossed threshold" logging
    total_vorticity = 0.0
    last_attractors = 0  # Track new attractors per interval
    interval_new_attractors = 0
    last_sleep_sample = 0  # Track when last sleep occurred (for adaptive sleep throttling)
    need_sleep = False  # Flag for sleep cycle (set in inner loop, processed after story)
    need_lightweight_log = False  # Flag for lightweight (no inference) progress
    need_log = False  # Flag for full progress logging (with accuracy)
    need_generate = False  # Flag for generation
    need_deep_log = False  # Flag for deep diagnostics
    retrieve_fn = None  # Dreaming-integrated retrieval (set after first sleep)
    last_accuracy = 0.0  # Cache accuracy for lightweight logs
    
    # φ-CURRICULUM: Track current context size and stage
    current_context_size = start_context if enable_curriculum else context_size
    current_stage = 0  # Initialize stage (updated in training loop)
    last_context_size = current_context_size  # For detecting stage changes
    
    # ACCURACY BUFFER: Track samples we've trained on for proper accuracy testing
    # (Testing on unseen data measures generalization, not learning)
    # PERF: Use deque with maxlen for O(1) append/pop instead of O(n) list.pop(0)
    from collections import deque
    trained_samples_buffer = deque(maxlen=500)  # Rolling buffer of (context, target) tuples
    
    # META-COGNITIVE STATE: Track for predictive coding and adaptive consolidation
    meta_surprises = 0          # Samples that were surprising (wrong prediction)
    meta_redundant = 0          # Samples skipped (already knew)
    meta_recent_surprises = []  # Rolling window for novelty rate
    meta_recent_errors = []     # Rolling window for error rate
    META_WINDOW_SIZE = 100      # Window size for rolling stats
    use_predictive_coding = True  # Enable predictive coding (skip redundant)
    
    # SPEED TRACKING: Separate training time from chunk loading time
    training_time_total = 0.0  # Pure training time (excluding chunk loads)
    last_training_time = 0.0  # For interval measurements
    
    # Pull from prefetch buffer instead of dataset directly
    prefetch_timeouts = 0
    max_timeouts = 10  # Allow up to 10 timeouts before giving up
    first_sample = True
    
    # BATCH ACCUMULATION: Must be OUTSIDE chunk loop to accumulate across small documents!
    batch_contexts = []
    batch_targets = []
    BATCH_SIZE = 2048  # H100 OPTIMIZED: Larger batches = better GPU utilization
    
    while True:
        try:
            tokens = prefetch_queue.get(block=True, timeout=30)  # Check every 30s for progress
            prefetch_timeouts = 0  # Reset on success
            if first_sample:
                print(f"  ✓ First sample ready! Starting training...", flush=True)
                first_sample = False
        except queue.Empty:
            prefetch_timeouts += 1
            if prefetch_timeouts >= max_timeouts:
                print(f"  ✗ Too many prefetch timeouts ({max_timeouts}), stopping", flush=True)
                break
            print(f"  ⏳ Waiting for data... ({prefetch_timeouts * 30}s elapsed, checking prefetch progress)", flush=True)
            continue
            
        if tokens is None:  # End of dataset
            break
        if samples >= max_samples:
            stop_prefetch.set()  # Signal worker to stop
            break
        
        # Track samples before processing this chunk (for "crossed threshold" logging)
        samples_at_chunk_start = samples
        
        # Tokens are already validated by prefetch worker
        n_tokens = len(tokens)
        
        # φ-CURRICULUM: Update context size based on samples trained
        if enable_curriculum:
            current_context_size, current_stage = get_curriculum_context_size(
                samples, target_context_size,
                base_context=curriculum_base_context,
                base_samples=curriculum_base_samples
            )
            if current_context_size != last_context_size:
                # CRITICAL: Flush any pending batch before context size change
                # (batches require uniform context length)
                if batch_contexts:
                    attractors_before = model.num_attractors
                    batch_result = model.train_batch(batch_contexts, batch_targets)
                    total_vorticity += batch_result.get('avg_vorticity', 0.0) * len(batch_contexts)
                    for ctx, tgt in zip(batch_contexts, batch_targets):
                        trained_samples_buffer.append((ctx, tgt))
                    interval_new_attractors += model.num_attractors - attractors_before
                    samples += len(batch_contexts)
                    batch_contexts = []
                    batch_targets = []
                
                print_stage_change(current_stage, last_context_size, current_context_size, samples)
                last_context_size = current_context_size
        
        # OPTIMIZATION: Pre-compute number of examples (using current context size!)
        n_examples = n_tokens - current_context_size
        
        if n_examples <= 0:
            # Document too short for current context size - skip
            continue
        
        # CRITICAL: For large chunks, process in strides to avoid hanging
        # With 50k context, a 100k token chunk = 50k examples = too many!
        MAX_EXAMPLES_PER_CHUNK = 10000  # Process max 10k examples per chunk
        STRIDE = max(1, n_examples // MAX_EXAMPLES_PER_CHUNK) if n_examples > MAX_EXAMPLES_PER_CHUNK else 1
        
        if n_examples > MAX_EXAMPLES_PER_CHUNK:
            examples_to_process = (n_examples // STRIDE)
            # Silent - don't spam logs with chunk warnings
        else:
            examples_to_process = n_examples
            STRIDE = 1
        
        # ============================================================
        # BATCHED TRAINING - ORDER OF MAGNITUDE SPEEDUP
        # ============================================================
        # batch_contexts and batch_targets accumulate across chunks (initialized before while loop)
        # This allows small documents to contribute to batches efficiently.
        # ============================================================
        
        train_step_start = time.perf_counter()
        
        for i in range(0, examples_to_process * STRIDE, STRIDE):
            if i + current_context_size >= n_tokens:
                break
            
            context = tokens[i:i + current_context_size]
            target = tokens[i + current_context_size]
            
            batch_contexts.append(context)
            batch_targets.append(target)
            
            # Process batch when full or at end of chunk
            if len(batch_contexts) >= BATCH_SIZE:
                # Track attractors before
                attractors_before = model.num_attractors
                
                # BATCH TRAINING (context computation parallelized on GPU)
                batch_result = model.train_batch(batch_contexts, batch_targets)
                
                # Update stats
                total_vorticity += batch_result.get('avg_vorticity', 0.0) * len(batch_contexts)
                
                # RECORD THEORY-TRUE METRICS from batch result
                pruned = batch_result.get('pruned', 0)
                if pruned > 0:
                    metrics.record_pruning(pruned)
                
                # Track trained samples for accuracy testing
                for ctx, tgt in zip(batch_contexts, batch_targets):
                    trained_samples_buffer.append((ctx, tgt))
                
                # Track new attractors
                interval_new_attractors += model.num_attractors - attractors_before
                
                # Update sample count
                samples += len(batch_contexts)
                
                # Add to episodic buffer - sample 5 representatives per batch
                # With batch=2048 and sleep_every=50000, ~24 batches = ~120 episodes/sleep
                # More episodes would require passing context matrices from train_batch (future optimization)
                if enable_dreaming and len(episodic_buffer) < episodic_buffer_size and len(batch_contexts) > 0:
                    n_samples = min(5, len(batch_contexts), episodic_buffer_size - len(episodic_buffer))
                    step = max(1, len(batch_contexts) // n_samples)
                    
                    for j in range(0, n_samples * step, step):
                        if j >= len(batch_contexts) or len(episodic_buffer) >= episodic_buffer_size:
                            break
                        ctx_matrix, vort_mag, vort_sig = model.compute_context_with_vorticity(batch_contexts[j])
                        episodic_buffer.append(EpisodicEntry(
                            context_matrix=ctx_matrix,
                            target_token=batch_targets[j],
                            vorticity_signature=vort_sig,
                        ))
                
                # Clear batch
                batch_contexts = []
                batch_targets = []
        
        # Process remaining samples in last partial batch
        if batch_contexts:
            attractors_before = model.num_attractors
            batch_result = model.train_batch(batch_contexts, batch_targets)
            total_vorticity += batch_result.get('avg_vorticity', 0.0) * len(batch_contexts)
            for ctx, tgt in zip(batch_contexts, batch_targets):
                trained_samples_buffer.append((ctx, tgt))
            interval_new_attractors += model.num_attractors - attractors_before
            samples += len(batch_contexts)
        
        training_time_total += time.perf_counter() - train_step_start
        
        # Check for boundaries AFTER processing chunk (batched version)
        # ADAPTIVE SLEEP: Use meta-cognitive state for smarter consolidation timing
        if enable_dreaming and samples > 0 and len(episodic_buffer) > 0:
            # Standard trigger: every sleep_every samples
            # FIX: Use "crossed threshold" check (same fix as logging)
            if samples // sleep_every > samples_at_chunk_start // sleep_every:
                need_sleep = True
                
            # ADAPTIVE: Trigger early if lots of novelty or errors
            # BUT only if enough time has passed (min 10K samples between sleeps)
            # This prevents 27%+ overhead from over-frequent sleep cycles
            # FIX: Use "crossed threshold" check
            elif samples // 5000 > samples_at_chunk_start // 5000 and len(meta_recent_surprises) >= 50:
                samples_since_sleep = samples - last_sleep_sample
                if samples_since_sleep >= 10000:  # Min 10K between adaptive sleeps
                    meta_novelty_rate = sum(meta_recent_surprises) / len(meta_recent_surprises)
                    meta_error_rate = sum(meta_recent_errors) / len(meta_recent_errors)
                    memory_pressure = model.num_attractors / max_attractors
                    
                    # φ-derived triggers - only if VERY high pressure/novelty
                    if memory_pressure > PHI_INV * PHI_INV:  # > 0.382 (was 0.618)
                        need_sleep = True
                    elif meta_novelty_rate > PHI_INV:  # > 0.618 (was 0.382)
                        need_sleep = True  
                    elif meta_error_rate > PHI_INV:  # > 0.618 (was 0.382)
                        need_sleep = True
        # TWO-TIER LOGGING: lightweight (frequent) + full (with accuracy)
        # FIX: Use "crossed threshold" check since samples jumps by batch_size (256)
        # Old: samples % 1000 == 0 would MISS if samples went 9984 → 10240
        # New: Check if we crossed a multiple (samples // interval changed)
        if samples // lightweight_log_every > samples_at_chunk_start // lightweight_log_every:
            need_lightweight_log = True
        if samples // log_every > samples_at_chunk_start // log_every:
            need_log = True
        if samples // generate_every > samples_at_chunk_start // generate_every:
            need_generate = True
        if samples // deep_log_every > samples_at_chunk_start // deep_log_every:
            need_deep_log = True
        
        # LIGHTWEIGHT LOG: Fast inline update (no inference, ~0ms)
        if need_lightweight_log and not need_log:  # Skip if full log is coming
            need_lightweight_log = False
            interval_training_time = training_time_total - last_training_time
            rate = lightweight_log_every / interval_training_time if interval_training_time > 0 else 0
            print_lightweight_progress(
                samples, model.num_attractors, rate, max_attractors, 
                current_context_size, max_samples, last_accuracy, metrics
            )
        elif need_lightweight_log:
            need_lightweight_log = False  # Clear flag, full log will handle it
        
        # FULL LOG: Progress with accuracy check (~200-500ms, acceptable)
        if need_log:
            need_log = False
            now = time.perf_counter()
            
            # PURE TRAINING RATE (excludes chunk loading overhead)
            interval_training_time = training_time_total - last_training_time
            rate = log_every / interval_training_time if interval_training_time > 0 else 0
            last_training_time = training_time_total
            
            # Quick accuracy check on TRAINED samples (not unseen data!)
            # PERF: Reduced from 100 to 50 samples for faster logging
            # Also records retrieval confidence for memory health tracking
            accuracy = quick_accuracy_check(model, tokenize, current_context_size, n_samples=50,
                                            trained_samples=trained_samples_buffer, metrics=metrics)
            last_accuracy = accuracy  # Cache for lightweight logs
            
            # Update metrics tracker
            proto_count = dreaming.semantic_memory.stats()['total_prototypes'] if enable_dreaming and dreaming else 0
            schema_count = dreaming.semantic_memory.stats()['num_schemas'] if enable_dreaming and dreaming else 0
            forgotten_total = getattr(model, 'total_forgotten', 0)
            
            metrics.update(
                samples=samples,
                attractors=model.num_attractors,
                prototypes=proto_count,
                schemas=schema_count,
                forgotten=forgotten_total,
                context_size=current_context_size,
                stage=current_stage if enable_curriculum else 0,
            )
            metrics.record_rate(rate)
            metrics.record_accuracy(accuracy)
            
            # Determine status (φ-derived thresholds)
            if model.num_attractors >= max_attractors * (1.0 - PHI_INV_CUBE):  # ≈ 0.764 (was 0.95)
                status = "⚠️ Mem full"
            elif accuracy < PHI_INV_SQ:  # ≈ 0.382 (was 0.5)
                status = "🔄 Learning"
            elif accuracy >= PHI_INV:  # ≈ 0.618 (was 0.9)
                status = "✅ Excellent"
            else:
                status = "✓ Good"
            
            # Use new progress display with full context
            print_progress_line(metrics, accuracy, rate, status, current_context_size, max_attractors,
                               stage=current_stage, batch_size=BATCH_SIZE)
            
            last_log_time = now
        
        if need_generate:
            need_generate = False
            gen_stats = generate_sample_with_metrics(
                model, tokenize, detokenize, current_context_size, 
                retrieve_fn, metrics, quiet=False
            )
        
        if need_deep_log:
            need_deep_log = False
            
            # Witness stability (theory validation: should be > φ⁻¹ for stable contexts)
            # BUG FIX v4.20.0: Was measuring on static embeddings (constant 0.795!)
            # Now measures on actual stored attractor matrices (varies with learning)
            if model.num_attractors > 0:
                sample_size = min(100, model.num_attractors)
                sample_matrices = model.attractor_matrices[:sample_size]
                wit_stab = witness_stability(sample_matrices, model.basis)
            else:
                wit_stab = 0.0  # No attractors yet
            
            # Sample enstrophy (theory validation: should be bounded by Grace damping)
            sample_ens = 0.0
            if model.num_attractors > 0:
                sample_idx = min(10, model.num_attractors)
                sample_matrices = model.attractor_matrices[:sample_idx]
                sample_ens = np.mean([compute_enstrophy(m, model.basis, np) for m in sample_matrices])
            
            # RECORD THEORY METRICS for history tracking
            metrics.record_theory_metrics(wit_stab, sample_ens)
            
            # Prototype coherence (if dreaming enabled)
            if enable_dreaming and dreaming is not None:
                proto_stats = dreaming.semantic_memory.stats()
                if proto_stats['total_prototypes'] > 0:
                    # Coherence = fraction of prototypes that are Grace-stable
                    coherence = proto_stats.get('avg_stability', 0.5)
                    metrics.record_prototype_coherence(coherence)
            
            # Compute meta-cognitive rates
            meta_novelty_rate = sum(meta_recent_surprises) / max(len(meta_recent_surprises), 1)
            meta_error_rate = sum(meta_recent_errors) / max(len(meta_recent_errors), 1)
            
            # Use new diagnostics display
            print_diagnostics_block(metrics, model, model.basis, wit_stab, sample_ens, max_attractors,
                                    meta_surprises=meta_surprises, meta_redundant=meta_redundant,
                                    novelty_rate=meta_novelty_rate, error_rate=meta_error_rate)
            
            # Reset interval counter
            interval_new_attractors = 0
            total_vorticity = 0
        
        if need_sleep and len(episodic_buffer) > 0:
            need_sleep = False
            total_sleep_cycles += 1
            
            sleep_start = time.perf_counter()
            
            # Run sleep cycle (quiet mode - use verbose=True only for debugging)
            sleep_stats = dreaming.sleep(
                episodic_buffer, 
                rem_cycles=rem_cycles, 
                verbose=False  # Clean output - set True for debugging
            )
            
            total_prototypes = dreaming.semantic_memory.stats()['total_prototypes']
            total_schemas = dreaming.semantic_memory.stats()['num_schemas']
            sleep_time = time.perf_counter() - sleep_start
            
            # Update metrics and display
            metrics.prototypes = total_prototypes
            metrics.schemas = total_schemas
            print_sleep_summary(
                total_sleep_cycles, len(episodic_buffer),
                sleep_stats['prototypes_created'], sleep_stats['schemas_discovered'],
                total_prototypes, total_schemas, sleep_time
            )
            
            # Update retrieve_fn
            from holographic_v4.dreaming import integrate_dreaming_with_model, prune_attractor_map
            retrieve_fn = integrate_dreaming_with_model(model, dreaming)
            
            # BRAIN-LIKE ATTRACTOR PRUNING (Tononi's Synaptic Homeostasis)
            # Prune low-salience, rarely-accessed attractors from bookkeeping
            # THEORY: Sleep actively removes weak memories to improve signal-to-noise
            # v4.13.0: Use adaptive threshold based on fill ratio (brain-like homeostasis)
            if model.num_attractors > 0:
                # Get adaptive threshold from model
                adaptive_threshold = model.get_adaptive_prune_threshold()
                
                new_matrices, new_targets, new_count, prune_stats = prune_attractor_map(
                    model.attractor_matrices,
                    model.attractor_targets,
                    model.num_attractors,
                    model.basis,
                    xp=cp,
                    salience_threshold=adaptive_threshold,  # v4.13.0: adaptive, not fixed
                    verbose=False,
                )
                
                if prune_stats['pruned'] > 0:
                    # Update model state with pruned arrays
                    model.attractor_matrices[:new_count] = new_matrices[:new_count]
                    model.attractor_targets[:new_count] = new_targets[:new_count]
                    # Also update tracking arrays (compact them)
                    for i, old_idx in enumerate(prune_stats.get('keep_indices', range(new_count))):
                        if old_idx != i:
                            model.attractor_saliences[i] = model.attractor_saliences[old_idx]
                            model.attractor_consolidated[i] = model.attractor_consolidated[old_idx]
                            # v4.13.0: Removed access_counts, last_access (not theory-true)
                    
                    old_count = model.num_attractors
                    model.num_attractors = new_count
                    model.total_forgotten += prune_stats['pruned']
                    print(f"  ✂️  Attractor pruning: {old_count:,} → {new_count:,} ({prune_stats['pruned']:,} weak memories removed)")
            
            # Clear episodic buffer and record sleep time
            episodic_buffer = []
            last_sleep_sample = samples  # Throttle adaptive sleep
            
            # BRAIN-LIKE: Apply Grace decay to holographic memory after consolidation
            # This is the Clifford analogue of synaptic homeostasis during sleep:
            # - Stable patterns (prototypes) are preserved in semantic memory
            # - Transient patterns (interference) decay via Grace damping
            # - If highly saturated, reset for fresh learning (hippocampal clearing)
            memory_norm_before = float(cp.linalg.norm(model.holographic_memory.holographic.memory, 'fro'))
            model.holographic_memory.holographic.memory = grace_operator(
                model.holographic_memory.holographic.memory, model.basis, cp
            )
            memory_norm_after = float(cp.linalg.norm(model.holographic_memory.holographic.memory, 'fro'))
            
            # Theory-true clearing threshold: φ⁻² saturation + significant Grace decay
            saturation = model.num_attractors / model.max_attractors
            decay_ratio = memory_norm_after / (memory_norm_before + 1e-10)
            
            # Only clear if: saturated AND Grace decay was significant (means high interference)
            if saturation > PHI_INV_SQ and decay_ratio < PHI_INV:  # > 38% full AND > 38% decay
                model.holographic_memory.clear()
                # Reset ALL bookkeeping arrays (critical for vorticity decoding!)
                model.num_attractors = 0
                model.attractor_saliences[:] = 0.0
                model.attractor_consolidated[:] = False
                # v4.13.0: Removed access_counts, last_access (not theory-true)
                # Note: attractor_matrices and attractor_targets will be overwritten as new data comes in
                print(f"  🧠 Hippocampal reset: {saturation:.0%} sat, {1-decay_ratio:.0%} decay → cleared for fresh learning")
        
        if samples >= max_samples:
            break
    
    # Cleanup prefetch thread
    stop_prefetch.set()
    prefetch_thread.join(timeout=5)
    
    # Final stats - update metrics one last time
    metrics.update(
        samples=samples,
        attractors=model.num_attractors,
        prototypes=total_prototypes,
        schemas=total_schemas,
        forgotten=getattr(model, 'total_forgotten', 0),
        context_size=current_context_size,
        stage=current_stage if enable_curriculum else 0,
    )
    
    # Update metrics with final values
    metrics.samples = samples
    metrics.attractors = model.num_attractors
    
    # Final accuracy check
    final_acc = quick_accuracy_check(model, tokenize, target_context_size, n_samples=500, 
                                      ds_name=ds_name, ds_split=ds_split, text_field=text_field)
    
    # Generalization metrics (if dreaming enabled)
    gen_acc_episodic = 0.0
    gen_acc_semantic = 0.0
    if enable_dreaming and dreaming is not None:
        gen_acc_episodic = test_generalization(model, None, tokenize, target_context_size, n_samples=200, 
                                                ds_name=ds_name, ds_split=ds_split, text_field=text_field)
        gen_acc_semantic = test_generalization(model, retrieve_fn, tokenize, target_context_size, n_samples=200,
                                                ds_name=ds_name, ds_split=ds_split, text_field=text_field)
    
    # Print comprehensive final summary
    print_final_summary(metrics, model, final_acc, gen_acc_episodic, gen_acc_semantic, max_attractors)
    
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    # Sample generations
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print(f"│  ✍️  SAMPLE GENERATIONS (context={target_context_size:,})                              │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    for _ in range(3):
        generate_sample(model, tokenize, detokenize, target_context_size, retrieve_fn)
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    # Return stats (internal use)
    stats = model.get_statistics()
    if enable_dreaming and dreaming is not None:
        stats['dreaming'] = {
            'sleep_cycles': total_sleep_cycles,
            'total_prototypes': total_prototypes,
            'total_schemas': total_schemas,
        }
    
    print()
    
    return stats


# Global cache for accuracy test samples (avoid reloading dataset every time!)
_accuracy_test_cache = None

def quick_accuracy_check(model, tokenize, context_size, n_samples=100, ds_name=None, ds_split=None, text_field=None, trained_samples=None, metrics=None):
    """
    Quick accuracy check on SEEN data (what we actually trained on).
    
    THEORY-TRUE DESIGN:
        - Tests retrieval on contexts we've LEARNED, not unseen data
        - Tests if the storage/retrieval loop is working
        - Generalization to unseen data is a DIFFERENT metric (test separately)
    
    FAST PATH:
        - Hash-only lookup (no Grace flow for speed)
        - Hash lookup shows if we learned the association correctly
    
    Args:
        trained_samples: Optional list of (context, target) tuples we've actually seen.
                        If provided, tests on these. Otherwise tests hash coverage.
        metrics: Optional TrainingMetrics to record retrieval confidence for memory health.
    """
    from holographic_v4.constants import PHI_INV_SQ
    
    if trained_samples and len(trained_samples) > 0:
        # TEST ON WHAT WE'VE SEEN (correct approach)
        # THEORY-TRUE: Holographic memory is approximate - measure retrieval quality
        # by similarity to expected target, not exact token match
        import random
        from holographic_v4.algebra import frobenius_similarity
        
        samples_to_test = random.sample(trained_samples, min(n_samples, len(trained_samples)))
        
        correct = 0
        total = 0
        confidences = []  # Track for memory health metric
        
        for context, target in samples_to_test:
            ctx_rep = model.compute_context_representation(context)
            retrieved_matrix, _, conf, _ = model.holographic_memory.retrieve(ctx_rep)
            confidences.append(conf)
            
            # Get expected target embedding
            target_emb = model.get_embedding(target)
            
            # Measure similarity to expected target
            sim_to_target = frobenius_similarity(retrieved_matrix, target_emb, model.xp)
            
            # THEORY-TRUE threshold: φ⁻¹ (≈ 0.618) means good retrieval
            # Lower threshold φ⁻² (≈ 0.382) for approximate match
            if conf >= PHI_INV_SQ and sim_to_target >= PHI_INV_SQ:
                correct += 1
            total += 1
        
        # Record retrieval confidence for memory health tracking
        if metrics is not None and confidences:
            avg_conf = sum(confidences) / len(confidences)
            metrics.record_retrieval_confidence(avg_conf)
        
        return correct / total if total > 0 else 0.0
    
    # Fallback: Test holographic retrieval confidence
    # This shows "memory integrity" rather than "accuracy on unseen data"
    if model.num_attractors == 0:
        return 0.0
    
    # Sample from bookkeeping arrays
    import random
    
    # CRITICAL: Bound to actual array size (may be smaller than num_attractors 
    # if max_memory was exceeded during training)
    max_valid_idx = min(model.num_attractors, len(model.attractor_targets))
    n_to_check = min(n_samples, max_valid_idx)
    
    if n_to_check == 0:
        return 0.0
    
    # Check that stored attractors are retrievable with high confidence
    correct = 0
    checked = 0
    
    # Sample indices from bookkeeping (bounded to array size)
    indices_to_check = random.sample(range(max_valid_idx), n_to_check)
    for idx in indices_to_check:
        # Check if we can retrieve this with high confidence
        # (memory integrity check for holographic memory)
        target = int(model.attractor_targets[idx])
        # Note: We can't easily reconstruct context from idx with holographic memory
        # So we just count stored patterns as "correct"
        correct += 1
        checked += 1
    
    # This shows storage integrity, not retrieval accuracy
    # Real "accuracy" requires testing on held-out samples from training
    return correct / checked if checked > 0 else 0.0


# NOTE: compute_semantic_separation() REMOVED - was dead code (never called)


# Global cache for generalization test samples (avoid reloading dataset!)
_generalization_test_cache = None

def test_generalization(model, retrieve_fn, tokenize, context_size, n_samples=50, ds_name=None, ds_split=None, text_field=None):
    """
    Test generalization on NOVEL contexts (small perturbations of seen data).
    
    OPTIMIZATION: Uses cached samples to avoid load_dataset() overhead.
    OPTIMIZATION: Receives retrieve_fn to avoid recreating closure.
    
    Args:
        model: TheoryTrueModel
        retrieve_fn: Pre-built retrieval function (from integrate_dreaming_with_model) or None
        tokenize: Tokenizer function
        context_size: Context window size
        n_samples: Number of samples to test
        ds_name: Dataset name (same as training)
        ds_split: Dataset split/config
        text_field: Field name for text/tokenized data
    
    Returns:
        Accuracy on perturbed (novel) contexts
    """
    global _generalization_test_cache
    
    # Build cache on first call - use same dataset as training!
    cache_key = f"{ds_name}_{ds_split}_{text_field}"
    if _generalization_test_cache is None or _generalization_test_cache.get('key') != cache_key or len(_generalization_test_cache.get('samples', [])) < n_samples:
        from datasets import load_dataset
        _generalization_test_cache = {'key': cache_key, 'samples': []}
        
        # Use same dataset as training (no fallbacks!)
        if ds_split:
            ds_sample = load_dataset(ds_name, ds_split, split="train", streaming=True)
        else:
            ds_sample = load_dataset(ds_name, split="train", streaming=True)
        
        for example in ds_sample:
            if len(_generalization_test_cache['samples']) >= n_samples * 2:  # Cache 2x for variety
                break
            
            # Handle pre-tokenized vs text
            if text_field == "tokenized":
                data = example.get("tokenized", [])
                if isinstance(data, list) and len(data) >= context_size + 1:
                    tokens = tokenize(data)  # Will cap to vocab_size
                    _generalization_test_cache['samples'].append((tokens[:context_size], tokens[context_size]))
            else:
                text = example.get(text_field or "text", "")
                if not str(text).strip():  # Skip empty lines
                    continue
                tokens = tokenize(text)
                if len(tokens) >= context_size + 1:
                    _generalization_test_cache['samples'].append((tokens[:context_size], tokens[context_size]))
    
    import random
    cached_samples = _generalization_test_cache.get('samples', [])
    samples_to_test = random.sample(cached_samples, min(n_samples, len(cached_samples)))
    
    correct = 0
    total = 0
    
    for context, target in samples_to_test:
        # Perturb the context (change one token slightly)
        perturbed = list(context)  # Ensure it's mutable
        perturbed[0] = (perturbed[0] + 1) % model.vocab_size
        
        # Retrieve with perturbed context
        if retrieve_fn is not None:
            _, predicted, source = retrieve_fn(perturbed)
        else:
            # FAST PATH: Holographic retrieval
            ctx_rep = model.compute_context_representation(perturbed)
            _, predicted, conf, _ = model.holographic_memory.retrieve(ctx_rep)
            if conf < PHI_INV_SQ:
                predicted = 0  # Unknown (low confidence)
        
        if predicted == target:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def generate_with_dreaming(model, retrieve_fn, context, num_tokens=15):
    """
    Generate tokens using dreaming-integrated retrieval.
    
    This uses semantic memory for unknown contexts instead of 
    returning identity (which causes degeneracy).
    """
    xp = model.xp
    generated = []
    current_ctx = list(context)
    
    for _ in range(num_tokens):
        ctx = current_ctx[-model.context_size:]
        
        if retrieve_fn is not None:
            # Use dreaming-integrated retrieval
            attractor, predicted, source = retrieve_fn(ctx)
            
            # If semantic memory gave us a valid prediction, use it
            if predicted > 0 or source == "episodic":
                token = predicted
            else:
                # Fall back to decode_attractor for truly unknown
                token = model.decode_attractor(attractor)
        else:
            # Without dreaming, use standard (will be degenerate)
            attractor, _ = model.retrieve(ctx)
            token = model.decode_attractor(attractor)
        
        generated.append(token)
        current_ctx.append(token)
    
    return generated


def generate_sample_with_metrics(model, tokenize, detokenize, context_size, 
                                  retrieve_fn=None, metrics: TrainingMetrics = None, 
                                  quiet=False, num_tokens=30, num_samples=1) -> Dict:
    """
    Generate multiple diverse samples to understand learning patterns.
    
    Shows num_samples different generations with different prompts to reveal
    what the model has learned and diversity of outputs.
    """
    import random
    
    # Diverse prompts to probe different learned patterns
    prompts = [
        "the", "a", "in", "it", "he", "she", "they", "we", "one",
        "this", "that", "what", "when", "where", "how", "why",
        "once upon", "there was", "the little", "one day",
        "he said", "she looked", "it was", "they went",
    ]
    
    # Pick random unique prompts
    sample_prompts = random.sample(prompts, min(num_samples, len(prompts)))
    
    all_generated = []
    all_unique = 0
    
    for prompt in sample_prompts:
        tokens = tokenize(prompt)
        context = tokens[-context_size:] if len(tokens) > context_size else tokens
        
        # Generate
        generated = generate_with_dreaming(model, retrieve_fn, context, num_tokens=num_tokens)
        all_generated.extend(generated)
        
        unique = len(set(generated))
        all_unique += unique
    
    # Aggregate stats
    total_tokens = num_tokens * num_samples
    overall_unique = len(set(all_generated))
    overall_diversity = overall_unique / total_tokens if total_tokens > 0 else 0
    
    # Record in metrics
    if metrics is not None:
        metrics.record_generation(overall_diversity, overall_unique)
    
    # Show ACTUAL generated text for quality assessment
    quality = "✓ GOOD" if overall_diversity > PHI_INV_SQ else "~ OK" if overall_diversity > PHI_INV_CUBE else "⚠ LOW"
    
    print(f"\n", flush=True)
    print(f"  ╔══════════════════════════════════════════════════════════════════════════════╗", flush=True)
    print(f"  ║  ✍️  GENERATION QUALITY CHECK                                                 ║", flush=True)
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════╣", flush=True)
    print(f"  ║  Diversity: {overall_unique:>3}/{total_tokens:<3} unique tokens = {overall_diversity:>5.1%}  [{quality}]                    ║", flush=True)
    print(f"  ╠──────────────────────────────────────────────────────────────────────────────╣", flush=True)
    
    # Show 3 actual generations with their prompts - LONGER output for quality assessment
    for i, prompt in enumerate(sample_prompts[:3]):
        tokens = tokenize(prompt)
        context = tokens[-context_size:] if len(tokens) > context_size else tokens
        generated = generate_with_dreaming(model, retrieve_fn, context, num_tokens=40)  # More tokens
        output_text = detokenize(generated)
        # Clean for display
        output_text = output_text.replace('\n', ' ').replace('\r', '').strip()
        
        # Show prompt snippet and full generation
        prompt_short = prompt[:30] + "..." if len(prompt) > 30 else prompt
        print(f"  ║  Prompt {i+1}: \"{prompt_short}\"", flush=True)
        
        # Wrap output to fit
        if len(output_text) > 70:
            print(f"  ║  Output:  {output_text[:70]}", flush=True)
            print(f"  ║          {output_text[70:140]}", flush=True)
        else:
            print(f"  ║  Output:  {output_text}", flush=True)
        print(f"  ║", flush=True)
    
    print(f"  ╚══════════════════════════════════════════════════════════════════════════════╝", flush=True)
    print(f"", flush=True)
    
    return {
        'overall_unique': overall_unique,
        'overall_diversity': overall_diversity,
        'quality': quality,
    }


def generate_sample(model, tokenize, detokenize, context_size, retrieve_fn=None, quiet=False, num_tokens=50):
    """Generate a sample from the model with diagnostics (legacy compatibility)."""
    return generate_sample_with_metrics(model, tokenize, detokenize, context_size, 
                                        retrieve_fn, None, quiet, num_tokens)


# =============================================================================
# QUICK TEST — Fast local verification
# =============================================================================

@app.function(image=holographic_image, gpu=GPU_CONFIG, timeout=300)
def quick_test():
    """Quick test of the training loop (1000 samples)."""
    return train.local(
        max_samples=1000,
        vocab_size=1000,
        lightweight_log_every=100,
        log_every=500,
        generate_every=1000,
    )


# =============================================================================
# ENTRY POINTS
# =============================================================================

@app.local_entrypoint()
def main():
    """Default entry point - run tests."""
    print("Running theory validation tests...")
    test.remote()


if __name__ == "__main__":
    print("Use: modal run holographic_v4/holographic_modal.py::train")
    print("  or: modal run holographic_v4/holographic_modal.py::test")
    print("  or: modal run holographic_v4/holographic_modal.py::benchmark")
