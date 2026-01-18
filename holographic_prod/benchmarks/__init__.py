"""
Holographic Memory NLP Benchmarking Suite
==========================================

Theory-True Benchmarks for Holographic Memory Architecture v5.11.0

This module provides comprehensive NLP benchmarking tools to evaluate
the holographic memory system against standard language modeling metrics
and compare with transformer baselines.

Architecture-Specific Benchmarks:
    1. Perplexity - Standard language modeling metric
    2. Generation Quality - Coherence, fluency, semantic fidelity
    3. Context Length Scaling - Verify "infinite context" claim
    4. Retrieval Accuracy - Episodic vs holographic vs combined
    5. Speed Benchmarks - Throughput, latency, memory usage

Standard NLP Benchmarks:
    6. GLUE - General Language Understanding Evaluation
    7. MMLU - Massive Multitask Language Understanding
    8. GSM-8K - Grade School Math Problems
    9. HumanEval - Code Generation
    10. MATH - Competition Math Problems

Theory-True Principles:
    - All benchmarks use real data, no synthetic/mock
    - No fallbacks that hide failures
    - Metrics derived from theory (φ-constants where applicable)
    - Explicit comparison to theoretical predictions
"""

__version__ = "1.1.0"

# Architecture-specific benchmarks
from .perplexity import (
    compute_perplexity,
    compute_cross_entropy,
    PerplexityBenchmark,
)
from .generation_quality import (
    measure_coherence,
    measure_fluency,
    measure_semantic_fidelity,
    GenerationQualityBenchmark,
)
from .context_length import (
    test_context_scaling,
    ContextLengthBenchmark,
)
from .retrieval_accuracy import (
    measure_retrieval_accuracy,
    RetrievalAccuracyBenchmark,
)
from .speed_benchmarks import (
    measure_throughput,
    measure_latency,
    SpeedBenchmark,
)

# Standard NLP benchmarks
from .standard_benchmarks import (
    # Benchmark classes
    GLUEBenchmark,
    MMLUBenchmark,
    GSM8KBenchmark,
    HumanEvalBenchmark,
    MATHBenchmark,
    # Unified runner
    StandardBenchmarkRunner,
    # Comparison utilities
    compare_to_transformers,
    print_comparison_table,
    TRANSFORMER_BASELINES,
    # Result types
    BenchmarkResult,
    BenchmarkType,
    # Task/subject lists
    GLUE_TASKS,
    MMLU_SUBJECTS,
    MATH_SUBJECTS,
)

# Few-shot benchmarks (like GPT evaluation)
from .few_shot_benchmarks import (
    FewShotEvaluator,
    FewShotGLUE,
    FewShotMMLU,
    FewShotGSM8K,
    FewShotBenchmarkRunner,
    FewShotResult,
)

__all__ = [
    # === Architecture-Specific ===
    # Perplexity
    "compute_perplexity",
    "compute_cross_entropy",
    "PerplexityBenchmark",
    # Generation Quality
    "measure_coherence",
    "measure_fluency",
    "measure_semantic_fidelity",
    "GenerationQualityBenchmark",
    # Context Length
    "test_context_scaling",
    "ContextLengthBenchmark",
    # Retrieval Accuracy
    "measure_retrieval_accuracy",
    "RetrievalAccuracyBenchmark",
    # Speed
    "measure_throughput",
    "measure_latency",
    "SpeedBenchmark",
    # === Standard NLP Benchmarks ===
    # Benchmark classes
    "GLUEBenchmark",
    "MMLUBenchmark",
    "GSM8KBenchmark",
    "HumanEvalBenchmark",
    "MATHBenchmark",
    # Unified runner
    "StandardBenchmarkRunner",
    # Comparison
    "compare_to_transformers",
    "print_comparison_table",
    "TRANSFORMER_BASELINES",
    # Types
    "BenchmarkResult",
    "BenchmarkType",
    # Task lists
    "GLUE_TASKS",
    "MMLU_SUBJECTS",
    "MATH_SUBJECTS",
    # === Few-Shot Benchmarks (GPT-style evaluation) ===
    "FewShotEvaluator",
    "FewShotGLUE",
    "FewShotMMLU",
    "FewShotGSM8K",
    "FewShotBenchmarkRunner",
    "FewShotResult",
]
