# Holographic Memory NLP Benchmarking Suite

**Version**: 1.1.0  
**Theory-True**: All benchmarks follow FIRM architecture principles

## Overview

This benchmarking suite provides comprehensive NLP evaluation tools for the holographic memory architecture, including both **architecture-specific metrics** and **standard industry benchmarks** for transformer comparison.

All benchmarks are designed to be theory-true:

- **Real data only** - No synthetic/mock data
- **No silent failures** - All errors are reported
- **φ-derived thresholds** - Where applicable
- **Explicit comparison** to theoretical predictions

---

## Standard NLP Benchmarks

These are the industry-standard benchmarks used to compare against transformers:

### GLUE (General Language Understanding Evaluation)

9 tasks testing language understanding: CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI.

```python
from holographic_prod.benchmarks import GLUEBenchmark

benchmark = GLUEBenchmark(memory, tokenizer=my_tokenizer)
result = benchmark.run(tasks=["sst2", "mrpc", "mnli"])
print(f"GLUE Score: {result.accuracy:.1f}%")
```

### MMLU (Massive Multitask Language Understanding)

57 subjects testing world knowledge: algebra, anatomy, law, medicine, etc.

```python
from holographic_prod.benchmarks import MMLUBenchmark

benchmark = MMLUBenchmark(memory)
result = benchmark.run(subjects=["abstract_algebra", "anatomy", "machine_learning"])
print(f"MMLU Score: {result.accuracy:.1f}%")
```

### GSM-8K (Grade School Math)

Linguistically diverse grade school math word problems.

```python
from holographic_prod.benchmarks import GSM8KBenchmark

benchmark = GSM8KBenchmark(memory)
result = benchmark.run(max_samples=500)
print(f"GSM-8K Accuracy: {result.accuracy:.1f}%")
```

### HumanEval (Code Generation)

Python code generation and execution.

```python
from holographic_prod.benchmarks import HumanEvalBenchmark

benchmark = HumanEvalBenchmark(memory)
result = benchmark.run()
print(f"pass@1: {result.accuracy:.1f}%")
```

### MATH (Competition Math)

Challenging math competition problems across 7 subjects.

```python
from holographic_prod.benchmarks import MATHBenchmark

benchmark = MATHBenchmark(memory)
result = benchmark.run()
print(f"MATH Accuracy: {result.accuracy:.1f}%")
```

### Running All Standard Benchmarks

```python
from holographic_prod.benchmarks import StandardBenchmarkRunner

runner = StandardBenchmarkRunner(memory, tokenizer=my_tokenizer)
results = runner.run_all(benchmarks=["glue", "mmlu", "gsm8k", "humaneval", "math"])

# Compare to transformers
from holographic_prod.benchmarks import print_comparison_table
print_comparison_table(results, baselines=["GPT-3.5", "LLaMA-2-70B"])
```

### Transformer Baselines (Reference)

| Benchmark | GPT-4 | GPT-3.5 | LLaMA-2-70B | Mistral-7B |
|-----------|-------|---------|-------------|------------|
| GLUE | 92.0% | 85.0% | 82.0% | 78.0% |
| MMLU | 86.4% | 70.0% | 68.9% | 62.5% |
| GSM-8K | 92.0% | 57.1% | 56.8% | 52.2% |
| HumanEval | 67.0% | 48.1% | 29.9% | 26.2% |
| MATH | 42.5% | 23.5% | 13.5% | — |

---

## Architecture-Specific Benchmarks

### 1. Perplexity (`perplexity.py`)

Standard language modeling metric measuring prediction quality.

```python
from holographic_prod.benchmarks import PerplexityBenchmark

benchmark = PerplexityBenchmark(memory, context_length=8)
result = benchmark.evaluate(test_sequences)
print(f"Perplexity: {result.perplexity:.2f}")
```

**Metrics**:
- Perplexity (lower is better)
- Cross-entropy (nats)
- Accuracy (%)
- Retrieval source breakdown (episodic vs holographic)

### 2. Generation Quality (`generation_quality.py`)

Evaluates quality of generated text.

```python
from holographic_prod.benchmarks import GenerationQualityBenchmark

benchmark = GenerationQualityBenchmark(memory)
result = benchmark.evaluate(prompts, num_tokens=100)
print(f"Coherence: {result.coherence_score:.4f}")
```

**Metrics**:
- Coherence - Semantic consistency across generated text
- Fluency - N-gram diversity and repetition
- Semantic Fidelity - Preservation of prompt meaning
- Witness Stability - Theory-specific: stability ratio (target: φ⁻² ≈ 0.382)
- Diversity - Unique tokens / total tokens
- Repetition Rate - Immediate repetitions

### 3. Context Length Scaling (`context_length.py`)

Tests the "infinite context" claim by verifying SO(4) properties at extreme lengths.

```python
from holographic_prod.benchmarks import ContextLengthBenchmark

benchmark = ContextLengthBenchmark(memory)
results = benchmark.run(context_lengths=[256, 1024, 4096, 16384, 65536])
```

**Metrics**:
- Frobenius norm (should be 2.0 for SO(4))
- Determinant (should be 1.0)
- Orthogonality error (should be ~0)
- Embedding time scaling (should be O(n))

**Theory Predictions**:
- Time complexity: O(n) linear
- Error scaling: O(√n)
- SO(4) validity: Should hold at any context length

### 4. Retrieval Accuracy (`retrieval_accuracy.py`)

Measures accuracy of different retrieval pathways.

```python
from holographic_prod.benchmarks import RetrievalAccuracyBenchmark

benchmark = RetrievalAccuracyBenchmark(memory)
result = benchmark.evaluate(test_sequences)
# Or detailed pathway analysis:
pathway_results = benchmark.evaluate_by_pathway(test_sequences)
```

**Metrics**:
- Overall accuracy
- Episodic accuracy (exact match pathway)
- Holographic accuracy (generalization pathway)
- Hit rates by pathway
- Conflict rate (episodic ≠ holographic)
- Synergy analysis (combined vs individual)

### 5. Speed Benchmarks (`speed_benchmarks.py`)

Measures throughput, latency, and memory efficiency.

```python
from holographic_prod.benchmarks import SpeedBenchmark

benchmark = SpeedBenchmark(memory)
results = benchmark.run()
print(f"Learn throughput: {results['learn_throughput'].tokens_per_second:,.0f} tok/s")
```

**Metrics**:
- Throughput (tokens/second for learn and retrieve)
- Latency (p50, p90, p99 in microseconds)
- GPU memory usage

## Running All Benchmarks

```python
from holographic_prod.benchmarks import (
    PerplexityBenchmark,
    GenerationQualityBenchmark,
    ContextLengthBenchmark,
    RetrievalAccuracyBenchmark,
    SpeedBenchmark,
)
from holographic_prod import HolographicMemory

# Create memory
memory = HolographicMemory(vocab_size=50000, max_levels=5)

# Train on data (assuming you have training sequences)
for seq in train_sequences:
    for i in range(8, len(seq)):
        memory.learn(seq[i-8:i], seq[i])

# Run all benchmarks
ppl_result = PerplexityBenchmark(memory).evaluate(test_sequences)
gen_result = GenerationQualityBenchmark(memory).evaluate(prompts)
ctx_result = ContextLengthBenchmark(memory).run()
acc_result = RetrievalAccuracyBenchmark(memory).evaluate(test_sequences)
speed_result = SpeedBenchmark(memory).run()
```

## Transformer Comparison

```python
from holographic_prod.benchmarks.perplexity import benchmark_perplexity_vs_transformer

# Compare to a known transformer baseline
comparison = benchmark_perplexity_vs_transformer(
    memory,
    test_sequences,
    transformer_ppl=25.0,  # Your baseline
)
print(f"Improvement: {comparison['improvement_percent']:.1f}%")
```

## Theory-True Principles

1. **No fallbacks** - All metrics come from real computations
2. **No mocking** - Real data, real operations
3. **φ-derived constants** - Stability targets use φ⁻² ≈ 0.382
4. **Explicit failures** - Timeouts and errors are reported, not hidden
5. **SO(4) verification** - Context scaling explicitly checks group properties

## Output Interpretation

### Perplexity
- **< 10**: Excellent (near memorization)
- **10-50**: Good (strong generalization)
- **50-100**: Moderate
- **> 100**: Poor (mostly guessing)

### Witness Stability
- **≈ 0.382** (φ⁻²): Optimal (theory-predicted attractor)
- **> 0.5**: Over-stable (may indicate collapse)
- **< 0.2**: Under-stable (may indicate noise)

### SO(4) Validity
- **Frobenius norm**: 2.0 ± 0.05
- **Determinant**: 1.0 ± 0.05
- **Orthogonality error**: < 0.1

## File Structure

```
benchmarks/
├── __init__.py              # Package exports
├── README.md                # This file
├── perplexity.py            # Perplexity benchmarks
├── generation_quality.py    # Generation quality metrics
├── context_length.py        # Context scaling tests
├── retrieval_accuracy.py    # Retrieval pathway analysis
├── speed_benchmarks.py      # Throughput and latency
└── standard_benchmarks.py   # GLUE, MMLU, GSM-8K, HumanEval, MATH
```

## Requirements

For standard NLP benchmarks, install the datasets library:

```bash
pip install datasets
```

Optional for code execution (HumanEval):

```bash
pip install human-eval
```

## Notes on Standard Benchmarks

1. **Data Loading**: Standard benchmarks load data from HuggingFace Hub automatically.
   First run may take a few minutes to download datasets.

2. **Tokenization**: You can pass a custom tokenizer (e.g., from `transformers` or `tiktoken`).
   If not provided, a simple word-level hash tokenization is used.

3. **Code Execution**: HumanEval requires executing generated code. The current
   implementation is conservative (always returns False for safety). For real
   evaluation, integrate with the `human-eval` package.

4. **GPQA**: Not yet implemented (requires specialized dataset access).

5. **Timeouts**: All benchmarks have configurable timeouts to prevent runaway execution.
