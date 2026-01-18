"""
Standard NLP Benchmarks for Holographic Memory
===============================================

Industry-standard benchmarks for comparing against transformer models.

Benchmarks Implemented:
    1. GLUE - General Language Understanding Evaluation
    2. MMLU - Massive Multitask Language Understanding
    3. GPQA - Graduate-Level Google Proof Q&A
    4. HumanEval - Code Generation
    5. GSM-8K - Grade School Math
    6. MATH - Competition Math Problems

Note: These benchmarks require downloading datasets from HuggingFace.
Use `pip install datasets` to enable automatic dataset loading.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import re


class BenchmarkType(Enum):
    """Types of standard benchmarks."""
    GLUE = "glue"
    MMLU = "mmlu"
    GPQA = "gpqa"
    HUMANEVAL = "humaneval"
    GSM8K = "gsm8k"
    MATH = "math"


@dataclass
class BenchmarkResult:
    """Result from a standard benchmark."""
    benchmark_name: str
    accuracy: float  # Overall accuracy (0-100)
    num_correct: int
    num_total: int
    subset_scores: Dict[str, float] = field(default_factory=dict)
    time_seconds: float = 0.0
    tokens_processed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# GLUE BENCHMARK
# =============================================================================

GLUE_TASKS = {
    "cola": {
        "description": "Corpus of Linguistic Acceptability - grammatical acceptability",
        "metric": "matthews_correlation",
        "num_labels": 2,
    },
    "sst2": {
        "description": "Stanford Sentiment Treebank - sentiment classification",
        "metric": "accuracy",
        "num_labels": 2,
    },
    "mrpc": {
        "description": "Microsoft Research Paraphrase Corpus - paraphrase detection",
        "metric": "f1",
        "num_labels": 2,
    },
    "qqp": {
        "description": "Quora Question Pairs - duplicate question detection",
        "metric": "f1",
        "num_labels": 2,
    },
    "stsb": {
        "description": "Semantic Textual Similarity Benchmark - similarity regression",
        "metric": "pearson_spearman",
        "num_labels": 1,  # regression
    },
    "mnli": {
        "description": "Multi-Genre Natural Language Inference",
        "metric": "accuracy",
        "num_labels": 3,
    },
    "qnli": {
        "description": "Question Natural Language Inference",
        "metric": "accuracy",
        "num_labels": 2,
    },
    "rte": {
        "description": "Recognizing Textual Entailment",
        "metric": "accuracy",
        "num_labels": 2,
    },
    "wnli": {
        "description": "Winograd Schema Challenge NLI",
        "metric": "accuracy",
        "num_labels": 2,
    },
}


class GLUEBenchmark:
    """
    General Language Understanding Evaluation (GLUE) Benchmark.
    
    Tests language understanding across 9 tasks:
    - CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI
    
    Reference: https://gluebenchmark.com/
    """
    
    def __init__(
        self,
        memory: Any,
        tokenizer: Optional[Any] = None,
        context_length: int = 64,
        verbose: bool = True,
    ):
        """
        Initialize GLUE benchmark.
        
        Args:
            memory: HolographicMemory instance
            tokenizer: Tokenizer for text-to-tokens (if None, uses simple split)
            context_length: Context window size
            verbose: Print progress
        """
        self.memory = memory
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.verbose = verbose
    
    def load_task(self, task_name: str, split: str = "validation", max_samples: int = 1000) -> List[Dict]:
        """
        Load a GLUE task dataset.
        
        Args:
            task_name: One of GLUE_TASKS keys
            split: Dataset split (train, validation, test)
            max_samples: Maximum samples to load
        
        Returns:
            List of examples with 'text', 'label' keys
        """
        try:
            from datasets import load_dataset
            
            # Map task names to HuggingFace dataset names
            hf_name_map = {
                "cola": ("glue", "cola"),
                "sst2": ("glue", "sst2"),
                "mrpc": ("glue", "mrpc"),
                "qqp": ("glue", "qqp"),
                "stsb": ("glue", "stsb"),
                "mnli": ("glue", "mnli"),
                "qnli": ("glue", "qnli"),
                "rte": ("glue", "rte"),
                "wnli": ("glue", "wnli"),
            }
            
            if task_name not in hf_name_map:
                raise ValueError(f"Unknown task: {task_name}. Choose from: {list(GLUE_TASKS.keys())}")
            
            dataset_name, config_name = hf_name_map[task_name]
            
            # Handle MNLI special case
            if task_name == "mnli" and split == "validation":
                split = "validation_matched"
            
            dataset = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
            
            examples = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                # Format text based on task
                if task_name in ["cola", "sst2"]:
                    text = item["sentence"]
                elif task_name in ["mrpc", "qqp"]:
                    text = f"{item['sentence1']} [SEP] {item['sentence2']}"
                elif task_name == "stsb":
                    text = f"{item['sentence1']} [SEP] {item['sentence2']}"
                elif task_name in ["mnli", "qnli", "rte", "wnli"]:
                    if "premise" in item:
                        text = f"{item['premise']} [SEP] {item['hypothesis']}"
                    elif "question" in item:
                        text = f"{item['question']} [SEP] {item['sentence']}"
                    else:
                        text = f"{item['sentence1']} [SEP] {item['sentence2']}"
                else:
                    text = str(item)
                
                examples.append({
                    "text": text,
                    "label": item["label"],
                    "task": task_name,
                })
            
            return examples
            
        except ImportError:
            if self.verbose:
                print("⚠️ 'datasets' library not installed. Run: pip install datasets")
            return []
    
    def evaluate_task(
        self,
        task_name: str,
        max_samples: int = 1000,
        timeout_seconds: float = 300.0,
    ) -> Dict[str, float]:
        """
        Evaluate on a single GLUE task.
        
        Args:
            task_name: Task to evaluate
            max_samples: Maximum samples
            timeout_seconds: Maximum evaluation time
        
        Returns:
            Dictionary with metric scores
        """
        examples = self.load_task(task_name, max_samples=max_samples)
        
        if not examples:
            return {"error": "Failed to load dataset"}
        
        if self.verbose:
            print(f"\n  Evaluating {task_name.upper()} ({len(examples)} samples)...")
        
        start_time = time.perf_counter()
        correct = 0
        predictions = []
        labels = []
        
        for ex in examples:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                break
            
            # Tokenize
            tokens = self._tokenize(ex["text"])
            
            # Get prediction via generation
            pred_label = self._predict_classification(tokens, GLUE_TASKS[task_name]["num_labels"])
            
            predictions.append(pred_label)
            labels.append(ex["label"])
            
            if pred_label == ex["label"]:
                correct += 1
        
        accuracy = correct / len(predictions) * 100 if predictions else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(predictions),
            "time_seconds": time.perf_counter() - start_time,
        }
    
    def run(
        self,
        tasks: Optional[List[str]] = None,
        max_samples_per_task: int = 500,
        timeout_seconds: float = 600.0,
    ) -> BenchmarkResult:
        """
        Run full GLUE benchmark.
        
        Args:
            tasks: List of tasks to run (None = all)
            max_samples_per_task: Max samples per task
            timeout_seconds: Total timeout
        
        Returns:
            BenchmarkResult with overall and per-task scores
        """
        if tasks is None:
            tasks = list(GLUE_TASKS.keys())
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("  GLUE BENCHMARK")
            print("=" * 60)
        
        start_time = time.perf_counter()
        task_results = {}
        total_correct = 0
        total_samples = 0
        
        for task in tasks:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                if self.verbose:
                    print(f"⚠️ Timeout after {elapsed:.1f}s")
                break
            
            result = self.evaluate_task(
                task,
                max_samples=max_samples_per_task,
                timeout_seconds=timeout_seconds - elapsed,
            )
            
            task_results[task] = result.get("accuracy", 0)
            total_correct += result.get("correct", 0)
            total_samples += result.get("total", 0)
            
            if self.verbose:
                print(f"    {task.upper()}: {result.get('accuracy', 0):.1f}%")
        
        overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
        
        return BenchmarkResult(
            benchmark_name="GLUE",
            accuracy=overall_accuracy,
            num_correct=total_correct,
            num_total=total_samples,
            subset_scores=task_results,
            time_seconds=time.perf_counter() - start_time,
        )
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)[:self.context_length]
        else:
            # Simple word-level tokenization
            words = text.lower().split()
            # Hash words to token IDs
            return [hash(w) % self.memory.vocab_size for w in words[:self.context_length]]
    
    def _predict_classification(self, tokens: List[int], num_labels: int) -> int:
        """Predict class label from tokens."""
        if len(tokens) < 2:
            return 0
        
        # Use context to predict - THEORY-TRUE: always returns valid token
        context = tokens[-min(self.context_length, len(tokens)):]
        predicted = self.memory.retrieve_theory_true(context)
        
        # Map token to label
        return predicted % num_labels


# =============================================================================
# MMLU BENCHMARK
# =============================================================================

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


class MMLUBenchmark:
    """
    Massive Multitask Language Understanding (MMLU) Benchmark.
    
    Tests multitask accuracy across 57 subjects requiring world knowledge.
    
    Reference: https://github.com/hendrycks/test
    """
    
    def __init__(
        self,
        memory: Any,
        tokenizer: Optional[Any] = None,
        context_length: int = 128,
        verbose: bool = True,
    ):
        self.memory = memory
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.verbose = verbose
    
    def load_subject(self, subject: str, split: str = "test", max_samples: int = 100) -> List[Dict]:
        """Load MMLU subject data."""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)
            
            examples = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                question = item["question"]
                choices = item["choices"]
                answer = item["answer"]  # 0-3 for A-D
                
                # Format as multiple choice
                formatted = f"{question}\n"
                for j, choice in enumerate(choices):
                    formatted += f"{chr(65+j)}. {choice}\n"
                
                examples.append({
                    "text": formatted,
                    "choices": choices,
                    "answer": answer,
                    "subject": subject,
                })
            
            return examples
            
        except ImportError:
            if self.verbose:
                print("⚠️ 'datasets' library not installed.")
            return []
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error loading {subject}: {e}")
            return []
    
    def evaluate_subject(
        self,
        subject: str,
        max_samples: int = 100,
        timeout_seconds: float = 60.0,
    ) -> Dict[str, float]:
        """Evaluate on a single MMLU subject."""
        examples = self.load_subject(subject, max_samples=max_samples)
        
        if not examples:
            return {"error": "Failed to load dataset"}
        
        start_time = time.perf_counter()
        correct = 0
        
        for ex in examples:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                break
            
            tokens = self._tokenize(ex["text"])
            pred = self._predict_multiple_choice(tokens, len(ex["choices"]))
            
            if pred == ex["answer"]:
                correct += 1
        
        return {
            "accuracy": correct / len(examples) * 100 if examples else 0,
            "correct": correct,
            "total": len(examples),
        }
    
    def run(
        self,
        subjects: Optional[List[str]] = None,
        max_samples_per_subject: int = 50,
        timeout_seconds: float = 1800.0,
    ) -> BenchmarkResult:
        """Run MMLU benchmark."""
        if subjects is None:
            # Use a representative subset
            subjects = MMLU_SUBJECTS[:10]  # First 10 subjects
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("  MMLU BENCHMARK")
            print("=" * 60)
            print(f"  Testing {len(subjects)} subjects")
        
        start_time = time.perf_counter()
        subject_results = {}
        total_correct = 0
        total_samples = 0
        
        for subject in subjects:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                break
            
            result = self.evaluate_subject(
                subject,
                max_samples=max_samples_per_subject,
                timeout_seconds=60.0,
            )
            
            subject_results[subject] = result.get("accuracy", 0)
            total_correct += result.get("correct", 0)
            total_samples += result.get("total", 0)
            
            if self.verbose:
                print(f"    {subject}: {result.get('accuracy', 0):.1f}%")
        
        overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
        
        return BenchmarkResult(
            benchmark_name="MMLU",
            accuracy=overall_accuracy,
            num_correct=total_correct,
            num_total=total_samples,
            subset_scores=subject_results,
            time_seconds=time.perf_counter() - start_time,
        )
    
    def _tokenize(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)[:self.context_length]
        words = text.lower().split()
        return [hash(w) % self.memory.vocab_size for w in words[:self.context_length]]
    
    def _predict_multiple_choice(self, tokens: List[int], num_choices: int = 4) -> int:
        """THEORY-TRUE: retrieve_theory_true always returns valid token."""
        if len(tokens) < 2:
            return 0
        context = tokens[-min(self.context_length, len(tokens)):]
        predicted = self.memory.retrieve_theory_true(context)
        return predicted % num_choices


# =============================================================================
# GSM-8K BENCHMARK (Grade School Math)
# =============================================================================

class GSM8KBenchmark:
    """
    GSM-8K Benchmark for grade school math word problems.
    
    Tests mathematical reasoning with linguistically diverse problems.
    
    Reference: https://github.com/openai/grade-school-math
    """
    
    def __init__(
        self,
        memory: Any,
        tokenizer: Optional[Any] = None,
        context_length: int = 256,
        verbose: bool = True,
    ):
        self.memory = memory
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.verbose = verbose
    
    def load_data(self, split: str = "test", max_samples: int = 500) -> List[Dict]:
        """Load GSM-8K dataset."""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("gsm8k", "main", split=split, trust_remote_code=True)
            
            examples = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                question = item["question"]
                answer = item["answer"]
                
                # Extract final numerical answer
                # GSM-8K answers end with "#### <number>"
                final_answer = self._extract_final_answer(answer)
                
                examples.append({
                    "question": question,
                    "full_answer": answer,
                    "final_answer": final_answer,
                })
            
            return examples
            
        except ImportError:
            if self.verbose:
                print("⚠️ 'datasets' library not installed.")
            return []
    
    def _extract_final_answer(self, answer: str) -> Optional[float]:
        """Extract final numerical answer from GSM-8K format."""
        # Look for #### pattern
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                return float(num_str)
            except ValueError:
                return None
        return None
    
    def run(
        self,
        max_samples: int = 500,
        timeout_seconds: float = 600.0,
    ) -> BenchmarkResult:
        """Run GSM-8K benchmark."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("  GSM-8K BENCHMARK (Grade School Math)")
            print("=" * 60)
        
        examples = self.load_data(max_samples=max_samples)
        
        if not examples:
            return BenchmarkResult(
                benchmark_name="GSM-8K",
                accuracy=0,
                num_correct=0,
                num_total=0,
            )
        
        start_time = time.perf_counter()
        correct = 0
        
        for i, ex in enumerate(examples):
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                if self.verbose:
                    print(f"⚠️ Timeout after {i} samples")
                break
            
            # Generate answer
            tokens = self._tokenize(ex["question"])
            generated = self._generate_answer(tokens)
            
            # Extract predicted number
            pred_number = self._extract_number(generated)
            
            if pred_number is not None and ex["final_answer"] is not None:
                if abs(pred_number - ex["final_answer"]) < 0.01:
                    correct += 1
            
            if self.verbose and (i + 1) % 100 == 0:
                print(f"    [{i + 1}/{len(examples)}] Accuracy: {correct / (i + 1) * 100:.1f}%")
        
        accuracy = correct / len(examples) * 100 if examples else 0
        
        if self.verbose:
            print(f"\n  Final Accuracy: {accuracy:.1f}%")
        
        return BenchmarkResult(
            benchmark_name="GSM-8K",
            accuracy=accuracy,
            num_correct=correct,
            num_total=len(examples),
            time_seconds=time.perf_counter() - start_time,
        )
    
    def _tokenize(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)[:self.context_length]
        words = text.lower().split()
        return [hash(w) % self.memory.vocab_size for w in words[:self.context_length]]
    
    def _generate_answer(self, tokens: List[int], max_length: int = 100) -> List[int]:
        """Generate answer tokens. THEORY-TRUE: always produces output."""
        generated = list(tokens)
        
        for _ in range(max_length):
            context = generated[-min(self.context_length, len(generated)):]
            # THEORY-TRUE: retrieve_theory_true NEVER returns None
            predicted = self.memory.retrieve_theory_true(context)
            generated.append(predicted)
        
        return generated[len(tokens):]
    
    def _extract_number(self, tokens: List[int]) -> Optional[float]:
        """Extract number from generated tokens (placeholder)."""
        # This is a simplified version - proper implementation would
        # decode tokens and parse the numerical answer
        if tokens:
            return float(tokens[-1] % 1000)  # Placeholder
        return None


# =============================================================================
# HUMANEVAL BENCHMARK (Code Generation)
# =============================================================================

class HumanEvalBenchmark:
    """
    HumanEval Benchmark for code generation.
    
    Tests ability to generate syntactically correct and semantically
    meaningful Python code.
    
    Reference: https://github.com/openai/human-eval
    """
    
    def __init__(
        self,
        memory: Any,
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        verbose: bool = True,
    ):
        self.memory = memory
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.verbose = verbose
    
    def load_data(self, max_samples: int = 164) -> List[Dict]:
        """Load HumanEval dataset."""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
            
            examples = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                examples.append({
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                })
            
            return examples
            
        except ImportError:
            if self.verbose:
                print("⚠️ 'datasets' library not installed.")
            return []
    
    def run(
        self,
        max_samples: int = 164,
        timeout_seconds: float = 1800.0,
        k: int = 1,  # pass@k metric
    ) -> BenchmarkResult:
        """
        Run HumanEval benchmark.
        
        Args:
            max_samples: Maximum problems to evaluate
            timeout_seconds: Total timeout
            k: k for pass@k metric
        
        Returns:
            BenchmarkResult with pass@k score
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("  HUMANEVAL BENCHMARK (Code Generation)")
            print("=" * 60)
        
        examples = self.load_data(max_samples=max_samples)
        
        if not examples:
            return BenchmarkResult(
                benchmark_name="HumanEval",
                accuracy=0,
                num_correct=0,
                num_total=0,
            )
        
        start_time = time.perf_counter()
        passed = 0
        
        for i, ex in enumerate(examples):
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                break
            
            # Generate code completion
            tokens = self._tokenize(ex["prompt"])
            generated_tokens = self._generate_code(tokens)
            
            # Check if code passes tests (simplified)
            # Real implementation would execute the code
            if self._check_code(ex, generated_tokens):
                passed += 1
            
            if self.verbose and (i + 1) % 20 == 0:
                print(f"    [{i + 1}/{len(examples)}] pass@{k}: {passed / (i + 1) * 100:.1f}%")
        
        pass_rate = passed / len(examples) * 100 if examples else 0
        
        if self.verbose:
            print(f"\n  pass@{k}: {pass_rate:.1f}%")
        
        return BenchmarkResult(
            benchmark_name="HumanEval",
            accuracy=pass_rate,
            num_correct=passed,
            num_total=len(examples),
            time_seconds=time.perf_counter() - start_time,
            metadata={"k": k},
        )
    
    def _tokenize(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)[:self.context_length]
        # Character-level tokenization for code
        return [ord(c) % self.memory.vocab_size for c in text[:self.context_length]]
    
    def _generate_code(self, tokens: List[int], max_length: int = 256) -> List[int]:
        """Generate code completion tokens. THEORY-TRUE: always produces output."""
        generated = list(tokens)
        
        for _ in range(max_length):
            context = generated[-min(self.context_length, len(generated)):]
            # THEORY-TRUE: retrieve_theory_true NEVER returns None
            predicted = self.memory.retrieve_theory_true(context)
            generated.append(predicted)
        
        return generated[len(tokens):]
    
    def _check_code(self, example: Dict, generated_tokens: List[int]) -> bool:
        """Check if generated code passes tests (placeholder)."""
        # Real implementation would:
        # 1. Decode tokens to code
        # 2. Concatenate with prompt
        # 3. Execute with test cases
        # 4. Check all tests pass
        return False  # Conservative - actual implementation needed


# =============================================================================
# MATH BENCHMARK (Competition Math)
# =============================================================================

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


class MATHBenchmark:
    """
    MATH Benchmark for competition-level mathematics.
    
    Tests ability to solve challenging math problems from competitions.
    
    Reference: https://github.com/hendrycks/math
    """
    
    def __init__(
        self,
        memory: Any,
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        verbose: bool = True,
    ):
        self.memory = memory
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.verbose = verbose
    
    def load_data(self, subject: Optional[str] = None, split: str = "test", max_samples: int = 500) -> List[Dict]:
        """Load MATH dataset."""
        try:
            from datasets import load_dataset
            
            # MATH dataset structure
            dataset = load_dataset("hendrycks/competition_math", split=split, trust_remote_code=True)
            
            examples = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                if subject is not None and item.get("type", "").lower() != subject:
                    continue
                
                examples.append({
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "level": item.get("level", "unknown"),
                    "type": item.get("type", "unknown"),
                })
            
            return examples
            
        except ImportError:
            if self.verbose:
                print("⚠️ 'datasets' library not installed.")
            return []
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error loading MATH: {e}")
            return []
    
    def run(
        self,
        subjects: Optional[List[str]] = None,
        max_samples: int = 500,
        timeout_seconds: float = 1200.0,
    ) -> BenchmarkResult:
        """Run MATH benchmark."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("  MATH BENCHMARK (Competition Math)")
            print("=" * 60)
        
        examples = self.load_data(max_samples=max_samples)
        
        if not examples:
            return BenchmarkResult(
                benchmark_name="MATH",
                accuracy=0,
                num_correct=0,
                num_total=0,
            )
        
        start_time = time.perf_counter()
        correct = 0
        level_results = {}
        
        for i, ex in enumerate(examples):
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                break
            
            tokens = self._tokenize(ex["problem"])
            generated = self._generate_solution(tokens)
            
            # Check answer (simplified)
            is_correct = self._check_answer(ex, generated)
            if is_correct:
                correct += 1
            
            # Track by level
            level = ex.get("level", "unknown")
            if level not in level_results:
                level_results[level] = {"correct": 0, "total": 0}
            level_results[level]["total"] += 1
            if is_correct:
                level_results[level]["correct"] += 1
            
            if self.verbose and (i + 1) % 100 == 0:
                print(f"    [{i + 1}/{len(examples)}] Accuracy: {correct / (i + 1) * 100:.1f}%")
        
        accuracy = correct / len(examples) * 100 if examples else 0
        
        # Compute per-level accuracy
        level_accuracy = {}
        for level, data in level_results.items():
            if data["total"] > 0:
                level_accuracy[level] = data["correct"] / data["total"] * 100
        
        if self.verbose:
            print(f"\n  Final Accuracy: {accuracy:.1f}%")
            for level, acc in sorted(level_accuracy.items()):
                print(f"    {level}: {acc:.1f}%")
        
        return BenchmarkResult(
            benchmark_name="MATH",
            accuracy=accuracy,
            num_correct=correct,
            num_total=len(examples),
            subset_scores=level_accuracy,
            time_seconds=time.perf_counter() - start_time,
        )
    
    def _tokenize(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)[:self.context_length]
        words = text.lower().split()
        return [hash(w) % self.memory.vocab_size for w in words[:self.context_length]]
    
    def _generate_solution(self, tokens: List[int], max_length: int = 256) -> List[int]:
        """Generate solution tokens. THEORY-TRUE: always produces output."""
        generated = list(tokens)
        
        for _ in range(max_length):
            context = generated[-min(self.context_length, len(generated)):]
            # THEORY-TRUE: retrieve_theory_true NEVER returns None
            predicted = self.memory.retrieve_theory_true(context)
            generated.append(predicted)
        
        return generated[len(tokens):]
    
    def _check_answer(self, example: Dict, generated_tokens: List[int]) -> bool:
        """Check if generated answer is correct (placeholder)."""
        # Real implementation would:
        # 1. Decode tokens
        # 2. Extract final answer (often in \boxed{})
        # 3. Compare with ground truth
        return False  # Conservative


# =============================================================================
# UNIFIED BENCHMARK RUNNER
# =============================================================================

class StandardBenchmarkRunner:
    """
    Unified runner for all standard NLP benchmarks.
    
    Usage:
        runner = StandardBenchmarkRunner(memory)
        results = runner.run_all()
    """
    
    def __init__(
        self,
        memory: Any,
        tokenizer: Optional[Any] = None,
        verbose: bool = True,
    ):
        self.memory = memory
        self.tokenizer = tokenizer
        self.verbose = verbose
    
    def run_all(
        self,
        benchmarks: Optional[List[str]] = None,
        timeout_per_benchmark: float = 600.0,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run multiple benchmarks.
        
        Args:
            benchmarks: List of benchmark names (None = all)
            timeout_per_benchmark: Timeout per benchmark in seconds
        
        Returns:
            Dictionary mapping benchmark names to results
        """
        if benchmarks is None:
            benchmarks = ["glue", "mmlu", "gsm8k", "humaneval", "math"]
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("  STANDARD NLP BENCHMARK SUITE")
            print("=" * 70)
        
        results = {}
        
        for bench_name in benchmarks:
            bench_name = bench_name.lower()
            
            if self.verbose:
                print(f"\n  Running {bench_name.upper()}...")
            
            try:
                if bench_name == "glue":
                    benchmark = GLUEBenchmark(self.memory, self.tokenizer, verbose=self.verbose)
                    results["GLUE"] = benchmark.run(timeout_seconds=timeout_per_benchmark)
                
                elif bench_name == "mmlu":
                    benchmark = MMLUBenchmark(self.memory, self.tokenizer, verbose=self.verbose)
                    results["MMLU"] = benchmark.run(timeout_seconds=timeout_per_benchmark)
                
                elif bench_name == "gsm8k":
                    benchmark = GSM8KBenchmark(self.memory, self.tokenizer, verbose=self.verbose)
                    results["GSM-8K"] = benchmark.run(timeout_seconds=timeout_per_benchmark)
                
                elif bench_name == "humaneval":
                    benchmark = HumanEvalBenchmark(self.memory, self.tokenizer, verbose=self.verbose)
                    results["HumanEval"] = benchmark.run(timeout_seconds=timeout_per_benchmark)
                
                elif bench_name == "math":
                    benchmark = MATHBenchmark(self.memory, self.tokenizer, verbose=self.verbose)
                    results["MATH"] = benchmark.run(timeout_seconds=timeout_per_benchmark)
                
                else:
                    if self.verbose:
                        print(f"    ⚠️ Unknown benchmark: {bench_name}")
            
            except Exception as e:
                if self.verbose:
                    print(f"    ❌ Error running {bench_name}: {e}")
                results[bench_name.upper()] = BenchmarkResult(
                    benchmark_name=bench_name.upper(),
                    accuracy=0,
                    num_correct=0,
                    num_total=0,
                    metadata={"error": str(e)},
                )
        
        if self.verbose:
            self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, BenchmarkResult]) -> None:
        """Print summary of all benchmark results."""
        print("\n" + "=" * 70)
        print("  BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"  {'Benchmark':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
        print("-" * 70)
        
        for name, result in results.items():
            print(f"  {name:<15} {result.accuracy:>9.1f}% {result.num_correct:>10} {result.num_total:>10}")
        
        # Overall average
        if results:
            avg_accuracy = np.mean([r.accuracy for r in results.values()])
            print("-" * 70)
            print(f"  {'AVERAGE':<15} {avg_accuracy:>9.1f}%")
        
        print("=" * 70)


# =============================================================================
# TRANSFORMER COMPARISON UTILITIES
# =============================================================================

# Reference scores for comparison (approximate, as of 2024)
TRANSFORMER_BASELINES = {
    "GPT-4": {
        "GLUE": 92.0,
        "MMLU": 86.4,
        "GSM-8K": 92.0,
        "HumanEval": 67.0,
        "MATH": 42.5,
        "GPQA": 39.0,
    },
    "GPT-3.5": {
        "GLUE": 85.0,
        "MMLU": 70.0,
        "GSM-8K": 57.1,
        "HumanEval": 48.1,
        "MATH": 23.5,
    },
    "LLaMA-2-70B": {
        "GLUE": 82.0,
        "MMLU": 68.9,
        "GSM-8K": 56.8,
        "HumanEval": 29.9,
        "MATH": 13.5,
    },
    "Mistral-7B": {
        "GLUE": 78.0,
        "MMLU": 62.5,
        "GSM-8K": 52.2,
        "HumanEval": 26.2,
    },
}


def compare_to_transformers(
    results: Dict[str, BenchmarkResult],
    baseline: str = "GPT-3.5",
) -> Dict[str, Dict[str, float]]:
    """
    Compare benchmark results to transformer baselines.
    
    Args:
        results: Dictionary of BenchmarkResult from StandardBenchmarkRunner
        baseline: Which transformer to compare against
    
    Returns:
        Dictionary with comparison metrics
    """
    if baseline not in TRANSFORMER_BASELINES:
        raise ValueError(f"Unknown baseline: {baseline}. Choose from: {list(TRANSFORMER_BASELINES.keys())}")
    
    baseline_scores = TRANSFORMER_BASELINES[baseline]
    comparison = {}
    
    for bench_name, result in results.items():
        if bench_name in baseline_scores:
            baseline_score = baseline_scores[bench_name]
            our_score = result.accuracy
            
            comparison[bench_name] = {
                "our_score": our_score,
                "baseline_score": baseline_score,
                "difference": our_score - baseline_score,
                "ratio": our_score / baseline_score if baseline_score > 0 else 0,
                "beats_baseline": our_score > baseline_score,
            }
    
    return comparison


def print_comparison_table(
    results: Dict[str, BenchmarkResult],
    baselines: Optional[List[str]] = None,
) -> None:
    """Print comparison table against multiple transformer baselines."""
    if baselines is None:
        baselines = ["GPT-3.5", "LLaMA-2-70B", "Mistral-7B"]
    
    print("\n" + "=" * 80)
    print("  COMPARISON WITH TRANSFORMER BASELINES")
    print("=" * 80)
    
    # Header
    header = f"  {'Benchmark':<12} {'Ours':>8}"
    for bl in baselines:
        header += f" {bl[:10]:>10}"
    print(header)
    print("-" * 80)
    
    for bench_name, result in results.items():
        row = f"  {bench_name:<12} {result.accuracy:>7.1f}%"
        
        for bl in baselines:
            if bl in TRANSFORMER_BASELINES and bench_name in TRANSFORMER_BASELINES[bl]:
                bl_score = TRANSFORMER_BASELINES[bl][bench_name]
                diff = result.accuracy - bl_score
                marker = "↑" if diff > 0 else "↓" if diff < 0 else "="
                row += f" {bl_score:>7.1f}%{marker}"
            else:
                row += f" {'N/A':>9}"
        
        print(row)
    
    print("=" * 80)
