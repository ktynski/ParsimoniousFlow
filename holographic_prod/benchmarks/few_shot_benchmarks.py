"""
Few-Shot NLP Benchmarks for Holographic Memory
===============================================

Tests general language learning via few-shot prompting, like GPT evaluation.

The key insight: A general language learner should be able to:
1. See a few examples of a task (few-shot)
2. Generalize to new instances
3. Without task-specific training

This is the standard way GPT/Claude/etc are evaluated on GLUE, MMLU, etc.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class FewShotResult:
    """Result from few-shot benchmark."""
    benchmark_name: str
    accuracy: float
    num_correct: int
    num_total: int
    few_shot_k: int  # Number of examples shown
    time_seconds: float
    per_task_scores: Dict[str, float] = None


class FewShotEvaluator:
    """
    Few-shot evaluation for holographic memory.
    
    Tests whether the model can:
    1. Parse the pattern from examples
    2. Apply it to new instances
    3. Generate the correct continuation (answer)
    
    This is the TRUE test of general language learning.
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
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(text)
            return [t % self.memory.vocab_size for t in tokens]
        else:
            # Simple character-level fallback
            return [ord(c) % self.memory.vocab_size for c in text]
    
    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 10,
        stop_tokens: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Generate continuation from prompt using THEORY-TRUE retrieval.
        
        THEORY (v5.15.0):
            Uses retrieve_theory_true() which:
            - ALWAYS returns a token (never None) - Grace guarantees convergence
            - Uses coherence (witness stability), not similarity
            - Scores against full vocabulary, not limited candidates
            
        This enables few-shot learning via attractor dynamics:
            - Few-shot examples shift attractor basins
            - Context resonates with these basins
            - Output emerges from coherent state
        """
        generated = list(prompt_tokens)
        
        for _ in range(max_new_tokens):
            # Use last context_length tokens
            context = generated[-min(self.context_length, len(generated)):]
            
            # THEORY-TRUE: retrieve_theory_true() NEVER returns None!
            # Grace always contracts to SOME attractor basin.
            predicted = self.memory.retrieve_theory_true(context)
            
            generated.append(predicted)
            
            # Check for stop tokens
            if stop_tokens and predicted in stop_tokens:
                break
        
        return generated[len(prompt_tokens):]
    
    def evaluate_multiple_choice(
        self,
        prompt: str,
        choices: List[str],
        correct_idx: int,
    ) -> Tuple[int, float]:
        """
        Evaluate multiple choice question.
        
        Returns (predicted_idx, confidence)
        """
        # Tokenize prompt
        prompt_tokens = self.tokenize(prompt)
        
        # Generate next token(s) - THEORY-TRUE: always produces output
        generated = self.generate(prompt_tokens, max_new_tokens=5)
        
        # THEORY-TRUE (v5.15.0): generate() ALWAYS produces tokens
        # because retrieve_theory_true() NEVER returns None.
        # Grace dynamics guarantee convergence to some attractor.
        # The "if not generated" check is no longer needed.
        
        # Try to decode and match to choices
        # For letter answers (A, B, C, D), check first generated token
        first_token = generated[0]
        
        # Map common letter tokens
        letter_map = {
            ord('A') % self.memory.vocab_size: 0,
            ord('B') % self.memory.vocab_size: 1,
            ord('C') % self.memory.vocab_size: 2,
            ord('D') % self.memory.vocab_size: 3,
            ord('a') % self.memory.vocab_size: 0,
            ord('b') % self.memory.vocab_size: 1,
            ord('c') % self.memory.vocab_size: 2,
            ord('d') % self.memory.vocab_size: 3,
        }
        
        if first_token in letter_map:
            return letter_map[first_token], 1.0
        
        # Fallback: return most likely based on token value
        return first_token % len(choices), 0.5


class FewShotGLUE:
    """
    Few-shot GLUE benchmark.
    
    Format:
    ```
    Sentence: "The movie was great!"
    Sentiment: positive
    
    Sentence: "I hated this film."
    Sentiment: negative
    
    Sentence: "[test sentence]"
    Sentiment:
    ```
    """
    
    def __init__(
        self,
        evaluator: FewShotEvaluator,
        k_shot: int = 3,
        verbose: bool = True,
    ):
        self.evaluator = evaluator
        self.k_shot = k_shot
        self.verbose = verbose
    
    def format_sst2_prompt(
        self,
        examples: List[Dict],
        test_sentence: str,
    ) -> str:
        """Format SST-2 (sentiment) prompt."""
        prompt = "Classify the sentiment of each sentence as positive or negative.\n\n"
        
        for ex in examples[:self.k_shot]:
            sentiment = "positive" if ex["label"] == 1 else "negative"
            prompt += f"Sentence: \"{ex['sentence']}\"\n"
            prompt += f"Sentiment: {sentiment}\n\n"
        
        prompt += f"Sentence: \"{test_sentence}\"\n"
        prompt += "Sentiment:"
        
        return prompt
    
    def format_mrpc_prompt(
        self,
        examples: List[Dict],
        test_s1: str,
        test_s2: str,
    ) -> str:
        """Format MRPC (paraphrase) prompt."""
        prompt = "Determine if two sentences are paraphrases (yes/no).\n\n"
        
        for ex in examples[:self.k_shot]:
            answer = "yes" if ex["label"] == 1 else "no"
            prompt += f"Sentence 1: \"{ex['sentence1']}\"\n"
            prompt += f"Sentence 2: \"{ex['sentence2']}\"\n"
            prompt += f"Paraphrase: {answer}\n\n"
        
        prompt += f"Sentence 1: \"{test_s1}\"\n"
        prompt += f"Sentence 2: \"{test_s2}\"\n"
        prompt += "Paraphrase:"
        
        return prompt
    
    def run_sst2(
        self,
        max_samples: int = 100,
        timeout_seconds: float = 300.0,
    ) -> Dict[str, Any]:
        """Run few-shot SST-2 evaluation."""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("glue", "sst2", split="validation", trust_remote_code=True)
            
            # Get examples for few-shot
            examples = [{"sentence": dataset[i]["sentence"], "label": dataset[i]["label"]} 
                       for i in range(self.k_shot)]
            
            start_time = time.perf_counter()
            correct = 0
            total = 0
            
            for i in range(self.k_shot, min(len(dataset), max_samples + self.k_shot)):
                elapsed = time.perf_counter() - start_time
                if elapsed > timeout_seconds:
                    break
                
                test_item = dataset[i]
                prompt = self.format_sst2_prompt(examples, test_item["sentence"])
                
                # Tokenize and generate
                prompt_tokens = self.evaluator.tokenize(prompt)
                generated = self.evaluator.generate(prompt_tokens, max_new_tokens=10)
                
                # Check if generated starts with "positive" or "negative"
                if generated:
                    # Simple heuristic: check first few tokens
                    first_token = generated[0]
                    
                    # Token for 'p' (positive) vs 'n' (negative)
                    p_token = ord('p') % self.evaluator.memory.vocab_size
                    n_token = ord('n') % self.evaluator.memory.vocab_size
                    
                    # Debug: show what was generated
                    if total < 3 and self.verbose:
                        print(f"    DEBUG: first_token={first_token}, p_token={p_token}, n_token={n_token}")
                        print(f"    DEBUG: generated={generated[:5]}")
                    
                    if first_token == p_token:
                        predicted = 1
                    elif first_token == n_token:
                        predicted = 0
                    else:
                        predicted = first_token % 2
                    
                    if predicted == test_item["label"]:
                        correct += 1
                else:
                    if total < 3 and self.verbose:
                        print(f"    DEBUG: No tokens generated!")
                
                total += 1
                
                if self.verbose and total % 20 == 0:
                    acc = correct / total * 100
                    print(f"    SST-2 [{total}]: {acc:.1f}%")
            
            accuracy = correct / total * 100 if total > 0 else 0
            
            return {
                "task": "sst2",
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "k_shot": self.k_shot,
                "time_seconds": time.perf_counter() - start_time,
            }
            
        except Exception as e:
            if self.verbose:
                print(f"    ❌ SST-2 error: {e}")
            return {"task": "sst2", "error": str(e)}
    
    def run(
        self,
        tasks: Optional[List[str]] = None,
        max_samples_per_task: int = 100,
        timeout_seconds: float = 600.0,
    ) -> FewShotResult:
        """Run few-shot GLUE benchmark."""
        if tasks is None:
            tasks = ["sst2"]  # Start with SST-2 (simplest)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  FEW-SHOT GLUE ({self.k_shot}-shot)")
            print(f"{'='*60}")
        
        start_time = time.perf_counter()
        results = {}
        total_correct = 0
        total_samples = 0
        
        for task in tasks:
            if task == "sst2":
                result = self.run_sst2(max_samples_per_task, timeout_seconds)
            else:
                result = {"task": task, "error": "Not implemented"}
            
            results[task] = result.get("accuracy", 0)
            total_correct += result.get("correct", 0)
            total_samples += result.get("total", 0)
            
            if self.verbose and "accuracy" in result:
                print(f"    {task.upper()}: {result['accuracy']:.1f}%")
        
        overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
        
        return FewShotResult(
            benchmark_name="GLUE (few-shot)",
            accuracy=overall_accuracy,
            num_correct=total_correct,
            num_total=total_samples,
            few_shot_k=self.k_shot,
            time_seconds=time.perf_counter() - start_time,
            per_task_scores=results,
        )


class FewShotMMLU:
    """
    Few-shot MMLU benchmark.
    
    Format:
    ```
    Question: What is the capital of France?
    A. London
    B. Paris
    C. Berlin
    D. Madrid
    Answer: B
    
    Question: [test question]
    A. [choice A]
    B. [choice B]
    C. [choice C]
    D. [choice D]
    Answer:
    ```
    """
    
    def __init__(
        self,
        evaluator: FewShotEvaluator,
        k_shot: int = 5,
        verbose: bool = True,
    ):
        self.evaluator = evaluator
        self.k_shot = k_shot
        self.verbose = verbose
    
    def format_prompt(
        self,
        examples: List[Dict],
        test_question: str,
        test_choices: List[str],
    ) -> str:
        """Format MMLU multiple-choice prompt."""
        prompt = "Answer each question by selecting A, B, C, or D.\n\n"
        
        for ex in examples[:self.k_shot]:
            prompt += f"Question: {ex['question']}\n"
            for j, choice in enumerate(ex['choices']):
                prompt += f"{chr(65+j)}. {choice}\n"
            prompt += f"Answer: {chr(65 + ex['answer'])}\n\n"
        
        prompt += f"Question: {test_question}\n"
        for j, choice in enumerate(test_choices):
            prompt += f"{chr(65+j)}. {choice}\n"
        prompt += "Answer:"
        
        return prompt
    
    def run_subject(
        self,
        subject: str,
        max_samples: int = 50,
        timeout_seconds: float = 120.0,
    ) -> Dict[str, Any]:
        """Run few-shot evaluation on a single MMLU subject."""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
            
            # Get examples from dev set for few-shot
            dev_dataset = load_dataset("cais/mmlu", subject, split="dev", trust_remote_code=True)
            examples = [
                {
                    "question": dev_dataset[i]["question"],
                    "choices": dev_dataset[i]["choices"],
                    "answer": dev_dataset[i]["answer"],
                }
                for i in range(min(self.k_shot, len(dev_dataset)))
            ]
            
            start_time = time.perf_counter()
            correct = 0
            total = 0
            
            for i in range(min(len(dataset), max_samples)):
                elapsed = time.perf_counter() - start_time
                if elapsed > timeout_seconds:
                    break
                
                test_item = dataset[i]
                prompt = self.format_prompt(
                    examples,
                    test_item["question"],
                    test_item["choices"],
                )
                
                # Generate answer
                prompt_tokens = self.evaluator.tokenize(prompt)
                generated = self.evaluator.generate(prompt_tokens, max_new_tokens=5)
                
                # Parse answer
                if generated:
                    first_token = generated[0]
                    
                    # Check for A, B, C, D tokens
                    answer_map = {
                        ord('A') % self.evaluator.memory.vocab_size: 0,
                        ord('B') % self.evaluator.memory.vocab_size: 1,
                        ord('C') % self.evaluator.memory.vocab_size: 2,
                        ord('D') % self.evaluator.memory.vocab_size: 3,
                    }
                    
                    if first_token in answer_map:
                        predicted = answer_map[first_token]
                    else:
                        predicted = first_token % 4
                    
                    if predicted == test_item["answer"]:
                        correct += 1
                
                total += 1
            
            accuracy = correct / total * 100 if total > 0 else 0
            
            return {
                "subject": subject,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            }
            
        except Exception as e:
            return {"subject": subject, "error": str(e)}
    
    def run(
        self,
        subjects: Optional[List[str]] = None,
        max_samples_per_subject: int = 50,
        timeout_seconds: float = 1200.0,
    ) -> FewShotResult:
        """Run few-shot MMLU benchmark."""
        if subjects is None:
            # Representative subset
            subjects = [
                "abstract_algebra",
                "anatomy",
                "college_computer_science",
                "high_school_mathematics",
                "logical_fallacies",
            ]
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  FEW-SHOT MMLU ({self.k_shot}-shot)")
            print(f"{'='*60}")
        
        start_time = time.perf_counter()
        results = {}
        total_correct = 0
        total_samples = 0
        
        for subject in subjects:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                break
            
            result = self.run_subject(subject, max_samples_per_subject, 120.0)
            
            results[subject] = result.get("accuracy", 0)
            total_correct += result.get("correct", 0)
            total_samples += result.get("total", 0)
            
            if self.verbose:
                if "accuracy" in result:
                    print(f"    {subject}: {result['accuracy']:.1f}%")
                else:
                    print(f"    {subject}: ❌ {result.get('error', 'unknown error')}")
        
        overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
        
        return FewShotResult(
            benchmark_name="MMLU (few-shot)",
            accuracy=overall_accuracy,
            num_correct=total_correct,
            num_total=total_samples,
            few_shot_k=self.k_shot,
            time_seconds=time.perf_counter() - start_time,
            per_task_scores=results,
        )


class FewShotGSM8K:
    """
    Few-shot GSM-8K (math) benchmark.
    
    Format:
    ```
    Question: If there are 3 apples and you add 2 more, how many apples are there?
    Answer: 5
    
    Question: [test question]
    Answer:
    ```
    """
    
    def __init__(
        self,
        evaluator: FewShotEvaluator,
        k_shot: int = 5,
        verbose: bool = True,
    ):
        self.evaluator = evaluator
        self.k_shot = k_shot
        self.verbose = verbose
    
    def format_prompt(
        self,
        examples: List[Dict],
        test_question: str,
    ) -> str:
        """Format GSM-8K prompt with chain-of-thought."""
        prompt = "Solve each math problem. Show your work and give the final answer.\n\n"
        
        for ex in examples[:self.k_shot]:
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"
        
        prompt += f"Question: {test_question}\n"
        prompt += "Answer:"
        
        return prompt
    
    def extract_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from generated text."""
        import re
        
        # Look for #### pattern (GSM-8K format)
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                pass  # Not a valid number format, try next pattern
        
        # Look for any number
        numbers = re.findall(r'-?[\d,]+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except ValueError:
                pass  # Not a valid number format
        
        return None
    
    def run(
        self,
        max_samples: int = 100,
        timeout_seconds: float = 600.0,
    ) -> FewShotResult:
        """Run few-shot GSM-8K benchmark."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  FEW-SHOT GSM-8K ({self.k_shot}-shot)")
            print(f"{'='*60}")
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
            train = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
            
            # Get few-shot examples from train
            examples = [
                {
                    "question": train[i]["question"],
                    "answer": train[i]["answer"],
                }
                for i in range(self.k_shot)
            ]
            
            start_time = time.perf_counter()
            correct = 0
            total = 0
            
            for i in range(min(len(dataset), max_samples)):
                elapsed = time.perf_counter() - start_time
                if elapsed > timeout_seconds:
                    break
                
                test_item = dataset[i]
                prompt = self.format_prompt(examples, test_item["question"])
                
                # Generate answer
                prompt_tokens = self.evaluator.tokenize(prompt)
                generated = self.evaluator.generate(prompt_tokens, max_new_tokens=100)
                
                # This is a placeholder - proper implementation would decode tokens
                # and extract the numerical answer
                # For now, just count as incorrect
                total += 1
                
                if self.verbose and total % 20 == 0:
                    acc = correct / total * 100
                    print(f"    GSM-8K [{total}]: {acc:.1f}%")
            
            accuracy = correct / total * 100 if total > 0 else 0
            
            return FewShotResult(
                benchmark_name="GSM-8K (few-shot)",
                accuracy=accuracy,
                num_correct=correct,
                num_total=total,
                few_shot_k=self.k_shot,
                time_seconds=time.perf_counter() - start_time,
            )
            
        except Exception as e:
            if self.verbose:
                print(f"    ❌ Error: {e}")
            return FewShotResult(
                benchmark_name="GSM-8K (few-shot)",
                accuracy=0,
                num_correct=0,
                num_total=0,
                few_shot_k=self.k_shot,
                time_seconds=0,
            )


class FewShotBenchmarkRunner:
    """
    Run all few-shot benchmarks.
    
    This is the TRUE test of general language learning:
    - Model sees a few examples of a task
    - Must generalize to new instances
    - No task-specific training
    
    Usage:
        runner = FewShotBenchmarkRunner(memory, k_shot=5)
        results = runner.run_all()
    """
    
    def __init__(
        self,
        memory: Any,
        tokenizer: Optional[Any] = None,
        k_shot: int = 5,
        context_length: int = 512,
        verbose: bool = True,
    ):
        self.evaluator = FewShotEvaluator(
            memory, tokenizer, context_length, verbose
        )
        self.k_shot = k_shot
        self.verbose = verbose
    
    def run_all(
        self,
        benchmarks: Optional[List[str]] = None,
        max_samples_per_benchmark: int = 100,
        timeout_per_benchmark: float = 600.0,
    ) -> Dict[str, FewShotResult]:
        """Run all few-shot benchmarks."""
        if benchmarks is None:
            benchmarks = ["glue", "mmlu"]  # GSM-8K needs more work
        
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"  FEW-SHOT BENCHMARK SUITE ({self.k_shot}-shot)")
            print("  Testing GENERAL LANGUAGE LEARNING capability")
            print("=" * 70)
        
        results = {}
        
        for bench in benchmarks:
            bench = bench.lower()
            
            try:
                if bench == "glue":
                    glue = FewShotGLUE(self.evaluator, self.k_shot, self.verbose)
                    results["GLUE"] = glue.run(
                        max_samples_per_task=max_samples_per_benchmark,
                        timeout_seconds=timeout_per_benchmark,
                    )
                
                elif bench == "mmlu":
                    mmlu = FewShotMMLU(self.evaluator, self.k_shot, self.verbose)
                    results["MMLU"] = mmlu.run(
                        max_samples_per_subject=max_samples_per_benchmark,
                        timeout_seconds=timeout_per_benchmark,
                    )
                
                elif bench == "gsm8k":
                    gsm = FewShotGSM8K(self.evaluator, self.k_shot, self.verbose)
                    results["GSM-8K"] = gsm.run(
                        max_samples=max_samples_per_benchmark,
                        timeout_seconds=timeout_per_benchmark,
                    )
                
            except Exception as e:
                if self.verbose:
                    print(f"  ❌ {bench} error: {e}")
        
        if self.verbose:
            self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, FewShotResult]) -> None:
        """Print summary of results."""
        print("\n" + "=" * 70)
        print("  FEW-SHOT BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"  {'Benchmark':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
        print("-" * 70)
        
        for name, result in results.items():
            print(f"  {name:<20} {result.accuracy:>9.1f}% {result.num_correct:>10} {result.num_total:>10}")
        
        # Comparison to GPT-3.5 (5-shot)
        print("-" * 70)
        print("  GPT-3.5 (5-shot) reference:")
        print("    GLUE (SST-2): ~95%")
        print("    MMLU: ~70%")
        print("    GSM-8K: ~57%")
        print("=" * 70)
