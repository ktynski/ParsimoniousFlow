"""
Generation Quality Benchmarking for Holographic Memory
======================================================

Theory-true metrics for evaluating text generation quality.

Key Metrics:
    1. Coherence - Semantic consistency across generated text
    2. Fluency - Grammatical and syntactic quality
    3. Semantic Fidelity - Preservation of meaning during generation
    4. Witness Stability - Theory-specific: witness energy during generation

Theory Connection:
    - Witness stability during generation indicates semantic preservation
    - Enstrophy (bivector energy) indicates structural diversity
    - Stability ratio (witness/total) should approach φ⁻² ≈ 0.382
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter


# Import constants
try:
    from holographic_prod.core.constants import PHI_INV_SQ
except ImportError:
    PHI_INV_SQ = 0.3819660112501051


@dataclass
class GenerationQualityResult:
    """Result of generation quality evaluation."""
    coherence_score: float  # 0-1, higher is better
    fluency_score: float  # 0-1, higher is better
    semantic_fidelity: float  # 0-1, higher is better
    witness_stability: float  # Should approach PHI_INV_SQ
    diversity_score: float  # 0-1, higher means more diverse
    repetition_rate: float  # 0-1, lower is better
    avg_generation_confidence: float
    num_tokens_generated: int
    generation_time_ms: float


def measure_coherence(
    generated_tokens: List[int],
    memory: Any,
    window_size: int = 8,
) -> float:
    """
    Measure coherence of generated text using holographic similarity.
    
    Coherence is measured by how well consecutive windows of text
    maintain semantic relationship (measured by frobenius_cosine
    between their context embeddings).
    
    Args:
        generated_tokens: List of generated token IDs
        memory: HolographicMemory instance
        window_size: Size of sliding window
    
    Returns:
        Coherence score (0-1, higher is better)
    """
    if len(generated_tokens) < window_size * 2:
        return 0.0
    
    try:
        from holographic_prod.core.algebra import frobenius_cosine
    except ImportError:
        # Fallback implementation
        def frobenius_cosine(a, b, xp=np):
            a_flat = a.flatten()
            b_flat = b.flatten()
            norm_a = np.linalg.norm(a_flat)
            norm_b = np.linalg.norm(b_flat)
            if norm_a < 1e-10 or norm_b < 1e-10:
                return 0.0
            return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))
    
    similarities = []
    
    for i in range(len(generated_tokens) - window_size * 2 + 1):
        window1 = generated_tokens[i:i + window_size]
        window2 = generated_tokens[i + window_size:i + window_size * 2]
        
        # Get embeddings for each window
        emb1 = memory.tower._embed_sequence(window1)
        emb2 = memory.tower._embed_sequence(window2)
        
        # Compute similarity
        sim = frobenius_cosine(emb1, emb2)
        similarities.append(sim)
    
    if not similarities:
        return 0.0
    
    # Coherence = average similarity (normalized to 0-1)
    avg_sim = np.mean(similarities)
    # Similarity can be negative, normalize to 0-1
    coherence = (avg_sim + 1) / 2
    
    return float(coherence)


def measure_fluency(
    generated_tokens: List[int],
    token_to_word: Optional[Dict[int, str]] = None,
    ngram_order: int = 3,
) -> float:
    """
    Measure fluency using n-gram statistics.
    
    Fluency is approximated by:
    1. N-gram diversity (not too repetitive)
    2. Absence of immediate repetition
    3. Reasonable token distribution
    
    Args:
        generated_tokens: List of generated token IDs
        token_to_word: Optional mapping from token IDs to words
        ngram_order: Order of n-grams to analyze
    
    Returns:
        Fluency score (0-1, higher is better)
    """
    if len(generated_tokens) < ngram_order:
        return 0.0
    
    # Extract n-grams
    ngrams = []
    for i in range(len(generated_tokens) - ngram_order + 1):
        ngram = tuple(generated_tokens[i:i + ngram_order])
        ngrams.append(ngram)
    
    if not ngrams:
        return 0.0
    
    # N-gram diversity: unique ngrams / total ngrams
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    diversity = unique_ngrams / total_ngrams
    
    # Immediate repetition penalty
    immediate_reps = sum(1 for i in range(len(generated_tokens) - 1) 
                        if generated_tokens[i] == generated_tokens[i + 1])
    rep_penalty = 1 - (immediate_reps / (len(generated_tokens) - 1)) if len(generated_tokens) > 1 else 1.0
    
    # Token distribution evenness (entropy-based)
    token_counts = Counter(generated_tokens)
    probs = np.array(list(token_counts.values())) / len(generated_tokens)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(token_counts)) if len(token_counts) > 1 else 1
    evenness = entropy / max_entropy if max_entropy > 0 else 0
    
    # Combined fluency score
    fluency = 0.4 * diversity + 0.3 * rep_penalty + 0.3 * evenness
    
    return float(np.clip(fluency, 0, 1))


def measure_semantic_fidelity(
    prompt_tokens: List[int],
    generated_tokens: List[int],
    memory: Any,
) -> float:
    """
    Measure semantic fidelity: how well generated text preserves prompt meaning.
    
    Uses holographic similarity between prompt embedding and generated
    text embedding.
    
    Args:
        prompt_tokens: Original prompt tokens
        generated_tokens: Generated continuation tokens
        memory: HolographicMemory instance
    
    Returns:
        Semantic fidelity score (0-1, higher means better preservation)
    """
    if not prompt_tokens or not generated_tokens:
        return 0.0
    
    try:
        from holographic_prod.core.algebra import frobenius_cosine
    except ImportError:
        def frobenius_cosine(a, b, xp=np):
            a_flat = a.flatten()
            b_flat = b.flatten()
            norm_a = np.linalg.norm(a_flat)
            norm_b = np.linalg.norm(b_flat)
            if norm_a < 1e-10 or norm_b < 1e-10:
                return 0.0
            return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))
    
    # Embed prompt and generated text
    prompt_emb = memory.tower._embed_sequence(prompt_tokens)
    gen_emb = memory.tower._embed_sequence(generated_tokens)
    
    # Compute similarity
    sim = frobenius_cosine(prompt_emb, gen_emb)
    
    # Normalize to 0-1
    fidelity = (sim + 1) / 2
    
    return float(fidelity)


def measure_witness_stability(
    generated_tokens: List[int],
    memory: Any,
    window_size: int = 8,
) -> Tuple[float, List[float]]:
    """
    Measure witness stability during generation.
    
    Theory-true metric: the witness (scalar + pseudoscalar) should remain
    stable during generation, with stability ratio approaching φ⁻² ≈ 0.382.
    
    Args:
        generated_tokens: Generated tokens
        memory: HolographicMemory instance
        window_size: Window size for stability computation
    
    Returns:
        (average_stability, stability_history)
    """
    if len(generated_tokens) < window_size:
        return 0.0, []
    
    try:
        from holographic_prod.core.algebra import witness, total_energy
    except ImportError:
        # Simplified fallback
        def witness(m, xp=np):
            return float(m[0, 0]**2 + m[3, 3]**2)
        def total_energy(m, xp=np):
            return float(np.sum(m**2))
    
    stabilities = []
    
    for i in range(len(generated_tokens) - window_size + 1):
        window = generated_tokens[i:i + window_size]
        emb = memory.tower._embed_sequence(window)
        
        w = witness(emb)
        total = total_energy(emb)
        
        if total > 1e-10:
            stability = w / total
        else:
            stability = 0.0
        
        stabilities.append(stability)
    
    avg_stability = np.mean(stabilities) if stabilities else 0.0
    
    return float(avg_stability), stabilities


class GenerationQualityBenchmark:
    """
    Comprehensive generation quality benchmarking.
    
    Usage:
        benchmark = GenerationQualityBenchmark(memory)
        result = benchmark.evaluate(prompts, num_tokens=100)
    """
    
    def __init__(
        self,
        memory: Any,
        context_length: int = 8,
        verbose: bool = True,
    ):
        """
        Initialize generation quality benchmark.
        
        Args:
            memory: HolographicMemory instance
            context_length: Context length for generation
            verbose: Print progress
        """
        self.memory = memory
        self.context_length = context_length
        self.verbose = verbose
    
    def evaluate(
        self,
        prompts: List[List[int]],
        num_tokens: int = 100,
        timeout_seconds: float = 300.0,
    ) -> GenerationQualityResult:
        """
        Evaluate generation quality on prompts.
        
        Args:
            prompts: List of prompt token sequences
            num_tokens: Number of tokens to generate per prompt
            timeout_seconds: Maximum evaluation time
        
        Returns:
            GenerationQualityResult with all metrics
        """
        start_time = time.perf_counter()
        
        all_coherence = []
        all_fluency = []
        all_fidelity = []
        all_stability = []
        all_diversity = []
        all_repetition = []
        all_confidences = []
        total_generated = 0
        
        for i, prompt in enumerate(prompts):
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                if self.verbose:
                    print(f"⚠️ Timeout after {elapsed:.1f}s")
                break
            
            # Generate tokens
            generated, confidences = self._generate(prompt, num_tokens)
            total_generated += len(generated)
            all_confidences.extend(confidences)
            
            if len(generated) > self.context_length:
                # Compute metrics
                coherence = measure_coherence(generated, self.memory, self.context_length)
                fluency = measure_fluency(generated)
                fidelity = measure_semantic_fidelity(prompt, generated, self.memory)
                stability, _ = measure_witness_stability(generated, self.memory, self.context_length)
                
                # Diversity and repetition
                unique_tokens = len(set(generated))
                diversity = unique_tokens / len(generated) if generated else 0
                reps = sum(1 for j in range(len(generated) - 1) 
                          if generated[j] == generated[j + 1])
                repetition = reps / (len(generated) - 1) if len(generated) > 1 else 0
                
                all_coherence.append(coherence)
                all_fluency.append(fluency)
                all_fidelity.append(fidelity)
                all_stability.append(stability)
                all_diversity.append(diversity)
                all_repetition.append(repetition)
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(prompts)}] Generated {total_generated} tokens")
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        
        result = GenerationQualityResult(
            coherence_score=np.mean(all_coherence) if all_coherence else 0.0,
            fluency_score=np.mean(all_fluency) if all_fluency else 0.0,
            semantic_fidelity=np.mean(all_fidelity) if all_fidelity else 0.0,
            witness_stability=np.mean(all_stability) if all_stability else 0.0,
            diversity_score=np.mean(all_diversity) if all_diversity else 0.0,
            repetition_rate=np.mean(all_repetition) if all_repetition else 0.0,
            avg_generation_confidence=np.mean(all_confidences) if all_confidences else 0.0,
            num_tokens_generated=total_generated,
            generation_time_ms=elapsed_ms,
        )
        
        if self.verbose:
            self._print_result(result)
        
        return result
    
    def _generate(
        self,
        prompt: List[int],
        num_tokens: int,
    ) -> Tuple[List[int], List[float]]:
        """Generate tokens from prompt."""
        generated = list(prompt)
        confidences = []
        
        for _ in range(num_tokens):
            if len(generated) < self.context_length:
                context = generated
            else:
                context = generated[-self.context_length:]
            
            # THEORY-TRUE (v5.15.0): retrieve_theory_true NEVER returns None
            # Grace guarantees convergence to SOME attractor
            predicted = self.memory.retrieve_theory_true(context)
            
            # Theory-true always outputs - get coherence for tracking
            # (confidence from theory-true is coherence, not probability)
            
            generated.append(predicted)
            # Track coherence from internal state (theory-true stores it)
            confidence = getattr(self.memory, '_last_coherence', 0.5)
            confidences.append(confidence)
        
        return generated[len(prompt):], confidences
    
    def _print_result(self, result: GenerationQualityResult) -> None:
        """Print formatted result."""
        print("\n" + "=" * 60)
        print("  GENERATION QUALITY BENCHMARK RESULTS")
        print("=" * 60)
        print(f"  Coherence:         {result.coherence_score:.4f}")
        print(f"  Fluency:           {result.fluency_score:.4f}")
        print(f"  Semantic Fidelity: {result.semantic_fidelity:.4f}")
        print(f"  Witness Stability: {result.witness_stability:.4f} (target: {PHI_INV_SQ:.4f})")
        print(f"  Diversity:         {result.diversity_score:.4f}")
        print(f"  Repetition Rate:   {result.repetition_rate:.4f}")
        print("-" * 60)
        print(f"  Avg Confidence:    {result.avg_generation_confidence:.4f}")
        print(f"  Tokens Generated:  {result.num_tokens_generated:,}")
        print(f"  Time:              {result.generation_time_ms:.1f} ms")
        print("=" * 60)
