"""
End-to-End Test: Grounded Embeddings Integration

Validates the complete flow:
1. Create corpus with semantic structure
2. Compute co-occurrence and grounded embeddings
3. Initialize HolographicMemory with grounded embeddings
4. Train and compare accuracy vs random embeddings
"""
import numpy as np
import time
from typing import List, Tuple

from holographic_prod.memory.holographic_memory_unified import HolographicMemory
from holographic_prod.core.grounded_embeddings import (
    compute_cooccurrence_from_corpus,
    create_grounded_embeddings,
)
from holographic_prod.core.algebra import frobenius_cosine


def create_structured_corpus(n_sentences: int = 1000, vocab_size: int = 100) -> List[List[int]]:
    """
    Create corpus with semantic structure (tokens that co-occur form groups).
    
    Structure:
    - Tokens 10-19 appear together (like "animals")
    - Tokens 20-29 appear together (like "actions")
    - Tokens 30-39 appear together (like "places")
    """
    np.random.seed(42)
    corpus = []
    
    for _ in range(n_sentences):
        group = np.random.choice([0, 1, 2])
        if group == 0:
            # "Animal" sentence: [article, animal, action, place]
            sentence = [
                1,  # article
                np.random.randint(10, 20),  # animal
                np.random.randint(20, 30),  # action
                np.random.randint(30, 40),  # place
            ]
        elif group == 1:
            # "Action" sentence: [subject, action, adverb, place]
            sentence = [
                np.random.randint(10, 20),  # subject
                np.random.randint(20, 30),  # action
                2,  # adverb
                np.random.randint(30, 40),  # place
            ]
        else:
            # "Place" sentence: [article, place, verb, animal]
            sentence = [
                1,  # article
                np.random.randint(30, 40),  # place
                3,  # verb
                np.random.randint(10, 20),  # animal
            ]
        corpus.append(sentence)
    
    return corpus


def test_e2e_grounded_vs_random():
    """
    Compare training with grounded vs random embeddings.
    
    Expected: Grounded embeddings should achieve higher accuracy
    with fewer samples due to O(√N) sample complexity.
    """
    print("=" * 70)
    print("END-TO-END TEST: Grounded vs Random Embeddings")
    print("=" * 70)
    
    vocab_size = 100
    n_corpus = 1000
    n_train = 500
    
    # Create corpus
    print("\n1. Creating structured corpus...")
    corpus = create_structured_corpus(n_corpus, vocab_size)
    print(f"   {len(corpus)} sentences, vocab_size={vocab_size}")
    
    # Create training samples from corpus
    train_samples = []
    for sentence in corpus[:n_train]:
        if len(sentence) >= 4:
            ctx = sentence[:3]
            tgt = sentence[3]
            train_samples.append((ctx, tgt))
    print(f"   {len(train_samples)} training samples")
    
    # Create grounded embeddings
    print("\n2. Computing grounded embeddings...")
    start = time.time()
    cooccur = compute_cooccurrence_from_corpus(corpus, vocab_size)
    grounded = create_grounded_embeddings(cooccur, vocab_size)
    grounding_time = time.time() - start
    print(f"   Grounding completed in {grounding_time:.2f}s")
    
    # Check semantic structure in grounded embeddings
    within_group_sims = []
    for base in [10, 20, 30]:
        for i in range(base, base + 5):
            for j in range(i + 1, base + 10):
                if j < vocab_size:
                    sim = frobenius_cosine(grounded[i], grounded[j], np)
                    within_group_sims.append(sim)
    
    print(f"   Within-group similarity: {np.mean(within_group_sims):.3f}")
    
    # Train with GROUNDED embeddings
    print("\n3. Training with GROUNDED embeddings...")
    start = time.time()
    model_grounded = HolographicMemory(
        vocab_size=vocab_size, 
        max_levels=2,
        grounded_embeddings=grounded,
        use_gpu=False
    )
    
    for ctx, tgt in train_samples:
        model_grounded.learn(ctx, tgt)
    
    grounded_train_time = time.time() - start
    
    # Evaluate grounded
    correct_grounded = 0
    for ctx, tgt in train_samples[:100]:
        pred, _ = model_grounded.retrieve_deterministic(ctx)
        if pred == tgt:
            correct_grounded += 1
    
    grounded_accuracy = correct_grounded / 100
    print(f"   Training time: {grounded_train_time:.2f}s")
    print(f"   Accuracy: {grounded_accuracy:.1%}")
    
    # Train with RANDOM embeddings
    print("\n4. Training with RANDOM embeddings...")
    start = time.time()
    model_random = HolographicMemory(
        vocab_size=vocab_size, 
        max_levels=2,
        grounded_embeddings=None,  # Random
        use_gpu=False
    )
    
    for ctx, tgt in train_samples:
        model_random.learn(ctx, tgt)
    
    random_train_time = time.time() - start
    
    # Evaluate random
    correct_random = 0
    for ctx, tgt in train_samples[:100]:
        pred, _ = model_random.retrieve_deterministic(ctx)
        if pred == tgt:
            correct_random += 1
    
    random_accuracy = correct_random / 100
    print(f"   Training time: {random_train_time:.2f}s")
    print(f"   Accuracy: {random_accuracy:.1%}")
    
    # Test GENERALIZATION (unseen but similar contexts)
    print("\n5. Testing GENERALIZATION to unseen contexts...")
    
    # Create test contexts that are similar to training but not identical
    test_contexts = []
    for _ in range(50):
        # Similar structure to training data
        animal = np.random.randint(10, 20)
        action = np.random.randint(20, 30)
        place = np.random.randint(30, 40)
        ctx = [1, animal, action]  # Same structure
        tgt = place
        test_contexts.append((ctx, tgt))
    
    # Evaluate on grounded
    correct_gen_grounded = 0
    for ctx, tgt in test_contexts:
        pred, _ = model_grounded.retrieve_deterministic(ctx)
        if pred == tgt:
            correct_gen_grounded += 1
    
    gen_grounded = correct_gen_grounded / len(test_contexts)
    
    # Evaluate on random
    correct_gen_random = 0
    for ctx, tgt in test_contexts:
        pred, _ = model_random.retrieve_deterministic(ctx)
        if pred == tgt:
            correct_gen_random += 1
    
    gen_random = correct_gen_random / len(test_contexts)
    
    print(f"   Grounded generalization: {gen_grounded:.1%}")
    print(f"   Random generalization: {gen_random:.1%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"                    | Grounded    | Random")
    print(f"  Training accuracy | {grounded_accuracy:>6.1%}      | {random_accuracy:>6.1%}")
    print(f"  Generalization    | {gen_grounded:>6.1%}      | {gen_random:>6.1%}")
    print(f"  Training time     | {grounded_train_time:>6.2f}s     | {random_train_time:>6.2f}s")
    print(f"  Grounding time    | {grounding_time:>6.2f}s     | N/A")
    
    improvement = grounded_accuracy - random_accuracy
    print(f"\n  Accuracy improvement: {improvement:+.1%}")
    
    if grounded_accuracy >= random_accuracy:
        print("\n✓ Grounded embeddings perform at least as well as random")
    else:
        print("\n⚠️ Unexpected: Random performed better (check test setup)")
    
    return {
        'grounded_accuracy': grounded_accuracy,
        'random_accuracy': random_accuracy,
        'grounded_generalization': gen_grounded,
        'random_generalization': gen_random,
    }


def test_grounding_performance():
    """
    Test that grounding phase is fast enough for production.
    """
    print("\n" + "=" * 70)
    print("GROUNDING PERFORMANCE TEST")
    print("=" * 70)
    
    vocab_sizes = [1000, 5000, 10000]
    corpus_sizes = [1000, 5000, 10000]
    
    results = []
    
    for vocab_size in vocab_sizes:
        for corpus_size in corpus_sizes:
            # Generate corpus
            np.random.seed(42)
            corpus = [list(np.random.randint(0, vocab_size, size=10)) for _ in range(corpus_size)]
            
            # Time grounding
            start = time.time()
            cooccur = compute_cooccurrence_from_corpus(corpus, vocab_size)
            grounded = create_grounded_embeddings(cooccur, vocab_size)
            elapsed = time.time() - start
            
            results.append({
                'vocab_size': vocab_size,
                'corpus_size': corpus_size,
                'time': elapsed,
            })
            
            print(f"  vocab={vocab_size:,}, corpus={corpus_size:,}: {elapsed:.2f}s")
    
    # Check performance targets
    print("\n  Performance targets:")
    for r in results:
        if r['vocab_size'] <= 5000 and r['corpus_size'] <= 5000:
            target = 5.0
        else:
            target = 30.0
        
        status = "✓" if r['time'] < target else "✗"
        print(f"  {status} vocab={r['vocab_size']:,}, corpus={r['corpus_size']:,}: "
              f"{r['time']:.2f}s (target: <{target}s)")
    
    return results


if __name__ == "__main__":
    results = test_e2e_grounded_vs_random()
    perf_results = test_grounding_performance()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
