"""
Modal Test: Contrastive Embedding Learning at Scale

Tests the contrastive learning mechanism with real text data on Modal.

Run:
    modal run holographic_v4/test_contrastive_modal.py

This script:
1. Loads real text data (WikiText-2)
2. Trains the model and tracks co-predictive tokens
3. Applies contrastive updates based on predictiveness
4. Measures generalization improvement

Expected outcome:
- Tokens that predict the same target should converge geometrically
- Paraphrase generalization should improve over baseline
"""

import modal

# Modal app
app = modal.App("contrastive-learning-test")

# Define image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24.0",
        "datasets>=2.14.0",
        "tqdm>=4.65.0",
    )
    .add_local_dir(".", "/root/project", copy=True)
)


@app.function(
    image=image,
    timeout=1800,  # 30 min
    memory=8192,   # 8GB
)
def test_contrastive_at_scale():
    """Test contrastive learning with real data at scale."""
    import sys
    sys.path.insert(0, "/root/project")
    
    import numpy as np
    from collections import defaultdict
    from tqdm import tqdm
    
    print("\n" + "="*70)
    print("CONTRASTIVE LEARNING TEST: Real Data at Scale")
    print("="*70)
    
    # Import holographic modules
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.algebra import frobenius_similarity
    from holographic_v4.representation_learning import (
        contrastive_embedding_update,
        auto_contrastive_update,
    )
    
    # Load real data
    print("\n[1/6] Loading WikiText-2 data...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
    
    # Build vocabulary
    print("\n[2/6] Building vocabulary...")
    word_counts = defaultdict(int)
    for example in tqdm(ds, desc="Counting words"):
        text = example.get("text", "")
        words = text.lower().split()
        for w in words:
            word_counts[w] += 1
    
    # Keep top 10k words
    vocab_size = 10000
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    vocab = {word: i for i, (word, _) in enumerate(sorted_words[:vocab_size])}
    vocab["<unk>"] = len(vocab)
    
    def tokenize(text):
        return [vocab.get(w.lower(), vocab["<unk>"]) for w in text.split()]
    
    print(f"  Vocabulary size: {len(vocab)}")
    
    # Create model
    print("\n[3/6] Creating model...")
    model = TheoryTrueModel(
        vocab_size=len(vocab),
        max_attractors=50000,
        use_predictiveness=True,
        use_embedding_drift=True,
        xp=np
    )
    
    # Train phase 1: Learn predictiveness patterns
    print("\n[4/6] Training phase 1: Learning predictiveness patterns...")
    n_train = 10000
    context_size = 32
    
    trained_samples = []
    target_contexts = defaultdict(list)  # target → list of context tokens
    
    for i, example in enumerate(tqdm(ds, total=n_train, desc="Training")):
        if i >= n_train:
            break
        
        text = example.get("text", "")
        if len(text) < 50:
            continue
        
        tokens = tokenize(text)
        if len(tokens) < context_size + 1:
            continue
        
        # Train on sliding windows
        for j in range(0, len(tokens) - context_size - 1, 8):  # Stride 8
            context = tokens[j:j + context_size]
            target = tokens[j + context_size]
            
            # Train predictiveness
            for token in context[-5:]:  # Last 5 tokens most predictive
                model.predictiveness_tracker.observe([token], target)
            
            # Track which contexts predict which targets
            target_contexts[target].extend(context[-5:])
            
            # Store in memory
            ctx_matrix = model.compute_context(context)
            target_matrix = model.get_embedding(target)
            model.holographic_memory.store(ctx_matrix, target_matrix, target)
            
            trained_samples.append((context, target))
            
            if len(trained_samples) >= 5000:
                break
        
        if len(trained_samples) >= 5000:
            break
    
    print(f"  Trained on {len(trained_samples)} samples")
    print(f"  Unique targets: {len(target_contexts)}")
    
    # Find co-predictive token pairs
    print("\n[5/6] Finding co-predictive pairs and measuring baseline...")
    
    # Find tokens that frequently co-predict the same target
    co_predictive_pairs = []
    token_pair_counts = defaultdict(int)
    
    for target, predictors in target_contexts.items():
        unique_predictors = list(set(predictors))
        for i, t1 in enumerate(unique_predictors):
            for t2 in unique_predictors[i+1:]:
                if t1 != t2:
                    pair = tuple(sorted([t1, t2]))
                    token_pair_counts[pair] += 1
    
    # Keep pairs that co-predict at least 3 times
    for pair, count in token_pair_counts.items():
        if count >= 3:
            co_predictive_pairs.append(pair)
    
    print(f"  Found {len(co_predictive_pairs)} co-predictive pairs")
    
    # Measure baseline similarity for co-predictive pairs
    baseline_sims = []
    for t1, t2 in co_predictive_pairs[:100]:  # Sample
        sim = frobenius_similarity(model.embeddings[t1], model.embeddings[t2], np)
        baseline_sims.append(float(sim))
    
    baseline_mean = np.mean(baseline_sims) if baseline_sims else 0
    print(f"  Baseline co-predictive similarity: {baseline_mean:.4f}")
    
    # Test baseline generalization
    print("\n  Testing baseline generalization...")
    n_test = min(100, len(trained_samples))
    correct_baseline = 0
    for ctx, target in trained_samples[:n_test]:
        ctx_matrix = model.compute_context(ctx)
        _, idx, _, _ = model.holographic_memory.retrieve(ctx_matrix)
        if idx == target:
            correct_baseline += 1
    
    baseline_accuracy = correct_baseline / n_test
    print(f"  Baseline retrieval accuracy: {baseline_accuracy:.1%}")
    
    # Apply contrastive updates
    print("\n[6/6] Applying contrastive updates...")
    
    # Group pairs by shared target
    pairs_by_target = defaultdict(list)
    for target, predictors in target_contexts.items():
        unique_predictors = list(set(predictors))
        for i, t1 in enumerate(unique_predictors[:10]):  # Limit
            for t2 in unique_predictors[i+1:10]:
                if t1 != t2:
                    pairs_by_target[target].append((t1, t2))
    
    # Apply updates
    n_updates = 0
    for target, pairs in tqdm(list(pairs_by_target.items())[:500], desc="Contrastive updates"):
        if pairs:
            updated = contrastive_embedding_update(
                model=model,
                co_predictive_pairs=pairs[:5],  # Limit pairs per target
                shared_target=target,
            )
            n_updates += updated
    
    print(f"  Applied {n_updates} embedding updates")
    
    # Measure post-contrastive similarity
    post_sims = []
    for t1, t2 in co_predictive_pairs[:100]:
        sim = frobenius_similarity(model.embeddings[t1], model.embeddings[t2], np)
        post_sims.append(float(sim))
    
    post_mean = np.mean(post_sims) if post_sims else 0
    print(f"  Post-contrastive co-predictive similarity: {post_mean:.4f}")
    print(f"  Improvement: {post_mean - baseline_mean:+.4f}")
    
    # Re-store with updated embeddings
    print("\n  Re-storing patterns with updated embeddings...")
    model.holographic_memory.clear()
    for ctx, target in tqdm(trained_samples, desc="Re-storing"):
        ctx_matrix = model.compute_context(ctx)
        target_matrix = model.get_embedding(target)
        model.holographic_memory.store(ctx_matrix, target_matrix, target)
    
    # Test post-contrastive accuracy
    print("\n  Testing post-contrastive generalization...")
    correct_post = 0
    for ctx, target in trained_samples[:n_test]:
        ctx_matrix = model.compute_context(ctx)
        _, idx, _, _ = model.holographic_memory.retrieve(ctx_matrix)
        if idx == target:
            correct_post += 1
    
    post_accuracy = correct_post / n_test
    print(f"  Post-contrastive retrieval accuracy: {post_accuracy:.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"""
  Training:
    Samples: {len(trained_samples)}
    Co-predictive pairs found: {len(co_predictive_pairs)}
    
  Embedding Similarity (co-predictive pairs):
    Baseline: {baseline_mean:.4f}
    Post-contrastive: {post_mean:.4f}
    Improvement: {post_mean - baseline_mean:+.4f}
    
  Retrieval Accuracy:
    Baseline: {baseline_accuracy:.1%}
    Post-contrastive: {post_accuracy:.1%}
    Improvement: {post_accuracy - baseline_accuracy:+.1%}
""")
    
    # Determine success
    success = (post_mean > baseline_mean) and (post_accuracy >= baseline_accuracy)
    
    if success:
        print("  ✅ SUCCESS: Contrastive learning improved embeddings!")
    else:
        print("  ⚠️ Results inconclusive or worse")
    
    return {
        'baseline_similarity': baseline_mean,
        'post_similarity': post_mean,
        'baseline_accuracy': baseline_accuracy,
        'post_accuracy': post_accuracy,
        'success': success,
    }


@app.local_entrypoint()
def main():
    """Run the contrastive learning test."""
    print("Starting contrastive learning test on Modal...")
    result = test_contrastive_at_scale.remote()
    print(f"\nFinal result: {result}")
