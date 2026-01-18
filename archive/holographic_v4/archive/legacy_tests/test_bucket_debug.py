"""
DEBUG: Why does retrieval fail when similarity = 1.0?

The semantic embeddings create perfect similarity, but bucketing fails.
Let's see what's happening with the 8D keys.
"""

import numpy as np
from typing import Dict, List

def simple_tokenize(text: str) -> List[str]:
    return text.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').split()

def create_semantic_embeddings(vocab: Dict[str, int], embedding_dim: int = 4) -> np.ndarray:
    """Same as before."""
    n_words = len(vocab)
    embeddings = np.zeros((n_words, embedding_dim, embedding_dim))
    for i in range(n_words):
        embeddings[i] = np.eye(embedding_dim)
    
    semantic_clusters = {
        'cat': np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        'feline': np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        'dog': np.array([[0.1, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        'canine': np.array([[0.1, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        'sat': np.array([[0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0]]),
        'rested': np.array([[0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0]]),
        'ran': np.array([[0, 0, 0.1, 0], [0, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0]]),
        'jogged': np.array([[0, 0, 0.1, 0], [0, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0]]),
    }
    
    for word, signature in semantic_clusters.items():
        if word in vocab:
            idx = vocab[word]
            embeddings[idx] = np.eye(embedding_dim) + signature
    
    np.random.seed(42)
    for i in range(n_words):
        embeddings[i] += 0.01 * np.random.randn(embedding_dim, embedding_dim)
    
    return embeddings.astype(np.float32)


def main():
    from holographic_v4.algebra import build_clifford_basis, geometric_product, grace_operator
    from holographic_v4.holographic_memory import VorticityWitnessIndex
    from holographic_v4.quotient import extract_witness
    
    print("\n" + "="*60)
    print("DEBUG: Bucket Key Analysis")
    print("="*60)
    
    # Setup
    test_cases = [
        ("the cat sat on the", "mat", "the feline rested on the"),
        ("the dog ran in the", "park", "the canine jogged in the"),
    ]
    
    all_words = set()
    for t, target, p in test_cases:
        all_words.update(simple_tokenize(t))
        all_words.add(target)
        all_words.update(simple_tokenize(p))
    
    vocab = {w: i for i, w in enumerate(sorted(all_words))}
    embeddings = create_semantic_embeddings(vocab)
    basis = build_clifford_basis()
    memory = VorticityWitnessIndex.create(basis, xp=np)
    
    def compute_context(tokens):
        token_ids = [vocab.get(t, 0) for t in tokens]
        ctx = np.eye(4, dtype=np.float32)
        for tid in token_ids:
            ctx = geometric_product(ctx, embeddings[tid])
        return grace_operator(ctx, basis, np)
    
    print("\n  Analyzing bucket keys:")
    print("  " + "-"*70)
    
    for train_ctx, target, para_ctx in test_cases:
        train_tokens = simple_tokenize(train_ctx)
        para_tokens = simple_tokenize(para_ctx)
        
        ctx_train = compute_context(train_tokens)
        ctx_para = compute_context(para_tokens)
        
        # Get 8D keys
        key_train = memory._vorticity_key(ctx_train)
        key_para = memory._vorticity_key(ctx_para)
        
        # Get witnesses
        wit_train = extract_witness(ctx_train, basis, np)
        wit_para = extract_witness(ctx_para, basis, np)
        
        print(f"\n  '{train_ctx}' vs '{para_ctx}'")
        print(f"    Train witness: σ={wit_train[0]:.4f}, p={wit_train[1]:.4f}")
        print(f"    Para witness:  σ={wit_para[0]:.4f}, p={wit_para[1]:.4f}")
        print(f"    Train key: {key_train}")
        print(f"    Para key:  {key_para}")
        print(f"    Keys match: {key_train == key_para}")
        
        # Check difference in each dimension
        diffs = [abs(a - b) for a, b in zip(key_train, key_para)]
        print(f"    Key diffs: {diffs}")
    
    print("\n  " + "="*70)
    print("\n  CONCLUSION:")
    if all(memory._vorticity_key(compute_context(simple_tokenize(t))) == 
           memory._vorticity_key(compute_context(simple_tokenize(p)))
           for t, _, p in test_cases):
        print("    Keys MATCH - issue is elsewhere")
    else:
        print("    Keys DON'T MATCH - bucketing resolution is too fine")
        print("    Even with similarity=1.0, tiny differences cause bucket mismatch")
        print("\n    SOLUTION: Use coarser bucketing for semantic retrieval")


if __name__ == "__main__":
    main()
