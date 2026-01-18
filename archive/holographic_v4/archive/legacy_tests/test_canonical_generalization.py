"""
TEST: Generalization with Proper Canonicalization

PROBLEM IDENTIFIED:
    Contexts with p=-0.004 and p=+0.004 go to different buckets.
    But per theory (bireflection), p and -p are EQUIVALENT!
    
SOLUTION:
    Canonicalize the witness using abs(p) for bucketing.
    This respects the bireflection symmetry σ ↔ 1-σ, p ↔ -p.

Run: python -m holographic_v4.test_canonical_generalization
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_INV_SQ = PHI_INV ** 2
PHI_INV_THREE = PHI_INV ** 3


def simple_tokenize(text: str) -> List[str]:
    return text.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').split()


def create_semantic_embeddings(vocab: Dict[str, int]) -> np.ndarray:
    """Create embeddings where synonyms are similar."""
    n_words = len(vocab)
    embeddings = np.zeros((n_words, 4, 4), dtype=np.float32)
    for i in range(n_words):
        embeddings[i] = np.eye(4)
    
    clusters = {
        'cat': np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        'feline': np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        'dog': np.array([[0.1, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        'canine': np.array([[0.1, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        'bird': np.array([[0.1, 0, 0, 0], [0, 0, 0, 0.1], [0, 0, 0, 0], [0, 0, 0, 0]]),
        'sparrow': np.array([[0.1, 0, 0, 0], [0, 0, 0, 0.1], [0, 0, 0, 0], [0, 0, 0, 0]]),
        'sat': np.array([[0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0]]),
        'rested': np.array([[0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0]]),
        'ran': np.array([[0, 0, 0.1, 0], [0, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0]]),
        'jogged': np.array([[0, 0, 0.1, 0], [0, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0]]),
        'flew': np.array([[0, 0, 0, 0.1], [0, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0]]),
        'soared': np.array([[0, 0, 0, 0.1], [0, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0]]),
        'ate': np.array([[0, 0, 0, 0], [0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0]]),
        'consumed': np.array([[0, 0, 0, 0], [0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0]]),
        'red': np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0.1, 0, 0, 0]]),
        'crimson': np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0.1, 0, 0, 0]]),
    }
    
    for word, sig in clusters.items():
        if word in vocab:
            embeddings[vocab[word]] = np.eye(4) + sig
    
    np.random.seed(42)
    for i in range(n_words):
        embeddings[i] += 0.01 * np.random.randn(4, 4)
    
    return embeddings.astype(np.float32)


@dataclass
class CanonicalSemanticIndex:
    """
    Semantic index with bireflection-aware canonicalization.
    
    KEY INSIGHT:
        Per theory, p and -p are equivalent (bireflection maps p → -p).
        We use abs(p) for bucketing to respect this symmetry.
        
    This also uses COARSER resolution (φ⁻³ instead of φ⁻⁶) for generalization.
    """
    buckets: Dict[Tuple, List[Tuple[np.ndarray, np.ndarray, int]]] = field(default_factory=dict)
    targets: Dict[Tuple, List[int]] = field(default_factory=dict)
    resolution: float = PHI_INV_THREE  # COARSER: φ⁻³ ≈ 0.236 (was φ⁻⁶ ≈ 0.056)
    basis: np.ndarray = None
    n_items: int = 0
    
    @classmethod
    def create(cls, basis: np.ndarray, resolution: float = PHI_INV_THREE):
        return cls(buckets={}, targets={}, resolution=resolution, basis=basis)
    
    def _canonical_key(self, M: np.ndarray) -> Tuple[int, int]:
        """
        Compute canonical 2D key using bireflection symmetry.
        
        THEORY:
            - σ (scalar) is the primary semantic coordinate
            - |p| (abs pseudoscalar) respects p ↔ -p symmetry
        """
        # Extract witness
        sigma = float(np.sum(self.basis[0] * M) / 4.0)
        pseudo = float(np.sum(self.basis[15] * M) / 4.0)
        
        # Canonicalize: use abs(pseudo) because p ↔ -p under bireflection
        abs_pseudo = abs(pseudo)
        
        # Quantize with COARSE resolution
        s_idx = int(np.floor(sigma / self.resolution))
        p_idx = int(np.floor(abs_pseudo / self.resolution))
        
        return (s_idx, p_idx)
    
    def store(self, context: np.ndarray, target: np.ndarray, target_idx: int):
        """Store with canonical key."""
        key = self._canonical_key(context)
        if key not in self.buckets:
            self.buckets[key] = []
            self.targets[key] = []
        self.buckets[key].append((context.copy(), target.copy(), target_idx))
        self.targets[key].append(target_idx)
        self.n_items += 1
    
    def retrieve(self, context: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """Retrieve using canonical key and within-bucket similarity."""
        from holographic_v4.quotient import vorticity_similarity
        
        key = self._canonical_key(context)
        
        if key not in self.buckets:
            return None, None, 0.0
        
        # Find best match in bucket using combined similarity
        best_match = None
        best_idx = None
        best_sim = -1
        
        for ctx, tgt, idx in self.buckets[key]:
            sim = vorticity_similarity(context, ctx, self.basis, np)
            if sim > best_sim:
                best_sim = sim
                best_match = tgt
                best_idx = idx
        
        return best_match, best_idx, best_sim


def test_canonical_generalization() -> Dict:
    """Test that canonical indexing enables generalization."""
    print("\n" + "="*60)
    print("TEST: Canonical Semantic Generalization")
    print("="*60)
    
    from holographic_v4.algebra import build_clifford_basis, geometric_product, grace_operator
    from holographic_v4.quotient import extract_witness
    
    test_cases = [
        ("the cat sat on the", "mat", "the feline rested on the"),
        ("the dog ran in the", "park", "the canine jogged in the"),
        ("the bird flew to the", "tree", "the sparrow soared to the"),
        ("she ate a red", "apple", "she consumed a crimson"),
    ]
    
    # Build vocab
    all_words = set()
    for t, target, p in test_cases:
        all_words.update(simple_tokenize(t))
        all_words.add(target)
        all_words.update(simple_tokenize(p))
    
    vocab = {w: i for i, w in enumerate(sorted(all_words))}
    embeddings = create_semantic_embeddings(vocab)
    basis = build_clifford_basis()
    
    # Create canonical index
    semantic_index = CanonicalSemanticIndex.create(basis)
    
    def compute_context(tokens):
        token_ids = [vocab.get(t, 0) for t in tokens]
        ctx = np.eye(4, dtype=np.float32)
        for tid in token_ids:
            ctx = geometric_product(ctx, embeddings[tid])
        return grace_operator(ctx, basis, np)
    
    print(f"\n  Resolution: φ⁻³ ≈ {PHI_INV_THREE:.3f} (coarse for generalization)")
    
    # Store training
    print("\n  Training:")
    for train_ctx, target, _ in test_cases:
        tokens = simple_tokenize(train_ctx)
        target_idx = vocab.get(target, 0)
        ctx = compute_context(tokens)
        target_emb = embeddings[target_idx]
        
        key = semantic_index._canonical_key(ctx)
        semantic_index.store(ctx, target_emb, target_idx)
        print(f"    Stored: '{train_ctx}' → '{target}' (key={key})")
    
    # Test paraphrases
    print("\n  Testing paraphrases:")
    correct = 0
    for train_ctx, target, para_ctx in test_cases:
        train_tokens = simple_tokenize(train_ctx)
        para_tokens = simple_tokenize(para_ctx)
        target_idx = vocab.get(target, 0)
        
        train_context = compute_context(train_tokens)
        para_context = compute_context(para_tokens)
        
        train_key = semantic_index._canonical_key(train_context)
        para_key = semantic_index._canonical_key(para_context)
        
        result, idx, conf = semantic_index.retrieve(para_context)
        
        key_match = "✓" if train_key == para_key else "✗"
        
        if idx == target_idx:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"    {status} '{para_ctx}' key={para_key} match={key_match} → expected '{target}', got idx={idx}")
    
    accuracy = correct / len(test_cases)
    
    print(f"\n  Accuracy: {accuracy:.1%} ({correct}/{len(test_cases)})")
    
    passed = accuracy >= 0.75  # Should get most right with canonical bucketing
    
    if passed:
        print(f"\n  ✅ PASS: Canonical bucketing enables generalization!")
    else:
        print(f"\n  ⚠️ PARTIAL: {accuracy:.1%} generalization")
    
    return {'test': 'canonical_generalization', 'accuracy': accuracy, 'passed': passed}


def test_key_stability() -> Dict:
    """Verify that canonical keys are stable across near-identical contexts."""
    print("\n" + "="*60)
    print("TEST: Canonical Key Stability")
    print("="*60)
    
    from holographic_v4.algebra import build_clifford_basis, grace_operator
    
    basis = build_clifford_basis()
    index = CanonicalSemanticIndex.create(basis)
    
    # Create base context
    np.random.seed(42)
    base = np.eye(4) + 0.1 * np.random.randn(4, 4)
    base = grace_operator(base.astype(np.float32), basis, np)
    base_key = index._canonical_key(base)
    
    print(f"\n  Base key: {base_key}")
    
    # Add small perturbations
    n_stable = 0
    n_tests = 20
    
    for i in range(n_tests):
        perturbed = base + 0.02 * np.random.randn(4, 4)
        perturbed = grace_operator(perturbed.astype(np.float32), basis, np)
        pert_key = index._canonical_key(perturbed)
        
        if pert_key == base_key:
            n_stable += 1
    
    stability = n_stable / n_tests
    
    print(f"  Small perturbations: {n_stable}/{n_tests} ({stability:.1%}) same key")
    
    passed = stability >= 0.5  # At least 50% stability
    
    if passed:
        print(f"\n  ✅ PASS: Keys are stable under perturbation")
    else:
        print(f"\n  ❌ FAIL: Keys are too unstable")
    
    return {'test': 'key_stability', 'stability': stability, 'passed': passed}


def run_all_tests() -> bool:
    """Run all tests."""
    print("\n" + "="*70)
    print("CANONICAL GENERALIZATION TEST")
    print("="*70)
    print("""
SOLUTION TO BUCKET MISMATCH:
    1. Use abs(p) for bucketing (respects bireflection symmetry p ↔ -p)
    2. Use coarser resolution φ⁻³ instead of φ⁻⁶
    3. This creates "semantic buckets" where similar meanings cluster
""")
    
    results = []
    
    results.append(test_key_stability())
    results.append(test_canonical_generalization())
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    for r in results:
        status = "✅" if r['passed'] else "❌"
        print(f"  {status} {r['test']}")
    
    print(f"\n  TOTAL: {passed}/{total} passed")
    
    if passed == total:
        print("""
  🎉 THEORY-TRUE GENERALIZATION ACHIEVED!
  
  The fix was simple but theory-grounded:
  1. Bireflection symmetry: p ↔ -p means use abs(p)
  2. Coarser resolution: φ⁻³ creates semantic neighborhoods
  
  This enables real text generalization while staying on theory.
""")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    import sys
    sys.exit(0 if success else 1)
