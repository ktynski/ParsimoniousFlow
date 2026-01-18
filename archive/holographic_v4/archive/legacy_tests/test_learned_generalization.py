"""
TEST: Can the system LEARN to generalize without hand-crafted embeddings?

THIS IS THE REAL TEST:
    - NO hand-crafted semantic embeddings
    - System must LEARN that similar contexts predict similar targets
    - Generalization emerges from training data, not from us telling it synonyms

THEORY-TRUE APPROACH:
    1. Train on enough data that co-occurrence reveals patterns
    2. Predictiveness tracker identifies content vs function words
    3. Dreaming consolidates similar-predicting contexts into prototypes
    4. Retrieval uses prototypes for generalization

Run: python -m holographic_v4.test_learned_generalization
"""

import numpy as np
from typing import Dict, List, Tuple
import sys

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_INV_SQ = PHI_INV ** 2


def simple_tokenize(text: str) -> List[str]:
    return text.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').split()


def test_learned_generalization_via_predictiveness() -> Dict:
    """
    Test if predictiveness tracking enables generalization.
    
    THEORY:
        Tokens that predict the same target become "semantically similar"
        because they have similar predictiveness profiles.
        
        If "cat" and "feline" both predict "meows", they should cluster.
    """
    print("\n" + "="*60)
    print("TEST: Learned Generalization via Predictiveness")
    print("="*60)
    
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.quotient import witness_similarity
    
    # Training data: synonyms appear in SAME contexts predicting SAME targets
    # The system should LEARN that these words are related
    training_data = [
        # Cat/feline contexts → "meows"
        ("the cat said", "meows"),
        ("a cat said", "meows"),
        ("the feline said", "meows"),
        ("a feline said", "meows"),
        
        # Dog/canine contexts → "barks"  
        ("the dog said", "barks"),
        ("a dog said", "barks"),
        ("the canine said", "barks"),
        ("a canine said", "barks"),
        
        # More examples to reinforce
        ("my cat always", "meows"),
        ("my feline always", "meows"),
        ("my dog always", "barks"),
        ("my canine always", "barks"),
    ]
    
    # Build vocabulary
    all_text = " ".join([f"{ctx} {tgt}" for ctx, tgt in training_data])
    vocab = {w: i for i, w in enumerate(sorted(set(simple_tokenize(all_text))))}
    print(f"\n  Vocabulary: {len(vocab)} words")
    
    # Create model with DEFAULT (random) embeddings
    model = TheoryTrueModel(
        vocab_size=len(vocab) + 10,
        max_attractors=1000,
        use_predictiveness=True,
        xp=np
    )
    
    # Train: record predictiveness for each token
    print("\n  Training predictiveness tracker...")
    for ctx_text, target_text in training_data:
        ctx_tokens = [vocab.get(t, 0) for t in simple_tokenize(ctx_text)]
        target_idx = vocab.get(target_text, 0)
        
        for token in ctx_tokens:
            model.predictiveness_tracker.observe([token], target_idx)
    
    # Check: do synonyms now have similar predictiveness?
    print("\n  Learned predictiveness (who predicts what):")
    
    cat_idx = vocab.get('cat', 0)
    feline_idx = vocab.get('feline', 0)
    dog_idx = vocab.get('dog', 0)
    canine_idx = vocab.get('canine', 0)
    
    cat_pred = model.predictiveness_tracker.predictiveness(cat_idx)
    feline_pred = model.predictiveness_tracker.predictiveness(feline_idx)
    dog_pred = model.predictiveness_tracker.predictiveness(dog_idx)
    canine_pred = model.predictiveness_tracker.predictiveness(canine_idx)
    
    print(f"    cat: {cat_pred:.3f}")
    print(f"    feline: {feline_pred:.3f}")
    print(f"    dog: {dog_pred:.3f}")
    print(f"    canine: {canine_pred:.3f}")
    
    # Check: are cat/feline and dog/canine similar?
    cat_feline_similar = abs(cat_pred - feline_pred) < 0.1
    dog_canine_similar = abs(dog_pred - canine_pred) < 0.1
    
    print(f"\n  cat ≈ feline? {cat_feline_similar} (diff={abs(cat_pred - feline_pred):.3f})")
    print(f"  dog ≈ canine? {dog_canine_similar} (diff={abs(dog_pred - canine_pred):.3f})")
    
    # Now test: can semantic contexts generalize?
    print("\n  Testing semantic context similarity...")
    
    # Compute semantic contexts for synonymous sentences
    ctx1 = model.compute_semantic_context([vocab.get(t, 0) for t in simple_tokenize("the cat said")])
    ctx2 = model.compute_semantic_context([vocab.get(t, 0) for t in simple_tokenize("the feline said")])
    ctx3 = model.compute_semantic_context([vocab.get(t, 0) for t in simple_tokenize("the dog said")])
    
    sim_cat_feline = witness_similarity(ctx1, ctx2, model.basis, np)
    sim_cat_dog = witness_similarity(ctx1, ctx3, model.basis, np)
    
    print(f"    'the cat said' ~ 'the feline said': {sim_cat_feline:.3f}")
    print(f"    'the cat said' ~ 'the dog said': {sim_cat_dog:.3f}")
    
    # Success if synonyms more similar than non-synonyms
    synonyms_closer = sim_cat_feline > sim_cat_dog
    
    passed = cat_feline_similar and dog_canine_similar and synonyms_closer
    
    if passed:
        print(f"\n  ✅ PASS: Predictiveness LEARNS semantic similarity!")
    else:
        print(f"\n  ❌ FAIL: Predictiveness alone insufficient")
        if not cat_feline_similar:
            print("     - cat/feline predictiveness differs")
        if not dog_canine_similar:
            print("     - dog/canine predictiveness differs")
        if not synonyms_closer:
            print("     - Synonymous contexts not more similar")
    
    return {
        'test': 'learned_generalization_via_predictiveness',
        'cat_feline_similar': cat_feline_similar,
        'dog_canine_similar': dog_canine_similar,
        'synonyms_closer': synonyms_closer,
        'passed': passed
    }


def test_learned_generalization_via_dreaming() -> Dict:
    """
    Test if dreaming consolidation enables generalization.
    
    THEORY:
        Episodes that predict the same target should cluster into prototypes.
        Retrieval from prototypes enables generalization.
    """
    print("\n" + "="*60)
    print("TEST: Learned Generalization via Dreaming")
    print("="*60)
    
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.dreaming import DreamingSystem, EpisodicEntry
    from holographic_v4.constants import DTYPE, PHI_INV_SQ
    
    # Same training data
    training_data = [
        ("the cat said", "meows"),
        ("a cat said", "meows"),
        ("the feline said", "meows"),
        ("a feline said", "meows"),
        ("the dog said", "barks"),
        ("a dog said", "barks"),
        ("the canine said", "barks"),
        ("a canine said", "barks"),
    ]
    
    all_text = " ".join([f"{ctx} {tgt}" for ctx, tgt in training_data])
    vocab = {w: i for i, w in enumerate(sorted(set(simple_tokenize(all_text))))}
    
    model = TheoryTrueModel(
        vocab_size=len(vocab) + 10,
        max_attractors=1000,
        use_predictiveness=True,
        xp=np
    )
    
    dreaming = DreamingSystem(
        basis=model.basis,
        use_salience=False,
        use_novelty=False,
        similarity_threshold=PHI_INV_SQ,
    )
    
    # Create episodes
    episodes = []
    for ctx_text, target_text in training_data:
        ctx_tokens = [vocab.get(t, 0) for t in simple_tokenize(ctx_text)]
        target_idx = vocab.get(target_text, 0)
        
        ctx_matrix = model.compute_context(ctx_tokens)
        episodes.append(EpisodicEntry(ctx_matrix.astype(DTYPE), target_idx))
    
    print(f"\n  Episodes: {len(episodes)}")
    
    # Consolidate
    dreaming.sleep(episodes, rem_cycles=3, verbose=False)
    
    # Check prototypes
    prototypes = []
    for level in dreaming.semantic_memory.levels:
        prototypes.extend(level)
    
    print(f"  Prototypes after consolidation: {len(prototypes)}")
    
    # Check if prototypes are target-coherent
    if prototypes:
        for i, proto in enumerate(prototypes):
            if hasattr(proto, 'target_distribution') and proto.target_distribution:
                targets = list(proto.target_distribution.keys())
                probs = list(proto.target_distribution.values())
                max_prob = max(probs) if probs else 0
                print(f"    Prototype {i}: targets={targets}, max_prob={max_prob:.2f}")
    
    # Test retrieval on novel context
    print("\n  Testing retrieval on novel context...")
    
    # Novel: "my cat said" (not in training, but similar to "the/a cat said")
    novel_tokens = [vocab.get(t, 0) for t in simple_tokenize("my cat said")]
    novel_ctx = model.compute_context(novel_tokens)
    
    results = dreaming.semantic_memory.retrieve(novel_ctx.astype(DTYPE), top_k=1)
    
    if results:
        proto, sim = results[0]
        if hasattr(proto, 'target_distribution') and proto.target_distribution:
            predicted = max(proto.target_distribution.keys(),
                           key=lambda k: proto.target_distribution[k])
            expected = vocab.get('meows', 0)
            correct = predicted == expected
            print(f"    'my cat said' → predicted target {predicted}, expected {expected}")
            print(f"    Correct: {correct}")
        else:
            print("    Prototype has no target distribution")
            correct = False
    else:
        print("    No retrieval results")
        correct = False
    
    passed = len(prototypes) >= 2 and correct
    
    if passed:
        print(f"\n  ✅ PASS: Dreaming consolidation enables generalization!")
    else:
        print(f"\n  ❌ FAIL: Dreaming consolidation insufficient")
    
    return {
        'test': 'learned_generalization_via_dreaming',
        'n_prototypes': len(prototypes),
        'correct_retrieval': correct,
        'passed': passed
    }


def run_all_tests() -> bool:
    """Run all tests."""
    print("\n" + "="*70)
    print("LEARNED GENERALIZATION TEST")
    print("="*70)
    print("""
THE REAL QUESTION:
    Can the system learn to generalize WITHOUT hand-crafted synonyms?
    
    NO cheating with: cat = feline (by construction)
    
    Instead: cat ≈ feline because they predict the same targets
""")
    
    results = []
    
    results.append(test_learned_generalization_via_predictiveness())
    results.append(test_learned_generalization_via_dreaming())
    
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
  🎉 TRULY THEORY-TRUE GENERALIZATION!
  
  The system learns semantic similarity from data alone:
  - No hand-crafted embeddings
  - Predictiveness tracking identifies co-predictive words
  - Dreaming consolidates similar-predicting contexts
  
  This is informationally parsimonious.
""")
    elif passed > 0:
        print("""
  ⚠️ PARTIAL: Some learning, but gaps remain.
  
  The theory is sound, but implementation needs work.
""")
    else:
        print("""
  ❌ FAIL: Cannot learn generalization from data alone.
  
  This is a significant limitation:
  - Either need more training data
  - Or need better consolidation algorithms
  - Or the random embeddings are too noisy
""")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
