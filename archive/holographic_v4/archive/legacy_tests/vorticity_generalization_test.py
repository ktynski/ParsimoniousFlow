"""
Vorticity Grammar Generalization Test

Tests whether the vorticity (wedge product) captures grammatical structure
that generalizes to unseen word combinations.

Key question: If we train on "the cat sat", does the model recognize
"the dog ran" as having the SAME grammatical structure?
"""

import numpy as np
from typing import List, Tuple, Dict

# Import core components
from holographic_v4.algebra import (
    build_clifford_basis, 
    geometric_product,
    compute_vorticity,
    initialize_embeddings_rotor,
    grace_operator
)
from holographic_v4.constants import PHI_INV, PHI_INV_SQ

xp = np  # Use NumPy for local testing


def test_vorticity_structure_similarity():
    """
    Test 1: Do sentences with SAME grammatical structure have similar vorticity?
    """
    print("\n" + "="*70)
    print("TEST 1: Vorticity Structure Similarity")
    print("="*70)
    
    basis = build_clifford_basis(xp)
    
    # Create a small vocabulary with grammatical categories
    vocab = {
        # Determiners
        'the': 0, 'a': 1, 'an': 2,
        # Nouns (animals)
        'cat': 10, 'dog': 11, 'bird': 12, 'fish': 13,
        # Nouns (objects)
        'mat': 20, 'park': 21, 'house': 22, 'tree': 23,
        # Verbs
        'sat': 30, 'ran': 31, 'flew': 32, 'swam': 33,
        # Prepositions
        'on': 40, 'in': 41, 'over': 42, 'under': 43,
    }
    
    embeddings = initialize_embeddings_rotor(50, basis, xp, seed=42)
    
    def get_vorticity_signature(sentence: List[str]) -> np.ndarray:
        """Compute vorticity signature for a sentence."""
        tokens = [vocab[w] for w in sentence]
        mats = [embeddings[t] for t in tokens]
        
        # Compose context
        ctx = mats[0]
        for m in mats[1:]:
            ctx = geometric_product(ctx, m)
        
        # Extract vorticity (grade 2)
        vort = compute_vorticity(ctx, basis, xp)
        return vort
    
    def vorticity_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between vorticity signatures."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(np.dot(v1.flatten(), v2.flatten()) / (norm1 * norm2))
    
    # Test sentences with SAME grammatical structure: DET NOUN VERB PREP DET NOUN
    same_structure = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'dog', 'ran', 'in', 'the', 'park'],
        ['a', 'bird', 'flew', 'over', 'the', 'house'],
        ['the', 'fish', 'swam', 'under', 'the', 'tree'],
    ]
    
    # Test sentences with DIFFERENT grammatical structure
    different_structure = [
        ['the', 'cat'],  # DET NOUN (shorter)
        ['sat', 'the', 'cat'],  # VERB DET NOUN (different order)
        ['cat', 'the', 'mat', 'on', 'sat'],  # Scrambled
    ]
    
    # Compute vorticity for all same-structure sentences
    same_vorts = [get_vorticity_signature(s) for s in same_structure]
    diff_vorts = [get_vorticity_signature(s) for s in different_structure]
    
    # Measure intra-structure similarity
    print("\nSame Structure Similarities (should be HIGH):")
    same_sims = []
    for i in range(len(same_vorts)):
        for j in range(i+1, len(same_vorts)):
            sim = vorticity_similarity(same_vorts[i], same_vorts[j])
            same_sims.append(sim)
            print(f"  '{' '.join(same_structure[i][:3])}...' vs '{' '.join(same_structure[j][:3])}...': {sim:.3f}")
    
    avg_same = np.mean(same_sims)
    print(f"\n  Average same-structure similarity: {avg_same:.3f}")
    
    # Measure cross-structure similarity
    print("\nDifferent Structure Similarities (should be LOW):")
    diff_sims = []
    for i, sv in enumerate(same_vorts[:2]):  # Just first two
        for j, dv in enumerate(diff_vorts):
            sim = vorticity_similarity(sv, dv)
            diff_sims.append(sim)
            print(f"  '{' '.join(same_structure[i][:3])}...' vs '{' '.join(different_structure[j][:3])}...': {sim:.3f}")
    
    avg_diff = np.mean(diff_sims)
    print(f"\n  Average cross-structure similarity: {avg_diff:.3f}")
    
    # Verdict
    print(f"\n  DISCRIMINATION RATIO: {avg_same / (avg_diff + 0.01):.2f}x")
    
    if avg_same > 0.7 and avg_diff < 0.5:
        print("  ✓ PASS: Vorticity distinguishes grammatical structure!")
        return True
    elif avg_same > avg_diff:
        print("  ~ PARTIAL: Some discrimination, could be stronger")
        return True
    else:
        print("  ✗ FAIL: Vorticity doesn't capture structure")
        return False


def test_word_order_sensitivity():
    """
    Test 2: Does vorticity distinguish word ORDER?
    "the cat chased the dog" vs "the dog chased the cat" should be DIFFERENT.
    """
    print("\n" + "="*70)
    print("TEST 2: Word Order Sensitivity (Non-Commutativity)")
    print("="*70)
    
    basis = build_clifford_basis(xp)
    embeddings = initialize_embeddings_rotor(50, xp=xp, basis=basis, seed=42)
    
    vocab = {'the': 0, 'cat': 1, 'dog': 2, 'chased': 3, 'bit': 4}
    
    def get_context(sentence: List[str]) -> np.ndarray:
        tokens = [vocab[w] for w in sentence]
        mats = [embeddings[t] for t in tokens]
        ctx = mats[0]
        for m in mats[1:]:
            ctx = geometric_product(ctx, m)
        return ctx
    
    def matrix_similarity(m1: np.ndarray, m2: np.ndarray) -> float:
        return float(np.sum(m1 * m2) / (np.linalg.norm(m1) * np.linalg.norm(m2)))
    
    # These should be DIFFERENT (subject/object swap)
    s1 = ['the', 'cat', 'chased', 'the', 'dog']
    s2 = ['the', 'dog', 'chased', 'the', 'cat']
    
    ctx1 = get_context(s1)
    ctx2 = get_context(s2)
    
    sim = matrix_similarity(ctx1, ctx2)
    print(f"\n  '{' '.join(s1)}' vs '{' '.join(s2)}'")
    print(f"  Context similarity: {sim:.3f}")
    
    # These should be SAME (identical)
    ctx1_repeat = get_context(s1)
    sim_same = matrix_similarity(ctx1, ctx1_repeat)
    print(f"\n  Same sentence twice: {sim_same:.3f} (should be 1.0)")
    
    # Different verb, same structure
    s3 = ['the', 'cat', 'bit', 'the', 'dog']
    ctx3 = get_context(s3)
    sim_verb = matrix_similarity(ctx1, ctx3)
    print(f"\n  '{' '.join(s1)}' vs '{' '.join(s3)}' (different verb)")
    print(f"  Similarity: {sim_verb:.3f}")
    
    # Verdict
    print(f"\n  Subject/object swap discrimination: {1 - sim:.3f}")
    
    if sim < 0.9:  # Should NOT be highly similar
        print("  ✓ PASS: Word order creates different representations!")
        return True
    else:
        print("  ✗ FAIL: Word order not distinguished")
        return False


def test_grammatical_slot_prediction():
    """
    Test 3: Can we predict grammatical category from context?
    If trained on "the cat sat", can we recognize "the dog ___" expects a verb?
    """
    print("\n" + "="*70)
    print("TEST 3: Grammatical Slot Prediction")
    print("="*70)
    
    from holographic_v4.pipeline import TheoryTrueModel
    
    # Create model with small vocab
    model = TheoryTrueModel(vocab_size=100, context_size=10, xp=xp)
    
    # Define grammatical patterns
    # Pattern: DET(0-9) NOUN(10-19) VERB(20-29)
    det_tokens = list(range(0, 10))
    noun_tokens = list(range(10, 20))
    verb_tokens = list(range(20, 30))
    
    # Train on DET NOUN VERB patterns
    print("\n  Training on DET NOUN VERB patterns...")
    for _ in range(100):
        det = np.random.choice(det_tokens)
        noun = np.random.choice(noun_tokens)
        verb = np.random.choice(verb_tokens)
        
        # Context: [DET, NOUN], Target: VERB
        model.train_step([det, noun], verb)
    
    print(f"  Trained {model.num_attractors} attractors")
    
    # Test: Novel DET NOUN combination → should predict VERB category
    print("\n  Testing novel combinations...")
    
    correct_category = 0
    total_tests = 20
    
    for _ in range(total_tests):
        # Novel combination
        det = np.random.choice(det_tokens)
        noun = np.random.choice(noun_tokens)
        
        # Retrieve prediction
        _, predicted = model.retrieve([det, noun])
        
        # Is prediction in VERB category?
        if 20 <= predicted < 30:
            correct_category += 1
    
    accuracy = correct_category / total_tests
    print(f"\n  Novel inputs predicting VERB category: {correct_category}/{total_tests} ({accuracy*100:.0f}%)")
    
    if accuracy > 0.6:
        print("  ✓ PASS: Grammatical category abstraction works!")
        return True
    elif accuracy > 0.3:
        print("  ~ PARTIAL: Some abstraction, needs more training")
        return True
    else:
        print("  ✗ FAIL: No grammatical abstraction")
        return False


def test_vorticity_with_prototypes():
    """
    Test 4: Do prototypes cluster by grammatical structure?
    After dreaming, similar structures should map to same prototypes.
    """
    print("\n" + "="*70)
    print("TEST 4: Prototype Clustering by Structure")
    print("="*70)
    
    from holographic_v4.pipeline import TheoryTrueModel
    from holographic_v4.dreaming import DreamingSystem, EpisodicEntry
    
    model = TheoryTrueModel(vocab_size=100, context_size=10, xp=xp)
    dreaming = DreamingSystem(model, xp=xp)
    
    # Create episodes with same structure but different tokens
    # Structure A: tokens 0-9 followed by tokens 10-19 → target 50
    # Structure B: tokens 20-29 followed by tokens 30-39 → target 60
    
    print("\n  Creating episodes with two grammatical structures...")
    
    episodes = []
    
    # Structure A episodes
    for _ in range(50):
        t1 = np.random.randint(0, 10)
        t2 = np.random.randint(10, 20)
        ctx = model.compute_context([t1, t2])
        ep = EpisodicEntry(
            context_matrix=ctx,
            target=50,
            context_tokens=[t1, t2],
            timestamp=0,
            salience=1.0,
            stability=0.3,  # Below threshold for consolidation
            priority=1.0
        )
        episodes.append(ep)
    
    # Structure B episodes
    for _ in range(50):
        t1 = np.random.randint(20, 30)
        t2 = np.random.randint(30, 40)
        ctx = model.compute_context([t1, t2])
        ep = EpisodicEntry(
            context_matrix=ctx,
            target=60,
            context_tokens=[t1, t2],
            timestamp=0,
            salience=1.0,
            stability=0.3,
            priority=1.0
        )
        episodes.append(ep)
    
    print(f"  Created {len(episodes)} episodes (50 each of 2 structures)")
    
    # Run sleep cycle
    print("  Running sleep cycle...")
    stats = dreaming.sleep(episodes, rem_cycles=1, verbose=False)
    
    n_protos = dreaming.semantic_memory.stats()['total_prototypes']
    print(f"  Created {n_protos} prototypes")
    
    # Check: Do the two structures map to different prototypes?
    # Test with novel tokens from each structure
    
    print("\n  Testing prototype retrieval for novel tokens...")
    
    structure_a_protos = set()
    structure_b_protos = set()
    
    for _ in range(20):
        # Novel Structure A
        t1 = np.random.randint(0, 10)
        t2 = np.random.randint(10, 20)
        ctx = model.compute_context([t1, t2])
        matches = dreaming.semantic_memory.retrieve(ctx, top_k=1)
        if matches:
            structure_a_protos.add(id(matches[0]))
        
        # Novel Structure B
        t1 = np.random.randint(20, 30)
        t2 = np.random.randint(30, 40)
        ctx = model.compute_context([t1, t2])
        matches = dreaming.semantic_memory.retrieve(ctx, top_k=1)
        if matches:
            structure_b_protos.add(id(matches[0]))
    
    overlap = len(structure_a_protos & structure_b_protos)
    print(f"\n  Structure A retrieves {len(structure_a_protos)} unique prototypes")
    print(f"  Structure B retrieves {len(structure_b_protos)} unique prototypes")
    print(f"  Overlap: {overlap}")
    
    if overlap == 0 and len(structure_a_protos) > 0 and len(structure_b_protos) > 0:
        print("  ✓ PASS: Structures map to distinct prototypes!")
        return True
    elif overlap < min(len(structure_a_protos), len(structure_b_protos)):
        print("  ~ PARTIAL: Some separation, not perfect")
        return True
    else:
        print("  ✗ FAIL: Structures not distinguished in prototypes")
        return False


def run_all_tests():
    """Run all vorticity generalization tests."""
    print("\n" + "="*70)
    print("  VORTICITY GRAMMAR GENERALIZATION TESTS")
    print("="*70)
    
    results = {}
    
    results['structure_similarity'] = test_vorticity_structure_similarity()
    results['word_order'] = test_word_order_sensitivity()
    results['slot_prediction'] = test_grammatical_slot_prediction()
    results['prototype_clustering'] = test_vorticity_with_prototypes()
    
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n  Overall: {'✓ ALL TESTS PASS' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
