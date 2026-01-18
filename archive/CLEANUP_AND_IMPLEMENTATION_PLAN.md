# Cleanup and Implementation Plan v4.21.0 → v4.23.0

## Status: ✅ COMPLETED (v4.23.0)

This plan has been FULLY IMPLEMENTED. See `__init__.py` for version history.

## Philosophy

**NO FALLBACKS. NO LEGACY. NO BACKWARDS COMPATIBILITY.**

If something doesn't work, it fails explicitly. The system either works correctly per theory or it doesn't work at all.

---

## PART 1: LEGACY CRUFT REMOVAL

### 1.1 Remove WitnessIndex (DEAD CODE)

**Rationale**: WitnessIndex creates only ~4 buckets. It's useless.

**Files to modify:**
- `holographic_memory.py`: Remove WitnessIndex class entirely
- `holographic_memory.py`: Remove `use_vorticity_index` flag - VorticityWitnessIndex is the ONLY option
- `__init__.py`: Remove WitnessIndex from exports
- All test files: Remove WitnessIndex tests and legacy mode tests

**Test (before removal):**
```python
def test_witness_index_is_dead():
    """Prove WitnessIndex is useless before removing it."""
    index = WitnessIndex.create(basis)
    # Store 1000 random contexts
    # Assert < 10 unique buckets
    # Assert retrieval accuracy < random chance
```

### 1.2 Remove Alternative Index Flags

**Remove:**
- `use_gpu_witness` flag (keep GPU implementation if needed, just make it default)
- `use_torus_symmetry` flag (if theory-true, always use it; if not, remove it)
- `use_vorticity_index` flag (it's always True now)

**Simplify to:**
```python
def create(cls, basis: Array, xp: ArrayModule = np) -> 'HybridHolographicMemory':
    holographic = HolographicMemory.create(basis, xp)
    witness_index = VorticityWitnessIndex.create(basis, xp=xp)
    return cls(holographic, witness_index, basis, xp=xp)
```

### 1.3 Remove Fallback Semantics

**Files with fallback behavior to fix:**

1. `pipeline.py::compute_semantic_context` - has `fallback_to_full=True`
   - **FIX**: Remove fallback. If no semantic tokens, return zero matrix (empty context)
   
2. `predictiveness.py::compute_semantic_context` - has `fallback_to_full=True`
   - **FIX**: Remove entirely, use pipeline.py version only

3. `quotient.py` - has `PHI_INV_SQ` fallback for empty arrays
   - **FIX**: Return explicit error or zero

4. `holographic_memory.py` - "Fallback: WitnessIndex"
   - **FIX**: VorticityWitnessIndex is the ONLY index, not a fallback

5. `resonance.py` - "Global fallback (factorized prior)"
   - **KEEP**: This is theory-true (distributed prior provides global coverage)

### 1.4 Remove Arbitrary Constants

**Replace with φ-derived values:**

| File | Current | Replace With |
|------|---------|--------------|
| `holographic_modal.py:298` | `0.3-0.7` | `PHI_INV_CUBE` to `PHI_INV` |
| `holographic_modal.py:304` | `< 0.3` | `< PHI_INV_CUBE` |
| `holographic_modal.py:308` | `> 0.5` | `> PHI_INV_SQ` |
| `holographic_modal.py:420` | `> 0.7` | `> 1-PHI_INV_CUBE` |
| `holographic_modal.py:679` | `> 0.1` | `> PHI_INV_CUBE - PHI_INV_SQ` |
| `theory_of_mind.py:648` | `* 0.5` | `* PHI_INV_SQ` |

### 1.5 Remove BPE Tokenizer

**Rationale**: Word tokenizer is theory-true. BPE is not.

**Files:**
- `holographic_modal.py`: Remove `tokenizer_type: str = "word"` choice, always use word
- Remove all BPE-related code paths

---

## PART 2: PHASE 2 - SEMANTIC TOKEN FILTERING (TEST-DRIVEN)

### 2.1 Theory

The witness is sensitive to ALL tokens: `Tr(A·B·C...) = sum of semantic contributions`

**Problem**: Function words (the, a, is) contribute noise but no semantic content.

**Solution**: Compute separate indices:
1. **Episodic Index**: All tokens → exact recall (current behavior)
2. **Semantic Index**: Content tokens only → generalization

### 2.2 Tests First

```python
# test_semantic_filtering.py

def test_function_words_identified():
    """High-frequency, low-information tokens identified correctly."""
    tracker = InformationTracker()
    # After training, function words should have high frequency but low predictiveness
    assert tracker.is_function_word("the") == True
    assert tracker.is_function_word("cat") == False

def test_semantic_context_filters_function_words():
    """Semantic context excludes function words."""
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    semantic_tokens = filter_semantic_tokens(tokens, tracker)
    assert "the" not in semantic_tokens
    assert "cat" in semantic_tokens
    assert "sat" in semantic_tokens
    assert "mat" in semantic_tokens

def test_semantic_context_similarity_invariant():
    """Semantic contexts of paraphrases should be similar."""
    ctx1 = compute_semantic_context(["the", "cat", "sat", "on", "mat"], model)
    ctx2 = compute_semantic_context(["a", "cat", "sits", "on", "a", "mat"], model)
    # These should map to same/similar semantic region
    sim = witness_similarity(ctx1, ctx2, basis)
    assert sim > PHI_INV  # Should be similar

def test_semantic_retrieval_generalizes():
    """Semantic retrieval finds related contexts."""
    # Store: "the cat sat on the mat" → predicts "warmly"
    # Query: "a feline rested on a rug"
    # Should retrieve "warmly" (semantic match)
    pass

def test_no_fallback_on_empty_semantic():
    """Empty semantic context is explicit zero, not fallback."""
    tokens = ["the", "a", "is", "on"]  # All function words
    ctx = compute_semantic_context(tokens, model, tracker)
    # Should be zero matrix, NOT full context
    assert np.allclose(ctx, np.zeros((4, 4)))
```

### 2.3 Implementation

```python
# In pipeline.py or new semantic_filtering.py

def compute_semantic_context(
    tokens: List[int], 
    embeddings: Array, 
    tracker: 'InformationTracker',
    basis: Array,
    xp: ArrayModule = np
) -> Array:
    """
    Compute semantic context from content tokens only.
    
    NO FALLBACK. If no content tokens, returns zero matrix.
    """
    # Filter to content tokens only
    content_tokens = [t for t in tokens if not tracker.is_function_word(t)]
    
    if len(content_tokens) == 0:
        return xp.zeros((4, 4), dtype=DTYPE)
    
    return compute_context(content_tokens, embeddings, basis, xp)


class DualIndexMemory:
    """
    Two-level indexing for episodic + semantic retrieval.
    
    THEORY:
        - Episodic: Exact match (8D keys, all tokens)
        - Semantic: Generalization (iterated Grace, content tokens only)
    """
    def __init__(self, basis: Array, xp: ArrayModule = np):
        self.episodic = VorticityWitnessIndex.create(basis, xp=xp)
        self.semantic = SemanticIndex.create(basis, xp=xp)
        
    def store(self, context: Array, target: Array, target_idx: int, 
              semantic_context: Optional[Array] = None):
        """Store in both indices."""
        self.episodic.store(context, target, target_idx)
        if semantic_context is not None:
            self.semantic.store(semantic_context, target, target_idx)
    
    def retrieve(self, context: Array, semantic_context: Optional[Array] = None):
        """Try episodic first, then semantic."""
        # Episodic (exact)
        result, idx, conf = self.episodic.retrieve(context)
        if conf > PHI_INV:  # High confidence = exact match
            return result, idx, conf, "episodic"
        
        # Semantic (generalization)
        if semantic_context is not None:
            sem_result, sem_idx, sem_conf = self.semantic.retrieve(semantic_context)
            if sem_conf > PHI_INV_SQ:  # Moderate confidence = semantic match
                return sem_result, sem_idx, sem_conf, "semantic"
        
        # No match - explicit failure, NOT fallback
        return None, None, 0.0, "no_match"
```

---

## PART 3: PHASE 3 - TARGET-AWARE CLUSTERING (TEST-DRIVEN)

### 3.1 Theory

Current clustering: Group contexts by similarity
Problem: Similar contexts may predict different targets

**Target-Aware**: Group by (target, context_similarity)
- Prototypes are prediction-coherent
- Each prototype maps to a specific target distribution

### 3.2 Tests First

```python
# test_target_aware_clustering.py

def test_prototypes_predict_single_target():
    """Each prototype should primarily predict one target."""
    system = DreamingSystem(...)
    # Store many (context, target) pairs
    # Run consolidation
    for proto in system.semantic_memory.prototypes:
        # Target distribution should be concentrated
        max_prob = max(proto.target_distribution.values())
        assert max_prob > PHI_INV  # > 61.8% for primary target

def test_same_target_contexts_cluster():
    """Contexts predicting same target should cluster together."""
    # "cat on mat" → "meow"
    # "feline on rug" → "meow"  
    # These should be in SAME prototype
    pass

def test_different_target_contexts_separate():
    """Contexts predicting different targets should NOT cluster."""
    # "cat on mat" → "meow"
    # "cat on mat" → "purr"
    # These should be in DIFFERENT prototypes (or same proto with mixed distribution)
    pass

def test_no_degenerate_prototypes():
    """No prototype should predict all targets equally."""
    for proto in prototypes:
        entropy = -sum(p * log(p) for p in proto.target_distribution.values())
        max_entropy = log(vocab_size)
        # Entropy should be much less than max
        assert entropy < max_entropy * PHI_INV_SQ
```

### 3.3 Implementation

```python
# In dreaming.py

def target_aware_consolidation(episodes: List[EpisodicEntry]) -> List[SemanticPrototype]:
    """
    Consolidate episodes into prediction-coherent prototypes.
    
    ALGORITHM:
        1. Group episodes by target token
        2. Within each target group, cluster by context similarity
        3. Merge clusters across targets only if contexts are VERY similar
    """
    # Step 1: Group by target
    by_target: Dict[int, List[EpisodicEntry]] = defaultdict(list)
    for ep in episodes:
        by_target[ep.target_token].append(ep)
    
    prototypes = []
    
    # Step 2: Cluster within target groups
    for target, target_episodes in by_target.items():
        if len(target_episodes) < 2:
            # Single episode → single prototype
            proto = SemanticPrototype(
                target_episodes[0].context_matrix,
                target_distribution={target: 1.0},
                n_examples=1
            )
            prototypes.append(proto)
            continue
        
        # Cluster similar contexts within this target group
        clusters = cluster_by_combined_similarity(
            [ep.context_matrix for ep in target_episodes],
            threshold=1 - PHI_INV_CUBE  # Conservative threshold
        )
        
        for cluster_indices in clusters:
            centroid = compute_centroid([target_episodes[i].context_matrix for i in cluster_indices])
            proto = SemanticPrototype(
                centroid,
                target_distribution={target: 1.0},
                n_examples=len(cluster_indices)
            )
            prototypes.append(proto)
    
    # Step 3: Optional cross-target merge for very similar contexts
    # (This creates prototypes with mixed target distributions)
    # Only merge if similarity > 1 - PHI_INV_CUBE AND targets are related
    
    return prototypes
```

---

## PART 4: EXECUTION ORDER

### Phase A: Cleanup (This Session)

1. [ ] Write tests proving WitnessIndex is dead
2. [ ] Remove WitnessIndex class and all references
3. [ ] Remove index selection flags (use_gpu_witness, use_torus_symmetry, use_vorticity_index)
4. [ ] Remove fallback_to_full from compute_semantic_context
5. [ ] Replace arbitrary constants with φ-derived values
6. [ ] Remove BPE tokenizer code
7. [ ] Run full test suite, ensure no regressions

### Phase B: Semantic Filtering (After Cleanup)

1. [ ] Write tests for semantic token filtering
2. [ ] Implement InformationTracker.is_function_word()
3. [ ] Implement filter_semantic_tokens()
4. [ ] Implement DualIndexMemory
5. [ ] Integrate with pipeline
6. [ ] Run generalization tests

### Phase C: Target-Aware Clustering (After Phase B)

1. [ ] Write tests for target-aware clustering
2. [ ] Implement target_aware_consolidation()
3. [ ] Update DreamingSystem to use it
4. [ ] Run full system tests

### Phase D: Validation Run

1. [ ] Run training with all fixes
2. [ ] Measure exact accuracy (target: >90%)
3. [ ] Measure generalization accuracy (target: >50%)
4. [ ] Document results

---

## SUCCESS CRITERIA

| Metric | Current | Target |
|--------|---------|--------|
| Exact retrieval accuracy | ~100% | >95% |
| Generalization accuracy | ~1% | >50% |
| Prototype count | ~16 | >100 |
| No fallback code | ❌ | ✅ |
| All constants φ-derived | ❌ | ✅ |
| No legacy code | ❌ | ✅ |

---

## VERSION: 4.21.0

After completion:
- Zero legacy code
- Zero fallbacks
- All φ-derived constants
- Dual-index (episodic + semantic)
- Target-aware prototypes
- Transformer-killer ready
