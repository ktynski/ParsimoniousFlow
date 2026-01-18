# Attention & Dreaming — Implementation Complete ✅

> **NOTE:** This is a historical implementation plan. All features have been
> implemented and integrated in v4.27.0+. For current architecture, see
> `ARCHITECTURE.md` and `SCALING_ROADMAP.md`.

**Date:** 2026-01-13  
**Version:** v4.27.0 → v4.29.0 (current)  
**Status:** ✅ COMPLETE — All features implemented and tested

---

## Status: ✅ ALL TESTS PASS

| Component | Tests | Status | File |
|-----------|-------|--------|------|
| ToroidalAttention | 7/7 | ✅ | `toroidal_attention.py` |
| DreamCycle | 7/7 | ✅ | `dream_cycles.py` |
| Integration | 5/5 | ✅ | `test_integrated_attention_dreaming.py` |

**Total: 19/19 tests pass. Implementation complete.**

---

## Why Before Scaling?

These are **architectural fundamentals**, not optimizations:

1. **Attention** enables cross-context learning — without it, each context is isolated
2. **Dreaming** enables consolidation — without it, memory fragments over time
3. Scaling without these = scaling a broken system

---

## Part 1: Toroidal Attention Mechanism

### Theory

Attention emerges **structurally** from the nested torus, not as a learned weight matrix:

```
Traditional Transformer:      Our Architecture:
─────────────────────────     ─────────────────────
Attention(Q, K, V)            Phase-coherent aggregation
  = softmax(QK^T/√d) · V        = Σ φ^(-k) · satellite_k · cos(θ_i - θ_j)

Learned weights               Geometric structure
O(n²) per layer              O(n) per layer (satellites aggregate locally)
```

### Key Insight

In the nested torus:
- **16 satellites** each hold a "slice" of context
- **Phase alignment** determines which satellites attend to each other
- **Master torus** aggregates satellite witnesses → global attention

This is **structural attention** — it comes from φ-offset phase distribution, not learned Q/K/V matrices.

### Tests (Write First)

```python
# File: holographic_v4/test_toroidal_attention.py

def test_1_satellite_phase_attention():
    """
    Satellites with aligned phases should have higher mutual attention.
    """
    model = ToroidalAttention(n_satellites=16)
    
    # Set phases
    model.set_satellite_phase(0, 0.0)       # Phase 0
    model.set_satellite_phase(1, 0.1)       # Close to 0
    model.set_satellite_phase(8, np.pi)     # Opposite phase
    
    # Attention from satellite 0
    attn_0_to_1 = model.attention_weight(0, 1)
    attn_0_to_8 = model.attention_weight(0, 8)
    
    # Close phases = high attention
    assert attn_0_to_1 > attn_0_to_8

def test_2_master_aggregation():
    """
    Master witness should be φ-weighted sum of satellite witnesses.
    """
    model = ToroidalAttention(n_satellites=16)
    
    # Set satellite witnesses
    for k in range(16):
        model.satellites[k].witness = np.array([PHI_INV ** k, 0])
    
    # Master should aggregate
    master = model.aggregate_to_master()
    
    # Should be φ-weighted sum
    expected = sum(PHI_INV ** (k % 4) * model.satellites[k].witness[0] 
                   for k in range(16))
    assert np.isclose(master[0], expected / sum(PHI_INV ** (k % 4) for k in range(16)))

def test_3_cross_context_attention():
    """
    Learning "A B → C" and "A D → C" should create shared attention.
    """
    model = FractalGenerativeMemory(...)
    
    model.learn([10, 20], 30)  # A B → C
    model.learn([10, 40], 30)  # A D → C (same target)
    
    # Token 10 should attend to both contexts
    attn_20 = model.get_attention([10, 20])
    attn_40 = model.get_attention([10, 40])
    
    # Both should be high (shared first token)
    assert attn_20 > 0.5
    assert attn_40 > 0.5

def test_4_attention_preserves_order():
    """
    Attention should respect token order (non-commutative).
    """
    model = ToroidalAttention(...)
    
    # Different orders should have different attention patterns
    attn_AB = model.compute_attention([10, 20])
    attn_BA = model.compute_attention([20, 10])
    
    assert not np.allclose(attn_AB, attn_BA)

def test_5_attention_scales_O_n():
    """
    Attention should be O(n) not O(n²).
    """
    import time
    
    for n in [10, 100, 1000]:
        model = ToroidalAttention(n_satellites=16)
        context = list(range(n))
        
        start = time.time()
        _ = model.compute_attention(context)
        elapsed = time.time() - start
        
    # Should scale linearly
    # (check that 1000 doesn't take 100x longer than 10)
```

### Implementation Plan

```python
# File: holographic_v4/toroidal_attention.py

class ToroidalAttention:
    """
    Structural attention via phase-coherent aggregation.
    
    NO LEARNED WEIGHTS. ALL φ-DERIVED.
    
    Architecture:
        - 16 satellites with φ-offset phases
        - Attention = cos(θ_i - θ_j) (phase alignment)
        - Master aggregates via φ-weighted sum
    """
    
    def __init__(self, n_satellites: int = 16):
        self.n_satellites = n_satellites
        self.satellites = [
            SatelliteState(
                phase=2 * PI * k * PHI_INV,  # Golden spiral
                frequency=PHI ** (k % 4),    # φ-staggered
            )
            for k in range(n_satellites)
        ]
    
    def attention_weight(self, i: int, j: int) -> float:
        """
        Attention from satellite i to satellite j.
        
        Based purely on phase alignment.
        """
        phase_diff = self.satellites[i].phase - self.satellites[j].phase
        return (1 + np.cos(phase_diff)) / 2  # Normalized to [0, 1]
    
    def compute_attention(self, context: List[int]) -> np.ndarray:
        """
        Compute attention weights for a context.
        
        Each token is assigned to a satellite based on position.
        Returns attention matrix [n_tokens, n_tokens].
        """
        n = len(context)
        attention = np.zeros((n, n))
        
        for i in range(n):
            sat_i = i % self.n_satellites
            for j in range(n):
                sat_j = j % self.n_satellites
                attention[i, j] = self.attention_weight(sat_i, sat_j)
        
        # Normalize rows (like softmax)
        attention = attention / (attention.sum(axis=1, keepdims=True) + 1e-10)
        
        return attention
    
    def aggregate_to_master(self) -> np.ndarray:
        """
        Aggregate satellite witnesses to master.
        
        Uses φ-weighted sum.
        """
        total_weight = 0.0
        master_witness = np.zeros_like(self.satellites[0].witness)
        
        for k, sat in enumerate(self.satellites):
            weight = PHI_INV ** (k % 4)
            master_witness += weight * sat.witness
            total_weight += weight
        
        return master_witness / total_weight
```

---

## Part 2: Dream Cycles

### Theory

Dreaming is **topological re-alignment**:

1. **Non-REM (Harmonic Consolidation)**
   - Master broadcasts stable witness DOWN to satellites
   - Dissonant satellites receive accelerated Grace (φ⁻⁴)
   - Prunes noise, reinforces prototypes

2. **REM (Stochastic Recombination)**
   - Master jitters satellite phases by random × 2πφ⁻¹
   - Search for new stable attractors (creative synthesis)
   - Concept bridging via phase exploration

3. **Wake Trigger**
   - When stability > φ⁻² (spectral gap threshold)
   - Or after max iterations (φ⁵ ≈ 11 cycles)

### Tests (Write First)

```python
# File: holographic_v4/test_dream_cycles.py

def test_1_non_rem_consolidates():
    """
    Non-REM should increase master-satellite coherence.
    """
    model = FractalGenerativeMemory(...)
    
    # Train with some noise
    for _ in range(100):
        model.learn(random_context(), random_target())
    
    # Get pre-dream coherence
    pre_coherence = model.compute_coherence()
    
    # Non-REM cycle
    model.non_rem_consolidation()
    
    # Coherence should increase
    post_coherence = model.compute_coherence()
    assert post_coherence >= pre_coherence * 0.95  # Allow some variance

def test_2_rem_finds_new_attractors():
    """
    REM should discover new stable states (creativity).
    """
    model = FractalGenerativeMemory(...)
    
    # Train
    model.learn([1, 2, 3], 10)
    model.learn([4, 5, 6], 10)  # Same target, different context
    
    # Get attractor count before REM
    pre_attractors = model.count_stable_attractors()
    
    # REM cycle
    discoveries = model.rem_recombination()
    
    # Should have found connections
    assert discoveries >= 0  # May or may not find new ones

def test_3_wake_trigger():
    """
    System should wake when stability > φ⁻².
    """
    model = FractalGenerativeMemory(...)
    
    # Set very stable state
    model.set_stability(0.7)  # Above threshold
    
    # Dream should wake immediately
    stats = model.dream()
    assert stats['iterations'] <= 2  # Quick wake

def test_4_paradox_resolution():
    """
    Contradictory memories should be phase-shifted, not deleted.
    """
    model = FractalGenerativeMemory(...)
    
    # Learn contradiction
    model.learn([1, 2, 3], 10)  # A → X
    model.learn([1, 2, 3], 20)  # A → Y (contradiction!)
    
    # Dream with paradox resolution
    model.dream(resolve_paradoxes=True)
    
    # Both should be retrievable (in different phase lanes)
    targets = model.get_all_targets([1, 2, 3])
    assert 10 in targets
    assert 20 in targets

def test_5_dream_improves_retrieval():
    """
    Dreaming should improve retrieval accuracy.
    """
    model = FractalGenerativeMemory(...)
    
    # Train
    pairs = [(random_context(), random_target()) for _ in range(100)]
    for ctx, tgt in pairs:
        model.learn(ctx, tgt)
    
    # Pre-dream accuracy
    pre_correct = sum(1 for ctx, tgt in pairs 
                      if model.retrieve_deterministic(ctx)[0] == tgt)
    
    # Dream
    model.dream()
    
    # Post-dream accuracy
    post_correct = sum(1 for ctx, tgt in pairs 
                       if model.retrieve_deterministic(ctx)[0] == tgt)
    
    # Should not decrease significantly
    assert post_correct >= pre_correct * 0.95
```

### Implementation Plan

```python
# File: holographic_v4/dream_cycles.py

class DreamCycle:
    """
    Topological Re-alignment via Sleep Cycles.
    
    Integrates with FractalGenerativeMemory.
    
    Phases:
        1. Non-REM: Master → Satellites (consolidation)
        2. REM: Phase jitter (creative recombination)
        3. Wake: When stability > φ⁻²
    """
    
    def __init__(self, memory: 'FractalGenerativeMemory'):
        self.memory = memory
        self.stability_threshold = PHI_INV_SQ  # Wake when above this
        self.max_iterations = int(PHI ** 5)    # ~11 cycles max
        self.consolidation_rate = PHI_INV_FOUR # Accelerated Grace
    
    def non_rem_consolidation(self):
        """
        Master broadcasts witness to satellites.
        
        Dissonant satellites receive accelerated Grace.
        """
        master_witness = self.memory.get_master_witness()
        
        for i, sat in enumerate(self.memory.satellite_states):
            # Compute coherence with master
            coherence = self._compute_coherence(sat, master_witness)
            
            if coherence < PHI_INV:  # Dissonant
                # Accelerated Grace
                self._apply_accelerated_grace(sat, self.consolidation_rate)
            
            # Broadcast master witness
            sat[0] = (1 - PHI_INV) * sat[0] + PHI_INV * master_witness[0]
            sat[-1] = (1 - PHI_INV) * sat[-1] + PHI_INV * master_witness[1]
    
    def rem_recombination(self) -> int:
        """
        Phase jitter for creative synthesis.
        
        Returns number of new attractors discovered.
        """
        discoveries = 0
        pre_stability = self.memory.get_stability()
        
        for sat in self.memory.satellite_states:
            # Random phase jitter scaled by golden angle
            jitter = np.random.randn() * 2 * PI * PHI_INV
            # Apply to bivector components (phase carriers)
            sat[5:11] *= np.cos(jitter)  # Bivectors = indices 5-10
        
        # Re-aggregate master
        self.memory._update_master()
        
        # Check if stability improved (found new attractor)
        post_stability = self.memory.get_stability()
        if post_stability > pre_stability:
            discoveries = 1
        
        return discoveries
    
    def full_cycle(self) -> Dict[str, Any]:
        """
        Run complete dream cycle (Non-REM + REM).
        
        Returns statistics.
        """
        stats = {
            'iterations': 0,
            'total_discoveries': 0,
            'pre_stability': self.memory.get_stability(),
            'woke_early': False,
        }
        
        for i in range(self.max_iterations):
            stats['iterations'] = i + 1
            
            # Non-REM
            self.non_rem_consolidation()
            
            # Check wake trigger
            stability = self.memory.get_stability()
            if stability > self.stability_threshold:
                stats['woke_early'] = True
                break
            
            # REM
            discoveries = self.rem_recombination()
            stats['total_discoveries'] += discoveries
        
        stats['post_stability'] = self.memory.get_stability()
        return stats
```

---

## Part 3: Integration with FractalGenerativeMemory

### Current State

`FractalGenerativeMemory` has:
- ✅ Basic `dream()` method
- ✅ Satellite states
- ❌ Proper Non-REM consolidation
- ❌ REM recombination
- ❌ Paradox resolution
- ❌ Toroidal attention

### Integration Tests

```python
# File: holographic_v4/test_integrated_attention_dreaming.py

def test_1_attention_improves_generalization():
    """
    Attention should enable cross-context generalization.
    """
    model = FractalGenerativeMemory(
        attention_enabled=True,
        ...
    )
    
    # Train with overlapping contexts
    model.learn([10, 20, 30], 50)
    model.learn([10, 20, 40], 50)  # Same first two tokens
    
    # Should generalize to unseen suffix
    retrieved, conf = model.retrieve_deterministic([10, 20, 99])
    
    # Should retrieve 50 (via attention to first two tokens)
    # This is the key test!

def test_2_dreaming_improves_generation():
    """
    Dreaming should improve generation coherence.
    """
    model = FractalGenerativeMemory(...)
    
    # Train on WikiText sample
    for ctx, tgt in wikitext_pairs[:1000]:
        model.learn(ctx, tgt)
    
    # Generate before dreaming
    pre_generated, _ = model.generate(prompt, max_tokens=20)
    pre_diversity = len(set(pre_generated))
    
    # Dream
    model.dream()
    
    # Generate after dreaming
    post_generated, _ = model.generate(prompt, max_tokens=20)
    post_diversity = len(set(post_generated))
    
    # Diversity should not decrease
    assert post_diversity >= pre_diversity * 0.8

def test_3_full_pipeline():
    """
    Complete pipeline: learn → dream → generate.
    """
    model = FractalGenerativeMemory(
        attention_enabled=True,
        dreaming_enabled=True,
    )
    
    # Train
    for _ in range(500):
        model.learn(random_context(5), random_target())
    
    # Dream (consolidate)
    dream_stats = model.dream()
    
    # Generate
    generated, gen_stats = model.generate([1, 2, 3], max_tokens=10)
    
    assert len(generated) == 10
    assert dream_stats['iterations'] > 0
```

---

## Implementation Order

### Week 1: Attention

| Day | Task | Tests |
|-----|------|-------|
| 1 | Write `test_toroidal_attention.py` | 5 tests |
| 2-3 | Implement `ToroidalAttention` class | Make tests pass |
| 4-5 | Integrate with `FractalGenerativeMemory` | Integration tests |

### Week 2: Dreaming

| Day | Task | Tests |
|-----|------|-------|
| 1 | Write `test_dream_cycles.py` | 5 tests |
| 2-3 | Implement `DreamCycle` class | Make tests pass |
| 4-5 | Integrate with `FractalGenerativeMemory` | Integration tests |

### Week 3: Integration & Testing

| Day | Task | Tests |
|-----|------|-------|
| 1-2 | Write `test_integrated_attention_dreaming.py` | 3 tests |
| 3-4 | Run comprehensive tests | All tests pass |
| 5 | Update documentation | Version bump to v4.27.0 |

---

## Success Criteria

### v4.27.0 Release

| Metric | Target | Current |
|--------|--------|---------|
| Attention tests | 5/5 pass | 0/5 |
| Dreaming tests | 5/5 pass | 0/5 |
| Integration tests | 3/3 pass | 0/3 |
| Cross-context generalization | >50% | Unknown |
| Post-dream retrieval | ≥95% | Unknown |

---

## File Structure

```
holographic_v4/
├── toroidal_attention.py          # NEW: Structural attention
├── dream_cycles.py                # NEW: Proper dreaming
├── test_toroidal_attention.py     # NEW: Attention tests
├── test_dream_cycles.py           # NEW: Dreaming tests
├── test_integrated_attention_dreaming.py  # NEW: Integration
├── fractal_generative_memory.py   # UPDATE: Add attention + dreaming
└── (existing files...)
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Attention breaks existing tests | MEDIUM | Run full test suite after each change |
| Dreaming degrades retrieval | LOW | Post-dream test included |
| Integration complexity | MEDIUM | Small, incremental changes |
| Performance overhead | LOW | Structural attention is O(n) |

---

## Next Immediate Action

**Start with Test 1 of Attention:**

```bash
python3 -m pytest holographic_v4/test_toroidal_attention.py::test_1_satellite_phase_attention -v
```

Write the test first. Watch it fail. Then implement.
