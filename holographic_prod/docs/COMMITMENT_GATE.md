# Commitment Gate — Basal Ganglia Analog

## Overview

The `CommitmentGate` implements the brain's action selection mechanism, specifically
the basal ganglia's role in deciding **when** to commit to an action, not just
**what** action to take.

This is a critical distinction from transformers, which must produce output at every
forward pass (forced commitment via softmax).

---

## The Problem: Transformers Have No Commitment Mechanism

```python
# TRANSFORMER: Forced commitment every step
logits = model(context)
token = softmax(logits).argmax()  # MUST commit, no "hold" option
```

This is fundamentally different from how the brain works. The brain has a
**commitment gate** that decides WHEN to release an action.

---

## The Three-Pathway Model

The basal ganglia implements action selection via three pathways:

```
                    ┌─────────────────────────────────────┐
                    │           STRIATUM                   │
                    │   (competing action representations) │
                    └────────────────┬────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
    ┌────────────┐           ┌────────────┐           ┌────────────┐
    │  DIRECT    │           │  INDIRECT  │           │ HYPERDIRECT│
    │    GO      │           │   NO-GO    │           │    STOP    │
    │ entropy<φ⁻²│           │ entropy>φ⁻²│           │ entropy>1.0│
    └─────┬──────┘           └─────┬──────┘           └─────┬──────┘
          │                        │                        │
          ▼                        ▼                        ▼
       COMMIT                    HOLD                  EMERGENCY
       (token)              (evolve more)               BRAKE
```

| Pathway | Condition | Action | Brain Structure |
|---------|-----------|--------|-----------------|
| **Direct** (GO) | Entropy < φ⁻² | Release action | Striatum → GPi |
| **Indirect** (NO-GO) | φ⁻² < Entropy < 1.0 | Suppress, hold | Striatum → GPe → STN |
| **Hyperdirect** (STOP) | Entropy > 1.0 | Emergency brake | Cortex → STN → GPi |

---

## φ-Derived Thresholds (NOT Arbitrary)

| Threshold | Value | Brain Analog |
|-----------|-------|--------------|
| `entropy_threshold` | φ⁻² ≈ 0.382 | Dopamine release threshold |
| `hyperdirect_threshold` | 1.0 | Emergency brake activation |

The threshold φ⁻² is the **spectral gap** of the Grace operator — the same
constant that governs semantic contraction. This is NOT a coincidence.

---

## Mathematical Formulation

The gate decision is based on Shannon entropy of the score distribution:

```python
H(p) = -Σ pᵢ log(pᵢ)  # Entropy of softmax(scores)

if H < φ⁻²:
    pathway = "direct"   # GO: confident, commit
    commit = True
elif H < 1.0:
    pathway = "indirect" # NO-GO: uncertain, hold
    commit = False
else:
    pathway = "hyperdirect"  # STOP: very uncertain
    commit = False
```

---

## Neurological Validation

The commitment gate exhibits the same failure modes as human neurological disorders:

### Parkinson's Disease

**Clinical:** Patients know what they want to say but can't release it.
"I know what I want to say, but I can't get it out."

**Architecture:** `entropy_threshold → 0` means the gate almost never commits.

```python
parkinsonian_gate = CommitmentGate(entropy_threshold=0.001)
result = parkinsonian_gate.decide(clear_scores, candidates)
assert result.committed is False  # Gate stuck closed
```

### Tourette's Syndrome

**Clinical:** Actions released before semantic planning complete.
Inhibitory control fails, motor outputs "leak".

**Architecture:** `entropy_threshold → ∞` means the gate always commits.

```python
tourettes_gate = CommitmentGate(entropy_threshold=10.0, hyperdirect_threshold=10.0)
result = tourettes_gate.decide(ambiguous_scores, candidates)
assert result.committed is True  # Gate stuck open
```

### Stuttering

**Clinical:** Repetition of function words, blocks at sentence boundaries.
The sentence is ready but the release mechanism stutters.

**Architecture:** Normal threshold, but high entropy at boundaries (`. vs , vs "`).

```python
gate = CommitmentGate()
boundary_scores = np.array([2.0, 1.0])  # Near-tie at boundary
result = gate.decide(boundary_scores, [".", ","])
assert result.committed is False  # Hesitation at boundary
assert result.pathway == "indirect"
```

### Akinetic Mutism

**Clinical:** Near silence, but patients can still understand language.
Complete failure to initiate action.

**Architecture:** Both thresholds at 0 means ANY entropy triggers emergency stop.

```python
akinetic_gate = CommitmentGate(entropy_threshold=0.0, hyperdirect_threshold=0.0)
result = akinetic_gate.decide(any_scores, candidates)
assert result.pathway == "hyperdirect"  # Complete failure to initiate
```

---

## Integration with Grace Evolution

When the gate holds (NO-GO), the semantic state evolves further via Grace:

```python
decision = gate.decide(scores, candidates)

if decision.committed:
    token = decision.token  # Direct pathway: GO
else:
    # Indirect/Hyperdirect pathway: NO-GO
    # Evolve state via Grace, then retry
    for _ in range(grace_steps):
        state = grace_operator(state, basis)
    decision = gate.forced_commit(new_scores, candidates)
```

This is the brain-analog pattern:
1. **Hesitate** when uncertain
2. **Evolve** semantic representation
3. **Commit** when ready

---

## Dopamine Analog

The `entropy_threshold` acts like dopamine level:

| Dopamine Level | Threshold | Behavior |
|----------------|-----------|----------|
| Low (Parkinson's) | → 0 | Hard to commit, gate stuck closed |
| Normal | φ⁻² ≈ 0.382 | Balanced commitment |
| High (mania) | → ∞ | Easy to commit, gate stuck open |

This explains why dopamine agonists help Parkinson's patients speak more fluently
— they effectively raise the commitment threshold.

---

## Usage

```python
from holographic_prod.core.commitment_gate import CommitmentGate, GateDecision

# Create gate with default (theory-derived) thresholds
gate = CommitmentGate()

# Decide based on score distribution
scores = np.array([5.0, 1.0, 0.5])
candidates = ["the", "a", "one"]
decision = gate.decide(scores, candidates)

if decision.committed:
    print(f"Committed to: {decision.token}")
    print(f"Pathway: {decision.pathway}")  # "direct"
    print(f"Suppressed: {decision.suppressed}")  # ["a", "one"]
else:
    print(f"Held, pathway: {decision.pathway}")  # "indirect" or "hyperdirect"
    print(f"Entropy: {decision.entropy}")
```

---

## Key Files

| File | Purpose |
|------|---------|
| `core/commitment_gate.py` | Main implementation |
| `core/attractor_generation.py` | Integration with generation loop |
| `tests/test_commitment_gate.py` | Comprehensive tests including neurological modes |

---

## Summary

The commitment gate is not a hack — it's a brain-analog mechanism that:

1. **Replaces forced softmax commitment** with a three-pathway gating system
2. **Uses φ⁻² threshold** (the spectral gap) as the commitment criterion
3. **Exhibits real neurological failure modes** when parameters are pushed to extremes
4. **Integrates with Grace evolution** for the "hesitate → evolve → commit" pattern
5. **Acts as a dopamine analog** for modulating readiness to act

This validates that the architecture captures real brain dynamics, not just
statistical patterns.
