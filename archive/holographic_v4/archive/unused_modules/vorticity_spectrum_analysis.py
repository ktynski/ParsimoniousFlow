"""
Vorticity Spectrum Analysis — Theory-True Brain-Like Approach
=============================================================

WHY THE FFT TEST FAILED:
------------------------
The original test computed FFT on vorticity MAGNITUDES. This is wrong because:
1. Magnitude is a SCALAR - loses directional information
2. Coherent text has structured DIRECTION, not higher magnitude
3. Random text can have high magnitude but chaotic direction (turbulence)

BRAIN-LIKE INSIGHT:
-------------------
The brain doesn't use FFT on firing rates. It uses:
1. PHASE COHERENCE - neurons that fire together (same phase) bind together
2. PREDICTABILITY - coherent sequences are predictable
3. DIRECTIONAL STABILITY - same "rotation direction" = same structure

THEORY-TRUE SOLUTION:
---------------------
Replace FFT with:
1. Phase Locking Value (PLV) - how aligned are vorticity directions?
2. Autocorrelation - does structure repeat?
3. Prediction error - can we predict next vorticity from previous?

These are what brains actually compute for binding and coherence detection.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_v4.algebra import (
    wedge_product, build_clifford_basis, decompose_to_coefficients,
    vorticity_similarity
)
from holographic_v4.constants import PHI, PHI_INV, PHI_INV_SQ, MATRIX_DIM

ArrayModule = type(np)
Array = np.ndarray


# =============================================================================
# BRAIN-LIKE METRIC 1: PHASE LOCKING VALUE (PLV)
# =============================================================================

def compute_plv(signatures: List[Array], xp: ArrayModule = np) -> float:
    """
    Phase Locking Value for vorticity signatures.
    
    BRAIN ANALOGY:
        PLV measures how synchronized neural oscillations are.
        High PLV = coherent binding (attention, working memory)
        Low PLV = no binding (noise, random activity)
    
    VORTICITY VERSION:
        PLV = |mean(exp(i*θ))| where θ is the "angle" between consecutive signatures.
        We approximate this using signature dot products (cosine of angle).
        
        High PLV = vorticity directions are STABLE (coherent text)
        Low PLV = vorticity directions FLUCTUATE (random/incoherent)
    
    Returns:
        PLV in [0, 1]. Higher = more coherent.
    """
    if len(signatures) < 2:
        return 0.0
    
    # Compute "phase differences" as cosine similarities
    phase_diffs = []
    for i in range(len(signatures) - 1):
        sim = vorticity_similarity(signatures[i], signatures[i+1], xp)
        # Map similarity [-1, 1] to angle [π, 0] then to unit circle
        # High similarity → small angle → phase locked
        phase_diffs.append(sim)
    
    # PLV = |mean(exp(i*θ))| ≈ |mean(cos(θ))| for small angles
    # Since we already have cos(θ) = sim, use magnitude of mean
    mean_sim = np.mean(phase_diffs)
    
    # PLV should be in [0, 1]
    # High positive mean = phase locked (coherent)
    # Near zero = random phase (incoherent)
    # Negative = anti-phase (structured but opposite)
    
    plv = abs(mean_sim)
    return float(plv)


# =============================================================================
# BRAIN-LIKE METRIC 2: DIRECTIONAL STABILITY
# =============================================================================

def compute_directional_stability(signatures: List[Array], xp: ArrayModule = np) -> float:
    """
    Measure how stable the vorticity direction is across the sequence.
    
    BRAIN ANALOGY:
        Stable direction = consistent attentional focus
        Unstable direction = mind wandering / distraction
    
    THEORY:
        Coherent narratives maintain a consistent "rotation axis"
        Random text has chaotic rotation (turbulence)
    
    Returns:
        Stability in [0, 1]. Higher = more coherent.
    """
    if len(signatures) < 3:
        return 1.0
    
    # Compute consecutive similarities
    sims = []
    for i in range(len(signatures) - 1):
        sim = vorticity_similarity(signatures[i], signatures[i+1], xp)
        sims.append(sim)
    
    # Stability = 1 - variance of similarities
    # Coherent text: sims are consistently high (low variance)
    # Random text: sims fluctuate wildly (high variance)
    
    variance = np.var(sims)
    # THEORY-TRUE: φ-power decay (NOT arbitrary 1/(1+4*var))
    stability = PHI_INV ** variance
    
    return float(stability)


# =============================================================================
# BRAIN-LIKE METRIC 3: PREDICTION ERROR (PREDICTIVE CODING)
# =============================================================================

def compute_prediction_error(signatures: List[Array], xp: ArrayModule = np) -> float:
    """
    Measure how predictable the vorticity sequence is.
    
    BRAIN ANALOGY:
        Predictive coding: brain constantly predicts next state
        Low prediction error = coherent, expected sequence
        High prediction error = surprising, novel, or random
    
    THEORY:
        For coherent text, vorticity evolves smoothly
        Simple linear prediction: sig[t+1] ≈ sig[t] + Δ where Δ is stable
    
    Returns:
        Predictability in [0, 1]. Higher = more coherent.
    """
    if len(signatures) < 3:
        return 1.0
    
    # Simple linear predictor: sig[t+1] ≈ 2*sig[t] - sig[t-1]
    # (Assumes constant velocity in signature space)
    
    errors = []
    for i in range(2, len(signatures)):
        # Linear prediction
        predicted = 2 * signatures[i-1] - signatures[i-2]
        actual = signatures[i]
        
        # Error as normalized distance
        error = xp.linalg.norm(actual - predicted) / (xp.linalg.norm(actual) + 1e-10)
        errors.append(float(error))
    
    # THEORY-TRUE: φ-power decay (NOT arbitrary 1/(1+x))
    mean_error = np.mean(errors)
    predictability = PHI_INV ** mean_error
    
    return float(predictability)


# =============================================================================
# BRAIN-LIKE METRIC 4: AUTOCORRELATION (MEMORY)
# =============================================================================

def compute_autocorrelation(signatures: List[Array], lag: int = 1, xp: ArrayModule = np) -> float:
    """
    Autocorrelation of vorticity signatures at given lag.
    
    BRAIN ANALOGY:
        Memory = correlation between past and present
        High autocorrelation = stable working memory
        Low autocorrelation = no memory / random
    
    THEORY:
        Coherent text has temporal structure (themes return)
        Random text has no temporal structure
    
    Returns:
        Autocorrelation in [-1, 1]. Higher = more memory.
    """
    if len(signatures) < lag + 1:
        return 0.0
    
    correlations = []
    for i in range(len(signatures) - lag):
        sim = vorticity_similarity(signatures[i], signatures[i + lag], xp)
        correlations.append(sim)
    
    return float(np.mean(correlations))


# =============================================================================
# COMBINED: BRAIN-LIKE COHERENCE SCORE
# =============================================================================

@dataclass
class VorticityCoherenceMetrics:
    """Brain-like coherence metrics for vorticity analysis."""
    plv: float                  # Phase Locking Value [0, 1]
    directional_stability: float  # Direction stability [0, 1]
    predictability: float       # Prediction accuracy [0, 1]
    autocorrelation_lag1: float # Memory at lag 1 [-1, 1]
    autocorrelation_lag3: float # Memory at lag 3 [-1, 1]
    overall_coherence: float    # Combined score [0, 1]


def compute_vorticity_coherence(
    signatures: List[Array],
    xp: ArrayModule = np
) -> VorticityCoherenceMetrics:
    """
    Compute brain-like coherence metrics for a sequence of vorticity signatures.
    
    This replaces the failed FFT approach with theory-true brain-like metrics:
    1. PLV - phase locking (binding)
    2. Directional stability - attentional focus
    3. Predictability - predictive coding
    4. Autocorrelation - working memory
    
    Args:
        signatures: List of [16] vorticity signatures
        xp: array module
        
    Returns:
        VorticityCoherenceMetrics with all metrics and combined score
    """
    plv = compute_plv(signatures, xp)
    stability = compute_directional_stability(signatures, xp)
    predictability = compute_prediction_error(signatures, xp)
    autocorr1 = compute_autocorrelation(signatures, lag=1, xp=xp)
    autocorr3 = compute_autocorrelation(signatures, lag=3, xp=xp)
    
    # Combined score using φ-derived weights (theory-true)
    # PLV and predictability most important (φ⁻¹)
    # Stability moderate (φ⁻²)
    # Autocorrelation supporting (φ⁻³)
    
    overall = (
        plv * PHI_INV +
        predictability * PHI_INV +
        stability * PHI_INV_SQ +
        max(0, autocorr1) * PHI_INV_SQ / 2 +  # Only positive contribution
        max(0, autocorr3) * PHI_INV_SQ / 2
    ) / (2 * PHI_INV + 2 * PHI_INV_SQ)  # Normalize
    
    return VorticityCoherenceMetrics(
        plv=plv,
        directional_stability=stability,
        predictability=predictability,
        autocorrelation_lag1=autocorr1,
        autocorrelation_lag3=autocorr3,
        overall_coherence=overall
    )


# =============================================================================
# TEST: VERIFY BRAIN-LIKE METRICS WORK
# =============================================================================

def test_brainlike_metrics():
    """Test that brain-like metrics discriminate coherent from random text."""
    print("\n" + "="*70)
    print("BRAIN-LIKE VORTICITY COHERENCE METRICS")
    print("="*70)
    
    from holographic_v4.pipeline import TheoryTrueModel
    
    class SimpleTokenizer:
        def __init__(self):
            self.word_to_id = {}
            self.id_to_word = {}
            self.next_id = 1
        def encode(self, text):
            tokens = []
            for word in text.lower().split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
                tokens.append(self.word_to_id[word])
            return tokens
    
    tokenizer = SimpleTokenizer()
    
    # Coherent text (clear narrative structure)
    coherent_texts = [
        "the young prince traveled across the vast kingdom seeking wisdom and adventure in distant lands",
        "she opened the old wooden door and stepped into the garden where flowers bloomed in every color",
        "the scientist carefully measured each ingredient before adding them to the bubbling solution",
    ]
    
    # Random/shuffled text (no narrative structure)
    import random
    random_texts = []
    for text in coherent_texts:
        words = text.split()
        shuffled = words.copy()
        random.shuffle(shuffled)
        random_texts.append(' '.join(shuffled))
    
    # Pre-encode
    for text in coherent_texts + random_texts:
        tokenizer.encode(text)
    
    model = TheoryTrueModel(vocab_size=tokenizer.next_id + 10)
    xp = model.xp
    basis = build_clifford_basis(xp)
    
    # Train briefly
    print("\n  Training model...")
    for text in coherent_texts * 5:
        tokens = tokenizer.encode(text)
        for i in range(1, len(tokens)):
            model.train_step(tokens[:i], tokens[i])
    
    def get_signatures(text):
        """Extract vorticity signatures for text."""
        tokens = tokenizer.encode(text)
        signatures = []
        for i in range(2, len(tokens) + 1):
            subset = tokens[:i]
            embeddings = xp.stack([model.embeddings[t] for t in subset])
            total_vort = xp.zeros((MATRIX_DIM, MATRIX_DIM))
            for j in range(len(embeddings) - 1):
                wedge = wedge_product(embeddings[j], embeddings[j+1], xp)
                total_vort = total_vort + wedge
            sig = decompose_to_coefficients(total_vort, basis, xp)
            signatures.append(sig)
        return signatures
    
    print("\n  Coherent texts:")
    coherent_metrics = []
    for i, text in enumerate(coherent_texts):
        sigs = get_signatures(text)
        metrics = compute_vorticity_coherence(sigs, xp)
        coherent_metrics.append(metrics)
        print(f"\n    Text {i+1}:")
        print(f"      PLV: {metrics.plv:.4f}")
        print(f"      Stability: {metrics.directional_stability:.4f}")
        print(f"      Predictability: {metrics.predictability:.4f}")
        print(f"      Autocorr(1): {metrics.autocorrelation_lag1:.4f}")
        print(f"      Autocorr(3): {metrics.autocorrelation_lag3:.4f}")
        print(f"      OVERALL: {metrics.overall_coherence:.4f}")
    
    print("\n  Random/shuffled texts:")
    random_metrics = []
    for i, text in enumerate(random_texts):
        sigs = get_signatures(text)
        metrics = compute_vorticity_coherence(sigs, xp)
        random_metrics.append(metrics)
        print(f"\n    Random {i+1}:")
        print(f"      PLV: {metrics.plv:.4f}")
        print(f"      Stability: {metrics.directional_stability:.4f}")
        print(f"      Predictability: {metrics.predictability:.4f}")
        print(f"      Autocorr(1): {metrics.autocorrelation_lag1:.4f}")
        print(f"      Autocorr(3): {metrics.autocorrelation_lag3:.4f}")
        print(f"      OVERALL: {metrics.overall_coherence:.4f}")
    
    # Compare averages
    avg_coherent = np.mean([m.overall_coherence for m in coherent_metrics])
    avg_random = np.mean([m.overall_coherence for m in random_metrics])
    
    avg_coherent_plv = np.mean([m.plv for m in coherent_metrics])
    avg_random_plv = np.mean([m.plv for m in random_metrics])
    
    avg_coherent_pred = np.mean([m.predictability for m in coherent_metrics])
    avg_random_pred = np.mean([m.predictability for m in random_metrics])
    
    print("\n" + "-"*60)
    print("  COMPARISON:")
    print("-"*60)
    print(f"\n  Average OVERALL coherence:")
    print(f"    Coherent: {avg_coherent:.4f}")
    print(f"    Random: {avg_random:.4f}")
    print(f"    Ratio: {avg_coherent / (avg_random + 1e-10):.2f}x")
    
    print(f"\n  Average PLV (phase locking):")
    print(f"    Coherent: {avg_coherent_plv:.4f}")
    print(f"    Random: {avg_random_plv:.4f}")
    
    print(f"\n  Average Predictability:")
    print(f"    Coherent: {avg_coherent_pred:.4f}")
    print(f"    Random: {avg_random_pred:.4f}")
    
    # Test: coherent should have HIGHER overall coherence
    passed = avg_coherent > avg_random
    
    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    if passed:
        print("\n  ✓ BRAIN-LIKE METRICS SUCCESSFULLY DISCRIMINATE")
        print("    Phase locking and predictability capture coherence")
        print("    that FFT on magnitudes missed.")
    else:
        print("\n  ✗ Metrics need refinement")
    
    print("\n" + "-"*60)
    print("  WHY THIS WORKS (Brain Analogy):")
    print("-"*60)
    print("""
    1. PLV (Phase Locking Value):
       - Brain: Synchronized neural oscillations = binding/attention
       - Vorticity: Consistent direction = coherent structure
       
    2. Predictability:
       - Brain: Predictive coding minimizes surprise
       - Vorticity: Coherent text has predictable evolution
       
    3. Directional Stability:
       - Brain: Stable attentional focus = working memory
       - Vorticity: Stable rotation axis = thematic consistency
       
    4. Autocorrelation:
       - Brain: Memory = correlation with past states
       - Vorticity: Themes return = positive autocorrelation
    
    The FFT approach failed because it measured MAGNITUDE spectrum,
    but coherence is in the DIRECTION (phase) of vorticity, not amplitude.
    
    This is exactly how brains work: binding is phase-based, not amplitude-based.
    """)
    
    return passed, coherent_metrics, random_metrics


def test_comparison_with_fft():
    """
    Direct comparison showing why brain-like metrics work better than FFT.
    """
    print("\n" + "="*70)
    print("COMPARISON: FFT vs BRAIN-LIKE METRICS")
    print("="*70)
    
    from holographic_v4.pipeline import TheoryTrueModel
    
    class SimpleTokenizer:
        def __init__(self):
            self.word_to_id = {}
            self.next_id = 1
        def encode(self, text):
            tokens = []
            for word in text.lower().split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.next_id
                    self.next_id += 1
                tokens.append(self.word_to_id[word])
            return tokens
    
    tokenizer = SimpleTokenizer()
    
    coherent = "the cat sat on the mat and the dog ran in the park while the bird flew over the tree"
    
    import random
    words = coherent.split()
    shuffled = words.copy()
    random.shuffle(shuffled)
    random_text = ' '.join(shuffled)
    
    for text in [coherent, random_text]:
        tokenizer.encode(text)
    
    model = TheoryTrueModel(vocab_size=tokenizer.next_id + 10)
    xp = model.xp
    basis = build_clifford_basis(xp)
    
    # Train
    for text in [coherent] * 10:
        tokens = tokenizer.encode(text)
        for i in range(1, len(tokens)):
            model.train_step(tokens[:i], tokens[i])
    
    def get_magnitudes_and_sigs(text):
        tokens = tokenizer.encode(text)
        magnitudes = []
        signatures = []
        for i in range(2, len(tokens) + 1):
            subset = tokens[:i]
            embeddings = xp.stack([model.embeddings[t] for t in subset])
            total_vort = xp.zeros((MATRIX_DIM, MATRIX_DIM))
            for j in range(len(embeddings) - 1):
                wedge = wedge_product(embeddings[j], embeddings[j+1], xp)
                total_vort = total_vort + wedge
            magnitudes.append(float(xp.linalg.norm(total_vort)))
            signatures.append(decompose_to_coefficients(total_vort, basis, xp))
        return magnitudes, signatures
    
    coh_mags, coh_sigs = get_magnitudes_and_sigs(coherent)
    rand_mags, rand_sigs = get_magnitudes_and_sigs(random_text)
    
    # FFT analysis (what failed)
    def fft_ratio(mags):
        if len(mags) < 4:
            return 0
        spectrum = np.abs(np.fft.fft(mags))[:len(mags)//2]
        low = np.sum(spectrum[:len(spectrum)//3])
        high = np.sum(spectrum[len(spectrum)//3:])
        return low / (high + 1e-10)
    
    coh_fft = fft_ratio(coh_mags)
    rand_fft = fft_ratio(rand_mags)
    
    # Brain-like analysis
    coh_brain = compute_vorticity_coherence(coh_sigs, xp)
    rand_brain = compute_vorticity_coherence(rand_sigs, xp)
    
    print(f"\n  Coherent text: '{coherent[:50]}...'")
    print(f"  Random text: '{random_text[:50]}...'")
    
    print(f"\n  FFT (low/high freq ratio):")
    print(f"    Coherent: {coh_fft:.4f}")
    print(f"    Random: {rand_fft:.4f}")
    print(f"    Discriminates? {'YES' if coh_fft > rand_fft else 'NO ← FFT FAILS'}")
    
    print(f"\n  Brain-like (overall coherence):")
    print(f"    Coherent: {coh_brain.overall_coherence:.4f}")
    print(f"    Random: {rand_brain.overall_coherence:.4f}")
    print(f"    Discriminates? {'YES ← BRAIN-LIKE WORKS' if coh_brain.overall_coherence > rand_brain.overall_coherence else 'NO'}")
    
    print(f"\n  Detailed brain-like metrics:")
    print(f"    PLV: Coherent={coh_brain.plv:.3f}, Random={rand_brain.plv:.3f}")
    print(f"    Stability: Coherent={coh_brain.directional_stability:.3f}, Random={rand_brain.directional_stability:.3f}")
    print(f"    Predictability: Coherent={coh_brain.predictability:.3f}, Random={rand_brain.predictability:.3f}")
    
    fft_works = coh_fft > rand_fft
    brain_works = coh_brain.overall_coherence > rand_brain.overall_coherence
    
    print(f"\n  VERDICT:")
    print(f"    FFT discriminates coherent from random: {fft_works}")
    print(f"    Brain-like discriminates coherent from random: {brain_works}")
    
    return brain_works


if __name__ == "__main__":
    passed, _, _ = test_brainlike_metrics()
    print("\n")
    test_comparison_with_fft()
