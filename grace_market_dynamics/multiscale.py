"""
Multi-Timescale Composition — Nested Clifford Analysis
=======================================================

Theory-true multi-scale analysis using nested Clifford encoders.

The key insight: market dynamics occur at multiple timescales.
A trade that looks good on 5-minute data may be noise on hourly data.

This module:
1. Runs parallel encoders at different timescales
2. Composes their witnesses via geometric product
3. Measures cross-scale coherence
4. Only trades when all scales agree
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.algebra import (
    build_clifford_basis,
    geometric_product,
    geometric_product_batch,
    frobenius_cosine,
    decompose_to_coefficients,
    reconstruct_from_coefficients,
    grace_operator,
)
from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ, GRADE_INDICES
from grace_market_dynamics.state_encoder import CliffordState, ObservableAgnosticEncoder


@dataclass
class ScaleConfig:
    """Configuration for a single timescale."""
    name: str
    aggregation_bars: int  # How many base bars to aggregate
    gram_window: int = 20
    weight: float = 1.0


@dataclass
class MultiscaleConfig:
    """Multi-timescale configuration."""
    scales: List[ScaleConfig] = field(default_factory=lambda: [
        ScaleConfig("fast", 1, gram_window=10, weight=0.3),
        ScaleConfig("medium", 5, gram_window=20, weight=0.4),
        ScaleConfig("slow", 20, gram_window=20, weight=0.3),
    ])
    
    # Coherence thresholds
    min_cross_scale_coherence: float = 0.3
    min_directional_agreement: float = 0.6
    
    # Composite signal
    require_all_agree: bool = True


@dataclass
class ScaleState:
    """State at a single timescale."""
    name: str
    clifford_state: Optional[CliffordState]
    chirality: float
    coherence: float
    signal: float  # -1 to +1
    n_updates: int


@dataclass
class MultiscaleState:
    """Combined multi-scale state."""
    scale_states: Dict[str, ScaleState]
    
    # Cross-scale metrics
    cross_scale_coherence: float = 0.0
    directional_agreement: float = 0.0
    
    # Composite Clifford state (product of scale witnesses)
    composite_matrix: Optional[np.ndarray] = None
    composite_chirality: float = 0.0
    
    # Final signal
    composite_signal: float = 0.0
    composite_confidence: float = 0.0


class MultiscaleEncoder:
    """
    Multi-timescale Clifford encoder.
    
    Runs parallel encoders at different aggregation levels,
    then composes their outputs for a unified signal.
    """
    
    def __init__(self, config: MultiscaleConfig = None):
        self.config = config or MultiscaleConfig()
        self.basis = build_clifford_basis(np)
        
        # Create encoder for each scale
        self.encoders: Dict[str, ObservableAgnosticEncoder] = {}
        self.aggregators: Dict[str, PriceAggregator] = {}
        
        for scale in self.config.scales:
            self.encoders[scale.name] = ObservableAgnosticEncoder(
                gram_window=scale.gram_window,
                use_log_returns=True,
                apply_grace=True
            )
            self.aggregators[scale.name] = PriceAggregator(scale.aggregation_bars)
        
        # State tracking
        self.scale_states: Dict[str, ScaleState] = {}
        self.last_state: Optional[MultiscaleState] = None
        
        # History for cross-scale analysis
        self.matrix_history: Dict[str, deque] = {
            s.name: deque(maxlen=20) for s in self.config.scales
        }
    
    def reset(self):
        """Reset all encoders."""
        for encoder in self.encoders.values():
            encoder.reset()
        for agg in self.aggregators.values():
            agg.reset()
        self.scale_states.clear()
        self.last_state = None
        for hist in self.matrix_history.values():
            hist.clear()
    
    def update(self, price: float) -> Optional[MultiscaleState]:
        """
        Process new price and update all scales.
        
        Args:
            price: New base-frequency price
            
        Returns:
            MultiscaleState if all scales have data, else None
        """
        # Update each scale
        for scale in self.config.scales:
            name = scale.name
            aggregator = self.aggregators[name]
            encoder = self.encoders[name]
            
            # Aggregate price
            agg_price = aggregator.update(price)
            
            if agg_price is not None:
                # Update encoder with aggregated price
                clifford_state = encoder.update(np.array([agg_price]))
                
                if clifford_state is not None:
                    # Extract metrics
                    chirality = clifford_state.pseudoscalar
                    mags = clifford_state.grade_magnitudes()
                    total = sum(mags.values()) + 1e-10
                    coherence = (mags['G0'] + mags['G4']) / total
                    
                    # Signal = chirality direction × coherence
                    signal = np.sign(chirality) * coherence
                    
                    # Store state
                    prev_state = self.scale_states.get(name)
                    n_updates = (prev_state.n_updates + 1) if prev_state else 1
                    
                    self.scale_states[name] = ScaleState(
                        name=name,
                        clifford_state=clifford_state,
                        chirality=chirality,
                        coherence=coherence,
                        signal=signal,
                        n_updates=n_updates
                    )
                    
                    # Store matrix for cross-scale analysis
                    M = reconstruct_from_coefficients(clifford_state.coefficients, self.basis, np)
                    self.matrix_history[name].append(M)
        
        # Check if all scales have data
        if len(self.scale_states) < len(self.config.scales):
            return None
        
        # Compute cross-scale metrics
        state = self._compute_multiscale_state()
        self.last_state = state
        return state
    
    def _compute_multiscale_state(self) -> MultiscaleState:
        """Compute combined multi-scale state."""
        # Collect matrices from each scale
        matrices = []
        signals = []
        weights = []
        
        for scale in self.config.scales:
            name = scale.name
            scale_state = self.scale_states.get(name)
            
            if scale_state and scale_state.clifford_state:
                M = reconstruct_from_coefficients(
                    scale_state.clifford_state.coefficients,
                    self.basis, np
                )
                matrices.append(M)
                signals.append(scale_state.signal)
                weights.append(scale.weight)
        
        if len(matrices) < 2:
            return MultiscaleState(
                scale_states=dict(self.scale_states)
            )
        
        matrices = np.array(matrices)
        signals = np.array(signals)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Cross-scale coherence: average pairwise similarity
        coherences = []
        for i in range(len(matrices)):
            for j in range(i + 1, len(matrices)):
                coh = abs(frobenius_cosine(matrices[i], matrices[j], np))
                coherences.append(coh)
        
        cross_scale_coherence = np.mean(coherences) if coherences else 0.0
        
        # Directional agreement: how many scales agree on direction
        signs = np.sign(signals)
        if np.abs(np.sum(signs)) == len(signs):
            directional_agreement = 1.0  # All agree
        else:
            # Fraction agreeing with majority
            majority_sign = np.sign(np.sum(signs * weights))
            agreeing = np.sum((signs == majority_sign) * weights)
            directional_agreement = agreeing
        
        # Composite matrix: geometric product of all scales
        composite_matrix = geometric_product_batch(matrices, np)
        
        # Composite chirality
        composite_coeffs = decompose_to_coefficients(composite_matrix, self.basis, np)
        composite_chirality = composite_coeffs[15]
        
        # Composite signal
        if self.config.require_all_agree:
            # Only signal if all agree
            if directional_agreement >= self.config.min_directional_agreement:
                composite_signal = np.sign(composite_chirality) * cross_scale_coherence
            else:
                composite_signal = 0.0
        else:
            # Weighted average
            composite_signal = np.sum(signals * weights)
        
        # Confidence based on agreement and coherence
        composite_confidence = directional_agreement * cross_scale_coherence
        
        return MultiscaleState(
            scale_states=dict(self.scale_states),
            cross_scale_coherence=cross_scale_coherence,
            directional_agreement=directional_agreement,
            composite_matrix=composite_matrix,
            composite_chirality=composite_chirality,
            composite_signal=composite_signal,
            composite_confidence=composite_confidence
        )
    
    def get_scale_summary(self) -> Dict[str, Dict]:
        """Get summary of all scales."""
        summary = {}
        for name, state in self.scale_states.items():
            summary[name] = {
                'chirality': state.chirality,
                'coherence': state.coherence,
                'signal': state.signal,
                'n_updates': state.n_updates
            }
        return summary


class PriceAggregator:
    """Aggregates prices over N bars."""
    
    def __init__(self, n_bars: int):
        self.n_bars = n_bars
        self.buffer: List[float] = []
    
    def reset(self):
        self.buffer.clear()
    
    def update(self, price: float) -> Optional[float]:
        """
        Add price and return aggregated value if complete.
        
        Returns:
            Aggregated price (close of period) if buffer full, else None
        """
        self.buffer.append(price)
        
        if len(self.buffer) >= self.n_bars:
            # Return last price (close) and reset buffer
            result = self.buffer[-1]
            self.buffer.clear()
            return result
        
        return None


def demo_multiscale():
    """Demo multi-scale analysis."""
    print("="*60)
    print("Multi-Timescale Composition Demo")
    print("="*60)
    
    np.random.seed(42)
    
    # Generate synthetic price series with multiple timescale structure
    n = 1000
    
    # Slow trend (20-bar)
    slow_trend = np.cumsum(np.random.randn(n // 20 + 1) * 0.02)
    slow_trend = np.repeat(slow_trend, 20)[:n]
    
    # Medium oscillation (5-bar)
    medium = np.sin(np.arange(n) * 2 * np.pi / 5) * 0.005
    
    # Fast noise
    fast_noise = np.random.randn(n) * 0.002
    
    prices = 100 * np.exp(slow_trend + medium + fast_noise)
    
    # Create multi-scale encoder
    config = MultiscaleConfig(
        scales=[
            ScaleConfig("fast", 1, gram_window=10, weight=0.2),
            ScaleConfig("medium", 5, gram_window=15, weight=0.4),
            ScaleConfig("slow", 20, gram_window=20, weight=0.4),
        ],
        min_cross_scale_coherence=0.2,
        min_directional_agreement=0.5,
        require_all_agree=True
    )
    
    encoder = MultiscaleEncoder(config)
    
    # Process prices
    print("\nProcessing price stream...")
    states = []
    
    for i, price in enumerate(prices):
        state = encoder.update(price)
        if state:
            states.append(state)
            
            if i % 200 == 0 and i > 0:
                print(f"\n  Bar {i}:")
                print(f"    Cross-scale coherence: {state.cross_scale_coherence:.4f}")
                print(f"    Directional agreement: {state.directional_agreement:.4f}")
                print(f"    Composite signal: {state.composite_signal:.4f}")
                print(f"    Composite confidence: {state.composite_confidence:.4f}")
                
                print("    Scale signals:")
                for name, scale_state in state.scale_states.items():
                    print(f"      {name}: signal={scale_state.signal:.4f}, coh={scale_state.coherence:.4f}")
    
    # Summary
    print("\n" + "-"*40)
    print("Multi-Scale Summary:")
    
    if states:
        coherences = [s.cross_scale_coherence for s in states]
        agreements = [s.directional_agreement for s in states]
        signals = [s.composite_signal for s in states]
        confidences = [s.composite_confidence for s in states]
        
        print(f"  Total states: {len(states)}")
        print(f"  Mean cross-scale coherence: {np.mean(coherences):.4f}")
        print(f"  Mean directional agreement: {np.mean(agreements):.4f}")
        print(f"  Mean composite signal: {np.mean(signals):.4f}")
        print(f"  Mean confidence: {np.mean(confidences):.4f}")
        
        # Signal distribution
        pos_signals = sum(1 for s in signals if s > 0.1)
        neg_signals = sum(1 for s in signals if s < -0.1)
        neutral = len(signals) - pos_signals - neg_signals
        
        print(f"\n  Signal distribution:")
        print(f"    Positive (>0.1): {pos_signals} ({pos_signals/len(signals)*100:.1f}%)")
        print(f"    Negative (<-0.1): {neg_signals} ({neg_signals/len(signals)*100:.1f}%)")
        print(f"    Neutral: {neutral} ({neutral/len(signals)*100:.1f}%)")
    
    print("\n✓ Multi-scale composition working")


if __name__ == "__main__":
    demo_multiscale()
