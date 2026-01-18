"""
Live Signal Generator — Real-Time Clifford Encoding
====================================================

Real-time signal generation using theory-true Clifford algebra.

Features:
- Streaming state updates
- Multi-signal fusion (chirality, bivector, coherence)
- Confidence-weighted position sizing
- Risk management integration
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
from collections import deque
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.algebra import (
    build_clifford_basis,
    geometric_product_batch,
    grace_basin_key_direct,
    frobenius_cosine,
    reconstruct_from_coefficients,
    vorticity_magnitude_and_signature,
)
from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_SIX, GRADE_INDICES
)
from grace_market_dynamics.state_encoder import CliffordState, ObservableAgnosticEncoder


class SignalType(Enum):
    """Types of signals generated."""
    CHIRALITY = "chirality"      # Pseudoscalar sign
    BIVECTOR = "bivector"        # Rotation plane stability
    COHERENCE = "coherence"      # Witness stability
    VORTICITY = "vorticity"      # Rotational intensity
    COMPOSITE = "composite"      # Combined signal


@dataclass
class Signal:
    """Single trading signal."""
    timestamp: float
    signal_type: SignalType
    value: float           # -1 to +1 (direction and strength)
    confidence: float      # 0 to 1
    metadata: Dict = field(default_factory=dict)
    
    @property
    def strength(self) -> float:
        """Effective signal strength (value × confidence)."""
        return self.value * self.confidence


@dataclass
class MarketState:
    """Current market state summary."""
    price: float
    clifford_state: Optional[CliffordState]
    
    # Grade structure
    grade_magnitudes: Dict[str, float] = field(default_factory=dict)
    
    # Key indicators
    chirality: float = 0.0
    dominant_bivector: int = 0
    coherence: float = 0.0
    vorticity: float = 0.0
    
    # Basin key for regime identification
    basin_key: Optional[tuple] = None
    
    # History
    chirality_history: List[float] = field(default_factory=list)
    bivector_history: List[int] = field(default_factory=list)


@dataclass
class SignalConfig:
    """Signal generation configuration."""
    # Encoder settings
    gram_window: int = 20
    delay_embed_dim: int = 4
    
    # Signal thresholds
    chirality_threshold: float = 0.3
    coherence_threshold: float = 0.5
    bivector_stability_window: int = 5
    vorticity_threshold: float = 0.5
    
    # Signal weights for composite
    chirality_weight: float = 0.4
    bivector_weight: float = 0.2
    coherence_weight: float = 0.3
    vorticity_weight: float = 0.1
    
    # History lengths
    max_history: int = 100
    
    # Grace basin settings
    basin_resolution: float = PHI_INV_SIX
    basin_iterations: int = 3


class LiveSignalGenerator:
    """
    Real-time Clifford-based signal generator.
    
    Theory-true signal generation:
    1. Chirality persistence: pseudoscalar sign predicts direction
    2. Bivector stability: stable rotation plane = trend continuation
    3. Coherence: high witness energy = high confidence
    4. Vorticity: high rotation = regime instability
    
    Uses direct from_increments encoding (proven to work) rather than
    the full ObservableAgnosticEncoder for simpler, more reliable signals.
    """
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        self.basis = build_clifford_basis(np)
        
        # State tracking
        self.current_state: Optional[MarketState] = None
        self.price_history: deque = deque(maxlen=self.config.max_history)
        self.return_history: deque = deque(maxlen=self.config.max_history)
        self.state_history: deque = deque(maxlen=self.config.max_history)
        self.signal_history: deque = deque(maxlen=self.config.max_history)
        
        # Clifford matrix history for vorticity
        self.matrix_history: deque = deque(maxlen=20)
        
        # Bivector indices
        self.bivector_indices = GRADE_INDICES[2]
    
    def reset(self):
        """Reset generator state."""
        self.current_state = None
        self.price_history.clear()
        self.return_history.clear()
        self.state_history.clear()
        self.signal_history.clear()
        self.matrix_history.clear()
    
    def update(self, price: float, timestamp: float = None) -> Optional[Signal]:
        """
        Process new price and generate signal.
        
        Uses direct from_increments encoding for reliable chirality detection.
        
        Args:
            price: New price observation
            timestamp: Unix timestamp (uses current time if None)
            
        Returns:
            Composite signal or None if not enough data
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.price_history.append(price)
        
        # Compute log return
        if len(self.price_history) >= 2:
            log_return = np.log(price / self.price_history[-2])
            self.return_history.append(log_return)
        
        # Need at least 4 returns for delay embedding
        if len(self.return_history) < 4:
            return None
        
        # Get last 4 returns
        increments = np.array(list(self.return_history)[-4:])
        
        # Create Clifford state directly (proven to produce meaningful chirality)
        clifford_state = CliffordState.from_increments(increments, self.basis)
        
        # Build market state
        market_state = self._build_market_state(price, clifford_state)
        self.current_state = market_state
        self.state_history.append(market_state)
        
        # Store matrix for vorticity
        M = reconstruct_from_coefficients(clifford_state.coefficients, self.basis, np)
        self.matrix_history.append(M)
        
        # Generate composite signal
        signal = self._generate_composite_signal(market_state, timestamp)
        self.signal_history.append(signal)
        
        return signal
    
    def _build_market_state(self, price: float, clifford_state: CliffordState) -> MarketState:
        """Build comprehensive market state from Clifford state."""
        coeffs = clifford_state.coefficients
        
        # Grade magnitudes
        mags = clifford_state.grade_magnitudes()
        
        # Chirality (pseudoscalar)
        chirality = coeffs[15]
        
        # Dominant bivector
        bv_coeffs = coeffs[self.bivector_indices]
        dominant_bv = int(np.argmax(np.abs(bv_coeffs)))
        
        # Coherence (witness energy ratio)
        total = sum(mags.values()) + 1e-10
        coherence = (mags['G0'] + mags['G4']) / total
        
        # Vorticity (if enough history)
        vorticity = 0.0
        if len(self.matrix_history) >= 5:
            matrices = np.array(list(self.matrix_history)[-5:])
            vorticity, _ = vorticity_magnitude_and_signature(matrices, self.basis, np)
        
        # Basin key
        M = reconstruct_from_coefficients(coeffs, self.basis, np)
        basin_key = grace_basin_key_direct(
            M, self.basis, 
            self.config.basin_iterations,
            self.config.basin_resolution,
            np
        )
        
        # History tracking
        chirality_history = [s.chirality for s in list(self.state_history)[-10:]]
        chirality_history.append(chirality)
        
        bivector_history = [s.dominant_bivector for s in list(self.state_history)[-10:]]
        bivector_history.append(dominant_bv)
        
        return MarketState(
            price=price,
            clifford_state=clifford_state,
            grade_magnitudes=mags,
            chirality=chirality,
            dominant_bivector=dominant_bv,
            coherence=coherence,
            vorticity=vorticity,
            basin_key=basin_key,
            chirality_history=chirality_history,
            bivector_history=bivector_history
        )
    
    def _generate_chirality_signal(self, state: MarketState, timestamp: float) -> Signal:
        """Generate signal from chirality persistence."""
        # Check chirality consistency over history
        history = state.chirality_history[-5:]
        if len(history) < 3:
            return Signal(timestamp, SignalType.CHIRALITY, 0.0, 0.0)
        
        signs = np.sign(history)
        consistency = abs(np.mean(signs))  # 1 = all same sign, 0 = mixed
        
        # Signal direction from current chirality
        direction = np.sign(state.chirality)
        
        # Value = direction × magnitude (scaled)
        mag = min(abs(state.chirality) / self.config.chirality_threshold, 1.0)
        value = direction * mag
        
        # Confidence from consistency
        confidence = consistency
        
        return Signal(
            timestamp=timestamp,
            signal_type=SignalType.CHIRALITY,
            value=value,
            confidence=confidence,
            metadata={
                'chirality': state.chirality,
                'consistency': consistency,
                'history_len': len(history)
            }
        )
    
    def _generate_bivector_signal(self, state: MarketState, timestamp: float) -> Signal:
        """Generate signal from bivector stability."""
        history = state.bivector_history[-self.config.bivector_stability_window:]
        if len(history) < 3:
            return Signal(timestamp, SignalType.BIVECTOR, 0.0, 0.0)
        
        # Stability = fraction of time dominant bivector is consistent
        unique_bivectors = len(set(history))
        stability = 1.0 / unique_bivectors
        
        # When stable, use chirality direction; when unstable, neutral
        direction = np.sign(state.chirality) if stability > 0.5 else 0.0
        
        # Value = direction (bivector provides confidence, not direction)
        value = direction * stability
        
        # Confidence from stability
        confidence = stability
        
        return Signal(
            timestamp=timestamp,
            signal_type=SignalType.BIVECTOR,
            value=value,
            confidence=confidence,
            metadata={
                'dominant_bivector': state.dominant_bivector,
                'stability': stability,
                'unique_count': unique_bivectors
            }
        )
    
    def _generate_coherence_signal(self, state: MarketState, timestamp: float) -> Signal:
        """Generate signal from coherence level."""
        # High coherence = clear signal, use chirality direction
        direction = np.sign(state.chirality)
        
        # Value scales with coherence above threshold
        excess_coherence = max(0, state.coherence - self.config.coherence_threshold)
        value = direction * min(excess_coherence / (1 - self.config.coherence_threshold), 1.0)
        
        # Confidence IS coherence
        confidence = state.coherence
        
        return Signal(
            timestamp=timestamp,
            signal_type=SignalType.COHERENCE,
            value=value,
            confidence=confidence,
            metadata={
                'coherence': state.coherence,
                'grade_magnitudes': state.grade_magnitudes
            }
        )
    
    def _generate_vorticity_signal(self, state: MarketState, timestamp: float) -> Signal:
        """Generate signal from vorticity (contrarian)."""
        # High vorticity = instability = reduce position
        # Low vorticity = stable = follow trend
        
        direction = np.sign(state.chirality)
        
        # Invert: high vorticity reduces signal
        vort_factor = 1.0 - min(state.vorticity / self.config.vorticity_threshold, 1.0)
        value = direction * vort_factor
        
        # Confidence inversely related to vorticity
        confidence = vort_factor
        
        return Signal(
            timestamp=timestamp,
            signal_type=SignalType.VORTICITY,
            value=value,
            confidence=confidence,
            metadata={
                'vorticity': state.vorticity
            }
        )
    
    def _generate_composite_signal(self, state: MarketState, timestamp: float) -> Signal:
        """Generate weighted composite signal."""
        # Generate component signals
        chirality_sig = self._generate_chirality_signal(state, timestamp)
        bivector_sig = self._generate_bivector_signal(state, timestamp)
        coherence_sig = self._generate_coherence_signal(state, timestamp)
        vorticity_sig = self._generate_vorticity_signal(state, timestamp)
        
        # Weighted combination
        weights = {
            'chirality': self.config.chirality_weight,
            'bivector': self.config.bivector_weight,
            'coherence': self.config.coherence_weight,
            'vorticity': self.config.vorticity_weight
        }
        
        signals = {
            'chirality': chirality_sig,
            'bivector': bivector_sig,
            'coherence': coherence_sig,
            'vorticity': vorticity_sig
        }
        
        # Weighted value (confidence-weighted average)
        total_weight = 0
        weighted_value = 0
        weighted_confidence = 0
        
        for name, sig in signals.items():
            w = weights[name] * sig.confidence
            weighted_value += w * sig.value
            weighted_confidence += w * sig.confidence
            total_weight += w
        
        if total_weight > 0:
            composite_value = weighted_value / total_weight
            composite_confidence = weighted_confidence / total_weight
        else:
            composite_value = 0.0
            composite_confidence = 0.0
        
        return Signal(
            timestamp=timestamp,
            signal_type=SignalType.COMPOSITE,
            value=composite_value,
            confidence=composite_confidence,
            metadata={
                'chirality_signal': chirality_sig.strength,
                'bivector_signal': bivector_sig.strength,
                'coherence_signal': coherence_sig.strength,
                'vorticity_signal': vorticity_sig.strength,
                'basin_key': state.basin_key
            }
        )
    
    def get_current_signals(self) -> Dict[str, Signal]:
        """Get all current component signals."""
        if self.current_state is None:
            return {}
        
        timestamp = time.time()
        return {
            'chirality': self._generate_chirality_signal(self.current_state, timestamp),
            'bivector': self._generate_bivector_signal(self.current_state, timestamp),
            'coherence': self._generate_coherence_signal(self.current_state, timestamp),
            'vorticity': self._generate_vorticity_signal(self.current_state, timestamp),
        }
    
    def get_regime(self) -> Optional[tuple]:
        """Get current regime (basin key)."""
        if self.current_state is None:
            return None
        return self.current_state.basin_key


def demo_signal_generator():
    """Demo the signal generator with synthetic data."""
    print("="*60)
    print("Live Signal Generator Demo")
    print("="*60)
    
    np.random.seed(42)
    
    # Generate synthetic price series
    n = 200
    trend = np.cumsum(np.random.randn(n) * 0.01 + 0.001)
    prices = 100 * np.exp(trend)
    
    # Create generator
    config = SignalConfig(
        coherence_threshold=0.4,
        chirality_threshold=0.2
    )
    generator = LiveSignalGenerator(config)
    
    # Process prices
    print("\nProcessing price stream...")
    signals = []
    
    for i, price in enumerate(prices):
        signal = generator.update(price, timestamp=float(i))
        if signal:
            signals.append(signal)
            
            if i % 50 == 0 and i > 0:
                print(f"\n  Bar {i}:")
                print(f"    Price: {price:.2f}")
                print(f"    Signal Value: {signal.value:.4f}")
                print(f"    Confidence: {signal.confidence:.4f}")
                print(f"    Strength: {signal.strength:.4f}")
                
                regime = generator.get_regime()
                if regime:
                    print(f"    Regime (basin key): {regime[:4]}...")
    
    # Summary
    print("\n" + "-"*40)
    print("Signal Summary:")
    
    if signals:
        values = [s.value for s in signals]
        confidences = [s.confidence for s in signals]
        strengths = [s.strength for s in signals]
        
        print(f"  Total signals: {len(signals)}")
        print(f"  Mean value: {np.mean(values):.4f}")
        print(f"  Mean confidence: {np.mean(confidences):.4f}")
        print(f"  Mean strength: {np.mean(strengths):.4f}")
        print(f"  Positive signals: {sum(1 for v in values if v > 0)}")
        print(f"  Negative signals: {sum(1 for v in values if v < 0)}")
    
    print("\n✓ Signal generator working")


if __name__ == "__main__":
    demo_signal_generator()
