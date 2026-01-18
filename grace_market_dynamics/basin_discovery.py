"""
Grace Flow Attractors — Basin Discovery for Regime Identification
=================================================================

Theory-true regime identification using Grace basin dynamics.

The Grace operator contracts Clifford multivectors toward stable attractors.
Different market regimes map to different attractor basins.

This module:
1. Discovers attractor basins from market data
2. Classifies new states by their basin
3. Tracks regime transitions
4. Provides regime-aware trading signals
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.algebra import (
    build_clifford_basis,
    grace_operator,
    grace_basin_key_direct,
    frobenius_cosine,
    decompose_to_coefficients,
    reconstruct_from_coefficients,
)
from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_SIX, PHI_INV_CUBE
)
from grace_market_dynamics.state_encoder import CliffordState, ObservableAgnosticEncoder


@dataclass
class Basin:
    """A Grace attractor basin."""
    key: tuple
    centroid: np.ndarray  # Average Clifford state in this basin
    
    # Statistics
    n_observations: int = 0
    
    # Return characteristics
    mean_forward_return: float = 0.0
    std_forward_return: float = 0.0
    
    # Transition probabilities (to other basin keys)
    transitions: Dict[tuple, int] = field(default_factory=dict)
    
    # Stability (how long states stay in this basin)
    avg_dwell_time: float = 0.0
    
    @property
    def transition_probs(self) -> Dict[tuple, float]:
        """Normalized transition probabilities."""
        total = sum(self.transitions.values()) + 1e-10
        return {k: v / total for k, v in self.transitions.items()}


@dataclass
class RegimeState:
    """Current regime classification."""
    basin_key: tuple
    basin: Optional[Basin]
    
    # State metrics
    stability: float = 0.0  # How well state fits basin
    dwell_time: int = 0     # How long in this basin
    
    # Transition prediction
    most_likely_next: Optional[tuple] = None
    transition_confidence: float = 0.0


@dataclass
class BasinDiscoveryConfig:
    """Basin discovery configuration."""
    # Grace iteration settings
    grace_iterations: int = 5
    basin_resolution: float = PHI_INV_SIX
    
    # Basin management
    min_observations_for_basin: int = 10
    max_basins: int = 100
    
    # Similarity thresholds
    basin_merge_threshold: float = 0.95  # Merge basins this similar
    
    # History
    max_history: int = 1000
    
    # Return lookahead
    forward_return_bars: int = 10


class BasinDiscovery:
    """
    Grace basin discovery and regime identification.
    
    Theory: The Grace operator has attractor dynamics.
    Repeatedly applying Grace to a Clifford state contracts
    it toward a stable fixed point. The basin key (discretized
    attractor) identifies which "regime" the market is in.
    """
    
    def __init__(self, config: BasinDiscoveryConfig = None):
        self.config = config or BasinDiscoveryConfig()
        self.basis = build_clifford_basis(np)
        
        # Encoder
        self.encoder = ObservableAgnosticEncoder(
            gram_window=20,
            use_log_returns=True,
            apply_grace=True
        )
        
        # Basin storage
        self.basins: Dict[tuple, Basin] = {}
        
        # State tracking
        self.price_history: List[float] = []
        self.key_history: List[tuple] = []
        self.state_history: List[CliffordState] = []
        
        # Current regime
        self.current_regime: Optional[RegimeState] = None
        self.dwell_counter: int = 0
    
    def reset(self):
        """Reset discovery state."""
        self.encoder.reset()
        self.basins.clear()
        self.price_history.clear()
        self.key_history.clear()
        self.state_history.clear()
        self.current_regime = None
        self.dwell_counter = 0
    
    def _compute_basin_key(self, state: CliffordState) -> tuple:
        """Compute basin key for a Clifford state."""
        M = reconstruct_from_coefficients(state.coefficients, self.basis, np)
        
        # Apply Grace iterations
        for _ in range(self.config.grace_iterations):
            M = grace_operator(M, self.basis, np)
        
        # Compute key from attractor
        return grace_basin_key_direct(
            M, self.basis,
            self.config.grace_iterations,
            self.config.basin_resolution,
            np
        )
    
    def _get_or_create_basin(self, key: tuple, state: CliffordState) -> Basin:
        """Get existing basin or create new one."""
        if key in self.basins:
            return self.basins[key]
        
        # Create new basin
        basin = Basin(
            key=key,
            centroid=state.coefficients.copy(),
            n_observations=0
        )
        
        # Limit basin count
        if len(self.basins) >= self.config.max_basins:
            # Remove least observed basin
            min_key = min(self.basins.keys(), 
                         key=lambda k: self.basins[k].n_observations)
            del self.basins[min_key]
        
        self.basins[key] = basin
        return basin
    
    def _update_basin_statistics(
        self,
        basin: Basin,
        state: CliffordState,
        forward_return: float,
        prev_key: Optional[tuple]
    ):
        """Update basin statistics with new observation."""
        # Update centroid (exponential moving average)
        alpha = 1.0 / (basin.n_observations + 1)
        basin.centroid = (1 - alpha) * basin.centroid + alpha * state.coefficients
        
        # Update return statistics
        n = basin.n_observations
        if n == 0:
            basin.mean_forward_return = forward_return
            basin.std_forward_return = 0.0
        else:
            old_mean = basin.mean_forward_return
            basin.mean_forward_return = (n * old_mean + forward_return) / (n + 1)
            # Welford's online variance
            if n > 0:
                delta = forward_return - old_mean
                delta2 = forward_return - basin.mean_forward_return
                var = basin.std_forward_return ** 2
                new_var = (n * var + delta * delta2) / (n + 1)
                basin.std_forward_return = np.sqrt(max(new_var, 0))
        
        # Update transition counts
        if prev_key is not None and prev_key != basin.key:
            basin.transitions[prev_key] = basin.transitions.get(prev_key, 0) + 1
        
        basin.n_observations += 1
    
    def update(self, price: float) -> Optional[RegimeState]:
        """
        Process new price and update regime classification.
        
        Args:
            price: New price observation
            
        Returns:
            RegimeState with current regime classification
        """
        self.price_history.append(price)
        
        # Update encoder
        state = self.encoder.update(np.array([price]))
        
        if state is None:
            return None
        
        self.state_history.append(state)
        
        # Compute basin key
        key = self._compute_basin_key(state)
        
        # Get previous key for transition tracking
        prev_key = self.key_history[-1] if self.key_history else None
        self.key_history.append(key)
        
        # Compute forward return (lagged update)
        forward_return = 0.0
        if len(self.price_history) > self.config.forward_return_bars:
            past_price = self.price_history[-self.config.forward_return_bars - 1]
            current_price = self.price_history[-1]
            forward_return = np.log(current_price / past_price)
            
            # Update old basin with forward return
            if len(self.key_history) > self.config.forward_return_bars:
                old_key = self.key_history[-self.config.forward_return_bars - 1]
                old_state = self.state_history[-self.config.forward_return_bars - 1]
                old_prev_key = self.key_history[-self.config.forward_return_bars - 2] if len(self.key_history) > self.config.forward_return_bars + 1 else None
                
                if old_key in self.basins:
                    self._update_basin_statistics(
                        self.basins[old_key],
                        old_state,
                        forward_return,
                        old_prev_key
                    )
        
        # Get or create basin
        basin = self._get_or_create_basin(key, state)
        
        # Track dwell time
        if prev_key == key:
            self.dwell_counter += 1
        else:
            # Regime transition
            if self.current_regime and self.current_regime.basin:
                # Update dwell time statistics for previous basin
                prev_basin = self.current_regime.basin
                n = prev_basin.n_observations
                if n > 0:
                    old_avg = prev_basin.avg_dwell_time
                    prev_basin.avg_dwell_time = (n * old_avg + self.dwell_counter) / (n + 1)
            
            self.dwell_counter = 1
        
        # Compute stability (how well state fits basin centroid)
        if basin.n_observations > 0:
            M_state = reconstruct_from_coefficients(state.coefficients, self.basis, np)
            M_centroid = reconstruct_from_coefficients(basin.centroid, self.basis, np)
            stability = abs(frobenius_cosine(M_state, M_centroid, np))
        else:
            stability = 1.0
        
        # Predict next regime
        most_likely_next = None
        transition_confidence = 0.0
        
        trans_probs = basin.transition_probs
        if trans_probs:
            most_likely_next = max(trans_probs.keys(), key=lambda k: trans_probs[k])
            transition_confidence = trans_probs[most_likely_next]
        
        # Build regime state
        self.current_regime = RegimeState(
            basin_key=key,
            basin=basin,
            stability=stability,
            dwell_time=self.dwell_counter,
            most_likely_next=most_likely_next,
            transition_confidence=transition_confidence
        )
        
        return self.current_regime
    
    def get_regime_signal(self) -> Tuple[float, float]:
        """
        Get trading signal based on current regime.
        
        Returns:
            (signal, confidence) based on basin return characteristics
        """
        if self.current_regime is None or self.current_regime.basin is None:
            return 0.0, 0.0
        
        basin = self.current_regime.basin
        
        # Only signal if enough observations
        if basin.n_observations < self.config.min_observations_for_basin:
            return 0.0, 0.0
        
        # Signal from expected return direction
        mean_ret = basin.mean_forward_return
        std_ret = basin.std_forward_return + 1e-10
        
        # Signal strength = mean return normalized by volatility
        signal = np.clip(mean_ret / std_ret, -1, 1)
        
        # Confidence from stability and observation count
        obs_confidence = min(basin.n_observations / 50, 1.0)
        confidence = self.current_regime.stability * obs_confidence
        
        return signal, confidence
    
    def get_regime_summary(self) -> Dict:
        """Get summary of discovered regimes."""
        if not self.basins:
            return {'n_basins': 0}
        
        basin_stats = []
        for key, basin in self.basins.items():
            basin_stats.append({
                'key': key[:4],  # First 4 elements for brevity
                'n_obs': basin.n_observations,
                'mean_return': basin.mean_forward_return * 10000,  # bps
                'std_return': basin.std_forward_return * 10000,
                'avg_dwell': basin.avg_dwell_time,
                'n_transitions': len(basin.transitions)
            })
        
        # Sort by observation count
        basin_stats.sort(key=lambda x: x['n_obs'], reverse=True)
        
        return {
            'n_basins': len(self.basins),
            'total_observations': sum(b.n_observations for b in self.basins.values()),
            'top_basins': basin_stats[:10]
        }


def demo_basin_discovery():
    """Demo basin discovery with synthetic data."""
    print("="*60)
    print("Grace Basin Discovery Demo")
    print("="*60)
    
    np.random.seed(42)
    
    # Generate synthetic data with regime changes
    n = 800
    prices = [100.0]
    
    # Regime 1: Bull trend (bars 0-200)
    for i in range(200):
        ret = np.random.randn() * 0.005 + 0.002
        prices.append(prices[-1] * np.exp(ret))
    
    # Regime 2: Choppy/Sideways (bars 200-400)
    for i in range(200):
        ret = np.random.randn() * 0.01
        prices.append(prices[-1] * np.exp(ret))
    
    # Regime 3: Bear trend (bars 400-600)
    for i in range(200):
        ret = np.random.randn() * 0.008 - 0.003
        prices.append(prices[-1] * np.exp(ret))
    
    # Regime 4: Bull again (bars 600-800)
    for i in range(200):
        ret = np.random.randn() * 0.006 + 0.002
        prices.append(prices[-1] * np.exp(ret))
    
    prices = np.array(prices)
    
    # Create basin discovery
    config = BasinDiscoveryConfig(
        grace_iterations=3,
        basin_resolution=PHI_INV_SIX,
        min_observations_for_basin=5,
        forward_return_bars=10
    )
    
    discovery = BasinDiscovery(config)
    
    # Process prices
    print("\nProcessing price stream with regime changes...")
    regimes = []
    signals = []
    
    for i, price in enumerate(prices):
        regime = discovery.update(price)
        if regime:
            regimes.append(regime)
            
            signal, conf = discovery.get_regime_signal()
            signals.append((signal, conf))
            
            if i in [100, 300, 500, 700]:
                print(f"\n  Bar {i} (Regime checkpoint):")
                print(f"    Price: {price:.2f}")
                print(f"    Basin key: {regime.basin_key[:4]}...")
                print(f"    Dwell time: {regime.dwell_time}")
                print(f"    Stability: {regime.stability:.4f}")
                if regime.basin:
                    print(f"    Basin mean return: {regime.basin.mean_forward_return*10000:.2f} bps")
                print(f"    Signal: {signal:.4f}, Confidence: {conf:.4f}")
    
    # Summary
    print("\n" + "-"*40)
    print("Basin Discovery Summary:")
    
    summary = discovery.get_regime_summary()
    print(f"  Total basins discovered: {summary['n_basins']}")
    print(f"  Total observations: {summary['total_observations']}")
    
    if summary.get('top_basins'):
        print("\n  Top Basins by Observation Count:")
        for i, basin in enumerate(summary['top_basins'][:5]):
            print(f"    {i+1}. Key: {basin['key']}")
            print(f"       Obs: {basin['n_obs']}, Mean ret: {basin['mean_return']:.2f} bps")
            print(f"       Avg dwell: {basin['avg_dwell']:.1f} bars")
    
    # Signal effectiveness
    if signals:
        sig_values = [s[0] for s in signals]
        confidences = [s[1] for s in signals]
        
        print(f"\n  Signal Statistics:")
        print(f"    Total signals: {len(signals)}")
        print(f"    Mean signal: {np.mean(sig_values):.4f}")
        print(f"    Mean confidence: {np.mean(confidences):.4f}")
        print(f"    Positive signals: {sum(1 for s in sig_values if s > 0.1)}")
        print(f"    Negative signals: {sum(1 for s in sig_values if s < -0.1)}")
    
    print("\n✓ Basin discovery working")


if __name__ == "__main__":
    demo_basin_discovery()
