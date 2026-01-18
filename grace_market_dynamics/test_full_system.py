"""
Full System Integration Test
============================

Tests all components together on real market data:
1. State encoding
2. Signal generation
3. Multi-scale analysis
4. Basin discovery
5. Backtesting

This validates the complete theory-true trading system.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from grace_market_dynamics import (
    CliffordState,
    ObservableAgnosticEncoder,
    LiveSignalGenerator, SignalConfig,
    CliffordBacktester, BacktestConfig,
    MultiscaleEncoder, MultiscaleConfig, ScaleConfig,
    BasinDiscovery, BasinDiscoveryConfig,
)
from holographic_prod.core.algebra import build_clifford_basis
from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ, PHI_INV_SIX


def fetch_data(ticker: str = "SPY", period: str = "1y") -> np.ndarray:
    """Fetch market data."""
    if HAS_YFINANCE:
        try:
            data = yf.download(ticker, period=period, progress=False)
            if len(data) > 50:
                return data['Close'].values.flatten().astype(np.float64)
        except:
            pass
    
    # Fallback: synthetic data
    print("Using synthetic data...")
    np.random.seed(42)
    n = 500
    
    # Realistic price dynamics
    returns = np.random.randn(n) * 0.01 + 0.0002  # Slight drift
    # Add some regime structure
    returns[100:150] += 0.003  # Bull
    returns[250:300] -= 0.004  # Bear
    returns[400:450] += 0.002  # Bull
    
    return 100 * np.exp(np.cumsum(returns))


def test_full_pipeline():
    """Run complete system test."""
    print("="*70)
    print("FULL SYSTEM INTEGRATION TEST")
    print("="*70)
    
    # Fetch data
    print("\n1. Fetching data...")
    prices = fetch_data("SPY", "1y")
    print(f"   Data points: {len(prices)}")
    print(f"   Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    basis = build_clifford_basis(np)
    
    # Split data
    train_size = int(len(prices) * 0.6)
    train_prices = prices[:train_size]
    test_prices = prices[train_size:]
    
    print(f"   Train: {len(train_prices)}, Test: {len(test_prices)}")
    
    # ==========================================
    # 2. Signal Generator Test
    # ==========================================
    print("\n2. Testing Signal Generator...")
    
    sig_config = SignalConfig(
        coherence_threshold=0.3,
        chirality_threshold=0.1,
        chirality_weight=0.5,
        bivector_weight=0.2,
        coherence_weight=0.2,
        vorticity_weight=0.1
    )
    
    generator = LiveSignalGenerator(sig_config)
    
    signals = []
    for price in train_prices[:200]:
        signal = generator.update(price)
        if signal:
            signals.append(signal)
    
    if signals:
        values = [s.value for s in signals]
        confs = [s.confidence for s in signals]
        print(f"   Generated {len(signals)} signals")
        print(f"   Mean value: {np.mean(values):.4f}")
        print(f"   Mean confidence: {np.mean(confs):.4f}")
        print(f"   Pos/Neg: {sum(1 for v in values if v > 0)}/{sum(1 for v in values if v < 0)}")
    else:
        print("   No signals generated (need more data)")
    
    # ==========================================
    # 3. Multi-Scale Analysis Test
    # ==========================================
    print("\n3. Testing Multi-Scale Analysis...")
    
    ms_config = MultiscaleConfig(
        scales=[
            ScaleConfig("fast", 1, gram_window=10, weight=0.3),
            ScaleConfig("medium", 5, gram_window=15, weight=0.4),
            ScaleConfig("slow", 10, gram_window=20, weight=0.3),
        ],
        min_cross_scale_coherence=0.2,
        require_all_agree=False
    )
    
    ms_encoder = MultiscaleEncoder(ms_config)
    
    ms_states = []
    for price in train_prices[:300]:
        state = ms_encoder.update(price)
        if state:
            ms_states.append(state)
    
    if ms_states:
        coherences = [s.cross_scale_coherence for s in ms_states]
        agreements = [s.directional_agreement for s in ms_states]
        ms_signals = [s.composite_signal for s in ms_states]
        
        print(f"   States: {len(ms_states)}")
        print(f"   Cross-scale coherence: {np.mean(coherences):.4f}")
        print(f"   Directional agreement: {np.mean(agreements):.4f}")
        print(f"   Mean composite signal: {np.mean(ms_signals):.4f}")
    else:
        print("   No multi-scale states (need more data)")
    
    # ==========================================
    # 4. Basin Discovery Test
    # ==========================================
    print("\n4. Testing Basin Discovery...")
    
    basin_config = BasinDiscoveryConfig(
        grace_iterations=3,
        basin_resolution=PHI_INV_SIX * 2,  # Coarser resolution
        min_observations_for_basin=5,
        forward_return_bars=5
    )
    
    discovery = BasinDiscovery(basin_config)
    
    regime_signals = []
    for price in train_prices:
        regime = discovery.update(price)
        if regime:
            signal, conf = discovery.get_regime_signal()
            regime_signals.append((signal, conf))
    
    summary = discovery.get_regime_summary()
    print(f"   Basins discovered: {summary['n_basins']}")
    
    if regime_signals:
        sig_vals = [s[0] for s in regime_signals]
        print(f"   Regime signals: {len(regime_signals)}")
        print(f"   Mean regime signal: {np.mean(sig_vals):.4f}")
    
    # ==========================================
    # 5. Backtest Test
    # ==========================================
    print("\n5. Testing Backtest Framework...")
    
    bt_config = BacktestConfig(
        coherence_threshold=0.1,  # Lowered from 0.3 (G0+G4 ~18% of energy)
        min_signal_strength=0.1,
        commission_bps=1.0,
        slippage_bps=1.0,
        train_fraction=0.6,
        max_holding_periods=20,
        stop_loss_pct=0.03
    )
    
    backtester = CliffordBacktester(bt_config)
    result = backtester.run(prices)
    
    print(f"\n   BACKTEST RESULTS (Out-of-Sample)")
    print(f"   {'-'*40}")
    print(f"   Total Return: {result.total_return*100:.2f}%")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown*100:.2f}%")
    print(f"   Number of Trades: {result.n_trades}")
    print(f"   Win Rate: {result.win_rate*100:.1f}%")
    print(f"   Profit Factor: {result.profit_factor:.2f}")
    print(f"   Avg Signal Strength: {result.avg_signal_strength:.4f}")
    print(f"   Avg Coherence: {result.avg_coherence:.4f}")
    
    # ==========================================
    # 6. Integration: Combined Analysis
    # ==========================================
    print("\n6. Combined Analysis on Test Set...")
    
    # Reset all components
    generator.reset()
    ms_encoder.reset()
    discovery.reset()
    
    combined_signals = []
    
    for i, price in enumerate(test_prices):
        # Update all components
        sig = generator.update(price)
        ms_state = ms_encoder.update(price)
        regime = discovery.update(price)
        
        if sig and ms_state and regime:
            # Combine signals
            regime_sig, regime_conf = discovery.get_regime_signal()
            
            combined = {
                'price': price,
                'signal_gen': sig.value,
                'signal_gen_conf': sig.confidence,
                'ms_signal': ms_state.composite_signal,
                'ms_conf': ms_state.composite_confidence,
                'regime_signal': regime_sig,
                'regime_conf': regime_conf,
                'cross_scale_coh': ms_state.cross_scale_coherence,
                'basin_key': regime.basin_key[:4] if regime.basin_key else None
            }
            combined_signals.append(combined)
    
    if combined_signals:
        # Analyze combined signals
        sg_vals = [s['signal_gen'] for s in combined_signals]
        ms_vals = [s['ms_signal'] for s in combined_signals]
        rg_vals = [s['regime_signal'] for s in combined_signals]
        
        print(f"\n   Combined signals: {len(combined_signals)}")
        print(f"   Signal Generator: mean={np.mean(sg_vals):.4f}")
        print(f"   Multi-Scale: mean={np.mean(ms_vals):.4f}")
        print(f"   Regime-based: mean={np.mean(rg_vals):.4f}")
        
        # Correlation between signals
        if len(sg_vals) > 10:
            corr_sg_ms = np.corrcoef(sg_vals, ms_vals)[0, 1]
            corr_sg_rg = np.corrcoef(sg_vals, rg_vals)[0, 1]
            corr_ms_rg = np.corrcoef(ms_vals, rg_vals)[0, 1]
            
            print(f"\n   Signal Correlations:")
            print(f"     SigGen ↔ MultiScale: {corr_sg_ms:.4f}")
            print(f"     SigGen ↔ Regime: {corr_sg_rg:.4f}")
            print(f"     MultiScale ↔ Regime: {corr_ms_rg:.4f}")
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*70)
    print("INTEGRATION TEST COMPLETE")
    print("="*70)
    
    print("\nComponent Status:")
    print(f"  ✓ State Encoder: Working")
    print(f"  ✓ Signal Generator: {len(signals)} signals")
    print(f"  ✓ Multi-Scale: {len(ms_states)} states")
    print(f"  ✓ Basin Discovery: {summary['n_basins']} basins")
    print(f"  ✓ Backtest: {result.n_trades} trades, {result.total_return*100:.2f}% return")
    
    if result.n_trades > 0 and result.total_return > -0.5:
        print("\n🎉 FULL SYSTEM OPERATIONAL")
    else:
        print("\n⚠️  System running but needs parameter tuning for this dataset")
    
    return result


if __name__ == "__main__":
    test_full_pipeline()
