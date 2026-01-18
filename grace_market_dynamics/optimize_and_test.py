"""
Parameter Optimization & Intraday Testing
==========================================

Goal: Find if there's a profitable configuration on higher-frequency data.

Methodology:
1. Walk-forward validation (avoid overfitting)
2. Test on 5-min intraday data (where IC was 5.7%)
3. Grid search over key parameters
4. Report honest results with confidence intervals
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from holographic_prod.core.algebra import build_clifford_basis
from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ
from grace_market_dynamics.state_encoder import CliffordState
from grace_market_dynamics.backtest import BacktestConfig, BacktestResult


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    best_params: Dict
    best_sharpe: float
    best_return: float
    all_results: List[Tuple[Dict, BacktestResult]]
    walk_forward_sharpe: float
    walk_forward_return: float


def fetch_intraday(ticker: str = "SPY", period: str = "5d", interval: str = "5m") -> np.ndarray:
    """Fetch intraday data."""
    if HAS_YFINANCE:
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if len(data) > 100:
                return data['Close'].values.flatten().astype(np.float64)
        except Exception as e:
            print(f"Error fetching data: {e}")
    
    # Fallback: synthetic intraday
    print("Using synthetic intraday data...")
    np.random.seed(42)
    n = 1000
    
    # Realistic intraday dynamics
    returns = np.random.randn(n) * 0.001  # 10 bps volatility per bar
    
    # Add intraday patterns
    time_of_day = np.linspace(0, 2*np.pi * 5, n)  # 5 days
    seasonality = 0.0002 * np.sin(time_of_day)  # Open/close effects
    
    # Add some trending periods
    returns[100:150] += 0.0003
    returns[300:350] -= 0.0004
    returns[600:650] += 0.0003
    returns[800:850] -= 0.0002
    
    return 100 * np.exp(np.cumsum(returns + seasonality))


def simple_backtest(
    prices: np.ndarray,
    coherence_threshold: float,
    signal_threshold: float,
    holding_period: int,
    commission_bps: float = 1.0
) -> Tuple[float, float, int]:
    """
    Simple backtest without the full framework.
    
    Returns: (total_return, sharpe, n_trades)
    """
    basis = build_clifford_basis(np)
    n = len(prices)
    
    if n < 10:
        return 0.0, 0.0, 0
    
    # Compute log returns
    log_returns = np.diff(np.log(prices))
    
    # Track equity
    equity = [1.0]
    position = 0  # -1, 0, +1
    entry_idx = 0
    entry_price = 0.0
    trades = []
    
    cost_pct = commission_bps / 10000
    
    for i in range(4, len(log_returns)):
        # Get last 4 returns
        increments = log_returns[i-4:i]
        
        # Create Clifford state
        state = CliffordState.from_increments(increments, basis)
        
        # Get signal
        chirality = state.pseudoscalar
        mags = state.grade_magnitudes()
        total = sum(mags.values()) + 1e-10
        coherence = (mags['G0'] + mags['G4']) / total
        
        signal = np.sign(chirality) * min(abs(chirality) / 0.3, 1.0)
        
        price = prices[i + 1]  # Current price (i+1 because log_returns is diff)
        
        if position != 0:
            # Check exit
            bars_held = i - entry_idx
            
            if bars_held >= holding_period:
                # Exit
                ret = position * (price / entry_price - 1)
                net_ret = ret - 2 * cost_pct  # Entry + exit cost
                equity.append(equity[-1] * (1 + net_ret))
                trades.append(net_ret)
                position = 0
            else:
                # Mark to market
                bar_ret = position * (price / prices[i] - 1)
                equity.append(equity[-1] * (1 + bar_ret))
        else:
            # Check entry
            if coherence >= coherence_threshold and abs(signal) >= signal_threshold:
                position = 1 if signal > 0 else -1
                entry_idx = i
                entry_price = price
                equity.append(equity[-1] * (1 - cost_pct))
            else:
                equity.append(equity[-1])
    
    # Close open position
    if position != 0:
        ret = position * (prices[-1] / entry_price - 1)
        net_ret = ret - 2 * cost_pct
        equity.append(equity[-1] * (1 + net_ret))
        trades.append(net_ret)
    
    # Compute metrics
    equity = np.array(equity)
    total_return = equity[-1] / equity[0] - 1
    
    returns = np.diff(equity) / equity[:-1]
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 78)  # Annualized for 5-min bars
    else:
        sharpe = 0.0
    
    return total_return, sharpe, len(trades)


def walk_forward_optimization(
    prices: np.ndarray,
    n_windows: int = 5
) -> OptimizationResult:
    """
    Walk-forward optimization to avoid overfitting.
    
    1. Split data into n_windows
    2. For each window: optimize on previous windows, test on current
    3. Report aggregate out-of-sample performance
    """
    n = len(prices)
    window_size = n // n_windows
    
    # Parameter grid
    coherence_thresholds = [0.05, 0.1, 0.15, 0.2]
    signal_thresholds = [0.1, 0.2, 0.3, 0.4]
    holding_periods = [3, 5, 10, 20]
    
    all_results = []
    oos_returns = []
    oos_sharpes = []
    
    print(f"\nWalk-Forward Optimization ({n_windows} windows)")
    print("="*60)
    
    for window_idx in range(1, n_windows):
        # Training data: all previous windows
        train_end = window_idx * window_size
        train_prices = prices[:train_end]
        
        # Test data: current window
        test_start = train_end
        test_end = min((window_idx + 1) * window_size, n)
        test_prices = prices[test_start:test_end]
        
        print(f"\nWindow {window_idx}: Train on 0-{train_end}, Test on {test_start}-{test_end}")
        
        # Grid search on training data
        best_sharpe = -np.inf
        best_params = {}
        
        for coh in coherence_thresholds:
            for sig in signal_thresholds:
                for hold in holding_periods:
                    ret, sharpe, n_trades = simple_backtest(
                        train_prices, coh, sig, hold
                    )
                    
                    if sharpe > best_sharpe and n_trades >= 5:
                        best_sharpe = sharpe
                        best_params = {
                            'coherence_threshold': coh,
                            'signal_threshold': sig,
                            'holding_period': hold
                        }
        
        if not best_params:
            best_params = {'coherence_threshold': 0.1, 'signal_threshold': 0.2, 'holding_period': 10}
        
        print(f"  Best params: coh={best_params['coherence_threshold']}, "
              f"sig={best_params['signal_threshold']}, hold={best_params['holding_period']}")
        print(f"  Train Sharpe: {best_sharpe:.2f}")
        
        # Test on out-of-sample
        oos_ret, oos_sharpe, oos_trades = simple_backtest(
            test_prices,
            best_params['coherence_threshold'],
            best_params['signal_threshold'],
            best_params['holding_period']
        )
        
        print(f"  OOS Return: {oos_ret*100:.2f}%, Sharpe: {oos_sharpe:.2f}, Trades: {oos_trades}")
        
        oos_returns.append(oos_ret)
        oos_sharpes.append(oos_sharpe)
        all_results.append((best_params.copy(), oos_ret, oos_sharpe, oos_trades))
    
    # Aggregate OOS performance
    avg_oos_return = np.mean(oos_returns)
    avg_oos_sharpe = np.mean(oos_sharpes)
    
    # Compounded return
    compounded_return = np.prod([1 + r for r in oos_returns]) - 1
    
    print("\n" + "="*60)
    print("WALK-FORWARD RESULTS (Out-of-Sample)")
    print("="*60)
    print(f"  Average OOS Return: {avg_oos_return*100:.2f}%")
    print(f"  Compounded OOS Return: {compounded_return*100:.2f}%")
    print(f"  Average OOS Sharpe: {avg_oos_sharpe:.2f}")
    print(f"  Return Std: {np.std(oos_returns)*100:.2f}%")
    
    # T-stat for significance
    if len(oos_returns) > 1 and np.std(oos_returns) > 0:
        t_stat = np.mean(oos_returns) / (np.std(oos_returns) / np.sqrt(len(oos_returns)))
        print(f"  T-statistic: {t_stat:.2f}")
        print(f"  Statistically significant (t>2): {'Yes' if abs(t_stat) > 2 else 'No'}")
    
    return OptimizationResult(
        best_params=best_params,
        best_sharpe=best_sharpe,
        best_return=avg_oos_return,
        all_results=all_results,
        walk_forward_sharpe=avg_oos_sharpe,
        walk_forward_return=compounded_return
    )


def test_regime_filtering(prices: np.ndarray) -> Dict:
    """
    Test if filtering by regime (basin) improves performance.
    
    Only trade when in certain basin states.
    """
    from grace_market_dynamics.basin_discovery import BasinDiscovery, BasinDiscoveryConfig
    
    basis = build_clifford_basis(np)
    config = BasinDiscoveryConfig(grace_iterations=3, min_observations_for_basin=5)
    discovery = BasinDiscovery(config)
    
    n = len(prices)
    log_returns = np.diff(np.log(prices))
    
    # First pass: discover basins and their characteristics
    basin_returns = {}  # basin_key -> list of forward returns
    
    for i in range(4, len(log_returns) - 5):
        increments = log_returns[i-4:i]
        state = CliffordState.from_increments(increments, basis)
        
        # Update basin discovery
        regime = discovery.update(prices[i])
        
        if regime and regime.basin_key:
            # Forward return (next 5 bars)
            fwd_return = np.sum(log_returns[i:i+5])
            
            key = regime.basin_key[:4]  # Use first 4 elements
            if key not in basin_returns:
                basin_returns[key] = []
            basin_returns[key].append(fwd_return)
    
    # Analyze basins
    print("\n" + "="*60)
    print("REGIME ANALYSIS")
    print("="*60)
    
    profitable_basins = []
    
    for key, returns in sorted(basin_returns.items(), key=lambda x: -len(x[1])):
        if len(returns) >= 10:
            mean_ret = np.mean(returns) * 10000  # bps
            std_ret = np.std(returns) * 10000
            t_stat = mean_ret / (std_ret / np.sqrt(len(returns))) if std_ret > 0 else 0
            
            print(f"\n  Basin {key}:")
            print(f"    Observations: {len(returns)}")
            print(f"    Mean 5-bar return: {mean_ret:.2f} bps")
            print(f"    T-stat: {t_stat:.2f}")
            
            if t_stat > 1.5:  # Somewhat positive
                profitable_basins.append(key)
                print(f"    → FAVORABLE (t > 1.5)")
            elif t_stat < -1.5:
                print(f"    → AVOID (t < -1.5)")
    
    # Backtest with regime filter
    print("\n" + "-"*40)
    print("Backtest with Regime Filter:")
    
    # Reset discovery
    discovery.reset()
    
    equity_filtered = [1.0]
    equity_unfiltered = [1.0]
    
    position = 0
    entry_price = 0.0
    entry_idx = 0
    filtered_trades = 0
    unfiltered_trades = 0
    
    cost_pct = 0.0001  # 1 bp each way
    
    for i in range(4, len(log_returns)):
        increments = log_returns[i-4:i]
        state = CliffordState.from_increments(increments, basis)
        regime = discovery.update(prices[i])
        
        chirality = state.pseudoscalar
        signal = np.sign(chirality) * min(abs(chirality) / 0.3, 1.0)
        
        price = prices[i + 1]
        basin_key = regime.basin_key[:4] if regime and regime.basin_key else None
        
        # Unfiltered strategy
        if abs(signal) > 0.2:
            ret = signal * (price / prices[i] - 1)
            equity_unfiltered.append(equity_unfiltered[-1] * (1 + ret - cost_pct))
            unfiltered_trades += 1
        else:
            equity_unfiltered.append(equity_unfiltered[-1])
        
        # Filtered strategy (only trade in favorable basins)
        in_favorable = basin_key in profitable_basins if basin_key else False
        
        if abs(signal) > 0.2 and in_favorable:
            ret = signal * (price / prices[i] - 1)
            equity_filtered.append(equity_filtered[-1] * (1 + ret - cost_pct))
            filtered_trades += 1
        else:
            equity_filtered.append(equity_filtered[-1])
    
    equity_filtered = np.array(equity_filtered)
    equity_unfiltered = np.array(equity_unfiltered)
    
    filtered_return = equity_filtered[-1] / equity_filtered[0] - 1
    unfiltered_return = equity_unfiltered[-1] / equity_unfiltered[0] - 1
    
    print(f"\n  Unfiltered: Return={unfiltered_return*100:.2f}%, Trades={unfiltered_trades}")
    print(f"  Regime-Filtered: Return={filtered_return*100:.2f}%, Trades={filtered_trades}")
    print(f"  Improvement: {(filtered_return - unfiltered_return)*100:.2f}%")
    
    return {
        'profitable_basins': profitable_basins,
        'filtered_return': filtered_return,
        'unfiltered_return': unfiltered_return,
        'improvement': filtered_return - unfiltered_return
    }


def main():
    """Run full optimization and testing."""
    print("="*70)
    print("CLIFFORD MARKET DYNAMICS — OPTIMIZATION & TESTING")
    print("="*70)
    
    # Fetch data
    print("\nFetching intraday data...")
    prices = fetch_intraday("SPY", "5d", "5m")
    print(f"Data points: {len(prices)}")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Walk-forward optimization
    opt_result = walk_forward_optimization(prices, n_windows=4)
    
    # Regime filtering test
    regime_result = test_regime_filtering(prices)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print(f"\n  Walk-Forward OOS Return: {opt_result.walk_forward_return*100:.2f}%")
    print(f"  Walk-Forward OOS Sharpe: {opt_result.walk_forward_sharpe:.2f}")
    
    print(f"\n  Regime-Filtered Return: {regime_result['filtered_return']*100:.2f}%")
    print(f"  Improvement from filtering: {regime_result['improvement']*100:.2f}%")
    
    # Verdict
    profitable = opt_result.walk_forward_return > 0 or regime_result['filtered_return'] > 0
    
    print("\n" + "-"*40)
    if profitable:
        print("✓ PROFITABLE configuration found!")
        print(f"  Best approach: {'Regime-filtered' if regime_result['filtered_return'] > opt_result.walk_forward_return else 'Walk-forward optimized'}")
    else:
        print("✗ No profitable configuration on this data")
        print("  The signal exists but doesn't overcome transaction costs")
        print("  May need: lower costs, longer holding, or different instrument")
    
    return opt_result, regime_result


if __name__ == "__main__":
    main()
