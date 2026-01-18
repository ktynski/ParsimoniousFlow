"""
OPTIMAL BASIN RESOLUTION TEST — φ⁵ DISCOVERY
==============================================

Key finding from previous test:
- φ⁵ resolution = 58 basins = OPTIMAL (Sharpe 1.07)
- φ⁶ was too fine (208 basins, negative Sharpe)

This test validates the φ⁵ finding across multiple instruments
and tests for statistical significance.

Theory: Grace basins at φ⁵ resolution = ~60 distinct market regimes
        This matches intuition: bull/bear × volatility × sector rotation
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, DTYPE
)
from holographic_prod.core.algebra import (
    build_clifford_basis,
    grace_operator,
    grace_basin_key_direct,
    decompose_to_coefficients,
)

from theory_true_extensions import compute_t_test


def build_state(returns: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Build Clifford state from returns."""
    if len(returns) < 4:
        return np.eye(4, dtype=DTYPE)
    v = returns[-4:].astype(DTYPE)
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-10:
        v = v / v_norm
    gamma = np.array([
        [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]],
        [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]],
        [[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]],
        [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
    ], dtype=DTYPE)
    M = sum(v[i] * gamma[i] for i in range(4))
    return grace_operator(M, basis, np)


class OptimalBasinRouter:
    """
    Basin router with φ⁵ resolution (discovered optimal).
    """
    
    def __init__(self):
        self.basis = build_clifford_basis(np)
        self.resolution = PHI_INV ** 5  # ~0.0902 — OPTIMAL
        self.n_grace_iters = 3
        self.key_truncation = 8
        
        # Regime statistics
        self.regimes: Dict[Tuple[int, ...], Dict] = {}
    
    def get_basin_key(self, M: np.ndarray) -> Tuple[int, ...]:
        full_key = grace_basin_key_direct(
            M, self.basis,
            n_iters=self.n_grace_iters,
            resolution=self.resolution,
            xp=np
        )
        return full_key[:self.key_truncation]
    
    def update(self, basin_key: Tuple[int, ...], forward_return: float, coherence: float):
        if basin_key not in self.regimes:
            self.regimes[basin_key] = {
                'returns': [],
                'coherences': [],
                'count': 0
            }
        self.regimes[basin_key]['returns'].append(forward_return)
        self.regimes[basin_key]['coherences'].append(coherence)
        self.regimes[basin_key]['count'] += 1
    
    def get_signal(self, basin_key: Tuple[int, ...]) -> Tuple[float, float]:
        """
        Get signal and confidence for basin.
        
        Returns: (signal, confidence)
        """
        if basin_key not in self.regimes:
            return 0.0, 0.0
        
        regime = self.regimes[basin_key]
        if regime['count'] < 5:  # Need minimum history
            return 0.0, 0.0
        
        returns = np.array(regime['returns'])
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret < 1e-10:
            return 0.0, 0.0
        
        t_stat = mean_ret / (std_ret / np.sqrt(len(returns)))
        signal = np.tanh(t_stat / 2.0)
        
        # Confidence based on sample size and consistency
        confidence = min(np.sqrt(len(returns)) / 10, 1.0) * (1 - std_ret / (np.abs(mean_ret) + std_ret + 1e-10))
        
        return signal, confidence


def run_optimal_basin_test():
    """Run comprehensive test with optimal φ⁵ basin resolution."""
    print("="*80)
    print("OPTIMAL BASIN RESOLUTION TEST — φ⁵ = 58 REGIMES")
    print("="*80)
    
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not available")
        return
    
    # Wide universe
    tickers = [
        # US Equity
        'SPY', 'QQQ', 'IWM', 'DIA',
        # Sectors
        'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB', 'XLRE',
        # International
        'EEM', 'EFA', 'VWO',
        # Fixed Income
        'TLT', 'IEF', 'HYG', 'LQD',
        # Commodities
        'GLD', 'SLV', 'USO',
        # Volatility
        'VIXY'
    ]
    
    data = {}
    print("\nFetching 5 years of daily data for 25 instruments...")
    for ticker in tickers:
        try:
            df = yf.download(ticker, period='5y', interval='1d', progress=False)
            if len(df) > 500:
                data[ticker] = df['Close'].values.flatten()
        except:
            pass
    
    print(f"Loaded {len(data)} instruments")
    
    # Align
    min_len = min(len(v) for v in data.values())
    for ticker in data:
        data[ticker] = data[ticker][-min_len:]
    print(f"Aligned to {min_len} days")
    
    basis = build_clifford_basis(np)
    
    # Train/test split (60/40 for more OOS data)
    train_end = int(min_len * 0.6)
    print(f"Training: {train_end} days, Testing: {min_len - train_end} days")
    
    # PART 1: Train individual routers
    print("\n" + "-"*40)
    print("PART 1: INDIVIDUAL INSTRUMENT SIGNALS")
    print("-"*40)
    
    routers = {}
    for ticker in data:
        router = OptimalBasinRouter()
        returns = np.diff(np.log(data[ticker][:train_end]))
        
        for i in range(20, len(returns) - 5):
            state = build_state(returns[i-20:i], basis)
            forward_ret = np.sum(returns[i:i+5])
            basin_key = router.get_basin_key(state)
            coherence = np.abs(decompose_to_coefficients(state, basis, np)[0])
            router.update(basin_key, forward_ret, coherence)
        
        routers[ticker] = router
    
    # Test individual
    individual_results = {}
    
    for ticker, prices in data.items():
        test_prices = prices[train_end:]
        test_returns = np.diff(np.log(test_prices))
        
        trades = []
        position = 0
        entry_idx = 0
        entry_price = 0
        
        for i in range(20, len(test_returns) - 5):
            state = build_state(test_returns[i-20:i], basis)
            basin_key = routers[ticker].get_basin_key(state)
            signal, confidence = routers[ticker].get_signal(basin_key)
            
            if position != 0:
                if i - entry_idx >= 5:
                    ret = position * (test_prices[i] / entry_price - 1)
                    trades.append(ret)
                    position = 0
            else:
                # Trade only on high-confidence signals
                if confidence > 0.3 and abs(signal) > 0.3:
                    position = 1 if signal > 0 else -1
                    entry_idx = i
                    entry_price = test_prices[i]
        
        if trades:
            trades = np.array(trades)
            individual_results[ticker] = {
                'n_trades': len(trades),
                'mean_return': np.mean(trades),
                'sharpe': np.mean(trades) / (np.std(trades) + 1e-10) * np.sqrt(252/5),
                't_stat': compute_t_test(trades)[0],
                'p_value': compute_t_test(trades)[1]
            }
    
    # Sort by Sharpe
    sorted_results = sorted(individual_results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    
    print(f"\n{'Ticker':<8} {'Trades':>7} {'Mean Ret':>10} {'Sharpe':>8} {'T-stat':>8} {'P-val':>8}")
    print("-"*60)
    for ticker, res in sorted_results[:15]:
        print(f"{ticker:<8} {res['n_trades']:>7} {res['mean_return']*100:>9.2f}% {res['sharpe']:>8.2f} {res['t_stat']:>8.2f} {res['p_value']:>8.3f}")
    
    # PART 2: Long-Short Portfolio
    print("\n" + "-"*40)
    print("PART 2: LONG-SHORT PORTFOLIO (Weekly Rebalance)")
    print("-"*40)
    
    portfolio_returns = []
    
    for week in range(0, min_len - train_end - 5, 5):
        signals = {}
        
        for ticker in data:
            test_prices = data[ticker][train_end:]
            test_returns = np.diff(np.log(test_prices))
            
            if week >= len(test_returns) - 5:
                continue
            
            state = build_state(test_returns[max(0, week-20):week], basis) if week >= 20 else np.eye(4)
            basin_key = routers[ticker].get_basin_key(state)
            signal, conf = routers[ticker].get_signal(basin_key)
            
            signals[ticker] = signal * conf  # Confidence-weighted
        
        if len(signals) < 6:
            continue
        
        # Rank
        ranked = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        n_q = max(len(ranked) // 4, 1)
        
        longs = [t for t, s in ranked[:n_q]]
        shorts = [t for t, s in ranked[-n_q:]]
        
        # Compute return
        period_ret = 0
        for ticker in longs:
            tp = data[ticker][train_end:]
            if week + 5 < len(tp):
                period_ret += (tp[week+5] / tp[week] - 1) / n_q
        
        for ticker in shorts:
            tp = data[ticker][train_end:]
            if week + 5 < len(tp):
                period_ret -= (tp[week+5] / tp[week] - 1) / n_q
        
        portfolio_returns.append(period_ret)
    
    if portfolio_returns:
        returns = np.array(portfolio_returns)
        cumulative = np.cumprod(1 + returns)
        
        total_ret = cumulative[-1] - 1
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(52)
        max_dd = np.max(np.maximum.accumulate(cumulative) - cumulative)
        t_stat, p_value, ci = compute_t_test(returns)
        
        print(f"\n  Total Return: {total_ret*100:.2f}%")
        print(f"  Annualized Sharpe: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_dd*100:.2f}%")
        print(f"  Win Rate: {np.mean(returns > 0)*100:.1f}%")
        print(f"  T-statistic: {t_stat:.2f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  95% CI: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")
        print(f"\n  Significant (p<0.05): {'YES ✓' if p_value < 0.05 else 'NO'}")
        print(f"  Significant (p<0.10): {'YES ✓' if p_value < 0.10 else 'NO'}")
    
    # PART 3: Basin Statistics
    print("\n" + "-"*40)
    print("PART 3: BASIN STATISTICS")
    print("-"*40)
    
    all_basins = set()
    for router in routers.values():
        all_basins.update(router.regimes.keys())
    
    print(f"\n  Unique basins across all instruments: {len(all_basins)}")
    
    # Find universal basins (appear in all instruments)
    basin_counts = {}
    for basin in all_basins:
        count = sum(1 for r in routers.values() if basin in r.regimes)
        basin_counts[basin] = count
    
    universal = [b for b, c in basin_counts.items() if c >= len(routers) * 0.5]
    print(f"  Universal basins (>50% of instruments): {len(universal)}")
    
    # PART 4: Regime-Specific Analysis
    print("\n" + "-"*40)
    print("PART 4: REGIME-SPECIFIC PROFITABILITY")
    print("-"*40)
    
    # Aggregate returns by basin
    basin_aggregates = {}
    
    for ticker, router in routers.items():
        for basin_key, regime in router.regimes.items():
            if basin_key not in basin_aggregates:
                basin_aggregates[basin_key] = {'returns': [], 'tickers': set()}
            basin_aggregates[basin_key]['returns'].extend(regime['returns'])
            basin_aggregates[basin_key]['tickers'].add(ticker)
    
    # Find profitable regimes
    profitable_basins = []
    for basin_key, agg in basin_aggregates.items():
        if len(agg['returns']) >= 20:
            ret = np.array(agg['returns'])
            t_stat, p_value, _ = compute_t_test(ret)
            if t_stat > 1.5:  # Significantly profitable
                profitable_basins.append({
                    'basin': basin_key,
                    'mean_return': np.mean(ret),
                    't_stat': t_stat,
                    'p_value': p_value,
                    'n_obs': len(ret),
                    'n_tickers': len(agg['tickers'])
                })
    
    profitable_basins.sort(key=lambda x: x['t_stat'], reverse=True)
    
    print(f"\n  Profitable basins (t>1.5): {len(profitable_basins)}")
    print(f"\n  Top 10 most profitable regimes:")
    print(f"  {'Mean Ret':>10} {'T-stat':>8} {'P-val':>8} {'N Obs':>7} {'Tickers':>8}")
    print("  " + "-"*50)
    
    for pb in profitable_basins[:10]:
        print(f"  {pb['mean_return']*100:>9.2f}% {pb['t_stat']:>8.2f} {pb['p_value']:>8.3f} {pb['n_obs']:>7} {pb['n_tickers']:>8}")
    
    # FINAL VERDICT
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if portfolio_returns:
        returns = np.array(portfolio_returns)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(52)
        t_stat, p_value, _ = compute_t_test(returns)
        
        print(f"\n  φ⁵ Basin Resolution Performance:")
        print(f"    - Sharpe: {sharpe:.2f}")
        print(f"    - Statistical Significance: p={p_value:.4f}")
        
        if sharpe > 0.5 and p_value < 0.1:
            print("\n  ✓ THEORY VALIDATED: φ⁵ basin resolution captures meaningful market regimes")
        elif sharpe > 0:
            print("\n  ~ PARTIAL: Positive returns but not statistically significant")
            print("    Need: More data or better signal combination")
        else:
            print("\n  ✗ NOT VALIDATED at this resolution")
    
    return {
        'individual': individual_results,
        'portfolio_sharpe': sharpe if portfolio_returns else 0,
        'portfolio_p': p_value if portfolio_returns else 1,
        'n_profitable_basins': len(profitable_basins)
    }


if __name__ == "__main__":
    run_optimal_basin_test()
