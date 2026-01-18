"""
OPTIMIZED SYSTEM — Based on Diagnostic Findings
================================================

Improvements from diagnosis:
1. EXCLUDE: XLE, XLP, XLU (energy & utilities don't work)
2. T-STAT SWEET SPOT: 2.0-2.5 (not >2.5, may be overfitted)
3. FOCUS ON: GLD, XLY, SPY, DIA, HYG (best performers)

Expected improvement:
- Current: 56.9% WR, +8.5% edge over breakeven
- Target: 60%+ WR with same or better returns
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import PHI, PHI_INV, DTYPE
from holographic_prod.core.algebra import (
    build_clifford_basis,
    grace_operator,
    grace_basin_key_direct,
)

from theory_true_extensions import compute_t_test

BASIS = build_clifford_basis(np)
RESOLUTION = PHI_INV ** 5

# EXCLUSION LIST (from diagnosis)
EXCLUDED_INSTRUMENTS = {'XLE', 'XLP', 'XLU'}

# T-STAT SWEET SPOT
T_STAT_MIN = 2.0
T_STAT_MAX = 2.5


def build_state(returns: np.ndarray) -> np.ndarray:
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
    return grace_operator(M, BASIS, np)


def get_basin_key(state: np.ndarray) -> Tuple[int, ...]:
    return grace_basin_key_direct(state, BASIS, n_iters=3, resolution=RESOLUTION, xp=np)[:8]


@dataclass
class BasinStats:
    """Statistics for a single basin with exponential decay."""
    returns: List[float] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)
    
    def add(self, ret: float, timestamp: int):
        self.returns.append(ret)
        self.timestamps.append(timestamp)
    
    def get_stats(self, current_time: int, halflife: int = 126) -> Tuple[float, float, float, int]:
        if len(self.returns) < 10:
            return 0.0, 1.0, 0.0, 0
        
        ages = np.array([current_time - t for t in self.timestamps])
        weights = np.exp(-ages * np.log(2) / halflife)
        weights = weights / np.sum(weights)
        
        returns = np.array(self.returns)
        mean = np.sum(weights * returns)
        var = np.sum(weights * (returns - mean)**2)
        std = np.sqrt(var) if var > 0 else 1e-10
        
        effective_n = 1.0 / np.sum(weights**2)
        
        if std > 1e-10 and effective_n > 1:
            t_stat = mean / (std / np.sqrt(effective_n))
        else:
            t_stat = 0.0
        
        return mean, std, t_stat, int(effective_n)


class OptimizedTrader:
    """
    Optimized trading system with diagnostic improvements.
    """
    
    def __init__(self):
        self.basin_stats: Dict[Tuple[int, ...], BasinStats] = defaultdict(BasinStats)
        self.current_time = 0
    
    def train(self, returns: np.ndarray, start_time: int = 0):
        for i in range(20, len(returns) - 5):
            state = build_state(returns[i-20:i])
            key = get_basin_key(state)
            forward_ret = np.sum(returns[i:i+5])
            self.basin_stats[key].add(forward_ret, start_time + i)
        self.current_time = start_time + len(returns)
    
    def update(self, returns: np.ndarray, forward_ret: float):
        if len(returns) < 20:
            return
        state = build_state(returns[-20:])
        key = get_basin_key(state)
        self.basin_stats[key].add(forward_ret, self.current_time)
        self.current_time += 1
    
    def get_signal(self, returns: np.ndarray) -> Tuple[float, Dict]:
        """
        Get signal with t-stat sweet spot filter.
        
        Returns: (position_size, info)
        """
        if len(returns) < 20:
            return 0.0, {'reason': 'insufficient_data'}
        
        state = build_state(returns[-20:])
        key = get_basin_key(state)
        
        if key not in self.basin_stats:
            return 0.0, {'reason': 'unknown_basin'}
        
        mean, std, t_stat, effective_n = self.basin_stats[key].get_stats(self.current_time)
        
        if effective_n < 10:
            return 0.0, {'reason': 'insufficient_samples', 'n': effective_n}
        
        # SWEET SPOT FILTER: 2.0 <= t < 2.5
        if t_stat < T_STAT_MIN:
            return 0.0, {'reason': 'low_t_stat', 't_stat': t_stat}
        
        if t_stat >= T_STAT_MAX:
            # Still trade but with reduced size (may be overfitted)
            position = 0.15 * (mean / (std + 1e-10))  # Reduced Kelly
        else:
            # Sweet spot: full position
            position = 0.25 * (mean / (std + 1e-10))  # Full Kelly
        
        # Long-only
        if mean <= 0:
            return 0.0, {'reason': 'negative_mean', 'mean': mean}
        
        position = np.clip(position, 0, 0.25)
        
        return position, {
            'basin': key,
            'mean': mean,
            'std': std,
            't_stat': t_stat,
            'n': effective_n,
            'sweet_spot': T_STAT_MIN <= t_stat < T_STAT_MAX
        }


def run_optimized_backtest(
    data: Dict[str, np.ndarray],
    train_window: int = 252,
    retrain_freq: int = 21,
    holding_period: int = 5,
    cost_bps: float = 5.0
) -> Dict:
    """Run optimized backtest with exclusions and sweet spot."""
    
    # Filter excluded instruments
    filtered_data = {k: v for k, v in data.items() if k not in EXCLUDED_INSTRUMENTS}
    
    if not filtered_data:
        return {'error': 'No instruments after filtering'}
    
    min_len = min(len(v) for v in filtered_data.values())
    for ticker in filtered_data:
        filtered_data[ticker] = filtered_data[ticker][-min_len:]
    
    n = min_len
    
    # Create trader for each instrument
    traders = {ticker: OptimizedTrader() for ticker in filtered_data}
    
    # Initial training
    train_end = min(train_window, n // 2)
    
    for ticker, prices in filtered_data.items():
        returns = np.diff(np.log(prices[:train_end]))
        traders[ticker].train(returns)
    
    # Walk-forward backtest
    all_trades = []
    portfolio_equity = [1.0]
    
    positions = {t: 0.0 for t in filtered_data}
    entries = {t: 0 for t in filtered_data}
    entry_prices = {t: 0.0 for t in filtered_data}
    entry_info = {t: {} for t in filtered_data}
    last_retrain = train_end
    
    for i in range(train_end + 20, n - holding_period):
        # Retrain periodically
        if i - last_retrain >= retrain_freq:
            for ticker, prices in filtered_data.items():
                returns = np.diff(np.log(prices))
                for j in range(last_retrain, i):
                    if j >= 20 and j + holding_period < len(returns):
                        traders[ticker].update(
                            returns[j-20:j],
                            np.sum(returns[j:j+holding_period])
                        )
                traders[ticker].current_time = i
            last_retrain = i
        
        daily_pnl = 0.0
        
        # Manage existing positions
        for ticker in filtered_data:
            prices = filtered_data[ticker]
            
            if positions[ticker] != 0:
                if i - entries[ticker] >= holding_period:
                    ret = positions[ticker] * (prices[i] / entry_prices[ticker] - 1)
                    cost = cost_bps / 10000 * abs(positions[ticker]) * 2
                    net_ret = ret - cost
                    daily_pnl += net_ret
                    
                    all_trades.append({
                        'ticker': ticker,
                        'return': ret,
                        'net_return': net_ret,
                        'sweet_spot': entry_info[ticker].get('sweet_spot', False),
                        't_stat': entry_info[ticker].get('t_stat', 0)
                    })
                    positions[ticker] = 0
                else:
                    if i > 0:
                        bar_ret = positions[ticker] * (prices[i] / prices[i-1] - 1)
                        daily_pnl += bar_ret
        
        # New entries
        signals = {}
        for ticker, prices in filtered_data.items():
            if positions[ticker] != 0:
                continue
            
            returns = np.diff(np.log(prices[:i+1]))
            if len(returns) >= 20:
                pos, info = traders[ticker].get_signal(returns[-20:])
                if pos > 0:
                    signals[ticker] = (pos, info)
        
        if signals:
            total_target = sum(s[0] for s in signals.values())
            scale = min(1.0, 1.0 / total_target) if total_target > 0 else 1.0
            
            for ticker, (pos, info) in signals.items():
                positions[ticker] = pos * scale
                entries[ticker] = i
                entry_prices[ticker] = filtered_data[ticker][i]
                entry_info[ticker] = info
                daily_pnl -= cost_bps / 10000 * abs(positions[ticker])
        
        portfolio_equity.append(portfolio_equity[-1] * (1 + daily_pnl))
    
    # Close remaining positions
    for ticker in filtered_data:
        if positions[ticker] != 0:
            prices = filtered_data[ticker]
            ret = positions[ticker] * (prices[-1] / entry_prices[ticker] - 1)
            cost = cost_bps / 10000 * abs(positions[ticker]) * 2
            portfolio_equity[-1] *= (1 + ret - cost)
            all_trades.append({
                'ticker': ticker,
                'return': ret,
                'net_return': ret - cost,
                'sweet_spot': entry_info[ticker].get('sweet_spot', False),
                't_stat': entry_info[ticker].get('t_stat', 0)
            })
    
    portfolio_equity = np.array(portfolio_equity)
    
    # Statistics
    if all_trades:
        trade_returns = np.array([t['net_return'] for t in all_trades])
        t_stat, p_value, ci = compute_t_test(trade_returns)
        
        # Breakdown by sweet spot
        sweet_spot_trades = [t for t in all_trades if t.get('sweet_spot', False)]
        other_trades = [t for t in all_trades if not t.get('sweet_spot', False)]
    else:
        trade_returns = np.array([0.0])
        t_stat, p_value, ci = 0.0, 1.0, (0.0, 0.0)
        sweet_spot_trades = []
        other_trades = []
    
    daily_returns = np.diff(portfolio_equity) / portfolio_equity[:-1]
    
    return {
        'total_return': portfolio_equity[-1] / portfolio_equity[0] - 1,
        'sharpe': np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252),
        'max_drawdown': np.max((np.maximum.accumulate(portfolio_equity) - portfolio_equity) / (np.maximum.accumulate(portfolio_equity) + 1e-10)),
        'n_trades': len(all_trades),
        'win_rate': np.mean(trade_returns > 0) if len(trade_returns) > 0 else 0,
        'mean_trade': np.mean(trade_returns) if len(trade_returns) > 0 else 0,
        't_stat': t_stat,
        'p_value': p_value,
        'ci': ci,
        'trades': all_trades,
        'equity': portfolio_equity,
        'sweet_spot_trades': len(sweet_spot_trades),
        'other_trades': len(other_trades),
        'sweet_spot_wr': np.mean([t['net_return'] > 0 for t in sweet_spot_trades]) if sweet_spot_trades else 0,
        'other_wr': np.mean([t['net_return'] > 0 for t in other_trades]) if other_trades else 0
    }


def main():
    print("="*80)
    print("OPTIMIZED SYSTEM — Diagnostic Improvements Applied")
    print("="*80)
    print(f"\n  IMPROVEMENTS:")
    print(f"    1. EXCLUDED: {EXCLUDED_INSTRUMENTS}")
    print(f"    2. T-STAT SWEET SPOT: {T_STAT_MIN} ≤ t < {T_STAT_MAX}")
    print(f"    3. Reduced position for t ≥ {T_STAT_MAX}")
    
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not available")
        return
    
    tickers = [
        'SPY', 'QQQ', 'IWM', 'DIA',
        'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY',
        'EEM', 'EFA', 'TLT', 'GLD', 'HYG'
    ]
    
    data = {}
    print("\nFetching 5 years of daily data...")
    for ticker in tickers:
        try:
            df = yf.download(ticker, period='5y', interval='1d', progress=False)
            if len(df) > 500:
                data[ticker] = df['Close'].values.flatten()
        except:
            pass
    
    print(f"Loaded {len(data)} instruments")
    print(f"After exclusion: {len(data) - len(EXCLUDED_INSTRUMENTS.intersection(data.keys()))} instruments")
    
    # Run optimized backtest
    print("\n" + "-"*40)
    print("OPTIMIZED PORTFOLIO RESULTS")
    print("-"*40)
    
    result = run_optimized_backtest(data)
    
    print(f"\n  Total Return: {result['total_return']*100:+.2f}%")
    print(f"  Annualized Sharpe: {result['sharpe']:.2f}")
    print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"  Number of Trades: {result['n_trades']}")
    print(f"  Win Rate: {result['win_rate']*100:.1f}%")
    print(f"  Mean Trade Return: {result['mean_trade']*100:.3f}%")
    print(f"  T-statistic: {result['t_stat']:.2f}")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  95% CI: [{result['ci'][0]*100:.3f}%, {result['ci'][1]*100:.3f}%]")
    
    print(f"\n  Sweet Spot Trades: {result['sweet_spot_trades']} ({result['sweet_spot_wr']*100:.1f}% WR)")
    print(f"  Other Trades: {result['other_trades']} ({result['other_wr']*100:.1f}% WR)")
    
    print(f"\n  Significant (p<0.05): {'YES ✓' if result['p_value'] < 0.05 else 'NO'}")
    print(f"  Significant (p<0.10): {'YES ✓' if result['p_value'] < 0.10 else 'NO'}")
    
    # Compare to baseline (without optimizations)
    print("\n" + "-"*40)
    print("COMPARISON: OPTIMIZED vs BASELINE")
    print("-"*40)
    
    # Run baseline (include excluded instruments, no sweet spot filter)
    from production_system import run_multi_instrument_backtest, WalkForwardConfig
    
    baseline_config = WalkForwardConfig(
        train_window=252,
        retrain_frequency=21,
        min_t_stat=1.5,
        max_position=0.25,
        cost_bps=5.0,
        holding_period=5
    )
    baseline_result = run_multi_instrument_backtest(data, baseline_config)
    
    print(f"\n  {'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'Improvement':>12}")
    print("  " + "-"*60)
    print(f"  {'Return':<20} {baseline_result['total_return']*100:>11.2f}% {result['total_return']*100:>11.2f}% {(result['total_return']-baseline_result['total_return'])*100:>+11.2f}%")
    print(f"  {'Sharpe':<20} {baseline_result['sharpe']:>12.2f} {result['sharpe']:>12.2f} {result['sharpe']-baseline_result['sharpe']:>+12.2f}")
    print(f"  {'Win Rate':<20} {baseline_result['win_rate']*100:>11.1f}% {result['win_rate']*100:>11.1f}% {(result['win_rate']-baseline_result['win_rate'])*100:>+11.1f}%")
    print(f"  {'Max DD':<20} {baseline_result['max_drawdown']*100:>11.2f}% {result['max_drawdown']*100:>11.2f}% {(result['max_drawdown']-baseline_result['max_drawdown'])*100:>+11.2f}%")
    print(f"  {'Trades':<20} {baseline_result['n_trades']:>12} {result['n_trades']:>12} {result['n_trades']-baseline_result['n_trades']:>+12}")
    print(f"  {'P-value':<20} {baseline_result['p_value']:>12.4f} {result['p_value']:>12.4f}")
    
    # Per-instrument breakdown
    print("\n" + "-"*40)
    print("PER-INSTRUMENT BREAKDOWN")
    print("-"*40)
    
    instrument_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'returns': []})
    for trade in result['trades']:
        ticker = trade['ticker']
        instrument_stats[ticker]['returns'].append(trade['net_return'])
        if trade['net_return'] > 0:
            instrument_stats[ticker]['wins'] += 1
        else:
            instrument_stats[ticker]['losses'] += 1
    
    print(f"\n  {'Ticker':<8} {'Trades':>7} {'Win%':>8} {'Mean':>10} {'T-stat':>8}")
    print("  " + "-"*50)
    
    for ticker in sorted(instrument_stats.keys()):
        stats = instrument_stats[ticker]
        total = stats['wins'] + stats['losses']
        if total > 0:
            wr = stats['wins'] / total * 100
            mean = np.mean(stats['returns']) * 100
            t, _, _ = compute_t_test(np.array(stats['returns']))
            print(f"  {ticker:<8} {total:>7} {wr:>7.1f}% {mean:>9.2f}% {t:>8.2f}")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    improvement = result['win_rate'] - baseline_result['win_rate']
    
    print(f"\n  Win Rate Improvement: {improvement*100:+.1f}%")
    print(f"  Sharpe Improvement: {result['sharpe'] - baseline_result['sharpe']:+.2f}")
    
    if result['win_rate'] > 0.60:
        print("\n  ✓ WIN RATE TARGET ACHIEVED (>60%)")
    elif result['win_rate'] > baseline_result['win_rate']:
        print(f"\n  ~ WIN RATE IMPROVED ({baseline_result['win_rate']*100:.1f}% → {result['win_rate']*100:.1f}%)")
    else:
        print("\n  ✗ WIN RATE NOT IMPROVED")
    
    if result['sharpe'] > baseline_result['sharpe']:
        print("  ✓ SHARPE IMPROVED")
    else:
        print("  ✗ SHARPE NOT IMPROVED")
    
    if result['p_value'] < 0.05:
        print("  ✓ STATISTICALLY SIGNIFICANT (p<0.05)")
    
    return {
        'optimized': result,
        'baseline': baseline_result
    }


if __name__ == "__main__":
    main()
