"""
HYBRID SYSTEM — Keep All Instruments, Size by Confidence
=========================================================

Finding: Exclusions hurt diversification.
New approach: Keep all, size positions by t-stat band.

Position sizing:
  - t < 1.5: No trade
  - 1.5 ≤ t < 2.0: Small position (10%)
  - 2.0 ≤ t < 2.5: Medium position (20%)  
  - t ≥ 2.5: Large position (25%)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
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

# Position sizing by t-stat band
POSITION_TIERS = {
    (1.5, 2.0): 0.10,   # Low confidence
    (2.0, 2.5): 0.20,   # Medium confidence
    (2.5, float('inf')): 0.25,  # High confidence
}


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


def get_position_size(t_stat: float, mean: float, std: float) -> float:
    """Get position size based on t-stat tier."""
    if t_stat < 1.5 or mean <= 0:
        return 0.0
    
    for (t_min, t_max), base_size in POSITION_TIERS.items():
        if t_min <= t_stat < t_max:
            # Kelly adjustment within tier
            kelly = mean / (std ** 2 + 1e-10)
            kelly = min(kelly, 1.0)  # Cap Kelly
            return base_size * kelly * 0.5  # Half-Kelly
    
    return 0.0


@dataclass
class BasinStats:
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


class HybridTrader:
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
    
    def get_signal(self, returns: np.ndarray) -> Tuple[float, str, Dict]:
        if len(returns) < 20:
            return 0.0, 'none', {}
        
        state = build_state(returns[-20:])
        key = get_basin_key(state)
        
        if key not in self.basin_stats:
            return 0.0, 'none', {'reason': 'unknown_basin'}
        
        mean, std, t_stat, effective_n = self.basin_stats[key].get_stats(self.current_time)
        
        if effective_n < 10:
            return 0.0, 'none', {'reason': 'insufficient_samples'}
        
        position = get_position_size(t_stat, mean, std)
        
        # Determine tier
        if t_stat >= 2.5:
            tier = 'high'
        elif t_stat >= 2.0:
            tier = 'medium'
        elif t_stat >= 1.5:
            tier = 'low'
        else:
            tier = 'none'
        
        return position, tier, {
            'mean': mean,
            'std': std,
            't_stat': t_stat,
            'n': effective_n
        }


def run_hybrid_backtest(
    data: Dict[str, np.ndarray],
    train_window: int = 252,
    retrain_freq: int = 21,
    holding_period: int = 5,
    cost_bps: float = 5.0
) -> Dict:
    """Run hybrid backtest with tiered position sizing."""
    
    min_len = min(len(v) for v in data.values())
    for ticker in data:
        data[ticker] = data[ticker][-min_len:]
    
    n = min_len
    traders = {ticker: HybridTrader() for ticker in data}
    
    train_end = min(train_window, n // 2)
    
    for ticker, prices in data.items():
        returns = np.diff(np.log(prices[:train_end]))
        traders[ticker].train(returns)
    
    all_trades = []
    portfolio_equity = [1.0]
    
    positions = {t: 0.0 for t in data}
    entries = {t: 0 for t in data}
    entry_prices = {t: 0.0 for t in data}
    entry_tiers = {t: 'none' for t in data}
    last_retrain = train_end
    
    tier_stats = {'low': [], 'medium': [], 'high': []}
    
    for i in range(train_end + 20, n - holding_period):
        if i - last_retrain >= retrain_freq:
            for ticker, prices in data.items():
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
        
        for ticker in data:
            prices = data[ticker]
            
            if positions[ticker] != 0:
                if i - entries[ticker] >= holding_period:
                    ret = positions[ticker] * (prices[i] / entry_prices[ticker] - 1)
                    cost = cost_bps / 10000 * abs(positions[ticker]) * 2
                    net_ret = ret - cost
                    daily_pnl += net_ret
                    
                    tier = entry_tiers[ticker]
                    if tier in tier_stats:
                        tier_stats[tier].append(net_ret)
                    
                    all_trades.append({
                        'ticker': ticker,
                        'return': ret,
                        'net_return': net_ret,
                        'tier': tier
                    })
                    positions[ticker] = 0
                else:
                    if i > 0:
                        bar_ret = positions[ticker] * (prices[i] / prices[i-1] - 1)
                        daily_pnl += bar_ret
        
        signals = {}
        for ticker, prices in data.items():
            if positions[ticker] != 0:
                continue
            
            returns = np.diff(np.log(prices[:i+1]))
            if len(returns) >= 20:
                pos, tier, info = traders[ticker].get_signal(returns[-20:])
                if pos > 0:
                    signals[ticker] = (pos, tier, info)
        
        if signals:
            total_target = sum(s[0] for s in signals.values())
            scale = min(1.0, 1.0 / total_target) if total_target > 0 else 1.0
            
            for ticker, (pos, tier, info) in signals.items():
                positions[ticker] = pos * scale
                entries[ticker] = i
                entry_prices[ticker] = data[ticker][i]
                entry_tiers[ticker] = tier
                daily_pnl -= cost_bps / 10000 * abs(positions[ticker])
        
        portfolio_equity.append(portfolio_equity[-1] * (1 + daily_pnl))
    
    for ticker in data:
        if positions[ticker] != 0:
            prices = data[ticker]
            ret = positions[ticker] * (prices[-1] / entry_prices[ticker] - 1)
            cost = cost_bps / 10000 * abs(positions[ticker]) * 2
            portfolio_equity[-1] *= (1 + ret - cost)
            all_trades.append({
                'ticker': ticker,
                'return': ret,
                'net_return': ret - cost,
                'tier': entry_tiers[ticker]
            })
            if entry_tiers[ticker] in tier_stats:
                tier_stats[entry_tiers[ticker]].append(ret - cost)
    
    portfolio_equity = np.array(portfolio_equity)
    
    if all_trades:
        trade_returns = np.array([t['net_return'] for t in all_trades])
        t_stat, p_value, ci = compute_t_test(trade_returns)
    else:
        trade_returns = np.array([0.0])
        t_stat, p_value, ci = 0.0, 1.0, (0.0, 0.0)
    
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
        'tier_stats': tier_stats
    }


def main():
    print("="*80)
    print("HYBRID SYSTEM — Tiered Position Sizing")
    print("="*80)
    print(f"\n  APPROACH: Keep all instruments, size by t-stat tier")
    print(f"    t < 1.5: No trade")
    print(f"    1.5 ≤ t < 2.0: Small position (10% × Kelly)")
    print(f"    2.0 ≤ t < 2.5: Medium position (20% × Kelly)")
    print(f"    t ≥ 2.5: Large position (25% × Kelly)")
    
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
    
    result = run_hybrid_backtest(data)
    
    print("\n" + "-"*40)
    print("HYBRID PORTFOLIO RESULTS")
    print("-"*40)
    
    print(f"\n  Total Return: {result['total_return']*100:+.2f}%")
    print(f"  Annualized Sharpe: {result['sharpe']:.2f}")
    print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"  Number of Trades: {result['n_trades']}")
    print(f"  Win Rate: {result['win_rate']*100:.1f}%")
    print(f"  Mean Trade Return: {result['mean_trade']*100:.3f}%")
    print(f"  T-statistic: {result['t_stat']:.2f}")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  95% CI: [{result['ci'][0]*100:.3f}%, {result['ci'][1]*100:.3f}%]")
    
    print(f"\n  Significant (p<0.05): {'YES ✓' if result['p_value'] < 0.05 else 'NO'}")
    
    # Tier breakdown
    print("\n" + "-"*40)
    print("PERFORMANCE BY TIER")
    print("-"*40)
    
    print(f"\n  {'Tier':<10} {'Trades':>8} {'Win%':>8} {'Mean':>10} {'EV Contrib':>12}")
    print("  " + "-"*55)
    
    total_ev = 0
    for tier in ['low', 'medium', 'high']:
        trades = result['tier_stats'][tier]
        if trades:
            trades = np.array(trades)
            wr = np.mean(trades > 0) * 100
            mean = np.mean(trades) * 100
            ev_contrib = np.sum(trades) / result['total_return'] * 100 if result['total_return'] != 0 else 0
            total_ev += np.sum(trades)
            print(f"  {tier:<10} {len(trades):>8} {wr:>7.1f}% {mean:>9.3f}% {ev_contrib:>11.1f}%")
    
    # Compare to baseline
    print("\n" + "-"*40)
    print("COMPARISON TO BASELINE")
    print("-"*40)
    
    from production_system import run_multi_instrument_backtest, WalkForwardConfig
    
    baseline_config = WalkForwardConfig(
        train_window=252,
        retrain_frequency=21,
        min_t_stat=1.5,
        max_position=0.25,
        cost_bps=5.0,
        holding_period=5
    )
    baseline = run_multi_instrument_backtest(data, baseline_config)
    
    print(f"\n  {'Metric':<20} {'Baseline':>12} {'Hybrid':>12} {'Change':>12}")
    print("  " + "-"*60)
    print(f"  {'Return':<20} {baseline['total_return']*100:>11.2f}% {result['total_return']*100:>11.2f}% {(result['total_return']-baseline['total_return'])*100:>+11.2f}%")
    print(f"  {'Sharpe':<20} {baseline['sharpe']:>12.2f} {result['sharpe']:>12.2f} {result['sharpe']-baseline['sharpe']:>+12.2f}")
    print(f"  {'Win Rate':<20} {baseline['win_rate']*100:>11.1f}% {result['win_rate']*100:>11.1f}% {(result['win_rate']-baseline['win_rate'])*100:>+11.1f}%")
    print(f"  {'Max DD':<20} {baseline['max_drawdown']*100:>11.2f}% {result['max_drawdown']*100:>11.2f}% {(result['max_drawdown']-baseline['max_drawdown'])*100:>+11.2f}%")
    print(f"  {'Trades':<20} {baseline['n_trades']:>12} {result['n_trades']:>12} {result['n_trades']-baseline['n_trades']:>+12}")
    print(f"  {'P-value':<20} {baseline['p_value']:>12.4f} {result['p_value']:>12.4f}")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    improvements = []
    if result['win_rate'] > baseline['win_rate']:
        improvements.append(f"Win Rate: {baseline['win_rate']*100:.1f}% → {result['win_rate']*100:.1f}%")
    if result['sharpe'] > baseline['sharpe']:
        improvements.append(f"Sharpe: {baseline['sharpe']:.2f} → {result['sharpe']:.2f}")
    if result['max_drawdown'] < baseline['max_drawdown']:
        improvements.append(f"Max DD: {baseline['max_drawdown']*100:.1f}% → {result['max_drawdown']*100:.1f}%")
    
    if improvements:
        print("\n  IMPROVEMENTS:")
        for imp in improvements:
            print(f"    ✓ {imp}")
    
    if not improvements:
        print("\n  NO IMPROVEMENT over baseline")
        print("  Tiered sizing doesn't help - uniform sizing may be optimal")
    
    if result['p_value'] < 0.05:
        print("\n  ✓ STATISTICALLY SIGNIFICANT (p<0.05)")
    
    # Which tier contributes most?
    print("\n  TIER CONTRIBUTION:")
    for tier in ['low', 'medium', 'high']:
        trades = result['tier_stats'][tier]
        if trades:
            ev = np.sum(trades)
            pct = ev / result['total_return'] * 100 if result['total_return'] != 0 else 0
            print(f"    {tier}: {pct:.1f}% of total return")
    
    return result


if __name__ == "__main__":
    main()
