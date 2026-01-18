"""
PROFITABLE BASIN TEST — Trade Only in Significant Regimes
==========================================================

Key finding: 4 basins with p < 0.1:
  - 0.43% mean, t=3.10, p=0.002 (***)
  - 0.44% mean, t=2.32, p=0.020 (**)
  - 0.52% mean, t=1.75, p=0.081 (*)
  - 0.31% mean, t=1.63, p=0.104

These ARE statistically significant. The question:
Can we trade ONLY when we're in these basins?

Theory: Grace basin routing = O(1) regime lookup
        If certain regimes are predictable, we should exploit them
"""

import numpy as np
from typing import Dict, Set, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import (
    PHI, PHI_INV, DTYPE
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


def run_profitable_basin_test():
    """Trade ONLY in statistically significant basins."""
    print("="*80)
    print("PROFITABLE BASIN TEST — Trade Only in Significant Regimes")
    print("="*80)
    
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
    
    min_len = min(len(v) for v in data.values())
    for ticker in data:
        data[ticker] = data[ticker][-min_len:]
    
    # Split: 50% train, 20% validate, 30% test
    train_end = int(min_len * 0.5)
    val_end = int(min_len * 0.7)
    
    print(f"Train: {train_end} days, Validation: {val_end - train_end} days, Test: {min_len - val_end} days")
    
    basis = build_clifford_basis(np)
    resolution = PHI_INV ** 5  # Optimal
    
    # PHASE 1: TRAINING — Discover basins and their statistics
    print("\n" + "-"*40)
    print("PHASE 1: TRAINING — Discover Basins")
    print("-"*40)
    
    basin_stats: Dict[Tuple[int, ...], Dict] = {}
    
    for ticker, prices in data.items():
        returns = np.diff(np.log(prices[:train_end]))
        
        for i in range(20, len(returns) - 5):
            state = build_state(returns[i-20:i], basis)
            key = grace_basin_key_direct(state, basis, n_iters=3, resolution=resolution, xp=np)[:8]
            forward_ret = np.sum(returns[i:i+5])
            
            if key not in basin_stats:
                basin_stats[key] = {'returns': [], 'direction': 0}
            basin_stats[key]['returns'].append(forward_ret)
    
    print(f"  Discovered {len(basin_stats)} unique basins")
    
    # PHASE 2: VALIDATION — Find profitable basins
    print("\n" + "-"*40)
    print("PHASE 2: VALIDATION — Identify Profitable Basins")
    print("-"*40)
    
    profitable_basins: Set[Tuple[int, ...]] = set()
    unprofitable_basins: Set[Tuple[int, ...]] = set()
    
    for basin_key, stats in basin_stats.items():
        if len(stats['returns']) < 30:  # Need enough samples
            continue
        
        ret = np.array(stats['returns'])
        t_stat, p_value, _ = compute_t_test(ret)
        mean_ret = np.mean(ret)
        
        if p_value < 0.1 and mean_ret > 0:
            profitable_basins.add(basin_key)
            stats['direction'] = 1
            print(f"  + Basin: mean={mean_ret*100:.3f}%, t={t_stat:.2f}, p={p_value:.3f}, n={len(ret)}")
        elif p_value < 0.1 and mean_ret < 0:
            unprofitable_basins.add(basin_key)
            stats['direction'] = -1
            print(f"  - Basin: mean={mean_ret*100:.3f}%, t={t_stat:.2f}, p={p_value:.3f}, n={len(ret)}")
    
    print(f"\n  Profitable basins: {len(profitable_basins)}")
    print(f"  Unprofitable basins (short): {len(unprofitable_basins)}")
    
    # PHASE 3: OUT-OF-SAMPLE TEST
    print("\n" + "-"*40)
    print("PHASE 3: OUT-OF-SAMPLE TEST")
    print("-"*40)
    
    # Test on validation set first
    print("\n  Validation Set:")
    
    val_trades = []
    for ticker, prices in data.items():
        val_prices = prices[train_end:val_end]
        val_returns = np.diff(np.log(val_prices))
        
        position = 0
        entry_idx = 0
        entry_price = 0
        
        for i in range(20, len(val_returns) - 5):
            state = build_state(val_returns[i-20:i], basis)
            key = grace_basin_key_direct(state, basis, n_iters=3, resolution=resolution, xp=np)[:8]
            
            if position != 0:
                if i - entry_idx >= 5:
                    ret = position * (val_prices[i] / entry_price - 1)
                    val_trades.append(ret)
                    position = 0
            else:
                if key in profitable_basins:
                    position = 1
                    entry_idx = i
                    entry_price = val_prices[i]
                elif key in unprofitable_basins:
                    position = -1
                    entry_idx = i
                    entry_price = val_prices[i]
    
    if val_trades:
        val_trades = np.array(val_trades)
        t_stat, p_value, ci = compute_t_test(val_trades)
        sharpe = np.mean(val_trades) / (np.std(val_trades) + 1e-10) * np.sqrt(252/5)
        
        print(f"    Trades: {len(val_trades)}")
        print(f"    Mean Return: {np.mean(val_trades)*100:.3f}%")
        print(f"    Sharpe: {sharpe:.2f}")
        print(f"    Win Rate: {np.mean(val_trades > 0)*100:.1f}%")
        print(f"    T-stat: {t_stat:.2f}, P-value: {p_value:.4f}")
    
    # True OOS test
    print("\n  TRUE OUT-OF-SAMPLE TEST:")
    
    oos_trades = []
    oos_details = []
    
    for ticker, prices in data.items():
        test_prices = prices[val_end:]
        test_returns = np.diff(np.log(test_prices))
        
        position = 0
        entry_idx = 0
        entry_price = 0
        
        for i in range(20, len(test_returns) - 5):
            state = build_state(test_returns[i-20:i], basis)
            key = grace_basin_key_direct(state, basis, n_iters=3, resolution=resolution, xp=np)[:8]
            
            if position != 0:
                if i - entry_idx >= 5:
                    ret = position * (test_prices[i] / entry_price - 1)
                    oos_trades.append(ret)
                    oos_details.append({'ticker': ticker, 'return': ret, 'type': 'long' if position > 0 else 'short'})
                    position = 0
            else:
                if key in profitable_basins:
                    position = 1
                    entry_idx = i
                    entry_price = test_prices[i]
                elif key in unprofitable_basins:
                    position = -1
                    entry_idx = i
                    entry_price = test_prices[i]
    
    if oos_trades:
        oos_trades = np.array(oos_trades)
        t_stat, p_value, ci = compute_t_test(oos_trades)
        sharpe = np.mean(oos_trades) / (np.std(oos_trades) + 1e-10) * np.sqrt(252/5)
        
        print(f"    Trades: {len(oos_trades)}")
        print(f"    Mean Return: {np.mean(oos_trades)*100:.3f}%")
        print(f"    Sharpe: {sharpe:.2f}")
        print(f"    Win Rate: {np.mean(oos_trades > 0)*100:.1f}%")
        print(f"    T-stat: {t_stat:.2f}")
        print(f"    P-value: {p_value:.4f}")
        print(f"    95% CI: [{ci[0]*100:.3f}%, {ci[1]*100:.3f}%]")
        
        # Long vs Short breakdown
        longs = [d['return'] for d in oos_details if d['type'] == 'long']
        shorts = [d['return'] for d in oos_details if d['type'] == 'short']
        
        print(f"\n    Long trades: {len(longs)}, Mean: {np.mean(longs)*100:.3f}%" if longs else "\n    No long trades")
        print(f"    Short trades: {len(shorts)}, Mean: {np.mean(shorts)*100:.3f}%" if shorts else "    No short trades")
    
    # PHASE 4: PORTFOLIO TEST — Only trade in profitable basins
    print("\n" + "-"*40)
    print("PHASE 4: PORTFOLIO — Long/Short Only in Significant Basins")
    print("-"*40)
    
    portfolio_returns = []
    
    test_len = min_len - val_end
    
    for week in range(0, test_len - 5, 5):
        longs = []
        shorts = []
        
        for ticker, prices in data.items():
            test_prices = prices[val_end:]
            test_returns = np.diff(np.log(test_prices))
            
            if week >= len(test_returns) - 5:
                continue
            
            state = build_state(test_returns[max(0, week-20):week] if week >= 20 else test_returns[:1], basis)
            key = grace_basin_key_direct(state, basis, n_iters=3, resolution=resolution, xp=np)[:8]
            
            if key in profitable_basins:
                longs.append(ticker)
            elif key in unprofitable_basins:
                shorts.append(ticker)
        
        if not longs and not shorts:
            continue  # No signal this week
        
        period_ret = 0
        n_positions = 0
        
        for ticker in longs:
            tp = data[ticker][val_end:]
            if week + 5 < len(tp):
                period_ret += (tp[week+5] / tp[week] - 1)
                n_positions += 1
        
        for ticker in shorts:
            tp = data[ticker][val_end:]
            if week + 5 < len(tp):
                period_ret -= (tp[week+5] / tp[week] - 1)
                n_positions += 1
        
        if n_positions > 0:
            period_ret /= n_positions
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
        print(f"  N Periods with signal: {len(returns)}")
        print(f"  T-statistic: {t_stat:.2f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  95% CI: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")
        
        print(f"\n  Significant (p<0.05): {'YES ✓' if p_value < 0.05 else 'NO'}")
        print(f"  Significant (p<0.10): {'YES ✓' if p_value < 0.10 else 'NO'}")
    
    # FINAL VERDICT
    print("\n" + "="*80)
    print("FINAL VERDICT — BASIN-BASED REGIME TRADING")
    print("="*80)
    
    if oos_trades is not None and len(oos_trades) > 0:
        t_stat, p_value, _ = compute_t_test(oos_trades)
        sharpe = np.mean(oos_trades) / (np.std(oos_trades) + 1e-10) * np.sqrt(252/5)
        
        print(f"\n  OOS Performance:")
        print(f"    - N Trades: {len(oos_trades)}")
        print(f"    - Sharpe: {sharpe:.2f}")
        print(f"    - P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("\n  ✓ STATISTICALLY SIGNIFICANT at p<0.05")
            print("    Theory VALIDATED: Grace basins identify predictable regimes")
        elif p_value < 0.1:
            print("\n  ~ MARGINALLY SIGNIFICANT at p<0.10")
            print("    Partial validation: Signal present but noisy")
        elif sharpe > 0:
            print("\n  ~ POSITIVE but not significant")
            print("    Need more data or regime refinement")
        else:
            print("\n  ✗ NOT PROFITABLE in OOS")
            print("    Basin statistics may have overfit")


if __name__ == "__main__":
    run_profitable_basin_test()
