"""
DIAGNOSE WIN RATE — Why 55.9% When P-value is Significant?
===========================================================

Hypothesis:
1. Asymmetric returns (winners > losers in size)
2. Some instruments/basins drag down average
3. Position sizing masks true edge

Let's find out.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
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


def diagnose():
    print("="*80)
    print("DIAGNOSING WIN RATE")
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
    print("\nFetching data...")
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
    
    train_end = int(min_len * 0.5)
    
    # Train basin statistics
    print("\nTraining basins...")
    basin_stats = defaultdict(lambda: {'returns': [], 'count': 0})
    
    for ticker, prices in data.items():
        returns = np.diff(np.log(prices[:train_end]))
        for i in range(20, len(returns) - 5):
            state = build_state(returns[i-20:i])
            key = get_basin_key(state)
            forward_ret = np.sum(returns[i:i+5])
            basin_stats[key]['returns'].append(forward_ret)
            basin_stats[key]['count'] += 1
    
    # Identify profitable basins
    profitable_basins = set()
    for key, stats in basin_stats.items():
        if len(stats['returns']) >= 30:
            ret = np.array(stats['returns'])
            t_stat, p_value, _ = compute_t_test(ret)
            if t_stat > 1.5 and np.mean(ret) > 0:
                profitable_basins.add(key)
    
    print(f"Profitable basins: {len(profitable_basins)}")
    
    # DIAGNOSTIC 1: Win rate by instrument
    print("\n" + "-"*40)
    print("DIAGNOSTIC 1: WIN RATE BY INSTRUMENT")
    print("-"*40)
    
    instrument_trades = defaultdict(list)
    
    for ticker, prices in data.items():
        test_prices = prices[train_end:]
        test_returns = np.diff(np.log(test_prices))
        
        position = 0
        entry_idx = 0
        entry_price = 0
        
        for i in range(20, len(test_returns) - 5):
            state = build_state(test_returns[i-20:i])
            key = get_basin_key(state)
            
            if position != 0:
                if i - entry_idx >= 5:
                    ret = (test_prices[i] / entry_price - 1)
                    instrument_trades[ticker].append(ret)
                    position = 0
            else:
                if key in profitable_basins:
                    position = 1
                    entry_idx = i
                    entry_price = test_prices[i]
    
    print(f"\n{'Ticker':<8} {'Trades':>7} {'Win%':>8} {'Mean':>10} {'Winners':>10} {'Losers':>10}")
    print("-"*60)
    
    all_trades = []
    for ticker in sorted(instrument_trades.keys()):
        trades = np.array(instrument_trades[ticker])
        all_trades.extend(trades)
        if len(trades) > 0:
            win_rate = np.mean(trades > 0) * 100
            mean_ret = np.mean(trades) * 100
            mean_win = np.mean(trades[trades > 0]) * 100 if np.any(trades > 0) else 0
            mean_loss = np.mean(trades[trades <= 0]) * 100 if np.any(trades <= 0) else 0
            print(f"{ticker:<8} {len(trades):>7} {win_rate:>7.1f}% {mean_ret:>9.2f}% {mean_win:>9.2f}% {mean_loss:>9.2f}%")
    
    # DIAGNOSTIC 2: Distribution of returns
    print("\n" + "-"*40)
    print("DIAGNOSTIC 2: RETURN DISTRIBUTION")
    print("-"*40)
    
    all_trades = np.array(all_trades)
    winners = all_trades[all_trades > 0]
    losers = all_trades[all_trades <= 0]
    
    print(f"\n  Total trades: {len(all_trades)}")
    print(f"  Winners: {len(winners)} ({len(winners)/len(all_trades)*100:.1f}%)")
    print(f"  Losers: {len(losers)} ({len(losers)/len(all_trades)*100:.1f}%)")
    
    print(f"\n  Mean winner: {np.mean(winners)*100:+.3f}%")
    print(f"  Mean loser: {np.mean(losers)*100:+.3f}%")
    print(f"  Win/Loss ratio: {abs(np.mean(winners)/np.mean(losers)):.2f}x")
    
    print(f"\n  Median winner: {np.median(winners)*100:+.3f}%")
    print(f"  Median loser: {np.median(losers)*100:+.3f}%")
    
    # Expected value breakdown
    ev_from_wins = len(winners)/len(all_trades) * np.mean(winners)
    ev_from_losses = len(losers)/len(all_trades) * np.mean(losers)
    
    print(f"\n  EV from wins: {ev_from_wins*100:+.4f}%")
    print(f"  EV from losses: {ev_from_losses*100:+.4f}%")
    print(f"  Net EV: {(ev_from_wins + ev_from_losses)*100:+.4f}%")
    
    # DIAGNOSTIC 3: Win rate by basin confidence
    print("\n" + "-"*40)
    print("DIAGNOSTIC 3: WIN RATE BY BASIN T-STAT")
    print("-"*40)
    
    trades_by_confidence = defaultdict(list)
    
    for ticker, prices in data.items():
        test_prices = prices[train_end:]
        test_returns = np.diff(np.log(test_prices))
        
        position = 0
        entry_idx = 0
        entry_price = 0
        entry_t = 0
        
        for i in range(20, len(test_returns) - 5):
            state = build_state(test_returns[i-20:i])
            key = get_basin_key(state)
            
            if position != 0:
                if i - entry_idx >= 5:
                    ret = (test_prices[i] / entry_price - 1)
                    # Bin by t-stat
                    if entry_t >= 2.5:
                        trades_by_confidence['t≥2.5'].append(ret)
                    elif entry_t >= 2.0:
                        trades_by_confidence['2.0≤t<2.5'].append(ret)
                    elif entry_t >= 1.5:
                        trades_by_confidence['1.5≤t<2.0'].append(ret)
                    position = 0
            else:
                if key in profitable_basins:
                    # Get basin t-stat at entry
                    if key in basin_stats and len(basin_stats[key]['returns']) >= 5:
                        ret_arr = np.array(basin_stats[key]['returns'])
                        t_stat = np.mean(ret_arr) / (np.std(ret_arr)/np.sqrt(len(ret_arr)) + 1e-10)
                        
                        position = 1
                        entry_idx = i
                        entry_price = test_prices[i]
                        entry_t = t_stat
    
    print(f"\n{'T-stat Range':<15} {'Trades':>7} {'Win%':>8} {'Mean':>10} {'Sharpe':>8}")
    print("-"*55)
    
    for conf_level in ['t≥2.5', '2.0≤t<2.5', '1.5≤t<2.0']:
        trades = np.array(trades_by_confidence[conf_level])
        if len(trades) > 0:
            win_rate = np.mean(trades > 0) * 100
            mean_ret = np.mean(trades) * 100
            sharpe = np.mean(trades) / (np.std(trades) + 1e-10) * np.sqrt(252/5)
            print(f"{conf_level:<15} {len(trades):>7} {win_rate:>7.1f}% {mean_ret:>9.3f}% {sharpe:>8.2f}")
    
    # DIAGNOSTIC 4: Temporal pattern
    print("\n" + "-"*40)
    print("DIAGNOSTIC 4: WIN RATE OVER TIME")
    print("-"*40)
    
    # Group trades by quarter
    quarterly_trades = defaultdict(list)
    trade_idx = 0
    
    for ticker, prices in data.items():
        test_prices = prices[train_end:]
        test_returns = np.diff(np.log(test_prices))
        
        position = 0
        entry_idx = 0
        entry_price = 0
        
        for i in range(20, len(test_returns) - 5):
            state = build_state(test_returns[i-20:i])
            key = get_basin_key(state)
            
            if position != 0:
                if i - entry_idx >= 5:
                    ret = (test_prices[i] / entry_price - 1)
                    # Quarter (63 trading days)
                    quarter = i // 63
                    quarterly_trades[quarter].append(ret)
                    trade_idx += 1
                    position = 0
            else:
                if key in profitable_basins:
                    position = 1
                    entry_idx = i
                    entry_price = test_prices[i]
    
    print(f"\n{'Quarter':<10} {'Trades':>7} {'Win%':>8} {'Mean':>10}")
    print("-"*40)
    
    for q in sorted(quarterly_trades.keys()):
        trades = np.array(quarterly_trades[q])
        if len(trades) > 5:
            win_rate = np.mean(trades > 0) * 100
            mean_ret = np.mean(trades) * 100
            print(f"Q{q:<9} {len(trades):>7} {win_rate:>7.1f}% {mean_ret:>9.2f}%")
    
    # DIAGNOSTIC 5: What's the REAL problem?
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)
    
    # Calculate what win rate we NEED for profitability
    avg_win = np.mean(winners)
    avg_loss = abs(np.mean(losers))
    
    breakeven_wr = avg_loss / (avg_win + avg_loss)
    
    print(f"\n  Average win: {avg_win*100:+.3f}%")
    print(f"  Average loss: {-avg_loss*100:.3f}%")
    print(f"  Breakeven win rate: {breakeven_wr*100:.1f}%")
    print(f"  Actual win rate: {len(winners)/len(all_trades)*100:.1f}%")
    print(f"  Edge over breakeven: {(len(winners)/len(all_trades) - breakeven_wr)*100:+.1f}%")
    
    # Is the edge from win rate or asymmetry?
    symmetric_ev = (len(winners)/len(all_trades) - 0.5) * (avg_win + avg_loss)
    asymmetry_ev = 0.5 * (avg_win - avg_loss)
    
    print(f"\n  Edge decomposition:")
    print(f"    From win rate > 50%: {symmetric_ev*100:+.4f}%")
    print(f"    From asymmetry (W>L): {asymmetry_ev*100:+.4f}%")
    
    # The REAL answer
    print("\n" + "-"*40)
    print("THE ANSWER")
    print("-"*40)
    
    actual_wr = len(winners)/len(all_trades)
    
    if avg_win > avg_loss:
        print(f"\n  ✓ Win/Loss asymmetry: {avg_win/avg_loss:.2f}x (winners bigger than losers)")
        print(f"    This means you can be profitable with {breakeven_wr*100:.1f}% win rate")
        print(f"    You have {actual_wr*100:.1f}% → {(actual_wr - breakeven_wr)*100:+.1f}% edge")
    else:
        print(f"\n  ✗ Winners smaller than losers ({avg_win/avg_loss:.2f}x)")
        print(f"    Need higher win rate to compensate")
    
    if actual_wr > 0.5:
        print(f"\n  ✓ Win rate above 50%: {actual_wr*100:.1f}%")
    else:
        print(f"\n  ✗ Win rate below 50%: {actual_wr*100:.1f}%")
    
    # Improvement suggestions
    print("\n" + "-"*40)
    print("IMPROVEMENT OPPORTUNITIES")
    print("-"*40)
    
    # Find best instruments
    best_instruments = []
    for ticker in instrument_trades:
        trades = np.array(instrument_trades[ticker])
        if len(trades) > 10:
            wr = np.mean(trades > 0)
            mean_ret = np.mean(trades)
            if wr > 0.55 or mean_ret > 0.002:
                best_instruments.append((ticker, wr, mean_ret))
    
    if best_instruments:
        print("\n  Best instruments (WR>55% or Mean>0.2%):")
        for ticker, wr, mean in sorted(best_instruments, key=lambda x: -x[2]):
            print(f"    {ticker}: {wr*100:.1f}% WR, {mean*100:.2f}% mean")
    
    # Find worst instruments
    worst_instruments = []
    for ticker in instrument_trades:
        trades = np.array(instrument_trades[ticker])
        if len(trades) > 10:
            wr = np.mean(trades > 0)
            if wr < 0.45:
                worst_instruments.append((ticker, wr, np.mean(trades)))
    
    if worst_instruments:
        print("\n  Worst instruments (WR<45%):")
        for ticker, wr, mean in sorted(worst_instruments, key=lambda x: x[1]):
            print(f"    {ticker}: {wr*100:.1f}% WR, {mean*100:.2f}% mean → EXCLUDE")


if __name__ == "__main__":
    diagnose()
