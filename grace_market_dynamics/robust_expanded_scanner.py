"""
ROBUST EXPANDED SCANNER — Fix Short Training Issue
===================================================

Previous test failed because:
1. Only 137 days training (need 300+)
2. Included volatile small caps with erratic behavior
3. Short 2y period limited by yfinance

Fix:
1. Use 5y period for more training data
2. Filter to liquid, established stocks
3. Require minimum 20 samples per basin
4. Focus on quality over quantity
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys
import os
from datetime import datetime
import json

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


# QUALITY UNIVERSE — Liquid, established stocks with long history
QUALITY_UNIVERSE = [
    # Major ETFs (most liquid, longest history)
    'SPY', 'QQQ', 'IWM', 'DIA', 'MDY', 'IJH', 'IJR', 'VOO', 'VTI', 'IVV',
    'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB', 'XLRE',
    'GLD', 'SLV', 'TLT', 'IEF', 'LQD', 'HYG', 'EEM', 'EFA', 'VNQ',
    
    # Mega Cap (most stable, long history)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST',
    'CAT', 'DE', 'HON', 'UNP', 'UPS', 'BA', 'GE', 'MMM',
    'PG', 'KO', 'PEP', 'PM', 'MO', 'CL',
    'NEE', 'DUK', 'SO', 'D', 'AEP',
    
    # Large Cap Growth
    'CRM', 'ADBE', 'CSCO', 'ACN', 'INTC', 'AMD', 'QCOM', 'TXN', 'NOW', 'IBM',
    'INTU', 'AMAT', 'LRCX', 'MU', 'ADI', 'MCHP', 'NXPI', 'KLAC',
    
    # Large Cap Value
    'BRK.B', 'V', 'MA', 'AXP', 'BLK', 'SCHW', 'CME', 'ICE',
    'LMT', 'RTX', 'NOC', 'GD', 'BA',
    
    # Large Healthcare
    'LLY', 'AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'SYK', 'BSX', 'MDT',
    
    # Consumer Leaders
    'DIS', 'CMCSA', 'NFLX', 'BKNG', 'MAR', 'HLT',
    'F', 'GM',
    
    # More ETFs for diversification
    'SMH', 'XBI', 'IBB', 'XOP', 'XME', 'XRT', 'KRE', 'KBE',
    'VGK', 'VWO', 'VEA', 'IEFA', 'IEMG',
    
    # REITs
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'PSA', 'O', 'DLR',
    
    # Utilities
    'SRE', 'XEL', 'ES', 'WEC', 'EXC', 'ED', 'PEG',
    
    # Materials
    'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'NUE',
]

# Remove duplicates
QUALITY_UNIVERSE = list(dict.fromkeys(QUALITY_UNIVERSE))

print(f"Quality universe: {len(QUALITY_UNIVERSE)} tickers")


def fetch_quality_data(tickers: List[str]) -> Dict[str, np.ndarray]:
    """Fetch 5-year data for quality stocks."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not available")
        return {}
    
    data = {}
    
    print(f"\n  Fetching 5-year data for {len(tickers)} quality stocks...")
    
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"    Progress: {i}/{len(tickers)} ({len(data)} loaded)")
        
        try:
            df = yf.Ticker(ticker).history(period='5y', interval='1d')
            if df is not None and len(df) > 500:  # Need at least 2 years
                prices = df['Close'].values.flatten()
                if len(prices) > 0 and not np.isnan(prices[-1]):
                    data[ticker] = prices
        except:
            pass
    
    print(f"  Loaded {len(data)} instruments with 5-year history")
    return data


def train_robust_basin_stats(
    data: Dict[str, np.ndarray], 
    train_days: int,
    min_samples: int = 20
) -> Dict[str, Dict]:
    """Train basin statistics with robust parameters."""
    print(f"\n  Training on {train_days} days (min {min_samples} samples per basin)...")
    
    basin_stats = {}
    
    for ticker, prices in data.items():
        if len(prices) < train_days + 30:
            continue
        
        train_prices = prices[:train_days]
        returns = np.diff(np.log(train_prices))
        returns = np.clip(returns, -0.2, 0.2)
        
        stats = defaultdict(lambda: {'returns': [], 'count': 0})
        
        for i in range(20, len(returns) - 5):
            state = build_state(returns[i-20:i])
            key = get_basin_key(state)
            fwd = np.sum(returns[i:i+5])
            if abs(fwd) < 0.5:
                stats[key]['returns'].append(fwd)
                stats[key]['count'] += 1
        
        for key in stats:
            rets = np.array(stats[key]['returns'])
            if len(rets) >= min_samples:  # Robust: require 20+ samples
                mean = np.mean(rets)
                std = np.std(rets)
                t = mean / (std / np.sqrt(len(rets)) + 1e-10)
                stats[key]['t_stat'] = t
                stats[key]['mean'] = mean
                stats[key]['samples'] = len(rets)
            else:
                stats[key]['t_stat'] = 0
                stats[key]['mean'] = 0
                stats[key]['samples'] = len(rets)
        
        basin_stats[ticker] = dict(stats)
    
    return basin_stats


def run_robust_backtest():
    """Run robust backtest with quality universe and long history."""
    print("="*80)
    print("ROBUST EXPANDED BACKTEST")
    print("="*80)
    print(f"\nTime: {datetime.now()}")
    print(f"Quality Universe: {len(QUALITY_UNIVERSE)} tickers")
    
    # Fetch 5-year data
    data = fetch_quality_data(QUALITY_UNIVERSE)
    
    if len(data) < 50:
        print("Not enough data")
        return
    
    # Align data
    min_len = min(len(v) for v in data.values())
    for t in data:
        data[t] = data[t][-min_len:]
    
    n = min_len
    train_end = int(n * 0.6)  # 60% train (~3 years)
    val_end = int(n * 0.8)    # 20% validation (~1 year)
    # 20% test (~1 year)
    
    print(f"\n  Total days: {n}")
    print(f"  Training: {train_end} days (~{train_end/252:.1f} years)")
    print(f"  Validation: {val_end - train_end} days")
    print(f"  Out-of-sample test: {n - val_end} days")
    
    # Train with robust parameters
    basin_stats = train_robust_basin_stats(data, train_end, min_samples=20)
    
    print(f"\n  Trained {len(basin_stats)} instruments")
    
    # Count high-quality basins
    total_basins = 0
    high_t_basins = 0
    for ticker, stats in basin_stats.items():
        for key, s in stats.items():
            if s.get('samples', 0) >= 20:
                total_basins += 1
                if s.get('t_stat', 0) >= 3:
                    high_t_basins += 1
    
    print(f"  Total basins (20+ samples): {total_basins}")
    print(f"  High-t basins (t≥3): {high_t_basins}")
    
    # Backtest
    print("\n  Running out-of-sample backtest...")
    
    all_trades = []
    daily_signals = []
    
    test_start = val_end + 20
    test_end = n - 5
    
    for day_idx in range(test_start, test_end):
        day_signals = []
        
        for ticker, prices in data.items():
            if ticker not in basin_stats:
                continue
            
            if day_idx >= len(prices):
                continue
            
            returns = np.diff(np.log(prices[:day_idx+1]))
            returns = np.clip(returns, -0.2, 0.2)
            
            if len(returns) < 20:
                continue
            
            state = build_state(returns[-20:])
            key = get_basin_key(state)
            
            stats = basin_stats[ticker].get(key, {})
            t_stat = stats.get('t_stat', 0)
            mean = stats.get('mean', 0)
            samples = stats.get('samples', 0)
            
            # Require t≥3 AND 20+ samples
            if t_stat >= 3 and mean > 0 and samples >= 20:
                day_signals.append({
                    'ticker': ticker,
                    't_stat': t_stat,
                    'expected': mean,
                    'samples': samples
                })
        
        daily_signals.append(len(day_signals))
        
        # Sort by t-stat, take top 10
        day_signals = sorted(day_signals, key=lambda x: -x['t_stat'])[:10]
        
        for sig in day_signals:
            ticker = sig['ticker']
            prices = data[ticker]
            
            if day_idx + 5 < len(prices):
                actual_ret = np.log(prices[day_idx + 5] / prices[day_idx])
                cost = 15 / 10000
                net_ret = actual_ret - cost
                
                all_trades.append({
                    'ticker': ticker,
                    't_stat': sig['t_stat'],
                    'expected': sig['expected'],
                    'actual': actual_ret,
                    'net': net_ret,
                    'won': net_ret > 0,
                    'samples': sig['samples']
                })
    
    # Results
    print("\n" + "="*60)
    print("ROBUST BACKTEST RESULTS")
    print("="*60)
    
    print(f"\n  Instruments: {len(data)}")
    print(f"  Test days: {test_end - test_start}")
    print(f"  Total trades: {len(all_trades)}")
    
    if not all_trades:
        print("  No trades generated")
        return
    
    net_returns = np.array([t['net'] for t in all_trades])
    t_stats = np.array([t['t_stat'] for t in all_trades])
    
    print(f"\n  SIGNAL DENSITY:")
    print(f"    Mean signals/day: {np.mean(daily_signals):.1f}")
    print(f"    Max signals/day: {np.max(daily_signals)}")
    print(f"    Days with 5+ signals: {np.sum(np.array(daily_signals) >= 5)}")
    print(f"    Days with 10+ signals: {np.sum(np.array(daily_signals) >= 10)}")
    
    print(f"\n  PERFORMANCE (all trades):")
    print(f"    Win rate: {np.mean(net_returns > 0)*100:.1f}%")
    print(f"    Mean return: {np.mean(net_returns)*100:.3f}%")
    print(f"    Std return: {np.std(net_returns)*100:.3f}%")
    sharpe = np.mean(net_returns) / (np.std(net_returns) + 1e-10) * np.sqrt(252/5)
    print(f"    Sharpe: {sharpe:.2f}")
    
    if len(net_returns) > 5:
        t_test, p_val, ci = compute_t_test(net_returns)
        print(f"    T-stat: {t_test:.2f}")
        print(f"    P-value: {p_val:.4f}")
        print(f"    Significant: {'YES ✓' if p_val < 0.05 else 'NO'}")
    
    # By t-stat tier
    print(f"\n  BY T-STAT TIER:")
    for thresh in [3, 4, 5]:
        mask = t_stats >= thresh
        if np.sum(mask) > 0:
            tier_rets = net_returns[mask]
            wr = np.mean(tier_rets > 0) * 100
            mr = np.mean(tier_rets) * 100
            print(f"    t≥{thresh}: {len(tier_rets)} trades, {wr:.1f}% win, {mr:.3f}% mean")
    
    # Today's signals (most recent day)
    print("\n" + "="*60)
    print("CURRENT TOP SIGNALS")
    print("="*60)
    
    current_signals = []
    for ticker, prices in data.items():
        if ticker not in basin_stats:
            continue
        
        returns = np.diff(np.log(prices))
        returns = np.clip(returns, -0.2, 0.2)
        
        if len(returns) < 20:
            continue
        
        state = build_state(returns[-20:])
        key = get_basin_key(state)
        
        stats = basin_stats[ticker].get(key, {})
        t_stat = stats.get('t_stat', 0)
        mean = stats.get('mean', 0)
        samples = stats.get('samples', 0)
        
        if t_stat >= 3 and mean > 0 and samples >= 20:
            current_signals.append({
                'ticker': ticker,
                't_stat': t_stat,
                'expected': mean,
                'samples': samples,
                'price': prices[-1]
            })
    
    current_signals = sorted(current_signals, key=lambda x: -x['t_stat'])
    
    print(f"\n  Found {len(current_signals)} actionable signals today")
    print(f"\n  {'Rank':<6} {'Ticker':<8} {'T-stat':>10} {'Expected':>12} {'Samples':>10}")
    print("  " + "-"*55)
    
    for i, sig in enumerate(current_signals[:15]):
        print(f"  {i+1:<6} {sig['ticker']:<8} {sig['t_stat']:>10.2f} {sig['expected']*100:>11.2f}% {sig['samples']:>10}")
    
    # Projection
    print("\n" + "="*60)
    print("REALISTIC PROJECTION")
    print("="*60)
    
    if len(all_trades) > 0 and np.mean(net_returns) > 0:
        avg_signals = np.mean(daily_signals)
        avg_trades = min(avg_signals, 10)
        daily_pnl = avg_trades * np.mean(net_returns) * 20000 * 0.10
        monthly_pnl = daily_pnl * 21
        
        print(f"\n  Capital: $20,000")
        print(f"  Avg signals/day: {avg_signals:.1f}")
        print(f"  Trades/day: {avg_trades:.0f}")
        print(f"  Win rate: {np.mean(net_returns > 0)*100:.0f}%")
        print(f"  Mean return: {np.mean(net_returns)*100:.3f}%")
        print(f"\n  Daily P&L: ${daily_pnl:.0f}")
        print(f"  Monthly P&L: ${monthly_pnl:.0f}")
        print(f"  Annual (compounded): ${20000 * ((1 + monthly_pnl/20000)**12 - 1):,.0f}")
    else:
        print("\n  Strategy not profitable — need to investigate further")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'instruments': len(data),
        'test_days': test_end - test_start,
        'total_trades': len(all_trades),
        'mean_signals_per_day': float(np.mean(daily_signals)) if daily_signals else 0,
        'win_rate': float(np.mean(net_returns > 0)) if len(net_returns) > 0 else 0,
        'mean_return': float(np.mean(net_returns)) if len(net_returns) > 0 else 0,
        'sharpe': float(sharpe) if 'sharpe' in dir() else 0,
        'current_signals': current_signals[:20]
    }
    
    with open('robust_backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n  Results saved to robust_backtest_results.json")


if __name__ == "__main__":
    run_robust_backtest()
