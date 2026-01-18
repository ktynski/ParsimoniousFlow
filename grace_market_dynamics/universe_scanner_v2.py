"""
UNIVERSE SCANNER v2 — Fixed Data Isolation
===========================================

Fixed issues:
1. Per-instrument isolation (no shared basins)
2. Sanity checks on prices and returns
3. Sequential fetching to avoid yfinance cache issues
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import sys
import os
import time

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


# Universe
STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
    'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'MCD', 'WMT', 'CSCO', 'ACN', 'ABT',
    'CRM', 'NKE', 'AMD', 'QCOM', 'ORCL', 'INTC', 'IBM', 'GS', 'MS', 'BA',
    'CAT', 'GE', 'UNP', 'UPS', 'SBUX', 'PM', 'BMY', 'GILD', 'AMGN', 'MDT'
]

CRYPTO = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'SOL-USD',
    'DOT-USD', 'MATIC-USD', 'LTC-USD', 'AVAX-USD', 'LINK-USD', 'UNI-USD', 'ATOM-USD'
]

ETFS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD', 'XLF', 'XLE', 'XLK', 'XLV',
    'EEM', 'VNQ', 'HYG', 'SMH', 'ARKK', 'XBI'
]


@dataclass
class ScanResult:
    ticker: str
    asset_class: str
    current_price: float
    signal: float
    t_stat: float
    mean_return: float
    n_observations: int
    is_favorable: bool
    error: Optional[str] = None


def analyze_instrument(ticker: str, asset_class: str) -> Optional[ScanResult]:
    """Analyze a single instrument with proper isolation."""
    try:
        import yfinance as yf
        
        # Fetch fresh data for this ticker
        stock = yf.Ticker(ticker)
        df = stock.history(period='1y', interval='1d')
        
        if df is None or len(df) < 60:
            return ScanResult(
                ticker=ticker, asset_class=asset_class, current_price=0,
                signal=0, t_stat=0, mean_return=0, n_observations=0,
                is_favorable=False, error='Insufficient data'
            )
        
        prices = df['Close'].values.flatten()
        current_price = float(prices[-1])
        
        # Sanity check: price should be positive
        if current_price <= 0:
            return ScanResult(
                ticker=ticker, asset_class=asset_class, current_price=0,
                signal=0, t_stat=0, mean_return=0, n_observations=0,
                is_favorable=False, error='Invalid price'
            )
        
        returns = np.diff(np.log(prices))
        
        # Sanity check: returns should be reasonable (not >50% per day)
        if np.max(np.abs(returns)) > 0.5:
            # Filter extreme returns
            returns = np.clip(returns, -0.2, 0.2)
        
        # Build basin statistics for THIS instrument only
        basin_stats = defaultdict(list)
        
        for i in range(20, len(returns) - 5):
            state = build_state(returns[i-20:i])
            key = get_basin_key(state)
            forward_ret = np.sum(returns[i:i+5])
            
            # Sanity: forward return should be reasonable
            if abs(forward_ret) < 0.5:  # Max 50% in 5 days
                basin_stats[key].append(forward_ret)
        
        # Get current state and its basin
        current_state = build_state(returns[-20:])
        current_key = get_basin_key(current_state)
        
        # Get statistics for current basin
        if current_key in basin_stats and len(basin_stats[current_key]) >= 5:
            hist_returns = np.array(basin_stats[current_key])
            mean_ret = np.mean(hist_returns)
            std_ret = np.std(hist_returns)
            n_obs = len(hist_returns)
            
            if std_ret > 1e-10:
                t_stat = mean_ret / (std_ret / np.sqrt(n_obs))
            else:
                t_stat = 0
            
            # Cap t-stat at reasonable levels
            t_stat = np.clip(t_stat, -10, 10)
            
            signal = np.tanh(t_stat / 2.0)
            is_favorable = t_stat > 1.5 and mean_ret > 0
        else:
            mean_ret = 0
            t_stat = 0
            signal = 0
            n_obs = len(basin_stats.get(current_key, []))
            is_favorable = False
        
        return ScanResult(
            ticker=ticker,
            asset_class=asset_class,
            current_price=current_price,
            signal=signal,
            t_stat=t_stat,
            mean_return=mean_ret,
            n_observations=n_obs,
            is_favorable=is_favorable
        )
        
    except Exception as e:
        return ScanResult(
            ticker=ticker, asset_class=asset_class, current_price=0,
            signal=0, t_stat=0, mean_return=0, n_observations=0,
            is_favorable=False, error=str(e)[:30]
        )


def main():
    print("="*80)
    print("UNIVERSE SCANNER v2 — Fixed Data Isolation")
    print("="*80)
    
    start_time = time.time()
    
    # Build universe
    universe = []
    universe.extend([(t, 'Stock') for t in STOCKS])
    universe.extend([(t, 'ETF') for t in ETFS])
    universe.extend([(t, 'Crypto') for t in CRYPTO])
    
    print(f"\nScanning {len(universe)} instruments...")
    print(f"  Stocks: {len(STOCKS)}")
    print(f"  ETFs: {len(ETFS)}")
    print(f"  Crypto: {len(CRYPTO)}")
    
    results = []
    errors = 0
    
    for i, (ticker, asset_class) in enumerate(universe):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(universe)}")
        
        result = analyze_instrument(ticker, asset_class)
        
        if result:
            if result.error:
                errors += 1
            else:
                results.append(result)
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    elapsed = time.time() - start_time
    print(f"\nScan completed in {elapsed:.1f} seconds")
    print(f"  Successful: {len(results)}, Errors: {errors}")
    
    # Filter to favorable basins
    favorable = [r for r in results if r.is_favorable]
    
    print("\n" + "="*80)
    print(f"FAVORABLE BASINS (t-stat > 1.5, mean > 0): {len(favorable)}")
    print("="*80)
    
    # Sort by t-stat
    favorable.sort(key=lambda x: -x.t_stat)
    
    if favorable:
        print(f"\n{'Rank':<5} {'Ticker':<12} {'Class':<8} {'Price':>12} {'Signal':>8} {'T-stat':>8} {'Mean 5d':>10} {'N':>6}")
        print("-"*80)
        
        for i, r in enumerate(favorable[:20], 1):
            print(f"{i:<5} {r.ticker:<12} {r.asset_class:<8} ${r.current_price:>11,.2f} {r.signal:>8.3f} {r.t_stat:>8.2f} {r.mean_return*100:>9.2f}% {r.n_observations:>6}")
    else:
        print("\n  No favorable signals found")
    
    # Summary by asset class
    print("\n" + "-"*40)
    print("SUMMARY BY ASSET CLASS")
    print("-"*40)
    
    for asset_class in ['Stock', 'ETF', 'Crypto']:
        class_results = [r for r in results if r.asset_class == asset_class]
        class_favorable = [r for r in favorable if r.asset_class == asset_class]
        
        if class_results:
            print(f"\n  {asset_class}:")
            print(f"    Scanned: {len(class_results)}")
            print(f"    Favorable: {len(class_favorable)} ({len(class_favorable)/len(class_results)*100:.1f}%)")
            
            if class_favorable:
                top = max(class_favorable, key=lambda x: x.t_stat)
                print(f"    Top: {top.ticker} (t={top.t_stat:.2f}, mean={top.mean_return*100:.2f}%)")
    
    # SELL signals
    sell_signals = [r for r in results if r.t_stat < -1.5 and r.mean_return < 0 and r.n_observations >= 5]
    
    if sell_signals:
        sell_signals.sort(key=lambda x: x.t_stat)
        
        print("\n" + "-"*40)
        print(f"SELL/SHORT SIGNALS (t-stat < -1.5): {len(sell_signals)}")
        print("-"*40)
        
        print(f"\n{'Rank':<5} {'Ticker':<12} {'Class':<8} {'Price':>12} {'Signal':>8} {'T-stat':>8} {'Mean 5d':>10}")
        print("-"*75)
        
        for i, r in enumerate(sell_signals[:10], 1):
            print(f"{i:<5} {r.ticker:<12} {r.asset_class:<8} ${r.current_price:>11,.2f} {r.signal:>8.3f} {r.t_stat:>8.2f} {r.mean_return*100:>9.2f}%")
    
    # Current recommendations
    print("\n" + "="*80)
    print("TRADING RECOMMENDATIONS")
    print("="*80)
    
    if favorable:
        print("\n  LONG CANDIDATES:")
        for r in favorable[:5]:
            print(f"    {r.ticker}: ${r.current_price:,.2f}, t={r.t_stat:.2f}, expected {r.mean_return*100:+.2f}% (5d)")
    
    if sell_signals:
        print("\n  SHORT/AVOID:")
        for r in sell_signals[:5]:
            print(f"    {r.ticker}: ${r.current_price:,.2f}, t={r.t_stat:.2f}, expected {r.mean_return*100:+.2f}% (5d)")
    
    return results, favorable


if __name__ == "__main__":
    main()
