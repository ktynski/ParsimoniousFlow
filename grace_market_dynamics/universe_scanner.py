"""
UNIVERSE SCANNER — Scan All Stocks & Crypto
=============================================

Scans a broad universe for current Grace basin signals:
- S&P 500 stocks
- NASDAQ 100
- Major cryptocurrencies
- ETFs

Identifies instruments currently in favorable basins.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# =============================================================================
# UNIVERSE DEFINITIONS
# =============================================================================

# S&P 500 sample (top 100 by market cap)
SP500_SAMPLE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
    'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'MCD', 'WMT', 'CSCO', 'ACN', 'ABT',
    'DHR', 'NEE', 'VZ', 'ADBE', 'NKE', 'CRM', 'TXN', 'PM', 'BMY', 'UPS',
    'RTX', 'QCOM', 'ORCL', 'HON', 'LOW', 'INTC', 'UNP', 'IBM', 'AMGN', 'CAT',
    'GE', 'BA', 'SBUX', 'PLD', 'INTU', 'GS', 'SPGI', 'DE', 'MS', 'BLK',
    'AMD', 'GILD', 'AXP', 'ISRG', 'MDT', 'ADI', 'CVS', 'SYK', 'BKNG', 'MDLZ',
    'TJX', 'VRTX', 'MMC', 'CB', 'ADP', 'REGN', 'PGR', 'CI', 'SO', 'MO',
    'DUK', 'CME', 'LRCX', 'ZTS', 'NOC', 'ITW', 'EOG', 'BSX', 'ETN', 'SCHW',
    'APD', 'SHW', 'FDX', 'CL', 'ATVI', 'MU', 'HUM', 'AON', 'ICE', 'PNC'
]

# NASDAQ 100 sample (tech heavy)
NASDAQ_SAMPLE = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST',
    'PEP', 'CSCO', 'ADBE', 'CMCSA', 'NFLX', 'AMD', 'INTC', 'TXN', 'QCOM', 'AMGN',
    'HON', 'INTU', 'SBUX', 'ISRG', 'GILD', 'ADI', 'BKNG', 'MDLZ', 'VRTX', 'REGN',
    'ADP', 'LRCX', 'MU', 'PYPL', 'SNPS', 'KLAC', 'ASML', 'CDNS', 'MELI', 'ORLY',
    'CSX', 'MNST', 'MAR', 'CTAS', 'PANW', 'KDP', 'MRVL', 'NXPI', 'PCAR', 'PAYX',
    'AZN', 'KHC', 'DXCM', 'LULU', 'ROST', 'ODFL', 'WDAY', 'EXC', 'CPRT', 'SGEN',
    'EA', 'FAST', 'VRSK', 'XEL', 'CTSH', 'IDXX', 'EBAY', 'ZS', 'ANSS', 'DLTR',
    'ILMN', 'WBD', 'BIIB', 'TEAM', 'FANG', 'JD', 'CRWD', 'DDOG', 'ZM', 'LCID'
]

# Cryptocurrencies (via yfinance)
CRYPTO = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'SOL-USD',
    'DOT-USD', 'MATIC-USD', 'LTC-USD', 'SHIB-USD', 'TRX-USD', 'AVAX-USD', 'LINK-USD',
    'ATOM-USD', 'UNI-USD', 'XMR-USD', 'ETC-USD', 'XLM-USD', 'BCH-USD',
    'FIL-USD', 'AAVE-USD', 'ALGO-USD', 'VET-USD', 'SAND-USD', 'MANA-USD',
    'AXS-USD', 'THETA-USD', 'ICP-USD', 'FTM-USD'
]

# Major ETFs
ETFS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'EEM', 'EFA',
    'TLT', 'IEF', 'LQD', 'HYG', 'GLD', 'SLV', 'USO', 'UNG', 'VNQ', 'XLF',
    'XLE', 'XLK', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB', 'XLRE', 'XLC',
    'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ', 'SMH', 'SOXX', 'XBI', 'IBB', 'KRE'
]


@dataclass
class ScanResult:
    """Result of scanning a single instrument."""
    ticker: str
    asset_class: str
    current_price: float
    basin_key: Tuple[int, ...]
    signal: float
    t_stat: float
    mean_return: float
    n_observations: int
    is_favorable: bool
    error: Optional[str] = None


def fetch_and_analyze(ticker: str, asset_class: str, yf) -> Optional[ScanResult]:
    """Fetch data and analyze a single ticker."""
    try:
        # Fetch 1 year of daily data for training + recent for signal
        df = yf.download(ticker, period='1y', interval='1d', progress=False, timeout=10)
        
        if df is None or len(df) < 60:
            return ScanResult(
                ticker=ticker, asset_class=asset_class, current_price=0,
                basin_key=(), signal=0, t_stat=0, mean_return=0,
                n_observations=0, is_favorable=False, error='Insufficient data'
            )
        
        prices = df['Close'].values.flatten()
        current_price = float(prices[-1])
        returns = np.diff(np.log(prices))
        
        # Build basin statistics from historical data
        basin_stats = defaultdict(list)
        
        for i in range(20, len(returns) - 5):
            state = build_state(returns[i-20:i])
            key = get_basin_key(state)
            forward_ret = np.sum(returns[i:i+5])
            basin_stats[key].append(forward_ret)
        
        # Get current state
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
            basin_key=current_key,
            signal=signal,
            t_stat=t_stat,
            mean_return=mean_ret,
            n_observations=n_obs,
            is_favorable=is_favorable
        )
        
    except Exception as e:
        return ScanResult(
            ticker=ticker, asset_class=asset_class, current_price=0,
            basin_key=(), signal=0, t_stat=0, mean_return=0,
            n_observations=0, is_favorable=False, error=str(e)[:50]
        )


def scan_universe(
    include_stocks: bool = True,
    include_crypto: bool = True,
    include_etfs: bool = True,
    max_workers: int = 10
) -> List[ScanResult]:
    """Scan the full universe."""
    
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not available")
        return []
    
    # Build universe
    universe = []
    
    if include_etfs:
        universe.extend([(t, 'ETF') for t in ETFS])
    
    if include_stocks:
        # Dedupe
        seen = set(t for t, _ in universe)
        for t in SP500_SAMPLE:
            if t not in seen:
                universe.append((t, 'Stock'))
                seen.add(t)
        for t in NASDAQ_SAMPLE:
            if t not in seen:
                universe.append((t, 'Stock'))
                seen.add(t)
    
    if include_crypto:
        universe.extend([(t, 'Crypto') for t in CRYPTO])
    
    print(f"\nScanning {len(universe)} instruments...")
    print(f"  Stocks: {sum(1 for _, c in universe if c == 'Stock')}")
    print(f"  ETFs: {sum(1 for _, c in universe if c == 'ETF')}")
    print(f"  Crypto: {sum(1 for _, c in universe if c == 'Crypto')}")
    
    results = []
    errors = 0
    
    # Parallel fetch
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_and_analyze, ticker, asset_class, yf): (ticker, asset_class)
            for ticker, asset_class in universe
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0:
                print(f"  Progress: {completed}/{len(universe)}")
            
            result = future.result()
            if result:
                if result.error:
                    errors += 1
                else:
                    results.append(result)
    
    print(f"\n  Completed: {len(results)} successful, {errors} errors")
    
    return results


def main():
    print("="*80)
    print("UNIVERSE SCANNER — Grace Basin Signals")
    print("="*80)
    
    start_time = time.time()
    
    # Scan everything
    results = scan_universe(
        include_stocks=True,
        include_crypto=True,
        include_etfs=True,
        max_workers=15
    )
    
    elapsed = time.time() - start_time
    print(f"\nScan completed in {elapsed:.1f} seconds")
    
    # Filter to favorable basins
    favorable = [r for r in results if r.is_favorable]
    
    print("\n" + "="*80)
    print(f"FAVORABLE BASINS (t-stat > 1.5, mean > 0): {len(favorable)}")
    print("="*80)
    
    # Sort by t-stat
    favorable.sort(key=lambda x: -x.t_stat)
    
    if favorable:
        print(f"\n{'Rank':<5} {'Ticker':<12} {'Class':<8} {'Price':>10} {'Signal':>8} {'T-stat':>8} {'Mean':>10} {'N':>6}")
        print("-"*80)
        
        for i, r in enumerate(favorable[:30], 1):
            print(f"{i:<5} {r.ticker:<12} {r.asset_class:<8} ${r.current_price:>9.2f} {r.signal:>8.3f} {r.t_stat:>8.2f} {r.mean_return*100:>9.2f}% {r.n_observations:>6}")
    
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
                print(f"    Top signal: {top.ticker} (t={top.t_stat:.2f}, mean={top.mean_return*100:.2f}%)")
    
    # CURRENT TRADING RECOMMENDATIONS
    print("\n" + "="*80)
    print("CURRENT TRADING RECOMMENDATIONS")
    print("="*80)
    
    # Top 10 overall
    print("\n  TOP 10 SIGNALS (by t-stat):")
    print(f"  {'Ticker':<12} {'Class':<8} {'Signal':>8} {'T-stat':>8} {'Expected 5d':>12}")
    print("  " + "-"*55)
    
    for r in favorable[:10]:
        expected = r.mean_return * 100
        print(f"  {r.ticker:<12} {r.asset_class:<8} {r.signal:>8.3f} {r.t_stat:>8.2f} {expected:>+11.2f}%")
    
    # Check for any SELL signals (strongly negative basins)
    sell_signals = [r for r in results if r.t_stat < -1.5 and r.mean_return < 0 and r.n_observations >= 5]
    
    if sell_signals:
        sell_signals.sort(key=lambda x: x.t_stat)
        
        print("\n  SELL/SHORT SIGNALS (t-stat < -1.5):")
        print(f"  {'Ticker':<12} {'Class':<8} {'Signal':>8} {'T-stat':>8} {'Expected 5d':>12}")
        print("  " + "-"*55)
        
        for r in sell_signals[:10]:
            expected = r.mean_return * 100
            print(f"  {r.ticker:<12} {r.asset_class:<8} {r.signal:>8.3f} {r.t_stat:>8.2f} {expected:>+11.2f}%")
    
    # Portfolio suggestion
    print("\n" + "-"*40)
    print("SUGGESTED PORTFOLIO")
    print("-"*40)
    
    # Diversified selection
    stocks = [r for r in favorable if r.asset_class == 'Stock'][:3]
    etfs = [r for r in favorable if r.asset_class == 'ETF'][:2]
    crypto = [r for r in favorable if r.asset_class == 'Crypto'][:2]
    
    portfolio = stocks + etfs + crypto
    
    if portfolio:
        print(f"\n  Equal-weight portfolio ({len(portfolio)} positions):")
        for r in portfolio:
            weight = 100 / len(portfolio)
            print(f"    {r.ticker}: {weight:.1f}% (t={r.t_stat:.2f}, exp={r.mean_return*100:.2f}%)")
        
        # Expected portfolio return
        avg_expected = np.mean([r.mean_return for r in portfolio]) * 100
        print(f"\n  Expected 5-day return: {avg_expected:+.2f}%")
    
    return results, favorable


if __name__ == "__main__":
    main()
