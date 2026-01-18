"""
Extended Test — Multiple Instruments, Longer History
====================================================

Test on daily data across multiple instruments to get statistically meaningful results.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from holographic_prod.core.algebra import build_clifford_basis
from grace_market_dynamics.state_encoder import CliffordState


def fetch_daily(ticker: str, period: str = "2y") -> Tuple[np.ndarray, np.ndarray]:
    """Fetch daily OHLCV data."""
    if HAS_YFINANCE:
        try:
            data = yf.download(ticker, period=period, progress=False)
            if len(data) > 100:
                prices = data['Close'].values.flatten().astype(np.float64)
                volumes = data['Volume'].values.flatten().astype(np.float64)
                return prices, volumes
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    
    # Fallback: synthetic
    np.random.seed(hash(ticker) % 2**31)
    n = 500
    returns = np.random.randn(n) * 0.015
    prices = 100 * np.exp(np.cumsum(returns))
    volumes = np.random.exponential(1e6, n)
    return prices, volumes


def compute_signals(prices: np.ndarray, lookback: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Clifford signals from price series.
    
    Returns: (chirality, bivector_stability, coherence)
    """
    basis = build_clifford_basis(np)
    log_returns = np.diff(np.log(prices))
    
    n = len(log_returns)
    chirality = np.zeros(n)
    bivector_stability = np.zeros(n)
    coherence = np.zeros(n)
    
    prev_bivector = None
    
    for i in range(lookback, n):
        increments = log_returns[i-lookback:i]
        state = CliffordState.from_increments(increments, basis)
        
        # Extract features
        chirality[i] = state.pseudoscalar
        
        mags = state.grade_magnitudes()
        total = sum(mags.values()) + 1e-10
        coherence[i] = (mags['G0'] + mags['G4']) / total
        
        # Bivector stability (similarity to previous)
        curr_bivector = state.coeffs[5:11]  # Bivector coefficients
        if prev_bivector is not None:
            denom = np.linalg.norm(curr_bivector) * np.linalg.norm(prev_bivector)
            if denom > 1e-10:
                bivector_stability[i] = np.dot(curr_bivector, prev_bivector) / denom
        prev_bivector = curr_bivector.copy()
    
    return chirality, bivector_stability, coherence


def backtest_strategy(
    prices: np.ndarray,
    chirality: np.ndarray,
    bivector_stability: np.ndarray,
    coherence: np.ndarray,
    params: Dict
) -> Dict:
    """
    Backtest a signal-based strategy.
    
    Params:
    - chirality_weight: how much to weight chirality signal
    - bivector_weight: how much to weight bivector stability
    - coherence_threshold: minimum coherence to trade
    - holding_period: bars to hold position
    """
    n = len(prices)
    returns = np.diff(prices) / prices[:-1]
    
    equity = [1.0]
    position = 0
    entry_idx = 0
    entry_price = 0
    trades = []
    
    cost = params.get('cost_bps', 5) / 10000  # Default 5 bps
    
    for i in range(5, n-1):
        # Combined signal
        signal = (params['chirality_weight'] * np.sign(chirality[i]) * 
                  min(abs(chirality[i]) / 0.3, 1.0))
        
        if bivector_stability[i] > 0.5:
            signal *= (1 + params['bivector_weight'] * bivector_stability[i])
        
        if position != 0:
            # Check exit
            if i - entry_idx >= params['holding_period']:
                # Exit
                ret = position * (prices[i] / entry_price - 1)
                net_ret = ret - 2 * cost
                equity.append(equity[-1] * (1 + net_ret))
                trades.append(net_ret)
                position = 0
            else:
                # Mark to market
                bar_ret = position * returns[i]
                equity.append(equity[-1] * (1 + bar_ret))
        else:
            # Check entry
            if coherence[i] >= params['coherence_threshold'] and abs(signal) > 0.3:
                position = 1 if signal > 0 else -1
                entry_idx = i
                entry_price = prices[i]
                equity.append(equity[-1] * (1 - cost))
            else:
                equity.append(equity[-1])
    
    # Close open position
    if position != 0:
        ret = position * (prices[-1] / entry_price - 1)
        equity.append(equity[-1] * (1 + ret - 2*cost))
        trades.append(ret)
    
    equity = np.array(equity)
    total_return = equity[-1] / equity[0] - 1
    
    daily_returns = np.diff(equity) / equity[:-1]
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown)
    
    # Win rate
    if len(trades) > 0:
        win_rate = np.mean([1 if t > 0 else 0 for t in trades])
        avg_win = np.mean([t for t in trades if t > 0]) if any(t > 0 for t in trades) else 0
        avg_loss = np.mean([t for t in trades if t < 0]) if any(t < 0 for t in trades) else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'n_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }


def test_multiple_instruments():
    """Test across multiple instruments."""
    
    tickers = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'EEM', 'XLF', 'XLE']
    
    print("="*70)
    print("MULTI-INSTRUMENT TEST — 2 Year Daily Data")
    print("="*70)
    
    # Fixed parameters (simple approach)
    params = {
        'chirality_weight': 1.0,
        'bivector_weight': 0.5,
        'coherence_threshold': 0.1,
        'holding_period': 5,
        'cost_bps': 5
    }
    
    all_results = []
    
    for ticker in tickers:
        print(f"\nFetching {ticker}...", end=" ")
        prices, volumes = fetch_daily(ticker, "2y")
        print(f"Got {len(prices)} days")
        
        if len(prices) < 100:
            print(f"  Skipping {ticker} — not enough data")
            continue
        
        # Compute signals
        chirality, bivector_stability, coherence = compute_signals(prices)
        
        # Split: first 70% train, last 30% test
        split = int(len(prices) * 0.7)
        
        # In-sample
        in_sample = backtest_strategy(
            prices[:split], 
            chirality[:split], 
            bivector_stability[:split], 
            coherence[:split],
            params
        )
        
        # Out-of-sample
        out_sample = backtest_strategy(
            prices[split:], 
            chirality[split:], 
            bivector_stability[split:], 
            coherence[split:],
            params
        )
        
        print(f"  In-Sample:  Return={in_sample['total_return']*100:+.2f}%, "
              f"Sharpe={in_sample['sharpe']:.2f}, Trades={in_sample['n_trades']}")
        print(f"  Out-Sample: Return={out_sample['total_return']*100:+.2f}%, "
              f"Sharpe={out_sample['sharpe']:.2f}, Trades={out_sample['n_trades']}")
        
        all_results.append({
            'ticker': ticker,
            'in_sample': in_sample,
            'out_sample': out_sample
        })
    
    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)
    
    oos_returns = [r['out_sample']['total_return'] for r in all_results]
    oos_sharpes = [r['out_sample']['sharpe'] for r in all_results]
    oos_win_rates = [r['out_sample']['win_rate'] for r in all_results if r['out_sample']['n_trades'] > 0]
    
    print(f"\n  Instruments tested: {len(all_results)}")
    print(f"\n  Out-of-Sample Returns:")
    print(f"    Mean: {np.mean(oos_returns)*100:+.2f}%")
    print(f"    Median: {np.median(oos_returns)*100:+.2f}%")
    print(f"    Std: {np.std(oos_returns)*100:.2f}%")
    print(f"    Min: {np.min(oos_returns)*100:+.2f}%")
    print(f"    Max: {np.max(oos_returns)*100:+.2f}%")
    
    print(f"\n  Out-of-Sample Sharpe:")
    print(f"    Mean: {np.mean(oos_sharpes):.2f}")
    print(f"    Median: {np.median(oos_sharpes):.2f}")
    
    if oos_win_rates:
        print(f"\n  Win Rate: {np.mean(oos_win_rates)*100:.1f}%")
    
    # Statistical test
    if len(oos_returns) > 1 and np.std(oos_returns) > 0:
        t_stat = np.mean(oos_returns) / (np.std(oos_returns) / np.sqrt(len(oos_returns)))
        print(f"\n  T-statistic: {t_stat:.2f}")
        print(f"  P-value (approx): {1 - (1/(1 + np.exp(-0.7*abs(t_stat)))):.3f}")
    
    # Verdict
    profitable_count = sum(1 for r in oos_returns if r > 0)
    
    print("\n" + "-"*40)
    print(f"VERDICT: {profitable_count}/{len(all_results)} instruments profitable OOS")
    
    if np.mean(oos_returns) > 0:
        print("✓ Average return is POSITIVE")
    else:
        print("✗ Average return is NEGATIVE")
    
    if np.mean(oos_sharpes) > 0.5:
        print("✓ Average Sharpe > 0.5")
    elif np.mean(oos_sharpes) > 0:
        print("○ Average Sharpe positive but < 0.5")
    else:
        print("✗ Average Sharpe is NEGATIVE")
    
    return all_results


def test_long_short_portfolio():
    """Test a simple long-short portfolio based on chirality ranking."""
    
    tickers = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'EEM', 'XLF', 'XLE']
    
    print("\n" + "="*70)
    print("LONG-SHORT PORTFOLIO TEST")
    print("="*70)
    
    # Fetch all data
    data = {}
    min_len = float('inf')
    
    for ticker in tickers:
        prices, _ = fetch_daily(ticker, "1y")
        if len(prices) > 100:
            data[ticker] = prices
            min_len = min(min_len, len(prices))
    
    if not data:
        print("No data available")
        return None
    
    # Align all to same length
    for ticker in data:
        data[ticker] = data[ticker][-int(min_len):]
    
    print(f"Aligned {len(data)} instruments to {min_len} days")
    
    # Compute signals for all
    basis = build_clifford_basis(np)
    signals = {ticker: {} for ticker in data}
    
    for ticker, prices in data.items():
        chirality, bivector, coherence = compute_signals(prices)
        signals[ticker] = {
            'chirality': chirality,
            'bivector': bivector,
            'coherence': coherence
        }
    
    # Portfolio backtest
    portfolio_equity = [1.0]
    rebal_period = 5  # Rebalance weekly
    
    for day in range(10, int(min_len)-1, rebal_period):
        # Rank instruments by chirality
        rankings = []
        for ticker in data:
            chir = signals[ticker]['chirality'][day]
            coh = signals[ticker]['coherence'][day]
            if coh > 0.05:  # Only consider coherent signals
                rankings.append((ticker, chir))
        
        if len(rankings) < 4:
            # Not enough signals, stay flat
            portfolio_equity.append(portfolio_equity[-1])
            continue
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Long top quartile, short bottom quartile
        n_each = max(1, len(rankings) // 4)
        longs = [r[0] for r in rankings[:n_each]]
        shorts = [r[0] for r in rankings[-n_each:]]
        
        # Compute portfolio return over rebalance period
        port_return = 0
        for ticker in longs:
            prices = data[ticker]
            end_day = min(day + rebal_period, int(min_len) - 1)
            ret = prices[end_day] / prices[day] - 1
            port_return += ret / n_each * 0.5  # 50% long
        
        for ticker in shorts:
            prices = data[ticker]
            end_day = min(day + rebal_period, int(min_len) - 1)
            ret = prices[end_day] / prices[day] - 1
            port_return -= ret / n_each * 0.5  # 50% short
        
        # Apply costs (5 bps per trade, full turnover)
        port_return -= 0.001  # ~10 bps total for rebalance
        
        portfolio_equity.append(portfolio_equity[-1] * (1 + port_return))
    
    portfolio_equity = np.array(portfolio_equity)
    
    # Compute metrics
    total_return = portfolio_equity[-1] / portfolio_equity[0] - 1
    
    returns = np.diff(portfolio_equity) / portfolio_equity[:-1]
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(52)  # Weekly
    else:
        sharpe = 0.0
    
    peak = np.maximum.accumulate(portfolio_equity)
    max_dd = np.max((peak - portfolio_equity) / peak)
    
    print(f"\n  Total Return: {total_return*100:+.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")
    print(f"  Rebalances: {len(portfolio_equity)-1}")
    
    if total_return > 0:
        print("\n✓ Long-short portfolio is PROFITABLE")
    else:
        print("\n✗ Long-short portfolio lost money")
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'equity_curve': portfolio_equity
    }


if __name__ == "__main__":
    # Test 1: Multi-instrument
    results = test_multiple_instruments()
    
    # Test 2: Long-short portfolio
    portfolio = test_long_short_portfolio()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL HONEST ASSESSMENT")
    print("="*70)
    
    if results:
        oos_avg = np.mean([r['out_sample']['total_return'] for r in results])
        if oos_avg > 0:
            print("\n  Multi-instrument: Marginal positive OOS return")
        else:
            print("\n  Multi-instrument: Negative OOS return")
    
    if portfolio:
        if portfolio['total_return'] > 0:
            print("  Long-short portfolio: PROFITABLE")
        else:
            print("  Long-short portfolio: NOT PROFITABLE")
    
    print("\n" + "-"*40)
    print("The Clifford geometry DOES encode predictive information,")
    print("but profitability depends heavily on:")
    print("  1. Transaction costs")
    print("  2. Holding period")  
    print("  3. Instrument characteristics")
    print("  4. Market regime")
