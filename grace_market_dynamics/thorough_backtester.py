"""
THOROUGH BACKTESTER — Validate Multi-Instrument, Multi-Timeframe Strategy
==========================================================================

REQUIREMENTS:
1. REAL historical data (no fake/mock)
2. Walk-forward validation (no lookahead bias)
3. Proper transaction costs and slippage
4. Statistical significance testing
5. Honest reporting with confidence intervals

Tests:
1. Single instrument, single timeframe (baseline)
2. Multiple instruments, single timeframe
3. Single instrument, multiple timeframes
4. Full: multiple instruments × multiple timeframes
5. Signal selection (top N by t-stat)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import sys
import os
import time
from datetime import datetime

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


@dataclass
class Trade:
    """Record of a single trade."""
    ticker: str
    timeframe: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    t_stat: float
    gross_return: float
    net_return: float
    
    
@dataclass
class BacktestResult:
    """Complete backtest results."""
    name: str
    n_trades: int
    win_rate: float
    mean_return: float
    std_return: float
    sharpe: float
    max_drawdown: float
    total_return: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))


class ThoroughBacktester:
    """
    Rigorous walk-forward backtester.
    """
    
    def __init__(
        self,
        train_ratio: float = 0.5,  # 50% train, 50% test
        min_basin_samples: int = 10,
        min_t_stat: float = 1.5,
        holding_period: int = 5,
        cost_bps: float = 10.0,  # Conservative: 10 bps round-trip
        slippage_bps: float = 5.0,  # 5 bps slippage
        max_positions: int = 20,
    ):
        self.train_ratio = train_ratio
        self.min_basin_samples = min_basin_samples
        self.min_t_stat = min_t_stat
        self.holding_period = holding_period
        self.cost_bps = cost_bps
        self.slippage_bps = slippage_bps
        self.max_positions = max_positions
        
        # Results storage
        self.all_trades: List[Trade] = []
        
    def _fetch_data(self, ticker: str, period: str, interval: str) -> Optional[np.ndarray]:
        """Fetch historical data."""
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            if df is None or len(df) < 100:
                return None
            return df['Close'].values.flatten()
        except:
            return None
    
    def _build_basin_stats(
        self, 
        returns: np.ndarray, 
        holding: int
    ) -> Dict[Tuple[int, ...], Dict]:
        """Build basin statistics from training data."""
        basin_stats = defaultdict(lambda: {'returns': [], 'count': 0})
        
        for i in range(20, len(returns) - holding):
            state = build_state(returns[i-20:i])
            key = get_basin_key(state)
            forward_ret = np.sum(returns[i:i+holding])
            
            # Sanity check
            if abs(forward_ret) < 0.5:
                basin_stats[key]['returns'].append(forward_ret)
                basin_stats[key]['count'] += 1
        
        # Compute statistics
        for key in basin_stats:
            rets = np.array(basin_stats[key]['returns'])
            if len(rets) >= self.min_basin_samples:
                basin_stats[key]['mean'] = np.mean(rets)
                basin_stats[key]['std'] = np.std(rets)
                basin_stats[key]['t_stat'] = (
                    basin_stats[key]['mean'] / 
                    (basin_stats[key]['std'] / np.sqrt(len(rets)) + 1e-10)
                )
            else:
                basin_stats[key]['mean'] = 0
                basin_stats[key]['std'] = 1
                basin_stats[key]['t_stat'] = 0
        
        return dict(basin_stats)
    
    def _get_signal(
        self, 
        returns: np.ndarray, 
        basin_stats: Dict
    ) -> Tuple[float, float]:
        """Get trading signal for current state."""
        if len(returns) < 20:
            return 0.0, 0.0
        
        state = build_state(returns[-20:])
        key = get_basin_key(state)
        
        if key not in basin_stats:
            return 0.0, 0.0
        
        stats = basin_stats[key]
        t_stat = stats.get('t_stat', 0)
        mean_ret = stats.get('mean', 0)
        
        if t_stat < self.min_t_stat or mean_ret <= 0:
            return 0.0, 0.0
        
        # Signal strength = t-stat
        return 1.0, t_stat
    
    def backtest_single(
        self, 
        ticker: str, 
        period: str = '2y',
        interval: str = '1d',
        holding: int = None
    ) -> Optional[BacktestResult]:
        """Backtest single instrument, single timeframe."""
        
        prices = self._fetch_data(ticker, period, interval)
        if prices is None:
            return None
        
        holding = holding or self.holding_period
        returns = np.diff(np.log(prices))
        returns = np.clip(returns, -0.2, 0.2)
        
        n = len(returns)
        train_end = int(n * self.train_ratio)
        
        # Build basin stats from training period
        basin_stats = self._build_basin_stats(returns[:train_end], holding)
        
        # Test on remaining data
        trades = []
        equity = [1.0]
        
        position = 0
        entry_idx = 0
        entry_price = 0
        entry_t = 0
        
        for i in range(train_end + 20, n - holding):
            signal, t_stat = self._get_signal(returns[i-20:i], basin_stats)
            
            if position != 0:
                # Check exit
                if i - entry_idx >= holding:
                    exit_price = prices[i]
                    gross_ret = position * (exit_price / entry_price - 1)
                    cost = (self.cost_bps + self.slippage_bps) / 10000
                    net_ret = gross_ret - cost
                    
                    trades.append(Trade(
                        ticker=ticker,
                        timeframe=interval,
                        entry_time=entry_idx,
                        exit_time=i,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=position,
                        t_stat=entry_t,
                        gross_return=gross_ret,
                        net_return=net_ret
                    ))
                    
                    equity.append(equity[-1] * (1 + net_ret))
                    position = 0
                else:
                    # Mark to market
                    bar_ret = position * (prices[i] / prices[i-1] - 1)
                    equity.append(equity[-1] * (1 + bar_ret))
            else:
                # Check entry
                if signal > 0:
                    position = 1  # Long only
                    entry_idx = i
                    entry_price = prices[i]
                    entry_t = t_stat
                    equity.append(equity[-1])
                else:
                    equity.append(equity[-1])
        
        # Close open position
        if position != 0:
            exit_price = prices[-1]
            gross_ret = position * (exit_price / entry_price - 1)
            cost = (self.cost_bps + self.slippage_bps) / 10000
            net_ret = gross_ret - cost
            trades.append(Trade(
                ticker=ticker, timeframe=interval, entry_time=entry_idx,
                exit_time=n-1, entry_price=entry_price, exit_price=exit_price,
                direction=position, t_stat=entry_t, gross_return=gross_ret,
                net_return=net_ret
            ))
            equity.append(equity[-1] * (1 + net_ret))
        
        if not trades:
            return None
        
        equity = np.array(equity)
        trade_returns = np.array([t.net_return for t in trades])
        
        t_stat_result, p_value, ci = compute_t_test(trade_returns)
        
        return BacktestResult(
            name=f"{ticker}_{interval}",
            n_trades=len(trades),
            win_rate=np.mean(trade_returns > 0),
            mean_return=np.mean(trade_returns),
            std_return=np.std(trade_returns),
            sharpe=np.mean(trade_returns) / (np.std(trade_returns) + 1e-10) * np.sqrt(252 / holding),
            max_drawdown=np.max((np.maximum.accumulate(equity) - equity) / (np.maximum.accumulate(equity) + 1e-10)),
            total_return=equity[-1] / equity[0] - 1,
            t_stat=t_stat_result,
            p_value=p_value,
            ci_lower=ci[0],
            ci_upper=ci[1],
            trades=trades,
            equity_curve=equity
        )
    
    def backtest_multi_instrument(
        self,
        tickers: List[str],
        period: str = '2y',
        interval: str = '1d',
        select_top_n: int = None
    ) -> BacktestResult:
        """Backtest multiple instruments with signal selection."""
        
        print(f"\n  Fetching data for {len(tickers)} instruments...")
        
        # Fetch all data
        data = {}
        for ticker in tickers:
            prices = self._fetch_data(ticker, period, interval)
            if prices is not None and len(prices) > 100:
                data[ticker] = prices
        
        print(f"  Loaded {len(data)} instruments")
        
        if len(data) < 3:
            return None
        
        # Align data
        min_len = min(len(v) for v in data.values())
        for ticker in data:
            data[ticker] = data[ticker][-min_len:]
        
        n = min_len
        train_end = int(n * self.train_ratio)
        
        print(f"  Training: {train_end} bars, Testing: {n - train_end} bars")
        
        # Build basin stats for each instrument
        print("  Building basin statistics...")
        basin_stats_all = {}
        for ticker, prices in data.items():
            returns = np.diff(np.log(prices[:train_end]))
            returns = np.clip(returns, -0.2, 0.2)
            basin_stats_all[ticker] = self._build_basin_stats(returns, self.holding_period)
        
        # Walk-forward test
        print("  Running walk-forward test...")
        
        all_trades = []
        equity = [1.0]
        
        positions = {t: 0 for t in data}
        entries = {t: 0 for t in data}
        entry_prices = {t: 0.0 for t in data}
        entry_t_stats = {t: 0.0 for t in data}
        
        for i in range(train_end + 20, n - self.holding_period):
            # Get all signals
            signals = {}
            for ticker, prices in data.items():
                returns = np.diff(np.log(prices[:i+1]))
                returns = np.clip(returns, -0.2, 0.2)
                
                if len(returns) >= 20:
                    signal, t_stat = self._get_signal(returns[-20:], basin_stats_all[ticker])
                    if signal > 0:
                        signals[ticker] = t_stat
            
            # Select top N signals if specified
            if select_top_n and len(signals) > select_top_n:
                sorted_signals = sorted(signals.items(), key=lambda x: -x[1])
                signals = dict(sorted_signals[:select_top_n])
            
            daily_pnl = 0.0
            
            # Manage existing positions
            for ticker in data:
                prices = data[ticker]
                
                if positions[ticker] != 0:
                    if i - entries[ticker] >= self.holding_period:
                        exit_price = prices[i]
                        gross_ret = positions[ticker] * (exit_price / entry_prices[ticker] - 1)
                        cost = (self.cost_bps + self.slippage_bps) / 10000
                        net_ret = gross_ret - cost
                        
                        all_trades.append(Trade(
                            ticker=ticker, timeframe=interval,
                            entry_time=entries[ticker], exit_time=i,
                            entry_price=entry_prices[ticker], exit_price=exit_price,
                            direction=positions[ticker], t_stat=entry_t_stats[ticker],
                            gross_return=gross_ret, net_return=net_ret
                        ))
                        
                        daily_pnl += net_ret / max(len([t for t in data if positions[t] != 0]), 1)
                        positions[ticker] = 0
                    else:
                        bar_ret = positions[ticker] * (prices[i] / prices[i-1] - 1)
                        daily_pnl += bar_ret / max(len([t for t in data if positions[t] != 0]), 1)
            
            # New entries
            active_positions = sum(1 for t in data if positions[t] != 0)
            
            for ticker, t_stat in signals.items():
                if positions[ticker] == 0 and active_positions < self.max_positions:
                    positions[ticker] = 1
                    entries[ticker] = i
                    entry_prices[ticker] = data[ticker][i]
                    entry_t_stats[ticker] = t_stat
                    active_positions += 1
            
            equity.append(equity[-1] * (1 + daily_pnl))
        
        # Close remaining positions
        for ticker in data:
            if positions[ticker] != 0:
                prices = data[ticker]
                exit_price = prices[-1]
                gross_ret = positions[ticker] * (exit_price / entry_prices[ticker] - 1)
                cost = (self.cost_bps + self.slippage_bps) / 10000
                net_ret = gross_ret - cost
                all_trades.append(Trade(
                    ticker=ticker, timeframe=interval,
                    entry_time=entries[ticker], exit_time=n-1,
                    entry_price=entry_prices[ticker], exit_price=exit_price,
                    direction=positions[ticker], t_stat=entry_t_stats[ticker],
                    gross_return=gross_ret, net_return=net_ret
                ))
        
        if not all_trades:
            return None
        
        equity = np.array(equity)
        trade_returns = np.array([t.net_return for t in all_trades])
        
        t_stat_result, p_value, ci = compute_t_test(trade_returns)
        
        return BacktestResult(
            name=f"Multi_{len(data)}_{interval}" + (f"_top{select_top_n}" if select_top_n else ""),
            n_trades=len(all_trades),
            win_rate=np.mean(trade_returns > 0),
            mean_return=np.mean(trade_returns),
            std_return=np.std(trade_returns),
            sharpe=np.mean(trade_returns) / (np.std(trade_returns) + 1e-10) * np.sqrt(252 / self.holding_period),
            max_drawdown=np.max((np.maximum.accumulate(equity) - equity) / (np.maximum.accumulate(equity) + 1e-10)),
            total_return=equity[-1] / equity[0] - 1,
            t_stat=t_stat_result,
            p_value=p_value,
            ci_lower=ci[0],
            ci_upper=ci[1],
            trades=all_trades,
            equity_curve=equity
        )
    
    def analyze_by_t_stat(self, trades: List[Trade]) -> Dict:
        """Analyze performance by t-stat tier."""
        tiers = {
            'low (1.5-2)': [],
            'medium (2-3)': [],
            'high (3-5)': [],
            'max (5+)': []
        }
        
        for t in trades:
            if 1.5 <= t.t_stat < 2:
                tiers['low (1.5-2)'].append(t.net_return)
            elif 2 <= t.t_stat < 3:
                tiers['medium (2-3)'].append(t.net_return)
            elif 3 <= t.t_stat < 5:
                tiers['high (3-5)'].append(t.net_return)
            elif t.t_stat >= 5:
                tiers['max (5+)'].append(t.net_return)
        
        results = {}
        for tier, returns in tiers.items():
            if returns:
                returns = np.array(returns)
                results[tier] = {
                    'n': len(returns),
                    'win_rate': np.mean(returns > 0),
                    'mean': np.mean(returns),
                    't_stat': np.mean(returns) / (np.std(returns) / np.sqrt(len(returns)) + 1e-10) if len(returns) > 1 else 0
                }
            else:
                results[tier] = {'n': 0, 'win_rate': 0, 'mean': 0, 't_stat': 0}
        
        return results


def main():
    print("="*80)
    print("THOROUGH BACKTESTER — Comprehensive Validation")
    print("="*80)
    print(f"\nStart time: {datetime.now()}")
    
    backtester = ThoroughBacktester(
        train_ratio=0.5,
        min_basin_samples=10,
        min_t_stat=1.5,
        holding_period=5,
        cost_bps=10.0,  # Conservative
        slippage_bps=5.0,
        max_positions=20
    )
    
    # =================================================================
    # TEST 1: SINGLE INSTRUMENT BASELINE
    # =================================================================
    print("\n" + "="*60)
    print("TEST 1: SINGLE INSTRUMENT BASELINE (SPY)")
    print("="*60)
    
    result = backtester.backtest_single('SPY', period='5y', interval='1d')
    
    if result:
        print(f"\n  {result.name}:")
        print(f"    Trades: {result.n_trades}")
        print(f"    Win Rate: {result.win_rate*100:.1f}%")
        print(f"    Mean Return: {result.mean_return*100:.3f}%")
        print(f"    Sharpe: {result.sharpe:.2f}")
        print(f"    Total Return: {result.total_return*100:.1f}%")
        print(f"    Max Drawdown: {result.max_drawdown*100:.1f}%")
        print(f"    T-stat: {result.t_stat:.2f}")
        print(f"    P-value: {result.p_value:.4f}")
        print(f"    95% CI: [{result.ci_lower*100:.3f}%, {result.ci_upper*100:.3f}%]")
        print(f"    Significant: {'YES ✓' if result.p_value < 0.05 else 'NO'}")
    
    # =================================================================
    # TEST 2: MULTIPLE INSTRUMENTS (ALL SIGNALS)
    # =================================================================
    print("\n" + "="*60)
    print("TEST 2: MULTIPLE INSTRUMENTS (All Signals)")
    print("="*60)
    
    tickers = [
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV',
        'GLD', 'TLT', 'EEM', 'VNQ', 'HYG', 'AAPL', 'MSFT', 'GOOGL',
        'AMZN', 'META', 'NVDA', 'JPM', 'BAC', 'WFC', 'GS', 'MS',
        'XOM', 'CVX', 'COP', 'PFE', 'JNJ', 'UNH', 'MRK', 'ABBV'
    ]
    
    result_all = backtester.backtest_multi_instrument(tickers, period='3y')
    
    if result_all:
        print(f"\n  {result_all.name}:")
        print(f"    Trades: {result_all.n_trades}")
        print(f"    Win Rate: {result_all.win_rate*100:.1f}%")
        print(f"    Mean Return: {result_all.mean_return*100:.3f}%")
        print(f"    Sharpe: {result_all.sharpe:.2f}")
        print(f"    Total Return: {result_all.total_return*100:.1f}%")
        print(f"    Max Drawdown: {result_all.max_drawdown*100:.1f}%")
        print(f"    T-stat: {result_all.t_stat:.2f}")
        print(f"    P-value: {result_all.p_value:.4f}")
        print(f"    Significant: {'YES ✓' if result_all.p_value < 0.05 else 'NO'}")
        
        # Analyze by t-stat tier
        tier_analysis = backtester.analyze_by_t_stat(result_all.trades)
        
        print(f"\n  Performance by T-stat Tier:")
        print(f"  {'Tier':<15} {'N':>8} {'Win%':>10} {'Mean':>12} {'T-stat':>10}")
        print("  " + "-"*55)
        for tier, stats in tier_analysis.items():
            if stats['n'] > 0:
                print(f"  {tier:<15} {stats['n']:>8} {stats['win_rate']*100:>9.1f}% {stats['mean']*100:>11.3f}% {stats['t_stat']:>10.2f}")
    
    # =================================================================
    # TEST 3: SIGNAL SELECTION (TOP N)
    # =================================================================
    print("\n" + "="*60)
    print("TEST 3: SIGNAL SELECTION (Top N by T-stat)")
    print("="*60)
    
    selection_results = []
    
    for top_n in [5, 10, 20, None]:
        result_sel = backtester.backtest_multi_instrument(
            tickers, period='3y', select_top_n=top_n
        )
        if result_sel:
            selection_results.append(result_sel)
    
    print(f"\n  {'Selection':<20} {'Trades':>8} {'Win%':>8} {'Mean':>10} {'Sharpe':>8} {'P-val':>10}")
    print("  " + "-"*70)
    
    for result in selection_results:
        name = result.name.split('_')[-1] if 'top' in result.name else 'All'
        print(f"  {name:<20} {result.n_trades:>8} {result.win_rate*100:>7.1f}% {result.mean_return*100:>9.3f}% {result.sharpe:>8.2f} {result.p_value:>10.4f}")
    
    # =================================================================
    # TEST 4: MULTIPLE TIMEFRAMES
    # =================================================================
    print("\n" + "="*60)
    print("TEST 4: MULTIPLE TIMEFRAMES (SPY)")
    print("="*60)
    
    timeframes = [
        ('1d', '5y', 5),    # Daily, 5 years, 5-day holding
        ('1h', '2y', 8),    # Hourly, 2 years, 8-hour holding (1 day)
        ('15m', '60d', 8),  # 15min, 60 days, 8-bar holding (2 hours)
    ]
    
    tf_results = []
    
    for interval, period, holding in timeframes:
        result_tf = backtester.backtest_single('SPY', period=period, interval=interval, holding=holding)
        if result_tf:
            tf_results.append(result_tf)
    
    print(f"\n  {'Timeframe':<15} {'Trades':>8} {'Win%':>8} {'Mean':>10} {'Sharpe':>8} {'P-val':>10}")
    print("  " + "-"*65)
    
    for result in tf_results:
        print(f"  {result.name:<15} {result.n_trades:>8} {result.win_rate*100:>7.1f}% {result.mean_return*100:>9.3f}% {result.sharpe:>8.2f} {result.p_value:>10.4f}")
    
    # =================================================================
    # TEST 5: EXPANDED UNIVERSE
    # =================================================================
    print("\n" + "="*60)
    print("TEST 5: EXPANDED UNIVERSE (100 instruments)")
    print("="*60)
    
    expanded_tickers = [
        # ETFs
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLU',
        'XLP', 'XLY', 'GLD', 'TLT', 'EEM', 'VNQ', 'HYG', 'LQD', 'SMH', 'XBI',
        # Large Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'CRM',
        # Financials
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'SCHW', 'BLK',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY',
        # Consumer
        'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'DIS', 'CMCSA',
        # Industrials
        'CAT', 'DE', 'HON', 'UNP', 'UPS', 'BA', 'GE', 'MMM', 'LMT', 'RTX',
        # Crypto ETFs
        'COIN', 'MSTR',
        # International
        'EFA', 'VWO', 'FXI', 'EWJ', 'EWZ', 'EWY', 'INDA', 'VGK',
    ]
    
    result_expanded = backtester.backtest_multi_instrument(
        expanded_tickers, period='2y', select_top_n=20
    )
    
    if result_expanded:
        print(f"\n  {result_expanded.name}:")
        print(f"    Trades: {result_expanded.n_trades}")
        print(f"    Win Rate: {result_expanded.win_rate*100:.1f}%")
        print(f"    Mean Return: {result_expanded.mean_return*100:.3f}%")
        print(f"    Sharpe: {result_expanded.sharpe:.2f}")
        print(f"    Total Return: {result_expanded.total_return*100:.1f}%")
        print(f"    Max Drawdown: {result_expanded.max_drawdown*100:.1f}%")
        print(f"    T-stat: {result_expanded.t_stat:.2f}")
        print(f"    P-value: {result_expanded.p_value:.4f}")
        print(f"    Significant: {'YES ✓' if result_expanded.p_value < 0.05 else 'NO'}")
        
        tier_analysis = backtester.analyze_by_t_stat(result_expanded.trades)
        
        print(f"\n  Performance by T-stat Tier (Expanded Universe):")
        print(f"  {'Tier':<15} {'N':>8} {'Win%':>10} {'Mean':>12} {'T-stat':>10}")
        print("  " + "-"*55)
        for tier, stats in tier_analysis.items():
            if stats['n'] > 0:
                print(f"  {tier:<15} {stats['n']:>8} {stats['win_rate']*100:>9.1f}% {stats['mean']*100:>11.3f}% {stats['t_stat']:>10.2f}")
    
    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    all_results = [r for r in [result, result_all, result_expanded] if r]
    if selection_results:
        all_results.extend(selection_results)
    if tf_results:
        all_results.extend(tf_results)
    
    significant = [r for r in all_results if r.p_value < 0.05]
    profitable = [r for r in all_results if r.total_return > 0]
    
    print(f"\n  Total tests: {len(all_results)}")
    print(f"  Statistically significant (p<0.05): {len(significant)}/{len(all_results)}")
    print(f"  Profitable: {len(profitable)}/{len(all_results)}")
    
    if all_results:
        avg_wr = np.mean([r.win_rate for r in all_results])
        avg_sharpe = np.mean([r.sharpe for r in all_results])
        
        print(f"\n  Average win rate: {avg_wr*100:.1f}%")
        print(f"  Average Sharpe: {avg_sharpe:.2f}")
    
    # Key finding
    if result_expanded:
        print(f"\n  KEY FINDING — Expanded Universe with Top 20 Selection:")
        print(f"    Win Rate: {result_expanded.win_rate*100:.1f}%")
        print(f"    Mean Return: {result_expanded.mean_return*100:.3f}%")
        print(f"    Sharpe: {result_expanded.sharpe:.2f}")
        print(f"    P-value: {result_expanded.p_value:.4f}")
        
        if result_expanded.p_value < 0.05:
            print("\n  ✓ STRATEGY VALIDATED at p<0.05")
        else:
            print("\n  ~ Strategy shows promise but not statistically significant")
    
    print(f"\n  End time: {datetime.now()}")


if __name__ == "__main__":
    main()
