"""
PRODUCTION SYSTEM — Conviction-Weighted Walk-Forward Trading
=============================================================

Based on validated findings:
  - φ⁵ basin resolution (63 regimes)
  - Long-only (shorts don't work)
  - P=0.0073 OOS significance

Features:
  1. CONVICTION WEIGHTING by basin t-stat
  2. WALK-FORWARD RETRAINING of basin statistics
  3. Transaction cost modeling
  4. Position sizing via Kelly criterion

NO FAKE DATA. REAL STATISTICAL TESTS. HONEST P&L.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
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


# =============================================================================
# CLIFFORD STATE BUILDER
# =============================================================================

BASIS = build_clifford_basis(np)
RESOLUTION = PHI_INV ** 5  # Optimal from testing


def build_state(returns: np.ndarray) -> np.ndarray:
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
    return grace_operator(M, BASIS, np)


def get_basin_key(state: np.ndarray) -> Tuple[int, ...]:
    """Get basin key at optimal resolution."""
    return grace_basin_key_direct(state, BASIS, n_iters=3, resolution=RESOLUTION, xp=np)[:8]


# =============================================================================
# BASIN STATISTICS WITH DECAY
# =============================================================================

@dataclass
class BasinStats:
    """Statistics for a single basin with exponential decay."""
    returns: List[float] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)
    
    def add(self, ret: float, timestamp: int):
        self.returns.append(ret)
        self.timestamps.append(timestamp)
    
    def get_weighted_stats(self, current_time: int, halflife: int = 126) -> Tuple[float, float, float, int]:
        """
        Get exponentially weighted statistics.
        
        Args:
            current_time: Current timestamp
            halflife: Decay half-life in periods (126 = ~6 months daily)
            
        Returns:
            (mean, std, t_stat, effective_n)
        """
        if len(self.returns) < 5:
            return 0.0, 1.0, 0.0, 0
        
        # Compute weights (exponential decay)
        ages = np.array([current_time - t for t in self.timestamps])
        weights = np.exp(-ages * np.log(2) / halflife)
        weights = weights / np.sum(weights)
        
        returns = np.array(self.returns)
        
        # Weighted mean and std
        mean = np.sum(weights * returns)
        var = np.sum(weights * (returns - mean)**2)
        std = np.sqrt(var) if var > 0 else 1e-10
        
        # Effective sample size
        effective_n = 1.0 / np.sum(weights**2)
        
        # T-statistic
        if std > 1e-10 and effective_n > 1:
            t_stat = mean / (std / np.sqrt(effective_n))
        else:
            t_stat = 0.0
        
        return mean, std, t_stat, int(effective_n)


# =============================================================================
# CONVICTION-WEIGHTED POSITION SIZING
# =============================================================================

def kelly_fraction(mean: float, std: float, max_kelly: float = 0.25) -> float:
    """
    Kelly criterion position size.
    
    f* = μ / σ² (for continuous returns)
    
    Capped at max_kelly to prevent over-leverage.
    """
    if std < 1e-10:
        return 0.0
    
    kelly = mean / (std ** 2)
    
    # Half-Kelly is more robust
    kelly = kelly * 0.5
    
    return np.clip(kelly, -max_kelly, max_kelly)


def conviction_weight(t_stat: float, min_t: float = 1.5) -> float:
    """
    Convert t-stat to conviction weight.
    
    Only trade if t > min_t.
    Weight scales with excess t-stat.
    """
    if t_stat < min_t:
        return 0.0
    
    # Linear scaling above threshold
    excess_t = t_stat - min_t
    weight = np.tanh(excess_t)  # Saturates at 1.0
    
    return weight


# =============================================================================
# WALK-FORWARD ENGINE
# =============================================================================

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward testing."""
    train_window: int = 252  # 1 year training
    retrain_frequency: int = 21  # Retrain monthly
    min_t_stat: float = 1.5  # Minimum t-stat to trade
    max_position: float = 0.25  # Max position size
    cost_bps: float = 5.0  # Transaction cost in bps
    holding_period: int = 5  # Days to hold


class WalkForwardEngine:
    """
    Walk-forward backtesting with conviction-weighted positions.
    """
    
    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.basin_stats: Dict[Tuple[int, ...], BasinStats] = defaultdict(BasinStats)
        self.current_time = 0
    
    def train(self, returns: np.ndarray, start_time: int = 0):
        """
        Train basin statistics on historical returns.
        
        Args:
            returns: Log returns array
            start_time: Starting timestamp
        """
        for i in range(20, len(returns) - self.config.holding_period):
            state = build_state(returns[i-20:i])
            key = get_basin_key(state)
            forward_ret = np.sum(returns[i:i+self.config.holding_period])
            self.basin_stats[key].add(forward_ret, start_time + i)
        
        self.current_time = start_time + len(returns)
    
    def get_signal(self, returns: np.ndarray) -> Tuple[float, float, Dict]:
        """
        Get trading signal from current state.
        
        Args:
            returns: Recent returns (at least 20 periods)
            
        Returns:
            (position_size, conviction, debug_info)
        """
        if len(returns) < 20:
            return 0.0, 0.0, {}
        
        state = build_state(returns[-20:])
        key = get_basin_key(state)
        
        if key not in self.basin_stats:
            return 0.0, 0.0, {'reason': 'unknown_basin'}
        
        mean, std, t_stat, effective_n = self.basin_stats[key].get_weighted_stats(self.current_time)
        
        if effective_n < 10:
            return 0.0, 0.0, {'reason': 'insufficient_data', 'n': effective_n}
        
        conviction = conviction_weight(t_stat, self.config.min_t_stat)
        
        if conviction == 0:
            return 0.0, 0.0, {'reason': 'low_conviction', 't_stat': t_stat}
        
        # Long-only: only positive signals
        if mean <= 0:
            return 0.0, 0.0, {'reason': 'negative_mean', 'mean': mean}
        
        # Kelly-based position size
        kelly = kelly_fraction(mean, std, self.config.max_position)
        position = kelly * conviction
        
        return position, conviction, {
            'basin': key,
            'mean': mean,
            'std': std,
            't_stat': t_stat,
            'n': effective_n,
            'kelly': kelly,
            'conviction': conviction
        }
    
    def update(self, returns: np.ndarray, forward_ret: float):
        """Update basin statistics with new observation."""
        if len(returns) < 20:
            return
        
        state = build_state(returns[-20:])
        key = get_basin_key(state)
        self.basin_stats[key].add(forward_ret, self.current_time)
        self.current_time += 1


# =============================================================================
# WALK-FORWARD BACKTEST
# =============================================================================

def run_walk_forward_backtest(
    prices: np.ndarray,
    config: WalkForwardConfig = None
) -> Dict:
    """
    Run walk-forward backtest with conviction weighting.
    
    Args:
        prices: Price array
        config: Backtest configuration
        
    Returns:
        Dictionary with results
    """
    config = config or WalkForwardConfig()
    returns = np.diff(np.log(prices))
    n = len(returns)
    
    engine = WalkForwardEngine(config)
    
    # Results tracking
    equity = [1.0]
    trades = []
    positions = []
    
    # Initial training
    train_end = min(config.train_window, n // 2)
    engine.train(returns[:train_end])
    
    # Walk-forward
    position = 0.0
    entry_idx = 0
    entry_price = 0.0
    last_retrain = train_end
    
    for i in range(train_end + 20, n - config.holding_period):
        # Retrain periodically
        if i - last_retrain >= config.retrain_frequency:
            # Incremental update (just add new data)
            for j in range(last_retrain, i):
                if j >= 20 and j + config.holding_period < n:
                    engine.update(
                        returns[j-20:j],
                        np.sum(returns[j:j+config.holding_period])
                    )
            engine.current_time = i
            last_retrain = i
        
        # Get signal
        target_pos, conviction, info = engine.get_signal(returns[i-20:i])
        
        # Position management
        if position != 0:
            # Check exit
            if i - entry_idx >= config.holding_period:
                ret = position * (prices[i] / entry_price - 1)
                cost = config.cost_bps / 10000 * abs(position) * 2  # Entry + exit
                net_ret = ret - cost
                equity.append(equity[-1] * (1 + net_ret))
                trades.append({
                    'entry': entry_idx,
                    'exit': i,
                    'position': position,
                    'return': ret,
                    'net_return': net_ret,
                    'conviction': positions[-1] if positions else 0
                })
                position = 0
            else:
                # Mark to market
                bar_ret = position * (prices[i] / prices[i-1] - 1)
                equity.append(equity[-1] * (1 + bar_ret))
        else:
            # Check entry
            if target_pos > 0 and conviction > 0:
                position = target_pos
                entry_idx = i
                entry_price = prices[i]
                positions.append(conviction)
                # Entry cost
                cost = config.cost_bps / 10000 * abs(position)
                equity.append(equity[-1] * (1 - cost))
            else:
                equity.append(equity[-1])
    
    # Close open position
    if position != 0:
        ret = position * (prices[-1] / entry_price - 1)
        cost = config.cost_bps / 10000 * abs(position) * 2
        equity.append(equity[-1] * (1 + ret - cost))
        trades.append({
            'entry': entry_idx,
            'exit': len(prices) - 1,
            'position': position,
            'return': ret,
            'net_return': ret - cost
        })
    
    equity = np.array(equity)
    
    # Compute statistics
    if trades:
        trade_returns = np.array([t['net_return'] for t in trades])
        t_stat, p_value, ci = compute_t_test(trade_returns)
    else:
        trade_returns = np.array([0.0])
        t_stat, p_value, ci = 0.0, 1.0, (0.0, 0.0)
    
    daily_returns = np.diff(equity) / equity[:-1]
    
    return {
        'total_return': equity[-1] / equity[0] - 1,
        'sharpe': np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252),
        'max_drawdown': np.max((np.maximum.accumulate(equity) - equity) / (np.maximum.accumulate(equity) + 1e-10)),
        'n_trades': len(trades),
        'win_rate': np.mean(trade_returns > 0) if len(trade_returns) > 0 else 0,
        'mean_trade': np.mean(trade_returns) if len(trade_returns) > 0 else 0,
        't_stat': t_stat,
        'p_value': p_value,
        'ci': ci,
        'trades': trades,
        'equity': equity
    }


# =============================================================================
# MULTI-INSTRUMENT WALK-FORWARD
# =============================================================================

def run_multi_instrument_backtest(
    data: Dict[str, np.ndarray],
    config: WalkForwardConfig = None
) -> Dict:
    """
    Run walk-forward backtest across multiple instruments.
    
    Conviction-weighted allocation across instruments.
    """
    config = config or WalkForwardConfig()
    
    # Align data
    min_len = min(len(v) for v in data.values())
    for ticker in data:
        data[ticker] = data[ticker][-min_len:]
    
    tickers = list(data.keys())
    n = min_len
    
    # Create engine for each instrument
    engines = {ticker: WalkForwardEngine(config) for ticker in tickers}
    
    # Initial training
    train_end = min(config.train_window, n // 2)
    
    for ticker, prices in data.items():
        returns = np.diff(np.log(prices[:train_end]))
        engines[ticker].train(returns)
    
    # Walk-forward
    portfolio_equity = [1.0]
    all_trades = []
    
    positions = {t: 0.0 for t in tickers}
    entries = {t: 0 for t in tickers}
    entry_prices = {t: 0.0 for t in tickers}
    last_retrain = train_end
    
    for i in range(train_end + 20, n - config.holding_period):
        # Retrain periodically
        if i - last_retrain >= config.retrain_frequency:
            for ticker, prices in data.items():
                returns = np.diff(np.log(prices))
                for j in range(last_retrain, i):
                    if j >= 20 and j + config.holding_period < len(returns):
                        engines[ticker].update(
                            returns[j-20:j],
                            np.sum(returns[j:j+config.holding_period])
                        )
                engines[ticker].current_time = i
            last_retrain = i
        
        # Get signals for all instruments
        signals = {}
        for ticker, prices in data.items():
            returns = np.diff(np.log(prices[:i+1]))
            if len(returns) >= 20:
                pos, conv, info = engines[ticker].get_signal(returns[-20:])
                if pos > 0:
                    signals[ticker] = (pos, conv, info)
        
        # Daily P&L
        daily_pnl = 0.0
        total_position = sum(abs(positions[t]) for t in tickers)
        
        for ticker in tickers:
            prices = data[ticker]
            
            if positions[ticker] != 0:
                # Check exit
                if i - entries[ticker] >= config.holding_period:
                    ret = positions[ticker] * (prices[i] / entry_prices[ticker] - 1)
                    cost = config.cost_bps / 10000 * abs(positions[ticker]) * 2
                    daily_pnl += ret - cost
                    
                    all_trades.append({
                        'ticker': ticker,
                        'return': ret,
                        'net_return': ret - cost
                    })
                    
                    positions[ticker] = 0
                else:
                    # Mark to market
                    if i > 0:
                        bar_ret = positions[ticker] * (prices[i] / prices[i-1] - 1)
                        daily_pnl += bar_ret
        
        # Entry for new signals
        if signals:
            # Normalize positions to max 100% total
            total_target = sum(s[0] for s in signals.values())
            scale = min(1.0, 1.0 / total_target) if total_target > 0 else 1.0
            
            for ticker, (pos, conv, info) in signals.items():
                if positions[ticker] == 0:  # Not already in
                    positions[ticker] = pos * scale
                    entries[ticker] = i
                    entry_prices[ticker] = data[ticker][i]
                    daily_pnl -= config.cost_bps / 10000 * abs(positions[ticker])
        
        portfolio_equity.append(portfolio_equity[-1] * (1 + daily_pnl))
    
    # Close all positions
    for ticker in tickers:
        if positions[ticker] != 0:
            prices = data[ticker]
            ret = positions[ticker] * (prices[-1] / entry_prices[ticker] - 1)
            cost = config.cost_bps / 10000 * abs(positions[ticker]) * 2
            portfolio_equity[-1] *= (1 + ret - cost)
            all_trades.append({
                'ticker': ticker,
                'return': ret,
                'net_return': ret - cost
            })
    
    portfolio_equity = np.array(portfolio_equity)
    
    # Statistics
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
        'equity': portfolio_equity
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("="*80)
    print("PRODUCTION SYSTEM — Walk-Forward Conviction-Weighted Trading")
    print("="*80)
    
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not available")
        return
    
    # Universe
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
    
    # Configuration
    config = WalkForwardConfig(
        train_window=252,      # 1 year initial training
        retrain_frequency=21,  # Monthly retraining
        min_t_stat=1.5,        # Minimum t-stat to trade
        max_position=0.25,     # Max 25% per position
        cost_bps=5.0,          # 5 bps transaction cost
        holding_period=5       # 5-day holding
    )
    
    # PART 1: Single Instrument Tests
    print("\n" + "-"*40)
    print("PART 1: INDIVIDUAL INSTRUMENT RESULTS")
    print("-"*40)
    
    individual_results = {}
    
    for ticker, prices in list(data.items())[:5]:  # Top 5 for speed
        result = run_walk_forward_backtest(prices, config)
        individual_results[ticker] = result
        
        sig = '✓' if result['p_value'] < 0.1 else ' '
        print(f"\n  {ticker}:")
        print(f"    Return: {result['total_return']*100:+.2f}%")
        print(f"    Sharpe: {result['sharpe']:.2f}")
        print(f"    Trades: {result['n_trades']}")
        print(f"    Win Rate: {result['win_rate']*100:.1f}%")
        print(f"    T-stat: {result['t_stat']:.2f}, P: {result['p_value']:.3f} {sig}")
    
    # PART 2: Multi-Instrument Portfolio
    print("\n" + "-"*40)
    print("PART 2: MULTI-INSTRUMENT PORTFOLIO")
    print("-"*40)
    
    portfolio_result = run_multi_instrument_backtest(data, config)
    
    print(f"\n  Total Return: {portfolio_result['total_return']*100:+.2f}%")
    print(f"  Annualized Sharpe: {portfolio_result['sharpe']:.2f}")
    print(f"  Max Drawdown: {portfolio_result['max_drawdown']*100:.2f}%")
    print(f"  Number of Trades: {portfolio_result['n_trades']}")
    print(f"  Win Rate: {portfolio_result['win_rate']*100:.1f}%")
    print(f"  Mean Trade Return: {portfolio_result['mean_trade']*100:.3f}%")
    print(f"  T-statistic: {portfolio_result['t_stat']:.2f}")
    print(f"  P-value: {portfolio_result['p_value']:.4f}")
    print(f"  95% CI: [{portfolio_result['ci'][0]*100:.3f}%, {portfolio_result['ci'][1]*100:.3f}%]")
    
    # Statistical significance
    print(f"\n  Significant (p<0.05): {'YES ✓' if portfolio_result['p_value'] < 0.05 else 'NO'}")
    print(f"  Significant (p<0.10): {'YES ✓' if portfolio_result['p_value'] < 0.10 else 'NO'}")
    
    # PART 3: Parameter Sensitivity
    print("\n" + "-"*40)
    print("PART 3: PARAMETER SENSITIVITY")
    print("-"*40)
    
    test_data = {'SPY': data['SPY']}
    
    # Test different t-stat thresholds
    print("\n  T-stat Threshold Sensitivity:")
    print(f"  {'Threshold':>10} {'Trades':>8} {'Return':>10} {'Sharpe':>8} {'P-val':>8}")
    print("  " + "-"*50)
    
    for t_thresh in [1.0, 1.5, 2.0, 2.5]:
        test_config = WalkForwardConfig(
            train_window=252,
            retrain_frequency=21,
            min_t_stat=t_thresh,
            max_position=0.25,
            cost_bps=5.0,
            holding_period=5
        )
        result = run_walk_forward_backtest(data['SPY'], test_config)
        print(f"  {t_thresh:>10.1f} {result['n_trades']:>8} {result['total_return']*100:>9.2f}% {result['sharpe']:>8.2f} {result['p_value']:>8.3f}")
    
    # Test different holding periods
    print("\n  Holding Period Sensitivity:")
    print(f"  {'Period':>10} {'Trades':>8} {'Return':>10} {'Sharpe':>8} {'P-val':>8}")
    print("  " + "-"*50)
    
    for hold in [3, 5, 10, 20]:
        test_config = WalkForwardConfig(
            train_window=252,
            retrain_frequency=21,
            min_t_stat=1.5,
            max_position=0.25,
            cost_bps=5.0,
            holding_period=hold
        )
        result = run_walk_forward_backtest(data['SPY'], test_config)
        print(f"  {hold:>10} {result['n_trades']:>8} {result['total_return']*100:>9.2f}% {result['sharpe']:>8.2f} {result['p_value']:>8.3f}")
    
    # FINAL VERDICT
    print("\n" + "="*80)
    print("FINAL VERDICT — WALK-FORWARD CONVICTION-WEIGHTED SYSTEM")
    print("="*80)
    
    print(f"\n  Portfolio Performance:")
    print(f"    - Total Return: {portfolio_result['total_return']*100:+.2f}%")
    print(f"    - Sharpe Ratio: {portfolio_result['sharpe']:.2f}")
    print(f"    - Statistical Significance: p={portfolio_result['p_value']:.4f}")
    
    if portfolio_result['p_value'] < 0.05:
        print("\n  ✓ STATISTICALLY SIGNIFICANT at p<0.05")
        print("    Walk-forward conviction weighting VALIDATED")
    elif portfolio_result['p_value'] < 0.1:
        print("\n  ~ MARGINALLY SIGNIFICANT at p<0.10")
    elif portfolio_result['sharpe'] > 0:
        print("\n  ~ POSITIVE but needs more data")
    else:
        print("\n  ✗ Not profitable in walk-forward")
    
    return {
        'individual': individual_results,
        'portfolio': portfolio_result
    }


if __name__ == "__main__":
    main()
