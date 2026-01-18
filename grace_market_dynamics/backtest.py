"""
Backtest Framework — Theory-True P&L Validation
================================================

Out-of-sample backtesting with:
- Transaction costs
- Position sizing based on coherence
- Proper train/test splits
- Multiple performance metrics

NO look-ahead bias, NO overfitting.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.algebra import (
    build_clifford_basis,
    geometric_product_batch,
    grace_basin_key_direct,
    frobenius_cosine,
    reconstruct_from_coefficients,
)
from holographic_prod.core.constants import PHI_INV, PHI_INV_SQ, PHI_INV_SIX
from grace_market_dynamics.state_encoder import CliffordState, ObservableAgnosticEncoder
from holographic_prod.core.algebra import get_cached_basis


class Position(Enum):
    """Trading position."""
    LONG = 1
    FLAT = 0
    SHORT = -1


@dataclass
class Trade:
    """Single trade record."""
    entry_time: int
    exit_time: int
    position: Position
    entry_price: float
    exit_price: float
    size: float
    signal_strength: float
    
    @property
    def pnl(self) -> float:
        """Gross P&L (before costs)."""
        return self.position.value * (self.exit_price - self.entry_price) * self.size
    
    @property
    def return_pct(self) -> float:
        """Return percentage."""
        return self.position.value * (self.exit_price / self.entry_price - 1)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    # Transaction costs
    commission_bps: float = 1.0  # Commission per side in basis points
    slippage_bps: float = 1.0   # Slippage per side in basis points
    
    # Position sizing
    base_position_size: float = 1.0
    scale_by_coherence: bool = True
    max_position_size: float = 2.0
    
    # Entry/Exit thresholds
    coherence_threshold: float = 0.6
    min_signal_strength: float = 0.3
    
    # Risk management
    max_holding_periods: int = 20
    stop_loss_pct: float = 0.02  # 2% stop loss
    
    # Train/test split
    train_fraction: float = 0.6
    
    # Encoder settings
    gram_window: int = 20
    delay_embed_dim: int = 4


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Performance metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    
    # Trade statistics
    n_trades: int = 0
    n_winning: int = 0
    n_losing: int = 0
    avg_holding_period: float = 0.0
    
    # Signal quality
    avg_signal_strength: float = 0.0
    avg_coherence: float = 0.0


class CliffordBacktester:
    """
    Theory-true backtester using Clifford algebra signals.
    
    Signal generation:
    1. Encode market state as Clifford multivector (direct from_increments)
    2. Track chirality (pseudoscalar sign)
    3. Measure coherence (stability)
    4. Generate position based on chirality + coherence
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.basis = get_cached_basis(np)
        self.return_history: List[float] = []
        
    def _compute_transaction_cost(self, price: float, size: float) -> float:
        """Compute total transaction cost for entry or exit."""
        cost_bps = self.config.commission_bps + self.config.slippage_bps
        return price * size * cost_bps / 10000
    
    def _get_signal(self, state: CliffordState) -> Tuple[float, float]:
        """
        Extract trading signal from Clifford state.
        
        Returns:
            (signal, coherence) where:
            - signal: -1 to +1 (direction and strength)
            - coherence: 0 to 1 (confidence)
        """
        # Chirality = pseudoscalar sign × magnitude
        chirality = state.pseudoscalar
        
        # Coherence from grade structure
        mags = state.grade_magnitudes()
        total = sum(mags.values()) + 1e-10
        
        # Coherence = how much energy is in stable grades (G0 + G4)
        witness_energy = mags['G0'] + mags['G4']
        coherence = witness_energy / total
        
        # Signal strength from chirality normalized by vector energy
        signal = np.sign(chirality) * min(abs(chirality) / (mags['G1'] + 1e-10), 1.0)
        
        return signal, coherence
    
    def _compute_position_size(self, coherence: float, signal_strength: float) -> float:
        """Compute position size based on signal quality."""
        base = self.config.base_position_size
        
        if self.config.scale_by_coherence:
            # Scale by coherence: higher coherence = larger position
            scale = coherence / self.config.coherence_threshold
            size = base * min(scale, self.config.max_position_size / base)
        else:
            size = base
        
        # Also scale by signal strength
        size *= abs(signal_strength)
        
        return min(size, self.config.max_position_size)
    
    def run(
        self,
        prices: np.ndarray,
        train_only: bool = False
    ) -> BacktestResult:
        """
        Run backtest on price series.
        
        Args:
            prices: Array of prices (close prices)
            train_only: If True, only use training period
            
        Returns:
            BacktestResult with all metrics
        """
        prices = np.asarray(prices).flatten()
        n = len(prices)
        
        # Train/test split
        split_idx = int(n * self.config.train_fraction)
        if train_only:
            prices = prices[:split_idx]
            n = len(prices)
        
        # Reset return history
        self.return_history = []
        
        # Results tracking
        trades: List[Trade] = []
        equity = [1.0]  # Starting capital normalized to 1
        
        # State tracking
        current_position = Position.FLAT
        entry_time = 0
        entry_price = 0.0
        position_size = 0.0
        entry_signal = 0.0
        holding_periods = 0
        
        signals = []
        coherences = []
        
        # Compute log returns
        log_returns = np.diff(np.log(prices))
        
        # Warm up period (need 4 returns for encoding)
        warmup_end = max(5, split_idx if not train_only else 10)
        
        # Main backtest loop
        start_idx = warmup_end if train_only else max(split_idx, 5)
        
        for i in range(start_idx, n - 1):
            # Get last 4 returns for encoding (i-1 is the latest complete return)
            if i < 4:
                equity.append(equity[-1])
                continue
            
            increments = log_returns[i-4:i]
            if len(increments) < 4:
                equity.append(equity[-1])
                continue
                
            # Create Clifford state directly
            state = CliffordState.from_increments(increments, self.basis)
            
            # Get signal
            signal, coherence = self._get_signal(state)
            signals.append(signal)
            coherences.append(coherence)
            
            # Current price
            price = prices[i]
            next_price = prices[i + 1]
            
            # Position management
            if current_position != Position.FLAT:
                holding_periods += 1
                
                # Check exit conditions
                should_exit = False
                
                # Max holding period
                if holding_periods >= self.config.max_holding_periods:
                    should_exit = True
                
                # Stop loss
                unrealized_return = current_position.value * (price / entry_price - 1)
                if unrealized_return < -self.config.stop_loss_pct:
                    should_exit = True
                
                # Signal reversal
                if signal * current_position.value < -self.config.min_signal_strength:
                    should_exit = True
                
                if should_exit:
                    # Exit trade
                    exit_cost_pct = (self.config.commission_bps + self.config.slippage_bps) / 10000
                    
                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=i,
                        position=current_position,
                        entry_price=entry_price,
                        exit_price=price,
                        size=position_size,
                        signal_strength=entry_signal
                    )
                    trades.append(trade)
                    
                    # Update equity using returns (not dollar P&L)
                    trade_return = trade.return_pct * position_size - exit_cost_pct
                    equity.append(equity[-1] * (1 + trade_return))
                    
                    # Reset position
                    current_position = Position.FLAT
                    holding_periods = 0
                else:
                    # Mark-to-market using returns
                    bar_return = current_position.value * (next_price / price - 1) * position_size
                    equity.append(equity[-1] * (1 + bar_return))
            
            else:
                # Check entry conditions
                if (coherence >= self.config.coherence_threshold and 
                    abs(signal) >= self.config.min_signal_strength):
                    
                    # Enter position
                    position_size = self._compute_position_size(coherence, signal)
                    entry_cost_pct = (self.config.commission_bps + self.config.slippage_bps) / 10000
                    
                    current_position = Position.LONG if signal > 0 else Position.SHORT
                    entry_time = i
                    entry_price = price
                    entry_signal = signal
                    holding_periods = 0
                    
                    # Deduct entry cost (percentage of position)
                    equity.append(equity[-1] * (1 - entry_cost_pct * position_size))
                else:
                    equity.append(equity[-1])
        
        # Close any open position at end
        if current_position != Position.FLAT:
            exit_cost_pct = (self.config.commission_bps + self.config.slippage_bps) / 10000
            trade = Trade(
                entry_time=entry_time,
                exit_time=n - 1,
                position=current_position,
                entry_price=entry_price,
                exit_price=prices[-1],
                size=position_size,
                signal_strength=entry_signal
            )
            trades.append(trade)
            trade_return = trade.return_pct * position_size - exit_cost_pct
            equity.append(equity[-1] * (1 + trade_return))
        
        # Compute metrics
        result = self._compute_metrics(trades, np.array(equity), signals, coherences)
        return result
    
    def _compute_metrics(
        self,
        trades: List[Trade],
        equity: np.ndarray,
        signals: List[float],
        coherences: List[float]
    ) -> BacktestResult:
        """Compute all performance metrics."""
        result = BacktestResult(
            trades=trades,
            equity_curve=equity
        )
        
        if len(equity) < 2:
            return result
        
        # Returns
        returns = np.diff(equity) / equity[:-1]
        result.total_return = (equity[-1] / equity[0] - 1)
        
        # Sharpe ratio (annualized, assuming daily)
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        result.max_drawdown = np.max(drawdown)
        
        # Trade statistics
        result.n_trades = len(trades)
        if trades:
            pnls = [t.pnl for t in trades]
            winning = [p for p in pnls if p > 0]
            losing = [p for p in pnls if p <= 0]
            
            result.n_winning = len(winning)
            result.n_losing = len(losing)
            result.win_rate = len(winning) / len(trades)
            result.avg_trade_pnl = np.mean(pnls)
            
            # Profit factor
            gross_profit = sum(winning) if winning else 0
            gross_loss = abs(sum(losing)) if losing else 1e-10
            result.profit_factor = gross_profit / gross_loss
            
            # Average holding period
            result.avg_holding_period = np.mean([t.exit_time - t.entry_time for t in trades])
        
        # Signal quality
        if signals:
            result.avg_signal_strength = np.mean(np.abs(signals))
        if coherences:
            result.avg_coherence = np.mean(coherences)
        
        return result


def run_backtest_suite(
    prices: np.ndarray,
    configs: List[BacktestConfig] = None
) -> Dict[str, BacktestResult]:
    """
    Run multiple backtests with different configurations.
    
    Args:
        prices: Price series
        configs: List of configurations to test
        
    Returns:
        Dictionary of config_name -> result
    """
    if configs is None:
        configs = [
            BacktestConfig(coherence_threshold=0.5, min_signal_strength=0.2),
            BacktestConfig(coherence_threshold=0.6, min_signal_strength=0.3),
            BacktestConfig(coherence_threshold=0.7, min_signal_strength=0.4),
        ]
    
    results = {}
    for i, config in enumerate(configs):
        name = f"config_{i}_coh{config.coherence_threshold}_sig{config.min_signal_strength}"
        backtester = CliffordBacktester(config)
        results[name] = backtester.run(prices)
    
    return results


def print_backtest_report(result: BacktestResult, name: str = "Backtest"):
    """Print formatted backtest report."""
    print(f"\n{'='*60}")
    print(f"{name} Results")
    print(f"{'='*60}")
    
    print(f"\n  Performance:")
    print(f"    Total Return: {result.total_return*100:.2f}%")
    print(f"    Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"    Max Drawdown: {result.max_drawdown*100:.2f}%")
    
    print(f"\n  Trade Statistics:")
    print(f"    Number of Trades: {result.n_trades}")
    print(f"    Win Rate: {result.win_rate*100:.1f}%")
    print(f"    Profit Factor: {result.profit_factor:.2f}")
    print(f"    Avg Trade P&L: {result.avg_trade_pnl*100:.4f}%")
    print(f"    Avg Holding Period: {result.avg_holding_period:.1f} bars")
    
    print(f"\n  Signal Quality:")
    print(f"    Avg Signal Strength: {result.avg_signal_strength:.4f}")
    print(f"    Avg Coherence: {result.avg_coherence:.4f}")


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Backtest Framework...")
    
    np.random.seed(42)
    n = 500
    
    # Generate synthetic trending + mean-reverting data
    trend = np.cumsum(np.random.randn(n) * 0.01 + 0.0005)  # Slight upward drift
    noise = np.random.randn(n) * 0.005
    prices = 100 * np.exp(trend + noise)
    
    # Run backtest
    config = BacktestConfig(
        coherence_threshold=0.4,
        min_signal_strength=0.2,
        train_fraction=0.5
    )
    
    backtester = CliffordBacktester(config)
    result = backtester.run(prices)
    
    print_backtest_report(result, "Synthetic Data Backtest")
    print("\n✓ Backtest framework working")
