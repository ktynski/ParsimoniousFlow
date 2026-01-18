"""
Intraday Data Test — Trading-Relevant Validation
=================================================

Tests the theory-true Clifford market dynamics on intraday (minute/hourly) data
where geometric structure should be more pronounced than on daily data.

Theory validation:
1. Higher-frequency data has more geometric structure (trend planes persist)
2. Chirality flips correlate with actual regime changes
3. Grace basin keys cluster by market microstructure
4. Multi-scale coherence identifies good trading windows
5. Vorticity spikes precede volatility events

Uses Yahoo Finance for real market data.
"""

import numpy as np
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not installed. Run: pip install yfinance")

from holographic_prod.core.algebra import (
    build_clifford_basis,
    geometric_product_batch,
    decompose_to_coefficients,
    reconstruct_from_coefficients,
    grace_operator,
    grace_basin_key_direct,
    frobenius_cosine,
    vorticity_magnitude_and_signature,
)
from holographic_prod.core.constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_SIX, GRADE_INDICES
)
from grace_market_dynamics.state_encoder import CliffordState, ObservableAgnosticEncoder


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_intraday_data(ticker: str, period: str = "5d", interval: str = "5m") -> Optional[np.ndarray]:
    """
    Fetch intraday data from Yahoo Finance.
    
    Args:
        ticker: Stock symbol (e.g., "SPY", "AAPL")
        period: Time period ("1d", "5d", "1mo")
        interval: Data interval ("1m", "5m", "15m", "1h")
        
    Returns:
        Array of close prices, or None if failed
    """
    if not HAS_YFINANCE:
        return None
    
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if len(data) < 50:
            print(f"Warning: Only {len(data)} data points for {ticker}")
            return None
        # Ensure we return a numpy array, not a pandas Series
        prices = data['Close'].values.flatten().astype(np.float64)
        return prices
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


# =============================================================================
# Theory-True Analysis Functions
# =============================================================================

def analyze_chirality_regime_correlation(
    prices: np.ndarray,
    basis: np.ndarray,
    window: int = 5,
    lookahead: int = 5
) -> Dict[str, float]:
    """
    Test if chirality sign correlates with subsequent returns.
    
    Theory: Consistent chirality sign (pseudoscalar) indicates directional bias
    that tends to persist.
    
    Returns:
        Correlation statistics
    """
    prices = np.asarray(prices).flatten()
    log_returns = np.diff(np.log(prices))
    
    chiralities = []
    future_returns = []
    
    for i in range(window, len(log_returns) - lookahead):
        # Get chirality from current window
        increments = log_returns[i-window:i]
        if len(increments) >= 4:
            state = CliffordState.from_increments(increments[-4:], basis)
            chirality = np.sign(state.pseudoscalar)
            
            # Get future return
            future_ret = np.sum(log_returns[i:i+lookahead])
            
            chiralities.append(chirality)
            future_returns.append(future_ret)
    
    chiralities = np.array(chiralities)
    future_returns = np.array(future_returns)
    
    # Compute correlation statistics
    pos_chirality_mask = chiralities > 0
    neg_chirality_mask = chiralities < 0
    
    pos_future = future_returns[pos_chirality_mask].mean() if pos_chirality_mask.sum() > 0 else 0
    neg_future = future_returns[neg_chirality_mask].mean() if neg_chirality_mask.sum() > 0 else 0
    
    # Information coefficient (correlation)
    if len(chiralities) > 10:
        ic = np.corrcoef(chiralities, future_returns)[0, 1]
    else:
        ic = 0.0
    
    return {
        'n_samples': len(chiralities),
        'pos_chirality_count': int(pos_chirality_mask.sum()),
        'neg_chirality_count': int(neg_chirality_mask.sum()),
        'pos_chirality_future_return': pos_future,
        'neg_chirality_future_return': neg_future,
        'information_coefficient': ic if not np.isnan(ic) else 0.0,
        'signal_value': pos_future - neg_future  # Spread between regimes
    }


def analyze_bivector_trend_correlation(
    prices: np.ndarray,
    basis: np.ndarray,
    window: int = 5,
    stability_threshold: int = 3
) -> Dict[str, float]:
    """
    Test if stable bivector plane correlates with trend persistence.
    
    Theory: When the dominant rotation plane is stable for multiple steps,
    the market is in a consistent regime.
    """
    prices = np.asarray(prices).flatten()
    log_returns = np.diff(np.log(prices))
    bivector_indices = GRADE_INDICES[2]
    
    def get_dominant_bivector(coeffs):
        return np.argmax(np.abs(coeffs[bivector_indices]))
    
    # Track bivector stability windows
    stable_windows = []
    unstable_windows = []
    
    dominant_history = []
    
    for i in range(window, len(log_returns) - 1):
        increments = log_returns[i-window:i]
        if len(increments) >= 4:
            state = CliffordState.from_increments(increments[-4:], basis)
            dominant = get_dominant_bivector(state.coeffs)
            dominant_history.append(dominant)
            
            # Check stability (same dominant for last N steps)
            if len(dominant_history) >= stability_threshold:
                recent = dominant_history[-stability_threshold:]
                is_stable = len(set(recent)) == 1
                
                # Get forward return
                forward_return = log_returns[i] if i < len(log_returns) else 0
                
                if is_stable:
                    stable_windows.append(forward_return)
                else:
                    unstable_windows.append(forward_return)
    
    stable_abs_mean = np.mean(np.abs(stable_windows)) if stable_windows else 0
    unstable_abs_mean = np.mean(np.abs(unstable_windows)) if unstable_windows else 0
    
    return {
        'n_stable_windows': len(stable_windows),
        'n_unstable_windows': len(unstable_windows),
        'stable_mean_abs_return': stable_abs_mean,
        'unstable_mean_abs_return': unstable_abs_mean,
        'stability_ratio': len(stable_windows) / max(len(stable_windows) + len(unstable_windows), 1),
        'predictability_edge': stable_abs_mean - unstable_abs_mean
    }


def analyze_vorticity_volatility_correlation(
    prices: np.ndarray,
    basis: np.ndarray,
    window: int = 10,
    lookahead: int = 5
) -> Dict[str, float]:
    """
    Test if vorticity spikes precede volatility events.
    
    Theory: High vorticity (rotational intensity) indicates regime instability
    that manifests as increased volatility.
    """
    prices = np.asarray(prices).flatten()
    log_returns = np.diff(np.log(prices))
    
    vorticity_mags = []
    future_volatilities = []
    
    for i in range(window, len(log_returns) - lookahead):
        # Build state sequence
        states = []
        for j in range(i - window, i):
            if j >= 3:
                inc = log_returns[j-3:j+1]
                state = CliffordState.from_increments(inc, basis)
                M = reconstruct_from_coefficients(state.coeffs, basis, np)
                states.append(M)
        
        if len(states) >= 5:
            states_arr = np.array(states)
            vort_mag, _ = vorticity_magnitude_and_signature(states_arr, basis, np)
            
            # Future volatility (realized)
            future_vol = np.std(log_returns[i:i+lookahead])
            
            vorticity_mags.append(vort_mag)
            future_volatilities.append(future_vol)
    
    vorticity_mags = np.array(vorticity_mags)
    future_volatilities = np.array(future_volatilities)
    
    # Correlation
    if len(vorticity_mags) > 10:
        corr = np.corrcoef(vorticity_mags, future_volatilities)[0, 1]
    else:
        corr = 0.0
    
    # Quartile analysis
    if len(vorticity_mags) > 20:
        q75 = np.percentile(vorticity_mags, 75)
        q25 = np.percentile(vorticity_mags, 25)
        
        high_vort_mask = vorticity_mags > q75
        low_vort_mask = vorticity_mags < q25
        
        high_vort_vol = future_volatilities[high_vort_mask].mean() if high_vort_mask.sum() > 0 else 0
        low_vort_vol = future_volatilities[low_vort_mask].mean() if low_vort_mask.sum() > 0 else 0
    else:
        high_vort_vol = low_vort_vol = 0
    
    return {
        'n_samples': len(vorticity_mags),
        'correlation': corr if not np.isnan(corr) else 0.0,
        'high_vorticity_future_vol': high_vort_vol,
        'low_vorticity_future_vol': low_vort_vol,
        'vol_spread': high_vort_vol - low_vort_vol
    }


def analyze_multiscale_trading_signal(
    prices: np.ndarray,
    basis: np.ndarray,
    windows: List[int] = [4, 16, 32]  # Reduced max window for intraday data
) -> Dict[str, float]:
    """
    Test multi-scale coherence as a trading signal.
    
    Theory: High cross-scale coherence indicates a clear, actionable market state.
    Low coherence indicates uncertainty.
    """
    log_returns = np.diff(np.log(np.asarray(prices).flatten()))
    
    def encode_at_scale(returns, window, start_idx):
        if start_idx < 0 or start_idx + window > len(returns):
            return None
        segment = returns[start_idx:start_idx + window]
        
        if len(segment) < window:
            return None
        
        if window == 4:
            inc = segment
        else:
            indices = np.linspace(0, window-1, 4, dtype=int)
            inc = segment[indices]
        
        state = CliffordState.from_increments(inc, basis)
        return reconstruct_from_coefficients(state.coeffs, basis, np)
    
    coherences = []
    forward_returns = []
    
    max_window = max(windows)
    
    for i in range(max_window, len(log_returns) - 1):
        contexts = []
        for w in windows:
            start_idx = i - w
            if start_idx >= 0:
                ctx = encode_at_scale(log_returns, w, start_idx)
                if ctx is not None:
                    contexts.append(ctx)
        
        if len(contexts) == len(windows) and len(contexts) >= 2:
            # Compute cross-scale coherence (average pairwise similarity)
            sims = []
            for j in range(len(contexts)):
                for k in range(j+1, len(contexts)):
                    sim = frobenius_cosine(contexts[j], contexts[k], np)
                    if not np.isnan(sim):
                        sims.append(abs(sim))
            
            if sims:
                coherence = np.mean(sims)
                forward_ret = float(abs(log_returns[i]))  # Ensure scalar
                
                if np.isfinite(coherence) and np.isfinite(forward_ret):
                    coherences.append(float(coherence))
                    forward_returns.append(forward_ret)
    
    coherences = np.array(coherences) if coherences else np.array([0.0])
    forward_returns = np.array(forward_returns) if forward_returns else np.array([0.0])
    
    # High coherence should correlate with larger moves (clearer signal)
    if len(coherences) > 20 and len(forward_returns) > 20:
        if len(coherences) == len(forward_returns):
            corr = np.corrcoef(coherences, forward_returns)[0, 1]
        else:
            # Align lengths
            min_len = min(len(coherences), len(forward_returns))
            corr = np.corrcoef(coherences[:min_len], forward_returns[:min_len])[0, 1]
        
        q75 = np.percentile(coherences, 75)
        q25 = np.percentile(coherences, 25)
        
        high_coh_mask = coherences > q75
        low_coh_mask = coherences < q25
        
        high_coh_ret = forward_returns[high_coh_mask].mean() if high_coh_mask.sum() > 0 else 0
        low_coh_ret = forward_returns[low_coh_mask].mean() if low_coh_mask.sum() > 0 else 0
    else:
        corr = high_coh_ret = low_coh_ret = 0
    
    return {
        'n_samples': len(coherences),
        'mean_coherence': float(coherences.mean()) if len(coherences) > 0 else 0,
        'correlation': corr if not np.isnan(corr) else 0.0,
        'high_coherence_abs_return': high_coh_ret,
        'low_coherence_abs_return': low_coh_ret,
        'signal_spread': high_coh_ret - low_coh_ret
    }


# =============================================================================
# Main Test Suite
# =============================================================================

def run_intraday_tests(ticker: str = "SPY", period: str = "5d", interval: str = "5m"):
    """
    Run comprehensive intraday tests on real market data.
    """
    print("="*70)
    print(f"INTRADAY THEORY-TRUE VALIDATION")
    print(f"Ticker: {ticker}, Period: {period}, Interval: {interval}")
    print("="*70)
    
    # Fetch data
    print("\nFetching data...")
    prices = fetch_intraday_data(ticker, period, interval)
    
    if prices is None:
        print("Failed to fetch data. Using synthetic intraday data.")
        # Generate synthetic intraday-like data
        np.random.seed(42)
        n_points = 500
        # Intraday pattern: higher volatility at open/close
        base_vol = 0.001
        time_of_day = np.linspace(0, 2*np.pi, n_points)
        volatility = base_vol * (1 + 0.5 * np.abs(np.sin(time_of_day)))
        returns = np.random.randn(n_points) * volatility
        
        # Add some trend periods
        returns[100:150] += 0.0005  # Bull period
        returns[300:350] -= 0.0005  # Bear period
        
        prices = 100 * np.exp(np.cumsum(returns))
    
    print(f"Data points: {len(prices)}")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    basis = build_clifford_basis(np)
    
    # Run analyses
    results = {}
    
    # 1. Chirality-Regime Correlation
    print("\n" + "-"*50)
    print("Test 1: Chirality-Regime Correlation")
    print("-"*50)
    chirality_results = analyze_chirality_regime_correlation(prices, basis)
    results['chirality'] = chirality_results
    
    print(f"  Samples: {chirality_results['n_samples']}")
    print(f"  Positive chirality count: {chirality_results['pos_chirality_count']}")
    print(f"  Negative chirality count: {chirality_results['neg_chirality_count']}")
    print(f"  Pos chirality → future return: {chirality_results['pos_chirality_future_return']*10000:.2f} bps")
    print(f"  Neg chirality → future return: {chirality_results['neg_chirality_future_return']*10000:.2f} bps")
    print(f"  Information Coefficient: {chirality_results['information_coefficient']:.4f}")
    print(f"  Signal Value (spread): {chirality_results['signal_value']*10000:.2f} bps")
    
    # 2. Bivector-Trend Correlation
    print("\n" + "-"*50)
    print("Test 2: Bivector Stability → Trend Persistence")
    print("-"*50)
    bivector_results = analyze_bivector_trend_correlation(prices, basis)
    results['bivector'] = bivector_results
    
    print(f"  Stable windows: {bivector_results['n_stable_windows']}")
    print(f"  Unstable windows: {bivector_results['n_unstable_windows']}")
    print(f"  Stability ratio: {bivector_results['stability_ratio']:.2%}")
    print(f"  Stable mean |return|: {bivector_results['stable_mean_abs_return']*10000:.2f} bps")
    print(f"  Unstable mean |return|: {bivector_results['unstable_mean_abs_return']*10000:.2f} bps")
    print(f"  Predictability edge: {bivector_results['predictability_edge']*10000:.2f} bps")
    
    # 3. Vorticity-Volatility Correlation
    print("\n" + "-"*50)
    print("Test 3: Vorticity → Future Volatility")
    print("-"*50)
    vorticity_results = analyze_vorticity_volatility_correlation(prices, basis)
    results['vorticity'] = vorticity_results
    
    print(f"  Samples: {vorticity_results['n_samples']}")
    print(f"  Correlation: {vorticity_results['correlation']:.4f}")
    print(f"  High vorticity → future vol: {vorticity_results['high_vorticity_future_vol']*10000:.2f} bps")
    print(f"  Low vorticity → future vol: {vorticity_results['low_vorticity_future_vol']*10000:.2f} bps")
    print(f"  Vol spread: {vorticity_results['vol_spread']*10000:.2f} bps")
    
    # 4. Multi-scale Trading Signal
    print("\n" + "-"*50)
    print("Test 4: Multi-Scale Coherence Signal")
    print("-"*50)
    multiscale_results = analyze_multiscale_trading_signal(prices, basis)
    results['multiscale'] = multiscale_results
    
    print(f"  Samples: {multiscale_results['n_samples']}")
    print(f"  Mean coherence: {multiscale_results['mean_coherence']:.4f}")
    print(f"  Correlation with |return|: {multiscale_results['correlation']:.4f}")
    print(f"  High coherence |return|: {multiscale_results['high_coherence_abs_return']*10000:.2f} bps")
    print(f"  Low coherence |return|: {multiscale_results['low_coherence_abs_return']*10000:.2f} bps")
    print(f"  Signal spread: {multiscale_results['signal_spread']*10000:.2f} bps")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    signals = []
    
    # Chirality signal strength
    ic = abs(chirality_results['information_coefficient'])
    signals.append(('Chirality IC', ic, ic > 0.02))
    
    # Vorticity-vol correlation
    vort_corr = abs(vorticity_results['correlation'])
    signals.append(('Vorticity→Vol Corr', vort_corr, vort_corr > 0.05))
    
    # Multi-scale coherence signal
    ms_corr = abs(multiscale_results['correlation'])
    signals.append(('Multi-scale Corr', ms_corr, ms_corr > 0.02))
    
    # Bivector stability edge
    bv_edge = abs(bivector_results['predictability_edge']) * 10000
    signals.append(('Bivector Edge (bps)', bv_edge, bv_edge > 0.5))
    
    for name, value, passed in signals:
        status = "✓" if passed else "○"
        print(f"  {status} {name}: {value:.4f}")
    
    passed_count = sum(1 for _, _, p in signals if p)
    print(f"\nTheory signals detected: {passed_count}/{len(signals)}")
    
    if passed_count >= 2:
        print("✓ Sufficient theory-true signals present in data")
    else:
        print("○ Limited theory signals (may need more data or different conditions)")
    
    return results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Intraday Theory-True Validation")
    parser.add_argument("--ticker", default="SPY", help="Stock ticker symbol")
    parser.add_argument("--period", default="5d", help="Data period (e.g., 1d, 5d, 1mo)")
    parser.add_argument("--interval", default="5m", help="Data interval (e.g., 1m, 5m, 15m, 1h)")
    
    args = parser.parse_args()
    
    run_intraday_tests(args.ticker, args.period, args.interval)
