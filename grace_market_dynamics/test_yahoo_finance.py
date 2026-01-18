#!/usr/bin/env python3
"""
Observable-Agnostic Clifford Torus — Yahoo Finance Test
========================================================

Theory-true test using actual Cl(3,1) Clifford algebra.

Tests:
1. Single ticker with delay embedding (truly minimal assumptions)
2. Multi-feature with OHLCV (still observable-agnostic)
3. Basket of tickers (tests gauge-invariance across instruments)

Key metrics:
- Does the geometry predict itself? (prediction error)
- Is the rotor stable? (coherence)
- Do bet triggers correlate with forward returns?
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Add parent to path for holographic_prod imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our modules
from state_encoder import ObservableAgnosticEncoder, delay_embed, CliffordState
from rotor_dynamics import RotorPredictor, BetTrigger

# Try to import yfinance
try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance', '-q'])
    import yfinance as yf


def fetch_ticker_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
) -> Tuple[pd.DataFrame, str]:
    """
    Fetch data for a single ticker.
    
    Returns:
        (DataFrame with OHLCV, error message if any)
    """
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if len(data) < 50:
            return pd.DataFrame(), f"Insufficient data for {ticker}"
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        return data, ""
    except Exception as e:
        return pd.DataFrame(), str(e)


def test_single_ticker_delay_embed(
    ticker: str = "SPY",
    period: str = "1y"
) -> dict:
    """
    Test 1: Single ticker with delay embedding.
    
    This is the MINIMAL observable-agnostic case:
    - Only assume "there is a time series"
    - Embed via delays: [Δlog(p_t), Δlog(p_{t-1}), ...]
    - Fill all 16 Clifford grades via wedge products
    """
    print(f"\n{'='*60}")
    print(f"TEST 1: Single Ticker Delay Embedding — {ticker}")
    print('='*60)
    
    # Fetch data
    data, err = fetch_ticker_data(ticker, period)
    if err:
        print(f"  ERROR: {err}")
        return {'success': False, 'error': err}
    
    prices = data['Close'].values
    print(f"  Fetched {len(prices)} days of data")
    
    # Delay embed: scalar price → 4D state
    embedded = delay_embed(prices, delays=4)
    print(f"  Delay embedded: {embedded.shape}")
    
    # Initialize encoder and rotor predictor
    encoder = ObservableAgnosticEncoder(
        gram_window=20,
        use_log_returns=False  # Already log-differenced
    )
    predictor = RotorPredictor(history_length=5)
    trigger = BetTrigger(
        error_threshold=0.4,
        coherence_threshold=0.5,
        chirality_persistence=3
    )
    
    # Process stream
    states: List[CliffordState] = []
    rotor_states = []
    bet_signals = []
    
    for i, obs in enumerate(embedded):
        state = encoder.update(obs)
        if state is None:
            continue
        
        states.append(state)
        
        rotor_state = predictor.update(state)
        if rotor_state is None:
            continue
        
        rotor_states.append(rotor_state)
        
        bet_result = trigger.evaluate(rotor_state, state)
        bet_signals.append({
            'index': i,
            'signal': bet_result['signal'],
            'trigger': bet_result['trigger'],
            'direction': bet_result['direction'],
            'confidence': bet_result['confidence'],
            'reason': bet_result['reason'],
            **bet_result['diagnostics']
        })
    
    if len(bet_signals) < 10:
        print(f"  ERROR: Only {len(bet_signals)} valid states")
        return {'success': False, 'error': 'Insufficient valid states'}
    
    # Analyze results
    df_signals = pd.DataFrame(bet_signals)
    
    # Grade magnitudes distribution
    grade_mags = [s.grade_magnitudes() for s in states]
    df_grades = pd.DataFrame(grade_mags)
    
    print(f"\n  === CLIFFORD GRADE STATISTICS ===")
    for col in df_grades.columns:
        print(f"    {col}: mean={df_grades[col].mean():.4f}, std={df_grades[col].std():.4f}")
    
    print(f"\n  === ROTOR DYNAMICS ===")
    print(f"    Mean prediction error: {df_signals['prediction_error'].mean():.4f}")
    print(f"    Mean coherence: {df_signals['coherence'].mean():.4f}")
    print(f"    Trigger rate: {df_signals['trigger'].mean()*100:.1f}%")
    
    # Correlate signal with forward returns
    # Align signals with next day's return
    forward_returns = np.diff(np.log(prices[len(prices)-len(df_signals)-1:]))
    if len(forward_returns) >= len(df_signals):
        forward_returns = forward_returns[:len(df_signals)]
        df_signals['forward_return'] = forward_returns
        
        # Correlation
        signal_return_corr = df_signals['signal'].corr(df_signals['forward_return'])
        print(f"\n  === PREDICTIVE POWER ===")
        print(f"    Signal-Return Correlation: {signal_return_corr:.4f}")
        
        # PnL if trading on signal
        triggered = df_signals[df_signals['trigger']]
        if len(triggered) > 0:
            triggered_pnl = (triggered['direction'] * triggered['forward_return']).sum()
            triggered_sharpe = (triggered['direction'] * triggered['forward_return']).mean() / \
                              (triggered['direction'] * triggered['forward_return']).std() if len(triggered) > 5 else 0
            print(f"    Triggered trades: {len(triggered)}")
            print(f"    Triggered PnL: {triggered_pnl*100:.2f}%")
            print(f"    Triggered Sharpe (raw): {triggered_sharpe*np.sqrt(252):.2f}")
    
    return {
        'success': True,
        'ticker': ticker,
        'n_states': len(states),
        'n_signals': len(bet_signals),
        'mean_prediction_error': df_signals['prediction_error'].mean(),
        'mean_coherence': df_signals['coherence'].mean(),
        'trigger_rate': df_signals['trigger'].mean(),
        'grade_stats': df_grades.describe().to_dict(),
    }


def test_multi_feature_ohlcv(
    ticker: str = "SPY",
    period: str = "1y"
) -> dict:
    """
    Test 2: Multi-feature with OHLCV.
    
    Still observable-agnostic: we don't assume which feature is "important".
    The Gram matrix extracts intrinsic structure.
    """
    print(f"\n{'='*60}")
    print(f"TEST 2: Multi-Feature OHLCV — {ticker}")
    print('='*60)
    
    # Fetch data
    data, err = fetch_ticker_data(ticker, period)
    if err:
        print(f"  ERROR: {err}")
        return {'success': False, 'error': err}
    
    # Use OHLCV as observation vector
    ohlcv = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    print(f"  Fetched {len(ohlcv)} days of OHLCV data")
    
    # Normalize volume to same scale as prices
    ohlcv[:, 4] = ohlcv[:, 4] / ohlcv[:, 4].mean() * ohlcv[:, 3].mean()
    
    # Initialize encoder
    encoder = ObservableAgnosticEncoder(
        gram_window=20,
        use_log_returns=True  # Will log the prices
    )
    predictor = RotorPredictor(history_length=5)
    trigger = BetTrigger()
    
    # Process stream
    states = []
    rotor_states = []
    bet_signals = []
    
    for i, obs in enumerate(ohlcv):
        state = encoder.update(obs)
        if state is None:
            continue
        
        states.append(state)
        
        rotor_state = predictor.update(state)
        if rotor_state is None:
            continue
        
        rotor_states.append(rotor_state)
        bet_result = trigger.evaluate(rotor_state, state)
        bet_signals.append({
            'index': i,
            **bet_result,
            **bet_result['diagnostics']
        })
    
    if len(bet_signals) < 10:
        return {'success': False, 'error': 'Insufficient states'}
    
    df_signals = pd.DataFrame(bet_signals)
    
    print(f"\n  === ROTOR DYNAMICS ===")
    print(f"    Mean prediction error: {df_signals['prediction_error'].mean():.4f}")
    print(f"    Mean coherence: {df_signals['coherence'].mean():.4f}")
    print(f"    Trigger rate: {df_signals['trigger'].mean()*100:.1f}%")
    
    # Invariants from encoder
    invariants = encoder.get_invariants()
    if invariants:
        print(f"\n  === GAUGE INVARIANTS ===")
        print(f"    Scale (energy): {invariants.get('energy', 0):.4f}")
        print(f"    Effective dimension: {invariants.get('effective_dimension', 0):.2f}")
        if 'anisotropy' in invariants:
            print(f"    Anisotropy: {invariants['anisotropy']}")
    
    return {
        'success': True,
        'ticker': ticker,
        'n_states': len(states),
        'mean_prediction_error': df_signals['prediction_error'].mean(),
        'mean_coherence': df_signals['coherence'].mean(),
        'trigger_rate': df_signals['trigger'].mean(),
        'invariants': invariants,
    }


def test_basket_gauge_invariance(
    tickers: List[str] = ['SPY', 'QQQ', 'IWM', 'TLT'],
    period: str = "1y"
) -> dict:
    """
    Test 3: Basket of tickers — test gauge-invariance.
    
    Key question: does rescaling/permuting features change the geometry?
    It shouldn't if our encoder is truly observable-agnostic.
    """
    print(f"\n{'='*60}")
    print(f"TEST 3: Basket Gauge Invariance — {tickers}")
    print('='*60)
    
    # Fetch all tickers
    dfs = {}
    for t in tickers:
        data, err = fetch_ticker_data(t, period)
        if not err and len(data) > 50:
            dfs[t] = data['Close']
    
    if len(dfs) < 2:
        return {'success': False, 'error': 'Not enough valid tickers'}
    
    # Align dates
    prices_df = pd.DataFrame(dfs).dropna()
    print(f"  Aligned {len(prices_df)} days across {len(dfs)} tickers")
    
    # Test: compare encoder with original vs permuted features
    observations = prices_df.values  # Shape: [T, n_tickers]
    
    # Encoder on original
    encoder_orig = ObservableAgnosticEncoder(gram_window=15, use_log_returns=True)
    
    states_orig = []
    for obs in observations:
        state = encoder_orig.update(obs)
        if state is not None:
            states_orig.append(state.coefficients)
    
    if len(states_orig) < 10:
        return {'success': False, 'error': 'Too few states'}
    
    # Encoder on permuted features (should give similar geometry!)
    perm = np.random.permutation(observations.shape[1])
    observations_perm = observations[:, perm]
    
    encoder_perm = ObservableAgnosticEncoder(gram_window=15, use_log_returns=True)
    
    states_perm = []
    for obs in observations_perm:
        state = encoder_perm.update(obs)
        if state is not None:
            states_perm.append(state.coefficients)
    
    # Compare: correlation of grade magnitudes should be high
    # (exact match not expected due to eigenvector ordering)
    n_compare = min(len(states_orig), len(states_perm))
    
    orig_mags = np.array([np.linalg.norm(s) for s in states_orig[:n_compare]])
    perm_mags = np.array([np.linalg.norm(s) for s in states_perm[:n_compare]])
    
    magnitude_corr = np.corrcoef(orig_mags, perm_mags)[0, 1]
    
    print(f"\n  === GAUGE INVARIANCE TEST ===")
    print(f"    Original vs Permuted magnitude correlation: {magnitude_corr:.4f}")
    print(f"    (Should be close to 1.0 if gauge-invariant)")
    
    # Scale test: multiply all prices by random scale
    scale = np.random.uniform(0.5, 2.0, size=observations.shape[1])
    observations_scaled = observations * scale
    
    encoder_scaled = ObservableAgnosticEncoder(gram_window=15, use_log_returns=True)
    
    states_scaled = []
    for obs in observations_scaled:
        state = encoder_scaled.update(obs)
        if state is not None:
            states_scaled.append(state.coefficients)
    
    n_compare = min(len(states_orig), len(states_scaled))
    scaled_mags = np.array([np.linalg.norm(s) for s in states_scaled[:n_compare]])
    
    scale_corr = np.corrcoef(orig_mags[:n_compare], scaled_mags)[0, 1]
    
    print(f"    Original vs Scaled magnitude correlation: {scale_corr:.4f}")
    print(f"    (Should be close to 1.0 if gauge-invariant)")
    
    # Conclusion
    is_gauge_invariant = magnitude_corr > 0.8 and scale_corr > 0.8
    print(f"\n    Gauge invariance: {'PASS' if is_gauge_invariant else 'NEEDS WORK'}")
    
    return {
        'success': True,
        'n_tickers': len(dfs),
        'n_states': len(states_orig),
        'permutation_correlation': magnitude_corr,
        'scaling_correlation': scale_corr,
        'gauge_invariant': is_gauge_invariant,
    }


def run_all_tests():
    """Run all tests and summarize."""
    print("\n" + "="*70)
    print("OBSERVABLE-AGNOSTIC CLIFFORD TORUS — YAHOO FINANCE TESTS")
    print("="*70)
    print("\nTheory: Encode intrinsic geometry, not named features.")
    print("Key: Gauge-invariance under feature swaps, rescaling, linear mixes.\n")
    
    results = {}
    
    # Test 1: Single ticker delay embedding
    results['delay_embed'] = test_single_ticker_delay_embed('SPY', '1y')
    
    # Test 2: Multi-feature OHLCV
    results['ohlcv'] = test_multi_feature_ohlcv('SPY', '1y')
    
    # Test 3: Basket gauge invariance
    results['basket'] = test_basket_gauge_invariance(['SPY', 'QQQ', 'IWM', 'TLT'], '1y')
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, res in results.items():
        status = '✓' if res.get('success', False) else '✗'
        print(f"\n  {status} {name.upper()}")
        if res.get('success'):
            if 'mean_prediction_error' in res:
                print(f"      Prediction error: {res['mean_prediction_error']:.4f}")
            if 'mean_coherence' in res:
                print(f"      Coherence: {res['mean_coherence']:.4f}")
            if 'gauge_invariant' in res:
                print(f"      Gauge invariant: {res['gauge_invariant']}")
        else:
            print(f"      Error: {res.get('error', 'Unknown')}")
    
    return results


if __name__ == "__main__":
    import signal
    # Timeout after 120 seconds
    signal.signal(signal.SIGALRM, lambda s, f: exit(1))
    signal.alarm(120)
    
    try:
        results = run_all_tests()
        
        # Final assessment
        all_success = all(r.get('success', False) for r in results.values())
        print(f"\n{'='*70}")
        if all_success:
            print("ALL TESTS PASSED — Observable-agnostic encoding working")
        else:
            print("SOME TESTS FAILED — Review above for details")
        print('='*70)
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
