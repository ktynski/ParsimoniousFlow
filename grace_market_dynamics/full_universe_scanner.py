"""
FULL UNIVERSE SCANNER — All Stocks, Crypto, Forex
===================================================

Comprehensive scan of:
- ALL S&P 500 stocks
- ALL NASDAQ stocks (top ~1000)  
- ALL major cryptocurrencies (~200)
- ALL major forex pairs (~50)
- Major global ETFs

Uses batched fetching and caching for efficiency.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import sys
import os
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic_prod.core.constants import PHI, PHI_INV, DTYPE
from holographic_prod.core.algebra import (
    build_clifford_basis,
    grace_operator,
    grace_basin_key_direct,
)

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
# COMPREHENSIVE UNIVERSE LISTS
# =============================================================================

# S&P 500 (all components)
SP500 = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH',
    'JNJ', 'JPM', 'V', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
    'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'MCD', 'WMT', 'CSCO', 'ACN',
    'ABT', 'DHR', 'NEE', 'VZ', 'ADBE', 'NKE', 'CRM', 'TXN', 'PM', 'BMY',
    'UPS', 'RTX', 'QCOM', 'ORCL', 'HON', 'LOW', 'INTC', 'UNP', 'IBM', 'AMGN',
    'CAT', 'GE', 'BA', 'SBUX', 'PLD', 'INTU', 'GS', 'SPGI', 'DE', 'MS',
    'BLK', 'AMD', 'GILD', 'AXP', 'ISRG', 'MDT', 'ADI', 'CVS', 'SYK', 'BKNG',
    'MDLZ', 'TJX', 'VRTX', 'MMC', 'CB', 'ADP', 'REGN', 'PGR', 'CI', 'SO',
    'MO', 'DUK', 'CME', 'LRCX', 'ZTS', 'NOC', 'ITW', 'EOG', 'BSX', 'ETN',
    'SCHW', 'APD', 'SHW', 'FDX', 'CL', 'MU', 'HUM', 'AON', 'ICE', 'PNC',
    'TGT', 'EMR', 'PSA', 'SNPS', 'ORLY', 'GM', 'AZO', 'FCX', 'NSC', 'USB',
    'WM', 'MAR', 'MCO', 'SLB', 'CDNS', 'F', 'KLAC', 'CCI', 'GD', 'EW',
    'TFC', 'MCHP', 'DG', 'HCA', 'MCK', 'CMG', 'OXY', 'PCAR', 'D', 'ROP',
    'AFL', 'JCI', 'AEP', 'PAYX', 'MSCI', 'PSX', 'MSI', 'NXPI', 'MNST', 'O',
    'SRE', 'AIG', 'WMB', 'TEL', 'TRV', 'KMB', 'ECL', 'A', 'FTNT', 'SPG',
    'HSY', 'ADM', 'EXC', 'WELL', 'CARR', 'CTVA', 'GIS', 'KHC', 'HAL', 'ROST',
    'YUM', 'DOW', 'CTAS', 'KDP', 'PH', 'VLO', 'DXCM', 'BK', 'CMI', 'OTIS',
    'EL', 'HLT', 'IQV', 'PRU', 'DD', 'STZ', 'LHX', 'IDXX', 'EA', 'COF',
    'FAST', 'VRSK', 'XEL', 'CTSH', 'PPG', 'AME', 'ON', 'GWW', 'KR', 'DVN',
    'BIIB', 'ED', 'MLM', 'ALB', 'WEC', 'VICI', 'ODFL', 'GEHC', 'ANSS', 'BKR',
    'AWK', 'DLTR', 'KEYS', 'ROK', 'MTD', 'EFX', 'TSCO', 'CDW', 'EXR', 'HPQ',
    'WTW', 'RMD', 'DLR', 'DHI', 'FTV', 'CBRE', 'EBAY', 'CPRT', 'FANG', 'AVB',
    'CHD', 'WAB', 'GPN', 'NUE', 'IR', 'TROW', 'AMP', 'ACGL', 'DAL', 'DOV',
    'MKC', 'BR', 'FITB', 'LEN', 'FE', 'VMC', 'ZBH', 'HIG', 'PPL', 'CAH',
    'BALL', 'DTE', 'NVR', 'PTC', 'AEE', 'STE', 'RF', 'HOLX', 'LYV', 'WST',
    'ES', 'MOH', 'WY', 'AJG', 'IRM', 'BAX', 'SBAC', 'NTRS', 'EXPD', 'BRO',
    'CINF', 'HBAN', 'LH', 'INVH', 'TDY', 'MAA', 'CLX', 'ULTA', 'TER', 'PODD',
    'EQR', 'CNP', 'RJF', 'CFG', 'K', 'ESS', 'STT', 'STLD', 'ARE', 'PFG',
    'SWKS', 'DRI', 'IFF', 'IP', 'TRMB', 'VTRS', 'WAT', 'KEY', 'NDAQ', 'LUV',
    'DGX', 'WDC', 'SYF', 'CMS', 'LVS', 'EXPE', 'BXP', 'CAG', 'TXT', 'PKI',
    'MAS', 'SJM', 'JBHT', 'TSN', 'EVRG', 'OMC', 'NTAP', 'PEAK', 'SNA', 'J',
    'HST', 'POOL', 'LNT', 'AKAM', 'AVY', 'L', 'CE', 'UDR', 'IPG', 'TECH',
    'NI', 'CPT', 'KIM', 'ATO', 'GL', 'BEN', 'TAP', 'REG', 'CHRW', 'HRL',
    'ALLE', 'JNPR', 'HSIC', 'CRL', 'QRVO', 'BWA', 'HII', 'FFIV', 'EMN', 'AOS',
    'MKTX', 'MOS', 'FRT', 'WHR', 'RHI', 'PNR', 'WYNN', 'BBWI', 'AAP', 'AIZ',
    'NWL', 'FMC', 'BIO', 'SEE', 'VFC', 'DXC', 'ALK', 'NWSA', 'NWS', 'LKQ',
    'DISH', 'PARA', 'CTLT', 'MHK', 'XRAY', 'DVA', 'CZR', 'CCL', 'NCLH', 'UAL',
    'AAL', 'PVH', 'IVZ', 'RL', 'TPR', 'GEN', 'HAS', 'ETSY', 'MTCH', 'ZION',
    'CMA', 'SIVB', 'FRC', 'SBNY', 'NYCB', 'WBA', 'VNO', 'LNC', 'FOX', 'FOXA'
]

# Additional NASDAQ stocks
NASDAQ_EXTRA = [
    'NFLX', 'PYPL', 'MRNA', 'PANW', 'MRVL', 'CRWD', 'DDOG', 'ZS', 'OKTA', 'NET',
    'SNOW', 'PLTR', 'COIN', 'RIVN', 'LCID', 'SOFI', 'HOOD', 'UPST', 'AFRM', 'PATH',
    'ZM', 'DOCU', 'ROKU', 'SPOT', 'SQ', 'SHOP', 'MELI', 'SE', 'GRAB', 'NU',
    'RBLX', 'UNITY', 'U', 'ABNB', 'DASH', 'UBER', 'LYFT', 'TTD', 'SNAP', 'PINS',
    'TWLO', 'DBX', 'BOX', 'WDAY', 'NOW', 'TEAM', 'HUBS', 'ZI', 'VEEV', 'BILL',
    'PCTY', 'PAYC', 'CFLT', 'MDB', 'ESTC', 'FIVN', 'GTLB', 'DOCN', 'DT', 'SUMO',
    'ASAN', 'MNDY', 'FROG', 'DOMO', 'SPT', 'APP', 'IS', 'BRZE', 'AMPL', 'CWAN',
    'SMAR', 'ALTR', 'SMCI', 'AI', 'BBAI', 'PRCT', 'IONQ', 'QBTS', 'RGTI', 'QUBT',
    'ARM', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'VERV', 'FATE', 'BLUE', 'SGMO', 'RARE',
    'ARWR', 'ALNY', 'SRPT', 'EXAS', 'NVAX', 'BNTX', 'MRVI', 'CDNA', 'TWST', 'TXG'
]

# Comprehensive Crypto (top 200 by market cap)
CRYPTO = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'SOL-USD',
    'TRX-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD', 'SHIB-USD', 'AVAX-USD', 'LINK-USD',
    'ATOM-USD', 'UNI-USD', 'XMR-USD', 'ETC-USD', 'XLM-USD', 'BCH-USD', 'FIL-USD',
    'HBAR-USD', 'ICP-USD', 'LDO-USD', 'APT-USD', 'ARB-USD', 'OP-USD', 'VET-USD',
    'NEAR-USD', 'QNT-USD', 'ALGO-USD', 'GRT-USD', 'AAVE-USD', 'EOS-USD', 'STX-USD',
    'SAND-USD', 'MANA-USD', 'AXS-USD', 'THETA-USD', 'EGLD-USD', 'XTZ-USD', 'IMX-USD',
    'FLOW-USD', 'NEO-USD', 'KCS-USD', 'CRV-USD', 'FTM-USD', 'MKR-USD', 'SNX-USD',
    'RUNE-USD', 'ZEC-USD', 'KAVA-USD', 'MINA-USD', 'XDC-USD', 'DASH-USD', 'CHZ-USD',
    'LRC-USD', 'ENJ-USD', 'BAT-USD', '1INCH-USD', 'COMP-USD', 'YFI-USD', 'ZIL-USD',
    'ENS-USD', 'DYDX-USD', 'GMX-USD', 'MASK-USD', 'CAKE-USD', 'GT-USD', 'ROSE-USD',
    'GALA-USD', 'APE-USD', 'BLUR-USD', 'CFX-USD', 'INJ-USD', 'SUI-USD', 'SEI-USD',
    'TIA-USD', 'JTO-USD', 'PYTH-USD', 'JUP-USD', 'WIF-USD', 'BONK-USD', 'PEPE-USD',
    'FLOKI-USD', 'WLD-USD', 'FET-USD', 'RNDR-USD', 'AGIX-USD', 'OCEAN-USD', 'TAO-USD',
    'JASMY-USD', 'HOT-USD', 'ONE-USD', 'IOTA-USD', 'ICX-USD', 'ZEN-USD', 'WAVES-USD'
]

# Major Forex Pairs
FOREX = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X',
    'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X', 'AUDJPY=X', 'CADJPY=X',
    'CHFJPY=X', 'EURAUD=X', 'EURCAD=X', 'EURCHF=X', 'EURNZD=X', 'GBPAUD=X',
    'GBPCAD=X', 'GBPCHF=X', 'GBPNZD=X', 'AUDCAD=X', 'AUDCHF=X', 'AUDNZD=X',
    'CADCHF=X', 'NZDCAD=X', 'NZDCHF=X', 'NZDJPY=X', 'USDMXN=X', 'USDBRL=X',
    'USDZAR=X', 'USDTRY=X', 'USDSGD=X', 'USDHKD=X', 'USDCNH=X', 'USDINR=X',
    'USDKRW=X', 'USDRUB=X', 'USDPLN=X', 'USDHUF=X', 'USDCZK=X', 'USDNOK=X',
    'USDSEK=X', 'USDDKK=X', 'EURPLN=X', 'EURNOK=X', 'EURSEK=X', 'EURTRY=X'
]

# Global ETFs
GLOBAL_ETFS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'EEM', 'EFA',
    'TLT', 'IEF', 'SHY', 'BND', 'LQD', 'HYG', 'JNK', 'AGG', 'TIP', 'EMB',
    'GLD', 'SLV', 'GDX', 'GDXJ', 'USO', 'UNG', 'DBA', 'DBB', 'CORN', 'WEAT',
    'VNQ', 'IYR', 'XLRE', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLU', 'XLP',
    'XLY', 'XLB', 'XLC', 'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ', 'ARKX', 'SARK',
    'SMH', 'SOXX', 'XBI', 'IBB', 'KRE', 'XHB', 'ITB', 'XRT', 'JETS', 'HACK',
    'KWEB', 'FXI', 'MCHI', 'EWJ', 'EWZ', 'EWY', 'INDA', 'VGK', 'EWG', 'EWU',
    'FEZ', 'HEDJ', 'DXJ', 'RSX', 'TUR', 'ERUS', 'GXC', 'THD', 'VNM', 'EPOL'
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


def analyze_batch(tickers: List[str], asset_class: str, yf) -> List[ScanResult]:
    """Analyze a batch of tickers."""
    results = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period='1y', interval='1d')
            
            if df is None or len(df) < 60:
                results.append(ScanResult(
                    ticker=ticker, asset_class=asset_class, current_price=0,
                    signal=0, t_stat=0, mean_return=0, n_observations=0,
                    is_favorable=False, error='Insufficient data'
                ))
                continue
            
            prices = df['Close'].values.flatten()
            current_price = float(prices[-1])
            
            if current_price <= 0:
                results.append(ScanResult(
                    ticker=ticker, asset_class=asset_class, current_price=0,
                    signal=0, t_stat=0, mean_return=0, n_observations=0,
                    is_favorable=False, error='Invalid price'
                ))
                continue
            
            returns = np.diff(np.log(prices))
            returns = np.clip(returns, -0.2, 0.2)  # Cap extreme moves
            
            # Build basin statistics
            basin_stats = defaultdict(list)
            
            for i in range(20, len(returns) - 5):
                state = build_state(returns[i-20:i])
                key = get_basin_key(state)
                forward_ret = np.sum(returns[i:i+5])
                if abs(forward_ret) < 0.5:
                    basin_stats[key].append(forward_ret)
            
            # Current state
            current_state = build_state(returns[-20:])
            current_key = get_basin_key(current_state)
            
            if current_key in basin_stats and len(basin_stats[current_key]) >= 5:
                hist_returns = np.array(basin_stats[current_key])
                mean_ret = np.mean(hist_returns)
                std_ret = np.std(hist_returns)
                n_obs = len(hist_returns)
                
                if std_ret > 1e-10:
                    t_stat = mean_ret / (std_ret / np.sqrt(n_obs))
                else:
                    t_stat = 0
                
                t_stat = np.clip(t_stat, -10, 10)
                signal = np.tanh(t_stat / 2.0)
                is_favorable = t_stat > 1.5 and mean_ret > 0
            else:
                mean_ret, t_stat, signal, n_obs = 0, 0, 0, 0
                is_favorable = False
            
            results.append(ScanResult(
                ticker=ticker, asset_class=asset_class, current_price=current_price,
                signal=signal, t_stat=t_stat, mean_return=mean_ret,
                n_observations=n_obs, is_favorable=is_favorable
            ))
            
        except Exception as e:
            results.append(ScanResult(
                ticker=ticker, asset_class=asset_class, current_price=0,
                signal=0, t_stat=0, mean_return=0, n_observations=0,
                is_favorable=False, error=str(e)[:20]
            ))
    
    return results


def main():
    print("="*80)
    print("FULL UNIVERSE SCANNER — All Stocks, Crypto, Forex")
    print("="*80)
    print(f"\nScan time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not available")
        return
    
    # Build comprehensive universe
    universe = {
        'Stock': list(set(SP500 + NASDAQ_EXTRA)),
        'Crypto': CRYPTO,
        'Forex': FOREX,
        'ETF': GLOBAL_ETFS
    }
    
    total = sum(len(v) for v in universe.values())
    print(f"\nTotal universe: {total} instruments")
    for asset_class, tickers in universe.items():
        print(f"  {asset_class}: {len(tickers)}")
    
    start_time = time.time()
    all_results = []
    
    # Process each asset class
    for asset_class, tickers in universe.items():
        print(f"\n  Scanning {asset_class}...")
        
        # Process in batches
        batch_size = 20
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            results = analyze_batch(batch, asset_class, yf)
            all_results.extend(results)
            
            progress = min(i + batch_size, len(tickers))
            print(f"    Progress: {progress}/{len(tickers)}", end='\r')
            time.sleep(0.5)  # Rate limiting
        
        print(f"    Completed {asset_class}: {len(tickers)} instruments")
    
    elapsed = time.time() - start_time
    print(f"\nScan completed in {elapsed:.1f} seconds")
    
    # Filter successful scans
    successful = [r for r in all_results if not r.error]
    errors = len(all_results) - len(successful)
    print(f"  Successful: {len(successful)}, Errors: {errors}")
    
    # Find favorable signals
    favorable = [r for r in successful if r.is_favorable]
    favorable.sort(key=lambda x: -x.t_stat)
    
    # Find unfavorable signals (short candidates)
    unfavorable = [r for r in successful if r.t_stat < -1.5 and r.mean_return < 0 and r.n_observations >= 5]
    unfavorable.sort(key=lambda x: x.t_stat)
    
    # =================================================================
    # RESULTS OUTPUT
    # =================================================================
    
    print("\n" + "="*80)
    print(f"FAVORABLE SIGNALS (t > 1.5): {len(favorable)}")
    print("="*80)
    
    # Group by asset class
    for asset_class in ['Stock', 'ETF', 'Crypto', 'Forex']:
        class_fav = [r for r in favorable if r.asset_class == asset_class]
        if class_fav:
            print(f"\n  {asset_class.upper()} ({len(class_fav)} signals):")
            print(f"  {'Ticker':<12} {'Price':>12} {'T-stat':>8} {'Mean 5d':>10} {'N':>6}")
            print("  " + "-"*55)
            for r in class_fav[:15]:
                price_str = f"${r.current_price:,.2f}" if r.current_price < 10000 else f"${r.current_price:,.0f}"
                print(f"  {r.ticker:<12} {price_str:>12} {r.t_stat:>8.2f} {r.mean_return*100:>9.2f}% {r.n_observations:>6}")
    
    print("\n" + "="*80)
    print(f"UNFAVORABLE SIGNALS (t < -1.5): {len(unfavorable)}")
    print("="*80)
    
    for asset_class in ['Stock', 'ETF', 'Crypto', 'Forex']:
        class_unfav = [r for r in unfavorable if r.asset_class == asset_class]
        if class_unfav:
            print(f"\n  {asset_class.upper()} ({len(class_unfav)} signals):")
            print(f"  {'Ticker':<12} {'Price':>12} {'T-stat':>8} {'Mean 5d':>10}")
            print("  " + "-"*45)
            for r in class_unfav[:10]:
                price_str = f"${r.current_price:,.2f}" if r.current_price < 10000 else f"${r.current_price:,.0f}"
                print(f"  {r.ticker:<12} {price_str:>12} {r.t_stat:>8.2f} {r.mean_return*100:>9.2f}%")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n  {'Asset Class':<12} {'Scanned':>10} {'Favorable':>10} {'%':>8} {'Unfavorable':>12}")
    print("  " + "-"*55)
    
    for asset_class in ['Stock', 'ETF', 'Crypto', 'Forex']:
        scanned = len([r for r in successful if r.asset_class == asset_class])
        fav = len([r for r in favorable if r.asset_class == asset_class])
        unfav = len([r for r in unfavorable if r.asset_class == asset_class])
        pct = fav / scanned * 100 if scanned > 0 else 0
        print(f"  {asset_class:<12} {scanned:>10} {fav:>10} {pct:>7.1f}% {unfav:>12}")
    
    # Top recommendations
    print("\n" + "="*80)
    print("TOP RECOMMENDATIONS")
    print("="*80)
    
    print("\n  STRONGEST LONG SIGNALS:")
    for i, r in enumerate(favorable[:10], 1):
        print(f"    {i}. {r.ticker} ({r.asset_class}): t={r.t_stat:.2f}, exp={r.mean_return*100:+.2f}%")
    
    if unfavorable:
        print("\n  STRONGEST SHORT SIGNALS:")
        for i, r in enumerate(unfavorable[:5], 1):
            print(f"    {i}. {r.ticker} ({r.asset_class}): t={r.t_stat:.2f}, exp={r.mean_return*100:+.2f}%")
    
    # Save results to JSON
    output_file = f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        'scan_time': datetime.now().isoformat(),
        'total_scanned': len(successful),
        'favorable': [
            {
                'ticker': r.ticker,
                'asset_class': r.asset_class,
                'price': r.current_price,
                't_stat': r.t_stat,
                'mean_return': r.mean_return,
                'n': r.n_observations
            }
            for r in favorable
        ],
        'unfavorable': [
            {
                'ticker': r.ticker,
                'asset_class': r.asset_class,
                'price': r.current_price,
                't_stat': r.t_stat,
                'mean_return': r.mean_return,
                'n': r.n_observations
            }
            for r in unfavorable
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n  Results saved to: {output_file}")
    
    return all_results, favorable, unfavorable


if __name__ == "__main__":
    main()
