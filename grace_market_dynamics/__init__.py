"""
Grace Market Dynamics — Observable-Agnostic Clifford Torus
==========================================================

Theory: Encode the *intrinsic geometry of the stream*, not "what you measured."

Key principles:
1. Gauge-invariance: representation unchanged under feature swapping, rescaling,
   linear mixes, adding/dropping channels
2. Coordinate-free: build state from Gram matrix eigenspectrum (intrinsic)
3. Fill 16 Clifford grades via wedge products across time lags

The 16D Clifford decomposition:
- Scalar (1): energy/scale
- Vector (4): current direction in canonical frame  
- Bivector (6): local "turning plane" (v_t ∧ v_{t-1})
- Trivector (4): 3-way regime texture
- Pseudoscalar (1): chirality / persistent spin

Modules:
- state_encoder: CliffordState, ObservableAgnosticEncoder
- rotor_dynamics: RotorPredictor, BetTrigger
- signal_generator: LiveSignalGenerator, Signal
- backtest: CliffordBacktester, BacktestResult
- multiscale: MultiscaleEncoder, MultiscaleState
- basin_discovery: BasinDiscovery, RegimeState
"""

from .state_encoder import CliffordState, ObservableAgnosticEncoder
from .rotor_dynamics import RotorPredictor, BetTrigger
from .signal_generator import LiveSignalGenerator, Signal, SignalConfig
from .backtest import CliffordBacktester, BacktestConfig, BacktestResult
from .multiscale import MultiscaleEncoder, MultiscaleConfig, MultiscaleState, ScaleConfig
from .basin_discovery import BasinDiscovery, BasinDiscoveryConfig, RegimeState

__all__ = [
    # Core encoding
    'CliffordState',
    'ObservableAgnosticEncoder',
    
    # Rotor dynamics
    'RotorPredictor',
    'BetTrigger',
    
    # Signal generation
    'LiveSignalGenerator',
    'Signal',
    'SignalConfig',
    
    # Backtesting
    'CliffordBacktester',
    'BacktestConfig',
    'BacktestResult',
    
    # Multi-timescale
    'MultiscaleEncoder',
    'MultiscaleConfig',
    'MultiscaleState',
    'ScaleConfig',
    
    # Regime identification
    'BasinDiscovery',
    'BasinDiscoveryConfig',
    'RegimeState',
]
