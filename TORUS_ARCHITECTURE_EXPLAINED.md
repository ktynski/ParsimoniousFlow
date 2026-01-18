# How the Multi-Scale Torus Architecture Finds Alpha

## The Core FSCTF Insight

The Self-Consistent Coherence-Maximizing Framework (FSCTF) says:

> **Reality must be self-consistent. When it isn't, it will restore consistency.**

This is not a prediction about *direction*. It's a statement about *invariants*. We don't predict "GILD will go up" — we detect "GILD is out of phase with its sector, and phases must synchronize."

## The Gauge Symmetry

### What Standard Finance Misses

Standard finance treats prices as absolute values:
- "GILD is at $95"
- "IBB is at $130"

But prices are **gauge-dependent**. Multiply all prices by 2 → nothing real changes. The "physics" is in the **connections** (log returns), not the prices themselves.

### The Real Invariant

The real invariant is **phase coherence in the hierarchy**:

```
Market → Sector → Industry → Company

Each level should be phase-aligned with its parent.
When it's not, that's a detectable symmetry violation.
```

## The Torus Structure

### Why a Torus?

Financial dynamics have **two natural cycles**:

1. **Price/Momentum Cycle (θ)**
   - Uptrend → Exhaustion → Downtrend → Capitulation → repeat
   - This is the familiar "market cycle"

2. **Volatility Cycle (φ)**
   - Low vol → Rising vol → High vol → Falling vol → repeat
   - This is the "fear/complacency" cycle

These two cycles are **independent** but **coupled**. Their state space is naturally T² = S¹ × S¹ — a **torus**.

### Phase Computation

For any instrument with returns r_t:

```
θ_t = arctan2(momentum_t, cumulative_return_t)

where:
  cumulative_return_t = Σ(r_{t-k}) for k = 0 to lookback
  momentum_t = d/dt(cumulative_return)

φ_t = arctan2(vol_momentum_t, vol_deviation_t)

where:
  vol_t = rolling_std(r)
  vol_deviation_t = vol_t - mean(vol)
  vol_momentum_t = d/dt(vol)
```

This maps the continuous price/vol dynamics to **angular position on the torus**.

## The Hierarchy Creates Phase Relationships

### Parent-Child Coupling

In a healthy hierarchy:
- IBB (parent) moves → GILD (child) moves proportionally
- Both should be at the **same phase** (θ_IBB ≈ θ_GILD)

But GILD has **idiosyncratic noise** (news, earnings, pipeline updates).

This noise pushes GILD's phase away from IBB's phase.

### Phase Error = Holonomy

The **phase error** is:

```
Δθ = θ_GILD - θ_IBB  (with angular wraparound)
Δφ = φ_GILD - φ_IBB

Combined signal = 0.7 * Δθ + 0.3 * Δφ
```

This is analogous to **holonomy** in differential geometry:
- Parallel transport a vector around a loop
- If it comes back rotated, there's **curvature**
- Here: "transport" GILD's expected phase from IBB's phase
- The difference is the "curvature" of the parent-child relationship

## Why This Creates Tradeable Alpha

### The Restoration Force

When phase error is large:

1. **GILD ahead of IBB (Δθ > 0)**: 
   - GILD has moved too fast relative to sector
   - It will **underperform** as it waits for sector to catch up
   - → SHORT GILD

2. **GILD behind IBB (Δθ < 0)**:
   - GILD has lagged the sector
   - It will **outperform** as it catches up
   - → LONG GILD

This is NOT momentum or mean-reversion in the traditional sense.

It's **phase synchronization** — a physical process that MUST happen because the parent-child relationship is CAUSAL.

### Why Biotech Specifically?

Biotech is ideal because:

1. **High Beta**: Strong coupling to parent (β ≈ 1.5-2.5)
2. **Idiosyncratic Events**: Clinical trials, FDA decisions create large phase dislocations
3. **Mean-Reverting Overreaction**: Market overreacts to news, then corrects
4. **Liquid Enough**: Can actually trade the signals

Compare to utilities (too stable, tiny phase errors) or tech megacaps (too dominant, they ARE the parent).

## Connection to Clifford Algebra

### Rotors as Phase Transport

In Clifford algebra, rotations are represented by **rotors**:

```
R = cos(θ/2) + sin(θ/2) * e_12
```

where e_12 is the bivector (the plane of rotation).

### Phase Composition

When we compose market moves:
- Each move is a rotor
- Composition is rotor multiplication (non-commutative!)
- Path-dependent → holonomy is well-defined

### The Bivector is Curvature

The **bivector** component of the holonomy rotor IS the curvature:

```
H = R_1 * R_2 * R_3 * ... around a loop

If H ≠ 1: there's curvature (exploitable)
log(H) = bivector = curvature "energy"
```

In our implementation, we simplify this to scalar phase angles, but the geometric structure is the same.

## Why This Is Different From Standard Stat Arb

### Standard Pair Trading

```
Spread = Price_A - β * Price_B
Trade when spread is extreme
Assumes: linear relationship, stationary spread
```

Problems:
- Spread can stay extreme for a long time
- Relationship (β) can change
- No theory for WHY it should revert

### Torus Phase Trading

```
Phase_error = θ_child - θ_parent (angular)
Trade when phase is misaligned
Assumes: phase must synchronize (physical necessity)
```

Advantages:
- Angular → bounded, natural mean-reversion
- Phase = cycle position, not level
- Restoration is FORCED by causal parent-child link

## The Information Geometry Perspective

### Fisher Information on the Torus

The torus has a natural Riemannian metric:

```
ds² = dθ² + dφ²  (flat metric in local coordinates)
```

But the **information geometry** gives us:

```
g_ij = E[∂_i log p * ∂_j log p]  (Fisher metric)
```

This tells us how "informative" each direction is.

### Geodesics = Optimal Paths

The geodesic on the torus is the "shortest path" between two phase states.

When phase error is large, the **gradient of the geodesic distance** points toward the restoration direction.

This IS the trading signal.

## Summary: Why the Architecture Finds Alpha

| Standard Approach | Torus Architecture |
|-------------------|-------------------|
| Prices are absolute | Only ratios matter (gauge symmetry) |
| Linear relationships | Angular/phase relationships |
| Spread can diverge forever | Phase is bounded (angular) |
| No theory for reversion | Reversion is physical necessity |
| Correlation-based | Causation-based (hierarchy) |
| Point predictions | Invariant restoration |

**The edge exists because:**

1. The torus correctly models the **cyclic nature** of market dynamics
2. The hierarchy captures **causal** parent-child relationships  
3. Phase error is a **symmetry violation** that MUST be restored
4. Biotech has the right properties (high beta, idiosyncratic noise, liquid)

**We're not predicting the future. We're detecting when the present violates an invariant that must be restored.**

---

## Mathematical Appendix: Phase Error Computation

```python
def compute_phase(returns, lookback=40):
    # Price/momentum phase
    cum_ret = returns.rolling(lookback).sum()
    momentum = cum_ret.diff(5)
    theta = np.arctan2(momentum, cum_ret)
    
    # Volatility phase
    vol = returns.rolling(lookback // 2).std()
    vol_mean = vol.rolling(lookback).mean()
    vol_dev = vol - vol_mean
    vol_mom = vol.diff(5)
    phi = np.arctan2(vol_mom, vol_dev)
    
    return theta, phi

def phase_error(parent_theta, parent_phi, child_theta, child_phi):
    # Angular difference with wraparound
    d_theta = np.arctan2(
        np.sin(child_theta - parent_theta),
        np.cos(child_theta - parent_theta)
    )
    d_phi = np.arctan2(
        np.sin(child_phi - parent_phi),
        np.cos(child_phi - parent_phi)
    )
    
    # Combined signal (weight momentum more)
    return 0.7 * d_theta + 0.3 * d_phi
```

The `arctan2` ensures proper angular arithmetic (handles wraparound at ±π).

The 0.7/0.3 weighting prioritizes the momentum cycle over the volatility cycle (empirically more predictive).
