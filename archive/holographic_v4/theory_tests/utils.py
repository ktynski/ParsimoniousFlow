"""
Statistical Utilities for Theory Validation Tests
==================================================

Provides rigorous statistical testing to avoid premature conclusions.
All tests require p < 0.05 AFTER Bonferroni correction.

FUNCTIONS:
    bootstrap_confidence_interval - Non-parametric CI
    permutation_test              - Non-parametric significance
    effect_size_cohens_d          - Standardized effect size
    is_significant                - Apply Bonferroni correction
    plot_theory_vs_measured       - Visualization (optional)

PRINCIPLE: No conclusions without statistical rigor!
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    effect_size: float
    is_significant: bool
    n_tests: int  # For Bonferroni correction context


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: 1D array of observations
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (0.95 = 95%)
        seed: Random seed for reproducibility
        
    Returns:
        (estimate, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    
    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_stats[i] = statistic(data[indices])
    
    # Compute percentiles
    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(bootstrap_stats, 100 * alpha))
    ci_high = float(np.percentile(bootstrap_stats, 100 * (1 - alpha)))
    estimate = statistic(data)
    
    return float(estimate), ci_low, ci_high


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: np.mean(a) - np.mean(b),
    n_permutations: int = 10000,
    alternative: str = 'two-sided',
    seed: int = 42
) -> Tuple[float, float]:
    """
    Non-parametric permutation test for comparing two groups.
    
    Args:
        group_a: First group observations
        group_b: Second group observations
        statistic: Function(a, b) -> scalar (default: difference in means)
        n_permutations: Number of permutations
        alternative: 'two-sided', 'greater', or 'less'
        seed: Random seed
        
    Returns:
        (observed_statistic, p_value)
    """
    rng = np.random.default_rng(seed)
    
    observed = statistic(group_a, group_b)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    
    # Count extreme values
    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_stat = statistic(perm_a, perm_b)
        
        if alternative == 'two-sided':
            if abs(perm_stat) >= abs(observed):
                count_extreme += 1
        elif alternative == 'greater':
            if perm_stat >= observed:
                count_extreme += 1
        else:  # 'less'
            if perm_stat <= observed:
                count_extreme += 1
    
    p_value = (count_extreme + 1) / (n_permutations + 1)
    return float(observed), float(p_value)


def effect_size_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for two groups.
    
    Interpretation:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large
        
    Args:
        group_a, group_b: Arrays of observations
        
    Returns:
        Cohen's d (positive if a > b)
    """
    n_a, n_b = len(group_a), len(group_b)
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std < 1e-10:
        return 0.0
    
    return float((mean_a - mean_b) / pooled_std)


def is_significant(
    p_value: float,
    alpha: float = 0.05,
    correction: str = 'bonferroni',
    n_tests: int = 1
) -> bool:
    """
    Determine significance with multiple comparison correction.
    
    Args:
        p_value: Observed p-value
        alpha: Significance level before correction
        correction: 'bonferroni', 'holm', or 'none'
        n_tests: Number of tests for correction
        
    Returns:
        True if significant after correction
    """
    if correction == 'none':
        threshold = alpha
    elif correction == 'bonferroni':
        threshold = alpha / n_tests
    elif correction == 'holm':
        # Holm step-down: simplified version (assumes this is the most significant)
        threshold = alpha / n_tests
    else:
        raise ValueError(f"Unknown correction: {correction}")
    
    return p_value < threshold


def compute_full_statistical_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_bootstrap: int = 1000,
    n_permutations: int = 10000,
    n_tests: int = 1,
    seed: int = 42
) -> StatisticalResult:
    """
    Run complete statistical analysis comparing two groups.
    
    Args:
        group_a, group_b: Arrays to compare
        n_bootstrap: Bootstrap iterations for CI
        n_permutations: Permutation test iterations
        n_tests: Number of tests (for Bonferroni)
        seed: Random seed
        
    Returns:
        StatisticalResult with all computed values
    """
    # Effect size
    d = effect_size_cohens_d(group_a, group_b)
    
    # Bootstrap CI on difference
    diff = group_a - group_b[:len(group_a)] if len(group_a) == len(group_b) else None
    if diff is not None:
        est, ci_lo, ci_hi = bootstrap_confidence_interval(diff, np.mean, n_bootstrap, seed=seed)
    else:
        # Use bootstrap on each group separately
        est_a, ci_a_lo, ci_a_hi = bootstrap_confidence_interval(group_a, np.mean, n_bootstrap, seed=seed)
        est_b, ci_b_lo, ci_b_hi = bootstrap_confidence_interval(group_b, np.mean, n_bootstrap, seed=seed+1)
        est = est_a - est_b
        ci_lo = ci_a_lo - ci_b_hi  # Conservative
        ci_hi = ci_a_hi - ci_b_lo
    
    # Permutation test
    stat, p = permutation_test(group_a, group_b, n_permutations=n_permutations, seed=seed)
    
    # Significance
    sig = is_significant(p, correction='bonferroni', n_tests=n_tests)
    
    return StatisticalResult(
        statistic=stat,
        p_value=p,
        ci_low=ci_lo,
        ci_high=ci_hi,
        effect_size=d,
        is_significant=sig,
        n_tests=n_tests
    )


# =============================================================================
# THEORY-SPECIFIC UTILITIES
# =============================================================================

def compute_lyapunov_exponent(
    trajectory: np.ndarray,
    perturbation: float = 1e-6
) -> float:
    """
    Estimate Lyapunov exponent from trajectory divergence.
    
    λ = lim (1/t) ln(|δx(t)| / |δx(0)|)
    
    Args:
        trajectory: [T, ...] sequence of states
        perturbation: Initial perturbation magnitude
        
    Returns:
        Estimated Lyapunov exponent
    """
    T = len(trajectory)
    if T < 2:
        return 0.0
    
    # Compute divergence from perturbed initial condition
    divergences = []
    for t in range(1, T):
        diff = trajectory[t] - trajectory[0]
        dist = np.linalg.norm(diff.flatten())
        if dist > 0:
            divergences.append(np.log(dist / perturbation) / t)
    
    if not divergences:
        return 0.0
    
    return float(np.mean(divergences))


def compute_mutual_information(
    X: np.ndarray,
    Y: np.ndarray,
    bins: int = 10
) -> float:
    """
    Estimate mutual information I(X;Y) using histogram binning.
    
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    Args:
        X, Y: 1D arrays of equal length
        bins: Number of histogram bins
        
    Returns:
        Estimated mutual information in bits
    """
    # Compute histograms
    p_x, _ = np.histogram(X, bins=bins, density=True)
    p_y, _ = np.histogram(Y, bins=bins, density=True)
    p_xy, _, _ = np.histogram2d(X, Y, bins=bins, density=True)
    
    # Normalize
    p_x = p_x / (p_x.sum() + 1e-10)
    p_y = p_y / (p_y.sum() + 1e-10)
    p_xy = p_xy / (p_xy.sum() + 1e-10)
    
    # Compute entropies
    H_x = -np.sum(p_x[p_x > 0] * np.log2(p_x[p_x > 0] + 1e-10))
    H_y = -np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0] + 1e-10))
    H_xy = -np.sum(p_xy[p_xy > 0] * np.log2(p_xy[p_xy > 0] + 1e-10))
    
    return float(max(0, H_x + H_y - H_xy))


def compute_enstrophy_from_matrix(M: np.ndarray, basis: np.ndarray) -> float:
    """
    Compute enstrophy (||grade-2||²) from 4×4 matrix.
    
    Uses grade-2 indices: [5, 6, 7, 8, 9, 10]
    
    Args:
        M: [4, 4] matrix
        basis: [16, 4, 4] Clifford basis
        
    Returns:
        Enstrophy (bivector energy)
    """
    grade2_indices = [5, 6, 7, 8, 9, 10]
    enstrophy = 0.0
    
    for idx in grade2_indices:
        coeff = np.sum(M * basis[idx]) / np.sum(basis[idx] * basis[idx])
        enstrophy += coeff ** 2
    
    return float(enstrophy)


def compute_grade_energy_distribution(M: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Compute energy distribution across grades 0-4.
    
    Args:
        M: [4, 4] matrix
        basis: [16, 4, 4] Clifford basis
        
    Returns:
        [5] array of grade energies (sum of squared coefficients per grade)
    """
    grade_indices = {
        0: [0],
        1: [1, 2, 3, 4],
        2: [5, 6, 7, 8, 9, 10],
        3: [11, 12, 13, 14],
        4: [15]
    }
    
    energies = np.zeros(5)
    for grade, indices in grade_indices.items():
        for idx in indices:
            coeff = np.sum(M * basis[idx]) / np.sum(basis[idx] * basis[idx])
            energies[grade] += coeff ** 2
    
    return energies


def assert_theory_prediction(
    measured: float,
    predicted: float,
    tolerance: float,
    description: str,
    allow_deviation: bool = False
) -> bool:
    """
    Assert that measured value matches theory prediction within tolerance.
    
    Args:
        measured: Empirically measured value
        predicted: Theoretically predicted value
        tolerance: Acceptable relative error
        description: What we're testing (for error messages)
        allow_deviation: If True, return bool instead of asserting
        
    Returns:
        True if within tolerance
        
    Raises:
        AssertionError if not within tolerance and allow_deviation=False
    """
    if predicted == 0:
        error = abs(measured)
    else:
        error = abs(measured - predicted) / abs(predicted)
    
    passed = error <= tolerance
    
    if not allow_deviation:
        assert passed, (
            f"Theory prediction failed for {description}:\n"
            f"  Predicted: {predicted:.6f}\n"
            f"  Measured:  {measured:.6f}\n"
            f"  Error:     {error:.2%} (tolerance: {tolerance:.2%})"
        )
    
    return passed


# =============================================================================
# VISUALIZATION (Optional — used if matplotlib available)
# =============================================================================

def plot_theory_vs_measured(
    predicted: np.ndarray,
    measured: np.ndarray,
    xlabel: str = "Predicted",
    ylabel: str = "Measured",
    title: str = "Theory vs Measurement",
    save_path: Optional[str] = None
) -> None:
    """
    Plot theory predictions against measured values.
    
    Args:
        predicted: Array of predictions
        measured: Array of measurements
        xlabel, ylabel, title: Plot labels
        save_path: If provided, save figure to this path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(predicted, measured, alpha=0.5, edgecolor='k', linewidth=0.5)
    
    # Perfect agreement line
    lims = [
        min(min(predicted), min(measured)),
        max(max(predicted), max(measured))
    ]
    ax.plot(lims, lims, 'r--', label='Perfect agreement')
    
    # Labels
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correlation
    corr = np.corrcoef(predicted, measured)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()
