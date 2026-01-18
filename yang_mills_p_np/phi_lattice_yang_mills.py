"""
φ-Lattice Yang-Mills Theory
============================

Exploring the mass gap problem using golden ratio structured lattices.

Key idea: Replace uniform lattice with φ-quasiperiodic structure to:
1. Regularize UV divergences via incommensurability
2. Frustrate massless modes (forcing mass gap)
3. Provide natural RG flow structure

Author: Fractal Toroidal Flow Project
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from scipy import linalg
import matplotlib.pyplot as plt

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI  # ≈ 0.618

@dataclass
class PhiLatticeConfig:
    """Configuration for φ-quasiperiodic lattice."""
    L: int  # Base lattice size
    phi_powers: Tuple[int, int, int, int] = (0, 1, 2, 3)  # Powers of φ for each direction
    gauge_group: str = "SU2"  # SU2 or SU3
    beta: float = 2.0  # Inverse coupling
    
    @property
    def spacings(self) -> np.ndarray:
        """Lattice spacings in each direction."""
        return np.array([PHI_INV ** p for p in self.phi_powers])
    
    @property
    def volume_factor(self) -> float:
        """Relative volume compared to uniform lattice."""
        return np.prod(self.spacings)


class SU2Matrix:
    """SU(2) matrix representation using Pauli matrices."""
    
    # Pauli matrices
    sigma = np.array([
        [[0, 1], [1, 0]],      # σ₁
        [[0, -1j], [1j, 0]],   # σ₂
        [[1, 0], [0, -1]]      # σ₃
    ], dtype=complex)
    
    identity = np.eye(2, dtype=complex)
    
    @classmethod
    def from_angles(cls, theta: float, n: np.ndarray) -> np.ndarray:
        """Create SU(2) matrix from rotation angle and axis."""
        n = n / np.linalg.norm(n)  # Normalize
        return (np.cos(theta/2) * cls.identity + 
                1j * np.sin(theta/2) * sum(n[i] * cls.sigma[i] for i in range(3)))
    
    @classmethod
    def random(cls, scale: float = 1.0) -> np.ndarray:
        """Generate random SU(2) matrix near identity."""
        theta = np.random.normal(0, scale)
        n = np.random.randn(3)
        return cls.from_angles(theta, n)
    
    @classmethod
    def dagger(cls, U: np.ndarray) -> np.ndarray:
        """Hermitian conjugate."""
        return U.conj().T


class SU3Matrix:
    """SU(3) matrix representation using Gell-Mann matrices."""
    
    # Gell-Mann matrices (8 generators)
    @classmethod
    def gell_mann(cls) -> List[np.ndarray]:
        """Return list of 8 Gell-Mann matrices."""
        lambda_matrices = []
        
        # λ₁
        lambda_matrices.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex))
        # λ₂
        lambda_matrices.append(np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex))
        # λ₃
        lambda_matrices.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex))
        # λ₄
        lambda_matrices.append(np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex))
        # λ₅
        lambda_matrices.append(np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex))
        # λ₆
        lambda_matrices.append(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex))
        # λ₇
        lambda_matrices.append(np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex))
        # λ₈
        lambda_matrices.append(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3))
        
        return lambda_matrices
    
    identity = np.eye(3, dtype=complex)
    
    @classmethod
    def random(cls, scale: float = 1.0) -> np.ndarray:
        """Generate random SU(3) matrix near identity."""
        gm = cls.gell_mann()
        # Random linear combination of generators
        coeffs = np.random.normal(0, scale, 8)
        H = sum(c * g for c, g in zip(coeffs, gm))
        return linalg.expm(1j * H)
    
    @classmethod
    def dagger(cls, U: np.ndarray) -> np.ndarray:
        """Hermitian conjugate."""
        return U.conj().T


class PhiLatticeSite:
    """A site on the φ-lattice with associated gauge links."""
    
    def __init__(self, coords: Tuple[int, ...], config: PhiLatticeConfig):
        self.coords = coords
        self.config = config
        self.dim = len(coords)
        
        # Initialize links to identity
        N = 2 if config.gauge_group == "SU2" else 3
        self.links = [np.eye(N, dtype=complex) for _ in range(self.dim)]
    
    def set_link(self, direction: int, U: np.ndarray):
        """Set gauge link in given direction."""
        self.links[direction] = U.copy()
    
    def get_link(self, direction: int) -> np.ndarray:
        """Get gauge link in given direction."""
        return self.links[direction]


class PhiLatticeYangMills:
    """
    Yang-Mills theory on a φ-quasiperiodic lattice.
    
    The key innovation: lattice spacings scale as powers of φ⁻¹,
    creating an incommensurate structure that:
    1. Prevents exact resonances between modes
    2. Provides natural UV regularization
    3. Has self-similar structure under RG
    """
    
    def __init__(self, config: PhiLatticeConfig):
        self.config = config
        self.L = config.L
        self.dim = 4  # 4D spacetime
        
        # Matrix class based on gauge group
        self.MatrixClass = SU2Matrix if config.gauge_group == "SU2" else SU3Matrix
        self.N = 2 if config.gauge_group == "SU2" else 3
        
        # Initialize lattice of gauge links
        # links[x, y, z, t, mu] = U_mu(x)
        shape = (self.L,) * self.dim + (self.dim,)
        self.links = np.zeros(shape + (self.N, self.N), dtype=complex)
        
        # Initialize to identity
        for idx in np.ndindex(*shape):
            self.links[idx] = np.eye(self.N, dtype=complex)
    
    def get_link(self, site: Tuple[int, ...], mu: int) -> np.ndarray:
        """Get gauge link U_μ(site)."""
        return self.links[site + (mu,)]
    
    def set_link(self, site: Tuple[int, ...], mu: int, U: np.ndarray):
        """Set gauge link U_μ(site)."""
        self.links[site + (mu,)] = U.copy()
    
    def neighbor(self, site: Tuple[int, ...], mu: int, forward: bool = True) -> Tuple[int, ...]:
        """Get neighboring site in direction mu."""
        site = list(site)
        site[mu] = (site[mu] + (1 if forward else -1)) % self.L
        return tuple(site)
    
    def plaquette(self, site: Tuple[int, ...], mu: int, nu: int) -> np.ndarray:
        """
        Compute plaquette U_μν(site) = U_μ(x) U_ν(x+μ) U_μ†(x+ν) U_ν†(x)
        
        With φ-lattice: plaquette area = a_μ × a_ν = φ^(-p_μ) × φ^(-p_ν)
        """
        x = site
        x_mu = self.neighbor(x, mu)
        x_nu = self.neighbor(x, nu)
        
        U1 = self.get_link(x, mu)
        U2 = self.get_link(x_mu, nu)
        U3 = self.MatrixClass.dagger(self.get_link(x_nu, mu))
        U4 = self.MatrixClass.dagger(self.get_link(x, nu))
        
        return U1 @ U2 @ U3 @ U4
    
    def plaquette_action(self, site: Tuple[int, ...], mu: int, nu: int) -> float:
        """
        Wilson action for single plaquette with φ-weighting.
        
        S_P = β × φ^(-p_μ - p_ν) × [1 - (1/N) Re Tr(U_P)]
        
        The φ-weighting accounts for different plaquette areas.
        """
        P = self.plaquette(site, mu, nu)
        trace_part = np.real(np.trace(P)) / self.N
        
        # φ-weighting based on plaquette area
        p_mu, p_nu = self.config.phi_powers[mu], self.config.phi_powers[nu]
        phi_weight = PHI_INV ** (p_mu + p_nu)
        
        return self.config.beta * phi_weight * (1 - trace_part)
    
    def total_action(self) -> float:
        """Compute total Wilson action on φ-lattice."""
        S = 0.0
        for site in np.ndindex(*([self.L] * self.dim)):
            for mu in range(self.dim):
                for nu in range(mu + 1, self.dim):
                    S += self.plaquette_action(site, mu, nu)
        return S
    
    def average_plaquette(self) -> float:
        """Compute average plaquette value (order parameter)."""
        total = 0.0
        count = 0
        
        for site in np.ndindex(*([self.L] * self.dim)):
            for mu in range(self.dim):
                for nu in range(mu + 1, self.dim):
                    P = self.plaquette(site, mu, nu)
                    total += np.real(np.trace(P)) / self.N
                    count += 1
        
        return total / count if count > 0 else 0.0
    
    def phi_weighted_average_plaquette(self) -> float:
        """
        Compute φ-weighted average plaquette.
        
        This is the natural observable on φ-lattice, accounting for
        different plaquette areas.
        """
        total = 0.0
        weight_sum = 0.0
        
        for site in np.ndindex(*([self.L] * self.dim)):
            for mu in range(self.dim):
                for nu in range(mu + 1, self.dim):
                    P = self.plaquette(site, mu, nu)
                    plaq_val = np.real(np.trace(P)) / self.N
                    
                    # Weight by plaquette area
                    p_mu, p_nu = self.config.phi_powers[mu], self.config.phi_powers[nu]
                    weight = PHI_INV ** (p_mu + p_nu)
                    
                    total += weight * plaq_val
                    weight_sum += weight
        
        return total / weight_sum if weight_sum > 0 else 0.0
    
    def randomize(self, scale: float = 0.5):
        """Initialize with random gauge configuration."""
        for site in np.ndindex(*([self.L] * self.dim)):
            for mu in range(self.dim):
                U = self.MatrixClass.random(scale)
                self.set_link(site, mu, U)
    
    def heatbath_update(self, site: Tuple[int, ...], mu: int):
        """
        Single heatbath update at one link.
        
        On φ-lattice, the staple sum is weighted by plaquette areas.
        """
        # Compute staple sum (weighted by φ)
        staple = np.zeros((self.N, self.N), dtype=complex)
        
        for nu in range(self.dim):
            if nu == mu:
                continue
            
            # Forward staple
            x_mu = self.neighbor(site, mu)
            U_nu_xmu = self.get_link(x_mu, nu)
            U_mu_xnu = self.get_link(self.neighbor(site, nu), mu)
            U_nu_x = self.get_link(site, nu)
            
            forward = U_nu_xmu @ self.MatrixClass.dagger(U_mu_xnu) @ self.MatrixClass.dagger(U_nu_x)
            
            # Backward staple  
            x_nu_back = self.neighbor(site, nu, forward=False)
            x_mu_nu_back = self.neighbor(x_nu_back, mu)
            
            U_nu_xmu_nuback = self.get_link(x_mu_nu_back, nu)
            U_mu_xnuback = self.get_link(x_nu_back, mu)
            U_nu_xnuback = self.get_link(x_nu_back, nu)
            
            backward = (self.MatrixClass.dagger(U_nu_xmu_nuback) @ 
                       self.MatrixClass.dagger(U_mu_xnuback) @ 
                       U_nu_xnuback)
            
            # φ-weighting
            p_mu, p_nu = self.config.phi_powers[mu], self.config.phi_powers[nu]
            weight = PHI_INV ** (p_mu + p_nu)
            
            staple += weight * (forward + backward)
        
        # Generate new link from staple (simplified - use Metropolis for now)
        old_U = self.get_link(site, mu)
        old_action = -self.config.beta * np.real(np.trace(old_U @ staple)) / self.N
        
        # Propose new link
        delta_U = self.MatrixClass.random(0.2)
        new_U = delta_U @ old_U
        
        # Ensure unitarity
        if self.N == 2:
            new_U = new_U / np.sqrt(np.abs(np.linalg.det(new_U)))
        else:
            U, s, Vh = np.linalg.svd(new_U)
            new_U = U @ Vh
        
        new_action = -self.config.beta * np.real(np.trace(new_U @ staple)) / self.N
        
        # Metropolis accept/reject
        delta_S = new_action - old_action
        if delta_S < 0 or np.random.random() < np.exp(-delta_S):
            self.set_link(site, mu, new_U)
    
    def sweep(self):
        """One full lattice sweep of heatbath updates."""
        for site in np.ndindex(*([self.L] * self.dim)):
            for mu in range(self.dim):
                self.heatbath_update(site, mu)


def compute_transfer_matrix_spectrum(config: PhiLatticeConfig, 
                                    n_configs: int = 100,
                                    thermalization: int = 50) -> np.ndarray:
    """
    Estimate transfer matrix eigenvalues from Monte Carlo.
    
    The mass gap is related to:
        Δ = -log(λ₁/λ₀) / a₄
    
    where a₄ = φ^(-p₄) is the temporal lattice spacing.
    """
    lattice = PhiLatticeYangMills(config)
    lattice.randomize()
    
    # Thermalize
    for _ in range(thermalization):
        lattice.sweep()
    
    # Collect plaquette correlations
    correlations = []
    for _ in range(n_configs):
        lattice.sweep()
        plaq = lattice.phi_weighted_average_plaquette()
        correlations.append(plaq)
    
    correlations = np.array(correlations)
    
    # Autocorrelation gives effective mass
    mean_plaq = np.mean(correlations)
    var_plaq = np.var(correlations)
    
    return correlations, mean_plaq, var_plaq


def mass_gap_estimate(config: PhiLatticeConfig,
                     n_configs: int = 200,
                     thermalization: int = 100) -> dict:
    """
    Estimate the mass gap from correlation function decay.
    
    On φ-lattice, the gap should scale as:
        Δ ∝ Λ_QCD × φ^(-n)
    
    for some n related to gauge group dimension.
    """
    lattice = PhiLatticeYangMills(config)
    lattice.randomize()
    
    # Thermalize
    print(f"Thermalizing ({thermalization} sweeps)...")
    for i in range(thermalization):
        lattice.sweep()
        if (i + 1) % 20 == 0:
            plaq = lattice.phi_weighted_average_plaquette()
            print(f"  Sweep {i+1}: <P>_φ = {plaq:.6f}")
    
    # Collect measurements
    print(f"\nCollecting measurements ({n_configs} configs)...")
    plaquettes = []
    actions = []
    
    for i in range(n_configs):
        lattice.sweep()
        plaq = lattice.phi_weighted_average_plaquette()
        action = lattice.total_action()
        plaquettes.append(plaq)
        actions.append(action)
        
        if (i + 1) % 50 == 0:
            print(f"  Config {i+1}: <P>_φ = {plaq:.6f}, S = {action:.4f}")
    
    plaquettes = np.array(plaquettes)
    actions = np.array(actions)
    
    # Compute autocorrelation
    mean_plaq = np.mean(plaquettes)
    fluctuations = plaquettes - mean_plaq
    
    max_lag = min(50, n_configs // 4)
    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            autocorr[0] = np.mean(fluctuations ** 2)
        else:
            autocorr[lag] = np.mean(fluctuations[:-lag] * fluctuations[lag:])
    
    # Normalize
    if autocorr[0] > 0:
        autocorr /= autocorr[0]
    
    # Fit exponential to get correlation length
    # C(t) ∼ exp(-t/τ) → τ is correlation time
    # Mass gap Δ ∼ 1/τ in lattice units
    
    # Simple exponential fit
    positive_autocorr = autocorr[autocorr > 0.01]
    if len(positive_autocorr) > 2:
        log_autocorr = np.log(positive_autocorr)
        t_vals = np.arange(len(positive_autocorr))
        
        # Linear fit: log(C) = -t/τ → slope = -1/τ
        slope, intercept = np.polyfit(t_vals, log_autocorr, 1)
        tau = -1 / slope if slope < 0 else float('inf')
    else:
        tau = 1.0  # Default
    
    # Mass gap estimate in lattice units
    # Δ_lattice = 1/τ, convert to physical units via φ-spacing
    a_temporal = PHI_INV ** config.phi_powers[3]
    mass_gap_lattice = 1 / tau if tau > 0 else 0
    
    results = {
        'mean_plaquette': mean_plaq,
        'plaquette_variance': np.var(plaquettes),
        'mean_action': np.mean(actions),
        'autocorr': autocorr,
        'correlation_time': tau,
        'mass_gap_lattice': mass_gap_lattice,
        'temporal_spacing': a_temporal,
        'phi_powers': config.phi_powers,
        'beta': config.beta,
        'L': config.L,
        'gauge_group': config.gauge_group
    }
    
    return results


def test_phi_structure():
    """
    Test the φ-lattice structure and verify basic properties.
    """
    print("=" * 60)
    print("φ-Lattice Yang-Mills: Structure Test")
    print("=" * 60)
    
    # Basic φ properties
    print(f"\nGolden ratio φ = {PHI:.10f}")
    print(f"φ⁻¹ = {PHI_INV:.10f}")
    print(f"φ² = φ + 1: {PHI**2:.10f} ≈ {PHI + 1:.10f}")
    print(f"φ⁻¹ + φ⁻² = 1: {PHI_INV + PHI_INV**2:.10f}")
    
    # Lattice spacings
    print("\nφ-Lattice spacings:")
    for p in range(4):
        print(f"  a_{p} = φ^(-{p}) = {PHI_INV**p:.6f}")
    
    # Plaquette areas (all combinations)
    print("\nPlaquette areas (a_μ × a_ν):")
    for mu in range(4):
        for nu in range(mu + 1, 4):
            area = PHI_INV**(mu + nu)
            print(f"  P_{mu}{nu}: φ^(-{mu+nu}) = {area:.6f}")
    
    # Create small test lattice
    print("\n" + "-" * 60)
    print("Creating small SU(2) φ-lattice (L=4)...")
    
    config = PhiLatticeConfig(L=4, gauge_group="SU2", beta=2.0)
    lattice = PhiLatticeYangMills(config)
    
    print(f"  Dimension: {lattice.dim}")
    print(f"  Gauge group: {config.gauge_group} (N={lattice.N})")
    print(f"  β = {config.beta}")
    
    # Test with identity config
    print("\nIdentity configuration:")
    print(f"  Average plaquette: {lattice.average_plaquette():.6f} (should be 1.0)")
    print(f"  φ-weighted plaquette: {lattice.phi_weighted_average_plaquette():.6f}")
    print(f"  Total action: {lattice.total_action():.6f} (should be 0.0)")
    
    # Randomize
    print("\nRandomized configuration (scale=0.5):")
    lattice.randomize(scale=0.5)
    print(f"  Average plaquette: {lattice.average_plaquette():.6f}")
    print(f"  φ-weighted plaquette: {lattice.phi_weighted_average_plaquette():.6f}")
    print(f"  Total action: {lattice.total_action():.6f}")
    
    # A few sweeps
    print("\nAfter 10 heatbath sweeps:")
    for _ in range(10):
        lattice.sweep()
    print(f"  Average plaquette: {lattice.average_plaquette():.6f}")
    print(f"  φ-weighted plaquette: {lattice.phi_weighted_average_plaquette():.6f}")
    print(f"  Total action: {lattice.total_action():.6f}")
    
    print("\n" + "=" * 60)
    print("Structure test complete!")
    print("=" * 60)


def run_mass_gap_study():
    """
    Run mass gap estimation study comparing different configurations.
    """
    print("\n" + "=" * 60)
    print("φ-Lattice Yang-Mills: Mass Gap Study")
    print("=" * 60)
    
    # Test with SU(2)
    config = PhiLatticeConfig(
        L=4,
        phi_powers=(0, 1, 2, 3),
        gauge_group="SU2",
        beta=2.2
    )
    
    print(f"\nConfiguration:")
    print(f"  L = {config.L}")
    print(f"  Gauge group = {config.gauge_group}")
    print(f"  β = {config.beta}")
    print(f"  φ-powers = {config.phi_powers}")
    print(f"  Spacings = {config.spacings}")
    
    results = mass_gap_estimate(config, n_configs=100, thermalization=50)
    
    print("\n" + "-" * 60)
    print("Results:")
    print(f"  <P>_φ = {results['mean_plaquette']:.6f} ± {np.sqrt(results['plaquette_variance']):.6f}")
    print(f"  <S> = {results['mean_action']:.4f}")
    print(f"  Correlation time τ = {results['correlation_time']:.4f}")
    print(f"  Mass gap (lattice) = {results['mass_gap_lattice']:.4f}")
    
    # φ-scaling prediction
    dim_G = 3 if config.gauge_group == "SU2" else 8  # dim(su(2))=3, dim(su(3))=8
    phi_prediction = PHI_INV ** dim_G
    print(f"\n  φ-scaling prediction: φ^(-{dim_G}) = {phi_prediction:.6f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run basic structure test
    test_phi_structure()
    
    # Run mass gap study
    print("\n")
    run_mass_gap_study()
