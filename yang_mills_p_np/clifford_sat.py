"""
Clifford-SAT: Encoding Boolean Satisfiability in Clifford Algebra
=================================================================

This module implements:
1. Encoding of SAT instances in Clifford algebra
2. φ-weighted SAT instances (φ-SAT)
3. Analysis of computational hardness via spectral properties
4. Testing the φ-barrier hypothesis

Key insight: The φ-structure that creates mass gaps in Yang-Mills
may also create computational barriers in SAT.

Author: Fractal Toroidal Flow Project
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from itertools import product
import time
from collections import defaultdict

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


# =============================================================================
# Part 1: Clifford Algebra Implementation for SAT
# =============================================================================

class CliffordAlgebra:
    """
    Clifford algebra Cl(n,0) for encoding n Boolean variables.
    
    Basis elements: 1, e₁, e₂, ..., eₙ, e₁e₂, ..., e₁e₂...eₙ
    Total dimension: 2^n
    """
    
    def __init__(self, n: int):
        """Initialize Cl(n,0) with n generators."""
        self.n = n
        self.dim = 2 ** n
        
        # Precompute basis element indices
        # Each basis element is indexed by a binary number indicating which eᵢ are present
        self.basis_labels = [self._binary_to_label(i) for i in range(self.dim)]
    
    def _binary_to_label(self, idx: int) -> str:
        """Convert binary index to basis element label like 'e1e3'."""
        if idx == 0:
            return "1"
        bits = []
        for i in range(self.n):
            if idx & (1 << i):
                bits.append(f"e{i+1}")
        return "".join(bits)
    
    def grade(self, idx: int) -> int:
        """Return the grade of basis element at index idx."""
        return bin(idx).count('1')
    
    def zero(self) -> np.ndarray:
        """Return zero multivector."""
        return np.zeros(self.dim)
    
    def scalar(self, s: float) -> np.ndarray:
        """Return scalar multivector."""
        M = self.zero()
        M[0] = s
        return M
    
    def basis(self, i: int) -> np.ndarray:
        """Return basis vector eᵢ (1-indexed)."""
        M = self.zero()
        M[1 << (i-1)] = 1.0
        return M
    
    def geometric_product(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute geometric product A * B in Cl(n,0).
        
        For orthonormal basis: eᵢeᵢ = 1 (positive signature)
        eᵢeⱼ = -eⱼeᵢ for i ≠ j
        """
        result = self.zero()
        
        for i in range(self.dim):
            if A[i] == 0:
                continue
            for j in range(self.dim):
                if B[j] == 0:
                    continue
                
                # Compute product of basis elements i and j
                # Result index is XOR of the two
                k = i ^ j
                
                # Compute sign from anticommutations
                sign = self._product_sign(i, j)
                
                result[k] += sign * A[i] * B[j]
        
        return result
    
    def _product_sign(self, i: int, j: int) -> int:
        """
        Compute sign of product of basis elements i and j.
        
        Count number of swaps needed to reorder.
        """
        sign = 1
        
        # For each bit in j, count how many bits in i are to its right
        # Each such bit requires a swap (anticommutation)
        for bit_j in range(self.n):
            if not (j & (1 << bit_j)):
                continue
            
            # Count bits in i that are at positions > bit_j
            for bit_i in range(bit_j + 1, self.n):
                if i & (1 << bit_i):
                    sign *= -1
        
        return sign
    
    def scalar_part(self, M: np.ndarray) -> float:
        """Extract scalar (grade-0) part."""
        return M[0]
    
    def grade_project(self, M: np.ndarray, k: int) -> np.ndarray:
        """Project to grade k."""
        result = self.zero()
        for i in range(self.dim):
            if self.grade(i) == k:
                result[i] = M[i]
        return result
    
    def norm_squared(self, M: np.ndarray) -> float:
        """Compute |M|² = sum of squares of components."""
        return np.sum(M ** 2)


# =============================================================================
# Part 2: SAT to Clifford Encoding
# =============================================================================

@dataclass
class Clause:
    """A clause is a disjunction of literals."""
    literals: List[int]  # Positive = variable, negative = negated variable
    
    def __str__(self):
        parts = []
        for lit in self.literals:
            if lit > 0:
                parts.append(f"x{lit}")
            else:
                parts.append(f"¬x{-lit}")
        return "(" + " ∨ ".join(parts) + ")"


@dataclass  
class SATInstance:
    """A SAT instance in CNF."""
    n_vars: int
    clauses: List[Clause]
    
    def __str__(self):
        return " ∧ ".join(str(c) for c in self.clauses)
    
    def evaluate(self, assignment: List[bool]) -> bool:
        """Evaluate the formula under given assignment."""
        for clause in self.clauses:
            clause_true = False
            for lit in clause.literals:
                var_idx = abs(lit) - 1
                var_val = assignment[var_idx]
                lit_val = var_val if lit > 0 else not var_val
                if lit_val:
                    clause_true = True
                    break
            if not clause_true:
                return False
        return True


class CliffordSATEncoder:
    """
    Encode SAT instances in Clifford algebra.
    
    Encoding scheme:
    - Variable xᵢ → basis vector eᵢ
    - Literal xᵢ → (1 + eᵢ)/2 (projector onto "true")
    - Literal ¬xᵢ → (1 - eᵢ)/2 (projector onto "false")  
    - Clause → sum of literal projectors
    - Formula → product of clause encodings
    
    A formula is satisfiable iff there exists an assignment such that
    the product of clause encodings has non-zero scalar part.
    """
    
    def __init__(self, n_vars: int):
        self.n = n_vars
        self.algebra = CliffordAlgebra(n_vars)
    
    def encode_literal(self, lit: int) -> np.ndarray:
        """Encode a literal as projector (1 ± eᵢ)/2."""
        var = abs(lit)
        e_i = self.algebra.basis(var)
        one = self.algebra.scalar(1.0)
        
        if lit > 0:
            # (1 + eᵢ)/2
            return 0.5 * (one + e_i)
        else:
            # (1 - eᵢ)/2
            return 0.5 * (one - e_i)
    
    def encode_clause(self, clause: Clause) -> np.ndarray:
        """
        Encode a clause as sum of literal projectors.
        
        This represents "at least one literal is true".
        """
        result = self.algebra.zero()
        for lit in clause.literals:
            result = result + self.encode_literal(lit)
        return result
    
    def encode_formula(self, sat: SATInstance) -> np.ndarray:
        """
        Encode entire formula as product of clause encodings.
        """
        if not sat.clauses:
            return self.algebra.scalar(1.0)
        
        result = self.encode_clause(sat.clauses[0])
        for clause in sat.clauses[1:]:
            clause_mv = self.encode_clause(clause)
            result = self.algebra.geometric_product(result, clause_mv)
        
        return result
    
    def evaluate_assignment(self, sat: SATInstance, assignment: List[bool]) -> float:
        """
        Evaluate the Clifford encoding for a specific assignment.
        
        The assignment determines a "collapse" of the multivector to a scalar.
        """
        # Create assignment multivector
        # For each variable, project onto the assigned value
        M = self.encode_formula(sat)
        
        # Project each variable
        for i, val in enumerate(assignment):
            proj = self.encode_literal(i+1 if val else -(i+1))
            M = self.algebra.geometric_product(proj, M)
            M = self.algebra.geometric_product(M, proj)
        
        return self.algebra.scalar_part(M)


# =============================================================================
# Part 3: φ-Weighted SAT (φ-SAT)
# =============================================================================

class PhiSATEncoder(CliffordSATEncoder):
    """
    φ-weighted SAT encoding.
    
    The φ-weighting adds a penalty based on grade:
    - Lower grades are "preferred" (weighted by φ^0 = 1)
    - Higher grades are "penalized" (weighted by φ^(-k))
    
    This creates an "energy landscape" where solutions with
    low-grade structure are easier to find.
    """
    
    def __init__(self, n_vars: int):
        super().__init__(n_vars)
    
    def phi_weight(self, M: np.ndarray) -> np.ndarray:
        """Apply φ-weighting: scale grade-k components by φ^(-k)."""
        result = self.algebra.zero()
        for i in range(self.algebra.dim):
            k = self.algebra.grade(i)
            result[i] = M[i] * PHI_INV ** k
        return result
    
    def encode_formula_phi(self, sat: SATInstance) -> np.ndarray:
        """Encode formula with φ-weighting."""
        M = self.encode_formula(sat)
        return self.phi_weight(M)
    
    def spectral_gap(self, sat: SATInstance) -> float:
        """
        Compute the "spectral gap" of the φ-weighted encoding.
        
        This measures how separated the satisfying assignments are
        from the unsatisfying ones in the Clifford space.
        """
        M = self.encode_formula_phi(sat)
        
        # Compute scalar part for each assignment
        values = []
        for bits in range(2 ** sat.n_vars):
            assignment = [(bits >> i) & 1 == 1 for i in range(sat.n_vars)]
            val = self.evaluate_assignment(sat, assignment)
            values.append((val, sat.evaluate(assignment)))
        
        # Find gap between satisfying and unsatisfying
        sat_values = [v for v, is_sat in values if is_sat]
        unsat_values = [v for v, is_sat in values if not is_sat]
        
        if not sat_values:
            return 0.0  # Unsatisfiable
        if not unsat_values:
            return float('inf')  # All assignments satisfy
        
        min_sat = min(abs(v) for v in sat_values) if sat_values else 0
        max_unsat = max(abs(v) for v in unsat_values) if unsat_values else 0
        
        return min_sat - max_unsat


# =============================================================================
# Part 4: SAT Instance Generation
# =============================================================================

def generate_random_3sat(n_vars: int, n_clauses: int, seed: int = None) -> SATInstance:
    """Generate a random 3-SAT instance."""
    if seed is not None:
        np.random.seed(seed)
    
    clauses = []
    for _ in range(n_clauses):
        # Choose 3 distinct variables
        vars = np.random.choice(range(1, n_vars + 1), size=3, replace=False)
        # Random polarity
        literals = [int(v) * (1 if np.random.random() > 0.5 else -1) for v in vars]
        clauses.append(Clause(literals))
    
    return SATInstance(n_vars, clauses)


def generate_hard_sat(n_vars: int, clause_ratio: float = 4.3) -> SATInstance:
    """
    Generate a hard SAT instance at the phase transition.
    
    The hardest 3-SAT instances occur around m/n ≈ 4.3
    where m = number of clauses and n = number of variables.
    """
    n_clauses = int(n_vars * clause_ratio)
    return generate_random_3sat(n_vars, n_clauses)


def generate_phi_structured_sat(n_vars: int, n_clauses: int) -> SATInstance:
    """
    Generate SAT instance with φ-structured clause weights.
    
    Clauses are arranged so that their "importance" follows φ-scaling.
    """
    clauses = []
    
    for i in range(n_clauses):
        # Clause involves variables with indices scaled by φ
        base_var = int(1 + (i * PHI) % n_vars)
        vars = [(base_var + j) % n_vars + 1 for j in range(3)]
        vars = list(set(vars))  # Ensure distinct
        while len(vars) < 3:
            vars.append(np.random.randint(1, n_vars + 1))
            vars = list(set(vars))
        vars = vars[:3]
        
        # Polarity based on φ
        literals = []
        for j, v in enumerate(vars):
            sign = 1 if ((i + j) * PHI) % 1 < 0.5 else -1
            literals.append(int(v) * sign)
        
        clauses.append(Clause(literals))
    
    return SATInstance(n_vars, clauses)


# =============================================================================
# Part 5: Hardness Analysis
# =============================================================================

def brute_force_sat(sat: SATInstance) -> Tuple[Optional[List[bool]], int]:
    """
    Brute force SAT solver. Returns (solution, steps).
    """
    steps = 0
    for bits in range(2 ** sat.n_vars):
        steps += 1
        assignment = [(bits >> i) & 1 == 1 for i in range(sat.n_vars)]
        if sat.evaluate(assignment):
            return assignment, steps
    return None, steps


def analyze_hardness(sat: SATInstance) -> Dict:
    """
    Analyze the computational hardness of a SAT instance.
    """
    n = sat.n_vars
    m = len(sat.clauses)
    
    # Solve it
    start_time = time.time()
    solution, steps = brute_force_sat(sat)
    solve_time = time.time() - start_time
    
    # Compute φ-properties
    encoder = PhiSATEncoder(n)
    
    # Encode the formula
    M = encoder.encode_formula(sat)
    M_phi = encoder.encode_formula_phi(sat)
    
    # Grade distribution
    grade_dist = defaultdict(float)
    for i in range(encoder.algebra.dim):
        k = encoder.algebra.grade(i)
        grade_dist[k] += M[i] ** 2
    
    # φ-weighted grade distribution
    phi_grade_dist = defaultdict(float)
    for i in range(encoder.algebra.dim):
        k = encoder.algebra.grade(i)
        phi_grade_dist[k] += M_phi[i] ** 2
    
    # Scalar content
    scalar_content = M[0] / (np.linalg.norm(M) + 1e-10)
    phi_scalar_content = M_phi[0] / (np.linalg.norm(M_phi) + 1e-10)
    
    return {
        'n_vars': n,
        'n_clauses': m,
        'clause_ratio': m / n,
        'satisfiable': solution is not None,
        'solution': solution,
        'steps': steps,
        'solve_time': solve_time,
        'grade_distribution': dict(grade_dist),
        'phi_grade_distribution': dict(phi_grade_dist),
        'scalar_content': scalar_content,
        'phi_scalar_content': phi_scalar_content,
        'norm': np.linalg.norm(M),
        'phi_norm': np.linalg.norm(M_phi),
    }


def test_phi_barrier_hypothesis(max_vars: int = 10, trials: int = 10):
    """
    Test the φ-barrier hypothesis.
    
    Hypothesis: φ-structured SAT instances are harder than random ones.
    """
    print("=" * 70)
    print("φ-BARRIER HYPOTHESIS TEST")
    print("=" * 70)
    
    results = []
    
    for n in range(3, max_vars + 1):
        m = int(4.3 * n)  # Phase transition ratio
        
        random_steps = []
        phi_steps = []
        
        for t in range(trials):
            # Random SAT
            sat_random = generate_random_3sat(n, m, seed=t*1000+n)
            _, steps_random = brute_force_sat(sat_random)
            random_steps.append(steps_random)
            
            # φ-structured SAT
            np.random.seed(t*1000+n)  # Same seed for fairness
            sat_phi = generate_phi_structured_sat(n, m)
            _, steps_phi = brute_force_sat(sat_phi)
            phi_steps.append(steps_phi)
        
        avg_random = np.mean(random_steps)
        avg_phi = np.mean(phi_steps)
        
        results.append({
            'n': n,
            'm': m,
            'avg_random_steps': avg_random,
            'avg_phi_steps': avg_phi,
            'ratio': avg_phi / avg_random if avg_random > 0 else 0,
        })
    
    # Print results
    print(f"\n{'n':<5} {'m':<5} {'Random Steps':<15} {'φ-Steps':<15} {'Ratio φ/Random':<15}")
    print("-" * 55)
    for r in results:
        print(f"{r['n']:<5} {r['m']:<5} {r['avg_random_steps']:<15.1f} {r['avg_phi_steps']:<15.1f} {r['ratio']:<15.3f}")
    
    # Analyze scaling
    ns = [r['n'] for r in results]
    random_scaling = [r['avg_random_steps'] for r in results]
    phi_scaling = [r['avg_phi_steps'] for r in results]
    
    print("\n" + "-" * 55)
    print("SCALING ANALYSIS:")
    
    # Fit exponential: steps ~ c * base^n
    if len(ns) >= 3:
        log_random = np.log(random_scaling)
        log_phi = np.log(phi_scaling)
        
        # Linear fit in log space
        coef_random = np.polyfit(ns, log_random, 1)
        coef_phi = np.polyfit(ns, log_phi, 1)
        
        base_random = np.exp(coef_random[0])
        base_phi = np.exp(coef_phi[0])
        
        print(f"\nRandom SAT: steps ~ {np.exp(coef_random[1]):.2f} × {base_random:.3f}^n")
        print(f"φ-SAT:      steps ~ {np.exp(coef_phi[1]):.2f} × {base_phi:.3f}^n")
        print(f"\nBase ratio: {base_phi/base_random:.3f}")
        print(f"φ scaling:  {base_phi:.3f} vs theoretical φ = {PHI:.3f}")
    
    return results


# =============================================================================
# Part 6: Main Analysis
# =============================================================================

def main():
    """Run all analyses."""
    
    print("=" * 70)
    print("CLIFFORD-SAT: Encoding SAT in Clifford Algebra")
    print("=" * 70)
    
    # Example SAT instance
    print("\n1. EXAMPLE ENCODING")
    print("-" * 70)
    
    # (x₁ ∨ x₂ ∨ ¬x₃) ∧ (¬x₁ ∨ x₃) ∧ (x₂ ∨ ¬x₃)
    sat = SATInstance(3, [
        Clause([1, 2, -3]),
        Clause([-1, 3]),
        Clause([2, -3])
    ])
    
    print(f"Formula: {sat}")
    
    encoder = CliffordSATEncoder(3)
    M = encoder.encode_formula(sat)
    
    print(f"\nClifford encoding (first few components):")
    for i in range(min(8, len(M))):
        if abs(M[i]) > 1e-10:
            print(f"  {encoder.algebra.basis_labels[i]}: {M[i]:.4f}")
    
    print(f"\nScalar part: {encoder.algebra.scalar_part(M):.4f}")
    print(f"Norm: {np.linalg.norm(M):.4f}")
    
    # Test all assignments
    print("\nAssignment evaluation:")
    for bits in range(8):
        assignment = [(bits >> i) & 1 == 1 for i in range(3)]
        is_sat = sat.evaluate(assignment)
        cliff_val = encoder.evaluate_assignment(sat, assignment)
        status = "✓" if is_sat else "✗"
        print(f"  {assignment} → SAT: {status}, Clifford: {cliff_val:.4f}")
    
    # Test φ-barrier hypothesis
    print("\n" + "=" * 70)
    test_phi_barrier_hypothesis(max_vars=8, trials=20)
    
    # Hardness analysis
    print("\n" + "=" * 70)
    print("HARDNESS ANALYSIS")
    print("=" * 70)
    
    for n in [4, 5, 6]:
        m = int(4.3 * n)
        sat = generate_hard_sat(n)
        analysis = analyze_hardness(sat)
        
        print(f"\nn={n}, m={m}:")
        print(f"  Satisfiable: {analysis['satisfiable']}")
        print(f"  Steps: {analysis['steps']}")
        print(f"  Scalar content: {analysis['scalar_content']:.4f}")
        print(f"  φ-scalar content: {analysis['phi_scalar_content']:.4f}")
    
    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
