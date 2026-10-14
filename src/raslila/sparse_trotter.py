"""Sparse Trotterized Evolution - for scaling beyond 10 qubits.

Implements:
- Sparse Pauli string representation of Hamiltonian
- First and second-order Trotter-Suzuki decomposition
- Commuting group optimization (parallelize commuting terms)
- Integration with PennyLane/Qiskit for hardware execution
"""
import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import itertools

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)
PAULI_MAP = {'X': X, 'Y': Y, 'Z': Z, 'I': I}


@dataclass
class PauliTerm:
    """Single Pauli string term with coefficient."""
    pauli_string: str  # e.g., "ZIZXII" (qubit 0 = leftmost/MSB)
    coeff: float

    def __post_init__(self):
        self.n_qubits = len(self.pauli_string)

    def to_matrix(self) -> np.ndarray:
        """Convert to dense matrix (only for small n)."""
        ops = [PAULI_MAP[c] for c in self.pauli_string]
        H = ops[0]
        for op in ops[1:]:
            H = np.kron(H, op)
        return self.coeff * H

    def to_sparse_pauli(self) -> List[Tuple[int, str, float]]:
        """Return list of (qubit, pauli, coeff) for non-identity terms."""
        return [(i, c, self.coeff) for i, c in enumerate(self.pauli_string) if c != 'I']

    def commutes_with(self, other: 'PauliTerm') -> bool:
        """Check if two Pauli terms commute."""
        if self.n_qubits != other.n_qubits:
            return False
        # Two Pauli strings commute iff they have even number of
        # positions where both are non-identity and different
        anticommute = 0
        for c1, c2 in zip(self.pauli_string, other.pauli_string):
            if c1 != 'I' and c2 != 'I' and c1 != c2:
                anticommute += 1
        return anticommute % 2 == 0


class SparseHamiltonian:
    """
    Sparse Hamiltonian representation for large-scale simulation.

    Stores Hamiltonian as list of PauliTerm objects.
    Supports Trotterized evolution, commuting group extraction,
    and export to PennyLane/Qiskit.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.terms: List[PauliTerm] = []

    def add_term(self, pauli_string: str, coeff: float):
        if abs(coeff) > 1e-12:
            self.terms.append(PauliTerm(pauli_string, coeff))

    def add_zz(self, i: int, j: int, coeff: float):
        """Add Z_i Z_j term."""
        s = ['I'] * self.n_qubits
        s[i] = 'Z'
        s[j] = 'Z'
        self.add_term(''.join(s), coeff)

    def add_x(self, i: int, coeff: float):
        """Add X_i term."""
        s = ['I'] * self.n_qubits
        s[i] = 'X'
        self.add_term(''.join(s), coeff)

    def add_y(self, i: int, coeff: float):
        """Add Y_i term."""
        s = ['I'] * self.n_qubits
        s[i] = 'Y'
        self.add_term(''.join(s), coeff)

    def add_xy(self, i: int, j: int, coeff: float, which: str = 'XY'):
        """Add X_i Y_j or X_i X_j or Y_i Y_j etc."""
        s = ['I'] * self.n_qubits
        s[i] = which[0]
        s[j] = which[1]
        self.add_term(''.join(s), coeff)

    @classmethod
    def from_raslila_ham(cls, raslila_ham: 'RaslilaHamiltonian') -> 'SparseHamiltonian':
        """Convert RaslilaHamiltonian to SparseHamiltonian."""
        ham = cls(raslila_ham.n)
        # Single-qubit Z biases
        for i, h in enumerate(raslila_ham.h_z):
            if abs(h) > 1e-12:
                ham.add_z(i, h)
        # Single-qubit X fields
        for i, h in enumerate(raslila_ham.h_x):
            if abs(h) > 1e-12:
                ham.add_x(i, h)
        # ZZ couplings
        for (i, j), v in raslila_ham.J.items():
            if abs(v) > 1e-12:
                ham.add_zz(i, j, v)
        # Interference terms
        a, b = raslila_ham.a, raslila_ham.b
        if abs(raslila_ham.alpha) > 1e-12:
            ham.add_xy(a, b, raslila_ham.alpha, 'XX')
        if abs(raslila_ham.beta) > 1e-12:
            ham.add_xy(a, b, raslila_ham.beta, 'YY')
        return ham

    def add_z(self, i: int, coeff: float):
        s = ['I'] * self.n_qubits
        s[i] = 'Z'
        self.add_term(''.join(s), coeff)

    def to_dense(self) -> np.ndarray:
        """Convert to dense matrix (only for verification)."""
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)
        for term in self.terms:
            H += term.to_matrix()
        return H

    def commuting_groups(self) -> List[List[PauliTerm]]:
        """
        Partition terms into commuting groups for parallel Trotterization.

        Uses greedy coloring on commutativity graph.
        """
        n = len(self.terms)
        if n == 0:
            return []

        # Build adjacency matrix (True = commute)
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i, n):
                comm = self.terms[i].commutes_with(self.terms[j])
                adj[i, j] = adj[j, i] = comm

        # Greedy graph coloring (each color = commuting group)
        colors = [-1] * n
        for i in range(n):
            used = {colors[j] for j in range(i) if not adj[i, j] and colors[j] != -1}
            c = 0
            while c in used:
                c += 1
            colors[i] = c

        groups = defaultdict(list)
        for i, c in enumerate(colors):
            groups[c].append(self.terms[i])

        return [groups[c] for c in sorted(groups.keys())]

    def trotter_step(self, state: np.ndarray, dt: float,
                     order: int = 1) -> np.ndarray:
        """
        Apply one Trotter step to state vector.

        order=1: first-order (Lie-Trotter) exp(-iHdt) ≈ ∏ exp(-iH_k dt)
        order=2: second-order (Suzuki) exp(-iHdt) ≈ ∏ exp(-iH_k dt/2) ∏ exp(-iH_k dt/2) reversed
        """
        if order == 1:
            return self._trotter_first_order(state, dt)
        elif order == 2:
            return self._trotter_second_order(state, dt)
        else:
            raise ValueError("Order must be 1 or 2")

    def _trotter_first_order(self, state: np.ndarray, dt: float) -> np.ndarray:
        psi = state.copy()
        for term in self.terms:
            U = self._expm_pauli(term, -dt)
            psi = U @ psi
        return psi

    def _trotter_second_order(self, state: np.ndarray, dt: float) -> np.ndarray:
        psi = state.copy()
        half_dt = dt / 2.0
        # Forward
        for term in self.terms:
            U = self._expm_pauli(term, -half_dt)
            psi = U @ psi
        # Backward
        for term in reversed(self.terms):
            U = self._expm_pauli(term, -half_dt)
            psi = U @ psi
        return psi

    def _expm_pauli(self, term: PauliTerm, t: float) -> np.ndarray:
        """Compute exp(-i * coeff * t * P) for Pauli string P."""
        # For Pauli string, exp(-i θ P) = cos(θ) I - i sin(θ) P
        # since P^2 = I
        theta = term.coeff * t
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Build P matrix
        P = term.to_matrix() / term.coeff if term.coeff != 0 else np.eye(2**self.n_qubits, dtype=complex)

        return cos_t * np.eye(2**self.n_qubits, dtype=complex) - 1j * sin_t * P

    def evolve(self, init_state: np.ndarray, total_time: float,
               n_steps: int, order: int = 2) -> np.ndarray:
        """Full Trotterized evolution."""
        dt = total_time / n_steps
        psi = init_state.copy()
        for _ in range(n_steps):
            psi = self.trotter_step(psi, dt, order)
        return psi

    def to_pennylane_hamiltonian(self):
        """Export to PennyLane Hamiltonian for hardware execution."""
        try:
            import pennylane as qml
            coeffs = []
            obs = []
            for term in self.terms:
                coeffs.append(term.coeff)
                # Build PennyLane observable
                pauli_ops = []
                for i, c in enumerate(term.pauli_string):
                    if c != 'I':
                        if c == 'X':
                            pauli_ops.append(qml.PauliX(i))
                        elif c == 'Y':
                            pauli_ops.append(qml.PauliY(i))
                        elif c == 'Z':
                            pauli_ops.append(qml.PauliZ(i))
                if pauli_ops:
                    obs.append(pauli_ops[0] if len(pauli_ops) == 1 else qml.prod(*pauli_ops))
                else:
                    obs.append(qml.Identity(0))
            return qml.Hamiltonian(coeffs, obs)
        except ImportError:
            raise ImportError("PennyLane not installed. pip install pennylane")

    def to_qiskit_operator(self):
        """Export to Qiskit SparsePauliOp."""
        try:
            from qiskit.quantum_info import SparsePauliOp
            pauli_strings = []
            coeffs = []
            for term in self.terms:
                pauli_strings.append(term.pauli_string)
                coeffs.append(term.coeff)
            return SparsePauliOp(pauli_strings, coeffs)
        except ImportError:
            raise ImportError("Qiskit not installed. pip install qiskit")


def build_raslila_sparse_ham(n_layers: int, n_universes: int,
                              interference_qubits: Tuple[int, int] = (0, 2)) -> SparseHamiltonian:
    """Build a sparse Hamiltonian template for given grid size."""
    n_qubits = n_layers * n_universes
    ham = SparseHamiltonian(n_qubits)

    def q(L, u): return L * n_universes + u
    eps = 1e-8  # small placeholder value to keep structure

    # Horizontal edges (within layer)
    for L in range(n_layers):
        for u in range(n_universes - 1):
            ham.add_zz(q(L, u), q(L, u+1), eps)

    # Vertical edges (across layers)
    for L in range(n_layers - 1):
        for u in range(n_universes):
            ham.add_zz(q(L, u), q(L+1, u), eps)

    # Bridge edge (Radha - primary entangled pair)
    if n_qubits >= 5:
        ham.add_zz(q(0, 0), q(1, 1), eps)

    # Interference terms (alpha XX + beta YY)
    a, b = interference_qubits
    ham.add_xy(a, b, eps, 'XX')  # alpha
    ham.add_xy(a, b, eps, 'YY')  # beta

    return ham


def update_sparse_ham_j(ham: SparseHamiltonian, J_dict: Dict[Tuple[int, int], float]):
    """Update J couplings in sparse Hamiltonian from MLP outputs."""
    # Remove old ZZ terms for edges in J_dict
    ham.terms = [t for t in ham.terms if not (
        sum(c != 'I' for c in t.pauli_string) == 2 and
        all(c in 'ZI' for c in t.pauli_string) and
        tuple(i for i, c in enumerate(t.pauli_string) if c == 'Z') in J_dict
    )]
    # Add new ZZ terms
    for (i, j), val in J_dict.items():
        if abs(val) > 1e-12:
            ham.add_zz(i, j, val)