"""True Quantum Annealing - time-dependent Hamiltonian evolution.

Implements the "cooling" metaphor from Phase 4 properly:
- Start with strong transverse field (high temperature/exploration)
- Gradually reduce transverse field, increase problem Hamiltonian (cooling)
- Final state is ground state of problem Hamiltonian
- Can add dissipation (Lindblad) for open quantum system simulation
"""
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from scipy.linalg import expm
from .hamiltonian import RaslilaHamiltonian


@dataclass
class AnnealingSchedule:
    """Defines the annealing schedule s(t) in [0,1]."""
    total_time: float
    n_steps: int
    schedule_type: str = "linear"  # linear, exponential, custom

    def __post_init__(self):
        self.dt = self.total_time / self.n_steps

    def s(self, step: int) -> float:
        """Schedule parameter at step (0 to n_steps)."""
        t = step / self.n_steps
        if self.schedule_type == "linear":
            return t
        elif self.schedule_type == "exponential":
            return 1.0 - np.exp(-5.0 * t)
        elif self.schedule_type == "cosine":
            return 0.5 * (1.0 - np.cos(np.pi * t))
        else:
            return t


class TimeDependentHamiltonian:
    """
    H(s) = (1-s) * H_driver + s * H_problem

    H_driver = -sum_i X_i  (transverse field, causes quantum fluctuations)
    H_problem = H_raslila (the normalized Raslila Hamiltonian)

    As s goes 0 -> 1, system evolves from ground state of driver
    to ground state of problem Hamiltonian.
    """

    def __init__(self, n_qubits: int, problem_ham: RaslilaHamiltonian,
                 driver_strength: float = 1.0):
        self.n_qubits = n_qubits
        self.problem_ham = problem_ham
        self.driver_strength = driver_strength

        # Build driver Hamiltonian: -sum X_i
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        I = np.eye(2, dtype=complex)
        self.H_driver = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        for i in range(n_qubits):
            ops = [X if j == i else I for j in range(n_qubits)]
            H_i = ops[0]
            for op in ops[1:]:
                H_i = np.kron(H_i, op)
            self.H_driver -= driver_strength * H_i

    def hamiltonian_at(self, s: float) -> np.ndarray:
        """Return H(s) = (1-s) * H_driver + s * H_problem."""
        H_problem = self.problem_ham.construct_dense()
        return (1.0 - s) * self.H_driver + s * H_problem

    def ground_state(self, s: float) -> np.ndarray:
        """Compute ground state of H(s) via eigendecomposition (small systems only)."""
        H = self.hamiltonian_at(s)
        evals, evecs = np.linalg.eigh(H)
        return evecs[:, 0]


def annealed_evolve(ham_builder: TimeDependentHamiltonian,
                    init_state: np.ndarray,
                    schedule: AnnealingSchedule,
                    add_dephasing: bool = False,
                    dephasing_rate: float = 0.01) -> np.ndarray:
    """
    Perform time-dependent Schrödinger evolution with optional dephasing.

    If add_dephasing=True, applies Lindblad dephasing after each step:
    rho -> (1-p) * rho + p * sum_i Z_i rho Z_i

    This simulates an open quantum system (NISQ device noise).
    """
    psi = init_state.copy()
    n = ham_builder.n_qubits
    dim = 2**n

    # Precompute Z operators for dephasing
    if add_dephasing:
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        Z_ops = []
        for i in range(n):
            ops = [Z if j == i else I for j in range(n)]
            Z_i = ops[0]
            for op in ops[1:]:
                Z_i = np.kron(Z_i, op)
            Z_ops.append(Z_i)

    for step in range(schedule.n_steps):
        s = schedule.s(step)
        H = ham_builder.hamiltonian_at(s)
        U = expm(-1j * H * schedule.dt)
        psi = U @ psi

        if add_dephasing:
            # Apply dephasing channel
            rho = np.outer(psi, psi.conj())
            rho_new = (1 - dephasing_rate) * rho
            for Z_i in Z_ops:
                rho_new += (dephasing_rate / n) * Z_i @ rho @ Z_i
            # Recover state (approximate, take dominant eigenvector)
            evals, evecs = np.linalg.eigh(rho_new)
            psi = evecs[:, -1]  # largest eigenvalue

    return psi


def quantum_annealing_inference(problem_ham: RaslilaHamiltonian,
                                 init_state: np.ndarray,
                                 total_time: float = 10.0,
                                 n_steps: int = 100,
                                 schedule_type: str = "linear",
                                 driver_strength: float = 1.0,
                                 add_dephasing: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Complete quantum annealing inference.

    Returns (final_state, info_dict).
    """
    schedule = AnnealingSchedule(total_time, n_steps, schedule_type)
    td_ham = TimeDependentHamiltonian(problem_ham.n, problem_ham, driver_strength)

    final_state = annealed_evolve(td_ham, init_state, schedule, add_dephasing, 0.01)

    probs = np.abs(final_state)**2
    info = {
        "schedule": schedule_type,
        "total_time": total_time,
        "n_steps": n_steps,
        "driver_strength": driver_strength,
        "dephasing": add_dephasing,
        "final_probs": probs
    }
    return final_state, info


def variational_quantum_annealing(problem_ham: RaslilaHamiltonian,
                                   init_state: np.ndarray,
                                   schedule_params: np.ndarray,
                                   param_circuit) -> np.ndarray:
    """
    Variational quantum annealing (VQA) - parameterize the schedule itself.

    schedule_params controls s(t) shape. param_circuit maps params -> schedule.
    This allows learning the optimal annealing path.
    """
    # Placeholder for variational schedule optimization
    # Would integrate with PennyLane/TFQ for gradient-based schedule optimization
    pass


def trotterized_annealing(pauli_terms: List[Tuple[str, float]],
                          init_state: np.ndarray,
                          schedule: AnnealingSchedule,
                          n_qubits: int) -> np.ndarray:
    """
    Trotterized annealing for sparse Hamiltonians (scaling to >10 qubits).

    pauli_terms: list of (pauli_string, coefficient) from hamiltonian.pauli_string_list()
    Uses first-order Trotter-Suzuki: exp(-i(H1+H2)dt) ≈ exp(-iH1dt)exp(-iH2dt)
    """
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    pauli_map = {'X': X, 'Y': Y, 'Z': Z, 'I': I}

    def build_term_op(pauli_string: str) -> np.ndarray:
        ops = [pauli_map[c] for c in pauli_string]
        H = ops[0]
        for op in ops[1:]:
            H = np.kron(H, op)
        return H

    # Precompute all term operators
    term_ops = [(build_term_op(p), coeff) for p, coeff in pauli_terms]

    psi = init_state.copy()
    for step in range(schedule.n_steps):
        s = schedule.s(step)
        # For annealing, scale terms by s
        # H(s) = (1-s)*H_driver + s*H_problem
        # Here pauli_terms is H_problem; H_driver = -sum X_i
        # Apply each term sequentially (Trotter)
        U_driver = np.eye(2**n_qubits, dtype=complex)
        for i in range(n_qubits):
            ops = [X if j == i else I for j in range(n_qubits)]
            H_i = ops[0]
            for op in ops[1:]:
                H_i = np.kron(H_i, op)
            U_driver = expm(1j * (1-s) * H_i * schedule.dt) @ U_driver

        psi = U_driver @ psi

        for H_term, coeff in term_ops:
            U_term = expm(-1j * s * coeff * H_term * schedule.dt)
            psi = U_term @ psi

    return psi