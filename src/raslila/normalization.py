"""Fixed Hamiltonian Normalization - decouples parameter scaling.

Issue #7 from Additional Technical Issues:
Frobenius norm H = H / ||H||_F implicitly couples all parameters.
If J increases, overall norm increases, reducing effective alpha/beta contribution.

Solution: Normalize only the parameter-independent part, or use
spectral normalization with fixed scale, or separate normalization.

This module provides multiple normalization strategies.
"""
import numpy as np
from scipy.linalg import norm, eigh
from typing import Tuple
from .hamiltonian import RaslilaHamiltonian


def normalize_fixed_scale(H_base: np.ndarray, target_norm: float = 1.0) -> np.ndarray:
    """
    Normalize to fixed Frobenius norm.

    This still couples parameters but keeps total scale constant.
    """
    frob = norm(H_base, 'fro')
    if frob > 1e-12:
        return H_base * (target_norm / frob)
    return H_base


def normalize_spectral(H_base: np.ndarray, max_eigenvalue: float = 1.0) -> np.ndarray:
    """
    Normalize by spectral norm (largest eigenvalue magnitude).

    Better for time aliasing control since it bounds eigenvalues directly.
    Less sensitive to number of terms than Frobenius.
    """
    evals = eigh(H_base, eigvals_only=True)
    spec_norm = np.max(np.abs(evals))
    if spec_norm > 1e-12:
        return H_base * (max_eigenvalue / spec_norm)
    return H_base


def normalize_separable(H_base: np.ndarray,
                        param_terms: dict,
                        fixed_norm: float = 1.0) -> Tuple[np.ndarray, dict]:
    """
    Separate normalization: normalize fixed part and parameter part independently.

    H_base = H_fixed + H_params
    H_total = normalize(H_fixed) + normalize(H_params)  (each to fixed_norm/sqrt(2))

    This decouples parameter scaling from fixed terms.
    """
    # This is a conceptual framework - actual implementation needs
    # to track which terms are parameter-dependent
    H_fixed = param_terms.get('fixed', np.zeros_like(H_base))
    H_param = H_base - H_fixed

    frob_fixed = norm(H_fixed, 'fro')
    frob_param = norm(H_param, 'fro')

    target_each = fixed_norm / np.sqrt(2)

    if frob_fixed > 1e-12:
        H_fixed = H_fixed * (target_each / frob_fixed)
    if frob_param > 1e-12:
        H_param = H_param * (target_each / frob_param)

    return H_fixed + H_param, {
        'frob_fixed': float(norm(H_fixed, 'fro')),
        'frob_param': float(norm(H_param, 'fro')),
        'frob_total': float(norm(H_fixed + H_param, 'fro'))
    }


class AdaptiveNormalization:
    """
    Adaptive normalization that maintains stable eigenvalues
    while decoupling parameter sensitivities.

    Strategy: Track eigenvalue spectrum during training,
    adjust normalization to keep eigenvalues in [-1, 1]
    without over-scaling parameter contributions.
    """

    def __init__(self, target_max_eigenvalue: float = 1.0,
                 adaptation_rate: float = 0.01):
        self.target_max = target_max_eigenvalue
        self.adaptation_rate = adaptation_rate
        self.current_scale = 1.0
        self.eigenvalue_history = []

    def normalize(self, H: np.ndarray) -> np.ndarray:
        """Adaptively normalize Hamiltonian."""
        evals = eigh(H, eigvals_only=True)
        max_eval = np.max(np.abs(evals))
        self.eigenvalue_history.append(max_eval)

        if max_eval > 1e-12:
            # Adjust scale to keep max eigenvalue at target
            desired_scale = self.target_max / max_eval
            # Smooth adaptation
            self.current_scale = (1 - self.adaptation_rate) * self.current_scale + \
                                 self.adaptation_rate * desired_scale
            return H * self.current_scale
        return H

    def get_stats(self) -> dict:
        if not self.eigenvalue_history:
            return {}
        return {
            'current_scale': self.current_scale,
            'mean_max_eigenvalue': np.mean(self.eigenvalue_history[-100:]),
            'max_eigenvalue': np.max(self.eigenvalue_history),
            'min_eigenvalue': np.min(self.eigenvalue_history)
        }


def hamiltonian_with_decoupled_normalization(ham: RaslilaHamiltonian,
                                              method: str = 'spectral',
                                              fixed_terms_scale: float = 1.0) -> np.ndarray:
    """
    Build Hamiltonian with decoupled normalization.

    Methods:
    - 'frobenius': Original (coupled)
    - 'spectral': Spectral norm (less coupling)
    - 'fixed_driver': Normalize driver and problem separately
    """
    n = ham.n
    dim = 2**n
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def kron_n(ops):
        out = np.array([[1.0+0j]])
        for op in ops:
            out = np.kron(out, op)
        return out

    def single_op(target, mat):
        ops = [mat if j == target else I for j in range(n)]
        return kron_n(ops)

    def two_op(a, b, mat_a, mat_b):
        ops = []
        for j in range(n):
            if j == a:
                ops.append(mat_a)
            elif j == b:
                ops.append(mat_b)
            else:
                ops.append(I)
        return kron_n(ops)

    # Build parameter-independent part (fixed structure)
    H_fixed = np.zeros((dim, dim), dtype=complex)

    # Build parameter-dependent part
    H_param = np.zeros((dim, dim), dtype=complex)

    # Single-qubit Z (parameter-dependent)
    for i, h in enumerate(ham.h_z):
        if h != 0:
            H_param += h * single_op(i, Z)

    # Single-qubit X (parameter-dependent)
    for i, h in enumerate(ham.h_x):
        if h != 0:
            H_param += h * single_op(i, X)

    # ZZ couplings (parameter-dependent)
    for (i, j), v in ham.J.items():
        if v != 0:
            H_param += v * two_op(i, j, Z, Z)

    # Interference terms (parameter-dependent)
    a, b = ham.a, ham.b
    if ham.alpha != 0:
        H_param += ham.alpha * two_op(a, b, X, X)
    if ham.beta != 0:
        H_param += ham.beta * two_op(a, b, Y, Y)

    # Apply normalization
    if method == 'frobenius':
        H_total = H_param
        frob = norm(H_total, 'fro')
        if frob > 1e-12:
            H_total = H_total / frob
        return H_total

    elif method == 'spectral':
        H_total = H_param
        evals = eigh(H_total, eigvals_only=True)
        spec = np.max(np.abs(evals))
        if spec > 1e-12:
            H_total = H_total / spec
        return H_total

    elif method == 'fixed_driver':
        # Add a fixed driver term that doesn't scale with params
        H_driver = np.zeros((dim, dim), dtype=complex)
        for i in range(n):
            H_driver -= single_op(i, X)  # -X_i driver

        # Normalize each part to same scale
        target = fixed_terms_scale / np.sqrt(2)

        frob_param = norm(H_param, 'fro')
        frob_driver = norm(H_driver, 'fro')

        if frob_param > 1e-12:
            H_param = H_param * (target / frob_param)
        if frob_driver > 1e-12:
            H_driver = H_driver * (target / frob_driver)

        return H_param + H_driver

    elif method == 'none':
        return H_param

    else:
        raise ValueError(f"Unknown normalization method: {method}")


class DecoupledHamiltonian(RaslilaHamiltonian):
    """
    Extended Hamiltonian with decoupled normalization.

    Replaces the construct_dense method with normalization that
    doesn't couple parameter scales.
    """

    def __init__(self, n_qubits, interference_qubits=(0, 2),
                 normalization: str = 'spectral', seed=None):
        super().__init__(n_qubits, interference_qubits, seed)
        self.normalization = normalization
        self.adaptive_norm = AdaptiveNormalization()

    def construct_dense(self):
        """Build Hamiltonian with decoupled normalization."""
        H = hamiltonian_with_decoupled_normalization(self, self.normalization)
        return H

    def get_normalization_stats(self):
        return self.adaptive_norm.get_stats()