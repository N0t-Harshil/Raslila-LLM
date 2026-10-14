"""Raslila Hamiltonian with all critical fixes.

Fixes applied:
  #1 (Tech Review #1) alpha/beta degeneracy:
     OLD: H += alpha*X0X2 + beta*X0X2  (same operator -> alpha+beta, singular QFIM)
     NEW: H += alpha*X0X2 + beta*Y0Y2  (non-commuting, independently learnable)
  #5 (Tech Review #5) unreachable target:
     OLD: only Z0Z4 + X0X2 terms; qubits 1,3,5 passive -> max prob 1/8.
     NEW: single-qubit Z biases h_z[i] + X transverse fields h_x[i] on ALL qubits.
          Every qubit is now active; full 6-bit target is reachable.
  #5b (qubit ordering): explicit endianness convention.
     qubit 0 = MSB (leftmost Kronecker factor).
     index_to_bits / bits_to_index document and enforce this.

Normalization (Frobenius) is kept to kill time aliasing, with a note that it
implicitly couples parameters (Additional Issue #7) — documented, not removed,
because fixed-time T=pi/2 evolution depends on it.
"""
import numpy as np
from scipy.linalg import norm
from collections import defaultdict

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)


def index_to_bits(idx, n):
    """Convert state index to bit list [b0..b_{n-1}], qubit0 = MSB.

    Example: index_to_bits(10, 6) -> [0,0,1,0,1,0] (binary 001010).
    """
    return [(idx >> (n - 1 - i)) & 1 for i in range(n)]


def bits_to_index(bits):
    n = len(bits)
    idx = 0
    for i, b in enumerate(bits):
        idx |= (int(b) & 1) << (n - 1 - i)
    return idx


def _kron_n(ops):
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def single_qubit_op(n, target, mat):
    ops = [mat if j == target else I for j in range(n)]
    return _kron_n(ops)


def two_qubit_op(n, a, b, mat_a, mat_b):
    ops = []
    for j in range(n):
        if j == a:
            ops.append(mat_a)
        elif j == b:
            ops.append(mat_b)
        else:
            ops.append(I)
    return _kron_n(ops)


class RaslilaHamiltonian:
    """Corrected normalized Hamiltonian.

    H_base = sum_i h_z[i] Z_i + sum_i h_x[i] X_i
           + sum_{i<j} J[i,j] Z_i Z_j
           + alpha * X_a X_b + beta * Y_a Y_b
    H_total = H_base / ||H_base||_F
    """

    def __init__(self, n_qubits, interference_qubits=(0, 2), seed=None):
        self.n = n_qubits
        self.a, self.b = interference_qubits
        rng = np.random.default_rng(seed)
        # Small random single-qubit terms so every qubit is active (Fix #5).
        self.h_z = rng.normal(0, 0.1, size=n_qubits)
        self.h_x = rng.normal(0, 0.1, size=n_qubits)
        self.J = defaultdict(float)
        self.alpha = 0.5
        self.beta = 0.5

    def copy(self):
        h = RaslilaHamiltonian(self.n, (self.a, self.b))
        h.h_z = self.h_z.copy()
        h.h_x = self.h_x.copy()
        h.J = self.J.copy()
        h.alpha = float(self.alpha)
        h.beta = float(self.beta)
        return h

    def construct_dense(self):
        n = self.n
        dim = 2 ** n
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n):
            if self.h_z[i] != 0:
                H += self.h_z[i] * single_qubit_op(n, i, Z)
            if self.h_x[i] != 0:
                H += self.h_x[i] * single_qubit_op(n, i, X)
        for (i, j), val in self.J.items():
            if val != 0:
                H += val * two_qubit_op(n, i, j, Z, Z)
        # FIX #1: non-commuting interference operators.
        if self.alpha != 0:
            H += self.alpha * two_qubit_op(n, self.a, self.b, X, X)
        if self.beta != 0:
            H += self.beta * two_qubit_op(n, self.a, self.b, Y, Y)
        frob = norm(H, "fro")
        if frob > 1e-12:
            H = H / frob
        # Return real part (H is Hermitian; imag parts ~0) as complex for expm.
        return H

    def frobenius_norm_unscaled(self):
        H = self.construct_dense()
        return float(norm(H, "fro"))

    def pauli_string_list(self):
        """Sparse description for future scaling (Trotterization)."""
        terms = []
        for i in range(self.n):
            terms.append((f"Z{i}", float(self.h_z[i])))
            terms.append((f"X{i}", float(self.h_x[i])))
        for (i, j), v in self.J.items():
            terms.append((f"Z{i}Z{j}", float(v)))
        terms.append((f"X{self.a}X{self.b}", float(self.alpha)))
        terms.append((f"Y{self.a}Y{self.b}", float(self.beta)))
        return [(k, v) for k, v in terms if v != 0]
