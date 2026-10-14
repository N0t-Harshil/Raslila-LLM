"""Unitary evolution, losses, marginal measurement, QFIM.

Clarifies Tech Review #6 ("inference is cooling" misrepresentation):
  The proof-of-concept uses FIXED UNITARY evolution
      |psi_final> = exp(-i H_total pi/2) |psi0>,
  NOT thermal annealing. Annealing would need a time-dependent Hamiltonian
  plus dissipation. `annealed_evolve` below is provided as an optional
  true-annealing sketch for future work; the verified results use unitary.
"""
import numpy as np
from scipy.linalg import expm
from .hamiltonian import index_to_bits

T_FIXED = np.pi / 2


def uniform_initial_state(n):
    dim = 2 ** n
    return np.ones(dim, dtype=complex) / np.sqrt(dim)


def evolve(H, init, T=T_FIXED):
    return expm(-1j * H * T) @ init


def state_probs(state):
    return np.abs(state) ** 2


def nll_loss(probs, target_idx, eps=1e-12):
    """Negative log-likelihood. FIX for Additional Issue #2 (MSE too weak)."""
    return float(-np.log(float(probs[target_idx]) + eps))


def mse_loss(probs, target_idx, target=1.0):
    return float((float(probs[target_idx]) - target) ** 2)


def marginal_prob(probs, n, qubits, bits):
    """Sum probs where given qubits equal given bits. qubit0 = MSB."""
    total = 0.0
    for idx in range(2 ** n):
        b = index_to_bits(idx, n)
        if all(b[q] == bit for q, bit in zip(qubits, bits)):
            total += float(probs[idx])
    return float(total)


def compute_qfim(get_state_fn, alpha, beta, psi, eps=1e-4):
    """2x2 Quantum Fisher Information Matrix via central finite differences."""
    da = (get_state_fn(alpha + eps, beta) - get_state_fn(alpha - eps, beta)) / (2 * eps)
    db = (get_state_fn(alpha, beta + eps) - get_state_fn(alpha, beta - eps)) / (2 * eps)

    def qf(a, b):
        return 4.0 * float(
            np.real(np.vdot(a, b) - np.vdot(a, psi) * np.vdot(psi, b))
        )

    return np.array([[qf(da, da), qf(da, db)], [qf(db, da), qf(db, db)]])


def annealed_evolve(h_builder, init, steps=20, T_total=np.pi / 2):
    """Optional TRUE-annealing sketch: linear schedule H(s)=(1-s)H0 + s H1.

    h_builder(s) must return the dense Hamiltonian at schedule point s in [0,1].
    Uses Trotterized stepwise unitary (still unitary; add Lindblad noise for
    real dissipation on hardware). Not used for the verified proof.
    """
    psi = init.copy()
    dt = T_total / steps
    for k in range(steps):
        s = (k + 0.5) / steps
        H = h_builder(s)
        psi = expm(-1j * H * dt) @ psi
    return psi
