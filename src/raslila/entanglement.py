"""Entanglement Entropy Monitoring - for analyzing quantum correlations.

Implements:
- Von Neumann entanglement entropy between subsystems
- Mutual information between universes
- Entanglement spectrum (Schmidt values)
- Monitoring during training/inference
- Visualization of entanglement structure
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.linalg import eigh


@dataclass
class EntanglementData:
    """Container for entanglement measurements."""
    von_neumann: Dict[str, float]
    renyi_2: Dict[str, float]
    mutual_info: Dict[Tuple[str, str], float]
    schmidt_values: Dict[str, np.ndarray]
    log_negativity: Dict[str, float]


def partial_trace(rho: np.ndarray, keep_qubits: List[int], n_qubits: int) -> np.ndarray:
    """
    Compute reduced density matrix by tracing out qubits.

    rho: full density matrix (2^n x 2^n)
    keep_qubits: list of qubit indices to keep (0 = MSB)
    n_qubits: total number of qubits
    """
    trace_qubits = [q for q in range(n_qubits) if q not in keep_qubits]
    if not trace_qubits:
        return rho

    # Reshape rho to tensor with 2n indices (bra and ket)
    # rho[i1,...,in, j1,...,jn] where ik, jk are qubit k basis
    # We want to trace over trace_qubits: sum over their basis
    tensor = rho.reshape([2] * (2 * n_qubits))

    # Indices: 0..n-1 = bra (row), n..2n-1 = ket (col)
    # Trace qubits: need to sum over both bra and ket indices
    trace_bra = trace_qubits
    trace_ket = [q + n_qubits for q in trace_qubits]
    all_trace = trace_bra + trace_ket

    # Sum over trace indices
    for axis in sorted(all_trace, reverse=True):
        tensor = np.sum(tensor, axis=axis)

    # Reshape back to matrix
    dim_keep = 2**len(keep_qubits)
    return tensor.reshape(dim_keep, dim_keep)


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute von Neumann entropy S = -Tr(rho log rho)."""
    evals = eigh(rho, eigvals_only=True)
    evals = evals[evals > 1e-12]
    return -np.sum(evals * np.log(evals))


def renyi_entropy(rho: np.ndarray, alpha: float = 2.0) -> float:
    """Compute Renyi entropy S_alpha = 1/(1-alpha) log Tr(rho^alpha)."""
    evals = eigh(rho, eigvals_only=True)
    evals = evals[evals > 1e-12]
    if alpha == 1.0:
        return von_neumann_entropy(rho)
    return np.log(np.sum(evals**alpha)) / (1.0 - alpha)


def mutual_information(rho_AB: np.ndarray,
                        qubits_A: List[int],
                        qubits_B: List[int],
                        n_qubits: int) -> float:
    """Compute mutual information I(A:B) = S(A) + S(B) - S(AB)."""
    rho_A = partial_trace(rho_AB, qubits_A, n_qubits)
    rho_B = partial_trace(rho_AB, qubits_B, n_qubits)
    S_A = von_neumann_entropy(rho_A)
    S_B = von_neumann_entropy(rho_B)
    S_AB = von_neumann_entropy(rho_AB)
    return max(0.0, S_A + S_B - S_AB)  # Ensure non-negative


def log_negativity(rho: np.ndarray,
                   qubits_A: List[int],
                   n_qubits: int) -> float:
    """Compute log negativity (entanglement measure for mixed states)."""
    rho_pt = partial_transpose(rho, qubits_A, n_qubits)
    evals = eigh(rho_pt, eigvals_only=True)
    neg_evals = evals[evals < 0]
    if len(neg_evals) == 0:
        return 0.0
    return np.log(1 + 2 * np.sum(np.abs(neg_evals)))


def partial_transpose(rho: np.ndarray, qubits_A: List[int], n_qubits: int) -> np.ndarray:
    """Partial transpose on subsystem A."""
    # Reshape to tensor: [2]*n (bra) + [2]*n (ket)
    tensor = rho.reshape([2] * (2 * n_qubits))

    # Transpose ket indices of qubits_A
    for q in qubits_A:
        ket_axis = q + n_qubits
        # Swap axis ket_axis with itself (transpose)
        # Actually we need to swap the two indices for each basis element
        # This is complex; use explicit matrix form
        pass

    # Fallback: explicit matrix partial transpose
    dim = 2**n_qubits
    rho_pt = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            # Check if bits on A match for transpose
            transpose = True
            for q in qubits_A:
                bit_i = (i >> (n_qubits - 1 - q)) & 1
                bit_j = (j >> (n_qubits - 1 - q)) & 1
                if bit_i != bit_j:
                    transpose = False
                    break

            if transpose:
                # Swap i and j on subsystem A
                i_new = i
                j_new = j
                for q in qubits_A:
                    bit_i = (i >> (n_qubits - 1 - q)) & 1
                    bit_j = (j >> (n_qubits - 1 - q)) & 1
                    i_new ^= (bit_i ^ bit_j) << (n_qubits - 1 - q)
                    j_new ^= (bit_i ^ bit_j) << (n_qubits - 1 - q)
                rho_pt[i_new, j_new] = rho[i, j]
            else:
                rho_pt[i, j] = rho[i, j]

    return rho_pt


def schmidt_decomposition(state: np.ndarray,
                          qubits_A: List[int],
                          n_qubits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Schmidt decomposition of pure state |psi> = sum_i lambda_i |a_i>|b_i>.

    Returns (schmidt_values, U_A, U_B).
    """
    dim_A = 2**len(qubits_A)
    dim_B = 2**(n_qubits - len(qubits_A))

    # Reshape state to matrix with A on rows, B on columns
    qubits_B = [q for q in range(n_qubits) if q not in qubits_A]

    # Create permutation: qubits_A first, then qubits_B
    perm = qubits_A + qubits_B

    # Apply permutation to state vector
    state_tensor = state.reshape([2] * n_qubits)
    state_tensor = np.transpose(state_tensor, perm)
    M = state_tensor.reshape(dim_A, dim_B)

    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    return S, U, Vh.T.conj()


def entanglement_spectrum(state: np.ndarray,
                          qubits_A: List[int],
                          n_qubits: int) -> np.ndarray:
    """Get Schmidt values squared (entanglement spectrum)."""
    S, _, _ = schmidt_decomposition(state, qubits_A, n_qubits)
    return S**2


def compute_all_entanglement(state: np.ndarray,
                              universe_qubits: Dict[str, List[int]],
                              n_qubits: int) -> EntanglementData:
    """
    Compute comprehensive entanglement data for all universe pairs.
    """
    rho = np.outer(state, state.conj())

    von_neumann = {}
    renyi_2 = {}
    schmidt_vals = {}

    for name, qubits in universe_qubits.items():
        rho_A = partial_trace(rho, qubits, n_qubits)
        von_neumann[name] = von_neumann_entropy(rho_A)
        renyi_2[name] = renyi_entropy(rho_A, alpha=2.0)
        schmidt_vals[name] = entanglement_spectrum(state, qubits, n_qubits)

    mutual_info = {}
    universe_names = list(universe_qubits.keys())
    for i, name_a in enumerate(universe_names):
        for name_b in universe_names[i+1:]:
            qubits_A = universe_qubits[name_a]
            qubits_B = universe_qubits[name_b]
            mi = mutual_information(rho, qubits_A, qubits_B, n_qubits)
            mutual_info[(name_a, name_b)] = mi

    log_neg = {}
    for name, qubits in universe_qubits.items():
        log_neg[name] = log_negativity(rho, qubits, n_qubits)

    return EntanglementData(
        von_neumann=von_neumann,
        renyi_2=renyi_2,
        mutual_info=mutual_info,
        schmidt_values=schmidt_vals,
        log_negativity=log_neg
    )


def monitor_entanglement_during_training(model, texts: List[str],
                                          n_universes: int = 3) -> List[EntanglementData]:
    """Monitor entanglement evolution during training on texts."""
    entanglement_history = []
    universe_qubits = {}
    for u in range(n_universes):
        universe_qubits[f'Universe_{u}'] = list(range(u, model.n_qubits, n_universes))

    for text in texts:
        result = model.forward(text)
        state = result['state']
        ed = compute_all_entanglement(state, universe_qubits, model.n_qubits)
        entanglement_history.append(ed)
    return entanglement_history


def print_entanglement_report(ed: EntanglementData):
    """Pretty print entanglement data."""
    print("=== Entanglement Report ===")
    print("\nVon Neumann Entropy (Universe vs Rest):")
    for name, val in sorted(ed.von_neumann.items()):
        print(f"  {name}: {val:.4f}")

    print("\nMutual Information (Universe Pairs):")
    for (a, b), mi in sorted(ed.mutual_info.items()):
        print(f"  {a} <-> {b}: {mi:.4f}")

    print("\nLog Negativity:")
    for name, ln in sorted(ed.log_negativity.items()):
        print(f"  {name}: {ln:.4f}")

    print("\nSchmidt Values (top 5 per universe):")
    for name, vals in sorted(ed.schmidt_values.items()):
        top5 = vals[:5]
        print(f"  {name}: {top5}")