"""Full corrected training loop.

Every fix from 10_Technical_Review + 13_Additional_Technical_Issues:
 1. alpha/beta independent (XX + YY) — in hamiltonian.py
 2. QNG sync before step — qng.sync(...)
 3. MLP grad correct shapes — in mlp.py
 4. grad_J via SEPARATE finite difference on J_AB (not reusing beta's loss_p)
 5. Reachable target — single-qubit X+Z on all qubits + NLL + track marginal
 6. Unitary (not cooling) — documented in evolution.py
 7. Fixed activation (pi/2 version, grad never zero) — in activation.py
 8. NLL loss instead of MSE
 9. Clipped reflection feature to [0,1]
 10. Small central-difference epsilons + documented (param-shift needs hardware gates)
 11. Explicit endianness (qubit0=MSB)
 12. Memory injection on J via resonance (NOT on alpha/beta)
"""
import numpy as np
from .hamiltonian import RaslilaHamiltonian, index_to_bits
from .mlp import TrainableMLP
from .qng import QNGOptimizer
from .memory import TopologicalMemory
from .evolution import (
    uniform_initial_state,
    evolve,
    state_probs,
    nll_loss,
    marginal_prob,
    compute_qfim,
    T_FIXED,
)

TARGET_INDEX = 10  # binary 001010 for 6 qubits, qubit0=MSB
ACTIVE_QUBITS = (0, 2, 4)
ACTIVE_BITS = tuple(index_to_bits(TARGET_INDEX, 6)[q] for q in ACTIVE_QUBITS)


def property_vector(step):
    """Clipped reflection (Fix: Additional Issue #3)."""
    usage = 0.2 + 0.1 * np.sin(step * 0.1)
    meaning = 0.9 + 0.05 * np.cos(step * 0.1)
    reflection = min(0.1 + 0.01 * step, 1.0)
    temporality = 0.8
    return np.array([usage, meaning, reflection, temporality])


def _loss_for_params(ham, init, target=TARGET_INDEX):
    H = ham.construct_dense()
    final = evolve(H, init, T_FIXED)
    probs = state_probs(final)
    return nll_loss(probs, target), final, probs


def train(n_qubits=6, steps=100, seed=42, lr_qng=0.05, lr_mlp=0.01,
          lr_single=0.02, log_every=20, verbose=True):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    mlp = TrainableMLP(seed=seed)
    ham = RaslilaHamiltonian(n_qubits, seed=seed)
    qng = QNGOptimizer(params=[ham.alpha, ham.beta], lr=lr_qng)
    init = uniform_initial_state(n_qubits)
    memory = TopologicalMemory()
    mem_pattern = np.array([0.2, 0.9, 0.3, 0.8])
    memory.save("seed", {(0, 4): 0.8}, mem_pattern)

    hist = {"loss": [], "prob_full": [], "prob_marg": [],
            "alpha": [], "beta": [], "J": []}
    eps_ab = 0.01   # central-diff epsilon for alpha/beta
    eps_j = 0.01    # separate epsilon for J (Fix #4)
    eps_h = 0.01    # epsilon for single-qubit fields

    for step in range(steps):
        p = property_vector(step)
        J_AB = mlp.forward(p)
        ham.J[(0, 4)] = J_AB

        # Correct memory injection on J (Fix: was on alpha/beta).
        factor = float(np.exp(-float(np.sum((p - mem_pattern) ** 2))))
        if step > 0 and step % 25 == 0:
            memory.save("snap", dict(ham.J), p)
        if "snap" in memory.snapshots:
            ham.J = memory.inject(dict(ham.J), "snap", min(factor, 0.3))
            J_AB = float(ham.J[(0, 4)])

        loss, final, probs = _loss_for_params(ham, init)
        p_full = float(probs[TARGET_INDEX])
        p_marg = marginal_prob(probs, n_qubits, ACTIVE_QUBITS, ACTIVE_BITS)

        # --- gradients for alpha, beta (central differences, small eps) ---
        base_a, base_b = float(ham.alpha), float(ham.beta)
        ham.alpha = base_a + eps_ab
        lp, _, _ = _loss_for_params(ham, init)
        ham.alpha = base_a - eps_ab
        lm, _, _ = _loss_for_params(ham, init)
        ham.alpha = base_a
        grad_a = (lp - lm) / (2 * eps_ab)

        ham.beta = base_b + eps_ab
        lp, _, _ = _loss_for_params(ham, init)
        ham.beta = base_b - eps_ab
        lm, _, _ = _loss_for_params(ham, init)
        ham.beta = base_b
        grad_b = (lp - lm) / (2 * eps_ab)
        grad_vec = np.array([grad_a, grad_b])

        # --- QFIM (now non-singular thanks to XX+YY fix) ---
        def get_state(a, b):
            hd = ham.copy()
            hd.alpha, hd.beta = a, b
            return evolve(hd.construct_dense(), init, T_FIXED)

        qfim = compute_qfim(get_state, base_a, base_b, final)

        # --- QNG update WITH sync (Fix #2) ---
        qng.sync([ham.alpha, ham.beta])
        new_params = qng.step(grad_vec, qfim)
        ham.alpha, ham.beta = float(new_params[0]), float(new_params[1])

        # --- grad_J via SEPARATE perturbation (Fix #4) ---
        j0 = float(ham.J[(0, 4)])
        ham.J[(0, 4)] = j0 + eps_j
        lp_j, _, _ = _loss_for_params(ham, init)
        ham.J[(0, 4)] = j0 - eps_j
        lm_j, _, _ = _loss_for_params(ham, init)
        ham.J[(0, 4)] = j0
        grad_J = (lp_j - lm_j) / (2 * eps_j)
        mlp.apply_grads(mlp.gradient(p, grad_J), lr=lr_mlp)

        # --- single-qubit fields via SGD (part of Fix #5: make target reachable) ---
        for arr_name in ("h_z", "h_x"):
            arr = getattr(ham, arr_name)
            grads = np.zeros_like(arr)
            for i in range(n_qubits):
                old = float(arr[i])
                arr[i] = old + eps_h
                lp_h, _, _ = _loss_for_params(ham, init)
                arr[i] = old - eps_h
                lm_h, _, _ = _loss_for_params(ham, init)
                arr[i] = old
                grads[i] = (lp_h - lm_h) / (2 * eps_h)
            arr -= lr_single * np.clip(grads, -1.0, 1.0)

        hist["loss"].append(float(loss))
        hist["prob_full"].append(p_full)
        hist["prob_marg"].append(p_marg)
        hist["alpha"].append(float(ham.alpha))
        hist["beta"].append(float(ham.beta))
        hist["J"].append(float(ham.J[(0, 4)]))
        if verbose and (step % log_every == 0 or step == steps - 1):
            print(f"Step {step:3d} | NLL {loss:.4f} | p_full {p_full:.4f} "
                  f"| p_marg {p_marg:.4f} | a {ham.alpha:+.3f} b {ham.beta:+.3f} "
                  f"| J {ham.J[(0,4)]:+.4f}")
    return {"mlp": mlp, "ham": ham, "qng": qng, "init": init,
            "history": hist, "memory": memory}
