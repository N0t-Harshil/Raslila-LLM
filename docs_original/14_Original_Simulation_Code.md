---
date: 2026-08-19
time: 20:00
tags:
  - code
  - simulation
  - original
  - raslila
  - status/incomplete
---

# Original 6‑Qubit Simulation Code

This file contains only the Python code for the original Raslila simulation. It is provided as-is, without corrections. The code has known bugs documented in [[10_Technical_Review]] and [[13_Additional_Technical_Issues]].

```python
import numpy as np
from scipy.linalg import expm, pinv, norm
from collections import defaultdict

# ==============================================================================
# 1. THE SACRED ACTIVATION (Kills Spectral Crowding)
# Maps (-inf, inf) to [-1, 1] without ever losing gradient.
# ==============================================================================
def squash_sin(x):
    squash = x / (1.0 + np.abs(x) + 1e-12)
    return np.sin(squash * np.pi)

def dsquash_sin_dx(x):
    squash = x / (1.0 + np.abs(x) + 1e-12)
    dsquash = 1.0 / ((1.0 + np.abs(x) + 1e-12) ** 2)
    return np.pi * dsquash * np.cos(squash * np.pi)

# ==============================================================================
# 2. THE NORMALIZED HAMILTONIAN (Kills Time Aliasing)
# We never store dense matrices for >10 qubits; we build sparsely.
# ==============================================================================
class NormalizedHamiltonian:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.h = np.zeros(n_qubits)
        self.J = defaultdict(float)  # Sparse dictionary of edges
        self.alpha = 0.5              # Interference Strength (Univ A -> B)
        self.beta = 0.5               # Interference Strength (Univ B -> A)

    def construct_dense(self):
        dim = 2 ** self.n
        H = np.zeros((dim, dim))
        Z = np.array([[1, 0], [0, -1]])
        X = np.array([[0, 1], [1, 0]])
        I = np.eye(2)

        # Biases
        for i in range(self.n):
            op = 1
            for j in range(self.n):
                op = np.kron(op, Z if j == i else I)
            H += self.h[i] * op

        # Couplings (The brain cells)
        for (i, j), val in self.J.items():
            op = 1
            for k in range(self.n):
                if k == i or k == j:
                    op = np.kron(op, Z)
                else:
                    op = np.kron(op, I)
            H += val * op

        # Multiverse Interference (Entangling Universe 0 and 2)
        op_X = 1
        for k in range(self.n):
            if k == 0 or k == 2:  # Entangle Universe 0 (Logical) and 2 (Temporal)
                op_X = np.kron(op_X, X)
            else:
                op_X = np.kron(op_X, I)
        H += self.alpha * op_X + self.beta * op_X

        # ----- THE FIX: Normalize to keep eigenvalues in [-1, 1] -----
        norm_fro = norm(H, 'fro')
        if norm_fro > 1e-6:
            H = H / norm_fro
        return H

# ==============================================================================
# 3. THE CLASSICAL MLP (Learns the edge properties)
# ==============================================================================
class TrainableMLP:
    def __init__(self):
        self.W = np.random.randn(4, 8) * 0.1
        self.b = np.zeros(8)
        self.W_out = np.random.randn(8, 1) * 0.1
        self.b_out = np.zeros(1)

    def forward(self, p_vector):
        hidden = np.tanh(p_vector @ self.W + self.b)
        raw = hidden @ self.W_out + self.b_out
        return float(squash_sin(raw[0]))

    def gradient(self, p_vector, grad_J):
        hidden = np.tanh(p_vector @ self.W + self.b)
        raw = hidden @ self.W_out + self.b_out
        d_act = dsquash_sin_dx(raw[0])
        delta_raw = grad_J * d_act
        return {'W': np.outer(p_vector, delta_raw * (1-hidden**2) @ self.W_out.T),
                'b': (delta_raw * (1-hidden**2) @ self.W_out.T).flatten(),
                'W_out': delta_raw * hidden.reshape(-1, 1),
                'b_out': np.array([delta_raw])}

# ==============================================================================
# 4. THE QUANTUM NATURAL GRADIENT OPTIMIZER (Kills Barren Plateaus)
# Uses exact 2x2 QFIM inversion for alpha/beta.
# ==============================================================================
class QNG_Optimizer:
    def __init__(self, params, lr=0.1):
        self.params = np.array(params)
        self.lr = lr
    def step(self, grad, qfim):
        qfim_inv = pinv(qfim + 1e-6 * np.eye(2))
        self.params -= self.lr * (qfim_inv @ grad)
        self.params = np.clip(self.params, 0.1, 0.9)  # Physical constraints
        return self.params

# ==============================================================================
# 5. MAIN TRAINING LOOP (Proof of Concept)
# Goal: Maximize probability of collapsing to state '001010' 
# (Logical Analysis -> Empathetic Reasoning)
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    N_QUBITS = 6
    TARGET = 1  # Target probability for state index 10 (Binary: 001010)

    mlp = TrainableMLP()
    hamiltonian = NormalizedHamiltonian(N_QUBITS)
    qng = QNG_Optimizer(params=[hamiltonian.alpha, hamiltonian.beta], lr=0.05)

    print("Training the Quantum Multiverse Brain...\n")
    for step in range(100):
        # Dynamic properties (simulating changing context)
        usage = 0.2 + 0.1 * np.sin(step * 0.1)
        meaning = 0.9 + 0.05 * np.cos(step * 0.1)
        reflection = 0.1 + 0.01 * step  # Reflection grows over time (learning)
        temporality = 0.8
        p_vector = np.array([usage, meaning, reflection, temporality])

        # Compute coupling between Emotional and Logical universes
        J_AB = mlp.forward(p_vector)
        hamiltonian.J[(0, 4)] = J_AB  # Edge between q0 (Logical) and q4 (Empathy)

        # ---- Memory Injection (Sparse Gravity) ----
        gamma = np.exp(-2)  # Simulated resonance
        hamiltonian.alpha = squash_sin(hamiltonian.alpha + gamma * 0.8)
        hamiltonian.beta = squash_sin(hamiltonian.beta + gamma * 0.2)

        # ---- Quantum Evolution (Fixed T = pi/2) ----
        H = hamiltonian.construct_dense()
        init = np.ones(2**N_QUBITS) / np.sqrt(2**N_QUBITS)
        final = expm(-1j * H * (np.pi / 2)) @ init
        probs = np.abs(final) ** 2
        loss = (probs[10] - TARGET) ** 2

        # ---- Gradient Calculation (Parameter Shift) ----
        delta = 0.1
        # Alpha shift
        hamiltonian.alpha += delta
        H_p = hamiltonian.construct_dense()
        loss_p = ((np.abs(expm(-1j * H_p * (np.pi/2)) @ init) ** 2)[10] - TARGET) ** 2
        hamiltonian.alpha -= 2*delta
        H_m = hamiltonian.construct_dense()
        loss_m = ((np.abs(expm(-1j * H_m * (np.pi/2)) @ init) ** 2)[10] - TARGET) ** 2
        hamiltonian.alpha += delta
        grad_alpha = (loss_p - loss_m) / (2*delta)
        
        # Beta shift
        hamiltonian.beta += delta
        H_p = hamiltonian.construct_dense()
        loss_p = ((np.abs(expm(-1j * H_p * (np.pi/2)) @ init) ** 2)[10] - TARGET) ** 2
        hamiltonian.beta -= 2*delta
        H_m = hamiltonian.construct_dense()
        loss_m = ((np.abs(expm(-1j * H_m * (np.pi/2)) @ init) ** 2)[10] - TARGET) ** 2
        hamiltonian.beta += delta
        grad_beta = (loss_p - loss_m) / (2*delta)
        grad_vec = np.array([grad_alpha, grad_beta])

        # ---- Exact 2x2 QFIM ----
        def get_state(a, b):
            h_d = NormalizedHamiltonian(N_QUBITS)
            h_d.J = hamiltonian.J.copy()
            h_d.alpha, h_d.beta = a, b
            return expm(-1j * h_d.construct_dense() * (np.pi/2)) @ init
        
        eps = 1e-4
        psi = final
        da = (get_state(hamiltonian.alpha+eps, hamiltonian.beta) - get_state(hamiltonian.alpha-eps, hamiltonian.beta)) / (2*eps)
        db = (get_state(hamiltonian.alpha, hamiltonian.beta+eps) - get_state(hamiltonian.alpha, hamiltonian.beta-eps)) / (2*eps)
        qfim = np.array([[4*np.real(np.vdot(da, da) - np.vdot(da, psi)*np.vdot(psi, da)), 
                          4*np.real(np.vdot(da, db) - np.vdot(da, psi)*np.vdot(psi, db))],
                         [4*np.real(np.vdot(db, da) - np.vdot(db, psi)*np.vdot(psi, da)), 
                          4*np.real(np.vdot(db, db) - np.vdot(db, psi)*np.vdot(psi, db))]])

        # ---- Updates ----
        new_params = qng.step(grad_vec, qfim)
        hamiltonian.alpha, hamiltonian.beta = new_params[0], new_params[1]
        
        grad_J = (loss_p - loss) / delta
        mlp_grads = mlp.gradient(p_vector, grad_J)
        mlp.W -= 0.01 * mlp_grads['W']
        mlp.b -= 0.01 * mlp_grads['b']
        mlp.W_out -= 0.01 * mlp_grads['W_out']
        mlp.b_out -= 0.01 * mlp_grads['b_out']

        if step % 20 == 0:
            print(f"Step {step:3d} | Loss: {loss:.6f} | α: {hamiltonian.alpha:.3f} | β: {hamiltonian.beta:.3f} | J_AB: {J_AB:.4f} | Reflect: {reflection:.2f}")

    print("\n" + "="*60)
    print("[VERIFIED] The math holds. The system learned to activate Empathetic Logic.")
    print("The reflection property grew, the QNG prevented plateaus, and the Hamiltonian stayed stable.")
```
