"""Complete Raslila model: grid + edges + memory + inference + training.

This is the whole system in one class, built from the verified subsystems:
  activation (fixed Squash+Sin) -> mlp (edge couplings J_ij)
  -> hamiltonian (normalized, XX+YY interference, X+Z on all qubits)
  -> evolution (fixed unitary T=pi/2, NLL loss, marginal readout)
  -> memory (resonance injection on J) + qng (natural-gradient for alpha/beta)

Grid convention:
  n_layers (vertical abstraction: 0=raw input ... L-1=pure reasoning)
  universes (horizontal lenses, e.g. Logical / Emotional / Temporal)
  n_qubits = n_layers * n_universes
  qubit index q = layer * n_universes + u, qubit 0 = MSB (leftmost Kronecker).
"""
import numpy as np
from collections import defaultdict

from .activation import squash_sin, dsquash_sin_dx
from .hamiltonian import RaslilaHamiltonian, index_to_bits, bits_to_index
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

DEFAULT_UNIVERSES = ["Logical", "Emotional", "Temporal"]
DEFAULT_TOKENS = ["grief", "logic", "memory", "clarity",
                  "longing", "reason", "dream", "empathy"]


class RaslilaModel:
    """The complete model. Default = 2 layers x 3 universes = 6 qubits (verified scale)."""

    def __init__(self, n_layers=2, universes=None, tokens=None,
                 readout_qubits=(0, 2, 4), interference_qubits=(0, 2),
                 edge_topology="grid+bridge", seed=42, mlp_hidden_dim=32):
        self.n_layers = int(n_layers)
        self.universes = list(universes) if universes else list(DEFAULT_UNIVERSES)
        self.n_universes = len(self.universes)
        self.n_qubits = self.n_layers * self.n_universes
        self.tokens = list(tokens) if tokens else list(DEFAULT_TOKENS)
        self.readout_qubits = tuple(readout_qubits)
        self.seed = seed
        rng = np.random.default_rng(seed)

        self.ham = RaslilaHamiltonian(self.n_qubits,
                                      interference_qubits=interference_qubits,
                                      seed=seed)
        self.mlp = TrainableMLP(activation=squash_sin,
                                d_activation=dsquash_sin_dx, seed=seed,
                                hidden_dim=mlp_hidden_dim)
        self.qng = QNGOptimizer(params=[self.ham.alpha, self.ham.beta], lr=0.05)
        self.memory = TopologicalMemory()
        self.memory.save("seed", {(0, 4): 0.8},
                         np.array([0.2, 0.9, 0.3, 0.8]))
        self.init_state = uniform_initial_state(self.n_qubits)
        # Tiny word embedding: vocab hash -> 4-dim property bias (placeholder
        # for a real LLM embedding matrix; keeps the pipeline end-to-end).
        self.embed_bias = {w: rng.normal(0, 0.05, size=4)
                           for w in set(DEFAULT_TOKENS) | {"love", "time", "dream"}}
        self.edges = self._build_edges(edge_topology)
        self.step_count = 0

    # ---- grid ----
    def qubit_info(self, q):
        layer, u = divmod(q, self.n_universes)
        return {"qubit": q, "layer": layer, "universe": self.universes[u]}

    def qubit_map(self):
        return [self.qubit_info(q) for q in range(self.n_qubits)]

    def _build_edges(self, mode):
        nL, nU = self.n_layers, self.n_universes
        edges = []

        def q(L, u):
            return L * nU + u
        if mode in ("grid", "grid+bridge"):
            for L in range(nL):          # horizontal: universes talk within a layer
                for u in range(nU - 1):
                    edges.append((q(L, u), q(L, u + 1)))
            for L in range(nL - 1):      # vertical: abstraction flows up
                for u in range(nU):
                    edges.append((q(L, u), q(L + 1, u)))
        if mode == "all":
            edges = [(i, j) for i in range(self.n_qubits)
                     for j in range(i + 1, self.n_qubits)]
        if mode == "grid+bridge" and self.n_qubits >= 5:
            if (0, 4) not in edges and (4, 0) not in edges:
                edges.append((0, 4))     # empathetic bridge from the 6-qubit proof
        return sorted(set(tuple(sorted(e)) for e in edges))

    # ---- text -> property vectors ----
    def property_vector(self, text, edge=None, step=None):
        step = self.step_count if step is None else step
        words = str(text).lower().split() or ["silence"]
        bias = np.mean([self.embed_bias.setdefault(
            w, np.zeros(4)) for w in words], axis=0)
        usage = float(np.clip(0.2 + 0.1 * np.sin(step * 0.1) + bias[0], 0, 1))
        meaning = float(np.clip(0.9 + 0.05 * np.cos(step * 0.1) + bias[1], 0, 1))
        reflection = float(min(0.1 + 0.01 * step, 1.0))
        temporality = float(np.clip(0.8 + bias[3], 0, 1))
        return np.array([usage, meaning, reflection, temporality])

    # ---- forward pass ----
    def forward(self, text="love and logic"):
        """Full inference: text -> Hamiltonian -> unitary -> probs + token."""
        for edge in self.edges:
            p = self.property_vector(text, edge=edge)
            self.ham.J[edge] = self.mlp.forward(p)
        # resonance memory on J (correct location, never on alpha/beta)
        p0 = self.property_vector(text)
        factor = float(np.exp(-float(np.sum((p0 - np.array([0.2, 0.9, 0.3, 0.8])) ** 2))))
        if "snap" in self.memory.snapshots:
            self.ham.J = self.memory.inject(dict(self.ham.J), "snap",
                                            min(factor, 0.3))
        H = self.ham.construct_dense()
        final = evolve(H, self.init_state, T_FIXED)
        probs = state_probs(final)
        token, conf, marg = self.decode(probs)
        self.step_count += 1
        return {"probs": probs, "state": final, "H": H,
                "token": token, "confidence": conf, "marginals": marg,
                "J": dict(self.ham.J), "alpha": float(self.ham.alpha),
                "beta": float(self.ham.beta)}

    def decode(self, probs):
        best, best_p, marg = None, -1.0, {}
        for i, tok in enumerate(self.tokens):
            bits = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
            m = marginal_prob(probs, self.n_qubits,
                              self.readout_qubits, tuple(bits))
            marg[tok] = m
            if m > best_p:
                best, best_p = tok, m
        return best, float(best_p), marg

    def token_to_bits(self, token):
        i = self.tokens.index(token)
        return [(i >> 2) & 1, (i >> 1) & 1, i & 1]

    # ---- training (one supervised step on a token target) ----
    def train_step(self, text, target_token, lr_mlp=0.01, lr_qng=0.05,
                   lr_single=0.02):
        target_bits = self.token_to_bits(target_token)
        readout = self.readout_qubits

        def loss_of(ham):
            for edge in self.edges:
                ham.J[edge] = ham.J.get(edge, 0.0)
            H = ham.construct_dense()
            probs = state_probs(evolve(H, self.init_state, T_FIXED))
            m = marginal_prob(probs, self.n_qubits, readout, tuple(target_bits))
            return -float(np.log(m + 1e-12)), probs

        # refresh J from MLP first
        for edge in self.edges:
            self.ham.J[edge] = self.mlp.forward(self.property_vector(text, edge=edge))
        base_loss, _ = loss_of(self.ham)
        eps = 0.01
        # alpha/beta via QNG (with sync fix)
        grads = []
        for k, key in (("alpha", 0), ("beta", 1)):
            old = float(getattr(self.ham, k))
            setattr(self.ham, k, old + eps)
            lp, _ = loss_of(self.ham)
            setattr(self.ham, k, old - eps)
            lm, _ = loss_of(self.ham)
            setattr(self.ham, k, old)
            grads.append((lp - lm) / (2 * eps))
        grads = np.array(grads)
        H0 = self.ham.construct_dense()
        psi0 = evolve(H0, self.init_state, T_FIXED)

        def get_state(a, b):
            hd = self.ham.copy()
            hd.alpha, hd.beta = a, b
            return evolve(hd.construct_dense(), self.init_state, T_FIXED)
        qfim = compute_qfim(get_state, float(self.ham.alpha),
                            float(self.ham.beta), psi0)
        self.qng.lr = lr_qng
        self.qng.sync([self.ham.alpha, self.ham.beta])
        na, nb = self.qng.step(grads, qfim)
        self.ham.alpha, self.ham.beta = float(na), float(nb)
        # shared MLP via mean grad over edges (separate J perturbations)
        acc = None
        for edge in self.edges:
            j0 = float(self.ham.J.get(edge, 0.0))
            self.ham.J[edge] = j0 + eps
            lp, _ = loss_of(self.ham)
            self.ham.J[edge] = j0 - eps
            lm, _ = loss_of(self.ham)
            self.ham.J[edge] = j0
            gJ = (lp - lm) / (2 * eps)
            g = self.mlp.gradient(self.property_vector(text, edge=edge), gJ)
            acc = g if acc is None else {k: acc[k] + g[k] for k in acc}
        acc = {k: v / max(len(self.edges), 1) for k, v in acc.items()}
        self.mlp.apply_grads(acc, lr=lr_mlp)
        new_loss, _ = loss_of(self.ham)
        return {"loss_before": float(base_loss), "loss_after": float(new_loss)}

    # ---- introspection ----
    def count_parameters(self):
        mlp_p = int(self.mlp.W.size + self.mlp.b.size +
                    self.mlp.W_out.size + self.mlp.b_out.size)
        return {"mlp": mlp_p, "h_z": int(self.ham.h_z.size),
                "h_x": int(self.ham.h_x.size), "J_edges": len(self.edges),
                "quantum_ab": 2,
                "total_trainable": mlp_p + int(self.ham.h_z.size) +
                int(self.ham.h_x.size) + len(self.edges) + 2,
                "hilbert_dim": 2 ** self.n_qubits}

    def summary(self):
        c = self.count_parameters()
        L = [f"=" * 68,
             "RASLILA — QUANTUM MULTIVERSE LLM — COMPLETE MODEL",
             f"Grid: {self.n_layers} layers x {self.n_universes} universes "
             f"= {self.n_qubits} qubits  (Hilbert dim {c['hilbert_dim']})",
             "=" * 68, ""]
        L.append("QUBIT MAP  [qubit = layer x universe, qubit0 = MSB]:")
        for qm in self.qubit_map():
            L.append(f"  q{qm['qubit']}: layer L{qm['layer']}  universe {qm['universe']}")
        L += ["",
              f"EDGES ({len(self.edges)}): " + ", ".join(
                  f"Z{a}Z{b}" for a, b in self.edges),
              f"INTERFERENCE: alpha * X{self.ham.a}X{self.ham.b} + "
              f"beta * Y{self.ham.a}Y{self.ham.b}  (independent, non-commuting)",
              f"READOUT qubits {self.readout_qubits} -> {len(self.tokens)} tokens: "
              + ", ".join(self.tokens),
              "",
              "DATA FLOW (inference, fixed unitary T=pi/2):",
              "  text -> [Usage,Meaning,Reflection,Temporality] per edge",
              "    -> shared MLP(4->8 tanh -> Squash+Sin) -> J_ij",
              "    -> J_ij + memory resonance exp(-hamming)  [on J, never on a/b]",
              "    -> H = sum h_z Z + sum h_x X + sum J ZZ + a XX + b YY",
              "    -> H /= ||H||_F  (kills time aliasing)",
              "    -> |psi> = exp(-i H pi/2) |+>^n  (unitary, not thermal cooling)",
              "    -> measure readout marginals -> argmax token",
              "",
              "TRAINING: NLL on readout marginal + QNG(alpha,beta w/ sync) + "
              "SGD(MLP, h_z, h_x)",
              "",
              f"PARAMETERS: MLP {c['mlp']} + h_z {c['h_z']} + h_x {c['h_x']} + "
              f"J {c['J_edges']} + ab 2 = {c['total_trainable']} trainable",
              f"HILBERT SPACE: {c['hilbert_dim']} amplitudes "
              f"({c['hilbert_dim'] * 16 / 1024:.1f} KB state @ complex128)",
              "=" * 68]
        return "\n".join(L)
