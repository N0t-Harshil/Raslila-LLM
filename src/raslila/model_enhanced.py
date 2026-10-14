"""Complete Raslila Model - Enhanced with all architecture pieces.

Integrates:
- Topological memory (enhanced with proper snapshots)
- True quantum annealing (time-dependent Hamiltonian)
- Sparse Trotterized evolution (scaling >10 qubits)
- Full language interface (tokenization, embedding, generation)
- Parameter-shift exact gradients
- Decoupled Hamiltonian normalization
- Entanglement entropy monitoring
"""
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

from .activation import squash_sin, dsquash_sin_dx
from .hamiltonian import RaslilaHamiltonian, index_to_bits, bits_to_index
from .mlp import TrainableMLP
from .qng import QNGOptimizer
from .memory import TopologicalMemory  # original simple memory
from .memory_enhanced import (TopologicalMemory as EnhancedTopologicalMemory,
                               TopologicalSnapshot,
                               extract_universe_phases,
                               extract_activation_order)
from .evolution import (
    uniform_initial_state,
    evolve,
    state_probs,
    nll_loss,
    marginal_prob,
    compute_qfim,
    T_FIXED,
)
from .annealing import (TimeDependentHamiltonian, AnnealingSchedule,
                         annealed_evolve, quantum_annealing_inference,
                         trotterized_annealing)
from .sparse_trotter import (SparseHamiltonian, PauliTerm,
                              build_raslila_sparse_ham, update_sparse_ham_j)
from .language import (Tokenizer, RaslilaEmbedding, QuantumReadout,
                        RaslilaLanguageModel, create_language_model)
from .parameter_shift import (ParameterShiftGradients,
                               HamiltonianParameterShift,
                               validate_parameter_shift)
from .normalization import (AdaptiveNormalization, DecoupledHamiltonian,
                             hamiltonian_with_decoupled_normalization)
from .entanglement import (compute_all_entanglement,
                            monitor_entanglement_during_training,
                            EntanglementData)

DEFAULT_UNIVERSES = ["Logical", "Emotional", "Temporal", "Spatial"]
DEFAULT_TOKENS = ["grief", "logic", "memory", "clarity",
                  "longing", "reason", "dream", "empathy"]


@dataclass
class ModelConfig:
    """Complete configuration for Raslila model."""
    n_layers: int = 2
    universes: List[str] = None
    tokens: List[str] = None
    readout_qubits: Tuple[int, ...] = (0, 2, 4)
    interference_qubits: Tuple[int, int] = (0, 2)
    edge_topology: str = "grid+bridge"
    seed: int = 42
    # Enhanced features
    use_enhanced_memory: bool = True
    use_annealing: bool = False
    use_sparse_trotter: bool = False
    normalization_method: str = 'spectral'  # 'frobenius', 'spectral', 'fixed_driver', 'none'
    use_parameter_shift: bool = True
    # MLP hidden dimension (controls model capacity)
    # 6H + 23 = total_params → for 568: H ≈ 91
    mlp_hidden_dim: int = 91


class RaslilaModel:
    """
    The complete Raslila Quantum Multiverse LLM.

    Grid: n_layers (vertical abstraction) x n_universes (horizontal lenses)
    = n_qubits total

    Architecture flow:
    Text -> Tokenizer -> Embedding -> Property vectors per edge
         -> Shared MLP -> J_ij couplings
         -> Enhanced Memory (topological snapshots, resonance)
         -> Hamiltonian (normalized, XX+YY interference, X+Z on all qubits)
         -> Evolution (unitary T=pi/2 OR quantum annealing)
         -> Quantum State -> Readout Marginals -> Token
    """

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        if config is None:
            config = ModelConfig(**kwargs)
        self.config = config

        self.n_layers = config.n_layers
        self.universes = config.universes or DEFAULT_UNIVERSES
        self.n_universes = len(self.universes)
        self.n_qubits = self.n_layers * self.n_universes
        self.tokens = config.tokens or DEFAULT_TOKENS
        self.readout_qubits = config.readout_qubits
        self.seed = config.seed
        rng = np.random.default_rng(config.seed)

        # Core Hamiltonian
        if config.normalization_method != 'frobenius':
            self.ham = DecoupledHamiltonian(
                self.n_qubits,
                interference_qubits=config.interference_qubits,
                normalization=config.normalization_method,
                seed=config.seed
            )
        else:
            self.ham = RaslilaHamiltonian(
                self.n_qubits,
                interference_qubits=config.interference_qubits,
                seed=config.seed
            )

        self.mlp = TrainableMLP(activation=squash_sin,
                                d_activation=dsquash_sin_dx, seed=config.seed,
                                hidden_dim=config.mlp_hidden_dim)
        self.qng = QNGOptimizer(params=[self.ham.alpha, self.ham.beta], lr=0.05)

        # Memory: enhanced or simple
        if config.use_enhanced_memory:
            self.memory = EnhancedTopologicalMemory(max_snapshots=1000)
        else:
            self.memory = TopologicalMemory()
            self.memory.save("seed", {(0, 4): 0.8},
                             np.array([0.2, 0.9, 0.3, 0.8]))

        self.init_state = uniform_initial_state(self.n_qubits)

        # Token embedding (for language interface)
        self.embed_bias = {w: rng.normal(0, 0.05, size=4)
                           for w in set(DEFAULT_TOKENS) | {"love", "time", "dream"}}

        # Graph structure
        self.edges = self._build_edges(config.edge_topology)

        # Sparse Hamiltonian for scaling
        if config.use_sparse_trotter:
            self.sparse_ham = build_raslila_sparse_ham(
                self.n_layers, self.n_universes, config.interference_qubits)
        else:
            self.sparse_ham = None

        # Parameter-shift gradients
        self.psg = None
        if config.use_parameter_shift:
            self.psg = HamiltonianParameterShift(
                self.ham, self.init_state,
                target_bits=self._default_target_bits(),
                readout_qubits=self.readout_qubits
            )

        # Entanglement tracking
        self.entanglement_history: List[EntanglementData] = []

        # Training state
        self.step_count = 0
        self.training_history: List[Dict] = []

    def _default_target_bits(self) -> Tuple[int, ...]:
        """Default target bits for readout qubits (token 'empathy' = 111)."""
        return (1, 1, 1)

    # ---- Grid utilities ----
    def qubit_info(self, q: int) -> Dict:
        layer, u = divmod(q, self.n_universes)
        return {"qubit": q, "layer": layer, "universe": self.universes[u]}

    def qubit_map(self) -> List[Dict]:
        return [self.qubit_info(q) for q in range(self.n_qubits)]

    def _build_edges(self, mode: str) -> List[Tuple[int, int]]:
        nL, nU = self.n_layers, self.n_universes
        edges = []

        def q(L, u): return L * nU + u
        if mode in ("grid", "grid+bridge"):
            for L in range(nL):
                for u in range(nU - 1):
                    edges.append((q(L, u), q(L, u + 1)))
            for L in range(nL - 1):
                for u in range(nU):
                    edges.append((q(L, u), q(L + 1, u)))
        if mode == "all":
            edges = [(i, j) for i in range(self.n_qubits)
                     for j in range(i + 1, self.n_qubits)]
        if mode == "grid+bridge" and self.n_qubits >= 5:
            if (0, 4) not in edges and (4, 0) not in edges:
                edges.append((0, 4))
        return sorted(set(tuple(sorted(e)) for e in edges))

    # ---- Property vectors ----
    def property_vector(self, text: str, edge=None, step=None) -> np.ndarray:
        step = self.step_count if step is None else step
        words = str(text).lower().split() or ["silence"]
        bias = np.mean([self.embed_bias.setdefault(w, np.zeros(4)) for w in words], axis=0)
        usage = float(np.clip(0.2 + 0.1 * np.sin(step * 0.1) + bias[0], 0, 1))
        meaning = float(np.clip(0.9 + 0.05 * np.cos(step * 0.1) + bias[1], 0, 1))
        reflection = float(min(0.1 + 0.01 * step, 1.0))
        temporality = float(np.clip(0.8 + bias[3], 0, 1))
        return np.array([usage, meaning, reflection, temporality])

    # ---- Forward pass (inference) ----
    def forward(self, text: str = "love and logic",
                use_annealing: Optional[bool] = None,
                return_entanglement: bool = False) -> Dict:
        """
        Full inference pass.

        Args:
            text: Input text
            use_annealing: Override config (True=quantum annealing, False=unitary)
            return_entanglement: Compute and return entanglement data
        """
        # Compute J couplings from MLP
        for edge in self.edges:
            p = self.property_vector(text, edge=edge)
            self.ham.J[edge] = self.mlp.forward(p)

        # Enhanced memory injection on J
        if isinstance(self.memory, EnhancedTopologicalMemory):
            p0 = self.property_vector(text)
            phases = extract_universe_phases(
                self.init_state, self.n_universes, self.n_layers)
            act_order = extract_activation_order(self.ham.J)
            # Save snapshot periodically
            if self.step_count > 0 and self.step_count % 25 == 0:
                self.memory.save_snapshot(
                    dict(self.ham.J), phases, act_order, p0,
                    label=f"step_{self.step_count}")
            # Inject resonant memories
            self.ham.J = self.memory.inject_memory(
                dict(self.ham.J), phases, p0, max_injections=3)
        else:
            # Original simple memory
            p0 = self.property_vector(text)
            factor = float(np.exp(-float(np.sum((p0 - np.array([0.2, 0.9, 0.3, 0.8]))**2))))
            if "snap" in self.memory.snapshots:
                self.ham.J = self.memory.inject(dict(self.ham.J), "snap", min(factor, 0.3))

        # Evolution: unitary or annealing
        use_ann = self.config.use_annealing if use_annealing is None else use_annealing
        if use_ann:
            final_state, anneal_info = quantum_annealing_inference(
                self.ham, self.init_state, total_time=10.0, n_steps=50)
            probs = np.abs(final_state)**2
            info = {"evolution": "annealing", **anneal_info}
        else:
            H = self.ham.construct_dense()
            final_state = evolve(H, self.init_state, T_FIXED)
            probs = state_probs(final_state)
            info = {"evolution": "unitary"}

        # Decode token
        token, conf, marg = self.decode(probs)

        # Entanglement monitoring
        entanglement = None
        if return_entanglement:
            universe_qubits = {}
            for u in range(self.n_universes):
                universe_qubits[self.universes[u]] = list(range(u, self.n_qubits, self.n_universes))
            entanglement = compute_all_entanglement(final_state, universe_qubits, self.n_qubits)
            self.entanglement_history.append(entanglement)

        self.step_count += 1

        result = {
            "probs": probs, "state": final_state, "H": self.ham.construct_dense(),
            "token": token, "confidence": conf, "marginals": marg,
            "J": dict(self.ham.J), "alpha": float(self.ham.alpha),
            "beta": float(self.ham.beta), "info": info
        }
        if entanglement:
            result["entanglement"] = entanglement
        return result

    # ---- Decoding ----
    def decode(self, probs: np.ndarray) -> Tuple[str, float, Dict]:
        best, best_p, marg = None, -1.0, {}
        for i, tok in enumerate(self.tokens):
            bits = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
            m = marginal_prob(probs, self.n_qubits, self.readout_qubits, tuple(bits))
            marg[tok] = m
            if m > best_p:
                best, best_p = tok, m
        return best, float(best_p), marg

    def token_to_bits(self, token: str) -> List[int]:
        i = self.tokens.index(token)
        return [(i >> 2) & 1, (i >> 1) & 1, i & 1]

    # ---- Training ----
    def train_step(self, text: str, target_token: str,
                   lr_mlp: float = 0.01, lr_qng: float = 0.05,
                   lr_single: float = 0.02,
                   use_parameter_shift: Optional[bool] = None) -> Dict:
        """Single supervised training step with exact or finite-diff gradients."""
        target_bits = tuple(self.token_to_bits(target_token))
        readout = self.readout_qubits

        def loss_of(ham):
            for edge in self.edges:
                ham.J[edge] = ham.J.get(edge, 0.0)
            H = ham.construct_dense()
            probs = state_probs(evolve(H, self.init_state, T_FIXED))
            m = marginal_prob(probs, self.n_qubits, readout, target_bits)
            return -float(np.log(m + 1e-12)), probs

        # Refresh J from MLP
        for edge in self.edges:
            self.ham.J[edge] = self.mlp.forward(self.property_vector(text, edge=edge))
        base_loss, _ = loss_of(self.ham)

        # Gradients for alpha, beta
        use_ps = self.config.use_parameter_shift if use_parameter_shift is None else use_parameter_shift
        if use_ps and self.psg:
            # Exact parameter-shift gradients via all_gradients()
            param_grads = self.psg.all_gradients()
            grads = np.array([param_grads['alpha'], param_grads['beta']])
        else:
            # Finite difference fallback
            eps = 0.01
            grads = []
            for k in ("alpha", "beta"):
                old = float(getattr(self.ham, k))
                setattr(self.ham, k, old + eps)
                lp, _ = loss_of(self.ham)
                setattr(self.ham, k, old - eps)
                lm, _ = loss_of(self.ham)
                setattr(self.ham, k, old)
                grads.append((lp - lm) / (2 * eps))
            grads = np.array(grads)

        # QFIM and QNG update
        H0 = self.ham.construct_dense()
        psi0 = evolve(H0, self.init_state, T_FIXED)

        def get_state(a, b):
            hd = self.ham.copy()
            hd.alpha, hd.beta = a, b
            return evolve(hd.construct_dense(), self.init_state, T_FIXED)

        qfim = compute_qfim(get_state, float(self.ham.alpha), float(self.ham.beta), psi0)
        self.qng.lr = lr_qng
        self.qng.sync([self.ham.alpha, self.ham.beta])
        na, nb = self.qng.step(grads, qfim)
        self.ham.alpha, self.ham.beta = float(na), float(nb)

        # MLP gradients (separate J perturbations)
        eps = 0.01
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

        # Single-qubit fields via parameter-shift or finite diff
        if use_ps and self.psg:
            param_grads = self.psg.all_gradients()
            for i in range(self.n_qubits):
                self.ham.h_z[i] -= lr_single * np.clip(param_grads.get(f'h_z[{i}]', 0), -1, 1)
                self.ham.h_x[i] -= lr_single * np.clip(param_grads.get(f'h_x[{i}]', 0), -1, 1)
        else:
            eps = 0.01
            for arr_name in ("h_z", "h_x"):
                arr = getattr(self.ham, arr_name)
                for i in range(self.n_qubits):
                    old = float(arr[i])
                    arr[i] = old + eps
                    lp, _ = loss_of(self.ham)
                    arr[i] = old - eps
                    lm, _ = loss_of(self.ham)
                    arr[i] = old
                    arr[i] -= lr_single * np.clip((lp - lm) / (2 * eps), -1, 1)

        new_loss, _ = loss_of(self.ham)
        self.training_history.append({
            "step": self.step_count, "loss": new_loss,
            "alpha": float(self.ham.alpha), "beta": float(self.ham.beta)
        })

        return {"loss_before": float(base_loss), "loss_after": float(new_loss)}

    # ---- Sparse Trotter Evolution (for scaling) ----
    def forward_sparse(self, text: str, total_time: float = 5.0,
                       n_steps: int = 20, trotter_order: int = 2) -> Dict:
        """Forward pass using sparse Trotterized evolution (for >10 qubits)."""
        if self.sparse_ham is None:
            raise ValueError("Model not initialized with use_sparse_trotter=True")

        # Update sparse Hamiltonian J from MLP
        J_dict = {}
        for edge in self.edges:
            p = self.property_vector(text, edge=edge)
            j_val = self.mlp.forward(p)
            self.ham.J[edge] = j_val
            J_dict[edge] = j_val
        update_sparse_ham_j(self.sparse_ham, J_dict)

        # Sparse Trotter evolution
        final_state = self.sparse_ham.evolve(self.init_state, total_time, n_steps, trotter_order)
        probs = np.abs(final_state)**2
        token, conf, marg = self.decode(probs)

        return {"probs": probs, "state": final_state, "token": token,
                "confidence": conf, "marginals": marg, "J": dict(self.ham.J)}

    # ---- Language Model Interface ----
    def create_language_interface(self, vocab_size: int = 256,
                                   texts: Optional[List[str]] = None) -> RaslilaLanguageModel:
        """Create full language model interface."""
        return create_language_model(
            n_layers=self.n_layers,
            n_universes=self.n_universes,
            vocab_size=vocab_size,
            texts=texts
        )

    # ---- Entanglement Analysis ----
    def get_entanglement_report(self, text: str = "love and logic") -> EntanglementData:
        """Get entanglement data for current state."""
        result = self.forward(text, return_entanglement=True)
        return result.get("entanglement")

    def analyze_entanglement_dynamics(self, texts: List[str]) -> List[EntanglementData]:
        """Track entanglement evolution across texts."""
        return monitor_entanglement_during_training(self, texts, self.n_universes)

    # ---- Parameter Validation ----
    def validate_gradients(self) -> Dict:
        """Validate parameter-shift vs finite-difference gradients."""
        if self.psg is None:
            return {"error": "Parameter-shift not enabled"}
        return validate_parameter_shift(
            self.ham, self.init_state,
            self._default_target_bits(), self.readout_qubits
        )

    # ---- Introspection ----
    def count_parameters(self) -> Dict:
        mlp_p = int(self.mlp.W.size + self.mlp.b.size +
                    self.mlp.W_out.size + self.mlp.b_out.size)
        return {
            "mlp": mlp_p,
            "h_z": int(self.ham.h_z.size),
            "h_x": int(self.ham.h_x.size),
            "J_edges": len(self.edges),
            "quantum_ab": 2,
            "total_trainable": mlp_p + int(self.ham.h_z.size) +
            int(self.ham.h_x.size) + len(self.edges) + 2,
            "hilbert_dim": 2 ** self.n_qubits,
            "memory_snapshots": len(self.memory.snapshots) if hasattr(self.memory, 'snapshots') else 0
        }

    def summary(self) -> str:
        c = self.count_parameters()
        L = [f"=" * 72,
             "RASLILA — QUANTUM MULTIVERSE LLM — ENHANCED COMPLETE MODEL",
             f"Grid: {self.n_layers} layers x {self.n_universes} universes "
             f"= {self.n_qubits} qubits  (Hilbert dim {c['hilbert_dim']})",
             f"Normalization: {self.config.normalization_method}  |  "
             f"Annealing: {self.config.use_annealing}  |  "
             f"Sparse Trotter: {self.config.use_sparse_trotter}  |  "
             f"Parameter-Shift: {self.config.use_parameter_shift}",
             "=" * 72, ""]
        L.append("QUBIT MAP [qubit = layer x universe, qubit0 = MSB]:")
        for qm in self.qubit_map():
            L.append(f"  q{qm['qubit']}: layer L{qm['layer']}  universe {qm['universe']}")
        L += ["",
              f"EDGES ({len(self.edges)}): " + ", ".join(f"Z{a}Z{b}" for a, b in self.edges),
              f"INTERFERENCE: alpha * X{self.ham.a}X{self.ham.b} + "
              f"beta * Y{self.ham.a}Y{self.ham.b}  (independent, non-commuting)",
              f"READOUT qubits {self.readout_qubits} -> {len(self.tokens)} tokens: "
              + ", ".join(self.tokens),
              "",
              "DATA FLOW:",
              "  TEXT -> Tokenizer -> Embedding -> [U,M,R,T] per edge",
              "    -> shared MLP(4->8 tanh -> Squash+Sin) -> J_ij",
              "    -> Enhanced Memory (topological snapshots, resonance)",
              "    -> Hamiltonian (Z, X, ZZ, XX, YY) with decoupled normalization",
              "    -> EVOLUTION: Unitary T=pi/2  OR  Quantum Annealing",
              "    -> Measure readout marginals -> Token",
              "",
              "TRAINING: NLL on readout marginal",
              "  QNG(alpha,beta w/ sync + parameter-shift exact gradients)",
              "  SGD(MLP, h_z, h_x w/ parameter-shift)",
              "",
              f"PARAMETERS: MLP {c['mlp']} + h_z {c['h_z']} + h_x {c['h_x']} + "
              f"J {c['J_edges']} + ab 2 = {c['total_trainable']} trainable",
              f"HILBERT SPACE: {c['hilbert_dim']} amplitudes "
              f"({c['hilbert_dim'] * 16 / 1024:.1f} KB state @ complex128)",
              f"MEMORY: {c['memory_snapshots']} topological snapshots",
              "=" * 72]
        return "\n".join(L)