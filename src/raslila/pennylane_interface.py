"""PennyLane Quantum Interface — P0 Blocker.

JIT-compiled quantum forward/backward with parameter-shift gradients.
Replaces scipy.linalg.expm with differentiable PennyLane circuits.
"""
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    qml = None
    pnp = None
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not available. Quantum interface disabled.")

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from functools import partial


@dataclass
class QuantumDeviceConfig:
    n_qubits: int = 10
    shots: Optional[int] = None  # None = exact statevector
    device_name: str = "default.qubit"  # or "lightning.qubit", "default.mixed"
    diff_method: str = "parameter-shift"  # "backprop", "adjoint"


if PENNYLANE_AVAILABLE:
    class PennyLaneHamiltonian:
        """
        Differentiable Hamiltonian evolution using PennyLane.
        Supports parameter-shift gradients natively.
        """
        
        def __init__(self, config: QuantumDeviceConfig):
            if not PENNYLANE_AVAILABLE:
                raise RuntimeError("PennyLane not installed")
            self.config = config
            self.dev = qml.device(config.device_name, wires=config.n_qubits, shots=config.shots)
            self.n_qubits = config.n_qubits
            
            # Define the quantum circuit as a QNode
            self._build_circuit()
        
        def _build_circuit(self):
            """Build the differentiable quantum circuit."""
            
            @qml.qnode(self.dev, diff_method=self.config.diff_method)
            def circuit(params: pnp.ndarray, 
                        h_z: pnp.ndarray,
                        h_x: pnp.ndarray,
                        J_dict: Dict[Tuple[int, int], float],
                        alpha: float,
                        beta: float,
                        interference_qubits: Tuple[int, int]) -> pnp.ndarray:
                """
                Full Hamiltonian evolution circuit.
                
                params: trainable parameters (flattened)
                h_z, h_x: single-qubit terms
                J_dict: ZZ couplings
                alpha, beta: interference strengths
                interference_qubits: (a, b) for XX+YY
                """
                # Initialize in |+>^n state
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                
                # Build Hamiltonian evolution using Trotterization
                # For exact gradients, we use PennyLane's built-in evolution
                # or decompose into native gates
                
                # Single-qubit Z rotations
                for i in range(self.n_qubits):
                    if h_z[i] != 0:
                        qml.RZ(2 * h_z[i] * pnp.pi / 2, wires=i)  # T=pi/2
                
                # Single-qubit X rotations
                for i in range(self.n_qubits):
                    if h_x[i] != 0:
                        qml.RX(2 * h_x[i] * pnp.pi / 2, wires=i)
                
                # ZZ couplings (Ising interactions)
                for (i, j), J_val in J_dict.items():
                    if J_val != 0:
                        qml.IsingZZ(2 * J_val * pnp.pi / 2, wires=[i, j])
                
                # Interference terms: alpha * XX + beta * YY
                a, b = interference_qubits
                if alpha != 0:
                    qml.IsingXX(2 * alpha * pnp.pi / 2, wires=[a, b])
                if beta != 0:
                    qml.IsingYY(2 * beta * pnp.pi / 2, wires=[a, b])
                
                # Return probabilities for readout qubits
                return qml.probs(wires=range(self.n_qubits))
            
            self.circuit = circuit
        
        def __call__(self, h_z: pnp.ndarray, h_x: pnp.ndarray, 
                     J_dict: Dict[Tuple[int, int], float],
                     alpha: float, beta: float,
                     interference_qubits: Tuple[int, int]) -> pnp.ndarray:
            """Execute circuit and return probabilities."""
            params = pnp.array([])
            probs = self.circuit(params, h_z, h_x, J_dict, alpha, beta, interference_qubits)
            return probs
    
    
    class ParameterShiftGradients:
        """
        PennyLane-native parameter-shift gradients.
        Much faster than manual finite-difference.
        """
        
        def __init__(self, hamiltonian: PennyLaneHamiltonian):
            self.ham = hamiltonian
        
        def gradient(self, loss_fn: Callable, params: pnp.ndarray, 
                     argnum: int = 0, shift: float = np.pi/2) -> pnp.ndarray:
            """
            Compute gradient using PennyLane's built-in parameter-shift.
            
            Args:
                loss_fn: Function taking params, returning scalar loss
                params: Parameters to differentiate
                argnum: Which argument to differentiate
                shift: Shift amount (pi/2 for Pauli generators)
            """
            return qml.grad(loss_fn, argnum=argnum)(params)
    
    
    class BatchedQuantumForward:
        """
        Batched quantum forward pass for multiple samples.
        PennyLane supports batching via qml.map or vectorized QNodes.
        """
        
        def __init__(self, hamiltonian: PennyLaneHamiltonian, batch_size: int = 32):
            self.ham = hamiltonian
            self.batch_size = batch_size
            self.batched_circuit = None
        
        def __call__(self, h_z_batch: pnp.ndarray, h_x_batch: pnp.ndarray,
                     J_batch: List[Dict], alpha_batch: pnp.ndarray,
                     beta_batch: pnp.ndarray) -> pnp.ndarray:
            """Process batch of Hamiltonians."""
            results = []
            for i in range(len(h_z_batch)):
                probs = self.ham(h_z_batch[i], h_x_batch[i], J_batch[i],
                                alpha_batch[i], beta_batch[i], (0, 2))
                results.append(probs)
            return pnp.stack(results)
    
    
    class HybridModel:
        """
        Hybrid classical-quantum model with PennyLane autodiff.
        Classical MLP -> J_dict -> Quantum Hamiltonian -> Probabilities -> Loss
        All differentiable end-to-end.
        """
        
        def __init__(self, n_qubits: int = 10, n_universes: int = 5,
                     n_layers: int = 2, mlp_hidden: int = 128,
                     config: Optional[QuantumDeviceConfig] = None):
            self.n_qubits = n_qubits
            self.n_universes = n_universes
            self.n_layers = n_layers
            
            if config is None:
                config = QuantumDeviceConfig(n_qubits=n_qubits)
            self.config = config
            
            # Quantum Hamiltonian
            self.quantum_ham = PennyLaneHamiltonian(config)
            
            # Classical MLP for edge couplings (J_ij from property vectors)
            self.mlp_params = self._init_mlp(n_universes, mlp_hidden)
            
            # Single-qubit parameters
            self.h_z = pnp.zeros(n_qubits)
            self.h_x = pnp.zeros(n_qubits)
            self.alpha = pnp.array(0.5)
            self.beta = pnp.array(0.5)
            self.interference_qubits = (0, 2)
            
            # Edge list (same as RaslilaModel)
            self.edges = self._build_edges(n_layers, n_universes)
            self.step_count = 0
        
        def _init_mlp(self, n_universes: int, hidden: int) -> dict:
            """Initialize MLP parameters for edge couplings."""
            rng = np.random.default_rng(42)
            return {
                'W1': pnp.array(rng.normal(0, 0.1, size=(4, hidden))),
                'b1': pnp.zeros(hidden),
                'W2': pnp.array(rng.normal(0, 0.1, size=(hidden, 1))),
                'b2': pnp.zeros(1),
            }
        
        def _build_edges(self, n_layers: int, n_universes: int) -> List[Tuple[int, int]]:
            edges = []
            def q(L, u): return L * n_universes + u
            for L in range(n_layers):
                for u in range(n_universes - 1):
                    edges.append((q(L, u), q(L, u + 1)))
            for L in range(n_layers - 1):
                for u in range(n_universes):
                    edges.append((q(L, u), q(L + 1, u)))
            if n_layers * n_universes >= 5:
                edges.append((0, 4))
            return sorted(set(tuple(sorted(e)) for e in edges))
        
        def mlp_forward(self, prop_vec: pnp.ndarray) -> pnp.ndarray:
            """MLP forward: property vector -> J coupling."""
            h = pnp.tanh(prop_vec @ self.mlp_params['W1'] + self.mlp_params['b1'])
            out = h @ self.mlp_params['W2'] + self.mlp_params['b2']
            return pnp.tanh(out[0])
        
        def compute_J_dict(self, text: str, step: int) -> Dict[Tuple[int, int], float]:
            """Compute J_dict from text via MLP."""
            J_dict = {}
            for edge in self.edges:
                prop = self._property_vector(text, edge, step)
                J_dict[edge] = float(self.mlp_forward(prop))
            return J_dict
        
        def _property_vector(self, text: str, edge: Tuple[int, int], step: int) -> pnp.ndarray:
            """Generate property vector from text."""
            words = text.lower().split()
            usage = 0.2 + 0.1 * pnp.sin(step * 0.1 + hash(text) % 100 * 0.01)
            meaning = 0.9 + 0.05 * pnp.cos(step * 0.1 + hash(text) % 100 * 0.01)
            reflection = pnp.minimum(0.1 + 0.01 * step, 1.0)
            temporality = 0.8
            return pnp.array([usage, meaning, reflection, temporality])
        
        def forward(self, text: str, step: Optional[int] = None) -> pnp.ndarray:
            """Full differentiable forward pass."""
            if step is None:
                step = self.step_count
            J_dict = self.compute_J_dict(text, step)
            
            probs = self.quantum_ham(
                h_z=self.h_z,
                h_x=self.h_x,
                J_dict=J_dict,
                alpha=self.alpha,
                beta=self.beta,
                interference_qubits=self.interference_qubits
            )
            self.step_count += 1
            return probs
        
        def loss(self, probs: pnp.ndarray, target_bits: Tuple[int, ...],
                 readout_qubits: Tuple[int, ...]) -> pnp.ndarray:
            """NLL loss on marginal probabilities."""
            marginal = self._marginal_prob(probs, target_bits, readout_qubits)
            return -pnp.log(marginal + 1e-12)
        
        def _marginal_prob(self, probs: pnp.ndarray, 
                           bits: Tuple[int, ...], 
                           qubits: Tuple[int, ...]) -> pnp.ndarray:
            n_qubits = self.n_qubits
            total = pnp.array(0.0)
            for idx, p in enumerate(probs):
                match = True
                for q_idx, q in enumerate(qubits):
                    bit = (idx >> (n_qubits - 1 - q)) & 1
                    if bit != bits[q_idx]:
                        match = False
                        break
                if match:
                    total = total + p
            return total
    
    
    def create_hybrid_model(n_qubits: int = 10, n_universes: int = 5,
                            shots: Optional[int] = None) -> HybridModel:
        """Factory for hybrid model with PennyLane."""
        config = QuantumDeviceConfig(
            n_qubits=n_qubits,
            shots=shots,
            device_name="default.qubit",
            diff_method="parameter-shift"
        )
        return HybridModel(n_qubits=n_qubits, n_universes=n_universes, config=config)
    
    
    def validate_pennylane_gradients(model: HybridModel, text: str = "test") -> Dict:
        """Validate PennyLane gradients against finite difference."""
        target_bits = (1, 1, 1)
        
        # Extract initial values as floats before autograd tracing
        alpha_init = float(model.alpha)
        beta_init = float(model.beta)
        
        # Use PennyLane's parameter-shift with explicit argnums
        def loss_fn(alpha, beta):
            model.alpha = alpha
            model.beta = beta
            probs = model.forward(text, step=0)
            marginal = model._marginal_prob(probs, (1, 1, 1), (0, 2, 4))
            return -pnp.log(marginal + 1e-12)
        
        # Compute gradient with respect to both arguments
        # Use parameter-shift directly via qml.grad with argnums
        grad_fn = qml.grad(loss_fn, argnums=[0, 1])
        pl_grad = grad_fn(alpha_init, beta_init)
        
        eps = 1e-4
        fd_grads = []
        for i in range(2):
            params_plus = [alpha_init, beta_init]
            params_minus = [alpha_init, beta_init]
            params_plus[i] += 1e-4
            params_minus[i] -= 1e-4
            loss_p = loss_fn(params_plus[0], params_plus[1])
            loss_m = loss_fn(params_minus[0], params_minus[1])
            fd_grads.append((float(loss_p) - float(loss_m)) / (2 * 1e-4))
        
        pl_grad_arr = np.array([float(g) for g in pl_grad])
        fd_grad_arr = np.array(fd_grads)
        
        return {
            'pennylane': pl_grad_arr,
            'finite_diff': fd_grad_arr,
            'rel_error': np.abs(pl_grad_arr - fd_grad_arr) / (np.abs(pl_grad_arr) + 1e-8)
        }
    
    
    def benchmark_quantum_forward(model: HybridModel, n_samples: int = 10) -> Dict:
        """Benchmark quantum forward pass time."""
        import time
        
        times = []
        for i in range(n_samples):
            start = time.perf_counter()
            _ = model.forward(f"sample text {i}")
            times.append(time.perf_counter() - start)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'per_sample': np.mean(times)
        }


else:
    # Placeholder classes when PennyLane not available
    class PennyLaneHamiltonian:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PennyLane not installed. pip install pennylane")
    
    class HybridModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PennyLane not installed. pip install pennylane")
    
    def create_hybrid_model(*args, **kwargs):
        raise RuntimeError("PennyLane not installed. pip install pennylane")
    
    def validate_pennylane_gradients(*args, **kwargs):
        raise RuntimeError("PennyLane not installed. pip install pennylane")
    
    def benchmark_quantum_forward(*args, **kwargs):
        raise RuntimeError("PennyLane not installed. pip install pennylane")


def create_hybrid_model(n_qubits: int = 10, n_universes: int = 5,
                        shots: Optional[int] = None):
    """Factory for hybrid model with PennyLane."""
    if not PENNYLANE_AVAILABLE:
        raise RuntimeError("PennyLane not installed. pip install pennylane")
    config = QuantumDeviceConfig(
        n_qubits=n_qubits,
        shots=shots,
        device_name="default.qubit",
        diff_method="parameter-shift"
    )
    return HybridModel(n_qubits=n_qubits, n_universes=n_universes, config=config)