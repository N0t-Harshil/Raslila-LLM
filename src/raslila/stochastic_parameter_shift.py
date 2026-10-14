"""Stochastic Parameter-Shift Gradients — P0 Blocker.

Full parameter-shift requires 2 circuit evaluations per parameter.
For 2M params: 4M evaluations/step = IMPOSSIBLE.

Solution: Stochastic parameter-shift — sample subset of parameters per step.
"""
import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    qml = None
    pnp = None
    PENNYLANE_AVAILABLE = False

from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import time


@dataclass
class StochasticShiftConfig:
    """Configuration for stochastic parameter-shift."""
    n_params_per_step: int = 100  # Number of parameters to estimate per step
    shift: float = np.pi / 2      # Shift amount (pi/2 for Pauli generators)
    use_control_variates: bool = True  # Reduce variance
    momentum: float = 0.9         # Momentum for gradient estimate
    minibatch_size: int = 32      # Samples per gradient estimate


class StochasticParameterShift:
    """
    Stochastic parameter-shift gradient estimator.
    
    Instead of computing gradient for all parameters (2n evaluations),
    sample a subset and use momentum to accumulate estimates.
    """
    
    def __init__(self, config: StochasticShiftConfig):
        self.config = config
        self.momentum_buffer = {}
        self.step_count = 0
    
    def estimate_gradient(self, 
                          loss_fn: Callable[[Dict[str, float]], float],
                          params: Dict[str, float],
                          param_names: List[str],
                          key: str = "default") -> Dict[str, float]:
        """
        Estimate gradient for subset of parameters.
        
        Args:
            loss_fn: Function taking param dict, returning scalar loss
            params: Current parameter values
            param_names: List of parameter names to consider
            key: Identifier for momentum buffer
            
        Returns:
            Gradient estimates for all params (sampled ones updated, others from momentum)
        """
        n_total = len(param_names)
        n_sample = min(self.config.n_params_per_step, n_total)
        
        # Sample parameters
        rng = np.random.default_rng(self.step_count)
        sampled_idx = rng.choice(n_total, size=n_sample, replace=False)
        sampled_names = [param_names[i] for i in sampled_idx]
        
        gradients = {}
        
        for name in sampled_names:
            old_val = params[name]
            shift = self.config.shift
            
            # Forward shift
            params[name] = old_val + shift
            loss_p = loss_fn(params)
            
            # Backward shift
            params[name] = old_val - shift
            loss_m = loss_fn(params)
            
            # Restore
            params[name] = old_val
            
            # Central difference
            grad = (loss_p - loss_m) / (2 * shift)
            gradients[name] = grad
        
        # Update momentum buffer
        if key not in self.momentum_buffer:
            self.momentum_buffer[key] = {name: 0.0 for name in param_names}
        
        # Update momentum for sampled params
        for name in sampled_names:
            self.momentum_buffer[key][name] = (
                self.config.momentum * self.momentum_buffer[key][name] +
                (1 - self.config.momentum) * gradients[name]
            )
        
        # Return momentum estimates for all params
        result = {name: self.momentum_buffer[key][name] for name in param_names}
        
        self.step_count += 1
        return result


class StochasticParameterShiftWithControlVariates:
    """
    Stochastic parameter-shift with control variates for variance reduction.
    
    Uses a cheap classical proxy (e.g., linear approximation) as control variate.
    """
    
    def __init__(self, config: StochasticShiftConfig, proxy_fn: Callable):
        self.config = config
        self.proxy_fn = proxy_fn  # Cheap classical approximation
        self.step_count = 0
        self.control_coeff = 0.0
    
    def estimate_gradient(self,
                          loss_fn: Callable,
                          params: Dict[str, float],
                          param_names: List[str],
                          key: str = "default") -> Dict[str, float]:
        n_total = len(param_names)
        n_sample = min(self.config.n_params_per_step, n_total)
        
        rng = np.random.default_rng(self.step_count)
        sampled_idx = rng.choice(n_total, size=n_sample, replace=False)
        sampled_names = [param_names[i] for i in sampled_idx]
        
        gradients = {}
        proxy_grads = {}
        
        # Compute proxy gradients (analytical, cheap)
        for name in param_names:
            proxy_grads[name] = self.proxy_fn(name, params)
        
        for name in sampled_names:
            old_val = params[name]
            shift = self.config.shift
            
            params[name] = old_val + shift
            loss_p = loss_fn(params)
            
            params[name] = old_val - shift
            loss_m = loss_fn(params)
            
            params[name] = old_val
            
            quantum_grad = (loss_p - loss_m) / (2 * shift)
            
            # Control variate correction
            # grad_corrected = quantum_grad - c * (proxy_grad - E[proxy_grad])
            # Simplified: use proxy as baseline
            corrected_grad = quantum_grad - self.control_coeff * proxy_grads[name]
            gradients[name] = corrected_grad
        
        # Update control coefficient (online)
        if len(gradients) > 1:
            quantum_vals = list(gradients.values())
            proxy_vals = [proxy_grads[n] for n in sampled_names]
            # Optimal c = Cov(quantum, proxy) / Var(proxy)
            if np.var(proxy_vals) > 1e-8:
                self.control_coeff = np.cov(quantum_vals, proxy_vals)[0,1] / np.var(proxy_vals)
        
        return gradients


class GradientAccumulator:
    """
    Accumulate stochastic gradients over multiple steps
    to simulate larger effective batch size.
    """
    
    def __init__(self, param_names: List[str], accumulation_steps: int = 4):
        self.param_names = param_names
        self.accumulation_steps = accumulation_steps
        self.buffer = {name: 0.0 for name in param_names}
        self.step_in_accum = 0
    
    def accumulate(self, gradients: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Accumulate gradients. Returns averaged gradients when ready."""
        for name, grad in gradients.items():
            if name in self.buffer:
                self.buffer[name] += grad
        
        self.step_in_accum += 1
        
        if self.step_in_accum >= self.accumulation_steps:
            # Return averaged gradients
            result = {name: val / self.accumulation_steps 
                     for name, val in self.buffer.items()}
            # Reset buffer
            self.buffer = {name: 0.0 for name in self.param_names}
            self.step_in_accum = 0
            return result
        return None


class LayerwiseStochasticShift:
    """
    Layer-wise stochastic parameter-shift.
    Different sampling rates for different parameter types:
    - Quantum params (alpha, beta, h_z, h_x): always compute (few)
    - Classical MLP params: sample heavily
    - Embedding params: sample moderately
    """
    
    def __init__(self, config: StochasticShiftConfig):
        self.config = config
        self.quantum_sampler = StochasticParameterShift(
            StochasticShiftConfig(n_params_per_step=10, shift=np.pi/2)
        )
        self.classical_sampler = StochasticParameterShift(
            StochasticShiftConfig(n_params_per_step=config.n_params_per_step - 10, 
                                 shift=1e-4)  # Small shift for classical finite-diff
        )
        self.embedding_sampler = StochasticParameterShift(
            StochasticShiftConfig(n_params_per_step=50, shift=1e-4)
        )
    
    def estimate_all_gradients(self,
                               loss_fn: Callable,
                               params: Dict[str, float],
                               quantum_param_names: List[str],
                               classical_param_names: List[str],
                               embedding_param_names: List[str],
                               step: int) -> Dict[str, float]:
        """Estimate gradients for all parameter groups with different strategies."""
        
        all_grads = {}
        
        # Quantum params: exact parameter-shift (few params)
        if quantum_param_names:
            q_grads = self.quantum_sampler.estimate_gradient(
                loss_fn, params, quantum_param_names, key="quantum"
            )
            all_grads.update(q_grads)
        
        # Classical params: stochastic finite-diff
        if classical_param_names:
            c_grads = self.classical_sampler.estimate_gradient(
                loss_fn, params, classical_param_names, key="classical"
            )
            all_grads.update(c_grads)
        
        # Embedding params: moderate sampling
        if embedding_param_names:
            e_grads = self.embedding_sampler.estimate_gradient(
                loss_fn, params, embedding_param_names, key="embedding"
            )
            all_grads.update(e_grads)
        
        return all_grads


def create_stochastic_optimizer(model_params: Dict[str, np.ndarray],
                                 quantum_params: List[str],
                                 classical_params: List[str],
                                 embedding_params: List[str],
                                 config: StochasticShiftConfig) -> Tuple:
    """
    Create stochastic parameter-shift optimizer setup.
    
    Returns:
        - loss_fn wrapper
        - gradient estimator
        - parameter update function
    """
    
    # Split parameters by type
    param_dict = {}
    for name, arr in model_params.items():
        param_dict[name] = float(arr.flatten()[0]) if arr.size == 1 else arr
    
    layerwise = LayerwiseStochasticShift(config)
    
    def loss_fn_wrapper(params_dict):
        # This would be called by the estimator
        # Actual implementation needs to map params_dict back to model
        pass
    
    def estimate_gradients(loss_fn, step):
        return layerwise.estimate_all_gradients(
            loss_fn, param_dict, quantum_params, classical_params, embedding_params, step
        )
    
    def update_params(gradients, lr=3e-4):
        for name, grad in gradients.items():
            if name in model_params:
                if isinstance(model_params[name], np.ndarray):
                    if model_params[name].size == 1:
                        model_params[name][...] -= lr * grad
                    else:
                        # Distribute gradient across array (simplified)
                        model_params[name] -= lr * grad / model_params[name].size
    
    return loss_fn_wrapper, estimate_gradients, update_params


def benchmark_stochastic_shift(n_params: int = 10000,
                                n_sample: int = 100,
                                n_steps: int = 10) -> Dict:
    """Benchmark stochastic parameter-shift vs full."""
    import time
    
    # Simulate loss function
    params = {f'p{i}': np.random.randn() for i in range(n_params)}
    param_names = list(params.keys())
    
    def dummy_loss(p):
        return sum(v**2 for v in p.values())
    
    config = StochasticShiftConfig(n_params_per_step=n_sample)
    estimator = StochasticParameterShift(config)
    
    # Time stochastic
    start = time.perf_counter()
    for step in range(n_steps):
        _ = estimator.estimate_gradient(dummy_loss, params, param_names)
    stochastic_time = time.perf_counter() - start
    
    # Time full (small n for feasibility)
    if n_params <= 100:
        start = time.perf_counter()
        for step in range(n_steps):
            grads = {}
            for name in param_names:
                old = params[name]
                shift = np.pi/2
                params[name] = old + shift
                lp = dummy_loss(params)
                params[name] = old - shift
                lm = dummy_loss(params)
                params[name] = old
                grads[name] = (lp - lm) / (2 * shift)
        full_time = time.perf_counter() - start
    else:
        full_time = float('inf')
    
    return {
        'stochastic_time': stochastic_time,
        'full_time': full_time,
        'speedup': full_time / stochastic_time if full_time < float('inf') else 'inf',
        'params_per_second': n_params * n_steps / stochastic_time
    }