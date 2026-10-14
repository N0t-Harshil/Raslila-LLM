"""Raslila: 3D Brain-Inspired Neural Architecture (classical) + quantum research foundation."""
from .activation import squash_sin, dsquash_sin_dx, squash_sin_original, scaled_tanh
from .hamiltonian import RaslilaHamiltonian, index_to_bits, bits_to_index
from .mlp import TrainableMLP
from .qng import QNGOptimizer
from .memory import TopologicalMemory, inject_memory
from .memory_enhanced import (TopologicalMemory as EnhancedTopologicalMemory, TopologicalSnapshot,
                               extract_universe_phases, extract_activation_order)
from .evolution import (
    uniform_initial_state,
    evolve,
    state_probs,
    nll_loss,
    mse_loss,
    marginal_prob,
    compute_qfim,
)
from .annealing import (TimeDependentHamiltonian, AnnealingSchedule,
                         annealed_evolve, quantum_annealing_inference)
from .sparse_trotter import (SparseHamiltonian, PauliTerm,
                              build_raslila_sparse_ham, update_sparse_ham_j)
from .language import (Tokenizer, RaslilaEmbedding, QuantumReadout,
                        RaslilaLanguageModel, create_language_model,
                        train_on_corpus)
from .parameter_shift import (ParameterShiftGradients,
                               HamiltonianParameterShift,
                               validate_parameter_shift)
from .normalization import (AdaptiveNormalization, DecoupledHamiltonian,
                             hamiltonian_with_decoupled_normalization)
from .entanglement import (compute_all_entanglement,
                            monitor_entanglement_during_training,
                            EntanglementData)
from .model import RaslilaModel, DEFAULT_UNIVERSES, DEFAULT_TOKENS
from .model_enhanced import RaslilaModel as EnhancedRaslilaModel, ModelConfig
from .data import (BPETokenizer, StreamingDataset, DataCollator,
                    create_dataloaders, create_synthetic_data,
                    DataConfig, get_dataloaders_for_mve,
                    ClassicalBaselineConfig)
from .classical_baseline import (ClassicalTransformer, RaslilaClassical,
                                  train_classical_baseline, GPTConfig)
from .stochastic_parameter_shift import (StochasticParameterShift, StochasticShiftConfig,
                                           LayerwiseStochasticShift, GradientAccumulator,
                                           StochasticParameterShiftWithControlVariates,
                                           create_stochastic_optimizer)
from .train import (TrainingConfig, KillMonitor, CheckpointManager, WandBLogger,
                      PreTrainingGates, train_raslila, run_mve)

# Raslila (classical 3D-grid architecture, active model)
from .raslila import (Raslila, RaslilaConfig, EdgeNetwork, TransformerNode,
                      RMSNorm, CausalSelfAttention, SwiGLU,
                      create_raslila)

# PennyLane interface (optional - will be None if not installed)
try:
    from .pennylane_interface import (PennyLaneHamiltonian, QuantumDeviceConfig,
                                       HybridModel, create_hybrid_model,
                                       validate_pennylane_gradients,
                                       benchmark_quantum_forward)
    from .pennylane_interface import ParameterShiftGradients as PLParameterShiftGradients
except ImportError:
    PennyLaneHamiltonian = QuantumDeviceConfig = HybridModel = create_hybrid_model = None
    validate_pennylane_gradients = benchmark_quantum_forward = PLParameterShiftGradients = None

__all__ = [
    "squash_sin", "dsquash_sin_dx", "squash_sin_original", "scaled_tanh",
    "RaslilaHamiltonian", "index_to_bits", "bits_to_index",
    "TrainableMLP", "QNGOptimizer",
    "TopologicalMemory", "inject_memory",
    "EnhancedTopologicalMemory", "TopologicalSnapshot",
    "extract_universe_phases", "extract_activation_order",
    "uniform_initial_state", "evolve", "state_probs", "nll_loss",
    "mse_loss", "marginal_prob", "compute_qfim",
    "TimeDependentHamiltonian", "AnnealingSchedule",
    "annealed_evolve", "quantum_annealing_inference",
    "SparseHamiltonian", "PauliTerm",
    "build_raslila_sparse_ham", "update_sparse_ham_j",
    "Tokenizer", "RaslilaEmbedding", "QuantumReadout",
    "RaslilaLanguageModel", "create_language_model",
    "train_on_corpus",
    "ParameterShiftGradients", "HamiltonianParameterShift",
    "validate_parameter_shift",
    "AdaptiveNormalization", "DecoupledHamiltonian",
    "hamiltonian_with_decoupled_normalization",
    "compute_all_entanglement", "monitor_entanglement_during_training",
    "EntanglementData",
    "RaslilaModel", "EnhancedRaslilaModel", "ModelConfig",
    "DEFAULT_UNIVERSES", "DEFAULT_TOKENS",
    "BPETokenizer", "StreamingDataset", "DataCollator",
    "create_dataloaders", "create_synthetic_data",
    "DataConfig", "get_dataloaders_for_mve",
    "ClassicalBaselineConfig",
    "ClassicalTransformer", "RaslilaClassical",
    "train_classical_baseline", "GPTConfig",
    "PennyLaneHamiltonian", "QuantumDeviceConfig",
    "PLParameterShiftGradients", "HybridModel", "create_hybrid_model",
    "validate_pennylane_gradients", "benchmark_quantum_forward",
    "StochasticParameterShift", "StochasticShiftConfig",
    "LayerwiseStochasticShift", "GradientAccumulator",
    "StochasticParameterShiftWithControlVariates",
    "create_stochastic_optimizer",
    "TrainingConfig", "KillMonitor", "CheckpointManager", "WandBLogger",
    "PreTrainingGates", "train_raslila", "run_mve",
    "Raslila", "RaslilaConfig", "EdgeNetwork", "TransformerNode",
    "RMSNorm", "CausalSelfAttention", "SwiGLU",
    "create_raslila",
]
