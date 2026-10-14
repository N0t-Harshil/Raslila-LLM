"""Complete Topological Memory System - Phase 3 realization.

Implements:
- Topological Snapshots: store shape of active edges (J values, phases, temporal order)
- Retrieval as Resonance: resonance factor exp(-Hamming) injects memory into J
- Memory as Dark Matter: gravitational pull on Hamiltonian dynamics
- Self-modification: learnable memory injection strengths
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from .activation import squash_sin


@dataclass
class TopologicalSnapshot:
    """Stores the topological shape of a significant cognitive event."""
    # Core topology: which edges were active and their coupling strengths
    J_values: Dict[Tuple[int, int], float]  # edge -> coupling
    # Phase relationships between universes at snapshot time
    universe_phases: Dict[int, float]  # universe_idx -> phase
    # Temporal signature: activation order of edges
    activation_order: List[Tuple[int, int]]  # sequence of (i,j) edges
    # Context vector that triggered this memory
    context_vector: np.ndarray  # [usage, meaning, reflection, temporality]
    # Semantic label (optional)
    label: str = ""
    # Strength of this memory (learned)
    strength: float = 1.0
    # When this snapshot was created
    timestamp: int = 0


class TopologicalMemory:
    """
    Complete topological memory system.

    Memories are not stored as vectors but as topological invariants - the shape
    of the quantum state's edge activations. Retrieval happens through resonance:
    when current activation pattern matches a stored topology, the memory injects
    its gravitational influence into the Hamiltonian.
    """

    def __init__(self, max_snapshots: int = 1000, resonance_threshold: float = 0.1):
        self.snapshots: List[TopologicalSnapshot] = []
        self.max_snapshots = max_snapshots
        self.resonance_threshold = resonance_threshold
        # Learnable memory injection strength per snapshot (meta-cognitive)
        self.injection_strengths: Dict[str, float] = {}
        self.step_count = 0

    def save_snapshot(self, J_current: Dict[Tuple[int, int], float],
                      universe_phases: Dict[int, float],
                      activation_order: List[Tuple[int, int]],
                      context_vector: np.ndarray,
                      label: str = "") -> str:
        """Save a topological snapshot of the current cognitive state."""
        snapshot = TopologicalSnapshot(
            J_values=dict(J_current),
            universe_phases=dict(universe_phases),
            activation_order=list(activation_order),
            context_vector=np.array(context_vector, dtype=float),
            label=label,
            strength=1.0,
            timestamp=self.step_count
        )
        self.snapshots.append(snapshot)
        self.injection_strengths[id(snapshot)] = 1.0

        # Maintain capacity
        if len(self.snapshots) > self.max_snapshots:
            self._forget_oldest()

        self.step_count += 1
        return f"snap_{len(self.snapshots)-1}"

    def _forget_oldest(self):
        """Remove weakest/oldest memories (simulating memory decay)."""
        # Sort by strength * recency, remove weakest
        scores = [(s.strength * (self.step_count - s.timestamp + 1), i)
                  for i, s in enumerate(self.snapshots)]
        scores.sort()
        remove_idx = scores[0][1]
        removed = self.snapshots.pop(remove_idx)
        self.injection_strengths.pop(id(removed), None)

    def compute_resonance(self, current_J: Dict[Tuple[int, int], float],
                          current_phases: Dict[int, float],
                          current_context: np.ndarray) -> List[Tuple[float, TopologicalSnapshot]]:
        """
        Compute resonance between current state and all stored snapshots.

        Returns list of (resonance_factor, snapshot) sorted by resonance descending.
        Resonance combines:
        - Edge pattern similarity (Hamming-like on J values)
        - Phase alignment between universes
        - Context vector similarity
        """
        resonances = []
        for snap in self.snapshots:
            # Edge pattern resonance
            all_edges = set(current_J.keys()) | set(snap.J_values.keys())
            if not all_edges:
                edge_resonance = 1.0
            else:
                diff = sum(abs(current_J.get(e, 0) - snap.J_values.get(e, 0))
                           for e in all_edges)
                max_possible = len(all_edges) * 2.0  # max diff per edge is 2
                edge_resonance = np.exp(-diff / max(1.0, max_possible))

            # Phase resonance
            phase_resonance = 1.0
            if snap.universe_phases and current_phases:
                phase_diffs = [abs(current_phases.get(u, 0) - snap.universe_phases.get(u, 0))
                               for u in set(current_phases) | set(snap.universe_phases)]
                if phase_diffs:
                    avg_phase_diff = np.mean(phase_diffs)
                    phase_resonance = np.exp(-avg_phase_diff / np.pi)

            # Context resonance
            ctx_diff = np.linalg.norm(current_context - snap.context_vector)
            context_resonance = np.exp(-ctx_diff / 2.0)

            # Combined resonance (geometric mean)
            total_resonance = (edge_resonance * phase_resonance * context_resonance) ** (1/3)
            total_resonance *= snap.strength * self.injection_strengths.get(id(snap), 1.0)

            if total_resonance > self.resonance_threshold:
                resonances.append((total_resonance, snap))

        resonances.sort(key=lambda x: x[0], reverse=True)
        return resonances

    def inject_memory(self, J_current: Dict[Tuple[int, int], float],
                      current_phases: Dict[int, float],
                      current_context: np.ndarray,
                      max_injections: int = 3) -> Dict[Tuple[int, int], float]:
        """
        Inject resonant memories into current edge couplings.

        This is the "dark matter gravity" - memories warp the Hamiltonian
        by modifying J couplings proportionally to resonance.
        """
        resonances = self.compute_resonance(J_current, current_phases, current_context)
        J_new = dict(J_current)

        for resonance, snap in resonances[:max_injections]:
            inject_strength = self.injection_strengths.get(id(snap), 1.0)
            for edge, mem_J in snap.J_values.items():
                old_J = J_new.get(edge, 0.0)
                # Resonance-weighted injection with activation bound
                delta = resonance * inject_strength * mem_J
                J_new[edge] = float(squash_sin(old_J + delta))

        return J_new

    def reinforce_memory(self, snapshot_id: str, reward: float):
        """Meta-learning: adjust memory injection strength based on outcome."""
        # Find snapshot by label or index
        for snap in self.snapshots:
            if f"snap_{self.snapshots.index(snap)}" == snapshot_id or snap.label == snapshot_id:
                self.injection_strengths[id(snap)] = np.clip(
                    self.injection_strengths.get(id(snap), 1.0) + 0.01 * reward, 0.1, 5.0)
                snap.strength = np.clip(snap.strength + 0.005 * reward, 0.1, 2.0)
                break

    def get_stats(self) -> Dict:
        return {
            "num_snapshots": len(self.snapshots),
            "avg_strength": np.mean([s.strength for s in self.snapshots]) if self.snapshots else 0,
            "labels": [s.label for s in self.snapshots if s.label]
        }


def extract_universe_phases(state_vector: np.ndarray, n_universes: int,
                            n_layers: int) -> Dict[int, float]:
    """Extract phase information per universe from quantum state."""
    phases = {}
    # Simplified: compute phase from reduced density matrix of each universe
    # For full implementation, would do partial trace
    dim = len(state_vector)
    n_qubits = int(np.log2(dim))
    for u in range(n_universes):
        # Aggregate phase across layers for this universe
        phase_sum = 0.0
        count = 0
        for L in range(n_layers):
            q = L * n_universes + u
            # Measure Z expectation as phase proxy
            prob_0 = sum(abs(state_vector[i])**2 for i in range(dim)
                         if ((i >> (n_qubits - 1 - q)) & 1) == 0)
            prob_1 = 1.0 - prob_0
            phase_sum += np.arctan2(prob_1, prob_0)  # phase in [-pi/2, pi/2]
            count += 1
        phases[u] = phase_sum / max(count, 1)
    return phases


def extract_activation_order(J_values: Dict[Tuple[int, int], float],
                             threshold: float = 0.1) -> List[Tuple[int, int]]:
    """Extract temporal activation order from edge couplings (strongest first)."""
    return [e for e, v in sorted(J_values.items(), key=lambda x: -abs(x[1]))
            if abs(v) > threshold]