"""Resonance memory injected on J (not alpha/beta).

Fixes Tech Review conceptual bug: original code injected memory into
alpha/beta:
    hamiltonian.alpha = squash_sin(alpha + gamma*0.8)
    hamiltonian.beta  = squash_sin(beta + gamma*0.2)
Correct location per Phase 3 spec:
    J_ij^new = Act(J_ij^old + exp(-Hamming) * J_ij^mem)

This module implements the correct version.
"""
import numpy as np
from .activation import squash_sin


class TopologicalMemory:
    """Minimal topological snapshot store."""

    def __init__(self):
        self.snapshots = {}  # key -> {"J": dict, "pattern": array}

    def save(self, key, J_dict, pattern):
        self.snapshots[key] = {"J": dict(J_dict), "pattern": np.array(pattern, float)}

    def resonance(self, current_pattern, key):
        mem = self.snapshots[key]["pattern"]
        cur = np.asarray(current_pattern, float)
        hamming = float(np.sum((cur - mem) ** 2))
        return float(np.exp(-hamming)), hamming

    def inject(self, J_current, key, factor, act=squash_sin):
        mem_J = self.snapshots[key]["J"]
        out = dict(J_current)
        for edge, mem_val in mem_J.items():
            old = float(out.get(edge, 0.0))
            out[edge] = float(act(old + factor * float(mem_val)))
        return out


def inject_memory(J_current, J_mem, current_pattern, mem_pattern, act=squash_sin):
    """One-shot resonance injection. Returns (J_new, factor, hamming)."""
    cur = np.asarray(current_pattern, float)
    mem = np.asarray(mem_pattern, float)
    hamming = float(np.sum((cur - mem) ** 2))
    factor = float(np.exp(-hamming))
    out = dict(J_current)
    for edge, mem_val in J_mem.items():
        out[edge] = float(act(float(out.get(edge, 0.0)) + factor * float(mem_val)))
    return out, factor, hamming
