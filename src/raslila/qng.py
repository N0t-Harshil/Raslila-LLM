"""Quantum Natural Gradient optimizer with sync fix (Tech Review #2).

OLD bug: qng.params initialized once to [0.5,0.5]; loop modified
hamiltonian.alpha/beta via memory injection but never synced qng.params,
so qng.step updated from a stale point, discarding injection.

NEW: call qng.sync([alpha, beta]) before every step.
"""
import numpy as np
from scipy.linalg import pinv


class QNGOptimizer:
    def __init__(self, params, lr=0.05, clip=(-1.0, 1.0)):
        self.params = np.array(params, dtype=float)
        self.lr = lr
        self.clip = clip

    def sync(self, params):
        self.params = np.array(params, dtype=float)
        return self.params

    def step(self, grad, qfim, reg=1e-6):
        grad = np.asarray(grad, dtype=float)
        qfim = np.asarray(qfim, dtype=float)
        qfim_inv = pinv(qfim + reg * np.eye(len(grad)))
        self.params = self.params - self.lr * (qfim_inv @ grad)
        lo, hi = self.clip
        self.params = np.clip(self.params, lo, hi)
        return self.params.copy()
