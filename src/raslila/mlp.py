"""Trainable MLP with CORRECTED backprop (Tech Review #3).

OLD buggy code:
    np.outer(p_vector, delta_raw * (1-hidden**2) @ self.W_out.T)
    -> (8,) @ (1,8) shape mismatch -> ValueError crash.

NEW: proper chain rule.
    hidden  (H,) = tanh(p @ W + b)
    raw     (1,) = hidden @ W_out + b_out
    out     scalar = Act(raw[0])
    dL/dW_out = delta_raw * hidden  -> (H,1)
    dL/db_out = delta_raw           -> (1,)
    dL/dhidden_pre = (delta_raw * W_out[:,0]) * (1-hidden^2) -> (H,)
    dL/dW = outer(p (4,), dL/dhidden_pre (H,)) -> (4,H)
    dL/db = dL/dhidden_pre -> (H,)
"""
import numpy as np
from .activation import squash_sin, dsquash_sin_dx


class TrainableMLP:
    def __init__(self, activation=squash_sin, d_activation=dsquash_sin_dx, seed=0, hidden_dim=8):
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.W = rng.normal(0, 0.1, size=(4, hidden_dim))
        self.b = np.zeros(hidden_dim)
        self.W_out = rng.normal(0, 0.1, size=(hidden_dim, 1))
        self.b_out = np.zeros(1)
        self.act = activation
        self.d_act = d_activation

    def forward(self, p_vector):
        hidden = np.tanh(p_vector @ self.W + self.b)
        raw = hidden @ self.W_out + self.b_out
        return float(self.act(raw[0]))

    def forward_cache(self, p_vector):
        hidden = np.tanh(p_vector @ self.W + self.b)
        raw = hidden @ self.W_out + self.b_out
        out = float(self.act(raw[0]))
        return out, hidden, raw

    def gradient(self, p_vector, grad_J):
        """Return dict of grads with correct shapes."""
        hidden = np.tanh(p_vector @ self.W + self.b)
        raw = hidden @ self.W_out + self.b_out
        d_act = float(self.d_act(raw[0]))
        delta_raw = float(grad_J) * d_act  # scalar
        grad_W_out = (delta_raw * hidden).reshape(-1, 1)  # (H,1)
        grad_b_out = np.array([delta_raw])  # (1,)
        grad_hidden_pre = (delta_raw * self.W_out[:, 0]) * (1.0 - hidden ** 2)  # (H,)
        grad_W = np.outer(p_vector, grad_hidden_pre)  # (4,H)
        grad_b = grad_hidden_pre.copy()  # (H,)
        return {"W": grad_W, "b": grad_b, "W_out": grad_W_out, "b_out": grad_b_out}

    def apply_grads(self, grads, lr=0.01, clip=1.0):
        for k in ("W", "b", "W_out", "b_out"):
            g = np.clip(grads[k], -clip, clip)
            setattr(self, k, getattr(self, k) - lr * g)
