"""Building blocks shared by the scaling-study models (baseline.py, grid.py).

Both model families use exactly the same modern recipe -- pre-norm RMSNorm,
rotary positions, per-head QK-norm, SwiGLU, tied embeddings -- so a difference
in loss between them comes from the architecture, not from the block recipe.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).type_as(x)


def rms_norm_nd(x, weight, eps=1e-6):
    """RMSNorm with a per-group weight that broadcasts against x."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * weight.float()).type_as(x)


def swiglu_hidden(d, mult=8 / 3, multiple_of=64):
    """LLaMA rule: ~8/3*d, rounded up so the matmuls stay tensor-core friendly."""
    h = int(mult * d)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


_ROPE_CACHE = {}


def rope_tables(seq_len, head_dim, theta, device):
    key = (seq_len, head_dim, float(theta), str(device))
    hit = _ROPE_CACHE.get(key)
    if hit is None:
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(seq_len, device=device).float()
        f = torch.outer(t, inv)
        emb = torch.cat([f, f], dim=-1)
        hit = (emb.cos(), emb.sin())
        _ROPE_CACHE[key] = hit
    return hit


def apply_rope(x, cos, sin):
    """x: [..., T, Dh]; cos/sin: [T, Dh]."""
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos.to(x.dtype) + rot * sin.to(x.dtype)


def init_linear_(w, fan_in, scale=1.0):
    nn.init.normal_(w, mean=0.0, std=scale / math.sqrt(fan_in))


def count_params(model, exclude_embedding=True):
    n = sum(p.numel() for p in model.parameters())
    if exclude_embedding and hasattr(model, 'tok_emb'):
        n -= model.tok_emb.weight.numel()
    return n
