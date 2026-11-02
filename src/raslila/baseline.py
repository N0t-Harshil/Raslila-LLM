"""Standard decoder-only transformer: the yardstick Raslila must beat.

Same recipe as grid.py (RMSNorm pre-norm, RoPE, QK-norm, SwiGLU, tied
embeddings, no dropout), so the scaling study compares architectures only.
"""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .nn_common import RMSNorm, swiglu_hidden, rope_tables, apply_rope, init_linear_


@dataclass
class BaselineConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    rope_theta: float = 10000.0
    activation_checkpoint: bool = False

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.n_heads, self.head_dim = h, d // h
        self.norm1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.norm2 = RMSNorm(d)
        f = swiglu_hidden(d)
        self.w12 = nn.Linear(d, 2 * f, bias=False)
        self.w3 = nn.Linear(f, d, bias=False)
        out_scale = 1.0 / (2 * cfg.n_layers) ** 0.5
        init_linear_(self.qkv.weight, d)
        init_linear_(self.o.weight, d, out_scale)
        init_linear_(self.w12.weight, d)
        init_linear_(self.w3.weight, f, out_scale)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.n_heads, self.head_dim).unbind(2)
        q = apply_rope(self.q_norm(q).transpose(1, 2), cos, sin)
        k = apply_rope(self.k_norm(k).transpose(1, 2), cos, sin)
        y = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D))
        a, b = self.w12(self.norm2(x)).chunk(2, dim=-1)
        return x + self.w3(F.silu(a) * b)


class Baseline(nn.Module):
    family = 'baseline'

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, idx, **_):
        B, T = idx.shape
        cos, sin = rope_tables(T, self.config.d_model // self.config.n_heads,
                               self.config.rope_theta, idx.device)
        x = self.tok_emb(idx)
        for blk in self.blocks:
            if self.config.activation_checkpoint and self.training:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        logits = F.linear(self.norm(x), self.tok_emb.weight)
        return {'logits': logits, 'exit_logits': [logits], 'sweeps': 0}

    def flops_per_token(self, seq_len, sweeps=None):
        """Training FLOPs per token (fwd+bwd = 3x fwd), Chinchilla accounting."""
        c = self.config
        f = swiglu_hidden(c.d_model)
        per_layer = 2 * (4 * c.d_model ** 2 + 3 * c.d_model * f) + 4 * seq_len * c.d_model
        head = 2 * c.d_model * c.vocab_size
        return 3 * (c.n_layers * per_layer + head)
