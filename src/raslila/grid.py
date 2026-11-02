"""Raslila v2: the 3D brain grid, rebuilt to run fast on GPUs.

WHAT IT IS
    A grid of transformer blocks, [layers, rows, cols]. Every block owns its own
    weights. Inside a sheet (one layer) blocks talk sideways to their grid
    neighbours (plus a few small-world long-range links); between sheets they
    talk upward and -- on the way back down -- through feedback. The deepest
    sheet is read out. With `max_think_cycles > 0` the model "thinks in loops":
    cycle 1 is one bottom-up pass, every further cycle goes down through the
    feedback links and back up, an answer is read out after every cycle, and at
    inference it stops as soon as the answer has settled.

WHY v2 (measured on the A100, 2026-11)
    v1 (raslila.py) ran its 125 blocks one at a time with tiny matmuls, and
    trained at ~1,000 tokens/s -- ~150-200x slower than a standard transformer
    of the same size. That alone makes it impossible to train at scale. v2 keeps
    the idea and changes the execution:

    - All blocks of a sheet run TOGETHER as one batched matmul (their weights
      are stacked [N, ...]), and attention for all of them is one fused
      flash-attention call. Sequential depth is layers x node_depth per sweep,
      like a normal transformer, instead of one step per block.
    - Blocks in a sheet update from the same snapshot (Jacobi), which is what
      makes them parallel. v1's in-sheet Gauss-Seidel order bought ~0.026 nats
      at a ~25x sequential cost.
    - Messages are cheap linear maps of normalised neighbour states instead of
      618 separate edge MLPs; message projections start at zero, so the model
      begins as clean parallel towers and learns how much to listen.
    - Between thinking cycles the state is re-normalised and the input is
      re-injected (v1's 7x-per-cycle growth sent the adaptive run to NaN).
    - The readout projects the whole deepest sheet (N * d_node) to the
      embedding width instead of averaging it down to one block's width.

    Grids are uniform here (every sheet rows x cols); v1 keeps per-layer shapes.
"""
import random
from dataclasses import dataclass, asdict, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .nn_common import RMSNorm, rms_norm_nd, swiglu_hidden, rope_tables, apply_rope


@dataclass
class GridConfig:
    vocab_size: int = 32768
    grid_shape: List[int] = field(default_factory=lambda: [4, 4, 4])  # [layers, rows, cols]
    d_node: int = 256          # width of every block
    n_heads: int = 4           # per block (head_dim = d_node / n_heads)
    node_depth: int = 2        # transformer blocks stacked inside one grid cell
    d_embed: int = 768         # token embedding / readout width (tied head)
    depth_sweeps: int = 2      # fixed-depth mode: alternating passes, last one upward
    max_think_cycles: int = 0  # >0: adaptive loops (overrides depth_sweeps)
    halt_confidence: float = 0.9
    halt_tol: float = 0.02
    lateral: bool = True
    diagonal: bool = False
    long_range_prob: float = 0.1
    max_long_range: int = 3
    feedback: bool = True
    connectivity_seed: int = 42
    rope_theta: float = 10000.0
    activation_checkpoint: bool = True

    def __post_init__(self):
        if len(self.grid_shape) != 3 or min(self.grid_shape) < 1:
            raise ValueError('grid_shape must be [layers, rows, cols] of positive ints')
        if self.d_node % self.n_heads:
            raise ValueError('d_node must be divisible by n_heads')

    def to_dict(self):
        return asdict(self)


def sheet_adjacency(rows, cols, diagonal, long_range_prob, max_long_range, rng):
    """Row-normalised [N, N] neighbour matrix of one sheet (no self loops)."""
    n = rows * cols
    a = torch.zeros(n, n)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr, dc) == (0, 0) or (not diagonal and dr and dc):
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        a[i, rr * cols + cc] = 1.0
            if n > 1 and rng.random() < long_range_prob:
                for _ in range(rng.randint(1, max(1, max_long_range))):
                    j = rng.randrange(n)
                    if j != i:
                        a[i, j] = 1.0
    deg = a.sum(1, keepdim=True).clamp_min(1.0)
    return a / deg


class GroupedBlock(nn.Module):
    """N independent pre-norm transformer blocks evaluated as one batched op."""

    def __init__(self, n, d, heads, out_scale):
        super().__init__()
        self.n, self.d, self.h, self.hd = n, d, heads, d // heads
        f = swiglu_hidden(d)
        self.f = f
        self.norm1 = nn.Parameter(torch.ones(n, 1, d))
        self.qkv = nn.Parameter(torch.randn(n, d, 3 * d) / d ** 0.5)
        self.o = nn.Parameter(torch.randn(n, d, d) * out_scale / d ** 0.5)
        self.q_norm = nn.Parameter(torch.ones(self.hd))
        self.k_norm = nn.Parameter(torch.ones(self.hd))
        self.norm2 = nn.Parameter(torch.ones(n, 1, d))
        self.w12 = nn.Parameter(torch.randn(n, d, 2 * f) / d ** 0.5)
        self.w3 = nn.Parameter(torch.randn(n, f, d) * out_scale / f ** 0.5)

    def forward(self, x, cos, sin):
        # x: [N, B, T, D]
        N, B, T, D = x.shape
        flat = x.reshape(N, B * T, D)
        # torch.bmm is on autocast's bf16 list, so under autocast these run in
        # bf16 on tensor cores; on CPU (fp32) they run as written.
        qkv = torch.bmm(rms_norm_nd(flat, self.norm1), self.qkv)
        qkv = qkv.view(N * B, T, 3, self.h, self.hd)
        q, k, v = qkv.unbind(2)
        q = apply_rope(rms_norm_nd(q, self.q_norm).transpose(1, 2), cos, sin)
        k = apply_rope(rms_norm_nd(k, self.k_norm).transpose(1, 2), cos, sin)
        v = v.transpose(1, 2).to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(N, B * T, D)
        flat = flat + torch.bmm(y, self.o).to(flat.dtype)
        a, b = torch.bmm(rms_norm_nd(flat, self.norm2), self.w12).chunk(2, dim=-1)
        flat = flat + torch.bmm(F.silu(a) * b, self.w3).to(flat.dtype)
        return flat.view(N, B, T, D)


class Sheet(nn.Module):
    """One layer of the grid: lateral + vertical messages, then the blocks."""

    def __init__(self, cfg, layer, adjacency, out_scale):
        super().__init__()
        L = cfg.grid_shape[0]
        n = cfg.grid_shape[1] * cfg.grid_shape[2]
        d = cfg.d_node
        self.register_buffer('adj', adjacency, persistent=False)
        self.lateral = bool(cfg.lateral) and n > 1
        if self.lateral:
            self.lat_norm = RMSNorm(d)
            self.lat_proj = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.lat_proj.weight)
        self.up = None
        self.down = None
        if layer > 0:                                    # message from the sheet below
            self.up_norm = RMSNorm(d)
            self.up = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.up.weight)
        if cfg.feedback and layer < L - 1:               # feedback from the sheet above
            self.down_norm = RMSNorm(d)
            self.down = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.down.weight)
        self.blocks = nn.ModuleList([GroupedBlock(n, d, cfg.n_heads, out_scale)
                                     for _ in range(cfg.node_depth)])

    def forward(self, h, src, ascending, cos, sin):
        N = h.shape[0]
        x = h
        if self.lateral:
            nb = (self.adj.to(h.dtype) @ h.reshape(N, -1)).view_as(h)
            x = x + self.lat_proj(self.lat_norm(nb))
        if src is not None:
            if ascending and self.up is not None:
                x = x + self.up(self.up_norm(src))
            elif not ascending and self.down is not None:
                x = x + self.down(self.down_norm(src))
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x


class RaslilaGrid(nn.Module):
    family = 'raslila'

    def __init__(self, cfg: GridConfig):
        super().__init__()
        self.config = cfg
        L, R, C = cfg.grid_shape
        self.L, self.N = L, R * C
        d, e = cfg.d_node, cfg.d_embed
        self.tok_emb = nn.Embedding(cfg.vocab_size, e)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.in_proj = nn.Linear(e, d, bias=False)
        self.role = nn.Parameter(torch.randn(L, self.N, 1, 1, d) * 0.02)
        rng = random.Random(cfg.connectivity_seed)       # wiring only; never the global RNG
        out_scale = 1.0 / (2 * L * cfg.node_depth) ** 0.5
        self.sheets = nn.ModuleList([
            Sheet(cfg, l, sheet_adjacency(R, C, cfg.diagonal, cfg.long_range_prob,
                                          cfg.max_long_range, rng), out_scale)
            for l in range(L)])
        self.cycle_norm = RMSNorm(d) if cfg.max_think_cycles > 0 else None
        self.ro_norm = nn.Parameter(torch.ones(self.N, 1, 1, d))
        self.ro_proj = nn.Linear(self.N * d, e, bias=False)
        self.norm = RMSNorm(e)

    # ---- pieces ---------------------------------------------------------
    def _sheet(self, l, h, src, ascending, cos, sin):
        if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self.sheets[l], h, src, ascending, cos, sin, use_reentrant=False)
        return self.sheets[l](h, src, ascending, cos, sin)

    def _readout(self, top):
        N, B, T, D = top.shape
        h = rms_norm_nd(top, self.ro_norm).permute(1, 2, 0, 3).reshape(B, T, N * D)
        return F.linear(self.norm(self.ro_proj(h)), self.tok_emb.weight)

    def _settled(self, logits, prev_p):
        p = torch.softmax(logits[:, -1].float(), dim=-1)
        if float(p.max(-1).values.min()) >= self.config.halt_confidence:
            return True, p
        if prev_p is not None and float((0.5 * (p - prev_p).abs().sum(-1)).max()) < self.config.halt_tol:
            return True, p
        return False, p

    # ---- forward -----------------------------------------------------------
    def forward(self, idx, think_cycles=None, halt=None):
        B, T = idx.shape
        cfg = self.config
        cos, sin = rope_tables(T, cfg.d_node // cfg.n_heads, cfg.rope_theta, idx.device)
        x0 = self.in_proj(self.tok_emb(idx))                       # [B, T, D]
        init = [x0.unsqueeze(0) + self.role[l].to(x0.dtype) for l in range(self.L)]
        S = list(init)
        sweeps = 0

        def sweep(ascending):
            nonlocal sweeps
            order = range(self.L) if ascending else reversed(range(self.L))
            for l in order:
                if ascending:
                    src = S[l - 1] if l > 0 else None
                else:
                    src = S[l + 1] if l < self.L - 1 else None
                S[l] = self._sheet(l, S[l], src, ascending, cos, sin)
            sweeps += 1

        cycles = cfg.max_think_cycles if think_cycles is None else int(think_cycles)
        if cycles <= 0:
            n = max(1, cfg.depth_sweeps)
            for i in range(n):
                sweep((i % 2) == ((n - 1) % 2))                     # last sweep ends at the top
            logits = self._readout(S[-1])
            return {'logits': logits, 'exit_logits': [logits], 'sweeps': sweeps, 'cycles_used': None}

        if halt is None:
            halt = not self.training
        sweep(True)
        logits = self._readout(S[-1])
        exits, used, prev_p = [logits], 1, None
        for c in range(2, cycles + 1):
            if halt:
                done, prev_p = self._settled(logits, prev_p)
                if done:
                    break
            if self.cycle_norm is not None:
                S = [self.cycle_norm(S[l]) + init[l] for l in range(self.L)]
            sweep(False)
            sweep(True)
            logits = self._readout(S[-1])
            exits.append(logits)
            used = c
        return {'logits': logits, 'exit_logits': exits, 'sweeps': sweeps, 'cycles_used': used}

    # ---- accounting ----------------------------------------------------------
    def flops_per_token(self, seq_len, sweeps=None, exits=1):
        """Training FLOPs per token (fwd+bwd = 3x fwd), Chinchilla accounting."""
        c = self.config
        d, e, n = c.d_node, c.d_embed, self.N
        f = swiglu_hidden(d)
        if sweeps is None:
            sweeps = max(1, c.depth_sweeps) if c.max_think_cycles <= 0 else 2 * c.max_think_cycles - 1
        block = 2 * (4 * d * d + 3 * d * f) + 4 * seq_len * d
        msgs = 2 * n * d + 2 * 2 * d * d
        per_sweep = self.L * n * (c.node_depth * block + msgs)
        readout = 2 * n * d * e + 2 * e * c.vocab_size
        return 3 * (2 * e * d + sweeps * per_sweep + exits * readout)
