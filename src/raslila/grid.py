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

v3 ROUTING (`routing: true`, 2026-11-08)
    The 50m/125m study showed v2 losing to the baseline (+0.19 / +0.12 val loss
    at equal size) and its 2nd thinking cycle not helping. In v2 every block
    received the same fixed AVERAGE of its neighbours through one linear map,
    upward only on the way up and downward only on the way down; nothing could
    choose. With routing, for every token every block decides whom to listen
    to: each sideways neighbour (and its long-range links), the block below
    (fresh this sweep), the block above (its previous state = feedback), or
    nobody. The choice is a small attention over those sources through a
    narrow channel of `msg_rank` dims (default d_node/4), so connections stay
    cheap and a block that does better alone can switch them off.
    Because feedback from above arrives inside every upward sweep, a thinking
    cycle is ONE upward sweep instead of down + up (cheaper loops).
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
    routing: bool = False      # v3: per-token learned choice of direction (see top)
    msg_rank: int = 0          # width of the routed message channel; 0 = d_node // 4
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


class Router(nn.Module):
    """Per token, every block of a sheet chooses which neighbours to listen to.

    Sources of block i: the blocks of its own sheet it is wired to (grid
    neighbours + long-range links; snapshot of this sheet), block i of the sheet
    below (fresh this sweep), block i of the sheet above (previous state:
    feedback), and a "nobody" slot. Scores are q_i . k_source in a `rank`-dim
    channel plus a learned per-(block, source) bias; a softmax over the allowed
    sources picks the mix. Everything is per token and position-wise, so it is
    causal. The output projection starts at zero: training starts from
    independent towers and learns how much to listen.
    """

    def __init__(self, n, d, rank, adjacency, has_below, has_above, lateral):
        super().__init__()
        self.n, self.d, self.r = n, d, rank
        self.has_below, self.has_above = has_below, has_above
        self.q_norm = nn.Parameter(torch.ones(n, 1, d))
        self.q = nn.Parameter(torch.randn(n, d, rank) / d ** 0.5)
        self.lat_norm = nn.Parameter(torch.ones(n, 1, d))
        self.kv_lat = nn.Parameter(torch.randn(n, d, 2 * rank) / d ** 0.5)
        if has_below:
            self.below_norm = nn.Parameter(torch.ones(n, 1, d))
            self.kv_below = nn.Parameter(torch.randn(n, d, 2 * rank) / d ** 0.5)
        if has_above:
            self.above_norm = nn.Parameter(torch.ones(n, 1, d))
            self.kv_above = nn.Parameter(torch.randn(n, d, 2 * rank) / d ** 0.5)
        self.o = nn.Parameter(torch.zeros(n, rank, d))
        # allowed sources per block: [n sideways | below | above | nobody]
        allow = torch.zeros(n, n + 3, dtype=torch.bool)
        if lateral:
            allow[:, :n] = adjacency > 0
        allow[:, n] = has_below
        allow[:, n + 1] = has_above
        allow[:, n + 2] = True
        self.register_buffer('allow', allow, persistent=False)
        self.route_bias = nn.Parameter(torch.zeros(n, n + 3))
        # One learned key per DIRECTION (source slot): lets a block choose
        # "look up" / "look left" / "listen to nobody" from its own state,
        # independently of what the neighbour currently holds.
        self.dir_key = nn.Parameter(torch.randn(n, n + 3, rank) * 0.02)

    def _kv(self, src, norm, w):
        N, B, T, D = src.shape
        return torch.bmm(rms_norm_nd(src.reshape(N, B * T, D), norm), w).chunk(2, dim=-1)

    def forward(self, x, below, above):
        N, B, T, D = x.shape
        flat = x.reshape(N, B * T, D)
        q = torch.bmm(rms_norm_nd(flat, self.q_norm), self.q)                   # [N, BT, r]
        k, v = self._kv(x, self.lat_norm, self.kv_lat)
        s_lat = torch.bmm(q.transpose(0, 1), k.permute(1, 2, 0))                # [BT, N, N]
        zero = s_lat.new_zeros(B * T, N, 1)
        allow = self.allow
        use_b = self.has_below and below is not None
        use_a = self.has_above and above is not None
        s_b = s_a = zero
        if use_b:
            kb, vb = self._kv(below, self.below_norm, self.kv_below)
            s_b = (q * kb).sum(-1).transpose(0, 1).unsqueeze(-1)
        if use_a:
            ka, va = self._kv(above, self.above_norm, self.kv_above)
            s_a = (q * ka).sum(-1).transpose(0, 1).unsqueeze(-1)
        if (self.has_below and not use_b) or (self.has_above and not use_a):
            allow = allow.clone()
            allow[:, N] = use_b
            allow[:, N + 1] = use_a
        s_dir = torch.bmm(q, self.dir_key.transpose(1, 2).to(q.dtype)).transpose(0, 1)  # [BT, N, N+3]
        scores = (torch.cat([s_lat, s_b, s_a, zero], dim=-1) + s_dir).float() * self.r ** -0.5
        scores = (scores + self.route_bias.float()).masked_fill(~allow, float('-inf'))
        w = torch.softmax(scores, dim=-1).to(v.dtype)                            # [BT, N, N+3]
        m = torch.bmm(w[..., :N], v.transpose(0, 1))                             # [BT, N, r]
        if use_b:
            m = m + w[..., N:N + 1] * vb.transpose(0, 1)
        if use_a:
            m = m + w[..., N + 1:N + 2] * va.transpose(0, 1)
        out = torch.bmm(m.transpose(0, 1), self.o)                               # [N, BT, D]
        return out.view(N, B, T, D)

    def flops_per_token(self):
        """Forward FLOPs per token per block (q, 3x k/v, out, scores, mix)."""
        n, r, d = self.n, self.r, self.d
        return 2 * d * r + 3 * (4 * d * r) + 2 * r * d + 4 * n * r + 2 * (n + 3) * r + 8 * r


class Sheet(nn.Module):
    """One layer of the grid: lateral + vertical messages, then the blocks."""

    def __init__(self, cfg, layer, adjacency, out_scale):
        super().__init__()
        L = cfg.grid_shape[0]
        n = cfg.grid_shape[1] * cfg.grid_shape[2]
        d = cfg.d_node
        self.register_buffer('adj', adjacency, persistent=False)
        self.router = None
        if cfg.routing:
            rank = int(cfg.msg_rank) or max(8, d // 4)
            self.router = Router(n, d, rank, adjacency, has_below=layer > 0,
                                 has_above=bool(cfg.feedback) and layer < L - 1,
                                 lateral=bool(cfg.lateral) and n > 1)
            self.lateral = False
            self.up = self.down = None
            self.blocks = nn.ModuleList([GroupedBlock(n, d, cfg.n_heads, out_scale)
                                         for _ in range(cfg.node_depth)])
            return
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

    def forward(self, h, below, above, ascending, cos, sin):
        N = h.shape[0]
        x = h
        if self.router is not None:
            x = x + self.router(h, below, above).to(h.dtype)
            for blk in self.blocks:
                x = blk(x, cos, sin)
            return x
        src = below if ascending else above
        if self.lateral:
            nb = (self.adj.to(h.dtype) @ h.reshape(N, -1)).view_as(h)
            x = x + self.lat_proj(self.lat_norm(nb)).to(h.dtype)
        if src is not None:
            if ascending and self.up is not None:
                x = x + self.up(self.up_norm(src)).to(h.dtype)
            elif not ascending and self.down is not None:
                x = x + self.down(self.down_norm(src)).to(h.dtype)
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
        # Residual branches added per readout path: layers x node_depth per
        # sweep, times the sweeps run between two re-normalisations (a thinking
        # cycle is down + up = 2; fixed depth is depth_sweeps). Scaling the
        # output projections by 1/sqrt(2 * branches) keeps the stream's variance
        # independent of how deep the loop runs (GPT-2 / LLaMA init rule).
        if cfg.max_think_cycles > 0:
            sweeps_per_path = 1 if cfg.routing else 2
        else:
            sweeps_per_path = max(1, cfg.depth_sweeps)
        out_scale = 1.0 / (2 * L * cfg.node_depth * sweeps_per_path) ** 0.5
        self.sheets = nn.ModuleList([
            Sheet(cfg, l, sheet_adjacency(R, C, cfg.diagonal, cfg.long_range_prob,
                                          cfg.max_long_range, rng), out_scale)
            for l in range(L)])
        self.cycle_norm = RMSNorm(d) if cfg.max_think_cycles > 0 else None
        self.ro_norm = nn.Parameter(torch.ones(self.N, 1, 1, d))
        self.ro_proj = nn.Linear(self.N * d, e, bias=False)
        self.norm = RMSNorm(e)

    # ---- pieces ---------------------------------------------------------
    def _sheet(self, l, h, below, above, ascending, cos, sin):
        if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self.sheets[l], h, below, above, ascending, cos, sin, use_reentrant=False)
        return self.sheets[l](h, below, above, ascending, cos, sin)

    def sweeps_for_cycles(self, cycles):
        """Sweeps run by `cycles` thinking cycles (v3: one each; v2: up, then down + up)."""
        return cycles if self.config.routing else 2 * cycles - 1

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
        # The residual stream is kept in fp32 (as the baseline's is: its
        # embedding output is fp32 and every block adds into it). Under bf16
        # autocast in_proj returns bf16, and summing 20-60 residual updates in
        # bf16 (8-bit mantissa) visibly costs quality. Matmuls still run in
        # bf16; only the running sum is fp32.
        x0 = self.in_proj(self.tok_emb(idx)).float()               # [B, T, D]
        init = [x0.unsqueeze(0) + self.role[l].float() for l in range(self.L)]
        S = list(init)
        sweeps = 0

        def sweep(ascending):
            nonlocal sweeps
            order = range(self.L) if ascending else reversed(range(self.L))
            for l in order:
                below = S[l - 1] if l > 0 else None
                above = S[l + 1] if l < self.L - 1 else None
                S[l] = self._sheet(l, S[l], below, above, ascending, cos, sin)
            sweeps += 1

        cycles = cfg.max_think_cycles if think_cycles is None else int(think_cycles)
        if cycles <= 0:
            n = max(1, cfg.depth_sweeps)
            for i in range(n):
                # v3: every sweep goes up (feedback arrives through the router);
                # v2: alternate, the last sweep ends at the top
                sweep(True if cfg.routing else (i % 2) == ((n - 1) % 2))
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
                S = [self.cycle_norm(S[l]).float() + init[l] for l in range(self.L)]
            if not cfg.routing:
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
            sweeps = max(1, c.depth_sweeps) if c.max_think_cycles <= 0 \
                else self.sweeps_for_cycles(c.max_think_cycles)
        block = 2 * (4 * d * d + 3 * d * f) + 4 * seq_len * d
        if c.routing:
            msgs = self.sheets[0].router.flops_per_token()
        else:
            msgs = 2 * n * d + 2 * 2 * d * d
        per_sweep = self.L * n * (c.node_depth * block + msgs)
        readout = 2 * n * d * e + 2 * e * c.vocab_size
        return 3 * (2 * e * d + sweeps * per_sweep + exits * readout)
