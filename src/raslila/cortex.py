"""Raslila v4: a 3D cortex that thinks in loops (docs/10_Research_and_v4_Design.md).

    tokens -> PRELUDE (unique layers) -> CORTEX (a full-width core of layers,
    looped r times, the input re-injected every loop) -> CODA (unique layers)
    -> next token

Why this shape: v2/v3 cut the width into many isolated narrow blocks and lost
to a same-size transformer by ~0.2 nats. Every looped model that works keeps
ONE full-width state (Huginn, Mixture-of-Recursions, TRM), and at EQUAL
PARAMETERS looped models beat a standard transformer, mostly on reasoning
(Saunshi et al. 2025; Geiping et al. 2025). Prelude/core/coda with unique
first and last layers is the best sharing pattern in both MoR ("Middle-Cycle")
and Saunshi ("middle looping").

The 3D cube: the core's d_model neurons are laid out as a cube
`cube_shape` (512 = 8x8x8, 1024 = 8x8x16). Every loop still mixes the whole
width (attention + MLP), so nothing is cut off. The layout is used to MEASURE
brain-likeness (scripts/brain_metrics.py) and, optionally, by a spatial
smoothness loss (`topo_weight`, TopoLM). TopoLM's loss cost +0.11 val loss at
its tuned weight, so it is off by default and is its own experiment.

Training recipe (Huginn, arXiv 2502.05171):
  * a random number of loops per step from a log-normal Poisson
    distribution (mean `mean_loops`, heavy tail up to `max_train_loops`),
    the same for all GPUs in a step (seeded by the step)
  * the state starts as random noise; the prelude output e is re-injected in
    every loop by concatenation + a learned adapter: s = core(A([s; e]))
  * only the FINAL output gets the loss; gradients flow through the last
    `backprop_loops` loops only, so memory does not grow with thinking time
  * "sandwich" norms (normalise before and after each sublayer): the norm
    setup that made Huginn's recurrence trainable at scale.
At evaluation the model is read out after several loop counts (`exit_at`),
which is how the accuracy-vs-thinking-time curve is measured.
"""
import math
import random
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .nn_common import RMSNorm, swiglu_hidden, rope_tables, apply_rope, init_linear_


@dataclass
class CortexConfig:
    vocab_size: int = 32768
    d_model: int = 1024
    n_heads: int = 16
    n_prelude: int = 2
    n_core: int = 6
    n_coda: int = 2
    cube_shape: Optional[List[int]] = None        # e.g. [8, 8, 16]; product must be d_model
    mean_loops: float = 4.0                       # mean loops per training step
    max_train_loops: int = 16                     # cap of the heavy tail during training
    backprop_loops: int = 4                       # gradients through the last k loops only
    test_loops: int = 8                           # default loops at evaluation / generation
    eval_loops: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    init_state_std: float = 1.0
    halt_kl: float = 5e-4                         # "am I done?": stop when the answer stops changing
    halt_min_loops: int = 2
    topo_weight: float = 0.0                      # TopoLM-style 3D smoothness loss (0 = off)
    topo_neighbourhoods: int = 4
    topo_radius: int = 2
    rope_theta: float = 10000.0
    activation_checkpoint: bool = False
    arch: str = 'cortex'

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if self.cube_shape is not None and math.prod(self.cube_shape) != self.d_model:
            raise ValueError('cube_shape {} does not multiply to d_model {}'.format(self.cube_shape, self.d_model))
        if min(self.n_core, self.backprop_loops, self.test_loops) < 1:
            raise ValueError('n_core, backprop_loops and test_loops must be >= 1')

    def to_dict(self):
        return asdict(self)


class SandwichBlock(nn.Module):
    """Transformer layer with norms before AND after each sublayer (Huginn).

    x^ = n2(x + Attn(n1(x)));  x = n4(x^ + MLP(n3(x^)))
    The trailing norms keep the state's scale fixed however many times the
    block is applied, which a plain pre-norm residual stream does not.
    """

    def __init__(self, d, heads, out_scale):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.n1, self.n2, self.n3, self.n4 = RMSNorm(d), RMSNorm(d), RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)
        f = swiglu_hidden(d)
        self.w12 = nn.Linear(d, 2 * f, bias=False)
        self.w3 = nn.Linear(f, d, bias=False)
        init_linear_(self.qkv.weight, d)
        init_linear_(self.o.weight, d, out_scale)
        init_linear_(self.w12.weight, d)
        init_linear_(self.w3.weight, f, out_scale)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, self.hd).unbind(2)
        q = apply_rope(self.q_norm(q).transpose(1, 2), cos, sin)
        k = apply_rope(self.k_norm(k).transpose(1, 2), cos, sin)
        y = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=True)
        x = self.n2(x + self.o(y.transpose(1, 2).reshape(B, T, D)))
        a, b = self.w12(self.n3(x)).chunk(2, dim=-1)
        return self.n4(x + self.w3(F.silu(a) * b))


def sample_loops(rng, mean, cap):
    """Log-normal Poisson (Huginn eq. 1-2): mostly near the mean, heavy tail."""
    rbar = max(1e-3, mean - 1.0)
    sigma = 0.5
    tau = rng.gauss(math.log(rbar) - sigma ** 2 / 2, sigma)
    lam = math.exp(tau)
    # Poisson by inversion (Python's random has no Poisson sampler)
    k, p, u = 0, math.exp(-lam), rng.random()
    c = p
    while u > c and k < 10 * cap:
        k += 1
        p *= lam / k
        c += p
    return int(min(cap, k + 1))


class RaslilaCortex(nn.Module):
    family = 'raslila'
    looped = True

    def __init__(self, cfg: CortexConfig):
        super().__init__()
        self.config = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        eff = cfg.n_prelude + cfg.mean_loops * cfg.n_core + cfg.n_coda
        out_scale = 1.0 / (2 * eff) ** 0.5
        mk = lambda n: nn.ModuleList([SandwichBlock(d, cfg.n_heads, out_scale) for _ in range(n)])
        self.prelude, self.core, self.coda = mk(cfg.n_prelude), mk(cfg.n_core), mk(cfg.n_coda)
        self.adapter = nn.Linear(2 * d, d, bias=False)
        init_linear_(self.adapter.weight, 2 * d)
        self.norm = RMSNorm(d)
        self._train_loop_rng = random.Random(0)
        self._compiled = False

    # ---- pieces ------------------------------------------------------------------
    def _run(self, blocks, x, cos, sin):
        for blk in blocks:
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        return x

    def _loop(self, s, e, cos, sin):
        return self._run(self.core, self.adapter(torch.cat([s, e], dim=-1)), cos, sin)

    def _readout(self, s, cos, sin):
        return F.linear(self.norm(self._run(self.coda, s, cos, sin)), self.tok_emb.weight)

    def compile_blocks(self):
        """torch.compile each layer, not the whole model: the loop count changes
        every step and would otherwise force a recompile per value."""
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        for blk in list(self.prelude) + list(self.core) + list(self.coda):
            blk.forward = torch.compile(blk.forward)
        self._compiled = True

    def sample_train_loops(self, rng):
        return sample_loops(rng, self.config.mean_loops, self.config.max_train_loops)

    # ---- forward -------------------------------------------------------------------
    def forward(self, idx, loops=None, exit_at=None, think_cycles=None, halt=None, return_state=False):
        """loops: loop count (training: sampled by the caller; default test_loops
        at eval, mean_loops in training). exit_at: list of loop counts to read
        out (evaluation only). think_cycles is accepted as an alias of loops."""
        cfg = self.config
        B, T = idx.shape
        cos, sin = rope_tables(T, cfg.d_model // cfg.n_heads, cfg.rope_theta, idx.device)
        if loops is None:
            loops = think_cycles
        if exit_at:
            exits_set = sorted(set(int(x) for x in exit_at))
            loops = exits_set[-1]
        elif loops is None:
            loops = int(round(cfg.mean_loops)) if self.training else cfg.test_loops
        loops = int(loops)

        e = self._run(self.prelude, self.tok_emb(idx), cos, sin)
        gen = torch.Generator(device='cpu').manual_seed(1234) if not self.training else None
        s = (torch.randn(e.shape, generator=gen) if gen is not None
             else torch.randn(e.shape, device=e.device)).to(e.device, e.dtype) * cfg.init_state_std

        exits, labels = [], []
        if halt and not self.training and not exit_at:
            # Adaptive thinking (Huginn sec. 6.1; the "am I done?" step HRM's
            # analysis found mattered): read out after every loop and stop as
            # soon as the next-token distribution at the last position stops
            # changing (KL below halt_kl). `loops` is then the cap.
            prev = None
            used = loops
            for r in range(1, loops + 1):
                s = self._loop(s, e, cos, sin)
                lg = self._readout(s, cos, sin)
                lp = lg[:, -1].float().log_softmax(-1)
                if prev is not None and r >= cfg.halt_min_loops:
                    kl = (prev.exp() * (prev - lp)).sum(-1).max()
                    if float(kl) < cfg.halt_kl:
                        used = r
                        break
                prev = lp
            return {'logits': lg, 'exit_logits': [lg], 'exit_loops': [used],
                    'sweeps': used, 'cycles_used': used}
        if exit_at:
            for r in range(1, loops + 1):
                s = self._loop(s, e, cos, sin)
                if r in exits_set:
                    exits.append(self._readout(s, cos, sin))
                    labels.append(r)
        else:
            k = cfg.backprop_loops if (self.training and torch.is_grad_enabled()) else loops
            n_free = max(0, loops - k)
            if n_free:
                with torch.no_grad():
                    for _ in range(n_free):
                        s = self._loop(s, e.detach(), cos, sin)
                s = s.detach()
                # Autocast caches the bf16 copy of each weight on first use and
                # reuses it for the rest of the region. A copy made under no_grad
                # has no autograd link, so the loops below would give the core and
                # adapter NO gradient (DDP then fails with "did not receive grad").
                torch.clear_autocast_cache()
            for _ in range(loops - n_free):
                s = self._loop(s, e, cos, sin)
            exits.append(self._readout(s, cos, sin))
            labels.append(loops)

        out = {'logits': exits[-1], 'exit_logits': exits, 'exit_loops': labels,
               'sweeps': loops, 'cycles_used': loops}
        if self.training and cfg.topo_weight > 0 and cfg.cube_shape:
            out['aux_loss'] = cfg.topo_weight * self.topo_loss(s)
        if return_state:
            out['state'] = s
        return out

    # ---- 3D spatial smoothness (TopoLM, arXiv 2410.11516, eq. 1) -------------------
    def unit_coords(self):
        sh = self.config.cube_shape
        g = torch.stack(torch.meshgrid(*[torch.arange(n) for n in sh], indexing='ij'), -1)
        return g.reshape(-1, 3).float()

    def topo_loss(self, s, max_tokens=1024):
        """Neighbouring cube neurons should respond alike: 0.5 * (1 - corr(r, 1/(d+1)))
        over random neighbourhoods, r = pairwise activation correlation."""
        cfg = self.config
        coords = self.unit_coords().to(s.device)
        x = s.reshape(-1, s.size(-1)).float()
        if x.size(0) > max_tokens:
            x = x[torch.randperm(x.size(0), device=x.device)[:max_tokens]]
        losses = []
        for _ in range(cfg.topo_neighbourhoods):
            c = coords[torch.randint(0, coords.size(0), (1,)).item()]
            idx = ((coords - c).abs().amax(-1) <= cfg.topo_radius).nonzero().squeeze(-1)
            if idx.numel() < 3:
                continue
            a = x[:, idx]
            a = (a - a.mean(0)) / (a.std(0) + 1e-5)
            corr = (a.t() @ a) / a.size(0)
            cc = coords[idx]
            inv = 1.0 / (torch.cdist(cc, cc, p=float('inf')) + 1.0)
            iu = torch.triu_indices(idx.numel(), idx.numel(), 1, device=s.device)
            r, dv = corr[iu[0], iu[1]], inv[iu[0], iu[1]]
            r, dv = r - r.mean(), dv - dv.mean()
            pear = (r * dv).sum() / (r.norm() * dv.norm() + 1e-8)
            losses.append(0.5 * (1 - pear))
        return torch.stack(losses).mean() if losses else s.new_zeros(())

    # ---- accounting -----------------------------------------------------------------
    def _layer_flops(self, seq_len):
        d = self.config.d_model
        f = swiglu_hidden(d)
        return 2 * (4 * d * d + 3 * d * f) + 4 * seq_len * d

    def flops_per_token(self, seq_len, sweeps=None, exits=1):
        """Training FLOPs per token for one step that ran `sweeps` loops: forward
        through every loop, backward (2x forward) only through the last
        backprop_loops loops plus prelude/coda. Chinchilla-style accounting."""
        c = self.config
        loops = int(sweeps if sweeps is not None else c.test_loops)
        L = self._layer_flops(seq_len)
        adapter = 2 * 2 * c.d_model * c.d_model
        loop_f = c.n_core * L + adapter
        head = 2 * c.d_model * c.vocab_size
        ends = (c.n_prelude + c.n_coda) * L + head
        fwd = ends + loops * loop_f
        bwd = 2 * (ends + min(loops, c.backprop_loops) * loop_f)
        return fwd + bwd

    def expected_flops_per_token(self, seq_len, n=4000):
        rng = random.Random(7)
        return sum(self.flops_per_token(seq_len, self.sample_train_loops(rng)) for _ in range(n)) / n

    def inference_flops_per_token(self, seq_len, loops=None):
        c = self.config
        loops = loops or c.test_loops
        L = self._layer_flops(seq_len)
        return (c.n_prelude + c.n_coda) * L + loops * (c.n_core * L + 4 * c.d_model ** 2) + 2 * c.d_model * c.vocab_size
