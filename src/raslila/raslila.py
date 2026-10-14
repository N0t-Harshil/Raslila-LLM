"""
Raslila - 3D Brain-Inspired Transformer Network
===============================================

Key Concepts:
- Transformer blocks arranged on a 3D grid (x, y, z) like brain regions
- Edges ARE neural networks that process information (not just weights)
- 3D connectivity like brain: local + long-range connections
- Purely classical - no quantum computing
- Each region is a transformer block; edges process messages

Structure:
- SPATIAL axes: dim_x (width) x dim_y (height) x dim_z (front-to-back).
  Every region holds its own persistent, separate state stream across the whole
  sequence -- the "slots" that binding theory requires and that a 1D transformer
  structurally cannot provide.
- PROCESSING DEPTH: `num_layers` message-passing rounds. Rounds reuse the same
  region weights (recurrent / multidirectional: information flows 1->2->1->3,
  not just forward), so depth costs compute but not parameters.
- `node_depth`: transformer blocks stacked inside a single region.

Production config (config.yaml): 5x5x5 = 125 regions, 5 rounds, 163.6M params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict
from dataclasses import dataclass
import math
import random


@dataclass
class RaslilaConfig:
    """Configuration for the Raslila architecture.

    Production defaults mirror config.yaml (the authoritative source).

    STRUCTURE — a deep grid, like the cortex:

        layer 2 of 5   [ ][ ][ ][ ][ ]      <- grid_w blocks, connected laterally
                      [ ][ ][ ][ ][ ]      <- grid_h rows
                      [ ][ ][ ][ ][ ]
                      [ ][ ][ ][ ][ ]
                      [ ][ ][ ][ ][ ]
                         ||
        layer 1 of 5   [ ][ ][ ][ ][ ]      <- vertically connected to layer 2
                      [ ][ ][ ][ ][ ]
                       ...

    Each of `depth_layers` layers is a 2D `grid_h` x `grid_w` array of
    transformer blocks. Blocks connect to their lateral neighbours inside a
    layer, and to the corresponding block in adjacent layers. Layers are
    processed SEQUENTIALLY, and every layer owns a distinct set of block
    weights (unlike a recurrent loop, which reuses one set).

    A conventional LLM is the degenerate 1D case: one flat chain of blocks.
    Raslila restores the 2D sheet within each layer and stacks sheets in depth.

    Fields:
        depth_layers : number of sequential layers (sheets)
        grid_h, grid_w : 2D block layout inside one layer
        num_layers   : extra message-passing ROUNDS inside each layer
                       (multidirectional cascades; weights are reused)
        node_depth   : transformer blocks stacked inside one region
    """
    # Deep-grid layout
    depth_layers: int = 5   # sequential layers (cortical sheets)
    grid_h: int = 5         # rows per layer
    grid_w: int = 5         # blocks per row
    layout: str = 'deep_grid'   # 'deep_grid' | 'flat3d' (legacy flat lattice)

    # Region configuration
    hidden_dim: int = 256
    num_heads: int = 8
    # Bidirectional depth sweeps. The cortex is not a feedforward stack:
    # information goes UP through the layers and comes back DOWN, refining the
    # hypothesis. A single bottom-up pass computes the feedback edges but never
    # uses them, because the lower layer has already been written. `depth_sweeps`
    # alternates direction, so every edge is traversed both ways per sweep pair.
    depth_sweeps: int = 2     # up-down traversals over the whole stack.
                              # 1 = plain bottom-up (no real feedback).
                              # 2 = up then down (recommended minimum).
                              # 3+ = repeated refinement passes.

    num_layers: int = 1     # cascade rounds INSIDE each layer (multidirectional)
    node_depth: int = 1     # transformer blocks per region
    ff_dim: int = 1024
    dropout: float = 0.1

    # Legacy flat-lattice dimensions (used only when layout='flat3d')
    dim_x: int = 5
    dim_y: int = 5
    dim_z: int = 5

    # Modern block recipe (2023-2026 consensus: pre-norm RMSNorm + RoPE + SwiGLU).
    # Each is the empirically-stronger replacement for the GPT-2-era default.
    use_rope: bool = True      # rotary positions (replaces learned absolute table)
    rope_theta: float = 10000.0
    use_swiglu: bool = True    # gated SiLU FFN (replaces plain GELU MLP)
    use_rmsnorm: bool = True   # RMSNorm + pre-norm layout (replaces post-norm LayerNorm)
    qk_norm: bool = True       # per-head RMSNorm on Q/K before RoPE (stability)
    tie_weights: bool = True   # share token embedding with output head
    lns: bool = False          # LayerNorm Scaling: scale norm outputs by 1/sqrt(depth)
                               # (cures pre-norm "curse of depth"; Sun et al. 2025)
    gated_messages: bool = False  # input-dependent sigmoid gate on edge messages
                               # (neuromodulation: regions accept/reject input)
    activation_checkpoint: bool = False
                               # recompute each region's activations in backward
                               # instead of storing them. At 125 regions x node_depth
                               # blocks the stored activations dominate memory and
                               # are the reason a 20B config OOMs while the weights
                               # alone fit. Costs ~30% step time, cuts activation
                               # memory from O(regions*blocks) to O(1).

    # Brain mechanisms (neuroscience-grounded, all A/B tested):
    mtp_steps: int = 0         # extra future-byte prediction heads (t+2..t+K).
                               # Hierarchical multi-timescale prediction (Caucheteux:
                               # cortex predicts distant+abstract, not just t+1).
                               # 0 = off. Auxiliary loss only; generation unchanged.
    mtp_weight: float = 0.3    # weight of the MTP auxiliary loss
    workspace_slots: int = 0   # Global Workspace bottleneck (Baars/Dehaene/Bengio):
                               # nodes compete to write into K shared slots, then
                               # all read the broadcast back. 0 = off (pure edge msgs).
    readout_topk: int = 0      # k-winners-take-all over regions at readout
                               # (sparse coding: <2% neurons active). 0 = dense.

    # Binding / conceptual blending (the 3D grid is the substrate: 18 regions
    # are persistent SEPARATE state streams, exactly the "slots" binding
    # theory requires and a 1D transformer cannot provide).
    bind_roles: int = 4      # K learned role vectors. Each region state is bound
                               # to every role multiplicatively (Tensor-Product
                               # style), yielding an N*K composite "blended space"
                               # instead of collapsing the region set to a mean.
                               # 0 = off. Measured -0.041 nats, so ON by default.
    bind_local: bool = False  # additionally bind each region with its GRID
                               # neighbours (spatial relations: the 3D advantage).

    # Cortical growth: the model starts small and adds regions as it learns
    # (a brain grows; dense models are fixed at init). Every region always
    # participates -- this is NOT sparse routing; growth is uniform, so all
    # regions keep exchanging messages (one integrated system).
    grow_every: int = 0        # check for growth every N optimizer steps (0=off)
    grow_min_scale: float = 0.004  # grow when relative loss improvement over the
                               # window is below this (i.e. a plateau)
    grow_step: int = 2         # how many regions to add per growth event
    grow_max_nodes: int = 64   # hard ceiling on total regions

    # Multidirectional recurrence (the brain is not feedforward: information
    # flows back and forth — 1→2→1→3→… — before anything settles):
    immediate_updates: bool = True   # Gauss-Seidel cascades: a region reads its
                               # neighbors' FRESH states (updated earlier in the
                               # same sweep), so signals cross the whole grid in
                               # one round instead of one hop per round.
                               # Measured -0.026 nats for free, so ON by default.
    priority_order: bool = False  # event-driven sweeps: visit nodes in order of
                               # descending incoming-message magnitude (most
                               # surprised first), like firing on change.
    settle_thresh: float = 0.0  # >0: halt message rounds early once the max
                               # message magnitude drops below this (relaxation
                               # to convergence, like Hopfield settling).
                               # num_layers then acts as the round cap.

    # Edge network config. Edge params scale with region count (618 edges at
    # 5x5x5), so this is the main lever for trimming a large grid.
    edge_hidden_dim: int = 128
    edge_layers: int = 2

    # Lateral inhibition. In cortex ~20% of neurons are inhibitory interneurons,
    # and the resulting competition is what makes a representation SELECTIVE.
    # Every edge message in Raslila used to be additive, so blocks could never
    # suppress a competing block and binding stayed permissive (anything binds to
    # anything). Signed edges let a block argue against its neighbours.
    inhibit: bool = True        # emit a signed excitatory/inhibitory message
    inhibit_neg: bool = True    # the inhibitory component subtracts
    inhibit_beta: float = 1.0   # weight on the inhibitory component

    # Learning from experience (training-side; read by scripts/train_raslila.py).
    # Humans revisit what they have seen -- sleep replay re-runs the day's
    # episodes at 10-20x -- and that is what consolidates temporary memories and
    # prevents forgetting. A single pass over text is the recipe for
    # catastrophic forgetting.
    replay_prob: float = 0.25    # fraction of steps drawn from ALREADY-SEEN text
    replay_weight: float = 1.5   # replayed batches count more than fresh ones, so
                                 # consolidating an old memory beats one exposure

    # Persistent state over time. A brain's neurons integrate across time; an
    # ordinary transformer only ever sees its attention window. Each block keeps
    # a gated linear recurrence (h_t = a*h_{t-1} + (1-a)*x_t) with a LEARNED
    # per-channel decay, so different channels get different timescales - slow
    # ones carry topic, fast ones carry syntax. This is the memory the attention
    # window cannot provide.
    state_dim: int = 0          # per-block recurrent state width (0 = off)
    state_decay_init: float = 0.9   # initial per-channel retention

    # Connectivity
    local_radius: int = 1
    long_range_prob: float = 0.1
    max_long_range: int = 3
    # Seed for the RANDOM TOPOLOGY only (which blocks get small-world shortcut
    # edges). Weight initialisation is seeded separately by the caller via
    # torch.manual_seed; this only fixes the wiring pattern, so two runs of the
    # same config always get the same graph while weights still vary.
    connectivity_seed: int = 42
    # Deep-grid connectivity. Lateral = inside a layer (2D). Vertical = between
    # adjacent layers. Diagonal adds the 4 diagonals within a layer.
    lateral: bool = True        # connect blocks left/right/up/down in a layer
    vertical: bool = True       # connect a block to the same cell in layer L±1
    dense_layers: bool = False  # every cell in L <-> every cell in L+1 (all-to-all
                                # BETWEEN sheets, while staying local inside one).
                                # This is the "cortical areas projecting to areas"
                                # reading of brain connectivity: sheets talk to
                                # sheets densely, but a neuron still only reaches
                                # its own column's neighbours locally -- which is
                                # what cortex actually does, unlike a fully
                                # connected graph.
                                # COST (5x5 grid, 125 regions): 618 -> 5421 edges
                                # (8.8x), edge params 22.9M -> 201.1M on the 50M
                                # config, and the retained exc_list goes 5.2 -> 45.5
                                # GB at batch 32. Feasible at 50M on an 80 GB A100;
                                # NOT feasible at 20B (would need batch reduction).
                                # Requires a fresh run: the edge key set changes, so
                                # an existing checkpoint will not load.
    diagonal: bool = False      # include in-layer diagonals
    feedback: bool = True       # allow L+1 -> L as well as L -> L+1. The cortex
                                # is not feedforward; feedback is what makes this
                                # a recurrent brain-like system rather than a
                                # plain deep stack.
    readout_layer: str = 'last'  # 'last' | 'all' — which layers feed the readout

    # Training
    # Production uses BPE-16k (id 0 = <pad|>). Byte-level is 257; the historical
    # quantum-subsystem head used 32000.
    vocab_size: int = 16384
    max_seq_len: int = 1024
    # embed_dim / num_classes are legacy quantum-subsystem fields, unused by the
    # classical Raslila (token embeddings use hidden_dim).
    embed_dim: int = 512
    num_classes: int = 8

    # Causal language modeling (True = next-token; False = bidirectional)
    causal: bool = True


class EdgeNetwork(nn.Module):
    """
    Edge Network - processes messages between nodes.
    This is the key innovation: edges COMPUTE, not just weight.
    """
    def __init__(self, hidden_dim: int, edge_hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = hidden_dim * 2
        # Honour edge_hidden_dim: hidden capacity, not the node width.
        h = max(8, int(edge_hidden_dim))
        for i in range(max(num_layers, 1)):
            if i == 0:
                layers.append(nn.Linear(in_dim, h))
            elif i == num_layers - 1:
                layers.append(nn.Linear(h, hidden_dim))
            else:
                layers.append(nn.Linear(h, h))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, src_feat: torch.Tensor, tgt_feat: torch.Tensor) -> torch.Tensor:
        """Process message from source to target"""
        combined = torch.cat([src_feat, tgt_feat], dim=-1)
        return self.net(combined)

    def signed(self, src_feat: torch.Tensor, tgt_feat: torch.Tensor):
        """Emit an excitatory AND an inhibitory message (lateral inhibition).

        Real cortex is ~20% inhibitory interneurons; without a subtractive path
        a block can only ever add to its neighbours, never argue against them,
        and binding stays permissive. Splitting the final layer's output in two
        gives every edge a sign, so competition is expressible.
        """
        combined = torch.cat([src_feat, tgt_feat], dim=-1)
        out = self.net(combined)
        return F.relu(out), -F.relu(-out)


class PersistentState(nn.Module):
    """Per-block recurrent state across token positions — the brain's integration.

    A transformer only integrates inside its attention window; between two
    positions it keeps nothing. A cortical neuron accumulates over time. This is
    a gated linear recurrence

        h_t = a * h_{t-1} + (1 - a) * x_t

    with a **learned per-channel** decay `a`. That detail is the point: different
    channels forget at different rates, so the block carries slow topic state and
    fast syntactic state side by side. A single scalar decay could only pick one
    timescale.

    Implemented with a cumulative-product scan so it is O(T) and parallel rather
    than a Python loop over positions (which would dominate step time).
    """

    def __init__(self, hidden_dim: int, state_dim: int,
                 decay_init: float = 0.9):
        super().__init__()
        self.state_dim = int(state_dim)
        self.in_proj = nn.Linear(hidden_dim, state_dim, bias=False)
        self.out_proj = nn.Linear(state_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        # Sigmoid-initialised around `decay_init`, so a fresh model starts with
        # a moderate, well-behaved memory rather than forgetting instantly.
        p = min(max(decay_init, 1e-3), 1 - 1e-3)
        logit = math.log(p / (1 - p))
        self.decay_logit = nn.Parameter(torch.full((self.state_dim,), logit))
        self.norm = RMSNorm(state_dim)

    def decay(self):
        return torch.sigmoid(self.decay_logit).clamp(1e-4, 1 - 1e-4)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        """x: [B,T,D]; state: [B,D] previous hidden state (or None).

        Returns (out [B,T,D], new_state [B,D]).
        """
        if state is None:
            state = x.new_zeros(x.shape[0], self.state_dim)
        u = self.in_proj(x.float())                      # [B,T,S]
        u = self.norm(u)
        a = self.decay().to(u.dtype)                     # [S]

        # h_t = a^t * h_0 + sum_{s<=t} a^(t-s) u_s
        # Computed with a cumulative product / cumulative sum (parallel scan).
        powers = torch.cumprod(
            torch.cat([a.view(1, 1, -1).expand(x.shape[0], 1, -1),
                       a.view(1, 1, -1).expand(x.shape[0], x.shape[1] - 1, -1)],
                      dim=1),
            dim=1,
        )                                               # [B,T,S] = a^t
        scaled = u / powers.clamp_min(1e-12)
        cum = torch.cumsum(scaled, dim=1)
        h = cum * powers                               # [B,T,S]
        # Drop the contribution of the incoming state from the first step only,
        # so the recurrence is exactly the linear form above.
        h = h + powers[:, :1] * state.unsqueeze(1).to(h.dtype)
        new_state = h[:, -1]
        return self.out_proj(h.to(x.dtype)), new_state


class RMSNorm(nn.Module):
    """Root-mean-square normalization (no mean centering). Matches LayerNorm
    quality in pre-norm transformers at lower cost (LLaMA/Mistral/Qwen recipe)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight).type_as(x)


def _rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0,
                     device=None, dtype=torch.float32):
    """Precompute RoPE cos/sin tables of shape [T, head_dim]."""
    assert head_dim % 2 == 0, "RoPE needs even head_dim"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(q, k, cos, sin):
    """Apply rotary embedding. q,k: [B, H, T, Dh]; cos,sin: [T, Dh]."""
    cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional per-head QK-norm and
    RoPE. Mathematically identical to MHA when both are off."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float,
                 qk_norm: bool = True, rope_theta: float = 10000.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must divide num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim % 2 == 0, "RoPE needs even head_dim"
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.rope_theta = rope_theta

    def forward(self, x, rope_cos=None, rope_sin=None, causal=True):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rope(q, k, rope_cos[:T], rope_sin[:T])
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    """Gated SiLU feed-forward: (SiLU(xW_gate) * xW_up) W_down.
    Consistently beats parameter-matched GELU MLPs (PaLM/LLaMA/Mistral)."""
    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        # 3 matrices vs 2: shrink width ~2/3 for param parity with a 4d MLP.
        inner = max(32, round(2 * ff_dim / 3 / 32) * 32)
        self.gate = nn.Linear(hidden_dim, inner)
        self.up = nn.Linear(hidden_dim, inner)
        self.down = nn.Linear(inner, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


def _gelu_mlp(hidden_dim: int, ff_dim: int, dropout: float):
    return nn.Sequential(
        nn.Linear(hidden_dim, ff_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(ff_dim, hidden_dim),
        nn.Dropout(0.1),
    )


class RegionBinder(nn.Module):
    """Binding / blending layer: composes region states into an emergent set.

    The binding problem (arXiv 2012.05208) argues neural nets fail because they
    cannot dynamically BIND separately-represented information. A weighted mean
    over regions is a LINEAR function of the region set, so no composite can
    ever emerge: "cat" and "on the mat" are averaged into mush.

    This module performs genuine composition (Tensor-Product style, Smolensky):
    each region state is bound MULTIPLICATIVELY to learned role vectors,
    producing features that are products of two region features. The resulting
    N*K (and optionally neighbour) composites are kept as a SET and handed to
    the readout, which attends over the set instead of collapsing it.

    Why the 3D grid helps: every region already carries a persistent, separate
    state stream, so we have the separated representations binding requires.
    A 1D transformer fuses each position into one vector immediately and
    structurally cannot provide this.
    """
    def __init__(self, hidden_dim: int, n_roles: int = 4,
                 bind_local: bool = False):
        super().__init__()
        self.n_roles = max(n_roles, 0)
        self.bind_local = bind_local
        if self.n_roles > 0:
            # Per-role modulation, applied before the multiplicative bind so each
            # composite is role-specific rather than a plain self-product.
            self.role_proj = nn.Linear(hidden_dim, self.n_roles * hidden_dim)
        if bind_local:
            # Second binder over grid neighbours: spatial-relation composites.
            self.local_proj = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, node_states, adjacency=None):
        """node_states: [B, T, N, D] -> (bound set [B, T, M, D], None).

        M = N*K composites (role binding), plus N composites when bind_local is
        on. Returns (None, None) only when BOTH binders are disabled.
        """
        if self.n_roles <= 0 and not self.bind_local:
            return None, None
        B, T, N, D = node_states.shape
        comps = []
        if self.n_roles > 0:
            # Role modulation: [B,T,N,K*D] -> [B,T,N,K,D]
            mod = self.role_proj(node_states).view(B, T, N, self.n_roles, D)
            # Multiplicative bind (FiLM/TPR-style): product of two learned
            # features. [B,T,N,1,D] * [B,T,N,K,D] -> [B,T,N,K,D]
            role_bound = node_states.unsqueeze(3) * mod
            comps.append(role_bound.reshape(B, T, N * self.n_roles, D))

        if self.bind_local and adjacency is not None:
            # Neighbour composites: bind each region with the mean of its grid
            # neighbours, capturing spatial relations across the 3D lattice.
            deg = adjacency.sum(0).clamp_min(1.0).view(1, 1, N, 1)
            nbr_mean = torch.einsum('ij,btjd->btid', adjacency.to(node_states.dtype),
                                    node_states) / deg
            pair = self.local_proj(node_states)
            a, b = pair[..., :D], pair[..., D:]
            # Product of a region feature and its neighbourhood's feature.
            local_bound = (a * b + b * nbr_mean)
            comps.append(local_bound)

        bound = torch.cat(comps, dim=2)
        return bound, None


class TransformerNode(nn.Module):
    """Transformer node at a 3D grid position: `num_blocks` stacked pre-norm
    blocks (RMSNorm + RoPE attention + SwiGLU by default). With all modern
    flags off it reproduces the legacy GPT-2-era block for ablations."""
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float,
                 num_blocks: int = 1, use_rope: bool = True,
                 use_swiglu: bool = True, use_rmsnorm: bool = True,
                 qk_norm: bool = True, rope_theta: float = 10000.0,
                 lns: bool = False):
        super().__init__()
        self.use_rope = use_rope
        self.lns = lns
        self.num_blocks = max(num_blocks, 1)
        blocks = []
        for _ in range(self.num_blocks):
            blocks.append(CausalSelfAttention(hidden_dim, num_heads, dropout,
                                             qk_norm=qk_norm, rope_theta=rope_theta))
            blocks.append(RMSNorm(hidden_dim) if use_rmsnorm else nn.LayerNorm(hidden_dim))
            blocks.append(SwiGLU(hidden_dim, ff_dim, dropout) if use_swiglu
                          else _gelu_mlp(hidden_dim, ff_dim, dropout))
            blocks.append(RMSNorm(hidden_dim) if use_rmsnorm else nn.LayerNorm(hidden_dim))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, rope_cos=None, rope_sin=None, causal=True):
        for b in range(self.num_blocks):
            # LNS: scale norm outputs by 1/sqrt(depth) so deep blocks keep
            # contributing instead of collapsing to identity (curse of depth).
            s = 1.0 / math.sqrt(b + 1) if self.lns else 1.0
            attn = self.blocks[4 * b]
            norm1 = self.blocks[4 * b + 1]
            ffn = self.blocks[4 * b + 2]
            norm2 = self.blocks[4 * b + 3]
            # Pre-norm: normalize before the sublayer, clean residual path.
            x = x + attn(norm1(x) * s,
                         rope_cos=rope_cos if self.use_rope else None,
                         rope_sin=rope_sin if self.use_rope else None,
                         causal=causal)
            x = x + ffn(norm2(x) * s)
        return x


class Raslila(nn.Module):
    """
    Raslila - Brain-Inspired Transformer Network.

    Nodes: Transformer blocks at (x, y, z) positions
    Edges: Neural networks that process messages between nodes
    3D connectivity: local + long-range (small-world)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim_x, self.dim_y, self.dim_z = config.dim_x, config.dim_y, config.dim_z
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len

        self.tokens = ["grief", "logic", "memory", "clarity",
                       "longing", "reason", "dream", "empathy"]

        # ---- Layout ----
        if getattr(config, 'layout', 'deep_grid') == 'flat3d':
            self.layout = 'flat3d'
            # A flat 3D lattice is a single layer of dim_x*dim_y*dim_z blocks.
            self.n_layers = 1
            self.grid_h = max(1, int(config.dim_x) * int(config.dim_y))
            self.grid_w = max(1, int(config.dim_z))
        else:
            self.layout = 'deep_grid'
            self.n_layers = max(1, int(config.depth_layers))
            self.grid_h = max(1, int(config.grid_h))
            self.grid_w = max(1, int(config.grid_w))
        self.per_layer = self.grid_h * self.grid_w
        self.n_nodes = self.n_layers * self.per_layer

        self.register_buffer('node_positions', self._create_positions())
        # index of each block within its layer, per layer: layer -> [block ids]
        self.register_buffer(
            'layer_index',
            torch.tensor([[l * self.per_layer + i for i in range(self.per_layer)]
                          for l in range(self.n_layers)], dtype=torch.long))

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        # GPT-2-style small init: with a tied output head, logits are h·E, so
        # default N(0,1) rows (norm ~sqrt(D)) would saturate softmax at init.
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        # Learned absolute positions are only needed when RoPE is off; skip the
        # table entirely otherwise (saves max_seq_len x hidden_dim dead params).
        if not config.use_rope:
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
            nn.init.normal_(self.pos_embedding.weight, std=0.02)

        # Node identity: give each brain region a location-aware role embedding
        # so nodes can specialize (logical / memory / spatial regions...).
        # Final layer zero-initialized (residual-branch style): roles start at
        # zero and grow only as needed, keeping early activations balanced.
        self.node_role_net = nn.Sequential(
            nn.Linear(3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        nn.init.zeros_(self.node_role_net[2].weight)
        nn.init.zeros_(self.node_role_net[2].bias)

        self.nodes = nn.ModuleList([
            TransformerNode(
                config.hidden_dim,
                config.num_heads,
                config.ff_dim,
                config.dropout,
                num_blocks=config.node_depth,
                use_rope=config.use_rope,
                use_swiglu=config.use_swiglu,
                use_rmsnorm=config.use_rmsnorm,
                qk_norm=config.qk_norm,
                rope_theta=config.rope_theta,
                lns=config.lns,
            ) for _ in range(self.n_nodes)
        ])

        # Persistent per-block state across token positions (the brain integrates
        # over time; an attention window does not). Zero width = off.
        self.n_blocks_with_state = max(0, int(getattr(config, 'state_dim', 0)))
        if self.n_blocks_with_state:
            self.states = nn.ModuleList([
                PersistentState(config.hidden_dim, self.n_blocks_with_state,
                                decay_init=getattr(config, 'state_decay_init', 0.9))
                for _ in range(self.n_nodes)
            ])
        else:
            self.states = nn.ModuleList()
        # Live state is inference-time only (training always starts from zeros,
        # exactly like a KV cache), so it is deliberately NOT in state_dict.
        self.persistent_state = [None] * self.n_nodes
        self.inhibit_beta = float(getattr(config, 'inhibit_beta', 1.0))

        self.edge_networks = nn.ModuleDict()
        # Buffer, not a plain attribute: must follow .to(device) or the binding
        # path (tensor math with adjacency) breaks on GPU, and it belongs in
        # state_dict so checkpoints are self-contained.
        self.register_buffer('adjacency', self._build_connectivity())
        self._create_edge_networks()
        self._build_in_edge_lists()

        # Activation checkpointing is applied at the CALL SITE in forward, not by
        # wrapping modules here. Two reasons: (1) FSDP wraps TransformerNode itself,
        # and a checkpoint_wrapper around it would double-wrap and break the
        # all-gather; (2) the node is the right recomputation unit -- it is the
        # expensive part of the hot loop, and checkpointing the whole forward would
        # recompute every sweep, not just one region.
        self.ckpt_nodes = bool(getattr(config, 'activation_checkpoint', False))

        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)
        self.input_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        if config.tie_weights:
            # Share the token table with the output head (GPT-2 style).
            self.output_proj.weight = self.token_embedding.weight

        # Learned node readout: weigh which region's opinion matters per token.
        self.node_readout = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

        # Pre-norm nets need a final normalization before the head: the
        # residual stream grows with depth, and without this the logits
        # explode (loss ~100 at init). Identity in legacy post-norm mode.
        self.final_norm = RMSNorm(config.hidden_dim) if config.use_rmsnorm else nn.Identity()

        # Shared neuromodulation gate for incoming edge messages.
        # Only built when actually enabled: the forward pass reads it solely
        # under `if self.config.gated_messages`, so with gating off these
        # parameters were registered but never used. Single-GPU runs never
        # noticed (autograd quietly leaves an unused param's grad as None),
        # but DistributedDataParallel treats a None grad as a hard error:
        #   "Expected to have finished reduction in the prior iteration ...
        #    Parameter indices which did not receive grad for rank N"
        # which made every multi-GPU run of the default config (gating off)
        # crash on the first step. Allocating only what is used is the real
        # fix; `find_unused_parameters=True` would have papered over it and
        # added per-step overhead forever.
        if config.gated_messages:
            self.msg_gate = nn.Linear(config.hidden_dim, 1)
            nn.init.zeros_(self.msg_gate.weight)
            nn.init.constant_(self.msg_gate.bias, 2.0)
        else:
            self.msg_gate = None

        # Multi-token prediction heads: predict t+2..t+K from the final
        # representation (hierarchical forecasting; auxiliary loss only).
        self.mtp_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.vocab_size)
            for _ in range(max(config.mtp_steps, 0))
        ])

        # Global workspace: K shared slots all regions compete to write to
        # and read back (bandwidth-limited broadcast; GWT bottleneck).
        self.n_ws_slots = max(config.workspace_slots, 0)
        if self.n_ws_slots > 0:
            self.ws_query = nn.Parameter(
                torch.randn(self.n_ws_slots, config.hidden_dim) * 0.02)
            self.ws_out = nn.Linear(config.hidden_dim, config.hidden_dim)
            nn.init.zeros_(self.ws_out.weight)
            nn.init.zeros_(self.ws_out.bias)

        # Binding / blending: composes regions into a set of emergent composites
        # so the readout can depend on "region A AND region B", not their mean.
        self.binder = RegionBinder(config.hidden_dim,
                                  n_roles=config.bind_roles,
                                  bind_local=config.bind_local)
        if config.bind_roles > 0 or config.bind_local:
            # Attention over the composite set, queried by the region summary.
            # Zero-init output => starts identical to the non-bound readout.
            self.bind_query = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.bind_key = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.bind_value = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.bind_out = nn.Linear(config.hidden_dim, config.hidden_dim)
            nn.init.zeros_(self.bind_out.weight)
            nn.init.zeros_(self.bind_out.bias)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def _create_positions(self):
        """3D position (layer, row, col) for every block, so the role embedding
        knows where each block sits in the deep grid."""
        positions = []
        if self.layout == 'flat3d':
            for z in range(self.config.dim_z):
                for y in range(self.config.dim_y):
                    for x in range(self.config.dim_x):
                        positions.append([x, y, z])
        else:
            for l in range(self.n_layers):
                for r in range(self.grid_h):
                    for c in range(self.grid_w):
                        positions.append([c, r, l])
        return torch.tensor(positions, dtype=torch.float32)

    def _build_connectivity(self):
        """Structured deep-grid connectivity.

        Inside a layer: 2D lateral neighbours (left/right/up/down, optionally
        diagonals) — cortical columns talking within a sheet.
        Between layers: the same cell in adjacent layers (feedforward), and
        optionally the reverse direction too (feedback). The cortex is not a
        feedforward stack, so `feedback` is on by default.
        """
        adj = torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.bool)
        if self.layout == 'flat3d':
            return self._build_flat3d_connectivity()

        cfg = self.config

        def bid(l, r, c):
            return l * self.per_layer + r * self.grid_w + c

        for l in range(self.n_layers):
            # lateral, inside the layer
            for r in range(self.grid_h):
                for c in range(self.grid_w):
                    i = bid(l, r, c)
                    if cfg.lateral:
                        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w:
                                j = bid(l, nr, nc)
                                adj[j, i] = True   # neighbour sends INTO i
                    if cfg.diagonal:
                        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w:
                                adj[bid(l, nr, nc), i] = True
            # vertical, between layers
            if cfg.vertical:
                if getattr(cfg, 'dense_layers', False):
                    # Every cell in L connects to every cell in L+1 (and back,
                    # via feedback). Sheets are fully coupled to each other,
                    # but connectivity WITHIN a sheet stays local -- which is
                    # the direction cortex actually goes: sparse local wiring,
                    # dense long-range projections between areas.
                    for r1 in range(self.grid_h):
                        for c1 in range(self.grid_w):
                            for r2 in range(self.grid_h):
                                for c2 in range(self.grid_w):
                                    if l + 1 < self.n_layers:
                                        adj[bid(l, r1, c1),
                                            bid(l + 1, r2, c2)] = True
                                    if cfg.feedback and l - 1 >= 0:
                                        adj[bid(l, r1, c1),
                                            bid(l - 1, r2, c2)] = True
                else:
                    for r in range(self.grid_h):
                        for c in range(self.grid_w):
                            if l + 1 < self.n_layers:
                                adj[bid(l, r, c), bid(l + 1, r, c)] = True
                            if cfg.feedback and l - 1 >= 0:
                                adj[bid(l, r, c), bid(l - 1, r, c)] = True

        # Sparse long-range shortcuts (small-world), never duplicating a link.
        # A LOCAL Random instance, not random.seed(...): calling the module-level
        # seed here silently reset the caller's global RNG every time a model was
        # constructed, which made training stochasticity (replay sampling, dropout
        # order) depend on when the model happened to be built.
        rng = random.Random(cfg.connectivity_seed)
        for i in range(self.n_nodes):
            if rng.random() >= cfg.long_range_prob:
                continue
            candidates = [j for j in range(self.n_nodes)
                          if j != i and not adj[j, i]]
            if not candidates:
                continue
            k = min(rng.randint(1, max(cfg.max_long_range, 1)), len(candidates))
            for j in rng.sample(candidates, k):
                adj[j, i] = True
        return adj

    def _build_flat3d_connectivity(self):
        """Legacy flat 3D lattice (local radius + long-range)."""
        dx_, dy_, dz_ = self.config.dim_x, self.config.dim_y, self.config.dim_z
        adj = torch.zeros(dx_ * dy_ * dz_, dx_ * dy_ * dz_, dtype=torch.bool)
        r = max(self.config.local_radius, 1)

        def idx(x, y, z):
            return z * (dx_ * dy_) + y * dx_ + x

        for z in range(dz_):
            for y in range(dy_):
                for x in range(dx_):
                    i = idx(x, y, z)
                    for dx in range(-r, r + 1):
                        for dy in range(-r, r + 1):
                            for dz in range(-r, r + 1):
                                if dx == dy == dz == 0:
                                    continue
                                if abs(dx) + abs(dy) + abs(dz) > r:
                                    continue
                                nx, ny, nz = x + dx, y + dy, z + dz
                                if 0 <= nx < dx_ and 0 <= ny < dy_ and 0 <= nz < dz_:
                                    adj[idx(nx, ny, nz), i] = True
        rng = random.Random(self.config.connectivity_seed)
        for i in range(self.n_nodes):
            if rng.random() >= self.config.long_range_prob:
                continue
            candidates = [j for j in range(self.n_nodes) if j != i and not adj[j, i]]
            if candidates:
                k = min(rng.randint(1, self.config.max_long_range), len(candidates))
                for j in rng.sample(candidates, k):
                    adj[j, i] = True
        return adj

    @torch.no_grad()
    def grow_regions(self, n_new, optimizer=None, verbose=True):
        """Cortical growth: append `n_new` regions to the grid, keeping every
        already-trained region untouched.

        Each newborn region is CLONED from a random existing region rather than
        randomly initialised. That is the "integration-friendly birth" finding
        from the structural-plasticity literature: fresh random units are
        forward-active but backward-starved and destabilise training, whereas
        cloned ones arrive pre-integrated. The clone's input is then mixed down
        so it does not exactly duplicate its parent (diversity without noise).

        All regions keep participating afterwards -- growth is dense, not routed.
        """
        n_new = int(max(0, n_new))
        if n_new == 0:
            return 0
        old_n = self.n_nodes
        if old_n + n_new > self.config.grow_max_nodes:
            n_new = max(0, self.config.grow_max_nodes - old_n)
            if n_new == 0:
                return 0
        new_n = old_n + n_new
        device = self.node_positions.device

        # 1. Positions: extend the lattice outward from the current grid so the
        #    new regions have sensible 3D identities.
        pos = self.node_positions.detach().cpu()
        if self.layout == 'deep_grid':
            # A grown block is a whole new cortical sheet cell; keep its layer
            # coordinate one past the deepest existing layer.
            extra = [[c, r, int(pos[:, 2].max()) + 1]
                     for r in range(self.grid_h) for c in range(self.grid_w)]
            extra = extra[:n_new]
            n_new = len(extra)
            if n_new == 0:
                return 0
            new_n = old_n + n_new
        else:
            dim_x, dim_y, dim_z = self.dim_x, self.dim_y, self.dim_z
            extra = []
            k = 0
            guard = 0
            while len(extra) < n_new and guard < n_new * 200:
                guard += 1
                cand = (k % dim_x, (k // dim_x) % dim_y, (k // (dim_x * dim_y)) % dim_z)
                k += 1
                if k > dim_x * dim_y * dim_z:
                    # lattice full: continue along +z like a deeper cortex layer
                    cand = (cand[0], cand[1], dim_z + len(extra))
                if any(torch.equal(pos.new_tensor(cand, dtype=torch.float32), p) for p in pos):
                    continue
                extra.append(cand)
        while len(extra) < n_new and self.layout != 'deep_grid':  # numeric fallback
            extra.append((0, 0, self.dim_z + len(extra)))
        new_pos = torch.cat([pos, torch.tensor(extra, dtype=torch.float32)], dim=0)
        self.node_positions = new_pos.to(device)

        # 2. Grow the block count, keeping the old indices intact.
        self.n_nodes = new_n
        if self.layout == 'deep_grid':
            # Each grown cell starts a new cortical layer.
            self.n_layers += 1
            new_layer = torch.tensor(
                [[l * self.per_layer + i for i in range(self.per_layer)]
                 for l in range(self.n_layers)], dtype=torch.long,
                device=self.layer_index.device)
            self.layer_index = new_layer
        else:
            self.dim_z = max(self.dim_z, max(p[2] for p in extra) + 1)

        # 3. New regions: clone a random existing one (pre-integrated birth).
        for _ in range(n_new):
            src_idx = random.randrange(old_n)
            clone = TransformerNode(
                self.config.hidden_dim, self.config.num_heads, self.config.ff_dim,
                self.config.dropout, num_blocks=self.config.node_depth,
                use_rope=self.config.use_rope, use_swiglu=self.config.use_swiglu,
                use_rmsnorm=self.config.use_rmsnorm, qk_norm=self.config.qk_norm,
                rope_theta=self.config.rope_theta, lns=self.config.lns,
            ).to(device)
            clone.load_state_dict(self.nodes[src_idx].state_dict())
            # Decouple the clone slightly so it is not an exact duplicate.
            for p in clone.parameters():
                p.add_(torch.randn_like(p) * 0.02)
            self.nodes.append(clone)
            # A newborn block also needs its own memory, or it starts blind.
            if self.n_blocks_with_state:
                st = PersistentState(
                    self.config.hidden_dim, self.n_blocks_with_state,
                    decay_init=getattr(self.config, 'state_decay_init', 0.9)).to(device)
                st.load_state_dict(self.states[src_idx].state_dict())
                for p in st.parameters():
                    p.add_(torch.randn_like(p) * 0.02)
                self.states.append(st)
        if self.n_blocks_with_state:
            self.persistent_state = self.persistent_state + [None] * n_new

        # 4. Connectivity: new region joins, wired to its 3D nearest neighbours
        #    plus a couple of long-range links (it recruits into the network).
        adj = torch.zeros(new_n, new_n, dtype=torch.bool, device=device)
        adj[:old_n, :old_n] = self.adjacency
        all_pos = self.node_positions.detach().cpu()
        for i in range(old_n, new_n):
            d = (all_pos - all_pos[i]).abs().sum(-1)
            d[i] = float('inf')
            k_near = min(4, new_n - 1)
            for j in d.topk(k_near, largest=False).indices.tolist():
                adj[i, j] = True
                adj[j, i] = True
        self.adjacency = adj
        # The wiring changed, so the cached incoming-edge lists are stale.
        self._build_in_edge_lists()

        # 5. Edge networks for the new connections (reuse the old ones).
        for i in range(new_n):
            for j in range(new_n):
                if adj[i, j] and (f"{i}_{j}" not in self.edge_networks):
                    self.edge_networks[f"{i}_{j}"] = EdgeNetwork(
                        self.config.hidden_dim, self.config.edge_hidden_dim,
                        self.config.edge_layers, self.config.dropout,
                    ).to(device)

        # 6. Register the newborn parameters with the optimizer so they train.
        if optimizer is not None:
            known = {id(p) for g in optimizer.param_groups for p in g['params']}
            fresh = [p for p in self.parameters() if id(p) not in known]
            if fresh:
                optimizer.add_param_group(
                    {'params': fresh,
                     'lr': optimizer.param_groups[0]['lr']})

        if verbose:
            print('  [growth] regions {} -> {} (+{}, {:.1f}M params total)'.format(
                old_n, new_n, n_new,
                sum(p.numel() for p in self.parameters()) / 1e6), flush=True)
        return n_new

    def reset_state(self):
        """Clear persistent per-block state (call between independent sequences).

        Mirrors the KV cache: state is inference-time context, not model weights,
        so it is never checkpointed and must be cleared between documents.
        """
        self.persistent_state = [None] * self.n_nodes

    def experience(self):
        """Read out what the network currently 'knows' as a memory bank.

        This is the mechanism behind `replay_prob` in the training script and it
        is also what makes the model's memory inspectable rather than an
        emergent accident: the hidden state after seeing text IS the memory, so
        it can be saved, replayed and inspected directly.

        Returns a list of [state_dim] tensors, one per block, all on CPU.
        """
        if not self.n_blocks_with_state:
            return []
        out = []
        for s in self.persistent_state:
            if s is None:
                continue
            out.append(s.detach().float().cpu())
        return out

    def absorb_experience(self, memories):
        """Load block states previously returned by `experience()`.

        Used to seed a fresh document with what earlier documents taught, which
        is the model-side half of learning from experience. Mismatched or empty
        input is ignored rather than raising, so a checkpoint written before
        persistent state existed still loads.
        """
        if not self.n_blocks_with_state or not memories:
            return 0
        sdim = self.n_blocks_with_state
        loaded = 0
        for i, mem in enumerate(memories[:self.n_nodes]):
            t = torch.as_tensor(mem).float()
            if t.numel() != sdim:
                continue
            # Persisted memory is stored per-block WITHOUT the batch dimension
            # ([S]); add it back here. Forgetting this made the forward
            # pass concatenate a [1,S] tensor against a [B,T,D] one.
            self.persistent_state[i] = t.reshape(1, -1).to(
                next(self.parameters()).device)
            loaded += 1
        return loaded

    def state_batch_ok(self, batch: int) -> bool:
        """True only if cached state is usable for a batch of this size.

        State tensors carry a batch dimension. Reusing state across a different
        batch size is a shape error, and silently broadcasting it would corrupt
        every sequence, so callers must check first.
        """
        if not self.n_blocks_with_state:
            return True
        for s in self.persistent_state:
            if s is None:
                return False
            if s.shape[0] != batch:
                return False
        return True

    def _create_edge_networks(self):
        """Create edge networks for each connection."""
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.adjacency[i, j]:
                    self.edge_networks[f"{i}_{j}"] = EdgeNetwork(
                        self.config.hidden_dim,
                        self.config.edge_hidden_dim,
                        self.config.edge_layers,
                        self.config.dropout,
                    )

    def _build_in_edge_lists(self):
        """Precompute, per node, the list of nodes with an edge INTO it.

        The forward pass used to enumerate incoming neighbours with
        `torch.where(self.adjacency[:, node_idx])[0].tolist()`. That reads a
        tensor on the compute device and copies the result back to the host, so
        it is a synchronisation point -- and it ran once per block per sweep.
        At 125 blocks x depth_sweeps=2 that is 250 host-device round trips
        inside a single forward pass, which is exactly the pattern XLA/TPU
        compiles worst and which made this model unviable on a TPU despite
        having plenty of compute available.

        The wiring is fixed for a given config, so the index lists are built
        once here, on the host, and never recomputed in the hot loop. Call this
        again after anything that changes `adjacency` (e.g. grow_regions).
        """
        self.in_edges = [
            torch.where(self.adjacency[:, i])[0].tolist() for i in range(self.n_nodes)
        ]

    def forward(self, input_ids, return_activations=False, return_entanglement=False,
                keep_state=False):
        """
        Forward pass through the Raslila network.

        Args:
            input_ids: [B, T] token ids OR string prompt
            return_activations: whether to return node activations
            return_entanglement: return inter-node coherence (classical analogue
                of entanglement: pairwise correlation of node activations)
            keep_state: carry each block's persistent state into the NEXT call,
                so memory survives across sequences (inference only, exactly
                like a KV cache). Defaults False, which is what training needs --
                every training batch must start from a clean state, otherwise
                gradients leak between unrelated examples.

        Returns:
            dict with keys: logits, final_repr, activations (optional),
                            token/confidence (string-prompt path only),
                            entanglement (optional)
        """
        from_string = isinstance(input_ids, str)
        if from_string:
            tokens = input_ids.lower().split()
            token_ids = []
            for tok in tokens:
                if tok in self.tokens:
                    token_ids.append(self.tokens.index(tok))
                else:
                    token_ids.append(0)
            if not token_ids:
                token_ids = [0]
            input_ids = torch.tensor([token_ids], dtype=torch.long,
                                     device=next(self.parameters()).device)

        B, T = input_ids.shape
        device = input_ids.device

        # Persistent state is inference context, not a weight. Only continue from
        # it when the caller explicitly asks AND the cached shapes actually match
        # this call; otherwise start clean. Training always starts clean, or
        # gradients would leak from one batch into the next.
        if not keep_state or not self.state_batch_ok(B):
            self.reset_state()

        tok_emb = self.token_embedding(input_ids)
        if self.config.use_rope:
            # Positions ride on Q/K rotation inside attention; no table needed.
            x = tok_emb
            rope_cos, rope_sin = build_rope_cache(
                T, self.hidden_dim // self.config.num_heads,
                theta=self.config.rope_theta, device=device,
            )
        else:
            pos_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
            pos_emb = self.pos_embedding(pos_ids)
            x = tok_emb + pos_emb
            rope_cos, rope_sin = None, None

        x = self.input_proj(x)

        # Give each node a location-aware role identity so nodes specialize.
        roles = self.node_role_net(self.node_positions)  # [N, D]
        node_states = x.unsqueeze(2) + roles.view(1, 1, self.n_nodes, self.hidden_dim)

        node_activations = []
        prev_mag = None  # per-node message magnitude (surprise) from last round

        # Which blocks participate in the readout. 'last' = only the deepest
        # layer (like a deep net's output stage); 'all' = every block.
        if self.layout == 'deep_grid' and getattr(self.config, 'readout_layer', 'last') == 'last':
            readout_ids = list(range((self.n_layers - 1) * self.per_layer,
                                     self.n_layers * self.per_layer))
        else:
            readout_ids = list(range(self.n_nodes))

        # BIDIRECTIONAL DEPTH SWEEPS. The cortex is not a feedforward stack: a
        # hypothesis propagates up and the higher layers send it back down. A
        # single bottom-up pass cannot do that -- the lower layer is written
        # before the upper one exists to talk back -- so `depth_sweeps` alternates
        # the traversal direction, which is what finally uses the feedback edges.
        n_sweeps = max(1, int(self.config.depth_sweeps))
        for sweep in range(n_sweeps):
            # Order matters for correctness, not just cost. The readout takes the
            # DEEPEST layer, so the final sweep must end at that layer --
            # otherwise the readout reads a state that was written before the
            # feedback refinement reached the layers below it, and the whole
            # point of the downward sweep is thrown away.
            # The readout takes the DEEPEST layer, so the FINAL sweep must run
            # upward and finish there. Picking a fixed starting polarity and
            # alternating does not guarantee that: with `sweep % 2 == 1` the
            # sequence for 2 sweeps was [down, up] (accidentally correct) but
            # for 1 sweep it was [down] -- the readout layer was computed FIRST,
            # and every lower layer was then computed and thrown away, because
            # readout_layer='last' reads only the deepest sheet. Measured: with
            # depth_sweeps=1, zeroing the transformer weights of every block
            # outside the readout layer changed the logits by exactly 0.0. The
            # lower 4/5 of the model was dead computation and received no
            # gradient at all.
            #
            # Choose the starting polarity so that alternation ENDS on an
            # upward sweep for any sweep count, keeping the feedback traversal
            # and the readout-ordering requirement consistent.
            ascending = (sweep % 2) == ((n_sweeps - 1) % 2)
            layer_order = (range(self.n_layers) if ascending
                           else reversed(range(self.n_layers)))
            for layer in layer_order:
                layer_ids = (list(range(layer * self.per_layer,
                                        (layer + 1) * self.per_layer))
                             if self.layout == 'deep_grid'
                             else list(range(self.n_nodes)))

                for round_idx in range(max(1, self.config.num_layers)):
                        # Visitation order: fixed sweep, or most-surprised-first.
                        if self.config.priority_order and prev_mag is not None:
                            order = [i for i in sorted(layer_ids,
                                                       key=lambda j: -float(prev_mag[j]))]
                        else:
                            order = layer_ids

                        # Live slot list: reads see fresh values written earlier in
                        # this sweep (multidirectional cascade). Frozen mode reads the
                        # round-start snapshot while writes accumulate separately.
                        live = self.config.immediate_updates
                        read = [node_states[:, :, i] for i in range(self.n_nodes)]
                        write = read if live else list(read)
                        src = write if live else read
                        mag = torch.zeros(self.n_nodes, device=device)

                        signed = self.config.inhibit
                        for node_idx in order:
                            node = self.nodes[node_idx]
                            if self.ckpt_nodes and self.training:
                                # Non-reentrant checkpoint: store this region's
                                # activations, recompute them in backward. At 20B
                                # this is the difference between OOM and fitting --
                                # stored activations scale with regions*node_depth,
                                # recomputed ones with just the readout path.
                                node_out = checkpoint(
                                    node, src[node_idx], rope_cos, rope_sin,
                                    self.config.causal, use_reentrant=False)
                            else:
                                node_out = node(
                                    src[node_idx], rope_cos=rope_cos,
                                    rope_sin=rope_sin,
                                    causal=self.config.causal,
                                )

                            # Per-block persistent state: integration across token
                            # positions, which attention alone cannot do.
                            if self.n_blocks_with_state:
                                mem, self.persistent_state[node_idx] = self.states[node_idx](
                                    node_out, self.persistent_state[node_idx])
                                node_out = node_out + mem

                            exc_list = []
                            inh_list = []
                            for src_t in self.in_edges[node_idx]:
                                edge_key = "{}_{}".format(src_t, node_idx)
                                if edge_key not in self.edge_networks:
                                    continue
                                src_feat = src[src_t]
                                if signed:
                                    e, i_ = self.edge_networks[edge_key].signed(
                                        src_feat, src[node_idx])
                                    # alpha/beta interference applies to the
                                    # excitatory part; the inhibitory part is pure.
                                    e = self.alpha * src_feat + self.beta * src[node_idx] + e
                                    exc_list.append(e)
                                    inh_list.append(i_ * self.inhibit_beta)
                                else:
                                    msg = self.edge_networks[edge_key](src_feat, src[node_idx])
                                    msg = self.alpha * src_feat + self.beta * src[node_idx] + msg
                                    exc_list.append(msg)

                            if exc_list:
                                aggregated = torch.stack(exc_list, dim=0).mean(dim=0)
                                if signed and inh_list:
                                    aggregated = aggregated - torch.stack(
                                        inh_list, dim=0).mean(dim=0)
                                mag[node_idx] = aggregated.detach().abs().mean()
                                if self.config.gated_messages:
                                    g = torch.sigmoid(self.msg_gate(node_out))
                                    node_out = node_out + g * aggregated
                                else:
                                    node_out = node_out + aggregated

                            write[node_idx] = node_out

                        node_states = torch.stack(write, dim=2)

                        # Global workspace: regions compete to write into shared slots
                        # (softmax over nodes = competition), then read the broadcast
                        # back. Linear-cost global coordination (GWT bottleneck).
                        if self.n_ws_slots > 0:
                            w_scores = torch.einsum('sd,btnd->btsn', self.ws_query, node_states)
                            w = torch.softmax(w_scores, dim=-1)
                            slots = torch.einsum('btsn,btnd->btsd', w, node_states)
                            r_scores = torch.einsum('btnd,btsd->btns', node_states, slots)
                            r = torch.softmax(r_scores, dim=-1)
                            back = torch.einsum('btns,btsd->btnd', r, slots)
                            node_states = node_states + self.ws_out(back)

                        prev_mag = mag

                        # Settle-to-convergence: stop exchanging once traffic dies down.
                        if (self.config.settle_thresh > 0
                                and float(mag.max()) < self.config.settle_thresh):
                            break

                if return_activations:
                    node_activations.append(node_states.clone())

        # Readout operates on the selected blocks only.
        readout_states = node_states[:, :, readout_ids, :]
        n_read = len(readout_ids)

        # Learned readout: weight each block region's contribution per token.
        node_weights = torch.softmax(self.node_readout(readout_states).squeeze(-1), dim=2)
        if self.config.readout_topk > 0:
            k = min(self.config.readout_topk, n_read)
            topv, topi = torch.topk(node_weights, k, dim=2)
            sparse = torch.zeros_like(node_weights).scatter_(2, topi, topv)
            node_weights = sparse / sparse.sum(dim=2, keepdim=True).clamp_min(1e-8)
        summary = (readout_states * node_weights.unsqueeze(-1)).sum(dim=2)

        # Binding readout: if enabled, compose the region set into emergent
        # composites and read from THAT set, instead of collapsing to a mean.
        bound, _ = self.binder(readout_states, self.adjacency[readout_ids][:, readout_ids])
        if bound is not None:
            # A query derived from the region summary attends over the bound
            # composites: the prediction can depend on specific pairings
            # ("A bound to role-k") rather than a linear blend of regions.
            q = self.bind_query(summary).unsqueeze(2)          # [B,T,1,D]
            k = self.bind_key(bound)                             # [B,T,M,D]
            v = self.bind_value(bound)                           # [B,T,M,D]
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            blended = self.bind_out(y.squeeze(2)) + summary      # residual: never
            # lose the plain path, so binding can only add signal.
            final_repr = blended
        else:
            final_repr = summary
        final_repr = self.final_norm(final_repr)
        logits = self.output_proj(final_repr)

        result = {'logits': logits, 'final_repr': final_repr}

        # Legacy 8-token display fields ('token'/'confidence') are computed ONLY
        # for the string-prompt demo path. Evaluating them on every tensor
        # forward cost T device syncs plus a full [B,T,vocab] softmax per
        # training step, for values no training code reads.
        if from_string:
            pred_ids = logits.argmax(dim=-1)
            pred_tokens = [self.tokens[t] if t < len(self.tokens) else "unk"
                           for t in pred_ids[0].tolist()]
            last_probs = F.softmax(logits[0, -1, :], dim=-1)
            result['token'] = pred_tokens[0] if pred_tokens else "unk"
            result['confidence'] = last_probs[pred_ids[0, -1]].item()

        if return_activations:
            result['activations'] = torch.stack(node_activations, dim=1)

        if return_entanglement:
            result['entanglement'] = self._compute_coherence(node_states)

        return result

    def _compute_coherence(self, node_states):
        """Classical analogue of entanglement: pairwise correlation between node
        activations. Returns a CoherenceData-like object with `von_neumann`
        (pairwise |corr|) and `mutual_info` (pairwise corr) dicts."""

        @dataclass
        class CoherenceData:
            von_neumann: Dict
            mutual_info: Dict
            renyi_2: Dict
            schmidt_values: Dict
            log_negativity: Dict

        # node_states: [B, T, N, D] -> [N, B*T*D] flattened per node
        flat = node_states.reshape(self.n_nodes, -1).float()
        flat = flat - flat.mean(dim=1, keepdim=True)
        norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        flat = flat / norm
        corr = torch.mm(flat, flat.t())  # [N, N] pairwise correlation (cosine sim)

        von_neumann = {}
        mutual_info = {}
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                key = "n{}_{}".format(i, j)
                c = corr[i, j].item()
                von_neumann[key] = abs(c)
                mutual_info[(i, j)] = c
        return CoherenceData(
            von_neumann=von_neumann,
            mutual_info=mutual_info,
            renyi_2={},
            schmidt_values={},
            log_negativity={},
        )

    def train_step(self, prompt, target_token, lr_mlp=0.01):
        """
        Single training step: predict next token and update edge MLPs + alpha/beta.

        Args:
            prompt: input text string
            target_token: target token from vocabulary
            lr_mlp: learning rate for MLP updates

        Returns:
            dict with loss_after, alpha, beta
        """
        self.train()

        tokens = prompt.lower().split()
        token_ids = []
        for tok in tokens:
            if tok in self.tokens:
                token_ids.append(self.tokens.index(tok))
            else:
                token_ids.append(0)

        if not token_ids:
            token_ids = [0]

        input_ids = torch.tensor([token_ids], dtype=torch.long,
                                 device=next(self.parameters()).device)

        if target_token in self.tokens:
            target_idx = self.tokens.index(target_token)
        else:
            target_idx = 0

        out = self.forward(input_ids)
        logits = out['logits']

        shift_logits = logits[:, -1:, :]
        target_ids = torch.tensor([[target_idx]], dtype=torch.long,
                                  device=next(self.parameters()).device)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            target_ids.view(-1)
        )

        edge_params = []
        for edge_net in self.edge_networks.values():
            edge_params.extend(list(edge_net.parameters()))

        if edge_params:
            grads = torch.autograd.grad(loss, edge_params, retain_graph=True, create_graph=False)
            with torch.no_grad():
                for param, grad in zip(edge_params, grads):
                    if grad is not None:
                        param.data -= lr_mlp * grad

        alpha_grad = torch.autograd.grad(loss, self.alpha, retain_graph=True)[0]
        beta_grad = torch.autograd.grad(loss, self.beta, retain_graph=True)[0]

        with torch.no_grad():
            self.alpha.data -= lr_mlp * alpha_grad
            self.beta.data -= lr_mlp * beta_grad
            self.alpha.data.clamp_(-1.0, 1.0)
            self.beta.data.clamp_(-1.0, 1.0)

        return {'loss_after': float(loss.item()), 'alpha': float(self.alpha),
                'beta': float(self.beta)}

    def full_train_step(self, input_ids, target_ids, optimizer=None, lr=3e-4,
                        grad_clip=1.0):
        """Full-parameter training step using standard cross-entropy next-token
        loss over ALL parameters (embeddings, transformer nodes, edge networks,
        output projection, alpha/beta). Use this for real language training.

        Args:
            input_ids: [B, T] token ids
            target_ids: [B, T] target token ids (same shape as input_ids for
                        next-token prediction)
            optimizer: torch optimizer; if None, creates a fresh AdamW
            lr: base learning rate
            grad_clip: global gradient norm clip

        Returns:
            dict with 'loss' (total, incl. MTP auxiliary) and 'main_loss'
            (next-token CE only); plus 'mtp_loss' when MTP heads are active.
        """
        input_ids = input_ids.to(next(self.parameters()).device)
        target_ids = target_ids.to(next(self.parameters()).device)

        created = False
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
            created = True

        optimizer.zero_grad(set_to_none=True)
        out = self.forward(input_ids)
        logits = out['logits']  # [B, T, vocab]
        main_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=0,
        )
        loss = main_loss
        mtp_loss = None
        if len(self.mtp_heads) > 0:
            # Hierarchical forecasting: from the representation at position t,
            # predict not just t+1 (main head) but t+2..t+K (auxiliary heads).
            # Uses the same batch: head k reads final_repr[:, :-k], target
            # target_ids[:, k:]. Main loss stays the reported metric.
            final_repr = out['final_repr']
            aux = []
            for k, head in enumerate(self.mtp_heads, start=2):
                if final_repr.size(1) <= k:
                    break
                aux.append(F.cross_entropy(
                    head(final_repr[:, :-k]).reshape(-1, logits.size(-1)),
                    target_ids[:, k:].reshape(-1),
                    ignore_index=0,
                ))
            if aux:
                mtp_loss = torch.stack(aux).mean()
                loss = loss + self.config.mtp_weight * mtp_loss
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()

        if created:
            del optimizer
        res = {'loss': float(loss.item()), 'main_loss': float(main_loss.item())}
        if mtp_loss is not None:
            res['mtp_loss'] = float(mtp_loss.item())
        return res

    def generate(self, prompt, max_new_tokens=16, temperature=1.0):
        """Generate text autoregressively from a prompt string.

        Args:
            prompt: input text string
            max_new_tokens: number of new tokens to sample
            temperature: sampling temperature (1.0 = softmax sampling)

        Returns:
            list of generated tokens
        """
        self.eval()
        tokens = prompt.lower().split()
        token_ids = []
        for tok in tokens:
            if tok in self.tokens:
                token_ids.append(self.tokens.index(tok))
            else:
                token_ids.append(0)
        if not token_ids:
            token_ids = [0]

        input_ids = torch.tensor([token_ids], dtype=torch.long,
                                 device=next(self.parameters()).device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.forward(input_ids)
                logits = out['logits']  # [1, T, vocab]
                last = logits[0, -1, :]
                if temperature <= 0:
                    next_id = last.argmax()
                else:
                    probs = F.softmax(last / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1).squeeze(0)
                input_ids = torch.cat([input_ids,
                                       next_id.unsqueeze(0).unsqueeze(0)], dim=1)
                if input_ids.size(1) >= self.max_seq_len:
                    break

        gen_tokens = []
        for tid in input_ids[0, len(token_ids):].tolist():
            if tid < len(self.tokens):
                gen_tokens.append(self.tokens[tid])
            else:
                gen_tokens.append("<t{}>".format(tid))
        return gen_tokens


def create_raslila(config_dict: dict):
    """Factory function to create Raslila from config dict."""
    config = RaslilaConfig(**config_dict)
    return Raslila(config)


if __name__ == "__main__":
    """Smoke demo using the production config (5x5x5 grid, depth 5)."""
    config = RaslilaConfig()

    model = Raslila(config)
    n = sum(p.numel() for p in model.parameters())
    print(f"Grid       : {config.dim_x}x{config.dim_y}x{config.dim_z} "
          f"= {model.n_nodes} regions (width x height x front-to-back)")
    print(f"Depth      : {config.num_layers} rounds x {config.node_depth} blocks")
    print(f"Edges      : {int(model.adjacency.sum())}")
    print(f"Params     : {n:,} ({n/1e6:.1f}M)")

    x = torch.randint(1, config.vocab_size, (2, 32))
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Output     : {tuple(out['logits'].shape)}")
    print("Raslila model working!")
