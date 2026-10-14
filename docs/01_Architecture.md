# 01 — Architecture

Raslila is a decoder-only language model. Input is a token sequence; output is a
probability distribution over the next token. Everything below is chosen from
published evidence or our own measurements (see [02_Research_Log.md](02_Research_Log.md)).

## The structure: a deep grid, not a chain

A conventional LLM is the **degenerate 1D case** — a flat chain of blocks:

```
[0]-[-]-[-]-[-]-[-]-[-]-[-]-[-]        one dimension, one pass forward
```

Raslila restores the **2D sheet inside each layer**, and stacks sheets in depth:

```
   layer 5     [0]-[-]-[-]-[-]-[-]      blocks connect laterally (2D sheet)
               [0]-[-]-[-]-[-]-[-]
               [0]-[-]-[-]-[-]-[-]
                |      |      |
   layer 4     [0]-[-]-[-]-[-]-[-]      ...and vertically to the layer above
               [0]-[-]-[-]-[-]-[-]
                |      |      |
   layer 3     [0]-[-]-[-]-[-]-[-]
                 ...               -> 5 layers x 5x5 = 125 blocks
   layer 1     [0]-[-]-[-]-[-]-[-]
```

**This is cortex-like**: each layer is a cortical sheet, blocks connect laterally
within the sheet, and sheets feed into each other in depth.

### Two axes, and they are different

| Knob | Kind | Current | Meaning |
|------|------|---------|---------|
| `depth_layers` | sequential depth | 5 | distinct sheets, **each with its own weights** |
| `grid_h`, `grid_w` | sheet size | 5, 5 | 2D blocks per sheet, connected laterally |
| `num_layers` | local cascade | 1 | extra message-passing rounds *inside* a sheet |
| `node_depth` | block depth | 1 | transformer blocks stacked inside one region |

Because each layer owns distinct weights, this is a **genuinely deep** network —
unlike the earlier flat-lattice design, where 5 rounds reused one weight set
(recurrent). Depth here costs both parameters and compute, as it should.

## Connectivity

| Connection | Default | Count | Meaning |
|---|---|---|---|
| `lateral` | on | within a sheet, 8-neighbour 2D (`diagonal: true`) | cortical columns talking laterally |
| `vertical` | on | same cell in layer L±1 | feedforward between sheets |
| `feedback` | on | L+1 → L as well | **cortex is not feedforward** |
| `long_range` | 30%, ≤8 hops | random | small-world white-matter tracts |

At 5×(5×5): **618 edges, mean in-degree 4.94**. Turning feedback off drops it
to 518 (mean 8.11); turning lateral off collapses it to 714 (mean 5.71) — the
degenerate case we are trying to escape.

> **Wiring changed 2026-10** from `diagonal: false` + `long_range_prob: 0.1` +
> `max_long_range: 3` (618 edges, mean in-degree 4.94; 518 feedback-off, 218
> lateral-off) to the brain-like values above. The cortex is small-world — dense
> local columns plus sparse long-range tracts — and it is not a Manhattan lattice,
> so cells reach diagonally. Measured cost on the 50M config: 74,194,142 →
> 92,589,790 params, 1378 → 1854 ms forward (+34%). **This invalidates every
> checkpoint taken before the change**: the edge key set differs, so an old `.pt`
> will not load into a new model.

### `depth_sweeps`: connections that exist vs connections that flow

Having feedback edges in the graph is **not** the same as using them. With a
single bottom-up pass, layer 1 is written before layer 2 has anything to send
back — so the feedback edges were present in the adjacency matrix and completely
unused in the executed model. The claim "the connections are bidirectional" was
true on paper and false in practice.

`depth_sweeps` fixes this by alternating traversal direction:

| Sweep | Traversal | |
|---|---|---|
| 0 | L5 → L1 | hypothesis comes down from the top |
| 1 | L1 → L5 | refinement propagates back up |
| 2 | L5 → L1 | again |
| … | alternating | |

**Direction order is a correctness issue, not a style choice.** The readout reads
the *deepest* layer, so the final sweep must end there. Ending on a downward pass
would mean the readout sees a state written before the lower layers responded —
discarding the whole point of the feedback. So the sequence is deliberately
down, up, down, up.

Cost: **~2× compute, zero extra parameters.** The weights are reused, so this is
a purely computational cost, and at `depth_sweeps=2` the parameter count is
byte-identical to `depth_sweeps=1` (163,605,123 both ways) — meaning any
difference in loss is attributable to the sweep and nothing else.

Verified: 2 sweeps change the output representation ~19% versus 1 sweep, 3 sweeps
~31%, and the deepest layer's state genuinely changes after feedback returns.

**Not yet verified on loss.** See [02 Research Log](02_Research_Log.md) §11.

## Data flow

```
tokens → token_embedding → input_proj → 125 block states (5 layers × 5×5)
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │  layer 1 → layer 2 → ... → layer 5            │  sequential
                    │  within a layer, blocks cascade and exchange  │  sheets
                    │  messages via edge MLPs (lateral + vertical    │
                    │  + feedback + small-world)                    │
                    └───────────────────────┬───────────────────────┘
                                            │
                          learned attention pooling over layer 5's 25 blocks
                          (+ binding over 25×4 emergent composites)
                                            │
                            final_norm → output head (tied) → logits
```

## Layer by layer, with justification

| Layer | Choice | Why (evidence) |
|-------|--------|-----------------|
| Positional | **RoPE** (rotary on Q/K) | Biggest single win in the literature; measured part of our −0.27 nats modern-vs-legacy gap |
| Normalization | **RMSNorm, pre-norm**, plus a final norm | LLaMA/Mistral/Qwen consensus. The final norm is load-bearing: without it pre-norm logits explode (init loss ~110) |
| Attention | Causal **self-attention** + per-head **QK-norm** | Causal is correctness, not a feature — without it the LM objective leaks the answer. QK-norm is cheap insurance (Gemma/OLMo/Qwen) |
| FFN | **SwiGLU** (gated SiLU) | Consistently beats parameter-matched GELU. Width auto-scaled ~2/3 for param parity |
| Regions | **5 sequential layers × 5×5 2D sheets = 125 blocks**, distinct weights per layer | A conventional LLM is the degenerate 1D case (one flat chain). Raslila restores the 2D sheet per layer and stacks sheets — cortex-like |
| Node identity | Learnable role embedding from (col, row, layer) position | Breaks block symmetry so regions specialize. Zero-init so early activations stay balanced |
| Edges | `EdgeNetwork` MLP per connection (618 at 5×(5×5)), plus learned α/β interference | The original "edges as first-class citizens" idea; the single thing a 1D transformer cannot do |
| Recurrence | Immediate (Gauss-Seidel) cascades **within** each sheet, plus **inter-layer feedback** | A block reads neighbours' *fresh* states; and L+1 feeds back to L because the cortex is not a feedforward stack. Measured −0.026 nats in the flat-lattice A/B |
| Binding | Multiplicative bind of the readout blocks to learned roles, read as a **set** | The binding problem fix. A weighted mean is *linear* in the block set, so no composite can emerge. Measured −0.041 nats — the largest single gain |
| Readout | Learned attention over the **last layer's** blocks | Deep-net semantics: the deepest sheet produces the output |
| Output | `output_proj` tied to `token_embedding` | Standard; requires GPT-2-style 0.02 init or the softmax saturates (init loss ~45) |
| Vocabulary | **BPE 199,995** (id 0 = `<pad>`), byte-level 257 optional | BPE is 5.43 chars/token vs bytes' 1.0 — a 5.4× cut in steps-per-epoch. See [04](04_Training_and_Data.md) |

## Growth (cortical growth, optional)

The model can start small and add whole layers as training plateaus. Two design
choices make this stable rather than destabilising:

1. **Newborn blocks are cloned, not randomly initialised.** The
   structural-plasticity literature is clear that fresh random units are
   "forward-active but backward-starved" and disrupt training; cloned ones arrive
   pre-integrated.
2. **Growth is dense, not routed.** Every block keeps participating. This is
   deliberately *not* a mixture-of-experts: nothing is switched off, because the
   brain is one integrated system, not a set of isolated specialists.

Because a layer is the unit of depth, growth adds `grid_h × grid_w` blocks at
once, so early growth should start from a small grid (e.g. 2 layers × 3×3) rather
than by expanding a large one. Set `grow_every: 500` to enable (0 = fixed size).

## Config that matters most

```yaml
raslila:
  layout: deep_grid
  # Deep grid: 2D sheet per layer, layers stacked in depth (cortex-like)
  depth_layers: 5                  # sequential layers, distinct weights each
  grid_h: 5                        # rows per layer
  grid_w: 5                        # blocks per row  -> 125 blocks
  num_layers: 1                    # cascade rounds INSIDE a layer
  node_depth: 1                    # blocks stacked inside one region
  depth_sweeps: 2                  # <-- THE VARIABLE UNDER TEST: bidirectional
                                   #     flow. 1 = feedforward, 2 = down+up.
  hidden_dim: 256
  vocab_size: 16384                # BPE tokens (id 0 = <pad>)
  # Connectivity (graph structure)
  lateral: true                    # 2D neighbours within a sheet
  vertical: true                   # same cell in layer L+/-1
  feedback: true                   # L+1 -> L edge exists
  diagonal: true                   # 4 -> 8 neighbour within a sheet
  long_range_prob: 0.3             # small-world shortcuts, <= 8 hops
  max_long_range: 8
  connectivity_seed: 42            # seeds the WIRING only, not the weights
  readout_layer: last              # deepest sheet feeds the readout
  # Measured wins on the OLD flat-lattice harness (enabled)
  use_rope: true
  use_swiglu: true
  use_rmsnorm: true
  immediate_updates: true          # multidirectional cascades  (-0.026 nats)
  bind_roles: 4                    # binding / conceptual blending   (-0.041 nats)
  # Implemented, structurally verified, UNPROVEN on loss -> ALL OFF
  inhibit: false                   # signed lateral inhibition edges
  state_dim: 0                     # per-block recurrence across tokens
  replay_prob: 0.0                 # experience replay
  # Tested, ties or worse (off)
  mtp_steps: 0
  workspace_slots: 0
  readout_topk: 0
  priority_order: false
  grow_every: 0
```

**The current run tests exactly one thing.** `depth_sweeps: 2` is on; `inhibit`,
`state_dim` and `replay_prob` are implemented but **off**. They are turned on one
at a time afterwards, and only kept if they beat this baseline. Enabling several
at once makes the result unattributable — that is how the project produced a
3-seed "win" that a 6-seed rerun erased.

> The binding/recurrence A/B results were measured on the earlier **flat-lattice**
> harness (18-region 3D grid, byte vocab). The mechanisms are unchanged and carry
> over to the deep grid, but those specific numbers are not re-measured here.

## Measured cost

| Config | Blocks | Params | s/step (CPU, b2×s64) |
|---|---|---|---|
| 5 × (5×5), **sweeps=1** (feedforward) | 125 | 163,605,123 | 7.4 |
| **5 × (5×5), sweeps=2 (ACTIVE)** | 125 | **163,605,123** | ~13.9 (1.9×) |
| 5 × (5×5), sweeps=3 | 125 | 163,605,123 | ~21 (2.8×) |
| 3 × (3×3), sweeps=1 | 27 | 37.4M | 1.2 (0.2×) |
| flat3d 5×5×5 (old layout) | 125 | 163,605,123 | 6.8 (0.9×) |

Note the parameter column: **sweeping costs compute, not parameters.** Rows 1
and 2 have byte-identical parameter counts (163,605,123), so a loss difference
between them can only come from the bidirectional flow.

Edge counts: 618 with feedback, 518 without, **218 with lateral off** — the
last being the degenerate case. `edge_hidden_dim` is the trim lever (at
618 edges and hidden 256: 128 → 163.6M, 64 → 133M, 32 → 118M).

Because `depth_sweeps=2` doubles step time at production size, the active
baseline uses **batch 2 × seq 64** while tuning. If the sweep does not pay for
itself in loss, that 2× is the argument against keeping it.

> **Note on the A/B harness.** `scripts/benchmark_gpt_vs_raslila.py` runs
> **byte-level (vocab 257)** with its own inline configs, so the published
> ablations stay reproducible and comparable with each other. They do *not* use
> the production BPE config. The loss scales differ accordingly (uniform init is
> 5.55 for 257 vs 9.73 for 16k), so never compare a benchmark number against a
> training log directly.
