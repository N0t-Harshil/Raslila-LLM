# 11 — Raslila v4: complete documentation and architecture breakdown

Status: 2026-11-17. Every number here was measured on 3x A100 80GB unless marked
"projection". Code: `src/raslila/cortex.py`. Study log and decision rules:
[09 Scaling Study](09_Scaling_Study.md). Research basis: [10 Research and v4 Design](10_Research_and_v4_Design.md).

---

## 1. What Raslila is, in one paragraph

Raslila v4 is a language model that **thinks in loops**. A standard transformer
passes each token through a fixed stack of layers once. Raslila has a short
entry stack (the *prelude*), a **thinking core** that it runs again and again on
its own output (the *loop*), and a short exit stack (the *coda*). The input is
re-injected into every loop, so the model keeps refining its answer instead of
forgetting the question. The core's 1024 neurons are laid out as a
**3D cube (8 x 8 x 16)**, so we can measure and train brain-like spatial
organisation. Because thinking reuses the same weights, Raslila gets more
quality **per parameter** than a standard transformer; the price is more compute
per token.

---

## 2. Architecture breakdown (150m model, `configs/scaling/raslila_150m.yaml`)

```
 tokens (ids)                                         [B, T]
   |
   v
 EMBEDDING  32,768 x 1024 (shared with the output)    [B, T, 1024]
   |
   v
 PRELUDE  2 x SandwichBlock (unique weights)  --> e   [B, T, 1024]   "read the input"
   |                                            |
   |   s0 = random noise (std 1)                |  e is re-injected
   v                                            v  every loop
 +---------------------- THINKING LOOP (r = 1 .. R) ----------------------+
 |  x = Adapter( concat[s, e] )      2048 -> 1024                         |
 |  s = Core(x) = 6 x SandwichBlock (the SAME 6 blocks every loop)        |
 |  the 1024 neurons of s are the 3D cube 8 x 8 x 16                      |
 +------------------------------------------------------------------------+
   |  s after R loops
   v
 CODA  2 x SandwichBlock (unique weights)                             "write the answer"
   |
   v
 RMSNorm -> logits = h . Embeddingᵀ                   [B, T, 32,768]  next-token
```

Effective depth = 2 + 6 x R + 2 layers. At R = 4 that is 28 layers deep with the
weights of 10.

### 2.1 Parameters

| part | what it is | parameters | share |
|---|---|---|---|
| Embedding / output head | 32,768 tokens x 1024, tied (one matrix for both) | 33.55M | 21% |
| Prelude | 2 SandwichBlocks, run once | 25.31M | 16% |
| **Thinking core** | **6 SandwichBlocks, run R times** | **75.92M** | **47%** |
| Adapter | Linear 2048 -> 1024: merges the state s with the input e | 2.10M | 1% |
| Coda | 2 SandwichBlocks, run once | 25.31M | 16% |
| Final norm | RMSNorm(1024) | 0.001M | |
| **Total** | | **162.18M** | |

The standard-transformer baseline of the same size (`baseline_150m`) is 10
plain layers of the same width: 160M parameters.

### 2.2 The SandwichBlock (one layer)

```
x^ = n2( x  + Attention( n1(x) ) )
x  = n4( x^ + SwiGLU-MLP( n3(x^) ) )
```

| piece | detail |
|---|---|
| width | d = 1024, 16 heads of 64 |
| attention | causal, fused QKV (1024 -> 3072), RoPE positions (theta 10,000), RMSNorm on q and k (QK-norm), PyTorch SDPA (flash) |
| MLP | SwiGLU, hidden 2752 (1024 -> 2x2752 -> 1024) |
| norms | four RMSNorms: before AND after each sublayer ("sandwich", from Huginn) |
| parameters | 12.65M per block |

Why sandwich norms: a normal pre-norm residual stream grows every time a layer
is applied. Run the same block 16 times and it blows up. The norms after each
sublayer keep the state at a fixed scale however many loops run; this is what
made Huginn's recurrence trainable at 3.5B.

### 2.3 The thinking loop

- **Start state:** random noise, std 1. The model must learn to converge from any
  start, which makes the loop a refinement process rather than a fixed pipeline.
- **Input injection:** every loop sees `concat[s, e]` through the adapter, so the
  original input never fades.
- **Same weights every loop:** extra thinking costs compute, not parameters or memory.

### 2.4 The 3D cube

The 1024 numbers of the state are given coordinates in an 8 x 8 x 16 cube. Every
loop still mixes all 1024 (attention + MLP are full width; nothing is cut off,
which was the mistake of v2/v3). The cube is used two ways:
1. **Measurement** (`scripts/brain_metrics.py`): do neighbouring neurons respond alike?
2. **Optional training loss** (`topo_weight`, from TopoLM): pushes neighbouring
   neurons to correlate, forming brain-like regions.
   `loss = 0.5 x (1 - corr(pairwise activity correlation, 1/(distance+1)))`
   over 4 random neighbourhoods of radius 2 per step.

---

## 3. How it is trained

| setting | value |
|---|---|
| objective | next-token prediction (cross-entropy on the final loop's output only) |
| data | 60% FineWeb-Edu, 10% Wikipedia, 15% FineMath, 15% code; deduplicated; one token per digit |
| tokenizer | BPE, 32,768 tokens |
| tokens | 3.2B (150m tier, ~20 tokens/parameter, Chinchilla); 1.0B (50m tier) |
| batch | 524,288 tokens per step (0.5M), sequence 1024 |
| optimizer | AdamW (0.9, 0.95), lr 5e-4 (150m) / 8e-4 (50m), warmup 2%, cosine to 10%, weight decay 0.1, grad clip 1.0 |
| precision / hardware | bf16 autocast, per-block torch.compile, DDP on 3x A100 80GB |
| **loops per step** | random, log-normal Poisson with mean 4, capped at 16 (Huginn), same count on every GPU in a step |
| **backprop** | only through the last 4 loops; earlier loops run without gradients, so memory does not grow with thinking time |

Random loop counts teach the model to give a usable answer after any number of
loops, which is what makes "think longer at test time" possible.

**Lean schedule** (`raslila_150m_lean`): mean 3 loops, backprop 2, test 4. Cuts
training compute from 2.22x to 1.61x the baseline.

**Topo variant** (`raslila_150m_topo`): same as raslila_150m plus the 3D loss at weight 0.5.

---

## 4. How it runs (inference)

| mode | how | used by |
|---|---|---|
| fixed loops | run R loops for every token (default R = 8; 4 is enough) | benchmarks, val loss |
| read out at several R | `exit_at=[1,2,4,8,16]`: one pass, answers after each | the "thinking" curves |
| whole-sequence halting | stop when the next-token distribution stops changing (KL < 5e-4) | text generation |
| **per-token adaptive (new)** | **each token stops when its own state stops changing** | `python run.py adaptive` (section 7) |

### 4.1 Compute per token (sequence 1024)

| model | layers run per token | inference GFLOPs / token |
|---|---|---|
| baseline_150m (160M) | 10 | 0.362 |
| Raslila, 1 loop | 2 + 6 + 2 | 0.366 |
| Raslila, 2 loops | 2 + 12 + 2 | 0.547 |
| Raslila, 3 loops | 2 + 18 + 2 | 0.729 |
| Raslila, 4 loops | 2 + 24 + 2 | 0.910 |
| Raslila, 8 loops | 2 + 48 + 2 | 1.634 |
| baseline_150m_iso (350M) | 25 | 0.804 |

Training FLOPs per token (forward + backward): baseline_150m 1.09e9; Raslila
2.40e9 (2.22x); lean 1.74e9 (1.61x); baseline_150m_iso 2.41e9.

Quality is the same at 4 and at 8 loops (section 5), so the 8-loop default
wastes about 45% of inference compute. Per-token halting attacks exactly that.

---

## 5. Results (measured)

### 5.1 Same size

| | baseline | Raslila v4 | diff (neg = Raslila better) | Raslila's training compute |
|---|---|---|---|---|
| 50m (42M / 43M) | 3.1395 | 3.1471 | +0.008 (tie) | 1.86x |
| 150m (160M / 162M) | 2.5929 | **2.5464** | **-0.047** | 2.22x |
| 150m lean | 2.5929 | 2.5663 | -0.027 | 1.61x |
| 150m topo | 2.5929 | 2.5637 | -0.029 | 2.22x |

Benchmarks at 150m (Raslila vs baseline_150m): LAMBADA 24.6% vs 22.0%,
ARC-easy 50.4% vs 47.8%, SciQ 78.1% vs 76.0%, and better bits-per-byte on
WikiText, GSM8K and HumanEval text. Won 9 of 13.

### 5.2 Equal training compute

A 350M standard transformer trained with the same FLOPs as Raslila
(`baseline_150m_iso`) reached **2.5082**, better than Raslila's 2.5464 (+0.038).
The lean version is +0.024 from the (interpolated) baseline of its compute:
inside the 0.03 parity band of the pre-registered rules.

### 5.3 Thinking: more loops help, up to about 4

| loops | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| val loss (raslila_150m) | 2.640 | 2.560 | 2.547 | 2.546 | 2.546 |
| LAMBADA | 21.3% | 23.6% | 24.5% | 24.6% | 24.7% |

The state becomes a fixed point after 3-4 loops in every run: loops beyond 4
cost compute and change nothing.

### 5.4 Brain metrics (`scripts/brain_metrics.py`)

| measurement | what it means | raslila_150m | lean | **topo** | 50m | baseline_50m |
|---|---|---|---|---|---|---|
| spatial r (3D clustering) | neighbours in the cube respond alike; shuffled control ~0.003 | 0.000 | 0.003 | **0.382** | 0.002 | 0.002 |
| PR last/first | dimensionality rises with depth (cortex ~2.25x) | 0.55x | 0.52x | 0.55x | 0.74x | 1.95x |

- **3D brain regions: yes, with the topo loss.** r = 0.38 in every loop vs 0.003
  for the shuffled control, at a cost of +0.017 val loss. Induced by the loss
  (as in TopoLM), not emergent.
- **Dimensionality hierarchy: no.** Raslila's dimensionality falls toward the
  output; the plain transformer's rises.

### 5.5 What can and cannot be claimed

Can claim:
- At 150M parameters, Raslila beats a standard transformer of the same size, at the cost of extra compute per token.
- One model can trade compute for quality at test time (1 to 4 loops).
- The 3D cube forms measurable brain-like spatial regions (r = 0.38 vs 0.003) for +0.017 loss.
- Lean v4 is within 0.024 of a standard model of equal training compute with ~35% fewer parameters.

Cannot claim (yet):
- More compute-efficient than a transformer (measured: it is not, at 150m).
- That the same-size advantage grows with scale (two sizes only; needs a third).
- Brain-like dimensionality hierarchy (measured: no).
- Anything at 10B+ (the report's 10B line is an extrapolation).

---

## 6. The main weakness: the loop stops working after 4

Every run converges to a fixed point by loop 3-4. That has two consequences:
1. Training spends compute on loops 5-16 (heavy tail of the sampler) that teach little.
2. The equal-compute gap: a deeper standard model uses the same FLOPs better.

Two fixes, in order:
1. **Spend fewer loops where they are not needed** (per-token adaptive thinking, section 7).
   Fixes inference cost now; no retraining.
2. **Train with adaptive depth** (Mixture-of-Recursions style: a learned router
   sends only unfinished tokens through the next loop, so training compute drops
   too). Needed to close the equal-compute gap.

---

## 7. Per-token adaptive thinking (new, 2026-11-17)

Idea: a token that is already settled after 2 loops should stop at 2; a hard
token should take 4 or more. Implemented in `RaslilaCortex.adaptive_forward`:

1. Run loop 1 for all tokens.
2. After each further loop, measure each token's relative state change
   `||s_r - s_(r-1)|| / ||s_(r-1)||`.
3. If it is below the threshold, that token is **done**: its state is frozen
   (later tokens can still attend to it) and it is not updated again.
4. Tokens still changing keep looping, up to the cap (16). Stop when all are done.

The check costs nothing (no extra readout). A threshold sweep gives the
quality-vs-compute curve.

**How to measure** (no training, ~minutes per run):
```bash
CUDA_VISIBLE_DEVICES=0 python run.py adaptive --run raslila_150m --run raslila_150m_lean --run raslila_150m_topo
```
Output per run: val loss and average loops per token at each threshold, how many
tokens stopped after 2, 3, 4, ... loops, inference GFLOPs, and the smallest
average that keeps fixed-8-loop quality (within 0.002). Saved to `runs/<run>/adaptive.json`.

**What it does and does not change:**
- It reduces **inference** compute (loops per token). It does not change the
  training compute, so the equal-compute training table does not move. That needs fix 2 above.
- This implementation still computes frozen tokens and discards the result, so
  the saving is in loops counted, not yet in wall-clock. Real speed-up needs
  gather-based execution (only active tokens through the core) or token-by-token
  generation with a KV cache.
- A token cannot stop after loop 1: loop 1 starts from noise, so its change is always large.

Result: **pending** (run the command above).

---

## 8. Variants and configs

| config | params | layout | loops (mean / backprop / test) | topo | tokens |
|---|---|---|---|---|---|
| `scaling/raslila_50m.yaml` | 43M | d 512, 2 + 4 + 2, cube 8x8x8 | 4 / 4 / 8 | 0 | 1.0B |
| `scaling/raslila_150m.yaml` | 162M | d 1024, 2 + 6 + 2, cube 8x8x16 | 4 / 4 / 8 | 0 | 3.2B |
| `scaling/variants/raslila_150m_lean.yaml` | 162M | same | 3 / 2 / 4 (cap 8) | 0 | 3.2B |
| `scaling/variants/raslila_150m_topo.yaml` | 162M | same | 4 / 4 / 8 | 0.5 | 3.2B |
| `scaling/baseline_150m.yaml` | 160M | 10 layers, d 1024 | - | - | 3.2B |
| `scaling/variants/baseline_150m_iso.yaml` | 350M | 25 layers, d 1024 | - | - | 3.2B |

---

## 9. Files

| file | role |
|---|---|
| `src/raslila/cortex.py` | the v4 model: SandwichBlock, loop, adaptive halting, topo loss, FLOP accounting |
| `src/raslila/baseline.py` | the standard transformer yardstick |
| `scripts/train_scaling.py` | training (DDP, compile, loop sampling, logging, checkpoints) |
| `scripts/benchmark.py` | 13 benchmarks, read out at several loop counts |
| `scripts/scaling_report.py` | `runs/scaling_report.md`: same-size, equal-compute, thinking tables, plots |
| `scripts/brain_metrics.py` | dimensionality hierarchy + 3D clustering, with shuffled controls |
| `scripts/adaptive_eval.py` | per-token adaptive thinking measurement |
| `COMMANDS.md` | every command to run on the A100 machine |

## 10. Commands

```bash
python run.py study --tiers 150m                 # train + benchmark + report
python run.py report && cat runs/scaling_report.md
python run.py brain --run raslila_150m_topo      # brain metrics
python run.py adaptive --run raslila_150m        # per-token adaptive thinking
python run.py bench --run raslila_150m --loops 1,2,4,8,16
```

## 11. Bugs found and fixed on the way

- **Core received no gradient under bf16** (`3a2f4e3`): autocast cached the bf16
  weight copies made in the no-grad loops and reused them in the gradient loops,
  so the core and adapter got no gradient on steps with more than 4 loops. Fixed
  with `torch.clear_autocast_cache()`.
- **Brain-metrics legend read as a verdict** (`47d25b6`): now prints an actual
  pass/fail against the shuffled control.

## 12. Next steps, in priority order

1. Run `adaptive` on the three 150m checkpoints (minutes): the inference-compute number.
2. Third size: Raslila v4 vs baseline at ~350M, same-size and equal-compute,
   to test whether the advantage grows with scale.
3. Adaptive training (learned per-token exit, MoR-style) to cut training compute.
4. Make the topo loss the default (it costs only 0.017 loss).
5. Second seed at 150m to put error bars on the -0.047.
