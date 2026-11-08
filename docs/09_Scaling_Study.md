# 09 — The scaling study: does the 3D grid beat a standard transformer?

*Started 2026-11-02. This is now the project's main line of work.*

## Why this study exists: the measured state of v1

The numbers below are from the A100 runs, read straight from the training logs.

| measurement | value |
|---|---|
| v1 model (`config.yaml`) | 163.6M params, 5×5×5 grid, 618 edge MLPs |
| tokens per training "step" | 8 × 128 = **1,024** |
| whole 20,000-step run | **20.5M tokens** in 20,071 s ≈ **1,000 tokens/s** |
| final val loss (sweeps=1) | **5.461** (FineWeb-Edu, 16k BPE) |
| A/B: bidirectional sweeps vs feedforward | **−0.009 nats** (within noise) at **2× compute** |
| adaptive-depth run | went to **NaN** at step ~1,600 (see the fix below) |
| standard 124M transformer, same GPU (public llm.c / nanoGPT results) | **~200,000 tokens/s** |

Three conclusions:

1. **v1 is ~150–200× slower per token than a standard transformer of the same
   size.** That alone rules out training it at any meaningful scale. The cause
   is how it runs, not what it is: 125 small blocks executed one after
   another, each with tiny matmuls, so the GPU sits almost idle.
2. **The loss is high because the model has barely trained** (20M tokens; a
   124M model normally sees 2–3B). It says little about the architecture either way.
3. **No brain mechanism has shown a measurable gain yet.** Feedback sweeps,
   inhibition, persistent state, gating and LNS each came out at about zero within
   noise (docs 02, 05, 07).

So a bigger v1 would not work: 10B parameters cannot even hold AdamW state on
one 80 GB GPU (~160 GB needed), and at v1's speed it would see almost no data.
What investors and labs accept as evidence for a new architecture is a
**scaling curve**: loss against compute at several sizes, against a strong
baseline trained identically. That is what this study produces.

## What changed

### The NaN in the adaptive run (fixed in both v1 and v2)
Each thinking cycle added the alpha/beta interference on top of every block's
residual output, so the state grew about **7× per cycle**: measured readout RMS
was 0.2 → 1.7 → 11 → 80 → 713 → 4,877 → 32,781 → 220,854 over 8 cycles. At 4
cycles in bf16 the gradients overflowed: grad_norm went 110 → 4e4 → 4e8 → 1e14
→ inf at steps 1200–1400, and the loss was NaN from step 1600. Fix: between
cycles the state is RMS-normalised and the original input re-injected (the
recurrent-depth recipe). After the fix: 0.2 → 3.9 → 5.8 → 6.4 → 6.5 → 6.6 → 6.5
→ 6.5, stable even at twice the trained loop count. Both trainers now also
**skip** an update whose gradient norm is not finite, instead of writing NaN into
every weight.

### Raslila v2 (`src/raslila/grid.py`): same idea, GPU-shaped
- A grid of transformer blocks `[layers, rows, cols]`, every block with its own
  weights, lateral neighbours plus small-world long-range links, upward
  messages, and downward feedback.
- **All blocks of a sheet run together** as one batched matmul, and attention
  for all of them is one flash-attention call. The sequential depth per sweep
  is `layers × node_depth`, as in a normal transformer.
- Blocks in a sheet update from the same snapshot (Jacobi). v1's in-sheet
  sequential order bought ~0.026 nats at ~25× sequential cost.
- Messages are linear maps of normalised neighbour states, **zero-initialised**,
  so the model starts as clean parallel towers and learns how much to listen.
- **Thinking loops**: cycle 1 = one bottom-up sweep; each further cycle = down
  through the feedback links and back up, with re-normalisation and input
  injection; an answer is read out after every cycle. Training samples 1..K
  cycles and supervises every exit. At inference it stops once the prediction
  is confident or has stopped changing.
- Readout projects the whole deepest sheet (N × d_node) to the embedding width.
- Activation checkpointing per sheet, so large grids fit in memory.

### The baseline (`src/raslila/baseline.py`)
A standard decoder-only transformer with the **same recipe**: RMSNorm
pre-norm, RoPE, QK-norm, SwiGLU, tied embeddings, no dropout. It is the
yardstick, not a second product. Raslila is only worth funding if it beats it.

### Data (`configs/data.yaml`, `src/raslila/data_mix.py`)
General knowledge + maths + code, interleaved by weight so every slice
(including validation) has the same mix:

| source | content | weight |
|---|---|---|
| HuggingFaceFW/fineweb-edu (sample-100BT) | educational web text | 60% |
| wikimedia/wikipedia (20231101.en) | encyclopedia | 10% |
| HuggingFaceTB/finemath (finemath-4plus) | step-by-step maths | 15% |
| codeparrot/codeparrot-clean | deduplicated Python | 15% |

32,768-token byte-level BPE with **one token per digit** (for maths), NFC,
`<|pad|>`=0, `<|endoftext|>`=1. Junk, stubs and exact duplicates are dropped
(with per-source thresholds; code and LaTeX are symbol-heavy). The 21B-token
cache is streamed to disk incrementally (42 GB, low RAM).

### The study configs (`configs/scaling/`)
Pairs matched on parameters; everything else is identical (checked by
`python run.py check`):

| tier | baseline | Raslila v2 | tokens | est. A100-hours* (B / R) |
|---|---|---|---|---|
| 50m | 42.5M, d512 × 8 | 41.9M, grid 3×3×3, d192 | 1.0B | 1 / 2 |
| 125m | 110.1M, d768 × 12 | 106.1M, grid 3×4×4, d256 | 2.5B | 4 / 10 |
| 350m | 337.2M, d1024 × 24 | 325.4M, grid 5×4×4, d384 | 7.0B | 36 / 87 |
| 1b | 977.9M, d2048 × 18 | 900.9M, grid 5×5×5, d512 | 20B | 281 / 642 |

\*At an assumed 40% MFU. The real number comes from `python run.py speed`.
Total ≈ 1,060 A100-hours ≈ **11 days on 4 A100s**, if Raslila v2 reaches the
assumed efficiency. Raslila trains with 1–2 thinking cycles (≈2.4× the
baseline's FLOPs per token), and the report plots **loss against FLOPs actually
spent**, so the extra compute is counted against it.

Shared recipe: seq 1024, 0.5M tokens per step, AdamW (0.9, 0.95), wd 0.1,
cosine to 10% after a 2% warmup, grad clip 1.0, bf16, same seed, and the same
validation windows for every model.

## How to run it (4× A100)

```bash
python run.py setup                 # packages + GPUs
python run.py check                 # configs are fair and build
python run.py smoke                 # both models train (seconds, no data)
python run.py data                  # corpus, once (hours; prints an ETA)
python run.py speed --tier 125m     # FIRST GO/NO-GO (minutes)
python run.py study                 # every model, one after another, on all 4 GPUs
python run.py status                # anytime: progress / val loss / ETA per model
python run.py report                # the comparison of every model (also run automatically)
```

`study` trains the models **one after another, each on all GPUs**:
baseline_50m → raslila_50m → baseline_125m → raslila_125m → … → raslila_1b.
After each model it **benchmarks** it and **refreshes the comparison**, so a
same-size head-to-head exists as soon as each pair is done. Options:
`--no-bench`, `--quick` (300 examples per task), `--run-code` (adds HumanEval
pass@1, which executes model-written Python), `--tiers 50m,125m`.

Run long commands inside `tmux`. Everything resumes automatically from
`runs/<name>/ckpt.pt` (and finished models/benchmarks are skipped), so after
any interruption just run the same command again.

### What gets logged (nothing is overwritten)
| file | contents |
|---|---|
| `logs/study.log` | timeline of the whole study: start / done / failed, final val loss, benchmark scores |
| `runs/<name>/train.log` | full training output of that model, appended across resumes |
| `runs/<name>/log.csv` | loss, lr, grad norm, tok/s, MFU every 10 steps; val loss every 250 |
| `runs/<name>/metrics.json` | params, tokens, FLOPs, speed, MFU, val-loss history, loss per thinking cycle |
| `runs/<name>/benchmark.json` (+ `benchmark.log`) | every benchmark score, per thinking cycle, sample outputs |
| `logs/<command>_<time>.log` | a copy of every command's output |

### The comparison (`runs/`, rebuilt after every model)
- `scaling_report.md`: training table, benchmark table (the better model at each size in bold, chance level shown), head-to-head per size (val-loss difference, speed ratio, benchmarks won/lost), the thinking-loop table (cycle 1 → 2 on val loss and on every benchmark), and the scaling fit with its 10B projection.
- `compare.csv`: one row per model with every number.
- `scaling.png`: loss vs compute. `curves.png`: val loss during training, every model overlaid. `benchmarks.png`: every benchmark, every model side by side.

### Benchmarks (`python run.py bench --run NAME` or `--all`)
Zero-shot log-likelihood scoring, as lm-evaluation-harness does it, so the
numbers are comparable to published small models:

| area | tasks |
|---|---|
| general knowledge / reasoning | HellaSwag, ARC-Easy, ARC-Challenge, SciQ, OpenBookQA, Winogrande, LAMBADA |
| maths | GSM8K bits-per-byte on worked solutions; GSM8K 3-shot exact answer (250 problems) |
| code | HumanEval bits-per-byte on reference solutions; HumanEval pass@1 (with `--run-code`) |
| language modelling | WikiText-2 bits per byte |

Expect small models to sit near chance on ARC-Challenge, Winogrande and
OpenBookQA, and at ~0–3% on GSM8K and HumanEval. That is normal below 1B and
~20B tokens. The bits-per-byte numbers are the sensitive ones at this scale. What
counts is Raslila vs the baseline at the same size, and how the gap moves
with size. A full benchmark takes minutes for 50m and roughly an hour for 1b
on one GPU.

## Measured speed (A100-SXM4-80GB, torch 2.5.1, compiled, 2026-11-05)

| tier | baseline tok/s | Raslila 1 cycle | Raslila 2 cycles | Raslila as trained (1-2 mixed) | MFU baseline / Raslila |
|---|---|---|---|---|---|
| 125m | 158,392 | 95,157 | 33,694 | ~50,000 (3.2x slower) | 39% / 30% |
| 1b | 21,582 | 15,472 | 5,120 | ~7,700 (2.8x slower) | 44% / 36% |

- torch.compile gives ~2.5x on both families (uncompiled: 22% / 9% MFU), so it is on by default (`--no-compile` disables it).
- Raslila without activation checkpointing is 1.28x faster at both sizes and fits, so its configs now have it off.
- Speed gate: with 1 cycle Raslila runs at 60-70% of the baseline's speed. **Passed.** The rest of its cost is the extra compute of the second cycle, which the report charges against it.
- Estimated study cost at these efficiencies: ~300 GPU-hours (baseline) + ~900 (Raslila), about **13 days on 4 GPUs or 18 on 3**. The 50m + 125m pairs take about 8 hours on 3 GPUs.
- The machine is shared. On 2026-11-05 GPU 2 was ~94% busy with another tenant's job. Multi-GPU training runs at the pace of the slowest GPU, so exclude a busy GPU (`CUDA_VISIBLE_DEVICES=0,1,3`). Tokens per step stay identical across families on any GPU count.

## Result 1: v2 at 50m / 125m (2026-11-05) and the v3 change

| tier | baseline val loss | Raslila v2 val loss | gap | compute multiplier |
|---|---|---|---|---|
| 50m | 3.1395 | 3.3313 | +0.19 | |
| 125m | 2.6793 | 2.8017 | +0.12 | 0.24x (Raslila matches a baseline using 1/4 of its compute) |

The baseline was also better on every bits-per-byte benchmark (wikitext, gsm8k,
humaneval: 5-9%), and the 2nd thinking cycle was slightly *worse* than the 1st
(+0.011 / +0.005). By rules 2 and 4 below, 350m/1b were not run with v2.
The v2 runs are kept in `runs/v2/`.

Diagnosis: in v2 every block received the same fixed average of its
neighbours through one linear map, upward only on the way up and downward
only on the way down. No block could choose what to listen to, so the cube
behaved like loosely coupled narrow towers.

**v3 (`routing: true`, now in every raslila config).** For every token, every
block chooses whom to listen to: each sideways neighbour and long-range link,
the block below (fresh), the block above (feedback from its previous state), or
nobody. The choice is a softmax over those sources, scored by content and by a
learned key per *direction*, through a narrow `msg_rank` channel (d_node/4).
Connections stay cheap, and a block can switch them off. Feedback now arrives
inside each upward sweep, so a thinking cycle is one sweep instead of two
(compute 2.4x -> 1.9x the baseline's per token at 50m). On a toy task where
each token must fetch from "below" or "above" depending on its own state, the
learned router reaches loss 0.04 versus 0.54 for v2-style fixed averaging.
Whether this closes the gap on language is what the 50m/125m rerun measures;
the decision rules below are unchanged.

## Decision rules (fixed before seeing the results)

1. **Speed gate (day 1).** `run.py speed`: if Raslila v2 runs at less than
   ~1/4 of the baseline's tokens/s at equal parameters (beyond its intended
   extra sweeps), optimise first (try `--compile`, larger micro-batches)
   before spending study time.
2. **Small-tier gate (~day 2–3).** After the 50m and 125m pairs: if Raslila's
   loss at equal FLOPs is clearly worse at both sizes **and** the gap is not
   shrinking, do not run 350m/1b as is. Change the design first.
3. **Main result (~week 2–3).** With 3–4 sizes per family, the report fits
   `loss = a·C^−b` and gives the **compute multiplier**: how much more compute
   the baseline needs to match Raslila. A multiplier > 1 with an equal or
   steeper slope is a result worth showing investors. Anything else is not, and
   saying so early is better than burning the remaining budget.
4. **Thinking loops.** The per-cycle validation loss shows whether a second
   cycle helps. If cycle 2 is not better than cycle 1, the loops are not
   earning their compute.

## Honest limits
- One seed per run. Differences under ~0.02 nats at a single size are noise;
  trust the curve, not one point.
- Learning rates follow GPT-3's per-size table and are shared by both families.
  They are tuned for standard transformers, which slightly favours the baseline;
  a short LR sweep for Raslila at 50m is cheap and worth doing if it is close.
- The 10B+ number in the report is an **extrapolation** of the fitted curves,
  labelled as such. It is how large runs are justified, but it is not a measurement.

## Final pre-launch fixes (2026-11-02, configs frozen after this)
- **Weight-decay bug fixed (trainer):** Raslila's per-block norm gains are
  stored as `[N, 1, D]` tensors, so the "decay only tensors with dim >= 2" rule
  was decaying them towards zero. Norm gains and grid roles are now excluded
  by name.
- **fp32 residual stream in Raslila v2:** the baseline's residual sum is fp32
  (its embedding output is fp32), while Raslila's was bf16 after `in_proj`.
  Summing dozens of updates in bf16 loses precision, so Raslila now keeps the
  running state in fp32 (matmuls stay bf16), making the comparison fair.
- **Init scaled for the loops:** output projections are scaled by
  `1/sqrt(2 * layers * node_depth * sweeps_per_cycle)`; measured readout RMS is
  flat (0.59-0.62) across 8 thinking cycles.
- **z-loss 1e-4 for both families** (PaLM) against late-run logit drift. The
  logged and validation losses are plain cross-entropy.
