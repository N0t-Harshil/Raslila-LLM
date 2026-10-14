# 06 — Known Limitations

Written to be read before anyone spends money on this project.

## The 5T question, answered straight

**A 5-trillion-parameter model cannot be trained by us.** Not "it's hard" — it is
outside the range of any single person or lab.

| Model | Total | Active/token | Who |
|-------|-------|--------------|-----|
| DeepSeek-V3 | 671B | 37B (5.5%) | large org, multi-cluster |
| Kimi K2 | 1T | 32B (3%) | large org, multi-cluster |
| Llama-4 Maverick | 400B | ~17B | large org |

Frontier labs train on tens of thousands of H100-class GPUs for months. Nobody
trains 5T; the largest openly documented runs are ~1T and they took industrial
clusters. The honest ceiling for a well-resourced individual is tens of
millions of parameters.

**What "5T, growing from 10M" can honestly be:**

- **The mechanism is real and we implemented it.** `grow_regions` starts small and
  adds a whole layer on plateaus, cloned from existing blocks
  (integration-friendly birth), wired into lateral, vertical and feedback
  connectivity. Verified working. Note that adding a layer is a coarse step —
  it multiplies blocks by `grid_h × grid_w` at once, so early growth is best
  done from a small grid rather than by carving up a large one.
- **Growing toward 5T is not the interesting part.** The literature is clear that
  growth is *not* symmetric to pruning, and that integration-friendly birth is the
  hard part. We handled that; scaling 5,000× is a funding problem, not a research
  problem.
- **The only proven route to 5T-class capability is sparse activation** — i.e. the
  mixture-of-experts idea. And that is precisely the mechanism we rejected on
  architectural grounds (segregation, not binding). So: 5T is reachable *only* by
  a route we deliberately declined. That is a real tension in this project's
  thesis, not something to paper over.
- **A brain is not a 5T-parameter model.** It has ~86B neurons, fixed
  genetically, that grow *synapses*. The defensible analogy is fixed massive
  structure + sparse activation + lifelong plasticity.

**Recommended framing.** Stop treating 5T as a target. Treat the growth mechanism
as a *scientific* contribution — "does a densely-integrated deep grid benefit from
cortical growth, the way a brain does?" — and demonstrate it at 10M→200M where
the answer is actually obtainable.

## Where our architecture currently loses

- **Against a well-chosen GPT, we are on par per parameter, not ahead.** The
  headline "beats GPT by 0.75 nats" came from comparing against a 16-layer
  narrow (h=384) stack, which trains poorly. A 6.5M 8-layer GPT reached ~2.56
  against our 2.50 at 28M. We win at *equal params* partly because our params
  are spread across many regions that re-process the full sequence — genuinely
  redundant compute.
- **Block re-processing is expensive at scale.** Every block runs a full
  transformer over the whole sequence. At 5 layers × 5×5 that is 125 transformer
  passes per forward, plus 618 edge-MLP calls per sweep issued from a Python
  loop. Measured 7.4 s/step on CPU at batch 2 × seq 64. This is launch-bound on
  GPU and CPU-bound locally. It is the direct cost of the grid thesis and has not
  been optimised away. Batching the edge networks or using a vectorised sparse
  gather is the obvious next engineering step. Adding `num_layers` cascades
  multiplies this linearly (2 rounds ≈ 1.9× step time, same parameters).
- **Edge parameters grow with the grid.** At 125 blocks the 618 edge networks
  are **42% of the model** (110M of 260M with the brain-like wiring; it was 29%,
  61M of 211M, at the old 618-edge wiring), and their count scales linearly with
  regions × neighbours. `edge_hidden_dim` is the trim lever, but shrinking it
  weakens the edge-compute idea that is the architecture's core novelty.
- **Most exotic mechanisms tie.** LNS, gating, MTP, workspace, top-k, priority —
  all ties. The honest reason: they help *deep* models, and we are shallow and
  wide. Binding and multidirectionality are the exceptions, and both are
  genuinely ours.
- **Depth now costs parameters, as it should.** The earlier recurrent design
  reused one weight set across rounds, so depth was free in parameters — which
  also meant it was not really a deep network. Each layer now owns distinct
  weights, so we cannot buy depth cheaply. Measured cost of the correction:
  ~9% slower per step than the flat lattice it replaced.

## Training budget, honestly

One epoch over 300M bytes at batch 4 × seq 128 is ~586k steps. We have run at
most ~4k. **Every quality claim rests on ≤0.2 epochs.** The smooth loss curves
say the model is learning, not that it is good.

## The byte-level tax — now fixed

Byte-level cost ~4× the tokens per unit of language versus BPE. At fixed compute
that is a 4× smaller effective dataset, and it was the largest single cause of the
garbage-looking output.

**RESOLVED:** we now train on BPE-200k at 5.43 chars/token, cutting one epoch
from ~150k steps to ~51k (at batch 8 × seq 128). See
[04_Training_and_Data.md](04_Training_and_Data.md).

## The architecture is spatial, and brains are temporal

The deep grid reproduces the brain's *spatial* organisation well and its
*temporal* organisation not at all. Every cortical neuron integrates over time;
Raslila carries nothing between tokens except attention keys and values, so there
is no recurrence in time — only across space and depth. It also has no inhibition
(all traffic excitatory, so blocks never compete), one memory system instead of
two, no replay, no input gating, and five structurally identical layers. These
are larger gaps than anything left in architectural tuning.

Full audit with fixes and a ranked roadmap:
[`07_Brain_Gap_Analysis.md`](07_Brain_Gap_Analysis.md).

## Over-parameterised for the data we have

260M parameters against 1.5B tokens — still well under one epoch, and 1.63× more
text per token than before. The model has more capacity than the budget can
usefully fill, which is the mirror image of the byte-level problem — we fixed
token efficiency, and now the grid is larger than the budget justifies. If the
loss plateaus early, `config_50m.yaml` (92.6M, or 78.9M at
`edge_hidden_dim: 64`) will train further on the same compute.
the same compute. The 5 × (5×5) grid is the right architecture to *have*; whether
it is the right size to *train now* is a budget question, not a design one.

## Gaps against a real brain

The architecture reproduces the brain's *spatial* organisation and not its
*temporal* or *learning* organisation. The ranked audit, with fixes, is in
[07_Brain_Gap_Analysis.md](07_Brain_Gap_Analysis.md).

Four mechanisms addressing those gaps are **implemented and structurally
verified**: bidirectional depth sweeps, signed lateral inhibition, persistent
per-block state, and experience replay. **Only the bidirectional sweep is
active**, because none of the other three has demonstrated a loss benefit and
activating them together would make the result unattributable. Each gets its own
training run, kept only if it wins.

## Our benchmarks have measured noise, not capability

Every mechanism we A/B'd "tied" — and then, on inspection, the benchmark turned
out to be the problem, three separate times:

- It fed **random bytes**, so the loss was pinned at ln(257) and every variant
  tied exactly. Random data cannot distinguish anything.
- It then trained a 257-vocab model on a **BPE-16k corpus** — the same silent
  data/vocab mismatch that once made the TPU trainer produce garbage without
  crashing.
- A **3-seed run reported inhibition as a win**; the 6-seed rerun said no
  significant change. Seed spread is ±0.009–0.016 nats, the same magnitude as
  every effect being measured.

Underneath all of it: **200 steps is ~0.2 epochs of our corpus.** A benchmark at
that horizon cannot detect a difference between architectures. Treat every
"no significant change" in this project as *unmeasured*, not as *disproved*.

## What we have not done at all

- **No held-out evaluation.** Train loss only — we cannot quote a valid
  perplexity number. This is a real gap.
- **No overfitting check.** With a few thousand steps on 85M tokens there is no
  risk yet, but there is also no evidence the model generalizes.
- **No multi-GPU run.** DDP code exists and is syntax-checked, never executed.
- **No generation benchmarks.** Samples are inspected by eye only.
- **Binding is one seed.** The −0.041 result is promising, not established.

## Bugs found by reading code, not by training

Twelve defects, including two that would have silently invalidated everything:
non-causal attention (the objective could read the answer) and a non-buffer
tensor that would crash the GPU run at step 0. Both were invisible in loss
curves. **Run the audit before trusting any new result** — see
[02_Research_Log.md §6](02_Research_Log.md).
