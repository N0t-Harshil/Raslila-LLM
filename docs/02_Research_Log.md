# 02 — Research Log

Every hypothesis we tested, what we expected, and what actually happened.
Negative results are kept because they are the most useful part.

All A/B runs: identical corpus, seed, optimizer, step count. Only the named
variable changed. Numbers are main next-token cross-entropy.

> **Units.** Every result below is **nats per byte**, measured with the
> byte-level (257-vocab) configuration that the benchmark harness pins. Production
> now uses BPE-199,995 at 5.43 chars/token (it was BPE-16k at 3.34), where
> the init loss is ln(199,995) = 12.2 instead of 5.55. The A/B comparisons in
> this table are internally consistent, but do not compare a number from here
> against a training log.
>
> **Harness note.** Entries 1–7 were measured on the earlier **flat-lattice**
> harness (a single 3D grid of regions with recurrent message-passing rounds).
> The architecture has since been rebuilt as a **deep grid** — 2D sheets stacked
> in depth with distinct weights per layer (see [01_Architecture.md](01_Architecture.md)).
> The *mechanisms* under test (binding, immediate cascades, modern blocks) are
> unchanged and still present, but these specific numbers were not re-measured
> on the deep grid. Entries 8+ are deep-grid-specific.

---

## 1. Does the 3D grid beat a flat stack at equal parameters?

**Hypothesis.** Regions that exchange messages via computing edges should beat a
plain deep transformer with the same parameter count.

**Setup.** GPT-style decoder (h=384, 16 layers) vs Raslila (18 regions, hidden
256, node_depth 1), both ~28M params, same byte corpus, 200 steps.

| Model | Params | Final loss |
|-------|--------|-----------|
| GPT stack | 28.6M | **3.25** |
| Raslila | 28.2M | **2.50** |

**Result.** Raslila wins by 0.75 nats. The architecture is sound.

**Caveat that later became important.** A *well-chosen* GPT does better than this
comparison implies: a 6.5M 8-layer GPT reached ~2.56 against our 2.50. The
param-matched win flatters us because narrow deep GPTs (h=384×16) train poorly.
Against a sensible GPT we are roughly on par per parameter, not dominant. See
[06_Known_Limitations.md](06_Known_Limitations.md).

---

## 2. Modern block recipe vs GPT-2-era blocks

**Hypothesis.** Pre-norm RMSNorm + RoPE + SwiGLU + QK-norm + tied head should beat
the 2019 recipe, as the literature consistently reports.

| | Final loss |
|---|---|
| Legacy block (abs positions, GELU, post-norm LayerNorm) | 2.60 |
| Modern block | **2.33** |

**Result.** −0.27 nats, gap still widening at 200 steps. Adopted.

**Two init bugs this A/B caught** (both would have silently ruined every run):

- **Missing final norm.** Pre-norm residual streams grow without bound; without a
  norm before the head, logits explode — init loss ~110. Legacy post-norm never
  had this problem, so it was invisible until we switched.
- **Unscaled tied-head init.** Default `N(0,1)` embedding rows × normalized hidden
  states saturate the softmax — init loss ~45. Fixed with GPT-2-style 0.02 init
  plus zero-init role projection. Both arms now start at ~5.6 (uniform = 5.55).

---

## 3. Multidirectional recurrence — the core thesis

**Hypothesis.** Information should flow *back and forth* between regions
(`1→2→1→3→…`), not one way. Modern LMs are feedforward; the brain is not.

Raslila's rounds already shared weights (recurrent), but each round used
**frozen snapshots**, so information advanced only one hop per round.

**Setup.** 4 arms, 150 steps, identical ~28M models.

| Arm | Final loss |
|-----|-----------|
| frozen snapshots, 2 rounds | 2.385 |
| **immediate cascades, 2 rounds** | **2.360** |
| + priority ordering (most-surprised first) | 2.382 |
| frozen, **4 rounds** (2× compute) | 2.398 |

**Result.** Immediate cascades win by −0.026 **for free**. The decisive line is
the last one: *doubling* the number of frozen rounds does nothing, while letting
a signal cross the grid in one sweep helps. It is the multidirectionality that
matters, not the pass count. Adopted as `immediate_updates: true`.

---

## 4. Binding / conceptual blending

**Hypothesis.** The 3D grid gives persistent *separate* region streams — exactly
the "slots" binding theory requires — but our readout collapsed them with a
weighted mean, which is *linear* in the region set and therefore cannot express
any composite. Binding regions multiplicatively into a set should let the model
represent "A **and** B", not their average.

**Setup.** 4 arms, 150 steps, identical ~28M models.

| Arm | Final loss |
|-----|-----------|
| base (weighted-mean readout) | 2.346 |
| **bind 4 roles** | **2.305** |
| bind 8 roles | 2.310 |
| bind 4 + 3D-neighbour binding | 2.335 |

**Result.** −0.041 nats — **the largest single gain of any mechanism we tried**.
K=4 ≈ K=8, so 4 roles suffice. Spatial (3D-neighbour) binding added nothing on
top, which is a genuine surprise: the *existence* of separated regions is what
enables binding; adding explicit spatial composites on top does not help further.

**Caveat.** Single seed, 150 steps, and the curve was noisy mid-run (binding was
briefly worse at step 100). Promising and enabled, but not yet proven.

---

## 5. Things that did not work (kept deliberately)

| Mechanism | Result | Interpretation |
|-----------|--------|----------------|
| LNS (LayerNorm Scaling) | 2.484 vs 2.487 — tie | Cures *deep*-stack variance growth. Our per-region depth is 1–4, so there is no curse to cure. Revisit if depth grows. |
| Gated messages (neuromodulation) | 2.500 vs 2.487 — tie | Input-dependent gating adds nothing while edge depth is 2. |
| Multi-token prediction (t+2..t+K) | 2.487 vs 2.472 — tie | Auxiliary targets are harder and need more steps to pay off. Most likely candidate to win on a full run. |
| Global workspace slots | 2.475 vs 2.472 — tie | GWT-faithful and linear-cost, but redundant while edge messages suffice. |
| Top-k sparse readout | 2.499 vs 2.472 — **worse** | Sparsity buys efficiency and robustness, not short-horizon loss. |
| Surprise-ordered sweeps | 2.382 vs 2.385 — tie | Ordering adds no signal the cascade does not already provide. |
| More frozen rounds | 2.398 — no gain | Covered in §3. |

The pattern: mechanisms that help *deep* models do nothing for ours, because we
are shallow and wide. That is consistent, and it is the honest reason the scorecard
is mostly 5s for these.

---

## 6. Tokenization — the largest single change, and it wasn't a loss A/B

**Observation.** Byte-level tokenization gives 1 token per character. English
averages ~4 characters per BPE token. So every unit of compute was spending 3–4×
more steps to represent the same language than necessary.

**Action.** Trained a byte-level BPE (16,384 tokens, id 0 reserved for `<pad|>`)
on the real corpus, and cached the encoded corpus as a flat integer array.

| | tokens | chars/token | steps/epoch @ batch 8 × seq 256 |
|---|---|---|---|
| byte-level | 307,187,000 | 1.00 | ~150k |
| **BPE 16k** | **85,152,000** | **3.34** | **~42k** |

**Result.** A 3.6× cut in steps-per-epoch at identical compute. This is worth
more than every architecture change in §1–§5 combined, and it cost no modelling
decisions at all. It is the main reason the model should stop producing
byte-soup within a Colab-sized budget.

---

## 7. Bugs found by auditing (not by training)

Twelve defects, several of which would have been invisible until they destroyed a run:

- **Non-causal attention** — the original `TransformerNode` accepted a mask that
  was never passed. Every position attended to future tokens, so next-token
  training could read the answer. This invalidated the very first loss numbers.
- **`node_positions` buffer unused** — all 18 regions started identical.
- **`local_radius` ignored** — connectivity hardcoded 6 directions.
- **`adjacency` was not a buffer** — never moved to GPU. Harmless until the
  binding path did tensor math with it, which would have crashed the GPU run at step 0.
- **Per-step performance bug** — a Python loop doing one device sync per token
  position, plus a full `[B,T,vocab]` softmax, on *every* forward pass, to populate
  two legacy display fields no training code reads.
- **`edge_hidden_dim` was dead config** — a ternary with identical branches meant
  every edge silently used full width. Fixing it cut the model 70.9M → 63.9M.
- **TPU trainer ignored BPE** — it always called `load_corpus()`, so with the BPE
  config it would have trained a 16k-vocab model on byte-level token ids. Silent
  garbage, no crash.
- **No data/vocab validation** — `tok_mode: bytes` with a 16k vocab trained
  happily on garbage. Now a hard error at startup.
- Plus: `bind_local` silently inert, `AttributeError` on bind-local-only configs,
  dead `pos_embedding` under RoPE, `NameError` on `--steps 0`, `ZeroDivisionError`
  on `--save-every 0`, unguarded vocab mask in generation, uint16 cache overflow
  risk on large vocabs, stale docstrings, shadowed encode/decode in the generator.

The lesson worth recording: the two worst bugs (non-causal attention, the GPU
crash) were both invisible in loss curves, and the two worst *data* bugs (TPU BPE,
vocab mismatch) would have silently wasted a full Colab session. Only reading
the code found any of them.

---

## 8. Deep grid: restoring the 2D sheet and stacking it in depth

**Hypothesis.** The earlier flat lattice had collapsed to something close to a 1D
chain: regions were reused every round, so "depth" was recurrence rather than
distinct learned layers, and the 2D structure was incidental. Making each layer a
2D sheet, stacking sheets in depth, and giving **every layer its own weights**
should be a genuinely deeper network than the recurrent design at equal parameter
count.

**Setup.** Small config (4 layers × 3×3, hidden 64) to verify the structure is
real, plus cost measurements at production size.

**Result 1 — it is a real deep net, not a relabelled lattice.** Perturbing the
weights of a single block in *layer 1* changes the output, confirming information
flows layer 1 → ... → layer N through distinct parameters rather than looping
through one weight set.

**Result 2 — causality survives the depth.** Changing token t+1 leaves logits at
all positions ≤ t bit-identical, across all 4 layers. No future-token leakage was
reintroduced by the sequential layer walk.

**Result 3 — the 2D structure is load-bearing, not decorative.** Connectivity
ablations at 5 layers × 5×5, all 163.6M params:

| Connectivity | Edges | Mean in-degree |
|---|---|---|
| lateral + vertical + feedback (default) | 618 | 4.94 |
| no feedback | 518 | 4.14 |
| **no lateral** | **218** | **1.74** |

Turning off lateral connectivity collapses the network to ~1.7 neighbours per
block, i.e. essentially the 1D chain the thesis claims to beat. The default is
the only setting that keeps blocks genuinely 2D.

**Result 4 — depth costs compute, and that is expected.** Measured CPU step time
at batch 2 × seq 64, 163.6M params:

| Config | s/step | vs default |
|---|---|---|
| 5 × (5×5), 1 round | 7.4 | 1.0× |
| 5 × (5×5), 2 rounds | 13.9 | 1.9× |
| 3 × (3×3), 1 round | 1.2 | 0.2× |
| flat 5×5×5 lattice (old design) | 6.8 | 0.9× |

The old recurrent design's advantage — depth was free in parameters — is
deliberately given up. Reusing one weight set across layers is what made it
*not* a deep network; each layer now owns distinct parameters, so depth costs
both. The deep grid is only ~9% slower per step than the flat lattice it
replaces, which is a fair price for the structural correction.

**Caveats.** Loss is **not yet measured** for this architecture — only structure
and cost are verified. The binding and immediate-cascade numbers in §1–§7 come
from the old harness and are not re-measured here. Growth now adds whole sheets
rather than scattered regions, so the "dense participation" property of early
growth is no longer literally true.

---

## 9. Bugs found while building the deep grid

- **`ModuleDict.get` does not exist** — the edge-message loop used
  `edge_networks.get(key)`, which raises `AttributeError` on *every* forward
  pass. It would have failed at training step 0. Replaced with a membership
  check.
- **`flat3d` legacy layout was unconstructible** — `dim_z` never reached the node
  count, so any `dim_z > 1` raised `IndexError: index N is out of bounds`. The
  legacy layout is now `dim_x*dim_y*dim_z` blocks in a single layer, verified at
  2×2×2, 3×3×2, and 4×4×2.

---

## 10. The gaps we should have been looking at

**Not a training experiment — an audit**, prompted by asking what a brain has that
we do not. Full write-up: [07_Brain_Gap_Analysis.md](07_Brain_Gap_Analysis.md).

The uncomfortable finding: we have spent every cycle refining the *geometry*, and
the geometry is now in good shape. What is missing is that Raslila has **no
organisation over time**.

- **No persistent state between tokens.** The only thing crossing a token
  boundary is attention keys and values. Every forward pass is independent of
  what the model "just" processed. A brain recognises a word in ~150 ms from a
  few letters because neurons integrate over time; we integrate over an attention
  window and no further. Scored **2/10**, and the largest single gap.
- **No inhibition.** ~20% of cortical neurons are inhibitory interneurons, and
  lateral competition is how the cortex decides what wins. Every edge message in
  Raslila is additive. Scored **1/10**. This also explains G3 below.
- **One memory system, not two.** No episodic memory at all — the model forms
  statistical tendencies, never a memory of anything. Scored **1/10**.
- **No replay.** We never revisit anything, which is the standard fix for
  catastrophic forgetting and the prerequisite for continual learning.
  Scored **1/10**.
- **Five homogeneous layers.** Real cortex has six layers with distinct cell types
  and roughly doubling receptive fields; ours are copies differing only by a
  positional embedding. Scored **3/10**.
- **No input gating.** The token embedding broadcasts to all 125 blocks
  identically; a brain's input is modulated by a thalamic relay before it reaches
  cortex.

And a correction to our own record: **`gated_messages` was a bad test of
neuromodulation.** We A/B'd a boolean gate on edge traffic and recorded it as a
tie, concluding the mechanism does not work. But neuromodulation is a
*continuous surprise signal that scales plasticity* — a learning-rate
modulation, not a capacity-control knob. The mechanism was misimplemented before
it was measured, so the negative result does not mean what we said it meant.

**Two things this changes about what to build next.** First, the top three
roadmap items are persistent per-block state, signed/inhibitory edges, and
experience replay — all time and sign, no geometry. Second, the sparsity story
needs reframing: `readout_topk` measured *slightly worse*, and we read that as
"sparsity does not help". But top-k at the output is the wrong place for it. Once
blocks actually compete (inhibition), inactive blocks can be *skipped* rather
than merely down-weighted — which would save the 125× compute, not just produce a
slightly worse number.

All three were built and none is active. See sections 11 and 12 for what happened
when we actually measured them.

---

## 11. The connections were not actually bidirectional

**Claim under test.** Raslila's feedback edges make information flow in both
directions between sheets, so a hypothesis can travel up and be refined coming
back down -- the way the cortex works.

**Setup.** Instrument the adjacency matrix against the execution order.

**Result -- the claim was false as executed.** The feedback edges were present in
the graph and completely unused:

| Pair | Edges in the graph | Edges actually traversed |
|---|---|---|
| L0 -> L1 | 5 | 5 |
| L1 -> L0 | **6** | **0** |

The forward pass walked layers strictly bottom-up. Layer 0 was written before
layer 1 had processed anything, so the L1 -> L0 edges fired against layer 1's raw
input state. The adjacency matrix said "bidirectional"; the code said otherwise.

**Fix.** `depth_sweeps` alternates the traversal direction. Verified after the
fix: 2 sweeps change the output representation **~19%** versus 1 sweep, 3 sweeps
**~31%**, at byte-identical parameter counts (163,605,123 either way) -- so any
loss difference is attributable to the sweep and nothing else.

**Result 2 -- direction order is a correctness issue.** The first implementation
started the sweep bottom-up and ended descending, which meant the readout (which
reads the *deepest* layer) saw a state written before the lower layers had
responded -- discarding the refinement. The sequence is now deliberately down,
up, down, up, so the final sweep always ends at the readout layer.

---

## 12. Three mechanisms built, one variable actually being tested

Built during the gap audit (section 10) and structurally verified:

| Mechanism | Config | Structurally verified | Loss effect |
|---|---|---|---|
| Bidirectional depth sweeps | `depth_sweeps` | yes | **under test now** |
| Signed lateral inhibition | `inhibit` | yes | none measurable |
| Persistent per-block state | `state_dim` | yes | none measurable |
| Experience replay | `replay_prob` | yes | not tested |

All four are implemented. **Only the first is active.** The others are off,
because none has shown a loss benefit and enabling them together would make the
bidirectional result unattributable. They get enabled one at a time, each kept
only if it beats the baseline.

### Two benchmark bugs, both of which faked a result

**Bug 1 -- random bytes as data.** The first version of the A/B fed uniformly
random bytes. Random bytes carry no structure, so the loss sat pinned at
ln(257) = 5.545 and **every variant tied exactly**. A tie that comes from the data
having no structure is not a result about the model. Replaced with the real
BPE-16k corpus.

**Bug 2 -- vocab/data mismatch.** The second version still assumed a 257-vocab
byte corpus while the cached corpus was BPE-16k (ids to 16383). Training a
257-vocab model on 16k-vocab data is the exact silent mismatch that previously
made the TPU trainer train garbage without crashing. Now validated at load.

### The 3-seed win that was not a win

At 200 steps on real BPE data, paired differences against baseline:

| Mechanism | 3 seeds | 6 seeds | Verdict |
|---|---|---|---|
| `depth_sweeps=2` | -0.0072 +/- 0.0091 | not retested | no significant change |
| `depth_sweeps=3` | -0.0010 +/- 0.0149 | not retested | no significant change |
| `inhibit` | **-0.0049 +/- 0.0034 "BETTER"** | **+0.0007 +/- 0.0088** | **not confirmed** |
| `state_dim=16` | -0.0056 +/- 0.0061 | -- | no significant change |
| all three | **-0.0094 +/- 0.0069 "BETTER"** | **+0.0025 +/- 0.0161** | **not confirmed** |

**Three seeds produced a confident "BETTER" that six seeds erased.** Seed spread
is +/-0.009-0.016 nats, the same size as every effect being measured. Any future
A/B here needs >=5 seeds and a paired comparison, and the default verdict is
"no significant change" until it says otherwise.

### Also fixed: a hardcoded seed was silently resetting global RNG

Probing why seeds behaved oddly showed `_build_connectivity` calling
`random.seed(42)` on the **global** RNG. Every model construction rewound the
caller's randomness, so training-time sampling (replay, dropout order) depended
on when the model happened to be built. It also meant the "seed sweep" varied
weights while the *wiring* stayed fixed.

Now a local `random.Random(connectivity_seed)` instance, with
`connectivity_seed` an explicit config field seeding topology only,
independently of weight init. Verified: same weight seed gives identical weights;
fixed `connectivity_seed` gives an identical graph across weight seeds; 42 vs 7
gives 115 vs 119 edges; global RNG untouched by construction.

**Methodological lesson.** 200 steps on a 165M model is at most ~0.2 epochs of
our corpus. Every mechanism "ties" at that horizon, which means the benchmark is
measuring noise, not capability. The honest conclusion is not that the mechanisms
do not work -- it is that we have not given them enough training to find out.

---

## 2026-10 — corpus swap and brain-like wiring (not A/B results, structural changes)

Two changes made outside the one-variable-at-a-time discipline, and deliberately
so: neither is a candidate mechanism, both are corrections to inputs the whole
project was starved of. Neither has a loss number yet.

**1. Corpus: Gutenberg → HuggingFace `Skylion007/openwebtext`.**

The binding constraint was never the architecture, it was tokens. The old corpus
was 85,151,573 tokens. That is 4.7x short of Chinchilla-optimal for the 20B
config (~415M needed) and, for the 50M config, 122.9M tokens consumed over a 30k
step run = 1.4 epochs of 448 books. The live 50M run went flat at ~14k steps
with loss stuck at 3.97-3.99; the honest reading is that it ran out of data
twice over, not that the LR schedule failed.

Now: 1,500,000,000 tokens cached (3.0 GB, uint16 at the then-current 16,384-token
vocab; the 2026-10 (2) entry below supersedes this to 6.0 GB, uint32 at 199,995
tokens), auto-fetched at training start by `src/raslila/hf_dataset.py`, rank 0
only, other ranks barrier-wait.
A 30k-step run consumes 8% of it -- no epoch boundary is ever crossed.
BPE is re-fitted on the new corpus (the fit source is recorded in
`data/tokenizer/meta.json`, so it re-fits when the corpus changes and not when
it does not).

**2. Wiring: 4-neighbour + 10% long-range → 8-neighbour + 30% long-range
(`diagonal: true`, `long_range_prob: 0.3`, `max_long_range: 8`).**

Rationale: the cortex is small-world (dense local columns plus sparse
long-range tracts) and is not a Manhattan lattice -- cells reach diagonally.
Measured cost, same 5x5x5 grid, same `connectivity_seed`:

| wiring | params | edges | fwd (CPU, batch 8) |
|---|---|---|---|
| original (`false`/0.1/3) | 50,508,323 | 618 | 1378 ms |
| brain-like (`true`/0.3/8) | 68,903,971 | 1114 | 1854 ms |

+1.37x params, +1.80x edges, +34% forward. Per config: 3M 2,944,555 ->
3,249,795; 15M 15,441,091 -> 19,335,331; 50M -> 68,903,971; 164M 163,605,123 ->
212,554,371; 20B 20,626,181,123 -> 20,773,012,995.

Note the 20B consequence: 332.4 GB of training state / 4 = **83.1 GB per rank,
over the 80 GB A100**. It no longer fits as written even with activation
checkpointing. `node_depth: 11` gives 17,632,260,995 params = 70.5 GB/rank with
~9.5 GB headroom, and is the recommended shape. See `config_20b.yaml`.

**Not measured yet:** whether the brain-like wiring actually beats the original
on loss. It is a prior borrowed from neuroscience, not a result. The honest
default, given this project's history with unreplicated wins, is "unproven"
until a paired >=5-seed run says otherwise.

**Everything before this entry is invalidated.** The edge key set changed, so
pre-2026-10 checkpoints will not load into the current model. Corpus numbers in
sections above are historical and refer to the deleted Gutenberg data.


## 2026-10 (2) - 200k-token vocabulary and a pruned config set

Structural change again, not an A/B result. Three things landed together.

**1. Tokenizer 16,384 -> 199,995 tokens.** Byte-level BPE, `min_frequency=2`,
fitted by the dataset pipeline on openwebtext. The sample size is a measured
quantity rather than a guess -- fitting a BPE stops early when no more pairs
occur twice, so a short sample silently under-fills the vocab and leaves dead
embedding rows:

| sample | realised vocab | % of 199,995 | fit time |
|---|---|---|---|
| 60M chars | 184,355 | 92.2% | 19s |
| 300M chars | 199,995 | 100% | 57s |
| 1,000M chars | 199,995 | 100% | 155s |

`tokenizer_sample_chars: 300000000` is therefore the smallest sample that
reaches the full vocab. The realised compression, measured on held-out English
prose, is **5.43 chars/token vs 3.34** at 16k -- 1.63x more text per token and
1.63x fewer steps per epoch. Init loss moves from ln(16384) = 9.70 to
ln(199995) = 12.20.

**2. Config set pruned to four files.** `config_3m.yaml` and `config_15m.yaml`
are deleted; `config_50m.yaml`, `config.yaml` (164M shape), `config_20b.yaml`
and `config_baseline_feedforward.yaml` remain. The A/B baseline is deliberately
kept alongside the 164M arm so both train together, and it still differs from
`config.yaml` by exactly one key (`raslila.depth_sweeps`).

`scripts/verify_20b_groundwork.py` used to read `config_3m.yaml` for its
activation-checkpoint equivalence check; it now inlines a tiny config, so the
script has no dependency on any config file.

**3. Parameter counts after the vocab change.** The vocab delta is
`vocab x (hidden + 1)`: `token_embedding.weight` is `vocab x hidden` (tied into
`output_proj`), and `output_proj.bias` is `vocab`. All counts meta-device, zero
allocation:

| config | before (16,384) | after (199,995) | embedding share |
|---|---|---|---|
| `config_50m.yaml` | 68,903,971 | **92,589,790** | 27.6% |
| `config.yaml` | 212,554,371 | **259,742,398** | 19.7% |
| `config_baseline_feedforward.yaml` | 212,554,371 | **259,742,398** | 19.7% |
| `config_20b.yaml` | 20,773,012,995 | **20,961,214,270** | 1.0% |

The "50M" and "164M" filenames are now labels, not sizes. Worth flagging: at
hidden 128 the embedding is **27.6% of the model**, which is a lot of parameters
sitting in a table rather than in the grid. That was the user's call and it is a
reasonable one at 200k -- it is what GPT-4o, Claude and every production model
of this scale do -- but if the 92.6M arm underperforms, the vocab is the first
thing to ablate.

**4. 20B memory consequence.** 20,961,214,270 x 16 bytes = 335.4 GB of training
state, 83.8 GB/rank over 4-way FSDP -- **3.8 GB over the 80 GB A100**. It no
longer fits as written. `node_depth: 11` gives 17,820,462,270 params = 71.3
GB/rank with ~8.7 GB headroom and is the recommended shape. See the
`config_20b.yaml` header.

**5. Token cache dtype.** vocab > 65,535 forces **uint32**, so the 1.5B-token
cache is **6.0 GB on disk, not 3.0 GB**. Peak build RAM is ~12 GB. `make_batch`
casts to `int64` before indexing, so nothing downstream sees the dtype.

**Still not measured:** whether brain-like wiring beats the original on loss,
and whether 200k vocab beats 16k on loss. Both are priors. The vocab one is a
stronger prior (every production model uses a large subword vocab), but it is
still a prior until a paired run says otherwise.
