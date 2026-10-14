# 00 — Index

Documentation vault for **Raslila**, a brain-inspired deep-grid transformer.
Original quantum research notes (15 files) are archived untouched in `../docs_original/`.

## The vault

| Doc | Covers |
|-----|--------|
| [README](README.md) | Entry point + one-paragraph summary |
| [01 Architecture](01_Architecture.md) | The model layer by layer, with the evidence for each choice |
| [02 Research Log](02_Research_Log.md) | Every hypothesis tested with measured results, including failures |
| [03 Brain Principles](03_Brain_Principles.md) | Neuroscience ↔ ML mapping, with sources, and where the analogy breaks |
| [04 Training and Data](04_Training_and_Data.md) | Corpus, tokenization, schedules, CPU/GPU/TPU runbooks |
| [05 Scorecards](05_Scorecards.md) | Per-component 0–10 scores and the honest verdict |
| [06 Known Limitations](06_Known_Limitations.md) | What doesn't work; what 5T would actually require |
| [**07 Brain Gap Analysis**](07_Brain_Gap_Analysis.md) | **What a real brain has that Raslila doesn't — 13 gaps, bad analogies, and a prioritised roadmap** |
| [**08 Runbook**](08_Runbook_Bidirectional_Test.md) | **Commands for the current experiment: train A/B, compare, sample, then activate the next candidate one at a time** |

Legacy top-level docs are retained and still accurate in spirit: `../README.md`,
`../PRE_TRAINING_DESIGN.md`, `../SOTA_COMPARISON.md`, `../VERIFICATION_REPORT.md`.

## Current state

| | |
|---|---|
| Model | Raslila — **5 layers × 5×5 2D sheets = 125 blocks**, hidden 256, **163,605,123 params**, 618 edges |
| Structure | a conventional LLM is the degenerate 1D case; Raslila restores the 2D sheet per layer and stacks sheets in depth |
| Wiring | **verified small-world**: `diagonal: false` (8-neighbour sheets) + `long_range_prob: 0.3` / `max_long_range: 8` (sparse long-range tracts). Was 618 edges / 4-neighbour; now 618. Costs +34% forward, +23% params |
| Active experiment | **`depth_sweeps: 2`** — the one variable under test. Feedback edges existed in the graph but were never traversed; the sweep makes flow bidirectional. 2× compute, 0 extra params |
| Built but off | `inhibit`, `state_dim`, `replay_prob` — verified structurally, no measured loss gain, activated one at a time only if they win |
| Objective | next-token prediction, **BPE 199,995 tokens** (5.43 chars/token) |
| Corpus | HuggingFace **`Skylion007/openwebtext`**, **1.5B tokens cached**, auto-fetched on first run. Replaces 448 Gutenberg books / 85.2M tokens (deleted) |
| Proven wins | binding (−0.041), multidirectional cascades (−0.026), modern blocks (−0.27) |
| Growth | cortical growth implemented (adds whole layers, clone-based birth) |
| Verdict | architecture 8/10, current model 3/10 — under-trained, not under-designed |

> **Deep grid, not a chain.** `depth_layers` = sequential layers with *distinct
> weights*; `grid_h`×`grid_w` = the 2D sheet inside one layer; blocks connect
> laterally within a sheet and vertically between sheets, with feedback.

## Current experiment

Everything except one thing is off. `config.yaml` runs `depth_sweeps: 2` and
nothing else new — inhibition, persistent state and replay are implemented,
verified structurally, and **disabled**, because none has beaten baseline and
turning several on at once would make the result unattributable.

That one thing is the fix for a claim that turned out to be false: Raslila's
feedback edges were in the adjacency matrix but **never executed**. The forward
pass walked layers strictly bottom-up, so L1→L0 fired against unprocessed state.
"Symmetric connections" was true on paper and false in the model.

The comparison to beat: `depth_sweeps: 1` (feedforward), same parameters, same
data. Cost of the change is 2× compute and zero parameters — if it does not win,
that 2× is the argument for reverting to 1.

## Where the gaps are

Raslila reproduces the brain's **spatial** organisation and not its temporal or
learning organisation. The largest remaining gaps are all about *time* and *sign*,
not geometry — see [07 Brain Gap Analysis](07_Brain_Gap_Analysis.md):

| Gap | Brain | Raslila |
|---|---|---|
| Persistent state over time | membrane integration, ~150 ms recognition | none — attention window only |
| Inhibition | ~20% interneurons, lateral competition | none, all traffic excitatory |
| Two memory systems | hippocampus (fast) + neocortex (slow) | one, gradient-trained |
| Replay / consolidation | sleep replay at 10–20× | none |
| Input gating | thalamic relay | none, input hits all 125 blocks |
| Layer heterogeneity | 6 layers, distinct cell types | 5 homogeneous copies |

Top of the roadmap: **persistent per-block state**, **inhibitory edges**, and
**experience replay**.

## The thesis in one paragraph

Modern LLMs process a token through a **one-way 1D chain**: `input→1→2→3→output`.
Brains do not. They are **sheets of regions**: each layer is a 2D surface where
regions connect laterally, sheets stack in depth and feed into each other (with
feedback — the cortex is not a feedforward stack), and information is *bound*:
separately-represented facts get combined into genuinely new structures rather
than averaged away. Raslila reproduces all of it. A conventional LLM is the
degenerate case of this architecture — a 1D chain with no lateral connectivity
and no binding — which is why the 3D/sheet structure is the thesis, not
decoration: binding theory requires exactly the separated, persistent streams a
2D sheet provides, and a 1D transformer structurally cannot offer them.

## Reproduce the key claims

```bash
# the architecture beats an equal-parameter GPT stack
python scripts/benchmark_gpt_vs_raslila.py --steps 200

# the modern block recipe earns its place
python scripts/benchmark_gpt_vs_raslila.py --mode legacy_vs_modern --steps 200

# multidirectional beats more passes
python scripts/benchmark_gpt_vs_raslila.py --mode recur --steps 150

# binding is the largest single gain
python scripts/benchmark_gpt_vs_raslila.py --mode bind --steps 150
```
