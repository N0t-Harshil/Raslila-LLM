# Raslila — Documentation Vault

> Living documentation for Raslila, a brain-inspired 3D-grid transformer.
> The original 15 quantum research notes are preserved untouched in `docs_original/`.

## Read in this order

| # | Document | What it covers |
|---|----------|----------------|
| 00 | [00_Index.md](00_Index.md) | Map of the whole vault + current status |
| 01 | [01_Architecture.md](01_Architecture.md) | The model, layer by layer, with the research behind each choice |
| 02 | [02_Research_Log.md](02_Research_Log.md) | Every hypothesis tested, with measured results (including failures) |
| 03 | [03_Brain_Principles.md](03_Brain_Principles.md) | The neuroscience ↔ ML mapping driving the design |
| 04 | [04_Training_and_Data.md](04_Training_and_Data.md) | Corpus, tokenization, schedules, hardware runbooks |
| 05 | [05_Scorecards.md](05_Scorecards.md) | Per-component scores and the honest verdict |
| 06 | [06_Known_Limitations.md](06_Known_Limitations.md) | What does not work, and what 5T would actually require |
| **07** | [**07_Brain_Gap_Analysis.md**](07_Brain_Gap_Analysis.md) | **What a real brain has that Raslila doesn't: 13 gaps, the bad analogies, and a prioritised roadmap** |
| **08** | [**08_Runbook_Bidirectional_Test.md**](08_Runbook_Bidirectional_Test.md) | **Exact commands for the current experiment: train A/B, compare, sample, then activate the next candidate one at a time** |

## The one-paragraph version

Raslila replaces the 1D chain of a conventional LLM with a **deep grid**: each of
5 layers is a 2D sheet of 25 transformer blocks that connect laterally, and the
sheets stack in depth feeding into each other — with feedback, because the cortex
is not a feedforward stack. Each layer owns distinct weights, so it is a
genuinely deep network. Blocks hold persistent, *separate* state streams, which
is what binding theory requires and what a 1D transformer structurally cannot
provide; they are bound multiplicatively into emergent composites and pooled by
learned attention rather than averaged away. As a research architecture it beats
an equal-parameter GPT stack; as a language model it is still under-trained, and
the bottleneck is training budget, not design.

A conventional LLM is the degenerate case of this architecture — one flat chain,
no lateral connectivity, no binding.

## Current status at a glance

- **Model**: **5 layers × 5×5 2D sheets = 125 blocks**, **618 edges**, **163.6M
  params** (`config.yaml`, the depth_sweeps A/B pair). `config_50m.yaml` (68.9M)
  and `config_20b.yaml` use the brain-like small-world wiring
- **Corpus**: HuggingFace `HuggingFaceFW/fineweb-edu` (`sample-10BT`), **1.5B
  tokens**, downloaded, cleaned, deduplicated and tokenized automatically on first
  run, with `<|endoftext|>` between documents; 16,384-token BPE
- **Loss**: historical CPU figure from an older 54M setup; not comparable to the current tokenizer and wiring.
- **Verdict**: architecture 8/10, current model 3/10 (undertrained) — see [05](05_Scorecards.md)

> **2026-10 changes invalidate earlier checkpoints and loss numbers:** the
> corpus is now FineWeb-Edu with document separators and a re-fitted tokenizer,
> and the training recipe changed (cosine LR, no decay on norms/biases). The
> brain-like wiring applies to the 50M/20B configs only; the A/B pair keeps the
> 618-edge graph on purpose. Details in [01](01_Architecture.md),
> [04](04_Training_and_Data.md) and the log in [02](02_Research_Log.md).

## What is still missing

The spatial design is in good shape. What Raslila lacks is the brain's
*organisation over time*: there is no persistent state between tokens, no
inhibition (all 125 blocks excite each other, never compete), one memory system
instead of two, no replay, no thalamic gating of input, and 5 structurally
identical layers. Full audit, including five places where our own brain
analogies do not hold up, in [07](07_Brain_Gap_Analysis.md). Top of the roadmap:
persistent per-block state, inhibitory edges, and experience replay.
