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

- **Model**: **5 layers × 5×5 2D sheets = 125 blocks**, **1,114 edges** (brain-like
  small-world wiring), **212.6M params** (`config.yaml`)
- **Corpus**: HuggingFace `Skylion007/openwebtext`, **1.5B tokens**, downloaded and
  tokenized automatically on first run — the old 85.2M-token Gutenberg corpus is gone
- **Loss**: historical CPU figure from an older 54M setup; not comparable to the current tokenizer and wiring.
- **Verdict**: architecture 8/10, current model 3/10 (undertrained) — see [05](05_Scorecards.md)

> **Two changes landed 2026-10** and both invalidate earlier checkpoints:
> the corpus swapped to openwebtext, and the wiring went brain-like
> (`diagonal: true`, `long_range_prob: 0.3`, `max_long_range: 8` — 618 → 1,114
> edges). Details in [01](01_Architecture.md) and [04](04_Training_and_Data.md).

## What is still missing

The spatial design is in good shape. What Raslila lacks is the brain's
*organisation over time*: there is no persistent state between tokens, no
inhibition (all 125 blocks excite each other, never compete), one memory system
instead of two, no replay, no thalamic gating of input, and 5 structurally
identical layers. Full audit, including five places where our own brain
analogies do not hold up, in [07](07_Brain_Gap_Analysis.md). Top of the roadmap:
persistent per-block state, inhibitory edges, and experience replay.
