# What the research says, and Raslila v4

Rewritten 2026-11-09 after reading the full papers (PDFs and extracted text in
`research/papers/`, not committed). An earlier version of this file was based on
abstracts and summaries; **three of its claims were wrong** and are corrected
below (marked ⚠).

Goal: keep the 3D thinking model, and make it better than a standard
transformer **at the same size**.

## Our results so far (docs/09)

| size | baseline | v2 (fixed averaging) | v3 (learned routing, d/4 channel) |
|---|---|---|---|
| 50m | **3.140** | 3.331 | 3.368 |
| 125m | **2.679** | 2.802 | 2.875 |

Both Raslila versions have the same parameters as the baseline and ~2x its
compute, and still lose. The 2nd thinking loop never helped.

## What each paper actually shows

### Looped / recurrent-depth language models

**Huginn: recurrent depth** (Geiping et al. 2025, arXiv 2502.05171)
- Design: prelude (2 layers) → core (4 layers, *full width*) looped r times → coda (2 layers).
  - The input embedding is re-injected every loop (concatenation + adapter worked best at scale).
  - The start state is random noise.
  - Loop count per step is random (log-normal Poisson, mean 32, heavy tail).
  - Backprop goes through the last 8 loops only.
  - **Only the final output gets a loss.**
- Scale: 3.5B parameters, 800B tokens, mostly code and math data.
- Main result (Table 4, same data, same 180B tokens, *equal parameters*):
  - Recurrent r=32 beats the fixed-depth twin: ARC-C 29.2 vs 27.0, GSM8K-CoT 9.0 vs 1.8, HellaSwag 48.8 vs 37.3.
  - With r=1 it is *far worse* (HellaSwag 29.2). The model only works when it loops.
- What it is not: this is **not equal compute**. At r=32 it spends roughly the FLOPs of a 32B model.
- Failures at scale:
  - The first two runs failed. In one, all tokens collapsed to the same hidden state. In the other, the core learned to ignore the loop, so 1 and 32 loops gave the same perplexity.
  - What fixed it: "sandwich" norms (norms before *and* after each sublayer), a learned adapter, and a much lower peak LR (4e-5).
- Facts and reasoning: the paper itself says the model has "fewer facts memorized" but "more capacity to reason about its context". Given the needed fact in the prompt, it closes most of its gap to bigger models.

**Independent analysis of Huginn** (Lu et al. 2025, arXiv 2507.02199)
- When the model must answer GSM8K directly, without writing out steps, more loops help only a little: 3.1% at r=4 → 4.9% at r=32, then flat.
- With written chain-of-thought it reaches 25–38%.
- They found no evidence of step-by-step "latent chain of thought" inside the loops.
- **Lesson:** loops alone do not make a small model reason like a person on math. Loops plus written reasoning is a different matter.

**Looped transformers for reasoning** (Saunshi et al., ICLR 2025, arXiv 2502.17416)
- Synthetic tasks (addition, multi-hop lookup, synthetic math): a k-layer model looped L times nearly matches a kL-layer model, with L× fewer parameters.
- Real language modelling, 1B models on 250B tokens:
  - Looped models have **worse perplexity** than the deeper baseline. They close only 34–50% of the gap from a same-parameter shallow model.
  - **On reasoning they do far better than their perplexity predicts.** A 12-layer model looped twice beats the 24-layer model on math word problems (34.3 vs 29.3) with half the parameters. On "reasoning primitives" (variable-assignment chains) every looped variant beats the 24-layer baseline.
  - On closed-book QA (memorised facts) the gain is no larger than perplexity predicts.
- Accuracy grows like log(effective depth). More loops keep helping, and help reasoning most.
- "Middle looping" (unique first and last layers, looped middle) has better perplexity than looping everything.
- A cheaper trick, a loss that keeps successive layer blocks similar, gives the reasoning bias at no perplexity cost.

**Mixture-of-Recursions** (Bae et al., NeurIPS 2025, arXiv 2507.10524)
- Shared middle layers. A small router decides, per token, how many loops it gets (expert-choice routing with an auxiliary loss, sigmoid, linear router).
- ⚠ **At the 135M scale it loses to the vanilla transformer.** It matches or beats vanilla only from about 360M up.
- Its isoFLOP wins come mostly from being cheaper per token, so it trains on more tokens for the same budget.
- With Middle-Cycle sharing (unique first and last layers) it was the best variant.
- **Lesson:** per-token adaptive depth is not a lever at our 50m/125m sizes. Revisit at ≥350m.

**Mixture-of-Depths** (Raposo et al. 2024, arXiv 2404.02258)
- Gains are modest: up to 1.5% on log-probability at equal FLOPs, or the same loss with fewer FLOPs per step.
- Only works when routed blocks alternate with normal full blocks.
- Random routing is much worse than learned routing.

**Relaxed Recursive Transformers** (Bae et al., ICLR 2025, arXiv 2410.20672)
- Converts an already-trained model into a looped one (plus per-loop LoRA).
- Strong results, but they depend on initialising from a pretrained model, not training from scratch. Useful later for scaling, not for our study.

### Brain-inspired reasoning models

**HRM** (Wang et al. 2025, arXiv 2506.21734) and the ARC Prize analysis
- 27M parameters, two recurrent modules (fast and slow).
- Encoder-only, trained per task on about 1000 puzzles with heavy augmentation. It is not a language model.
- Real brain-like finding: the slow module's activity becomes about 3× higher-dimensional than the fast one's (participation ratio 90 vs 30). The ratio matches the mouse cortex (~2.25×), and it emerges only with training.
- The ARC Prize team found the "outer loop" (deep supervision with refinement) drove the gains. The hierarchy itself was about level with a same-size transformer.

**TRM** (Jolicoeur-Martineau 2025, arXiv 2510.04871)
- One 2-layer network, recursed. Beats HRM: Sudoku 87% vs 55%, ARC-1 45% vs 40%.
- Ablations:
  - Deep supervision doubled ARC accuracy (19→39%); the recursion inside each step mattered little.
  - Backprop through the full last recursion beats HRM's 1-step gradient (87% vs 57%).
  - **Splitting the state into more parts hurt** (7 parts: 78% vs 2 parts: 87%).
  - **Two separate networks were worse than one shared** (82% vs 87%).
  - MoE "decreased generalization massively".
- Small data, puzzles, not generative.

**TopoLM** (Rathi et al., ICLR 2025 oral, arXiv 2410.11516)
- A GPT-2-small whose units are laid out on a 2D grid, plus a spatial-smoothness loss (nearby units should respond alike).
- Result: verb/noun and language-selective clusters that match fMRI data.
- ⚠ **Not free:**
  - Validation loss 3.075 vs 2.966 (+0.11, at the heavily tuned weight λ=2.5).
  - BLiMP 0.71 vs 0.76, Brain-Score 0.78 vs 0.80.
  - GLUE 0.68 vs 0.65 (better).
- A model with *local wiring* instead (Topoformer) formed clusters that did **not** match the brain.

**Continuous Thought Machines** (Sakana, NeurIPS 2025, arXiv 2505.05522)
- Neuron-level timing. Mazes and ImageNet. Not applied to language yet.

**Energy-Based Transformers** (Gladstone et al. 2025, arXiv 2507.02092)
- Thinking = gradient descent on a learned energy.
- Claims a faster *scaling rate* than Transformer++.
- At their sizes absolute perplexity is worse: 33.4 vs 31.4.
- Training needs second-order gradients at roughly 2× sequence length, and is sensitive to hyperparameters.
- Too expensive and risky for our budget now.

### Splitting width into blocks

**Compute Better Spent** (Qiu et al., ICML 2024, arXiv 2406.06248)
- ⚠ The structures that match or beat dense layers (Monarch, BTT) are **not isolated blocks**. They put a permutation between two block layers, so every block mixes with every other block in every layer.
- On GPT-2 language modelling, BTT only *ties* dense once the output-head compute is excluded.
- Narrower blocks need a **higher** learning rate. Under µP the LR scales with 1/fan-in, so a 192-wide block wants about 512/192 ≈ 2.7× the LR of a 512-wide layer. With the dense model's LR, structured layers "significantly underperform".
- Weight normalisation was needed to stop activations growing without bound.

## What went wrong in v2/v3 (diagnosis, now evidence-based)

1. **Isolated narrow blocks.** v2/v3 cut the 512-wide stream into 27 blocks of 192 that talk only through averaged or low-rank messages. That is a block-diagonal model without a mixing path. The structured-layer paper shows that loses unless every layer mixes all blocks. TRM found splitting the state into more pieces hurts.
2. **Wrong learning rate for narrow blocks.** We used the dense model's LR. µP says ~2.7× higher at 50m.
3. **Every loop was graded as a final answer.** Huginn grades only the end, with random loop counts. TRM/HRM grade successive refinements with the state detached between steps. We did neither, and loop 2 never helped.
4. **Too few loops**: 1–2, versus Huginn's mean of 32.
5. **Only perplexity was measured.** Every looped paper finds the benefit on reasoning, not perplexity.

## The honest bottom line

- No published loop or brain design beats a standard transformer on **raw language
  loss at equal compute at 50m–125m**:
  - MoR loses at 135M.
  - Looped models lose on perplexity.
  - TopoLM costs 0.11.
  - EBT is worse in absolute terms.
- What *is* well supported:
  - **Equal parameters:** a looped full-width model beats a same-size standard model on every metric (Saunshi, Huginn), because it uses more compute per parameter.
  - **Reasoning:** at the same perplexity or fewer parameters, looped models do much better on math and multi-step reasoning, and keep improving with more loops (log-depth scaling).
- So "better at a smaller size" is achievable if stated precisely:
  - **same parameter count, better on reasoning, improving with thinking time**
  - with the extra compute reported openly.
- "Thinks like a human" needs care. Latent loops alone gave Huginn almost nothing on GSM8K without written reasoning.

## Raslila v4: 3D cortex with a recurrent core

```
tokens -> PRELUDE: 2 normal full-width layers (unique weights)
       -> CORTEX: a full-width core (e.g. 4 layers) looped r times
            state_{t+1} = Core( Adapter([state_t ; input]) )
            * the core's d neurons form a 3D cube (512 = 8x8x8 at 50m):
              its "3D brain" layout, used for analysis, and optionally an
              ablated 3D spatial-smoothness loss (TopoLM, small weight; costs
              must be measured, not assumed)
            * full attention and full MLP across the whole width every loop:
              no isolated blocks
       -> CODA: 2 normal layers -> next token
```

Training recipe, each piece from a paper:
- random loop count per step (log-normal Poisson; mean ~6 at 50m), random start state, input re-injected via concat + adapter (Huginn)
- loss on the final output only; backprop through the last k loops (k = 4–8) (Huginn)
- sandwich norms and a lower core LR to prevent collapse and the core ignoring the loop. Monitor the token-correlation and "does r change the loss" signals Huginn used to catch failed runs.
- unique first and last layers, looped middle (Middle-Cycle / middle looping: best in both MoR and Saunshi)

Not now: per-token adaptive depth (loses at our sizes per MoR; revisit at
≥350m), energy-based thinking (cost), isolated block grids (refuted).

### What we measure (and claim)
1. **Equal parameters** vs baseline: val loss and all benchmarks. Literature predicts a win.
2. **Equal compute** vs a deeper baseline: reported honestly, expected to lose on perplexity.
3. **Reasoning that scales with thinking:** accuracy vs loop count (1, 2, 4, 8, 16, 32) on
   - new cheap synthetic reasoning tests that small models can show: variable-assignment chains, multi-hop lookup, multi-digit addition (Saunshi's "reasoning primitives")
   - plus arc/gsm8k
4. **Reasoning at equal perplexity** (Saunshi's isoplot): downstream accuracy vs val loss over training, Raslila vs baseline.
5. **Brain-likeness, measured not claimed:**
   - dimensionality hierarchy across the cube (HRM's participation ratio)
   - clustering of the 3D cube's units (Moran's I, TopoLM)

### Implemented (2026-11-09): tier 150m
- Model: `src/raslila/cortex.py`.
- Configs:
  - `configs/scaling/raslila_150m.yaml`: 162M parameters. d 1024, prelude 2 + core 6 + coda 2, cube 8x8x16, mean 4 loops, backprop through the last 4, headline at 8 loops.
  - `configs/scaling/baseline_150m.yaml`: 160M parameters, 10 layers x 1024.
- Same tokens (3.2B), LR (5e-4) and batch for both.
- v4 spends ~2.2x training compute per token, reported in the head-to-head table.
- Variant: `configs/scaling/variants/raslila_150m_topo.yaml` (3D brain-region loss on, weight 0.5).
- Evaluation:
  - `benchmark.py` scores v4 at 1/2/4/8/16 loops and adds `var_chain` (0–3 hops) and `add2`.
  - `scaling_report.py` draws `thinking.png`.
  - `brain_metrics.py` measures the dimensionality hierarchy and 3D clustering.

Decision: if raslila_150m does not beat baseline_150m at equal parameters (val
loss and the reasoning tasks), the literature-backed premise fails at our scale
and we stop before going bigger. Next checks if it is close: core learning rate
(Huginn needed a lower one), more loops.

### Honest risks
- Huginn's training was fragile (2 of 3 runs failed) at 3.5B; small scale was reported as robust.
- Reasoning benchmarks at 50m are near the floor. That's why the synthetic reasoning tests are needed.
- Equal-parameter wins cost extra compute at inference. That is the trade, and it must be stated.
