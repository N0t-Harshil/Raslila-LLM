# What the research says, and Raslila v4

Written 2026-11-09, after Raslila v2 and v3 both lost to the baseline by ~0.2
val loss at 50m and 125m (docs/09). Goal: keep the 3D thinking model, and make
it better than a standard transformer **at the same size**.

## Our results so far

| size | baseline | v2 (fixed averaging) | v3 (learned routing, d/4 channel) |
|---|---|---|---|
| 50m | **3.140** | 3.331 | 3.368 |
| 125m | **2.679** | 2.802 | 2.875 |

The 2nd thinking loop never helped (slightly worse in all four runs).

## Papers that matter for us

| paper | what they built | what they found | lesson for Raslila |
|---|---|---|---|
| **Recurrent depth / Huginn** (Geiping et al. 2025, [arXiv 2502.05171](https://arxiv.org/abs/2502.05171)) | 3.5B model: a few normal layers, then one **full-width** block looped many times, then a few layers out. Trained on 800B tokens with a *random* number of loops per step | Accuracy on reasoning tasks rises with more loops at test time. Others found the gains on language modest ([2507.02199](https://arxiv.org/abs/2507.02199)) | Loop a **full-width** core, train with random loop counts, and supervise only the final answer |
| **Looped transformers** (Saunshi et al., ICLR 2025, [2502.17416](https://arxiv.org/abs/2502.17416)) | k layers looped L times | ≈ a kL-layer model on reasoning (addition, multi-hop, math); **worse perplexity** at equal FLOPs, but competitive on math and code | Loops buy **reasoning**, not memorized knowledge. Judging us on val loss alone hides the benefit |
| **Mixture-of-Recursions** (Bae et al., NeurIPS 2025, [2507.10524](https://arxiv.org/abs/2507.10524)) | Shared layers reused; a small router gives **each token its own number of loops** | At 135M–1.7B: lower perplexity than a vanilla transformer at equal training FLOPs, with fewer parameters and higher throughput | Per-token "think harder on hard words" is the documented way to **win at small size** |
| **Mixture-of-Depths** (Raposo et al. 2024, [2404.02258](https://arxiv.org/abs/2404.02258)) | Router picks which tokens a layer processes; the rest skip it | Matches or beats isoFLOP baselines; learned routing ≫ random routing | Routing should decide *how much compute a token gets*, not just carry messages |
| **HRM** (Wang et al. 2025, [2506.21734](https://arxiv.org/abs/2506.21734)) and the [ARC Prize analysis](https://arcprize.org/blog/hrm-analysis) | Brain-inspired 27M model, fast and slow modules | The **outer refine loop** (predict, "am I done?", refine) drove the gains; the brain-like hierarchy itself barely mattered against a same-size transformer | Brain-like structure alone doesn't win; **iterative refinement with deep supervision** does |
| **TRM** (Jolicoeur-Martineau 2025, [2510.04871](https://arxiv.org/abs/2510.04871)) | One 2-layer, 7M network applied recursively | 45% ARC-AGI-1 (HRM: 40%) | Small model + many refinement steps can beat bigger models on reasoning (puzzles, heavy data augmentation; not language) |
| **TopoLM** (Rathi et al., ICLR 2025 oral, [2410.11516](https://arxiv.org/abs/2410.11516)) | Transformer units placed on a 2D sheet + a **spatial smoothness loss** | Brain-like language regions emerge, matching fMRI data, at "virtually no cost" to performance | The honest way to make the cube *brain-like*: lay neurons out in space and let regions form, without cutting the model's width |
| **Structured / block-diagonal layers** ([2406.06248](https://arxiv.org/abs/2406.06248), [2406.16450](https://arxiv.org/abs/2406.16450)) | Replace dense layers with block-diagonal ("columns") | Can match dense *only* with a dense/global mixing path **and** µP-style learning-rate and init scaling; with dense-model LRs they "significantly underperform" | **This is exactly v2/v3:** isolated narrow blocks, no dense path, baseline LRs |
| **Energy-Based Transformers** (Gladstone et al. 2025, [2507.02092](https://arxiv.org/abs/2507.02092)) | Thinking = iteratively minimising a learned energy | Authors report up to 35% faster scaling and more gain from thinking than Transformer++ (not yet replicated) | Promising but expensive; later option |
| **Continuous Thought Machines** (Sakana, NeurIPS 2025, [2505.05522](https://arxiv.org/abs/2505.05522)) | Neuron-level timing and synchronization | Interesting adaptive "thinking"; results not state of the art | Inspiration, not a base |

## Why v2 and v3 lost (the diagnosis the papers point to)

1. **The cube cut the model's width into small isolated blocks** (27 blocks of
   width 192 at 50m, vs one stream of width 512). Every successful looped model
   (Huginn, MoR, TRM) keeps one full-width state. Block-structured layers only
   work with a dense mixing path and their own LR scaling. We had neither, and
   v3's narrower d/4 channel made it worse, as the data showed.
2. **Every loop was graded as a final answer.** We supervised cycle 1 and
   cycle 2 equally, so cycle 1 learned to finish the job and cycle 2 had
   nothing left to add. Huginn supervises only the end; HRM/TRM supervise
   *successive refinements* of the same answer.
3. **Two loops is almost no thinking.** Huginn samples dozens; MoR lets each
   token choose.
4. **We only measured perplexity**, where looped models are known to look
   worse. Their gains show up on reasoning, and in **accuracy vs. thinking
   time**.

## Raslila v4: the 3D cortex that thinks

Keep everything in the story (3D brain, regions, deciding where to go, thinking
in loops); change how it is built so it no longer fights the research.

```
tokens -> PRELUDE (2 normal full-width layers)
       -> CORTEX CUBE, looped:  state_{t+1} = Cube(state_t + input)
            * one full-width stream (no cutting into tiny towers)
            * the stream's neurons are laid out in 3D: [sheets, rows, cols];
              each sheet = attention across the whole width (global, like
              long-range cortical links) + a column-structured feed-forward
              (each grid column has its own MLP: local cortical columns)
              with a dense mixing path and µP-scaled learning rates
            * a spatial smoothness loss (TopoLM) so neighbouring columns
              specialise together: brain-like regions emerge in the cube
            * per-token router (MoR): every token decides how many loops it
              needs -> easy words exit early, hard ones keep thinking
       -> CODA (2 normal layers) -> next token
```

Training changes, each taken from a paper above:
- random loop count per step (e.g. 1–8, mean ~4), input re-injected every loop (Huginn)
- loss on the final answer, plus a light "refine" loss on intermediate loops (HRM/TRM deep supervision)
- backprop through the last few loops only, which keeps memory flat however long it thinks (Huginn)

### What we claim, and how it is measured
- **Same parameters as the baseline**, so "better at a smaller size" means
  matching or beating a larger baseline with the same or fewer parameters.
- Report FLOPs honestly: thinking costs compute, which is the whole point.
- New headline chart: **accuracy on gsm8k / arc / hellaswag vs. number of
  thinking loops**. A standard transformer is a flat line; v4 should rise.
- Decision rules stay: no 350m/1b until 50m and 125m show a real win.

### Experiments, at 50m (~2–3 h each on 3 GPUs), in order
1. **v4-core**: prelude + full-width looped cube + coda, random loops, final-only loss
2. \+ column-structured feed-forward (the cube's columns) with µP scaling
3. \+ per-token router (adaptive loops)
4. \+ topographic loss (brain-like regions; checks that it is ~free)

Each is compared with the existing baseline_50m. Anything that loses is dropped
before the next step. The winner goes to 125m.

### Honest risks
- On perplexity, looped models usually tie or lose at equal FLOPs (Saunshi).
  The win to aim for is equal **parameters** plus reasoning that improves with thinking.
- HRM/TRM wins are on puzzles with heavy augmentation, not language.
- MoR's results are the authors' own at ≤1.7B; we would be replicating them.
