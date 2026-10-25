# Raslila: Brain-Inspired Deep-Grid Transformer

> Original quantum research notes (15 files) are preserved untouched in `docs_original/`.
> Raslila has since evolved from a 6-qubit quantum proof-of-concept into a
> **purely classical, brain-inspired deep grid** with edge networks that compute,
> trainable end-to-end on GPU.

## What this is

Raslila is a **classical neural architecture** that arranges transformer blocks
into a **deep grid**: each of 5 layers is a 2D sheet of 5×5 blocks (like a
cortical sheet), blocks connect laterally within a sheet and vertically between
sheets, and **edges that are neural networks** (not just weights) process the
messages between them. It keeps the brain-inspired ideas from the original
quantum research — multi-perspective reasoning, edge-centric computation,
interference dynamics — but is trainable on standard GPUs.

A conventional LLM is the **degenerate 1D case** of this design: one flat chain
with no lateral connectivity and no binding.

**What is still missing.** The spatial design is in good shape; the brain's
*organisation over time* is not reproduced. Audited in
[`docs/07_Brain_Gap_Analysis.md`](docs/07_Brain_Gap_Analysis.md) — no persistent
state between tokens, no inhibition, one memory system instead of two, no replay,
no input gating, 5 identical layers.

**What is being tested right now: exactly one thing.** `depth_sweeps: 2`.
Raslila's feedback edges were in the adjacency matrix but **never executed** —
the forward pass walked layers strictly bottom-up, so the L+1→L edges fired
against unprocessed state. "Bidirectional connections" was true on paper and false
in the model. The sweep alternates traversal direction and costs **2× compute and
zero extra parameters** (163,605,123 either way), so any loss difference is
attributable to the sweep alone.

Inhibition, persistent state and replay are all implemented and structurally
verified, and all **off**. Each gets its own training run and is kept only if it
beats this baseline — enabling several at once makes the result unattributable,
which is how a confident 3-seed "win" got erased by a 6-seed rerun.

**Recent changes (2026-10) — these invalidate every earlier checkpoint:**

1. **Corpus is HuggingFace FineWeb-Edu (`sample-10BT`)**, 1.5B tokens,
   downloaded, cleaned and tokenized automatically at training start into
   `data/fineweb_edu_tokens.npy` (~3 GB, uint16). It replaced openwebtext:
   FineWeb-Edu is filtered for educational quality and deduplicated. The
   pipeline also normalises whitespace, drops stub/junk documents and exact
   duplicates, shuffles the stream, and puts `<|endoftext|>` between documents
   so the model sees document boundaries. 2M tokens are held out for val loss.
   The Gutenberg corpus is kept only as the `training.data_dir` fallback.
   Details: [`src/raslila/hf_dataset.py`](src/raslila/hf_dataset.py).
2. **Wiring depends on the config.** `config_50m.yaml` and `config_20b.yaml` use
   brain-like small-world wiring (`diagonal: true`, `long_range_prob: 0.3`,
   `max_long_range: 8`). The `config.yaml` / `config_baseline_feedforward.yaml`
   A/B pair deliberately keeps the original 618-edge graph (`false / 0.1 / 3`)
   so `depth_sweeps` is the only variable; brain-like wiring is its own later
   experiment. `scripts/check_configs.py` enforces both.
3. **16,384-token BPE, re-fitted on FineWeb-Edu** on the first training run
   (with NFC normalisation and `<|pad|>`=0, `<|endoftext|>`=1). A 199,995-token
   vocab was considered but not adopted: at hidden 256 it puts ~47M parameters
   into the embedding and a 200k-way softmax on every token.
4. **Training recipe**: cosine LR decay to `min_lr`, AdamW betas (0.9, 0.95),
   no weight decay on norms/biases/scalars, and edge networks run as one
   batched matmul per block (same parameters and checkpoint keys as before).

A checkpoint taken before any of these **will not load**: the edge key set and
the embedding table both differ. Start fresh.

**Recent pivot (2026-09):** The active model is the classical **Raslila**
transformer (`src/raslila/raslila.py`) — *no quantum computation*. The original
quantum 6-qubit simulation remains as a verified research foundation in
`docs_original/` and the quantum subsystem modules.

**Full documentation lives in [`docs/`](docs/README.md)** — architecture with
per-layer evidence, a research log with measured results (including failures),
the neuroscience↔ML mapping, training/data runbooks, scorecards, and an honest
limitations document.

## Project layout

```
Raslila\
  docs_original\        # all 15 original quantum research notes (archived, untouched)
  src\raslila\          # the package
    raslila.py          # ★ ACTIVE Raslila transformer (EdgeNetwork, TransformerNode, Raslila)
    hf_dataset.py       # ★ HuggingFace corpus fetch + BPE tokenize + cache
    model.py            # quantum RaslilaModel (6-qubit, verified)
    model_enhanced.py   # enhanced quantum model (annealing, sparse Trotter, language)
    gpu_hamiltonian.py  # GPU Hamiltonian with Krylov subspace evolution
    hamiltonian.py      # XX+YY independent interference (quantum)
    activation.py       # Squash+Sin + tanh activations
    mlp.py              # Trainable MLP (edge couplings)
    qng.py              # Quantum Natural Gradient optimizer
    memory.py           # Topological memory (resonance on J)
    evolution.py        # unitary evolution, NLL loss, QFIM
    language.py         # tokenization / embedding interface
    sparse_trotter.py   # sparse Trotter evolution (>10 qubits)
    parameter_shift.py  # exact parameter-shift gradients
    data.py             # dataloaders
    trainer.py          # training loops
    entanglement.py     # entanglement entropy monitoring
    normalization.py    # Hamiltonian normalization
    annealing.py        # quantum annealing
    pennylane_interface.py  # optional PennyLane backend
    stochastic_parameter_shift.py  # sampled gradients
    classical_baseline.py  # GPT-2-style classical baseline
  scripts\
    train_config.py     # train from YAML config (config.yaml)
    train_2m_gpu.py     # GPU training for 2M model
    train_real.py       # training on curated real text
    micro_real.py / quick_real.py  # small real-text experiments
    test_raslila.py / test_raslila_quick.py / test_raslila_train.py  # Raslila tests
    train_raslila.py      # Raslila trainer (FSDP, DDP, adaptive LR, BPE)
    train_raslila_tpu.py  # same trainer on torch_xla
    get_dataset.py        # ★ download + tokenize a HuggingFace corpus
    run_ab.py             # ★ run the depth_sweeps A/B (both arms, preflighted)
    generate_raslila.py   # text generation from a checkpoint
    raslila_live.py       # ★ always-on interactive session (rests when idle, input mid-answer)
    verify_20b_groundwork.py  # 20B param count, activation ckpt, FSDP equivalence
    benchmark_gpt_vs_raslila.py  # GPT vs Raslila + legacy/modern ablations
  config.yaml           # ★ 163.6M config (edit to scale model up/down)
  config_50m.yaml       # 68.9M production shape (brain-like wiring)
  config_20b.yaml       # 21.0B, FSDP + activation_checkpoint target
  config_baseline_feedforward.yaml  # A/B control for config.yaml (depth_sweeps: 1)
  config_adaptive.yaml  # config.yaml + adaptive depth (think until settled)
  data/tokenizer/       # trained BPE tokenizer (16k; re-fitted on FineWeb-Edu on first run)
  data/gutenberg/       # 17 books, kept only as the benchmark script's default corpus
  docs/                 # ★ full documentation vault
  architecture.png      # architecture diagram
  requirements.txt
  README.md
  PRE_TRAINING_DESIGN.md   # gap analysis + pre-training plan
  VERIFICATION_REPORT.md   # verification results
  SOTA_COMPARISON.md       # component-by-component vs state of the art
```

## The Raslila architecture (`raslila.py`)

> Full layer-by-layer detail with evidence: [`docs/01_Architecture.md`](docs/01_Architecture.md)

**Structure — a deep grid, not a chain.** A conventional LLM is the degenerate 1D case, one flat chain of blocks. Raslila makes each layer a **2D sheet of 25 blocks** connected laterally, and stacks **5 sheets** in depth:

```
   layer 5    [0]-[-]-[-]-[-]-[-]     2D sheet: blocks connect laterally
              [0]-[-]-[-]-[-]-[-]
              [0]-[-]-[-]-[-]-[-]
               |      |      |
   layer 4    [0]-[-]-[-]-[-]-[-]     ...and vertically to the sheet above
              [0]-[-]-[-]-[-]-[-]
               |      |      |
   ...
   layer 1    [0]-[-]-[-]-[-]-[-]     5 layers x 5x5 = 125 blocks
```

This is cortex-like: each layer is a cortical sheet, blocks connect within the sheet, and sheets feed each other in depth. **Each layer owns distinct weights**, so it is a genuinely deep network.

- **Blocks**: Transformer blocks with (col, row, layer) positional identity, so regions specialize
- **Edges**: `EdgeNetwork` MLPs that *process* messages (not just weight them) — **618 connections** in `config.yaml`
- **Connectivity (brain-like small-world, in `config_50m.yaml` / `config_20b.yaml`; the `config.yaml` A/B pair uses 4-neighbour lateral + 10% shortcuts)**: `lateral` (8-neighbour within a sheet, `diagonal: true`) + `vertical` (same cell, layer L+/-1) + **`feedback`** (L+1 -> L, because the cortex is not feedforward) + **30% long-range shortcuts** (`long_range_prob: 0.3`, `max_long_range: 8`). This is the cortex's actual topology: dense local columns plus sparse long-range white-matter tracts, on a lattice that is not Manhattan
- **Bidirectional flow** (`depth_sweeps: 2`, **the active experiment**): a hypothesis travels down and back up, alternating traversal direction. Direction order is deliberate — the readout reads the deepest layer, so the final sweep must *end* there or the refinement is discarded. Costs 2× compute, zero parameters
- **Interference**: learned `alpha`/`beta` blend source/target features on every message
- **Causal masking**: position t only sees positions ≤ t (proper next-token LM)
- **Learned readout**: attention over the deepest sheet's blocks, not a mean
- **Modern block recipe**: pre-norm **RMSNorm**, **RoPE**, **SwiGLU**, **QK-norm**, **tied** head, final norm — measured **−0.27 nats vs the GPT-2-era block** (`--mode legacy_vs_modern`)
- **Multidirectional recurrence**: `immediate_updates` lets a block read neighbours' *fresh* states, so signals cross a sheet in one sweep. Measured **−0.026 nats**, free
- **Binding / conceptual blending** (`bind_roles: 4`): blocks are bound *multiplicatively* to learned role vectors (Tensor-Product style) into an emergent composite set, and the readout attends over that set **instead of collapsing to a weighted mean**. Measured **−0.041 nats** — the largest single gain tested. This is why the 2D sheet matters: binding theory needs separated, persistent streams, which a 1D chain cannot provide
- **Cortical growth** (optional, `grow_every: 0` off): adds whole layers on plateaus, cloned from existing blocks. See [`docs/06_Known_Limitations.md`](docs/06_Known_Limitations.md)

Measured sizes (BPE vocab 16,384):

| Layout | Blocks | Params |
|---|---|---|
| 5 × (5×5), hidden 128, brain-like wiring (`config_50m.yaml`) | 125 | **68.9M** |
| 5 x (5x5), hidden 256, 618 edges (**current**, `config.yaml`) | 125 | **163.6M** |
| 5 × (5×5), hidden 256, node_depth 2 | 125 | 357M |
| 5 × (5×5), hidden 256, node_depth 3 | 125 | 455M |
| 5 × (5×5), hidden 1024, node_depth 11 (`config_20b.yaml`) | 125 | 21.0B |

`node_depth` dominates parameters; **edge parameters scale with block count** - in `config.yaml` (125 blocks, 618 edges) they are **37% of the model** (61.0M of 163.6M), so `edge_hidden_dim` is the clean trim lever.

## Quickstart

**Easiest: `run.py` does everything.** `python run.py` lists the commands.

```bash
python run.py setup                                   # install, show GPU
python run.py check                                   # validate configs
python run.py data                                    # FineWeb-Edu, once (~3 GB)
python run.py smoke                                   # 3-step sanity run
python run.py ab                                      # depth_sweeps A/B (1 GPU per arm)
python run.py compare                                 # A/B verdict
python run.py train --config config_adaptive.yaml     # adaptive-depth model
python run.py eval  --config config_adaptive.yaml     # loss, speed, thinking cycles
python run.py live                                    # always-on chat session
python run.py train --config config.yaml --resume latest   # continue an interrupted run
python run.py all                                     # the whole pipeline in order
```

Every command logs to `logs/<command>_<time>.log`. Over SSH, run long jobs in
`tmux`/`screen`. The scripts below are what `run.py` calls, for finer control.


```powershell
pip install -r requirements.txt

# The corpus downloads itself on the first training run (rank 0 only; the other
# ranks wait, then load the cache). Pre-fetch only to avoid the first-run wait:
python scripts/get_dataset.py --config config.yaml
python scripts/get_dataset.py --config config.yaml --max-tokens 50000000  # small test slice

# Single A100 80GB: just run the trainer. The first run fits the tokenizer on
# FineWeb-Edu, then streams and cleans 1.5B tokens into
# data/fineweb_edu_tokens.npy (~3 GB); later runs reuse both.
python scripts/check_configs.py                 # must print "All active config checks passed."
python scripts/train_raslila.py --steps 3 --batch-size 1 --seq-len 32 --out-dir ck_smoke --save-every 0
python scripts/run_ab.py --steps 20000          # the depth_sweeps A/B (both arms)

# Raslila quick training sanity check (18 regions)
python scripts/test_raslila_quick.py

# Train Raslila from config.yaml (single GPU / CPU)
python scripts/train_raslila.py

# Run the bidirectional A/B: config.yaml (depth_sweeps=2) then the feedforward
# control. Refuses to launch unless the two configs differ by exactly that one
# key -- a drifted A/B is worse than no A/B.
python scripts/run_ab.py --steps 20000
python scripts/run_ab.py --steps 20000 --compare-only   # just print the table

# Multi-GPU: DDP by default; add --fsdp for anything above ~5B params
torchrun --standalone --nproc_per_node=4 scripts/train_raslila.py --config config.yaml
torchrun --standalone --nproc_per_node=4 scripts/train_raslila.py --config config_20b.yaml --fsdp

# Verify the 20B groundwork before committing GPU-days (param count, activation
# checkpointing, FSDP-vs-single-process loss equivalence)
python scripts/verify_20b_groundwork.py

# Generate text from a checkpoint
python scripts/generate_raslila.py --checkpoint checkpoints/raslila/raslila_step1000.pt \
    --prompt "Once upon a time" --tokens 200

# Adaptive depth: train the "think until settled" variant (same 163.6M params)
python scripts/train_raslila.py --config config_adaptive.yaml

# Always-on live session: stays loaded, rests when idle, takes input mid-answer
python scripts/raslila_live.py --checkpoint checkpoints/raslila_adaptive/raslila_step19999.pt \
    --inbox data/live_inbox --offload

# Prove the architecture claims (equal-param GPT, block recipe, recurrence, binding)
python scripts/benchmark_gpt_vs_raslila.py --steps 200
python scripts/benchmark_gpt_vs_raslila.py --mode legacy_vs_modern --steps 200
python scripts/benchmark_gpt_vs_raslila.py --mode recur --steps 150
python scripts/benchmark_gpt_vs_raslila.py --mode bind --steps 150

# Quantum research subsystem (original 6-qubit model, ~25k params)
python scripts/quick_real.py
python scripts/train_real.py

# GPU training of the quantum Hamiltonian model (needs CUDA)
python scripts/train_config.py        # from config.yaml
python scripts/train_2m_gpu.py        # 4xH100 variant
```

> Note: `torch` is a required dependency for the Raslila model. Quantum modules
> (`scipy`, `numpy`) are still used for the quantum subsystem and PennyLane is
> optional for hardware/hybrid backends.

## Any grid shape

5 × 5 × 5 is only the default. Two ways to set the shape:

- **`grid_shape: [layers, rows, cols]`**: three independent numbers, any size.
  `[7, 3, 10]` = 7 stacked sheets of 3×10 blocks (210 blocks); `[2, 9, 4]`,
  `[10, 2, 2]` and `[1, 1, 12]` all work. (The older `depth_layers` / `grid_h` /
  `grid_w` keys are the same thing written separately.)
- **`layer_shapes`**: one `[rows, cols]` sheet per layer, shallow to deep, for
  sheets of *different* sizes. Use this or `grid_shape`, not both. Layers can differ in size the way
  cortical areas do: a pyramid `[[8, 8], [6, 6], [4, 4], [2, 2]]`, a funnel, an
  hourglass `[[4, 4], [1, 1], [4, 4]]`, or a strip `[[1, 32]]`.

Between sheets of different sizes each block links to the block at the same
relative position in the next sheet, mapped in both directions, so every block
gets a vertical (and feedback) link. Lateral, diagonal, long-range,
`dense_layers`, adaptive depth and growth all follow the shape, and a grown
layer copies the deepest sheet's shape. A uniform box gives the same wiring,
parameter names and outputs as before this option existed. Parameter count
scales with blocks and edges, so check `Raslila params:` and the `Grid:` line
printed at the start of training.

## Adaptive depth and the live session

**Think until settled** (`max_think_cycles`, on in `config_adaptive.yaml`). A
fixed-depth model spends the same compute on "the cat sat on the" as on a hard
continuation. With `max_think_cycles: N` the grid thinks in cycles: cycle 1 is
one bottom-up pass; each further cycle sends the signal back down through the
feedback edges and up again, and an answer is read off the deepest sheet after
every cycle. At inference the loop stops as soon as the next-token prediction is
confident (`halt_confidence`) or has stopped changing (`halt_tol`), so easy
tokens exit after cycle 1 and hard ones keep looping through the 3D grid, up to
N. Training samples 1..N cycles per step and scores every cycle's answer, so an
early exit is a complete prediction. `max_think_cycles: 0` (the default) is the
old fixed-depth path, bit-identical to before.

**Always on** (`scripts/raslila_live.py`). The session does not exit when an
answer ends. It keeps the model and the whole conversation in memory and waits
on an input queue (no compute while idle). After `--idle-sleep` seconds it rests:
frees cached GPU memory and, with `--offload`, moves the weights to CPU. Input
wakes it with the conversation intact. You can type **while it is answering**:
the new text is added to the context and generation continues from there,
without restarting. Files dropped into `--inbox DIR` are read the same way.
`/stop`, `/more`, `/think N`, `/status`, `/reset`, `/save`, `/quit`; Ctrl+C
stops the current answer, not the session.

## Configuration (`config.yaml`)

Edit `config.yaml` to scale the model up or down. Key knobs:

| Key | What it controls | Scale direction |
|-----|------------------|-----------------|
| `dataset.name` / `dataset.max_tokens` | which HuggingFace corpus, how much of it | up = more tokens |
| `raslila.vocab_size` + `dataset.vocab_size` | BPE vocab (must match both); 16,384 now | up = better compression, +embedding params |
| `raslila.diagonal` + `long_range_prob` | brain-like wiring (on in 50M/20B, off in the A/B pair) | up = denser, +params |
| `raslila.node_depth` | transformer blocks per region — the dominant param lever | up = more capacity |
| `raslila.edge_hidden_dim` | edge network width; 37% of `config.yaml` | down = trim params |
| `raslila.activation_checkpoint` | recompute activations instead of storing them | on = fits bigger models |
| `training.*` | LR, batch, steps, precision | — |
| `distributed.fsdp` / CLI `--fsdp` | shard params+grads+optimizer across ranks | required above ~5B |
| `model.n_qubits` | quantum layer count (research) | — |
| `mlp.hidden_dim` | edge MLP capacity | up = more capacity |
| `embeddings.vocab_size` | vocabulary size | up = more tokens |
| `embeddings.embed_dim` | token embedding dim | up = more capacity |
| `distributed.backend` | `nccl` (server) / `gloo` (Colab w/o CUDA torch) | — |

**Colab note**: Colab's default PyTorch may lack NCCL — set
`distributed.backend: "gloo"` or install the CUDA build of PyTorch.

## All quantum fixes (traceable to reviews)

| # | Bug | Fix location |
|---|-----|--------------|
| 1 | alpha/beta degenerate (`alpha*X0X2 + beta*X0X2`) | `hamiltonian.py`: `alpha*X0X2 + beta*Y0Y2` |
| 2 | QNG stale params (memory injection discarded) | `qng.py sync()` + `trainer.py` calls it every step |
| 3 | MLP grad shape crash `(8,) @ (1,8)` | `mlp.py gradient()` proper chain rule, numeric-checked |
| 4 | `grad_J` reused beta's `loss_p` | `trainer.py` separate J perturbation |
| 5 | Unreachable target (qubits 1,3,5 passive, max 1/8) | `hamiltonian.py` X+Z fields on all qubits; marginal tracking |
| 6 | "Cooling" mislabel for unitary evolution | `evolution.py` docs + `annealed_evolve` sketch for real annealing |
| 7 | Squash+Sin `f'(±1)=0` stalls training | `activation.py squash_sin`: `sin((pi/2)s)`, grad never 0 for finite x |
| 8 | MSE weak on probabilities | `evolution.py nll_loss`, trainer uses NLL |
| 9 | Reflection `0.1+0.01*step` unbounded → tanh saturation | `trainer.py property_vector()` clips to 1.0 |
| 10 | Finite-diff `delta=0.1` noisy; param-shift noted | small epsilons (0.01), documented |
| 11 | Qubit endianness ambiguous | `index_to_bits`/`bits_to_index`, qubit0=MSB |
| 12 | Memory injected on alpha/beta (wrong) | `memory.py` + trainer inject on J via `exp(-hamming)` |
| 13 | Frobenius norm couples params implicitly | documented in `hamiltonian.py` (kept: needed for T=pi/2 stability) |

## Honest scope

- The 6-qubit quantum demo **learns verifiably** (NLL 4.16→4.05, p_full +12%,
  alpha/beta diverge, QFIM non-singular) but gains are **modest** — it proves
  trainability, not supremacy, on a single-step unitary.
- The **Raslila** model (`raslila.py`) is the current focus: classical, trainable,
  scales to millions of params on GPU. It keeps the brain-inspired edge-centric
  design in a purely classical PyTorch form suitable for real language modeling.
- See `PRE_TRAINING_DESIGN.md` for the honest gap analysis and scaling plan, and
  `VERIFICATION_REPORT.md` for verification results.
