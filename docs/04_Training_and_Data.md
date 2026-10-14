# 04 — Training and Data

## Corpus

**Current: HuggingFace `Skylion007/openwebtext`, downloaded automatically the
first time training starts.**

| Corpus | Size | Tokens | Use |
|--------|------|--------|-----|
| `Skylion007/openwebtext` (streamed, not downloaded whole) | ~9.5B available | **1,500,000,000 cached** | **current training** |
| `data/gutenberg/` | 10.5 MB | ~10M bytes | early validation; `--data-dir` default for the benchmark script |

The previous training corpus — 448 Project Gutenberg books, 313 MB, 85,151,573
BPE tokens — **has been removed**. It was the binding constraint on the project,
not the architecture:

- for the 20B config it was **4.7× short** of Chinchilla-optimal (~400M needed);
- for the 50M config a 30k-step run at 4,096 tok/step consumes 122.9M tokens, so
  the old corpus was **1.4 epochs**, and the observed run went flat by step ~14k;
- it was 1860s literary prose, an odd match for anything you would deploy.

openwebtext is ~9.5B tokens of modern web text. At `max_tokens: 1.5e9` the cache
is 6.0 GB on disk (uint32 — the 199,995-token vocab no longer fits in uint16),
and a 30k-step 50M run at batch 8 × seq 128 consumes 30.7M tokens, **2% of it**
— no epoch boundary is ever crossed, every batch is effectively fresh.

### How the pipeline works

Implemented in [`src/raslila/hf_dataset.py`](../src/raslila/hf_dataset.py). All
steps are cached, so a re-run is a no-op.

```bash
# training does this for you; run it standalone only to pre-fetch or force
python scripts/get_dataset.py --config config_50m.yaml
python scripts/get_dataset.py --config config_50m.yaml --force     # rebuild
python scripts/get_dataset.py --config config_50m.yaml --list      # show settings
```

1. **stream** the dataset (`datasets.load_dataset(..., streaming=True)`). The
   40 GB of parquet is never written out; only the shards needed to reach
   `max_tokens` are pulled, and HF caches them under `~/.cache/huggingface`.
2. **fit the BPE tokenizer** — only if it is missing, or was fitted on a
   *different* corpus or sample size. The fitted source is recorded in
   `data/tokenizer/meta.json`, so switching corpora re-fits automatically and
   re-running does not.
3. **tokenize** into `data/openwebtext_tokens.npy`.

Under `torchrun` **rank 0 does all of this** and the other ranks wait on a
barrier before loading, so the download happens exactly once.

### Configuration

```yaml
dataset:
  name: "Skylion007/openwebtext"
  split: "train"
  text_field: "text"
  max_tokens: 1500000000
  cache: "data/openwebtext_tokens.npy"
  tokenizer: "data/tokenizer/raslila_bpe.json"
  vocab_size: 16384           # must equal raslila.vocab_size
  train_tokenizer: true        # re-fit BPE on this corpus, then reuse it
  tokenizer_sample_chars: 300000000
```

**Legacy fallback.** Delete the `dataset:` section and set
`training.data_dir` to a directory of `.txt` files; the trainer then uses the
old local-corpus path unchanged.

**Requirements.** `datasets` and `tokenizers` must be installed; the trainer
exits with install instructions rather than failing mid-run. See
`requirements.txt`.

### Removing the old corpus

`data/corpus/`, `data/corpus_tokens.npy`, `data/synthetic/`, `data/benchmark/`,
`data/compare/` and `data/foundation/` were deleted with this change (483 MB,
58,000 files). Nothing references them. `data/gutenberg/` was kept because
`benchmark_gpt_vs_raslila.py` uses it as the `--data-dir` default.

Three scripts still reference the removed files:

- `scripts/pretokenize_corpus.py` and `scripts/build_corpus.py` now exit with a
  clear message; use `scripts/get_dataset.py` for the active corpus.
- `scripts/ab_brain_mechanisms.py` now prefers `data/openwebtext_tokens.npy` and
  only falls back to the old name if you still have one.

The `training.data_dir` fallback no longer shares one hardcoded cache. It keys
`data/corpus_tokens_<hash>.npy` on the directories you asked for, because the
single fixed path meant pointing `data_dir` at a second corpus silently reused
the first corpus's tokens — you would train on the wrong data with nothing
but a suspicious loss curve to notice.

## Tokenization — the biggest quality lever (now implemented)

We began **byte-level** (257 vocab). Principled and tokenizer-free, but brutally
token-inefficient: English averages ~4 characters per BPE token and exactly 1
byte per our token — a 3–4× tax on effective epochs at identical compute.

**Now default: byte-level BPE, 199,995 tokens, measured 5.43 characters/token
on hold-out English prose (the old 16k vocab measured 3.34 on Gutenberg).**

The sample size is a measured quantity, not a guess: on openwebtext with
`min_frequency=2`, 60M chars only reaches 184,355 merges (92.2% of target) and
the shortfall is wasted embedding rows, so the configs fit on 300M chars —
the smallest sample measured to reach the full 199,995.

```bash
python scripts/train_tokenizer.py --vocab-size 16384   # legacy, local .txt corpora
python scripts/pretokenize_corpus.py                    # exits: superseded by get_dataset.py
```

Both commands above are now **legacy**: the tokenizer and the token cache are
built by the dataset pipeline (`scripts/get_dataset.py`, described above) when
`dataset.train_tokenizer` is true. They are kept only for a local `.txt` corpus
via the `training.data_dir` fallback.

Effect of BPE versus byte-level on the old corpus (the reason the switch happened):

| | tokens | tokens/char |
|---|---|---|
| byte-level | 307,187,000 | 1.00 |
| **BPE 16k** | **85,152,000** | **3.34** |

Same 313 MB of text now yields 85M tokens, each carrying ~3.3 characters of
linguistic structure instead of one byte. That is a **3.6× increase in effective
epochs at the same compute** — worth more than any architecture change we tested.
It is the main reason the model should stop producing byte-soup within a
Colab-sized budget.

`id 0` is reserved for `<|pad|>` so the training loop's `ignore_index=0` is
unchanged. Switch back with `tok_mode: bytes` and `vocab_size: 257` if desired —
though with `dataset:` set the trainer forces `bpe`, since byte-level ids cannot
represent a streamed web corpus.

## Schedule

Adaptive LR, all configurable in `config.yaml`:

1. **Warmup ramp** over `warmup_steps` (default 500).
2. **Plateau decay** — when the 50-step window fails to improve on the best loss
   by more than `lr_threshold` (0.1%) for `lr_patience` steps, multiply the LR by
   `lr_factor` (0.5).

**Known trap, hit already.** Plateau decay froze learning at 0.2 epochs in an
early run (LR decayed to 5.7e-10 with the model still at loss 2.5). For long
runs, prefer a long fixed cosine schedule over aggressive plateau decay.

> The unused `scheduler` key was removed from the active configs. `adaptive_lr`
> controls optional plateau decay after warmup; with it disabled, LR stays at the
> base rate after warmup. The trainer does not implement cosine decay.


Full per-step logging (`step, lr, loss, alpha, beta, grad_norm, time_s`) goes to
`checkpoints/<run>/train_log.csv` as well as stdout.

## Hardware runbooks

Model under discussion: **212.6M params** (125 blocks = 5 layers × 5×5,
**618 edges** with the brain-like wiring, BPE 16k) — 163.6M / 618 edges was
the pre-wiring shape the memory table below was measured on. Edge networks add
1.37× to the parameter count at this hidden width, so scale the numbers below by
~1.3 if you are budgeting a run.
AdamW keeps roughly 3× parameters in RAM, plus activations for the
message-passing loop.

### CPU (this machine or Colab)

```bash
python scripts/train_raslila.py --steps 4000 --batch-size 4 --seq-len 128 \
    --out-dir checkpoints/raslila_cpu
```

~22 s/step at batch 2 × seq 64 on 8 threads; expect ~30 s/step at batch 4 ×
seq 128. **4000 steps is a multi-day CPU run** — use CPU for smoke tests only.
Colab CPU (24 vCPU) will be comparable.

A quick timing check before committing:

```bash
python scripts/train_raslila.py --steps 10 --batch-size 2 --seq-len 64 --out-dir /tmp/t
```

### GPU memory guide (163.6M model, BPE 16k)

| Batch × Seq | Rough memory | Fits |
|---|---|---|
| 4 × 128 | ~6 GB | Colab T4 (12 GB) ✓ |
| 8 × 128 | ~10 GB | Colab T4, tight ✓ |
| 16 × 256 | ~26 GB | A100 40 GB ✓ |
| 32 × 512 | ~80 GB | multi-GPU |

If you hit OOM, lower `batch_size` first, or keep the batch and raise
`gradient_accumulation` (same effective batch, less memory).

### Colab GPU (fp16 required on T4)

```python
# Colab TPU images ship a mismatched torch; align both before installing XLA.
!pip install -q "torch==2.8.*" "torchvision==0.23.*" "torchaudio==2.8.*"
```

```bash
python scripts/train_raslila.py --mixed-precision fp16
```

`config.yaml` drives everything else (batch, seq, steps, data_dir, out_dir,
save_every). A 260M checkpoint is ~3 GB with AdamW state — prune between runs.

### Colab TPU (single core, if Colab gives you one)

Check first — Colab often hands you a TPU-labelled VM with no accelerator:

```python
import os
print(os.environ.get('COLAB_TPU_ADDR'))   # None => no TPU, pick GPU instead
```

torch and torch_xla major.minor **must** match or `_XLAC` fails to load.

```bash
python scripts/train_raslila_tpu.py --steps 8000 --out-dir /content/drive/MyDrive/raslila_ckpt
```

One process owns the TPU per session lifetime. Checkpoints save as CPU tensors so
they load on any device.

### A100 (yours)

```bash
python scripts/train_raslila.py          # config is already bf16-native
```

Raise `batch_size` / `seq_len` in config for a 40 GB+ card.

## Epochs — no longer the bottleneck

There is no epoch boundary to reason about any more. The corpus is
**1,500,000,000 tokens** and a 30k-step run at batch 32 × seq 128 (4,096
tok/step) consumes 122.9M of them — **8% of the corpus, 0.08 epochs**. With
random contiguous window sampling, every batch is effectively fresh text for the
whole run.

For scale, one full pass over 1.5B tokens would be:

| batch × seq | tok/step | steps/epoch |
|---|---|---|
| 8 × 256 | 2,048 | 732,422 |
| 32 × 128 | 4,096 | 366,211 |

(The old 85M-token corpus needed ~42k steps per epoch at 8 × 256, which is why
every run so far was 1–3 epochs of the same 448 books.)

Note also that our loss is nats per *token*, not per byte, so numbers are not
directly comparable to the byte-level history in [02](02_Research_Log.md)
(9.73 at init for a 16k vocab vs 5.6 for 257).

## Generation

```bash
python scripts/generate_raslila.py \
    --checkpoint checkpoints/raslila_cpu/raslila_step399.pt \
    --prompt "It was a dark and stormy night" --tokens 150 --temperature 0.7
```

Byte-sampled, masked to the trained byte range, decoded back to text.
