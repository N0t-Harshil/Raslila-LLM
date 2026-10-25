# 04 — Training and Data

## Corpus

**Current: HuggingFace `HuggingFaceFW/fineweb-edu` (config `sample-10BT`),
streamed, cleaned and tokenized automatically the first time training starts.**

| Corpus | Size | Tokens | Use |
|--------|------|--------|-----|
| `HuggingFaceFW/fineweb-edu`, `sample-10BT` (streamed) | ~10B available | **1,500,000,000 cached** | **current training** |
| `Skylion007/openwebtext` | ~9.5B available | — | previous corpus (2026-10, superseded) |
| `data/corpus/` (448 Gutenberg books) | 313 MB | 85,151,573 | `training.data_dir` fallback only |
| `data/gutenberg/` | 10.5 MB | ~10M bytes | `--data-dir` default for the benchmark script |

**Why FineWeb-Edu.** openwebtext is a scrape of pages linked from Reddit with no
quality filter. FineWeb-Edu is Common Crawl filtered by an educational-quality
classifier and deduplicated; the FineWeb paper (arXiv 2406.17557) measured it
well ahead of other open web corpora on knowledge and reasoning benchmarks at
small model sizes, which is the regime this project trains in.

**Why not Gutenberg.** The 85M-token book corpus was the binding constraint: 4.7x
short of Chinchilla-optimal for the 20B config, 1.4 epochs for a 30k-step 50M
run (flat by step ~14k), and 1860s literary prose. It is still on disk and is
used only if the `dataset:` section is removed.

At `max_tokens: 1.5e9` the cache is **~3.0 GB** (uint16 — the 16,384-token vocab
fits), and a 20k-step run at batch 8 x seq 128 x accumulation 8 consumes 164M
tokens, **11% of it** — no epoch boundary is crossed.

### How the pipeline works

Implemented in [`src/raslila/hf_dataset.py`](../src/raslila/hf_dataset.py). All
steps are cached, so a re-run is a no-op.

```bash
# training does this for you; run it standalone only to pre-fetch or force
python scripts/get_dataset.py --config config.yaml
python scripts/get_dataset.py --config config.yaml --force     # rebuild
python scripts/get_dataset.py --config config.yaml --list      # show settings
```

1. **stream** the dataset (`datasets.load_dataset(..., streaming=True)`) through
   a shuffle buffer (`shuffle_seed`, `shuffle_buffer`), so the validation head is
   a random sample rather than the first crawl files. Only the shards needed to
   reach `max_tokens` are pulled; HF caches them under `~/.cache/huggingface`.
2. **clean** every document (`clean()`): normalise line endings, strip trailing
   spaces, collapse runs of blank lines; drop documents shorter than `min_chars`
   (200) or less than `min_alpha_frac` (60%) letters/whitespace; drop exact
   duplicates of an already-kept document (8-byte BLAKE2 hash). FineWeb-Edu is
   already filtered, so on a 3M-token sample this dropped nothing — it is a
   safety net that matters if you point `dataset.name` at a rawer corpus.
3. **fit the BPE tokenizer** — only if it is missing, or was fitted on a
   *different* corpus, sample size or recipe. The fitted source is recorded in
   `data/tokenizer/meta.json` as `name::config::chars::recipe`, so switching
   corpora re-fits automatically and re-running does not.
4. **tokenize** with `encode_batch` (all CPU cores) into
   `data/fineweb_edu_tokens.npy`, appending **`<|endoftext|>` after every
   document**. Before this the documents were concatenated with no separator,
   so the model was trained to continue one web page straight into an unrelated
   one and had no way to learn where a document ends.
5. write **`<cache>.json`** recording the tokenizer source, dataset and
   `max_tokens`, plus kept/dropped document counts. If a later run wants a
   different tokenizer or corpus, the cache is rebuilt instead of silently
   reused with mismatched token ids.

Under `torchrun` **rank 0 does all of this** and the other ranks wait on a
barrier before loading, so the download happens exactly once.

### Configuration

```yaml
dataset:
  name: "HuggingFaceFW/fineweb-edu"
  config: "sample-10BT"
  split: "train"
  text_field: "text"
  max_tokens: 1500000000
  cache: "data/fineweb_edu_tokens.npy"
  tokenizer: "data/tokenizer/raslila_bpe.json"
  vocab_size: 16384             # must equal raslila.vocab_size
  train_tokenizer: true          # re-fit BPE on this corpus, then reuse it
  tokenizer_sample_chars: 300000000
  shuffle_seed: 1234
  min_chars: 200
  min_alpha_frac: 0.6
  dedup: true
  val_tokens: 2000000            # held-out head of the cache, used for val loss
```

**Validation.** The first `val_tokens` of the (shuffled) cache are held out and
never trained on. Every `eval_every` steps the trainer reports val loss averaged
over up to 128 evenly spaced windows; it is logged to `train_log.csv`.

**Legacy fallback.** Delete the `dataset:` section and set
`training.data_dir` to a directory of `.txt` files; the trainer then uses the
old local-corpus path unchanged. It keys its cache as
`data/corpus_tokens_<hash>.npy` on the directories you asked for, so pointing
`data_dir` at a second corpus cannot silently reuse the first corpus's tokens.

**Requirements.** `datasets` and `tokenizers` must be installed; the trainer
exits with install instructions rather than failing mid-run. See
`requirements.txt`.

**Old scripts.** `scripts/pretokenize_corpus.py`, `scripts/build_corpus.py` and
`scripts/train_tokenizer.py` belong to the local `.txt` path; use
`scripts/get_dataset.py` for the active corpus. `scripts/ab_brain_mechanisms.py`
reads `data/fineweb_edu_tokens.npy`, falling back to older caches if present.

## Tokenization — the biggest quality lever (now implemented)

We began **byte-level** (257 vocab). Principled and tokenizer-free, but brutally
token-inefficient: English averages ~4 characters per BPE token and exactly 1
byte per our token — a 3–4× tax on effective epochs at identical compute.

**Now default: byte-level BPE, 16,384 tokens, fitted on a 300M-char sample of
FineWeb-Edu, measured ~4.3 characters/token on FineWeb-Edu text** (from a small
20M-char test fit; the full fit should be slightly better). Specials:
`<|pad|>` = 0 (the loss `ignore_index`) and `<|endoftext|>` = 1 (document
separator). Text is NFC-normalised before byte-level pre-tokenisation, so
composed and decomposed accents share merges.

A 199,995-token vocab was trialled on openwebtext (5.43 chars/token vs 3.34)
and **not adopted**: at hidden 256 it puts ~47M parameters into the embedding
and output bias and makes every step pay for a 200k-way softmax, for a model
whose budget is better spent on the grid.

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

## Schedule and optimiser

Set by `training.lr_schedule`:

- **`cosine` (all active configs)** — linear warmup over `warmup_steps`, then
  cosine decay from `learning_rate` to `min_lr` over the run's steps. The
  horizon is `start_step + --steps`, so a resumed run decays over its own
  extended length.
- **`plateau`** (the old rule) — warmup, then when the 50-step window fails to
  improve on the best loss by `lr_threshold` for `lr_patience` steps, multiply
  the LR by `lr_factor`, floored at `min_lr`. Only active with `adaptive_lr: true`.

**Why cosine is now the default.** Plateau decay froze learning at 0.2 epochs in
an early run (LR decayed to 5.7e-10 with the model still at loss 2.5), and it
reacts to batch-to-batch loss noise. A fixed step budget wants a fixed schedule.

**AdamW** uses betas (0.9, 0.95) and the fused CUDA kernel when available.
Weight decay applies **only to 2-D weight matrices** (including the tied
embedding); RMSNorm gains, biases, the alpha/beta interference scalars and the
persistent-state decay logits are not decayed. Previously every parameter was
decayed at 0.1, which pulled norm gains and alpha/beta toward zero. Resuming a
checkpoint saved before this split restores the weights and restarts the AdamW
moments, with a warning.

Full per-step logging (`step, lr, loss, val_loss, alpha, beta, grad_norm, time_s`) goes to
`checkpoints/<run>/train_log.csv` as well as stdout.

## Hardware runbooks

Model under discussion: **163.6M params** (`config.yaml`: 125 blocks = 5 layers
x 5x5, **618 edges**, BPE 16k; edge networks are 61.0M of it, 37%).
`config_50m.yaml` is 68.9M with the brain-like wiring.
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
save_every). A 163.6M checkpoint is ~2 GB with AdamW state — prune between runs.

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
python scripts/check_configs.py          # must pass before a run
python scripts/train_raslila.py          # config is already bf16-native
python scripts/run_ab.py --steps 20000   # the depth_sweeps A/B
```

The trainer enables TF32 matmuls on Ampere. Edge networks run as one batched
matmul per block instead of ~1,236 separate tiny MLP calls per forward, which
matters most on a GPU (launch-bound) — the CPU gain measured 10-20%. On an
80 GB card there is room to raise `batch_size` / `seq_len`; keep both A/B arms
identical if you do.

## Epochs — no longer the bottleneck

There is no epoch boundary to reason about. The corpus is **1,500,000,000
tokens**; a 20k-step run at batch 8 x seq 128 x accumulation 8 consumes 164M of
them — **11% of the corpus**. With random contiguous window sampling, every
batch is effectively fresh text for the whole run.

For scale, one full pass over 1.5B tokens would be:

| batch x seq | tok/step | steps/epoch |
|---|---|---|
| 8 x 256 | 2,048 | 732,422 |
| 32 x 128 | 4,096 | 366,211 |

Note also that our loss is nats per *token*, not per byte, so numbers are not
directly comparable to the byte-level history in [02](02_Research_Log.md)
(9.73 at init for a 16k vocab vs 5.6 for 257).

## Generation

```bash
python scripts/generate_raslila.py \
    --checkpoint checkpoints/raslila_cpu/raslila_step399.pt \
    --prompt "It was a dark and stormy night" --tokens 150 --temperature 0.7
```

Samples BPE tokens (never the pad id 0) at the trained context length, stops
early if the model emits `<|endoftext|>`, and decodes back to text.
