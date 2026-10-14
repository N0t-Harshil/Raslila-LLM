# 08 — Runbook: Testing the Bidirectional Sweep

How to run the current experiment. **One variable:** `depth_sweeps` = 2 (bidirectional)
vs 1 (feedforward). Everything else is identical.

---

## The experiment

| | Config | Meaning |
|---|---|---|
| **A (candidate)** | `config.yaml` | `depth_sweeps: 2` — down, then up |
| **B (baseline)** | `config_baseline_feedforward.yaml` | `depth_sweeps: 1` — feedforward only |

Both are **163,605,123 parameters** (125 blocks, 618 edges), so a loss difference
can only come from the traversal direction. Both run with `inhibit: false`,
`state_dim: 0`, `replay_prob: 0.0`, and both pull the same corpus
(`data/corpus`, 448 Gutenberg books → 85,151,573 BPE-16k tokens) with the same
wiring (`connectivity_seed: 42`).

Verified: the two files differ by exactly one key (`raslila.depth_sweeps`).

> **Correction, 2026-10.** An earlier version of this runbook claimed the wiring
> had gone "brain-like" (`diagonal: true`, `long_range_prob: 0.3`,
> `max_long_range: 8`) and the vocab had moved to a 200k-token openwebtext
> tokenizer, and stated that both changes had been applied to *both* arms.
>
> **That was false.** The changes were applied to `config.yaml` only. The
> baseline still carried the old values, so the two arms differed by six keys
> instead of one — the exact failure mode this runbook exists to prevent. It
> surfaced only when a post-merge check reported 259,742,398 parameters against
> the documented 163.6M.
>
> Everything has been reverted to the verified configuration: BPE-16k
> (16384, matching `data/corpus_tokens.npy` whose max id is 16383),
> `long_range_prob: 0.1`, `max_long_range: 3`, `diagonal: false`, 618 edges.
> The brain-like wiring is a legitimate experiment but a *different* one — it
> gets its own run after this question is settled, not a silent change to this
> one.
>
> Two lessons worth keeping: a doc claiming a change was "applied to both arms"
> is not evidence that it was, and a parameter-count assertion is a cheap guard
> against a config that has drifted away from what the docs describe.

**The decision rule, fixed before you look at the numbers:** keep
`depth_sweeps: 2` only if it reaches a **lower loss at equal steps**. It costs
2× compute. If it ties or loses, revert to 1 — that 2× buys nothing.

---

## Prerequisites

The corpus is already cached locally and is **not** in git. It must be present
before training, because `train_raslila.py` expects the directory:

```
data/corpus/               448 Gutenberg .txt files, ~299 MB
data/corpus_tokens.npy     85,151,573 BPE-16k tokens  (162 MB)
data/tokenizer/raslila_bpe.json   (committed; 13.9 MB)
```

If you cloned fresh, build it in this order:

```bash
python scripts/build_corpus.py         # -> data/corpus/*.txt
python scripts/train_tokenizer.py      # -> data/tokenizer/
python scripts/pretokenize_corpus.py   # -> data/corpus_tokens.npy
```

**Verify before a long run** — this catches the vocab/data mismatch that
silently trains on garbage:

```bash
python -c "import numpy as np,yaml; a=np.load('data/corpus_tokens.npy'); v=yaml.safe_load(open('config.yaml'))['raslila']['vocab_size']; print(len(a),a.max(),v,a.max()<v)"
# expect:  85151573 16383 16384 True
# if the last value is False the vocab does not match the data -> STOP
```

`scripts/get_dataset.py` also exists and pulls a HuggingFace corpus, but the
active configuration uses the Gutenberg pipeline above. Mixing the two will
mismatch the cached tokenizer.

---

## Step 1 — Smoke test (2 minutes, run this first)

Catches config, data, and OOM problems before committing GPU hours.

```bash
python scripts/train_raslila.py --steps 3 --batch-size 1 --seq-len 32 --out-dir ck_smoke
python scripts/generate_raslila.py --checkpoint ck_smoke/raslila_step2.pt --prompt "The" --tokens 12
```

Expected: `Raslila params: 163,605,123 (163.6M)`, a decreasing loss, a saved
checkpoint, and 12 generated tokens. Delete it after:

```bash
rm -r ck_smoke
```

---

## Step 2 — GPU training

Your local machine is **CPU-only** (`torch 2.9.0+cpu`). The real runs need an
A100. On Colab:

```python
!nvidia-smi
!pip install -q torch --index-url https://download.pytorch.org/whl/cu124
```

**Recommended: use the launcher.** It runs both arms for you, writes them to
separate out-dirs, and prints the final-loss comparison. Crucially it
preflights the pair first and refuses to launch unless the two configs differ
by exactly `raslila.depth_sweeps` — a drifted A/B that *looks* controlled is
worse than no A/B, and it is what erased an earlier 3-seed win.

```bash
python scripts/run_ab.py --steps 20000            # both arms, all 4 GPUs each
python scripts/run_ab.py --steps 20000 --parallel # both at once, 2 GPUs each
python scripts/run_ab.py --compare-only           # just the comparison table
```

The manual commands below are what the launcher runs, kept for when you want
one arm on its own:

```bash
# A: bidirectional (the candidate)
python scripts/train_raslila.py \
  --config config.yaml \
  --out-dir ck_bidir \
  --steps 20000 \
  --batch-size 8 --seq-len 128 \
  --lr 3e-4 --warmup 500 \
  --save-every 2000 \
  --mixed-precision bf16 \
  --seed 42
```

```bash
# B: feedforward baseline, identical except the one key
python scripts/train_raslila.py \
  --config config_baseline_feedforward.yaml \
  --out-dir ck_feed \
  --steps 20000 \
  --batch-size 8 --seq-len 128 \
  --lr 3e-4 --warmup 500 \
  --save-every 2000 \
  --mixed-precision bf16 \
  --seed 42
```

Notes:
- Use `bf16` on an A100. On a T4 use `fp16` (T4 has no bf16).
- **Same `--seed` for both.** Then repeat with `--seed 43` and `--seed 44`. One
  seed is not enough — seed spread here is ±0.009–0.016 nats, the same size as
  the effects being measured.
- If OOM: lower `--batch-size` first, or raise `--accum` to hold the effective
  batch. Do not change `--seq-len` — it changes the experiment.

---

## Step 3 — Compare

```bash
python scripts/compare_runs.py ck_bidir ck_feed --min-steps 20000
```

The script prints the tail loss for both runs and applies the decision rule
mechanically — the rule is coded in, not judged by hand, so the result cannot be
rationalised after the fact:

```
VERDICT: KEEP depth_sweeps=2   candidate better by >0.01 nats
VERDICT: REVERT to depth_sweeps=1   baseline better
VERDICT: TIE   within ±0.01 nats — 2x compute bought nothing, revert
```

`--min-steps` makes it refuse to compare runs that stopped early, which is the
most likely way to get a misleading result (a crashed run looks "better" simply
because it never got worse).

Raw curves: `<out-dir>/train_log.csv`, columns
`step,lr,loss,alpha,beta,grad_norm,time_s`. Loss there is a windowed average.

**Checkpoint names are zero-indexed.** `--steps 20000` produces
`raslila_step19999.pt`, not `step19000`. With `--save-every 2000` you get
`step1999, step3999, … step17999, step19999`.

---

## Step 4 — Sample from the winner

```bash
python scripts/generate_raslila.py \
  --checkpoint ck_bidir/raslila_step19999.pt \
  --prompt "The old house stood at the end of the lane, and" \
  --tokens 120 --temperature 0.8
```

Save samples from **both** runs at the same temperature. Grep for repetition —
the recurring failure mode is loops, not gibberish, and it is worth documenting
whichever run loops more.

---

## If a run dies

Resume from the last checkpoint (tested working):

```bash
python scripts/train_raslila.py \
  --config config.yaml --out-dir ck_bidir \
  --resume ck_bidir/raslila_step17999.pt \
  --steps 20000 --batch-size 8 --seq-len 128 \
  --mixed-precision bf16 --seed 42
```

`--steps` is the **total** target, not the number of extra steps — it resumes and
continues to 20000. Optimizer state, LR and plateau counters are restored.

---

## Step 5 — Fast regression suite (no GPU needed)

Run after any code change, before launching a long job:

```bash
python scripts/test_raslila_quick.py
```

(That script uses its own small inline config, so its parameter count is not the
production 163.6M — it is a smoke test for wiring, not a size check.)

`python scripts/ab_brain_multiseed.py` is for mechanism screening only, and its
verdict is only meaningful at **≥5 seeds**. Do not use it to decide the
bidirectional question — it runs 200 steps, which is ~0.2 epochs and cannot
resolve an effect this size. Use the two real training runs instead.

---

## After A/B completes

Next candidates, **one at a time**, each compared against the winner of the
previous round and kept only if it wins:

| Order | Candidate | Config key | Baseline to beat |
|---|---|---|---|
| 1 | Bidirectional sweeps | `depth_sweeps` | *(running now)* |
| 2 | Signed lateral inhibition | `inhibit: true` | winner of #1 |
| 3 | Persistent per-block state | `state_dim: 32` | winner of #2 |
| 4 | Experience replay | `replay_prob: 0.25` | winner of #3 |

Protocol rules, from §12 of the research log:
1. Change **one** key. Several at once makes the result unattributable.
2. **≥5 seeds**, paired, and report the seed spread next to the effect.
3. Default verdict is "no significant change" until proven otherwise. A 3-seed
   "win" in this project evaporated under 6 seeds.
4. Record negative results. They are the most useful entries in the log.

---

## Files involved

| Path | Role |
|---|---|
| `config.yaml` | Arm A — `depth_sweeps: 2` |
| `config_baseline_feedforward.yaml` | Arm B — `depth_sweeps: 1`, identical otherwise |
| `scripts/train_raslila.py` | training; `--resume` for crash recovery |
| `scripts/compare_runs.py` | comparison + mechanical verdict |
| `scripts/generate_raslila.py` | sampling |
| `docs/02_Research_Log.md` §11–12 | how the bidirectional claim was found false on execution |

## Related

- [02 Research Log](02_Research_Log.md) §11–12 — how the bidirectional claim was
  found to be false on execution, and what the benchmarks got wrong
- [01 Architecture](01_Architecture.md) — `depth_sweeps` semantics and cost
- [07 Brain Gap Analysis](07_Brain_Gap_Analysis.md) — what the remaining
  candidates are meant to fix
- [04 Training and Data](04_Training_and_Data.md) — corpus and schedules