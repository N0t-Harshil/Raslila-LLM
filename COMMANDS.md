# Every command, in order (4x A100)

Run everything inside `tmux` so a dropped connection doesn't stop it:
`tmux new -s raslila` to start, `Ctrl+B` then `D` to detach, `tmux attach -t raslila` to come back.
Every training run resumes from its last checkpoint, so after any stop just run the same command again.

## 0. One time: fresh, fast environment
```bash
cd ~/work/shared/Raslila-LLM
git pull
conda deactivate                          # leave (base); the new env must not mix with conda
bash scripts/make_env.sh                  # ~5 min: newest torch for your driver + only what's needed
source ~/raslila-fast/bin/activate        # EVERY new terminal / tmux window
hf auth login                             # once; paste your HF read token (never put it in a file in the repo)
hf auth whoami                            # must print your username
```
The prompt should now start with `(raslila-fast)`, not `(base)`.

## 1. Checks (minutes)
```bash
python run.py setup                       # packages + GPUs
python run.py check                       # study configs are fair and build
python run.py smoke                       # both models train
python run.py gpucheck                    # multi-GPU training works (compiled), 12 steps
```

## 2. Data (once, hours)
```bash
python run.py data                        # ~42 GB in data/mix/, prints an ETA every minute
```
It is finished when the last line says `Cache complete`. Re-running it is a no-op.

## 3. Speed (minutes, can run while the data builds)
```bash
python run.py speed --tier 125m --compile
python run.py speed --tier 1b --compile
```
Each prints tokens/s for the baseline and Raslila, with activation checkpointing ON and OFF.
Already done on 2026-11-05; the results are in docs/09_Scaling_Study.md ("Measured speed").

## 4. Train + benchmark + compare (days)
The fast settings are now the defaults (torch.compile on; Raslila without activation
checkpointing; micro-batches set from the A100 speed test). No extra flags are needed.

The machine is shared: if `nvidia-smi` shows a GPU busy with someone else's job (GPU 2 on
2026-11-05), leave it out, because multi-GPU training runs at the speed of the slowest GPU:
```bash
nvidia-smi                                                        # which GPUs are free?
CUDA_VISIBLE_DEVICES=0,1,3 python run.py gpucheck --tier 350m     # memory check, bigger sizes
CUDA_VISIBLE_DEVICES=0,1,3 python run.py gpucheck --tier 1b
# first the small sizes (~8 hours on 3 GPUs): the first real Raslila-vs-baseline result
CUDA_VISIBLE_DEVICES=0,1,3 python run.py study --tiers 50m,125m
# then, if Raslila holds up, everything (finished models are skipped; ~2.5 weeks on 3 GPUs)
CUDA_VISIBLE_DEVICES=0,1,3 python run.py study
```
Keep the same GPUs for the whole study. If all 4 are free, drop the `CUDA_VISIBLE_DEVICES=...` part.
Add `--run-code` to include HumanEval pass@1 (runs model-written Python). `--no-compile` turns compile off.

### 4c. The proof runs (2026-11-11): second size, equal compute, lean, brain regions

One command trains all four in order (~26 h + ~1 h benchmarks). Finished runs are skipped,
so if it stops, run the same command again.

```bash
cd ~/work/shared/Raslila-LLM && git pull && source ~/raslila-fast/bin/activate
CUDA_VISIBLE_DEVICES=0,1,3 python run.py gpucheck --tier 50m
CUDA_VISIBLE_DEVICES=0,1,3 python run.py study --tiers 50m,150m     --variant baseline_150m_iso --variant raslila_150m_lean --variant raslila_150m_topo
python run.py brain --run raslila_150m_topo --run raslila_150m_lean --run raslila_50m
cat runs/scaling_report.md
```

Read "Head to head at the same size" and "Equal training compute" in `runs/scaling_report.md`;
the decision rules are in docs/09 (Result 2).

### 4a. Raslila v4 at 150m (current, 2026-11-09): the 3D cortex that thinks in loops
Design and the papers behind it: docs/10_Research_and_v4_Design.md. Model: src/raslila/cortex.py.
160M baseline vs 162M Raslila v4 (same parameters; v4 spends ~2.2x compute per token by looping).
```bash
git pull
mkdir -p runs/v3 && mv runs/raslila_50m runs/raslila_125m runs/v3/ 2>/dev/null   # keep old grid runs out of the fit
python run.py smoke                                                # now also tests v4
CUDA_VISIBLE_DEVICES=0,1,3 python run.py gpucheck --tier 150m      # memory check (minutes)
CUDA_VISIBLE_DEVICES=0,1,3 python run.py study                     # default tier: 150m (~3 h + ~6 h)
python run.py brain --run raslila_150m --run baseline_150m         # brain-likeness, measured
```
Optional, after that: the 3D brain-region loss as its own experiment (costs val loss in TopoLM):
`CUDA_VISIBLE_DEVICES=0,1,3 python run.py study --variant raslila_150m_topo`
Results: `runs/scaling_report.md` and `runs/thinking.png` (accuracy vs thinking loops; the baseline is the flat line).
More loops at test time: `python run.py bench --run raslila_150m --loops 1,2,4,8,16,32 --force`

### 4b. Raslila v3 (learned routing, 2026-11-08): rerun the small sizes
v2 lost at 50m/125m (see docs/09). Keep its runs aside and train v3; the finished
baselines are reused (skipped), so this trains only the two Raslila models:
```bash
git pull
mkdir -p runs/v2 && mv runs/raslila_* runs/v2/
CUDA_VISIBLE_DEVICES=0,1,3 python run.py gpucheck --tier 125m    # v3 + compile on 3 GPUs, minutes
CUDA_VISIBLE_DEVICES=0,1,3 python run.py study --tiers 50m,125m
```

## 5. Watch it (any time, in another tmux window)
```bash
source ~/raslila-fast/bin/activate
python run.py status                      # progress, val loss, ETA of every model
tail -f logs/study.log                    # timeline: start / done / scores
tail -f runs/raslila_125m/train.log       # live training output of one model
nvidia-smi                                # GPU use (should be ~100% on all 4)
```

## 6. Results
```bash
python run.py report                      # -> runs/scaling_report.md (open it in Jupyter)
```
Also in `runs/`: `compare.csv` (every number), `curves.png`, `benchmarks.png`, `scaling.png`.

## Extra
```bash
python run.py bench --run raslila_125m            # benchmark one model again
python run.py bench --all --quick                 # quick benchmarks of every model
python run.py bench --run raslila_125m --think 4  # let Raslila think 4 loops
python run.py generate --run raslila_125m --prompt "def fibonacci(n):"
python run.py live --run raslila_125m             # chat with a model
python run.py scale --tier 125m --family raslila --gpus 4   # one model by hand
```

## If something goes wrong
| problem | fix |
|---|---|
| `CUDA out of memory` | halve `micro_batch` in that model's config (configs/scaling/<model>.yaml) and run the same command again; results are unaffected |
| `torch.compile: FAILED` in make_env | `conda install -y -c conda-forge gxx_linux-64`, then run without `--compile` meanwhile |
| `unauthenticated requests` warning | `hf auth login` again in this env |
| a model FAILED in the study | `less runs/<model>/train.log`, paste the end of it |
| anything stopped | run the same command again: it resumes |
