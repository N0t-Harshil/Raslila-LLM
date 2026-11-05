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
python run.py gpucheck                    # multi-GPU training works on all 4 GPUs
python run.py gpucheck --compile          # ... and with torch.compile
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
Use whatever is fastest in the study command below (`--compile`, `--no-ckpt`).

## 4. Train + benchmark + compare (days)
```bash
# first the small sizes (~1 day): the first real Raslila-vs-baseline result
python run.py study --tiers 50m,125m --compile --no-ckpt
# then, if Raslila holds up, everything (finished models are skipped)
python run.py study --compile --no-ckpt
```
Leave out `--compile` / `--no-ckpt` if the speed test or gpucheck says they don't help or don't fit.
For the big sizes, use `--no-ckpt` only if the **1b** speed test showed `ckpt off` fitting at a micro-batch of at least 2
(the 1b config's value); otherwise run `python run.py study --compile` for those.
Add `--run-code` to include HumanEval pass@1 (runs model-written Python).

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
| `CUDA out of memory` | remove `--no-ckpt`, or add `--micro-batch 4` to `scale` |
| `torch.compile: FAILED` in make_env | `conda install -y -c conda-forge gxx_linux-64`, then run without `--compile` meanwhile |
| `unauthenticated requests` warning | `hf auth login` again in this env |
| a model FAILED in the study | `less runs/<model>/train.log`, paste the end of it |
| anything stopped | run the same command again: it resumes |
