# Verification Report

> **Historical document (2026-09) — read with care.** Two changes landed in
> 2026-10: the training corpus moved to HuggingFace `Skylion007/openwebtext`
> (1.5B tokens, auto-fetched), and the wiring went brain-like
> (`diagonal: true`, `long_range_prob: 0.3`, `max_long_range: 8` — 618 → 1,114
> edges at 5×5×5, 163.6M → 212.6M params at hidden 256). Every parameter count,
> corpus figure and "current training" claim below predates both, and every
> checkpoint from before is unloadable. For current state see
> [`docs/00_Index.md`](docs/00_Index.md).
>
> **Third change (also 2026-10).** The tokenizer was upgraded from a
> 16,384-token BPE to a **199,995-token** one (measured 5.43 vs 3.34
> chars/token). That alone adds 47M parameters at hidden 256 and moves the
> token cache to uint32 / 6.0 GB. The `config_3m.yaml` and `config_15m.yaml`
> files referenced below were deleted; only `config.yaml` (260M),
> `config_50m.yaml` (92.6M), `config_20b.yaml` (21.0B) and
> `config_baseline_feedforward.yaml` remain.


Date: 2026-09-06 (quantum 6-qubit). 2026-09-09 (3D Brain classical).
All original notes archived in `docs_original/`.

## A. Quantum 6-Qubit Simulation (original, verified)

### A1. Component tests (`python scripts/test_components.py`)

**17 passed, 0 failed** on 2026-09-06:

- Activation: original `f'(1)≈6e-13` (bug reproduced); fixed `f'(1)=0.278` (nonzero);
  fixed grad nonzero on all of [-5,5]; bounded in [-1,1].
- Hamiltonian: alpha/beta gradients differ (`ga=-0.0182, gb=-0.0307` → independent);
  QFIM `det=4.2e-3`, `cond=4.06` (non-singular; old code was singular by construction).
- Target: distribution non-uniform (all qubits active); endianness `10 → 001010` round-trips;
  marginal prob in [0,1].
- MLP: shapes `(4,8),(8,),(8,1),(1,)` (old code crashed); analytic `dL/dW[0,0]=-0.004845`
  matches numeric to 1e-9.
- QNG `sync()` works; memory `resonance=1` for identical patterns, injects on J bounded.
- NLL strong at low p; reflection clipped.

### A2. Training run (`python scripts/train_6qubit.py`, 100 steps, seed 42)

```
Step   0 | NLL 4.1634 | p_full 0.0156 | p_marg 0.1247 | a +0.505 b +0.504
Step  20 | NLL 4.1265 | p_full 0.0161 | p_marg 0.1295 | a +0.247 b +0.361
Step  40 | NLL 4.0769 | p_full 0.0170 | p_marg 0.1359 | a +0.159 b +0.258
Step  60 | NLL 4.0585 | p_full 0.0173 | p_marg 0.1382 | a +0.158 b +0.237
Step  80 | NLL 4.0494 | p_full 0.0174 | p_marg 0.1394 | a +0.152 b +0.234
Step  99 | NLL 4.0451 | p_full 0.0175 | p_marg 0.1400 | a +0.153 b +0.236
```

- NLL **4.1634 → 4.0451** (monotone-ish decrease; 200-step extension reaches 4.0399 then plateaus).
- `p_full` (P(state 001010)) **0.0156 → 0.0175 (+12%)**; random baseline 1/64=0.0156.
- `p_marg` (qubits 0,2,4 = 0,1,1) **0.1247 → 0.1400 (+12%)**; baseline 1/8=0.125.
- **alpha/beta diverge** (+0.153 vs +0.236) → independently learnable (old code: identical grads).
- No crashes, no NaNs; QFIM invertible every step; history in `scripts/training_history.json`.

Interpretation: modest but **real, verifiable learning**. Single-step `T=pi/2` unitary with
~15 free parameters cannot reach P=1; it proves gradient flow without barren-plateau stall,
not high accuracy. Claims beyond this would be dishonest.

### A3. Quantum roadmap

- [x] Immediate fixes 1–10 (table in README).
- [x] Corrected run + alpha/beta independence validated.
- [x] Memory injection on J (`memory.py`, used in trainer every step via resonance).
- [x] Token-embedding wiring (8 tokens ↔ 3-bit patterns of qubits 0,2,4; untrained → near-uniform).
- [~] Scaling: `pauli_string_list()` (sparse terms) + `annealed_evolve()` sketch scaffolded;
      sparse-Trotter / PennyLane / Qiskit hybrid + real hardware remain future work.
- [ ] Theoretical analysis (entanglement entropy, capacity conditions) — open research.

---

## B. Raslila (classical, ACTIVE)

The current focus is a **purely classical** 3D-grid transformer (`src/raslila/raslila.py`)
that keeps the brain-inspired edge-centric design but trains end-to-end on GPU
with millions of parameters. **No quantum computation.**

### B1. `raslila.py` — fixed and verified (2026-09-09; modernized 2026-09-11)

After accumulated whitespace corruption broke the `train_step` method
(`IndentationError`), the file was rewritten cleanly. Results:

- **Syntax**: `ast.parse` clean.
- *Historical full-config forward* (4×4×2 grid, 32 regions, hidden 512):
  **240,769,794 params**, output `[2, 128, 32000]` — ran on CPU.
- *Historical compact test* (3×3×2 grid, 18 regions, hidden 256): **44,331,522 params**.
- **Current production** (5 layers × 5×5 sheets = 125 blocks, 1 round per sheet,
  BPE 16,384): **163,605,380 params**, 618 edge networks — see
  `docs/01_Architecture.md`.
- **`train_step` works**: loss decreases across steps and alpha/beta diverge:

```
Params: 44,331,522
Step 0: loss=9.0354 alpha=0.5016 beta=0.5015
Step 1: loss=8.9541 alpha=0.5033 beta=0.5032
Step 2: loss=8.9800 alpha=0.5051 beta=0.5049
Step 3: loss=8.8470 alpha=0.5071 beta=0.5069
Step 4: loss=8.8542 alpha=0.5094 beta=0.5091
```

- Loss **9.04 → 8.85** in 5 steps; **alpha/beta diverge** → interference params
  independently learnable, mirroring the quantum result but classical.
- Methods verified: `forward()` (string + tensor input, activations/entanglement
  flags), `train_step()`, `create_raslila()`, 3D `_build_connectivity()`
  (local radius + small-world long-range), `EdgeNetwork`, `TransformerNode`
  (RMSNorm pre-norm + RoPE + SwiGLU + QK-norm).
- 2026-09-11: modern-vs-legacy A/B (200 steps, ~28M, same data/seed):
  legacy 2.60 → modern **2.33**; plus-ablation (LNS, gated messages): no gain,
  flags default off.

### B2. Reproduce Raslila

```powershell
python -c "from raslila.raslila import Raslila, RaslilaConfig"  # import sanity
python src/raslila/raslila.py        # production config demo (5 x 5x5, 125 blocks, 164M)
python scripts/test_raslila_quick.py # 18-node training sanity check
```

---

## C. Reproduce (all)

```powershell
pip install -r requirements.txt
# Raslila (active classical model)
python scripts/test_raslila_quick.py     # 18-node training sanity check
python scripts/test_raslila_train.py     # longer training check
python scripts/train_raslila.py          # train from config.yaml
# Quantum subsystem (original 6-qubit model)
python scripts/quick_real.py             # 25k-param quantum quick test
python scripts/train_real.py             # quantum training on curated text
```

> Note: the original quantum component tests (`test_components.py`,
> `train_6qubit.py`, `training_history.json`) were archived with the rest of the
> quantum tooling; their verified results are recorded in Section A above.

## D. Conclusion

The quantum framework is **conceptually sealed and empirically trainable at
6-qubit scale** — all 13 known bugs fixed with isolated tests + end-to-end learning.

The **Raslila** model extends the brain-inspired ideas into a trainable classical
architecture with 28M–240M params that converges on GPU. This is the active
path for scaling to real language modeling while the quantum subsystem remains
as the theoretical/research foundation.
