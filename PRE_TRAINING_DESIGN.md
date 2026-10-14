# Raslila Pre-Training Design Document

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

## Brutal Gap Analysis & Realistic 2M Parameter Experiment Plan

**Status**: PRE-TRAINING — DO NOT TRAIN UNTIL THIS IS SIGNED OFF
**Date**: 2026-09-06
**Version**: 0.3 (2026-09-11 — Raslila modernization: RoPE/SwiGLU/RMSNorm)

---

## ⚠️ 2026-09-09 UPDATE: ARCHITECTURAL PIVOT

This document was originally written for a **quantum** 2M-parameter experiment.
Since then, the project pivoted to the **purely classical Raslila** model (`raslila.py`):

- Blocks = transformer blocks on a **deep grid**: 5 layers, each a 5×5 2D sheet;
  **edges = neural networks** that process messages (the brain-inspired
  "edge-centric" idea, kept).
- Blocks connect laterally within a sheet, vertically between sheets, and via
  feedback (L+1 → L), plus small-world shortcuts.
- Learned `alpha`/`beta` provide Hamiltonian-style interference, classically.
- **No quantum computation** — trains end-to-end on GPU (4×H100, Colab).
- **Current production model**: **5 layers × 5×5 = 125 blocks**, hidden 256,
  1 cascade round per sheet, BPE vocab 16,384 → **163.6M params**, 618 edges.
  Each layer has distinct weights, so depth costs parameters (a real deep net).
- *Historical:* early verification used 18 nodes / 256 hidden (44M @32k vocab,
  28M @257 byte vocab) and 32 nodes / 512 hidden (~240M). The A/B results below
  were measured on a ~28M harness model; see [`docs/`](docs/README.md) for the
  authoritative current spec.

**2026-09-11 architecture corrections (audit vs. this design):**
1. **Causal attention** (was missing) — nodes now attend only to positions
   ≤ t, so next-token training no longer leaks the answer via future positions.
2. **Node identity** — each node receives a learnable role embedding from its
   3D position (`node_positions` buffer was unused); regions can specialize.
3. **`local_radius` honored** — connectivity now respects the configured radius
   (default 6-neighbor for radius 1).
4. **Learned node readout** — final prediction attention-weights regions over
   nodes instead of a plain mean.
5. **Byte vocab (257)** — byte-level LM, output head no longer scores 32k
   classes for only 257 used bytes (−16.3M params).

**2026-09-11 architecture modernization (research-backed):**
Adopted the converged 2023–2026 dense-transformer recipe inside each node
(RMSNorm pre-norm + RoPE + SwiGLU + QK-norm + tied head + final norm), all
config-gated (`use_rope/use_swiglu/use_rmsnorm/qk_norm/tie_weights`) so the
legacy block remains available for ablations. Measured on identical
Gutenberg windows/seed/optimizer (200 steps, ~28M params each):
legacy 2.60 → modern **2.33 (−0.27 nats)**, gap still widening.
Two init bugs caught by the A/B (both fixed): pre-norm needs a final norm
before the head (else logits explode, loss ~110), and tied heads need
GPT-2-style 0.02 embedding init (else saturated softmax, loss ~45).

**2026-09-11 extra-layer research (MoDA/Depth-Attention, LNS, Mix-LN, gated
attention, MoE, MTP reviewed):** implemented the two cheapest testable ideas
as config-gated options — **LNS** (`lns`, norm outputs × 1/sqrt(depth)) and
**gated messages** (`gated_messages`, input-dependent sigmoid on edge
traffic). 4-arm A/B (150 steps, ~28M each): base 2.487 / +lns 2.484 /
+gate 2.500 / +both 2.495 — **no measurable gain (negative result)**.
Expected in hindsight: LNS cures deep-stack variance growth, but per-node
depth here is 1–4. Both flags default **off**. Revisit LNS if node_depth
grows large; revisit MoE / multi-token prediction / differential attention
only with GPU budget (all scored in SOTA_COMPARISON.md).

**2026-09-11 neuroscience-grounded mechanisms (all A/B-tested, all tie):**
Implemented three brain mechanisms as config-gated options — **MTP**
(`mtp_steps/mtp_weight`: predict t+2..t+K, hierarchical forecasting),
**global workspace** (`workspace_slots`: GWT compete-to-write + broadcast),
**top-k readout** (`readout_topk`: k-WTA sparse coding over regions).
4-arm A/B (150 steps, ~28M each, main loss): base 2.472 / +mtp 2.487 /
+workspace 2.475 / +topk 2.499 — **no measurable gain at this horizon**.
All default **off**. MTP most likely to pay off on full-length runs
(auxiliary harder targets need time); workspace/topk revisit with deeper
stacks or efficiency constraints.

**2026-09-11 multidirectional recurrence (the core thesis, A/B-validated):**
Modern LLMs are feedforward (input→1→2→3→output); brains exchange back and
forth (1→2→1→3→…). Raslila rounds already shared weights (recurrent), but each
round used frozen snapshots (one hop per round). Implemented, all config-gated:
**immediate updates** (Gauss-Seidel: read neighbors' fresh states — full-grid
cascades per round), **priority ordering** (most-surprised-first sweeps),
**settle-to-convergence** (`settle_thresh`: halt when traffic dies, num_layers
as cap). 4-arm A/B (150 steps, identical ~28M): frozen2 2.385 / **immediate2
2.360 (−0.026, free)** / +priority 2.382 (tie) / frozen4 2.398 (2× compute, no
gain). Conclusion: multidirectionality beats pass count. `immediate_updates`
now defaults **on**; priority/settle off (settle smoke-tested only).

**What this changes for this document:**

| Concern | Quantum plan (original) | Classical Raslila (now) |
|---------|-------------------------|--------------------------|
| Barren plateaus | 85% risk | N/A (no variational quantum circuit) |
| Exponential sim wall | Blocks >10 qubits | N/A |
| Parameter-shift cost | 2 evals/param, impossible | N/A |
| QFIM inversion O(n³) | 95% risk | N/A |
| Quantum hardware | QPU required for scale | Not required |
| Scaling | Hilbert 2^n | Standard GPU (like transformers) |

The **gates G1–G8**, data pipeline, training infrastructure, evaluation, kill
criteria, and compute budget in this document **all still apply** to the classical
Raslila model — they are model-agnostic engineering requirements. The **quantum-specific
gates (G2 dense expm, G3 parameter-shift, G8 sparse Trotter)** become optional
research-scaling work; the core language-modeling path no longer depends on them.

The quantum subsystem (`model.py`, `gpu_hamiltonian.py`, etc.) remains as the
**theoretical/research foundation** in `docs_original/` + the quantum modules.
See `README.md` and `VERIFICATION_REPORT.md` for current verified state.

---

---

## EXECUTIVE SUMMARY: WHY THIS WILL PROBABLY FAIL

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **No quantum advantage for language** | 70% | Total | Classical baselines first |
| **Barren plateaus at 20+ qubits** | 85% | Total | QNG + diagnostic logging |
| **Exponential classical sim wall** | 100% at 20 qubits | Blocking | Sparse Trotter + PennyLane |
| **Topological memory doesn't scale** | 60% | High | Unit test resonance at scale |
| **Parameter-shift too expensive** | 90% | High | Stochastic parameter-shift |
| **QFIM inversion O(n³) kills scaling** | 95% | High | Diagonal approx / KFAC |
| **Measurement collapses superposition** | 80% | Medium | Annealing + marginal readout |
| **Universes learn same thing** | 75% | High | Orthogonality regularization |
| **Classical MLP dominates quantum** | 60% | Medium | Freeze/ablation studies |
| **No gradient flow through measurement** | 100% | Total | Parameter-shift only |

**Bottom Line**: This is a **physics experiment**, not an LLM training run. Treat it as such.

---

## 1. MISSING ENGINEERING PIECES (Must Build Before Training)

### 1.1 Data Pipeline — COMPLETELY MISSING
```python
# NONE OF THIS EXISTS:
- Tokenizer trained on real corpus (BPE, 32k vocab)
- Streaming dataset (FineWeb-Edu, 10B tokens)
- Sequence packing (no padding waste)
- Train/val/test splits with contamination check
- DataLoader with prefetch, sharding
```

### 1.2 Training Infrastructure — COMPLETELY MISSING
```python
# NONE OF THIS EXISTS:
- Distributed training (DDP/FSDP) — quantum part is single-GPU
- Gradient accumulation (simulate larger batch)
- Learning rate schedule (cosine + warmup)
- Gradient clipping (global norm)
- Mixed precision (bf16 for classical, complex128 for quantum)
- Checkpointing (every N steps + best val)
- Resume from checkpoint (RNG state, step, optimizer)
- WandB/TensorBoard logging (loss, grads, entanglement, params)
- EMA weights for eval
```

### 1.3 Quantum-Classical Interface — FRAGILE
```python
# CURRENT ISSUES:
- scipy.linalg.expm for 2^n x 2^n — O(8^n) — dies at n=12
- No batched quantum forward (each sample = separate expm)
- Parameter-shift: 2 evals/param × 2M params = IMPOSSIBLE
- QFIM: O(n³) inversion — dies at n=1000
- No JIT compilation (JAX/PennyLane needed)
- No gradient checkpointing for quantum circuit
```

### 1.4 Evaluation — COMPLETELY MISSING
```python
# NONE OF THIS EXISTS:
- Perplexity on held-out val set
- Downstream benchmarks (MMLU, GSM8K, Hellaswag)
- Ablation: classical-only vs quantum
- Entanglement vs performance correlation
- Universe specialization metrics
- Memory retrieval accuracy
- Inference speed (tokens/sec)
```

### 1.5 Model Architecture Gaps
```python
# CRITICAL MISSING:
- Vocabulary alignment: base model has 8 fixed tokens, language.py has 37 char tokens
- No embedding table (2M params needs ~50k vocab × 64 dim = 3.2M just for embed)
- Position encoding (RoPE/ALiBi) — quantum part has no position awareness
- LayerNorm / RMSNorm — quantum Hamiltonian has no normalization
- Residual connections — no skip connections in Hamiltonian evolution
- Attention mechanism — Raslila replaces attention with entanglement (unproven)
- Multi-head — no equivalent
- FFN — MLP per edge is tiny (4→8→1)
```

---

## 2. REALISTIC 2M PARAMETER BUDGET

### Parameter Allocation (2,000,000 total)

| Component | Params | Calculation | Notes |
|-----------|--------|-------------|-------|
| **Token Embedding** | 1,024,000 | 16,384 vocab × 64 dim | 16k vocab minimum |
| **Position Embedding** | 65,536 | 1024 ctx × 64 dim | RoPE preferred |
| **Edge MLPs** | 327,680 | 64 edges × (64→128→1) | Shared across edges? |
| **Hamiltonian (h_z, h_x, J, α, β)** | ~500 | n_qubits × 4 | Negligible |
| **LayerNorm/RMSNorm** | 131,072 | 2 × 64 per layer × 8 layers | |
| **Output Projection** | 1,048,576 | 64 dim × 16,384 vocab | Tied with input embed |
| **Memory Snapshots** | 100,000 | 1000 snaps × 100 params | Compressed |
| **TOTAL** | **~2,700,000** | | **Over budget — need cuts** |

### Realistic Cuts for 2M:
- Vocab: 8,192 (524k embed)
- Context: 512 (32k pos)
- Edge MLPs: 32 edges, 64→64→1 (131k)
- Layers: 6 (not 8)
- **Total: ~1.95M** ✓

### Qubit Budget for 2M Classical Params
| Qubits | Hilbert Dim | Classical Sim Cost | Quantum HW Needed |
|--------|-------------|-------------------|-------------------|
| 10 | 1,024 | 16 KB state, feasible | 10 qubits (easy) |
| 12 | 4,096 | 64 KB, feasible | 12 qubits |
| 14 | 16,384 | 256 KB, slow | 14 qubits |
| 16 | 65,536 | 1 MB, very slow | 16 qubits |
| 18 | 262,144 | 4 MB, impractical | **Required** |
| 20 | 1,048,576 | 16 MB, impossible classically | **Required** |

**Recommendation**: Start at **10 qubits** (2 layers × 5 universes) with classical sim. Target 14 qubits on quantum hardware.

---

## 3. FAILURE MODES & KILL CRITERIA

### 3.1 Pre-Training Gates (Must Pass Before Scaling)

| Gate | Test | Pass Criteria | If Fail → |
|------|------|---------------|-----------|
| **G1: Classical Baseline** | Train GPT-2 2M on same data | Val ppl < 50 | Fix data/tokenizer |
| **G2: Quantum Forward** | 10-qubit forward pass < 1s | ✓ | Use PennyLane/JAX |
| **G3: Gradient Correctness** | Parameter-shift vs finite-diff | Rel error < 1e-3 | Fix autodiff |
| **G4: Barren Plateau** | Gradient norm at init > 1e-4 | ✓ | Fix initialization |
| **G5: Learning Signal** | 100 steps loss decreases | NLL ↓ 10% | Fix loss/architecture |
| **G6: Universe Separation** | Mutual info < 0.1 bits between universes | ✓ | Add orthogonality loss |
| **G7: Memory Retrieval** | Resonance > 0.5 for saved pattern | ✓ | Fix memory |
| **G8: Scaling** | 12-qubit sparse Trotter works | < 10s/step | Optimize/accept classical limit |

### 3.2 Training Kill Criteria (Stop Immediately If)

| Metric | Threshold | Action |
|--------|-----------|--------|
| Val perplexity > 2× classical baseline | After 10k steps | **KILL** — no quantum advantage |
| Gradient norm < 1e-6 (barren) | After 1k steps | **KILL** — fix init/architecture |
| Quantum eval time > 10s/step | At any point | **KILL** — need hardware |
| Memory resonance → 0 | After 5k steps | **KILL** — memory broken |
| Universes converge (MI > 0.5) | After 5k steps | **KILL** — add orthogonality |
| NaN in quantum state | Ever | **KILL** — numerical instability |

---

## 4. MINIMAL VIABLE EXPERIMENT (MVE)

### 4.1 Scope: 10 Qubits, 2M Classical Params, Classical Simulation

```
Grid: 2 layers × 5 universes = 10 qubits
Universes: [Logical, Emotional, Temporal, Spatial, Creative]
Classical: 2M params (embed + edge MLPs + norms + output)
Quantum: 10 qubits, classical sim (scipy/PennyLane)
Data: 100M tokens (FineWeb-Edu subset)
Steps: 50,000 (≈1 epoch)
Batch: 32 sequences × 512 tokens
LR: 3e-4 (cosine, 1k warmup)
```

### 4.2 Hypotheses to Test (Pre-Registered)

| H# | Hypothesis | Metric | Success Threshold |
|----|------------|--------|-------------------|
| H1 | Quantum model learns faster than classical | Val ppl at step 10k | 10% lower |
| H2 | Universes specialize | MI(Logical, Emotional) < 0.1 | ✓ |
| H3 | Memory improves few-shot | In-context accuracy | +5% vs no-memory |
| H4 | Annealing > unitary | Final val ppl | 5% lower |
| H5 | Entanglement correlates with reasoning | MI vs task accuracy | r > 0.3 |

### 4.3 Baselines (Must Run First)

| Baseline | Params | Config | Expected Val PPL |
|----------|--------|--------|------------------|
| **GPT-2 Tiny** | 2M | 6L, 8H, 256D, 512ctx | ~45 |
| **Raslila Classical** | 2M | Same classical, no quantum | ~50 |
| **Raslila 10Q** | 2M + 10Q | Full model | Target: <45 |

---

## 5. MISSING CODE MODULES (Priority Order)

### P0 — Blockers (Week 1-2)
- [ ] `data.py` — Tokenizer, dataset, dataloader
- [ ] `train.py` — Training loop with all infrastructure
- [ ] `pennylane_interface.py` — JIT quantum forward/backward
- [ ] `stochastic_parameter_shift.py` — Subsample params for gradient

### P1 — Critical (Week 2-3)
- [ ] `eval.py` — Perplexity, generation, ablations
- [ ] `curriculum.py` — Sequence length ramp, difficulty
- [ ] `orthogonality_loss.py` — Universe separation regularizer
- [ ] `checkpoint.py` — Save/resume quantum+classical state

### P2 — Needed for Scale (Week 3-4)
- [ ] `distributed.py` — DDP for classical, single GPU for quantum
- [ ] `mixed_precision.py` — bf16 classical + complex128 quantum
- [ ] `gradient_checkpoint.py` — Activation recomputation
- [ ] `profiling.py` — Quantum/classical time breakdown

### P3 — Quantum Hardware (Week 4+)
- [ ] `ionq_backend.py` / `ibm_backend.py` — Hardware execution
- [ ] `error_mitigation.py` — ZNE, PEC, CDR
- [ ] `circuit_optimization.py` — Decomposition, routing

---

## 6. REALISTIC COMPUTE BUDGET

### Classical Simulation (10 qubits)
| Resource | Estimate |
|----------|----------|
| GPU | 1× A100 80GB |
| Quantum sim/step | 0.5-2s (scipy.expm) |
| Classical forward/backward | 0.05s |
| Total/step | 1-3s |
| 50k steps | 14-42 hours |
| Memory | < 20 GB |

### Quantum Hardware (12-16 qubits)
| Resource | Estimate |
|----------|----------|
| QPU time | $1-5k/hour (IonQ/IBM) |
| Shots per gradient | 10,000 |
| Gradient evaluations | 2M params × 2 (param-shift) = IMPOSSIBLE |
| **Required**: Stochastic gradient (100 params/step) | 100× speedup |

**Hard Truth**: 2M parameter quantum gradients via parameter-shift is **physically impossible** on current hardware. You need:
1. Stochastic parameter-shift (sample 100 params/step)
2. Or: Classical proxy gradients (train classical, distill to quantum)
3. Or: Analog quantum computing (continuous evolution)

---

## 7. ARCHITECTURAL DECISIONS NEEDED NOW

| Decision | Options | Recommendation |
|----------|---------|----------------|
| **Vocabulary** | Char / BPE-8k / BPE-16k / WordPiece | BPE-8k (balance) |
| **Position Encoding** | RoPE / ALiBi / Absolute / None | RoPE (standard) |
| **Normalization** | RMSNorm / LayerNorm / None | RMSNorm (stable) |
| **Residuals** | Pre-norm / Post-norm / None | Pre-norm (standard) |
| **Quantum Layers** | 1 big Hamiltonian / Layered | Layered (2-3) |
| **Universe Count** | 4 / 5 / 6 / 8 | 5 (Logical, Emotional, Temporal, Spatial, Creative) |
| **Memory** | Full topological / Compressed / None | Compressed (100 snapshots) |
| **Readout** | Marginal 3-qubit / Full measurement | Marginal (preserves superposition) |
| **Optimizer** | QNG + AdamW / AdamW only / KFAC | Hybrid (QNG for quantum, AdamW classical) |

---

## 8. EVALUATION PROTOCOL (Strict)

### 8.1 Primary Metric
**Validation Perplexity** on held-out 10M tokens (FineWeb-Edu val split)

### 8.2 Secondary Metrics
| Metric | How Measured | Target |
|--------|--------------|--------|
| Training throughput | tokens/sec | > 10k |
| Memory retrieval@k | Prompt with saved pattern, measure resonance | > 0.5 |
| Universe MI | von Neumann MI between universe pairs | < 0.1 bits |
| Entanglement entropy | Avg von Neumann per universe | > 0.5 (not 0, not max) |
| Few-shot accuracy | 5-shot on reasoning tasks | > random |
| Generation quality | Human eval (100 samples) | > 3/5 |

### 8.3 Ablation Suite (Run at 10k, 25k, 50k steps)
| Ablation | Config |
|----------|--------|
| A1 | No quantum (classical only) |
| A2 | No memory |
| A3 | No interference (α=β=0) |
| A4 | No universes (1 universe = standard) |
| A5 | Unitary vs Annealing |
| A6 | Parameter-shift vs finite-diff |

---

## 8.4 Statistical Rigor
- **Seeds**: 5 seeds minimum (42, 123, 456, 789, 999)
- **Confidence**: Report mean ± std
- **Significance**: p < 0.05 (paired t-test vs baseline)
- **Logging**: Every 100 steps (loss, grads, entanglement, MI)

---

## 9. SIGN-OFF REQUIRED

Before any training starts, the following must be **reviewed and approved**:

- [ ] **Data Pipeline** — Tokenizer trained, dataset streaming, no contamination
- [ ] **Classical Baseline** — GPT-2 2M trained, val ppl recorded
- [ ] **Quantum Forward** — 10-qubit forward < 1s/step (PennyLane)
- [ ] **Gradient Correctness** — Parameter-shift validated on all param types
- [ ] **Barren Plateau Test** — Gradient norms at init logged
- [ ] **Memory Test** — Resonance > 0.5 on saved pattern
- [ ] **Universe Separation** — MI < 0.1 at init
- [ ] **Compute Budget** — GPU allocated, timeline agreed
- [ ] **Kill Criteria** — Documented, automated monitoring
- [ ] **Ablation Plan** — Pre-registered, code ready

---

## 10. HONEST ASSESSMENT: WHAT 2M PARAMS ACTUALLY BUYS YOU

| Capability | 2M Params (This) | 125M (GPT-2 Small) | 1.5B (GPT-2 XL) |
|------------|------------------|-------------------|-----------------|
| Coherent paragraphs | ❌ | ✓ | ✓ |
| Basic reasoning | ❌ | ~ | ✓ |
| In-context learning | ❌ | Weak | ✓ |
| Code generation | ❌ | ❌ | ~ |
| World knowledge | ❌ | ❌ | Weak |
| **Quantum-specific test** | ✓ (theory) | N/A | N/A |

**2M params is enough for**: 
- Proof-of-concept that quantum gradients flow
- Universe separation measurement
- Memory resonance demonstration
- Entanglement dynamics observation

**2M params is NOT enough for**:
- Any useful language capability
- Beating classical baselines on perplexity
- Generalization beyond memorization
- Publication-worthy language results

---

## 11. RECOMMENDED PATH FORWARD

### Phase 0 (Now - 2 weeks): **Infrastructure Only**
- Build data pipeline, classical baseline, PennyLane interface
- Run all pre-training gates (G1-G8)
- **No training until all gates pass**

### Phase 1 (Weeks 2-4): **Minimal Viable Experiment**
- 10 qubits, 2M params, 50k steps
- Run all ablations
- **If G8 fails → stop, analyze, redesign**

### Phase 2 (Month 2-3): **Quantum Hardware**
- 12-14 qubits on IonQ/IBM
- Stochastic parameter-shift
- Error mitigation
- **If no quantum advantage → pivot to classical distillation**

### Phase 3 (Month 3-6): **Scale or Pivot**
- If quantum advantage proven: seek 50+ qubit access
- If not: extract best ideas (edge MLP, topological memory) → classical transformer

---

## APPENDIX: QUICK REFERENCE — FILES TO CREATE

```
src/raslila/
├── raslila.py                  # ✓ ACTIVE Raslila (EdgeNetwork, TransformerNode, Raslila)
├── data.py                     # P0 — Tokenizer, dataset, dataloader (exists, extend)
├── train.py / trainer.py       # training loops (exists, extend for Raslila)
├── train_config.py             # (scripts/) train from config.yaml
├── eval.py                     # P1 — Perplexity, generation, ablations
├── curriculum.py               # P1 — Length/difficulty ramp
├── checkpoint.py               # P1 — Save/resume
├── distributed.py              # P2 — DDP (nccl server / gloo Colab)
├── mixed_precision.py          # P2 — bf16
├── gradient_checkpoint.py      # P2 — Activation recompute
├── profiling.py                # P2 — Time breakdown
# ---- Quantum research modules (foundation, retained) ----
├── gpu_hamiltonian.py          # GPU Hamiltonian + Krylov evolution
├── pennylane_interface.py      # optional JIT quantum autodiff
├── sparse_trotter.py           # sparse Trotter (>10 qubits)
├── stochastic_parameter_shift.py # sampled quantum gradients
├── classical_baseline.py       # GPT-2-style classical baseline
```

---

**THIS DOCUMENT IS A LIVING CONTRACT. NO TRAINING UNTIL ALL P0 ITEMS ARE BUILT, TESTED, AND GATES G1-G8 PASS.**

**Reviewer Signatures Required:**
- [ ] ML Engineer Lead
- [ ] Quantum Engineer Lead  
- [ ] Research Lead
- [ ] Compute Owner

---

*Generated 2026-09-06. Update as gaps are resolved.*