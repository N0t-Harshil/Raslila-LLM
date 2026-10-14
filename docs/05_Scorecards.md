# 05 — Scorecards

Scored 0–10 against a same-budget GPT stack on identical data, and against
published results. 10 = best in class. Full table also in
`../SOTA_COMPARISON.md`.

## Components

| # | Component | Score | Note |
|---|-----------|-------|------|
| 1 | Causal masking | **10** | Correctness, not a feature. Without it the objective leaks. |
| 2 | Binding / conceptual blending | **8** | −0.041 nats on the retired setup; unreplicated on the current setup. |
| 3 | Immediate cascades (multidirectional) | **8** | −0.026 nats on the retired setup; unreplicated on the current setup. |
| 4 | node_depth | **8** | Converts grid breadth into real per-token depth. |
| 5 | RoPE | **9** | Modern block recipe; the −0.27 figure is unreplicated on current setup. |
| 6 | RMSNorm + pre-norm + final norm | **8** | Final norm is load-bearing (init loss 110 without it). |
| 7 | SwiGLU | **8** | Consistent literature win (−0.11). |
| 8 | Edge networks (edges compute) | **8** | Core novelty; trains end-to-end. |
| 9 | BPE vocabulary | **?** | Current 199,995-token setup; prior byte-level figures do not transfer. |
| 10 | Init scheme (0.02 emb, zero-init roles) | **8** | Both measured at init (5.6 ≈ uniform 5.55). |
| 11 | Learned node readout | **7** | Strictly more expressive than mean pooling; cheap. |
| 12 | Deep-grid topology (2D sheets, stacked, with feedback) | **7** | Structurally sound; unproven vs plain stacking at scale. |
| 13 | Persistent state over time | **2** | **Biggest remaining gap.** No recurrence between tokens — only attention across the window. A brain integrates over time; we do not |
| 14 | Inhibition / lateral competition | **1** | ~20% of cortical neurons are interneurons; we have zero. Every message is additive, so blocks never compete |
| 15 | Layer heterogeneity | **3** | 5 structurally identical layers. Real cortex has 6 layers with distinct cell types and doubling receptive fields |
| 16 | Two memory systems (hippocampus + neocortex) | **1** | One weight set, one rule, one timescale. No episodic memory at all |
| 17 | Replay / consolidation | **1** | Never revisits anything. The standard fix for catastrophic forgetting, and we have no continual learning |
| 13 | Message-passing rounds | **7** | Works; more rounds cost linearly, gain little. |
| 14 | Node role identity | **6** | Breaks symmetry; effect entangled with edges. |
| 15 | QK-norm | **7** | Cheap insurance; not isolated yet. |
| 16 | Tied embedding/head | **7** | Needs 0.02 init to not saturate. |
| 17 | Adaptive LR + logging | **7** | Good operability; plateau decay froze learning once. |
| 18 | Corpus pipeline | **7** | Rewritten 2026-10: 1.5B-token openwebtext corpus, auto-fetched. The 448-book version scored the same 7 and is deleted. |
| 19 | α/β interference | **6** | Learns; small consistent effect. |
| 20 | Multi-token prediction | **6** | Tie at 150 steps; best dormant candidate. |
| 21 | Global workspace | **6** | Tie; redundant while edges suffice. |
| 22 | LNS | **5** | Tie — no depth curse to cure at our depth. |
| 23 | Gated messages | **5** | Tie. But tested as a boolean, not as a surprise→plasticity signal — the mechanism was misimplemented. See 07 §G9 |
| 24 | Priority sweeps | **5** | Tie. |
| 25 | Top-k sparse readout | **5** | Slightly worse; efficiency play, not loss. |

### Gaps against a real brain

Scored against the brain, not against a baseline. Full audit and fixes:
[07_Brain_Gap_Analysis.md](07_Brain_Gap_Analysis.md).

| # | Brain property | Score | Honest note |
|---|---|---|---|
| 26 | Persistent state over time | **2** | **Biggest remaining gap.** No recurrence between tokens — only attention across the window. A brain integrates over time; we do not. **Built (`state_dim`), off** — no measurable loss gain at 200 steps |
| 27 | Inhibition / lateral competition | **1** | ~20% of cortical neurons are interneurons; we have zero. Every message is additive, so blocks never compete. **Built (`inhibit`), off** — a 3-seed win did not survive a 6-seed rerun |
| 28 | Layer heterogeneity | **3** | 5 structurally identical layers. Real cortex has 6 layers with distinct cell types and doubling receptive fields |
| 29 | Two memory systems (hippocampus + neocortex) | **1** | One weight set, one rule, one timescale. No episodic memory at all |
| 30 | Replay / consolidation | **1** | Never revisits anything — the standard fix for catastrophic forgetting, and we have no continual learning. **Built (`replay_prob`), off** |
| 31 | Input gating (thalamic relay) | **2** | Token embedding broadcasts to all 125 blocks identically. Attention in a brain is partly pre-cortical gain control |
| 32 | Oscillations (gamma/theta) | **1** | Nothing. We solved binding in space, not in time |
| 33 | Neuromodulation | **3** | Present but as a boolean gate, not a continuous surprise signal modulating plasticity |
| 34 | Use-shaped plasticity / critical periods | **2** | Uniform LR for every parameter, forever. No critical periods, no per-unit plasticity decay |
| 35 | Settle-to-convergence | **5** | Smoke-tested only. |
| 36 | **Bidirectional depth flow** | **?** | **Active, under test.** The feedback edges existed in the graph but were never traversed; `depth_sweeps=2` fixes it. Cost: 2× compute, 0 extra parameters. Loss effect unknown |

## Candidate layers reviewed, not adopted

| Candidate | Fit | Verdict |
|-----------|-----|---------|
| Depth-Attention / MoDA | 7 | Strong 2025–26 results; natural fit for stacked blocks. Implementation cost moderate-high. |
| Multi-token prediction | 7 | Proven (DeepSeek MTP); training-side only. Next training upgrade. |
| MoE (mixture-of-experts) | 6 | Proven route to scale, **deliberately rejected** — it is segregation, not binding, and matches the brain poorly. |
| MegaByte / BLT patching | 7 | The principled byte-level direction; a full redesign. Long-term. |
| Muon optimizer | 6 | Promising AdamW alternative; unproven in our loop. Cheap to A/B. |
| Differential attention | 6 | Gains shown at 3B+, unknown at 30M. |
| GQA / MQA | 4 | Inference-memory win, quality-neutral; unneeded at our scale. |
| Z-loss | 3 | Matters for 50k vocabs; negligible at 257. |
| Sliding-window attention | 2 | Irrelevant at seq 128. |

## Honest verdict

| Axis | Score |
|------|-------|
| **As a research architecture** | **8/10** — original where it counts, modern where it matters, honestly tested |
| **As a language model today** | **3/10** — an under-trained 64M net with byte-soup output |
| **As evidence the thesis holds** | **7/10** — beats equal-param GPT, binding and multidirectionality both show real gains |

The historical scorecards reflect earlier data and vocabulary setups. The claimed
loss gains have not been revalidated on the current tokenizer and wiring.
