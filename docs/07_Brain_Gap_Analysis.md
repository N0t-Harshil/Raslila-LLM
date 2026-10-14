# 07 — Brain Comparison: What Raslila Still Lacks

An honest audit of the deep grid against a real brain. For each gap: what the
brain does, what Raslila does, why it matters, and whether it is worth building.

This document deliberately separates **three different things** that are easy to
conflate:

- **A missing mechanism** — the brain has it, we do not.
- **A missing dynamic** — the brain does it *in time*; we do it in one static pass.
- **A bad analogy** — we claim a brain parallel that does not hold up. These are
  the most valuable entries, because they cut against the project's own thesis.

Bottom line up front: Raslila reproduces the brain's **spatial** organisation
fairly well and its **learning** organisation not at all. It has no persistent
temporal state, no inhibition, no second memory system, no replay, and no
development. Those five gaps are larger than anything left in the architectural
fine-tuning.

---

## Scorecard

| Brain property | Raslila | Verdict |
|---|---|---|
| 2D sheets stacked in depth | 5 layers × 5×5 sheets | **Good** — the core thesis holds |
| Distinct cell types per layer | 5 homogeneous copies | **Missing** — see G7 |
| Lateral connectivity | 4-neighbour within sheet | **Partial** — no distance decay, no long-range tuning |
| Recurrent in time | none (attention window only) | **Missing, biggest gap** — G1 |
| Inhibition (~20% interneurons) | zero, all traffic excitatory | **Missing** — G2 |
| Sparse activation (~1–5%) | dense: all 125 blocks every token | **Missing** — G3 |
| Thalamic relay + gating | none, input hits all blocks | **Missing** — G4 |
| Oscillations (gamma/theta) | none | **Missing** — G5 |
| Two memory systems | one, gradient-trained | **Missing** — G6 |
| Replay / consolidation | none | **Missing** — G8 |
| Neuromodulation | a boolean gate (A/B: no gain) | **Partial, misimplemented** — G9 |
| Plasticity shaped by use | gradient descent, uniform in time | **Missing** — G10 |
| Local dendritic computation | point-to-point MLP, no compartments | **Missing** — G11 |
| Energy limit (~20W) | unconstrained | **Deliberately absent** — G12 |
| Sparse noisy spike codes | dense analog | **Deliberately absent** — G13 |
| Binding problem | multiplicative binding, best measured gain | **Partial** — mechanism measured, problem not claimed |

And five entries in that table are **deliberately absent** rather than missing:
energy limits (G12), spike codes (G13), and so on. Those are documented as
non-goals, not oversights.

---

## The big five

### G1 — No persistent state over time. The largest gap.

**Brain.** Every cortical neuron integrates over time: membrane potential
accumulating inputs, firing when threshold crossed. Information persists in
*membrane state*, not in a token buffer. A human recognises a word in ~150 ms
from ~4 letters of context; there is no 150 ms of tokens to attend over.

**Raslila.** The only thing crossing a token boundary is attention keys/values.
Between tokens the network has no state at all. Every forward pass is
independent. This means the model has *no recurrence in time*, only recurrence
across space and depth.

**Why it matters.** This is the single biggest structural difference remaining.
A 1D transformer and a brain differ here far more than a 1D transformer and a
2D grid do — and we spent all our effort on the second difference.

**Fix.** Give each block a persistent state: a gated linear recurrence
(`h_t = a·h_{t−1} + (1−a)·f(x_t)`, with learned per-channel `a`) or fast-weight
updates. Costs ~nothing in parameters, gives unbounded effective context, and is
the standard modern answer (Mamba, RWKV, Titans). Crucially, per-channel `a`
gives **different neurons different timescales**, which is itself brain-like —
and gets us the multi-scale prediction that `mtp_steps` tried and failed to
deliver. This is the highest-value change available.

### G2 — No inhibition at all.

**Brain.** ~20% of cortical neurons are inhibitory interneurons. They generate
lateral inhibition, which is what produces sparse codes, gain control, and
competition between representations. Inhibition is not a minor detail; it is how
the cortex decides *what wins*.

**Raslila.** Every edge message is additive. `alpha` is learned but unconstrained
in sign, and normalisation is RMSNorm, not competition. There is no mechanism
where one block can suppress a competing block.

**Why it matters.** Binding requires competition — "A and B" is only meaningful
if unrelated C is *not* also bound. Our binding is currently permissive: every
block can be bound to every role. Inhibition is what makes selection meaningful,
and it is the mechanism behind real sparsity.

**Fix.** Sign-constrain edge outputs: split each edge MLP into an excitatory and
an inhibitory half, or give edges a learned negative gate. Cheap. Secondarily,
competition-style readout (winner-take-all over *roles*) rather than softmax over
blocks would make binding selective. Note our `readout_topk` A/B came out
slightly *worse* — but that was sparsity applied at the wrong place (output
selection) rather than as lateral competition between blocks.

### G3 — Dense where the brain is sparse.

**Brain.** Fewer than ~5% of neurons are active at any moment (often quoted <2%
for spikes). Cortex is sparse in space, in time, and in synapse count.

**Raslila.** All 125 blocks run a full transformer over the whole sequence on
every forward pass. 7.4 s/step on CPU, 125 redundant transformer passes.

**Why it matters.** This is a pure compute tax, and it is what makes the deep grid
expensive. But — and this is the uncomfortable part — the A/B was *slightly
worse* with sparse readout. So the honest statement is: we pay 125× compute and
cannot yet turn it into an efficiency win. That is the weakest economic argument
in the project.

**Fix.** Do not solve this with MoE — rejected for good reason (see
[03](03_Brain_Principles.md) §5). Solve it with **inhibition-driven** sparsity
(G2): once blocks compete, inactive blocks can be *skipped*, not merely
down-weighted. That gives real FLOP savings, which top-k at readout never did.

### G4 — No thalamus.

**Brain.** Every sensory signal except olfaction passes through the thalamus
before reaching cortex. The thalamus is not a relay, it is a gate: it decides
what is allowed into awareness, and it pulses in sleep-state-dependent rhythms.

**Raslila.** `input_proj` broadcasts the token embedding to all 125 blocks
identically at line 860. Nothing selects what matters.

**Why it matters.** Cortical input is *modulated*, not uniform. Attention in a
brain is partly this: a gain-control stage before the cortex proper.

**Fix.** A learned per-block gain over the injected input — a cheap soft gate
over blocks at the input, rather than at the readout. Much better placed than
`readout_topk` because it saves the compute too. Roughly 10 lines.

### G5 — No oscillations.

**Brain.** Gamma (30–80 Hz) is associated with binding and feature integration;
theta (4–8 Hz) with memory and hippocampal replay; alpha (8–12 Hz) with
suppression. Timing and phase carry information, not just presence.

**Raslila.** Nothing. We are a static feedforward-plus-spatial-recurrent
network.

**Why it matters.** Oscillations are how the brain separates simultaneous
representations in *time* rather than by hard routing. Arguably the same problem
binding solves in *space*. We solved it in space and not in time.

**Fix.** Honestly, hard. Explicit simulated oscillation means phase state across
the sequence, which overlaps heavily with G1 — if we implement persistent state
with per-channel decay constants, we get frequency-dependent filtering for free
and can measure whether "gamma-like" fast channels help binding. Cheap version:
just give G1's recurrence per-channel decay rates and check if different
timescales emerge. Only worth building if that shows something.

---

## The structural gaps

### G6 — One memory system, not three.

**Brain.** Three learning systems with different rules and timescales:

| System | Rule | Timescale | Role |
|---|---|---|---|
| Hippocampus | one-shot, sparse, allosynaptic | seconds | episodic: *where/when* |
| Neocortex | slow, many-shot, gradient-like | days–years | semantic: *what* |
| Cerebellum / basal ganglia | error-driven, supervised-ish | fast | procedural, action |

**Raslila.** One set of weights, one rule (backprop), one timescale. Every
generalisation is smeared across all parameters simultaneously.

**Why it matters.** The hippocampus is why humans learn a *specific* event from
one exposure and the neocortex is why we then generalise slowly. We have only the
slow half, and no episodic memory at all — so the model cannot form a memory of
anything, only statistical tendencies.

**Fix.** A sparse episodic store: a key-value memory over past context, written
sparsely, read by similarity, trained with a distinct objective and a distinct
learning rate. This is the Hopfield-network route (Ramsauer et al. 2024) and it
is a well-trodden path. Expensive to get right; conceptually the most faithful
single addition.

### G7 — Homogeneous layers.

**Brain.** The six cortical layers are not copies. Layer 4 receives thalamic
input and is dominated by spiny stellate cells; layer 5 sends output to the
brainstem and is thick; layer 2/3 does long-range lateral integration; layer 6
sends feedback to the thalamus. Different cells, different wiring, different
receptive fields — and receptive field size roughly doubles per layer.

**Raslila.** Five structurally identical layers, distinguished only by a
positional role embedding.

**Why it matters.** Homogeneous depth is *less* expressive per layer than
heterogeneous depth. Our measured feedback gain suggests depth is doing work; the
cheap upgrade is making each layer do a *different* job.

**Fix.** Per-layer config: distinct hidden widths, and — most importantly —
**layer-specific input injection** (layer 1 gets the token embedding, layer 5
gets a global summary, middle layers get neither). Receptive-field doubling via
increasing attention span per layer is a direct and cheap analogy.

### G8 — No replay, no consolidation.

**Brain.** During sleep, hippocampal cells replay the day's sequences at 10–20×
speed; replay is what consolidates temporary memories into the neocortex.
Waking rest and sleep are not interchangeable.

**Raslila.** Training is one pass over text, no replay, no consolidation phase.
Nothing is ever re-visited.

**Why it matters.** Replay is the brain's defence against catastrophic
forgetting, and it is what makes *continual* learning possible at all. Directly
relevant to G10.

**Fix.** Cheap and high-value: during training, with some probability, sample a
segment from *previously seen* text and give it extra weight, or re-run old
batches at a low rate. This is experience replay, and it is the standard fix for
forgetting in continual learning. Costs nothing but compute.

### G9 — Neuromodulation, misimplemented.

**Brain.** Dopamine, serotonin, acetylcholine and noradrenaline modulate
*plasticity and gain* globally and by region. Dopamine is not a signal, it is a
learning-rate modulation: "this was unexpected, update more."

**Raslila.** `gated_messages` is a boolean gate on edge traffic. A/B: no gain.

**Why the A/B was uninformative.** A boolean gate is not neuromodulation. The
brain version is a *continuous scalar derived from prediction error* that scales
the update. We tested a capacity-control knob, not a learning-rate signal, and
concluded the mechanism doesn't work. That conclusion is not supported.

**Fix.** A global surprise signal (readout logit entropy, or TD-style error)
feeding a learned multiplier on plasticity or on edge gain. Different mechanism,
worth re-testing. Flag: this is where we were wrong, and the honest move is to
re-open it rather than leave it in the "tied" pile.

### G10 — Plasticity not shaped by use.

**Brain.** Plasticity is not uniform. Critical periods open and close; heavily used
circuits lose plasticity while unused ones retain it; structural plasticity
prunes what is unused. Learning rate is a *function of experience*.

**Raslila.** Every parameter gets the same optimizer, at the same rate, forever.

**Why it matters.** This is the mechanism-level version of G6, and it is what
makes lifelong learning possible in biology. It is also what growth needs: a
model that grows needs its new units to have usable plasticity, and old units to
stay stable.

**Fix.** Per-block usage counters → per-block learning-rate scaling, or simply a
higher LR for recently grown blocks. Cheap, and directly testable.

### G11 — No dendritic structure.

**Brain.** A single neuron receives ~10,000 synapses across thousands of dendritic
branches, each branch performing local nonlinear integration, and spines are the
individual plastic units. Computation is distributed across a tree, not across
flat fan-in.

**Raslila.** Each edge is a small MLP over two 256-d vectors, with no dendritic
sub-compartments. `edge_hidden_dim` (128) is the only nod to local width.

**Why it matters.** Our "edges as neural networks" claim is real but modest: an
edge MLP is not a dendrite. A dendritic compartment would give per-synapse
nonlinear gating and local saturation, which fan-in weights structurally cannot.

**Fix.** If pursuing: make the edge MLP a small *tree* with per-compartment
saturation and a spine-level gate. Genuinely more brain-like, but expect the same
outcome as `edge_hidden_dim` tuning: marginal.

### G12 — Energy is unconstrained. *(Deliberate, but must be stated.)*

**Brain.** ~20 W for the whole brain, ~2% of body mass, ~20% of resting energy.
Metabolic cost is a hard evolutionary constraint that shaped sparsity (G3),
locality (short axons), and myelination.

**Raslila.** No energy term anywhere. Cost is whatever the hardware gives us.

**Why it matters.** Not for accuracy — for *research honesty*. Raslila's deep
grid is ~125× more redundant per token than a brain pathway, and it has never had
to answer for it. Any claim that this architecture is "efficient" needs an energy
denominator, and it does not have one.

**Fix.** Add FLOPs-per-token and active-parameter-count to the benchmark output.
Not a model change — a measurement change, and arguably overdue.

---

## Where the analogy is wrong, not just incomplete

The most useful entries here, because they cut against the thesis.

**Bad analogy 1 — "5 layers ≈ cortical depth."** Real cortex is ~2–4 mm of tissue
with 6 layers and *hundreds* of mm of lateral extent, folded into gyri. Our
5 × 5×5 is a 5-cell-deep sandwich, not a folded sheet. Folding exists to
maximise surface area within a skull — a geometric constraint we have no
analogue for. Calling 5 layers "cortical depth" overstates the correspondence.

**Bad analogy 2 — "125 blocks ≈ 125 brain regions."** Real cortical columns are
~50 µm wide and contain on the order of 10⁴ neurons; "region" is a functional,
not anatomical, unit. A block is a convenient engineering unit with no
neuroanatomical counterpart.

**Bad analogy 3 — "Edges ≈ synapses."** 618 learned MLPs is a toy relative to
10¹⁴–10¹⁵ synapses, and unlike real synapses ours are all present, all strong,
all fast, and structurally never pruned. Our growth adds blocks; the brain grows
*connections within* blocks.

**Bad analogy 4 — "Positional role embeddings ≈ cortical maps."** We feed
(col, row, layer) through a 3→256 MLP and call it a map. Tonotopic and
retinotopic maps are established by experience through a developmental process we
do not model (G10).

**Bad analogy 5 — "Binding solves the binding problem."** Our −0.041 nats shows
multiplicative composition is *useful*, i.e. a weighted mean is a restricted
function class and Tensor-Product style composition is not. That is a real
result and it is about our loss function. It says nothing about whether the
binding problem in brains is solved. We over-claim here and should stop.

---

## Status: four built, one active

| # | Change | Gap | Built | Active | Measured on loss |
|---|---|---|---|---|---|
| 0 | **Bidirectional depth sweeps** | §11 | yes | **yes — under test** | pending real run |
| 2 | Signed / inhibitory edges | G2 | yes | no | no significant change |
| 1 | Persistent per-block state | G1 | yes | no | no significant change |
| 3 | Experience replay | G8 | yes | no | not tested |
| 4 | Input-side gating | G4 | no | no | — |
| 5 | Layer heterogeneity | G7 | no | no | — |
| 6 | Neuromodulation as surprise→plasticity | G9 | no | no | — |
| 7 | Usage-scaled plasticity | G10 | no | no | — |
| 8 | Episodic Hopfield memory | G6 | no | no | — |
| 9 | FLOPs + active-params in benchmarks | G12 | no | no | — |
| 10 | Inhibition-driven compute skipping | G3 | no | no | — |

The original ordering below is retained as written, but note what actually
happened: **items 1–3 are built, and all three failed to show a loss benefit at
200 steps.** That is not evidence they are useless — 200 steps is ~0.2 epochs, so
the benchmark was measuring noise (§12 of the research log). It does mean nothing
gets activated until it wins a real training run.

**Why only one is active.** Turning several on together would make the result
unattributable, which is exactly the mistake that produced a confident 3-seed
"win" for inhibition that a 6-seed rerun erased. Order of operations:
1. Train `depth_sweeps=2` against `depth_sweeps=1`. Keep 2 only if it wins.
2. Add `inhibit`. Keep only if it wins.
3. Add `state_dim`. Keep only if it wins.
4. Add `replay_prob`. Keep only if it wins.

Every comparison at ≥5 seeds, paired, with the seed spread reported alongside
the effect.

| # | Change | Gap | Cost | Expected value | Risk |
|---|---|---|---|---|---|
| 1 | **Persistent per-block state** (gated linear recurrence) | G1 | Medium | **Very high** | Medium — may change convergence |
| 2 | **Signed / inhibitory edges** | G2 | Low | **High** | Low |
| 3 | **Experience replay in training loop** | G8 | Low | **High** | Very low |
| 4 | **Input-side gating instead of readout top-k** | G4 | Very low | Medium-high | Low |
| 5 | **Layer heterogeneity** (per-layer width, input injection, receptive-field growth) | G7 | Low | Medium | Low |
| 6 | **Neuromodulation as continuous surprise→plasticity** | G9 | Low | Medium | Low — needs re-benchmark |
| 7 | **Usage-scaled per-block plasticity** | G10 | Low | Medium | Low |
| 8 | **Episodic Hopfield memory** | G6 | High | Medium-high | High |
| 9 | **FLOPs + active-params in benchmarks** | G12 | Very low | Indirect, but overdue | None |
| 10 | **Inhibition-driven compute skipping** | G3 | Medium | Efficiency, not loss | Medium |
| 11 | Oscillatory timescales from G1's decay rates | G5 | Low | Unknown | Only after #1 |
| 12 | Dendritic compartment edges | G11 | High | Low | High |

**The uncomfortable conclusion.** Items 1–3 are all about *time* and *sign* — not
about geometry. We have been refining the spatial organisation, which was the
thesis, and the thesis is now in good shape. What is missing is not another
lateral connection; it is that the model has no memory of anything that is not in
the attention window, cannot suppress a wrong answer, and never re-visits what it
learned.

## What not to build

- **MoE / routed sparsity** — rejected; it is segregation, the opposite of binding.
- **More geometric variants** (larger grids, diagonal edges, folded layouts). The
  geometry is settled and flat in returns; we measured lateral off as the only
  connectivity setting that matters.
- **A second binding mechanism.** One works and is measured. A second is a tie at
  best.
- **Anything requiring a 5T budget.** See [06_Known_Limitations.md](06_Known_Limitations.md).

---

## Related

- [03 Brain Principles](03_Brain_Principles.md) — the neuroscience ↔ ML mapping this extends
- [02 Research Log](02_Research_Log.md) — measured results, including the failures
- [05 Scorecards](05_Scorecards.md) — per-component scores
- [06 Known Limitations](06_Known_Limitations.md) — what 5T would actually require