# 03 — Brain Principles

Every Raslila design choice traces to a specific, cited finding about brains —
not vibes. Where the mapping is weak or contested, we say so.

## 1. Continuous prediction and surprise

**Finding.** Human ECoG recordings (Nature Neuroscience, 2022) show the brain
engaged in continuous next-word prediction *before* word onset, computes
prediction error after, and represents words with contextual embeddings — the
same three principles as an autoregressive LM.

**Our model.** Byte-level causal next-byte prediction. This is not a metaphor;
it is the same computation, and it is the reason Raslila's activations map onto
brain responses the way GPT-2's do.

## 2. Hierarchical, multi-timescale prediction

**Finding.** (Caucheteux et al., 2022, 304 subjects, fMRI) The cortex is
hierarchical: frontoparietal regions forecast *distant and abstract* content
while temporal regions forecast the immediate. LMs predict only t+1, which
predicts bland text and repetition loops.

**Our model.** `mtp_steps` — auxiliary heads predicting t+2..t+K. Implemented,
A/B-tested, currently a **tie**; the most likely of the dormant mechanisms to
earn its place on a full-length run.

## 3. Sparse activation

**Finding.** <2% of cortical neurons are active at any moment; connectivity is
also sparse (<5% local). Winners-take-all circuits (lateral inhibition) produce
this.

**Our model.** `readout_topk` (k-winners-take-all over regions). Measured
*slightly worse* than dense pooling at 150 steps. Sparsity buys energy and
robustness, not short-horizon loss — so it stays off, but it is the right lever
if efficiency ever matters.

## 4. Global workspace

**Finding.** Baars/Dehaene: a small capacity workspace binds selected content
into one coherent percept and broadcasts it, which is what makes information
globally available. Crucially, GWT is a *binding* theory — the opposite of
mixture-of-experts routing.

**Our model.** `workspace_slots` — regions compete (softmax) to write into K
slots, then read the broadcast back; linear cost. A **tie** in A/B, redundant
while edge messages do the job.

## 5. The binding problem — the deepest one

**Finding.** arXiv 2012.05208 argues this is *the* underlying cause of LLMs
falling short of human-level generalization: their inability to "dynamically and
flexibly bind information that is distributed throughout the network."
Conceptual blending (Fauconnier & Turner) is the cognitive-science account of
how humans fuse unrelated things into genuinely new structures.

**Our model.** This drove the largest measured gain. Two parts:

1. **The grid is the precondition.** Binding theory requires information to
   stay separated at the representational level, "such that their features do
   not interfere with each other, even when composed." Raslila's 125 blocks each
   hold a persistent separate stream. A 1D transformer fuses each position into
   one vector immediately and *structurally cannot* provide this. This is why
   the grid matters, in the most concrete sense.
2. **The composition operator.** Blocks are bound multiplicatively to learned
   roles (Tensor-Product style) into an emergent composite set, and the readout
   attends over that set instead of collapsing it. A weighted mean is *linear* in
   the region set — no composite is reachable through it.

**What we rejected.** Mixture-of-experts. It is the proven route to very large
sparse models, but it is *segregation*: tokens are dispatched to isolated
specialists that never see each other's representations. That is the opposite
mechanism from binding, and it matches the brain poorly — the brain is one
integrated system, not a set of routers. Raslila's growth is likewise dense:
new regions participate immediately, nothing is switched off.

## 6. Multidirectional recurrence

**Finding.** Hierarchical processing in the brain is recurrent and bidirectional —
information moves up and down, refining partial hypotheses, rather than flowing
once through a stack.

**Our model.** `immediate_updates` (Gauss-Seidel cascades) plus configurable
rounds, priority ordering, and settle-to-convergence. Measured −0.026 nats, and
the control experiment (more frozen rounds help nothing) is what makes the case.

## 7. Growth and plasticity

**Finding.** Structural-plasticity work (2025–2026) shows growth and pruning are
*not* symmetric: newborn units are "forward-active but backward-starved," which
is why naive growth destabilizes training. Integration-friendly birth interventions
(Rand. Smooth-Leaky) recover most of the loss. Separately, continual-learning
studies find growing networks retain plasticity where fixed nets lose it.

**Our model.** `grow_regions` clones an existing region rather than
randomly initialising the newborn — pre-integrated birth. Implemented and
tested; growth is off by default (`grow_every: 0`) until a run justifies it.

## Where the analogy breaks

Honesty about limits, since the whole point of the project is the brain:

- A real brain does **not** start empty and grow to 5T parameters. It has ~86B
  neurons fixed genetically and grows *synapses*. The truer analogy is fixed
  massive structure + sparse activation + lifelong weight plasticity — of which
  Raslila has the last two.
- Brains do not use attention-over-256-classes to write language. Our byte-level
  head is an engineering choice for token-free training, not a brain analogue.
- The binding problem is **unsolved**. Our −0.041 nats is evidence the direction
  is real, not that it is solved. It shows multiplicative composition beats a
  weighted mean on our loss — that is a statement about function classes, not
  about cognition.
- **We have the brain's spatial organisation, not its temporal one.** Every
  cortical neuron integrates over time; Raslila carries nothing between tokens
  except attention keys and values. This is the largest remaining gap, and it is
  about *time*, not geometry.
- We have no inhibition. ~20% of cortical neurons are inhibitory interneurons
  and lateral competition is how the cortex decides what wins; every edge message
  in Raslila is additive.

## What the comparison found

[`07_Brain_Gap_Analysis.md`](07_Brain_Gap_Analysis.md) audits all of this
properly: 13 gaps scored against a real brain, the fixes ranked by value per unit
of effort, five places where our own brain analogies do not hold up, and what not
to build. Headline: the top three items on the roadmap are all about time and
sign — persistent per-block state, inhibitory edges, experience replay — not more
geometry.
