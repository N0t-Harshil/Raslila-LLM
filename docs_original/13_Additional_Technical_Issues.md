---
date: 2026-08-19
time: 16:00
tags:
  - review
  - bugs
  - activation
  - loss
  - gradients
  - raslila
  - status/review
---

# Additional Technical Issues

## Introduction

This note documents further problems discovered during deeper analysis of the Raslila simulation. These issues go beyond the six already listed in [[10_Technical_Review]] and include silent failures that could make training appear to work while actually learning nothing useful. Each issue is described with its cause, consequence, and recommended fix.

## 1. Squash+Sin Activation Has Zero Gradient at \(x = \pm 1\)

**Original Claim**: "Squash+Sin never kills gradients."  
**Reality**: The derivative of Squash+Sin vanishes exactly at \(x = \pm 1\).

### Derivation

The activation is:

$$
f(x) = \sin\left( \frac{\pi x}{1 + |x|} \right)
$$

Let \(s(x) = \frac{x}{1+|x|}\). Then \(f(x) = \sin(\pi s(x))\). The derivative is:

$$
f'(x) = \pi s'(x) \cos(\pi s(x))
$$

where \(s'(x) = \frac{1}{(1+|x|)^2}\).

At \(x = 1\):

$$
s(1) = \frac{1}{2}, \quad \cos(\pi/2) = 0 \quad \Rightarrow \quad f'(1) = 0
$$

Similarly at \(x = -1\). Therefore, if the MLP outputs a raw value near ±1, the gradient through the activation is zero, causing the MLP to stop learning for that edge. This is a critical bug because it can silently stall training.

**Fix**: Use an activation that is bounded and has non-zero derivative everywhere, e.g., \(\tanh(x)\), or modify Squash+Sin to avoid zero derivative by adding a small linear term. Alternatively, ensure inputs to the activation are kept away from ±1 by scaling.

## 2. Mean Squared Error Loss on Probability Is Weak

The current loss:

```python
loss = (probs[10] - TARGET) ** 2
```

with `TARGET = 1`. Probabilities are bounded in [0,1]. The gradient of MSE with respect to probability is linear: \(2(p-1)\). When \(p\) is small (initial random state ~ 1/64), the gradient is about -2, which is weak and does not provide strong learning signal.

**Better loss**: Negative log-likelihood:

$$
L = -\log(p_{\text{target}})
$$

Gradient: \(-1/p\). This blows up when \(p\) is small, giving strong gradients to push the system toward the target. It also naturally saturates as \(p \to 1\). This is the standard loss for probability distributions.

**Fix**: Replace MSE with `-np.log(probs[10] + 1e-12)`.

## 3. Reflection Feature Grows Unbounded

In the main loop:

```python
reflection = 0.1 + 0.01 * step
```

By step 100, reflection = 1.1. This value is outside the typical input range [0,1] and can cause the MLP's hidden tanh to saturate, leading to vanishing gradients inside the MLP. The other inputs (usage, meaning, temporality) stay roughly in [0,1].

**Fix**: Clip or normalize reflection:

```python
reflection = min(0.1 + 0.01 * step, 1.0)
```

or rescale with a sigmoid.

## 4. Finite Difference Is Not Parameter-Shift

The code estimates gradients using central finite difference with `delta = 0.1`. While this is a valid approximation, it is noisy and not exact. For quantum gates, the **parameter-shift rule** provides exact gradients by shifting parameters by \(\pi/2\). Using finite difference can lead to inaccurate gradients, especially in the presence of noise or for small steps.

**Fix**: Use exact parameter-shift rules for the quantum parameters, or implement automatic differentiation via libraries like PennyLane, TensorFlow Quantum, or Qiskit.

## 5. Qubit Ordering Ambiguity

The Hamiltonian construction uses qubit 0 as the most significant bit (leftmost) in the Kronecker product. The target state is defined as index 10 (binary `001010`). Depending on the endianness convention, the mapping to physical qubits may be reversed. This can cause confusion when interpreting which universe/layer is being measured.

**Fix**: Explicitly define the endianness convention and ensure the target state index corresponds to the desired measurement basis. In quantum computing, qubit 0 is often the least significant bit; it may be safer to define the target in terms of individual qubit measurement probabilities.

## 6. Memory Injection + QNG Sync Mismatch (Reiterated)

This issue is already in [[10_Technical_Review]] but is worth repeating. The memory injection modifies \(\alpha/\beta\), but the QNG optimizer's internal parameters are not synced. This causes the update to be computed at a different point than where the gradient was evaluated, introducing a systematic bias.

**Fix**: Always sync `qng.params` with the current Hamiltonian \(\alpha/\beta\) before calling `qng.step`.

## 7. Hamiltonian Normalization Implicitly Couples All Parameters

The Hamiltonian is normalized by its Frobenius norm:

```python
H = H / norm(H, 'fro')
```

If the magnitude of \(J_{AB}\) increases, the overall norm increases, causing the relative contribution of \(\alpha/\beta\) terms to decrease. This means that the effective learning rate for \(\alpha/\beta\) is not constant but depends on the scale of other parameters. This can interfere with training dynamics.

**Fix**: Use a Hamiltonian with a fixed norm or normalize only the parameter-independent part. Alternatively, use a spectral norm that is less sensitive to the number of terms.

## Summary Table

| Issue | Description | Severity |
|-------|-------------|----------|
| Squash+Sin gradient zero | \(f'(±1) = 0\) exactly | <span style="color:#b56576">Critical</span> |
| MSE loss on probability | Weak gradients for low probability | <span style="color:#b56576">High</span> |
| Unbounded reflection input | Feature >1, can saturate MLP | <span style="color:#d2a679">Medium</span> |
| Finite difference not parameter-shift | Approximate, noisy | <span style="color:#d2a679">Medium</span> |
| Qubit ordering ambiguity | Target state may not match intended qubits | <span style="color:#d4c685">Low</span> |
| Memory injection + QNG sync | Learning point mismatch | <span style="color:#b56576">High</span> |
| Hamiltonian normalization coupling | Implicit parameter coupling | <span style="color:#d2a679">Medium</span> |

These issues must be resolved before the simulation can be considered a valid proof of the Raslila concept.

## Next Steps

- Incorporate all fixes into a corrected version of the code.
- Test each component individually (activation gradient, loss gradient, parameter shift) before full training.
- Document the corrected simulation in a new note.

## Status

<span style="color:#b56576">Additional issues identified – fixes required.</span>