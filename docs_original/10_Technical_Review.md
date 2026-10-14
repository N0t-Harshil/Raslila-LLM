---
date: 2026-08-19
time: 14:30
tags:
  - review
  - bugs
  - corrections
  - raslila
  - status/review
---

# Technical Review – Identified Issues

## Introduction

This note documents the six critical bugs found in the 6‑qubit simulation. Each bug is explained with its cause, consequence, and recommended fix. These issues prevent the simulation from being a valid proof of the Raslila concept. The review was performed on 2026-08-19 and reflects the state of the code as originally written.

## 1. α/β Degeneracy

**Location**: `NormalizedHamiltonian.construct_dense`  
**Code snippet**:
```python
if k == 0 or k == 2:
    op_X = np.kron(op_X, X)
...
H += self.alpha * op_X + self.beta * op_X
```

**Problem**: Both \(\alpha\) and \(\beta\) multiply the same Pauli string \(X_0 X_2\). Thus, the actual term is \((\alpha+\beta)X_0X_2\). The parameters are not separately identifiable; their gradients will be identical, and the QFIM will be singular or nearly singular. This means QNG cannot distinguish between them, and training may fail or behave erratically.

**Fix**: Use non-commuting operators, for example \(\alpha X_0 X_2 + \beta Y_0 Y_2\), or use different qubit pairs. This will make the parameters independent and the QFIM invertible.

## 2. QNG Optimizer Out of Sync

**Location**: Main training loop  
**Code snippet**:
```python
qng = QNG_Optimizer(params=[hamiltonian.alpha, hamiltonian.beta], lr=0.05)
...
hamiltonian.alpha = squash_sin(hamiltonian.alpha + gamma * 0.8)
hamiltonian.beta = squash_sin(hamiltonian.beta + gamma * 0.2)
...
new_params = qng.step(grad_vec, qfim)
```

**Problem**: The `qng` object stores `self.params` initialized to [0.5, 0.5]. Inside the loop, \(\alpha\) and \(\beta\) are modified by memory injection, but `qng.params` is never updated to reflect these changes. When `qng.step` is called, it updates from its stale internal parameters, effectively discarding the memory injection and updating from the wrong point. This creates a systematic bias.

**Fix**: Before calling `qng.step`, sync with the current Hamiltonian values:
```python
qng.params = np.array([hamiltonian.alpha, hamiltonian.beta])
```

## 3. MLP Gradient Shape Error

**Location**: `TrainableMLP.gradient`  
**Code snippet**:
```python
return {'W': np.outer(p_vector, delta_raw * (1-hidden**2) @ self.W_out.T),
        'b': (delta_raw * (1-hidden**2) @ self.W_out.T).flatten(),
        'W_out': delta_raw * hidden.reshape(-1, 1),
        'b_out': np.array([delta_raw])}
```

**Problem**: The expression `(1-hidden**2) @ self.W_out.T` attempts to multiply a vector of shape `(8,)` by a matrix of shape `(1,8)`, which raises a `ValueError` due to incompatible dimensions. The gradient computation for the hidden layer is incorrect. This will crash the training loop on the first iteration.

**Fix**: Compute gradients correctly using backpropagation:
```python
grad_hidden = delta_raw * self.W_out[:, 0]
grad_W = np.outer(p_vector, grad_hidden * (1 - hidden**2))
grad_b = grad_hidden * (1 - hidden**2)
grad_W_out = delta_raw * hidden.reshape(-1, 1)
grad_b_out = np.array([delta_raw])
```

## 4. `grad_J` Uses Wrong Perturbation

**Location**: Main loop, after β shift  
**Code snippet**:
```python
grad_J = (loss_p - loss) / delta
```
Here `loss_p` is the loss after perturbing \(\beta\), not after perturbing \(J_{AB}\). Thus, the gradient for the MLP is actually the \(\beta\) gradient, not the coupling gradient. The MLP is trained with a gradient that has nothing to do with its output.

**Fix**: Perform a separate finite difference on \(J_{AB}\):
```python
delta_J = 0.01
J_old = hamiltonian.J[(0,4)]
hamiltonian.J[(0,4)] = J_old + delta_J
loss_p = ...  # recompute loss
hamiltonian.J[(0,4)] = J_old - delta_J
loss_m = ...
hamiltonian.J[(0,4)] = J_old
grad_J = (loss_p - loss_m) / (2*delta_J)
```

## 5. Target State Unreachable

**Location**: Hamiltonian construction and target definition  
**Problem**: The Hamiltonian contains only \(Z_0Z_4\) and \(X_0X_2\). Qubits 1, 3, and 5 have no terms acting on them; they evolve under the identity. The initial state is uniform superposition. For any passive qubit, the probability of measuring 0 or 1 remains 1/2. Therefore, the maximum probability for any specific 6-bit target is \(1/2^3 = 1/8\). The loss target `TARGET = 1` is impossible to reach.

**Fix**: Either add terms on all qubits (e.g., transverse fields \(X_i\) for all \(i\)) or measure only the active qubits and set a 3-bit target. The simplest fix is to define the target on the active subspace and ignore passive qubits in the measurement.

## 6. "Inference is Cooling" Misrepresentation

**Location**: Throughout documentation  
**Problem**: The documents describe inference as a cooling process where the system settles into the lowest energy state. However, the code uses fixed unitary evolution \(e^{-iH\pi/2}\). There is no dissipation, temperature, or annealing schedule. This is not quantum annealing; it is a deterministic rotation.

**Fix**: Either change the narrative to accurately describe unitary evolution, or implement a true annealing process with a time-dependent Hamiltonian and dissipation. For the current proof-of-concept, unitary evolution is acceptable, but the metaphor must be adjusted.

## Summary

All six issues must be fixed before the simulation can be considered a valid proof of the Raslila concept. The fixes are relatively straightforward but require careful implementation and testing.

## Status

<span style="color:#b56576">Review completed – fixes required.</span>