---
date: 2026-08-15
time: 14:00
tags:
  - math
  - equations
  - core
  - raslila
  - status/completed
---

# The Living Mathematics

## Introduction

This note presents the complete mathematical formalism of Raslila. Each equation is explained in detail, including what each symbol means and how it contributes to the overall system. The equations are the logical skeleton of the vision. They have been stress-tested conceptually, but the implementation requires corrections noted elsewhere.

## 1. Multiverse State (Superposition)

$$
|\Psi_{l,u}\rangle = \sum_{k} c_k |k\rangle
$$

**Explanation**:  
The entire system is described by a quantum state \(|\Psi\rangle\) living in a Hilbert space. The indices \(l\) and \(u\) label the vertical layer and horizontal universe, respectively, but for the whole system we can write the state as a superposition over all computational basis states \(|k\rangle\). The coefficients \(c_k\) are complex amplitudes; their squared magnitudes give the probability of measuring the system in state \(|k\rangle\). This equation embodies **superposition**—the system is simultaneously in all possible configurations, with different universes and layers represented in different basis states. The normalization condition is \(\sum_k |c_k|^2 = 1\).

## 2. Dynamic Edge Coupling

$$
J_{ij} = \sin\left( \frac{\pi \cdot \text{MLP}([\text{Usage}, \text{Meaning}, \text{Reflection}, \text{Temporality}])}{1 + |\text{MLP}(\dots)|} \right)
$$

**Explanation**:  
The coupling strength \(J_{ij}\) between qubits \(i\) and \(j\) is not a static number; it is computed by a classical neural network (MLP) that takes a 4-dimensional vector as input. That vector contains:

- **Usage**: How often the edge has been used recently.
- **Meaning**: The semantic role of the connection.
- **Reflection**: The strength of past successes or failures associated with this edge (memory).
- **Temporality**: When the edge last fired.

The MLP outputs a raw scalar, which is then passed through the Squash+Sin activation to produce a coupling in \([-1,1]\). This makes edges **dynamic** and **self-aware**—they change based on context and memory. The equation is a form of neuromodulation, where classical information shapes quantum interactions.

## 3. Normalized Hamiltonian

$$
H_{\text{base}} = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + \alpha X_a X_b + \beta X_b X_a
$$

$$
H_{\text{total}} = \frac{H_{\text{base}}}{\|H_{\text{base}}\|_F}
$$

**Explanation**:  
The Hamiltonian \(H\) is the energy operator that governs the system's evolution. It consists of three parts:

- **Local fields** \(\sum_i h_i Z_i\): Each qubit \(i\) has a bias \(h_i\) that tilts its energy landscape, favoring one state over another. \(Z_i\) is the Pauli-Z operator acting on qubit \(i\).
- **Pairwise couplings** \(\sum_{i<j} J_{ij} Z_i Z_j\): Each pair of qubits \(i,j\) has an interaction strength \(J_{ij}\). This is the "brain cell" term, where edges carry the computational power.
- **Multiverse interference** \(\alpha X_a X_b + \beta X_b X_a\): This term entangles two specific qubits (universes) \(a\) and \(b\) using Pauli-X operators. The parameters \(\alpha\) and \(\beta\) control the strength of interference in each direction.

**Critical note**: In the original implementation, \(X_a X_b = X_b X_a\), causing degeneracy. A corrected version must use non-commuting terms, for example \(\alpha X_a X_b + \beta Y_a Y_b\), to make \(\alpha\) and \(\beta\) independently learnable.

Finally, the entire Hamiltonian is normalized by its Frobenius norm to ensure eigenvalues are bounded. This is crucial for stable evolution and to prevent time aliasing.

## 4. Fixed Evolution (Inference)

$$
|\Psi_{\text{final}}\rangle = e^{-i H_{\text{total}} \frac{\pi}{2}} |\Psi_0\rangle
$$

**Explanation**:  
Inference is performed by evolving the initial state \(|\Psi_0\rangle\) under the Hamiltonian for a fixed time \(T = \pi/2\). The initial state is typically a uniform superposition over all basis states. The unitary operator \(e^{-iH\pi/2}\) rotates the state in Hilbert space, and the resulting final state encodes the answer. Measurement probabilities are obtained by \(|\langle k|\Psi_{\text{final}}\rangle|^2\). This is **not** cooling or annealing; it is unitary evolution. The "cooling" metaphor in [[08_Intuitive_Physics]] must be understood as an analogy, not literal thermal relaxation.

## 5. Quantum Natural Gradient

$$
\theta_{t+1} = \theta_t - \eta \cdot \left( \text{Re}[\langle \partial_i \Psi | \partial_j \Psi \rangle - \langle \partial_i \Psi | \Psi \rangle \langle \Psi | \partial_j \Psi \rangle] \right)^{-1} \nabla L
$$

**Explanation**:  
This is the update rule for the quantum parameters \(\theta = [\alpha, \beta]\). The matrix \(G_{ij} = \text{Re}[\langle \partial_i \Psi | \partial_j \Psi \rangle - \langle \partial_i \Psi | \Psi \rangle \langle \Psi | \partial_j \Psi \rangle]\) is the Quantum Fisher Information Matrix (QFIM). It captures the local geometry of the quantum state space. By preconditioning the gradient with the inverse of the QFIM, the optimizer takes steps that are isotropic in the natural geometry, avoiding barren plateaus. In practice, the QFIM is computed using finite differences or parameter-shift rules.

## 6. Sparse Memory Injection

$$
J_{ij}^{\text{new}} = \text{Act}\left( J_{ij}^{\text{old}} + e^{-\text{Hamming}} \cdot J_{ij}^{\text{mem}} \right)
$$

**Explanation**:  
When a memory is retrieved, its stored coupling \(J_{ij}^{\text{mem}}\) is injected into the current coupling \(J_{ij}^{\text{old}}\). The strength of injection is weighted by \(e^{-\text{Hamming}}\), where Hamming distance measures how different the current activation pattern is from the stored pattern. If the patterns are similar, the exponential is close to 1, and the memory strongly influences the edge. If dissimilar, the influence fades. The result is passed through an activation to keep it within bounds. This equation embodies the resonance-based memory described in [[03_Phase3_Memory_Revolution]].

## Summary

These six equations form the mathematical backbone of Raslila. They combine classical MLPs, quantum superposition, entanglement, normalized evolution, natural gradient optimization, and resonance-based memory. The next step is to implement them correctly in code and validate the training process.

## Status

<span style="color:#7a9e7a">Equations formalized.</span>  
<span style="color:#b56576">\(\alpha/\beta\) are not independent in the current code – see [[10_Technical_Review]].</span>