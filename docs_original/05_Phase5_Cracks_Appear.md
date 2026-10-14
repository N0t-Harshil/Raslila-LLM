---
date: 2026-08-14
time: 10:00
tags:
  - phase
  - problems
  - optimisation
  - raslila
  - status/completed
---

# Phase 5: The Cracks Appear

## Introduction

After formalizing the mathematics and deciding to use quantum simulation, we attempted to run the first numerical experiments. We expected the system to learn, but instead we hit three major walls that threatened the entire project. These problems were not immediately obvious; they emerged only when we tried to train the system. They are classic challenges in variational quantum algorithms.

## Problem 1: Time Aliasing

The first problem involved the evolution time parameter \(T\) in the unitary \(e^{-iHT}\). We treated \(T\) as a trainable parameter and used gradient descent to optimize it. However, the eigenvalues of the Hamiltonian were not bounded, and the evolution became unstable. The quantum state started spinning wildly, and the loss function oscillated chaotically. This is known as **time aliasing**: because \(e^{-iHT}\) is periodic, different values of \(T\) can produce the same evolution, causing the optimizer to get confused.

**Cause**: The Hamiltonian \(H\) was not normalized, so its eigenvalues could be arbitrarily large. Multiplying by a large \(T\) caused rapid phase oscillations that overwhelmed the gradient signal. The loss function became highly oscillatory, with no stable descent direction.

## Problem 2: Barren Plateaus

The second problem was that standard optimizers like Adam got lost in the loss landscape. The gradient was essentially zero in most regions, a phenomenon known as **barren plateaus**. In high-dimensional quantum systems, the loss function tends to be exponentially flat, making it impossible for gradient-based methods to find a good direction. This is a well-known challenge in variational quantum algorithms and is caused by the fact that random quantum states have exponentially small overlap with any specific target state.

**Cause**: The loss function is an average over the exponentially large Hilbert space. For a random parameter configuration, the probability of measuring the target state is exponentially small, and its gradient is even smaller. The optimizer cannot distinguish between directions and gets stuck.

## Problem 3: Spectral Crowding

The third problem involved the activation functions used to map classical MLP outputs to coupling strengths. We initially used ReLU and Sigmoid. ReLU outputs unbounded positive values, which pushed the Hamiltonian couplings beyond the physical range of qubits. Sigmoid outputs values in (0,1), but its gradient vanishes at the extremes, and it caused many couplings to saturate. This led to a loss of diversity in the Hamiltonian's eigenvalues, a phenomenon we call **spectral crowding**. The eigenvalues became clustered, reducing the system's expressiveness and making it harder to distinguish different computational states.

**Cause**: The activations were not designed to respect the physical bounds of quantum couplings and to maintain non-zero gradients across their domain.

## The Impact

These three problems together meant that the system could not be trained effectively. Time aliasing made the evolution unstable, barren plateaus made the loss landscape flat, and spectral crowding limited the Hamiltonian's ability to represent complex functions. We needed to fix all three before the project could proceed.

## Status

<span style="color:#b56576">Problems identified and clearly diagnosed.</span>  
<span style="color:#d2a679">Solutions proposed in the next phase.</span>