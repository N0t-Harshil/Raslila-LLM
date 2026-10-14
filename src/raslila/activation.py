"""Activation functions.

Fixes Additional Issue #1: original Squash+Sin has zero gradient at x = +-1.

Original:
    f(x) = sin(pi * x / (1+|x|))
    f'(x) = pi * s'(x) * cos(pi*s(x)), s = x/(1+|x|)
    At x=1: s=1/2, cos(pi/2)=0 -> f'=0. Same at x=-1.

Fixed (default `squash_sin`):
    f(x) = sin((pi/2) * x / (1+|x|))
    f'(x) = (pi/2) * s'(x) * cos((pi/2)*s(x))
    cos = 0 only when s = +-1, i.e. x -> +-inf. Never zero for finite x.
    Still maps (-inf,inf) -> (-1,1). Preserves the spirit, kills the bug.

Also provides scaled_tanh as an alternative.
"""
import numpy as np

_EPS = 1e-12


def _squash(x):
    return x / (1.0 + np.abs(x) + _EPS)


def squash_sin_original(x):
    """Original buggy activation (kept for comparison/testing)."""
    s = _squash(x)
    return np.sin(s * np.pi)


def dsquash_sin_original_dx(x):
    s = _squash(x)
    ds = 1.0 / ((1.0 + np.abs(x) + _EPS) ** 2)
    return np.pi * ds * np.cos(s * np.pi)


def squash_sin(x):
    """FIXED Squash+Sin. Default activation. Derivative never zero for finite x."""
    s = _squash(x)
    return np.sin(s * np.pi / 2.0)


def dsquash_sin_dx(x):
    s = _squash(x)
    ds = 1.0 / ((1.0 + np.abs(x) + _EPS) ** 2)
    return (np.pi / 2.0) * ds * np.cos(s * np.pi / 2.0)


def scaled_tanh(x):
    return np.tanh(x)


def dscaled_tanh_dx(x):
    t = np.tanh(x)
    return 1.0 - t * t
