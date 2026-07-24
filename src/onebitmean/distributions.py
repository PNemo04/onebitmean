"""Synthetic finite-moment laws used by the reproducibility experiments."""

from __future__ import annotations

import math

import numpy as np


def normal_with_kth_moment(
    rng: np.random.Generator,
    count: int,
    *,
    k: float,
    sigma: float,
    mean: float = 0.0,
) -> np.ndarray:
    absolute_moment = 2.0 ** (k / 2.0) * math.gamma((k + 1.0) / 2.0) / math.sqrt(math.pi)
    scale = sigma / absolute_moment ** (1.0 / k)
    return mean + scale * rng.standard_normal(count)


def symmetric_pareto_with_kth_moment(
    rng: np.random.Generator,
    count: int,
    *,
    k: float,
    alpha: float,
    sigma: float,
    mean: float = 0.0,
) -> np.ndarray:
    """Sample a symmetric Pareto law with exact central kth moment sigma^k."""
    if alpha <= max(k, 1.0):
        raise ValueError("alpha must exceed max(k, 1)")
    minimum = sigma * ((alpha - k) / alpha) ** (1.0 / k)
    magnitudes = minimum * (1.0 - rng.random(count)) ** (-1.0 / alpha)
    signs = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    return mean + signs * magnitudes


def sparse_symmetric_outliers(
    rng: np.random.Generator,
    count: int,
    *,
    k: float,
    sigma: float,
    amplitude: float,
    mean: float = 0.0,
) -> np.ndarray:
    """A zero-or-symmetric-outlier law saturating the kth-moment constraint."""
    probability = (sigma / amplitude) ** k
    if probability > 1.0:
        raise ValueError("amplitude must be at least sigma")
    active = rng.random(count) < probability
    signs = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    return mean + active * signs * amplitude

