"""Robust aggregation primitives used by both refinement backends."""

from __future__ import annotations

import math

import numpy as np


def mom_block_count(failure_probability: float) -> int:
    """Return the block count used in the manuscript's median-of-means lemma."""
    if not 0 < failure_probability < 0.5:
        raise ValueError("failure_probability must lie in (0, 1/2)")
    return max(1, math.ceil(8.0 * math.log(1.0 / failure_probability)))


def median_of_means(values: np.ndarray, blocks: int) -> float:
    """Compute the upper median of equal-sized contiguous block means.

    At most ``blocks - 1`` trailing observations are discarded, exactly as in
    the proof.  The input order is immaterial for i.i.d. observations.
    """
    observations = np.asarray(values, dtype=float).reshape(-1)
    if observations.size == 0:
        raise ValueError("values must be non-empty")
    if blocks < 1 or blocks > observations.size:
        raise ValueError("blocks must be between one and the sample count")
    block_size = observations.size // blocks
    used = observations[: block_size * blocks].reshape(blocks, block_size)
    means = used.mean(axis=1)
    return float(np.partition(means, blocks // 2)[blocks // 2])

