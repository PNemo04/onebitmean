"""Coding-theoretic non-adaptive localization used by both backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._utils import fingerprint, readonly


def _bin_indices(samples: np.ndarray, lam: float, bin_width: float, bins: int) -> np.ndarray:
    indices = np.floor((samples + lam) / bin_width).astype(np.int64)
    return np.clip(indices, 0, bins - 1)


def _is_balanced(codebook: np.ndarray, tolerance: float = 0.01) -> bool:
    bins, length = codebook.shape
    lower = (0.5 - tolerance) * length
    upper = (0.5 + tolerance) * length
    for left in range(bins):
        if left + 1 == bins:
            break
        distances = np.count_nonzero(codebook[left + 1 :] != codebook[left], axis=1)
        if np.any(distances < lower) or np.any(distances > upper):
            return False
    return True


@dataclass(frozen=True)
class LocalizationPlan:
    lam: float
    sigma: float
    delta: float
    profile: str
    bin_edges: np.ndarray
    codebook: np.ndarray

    @property
    def query_count(self) -> int:
        return int(self.codebook.shape[1])

    def fingerprint(self) -> str:
        return fingerprint(
            {"backend": "localization", "lam": self.lam, "sigma": self.sigma,
             "delta": self.delta, "profile": self.profile},
            self.bin_edges,
            self.codebook,
        )

    def encode(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=float).reshape(-1)
        if samples.size != self.query_count:
            raise ValueError("sample count does not match localization query count")
        if self.codebook.shape[0] == 1:
            return np.zeros(0, dtype=np.uint8)
        width = float(self.bin_edges[1] - self.bin_edges[0])
        indices = _bin_indices(samples, self.lam, width, self.codebook.shape[0])
        return self.codebook[indices, np.arange(self.query_count)]

    def decode_interval(self, bits: np.ndarray) -> tuple[float, float]:
        if self.codebook.shape[0] == 1:
            return -self.lam, self.lam
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if bits.size != self.query_count:
            raise ValueError("bit count does not match localization query count")
        scores = np.count_nonzero(self.codebook != bits[None, :], axis=1)
        selected = int(np.argmin(scores))
        left_bin = max(0, selected - 2)
        right_bin = min(self.codebook.shape[0] - 1, selected + 2)
        return float(self.bin_edges[left_bin]), float(self.bin_edges[right_bin + 1])


def build_localization_plan(
    *,
    lam: float,
    sigma: float,
    delta: float,
    seed: int,
    profile: Literal["certified", "research"],
    max_attempts: int = 128,
) -> LocalizationPlan:
    """Build the fixed codebook plan from Lau--Scarlett Theorem 16.

    The certified profile retains the theorem's code-length coefficient and
    checks its 0.49--0.51 distance condition.  The research profile uses a
    shorter random code only for executable sanity tests.
    """
    if not np.isfinite(lam) or not np.isfinite(sigma) or lam < sigma or sigma <= 0:
        raise ValueError("require finite lam >= sigma > 0")
    if not 0 < delta < 0.5:
        raise ValueError("delta must lie in (0, 1/2)")
    if profile not in {"certified", "research"}:
        raise ValueError("profile must be 'certified' or 'research'")
    h = 20.0 * sigma
    if 2.0 * lam <= h:
        return LocalizationPlan(
            lam=float(lam), sigma=float(sigma), delta=float(delta), profile=profile,
            bin_edges=readonly(np.array([-lam, lam], dtype=float)),
            codebook=readonly(np.zeros((1, 0), dtype=np.uint8)),
        )
    bins = int(math.ceil(2.0 * lam / h))
    edges = np.linspace(-lam, lam, bins + 1)
    coefficient = 10000.0 if profile == "certified" else 64.0
    length = int(math.ceil(coefficient * (math.log(bins) + math.log(1.0 / delta))))
    rng = np.random.default_rng(seed)
    codebook = None
    for _ in range(max_attempts):
        candidate = rng.integers(0, 2, size=(bins, length), dtype=np.uint8)
        if profile == "research" or _is_balanced(candidate):
            codebook = candidate
            break
    if codebook is None:
        raise RuntimeError("failed to draw a balanced deterministic codebook")
    return LocalizationPlan(
        lam=float(lam),
        sigma=float(sigma),
        delta=float(delta),
        profile=profile,
        bin_edges=readonly(edges),
        codebook=readonly(codebook),
    )

