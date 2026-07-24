"""Finite-scale safe-periodic refinement from the main construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._utils import fingerprint, readonly, validate_refinement_inputs
from .statistics import median_of_means


def residue(x: np.ndarray | float, length: np.ndarray | float, phase: np.ndarray | int) -> np.ndarray:
    """Evaluate the shifted residue ``rho_{L,b}(x)`` in ``[0,L)``."""
    x_array = np.asarray(x, dtype=float)
    length_array = np.asarray(length, dtype=float)
    phase_array = np.asarray(phase, dtype=float)
    return np.mod(x_array - 0.5 * phase_array * length_array, length_array)


def distance_to_grid(center: float, length: float, phase: int) -> float:
    position = float(residue(center, length, phase))
    return min(position, length - position)


def safe_phase(center: float, length: float) -> int:
    """Choose the deterministic safe phase, breaking ties in favor of zero."""
    distance_zero = distance_to_grid(center, length, 0)
    distance_one = distance_to_grid(center, length, 1)
    return int(distance_one > distance_zero)


def terminal_scale_count(k: float, tau: float, epsilon: float) -> int:
    """Return the smallest terminal index in Equation (terminal-J)."""
    tail_constant = 5.0 * 4.0 ** (k - 1.0)
    length = 8.0 * tau
    for index in range(1024):
        if tail_constant * tau**k / length ** (k - 1.0) <= epsilon / 4.0:
            return max(1, index)
        length *= 2.0
    raise OverflowError("terminal scale exceeded 1024 dyadic levels")


@dataclass(frozen=True)
class DyadicPlan:
    """A frozen public query plan for the dyadic refinement backend."""

    k: float
    sigma: float
    epsilon: float
    tau: float
    lengths: np.ndarray
    probabilities: np.ndarray
    base_phase: np.ndarray
    base_threshold: np.ndarray
    correction_scale: np.ndarray
    correction_phase: np.ndarray
    correction_next_phase: np.ndarray
    correction_threshold: np.ndarray

    @property
    def terminal_index(self) -> int:
        return int(self.probabilities.size)

    def fingerprint(self) -> str:
        return fingerprint(
            {"backend": "dyadic", "k": self.k, "sigma": self.sigma,
             "epsilon": self.epsilon, "tau": self.tau},
            self.lengths,
            self.probabilities,
            self.base_phase,
            self.base_threshold,
            self.correction_scale,
            self.correction_phase,
            self.correction_next_phase,
            self.correction_threshold,
        )

    def encode_base(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=float).reshape(-1)
        if samples.size != self.base_phase.size:
            raise ValueError("base sample count does not match the query plan")
        coordinates = residue(samples, self.lengths[0], self.base_phase)
        return coordinates >= self.base_threshold

    def encode_correction(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=float).reshape(-1)
        if samples.size != self.correction_scale.size:
            raise ValueError("correction sample count does not match the query plan")
        length = self.lengths[self.correction_scale]
        fine = residue(samples, 2.0 * length, self.correction_next_phase)
        coarse = residue(samples, length, self.correction_phase)
        return (fine - coarse) >= self.correction_threshold

    def decode_base_statistics(self, bits: np.ndarray, center: float) -> np.ndarray:
        bits = np.asarray(bits, dtype=float).reshape(-1)
        if bits.size != self.base_phase.size:
            raise ValueError("base bit count does not match the query plan")
        chosen = safe_phase(center, float(self.lengths[0]))
        selected = self.base_phase == chosen
        reference = self.base_threshold <= residue(
            center, self.lengths[0], self.base_phase
        )
        return 2.0 * selected * self.lengths[0] * (bits - reference)

    def decode_correction_statistics(self, bits: np.ndarray, center: float) -> np.ndarray:
        bits = np.asarray(bits, dtype=float).reshape(-1)
        if bits.size != self.correction_scale.size:
            raise ValueError("correction bit count does not match the query plan")
        selected_phases = np.array(
            [safe_phase(center, float(length)) for length in self.lengths],
            dtype=np.uint8,
        )
        scale = self.correction_scale
        length = self.lengths[scale]
        selected = (
            (self.correction_phase == selected_phases[scale])
            & (self.correction_next_phase == selected_phases[scale + 1])
        )
        fine_center = residue(center, 2.0 * length, self.correction_next_phase)
        coarse_center = residue(center, length, self.correction_phase)
        reference = self.correction_threshold <= (fine_center - coarse_center)
        importance = 12.0 * length / self.probabilities[scale]
        return importance * selected * (bits - reference)

    def ideal_conditional_block_moments(
        self, samples: np.ndarray, center: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Integrate each block's public coins exactly, conditional on data.

        Returns base mean, base second moment, correction mean, and correction
        second moment, in that order.
        """
        samples = np.asarray(samples, dtype=float).reshape(-1)
        selected_phases = np.array(
            [safe_phase(center, float(length)) for length in self.lengths],
            dtype=np.uint8,
        )
        differences = np.stack([
            residue(samples, length, phase) - residue(center, length, phase)
            for length, phase in zip(self.lengths, selected_phases)
        ])
        base_second = 2.0 * self.lengths[0] * np.abs(differences[0])
        increments = np.diff(differences, axis=0)
        correction_second = np.sum(
            12.0
            * self.lengths[:-1, None]
            * np.abs(increments)
            / self.probabilities[:, None],
            axis=0,
        )
        return (
            differences[0],
            base_second,
            differences[-1] - differences[0],
            correction_second,
        )

    def ideal_conditional_moments(
        self, samples: np.ndarray, center: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the combined conditional mean and second-moment sum.

        The protocol uses independent base and correction samples.  Pairing
        their conditional expressions on one array here is only a
        variance-reduced device for integrating their common data law.
        """
        base_mean, base_second, correction_mean, correction_second = (
            self.ideal_conditional_block_moments(samples, center)
        )
        return base_mean + correction_mean, base_second + correction_second

    def decode(
        self,
        base_bits: np.ndarray,
        correction_bits: np.ndarray,
        center: float,
        blocks: int = 1,
    ) -> float:
        base = self.decode_base_statistics(base_bits, center)
        correction = self.decode_correction_statistics(correction_bits, center)
        return center + median_of_means(base, blocks) + median_of_means(correction, blocks)


def build_dyadic_plan(
    *,
    k: float,
    sigma: float,
    epsilon: float,
    tau: float,
    base_samples: int,
    correction_samples: int,
    seed: int,
) -> DyadicPlan:
    """Compile all dyadic encoder queries before any bit is observed."""
    validate_refinement_inputs(k, sigma, epsilon, tau)
    if base_samples < 1 or correction_samples < 1:
        raise ValueError("both refinement blocks need at least one sample")
    terminal_index = terminal_scale_count(k, tau, epsilon)
    lengths = 8.0 * tau * 2.0 ** np.arange(terminal_index + 1, dtype=float)
    weights = 2.0 ** (np.arange(terminal_index, dtype=float) * (2.0 - k) / 2.0)
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(seed)
    base_phase = rng.integers(0, 2, size=base_samples, dtype=np.uint8)
    base_threshold = rng.random(base_samples) * lengths[0]
    correction_scale = rng.choice(
        terminal_index, size=correction_samples, p=probabilities
    ).astype(np.int32)
    correction_phase = rng.integers(0, 2, size=correction_samples, dtype=np.uint8)
    correction_next_phase = rng.integers(
        0, 2, size=correction_samples, dtype=np.uint8
    )
    selected_lengths = lengths[correction_scale]
    correction_threshold = selected_lengths * (
        -1.0 + 3.0 * rng.random(correction_samples)
    )
    return DyadicPlan(
        k=float(k),
        sigma=float(sigma),
        epsilon=float(epsilon),
        tau=float(tau),
        lengths=readonly(lengths),
        probabilities=readonly(probabilities),
        base_phase=readonly(base_phase),
        base_threshold=readonly(base_threshold),
        correction_scale=readonly(correction_scale),
        correction_phase=readonly(correction_phase),
        correction_next_phase=readonly(correction_next_phase),
        correction_threshold=readonly(correction_threshold),
    )
