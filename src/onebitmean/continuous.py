"""Continuous random-grid refinement from the alternative construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._utils import fingerprint, readonly, validate_refinement_inputs
from .statistics import median_of_means


_A = 0.25
_C_A = math.log(15.0 / 7.0)


def cell_sign(
    seed: np.ndarray, offset: np.ndarray, index: np.ndarray
) -> np.ndarray:
    """Evaluate an exact pairwise-independent coloring on int64 cells.

    For a public uniform ``seed`` in GF(2)^64 and an independent uniform
    ``offset`` bit, the color is ``(-1)^(<seed,index> xor offset)``.  Distinct
    64-bit cell words give linearly independent affine evaluation vectors, so
    every pair of colors is independent and Rademacher-distributed.
    """
    words = np.asarray(index, dtype=np.int64).view(np.uint64)
    inner_product = np.bitwise_count(
        words & np.asarray(seed, dtype=np.uint64)
    ) & np.uint8(1)
    bits = inner_product ^ np.asarray(offset, dtype=np.uint8)
    return np.where(bits == 1, 1.0, -1.0)


def continuous_cutoffs(k: float, tau: float, epsilon: float) -> tuple[float, float]:
    lower = epsilon / 14.0
    upper = 4.0 * (8.0 * tau**k / epsilon) ** (1.0 / (k - 1.0))
    return lower, upper


def _density_normalizer(k: float, tau: float, lower: float, upper: float) -> float:
    if k < 2.0:
        exponent = 2.0 - k
        return (upper**exponent - lower**exponent) / exponent
    if k == 2.0:
        return math.log(upper / lower)
    light_mass = (tau - lower) / tau
    exponent = 2.0 - k
    tail_mass = tau ** (k - 2.0) * (
        upper**exponent - tau**exponent
    ) / exponent
    return light_mass + tail_mass


def _sample_scales(
    rng: np.random.Generator,
    count: int,
    k: float,
    tau: float,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    uniforms = rng.random(count)
    if k < 2.0:
        exponent = 2.0 - k
        normalizer = (upper**exponent - lower**exponent) / exponent
        scales = (
            lower**exponent
            + uniforms * (upper**exponent - lower**exponent)
        ) ** (1.0 / exponent)
        density = scales ** (1.0 - k) / normalizer
        return scales, density
    if k == 2.0:
        normalizer = math.log(upper / lower)
        scales = lower * np.exp(uniforms * normalizer)
        density = 1.0 / (normalizer * scales)
        return scales, density

    light_mass = (tau - lower) / tau
    exponent = 2.0 - k
    tail_mass = tau ** (k - 2.0) * (
        upper**exponent - tau**exponent
    ) / exponent
    normalizer = light_mass + tail_mass
    choose_light = uniforms < light_mass / normalizer
    scales = np.empty(count, dtype=float)
    light_uniform = rng.random(int(choose_light.sum()))
    scales[choose_light] = lower + (tau - lower) * light_uniform
    tail_uniform = rng.random(int((~choose_light).sum()))
    scales[~choose_light] = (
        tau**exponent
        + tail_uniform * (upper**exponent - tau**exponent)
    ) ** (1.0 / exponent)
    q = np.where(
        scales <= tau,
        1.0 / tau,
        tau ** (k - 2.0) * scales ** (1.0 - k),
    )
    return scales, q / normalizer


@dataclass(frozen=True)
class ContinuousPlan:
    """A frozen public query plan for the continuous random-grid backend."""

    k: float
    sigma: float
    epsilon: float
    tau: float
    lower_scale: float
    upper_scale: float
    scale: np.ndarray
    density: np.ndarray
    shift: np.ndarray
    coloring_seed: np.ndarray
    coloring_offset: np.ndarray

    def fingerprint(self) -> str:
        return fingerprint(
            {"backend": "continuous", "k": self.k, "sigma": self.sigma,
             "epsilon": self.epsilon, "tau": self.tau,
             "lower_scale": self.lower_scale, "upper_scale": self.upper_scale},
            self.scale,
            self.density,
            self.shift,
            self.coloring_seed,
            self.coloring_offset,
        )

    def encode(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=float).reshape(-1)
        if samples.size != self.scale.size:
            raise ValueError("sample count does not match the query plan")
        cells = np.floor((samples + self.shift) / self.scale).astype(np.int64)
        return cell_sign(self.coloring_seed, self.coloring_offset, cells) > 0

    def decode_statistics(self, bits: np.ndarray, center: float) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if bits.size != self.scale.size:
            raise ValueError("bit count does not match the query plan")
        position = (center + self.shift) / self.scale
        center_cell = np.floor(position).astype(np.int64)
        fractional = position - center_cell
        gate = (_A <= fractional) & (fractional <= 1.0 - _A)
        sample_sign = 2.0 * bits.astype(float) - 1.0
        center_sign = cell_sign(
            self.coloring_seed, self.coloring_offset, center_cell
        )
        right_sign = cell_sign(
            self.coloring_seed, self.coloring_offset, center_cell + 1
        )
        left_sign = cell_sign(
            self.coloring_seed, self.coloring_offset, center_cell - 1
        )
        return (
            gate
            * (sample_sign - center_sign)
            * (right_sign - left_sign)
            / (_C_A * self.density)
        )

    @staticmethod
    def _clipped_integral(
        lower: np.ndarray,
        upper: np.ndarray,
        domain_lower: float,
        domain_upper: float,
        antiderivative,
    ) -> np.ndarray:
        left = np.maximum(lower, domain_lower)
        right = np.minimum(upper, domain_upper)
        valid = right > left
        result = np.zeros_like(left, dtype=float)
        if np.any(valid):
            result[valid] = antiderivative(right[valid], valid) - antiderivative(left[valid], valid)
        return result

    def _ideal_mean(self, displacement: np.ndarray) -> np.ndarray:
        magnitude = np.abs(displacement)
        sign = np.sign(displacement)

        first_lower = magnitude / (2.0 - _A)
        first_upper = magnitude / (1.0 + _A)
        first = self._clipped_integral(
            first_lower,
            first_upper,
            self.lower_scale,
            self.upper_scale,
            lambda r, valid: (2.0 - _A) * r - magnitude[valid] * np.log(r),
        )

        second_lower = magnitude / (1.0 + _A)
        second_upper = magnitude / (1.0 - _A)
        second = self._clipped_integral(
            second_lower,
            second_upper,
            self.lower_scale,
            self.upper_scale,
            lambda r, valid: (1.0 - 2.0 * _A) * r,
        )

        third_lower = magnitude / (1.0 - _A)
        third_upper = magnitude / _A
        third = self._clipped_integral(
            third_lower,
            third_upper,
            self.lower_scale,
            self.upper_scale,
            lambda r, valid: magnitude[valid] * np.log(r) - _A * r,
        )
        return sign * (first + second + third) / _C_A

    def _integral_inverse_density(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        valid = upper > lower
        result = np.zeros_like(lower, dtype=float)
        normalizer = _density_normalizer(
            self.k, self.tau, self.lower_scale, self.upper_scale
        )
        if self.k <= 2.0:
            exponent = self.k
            result[valid] = normalizer * (
                upper[valid] ** exponent - lower[valid] ** exponent
            ) / exponent
            return result

        light_right = np.minimum(upper, self.tau)
        light_valid = light_right > lower
        result[light_valid] += normalizer * self.tau * (
            light_right[light_valid] - lower[light_valid]
        )
        tail_left = np.maximum(lower, self.tau)
        tail_valid = upper > tail_left
        result[tail_valid] += normalizer * self.tau ** (2.0 - self.k) * (
            upper[tail_valid] ** self.k - tail_left[tail_valid] ** self.k
        ) / self.k
        return result

    def _integral_linear_crossing(
        self, lower: np.ndarray, upper: np.ndarray, magnitude: np.ndarray
    ) -> np.ndarray:
        valid = upper > lower
        result = np.zeros_like(lower, dtype=float)
        normalizer = _density_normalizer(
            self.k, self.tau, self.lower_scale, self.upper_scale
        )
        if self.k <= 2.0:
            result[valid] = normalizer * (
                magnitude[valid]
                * (upper[valid] ** (self.k - 1.0) - lower[valid] ** (self.k - 1.0))
                / (self.k - 1.0)
                - _A
                * (upper[valid] ** self.k - lower[valid] ** self.k)
                / self.k
            )
            return result

        light_right = np.minimum(upper, self.tau)
        light_valid = light_right > lower
        result[light_valid] += normalizer * self.tau * (
            magnitude[light_valid] * np.log(light_right[light_valid] / lower[light_valid])
            - _A * (light_right[light_valid] - lower[light_valid])
        )
        tail_left = np.maximum(lower, self.tau)
        tail_valid = upper > tail_left
        result[tail_valid] += normalizer * self.tau ** (2.0 - self.k) * (
            magnitude[tail_valid]
            * (upper[tail_valid] ** (self.k - 1.0) - tail_left[tail_valid] ** (self.k - 1.0))
            / (self.k - 1.0)
            - _A
            * (upper[tail_valid] ** self.k - tail_left[tail_valid] ** self.k)
            / self.k
        )
        return result

    def ideal_conditional_moments(
        self, samples: np.ndarray, center: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the ideal public grid, coloring, and scale law exactly."""
        displacement = np.asarray(samples, dtype=float).reshape(-1) - center
        magnitude = np.abs(displacement)
        conditional_mean = self._ideal_mean(displacement)

        low_left = np.full_like(magnitude, self.lower_scale)
        low_right = np.minimum(self.upper_scale, magnitude / (1.0 - _A))
        low = 0.5 * self._integral_inverse_density(low_left, low_right)

        mid_left = np.maximum(self.lower_scale, magnitude / (1.0 - _A))
        mid_right = np.minimum(self.upper_scale, magnitude / _A)
        middle = self._integral_linear_crossing(
            mid_left, mid_right, magnitude
        )
        conditional_second = 4.0 * (low + middle) / (_C_A**2)
        return conditional_mean, conditional_second

    def decode(self, bits: np.ndarray, center: float, blocks: int = 1) -> float:
        statistics = self.decode_statistics(bits, center)
        return center + median_of_means(statistics, blocks)


def build_continuous_plan(
    *,
    k: float,
    sigma: float,
    epsilon: float,
    tau: float,
    samples: int,
    seed: int,
) -> ContinuousPlan:
    """Compile all continuous-grid encoder queries before any bit is observed."""
    validate_refinement_inputs(k, sigma, epsilon, tau)
    if samples < 1:
        raise ValueError("samples must be positive")
    lower, upper = continuous_cutoffs(k, tau, epsilon)
    rng = np.random.default_rng(seed)
    scales, density = _sample_scales(rng, samples, k, tau, lower, upper)
    shifts = rng.random(samples) * scales
    coloring_seed = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=samples,
        dtype=np.uint64,
        endpoint=True,
    )
    coloring_offset = rng.integers(0, 2, size=samples, dtype=np.uint8)
    return ContinuousPlan(
        k=float(k),
        sigma=float(sigma),
        epsilon=float(epsilon),
        tau=float(tau),
        lower_scale=float(lower),
        upper_scale=float(upper),
        scale=readonly(scales),
        density=readonly(density),
        shift=readonly(shifts),
        coloring_seed=readonly(coloring_seed),
        coloring_offset=readonly(coloring_offset),
    )
