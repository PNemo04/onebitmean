#!/usr/bin/env python3
"""Run reproducible mechanism, scaling, localization, and timing checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import fields
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy.stats import beta

from onebitmean.continuous import ContinuousPlan, build_continuous_plan
from onebitmean.distributions import (
    normal_with_kth_moment,
    symmetric_pareto_with_kth_moment,
)
from onebitmean.dyadic import DyadicPlan, build_dyadic_plan, residue, safe_phase
from onebitmean.localization import build_localization_plan


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def distribution_samples(
    rng: np.random.Generator,
    count: int,
    *,
    k: float,
    sigma: float,
    mean: float,
) -> np.ndarray:
    if k > 2.0:
        return normal_with_kth_moment(
            rng, count, k=k, sigma=sigma, mean=mean
        )
    alpha = 2.3 if k == 2.0 else 1.7
    return symmetric_pareto_with_kth_moment(
        rng, count, k=k, alpha=alpha, sigma=sigma, mean=mean
    )


def target_rate(k: float, tau: float, epsilon: float) -> float:
    ratio = tau / epsilon
    if k > 2.0:
        return ratio**2
    if k == 2.0:
        return ratio**2 * math.log(math.e * ratio)
    return ratio ** (k / (k - 1.0))


def plan_bytes(plan: DyadicPlan | ContinuousPlan) -> int:
    total = 0
    for field in fields(plan):
        value = getattr(plan, field.name)
        if isinstance(value, np.ndarray):
            total += value.nbytes
    return total


def confidence_record(
    statistics: np.ndarray,
    squares: np.ndarray,
    *,
    target: float,
    epsilon: float,
    normalization: float,
) -> dict[str, float]:
    count = statistics.size
    estimate = float(statistics.mean())
    mean_se = float(statistics.std(ddof=1) / math.sqrt(count))
    square_mean = float(squares.mean())
    square_se = float(squares.std(ddof=1) / math.sqrt(count))
    return {
        "estimate": estimate,
        "target": target,
        "signed_bias_over_epsilon": (estimate - target) / epsilon,
        "mean_ci95_halfwidth_over_epsilon": 1.96 * mean_se / epsilon,
        "second_moment": square_mean,
        "second_moment_se": square_se,
        "normalized_second_moment": square_mean / normalization,
        "normalized_second_moment_ci95_halfwidth": 1.96 * square_se / normalization,
    }


def run_scaling(profile: str) -> list[dict[str, float | str | int]]:
    count = 60_000 if profile == "quick" else 400_000
    epsilons = [0.32, 0.16] if profile == "quick" else [0.40, 0.25, 0.16, 0.10, 0.063]
    sigma = 1.0
    mean = 0.20
    center = 0.0
    rows: list[dict[str, float | str | int]] = []
    case_seed = 1000
    for k in (3.0, 2.0, 1.5):
        tau = (2.0 ** (k - 1.0) * (sigma**k + abs(mean - center) ** k)) ** (1.0 / k)
        for epsilon in epsilons:
            for backend in ("dyadic", "continuous"):
                rng = np.random.default_rng(case_seed)
                case_seed += 1
                if backend == "dyadic":
                    plan = build_dyadic_plan(
                        k=k,
                        sigma=sigma,
                        epsilon=epsilon,
                        tau=tau,
                        base_samples=count,
                        correction_samples=count,
                        seed=case_seed,
                    )
                    base_samples = distribution_samples(
                        rng, count, k=k, sigma=sigma, mean=mean
                    )
                    correction_samples = distribution_samples(
                        rng, count, k=k, sigma=sigma, mean=mean
                    )
                    start = time.perf_counter()
                    base_bits = plan.encode_base(base_samples)
                    correction_bits = plan.encode_correction(correction_samples)
                    encode_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    base_statistics = plan.decode_base_statistics(base_bits, center)
                    correction_statistics = plan.decode_correction_statistics(
                        correction_bits, center
                    )
                    decode_seconds = time.perf_counter() - start
                    raw_statistics = base_statistics + correction_statistics
                    statistics, squares = plan.ideal_conditional_moments(
                        base_samples, center
                    )
                    base_mean, _, _, _ = plan.ideal_conditional_block_moments(
                        base_samples, center
                    )
                    _, _, correction_mean, _ = plan.ideal_conditional_block_moments(
                        correction_samples, center
                    )
                    crosscheck_differences = (
                        base_statistics - base_mean
                        + correction_statistics - correction_mean
                    )
                    total_queries = 2 * count
                    terminal_index = plan.terminal_index
                else:
                    plan = build_continuous_plan(
                        k=k,
                        sigma=sigma,
                        epsilon=epsilon,
                        tau=tau,
                        samples=count,
                        seed=case_seed,
                    )
                    samples = distribution_samples(
                        rng, count, k=k, sigma=sigma, mean=mean
                    )
                    start = time.perf_counter()
                    bits = plan.encode(samples)
                    encode_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    raw_statistics = plan.decode_statistics(bits, center)
                    decode_seconds = time.perf_counter() - start
                    statistics, squares = plan.ideal_conditional_moments(
                        samples, center
                    )
                    crosscheck_differences = raw_statistics - statistics
                    total_queries = count
                    terminal_index = -1
                rate = target_rate(k, tau, epsilon)
                normalization = epsilon**2 * rate
                record = confidence_record(
                    statistics,
                    squares,
                    target=mean - center,
                    epsilon=epsilon,
                    normalization=normalization,
                )
                empirical_moment = float(np.mean(np.abs(
                    (base_samples if backend == "dyadic" else samples) - mean
                ) ** k))
                crosscheck_standard_error = float(
                    crosscheck_differences.std(ddof=1) / math.sqrt(count)
                )
                implementation_crosscheck_z = float(
                    crosscheck_differences.mean() / crosscheck_standard_error
                ) if crosscheck_standard_error > 0 else 0.0
                rows.append({
                    "backend": backend,
                    "k": k,
                    "epsilon": epsilon,
                    "sigma": sigma,
                    "tau": tau,
                    "draws_per_block": count,
                    "total_queries": total_queries,
                    "terminal_index": terminal_index,
                    "target_rate": rate,
                    "empirical_central_kth_moment": empirical_moment,
                    "encode_nanoseconds_per_query": 1e9 * encode_seconds / total_queries,
                    "decode_nanoseconds_per_query": 1e9 * decode_seconds / total_queries,
                    "public_plan_bytes_per_query": plan_bytes(plan) / total_queries,
                    "plan_fingerprint": plan.fingerprint(),
                    "implementation_crosscheck_z": implementation_crosscheck_z,
                    **record,
                })
    return rows


def _scale_probabilities(levels: int, assumed_k: float) -> np.ndarray:
    weights = 2.0 ** (np.arange(levels) * (2.0 - assumed_k) / 2.0)
    return weights / weights.sum()


def run_allocation(profile: str) -> list[dict[str, float | str]]:
    sigma = 1.0
    epsilon = 0.10
    mean = 0.20
    center = 0.0
    rows: list[dict[str, float | str]] = []
    for case, k in enumerate((3.0, 2.0, 1.5)):
        tau = (2.0 ** (k - 1.0) * (sigma**k + abs(mean - center) ** k)) ** (1.0 / k)
        plan = build_dyadic_plan(
            k=k,
            sigma=sigma,
            epsilon=epsilon,
            tau=tau,
            base_samples=1,
            correction_samples=1,
            seed=5100 + case,
        )
        lengths = plan.lengths[:-1]
        coefficients = tau**k * lengths ** (2.0 - k)
        candidates = {
            "matched": plan.probabilities,
            "uniform": np.full(plan.terminal_index, 1.0 / plan.terminal_index),
            "light-tail": _scale_probabilities(plan.terminal_index, 3.0),
            "heavy-tail": _scale_probabilities(plan.terminal_index, 1.5),
        }
        objective = {
            name: float(np.sum(coefficients / probabilities))
            for name, probabilities in candidates.items()
        }
        baseline = objective["matched"]
        for name in ("matched", "uniform", "light-tail", "heavy-tail"):
            rows.append({
                "k": k,
                "allocation": name,
                "objective": objective[name],
                "ratio_to_matched": objective[name] / baseline if baseline > 0 else 1.0,
                "terminal_index": float(plan.terminal_index),
                "objective_type": "deterministic kth-moment envelope",
            })
    return rows


def run_mechanisms() -> dict[str, float | str]:
    length = 8.0
    center = 0.0
    sample = -0.5
    safe = safe_phase(center, length)
    unsafe = 1 - safe
    safe_base = float(residue(sample, length, safe) - residue(center, length, safe))
    unsafe_base = float(residue(sample, length, unsafe) - residue(center, length, unsafe))
    safe_correction = float(
        (residue(sample, 2.0 * length, safe_phase(center, 2.0 * length))
         - residue(sample, length, safe))
        - (residue(center, 2.0 * length, safe_phase(center, 2.0 * length))
           - residue(center, length, safe))
    )
    unsafe_correction = float(
        (residue(sample, 2.0 * length, unsafe)
         - residue(sample, length, unsafe))
        - (residue(center, 2.0 * length, unsafe)
           - residue(center, length, unsafe))
    )
    rng = np.random.default_rng(6100)
    max_telescope_residual = 0.0
    for _ in range(1000):
        c = float(rng.normal() * 10.0)
        x = float(rng.normal() * 30.0)
        lengths = 2.3 * 2.0 ** np.arange(10)
        phases = [safe_phase(c, float(value)) for value in lengths]
        differences = np.array([
            residue(x, value, phase) - residue(c, value, phase)
            for value, phase in zip(lengths, phases)
        ])
        residual = abs(differences[0] + np.diff(differences).sum() - differences[-1])
        max_telescope_residual = max(max_telescope_residual, float(residual))
    return {
        "center": center,
        "sample": sample,
        "safe_phase": float(safe),
        "safe_base_displacement": safe_base,
        "unsafe_base_displacement": unsafe_base,
        "safe_correction_displacement": safe_correction,
        "unsafe_correction_displacement": unsafe_correction,
        "maximum_telescope_residual": max_telescope_residual,
        "interpretation": "Safe phases preserve local displacement; telescoping is pointwise exact.",
    }


def clopper_pearson_upper(failures: int, trials: int, confidence: float = 0.95) -> float:
    if failures == trials:
        return 1.0
    return float(beta.ppf(confidence, failures + 1, trials - failures))


def run_localization(profile: str) -> list[dict[str, float | int | str]]:
    trials = 250 if profile == "quick" else 2000
    lam = 200.0
    sigma = 1.0
    delta = 0.05
    plan = build_localization_plan(
        lam=lam,
        sigma=sigma,
        delta=delta,
        seed=7100,
        profile="research",
    )
    certified = build_localization_plan(
        lam=lam,
        sigma=sigma,
        delta=delta,
        seed=7101,
        profile="certified",
    )
    rows: list[dict[str, float | int | str]] = []
    for case, mean in enumerate((-150.0, 3.0, 151.0)):
        rng = np.random.default_rng(7200 + case)
        failures = 0
        for _ in range(trials):
            samples = symmetric_pareto_with_kth_moment(
                rng,
                plan.query_count,
                k=1.5,
                alpha=1.7,
                sigma=sigma,
                mean=mean,
            )
            bits = plan.encode(samples)
            left, right = plan.decode_interval(bits)
            failures += int(not (left <= mean <= right))
        rows.append({
            "mean": mean,
            "trials": trials,
            "failures": failures,
            "empirical_failure_probability": failures / trials,
            "clopper_pearson_95_upper": clopper_pearson_upper(failures, trials),
            "research_query_count": plan.query_count,
            "certified_query_count": certified.query_count,
            "profile": "research (empirical); certified count reported only",
            "plan_fingerprint": plan.fingerprint(),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    scaling = run_scaling(args.profile)
    allocation = run_allocation(args.profile)
    localization = run_localization(args.profile)
    mechanisms = run_mechanisms()
    write_csv(RESULTS / "scaling.csv", scaling)
    write_csv(RESULTS / "allocation.csv", allocation)
    write_csv(RESULTS / "localization.csv", localization)
    metadata = {
        "profile": args.profile,
        "wall_clock_seconds": time.perf_counter() - start,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "master_seed_schedule": "explicit deterministic integer seeds in run_validation.py",
        "mechanisms": mechanisms,
        "summary": {
            "maximum_absolute_bias_over_epsilon": max(
                abs(float(row["signed_bias_over_epsilon"])) for row in scaling
            ),
            "maximum_absolute_implementation_crosscheck_z": max(
                abs(float(row["implementation_crosscheck_z"])) for row in scaling
            ),
            "median_public_plan_bytes_per_query": {
                backend: float(np.median([
                    float(row["public_plan_bytes_per_query"])
                    for row in scaling if row["backend"] == backend
                ]))
                for backend in ("dyadic", "continuous")
            },
            "median_encode_plus_decode_nanoseconds_per_query": {
                backend: float(np.median([
                    float(row["encode_nanoseconds_per_query"])
                    + float(row["decode_nanoseconds_per_query"])
                    for row in scaling if row["backend"] == backend
                ]))
                for backend in ("dyadic", "continuous")
            },
        },
    }
    with (RESULTS / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps({
        "profile": args.profile,
        "scaling_rows": len(scaling),
        "allocation_rows": len(allocation),
        "localization_rows": len(localization),
        "wall_clock_seconds": metadata["wall_clock_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
