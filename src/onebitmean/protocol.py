"""Uniform compile/encode/decode facade for the two refinement backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from .continuous import ContinuousPlan, build_continuous_plan
from .dyadic import DyadicPlan, build_dyadic_plan


Backend = Literal["dyadic", "continuous"]


@dataclass(frozen=True)
class CompiledRefinement:
    """A backend-neutral public query plan.

    Arrays are grouped by named sample blocks so that both the two-block
    dyadic construction and the one-block continuous construction share the
    same external interface.
    """

    backend: Backend
    plan: DyadicPlan | ContinuousPlan

    def __post_init__(self) -> None:
        expected = "dyadic" if isinstance(self.plan, DyadicPlan) else "continuous"
        if self.backend != expected:
            raise ValueError(
                f"backend label {self.backend!r} does not match {expected} plan"
            )

    @property
    def query_counts(self) -> dict[str, int]:
        if isinstance(self.plan, DyadicPlan):
            return {
                "base": int(self.plan.base_phase.size),
                "correction": int(self.plan.correction_scale.size),
            }
        return {"refinement": int(self.plan.scale.size)}

    def fingerprint(self) -> str:
        return self.plan.fingerprint()

    def encode(self, samples: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        expected = set(self.query_counts)
        if set(samples) != expected:
            raise ValueError(f"expected sample blocks {sorted(expected)}")
        if isinstance(self.plan, DyadicPlan):
            return {
                "base": self.plan.encode_base(samples["base"]),
                "correction": self.plan.encode_correction(samples["correction"]),
            }
        return {"refinement": self.plan.encode(samples["refinement"])}

    def decode(
        self,
        bits: Mapping[str, np.ndarray],
        *,
        center: float,
        blocks: int = 1,
    ) -> float:
        expected = set(self.query_counts)
        if set(bits) != expected:
            raise ValueError(f"expected bit blocks {sorted(expected)}")
        if isinstance(self.plan, DyadicPlan):
            return self.plan.decode(
                bits["base"], bits["correction"], center, blocks=blocks
            )
        return self.plan.decode(bits["refinement"], center, blocks=blocks)

    def audit_immutable(self, operation) -> None:
        """Run ``operation`` and fail if it mutates the public query plan."""
        before = self.fingerprint()
        operation()
        after = self.fingerprint()
        if before != after:
            raise RuntimeError("public query plan changed after compilation")


def compile_refinement(
    *,
    backend: Backend,
    k: float,
    sigma: float,
    epsilon: float,
    tau: float,
    refinement_samples: int,
    seed: int,
    base_samples: int | None = None,
) -> CompiledRefinement:
    """Compile either construction behind one backend-neutral interface."""
    if backend == "dyadic":
        plan = build_dyadic_plan(
            k=k,
            sigma=sigma,
            epsilon=epsilon,
            tau=tau,
            base_samples=refinement_samples if base_samples is None else base_samples,
            correction_samples=refinement_samples,
            seed=seed,
        )
    elif backend == "continuous":
        if base_samples is not None:
            raise ValueError("base_samples applies only to the dyadic backend")
        plan = build_continuous_plan(
            k=k,
            sigma=sigma,
            epsilon=epsilon,
            tau=tau,
            samples=refinement_samples,
            seed=seed,
        )
    else:
        raise ValueError("backend must be 'dyadic' or 'continuous'")
    return CompiledRefinement(backend=backend, plan=plan)
