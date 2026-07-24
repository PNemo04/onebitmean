import numpy as np
import pytest

from onebitmean.dyadic import build_dyadic_plan
from onebitmean.protocol import CompiledRefinement, compile_refinement


@pytest.mark.parametrize("backend", ["dyadic", "continuous"])
def test_common_protocol_interface_is_decoder_centered_and_immutable(backend: str) -> None:
    compiled = compile_refinement(
        backend=backend,
        k=2.0,
        sigma=1.0,
        epsilon=0.2,
        tau=1.0,
        refinement_samples=20_000,
        seed=43,
    )
    center = 0.1
    samples = {
        name: np.full(count, center + 0.5)
        for name, count in compiled.query_counts.items()
    }
    bits = compiled.encode(samples)
    compiled.audit_immutable(lambda: compiled.decode(bits, center=center))
    with pytest.raises(ValueError):
        compiled.encode({"wrong": np.zeros(1)})


def test_common_protocol_rejects_a_mismatched_backend_label() -> None:
    plan = build_dyadic_plan(
        k=2.0,
        sigma=1.0,
        epsilon=0.2,
        tau=1.0,
        base_samples=2,
        correction_samples=2,
        seed=47,
    )
    with pytest.raises(ValueError, match="does not match"):
        CompiledRefinement(backend="continuous", plan=plan)
