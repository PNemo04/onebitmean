import numpy as np

from onebitmean.dyadic import (
    build_dyadic_plan,
    distance_to_grid,
    residue,
    safe_phase,
)


def test_safe_phase_distance_is_at_least_quarter_period() -> None:
    rng = np.random.default_rng(7)
    centers = rng.normal(size=2000) * 100.0
    lengths = np.exp(rng.uniform(-4.0, 5.0, size=2000))
    for center, length in zip(centers, lengths):
        phase = safe_phase(float(center), float(length))
        assert distance_to_grid(float(center), float(length), phase) >= length / 4.0 - 1e-11


def test_pointwise_telescope_is_exact() -> None:
    rng = np.random.default_rng(11)
    for _ in range(200):
        center = float(rng.normal() * 4.0)
        sample = float(rng.normal() * 12.0)
        lengths = 1.7 * 2.0 ** np.arange(8)
        phases = [safe_phase(center, float(length)) for length in lengths]
        values_x = np.array([residue(sample, length, phase) for length, phase in zip(lengths, phases)])
        values_c = np.array([residue(center, length, phase) for length, phase in zip(lengths, phases)])
        base = values_x[0] - values_c[0]
        correction = np.diff(values_x - values_c).sum()
        assert np.isclose(base + correction, values_x[-1] - values_c[-1], atol=1e-12)


def test_dyadic_plan_is_frozen_and_decoder_only_centering_is_unbiased() -> None:
    count = 300_000
    center = 0.37
    sample = center + 0.61
    plan = build_dyadic_plan(
        k=3.0,
        sigma=1.0,
        epsilon=0.1,
        tau=1.0,
        base_samples=count,
        correction_samples=count,
        seed=19,
    )
    before = plan.fingerprint()
    base_bits = plan.encode_base(np.full(count, sample))
    correction_bits = plan.encode_correction(np.full(count, sample))
    estimate = (
        plan.decode_base_statistics(base_bits, center).mean()
        + plan.decode_correction_statistics(correction_bits, center).mean()
    )
    conditional_mean, conditional_second = plan.ideal_conditional_moments(
        np.full(count, sample), center
    )
    base_mean, base_second, correction_mean, correction_second = (
        plan.ideal_conditional_block_moments(np.full(count, sample), center)
    )
    raw_base = plan.decode_base_statistics(base_bits, center)
    raw_correction = plan.decode_correction_statistics(correction_bits, center)
    raw_second = raw_base**2 + raw_correction**2
    assert abs(estimate - (sample - center)) < 0.04
    assert np.allclose(conditional_mean, sample - center)
    assert np.allclose(base_mean + correction_mean, conditional_mean)
    assert np.allclose(base_second + correction_second, conditional_second)
    second_se = raw_second.std(ddof=1) / np.sqrt(count)
    assert abs(raw_second.mean() - conditional_second[0]) <= 5.0 * second_se
    assert plan.fingerprint() == before
    assert not plan.base_phase.flags.writeable
    assert not plan.correction_threshold.flags.writeable
