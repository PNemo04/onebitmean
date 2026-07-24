import numpy as np

from onebitmean.continuous import build_continuous_plan, cell_sign


def test_affine_cell_colors_are_exactly_pairwise_independent() -> None:
    seeds = np.repeat(np.arange(256, dtype=np.uint64), 2)
    offsets = np.tile(np.array([0, 1], dtype=np.uint8), 256)
    left = cell_sign(seeds, offsets, np.full(512, 3, dtype=np.int64))
    right = cell_sign(seeds, offsets, np.full(512, 97, dtype=np.int64))
    joint = {
        (left_sign, right_sign): int(np.count_nonzero(
            (left == left_sign) & (right == right_sign)
        ))
        for left_sign in (-1.0, 1.0)
        for right_sign in (-1.0, 1.0)
    }
    assert set(joint.values()) == {128}


def test_continuous_exact_window_identity_and_plan_fingerprint() -> None:
    count = 350_000
    center = -0.23
    displacement = 0.8
    plan = build_continuous_plan(
        k=2.0,
        sigma=1.0,
        epsilon=0.1,
        tau=1.0,
        samples=count,
        seed=23,
    )
    before = plan.fingerprint()
    bits = plan.encode(np.full(count, center + displacement))
    statistics = plan.decode_statistics(bits, center)
    conditional_mean, conditional_second = plan.ideal_conditional_moments(
        np.full(count, center + displacement), center
    )
    standard_error = statistics.std(ddof=1) / np.sqrt(count)
    assert abs(statistics.mean() - displacement) <= 5.0 * standard_error + 0.01
    assert np.allclose(conditional_mean, displacement, atol=1e-12)
    square_standard_error = np.std(statistics**2, ddof=1) / np.sqrt(count)
    assert abs(np.mean(statistics**2) - conditional_second[0]) <= 6.0 * square_standard_error
    assert plan.fingerprint() == before
    assert not plan.coloring_seed.flags.writeable
    assert not plan.coloring_offset.flags.writeable


def test_continuous_statistics_vanish_in_same_cell_control_case() -> None:
    plan = build_continuous_plan(
        k=1.5,
        sigma=1.0,
        epsilon=0.2,
        tau=1.0,
        samples=10_000,
        seed=29,
    )
    center = 0.0
    bits = plan.encode(np.full(plan.scale.size, center))
    statistics = plan.decode_statistics(bits, center)
    assert np.count_nonzero(statistics) == 0
