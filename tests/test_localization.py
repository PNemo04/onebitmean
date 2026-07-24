import numpy as np

from onebitmean.localization import build_localization_plan


def test_research_localizer_recovers_a_noise_free_bin_without_adaptation() -> None:
    plan = build_localization_plan(
        lam=100.0,
        sigma=1.0,
        delta=0.05,
        seed=31,
        profile="research",
    )
    before = plan.fingerprint()
    mean = 37.0
    bits = plan.encode(np.full(plan.query_count, mean))
    left, right = plan.decode_interval(bits)
    assert left <= mean <= right
    assert right - left <= 100.0
    assert plan.fingerprint() == before


def test_certified_profile_builds_a_balanced_two_word_codebook() -> None:
    plan = build_localization_plan(
        lam=21.0,
        sigma=1.0,
        delta=0.1,
        seed=37,
        profile="certified",
    )
    assert plan.codebook.shape[0] == 3
    length = plan.query_count
    for left in range(plan.codebook.shape[0]):
        for right in range(left + 1, plan.codebook.shape[0]):
            distance = np.count_nonzero(plan.codebook[left] != plan.codebook[right])
            assert 0.49 * length <= distance <= 0.51 * length


def test_trivial_localization_needs_no_queries() -> None:
    plan = build_localization_plan(
        lam=5.0,
        sigma=1.0,
        delta=0.1,
        seed=41,
        profile="certified",
    )
    assert plan.query_count == 0
    assert plan.decode_interval(np.array([], dtype=np.uint8)) == (-5.0, 5.0)

