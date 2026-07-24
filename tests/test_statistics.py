import numpy as np
import pytest

from onebitmean.statistics import median_of_means, mom_block_count


def test_median_of_means_uses_equal_blocks_and_upper_median() -> None:
    values = np.array([0.0, 0.0, 3.0, 3.0, 100.0, 100.0, -9.0])
    assert median_of_means(values, blocks=3) == 3.0


def test_mom_block_count_and_validation() -> None:
    assert mom_block_count(0.1) >= 8
    with pytest.raises(ValueError):
        mom_block_count(0.5)
    with pytest.raises(ValueError):
        median_of_means(np.array([]), blocks=1)

