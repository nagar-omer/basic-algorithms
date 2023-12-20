import pytest
from basic_algorithms.sort.quick_sort import _pick_pivot, quick_sort
import numpy as np


def test_pick_pivot():
    assert _pick_pivot('first', 5) == 0
    assert _pick_pivot('first', 10) == 0
    assert _pick_pivot('random', 5) in range(5)
    with pytest.raises(Exception):
        _pick_pivot('random', 0)


def test_quick_sort():
    arr = [1, 3, 5, 1, 0.2, 10, 0, 399, 4, 4, 5, 6, 1 , 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert quick_sort(arr) == sorted(arr)


def test_merge_sort_random():
    np.random.seed(2026)
    arr = np.random.randint(10000, size=1000).tolist()
    assert quick_sort(arr) == sorted(arr)
