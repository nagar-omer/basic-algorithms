import pytest

from basic_algorithms.radix_sort import radix_sort, _get_max_digits, _split_by_polarity
import numpy as np


def test_get_max_digits():
    arr = [1, 3, 5, 1, 0.2, 10, 0, -399, 4, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, -8, 9, 10]
    assert _get_max_digits(arr) == 3


def test_split_by_polarity():
    arr1 = np.random.randint(10000, size=1000)
    arr2 = -np.random.randint(10, 1000, size=100)
    neg, pos = _split_by_polarity(arr1.tolist() + arr2.tolist())
    assert np.all(np.asarray(neg) < 0), "found positive in negatives"
    assert np.all(np.asarray(pos) >= 0), "found negative in positives"
    assert len(neg) == len(arr2), "negatives length is not correct"
    assert len(pos) == len(arr1), "positives length is not correct"
    assert np.all([v in arr2 for v in neg]), "negatives are not in original array"
    assert np.all([v in arr1 for v in pos]), "positives are not in original array"


def test_value_error_for_non_integers():
    with pytest.raises(ValueError):
        radix_sort([1, 2, 3, 4., 5, 6, 7, 8, 9, 10.0])


def test_quick_sort():
    arr = [1, 3, 5, 1, 2, 10, 0, 399, 4, 4, 5, 6, 1 , 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert radix_sort(arr) == sorted(arr)


def test_merge_sort_random():
    np.random.seed(2026)
    arr1 = np.random.randint(10000, size=1000).astype(int).tolist()
    arr2 = (-np.random.randint(10, 1000, size=100)).astype(int).tolist()
    assert radix_sort(arr1 + arr2) == sorted(arr1 + arr2)
