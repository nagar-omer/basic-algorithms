from basic_algorithms.search.bubble_sort import bubble_sort
import numpy as np


def test_quick_sort():
    arr = [1, 3, 5, 1, 0.2, 10, 0, 399, 4, 4, 5, 6, 1 , 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert bubble_sort(arr) == sorted(arr)


def test_quick_sort_random():
    np.random.seed(2026)
    arr = np.random.randint(10000, size=1000).tolist()
    assert bubble_sort(arr) == sorted(arr)
