from basic_algorithms.search.merge_sort import merge_sorted_arrays, merge_sort


def test_merge_sorted_arrays():
    arr1, arr2, arr3 = [1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert arr3 == merge_sorted_arrays(arr1, arr2), 'Arrays are not merged correctly'


def test_merge_sorted_arrays_empty_arrays():
    arr1, arr2, arr3 = [], [], []
    assert arr3 == merge_sorted_arrays(arr1, arr2), 'Arrays are not merged correctly'


def test_merge_sorted_arrays_one_empty_array():
    arr1, arr2 = [], [2, 4, 6, 9, 100]
    assert arr2 == merge_sorted_arrays(arr1, arr2), 'Arrays are not merged correctly'


def test_merge_sort():
    arr = [1, 3, 5, 1, 0.2, 10, 0, 399, 4, 4, 5, 6, 1 , 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert merge_sort(arr) == sorted(arr)
