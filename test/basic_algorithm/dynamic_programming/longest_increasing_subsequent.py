from basic_algorithms.dynamic_programming.longest_increasing_subsequent import longest_increasing_subsequence


def test_empty():
    assert longest_increasing_subsequence([]) == [], "Empty list"


def test_single_element():
    assert longest_increasing_subsequence([1]) == [], "Single element list"


def test_increasing():
    assert longest_increasing_subsequence([1, 2, 3]) == [1, 2, 3], 'Increasing list'


def test_decreasing():
    assert longest_increasing_subsequence([5, 4, 3, 2, 1]) == [], 'Decreasing list'


def test_increasing_and_decreasing():
    assert longest_increasing_subsequence([2, 1, 0, 1, 2, 3, 4, 5]) == [0, 1, 2, 3, 4, 5], 'Increasing and decreasing list'
