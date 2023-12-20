import numpy as np


def _pick_pivot(method, n):
    """
    Pick a pivot index from a list of length n
    :param method: first or random
    :param n: length of the list
    :return: pivot index
    """
    if n <= 0:
        raise ValueError('n must be positive, > 1')

    if method == 'random':
        return np.random.randint(n)
    elif method == 'first':
        return 0
    else:
        raise ValueError('Invalid method')


def quick_sort(arr):
    """
    Quick sort algorithm
    :param arr: list of values to be sorted
    :return: sorted list ~O(nlogn)
    """

    # stop condition -> nothing to sort
    if len(arr) <= 1:
        return arr

    # pick pivot
    pivot = _pick_pivot('first', len(arr))
    pivot_val = arr[pivot]

    # create partition
    smaller_than_pivot, larger_than_pivot = [], []
    for i, val in enumerate(arr):
        if i == pivot:
            continue
        if val < pivot_val:
            smaller_than_pivot.append(val)
        else:
            larger_than_pivot.append(val)

    # recursive call -> sort left and right partitions
    return quick_sort(smaller_than_pivot) + [pivot_val] + quick_sort(larger_than_pivot)
