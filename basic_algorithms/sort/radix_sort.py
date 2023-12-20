from typing import List
import numpy as np


def _get_max_digits(arr):
    """
    Get the number of digits in the largest number in the array
    :param arr: array to be sorted
    :return: number of digits in the largest number
    """
    if len(arr) == 0:
        return 0
    return len(str(np.abs(arr).astype(int).max()))


def _split_by_polarity(arr):
    """
    Split array into two arrays: one with positive numbers and one with negative numbers
    :param arr: array to be split
    :return: negatives, positives
    """
    negatives, positives = [], []
    for val in arr:
        if val < 0:
            negatives.append(val)
        else:
            positives.append(val)
    return negatives, positives


def get_digit(val, digit):
    """
    Get the digit at the specified index: 0 is the LSD, 1 is the second LSD, etc.
    :param val: value to get digit from
    :param digit: index of digit to get
    :return: digit at index
    """
    if digit + 1 > len(str(abs(val))):
        return 0

    return int(str(val)[-digit-1])


def lsb_sort(arr):
    """
    Least significant bit sort algorithm
    sort by LSD, then by second LSD, etc.
    :param arr: array to be sorted (positive integers only or negative integers only)
    :return: sorted array ~O(n + k)
    """
    # nothing to sort
    if len(arr) <= 1:
        return arr

    # get the number of digits in the largest number
    k_digits = _get_max_digits(arr)
    # sort by each digit
    for digit in range(k_digits):
        # throw to buckets (each bucket for a specific digit)
        buckets = [[] for _ in range(10)]
        for val in arr:
            buckets[get_digit(val, digit)].append(val)

        # merge buckets back into array -> sorted by digit
        sorted_by_digit = []
        for bucket in buckets:
            sorted_by_digit.extend(bucket)
        arr = sorted_by_digit
    return arr


def radix_sort(arr: List[int]):
    """
    Radix sort algorithm - for integers only - STABLE
    :param arr: array to be sorted
    :return: sorted array ~O(n + k)
    """
    for i in range(len(arr)):
        if not isinstance(arr[i], int):
            raise ValueError("Radix sort can only be applied to integers")

    negatives, positives = _split_by_polarity(arr)
    sorted_negatives = lsb_sort(negatives)
    sorted_positives = lsb_sort(positives)
    return sorted_negatives[::-1] + sorted_positives

