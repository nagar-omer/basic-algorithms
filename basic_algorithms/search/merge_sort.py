
def merge_sorted_arrays(arr1, arr2):
    """
    Merge two sorted arrays
    :param arr1: array 1
    :param arr2: array 2
    :return: sorted array of merged values O(n1 + n2)
    """
    # get lengths and init merged array
    idx1, idx2 = 0, 0
    len1, len2 = len(arr1), len(arr2)
    merged = []

    # while merged array is not full
    for _ in range(len1 + len2):
        # if both arrays have values, compare and append the smaller one
        if idx1 < len1 and idx2 < len2:
            if arr1[idx1] < arr2[idx2]:
                merged.append(arr1[idx1])
                idx1 += 1
            else:
                merged.append(arr2[idx2])
                idx2 += 1
        # if one of the arrays is empty, append the rest of the other array
        elif idx1 < len1:
            merged.append(arr1[idx1])
            idx1 += 1
        elif idx2 < len2:
            merged.append(arr2[idx2])
            idx2 += 1
    return merged


def merge_sort(arr):
    """
    Merge sort algorithm
    complexity: O(nlogn)
    :param arr: array to be sorted
    :return: sorted array ~O(nlogn)
    """
    if len(arr) <= 1:
        return arr

    # split array in half
    split_pint = len(arr) // 2
    left, right = arr[:split_pint], arr[split_pint:]
    return merge_sorted_arrays(merge_sort(left), merge_sort(right))

