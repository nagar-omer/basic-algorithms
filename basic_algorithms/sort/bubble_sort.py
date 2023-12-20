

def bubble_sort(arr):
    """
    Bubble sort algorithm
    :param arr: array to be sorted
    :return: sorted array ~O(n^2)
    """

    sort_up_to = len(arr) - 1
    swapped = True

    # keep swapping until no swaps are made
    while swapped:
        swapped = False
        for i in range(sort_up_to):
            # swap if out of order
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        # max value is now at the end, so we don't need to check it again
        sort_up_to -= 1

    return arr
