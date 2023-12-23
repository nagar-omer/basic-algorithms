from basic_algorithms.data_structures.heap import Heap, heap_sort
import numpy as np


def test_heap_length():
    arr = [4] * 10
    heap = Heap(heap_type='min')
    for val in arr:
        heap.push(val)
    assert len(heap) == len(arr)


def test_heap_push_min():
    arr = [100, 2, 1, 43, 8, 1, 33]
    heap = Heap(heap_type='min')
    heap.push_array(arr)
    assert heap.peek(0) == 1, 'Heap did not push correctly, top should be 1'


def test_heap_push_max():
    arr = [100, 2, 1, 43, 8, 1, 33]
    heap = Heap(heap_type='max')
    heap.push_array(arr)
    assert heap.peek(0) == 100, 'Heap did not push correctly, top should be 100'


def test_heap_is_balanced():
    arr = [100, 2, 1, 43, 8, 1, 33, 0, 0, 12, 342, 2, 33, 123, 53, 121, 123 ,121, 2, 1, 22 ,124, 12, 4, 12, 4]
    heap = Heap(heap_type='max')
    heap.push_array(arr)
    assert len(heap._heap) == len(arr), 'Heap is not balanced'


def test_merge_sort():
    arr = [1, 3, 5, 1, 0.2, 10, 0, 399, 4, 4, 5, 6, 1 , 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert heap_sort(arr) == sorted(arr)


def test_heap_sort_random():
    np.random.seed(2026)
    arr = np.random.randint(10000, size=1000).tolist()
    assert heap_sort(arr) == sorted(arr)
