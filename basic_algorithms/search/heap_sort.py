from numbers import Number
from typing import List
import numpy as np


class Heap:
    """
    Heap data structure
    """
    def __init__(self, heap_type='min'):
        """
        Init heap
        :param heap_type: min or max heap
        """
        # init heap
        # heap arrangement [root, r1, r2, r11, r12, r21, r22, r23, r24, ... ]
        self._heap = []
        self._heap_type = heap_type

    def _is_valid(self, parent: int, child: int):
        """
        Check if parent and child are in valid order
        :param parent: parent index
        :param child: child index
        :return:
        """
        assert parent < len(self._heap) and child < len(self._heap), 'Invalid parent or child index'
        if self._heap_type == 'min':
            return self._heap[parent] <= self._heap[child]
        elif self._heap_type == 'max':
            return self._heap[parent] >= self._heap[child]

    def _left_child(self, pos: int):
        """
        Get left child index
        :param pos: parent index
        :return: left child index
        """
        return 2 * pos + 1

    def _right_child(self, pos: int):
        """
        Get right child index
        :param pos: parent index
        :return: right child index
        """
        return 2 * pos + 2

    def _parent(self, pos: int):
        """
        Get parent index
        :param pos: child index
        :return: parent index
        """
        assert pos > 0, f'Invalid child index {pos}'
        return (pos - 1) // 2

    def _swap(self, parent: int, child: int):
        """
        swap parent and child
        :param parent: parent index
        :param child: child index
        """
        assert self._parent(child) == parent, 'Invalid parent or child index'
        self._heap[parent], self._heap[child] = self._heap[child], self._heap[parent]

    def _bubble_up(self, pos: int = None):
        """
        Bubble up the value at pos until heap is sorted
        :param pos: index of unsorted value
        :return:
        """

        # default pos is last element
        if pos is None:
            pos = len(self._heap) - 1

        # stop condition -> reached root
        if pos == 0:
            return

        parent = self._parent(pos)
        # stop condition -> heap is sorted
        if self._is_valid(parent, pos):
            return

        # swap parent and child and continue heapifying
        self._swap(parent, pos)
        return self._bubble_up(parent)

    def push(self, value: Number):
        """
        Insert value into heap
        :param value: value to add
        """
        self._heap.append(value)
        self._bubble_up()

    def push_array(self, arr: list):
        """
        Insert value into heap
        :param arr: list of values to add
        """
        for val in arr:
            self.push(val)

    def _get_swap_child(self, left: int, right: int):
        """
        Get child to swap with parent in case of heap violation
        behaviour depends on heap type
        :param left: left child index
        :param right: right child index
        :return: index of child to swap with parent
        """
        if self._heap_type == 'min':
            return left if self._heap[left] < self._heap[right] else right
        elif self._heap_type == 'max':
            return left if self._heap[left] > self._heap[right] else right
        else:
            raise ValueError(f'Invalid heap type {self._heap_type}')

    def _get_children(self, parent: int) -> List[int]:
        """
        Get children of node
        :return: list of children
        """
        left, right = self._left_child(parent), self._right_child(parent)
        valid_children = [child for child in [left, right] if child < len(self._heap)]
        return valid_children

    def _heapify(self, pos: int = 0):
        """
        Rearrange heap after pop
        :param pos: position to start heapifying
        """

        # get children of node and check if heap is valid
        children = self._get_children(pos)
        if np.all([self._is_valid(parent=pos, child=c) for c in children]):
            return

        # nothing to do if no children -> reached leaf
        if not children:
            return

        # swap parent with child and continue heapifying
        swap_with = children[0] if len(children) == 1 else self._get_swap_child(children[0], children[1])
        self._swap(pos, swap_with)
        self._heapify(swap_with)

    def pop(self) -> Number:
        """
        Pop root element from heap (min or max depending on heap type)
        :return: value of root element
        """
        # swap root and last element
        return_val = self._heap[0]
        self._heap[0] = self._heap[-1]
        self._heap = self._heap[:-1]

        # rearrange heap
        self._heapify()

        return return_val

    def peek(self, pos: int = 0) -> Number:
        """
        Peek at value at a specific pos
        :param pos: position to peek at (default: root)
        :return: value at pos
        """
        return self._heap[pos]

    def __len__(self):
        return len(self._heap)


def heap_sort(arr):
    """
    Sort array using heap sort
    complexity: O(n log n)
    :param arr: array to sort
    :return: sorted array
    """
    heap = Heap(heap_type='min')
    for val in arr:
        heap.push(val)
    return [heap.pop() for _ in range(len(arr))]

