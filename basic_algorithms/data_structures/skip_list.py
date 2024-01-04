import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """
    Skip list node
    """
    key: float = field(default=None, compare=True)
    value: Any = field(default=None, compare=False)
    next = None
    prev = None
    up = None
    down = None

    def __init__(self, key: float, value: Any, layer: int):
        self.value = value
        self.key = key
        self.layer = layer
        self.next = None
        self.prev = None
        self.up = None
        self.down = None

    def __repr__(self):
        return f'{self.key}'


class Tower:
    """
    Tower of nodes
    """
    def __init__(self, key: float, value: Any, max_height: int, p: float):
        # initialize tower with a single node
        self._len = 0
        self.head = Node(key=key, value=value, layer=0)
        self.tail = self.head

        # add nodes to the tower (with probability p for each node)
        for level in range(1, max_height + 1):
            if random.random() > p:
                break
            self._add(Node(key=key, value=None, layer=level))

    def _add(self, node):
        """
        Add node to the tower
        :param node: node to add
        :return: None
        """
        self.head.up = node
        node.down = self.head
        self.head = node
        self._len += 1

    def __iter__(self):
        """
        Iterate over the tower
        :return: node
        """
        node = self.head
        while node is not None:
            yield node
            node = node.down

    def __len__(self):
        """
        Get the length of the tower
        :return: length
        """
        return self._len


class SkipList:
    def __init__(self, max_height=16, p=0.5):
        """
        Skip list data structure
        :param max_height: maximum height of a node
        :param p: probability of increasing the height of a node
        """

        self._len = 0

        # set parameters
        self._max_height = max_height
        self._p = p

        # set list boundaries
        self._head = Tower(key=-float('inf'), value=None, max_height=max_height, p=1)
        self._tail = Tower(key=float('inf'), value=None, max_height=max_height, p=1)

        self._cursor = self._head.head

        # connect head and tail
        for head_node, tail_node in zip(self._head, self._tail):
            head_node.next = tail_node
            tail_node.prev = head_node

    def find_closest(self, key: float):
        """
        Find value in the skip list that is closest to the given value (smaller or equal)
        :param key: identifier of the value
        :return: node
        """
        node = self._head.head
        while node.down is not None:
            node = node.down
            while node.next.key < key:
                node = node.next
        return node

    def insert(self, value: float):
        """
        Insert value into the skip list
        :param value:
        :return:
        """
        # create new tower
        new_tower = Tower(key=value, value=None, max_height=self._max_height, p=self._p)
        new_node = new_tower.tail
        closest_node = self.find_closest(value)

        # insert new tower (connection between new key and larger key)
        closest_node.next.prev = new_node
        new_node.next = closest_node.next

        # insert new tower (connection between new key and smaller key)
        closest_node.next = new_node
        new_node.prev = closest_node

        smaller_node = new_node.prev
        while new_node.up is not None:
            # move one level up
            new_node = new_node.up

            # search for the closest smaller node with a higher layer
            while smaller_node.up is None:
                smaller_node = smaller_node.prev
            smaller_node = smaller_node.up

            # connect larger node to new node
            smaller_node.next.prev = new_node
            new_node.next = smaller_node.next

            # connect smaller node to new node
            smaller_node.next = new_node
            new_node.prev = smaller_node

        self._len += 1

    def get_current_node(self):
        return self._cursor

    def move_cursor_to_next(self):
        if self._cursor.next is not None:
            self._cursor = self._cursor.next

    def move_cursor_to_closest_larger_key(self, key):
        """
        Move cursor to the node with the closest larger key
        :param key: key to compare with
        :return: None
        """
        node = self._tail.head
        while node.down is not None:
            node = node.down
            while node.prev.key > key:
                node = node.prev

        self._cursor = node

    def __repr__(self):
        """
        Get string representation of the skip list
        :return: string
        """
        node = self._head.tail

        build_str = ''
        while node is not None:
            build_str += f'{node} -> '
            node = node.next

        return build_str



if __name__ == '__main__':
    sl = SkipList()
    sl.insert(1)
    print(sl)
    sl.insert(3)
    print(sl)
    sl.insert(2)
    print(sl)
    sl.insert(5)
    print(sl)
    sl.insert(4)
    print(sl)
    e = 0

