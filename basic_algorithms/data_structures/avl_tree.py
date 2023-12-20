from dataclasses import dataclass, field
from typing import Union, Optional, Any


@dataclass(order=True)
class TreeNode:
    """
    Helper class for max sequence up to n items.
    """
    key: Union[int, float] = field(compare=True)
    color: str = field(compare=False)
    value: [Any] = field(default=None, compare=False)
    smaller_child: Optional['TreeNode'] = field(default=None, compare=False)
    larger_child: Optional['TreeNode'] = field(default=None, compare=False)
    parent: Optional['TreeNode'] = field(default=None, compare=False)


class AVLTree:
    def __init__(self):
        self.root = None

    def _bst_insert(self, key, value, node) -> TreeNode:
        # four cases:
        # key < node.key  ->  smaller child is None or smaller child is not None
        # key >= node.key ->  larger child is None or larger child is not None
        if node.smaller_child is None and key < node.key:
            node.smaller_child = TreeNode(key, value=value, parent=node, color='red')
            return node.smaller_child
        elif node.larger_child is None and key >= node.key:
            node.larger_child = TreeNode(key, value=value, parent=node, color='red')
            return node.larger_child
        elif node.smaller_child is not None and key < node.key:
            return self._bst_insert(key, value, node=node.smaller_child)
        elif node.larger_child is not None and key >= node.key:
            return self._bst_insert(key, value, node=node.larger_child)
        else:
            raise ValueError(f'Invalid case for key={key}, node.key={node.key}, '
                             f'node.smaller_child={node.smaller_child}, node.larger_child={node.larger_child}')

    def _get_relatives(self, node):
        parent = node.parent
        if parent.parent == None:
            return parent, None, None
        grandparent = parent.parent
        uncle = grandparent.smaller_child if grandparent.smaller_child != parent else grandparent.larger_child
        return parent, grandparent, uncle

    def _is_right_child(self, parent, node):
        return parent.larger_child == node

    def _is_left_child(self, parent, node):
        return parent.smaller_child == node

    def _left_rotate(self, node, parent, grandparent, uncle):
        """
        assuming:
            1. node is a larger than parent
            2. uncle is larger than grandparent
            3. parent is red -> has only one child
        :param node:
        :param parent:
        :param grandparent:
        :param uncle:
        """
        # node > parent > grandparent > uncle
        # 1. swap parent and node
        # 2. make parent smaller child of node
        parent.larger_child = None
        parent.smaller_child = None
        parent.parent = node

        node.larger_child = None
        node.smaller_child = parent
        node.parent = grandparent

    def _right_rotate(self, node, parent, grandparent, uncle):
        """
        assuming:
            1. node is a smaller than parent
            2. uncle is smaller than grandparent
            3. parent is red -> has only one child
        :param node:
        :param parent:
        :param grandparent:
        :param uncle:
        :return:
        """
        # node < parent < grandparent < uncle
        # 1. swap parent and node
        # 2. make parent larger child of node
        parent.larger_child = None
        parent.smaller_child = None
        parent.parent = node

        node.larger_child = parent
        node.smaller_child = None
        node.parent = grandparent

    def _post_insert_fixup(self, node):
        # stop condition - nothing to fix the tree is valid
        if node.parent is None or node.parent.color == 'black':
            return

        parent, grandparent, uncle = self._get_relatives(node)

        # parent is the root of the tree
        if grandparent is None:
            parent.color = 'black'
            return

        # uncle is red
        if uncle is not None and uncle.color == 'red':
            parent.color = 'black'
            uncle.color = 'black'
            grandparent.color = 'red'
            self._post_insert_fixup(grandparent)
            return

        # node < parent < grandparent < uncle
        if self._is_left_child(parent, node) and self._is_left_child(grandparent, parent):
            # node is smaller than parent
            # parent is smaller than grandparent
            # uncle is larger than grandparent
            self._right_rotate(node, parent, grandparent, uncle)
            parent.color = 'black'
            grandparent.color = 'red'
            return

    def insert(self, key, value=None):
        """
        Insert a new node into the tree
        :param key: key of the new node
        :param value: value of the new node
        :return: None
        """
        if self.root is None:
            self.root = TreeNode(key=key, value=value, color='black')
        else:
            node = self._bst_insert(key, value, self.root)
            self._post_insert_fixup(node)

    