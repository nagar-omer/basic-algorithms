from typing import Optional, Any, Union
from dataclasses import dataclass, field
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize


@dataclass(order=True)
class TreeNode:
    """
    Helper class for max sequence up to n items.
    """
    key: Union[int, float] = field(compare=True)
    value: [Any] = field(default=None, compare=False)
    smaller_child: Optional['TreeNode'] = field(default=None, compare=False)
    larger_child: Optional['TreeNode'] = field(default=None, compare=False)
    parent: Optional['TreeNode'] = field(default=None, compare=False)


class binarySearchTree:
    def __init__(self):
        self._root = None

    def _insert(self, key: Union[int, float], value: Any, node: TreeNode):
        """
        Recursive insert function
        insert key as a leaf in the tree
        :param key: sort-by value
        :param value: content to store
        :param node: potential parent node
        """
        if node is None:
            return

        # four cases:
        # key < node.key  ->  smaller child is None or smaller child is not None
        # key >= node.key ->  larger child is None or larger child is not None
        if node.smaller_child is None and key < node.key:
            node.smaller_child = TreeNode(key, value=value, parent=node)
        elif node.larger_child is None and key >= node.key:
            node.larger_child = TreeNode(key, value=value, parent=node)
        elif node.smaller_child is not None and key < node.key:
            self._insert(key, value, node=node.smaller_child)
        elif node.larger_child is not None and key >= node.key:
            self._insert(key, value, node=node.larger_child)
        else:
            raise ValueError(f'Invalid case for key={key}, node.key={node.key}, '
                             f'node.smaller_child={node.smaller_child}, node.larger_child={node.larger_child}')

    def insert(self, key, value=None):
        """
        Insert a key-value pair into the tree
        :param key: sort-by value
        :param value: content to store
        """
        if self._root is None:
            self._root = TreeNode(key, value=value)
        else:
            self._insert(key, value, node=self._root)

    def _find(self, key: Union[int, float], node: TreeNode) -> Optional[TreeNode]:
        if node is None:
            return None
        if node.key == key:
            return node

        if key < node.key:
            return self._find(key, node=node.smaller_child)
        else:
            return self._find(key, node=node.larger_child)

    def find(self, key: Union[int, float]) -> Optional[TreeNode]:
        """
        Find a node in the tree by key
        :param key: key to search for
        :return: node with matching key
        """
        return self._find(key, node=self._root)


    def _remove_node(self, node: TreeNode):
        """
        Remove a node from the tree (without deleting the node object)
        :param node: node to remove
        """
        if node is None:
            return

        parent = node.parent
        # special case - root node with no children
        if node.smaller_child is None and node.larger_child is None and parent is None:
            self._root = None
            return
        # case 1 - node has no children
        if node.smaller_child is None and node.larger_child is None:
            if parent.smaller_child == node:
                parent.smaller_child = None
            else:
                parent.larger_child = None
            return
        # case 2 - node has one larger child
        if node.smaller_child is None and node.larger_child is not None:
            if parent.smaller_child == node:
                parent.smaller_child = node.larger_child
            else:
                parent.larger_child = node.larger_child
            return
        # case 3 - node has one smaller child
        if node.smaller_child is not None and node.larger_child is None:
            if parent.smaller_child == node:
                parent.smaller_child = node.smaller_child
            else:
                parent.larger_child = node.smaller_child
            return
        # case 4 - node has two children
        if node.smaller_child is not None and node.larger_child is not None:
            # swap the smallest node with the node to be removed
            node.key, node.smaller_child.key = node.smaller_child.key, node.key
            node.value, node.smaller_child.value = node.smaller_child.value, node.value
            # remove the smallest node
            self._remove_node(node.smaller_child)

    def pop(self, key) -> Optional[TreeNode]:
        node = self.find(key)
        if node is None:
            return None

        self._remove_node(node)
        return node

    def to_networkx(self, node=None, gnx=None, parent_id=None):
        """
        Convert the tree to a networkx graph
        :param node: node to start from
        :param gnx: accumulated graph
        :param parent_id: parent node id
        :return: networkx graph
        """
        if node is None:
            node = self._root if node is None else node
            gnx = nx.DiGraph() if gnx is None else gnx
            parent_id = f"#0 - {node.key}"
            gnx.add_node(parent_id, value=node.value, key=node.key)
        if node.smaller_child is not None:
            new_node_id = f"#{gnx.number_of_nodes()} - {node.smaller_child.key}"
            gnx.add_node(new_node_id, value=node.smaller_child.value, key=node.smaller_child.key)
            gnx.add_edge(parent_id, new_node_id)
            self.to_networkx(node=node.smaller_child, gnx=gnx, parent_id=new_node_id)
        if node.larger_child is not None:
            new_node_id = f"#{gnx.number_of_nodes()} - {node.larger_child.key}"
            gnx.add_node(new_node_id, value=node.larger_child.value, key=node.larger_child.key)
            gnx.add_edge(parent_id, new_node_id)
            self.to_networkx(node=node.larger_child, gnx=gnx, parent_id=new_node_id)
        return gnx

    def _get_gnx_children(self, gnx: nx.DiGraph, node):
        """
        Get the children of a node in a graph by ordering - smaller child first, larger child second
        :param gnx: networkx graph
        :param node: node id (# - key)
        :return: (smaller_child, larger_child), None if no children
        """
        assert gnx.has_node(node), f"Node {node} not in graph"
        parent_key = gnx.nodes[node]['key']
        children = list(gnx.successors(node))
        if not children:
            return None, None
        if len(children) == 1:
            if gnx.nodes[children[0]]['key'] < parent_key:
                return children[0], None
            else:
                return None, children[0]
        if len(children) == 2:
            return (children[0], children[1]) if gnx.nodes[children[0]]['key'] < gnx.nodes[children[1]]['key'] else \
                   (children[1], children[0])

    def _get_pos(self, gnx: nx.DiGraph, node=None, pos_dict=None, depth_dict=None, max_depth=None, horizontal_spacing=1, vertical_spacing=1):
        """
        Get the position of each node in the graph so it can be drawn as a tree
        :param gnx: the graph as a networkx DiGraph
        :param node: current node to get position for
        :param pos_dict: accumulated position dictionary
        :param depth_dict: dictionary of node depths (calculated once at the beginning)
        :param max_depth: maximum depth of the tree (calculated once at the beginning)
        :param horizontal_spacing: minimum horizontal spacing between nodes (on leaf level)
        :param vertical_spacing: vertical spacing between nodes
        :return: position dictionary on 2D plane
        """
        root_id = [node for node in gnx.nodes if gnx.in_degree(node) == 0][0]
        if node is None:
            depth_dict = nx.shortest_path_length(gnx, source=root_id)
            max_depth = max(depth_dict.values())
            pos_dict = {root_id: (0, nx.dag_longest_path_length(gnx))}
            node = root_id

        smaller_child, larger_child = self._get_gnx_children(gnx, node)
        horizontal_shift = (2 ** (max_depth - depth_dict[node])) * (horizontal_spacing)
        if smaller_child is not None:
            pos_dict[smaller_child] = (pos_dict[node][0] - horizontal_shift, pos_dict[node][1] - vertical_spacing)
            self._get_pos(gnx=gnx, node=smaller_child, pos_dict=pos_dict, depth_dict=depth_dict, max_depth=max_depth)
        if larger_child is not None:
            pos_dict[larger_child] = (pos_dict[node][0] + horizontal_shift, pos_dict[node][1] - vertical_spacing)
            self._get_pos(gnx=gnx, node=larger_child, pos_dict=pos_dict, depth_dict=depth_dict, max_depth=max_depth)
        return pos_dict

    def draw(self):
        """
        Draw the tree using networkx
        :return:
        """
        gnx = self.to_networkx()
        gnx_depth = nx.dag_longest_path_length(gnx)
        n_leafs = len([node for node in gnx.nodes if gnx.out_degree(node) == 0])
        pos_dict = self._get_pos(gnx=gnx)
        max_key = max([data["key"] for node, data in gnx.nodes(data=True)])
        colors = [cm.RdYlGn(data["key"] / max_key) for node, data in gnx.nodes(data=True)]

        fig, ax = plt.subplots(1, 1, figsize=(n_leafs, gnx_depth))
        nx.draw_networkx_nodes(gnx, pos=pos_dict, node_color=colors, node_size=20, cmap=plt.cm.RdYlGn, ax=ax)
        # nx.draw_networkx_labels(gnx, labels={node: data["key"] for node, data in gnx.nodes(data=True)},
        #                         pos={node: (x + 0.2, y) for node, (x, y) in pos_dict.items()}, font_size=8, ax=ax)
        nx.draw_networkx_edges(gnx, pos=pos_dict, width=0.5, arrowsize=1, ax=ax)

        # Create a ScalarMappable for the colorbar
        norm = Normalize(vmin=0, vmax=max_key)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
        sm.set_array([])

        # Add the colorbar to the figure
        plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
        plt.show()


if __name__ == '__main__':
    bst = binarySearchTree()
    for _ in range(100):
        bst.insert(np.random.randint(100))

    gnx = bst.draw()
