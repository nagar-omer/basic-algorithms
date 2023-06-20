from basic_algorithms.graph.floyd_warshall import floyd_warshall
import numpy as np
import  networkx as nx


def test_shortest_path():
    gnx = nx.DiGraph()
    gnx.add_edge(0, 1, weight=1)
    gnx.add_edge(0, 2, weight=2)
    gnx.add_edge(1, 3, weight=3)
    gnx.add_edge(2, 3, weight=4)
    gnx.add_edge(3, 4, weight=5)
    gnx.add_edge(4, 0, weight=6)

    dist, path, node_index = floyd_warshall(gnx=gnx, weight='weight')

    for src_index, src in enumerate(node_index):
        for dst_index, dst in enumerate(node_index):
            p = nx.bellman_ford_path(gnx, src_index, dst_index)
            assert p == path[src_index][dst_index]
