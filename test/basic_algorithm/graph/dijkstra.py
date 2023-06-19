from basic_algorithms.graph.dijkstra import dijkstra
import numpy as np
import networkx as nx


def test_shortest_path():
    gnx = nx.DiGraph()
    gnx.add_edge(0, 1, weight=1)
    gnx.add_edge(0, 2, weight=2)
    gnx.add_edge(1, 3, weight=3)
    gnx.add_edge(2, 3, weight=4)
    gnx.add_edge(3, 4, weight=5)
    gnx.add_edge(4, 0, weight=6)

    dist_dict = dijkstra(gnx=gnx, source=0, weight='weight')
    assert dist_dict[0][1] == 0
    assert dist_dict[1][1] == 1
    assert dist_dict[2][1] == 2
    assert dist_dict[3][1] == 4
    assert dist_dict[4][1] == 9

    assert nx.bellman_ford_path(gnx, 0, 0) == dist_dict[0][0]
    assert nx.bellman_ford_path(gnx, 0, 1) == dist_dict[1][0]
    assert nx.bellman_ford_path(gnx, 0, 2) == dist_dict[2][0]
    assert nx.bellman_ford_path(gnx, 0, 3) == dist_dict[3][0]
    assert nx.bellman_ford_path(gnx, 0, 4) == dist_dict[4][0]
