from typing import Tuple
import networkx as nx
import numpy as np

DIST_TO_SOURCE = 'dist'
SOURCE_NODE = 'source'
SHORTEST_PATH = 'path'


def _relax(gnx: nx.DiGraph, u: int, v: int, weight_col: str) -> bool:
    """
    Relaxation step in Belman-Ford algorithm
    check that the distance from source to v is shorter when going through u than the current distance
    if so, update the distance and source node
    :param gnx: graph object
    :param u: source node
    :param v: target node
    :param weight_col: weight key in edge data
    :return: True if distance was updated, False otherwise
    """

    # get distances from source
    u_dist, v_dist = gnx.nodes[u][DIST_TO_SOURCE], gnx.nodes[v][DIST_TO_SOURCE]
    weight = gnx.edges[u, v].get(weight_col, 1)
    # check if distance from source to v is shorter when going through u
    if u_dist + weight < v_dist:
        gnx.nodes[v][DIST_TO_SOURCE] = u_dist + weight
        gnx.nodes[v][SOURCE_NODE] = u
        gnx.nodes[v][SHORTEST_PATH] = gnx.nodes[u][SHORTEST_PATH] + [v]
        return True
    return False


def _edge_iter(gnx: nx.DiGraph, source: int):
    """
    Define iteration order for edges through all steps of Belman-Ford algorithm
    This function follows BFS to reduce the number of iterations at each step to reachable nodes only
    :param gnx: graph object
    :param source: source node
    :return: incremental list of edges to iterate over
    """

    iter_list, prev_level, next_level = set(), [source], []
    # do |V|-1 times
    for _ in range(len(gnx.nodes) - 1):
        # iterate over all nodes that are reachable in K steps
        for node_id in prev_level:
            # add nodes that are reachable in K+1 steps
            successors = list(gnx.successors(node_id))
            next_level.extend(successors)
            iter_list = iter_list.union([(node_id, dst) for dst in successors])
        prev_level = next_level
        next_level = []
        yield iter_list


def _find_shortest_path(gnx: nx.DiGraph, source: int, weight_col: str):
    """
    Find shortest path from source to all other nodes in the graph
    :param gnx: graph object
    :param source: source node
    :param weight_col: weight key in edge data
    """
    # do |V|-1 times
    for edge_list in _edge_iter(gnx, source):
        for u, v in edge_list:
            _relax(gnx=gnx, u=u, v=v, weight_col=weight_col)


def _is_negative_cycles(gnx: nx.DiGraph, weight_col: str):
    """
    Check if there are negative cycles in the graph
    NOTE: this function assumes that the graph has been relaxed |V|-1 times
    :param gnx: graph object
    :param weight_col: weight key in edge data
    :return:
    """
    for u, v, data in gnx.edges(data=True):
        if gnx.nodes[u][DIST_TO_SOURCE] + data.get(weight_col, 1) < gnx.nodes[v][DIST_TO_SOURCE]:
            return True
    return False


def belman_ford(gnx: nx.Graph, source: int, weight: str = 'weight') -> Tuple[dict, bool]:
    """
    Belman-Ford algorithm for finding shortest paths in a graph with negative edges
    this a single source shortest path algorithm
    O(|V|*|E|)
    :return:  dictionary with shortest paths from source to all other nodes
    """
    # convert to directed graph
    gnx = gnx if gnx.is_directed() else gnx.to_directed()

    # initialize graph distances
    for node_id, data in gnx.nodes(data=True):
        data[DIST_TO_SOURCE] = np.inf if node_id != source else 0
        data[SHORTEST_PATH] = [0]

    # find shortest paths and detect negative cycles
    _find_shortest_path(gnx=gnx, source=source, weight_col=weight)
    is_neg = _is_negative_cycles(gnx=gnx, weight_col=weight)

    # convert distances to dictionary
    dist_dict = {node_id: (data[SHORTEST_PATH], data[DIST_TO_SOURCE]) for node_id, data in gnx.nodes(data=True)}
    return dist_dict, is_neg
