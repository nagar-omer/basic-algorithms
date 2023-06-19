from dataclasses import dataclass, field
from heapq import heappush, heappop
import networkx as nx
import numpy as np

DIST_TO_SOURCE = 'dist'
SOURCE_NODE = 'source'
SHORTEST_PATH = 'path'


@dataclass(order=True)
class QueueNodeItem:
    """
    Helper class for priority queue
    """
    dist: int
    node_id: int = field(compare=False)


def dijkstra(gnx: nx.Graph, source, weight) -> dict:
    """
    Find shortest path from source to all other nodes in the graph
    This version is implemented using priority queue
    NOTE: graph is assumed to be DAG (directed acyclic graph)
    complexity: O(|E|log|V| + |V|log|V|) can be reduced to O(|E| + |V|log|V|) using Fibonacci heap

    :param gnx: graph object
    :param source: source node
    :param weight: weight key in edge data
    :return: dictionary of distances from source to all other nodes
    """
    gnx = gnx if gnx.is_directed() else gnx.to_directed()

    # initialize graph distances
    for node_id, data in gnx.nodes(data=True):
        data[DIST_TO_SOURCE] = np.inf if node_id != source else 0
        data[SHORTEST_PATH] = [0] if node_id == source else []

    # init Q and visited
    queue, visited = [QueueNodeItem(node_id=source, dist=0)], set()

    while len(queue) != 0:
        # pop and check not visited yet
        closest_node = heappop(queue)
        if closest_node.node_id in visited:
            continue

        # mark node as visited ->
        # queue may contain duplicate nodes, but only the closest (with respect to chosen path) is considered
        visited.add(closest_node.node_id)
        for neighbor in gnx.successors(closest_node.node_id):
            # check if distance from source to v is shorter when going through u
            alternative_dist = closest_node.dist + gnx.edges[closest_node.node_id, neighbor][weight]
            if alternative_dist < gnx.nodes[neighbor][DIST_TO_SOURCE]:
                # push to queue and update distance in graph database
                heappush(queue, QueueNodeItem(node_id=neighbor, dist=alternative_dist))
                gnx.nodes[neighbor][DIST_TO_SOURCE] = alternative_dist
                gnx.nodes[neighbor][SHORTEST_PATH] = gnx.nodes[closest_node.node_id][SHORTEST_PATH] + [neighbor]

    # return dictionary of distances
    dist_dict = {node_id: (data[SHORTEST_PATH], data[DIST_TO_SOURCE]) for node_id, data in gnx.nodes(data=True)}
    return dist_dict
