from typing import Tuple
import networkx as nx
import numpy as np
DIST_TO_SOURCE = 'dist'
SOURCE_NODE = 'source'
SHORTEST_PATH = 'path'


def floyd_warshall(gnx: nx.Graph, weight='weight') -> Tuple[np.ndarray, list, list]:
    # map node ids to 0-N indices
    idx_to_node = list(gnx.nodes())
    node_idx = {node_id: i for i, node_id in enumerate(idx_to_node)}

    # create distance matrix
    dist = np.ones((gnx.number_of_nodes(), gnx.number_of_nodes())) * np.inf
    path = [[[src, dst] if gnx.has_edge(idx_to_node[src], idx_to_node[dst]) else ([src] if src == dst else [])
             for dst in range(gnx.number_of_nodes())] for src in range(gnx.number_of_nodes())]

    # first step fill single edge path / 0 for u->u path
    for src, dst, data in gnx.edges(data=True):
        dist[node_idx[src], node_idx[dst]] = data.get(weight, 1)

    for u in gnx.nodes():
        dist[node_idx[u], node_idx[u]] = 0

    # longest path possible goes over all vertices -> hence after |V| steps were done
    for k in range(gnx.number_of_nodes()):
        # improve path / find new path
        for src in range(gnx.number_of_nodes()):
            for dst in range(gnx.number_of_nodes()):
                # recursive rule
                # the shortest path (from src to dst) using 0-k nodes equals to min of the following two:
                #   1. sortest path using 0-[k-1] nodes
                #   2. shortest path (from src to k) using 0-[k-1] nodes +
                #      shortest path (from k to dst) using 0-[k-1] nodes
                dist_without_k = dist[src, dst]
                dist_with_k = dist[src, k] + dist[k, dst]
                if dist_with_k < dist_without_k:
                    dist[src, dst] = dist_with_k
                    path[src][dst] = path[src][k][:-1] + path[k][dst]

    return dist, path, idx_to_node




