import numpy as np

"""
0-1 Knapsack Problem
====================

In the 0–1 Knapsack problem, we are given a set of items, each with a weight and a value,
and we need to determine the number of each item to include in a collection so that the
total weight is less than or equal to a given limit and the total value is as large as possible.
"""


def knapsack(weights, values, limit, item_names=None):
    """
    Dynamic-Programing based solution do the 0-1 Knapsack problem.

    KS will be filled only on its upper diagonal level.
    KS[w, v][0..j] = max
            k[w, v][0..j-1](don't add),
            KS[w, v][0..j-1] + v (add W + w <= limit) otherwise 0
    """
    assert len(weights) == len(values), "weights and values must have the same length"
    n_items = len(weights)
    knapsack_weight = np.zeros((n_items + 1, n_items + 1))
    knapsack_val = np.zeros((n_items + 1, n_items + 1))

    for item in range(1, n_items):
        for sol in range(item, n_items):
            sol_weight = knapsack_weight[] weights[item]
