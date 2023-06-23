import numpy as np

"""
0-1 Knapsack Problem
====================

In the 0–1 Knapsack problem, we are given a set of items, each with a weight and a value,
and we need to determine the number of each item to include in a collection so that the
total weight is less than or equal to a given limit and the total value is as large as possible.
"""


def knapsack(weights: list, values: list, capacity: int, item_names: list = None):
    """
    Dynamic-Programing based solution do the 0-1 Knapsack problem.

    KS is a matrix of dimensions (n_items + 1) x (capacity + 1)
    KS[i, j] = max value of items 0..i-1 with capacity limit of j
    to calculate KS[i, j] we have two options:
        1. don't add item i-1 to the knapsack ->
           the maximum value is the maximum value without i but with the same capacity limit (KS[i-1, j])
        2. add item i-1 to the knapsack ->
           the maximum value is the value of item i-1 + the maximum value of items 0..i-1 with capacity limit of j-w
    KS[w, v][0..j] = max
            k[i-1, j] (don't add item j),
            KS[i-1, j-w[j]] + w[j] (add item j)

    :param weights: list of weights of items
    :param values: list of values of items
    :param capacity: capacity of knapsack
    :param item_names: (Optional) list of names of items
    """
    assert len(weights) == len(values), "weights and values must have the same length"
    assert item_names is None or len(item_names) == len(weights), "item_names must have the same length as weights"
    assert isinstance(capacity, int), "capacity must be an integer"
    assert np.all([isinstance(w, int) for w in weights]), "weights must be integers"

    n_items = len(weights)
    knapsack_values = np.zeros((n_items + 1, capacity + 1))
    knapsack_set = [[set() for _ in range(capacity + 1)] for _ in range(n_items + 1)]

    # initialize first row and column to 0

    # zero capacity -> no items
    for i in range(n_items + 1):
        knapsack_values[i, 0] = 0
    # no items -> no gain
    for j in range(capacity + 1):
        knapsack_values[0, j] = 0

    for item in range(1, n_items + 1):
        # correct item index in weights and values arrays -> KS has extra row and column for 0 items and 0 capacity
        item_idx = item - 1

        for weight_limit in range(1, capacity + 1):
            # don't add item -> optimal value equals to gain without item with same capacity limit
            gain_wo_item = knapsack_values[item - 1, weight_limit]

            # add item -> optimal value equals to gain of item + gain w/o item with  limit of capacity - items weight
            if weight_limit - weights[item_idx] < 0:
                # item doesn't fit in knapsack with current weight limit
                gain_w_item = -1
            else:
                gain_w_item = values[item_idx] + knapsack_values[item - 1, weight_limit - weights[item_idx]]

            # Choose a solution with fewer items | >: prefer to add item if possible
            if gain_wo_item >= gain_w_item:
                # don't add item
                knapsack_values[item, weight_limit] = gain_wo_item
                knapsack_set[item][weight_limit] = knapsack_set[item - 1][weight_limit].copy()
            else:
                # add item
                knapsack_values[item, weight_limit] = gain_w_item
                knapsack_set[item][weight_limit] = knapsack_set[item - 1][weight_limit - weights[item_idx]].copy()
                knapsack_set[item][weight_limit].add(item_idx)

    # collect optimal solution
    optimal_gain = knapsack_values[n_items, capacity]
    optimal_item_set = list(knapsack_set[n_items][capacity])
    if item_names is not None:
        optimal_item_set = [item_names[i] for i in optimal_item_set]

    return optimal_gain, optimal_item_set
