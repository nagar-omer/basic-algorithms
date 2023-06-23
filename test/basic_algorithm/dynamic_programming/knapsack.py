from basic_algorithms.dynamic_programming.knapsack import knapsack


def test_knapsack():
    item_names = ['BEST_1', 'NA', 'NA', 'BEST_2', 'NA', 'NA']
    value = [20, 5, 10, 40, 15, 25]
    weight = [1, 2, 3, 8, 7, 4]
    W = 10
    optimal_gain, optimal_set = knapsack(weight, value, W, item_names=item_names)
    assert optimal_gain == 60
    assert optimal_set == ['BEST_1', 'BEST_2']


def test_empty_item_list():
    value = []
    weight = []
    W = 10
    optimal_gain, optimal_set = knapsack(weight, value, W)
    assert optimal_gain == 0
    assert optimal_set == []


def test_no_solution():
    value = [10, 20, 30]
    weight = [10, 20, 30]
    W = 5
    optimal_gain, optimal_set = knapsack(weight, value, W)
    assert optimal_gain == 0
    assert optimal_set == []

