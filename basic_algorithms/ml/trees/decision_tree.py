import functools
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Any

import numpy as np
from basic_algorithms.ml.trees.tree_utils import split_continuous_feature


@dataclass(order=False)
class LabelNode:
    """
    Decision tree leaf
    """
    label: Any = field(compare=False)
    type: str = field(default='label', compare=False)


@dataclass(order=False)
class SplitNode:
    """
    Decision tree node
    """
    feature: int = field(compare=False)
    gain: float = field(compare=False)
    current_label: Any = field(compare=False)
    children: dict = field(default_factory=dict, compare=False)
    type: str = field(default='split', compare=False)


class DecisionTree:
    def __init__(self, discrete_features: List = None, criterion: str = 'gini', n_split: int = 10,
                 split_method: str = 'quantile', max_depth: int = 20, min_impurity_decrease: float = 0.0):
        """
        Decision tree classifier based on C4.5 algorithm
        :param discrete_features: list of indices of discrete features
        :param criterion: gini | entropy
        :param n_split: number of intervals for continuous features
        :param split_method: quantile | mean | random - method to split continuous features
        :param max_depth: maximum depth of the tree
        :param min_impurity_decrease: minimum impurity decrease to split a node
        """

        # tree-alg parameters
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_impurity_decrease = min_impurity_decrease

        # discretization parameters for continuous features
        self._split_method = split_method
        self._n_split = n_split

        # features by type
        self._discrete_features = [] if discrete_features is None else discrete_features
        self._continuous_splits = None

        # tree structure attributes
        self._root = None
        self._classes = None

    def _bin_data(self, data):
        discrete_data = data[:, self._discrete_features]
        continuous_data = data[:, [i for i in range(data.shape[1]) if i not in self._discrete_features]]
        # fit continuous splits if not fitted
        if self._continuous_splits is None:
            self._continuous_splits = split_continuous_feature(continuous_data, self._n_split, method=self._split_method)
        # bin continuous data
        continuous_data = np.hstack([np.digitize(continuous_data[:, i], self._continuous_splits[:, i])[:, np.newaxis]
                                     for i in range(continuous_data.shape[1])]).astype(np.uint8)
        return np.hstack([discrete_data, continuous_data])

    def _calculate_entropy(self, y):
        """
        Calculate entropy of a dataset
        :param y: array of labels
        :return: entropy
        """
        distribution = dict(map(lambda item: (item[0], item[1] / len(y)), list(Counter(y).items())))
        entropy = 0
        for val, prob in distribution.items():
            entropy -= prob * np.log2(prob)
        return entropy

    def _calculate_gini(self, y):
        """
        Calculate gini of a dataset
        :param y: array of labels
        :return: gini
        """
        distribution = dict(map(lambda item: (item[0], item[1] / len(y)), list(Counter(y).items())))
        gini = 0
        for val, prob in distribution.items():
            gini += prob * (1 - prob)
        return gini

    def _get_criterion(self):
        """
        :return: criterion function according to defined in constructor
        """
        options = {
            'gini': self._calculate_gini,
            'entropy': self._calculate_entropy
        }
        assert self._criterion in options.keys(), \
            f"Invalid criterion {self._criterion}, expected one of {options.keys()}"

        return options[self._criterion]

    def _calculate_gain(self, X, y):
        """
        Calculate gain of a split - the gain is given for any feature and is calculated according:
            gain = H(S) - sum(p(Sv)H(Sv))
        where:
            H(S) is the entropy of the dataset
            Sv is a subset of S according to a specific value (v) of a feature
            p(Sv) is the probability of Sv in S - p(Sv) = |Sv| / |S|

        :param X: array of features
        :param y: array of labels
        :param feature: feature index
        :return: gain
        """
        criterion = self._get_criterion()

        # get idx of discrete features -> p(Sv) = |Sv| / |S|
        get_distribution = lambda column: dict(map(lambda item: (item[0], item[1] / X.shape[0]), list(Counter(column).items())))
        features_distribution = [get_distribution(X[:, i]) for i in range(X.shape[1])]
        # get H(S)
        prior_entropy = criterion(y)

        # calculate for each feature p(Sv)H(Sv) and sum
        all_gains = np.zeros(X.shape[1])
        for col in range(X.shape[1]):
            posterior_entropy = 0
            for val, prob in features_distribution[col].items():
                sub_group = y[X[:, col] == val]
                # p(Sv)H(Sv)
                posterior_entropy += prob * criterion(sub_group)

            # gain = H(S) - sum(p(Sv)H(Sv))
            feature_gain = prior_entropy - posterior_entropy
            all_gains[col] = feature_gain
        return all_gains

    def _build_tree(self, X, y, depth=0, features_not_used=None):
        """
        Build the decision tree recursively using C4.5 algorithm
        :param X: array of features - (Nxd)
        :param y: array of labels - (N,)
        :param depth: current depth of the tree (used for stopping condition)
        :param features_not_used: available features to split by
        :return: current node (split or leaf)
        """

        # use all features if not specified
        features_not_used = list(range(X.shape[1])) if features_not_used is None else features_not_used

        label_count = Counter(y)
        most_common_label = label_count.most_common(1)[0][0]

        # STOP condition - only one class in the node || max depth reached || no more features to split by
        if len(label_count) == 1 or depth >= self._max_depth or len(features_not_used) == 0:
            return LabelNode(label=most_common_label)

        # get gain for each feature {feature-index: gain earned by splitting by feature}
        all_gains = self._calculate_gain(X[:, features_not_used], y)
        argmax_gain = np.argmax(all_gains)
        best_feature = features_not_used[argmax_gain]
        max_gain = all_gains[argmax_gain].item()
        features_not_used = list(filter(lambda f: f != best_feature, features_not_used))

        # STOP condition - gain is too small
        if max_gain < self._min_impurity_decrease:
            return LabelNode(label=most_common_label)

        # get best feature to split by, create node and split recursively
        node = SplitNode(
            feature=best_feature,
            gain=max_gain,
            current_label=most_common_label
        )

        # split by best feature possible values
        for val in np.unique(X[:, best_feature]):
            val_indices = np.where(X[:, best_feature] == val)

            # STOP condition - no more samples to split by
            if val_indices[0].shape[0] == 0:
                node.children[val] = LabelNode(label=most_common_label)
                continue

            # split recursively
            node.children[val] = self._build_tree(X=X[val_indices], y=y[val_indices],
                                                  depth=depth+1,
                                                  features_not_used=features_not_used)

        return node

    def fit(self, X, y):
        """
        Fit the decision tree classifier
        :param X: array of features
        :param y: array of labels
        """
        assert X.ndim == 2 and y.ndim == 1, "Invalid data shape, expected (n_samples, n_features) and (n_samples,)"
        assert X.shape[0] == y.shape[0], "Invalid data shape, expected (n_samples, n_features) and (n_samples,)"

        X = self._bin_data(X)
        self._classes = np.unique(y)
        self._root = self._build_tree(X, y)

    def _predict(self, x):
        """
        Predict label for a given sample
        :param x: sample
        :return: predicted label
        """
        node = self._root
        while node.type == 'split':
            node = node.children.get(x[node.feature], LabelNode(label=node.current_label))
        return node.label

    def predict(self, X):
        """
        Predict labels for a given array of features
        :param X: array of features
        :return: array of predicted labels
        """
        X = self._bin_data(X)
        return np.array([self._predict(x) for x in X])


if __name__ == '__main__':
    from basic_algorithms.ml.data.data_loader import load_diabetes
    X, Y = load_diabetes()
    X[np.where(X[:, 4] == 0), 4] = X[np.where(X[:, 4] != 0), 4].mean()

    clf = DecisionTree(discrete_features=[1, 3], criterion='entropy', max_depth=10, min_impurity_decrease=0.1)
    N = 350
    clf.fit(X[:N], Y[:N])

    pred = clf.predict(X[N:])
    y_true = Y[N:]

    tp = 0
    for y, y_hat in zip(y_true, pred):
        if y == y_hat:
            tp += 1
    accuracy = tp / len(y_true)
    print(f'Accuracy: {accuracy}')
    e = 0