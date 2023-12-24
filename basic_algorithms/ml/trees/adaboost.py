from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from basic_algorithms.ml.trees.tree_utils import split_continuous_feature
import random


@dataclass
class Stump:
    """
    Stump classifier
    """
    feature: int = field(compare=False)
    classification: dict = field(default_factory=dict, compare=False)
    amount_of_say: float = field(default=0.0, compare=True)
    default_prediction: int = field(default=0, compare=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class for each row in X
        :param X: data to predict
        :return: predictions
        """
        return np.array([self.classification.get(x[self.feature], self.default_prediction) for x in X])


class AdaBoost:
    def __init__(self, n_estimators: int = 1, discrete_features: list = None,
                 n_split: int = 10, split_method: str = 'quantile'):
        """
        AdaBoost classifier
        :param n_estimators: number of estimators
        :param max_depth: maximum depth of each estimator
        """

        # tree-alg parameters
        self._n_estimators = n_estimators
        self._stumps = list()

        # discretization parameters for continuous features
        self._discrete_features = [] if discrete_features is None else discrete_features
        self._continuous_splits = None
        self._split_method = split_method
        self._n_split = n_split

    def _bin_data(self, data: np.ndarray):
        discrete_data = data[:, self._discrete_features]
        continuous_data = data[:, [i for i in range(data.shape[1]) if i not in self._discrete_features]]
        # fit continuous splits if not fitted
        if self._continuous_splits is None:
            self._continuous_splits = split_continuous_feature(continuous_data, self._n_split, method=self._split_method)
        # bin continuous data
        continuous_data = np.hstack([np.digitize(continuous_data[:, i], self._continuous_splits[:, i])[:, np.newaxis]
                                     for i in range(continuous_data.shape[1])]).astype(np.uint8)
        return np.hstack([discrete_data, continuous_data])

    def _sample_data(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # if all samples has equal weight
        if np.unique(weights).shape[0] == 1:
            return np.arange(X.shape[0])

        return np.asarray(random.choices(population=np.arange(X.shape[0]), weights=weights, k=X.shape[0]))

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


    def _get_gini(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
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

        # get idx of discrete features -> p(Sv) = |Sv| / |S|
        get_distribution = lambda column: dict(
            map(lambda item: (item[0], item[1] / X.shape[0]), list(Counter(column).items())))
        features_distribution = [get_distribution(X[:, i]) for i in range(X.shape[1])]
        # get H(S)
        prior_entropy = self._calculate_gini(y)

        # calculate for each feature p(Sv)H(Sv) and sum
        all_gains = np.zeros(X.shape[1])
        for col in range(X.shape[1]):
            posterior_entropy = 0
            for val, prob in features_distribution[col].items():
                sub_group = y[X[:, col] == val]
                # p(Sv)H(Sv)
                posterior_entropy += prob * self._calculate_gini(sub_group)

            # gain = H(S) - sum(p(Sv)H(Sv))
            feature_gain = prior_entropy - posterior_entropy
            all_gains[col] = feature_gain
        return all_gains


    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2, 'X must be a 2D array'
        assert y.ndim == 1, 'y must be a 1D array'
        assert X.shape[0] == y.shape[0], 'X and y must have the same number of rows'

        # initialize weights
        weights = np.ones_like(y) / y.shape[0]
        for i in range(self._n_estimators):
            # 1. sample data according to weights (not in the first iteration)
            data_indices = self._sample_data(X, weights)
            Xi, yi, wi = X[data_indices], y[data_indices], weights[data_indices]

            # 2. fit a new stump according to the best gini index
            all_features_gini = self._get_gini(Xi, yi)
            best_feature = np.argmin(all_features_gini)
            best_gini = all_features_gini[best_feature]
            majority_class = Counter(yi).most_common(1)[0][0]

            new_stump = Stump(feature=best_feature,
                              # majority class for each value of the feature
                              classification={val: Counter(yi[Xi[:, best_feature] == val]).most_common()[0][0]
                                              for val in np.unique(Xi[:, best_feature])},
                              # default prediction for values not in the stump
                              default_prediction=majority_class,
                              # assigned after checking error rate
                              amount_of_say=0.0)

            # get error rate
            pred_i = new_stump.predict(Xi)
            error_rate = np.sum(wi[yi != pred_i])
            assert error_rate <= 1, "Error cannot be larger than 1"

            # add stump
            amount_of_say = 0.5 * np.log((1 - error_rate) / error_rate)
            new_stump.amount_of_say = amount_of_say
            self._stumps.append(new_stump)

            # 4. update weights
            predictions = new_stump.predict(Xi)
            correct = np.unique(data_indices[predictions == yi])
            incorrect = np.unique(data_indices[predictions != yi])

            weights[correct] *= np.exp(-amount_of_say)
            weights[incorrect] *= np.exp(amount_of_say)

            # 5. normalization so it will sun to 1
            weights /= weights.sum()

    def predict(self, X: np.ndarray):
        # get prediction from all stumps
        candidates = np.hstack([stump.predict(X)[:, np.newaxis] for stump in self._stumps])

        # for each sample get argmax sum(Weight(y_hat))
        pred = np.zeros(X.shape[0])
        for sample_i, c in enumerate(candidates):
            weights = {}
            for stump_i, label in enumerate(c):
                weights[label] = weights.get(label, 0) + self._stumps[stump_i].amount_of_say
            pred[sample_i] = max(weights.keys(), key=weights.get)
        return pred

