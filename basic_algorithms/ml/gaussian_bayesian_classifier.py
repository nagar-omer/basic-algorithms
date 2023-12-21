import functools

import numpy as np
from scipy.stats import norm


class NaiveBayesianGaussianClassifier:
    def __init__(self, discrete_features=None):
        self._classes = None
        self._means = None
        self._variances = None
        self._priors = None
        self._discrete_features = [] if discrete_features is None else discrete_features
        self._d = None

    def calculate_discrete_likelihood(self, discrete_features):
        """
        Calculate likelihood of discrete features
        :param X: array of discrete features (assuming categorical)
        :return: probability of each discrete feature
        """
        discrete_likelihood = {}
        for column in range(discrete_features.shape[1]):
            feature_col = discrete_features[:, column]
            feature_values = np.unique(feature_col)
            n_samples = len(feature_col)
            discrete_likelihood[column] = {v: len(feature_col[feature_col == v]) / n_samples for v in feature_values}
        return discrete_likelihood

    def fit(self, X, y):
        self._d = X.shape[1]
        discrete_features = X[:, self._discrete_features]
        continuous_features = X[:, [i for i in range(X.shape[1]) if i not in self._discrete_features]]

        # prior distribution of each class
        self._classes = list(np.unique(y))
        self._priors = {c: len(y[y == c]) / len(y) for c in self._classes}

        # calculate mean and covariance for continuous features - assume gaussian distribution
        self._means = {c: np.mean(continuous_features[np.where(y == c)], axis=0) for c in self._classes}
        self._variances = {c: np.var(continuous_features[np.where(y == c)], axis=0) for c in self._classes}

        # count discrete features
        self._discrete_likelihood = {c: self.calculate_discrete_likelihood(discrete_features[np.where(y == c)]) for c in self._classes}

    def _calculate_posterior(self, x, c):
        """
        Calculate posterior of a sample
        :param x: sample
        :param c: class
        :return: posterior
        """

        assert x.ndim == 2 and x.shape[1] == self._d, \
            "Incorrect sample shape, expected (n_samples, n_features)"
        assert c in self._classes, f"Invalid class {c}, expected one of {self._classes}"

        # calculate likelihood of continuous features
        continuous_features = x[:, [i for i in range(x.shape[1]) if i not in self._discrete_features]]
        discrete_features = x[:, self._discrete_features]

        continuous_likelihood = np.ones_like(continuous_features) if continuous_features.shape[1] > 0 else (
            np.ones((continuous_features.shape[0], 1)))
        for i in range(continuous_features.shape[1]):
            continuous_likelihood[:, i] = norm.pdf(continuous_features[:, i], self._means[c][i], np.sqrt(self._variances[c][i]))

        # c - class | i - feature index | v - feature value
        discrete_likelihood = np.ones_like(discrete_features) if discrete_features.shape[1] > 0 else (
            np.ones((discrete_features.shape[0], 1)))
        for i in range(discrete_features.shape[1]):
            discrete_likelihood[:, i] = [self._discrete_likelihood.get(c, {}).get(i, {}).get(v, 0)
                                         for v in discrete_features[:, i]]

        likelihood = np.multiply(np.cumprod(discrete_likelihood, axis=0)[:, -1],
                                 np.cumprod(continuous_likelihood, axis=0)[:, -1])
        return likelihood * self._priors[c]

    def predict(self, X):
        assert (X.ndim == 1 and X.shape[0] == self._d) or (X.ndim == 2 and X.shape[1] == self._d), \
            f"Incorrect sample shape: {X.shape}, expected {(self._d,)} or ({X.shape[0]}, {self._d})"

        if X.ndim == 1:
            X = X[np.newaxis, :]

        # calculate posterior for each class & return class with highest posterior
        posteriors = {c: self._calculate_posterior(X, c) for c in self._classes}
        pred_index = np.vstack([posteriors[c] for c in self._classes]).T.argmax(axis=1)
        pred = np.asarray(list(map(lambda i: self._classes[i], pred_index)))
        return pred


if __name__ == '__main__':
    from basic_algorithms.ml.data.data_loader import load_diabetes
    X, Y = load_diabetes()
    X[np.where(X[:, 4] == 0), 4] = X[np.where(X[:, 4] != 0), 4].mean()

    clf = NaiveBayesianGaussianClassifier(discrete_features=[1, 3])
    clf.fit(X, Y)
    pred = clf.predict(X)

    tp = 0
    for i in range(len(Y)):
        if Y[i] == pred[i]:
            tp += 1
    accuracy = tp / len(Y)
    print(f'Accuracy: {accuracy}')
    e = 0
