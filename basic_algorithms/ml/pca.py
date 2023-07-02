import numpy as np


class PCA:
    def __init__(self, n_components, bias=False):
        self.n_components = n_components
        self._mat_change_basis = None
        self._bias = bias
        self._mu  =  None
        self._std = None

    def _standardize_fit(self, samples):
        """
        Standardize a set of samples according to the formula (Z-score):
        X = (X - mu) / std
        :param samples: features of shape (n_samples, n_features)
        :return: standardized features of shape (n_samples, n_features)
        """
        self._mu = samples.mean(axis=0)
        self._std = samples.std(axis=0)
        return (samples - self._mu) / self._std

    def _standardize_transform(self, samples):
        """
        Standardize a set of samples according to the formula (Z-score):
        X = (X - mu) / std
        :param samples: features of shape (n_samples, n_features)
        :return: standardized features of shape (n_samples, n_features)
        """
        assert self._mu is not None and self._std is not None, "Standartization failed - Must fit before transform"
        return (samples - self._mu) / self._std

    def _covariance(self, samples):
        """
        Compute the covariance matrix of a set of samples according to the formula:
        cov(X, Y) = E[XY] - E[X]E[Y]
        :param samples: samples of shape (n_samples, n_features)
        :return: covariance matrix of shape (n_features, n_features)
        """
        Exy = np.dot(samples.T, samples) / samples.shape[0]
        Ex = np.expand_dims(samples.mean(axis=0), axis=0)
        ExEy = np.dot(Ex.T, Ex)
        cov = Exy - ExEy
        return cov

    def fit(self, X):
        """
        Fit the PCA model according to the given training data.
        :param X: samples of shape (n_samples, n_features)
        """
        X = self._standardize_fit(X)
        cov = self._covariance(X)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvalues, eigenvectors = zip(*[(eigenvalues[i], eigenvectors[i]) for i in
                                          eigenvalues.argsort()[::-1][:self.n_components]])
        self._mat_change_basis = np.vstack(eigenvectors).T

    def transform(self, X):
        """
        Apply dimensionality reduction to X according to the fitted PCA model.
        :param X: samples of shape (n_samples, n_features)
        :return: samples of shape (n_samples, n_components)
        """
        X = self._standardize_transform(X)
        return np.dot(X, self._mat_change_basis)

    def fit_transform(self, X):
        """
        Fit the PCA model according to the given training data, and apply dimensionality reduction to X.
        :param X: samples of shape (n_samples, n_features)
        :return: samples of shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)
