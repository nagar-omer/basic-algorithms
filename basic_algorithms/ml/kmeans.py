import numpy as np


class KMeans:
    def __init__(self, n_clusters: int = 5, max_iter: int = 100, tol: float = 1e-4):
        """
        KMeans clustering algorithm
        :param n_clusters: number of clusters
        :param max_iter: maximum number of iterations
        :param tol: tolerance for convergence
        """
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._centroids = None
        self._labels = None

    def _init_centroids(self, X: np.ndarray):
        """
        Initialize centroids according to the kmeans++ algorithm
        :param X: data
        :return: None
        """
        self._centroids = X[np.random.randint(X.shape[0])][np.newaxis, :]
        # pruning for better performance
        candidates = X[np.random.choice(np.arange(X.shape[0]), size=min(1000, X.shape[0]), replace=False)]
        for _ in range(self._n_clusters - 1):
            distances = np.hstack([np.linalg.norm(candidates - centroid, axis=1)[:, np.newaxis]
                                   for centroid in self._centroids]).min(axis=1)
            distances /= np.sum(distances)
            self._centroids = np.vstack([self._centroids,
                                         candidates[np.random.choice(np.arange(candidates.shape[0]), p=distances)]])

    def _estimate_labels(self, X: np.ndarray):
        """
        Estimate labels for each row in X
        :param X: data
        :return: labels
        """
        self._labels = np.argmin(np.hstack([np.linalg.norm(X - centroid, axis=1)[:, np.newaxis]
                                            for centroid in self._centroids]), axis=1)

    def _maximize_centroids(self, X: np.ndarray):
        """
        Maximize centroids according to the kmeans algorithm
        :param X: data
        :return: None
        """
        for i in range(self._n_clusters):
            self._centroids[i] = np.mean(X[self._labels == i], axis=0)

    def fit(self, X):
        """
        Fit the model to the data
        :return:
        """
        self._init_centroids(X)
        # iterate max_iter times
        for i in range(self._max_iter):
            centroids_before = self._centroids.copy()
            self._estimate_labels(X)
            self._maximize_centroids(X)
            # check for convergence
            if np.linalg.norm(self._centroids - centroids_before, axis=1).mean() < self._tol:
                break
        return self

    def predict(self, X: np.ndarray):
        """
        Predict labels for X
        :param X: data
        :return: labels
        """
        return np.argmin(np.hstack([np.linalg.norm(X - centroid, axis=1)[:, np.newaxis]
                                    for centroid in self._centroids]), axis=1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    X = np.random.rand(100, 2)
    mean1, cov1 = [0, 0], [[1, 0], [0, 1]]
    mean2, cov2 = [10, 10], [[1, 0], [0, 1]]
    mean3, cov3 = [5, 5], [[2, 0], [0, 2]]
    mean4, cov4 = [2, 2], [[0.3, 0], [0, 0.3]]
    X = np.vstack([np.random.multivariate_normal(mean1, cov1, 200),
                   np.random.multivariate_normal(mean2, cov2, 200),
                   np.random.multivariate_normal(mean3, cov3, 200),
                   np.random.multivariate_normal(mean4, cov4, 200)])
    kmeans = KMeans(n_clusters=4).fit(X)
    pred = kmeans.predict(X)
    plt.scatter(*X[pred == 0].T.tolist(), color='red')
    plt.scatter(*X[pred == 1].T.tolist(), color='blue')
    plt.scatter(*X[pred == 2].T.tolist(), color='green')
    plt.scatter(*X[pred == 3].T.tolist(), color='orange')
    plt.show()
    e = 0
