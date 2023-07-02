import numpy as np
from tqdm.auto import tqdm


class BinarySVMClassifier:
    """
    Binary SVM classifier.
    """
    def __init__(self, max_iter: int = 10000, tol: float = 1e-4, reg: float = 1e-2,
                 margin: float = 1.0, lr: float = 1e-6):
        """
        Binary SVM classifier
        :param max_iter: max number of iterations
        :param tol: tolerance for stopping criteria
        :param reg: regularization parameter (lambda)
        :param margin: margin parameter (alpha)
        :param lr: learning rate
        """
        self._weights = None
        self._margin = margin
        self._max_iter = max_iter
        self._lambda = reg
        self._lr = lr
        self._tol = tol
        self.pbar = None

    def _init_weights(self, shape: tuple):
        """
        Init weights using Xavier initialization.
        :param shape: shape of the weights matrix
        """
        # init weights using Xavier initialization
        scale = 1 / max(1., (2 + 2) / 2.)
        limit = np.sqrt(3.0 * scale)
        self._weights = np.random.uniform(-limit, limit, size=shape)

    def _progress_report(self, iter: int, metrics: dict):
        """
        Report progress to tqdm progress bar.
        :param iter: iteration number
        :param metrics: metrics to report
        """
        assert self.pbar is not None, 'Error pbar referenced before initialization'
        self.pbar.update(1)
        self.pbar.set_description(f"iter: {iter} | {' | '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}")

    def _criterion(self, y: np.ndarray, y_hat: np.ndarray):
        """
        loss function - hinge loss
        :param y: ground truth
        :param y_hat: prediction
        :return: max(0, a - y * y_hat) + b * ||w||^2
        """
        # L = a - y * y_hat + b * ||w||^2
        loss = np.maximum(0, self._margin - y * y_hat) + self._lambda * np.dot(self._weights, self._weights.T)
        return loss.item()

    def _grad(self, x: np.ndarray, y: np.ndarray, y_hat: np.ndarray):
        """
        Gradient of the loss function
        :param x: sample
        :param y: ground truth
        :param y_hat: prediction
        :return: gradient of the hinge loss function
        """
        regularization_grad = 2 * self._lambda * self._weights
        prediction_grad = 0 if (y * y_hat > self._margin) else -y * x
        return prediction_grad + regularization_grad

    def transform_sign(self, Y: np.ndarray):
        """
        Transform Y to {-1, 1}
        :param Y: ground truth
        :return: Y transformed to {-1, 1}
        """
        return np.where(Y == 0, -1, Y)

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = True):
        """
        Train the model
        :param X: samples
        :param Y: ground truth
        :param verbose: report progress and metrics (default: True)
        """
        # transform Y to {-1, 1}
        Y = self.transform_sign(Y)
        assert set(np.unique(Y)).intersection({-1, 1}) == {-1, 1}, 'Y must be binary {-1, 1}'

        # init progress bar
        self.pbar = tqdm(disable=not verbose)

        # init weights
        assert X.shape[0] == Y.shape[0], 'X and Y must have the same number of samples'
        self._init_weights((1, X.shape[1]))


        # train loop
        iter, train = 1, True
        while train:
            last_weights = self._weights.copy()
            total_loss = 0

            # gradient descent - no batch
            for x, y in zip(X, Y):
                y_hat = np.dot(self._weights, x)
                total_loss += self._criterion(y, y_hat)
                grad = self._grad(x, y, y_hat, reg_decay=1/iter)
                self._weights = self._weights - self._lr * grad

            # check convergence / max iterations reached
            if np.linalg.norm(self._weights - last_weights) < self._tol or iter == self._max_iter:
                train = False
            iter += 1

            # progress report
            metrics = {
                'loss': total_loss / X.shape[0],
                'accuracy': 1 - np.sum(np.abs(np.clip(self.predict(X), 0, 1).squeeze(1) - np.clip(Y, 0, 1))) / X.shape[0]
            }
            self._progress_report(iter=iter, metrics=metrics)

        self.pbar.close()
        self.pbar = None

    def predict(self, X: np.ndarray):
        """
        Predict using the trained model
        :param X: samples
        :return: predictions {-1, 1}
        """
        assert self._weights is not None, 'Model not trained yet'
        return np.sign(np.dot(X, self._weights.T))


if __name__ == '__main__':
    from basic_algorithms.ml.data.data_loader import load_diabetes
    X, Y = load_diabetes()
    clf = BinarySVMClassifier()
    clf.fit(X, Y)
