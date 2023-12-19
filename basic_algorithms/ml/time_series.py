import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize


def window_iterator(time_series: np.ndarray, window_size: int):
    """
    This function creates a sliding window iterator over a time series, it returns a tuple of (window, label)
    where window is a list of values of size window_size and label is the next value in the time series
    :param time_series:
    :param window_size:
    :return:
    """
    num_samples = len(time_series) - window_size + 1
    moving_window = np.arange(window_size)[np.newaxis, :] + np.arange(num_samples)[:, np.newaxis]
    label_index = np.arange(window_size, len(time_series))

    for window, label in zip(moving_window, label_index):
        yield time_series[window], time_series[label]
    yield time_series[moving_window[-1]], None


class AutoregressiveClassifier:
    def __init__(self, lag: int = 1):
        """
        Autoregressive model of order p (AR(p))
        :param lag: lag order
        """
        self._lag = lag
        self._clf = None

    @property
    def coef_(self):
        return self._clf.coef_

    def fit(self, time_series: np.ndarray):
        """
        :param time_series: list of time series
        :return: AR(p) model
        """

        # last window doesn't have a label
        samples, labels = zip(*window_iterator(time_series, self._lag))
        self._clf = LinearRegression().fit(samples[:-1], labels[:-1])

        return self

    def predict(self, time_series: np.ndarray):
        """
        Predict the next value in the time series
        :param time_series: array of shape (N, )
        :return: array of shape (N + 1, ) where  first p elements are nan
        """
        if self._clf is None:
            self._clf = self.fit(time_series)

        samples, labels = zip(*window_iterator(time_series, self._lag))
        return np.hstack([np.zeros(self._lag) * np.nan, self._clf.predict(samples)])


class ARMAClassifier:
    """
    Autoregressive moving average model of order p, q (ARMA(p, q))
    """
    def __init__(self, p: int, q: int):
        self._ar_coeffs = None
        self._ma_coeffs = None
        self._p = p
        self._q = q

    def fit(self, time_series: np.ndarray):
        samples, y = zip(*window_iterator(time_series, self._p))

        # Create lagged data for AR(p)
        ar_clf = AutoregressiveClassifier(lag=self._p).fit(time_series)
        self._ar_coeffs = ar_clf.coef_

        # Calculate residuals for MA(q),
        #   first p elements in residuals do not have a corresponding label
        #   last element in prediction does not have a corresponding label
        residuals = y[:-1] - ar_clf.predict(time_series)[self._p:-1]
        initial_ma_params = np.zeros(self._q)

        # Function to be minimized for MA coefficients (minimum likelihood estimation)
        def MLE(params):
            ma_params = params
            ma_residuals = residuals.copy()
            for t in range(self._q, len(ma_residuals)):
                ma_residuals[t] -= np.dot(ma_params, ma_residuals[t - self._q:t])
            return np.sum(ma_residuals[self._q:] ** 2)

        # Minimize the MLE function to get MA coefficients
        result = minimize(MLE, initial_ma_params, method='BFGS')
        self._ma_coeffs = result.x
        return self

    def predict(self, time_series: np.ndarray):
        """
        give full prediction of ARMA(p, q) model to the whole time series
        :param time_series: array of shame (N, )
        :return: array of shape (N + 1, ) where  first n+q elements are nan
        """
        samples, labels = zip(*window_iterator(time_series, self._p))
        ar_pred = (self._ar_coeffs * np.asarray(samples)).sum(axis=1)
        # last element in prediction does not have a corresponding label
        residuals = labels[:-1] - ar_pred[-1]

        # residual shape = (|time_series| - self._p  - self._q + 1) x self._q
        residuals = np.vstack(list(zip(*window_iterator(residuals, self._q)))[0])
        ma_pred = (self._ma_coeffs * residuals).sum(axis=1)

        arma_pred = ma_pred + ar_pred[self._q:]
        return np.hstack([np.zeros(self._p + self._q) * np.nan, arma_pred])


def sin_time_series(n: int = 1000, noise: float = 0.1):
    data = np.sin(np.linspace(0, 10 * np.pi , n)) + np.random.normal(0, noise, n)
    return data


def poly_time_series(n: int = 1000, noise: float = 0.1):
    data = np.linspace(0, 3, n) ** 2 + np.random.normal(0, noise, n)
    return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = sin_time_series(noise=0) + poly_time_series(noise=0.05)

    clf_ = AutoregressiveClassifier(lag=2).fit(data[:600])
    prediction_2 = np.hstack([np.zeros(600) * np.nan, clf_.predict(data[600:])])
    clf_ = AutoregressiveClassifier(lag=16).fit(data[:600])
    prediction_16 = np.hstack([np.zeros(600) * np.nan, clf_.predict(data[600:])])
    clf_ = AutoregressiveClassifier(lag=128).fit(data[:600])
    prediction_128 = np.hstack([np.zeros(600) * np.nan, clf_.predict(data[600:])])

    clf_ = ARMAClassifier(p=2, q=2).fit(data[:600])
    arma_prediction_2 = np.hstack([np.zeros(600) * np.nan, clf_.predict(data[600:])])
    clf_ = ARMAClassifier(p=16, q=16).fit(data[:600])
    arma_prediction_16 = np.hstack([np.zeros(600) * np.nan, clf_.predict(data[600:])])

    plt.figure(figsize=(50, 10))
    plt.title('Autoregressive model')
    plt.plot(data[:600], label='data')
    plt.plot(prediction_2, label='AR prediction lag 2')
    plt.plot(prediction_16, label='AR prediction lag 16')
    plt.plot(prediction_128, label='AR prediction lag 128')
    plt.plot(arma_prediction_2, label='ARMA prediction p,q 2, 2')
    plt.plot(arma_prediction_16, label='ARMA prediction p,q 16, 16')
    plt.legend()
    plt.show()
