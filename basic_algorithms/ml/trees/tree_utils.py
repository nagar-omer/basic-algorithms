import numpy as np


def split_continuous_feature(data, n_split, method='quantile'):
    """
    Find the best split for a dataset.
    :param data: dataset to split
    :param n_split: number of splits
    :param method: method to split by: quantile | mean | random
    """
    assert data.ndim < 3, f"Invalid data shape {data.shape}, expected (n_samples, n_features)"
    assert n_split > 1, f"Invalid n_split {n_split}, expected > 1"
    if data.ndim == 1:
        data = data[:, np.newaxis]

    # sort data
    data = data[np.argsort(data[:, 0])]
    if method == 'quantile':
        # split data into n_split quantiles
        quantiles = np.linspace(0, 1, n_split + 1)[1:-1]
        split_points = np.quantile(data, quantiles, axis=0)
    elif method == 'mean':
        # split data into n_split equal parts
        split_points = np.linspace(np.min(data, axis=0), np.max(data, axis=0), n_split + 1)[1:-1, :]
    elif method == 'random':
        # split data into n_split random parts
        get_rand_splits = lambda v_min, v_max: np.sort(np.random.uniform(v_min, v_max, n_split - 1))
        split_points = np.vstack([get_rand_splits(v_min, v_max) for v_min, v_max in
                                  zip(np.min(data, axis=0), np.max(data, axis=0))]).T
    else:
        raise ValueError(f"Invalid method {method}, expected one of ['quantile', 'mean']")

    return split_points


if __name__ == '__main__':
    data = np.random.uniform(0, 1, (100, 3))
    split_continuous_feature(data, 5, method='quantile')

