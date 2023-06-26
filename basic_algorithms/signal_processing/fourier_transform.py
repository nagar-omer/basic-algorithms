import numpy as np


def dft_1d(x: np.ndarray):
    """
    DFT of 1D signal
    X[k] = SUM_n(x[n]*exp(-2*pi*i* k*n/N ))

    :param x: 1D signal
    :return: DFT of x
    """
    assert len(x.shape) == 1, "Input must be 1D"

    res = np.zeros(len(x), dtype=np.complex128)
    N = len(x)

    # calculate the sum of each element
    for k in range(len(res)):
        for n, xn in enumerate(x):
            res[k] += xn * np.exp((-2*np.pi*1j*k*n)/N)
    return res


def idft_1d(x: np.ndarray):
    """
    IDFT of 1D signal
    X[n] = 1/N * SUM_k(x[n]*exp(2*pi*i*k*n/N))

    :param x: 1D signal
    :return: IDFT of x
    """
    assert len(x.shape) == 1, "Input must be 1D"

    res = np.zeros(len(x), dtype=np.complex128)
    N = len(x)

    # calculate the sum of each element
    for n in range(len(res)):
        for k, xk in enumerate(x):
            res[n] += xk * np.exp((2*np.pi*1j*k*n)/N)
    return res / N


def dft_2d(f: np.ndarray):
    """
    DFT of 2D signal
    F[omega_x][omega_y] = SUM_x SUM_y (f[x][y] * exp(-2*pi*i* ((omega_x*x)/M + (omega_y*y)/N) )

    :param f: 2D signal
    :return: DFT of f
    """
    assert len(f.shape) == 2, "Input must be 2D"
    res = np.zeros(f.shape, dtype=np.complex128)
    M, N = f.shape

    # calculate the sum of each element
    for omega_x in range(f.shape[0]):
        for omega_y in range(f.shape[1]):
            for x in range(f.shape[0]):
                for y in range(f.shape[1]):
                    res[omega_x, omega_y] += f[x, y] * np.exp(-2*np.pi*1j*((omega_x*x)/M + (omega_y*y)/N))

    return res


def idft_2d(F: np.ndarray):
    """
    IDFT of 2D signal
    f[x][y] = 1/MN * SUM_wx SUM_wy (F[x][y] * exp(2*pi*i* ((wx*x)/M + (wy*y)/N) )

    :param F: 2D signal
    :return: IDFT of F
    """
    assert len(F.shape) == 2, "Input must be 2D"
    res = np.zeros(F.shape, dtype=np.complex128)
    M, N = F.shape

    # calculate the sum of each element
    for x in range(F.shape[0]):
        for y in range(F.shape[1]):
            for omega_x in range(F.shape[0]):
                for omega_y in range(F.shape[1]):
                    res[x, y] += F[omega_x, omega_y] * np.exp(2*np.pi*1j*((omega_x*x)/M + (omega_y*y)/N))

    return res / (M * N)


if __name__ == '__main__':
    x = np.asarray([[1, 2 - 1j, -1j, -1 + 2j]]).repeat(4, 0)
    x[1] += 0.5*x[0]
    x[2] += 0.5*x[1]
    x[3] += 0.5*x[2]

    x.clip(-1, 1)
    y = dft_2d(x)
    x_ = idft_2d(y)

    x_ == x
    np.testing.assert_almost_equal(x, x_)

    e = 0