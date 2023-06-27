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


def fft_1d(x: np.ndarray):
    """
    recursive function that follows the rule:
    DFT[k] = <FFT[EVEN] + f * FFT[ODD]><FFT[EVEN] + f * FFT[ODD]>
    """
    assert len(x.shape) == 1, "Input must be 1D"
    N = x.shape[0]
    assert N % 2 == 0, "Input must be a power of 2"

    # stop condition
    if N <= 2:
        return dft_1d(x)

    # split to and odd -> apply recursive function
    res_even = fft_1d(x[0::2])
    res_odd = fft_1d(x[1::2])
    factor = np.exp(-2j*np.pi*np.arange(N)/N)
    return np.hstack([res_even, res_even]) + factor * np.hstack([res_odd, res_odd])


def ifft_1d(x: np.ndarray):
    """
    IFFT implementation of 1D signal  using recursive FFT
    """
    assert len(x.shape) == 1, "Input must be 1D"
    N = x.shape[0]
    return np.conj(fft_1d(np.conj(x))) / N
