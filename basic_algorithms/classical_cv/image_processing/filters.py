import numpy as np
from scipy import special
from scipy.signal import fftconvolve
from joblib import Parallel, delayed


def unit_impulse_kernel(N: int):
    """
    Return a unit impulse kernel of size N x N.
    """
    kernel = np.zeros((N, N))
    kernel[N // 2, N // 2] = 1
    return kernel


def sinc_kernel(omega: float, N: int):
    """
    Create a sinc kernel of size N x N, using Bessel function
    :param omega: angular frequency (in radians)
    :param N: kernel size
    :return: sinc kernel
    """

    # N must be odd
    assert N % 2 == 1, "N must be odd"

    # Compute sinc kernel
    def _sinc(x, y):
        z = np.sqrt((x - (N - 1)/2) ** 2 + (y - (N - 1)/2) ** 2)
        out = omega * special.j1(omega * z) / (2 * np.pi * z)
        return np.nan_to_num(out, 0)

    # apply sinc kernel to each pixel
    kernel = np.fromfunction(_sinc, [N, N])
    return kernel / kernel.sum()


def apply_filter(image: np.ndarray, kernel: np.ndarray, n_jobs: int = -1):
    # apply kernel to image along channel axis
    if image.ndim == 2:
        filtered = fftconvolve(image, kernel, mode='same')
    elif image.ndim == 3:
        x = Parallel(n_jobs=n_jobs)(delayed(fftconvolve)(image[:, :, c], kernel, mode='same')
                                    for c in range(image.shape[2]))
        filtered = np.stack(x, axis=2).astype(image.dtype)
    else:
        raise NotImplementedError
    return filtered


def low_pass_filter(image: np.ndarray, omega: float, kernel_size: int = 15, n_jobs: int = -1):
    """
    Apply a low-pass filter to an image, using a sinc kernel.
    :param image: The image to apply the low-pass filter to.
    :param omega: The frequency of the low-pass filter (in radians).
    :param kernel_size: The size of the sinc kernel.
    :param n_jobs: The number of jobs to run in parallel.
    :return: filtered image.
    """

    # get kernel
    kernel = sinc_kernel(omega, N=kernel_size)
    filtered = apply_filter(image, kernel, n_jobs=n_jobs)
    return filtered


def high_pass_filter(image: np.ndarray, omega: float, kernel_size: int = 15, n_jobs: int = -1):
    """
    Apply a high-pass filter to an image.
    """
    kernel = unit_impulse_kernel(kernel_size) - sinc_kernel(omega, N=kernel_size)
    filtered = apply_filter(image, kernel, n_jobs=n_jobs)
    return filtered


def DoG(image: np.ndarray):
    """
    Computes the Difference of Gaussian's (DoG) of an image.
    :param image: The image to compute the DoG of.
    :return: The DoG of the image.
    """
    pass


def LoG(image: np.ndarray):
    """
    Computes the Laplacian of Gaussian's (LoG) of an image.
    :param image: The image to compute the LoG of.
    :return: The LoG of the image.
    """
    pass

