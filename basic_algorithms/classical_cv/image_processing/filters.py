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


def sinc_kernel(omega: float, kernel_size: int):
    """
    Create a sinc kernel of size N x N, using Bessel function
    :param omega: angular frequency (in radians)
    :param kernel_size: kernel size
    :return: sinc kernel
    """

    # N must be odd
    assert kernel_size % 2 == 1, "N must be odd"

    # Compute sinc kernel
    def _sinc(x, y):
        z = np.sqrt((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)
        out = omega * special.j1(omega * z) / (2 * np.pi * z)
        return np.nan_to_num(out, 0)

    # apply sinc kernel to each pixel
    kernel = np.fromfunction(_sinc, [kernel_size, kernel_size])
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
    kernel = sinc_kernel(omega, kernel_size=kernel_size)
    filtered = apply_filter(image, kernel, n_jobs=n_jobs)
    return filtered


def high_pass_filter(image: np.ndarray, omega: float, kernel_size: int = 15, n_jobs: int = -1):
    """
    Apply a high-pass filter to an image.
    """
    kernel = unit_impulse_kernel(kernel_size) - sinc_kernel(omega, kernel_size=kernel_size)
    filtered = apply_filter(image, kernel, n_jobs=n_jobs)
    return filtered


def band_reject_filter(image: np.ndarray, omega_low: float, omega_high: float,
                       kernel_size: int = 15, n_jobs: int = -1):
    """
    Apply a band-reject filter to an image.
    :param image: The image to apply the band-reject filter to.
    :param omega_low: The frequency of the low-pass filter (in radians).
    :param omega_high: The frequency of the high-pass filter (in radians).
    :param kernel_size: The size of the sinc kernel.
    :param n_jobs: The number of jobs to run in parallel.
    :return: filtered image.
    """
    # get kernels for low-pass and high-pass
    low_kernel = sinc_kernel(omega_low, kernel_size=kernel_size)
    high_kernel = unit_impulse_kernel(kernel_size) - sinc_kernel(omega_high, kernel_size=kernel_size)

    # compute band-pass kernel & normalize
    band_pass_kernel = high_kernel + low_kernel

    # apply band-pass filter
    filtered = apply_filter(image, band_pass_kernel, n_jobs=n_jobs)
    return filtered


def band_pass_filter(image: np.ndarray, omega_low: float, omega_high: float,
                     kernel_size: int = 15, n_jobs: int = -1):
    """
    Apply a band-pass filter to an image.
    :param image: image to apply the band-pass filter to.
    :param omega_low: low-pass filter frequency (in radians).
    :param omega_high: high-pass filter frequency (in radians).
    :param kernel_size: size of the sinc kernel.
    :param n_jobs: number of jobs to run in parallel.
    :return: filtered image.
    """

    # get kernels for low-pass and high-pass
    low_kernel = sinc_kernel(omega_low, kernel_size=kernel_size)
    high_kernel = sinc_kernel(omega_high, kernel_size=kernel_size)

    # compute band-pass kernel & normalize
    band_pass_kernel = high_kernel - low_kernel

    # apply band-pass filter
    filtered = apply_filter(image, band_pass_kernel, n_jobs=n_jobs)
    return filtered


def gaussian_kernel(sigma: float, kernel_size: int):
    def _2d_gaussian(x, y):

        nominator = np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma**2))
        denominators = 2 * np.pi * sigma ** 2
        return nominator / denominators

    kernel = np.fromfunction(_2d_gaussian, [kernel_size, kernel_size])
    return kernel / kernel.sum()


def dog_filter(image: np.ndarray, sigma_low: int, sigma_high: int, kernel_size: int = 15, n_jobs: int = -1):
    """
    Computes the Difference of Gaussian's (DoG) of an image - type of band-pass filter.
    :param image: The image to compute the DoG of.
    :param sigma_low: low-gaussian sigma.
    :param sigma_high: high-gaussian sigma.
    :param kernel_size: kernel dim is - kernel size x kernel size.
    :param n_jobs: Number of jobs to run in parallel.
    :return: The DoG of the image.
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    assert sigma_low > 0 and sigma_high > 0, "sigma_low and sigma_high must be positive"
    assert sigma_low < sigma_high, "sigma_low must be smaller than sigma_high"

    kernel_low = gaussian_kernel(sigma_low, kernel_size=kernel_size)
    kernel_high = gaussian_kernel(sigma_high, kernel_size=kernel_size)

    dog_kernel = kernel_low - kernel_high
    filtered = apply_filter(image, dog_kernel, n_jobs=n_jobs)
    return filtered


def log_filter(image: np.ndarray):
    """
    Computes the Laplacian of Gaussian's (LoG) of an image.
    :param image: The image to compute the LoG of.
    :return: The LoG of the image.
    """
    pass


if __name__ == '__main__':
    import imageio.v3 as imageio
    import matplotlib.pyplot as plt

    image = imageio.imread('../data/lena.jpg')

    plt.imshow(image)
    plt.show()

    image = dog_filter(image, sigma_low=2, sigma_high=4, kernel_size=51)
    plt.imshow(image)
    plt.show()
