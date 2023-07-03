import numpy as np
from skimage import color
from basic_algorithms.classical_cv.image_processing.filters import apply_filter, gaussian_kernel
from basic_algorithms.classical_cv.image_processing import non_maximal_suppression


def haris_coroner_detector(image: np.ndarray, k=0.06, sigma_window=3, window_size=51, threshold=None, nms_window_size=101):
    """
    good explanartion here: https://www.baeldung.com/cs/harris-corner-detection
    :param image: input image
    :param sigma_window: sigma for gaussian filter used to smooth partial derivatives
    :param window_size: size of the sliding window used to calculate partial derivatives
    :param threshold: threshold for non-maximal suppression
    :param nms_window_size: size of the sliding window used to calculate non-maximal suppression
    :param k: k is a constant to chose in the range [0.04, 0.06].
              used as a factor in harris score: det(M) - k * tr(M)^2
    """
    assert 0.04 <= k <= 0.06, "k should be in range of [0.04, 0.06]"

    # convert to grayscale
    grayscale_image = color.rgb2gray(image) if image.ndim == 3 else image

    # compose kernels (for calculation of partial derivatives)
    kernel_x = np.asarray([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # get partial derivatives for each pixel
    grad_x = apply_filter(grayscale_image, kernel_x, n_jobs=1)
    grad_y = apply_filter(grayscale_image, kernel_y, n_jobs=1)

    # calculate Ixx, Iyy, Ixy & smooth using gaussian filter to smooth the partial derivatives
    gauss_kernel = gaussian_kernel(sigma=sigma_window, kernel_size=window_size)
    Ixx = apply_filter(grad_x * grad_x, gauss_kernel, n_jobs=1)
    Iyy = apply_filter(grad_y * grad_y, gauss_kernel, n_jobs=1)
    Ixy = apply_filter(grad_x * grad_y, gauss_kernel, n_jobs=1)

    # generate hessian matrix for each pixel
    hessian = np.zeros((*image.shape[:2], 2, 2))
    hessian[..., 0, 0] = Ixx
    hessian[..., 0, 1] = Ixy
    hessian[..., 1, 0] = Ixy
    hessian[..., 1, 1] = Iyy

    # calculate eigenvalues & and Haris score
    eigvals, _ = np.linalg.eig(hessian)
    haris_score = eigvals[..., 0] * eigvals[..., 1] - k * np.trace(eigvals[..., 0] + eigvals[..., 1])**2

    # apply thresholding
    haris_score = (haris_score - haris_score.min()) / (haris_score.max() - haris_score.min())
    threshold = np.quantile(haris_score.flatten(), 0.9) if threshold is None else threshold
    corner_candidates = haris_score.copy()
    corner_candidates[haris_score < threshold] = 0

    # apply non maximum suppression
    # x1, x2, y1, y2 for each pixel
    def pixel_window(x, y):
        margin = nms_window_size // 2
        return np.asarray([np.clip(x - margin, 0, image.shape[0]), np.clip(y - margin, 0, image.shape[1]),
                           np.clip(x + margin, 0, image.shape[0]), np.clip(y + margin, 0, image.shape[1])])
    sliding_window = np.transpose(np.fromfunction(pixel_window, image.shape[:2]), (1, 2, 0))
    corners = non_maximal_suppression.nms(corner_candidates, sliding_window)
    return corners

