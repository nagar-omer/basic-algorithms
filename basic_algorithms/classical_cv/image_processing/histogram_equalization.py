from typing import Callable

import numpy as np
from joblib import Parallel, delayed


def split_image(image: np.ndarray, n_rows: int, n_cols: int):
    """
    Split an image into n_rows x n_cols blocks
    :param image: image to split
    :param n_rows: number of rows to split the image into
    :param n_cols: number of columns to split the image into
    :return: a list of blocks
    """

    # get image/block dimensions dimensions
    height, width = image.shape[:2]

    # get row and column split indices
    row_splits = np.linspace(0, height, n_rows + 1).astype(np.uint32)
    col_splits = np.linspace(0, width, n_cols + 1).astype(np.uint32)

    # split the image into blocks
    image_blocks = []
    for start_row, end_row in zip(row_splits[:-1], row_splits[1:]):
        image_blocks.append([image[start_row:end_row, start_col:end_col]
                             for start_col, end_col in zip(col_splits[:-1], col_splits[1:])])

    return image_blocks


def merge_blocks(image_blocks: list):
    """
    Merge a list of image blocks into a single image
    :param image_blocks: image blocks to merge
    :return: image
    """
    return np.vstack([np.hstack(block_row) for block_row in image_blocks])


def _pdf(image: np.ndarray, bins: int = 256, normalize: bool = True):
    """
    Compute the cumulative distribution function of an image
    :param image: image to compute the cdf of
    :param bins: number of bins to use (default: 256)
    :param normalize: if True return probability density function else return counts (default: True)
    :return: the cdf of the image
    """
    pdf = np.zeros(bins)
    for pixel in image:
        pdf[pixel] += 1
    return (pdf / np.sum(pdf)) if normalize else pdf


def _cdf(image: np.ndarray, bins: int = 256, normalize: bool = True):
    """
    Compute the cumulative distribution function of an image
    :param image: image to compute the cdf of
    :param bins: number of bins to use  (default: 256)
    :param normalize: if True return cumulative density function else return counts  (default: True)
    :return: the cdf of the image
    """
    pdf = _pdf(image, bins, normalize)
    cdf = pdf.cumsum()
    return cdf


def histogram_equalization(image: np.ndarray, alpha: float = 1.0, cdf: np.ndarray = None):
    """
    Transform function that that results in an image with a uniform histogram
    To achieve this, a transformation T must satisfy the following conditions:

    s = T(r)
    PT(s) = PR(r) * |dr/ds| = 1/L    (1)
    where L is the number of gray levels in the image.

    The transformation T is that satisfies the above condition is defined as:
    T(r) = (L-1) * integral_w_{0..r}(PR(w)) = sum(pixel<=r)  - commutative histogram

    proof:
    ds/dr = d(T(r))/dr = d(L * integral_w_{0..r}(PR(w)))/dr = L * PR(pixel<=r)

    apply the transformation (1)
    PT(s) = PR(r) * |dr/ds| = PR(r) / L * PR(pixel<=r)) = 1/L

    :param image: image to be transformed
    :param alpha: linear blend coefficient (default: 1.0)
    :param cdf: precomputed cdf of the image (default: None)
    :return: equalized image according to the above transformation
    """
    assert image.ndim in [2, 3], "Image must be grayscale or RGB"
    assert image.dtype == np.uint8, "Image must be uint8"
    assert 0 <= alpha <= 1, "Alpha must be in [0, 1]"

    image_cdf = _cdf(image.flatten(), bins=256) if cdf is None else cdf

    @np.vectorize
    def transform(pixel):
        return np.uint8(255 * image_cdf[pixel])

    return (alpha * transform(image) + (1 - alpha) * image).astype(np.uint8)


def block_histogram_equalization(image: np.ndarray, n: int = 8, alpha: float = 1.):
    """
    Perform histogram equalization on
    :param image: image to equalize
    :param n: number of blocks to split the image into (image will be split into n x n blocks)
    :param alpha: linear blend coefficient (default: 1.0)
    :return: equalized image
    """

    # split
    block_image = split_image(image, n, n)
    # equalize
    equalized_blocks = [[histogram_equalization(block, alpha=alpha) for block in row_block]
                        for row_block in block_image]
    # merge blocks
    equalized_image = merge_blocks(equalized_blocks)
    return equalized_image


def sliding_cdf(image: np.ndarray, n: int = 8, n_jobs: int = -1):
    """
    Compute the cumulative distribution function of a sliding window around a pixel
    complexity: O(n^2 + n * N_pixel) | N_pixel >> n^2 -> O(n*N_pixel)
    :param image: image to compute the cdf of
    :param n: size of the sliding window
    :param n_jobs: number of parallel jobs to run (default: -1)
    :return: the cdf of the sliding window
    """
    assert n % 2 == 1, "n must be odd"
    assert image.ndim in [2, 3], "Image must be grayscale or RGB"
    assert image.dtype == np.uint8, "Image must be uint8"

    pdf_mat = np.zeros((1, image.shape[1], 256))

    # init first block
    margin = n // 2
    pdf_mat[0, 0] = _pdf(image[:margin + 1, :margin + 1].flatten(), bins=256, normalize=False)

    # init rest of first row
    for j in range(1, image.shape[1]):
        remove_col = image[:margin + 1, j - margin].flatten() if j >= margin else []
        add_col = image[:margin + 1, j + margin].flatten() if j + margin < image.shape[1] else []

        pdf_mat[0, j] = pdf_mat[0, j-1] \
                        - _pdf(remove_col, bins=256, normalize=False) \
                        + _pdf(add_col, bins=256, normalize=False)

    # function to compute the cdf of a sliding window for entire row (assuming window is already computed)
    def job_sliding_pdf_block(base_hist: np.ndarray, col_idx: int):
        # container for the cdf of the sliding window for requested column
        col_pdf_mat = np.zeros((image.shape[0], 256))
        col_pdf_mat[0, :] = base_hist
        # column boundaries
        col_left, col_right = max(0, col_idx - margin),  min(col_idx + margin + 1, image.shape[1])

        # compute the cdf of the sliding window for each row -> for each pixel get diff
        for i in range(1, image.shape[0]):
            remove_row = image[i - margin, col_left:col_right].flatten() \
                if i >= margin else []
            add_row = image[i + margin, col_left:col_right].flatten() \
                if i + margin < image.shape[0] else []

            col_pdf_mat[i] = col_pdf_mat[i-1] \
                             - _pdf(remove_row, bins=256, normalize=False) \
                             + _pdf(add_row, bins=256, normalize=False)
        return col_pdf_mat

    # fill rest of matrix in parallel
    x = Parallel(n_jobs=n_jobs)(delayed(job_sliding_pdf_block)(pdf_mat[0, j], j) for j in range(image.shape[1]))
    pdf_mat = np.stack(x).swapaxes(0, 1)


    # convert to probabilities
    pdf_mat = pdf_mat / np.sum(pdf_mat, axis=2, keepdims=True).clip(min=1e-10)
    cdf_mat = pdf_mat.cumsum(axis=2)
    return cdf_mat


def sliding_histogram_equalization(image: np.ndarray, n: int = 129, alpha: float = 1.):
    """
    Perform histogram equalization on an image using a sliding window
    :param image: image to equalize
    :param n: size of the sliding window, n should be an odd number(default: 129)
    :param alpha: linear blend coefficient (default: 1.0)
    :return: equalized image
    """

    # calculate cdf matrix (sliding window)
    cdf = sliding_cdf(image, n=n)

    # apply equalization
    eq_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # for each pixel get histogram equalization value from cdf matrix
            if image.ndim == 3:
                for k in range(image.shape[2]):
                    eq_image[i, j, k] = cdf[i, j][image[i, j, k]] * 255
            else:
                eq_image[i, j] = cdf[i, j][eq_image[i, j]] * 255
    return (alpha * eq_image + (1 - alpha) * image).astype(np.uint8)


def map_to_closest_block(height: int, width: int, n_rows: int, n_cols: int) -> Callable:
    """
    Map each pixel to up to closest 2x2 blocks according to distance from block center
    :param height: original image height
    :param width: original image width
    :param n_rows: n blocks along rows
    :param n_cols: n blocks along cols
    :return: mapping function from each pixel to closest 2x2 blocks
    """

    # calculate block centers
    row_splits = np.linspace(0, height, n_rows + 1).astype(np.uint32)
    col_splits = np.linspace(0, width, n_cols + 1).astype(np.uint32)

    row_center = row_splits[:-1] + (row_splits[1:] - row_splits[:-1]) // 2
    col_center = col_splits[:-1] + (col_splits[1:] - col_splits[:-1]) // 2

    # calculate closest 2 blocks for each pixel - along rows and cols
    row_dist_from_center = np.abs(np.expand_dims(np.arange(height), 1).repeat(row_center.shape[0], 1) -
                                  np.expand_dims(row_center, 0).repeat(height, 0))
    row_top_two = np.argsort(row_dist_from_center, axis=1)[:, :2]
    col_dist_from_center = np.abs(np.expand_dims(np.arange(width), 1).repeat(col_center.shape[0], 1) -
                                  np.expand_dims(col_center, 0).repeat(width, 0))
    col_top_two = np.argsort(col_dist_from_center, axis=1)[:, :2]

    box_height, box_width = height // n_rows, width // n_cols

    # mapping function from each pixel to closest 2x2 blocks
    def _map_func(i, j):
        # get closest 2 blocks along rows and cols and pixel distance from their center
        closest_boxes_vertical = row_top_two[i]
        closest_boxes_horizontal = col_top_two[j]
        dist_vertical = row_dist_from_center[i, row_top_two[i]]
        dist_horizontal = col_dist_from_center[j, col_top_two[j]]

        # get boxes
        boxes = []
        for box_vert, dist_vert in zip(closest_boxes_vertical, dist_vertical):
            for box_horiz, dist_horiz in zip(closest_boxes_horizontal, dist_horizontal):
                # drop if distance from center is larger than box size
                if dist_horiz > box_height or dist_vert > box_width:
                    continue

                boxes.append([box_vert, box_horiz, (1 - dist_vert/box_height) * (1 - dist_horiz/box_width)])

        # normalize weights
        boxes = [(x, y, weight) for x, y, weight in boxes]
        return boxes

    return _map_func


def adaptive_histogram_equalization(image, n=8):
    """
    Perform adaptive histogram equalization on an image
    :param image: image to equalize
    :param n: number of blocks along each axis (default: 8)
    :return: equalized image
    """

    # split image to blocks and calculate cdf for each block
    image_blocks = split_image(image, n, n)
    cdf_blocks = np.zeros((n, n, 256))
    for i in range(n):
        for j in range(n):
            cdf_blocks[i, j] = _cdf(image_blocks[i][j])

    # map each pixel to closest 2x2 blocks
    spline_map = map_to_closest_block(*image.shape[:2], n, n)

    # apply equalization
    equalized_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            boxes = spline_map(i, j)
            for box_x, box_y, weight in boxes:
                equalized_image[i, j] += (255 * weight * cdf_blocks[box_x, box_y][image[i, j]]).astype(np.uint8)

    return equalized_image


if __name__ == '__main__':
    import imageio.v3 as imageio
    import matplotlib.pyplot as plt

    image = imageio.imread('../data/lena.jpg')

    eq_image1 = histogram_equalization(image)
    eq_image2 = block_histogram_equalization(image)
    eq_image3 = sliding_histogram_equalization(image)
    eq_image4 = adaptive_histogram_equalization(image)

    plt.imshow(image)
    plt.title('Original Image')
    plt.show()

    plt.imshow(eq_image1)
    plt.title('Histogram Equalization')
    plt.show()

    plt.imshow(eq_image2)
    plt.title('Block Histogram Equalization')
    plt.show()

    plt.imshow(eq_image3)
    plt.title('Sliding Window Histogram Equalization')
    plt.show()

    plt.imshow(eq_image4)
    plt.title('Adaptive Histogram Equalization')
    plt.show()
