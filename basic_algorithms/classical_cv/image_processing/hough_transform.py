import numpy as np
import cv2
from skimage import color


def line_hough_transform(image, theta_range=None, d_range=None, n_theta=None, n_rho=None, hough_threshold=0.9):

    # set parameters
    height, width = image.shape[:2]
    d_range = [-np.sqrt(height**2 + width**2), np.sqrt(height**2 + width**2)] if d_range is None else d_range
    theta_range = [11.25, 180] if theta_range is None else theta_range
    n_theta = 16 if n_theta is None else n_theta
    n_rho = int(np.sqrt(height**2 + width**2) / 1) if n_rho is None else n_rho

    # hough space matrix
    theta_axis = np.linspace(theta_range[0], theta_range[1], n_theta)
    rho_bins = np.linspace(d_range[0], d_range[1], n_rho + 1).astype(int)
    rho_axis = (rho_bins[1:] + rho_bins[:-1]) / 2
    hough_space = np.zeros((n_theta, n_rho))

    # function to extract rho according to theta and pixel coordinates
    def _find_rho(x, y, theta, is_feature):
        return is_feature * x * np.cos(theta) + y * np.sin(theta)

    # get edges and arguments to calculate rho values
    edges = cv2.Canny(image, 140, 160)
    image_coordinates = np.transpose(np.stack(np.meshgrid(np.arange(width), np.arange(height))), (1, 2, 0))
    is_feature = (edges > np.quantile(edges.flatten(), 0.8)).astype(np.uint8)

    # calculate rho values
    for i_theta, theta in enumerate(theta_axis):
        # calculate rho values using theta, pixel coordinates and edges
        theta_vals = np.ones_like(edges) * theta
        rho_vals = _find_rho(image_coordinates[:, :, 0], image_coordinates[:, :, 1], theta_vals, is_feature)

        # rho > 0 -> found matching line. fill hough space matrix accordingly
        hist_rho, _ = np.histogram(rho_vals[rho_vals > 0], bins=rho_bins)
        hough_space[i_theta, :] = hist_rho

    i_theta, i_rho = np.where(hough_space >= np.quantile(hough_space, 0.9))
    theta, rho = theta_axis[i_theta], rho_axis[i_rho]
    convert_hough_space_to_mask(edges, theta, rho, image.shape)


def draw_line(image, edges, theta, rho):
    """
    Draws a line on the image.
    """
    # get pixel coordinates

    new_lines = np.zeros(edges.shape, dtype=np.uint8)
    x = np.arange(edges.shape[1])
    y = ((rho - x * np.cos(theta)) / np.sin(theta)).round()

    to_draw = np.where(np.logical_and(y >= 0, y < edges.shape[0]))
    x, y = x[to_draw].astype(np.uint32), y[to_draw].astype(np.uint32)
    new_lines[y, x] = 1
    # new_lines = np.minimum(new_lines, edges)
    return np.maximum(image, new_lines)


def convert_hough_space_to_mask(edges, theta_axis, rho_axis, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)

    for theta in theta_axis:
        for rho in rho_axis:
            mask = draw_line(mask, edges, theta, rho)
    e = 0


if __name__ == '__main__':
    import imageio.v3 as imageio
    import matplotlib.pyplot as plt

    image = imageio.imread('../data/charuco_board.png')

    # plt.imshow(image)
    # plt.show()

    image = line_hough_transform(image)
    plt.imshow(image, cmap='gray')
    plt.show()


