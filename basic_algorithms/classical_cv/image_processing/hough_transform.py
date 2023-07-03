import numpy as np
import cv2
from skimage import color


def line_hough_transform(image, theta_range=None, d_range=None, n_theta=None, n_rho=None):

    # convert to grayscale if needed
    # if image.ndim == 3:
    #     image = color.rgb2gray(image)

    # set parameters
    height, width = image.shape[:2]
    d_range = [-np.sqrt(height**2 + width**2), np.sqrt(height**2 + width**2)] if d_range is None else d_range
    theta_range = [0, 2 * np.pi] if theta_range is None else theta_range
    n_theta = 16 if n_theta is None else n_theta
    n_rho = int(np.sqrt(height**2 + width**2) / 8) if n_rho is None else n_rho

    # hough space matrix
    theta_vals = np.linspace(theta_range[0], theta_range[1], n_theta)
    rho_vals = np.linspace(d_range[0], d_range[1], n_rho).astype(int)
    hough_space = np.zeros((n_theta, n_rho))

    # function to extract rho according to theta and pixel coordinates
    def _find_rho(x, y, theta, is_feature):
        return is_feature * x * np.cos(theta) + y * np.sin(theta)

    # arguments
    edges = cv2.Canny(image, 140, 160)
    image_coordinates = np.transpose(np.stack(np.meshgrid(np.arange(width), np.arange(height))), (1, 2, 0))
    is_feature = (edges > np.quantile(edges.flatten(), 0.8)).astype(np.uint8)

    for theta in theta_vals:
        theta_vals = np.ones_like(edges) * theta
        rho_vals = _find_rho(image_coordinates[:, :, 0], image_coordinates[:, :, 1], theta_vals, is_feature)

        EEE = rho_vals[rho_vals > 0], theta_vals[rho_vals > 0]

        e = 0


if __name__ == '__main__':
    import imageio.v3 as imageio
    import matplotlib.pyplot as plt

    image = imageio.imread('../data/coin.jpg')

    # plt.imshow(image)
    # plt.show()

    image = line_hough_transform(image)
    plt.imshow(image, cmap='gray')
    plt.show()


