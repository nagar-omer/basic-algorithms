import numpy as np


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Computes the intersection over union of two bounding boxes.
    :param box1: bounding box 1
    :param box2: bounding box 2
    :return: iou score
    """

    # get box coordinates
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    x31, y31, x32, y32 = max(x12, x21), max(y11, y21), min(x12, x22), min(y12, y22)

    # compute the area of all blocks
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    area_intersect = max(0, (x32 - x31) * (y32 - y31))

    # compute the intersection over union
    return area_intersect / (area1 + area2 - area_intersect)


def argsort_2d_grid(grid: np.ndarray, **kwargs) -> np.ndarray:
    """
    Perform argsort of a 2D grid.
    :param grid: grid to perform argsort on
    :param kwargs: additional arguments to pass to np.argsort
    :return: argsorted grid of size MxNx2
    """
    assert grid.ndim == 2
    return np.dstack(np.unravel_index(np.argsort(grid.ravel(), **kwargs), grid.shape))


def nms(grid_score: np.ndarray, grid_window: np.ndarray):
    """
    Perform non-maximum suppression on a 2D grid.
    :param grid_score: pixel score
    :param grid_window: referral window
    :return: non-maximum suppressed grid of size MxNx2
    """
    assert grid_score.ndim == 2, "grid_score must be 2D"
    assert grid_score.shape[:2] == grid_window.shape[:2], "grid_score and grid_window must have the same shape"

    return np.dstack(np.unravel_index(np.argsort(grid.ravel(), kind='mergesort'), grid.shape))