from typing import Tuple, Union

import numpy as np


def iou(box1: Union[np.ndarray, tuple, list], box2: Union[np.ndarray, tuple, list]) -> float:
    """
    Computes the intersection over union of two bounding boxes.
    :param box1: bounding box 1
    :param box2: bounding box 2
    :return: iou score
    """

    # get box coordinates
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    x31, y31, x32, y32 = max(x11, x21), max(y11, y21), min(x12, x22), min(y12, y22)

    # compute the area of all blocks
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    area_intersect = ((x32 - x31) * (y32 - y31)) if x32 > x31 and y32 > y31 else 0

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
    return np.dstack(np.unravel_index(np.argsort(grid.ravel(), **kwargs), grid.shape))[0]


def nms(grid_score: np.ndarray, grid_window: np.ndarray, skip_threshold: float = 1e-1, iou_threshold: float = 0.5):
    """
    Perform non-maximum suppression on a 2D grid.
    :param grid_score: pixel score
    :param grid_window: referral window
    :param skip_threshold: threshold for skipping detections
    :param iou_threshold: threshold for deciding whether to keep a detection
    :return: non-maximum suppressed grid of size MxNx2
    """
    assert grid_score.ndim == 2, "grid_score must be 2D"
    assert grid_score.shape[:2] == grid_window.shape[:2], "grid_score and grid_window must have the same shape"

    selected_coordinates, selected_bbox = [], []

    # loop all candidates in descending order (by score)
    candidates = argsort_2d_grid(grid_score)[::-1]
    candidates = candidates[np.where(np.asarray([grid_score[c[0], c[1]] for c in candidates]) >= skip_threshold)]
    for _, (i, j) in enumerate(candidates):
        # check IoU against previous selected maxima points
        new_box = grid_window[i, j]
        add_to_maximal = True
        for box in selected_bbox:
            if iou(box, new_box) > iou_threshold:
                add_to_maximal = False
                break

        # if IoU is small enough with respect to the previous selected maxima, add it to the list
        if add_to_maximal:
            selected_bbox.append(new_box)
            selected_coordinates.append([i, j])
    return np.asarray(selected_coordinates)
