#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from toadui.helpers.images import blank_image
from toadui.helpers.colors import pick_contrasting_gray_color

# For type hints
from numpy import ndarray
from toadui.helpers.types import COLORU8


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def draw_play_pause_icons(
    image_hw,
    color: COLORU8 = (60, 60, 225),
    color_symbol: COLORU8 | None = None,
) -> tuple[ndarray, ndarray]:
    """
    Helper used to draw the conventional triangle (▶) and bar (⏸︎) icons
    Returns:
        triangle_icon, bar_icon
    """

    # Fill in default color if missing
    if color_symbol is None:
        color_symbol = pick_contrasting_gray_color(color)

    # Figure out available drawing space
    img_h, img_w = image_hw[0:2]
    small_side = min(img_h, img_w)
    pad = max(10, small_side // 2)
    avail_side = small_side - pad
    half_avail = avail_side // 2

    # Figure out shape boundaries (done a bit strangely to force centering)
    x_mid, y_mid = (img_w - 1) / 2, (img_h - 1) / 2
    x1, y1 = [round(val - 0.25 - half_avail) for val in (x_mid, y_mid)]
    x2, y2 = [round(val + 0.25 + half_avail) for val in (x_mid, y_mid)]

    # Draw right-pointing triangle for play state
    triangle_img = blank_image(img_h, img_w, color)
    poly_px = [(x1, y1), (x2, round(y_mid)), (x1, y2)]
    cv2.fillConvexPoly(triangle_img, np.int32(poly_px), color_symbol, cv2.LINE_AA)

    # Draw 'double bars' for pause state
    bar_img = blank_image(img_h, img_w, color)
    barw = round(avail_side / 3)
    pt1, pt2 = (x1, y1), (x1 + barw, y2)
    pt3, pt4 = (x2 - barw, y1), (x2, y2)
    cv2.rectangle(bar_img, pt1, pt2, color_symbol, -1)
    cv2.rectangle(bar_img, pt3, pt4, color_symbol, -1)

    return triangle_img, bar_img
