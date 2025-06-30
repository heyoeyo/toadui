#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def draw_grid(
    image_uint8: ndarray | int | tuple[int, int],
    num_tiles: int = 10,
    line_color: tuple[int, int, int] = (255, 255, 255),
    line_thickness: int = 1,
    line_bg_color: None | tuple[int, int, int] = None,
    use_wraparound_sampling: bool = False,
) -> ndarray:
    """
    Helper used to draw a basic grid pattern onto an image.
    The num_tiles setting determines how many tiles will
    be placed along the shorter-axis of the image.
    """

    # Handle special case inputs where we interpret the input as a size for producing a blank background
    # -> If given a single integer, treat it as a square sizing
    # -> If given a tuple, treat it as a height/width sizing
    if isinstance(image_uint8, int):
        image_uint8 = (image_uint8, image_uint8)
    if isinstance(image_uint8, (tuple, list)):
        image_uint8 = np.zeros((image_uint8[0], image_uint8[1], 3), dtype=np.uint8)

    # Figure out tile sizing, based on area setting
    img_h, img_w = image_uint8.shape[0:2]
    tile_side_length = max(1, min(img_h, img_w) / num_tiles)

    # Figure out tile start/end drawing points
    x1, y1 = 0, 0
    x2, y2 = img_w - 1, img_h - 1
    if use_wraparound_sampling:
        half_step = tile_side_length * 0.5
        x1, x2 = x1 + half_step, x2 - half_step
        y1, y2 = y1 + half_step, y2 - half_step

    # Pre-determine all grid line x/y positions for drawing
    num_x_lines = int(round(img_w / tile_side_length)) + (0 if use_wraparound_sampling else 1)
    num_y_lines = int(round(img_h / tile_side_length)) + (0 if use_wraparound_sampling else 1)
    x_px_list = np.round(np.linspace(x1, x2, num_x_lines)).astype(np.int32).tolist()
    y_px_list = np.round(np.linspace(y1, y2, num_y_lines)).astype(np.int32).tolist()

    # Draw grid lines with/without background coloring as needed
    need_bg_lines = line_bg_color is not None
    bg_thickness = line_thickness + 1 + (line_thickness % 2) if line_thickness > 1 else 2
    thick_iter = [bg_thickness, line_thickness] if need_bg_lines else [line_thickness]
    color_iter = [line_bg_color, line_color] if need_bg_lines else [line_color]
    for color, thick in zip(color_iter, thick_iter):
        for x_px in x_px_list:
            pt1, pt2 = (x_px, -1), (x_px, img_h + 1)
            cv2.line(image_uint8, pt1, pt2, color, thick)
        for y_px in y_px_list:
            pt1, pt2 = (-1, y_px), (img_w + 1, y_px)
            cv2.line(image_uint8, pt1, pt2, color, thick)

    return image_uint8
