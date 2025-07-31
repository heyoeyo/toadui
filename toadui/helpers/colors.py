#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

# For type hints
from typing import Iterable
from toadui.helpers.types import COLORU8


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def interpret_coloru8(color: COLORU8 | int | float | None, fallback_color=None) -> COLORU8 | None:
    """
    Helper used to interpret color inputs with convenience features
    - Supports float inputs (will be rounded to nearest integer in 0-to-255 range)
    - Supports single value inputs (will be converted to 3-tuple)
    - Supports fallback colors (e.g. if None is given, replace with fallback)

    Returns:
        color_3ch_uint8 (or None if no color or fallback)
    """

    # Try to replace missing color with fallback, if none available, bail
    out_color = color
    if color is None:
        out_color = fallback_color
    if out_color is None:
        return None

    # Convert to tuple of 3 integers
    if not isinstance(out_color, Iterable):
        out_color = (out_color, out_color, out_color)
    return tuple(int(min(255, max(0, round(val)))) for val in out_color)


# .....................................................................................................................


def convert_color(color: COLORU8, conversion_code: int) -> tuple[int, int, int]:
    """
    Helper used to convert singular color values, without requiring a full image
    For example:
        bgr_color = (12, 23, 34)
        hsv_color = convert_color(bgr_color, cv2.BGR2HSV_FULL)
        -> hsv_color = (21, 165, 34)
    """

    color_as_img = np.expand_dims(np.uint8(color), (0, 1))
    converted_color_as_img = cv2.cvtColor(color_as_img, conversion_code)

    return tuple(converted_color_as_img.squeeze().tolist())


# .....................................................................................................................


def scale_color_with_conversion(
    color: COLORU8,
    channel_scale_factors: tuple[float, float, float] = (1, 1, 1),
    conversion_to_code: int = cv2.COLOR_BGR2HSV_FULL,
    conversion_from_code: int = cv2.COLOR_HSV2BGR_FULL,
) -> COLORU8:
    """
    Helper used to adjust 3-channel color values by first converting to a different
    coor space, then scaling the channels, then converting to another color space
    (typically converting back to the original space)

    If a negative value is given, this is interpreted as meaning to invert the value
    (e.g. if the value is 20, flip it to 255 - 20 = 235), and then scale toward/away
    from the original value, depending on if the scale is above/below 1, respectively.
    """
    adj_color = []
    conv_color = convert_color(color, conversion_to_code)
    for col, scale in zip(conv_color, channel_scale_factors):
        new_col = col * scale if scale >= 0 else col + (255 - 2 * col) * abs(scale)
        new_col = max(0, min(255, round(new_col)))
        adj_color.append(new_col)
    return convert_color(adj_color, conversion_from_code)


def adjust_as_hsv(color_bgr: COLORU8, h=1, s=1, v=1) -> COLORU8:
    """Helper used to adjust a BGR color using HSV scaling values. Assumes input is in BGR format"""
    return scale_color_with_conversion(color_bgr, (h, s, v), cv2.COLOR_BGR2HSV_FULL, cv2.COLOR_HSV2BGR_FULL)


def adjust_as_hls(color_bgr: COLORU8, h=1, l=1, s=1) -> COLORU8:
    """Helper used to adjust a BGR color using HLS scaling values. Assumes input is in BGR format"""
    return scale_color_with_conversion(color_bgr, (h, l, s), cv2.COLOR_BGR2HLS_FULL, cv2.COLOR_HLS2BGR_FULL)


def adjust_as_lab(color_bgr: COLORU8, l=1, a=1, b=1) -> COLORU8:
    """Helper used to adjust a BGR color using HLS scaling values. Assumes input is in BGR format"""
    return scale_color_with_conversion(color_bgr, (l, a, b), cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR)


# .....................................................................................................................


def adjust_gamma(color: COLORU8, gamma: float | tuple[float, float, float] = 1) -> COLORU8:
    """
    Apply gamma correction to a given color value (e.g. result = color ^ gamma)

    Note that the gamma value can be provided as a 3-channel value,
    so that each color component (e.g. BGR) has a different gamma correction!
    -> This is mainly meant for alternate color spaces

    Returns: gamma_correct_color_uint8
    """

    # Force gamma to be 3-channel value
    if isinstance(gamma, (float, int)):
        gamma = [float(gamma)] * 3

    color_float = [(col / 255.0) ** gam for col, gam in zip(color, gamma)]
    return tuple(round(col * 255) for col in color_float)


# .....................................................................................................................


def lerp_colors(color_a: COLORU8, color_b: COLORU8, b_weight: float | tuple[float, float, float] = 0.5) -> COLORU8:
    """
    Linear interpolation applied between two 3-channel color values.
    The b_weight decides how much of color_b should be in the output.
    If a 3-tuple is given for b_weight, then each channel will be interpolated
    based on the corresponding 3-tuble of weights.

    Returns: interpolated_color_uint8
    """

    # Convert to 3-channel mix value if we don't get a listing
    if isinstance(b_weight, (float, int)):
        b_weight = (b_weight, b_weight, b_weight)

    mix_color = (round(a * (1 - t) + b * t) for a, b, t in zip(color_a, color_b, b_weight))
    return tuple(min(255, max(0, ch)) for ch in mix_color)


# .....................................................................................................................


def pick_contrasting_gray_color(
    color_bgr: COLORU8,
    contrast: float = 1,
    color_lerp_weight: float = 0,
    threshold_pt=0.5,
) -> COLORU8:
    """
    Helper used to pick 'high contrast' gray value, given a reference color.
    This is particularly useful for selecting colors for text that sits on
    top of arbitrary background colors.

    - The contrast setting can be used to pick a gray value that is
      less 'extreme' compared to the input. A contrast of 0 will
      match the gray value of the input color.
    - The color_lerp_weight setting can be used to mix in some amount
      of the input color, giving a potentionally non-gray value.
      This can be helpful for giving a 'grayed out' version of a color.
    - The threshold_pt setting determines how the opposing gray value
      is chosen. Lower values will favour darker grays and vice versa.

    Returns: contrasting_color_uint8
    """

    old_l, _, _ = convert_color(color_bgr, cv2.COLOR_BGR2LAB)
    new_l = 0 if old_l > 255 * threshold_pt else 255
    new_l = new_l * contrast + (1 - contrast) * old_l
    new_l = max(0, min(255, round(new_l)))

    # Interpolate back towards input color, if needed
    new_color = convert_color((new_l, 128, 128), cv2.COLOR_LAB2BGR)
    if color_lerp_weight > 0:
        new_color = lerp_colors(new_color, color_bgr, color_lerp_weight)

    return new_color
