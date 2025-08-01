#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from dataclasses import dataclass
from base64 import b64decode, b64encode

import cv2
import numpy as np

# For type hints
from typing import Iterable
from numpy import ndarray
from toadui.helpers.types import COLORU8, IMGSHAPE_HW, XYNORM, XYPX, XY1XY2PX


# ---------------------------------------------------------------------------------------------------------------------
# %% Types


@dataclass
class CropData:
    """
    Container data used to hold information used to crop an image,
    along with some convenience functions.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def crop(self, frame: ndarray) -> ndarray:
        return frame[self.y1 : self.y2, self.x1 : self.x2]

    @property
    def xy1(self) -> tuple[int, int]:
        return (self.x1, self.y1)

    @property
    def xy2(self) -> tuple[int, int]:
        return (self.x2, self.y2)

    @property
    def xy1xy2(self) -> XY1XY2PX:
        return ((self.x1, self.y1), (self.x2, self.y2))

    def get_yx_slices(self) -> tuple[slice, slice]:
        """
        Get x/y indexing slices for cropping
        For example:
            y_slice, x_slice = crop_data.get_xy_slices()
            crop_img = image[y_slice, x_slice]
        """
        return slice(self.y1, self.y2), slice(self.x1, self.x2)

    @classmethod
    def from_xy1_xy2_norm(cls, image_shape: IMGSHAPE_HW, xy1_norm: XYNORM, xy2_norm: XYNORM):
        """
        Create crop data from normalized xy1 & xy2 coordinates (and a frame shape)
        Returns:
            new CropData
        """

        # For convenience
        img_h, img_w = image_shape[0:2]
        x_scale = img_w - 1
        y_scale = img_h - 1

        # Compute crop coords
        xy1_px = (round(xy1_norm[0] * x_scale), round(xy1_norm[1] * y_scale))
        xy2_px = (round(xy2_norm[0] * x_scale), round(xy2_norm[1] * y_scale))

        return cls(*xy1_px, *xy2_px)

    def is_valid(self) -> bool:
        """Check if crop data would produce a valid (e.g. non-empty) crop result"""
        return (self.x2 > self.x1) and (self.y2 > self.y1) and (self.x1 >= 0) and (self.y1 >= 0)


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def blank_image(height: int, width: int, bgr_color: None | int | COLORU8 = None) -> ndarray:
    """Helper used to create a blank image of a given size (and optionally provide a fill color)"""

    # If no color is given, default to zeros
    if bgr_color is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # If only 1 number is given for the color, duplicate it to form a gray value
    if isinstance(bgr_color, int):
        bgr_color = (bgr_color, bgr_color, bgr_color)

    return np.full((height, width, 3), bgr_color, dtype=np.uint8)


def blank_image_1ch(height: int, width: int, gray_value: int = 0) -> ndarray:
    """Helper used to create a blank mask (i.e. grayscale/no channels) of a given size"""
    return np.full((height, width), gray_value, dtype=np.uint8)


def adjust_image_gamma(image_uint8: ndarray, gamma: float | Iterable[float] = 1.0) -> ndarray:
    """
    Helper used to apply gamma correction to an entire image.
    If multiple gamma values are provided, they will be applied
    separately, per-channel, to the input image.
    Returns:
        gamma_corrected_image
    """

    # If we get multiple gamma values, assume we need to apply them per-channel
    if isinstance(gamma, Iterable):
        for ch_idx, ch_gamma in enumerate(gamma):
            image_uint8[:, :, ch_idx] = adjust_image_gamma(image_uint8[:, :, ch_idx], ch_gamma)
        return image_uint8

    # In case we get gamma of 1, do nothing (helpful for iterable case)
    if gamma == 1:
        return image_uint8
    return np.round(255 * np.pow(image_uint8.astype(np.float32) * (1.0 / 255.0), gamma)).astype(np.uint8)


def make_horizontal_gradient_image(
    image_hw: IMGSHAPE_HW,
    left_color: COLORU8 = (0, 0, 0),
    right_color: COLORU8 = (255, 255, 255),
) -> ndarray:
    h, w = image_hw[0:2]
    weight = np.linspace(0, 1, w, dtype=np.float32)
    weight = np.expand_dims(weight, axis=(0, 2))
    col_1px = (1.0 - weight) * np.float32(left_color) + weight * np.float32(right_color)
    return np.tile(col_1px.astype(np.uint8), (h, 1, 1))


def make_vertical_gradient_image(
    image_hw: IMGSHAPE_HW,
    top_color: COLORU8 = (0, 0, 0),
    bottom_color: COLORU8 = (255, 255, 255),
) -> ndarray:
    h, w = image_hw[0:2]
    weight = np.linspace(0, 1, h, dtype=np.float32)
    weight = np.expand_dims(weight, axis=(1, 2))
    col_1px = (1.0 - weight) * np.float32(top_color) + weight * np.float32(bottom_color)
    return np.tile(col_1px.astype(np.uint8), (1, w, 1))


def dirty_blur(
    image_uint8: ndarray,
    blur_strength: float = 2,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT,
) -> ndarray:
    """Function used to apply a blurring effect based on re-sampling"""

    img_hw = image_uint8.shape[0:2]
    yxsample = [np.linspace(0, s - 1, s, dtype=np.float32) for s in img_hw]
    ygrid, xgrid = np.meshgrid(*yxsample, indexing="ij")

    rad_grid = np.random.randn(*img_hw) * blur_strength
    ang_grid = np.random.randn(*img_hw) * (np.pi * 2.0)
    xgrid += np.cos(ang_grid) * rad_grid
    ygrid += np.sin(ang_grid) * rad_grid

    return cv2.remap(image_uint8, xgrid, ygrid, interpolation, borderMode=border_mode)


def convert_xy_norm_to_px(image_shape: IMGSHAPE_HW, *xy_norm_coords: XYNORM) -> tuple[XYPX] | XYPX:
    """
    Helper used to convert normalized xy coordinates to pixel coordinates,
    based on the provided image shape.

    Note that any number of input xy coordinates can be given, and the
    same number of coordinates will be returned by the function as a tuple.
    However, if a single xy coord is provided, then the function will output
    the converted coordinate directly (as opposed to a tuple of 1 xy coord).

    Returns:
        *xy_px_coords
    """

    # For convenience
    img_h, img_w = image_shape[0:2]
    x_scale = img_w - 1
    y_scale = img_h - 1

    # Compute crop coords
    xy_px_list = tuple((round(x * x_scale), round(y * y_scale)) for x, y in xy_norm_coords)

    return xy_px_list if len(xy_px_list) > 1 else xy_px_list[0]


def histogram_equalization(
    image_uint8: ndarray,
    min_pct: float = 0.0,
    max_pct: float = 1.0,
    channel_index: int | None = None,
) -> ndarray:
    """
    Function used to perform histogram equalization on a uint8 image.
    This function uses the built-in opencv function: cv2.equalizeHist(...)
    when using a (0, 1) min/max range.

    When a range other than (0,1) is given, the input
    will be equalized between this range.

    If a multi-channel image is provided, then each channel will
    be independently equalized!

    Returns:
        image_uint8_equalized
    """

    # If a channel index is given, equalize only that channel
    if channel_index is not None:
        result = image_uint8.copy()
        result[:, :, channel_index] = histogram_equalization(image_uint8[:, :, channel_index], min_pct, max_pct, None)
        return result

    # Make sure min/max are properly ordered & separated
    min_uint8, max_uint8 = [int(round(255 * value)) for value in sorted((min_pct, max_pct))]
    min_uint8, max_uint8 = (254, 255) if min_uint8 >= 254 else (min_uint8, max_uint8)
    min_uint8, max_uint8 = (0, 1) if max_uint8 <= 1 else (min_uint8, max_uint8)
    if min_uint8 <= 0 and max_uint8 >= 255:
        if image_uint8.ndim == 1:
            return cv2.equalizeHist(image_uint8).squeeze()
        elif image_uint8.ndim == 2:
            return cv2.equalizeHist(image_uint8)
        elif image_uint8.ndim == 3:
            num_channels = image_uint8.shape[2]
            return np.dstack([histogram_equalization(image_uint8[:, :, c]) for c in range(num_channels)])

    # Compute histogram of input
    h_bins = np.arange(257, dtype=np.int32)
    bin_counts, _ = np.histogram(image_uint8, h_bins)

    # Compute cumulative density function from histogram counts
    # -> Normalize cdf between min_pct & max_pct, so equalization applies to this range only!
    cdf = bin_counts.cumsum()
    cdf_min, cdf_max = cdf[0], cdf[-1]
    cdf_norm = min_pct + (max_pct - min_pct) * (cdf - cdf_min) / (cdf_max - cdf_min)

    # Apply LUT mapping to input
    equalization_lut = np.round(cdf_norm * 255).astype(np.uint8)
    return equalization_lut[image_uint8]


def encode_image_b64str(image: ndarray, file_extention: str = ".png") -> str:
    """Helper used to encode images in a string (base-64) format"""
    ok_encode, encoded_array = cv2.imencode(file_extention, image)
    assert ok_encode, f"Error while base-64 encoding ({file_extention}) image!"
    return b64encode(encoded_array).decode("utf-8")


def decode_b64str_image(b64_str: str) -> ndarray:
    """Helper used to decode images from a base-64 encoding"""
    decoded_bytes = b64decode(b64_str)
    decoded_array = np.frombuffer(decoded_bytes, np.uint8)
    return cv2.imdecode(decoded_array, cv2.IMREAD_UNCHANGED)
