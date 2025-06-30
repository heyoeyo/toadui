#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from base64 import b64decode, b64encode

import cv2
import numpy as np

# For type hints
from typing import Iterable
from numpy import ndarray
from toadui.helpers.types import COLORU8, IMGSHAPE_HW, HWPX, XYNORM, XY1XY2PX, XYPX


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


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


def draw_box_outline(image_uint8: ndarray, color: COLORU8 | None = (0, 0, 0), thickness=1) -> ndarray:
    """
    Helper used to draw a box outline around the outside of a given image.
    If the color given is None, then no box will be drawn
    (can be used to conditionally disable outline)
    Returns:
        image_with_outline_uint8
    """

    # Bail if no color or thickness
    if color is None or thickness <= 0:
        return image_uint8

    img_h, img_w = image_uint8.shape[0:2]
    x1, y1 = thickness - 1, thickness - 1
    x2, y2 = img_w - thickness, img_h - thickness

    # Technique for rendering border depends on the thickness, due to oddities
    # in the way opencv line drawing works
    if thickness < 4:
        # The width (in pixels) of lines drawn by opencv for thickness values
        # less than 4, follows the formula: line width (px) = 2*thickness - 1
        # The sizes are always odd, so we can get the target border thickness
        # by simply drawing lines with corners matching the image itself.
        x1, y1 = 0, 0
        x2, y2 = img_w - 1, img_h - 1
        cv2.rectangle(image_uint8, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_4)
    elif thickness < 8:
        # For thickness values 4 or greater, opencv follows a different
        # pattern of line widths, given by the formula:
        #   line width (px) = (thickness + 2 if odd else 1)
        # So for example, a thickness of 4 is also 5 pixels wide. A thickness
        # of 5 is drawn 7 pixels wide, thickness 9 is drawn 11 pixels wide etc.
        # So we need an extra offset to maintain correct border sizing.
        extra_offset = thickness // 2
        x1, y1 = extra_offset - 1, extra_offset - 1
        x2, y2 = img_w - extra_offset, img_h - extra_offset
        cv2.rectangle(image_uint8, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_4)
    else:
        # Opencv rounds the corners of rectangles, though this is only
        # noticable at higher thickness values. Here we switch to drawing
        # the border using filled rectangles to maintain consistent border
        # sizing (we don't default to doing this because it's slower).
        xl, xr = thickness - 1, img_w - thickness
        yt, yb = thickness - 1, img_h - thickness
        cv2.rectangle(image_uint8, (0, 0), (img_w, yt), color, -1, cv2.LINE_4)
        cv2.rectangle(image_uint8, (0, yb), (img_w, img_h), color, -1, cv2.LINE_4)
        cv2.rectangle(image_uint8, (0, 0), (xl, img_h), color, -1, cv2.LINE_4)
        cv2.rectangle(image_uint8, (xr, 0), (img_w, img_h), color, -1, cv2.LINE_4)

    return image_uint8


def draw_drop_shadow(
    image_uint8: ndarray,
    left=2,
    top=4,
    right=2,
    bottom=0,
    color: COLORU8 = (0, 0, 0),
    blur_strength: float = 3,
    blur_sharpness: float = 1,
) -> ndarray:

    # Bail if we're not shadowing
    if all(val <= 0 for val in [left, top, right, bottom, blur_strength]):
        return image_uint8

    # For convenience
    img_h, img_w = image_uint8.shape[0:2]
    x1, y1 = 0, 0
    x2, y2 = img_w - 1, img_h - 1
    xl, xr = x1 + left, x2 - right
    yt, yb = y1 + top, y2 - bottom

    # Draw lines around edges & blur to create drop-shadow effect
    shadow_img = blank_image_1ch(img_h, img_w, 255)
    if top > 0:
        cv2.rectangle(shadow_img, (x1, y1), (x2, yt), 0, -1)
    if left > 0:
        cv2.rectangle(shadow_img, (x1, y1), (xl, y2), 0, -1)
    if right > 0:
        cv2.rectangle(shadow_img, (xr, y1), (x2, y2), 0, -1)
    if bottom > 0:
        cv2.rectangle(shadow_img, (x1, yb), (x2, y2), 0, -1)
    shadow_img = np.float32(cv2.GaussianBlur(shadow_img, None, blur_strength)) * (1.0 / 255.0)

    # Scale input by shadow amount. Equivalent to a black (0,0,0) drop-shadow
    if blur_sharpness != 1:
        shadow_img = np.pow(shadow_img, blur_sharpness)
    shadow_img = np.expand_dims(shadow_img, axis=2)
    result_f32 = shadow_img * np.float32(image_uint8)

    # If we have a non-black shadow, then we treat the step above as a weighting
    # -> So add an inversely weighted copy of the color we want to get the final result
    is_black_shadow = all(col == 0 for col in color)
    if not is_black_shadow:
        full_color_img = np.float32(blank_image(img_h, img_w, color))
        result_f32 += (1.0 - shadow_img) * full_color_img
        result_f32 = np.clip(result_f32, 0, 255)

    return np.round(result_f32).astype(np.uint8)


def draw_normalized_polygon(
    image_uint8: ndarray,
    polygon_xy_norm_list: list[tuple[float, float]] | ndarray,
    color: COLORU8 = (0, 255, 255),
    thickness: int = 1,
    bg_color: COLORU8 | None = None,
    line_type: int = cv2.LINE_AA,
    is_closed: bool = True,
) -> ndarray:
    """
    Helper used to draw polygons from 0-to-1 normalized xy coordinates.
    Expects coordinates in the form:
        xy_norm = [(0,0), (0.5, 0), (0.5, 0.75), (1, 1), (0, 1), etc.]
    """

    # Force input to be an array to make normalization easier
    xy_norm_array = polygon_xy_norm_list
    if not isinstance(polygon_xy_norm_list, ndarray):
        xy_norm_array = np.float32(polygon_xy_norm_list)

    # Convert normalized xy into pixel units
    img_h, img_w = image_uint8.shape[0:2]
    norm_to_px_scale = np.float32((img_w - 1, img_h - 1))
    xy_px_array = np.int32(np.round(xy_norm_array * norm_to_px_scale))

    # Draw polygon with background if needed
    if bg_color is not None:
        bg_thick = max(0, thickness) + 1
        cv2.polylines(image_uint8, [xy_px_array], is_closed, bg_color, bg_thick, line_type)

    # Draw polygon outline, or filled in shape if using negative thickness value
    if thickness < 0:
        return cv2.fillPoly(image_uint8, [xy_px_array], color, line_type)
    return cv2.polylines(image_uint8, [xy_px_array], is_closed, color, thickness, line_type)


def draw_rectangle_norm(
    image_uint8: ndarray,
    xy1_norm: XYNORM,
    xy2_norm: XYNORM,
    color: COLORU8 = (0, 0, 0),
    thickness: int = -1,
    pad_xy1xy2_px: XY1XY2PX | XYPX = (0, 0),
    inset_outline: bool = True,
) -> ndarray:
    """
    Helper used to draw a rectangle onto an image, using normalized coordinates
    """

    pad_xy1, pad_xy2 = pad_xy1xy2_px
    img_h, img_w = image_uint8.shape[0:2]
    norm_to_px_scale = np.float32((img_w - 1, img_h - 1))
    x1_px, y1_px = np.int32(np.round(np.float32(xy1_norm) * norm_to_px_scale + np.float32(pad_xy1)))
    x2_px, y2_px = np.int32(np.round(np.float32(xy2_norm) * norm_to_px_scale - np.float32(pad_xy2)))

    if inset_outline and thickness > 1:
        inset_amt = thickness - 1
        x1_px, y1_px = x1_px + inset_amt, y1_px + inset_amt
        x2_px, y2_px = x2_px - inset_amt, y2_px - inset_amt

    pt1, pt2 = (x1_px, y1_px), (x2_px, y2_px)
    return cv2.rectangle(image_uint8, pt1, pt2, color, thickness)


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


def get_image_hw_to_fill(image_shape: IMGSHAPE_HW, target_hw: HWPX) -> tuple[int, int]:
    """
    Helper used to find the sizing (height & width) of a given image
    if it is scaled to fit in the target height & width, assuming the
    aspect ratio of the image is preserved.
    For example, to fit a 100x200 image into a 600x600 square space,
    while preserving aspect ratio, the image would be scaled to 300x600

    Returns:
        output_height, output_width
    """

    img_h, img_w = image_shape[0:2]
    targ_h, targ_w = target_hw

    scale = min(targ_h / img_h, targ_w / img_w)
    out_h = round(scale * img_h)
    out_w = round(scale * img_w)

    return out_h, out_w


def get_image_hw_for_max_height(image_shape: IMGSHAPE_HW, max_height_px: int = 800) -> tuple[int, int]:
    """
    Helper used to find the height & width of a given image if it
    is scaled to fit to a given target height, assuming the aspect
    ratio is preserved.
    For example, to fit a (HxW) 100x200 image to a max height of
    500, the image would be scaled to 500x1000

    Returns:
        output_height, output_width
    """

    img_h, img_w = image_shape[0:2]
    scale = max_height_px / img_h
    out_h = round(scale * img_h)
    out_w = round(scale * img_w)

    return out_h, out_w


def get_image_hw_for_max_width(image_shape: IMGSHAPE_HW, max_width_px: int = 800) -> tuple[int, int]:
    """
    Helper used to find the height & width of a given image if it
    is scaled to fit to a given target width, assuming the aspect
    ratio is preserved.
    For example, to fit a (HxW) 100x200 image to a max width of
    500, the image would be scaled to 250x500

    Returns:
        output_height, output_width
    """

    img_h, img_w = image_shape[0:2]
    scale = max_width_px / img_w
    out_h = round(scale * img_h)
    out_w = round(scale * img_w)

    return out_h, out_w


def get_image_hw_for_max_side_length(image_shape: IMGSHAPE_HW, max_side_length: int = 800) -> tuple[int, int]:
    """
    Helper used to find the height & width of a given image if it
    is scaled to a target max side length, assuming the aspect
    ratio is preserved.
    For example, to fit a (HxW) 100x200 image to a max side length
    of 500, the image would be scaled to 250x500

    Returns:
        output_height, output_width
    """

    img_h, img_w = image_shape[0:2]
    scale = min(max_side_length / img_w, max_side_length / img_h)
    out_h = round(scale * img_h)
    out_w = round(scale * img_w)

    return out_h, out_w


def pad_to_hw(
    image: ndarray,
    output_hw: HWPX,
    border_type: int = cv2.BORDER_CONSTANT,
    border_color: COLORU8 = (0, 0, 0),
    align_xy: XYNORM = (0.5, 0.5),
) -> ndarray:
    """
    Helper used to pad out an image to match a given output height & width.
    Uses opencv 'copyMakeBorder' internally and can used the border types
    of that function. See: cv2.BORDER_... constants.

    The 'align_xy' argument can be used to determine how padding is allocated.
    The default padding places the original image in the center, but alignment
    can be set to left/top-align (e.g align_xy = (0, 0)) for example.

    If the output height or width is smaller than the given image, then the
    image height or width will remain as-is (i.e. it won't be scaled/cropped)

    Returns:
        padded_image
    """

    # For convenience
    img_h, img_w = image.shape[0:2]
    out_h, out_w = output_hw[0:2]
    align_x, align_y = np.clip(align_xy, 0.0, 1.0).tolist()

    # Figure out how much total padding is needed
    available_h = max(0, out_h - img_h)
    available_w = max(0, out_w - img_w)

    # Split the top/bottom, left/right padding spacing, based on alignment
    pad_top, pad_left = int(available_h * align_y), int(available_w * align_x)
    pad_bot, pad_right = int(available_h - pad_top), int(available_w - pad_left)
    return cv2.copyMakeBorder(image, pad_top, pad_bot, pad_left, pad_right, border_type, value=border_color)


def scale_and_pad_to_fit_hw(
    image: ndarray,
    output_hw: IMGSHAPE_HW,
    interpolation_type: int = cv2.INTER_AREA,
    pad_border_type: int = cv2.BORDER_CONSTANT,
    pad_color: COLORU8 = (0, 0, 0),
    pad_align_xy: XYNORM = (0.5, 0.5),
) -> ndarray:
    """
    Helper function which scales a given image so that it fits inside a
    target height & width. If the original image aspect ratio does not match
    the target sizing, then the image will be padded to fit.
    """

    # Resize to fit inside target sizing
    out_h, out_w = output_hw
    scale_h, scale_w = get_image_hw_to_fill(image.shape, output_hw)
    out_image = cv2.resize(image, dsize=(scale_w, scale_h), interpolation=interpolation_type)

    # Pad to fit if needed
    scaled_h, scaled_w = out_image.shape[0:2]
    if scaled_h < out_h or scaled_w < out_w:
        out_image = pad_to_hw(out_image, output_hw, pad_border_type, pad_color, pad_align_xy)

    return out_image


def histogram_equalization(
    image_uint8: ndarray,
    min_pct: float = 0.0,
    max_pct: float = 1.0,
    channel_index: int | None = None,
) -> ndarray:
    """
    Function used to perform histogram equalization on a uint8 image.
    This function uses the built-in opencv function: cv2.equalizeHist(...)
    When the min/max thresholds are not set (since it works faster),
    however this implementation also supports truncating the low/high
    end of the input.

    This means that equalization can be performed over a subset of
    the input value range, which makes better use of the value range
    when using thresholded inputs.

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
    min_value, max_value = [int(round(255 * value)) for value in sorted((min_pct, max_pct))]
    max_value = max(max_value, min_value + 1)
    if min_value == 0 and max_value == 255:
        if image_uint8.ndim == 1:
            return cv2.equalizeHist(image_uint8).squeeze()
        elif image_uint8.ndim == 2:
            return cv2.equalizeHist(image_uint8)
        else:
            num_channels = image_uint8.shape[2]
            return np.dstack([image_uint8[:, :, c] for c in range(num_channels)])

    # Compute histogram of input
    num_bins = 1 + max_value - min_value
    bin_counts, _ = np.histogram(image_uint8, num_bins, range=(min_value, max_value))

    # Compute cdf of histogram counts
    cdf = bin_counts.cumsum()
    cdf_min, cdf_max = cdf.min(), cdf.max()
    cdf_norm = (cdf - cdf_min) / float(max(cdf_max - cdf_min, 1))
    cdf_uint8 = np.uint8(255 * cdf_norm)

    # Extend cdf to match 256 lut sizing, in case we skipped min/max value ranges
    low_end = np.zeros(min_value, dtype=np.uint8)
    high_end = np.full(255 - max_value, 255, dtype=np.uint8)
    equalization_lut = np.concatenate((low_end, cdf_uint8, high_end))

    # Apply LUT mapping to input
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
