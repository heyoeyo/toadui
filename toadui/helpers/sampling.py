#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def make_xy_coordinate_mesh(
    mesh_hw: int | tuple[int, int],
    xy1: tuple[float, float] = (0, 0),
    xy2: tuple[float, float] = (1, 1),
    use_wraparound_sampling=False,
    dtype=np.float32,
) -> ndarray:
    """
    Returns a grid of xy coordinate values (between the given xy1/xy2 values).
    The shape of the result is: HxWx2
    -> Where H, W are from the given mesh_hw
    -> The '2' at the end corresponds to the (x,y) coordinate values at each 'pixel'
    """

    # Support providing height/width as a single value (interpret as square sizing)
    if isinstance(mesh_hw, (int, float)):
        mesh_hw = round(mesh_hw)
        mesh_hw = (mesh_hw, mesh_hw)
    if isinstance(use_wraparound_sampling, (int, bool)):
        use_wraparound_sampling = (bool(use_wraparound_sampling), bool(use_wraparound_sampling))

    # For convenience
    mesh_h, mesh_w = np.int32(mesh_hw[0:2])
    x1, y1 = xy1
    x2, y2 = xy2
    wraparound_x, wrap_around_y = use_wraparound_sampling

    # With wrap-around sampling endpoints are chosen so that first/last samples are 'one step' apart
    # -> For example, no-wrap indexing: [0, 0.5, 1] vs. wrap indexing: [0.167, 0.5, 0.833]
    if wraparound_x:
        half_x_step = 0.5 * (x2 - x1) / mesh_w
        x1 = x1 + half_x_step
        x2 = x2 - half_x_step
    if wrap_around_y:
        half_y_step = 0.5 * (y2 - y1) / mesh_h
        y1 = y1 + half_y_step
        y2 = y2 - half_y_step

    # Form xy mesh
    x_idx = np.linspace(x1, x2, mesh_w, dtype=dtype)
    y_idx = np.linspace(y1, y2, mesh_h, dtype=dtype)
    return np.dstack(np.meshgrid(x_idx, y_idx, indexing="xy"))


def make_xy_complex_mesh(
    mesh_hw: int | tuple[int, int],
    xy1: tuple[float, float] = (0, 0),
    xy2: tuple[float, float] = (1, 1),
    use_wraparound_sampling=False,
    dtype=np.complex64,
) -> ndarray:
    """
    Variation of creating an xy mesh. In this case, the xy coordinates are
    encoded as a single '2D' matrix, but each entry is a complex number
    of the form: x + iy
    -> Where x,y are the xy mesh coordinates
    """
    x1, y1 = xy1
    x2, y2 = xy2
    xy_mesh = make_xy_coordinate_mesh(mesh_hw, (x1, y1 * 1j), (x2, y2 * 1j), use_wraparound_sampling, dtype)
    return np.sum(xy_mesh, axis=2)


def resample_with_xy_mesh(
    image_uint8: ndarray,
    xy_mesh: ndarray,
    xy1=(0, 0),
    xy2=(1, 1),
    border=cv2.BORDER_REFLECT,
    interpolation=cv2.INTER_LINEAR,
) -> ndarray:

    img_h, img_w = image_uint8.shape[0:2]
    x1, y1 = xy1
    x2, y2 = xy2

    xmap = (img_w - 1) * (xy_mesh[:, :, 0] - x1) / (x2 - x1)
    ymap = (img_h - 1) * (xy_mesh[:, :, 1] - y1) / (y2 - y1)
    return cv2.remap(image_uint8, xmap, ymap, interpolation, borderMode=border)


def resample_with_complex_mesh(
    image_uint8: ndarray,
    complex_mesh: ndarray,
    xy1=(0, 0),
    xy2=(1, 1),
    border=cv2.BORDER_REFLECT,
    interpolation=cv2.INTER_LINEAR,
) -> ndarray:
    """
    Re-samples an image based on a grid of xy coordinates,
    represented as a single complex-valued 2D matrix.
    Each value in the matrix is assumed to be an xy
    coordinate as a complex number: z = x + iy

    For each entry in the grid, the complex value is used to
    'look up' the RGB color value of the given image, which is
    then painted into the corresponding grid position.

    Returns: resampled_image_uint8
    -> The output will have the same size as the complex mesh
    """
    img_h, img_w = image_uint8.shape[0:2]
    x1, y1 = xy1
    x2, y2 = xy2

    xmap = (img_w - 1) * (complex_mesh.real - x1) / (x2 - x1)
    ymap = (img_h - 1) * (complex_mesh.imag - y1) / (y2 - y1)
    return cv2.remap(image_uint8, xmap, ymap, interpolation, borderMode=border)


# ---------------------------------------------------------------------------------------------------------------------
# %% SDFs


def pointwise_minimum_of_many(*sdfs: ndarray) -> ndarray:
    """
    Helper used to compute the minimum of many arrays.
    Note that this preserves the shape of the inputs!

    This function behaves as if the np.minimum(...)
    function accepted an arbitrary number of inputs:
        np.minimum(arr1, arr2, arr3, arr4, ... etc)

    This can be used to 'union' together multiple sdfs

    Returns:
        minimum_of_inputs
    """

    if len(sdfs) == 1:
        return sdfs[0]

    min_result = np.minimum(sdfs[0], sdfs[1])
    if len(sdfs) == 2:
        return min_result

    for remaining_sdf in sdfs[2:]:
        min_result = np.minimum(min_result, remaining_sdf)
    return min_result


def sdf_circle(mesh_xy: ndarray, xy_center: tuple[float, float], radius: float) -> ndarray:
    """
    Helper used to calculate the signed-distance-function (SDF) for a circle.
    This can be used to help draw 'perfect' circles of arbitrary radius,
    unlike the built-in opencv drawing function.

    Notes:
    - The mesh_xy is expected to be a HxWx2 array, containing xy coordinate values
    - Units for the center/radius are based on the provided xy mesh
    - This does not 'draw' a circle into an image, it only computes the sdf

    Returns:
        sdf_of_circle
    """

    return np.linalg.norm(mesh_xy - np.float32(xy_center), ord=2, axis=2) - radius


def smoothstep(edge0: float, edge1: float, x: ndarray | float) -> ndarray | float:
    """CPU implementation of common shader function"""
    x = (x - edge0) / (edge1 - edge0)
    x = np.clip(x, 0, 1)
    return x * x * (3.0 - 2.0 * x)


def lerp_cos(a: float, b: float, t: ndarray | float) -> ndarray | float:
    weight = 0.5 - 0.5 * np.cos(t * np.pi)
    return a * (1.0 - weight) + b * weight
