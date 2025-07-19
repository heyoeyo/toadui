#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2

from toadui.base import BaseCallback, CBEventXY, CBRenderSizing
from toadui.helpers.images import blank_image
from toadui.helpers.sizing import get_image_hw_to_fit_by_ar, get_image_hw_to_fit_region
from toadui.helpers.styling import UIStyle

# For type hints
from numpy import ndarray
from toadui.helpers.types import HWPX, SelfType
from toadui.helpers.ocv_types import OCVInterp


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class DynamicImage(BaseCallback):
    """Element used to hold images that may be updated (using the .set_image(...) function)"""

    # .................................................................................................................

    def __init__(
        self,
        image: ndarray | None = None,
        min_side_length: int = 128,
        resize_interpolation: OCVInterp = None,
        is_flexible_h: bool = True,
        is_flexible_w: bool = True,
    ):

        # Inherit from parent
        super().__init__(min_side_length, min_side_length, is_flexible_h, is_flexible_w)

        # Store sizing info
        self._min_side_length = min_side_length
        self._targ_h = -1
        self._targ_w = -1

        # Store state for mouse interaction
        self._is_clicked = False
        self._mouse_xy = CBEventXY.default()

        # Default to blank square image if given 'None' image input
        init_image = blank_image(min_side_length, min_side_length) if image is None else image
        self._full_image = None
        self._render_image = blank_image(1, 1)
        self.set_image(init_image)

        # Set up element styling
        self.style = UIStyle(interpolation=resize_interpolation)

    # .................................................................................................................

    def get_render_hw(self) -> HWPX:
        """
        Report the most recent render resolution of the image. This can be
        used to scale new images to match previous render sizes, which can
        help reduce jittering when giving images that repeatedly change size.
        Returns:
            render_height, render_width
        """
        return self._render_image.shape[0:2]

    def set_image(self, image: ndarray) -> SelfType:
        """Set the internally held image data"""
        self._full_image = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self._targ_h = -1
        self._targ_w = -1
        return self

    def read_mouse_xy(self) -> tuple[bool, CBEventXY]:
        """
        Read most recent mouse interaction, including whether the mouse was clicked.
        Returns:
            is_clicked, mouse_xy_event
        """
        is_clicked, self._is_clicked = self._is_clicked, False
        return is_clicked, self._mouse_xy

    def save(self, save_path: str) -> None:
        """Save current image data to the file system"""
        cv2.imwrite(save_path, self._full_image)

    # .................................................................................................................

    def _on_left_click(self, cbxy, cbflags) -> None:
        self._is_clicked = True
        self._mouse_xy = cbxy
        return

    def _on_move(self, cbxy, cbflags) -> None:
        self._mouse_xy = cbxy
        return

    # .................................................................................................................

    def _get_width_given_height(self, h: int) -> int:
        img_h, img_w = self._full_image.shape[0:2]
        scaled_w = max(self._cb_rdr.min_w, round(img_w * h / img_h))
        return scaled_w

    def _get_height_given_width(self, w: int) -> int:
        img_h, img_w = self._full_image.shape[0:2]
        scaled_h = max(self._cb_rdr.min_h, round(img_h * w / img_w))
        return scaled_h

    def _get_height_and_width_without_hint(self) -> HWPX:
        return self._full_image.shape[0:2]

    def _get_dynamic_aspect_ratio(self):
        is_flexible = self._cb_rdr.is_flexible_h and self._cb_rdr.is_flexible_h
        return self._full_image.shape[1] / self._full_image.shape[0] if is_flexible else None

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-render to target size if needed
        if self._targ_h != h or self._targ_w != w:
            self._targ_h = h
            self._targ_w = w

            # Scale image to fit within given sizing
            img_h, img_w = self._full_image.shape[0:2]
            scale = min(h / img_h, w / img_w)
            fill_wh = (round(scale * img_w), round(scale * img_h))
            scaled_image = cv2.resize(self._full_image, dsize=fill_wh, interpolation=self.style.interpolation)

            # Store rendered result for re-use
            self._render_image = scaled_image

        return self._render_image

    # .................................................................................................................


class StretchImage(DynamicImage):
    """
    Element used to hold images that can be updated and are meant to
    'stretch to fill' the space that they have available. By default,
    this class targets a specific aspect ratio for rendering, for example,
    stretching an image to fill a square (aspect ratio of 1). If the target
    aspect ratio is set to None, then the image aspect ratio will be used
    if there is space to do so, otherwise the image will stretch to
    fill whatever space is available.
    """

    # .................................................................................................................

    def __init__(
        self,
        image: ndarray,
        aspect_ratio: float | None = 1,
        min_h: int = 128,
        min_w: int = 128,
        resize_interpolation: OCVInterp = None,
        is_flexible_h=True,
        is_flexible_w=True,
    ):

        # Precompute aspect ratio settings for scaling
        self._has_ar = aspect_ratio is not None
        self._w_over_h = max(aspect_ratio, 0.001) if self._has_ar else -1
        self._h_over_w = 1.0 / self._w_over_h

        # Override parent sizing & styling
        super().__init__(image, resize_interpolation=resize_interpolation)
        self._cb_rdr = CBRenderSizing(min_h, min_w, is_flexible_h, is_flexible_w)

    # .................................................................................................................

    def _get_width_given_height(self, h: int) -> int:
        ar = self._w_over_h if self._has_ar else self._full_image.shape[1] / self._full_image.shape[0]
        return max(self._cb_rdr.min_w, round(h * ar))

    def _get_height_given_width(self, w: int) -> int:
        ar_inv = self._h_over_w if self._has_ar else self._full_image.shape[0] / self._full_image.shape[1]
        return max(self._cb_rdr.min_h, round(w * ar_inv))

    def _get_height_and_width_without_hint(self) -> HWPX:
        return self._full_image.shape[0:2]

    def _get_dynamic_aspect_ratio(self):
        return self._w_over_h if self._has_ar else None

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-render to target size if needed
        if self._targ_h != h or self._targ_w != w:
            self._targ_h = h
            self._targ_w = w
            self._render_image = cv2.resize(self._full_image, dsize=(w, h), interpolation=self.style.interpolation)

        return self._render_image

    # .................................................................................................................
