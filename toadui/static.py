#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2

from toadui.base import BaseCallback
from toadui.helpers.styling import UIStyle
from toadui.helpers.text import TextDrawer
from toadui.helpers.images import blank_image
from toadui.helpers.sizing import get_image_hw_to_fit_region
from toadui.helpers.drawing import draw_box_outline
from toadui.helpers.colors import pick_contrasting_gray_color

# For type hints
from numpy import ndarray
from toadui.helpers.types import COLORU8


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class StaticImage(BaseCallback):

    # .................................................................................................................

    def __init__(self, image: ndarray, min_scale_factor: float = 0.05, max_scale_factor: float | None = None):

        # Store image for re-use when rendering
        image_3ch = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self._cached_img = image_3ch
        self._render_image = image_3ch.copy()
        self._targ_h = 0
        self._targ_w = 0

        # Set up sizing limits
        img_w, img_h = image.shape[0:2]
        min_h = int(img_h * min_scale_factor)
        min_w = int(img_w * min_scale_factor)
        super().__init__(min_h, min_w, is_flexible_h=True, is_flexible_w=True)

        # Record a max sizing, if needed
        self._max_w = 1_000_000 if max_scale_factor is None else round(max_scale_factor * img_w)
        self._max_h = 1_000_000 if max_scale_factor is None else round(max_scale_factor * img_h)

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        h, w = min(h, self._max_h), min(w, self._max_w)
        img_h, img_w = self._render_image.shape[0:2]
        if self._targ_h != h or self._targ_w != w:
            fill_h, fill_w = get_image_hw_to_fit_region(self._cached_img.shape, (h, w))
            self._render_image = cv2.resize(self._cached_img, dsize=(fill_w, fill_h))
            self._targ_h = h
            self._targ_w = w

        return self._render_image

    # .................................................................................................................

    def _get_width_given_height(self, h: int) -> int:
        h = min(h, self._max_h)
        img_h, img_w = self._cached_img.shape[0:2]
        scaled_w = round(img_w * h / img_h)
        return scaled_w

    def _get_height_given_width(self, w: int) -> int:
        w = min(w, self._max_w)
        img_h, img_w = self._cached_img.shape[0:2]
        scaled_h = round(img_h * w / img_w)
        return scaled_h

    def _get_height_and_width_without_hint(self) -> tuple[int, int]:
        img_h, img_w = self._cached_img.shape[0:2]
        return img_h, img_w

    # .................................................................................................................


class StaticMessageBar(BaseCallback):

    # .................................................................................................................

    def __init__(
        self,
        *messages: str,
        text_scale: float = 0.5,
        color: COLORU8 = (150, 110, 15),
        height: int = 40,
        space_equally: bool = False,
        is_flexible_w: bool = True,
    ):

        # Store messages with front/back padding for nicer spacing on display (and skip 'None' entries)
        self._msgs_list = [f" {msg}  " for msg in messages if msg is not None]

        # Store visual settings
        self._base_image = blank_image(1, 1, color)
        self._cached_img = self._base_image.copy()
        text_color = pick_contrasting_gray_color(color)

        # Make sure our text sizing fits in the given bar height
        text_draw = TextDrawer(scale=text_scale, color=text_color)
        txt_h, _, _ = text_draw.get_text_size("".join(self._msgs_list))
        if txt_h > height:
            new_scale = text_scale * (height / txt_h) * 0.8
            text_draw.style.scale = new_scale

        # Record message widths, used to assign space when minimum drawing size
        msg_widths = [text_draw.get_text_size(m)[1] for m in self._msgs_list]
        total_msg_w = sum(msg_widths)

        # Pre-compute the relative x-positioning of each message for display
        cumulative_w = [sum(msg_widths[:k]) for k in range(len(self._msgs_list))]
        self._msg_x_norms = [(cum_w + 0.5 * msg_w) / total_msg_w for cum_w, msg_w in zip(cumulative_w, msg_widths)]
        if space_equally:
            num_msgs = len(msg_widths)
            self._msg_x_norms = [(k + 0.5) / num_msgs for k in range(num_msgs)]
            total_msg_w = max(msg_widths) * num_msgs
        self._space_equal = space_equally

        # Set up element styling
        self.style = UIStyle(
            color=color,
            outline_color=(0, 0, 0),
            text=text_draw,
        )

        # Inherit from parent & render initial image to cache results
        super().__init__(height, total_msg_w, is_flexible_h=False, is_flexible_w=is_flexible_w)
        self.render(height, total_msg_w)

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-draw image when sizing has changed
        img_h, img_w = self._cached_img.shape[0:2]
        if img_h != h or img_w != w:
            msg_img = blank_image(h, w, self.style.color)
            for msg_str, x_norm in zip(self._msgs_list, self._msg_x_norms):
                msg_img = self.style.text.xy_norm(msg_img, msg_str, (x_norm, 0.5), anchor_xy_norm=(0.5, 0.5))
            msg_img = draw_box_outline(msg_img, self.style.outline_color)

            # Cache image for re-use
            self._cached_img = msg_img

        return self._cached_img

    # .................................................................................................................


class HSeparator(BaseCallback):

    # .................................................................................................................

    def __init__(
        self,
        width: int = 2,
        color: COLORU8 = (20, 20, 20),
        label: str | None = None,
        is_flexible_h: bool = True,
        is_flexible_w: bool = False,
    ):
        self._cached_img = blank_image(1, width, color)
        self._label = label
        self.style = UIStyle(
            color=color,
            text=None if label is None else TextDrawer(0.35, 1, pick_contrasting_gray_color(color)),
        )
        super().__init__(1, width, is_flexible_h=is_flexible_h, is_flexible_w=is_flexible_w)

    # .................................................................................................................

    @classmethod
    def many(cls, num_separators: int, width: int = 2, color: COLORU8 = (20, 20, 20)):
        return [cls(width, color) for _ in range(num_separators)]

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:
        img_h, img_w = self._cached_img.shape[0:2]
        if img_h != h or img_w != w:
            self._cached_img = blank_image(h, w, self.style.color)
            if self._label is not None:
                self._cached_img = self.style.text.xy_centered(self._cached_img, self._label)
        return self._cached_img

    # .................................................................................................................


class VSeparator(BaseCallback):

    # .................................................................................................................

    def __init__(
        self,
        height: int = 2,
        color: COLORU8 = (20, 20, 20),
        label: str | None = None,
        is_flexible_h: bool = False,
        is_flexible_w: bool = True,
    ):
        self._cached_img = blank_image(height, 1, color)
        self._label = label
        self.style = UIStyle(
            color=color,
            text=None if label is None else TextDrawer(0.35, 1, pick_contrasting_gray_color(color)),
        )
        super().__init__(height, 1, is_flexible_h=is_flexible_h, is_flexible_w=is_flexible_w)

    # .................................................................................................................

    @classmethod
    def many(cls, num_separators: int, height: int = 2, color: COLORU8 = (20, 20, 20)):
        return [cls(height, color) for _ in range(num_separators)]

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:
        img_h, img_w = self._cached_img.shape[0:2]
        if img_h != h or img_w != w:
            self._cached_img = blank_image(h, w, self.style.color)
            if self._label is not None:
                self._cached_img = self.style.text.xy_centered(self._cached_img, self._label)
        return self._cached_img

    # .................................................................................................................
