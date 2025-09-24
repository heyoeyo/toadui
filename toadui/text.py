#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2

from toadui.base import CachedBgFgElement
from toadui.helpers.styling import UIStyle
from toadui.helpers.text import TextDrawer, find_minimum_text_width
from toadui.helpers.images import blank_image
from toadui.helpers.drawing import draw_box_outline
from toadui.helpers.colors import interpret_coloru8, pick_contrasting_gray_color

# For type hints
from numpy import ndarray
from toadui.helpers.types import COLORU8, SelfType


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class TextBlock(CachedBgFgElement):
    """UI element used to display text. The text can be modified using .set_text(...)"""

    # .................................................................................................................

    def __init__(
        self,
        text: str = "",
        color: COLORU8 | int = (30, 25, 25),
        text_scale: float = 0.35,
        max_characters: int = 8,
        height: int = 40,
        is_flexible_w: bool = True,
    ):

        # Set up text drawing config
        text_str = str(text)
        self._curr_text = text_str
        self._prev_value = None
        self._value_txtdraw = TextDrawer(text_scale)

        # Set up element styling
        color = interpret_coloru8(color)
        fg_color = pick_contrasting_gray_color(color)
        txtdraw = TextDrawer(scale=text_scale, color=fg_color, max_height=height)
        self.style = UIStyle(
            color=color,
            text=TextDrawer(scale=text_scale, color=fg_color, max_height=height),
            text_align_xy=(0.5, 0.5),
            text_offset_xy_px=(0, 0),
            outline_color=(0, 0, 0),
        )

        # Set up element sizing
        txt_w = find_minimum_text_width(txtdraw, max_characters)
        super().__init__(height, txt_w, is_flexible_h=False, is_flexible_w=is_flexible_w)

    # .................................................................................................................

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self._curr_text})"

    # .................................................................................................................

    def set_text(self, text: str) -> SelfType:
        if text != self._prev_value:
            self._prev_value = text
            self._curr_text = str(text)
            self.request_fg_repaint()
        return self

    # .................................................................................................................

    def _rerender_bg(self, h: int, w: int) -> ndarray:
        return blank_image(h, w, self.style.color)

    def _rerender_fg(self, bg_image: ndarray) -> ndarray:
        self.style.text.xy_norm(
            bg_image,
            self._curr_text,
            self.style.text_align_xy,
            offset_xy_px=self.style.text_offset_xy_px,
        )
        return draw_box_outline(bg_image, color=self.style.outline_color)

    # .................................................................................................................


class PrefixedTextBlock(TextBlock):
    """
    UI element used to display text with (typically unchanging) prefix/suffix components.
    This can be useful for providing labeled values.
    """

    # .................................................................................................................

    def __init__(
        self,
        prefix: str = "Label: ",
        initial_value: str = "-",
        suffix: str = "",
        color: COLORU8 | int = (30, 25, 25),
        text_scale: float = 0.35,
        max_characters: int | str = 8,
        height: int = 40,
        is_flexible_w: bool = True,
    ):

        # Set up text drawing config
        self._prefix = str(prefix)
        self._suffix = str(suffix)
        self._prev_value = None
        self._curr_text = ""

        # Convert max character strings to character count
        if isinstance(max_characters, str):
            max_characters = len(max_characters)

        spacer_txt = "*" * max_characters
        init_txt = f"{prefix}{spacer_txt}{suffix}"
        adjusted_max_characters = len(prefix) + max_characters + len(suffix)
        super().__init__(init_txt, color, text_scale, adjusted_max_characters, height, is_flexible_w)
        self.set_text(initial_value)

    # .................................................................................................................

    def set_prefix_suffix(self, new_prefix: str | None = None, new_suffix: str | None = None) -> SelfType:
        """Update prefix and/or suffix. Inputs left as 'None' won't be modified"""
        if new_prefix is not None:
            self._prefix = str(new_prefix)
            self._update_reported_text()
        if new_suffix is not None:
            self._suffix = str(new_suffix)
            self._update_reported_text()
        return self

    def set_text(self, text: str) -> SelfType:
        """Update reported value"""
        if text != self._prev_value:
            self._prev_value = text
            self._update_reported_text()
        return self

    def _update_reported_text(self) -> None:
        self._curr_text = f"{self._prefix}{self._prev_value}{self._suffix}"
        self.request_fg_repaint()
        return None

    # .................................................................................................................


class TwoLineTextBlock(CachedBgFgElement):
    """UI element that displays text with 2 lines, one above the other"""

    # .................................................................................................................

    def __init__(
        self,
        line_1: str,
        line_2: str,
        color_l1: COLORU8 | int = (160, 160, 160),
        color_l2: COLORU8 | int = (255, 255, 255),
        color_bg: COLORU8 | int = (64, 53, 52),
        l1_text_scale: float = 0.35,
        l2_text_scale: float = 0.35,
        height: int = 50,
        minimum_width: int | None = None,
        is_flexible_w: bool = True,
    ):

        # Storage for text being drawn (expected to be re-used)
        self._l1_str: str = str(line_1)
        self._l2_str: str = str(line_2)

        # Set up element styling
        max_l1_h = round(height * l1_text_scale / (l1_text_scale + l2_text_scale))
        max_l2_h = height - max_l1_h
        text_l1 = TextDrawer(l1_text_scale, max_height=max_l1_h, color=color_l1)
        text_l2 = TextDrawer(l2_text_scale, max_height=max_l2_h, color=color_l2)
        self.style = UIStyle(
            color=interpret_coloru8(color_bg),
            outline_color=(0, 0, 0),
            text_l1=text_l1,
            text_l2=text_l2,
            text_align_xy_l1=(0, 0.5),
            text_align_xy_l2=(0, 0.5),
            text_anchor_xy_l1=None,
            text_anchor_xy_l2=None,
            text_offset_xy_px_l1=(5, 0),
            text_offset_xy_px_l2=(5, 0),
        )

        # Set up element sizing
        if minimum_width is None:
            _, txt1_w, _ = text_l1.get_text_size(f"--{line_1}--")
            _, txt2_w, _ = text_l2.get_text_size(f"--{line_2}--")
            minimum_width = max(txt1_w, txt2_w)
        super().__init__(height, minimum_width, is_flexible_h=False, is_flexible_w=is_flexible_w)

    # .................................................................................................................

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self._l1_str}: {self._l2_str})"

    # .................................................................................................................

    def set_text(self, line_2: str | None = None, line_1: str | None = None) -> SelfType:
        """Update text. Inputs left as 'None' will not be altered"""
        if line_2 is not None:
            self._l2_str = str(line_2)
            self.request_fg_repaint()
        if line_1 is not None:
            self._l1_str = str(line_1)
            self.request_fg_repaint()
        return self

    # .................................................................................................................

    def _rerender_bg(self, h: int, w: int) -> ndarray:
        return blank_image(h, w, self.style.color)

    def _rerender_fg(self, bg_image: ndarray) -> ndarray:

        # For convenience
        l1_xy_off = self.style.text_offset_xy_px_l1
        l2_xy_off = self.style.text_offset_xy_px_l2
        l1_anc = self.style.text_anchor_xy_l1
        l2_anc = self.style.text_anchor_xy_l2

        # Figure out offset y-positioning (to get 2-line effect)
        l1_x, l1_y = self.style.text_align_xy_l1
        l2_x, l2_y = self.style.text_align_xy_l2
        l1_scale = self.style.text_l1.style.scale
        l2_scale = self.style.text_l2.style.scale
        l1_weight = l1_scale / (l1_scale + l2_scale)
        l2_weight = 1.0 - l1_weight
        l1_yf = l1_y * l1_weight
        l2_yf = (l2_y * l2_weight) + l1_weight

        # Re-draw line 1 & line 2 text, offset from center
        self.style.text_l1.xy_norm(bg_image, self._l1_str, (l1_x, l1_yf), l1_anc, l1_xy_off)
        self.style.text_l2.xy_norm(bg_image, self._l2_str, (l2_x, l2_yf), l2_anc, l2_xy_off)

        return draw_box_outline(bg_image, color=self.style.outline_color)

    # .................................................................................................................
