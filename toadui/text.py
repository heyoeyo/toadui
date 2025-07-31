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


class SubtitledTextBlock(CachedBgFgElement):
    """UI element that displays text with 2 lines. Meant to be used as a title block."""

    # .................................................................................................................

    def __init__(
        self,
        title: str,
        subtitle: str,
        color: COLORU8 | int = (64, 53, 52),
        text_scale: float = 0.5,
        height: int = 80,
        is_flexible_w: bool = True,
    ):

        # Storage for text being drawn (expected to be re-used)
        self._title_str: str = str(title)
        self._subtitle_str: str = str(subtitle)

        # Set up element styling
        half_height = height // 2
        text_title = TextDrawer(text_scale * 1.25, font=cv2.FONT_HERSHEY_DUPLEX, max_height=half_height)
        text_subtitle = TextDrawer(text_scale, max_height=half_height)
        self.style = UIStyle(
            color=interpret_coloru8(color),
            outline_color=(0, 0, 0),
            text_title=text_title,
            text_subtitle=text_subtitle,
        )

        # Set up element sizing
        ref_txt = f"  {title}  " if len(title) > len(subtitle) else f"  {subtitle}  "
        _, txt_w, _ = text_title.get_text_size(ref_txt)
        super().__init__(height, txt_w, is_flexible_h=False, is_flexible_w=is_flexible_w)

    # .................................................................................................................

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self._title_str}: {self._subtitle_str})"

    # .................................................................................................................

    def set_text(self, title: str | None = None, subtitle: str | None = None) -> SelfType:
        """Update text and/or subtitle. Inputs left as 'None' will not be altered"""
        if title is not None:
            self._title_str = str(title)
            self.request_fg_repaint()
        if subtitle is not None:
            self._subtitle_str = str(subtitle)
            self.request_fg_repaint()
        return self

    # .................................................................................................................

    def _rerender_bg(self, h: int, w: int) -> ndarray:
        return blank_image(h, w, self.style.color)

    def _rerender_fg(self, bg_image: ndarray) -> ndarray:

        # Re-draw title & subtitle text over background
        title_h, _, _ = self.style.text_title.get_text_size(self._title_str)
        self.style.text_title.xy_norm(bg_image, self._title_str, (0.5, 0.5), offset_xy_px=(0, -title_h))
        sub_h, _, _ = self.style.text_title.get_text_size(self._subtitle_str)
        self.style.text_subtitle.xy_norm(bg_image, self._subtitle_str, (0.5, 0.5), offset_xy_px=(0, sub_h))

        return draw_box_outline(bg_image, color=self.style.outline_color)

    # .................................................................................................................
