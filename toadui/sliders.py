#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from toadui.base import CachedBgFgElement
from toadui.helpers.styling import UIStyle
from toadui.helpers.colors import pick_contrasting_gray_color, lerp_colors
from toadui.helpers.text import TextDrawer
from toadui.helpers.images import blank_image, draw_box_outline

# Typing
from numpy import ndarray
from toadui.helpers.types import COLORU8, SelfType


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class Slider(CachedBgFgElement):
    """
    Simple horizontal slider. Intended to replace built-in opencv trackbars
    """

    # .................................................................................................................

    def __init__(
        self,
        label: str,
        value: float = 0.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        step: float = 0.05,
        color: COLORU8 = (40, 40, 40),
        indicator_width: int = 1,
        text_scale: float = 0.5,
        marker_step: float | None = None,
        enable_value_display: bool = True,
        height: int = 40,
        minimum_width: int = 64,
    ):

        # Make sure the given values make sense
        min_val, max_val = sorted((min_val, max_val))
        initial_value = min(max_val, max(min_val, value))

        # Storage for slider value
        self._label = label
        self._initial_value = initial_value
        self._slider_value = initial_value
        self._slider_min = min_val
        self._slider_max = max_val
        self._slider_step = step
        self._slider_delta = max(self._slider_max - self._slider_min, 1e-9)
        self._marker_x_norm = _get_norm_marker_positions(min_val, max_val, marker_step)
        self._max_precision = _get_step_precision(step)

        # Storage for slider state
        self._is_changed = True
        self._enable_value_display = enable_value_display

        # Set up text drawing
        txt_h = height * 0.8
        fg_color = pick_contrasting_gray_color(color)
        fg_text = TextDrawer(scale=text_scale, color=fg_color, max_height=txt_h)
        bg_text = TextDrawer(scale=text_scale, color=lerp_colors(fg_color, color, 0.55), max_height=txt_h)

        # Set up element styling
        self.style = UIStyle(
            color=color,
            indicator_width=indicator_width,
            indicator_color=fg_color,
            marker_color=lerp_colors(fg_color, color, 0.85),
            marker_width=1,
            marker_pad=5,
            outline_color=(0, 0, 0),
            fg_text=fg_text,
            bg_text=bg_text,
        )

        # Inherit from parent
        _, label_w, _ = fg_text.get_text_size(self._label)
        min_w = max(label_w, minimum_width)
        super().__init__(height, min_w, is_flexible_h=False, is_flexible_w=True)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name} ({self._label}"

    # .................................................................................................................

    def read(self) -> tuple[bool, float | int]:
        """Read current slider value. Returns: is_changed, slider_value"""
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._slider_value

    def set(self, slider_value: int | float, use_as_default_value: bool = True) -> SelfType:
        new_value = max(self._slider_min, min(self._slider_max, slider_value))
        if use_as_default_value:
            self._initial_value = new_value
        self._is_changed |= new_value != self._slider_value
        self._slider_value = new_value
        self.request_fg_repaint()
        return self

    def reset(self) -> SelfType:
        self.set(self._initial_value, use_as_default_value=False)
        return self

    def increment(self, num_increments: int = 1) -> SelfType:
        return self.set(self._slider_value + self._slider_step * num_increments, use_as_default_value=False)

    def decrement(self, num_decrements: int = 1) -> SelfType:
        return self.set(self._slider_value - self._slider_step * num_decrements, use_as_default_value=False)

    def set_is_changed(self, is_changed: bool = True) -> SelfType:
        """Helper used to artificially toggle is_changed flag, useful for forcing read updates (e.g. on startup)"""
        self._is_changed = is_changed
        return self

    # .................................................................................................................

    def _on_left_down(self, cbxy, cbflags) -> None:

        # Ignore clicks outside of the slider
        if not cbxy.is_in_region:
            return

        # Update slider as if dragging
        self._on_drag(cbxy, cbflags)
        return

    def _on_drag(self, cbxy, cbflags) -> None:

        # Update slider value while dragging
        new_slider_value = self._mouse_x_norm_to_slider_value(cbxy.xy_norm[0])
        value_is_changed = new_slider_value != self._slider_value
        if value_is_changed:
            self._is_changed = True
            self._slider_value = new_slider_value
            self.request_fg_repaint()

        return

    def _on_right_click(self, cbxy, cbflags) -> None:
        self.reset()
        return

    # .................................................................................................................

    def _mouse_x_norm_to_slider_value(self, x_norm: float) -> float | int:
        """Helper used to convert normalized mouse position into slider values"""

        # Map normalized x position to slider range, snapped to step increments
        slider_x = (x_norm * self._slider_delta) + self._slider_min
        slider_x = round(slider_x / self._slider_step) * self._slider_step

        # Finally, make sure the slider value doesn't go out of range
        return max(self._slider_min, min(self._slider_max, slider_x))

    # .................................................................................................................

    def _rerender_bg(self, h: int, w: int) -> ndarray:

        # Draw marker lines
        new_img = blank_image(h, w, self.style.color)
        mrk_pad_offset = 1 + self.style.marker_pad
        for x_norm in self._marker_x_norm:
            x_px = round(w * x_norm)
            pt1, pt2 = (x_px, mrk_pad_offset), (x_px, h - 1 - mrk_pad_offset)
            cv2.line(new_img, pt1, pt2, self.style.marker_color, self.style.marker_width, cv2.LINE_4)

        # Draw label
        return self.style.bg_text.xy_norm(new_img, self._label, (0, 0.5), offset_xy_px=(5, 0))

    def _rerender_fg(self, bg_image: ndarray) -> ndarray:

        # Get image sizing for norm-to-px conversions
        img_h, img_w = bg_image.shape[0:2]
        max_x_px = img_w - 1

        # Draw indicator line
        slider_norm = (self._slider_value - self._slider_min) / self._slider_delta
        line_x_px = round(slider_norm * max_x_px)
        pt1, pt2 = (line_x_px, 0), (line_x_px, img_h)
        new_img = cv2.line(bg_image, pt1, pt2, self.style.indicator_color, self.style.indicator_width)

        # Draw text beside indicator line to show current value if needed
        if self._enable_value_display:
            value_str = f"{float(self._slider_value):.{self._max_precision}f}"
            _, txt_w, _ = self.style.fg_text.get_text_size(value_str)

            # Draw the text to the left or right of the indicator line, depending on where the image border is
            is_near_right_edge = line_x_px + txt_w + 10 > max_x_px
            anchor_xy_norm = (1, 0.5) if is_near_right_edge else (0, 0.5)
            offset_xy_px = (-5, 0) if is_near_right_edge else (5, 0)
            self.style.fg_text.xy_norm(new_img, value_str, (slider_norm, 0.5), anchor_xy_norm, offset_xy_px)

        return draw_box_outline(new_img, self.style.outline_color)

    # .................................................................................................................


class MultiSlider(CachedBgFgElement):
    """
    Variant of a horizontal slider which has more than 1 control point
    This can be useful for setting min/max limits on a single value, for example.

    The number of control points is determined by the number of initial values provided.
    """

    # .................................................................................................................

    def __init__(
        self,
        label: str,
        values: int | float | list[int | float],
        min_val: float = 0.0,
        max_val: float = 1.0,
        step: float = 0.05,
        color: COLORU8 = (40, 40, 40),
        indicator_width: int = 1,
        text_scale: float = 0.5,
        marker_step: float | None = None,
        enable_value_display: bool = True,
        fill_color: COLORU8 | None = None,
        height: int = 40,
        minimum_width: int = 64,
    ):

        # Force to list-type, so we can handle single values as if they are multi-values
        if isinstance(values, (int, float)):
            values = tuple(values)

        # Make sure the given values make sense
        is_int = all(isinstance(var, int) for var in [min_val, max_val, step])
        data_dtype = np.int32 if is_int else np.float32
        min_val, max_val = sorted((min_val, max_val))
        initial_values = np.clip(np.array(sorted(values), dtype=data_dtype), min_val, max_val)

        # Storage for slider value
        self._label = label
        self._initial_values = initial_values
        self._slider_values = initial_values.copy()
        self._slider_min = min_val
        self._slider_max = max_val
        self._slider_step = step
        self._slider_delta = max(self._slider_max - self._slider_min, 1e-9)
        self._marker_x_norm = _get_norm_marker_positions(min_val, max_val, marker_step)
        self._max_precision = _get_step_precision(step)
        self._is_filled = fill_color is not None

        # Storage for slider state
        self._is_changed = True
        self._enable_value_display = enable_value_display
        self._drag_idx = 0

        # Set up text drawing
        txt_h = height * 0.8
        fg_color = pick_contrasting_gray_color(color)
        fg_text = TextDrawer(scale=text_scale, color=fg_color, max_height=txt_h)
        bg_text = TextDrawer(scale=text_scale, color=lerp_colors(fg_color, color, 0.55), max_height=txt_h)

        # Set up element styling
        self.style = UIStyle(
            color=color,
            indicator_width=indicator_width,
            indicator_color=fg_color,
            marker_color=lerp_colors(fg_color, color, 0.85),
            marker_width=1,
            marker_pad=5,
            fill_color=fill_color if self._is_filled else (255, 255, 255),
            fill_weight=0.5,
            fg_text=fg_text,
            bg_text=bg_text,
            outline_color=(0, 0, 0),
        )

        # Inherit from parent
        _, label_w, _ = fg_text.get_text_size(self._label)
        min_w = max(label_w, minimum_width)
        super().__init__(height, min_w, is_flexible_h=False, is_flexible_w=True)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name} ({self._label})"

    # .................................................................................................................

    def read(self) -> tuple[bool, float | int]:
        """Read slider values. Returns: is_changed, [slider_value1, slider_value2, ..., etc.]"""
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, np.sort(self._slider_values).tolist()

    def set(self, new_values: tuple[int | float], use_as_default_values: bool = True) -> SelfType:

        # Force new values to array-like data, within slider range
        if isinstance(new_values, (int, float)):
            new_values = [new_values]
        new_values = np.clip(np.array(sorted(new_values)), self._slider_min, self._slider_max)

        # Use new values as default if needed
        if use_as_default_values:
            self._initial_values = new_values.copy()

        # Check if new values are actually different and store
        self._is_changed |= np.allclose(new_values, self._slider_values)
        self._slider_values = new_values
        self.request_fg_repaint()

        return self

    def reset(self) -> SelfType:
        self.set(self._initial_values, use_as_default_values=False)
        return self

    def set_is_changed(self, is_changed=True) -> SelfType:
        """Helper used to artificially toggle is_changed flag, useful for forcing read updates"""
        self._is_changed = is_changed
        return self

    # .................................................................................................................

    def _on_left_down(self, cbxy, cbflags) -> None:
        """Update closest slider point on click and record index for dragging"""

        # Ignore clicks outside of the slider
        if not cbxy.is_in_region:
            return

        # Update closest click, redardless of whether we actually change values
        # (this is important for dragging to work properly)
        new_slider_value = self._mouse_x_norm_to_slider_value(cbxy.xy_norm[0])
        closest_idx = np.argmin((np.abs(self._slider_values - new_slider_value)))
        self._drag_idx = closest_idx

        # Update slider only if value changes
        is_value_changed = new_slider_value != self._slider_values[closest_idx]
        if is_value_changed:
            self._is_changed = True
            self._slider_values[closest_idx] = new_slider_value
            self.request_fg_repaint()

        return

    def _on_drag(self, cbxy, cbflags) -> None:
        """Update a single slider point on drag (determined by closest on left click)"""

        # Update slider value while dragging, only if values change
        new_slider_value = self._mouse_x_norm_to_slider_value(cbxy.xy_norm[0])
        is_value_changed = new_slider_value != self._slider_values[self._drag_idx]
        if is_value_changed:
            self._is_changed = True
            self._slider_values[self._drag_idx] = new_slider_value
            self.request_fg_repaint()

        return

    def _on_left_up(self, cbxy, cbflags) -> None:
        """For slight efficiency gain, sort values after modifications are complete"""
        self._slider_values = np.sort(self._slider_values)
        return

    def _on_right_click(self, cbxy, cbflags) -> None:
        """Reset slider position on right click"""
        self.reset()
        return

    # .................................................................................................................

    def _mouse_x_norm_to_slider_value(self, x_norm: float) -> float | int:
        """Helper used to convert normalized mouse position into slider values"""

        # Map normalized x position to slider range, snapped to step increments
        slider_x = (x_norm * self._slider_delta) + self._slider_min
        slider_x = round(slider_x / self._slider_step) * self._slider_step
        slider_x = max(self._slider_min, min(self._slider_max, slider_x))

        return slider_x

    # .................................................................................................................

    def _rerender_bg(self, h: int, w: int) -> ndarray:

        # Draw marker lines
        new_img = blank_image(h, w, self.style.color)
        mrk_pad_offset = 1 + self.style.marker_pad
        for x_norm in self._marker_x_norm:
            x_px = round(w * x_norm)
            pt1, pt2 = (x_px, mrk_pad_offset), (x_px, h - 1 - mrk_pad_offset)
            cv2.line(new_img, pt1, pt2, self.style.marker_color, self.style.marker_width)

        # Draw label
        return self.style.bg_text.xy_norm(new_img, self._label, (0, 0.5), offset_xy_px=(5, 0))

    def _rerender_fg(self, image: ndarray) -> ndarray:

        # Get image sizing for norm-to-px conversions
        img_h, img_w = image.shape[0:2]
        max_x_px = img_w - 1

        # Draw filled in region between min/max values, if needed
        if self._is_filled:

            # Figure out highlight region bounds
            x1_norm = (np.min(self._slider_values) - self._slider_min) / self._slider_delta
            x2_norm = (np.max(self._slider_values) - self._slider_min) / self._slider_delta
            x1_px = round(x1_norm * max_x_px)
            x2_px = round(x2_norm * max_x_px)

            # Mix in another color to indicator highlighted region
            fill_w = max(0, x2_px - x1_px)
            if fill_w > 0:
                fill_img = np.full((img_h, fill_w, 3), self.style.fill_color, dtype=np.uint8)
                orig_region = image[:, x1_px:x2_px, :]
                f_weight, inv_weight = self.style.fill_weight, 1 - self.style.fill_weight
                image[:, x1_px:x2_px, :] = cv2.addWeighted(orig_region, inv_weight, fill_img, f_weight, 0)

        # Draw indicator line(s)
        for value in self._slider_values:
            value_norm = (value - self._slider_min) / self._slider_delta
            line_x_px = round(value_norm * max_x_px)
            pt1, pt2 = (line_x_px, 0), (line_x_px, img_h)
            image = cv2.line(image, pt1, pt2, self.style.indicator_color, self.style.indicator_width)

            # Draw text beside indicator line to show current value if needed
            if self._enable_value_display:
                value_str = f"{float(value):.{self._max_precision}f}"
                _, txt_w, _ = self.style.fg_text.get_text_size(value_str)

                # Draw the text to the left or right of the indicator line, depending on where the image border is
                is_near_right_edge = line_x_px + txt_w + 10 > max_x_px
                anchor_xy_norm = (1, 0.5) if is_near_right_edge else (0, 0.5)
                offset_xy_px = (-5, 0) if is_near_right_edge else (5, 0)
                self.style.fg_text.xy_norm(image, value_str, (value_norm, 0.5), anchor_xy_norm, offset_xy_px)

        return draw_box_outline(image, self.style.outline_color)

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def _get_norm_marker_positions(
    min_value: int | float,
    max_value: int | float,
    marker_step: int | float,
) -> ndarray:
    """
    Helper used to compute the location of slider marker indicators.
    If steps_per_marker is None, then no markers will be returned
    """

    # Bail if we don't need steps
    if marker_step is None:
        return np.float32([])

    # Figure out where the marker boundaries are
    marker_min = marker_step * (min_value // marker_step)
    marker_max = marker_step * (2 + (max_value // marker_step))

    # Calculate normalized marker coordinates for drawing
    marker_pts = np.arange(marker_min, marker_max, marker_step, dtype=np.float32)
    marker_x_norm = (marker_pts - min_value) / max(max_value - min_value, 1e-9)

    return marker_x_norm


def _get_step_precision(slider_step_size: int | float) -> int:
    """
    Helper used to decide how many digits to display when showing
    a slider value indicator, based on step sizing.

    For example, for a step size of 0.05, we would want a
    precision of 2 decimal places. For a step of 5, we would
    want a precision of 0. This function includes extra checks
    to try to handle weird floating points issues,
    for example a step size of (0.1 + 0.2) = 0.30000000000000004
    will return a precision of 1
    """

    step_as_str = str(slider_step_size)
    step_dec_str = step_as_str.split(".")[-1] if "." in step_as_str else ""
    num_dec_places = len(step_dec_str)
    if num_dec_places >= 7:
        num_trunc = 2
        num_places_truncated = len(step_dec_str[:-num_trunc].rstrip("0"))
        print(step_as_str, num_dec_places, num_places_truncated)
        is_much_smaller = 0 < num_places_truncated < (num_dec_places - num_trunc)
        if is_much_smaller:
            num_dec_places = num_places_truncated
    return num_dec_places

    step_fractional = slider_step_size % 1
    if step_fractional == 0:
        return 0

    return int(np.ceil(-np.log10(max(slider_step_size, 1e-9))))
