#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from toadui.base import BaseCallback, CachedBgFgElement
from toadui.helpers.text import TextDrawer
from toadui.helpers.images import blank_image
from toadui.helpers.drawing import draw_box_outline, draw_drop_shadow, draw_rectangle_norm
from toadui.helpers.colors import adjust_as_hsv, pick_contrasting_gray_color, lerp_colors
from toadui.helpers.styling import UIStyle
from toadui.helpers.sizing import get_image_hw_to_fit_region, resize_hw

# For type hints
from numpy import ndarray
from toadui.helpers.types import COLORU8, SelfType
from toadui.helpers.ocv_types import OCVInterp


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class ToggleButton(BaseCallback):

    # .................................................................................................................

    def __init__(
        self,
        label: str,
        default_state: bool = False,
        color_on: COLORU8 = (70, 65, 180),
        color_off: COLORU8 | None = None,
        text_color_on: COLORU8 | None = None,
        text_color_off: COLORU8 | None = None,
        height: int = 40,
        text_scale: float = 0.5,
        is_flexible_w: bool = True,
    ):

        # Storage for cached data
        self._cached_off_img = blank_image(1, 1)
        self._cached_on_img = blank_image(1, 1)

        # Storage for toggle state
        self._is_on = default_state
        self._is_changed = True

        # Figure out missing colors, if needed
        color_drop_shadow = adjust_as_hsv(color_on, 1, 1.5, 0.35)
        if text_color_on is None:
            text_color_on = pick_contrasting_gray_color(color_on)
        if color_off is None:
            color_off = adjust_as_hsv(color_on, 1, 0.5, 0.8)
        if text_color_off is None:
            text_color_off = lerp_colors(color_off, text_color_on, 0.5)

        # Set up element styling
        self._label = f" {label} "
        self.style = UIStyle(
            color_on=color_on,
            color_off=color_off,
            color_drop_shadow=color_drop_shadow,
            enable_drop_shadow=True,
            text_on=TextDrawer(scale=text_scale, color=text_color_on, max_height=height),
            text_off=TextDrawer(scale=text_scale, color=text_color_off, max_height=height),
        )

        # Inherit from parent
        _, btn_w, _ = self.style.text_on.get_text_size(self._label)
        super().__init__(height, btn_w, is_flexible_h=False, is_flexible_w=is_flexible_w)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name} ({self._label})"

    # .................................................................................................................

    @classmethod
    def many(
        cls,
        *labels: list[str],
        default_state: bool = False,
        color_on: COLORU8 = (70, 65, 180),
        color_off: COLORU8 | None = None,
        height: int = 40,
        text_scale: float = 0.5,
    ):
        """Helper used to create multiple toggle buttons of the same style, all at once"""

        # Make sure labels iterable is held as a list of strings
        labels = [labels] if isinstance(labels, str) else [str(label) for label in labels]
        return [cls(l, default_state, color_on, color_off, height=height, text_scale=text_scale) for l in labels]

    # .................................................................................................................

    def read(self) -> tuple[bool, bool]:
        """Returns: is_changed, current_state"""
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._is_on

    def toggle(self, new_state: bool | None = None) -> bool:
        """Toggle current state (or set to True/False if given an input). Returns: new_state"""

        old_state = self._is_on
        self._is_on = not self._is_on if new_state is None else new_state
        if old_state != self._is_on:
            self._is_changed = True

        return self._is_on

    def set_is_changed(self, is_changed: bool = True) -> SelfType:
        """Artificially set change state"""
        self._is_changed = is_changed
        return self

    # .................................................................................................................

    def _on_left_click(self, cbxy, cbflags) -> None:
        self.toggle()

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-draw on/off button states if size changes
        if h != self._cached_on_img.shape[0] or w != self._cached_on_img.shape[1]:

            # Draw off state
            off_img = blank_image(h, w, self.style.color_off)
            off_img = self.style.text_off.xy_centered(off_img, self._label)

            # Draw on state with drop-shadow if needed
            on_img = blank_image(h, w, self.style.color_on)
            txt_offset = (0, 0)
            if self.style.enable_drop_shadow:
                on_img = draw_drop_shadow(on_img, color=self.style.color_drop_shadow, blur_sharpness=0.5)
                txt_offset = (0, 2)
            on_img = self.style.text_on.xy_centered(on_img, self._label, offset_xy_px=txt_offset)

            # Store images for re-use
            self._cached_on_img = draw_box_outline(on_img)
            self._cached_off_img = draw_box_outline(off_img)

        # Draw button label
        btn_img = self._cached_on_img if self._is_on else self._cached_off_img
        if self.is_hovered():
            btn_img = draw_box_outline(btn_img.copy(), (255, 255, 255))

        return btn_img

    # .................................................................................................................


class ToggleImageButton(BaseCallback):

    def __init__(
        self,
        default_image: ndarray,
        toggle_image: ndarray | None = None,
        default_state: bool = False,
        include_box_outline: bool = True,
        resize_interpolation: OCVInterp = cv2.INTER_AREA,
        height: int = 40,
        minimum_width: int = 40,
        is_flexible_h: bool = False,
        is_flexible_w: bool = True,
    ):

        # Fill in missing image data, if needed
        if default_image.ndim < 3:
            default_image = cv2.cvtColor(default_image, cv2.COLOR_GRAY2BGR)
        if toggle_image is None:
            toggle_image = default_image

        # Storage for cached data
        self._default_ar = default_image.shape[1] / default_image.shape[0]
        self._off_img = toggle_image if default_state else default_image
        self._on_img = default_image if default_state else toggle_image
        self._cached_off_img = blank_image(1, 1)
        self._cached_on_img = blank_image(1, 1)

        # Storage for toggle state
        self._is_on = default_state
        self._is_changed = True

        # Set up element styling
        self.style = UIStyle(
            outline_color=(0, 0, 0) if include_box_outline else None,
            outline_hover_color=(255, 255, 255) if include_box_outline else None,
            resize_interpolation=resize_interpolation,
        )

        # Inherit from parent
        super().__init__(height, minimum_width, is_flexible_h, is_flexible_w)

    # .................................................................................................................

    def read(self) -> tuple[bool, bool]:
        """Returns: is_changed, current_state"""
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._is_on

    def toggle(self, new_state: bool | None = None) -> bool:
        """Toggle current state (or set to True/False if given an input). Returns: new_state"""
        old_state = self._is_on
        self._is_on = not self._is_on if new_state is None else new_state
        if old_state != self._is_on:
            self._is_changed = True
        return self._is_on

    def set_is_changed(self, is_changed: bool = True) -> SelfType:
        """Artificially set change state"""
        self._is_changed = is_changed
        return self

    # .................................................................................................................

    def _on_left_click(self, cbxy, cbflags) -> None:
        self.toggle()

    # .................................................................................................................

    def _get_dynamic_aspect_ratio(self):
        if self._cb_rdr.is_flexible_h and self._cb_rdr.is_flexible_w:
            return self._default_ar
        return None

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-draw on/off button states if size changes
        if h != self._cached_on_img.shape[0] or w != self._cached_on_img.shape[1]:

            off_hw = get_image_hw_to_fit_region(self._off_img.shape, (h, w))
            on_hw = get_image_hw_to_fit_region(self._on_img.shape, (h, w))
            off_img = resize_hw(self._off_img, off_hw, self.style.resize_interpolation)
            on_img = resize_hw(self._on_img, on_hw, self.style.resize_interpolation)

            # Store images for re-use
            self._cached_off_img = draw_box_outline(off_img, self.style.outline_color)
            self._cached_on_img = draw_box_outline(on_img, self.style.outline_color)

        # Draw button label
        btn_img = self._cached_on_img if self._is_on else self._cached_off_img
        if self.is_hovered():
            btn_img = draw_box_outline(btn_img.copy(), self.style.outline_hover_color)

        return btn_img


class ImmediateButton(BaseCallback):

    # .................................................................................................................

    def __init__(
        self,
        label: str,
        color: COLORU8 = (100, 80, 90),
        text_color: COLORU8 | None = None,
        height: int = 40,
        text_scale: float = 0.5,
        is_flexible_w: bool = True,
    ):

        # Storage for cached button image
        self._cached_img = blank_image(1, 1)

        # Storage for button state
        self._is_changed = False

        # Set default text color, if none provided
        if text_color is None:
            text_color = pick_contrasting_gray_color(color)

        # Set up element styling
        self._label = f" {label} "
        self.style = UIStyle(
            color=color,
            text=TextDrawer(scale=text_scale, color=text_color, max_height=height),
        )

        # Inherit from parent
        _, btn_w, _ = self.style.text.get_text_size(self._label)
        super().__init__(height, btn_w, is_flexible_h=False, is_flexible_w=is_flexible_w)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name} ({self._label})"

    # .................................................................................................................

    @classmethod
    def many(
        cls,
        *labels: list[str],
        color: COLORU8 = (100, 80, 90),
        text_color: COLORU8 | None = None,
        height: int = 40,
        text_scale: float = 0.5,
    ):
        """Helper used to create multiple immediate buttons of the same style, all at once"""

        # Make sure labels iterable is held as a list of strings
        labels = [labels] if isinstance(labels, str) else [str(label) for label in labels]
        return [cls(label, color, text_color, height, text_scale) for label in labels]

    # .................................................................................................................

    def read(self) -> bool:
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed

    def click(self) -> SelfType:
        self._is_changed = True
        return self

    # .................................................................................................................

    def _on_left_click(self, cbxy, cbflags) -> None:
        self.click()

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Redraw button only when sizing changes
        if h != self._cached_img.shape[0] or w != self._cached_img.shape[1]:
            btn_img = blank_image(h, w, self.style.color)
            btn_img = self.style.text.xy_centered(btn_img, self._label)
            self._cached_img = draw_box_outline(btn_img)

        if self.is_hovered():
            return draw_box_outline(self._cached_img.copy(), (255, 255, 255))

        return self._cached_img

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Groupings


class RadioConstraint:
    """
    Class which can wrap-around toggle elements to enforce a 'radio constraint'
    This means that only 1 element can be active at a time
    (like the convenitonal UI element referred to as 'radio buttons')

    This is not a layout item however, so items can be independently
    arranged in a UI while still using this element.
    """

    def __init__(self, *items: ToggleButton, active_index=0):

        self._items = tuple(items)
        self._active_idx = active_index

        self._bit_state = np.zeros(len(self._items), dtype=np.bool)
        self._bit_state[active_index] = True
        self._enforce_radio_constraint(active_index)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        item_names = ", ".join((str(item) for item in self._items))
        return f"{cls_name} [{item_names}]"

    def read(self) -> tuple[bool, int, ToggleButton]:
        """
        Reads all items in the radio group and enforces the radio constraint (e.g. only 1 active)
        Returns:
            is_changed, active_index, active_item
        """

        # Read all item states
        new_bit_state = np.zeros_like(self._bit_state)
        for item_idx, item in enumerate(self._items):
            _, item_state = item.read()
            new_bit_state[item_idx] = item_state

        # Look for changes in states
        bit_change_state = np.bitwise_xor(new_bit_state, self._bit_state)
        num_bits_changed = np.sum(bit_change_state)
        is_changed = num_bits_changed == 1

        # If nothing changes, repeat prior outputs
        if num_bits_changed == 0:
            return is_changed, self._active_idx, self._items[self._active_idx]

        # If more than 1 button changes in a single update, restore previous state
        # (which item should be active is ambiguous in this case)
        if num_bits_changed > 1:
            bit_change_state = self._bit_state

        # Force items into state satisfying radio constraint
        self._bit_state = bit_change_state
        self._active_idx = np.argmax(bit_change_state)
        self._enforce_radio_constraint()

        return is_changed, self._active_idx, self._items[self._active_idx]

    def _enforce_radio_constraint(self, active_index: int | None = None) -> SelfType:
        """Loop over all group items and force all but 1 to be off"""

        if active_index is None:
            active_index = self._active_idx

        for item_idx, item in enumerate(self._items):
            is_on = item_idx == active_index
            item.toggle(is_on)
            item.set_is_changed(False)

        return self

    def __iter__(self):
        return iter(self._items)


class RadioBar(CachedBgFgElement):
    """
    Special-case implementation of a row of toggle buttons,
    where only one button is meant to be active at any given time.
    This can be done 'manually' using radio constraints and
    individual toggle button elements, however, this element
    may be simpler to use and is more graphically coherent.
    """

    def __init__(
        self,
        *labels,
        active_index=0,
        color_on=(95, 90, 75),
        color_off=None,
        text_scale=0.5,
        proportional_sizing=False,
        height=40,
        is_flexible_w=True,
    ):

        self._labels = tuple(str(label) for label in labels)
        self._cached_img = blank_image(1, 1)
        self._active_idx = active_index
        self._is_changed = True
        self._enable_wrap_around = False

        if color_off is None:
            color_off = adjust_as_hsv(color_on, 1, 0.35, 0.65)

        # Set up element styling
        text_color_on = pick_contrasting_gray_color(color_on)
        text_color_off = adjust_as_hsv(text_color_on, 1, 0, 0.5)
        self.style = UIStyle(
            color_on=color_on,
            color_off=color_off,
            text_on=TextDrawer(scale=text_scale, color=text_color_on, max_height=height),
            text_off=TextDrawer(scale=text_scale, color=text_color_off, max_height=height),
            outline_color=(0, 0, 0),
        )

        # Figure out sizing per label/button
        # -> Also pre-compute rectangle bounding box & mid-point, for rendering
        w_per_txt = [self.style.text_on.get_text_size(f" {label} ").w for label in self._labels]
        self._btn_cumw_norm = np.cumsum(w_per_txt) / sum(w_per_txt)
        if not proportional_sizing:
            self._btn_cumw_norm = np.linspace(0, 1, 1 + len(self._labels), dtype=np.float32)[1:]
        self._btn_x1x2_norm = np.float32(list(zip([0, *self._btn_cumw_norm], self._btn_cumw_norm)))
        self._btn_xmid_norm = np.mean(self._btn_x1x2_norm, axis=1).tolist()

        # Inherit from parent
        min_w = sum(w_per_txt)
        super().__init__(height, min_w, is_flexible_h=False, is_flexible_w=is_flexible_w)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        num_items = len(self._labels)
        return f"{cls_name} ({num_items} options)"

    # .................................................................................................................

    def enable_wrap_around(self, enable: bool = True) -> SelfType:
        """Enable/disable wrap-around when cycling entries"""
        self._enable_wrap_around = enable
        return self

    def read(self) -> tuple[bool, int, str]:
        """
        Read current selected item:
        Returns:
            is_changed, active_index, active_label
        """
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._active_idx, self._labels[self._active_idx]

    def next(self, increment: int = 1) -> SelfType:
        """Switch to the next entry"""

        # Handle wrap/no-wrap update
        num_items = len(self._labels)
        new_idx = self._active_idx + increment
        new_idx_no_wrap = max(0, min(new_idx, num_items - 1))
        new_idx_wrap = new_idx % num_items
        new_idx = new_idx_wrap if self._enable_wrap_around else new_idx_no_wrap

        # Store results
        return self.set_index(new_idx)

    def prev(self, decrement=1) -> SelfType:
        """Function used to select the previous option (with wrap-around)"""
        return self.next(-decrement)

    def set_label(self, item_label: str) -> SelfType:
        idx_of_item = self._labels.index(item_label)
        return self.set_index(idx_of_item)

    def set_index(self, item_index: int) -> SelfType:
        num_items = len(self._labels)
        item_index = max(0, min(num_items, item_index))
        is_changed = item_index != self._active_idx
        if is_changed:
            self._active_idx = item_index
            self._is_changed = True
            self.request_fg_repaint()
        return self

    def set_is_changed(self, is_changed: bool = True) -> SelfType:
        """Artificially set change state"""
        self._is_changed = is_changed
        return self

    # .................................................................................................................

    def _on_left_click(self, cbxy, cbflags) -> None:

        # Figure out which button was clicked
        click_x_norm, _ = cbxy.xy_norm
        click_idx = len(self._labels)
        for btn_idx, btn_x2_norm in enumerate(self._btn_cumw_norm):
            if click_x_norm < btn_x2_norm:
                click_idx = btn_idx
                break

        self.set_index(click_idx)

        return

    # .................................................................................................................

    def _rerender_bg(self, h: int, w: int) -> ndarray:

        new_img = blank_image(h, w, self.style.color_off)
        for label, label_x_norm in zip(self._labels, self._btn_xmid_norm):
            txt_xy = (label_x_norm, 0.5)
            self.style.text_off.xy_norm(new_img, label, txt_xy, anchor_xy_norm=(0.5, 0.5))

        return draw_box_outline(new_img, self.style.outline_color)

    def _rerender_fg(self, bg_image: ndarray) -> ndarray:

        img_h, img_w = bg_image.shape[0:2]
        btn_x1x2_px = self._btn_x1x2_norm[self._active_idx] * (img_w - 1)
        x1_px, x2_px = np.int32(np.round(btn_x1x2_px)).tolist()
        pt1 = (x1_px + 1, 0)
        pt2 = (x2_px - 1, img_h - 1)

        # Paint over active button with 'on' colors
        txt_xy_norm = (self._btn_xmid_norm[self._active_idx], 0.5)
        new_img = cv2.rectangle(bg_image, pt1, pt2, self.style.color_on, -1)
        self.style.text_on.xy_norm(new_img, self._labels[self._active_idx], txt_xy_norm, anchor_xy_norm=(0.5, 0.5))

        return draw_box_outline(new_img)

    def _post_rerender(self, image: ndarray) -> ndarray:

        # Bail if user isn't hovering, no need to update render
        if not self.is_hovered():
            return image

        # Figure out which item is being hovered
        evt_xy = self.get_event_xy()
        hover_x_norm, _ = evt_xy.xy_norm
        hover_idx = len(self._labels)
        for btn_idx, btn_x2_norm in enumerate(self._btn_cumw_norm):
            if hover_x_norm < btn_x2_norm:
                hover_idx = btn_idx
                break

        # Bail if active item if hovered
        if hover_idx == self._active_idx:
            return image

        # Create new image to draw on, so we don't destroy the cached results
        hover_img = image.copy()
        hover_label = self._labels[hover_idx]
        hover_color = lerp_colors(self.style.color_on, self.style.color_off, 0.75)

        # Draw an in-between background color to suggest hover state
        x1_norm, x2_norm = self._btn_x1x2_norm[hover_idx]
        xy1_norm, xy2_norm = (x1_norm, 0), (x2_norm, 1)
        draw_rectangle_norm(hover_img, xy1_norm, xy2_norm, hover_color, -1, pad_xy1xy2_px=(1, 1))

        # Re-draw the button text in the on-state to indicate hover
        txt_xy = (self._btn_xmid_norm[hover_idx], 0.5)
        self.style.text_on.xy_norm(hover_img, hover_label, txt_xy, anchor_xy_norm=(0.5, 0.5))

        return hover_img
