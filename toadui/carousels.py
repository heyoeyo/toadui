#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from pathlib import Path

import numpy as np

from toadui.base import CachedBgFgElement
from toadui.text import TextDrawer
from toadui.helpers.images import blank_image
from toadui.helpers.drawing import draw_box_outline, draw_normalized_polygon
from toadui.helpers.styling import UIStyle
from toadui.helpers.colors import interpret_coloru8, pick_contrasting_gray_color

# For type hints
from typing import Any, Iterable, Callable
from numpy import ndarray
from toadui.helpers.types import COLORU8, SelfType


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class TextCarousel(CachedBgFgElement):
    """
    Element used to provide a single-item selector where the selection
    can be changed by using a left or right button, with wrap-around when
    reaching the beggining or end of the listing.

    Items in the carousel are represented using key-value pairs. The
    'keys' act as the labels shown on the UI element itself, while the
    selected value will be made available when reading from the element.
    Alternatively, items can be given as a list/tuple, in which case
    they will be automatically assigned integer indexes as their values.
    """

    # .................................................................................................................

    def __init__(
        self,
        key_value_pairs: dict | tuple | list,
        color: COLORU8 | int = (60, 60, 60),
        height: int = 40,
        minimum_width: int = 128,
        text_scale: float = 0.5,
        center_deadspace: float = 0.05,
    ):

        # If we don't get a dictionary-like input, convert it to one
        is_dictlike = all(hasattr(key_value_pairs, attr) for attr in ["keys", "values"])
        if not is_dictlike:
            key_value_pairs = {k: idx for idx, k in enumerate(key_value_pairs)}

        # Store basic state
        self._init_idx = 0
        self._is_changed = True
        self._enable_wrap_around = True
        self._curr_idx = self._init_idx
        self._keys = tuple(key_value_pairs.keys())
        self._values = tuple(key_value_pairs.values())
        self._label_strs = tuple(str(k) for k in self._keys)
        self._last_change_dir = 1

        # Storage for rendering hover images (e.g. with 'arrow is filled' indicators)
        self._cached_l_hover_bg = blank_image(1, 1)
        self._cached_r_hover_bg = blank_image(1, 1)
        self._cached_l_hover_img = blank_image(1, 1)
        self._cached_r_hover_img = blank_image(1, 1)

        # Store interaction settings
        center_deadspace = min(0.99, max(0.01, center_deadspace))
        self._left_edge_x_norm = 0.5 - center_deadspace
        self._right_edge_x_norm = 0.5 + center_deadspace

        # Set up element styling
        color = interpret_coloru8(color)
        fg_color = pick_contrasting_gray_color(color)
        self.style = UIStyle(
            color=color,
            arrow_color=fg_color,
            arrow_thickness=1,
            arrow_width_px=round(0.8 * height),
            text=TextDrawer(scale=text_scale, color=fg_color),
            outline_color=(0, 0, 0),
        )

        # Inherit from parent
        super().__init__(height, minimum_width, is_flexible_h=False, is_flexible_w=True)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        num_items = len(self._keys)
        return f"{cls_name} ({num_items} items)"

    def __len__(self) -> int:
        return len(self._keys)

    # .................................................................................................................

    def enable_wrap_around(self, enable: bool = True) -> SelfType:
        """Enable/disable wrap-around when cycling carousel entries"""
        self._enable_wrap_around = enable
        return self

    # .................................................................................................................

    def reset(self) -> SelfType:
        is_changed = self._init_idx != self._curr_idx
        self._curr_idx = self._init_idx
        self._is_changed = is_changed
        self._last_change_dir = 1
        self.request_fg_repaint(is_changed)
        return self

    def next(self, increment: int = 1) -> SelfType:
        """Cycle to the next entry"""

        # Handle wrap/no-wrap update
        num_items = len(self)
        new_idx = self._curr_idx + increment
        new_idx_no_wrap = max(0, min(new_idx, num_items - 1))
        new_idx_wrap = new_idx % num_items
        new_idx = new_idx_wrap if self._enable_wrap_around else new_idx_no_wrap

        return self.set_index(new_idx, use_as_default_value=False)

    def prev(self, decrement: int = 1) -> SelfType:
        """Cycle to the previous entry"""
        return self.next(-decrement)

    # .................................................................................................................

    def read(self) -> tuple[bool, Any, Any]:
        """Read current carousel selection. Returns: is_changed, current_key, current_value"""
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._keys[self._curr_idx], self._values[self._curr_idx]

    # .................................................................................................................

    def set_index(self, item_index: int, use_as_default_value: bool = False) -> SelfType:
        """Set carousel to a specific item index. Does nothing if given an index outside of the carousel range"""

        num_items = len(self)
        is_valid = 0 <= item_index < num_items
        if is_valid:
            is_changed = item_index != self._curr_idx
            if is_changed:
                # Figure out which direction we're moving
                diff_up = (item_index - self._curr_idx) % num_items
                diff_down = (self._curr_idx - item_index) % num_items
                self._last_change_dir = 1 if diff_up <= diff_down else -1

                self._is_changed = True
                self._curr_idx = item_index
                self.request_fg_repaint()

            if use_as_default_value:
                self._init_idx = item_index

        return self

    def set_key(self, key: Any, use_as_default_value: bool = False) -> SelfType:
        """Set the carousel to a specific value (does nothing if given an unrecognized value)"""

        is_valid = key in self._keys
        if is_valid:
            new_idx = self._keys.index(key)
            self.set_index(new_idx, use_as_default_value)

        return self

    # .................................................................................................................

    def add_entry(
        self,
        new_key_value_pair: tuple[Any, Any],
        insert_index: int | None = None,
        set_to_new_entry: bool = False,
    ) -> SelfType:
        """
        Add a new value to the carousel listing. The new entry will be
        added to the end of the listing by default, but placement can
        be adjusted using the insert_index.
        """

        # If we don't get an insertion index, assume we add to the end
        insert_index = insert_index if insert_index is not None else len(self._keys)

        # Special case, if we don't get a tuple, assume we're given a single value, and force it into a pair
        if not isinstance(new_key_value_pair, (tuple, list)):
            new_key_value_pair = (new_key_value_pair, insert_index)

        new_key, new_value = new_key_value_pair
        already_exists = new_key in self._keys
        if not already_exists:
            keys_list, vals_list = list(self._keys), list(self._values)
            keys_list.insert(insert_index, new_key)
            vals_list.insert(insert_index, new_value)
            self._keys = tuple(keys_list)
            self._values = tuple(vals_list)
            self._label_strs = tuple(str(v) for v in self._keys)

            is_changed = insert_index == self._curr_idx
            self._is_changed |= is_changed
            self.request_fg_repaint(is_changed)
            if set_to_new_entry:
                self.set_key(new_key)

        return self

    # .................................................................................................................

    def remove_entry_by_index(self, index: int) -> tuple[Any, Any]:
        """
        Removes an entry from the carousel based on the provided index.
        Returns:
            removed_key, removed_value

        - Negative indexing can be used (e.g. index=-1 will remove the last item)
        - Using an index greater than the number of elements will do nothing
          (will return (None, None) in this case)
        """

        # Handle negative indexing
        if index < 0:
            index = max(0, len(self._keys) - index)

        # If we're asked to remove an index outside of the carousel range, do nothing
        removed_key, removed_value = None, None
        if index >= len(self._keys) or index < 0:
            return removed_key, removed_value

        # Convert internal data to lists so we can modify them
        kvl_lists = [list(data) for data in (self._keys, self._values, self._label_strs)]
        popped_items = [data.pop(index) for data in kvl_lists]

        # Convert modified datasets back into tuples for storage
        new_keys, new_values, new_labels = [tuple(data) for data in kvl_lists]
        self._keys = new_keys
        self._values = new_values
        self._label_strs = new_labels

        # Sanity check
        num_items = len(self)
        assert num_items > 0, f"Error! No items in {self.__class__.__name__}"

        # Update current indexing if needed
        if index < self._curr_idx:
            self._curr_idx = max(0, self._curr_idx - 1)
        elif index == self._curr_idx:
            move_right_on_delete = self._last_change_dir > 0
            next_idx = self._curr_idx if move_right_on_delete else self._curr_idx - 1
            next_idx = next_idx % num_items if self._enable_wrap_around else max(0, min(next_idx, num_items - 1))
            self.set_index(next_idx, use_as_default_value=False)
            self._is_changed = True

        # Handle changes to 'default' value
        if index < self._init_idx:
            self._init_idx = max(0, self._init_idx - 1)
        elif index == self._init_idx:
            # If we remove the default value, just reset back to 0th index
            self._init_idx = 0

        # Return the key of the removed item (may be useful for verification?)
        removed_key, removed_value = popped_items[0:2]
        return removed_key, removed_value

    def remove_entry_by_key(self, key: Any | None) -> tuple[Any, Any]:
        """Remove an item from the carousel based on it's key. Returns: removed_key, removed_value"""

        is_valid = key in self._keys
        if is_valid:
            index_to_remove = self._keys.index(key)
            return self.remove_entry_by_index(index_to_remove)

        return None, None

    def remove_current_entry(self) -> tuple[Any, Any]:
        """ "Remove the currently selected carousel item. Returns: removed_key, removed_value"""
        return self.remove_entry_by_index(self._curr_idx)

    # .................................................................................................................

    def _on_left_click(self, cbxy, cbflags) -> None:

        x_click = cbxy.xy_norm[0]
        if x_click < self._left_edge_x_norm:
            self.prev()
        elif x_click > self._right_edge_x_norm:
            self.next()

        return

    def _on_right_click(self, cbxy, cbflags) -> None:
        self.reset()
        return

    # .................................................................................................................

    def _rerender_bg(self, h: int, w: int) -> ndarray:

        # Create base image for drawing arrow shape
        arrow_base = blank_image(h, self.style.arrow_width_px, self.style.color)
        tri_norm = np.float32([(0.2, 0.5), (0.8, 0.2), (0.8, 0.8)])

        # Draw outlined arrows and filled in copies
        arrow_thick = self.style.arrow_thickness
        arrow_color = self.style.arrow_color
        l_arrow_lines = draw_normalized_polygon(arrow_base.copy(), tri_norm, arrow_color, arrow_thick)
        l_arrow_fill = draw_normalized_polygon(arrow_base.copy(), tri_norm, arrow_color, -1)
        r_arrow_lines, r_arrow_fill = [np.fliplr(l_img) for l_img in [l_arrow_lines, l_arrow_fill]]

        # Create spacer image for drawing text (as part of foreground render)
        text_w = w - 2 * self.style.arrow_width_px
        text_space_img = blank_image(h, text_w, self.style.color)

        # Combine arrow images & text spacer. Storing hover images for re-use
        self._cached_l_hover_bg = draw_box_outline(np.hstack((l_arrow_fill, text_space_img, r_arrow_lines)))
        self._cached_r_hover_bg = draw_box_outline(np.hstack((l_arrow_lines, text_space_img, r_arrow_fill)))
        return draw_box_outline(np.hstack((l_arrow_lines, text_space_img, r_arrow_lines)))

    # .................................................................................................................

    def _rerender_fg(self, base_image: ndarray) -> ndarray:

        # Draw new text label
        curr_label = self._label_strs[self._curr_idx]
        new_img = self.style.text.xy_centered(base_image, curr_label)

        # Update text on left/right hover backgrounds
        self._cached_l_hover_img = self.style.text.xy_centered(self._cached_l_hover_bg.copy(), curr_label)
        self._cached_r_hover_img = self.style.text.xy_centered(self._cached_r_hover_bg.copy(), curr_label)

        return new_img

    # .................................................................................................................

    def _post_rerender(self, image: ndarray) -> ndarray:

        # Switch to showing filled arrow when hovering left/right
        out_img = image
        if self.is_hovered():
            x_norm, _ = self.get_event_xy().xy_norm
            if x_norm < self._left_edge_x_norm:
                out_img = self._cached_l_hover_img
            elif x_norm > self._right_edge_x_norm:
                out_img = self._cached_r_hover_img

        return out_img

    # .................................................................................................................


class PathCarousel(TextCarousel):
    """
    Helper variant of a TextCarousel.
    This version allows for providing file or folder paths as the input listing.

    If a folder path is given, then the contents of the folder will be
    listed out as the carousel entries.
    If a list of strings or paths is given, then these are assumed
    to be file/folder paths and each file/folder name will become an
    entry in the carousel.
    """

    # .................................................................................................................

    def __init__(
        self,
        folder_path_or_path_list: str | list[str] | Path | list[Path],
        color: COLORU8 | int = (60, 60, 60),
        height: int = 40,
        minimum_width: int = 128,
        text_scale: float = 0.5,
        center_deadspace: float = 0.05,
        sort_by_name: bool = True,
        sort_key: Callable | None = None,
    ):

        # Force into pathlib type for convenience
        if isinstance(folder_path_or_path_list, str):
            folder_path_or_path_list = Path(folder_path_or_path_list)

        # Handle single path input (e.g. is a directory or file) vs. list of paths
        paths_list = None
        if isinstance(folder_path_or_path_list, Path):
            is_dir = folder_path_or_path_list.is_dir()
            paths_list = [p for p in folder_path_or_path_list.iterdir()] if is_dir else [folder_path_or_path_list]
        elif isinstance(folder_path_or_path_list, Iterable):
            paths_list = [Path(p) for p in folder_path_or_path_list]
        else:
            raise TypeError("Error! Expecting path or list of paths as input")

        # Sort paths if needed
        if sort_key is not None:
            assert callable(sort_key), "sort_key must be a function, for use in: sorted(paths_list, key=sort_key)"
            paths_list = sorted(paths_list, key=sort_key)
        elif sort_by_name:
            paths_list = sorted(paths_list, key=lambda p: p.name.lower())

        # Inherit from parent
        kv_pairs = {p.name: p for p in paths_list}
        super().__init__(kv_pairs, color, height, minimum_width, text_scale, center_deadspace)

        # Storage for previously loaded data (if loading from path), used to avoid repeat loading
        self._loaded_data = None

    def read(self) -> tuple[bool, str, Path]:
        """Returns: is_changed, current_file_name, current_file_path"""
        return super().read()

    # .................................................................................................................
