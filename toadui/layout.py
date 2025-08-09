#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from toadui.base import BaseCallback
from toadui.helpers.styling import UIStyle

# For type hints
from typing import Iterable, Any
from numpy import ndarray
from toadui.base import BaseOverlay, CBRenderSizing
from toadui.helpers.types import SelfType, HWPX


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class HStack(BaseCallback):
    """Layout element which stacks UI items together horizontally"""

    # .................................................................................................................

    def __init__(
        self,
        *items: BaseCallback,
        flex: Iterable[float | None] | None = None,
        min_w: int | None = None,
        error_on_size_constraints: bool = False,
    ):

        # Inherit from parent, with dummy values (needed so we can get child iterator!)
        super().__init__(32, 32)
        self._append_cb_children(*items)

        # Use child sizing to determine stack sizing
        tallest_child_min_h = max(child._cb_rdr.min_h for child in self)
        total_child_min_w = sum(child._cb_rdr.min_w for child in self)
        is_flex_h = any(child._cb_rdr.is_flexible_h for child in self)
        is_flex_w = any(child._cb_rdr.is_flexible_w for child in self)
        min_h = tallest_child_min_h
        min_w = max(total_child_min_w, min_w if min_w is not None else 1)
        self._cb_rdr = CBRenderSizing(min_h, min_w, is_flex_h, is_flex_w)

        # Default to sizing by aspect ratio if no flex values are given
        # -> Don't use AR sizing if not flexible (implies stacking doesn't have target AR)
        multiple_ar_children = sum(child._get_dynamic_aspect_ratio() is not None for child in self) > 1
        self._size_by_ar = flex is None and (is_flex_w and is_flex_h) and multiple_ar_children

        # Pre-compute sizing info for handling flexible sizing
        is_flex_w_per_child = [child._cb_rdr.is_flexible_w for child in self]
        flex_per_child_list = _read_flex_values(is_flex_w_per_child, flex)
        fixed_width_of_children = 0
        for child, flex_val in zip(self, flex_per_child_list):
            is_flexible = flex_val > 1e-2
            fixed_width_of_children += 0 if is_flexible else child._cb_rdr.min_w
        self._fixed_flex_width = fixed_width_of_children
        self._cumlative_flex = np.cumsum(flex_per_child_list, dtype=np.float32)
        self._flex_debug = flex_per_child_list
        self._error_on_constraints = error_on_size_constraints

        # Set up element styling when having to pad child items
        self.style = UIStyle(
            pad_color=(0, 0, 0),
            pad_border_type=cv2.BORDER_CONSTANT,
        )

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Set up starting stack point, used to keep track of child callback regions
        x_stack = self._cb_region.x1
        y_stack = self._cb_region.y1

        if self._size_by_ar:
            w_per_child_list = [child._get_width_given_height(h) for child in self]
            total_w = sum(w_per_child_list)
            if total_w != w:
                fix_list = []
                flex_list = []
                for child, child_w in zip(self, w_per_child_list):
                    fix_list.append(0 if child._cb_rdr.is_flexible_w else child._cb_rdr.min_w)
                    flex_list.append(child_w if child._cb_rdr.is_flexible_w else 0)

                avail_w = w - sum(fix_list)
                cumulative_w = np.cumsum(flex_list, dtype=np.float32) * avail_w / sum(flex_list)
                flex_list = np.diff(np.int32(np.round(cumulative_w)), prepend=0).tolist()
                w_per_child_list = [fixed_w if fixed_w > 0 else flex_w for fixed_w, flex_w in zip(fix_list, flex_list)]
        else:
            # Assign per-element sizing, taking into account flex scaling
            avail_w = max(0, w - self._fixed_flex_width)
            flex_w_px = np.diff(np.int32(np.round(self._cumlative_flex * avail_w)), prepend=0).tolist()
            w_per_child_list = [flex_w if flex_w > 0 else child._cb_rdr.min_w for child, flex_w in zip(self, flex_w_px)]

        # Have each child item draw itself
        imgs_list = []
        for child, ch_render_w in zip(self, w_per_child_list):
            frame = child._render_up_to_size(h, ch_render_w)
            frame_h, frame_w = frame.shape[0:2]

            # Crop overly-tall images
            # -> Don't need to crop wide images, since h-stacking won't break!
            if frame_h > h:
                print(
                    f"Render sizing error! Expecting height: {h}, got {frame_h} ({child})",
                    "-> Will crop!",
                    sep="\n",
                )
                frame = frame[:h, :, :]
                frame_h = h

            # Adjust frame height if needed
            tpad, lpad, bpad, rpad = 0, 0, 0, 0
            need_pad = (frame_h < h) or (frame_w < ch_render_w)
            if need_pad:
                available_h = h - frame_h
                available_w = max(0, ch_render_w - frame_w)
                tpad, lpad = available_h // 2, available_w // 2
                bpad, rpad = available_h - tpad, available_w - lpad
                ptype = self.style.pad_border_type
                pcolor = self.style.pad_color
                frame = cv2.copyMakeBorder(frame, tpad, bpad, lpad, rpad, ptype, value=pcolor)

            # Store image
            imgs_list.append(frame)

            # Provide callback region to child item
            x1, y1 = x_stack + lpad, y_stack + tpad
            x2, y2 = x1 + frame_w, y1 + frame_h
            child._cb_region.resize(x1, y1, x2, y2)

            # Update stacking point for next child
            x_stack = x2 + rpad

        return np.hstack(imgs_list)

    # .................................................................................................................

    def _get_dynamic_aspect_ratio(self) -> float | None:
        if self._cb_rdr.is_flexible_h and self._cb_rdr.is_flexible_w:
            child_ar = (c._get_dynamic_aspect_ratio() for c in self)
            ar = sum(ar if ar is not None else 0 for ar in child_ar)
            return None if ar == 0 else ar
        return None

    def _get_width_given_height(self, h: int) -> int:

        if not self._cb_rdr.is_flexible_w:
            return self._cb_rdr.min_w

        # Ask child elements for desired width and use total
        w_per_child_list = [child._get_width_given_height(h) for child in self]
        total_child_w = sum(w_per_child_list)
        return max(self._cb_rdr.min_w, total_child_w)

    def _get_height_given_width(self, w: int) -> int:
        """
        For h-stacking, we normally want to set a height since this must be shared
        for all elements in order to stack horizontally. Here we don't know the height.

        If sizing by aspect ratio, we calculate the height from knowing that all
        items must stack to the target width, while sharing the same height:
            target_w = (h * ar1) + (h * ar2) + (h * ar3) + ...
            target_w = h * (ar1 + ar2 + ar3)
            Therefore, h = target_w / sum(ar for all item aspect ratios)

        If sizing by flex values, we first figure out how much 'width' each child
        should be assigned. Then each child is asked for it's render height, given
        the assigned width. We take the 'tallest' child height as the height for stacking.

        Returns:
            render_height
        """

        # Use fixed height if not flexible
        if not self._cb_rdr.is_flexible_h:
            return self._cb_rdr.min_h

        # Allocate height based on child aspect ratios
        if self._size_by_ar:
            avail_w = max(0, w - self._fixed_flex_width)
            child_ar = (c._get_dynamic_aspect_ratio() for c in self)
            h = avail_w / sum(ar if ar is not None else 0 for ar in child_ar)
            return max(self._cb_rdr.min_h, round(h))

        # Figure out per-element width based on flex assignment
        avail_w = max(0, w - self._fixed_flex_width)
        flex_sizing_px = np.diff(np.int32(np.round(self._cumlative_flex * avail_w)), prepend=0).tolist()
        w_per_child_list = [max(child._cb_rdr.min_w, flex_w) for child, flex_w in zip(self, flex_sizing_px)]

        # Sanity check
        if self._error_on_constraints:
            total_computed_w = sum(w_per_child_list)
            assert total_computed_w == w, f"Error computing target widths ({self})! Target: {w}, got {total_computed_w}"

        # We'll say our height, given the target width, is that of the tallest child element
        h_per_child_list = (child._get_height_given_width(w) for child, w in zip(self, w_per_child_list))
        return max(h_per_child_list)

    # .................................................................................................................


class VStack(BaseCallback):
    """Layout element which stacks UI items together vertically"""

    # .................................................................................................................

    def __init__(
        self,
        *items: BaseCallback,
        flex: tuple | None = None,
        min_h: int | None = None,
        error_on_size_constraints: bool = False,
    ):

        # Inherit from parent, with dummy values (needed so we can get child iterator!)
        super().__init__(32, 32)
        self._append_cb_children(*items)

        # Update stack sizing based on children
        total_child_min_h = sum(child._cb_rdr.min_h for child in self)
        widest_child_min_w = max(child._cb_rdr.min_w for child in self)
        is_flex_h = any(child._cb_rdr.is_flexible_h for child in self)
        is_flex_w = any(child._cb_rdr.is_flexible_w for child in self)
        min_h = max(total_child_min_h, min_h if min_h is not None else 1)
        min_w = widest_child_min_w
        self._cb_rdr = CBRenderSizing(min_h, min_w, is_flex_h, is_flex_w)

        # Default to sizing by aspect ratio if not given flex values
        # -> Don't use AR sizing if not flexible (implies stacking doesn't have target AR)
        multiple_ar_children = sum(child._get_dynamic_aspect_ratio() is not None for child in self) > 1
        self._size_by_ar = flex is None and (is_flex_w and is_flex_h) and multiple_ar_children

        # Pre-compute sizing info for handling flexible sizing
        is_flex_h_per_child = [child._cb_rdr.is_flexible_h for child in self]
        flex_per_child_list = _read_flex_values(is_flex_h_per_child, flex)
        fixed_height_of_children = 0
        for child, flex_val in zip(self, flex_per_child_list):
            is_flexible = flex_val > 1e-3
            fixed_height_of_children += 0 if is_flexible else child._cb_rdr.min_h
        self._fixed_flex_height = fixed_height_of_children
        self._cumlative_flex = np.cumsum(flex_per_child_list, dtype=np.float32)
        self._flex_debug = flex_per_child_list
        self._error_on_constraints = error_on_size_constraints

        # Set up element styling when having to pad child items
        self.style = UIStyle(
            pad_color=(0, 0, 0),
            pad_border_type=cv2.BORDER_CONSTANT,
        )

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Set up starting stack point, used to keep track of child callback regions
        x_stack = self._cb_region.x1
        y_stack = self._cb_region.y1

        if self._size_by_ar:
            h_per_child_list = [child._get_height_given_width(w) for child in self]
            total_h = sum(h_per_child_list)
            if total_h != h:
                fix_list = []
                flex_list = []
                for child, child_h in zip(self, h_per_child_list):
                    fix_list.append(0 if child._cb_rdr.is_flexible_h else child._cb_rdr.min_h)
                    flex_list.append(child_h if child._cb_rdr.is_flexible_h else 0)

                avail_h = h - sum(fix_list)
                cumulative_h = np.cumsum(flex_list, dtype=np.float32) * avail_h / sum(flex_list)
                flex_list = np.diff(np.int32(np.round(cumulative_h)), prepend=0).tolist()
                h_per_child_list = [fixed_h if fixed_h > 0 else flex_h for fixed_h, flex_h in zip(fix_list, flex_list)]
        else:
            # Assign per-element sizing, taking into account flex scaling
            avail_h = max(0, h - self._fixed_flex_height)
            flex_h_px = np.diff(np.int32(np.round(self._cumlative_flex * avail_h)), prepend=0).tolist()
            h_per_child_list = [flex_h if flex_h > 0 else child._cb_rdr.min_h for child, flex_h in zip(self, flex_h_px)]

        # Have each child item draw itself
        imgs_list = []
        for child, ch_render_h in zip(self, h_per_child_list):
            frame = child._render_up_to_size(ch_render_h, w)
            frame_h, frame_w = frame.shape[0:2]

            # Crop overly-wide images
            # -> Don't need to crop tall images, since v-stacking won't break!
            if frame_w > w:
                print(
                    f"Render sizing error! Expecting width: {w}, got {frame_w} ({child})",
                    "-> Will crop!",
                    sep="\n",
                )
                frame = frame[:, :w, :]
                frame_w = w

            # Adjust frame width if needed
            tpad, lpad, bpad, rpad = 0, 0, 0, 0
            need_pad = (frame_w < w) or (frame_h < ch_render_h)
            if need_pad:
                available_w = w - frame_w
                available_h = max(0, ch_render_h - frame_h)
                tpad, lpad = available_h // 2, available_w // 2
                bpad, rpad = available_h - tpad, available_w - lpad
                ptype = self.style.pad_border_type
                pcolor = self.style.pad_color
                frame = cv2.copyMakeBorder(frame, tpad, bpad, lpad, rpad, ptype, value=pcolor)
                # print(" vpad->", tpad, bpad, lpad, rpad)

            # Store image
            imgs_list.append(frame)

            # Provide callback region to child item
            x1, y1 = x_stack + lpad, y_stack + tpad
            x2, y2 = x1 + frame_w, y1 + frame_h
            child._cb_region.resize(x1, y1, x2, y2)

            # Update stacking point for next child
            y_stack = y2 + bpad

        out_img = np.vstack(imgs_list)
        if len(self._cb_parent_list) == 0:
            out_h, out_w = out_img.shape[0:2]
            x1, y1 = self._cb_region.x1, self._cb_region.y1
            x2, y2 = x1 + out_w, y1 + out_h
            self._cb_region.resize(x1, y1, x2, y2)

        return out_img

    # .................................................................................................................

    def _get_dynamic_aspect_ratio(self) -> float | None:
        if self._cb_rdr.is_flexible_h and self._cb_rdr.is_flexible_w:
            child_ar = (c._get_dynamic_aspect_ratio() for c in self)
            inv_ar_sum = sum(1 / ar if ar is not None else 0 for ar in child_ar)
            return None if inv_ar_sum == 0 else 1 / inv_ar_sum

        return None

    def _get_height_given_width(self, w: int) -> int:

        # Use fixed height if not flexible
        if not self._cb_rdr.is_flexible_h:
            return self._cb_rdr.min_h

        # Ask child elements for desired width and use total
        h_per_child_list = [child._get_height_given_width(w) for child in self]
        total_child_h = sum(h_per_child_list)
        return max(self._cb_rdr.min_h, total_child_h)

    def _get_width_given_height(self, h: int) -> int:
        """
        For v-stacking, we normally want to set a width since this must be shared
        for all elements in order to stack vertically. Here we don't know the width.

        If sizing by aspect ratio, we calculate the width from knowing that all
        items must stack to the target height, while sharing the same width:
            target_h = (w / ar1) + (w / ar2) + (w / ar3) + ...
            target_h = w * (1/ar1 + 1/ar2 + 1/ar3)
            Therefore, w = target_h / sum(1/ar for all item aspect ratios)

        If sizing by flex values, we first figure out how much 'height' each child
        should be assigned. Then each child is asked for it's render width, given
        the assigned height. We take the 'widest' child width as the width for stacking.

        Returns:
            render_width
        """

        # Use fixed width if not flexible
        if not self._cb_rdr.is_flexible_w:
            return self._cb_rdr.min_w

        # Allocate width based on child aspect ratios
        if self._size_by_ar:
            avail_h = max(1, h - self._fixed_flex_height)
            child_ar = (c._get_dynamic_aspect_ratio() for c in self)
            w = avail_h / sum(1 / ar if ar is not None else 0 for ar in child_ar)
            return max(self._cb_rdr.min_w, round(w))

        # Figure out per-element height based on flex assignment
        avail_h = max(0, h - self._fixed_flex_height)
        flex_sizing_px = np.diff(np.int32(np.round(self._cumlative_flex * avail_h)), prepend=0).tolist()
        h_per_child_list = [max(child._cb_rdr.min_h, flex_h) for child, flex_h in zip(self, flex_sizing_px)]

        # Sanity check
        if self._error_on_constraints:
            total_computed_h = sum(h_per_child_list)
            assert total_computed_h == h, f"Error computing target height ({self})! Target: {h}, got {total_computed_h}"

        # We'll say our width, given the target height, is that of the widest child element
        w_per_child_list = (child._get_width_given_height(h=h) for child, h in zip(self, h_per_child_list))
        return max(w_per_child_list)

    # .................................................................................................................


class GridStack(BaseCallback):
    """
    Layout which combines elements into a grid with a specified number of rows and columns
    Items should be given in 'row-first' format (i.e. items fill out grid row-by-row)
    """

    # .................................................................................................................

    def __init__(self, *items, num_rows=None, num_columns=None, target_aspect_ratio=1):

        super().__init__(128, 128)
        self._append_cb_children(*items)

        # Fill in missing row/column counts
        num_items = len(items)
        if num_rows is None and num_columns is None:
            num_rows, num_columns = self.get_row_column_by_aspect_ratio(num_items, target_aspect_ratio)
        elif num_rows is None:
            num_rows = int(np.ceil(num_items / num_columns))
        elif num_columns is None:
            num_columns = int(np.ceil(num_items / num_rows))
        self._num_rows = num_rows
        self._num_cols = num_columns

        # Update stack sizing based on children
        min_h_per_row = []
        for _, child_per_row in self.row_iter():
            min_h_per_row.append(max(child._cb_rdr.min_h for child in child_per_row))
        min_w_per_col = []
        for _, child_per_col in self.column_iter():
            min_w_per_col.append(max(child._cb_rdr.min_w for child in child_per_col))
        total_min_h = sum(min_h_per_row)
        total_min_w = sum(min_w_per_col)
        is_flex_h = any(child._cb_rdr.is_flexible_h for child in self)
        is_flex_w = any(child._cb_rdr.is_flexible_w for child in self)
        self._cb_rdr = CBRenderSizing(total_min_h, total_min_w, is_flex_h, is_flex_w)

        self.style = UIStyle(
            pad_color=(0, 0, 0),
            pad_border_type=cv2.BORDER_CONSTANT,
        )

    # .................................................................................................................

    def get_row_columns(self) -> tuple[int, int]:
        """Get current row/column count of the grid layout"""
        return (self._num_rows, self._num_cols)

    # .................................................................................................................

    def transpose(self) -> SelfType:
        """Flip number of rows & columns"""
        self._num_rows, self._num_cols = self._num_cols, self._num_rows
        return self

    # .................................................................................................................

    def row_iter(self) -> tuple[int, tuple]:
        """
        Iterator over items per row. Example:
            for row_idx, items_in_row in grid.row_iter():
                # ... Do something with each row ...

                for col_idx, item for enumerate(items_in_row):
                    # ... Do something with each item ...
                    pass
                pass
        """

        for row_idx in range(self._num_rows):
            idx1 = row_idx * self._num_cols
            idx2 = idx1 + self._num_cols
            items_in_row = self[idx1:idx2]
            if len(items_in_row) == 0:
                break
            yield row_idx, tuple(items_in_row)

        return

    # .................................................................................................................

    def column_iter(self) -> tuple[int, tuple]:
        """
        Iterator over items per column. Example:
            for col_idx, items_in_column in grid.column_iter():
                # ... Do something with each column ...

                for row_idx, item for enumerate(items_in_column):
                    # ... Do something with each item ...
                    pass
                pass
        """

        num_items = len(self)
        for col_idx in range(self._num_cols):
            item_idxs = [col_idx + row_idx * self._num_cols for row_idx in range(self._num_rows)]
            items_in_column = tuple(self[item_idx] for item_idx in item_idxs if item_idx < num_items)
            if len(items_in_column) == 0:
                break
            yield col_idx, items_in_column

        return

    # .................................................................................................................

    def grid_iter(self) -> tuple[int, int, BaseCallback]:
        """
        Iterator over all items while returning row/column index. Example:
            for row_idx, col_idx, item in grid.grid_iter():
                # ... Do something with each item ...
                pass
        """

        for item_idx, item in enumerate(self):
            row_idx = item_idx // self._num_cols
            col_idx = item_idx % self._num_cols
            yield row_idx, col_idx, item

        return

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Set up starting stack point, used to keep track of child callback regions
        x_stack = self._cb_region.x1
        y_stack = self._cb_region.y1

        # Figure out how tall each row should be
        ideal_h_per_row = h // self._num_rows
        h_gap = h % self._num_rows
        h_per_row = [ideal_h_per_row + int(idx < h_gap) for idx in range(self._num_rows)]

        # Figure out how wide each column should be
        ideal_w_per_col = w // self._num_cols
        w_gap = w % self._num_cols
        w_per_col = [ideal_w_per_col + int(idx < w_gap) for idx in range(self._num_cols)]

        # Render all child items to target sizing
        row_images_list = []
        for row_idx, children_per_row in self.row_iter():

            row_height = h_per_row[row_idx]
            col_images_list = []
            for col_idx, child in enumerate(children_per_row):
                col_width = w_per_col[col_idx]
                frame = child._render_up_to_size(row_height, col_width)
                frame_h, frame_w = frame.shape[0:2]

                # Adjust frame width if needed
                tpad, bpad, lpad, rpad = 0, 0, 0, 0
                if (frame_h < row_height) or (frame_w < col_width):
                    available_h, available_w = row_height - frame_h, col_width - frame_w
                    lpad = available_w // 2
                    rpad = available_w - lpad
                    tpad = available_h // 2
                    bpad = available_h - tpad
                    ptype = self.style.pad_border_type
                    pcolor = self.style.pad_color
                    frame = cv2.copyMakeBorder(frame, tpad, bpad, lpad, rpad, ptype, value=pcolor)

                # Crop oversized heights
                if frame_h > row_height:
                    print(
                        f"Render sizing error! ({child})",
                        f"  Expecting height: {h}, got {frame_h}",
                        "-> Will crop!",
                        sep="\n",
                    )
                    frame = frame[:row_height, :, :]
                    frame_h = row_height

                # Crop oversized widths
                if frame_w > col_width:
                    print(
                        f"Render sizing error! ({child})",
                        f"  Expecting width: {w}, got {frame_w}",
                        "-> Will crop!",
                        sep="\n",
                    )
                    frame = frame[:, :col_width, :]
                    frame_w = col_width

                # Store image for forming row-images
                col_images_list.append(frame)

                # Provide callback region to child item
                x1, y1 = x_stack + lpad, y_stack + tpad
                x2, y2 = x1 + frame_w, y1 + frame_h
                child._cb_region.resize(x1, y1, x2, y2)

                # Update x-stacking point for each column
                x_stack = x2 + rpad

            # Combine all column images to form one row image, padding if needed
            one_row_image = np.hstack(col_images_list)
            _, one_row_w = one_row_image.shape[0:2]
            if one_row_w < w:
                pad_w = w - one_row_w
                ptype = self.style.pad_border_type
                pcolor = self.style.pad_color
                one_row_image = cv2.copyMakeBorder(one_row_image, 0, 0, 0, pad_w, ptype, value=pcolor)
            row_images_list.append(one_row_image)

            # Reset x-stacking point & update y-stacking point, for each completed row
            x_stack = self._cb_region.x1
            y_stack = y_stack + row_height

        return np.vstack(row_images_list)

    # .................................................................................................................

    def _get_height_and_width_without_hint(self) -> HWPX:
        """Set height to the total of largest heights per row, width to the total largest widths per column"""

        # Set height based on largest heights per row
        max_h_per_row = []
        for _, items_per_row in self.row_iter():
            max_h_per_row.append(max(item._cb_rdr.min_h for item in items_per_row))
        height = sum(max_h_per_row)

        # Set width based on largest widths per column
        max_w_per_col = []
        for _, items_per_col in self.column_iter():
            max_w_per_col.append(max(item._cb_rdr.min_w for item in items_per_col))
        width = sum(max_w_per_col)

        return height, width

    def _get_height_given_width(self, w: int) -> int:
        """Set height to the sum of the tallest elements per row"""

        # Figure out width of each column (assuming equal assignment)
        ideal_w_per_col = w // self._num_cols
        w_gap = w % self._num_cols
        w_per_col = [ideal_w_per_col + int(idx < w_gap) for idx in range(self._num_cols)]

        # Sum up all widths per row
        heights_per_row = [[] for _ in range(self._num_rows)]
        for row_idx, col_idx, child in self.grid_iter():
            heights_per_row[row_idx].append(child._get_height_given_width(w=w_per_col[col_idx]))

        # Set height to the total height based on tallest item per row stacked together
        max_height_per_row = [max(row_heights) for row_heights in heights_per_row]
        return sum(max_height_per_row)

    def _get_width_given_height(self, h: int) -> int:
        """Set width to the widest row"""

        # Figure out height of each row (assuming equal assignment)
        ideal_h_per_row = h // self._num_rows
        h_gap = h % self._num_rows
        h_per_row = [ideal_h_per_row + int(idx < h_gap) for idx in range(self._num_rows)]

        # Sum up all widths per row
        total_w_per_row = [0] * self._num_rows
        for row_idx, col_idx, child in self.grid_iter():
            total_w_per_row[row_idx] += child._get_width_given_height(h=h_per_row[row_idx])

        # Set width to the widest row
        return max(total_w_per_row)

    # .................................................................................................................

    @staticmethod
    def get_row_column_options(num_items: int) -> tuple[tuple[int, int]]:
        """
        Helper used to get all possible neatly divisible combinations of (num_rows, num_columns)
        for a given number of items, in order of fewest rows -to- most rows.
        For example for num_items = 6, returns:
            ((1, 6), (2, 3), (3, 2), (6, 1))
            -> This is meant to be interpreted as:
                (1 row, 6 columns) OR (2 rows, 3 columns) OR (3 rows, 2 columns) OR (6 rows, 1 column)

        As another example, for num_items = 12, returns:
            ((1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1))
        """
        return tuple((k, num_items // k) for k in range(1, 1 + num_items) if (num_items % k) == 0)

    # .................................................................................................................

    @staticmethod
    def get_aspect_ratio_similarity(
        row_column_options: tuple[tuple[int, int]], target_aspect_ratio: float
    ) -> list[float]:
        """
        Compute similarity score (0 to 1) indicating how close of match
        each row/column option is to the target aspect ratio.

        Note that the row_column_options are expected to come from the
        .get_row_column_options(...) method
        """
        target_theta, pi_over_2 = np.arctan(target_aspect_ratio), np.pi / 2
        difference_scores = (abs(np.arctan(col / row) - target_theta) for row, col in row_column_options)
        return tuple(float(1.0 - (diff / pi_over_2)) for diff in difference_scores)

    # .................................................................................................................

    @classmethod
    def get_row_column_by_aspect_ratio(cls, num_items: int, target_aspect_ratio: float = 1.0) -> tuple[int, int]:
        """
        Helper used to choose the number of rows & columns to best match a target aspect ratio
        Returns: (num_rows, num_columns)
        """

        rc_options = cls.get_row_column_options(num_items)
        ar_similarity = cls.get_aspect_ratio_similarity(rc_options, target_aspect_ratio)
        best_match_idx = np.argmax(ar_similarity)

        return rc_options[best_match_idx]

    # .................................................................................................................


class OverlayStack(BaseCallback):
    """
    Element used to combine multiple overlays onto a single base item.
    (i.e. stacks overlays ontop of one another).

    This is mainly intended for better efficiency when using many overlays together.
    """

    # .................................................................................................................

    def __init__(self, base_item: BaseCallback, *overlay_items: BaseOverlay, suppress_callbacks_to_base: bool = False):

        # Inherit from parent and copy base item render limits
        super().__init__(32, 32)

        # Clear overlay children, so we don't get duplicate base item calls
        # (only this parent instance will handle callbacks & pass these down to children)
        self._cb_rdr = self._base_item._cb_rdr.copy()
        for olay in overlay_items:
            olay._cb_rdr = self._base_item._cb_rdr.copy()
            olay._cb_child_list.clear()
        self._overlay_items = tuple(overlay_items)

        # Store base & overlays for future reference and enable callbacks for children
        self._base_item = base_item
        if not suppress_callbacks_to_base:
            self._append_cb_children(self._base_item)
        self._append_cb_children(*overlay_items)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        olay_names = [str(olay) for olay in self._overlay_items]
        return f"{cls_name} [{self._base_item} | {', '.join(olay_names)}]"

    # .................................................................................................................

    def add_overlays(self, *overlay_items: BaseOverlay) -> SelfType:
        """Function used to add overlays (after init)"""

        olays_list = list(self._overlay_items)
        olays_list.extend(overlay_items)
        self._overlay_items = tuple(olays_list)
        self._append_cb_children(*overlay_items)

        return self

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Set up starting stack point, used to keep track of child callback regions
        x1 = self._cb_region.x1
        y1 = self._cb_region.y1

        # Have base item provide the base frame rendering and overlays handle drawing over-top
        base_frame = self._base_item._render_up_to_size(h, w).copy()
        base_h, base_w = base_frame.shape[0:2]

        x2, y2 = x1 + base_w, y1 + base_h
        self._base_item._cb_region.resize(x1, y1, x2, y2)
        for overlay in self._overlay_items:
            overlay._cb_region.resize(x1, y1, x2, y2)
            base_frame = overlay._render_overlay(base_frame)

        return base_frame

    # .................................................................................................................

    def _get_dynamic_aspect_ratio(self):
        return self._base_item._get_dynamic_aspect_ratio()

    def _get_height_and_width_without_hint(self) -> HWPX:
        return self._base_item._get_height_and_width_without_hint()

    def _get_height_given_width(self, w: int) -> int:
        return self._base_item._get_height_given_width(w)

    def _get_width_given_height(self, h: int) -> int:
        return self._base_item._get_width_given_height(h)

    # .................................................................................................................


class Swapper(BaseCallback):
    """
    Special layout item which allows for swapping between elements.
    This can be used to switch between different UI layouts, for example.

    Items can be swapped between using .set_swap_index(...), which
    will switch to items based on the indexing order used when
    initializing the swap instance.
    Alternatively, an optional 'keys' init argument can be used
    to assign labels, which can be swapped between using
    the .set_swap_key(...) function.
    """

    # .................................................................................................................

    def __init__(self, *swap_items: BaseCallback, initial_index: int = 0, keys: Iterable[Any] | None = None):

        # Fill in missing keys
        swap_items = tuple(swap_items)
        if keys is None:
            keys = range(len(swap_items))
        keys = tuple(keys)

        # Set up per-item/key removal flag
        is_none_keys = [k is None for k in keys]
        is_none_items = [item is None for item in swap_items]
        need_remove_list = [any(none_key_or_item) for none_key_or_item in zip(is_none_keys, is_none_items)]

        items = tuple(item for item, remove in zip(swap_items, need_remove_list) if not remove)
        keys = tuple(key for key, remove in zip(keys, need_remove_list) if not remove)
        # reverse_key_lut = {item: key for item, key in zip(items, keys)}

        # Sanity check
        num_keys, num_items = len(keys), len(items)
        assert len(set(keys)) == num_keys, f"Cannot have duplicate keys: {keys}"
        assert num_keys == num_items, f"Number of keys ({num_keys}) must match number of swap items ({num_items})"

        # Store items for swapping
        self._items: tuple[BaseCallback] = items
        self._swap_idx: int = initial_index
        self._num_items: int = len(self._items)
        self._key_lut = {key: idx for idx, key in enumerate(keys)}

        item_rdr = self._items[initial_index]._cb_rdr
        super().__init__(item_rdr.min_h, item_rdr.min_w, item_rdr.is_flexible_h, item_rdr.is_flexible_w)

    # .................................................................................................................

    def _update_render_sizing(self):
        item_rdr = self._items[self._swap_idx]._cb_rdr
        self._cb_rdr = CBRenderSizing(item_rdr.min_h, item_rdr.min_w, item_rdr.is_flexible_h, item_rdr.is_flexible_w)
        return None

    # .................................................................................................................

    def set_swap_index(self, swap_index: int) -> BaseCallback:
        """
        Swap to a new item, by index
        Returns:
            current_swap_item
        """
        if 0 <= swap_index < self._num_items:
            if swap_index != self._swap_idx:
                self._swap_idx = swap_index
                self._update_render_sizing()
        return self._items[self._swap_idx]

    def set_swap_key(self, swap_key: Any) -> BaseCallback:
        """
        Swap to a new item, by key name. Keys can be specified on init.
        Returns:
            current_swap_item
        """
        new_idx = self._key_lut[swap_key]
        return self.set_swap_index(new_idx)

    # .................................................................................................................

    def next(self, increment=1) -> BaseCallback:
        new_idx = (self._swap_idx + increment) % self._num_items
        return self.set_swap_index(new_idx)

    def prev(self, decrement=1) -> BaseCallback:
        return self.next(-decrement)

    # .................................................................................................................

    def _render_up_to_size(self, h, w):
        parent = self._cb_region
        item = self._items[self._swap_idx]
        item._cb_region.resize(parent.x1, parent.y1, parent.x2, parent.y2)
        return item._render_up_to_size(h, w)

    def _get_dynamic_aspect_ratio(self):
        return self._items[self._swap_idx]._get_dynamic_aspect_ratio()

    def _get_height_and_width_without_hint(self):
        return self._items[self._swap_idx]._get_height_and_width_without_hint()

    def _get_height_given_width(self, w):
        return self._items[self._swap_idx]._get_height_given_width(w)

    def _get_width_given_height(self, h):
        return self._items[self._swap_idx]._get_width_given_height(h)

    def _cb_iter(self, global_x_px: int, global_y_px: int):
        if not self._cb_state.disabled:
            child = self._items[self._swap_idx]
            if not child._cb_state.disabled:
                yield from child._cb_iter(global_x_px, global_y_px)
        return

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def _read_flex_values(
    item_is_flexible_list: list[bool],
    flex: tuple[float] | None,
    allow_undersizing: bool = True,
) -> tuple[float]:
    """
    Helper used to compute normalized flex values
    - 'None' values are interpreted as 'fallback' to item flexibility
    - If allow_undersizing is False, then normalized flex values will sum to 1
    - If allow_undersizing is True, then values less than 1 will remain as-is
      (so sum can be less than 1). This allows for 'shrinking' UI elements
      below given space allocation. For example, flex=(0.25, 0.25), would
      result in the items taking up only half of the total available space.
    """

    # If no flex sizing is given, default to 'fallback' for every item
    if flex is None:
        flex = [None] * len(item_is_flexible_list)

    # Sanity check, make sure we have flex sizing for each callback item
    flex = tuple(flex)
    num_items = len(item_is_flexible_list)
    assert len(flex) == num_items, f"Flex error! Must match number of entries ({num_items}), got: flex={flex}"

    # Iterpret flex values of None as fallback to item flexibility
    out_flex = (val if val is not None else float(is_flex) for val, is_flex in zip(flex, item_is_flexible_list))
    out_flex = tuple(max(0, val) for val in out_flex)

    # Normalize flex values so they can be used as weights when deciding render sizes
    total_flex = sum(out_flex)
    if allow_undersizing and total_flex < 0.99:
        total_flex = 1
    elif total_flex <= 0:
        total_flex = 1
    return tuple(float(val) / float(total_flex) for val in out_flex)
