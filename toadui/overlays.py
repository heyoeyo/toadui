#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from toadui.base import BaseCallback, BaseOverlay, CBEventXY
from toadui.helpers.styling import UIStyle
from toadui.helpers.images import draw_normalized_polygon
from toadui.helpers.text import TextDrawer

# Typing
from numpy import ndarray
from toadui.helpers.types import COLORU8, XYPX, XYNORM, HWPX, XY1XY2NORM, SelfType
from toadui.helpers.ocv_types import OCVInterp, OCVLineType, OCVFont


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class DrawPolygonsOverlay(BaseOverlay):
    """Simple overlay which draws polygons over top of base images"""

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        color: COLORU8 = (0, 255, 255),
        bg_color: COLORU8 | None = None,
        thickness: int = 2,
        line_type: OCVLineType = cv2.LINE_AA,
        is_closed: bool = True,
    ):

        self._poly_xy_norm_list: list[XYNORM] = []

        self.style = UIStyle(
            color=color,
            color_bg=bg_color,
            thickness=thickness,
            line_type=line_type,
            is_closed=is_closed,
        )

        super().__init__(base_item)

    # .................................................................................................................

    def clear(self) -> SelfType:
        self._poly_xy_norm_list = []
        return self

    # .................................................................................................................

    def set_polygons(self, *polygon_xy_norm_list: list[XYNORM] | ndarray) -> SelfType:
        """
        Set or replace polygons. Polygons should be given as either a list/tuple
        of normalized xy coordinates or as an Nx2 numpy array, where N is the
        number of points in the polygon. More than one polygon can be provided.

        For example:
            poly1 = np.float32([(0.25,0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
            poly2 = [(0.25, 0.25), (0.85, 0.5), (0.25, 0.5)]
            set_polygons(poly1, poly2)
        """

        # if isinstance(polygon_xy_norm_list, ndarray):
        #     polygon_xy_norm_list = [polygon_xy_norm_list]
        self._poly_xy_norm_list = tuple([*polygon_xy_norm_list])

        return self

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        if self._poly_xy_norm_list is None:
            return frame

        if len(self._poly_xy_norm_list) == 0:
            return frame

        out_frame = frame.copy()
        for poly in self._poly_xy_norm_list:
            out_frame = draw_normalized_polygon(
                out_frame,
                poly,
                self.style.color,
                self.style.thickness,
                self.style.color_bg,
                self.style.line_type,
                self.style.is_closed,
            )

        return out_frame

    # .................................................................................................................


class DrawMaskOverlay(BaseOverlay):
    """Simple overlay which draws a binary mask over top of a base image"""

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        mask_color: COLORU8 = (90, 0, 255),
        scaling_interpolation: OCVInterp = cv2.INTER_NEAREST,
    ):

        # Storage for mask data
        self._inv_mask = None
        self._mask_bgr = None
        self._interpolation = scaling_interpolation
        self._color = np.uint8(mask_color)

        # Storage for cached data
        self._cached_h = None
        self._cached_w = None
        self._cached_mask_bgr = None
        self._cached_inv_mask = None

        super().__init__(base_item)

    # .................................................................................................................

    def clear(self) -> SelfType:
        """Clear all mask data"""
        self._inv_mask = None
        self._mask_bgr = None
        self._cached_inv_mask = None
        self._cached_mask_bgr = None
        self._cached_h = None
        self._cached_w = None
        return self

    # .................................................................................................................

    def set_mask(self, mask: ndarray, mask_threshold: int = 0) -> SelfType:
        """Update mask used for overlay"""

        mask_bin = np.uint8(mask > mask_threshold)
        self._mask_bgr = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR) * self._color
        self._inv_mask = cv2.cvtColor(cv2.bitwise_not(mask_bin * 255), cv2.COLOR_GRAY2BGR)

        # Update cached values, in case mask already matches target frame size
        self._cached_mask_bgr = self._mask_bgr
        self._cached_inv_mask = self._inv_mask
        self._cached_h, self._cached_w = mask_bin.shape[0:2]

        return self

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        # Skip overlay if we don't have a mask
        if self._mask_bgr is None:
            return frame

        # Re-build cached masks if sizing changes
        frame_h, frame_w = frame.shape[0:2]
        if (frame_h != self._cached_h) or (frame_w != self._cached_w):
            frame_wh = (frame_w, frame_h)
            self._cached_inv_mask = cv2.resize(self._inv_mask, dsize=frame_wh, interpolation=self._interpolation)
            self._cached_mask_bgr = cv2.resize(self._mask_bgr, dsize=frame_wh, interpolation=self._interpolation)
            self._cached_h, self._cached_w = frame_h, frame_w

        inv_frame = cv2.bitwise_and(frame, self._cached_inv_mask)
        return cv2.add(inv_frame, self._cached_mask_bgr)

    # .................................................................................................................


class TextOverlay(BaseOverlay):
    """Overlay used to draw text over a base image"""

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        xy_norm: XYNORM = (0.5, 0.5),
        scale: float = 0.5,
        thickness: int = 1,
        color: COLORU8 = (255, 255, 255),
        bg_color: COLORU8 = (0, 0, 0),
        font: OCVFont = cv2.FONT_HERSHEY_SIMPLEX,
        line_type: OCVLineType = cv2.LINE_AA,
        anchor_xy_norm: XYNORM | None = None,
        offset_xy_px: XYPX = (0, 0),
    ):
        super().__init__(base_item)
        self._text = None
        self._xy_norm = xy_norm
        self._anchor_xy_norm = anchor_xy_norm
        self._offset_xy_px = offset_xy_px
        self.style = UIStyle(text=TextDrawer(scale, thickness, color, bg_color, font, line_type))
        self.style.text = TextDrawer(scale, thickness, color, bg_color, font, line_type)

    # .................................................................................................................

    def set_text(self, text: str | None) -> SelfType:
        self._text = text
        return self

    # .................................................................................................................

    def set_postion(
        self,
        xy_norm: XYNORM | None = None,
        anchor_xy_norm: XYNORM | None = None,
        offset_xy_px: XYPX | None = None,
    ) -> SelfType:

        if xy_norm is not None:
            self._xy_norm = xy_norm
        if anchor_xy_norm is not None:
            self._anchor_xy_norm = anchor_xy_norm
        if offset_xy_px is not None:
            self._offset_xy_px = offset_xy_px

        return self

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        if self._text is None:
            return frame

        return self.style.text.xy_norm(frame, self._text, self._xy_norm, self._anchor_xy_norm, self._offset_xy_px)

    # .................................................................................................................


class PointClickOverlay(BaseOverlay):
    """
    Overlay which allows for clicking to add points over top of a base image
    Multiple points can be added by shift clicking. Right click removes points.
    """

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        color: COLORU8 = (0, 255, 255),
        radius: int = 4,
        bg_color: COLORU8 | None = (0, 0, 0),
        thickness: int = -1,
        line_type: OCVLineType = cv2.LINE_AA,
        max_points: int | None = None,
    ):
        # Inherit from parent
        super().__init__(base_item)

        # Store point state
        self._xy_norm_list: list[tuple[float, float]] = []
        self._is_changed = False
        self._max_points = int(max_points) if max_points is not None else 1_000_000
        assert self._max_points > 0, "Must have max_points > 0"

        self.style = UIStyle(
            color_fg=color,
            color_bg=bg_color,
            radius_fg=radius,
            radius_bg=radius if thickness > 0 else radius + 1,
            thickness_fg=thickness,
            thickness_bg=max(1 + thickness, 2 * thickness) if thickness > 0 else thickness,
            line_type=line_type,
        )

    # .................................................................................................................

    def clear(self, flag_is_changed: bool = True) -> SelfType:
        self._is_changed = (len(self._xy_norm_list) > 0) and flag_is_changed
        self._xy_norm_list = []
        return self

    # .................................................................................................................

    def read(self) -> tuple[bool, tuple]:
        """Returns: is_changed, xy_norm_list"""
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, tuple(self._xy_norm_list)

    # .................................................................................................................

    def _on_left_click(self, cbxy: CBEventXY, cbflags) -> None:

        # Add point if shift clicked or update point otherwise
        new_xy_norm = cbxy.xy_norm
        if cbflags.shift_key:
            self.add_points(new_xy_norm)
        else:
            if len(self._xy_norm_list) == 0:
                self._xy_norm_list = [new_xy_norm]
            else:
                self._xy_norm_list[-1] = new_xy_norm

        self._is_changed = True

        return

    def _on_right_click(self, cbxy: CBEventXY, cbflags) -> None:
        self.remove_closest(cbxy.xy_norm, cbxy.hw_px)
        return

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        # Convert points to pixel coords for drawing
        frame_h, frame_w = frame.shape[0:2]
        norm_to_px_scale = np.float32((frame_w - 1, frame_h - 1))
        xy_px_list = [np.int32(xy_norm * norm_to_px_scale) for xy_norm in self._xy_norm_list]

        # Draw each point as a circle with a background if needed
        if self.style.color_bg is not None:
            for xy_px in xy_px_list:
                cv2.circle(
                    frame,
                    xy_px,
                    self.style.radius_bg,
                    self.style.color_bg,
                    self.style.thickness_bg,
                    self.style.line_type,
                )
        for xy_px in xy_px_list:
            cv2.circle(
                frame,
                xy_px,
                self.style.radius_fg,
                self.style.color_fg,
                self.style.thickness_fg,
                self.style.line_type,
            )

        return frame

    # .................................................................................................................

    def add_points(self, *xy_norm_points: XYNORM) -> SelfType:

        if len(xy_norm_points) == 0:
            return self

        # Remove earlier points if needed
        if len(self._xy_norm_list) >= self._max_points:
            self._xy_norm_list.pop(0)

        self._xy_norm_list.extend(xy_norm_points)
        self._is_changed = True

        return self

    # .................................................................................................................

    def remove_closest(self, xy_norm: XYNORM, frame_hw: HWPX | None = None) -> None | XYNORM:

        # Can't remove points if there aren't any!
        if len(self._xy_norm_list) == 0:
            return None

        # Default to 'fake' pixel count if not given (so we can re-use the same calculations)
        if frame_hw is None:
            frame_hw = (10, 10)
        frame_h, frame_w = frame_hw
        norm_to_px_scale = np.float32((frame_w - 1, frame_h - 1))

        # Find the point closest to the given (x,y) for removal
        xy_px_array = np.int32([np.int32(xy_norm * norm_to_px_scale) for xy_norm in self._xy_norm_list])
        input_array = np.int32(xy_norm * norm_to_px_scale)
        dist_to_pts = np.linalg.norm(xy_px_array - input_array, ord=2, axis=1)
        closest_pt_idx = np.argmin(dist_to_pts)

        # Remove the point closest to the click and finish
        closest_xy_norm = self._xy_norm_list.pop(closest_pt_idx)
        self._is_changed = True

        return closest_xy_norm

    # .................................................................................................................


class BoxSelectOverlay(BaseOverlay):

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        color: COLORU8 = (0, 255, 255),
        thickness: int = 1,
        bg_color: COLORU8 | None = (0, 0, 0),
    ):
        super().__init__(base_item)
        self._xy1xy2_norm_list: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._xy1xy2_norm_inprog = None
        self._is_changed = False

        # Store display config
        self._fg_color = color
        self._bg_color = bg_color
        self._fg_thick = thickness
        self._bg_thick = thickness + 1
        self._ltype = cv2.LINE_4

    # .................................................................................................................

    def style(self, color=None, thickness=None, bg_color=None, bg_thickness=None) -> SelfType:
        """Update box styling. Any settings given as None will remain unchanged"""

        if color is not None:
            self._fg_color = color
        if thickness is not None:
            self._fg_thick = thickness
        if bg_color is not None:
            self._bg_color = bg_color if bg_color != -1 else None
        if bg_thickness is not None:
            self._bg_thick = bg_thickness

        return self

    # .................................................................................................................

    def clear(self, flag_is_changed: bool = True) -> SelfType:
        had_boxes = (len(self._xy1xy2_norm_list) > 0) or (self._xy1xy2_norm_inprog is not None)
        self._is_changed = had_boxes and flag_is_changed
        self._xy1xy2_norm_list = []
        self._xy1xy2_norm_inprog = None
        return self

    # .................................................................................................................

    def read(self, include_in_progress_box: bool = True) -> tuple[bool, tuple]:
        """Returns: is_changed, box_xy1xy2_list"""

        # Toggle change state, if needed
        is_changed = self._is_changed
        self._is_changed = False

        # Get list of boxes including in-progress box if needed
        out_list = self._xy1xy2_norm_list
        if include_in_progress_box:
            is_valid, extra_tlbr = self._make_inprog_tlbr()
            extra_xy1xy2_list = [extra_tlbr] if is_valid else []
            out_list = self._xy1xy2_norm_list + extra_xy1xy2_list

        return is_changed, tuple(out_list)

    # .................................................................................................................

    def _on_left_down(self, cbxy: CBEventXY, cbflags):

        # Ignore clicks outside of region
        if not cbxy.is_in_region:
            return

        # Begin new 'in-progress' box
        self._xy1xy2_norm_inprog = [cbxy.xy_norm, cbxy.xy_norm]

        # Remove newest box if we're not shift-clicking
        if not cbflags.shift_key:
            if len(self._xy1xy2_norm_list) > 0:
                self._xy1xy2_norm_list.pop()

        self._is_changed = True

        return

    def _on_drag(self, cbxy: CBEventXY, cbflags):

        # Update second in-progress box point
        if self._xy1xy2_norm_inprog is not None:
            new_xy = np.clip(cbxy.xy_norm, 0.0, 1.0)
            self._xy1xy2_norm_inprog[1] = tuple(new_xy)
            self._is_changed = True

        return

    def _on_left_up(self, cbxy: CBEventXY, cbflags):

        is_valid, new_tlbr = self._make_inprog_tlbr()
        if is_valid:
            self._xy1xy2_norm_list.append(new_tlbr)
            self._is_changed = True
        self._xy1xy2_norm_inprog = None

        return

    def _on_right_click(self, cbxy: CBEventXY, cbflags):
        self.remove_closest(cbxy.xy_norm, cbxy.hw_px)
        return

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        # Check if we need to draw an in-progress box
        is_valid, new_tlbr = self._make_inprog_tlbr()
        extra_tlbr = [new_tlbr] if is_valid else []
        boxes_to_draw = self._xy1xy2_norm_list + extra_tlbr

        frame_h, frame_w = frame.shape[0:2]
        norm_to_px_scale = np.float32((frame_w - 1, frame_h - 1))
        box_px_list = []
        for box in boxes_to_draw:
            box = np.int32([xy_norm * norm_to_px_scale for xy_norm in box])
            box_px_list.append(box)

        if self._bg_color is not None:
            for xy1_px, xy2_px in box_px_list:
                cv2.rectangle(frame, xy1_px, xy2_px, self._bg_color, self._bg_thick, self._ltype)
        for xy1_px, xy2_px in box_px_list:
            cv2.rectangle(frame, xy1_px, xy2_px, self._fg_color, self._fg_thick, self._ltype)

        return frame

    # .................................................................................................................

    def add_boxes(self, *xy1xy2_norm_list) -> SelfType:

        if len(xy1xy2_norm_list) == 0:
            return self

        self._xy1xy2_norm_list.extend(xy1xy2_norm_list)
        self._is_changed = True

        return self

    # .................................................................................................................

    def remove_closest(self, xy_norm: XYNORM, frame_hw: HWPX = None) -> None | XY1XY2NORM:

        # Can't remove boxes if there aren't any!
        if len(self._xy1xy2_norm_list) == 0:
            return None

        # Default to 'fake' pixel count if not given (so we can re-use the same calculations)
        if frame_hw is None:
            frame_hw = (10, 10)
        frame_h, frame_w = frame_hw
        norm_to_px_scale = np.float32((frame_w - 1, frame_h - 1))

        # For each box, find the distance to the closest corner
        input_array = np.int32(xy_norm * norm_to_px_scale)
        closest_dist_list = []
        for (x1, y1), (x2, y2) in self._xy1xy2_norm_list:
            xy_px_array = np.float32([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]) * norm_to_px_scale
            dist_to_pts = np.linalg.norm(xy_px_array - input_array, ord=2, axis=1)
            closest_dist_list.append(min(dist_to_pts))

        # Among all boxes, remove the one with the closest corner to the given click
        closest_pt_idx = np.argmin(closest_dist_list)
        closest_xy1xy2_norm = self._xy1xy2_norm_list.pop(closest_pt_idx)
        self._is_changed = True

        return closest_xy1xy2_norm

    # .................................................................................................................

    def _make_inprog_tlbr(self) -> tuple[bool, XY1XY2NORM]:
        """
        Helper used to make a 'final' box out of in-progress data
        Includes re-arranging points to be in proper top-left/bottom-right order
        as well as discarding boxes that are 'too small'
        """

        new_tlbr = None
        is_valid = self._xy1xy2_norm_inprog is not None
        if is_valid:

            # Re-arrange points to make sure first xy is top-left, second is bottom-right
            xy1_xy2 = np.float32(self._xy1xy2_norm_inprog)
            tl_xy_norm = xy1_xy2.min(0)
            br_xy_norm = xy1_xy2.max(0)

            # Make sure the box is not infinitesimally small
            xy_diff = br_xy_norm - tl_xy_norm
            is_valid = np.all(xy_diff > 1e-4)
            if is_valid:
                new_tlbr = (tl_xy_norm.tolist(), br_xy_norm.tolist())

        return is_valid, new_tlbr

    # .................................................................................................................


class EditBoxOverlay(BaseOverlay):
    """
    Overlay used to provide a 'crop-box' or similar UI
    The idea being to have a single box that can be modified
    by clicking and dragging the corners or sides, or otherwise
    fully re-drawn by clicking far enough away from the box.
    It is always assumed that there is 1 box!

    This differs from the regular 'box select overlay' which
    re-draws boxes on every click and supports multiple boxes
    """

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        frame_shape=None,
        color: COLORU8 = (0, 255, 255),
        thickness: int = 1,
        bg_color: COLORU8 | None = (0, 0, 0),
        indicator_base_radius: int = 6,
        interaction_distance_px: float = 100,
        minimum_box_area_norm: float = 5e-5,
    ):
        # Inherit from parent
        super().__init__(base_item)

        # Store box points in format that supports 'mid points'
        self._x_norms = np.float32([0.0, 0.5, 1.0])
        self._y_norms = np.float32([0.0, 0.5, 1.0])
        self._prev_xy_norms = (self._x_norms, self._y_norms)
        self._is_changed = False

        # Store indexing used to specify which of the box points is being modified, if any
        self._is_modifying = False
        self._xy_modify_idx = (2, 2)
        self._mouse_xy_norm = (0.0, 0.0)

        # Store sizing of frame being cropped, only use when 'nudging' the crop box
        self._full_frame_hw = frame_shape[0:2] if frame_shape is not None else (100, 100)

        # Store thresholding settings
        self._minimum_area_norm = minimum_box_area_norm
        self._interact_dist_px_threshold = interaction_distance_px

        # Store display config
        self._fg_color = color
        self._bg_color = bg_color
        self._fg_thick = thickness
        self._bg_thick = thickness + 1
        self._ltype = cv2.LINE_4
        self._ind_base_radius = indicator_base_radius
        self._ind_fg_radius = self._ind_base_radius + self._fg_thick
        self._ind_bg_radius = self._ind_fg_radius + (self._bg_thick - self._fg_thick)
        self._ind_ltype = cv2.LINE_AA

    # .................................................................................................................

    def style(self, color=None, thickness=None, bg_color=None, bg_thickness=None) -> SelfType:
        """Update box styling. Any settings given as None will remain unchanged"""

        if color is not None:
            self._fg_color = color
        if thickness is not None:
            self._fg_thick = thickness
            self._ind_fg_radius = self._ind_base_radius + self._fg_thick
        if bg_color is not None:
            self._bg_color = bg_color if bg_color != -1 else None
        if bg_thickness is not None:
            self._bg_thick = bg_thickness
            self._ind_bg_radius = self._ind_fg_radius + (self._bg_thick - self._fg_thick)

        return self

    # .................................................................................................................

    def clear(self) -> SelfType:
        """Reset box back to entire frame size"""
        self._x_norms = np.float32([0.0, 0.5, 1.0])
        self._y_norms = np.float32([0.0, 0.5, 1.0])
        self._is_changed = True
        self._is_modifying = False
        return self

    # .................................................................................................................

    def read(self) -> tuple[bool, bool, XY1XY2NORM]:
        """
        Read current box state
        Returns:
            is_changed, is_valid, box_xy1xy2_norm
            -> 'is_box_valid' is based on the minimum box area setting
            -> box_xy1xy2_norm is in format: ((x1, y1), (x2, y2))
        """

        # Toggle change state, if needed
        is_changed = self._is_changed
        self._is_changed = False

        # Get top-left/bottom-right output if it exists
        x1, _, x2 = sorted(self._x_norms.tolist())
        y1, _, y2 = sorted(self._y_norms.tolist())
        box_xy1xy2_norm = ((x1, y1), (x2, y2))
        is_valid = ((x2 - x1) * abs(y2 - y1)) > self._minimum_area_norm

        return is_changed, is_valid, box_xy1xy2_norm

    # .................................................................................................................

    def set_box(self, xy1xy2_norm: XY1XY2NORM) -> SelfType:
        """
        Update box coordinates. Input is expected in top-left/bottom-right format:
            ((x1, y1), (x2, y2))
        """

        (x1, y1), (x2, y2) = xy1xy2_norm
        x_mid, y_mid = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        self._x_norms = np.float32((x1, x_mid, x2))
        self._y_norms = np.float32((y1, y_mid, y2))
        self._is_changed = True
        self._is_modifying = False

        return self

    # .................................................................................................................

    def nudge(self, left: int = 0, right: int = 0, up: int = 0, down: int = 0) -> SelfType:
        """Helper used to move the position of a point (nearest to the mouse) by some number of pixels"""

        # Figure out which point to nudge
        (x_idx, y_idx), _, _ = self._check_xy_interaction(self._mouse_xy_norm, self._full_frame_hw)

        # Handle left/right nudge
        is_leftright_nudgable = x_idx != 1
        leftright_nudge = right - left
        if is_leftright_nudgable and leftright_nudge != 0:
            _, w_px = self._full_frame_hw
            old_x_norm = self._x_norms[x_idx]
            old_x_px = old_x_norm * (w_px - 1)
            new_x_px = old_x_px + leftright_nudge
            new_x_norm = new_x_px / (w_px - 1)
            new_x_norm = np.clip(new_x_norm, 0.0, 1.0)

            # Update target x coord and re-compute midpoint
            self._x_norms[x_idx] = new_x_norm
            self._x_norms[1] = (self._x_norms[0] + self._x_norms[-1]) * 0.5

        # Handle up/down nudge
        is_updown_nudgable = y_idx != 1
        updown_nudge = down - up
        if is_updown_nudgable and updown_nudge != 0:
            h_px, _ = self._full_frame_hw
            old_y_norm = self._y_norms[y_idx]
            old_y_px = old_y_norm * (h_px - 1)
            new_y_px = old_y_px + updown_nudge
            new_y_norm = new_y_px / (h_px - 1)
            new_y_norm = np.clip(new_y_norm, 0.0, 1.0)

            # Update target x coord and re-compute midpoint
            self._y_norms[y_idx] = new_y_norm
            self._y_norms[1] = (self._y_norms[0] + self._y_norms[-1]) * 0.5

        # Assume we've changed the box
        self._is_changed = True

        return self

    # .................................................................................................................

    def _on_move(self, cbxy: CBEventXY, cbflags):
        # Record mouse position for rendering 'closest point' indicator on hover
        self._mouse_xy_norm = cbxy.xy_norm
        return

    def _on_left_down(self, cbxy: CBEventXY, cbflags):
        """Create a new box or modify exist box based on left-click position"""

        # Ignore clicks outside of region
        if not cbxy.is_in_region:
            return

        # Record 'previous' box, in case we need to reset (happens if user draws invalid box)
        self._prev_xy_norms = (self._x_norms, self._y_norms)

        # Figure out if we're 'modifying' the box or drawing a new one
        xy_idx, _, is_interactive_dist = self._check_xy_interaction(cbxy.xy_norm, cbxy.hw_px)
        is_new_click = not is_interactive_dist or cbflags.shift_key

        # Either modify an existing point or reset/re-draw the box if clicking away from existing points
        self._xy_modify_idx = xy_idx
        if is_new_click:
            # We modify the 'last' xy coord on new boxes, by convention
            self._xy_modify_idx = (2, 2)
            new_x, new_y = cbxy.xy_norm
            self._x_norms = np.float32((new_x, new_x, new_x))
            self._y_norms = np.float32((new_y, new_y, new_y))

        self._is_modifying = True
        self._is_changed = True

        return

    def _on_drag(self, cbxy: CBEventXY, cbflags):
        """Modify box corner or midpoint when dragging"""

        # Bail if no points are being modified (shouldn't happen...?)
        if not self._is_modifying:
            return

        # Don't allow dragging out-of-bounds!
        new_x, new_y = np.clip(cbxy.xy_norm, 0.0, 1.0)

        # Update corner points (if they're the ones being modified) and re-compute mid-points
        x_mod_idx, y_mod_idx = self._xy_modify_idx
        if x_mod_idx != 1:
            self._x_norms[x_mod_idx] = new_x
            self._x_norms[1] = (self._x_norms[0] + self._x_norms[2]) * 0.5
        if y_mod_idx != 1:
            self._y_norms[y_mod_idx] = new_y
            self._y_norms[1] = (self._y_norms[0] + self._y_norms[2]) * 0.5

        # Assume box is changed by dragging update
        self._is_changed = True

        return

    def _on_left_up(self, cbxy: CBEventXY, cbflags):
        """Stop modifying box on left up"""

        # Reset modifier indexing
        self._is_modifying = False

        # Reset if the resulting box is too small
        h_px, w_px = cbxy.hw_px
        box_w = int(np.abs(self._x_norms[0] - self._x_norms[2]) * (h_px - 1))
        box_h = int(np.abs(self._y_norms[0] - self._y_norms[1]) * (w_px - 1))
        box_area_norm = (box_h * box_w) / (h_px * w_px)
        if box_area_norm < self._minimum_area_norm:
            self._x_norms, self._y_norms = self._prev_xy_norms
            self._is_changed = True

        return

    def _on_right_click(self, cbxy: CBEventXY, cbflags):
        self.clear()
        return

    # .................................................................................................................

    def _check_xy_interaction(
        self,
        target_xy_norm: XYNORM,
        frame_hw: HWPX | None = None,
    ) -> tuple[tuple[int, int], tuple[float, float], bool]:
        """
        Helper used to check which of the box points (corners or midpoints)
        are closest to given target xy coordinate, and what the x/y distance
        ('manhattan distance') is to the closest point. Used to determine
        which points may be interacted with for dragging/modifying the box.

        Returns:
            closest_xy_index, closest_xy_distance_px, is_interactive_distance
            -> Indexing is with respect to self._x_norms & self._y_norms
        """

        # Default to 'fake' pixel count if not given (so we can re-use the same calculations)
        if frame_hw is None:
            frame_hw = (2.0, 2.0)
        h_scale, w_scale = tuple(np.float32(size - 1.0) for size in frame_hw)
        target_x, target_y = target_xy_norm

        # Find closest x point on box
        x_dists = np.abs(self._x_norms - target_x)
        closest_x_index = np.argmin(x_dists)
        closest_x_dist_px = x_dists[closest_x_index] * w_scale

        # Find closest y point on box
        y_dists = np.abs(self._y_norms - target_y)
        closest_y_index = np.argmin(y_dists)
        closest_y_dist_px = y_dists[closest_y_index] * h_scale

        # Check if the point is within interaction distance
        closest_xy_index = (closest_x_index, closest_y_index)
        closest_xy_dist_px = (closest_x_dist_px, closest_y_dist_px)
        is_interactive = all(dist < self._interact_dist_px_threshold for dist in closest_xy_dist_px)
        if is_interactive:
            is_center_point = all(idx == 1 for idx in closest_xy_index)
            is_interactive = not is_center_point

        return closest_xy_index, closest_xy_dist_px, is_interactive

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        # Get sizing info
        frame_hw = frame.shape[0:2]
        h_scale, w_scale = tuple(float(size - 1.0) for size in frame_hw)
        all_x_px = tuple(int(x * w_scale) for x in self._x_norms)
        all_y_px = tuple(int(y * h_scale) for y in self._y_norms)
        xy1_px, xy2_px = (all_x_px[0], all_y_px[0]), (all_x_px[-1], all_y_px[-1])

        # Figure out whether we should draw interaction indicator & where
        need_draw_indicator = True
        if self._is_modifying:
            # If user if modifying the box, choose the modified point for drawing
            # -> We want to always draw the indicator for the point being dragged, even if
            #    the mouse is closer to some other point (can happen when dragging mid points)
            close_x_px = all_x_px[self._xy_modify_idx[0]]
            close_y_px = all_y_px[self._xy_modify_idx[1]]

        else:
            # If user isn't already interacting, we'll draw an indicator if the mouse is
            # close enough to a corner or mid point on the box. But we have to figure
            # out which point that would be every time we re-render, in case the mouse moved!
            (x_idx, y_idx), _, is_interactive_dist = self._check_xy_interaction(self._mouse_xy_norm, frame_hw)
            close_x_px = all_x_px[x_idx]
            close_y_px = all_y_px[y_idx]
            is_inbounds = np.min(self._mouse_xy_norm) > 0.0 and np.max(self._mouse_xy_norm) < 1.0
            need_draw_indicator = is_interactive_dist and is_inbounds
        closest_xy_px = (close_x_px, close_y_px)

        # Draw all background coloring first, so it appears entirely 'behind' the foreground
        if self._bg_color is not None:
            if need_draw_indicator:
                cv2.circle(frame, closest_xy_px, self._ind_bg_radius, self._bg_color, -1, self._ind_ltype)
            cv2.rectangle(frame, xy1_px, xy2_px, self._bg_color, self._bg_thick, self._ltype)

        # Draw box + interaction indicator circle in foreground color
        if need_draw_indicator:
            cv2.circle(frame, closest_xy_px, self._ind_fg_radius, self._fg_color, -1, self._ind_ltype)
        cv2.rectangle(frame, xy1_px, xy2_px, self._fg_color, self._fg_thick, self._ltype)

        return frame

    # .................................................................................................................
