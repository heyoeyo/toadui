#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from itertools import product

import cv2
import numpy as np

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class HeatmapOfLife:
    """
    Simple 'Game-of-Life' implementation with wrap-around and including a heatmap.

    - Place patterns using the .place_pattern(...) function
    - Update the grid state using the .step() function
    - For display, use .get_cells_image() or .get_heatmap_image() functions
    """

    def __init__(self, initial_height=32, initial_width=32):

        # Allocate storage for cells/heatmap
        initial_hw = [round(val) for val in (initial_height, initial_width)]
        self.cells = np.zeros(initial_hw, dtype=np.uint8)
        self.heatmap = np.zeros(initial_hw, dtype=np.float32)
        self.total_iters = 0
        self._weight = 0.995

        # Pre-generate the neighbourhood iterator (excluding center index)
        full_iter = product([-1, 0, 1], [-1, 0, 1])
        self._nb_iter = tuple((dx, dy) for dx, dy in full_iter if ((dx != 0) or (dy != 0)))
        self._neighbor_count = np.empty_like(self.cells)

    def get_hw(self) -> tuple[int, int]:
        """Return (height, width) of cell grid"""
        return self.cells.shape[0:2]

    def get_cells_image(self) -> ndarray:
        """
        Returns cell state as a uint8 grayscale image.
        Cells that are 'off' have values of 0
        Cells that are 'on' have values of 255
        """
        return 255 * self.cells

    def get_heatmap_image(self) -> ndarray:
        """Returns heatmap as a uint8 grayscale image"""
        return np.uint8(255 * self.heatmap)

    def set_heatmap_weight(self, new_weight: float):
        """
        Adjust heatmap weighting (note: adjustments are non-linear!)
        Weights are expected to be between 0.0 and 1.0. Larger weights
        lead to more 'memory' (i.e. prior cell state persist longer).
        A weight of 0 effectively disables the heatmap.
        """
        self._weight = 1.0 - (1.0 - new_weight) ** 4
        return self

    def step(self) -> int:
        """
        Run game-of-life update rules
        Returns: total_iterations
        """

        # Count up neighbour cells (with wrap-around on edges)
        self._neighbor_count.fill(0)
        for dx, dy in self._nb_iter:
            self._neighbor_count += np.roll(self.cells, (dy, dx), axis=(0, 1))

        # Game-of-life update rules
        survive = np.bitwise_and(self.cells, (self._neighbor_count - 2) <= 1)
        born = np.bitwise_and(np.bitwise_not(self.cells), self._neighbor_count == 3)
        self.cells = np.bitwise_or(survive, born)

        self.heatmap = np.maximum(self.heatmap * self._weight, np.float32(self.cells))
        self.total_iters += 1

        return self.total_iters

    def clear(self):
        """Clear all cell states (and heatmap) and reset iteration count to 0"""
        self.cells.fill(0)
        self.heatmap.fill(0)
        self.total_iters = 0
        return self

    def set_new_size(self, new_h: int | float, new_w: int | float):

        # Force sizing to be even, for nicer up/down scaling
        new_h, new_w = [2 * round(val * 0.5) for val in [new_h, new_w]]

        # Bail if we don't need to change sizing
        curr_h, curr_w = self.cells.shape[0:2]
        h_diff, w_diff = round(curr_h - new_h), round(curr_w - new_w)
        needs_resize = (h_diff != 0) or (w_diff != 0)
        if not needs_resize:
            return self

        # Handle up- vs. down-scaling
        new_cells = self.cells
        new_heatmap = self.heatmap
        needs_downsize = (h_diff > 0) or (w_diff > 0)
        dh, dw = [abs(diff // 2) for diff in (h_diff, w_diff)]
        if needs_downsize:
            dh, dw = [max(1, delta) for delta in (dh, dw)]
            hslice, wslice = slice(None), slice(None)
            hslice = slice(dh, -dh) if dh > 0 else slice(None)
            wslice = slice(dw, -dw) if dw > 0 else slice(None)
            new_cells = new_cells[hslice, wslice]
            new_heatmap = new_heatmap[hslice, wslice]
        else:
            tblr_inc = [dh, dh, dw, dw]
            new_cells = cv2.copyMakeBorder(new_cells, *tblr_inc, cv2.BORDER_CONSTANT)
            new_heatmap = cv2.copyMakeBorder(new_heatmap, *tblr_inc, cv2.BORDER_CONSTANT)

        # Store re-sized results
        self.cells = new_cells
        self.heatmap = new_heatmap
        self._neighbor_count = np.empty_like(new_cells)

        return self

    def rotate(self, num_steps_CCW=-1):
        """Rotate cells by 90 degrees"""
        self.cells = np.rot90(self.cells, num_steps_CCW)
        self.heatmap = np.rot90(self.heatmap, num_steps_CCW)
        self._neighbor_count = np.empty_like(self.cells)
        return self

    def place_pattern(
        self,
        pattern: ndarray,
        xy_position=(0.5, 0.5),
        pattern_anchor=(0.5, 0.5),
        use_xor=False,
        clear_existing=True,
    ):
        """
        Function used to place new patterns into the game-of-life cells
        Patterns are expected to be given as uint8 numpy arrays with 0/1 values.
        Placement is given using normalized xy coordinates (e.g. 0.0 to 1.0),
        with (0,0) being the top-left, (1,1) being bottom-right.
        """

        # Wipe out existing state, if needed
        if clear_existing:
            self.clear()

        # When using xor, we need to keep track of changes for updating the heatmap!
        prev_cells, xor_diff = None, None
        if use_xor:
            prev_cells = self.cells.copy()

        self.cells = place_pattern(self.cells, pattern, xy_position, pattern_anchor, use_xor)

        # Delete xor'd sections from the heatmap
        if use_xor:
            xor_diff = 1 - np.abs(prev_cells - self.cells)
            self.heatmap = self.heatmap * xor_diff
        self.heatmap = np.maximum(self.heatmap, np.float32(self.cells))

        return self


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def place_pattern(cells, pattern, xy_position=(0.5, 0.5), pattern_anchor=(0.5, 0.5), use_xor=False) -> ndarray:
    """Helper used to place patterns on an existing grid"""

    # If we don't get an array, assume we got a function to build an array, so call it
    if not isinstance(pattern, np.ndarray):
        pattern = pattern()

    # Get sizing and force pattern to be smaller than the grid
    ch, cw = cells.shape[0:2]
    ph, pw = pattern.shape[0:2]
    if ph > ch:
        py1 = (ph - ch) // 2
        py2 = py1 + ch
        pattern = pattern[py1:py2, :]
        ph = ch
    if pw > cw:
        px1 = (pw - cw) // 2
        px2 = px1 + cw
        pattern = pattern[:, px1:px2]
        pw = cw

    # Find bounding grid indices based on pattern placement
    x_pos, y_pos = xy_position
    x_anc, y_anc = pattern_anchor
    gx1 = round(x_pos * (cw - 1) - x_anc * (pw - 1))
    gy1 = round(y_pos * (ch - 1) - y_anc * (ph - 1))
    gx2 = gx1 + pw
    gy2 = gy1 + ph

    # Compute rolling offsets (handles cases where shape is applied on wrap-around borders)
    roll_y = max(-gy1, 0) - max(gy2 - ch, 0)
    roll_x = max(-gx1, 0) - max(gx2 - cw, 0)
    ry1, ry2 = [val + roll_y for val in [gy1, gy2]]
    rx1, rx2 = [val + roll_x for val in [gx1, gx2]]

    # Place pattern with wrap-around support
    new_cells = np.roll(cells, (roll_y, roll_x), axis=(0, 1))
    new_cells[ry1:ry2, rx1:rx2] = cv2.bitwise_xor(new_cells[ry1:ry2, rx1:rx2], pattern) if use_xor else pattern
    new_cells = np.roll(new_cells, (-roll_y, -roll_x), axis=(0, 1))

    return new_cells


def make_glider() -> ndarray:
    gd = np.zeros((3, 3), dtype=np.uint8)
    gd[0, 1] = 1
    gd[1, 2] = 1
    gd[2, :] = 1
    return gd


def make_r_pentamino() -> ndarray:
    gd = np.zeros((3, 3), dtype=np.uint8)
    gd[0, 1:] = 1
    gd[1, :2] = 1
    gd[2, 1] = 1
    return gd


def make_acorn() -> ndarray:
    gd = np.zeros((3, 7), dtype=np.uint8)
    gd[0, 1] = 1
    gd[1, 3] = 1
    gd[2, 0:2] = 1
    gd[2, 4:] = 1
    return gd


def make_lidka() -> ndarray:
    gd = np.zeros((9, 15), dtype=np.uint8)
    gd[0, 10:13] = 1
    gd[2, 11:13] = 1
    gd[2, 14] = 1
    gd[3, 12] = 1
    gd[3, 14] = 1
    gd[4, 14] = 1
    gd[6, 1] = 1
    gd[7, 0] = 1
    gd[7, 2] = 1
    gd[8, 1] = 1
    return gd


def make_rabbits() -> ndarray:
    gd = np.zeros((3, 7), dtype=np.uint8)
    gd[0, 0] = 1
    gd[0, 4:7] = 1
    gd[1, 0:3] = 1
    gd[1, 5] = 1
    gd[2, 1] = 1
    return gd


def make_bunnies() -> ndarray:
    gd = np.zeros((8, 4), dtype=np.uint8)
    gd[0, -1] = 1
    gd[1, 0] = 1
    gd[2, 1:3] = 1
    gd[3, 0] = 1
    gd[5, 1] = 1
    gd[6, 2:4] = 1
    gd[7, 1] = 1
    return np.rot90(gd)


def make_ark() -> ndarray:
    gd = np.zeros((29, 32), dtype=np.uint8)
    gd[0, 27] = 1
    gd[1, 28] = 1
    gd[2, 29] = 1
    gd[3, 28] = 1
    gd[4, 27] = 1
    gd[5, 29:] = 1
    gd[25, 0:2] = 1
    gd[26, 2] = 1
    gd[27, 2] = 1
    gd[28, 3:7] = 1
    return gd


def make_toad() -> ndarray:
    gd = np.zeros((2, 4), dtype=np.uint8)
    gd[0, 1:] = 1
    gd[1, :-1] = 1
    return gd


def make_block_engine() -> ndarray:
    gd = np.zeros((5, 5), dtype=np.uint8)
    gd[0, :3] = 1
    gd[0, 4] = 1
    gd[1, 0] = 1
    gd[2, 3:] = 1
    gd[3, 1] = 1
    gd[3, 2] = 1
    gd[3, 4] = 1
    gd[4, 0] = 1
    gd[4, 2] = 1
    gd[4, 4] = 1
    return gd


def make_pulsar() -> ndarray:
    gd_6x6 = np.zeros((6, 6), dtype=np.uint8)
    gd_6x6[0, 2:-1] = 1
    gd_6x6[-1, 2:-1] = 1
    gd_6x6[2:-1, 0] = 1
    gd_6x6[2:-1, -1] = 1

    gd = np.zeros((13, 13), dtype=np.uint8)
    gd[:6, :6] = gd_6x6
    gd[7:, :6] = np.flipud(gd_6x6)
    gd[:6, 7:] = np.fliplr(gd_6x6)
    gd[7:, 7:] = np.fliplr(np.flipud(gd_6x6))
    return gd


def make_penta_decathlon() -> ndarray:
    gd_5x4 = np.zeros((5, 4), dtype=np.uint8)
    gd_5x4[0, -1] = 1
    gd_5x4[1, -2] = 1
    gd_5x4[2, -3] = 1
    gd_5x4[4, 0] = 1

    gd = np.zeros((10, 9), dtype=np.uint8)
    gd[:5, :4] = gd_5x4
    gd[5:, :4] = np.flipud(gd_5x4)
    gd[:5, 5:] = np.fliplr(gd_5x4)
    gd[5:, 5:] = np.fliplr(np.flipud(gd_5x4))
    gd[0, 4] = 1
    gd[-1, 4] = 1
    return gd


def make_44P5H2V0() -> ndarray:
    gd_11x6 = np.zeros((11, 6), dtype=np.uint8)
    gd_11x6[0, 4] = 1
    gd_11x6[1, 3:] = 1
    gd_11x6[2, 2] = 1
    gd_11x6[2, 5] = 1
    gd_11x6[3, 1:4] = 1
    gd_11x6[4, 2] = 1
    gd_11x6[4, 4] = 1
    gd_11x6[5, 4:] = 1
    gd_11x6[6, 0] = 1
    gd_11x6[6, 5] = 1
    gd_11x6[7, 5] = 1
    gd_11x6[8, 0:2] = 1
    gd_11x6[8, 5] = 1
    gd_11x6[9, 2] = 1
    gd_11x6[9, 5] = 1
    gd_11x6[10, 4] = 1

    gd = np.zeros((11, 15), dtype=np.uint8)
    gd[:, :6] = gd_11x6
    gd[:, 9:] = np.fliplr(gd_11x6)

    return gd


def make_achims_p16() -> ndarray:
    gd_4x4 = np.zeros((4, 4), dtype=np.uint8)
    gd_4x4[0, 2:] = 1
    gd_4x4[1, 1] = 1
    gd_4x4[1, 3] = 1
    gd_4x4[2, 0:2] = 1
    gd_4x4[2, 3] = 1
    gd_4x4[3, 2] = 1

    gd = np.zeros((13, 13), dtype=np.uint8)
    gd[0:4, 2:6] = gd_4x4
    gd[7:11, 0:4] = np.rot90(gd_4x4, 1)
    gd[9:13, 7:11] = np.rot90(gd_4x4, 2)
    gd[2:6, 9:13] = np.rot90(gd_4x4, 3)

    return gd


def make_queen_bee() -> ndarray:
    gd = np.zeros((5, 7), dtype=np.uint8)
    gd[0, 3] = 1
    gd[1, (2, 4)] = 1
    gd[2, (1, 5)] = 1
    gd[3, 2:5] = 1
    gd[4, 0:2] = 1
    gd[4, 5:7] = 1
    return gd


def make_gosper_gun() -> ndarray:
    gd_a = np.zeros((6, 7), dtype=np.uint8)
    gd_a[0, 2:5] = 1
    gd_a[1, (1, 5)] = 1
    gd_a[2, (0, 6)] = 1
    gd_a[3, (1, 5)] = 1
    gd_a[4, 2:5] = 1
    gd_a[5, 2:5] = 1

    gd_b = np.zeros((5, 7), dtype=np.uint8)
    gd_b[0, 2:5] = 1
    gd_b[1, (1, 2, 4, 5)] = 1
    gd_b[2, (1, 2, 4, 5)] = 1
    gd_b[3, 1:6] = 1
    gd_b[4, (0, 1, 5, 6)] = 1

    gd_sq = np.ones((2, 2), dtype=np.uint8)
    gd = np.zeros((36, 9), dtype=np.uint8)
    gd[0:2, 3:5] = gd_sq.copy()
    gd[34:36, 5:7] = gd_sq.copy()
    gd[11:17, 0:7] = gd_a
    gd[21:26, 2:9] = gd_b

    return gd
