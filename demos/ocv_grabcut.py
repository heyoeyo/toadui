#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse
from time import perf_counter

import cv2
import numpy as np

from toadui.window import DisplayWindow
from toadui.video import VideoPlaybackSlider, read_webcam_string, ask_for_video_path, load_looping_video_or_image
from toadui.sliders import Slider
from toadui.images import FixedARImage
from toadui.text import PrefixedTextBlock
from toadui.colormaps import apply_colormap
from toadui.buttons import ToggleButton, ImmediateButton
from toadui.layout import VStack, HStack, OverlayStack
from toadui.overlays import EditBoxOverlay, MousePaintOverlay
from toadui.patterns.checker import CheckerPattern
from toadui.helpers.drawing import draw_circle_norm, draw_normalized_polygon
from toadui.helpers.sizing import resize_hw, get_image_hw_for_max_side_length
from toadui.helpers.images import histogram_equalization, convert_xy_norm_to_px
from toadui.helpers.data_management import ValueChangeTracker, UndoRedoList

# For type hints
from numpy import ndarray
from toadui.helpers.types import XY1XY2NORM


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set arg defaults
default_input_path = None
default_display_size = 800
default_max_side_length = 500

# Define script arguments
parser = argparse.ArgumentParser(description="Demo of OpenCV grab-cut implementation")
parser.add_argument("-i", "--input_path", default=default_input_path, type=str, help="Path to video or image")
parser.add_argument("-d", "--display_size", default=default_display_size, type=int, help="Initial window size")
parser.add_argument("-cam", "--use_webcam", action="store_true", help="Use webcam as video source")
parser.add_argument(
    "-x",
    "--max_size",
    default=default_max_side_length,
    type=int,
    help="Max side length allowed (prevents execessive CPU usage)",
)

# For convenience
args = parser.parse_args()
input_path = args.input_path if not args.use_webcam else "cam"
display_size = args.display_size
max_side_length = args.max_size


# ---------------------------------------------------------------------------------------------------------------------
# %% Helpers


class GrabCutData:
    """Helper used to run grabcut with rectangle/mask inputs"""

    def __init__(self):
        self._iter = 1
        self._fg_model = np.zeros((1, 65), dtype=np.float64)
        self._bg_model = np.zeros((1, 65), dtype=np.float64)

    def set_iterations(self, num_iterations: int) -> None:
        self._iter = max(1, num_iterations)
        return

    def make_blank_mask(self, image_uint8: ndarray) -> ndarray:
        return np.full(image_uint8.shape[0:2], cv2.GC_PR_BGD, dtype=np.uint8)

    def box_cut(self, image_uint8: ndarray, box_xy1xy2_norm: XY1XY2NORM | None) -> tuple[bool, ndarray]:
        """Perform grabcut using a bounding box. Also acts as a mask initializer"""

        # Figure out bounding box xy/wh in pixel units
        is_valid_box = box_xy1xy2_norm is not None
        if is_valid_box:
            (x1_px, y1_px), (x2_px, y2_px) = convert_xy_norm_to_px(image_uint8.shape, *box_xy1xy2_norm)
            w_px, h_px = max(1, x2_px - x1_px + 1), max(1, y2_px - y1_px + 1)
            img_h, img_w = image_uint8.shape[0:2]
            is_valid_box = not ((h_px == img_h) and (w_px == img_w))

        # Run grab cut using box or generate an empty mask
        if is_valid_box:
            init_mask = None
            gc_bg, gc_fg = None, None
            gc_rect = (x1_px, y1_px, w_px, h_px)
            gc_iter = self._iter
            gc_mode = cv2.GC_INIT_WITH_RECT
            mask, out_bg, out_fg = cv2.grabCut(image_uint8, init_mask, gc_rect, gc_bg, gc_fg, gc_iter, gc_mode)
        else:
            mask = self.make_blank_mask(image_uint8)
            out_bg = np.zeros((1, 65), dtype=np.float64)
            out_fg = np.zeros((1, 65), dtype=np.float64)

        # Store models for additional mask-cut updates
        self._bg_model = out_bg
        self._fg_model = out_fg

        is_blank_mask = not is_valid_box
        return is_blank_mask, mask

    def mask_cut(self, image_uint8: ndarray, mask: ndarray, update_fgbg_models: bool = False) -> ndarray:
        """Perform grabcut using a foreground/background mask"""

        # Run grabcut using a mask as input (and previous fg/bg models from box-cut)
        gc_bg = self._bg_model.copy()
        gc_fg = self._fg_model.copy()
        gc_rect = None
        gc_iter = self._iter
        gc_mode = cv2.GC_EVAL if update_fgbg_models else cv2.GC_EVAL_FREEZE_MODEL
        out_mask, out_bg, out_fg = cv2.grabCut(image_uint8, mask, gc_rect, gc_bg, gc_fg, gc_iter, gc_mode)
        if update_fgbg_models:
            self._fg_model = out_bg
            self._bg_model = out_bg

        return out_mask


class TrailKeeper(UndoRedoList):
    """Storage for trail data. Manages drawing & undo/redo/clear capability"""

    def add_trail(self, trail_xy: ndarray, trail_lmr, brush_size_norm: float) -> bool:
        is_valid_trail = len(trail_xy) > 0
        if is_valid_trail:
            self.append((np.float32(trail_xy), trail_lmr, brush_size_norm))
        return is_valid_trail

    def draw_trails(self, image_uint8: ndarray, line_type=cv2.LINE_8) -> ndarray:
        """Draws paint trails onto an image, using grabcut FG/BG 'color' coding"""

        bg_color, is_closed = None, False
        for trail_xy, lmr_idx, bsize_norm in self:
            brush_rad_f32 = bsize_norm * min(image_uint8.shape[0:2]) * 0.5
            color = cv2.GC_FGD if lmr_idx == 0 else cv2.GC_BGD

            # Draw a circle if trail is only 1 xy coord. otherwise draw as polyline
            num_xy = len(trail_xy)
            if num_xy == 1:
                brush_rad_px = round(brush_rad_f32)
                draw_circle_norm(image_uint8, trail_xy[0], brush_rad_px, color, -1, line_type)
            else:
                brush_thick_px = max(1, round(2 * brush_rad_f32))
                draw_normalized_polygon(image_uint8, trail_xy, color, brush_thick_px, bg_color, line_type, is_closed)

        return image_uint8


# Initialize data management objects used in video loop
trail_keeper = TrailKeeper()
frame_idx_tracker = ValueChangeTracker(-1)
grabcutter = GrabCutData()

# Define special colormap used to interpret grab-cut masking results
# -> Colors for user-defined BG/FG & GC-predicted BG/FG
fg_hint_color, bg_hint_color = (0, 255, 0), (0, 0, 255)
gc_cmap = np.zeros((1, 256, 3), dtype=np.uint8)
gc_cmap[:, cv2.GC_BGD, :] = bg_hint_color
gc_cmap[:, cv2.GC_FGD, :] = fg_hint_color
gc_cmap[:, cv2.GC_PR_BGD, :] = (0, 0, 0)
gc_cmap[:, cv2.GC_PR_FGD, :] = (255, 255, 255)

# Define alternate colormap, used for binary mask display (i.e. no colors)
bin_cmap = gc_cmap.copy()
bin_cmap[:, cv2.GC_BGD, :] = (0, 0, 0)
bin_cmap[:, cv2.GC_FGD, :] = (255, 255, 255)

# Set up checkerboard rendering used for previews
checker = CheckerPattern()


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup UI

# Handle webcam inputs
input_path = ask_for_video_path(input_path, path_type="video or image", allow_webcam_inputs=True)
is_webcam_source, input_path = read_webcam_string(input_path)
is_image_source, vreader = load_looping_video_or_image(input_path)
is_video_source = is_webcam_source or (not is_image_source)
sample_frame = vreader.get_sample_frame()
img_h, img_w = sample_frame.shape[0:2]

# Define display elements
img_elem = FixedARImage(sample_frame)
mask_elem = FixedARImage(sample_frame, resize_interpolation=cv2.INTER_NEAREST)
playback_slider = VideoPlaybackSlider(vreader)
box_olay = EditBoxOverlay(None, allow_right_click_clear=False)
paint_img_olay = MousePaintOverlay(None, allow_right_click=True)
paint_mask_olay = MousePaintOverlay(mask_elem, allow_right_click=True)
olay_stack = OverlayStack(img_elem, box_olay, paint_img_olay)

# Create top level elements
undo_btn, redo_btn, clear_btn = ImmediateButton.many("Undo", "Redo", "Clear")
use_paint_btn = ToggleButton("Paint", color_on=(60, 195, 30), text_color_on=255)
show_binmask_btn = ToggleButton("Binary", color_on=(195, 100, 35), text_color_on=255)
hidden_force_btn = ImmediateButton("Force Re-cut")

# Create lower control elements
enable_box_btn = ToggleButton("Include Box", default_state=True, color_on=(100, 115, 120), text_color_on=255)
use_heq_btn = ToggleButton("Boost", color_on=(110, 100, 125))
preview_btn = ToggleButton("Preview", color_on=(100, 120, 100))
brush_slider_slider = Slider("Brush Size", 2.5, 0, 50, step=0.5)
max_img_size_slider = Slider("Image Size", 220, 50, min(img_h, max_side_length), step=5)
iter_slider = Slider("Iterations", 1, 1, 10, step=1)
time_txt_block = PrefixedTextBlock("", suffix=" ms")

# Stack elements together to form layout for display
show_playback_bar = not (is_image_source or is_webcam_source)
ui_layout = VStack(
    HStack(undo_btn, redo_btn, clear_btn),
    HStack(use_paint_btn, show_binmask_btn),
    HStack(olay_stack, paint_mask_olay),
    playback_slider if is_video_source else None,
    HStack(enable_box_btn, use_heq_btn, preview_btn),
    brush_slider_slider,
    HStack(max_img_size_slider, iter_slider, time_txt_block, flex=(1, 1, 0)),
)

# Set paint colors
paint_img_olay.style.color_left_paint = fg_hint_color
paint_img_olay.style.color_right_paint = bg_hint_color
paint_mask_olay.style.color_left_paint = fg_hint_color
paint_mask_olay.style.color_right_paint = bg_hint_color


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Display loop ***

# Set up display window and attach UI for mouse interactions
window = DisplayWindow(display_fps=min(vreader.get_framerate(), 60))
window.enable_size_control(display_size)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callbacks(
    {
        "Play/Pause the video": {" ": vreader.toggle_pause} if is_video_source else None,
        "Step video backwards/forwards": (
            {",": vreader.prev_frame, ".": vreader.next_frame} if is_video_source else None
        ),
        "Adjust brush size": {"[": brush_slider_slider.decrement, "]": brush_slider_slider.increment},
        "Toggle box/paint control": {"t": use_paint_btn.toggle},
        "Toggle binary view": {"y": show_binmask_btn.toggle},
        "Toggle use of box cut": {"i": enable_box_btn.toggle},
        "Toggle (internal) contrast boost": {"b": use_heq_btn.toggle},
        "Toggle preview": {"p": preview_btn.toggle},
        "Undo paint": {"z": undo_btn.click, "u": undo_btn.click},
        "Redo paint": {"r": redo_btn.click},
        "Clear painting": {"c": clear_btn.click},
        "Re-roll box cut": {"f": hidden_force_btn.click},
    }
)

# Provide details about demo usage
window.report_keypress_descriptions()
print(
    "- In paint mode, use left/right click to paint FG/BG regions",
    "- Box-cut can be stochastic. Use re-roll to get mask variations",
    "- Using contrast boost (histogram equalization) can also give mask variations",
    sep="\n",
)

is_blank_mask = True
base_mask, out_mask = None, None
with window.auto_close(vreader.release):

    # Prevent non-webcam sources from playing on start up (grabcut can be slow to run frame-by-frame!)
    vreader.toggle_pause(not is_webcam_source)

    for is_paused, frame_idx, frame in vreader:
        playback_slider.update_state(is_paused, frame_idx)
        is_fidx_changed = frame_idx_tracker.is_changed(frame_idx)

        # Read controls
        is_enable_box_changed, enable_box_cut = enable_box_btn.read()
        is_imgsize_changed, max_img_size = max_img_size_slider.read()
        is_itercount_changed, iter_count = iter_slider.read()
        is_box_moved, is_box_valid, box_xy1xy2_norm = box_olay.read()
        is_brush_size_changed, brush_size_pct = brush_slider_slider.read()
        is_heq_changed, use_heq = use_heq_btn.read()
        is_show_binmask_changed, show_binmask = show_binmask_btn.read()
        _, show_preview = preview_btn.read()
        need_forced_recut = hidden_force_btn.read()

        # Read trail painting overlays
        is_img_trail_finished, img_trail_xy, img_trail_lmr = paint_img_olay.read_trail()
        is_mask_trail_finished, mask_trail_xy, mask_trail_lmr = paint_mask_olay.read_trail()

        # Enable/disable overlays based on box vs. paint control
        is_tool_changed, use_paint = use_paint_btn.read()
        if is_tool_changed:
            box_olay.enable(not use_paint)
            paint_img_olay.enable(use_paint).enable_render(use_paint)

        # Auto-switch to painting when disabling box cut (better user experience)
        if is_enable_box_changed:
            box_olay.enable_render(enable_box_cut)
            if not enable_box_cut:
                use_paint_btn.toggle(True)

        # Update image used for all processing
        scale_hw = get_image_hw_for_max_side_length(frame.shape, max_img_size)
        scaled_frame = resize_hw(frame, scale_hw, interpolation=cv2.INTER_NEAREST)
        if use_heq:
            yuv_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2YUV)
            yuv_frame = histogram_equalization(yuv_frame, channel_index=0)
            scaled_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)

        # When brush size is changed, update painter overlays, so brush location indicator is sized correctly
        brush_size_norm = brush_size_pct / 100.0
        if is_brush_size_changed:
            paint_img_olay.set_brush_size(brush_size_norm)
            paint_mask_olay.set_brush_size(brush_size_norm)

        if is_itercount_changed:
            grabcutter.set_iterations(iter_count)

        # Decide if we need to re-do grabcut using box input
        is_img_changed = is_imgsize_changed or is_heq_changed or is_fidx_changed
        is_box_changed = (is_box_valid and is_box_moved) or is_enable_box_changed
        is_gcut_changed = is_itercount_changed or (enable_box_cut and need_forced_recut)
        need_box_recut = is_img_changed or is_box_changed or is_gcut_changed

        # Update trail state from undo/redo/clear buttons
        need_redraw_trails = False
        if undo_btn.read():
            need_redraw_trails |= trail_keeper.undo()
        if redo_btn.read():
            need_redraw_trails |= trail_keeper.redo()
        if clear_btn.read():
            need_redraw_trails |= trail_keeper.clear()

        # Record trails painted by user
        is_trail_painted = is_img_trail_finished or is_mask_trail_finished
        if is_trail_painted:
            is_valid_img_trail = trail_keeper.add_trail(img_trail_xy, img_trail_lmr, brush_size_norm)
            is_valid_mask_trail = trail_keeper.add_trail(mask_trail_xy, mask_trail_lmr, brush_size_norm)
            need_redraw_trails |= is_valid_img_trail or is_valid_mask_trail

        # Re-cut the base mask from box selection, if needed
        t1 = perf_counter()
        if need_box_recut:
            is_blank_mask, base_mask = grabcutter.box_cut(scaled_frame, box_xy1xy2_norm if enable_box_cut else None)
            out_mask = base_mask.copy()
            need_redraw_trails = True

        # Update trail drawing
        if need_redraw_trails:
            if len(trail_keeper) == 0:
                out_mask = base_mask.copy()
            else:
                trail_mask = trail_keeper.draw_trails(base_mask.copy())
                out_mask = grabcutter.mask_cut(scaled_frame, trail_mask, update_fgbg_models=is_blank_mask)

        # Report time needed to perform grabcut
        is_mask_updated = need_box_recut or need_redraw_trails
        if is_mask_updated:
            t2 = perf_counter()
            time_taken_ms = round(1000 * (t2 - t1))
            time_txt_block.set_text(time_taken_ms)

        # Update mask display if mask is updated or coloring is toggled
        if is_mask_updated or is_show_binmask_changed:
            color_mask = apply_colormap(out_mask, bin_cmap if show_binmask else gc_cmap)
            mask_elem.set_image(color_mask)

        # In preview mode, show checkerboard where grabcut predicts 'background'
        disp_img = frame
        if show_preview:
            bin_mask = np.bitwise_or(out_mask == cv2.GC_PR_FGD, out_mask == cv2.GC_FGD)
            disp_img = checker.render_from_mask(frame, bin_mask)

        # Update displayed image & render
        img_elem.set_image(disp_img)
        display_image = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        pass
    pass
