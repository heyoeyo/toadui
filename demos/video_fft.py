#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse

import cv2
import numpy as np

from toadui.window import DisplayWindow, KEY
from toadui.video import VideoPlaybackSlider, read_webcam_string, ask_for_video_path, load_looping_video_or_image
from toadui.images import FixedARImage
from toadui.layout import VStack, HStack, Swapper
from toadui.buttons import ToggleButton, ImmediateButton, RadioBar
from toadui.sliders import Slider, ColorSlider
from toadui.overlays import MousePaintOverlay
from toadui.colormaps import make_hsv_rainbow_colormap
from toadui.helpers.drawing import draw_normalized_polygon
from toadui.helpers.sizing import get_image_hw_for_max_side_length, resize_hw
from toadui.helpers.data_management import UndoRedoList

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set arg defaults
default_input_path = None
default_display_size = 800

# Define script arguments
parser = argparse.ArgumentParser(description="Demo showing FFT of image or video")
parser.add_argument("-i", "--input_path", default=default_input_path, type=str, help="Path to video")
parser.add_argument("-d", "--display_size", default=default_display_size, type=int, help="Initial window size")
parser.add_argument("-cam", "--use_webcam", action="store_true", help="Use webcam as video source")
parser.add_argument("-o", "--no_optimal_fft", action="store_false", help="Disable FFT optimal sizing")
parser.add_argument("-s", "--no_shift_fft", action="store_false", help="Disable FFT center-shifting")

# For convenience
args, unknown_args = parser.parse_known_args()
input_path = args.input_path if not args.use_webcam else "cam"
display_size = args.display_size
allow_optimal_sizing = args.no_optimal_fft
default_center_fft = args.no_shift_fft


# ---------------------------------------------------------------------------------------------------------------------
# %% Helpers


def draw_trails(image: ndarray, trail_xy: list, brush_size_norm: float, color: float, line_type=cv2.LINE_8) -> ndarray:
    """Helper used to draw trails onto an image, with normalized brush sizing"""
    thick_px = max(1, round(brush_size_norm * min(image.shape[0:2])))
    return draw_normalized_polygon(image, trail_xy, round(color), thick_px, line_type=line_type, is_closed=False)


def get_optimal_sizing(frame: ndarray, allow_optimal_sizing: bool, padding_value: int = 0) -> tuple[ndarray, int, int]:
    """
    Helper used to pad a frame for 'optimal sizing' of FFT computation.
    This can help speed up the FFT computation, but results in a larger
    'image', which will need to be cropped to get back the original image.
    - Padding is added to the bottom/right of the frame
    Returns: padded_frame, padding_x, padding_y
    """
    pad_x, pad_y = 0, 0
    if allow_optimal_sizing:
        img_h, img_w = frame.shape[0:2]
        optimal_h = cv2.getOptimalDFTSize(img_h)
        optimal_w = cv2.getOptimalDFTSize(img_w)
        pad_x, pad_y = [max(0, value) for value in (optimal_w - img_w, optimal_h - img_h)]
        is_non_optimal_sizing = (pad_x > 0) or (pad_y > 0)
        if is_non_optimal_sizing:
            frame = cv2.copyMakeBorder(frame, 0, pad_y, 0, pad_x, borderType=cv2.BORDER_CONSTANT, value=padding_value)

    return frame, pad_x, pad_y


def get_fft_magphase(gray_frame: ndarray, use_centering: bool) -> tuple[ndarray, ndarray]:
    """Helper used to compute the FFT & return the magnitude & phase separately. Returns: magnitude, phase"""
    fft_frame = cv2.dft(gray_frame.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_frame_shift = np.fft.fftshift(fft_frame) if use_centering else fft_frame
    mag_f32, phase_f32 = cv2.cartToPolar(fft_frame_shift[:, :, 0], fft_frame_shift[:, :, 1])
    return mag_f32, phase_f32


def invert_fft(magnitude_f32: ndarray, phase_f32: ndarray, use_centering: bool, pad_x: int, pad_y: int) -> ndarray:
    """
    Helper used to invert an FFT from the magnitude & phase components.
    Can also 'undo' padding if input was padded for optimal sizing.
    Returns: grayscale_frame
    """
    fft_frame_shift = np.dstack(cv2.polarToCart(mag_f32, phase_f32))
    fft_frame = np.fft.ifftshift(fft_frame_shift) if use_centering else fft_frame_shift
    gray_frame = cv2.dft(fft_frame, flags=cv2.DFT_INVERSE)[:, :, 0]
    is_non_optimal_sizing = (pad_x > 0) or (pad_y > 0)
    if is_non_optimal_sizing:
        yslice = slice(0, None if pad_y == 0 else -pad_y)
        xslice = slice(0, None if pad_x == 0 else -pad_x)
        gray_frame = gray_frame[yslice, xslice]

    return gray_frame


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup UI

# Handle webcam inputs
input_path = ask_for_video_path(input_path, allow_webcam_inputs=True)
is_webcam_source, input_path = read_webcam_string(input_path)
is_image_source, vreader = load_looping_video_or_image(input_path, display_size_px=600)

# Define UI elements
img_elem = FixedARImage(vreader.get_sample_frame())
fft_elem = FixedARImage(vreader.get_sample_frame(), resize_interpolation=cv2.INTER_NEAREST)
playback_slider = VideoPlaybackSlider(vreader)
paint_olay = MousePaintOverlay(fft_elem)

# UI elements associated with magnitude/phase selection
radio_labels = ["Magnitude", "Phase"]
mag_phase_select = RadioBar(*radio_labels, height=40)
magcolor_slider = ColorSlider("Magnitude paint", (0, 255))
phasecolor_slider = ColorSlider("Phase paint", make_hsv_rainbow_colormap())
swap_col_slider = Swapper(magcolor_slider, phasecolor_slider, keys=radio_labels)

# Control elements
imgsize_slider = Slider("Image Size", 600, 20, 1440, step=5)
blur_slider = Slider("Blur", 0, 0, 15, step=0.1)
brush_size_slider = Slider("Brush Size", 0.1, 0, 0.5, step=0.005)
use_center_shift_btn = ToggleButton("Center", default_state=default_center_fft)
draw_effect_slider = Slider("Draw effect", 1, 0, 1, 0.05)
undo_btn = ImmediateButton("Undo")
redo_btn = ImmediateButton("Redo", (160, 120, 50))
clear_btn = ImmediateButton("Clear", (70, 40, 170))

# Stack elements together to form layout for display
show_playback_bar = not is_image_source
ui_layout = VStack(
    HStack(VStack(imgsize_slider, img_elem, blur_slider), VStack(mag_phase_select, paint_olay, swap_col_slider)),
    HStack(undo_btn, redo_btn, clear_btn),
    playback_slider if show_playback_bar else None,
    HStack(draw_effect_slider, brush_size_slider),
)


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Display loop ***

# Set up display window and attach UI for mouse interactions
window = DisplayWindow(display_fps=min(vreader.get_framerate(), 60))
window.enable_size_control(display_size)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callbacks(
    {
        "Play/Pause the video": {" ": vreader.toggle_pause},
        "Step video backwards/forwards": {",": vreader.prev_frame, ".": vreader.next_frame},
        "Adjust image size": {"[": imgsize_slider.decrement, "]": imgsize_slider.increment},
        "Adjust brush size": {";": brush_size_slider.decrement, "'": brush_size_slider.increment},
        "Toggle FFT center-shift": {"s": use_center_shift_btn.toggle},
        "Toggle Magnitude/Phase": {KEY.TAB: mag_phase_select.next},
        "Clear current painting": {"c": clear_btn.click},
        "Undo current paint": {"z": undo_btn.click, "u": undo_btn.click},
        "Redo current paint": {"r": redo_btn.click},
    }
).report_keypress_descriptions()

# Set up initial values so state is well defined on first frame
show_mag = True
mag_paint_data = UndoRedoList()
phase_paint_data = UndoRedoList()
curr_paint_data = mag_paint_data if show_mag else phase_paint_data
TWOPI = np.float32(2.0 * np.pi)
with window.auto_close(vreader.release):

    for is_paused, frame_idx, frame in vreader:
        playback_slider.update_state(is_paused, frame_idx)

        # Read controls
        _, max_img_size = imgsize_slider.read()
        _, blur_strength = blur_slider.read()
        is_brush_size_changed, brush_size_norm = brush_size_slider.read()
        _, draw_effect_norm = draw_effect_slider.read()
        _, use_centering = use_center_shift_btn.read()
        is_magphase_changed, _, mag_or_phase_label = mag_phase_select.read()
        _, magcolor, _ = magcolor_slider.read()
        _, phasecolor, _ = phasecolor_slider.read()

        # Switch color sliders
        if is_magphase_changed:
            show_mag = mag_or_phase_label == "Magnitude"
            swap_col_slider.set_swap_key(mag_or_phase_label)
            curr_paint_data = mag_paint_data if show_mag else phase_paint_data

        # Handle image resizing
        scale_hw = get_image_hw_for_max_side_length(frame.shape, max_img_size)
        scaled_frame = resize_hw(frame, scale_hw)

        # Handle (pre-) blurring
        if blur_strength > 0:
            scaled_frame = cv2.GaussianBlur(scaled_frame.copy(), (0, 0), blur_strength)

        if is_brush_size_changed:
            paint_olay.set_brush_size(brush_size_norm)

        # Handle painting
        is_paint_finished, trail_xy, trail_lmr = paint_olay.read_trail()
        if is_paint_finished:
            # If user only clicks once, repeat point so we can draw it as a polyline
            if len(trail_xy) == 1:
                trail_xy.append(trail_xy[0])
            curr_color = magcolor if show_mag else phasecolor
            curr_paint_data.append((trail_xy, brush_size_norm, curr_color))

        # Handle undo/redo/clear (handles magnitude/phase painting independently!)
        if clear_btn.read():
            curr_paint_data.clear()
        if undo_btn.read():
            curr_paint_data.undo()
        if redo_btn.read():
            curr_paint_data.redo()

        # Get FFT version of frame
        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        gray_frame, pad_x, pad_y = get_optimal_sizing(gray_frame, allow_optimal_sizing)
        mag_f32, phase_f32 = get_fft_magphase(gray_frame, use_centering)

        # Compute the magnitude max value if user has painted on magnitude data (will need it in various points)
        fft_mag_max = mag_f32.max() if show_mag or len(mag_paint_data) > 0 else 1

        # Build 'color' FFT image for display
        if show_mag:
            logmag_f32 = np.log(mag_f32)
            fft_logmin, fft_logmax = logmag_f32.min(), np.log(fft_mag_max)
            logmag_norm_f32 = (logmag_f32 - fft_logmin) / (fft_logmax - fft_logmin)
            logmag_uint8 = np.round(255 * logmag_norm_f32).astype(np.uint8)
            for trail_xy, bsize_norm, color_norm in mag_paint_data:
                logmag_uint8 = draw_trails(logmag_uint8, trail_xy, bsize_norm, color=255 * color_norm)
            fft_hsv = np.zeros((logmag_uint8.shape[0], logmag_uint8.shape[1], 3), dtype=np.uint8)
            fft_hsv[:, :, 2] = logmag_uint8
        else:
            phase_uint8 = np.round(255 * (phase_f32 / TWOPI)).astype(np.uint8)
            for trail_xy, bsize_norm, color_norm in phase_paint_data:
                phase_uint8 = draw_trails(phase_uint8, trail_xy, bsize_norm, color=255 * color_norm)
            fft_hsv = np.full((phase_uint8.shape[0], phase_uint8.shape[1], 3), 255, dtype=np.uint8)
            fft_hsv[:, :, 0] = phase_uint8
        fft_uint8 = cv2.cvtColor(fft_hsv, cv2.COLOR_HSV2BGR_FULL)

        # If user has drawn on FFT, invert it to get back the modified input image for display
        mod_scaled_frame = None
        have_paint_data = len(mag_paint_data) > 0 or len(phase_paint_data) > 0
        if have_paint_data:

            # Paint onto FFT data itself (so that inverting gives modified original image)
            fft_logmax = np.log(fft_mag_max)
            for trail_xy, bsize_norm, color_norm in mag_paint_data:
                mag_f32 = draw_trails(mag_f32, trail_xy, bsize_norm, color=np.exp(fft_logmax * color_norm))
            for trail_xy, bsize_norm, color_norm in phase_paint_data:
                phase_f32 = draw_trails(phase_f32, trail_xy, bsize_norm, color=TWOPI * color_norm)
            mod_gray_frame = invert_fft(mag_f32, phase_f32, use_centering, pad_x, pad_y)

            # Normalize modified grayscale frame for display
            mod_min, mod_max = mod_gray_frame.min(), mod_gray_frame.max()
            mod_gray_norm_f32 = (mod_gray_frame - mod_min) / (mod_max - mod_min)
            mod_gray_uint8 = np.round(255 * mod_gray_norm_f32).astype(np.uint8)

            # Replace original grayscale with FFT-modified version
            yuv_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2YUV)
            if draw_effect_norm < 1:
                w_orig, w_mod = (1 - draw_effect_norm), draw_effect_norm
                orig_gray = yuv_frame[:, :, 0]
                mod_gray_uint8 = cv2.addWeighted(orig_gray, w_orig, mod_gray_uint8, w_mod, 0.0)
            yuv_frame[:, :, 0] = mod_gray_uint8
            mod_scaled_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)

        # Update displayed image & render
        img_elem.set_image(mod_scaled_frame if have_paint_data else scaled_frame)
        fft_elem.set_image(fft_uint8)
        display_image = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        pass
    pass
