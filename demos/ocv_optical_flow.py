#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse
from time import perf_counter

import cv2
import numpy as np

from toadui.window import DisplayWindow, KEY
from toadui.video import LoopingVideoReader, VideoPlaybackSlider, read_webcam_string, ask_for_video_path
from toadui.images import DynamicImage, FixedARImage
from toadui.sliders import Slider, MultiSlider
from toadui.buttons import ToggleButton
from toadui.text import PrefixedTextBlock
from toadui.layout import VStack, HStack
from toadui.static import HSeparator
from toadui.colormaps import apply_colormap, make_colormap_from_keypoints
from toadui.overlays import HoverLabelOverlay
from toadui.helpers.colors import convert_color
from toadui.helpers.sizing import get_image_hw_for_max_side_length, resize_hw
from toadui.helpers.data_management import MaxLengthKVStorage

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set arg defaults
default_input_path = None
default_display_size = 900
default_max_skip = 10

# Define script arguments
parser = argparse.ArgumentParser(description="Demo of OpenCV optical flow (Farneback) implementation")
parser.add_argument("-i", "--input_path", default=default_input_path, type=str, help="Path to video")
parser.add_argument("-d", "--display_size", default=default_display_size, type=int, help="Initial window size")
parser.add_argument("-n", "--max_num_skip", default=default_max_skip, type=int, help="Max frame skip amount")
parser.add_argument("-cam", "--use_webcam", action="store_true", help="Use webcam as video source")
parser.add_argument("-gray", "--use_gray_dxdy", action="store_true", help="Use grayscale colormaps for dx & dy")

# For convenience
args = parser.parse_args()
input_path = args.input_path if not args.use_webcam else "cam"
display_size = args.display_size
use_grayscale_dxdy = args.use_gray_dxdy
max_frame_skip = max(1, args.max_num_skip)


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def optflow_to_false_color(optical_flow_result: ndarray, magnitude_scale_factor: float) -> ndarray:
    """
    Helper used to create false-color optical flow image.
    Flow magnitude is encoded with brightness, while flow
    direction is encoded with color:
        right movement -> red
        upward movement -> purple
        left movement -> teal
        downward movement -> green

    Returns:
        optical_flow_image_bgr
    """

    # Compute polar magnitude/angle from x/y components
    mag_flow, ang_flow = cv2.cartToPolar(xy_oflow[:, :, 0], xy_oflow[:, :, 1])
    two_pi = 2.0 * np.pi

    out_shape = (optical_flow_result.shape[0], optical_flow_result.shape[1], 3)
    hsv_img = np.empty(out_shape, dtype=np.uint8)
    hsv_img[:, :, 0] = np.round(255 * (ang_flow / two_pi)).astype(np.uint8)
    hsv_img[:, :, 1] = 255
    hsv_img[:, :, 2] = np.round(np.clip(mag_flow / magnitude_scale_factor, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR_FULL)


def optflow_to_dxdy_false_color(
    optical_flow_result: ndarray,
    magnitude_scale_factor: float,
    left_right_colormap: ndarray | int | None,
    up_down_colormap: ndarray | int | None,
) -> tuple[ndarray, ndarray]:
    """
    Helper used to create false-color images for delta-x/delta-y
    components of the optical flow prediction.
    Returns:
        delta_x_image_bgr, delta_y_image_bgr
    """
    # Map dx/dy components from a (-X, +X) to a (0, 1) range
    xy_color_scale = magnitude_scale_factor * 0.5
    dxy_img = np.uint8(255 * np.clip((optical_flow_result + xy_color_scale) / (2 * xy_color_scale), 0, 1))

    dx_flow = apply_colormap(dxy_img[:, :, 0], left_right_colormap)
    dy_flow = apply_colormap(dxy_img[:, :, 1], up_down_colormap)
    return dx_flow, dy_flow


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup UI

# Handle webcam inputs
input_path = ask_for_video_path(input_path, allow_webcam_inputs=True)
is_webcam_source, input_path = read_webcam_string(input_path)
vreader = LoopingVideoReader(input_path)
input_max_side_length = max(vreader.get_sample_frame().shape)
sample_frame = vreader.get_sample_frame()

# Set up playback & loop bounds controls
playback_slider = VideoPlaybackSlider(vreader)
loop_bounds_slider = MultiSlider(
    "",
    (0, 1),
    step=0.01,
    fill_color=(120, 150, 120),
    height=20,
    enable_value_display=False,
    color=(0, 0, 0),
)
playback_ctrl = VStack(
    HStack(HSeparator(50, (0, 0, 0), label="A-B:", is_flexible_h=False), loop_bounds_slider),
    playback_slider,
)

# Define main display elements
# -> using FixedAR helps avoid UI resizing when image is resized!
input_img_elem = DynamicImage(sample_frame)
flow_img_elem = FixedARImage(sample_frame)
dx_img_elem = FixedARImage(sample_frame, min_side_length=32)
dy_img_elem = FixedARImage(sample_frame, min_side_length=32)
time_text_block = PrefixedTextBlock("", 0, " ms")

# Create title indicator overlays
hov_flow_olay = HoverLabelOverlay(flow_img_elem, "Optical Flow")
hov_dx_olay = HoverLabelOverlay(dx_img_elem, "Delta X")
hov_dy_olay = HoverLabelOverlay(dy_img_elem, "Delta Y")

# Set up optical flow controls
image_size_slider = Slider("Image size", 600, 200, input_max_side_length, step=20)
skip_slider = Slider("Frame skip", 0, 0, max_frame_skip, step=1, marker_step=max_frame_skip // 5)
color_scale_slider = Slider("Color scaling", 10, 0.1, 50, step=0.5)
pyr_scale_slider = Slider("Pyramid Scale", 0.5, 0, 0.95, step=0.05, marker_step=0.25)
levels_slider = Slider("Pyramid Levels", 3, 0, 5, step=1, marker_step=1)
winsize_slider = Slider("Window Size", 15, 1, 50, step=1, marker_step=5)
iterations_slider = Slider("Iterations", 3, 1, 10, step=1, marker_step=1)
poly_n_slider = Slider("Poly N", 5, 0, 20, step=1, marker_step=5)
poly_sigma_slider = Slider("Poly Sigma", 1.2, 0, 8, step=0.05, marker_step=1)
reuse_input_btn = ToggleButton("Reuse flow", color_on=(145, 125, 60))
gaussian_btn = ToggleButton("Gauss Filter", color_on=(105, 40, 125))

# Stack elements together to form layout for display
is_wide_input = (vreader.shape[1] / vreader.shape[0]) > 1.25
main_display_block = HStack(input_img_elem, hov_flow_olay, VStack(hov_dx_olay, hov_dy_olay))
if is_wide_input:
    main_display_block = VStack(HStack(input_img_elem, hov_flow_olay), HStack(hov_dx_olay, hov_dy_olay))
show_playback_bar = not is_webcam_source

# Build complete layout
ui_layout = VStack(
    HStack(image_size_slider, time_text_block, flex=(1, 0)),
    main_display_block,
    playback_ctrl if show_playback_bar else None,
    color_scale_slider,
    HStack(pyr_scale_slider, levels_slider),
    winsize_slider,
    iterations_slider,
    HStack(poly_n_slider, poly_sigma_slider),
    skip_slider,
    HStack(reuse_input_btn, gaussian_btn, flex=(0.25, 0.25)),
)


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Display loop ***

# Set up dx/dy flow color schemes
lrud_angles_uint8 = [round((angle / 360) * 255) for angle in (180, 0, 90, 270)]
l_bgr, r_bgr, u_bgr, d_bgr = [convert_color([col, 255, 255], cv2.COLOR_HSV2BGR_FULL) for col in lrud_angles_uint8]
left_right_cmap = make_colormap_from_keypoints([l_bgr, (0, 0, 0), r_bgr])
up_down_cmap = make_colormap_from_keypoints([d_bgr, (0, 0, 0), u_bgr])
if use_grayscale_dxdy:
    left_right_cmap = None
    up_down_cmap = None

# Set up display window and attach UI for mouse interactions
window = DisplayWindow(display_fps=min(vreader.get_framerate(), 60))
window.enable_size_control(display_size)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callbacks(
    {
        "Play/Pause the video": {" ": vreader.toggle_pause},
        "Step video backwards/forwards": {",": vreader.prev_frame, ".": vreader.next_frame},
        "Adjust image sizing": {KEY.D_ARROW: image_size_slider.decrement, KEY.U_ARROW: image_size_slider.increment},
        "Adjust color scaling": {"[": color_scale_slider.decrement, "]": color_scale_slider.increment},
        "Toggle flow re-use": {"r": reuse_input_btn.toggle},
        "Toggle Gaussian filter": {"g": gaussian_btn.toggle},
    }
)
window.report_keypress_descriptions()


prev_frames_dict = MaxLengthKVStorage(2 + max_frame_skip)
prev_oflow_dict = MaxLengthKVStorage(2 + max_frame_skip)
with window.auto_close(vreader.release):

    # Set up initial loop boundaries, with special handling for webcams
    loop_a_idx, loop_b_idx = 0, vreader.get_frame_count()
    loop_b_idx = loop_b_idx if not is_webcam_source else 2**32

    for is_paused, frame_idx, frame in vreader:
        playback_slider.update_state(is_paused, frame_idx)
        is_playing = not is_paused

        # Read controls
        is_side_length_changed, max_side_length = image_size_slider.read()
        is_loop_bounds_changed, (loop_start_norm, loop_end_norm) = loop_bounds_slider.read()
        _, is_playback_adjusting = playback_slider.read()
        _, color_scale = color_scale_slider.read()
        _, pyr_scale = pyr_scale_slider.read()
        _, levels = levels_slider.read()
        _, winsize = winsize_slider.read()
        _, iters = iterations_slider.read()
        _, poly_n = poly_n_slider.read()
        _, poly_sigma = poly_sigma_slider.read()
        _, skip_n = skip_slider.read()
        _, reuse_input = reuse_input_btn.read()
        _, use_gauss = gaussian_btn.read()

        # Handle user loop bounding
        if is_loop_bounds_changed and not is_webcam_source:
            loop_a_idx = round(vreader.get_frame_count() * loop_start_norm)
            loop_b_idx = round(vreader.get_frame_count() * loop_end_norm)
        if frame_idx > loop_b_idx or frame_idx < loop_a_idx:
            vreader.set_playback_position(loop_a_idx)

        # Previous optical flow re-use is invalidated when changing frame size
        if is_side_length_changed:
            prev_oflow_dict.clear()

        # Create scaled frame for computing optflow
        flow_hw = get_image_hw_for_max_side_length(frame.shape, max_side_length=max_side_length)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frames_dict.store(frame_idx, gray_frame)

        # Decide if we can actually compute the optical flow
        prev_sample_idx = frame_idx - skip_n - 1
        prev_frame = prev_frames_dict.get(prev_sample_idx)
        is_valid_prev_frame = prev_frame is not None
        if is_valid_prev_frame:

            # Grab previous flow results (for re-use input)
            prev_oflow = prev_oflow_dict.get(prev_sample_idx)
            is_valid_prev_oflow = prev_oflow is not None
            use_prev_oflow = reuse_input and is_valid_prev_oflow

            # Set up optflow flags
            flags = int(use_gauss) * cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            flags += int(use_prev_oflow) * cv2.OPTFLOW_USE_INITIAL_FLOW

            # Resize previous frame to match current frame or else opt flow fails
            flow_h, flow_w = flow_hw
            prev_h, prev_w = prev_frame.shape[0:2]
            if prev_h != flow_h or prev_w != flow_w:
                prev_frame = resize_hw(prev_frame, flow_hw)

            # Run optical flow with timing
            t1 = perf_counter()
            curr_frame = resize_hw(gray_frame, flow_hw)
            xy_oflow = cv2.calcOpticalFlowFarneback(
                prev_frame,
                curr_frame,
                prev_oflow.copy() if use_prev_oflow else None,
                pyr_scale,
                levels,
                winsize,
                iters,
                poly_n,
                poly_sigma,
                flags,
            )
            t2 = perf_counter()
            time_text_block.set_text(str(round(1000 * (t2 - t1))))
            prev_oflow_dict.store(frame_idx, xy_oflow)

            # Generate optical flow false-color images
            bgr_flow = optflow_to_false_color(xy_oflow, color_scale)
            dx_flow, dy_flow = optflow_to_dxdy_false_color(xy_oflow, color_scale, left_right_cmap, up_down_cmap)

        else:
            # Create blank frames to indicate missing flow info when input is invalid
            bgr_flow = np.zeros((*flow_hw, 3), dtype=np.uint8)
            dx_flow = bgr_flow.copy()
            dy_flow = bgr_flow.copy()

        # Update displayed images & render
        input_img_elem.set_image(frame)
        flow_img_elem.set_image(bgr_flow)
        dx_img_elem.set_image(dx_flow)
        dy_img_elem.set_image(dy_flow)
        display_image = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        pass
    pass
