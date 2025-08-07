#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse
from time import perf_counter

import cv2
import numpy as np

from toadui.window import DisplayWindow, KEY
from toadui.video import VideoPlaybackSlider, read_webcam_string, ask_for_video_path, load_looping_video_or_image
from toadui.images import FixedARImage
from toadui.layout import VStack, HStack, Swapper
from toadui.buttons import RadioBar, ToggleButton
from toadui.sliders import Slider
from toadui.carousels import TextCarousel
from toadui.text import PrefixedTextBlock
from toadui.helpers.images import dirty_blur, kuwahara_filter
from toadui.helpers.sizing import get_image_hw_for_max_side_length, resize_hw


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set arg defaults
default_input_path = None
default_display_size = 900
default_operate_size = 600

# Define script arguments
parser = argparse.ArgumentParser(description="Demo of OpenCV low-pass filtering (e.g. blurring)")
parser.add_argument("-i", "--input_path", default=default_input_path, type=str, help="Path to video")
parser.add_argument("-d", "--display_size", default=default_display_size, type=int, help="Initial window size")
parser.add_argument("-s", "--operating_size", default=default_operate_size, type=int, help="Initial working frame size")
parser.add_argument("-cam", "--use_webcam", action="store_true", help="Use webcam as video source")

# For convenience
args = parser.parse_args()
input_path = args.input_path if not args.use_webcam else "cam"
display_size = args.display_size
init_operating_size = args.operating_size


# ---------------------------------------------------------------------------------------------------------------------
# %% Helpers


def get_xy_with_ar_adjustment(size: int | float, ar_adjust: float, round_result: bool = True) -> tuple[int | float]:
    """
    Helper used to convert 'size' & 'aspect adjustment' into (size_x, size_y) values.
    When ar_adjust is -1 -> (size, 0)
         ar_adjust is  0 -> (size, size)
         ar_adjust is +1 -> (0, size)
    """
    adj_w = float(size * np.clip(1 - ar_adjust, 0, 1))
    adj_h = float(size * np.clip(1 + ar_adjust, 0, 1))
    return tuple(round(val) for val in (adj_w, adj_h)) if round_result else (adj_w, adj_h)


# Define nice 'label' mapping to ocv morphological constants
morpho_shapes_lut = {"Ellipse": cv2.MORPH_ELLIPSE, "Rectangle": cv2.MORPH_RECT, "Cross": cv2.MORPH_CROSS}
morpho_ops_lut = {
    "Dilate": cv2.MORPH_DILATE,
    "Erode": cv2.MORPH_ERODE,
    "Open": cv2.MORPH_OPEN,
    "Close": cv2.MORPH_CLOSE,
}


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup UI

# Handle webcam inputs
input_path = ask_for_video_path(input_path, path_type="video or image", allow_webcam_inputs=True)
is_webcam_source, input_path = read_webcam_string(input_path)
is_image_source, vreader = load_looping_video_or_image(input_path)
sample_frame = vreader.get_sample_frame()
img_h, img_w = sample_frame.shape[0:2]

# Figure out image size bounds
largest_size = max(img_h, img_w)
smallest_size = max(1, min(largest_size // 2, 50))
initial_size = min(init_operating_size, largest_size)
show_imgsize_slider = largest_size > 50

# Define elements shared for all filters
img_elem = FixedARImage(vreader.get_sample_frame(), resize_interpolation=cv2.INTER_NEAREST)
playback_slider = VideoPlaybackSlider(vreader)
img_size_slider = Slider("Image size", initial_size, smallest_size, largest_size, step=5)
hidden_show_orig_btn = ToggleButton("Show Original")
time_txt = PrefixedTextBlock("", "-", " ms")

# Box blur UI
bblur_ksize_slider = Slider("Kernel Size", 3, 1, 50, step=1)
bblur_aradj_slider = Slider("Aspect Adjust", 0, -1, 1, step=0.05)
bblur_ui = HStack(bblur_ksize_slider, bblur_aradj_slider)

# Gaussian blur UI
gblur_ksize_slider = Slider("Kernel Size", 3, 0, 50, step=1)
gblur_aradj_slider = Slider("Aspect Adjust", 0, -1, 1, step=0.05)
gblur_sigma_slider = Slider("Gaussian Sigma", 0, 0, 25, step=0.25)
gblur_use_sigma_sizing_btn = ToggleButton("Use sigma sizing")
gblur_ui = HStack(gblur_ksize_slider, gblur_aradj_slider, gblur_sigma_slider)

# Bilateral filter UI
bilateral_d_slider = Slider("Diameter", 9, 1, 51, step=1)
bilateral_sig_color_slider = Slider("Sigma Color", 75, 0, 500, step=1)
bilateral_sig_space_slider = Slider("Sigma Space", 75, 0, 500, step=1)
bilater_ui = HStack(bilateral_d_slider, bilateral_sig_color_slider, bilateral_sig_space_slider)

# Median blur UI
mblur_ksize_slider = Slider("Kernel Size", 5, 0, 21, step=1)
mblur_iter_slider = Slider("Iterations", 1, 1, 5, step=1)
medianblur_ui = HStack(mblur_ksize_slider, mblur_iter_slider)

# Morphological filter UI
morpho_ksize_slider = Slider("Kernel Size", 5, 1, 50, step=1)
morpho_aradj_slider = Slider("Aspect Adjust", 0, -1, 1, step=0.05)
morpho_iter_slider = Slider("Iterations", 1, 1, 50, step=1)
morpho_operation_menu = TextCarousel(morpho_ops_lut)
morpho_shape_menu = TextCarousel(morpho_shapes_lut)
morpho_ui = VStack(
    HStack(morpho_operation_menu, morpho_shape_menu),
    HStack(morpho_ksize_slider, morpho_aradj_slider, morpho_iter_slider),
)

# Kuwahara UI
kuwa_winsize_slider = Slider("Window Size", 3, 0, 21, step=1)
kuwa_aradj_slider = Slider("Aspect Adjust", 0, -1, 1, step=0.05)
kuwa_iter_slider = Slider("Iterations", 1, 1, 5, step=1)
kuwahara_ui = HStack(kuwa_winsize_slider, kuwa_aradj_slider, kuwa_iter_slider)

# Dirty blur UI
dblur_strength_slider = Slider("Blur Strength", 3, 0, 50, step=0.1)
dblur_aradj_slider = Slider("Aspect Adjust", 0, -1, 1, step=0.05)
dblur_seed_slider = Slider("Random Seed", 50, 0, 100, step=1)
dirtyblur_ui = HStack(dblur_strength_slider, dblur_aradj_slider, dblur_seed_slider)

# Build filter selection & swapper used to switch controls
radio_labels = ("Box", "Gaussian", "Bilateral", "Median", "Morpho", "Kuwahara", "Dirty")
filter_select = RadioBar(*radio_labels, height=60).set_label("Morpho")
swap_filter_ctrls_ui = Swapper(
    bblur_ui,
    gblur_ui,
    bilater_ui,
    medianblur_ui,
    morpho_ui,
    kuwahara_ui,
    dirtyblur_ui,
    keys=radio_labels,
)

# Stack elements together to form layout for display
show_playback_bar = not (is_webcam_source or is_image_source)
ui_layout = VStack(
    filter_select,
    HStack(img_size_slider, time_txt, flex=(1, 0)) if show_imgsize_slider else None,
    img_elem,
    playback_slider if show_playback_bar else None,
    swap_filter_ctrls_ui,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Display loop ***

# Set up display window and attach UI for mouse interactions
window = DisplayWindow(display_fps=min(vreader.get_framerate(), 60))
window.enable_size_control(display_size)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callbacks(
    {
        "Toggle playback": {" ": vreader.toggle_pause} if not is_image_source else None,
        "Step video backwards/forwards": (
            {",": vreader.prev_frame, ".": vreader.next_frame} if show_playback_bar else None
        ),
        "Switch image filter": {KEY.L_ARROW: filter_select.prev, KEY.R_ARROW: filter_select.next},
        "Adjust image size": (
            {"[": img_size_slider.decrement, "]": img_size_slider.increment} if show_imgsize_slider else None
        ),
        "Toggle original image": {KEY.TAB: hidden_show_orig_btn.toggle},
    }
).report_keypress_descriptions()
print("- Right click sliders to reset values")

# Prevent over-use of CPU (bilateral filter can lock up machine with some settings!)
cv2.setNumThreads(cv2.getNumberOfCPUs() // 2)

# Set up morphological kernel on startup, to ensure it exists for first-use
morpho_kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (1, 1))
with window.auto_close(vreader.release):

    for is_paused, frame_idx, frame in vreader:
        playback_slider.update_state(is_paused, frame_idx)

        # Read shared controls (these don't depend on selected filter)
        _, max_img_size = img_size_slider.read()
        _, show_original = hidden_show_orig_btn.read()

        # Only change controls UI on swap change (e.g. don't re-set every frame)
        is_filter_changed, _, filter_key = filter_select.read()
        if is_filter_changed:
            swap_filter_ctrls_ui.set_swap_key(filter_key)

        # Disable filtering when showing original (saves cpu usage)
        if show_original:
            filter_key = None

        # Resize frame as needed (has a significant effect on blurring!)
        scale_hw = get_image_hw_for_max_side_length(frame.shape, max_img_size)
        scaled_frame = resize_hw(frame, scale_hw, interpolation=cv2.INTER_NEAREST)

        # Big ugly if/else-if to handle different filter control reading + logic
        t1 = perf_counter()
        if filter_key == "Box":
            _, bblur_ksize = bblur_ksize_slider.read()
            _, bblur_aradj = bblur_aradj_slider.read()

            bblur_xy = get_xy_with_ar_adjustment(bblur_ksize, bblur_aradj)
            bblur_xy = [max(1, size) for size in bblur_xy]
            result = cv2.blur(scaled_frame, bblur_xy)

        elif filter_key == "Gaussian":
            _, gblur_ksize = gblur_ksize_slider.read()
            _, gblur_aradj = gblur_aradj_slider.read()
            _, gblur_sigma = gblur_sigma_slider.read()

            # Switch between auto-ksize vs. auto-sigma for gaussian blurring
            use_ksize = gblur_ksize > 0
            use_sigma_size = gblur_ksize == 0 and gblur_sigma > 0
            if use_ksize:
                gblur_xy = [1 + 2 * size for size in get_xy_with_ar_adjustment(gblur_ksize - 1, gblur_aradj)]
                gblur_sig_x = gblur_sig_y = gblur_sigma
                result = cv2.GaussianBlur(scaled_frame, gblur_xy, gblur_sig_x, sigmaY=gblur_sig_y)

            elif use_sigma_size:
                gblur_xy = (0, 0)
                gblur_sig_x, gblur_sig_y = get_xy_with_ar_adjustment(gblur_sigma, gblur_aradj, round_result=False)
                gblur_sig_x, gblur_sig_y = [max(1e-6, sig) for sig in (gblur_sig_x, gblur_sig_y)]
                result = cv2.GaussianBlur(scaled_frame, gblur_xy, gblur_sig_x, sigmaY=gblur_sig_y)
            else:
                result = scaled_frame

        elif filter_key == "Bilateral":
            _, bl_diameter = bilateral_d_slider.read()
            _, bl_sig_color = bilateral_sig_color_slider.read()
            _, bl_sig_space = bilateral_sig_space_slider.read()
            result = cv2.bilateralFilter(scaled_frame, bl_diameter, bl_sig_color, bl_sig_space)

        elif filter_key == "Median":
            _, mblur_ksize = mblur_ksize_slider.read()
            _, mblur_iter = mblur_iter_slider.read()

            mblur_res = scaled_frame
            for _ in range(mblur_iter):
                mblur_res = cv2.medianBlur(mblur_res, 1 + 2 * mblur_ksize)
            result = mblur_res

        elif filter_key == "Morpho":
            is_morpho_ksize_changed, morpho_ksize = morpho_ksize_slider.read()
            is_morpho_aradj_changed, morpho_aradj = morpho_aradj_slider.read()
            _, morpho_iters = morpho_iter_slider.read()
            _, _, morpho_op = morpho_operation_menu.read()
            is_morpho_shape_changed, _, morpho_shape = morpho_shape_menu.read()

            if morpho_ksize > 1:
                if is_morpho_shape_changed or is_morpho_ksize_changed or is_morpho_aradj_changed:
                    max_y = 1 + int(morpho_shape == cv2.MORPH_ELLIPSE)  # weird bug in opencv?
                    morpho_xy = get_xy_with_ar_adjustment(morpho_ksize, morpho_aradj)
                    morpho_xy = (max(1, morpho_xy[0]), max(max_y, morpho_xy[1]))
                    morpho_kernel = cv2.getStructuringElement(morpho_shape, morpho_xy)
                result = cv2.morphologyEx(scaled_frame, morpho_op, morpho_kernel, iterations=morpho_iters)
            else:
                result = scaled_frame

        elif filter_key == "Dirty":
            _, dblur_strength = dblur_strength_slider.read()
            _, dblur_aradj = dblur_aradj_slider.read()
            _, dblur_seed = dblur_seed_slider.read()
            seed = None if dblur_seed == 0 else dblur_seed
            result = dirty_blur(scaled_frame, dblur_strength, dblur_aradj, random_seed=seed)

        elif filter_key == "Kuwahara":
            _, kuwa_winsize = kuwa_winsize_slider.read()
            _, kuwa_aradj = kuwa_aradj_slider.read()
            _, kuwa_iter = kuwa_iter_slider.read()

            kuwa_xy = get_xy_with_ar_adjustment(kuwa_winsize, kuwa_aradj)
            kuwa_res = scaled_frame
            for _ in range(kuwa_iter):
                kuwa_res = kuwahara_filter(kuwa_res, kuwa_xy)
            result = kuwa_res

        else:
            result = scaled_frame

        # Compute per-frame filtering time
        t2 = perf_counter()
        time_taken_ms = round(1000 * (t2 - t1))
        time_txt.set_text(time_taken_ms)

        # Update displayed image & render
        img_elem.set_image(scaled_frame if show_original else result)
        display_image = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        pass
    pass
