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
from toadui.sliders import Slider, MultiSlider
from toadui.carousels import TextCarousel
from toadui.text import PrefixedTextBlock
from toadui.helpers.sizing import get_image_hw_for_max_side_length, resize_hw
from toadui.helpers.data_management import MaxLengthKVStorage
from toadui.helpers.text import TextDrawer
from toadui.helpers.sampling import cosine_interp


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set arg defaults
default_input_path = None
default_display_size = 900
default_operate_size = 800
default_max_buffer_mb = 250

# Define script arguments
parser = argparse.ArgumentParser(description="Demo of OpenCV edge detection functions")
parser.add_argument("-i", "--input_path", default=default_input_path, type=str, help="Path to video")
parser.add_argument("-d", "--display_size", default=default_display_size, type=int, help="Initial window size")
parser.add_argument("-s", "--operating_size", default=default_operate_size, type=int, help="Initial working frame size")
parser.add_argument("-cam", "--use_webcam", action="store_true", help="Use webcam as video source")
parser.add_argument(
    "-ram",
    "--max_buffer_ram_mb",
    default=default_max_buffer_mb,
    type=int,
    help="Max RAM for storing prior frames, for videos only (default: {default_max_buffer_mb})",
)

# For convenience
args = parser.parse_args()
input_path = args.input_path if not args.use_webcam else "cam"
display_size = args.display_size
init_operating_size = args.operating_size
max_buffer_mb = args.max_buffer_ram_mb


# ---------------------------------------------------------------------------------------------------------------------
# %% Helpers


def get_xy_with_ar_adjustment(size: int | float, ar_adjust: float, round_result: bool = True) -> tuple[int | float]:
    """
    Helper used to convert 'size' & 'aspect_adjustment' into (size_x, size_y) values.
    When ar_adjust is -1 -> (size, 0)
         ar_adjust is  0 -> (size, size)
         ar_adjust is +1 -> (0, size)
    """
    adj_w = float(size * np.clip(1 - ar_adjust, 0, 1))
    adj_h = float(size * np.clip(1 + ar_adjust, 0, 1))
    return tuple(round(val) for val in (adj_w, adj_h)) if round_result else (adj_w, adj_h)


# Define nice 'label' mapping to ocv morphological constants
morpho_shapes_lut = {"Ellipse": cv2.MORPH_ELLIPSE, "Rectangle": cv2.MORPH_RECT, "Cross": cv2.MORPH_CROSS}
morpho_ops_lut = {"Gradient": cv2.MORPH_GRADIENT, "Blackhat": cv2.MORPH_BLACKHAT, "Tophat": cv2.MORPH_TOPHAT}


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup UI

# Handle webcam inputs
input_path = ask_for_video_path(input_path, path_type="video or image", allow_webcam_inputs=True)
is_webcam_source, input_path = read_webcam_string(input_path)
is_image_source, vreader = load_looping_video_or_image(input_path)
is_video_source = is_webcam_source or (not is_image_source)
sample_frame = vreader.get_sample_frame()
img_h, img_w = sample_frame.shape[0:2]

# Figure out image size bounds
largest_size = max(img_h, img_w)
smallest_size = max(1, min(largest_size // 2, 50))
initial_size = min(init_operating_size, largest_size)
show_imgsize_slider = largest_size > 50

# Define elements shared for all filters
img_elem = FixedARImage(vreader.get_sample_frame(), resize_interpolation=cv2.INTER_AREA)
playback_slider = VideoPlaybackSlider(vreader)
img_size_slider = Slider("Image size", initial_size, smallest_size, largest_size, step=5)
preblur_slider = Slider("Pre-blur", 1, 0, 5, step=0.1, marker_step=1)
show_orig_btn = ToggleButton("Show Original", color_on=(80, 110, 90))
invert_btn = ToggleButton("Invert", color_on=100)
overlay_btn = ToggleButton("Overlay", color_on=(135, 120, 50))
time_txt = PrefixedTextBlock("", "-", " ms")

# Sobel UI
sobel_ksize_slider = Slider("Kernel Size", 3, 1, 31, step=2)
sobel_order_slider = Slider("Order", 1, 1, 5, step=1)
sobel_aradj_slider = Slider("Aspect Adjust", 0, -1, 1, step=0.05, marker_step=1).set(1, use_as_default_value=False)
sobel_scale_slider = Slider("Scale", 1, -100, 100, step=0.5, marker_step=100).set(10, use_as_default_value=False)
sobel_offset_slider = Slider("Offset", 0, -512, 512, step=1, marker_step=512)
sobel_use_scharr = ToggleButton(" Use Scharr ", color_on=(90, 50, 70))
sobel_ui = VStack(
    HStack(sobel_ksize_slider, sobel_order_slider, sobel_aradj_slider),
    HStack(sobel_scale_slider, sobel_offset_slider, sobel_use_scharr, flex=(1, 1, 0)),
)

# Laplacian UI
lapa_ksize_slider = Slider("Kernel Size", 3, 1, 31, step=2, marker_step=6)
lapa_scale_slider = Slider("Scale", 1, -100, 100, step=0.5, marker_step=100).set(8, use_as_default_value=False)
lapa_offset_slider = Slider("Offset", 0, -512, 512, step=1, marker_step=512)
lapa_ui = VStack(lapa_ksize_slider, HStack(lapa_scale_slider, lapa_offset_slider))

# Unsharp UI
unsharp_ksize_slider = Slider("Kernel Size", 1, 0, 10, step=0.1)
unsharp_aradj_slider = Slider("Aspect Adjust", 0, -1, 1, step=0.05, marker_step=1)
unsharp_scale_slider = Slider("Scale", 0, -100, 100, step=0.5, marker_step=100).set(10, use_as_default_value=False)
unsharp_offset_slider = Slider("Offset", 0, 0, 512, step=1)
unsharp_ui = VStack(
    HStack(unsharp_ksize_slider, unsharp_aradj_slider, flex=(3, 1)),
    HStack(unsharp_scale_slider, unsharp_offset_slider),
)

# Canny UI
canny_thresh_slider = MultiSlider("Thresholds", (0, 255), 0, 255, step=1, marker_step=32)
canny_aperture_slider = Slider("Aperture", 3, 3, 7, step=2, marker_step=2)
canny_use_l2_btn = ToggleButton("Use L2 Gradient", color_on=(10, 80, 200))
canny_ui = VStack(canny_thresh_slider, HStack(canny_aperture_slider, canny_use_l2_btn))

# Morphological filter UI
morpho_ksize_slider = Slider("Kernel Size", 5, 1, 50, step=1)
morpho_aradj_slider = Slider("Aspect Adjust", 0, -1, 1, step=0.05, marker_step=1)
morpho_iter_slider = Slider("Iterations", 1, 1, 50, step=1)
morpho_operation_menu = TextCarousel(morpho_ops_lut)
morpho_shape_menu = TextCarousel(morpho_shapes_lut)
morpho_ui = VStack(
    HStack(morpho_operation_menu, morpho_shape_menu),
    HStack(morpho_ksize_slider, morpho_aradj_slider, morpho_iter_slider),
)

# Frame delta UI
fdelta_txt = TextDrawer()
max_num_frames = max(1, int(max_buffer_mb * 1_000_000 / sample_frame.nbytes))
prev_frames_f32_dict = MaxLengthKVStorage(max_length=1 if not (is_video_source) else max_num_frames)
framedelta_index_slider = Slider("Back Index", min(5, max_num_frames), 1, max_num_frames, step=1)
framedelta_scale_slider = Slider("Scale", 1, -50, 50, step=0.5, marker_step=50)
framedelta_offset_slider = Slider("Offset", 0, 0, 512, step=1)
frame_delta_ui = VStack(
    framedelta_index_slider,
    HStack(framedelta_scale_slider, framedelta_offset_slider),
)

# Build filter selection & swapper used to switch controls
radio_labels = ("Sobel", "Laplacian", "Unsharp", "Canny", "Morpho", "Delta" if is_video_source else None)
filter_select = RadioBar(*radio_labels).set_label("Unsharp")
swap_filter_ctrls_ui = Swapper(
    sobel_ui,
    lapa_ui,
    unsharp_ui,
    canny_ui,
    morpho_ui,
    frame_delta_ui,
    keys=radio_labels,
)

# Stack elements together to form layout for display
show_playback_bar = not (is_webcam_source or is_image_source)
ui_layout = VStack(
    filter_select,
    HStack(img_size_slider, preblur_slider, time_txt, flex=(1, 1, 0), min_w=600),
    img_elem,
    playback_slider if show_playback_bar else None,
    HStack(show_orig_btn, invert_btn, overlay_btn),
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
        "Play/Pause the video": {" ": vreader.toggle_pause} if not is_image_source else None,
        "Step video backwards/forwards": (
            {",": vreader.prev_frame, ".": vreader.next_frame} if is_video_source else None
        ),
        "Switch filter": {KEY.L_ARROW: filter_select.prev, KEY.R_ARROW: filter_select.next},
        "Adjust image size": (
            {"[": img_size_slider.decrement, "]": img_size_slider.increment} if show_imgsize_slider else None
        ),
        "Adjust pre-blur": {"v": preblur_slider.decrement, "b": preblur_slider.increment},
        "Toggle original image": {KEY.TAB: show_orig_btn.toggle},
        "Toggle invert": {"i": invert_btn.toggle},
        "Toggle overlay": {"o": overlay_btn.toggle},
    }
).report_keypress_descriptions()
print("- Right click sliders to reset values")

# Set up morphological kernel on startup, to ensure it exists for first-use
morpho_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
with window.auto_close(vreader.release):

    for is_paused, frame_idx, frame in vreader:
        playback_slider.update_state(is_paused, frame_idx)

        # Read shared controls (these don't depend on selected filter)
        is_img_size_changed, max_img_size = img_size_slider.read()
        is_show_orig_changed, show_original = show_orig_btn.read()
        _, invert_edges = invert_btn.read()
        is_overlay_changed, show_overlay = overlay_btn.read()
        _, preblur_ksize = preblur_slider.read()

        # Disable original/overlay when user toggles overlay/original buttons
        if is_show_orig_changed and show_original:
            overlay_btn.toggle(False)
        if is_overlay_changed and show_overlay:
            show_orig_btn.toggle(False)

        # Only change controls UI on swap change (e.g. don't re-set every frame)
        is_filter_changed, _, filter_key = filter_select.read()
        if is_filter_changed:
            swap_filter_ctrls_ui.set_swap_key(filter_key)

        # Disable filtering when showing original (saves cpu usage)
        if show_original:
            filter_key = None

        # Resize frame as needed
        scale_hw = get_image_hw_for_max_side_length(frame.shape, max_img_size)
        scaled_frame = resize_hw(frame, scale_hw)
        grayframe = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        if preblur_ksize > 0:
            grayframe = cv2.GaussianBlur(grayframe, (0, 0), preblur_ksize)

        # # Big ugly if/else-if to handle different filter control reading + logic
        t1 = perf_counter()
        if filter_key == "Sobel":
            _, s_ksize = sobel_ksize_slider.read()
            _, s_order = sobel_order_slider.read()
            _, s_aradj = sobel_aradj_slider.read()
            _, s_scale = sobel_scale_slider.read()
            _, s_offset = sobel_offset_slider.read()
            _, s_use_scharr = sobel_use_scharr.read()

            # Compute x/y derivative orders, with constraints imposed by OCV function
            s_dx, s_dy = get_xy_with_ar_adjustment(s_order, s_aradj)
            s_dx = min(s_dx, s_ksize - 1)
            s_dy = min(s_dy, s_ksize - 1)
            if s_dx == 0 and s_dy == 0:
                s_dx, s_dy = 1, 1

            # Special case (see cv2.Sobel docs). (dx,dy) must be (0,1) or (1,0) or else function fails
            if s_use_scharr:
                s_ksize = cv2.FILTER_SCHARR
                s_dy = 1 if s_aradj > 0 else 0
                s_dx = 1 - s_dy

            result = cv2.Sobel(grayframe, -1, s_dx, s_dy, ksize=s_ksize, scale=s_scale, delta=s_offset)

        elif filter_key == "Laplacian":
            _, l_ksize = lapa_ksize_slider.read()
            _, l_scale = lapa_scale_slider.read()
            _, l_offset = lapa_offset_slider.read()
            result = cv2.Laplacian(grayframe, -1, ksize=l_ksize, scale=l_scale, delta=l_offset)

        elif filter_key == "Canny":
            _, (c_thresh1, c_thresh2) = canny_thresh_slider.read()
            _, c_aperture = canny_aperture_slider.read()
            _, c_use_l2 = canny_use_l2_btn.read()
            result = cv2.Canny(grayframe, c_thresh1, c_thresh2, apertureSize=c_aperture, L2gradient=c_use_l2)

        elif filter_key == "Unsharp":
            _, u_ksize = unsharp_ksize_slider.read()
            _, u_aradj = unsharp_aradj_slider.read()
            _, u_scale = unsharp_scale_slider.read()
            _, u_offset = unsharp_offset_slider.read()

            # Handle aspect adjustment & edge cases (gaussian-blur doesn't like 0-ksize & 0-sigma)
            u_sig_x, u_sig_y = get_xy_with_ar_adjustment(u_ksize, u_aradj, round_result=False)
            kx, ky = [1 if sig == 0 else 0 for sig in (u_sig_x, u_sig_y)]

            # Compute 'image details' between image & blurred-copy of image
            # -> Intuition: image = blurred_image + image_details
            #           so: image_details = image - blurred_image
            gray_f32 = grayframe.astype(np.float32)
            blur_frame_f32 = cv2.GaussianBlur(gray_f32, (kx, ky), sigmaX=u_sig_x, sigmaY=u_sig_y)
            result_f32 = gray_f32 - blur_frame_f32
            result_f32 = (2.0 * u_scale) * result_f32 + u_offset
            result = np.clip(np.round(result_f32), 0, 255).astype(np.uint8)

        elif filter_key == "Morpho":
            is_morpho_ksize_changed, morpho_ksize = morpho_ksize_slider.read()
            is_morpho_aradj_changed, morpho_aradj = morpho_aradj_slider.read()
            _, morpho_iters = morpho_iter_slider.read()
            _, _, morpho_op = morpho_operation_menu.read()
            is_morpho_shape_changed, _, morpho_shape = morpho_shape_menu.read()

            # Only re-build morphological kernel when it changed
            kernel_is_changed = any((is_morpho_shape_changed, is_morpho_ksize_changed, is_morpho_aradj_changed))
            if is_filter_changed or kernel_is_changed:
                max_y = 1 + int(morpho_shape == cv2.MORPH_ELLIPSE)  # weird bug in opencv?
                morpho_xy = get_xy_with_ar_adjustment(morpho_ksize, morpho_aradj)
                morpho_xy = (max(1, morpho_xy[0]), max(max_y, morpho_xy[1]))
                morpho_kernel = cv2.getStructuringElement(morpho_shape, morpho_xy)
            result = cv2.morphologyEx(grayframe, morpho_op, morpho_kernel, iterations=morpho_iters)

        elif filter_key == "Delta":
            _, fdelta_idx = framedelta_index_slider.read()
            _, fdelta_scale = framedelta_scale_slider.read()
            _, fdelta_offset = framedelta_offset_slider.read()

            # Wipe out storage if sizing changes (we can't do frame-to-frame subtraction otherwise)
            if is_img_size_changed:
                prev_frames_f32_dict.clear()

            # Get previous frame, if possible
            back_idx = frame_idx - fdelta_idx
            prev_frame = prev_frames_f32_dict.get(back_idx, None)
            is_valid_prev_frame = prev_frame is not None

            # Compute frame delta or give feedback that there is no previous frame
            gray_f32 = grayframe.astype(np.float32)
            if is_valid_prev_frame:
                result_f32 = gray_f32 - prev_frame
                result_f32 = (fdelta_scale) * result_f32 + fdelta_offset
                result = np.clip(np.round(result_f32), 0, 255).astype(np.uint8)
            else:
                result = np.zeros_like(grayframe)
                fdelta_txt.xy_centered(result, "No previous frame!")

            # Store previous frames
            if not is_paused:
                prev_frames_f32_dict.store(frame_idx, gray_f32)

        else:
            # If we get here, we're not doing any filtering, but we need 'result' to exist so just use frame
            result = grayframe

        # Do extra processing to handle different display modes
        if show_original:
            result = scaled_frame

        elif show_overlay:
            # To make overlay, we modify the Y (brightness) channel of YUV of original image
            yuv_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2YUV)
            y_frame = yuv_frame[:, :, 0].astype(np.float32)

            # Interpolate between image brightness and blank color frame using edge results as weighting
            edge_color_frame = np.full_like(y_frame, 255) if invert_edges else np.zeros_like(y_frame)
            new_y_frame = cosine_interp(y_frame, edge_color_frame, np.float32(result) / 255.0)

            # Replace original brightness channel (and restore BGR) for final output
            yuv_frame[:, :, 0] = new_y_frame.astype(np.uint8)
            result = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)

        elif invert_edges:
            result = 255 - result

        # Compute per-frame filtering time
        t2 = perf_counter()
        time_taken_ms = round(1000 * (t2 - t1))
        time_txt.set_text(time_taken_ms)

        # Update displayed image & render
        img_elem.set_image(result)
        display_image = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        pass
    pass
