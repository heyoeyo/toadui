#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse

import cv2
import numpy as np

from toadui.video import VideoPlaybackSlider, load_looping_video_or_image, read_webcam_string, ask_for_video_path
from toadui.window import DisplayWindow, KEY
from toadui.buttons import ToggleButton, RadioBar
from toadui.sliders import MultiSlider
from toadui.images import DynamicImage
from toadui.plots import SimpleHistogramPlot
from toadui.layout import VStack, HStack, GridStack, Swapper
from toadui.colormaps import apply_colormap, make_colormap_from_keypoints
from toadui.helpers.images import histogram_equalization


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set arg defaults
default_input_path = None
default_display_size = 900
default_histogram_bins = 256
default_cspace = "RGB"

# Define script arguments
parser = argparse.ArgumentParser(description="Demo showing video or image data in different color spaces")
parser.add_argument("-i", "--input_path", default=default_input_path, type=str, help="Path to video or image")
parser.add_argument("-d", "--display_size", default=default_display_size, type=int, help="Initial window size")
parser.add_argument("-cam", "--use_webcam", action="store_true", help="Use webcam as video source")
parser.add_argument(
    "-b", "--histogram_bins", default=default_histogram_bins, type=int, help="Bin count used to compute histograms"
)
parser.add_argument("-s", "--initial_cspace", default=default_cspace, type=str, help="Color space used on startup")
parser.add_argument("-fw", "--force_wide", default=False, action="store_true", help="Force wide UI")
parser.add_argument("-ft", "--force_tall", default=False, action="store_true", help="Force tall UI")

# For convenience
args = parser.parse_args()
input_path = args.input_path if not args.use_webcam else "cam"
display_size = args.display_size
num_histo_bins = args.histogram_bins
initial_cspace = args.initial_cspace
force_wide_ui = args.force_wide
force_tall_ui = args.force_tall and not force_wide_ui


# ---------------------------------------------------------------------------------------------------------------------
# %% Helpers


class ColorSplitter:
    """
    Class used to split bgr images into channel components after performing a color space conversion.
    Also includes support for false-color mapping for representing single-channel components.
    """

    def __init__(self, convert_forward_code, convert_backward_code, ch1_cmap, ch2_cmap, ch3_cmap):
        self.forward_code = convert_forward_code
        self.backward_code = convert_backward_code
        self.c1_cmap = ch1_cmap
        self.c2_cmap = ch2_cmap
        self.c3_cmap = ch3_cmap

    def split_channels(self, frame):
        return cv2.split(cv2.cvtColor(frame, self.forward_code))

    def rebuild_3ch_image(self, ch1_frame, ch2_frame, ch3_frame):
        return cv2.cvtColor(np.dstack((ch1_frame, ch2_frame, ch3_frame)), self.backward_code)

    def create_false_colors(self, ch1_frame, ch2_frame, ch3_frame):
        ch1_frame = apply_colormap(ch1_frame, self.c1_cmap)
        ch2_frame = apply_colormap(ch2_frame, self.c2_cmap)
        ch3_frame = apply_colormap(ch3_frame, self.c3_cmap)
        return ch1_frame, ch2_frame, ch3_frame


class RGBSplitter(ColorSplitter):
    def __init__(self):
        r_cmap = make_colormap_from_keypoints([(0, 0, 0), (0, 0, 255)])
        g_cmap = make_colormap_from_keypoints([(0, 0, 0), (0, 255, 0)])
        b_cmap = make_colormap_from_keypoints([(0, 0, 0), (255, 0, 0)])
        super().__init__(cv2.COLOR_BGR2RGB, cv2.COLOR_RGB2BGR, r_cmap, g_cmap, b_cmap)


class LABSplitter(ColorSplitter):
    def __init__(self):
        l_cmap = make_colormap_from_keypoints([(0, 0, 0), (255, 255, 255)])
        a_cmap = make_colormap_from_keypoints([(0, 255, 0), (0, 0, 255)])
        b_cmap = make_colormap_from_keypoints([(255, 0, 0), (0, 255, 255)])
        super().__init__(cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR, l_cmap, a_cmap, b_cmap)


class LUVSplitter(ColorSplitter):
    def __init__(self):
        l_cmap = make_colormap_from_keypoints([(0, 0, 0), (255, 255, 255)])
        u_cmap = make_colormap_from_keypoints([(145, 160, 0), (128, 128, 128), (100, 0, 255)])
        v_cmap = make_colormap_from_keypoints([(255, 0, 200), (128, 128, 128), (0, 120, 140)])
        super().__init__(cv2.COLOR_BGR2LUV, cv2.COLOR_LUV2BGR, l_cmap, u_cmap, v_cmap)


class HSVSplitter(ColorSplitter):
    def __init__(self):
        full_hue_keypoints = [(0, 255, 255), (255, 255, 255)]
        h_cmap = make_colormap_from_keypoints(full_hue_keypoints, color_conversion_code=cv2.COLOR_HSV2BGR_FULL)
        s_cmap = make_colormap_from_keypoints([(0, 0, 0), (255, 255, 255)])
        v_cmap = make_colormap_from_keypoints([(0, 0, 0), (255, 255, 255)])
        super().__init__(cv2.COLOR_BGR2HSV_FULL, cv2.COLOR_HSV2BGR_FULL, h_cmap, s_cmap, v_cmap)


class HLSSplitter(ColorSplitter):
    def __init__(self):
        full_hue_keypoints = [(0, 128, 255), (255, 128, 255)]
        h_cmap = make_colormap_from_keypoints(full_hue_keypoints, color_conversion_code=cv2.COLOR_HLS2BGR_FULL)
        l_cmap = make_colormap_from_keypoints([(0, 0, 0), (255, 255, 255)])
        s_cmap = make_colormap_from_keypoints([(0, 0, 0), (255, 255, 255)])
        super().__init__(cv2.COLOR_BGR2HLS_FULL, cv2.COLOR_HLS2BGR_FULL, h_cmap, l_cmap, s_cmap)


class YUVSplitter(ColorSplitter):
    def __init__(self):
        y_cmap = make_colormap_from_keypoints([(0, 0, 0), (255, 255, 255)])
        u_cmap = make_colormap_from_keypoints([(0, 180, 128), (255, 80, 128)])
        v_cmap = make_colormap_from_keypoints([(128, 200, 0), (128, 55, 255)])
        super().__init__(cv2.COLOR_BGR2YUV, cv2.COLOR_YUV2BGR, y_cmap, u_cmap, v_cmap)


class YCrCbSplitter(ColorSplitter):
    def __init__(self):
        y_cmap = make_colormap_from_keypoints([(0, 0, 0), (255, 255, 255)])
        cr_cmap = make_colormap_from_keypoints([(0, 220, 85), (128, 40, 255)])
        cb_cmap = make_colormap_from_keypoints([(0, 170, 128), (255, 85, 128)])
        super().__init__(cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR, y_cmap, cr_cmap, cb_cmap)


class XYZSplitter(ColorSplitter):
    def __init__(self):
        x_cmap = make_colormap_from_keypoints([(110, 245, 0), (110, 0, 255)])
        y_cmap = make_colormap_from_keypoints([(140, 0, 255), (90, 255, 0)])
        z_cmap = make_colormap_from_keypoints([(0, 115, 220), (250, 128, 90)])
        super().__init__(cv2.COLOR_BGR2XYZ, cv2.COLOR_XYZ2BGR, x_cmap, y_cmap, z_cmap)


def adjust_channel_frame(channel_frame, low_threshold: float, high_threshold: float, use_histogram_equalization: bool):
    """Helper function used to handle per-channel frame processing (equalization + thresholding)"""
    if use_histogram_equalization:
        channel_frame = histogram_equalization(channel_frame, low_threshold, high_threshold)
    elif low_threshold > 0 or high_threshold < 1:
        channel_frame = np.clip(channel_frame, round(255 * low_threshold), round(255 * high_threshold))

    return channel_frame


# Define lookup table for how to handle colorspace splitting & single-channel titles
cspace_splitters_and_titles_lut = {
    "RGB": [RGBSplitter(), ("Red", "Green", "Blue")],
    "LAB": [LABSplitter(), ("Lightness", "Green-Red", "Blue-Yellow")],
    "LUV": [LUVSplitter(), ("Lightness", "Green-Red", "Blue-Yellow")],
    "HSV": [HSVSplitter(), ("Hue", "Saturation", "Value")],
    "HLS": [HLSSplitter(), ("Hue", "Lightness", "Saturation")],
    "YUV": [YUVSplitter(), ("Luma", "Chroma Blue", "Chroma Red")],
    "YCrCb": [YCrCbSplitter(), ("Luma", "Chroma Red", "Chroma Blue")],
    "XYZ": [XYZSplitter(), ("X", "Y", "Z")],
}
valid_cspaces = tuple(cspace_splitters_and_titles_lut.keys())
assert (
    initial_cspace in valid_cspaces
), f"Invalid initial color space ({initial_cspace}).\nMust be one of: {', '.join(valid_cspaces)}"


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup input source

# Handle webcam inputs
input_path = ask_for_video_path(input_path, path_type="video or image", allow_webcam_inputs=True)
is_webcam_source, input_path = read_webcam_string(input_path)
is_image_source, vreader = load_looping_video_or_image(input_path, display_size)


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup UI

# Create main display components
sample_frame = vreader.get_sample_frame()
playback_slider = VideoPlaybackSlider(vreader)
bgr_img_elem, ch1_img_elem, ch2_img_elem, ch3_img_elem = [DynamicImage(sample_frame) for _ in range(4)]
input_ar = sample_frame.shape[1] / sample_frame.shape[0]

# Set up histogram plots for each channel
histo_bin_range = (0, 255, num_histo_bins)
ch1_histo_plot = SimpleHistogramPlot("Channel 1", histo_bin_range, aspect_ratio=input_ar)
ch2_histo_plot = SimpleHistogramPlot("Channel 2", histo_bin_range, aspect_ratio=input_ar)
ch3_histo_plot = SimpleHistogramPlot("Channel 3", histo_bin_range, aspect_ratio=input_ar)

# Create swapper for each channel image (so we can switch between false-color & histograms)
ch_swap_keys = ("channel", "histogram")
ch1_swap = Swapper(ch1_img_elem, ch1_histo_plot, keys=ch_swap_keys)
ch2_swap = Swapper(ch2_img_elem, ch2_histo_plot, keys=ch_swap_keys)
ch3_swap = Swapper(ch3_img_elem, ch3_histo_plot, keys=ch_swap_keys)

# Create color space selector & mode toggle buttons
cspace_radio = RadioBar(*valid_cspaces, label_padding=2).set_label(initial_cspace)
show_channels_btn = ToggleButton("Show Channels", color_on=(110, 110, 70), default_state=True)
show_histos_btn = ToggleButton("Show Histograms", color_on=(110, 60, 110))
disable_adjustments_btn = ToggleButton("Disable Adjustments", color_on=(90, 110, 110))

# Create thresholding sliders for each channel + histo-equalize toggle
ch1_thresh = MultiSlider("Channel 1 Thresholds", (0, 1), 0, 1, 0.01, marker_step=0.25, fill_color=(40, 50, 70))
ch2_thresh = MultiSlider("Channel 2 Thresholds", (0, 1), 0, 1, 0.01, marker_step=0.25, fill_color=(50, 70, 60))
ch3_thresh = MultiSlider("Channel 3 Thresholds", (0, 1), 0, 1, 0.01, marker_step=0.25, fill_color=(70, 50, 40))
heq1_btn = ToggleButton(" Equalize ", color_on=(80, 80, 135))
heq2_btn = ToggleButton(" Equalize ", color_on=(80, 135, 80), text_color_on=255)
heq3_btn = ToggleButton(" Equalize ", color_on=(135, 80, 80))

# Set up main display configuration
num_grid_rows = 2
is_very_tall, is_very_wide = input_ar < 0.5, input_ar > 2
num_grid_rows = 4 if (is_very_wide or force_wide_ui) else num_grid_rows
num_grid_rows = 1 if (is_very_tall or force_tall_ui) else num_grid_rows
img_swap_elem = Swapper(
    GridStack(bgr_img_elem, ch1_swap, ch2_swap, ch3_swap, num_rows=num_grid_rows),
    bgr_img_elem,
    keys=("grid", "frame-only"),
)

# Build final layout
slider_flex = (1, 0)
ui_layout = VStack(
    cspace_radio,
    img_swap_elem,
    playback_slider if not (is_webcam_source or is_image_source) else None,
    HStack(show_channels_btn, show_histos_btn, disable_adjustments_btn),
    HStack(ch1_thresh, heq1_btn, flex=slider_flex),
    HStack(ch2_thresh, heq2_btn, flex=slider_flex),
    HStack(ch3_thresh, heq3_btn, flex=slider_flex),
)


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Display loop ***

# Set up display window and pass UI events to layout object so UI can respond to mouse
window = DisplayWindow(display_fps=min(vreader.get_framerate(), 60))
window.enable_size_control(display_size)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callbacks(
    {
        "Toggle playback": {" ": vreader.toggle_pause} if not is_image_source else None,
        "Switch colorspaces": {KEY.L_ARROW: cspace_radio.prev, KEY.R_ARROW: cspace_radio.next},
        "Toggle channel view": {"c": show_channels_btn.toggle},
        "Toggle original": {"d": disable_adjustments_btn.toggle},
        "Toggle histograms": {"h": show_histos_btn.toggle},
        "Toggle equalization": {"1": heq1_btn.toggle, "2": heq2_btn.toggle, "3": heq3_btn.toggle},
        "Toggle histogram log scale": {
            "l": (ch1_histo_plot.toggle_log_scale, ch2_histo_plot.toggle_log_scale, ch3_histo_plot.toggle_log_scale)
        },
        "Toggle histogram bar plot": {
            "b": (ch1_histo_plot.toggle_bar_plot, ch2_histo_plot.toggle_bar_plot, ch3_histo_plot.toggle_bar_plot)
        },
    }
)
window.report_keypress_descriptions()

# Make sure initial colorspace is selected
splitter_select = cspace_splitters_and_titles_lut[initial_cspace]
with window.auto_close(vreader.release):

    for is_paused, frame_idx, frame in vreader:
        playback_slider.update_state(is_paused, frame_idx)

        # Read controls
        _, enable_heq1 = heq1_btn.read()
        _, enable_heq2 = heq2_btn.read()
        _, enable_heq3 = heq3_btn.read()
        is_cspace_changed, cspace_idx, cspace_label = cspace_radio.read()
        _, (ch1_low_thresh, ch1_high_thresh) = ch1_thresh.read()
        _, (ch2_low_thresh, ch2_high_thresh) = ch2_thresh.read()
        _, (ch3_low_thresh, ch3_high_thresh) = ch3_thresh.read()
        is_show_channels_changed, show_channels = show_channels_btn.read()
        is_show_histograms_changed, show_histograms = show_histos_btn.read()
        _, disable_adjustments = disable_adjustments_btn.read()

        # Swap between full image & grid view
        if is_show_channels_changed:
            img_swap_elem.set_swap_key("grid" if show_channels else "frame-only")

        # Swap between displaying channel images & histogram plots
        if is_show_histograms_changed:
            ch_swap_key = "histogram" if show_histograms else "channel"
            ch1_swap.set_swap_key(ch_swap_key)
            ch2_swap.set_swap_key(ch_swap_key)
            ch3_swap.set_swap_key(ch_swap_key)

            # Automatically show channels if user toggles histograms on
            if show_histograms:
                show_channels_btn.toggle(True)

        # Switch color splitter instance
        if is_cspace_changed:
            splitter_select, ch123_titles = cspace_splitters_and_titles_lut[cspace_label]
            ch1_histo_plot.set_title(ch123_titles[0])
            ch2_histo_plot.set_title(ch123_titles[1])
            ch3_histo_plot.set_title(ch123_titles[2])

        # Apply per-channel adjustments
        ch1_frame, ch2_frame, ch3_frame = splitter_select.split_channels(frame)
        ch1_frame = adjust_channel_frame(ch1_frame, ch1_low_thresh, ch1_high_thresh, enable_heq1)
        ch2_frame = adjust_channel_frame(ch2_frame, ch2_low_thresh, ch2_high_thresh, enable_heq2)
        ch3_frame = adjust_channel_frame(ch3_frame, ch3_low_thresh, ch3_high_thresh, enable_heq3)

        # Determine which input image to show (i.e. with/without channel adjustments)
        bgr_frame = frame
        if not disable_adjustments:
            bgr_frame = splitter_select.rebuild_3ch_image(ch1_frame, ch2_frame, ch3_frame)
        bgr_img_elem.set_image(bgr_frame)

        # Only do work to update histogram or false-color channel images, depending on which is showing
        if show_channels:
            if show_histograms:
                ch1_histo_plot.set_data(ch1_frame)
                ch2_histo_plot.set_data(ch2_frame)
                ch3_histo_plot.set_data(ch3_frame)
            else:
                ch1_bgr, ch2_bgr, ch3_bgr = splitter_select.create_false_colors(ch1_frame, ch2_frame, ch3_frame)
                ch1_img_elem.set_image(ch1_bgr)
                ch2_img_elem.set_image(ch2_bgr)
                ch3_img_elem.set_image(ch3_bgr)

        # Render display image
        display_image = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        pass
    pass
