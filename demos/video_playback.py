#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse

from toadui.window import DisplayWindow, KEY
from toadui.video import LoopingVideoReader, VideoPlaybackSlider, read_webcam_string, ask_for_video_path
from toadui.images import DynamicImage
from toadui.text import PrefixedTextBlock
from toadui.layout import VStack, HStack
from toadui.helpers.images import draw_circle_norm


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set arg defaults
default_input_path = None
default_display_size = 800

# Define script arguments
parser = argparse.ArgumentParser(description="Demo of controllable video playback")
parser.add_argument("-i", "--input_path", default=default_input_path, type=str, help="Path to video")
parser.add_argument("-d", "--display_size", default=default_display_size, type=int, help="Initial window size")
parser.add_argument("-cam", "--use_webcam", action="store_true", help="Use webcam as video source")

# For convenience
args = parser.parse_args()
input_path = args.input_path if not args.use_webcam else "cam"
display_size = args.display_size


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup UI

# Handle webcam inputs
input_path = ask_for_video_path(input_path, allow_webcam_inputs=True)
is_webcam_source, input_path = read_webcam_string(input_path)
vreader = LoopingVideoReader(input_path)

# Define UI elements
img_elem = DynamicImage(vreader.get_sample_frame())
playback_slider = VideoPlaybackSlider(vreader)
xynorm_block = PrefixedTextBlock("Mouse XY: ", suffix="", max_characters=16)
xypx_block = PrefixedTextBlock("Mouse XY: ", suffix=" px", max_characters=16)

# Stack elements together to form layout for display
show_playback_bar = not is_webcam_source
ui_layout = VStack(
    HStack(xynorm_block, xypx_block, flex=(1, 1), min_w=600),
    img_elem,
    playback_slider if show_playback_bar else None,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Display loop ***

# Set up display window and attach UI for mouse interactions
window = DisplayWindow(display_fps=min(vreader.get_framerate(), 60))
window.enable_size_control(display_size)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callback(" ", vreader.toggle_pause, "Play/Pause the video")
window.attach_keypress_callback(
    (KEY.L_ARROW, KEY.R_ARROW), (vreader.prev_frame, vreader.next_frame), "Step video backwards/forwards"
)

print("Keyboard Controls:")
window.report_keypress_descriptions(print_directly=True)

with window.auto_close(vreader.release):

    for is_paused, frame_idx, frame in vreader:
        playback_slider.update_state(is_paused, frame_idx)

        # Display mouse coordinates when user hovers the image & draw circle with outline
        if img_elem.is_hovered():
            is_clicked, evt = img_elem.read_mouse_xy()
            xynorm_block.set_text(f"({evt.xy_norm.x:.2f}, {evt.xy_norm.y:.2f})")
            xypx_block.set_text(f"({evt.xy_px.x}, {evt.xy_px.y})")
            frame = draw_circle_norm(frame, evt.xy_norm, radius_px=5)
            frame = draw_circle_norm(frame.copy(), evt.xy_norm, radius_px=6, color=(0, 0, 0), thickness=1)

        # Update displayed image & render
        img_elem.set_image(frame)
        display_image = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        pass
    pass
