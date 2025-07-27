#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse

import cv2

from toadui.window import DisplayWindow, KEY
from toadui.video import VideoPlaybackSlider, ask_for_path_if_missing, load_looping_video_or_image
from toadui.images import DynamicImage, FixedARImage, ZoomImage
from toadui.text import PrefixedTextBlock
from toadui.layout import VStack, HStack, Swapper
from toadui.static import VSeparator, HSeparator
from toadui.buttons import ImmediateButton, ToggleImageButton
from toadui.sliders import Slider
from toadui.overlays import EditBoxOverlay, DrawRectangleOverlay
from toadui.helpers.images import CropData
from toadui.helpers.icons import draw_lock_icons
from toadui.helpers.pathing import modify_file_path, simplify_path


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
# %% Setup UI elements

# Handle video vs. image inputs
input_path = ask_for_path_if_missing(input_path, path_type="image or video")
is_image_source, vreader = load_looping_video_or_image(input_path)
sample_frame = vreader.get_sample_frame()

# Define UI elements
playback_slider = VideoPlaybackSlider(vreader)
show_playback_bar = not is_image_source

# Define image elements (shared between layouts)
main_img_elem = DynamicImage(sample_frame)
zoom_img_elem = ZoomImage(sample_frame)
crop_img_elem = FixedARImage(aspect_ratio=1)

# Create button/text elements
block_sample_text = " (1234, 1234) "
xy1_text_block = PrefixedTextBlock("Crop XY1: ", max_characters=block_sample_text)
xy2_text_block = PrefixedTextBlock("Crop XY2: ", max_characters=block_sample_text)
hw_text_block = PrefixedTextBlock("Crop HW: ", max_characters=block_sample_text)
stack_text_blocks = HStack(xy1_text_block, xy2_text_block, hw_text_block)
zoom_slider = Slider("Zoom", 0.75, min_val=0, max_val=1, step=0.05, marker_step=0.25, enable_value_display=False)
save_btn = ImmediateButton("Save", color=(125, 185, 0), text_color=(255, 255, 255), text_scale=0.75)

# Create lock/unlock icons for zoom controls
lock_icon, unlockicon = draw_lock_icons(locked_color=(80, 95, 80))
lock_btn = ToggleImageButton(unlockicon, lock_icon)
zoom_slider_and_btn = HStack(zoom_slider, lock_btn, flex=(1, 0))

# Set up overlays for interactions
box_olay = EditBoxOverlay(main_img_elem, color=(0, 255, 0))
zoom_box_olay = DrawRectangleOverlay(zoom_img_elem, color=(0, 255, 0), thickness=1)

# Make (consistently styled) separators
sep_size, sep_color = 8, (40, 40, 40)
vsep = VSeparator(sep_size, sep_color)
hsep = HSeparator(sep_size, sep_color)


# ---------------------------------------------------------------------------------------------------------------------
# %% Build multiple UI layouts

# Define a 'normal' UI layout
normal_stack_side_elem = VStack(zoom_box_olay, zoom_slider_and_btn, vsep, crop_img_elem, save_btn)
main_stack = HStack(box_olay, hsep, normal_stack_side_elem)
normal_layout = VStack(
    stack_text_blocks,
    main_stack,
    playback_slider if show_playback_bar else None,
)

# Define a 'wide' layout for wide input images
wide_zoom_stack = VStack(zoom_box_olay, zoom_slider_and_btn)
wide_crop_stack = VStack(crop_img_elem, save_btn)
wide_stack_side_elem = HStack(wide_zoom_stack, hsep, wide_crop_stack)
wide_stack_side_elem
wide_layout = VStack(
    box_olay,
    playback_slider if show_playback_bar else None,
    stack_text_blocks,
    wide_stack_side_elem,
)

# Define a 'minimal' layout, that doesn't show zoom/crop preview
minimal_layout = VStack(
    HStack(stack_text_blocks, save_btn, flex=(5, 1)),
    box_olay,
    playback_slider if show_playback_bar else None,
)

# Create UI with multiple layouts that can be swapped between
ui_layout = Swapper(normal_layout, minimal_layout, wide_layout)
is_wide_input = (vreader.shape[1] / vreader.shape[0]) > 2
if is_wide_input:
    ui_layout = Swapper(wide_layout, minimal_layout, normal_layout)


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Display loop ***

# Set up display window and attach UI for mouse interactions
window = DisplayWindow(display_fps=min(vreader.get_framerate(), 60))
window.enable_size_control(display_size)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callbacks(
    {
        "Toggle playback": {" ": vreader.toggle_pause} if show_playback_bar else None,
        "Step video backwards/forwards": (
            {",": vreader.prev_frame, ".": vreader.next_frame} if show_playback_bar else None
        ),
        "Change layout": {KEY.TAB: ui_layout.next},
        "Toggle zoom lock": {"z": lock_btn.toggle},
        "Nudge crop region": {
            KEY.L_ARROW: lambda: box_olay.nudge(left=1),
            KEY.R_ARROW: lambda: box_olay.nudge(right=1),
            KEY.U_ARROW: lambda: box_olay.nudge(up=1),
            KEY.D_ARROW: lambda: box_olay.nudge(down=1),
        },
        "Adjust zoom": {
            "[": zoom_slider.decrement,
            "]": zoom_slider.increment,
        },
        "Save cropped image": {"s": save_btn.click},
    },
)
window.report_keypress_descriptions()
print("- Middle click image to lock zoom!", "- Right click to reset crop region", sep="\n")

# Initialize crop data to make sure it's available in the loop
crop_frame = sample_frame.copy()
crop_data = CropData.from_xy1_xy2_norm(vreader.shape, (0, 0), (1, 1))
with window.auto_close(vreader.release):

    for is_paused, frame_idx, frame in vreader:
        playback_slider.update_state(is_paused, frame_idx)

        # We don't need to update displayed images if using a static source
        # (not strictly necessary, slightly more efficient for static images)
        if not is_image_source:
            zoom_img_elem.set_image(frame)
            main_img_elem.set_image(frame)

        # Read controls
        is_box_changed, is_box_valid, cropbox_xy1xy2_norm = box_olay.read()
        is_zoom_slider_changed, zoom_factor = zoom_slider.read()
        is_hovering_main_img = main_img_elem.is_hovered()
        _, is_zoom_locked = lock_btn.read()
        is_save_clicked = save_btn.read()

        # Center zoom on user cursor location, when hovering main image
        if not is_zoom_locked and (is_hovering_main_img or is_box_changed):
            zoom_img_elem.set_zoom_center(main_img_elem.get_event_xy().xy_norm)

        # Adjust zoom in/out amount
        if is_zoom_slider_changed:
            zoom_img_elem.set_zoom_factor(zoom_factor)

        # Update cropped image
        is_playing = not is_paused
        is_crop_changed = is_box_changed and is_box_valid
        if is_crop_changed or is_playing:
            box_olay.set_frame_shape(frame.shape)
            crop_data = CropData.from_xy1_xy2_norm(frame.shape, *cropbox_xy1xy2_norm)
            if crop_data.is_valid():
                crop_frame = crop_data.crop(frame)
                crop_img_elem.set_image(crop_frame)

                # Update crop text reporting
                crop_h, crop_w = crop_frame.shape[0:2]
                xy1_text_block.set_text(f"({crop_data.x1}, {crop_data.y1})")
                xy2_text_block.set_text(f"({crop_data.x2}, {crop_data.y2})")
                hw_text_block.set_text(f"({crop_h}, {crop_w})")

        # Update crop box indicator on zoom image
        if is_crop_changed or zoom_img_elem.is_changed():
            zcrop_xy1_norm, zcrop_xy2_norm = zoom_img_elem.map_full_to_zoom_coordinates(
                crop_data.xy1, crop_data.xy2, input_is_normalized=False, normalize_output=True
            )
            zoom_box_olay.set_rectangles([zcrop_xy1_norm, zcrop_xy2_norm])

        # Update display
        display_image = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        # Handle saving
        if is_save_clicked:
            spacer_str = " " if " " in input_path else "_"
            frame_suffix = f"_{frame_idx}" if show_playback_bar else ""
            crop_suffix = f"({crop_data.x1},{crop_data.y1})-to-({crop_data.x2},{crop_data.y2})"
            save_suffix = f"{spacer_str}{crop_suffix}{frame_suffix}"
            save_path = modify_file_path(input_path, save_suffix, ".png")
            cv2.imwrite(save_path, crop_frame)
            print("", "Saved cropped image:", simplify_path(save_path), sep="\n", flush=True)
        pass
    pass
