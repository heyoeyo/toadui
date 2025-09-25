#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse
import datetime as dt
from pathlib import Path

import cv2
import numpy as np

from toadui.window import DisplayWindow, KEY
from toadui.cli import ask_for_path_if_missing
from toadui.images import FixedARImage
from toadui.text import TwoLineTextBlock
from toadui.sliders import Slider
from toadui.buttons import ImmediateButton, ToggleButton, ImmediateImageButton
from toadui.layout import VStack, HStack
from toadui.static import VSeparator
from toadui.carousels import PathCarousel
from toadui.helpers.icons import draw_rotating_arrow_icons
from toadui.helpers.pathing import save_path_counter, modify_file_path, simplify_path


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set arg defaults
default_input_path = None
default_display_size = 900

# Define script arguments
parser = argparse.ArgumentParser(description="Demo of a simple image viewer")
parser.add_argument("-i", "--input_path", default=default_input_path, type=str, help="Path to image or folder")
parser.add_argument("-d", "--display_size", default=default_display_size, type=int, help="Initial window size")
parser.add_argument("-png", action="store_true", help="Always save as png (lossless)")
parser.add_argument("-n", "--no_save", action="store_true", help="Disable saving")
parser.add_argument("-s", "--single", action="store_true", help="Don't check parent folder when loading a single image")

# For convenience
args = parser.parse_args()
input_path = args.input_path
display_size = args.display_size
save_as_png = args.png
enable_save = not args.no_save
search_parent_folder = not args.single


# ---------------------------------------------------------------------------------------------------------------------
# %% Helper functions


def get_image_wxh_string(image_shape) -> str:
    """Helper used to report image sizing"""
    return f"{image_shape[1]} x {image_shape[0]}"


def get_file_info_strings(file_stats, dt_format="%Y/%m/%d %H:%M:%S %p") -> tuple[str, str, str]:
    """
    Helper used to report file size/timestamps.
    Returns: time_create_string, time_modified_str, size_string
    """

    # For convenience
    size_bytes = file_stats.st_size
    created_timestamp = file_stats.st_ctime
    modified_timestamp = file_stats.st_mtime

    # Create time strings
    created_str = dt.datetime.fromtimestamp(created_timestamp).strftime(dt_format)
    modified_str = dt.datetime.fromtimestamp(modified_timestamp).strftime(dt_format)

    # Figure out file sizing reporting string
    size_suffix_list = ("B", "KB", "MB", "GB", "TB", "PB")
    size_power1000 = int(np.log10(max(1, size_bytes)) // 3)
    size_power1000 = min(size_power1000, len(size_suffix_list) - 1)
    size_report = size_bytes / (1000**size_power1000)
    size_str = f"{size_report:.3g} {size_suffix_list[size_power1000]}"

    return created_str, modified_str, size_str


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup UI

# Get pathing to image or folder of images
input_path = ask_for_path_if_missing(input_path, path_type="image or folder", allow_folders=True)

# Switch to parent folder pathing, if needed
input_path = Path(input_path)
initial_file_select = None
if search_parent_folder and input_path.is_file():
    initial_file_select = input_path.name
    input_path = input_path.parent

# Define main UI elements
imgsize_slider = Slider("Image Size (%)", 100, 0, 100, step=1, marker_step=25)
img_elem = FixedARImage(aspect_ratio=1, min_side_length=256)
file_selector = PathCarousel(input_path, height=50)
if initial_file_select is not None:
    file_selector.set_key(initial_file_select)

# Create reporting blocks
created_txt_block = TwoLineTextBlock("Created:", "00/00/0000 00:00:00 AM")
modified_txt_block = TwoLineTextBlock("Modified:", "00/00/0000 00:00:00 AM")
wh_txt_block = TwoLineTextBlock("Dimensions (WxH):", "100x100")
size_txt_block = TwoLineTextBlock("Size:", "1MB")
edit_wh_txt_block = TwoLineTextBlock("Edited (WxH):", "100x100")

# Create editing controls
l_rotarrow, r_rotarrow = draw_rotating_arrow_icons(color_bg=(40, 40, 40), scale_norm=0.8, side_length_px=50)
rot_left_btn = ImmediateImageButton(l_rotarrow)
rot_right_btn = ImmediateImageButton(r_rotarrow)
save_btn = ImmediateButton("  Save  ", (40, 10, 160))
nearest_interp_btn = ToggleButton("Use nearest interpolation")

# Build full UI
show_file_select = len(file_selector) > 1
ui_layout = VStack(
    imgsize_slider,
    HStack(
        img_elem,
        VStack(
            created_txt_block,
            modified_txt_block,
            wh_txt_block,
            size_txt_block,
            VSeparator(8, color=(30, 25, 20), is_flexible_h=True),
            edit_wh_txt_block,
            HStack(rot_left_btn, rot_right_btn),
            save_btn if enable_save else None,
        ),
        flex=(1, 0),
    ),
    file_selector if show_file_select else None,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Display loop ***

# Set up display window and attach UI for mouse interactions
window = DisplayWindow(display_fps=60)
window.enable_size_control(display_size, minimum=ui_layout.get_min_hw().h)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callbacks(
    {
        "Cycle images": {KEY.L_ARROW: file_selector.prev, KEY.R_ARROW: file_selector.next},
        "Rotate image": {KEY.U_ARROW: rot_left_btn.click, KEY.D_ARROW: rot_right_btn.click},
        "Adjust image scaling": {"[": imgsize_slider.decrement, "]": imgsize_slider.increment},
        "Toggle scaling interpolation": {"i": nearest_interp_btn.toggle},
        "Save image": {"s": save_btn.click} if enable_save else None,
    }
).report_keypress_descriptions()

# Set initial initial image state
src_frame = np.zeros((window.size, window.size, 3), dtype=np.uint8)
scale_frame = src_frame
num_rots = 0

with window.auto_close():

    while True:

        # Switch image inputs
        is_file_changed, file_name, file_path = file_selector.read()
        if is_file_changed:
            # Remove folders & non-image files from carousel
            if file_path.is_dir():
                file_selector.remove_current_entry()
                continue
            src_frame = cv2.imread(file_path)
            if src_frame is None:
                print("Unable to read file:", file_name)
                file_selector.remove_current_entry()
                continue

            # Update file reporting
            filecreate_str, filemod_str, filesize_str = get_file_info_strings(file_path.stat())
            created_txt_block.set_text(filecreate_str)
            modified_txt_block.set_text(filemod_str)
            size_txt_block.set_text(filesize_str)
            wh_txt_block.set_text(get_image_wxh_string(src_frame.shape))

            # Force reset or re-read of controls
            num_rots = 0
            imgsize_slider.set_is_changed()
            nearest_interp_btn.set_is_changed()

        # Read controls
        is_imgsize_changed, img_size_pct = imgsize_slider.read()
        is_imginterp_changed, use_nearest_interp = nearest_interp_btn.read()
        is_rotleft = rot_left_btn.read()
        is_rotright = rot_right_btn.read()
        is_save_clicked = save_btn.read()

        # Handle down-scaling
        need_scale_changed = is_imgsize_changed or is_imginterp_changed
        if need_scale_changed:
            scale_frame = src_frame.copy()
            img_interp = cv2.INTER_NEAREST_EXACT if use_nearest_interp else cv2.INTER_AREA
            if img_size_pct < 100:
                min_img_scale = 3 / min(src_frame.shape[0:2])
                scale_norm = max(min_img_scale, img_size_pct / 100)
                scale_frame = cv2.resize(src_frame, dsize=None, fx=scale_norm, fy=scale_norm, interpolation=img_interp)

            # Update display element to match frame interpolation, so we aren't mixing effects
            img_elem.style.interpolation = img_interp

            # Update reported scaled sizing
            edit_wh_txt_block.set_text(get_image_wxh_string(scale_frame.shape))
            edit_frame = scale_frame

        # Handle rotations (applied after scaling!)
        if is_rotleft or is_rotright or need_scale_changed:
            num_rots = (num_rots + int(is_rotleft) - int(is_rotright)) % 4
            edit_frame = np.rot90(scale_frame, num_rots)
            edit_wh_txt_block.set_text(get_image_wxh_string(edit_frame.shape))

        # Update displayed image & render
        img_elem.set_image(edit_frame)
        display_image = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        # Handle saving
        if is_save_clicked:
            save_path = modify_file_path(file_path, "_edit", new_file_extension=".png" if save_as_png else None)
            save_path = save_path_counter(save_path)
            cv2.imwrite(save_path, edit_frame)
            print("", "Saved image:", simplify_path(save_path), sep="\n", flush=True)

        pass
    pass
