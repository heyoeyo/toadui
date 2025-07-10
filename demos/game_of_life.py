#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse

import cv2
import numpy as np

import toadui.patterns.game_of_life as gol
from toadui.window import DisplayWindow, KEY
from toadui.sliders import Slider
from toadui.images import DynamicImage
from toadui.carousels import TextCarousel
from toadui.layout import VStack, HStack
from toadui.buttons import ImmediateButton, ToggleButton
from toadui.text import TextBlock
from toadui.overlays import DrawMaskOverlay
from toadui.colormaps import ColormapsBar, make_tree_colormap, load_colormap
from toadui.helpers.images import get_image_hw_for_max_side_length
from toadui.helpers.loops import TickRateLimiter
from toadui.helpers.data_management import ValueChangeTracker

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set arg defaults
default_initial_size = 256
default_max_size = 1024
default_aspect_ratio = 1
default_seed = 1186804540
default_display_size = 900
default_framerate = 60
default_image_path = None
default_cmap_path = None

# Define script arguments
parser = argparse.ArgumentParser(description="Demo of Conway's game of life")
parser.add_argument("-s", "--initial_height", default=default_initial_size, type=int, help="Grid height on load")
parser.add_argument("-x", "--max_height", default=default_max_size, type=int, help="Maximum allowable grid height")
parser.add_argument("-a", "--aspect_ratio", default=default_aspect_ratio, type=float, help="Grid aspect ratio")
parser.add_argument("-e", "--seed", default=default_seed, help="Initial random seed")
parser.add_argument("-d", "--display_size", default=default_display_size, type=int, help="Initial window size")
parser.add_argument("-r", "--framerate", default=default_framerate, type=float, help="Display framerate")
parser.add_argument("-n", "--no_clear", action="store_true", help="Disable auto-clear when adding noise/tiling")
parser.add_argument("-i", "--image_path", default=default_image_path, type=str, help="Path to a custom image")
parser.add_argument("-c", "--cmap_path", default=default_cmap_path, type=str, help="Path to a custom colormap image")
parser.add_argument("-p", "--pause", action="store_true", help="Pause on start-up")

# For convenience
args = parser.parse_args()
init_height = args.initial_height
MAX_H = args.max_height
ASPECTRATIO = args.aspect_ratio
init_seed = args.seed
init_display_size = args.display_size
DISPLAY_FPS = args.framerate
AUTO_CLEAR = not args.no_clear
input_image_path = args.image_path
input_colormap_path = args.cmap_path
play_on_start = not args.pause


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Load custom image (and convert to binary pattern for game of life), if provided
loaded_image = None
if input_image_path is not None:
    loaded_image = cv2.imread(input_image_path)
    assert loaded_image is not None, f"Error reading input image: {input_image_path}"

    # Force to grayscale
    if loaded_image.ndim == 3:
        loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)

    # Downsize the loaded image if needed
    loaded_h, loaded_w = loaded_image.shape[0:2]
    target_size = max(MAX_H, MAX_H * ASPECTRATIO) // 2
    if max(loaded_h, loaded_w) > target_size:
        scale_h, scale_w = get_image_hw_for_max_side_length(loaded_image.shape, target_size)
        loaded_image = cv2.resize(loaded_image, (scale_w, scale_h), interpolation=cv2.INTER_NEAREST_EXACT)
    loaded_image = np.uint8(loaded_image > np.mean(loaded_image))

# Load custom colormap, if provided
loaded_cmap = None
if input_colormap_path is not None:
    ok_load, loaded_cmap = load_colormap(input_colormap_path)
    assert ok_load, f"Error reading input colormap: {input_colormap_path}"


# ---------------------------------------------------------------------------------------------------------------------
# %% Helper functions


def make_tiled_pattern(pattern: ndarray, tiled_hw: tuple[int, int], pad_factor: float = 0.5) -> ndarray:
    """Helper used to tile a (padded) pattern to a target size"""

    # Create padded pattern for tiling (usually more interesting with space between tiles)
    ph, pw = [round(pad_factor * side) for side in pattern.shape[0:2]]
    padded_pattern = cv2.copyMakeBorder(pattern, ph, ph, pw, pw, cv2.BORDER_CONSTANT)

    # Figure out how many tiles we need
    targ_h, targ_w = tiled_hw
    tile_h, tile_w = padded_pattern.shape[0:2]
    num_x_tiles = max(1, targ_w // tile_w)
    num_y_tiles = max(1, targ_h // tile_h)

    return np.tile(padded_pattern, (num_y_tiles, num_x_tiles))


def make_noise_pattern(noise_hw: tuple[int, int], seed: int | None = None, reduction_factor: int = 5) -> ndarray:
    """Helper used to make a binary noise pattern with fewer 1's than 0's"""

    reduction_factor = max(2, reduction_factor)
    noise_gen = np.random.default_rng(seed)
    return noise_gen.integers(0, reduction_factor, noise_hw, dtype=np.uint8) // (reduction_factor - 1)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up game of life & patterns

pattern_lut = {
    "Acorn": gol.make_acorn,
    "Bunnies": gol.make_bunnies,
    "Lidka": gol.make_lidka,
    "R-Pentamino": gol.make_r_pentamino,
    "Rabbits": gol.make_rabbits,
    "Queen Bee": gol.make_queen_bee,
    "44P5H2V0": gol.make_44P5H2V0,
    "Ark": gol.make_ark,
    "Glider": gol.make_glider,
    "Gosper Gun": gol.make_gosper_gun,
    "Achim's P16": gol.make_achims_p16,
    "Penta-decathlon": gol.make_penta_decathlon,
    "Pulsar": gol.make_pulsar,
    "Toad": gol.make_toad,
    "Point": lambda: np.ones((1, 1), dtype=np.uint8),
}


life = gol.HeatmapOfLife(init_height, init_height * ASPECTRATIO)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up UI

# Define UI elements
cmap_bar = ColormapsBar(cv2.COLORMAP_INFERNO, make_tree_colormap(), None)
img_elem = DynamicImage(life.get_heatmap_image(), resize_interpolation=cv2.INTER_NEAREST)
pattern_selector = TextCarousel(pattern_lut)
count_blk = TextBlock(max_characters=10)
noise_btn, tile_btn, rot_btn = ImmediateButton.many("Noise", "Tile", "Rotate")
size_slider = Slider("Size", init_height, 32, MAX_H, 32, marker_step=256)
speed_slider = Slider("Speed", 70, 0, 100, 1, marker_step=50)
weight_slider = Slider("Heatmap", 0.8, 0, 1, 0.01, marker_step=0.1)
play_btn = ToggleButton("Play", play_on_start, (80, 225, 40), text_color_on=(255, 255, 255))
step_btn = ImmediateButton("Step", (225, 120, 45), text_color=(255, 255, 255))
clear_btn = ImmediateButton("Clear", (85, 45, 225), text_color=(255, 255, 255))

# Set up image mouse interaction
mask_olay = DrawMaskOverlay(img_elem, scaling_interpolation=cv2.INTER_NEAREST)

# Stack elements together to form layout for display
ui_layout = VStack(
    HStack(cmap_bar, count_blk, min_w=300),
    mask_olay,
    pattern_selector,
    HStack(noise_btn, tile_btn, rot_btn),
    size_slider,
    speed_slider,
    weight_slider,
    HStack(play_btn, step_btn, clear_btn),
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set initial state

# Add loaded colormap to UI, if provided
if loaded_cmap is not None:
    cmap_bar.add_colormap(loaded_cmap, 0)

# Set initial state of cells (use loaded image if provided, otherwise use noise)
if loaded_image is None:
    noise_btn.click()
else:
    # Add loaded image to shape listing so user can click to add it
    pattern_selector.add_entry(("Custom", lambda: loaded_image), 0, set_to_new_entry=True)

    # Use tiling to set initial cell state to loaded image
    loaded_size = loaded_image.shape[0] * 2
    if loaded_size > init_height:
        size_slider.set(loaded_image.shape[0] * 2)
    tile_btn.click()

# Random seed used for (reproducable) noise patterns
rng_seed = np.random.randint(0, 2**31) if init_seed is None else int(init_seed)

# Variables to help adjust per-loop update speed
num_updates_per_frame = 1
loop_limiter = TickRateLimiter()

# Variable for holding currently selected pattern
mask_life = np.zeros(life.get_hw(), dtype=np.uint8)
curr_pattern_func = gol.make_acorn
curr_pattern = curr_pattern_func()
hover_changed = ValueChangeTracker(False)


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Main Loop ***

# Set up display window and pass UI events to layout object so UI can respond to mouse
window = DisplayWindow(display_fps=DISPLAY_FPS)
window.enable_size_control(init_display_size)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callbacks(
    {
        "Change placeable pattern": {KEY.L_ARROW: pattern_selector.prev, KEY.R_ARROW: pattern_selector.next},
        "Adjust grid size": {KEY.D_ARROW: size_slider.decrement, KEY.U_ARROW: size_slider.increment},
        "Cycle colormapping": {KEY.TAB: cmap_bar.next},
        "Play/Pause": {" ": play_btn.toggle},
        "Advance one step": {"s": step_btn.click},
        "Reset to new noise state": {"n": noise_btn.click},
        "Tile current selected shape": {"t": tile_btn.click},
        "Rotate 90 degrees clockwise": {"r": rot_btn.click},
        "Clear entire state": {"c": clear_btn.click},
    }
)

window.report_keypress_descriptions()

with window.auto_close():

    while True:

        # Clear cells & pause (do this before anything else, so changes take effect on this frame!)
        if clear_btn.read():
            play_btn.toggle(False)
            life.clear()

        # Read controls
        _, is_playing = play_btn.read()
        is_shape_changed, _, curr_pattern_func = pattern_selector.read()
        need_step = step_btn.read()
        is_size_changed, size_value = size_slider.read()
        is_speed_changed, speed_value = speed_slider.read()
        is_weight_changed, heatweight_value = weight_slider.read()
        need_noise = noise_btn.read()
        need_tile = tile_btn.read()
        need_rotation = rot_btn.read()

        # Force pause if user steps
        if need_step:
            play_btn.toggle(False)

        # Update pre-computed pattern if needed
        if is_shape_changed:
            curr_pattern = curr_pattern_func()

        # Alter the cell grid size as well as the mask overlay size
        if is_size_changed:
            new_h, new_w = size_value, int(round(size_value * ASPECTRATIO))
            life.set_new_size(new_h, new_w)
            mask_life = np.zeros(life.get_hw(), dtype=np.uint8)

        # Clear mask whenever we hover off the image (avoids repeatedly clearing)
        is_img_hover = img_elem.is_hovered()
        if hover_changed(is_img_hover) and not is_img_hover:
            mask_olay.clear()

        # Draw a pattern indicator when hovering the display (and add to GoL on click)
        if is_img_hover:

            # Update cells with shape if clicked
            is_img_clicked, img_mouse_xy = img_elem.read_mouse_xy()
            if is_img_clicked:
                life.place_pattern(curr_pattern, img_mouse_xy.xy_norm, use_xor=True, clear_existing=False)

            # Draw mask indicator to show placement of selected pattern
            if is_shape_changed or is_size_changed or window.mouse.is_moved():
                mask_life.fill(0)
                mask_life = gol.place_pattern(mask_life, curr_pattern, img_mouse_xy.xy_norm)
                mask_olay.set_mask(mask_life)

        if need_rotation:
            life.rotate()

        if is_weight_changed:
            life.set_heatmap_weight(heatweight_value)

        if need_noise:
            print("Noise seed:", rng_seed)
            noise_hw = [val // 2 for val in life.get_hw()]
            noise_pattern = make_noise_pattern(noise_hw, rng_seed)
            rng_seed = (rng_seed + 1) % (2**31)
            life.place_pattern(noise_pattern, clear_existing=AUTO_CLEAR)

        if need_tile:
            tiled_hw = [val // 2 for val in life.get_hw()]
            tiled_pattern = make_tiled_pattern(curr_pattern, tiled_hw)
            life.place_pattern(tiled_pattern, clear_existing=AUTO_CLEAR)

        if is_speed_changed:
            speed_norm = speed_value / 50.0
            loop_limiter.set_rate_lerp(DISPLAY_FPS, 0, speed_norm)
            num_updates_per_frame = max(1, round((speed_norm - 1) * 20))

        # Do life updates
        need_update = loop_limiter.tick()
        if (is_playing and need_update) or need_step:
            num_steps = 1 if need_step else num_updates_per_frame
            for _ in range(num_steps):
                life.step()
        count_blk.set_text(life.total_iters)

        # Update visuals and display
        hmap_image = cmap_bar.apply_colormap(life.get_heatmap_image())
        img_elem.set_image(hmap_image)
        display_image = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_image)
        if req_break:
            break
