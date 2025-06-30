#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse

import cv2
import numpy as np

from toadui.video import VideoPlaybackSlider, load_looping_video_or_image, read_webcam_string
from toadui.window import DisplayWindow, KEY
from toadui.buttons import ImmediateButton, ToggleButton
from toadui.sliders import Slider
from toadui.images import DynamicImage
from toadui.carousels import TextCarousel
from toadui.layout import VStack, HStack
from toadui.colormaps import apply_colormap, make_wa_rainbow_colormap
from toadui.static import StaticMessageBar
from toadui.helpers.sampling import make_xy_complex_mesh, resample_with_complex_mesh
from toadui.patterns.misc import draw_grid
import toadui.patterns.truchet as truchet

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set arg defaults
default_input_path = None
default_display_size = 900

# Define script arguments
parser = argparse.ArgumentParser(description="Demo of manipulating a video or image data interactively")
parser.add_argument("-i", "--input_path", default=default_input_path, type=str, help="Path to video or image")
parser.add_argument("-d", "--display_size", default=default_display_size, type=int, help="Initial window size")
parser.add_argument("-s", "--smith_tiles", action="store_true", help="Use smith tiles (when no input provided)")
parser.add_argument("-g", "--grid", action="store_true", help="Use a grid image (when no input provided)")
parser.add_argument("-cam", "--use_webcam", action="store_true", help="Use webcam as video source")

# For convenience
args = parser.parse_args()
input_path = args.input_path if not args.use_webcam else "cam"
display_size = args.display_size
use_smith_tiles = args.smith_tiles
use_grid_image = args.grid

# Handle webcam inputs
is_webcam_source, input_path = read_webcam_string(input_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup input source

# Generate a static image if we're not given an input source
if input_path is None:

    tile_side, tile_count, tile_thick_pct, tile_fg_color = 80, 10, 0.5, (127, 127, 127)
    total_side = tile_count * tile_side
    if use_smith_tiles:
        tiles_set = truchet.make_truchet_tiles_smith(tile_side, tile_thick_pct, tile_fg_color)
        static_image = truchet.draw_truchet(total_side, tiles_set)
    elif use_grid_image:
        grid_side = tile_side * tile_count
        grid_thick = max(1, round(tile_side * tile_thick_pct * 0.5))
        static_image = draw_grid(grid_side, tile_count, tile_fg_color, grid_thick, use_wraparound_sampling=True)
    else:
        tiles_set = truchet.make_truchet_tiles_diagonal(tile_side, tile_thick_pct, tile_fg_color)
        static_image = truchet.draw_truchet(total_side, tiles_set)

    # Use image as input 'path' (reader will properly interpret this)
    input_path = static_image

# Set up 'video reader' for UI visualization, so we can work with continuous stream of 'frames'
# (this works even for static images, they're treated as a video with 1 frame repeating)
is_image_source, vreader = load_looping_video_or_image(input_path, display_size)


# ---------------------------------------------------------------------------------------------------------------------
# %% Warping functions


def linear_of_z(z: ndarray, t: float = 1) -> ndarray:
    ang = t * 2 * np.pi
    return z * (np.cos(ang) + 1j * np.sin(ang)).astype(z.dtype)


def rational_of_z(z: ndarray, t: float = 1) -> ndarray:
    """f(z) = (z - 1) / (z^2 + z + 1), with additional scaling terms"""
    scale = 1 / 2
    zsquared = (z * z) / scale
    numer = z - t * scale
    denom = (zsquared + z) * t + scale
    return numer / denom


def twirl_of_z(z: ndarray, t: float = 1) -> ndarray:
    """
    Warp effect from Unity shader nodes:
    https://docs.unity3d.com/Packages/com.unity.shadergraph@6.9/manual/Twirl-Node.html
    """
    scaled_dist = t * np.absolute(z)
    cos_val = np.cos(scaled_dist)
    sin_val = np.sin(scaled_dist)
    xtwirl = cos_val * z.real - sin_val * z.imag
    ytwirl = sin_val * z.real + cos_val * z.imag
    return z + (xtwirl + ytwirl * 1j)


# Bundle all warp functions together for selection
INV_PI = 1.0 / np.pi
HALF_PI = 0.5 * np.pi
warp_lut = {
    "Linear": linear_of_z,
    "Sine": lambda z, t: np.sin(z * HALF_PI) * t,
    "Cosine": lambda z, t: np.cos(z * HALF_PI) * t,
    "Tangent": lambda z, t: np.tan(z * HALF_PI) * t,
    "Natural Log": lambda z, t: np.log(z) * t * ((vreader.shape[0] + 1) / (200 * np.pi)),
    "Inverse": lambda z, t: np.pow(z, -t),
    "Exponential": lambda z, t: np.exp(z * HALF_PI) * t,
    "Sphere": lambda z, t: z * (1 - t * (z.real**2 + z.imag**2)),
    "Twirl": twirl_of_z,
    "Rational": rational_of_z,
}

# Bundle border fil options together for selection
border_lut = {"Reflect": cv2.BORDER_REFLECT, "Wrap": cv2.BORDER_WRAP, "Blank": cv2.BORDER_CONSTANT}


# ---------------------------------------------------------------------------------------------------------------------
# %% Setup UI

# Define UI elements
img_elem = DynamicImage(vreader.get_sample_frame())
playback_slider = VideoPlaybackSlider(vreader)
border_selector = TextCarousel(border_lut)
warp_selector = TextCarousel(warp_lut, minimum_width=500).set_key("Tangent")
reset_btn = ImmediateButton("Reset XY")
zoom_slider = Slider("Zoom", 0, -2, 2, 0.1, marker_step=0.5, text_scale=0.5)
rval_slider = Slider("Rainbow Speed", 0, 0, 10, 1).set(2, use_as_default_value=False)
rbow_toggle = ToggleButton("Rainbow")
xspeed_slider = Slider("X Speed", 0, -1, 1, 0.01, marker_step=0.5).set(0.25, use_as_default_value=False)
yspeed_slider = Slider("Y Speed", 0, -1, 1, 0.01, marker_step=0.5)
tval_slider = Slider("Effect", 1, -2, 2, 0.01, marker_step=0.25)
msg_bar = StaticMessageBar("Right click sliders to reset", text_scale=0.35, height=30)

# Stack elements together to form layout for display
show_playback_bar = not (is_webcam_source or is_image_source)
ui_layout = VStack(
    img_elem,
    playback_slider if show_playback_bar else None,
    warp_selector,
    tval_slider,
    zoom_slider,
    HStack(xspeed_slider, yspeed_slider, reset_btn, flex=(1, 1, 0.5)),
    HStack(rval_slider, rbow_toggle, flex=(0.8, 0.2)),
    msg_bar,
)

# Set up base xy coordinates for warping
video_hw = vreader.shape[0:2]
square_grid_xy = np.float32((1, 1))
sample_grid_xy = min(video_hw) / np.float32(video_hw)
zgrid_base = make_xy_complex_mesh(max(video_hw), -square_grid_xy, square_grid_xy, use_wraparound_sampling=True)
zgrid = zgrid_base.copy()
warpmap = zgrid_base.copy()

# Build rainbow colormap & pre-compute all rotated versions for animating
rbow_cmap = make_wa_rainbow_colormap()
rbow_2d = np.empty((rbow_cmap.shape[1], 1, rbow_cmap.shape[1], 3), dtype=np.uint8)
for idx in range(rbow_cmap.shape[1]):
    rbow_2d[idx, 0] = np.roll(rbow_cmap, idx, axis=1)


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Display loop ***

# Set up display window and pass UI events to layout object so UI can respond to mouse
fps = min(vreader.get_framerate(), 60)
window = DisplayWindow(display_fps=fps)
window.enable_size_control(display_size)
window.attach_mouse_callbacks(ui_layout)
window.attach_many_keypress_callbacks(
    [
        (" ", vreader.toggle_pause, "Play/pause video") if not is_image_source else None,
        ((",", "."), (tval_slider.decrement, tval_slider.increment), "Adjust effect strength"),
        ((KEY.D_ARROW, KEY.U_ARROW), (zoom_slider.decrement, zoom_slider.increment), "Adjust zoom level"),
        (("z", "x"), (xspeed_slider.decrement, xspeed_slider.increment), "Adjust x speed"),
        (("t", "y"), (yspeed_slider.decrement, yspeed_slider.increment), "Adjust y speed"),
        ((KEY.L_ARROW, KEY.R_ARROW), (warp_selector.prev, warp_selector.next), "Switch warp function"),
        (KEY.TAB, (xspeed_slider.reset, yspeed_slider.reset), "Reset X/Y speed"),
        ("b", border_selector.next, "Switch border mode"),
        ("r", rbow_toggle.toggle, "Toggle rainbow coloring"),
    ]
)
print("", "***** Keypress controls: *****", sep="\n")
window.report_keypress_descriptions(print_directly=True)

# Initialize values used inside of loop
rbow_idx = 0
x_offset, y_offset = 0, 0
xy_speed_scale = 0.35
with window.auto_close(vreader.release):

    # Video loops forever!
    for is_paused, frame_idx, frame in vreader:
        playback_slider.update_state(is_paused, frame_idx)

        # Read controls
        _, _, border_type = border_selector.read()
        is_warp_changed, _, warp_func = warp_selector.read()
        need_xy_reset = reset_btn.read()
        is_zoom_changed, zoom_value = zoom_slider.read()
        _, x_speed = xspeed_slider.read()
        _, y_speed = yspeed_slider.read()
        is_t_changed, t_value = tval_slider.read()
        _, use_rbow_colors = rbow_toggle.read()
        _, r_speed = rval_slider.read()

        # Reset scrolling
        if need_xy_reset:
            x_offset, y_offset = 0, 0
            _, x_speed = xspeed_slider.reset().read()
            _, y_speed = yspeed_slider.reset().read()

        # Re-build base xy sampling grid when zooming in/out
        if is_zoom_changed:
            zoom_amt = 1 / (1 + zoom_value) if zoom_value > 0 else 1 - zoom_value
            zgrid = zgrid_base * zoom_amt

        # Re-build warp map when new function is selected
        if is_t_changed or is_warp_changed or is_zoom_changed:
            warpmap = warp_func(zgrid, t_value)

        # Update 'movement' of warping pattern
        if x_speed != 0 or y_speed != 0:
            dt_scale = window.dt * xy_speed_scale
            x_offset = (x_offset - x_speed * dt_scale) % sample_grid_xy[0]
            y_offset = (y_offset - y_speed * dt_scale) % sample_grid_xy[1]

        # Apply warping!
        xy_offset = (x_offset + y_offset * 1j) * 4
        frame = resample_with_complex_mesh(frame, warpmap + xy_offset, -sample_grid_xy, sample_grid_xy, border_type)

        # Switch to rainbow color map, if needed
        if use_rbow_colors:
            rbow_idx = (rbow_idx + r_speed) % rbow_2d.shape[0]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = apply_colormap(frame, rbow_2d[rbow_idx])

        # Update displayed image & render
        img_elem.set_image(frame)
        display_image = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        pass
    pass
