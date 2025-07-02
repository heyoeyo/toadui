# ToadUI

A UI library with warts! ToadUI is a helper UI library built on top of the basic UI provided by OpenCV. As such, it inherits much of the UI limitations of OpenCV. The goal is to provide ready-to-use UI elements that are not already part of OpenCV (e.g. buttons).

<p align="center">
  <img src="https://github.com/user-attachments/assets/285c9cfa-1a4e-4347-bb1f-1b8bf4a1a242">
</p>

This library is still in early development.


## Who is this for?

ToadUI is intended for quick prototyping and is geared towards handling video (or video-like) visualizations. For example, earlier versions of the library have been used to interactively visualize the internal state of vision transformer models (see experiments from [MuggledDPT](https://github.com/heyoeyo/muggled_dpt/tree/main/experiments) and [MuggledSAM](https://github.com/heyoeyo/muggled_sam/tree/main/experiments)).


### Basic features

- Stacking/tiling style layout. No need to position anything
- Built-in support for basic elements (buttons, toggles, sliders etc.)
- Built-in support for basic video playback/control
- Built-in support for colormapping (useful for data visualization)
- Built-in support for point/box/polygon drawing (e.g. ROI selection)

### Better than XYZ alternative?

This library is not meant for making professional or consumer-facing UIs. It's also poorly suited to anything requiring text input from users, at least for now. It's mainly recommended for those who are already using OpenCV.

If you're looking to make a UI for a _product_, this library probably isn't suitable!


## Usage

ToadUI uses a hybrid system that blends concepts from both [retained mode](https://en.wikipedia.org/wiki/Retained_mode) and [immediate mode](https://en.wikipedia.org/wiki/Immediate_mode_(computer_graphics)) design. As a user of the library, this means you define the structure and layout of your UI elements ahead of time (a bit like HTML), but interactions and rendering are handled in a synchronous, on-demand way. This isn't particularly efficient, but lends itself to code that's easy to write.

### Install

ToadUI is still in early development and is therefore not recommended for use in any serious projects. However, it can be installed using pip or uv if you would like to experiment with it. It can be installed into an existing project using Github:

<details>
<summary>Using pip</summary>

### Install using pip:

After [cloning this repo](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository), make sure to create a [virtual environment](https://docs.python.org/3/library/venv.html), using:
```bash
# For linux or mac:
python3 -m venv .venv
source .venv/bin/activate

# For windows (cmd):
python -m venv .venv
.venv\Scripts\activate
```
And then install in [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) using:
```bash
pip install -e .
```
</details>

<details>
<summary>Using uv</summary>

### Install using uv:

If you prefer to use something like [uv](https://docs.astral.sh/uv/) to install the project, then after cloning the repo you can simply run something like:
```bash
uv run demos/game_of_life.py
```
Which will install the project and run one of the included demos.

</details>


<details>
<summary>Install from Github</summary>

### Installing from github:

This repo can be installed into an existing project by using github:

#### Using pip:
```bash
# Be sure to activate a virtual environment before installing!
pip install git+https://github.com/heyoeyo/toadui
```

#### Using UV:
```bash
uv venv && uv pip install git+https://github.com/heyoeyo/toadui
```
</details>


### Simple Example

The following example creates a UI for a simple video player:

```python
import toadui as tui

vreader = tui.LoopingVideoReader("/path/to/video.mp4")

# Define UI elements
img_elem = tui.DynamicImage()
playback_slider = tui.VideoPlaybackSlider(vreader)
ui_layout = tui.VStack(img_elem, playback_slider)

# Set up display window and attach UI for mouse interactions
window = tui.DisplayWindow(display_fps=vreader.get_framerate())
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callback(" ", vreader.toggle_pause)

while True:
    is_paused, frame_idx, frame = vreader.read()
    playback_slider.update_state(is_paused, frame_idx)

    # Update displayed image & render
    img_elem.set_image(frame)
    display_image = ui_layout.render(h=800)
    req_break, keypress = window.show(display_image)
    if req_break:
        break

# Clean up
vreader.release()
window.close()
```

The result is a simple video player with playback control, it can be paused/unpaused by pressing the spacebar or the included button next to the playback slider.

<p align="center">
  <img src="https://github.com/user-attachments/assets/33ab98ba-71a6-43fb-b247-e66e33b37baa">
</p>