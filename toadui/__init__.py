# Special check to force x11 use on linux, since opencv doesn't have wayland support
# -> Without this, user is spammed with: qt.qpa.plugin: Could not find the Qt platform plugin "wayland"
import sys
import os

if sys.platform.startswith("linux"):
    is_wayland_platform = any(key in os.environ for key in ["DISPLAY", "WAYLAND_DISPLAY"])
    if is_wayland_platform:
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    pass

from toadui.window import DisplayWindow, KEY
from toadui.video import (
    LoopingVideoReader,
    ImageAsVideoReader,
    ReversibleLoopingVideoReader,
    VideoPlaybackSlider,
    load_looping_video_or_image,
    read_webcam_string,
)
from toadui.cli import (
    HistoryJSON,
    ask_for_media_path,
    ask_for_path_if_missing,
    select_from_options,
)
from toadui.images import DynamicImage, StretchImage, FixedARImage, ZoomImage
from toadui.layout import HStack, VStack, GridStack, OverlayStack, Swapper, HSeparator, VSeparator, Padded
from toadui.carousels import TextCarousel, PathCarousel
from toadui.colormaps import ColormapsBar
from toadui.sliders import Slider, MultiSlider, ColorSlider
from toadui.text import TextBlock, PrefixedTextBlock, TwoLineTextBlock, MessageBar
from toadui.buttons import (
    ToggleButton,
    ToggleImageButton,
    ImmediateButton,
    ImmediateImageButton,
    RadioConstraint,
    RadioBar,
)
from toadui.plots import SimpleHistogramPlot
from toadui.overlays import (
    DrawRectangleOverlay,
    DrawPolygonsOverlay,
    DrawMaskOverlay,
    DrawOutlineOverlay,
    DrawCustomOverlay,
    TextOverlay,
    MousePaintOverlay,
    HoverLabelOverlay,
    PointClickOverlay,
    BoxSelectOverlay,
    EditBoxOverlay,
    GridSelectOverlay,
)

__version__ = "0.1alpha"
