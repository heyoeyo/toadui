#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from enum import IntEnum
from time import perf_counter

import cv2
import numpy as np

from toadui.helpers.window import WindowContextManager, WindowTrackbar, CallbackSequencer, MouseEventsCallback


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class DisplayWindow:
    """
    Class used to manage opencv window, mostly to make trackbars & callbacks easier to organize.
    The most recent mouse events can be accessed from: window.mouse (e.g. window.mouse.xy)
    """

    WINDOW_CLOSE_KEYS_SET = {ord("q"), 27}  # q, esc

    def __init__(self, window_title="Display - esc to close", display_fps=60):

        # Clear any existing window with the same title
        # -> This forces the window to 'pop-up' when initialized, in case a 'dead' window was still around
        # -> Without this, rendering will resume inside the existing window, which remains hidden
        try:
            cv2.destroyWindow(window_title)
        except cv2.error:
            pass

        # Store window state
        self.title = window_title
        self._frame_delay_ms = 1000 // display_fps
        self._last_display_sec = -self._frame_delay_ms
        self.dt = 1.0 / display_fps
        self.size = None

        # Allocate variables for use of callbacks
        self._keypress_callbacks_dict: dict[int, callable] = {}
        self._keypress_descriptions = []

        # Fill in blank image to begin (otherwise errors before first image can cause UI to freeze!)
        cv2.namedWindow(self.title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        self.show(np.zeros((50, 50, 3), dtype=np.uint8), 100)

        # Set up callbacks on the window
        self.mouse = MouseEventsCallback()
        self._mouse_cbs = CallbackSequencer(self.mouse)
        cv2.setMouseCallback(self.title, self._mouse_cbs)

    def __repr__(self):
        return f"{self.title}  |  event: {self.mouse.event}, flags: {self.mouse.flags}"

    def move(self, x=None, y=None):
        """Wrapper around cv2.moveWindow. If None is given for x or y, will try to re-use existing value"""
        if x is None or y is None:
            old_x, old_y, _, _ = self.get_xywh()
            x = old_x if x is None else x
            y = old_y if y is None else y
        cv2.moveWindow(self.title, x, y)
        return self

    def get_xywh(self) -> tuple[int | None, int | None, int | None, int | None]:
        """
        Wrapper around cv2.getWindowImageRect, with exception handling.
        On success, returns:
            (x, y, width, height)
        On failure (e.g. window is closed), returns None for all values
        """
        try:
            x, y, w, h = cv2.getWindowImageRect(self.title)
        except cv2.error:
            x, y, w, h = None, None, None, None
        return x, y, w, h

    def add_trackbar(self, trackbar_name: str, max_value: int, initial_value: int = 0) -> WindowTrackbar:
        """
        Add built-in (opencv) trackbar to the window
        - These have limited capability and may render inconsistently across platforms
        - Not recommended, except for simple experimentation
        - Use a slider UI element instead
        """
        return WindowTrackbar(self.title, trackbar_name, max_value, initial_value)

    def attach_mouse_callbacks(self, *callbacks):
        """
        Attach callbacks for handling mouse events
        Callback functions should have a call signature as folows:

            def callback(event: int, x: int, y: int, flags: int, params: Any) -> None:

                # Example to handle left-button down event
                if event == EVENT_LBUTTONDOWN:
                    print("Mouse xy:", x, y)

                return
        """

        # Sanity check. Make sure we're given callbacks
        # -> We assume we're given objects with a ._on_opencv_event(...) function first
        # -> Otherwise assume we're given a direct function/callable
        actual_cbs = []
        for cb in callbacks:
            if hasattr(cb, "_on_opencv_event"):
                actual_cbs.append(cb._on_opencv_event)
            elif callable(cb):
                actual_cbs.append(cb)
            else:
                print(f"Invalid callback, cannot attach to window ({self.title}):\n{cb}")

        self._mouse_cbs.add(*actual_cbs)
        return self

    def attach_keypress_callback(self, keycodes: tuple[int | str] | int | str, callbacks, description=None):
        """
        Attach a callback for handling a keypress event
        Keycodes can be given as strings (i.e. the actual key, like 'a') or for
        keys that don't have simple string representations (e.g. the Enter key),
        the raw keycode integer can be given. To figure out what these are,
        print out the window keypress result while pressing the desired key!

        Callbacks should have no input arguments and no return values.

        Multi-keycode & multi-callbacks are supported as inputs:
            1. One keycode, one callback. Normal usage
            2. One keycode, many callbacks (e.g. one key triggers all callbacks)
            3. One callback, many keycodes (e.g. many keys trigger the same callback)
            4. Many keys, manycallbacks. Must be matching lengths. Allows for using a
               single description for multiple keys. Useful for key pairs used for
               adjusting quantities (e.g. decrement/increment pairs)
        """

        # Force key/callback to be lists, for handling more general scenario
        kcs_iter = keycodes if isinstance(keycodes, (tuple, list)) else [keycodes]
        cbs_iter = callbacks if isinstance(callbacks, (tuple, list)) else [callbacks]

        # Duplicate keycodes or callbacks to get matching lengths
        num_kcs, num_cbs = len(kcs_iter), len(cbs_iter)
        one_key_many_callbacks = num_kcs == 1 and num_cbs > 1
        one_callback_many_keys = num_kcs > 1 and num_cbs == 1
        if one_key_many_callbacks:
            cbs_iter = [_many_callbacks_to_one(*cbs_iter)]
        elif one_callback_many_keys:
            cbs_iter = [cbs_iter[0]] * len(kcs_iter)
        else:
            assert num_kcs == num_cbs, f"Error, mismatching keycode ({num_kcs}) & callback ({num_cbs}) lengths!"

        # Record callbacks and generate 'nice' key names for reporting
        nice_keynames = []
        for kcode, cb in zip(kcs_iter, cbs_iter):
            if isinstance(kcode, str):
                kcode = ord(kcode.lower())
            self._keypress_callbacks_dict[kcode] = cb
            is_known_key, name = KEY.get_value_name(kcode)
            nice_keynames.append(name if is_known_key else chr(kcode))

        # Try to record a print-friendly key name
        str_joiner = " or " if one_callback_many_keys else ("/" if "/" not in nice_keynames else " ")
        desc_keyname = str_joiner.join(nice_keynames)
        self._keypress_descriptions.append((desc_keyname, "no description" if description is None else description))

        return self

    def attach_many_keypress_callbacks(self, keycode_callback_desc_list: list | tuple):
        """
        Helper used to attach multiple keypress/callback/descriptions at the same time.
        Equivalent to running 'attach_keypress_callback' in a loop.
        Each entry should be encoded as: [keycode, callback_function, description (optional)]

        If 'None' is given, then the entry will be skipped. This can be useful for
        conditonally disabling entries in the list.

        Returns self
        """

        for entry in keycode_callback_desc_list:
            if entry is not None:
                self.attach_keypress_callback(*entry)

        return self

    def report_keypress_descriptions(self, print_directly=False) -> [list]:
        """Helper used to print out a list of all keypress callback descriptions"""
        if print_directly:
            strs_to_print = [f"{name}: {desc}" for name, desc in self._keypress_descriptions if desc is not None]
            print(*strs_to_print, sep="\n", flush=True)
        return self._keypress_descriptions

    def run_keypress_callbacks(self, keypress):
        """
        Helper used to run any attached keypress callbacks. This happens automatically
        when calling window.show(...). The only reason to use this function is if
        manually using cv2.imshow(...) & cv2.waitKey(...)
        """
        for cb_keycode, cb in self._keypress_callbacks_dict.items():
            if keypress == cb_keycode:
                cb()
        return self

    def show(self, image, frame_delay_ms=None) -> [bool, int]:
        """
        Function which combines both opencv functions: 'imshow' and 'waitKey'
        This is meant as a convenience function in cases where only a single window is being displayed.
        If more than one window is displayed, it is better to use 'imshow' and 'waitKey' separately,
        so that 'waitKey' is only called once!
        Returns:
            request_close, keypress
        """

        # Figure out frame delay (to achieve target FPS) if we're not given one
        if frame_delay_ms is None:
            time_elapsed_ms = round(1000 * (perf_counter() - self._last_display_sec))
            frame_delay_ms = max(self._frame_delay_ms - time_elapsed_ms, 1)

        cv2.imshow(self.title, image)
        keypress = cv2.waitKey(int(frame_delay_ms)) & 0xFF
        curr_time_sec = perf_counter()
        self.dt, self._last_display_sec = curr_time_sec - self._last_display_sec, curr_time_sec
        # self._last_display_sec = perf_counter()

        self.run_keypress_callbacks(keypress)
        request_close = keypress in self.WINDOW_CLOSE_KEYS_SET

        return request_close, keypress

    def imshow(self, image):
        """
        Wrapper around opencv imshow, fills in 'winname' with the window title.
        Doesn't include any of the additional checks/features of using .show(...)
        """
        cv2.imshow(self.title, image)
        return self

    @classmethod
    def waitKey(cls, frame_delay_ms=1) -> [bool, int]:
        """
        Wrapper around opencv waitkey (triggers draw to screen)
        Returns:
            request_close, keypress
        """

        keypress = cv2.waitKey(int(frame_delay_ms)) & 0xFF
        request_close = keypress in cls.WINDOW_CLOSE_KEYS_SET
        return request_close, keypress

    def close(self) -> bool:
        """Close window. Returns: was_open"""
        was_open = False
        try:
            cv2.destroyWindow(self.title)
            was_open = True
        except cv2.error:
            pass

        return was_open

    def auto_close(self, *clean_up_functions) -> WindowContextManager:
        """
        Context manager for auto-closing a window when finished.
        Accepts callback functions, which will be executed after the window is closed,
        even if an error occurs. This acts a bit like the 'finally' block of
        a try/except statement.

        Example usage:
            window = DisplayWindow("My Window")
            vreader = make_video_reader(...)
            # ... other setup code ...

            with window.auto_close(vreader.close):
                for frame in video:
                    window.show(frame)

            # ... Window & vreader will be closed ...
        """
        return WindowContextManager(self.title, *clean_up_functions)

    def enable_size_control(
        self,
        initial_size=900,
        minimum=350,
        maximum=4000,
        step=50,
        decrement_key="-",
        increment_key="=",
    ):
        """
        For convenience, enables use of the .size attribute on the window for render sizing.
        Also adds keypress callbacks for adjusting the sizing.

        Note that this alone does not auto-handle sizing!
        This is intended to be used to set sizing when rendering
        a final display, for example:

            window.enable_size_control(...)
            # ...
            # During render loop:
            ui_result.render(h=window.size)

        This is a bit of an ugly hack to help with a common use-case.
        For more involved size control, it is much better to manually
        manage the size variable (e.g. outside of the window object!).
        """

        initial_size, minimum, maximum, step = [int(val) for val in (initial_size, minimum, maximum, step)]
        self.size = initial_size

        def increase_size():
            self.size = min(self.size + step, maximum)

        def decrease_size():
            self.size = max(self.size - step, minimum)

        self.attach_keypress_callback(
            (decrement_key, increment_key), (decrease_size, increase_size), "Adjust window size"
        )

        return self


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def _many_callbacks_to_one(*callbacks) -> None:
    """
    Helper used to create a 'single' callback function which
    itself calls multiple callbacks in sequence
    """

    def _one_callback():
        for cb in callbacks:
            cb()

    return _one_callback


# ---------------------------------------------------------------------------------------------------------------------
# %% Define window key codes


class KEY(IntEnum):

    L_ARROW = 81
    U_ARROW = 82
    R_ARROW = 83
    D_ARROW = 84

    ESC = 27
    ENTER = 13
    BACKSPACE = 8
    SPACEBAR = ord(" ")
    TAB = ord("\t")

    SHIFT = 225
    ALT = 233
    CAPSLOCK = 229
    # CTRL = None # No key code for this one surprisingly!?

    @classmethod
    def get_value_name(cls, value):
        contains_value = value in tuple(cls)
        name = cls(value).name if contains_value else None
        return contains_value, name
