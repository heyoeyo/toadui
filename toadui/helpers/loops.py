#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from time import perf_counter

# For type hints
from .types import SelfType


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class TickRateLimiter:
    """
    Simple update limiter, intended for reducing the rate at which something updates.
    A common use would be to prevent some update from occurring on every iteration
    of a loop (e.g. a render loop), and instead have it update every 2 iterations,
    or 3 or 10 etc. Basic usage come from calling .tick():

        rate_limiter = TickRateLimiter(3, update_on_first_call=True)
        for idx in range(10):
            if rate_limiter.tick():
                print(idx, "Tick")
            else:
                print(idx)
        # Will print out ticks on indexes 0, 3, 6 & 9
    """

    def __init__(self, num_ticks_per_update: int = 1, update_on_first_call: bool = True):
        self._num_ticks_per_update = num_ticks_per_update
        self._curr_ticks = num_ticks_per_update if update_on_first_call else 0

    def tick(self) -> bool:
        """
        Count 1 tick and return whether an update should occur
        (When an update is meant to occur, the tick counter resets)
        Returns:
            need_update
        """
        self._curr_ticks += 1
        need_update = self._curr_ticks >= self._num_ticks_per_update
        if need_update:
            self._curr_ticks = 0
        return need_update

    def set_rate(self, num_ticks_per_update: int | float) -> SelfType:
        """Update the tick rate"""
        self._num_ticks_per_update = num_ticks_per_update
        return self

    def set_rate_lerp(self, t0_value: float, t1_value: float, t: float) -> SelfType:
        """
        Helper for setting the tick count based on a variable control
        Sets count according to:
            t0_value * (1-t) + t1_value * t
        Note: There is no clamping on the weighting (t) value!
        """
        self._num_ticks_per_update = t0_value * (1.0 - t) + t1_value * t
        return self

    def force_update(self, update: bool = True) -> SelfType:
        """Helper used to force an update on the next tick"""
        self._curr_ticks = self._num_ticks_per_update
        return self

    def reset(self) -> SelfType:
        """Helper used to reset the tick counter, in order to delay updates"""
        self._curr_ticks = 0
        return self


class FPSLimiter:
    """Simple FPS update limiter"""

    def __init__(self, target_frames_per_second: float = 60.0, update_on_first_call: bool = True):
        self._sec_per_frame = 1.0 / target_frames_per_second
        self._next_update_sec = -1 if update_on_first_call else perf_counter() + self._sec_per_frame
        self._last_tick_sec = perf_counter()

    def tick(self) -> tuple[bool, float]:
        """
        Function used to check whether enough time has passed to require an update
        Returns:
            need_update, delta_t_seconds
        """

        curr_time_sec = perf_counter()
        need_update = curr_time_sec >= self._next_update_sec
        if need_update:
            num_updates = 1 + (curr_time_sec - self._next_update_sec) // self._sec_per_frame
            self._next_update_sec += num_updates * self._sec_per_frame

        # Keep track of how much time has passed
        delta_t_sec = curr_time_sec - self._last_tick_sec
        self._last_tick_sec = curr_time_sec

        return need_update, delta_t_sec

    def set_rate(self, frames_per_second: float) -> SelfType:
        self._sec_per_frame = 1.0 / max(1, frames_per_second)
        return self

    def set_rate_lerp(self, t0_value: float, t1_value: float, t: float) -> SelfType:
        """
        Helper for setting the frame rate based on a variable control
        using linear interpolation between two value.
        Sets fps according to:
            t0_value * (1-t) + t1_value * t
        Note: There is no clamping on the weighting (t) value!
        """

        target_fps = t0_value * (1.0 - t) + t1_value * t
        self._sec_per_frame = 1.0 / max(1, target_fps)
        return self

    def __iter__(self) -> SelfType:
        return self

    # .................................................................................................................

    def __next__(self) -> tuple[bool, float]:
        """
        Iterator that 'ticks' the FPS counter.
        Returns: need_update, delta_t_seconds
        (when appropriate amount of time has passed, based on target frame rate)
        """

        return self.tick()


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions
