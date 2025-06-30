#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from types import SimpleNamespace


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class UIStyle(SimpleNamespace):
    def copy(self):
        return UIStyle(**self.__dict__)

    def add(self, **kwargs):
        self.__dict__.update(kwargs)
        return self


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def get_background_thickness(foreground_thickness):
    """
    Helper used to choose a 'background' thickness for use in plotting
    lines or text with opencv. The goal being to create an outline effect,
    ideally about 1px around the drawn item. The rules for doing this seem
    to vary slightly with the original thickness
    """
    return foreground_thickness + 1 + (foreground_thickness % 2) if foreground_thickness > 1 else 2
