#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
from pathlib import Path


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def modify_file_path(
    original_file_path: str | Path,
    filename_suffix: str | None = "_modified",
    new_file_extension: str | None = None,
    new_folder_path: str | None = None,
) -> str | Path:
    """
    Helper used to build a modified version of a file path.
    - Supports addition of a filename suffix, eg: /path/to/file.mp4 -> /path/to/file_suffix.mp4
    - Supports modifying the file extension
    - Supports changing the parent folder path

    Returns:
        modified_file_path
    """

    # Sanity check. Make sure we're dealing with a pathlib object
    assert isinstance(original_file_path, (str, Path)), f"Must provide string or Path! Got: {type(original_file_path)}"
    is_input_str = isinstance(original_file_path, str)
    original_file_path: Path = Path(original_file_path) if is_input_str else original_file_path
    orig_name_only: str = original_file_path.stem

    # Fill in missing suffix
    if filename_suffix is None:
        filename_suffix = ""

    # Make sure we have a valid file extension
    if new_file_extension is None:
        new_file_extension = original_file_path.suffix
    if not new_file_extension.startswith(".") and new_file_extension != "":
        new_file_extension = f".{new_file_extension}"

    # Fill in missing folder path
    if new_folder_path is None:
        new_folder_path = original_file_path.parent

    # Build new file path
    new_full_file_name = f"{orig_name_only}{filename_suffix}{new_file_extension}"
    new_path_str = os.path.join(new_folder_path, new_full_file_name)
    return new_path_str if is_input_str else Path(new_path_str)


def simplify_path(
    path: str | Path, max_parent_folders: int = 1, include_user_home: bool = False, filler_string: str = "..."
) -> str | Path:
    """
    Helper used to make 'nice' pathing string, intended for printing/reporting to end user.
    For example, given a long path like:
        input = "/user/home/path/to/something/in/many/folders/on/some/system/file.txt"
    This function will output:
        output = ".../system/file.txt"
    The user home pathing can be (optionally) included, giving:
        output = "~/.../system/file.txt"

    Returns:
        simplified_path
    """

    # Make sure we're dealing with a path
    is_input_str = isinstance(path, str)
    input_path: str = path if is_input_str else str(path)

    # Simplify user home pathing
    new_path = input_path
    has_home_path = False
    if include_user_home:
        user_home_path = os.path.expanduser("~")
        new_path = path.replace(user_home_path, "~")
        has_home_path = new_path != input_path

    # Remove middle path components, if needed
    out_path = str(new_path)
    path_parts = Path(new_path).parts
    num_allowable_parts = 1 + int(has_home_path) + max(max_parent_folders, 0)
    if len(path_parts) > num_allowable_parts:
        new_parts = list(reversed(path_parts))[0 : (1 + max_parent_folders)]
        new_parts.append(filler_string)
        if has_home_path:
            new_parts.append("~")
        out_path = os.path.join(*reversed(new_parts))

    return out_path if is_input_str else Path(out_path)
