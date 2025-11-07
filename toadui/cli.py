#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os.path as osp
from pathlib import Path
from time import sleep
import json

# For type hints
from typing import Iterable, Callable, Any
from toadui.helpers.types import SelfType


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class HistoryJSON:
    """
    Class used to manage loading/saving to a 'history' JSON file.

    The point of this being to store & re-use important settings,
    typically for re-using file loading paths.
    Typical usage:

        # Load existing history, if any
        history = HistoryJSON()
        _, history_loadpath = history.read("load_path")
        _, history_setting = history.read("important_setting")

        # Use history values as default
        load_path = ask_to_load_file(default=history_loadpath)
        setting = ask_for_important_setting(default=history_setting)

        # Store new results
        history.store(load_path=load_path, important_setting=setting)

    """

    # .................................................................................................................

    def __init__(self, save_folder: str | Path | None = None, save_name: str = ".history"):

        # Set up history save file pathing
        if save_folder is None:
            save_folder = __file__
        save_folder = Path(save_folder)
        save_folder = save_folder.expanduser()
        save_folder = save_folder.parent if save_folder.is_file() else save_folder
        if not save_folder.exists():
            save_folder = ""
        save_path = Path(save_folder).joinpath(save_name)

        self._filepath = save_path
        self._history_dict = {}
        self.reload()

    # .................................................................................................................

    def read(self, key: str, value_if_missing: Any = None) -> tuple[bool, Any]:
        """
        Read from history data
        Returns:
            is_valid_key, loaded_value
        """
        is_valid_key = key in self._history_dict.keys()
        loaded_value = self._history_dict.get(key, value_if_missing)
        return is_valid_key, loaded_value

    def store(self, **key_value_kwargs: Any) -> SelfType:
        """
        Update and save history data. Use as follows:
            history.store(some_setting=5, another_setting="hello", load_path="/path/to/data")

        This will save data to a JSON file, with the structure:
            {
                "some_setting": 5,
                "another_setting": "hello",
                "load_path: "/path/to/data",
            }
        Which can be read from using the .read(...) function.
        """

        # Force path types to strings (otherwise not serializable)
        key_value_kwargs = {k: v if not isinstance(v, Path) else str(v) for k, v in key_value_kwargs.items()}

        # Check if new data is valid
        new_history_dict = {**self._history_dict, **key_value_kwargs}
        try:
            json.dumps(new_history_dict)
            is_valid_json = True
        except TypeError:
            is_valid_json = False
            print(
                "",
                f"*** {self.__class__.__name__} Error ***",
                "Cannot store history, invalid as json:",
                new_history_dict,
                sep="\n",
                flush=True,
            )

        # Only re-write history data if the json is valid
        if is_valid_json:
            with open(self._filepath, "w") as outfile:
                json.dump(new_history_dict, outfile, indent=2)
            self._history_dict = new_history_dict

        return self

    # .................................................................................................................

    def reload(self) -> SelfType:
        """
        Reload an existing history file (if any). This should not need to be
        called under normal circumstances. It is helpful for initial loading
        mainly (or if something is changing the histrory file externally).
        """

        try:
            with open(self._filepath, "r") as infile:
                history_dict = json.load(infile)

        except json.JSONDecodeError:
            history_dict = {}
            print(
                "",
                f"*** {self.__class__.__name__} Error ***",
                "History file is corrupted! Unable to load...",
                f"@ {self._filepath}",
                sep="\n",
                flush=True,
            )

        except FileNotFoundError:
            history_dict = {}
        self._history_dict = history_dict

        return self

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def ask_for_path_if_missing(
    path: str | None = None,
    default_path: str | None = None,
    allow_files: bool = True,
    allow_folders: bool = False,
    prompt: str = "Enter path to file: ",
    special_case_check: Callable[str, bool] | None = None,
    quit_on_keyboard_interrupt: bool = True,
) -> str:
    """
    Helper used to provide a basic command-line prompt asking for a file/folder path
    If avalid path is given as input, then the prompt will be skipped entirely.

    If no path (or an invalid path) is given, then the user will be prompted to
    provide a valid path to a file/folder. The user will be re-prompted if they
    give an invalid path.

    Also includes support for 'default' inputs, which will be shown above the user prompt.

    A 'special_case_check' function can be provided to perform custom checks on the
    validity of user inputs. This function must take a single string as input and
    return True if the string is a valid input, otherwise false. For example, this
    can be used to accept special keywords:
        special_case_check = lambda s: "keyword" in s
    -> With this, if a user types in: 'keyword', it will be returned as a valid path

    Returns:
        valid_path
    """

    # Sanity check
    assert allow_files or allow_folders, "Must allow at least one of files and folders!"

    # Bail if we get a good path
    path = _clean_path_str(path)
    if osp.exists(path):
        return path

    # Use dummy special case check if not provided
    if special_case_check is None:
        special_case_check = lambda s: False
    else:
        assert callable(special_case_check), "special_case_check must be a function!"

    # Bail if given a valid custom path
    if special_case_check(path):
        return path

    # Wipe out bad default paths
    if default_path is not None:
        if not (osp.exists(default_path) or special_case_check(default_path)):
            default_path = None

    # Pad prompt text to align with default text if needed (only if using a short custom prompt)
    num_trailing_space = len(prompt) - len(prompt.rstrip(" "))
    prompt_for_default = "(default:" + "".join([" "] * num_trailing_space)
    if len(prompt) < len(prompt_for_default) and default_path is not None:
        prompt = prompt.rjust(len(prompt_for_default), " ")

    # Set up prompt text and default if needed
    default_msg_spacing = " " * (len(prompt) - len(prompt_for_default) - 0)
    default_msg = "" if default_path is None else f"{default_msg_spacing}{prompt_for_default}{default_path})"

    # Keep asking for a path until it points to something
    try:
        while True:

            # Print empty line for spacing and default hint if available
            print("", flush=True)
            if default_path is not None:
                print(default_msg, flush=True)

            # Ask user for path, and fallback to default if nothing is given
            path = _clean_path_str(input(prompt))
            if path == "" and default_path is not None:
                path = default_path

            # Check custom validations
            if special_case_check(path):
                break

            # Stop asking once we get a valid path
            if osp.exists(path):
                if osp.isfile(path) and allow_files:
                    break
                if osp.isdir(path) and allow_folders:
                    break
            print("", "", "Invalid path!", sep="\n", flush=True)
            sleep(0.75)

    except KeyboardInterrupt:
        if quit_on_keyboard_interrupt:
            print()
            quit()
        raise KeyboardInterrupt()

    return path


# .....................................................................................................................


def ask_for_media_path(
    path: str | None = None,
    allow_image=True,
    allow_video=True,
    allow_webcam=True,
    allow_folder=False,
    prompt: str | None = None,
    quit_on_keyboard_interrupt: bool = True,
    default_path: str | None = None,
    history_folder: str | Path | None = None,
    history_name: str | None = ".path_history",
) -> str:
    """
    Helper function used ask for paths to visual media, or folders.
    This is similar to the ask_for_path_if_missing(...) function,
    except three notable differences:
        - can accept webcam inputs (anything with 'cam' in the name)
          even though these are not file/folder paths
        - will auto-generate the prompt text (e.g. 'Enter path to ___')
          based on which inputs are allowed
        - includes a built-in history saving function. This
          automatically fills in default path using prior results
          (based on prompt). Can be disabled by setting history_name=None

    Note that this function does not validate images/videos/webcam
    inputs, other than to confirm the file path exists.
    (or contains 'cam' if webcams are allowed)
    """

    # Sanity check
    allow_files = any((allow_image, allow_video, allow_webcam))
    assert allow_files or allow_folder, "Must allow folders or at least one file type!"

    # Auto-generate prompt message if needed
    if prompt is None:
        prompt_lut = {"video": allow_video, "image": allow_image, "folder": allow_folder, "cam": allow_webcam}
        allowed_types = [key for key, val in prompt_lut.items() if val]
        prompt_types = allowed_types[0]
        if len(allowed_types) > 1:
            comma_str = ", ".join(allowed_types[0:-1])
            or_str = f" or {allowed_types[-1]}"
            prompt_types = "".join((comma_str, or_str))
        prompt = f"Enter path to {prompt_types}: "

    # Generate a default path from history
    enable_history = history_name is not None
    if enable_history:
        hist = HistoryJSON(history_folder, history_name)
        if default_path is None:
            _, default_path = hist.read(prompt)

    # Ask user for path
    check_for_webcam_func = lambda s: ("cam" in str(s).lower()) and (not osp.exists(s))
    selected_path = ask_for_path_if_missing(
        path,
        default_path,
        allow_files=allow_files,
        allow_folders=allow_folder,
        prompt=prompt,
        special_case_check=check_for_webcam_func if allow_webcam else None,
        quit_on_keyboard_interrupt=quit_on_keyboard_interrupt,
    )

    # Store result for re-use if it isn't already what we're using
    if enable_history and (default_path != selected_path):
        hist.store(**{prompt: selected_path})

    return selected_path


# .....................................................................................................................


def select_from_options(
    menu_options: Iterable[str],
    default_option: str | None = None,
    menu_message: str = "Select option:",
    special_case_check: Callable[str, bool] | None = None,
    quit_on_keyboard_interrupt: bool = True,
) -> str:
    """
    Function which provides a simple ui for selecting an item from a 'menu'.
    A default can be provided, which will highlight a matching entry in the menu
    (if present), and will be used if the user does not enter a selection.
    For example:

    Select option:

      1: option A
      2: option B (default)
      3: option C

    Enter selection: <user input here>

    Entries are 'selected' by entering their list index, or can be selected by providing
    a partial string match. Returns: selected_option
    """

    # Convert input to list for predictability
    options_list = [str(item) for item in menu_options]

    # Wipe out bad default paths
    default_is_available = default_option is not None
    if default_is_available:
        if not osp.exists(default_option):
            default_option = None
            default_is_available = False

    # Add default to menu, if it isn't already included
    if default_is_available:
        default_in_listing = any(default_option == item for item in options_list)
        if not default_in_listing:
            options_list.append(default_option)

    # Create menu listing strings for each option for display
    menu_item_strs = []
    for idx, item in enumerate(options_list):
        menu_str = f" {1+idx:>2}: {item}"
        is_default = item == default_option
        if is_default:
            menu_str += " (default)"
        menu_item_strs.append(menu_str)

    # Set up prompt text and feedback printing
    prompt_txt = "Enter selection: "
    feedback_prefix = " " * (len(prompt_txt) - len("-->") - 1) + "-->"

    # Keep giving menu until user selects something
    selected_option = None
    try:
        while True:

            # Provide prompt to ask user to select an item
            print("", menu_message, "", *menu_item_strs, "", sep="\n")
            user_selection = _clean_path_str(input("Enter selection: "))

            # Use the default if the user didn't enter anything
            if user_selection == "" and default_is_available:
                selected_option = default_option
                break

            # Check if user entered a number matching an item in the list
            try:
                idx_select = int(user_selection) - 1
                selected_option = options_list[idx_select]
                break
            except (ValueError, IndexError):
                # Happens if user didn't input an integer selecting an item in the menu
                pass

            # Check custom validations
            if special_case_check(user_selection):
                break

            # Check if the user entered a string that matches to some part of one of the entries
            filtered_names = [item for item in options_list if user_selection in item]
            if len(filtered_names) == 1:
                user_selected_name = filtered_names[0]
                idx_select = options_list.index(user_selected_name)
                selected_option = options_list[idx_select]
                break

            # If we get here, we didn't get a valid input. So warn user and repeat prompt
            print("", "", "Invalid selection!", sep="\n", flush=True)
            sleep(0.75)

    except KeyboardInterrupt:
        if quit_on_keyboard_interrupt:
            print()
            quit()

    print(f"{feedback_prefix} {selected_option}")
    return selected_option


# ---------------------------------------------------------------------------------------------------------------------
# %% Helpers


def _clean_path_str(path: str | None = None) -> str:
    """
    Helper used to interpret user-given paths correctly
    - Removes trailing white space
    - Removes surrounding quotations
    - Expands user pathing (e.g. '~/Desktop' is expanded to '<user home folder path>/Desktop')
    """

    path_str = "" if path is None else str(path)
    path_str = path_str.strip()
    path_str = path_str.removeprefix("'").removesuffix("'")
    path_str = path_str.removeprefix('"').removesuffix('"')
    return osp.expanduser(path_str)


def _convert_to_set_of_strings(iterable_of_strs: None | str | Iterable[str]) -> set[str]:
    """
    Helper used to make a set of string
    - If given None, returns empty set
    - If given a single string, returns set([string])
    - Otherwise returns: set(iterable_of_strs)
    """
    if iterable_of_strs is None:
        return set()
    if isinstance(iterable_of_strs, str):
        iterable_of_strs = tuple([iterable_of_strs])
    return set(iterable_of_strs)
