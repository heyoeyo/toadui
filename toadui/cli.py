#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os.path as osp
from time import sleep

# For type hints
from typing import Iterable, Callable


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def ask_for_path_if_missing(
    path: str | None = None,
    default_path: str | None = None,
    path_type: str = "file",
    allow_files: bool = True,
    allow_folders: bool = False,
    special_case_check: Callable[str, bool] | None = None,
    quit_on_keyboard_interupt=True,
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

    # Wipe out bad default paths
    if default_path is not None:
        if not osp.exists(default_path):
            default_path = None

    # Use dummy special case check if not provided
    if special_case_check is None:
        special_case_check = lambda s: False
    else:
        assert callable(special_case_check), "special_case_check must be a function!"
    
    # Bail if given a valid custom path
    if special_case_check(path):
        return path

    # Set up prompt text and default if needed
    prompt_txt = f"Enter path to {path_type}: "
    default_msg_spacing = " " * (len(prompt_txt) - len("(default:") - 1)
    default_msg = "" if default_path is None else f"{default_msg_spacing}(default: {default_path})"

    # Keep asking for a path until it points to something
    try:
        while True:

            # Print empty line for spacing and default hint if available
            print("", flush=True)
            if default_path is not None:
                print(default_msg, flush=True)

            # Ask user for path, and fallback to default if nothing is given
            path = _clean_path_str(input(prompt_txt))
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
            print("", "", f"Invalid {path_type} path!", sep="\n", flush=True)
            sleep(0.75)

    except KeyboardInterrupt:
        if quit_on_keyboard_interupt:
            print()
            quit()
        raise KeyboardInterrupt()

    return path


# .....................................................................................................................


def select_from_options(
    menu_options: Iterable[str],
    default_option: str | None = None,
    menu_message: str = "Select option:",
    special_case_check: Callable[str, bool] | None = None,
    quit_on_keyboard_interupt: bool = True,
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
        if quit_on_keyboard_interupt:
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
