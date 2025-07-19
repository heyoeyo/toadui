#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import unittest

import numpy as np

import toadui.helpers.drawing as tst


# ---------------------------------------------------------------------------------------------------------------------
# %% Tests


class TestBoxOutline(unittest.TestCase):

    def test_draw_box_outline(self):

        test_size = 100
        non_border_value = 0
        border_value = 1
        zeros_img = np.full((test_size, test_size, 3), non_border_value, dtype=(np.uint8))
        bcolor = [border_value] * 3
        mid_idx = test_size // 2

        for thickness in [1, 2, 3, 4, 5, 10, 20]:
            test_img = tst.draw_box_outline(zeros_img.copy(), bcolor, thickness=thickness)

            # Grab 1px line of (Red) values
            # -> Want to test that values look like: [border, border, ...., border, non-border]
            #    where the number of 'border' values matches the thickness setting
            slice_idx = thickness + 1
            l_slice = test_img[mid_idx, 0:slice_idx, 0]
            r_slice = test_img[mid_idx, -slice_idx:, 0]
            t_slice = test_img[0:slice_idx, mid_idx, 0]
            b_slice = test_img[-slice_idx:, mid_idx, 0]

            correct_border_size = all(s.sum() == thickness * border_value for s in [l_slice, r_slice, t_slice, b_slice])
            self.assertTrue(correct_border_size, "Wrong border size")

            last_pixel_not_part_of_border = all(s[-1] == non_border_value for s in [l_slice, t_slice])
            self.assertTrue(last_pixel_not_part_of_border, "Border offset incorrectly (top/left)")

            first_pixel_not_part_of_border = all(s[0] == non_border_value for s in [r_slice, b_slice])
            self.assertTrue(first_pixel_not_part_of_border, "Border offset incorrectly (top/left)")

        return


# %%

if __name__ == "__main__":
    unittest.main()
