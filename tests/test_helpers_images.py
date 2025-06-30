#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import unittest

import numpy as np

import toadui.helpers.images as tst


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


class TestBase64Images(unittest.TestCase):

    def test_encode(self):
        """
        - Check that a blank 1x1 image matches a target base64 (.png) encoding
        - Check the same image as .jpg matches a target length
        """
        test_img = np.zeros((1, 1), dtype=np.uint8)

        png_enc_result = tst.encode_image_b64str(test_img)
        target_result = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVQIHWNgAAAAAgABz8g15QAAAABJRU5ErkJggg=="
        self.assertEqual(png_enc_result, target_result, "Error encoding image with base64")

        jpg_enc_result = tst.encode_image_b64str(test_img, ".jpg")
        self.assertEqual(len(jpg_enc_result), 444, "Error encoding jpg image with base64")

        return

    def test_decode(self):
        """Check that base64 encoding and image then decoding it, reproduces the original image"""
        
        test_img = np.zeros((10, 20), dtype=np.uint8)
        enc_result = tst.encode_image_b64str(test_img, ".png")
        dec_result = tst.decode_b64str_image(enc_result)
        self.assertTrue(np.allclose(test_img, dec_result), "Error with base64 decoding (Image mismatch)")
        
        return


# %%

if __name__ == "__main__":
    unittest.main()
