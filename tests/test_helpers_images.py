#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import unittest

import numpy as np

import toadui.helpers.images as tst


# ---------------------------------------------------------------------------------------------------------------------
# %% Tests


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
