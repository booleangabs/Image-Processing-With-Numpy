"""
MIT License

Copyright (c) 2021 booleangabs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# site-packages
import numpy as np

# local
from context import image_processing as ipn
import matplotlib.image as mpimg


def MSE(target: np.ndarray, estimate: np.ndarray) -> float:
    """Mean squared error (MSE)

    Args:
        target (np.ndarray): Target
        estimate (np.ndarray): Estimate (same shape and dtype as target)

    Returns:
        np.ndarray: MSE
    """
    return ((target - estimate)**2).mean()


img = ipn.read_image("inputs/peppers3.tif", ipn.READ_COLOR)
img_mono = mpimg.imread("inputs/peppers2.tif")[..., 0]

img_bgr = ipn.convert_color(img, ipn.COLOR_RGB2BGR)
img_recon = ipn.convert_color(img_bgr, ipn.COLOR_BGR2RGB)
print(f"BGR conversion error (MSE): {MSE(img, img_recon)}")
ipn.show(img_bgr)

img_gray = ipn.convert_color(img, ipn.COLOR_RGB2GRAY)
print(f"Grayscale conversion error (MSE): {MSE(img_mono, img_gray)}")
ipn.show(img_gray)

img_hsv = ipn.convert_color(img, ipn.COLOR_RGB2HSV)
img_recon = ipn.convert_color(img_hsv, ipn.COLOR_HSV2RGB)
print(f"HSV conversion error (MSE): {MSE(img, img_recon)}")
ipn.show(img_hsv)

img_hls = ipn.convert_color(img, ipn.COLOR_RGB2HLS)
img_recon = ipn.convert_color(img_hls, ipn.COLOR_HLS2RGB)
print(f"HLS conversion error (MSE): {MSE(img, img_recon)}")
ipn.show(img_hls)

img_xyz = ipn.convert_color(img, ipn.COLOR_RGB2XYZ)
img_recon = ipn.convert_color(img_xyz, ipn.COLOR_XYZ2RGB)
print(f"XYZ conversion error (MSE): {MSE(img, img_recon)}")
ipn.show(img_xyz)
