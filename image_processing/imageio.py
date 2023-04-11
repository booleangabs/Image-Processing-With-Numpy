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
import matplotlib.image as mpimg
import numpy as np

# local
import image_processing.color_conversion as ccv
import image_processing.constants as cts

# native
import os
import warnings


def read_image(path: str, mode: int = cts.READ_COLOR) -> np.ndarray:
    """Reads image
    Extension is automatically detected

    Args:
        path (str): Path to input image
        mode (int): Image reading mode. Options are
                READ_COLOR, READ_GRAY

    Raises:
        FileNotFoundError: If the image cannot be found on 'path'

    Returns:
        np.ndarray: Image as array
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Can't find image file ({path}).")
    image = mpimg.imread(path)
    if len(image.shape) > 2 and mode == cts.READ_GRAY:
        image = ccv.convert_color(image, cts.COLOR_RGB2GRAY)
    return image

def write_image(image: np.ndarray, path: str):
    """Writes image
    Wraps matplotlib's imsave function. The gray colormap application is 
    ignored if image is RGB
    
    Args:
        image (np.ndarray): Image as array
        path (str): Path to output image
    """
    if os.path.isfile(path):
        warnings.warn("File exists. Current file will be overwritten!")
    mpimg.imsave(path, image, cmap="gray")
