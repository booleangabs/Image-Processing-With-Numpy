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
import image_processing.constants as cts

# local
from image_processing.utils import merge

    
def impl_invert_order(image: np.ndarray) -> np.ndarray:
    """RGB2BGR/BGR2RGB
    Invert channel order (rgb to bgr-like transformations)

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Converted Image
    """
    return image[..., ::-1]

def impl_rgb2gray(image: np.ndarray) -> np.array:
    """RGB2GRAY
    Converts RGB to grayscale (Y channel)

    Args:
        image (np.ndarray): Input image

    Returns:
        np.array: Converted Image
    """
    return np.ceil(0.2989 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])

def impl_gray2rgb(image: np.ndarray) -> np.ndarray:
    """GRAY2RGB
    Repeat grayscale image along three channels

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Converted Image
    """
    return merge([image.copy(), image.copy(), image.copy()])

def impl_rgb2rgba(image: np.ndarray) -> np.ndarray:
    """RGB2RGBA
    Creates the alpha channel filling it with 255

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Converted Image
    """
    result = np.zeros((image.shape[0], image.shape[1], 4), dtype=image.dtype)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i][j] = np.append(image[i][j], 255)
    return result

def impl_rgba2rgb(image: np.ndarray) -> np.ndarray:
    """RGBA2RGB
    Drops the alpha channel

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Converted Image
    """
    return image[..., :3]

def impl_rgb2hsv(image: np.ndarray) -> np.ndarray:
    """RGB2HSV

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Converted Image
    """
    def convertPixel(pixel):
        """Wraps the pixelwise calculations for HSV"""
        M = pixel.max(), np.argmax(pixel)
        m = pixel.min(), np.argmin(pixel)
        c = M[0] - m[0]
        v = M[0]
        s = 0 if M[0] == 0 else c / M[0]
        h = None
        if c == 0:
            h = 0
        else:
            if M[1] == pixel[0]:
                a = 6 if pixel[1] < pixel[2] else 0
                h = ((pixel[1] - pixel[2]) / c) + a
            elif M[1] == pixel[1]:
                h = ((pixel[2] - pixel[1]) / c) + 2
            else:
                h = ((pixel[0] - pixel[1]) / c) + 4
            h /= 6
        h *= 180
        s *= 255
        v *= 255
        return np.uint8([h, s, v])
    
    division_factor = 1 if image.max() <= 1 else 255
    result = np.zeros_like(image).astype(image.dtype)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i][j] = convertPixel(image[i][j] / division_factor)
    return result

def impl_hsv2rgb(image: np.ndarray) -> np.ndarray:
    """HSV2RGB

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Converted Image
    """
    def convertPixel(pixel):
        """Wraps the pixelwise calculations for HSV"""
        if pixel[2] == 0:
            return np.zeros((3, ))
        h, s, v = pixel
        h /= 30
        s /= 255
        v /= 255
        switch = int(h)
        frac = h - switch
        c = [v * (1 - s), 
                v * (1 - (s * frac)), 
                v * (1 - (s * (1 - frac)))]
        if switch == 0:
            r, g, b = v, c[2], c[0]
        elif switch == 1:
            r, g, b = c[1], v, c[0]
        elif switch == 2:
            r, g, b = c[0], v, c[2]
        elif switch == 3:
            r, g, b = c[0], c[1], v
        elif switch == 4:
            r, g, b = c[2], c[1], v
        else:
            r, g, b = v, c[0], c[1]
        r *= 255
        g *= 255
        b *= 255
        return np.uint8([r, g, b])
    
    result = np.zeros_like(image).astype(image.dtype)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i][j] = convertPixel(image[i][j])
    return result

conversion_methods = {
    cts.COLOR_RGB2BGR: impl_invert_order,
    cts.COLOR_BGR2RGB: impl_invert_order,
    cts.COLOR_RGB2GRAY: impl_rgb2gray,
    cts.COLOR_GRAY2RGB: impl_gray2rgb,
    cts.COLOR_RGB2RGBA: impl_rgb2rgba,
    cts.COLOR_RGBA2RGB: impl_rgba2rgb,
    cts.COLOR_RGB2HSV: impl_rgb2hsv,
    cts.COLOR_HSV2RGB: impl_hsv2rgb
}

def convert_color(image: np.ndarray, mode: int) -> np.ndarray:
    """Color conversion. Options for mode are
            COLOR_RGB2BGR, COLOR_BGR2RGB, COLOR_RGB2GRAY
            COLOR_GRAY2RGB, COLOR_RGB2RGBA, COLOR_RGBA2RGB
            COLOR_RGB2HSV, COLOR_HSV2RGB, COLOR_RGB2HSL*
            COLOR_HSL2RGB*, COLOR_RGB2LAB*, COLOR_LAB2RGB*

    * to be implemented in the future
    
    Args:
        image (np.ndarray): Input image
        mode (int): Color conversion flag. (COLOR_ prefix)

    Returns:
        np.ndarray: _description_
    """
    if not mode in conversion_methods.keys():
        raise RuntimeError(f"Conversion {mode} is not available or not a valid option.")
    return conversion_methods[mode](image).astype(image.dtype)