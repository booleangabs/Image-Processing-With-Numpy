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
from image_processing.utils import merge, split, clip_to_range

    
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
    if image.dtype != float:
        image_float = image.astype("float64") / 255
    else:
        image_float = image.copy()

    R, G, B = split(image_float)
    M = image_float.max(axis=2)
    m = image_float.min(axis=2)
    
    V = M.copy()
    S = np.zeros_like(V)
    delta = M - m
    S[M != 0] = delta[M != 0] / M[M != 0]
    
    wr = np.where(np.bitwise_and(M == R, delta != 0))
    wg = np.where(np.bitwise_and(M == G, delta != 0))
    wb = np.where(np.bitwise_and(M == B, delta != 0))
    all_eq = np.bitwise_and(R == G, G == B)
    H = np.zeros_like(V)
    H[wr] = 60 * ((G[wr] - B[wr]) / delta[wr])
    H[wg] = 120 + 60 * (B[wg] - R[wg]) / delta[wg]
    H[wb] = 240 + 60 * (R[wb] - G[wb]) / delta[wb]
    H[all_eq] = 0
    H[H < 0] += 360 
    
    if image.dtype != float:
        return np.uint8(merge([H / 2, S * 255, V * 255]))
    else:
        return merge([H, S, V])

def impl_hsv2rgb(image: np.ndarray) -> np.ndarray:
    """HSV2RGB

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Converted Image
    """
    if image.dtype != float:
        image_float = image.astype("float64")
        image_float[..., 0] *= 2
        image_float[..., 1:] = image_float[..., 1:] / 255
    else:
        image_float = image.copy()
    
    H, S, V = split(image_float)
    
    result = np.zeros(image.shape, dtype="float64")
    C = S * V
    X = C * (1 - np.abs(((H / 60) % 2) - 1))
    M = V - C
    
    mask = H < 60
    result[mask] = np.dstack([C[mask], X[mask], np.zeros_like(C[mask])])
    
    mask = np.bitwise_and(H >= 60, H < 120)
    result[mask] = np.dstack([X[mask], C[mask], np.zeros_like(C[mask])])
    
    mask = np.bitwise_and(H >= 120, H < 180)
    result[mask] = np.dstack([np.zeros_like(C[mask]), C[mask], X[mask]])
    
    mask = np.bitwise_and(H >= 180, H < 240)
    result[mask] = np.dstack([np.zeros_like(C[mask]), X[mask], C[mask]])
    
    mask = np.bitwise_and(H >= 240, H < 300)
    result[mask] = np.dstack([X[mask], np.zeros_like(C[mask]), C[mask]])
    
    mask = H >= 300
    result[mask] = np.dstack([C[mask], np.zeros_like(C[mask]), X[mask]])
    
    result += merge([M] * 3)   

    if image.dtype != float:
        return np.uint8(result * 255)
    else:
        return result
    
def impl_rgb2hls(image: np.ndarray) -> np.ndarray:
    """RGB2HLS

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Converted Image
    """
    if image.dtype != float:
        image_float = image.astype("float64") / 255
    else:
        image_float = image.copy()

    R, G, B = split(image_float)
    M = image_float.max(axis=2)
    m = image_float.min(axis=2)
    
    L = (M + m) / 2
    S = np.zeros_like(L)
    delta = M - m
    mask = np.bitwise_and(L < 0.5, L != 0)
    S[mask] = delta[mask] / (2 * L)[mask]
    mask = (L >= 0.5)
    S[mask] = delta[mask] / (2 * (1 - L))[mask]
        
    wr = np.where(np.bitwise_and(M == R, delta != 0))
    wg = np.where(np.bitwise_and(M == G, delta != 0))
    wb = np.where(np.bitwise_and(M == B, delta != 0))
    all_eq = np.bitwise_and(R == G, G == B)
    H = np.zeros_like(L)
    H[wr] = 60 * ((G[wr] - B[wr]) / delta[wr])
    H[wg] = 120 + 60 * (B[wg] - R[wg]) / delta[wg]
    H[wb] = 240 + 60 * (R[wb] - G[wb]) / delta[wb]
    H[all_eq] = 0
    H[H < 0] += 360 
    
    if image.dtype != float:
        return np.uint8(merge([H / 2, L * 255, S * 255]))
    else:
        return merge([H, L, S])

def impl_hls2rgb(image: np.ndarray) -> np.ndarray:
    """HLS2RGB

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Converted Image
    """
    if image.dtype != float:
        image_float = image.astype("float64")
        image_float[..., 0] *= 2
        image_float[..., 1:] = image_float[..., 1:] / 255
    else:
        image_float = image.copy()
    
    H, L, S = split(image_float)
    
    result = np.zeros(image.shape, dtype="float64")
    C = S * (1 - np.abs(2 * L - 1))
    X = C * (1 - np.abs(((H / 60) % 2) - 1))
    M = L - (C / 2)
    
    mask = H < 60
    result[mask] = np.dstack([C[mask], X[mask], np.zeros_like(C[mask])])
    
    mask = np.bitwise_and(H >= 60, H < 120)
    result[mask] = np.dstack([X[mask], C[mask], np.zeros_like(C[mask])])
    
    mask = np.bitwise_and(H >= 120, H < 180)
    result[mask] = np.dstack([np.zeros_like(C[mask]), C[mask], X[mask]])
    
    mask = np.bitwise_and(H >= 180, H < 240)
    result[mask] = np.dstack([np.zeros_like(C[mask]), X[mask], C[mask]])
    
    mask = np.bitwise_and(H >= 240, H < 300)
    result[mask] = np.dstack([X[mask], np.zeros_like(C[mask]), C[mask]])
    
    mask = H >= 300
    result[mask] = np.dstack([C[mask], np.zeros_like(C[mask]), X[mask]])
    
    result += merge([M] * 3)   

    if image.dtype != float:
        return np.uint8(result * 255)
    else:
        return result
    
def impl_rgb2xyz(image: np.ndarray) -> np.ndarray:
    """RGB2XYZ

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Converted Image
    """
    if image.dtype != float:
        R, G, B = split(image.astype("float64") / 255)
    else:
        R, G, B = split(image)
      
    X = 0.412453 * R + 0.357580 * G + 0.180423 * B
    Y = 0.212671 * R + 0.715160 * G + 0.072169 * B
    Z = 0.019334 * R + 0.119193 * G + 0.950227 * B
    
    result = merge([X, Y, clip_to_range(Z, 0, 1)])
    if image.dtype != float:
        return np.uint8(result * 255)
    else:
        return result
    
def impl_xyz2rgb(image: np.ndarray) -> np.ndarray:
    """XYZ2RGB

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Converted Image
    """
    if image.dtype != float:
        X, Y, Z = split(image.astype("float64"))
    else:
        X, Y, Z = split(image)
        
    R = 3.240479 * X - 1.53715 * Y - 0.498535 * Z
    G = -0.969256 * X + 1.875991 * Y + 0.041556 * Z
    B = 0.055648 * X - 0.204043 * Y + 1.057311 * Z
    
    result = merge([R, G, B])
    if image.dtype != float:
        return np.uint8(clip_to_range(result, 0, 255))
    else:
        return clip_to_range(result, 0, 1)

conversion_methods = {
    cts.COLOR_RGB2BGR: impl_invert_order,
    cts.COLOR_BGR2RGB: impl_invert_order,
    cts.COLOR_RGB2GRAY: impl_rgb2gray,
    cts.COLOR_GRAY2RGB: impl_gray2rgb,
    cts.COLOR_RGB2RGBA: impl_rgb2rgba,
    cts.COLOR_RGBA2RGB: impl_rgba2rgb,
    cts.COLOR_RGB2HSV: impl_rgb2hsv,
    cts.COLOR_HSV2RGB: impl_hsv2rgb,
    cts.COLOR_RGB2HLS: impl_rgb2hls,
    cts.COLOR_HLS2RGB: impl_hls2rgb,
    cts.COLOR_RGB2XYZ: impl_rgb2xyz,
    cts.COLOR_XYZ2RGB: impl_xyz2rgb
}

def convert_color(image: np.ndarray, mode: int) -> np.ndarray:
    """Color conversion. Options for mode are
            COLOR_RGB2BGR, COLOR_BGR2RGB, COLOR_RGB2GRAY
            COLOR_GRAY2RGB, COLOR_RGB2RGBA, COLOR_RGBA2RGB
            COLOR_RGB2HSV, COLOR_HSV2RGB, COLOR_RGB2HSL
            COLOR_HSL2RGB, COLOR_RGB2XYZ, COLOR_XYZ2RGB

    * more options to be implemented in the future
    
    Args:
        image (np.ndarray): Input image
        mode (int): Color conversion flag. (COLOR_ prefix)

    Returns:
        np.ndarray: _description_
    """
    if not mode in conversion_methods.keys():
        raise RuntimeError(f"Conversion {mode} is not available or not a valid option.")
    return conversion_methods[mode](image).astype(image.dtype)