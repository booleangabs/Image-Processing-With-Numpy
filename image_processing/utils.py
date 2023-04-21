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
import matplotlib.pyplot as plt

# local
import image_processing.constants as cts


def show(image: np.ndarray) -> None:
    """Show image
    Wraps matplotlib's imshow and show

    Args:
        image (np.ndarray): Input image
    """
    plt.axis("off")
    
    if len(image.shape) > 2:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap="gray")
    plt.show()

def normalize(image: np.ndarray) -> np.ndarray:
    """Map pixel values to [0, 1]

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Normalized image
    """
    return (image - image.min()) / (image.max() - image.min())

def map_to_range(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """Maps pixel values to [low, high]

    Args:
        image (np.ndarray): Input image
        low (float): Lower bound for pixel value (and new min)
        high (float): Upperbound for pixel value (and new max)

    Returns:
        np.ndarray: Image with remapped values
    """
    norm = normalize(image)
    return norm * (high - low) + low

def clip_to_range(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clips pixel values to [low, high]

    Args:
        image (np.ndarray): Input image
        low (float): Lower bound for pixel value
        high (float): Upper bound for pixel value

    Returns:
        np.ndarray: Image with clipped values
    """
    result = image.copy()
    result[image < low] = low
    result[image > high] = high
    return result

def split(image: np.ndarray) -> list:
    """Splits image into a list of its channels

    Args:
        image (np.ndarray): Input image (H x W x C)

    Returns:
        list: List containing the C channels
    """
    assert len(image.shape) > 2, "Cannot split single channel images."
    result = []
    for i in range(image.shape[2]):
        result.append(image[..., i])
    return result

def merge(channels: list) -> np.ndarray:
    """Merge list of channels into a single image

    Args:
        channels (list): List of chanels

    Returns:
        np.ndarray: Composed image
    """
    return np.dstack(channels)
        
__all__ = [
    "show",
    "normalize",
    "map_to_range",
    "clip_to_range",
    "split",
    "merge"
]