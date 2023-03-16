import matplotlib.image as mpimg
import image_processing.color_conversion as ccv
import image_processing.constants as cts
import os
import numpy as np
import warnings

def read_image(path: str, mode: int) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Can't find image file ({path}).")
    image = mpimg.imread(path)
    if len(image.shape) > 2 and mode == cts.READ_GRAY:
        image = ccv.convert_color(image, cts.COLOR_RGB2GRAY)
    return image

def write_image(image: np.ndarray, path: str):
    if os.path.isfile(path):
        warnings.warn("File exists. The execution will overwrite the current file.")
    mpimg.imsave(path, image, cmap="gray")
