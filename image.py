import numpy as np
import matplotlib.pyplot as plt
from constants import Constants as cts
    
class ImageReader:
    def __init__(self, mode):
        assert mode in [0, 1, 2], 'Insert a valid mode. Print cts.__doc__ for help.'
        self.mode = mode
    
    def __call__(self, path: str) -> np.array:
        from cv2 import imread, cvtColor
        if self.mode == cts.ird_color:
            image = imread(path)
            image = cvtColor(image, 4)
        elif self.mode == cts.ird_gray:
            image = imread(path, 0)
        else:
            image = imread(path, -1)
        return image.astype('float32')
    
def mapToRange(image: np.array, low: float, high: float) -> np.array:
    norm = (image - image.min()) / (image.max() - image.min())
    return norm * (high - low) + low

def clipToRange(image: np.array, low: float, high: float) -> np.array:
    result = image.copy()
    result[image < low] = low
    result[image > high] = high
    return result

def convertColor(image: np.array, mode: int):
    assert 0 <= mode <= 11, "Check Constants class' docs for valid modes."
    if mode in (cts.ccv_rgb2bgr, cts.ccv_bgr2rgb):
        result = image[..., ::-1]
    elif mode == cts.ccv_rgb2gray:
        result = np.ceil(0.2989 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])
    elif mode == cts.ccv_gray2rgb:
        result = merge([image]*3)
    elif mode == cts.ccv_rgb2rgba:
        result = image.copy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i][j] = np.append(result[i][j], 255)
    elif mode == cts.ccv_rgba2rgb:
        result = image[..., :3]
    return result

def split(image: np.array):
    assert len(image.shape) > 2, "Cannot use it on single channel images."
    result = []
    for i in range(image.shape[2]):
        result.append(image[..., i])
    return result

def merge(channels: list):
    return np.dstack(channels)

def show(image: np.array):
    plt.axis('off')
    if len(image.shape) == 2:
        plt.imshow(image.astype('uint8'), cmap='gray')
    else:
        plt.imshow(image.astype('uint8'))
    plt.show()
        