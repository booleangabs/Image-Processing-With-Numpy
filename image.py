import numpy as np
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