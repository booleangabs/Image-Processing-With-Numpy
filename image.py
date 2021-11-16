import numpy as np
import matplotlib.pyplot as plt
from constants import Constants as cts 
import warnings; warnings.filterwarnings('ignore')   
    
class ImageReader:
    def __init__(self, mode):
        assert mode in [0, 1, 2], 'Insert a valid mode. Print cts.__doc__ for help.'
        self.mode = mode
    
    def __call__(self, path: str) -> np.ndarray:
        from cv2 import imread, cvtColor
        if self.mode == cts.ird_color:
            image = imread(path)
            image = cvtColor(image, 4)
        elif self.mode == cts.ird_gray:
            image = imread(path, 0)
        else:
            image = imread(path, -1)
        return image.astype('float32')
    
class ColorConverter:
    def __init__(self, mode: int):
        assert mode in range(12), 'Please use a valid mode'
        self.mode = mode
        
    def __call__(self, image: np.array) -> np.array:
        if self.mode in (cts.ccv_rgb2bgr, cts.ccv_bgr2rgb):
            result = self.__invertOrder(image)
        elif self.mode == cts.ccv_rgb2gray:
            result = self.__rgb2Gray(image)
        elif self.mode == cts.ccv_gray2rgb:
            result = self.__gray2Rgb(image)
        elif self.mode == cts.ccv_rgb2rgba:
            result = self.__rgb2Rgba(image)
        elif self.mode == cts.ccv_rgba2rgb:
            result = self.__rgba2Rgb(image)
        return result.astype('uint8')
    
    def __invertOrder(self, image):
        return image[..., ::-1]
    
    def __rgb2Gray(self, image):
        return np.ceil(0.2989 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])
    
    def __gray2Rgb(self, image):
        return merge([image] * 3)
    
    def __rgb2Rgba(self, image):
        result = np.zeros((image.shape[0], image.shape[1], 4))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i][j] = np.append(image[i][j], 255)
        return result
    
    def __rgba2Rgb(self, image):
        return image[..., :3]
    
def mapToRange(image: np.ndarray, low: float, high: float) -> np.ndarray:
    norm = (image - image.min()) / (image.max() - image.min())
    return norm * (high - low) + low

def clipToRange(image: np.ndarray, low: float, high: float) -> np.ndarray:
    result = image.copy()
    result[image < low] = low
    result[image > high] = high
    return result

def split(image: np.ndarray) -> list:
    assert len(image.shape) > 2, 'Cannot use it on single channel images.'
    result = []
    for i in range(image.shape[2]):
        result.append(image[..., i])
    return result

def merge(channels: list) -> np.ndarray:
    return np.dstack(channels)

def show(image: np.ndarray):
    plt.axis('off')
    if len(image.shape) == 2:
        plt.imshow(image.astype('uint8'), cmap='gray')
    else:
        plt.imshow(image.astype('uint8'))
    plt.show()
        