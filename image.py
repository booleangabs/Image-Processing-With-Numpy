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
        self.conversion_methods = {
            0: self.__invertOrder,
            1: self.__invertOrder,
            2: self.__rgb2Gray,
            3: self.__gray2Rgb,
            4: self.__rgb2Rgba,
            5: self.__rgba2Rgb,
            6: self.__rgb2Hsv
            }
        
    def __call__(self, image: np.array) -> np.array:
        result = self.conversion_methods[self.mode](image)
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
    
    def __rgb2Hsv(self, image):
        def convertPixel(pixel):
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
        result = np.zeros_like(image).astype('uint8')
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i][j] = convertPixel(image[i][j] / division_factor)
        return result
    
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
        