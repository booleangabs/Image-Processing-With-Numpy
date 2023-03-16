import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import image_processing.constants as cts
import warnings; 
warnings.filterwarnings('ignore')   


class Image():
    def __init__(self, path: str, mode: int = cts.READ_COLOR):
        self.data = mpimg.imread(path)
        if mode == cts.READ_GRAY and len(self.data.shape) > 2:
            cc = ColorConverter(cts.COLOR_RGB2GRAY)
            self.data = cc(self.data)
        self.mode = mode
    
    def show(self):
        plt.axis("off")
        if self.mode == cts.READ_COLOR:
            plt.imshow(self.data.astype("uint8"))
        else:
            plt.imshow(self.data, cmap="gray")
        plt.show()
    
class ColorConverter:
    def __init__(self, mode: int):
        assert mode in range(12), 'Please use a valid mode'
        self.mode = mode
        self.conversion_methods = {
            cts.COLOR_RGB2BGR: self.__invertOrder,
            cts.COLOR_BGR2RGB: self.__invertOrder,
            cts.COLOR_RGB2GRAY: self.__rgb2Gray,
            cts.COLOR_GRAY2RGB: self.__gray2Rgb,
            cts.COLOR_RGB2RGBA: self.__rgb2Rgba,
            cts.COLOR_RGBA2RGB: self.__rgba2Rgb,
            cts.COLOR_RGB2HSV: self.__rgb2Hsv,
            cts.COLOR_HSV2RGB: self.__hsv2Rgb
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
    
    def __hsv2Rgb(self, image):
        def convertPixel(pixel):
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
        
        result = np.zeros_like(image).astype('uint8')
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i][j] = convertPixel(image[i][j])
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
        