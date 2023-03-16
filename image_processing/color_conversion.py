import numpy as np
import image_processing.constants as cts
from image_processing.utils import merge

    
def impl_invertOrder(image: np.ndarray):
    return image[..., ::-1]

def impl_rgb2Gray(image: np.ndarray):
    return np.ceil(0.2989 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])

def impl_gray2Rgb(image: np.ndarray):
    return merge([image] * 3)

def impl_rgb2Rgba(image: np.ndarray):
    result = np.zeros((image.shape[0], image.shape[1], 4))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i][j] = np.append(image[i][j], 255)
    return result

def impl_rgba2Rgb(image: np.ndarray):
    return image[..., :3]

def impl_rgb2Hsv(image: np.ndarray):
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

def impl_hsv2Rgb(image: np.ndarray):
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

conversion_methods = {
    cts.COLOR_RGB2BGR: impl_invertOrder,
    cts.COLOR_BGR2RGB: impl_invertOrder,
    cts.COLOR_RGB2GRAY: impl_rgb2Gray,
    cts.COLOR_GRAY2RGB: impl_gray2Rgb,
    cts.COLOR_RGB2RGBA: impl_rgb2Rgba,
    cts.COLOR_RGBA2RGB: impl_rgba2Rgb,
    cts.COLOR_RGB2HSV: impl_rgb2Hsv,
    cts.COLOR_HSV2RGB: impl_hsv2Rgb
}

def convert_color(image: np.ndarray, mode: int):
    return conversion_methods[mode](image).astype(image.dtype)