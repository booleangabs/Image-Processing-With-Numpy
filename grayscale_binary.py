import numpy as np
import constants as cts
from histogram import Histogram

def threshold(image: np.array, value: float, max_value: float, mode: int) -> tuple:
    assert 0 <= mode < len([i for i in dir(cts) if not(i.startswith('__'))]), \
        "Invalid threshold mode"
    result = image.copy()
    if mode == cts.thr_binary:
        result[result <= value] = 0
        result[result > value] = max_value
    elif mode == cts.thr_inverse:
        result = max_value - threshold(image, value, max_value, cts.thr_binary)[1]
    elif mode == cts.thr_tozero:
        result[result <= value] = 0
    elif mode == cts.thr_tomax:
        result[result <= value] = max_value
    elif mode == cts.thr_otsu:
        hist = Histogram(result).hist
        max_icv = 0
        for i in range(255):
            w_0 = w_1 = u_0 = u_1 = 0
            for j in range(i+1):
                w_0 += hist[j]
                u_0 += j * hist[j]
            for j in range(i+1, 255):
                w_1 += hist[j]
                u_1 += j * hist[j]
            icv = w_0 * w_1 * ((u_0/w_0 - u_1/w_1)**2) if (w_0 != 0) & (w_1 != 0) else 0
            if max_icv < icv:
                value = i
                max_icv = icv
    return value, result

def countNonZero(image: np.array) -> int:
    return (image.shape[0] * image.shape[1]) - (image == 0).sum()

def dilate(image: np.array, kernel: np.array, iterations: int=1) -> np.array:
    for _ in range(iterations):
        result = np.zeros_like(image)
        ks = kernel.shape[0]
        c = ks // 2
        padded = np.pad(image, ((c, c), (c, c)))
        mask = kernel.astype('bool')
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                patch = padded[i:i + ks, j:j + ks]
                result[i][j] = patch[mask].max()
        return result

def erode(image: np.array, kernel: np.array, iterations: int=1):
    for _ in range(iterations):
        result = np.zeros_like(image)
        ks = kernel.shape[0]
        c = ks // 2
        padded = np.pad(image, ((c, c), (c, c)))
        mask = kernel.astype('bool')
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                patch = padded[i:i + ks, j:j + ks]
                result[i][j] = patch[mask].min()
        return result

def opening(image: np.array, kernel: np.array, iterations: int=1) -> np.array:
    dilated = image
    for _ in range(iterations):
        eroded = erode(dilated, kernel)
        dilated = dilate(eroded, kernel)
    return dilated

def closing(image: np.array, kernel: np.array, iterations: int=1) -> np.array:
    eroded = image
    for _ in range(iterations):
        dilated = dilate(eroded, kernel)
        eroded = dilate(dilated, kernel)
    return eroded

def morphologyGradient(image: np.array, kernel: np.array) -> np.array:
    return dilate(image, kernel) - erode(image, kernel)

def getShapedKernel(size: int, shape: str) -> np.array:
    assert (size % 2 == 1) & (size > 1), "Size must be odd and bigger than one"
    size = (size, size)
    if shape == cts.mph_square:
        kernel = np.ones(size)
    elif shape == cts.mph_cross:
        kernel = np.zeros(size)
        kernel[size[0] // 2, :] = 1
        kernel[:, size[1] // 2] = 1
    else:
        c = size[1] // 2
        y, x = np.ogrid[:c * 2 + 1, :c * 2 + 1]
        kernel = (np.hypot(x - c, y - c) <= c)
    return kernel.astype('uint8')
