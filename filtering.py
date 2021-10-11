import numpy as np
from image import mapToRange, clipToRange
from constants import Constants as cts

def boxBlur(image: np.array, size: int=3) -> np.array:
    kernel = np.ones((size, size))
    return convolve(image, kernel)

def gaussianBlur(image: np.array, sigma: float, size: int=3) -> np.array:
    ksm1o2 = (size - 1) / 2
    sub = (-1 / (2 * sigma**2))
    gaussian = np.e ** (sub * np.linspace(-ksm1o2, ksm1o2, size)**2)
    gaussian = np.outer(gaussian, gaussian.T)
    gaussian /= gaussian.sum()
    return convolve(image, gaussian)

def medianBlur(image: np.array, size: int=3) -> np.array:
    result = np.zeros_like(image)
    c = size // 2
    padded = np.pad(image, ((c, c), (c, c))).astype('uint8')
    mask = np.ones((size, size)).astype('bool')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded[i:i + size, j:j + size]
            array = patch[mask].flatten().copy()
            array.sort()
            result[i][j] = array[size**2 // 2]
    return result

def sharpen(image: np.array, threshold: float, sigma: float, alpha: float=0.33333, size: int=3) -> np.array:
    assert (threshold > 0) & (0 <= alpha <= 1) & (sigma > 0), "Insert valid values for the parameters."
    smooth = gaussianBlur(image, sigma, size)
    sharpened = (1 - alpha) * image  - alpha * smooth
    sharpened = clipToRange(image, 0, 255)
    mask = np.abs(image - smooth) < threshold
    np.copyto(sharpened, image, where=mask)
    return sharpened

def sobel(image: np.array, size: int=3) -> np.array:
    assert size in [3, 5, 7], "Sobel operators available have 3, 5 or 7 as size"
    results = {}
    results['x'] = convolve(image, cts.dvf[f"sobel_x{size}"])
    results['y'] = convolve(image, cts.dvf[f"sobel_y{size}"])
    results['orientation'] = np.arctan2(results['y'], results['x'])
    results['magnitude'] = np.hypot(results['y'], results['x'])
    results['x_r'] = mapToRange(results['x'], 0, 255)
    results['y_r'] = mapToRange(results['y'], 0, 255)
    return results

def laplacian(image: np.array) -> np.array:
    laplacian_k = np.float32([[-1, -1, -1], 
                              [-1, 8, -1], 
                              [-1, -1, -1]])
    return convolve(image, laplacian_k)

def convolve(image: np.array, kernel: np.array, flipped: bool=True) -> np.array:
    result = np.zeros_like(image)
    filter_ = kernel.copy()
    ks = filter_.shape[0]
    c = ks // 2
    padded = np.pad(image, ((c, c), (c, c)))
    if flipped:
        filter_ = np.rot90(kernel, 2)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded[i:i + ks, j:j + ks]
            result[i][j] = (patch * filter_).sum()
    return result