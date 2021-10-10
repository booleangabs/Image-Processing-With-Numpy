import numpy as np

def boxBlur(image: np.array, size: int=3) -> np.array:
    kernel = np.ones((size, size))
    return convolve(image, kernel)

def gaussianBlur(image: np.array, sigma: float, size: int=3) -> np.array:
    ksm1o2 = (size - 1) / 2
    sub = (-1 / (2 * sigma**2))
    gaussian = np.e ** (sub * np.linspace(-ksm1o2, ksm1o2, size)**2)
    gaussian = np.outer(gaussian, gaussian.T)
    return gaussian / gaussian.sum()

def medianBlur(image: np.array, size: int=3):
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

def sharpen():
    pass

def sobel():
    pass

def laplacian():
    pass

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
    
    
    
    
    return