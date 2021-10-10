import numpy as np

def boxBlur():
    pass

def gaussianBlur():
    pass

def medianBlur():
    pass

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