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

def convolve2d(image: np.array, kernel: np.array) -> np.array:
    from cv2 import filter2d
    return filter2d(image.astype('float32'), -1, kernel.astype('float32'))