import statistics as stc
import numpy as np
import matplotlib.pyplot as plt

class Constants:
    # Thresholding
    thr_binary = 0
    thr_inverse = 1
    thr_tozero = 2
    thr_tomax = 3
    thr_otsu = 4
    
class ImageReader:
    def __init__(self, mode):
        self.mode = mode
    
    def read(self, path: str) -> np.array:
        from cv2 import imread, cvtColor, COLOR_BGR2RGB, COLOR_BGR2GRAY
        image = imread(path)
        if self.mode == 'color':
            image = cvtColor(image, COLOR_BGR2RGB)
        else:
            image = cvtColor(image, COLOR_BGR2GRAY)
        return image.astype('float32')

# Utils

def histogram(image: np.array, normalize: bool= True) -> dict:
    hist = dict()
    total = image.shape[0]*image.shape[1]
    for i in range(255):
        count = (image == i).sum()
        if normalize:
            count =  np.round(count / total, 4)
        hist[i] = count
    return hist
        
def plotHistogram(image: np.array, normalize: bool= True, color: str= 'black'):
    hist = histogram(image, normalize)
    plt.bar(list(hist.keys()), hist.values(), width=1, color=color)
    plt.show()

# Grayscale image processing

def threshold(image: np.array, value: float, max_value: float, mode: int) -> tuple:
    assert 0 <= mode < len([i for i in dir(Constants) if not(i[:2] == '__')]), \
        "Invalid threshold mode"
    result = image.copy()
    if mode == Constants.thr_binary:
        result[result < value] = 0
        result[result >= value] = max_value
    elif mode == Constants.thr_inverse:
        result = max_value - threshold(image, value, max_value, Constants.thr_binary)[1]
    elif mode == Constants.thr_tozero:
        result[result < value] = 0
    elif mode == Constants.thr_tomax:
        result[result < value] = max_value
    elif mode == Constants.thr_otsu:
        hist = histogram(result)
        max_icv = 0
        for i in range(255):
            w_0 = w_1 = u_0 = u_1 = 0
            for j in range(i):
                w_0 += hist[j]
                u_0 += j * hist[j]
            for j in range(i, 255):
                w_1 += hist[j]
                u_1 += j * hist[j]
            icv = w_0 * w_1 * ((u_0/w_0 - u_1/w_1)**2) if (w_0 != 0) & (w_1 != 0) else 0
            if max_icv < icv:
                value = i
                max_icv = icv
    return value, result

def countNonZero():
    pass

def dilate():
    pass

def erode():
    pass

def opening():
    pass

def closing():
    pass

def getShapedKernel():
    pass

# Filtering

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
    