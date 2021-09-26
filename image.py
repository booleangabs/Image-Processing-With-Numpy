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
    
class Image:
    def __init__(self, path, mode: str='color'):
        assert mode in ('color', 'gray'), "Please choose a valid mode ('color', 'gray')"
        self.data = ImageReader(mode).read(path)
        self.shape = self.data.shape
        self.dt = self.data.dtype
        self.mode = mode
        
    def __repr__(self):
        return f"{self.shape[0]}x{self.shape[1]} image - {self.dt}"
    
    def show(self):
        figure = plt.figure()
        axis = figure.add_subplot()
        if len(self.shape) == 2:
            plt.imshow(self.data, cmap='gray')
        else:
            plt.imshow(self.data)
        plt.show()
    
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
        return image
    
def threshold(image: Image, value: float, max_value: float, mode: int) -> tuple:
    assert 0 <= mode < len([i for i in dir(Constants) if not(i[:2] == '__')]), \
        "Invalid threshold mode"
    result = image.data.copy()
    if mode == Constants.thr_binary:
        result[result < value] = 0
        result[result >= value] = max_value
    elif mode == Constants.thr_inverse:
        result = max_value - threshold(image, value, max_value, Constants.thr_binary)[1]
    elif mode == Constants.thr_tozero:
        result[result < value] = 0
    elif mode == Constants.thr_tomax:
        result[result < value] = max_value
    else: # Otsu's method
        pass
    return value, result

def histogram(image: Image, normalize: bool= True) -> dict:
    hist = dict()
    total = image.shape[0]*image.shape[1]
    for i in range(255):
        count = (image.data == i).sum()
        if normalize:
            count =  np.round(count / total, 4)
        hist[i] = count
    return hist
        
def plotHistogram(image: Image, normalize: bool= True, color: str= 'black'):
    hist = histogram(image, normalize)
    plt.bar(list(hist.keys()), hist.values(), width=1, color=color)
    plt.plot([30,30],[0,0.025])
    plt.show()