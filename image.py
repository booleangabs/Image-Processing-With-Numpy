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
    def __init__(self, path):
        self.data = ImageReader().read(path)
        self.shape = self.data.shape
        self.dt = self.data.dtype
        
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
    def __init__(self):
        pass
    
    def read(self, path: str) -> np.array:
        from cv2 import imread, cvtColor, COLOR_BGR2RGB
        image = imread(path)
        image = cvtColor(image, COLOR_BGR2RGB)
        return image
    
def threshold(image: Image, value: float, max_value: float, mode: int) -> tuple:
    assert 0 <= mode < len([i for i in dir(Constants) if not(i[:2] == '__')]), \
        "Invalid threshold mode"
    if mode == Constants.thr_binary:
        pass
    elif mode == Constants.thr_inverse:
        pass
    elif mode == Constants.thr_tozero:
        pass
    elif mode == Constants.thr_tomax:
        pass
    else:
        pass
        
    return