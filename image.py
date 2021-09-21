import statistics as stc
import numpy as np
import matplotlib.pyplot as plt

class Image:
    def __init__(self, path):
        self.data = ImageReader.read(path)
        self.shape = self.data.shape
        self.dt = self.data.dtype
        
    def __repr__(self):
        return f"{self.shape[0]}x{self.shape[1]} image - {self.dt}"
    
    def show(self):
        figure = plt.figure()
        axis = plt.add_subplot()
        if len(self.shape) == 2:
            plt.imshow(self.data, cmap='gray')
        else:
            plt.imshow(self.data)
        plt.show()
    
    
class ImageReader:
    def __init__(self):
        pass
    
    def read(self, path: str) -> np.array:
        pass
    
    def __readJpeg(self, path: str) -> np.array:
        pass
    
    def __readPng(self, path: str) -> np.array:
        pass
    