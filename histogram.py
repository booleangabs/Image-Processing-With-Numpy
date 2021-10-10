import numpy as np
import matplotlib.pyplot as plt

class Histogram:
    def __init__(self, image: np.array, normalize: bool= True):
        self.is_norm = normalize
        self.image = image
        self.pixel_count = image.shape[0]*image.shape[1]
        self(image, normalize)
        
    def __call__(self, image: np.array, normalize: bool= True) -> dict:
        self.hist = dict()
        self.cumulative_sum = np.zeros((256, ))
        for i in range(256):
            count = (image == i).sum()
            if normalize:
                count =  np.round(count / self.pixel_count, 4)
            self.hist[i] = count
            self.cumulative_sum[i] += self.cumulative_sum[i - 1] + count
        
    def plot(self, color: str='black'):
        plt.bar(self.hist.keys(), self.hist.values(), width=1, color=color)
        plt.show()
        
    def normalize(self):
        if self.is_norm:
            pass
        else:
            for i in range(256):
                self.hist[i] /= self.pixel_count
            self.is_norm = True
            
        
        
        
        