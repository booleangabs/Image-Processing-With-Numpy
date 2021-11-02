import numpy as np
from image import mapToRange
import warnings
warnings.filterwarnings('ignore')

def negative(image: np.ndarray) -> np.ndarray:
    return 255 - image

def log(image: np.ndarray, c: float=1) -> np.ndarray:
    return mapToRange(c * np.log(1 + image + 1e15), 0, 255)

def gamma(image: np.ndarray, gamma: float, c: float=1) -> np.ndarray:
    return mapToRange(c * ((image + 1e10) ** gamma), 0, 255)