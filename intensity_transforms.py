import numpy as np

def negative(image: np.ndarray) -> np.ndarray:
    return 255 - image

def log(image: np.ndarray, c: float=1) -> np.ndarray:
    return c * np.log(1 + image)

def gamma(image: np.ndarray, gamma: float, c: float=1) -> np.ndarray:
    return c * (image ** gamma)
