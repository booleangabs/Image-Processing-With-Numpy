import numpy as np
from image import mapToRange
from filtering import gaussianBlur, convolve

def canny(image: np.ndarray, t0: float, t1: float, blur_size: int=3, sigma: float=5):
    assert blur_size % 2 == 1, f"Gaussian kernel size expected to be odd. Got {blur_size}."
    assert sigma > 0, "Gaussian kernel variance must be bigger than 0."
    assert (0 <= t0 < t1 <= 255), "The following condition must be met: 0 <= t0 < t1 <= 255"
    
    def getPositions(i, j, angle):
        if angle == 0 or angle == 180:
            return [[i, j - 1], [i, j + 1]]
        if angle == 45:
            return [[i - 1, j - 1], [i + 1, j + 1]]
        if angle == 90:
            return [[i - 1, j], [i + 1, j]]
        else:
            return [[i + 1, j - 1], [i - 1, j + 1]]
        
    blurred = gaussianBlur(image, sigma, blur_size)
    kernel = np.zeros((3, 3))
    kernel[:, 1] = np.float32([1, -2, 1])
    grad_x = convolve(blurred, kernel.T, flipped=False)
    grad_y = convolve(blurred, kernel, flipped=False)
    magnitude = np.hypot(grad_x, grad_y)
    magnitude = mapToRange(magnitude, 0, 255)
    orientation = np.arctan2(grad_y, grad_x)
    orientation = np.abs(np.rad2deg(orientation))
    orientation = (orientation // 45) * 45
    
    suppressed = np.zeros_like(image)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            previous_, next_ = getPositions(i, j, orientation[i][j])
            if (magnitude[i][j] >= magnitude[previous_[0]][previous_[1]]) \
                and (magnitude[i][j] >= magnitude[next_[0]][next_[1]]):
                suppressed[i][j] = magnitude[i][j]
            
    double_thresh, result = np.zeros_like(image), np.zeros_like(image)
    double_thresh[suppressed > t1] = 255
    double_thresh[suppressed < t0] = 0
    double_thresh[(suppressed >= t0) & (suppressed <= t1)] = 75
    
    padded = np.pad(double_thresh, ((1, 1), (1, 1)))
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            patch = padded[i:i + 3, j:j + 3] # 8-neighbourhood
            if (patch > 0).any() and double_thresh[i][j] == 75:
                result[i][j] = 255
    return result 
    