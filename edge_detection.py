import numpy as np
from image import clipToRange, mapToRange
from filtering import gaussianBlur, sobel

def canny(image: np.ndarray, t0: float, t1: float, blur_size: int=3, sigma: float=5):
    assert blur_size % 2 == 1, f"Gaussian kernel size expected to be odd. Got {blur_size}."
    assert sigma > 0, "Gaussian kernel variance must be bigger than 0."
    assert (0 <= t0 < t1 <= 255), "The following condition must be met: 0 <= t0 < t1 <= 255"
    blurred = gaussianBlur(image, sigma, blur_size)
    grad = sobel(blurred)
    grad['magnitude'] = mapToRange(grad['magnitude'], 0, 255)
    orientation = np.rad2deg(grad['orientation'])
    directions = np.array([0, 45, 90, 135])
    orientation = np.digitize(orientation, directions)
    for i in range(len(directions)):
        orientation[orientation == i + 1] = directions[i]
    suppressed = np.zeros_like(image)
    
    padded = np.pad(grad['magnitude'], ((1, 1), (1, 1)))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if orientation[i][j] == 0:
                mags = (padded[i][j - 1], padded[i][j + 1])
            elif orientation[i][j] == 45:
                mags = (padded[i + 1][j - 1], padded[i - 1][j + 1])
            elif orientation[i][j] == 0:
                mags = (padded[i - 1][j], padded[i + 1][j])
            else:
                mags = (padded[i - 1][j - 1], padded[i + 1][j + 1])
            
            if ((grad['magnitude'][i][j] > mags[0]) and (grad['magnitude'][i][j] > mags[1])):
                suppressed[i][j] = grad['magnitude'][i][j]
    
    double_thresh, result = np.zeros_like(image), np.zeros_like(image)
    double_thresh[suppressed > t1] = 255
    double_thresh[suppressed < t0] = 0
    padded = np.pad(double_thresh, ((1, 1), (1, 1)))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded[i:i + 3, j:j + 3] # 8-neighbourhood
            if (patch > 0).any():
                result[i][j] = 255
    return result
    