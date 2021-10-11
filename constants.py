import numpy as np

class Constants:
    '''
    Image reading (ird)
    - ird_color: RGB Image
    - ird_gray: Grayscale Image
    - ird_rgba: RGBA Image
    
    Thresholding (thr)
    - thr_binary: Pixels values <= than threshold turn into 0 otherwise max
    - thr_inverse: Pixels values <= than threshold turn into max otherwise 0
    - thr_tozero: Pixels values <= than threshold turn into 0 otherwise does not change
    - thr_tomax: Pixels values <= than threshold turn into max otherwise does not change
    - thr_otsu: Finds a threshold value maximizing inter-class variance (ICV)
    
    Derivative filters (dvf)
    - sobel_x3: 3x3 Sobel kernel in the x direction
    - sobel_y3: 3x3 Sobel kernel in the y direction
    - sobel_x5: 5x5 Sobel kernel in the x direction
    - sobel_y5: 5x5 Sobel kernel in the y direction
    - sobel_x7: 7x7 Sobel kernel in the x direction
    - sobel_y7: 7x7 Sobel kernel in the x direction
    '''
    # Image reading (ird)
    ird_color = 0
    ird_gray = 1
    ird_rgba = 2
    
    # Thresholding (thr)
    thr_binary = 0
    thr_inverse = 1
    thr_tozero = 2
    thr_tomax = 3
    thr_otsu = 4
    
    # Image morphology (mph)
    mph_square = 0
    mph_circle = 1
    mph_cross = 2
    
    # Derivative filter (dvf)
    dvf = {'sobel_x3': -np.float32([[-1.,  0.,  1.],
                                   [-2.,  0.,  2.],
                                   [-1.,  0.,  1.]]),
           'sobel_x5': -np.float32([[ -1.,  -2.,   0.,   2.,   1.],
                                   [ -4.,  -8.,   0.,   8.,   4.],
                                   [ -6., -12.,   0.,  12.,   6.],
                                   [ -4.,  -8.,   0.,   8.,   4.],
                                   [ -1.,  -2.,   0.,   2.,   1.]]),
           'sobel_x7': -np.float32([[  -1.,   -4.,   -5.,    0.,    5.,    4.,    1.],
                                   [  -6.,  -24.,  -30.,    0.,   30.,   24.,    6.],
                                   [ -15.,  -60.,  -75.,    0.,   75.,   60.,   15.],
                                   [ -20.,  -80., -100.,    0.,  100.,   80.,   20.],
                                   [ -15.,  -60.,  -75.,    0.,   75.,   60.,   15.],
                                   [  -6.,  -24.,  -30.,    0.,   30.,   24.,    6.],
                                   [  -1.,   -4.,   -5.,    0.,    5.,    4.,    1.]])}
    dvf['sobel_y3'] = dvf['sobel_x3'].T
    dvf['sobel_y5'] = dvf['sobel_x5'].T
    dvf['sobel_y7'] = dvf['sobel_x7'].T