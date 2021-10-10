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