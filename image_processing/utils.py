import numpy as np
import matplotlib.pyplot as plt
import image_processing.constants as cts

def show(image: np.ndarray) -> None:
    plt.axis("off")
    if len(image.shape) > 2:
        plt.imshow(image.astype("uint8"))
    else:
        plt.imshow(image, cmap="gray")
    plt.show()

def normalize(image: np.ndarray) -> np.ndarray:
    return (image - image.min()) / (image.max() - image.min())

def map_to_range(image: np.ndarray, low: float, high: float) -> np.ndarray:
    norm = normalize(image)
    return norm * (high - low) + low

def clip_to_range(image: np.ndarray, low: float, high: float) -> np.ndarray:
    result = image.copy()
    result[image < low] = low
    result[image > high] = high
    return result

def split(image: np.ndarray) -> list:
    assert len(image.shape) > 2, "Cannot split single channel images."
    result = []
    for i in range(image.shape[2]):
        result.append(image[..., i])
    return result

def merge(channels: list) -> np.ndarray:
    return np.dstack(channels)
        