from enum import Enum

import PIL.Image
import numpy as np


class RGBChannel(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


def read_image(path: str) -> np.ndarray:
    img = PIL.Image.open(path)
    return pillow2numpy(img)


def pillow2numpy(img: PIL.Image.Image) -> np.ndarray:
    return np.array(img)
