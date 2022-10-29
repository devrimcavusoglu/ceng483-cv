import os
import pickle
from enum import Enum
from typing import Dict, Any

import PIL.Image
import numpy as np


class RGBChannel(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


def read_image(path: str) -> np.ndarray:
    img = PIL.Image.open(path)
    return pillow2numpy(img)


def read_pickle(fp: str) -> Any:
    try:
        with open(fp, "rb") as pkl:
            _obj = pickle.load(pkl)
    except FileNotFoundError:
        return
    else:
        return _obj


def write_pickle(obj: Dict, fp: str, overwrite: bool = True) -> None:
    """Saves a dictionary as json file to given fp."""
    if os.path.exists(fp) and not overwrite:
        raise ValueError(f"Path {fp} already exists. To overwrite, use overwrite=True.")

    with open(fp, "wb") as pkl:
        pickle.dump(obj, pkl)


def pillow2numpy(img: PIL.Image.Image) -> np.ndarray:
    # noinspection PyTypeChecker
    return np.array(img)


def vnormalize(v: np.ndarray):
    return v / np.linalg.norm(v)



