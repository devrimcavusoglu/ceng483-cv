import os
import pickle
from enum import Enum
from typing import Dict, Any, Tuple

import PIL.Image
import numpy as np


class RGBChannel(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


def pillow2numpy(img: PIL.Image.Image) -> np.ndarray:
    # noinspection PyTypeChecker
    return np.array(img)


def numpy2pillow(img: np.ndarray) -> PIL.Image.Image:
    return PIL.Image.fromarray(img)


def vnormalize(v: np.ndarray):
    return v / np.linalg.norm(v)


def read_image(path: str) -> np.ndarray:
    img = PIL.Image.open(path).convert('RGB')
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


if __name__ == "__main__":
    from pathlib import Path

    p = r"D:\lab\projects\ceng483-cv\the1\dataset\query_1\Acadian_Flycatcher_0016_887710060.jpg"
    export_dir = Path(r"D:\lab\projects\ceng483-cv\the1\dataset\test")
    img = read_image(p)
    grid = (1, 1)
    crops = apply_grid(img, grid)
    print("Image shape:", img.shape)
    for i, crop in enumerate(crops):
        print(crop.shape)
        pil_crop = numpy2pillow(crop)
        pil_crop.save(export_dir / f"crop_{i}.jpg")
