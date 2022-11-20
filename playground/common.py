import numpy as np
from PIL import Image

from playground import FIXTURES_ROOT


GAUSSIAN_1D = np.array([1, 2, 1])
DIFF_1D = np.array([1, 0, -1])


def load_lena() -> Image.Image:
    fp = FIXTURES_ROOT / "lena_std.tif"
    return Image.open(fp)


def load_fixture(path: str) -> Image.Image:
    fp = FIXTURES_ROOT / path
    return Image.open(fp)
