import numpy as np
from PIL import Image

from playground import FIXTURES_ROOT

GAUSSIAN_1D = np.array([1, 2, 1])
DIFF_1D = np.array([1, 0, -1])


def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def load_lena() -> Image.Image:
    fp = FIXTURES_ROOT / "lena_std.tif"
    return Image.open(fp)


def load_fixture(path: str) -> Image.Image:
    fp = FIXTURES_ROOT / path
    return Image.open(fp)
