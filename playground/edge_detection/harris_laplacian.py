import numpy as np
import PIL.Image
from scipy.signal import convolve2d

from playground.common import load_lena
from playground.edge_detection.base import Operator
from playground.edge_detection.harris import HarrisOperator


LAPLACIAN_KERNEL = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
])

SUM_FILTER = np.tile([1, 1, 1], 3).reshape(3, 3)
MEAN_FILTER = (1/9) * SUM_FILTER



class HarrisLaplacianOperator(Operator):
    SCALES = [1.2, 1.4, 2.5, 3.6]

    def __init__(self):
        self.harris = HarrisOperator()

    def __call__(self, input_array: np.ndarray, verbose: int = 0, **kwargs) -> np.ndarray:
        # WIP still
        multiscale_outputs = []
        for scale in self.SCALES:
            scaled = convolve2d(input_array, scale * LAPLACIAN_KERNEL, mode="same")
            # multiscale_outputs.append(scaled)
            multiscale_outputs.append(self.harris(scaled))
        multiscale_outputs = np.dstack(multiscale_outputs)
        multiscale_outputs = np.max(multiscale_outputs, axis=-1)
        multiscale_outputs = 255 * multiscale_outputs / multiscale_outputs.max()
        if verbose:
            # Horizontally stack images, using same convolution to match shapes
            PIL.Image.fromarray(multiscale_outputs).show()
        return multiscale_outputs


def lena_grayscale():
    lena = load_lena().convert('L')
    img = np.array(lena)
    harris = HarrisLaplacianOperator()
    out = harris(img, verbose=1)


if __name__ == "__main__":
    lena_grayscale()
