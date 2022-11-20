from typing import Tuple, Union

import numpy as np
from scipy.signal import convolve2d
from PIL import Image

from playground.common import DIFF_1D, GAUSSIAN_1D, load_lena
from playground.edge_detection.base import Operator


class SobelOperator(Operator):
    def __init__(self):
        self.kernel_x = GAUSSIAN_1D.reshape(-1, 1) @ DIFF_1D.reshape(1, -1)
        self.kernel_y = DIFF_1D.reshape(-1, 1) @ GAUSSIAN_1D.reshape(1, -1)

    def __call__(
            self, input_array: np.ndarray, return_magnitudes: bool = False, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        gx = convolve2d(input_array, self.kernel_x, mode="same")
        gy = convolve2d(input_array, self.kernel_y, mode="same")

        if return_magnitudes:
            return gx, gy
        return np.sqrt(gx ** 2 + gy ** 2)


def sobel_grayscale():
    pil_img = load_lena().convert("L")
    img = np.array(pil_img)
    sobel = SobelOperator()
    g = sobel(img)
    # Horizontally stack images, using same convolution to match shapes
    Image.fromarray(np.hstack([img, g])).show()


def sobel_rgb():
    lena = load_lena()
    pil_img = lena.convert('RGB')
    pil_img_l = lena.convert('L')
    img = np.array(pil_img)
    img_l = np.array(pil_img_l)
    sobel = SobelOperator()
    out = []
    for c in range(3):
        conv_out = sobel(img[:, :, c])
        out.append(conv_out)
    out = np.array(out)
    Image.fromarray(np.hstack(out)).show()
    # out = np.dstack(out).mean(axis=-1)
    out = np.dot(out.transpose((1, 2, 0)), [0.2989, 0.5870, 0.1140])
    filtered_img = Image.fromarray(out)
    filtered_img.show()
    sobel_l = sobel(img_l)
    Image.fromarray(sobel_l).show()
    print((sobel_l == out).all())
    print(np.sqrt(np.sum(np.square((sobel_l - out) / 255))))


if __name__ == "__main__":
    sobel_grayscale()
