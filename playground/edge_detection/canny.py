import enum

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from playground.common import load_lena
from playground.edge_detection.base import Operator
from playground.edge_detection.sobel import SobelOperator


GAUSSIAN_KERNEL = (1/25) * np.array([
        [2, 4, 5, 4, 2],
        [4, 9, 12, 9, 4],
        [5, 12, 15, 12, 5],
        [4, 9, 12, 9, 4],
        [2, 4, 5, 4, 2]
])


class Direction(enum.Enum):
    NORTH = ((0, 1), (2, 1))
    SOUTH = ((0, 1), (2, 1))
    WEST = ((1, 1), (1, 2))
    EAST = ((1, 1), (1, 2))
    NORTHWEST = ((0, 0), (2, 2))
    NORTHEAST = ((0, 2), (2, 0))
    SOUTHWEST = ((0, 2), (2, 0))
    SOUTHEAST = ((0, 0), (2, 2))

    @classmethod
    def from_radians(cls, angle: float):
        t = np.pi / 8
        if angle < t:
            return cls.EAST
        elif angle < 3 * t:
            return cls.NORTHEAST
        elif angle < 5 * t:
            return cls.NORTH
        elif angle < 7 * t:
            return cls.NORTHWEST
        elif angle < 9 * t:
            return cls.WEST
        elif angle < 11 * t:
            return cls.SOUTHWEST
        elif angle < 13 * t:
            return cls.SOUTH
        elif angle < 15 * t:
            return cls.SOUTHEAST
        else:  # 15pi/8 to 2pi
            return cls.EAST


class CannyOperator(Operator):
    def __init__(self, smooth_before_sobel: bool = False):
        self.smooth_before_sobel = smooth_before_sobel
        self.sobel = SobelOperator()

    def __call__(self, input_array: np.ndarray, th: float = 100, tl: float = 50, verbose: int = 0, **kwargs) -> np.ndarray:
        if self.smooth_before_sobel:
            input_array = convolve2d(input_array, GAUSSIAN_KERNEL, mode="same")
        gx, gy = self.sobel(input_array, return_magnitudes=True)
        sobel_output = np.sqrt(gx ** 2 + gy ** 2)
        theta = np.arctan2(gy, gx)
        nms_output = self.non_maximum_supression(sobel_output, theta)
        canny_output = self.hysteresis_thresholding(nms_output, theta, th, tl)
        if verbose:
            # Horizontally stack images, using same convolution to match shapes
            Image.fromarray(np.hstack([input_array, sobel_output, nms_output, canny_output])).show()
        return canny_output

    @staticmethod
    def non_maximum_supression(edges, theta):
        m, n = edges.shape
        edges = np.pad(edges, 1)  # zero pad for convenience
        for i in range(1, m+1):
            for j in range(1, n+1):
                direction = Direction.from_radians(theta[i-1, j-1].item())
                c1, c2 = direction.value  # which indexes to compare (along gradient direction)
                if (
                        edges[i, j] < edges[i+c1[0]-1, j+c1[1]-1]
                        or edges[i, j] < edges[i+c2[0]-1, j+c2[1]-1]
                ):
                    edges[i, j] = 0
        return edges[1:-1, 1:-1]  # unpad

    @staticmethod
    def hysteresis_thresholding(edges, theta, th, tl):
        m, n = edges.shape
        edges = np.pad(edges, 1)  # zero pad for convenience
        for i in range(1, m+1):
            for j in range(1, n+1):
                if edges[i, j] < tl:
                    edges[i, j] = 0
                elif edges[i, j] < th:
                    direction = Direction.from_radians(theta[i-1, j-1].item())
                    # The gradient direction is perpendicular to the edge, so
                    # we need compare the points along the line for continuity
                    if direction in [Direction.NORTH, Direction.SOUTH]:
                        direction = Direction.EAST
                    elif direction in [Direction.EAST, Direction.WEST]:
                        direction = Direction.NORTH
                    elif direction in [Direction.SOUTHEAST, Direction.NORTHWEST]:
                        direction = Direction.NORTHEAST
                    else:
                        direction = Direction.NORTHWEST
                    c1, c2 = direction.value  # which indexes to compare
                    if edges[i+c1[0]-1, j+c1[1]-1] < th or edges[i+c1[0]-1, j+c1[1]-1] < th:
                        edges[i, j] = 0
        return edges[1:-1, 1:-1]  # unpad


def canny_grayscale():
    pil_img = load_lena().convert("L")
    img = np.array(pil_img)
    sobel = SobelOperator()
    canny = CannyOperator()
    g_sobel = sobel(img)
    g = canny(img, 100, 50, verbose=1)


def canny_rgb():
    lena = load_lena()
    pil_img = lena.convert('RGB')
    pil_img_l = lena.convert('L')
    img = np.array(pil_img)
    img_l = np.array(pil_img_l)
    canny = CannyOperator()
    out = []
    for c in range(3):
        conv_out = canny(img[:, :, c])
        out.append(conv_out)
    out = np.array(out)
    Image.fromarray(np.hstack(out)).show()
    Image.fromarray(np.dstack(out), mode="RGB").show()
    # out = np.dstack(out).mean(axis=-1)
    out = np.dot(out.transpose((1, 2, 0)), [0.2989, 0.5870, 0.1140])
    filtered_img = Image.fromarray(out)
    filtered_img.show()
    canny_l = canny(img_l)
    Image.fromarray(canny_l).show()
    print((canny_l == out).all())
    print(np.sqrt(np.sum(np.square((canny_l - out) / 255))))


if __name__ == "__main__":
    canny_grayscale()
