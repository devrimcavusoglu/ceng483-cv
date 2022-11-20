import enum

import numpy as np
import PIL.Image
from scipy.signal import convolve2d

from playground.edge_detection.base import Operator
from playground.edge_detection.sobel import SobelOperator


GAUSSIAN_KERNEL = np.array([
        [2, 4, 5, 4, 2],
        [4, 9, 12, 9, 4],
        [5, 12, 15, 12, 5],
        [4, 9, 12, 9, 4],
        [2, 4, 5, 4, 2]
])

SUM_FILTER = np.tile([1, 1, 1], 3).reshape(3, 3)
MEAN_FILTER = (1/9) * SUM_FILTER


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


class HarrisOperator(Operator):
    def __init__(self):
        self.sobel = SobelOperator()

    def __call__(self, input_array: np.ndarray, thresh: float = 100, alpha: float = 0.04, nms_filter: int = 3, verbose: int = 0, **kwargs) -> np.ndarray:
        R = self.compute_response(input_array, alpha)
        R_t = self.get_corners(R)
        nms_output = self.local_non_maximum_supression(R_t, filter_size=nms_filter)
        if verbose:
            # Horizontally stack images, using same convolution to match shapes
            PIL.Image.fromarray(np.hstack([input_array, R, R_t, nms_output])).show()
        return nms_output

    def compute_response(self, image, alpha):
        gx, gy = self.sobel(image, return_magnitudes=True)
        gj = gx * gy
        # to find M in a vectorized way, convolve with a 3x3 filter
        # that averages the local region.
        gx = convolve2d(gx, MEAN_FILTER, mode="same")
        gy = convolve2d(gy, MEAN_FILTER, mode="same")
        gj = convolve2d(gj, MEAN_FILTER, mode="same")
        H = np.array([[gx ** 2, gj], [gj, gy ** 2]])  # (2,2,m,n)
        H_det = np.linalg.det(H.transpose((2, 3, 0, 1)))  # (m,n)
        H_trace = np.trace(H)  # (m,n)
        return H_det - alpha * (H_trace ** 2)

    def get_corners(self, R: np.ndarray):
        idx = np.where(R < np.quantile(R[R > 0], 0.95))  # Remove with positive 95% quantile
        R_t = R.copy()
        R_t[idx] = 0
        return R_t

    @staticmethod
    def local_non_maximum_supression(corners: np.ndarray, filter_size: int):
        """
        Local non-maximum supression.

        :param corners:
        :param filter_size:
        :return:
        """
        m, n = corners.shape
        offset = filter_size//2
        corners = np.pad(corners, offset)  # zero pad for convenience
        for i in range(offset, m+offset):
            for j in range(offset, n+offset):
                patch = corners[i - offset:i + offset+1, j - offset:j + offset+1]
                if corners[i, j] < patch.max():
                    corners[i, j] = 0
        return corners[offset:-offset, offset:-offset]  # unpad


if __name__ == "__main__":
    from playground.common import load_fixture

    fixture = "checkerboard.png"
    # fixture = "lena_std.tif"

    pil_img = load_fixture(fixture).convert("L")
    img = np.array(pil_img)
    harris = HarrisOperator()
    g = harris(img, 100, verbose=1, nms_filter=7)
