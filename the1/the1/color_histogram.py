from abc import ABC, abstractmethod
from collections import Counter
from typing import Union, Tuple, Optional, Dict

import numpy as np

from the1.utils import read_image, RGBChannel


class ColorHistogram(ABC):
    _max_value: int = 255
    _min_value: int = 0

    def __init__(self, image: np.ndarray):
        self._image = image

    @property
    def image(self):
        return self._image

    @classmethod
    def from_path(cls, path: str):
        img = read_image(path)
        return cls(img)

    @abstractmethod
    def count(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("`count()` is not implemented.")

    def compute(self, grid: Optional[Union[int, Tuple[int, int]]] = None, nbins: Optional[int] = 256, **kwargs) -> np.ndarray:
        bin_size = np.ceil(self._max_value / nbins)
        bins = np.arange(self._min_value, self._max_value + 1, bin_size)
        counts = self.count(bins)
        return counts


class ColorHistogram3D(ColorHistogram):
    pass


class ColorHistogramPerChannel(ColorHistogram):
    def _fill_empty(self, count_dict: Dict[int, int]) -> None:
        for bin, count in count_dict.items():
            pass

    def count(self, bins: np.ndarray) -> np.ndarray:
        channel_counts = []
        for cname in RGBChannel:
            cimg = self.image[:, :, cname.value]

            # returned indices starts from 1, assuring 0-indexing by subtracting 1
            indices = np.digitize(cimg.ravel(), bins) - 1
            counts = np.bincount(indices, minlength=self._max_value + 1)
            channel_counts.append(counts)
        return np.stack(channel_counts)


if __name__ == "__main__":
    img_path = r"D:\lab\projects\ceng483-cv\the1\dataset\support_96\Acadian_Flycatcher_0016_887710060.jpg"
    img = ColorHistogramPerChannel.from_path(img_path)
    print(img.compute())
