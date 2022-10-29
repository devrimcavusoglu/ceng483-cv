from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Tuple, Optional, List

import numpy as np

from the1 import DATASET_CACHE_DIR
from the1.evaluate import js_divergence
from the1.utils import read_image, RGBChannel, write_pickle, read_pickle


class ColorHistogram(ABC):
    _max_value: int = 255
    _min_value: int = 0

    def __init__(self, grid: Optional[Union[int, Tuple[int, int]]] = 1, nbins: Optional[int] = 256):
        self.grid = grid
        self.nbins = nbins

    @classmethod
    def cache_dataset(cls, data_dir: Union[str, Path], cache_dir: Union[str, Path]):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)

        results = {}
        hist = cls()
        for path in data_dir.glob("*"):
            path = str(path)
            embedding = hist.compute_from_path(path)
            results[path] = embedding
        cache_file = cache_dir / "support.pkl"
        write_pickle(results, str(cache_file))

    def get_cache_dir(self):
        return DATASET_CACHE_DIR / self.__class__.__name__

    def preprocess(self, image):
        return image,

    def compute_from_path(self, image_path: str) -> np.ndarray:
        image = read_image(image_path)
        return self.compute_from_image(image)

    def compute_from_image(self, image: np.ndarray):
        bin_size = np.ceil(self._max_value / self.nbins)
        bins = np.arange(self._min_value, self._max_value + 1, bin_size)
        return self.count(image, bins)

    def evaluate(self, query_path: Union[str, Path]):
        if isinstance(query_path, str):
            query_path = Path(query_path)
        cache_dir = self.get_cache_dir()
        cache_file = cache_dir / "support.pkl"
        support_embeddings = read_pickle(str(cache_file))

        res = []
        for path in query_path.glob("*"):
            q_embedding = self.compute_from_path(str(path))
            scores = self._evaluate(q_embedding, list(support_embeddings.values()))
            res.append(scores)
        return res

    @abstractmethod
    def _evaluate(self, q: np.ndarray, s: List[np.ndarray]) -> List:
        raise NotImplementedError("`_evaluate()` is not implemented.")

    @abstractmethod
    def count(self, image, bins: np.ndarray) -> np.ndarray:
        raise NotImplementedError("`count()` is not implemented.")


class ColorHistogramPerChannel(ColorHistogram):
    def count(self, image: np.ndarray, bins: np.ndarray) -> np.ndarray:
        channel_counts = []
        for cname in RGBChannel:
            cimg = image[:, :, cname.value]

            # returned indices starts from 1, assuring 0-indexing by subtracting 1
            indices = np.digitize(cimg.ravel(), bins) - 1
            counts = np.bincount(indices, minlength=len(bins))
            channel_counts.append(counts)
        return np.stack(channel_counts)

    def _evaluate(self, q: np.ndarray, s: List[np.ndarray]) -> List:
        # vectorized_js = np.vectorize(js_divergence, signature='(n),(m)->(k)')
        # return vectorized_js(q, s).item()
        res = []
        for emb in s:
            res.append(js_divergence(q, emb))
        return res


class ColorHistogram3D(ColorHistogram):
    pass


if __name__ == "__main__":
    img_path = r"D:\lab\projects\ceng483-cv\the1\dataset\support_96\Acadian_Flycatcher_0016_887710060.jpg"
    img = ColorHistogramPerChannel()
    print(img.compute_from_path(img_path))
