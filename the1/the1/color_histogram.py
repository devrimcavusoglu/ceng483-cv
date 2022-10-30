import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Literal

import numpy as np
from tqdm import tqdm

from the1 import DATASET_CACHE_DIR, SUPPORT_DIR, CACHE_FILENAME
from the1.evaluate import js_divergence
from the1.utils import read_image, RGBChannel, write_pickle, read_pickle


class ColorHistogram(ABC):
    _max_value: int = 255
    _min_value: int = 0

    def __init__(self, grid: Optional[Union[int, Tuple[int, int]]] = 1, nbins: Optional[int] = 256):
        self.grid = grid
        self.nbins = nbins
        self.cache_dir = DATASET_CACHE_DIR / self.__class__.__name__
        self.cache_filepath = self.cache_dir / CACHE_FILENAME

    def cache_dataset(self, force_recache: bool):
        if force_recache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        elif self.is_dataset_cached():
            return
        results = {}
        for path in SUPPORT_DIR.glob("*"):
            path = str(path)
            embedding = self.compute_from_path(path)
            results[path] = embedding

        self.cache_filepath.parent.mkdir(parents=True)
        write_pickle(results, str(self.cache_filepath))

    def is_dataset_cached(self):
        return self.cache_dir.exists()

    def preprocess(self, image):
        return image,

    def compute_from_path(self, image_path: str) -> np.ndarray:
        image = read_image(image_path)
        return self.compute_from_image(image)

    def compute_from_image(self, image: np.ndarray):
        bin_size = np.ceil(self._max_value / self.nbins)
        bins = np.arange(self._min_value, self._max_value + 1, bin_size)
        return self.embed(image, bins)

    def evaluate_single(self, q_path: Path, s: Dict[str, np.ndarray], topk: int) -> Literal[0, 1]:
        q = self.compute_from_path(str(q_path))
        res = {}
        for img in s:
            res[img] = self.score(q, s[img])

        res = list(sorted(res.items(), key=lambda x: x[1]))  # sort by scores
        match = any([Path(res[k][0]).name == q_path.name for k in range(topk)])
        if match:
            return 1
        return 0

    def evaluate(self, query_dir: Union[str, Path], force_recache: bool = False, topk: int = 1):
        self.cache_dataset(force_recache=force_recache)
        if isinstance(query_dir, str):
            query_dir = Path(query_dir)
        support_embeddings = read_pickle(str(self.cache_filepath))

        res = []
        for path in tqdm(query_dir.glob("*"), desc=f"Evaluating query path -> '{query_dir}'", total=len(list(query_dir.glob("*")))):
            score = self.evaluate_single(path, support_embeddings, topk)
            res.append(score)
        overall_score = np.mean(res)
        tqdm.write(f"Top-{topk} Accuracy: %{100*overall_score:.2f}")
        time.sleep(0.01)  # Allow tqdm to write stdout properly

    @abstractmethod
    def score(self, q: np.ndarray, s: np.ndarray) -> float:
        raise NotImplementedError("`_evaluate()` is not implemented.")

    @abstractmethod
    def embed(self, image, bins: np.ndarray) -> np.ndarray:
        raise NotImplementedError("`count()` is not implemented.")


class ColorHistogramPerChannel(ColorHistogram):
    def embed(self, image: np.ndarray, bins: np.ndarray) -> np.ndarray:
        channel_counts = []
        for cname in RGBChannel:
            cimg = image[:, :, cname.value]

            # returned indices starts from 1, assuring 0-indexing by subtracting 1
            indices = np.digitize(cimg.ravel(), bins) - 1
            counts = np.bincount(indices, minlength=len(bins))
            channel_counts.append(counts)
        return np.stack(channel_counts)

    def score(self, q: np.ndarray, s: np.ndarray) -> float:
        rgbscores = []
        for channel in RGBChannel:
            axis = channel.value
            score = js_divergence(q[axis], s[axis])
            rgbscores.append(score)
        return float(np.mean(rgbscores))


class ColorHistogram3D(ColorHistogram):
    def embed(self, image: np.ndarray, bins: np.ndarray) -> np.ndarray:
        m = len(bins)
        H = np.zeros((m, m, m))
        indices = []
        for cname in RGBChannel:
            cimg = image[:, :, cname.value]
            # returned indices starts from 1, assuring 0-indexing by subtracting 1
            inds = np.digitize(cimg.ravel(), bins) - 1
            indices.append(inds)
        indices = np.array(indices)
        niter = indices.shape[1]
        for i in range(niter):
            idx = indices[:, i]
            H[idx[0], idx[1], idx[2]] += 1
        return H

    def score(self, q: np.ndarray, s: np.ndarray):
        return js_divergence(q.ravel(), s.ravel())
