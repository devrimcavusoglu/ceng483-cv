import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Literal, List

import numpy as np
from tqdm import tqdm

from the1 import DATASET_CACHE_DIR, SUPPORT_DIR, CACHE_FILENAME
from the1.evaluate import js_divergence
from the1.utils import read_image, RGBChannel, write_pickle, read_pickle


class ColorHistogram(ABC):
    _max_value: int = 255
    _min_value: int = 0

    def __init__(self, grid: Optional[Tuple[int, int]] = (1, 1), bin_size: Optional[int] = 1):
        if any([g <= 0 for g in grid]):
            raise ValueError("`grid` must contain positive values only.")
        elif len(grid) == 1:
            grid = grid[0], grid[0]  # if integer, grid is assumed to be square
        elif len(grid) == 3:
            raise ValueError("`grid` must be tuple of 2 (n_rows, n_cols).")
        self.grid = tuple(grid)
        self.bin_size = bin_size
        self.cache_dir = DATASET_CACHE_DIR / self.__class__.__name__
        self.cache_filepath = self.cache_dir / CACHE_FILENAME
        self._setup(bin_size)

    @property
    def range(self):
        return self._max_value - self._min_value + 1

    def _setup(self, bin_size: int):
        self.nbins = int(np.ceil(self.range / bin_size))
        self.bins = np.arange(self._min_value, self._max_value + 1, bin_size)
        print("Setup completed!")
        print(f"--> bin-size: {bin_size} | nbins: {self.nbins}")

    def cache_dataset(self):
        print("Cleaning up cache dir.")
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        results = {}
        for path in tqdm(SUPPORT_DIR.glob("*"), desc="Caching support set", total=len(list(SUPPORT_DIR.glob("*")))):
            path = str(path)
            embedding = self.compute_from_path(path)
            results[path] = embedding

        self.cache_filepath.parent.mkdir(parents=True)
        write_pickle(results, str(self.cache_filepath))

    def is_dataset_cached(self):
        return self.cache_dir.exists()

    def preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        r, c = self.grid
        crops = []
        h, w, _ = image.shape
        r_size = h // r
        c_size = w // c
        for g_i in range(r):
            for g_j in range(c):
                crop = image[
                       r_size * g_i: r_size * (g_i + 1),
                       c_size * g_j: c_size * (g_j + 1),
                       :
                       ]
                crops.append(crop)
        return crops

    def compute_from_path(self, image_path: str) -> np.ndarray:
        image = read_image(image_path)
        return self.compute_from_image(image)

    def compute_from_image(self, image: np.ndarray):
        crops = self.preprocess(image)
        crop_embeddings = []
        for crop in crops:
            crop_embeddings.append(self.embed(crop, self.bins))
        return np.stack(crop_embeddings)

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

    def evaluate(self, query_dir: Union[str, Path], topk: int = 1):
        if isinstance(query_dir, str):
            query_dir = Path(query_dir)
        support_embeddings = read_pickle(str(self.cache_filepath))

        res = []
        for path in tqdm(query_dir.glob("*"), desc=f"Evaluating query path -> '{query_dir}'", total=len(list(query_dir.glob("*")))):
            score = self.evaluate_single(path, support_embeddings, topk)
            res.append(score)
        overall_score = np.mean(res)
        tqdm.write(f"--> Top-{topk} Accuracy for '{query_dir.name}': %{100*overall_score:.2f}\n")
        time.sleep(0.1)  # Allow tqdm to write stdout properly

    @abstractmethod
    def score(self, q: np.ndarray, s: np.ndarray) -> float:
        raise NotImplementedError("`_evaluate()` is not implemented.")

    @abstractmethod
    def embed(self, image, bins: np.ndarray) -> np.ndarray:
        raise NotImplementedError("`count()` is not implemented.")


class ColorHistogramPerChannel(ColorHistogram):
    def embed(self, image: np.ndarray, bins: np.ndarray) -> np.ndarray:
        channel_counts = []
        m = len(bins)
        for cname in RGBChannel:
            cimg = image[:, :, cname.value]

            # returned indices starts from 1, assuring 0-indexing by subtracting 1
            indices = np.digitize(cimg.ravel(), bins) - 1
            counts = np.bincount(indices, minlength=m)
            channel_counts.append(counts)
        return np.stack(channel_counts)  # (g,c,m) - g: ngrids, c: channel, m: nbins

    def score(self, q: np.ndarray, s: np.ndarray) -> float:
        f = np.vectorize(js_divergence, signature="(n),(n) -> ()", excluded={"normalize"})
        scores = f(q, s)  # (g,c) - g: ngrids, c: channel
        return np.mean(scores).item()


class ColorHistogram3D(ColorHistogram):
    def _setup(self, bin_size: int):
        super()._setup(bin_size)
        if self.nbins > 16:
            raise ValueError("Number of bins must be smaller than 16 (4096 in total) for 3d histogram.")

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
        for n_i in range(niter):
            i, j, k = indices[:, n_i]
            H[i, j, k] += 1
        return H  # (g,m,m,m) - g: ngrids, m: nbins

    def score(self, q: np.ndarray, s: np.ndarray) -> float:
        f = np.vectorize(js_divergence, signature="(n),(n) -> ()", excluded={"normalize"})
        n = q.shape[0]
        q = q.reshape((n, -1))
        s = s.reshape((n, -1))
        scores = f(q, s)  # (g) - g: ngrids
        return np.mean(scores).item()
