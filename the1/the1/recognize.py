from pathlib import Path
from typing import Union

from the1.color_histogram import ColorHistogram3D, ColorHistogramPerChannel


class InstanceRecognizer:
    def __init__(self, color_histogram: Union[ColorHistogram3D, ColorHistogramPerChannel]):
        self.color_histogram = color_histogram

    def predict(self, query_path: Union[str, Path]):
        if isinstance(query_path, str):
            query_path = Path(query_path)
        queries = []





