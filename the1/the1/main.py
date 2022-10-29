import argparse

from the1 import DATASET_DIR
from the1.color_histogram import ColorHistogram3D, ColorHistogramPerChannel, ColorHistogram


def compute_data_cache(hist: ColorHistogram) -> str:
    cache_dir = hist.get_cache_dir()
    if cache_dir.exists():
        return cache_dir
    else:
        cache_dir.mkdir(parents=True)

    support_set = DATASET_DIR / "support_96"
    hist.cache_dataset(support_set, cache_dir)
    return cache_dir


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=False, default=1, nargs="+", help="Grid to be applied on image before the algorithm.")
    parser.add_argument("--bins", type=int, required=False, default=256, help="Number of bins.")
    parser.add_argument("--type", type=str, choices=["3d", "per-channel"], help="Histogram type to be used.")
    return parser.parse_args()


def main(args):
    if args.type == "3d":
        hist = ColorHistogram3D(**args)
    else:
        hist = ColorHistogramPerChannel(**args)

    # noinspection PyTypeChecker
    cache_dir = compute_data_cache(hist)
    hist.evaluate(cache_dir)


if __name__ == "__main__":
    args = create_args()
    main(args)
