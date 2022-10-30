import argparse

from the1 import QUERY_DIRS
from the1.color_histogram import ColorHistogram3D, ColorHistogramPerChannel


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["3d", "per-channel"], required=True,
                        help="Histogram type to be used.")
    parser.add_argument("--grid", type=int, required=False, default=1, nargs="+", help="Grid to be applied on image before the algorithm.")
    parser.add_argument("--bins", type=int, required=False, default=256, help="Number of bins.")
    parser.add_argument("--force-recache", action="store_true",
                        help="Whether to forcefully cache regardless of cache existence.")
    parser.add_argument("--topk", type=int, default=1)
    return parser.parse_args()


def main(args):
    grid = args.grid
    nbins = args.bins
    if args.type == "3d":
        if args.bins > 16:
            raise ValueError("Number of bins must be smaller than 16 (4096 in total) for 3d histogram.")
        hist = ColorHistogram3D(grid, nbins)
    else:
        hist = ColorHistogramPerChannel(grid, nbins)

    for query_dir in QUERY_DIRS:
        hist.evaluate(query_dir, force_recache=args.force_recache, topk=args.topk)


if __name__ == "__main__":
    args = create_args()
    main(args)
