import argparse
import threading

from the1 import QUERY_DIRS
from the1.color_histogram import ColorHistogram3D, ColorHistogramPerChannel


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["3d", "per-channel"], required=True,
                        help="Histogram type to be used.")
    parser.add_argument("--grid", type=int, required=False, default=[1], nargs="+",
                        help="Grid to be applied on image before the algorithm. "
                             "If integer, grid is assumed to be square.")
    parser.add_argument("--bs", type=int, required=False, default=1, help="Bin size. Determines the number of bins.")
    parser.add_argument("--topk", type=int, default=1, help="Used to report Top-k Accuracy.")
    parser.add_argument("--multithreaded", action="store_true",
                        help="Starts each query set evaluation as a separate thread. Thread safe.")
    return parser.parse_args()


def main(args):
    grid = args.grid
    bs = args.bs
    if args.type == "3d":
        hist = ColorHistogram3D(grid, bs)
    else:
        hist = ColorHistogramPerChannel(grid, bs)

    hist.cache_dataset()
    if args.multithreaded:
        threads = []
        for query_dir in QUERY_DIRS:
            threads.append(threading.Thread(target=hist.evaluate, args=(query_dir, args.topk)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    else:
        for query_dir in QUERY_DIRS:
            hist.evaluate(query_dir, topk=args.topk)


if __name__ == "__main__":
    args = create_args()
    main(args)
