import argparse
import threading

from the1 import QUERY_DIRS
from the1.color_histogram import ColorHistogram3D, ColorHistogramPerChannel


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["3d", "per-channel"], required=True,
                        help="Histogram type to be used.")
    parser.add_argument("--grid", type=int, required=False, default=1, nargs="+", help="Grid to be applied on image before the algorithm.")
    parser.add_argument("--bs", type=int, required=False, default=1, help="Bin size. Determines the number of bins.")
    # parser.add_argument("--force-recache", action="store_true",
    #                     help="Whether to forcefully cache regardless of cache existence.")
    parser.add_argument("--topk", type=int, default=1)
    return parser.parse_args()


def main(args):
    grid = args.grid
    bs = args.bs
    if args.type == "3d":
        hist = ColorHistogram3D(grid, bs)
    else:
        hist = ColorHistogramPerChannel(grid, bs)

    hist.cache_dataset()
    threads = []
    for query_dir in QUERY_DIRS:
        threads.append(threading.Thread(target=hist.evaluate, args=(query_dir, args.topk)))
        # hist.evaluate(query_dir, topk=args.topk)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    args = create_args()
    main(args)
