# CV THE-1: Instance Recognition with Color Histograms

Implementation is done under `./the1`, and `the1` is designed as python package. Add `the1` to `PYTHONPATH`, and 
access the help by:

```shell
python the1/main.py --help
usage: main.py [-h] --type {3d,per-channel} [--grid GRID [GRID ...]] [--bs BS] [--topk TOPK] [--multithreaded]

optional arguments:
  -h, --help            show this help message and exit
  --type {3d,per-channel}
                        Histogram type to be used.
  --grid GRID [GRID ...]
                        Grid to be applied on image before the algorithm. If integer, grid is assumed to be square.
  --bs BS               Bin size. Determines the number of bins.
  --topk TOPK           Used to report Top-k Accuracy.
  --multithreaded       Starts each query set evaluation as a separate thread. Thread safe.
```

**NOTE:** If `--multithreaded` flag is passed, evaluation for different query sets are started as different threads, 
and since they share a read-only resource, it's thread safe w.r.t shared resource (cached pickle). However, due to progress bar (using `tqdm`) the stdout writes (top-k accuracy reports) are messy due to race condition :).
