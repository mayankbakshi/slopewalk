"""Measures how long the widgets take to redraw after a slider move, for the example networks.

Run from the repository root:  python tools/measure_redraw.py [repeats]

For each case it builds the widgets without a front end (display does nothing), then moves the slider of w1 back
and forth and times each move: drawing every panel, one draw of the canvas, and the PNG sent to the image widget.
It prints the median and the largest time in milliseconds. "still" is the time of a still picture (a new pyplot
figure, saved as PNG at 100 dpi with a tight bounding box, as the inline backend shows it).
"""

import contextlib
import io
import os
import statistics
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from render_examples import CASES  # noqa: E402

from slopewalk import explore, xor_centres  # noqa: E402


def wide():
    torch.manual_seed(0)
    return (nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 1)), *xor_centres())


TIMED = {
    "xor centres, 13 weights": CASES["2_xor_centres"][0],
    "xor clusters": CASES["3_xor_clusters"][0],
    "two hidden layers": CASES["5_two_hidden"][0],
    "49 weights": wide,
}


def live_times(build, repeats):
    import IPython.display

    IPython.display.display = lambda *a, **k: None
    os.environ.pop("NB_STATIC", None)
    net, X, y, groups = build()
    s = explore(net, X, y, groups)
    slider = s.controller.sliders[0]
    start = slider.value
    times = []
    for i in range(repeats):
        t0 = time.perf_counter()
        slider.value = start + (0.25 if i % 2 == 0 else 0.0)
        times.append(1000 * (time.perf_counter() - t0))
    return times


def still_times(build, repeats):
    os.environ["NB_STATIC"] = "1"
    plt.show = lambda *a, **k: None
    net, X, y, groups = build()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):  # explore prints a line with every still picture
            s = explore(net, X, y, groups)
        s.fig.savefig(io.BytesIO(), format="png", dpi=100, bbox_inches="tight")
        times.append(1000 * (time.perf_counter() - t0))
        plt.close("all")
    os.environ.pop("NB_STATIC", None)
    return times


def main(repeats=12):
    print(f"{'case':26s} {'slider move: median':>20s} {'max':>6s}   {'still picture: median':>22s}")
    for name, build in TIMED.items():
        live = live_times(build, repeats)
        still = still_times(build, max(3, repeats // 3))
        print(f"{name:26s} {statistics.median(live):17.0f} ms {max(live):5.0f}   {statistics.median(still):19.0f} ms")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
