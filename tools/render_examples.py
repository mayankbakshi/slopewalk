"""Renders the example figures as still pictures (PNG), for looking at them and for before/after comparisons.

Run from the repository root:  python tools/render_examples.py baseline/phase0

The cases are those of examples/slopewalk_demo.ipynb (built by examples/build_demo.py), plus the XOR network at
step 40 of the reference run (seed 0, eta 0.5, full batch). The tests import `CASES`, `reference_data` and
`reference_net` from here.
"""

import os
import pathlib
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from slopewalk import clusters, explore  # noqa: E402

CENTERS = [(-1, 1), (1, 1), (-1, -1), (1, -1)]  # A, B, C, D: top left, top right, bottom left, bottom right
LABELS = [1, 0, 0, 1]


def reference_data():
    """The XOR data of the reference run: 100 points per centre, noise 0.35, generator seed 0."""
    g = torch.Generator().manual_seed(0)
    centres = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    X = torch.cat([c + 0.35 * torch.randn(100, 2, generator=g) for c in centres])
    y = torch.tensor([0.0, 0.0, 1.0, 1.0]).repeat_interleave(100)
    return X, y


def reference_net(steps=40, eta=0.5, seed=0):
    """The 2-3-1 tanh network after `steps` full-batch gradient steps from the random start `seed`."""
    X, y = reference_data()
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    for _ in range(steps):
        R = F.binary_cross_entropy_with_logits(net(X).squeeze(), y)
        R.backward()
        with torch.no_grad():
            for p in net.parameters():
                p -= eta * p.grad
                p.grad = None
    return net


def two_point():
    net = nn.Sequential(nn.Linear(2, 1))
    with torch.no_grad():
        net[0].weight[:] = torch.tensor([[0.5, 0.5]])
        net[0].bias[:] = 0.0
    return net, [(-1, 1), (1, 1)], [1, 0], None


def xor_centres():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    return (net, *clusters(CENTERS, LABELS))


def xor_clusters():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    return (net, *clusters(CENTERS, LABELS, noise=0.35))


def relu4():
    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    return (net, *clusters(CENTERS, LABELS))


def two_hidden():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2), nn.Tanh(), nn.Linear(2, 1))
    return (net, *clusters(CENTERS, LABELS))


def six_on_a_line():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1))
    return net, [(-2, 0), (-1, 0), (0, 0), (0.5, 0), (1, 0), (2, 0)], [0, 0, 1, 1, 0, 0], None


def xor_step40():
    return (reference_net(40), *clusters(CENTERS, LABELS))


# name -> (builder returning net, X, y, groups; number of panels)
CASES = {
    "1_two_points": (two_point, 2),
    "2_xor_centres": (xor_centres, 4),
    "3_xor_clusters": (xor_clusters, 4),
    "4_relu4": (relu4, 4),
    "5_two_hidden": (two_hidden, 4),
    "6_six_on_a_line": (six_on_a_line, 6),
    "7_xor_step40": (xor_step40, 4),
}


def render_all(outdir, dpi=90):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ["NB_STATIC"] = "1"
    saved = []
    plt.show = lambda *a, **k: None
    for name, (build, _) in CASES.items():
        path = outdir / f"{name}.png"
        net, X, y, groups = build()
        explore(net, X, y, groups)
        plt.gcf().savefig(path, dpi=dpi)
        plt.close("all")
        saved.append(path)
        print("wrote", path)
    return saved


if __name__ == "__main__":
    render_all(sys.argv[1] if len(sys.argv) > 1 else "baseline")
