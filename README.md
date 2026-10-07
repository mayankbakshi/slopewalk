# slopewalk: gradient descent by hand on small PyTorch networks

A teaching tool for small networks. You write a small network as in PyTorch and give it a few points. slopewalk draws one copy of the network per point, with every weight on its line, and gives you a slider per weight, the slopes, and the loss. Moving a slider or pressing `gradient step` changes the weights of the network itself.

## Install

Python 3.9 or newer. This also installs numpy, matplotlib, torch and ipywidgets when they are missing.

    pip install git+https://github.com/mayankbakshi/slopewalk

In a notebook (Colab, Jupyter, VS Code): `%pip install -q git+https://github.com/mayankbakshi/slopewalk`, then restart the kernel if the cell asks for it. Without installing: paste `dist/slopewalk_single.py` (the whole package in one file) into a notebook cell.

## Use

```python
import torch.nn as nn
from slopewalk import explore, clusters

net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))      # written as in PyTorch
X, y, groups = clusters([(-1, 1), (1, 1), (-1, -1), (1, -1)], [1, 0, 0, 1])   # the four XOR centres
explore(net, X, y, groups)
```

1. `explore(net, X, y, groups=None, eta=1.0, title=None)`: `X` holds the inputs (n x d), `y` the classes (0 or 1). One panel per group of points; without `groups`, one panel per point (up to 8 points) or per class. Returns a session: `save("state.png")` writes the current picture, `step(k)` takes k gradient steps, `reset()` returns to the start, `weights()` and `history` give the weights and R after each step.
2. `clusters(centers, labels, noise=0.0, n_per=50, seed=0)`: the centres themselves (`noise = 0`), or `n_per` points around each centre with standard deviation `noise`. Returns `X, y, groups`.
3. `xor_centres(noise=0.0, n_per=50, seed=0)` and `two_points()`: the built-in examples, as `X, y, groups`.
4. `weight_table(net)`: name, layer, unit, input and value of every weight, in the order of the names.
5. The sliders change the weights of `net` itself. After the cell, `net` holds the last weights; to start again, rebuild or reload the network.

## What it draws

1. One copy of the network per point or group (red: class 1; blue: class 0). With several points in a group, every value is a mean ± standard deviation over the group.
2. Each weight and bias on its line (line width: its size; orange: negative; blue: changed by you). Names `w1, w2, ...` on the lines for networks with at most 9 weights; otherwise the values only, and the slider groups say which inputs each unit's weights belong to.
3. Each unit as pre-activation z, activation function (a small curve; dot: the value of z; shaded: the group's spread) and value h. Then the score a, the sigmoid, p for class 1, and the log loss with its change since the start.
4. Below: the boundary p = 0.5 in the input plane (two inputs only), R (the average log loss) over the gradient steps taken, and for each weight the change of R for +0.1 and the slope dR/dw from `R.backward()`.

Controls: a slider per weight (a drop-down list and one slider above 30 weights), `eta`, `gradient step` (w ← w − eta · dR/dw for every weight), `10 steps`, `reset`.

## Supported networks

`nn.Sequential` of `nn.Linear` layers with `nn.Tanh`, `nn.ReLU`, `nn.Sigmoid` or nothing between them, ending in `nn.Linear(..., 1)`: its output is the score a of class 1, and R is the cross-entropy, as `F.binary_cross_entropy_with_logits`. Anything else raises `ValueError`.

Weight names: layer by layer, unit by unit, each unit's incoming weights in the order of its inputs, then its bias. For `nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))` this gives w1, w2, w3 (unit 1: x1, x2, bias), w4 to w6 (unit 2), w7 to w9 (unit 3), w10 to w12 (output weights) and w13 (output bias). Networks with more than a few hidden units work but the drawing gets crowded.

## Where it runs

1. Google Colab: run `%pip install -q git+https://github.com/mayankbakshi/slopewalk` in the first cell. Colab has ipywidgets.
2. Jupyter or VS Code: install the package and ipywidgets once (`%pip install ipywidgets`, then restart the kernel).
3. Without ipywidgets the cell draws the starting picture and says how to get the sliders. With the environment variable `NB_STATIC=1` (used to make HTML copies of notebooks) it draws the starting picture only.

`examples/slopewalk_demo.ipynb` shows six uses.

## Development

    python -m pytest                         # the tests (MPLBACKEND=Agg)
    ruff check . && ruff format --check .    # lint and format
    python tools/make_single_file.py         # rebuilds dist/slopewalk_single.py; --check tells whether it is current
    python tools/render_examples.py out      # the example figures as PNG

CI (GitHub Actions, Linux) runs the tests with Python 3.9 and the oldest supported versions (torch 1.12.1, numpy 1.21, matplotlib 3.5, ipywidgets 7.6), with Python 3.12 and the versions of Colab's 2026.07 runtime, and with Python 3.13 and the newest releases.
