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

1. `explore(net, X, y, groups=None, eta=1.0, title=None)`: `X` holds the inputs (n x d), `y` the classes (0 or 1; 0 to K-1 for a network with K outputs, see below). One panel per group of points; without `groups`, one panel per point (up to 8 points) or per class.
2. `clusters(centers, labels, noise=0.0, n_per=50, seed=0)`: the centres themselves (`noise = 0`), or `n_per` points around each centre with standard deviation `noise`. Returns `X, y, groups`.
3. `xor_centres(noise=0.0, n_per=50, seed=0)` and `two_points()`: the built-in examples, as `X, y, groups`. For regression, make the points with torch or numpy, for example `x = torch.linspace(-2, 2, 16)[:, None]` and `y = torch.sin(1.5 * x[:, 0])`.
4. `weight_table(net)`: name, layer, unit, input and value of every weight, in the order of the names.
5. `slopewalk.help()`: the functions with their arguments, the options of `explore` and the methods of the session it returns; `slopewalk.help("explore")` prints one of them in full.
6. The sliders change the weights of `net` itself. After the cell, `net` holds the last weights; to start again, rebuild or reload the network.

Options of `explore`, all keyword-only; without them the picture is as above:

| option | effect |
|---|---|
| `task="regression"` | the output is a prediction and R the mean squared error; `y` holds numbers. Labels other than 0 and 1 without this option raise an error. |
| `params=["w10", "w11"]`, `"output"`, `"layer:1"` | only these weights get sliders and move in a step; the others are drawn lighter and stay fixed. |
| `views=("backprop", "slice", "line", "hidden", "momentum", "cases")` | extra views, see below. |
| `backprop_step=3`, `backprop_case="B"` | the stage of the backward pass drawn, and the group whose case the derivation follows; a step above 0 turns the view on. |
| `case_slopes="w10"` | the per-case panel for this weight. |
| `slice_weights=("w10", "w11")` | the weights of the slice panel (one or two). |
| `beta=0.9`, `lookahead=30` | momentum and the number of steps in the momentum panel. |
| `log_scale=True` | R on a log axis. |
| `layout="wide"`, `"standard"`, `width=27` | the default `"auto"` is wide where the sliders run in Jupyter or VS Code, and standard in Colab (whose notebooks have a fixed width) and for still pictures. Wide: the picture at the full width, with the network panels side by side (up to four) and the lower panels in one row (up to six), each as wide as suits it. Standard: two network panels and three lower panels per row, with larger text on small screens. `width` sets the figure width in inches. |
| `theme="dark"`, `"light"` | the colours of the picture and its controls. The default `"auto"` follows the editor's theme where the kernel can read it: VS Code, and JupyterLab or Notebook 7, on the same machine. Colab and remote kernels do not tell the kernel their theme, so there it is light. `SLOPEWALK_THEME=dark` (or `light`) in the environment chooses for every notebook, and a switch under the controls changes both. |

With the backward pass on, the widgets have buttons to step back and forth through the stages and a picker for the case of the Worked example; the panel headed "Worked example" below the networks gives the formula of the current stage under its heading. In this view the names of the values (z, h, a, p, δ, ∂ℓ/∂h) are written in panel A only; the other panels show the values. `record_backprop("backprop.gif")` writes one frame per stage.

`explore` returns a session: `step(k)` takes k gradient steps (the sliders follow), `step_back()` undoes the last one (and any slider move made after it), `reset()` returns to the start, `predict()` gives the first-order change of R for the next step, `lookahead()` the next steps of gradient descent and of momentum, `case_slopes()` each case's slope along every weight, `save("state.png")` writes the current picture and `record("run.gif", steps=40)` a GIF of a run. `history`, `predictions` and `path` hold R, the predicted and actual change, and the weights after each step.

## What it draws

1. One copy of the network per point or group (red: class 1; blue: class 0; with more classes, teal: class 2 and magenta: class 3). With several points in a group, every value is a mean ± standard deviation over the group.
2. Each weight and bias on its line (line width: its size; dark: positive; orange: negative). The weights are the same for every case, so the first panel carries their numbers and the others show the lines only, marked "weights as in A". Names `w1, w2, ...` on the lines for networks with at most 9 weights; otherwise the values only. A weight moved by hand since the last step has a bold label, and the one moved last keeps its name. Blue is kept for class 0. The slider groups say which inputs each unit's weights belong to.
3. Each unit as pre-activation z, activation function (a small curve; dot: the value of z; shaded: the group's spread) and value h. Then the score a, the sigmoid, p for class 1, and the log loss with its change since the start. With K classes: a score a0, a1, ... per class, one softmax box over them, p for each class, and the cross-entropy; the p of the group's own class has its class colours, since the loss is −log of that p.
4. Below: the boundary p = 0.5 in the input plane (two inputs; dashed: the boundary at the start; one input: p against x1; with K classes, the most likely class over the plane, shaded by how sure the network is, with the boundaries between the classes, or with one input p of each class against x1), R (the average log loss) over the gradient steps taken with the first-order prediction of each step (hollow markers), and a table with each weight's value, the change of R for +0.1 and the slope dR/dw from `R.backward()`.
5. The extra views, in rows of three below:
   - `backprop`: backpropagation step by step, from the output back to the first layer. The stages: δa = ∂ℓ/∂a at the output (p − y; pk − yk for each class k, with yk 1 for the case's class and 0 otherwise; or 2(ŷ − y)); the slopes of the output weights (δ × the input of the line); then for each hidden layer ∂ℓ/∂h (the sum over the next units of weight × δ), δ = act′(z) × ∂ℓ/∂h (with the tangent drawn on the activation curve), and the slopes of that layer's weights; finally the average over the cases, which is the slope column of the table. Each stage adds its values under the boxes of every network panel and fades the boxes and lines it does not use, and a text panel writes out the numbers for one case;
   - `cases`: each case's slope of its own loss along one weight, by group, and their average, which is dR/dw;
   - `slice`: R over one or two weights with the others fixed, the start, the path of the steps and the direction of −∇R;
   - `line`: R along the next step, w − t·∇R, with the tangent line (the first-order prediction) and where the step of eta lands;
   - `hidden`: each case at its values in the last hidden layer, with the output unit's boundary (a line for two units, a plane for three); with K classes, the most likely class over (h1, h2) for two units, and the cases coloured by class otherwise;
   - `momentum`: R over the next steps of gradient descent and of momentum (each step keeps the share beta of the last one).

Under the picture, the status shows R and the change of R in the last action in large type (green when R went down, red when it went up). The last action is a step, ten steps, back one step or a slider move, and the note under the change says which; for steps it adds the first-order prediction. Then come the predicted change for the next step, the size of the gradient and the accuracy. One sentence says what the last action did (for example "w10 changed from 0.49 to 0.70: R went from 0.5680 to 0.5619, down 0.0061.").

Under the status are the controls: one line on what to do, a row with `eta`, `gradient step` (w ← w − eta · dR/dw for every weight), a box with a number of steps (10 at first, 1 to 1000) and the button that takes that many (`10 steps`, its label following the box), `back one step` and `reset` (in a narrow window the four buttons move below eta together), and a slider per weight in a column per unit (a drop-down list and one slider above 30 weights; a slider widens its range when a step takes its weight past the end). Sliders apply when they are released. The buttons are disabled while the picture is drawn, so that clicks do not pile up. Every control has a tooltip. Last come two switches: the layout of the picture (wide or standard) and the colours of the picture and the controls (light, or dark for a dark notebook theme). The controls take the colours of the picture whatever the notebook's own theme. Tabs appear only for what the call asks for:

1. **Backpropagation**, when `views` includes `"backprop"` (or `backprop_step` is above 0): `previous stage` and `next stage` move through the stages (each stops at its end), and `Worked example` chooses the case whose numbers are written out. The formula of the stage is under the heading of the Worked example.
2. **Views**, when other views are asked for: a checkbox per view, with its settings beside it; the settings of a view that is off are disabled.

Changing tabs leaves the picture as it is.

## Supported networks

`nn.Sequential` of `nn.Linear` layers (with bias) with `nn.Tanh`, `nn.ReLU`, `nn.Sigmoid` or nothing between them, ending in `nn.Linear(..., 1)`: its output is the score a of class 1, and R is the cross-entropy, as `F.binary_cross_entropy_with_logits`; with `task="regression"` it is the prediction and R the mean squared error, as `F.mse_loss`. A network that ends in `nn.Linear(..., K)` (K from 2 to 4) has one score per class: the labels are 0 to K-1, and R is the cross-entropy, as `F.cross_entropy`. The task follows from the outputs (`task="multiclass"` for K outputs). The network must be on the CPU in float32 or float64. Anything else, and points with NaN, the wrong shape or labels that do not fit the outputs, raise `ValueError` with the reason.

Weight names: layer by layer, unit by unit, each unit's incoming weights in the order of its inputs, then its bias. For `nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))` this gives w1, w2, w3 (unit 1: x1, x2, bias), w4 to w6 (unit 2), w7 to w9 (unit 3), w10 to w12 (output weights) and w13 (output bias). Rows grow with the number of units up to 12 per layer; beyond that the drawing gets crowded and the values move from the lines to the table.

## Where it runs

1. Google Colab: run `%pip install -q git+https://github.com/mayankbakshi/slopewalk` in the first cell. Colab has ipywidgets.
2. Jupyter or VS Code: install the package and ipywidgets once (`%pip install ipywidgets`, then restart the kernel).
3. Without ipywidgets the cell draws the starting picture and says how to get the sliders. With the environment variable `NB_STATIC=1` (used to make HTML copies of notebooks) it draws the starting picture only.

`examples/slopewalk_demo.ipynb` shows thirteen uses, including the views, regression, a recorded run, backpropagation step by step, the wide layout and three classes.

## Development

    python -m pytest                         # the tests (MPLBACKEND=Agg)
    ruff check . && ruff format --check .    # lint and format
    python tools/make_single_file.py         # rebuilds dist/slopewalk_single.py; --check tells whether it is current
    python tools/render_examples.py out      # the example figures as PNG
    python tools/measure_redraw.py           # how long a slider move takes to redraw

CI (GitHub Actions, Linux) runs the tests with Python 3.9 and the oldest supported versions (torch 1.12.1, numpy 1.21, matplotlib 3.5, ipywidgets 7.6), with Python 3.12 and the versions of Colab's 2026.07 runtime, and with Python 3.13 and the newest releases.
