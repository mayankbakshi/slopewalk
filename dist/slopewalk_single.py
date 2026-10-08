"""slopewalk 0.1.0.dev0: gradient descent by hand on small PyTorch networks, in one file.

Built by tools/make_single_file.py from the package at https://github.com/mayankbakshi/slopewalk; edit the package,
not this file. Same use as the package:

    net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    X, y, groups = clusters([(-1, 1), (1, 1), (-1, -1), (1, -1)], [1, 0, 0, 1])
    explore(net, X, y, groups)
"""

import io
import math
import os
import re
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import BoxStyle, Rectangle

__version__ = "0.1.0.dev0"


# ------------------------ model.py ------------------------

# activation module -> (name, numpy function, half-width of the curve drawn in the glyph)
ACTIVATIONS = {
    nn.Tanh: ("tanh", np.tanh, 3.0),
    nn.ReLU: ("ReLU", lambda t: np.maximum(t, 0), 3.0),
    nn.Sigmoid: ("sigmoid", lambda t: 1 / (1 + np.exp(-t)), 6.0),
}
SIGMOID = ("sigmoid", lambda t: 1 / (1 + np.exp(-t)), 6.0)
SUPPORTED = "nn.Linear with nn.Tanh, nn.ReLU or nn.Sigmoid between them"
TASKS = ("binary", "regression")
BATCH_CELLS = 400_000  # risk_batch handles at most this many (weight vector, point) pairs at once

Weight = namedtuple("Weight", "name layer unit input value")
# hidden: [(z, h), ...] per hidden layer; a: the score (binary) or the prediction (regression); p: sigmoid(a) for
# binary, None for regression; loss: per point
Values = namedtuple("Values", "hidden a p loss")


class Layer:
    """An nn.Linear and the activation after it: act is (name, f, lim) or None, module the activation module or None."""

    def __init__(self, linear):
        self.linear = linear
        self.act = None
        self.module = None

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features


def parse_layers(net):
    """The layers of net, or ValueError when net is not of the supported kind."""
    mods = list(net) if isinstance(net, nn.Sequential) else [net]
    layers = []
    for m in mods:
        if isinstance(m, nn.Linear):
            layers.append(Layer(m))
        elif type(m) in ACTIVATIONS and layers and layers[-1].act is None:
            layers[-1].act = ACTIVATIONS[type(m)]
            layers[-1].module = m
        elif isinstance(m, nn.Identity):
            continue
        else:
            raise ValueError(f"slopewalk: unsupported layer {m}. Use {SUPPORTED}.")
    if not layers or layers[-1].out_features != 1 or layers[-1].act is not None:
        raise ValueError(
            "slopewalk: the network must end with nn.Linear(..., 1) (the score of class 1, or the prediction for "
            "task='regression'), without an activation."
        )
    for L in layers:
        if L.linear.bias is None:
            raise ValueError(f"slopewalk: {L.linear} has no bias. Use nn.Linear with bias=True.")
    return layers


class Model:
    """The parsed network.

    The weights are one list in the order of their names: layer by layer, unit by unit, each unit's incoming weights
    in the order of its inputs, then its bias. For nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)) this
    gives w1 to w3 (unit 1), w4 to w6, w7 to w9, w10 to w12 (output weights) and w13 (output bias).
    task: "binary" (R is the log loss, as F.binary_cross_entropy_with_logits) or "regression" (R is the mean
    squared error, as F.mse_loss).
    """

    def __init__(self, net, task="binary"):
        if task not in TASKS:
            raise ValueError(f"slopewalk: task must be 'binary' or 'regression', not {task!r}.")
        self.net = net
        self.task = task
        self.layers = parse_layers(net)
        self.linears = [L.linear for L in self.layers]
        tensors = [t for L in self.linears for t in (L.weight, L.bias)]
        if any(t.device.type != "cpu" for t in tensors):
            raise ValueError("slopewalk works on networks on the CPU. Move the network there first: net.cpu().")
        dtypes = {t.dtype for t in tensors}
        if len(dtypes) > 1 or not dtypes <= {torch.float32, torch.float64}:
            found = ", ".join(sorted(str(d).replace("torch.", "") for d in dtypes))
            raise ValueError(
                f"slopewalk: the network's weights are {found}. Use float32 (the default) or float64 for all layers: "
                "net.float() or net.double()."
            )
        self.dtype = tensors[0].dtype
        self.n_inputs = self.layers[0].in_features
        self.hidden = [L.out_features for L in self.layers[:-1]]
        self.n_weights = sum((L.in_features + 1) * L.out_features for L in self.layers)
        self.names = [f"w{k + 1}" for k in range(self.n_weights)]
        # where each weight sits: (layer, unit, input), input -1 for the bias
        self.where = [
            (l, j, i)
            for l, L in enumerate(self.layers)
            for j in range(L.out_features)
            for i in list(range(L.in_features)) + [-1]
        ]

    # ---------------------------------------------------------------- names and places

    def unit_name(self, layer, j):
        """The index written after z and h: "2" with one hidden layer, "1,2" with several."""
        return f"{j + 1}" if len(self.hidden) == 1 else f"{layer + 1},{j + 1}"

    def unit_title(self, layer, j):
        if layer == len(self.layers) - 1:
            return "output unit"
        return f"unit {j + 1}" if len(self.hidden) == 1 else f"layer {layer + 1}, unit {j + 1}"

    def input_labels(self, layer):
        """What feeds the units of a layer: x1, x2, ... for the first, h1, h2, ... (or h1,1, h1,2, ...) after it."""
        n = self.layers[layer].in_features
        if layer == 0:
            return [f"x{i + 1}" for i in range(n)]
        return [f"h{i + 1}" if len(self.hidden) == 1 else f"h{layer},{i + 1}" for i in range(n)]

    def layer_indices(self, layer):
        """The indices of the weights of one layer (from 0)."""
        return [k for k, (l, _, _) in enumerate(self.where) if l == layer]

    def select(self, params):
        """The indices of the weights that move: params None (all), a list of names, a string of names separated by
        commas, "output" (the weights of the output unit) or "layer:k" (the weights of layer k, from 1)."""
        if params is None:
            return list(range(self.n_weights))
        n_layers = len(self.layers)
        if isinstance(params, str):
            text = params.strip()
            if text == "output":
                return self.layer_indices(n_layers - 1)
            m = re.fullmatch(r"layer\s*:\s*(\d+)", text)
            if m:
                k = int(m.group(1))
                if not 1 <= k <= n_layers:
                    raise ValueError(f"slopewalk: params='{text}': this network has layers 1 to {n_layers}.")
                return self.layer_indices(k - 1)
            params = [p.strip() for p in text.split(",") if p.strip()]
        names = list(params)
        unknown = [p for p in names if p not in self.names]
        if unknown or not names:
            raise ValueError(
                f"slopewalk: params {unknown or names} does not name weights of this network. Use names from w1 to "
                f"w{self.n_weights}, 'output', or 'layer:k' with k from 1 to {n_layers}."
            )
        return sorted({self.names.index(p) for p in names})

    # ---------------------------------------------------------------- weights in and out

    def get_weights(self):
        """The weights as Python floats, in the order of their names."""
        return [
            v
            for L in self.linears
            for j in range(L.out_features)
            for v in L.weight[j].detach().tolist() + [L.bias[j].item()]
        ]

    def set_weights(self, w):
        """Writes w (in the order of the names) into the network, in place."""
        k = 0
        with torch.no_grad():
            for L in self.linears:
                m = L.in_features
                for j in range(L.out_features):
                    L.weight[j].copy_(torch.tensor(w[k : k + m], dtype=L.weight.dtype))
                    L.bias[j].fill_(float(w[k + m]))
                    k += m + 1

    def weight_table(self):
        rows, k = [], 0
        w = self.get_weights()
        for layer, L in enumerate(self.layers):
            inputs = self.input_labels(layer) + ["bias"]
            for j in range(L.out_features):
                for name in inputs:
                    rows.append(Weight(self.names[k], layer + 1, j + 1, name, w[k]))
                    k += 1
        return rows

    # ---------------------------------------------------------------- loss, values, R and slopes

    def loss(self, a, y, reduction="mean"):
        """The loss of the task: log loss on the score a (binary) or squared error of the prediction a (regression)."""
        if self.task == "regression":
            return F.mse_loss(a, y, reduction=reduction)
        return F.binary_cross_entropy_with_logits(a, y, reduction=reduction)

    def forward(self, X, y, w=None):
        """Values at w (default: the current weights): (z, h) per hidden layer, then a, p and the loss, per point."""
        if w is not None:
            self.set_weights(w)
        hidden = []
        with torch.no_grad():
            t = X
            for L in self.layers[:-1]:
                z = L.linear(t)
                t = L.module(z) if L.module is not None else z
                hidden.append((z.numpy(), t.numpy()))
            a = self.layers[-1].linear(t).squeeze(1)
            p = torch.sigmoid(a).numpy() if self.task == "binary" else None
            loss = self.loss(a, y, reduction="none")
        return Values(hidden, a.numpy(), p, loss.numpy())

    def risk(self, X, y, w=None):
        """R: the average loss over all points (log loss for binary, squared error for regression)."""
        if w is not None:
            self.set_weights(w)
        with torch.no_grad():
            return float(self.loss(self.net(X).reshape(-1), y))

    def slopes(self, X, y, w=None):
        """dR/dw for every weight, from R.backward(), in the order of the names.

        Leaves the network as it was: the .grad of every parameter and its requires_grad flag are untouched.
        """
        if w is not None:
            self.set_weights(w)
        params = [p for L in self.linears for p in (L.weight, L.bias)]
        flags = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(True)
        try:
            with torch.enable_grad():
                R = self.loss(self.net(X).reshape(-1), y)
                grads = torch.autograd.grad(R, params)
        finally:
            for p, flag in zip(params, flags):
                p.requires_grad_(flag)
        return np.array(
            [
                v
                for gw, gb in zip(grads[0::2], grads[1::2])
                for j in range(gw.shape[0])
                for v in gw[j].tolist() + [gb[j].item()]
            ]
        )

    def risk_batch(self, X, y, W):
        """R for every row of W (G x n_weights), in one batched forward pass; the network is not changed."""
        W = torch.as_tensor(np.asarray(W, dtype=np.float64), dtype=self.dtype)
        if W.dim() == 1:
            W = W[None, :]
        n = X.shape[0]
        chunk = max(1, BATCH_CELLS // max(n, 1))
        out = []
        with torch.no_grad():
            for start in range(0, W.shape[0], chunk):
                Wc = W[start : start + chunk]
                G = Wc.shape[0]
                t = X.expand(G, *X.shape)
                k = 0
                for L in self.layers:
                    m, o = L.in_features, L.out_features
                    block = Wc[:, k : k + o * (m + 1)].reshape(G, o, m + 1)
                    z = torch.einsum("gni,goi->gno", t, block[:, :, :m]) + block[:, None, :, m]
                    t = L.module(z) if L.module is not None else z
                    k += o * (m + 1)
                losses = self.loss(t.reshape(G, n), y.expand(G, n), reduction="none")
                out.append(losses.mean(1))
        return torch.cat(out).numpy().astype(float)

    def case_slopes(self, X, y, w=None):
        """For every case i and weight k, the slope of case i's own loss along w_k, as an n x n_weights array.

        The mean over the cases is dR/dw. One backward pass: the summed losses are differentiated with respect to
        every layer's pre-activations z; since z_i only affects loss_i, this gives each case's slope with respect
        to each unit, and the product with the unit's inputs (and 1 for the bias) gives the slope along each weight.
        The network's parameters, their .grad and their requires_grad flags are not touched.
        """
        if w is not None:
            self.set_weights(w)
        with torch.enable_grad():
            h = X.detach()
            inputs, zs = [], []
            for L in self.layers:
                inputs.append(h)
                z = F.linear(h, L.linear.weight.detach(), L.linear.bias.detach())
                if not z.requires_grad:
                    z.requires_grad_(True)
                zs.append(z)
                h = L.module(z) if L.module is not None else z
            losses = self.loss(h.reshape(-1), y, reduction="none")
            deltas = torch.autograd.grad(losses.sum(), zs)
        n = X.shape[0]
        blocks = []
        for h_in, d in zip(inputs, deltas):
            with_one = torch.cat([h_in.detach(), torch.ones(n, 1, dtype=h_in.dtype)], dim=1)  # inputs, then 1
            blocks.append((d[:, :, None] * with_one[:, None, :]).reshape(n, -1))  # unit by unit
        return torch.cat(blocks, dim=1).detach().numpy().astype(float)

    def accuracy(self, X, y, w=None):
        """The share of cases on the correct side of the boundary (binary only)."""
        if w is not None:
            self.set_weights(w)
        with torch.no_grad():
            a = self.net(X).reshape(-1)
        return float(((a >= 0).to(y.dtype) == y).to(torch.float64).mean())


def weight_table(net):
    """One row per weight: name, layer (from 1), unit (from 1), input ("x1", "h2", "h1,2" or "bias") and value."""
    return Model(net).weight_table()

# ------------------------ data.py ------------------------

XOR_CENTRES = [(-1, 1), (1, 1), (-1, -1), (1, -1)]  # A, B, C, D: top left, top right, bottom left, bottom right
XOR_LABELS = [1, 0, 0, 1]
REGRESSION_GROUPS = 4  # regression with more than 8 points: this many groups of neighbouring points along x1


def clusters(centers, labels, noise=0.0, n_per=50, seed=0):
    """Points for explore: the centres themselves (noise = 0), or n_per points around each centre, with standard
    deviation noise. Returns X (n x d), y (n) and groups (n): the number of the centre of each point."""
    C = torch.tensor(centers, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.float32)
    if noise > 0:
        g = torch.Generator().manual_seed(seed)
        X = torch.cat([c + noise * torch.randn(n_per, C.shape[1], generator=g) for c in C])
        return X, Y.repeat_interleave(n_per), torch.arange(len(C)).repeat_interleave(n_per)
    return C, Y, torch.arange(len(C))


def xor_centres(noise=0.0, n_per=50, seed=0):
    """The four XOR centres A to D (class 1 where x1 and x2 differ in sign), or clusters around them."""
    return clusters(XOR_CENTRES, XOR_LABELS, noise, n_per, seed)


def two_points():
    """A = (-1, 1) of class 1 and B = (1, 1) of class 0: the warm-up without a hidden layer."""
    return clusters([(-1, 1), (1, 1)], [1, 0])


def sine_points(n=16, noise=0.1, seed=0):
    """A regression example with one input: n points on y = sin(1.5 x) for x from -2 to 2, plus Gaussian noise with
    standard deviation noise. Returns X (n x 1), y (n) and groups (n): four groups of neighbouring points along x.
    Use it with task="regression"."""
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(-2, 2, n)
    y = torch.sin(1.5 * x) + noise * torch.randn(n, generator=g)
    return x[:, None], y, torch.arange(n) * REGRESSION_GROUPS // n


class Data:
    """The points of one explore call: X (n x d), y (n), groups (n), and which points belong to each group."""

    def __init__(self, X, y, groups, task="binary"):
        self.X, self.y, self.groups, self.task = X, y, groups, task
        self.n, self.d = X.shape
        self.gids = sorted(set(groups.tolist()))
        self.G = len(self.gids)
        self.gsel = [(groups == g).numpy() for g in self.gids]

    def stats(self, arr, c):
        """Mean and standard deviation over the points of group c of the rows of arr; the sd is 0 for a single point."""
        sel = arr[self.gsel[c]]
        return sel.mean(0), (sel.std(0) if sel.shape[0] > 1 else 0 * sel.mean(0))


def as_numbers(v, name):
    """v as a float64 array, or ValueError naming the argument."""
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    try:
        a = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"slopewalk: {name} must hold numbers: a list of points of equal length, an array or a tensor."
        ) from e
    return a


def check_finite(a, name):
    bad = ~np.isfinite(a)
    if bad.any():
        rows = np.unique(np.nonzero(bad)[0])
        raise ValueError(
            f"slopewalk: {name} contains NaN or inf at {len(rows)} point(s), the first at index {rows[0]}. "
            "Remove those points or replace the values."
        )


def infer_task(y, task):
    """The task: as given, or binary when every label is 0 or 1. Raises ValueError rather than guess."""
    binary = bool(np.isin(y, (0.0, 1.0)).all())
    if task not in (None, "binary", "regression"):
        raise ValueError(f"slopewalk: task must be 'binary' or 'regression', not {task!r}.")
    if task == "regression":
        return task
    if not binary:
        other = y[~np.isin(y, (0.0, 1.0))][0]
        raise ValueError(
            f"slopewalk: y has values other than 0 and 1 (for example {other:g}). For classification the labels must "
            "be 0 or 1; for a numerical target pass task='regression'."
        )
    return "binary"


def prepare(X, y, groups=None, n_inputs=None, dtype=torch.float32, task=None):
    """The checked tensors of an explore call, as a Data object.

    X: n points with n_inputs numbers each (a list of points, an array or a tensor). A flat X is one point, except for
    a network with one input, where it is n points. y: n labels (0 or 1) or targets (task='regression').
    groups None: one group per point up to 8 points; more points: one group per class (binary), or for regression
    four groups of neighbouring points along x1 (one input) or a single group (several inputs).
    """
    Xa = as_numbers(X, "X")
    if Xa.ndim == 0:
        raise ValueError("slopewalk: X must hold at least one point.")
    if Xa.ndim == 1:
        if n_inputs == 1:
            Xa = Xa[:, None]
        elif n_inputs is None or Xa.shape[0] == n_inputs:
            Xa = Xa[None, :]
        else:
            raise ValueError(
                f"slopewalk: X has {Xa.shape[0]} numbers, but the network has {n_inputs} inputs. Give one point as "
                f"{n_inputs} numbers or several points as an n x {n_inputs} array."
            )
    if Xa.ndim != 2:
        raise ValueError(f"slopewalk: X must be an n x d array of points; it has shape {Xa.shape}.")
    n, d = Xa.shape
    if n == 0:
        raise ValueError("slopewalk: X has no points.")
    if n_inputs is not None and d != n_inputs:
        raise ValueError(f"slopewalk: the points of X have {d} inputs, but the network has {n_inputs}.")
    ya = as_numbers(y, "y").reshape(-1)
    if ya.shape[0] != n:
        raise ValueError(f"slopewalk: y has {ya.shape[0]} values for {n} points; give one label per point.")
    check_finite(Xa, "X")
    check_finite(ya, "y")
    task = infer_task(ya, task)
    if groups is None:
        if n <= 8:
            ga = np.arange(n)
        elif task == "binary":
            ga = ya.astype(np.int64)
        elif d == 1:
            order = np.argsort(Xa[:, 0], kind="stable")
            ga = np.empty(n, dtype=np.int64)
            ga[order] = np.arange(n) * REGRESSION_GROUPS // n
        else:
            ga = np.zeros(n, dtype=np.int64)
    else:
        g = as_numbers(groups, "groups").reshape(-1)
        if g.shape[0] != n:
            raise ValueError(f"slopewalk: groups has {g.shape[0]} values for {n} points; give one number per point.")
        if not (np.isfinite(g).all() and (g == np.round(g)).all()):
            raise ValueError("slopewalk: groups must hold one whole number per point.")
        ga = g.astype(np.int64)
    Xt = torch.as_tensor(np.ascontiguousarray(Xa), dtype=dtype)  # reversed or sliced arrays need a copy
    yt = torch.as_tensor(np.ascontiguousarray(ya), dtype=dtype)
    return Data(Xt, yt, torch.as_tensor(np.ascontiguousarray(ga), dtype=torch.int64), task)

# ------------------------ layout.py ------------------------

BW = 0.27  # half-width of a value box; lines end this far from the centre of the box
TABLE_ROWS = 15  # rows per column of the slope table
LABEL_FRACTIONS = np.linspace(0.14, 0.86, 13)  # where along a line a label may sit
LABEL_SHIFTS = (0.0, 0.8, -0.8, 1.6, -1.6, 2.4, -2.4, 3.2, -3.2, 4.0, -4.0, 4.8, -4.8)  # sideways shifts
LABEL_TOLERANCE = 0.02  # a label may overlap other labels or the boxes by at most this fraction of its area
BOTTOM_PER_ROW = 3  # panels per row below the networks
BOTTOM_HEIGHT = 4.2  # height in inches of one row of those panels


class Layout:
    """Positions for a network with n_inputs inputs and the hidden layers `hidden` (a list of unit counts).

    task "binary" ends each panel with a, the sigmoid, p and the loss box; "regression" with the prediction and the
    loss box.
    """

    def __init__(self, n_inputs, hidden, task="binary"):
        self.n_inputs, self.hidden, self.task = n_inputs, hidden, task
        self.blocks, x = [], 1.75
        for _ in hidden:
            self.blocks.append((x, x + 0.75, x + 1.5))  # x of the z box, of the glyph and of the h box
            x += 2.75
        if task == "regression":
            self.x_a, self.x_sigmoid, self.x_p, self.x_loss = x, None, None, x + 1.1
        else:
            self.x_a, self.x_sigmoid, self.x_p, self.x_loss = x, x + 0.75, x + 1.5, x + 2.4
        self.y_inputs = np.linspace(1.95, 0.45, n_inputs) if n_inputs > 1 else np.array([1.2])
        self.xlim = (-0.95, self.x_loss + 0.6)
        self.ylim = (-0.05, 3.0)

    @staticmethod
    def unit_ys(m):
        """y of the m units of one layer, top to bottom."""
        return np.linspace(2.2, 0.2, m) if m > 1 else np.array([1.2])

    @staticmethod
    def label_fraction(si, j, ns, nt):
        """Where along a line (0: source, 1: target) its label sits; staggered so that labels into one unit differ."""
        return 0.5 if ns * nt == 1 else 0.22 + 0.56 * (si * nt + j) / (ns * nt - 1)

    @staticmethod
    def show_names(n_weights):
        """Names and values on the lines for small networks, values only above 9 weights."""
        return n_weights <= 9

    def per_row(self):
        return 2 if self.x_loss + 1.55 <= 9 else 1

    def rows(self, n_panels):
        return -(-n_panels // self.per_row())

    def row_height(self, with_sd=False):
        """Height of a panel row in inches: taller when a layer has more than 3 units (up to four times the usual,
        for 12 units), and by a fifth when the boxes carry a standard deviation line (groups of points)."""
        return 2.9 * min(4.0, max(1.0, max(self.hidden, default=1) / 3)) * (1.2 if with_sd else 1.0)

    @staticmethod
    def bottom_rows(n_bottom):
        return -(-n_bottom // BOTTOM_PER_ROW)

    def figsize(self, n_panels, with_sd=False, n_bottom=BOTTOM_PER_ROW):
        return (15, self.row_height(with_sd) * self.rows(n_panels) + BOTTOM_HEIGHT * self.bottom_rows(n_bottom))

    @staticmethod
    def table_fontsize(n_weights):
        return 9.5 if n_weights > 15 else 11


def overlap_area(a, b):
    """The area shared by two boxes (x0, y0, x1, y1)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def overlap_matrix(A, B):
    """The areas shared by every box of A (m x 4) with every box of B (k x 4), as an m x k array."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    w = np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0])
    h = np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1])
    return np.clip(w, 0, None) * np.clip(h, 0, None)


def place_labels(edges, sizes, obstacles, defaults, pad=0.0):
    """Where each line's label goes, so that labels avoid the boxes and each other.

    edges: [((x0, y0), (x1, y1)), ...]; sizes: [(width, height), ...] of the labels; obstacles: boxes
    (x0, y0, x1, y1) to keep clear of; defaults: the preferred fraction along each edge; all in data units.
    Greedy, edge by edge: of the fractions in LABEL_FRACTIONS, on the line or shifted to either side of it by
    multiples of the label's height, the place with the least overlap with the obstacles and the labels placed so
    far wins, with a slight preference for the default fraction on the line.
    Returns (positions, boxes, worst_labels, worst_boxes): the centre of each label, its box, and the largest
    overlap of any label with other labels and with the obstacles, as fractions of the label's own area.
    """
    obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 4)
    fractions = LABEL_FRACTIONS[:, None]
    shifts = np.asarray(LABEL_SHIFTS, dtype=float)[None, :]
    placed = np.zeros((0, 4))
    positions, boxes, worst_labels, worst_boxes = [], [], 0.0, 0.0
    for (p0, p1), (lw, lh), t0 in zip(edges, sizes, defaults):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        nx, ny = -dy / length, dx / length  # the unit normal of the line
        offset = shifts * lh
        cx = (p0[0] + fractions * dx + offset * nx).ravel()
        cy = (p0[1] + fractions * dy + offset * ny).ravel()
        cand = np.stack([cx - lw / 2 - pad, cy - lh / 2 - pad, cx + lw / 2 + pad, cy + lh / 2 + pad], axis=1)
        with_boxes = overlap_matrix(cand, obstacles).sum(1)
        with_labels = overlap_matrix(cand, placed).sum(1)
        score = with_boxes + with_labels + 1e-4 * (np.abs(fractions - t0) + np.abs(offset)).ravel()
        best = int(np.argmin(score))
        placed = np.vstack([placed, cand[best]])
        positions.append((float(cx[best]), float(cy[best])))
        boxes.append(tuple(float(v) for v in cand[best]))
        area = (lw + 2 * pad) * (lh + 2 * pad)
        worst_labels = max(worst_labels, float(with_labels[best]) / area)
        worst_boxes = max(worst_boxes, float(with_boxes[best]) / area)
    return positions, boxes, worst_labels, worst_boxes

# ------------------------ draw.py ------------------------

class Theme:
    """Colours, fonts and sizes. Red is class 1 and blue class 0.

    Only the labels on the weight lines use edge_label_size, and only the weight lines use zorder 1.
    """

    class_colour = {0: "blue", 1: "red"}
    class_tint = {0: "#E5ECFF", 1: "#FFE5E5"}
    mixed_colour, mixed_tint = "black", "#EEEEEE"
    neutral_colour, neutral_tint = "0.15", "#EEEEEE"  # regression panels
    positive, negative, negative_text, changed = "black", "tab:orange", "#B35900", "tab:blue"
    frozen_alpha, frozen_text, frozen_fill = 0.35, "0.55", "#F5F5F5"  # weights that do not move (params)
    hidden = "green"  # the h boxes, the dot and the band on the glyphs, the history line
    arrow, curve, node_edge = "0.45", "0.35", "gray"
    start_colour, prediction_colour = "0.35", "0.45"
    gd_colour, momentum_colour = "green", "#B9852B"
    value_box = dict(boxstyle="round,pad=0.25", fc="#F2F2F2", ec="0.7", lw=0.8)
    edge_label_size, glyph_name_size, value_size, node_size, title_size, suptitle_size = 8.5, 8, 9.5, 10, 11.5, 13
    note_size = 9
    plane_cmap, points_cmap, value_cmap = "RdBu_r", "bwr", "viridis"
    table_size, table_title_size, table_row_height = 10, 10.5, 0.058  # row height as a fraction of the panel
    table_edge, table_head, table_highlight = "0.8", "#F2F2F2", "#FFF1D6"


THEME = Theme()


def letter(c):
    """A, B, C, ... for the groups."""
    return chr(65 + c) if c < 26 else str(c)


def num(m, sd, sign=True):
    """A value, with its standard deviation on a second line when it is not zero."""
    s = f"{m:+.2f}" if sign else f"{m:.2f}"
    return s + (f"\n±{sd:.2f}" if np.any(sd > 0) else "")


def glyph(ax, xc, yc, act, m, sd, T, height=0.5):
    """The activation function as a small curve; a dot at m and, for groups, a band from m - sd to m + sd.
    height: of the box, in data units (smaller when the units of a layer are close together).

    Returns the box (x0, y0, x1, y1) it occupies, including its name below.
    """
    name, f, lim = act
    w_, h_ = 0.62, height
    ax.add_patch(Rectangle((xc - w_ / 2, yc - h_ / 2), w_, h_, fc="white", ec="0.75", lw=0.8, zorder=2))
    u = np.linspace(-lim, lim, 100)

    def fx(t):
        return xc - w_ / 2 + (np.clip(t, -lim, lim) + lim) / (2 * lim) * w_

    vals = f(u)
    g = (vals - vals.min()) / (vals.max() - vals.min()) - 0.5
    ax.plot(fx(u), yc + 0.85 * h_ * g, color=T.curve, lw=1.2, zorder=3)
    if sd > 0:
        band = Rectangle(
            (fx(m - sd), yc - h_ / 2), max(fx(m + sd) - fx(m - sd), 0.01), h_, fc=T.hidden, alpha=0.18, ec="none"
        )
        band.set_zorder(2)
        ax.add_patch(band)
    ym = yc + 0.85 * h_ * ((f(np.clip(m, -lim, lim)) - vals.min()) / (vals.max() - vals.min()) - 0.5)
    ax.plot([fx(m)], [ym], "o", color=T.hidden, ms=4, zorder=4)
    ax.text(xc, yc - h_ / 2 - 0.04, name, fontsize=T.glyph_name_size, ha="center", va="top", color=T.curve)
    return (xc - w_ / 2, yc - h_ / 2 - 0.16, xc + w_ / 2, yc + h_ / 2)


def data_per_pixel(ax):
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0, 0), (1, 1)])
    return x1 - x0, y1 - y0


def get_renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        return None


def text_size(ax, renderer, text, fontsize):
    """Width and height of a one-line text in data units of ax."""
    if renderer is not None:
        w, h, d = renderer.get_text_width_height_descent(text, FontProperties(size=fontsize), ismath=False)
        h += d
    else:  # no renderer (some backends): a rough estimate
        px = fontsize * ax.figure.dpi / 72
        w, h = 0.6 * px * len(text), 1.2 * px
    sx, sy = data_per_pixel(ax)
    return w * sx, h * sy


def box_obstacles(ax, renderer, skip_size):
    """The boxes of the texts drawn so far (nodes, value boxes), in data units, except texts of size skip_size."""
    if renderer is None:
        return []
    inv = ax.transData.inverted()
    boxes = []
    for t in ax.texts:
        patch = t.get_bbox_patch()
        if patch is None or t.get_fontsize() == skip_size:
            continue
        bb = t.get_window_extent(renderer)
        pad = (
            (0.65 if isinstance(patch.get_boxstyle(), BoxStyle.Circle) else 0.35)
            * t.get_fontsize()
            * ax.figure.dpi
            / 72
        )
        (x0, y0), (x1, y1) = inv.transform([(bb.x0 - pad, bb.y0 - pad), (bb.x1 + pad, bb.y1 + pad)])
        boxes.append((x0, y0, x1, y1))
    return boxes


def group_style(S, c):
    """(colour, tint, mixed, label) of group c: the class colours, black for mixed classes, grey for regression."""
    D, T = S.data, S.theme
    if D.task == "regression":
        return T.neutral_colour, T.neutral_tint, False, None
    labs = D.y[D.gsel[c]]
    yi = int(round(float(labs.mean())))
    mixed = bool(labs.min() != labs.max())
    if mixed:
        return T.mixed_colour, T.mixed_tint, True, yi
    return T.class_colour[yi], T.class_tint[yi], False, yi


def weight_colour(T, k, w, changed, frozen):
    if frozen:
        return T.positive if w[k] >= 0 else T.negative
    return T.changed if k in changed else (T.positive if w[k] >= 0 else T.negative)


def draw_edge_labels(ax, S, F, edges, obstacles):
    """The labels on the lines: names and values for small networks, values only for larger ones, none when
    even the values cannot be placed without overlaps (the values stay on the sliders and in the table).
    Returns the mode used: "names", "values" or "none"."""
    M, L, T = S.model, S.layout, S.theme
    w = F.w
    renderer = get_renderer(ax.figure)
    obstacles = obstacles + box_obstacles(ax, renderer, T.edge_label_size)
    segments = [(p0, p1) for _, p0, p1, _ in edges]
    defaults = [t0 for _, _, _, t0 in edges]
    modes = ["names", "values"] if L.show_names(len(w)) else ["values"]
    for mode in modes:
        labels = [f"{M.names[k]} = {w[k]:.2f}" if mode == "names" else f"{w[k]:.2f}" for k, _, _, _ in edges]
        sizes = [text_size(ax, renderer, s, T.edge_label_size) for s in labels]
        positions, _, worst_labels, worst_boxes = place_labels(segments, sizes, obstacles, defaults, pad=0.03)
        if max(worst_labels, worst_boxes) <= LABEL_TOLERANCE:
            break
    else:
        return "none"
    for (k, _, _, _), label, (cx, cy) in zip(edges, labels, positions):
        frozen = k not in F.moving
        hot = k in F.changed and not frozen
        colour = T.frozen_text if frozen else (T.changed if hot else (T.positive if w[k] >= 0 else T.negative_text))
        ax.text(
            cx,
            cy,
            label,
            fontsize=T.edge_label_size,
            ha="center",
            va="center",
            zorder=3,
            color=colour,
            fontweight="bold" if hot else "normal",
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.9),
        )
    return mode


def draw_case(ax, S, F, c):
    """One copy of the network for group c: the lines with their weights, the units, the output and the loss.

    The labels on the lines are drawn afterwards by draw_edge_labels, once the figure's layout is fixed.
    Returns (edges, obstacles) for them.
    """
    M, D, L, T = S.model, S.data, S.layout, S.theme
    out, w = F.out, F.w
    col, tint, mixed, yi = group_style(S, c)
    arrow = dict(arrowstyle="-|>", color=T.arrow, lw=1.0, shrinkA=0, shrinkB=0)
    node = dict(fontsize=T.node_size, ha="center", va="center", zorder=4)
    value = dict(fontsize=T.value_size, ha="center", va="center", zorder=5, bbox=T.value_box)
    edges, obstacles = [], []  # edges: (k, source, target, preferred label fraction); obstacles: boxes to keep clear
    src = [(0.12, yy) for yy in L.y_inputs]
    src_x, k = 0.0, 0
    for l, layer in enumerate(M.layers):
        last = l == len(M.layers) - 1
        tx = (L.x_a if last else L.blocks[l][0]) - BW
        tys = np.array([1.2]) if last else L.unit_ys(layer.out_features)
        bias = (src_x if l == 0 else src_x + 0.45, 2.8)
        ax.text(*bias, "1", bbox=dict(boxstyle="circle", fc="white", ec=T.node_edge), **node)
        sources = src + [(bias[0] + 0.1, bias[1] - 0.1)]
        ns, nt = len(sources), len(tys)
        for j, ty in enumerate(tys):
            for si, (sx, sy) in enumerate(sources):
                edges.append((k, (sx, sy), (tx, ty), L.label_fraction(si, j, ns, nt)))
                k += 1
        if not last:
            XZ, XG, XH = L.blocks[l]
            glyph_height = min(0.5, 0.75 * (tys[0] - tys[1])) if len(tys) > 1 else 0.5
            (zm, zs), (hm, hs) = D.stats(out.hidden[l][0], c), D.stats(out.hidden[l][1], c)
            for j, ty in enumerate(tys):
                nm = M.unit_name(l, j)
                ax.text(XZ, ty, f"z{nm}\n" + num(zm[j], zs[j]), **value)
                if layer.act:
                    ax.annotate("", xy=(XG - 0.31, ty), xytext=(XZ + BW, ty), arrowprops=arrow)
                    obstacles.append(glyph(ax, XG, ty, layer.act, zm[j], zs[j], T, height=glyph_height))
                    ax.annotate("", xy=(XH - BW, ty), xytext=(XG + 0.31, ty), arrowprops=arrow)
                else:
                    ax.annotate("", xy=(XH - BW, ty), xytext=(XZ + BW, ty), arrowprops=arrow)
                ax.text(XH, ty, f"h{nm}\n" + num(hm[j], hs[j]), color=T.hidden, **value)
            src = [(XH + BW, ty) for ty in tys]
            src_x = XH
    xm, xs = D.stats(D.X.numpy(), c)
    for i, yy in enumerate(L.y_inputs):
        ax.text(0, yy, f"x{i + 1}", color=col, bbox=dict(boxstyle="circle", fc=tint, ec=col), **node)
        ax.text(
            -0.24,
            yy,
            num(xm[i], xs[i]),
            fontsize=T.node_size,
            ha="right",
            va="center",
            color=col,
            bbox=dict(boxstyle="round,pad=0.25", fc=tint, ec=col, lw=0.8),
        )
    (am, as_), (lm, _) = D.stats(out.a, c), D.stats(out.loss, c)
    npts = int(D.gsel[c].sum())
    where = "(" + ", ".join(f"{v:.1f}" for v in xm) + ")"
    lead = f"{letter(c)}: {'mean of ' + str(npts) + ' points ' if npts > 1 else ''}{where}"
    change = f"change\nfrom start\n{lm - S.loss0[c]:+.3f}"
    loss_box = dict(boxstyle="round,pad=0.3", fc=tint, ec=col, lw=0.8)
    if D.task == "regression":
        ax.text(
            L.x_a, 1.2, "ŷ\n" + num(am, as_), fontsize=T.node_size, ha="center", va="center", zorder=5, bbox=T.value_box
        )
        ax.annotate("", xy=(L.x_loss - 0.42, 1.2), xytext=(L.x_a + BW, 1.2), arrowprops=arrow)
        ym, ys = D.stats(D.y.numpy(), c)
        ytxt = f"y = {ym:+.2f}" if npts == 1 else f"y {ym:+.2f}\n±{ys:.2f}"
        ax.text(
            L.x_loss,
            1.2,
            f"{ytxt}\nloss {lm:.3f}\n{change}",
            fontsize=T.node_size,
            ha="center",
            va="center",
            color=col,
            bbox=loss_box,
        )
        title = f"{lead}, y = {ym:+.2f}:  loss {lm:.3f}"
    else:
        pm, ps = D.stats(out.p, c)
        ax.text(
            L.x_a, 1.2, "a\n" + num(am, as_), fontsize=T.node_size, ha="center", va="center", zorder=5, bbox=T.value_box
        )
        ax.annotate("", xy=(L.x_sigmoid - 0.31, 1.2), xytext=(L.x_a + BW, 1.2), arrowprops=arrow)
        obstacles.append(glyph(ax, L.x_sigmoid, 1.2, SIGMOID, am, as_, T))
        ax.annotate("", xy=(L.x_p - BW, 1.2), xytext=(L.x_sigmoid + 0.31, 1.2), arrowprops=arrow)
        ax.text(
            L.x_p,
            1.2,
            "p\n" + num(pm, ps, sign=False),
            fontsize=T.node_size,
            ha="center",
            va="center",
            zorder=5,
            bbox=T.value_box,
        )
        ytxt = "y mixed" if mixed else f"y = {yi}"
        ax.text(
            L.x_loss,
            1.2,
            f"{ytxt}\nloss {lm:.3f}\n{change}",
            fontsize=T.node_size,
            ha="center",
            va="center",
            color=col,
            bbox=loss_box,
        )
        what = "classes mixed" if mixed else f"class {yi}"
        title = f"{lead}, {what}:  loss {lm:.3f}"
    ax.set_title(title, fontsize=T.title_size, color=col)
    ax.set_xlim(*L.xlim)
    ax.set_ylim(*L.ylim)
    ax.axis("off")
    for k, (sx, sy), (tx, ty), _ in edges:
        frozen = k not in F.moving
        ax.plot(
            [sx, tx],
            [sy, ty],
            color=weight_colour(T, k, w, F.changed, frozen),
            lw=min(0.6 + 1.3 * abs(w[k]), 7),
            alpha=T.frozen_alpha if frozen else 1.0,
            zorder=1,
            solid_capstyle="round",
        )
    return edges, obstacles


# ---------------------------------------------------------------- the input plane


def net_on(S, points, w):
    """The network's output (score or prediction) on an array of points at the weights w; leaves the network at w."""
    S.model.set_weights(w)
    with torch.no_grad():
        return S.model.net(torch.as_tensor(points, dtype=S.model.dtype)).reshape(-1).numpy()


def start_values(S, key, points):
    """The network's output on points at the starting weights, computed once per session and key."""
    cache = S.cache.setdefault("start", {})
    if key not in cache:
        cache[key] = net_on(S, points, S.w0)
    return cache[key]


def plane_grid(S):
    X = S.data.X.numpy()
    lo, hi = min(-2.0, float(X.min()) - 0.5), max(2.0, float(X.max()) + 0.5)
    u = np.linspace(lo, hi, 101)
    P1, P2 = np.meshgrid(u, u)
    return P1, P2, np.c_[P1.ravel(), P2.ravel()]


def line_grid(S):
    x = S.data.X.numpy()[:, 0]
    lo, hi = float(x.min()), float(x.max())
    pad = 0.15 * (hi - lo) if hi > lo else 1.0
    return np.linspace(lo - pad, hi + pad, 201)


def draw_group_marks(ax, S, xs, ys, size=160):
    """The group means as large markers with their letters (class colours, or white for regression)."""
    D, T = S.data, S.theme
    for c in range(D.G):
        col, _, _, yi = group_style(S, c)
        face = "white" if D.task == "regression" else T.class_colour[yi]
        ax.scatter(xs[c], ys[c], s=size, color=face, edgecolor="k", zorder=4)
        ax.text(xs[c], ys[c], f"  {letter(c)}", fontsize=12, va="bottom", zorder=5)


def draw_input_plane(ax, S, F):
    """Binary, two inputs: p for class 1 over the plane, the boundary p = 0.5 and the starting boundary (dashed).
    One input: p or the prediction against x1, with the points. Regression, two inputs: the prediction over the plane.
    """
    D, T = S.data, S.theme
    w = F.w
    moved = bool(F.changed)
    X, y = D.X.numpy(), D.y.numpy()
    gx = [D.stats(X, c)[0] for c in range(D.G)]
    if D.d == 1:
        u = line_grid(S)
        a = net_on(S, u[:, None], w)
        out = 1 / (1 + np.exp(-a)) if D.task == "binary" else a
        if moved:
            a0 = start_values(S, "line", u[:, None])
            out0 = 1 / (1 + np.exp(-a0)) if D.task == "binary" else a0
            ax.plot(u, out0, "--", color=T.start_colour, lw=1.0, label="start")
        ax.plot(u, out, color=T.neutral_colour, lw=2, label="p for class 1" if D.task == "binary" else "prediction")
        if D.task == "binary":
            ax.scatter(X[:, 0], y, c=y, cmap=T.points_cmap, vmin=0, vmax=1, s=18, edgecolor="k", lw=0.3, zorder=3)
            ax.axhline(0.5, color="0.7", lw=0.8, zorder=0)
            gy = [float(D.stats(y, c)[0]) for c in range(D.G)]
            ax.set_ylim(-0.08, 1.08)
            ax.set_ylabel("p")
            ax.set_title("p for class 1 against x1")
        else:
            ax.scatter(X[:, 0], y, s=18, color="0.25", zorder=3)
            gy = [float(D.stats(y, c)[0]) for c in range(D.G)]
            ax.set_ylabel("y")
            ax.set_title("prediction and the points")
        draw_group_marks(ax, S, [float(v[0]) for v in gx], gy, size=110)
        ax.set_xlabel("x1")
        if moved:
            ax.legend(fontsize=8, loc="best")
        S.model.set_weights(w)
        return
    if D.d != 2:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{D.d} inputs: no picture of the input plane", ha="center", va="center")
        return
    P1, P2, grid = plane_grid(S)
    a = net_on(S, grid, w).reshape(P1.shape)
    if D.task == "binary":
        P = 1 / (1 + np.exp(-a))
        ax.contourf(P1, P2, P, levels=np.linspace(0, 1, 11), cmap=T.plane_cmap, alpha=0.5)
        if P.min() < 0.5 < P.max():
            ax.contour(P1, P2, P, levels=[0.5], colors="k")
        if moved:
            a0 = start_values(S, "plane", grid).reshape(P1.shape)
            if a0.min() < 0 < a0.max():
                ax.contour(P1, P2, a0, levels=[0.0], colors=T.start_colour, linestyles="--", linewidths=1.0)
        if D.n > D.G:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=T.points_cmap, s=4, vmin=0, vmax=1)
        for c in range(D.G):
            yi = group_style(S, c)[3]
            ax.scatter(*gx[c], s=160, color=T.class_colour[yi], edgecolor="k", zorder=3)
            ax.text(gx[c][0] + 0.15, gx[c][1] + 0.15, letter(c), fontsize=12)
        ax.set_title("p for class 1 and the boundary")
    else:
        lo, hi = float(min(a.min(), y.min())), float(max(a.max(), y.max()))
        levels = np.linspace(lo, hi, 12) if hi > lo else 12
        cs = ax.contourf(P1, P2, a, levels=levels, cmap=T.value_cmap, alpha=0.6)
        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap=T.value_cmap,
            vmin=cs.norm.vmin,
            vmax=cs.norm.vmax,
            s=18,
            edgecolor="k",
            lw=0.4,
            zorder=3,
        )
        for c in range(D.G):
            ax.text(gx[c][0] + 0.15, gx[c][1] + 0.15, letter(c), fontsize=12)
        ax.set_title("prediction over the input plane; points coloured by y")
    ax.set_aspect("equal")
    S.model.set_weights(w)


# ---------------------------------------------------------------- R over the steps


def draw_history(ax, S, F):
    """R over the gradient steps taken, the first-order prediction of each step (hollow markers) and a note on the
    last step and the next one."""
    T = S.theme
    history = F.history
    ax.plot(range(len(history)), history, "o-", color=T.hidden, ms=4, label="R after the step")
    predicted = [(k + 1, p.predicted) for k, p in enumerate(F.predictions)]
    if S.log_scale:
        predicted = [(k, v) for k, v in predicted if v > 0]
    if predicted:
        ks, vs = zip(*predicted)
        ax.plot(ks, vs, "o", mfc="white", mec=T.prediction_colour, ms=6, ls="none", label="first-order prediction")
        ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("gradient steps taken")
    ax.set_ylabel("R")
    values = list(history) + [v for _, v in predicted]
    if S.log_scale:
        positive = [v for v in values if v > 0] or [1.0]
        ax.set_yscale("log")
        ax.set_ylim(min(positive) * 0.7, max(positive) * 1.4)
    else:
        floor = 1.1 if S.task == "binary" else 0.0
        ax.set_ylim(min(0.0, min(values)), max(floor, max(values) * 1.05))
    ax.set_title(f"R = {F.R:.3f}")
    if len(history) < 12:
        ax.set_xticks(range(len(history)))
    notes = [f"next step, eta {F.eta:g}: predicted change {F.next_change:+.4f}"]
    if F.predictions:
        last = F.predictions[-1]
        notes.append(
            f"last step: predicted {last.predicted - last.before:+.4f}, actual {last.actual - last.before:+.4f}"
        )
    ax.text(
        0.98, 0.97, "\n".join(notes), transform=ax.transAxes, ha="right", va="top", fontsize=T.note_size, color="0.25"
    )


# ---------------------------------------------------------------- the slope table


def draw_slope_table(ax, S, F):
    """A table with one row per weight: its value, the change of R when it alone grows by 0.1, and the slope dR/dw.

    The row with the steepest slope among the weights that move is highlighted; the rows of weights that do not move
    are grey. Above 15 rows the table continues in a second block to the right; above 30, both blocks grow taller.
    """
    ax.axis("off")
    T = S.theme
    w, g, deltas = F.w, F.g, F.deltas
    rows = [[S.model.names[k], f"{w[k]:+.2f}", f"{deltas[k]:+.4f}", f"{g[k]:+.3f}"] for k in range(len(w))]
    moving = sorted(F.moving)
    steepest = moving[int(np.argmax(np.abs(g[moving])))]
    per_block = max(TABLE_ROWS, -(-len(rows) // 2))  # at most two blocks, taller ones for many weights
    blocks = [rows[i : i + per_block] for i in range(0, len(rows), per_block)]
    n_blocks = len(blocks)
    fontsize = T.table_size if n_blocks == 1 else max(T.table_size - 2, 8)
    headers = (
        ["weight", "value", "change of R for +0.1", "slope dR/dw"]
        if n_blocks == 1
        else ["w", "value", "R, +0.1", "slope"]
    )
    ax.set_title("Change of R when one weight grows by 0.1,\nand the slope dR/dw", fontsize=T.table_title_size)
    gap = 0.03
    width = (1 - gap * (n_blocks - 1)) / n_blocks
    for b, block in enumerate(blocks):
        height = min(1.0, T.table_row_height * (len(block) + 1))
        tbl = ax.table(
            cellText=block,
            colLabels=headers,
            colWidths=[0.15, 0.18, 0.41, 0.26],
            cellLoc="right",
            colLoc="center",
            bbox=[b * (width + gap), 1 - height, width, height],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(T.table_edge)
            text = cell.get_text()
            if r == 0:
                cell.set_facecolor(T.table_head)
                text.set_fontweight("bold")
                continue
            k = b * per_block + r - 1
            if c > 0:
                text.set_family("monospace")
            if k not in F.moving:
                cell.set_facecolor(T.frozen_fill)
                text.set_color(T.frozen_text)
            elif k == steepest:
                cell.set_facecolor(T.table_highlight)
                text.set_fontweight("bold")

# ------------------------ views.py ------------------------

VIEWS = ("cases", "slice", "line", "hidden", "momentum")


def message(ax, text):
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True)


def point_colours(S):
    """A colour per case: its class colour (binary) or a colour for its target y (regression)."""
    D, T = S.data, S.theme
    y = D.y.numpy()
    if D.task == "binary":
        return [T.class_colour[int(v)] for v in y]
    import matplotlib

    cmap = (
        matplotlib.colormaps[T.value_cmap] if hasattr(matplotlib, "colormaps") else matplotlib.cm.get_cmap(T.value_cmap)
    )
    span = float(y.max() - y.min()) or 1.0
    return [cmap((v - float(y.min())) / span) for v in y]


def moving_slopes(F):
    """The slopes of the weights that move; zero for the others."""
    g = np.zeros(len(F.w))
    idx = sorted(F.moving)
    g[idx] = F.g[idx]
    return g


# ---------------------------------------------------------------- each case's slope


def draw_cases(ax, S, F):
    """One strip per group: each case's slope of its own loss along the chosen weight; a short bar at the group mean;
    a dashed line at the average over all cases, which is the slope dR/dw of the table."""
    D, T = S.data, S.theme
    k = S.case_weight
    name = S.model.names[k]
    v = F.cases[:, k]
    colours = point_colours(S)
    for c in range(D.G):
        idx = np.nonzero(D.gsel[c])[0]
        m = len(idx)
        xs = c + (np.linspace(-0.22, 0.22, m) if m > 1 else np.zeros(1))
        order = np.argsort(v[idx], kind="stable")  # spread the dots along x in the order of their values
        ax.scatter(
            xs,
            v[idx][order],
            s=40 if m <= 8 else 10,
            c=[colours[i] for i in idx[order]],
            edgecolor="k",
            lw=0.3,
            zorder=3,
        )
        ax.hlines(float(v[idx].mean()), c - 0.32, c + 0.32, color="k", lw=2, zorder=4)
    avg = float(v.mean())
    ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)
    ax.axhline(avg, ls="--", color=T.hidden, lw=1.5, zorder=2, label=f"average {avg:+.4f} = dR/d{name}")
    ax.set_xticks(range(D.G))
    ax.set_xticklabels([letter(c) for c in range(D.G)])
    ax.set_xlim(-0.6, D.G - 0.4)
    ax.set_ylabel("slope of the case's own loss")
    formula = "(p - y) da/dw" if D.task == "binary" else "2(ŷ - y) dŷ/dw"
    ax.set_title(f"each case's slope along {name}, {formula};\nbars: group means; dashed: the average", fontsize=10)
    ax.legend(fontsize=8, loc="best")


# ---------------------------------------------------------------- R over one or two weights


def path_points(S, idx):
    """The steps taken, projected on the weights idx, with NaN rows where a new segment starts."""
    P = np.asarray(S.path, dtype=float)
    return P[:, idx]


def window(values, minimum=1.0):
    """A range around the given numbers with a margin."""
    values = np.asarray([v for v in np.ravel(values) if np.isfinite(v)])
    lo, hi = float(values.min()), float(values.max())
    half = max(minimum, 0.65 * (hi - lo))
    centre = 0.5 * (lo + hi)
    return centre - half, centre + half


def draw_slice(ax, S, F):
    """R over the slice weights with the other weights at their current values: contours (two weights) or a curve
    with its tangent (one weight); the start, the path of the steps, the current point and the direction of -grad R.
    With the momentum view on, the next steps of both methods as well."""
    D, M, T = S.data, S.model, S.theme
    idx = list(S.slice_weights)
    names = [M.names[k] for k in idx]
    w = np.asarray(F.w, dtype=float)
    w0 = np.asarray(S.w0, dtype=float)
    path = path_points(S, idx)
    others_move = bool(set(F.moving) - set(idx))
    note = "the other weights at their current values" if others_move or len(F.moving) != len(idx) else ""
    look = F.lookahead
    pts = [w[idx], w0[idx], path]
    if look:
        pts += [look["gd"]["path"][:, idx], look["momentum"]["path"][:, idx]]
    if len(idx) == 1:
        a = idx[0]
        lo, hi = window(np.concatenate([np.ravel(p) for p in pts]))
        u = np.linspace(lo, hi, 161)
        W = np.repeat(w[None, :], len(u), axis=0)
        W[:, a] = u
        R = M.risk_batch(D.X, D.y, W)
        ax.plot(u, R, color="k", lw=1.8, label="R")
        ax.plot(u, F.R + F.g[a] * (u - w[a]), "--", color=T.hidden, lw=1.2, label=f"tangent, slope {F.g[a]:+.3f}")
        ax.plot([w[a]], [F.R], "o", color="k", zorder=5)
        if w0[a] != w[a]:
            ax.axvline(w0[a], color=T.start_colour, ls=":", lw=1, label="start")
        span = float(R.max() - R.min()) or 1.0
        ax.set_ylim(float(R.min()) - 0.1 * span, float(R.max()) + 0.1 * span)
        ax.set_xlabel(names[0])
        ax.set_ylabel("R")
        ax.legend(fontsize=8, loc="best")
        ax.set_title(f"R over {names[0]}" + (f",\n{note}" if note else ""), fontsize=10)
        return
    a, b = idx
    flat = np.concatenate([p.reshape(-1, 2) for p in pts])
    xa, xb = window(flat[:, 0]), window(flat[:, 1])
    n = 41
    ua, ub = np.linspace(*xa, n), np.linspace(*xb, n)
    A, B = np.meshgrid(ua, ub)
    W = np.repeat(w[None, :], n * n, axis=0)
    W[:, a], W[:, b] = A.ravel(), B.ravel()
    R = M.risk_batch(D.X, D.y, W).reshape(n, n)
    levels = np.unique(np.quantile(R, [0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0]))
    if len(levels) > 1:
        ax.contourf(A, B, R, levels=levels, cmap="Greys_r", alpha=0.35)
        ax.contour(A, B, R, levels=levels, colors="0.35", linewidths=0.6)
    if look:
        for key, colour, label in (
            ("gd", T.gd_colour, "next steps, gradient descent"),
            ("momentum", T.momentum_colour, f"next steps, momentum {S.beta:g}"),
        ):
            p = look[key]["path"][:, idx]
            ax.plot(p[:, 0], p[:, 1], "o-", color=colour, ms=2.5, lw=1.2, label=label, zorder=3)
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], "o-", color="0.1", ms=3, lw=1.2, label="steps taken", zorder=4)
    ax.plot([w0[a]], [w0[b]], "o", mfc="white", mec="k", ms=7, label="start", zorder=5)
    ax.plot([w[a]], [w[b]], "o", color="k", ms=6, zorder=6)
    g = np.array([F.g[a], F.g[b]])
    norm = float(np.hypot(*g))
    if norm > 0:
        length = 0.25 * (xa[1] - xa[0])
        d = -g / norm * length
        ax.annotate(
            "",
            xy=(w[a] + d[0], w[b] + d[1]),
            xytext=(w[a], w[b]),
            arrowprops=dict(arrowstyle="-|>", color=T.hidden, lw=1.8),
            zorder=7,
        )
    ax.set_xlim(*xa)
    ax.set_ylim(*xb)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.legend(fontsize=7, loc="best")
    title = f"R over {names[0]} and {names[1]}; arrow: direction of -grad R"
    ax.set_title(title + (f"\n{note}" if note else ""), fontsize=10)


# ---------------------------------------------------------------- R along the next step


def draw_line(ax, S, F):
    """R along the direction of the next step, w - t g (g: the slopes of the weights that move), for t from 0 to
    2.5 eta: the actual R, the tangent line R - t |g|^2 (the first-order prediction), where the step lands at
    t = eta on both, and the lowest point on this line."""
    D, M, T = S.data, S.model, S.theme
    g = moving_slopes(F)
    g2 = float((g**2).sum())
    if g2 == 0:
        message(ax, "every slope is 0:\nno step")
        return
    eta = F.eta
    ts = np.linspace(0, 2.5 * eta, 101)
    w = np.asarray(F.w, dtype=float)
    R = M.risk_batch(D.X, D.y, w[None, :] - ts[:, None] * g[None, :])
    R_eta = float(M.risk_batch(D.X, D.y, (w - eta * g)[None, :])[0])
    pred = F.R - eta * g2
    tangent = F.R - ts * g2
    ax.plot(ts, R, color="k", lw=1.8, label="R along the step")
    ax.plot(ts, tangent, "--", color=T.hidden, lw=1.2, label="first-order prediction")
    ax.axvline(eta, color="0.7", lw=0.8, zorder=0)
    ax.plot([eta], [R_eta], "o", color="k", ms=6, zorder=5, label=f"step of eta: R {R_eta:.4f}")
    ax.plot([eta], [pred], "o", mfc="white", mec=T.hidden, ms=7, zorder=5, label=f"predicted: {pred:.4f}")
    best = int(np.argmin(R))
    if 0 < best < len(ts) - 1:  # a minimum inside the range shown, not R still falling at its end
        ax.plot(
            [ts[best]],
            [R[best]],
            "v",
            color=T.momentum_colour,
            ms=8,
            zorder=5,
            label=f"lowest on the line: t = {ts[best]:.3g}",
        )
    top = max(float(R.max()), F.R)
    bottom = min(float(R.min()), pred)
    span = (top - bottom) or 1.0
    ax.set_ylim(bottom - 0.08 * span, top + 0.08 * span)
    ax.set_xlim(0, ts[-1])
    ax.set_xlabel("t, the step size along -grad R")
    ax.set_ylabel("R")
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"R along the next step, eta = {eta:g}", fontsize=10)


# ---------------------------------------------------------------- the hidden layer


def hidden_width(S):
    return S.model.hidden[-1] if S.model.hidden else 0


def needs_3d(S):
    return "hidden" in S.views and hidden_width(S) == 3


def output_weights(S, w):
    """The output unit's weights for the last hidden layer and its bias."""
    k = S.model.layer_indices(len(S.model.layers) - 1)
    v = np.asarray([w[i] for i in k], dtype=float)
    return v[:-1], float(v[-1])


def draw_hidden(ax, S, F):
    """The cases at the values h of the last hidden layer. The output unit is a linear function of h: its boundary
    a = 0 is a point (one unit), a line (two) or a plane (three). With more units: the score a against the largest
    other direction of h."""
    D, T = S.data, S.theme
    m = hidden_width(S)
    if m == 0:
        message(ax, "no hidden layer:\nthe output unit sees x itself")
        return
    H = np.asarray(F.out.hidden[-1][1], dtype=float)
    v, b = output_weights(S, F.w)
    colours = point_colours(S)
    small = 10 if D.n > 40 else 40
    binary = D.task == "binary"
    gh = [D.stats(H, c)[0] for c in range(D.G)]
    label = "h" if len(S.model.hidden) == 1 else f"h{len(S.model.hidden)},"
    if m == 1:
        h = H[:, 0]
        lo, hi = window(h, minimum=0.6)
        u = np.linspace(lo, hi, 101)
        a = v[0] * u + b
        if binary:
            ax.plot(u, 1 / (1 + np.exp(-a)), color=T.neutral_colour, lw=1.5, label="p = sigmoid(a)")
            ax.scatter(h, D.y.numpy(), c=colours, s=small, edgecolor="k", lw=0.3, zorder=3)
            if v[0] != 0 and lo < -b / v[0] < hi:
                ax.axvline(-b / v[0], color="k", lw=1, label="boundary a = 0")
            ax.set_ylabel("p, and the class of each case")
        else:
            ax.plot(u, a, color=T.neutral_colour, lw=1.5, label="prediction")
            ax.scatter(h, D.y.numpy(), c=colours, s=small, edgecolor="k", lw=0.3, zorder=3)
            ax.set_ylabel("y")
        ax.set_xlabel(f"{label}1")
        ax.legend(fontsize=8, loc="best")
    elif m == 2:
        x0, x1 = window(H[:, 0], minimum=0.6), window(H[:, 1], minimum=0.6)
        U1, U2 = np.meshgrid(np.linspace(*x0, 81), np.linspace(*x1, 81))
        a = v[0] * U1 + v[1] * U2 + b
        if binary:
            P = 1 / (1 + np.exp(-a))
            ax.contourf(U1, U2, P, levels=np.linspace(0, 1, 11), cmap=T.plane_cmap, alpha=0.4)
            if P.min() < 0.5 < P.max():
                ax.contour(U1, U2, P, levels=[0.5], colors="k")
        else:
            ax.contour(U1, U2, a, levels=10, cmap=T.value_cmap, linewidths=0.8)
        ax.scatter(H[:, 0], H[:, 1], c=colours, s=small, edgecolor="k", lw=0.3, zorder=3)
        for c in range(D.G):
            ax.text(gh[c][0], gh[c][1], f"  {letter(c)}", fontsize=11, zorder=4)
        ax.set_xlim(*x0)
        ax.set_ylim(*x1)
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel(f"{label}2")
    elif m == 3 and hasattr(ax, "set_zlim"):
        ax.scatter(H[:, 0], H[:, 1], H[:, 2], c=colours, s=small, edgecolor="k", linewidths=0.3, depthshade=False)
        r = [window(H[:, i], minimum=0.6) for i in range(3)]
        if binary and np.any(v != 0):
            i = int(np.argmax(np.abs(v)))  # solve the plane for the coordinate with the largest weight
            j, k = [q for q in range(3) if q != i]
            Gj, Gk = np.meshgrid(np.linspace(*r[j], 6), np.linspace(*r[k], 6))
            Gi = np.clip(-(v[j] * Gj + v[k] * Gk + b) / v[i], *r[i])
            coords = [None, None, None]
            coords[i], coords[j], coords[k] = Gi, Gj, Gk
            ax.plot_surface(*coords, color="0.5", alpha=0.25, linewidth=0)
        for q in range(3):
            getattr(ax, f"set_{'xyz'[q]}lim")(*r[q])
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel(f"{label}2")
        ax.set_zlabel(f"{label}3")
        ax.view_init(elev=20, azim=-60)
    else:
        a = H @ v + b
        unit = v / (np.linalg.norm(v) or 1.0)
        rest = H - np.outer(H @ unit, unit)
        rest = rest - rest.mean(0)
        _, _, vt = np.linalg.svd(rest, full_matrices=False)
        other = rest @ vt[0]
        ax.scatter(a, other, c=colours, s=small, edgecolor="k", lw=0.3, zorder=3)
        if binary:
            ax.axvline(0.0, color="k", lw=1)
        ax.set_xlabel("a = output unit's score" if binary else "prediction")
        ax.set_ylabel("largest other direction of h")
    what = "the boundary a = 0" if binary else "coloured by y"
    ax.set_title(f"the cases at their hidden values,\n{what}", fontsize=10)


# ---------------------------------------------------------------- gradient descent and momentum


def draw_momentum(ax, S, F):
    """R over the next steps from the current weights: gradient descent and heavy-ball momentum, where each step
    keeps the share beta of the last one."""
    T = S.theme
    look = F.lookahead
    gd, mo = look["gd"]["R"], look["momentum"]["R"]
    steps = range(len(gd))
    ax.plot(steps, gd, color=T.gd_colour, lw=1.8, label=f"gradient descent: R {gd[-1]:.4f}")
    ax.plot(steps, mo, color=T.momentum_colour, lw=1.8, label=f"momentum, beta {S.beta:g}: R {mo[-1]:.4f}")
    if S.log_scale and min(min(gd), min(mo)) > 0:
        ax.set_yscale("log")
    ax.set_xlabel("steps from the current weights")
    ax.set_ylabel("R")
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"the next {len(gd) - 1} steps, eta {F.eta:g}", fontsize=10)


VIEW_DRAWERS = {
    "cases": draw_cases,
    "slice": draw_slice,
    "line": draw_line,
    "hidden": draw_hidden,
    "momentum": draw_momentum,
}

# ------------------------ picture.py ------------------------

Frame = namedtuple("Frame", "w out R g deltas changed moving history predictions eta next_change cases lookahead")
LIVE_DPI = 100  # resolution of the picture shown by the widgets


def make_frame(S, w=None):
    """The numbers every panel needs at the weights w (default: the current ones). Leaves the network at w."""
    M, D = S.model, S.data
    w = S.weights() if w is None else [float(v) for v in w]
    out = M.forward(D.X, D.y, w)
    R = M.risk(D.X, D.y, w)
    g = M.slopes(D.X, D.y, w)
    deltas = []
    for k in range(len(w)):
        wk = list(w)
        wk[k] += 0.1
        deltas.append(M.risk(D.X, D.y, wk) - R)
    M.set_weights(w)
    moving = set(S.active)
    changed = {k for k in range(len(w)) if abs(w[k] - S.w0[k]) > 1e-9}
    next_change = -S.eta * float(sum(g[k] ** 2 for k in moving))
    cases = M.case_slopes(D.X, D.y) if "cases" in S.views else None
    lookahead = S.lookahead(w=w) if "momentum" in S.views else None
    M.set_weights(w)
    return Frame(
        w,
        out,
        R,
        g,
        deltas,
        changed,
        moving,
        list(S.history),
        list(S.predictions),
        S.eta,
        next_change,
        cases,
        lookahead,
    )


def bottom_panels(S):
    """The panels below the networks, in order: the input plane, R over the steps, the table, then the views."""
    return ["plane", "history", "table"] + [v for v in VIEWS if v in S.views]


class Picture:
    """The figure of a session for its current views. refresh(w) draws it at the weights w.

    pyplot True: a pyplot figure (still pictures, shown with plt.show). False: an off-screen figure at dpi, for the
    widgets and for record(), which pyplot never shows.
    """

    def __init__(self, S, pyplot=False, dpi=LIVE_DPI):
        self.S = S
        D, L = S.data, S.layout
        self.panels = bottom_panels(S)
        per_row, rows = L.per_row(), L.rows(D.G)
        with_sd = any(sel.sum() > 1 for sel in D.gsel)  # groups of points: the boxes show mean and standard deviation
        n_bottom = L.bottom_rows(len(self.panels))
        size = L.figsize(D.G, with_sd, len(self.panels))
        if pyplot:
            fig = plt.figure(figsize=size)
        else:
            fig = Figure(figsize=size, dpi=dpi)
            FigureCanvasAgg(fig)
        heights = [L.row_height(with_sd)] * rows + [BOTTOM_HEIGHT] * n_bottom
        gs = fig.add_gridspec(rows + n_bottom, BOTTOM_PER_ROW, height_ratios=heights)
        self.case_axes = []
        for c in range(D.G):
            r, cc = divmod(c, per_row)
            self.case_axes.append(fig.add_subplot(gs[r, :] if per_row == 1 else gs[r, :].subgridspec(1, 2)[0, cc]))
        self.axes = {}
        for i, name in enumerate(self.panels):
            r, cc = divmod(i, BOTTOM_PER_ROW)
            extra = {"projection": "3d"} if name == "hidden" and needs_3d(S) else {}
            self.axes[name] = fig.add_subplot(gs[rows + r, cc], **extra)
        if S.title:
            fig.suptitle(S.title, fontsize=S.theme.suptitle_size)
        self.fig = fig
        self.laid_out = False

    def refresh(self, w=None):
        """Draws every panel at the weights w (default: the current ones) and returns the figure."""
        S = self.S
        F = make_frame(S, w)
        if self.laid_out:
            for ax in self.case_axes + list(self.axes.values()):
                ax.clear()
        parts = [draw_case(ax, S, F, c) for c, ax in enumerate(self.case_axes)]
        draw_input_plane(self.axes["plane"], S, F)
        draw_history(self.axes["history"], S, F)
        draw_slope_table(self.axes["table"], S, F)
        for name in self.panels[3:]:
            VIEW_DRAWERS[name](self.axes[name], S, F)
            S.model.set_weights(F.w)
        if not self.laid_out:
            self.layout()
        for ax, (edges, obstacles) in zip(self.case_axes, parts):
            draw_edge_labels(ax, S, F, edges, obstacles)
        S.model.set_weights(F.w)
        return self.fig

    def layout(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # 3-D axes do not take part in tight_layout
            self.fig.tight_layout()
        self.laid_out = True

    def png(self):
        """The picture as PNG bytes, from one draw of the off-screen canvas."""
        from PIL import Image

        canvas = self.fig.canvas
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        buf = io.BytesIO()
        Image.fromarray(rgba[..., :3]).save(buf, format="png", compress_level=3)
        return buf.getvalue()

# ------------------------ static.py ------------------------

STATIC_MESSAGE = (
    "Still picture of the start. "
    "Run the notebook (Colab, or Jupyter / VS Code with ipywidgets) to get the sliders and buttons."
)
NO_WIDGETS_MESSAGE = (
    "No sliders: the package ipywidgets is missing. "
    "Run  %pip install ipywidgets  in a cell, restart the kernel and run again,\n"
    "or open the notebook in Google Colab, where ipywidgets is installed."
)


def still_picture(session, message):
    """Prints the message and draws the start once. Returns the session."""
    print(message)
    session.draw()
    plt.show()
    return session

# ------------------------ widgets.py ------------------------

VIEW_LABELS = {
    "cases": "each case's slope",
    "slice": "R over two weights",
    "line": "R along the next step",
    "hidden": "the hidden layer",
    "momentum": "momentum",
}


def slider_range(v, lo=-4.0, hi=4.0):
    """The range of a weight's slider: lo to hi, widened on the side that v leaves, to floor(v) - 1 or ceil(v) + 1."""
    if v < lo:
        lo = math.floor(v) - 1.0
    if v > hi:
        hi = math.ceil(v) + 1.0
    return lo, hi


def name_ranges(names):
    """w1, w2, w3, w5 -> "w1 to w3, w5"."""
    nums = sorted(int(n[1:]) for n in names)
    parts, start = [], None
    for i, k in enumerate(nums):
        if start is None:
            start = k
        if i + 1 == len(nums) or nums[i + 1] != k + 1:
            parts.append(f"w{start}" if start == k else f"w{start} to w{k}")
            start = None
    return ", ".join(parts)


def readout_html(r):
    """The readout strip: one small card per number; an increase of R in the last step is shown in red."""

    def card(label, value, colour="#222"):
        return (
            '<div style="border:1px solid #DDD;border-radius:6px;padding:3px 10px;margin:2px 6px 2px 0;'
            f'background:#FAFAFA"><div style="font-size:11px;color:#666">{label}</div>'
            f'<div style="font-size:15px;font-family:monospace;color:{colour}">{value}</div></div>'
        )

    cards = [card("R", f"{r['R']:.4f}"), card("size of the gradient", f"{r['grad_norm']:.4f}")]
    if r["accuracy"] is not None:
        cards.append(card("accuracy", f"{r['accuracy']:.3f}"))
    cards.append(card(f"next step, eta {r['eta']:g}: predicted change of R", f"{r['next_change']:+.4f}"))
    if r["last"] is not None:
        predicted, actual = r["last"]
        colour = "#B00020" if actual > 0 else "#1B6E3A"
        cards.append(card("last step: predicted, actual change", f"{predicted:+.4f}, {actual:+.4f}", colour))
    return '<div style="display:flex;flex-wrap:wrap">' + "".join(cards) + "</div>"


class Controller:
    """Builds the widgets for a session and keeps them, the network and the picture in step."""

    def __init__(self, session, dpi=LIVE_DPI):
        import ipywidgets as widgets

        S = self.session = session
        self.dpi = dpi
        self.busy = False
        self.picture = None
        self.use_output = os.environ.get("SLOPEWALK_DISPLAY") == "output"  # fallback: the PNG through display()
        self.index = list(S.active)  # the weight of each slider
        self.sliders = []
        for k in self.index:
            lo, hi = slider_range(S.w0[k])
            self.sliders.append(
                widgets.FloatSlider(
                    value=S.w0[k],
                    min=lo,
                    max=hi,
                    step=0.001,
                    description=S.names[k],
                    readout_format=".2f",
                    continuous_update=False,
                    layout=widgets.Layout(width="230px"),
                )
            )
        panel = self.grouped_panel(widgets) if len(self.sliders) <= 30 else self.dropdown_panel(widgets)
        frozen = [n for k, n in enumerate(S.names) if k not in set(self.index)]
        if frozen:
            note = widgets.HTML(f"<i>fixed (drawn lighter, no slider): {name_ranges(frozen)}</i>")
            panel = widgets.VBox([panel, note])
        small = S.eta < 0.1
        self.eta = widgets.FloatSlider(
            value=S.eta,
            min=min(0.1, S.eta),
            max=max(5.0, S.eta),
            step=0.01 if small else 0.1,
            description="eta",
            readout_format=".2f" if small else ".1f",
        )
        self.buttons = {name: widgets.Button(description=name) for name in ("gradient step", "10 steps", "reset")}
        self.readout = widgets.HTML()
        self.image = widgets.Image(format="png", layout=widgets.Layout(max_width="100%", height="auto"))
        self.out = widgets.Output()
        self.more = self.views_panel(widgets)
        for s in self.sliders:
            s.observe(self.on_slider, names="value")
        self.eta.observe(self.on_eta, names="value")
        self.buttons["gradient step"].on_click(lambda _: self.steps(1))
        self.buttons["10 steps"].on_click(lambda _: self.steps(10))
        self.buttons["reset"].on_click(lambda _: self.reset())
        controls = widgets.HBox([self.eta, *self.buttons.values()])
        shown = [self.out] if self.use_output else [self.image, self.out]
        self.widgets = [panel, controls, self.more, self.readout, *shown]

    # ---------------------------------------------------------------- building

    def grouped_panel(self, widgets):
        """Up to 30 sliders: one box per unit with weights that move, headed by the names of their inputs."""
        S = self.session
        M = S.model
        moving = set(self.index)
        boxes, k = [], 0
        for layer, L in enumerate(M.layers):
            labels = M.input_labels(layer) + ["bias"]
            for j in range(L.out_features):
                ks = [k + i for i in range(L.in_features + 1) if k + i in moving]
                if ks:
                    names = ", ".join(labels[q - k] for q in ks)
                    head = widgets.HTML(f"<b>{M.unit_title(layer, j)}</b>: weights for {names}")
                    boxes.append(widgets.VBox([head] + [self.sliders[self.index.index(q)] for q in ks]))
                k += L.in_features + 1
        return widgets.Box(boxes, layout=widgets.Layout(flex_flow="row wrap"))

    def dropdown_panel(self, widgets):
        """More than 30 sliders: choose one weight, then move it."""
        names = [self.session.names[k] for k in self.index]
        pick = widgets.Dropdown(options=[(nm, i) for i, nm in enumerate(names)], value=0, description="weight")
        holder = widgets.HBox([self.sliders[0]])
        pick.observe(lambda ch: setattr(holder, "children", [self.sliders[ch["new"]]]), names="value")
        return widgets.HBox([pick, holder])

    def views_panel(self, widgets):
        """The "More views" section: a checkbox per view and their settings."""
        S = self.session
        names = list(S.names)
        self.view_boxes = {
            v: widgets.Checkbox(value=v in S.views, description=VIEW_LABELS[v], indent=False) for v in VIEWS
        }
        self.case_pick = widgets.Dropdown(options=names, value=names[S.case_weight], description="case slopes of")
        a = S.slice_weights[0]
        b = S.slice_weights[1] if len(S.slice_weights) > 1 else None
        self.slice_pick = [
            widgets.Dropdown(options=names, value=names[a], description="slice over"),
            widgets.Dropdown(options=["none"] + names, value="none" if b is None else names[b], description="and"),
        ]
        self.beta = widgets.FloatSlider(value=S.beta, min=0.0, max=0.99, step=0.01, description="beta")
        self.ahead = widgets.IntSlider(value=S.lookahead_steps, min=1, max=200, description="steps ahead")
        self.log_box = widgets.Checkbox(value=S.log_scale, description="R on a log scale", indent=False)
        for w in [*self.view_boxes.values(), self.case_pick, *self.slice_pick, self.beta, self.ahead, self.log_box]:
            w.observe(self.on_views, names="value")
        box = widgets.VBox(
            [
                widgets.HBox(
                    list(self.view_boxes.values()) + [self.log_box], layout=widgets.Layout(flex_flow="row wrap")
                ),
                widgets.HBox([self.case_pick, *self.slice_pick]),
                widgets.HBox([self.beta, self.ahead]),
            ]
        )
        acc = widgets.Accordion(children=[box])
        acc.set_title(0, "More views")
        acc.selected_index = None
        return acc

    # ---------------------------------------------------------------- state

    def values(self):
        """The full weight vector: the sliders for the weights that move, the network for the others."""
        w = self.session.weights()
        for k, s in zip(self.index, self.sliders):
            w[k] = s.value
        return w

    def on_slider(self, *_):
        if self.busy:
            return
        self.session.model.set_weights(self.values())
        self.redraw()

    def on_eta(self, change):
        self.session.eta = float(change["new"])
        if not self.busy:
            self.redraw()

    def on_views(self, *_):
        if self.busy:
            return
        S = self.session
        names = list(S.names)
        S.views = tuple(v for v in VIEWS if self.view_boxes[v].value)
        S.case_weight = names.index(self.case_pick.value)
        first = names.index(self.slice_pick[0].value)
        second = self.slice_pick[1].value
        S.slice_weights = [first] if second in ("none", self.slice_pick[0].value) else [first, names.index(second)]
        S.beta = float(self.beta.value)
        S.lookahead_steps = int(self.ahead.value)
        S.log_scale = bool(self.log_box.value)
        self.picture = None  # the panels changed: a new figure
        self.redraw()

    def sync(self):
        """After steps, reset or record in code: the sliders show the network's weights, then one redraw."""
        self.busy = True
        try:
            w = self.session.weights()
            for k, s in zip(self.index, self.sliders):
                lo, hi = slider_range(w[k], s.min, s.max)
                if hi > s.max:
                    s.max = hi
                if lo < s.min:
                    s.min = lo
                s.value = w[k]
        finally:
            self.busy = False
        self.redraw()

    def steps(self, m):
        """m gradient steps from the slider values with the current eta; the session then updates the sliders."""
        S = self.session
        S.eta = float(self.eta.value)
        S.step(m, S.eta, self.values())

    def reset(self):
        """Back to the weights at the start, with no steps taken, and the sliders' starting ranges."""
        S = self.session
        S.reset()
        self.busy = True
        try:
            for k, s in zip(self.index, self.sliders):
                lo, hi = slider_range(S.w0[k])
                s.min, s.max = lo, hi
        finally:
            self.busy = False

    def redraw(self):
        S = self.session
        with self.out:
            if self.picture is None:
                self.picture = Picture(S, dpi=self.dpi)
            w = self.values()
            self.picture.refresh(w)
            S.fig = self.picture.fig
            png = self.picture.png()
            self.readout.value = readout_html(S.readouts(self.eta.value, w))
            if self.use_output:
                from IPython.display import Image, clear_output, display

                clear_output(wait=True)
                display(Image(png))
            else:
                self.image.value = png

    def show(self):
        from IPython.display import display

        display(*self.widgets)
        self.redraw()
        return self.session

# ------------------------ session.py ------------------------

# one gradient step: R before it, R predicted to first order (R - eta * |g|^2), and R after it
Prediction = namedtuple("Prediction", "before predicted actual")


def check_eta(eta):
    if not (isinstance(eta, (int, float, np.floating, np.integer)) and math.isfinite(eta) and eta > 0):
        raise ValueError(f"slopewalk: eta must be a positive number, not {eta!r}.")
    return float(eta)


def check_beta(beta):
    if not (isinstance(beta, (int, float, np.floating, np.integer)) and 0 <= beta < 1):
        raise ValueError(
            f"slopewalk: beta (the share of the last step kept by momentum) must be in [0, 1), not {beta!r}."
        )
    return float(beta)


def as_views(views):
    """The views as a tuple in the standard order; a string may list them separated by commas."""
    if views is None:
        return ()
    if isinstance(views, str):
        views = [v.strip() for v in views.split(",") if v.strip()]
    views = list(views)
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise ValueError(f"slopewalk: unknown view(s) {unknown}; choose from {', '.join(VIEWS)}.")
    return tuple(v for v in VIEWS if v in views)


class Session:
    """What explore returns.

    weights(), risk(), slopes(), predict(), step(k), reset() work on the network itself; lookahead() and
    case_slopes() leave it as it is. save(path) writes the current picture and record(path) a GIF of a run.
    history holds R at the start and after every step; predictions one Prediction per step; path the weights after
    every step (a row of NaN marks a jump made with the sliders); w0 the weights at the start; active the indices of
    the weights that move; controller the widgets or None.
    """

    def __init__(
        self,
        net,
        X,
        y,
        groups=None,
        eta=1.0,
        title=None,
        theme=None,
        *,
        task=None,
        params=None,
        views=(),
        case_slopes=None,
        slice_weights=None,
        beta=0.9,
        lookahead=30,
        log_scale=False,
    ):
        self.model = Model(net)
        self.data = prepare(X, y, groups, n_inputs=self.model.n_inputs, dtype=self.model.dtype, task=task)
        self.task = self.model.task = self.data.task
        self.layout = Layout(self.model.n_inputs, self.model.hidden, self.task)
        self.theme = theme or THEME
        self.title = title
        self.eta = check_eta(eta)
        self.active = self.model.select(params)
        self.views = as_views(views)
        output = self.model.layer_indices(len(self.model.layers) - 1)
        self.case_weight = output[0] if case_slopes is None else self.weight_index(case_slopes)
        if case_slopes is not None:
            self.views = as_views(self.views + ("cases",))
        if slice_weights is None:
            self.slice_weights = list(self.active) if len(self.active) <= 2 else output[:2]
        else:
            chosen = [slice_weights] if isinstance(slice_weights, (str, int, np.integer)) else list(slice_weights)
            self.slice_weights = [self.weight_index(v) for v in chosen]
            if not 1 <= len(self.slice_weights) <= 2 or len(set(self.slice_weights)) != len(self.slice_weights):
                raise ValueError("slopewalk: slice_weights takes one weight or two different weights.")
            self.views = as_views(self.views + ("slice",))
        self.beta = check_beta(beta)
        if int(lookahead) < 1:
            raise ValueError("slopewalk: lookahead must be at least 1 step.")
        self.lookahead_steps = int(lookahead)
        self.log_scale = bool(log_scale)
        self.w0 = self.model.get_weights()
        start = self.model.forward(self.data.X, self.data.y)
        self.loss0 = [self.data.stats(start.loss, c)[0] for c in range(self.data.G)]
        self.history = [self.risk(self.w0)]
        self.predictions = []
        self.path = [list(self.w0)]
        self.cache = {}
        self.fig = None
        self.controller = None

    @property
    def names(self):
        return self.model.names

    def weight_index(self, v):
        """The index of a weight given by its name ("w10") or its index."""
        n = self.model.n_weights
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool) and 0 <= v < n:
            return int(v)
        if isinstance(v, str) and v in self.model.names:
            return self.model.names.index(v)
        raise ValueError(f"slopewalk: {v!r} is not a weight of this network; the weights are w1 to w{n}.")

    def weights(self):
        return self.model.get_weights()

    def risk(self, w=None):
        """R at the weights w (default: the current ones). Sets the network to w."""
        return self.model.risk(self.data.X, self.data.y, w)

    def slopes(self, w=None):
        """dR/dw at the weights w (default: the current ones). Sets the network to w."""
        return self.model.slopes(self.data.X, self.data.y, w)

    def predict(self, eta=None, w=None):
        """The first-order prediction of the change of R for one step from w: -eta times the sum of the squared
        slopes of the weights that move."""
        eta = self.eta if eta is None else check_eta(eta)
        g = self.slopes(w)
        return -eta * float(sum(g[k] ** 2 for k in self.active))

    def step(self, k=1, eta=None, w=None):
        """k gradient steps, w <- w - eta * dR/dw for the weights that move, from w (default: the current weights).
        Records R, the prediction and the weights of every step. Returns the new weights."""
        eta = self.eta if eta is None else check_eta(eta)
        w = [float(v) for v in (self.weights() if w is None else w)]
        if any(abs(a - b) > 1e-12 for a, b in zip(w, self.path[-1])):
            self.path.append([math.nan] * len(w))  # the sliders moved the weights: a new segment of the path
            self.path.append(list(w))
        R = self.risk(w)
        for _ in range(k):
            g = self.slopes(w)
            g2 = float(sum(g[i] ** 2 for i in self.active))
            for i in self.active:
                w[i] -= eta * float(g[i])
            R_new = self.risk(w)
            self.predictions.append(Prediction(R, R - eta * g2, R_new))
            self.history.append(R_new)
            self.path.append(list(w))
            R = R_new
        self.model.set_weights(w)
        self.changed()
        return w

    def reset(self):
        """Back to the weights at the start, with no steps taken. Returns the weights."""
        self.model.set_weights(self.w0)
        self.history = [self.risk(self.w0)]
        self.predictions = []
        self.path = [list(self.w0)]
        self.changed()
        return list(self.w0)

    def lookahead(self, steps=None, eta=None, beta=None, w=None):
        """The next steps from w (default: the current weights) of gradient descent and of heavy-ball momentum
        (each step keeps the share beta of the last one: dw <- beta * dw - eta * g), moving only the weights that move.
        The network is left as it was. Returns {"gd": {"R": [...], "path": array}, "momentum": {...}}."""
        steps = self.lookahead_steps if steps is None else int(steps)
        eta = self.eta if eta is None else check_eta(eta)
        beta = self.beta if beta is None else check_beta(beta)
        now = self.weights()
        start = np.asarray(now if w is None else w, dtype=float)
        idx = list(self.active)
        out = {}
        for key, b in (("gd", 0.0), ("momentum", beta)):
            x, v = start.copy(), np.zeros_like(start)
            Rs, path = [self.risk(x)], [x.copy()]
            for _ in range(steps):
                g = self.slopes(x)
                v[idx] = b * v[idx] - eta * g[idx]
                x = x + v
                Rs.append(self.risk(x))
                path.append(x.copy())
            out[key] = {"R": Rs, "path": np.array(path)}
        self.model.set_weights(now)
        return out

    def case_slopes(self, w=None):
        """Each case's slope of its own loss along every weight, as an n x n_weights array; the mean of a column is
        the slope dR/dw of that weight."""
        return self.model.case_slopes(self.data.X, self.data.y, w)

    def readouts(self, eta=None, w=None):
        """The numbers of the readout strip: R, the size of the gradient of the weights that move, the accuracy
        (binary), the predicted change of R for the next step, and the last step's predicted and actual change."""
        eta = self.eta if eta is None else check_eta(eta)
        w = self.weights() if w is None else list(w)
        g = self.slopes(w)
        g2 = float(sum(g[k] ** 2 for k in self.active))
        last = self.predictions[-1] if self.predictions else None
        out = {
            "R": self.risk(w),
            "grad_norm": math.sqrt(g2),
            "accuracy": self.model.accuracy(self.data.X, self.data.y, w) if self.task == "binary" else None,
            "eta": eta,
            "next_change": -eta * g2,
            "last": None if last is None else (last.predicted - last.before, last.actual - last.before),
        }
        self.model.set_weights(w)
        return out

    def changed(self):
        """Tells the widgets, if any, that the weights changed in code."""
        if self.controller is not None:
            self.controller.sync()

    def draw(self, w=None, history=None):
        """Draws the picture at w (default: the current weights) as a new pyplot figure and keeps it as self.fig."""
        saved = self.history
        if history is not None:
            self.history = list(history)
        try:
            self.fig = Picture(self, pyplot=True).refresh(w)
        finally:
            self.history = saved
        return self.fig

    def save(self, path, dpi=120):
        """Saves the picture at the current weights to path (PNG, PDF or SVG by its extension)."""
        picture = Picture(self, dpi=dpi)
        picture.refresh()
        picture.fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    def record(self, path, steps=40, eta=None, every=1, fps=6, dpi=80):
        """Takes `steps` gradient steps and writes the picture after every `every` steps (and after the last) to an
        animated GIF; .mp4 also works when ffmpeg is installed. The first frame is the current state. Afterwards the
        network holds the final weights, as after step(steps). Returns path."""
        from matplotlib.animation import FFMpegWriter, PillowWriter

        path = str(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".gif":
            writer = PillowWriter(fps=fps)
        elif ext == ".mp4":
            if not FFMpegWriter.isAvailable():
                raise ValueError(
                    "slopewalk: writing .mp4 needs ffmpeg, which was not found. Use a path ending in .gif."
                )
            writer = FFMpegWriter(fps=fps)
        else:
            raise ValueError(f"slopewalk: record writes .gif files (or .mp4 with ffmpeg), not {ext or 'no extension'}.")
        steps, every = int(steps), max(1, int(every))
        picture = Picture(self, dpi=dpi)
        controller, self.controller = self.controller, None  # update the widgets once, at the end
        try:
            with writer.saving(picture.fig, path, dpi):
                picture.refresh()
                writer.grab_frame()
                for i in range(1, steps + 1):
                    self.step(1, eta)
                    if i % every == 0 or i == steps:
                        picture.refresh()
                        writer.grab_frame()
        finally:
            self.controller = controller
        self.changed()
        return path

    def _ipython_display_(self):
        pass  # the picture and the controls are shown by explore itself

    def __repr__(self):
        return f"<slopewalk session: {len(self.names)} weights, R = {self.risk():.3f}>"


def explore(
    net,
    X,
    y,
    groups=None,
    eta=1.0,
    title=None,
    *,
    task=None,
    params=None,
    views=(),
    case_slopes=None,
    slice_weights=None,
    beta=0.9,
    lookahead=30,
    log_scale=False,
):
    """Gradient descent by hand on `net`, with inputs X (n x d) and labels y.

    net: nn.Sequential of nn.Linear layers with nn.Tanh, nn.ReLU, nn.Sigmoid or nothing between them, ending in
    nn.Linear(..., 1). Its output is the score a of class 1 (labels 0 or 1; R is the average log loss) or, with
    task="regression", the prediction (R is the mean squared error).
    groups: a number per point; one panel per group. None: one panel per point (up to 8 points), else one per class.
    eta: the step size of the gradient step buttons (a slider).
    params: the weights that move: a list of names such as ["w10", "w11"], "output" or "layer:k". The others are
    drawn lighter and stay fixed.
    views: extra panels, any of "cases" (each case's slope along one weight; case_slopes="w10" chooses it), "slice"
    (R over one or two weights; slice_weights chooses them), "line" (R along the next step), "hidden" (the cases at
    their hidden values) and "momentum" (the next `lookahead` steps of gradient descent and of momentum beta).
    log_scale: R on a log axis.

    Draws one copy of the network per group (every value a mean ± standard deviation over the group), with each
    weight on its line, the units as z, activation and h, then the output and the loss with its change from the
    start. Below: the input plane, R over the steps taken with the first-order prediction of each step, and a table
    with each weight's value, the change of R for +0.1 and the slope dR/dw from R.backward().
    With ipywidgets: sliders for the weights, eta, `gradient step`, `10 steps`, `reset`, a strip of readouts and a
    "More views" section. The sliders change the weights of `net` itself. Without ipywidgets, or with NB_STATIC=1,
    a still picture of the start.
    Returns a Session: step(k), reset(), save(path), record(path), weights(), history.
    """
    session = Session(
        net,
        X,
        y,
        groups,
        eta,
        title,
        task=task,
        params=params,
        views=views,
        case_slopes=case_slopes,
        slice_weights=slice_weights,
        beta=beta,
        lookahead=lookahead,
        log_scale=log_scale,
    )
    if os.environ.get("NB_STATIC") == "1":
        return still_picture(session, STATIC_MESSAGE)
    try:
        import ipywidgets  # noqa: F401
    except ImportError:
        return still_picture(session, NO_WIDGETS_MESSAGE)

    session.controller = Controller(session)
    return session.controller.show()

__all__ = [
    "explore",
    "clusters",
    "xor_centres",
    "two_points",
    "sine_points",
    "weight_table",
    "Session",
    "__version__",
]
