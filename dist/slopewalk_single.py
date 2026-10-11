"""slopewalk 0.1.0.dev0: gradient descent by hand on small PyTorch networks, in one file.

Built by tools/make_single_file.py from the package at https://github.com/mayankbakshi/slopewalk; edit the package,
not this file. Same use as the package:

    net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    X, y, groups = clusters([(-1, 1), (1, 1), (-1, -1), (1, -1)], [1, 0, 0, 1])
    explore(net, X, y, groups)
"""

import copy
import inspect
import io
import math
import os
import re
import sys
import textwrap
from collections import namedtuple
import json
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import BoxStyle, Rectangle
from matplotlib.text import Text
from pathlib import Path

__version__ = "0.1.0.dev0"


# ------------------------ optim.py ------------------------

OPTIMIZERS = ("gd", "momentum", "adam")


def check_optimizer(kind):
    if kind not in OPTIMIZERS:
        raise ValueError(f"slopewalk: optimizer must be 'gd', 'momentum' or 'adam', not {kind!r}.")
    return kind


def check_decay(weight_decay):
    if not (isinstance(weight_decay, (int, float)) and math.isfinite(weight_decay) and weight_decay >= 0):
        raise ValueError(f"slopewalk: weight_decay must be a number from 0 up, not {weight_decay!r}.")
    return float(weight_decay)


def check_beta2(beta2):
    if not (isinstance(beta2, (int, float)) and 0 <= beta2 < 1):
        raise ValueError(
            f"slopewalk: beta2 (Adam's share of the last squared slopes) must be in [0, 1), not {beta2!r}."
        )
    return float(beta2)


class Optimizer:
    """The step rule with its state: the velocity of momentum, or Adam's m, v and step count t."""

    def __init__(self, kind="gd", weight_decay=0.0, beta=0.9, beta2=0.999, eps=1e-8):
        self.kind = check_optimizer(kind)
        self.weight_decay = check_decay(weight_decay)
        self.beta, self.beta2, self.eps = beta, check_beta2(beta2), eps
        self.reset()

    def reset(self):
        """Forgets the steps taken: the next step is a first step."""
        self.velocity = {}  # momentum: per weight
        self.m, self.v, self.t = {}, {}, 0  # Adam

    def copy(self):
        """A copy of the state, for back one step."""
        out = Optimizer(self.kind, self.weight_decay, self.beta, self.beta2, self.eps)
        out.velocity, out.m, out.v, out.t = dict(self.velocity), dict(self.m), dict(self.v), self.t
        return out

    def set(self, kind=None, weight_decay=None, beta=None, beta2=None):
        """Changes the rule or its numbers; a new rule forgets the state."""
        if kind is not None and kind != self.kind:
            self.kind = check_optimizer(kind)
            self.reset()
        if weight_decay is not None:
            self.weight_decay = check_decay(weight_decay)
        if beta is not None:
            self.beta = float(beta)
        if beta2 is not None:
            self.beta2 = check_beta2(beta2)

    def direction(self, g, w, active, eta):
        """The change dw of each weight that the next step would make, as {k: dw}, without taking it."""
        return self.take(g, w, active, eta, self.copy())

    def step(self, g, w, active, eta):
        """The change dw of each weight for this step, as {k: dw}; the state moves on."""
        return self.take(g, w, active, eta, self)

    @staticmethod
    def take(g, w, active, eta, state):
        """The step of the rule in `state` (whose state moves on), for the weights in `active`."""
        out = {}
        lam, kind = state.weight_decay, state.kind
        if kind == "adam":
            state.t += 1
            b1, b2 = state.beta, state.beta2
            fix1, fix2 = 1 - b1**state.t, 1 - b2**state.t
        for k in active:
            gk = float(g[k]) + lam * float(w[k])
            if kind == "gd":
                out[k] = -eta * gk
            elif kind == "momentum":
                state.velocity[k] = state.beta * state.velocity.get(k, 0.0) - eta * gk
                out[k] = state.velocity[k]
            else:
                state.m[k] = b1 * state.m.get(k, 0.0) + (1 - b1) * gk
                state.v[k] = b2 * state.v.get(k, 0.0) + (1 - b2) * gk * gk
                out[k] = -eta * (state.m[k] / fix1) / (math.sqrt(state.v[k] / fix2) + state.eps)
        return out

# ------------------------ model.py ------------------------

# activation module -> (name, numpy function, half-width of the curve drawn in the glyph, numpy derivative)
ACTIVATIONS = {
    nn.Tanh: ("tanh", np.tanh, 3.0, lambda t: 1 - np.tanh(t) ** 2),
    nn.ReLU: ("ReLU", lambda t: np.maximum(t, 0), 3.0, lambda t: (np.asarray(t) > 0).astype(float)),
    nn.Sigmoid: (
        "sigmoid",
        lambda t: 1 / (1 + np.exp(-t)),
        6.0,
        lambda t: np.exp(-t) / (1 + np.exp(-t)) ** 2,
    ),
}
SIGMOID = ACTIVATIONS[nn.Sigmoid]
SUPPORTED = "nn.Linear with nn.Tanh, nn.ReLU or nn.Sigmoid between them"
TASKS = ("binary", "multiclass", "regression")
MAX_OUTPUTS = 4  # one colour per class, none of them a colour that means something else; or up to 4 targets
BATCH_CELLS = 400_000  # risk_batch handles at most this many (weight vector, point) pairs at once

Weight = namedtuple("Weight", "name layer unit input value")
# hidden: [(z, h), ...] per hidden layer; a: the score (binary), the scores of the classes (multiclass, n x K) or the
# prediction (regression, n x K for K targets); p: sigmoid(a) for binary, softmax(a) for multiclass (n x K), None for
# regression; loss: per point
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
    if not layers or layers[-1].act is not None:
        raise ValueError(
            "slopewalk: the network must end with nn.Linear, without an activation: nn.Linear(..., 1) for the score "
            "of class 1 (or the prediction for task='regression'), nn.Linear(..., K) for the scores of K classes (or K "
            "targets with task='regression')."
        )
    if layers[-1].out_features > MAX_OUTPUTS:
        k = layers[-1].out_features
        raise ValueError(
            f"slopewalk: the network ends with nn.Linear(..., {k}), {k} outputs; slopewalk draws up to {MAX_OUTPUTS}: "
            f"the scores of up to {MAX_OUTPUTS} classes, one colour each, or up to {MAX_OUTPUTS} targets."
        )
    for L in layers:
        if L.linear.bias is None:
            raise ValueError(f"slopewalk: {L.linear} has no bias. Use nn.Linear with bias=True.")
    return layers


def check_task(task, n_outputs):
    """The task for a network with n_outputs outputs: as given, or (None) multiclass for several outputs and binary
    for one. Raises ValueError when the task and the outputs do not fit together."""
    if task not in (None,) + TASKS:
        raise ValueError(f"slopewalk: task must be 'binary', 'multiclass' or 'regression', not {task!r}.")
    if task is None:
        return "multiclass" if n_outputs > 1 else "binary"
    if task == "multiclass" and n_outputs == 1:
        raise ValueError(
            "slopewalk: task='multiclass' needs one output per class: end the network with nn.Linear(..., K) for K "
            "classes."
        )
    if task == "binary" and n_outputs > 1:
        raise ValueError(
            f"slopewalk: the network has {n_outputs} outputs: one score per class is task='multiclass', and "
            f"{n_outputs} targets task='regression'. For task='binary', end it with nn.Linear(..., 1)."
        )
    return task


class Model:
    """The parsed network.

    The weights are one list in the order of their names: layer by layer, unit by unit, each unit's incoming weights
    in the order of its inputs, then its bias. For nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)) this
    gives w1 to w3 (unit 1), w4 to w6, w7 to w9, w10 to w12 (output weights) and w13 (output bias).
    task: "binary" (one output, the score of class 1; R is the log loss, as F.binary_cross_entropy_with_logits),
    "multiclass" (K outputs, the scores of the classes 0 to K-1; R is the cross-entropy, as F.cross_entropy) or
    "regression" (one output per target; R is the mean squared error over all cases and targets, as F.mse_loss, so
    each case's loss is the mean over its targets). None: multiclass for a network with several outputs, else binary.
    """

    def __init__(self, net, task=None):
        self.net = net
        self.layers = parse_layers(net)
        self.n_outputs = self.layers[-1].out_features
        self.task = check_task(task, self.n_outputs)
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
        """The title of a unit's slider group. Classes count from 0, as their labels; targets from 1, as inputs."""
        if layer == len(self.layers) - 1:
            if self.n_outputs == 1:
                return "output unit"
            return f"output unit, class {j}" if self.task == "multiclass" else f"output unit {j + 1}"
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
        """The loss of the task: log loss on the score a (binary), cross-entropy on the scores a (n x K, multiclass;
        the labels y are the classes) or squared error of the prediction a (regression; with several targets, a and y
        are n x K and a case's loss is the mean over its targets, so that R is F.mse_loss of all of them)."""
        if self.task == "regression":
            if reduction == "none" and self.n_outputs > 1:
                return F.mse_loss(a, y, reduction="none").mean(-1)
            return F.mse_loss(a, y, reduction=reduction)
        if self.task == "multiclass":
            return F.cross_entropy(a, y.long(), reduction=reduction)
        return F.binary_cross_entropy_with_logits(a, y, reduction=reduction)

    def scores(self, t):
        """The network's output t (n x outputs) as the loss takes it: n numbers for one output, else n x K."""
        return t.reshape(-1) if self.n_outputs == 1 else t

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
            a = self.scores(self.layers[-1].linear(t))
            if self.task == "binary":
                p = torch.sigmoid(a).numpy()
            elif self.task == "multiclass":
                p = torch.softmax(a, dim=1).numpy()
            else:
                p = None
            loss = self.loss(a, y, reduction="none")
        return Values(hidden, a.numpy(), p, loss.numpy())

    def risk(self, X, y, w=None):
        """R: the average loss over all points (log loss for binary, cross-entropy for multiclass, squared error for
        regression)."""
        if w is not None:
            self.set_weights(w)
        with torch.no_grad():
            return float(self.loss(self.scores(self.net(X)), y))

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
                R = self.loss(self.scores(self.net(X)), y)
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
                if self.task == "multiclass":  # one row of K scores per (weight vector, point), in that order
                    losses = self.loss(t.reshape(G * n, -1), y.repeat(G), reduction="none").reshape(G, n)
                elif self.n_outputs > 1:  # several targets: a case's loss is the mean over them
                    losses = ((t - y) ** 2).mean(-1)
                else:
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
            losses = self.loss(self.scores(h), y, reduction="none")
            deltas = torch.autograd.grad(losses.sum(), zs)
        n = X.shape[0]
        blocks = []
        for h_in, d in zip(inputs, deltas):
            with_one = torch.cat([h_in.detach(), torch.ones(n, 1, dtype=h_in.dtype)], dim=1)  # inputs, then 1
            blocks.append((d[:, :, None] * with_one[:, None, :]).reshape(n, -1))  # unit by unit
        return torch.cat(blocks, dim=1).detach().numpy().astype(float)

    def accuracy(self, X, y, w=None):
        """The share of cases classified correctly: on the correct side of the boundary (binary), or with the
        largest score for their own class (multiclass)."""
        if w is not None:
            self.set_weights(w)
        with torch.no_grad():
            out = self.net(X)
        if self.task == "multiclass":
            right = out.argmax(1) == y.long()
        else:
            right = (out.reshape(-1) >= 0).to(y.dtype) == y
        return float(right.to(torch.float64).mean())


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
    if len(centers) != len(labels):
        raise ValueError(
            f"slopewalk: {len(centers)} centres but {len(labels)} labels; give one label per centre, in the same order."
        )
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


class Data:
    """The points of one explore call: X (n x d), y (n), groups (n), and which points belong to each group.
    n_classes: 2 for binary, K for multiclass (the labels 0 to K-1), None for regression."""

    def __init__(self, X, y, groups, task="binary", n_classes=None):
        self.X, self.y, self.groups, self.task = X, y, groups, task
        self.n_classes = n_classes if n_classes is not None else (2 if task == "binary" else None)
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


def infer_task(y, task, n_outputs=1):
    """The task: as given, or from the network's outputs and the labels. Raises ValueError rather than guess.
    Several outputs: multiclass, with whole-number labels from 0 to K-1. One output: binary when every label is 0 or
    1; regression only when asked for."""
    task = check_task(task, n_outputs)
    if task == "regression":
        return task
    if task == "multiclass":
        bad = y[(y != np.round(y)) | (y < 0) | (y >= n_outputs)]
        if len(bad):
            raise ValueError(
                f"slopewalk: y has the label {bad[0]:g}, but the network has {n_outputs} outputs, one per class: the "
                f"labels must be whole numbers from 0 to {n_outputs - 1}."
            )
        return task
    others = y[~np.isin(y, (0.0, 1.0))]
    if len(others):
        if ((y == np.round(y)) & (y >= 0)).all():
            k = int(y.max()) + 1
            hint = f"for {k} classes, end the network with nn.Linear(..., {k}) and use the labels 0 to {k - 1}"
        else:
            hint = "for K classes, end the network with nn.Linear(..., K) and use the labels 0 to K-1"
        raise ValueError(
            f"slopewalk: y has values other than 0 and 1 (for example {others[0]:g}). With one output the labels "
            f"must be 0 or 1; {hint}; for a numerical target pass task='regression'."
        )
    return "binary"


def check_points(X, n_inputs):
    """X as an n x d array of points, or ValueError. A flat X is one point, except for a network with one input."""
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
    return Xa


def check_targets(y, n, task, n_outputs):
    """y as n labels or targets, or as an n x K array of targets (task='regression', K outputs), or ValueError."""
    ya = as_numbers(y, "y")
    if ya.ndim == 2 and ya.shape[1] == 1:
        ya = ya[:, 0]
    if ya.ndim == 2:  # one column per target
        if task != "regression":
            raise ValueError(
                f"slopewalk: y has {ya.shape[1]} columns, one per target: pass task='regression' (for classes, y holds "
                "one label per point)."
            )
        if ya.shape[0] != n:
            raise ValueError(f"slopewalk: y has {ya.shape[0]} rows for {n} points; give one row of targets per point.")
        if ya.shape[1] != n_outputs:
            raise ValueError(
                f"slopewalk: y has {ya.shape[1]} targets per point, but the network has {n_outputs} output(s); end it "
                f"with nn.Linear(..., {ya.shape[1]})."
            )
    elif ya.ndim > 2:
        raise ValueError(
            f"slopewalk: y must hold one value per point, or one row of targets per point, not {ya.shape}."
        )
    else:
        ya = ya.reshape(-1)
        if ya.shape[0] != n:
            raise ValueError(f"slopewalk: y has {ya.shape[0]} values for {n} points; give one label per point.")
        if task == "regression" and n_outputs > 1:
            raise ValueError(
                f"slopewalk: the network has {n_outputs} outputs, so y needs {n_outputs} targets per point: an n x "
                f"{n_outputs} array, one column per output."
            )
    return ya


def make_groups(groups, Xa, ya, task):
    """The group of each point: as given, or by default one per point up to 8 points, else one per class, or for
    regression four groups along x1 (one input) or one group (several inputs)."""
    n, d = Xa.shape
    if groups is None:
        if n <= 8:
            ga = np.arange(n)
        elif task in ("binary", "multiclass"):
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
    return ga


def prepare(X, y, groups=None, n_inputs=None, dtype=torch.float32, task=None, n_outputs=1):
    """The checked tensors of an explore call, as a Data object.

    X: n points with n_inputs numbers each (a list of points, an array or a tensor). A flat X is one point, except for
    a network with one input, where it is n points. y: n labels (0 or 1; 0 to K-1 for a network with K outputs) or
    targets (task='regression': n numbers, or n x K for a network with K outputs). groups None: one group per point
    up to 8 points; more points: one group per class (classification), or for regression four groups of neighbouring
    points along x1 (one input) or a single group (several inputs).
    """
    Xa = check_points(X, n_inputs)
    n, d = Xa.shape
    ya = check_targets(y, n, task, n_outputs)
    check_finite(Xa, "X")
    check_finite(ya, "y")
    task = infer_task(ya, task, n_outputs)
    ga = make_groups(groups, Xa, ya, task)
    Xt = torch.as_tensor(np.ascontiguousarray(Xa), dtype=dtype)  # reversed or sliced arrays need a copy
    yt = torch.as_tensor(np.ascontiguousarray(ya), dtype=dtype)
    n_classes = n_outputs if task == "multiclass" else None
    return Data(Xt, yt, torch.as_tensor(np.ascontiguousarray(ga), dtype=torch.int64), task, n_classes)

# ------------------------ layout.py ------------------------

BW = 0.27  # half-width of a value box; lines end this far from the centre of the box
TABLE_ROWS = 15  # rows per column of the slope table
LABEL_FRACTIONS = np.linspace(0.14, 0.86, 13)  # where along a line a label may sit
LABEL_SHIFTS = (0.0, 0.8, -0.8, 1.6, -1.6, 2.4, -2.4, 3.2, -3.2, 4.0, -4.0, 4.8, -4.8)  # sideways shifts
LABEL_TOLERANCE = 0.02  # a label may overlap other labels or the boxes by at most this fraction of its area
BOTTOM_PER_ROW = 3  # panels per row below the networks
BOTTOM_HEIGHT = 4.2  # height in inches of one row of those panels
TABLE_SHARE = 0.8  # the share of the table panel's height that its rows may take; the title and margins take the rest
UNIT_SPACING = 1.37  # inches between the units of a layer that a taller row keeps (0.9 in on the drawn panel)
LAYOUTS = ("standard", "wide")
WIDTH = {"standard": 15.0, "wide": 27.0}  # default figure width in inches
WIDE_PER_ROW, WIDE_BOTTOM_PER_ROW = 4, 6  # the most network panels and lower panels per row of the wide layout
NET_GAP, PANEL_GAP = 0.5, 0.8  # inches between network panels, and between the panels below them
OUTPUT_GAP = 0.3  # extra room before several output units (more would make the panel too wide for two in a row)
TABLE_INCH_PER_POINT = 0.26  # the width a block of the table needs per point of font, with the short headers
ZOOM = 1.5  # a single network panel (one group shown): this much larger than one of two side by side
TEXT_SCALE = {"standard": 1.0, "wide": 1.1}  # text in the network panels: the wide picture is shown smaller
# the wide layout gives each lower panel a base width in inches that suits it (for a row 4.2 inches tall), and the
# width left over in its row goes to the panels that can use it, in these proportions
WIDE_BASE = {
    "plane": 4.6,
    "hidden": 4.6,
    "slice": 4.8,
    "table": 5.0,
    "history": 5.0,
    "line": 5.0,
    "cases": 5.0,
    "momentum": 5.0,
    "backprop": 10.5,
}
HISTORY_MIN = 3.6  # the narrowest R over the steps gets in a full row of the wide layout
WIDE_STRETCH = {"history": 3.0, "table": 0.5, "line": 1.0, "cases": 1.0, "momentum": 1.0, "backprop": 1.0}


class Layout:
    """Positions for a network with n_inputs inputs and the hidden layers `hidden` (a list of unit counts).

    task "binary" ends each panel with a, the sigmoid, p and the loss box; "multiclass" with a score a and a p for
    each of the `outputs` classes, the softmax between them, and the loss box; "regression" with the prediction and
    the loss box. style "standard": a figure 15 inches wide, two network panels per row (one for deep networks) and
    three lower panels per row. "wide": 27 inches, up to four network panels in a row (two for deep networks) and up
    to six lower panels in a row, each as wide as suits it, for wide screens. width (inches) overrides the width of
    either. The network panels keep their size and spacing whatever panels are below them.
    """

    def __init__(self, n_inputs, hidden, task="binary", style="standard", width=None, outputs=1):
        if style not in LAYOUTS:
            raise ValueError(f"slopewalk: layout must be 'standard' or 'wide', not {style!r}.")
        if width is not None and not (isinstance(width, (int, float)) and 6 <= width <= 60):
            raise ValueError(f"slopewalk: width is the figure width in inches, between 6 and 60, not {width!r}.")
        self.n_inputs, self.hidden, self.task, self.style, self.outputs = n_inputs, hidden, task, style, outputs
        self.zoom = 1.0  # ZOOM when one network panel is shown (see focus)
        self.width = float(width) if width is not None else WIDTH[style]
        self.text_scale = f = TEXT_SCALE[style]
        extra = 1.2 * (f - 1)  # larger text: the loss box moves right, clear of the box before it
        self.blocks, x = [], 1.75
        for _ in hidden:
            self.blocks.append((x, x + 0.75, x + 1.5))  # x of the z box, of the glyph and of the h box
            x += 2.75
        if outputs > 1:  # the lines fan out to several output units: more room for their labels
            x += OUTPUT_GAP
        if task == "regression":
            gap = 1.1 if outputs == 1 else 1.6  # several targets: room for their arrows into the loss box
            self.x_a, self.x_sigmoid, self.x_p, self.x_loss = x, None, None, x + gap + extra
        else:
            loss_gap = 2.4 if outputs == 1 else 2.5  # several classes: p boxes with a ± line next to the loss box
            self.x_a, self.x_sigmoid, self.x_p, self.x_loss = x, x + 0.75, x + 1.5, x + loss_gap + extra
        self.x_input_value = -0.24 * f  # the right edge of an input's value box, left of its circle
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

    def narrow(self):
        """A network narrow enough for two panels side by side in the standard figure (at most one hidden layer)."""
        return self.x_loss + 1.55 <= 9

    def per_row(self, n_panels=None):
        """Network panels per row: 2 (or 1 for deep networks); the wide layout puts up to 4 (or 2) in a row."""
        if self.style == "wide":
            most = WIDE_PER_ROW if self.narrow() else 2
            return max(1, min(most, n_panels if n_panels else most))
        return 2 if self.narrow() else 1

    def rows(self, n_panels):
        return -(-n_panels // self.per_row(n_panels))

    def focus(self, alone):
        """One network panel shown (alone True): it is drawn ZOOM times larger, text included."""
        self.zoom = ZOOM if alone else 1.0

    def panel_share(self):
        """The share of the row's width that a single shown panel takes: ZOOM times one of two side by side, at most
        the row (a deep network takes the row anyway)."""
        return min(1.0, self.zoom / 2) if self.narrow() else 1.0

    def row_height(self, with_sd=False, backprop=False):
        """Height of a panel row in inches: 2.9 for up to 3 units in a layer, then growing so that the units of the
        widest layer stay about 0.9 in apart (the outputs count as a layer), up to four times the usual for 12 units;
        a fifth more when the boxes carry a standard deviation line (groups of points), and 40% more when the backward
        pass adds its boxes under the units."""
        m = max(list(self.hidden) + [self.outputs])
        units = min(4.0, max(1.0, UNIT_SPACING * (m - 1) / 2.9))
        return 2.9 * units * (1.2 if with_sd else 1.0) * (1.4 if backprop else 1.0) * self.text_scale * self.zoom

    def table_blocks(self, n_weights):
        """(blocks, width of a block in inches) of the slope table: one block up to TABLE_ROWS weights, else two."""
        blocks = 2 if n_weights > TABLE_ROWS else 1
        panel = WIDE_BASE["table"] if self.style == "wide" else self.width / BOTTOM_PER_ROW
        if self.style == "wide" and blocks == 2:
            panel = 7.5
        return blocks, (panel - PANEL_GAP) * (1 - 0.03 * (blocks - 1)) / blocks

    def first_row_height(self, n_weights, font, leading):
        """Height in inches of the first row below the networks, which holds the slope table: BOTTOM_HEIGHT, or more
        when the table's rows would not fit at the font its width allows (up to `font` points, a row `leading` times
        that), up to twice BOTTOM_HEIGHT; above that the table's font shrinks."""
        blocks, block_inches = self.table_blocks(n_weights)
        size = max(8.0, min(font, block_inches / TABLE_INCH_PER_POINT))
        rows = min(n_weights, max(TABLE_ROWS, -(-n_weights // 2))) + 1  # the header and the longest block
        needed = rows * size * leading / 72 / TABLE_SHARE
        if needed <= BOTTOM_HEIGHT * 1.05:  # a small shrink of the font is better than a taller row
            return BOTTOM_HEIGHT
        return min(2 * BOTTOM_HEIGHT, needed)

    def lower_rows(self, panels, n_weights=0):
        """The rows of the panels below the networks: lists of (panel, width), with widths in inches that add up to
        the width of the figure.

        standard: three columns of equal width; the derivation of the backward pass takes a whole row, and the spare
        columns of a short last row stay empty (panel None). wide: up to six columns in a row (the derivation counts
        two), the rows about equally full and every row filled: each panel gets its base width, and the width left
        over goes to the panels that can use it (mostly R over the steps).
        """
        wide = self.style == "wide"
        spans = [(2 if wide else BOTTOM_PER_ROW) if p == "backprop" else 1 for p in panels]
        cap = -(-sum(spans) // -(-sum(spans) // WIDE_BOTTOM_PER_ROW)) if wide else BOTTOM_PER_ROW
        rows, row, used = [], [], 0
        for name, span in zip(panels, spans):
            if row and used + span > cap:
                rows.append(row)
                row, used = [], 0
            row.append((name, span))
            used += span
        rows.append(row)
        if not wide:
            third = self.width / BOTTOM_PER_ROW
            return [[(n, third * k) for n, k in r] + [(None, third)] * (cap - sum(k for _, k in r)) for r in rows]
        return [self.fill([n for n, _ in r], n_weights) for r in rows]

    def fill(self, names, n_weights=0):
        """Widths for one row of the wide layout: base widths, and the rest of the row shared by WIDE_STRETCH."""
        base = [WIDE_BASE[n] for n in names]
        if "table" in names and n_weights > TABLE_ROWS:
            base[names.index("table")] = 7.5  # two blocks of rows side by side
        spare = self.width - sum(base)
        if spare < 0 and "history" in names:  # a full row: R over the steps gives up width first, the table keeps it
            i = names.index("history")
            base[i] -= min(-spare, max(0.0, base[i] - HISTORY_MIN))
            spare = self.width - sum(base)
        if spare <= 0:
            return [(n, b * self.width / sum(base)) for n, b in zip(names, base)]
        stretch = [WIDE_STRETCH.get(n, 0.0) for n in names]
        if sum(stretch) == 0:
            stretch = [1.0] * len(names)
        return [(n, b + spare * k / sum(stretch)) for n, b, k in zip(names, base, stretch)]

    def wspace(self, n, gap):
        """The wspace of a row of n panels with gaps of `gap` inches (wspace is a fraction of the mean panel width)."""
        return 0.0 if n < 2 else n * gap / max(self.width - 0.6 - (n - 1) * gap, 1.0)

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

# ------------------------ theme.py ------------------------

class Theme:
    """Colours, fonts and sizes of the picture. Red is class 1 and blue class 0 (with more classes, teal is class 2 and
    magenta class 3); green is the hidden layer; purple is the backward pass; orange marks negative weights. Text in
    boxes is ink: a box carries its meaning in its border and fill. Ratios are WCAG contrast against paper unless said
    otherwise.

    Only the labels on the weight lines use edge_label_size, and only the weight lines use zorder 1.
    """

    paper, ink, muted = "#FFFFFF", "#1F2328", "#59636E"  # ink 15.8:1, muted 6.1:1
    class_colour = {0: "#0969DA", 1: "#CF222E", 2: "#00796B", 3: "#B4237F"}  # 5.2, 5.4, 5.3, 6.0:1: titles, marks
    class_tint = {0: "#E3EFFD", 1: "#FDECEC", 2: "#DFF3F0", 3: "#FBEAF3"}  # box fills; the text on them is ink
    mixed_colour, mixed_tint = "#1F2328", "#EEF1F4"
    neutral_colour, neutral_tint = "#3D444D", "#EEF1F4"  # regression panels
    positive, negative, negative_text = "#1F2328", "#D1600A", "#B04800"  # lines 15.8:1 and 3.9:1, numbers 5.6:1
    line_width, line_slope, line_max = 1.0, 1.3, 7  # a weight's line: line_width + line_slope * |w|, at most line_max
    frozen_alpha, frozen_text, frozen_fill = 0.35, "#59636E", "#F6F8FA"  # weights that do not move (params)
    hidden = "#1A7F37"  # the hidden layer only: the border of the h boxes, the dot and the band on the curves
    arrow, curve, node_edge = "#8C959F", "#59636E", "#8C959F"
    start_colour, prediction_colour = "#59636E", "#8C959F"
    gd_colour, momentum_colour = "#59636E", "#9A6700"  # gradient descent solid, momentum dashed
    value_box = dict(boxstyle="round,pad=0.25", fc="#F6F8FA", ec="#C4CBD3", lw=0.8)
    edge_label_size, glyph_name_size, value_size, node_size, title_size, suptitle_size = 8.5, 8, 9.5, 10, 11.5, 13
    note_size = 11  # notes inside panels, and the lines of the worked example (smaller only if they do not fit)
    panel_title_size, legend_size, letter_size = 12, 9.5, 12  # the panels below the networks
    tick_size, axis_label_size = 10.5, 11
    plane_alpha, value_cmap = 0.4, "viridis"
    table_size, table_leading = 12, 1.45  # the table's largest font; a row is table_leading times the font
    table_edge, table_head, table_highlight = "#D0D7DE", "#EEF1F4", "#FFF4C2"
    backprop_colour, backprop_tint, backprop_size = "#7A45D6", "#F4EFFC", 9  # 5.8:1, and 5.1:1 on its tint
    # surfaces and neutral marks: every colour of the picture lives in this class, so that a theme can change them all
    marker_edge, boundary = "#1F2328", "#1F2328"
    note_text, derivation_text, derivation_muted, point_neutral = "#59636E", "#1F2328", "#8C959F", "#59636E"
    glyph_fill, glyph_edge, node_fill, label_backing = "#FFFFFF", "#C4CBD3", "#FFFFFF", "#FFFFFF"
    reference_line, zero_line, landscape_line, plane_surface, path_colour = (
        "#C4CBD3",
        "#D0D7DE",
        "#8C959F",
        "#8C959F",
        "#1F2328",
    )
    landscape_low = "#59636E"  # R over two weights: low R in this colour, high R in paper
    spine = "#8C959F"  # the frames of the panels below the networks
    control_button, control_selected = "#EEF1F4", "#C4CBD3"  # the controls: buttons, and the chosen toggle button
    # what a stage of the backward pass does not use: boxes and curves recede to 30%, their numbers to 50% so that they
    # stay readable (3.2:1 on light paper, 4.9:1 on dark), lines to 12% (context only: the table keeps every weight)
    faded_alpha, faded_text_alpha, faded_line_alpha = 0.3, 0.5, 0.12

    def plane_colormap(self):
        """p for class 1 over the input plane: class 0 colour, paper at p = 0.5, class 1 colour."""
        return LinearSegmentedColormap.from_list(
            "slopewalk_plane", [self.class_colour[0], self.paper, self.class_colour[1]]
        )

    def landscape_colormap(self):
        return LinearSegmentedColormap.from_list("slopewalk_landscape", [self.landscape_low, self.paper])

    network_sizes = ("edge_label_size", "glyph_name_size", "value_size", "node_size", "title_size", "backprop_size")

    def scaled(self, factor):
        """A copy whose text in the network panels is `factor` times larger; the panels below keep their sizes."""
        out = copy.copy(self)
        for name in self.network_sizes:
            setattr(out, name, round(getattr(self, name) * factor, 2))
        return out


class DarkTheme(Theme):
    """The same picture for a dark notebook theme, in high contrast: light ink on near-black paper, bright class
    colours, and frames and box edges that stand out. Ratios against #0D1117."""

    paper, ink, muted = "#0D1117", "#F0F3F6", "#B1BAC4"  # ink 17.0:1, muted 9.6:1
    class_colour = {0: "#71B7FF", 1: "#FF9492", 2: "#4FD8C8", 3: "#FF8DCA"}  # 8.9, 8.9, 10.8 and 8.9:1
    class_tint = {0: "#0F2742", 1: "#3A1519", 2: "#0E2E2B", 3: "#36162A"}
    mixed_colour, mixed_tint = "#F0F3F6", "#21262D"
    neutral_colour, neutral_tint = "#D1D9E0", "#21262D"
    positive, negative, negative_text = "#E6EDF3", "#FFA657", "#FFA657"  # lines 16.0:1, negative 9.8:1
    frozen_text, frozen_fill = "#9EA7B3", "#161B22"
    hidden = "#56D364"
    arrow, curve, node_edge = "#9EA7B3", "#C9D1D9", "#9EA7B3"
    start_colour, prediction_colour = "#B1BAC4", "#9EA7B3"
    gd_colour, momentum_colour = "#B1BAC4", "#F0B72F"
    value_box = dict(boxstyle="round,pad=0.25", fc="#161B22", ec="#7A828E", lw=0.8)  # edges 4.9:1
    plane_alpha = 0.5
    table_edge, table_head, table_highlight = "#6E7681", "#21262D", "#4D3B00"
    backprop_colour, backprop_tint = "#CB9EFF", "#261A3D"  # 8.9:1, and 7.6:1 on its tint
    marker_edge, boundary = "#0D1117", "#F0F3F6"
    note_text, derivation_text, derivation_muted, point_neutral = "#B1BAC4", "#F0F3F6", "#9EA7B3", "#B1BAC4"
    glyph_fill, glyph_edge, node_fill, label_backing = "#0D1117", "#7A828E", "#0D1117", "#0D1117"
    reference_line, zero_line, landscape_line, plane_surface, path_colour = (
        "#6E7681",
        "#57606A",
        "#8B949E",
        "#8B949E",
        "#F0F3F6",
    )
    landscape_low = "#D1D9E0"
    spine = "#7A828E"
    control_button, control_selected = "#21262D", "#4F5864"  # ink on them 13.7:1 and 6.5:1


THEME = Theme()
DARK_THEME = DarkTheme()
THEMES = {"light": THEME, "dark": DARK_THEME}


def as_theme(theme):
    """A Theme from "light", "dark", a Theme, or None (light)."""
    if theme is None:
        return THEME
    if isinstance(theme, str):
        if theme not in THEMES:
            raise ValueError(f"slopewalk: theme must be 'light' or 'dark', not {theme!r}.")
        return THEMES[theme]
    return theme

# ------------------------ draw.py ------------------------

def halo(T):
    """An outline in the background colour, so that group letters stay legible over points."""
    return [patheffects.withStroke(linewidth=3, foreground=T.paper)]


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
    name, f, lim = act[:3]
    w_, h_ = 0.62, height
    ax.add_patch(Rectangle((xc - w_ / 2, yc - h_ / 2), w_, h_, fc=T.glyph_fill, ec=T.glyph_edge, lw=0.8, zorder=2))
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
    """(colour, tint, mixed, label) of group c: the class colours, black for mixed classes, grey for regression. The
    label of a mixed group is its most common class (the lower one in a tie)."""
    D, T = S.data, S.theme
    if D.task == "regression":
        return T.neutral_colour, T.neutral_tint, False, None
    labs = D.y[D.gsel[c]]
    yi = int(np.argmax(np.bincount(labs.numpy().astype(np.int64))))
    mixed = bool(labs.min() != labs.max())
    if mixed:
        return T.mixed_colour, T.mixed_tint, True, yi
    return T.class_colour[yi], T.class_tint[yi], False, yi


@contextmanager
def tagging(ax, key):
    """Tags the texts, patches and lines drawn in the block with what they show, for example ("z", 0) for the z boxes
    of the first layer, so that fade() can find them."""
    seen = {id(a) for a in [*ax.texts, *ax.patches, *ax.lines]}
    yield
    for a in [*ax.texts, *ax.patches, *ax.lines]:
        if id(a) not in seen:
            a.slopewalk_key = key


def fade(ax, keep, T):
    """Fades what a network panel shows but the current stage of the backward pass does not use: every tagged text,
    box and curve whose tag is not in keep (None: nothing fades). Numbers keep more of their strength than boxes."""
    if keep is None:
        return
    for a in [*ax.texts, *ax.patches, *ax.lines]:
        key = getattr(a, "slopewalk_key", None)
        if key is None or key in keep:
            continue
        a.set_alpha(T.faded_text_alpha if isinstance(a, Text) else T.faded_alpha)
        patch = a.get_bbox_patch() if hasattr(a, "get_bbox_patch") else None
        if patch is not None:
            patch.set_alpha(T.faded_alpha)
        if getattr(a, "arrow_patch", None) is not None:  # the arrows between boxes
            a.arrow_patch.set_alpha(T.faded_alpha)


def weight_colour(T, k, w):
    """The colour of a weight's line: its sign. Blue is kept for class 0; a weight moved by hand is marked by its
    label instead (its name, in bold)."""
    return T.positive if w[k] >= 0 else T.negative


def label_variants(S, F, edges, spec):
    """The label sets to try, in order: (mode, labels). Names and values for small networks, values only for larger
    ones; with values only, the weight moved last by a slider first keeps its name, so that its line can be found."""
    M, w = S.model, F.w
    ks = [k for k, _, _, _ in edges]
    if spec is not None:  # names, values, then values with fewer digits where the lines are crowded
        out = [("names", [spec[k][0] for k in ks])] if S.layout.show_names(len(w)) else []
        out.append(("values", [spec[k][1] for k in ks]))
        if any(len(spec[k]) > 3 for k in ks):
            out.append(("values", [spec[k][3] if len(spec[k]) > 3 else spec[k][1] for k in ks]))
        return out
    named = "{} = {:.2f}"
    if S.layout.show_names(len(w)):
        return [("names", [named.format(M.names[k], w[k]) for k in ks]), ("values", [f"{w[k]:.2f}" for k in ks])]
    out = []
    moved = getattr(S, "last_moved", None)
    if moved in ks:
        out.append(("values", [named.format(M.names[k], w[k]) if k == moved else f"{w[k]:.2f}" for k in ks]))
    return out + [("values", [f"{w[k]:.2f}" for k in ks])]


def draw_edge_labels(ax, S, F, edges, obstacles, spec=None):
    """The labels on the lines: names and values for small networks, values only for larger ones. When the labels
    of every line cannot be placed without overlaps, each layer gets its labels if they fit, from the first layer on
    (the values of the others stay on the sliders and in the table).
    spec: {k: (long text, short text, colour[, shorter text])} replaces the labels (only those lines get one), as in
    the backward pass. Returns the mode used: "names", "values" or "none"."""
    T = S.theme
    renderer = get_renderer(ax.figure)
    obstacles = obstacles + box_obstacles(ax, renderer, T.edge_label_size)
    if spec is not None:
        edges = [e for e in edges if e[0] in spec]
        if not edges:
            return "none"
    fit = fit_labels(ax, S, F, edges, obstacles, spec, renderer)
    if fit is not None:
        put_labels(ax, S, F, edges, fit[1], fit[2], spec)
        return fit[0]
    mode = "none"
    for layer in range(len(S.model.layers)):  # layer by layer, each clear of the labels placed before it
        part = [e for e in edges if S.model.where[e[0]][0] == layer]
        fit = fit_labels(ax, S, F, part, obstacles, spec, renderer) if part else None
        if fit is not None:
            put_labels(ax, S, F, part, fit[1], fit[2], spec)
            obstacles = obstacles + list(fit[3])
            mode = "values"
    return mode


def fit_labels(ax, S, F, edges, obstacles, spec, renderer):
    """The first label set of label_variants that fits on the lines edges: (mode, texts, positions, boxes) or None."""
    T = S.theme
    segments = [(p0, p1) for _, p0, p1, _ in edges]
    defaults = [t0 for _, _, _, t0 in edges]
    for variant, texts in label_variants(S, F, edges, spec):
        sizes = [text_size(ax, renderer, s, T.edge_label_size) for s in texts]
        places, boxes, worst_labels, worst_boxes = place_labels(segments, sizes, obstacles, defaults, pad=0.03)
        if max(worst_labels, worst_boxes) <= LABEL_TOLERANCE:
            return variant, texts, places, boxes
    return None


def put_labels(ax, S, F, edges, labels, positions, spec):
    """Draws the labels of the lines edges at their positions."""
    T, w = S.theme, F.w
    for (k, _, _, _), label, (cx, cy) in zip(edges, labels, positions):
        frozen = k not in F.moving
        hot = k in F.by_hand and not frozen and spec is None
        colour = T.frozen_text if frozen else (T.positive if w[k] >= 0 else T.negative_text)
        if spec is not None:
            colour = spec[k][2]
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
            bbox=dict(boxstyle="round,pad=0.08", fc=T.label_backing, ec="none", alpha=0.9),
        )


def draw_shared_note(ax, S):
    """Network panels after the first: their lines carry no numbers, because the weights are those of the first
    panel (A, or the first group shown)."""
    T = S.theme
    note = f"weights as in {letter(S.shown[0])}"
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=T.glyph_name_size, color=T.curve)


def style_panel(ax, T):
    """The panels below the networks in the theme's colours and sizes: background, frame, ticks, axis labels, titles,
    legend, and the panes of a 3-D panel. Set after drawing, so that tick labels made later follow too."""
    ax.set_facecolor(T.paper)
    for spine in ax.spines.values():
        spine.set_color(T.spine)
    ax.tick_params(labelsize=T.tick_size, colors=T.muted)
    axes = [ax.xaxis, ax.yaxis]
    if hasattr(ax, "zaxis"):
        axes.append(ax.zaxis)
        ax.tick_params(axis="z", labelsize=T.tick_size, colors=T.muted)
    for axis in axes:
        axis.label.set_size(T.axis_label_size)
        axis.label.set_color(T.ink)
        if hasattr(axis, "set_pane_color"):
            axis.set_pane_color(to_rgba(T.value_box["fc"]))
    for title in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
        if title is not None:
            title.set_color(T.ink)
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor(T.paper)
        legend.get_frame().set_edgecolor(T.reference_line)
        for text in legend.get_texts():
            text.set_color(T.ink)


def draw_case(ax, S, F, c, styles=None):
    """One copy of the network for group c: the lines with their weights, the units, the output and the loss.

    The labels on the lines are drawn afterwards by draw_edge_labels, once the figure's layout is fixed.
    styles: {k: (colour or None, alpha, width or None)} overrides the lines' look (the backward pass). The boxes are
    tagged with what they show (see tagging), so that the backward pass can fade those its stage does not use.
    Returns (edges, obstacles).
    """
    M, D, L, T = S.model, S.data, S.layout, S.theme
    out, w = F.out, F.w
    col, tint, mixed, yi = group_style(S, c)
    arrow = dict(arrowstyle="-|>", color=T.arrow, lw=1.0, shrinkA=0, shrinkB=0)
    node = dict(fontsize=T.node_size, ha="center", va="center", zorder=4, color=T.ink)
    value = dict(fontsize=T.value_size, ha="center", va="center", zorder=5, bbox=T.value_box, color=T.ink)
    h_value = dict(value, bbox=dict(T.value_box, ec=T.hidden, lw=1.2))  # the hidden layer: a green border
    edges, obstacles = [], []  # edges: (k, source, target, preferred label fraction); obstacles: boxes to keep clear
    # in the backpropagation view the names of the values are written once, in panel A; the other panels show values
    tag = (lambda name: f"{name}\n") if c == S.shown[0] or F.stage is None else (lambda name: "")
    src = [(0.12, yy) for yy in L.y_inputs]
    src_x, k = 0.0, 0
    for l, layer in enumerate(M.layers):
        last = l == len(M.layers) - 1
        tx = (L.x_a if last else L.blocks[l][0]) - BW
        tys = L.unit_ys(layer.out_features)  # one output unit sits at 1.2, the middle
        bias = (src_x if l == 0 else src_x + 0.45, 2.8)
        with tagging(ax, ("bias", l)):
            ax.text(*bias, "1", bbox=dict(boxstyle="circle", fc=T.node_fill, ec=T.node_edge), **node)
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
                with tagging(ax, ("z", l)):
                    ax.text(XZ, ty, tag(f"z{nm}") + num(zm[j], zs[j]), **value)
                with tagging(ax, ("act", l)):
                    if layer.act:
                        ax.annotate("", xy=(XG - 0.31, ty), xytext=(XZ + BW, ty), arrowprops=arrow)
                        obstacles.append(glyph(ax, XG, ty, layer.act, zm[j], zs[j], T, height=glyph_height))
                        ax.annotate("", xy=(XH - BW, ty), xytext=(XG + 0.31, ty), arrowprops=arrow)
                    else:
                        ax.annotate("", xy=(XH - BW, ty), xytext=(XZ + BW, ty), arrowprops=arrow)
                with tagging(ax, ("h", l)):
                    ax.text(XH, ty, tag(f"h{nm}") + num(hm[j], hs[j]), **h_value)
            src = [(XH + BW, ty) for ty in tys]
            src_x = XH
    xm, xs = D.stats(D.X.numpy(), c)
    with tagging(ax, ("x",)):
        for i, yy in enumerate(L.y_inputs):
            ax.text(0, yy, f"x{i + 1}", bbox=dict(boxstyle="circle", fc=tint, ec=col), **node)
            ax.text(
                L.x_input_value,
                yy,
                num(xm[i], xs[i]),
                fontsize=T.node_size,
                ha="right",
                va="center",
                color=T.ink,
                bbox=dict(boxstyle="round,pad=0.25", fc=tint, ec=col, lw=0.8),
            )
    (am, as_), (lm, _) = D.stats(out.a, c), D.stats(out.loss, c)
    npts = int(D.gsel[c].sum())
    where = "(" + ", ".join(f"{v:.1f}".replace("-0.0", "0.0") if abs(v) < 0.05 else f"{v:.1f}" for v in xm) + ")"
    lead = f"{letter(c)}: {'mean of ' + str(npts) + ' points ' if npts > 1 else ''}{where}"
    change = f"change\nfrom start\n{lm - S.loss0[c]:+.3f}"
    loss_box = dict(boxstyle="round,pad=0.3", fc=tint, ec=col, lw=0.8)
    if D.task == "multiclass":
        title = draw_classes(ax, S, F, c, tag, obstacles, arrow, (lead, change, loss_box))
    elif D.task == "regression" and M.n_outputs > 1:
        title = draw_targets(ax, S, F, c, tag, arrow, (lead, change, loss_box))
    elif D.task == "regression":
        box_a = dict(fontsize=T.node_size, ha="center", va="center", zorder=5, bbox=T.value_box, color=T.ink)
        with tagging(ax, ("a",)):
            ax.text(L.x_a, 1.2, tag("ŷ") + num(am, as_), **box_a)
        ax.annotate("", xy=(L.x_loss - 0.42 * L.text_scale, 1.2), xytext=(L.x_a + BW, 1.2), arrowprops=arrow)
        ym, ys = D.stats(D.y.numpy(), c)
        ytxt = f"y = {ym:+.2f}" if npts == 1 else f"y {ym:+.2f}\n±{ys:.2f}"
        with tagging(ax, ("loss",)):
            ax.text(
                L.x_loss,
                1.2,
                f"{ytxt}\nloss {lm:.3f}\n{change}",
                fontsize=T.node_size,
                ha="center",
                va="center",
                color=T.ink,
                bbox=loss_box,
            )
        title = f"{lead}, y = {ym:+.2f}:  loss {lm:.3f}"
    else:
        pm, ps = D.stats(out.p, c)
        box_a = dict(fontsize=T.node_size, ha="center", va="center", zorder=5, bbox=T.value_box, color=T.ink)
        with tagging(ax, ("a",)):
            ax.text(L.x_a, 1.2, tag("a") + num(am, as_), **box_a)
        with tagging(ax, ("out_act",)):
            ax.annotate("", xy=(L.x_sigmoid - 0.31, 1.2), xytext=(L.x_a + BW, 1.2), arrowprops=arrow)
            obstacles.append(glyph(ax, L.x_sigmoid, 1.2, SIGMOID, am, as_, T))
            ax.annotate("", xy=(L.x_p - BW, 1.2), xytext=(L.x_sigmoid + 0.31, 1.2), arrowprops=arrow)
        with tagging(ax, ("p",)):
            ax.text(L.x_p, 1.2, tag("p") + num(pm, ps, sign=False), **box_a)
        ytxt = "y mixed" if mixed else f"y = {yi}"
        with tagging(ax, ("loss",)):
            ax.text(
                L.x_loss,
                1.2,
                f"{ytxt}\nloss {lm:.3f}\n{change}",
                fontsize=T.node_size,
                ha="center",
                va="center",
                color=T.ink,
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
        colour, alpha = weight_colour(T, k, w), T.frozen_alpha if frozen else 1.0
        width = min(T.line_width + T.line_slope * abs(w[k]), T.line_max)
        if styles is not None:
            colour = styles[k][0] or weight_colour(T, k, w)
            alpha = styles[k][1] * (T.frozen_alpha if frozen else 1.0)
            width = styles[k][2] or width
        ax.plot(
            [sx, tx],
            [sy, ty],
            color=colour,
            lw=width,
            alpha=alpha,
            zorder=1,
            solid_capstyle="round",
        )
    return edges, obstacles


def draw_targets(ax, S, F, c, tag, arrow, parts):
    """The output of network panel c for K targets (regression): a prediction box ŷ1, ŷ2, ... per target, each with
    an arrow into the loss box, which lists the targets y1, y2, ... (the loss is the mean of the K squared errors).
    Returns the panel title."""
    M, D, L, T = S.model, S.data, S.layout, S.theme
    lead, change, loss_box = parts
    K = M.n_outputs
    (am, as_), (ym, ys), (lm, _) = D.stats(F.out.a, c), D.stats(D.y.numpy(), c), D.stats(F.out.loss, c)
    box = dict(fontsize=T.node_size, ha="center", va="center", zorder=5, bbox=T.value_box, color=T.ink)
    end = (L.x_loss - 0.5 * L.text_scale, 1.2)
    for k, ty in enumerate(L.unit_ys(K)):
        with tagging(ax, ("a",)):
            ax.text(L.x_a, ty, tag(f"ŷ{k + 1}") + num(am[k], as_[k]), **box)
        ax.annotate("", xy=end, xytext=(L.x_a + BW, ty), arrowprops=arrow)
    one = int(D.gsel[c].sum()) == 1
    targets = [f"y{k + 1} = {ym[k]:+.2f}" if one else f"y{k + 1} {ym[k]:+.2f} ±{ys[k]:.2f}" for k in range(K)]
    with tagging(ax, ("loss",)):
        ax.text(
            L.x_loss,
            1.2,
            "\n".join(targets) + f"\nloss {lm:.3f}\n{change}",
            fontsize=T.node_size,
            ha="center",
            va="center",
            color=T.ink,
            bbox=loss_box,
        )
    return f"{lead}:  loss {lm:.3f}"


def draw_classes(ax, S, F, c, tag, obstacles, arrow, parts):
    """The output of network panel c for K classes: a score box a0, a1, ... per class, the softmax over all of them,
    a p box per class, and the loss box. The p box of the group's own class has the class colours: its p is the one
    the loss takes (loss = -log p of the case's class). Returns the panel title."""
    M, D, L, T = S.model, S.data, S.layout, S.theme
    lead, change, loss_box = parts
    col, tint, mixed, yi = group_style(S, c)
    tys = L.unit_ys(M.n_outputs)
    (am, as_), (pm, ps), (lm, _) = D.stats(F.out.a, c), D.stats(F.out.p, c), D.stats(F.out.loss, c)
    box = dict(fontsize=T.node_size, ha="center", va="center", zorder=5, bbox=T.value_box, color=T.ink)
    half = 0.31  # the softmax box: as wide as a glyph, as tall as the scores
    lo, hi = tys[-1] - 0.25, tys[0] + 0.25
    with tagging(ax, ("out_act",)):  # named below the box, like the curves of the activations
        ax.add_patch(Rectangle((L.x_sigmoid - half, lo), 2 * half, hi - lo, fc=T.glyph_fill, ec=T.glyph_edge, lw=0.8))
        ax.text(L.x_sigmoid, lo - 0.04, "softmax", fontsize=T.glyph_name_size, ha="center", va="top", color=T.curve)
    obstacles.append((L.x_sigmoid - half, lo - 0.16, L.x_sigmoid + half, hi))
    for k, ty in enumerate(tys):
        with tagging(ax, ("a",)):
            ax.text(L.x_a, ty, tag(f"a{k}") + num(am[k], as_[k]), **box)
        with tagging(ax, ("out_act",)):
            ax.annotate("", xy=(L.x_sigmoid - half, ty), xytext=(L.x_a + BW, ty), arrowprops=arrow)
            ax.annotate("", xy=(L.x_p - BW, ty), xytext=(L.x_sigmoid + half, ty), arrowprops=arrow)
        own = not mixed and k == yi
        style = dict(box, bbox=dict(T.value_box, fc=tint, ec=col, lw=1.6)) if own else box
        with tagging(ax, ("p",)):
            ax.text(L.x_p, ty, tag(f"p{k}") + num(pm[k], ps[k], sign=False), **style)
    ytxt = "y mixed" if mixed else f"y = {yi}"
    with tagging(ax, ("loss",)):
        ax.text(
            L.x_loss,
            1.2,
            f"{ytxt}\nloss {lm:.3f}\n{change}",
            fontsize=T.node_size,
            ha="center",
            va="center",
            color=T.ink,
            bbox=loss_box,
        )
    what = "classes mixed" if mixed else f"class {yi}"
    return f"{lead}, {what}:  loss {lm:.3f}"


# ---------------------------------------------------------------- the input plane


def net_on(S, points, w):
    """The network's output on an array of points at the weights w: the score or the prediction (one number per
    point), or the scores of the classes (multiclass: a row per point). Leaves the network at w."""
    S.model.set_weights(w)
    with torch.no_grad():
        out = S.model.net(torch.as_tensor(points, dtype=S.model.dtype))
    return out.numpy() if out.shape[1] > 1 else out.reshape(-1).numpy()


def softmax(A):
    """The probabilities of the classes from their scores, along the last axis."""
    E = np.exp(A - A.max(-1, keepdims=True))
    return E / E.sum(-1, keepdims=True)


def class_shading(T, A):
    """An RGB image of a grid of scores A (ny x nx x K): at each point the colour of the class with the largest score,
    the stronger the surer, in five bands as in the binary picture: paper where all classes are equally likely, a share
    plane_alpha of the class colour where one class is certain."""
    K = A.shape[-1]
    P = softmax(A)
    sure = (P.max(-1) - 1 / K) / (1 - 1 / K)
    band = (np.minimum(np.floor(sure * 5), 4) + 0.5) / 5
    colours = np.array([to_rgb(T.class_colour[k]) for k in range(K)])
    share = (T.plane_alpha * band)[..., None]
    return np.asarray(to_rgb(T.paper)) * (1 - share) + colours[P.argmax(-1)] * share


def class_boundaries(ax, U1, U2, A, **style):
    """The boundaries between the classes over a grid of scores A (ny x nx x K): where class k's score equals the
    largest of the others, for each k (a boundary between two classes is drawn once for each)."""
    K = A.shape[-1]
    for k in range(K):
        d = A[..., k] - np.delete(A, k, axis=-1).max(-1)
        if d.min() < 0 < d.max():
            ax.contour(U1, U2, d, levels=[0.0], **style)


def test_handle(S):
    """The legend entry of the test data, or None without test data."""
    if S.test is None:
        return None
    return Line2D([], [], marker="x", ls="none", color=S.theme.ink, markersize=6, label="test data")


def class_legend(ax, S, winners=None):
    """A legend of the class colours. winners: the classes that are the most likely somewhere in the picture; the
    others are marked "no region", so that a class missing from the picture is not taken for a drawing error."""
    T = S.theme
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            color=T.class_colour[k],
            mec=T.marker_edge,
            label=f"class {k}" if winners is None or k in winners else f"class {k}: no region",
        )
        for k in range(S.data.n_classes)
    ]
    handles += [h for h in [test_handle(S)] if h is not None]
    ax.legend(handles=handles, fontsize=T.legend_size, loc="best")


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


def test_marks(ax, S, where, legend=True):
    """The test data as crosses: where(X_test, y_test) gives (x, y, colours) of the marks; legend: name them once."""
    if S.test is None:
        return
    x, y, colours = where(S.test.X.numpy(), S.test.y.numpy())
    ax.scatter(x, y, marker="x", s=22, c=colours, linewidths=1.0, zorder=3, label="test data" if legend else None)


def draw_group_marks(ax, S, xs, ys, size=160):
    """The group means as large markers with their letters (class colours, or white for regression)."""
    D, T = S.data, S.theme
    for c in range(D.G):
        col, _, _, yi = group_style(S, c)
        face = "white" if D.task == "regression" else T.class_colour[yi]
        ax.scatter(xs[c], ys[c], s=size, color=face, edgecolor=T.marker_edge, zorder=4)
        ax.text(
            xs[c],
            ys[c],
            f"  {letter(c)}",
            fontsize=T.letter_size,
            va="bottom",
            zorder=5,
            path_effects=halo(T),
            color=T.ink,
        )


def draw_input_plane(ax, S, F):
    """The panel below the networks that shows the inputs: by the task and the number of inputs, p for class 1 or
    the prediction against x1 (one input), the plane (two inputs), the most likely class (several classes), each
    target's curve or the prediction against the target (several targets, or more inputs than a plane shows)."""
    D, T = S.data, S.theme
    X = D.X.numpy()
    gx = [D.stats(X, c)[0] for c in range(D.G)]
    if D.task == "multiclass" and D.d <= 2:
        draw_class_plane(ax, S, F, gx)
    elif D.task == "regression" and (S.model.n_outputs > 1 or D.d > 2):
        if D.d == 1:
            draw_target_curves(ax, S, F)
        else:  # more inputs than a plane shows, or several targets over a plane
            draw_parity(ax, S, F)
    elif D.d == 1:
        draw_curve(ax, S, F, gx)
    elif D.d == 2:
        draw_plane(ax, S, F, gx)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{D.d} inputs: no picture of the input plane", ha="center", va="center", color=T.muted)
    S.model.set_weights(F.w)


def draw_curve(ax, S, F, gx):
    """One input: p for class 1 (binary) or the prediction (regression) against x1, with the points and the groups'
    means; once the weights moved, the starting curve, dashed."""
    D, T = S.data, S.theme
    w = F.w
    moved = bool(F.changed)
    X, y = D.X.numpy(), D.y.numpy()
    u = line_grid(S)
    a = net_on(S, u[:, None], w)
    out = 1 / (1 + np.exp(-a)) if D.task == "binary" else a
    if moved:
        a0 = start_values(S, "line", u[:, None])
        out0 = 1 / (1 + np.exp(-a0)) if D.task == "binary" else a0
        ax.plot(u, out0, "--", color=T.start_colour, lw=1.0, label="start")
    ax.plot(u, out, color=T.neutral_colour, lw=2, label="p for class 1" if D.task == "binary" else "prediction")
    if D.task == "binary":
        colours = [T.class_colour[int(v)] for v in y]
        ax.scatter(X[:, 0], y, c=colours, s=18, edgecolor=T.marker_edge, lw=0.3, zorder=3)
        test_marks(ax, S, lambda Xt, yt: (Xt[:, 0], yt, [T.class_colour[int(v)] for v in yt]))
        ax.axhline(0.5, color=T.reference_line, lw=0.8, zorder=0)
        gy = [float(D.stats(y, c)[0]) for c in range(D.G)]
        ax.set_ylim(-0.08, 1.08)
        ax.set_ylabel("p")
        ax.set_title("p for class 1 against x1", fontsize=T.panel_title_size)
    else:
        ax.scatter(X[:, 0], y, s=18, color=T.point_neutral, zorder=3)
        test_marks(ax, S, lambda Xt, yt: (Xt[:, 0], yt, T.point_neutral))
        gy = [float(D.stats(y, c)[0]) for c in range(D.G)]
        ax.set_ylabel("y")
        ax.set_title("ŷ against x1, and the points", fontsize=T.panel_title_size)
    draw_group_marks(ax, S, [float(v[0]) for v in gx], gy, size=110)
    ax.set_xlabel("x1")
    if moved or S.test is not None:
        ax.legend(fontsize=T.legend_size, loc="best")


def draw_plane(ax, S, F, gx):
    """Two inputs. Binary: p for class 1 over the plane, the boundary p = 0.5 and the starting boundary (dashed).
    Regression: the prediction over the plane, with the points coloured by y."""
    D, T = S.data, S.theme
    w = F.w
    moved = bool(F.changed)
    X, y = D.X.numpy(), D.y.numpy()
    P1, P2, grid = plane_grid(S)
    a = net_on(S, grid, w).reshape(P1.shape)
    if D.task == "binary":
        P = 1 / (1 + np.exp(-a))
        ax.contourf(P1, P2, P, levels=np.linspace(0, 1, 11), cmap=T.plane_colormap(), alpha=T.plane_alpha)
        if P.min() < 0.5 < P.max():
            ax.contour(P1, P2, P, levels=[0.5], colors=T.boundary)
        if moved:
            a0 = start_values(S, "plane", grid).reshape(P1.shape)
            if a0.min() < 0 < a0.max():
                ax.contour(P1, P2, a0, levels=[0.0], colors=T.start_colour, linestyles="--", linewidths=1.0)
        if D.n > D.G:
            ax.scatter(X[:, 0], X[:, 1], c=[T.class_colour[int(v)] for v in y], s=4)
        test_marks(ax, S, lambda Xt, yt: (Xt[:, 0], Xt[:, 1], [T.class_colour[int(v)] for v in yt]))
        for c in range(D.G):
            yi = group_style(S, c)[3]
            ax.scatter(*gx[c], s=160, color=T.class_colour[yi], edgecolor=T.marker_edge, zorder=3)
            ax.text(
                gx[c][0] + 0.15, gx[c][1] + 0.15, letter(c), fontsize=T.letter_size, path_effects=halo(T), color=T.ink
            )
        ax.set_title("p for class 1 over the input plane,\nand the boundary p = 0.5", fontsize=T.panel_title_size)
        if S.test is not None:
            ax.legend(fontsize=T.legend_size, loc="best")
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
            edgecolor=T.marker_edge,
            lw=0.4,
            zorder=3,
        )
        for c in range(D.G):
            ax.text(
                gx[c][0] + 0.15, gx[c][1] + 0.15, letter(c), fontsize=T.letter_size, path_effects=halo(T), color=T.ink
            )
        ax.set_title("ŷ over the input plane;\npoints coloured by y", fontsize=T.panel_title_size)
    ax.set_aspect("equal")


def draw_class_plane(ax, S, F, gx):
    """Multiclass. Two inputs: the most likely class over the plane, shaded by how sure the network is, with the
    boundaries between the classes and, once the weights moved, the starting ones (dashed). One input: p of each class
    against x1, each case on the curve of its own class, so that its height is the p its loss takes."""
    D, T = S.data, S.theme
    w = F.w
    moved = bool(F.changed)
    X, y = D.X.numpy(), D.y.numpy()
    K = D.n_classes
    colours = [T.class_colour[int(v)] for v in y]
    if D.d == 1:
        u = line_grid(S)
        P = softmax(net_on(S, u[:, None], w))
        if moved:
            P0 = softmax(start_values(S, "line", u[:, None]))
            for k in range(K):
                ax.plot(u, P0[:, k], "--", color=T.class_colour[k], lw=1.0, alpha=0.6)
        for k in range(K):
            ax.plot(u, P[:, k], color=T.class_colour[k], lw=2, label=f"p{k}")
        own = F.out.p[np.arange(D.n), y.astype(np.int64)]
        ax.scatter(X[:, 0], own, c=colours, s=18, edgecolor=T.marker_edge, lw=0.3, zorder=3)
        if S.test is not None:  # the test cases on the curve of their class as well
            Pt = softmax(net_on(S, S.test.X.numpy(), F.w))
            yt = S.test.y.numpy().astype(np.int64)
            test_marks(
                ax, S, lambda Xt, _: (Xt[:, 0], Pt[np.arange(len(yt)), yt], [T.class_colour[int(v)] for v in yt])
            )
        draw_group_marks(ax, S, [float(v[0]) for v in gx], [float(D.stats(own, c)[0]) for c in range(D.G)], size=110)
        ax.set_ylim(-0.08, 1.08)
        ax.set_xlabel("x1")
        ax.set_ylabel("p")
        ax.legend(fontsize=T.legend_size, loc="best")
        ax.set_title("p of each class against x1;\neach case on the curve of its class", fontsize=T.panel_title_size)
        return
    P1, P2, grid = plane_grid(S)
    A = net_on(S, grid, w).reshape(*P1.shape, K)
    lo, hi = float(P1[0, 0]), float(P1[0, -1])
    ax.imshow(class_shading(T, A), extent=(lo, hi, lo, hi), origin="lower", interpolation="bilinear", zorder=0)
    class_boundaries(ax, P1, P2, A, colors=T.boundary)
    if moved:
        A0 = start_values(S, "plane", grid).reshape(*P1.shape, K)
        class_boundaries(ax, P1, P2, A0, colors=T.start_colour, linestyles="--", linewidths=1.0)
    if D.n > D.G:
        ax.scatter(X[:, 0], X[:, 1], c=colours, s=4)
    test_marks(ax, S, lambda Xt, yt: (Xt[:, 0], Xt[:, 1], [T.class_colour[int(v)] for v in yt]))
    for c in range(D.G):
        yi = group_style(S, c)[3]
        ax.scatter(*gx[c], s=160, color=T.class_colour[yi], edgecolor=T.marker_edge, zorder=3)
        ax.text(gx[c][0] + 0.15, gx[c][1] + 0.15, letter(c), fontsize=T.letter_size, path_effects=halo(T), color=T.ink)
    class_legend(ax, S, set(np.unique(A.argmax(-1)).tolist()))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title("the most likely class over the input plane,\nand the boundaries", fontsize=T.panel_title_size)


LINE_STYLES, MARKERS = ("-", "--", ":", "-."), ("o", "s", "^", "D")  # one per target: regression stays neutral


def target_marks(T, k):
    """The marker of target k: its own shape, filled for targets 1 and 3, hollow for 2 and 4, so that the points of
    two targets differ by more than a small shape."""
    hollow = k % 2 == 1
    return dict(
        marker=MARKERS[k],
        facecolor=T.paper if hollow else T.point_neutral,
        edgecolor=T.point_neutral if hollow else T.marker_edge,
        linewidths=1.2 if hollow else 0.4,
    )


def target_handle(T, k, line=True, label=None):
    """A legend entry for target k: its line style and its marker together."""
    m = target_marks(T, k)
    return Line2D(
        [],
        [],
        linestyle=LINE_STYLES[k] if line else "none",
        color=T.neutral_colour,
        lw=2,
        marker=m["marker"],
        markersize=7,
        markerfacecolor=m["facecolor"],
        markeredgecolor=m["edgecolor"],
        label=label or f"ŷ{k + 1} and y{k + 1}",
    )


def draw_target_curves(ax, S, F):
    """Regression, one input, several targets: the prediction of each target against x1, one line style each, with
    the points of that target, one marker each; the legend pairs each line with its marker. Once the weights moved,
    the starting curves, thin."""
    D, T = S.data, S.theme
    u = line_grid(S)
    A = net_on(S, u[:, None], F.w)
    X, Y = D.X.numpy(), D.y.numpy()
    handles = []
    if F.changed:
        A0 = start_values(S, "line", u[:, None])
        for k in range(A.shape[1]):
            ax.plot(u, A0[:, k], LINE_STYLES[k], color=T.start_colour, lw=1.0)
        handles.append(Line2D([], [], color=T.start_colour, lw=1.0, label="start"))
    for k in range(A.shape[1]):
        ax.plot(u, A[:, k], LINE_STYLES[k], color=T.neutral_colour, lw=2, label=f"ŷ{k + 1}")
        ax.scatter(X[:, 0], Y[:, k], s=40, zorder=3, **target_marks(T, k))
        test_marks(ax, S, lambda Xt, Yt, k=k: (Xt[:, 0], Yt.reshape(len(Xt), -1)[:, k], T.point_neutral), k == 0)
        handles.append(target_handle(T, k))
    handles += [h for h in [test_handle(S)] if h is not None]
    ax.set_xlabel("x1")
    ax.set_ylabel("ŷ and y")
    ax.legend(handles=handles, fontsize=T.legend_size, loc="best")
    ax.set_title("ŷ of each target against x1,\nand the points", fontsize=T.panel_title_size)


def draw_parity(ax, S, F):
    """Regression with several targets over two inputs, or with more inputs than a plane shows: each case's
    prediction against its target, one marker per target, and the diagonal where they are equal. With several groups,
    a cross at each group's mean (first target) with its letter beside it."""
    D, T = S.data, S.theme
    Y = D.y.numpy().reshape(D.n, -1)
    A = np.asarray(F.out.a, dtype=float).reshape(D.n, -1)
    K = Y.shape[1]
    lo, hi = float(min(Y.min(), A.min())), float(max(Y.max(), A.max()))
    pad = 0.08 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color=T.muted, lw=1.4, zorder=1)
    handles = [Line2D([], [], color=T.muted, lw=1.4, label="ŷ = y")]
    for k in range(K):
        ax.scatter(Y[:, k], A[:, k], s=36 if D.n <= 40 else 16, zorder=3, **target_marks(T, k))
    if S.test is not None:
        Yt = S.test.y.numpy().reshape(S.test.n, -1)
        At = np.asarray(net_on(S, S.test.X.numpy(), F.w), dtype=float).reshape(S.test.n, -1)
        for k in range(K):
            test_marks(ax, S, lambda *_, k=k: (Yt[:, k], At[:, k], T.point_neutral), k == 0)
        handles.append(target_handle(T, k, line=False, label=f"target {k + 1}" if K > 1 else "cases"))
    if D.G > 1:
        for c in range(D.G):
            sel = D.gsel[c]
            my, ma = float(Y[sel, 0].mean()), float(A[sel, 0].mean())
            ax.plot([my], [ma], "+", color=T.ink, ms=10, mew=1.6, zorder=4)
            ax.annotate(
                letter(c),
                xy=(my, ma),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=T.letter_size,
                path_effects=halo(T),
                color=T.ink,
                zorder=5,
            )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    handles += [h for h in [test_handle(S)] if h is not None]
    ax.set_xlabel("target y")
    ax.set_ylabel("prediction ŷ")
    ax.legend(handles=handles, fontsize=T.legend_size, loc="upper left")
    ax.set_title("ŷ against the target y;\non the diagonal ŷ = y", fontsize=T.panel_title_size)


# ---------------------------------------------------------------- R over the steps


def draw_history(ax, S, F):
    """R over the gradient steps taken, the first-order prediction of each step (hollow markers) and a note on the
    last step and the next one."""
    T = S.theme
    history = F.history
    ax.plot(range(len(history)), history, "o-", color=T.ink, ms=4, label="R after the step")
    test = [v for v in F.history_test if v is not None]
    if test:
        ax.plot(range(len(test)), test, "x--", color=T.muted, ms=5, lw=1.2, label="R on the test data")
    predicted = [(k + 1, p.predicted) for k, p in enumerate(F.predictions)]
    if S.log_scale:
        predicted = [(k, v) for k, v in predicted if v > 0]
    if predicted:
        ks, vs = zip(*predicted)
        ax.plot(ks, vs, "o", mfc=T.paper, mec=T.prediction_colour, ms=6, ls="none", label="first-order prediction")
    if predicted or test:
        ax.legend(fontsize=T.legend_size, loc="lower left")
    ax.set_xlabel("gradient steps taken")
    ax.set_ylabel("R")
    values = list(history) + [v for _, v in predicted] + test
    if S.log_scale:
        positive = [v for v in values if v > 0] or [1.0]
        ax.set_yscale("log")
        ax.set_ylim(min(positive) * 0.7, max(positive) * 1.4)
    else:  # fitted to the values, so that progress shows; R is never below 0
        lo, hi = min(values), max(values)
        pad = 0.12 * (hi - lo) if hi > lo else max(0.1 * abs(hi), 0.01)
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.set_title(f"R = {F.R:.4f}", fontsize=T.panel_title_size)
    if len(history) < 12:
        ax.set_xticks(range(len(history)))
    notes = [f"next step, eta {F.eta:g}: predicted change {F.next_change:+.4f}"]
    if F.predictions:
        last = F.predictions[-1]
        notes.append(
            f"last step: predicted {last.predicted - last.before:+.4f}, actual {last.actual - last.before:+.4f}"
        )
    ax.text(
        0.98,
        0.97,
        "\n".join(notes),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=T.note_size,
        color=T.note_text,
    )


# ---------------------------------------------------------------- the slope table


def height_inches(ax):
    """The height of a panel in inches, from the height ratios of the figure's grid: the same for every frame."""
    spec = ax.get_subplotspec()
    if spec is None:
        return BOTTOM_HEIGHT
    top = spec.get_topmost_subplotspec()
    grid = top.get_gridspec()
    ratios = list(grid.get_height_ratios() or [1] * grid.nrows)
    return sum(ratios[top.rowspan.start : top.rowspan.stop]) / sum(ratios) * ax.figure.get_figheight()


def width_share(ax):
    """The panel's share of the width of its row, from the width ratios of its grid: the same for every frame,
    before and after the layout."""
    spec = ax.get_subplotspec()
    if spec is None:
        return 1 / 3
    grid = spec.get_gridspec()
    ratios = list(grid.get_width_ratios() or [1] * grid.ncols)
    return sum(ratios[spec.colspan.start : spec.colspan.stop]) / sum(ratios)


TABLE_SHARES = (0.17, 0.18, 0.39, 0.26)  # the columns' shares of a block of the table
TABLE_HEADERS = (  # the longest that fit; "weight" always
    ("weight", "value", "change of R for +0.1", "slope dR/dw"),
    ("weight", "value", "ΔR for +0.1", "dR/dw"),
)


def table_fit(ax, rows, block_inches, most):
    """The font, headers and column shares of a block of the table that is block_inches wide, with the font at most
    `most` points (what the rows allow). The usual shares with the longest headers that fit at that font; when none
    fit, shares in proportion to what each column needs, with the headers that allow the largest font."""
    renderer = get_renderer(ax.figure)
    dpi = ax.figure.dpi

    def inches(text, size, **font):  # the width of a text at a font size, in inches
        if renderer is None:
            return 0.6 * size / 72 * len(text)
        props = FontProperties(size=size, **font)
        return renderer.get_text_width_height_descent(text, props, ismath=False)[0] / dpi

    cells = [max(inches(r[c], 10, family="monospace" if c else None) for r in rows) / 10 for c in range(4)]

    def needs(headers):  # inches per point of font, per column: numbers right-aligned with a margin, headers centred
        heads = [inches(h, 9, weight="bold") / 10 for h in headers]
        return [max(cell / 0.8, head / 0.9) for cell, head in zip(cells, heads)]

    for headers in TABLE_HEADERS:
        if all(n * most <= s * block_inches for n, s in zip(needs(headers), TABLE_SHARES)):
            return most, list(headers), list(TABLE_SHARES)
    options = [(min(most, block_inches / sum(needs(h))), h) for h in TABLE_HEADERS]
    size, headers = max(options, key=lambda o: o[0])  # the first of the largest: the longer headers in a tie
    need = needs(headers)
    return size, list(headers), [n / sum(need) for n in need]


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
    space = height_inches(ax) * 72 * TABLE_SHARE  # points of the panel's height for the rows
    rows_needed = max(len(b) for b in blocks) + 1  # the header and the longest block
    gap = 0.03
    width = (1 - gap * (n_blocks - 1)) / n_blocks
    block_inches = (ax.figure.get_figwidth() * width_share(ax) - PANEL_GAP) * width
    fontsize, headers, shares = table_fit(
        ax, rows, block_inches, min(T.table_size, space / rows_needed / T.table_leading)
    )
    fontsize = max(8.0, fontsize)
    ax.set_title("Change of R when one weight grows by 0.1,\nand the slope dR/dw", fontsize=T.panel_title_size)
    for b, block in enumerate(blocks):
        height = min(1.0, fontsize * T.table_leading * (len(block) + 1) / space)
        tbl = ax.table(
            cellText=block,
            colLabels=headers,
            colWidths=shares,
            cellLoc="right",
            colLoc="center",
            bbox=[b * (width + gap), 1 - height, width, height],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(T.table_edge)
            cell.set_facecolor(T.paper)
            text = cell.get_text()
            text.set_color(T.ink)
            if r == 0:
                cell.set_facecolor(T.table_head)
                text.set_fontweight("bold")
                text.set_fontsize(0.9 * fontsize)  # the headers a little smaller than the numbers
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

VIEWS = ("backprop", "cases", "slice", "line", "hidden", "momentum")


def message(ax, T, text):
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, color=T.muted)


def point_colours(S):
    """A colour per case: its class colour (classification) or a colour for its target y (regression)."""
    D, T = S.data, S.theme
    y = D.y.numpy()
    if D.task in ("binary", "multiclass"):
        return [T.class_colour[int(v)] for v in y]
    if y.ndim > 1:  # several targets: no single value to colour by
        return [T.point_neutral] * len(y)
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
            edgecolor=T.marker_edge,
            lw=0.3,
            zorder=3,
        )
        ax.hlines(float(v[idx].mean()), c - 0.32, c + 0.32, color=T.ink, lw=2, zorder=4)
    avg = float(v.mean())
    ax.axhline(0.0, color=T.zero_line, lw=0.8, zorder=0)
    ax.axhline(avg, ls="--", color=T.ink, lw=1.5, zorder=2, label=f"average {avg:+.4f} = dR/d{name}")
    ax.set_xticks(range(D.G))
    ax.set_xticklabels([letter(c) for c in range(D.G)])
    ax.set_xlim(-0.6, D.G - 0.4)
    ax.set_ylabel("slope of the case's own loss")
    formula = {"binary": "(p - y) da/dw", "multiclass": "Σk (pk - yk) dak/dw", "regression": "2(ŷ - y) dŷ/dw"}[D.task]
    if D.task == "regression" and S.model.n_outputs > 1:
        formula = "Σk 2(ŷk - yk)/K dŷk/dw"
    title = f"each case's slope along {name}, {formula};\nbars: group means; dashed: the average"
    ax.set_title(title, fontsize=T.panel_title_size)
    ax.legend(fontsize=T.legend_size, loc="best")


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
        ax.plot(u, R, color=T.ink, lw=1.8, label="R")
        ax.plot(
            u, F.R + F.g[a] * (u - w[a]), "--", color=T.prediction_colour, lw=1.2, label=f"tangent, slope {F.g[a]:+.3f}"
        )
        ax.plot([w[a]], [F.R], "o", color=T.ink, zorder=5)
        if w0[a] != w[a]:
            ax.axvline(w0[a], color=T.start_colour, ls=":", lw=1, label="start")
        span = float(R.max() - R.min()) or 1.0
        ax.set_ylim(float(R.min()) - 0.1 * span, float(R.max()) + 0.1 * span)
        ax.set_xlabel(names[0])
        ax.set_ylabel("R")
        ax.legend(fontsize=T.legend_size, loc="best")
        ax.set_title(f"R over {names[0]}" + (f",\n{note}" if note else ""), fontsize=T.panel_title_size)
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
        ax.contourf(A, B, R, levels=levels, cmap=T.landscape_colormap(), alpha=0.35)
        ax.contour(A, B, R, levels=levels, colors=T.landscape_line, linewidths=0.6)
    if look:
        for key, colour, style, label in (
            ("gd", T.gd_colour, "o-", "next steps, gradient descent"),
            ("momentum", T.momentum_colour, "o--", f"next steps, momentum {S.beta:g}"),  # dashed: not by colour alone
        ):
            p = look[key]["path"][:, idx]
            ax.plot(p[:, 0], p[:, 1], style, color=colour, ms=2.5, lw=1.2, label=label, zorder=3)
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], "o-", color=T.path_colour, ms=3, lw=1.2, label="steps taken", zorder=4)
    ax.plot([w0[a]], [w0[b]], "o", mfc=T.paper, mec=T.marker_edge, ms=7, label="start", zorder=5)
    ax.plot([w[a]], [w[b]], "o", color=T.ink, ms=6, zorder=6)
    g = np.array([F.g[a], F.g[b]])
    norm = float(np.hypot(*g))
    if norm > 0:
        length = 0.25 * (xa[1] - xa[0])
        d = -g / norm * length
        ax.annotate(
            "",
            xy=(w[a] + d[0], w[b] + d[1]),
            xytext=(w[a], w[b]),
            arrowprops=dict(arrowstyle="-|>", color=T.ink, lw=1.8),
            zorder=7,
        )
    ax.set_xlim(*xa)
    ax.set_ylim(*xb)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.legend(fontsize=T.legend_size, loc="best")
    title = f"R over {names[0]} and {names[1]},\nwith the direction of -grad R"
    ax.set_title(title + (f"\n{note}" if note else ""), fontsize=T.panel_title_size)


# ---------------------------------------------------------------- R along the next step


def draw_line(ax, S, F):
    """R along the direction of the next step, w + t d (d: the step per unit of eta; -g, the slopes of the weights
    that move, for plain gradient descent), for t from 0 to 2.5 eta: the actual R, the tangent line R + t (g . d)
    (the first-order prediction), where the step lands at t = eta on both, and the lowest point on this line."""
    D, M, T = S.data, S.model, S.theme
    g = moving_slopes(F)
    d = np.asarray(F.direction, dtype=float)
    slope = float((g * d).sum())  # the slope of R along the step, g . d
    if not np.any(d):
        message(ax, T, "every slope is 0:\nno step")
        return
    eta = F.eta
    ts = np.linspace(0, 2.5 * eta, 101)
    w = np.asarray(F.w, dtype=float)
    R = M.risk_batch(D.X, D.y, w[None, :] + ts[:, None] * d[None, :])
    R_eta = float(M.risk_batch(D.X, D.y, (w + eta * d)[None, :])[0])
    pred = F.R + eta * slope
    tangent = F.R + ts * slope
    ax.plot(ts, R, color=T.ink, lw=1.8, label="R along the step")
    ax.plot(ts, tangent, "--", color=T.prediction_colour, lw=1.2, label="first-order prediction")
    ax.axvline(eta, color=T.reference_line, lw=0.8, zorder=0)
    ax.plot([eta], [R_eta], "o", color=T.ink, ms=6, zorder=5, label=f"step of eta: R {R_eta:.4f}")
    ax.plot([eta], [pred], "o", mfc=T.paper, mec=T.prediction_colour, ms=7, zorder=5, label=f"predicted: {pred:.4f}")
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
    plain = S.optimizer.kind == "gd" and S.optimizer.weight_decay == 0
    ax.set_xlabel("t, the step size along -grad R" if plain else f"t, the step size along the {S.optimizer.kind} step")
    ax.set_ylabel("R")
    ax.legend(fontsize=T.legend_size, loc="best")
    ax.set_title(f"R along the next step, eta {eta:g}", fontsize=T.panel_title_size)


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


def output_layer(S, w):
    """The output layer as a matrix (one row per class, one column per unit of the last hidden layer) and its biases."""
    k = S.model.layer_indices(len(S.model.layers) - 1)
    V = np.asarray([w[i] for i in k], dtype=float).reshape(S.model.n_outputs, -1)
    return V[:, :-1], V[:, -1]


def draw_hidden_targets(ax, S, F, H, small, label):
    """draw_hidden for K targets (regression). One unit: the prediction of each target against h, one line style
    each, with the targets of the cases. More units: the cases at their hidden values (two units as they are, three
    in 3-D, more along the two directions in which h varies most)."""
    D, T = S.data, S.theme
    m = H.shape[1]
    V, b = output_layer(S, F.w)
    Y = D.y.numpy()
    dots = dict(s=small, color=T.point_neutral, edgecolor=T.marker_edge)
    what = "ŷ of each target against h" if m == 1 else "the points of the cases"
    if m == 1:
        lo, hi = window(H[:, 0], minimum=0.6)
        u = np.linspace(lo, hi, 101)
        for k in range(len(b)):
            ax.plot(u, V[k, 0] * u + b[k], LINE_STYLES[k], color=T.neutral_colour, lw=1.5)
            ax.scatter(H[:, 0], Y[:, k], s=max(small, 30), zorder=3, **target_marks(T, k))
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel("ŷ and y")
        ax.legend(handles=[target_handle(T, k) for k in range(len(b))], fontsize=T.legend_size, loc="best")
    elif m == 3 and hasattr(ax, "set_zlim"):
        ax.scatter(H[:, 0], H[:, 1], H[:, 2], linewidths=0.3, depthshade=False, **dots)
        for q in range(3):
            getattr(ax, f"set_{'xyz'[q]}lim")(*window(H[:, q], minimum=0.6))
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel(f"{label}2")
        ax.set_zlabel(f"{label}3")
        ax.view_init(elev=20, azim=-60)
    else:
        if m == 2:
            two = H
            ax.set_xlabel(f"{label}1")
            ax.set_ylabel(f"{label}2")
        else:
            centred = H - H.mean(0)
            two = centred @ np.linalg.svd(centred, full_matrices=False)[2][:2].T
            ax.set_xlabel("direction of h with the largest spread")
            ax.set_ylabel("the next one")
        ax.scatter(two[:, 0], two[:, 1], lw=0.3, zorder=3, **dots)
    ax.set_title(f"the cases at their hidden values,\n{what}", fontsize=T.panel_title_size)


def draw_hidden_classes(ax, S, F, H, colours, small, label):
    """draw_hidden for K classes. One unit: p of each class against h, each case on the curve of its class. Two
    units: the most likely class over the (h1, h2) plane, with the boundaries. Three: the cases in 3-D; more: along
    the two directions in which h varies most; both coloured by class, without boundaries."""
    D, T = S.data, S.theme
    m = H.shape[1]
    V, b = output_layer(S, F.w)
    if m == 1:
        lo, hi = window(H[:, 0], minimum=0.6)
        u = np.linspace(lo, hi, 101)
        P = softmax(u[:, None] * V[None, :, 0] + b)
        for k in range(D.n_classes):
            ax.plot(u, P[:, k], color=T.class_colour[k], lw=1.5, label=f"p{k}")
        own = F.out.p[np.arange(D.n), D.y.numpy().astype(np.int64)]
        ax.scatter(H[:, 0], own, c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel("p of each class; each case on its own")
        ax.legend(fontsize=T.legend_size, loc="best")
        what = "p of each class"
    elif m == 2:
        x0, x1 = window(H[:, 0], minimum=0.6), window(H[:, 1], minimum=0.6)
        U1, U2 = np.meshgrid(np.linspace(*x0, 81), np.linspace(*x1, 81))
        A = U1[..., None] * V[:, 0] + U2[..., None] * V[:, 1] + b
        ax.imshow(class_shading(T, A), extent=(*x0, *x1), origin="lower", aspect="auto", interpolation="bilinear")
        class_boundaries(ax, U1, U2, A, colors=T.boundary)
        ax.scatter(H[:, 0], H[:, 1], c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
        gh = [D.stats(H, c)[0] for c in range(D.G)]
        for c in range(D.G):
            ax.text(
                gh[c][0],
                gh[c][1],
                f"  {letter(c)}",
                fontsize=T.letter_size,
                zorder=4,
                path_effects=halo(T),
                color=T.ink,
            )
        ax.set_xlim(*x0)
        ax.set_ylim(*x1)
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel(f"{label}2")
        what = "the most likely class"
    elif m == 3 and hasattr(ax, "set_zlim"):
        ax.scatter(
            H[:, 0], H[:, 1], H[:, 2], c=colours, s=small, edgecolor=T.marker_edge, linewidths=0.3, depthshade=False
        )
        for q in range(3):
            getattr(ax, f"set_{'xyz'[q]}lim")(*window(H[:, q], minimum=0.6))
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel(f"{label}2")
        ax.set_zlabel(f"{label}3")
        ax.view_init(elev=20, azim=-60)
        what = "coloured by class"
    else:
        centred = H - H.mean(0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        two = centred @ vt[:2].T
        ax.scatter(two[:, 0], two[:, 1], c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
        ax.set_xlabel("direction of h with the largest spread")
        ax.set_ylabel("the next one")
        what = "coloured by class"
    ax.set_title(f"the cases at their hidden values,\n{what}", fontsize=T.panel_title_size)


def draw_hidden(ax, S, F):
    """The cases at the values h of the last hidden layer. The output unit is a linear function of h: its boundary
    a = 0 is a point (one unit), a line (two) or a plane (three). With more units: the score a against the largest
    other direction of h."""
    D, T = S.data, S.theme
    m = hidden_width(S)
    if m == 0:
        message(ax, T, "no hidden layer:\nthe output unit sees x itself")
        return
    H = np.asarray(F.out.hidden[-1][1], dtype=float)
    colours = point_colours(S)
    small = 10 if D.n > 40 else 40
    label = "h" if len(S.model.hidden) == 1 else f"h{len(S.model.hidden)},"
    if D.task == "multiclass":
        draw_hidden_classes(ax, S, F, H, colours, small, label)
        return
    if S.model.n_outputs > 1:
        draw_hidden_targets(ax, S, F, H, small, label)
        return
    v, b = output_weights(S, F.w)
    binary = D.task == "binary"
    gh = [D.stats(H, c)[0] for c in range(D.G)]
    if m == 1:
        h = H[:, 0]
        lo, hi = window(h, minimum=0.6)
        u = np.linspace(lo, hi, 101)
        a = v[0] * u + b
        if binary:
            ax.plot(u, 1 / (1 + np.exp(-a)), color=T.neutral_colour, lw=1.5, label="p = sigmoid(a)")
            ax.scatter(h, D.y.numpy(), c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
            if v[0] != 0 and lo < -b / v[0] < hi:
                ax.axvline(-b / v[0], color=T.boundary, lw=1, label="boundary a = 0")
            ax.set_ylabel("p, and the class of each case")
        else:
            ax.plot(u, a, color=T.neutral_colour, lw=1.5, label="prediction")
            ax.scatter(h, D.y.numpy(), c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
            ax.set_ylabel("y")
        ax.set_xlabel(f"{label}1")
        ax.legend(fontsize=T.legend_size, loc="best")
    elif m == 2:
        x0, x1 = window(H[:, 0], minimum=0.6), window(H[:, 1], minimum=0.6)
        U1, U2 = np.meshgrid(np.linspace(*x0, 81), np.linspace(*x1, 81))
        a = v[0] * U1 + v[1] * U2 + b
        if binary:
            P = 1 / (1 + np.exp(-a))
            ax.contourf(U1, U2, P, levels=np.linspace(0, 1, 11), cmap=T.plane_colormap(), alpha=T.plane_alpha)
            if P.min() < 0.5 < P.max():
                ax.contour(U1, U2, P, levels=[0.5], colors=T.boundary)
        else:
            ax.contour(U1, U2, a, levels=10, cmap=T.value_cmap, linewidths=0.8)
        ax.scatter(H[:, 0], H[:, 1], c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
        for c in range(D.G):
            ax.text(
                gh[c][0],
                gh[c][1],
                f"  {letter(c)}",
                fontsize=T.letter_size,
                zorder=4,
                path_effects=halo(T),
                color=T.ink,
            )
        ax.set_xlim(*x0)
        ax.set_ylim(*x1)
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel(f"{label}2")
    elif m == 3 and hasattr(ax, "set_zlim"):
        ax.scatter(
            H[:, 0], H[:, 1], H[:, 2], c=colours, s=small, edgecolor=T.marker_edge, linewidths=0.3, depthshade=False
        )
        r = [window(H[:, i], minimum=0.6) for i in range(3)]
        if binary and np.any(v != 0):
            i = int(np.argmax(np.abs(v)))  # solve the plane for the coordinate with the largest weight
            j, k = [q for q in range(3) if q != i]
            Gj, Gk = np.meshgrid(np.linspace(*r[j], 6), np.linspace(*r[k], 6))
            Gi = np.clip(-(v[j] * Gj + v[k] * Gk + b) / v[i], *r[i])
            coords = [None, None, None]
            coords[i], coords[j], coords[k] = Gi, Gj, Gk
            ax.plot_surface(*coords, color=T.plane_surface, alpha=0.25, linewidth=0)
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
        ax.scatter(a, other, c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
        if binary:
            ax.axvline(0.0, color=T.boundary, lw=1)
        ax.set_xlabel("a = output unit's score" if binary else "prediction")
        ax.set_ylabel("largest other direction of h")
    what = "the boundary a = 0" if binary else "coloured by y"
    ax.set_title(f"the cases at their hidden values,\n{what}", fontsize=T.panel_title_size)


# ---------------------------------------------------------------- gradient descent and momentum


def draw_momentum(ax, S, F):
    """R over the next steps from the current weights: gradient descent and heavy-ball momentum, where each step
    keeps the share beta of the last one."""
    T = S.theme
    look = F.lookahead
    gd, mo = look["gd"]["R"], look["momentum"]["R"]
    steps = range(len(gd))
    ax.plot(steps, gd, color=T.gd_colour, lw=1.8, label=f"gradient descent: R {gd[-1]:.4f}")
    ax.plot(steps, mo, "--", color=T.momentum_colour, lw=1.8, label=f"momentum, beta {S.beta:g}: R {mo[-1]:.4f}")
    if S.log_scale and min(min(gd), min(mo)) > 0:
        ax.set_yscale("log")
    ax.set_xlabel("steps from the current weights")
    ax.set_ylabel("R")
    ax.legend(fontsize=T.legend_size, loc="best")
    ax.set_title(f"the next {len(gd) - 1} steps, eta {F.eta:g}", fontsize=T.panel_title_size)


VIEW_DRAWERS = {
    "cases": draw_cases,
    "slice": draw_slice,
    "line": draw_line,
    "hidden": draw_hidden,
    "momentum": draw_momentum,
}

# ------------------------ backprop.py ------------------------

Stage = namedtuple("Stage", "kind layer text")  # kind: forward, output, weights, to_h, delta, average
Backward = namedtuple("Backward", "inputs z out delta dh deriv grads")  # per case, as numpy arrays


def layer_title(model, l):
    if l == len(model.layers) - 1:
        return "the output weights"
    if len(model.hidden) == 1:
        return "the first-layer weights"
    return f"the weights into layer {l + 1}"


def act_name(model, l):
    act = model.layers[l].act
    return act[0] if act else "no activation"


def backprop_stages(model):
    """The stages of the backward pass for this network, from the forward pass to the average over the cases."""
    L = len(model.layers)
    regression = model.task == "regression"
    output = {
        "binary": "at the output: δa = ∂ℓ/∂a = p - y",
        "multiclass": "at the output: δak = ∂ℓ/∂ak = pk - yk (yk is 1 for the case's class, else 0)",
        "regression": "at the output: δa = ∂ℓ/∂a = 2(ŷ - y)",
    }[model.task]
    if regression and model.n_outputs > 1:
        output = "at the output: δak = ∂ℓ/∂ak = 2(ŷk - yk)/K (ℓ is the mean over the K targets)"
    out = [
        Stage("forward", None, "forward pass: z, h and " + ("ŷ" if regression else "a and p") + " of every case"),
        Stage("output", L - 1, output),
        Stage("weights", L - 1, f"slopes of {layer_title(model, L - 1)}: δ of the unit × the input of the line"),
    ]
    for l in range(L - 2, -1, -1):
        h = "h" if len(model.hidden) == 1 else f"h{l + 1},"
        act = act_name(model, l)
        out += [
            Stage("to_h", l, f"back to {h}: ∂ℓ/∂h = sum over the next units of weight × δ"),
            Stage("delta", l, f"through {act}: δ = {act}′(z) × ∂ℓ/∂h"),
            Stage("weights", l, f"slopes of {layer_title(model, l)}: δ of the unit × the input of the line"),
        ]
    out.append(Stage("average", None, "R is the average loss: dR/dw = the average of the cases' slopes (the table)"))
    return out


def derivative(layer, z, out):
    """The derivative of the layer's activation at z (out: its value there), elementwise."""
    m = layer.module
    if m is None or isinstance(m, nn.Identity):
        return torch.ones_like(z)
    if isinstance(m, nn.Tanh):
        return 1 - out**2
    if isinstance(m, nn.Sigmoid):
        return out * (1 - out)
    if isinstance(m, nn.ReLU):
        return (z > 0).to(z.dtype)
    raise ValueError(f"slopewalk: no derivative for {m}")


def backward_pass(model, X, y, w=None):
    """The forward and backward values of every case, with the formulas of the stages (no autograd).

    inputs[l]: the inputs of layer l (x, then the h of the layer before); z[l] and out[l]: its pre-activations and
    values; delta[l] = dl/dz[l] (the last layer: dl/da); dh[l] = dl/dout[l] for the hidden layers; deriv[l]: the
    activation's derivative at z[l]; grads: each case's slope along every weight, n x n_weights, in the order of the
    names. The network is left at w.
    """
    if w is not None:
        model.set_weights(w)
    layers = model.layers
    n = X.shape[0]
    with torch.no_grad():
        inputs, zs, outs = [], [], []
        h = X
        for layer in layers:
            inputs.append(h)
            z = layer.linear(h)
            h = layer.module(z) if layer.module is not None else z
            zs.append(z)
            outs.append(h)
        a = zs[-1]
        if model.task == "multiclass":  # p - y for every class, with y one-hot
            d_out = torch.softmax(a, dim=1) - nn.functional.one_hot(y.long(), a.shape[1]).to(a.dtype)
        elif model.task == "binary":
            d_out = torch.sigmoid(a) - y[:, None]
        else:  # the mean squared error over the K targets of a case
            d_out = 2 * (a - y.reshape(a.shape[0], -1)) / a.shape[1]
        deltas = [None] * len(layers)
        dh = [None] * len(layers)
        deriv = [None] * len(layers)
        deltas[-1] = d_out
        for l in range(len(layers) - 2, -1, -1):
            dh[l] = deltas[l + 1] @ layers[l + 1].linear.weight
            deriv[l] = derivative(layers[l], zs[l], outs[l])
            deltas[l] = deriv[l] * dh[l]
        blocks = []
        for h_in, d in zip(inputs, deltas):
            with_one = torch.cat([h_in, torch.ones(n, 1, dtype=h_in.dtype)], dim=1)
            blocks.append((d[:, :, None] * with_one[:, None, :]).reshape(n, -1))
        grads = torch.cat(blocks, dim=1)

    def arr(ts):
        return [None if t is None else t.detach().numpy().astype(float) for t in ts]

    return Backward(arr(inputs), arr(zs), arr(outs), arr(deltas), arr(dh), arr(deriv), grads.numpy().astype(float))


def traced_case(S, c):
    """The case whose backward pass the derivation shows for group c: the point nearest the group's mean."""
    D = S.data
    idx = np.nonzero(D.gsel[c])[0]
    X = D.X.numpy()[idx]
    return int(idx[np.argmin(((X - X.mean(0)) ** 2).sum(1))])


# ---------------------------------------------------------------- the network panels


def stage_index(S):
    return max(0, min(S.backprop_step, len(S.backprop_stages) - 1))


def line_styles(S, F):
    """Colour, opacity and width of each weight's line at the current stage, or None for the usual drawing. The
    lines the stage does not use are thin and faint, in the weight's own colour."""
    st = F.stage
    if st is None or st.kind == "forward":
        return None
    M, T = S.model, S.theme
    styles = {k: (None, T.faded_line_alpha, T.line_width) for k in range(M.n_weights)}
    if st.kind == "weights":
        for k in M.layer_indices(st.layer):
            styles[k] = (T.backprop_colour, 1.0, None)
    elif st.kind == "average":
        for k in styles:
            styles[k] = (T.backprop_colour, 1.0, None)
    elif st.kind == "to_h":
        for k in M.layer_indices(st.layer + 1):
            if M.where[k][2] >= 0:  # not the bias
                styles[k] = (None, 1.0, None)
    return styles


def focus(S, F):
    """The boxes the current stage works with, as the tags of draw_case and overlay; the others are faded. None at
    the forward pass and outside the backward pass.

    output: a, p and y give δa. Slopes of layer l: δ of its units times their inputs (x, or the h before). Back to h:
    the δ of the next units and the weights to them give ∂ℓ/∂h. Through the activation: z, its curve and ∂ℓ/∂h give δ.
    Average: the slopes on the lines, no box."""
    st = F.stage
    if st is None or st.kind == "forward":
        return None
    last = len(S.model.layers) - 1

    def units(l):  # where the δ of layer l is drawn: under its z boxes, or under a for the output
        return {("a",), ("δa",)} if l == last else {("z", l), ("δ", l)}

    if st.kind == "output":
        return {("a",), ("out_act",), ("p",), ("loss",), ("δa",)}
    if st.kind == "weights":
        l = st.layer
        return units(l) | ({("x",)} if l == 0 else {("h", l - 1)}) | {("bias", l)}
    if st.kind == "to_h":
        l = st.layer
        return units(l + 1) | {("h", l), ("dh", l)}
    if st.kind == "delta":
        l = st.layer
        return {("z", l), ("act", l), ("h", l), ("dh", l), ("δ", l)}
    return {("bias", l) for l in range(last + 1)}  # average: the slopes on every line, and the bias nodes they start at


def shared_labels(F):
    """True when the labels on the lines are the same in every network panel (the weights themselves), so that the
    first panel alone carries them. The slopes of the backward pass differ from case to case and go in every panel."""
    return F.stage is None or F.stage.kind in ("forward", "to_h")


def label_spec(S, F, c):
    """The labels on the lines at the current stage: {k: (long text, short text, colour)}, or None for the usual
    labels. Weights stages: each line's slope (mean over the group); back to h: the weights used; else none."""
    st = F.stage
    if st is None or st.kind == "forward":
        return None
    M, T, D = S.model, S.theme, S.data
    g = F.bp.grads[D.gsel[c]].mean(0)
    spec = {}
    if st.kind in ("weights", "average"):
        ks = M.layer_indices(st.layer) if st.kind == "weights" else range(M.n_weights)
        for k in ks:
            long = f"∂ℓ/∂{M.names[k]} = {g[k]:+.3f}" if c == S.shown[0] else f"{g[k]:+.3f}"  # names once
            spec[k] = (long, f"{g[k]:+.3f}", T.backprop_colour, f"{g[k]:+.2f}")
    elif st.kind == "to_h":
        for k in M.layer_indices(st.layer + 1):
            if M.where[k][2] >= 0:
                colour = T.positive if F.w[k] >= 0 else T.negative_text
                spec[k] = (f"{M.names[k]} = {F.w[k]:.2f}", f"{F.w[k]:.2f}", colour)
    return spec


def box(ax, x, y, text, T):
    """A backward value under the forward box at (x, y), far enough down to clear it: both boxes have the same number
    of lines (two for single points, three with a standard deviation)."""
    lines = text.count("\n") + 1
    drop = lines * 1.2 * (T.value_size + T.backprop_size) / 2 + 0.25 * T.value_size + 0.2 * T.backprop_size + 3
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, -max(31.0, drop)),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=T.backprop_size,
        color=T.backprop_colour,
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", fc=T.backprop_tint, ec=T.backprop_colour, lw=0.8),
    )


def tangent(ax, xc, yc, act, m, T, height):
    """The tangent of the activation at m on its glyph: its slope is the derivative used at this stage."""
    name, f, lim, df = act
    w_, h_ = 0.62, height
    u = np.linspace(-lim, lim, 100)
    vals = f(u)
    lo, span = vals.min(), vals.max() - vals.min()
    m = float(np.clip(m, -lim, lim))
    ts = np.linspace(max(-lim, m - 0.35 * lim), min(lim, m + 0.35 * lim), 2)
    ys = f(m) + df(m) * (ts - m)

    def gx(t):
        return xc - w_ / 2 + (t + lim) / (2 * lim) * w_

    def gy(v):
        return yc + 0.85 * h_ * ((v - lo) / span - 0.5)

    ax.plot(gx(ts), gy(ys), color=T.backprop_colour, lw=1.6, zorder=5)


def overlay(ax, S, F, c, edges):
    """The backward values on network panel c, for every stage reached so far: delta at the output, dl/dh under the
    h boxes, delta under the z boxes; at the "through the activation" stage, the tangents on the glyphs; at the
    "back to h" stage, arrows back along the lines used."""
    st = F.stage
    if st is None or st.kind == "forward":
        return
    M, D, L, T = S.model, S.data, S.layout, S.theme
    bp = F.bp
    sel = D.gsel[c]
    reached = S.backprop_stages[1 : stage_index(S) + 1]
    tag = (lambda name: f"{name}\n") if c == S.shown[0] else (lambda name: "")  # the names once, first panel
    for s in reached:
        if s.kind == "output":  # under each score: one box, or one per class
            with tagging(ax, ("δa",)):
                for k, ty in enumerate(L.unit_ys(M.n_outputs)):
                    m, sd = D.stats(bp.delta[-1][:, k], c)
                    box(ax, L.x_a, ty, tag("δ" + output_name(M, k)) + num(m, sd), T)
        elif s.kind in ("to_h", "delta"):
            l = s.layer
            XZ, XG, XH = L.blocks[l]
            tys = L.unit_ys(M.layers[l].out_features)
            values = bp.dh[l] if s.kind == "to_h" else bp.delta[l]
            with tagging(ax, ("dh", l) if s.kind == "to_h" else ("δ", l)):
                for j, ty in enumerate(tys):
                    m, sd = D.stats(values[:, j], c)
                    nm = M.unit_name(l, j)
                    if s.kind == "to_h":
                        box(ax, XH, ty, tag(f"∂ℓ/∂h{nm}") + num(m, sd), T)
                    else:
                        box(ax, XZ, ty, tag(f"δ{nm}") + num(m, sd), T)
    if st.kind == "delta" and M.layers[st.layer].act:
        l = st.layer
        XZ, XG, XH = L.blocks[l]
        tys = L.unit_ys(M.layers[l].out_features)
        height = min(0.5, 0.75 * (tys[0] - tys[1])) if len(tys) > 1 else 0.5
        z = bp.z[l][sel].mean(0)
        for j, ty in enumerate(tys):
            tangent(ax, XG, ty, M.layers[l].act, z[j], T, height)
    if st.kind == "to_h":
        arrow = dict(arrowstyle="-|>", color=T.backprop_colour, lw=1.4, shrinkA=0, shrinkB=0)
        for k, p0, p1, _ in edges:
            if M.where[k][0] == st.layer + 1 and M.where[k][2] >= 0:
                a = (p0[0] + 0.62 * (p1[0] - p0[0]), p0[1] + 0.62 * (p1[1] - p0[1]))
                b = (p0[0] + 0.38 * (p1[0] - p0[0]), p0[1] + 0.38 * (p1[1] - p0[1]))
                ax.annotate("", xy=b, xytext=a, arrowprops=arrow, zorder=4)


# ---------------------------------------------------------------- the derivation panel


def factor(v, digits=3):
    """A number as the second factor of a product: in parentheses when it is negative."""
    text = f"{v:+.{digits}f}"
    return f"({text})" if v < 0 else text


def derivation_lines(S, F, i):
    """The steps of the backward pass for case i, as (stage index, text) pairs."""
    M = S.model
    bp = F.bp
    yi = S.data.y[i].numpy()
    y = float(yi) if yi.ndim == 0 else None  # None: several targets
    a = bp.z[-1][i, 0]
    regression = S.task == "regression"
    out = []
    for s_idx, st in enumerate(S.backprop_stages):
        if S.task == "multiclass" and st.kind in ("forward", "output"):
            out += [(s_idx, text) for text in class_lines(bp.z[-1][i], int(y), st.kind)]
        elif y is None and st.kind in ("forward", "output"):
            out += [(s_idx, text) for text in target_lines(bp.z[-1][i], yi, st.kind)]
        elif st.kind == "forward":
            if regression:
                out.append((s_idx, f"forward: ŷ = {a:+.3f}, y = {y:+.3f}"))
            else:
                p = 1 / (1 + np.exp(-a))
                out.append((s_idx, f"forward: a = {a:+.3f}, p = {p:.3f}, y = {y:g}"))
        elif st.kind == "output":
            d = bp.delta[-1][i, 0]
            if regression:
                out.append((s_idx, f"δa = 2(ŷ - y) = 2({a:+.3f} - {y:+.3f}) = {d:+.3f}"))
            else:
                p = 1 / (1 + np.exp(-a))
                out.append((s_idx, f"δa = p - y = {p:.3f} - {y:g} = {d:+.3f}"))
        elif st.kind == "weights":
            l = st.layer
            labels = M.input_labels(l)
            for k in M.layer_indices(l):
                _, j, src = M.where[k]
                unit = output_name(M, j) if l == len(M.layers) - 1 else M.unit_name(l, j)
                d = bp.delta[l][i, j]
                if src < 0:
                    out.append((s_idx, f"{M.names[k]}: δ{unit} × 1 = {bp.grads[i, k]:+.3f}"))
                else:
                    v = bp.inputs[l][i, src]
                    out.append(
                        (
                            s_idx,
                            f"{M.names[k]}: δ{unit} × {labels[src]} = {d:+.3f} × {factor(v)} = {bp.grads[i, k]:+.3f}",
                        )
                    )
        elif st.kind == "to_h":
            l = st.layer
            nxt = M.layers[l + 1]
            Wn = nxt.linear.weight.detach().numpy()
            for j in range(M.layers[l].out_features):
                nm = M.unit_name(l, j)
                if nxt.out_features == 1:
                    k = M.layer_indices(l + 1)[j]
                    unit = output_name(M, 0) if l + 1 == len(M.layers) - 1 else M.unit_name(l + 1, 0)
                    d = bp.delta[l + 1][i, 0]
                    product = f"{Wn[0, j]:+.3f} × {factor(d)} = {bp.dh[l][i, j]:+.3f}"
                    out.append((s_idx, f"∂ℓ/∂h{nm} = {M.names[k]} × δ{unit} = {product}"))
                else:  # a sum over the next units, two terms to a line so that the lines fit their column
                    terms = [f"{Wn[q, j]:+.2f} × {factor(bp.delta[l + 1][i, q])}" for q in range(nxt.out_features)]
                    parts = [" + ".join(terms[q : q + 2]) for q in range(0, len(terms), 2)]
                    parts[0] = f"∂ℓ/∂h{nm} = {parts[0]}"
                    parts[1:] = [f"      + {part}" for part in parts[1:]]
                    parts[-1] += f" = {bp.dh[l][i, j]:+.3f}"
                    out += [(s_idx, part) for part in parts]
        elif st.kind == "delta":
            l = st.layer
            act = act_name(S.model, l)
            for j in range(M.layers[l].out_features):
                nm = M.unit_name(l, j)
                out.append(
                    (
                        s_idx,
                        f"δ{nm} = {act}′(z{nm}) × ∂ℓ/∂h{nm} = {bp.deriv[l][i, j]:.3f} × {factor(bp.dh[l][i, j])} "
                        f"= {bp.delta[l][i, j]:+.3f}",
                    )
                )
        elif st.kind == "average":
            k = M.layer_indices(len(M.layers) - 1)[0]
            mean = bp.grads[:, k].mean()
            out.append((s_idx, f"dR/d{M.names[k]} = mean over the {S.data.n} cases = {mean:+.4f}"))
    return out


def output_name(model, j):
    """The name of output unit j: a with one output; a0, a1, ... with one per class (as the labels); a1, a2, ... with
    one per target (as the inputs x1, x2)."""
    if model.n_outputs == 1:
        return "a"
    return f"a{j}" if model.task == "multiclass" else f"a{j + 1}"


def target_lines(a, y, kind):
    """The lines of the forward and output stages for K targets: the predictions a and the targets y of one case."""
    K = len(a)
    if kind == "forward":
        predictions = ", ".join(f"{v:+.3f}" for v in a)
        targets = ", ".join(f"{v:+.3f}" for v in y)
        return [f"forward: ŷ = ({predictions})", f"targets: y = ({targets})"]
    return [
        f"δa{k + 1} = 2(ŷ{k + 1} - y{k + 1})/{K} = 2({a[k]:+.3f} - {factor(y[k])})/{K} = {2 * (a[k] - y[k]) / K:+.3f}"
        for k in range(K)
    ]


def class_lines(a, y, kind):
    """The lines of the forward and output stages for K classes: the scores a of one case and its class y."""
    p = np.exp(a - a.max())
    p /= p.sum()
    if kind == "forward":
        scores = ", ".join(f"{v:+.3f}" for v in a)
        probabilities = ", ".join(f"{v:.3f}" for v in p)
        return [f"forward: a = ({scores}), y = {y}", f"p = softmax(a) = ({probabilities})"]
    return [
        f"δa{k} = p{k} - {int(k == y)} = {p[k]:.3f} - {int(k == y)} = {p[k] - (k == y):+.3f}" for k in range(len(a))
    ]


def fitting_size(ax, T, texts, columns, floor=9.5):
    """The largest font from T.note_size down to floor (in half points) at which the longest text, in bold, fits its
    column of the panel. Measured in inches from the panel's share of the figure, so the same in every frame."""
    renderer = get_renderer(ax.figure)
    if renderer is None or not texts:
        return T.note_size
    column = (ax.figure.get_figwidth() * width_share(ax) - PANEL_GAP) / columns * 0.96
    size = T.note_size
    while size > floor:
        props = FontProperties(size=size, weight="bold")
        longest = max(renderer.get_text_width_height_descent(t, props, ismath=False)[0] for t in texts)
        if longest / ax.figure.dpi <= column:
            break
        size -= 0.5
    return size


def draw_derivation(ax, S, F):
    """The Worked example: the backward pass of one case, step by step; the stages reached so far, the current one in
    bold. Its heading, "Worked example" in the colour of the backward pass, sits above the title of the stage."""
    T = S.theme
    ax.axis("off")
    c = max(0, min(S.backprop_case, S.data.G - 1))
    i = traced_case(S, c)
    step = stage_index(S)
    stage = S.backprop_stages[step]
    lines = [(s, t) for s, t in derivation_lines(S, F, i) if s <= step]
    per_column, columns = 12, 2  # the steps run down the first column, then the second
    most = per_column * columns
    shown = lines[-most:]
    if S.data.gsel[c].sum() == 1:
        case = f": case {letter(c)}"
    else:
        case = f": the case nearest the mean of group {letter(c)}"
    title = ax.set_title(
        f"stage {step} of {len(S.backprop_stages) - 1}: {stage.text}", fontsize=T.panel_title_size, loc="left"
    )
    # the heading above the title: "Worked example" in the colour of the backward pass, then the case in ink
    head = ax.annotate(
        "Worked example",
        xy=(0, 1),
        xycoords=title,
        xytext=(0, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=T.panel_title_size,
        fontweight="bold",
        color=T.backprop_colour,
    )
    ax.annotate(case, xy=(1, 0), xycoords=head, ha="left", va="bottom", fontsize=T.panel_title_size, color=T.ink)
    dy = 0.96 / per_column
    if len(lines) > most:
        shown = [(None, "... (earlier steps above)")] + shown[1:]
    fontsize = fitting_size(ax, T, [t for _, t in derivation_lines(S, F, i)], columns)
    for n, (s, text) in enumerate(shown):
        current = s == step
        col, row = divmod(n, per_column)
        ax.text(
            0.5 * col,
            0.98 - row * dy,
            text,
            transform=ax.transAxes,
            fontsize=fontsize,
            va="top",
            color=T.backprop_colour if current else (T.derivation_muted if s is None else T.derivation_text),
            fontweight="bold" if current else "normal",
        )

# ------------------------ picture.py ------------------------

Frame = namedtuple(
    "Frame",
    "w out R g deltas changed moving history predictions eta next_change cases lookahead bp stage by_hand "
    "direction history_test R_test",
)
LIVE_DPI = 100  # resolution of the picture shown by the widgets


def make_frame(S, w=None):
    """The numbers every panel needs at the weights w (default: the current ones). Leaves the network at w."""
    M, D = S.model, S.data
    w = S.weights() if w is None else [float(v) for v in w]
    out = M.forward(D.X, D.y, w)
    R = M.risk(D.X, D.y, w)
    g = M.slopes(D.X, D.y, w)
    W = np.asarray(w, dtype=float)[None, :] + 0.1 * np.eye(len(w))  # each weight alone grown by 0.1: one batched pass
    deltas = list(M.risk_batch(D.X, D.y, W) - R)
    M.set_weights(w)
    moving = set(S.active)
    changed = {k for k in range(len(w)) if abs(w[k] - S.w0[k]) > 1e-9}  # moved from the start in any way
    ref = S.path[-1]  # the weights after the last step (or at the start): a difference was made by hand
    by_hand = {k for k in range(len(w)) if abs(w[k] - ref[k]) > 1e-6 * max(1.0, abs(ref[k]))}
    dw = S.next_step(S.eta, w, g)  # the step the optimizer would take, without taking it
    next_change = float(sum(g[k] * dw[k] for k in dw))
    direction = np.zeros(len(w))  # the step per unit of eta: -g for plain gradient descent
    for k, d in dw.items():
        direction[k] = d / S.eta
    R_test = S.risk_test(w)
    cases = M.case_slopes(D.X, D.y) if "cases" in S.views else None
    lookahead = S.lookahead(w=w) if "momentum" in S.views else None
    bp = backward_pass(M, D.X, D.y, w) if "backprop" in S.views else None
    stage = S.backprop_stages[stage_index(S)] if "backprop" in S.views else None
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
        bp,
        stage,
        by_hand,
        direction,
        list(S.history_test),
        R_test,
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
        shown = list(S.shown)  # the groups drawn as network panels
        per_row, rows = L.per_row(len(shown)), L.rows(len(shown))
        with_sd = any(sel.sum() > 1 for sel in D.gsel)  # groups of points: the boxes show mean and standard deviation
        backprop = "backprop" in S.views  # taller rows: room for the backward boxes under the units
        lower = L.lower_rows(self.panels, S.model.n_weights)
        n_bottom = len(lower)
        first = L.first_row_height(S.model.n_weights, S.theme.table_size, S.theme.table_leading)
        bottoms = [first] + [BOTTOM_HEIGHT] * (n_bottom - 1)  # the first lower row holds the table
        size = (L.width, L.row_height(with_sd, backprop) * rows + sum(bottoms))
        if pyplot:
            fig = plt.figure(figsize=size, facecolor=S.theme.paper)
        else:
            fig = Figure(figsize=size, dpi=dpi, facecolor=S.theme.paper)
            FigureCanvasAgg(fig)
        heights = [L.row_height(with_sd, backprop)] * rows + bottoms
        gs = fig.add_gridspec(rows + n_bottom, 1, height_ratios=heights)  # every row has its own grid and spacing
        self.case_axes = []
        share = L.panel_share()  # one panel alone: larger than one of two side by side, in the middle of its row
        for r in range(rows):
            if len(shown) == 1 and share < 1.0:
                side = (1 - share) / 2
                cells = gs[r, 0].subgridspec(1, 3, width_ratios=[side, share, side], wspace=0)
                self.case_axes.append(fig.add_subplot(cells[0, 1]))
            elif per_row == 1:
                self.case_axes.append(fig.add_subplot(gs[r, 0]))
            else:
                cells = gs[r, 0].subgridspec(1, per_row, wspace=L.wspace(per_row, NET_GAP))
                n = min(per_row, len(shown) - r * per_row)
                self.case_axes += [fig.add_subplot(cells[0, c]) for c in range(n)]
        self.shown = shown
        self.axes = {}
        for r, row in enumerate(lower):
            widths = [w for _, w in row]
            cells = gs[rows + r, 0].subgridspec(1, len(row), width_ratios=widths, wspace=L.wspace(len(row), PANEL_GAP))
            for c, (name, _) in enumerate(row):
                if name is not None:
                    extra = {"projection": "3d"} if name == "hidden" and needs_3d(S) else {}
                    self.axes[name] = fig.add_subplot(cells[0, c], **extra)
        if S.title:
            fig.suptitle(S.title, fontsize=S.theme.suptitle_size, color=S.theme.ink)
        self.fig = fig
        self.laid_out = False

    def refresh(self, w=None):
        """Draws every panel at the weights w (default: the current ones) and returns the figure."""
        S = self.S
        F = make_frame(S, w)
        if self.laid_out:
            for ax in self.case_axes + list(self.axes.values()):
                ax.clear()
        styles = line_styles(S, F)
        if list(S.shown) != self.shown:
            raise ValueError("slopewalk: the groups shown changed; build the picture again.")
        parts = [draw_case(ax, S, F, c, styles) for c, ax in zip(self.shown, self.case_axes)]
        keep = focus(S, F)  # the backward pass fades what its stage does not use
        for c, ax, (edges, _) in zip(self.shown, self.case_axes, parts):
            overlay(ax, S, F, c, edges)
            fade(ax, keep, S.theme)
        draw_input_plane(self.axes["plane"], S, F)
        draw_history(self.axes["history"], S, F)
        draw_slope_table(self.axes["table"], S, F)
        drawers = dict(VIEW_DRAWERS, backprop=draw_derivation)
        for name in self.panels[3:]:
            drawers[name](self.axes[name], S, F)
            S.model.set_weights(F.w)
        for ax in self.axes.values():
            style_panel(ax, S.theme)
        if not self.laid_out:
            self.layout()
        shared = shared_labels(F)  # the weights are the same in every panel: the first panel carries their labels
        first = "none"
        for i, (c, ax, (edges, obstacles)) in enumerate(zip(self.shown, self.case_axes, parts)):
            if i == 0 or not shared:
                mode = draw_edge_labels(ax, S, F, edges, obstacles, label_spec(S, F, c))
                first = mode if i == 0 else first
            elif first != "none":
                draw_shared_note(ax, S)
        S.model.set_weights(F.w)
        self.frame = F
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

# The look of the control area in one place. Text takes the colour of the root box, which the colours switch sets
# (see control_css); quiet text is the same colour at a lower opacity. The two colours of the last step's change keep
# at least 3.9:1 on white and on dark backgrounds (#1E1E1E, #111111), enough for the large text they are used for.
UI = {
    "quiet": "opacity:var(--slopewalk-quiet,0.72)",  # 0.72 on light paper, 0.85 on dark (control_css)
    "text": "font-size:14px",
    "hint": "font-size:13px",
    "small": "font-size:12px",
    "value": "font-size:15px",
    "big": "font-size:22px;font-weight:600",
    "numbers": "font-variant-numeric:tabular-nums",
    "down": "#2E8B57",
    "up": "#E53935",
    "short": "250px",  # weight sliders and drop-down lists
    "long": "320px",  # eta, stage and the settings of the views
    "label": "40px",  # the label column of the weight sliders, eta and stage
    "setting_label": "100px",  # the label column of the other controls
    "column": "270px",  # the narrowest column of the slider grid
    "gap": "16px",
    "count": "72px",  # the box with the number of steps
}
# The controls take the colours of the picture: the root box carries the class slopewalk-light or slopewalk-dark, and
# control_css sets for each the CSS variables from which ipywidgets 7 and 8 take their colours, so that the controls
# match the picture whatever the notebook's own theme. The accent is the primary button, the active slider handle and
# the focus ring (pressed, normal, focus); white text on the normal accent has 4.6:1 in both.
CONTROL_ACCENT = {"light": ("#0D47A1", "#1976D2", "#64B5F6"), "dark": ("#1158C7", "#1F6FEB", "#58A6FF")}
TITLES = {"core": "Weights and steps", "backprop": "Backpropagation", "views": "Views"}
VIEW_LABELS = {
    "backprop": "backward pass",
    "cases": "each case's slope",
    "slice": "R over two weights",
    "line": "R along the next step",
    "hidden": "the hidden layer",
    "momentum": "momentum",
}
VIEW_HELP = {
    "cases": "every case's slope along one weight; their average is the slope in the table",
    "slice": "contours of R over two weights, with the path of the steps",
    "line": "R along the next step, next to its first-order prediction",
    "hidden": "the cases at their hidden values, with the output unit's boundary",
    "momentum": "the next steps of gradient descent and of momentum",
}
LOOK = {  # where to look in the picture at each stage of the backward pass
    "forward": "The picture shows the forward pass. Press next stage to start at the output.",
    "output": "Under a, the purple box is δa = p - y of each case.",
    "output multiclass": "Under each score ak, the purple box is δak = pk - yk of each case: yk is 1 for the case's "
    "class and 0 for the others.",
    "output regression": "Under ŷ, the purple box is δa = 2(ŷ - y) of each case.",
    "output regression several": "Under each ŷk, the purple box is δak = 2(ŷk - yk)/K of each case, K the number of "
    "targets.",
    "weights": "On the purple lines, each number is δ of the unit times the input of its line.",
    "to_h": "The purple arrows carry δ back along the weights to each h.",
    "delta": "The purple tangents on the curves: their slope multiplies ∂ℓ/∂h to give δ.",
    "average": "The average of these slopes over the cases is the slope column of the table, what R.backward() gives.",
}
HINTS = {
    "core": "Press gradient step to move every weight against its slope, or set a number in the box to take that many "
    "steps. Drag a slider to change one weight.",
    "backprop": "Step from the output back to the first layer. The Worked example below the networks writes out the "
    "numbers for one case.",
    "views": "Tick a box to add a view under the network.",
}
BUTTONS = ("gradient step", "10 steps", "back one step", "reset")  # "10 steps" is the key; its label follows the box
MAX_STEPS = 1000  # the most steps the steps button takes at once


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


def dropdown_arrow(colour):
    """The arrow of a drop-down list in the given colour, as a CSS url of an SVG."""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 18 18'>"
        f"<path fill='{colour}' d='M5.2 5.9 9 9.7l3.8-3.8 1.2 1.2-5 5-5-5z'/></svg>"
    )
    return 'url("data:image/svg+xml;utf8,' + svg.replace("<", "%3C").replace(">", "%3E").replace("#", "%23") + '")'


def control_css():
    """A style element with the colours of the controls for each theme of the picture. ipywidgets resolves its own
    variables (--jp-widgets-color and the rest) once, at the root of the page, so the box sets those as well as the
    base colours; the slider handle of ipywidgets 8 has fixed colours and gets a rule of its own."""
    rules = []
    for name, T in THEMES.items():
        pressed, accent, focus = CONTROL_ACCENT[name]
        edge = T.value_box["ec"]
        values = {
            "color-scheme": name,
            "background-color": T.paper,
            "color": T.ink,
            "padding": "4px 8px",
            "--jp-layout-color0": T.paper,
            "--jp-layout-color1": T.paper,
            "--jp-layout-color2": T.control_button,
            "--jp-layout-color3": T.control_selected,
            "--jp-ui-font-color0": T.ink,
            "--jp-ui-font-color1": T.ink,
            "--jp-ui-font-color2": T.muted,
            "--jp-content-font-color1": T.ink,
            "--jp-ui-inverse-font-color0": "#FFFFFF",
            "--jp-ui-inverse-font-color1": "#FFFFFF",
            "--jp-border-color1": edge,
            "--jp-border-color2": T.table_edge,
            "--jp-brand-color0": pressed,
            "--jp-brand-color1": accent,
            "--jp-brand-color2": focus,
            "--jp-widgets-color": T.ink,
            "--jp-widgets-label-color": T.ink,
            "--jp-widgets-readout-color": T.ink,
            "--jp-widgets-input-color": T.ink,
            "--jp-widgets-input-background-color": T.value_box["fc"],
            "--jp-widgets-input-border-color": edge,
            "--jp-widgets-input-focus-border-color": focus,
            "--jp-widgets-slider-handle-border-color": edge,
            "--jp-widgets-slider-handle-background-color": T.paper if name == "light" else T.ink,
            "--jp-widgets-slider-active-handle-color": accent,
            "--jp-widgets-dropdown-arrow": dropdown_arrow(T.ink),
            "--slopewalk-quiet": "0.72" if name == "light" else "0.85",
        }
        rules.append(f".slopewalk-{name}{{" + ";".join(f"{k}:{v}" for k, v in values.items()) + "}")
        handle = ", ".join(f".slopewalk-{name} .{c} .noUi-handle" for c in ("widget-slider", "jupyter-widget-slider"))
        rules.append(
            f"{handle}{{background:var(--jp-widgets-slider-handle-background-color);"
            "border-color:var(--jp-widgets-slider-handle-border-color)}"
        )
        # VS Code puts every widget output on white (.cell-output-ipywidget-background{background:#fff!important}),
        # which shows as a white frame around a dark picture: the output that holds the box takes its paper instead
        rules.append(f".cell-output-ipywidget-background:has(.slopewalk-{name}){{background:{T.paper}!important}}")
        # and VS Code shades every second table row, which marks the row of numbers in the status
        rules.append(f".slopewalk-{name} table tr{{background-color:transparent}}")
    return "<style>" + "".join(rules) + "</style>"


def theme_class(S):
    """The class of the root box for the session's colours: light or dark, by the brightness of the paper."""
    from matplotlib.colors import to_rgb

    if S.theme_name in THEMES:
        return f"slopewalk-{S.theme_name}"
    return "slopewalk-dark" if sum(to_rgb(S.theme.paper)) < 1.5 else "slopewalk-light"


def span(text, *styles):
    return f'<span style="{";".join(UI[s] for s in styles)}">{text}</span>'


def readout_html(r, action=None):
    """The status numbers: a row of labels, a row of values and a row of notes. R and the change of R in the last
    action (a step, ten steps, back one step or a slider move; action: (change, note)) are large; the other numbers
    are smaller. The columns stay the same before and after the first action."""
    if action is None:
        change, note = span("none yet", "value", "quiet"), ""
    else:
        actual, note = action
        colour = UI["up"] if actual > 0 else UI["down"]
        change = f'<span style="{UI["big"]};color:{colour}">{actual:+.4f}</span>'
    cells = [
        ("R", span(f"{r['R']:.4f}", "big"), ""),
        ("change of R in the last action", change, note),
        (f"next step, eta {r['eta']:g}: predicted change", span(f"{r['next_change']:+.4f}", "value"), ""),
        ("size of the gradient", span(f"{r['grad_norm']:.4f}", "value"), ""),
    ]
    if r["accuracy"] is not None:
        cells.append(("accuracy", span(f"{r['accuracy']:.3f}", "value"), ""))
    if r.get("R_test") is not None:
        cells.append(("R on the test data", span(f"{r['R_test']:.4f}", "value"), ""))
    if r.get("accuracy_test") is not None:
        cells.append(("test accuracy", span(f"{r['accuracy_test']:.3f}", "value"), ""))
    pad = f"padding:0 32px 0 0;text-align:left;vertical-align:bottom;{UI['numbers']}"
    rows = [
        "".join(f'<td style="{pad}">{span(label, "small", "quiet")}</td>' for label, _, _ in cells),
        "".join(f'<td style="{pad}">{value}</td>' for _, value, _ in cells),
        "".join(f'<td style="{pad}">{span(note, "small", "quiet") if note else ""}</td>' for _, _, note in cells),
    ]
    return '<table style="border-collapse:collapse">' + "".join(f"<tr>{row}</tr>" for row in rows) + "</table>"


def message_html(text):
    return f'<div style="{UI["text"]};margin:8px 0 2px 0;min-height:1.4em">{text}</div>'


def text_html(text):
    return f'<div style="{UI["hint"]};{UI["quiet"]};margin:2px 0 8px 0">{text}</div>'


def tip(widget, text):
    """A tooltip, on ipywidgets 7 and 8."""
    if hasattr(widget, "tooltip"):
        widget.tooltip = text
    elif hasattr(widget, "description_tooltip"):
        widget.description_tooltip = text
    return widget


def change_words(before, after):
    d = after - before
    if abs(d) < 5e-5:
        return f"R stayed at {after:.4f}"
    return f"R went from {before:.4f} to {after:.4f}, {'down' if d < 0 else 'up'} {abs(d):.4f}"


class Controller:
    """Builds the widgets for a session and keeps them, the network and the picture in step."""

    def __init__(self, session, dpi=LIVE_DPI):
        import ipywidgets as widgets

        S = self.session = session
        self.dpi = dpi
        self.busy = False  # True while the sliders are set from code: their observers then do nothing
        self.depth = 0  # how many actions are running; the buttons are disabled while it is above 0
        self.picture = None
        self.last_R = None
        self.action = None  # the change of R in the last action and a note on it, for the status
        self.known = len(S.history)  # the length of the history when the controls last looked
        self.use_output = os.environ.get("SLOPEWALK_DISPLAY") == "output"  # fallback: the PNG through display()
        self.index = list(S.active)  # the weight of each slider
        self.sliders = []
        for k in self.index:
            lo, hi = slider_range(S.w0[k])
            slider = widgets.FloatSlider(
                value=S.w0[k],
                min=lo,
                max=hi,
                step=0.001,
                description=S.names[k],
                readout_format=".2f",
                continuous_update=False,
                style={"description_width": UI["label"]},
                layout=widgets.Layout(width=UI["short"]),
            )
            self.sliders.append(tip(slider, f"{S.names[k]}: drag to change this weight in the network"))
        self.panel = self.grouped_panel(widgets) if len(self.sliders) <= 30 else self.dropdown_panel(widgets)
        frozen = [n for k, n in enumerate(S.names) if k not in set(self.index)]
        self.frozen_note = widgets.HTML(text_html(f"Fixed, drawn lighter: {name_ranges(frozen)}.") if frozen else "")

        small = S.eta < 0.1
        self.eta = widgets.FloatSlider(
            value=S.eta,
            min=min(0.1, S.eta),
            max=max(5.0, S.eta),
            step=0.01 if small else 0.1,
            description="eta",
            readout_format=".2f" if small else ".1f",
            continuous_update=False,
            style={"description_width": UI["label"]},
            layout=widgets.Layout(width=UI["long"]),
        )
        tip(self.eta, "eta, the step size: a gradient step moves each weight by eta times its slope")
        self.buttons = {name: widgets.Button(description=name) for name in BUTTONS}
        self.buttons["gradient step"].button_style = "primary"
        tip(self.buttons["gradient step"], "every weight moves against its slope: w <- w - eta x dR/dw")
        self.count = tip(  # the box and its button read as one control: "10 | steps"
            widgets.BoundedIntText(
                value=10, min=1, max=MAX_STEPS, layout=widgets.Layout(width=UI["count"], margin="2px 0 2px 2px")
            ),
            f"how many gradient steps the button to the right takes, 1 to {MAX_STEPS}",
        )
        self.buttons["10 steps"].layout.margin = "2px 2px 2px 0"
        self.count.observe(self.on_count, names="value")
        self.on_count()
        tip(self.buttons["back one step"], "undo the last gradient step, and any slider move made after it")
        tip(self.buttons["reset"], "back to the weights the cell started from")
        self.layout_pick = tip(
            widgets.ToggleButtons(options=["wide", "standard"], value=S.layout.style, style={"button_width": "80px"}),
            "wide: all networks in one row; standard: two per row, with larger text on small screens",
        )
        self.layout_pick.observe(self.on_layout, names="value")
        self.theme_pick = tip(
            widgets.ToggleButtons(
                options=["light", "dark"],
                value=S.theme_name if S.theme_name in ("light", "dark") else "light",
                style={"button_width": "80px"},
            ),
            "the colours of the picture and the controls: light, or dark for a dark notebook theme",
        )
        self.theme_pick.observe(self.on_theme, names="value")

        def label(text, left="0"):
            return widgets.HTML(span(text, "small", "quiet"), layout=widgets.Layout(margin=f"0 12px 0 {left}"))

        row = [label("layout of the picture"), self.layout_pick, label("colours", "32px"), self.theme_pick]
        self.show_boxes = {}
        if S.focus:  # the call asked for one group: the others can be added, any one can be taken away
            row.append(label("show", "32px"))
            for c in range(S.data.G):
                box = widgets.Checkbox(
                    value=c in S.shown,
                    description=letter(c),
                    indent=False,
                    layout=widgets.Layout(width="52px"),
                )
                tip(box, f"draw the network for group {letter(c)}")
                box.observe(self.on_show, names="value")
                self.show_boxes[c] = box
                row.append(box)
        self.layout_row = widgets.HBox(
            row, layout=widgets.Layout(align_items="center", flex_flow="row wrap", margin="14px 0 0 0")
        )

        self.message = widgets.HTML()
        self.readout = widgets.HTML()
        self.image = widgets.Image(format="png", layout=widgets.Layout(max_width="100%", height="auto"))
        self.out = widgets.Output()
        self.status = widgets.VBox([self.readout, self.message], layout=widgets.Layout(margin="6px 0 8px 0"))

        b = self.buttons
        row = [b["gradient step"], self.count, b["10 steps"], b["back one step"], b["reset"]]
        self.button_row = widgets.HBox(row, layout=widgets.Layout(flex_flow="row nowrap", align_items="center"))
        steps_row = widgets.HBox(  # in a narrow window the buttons move below eta together
            [self.eta, self.button_row],
            layout=widgets.Layout(align_items="center", flex_flow="row wrap", margin="0 0 12px 0"),
        )
        hint = HINTS["core"]
        if S.task == "regression" and S.model.n_outputs > 1:
            hint += " The targets count from 1, like the inputs x1, x2."
        rows = [widgets.HTML(text_html(hint)), steps_row]
        if S.optimizer_asked:
            rows.append(self.optimizer_row(widgets))
        core = widgets.VBox(rows + [self.panel, self.frozen_note])
        # tabs only for what the call asked for; the backward pass stays on while its tab exists
        self.kinds, panels = ["core"], {"core": core}
        if "backprop" in S.views:
            self.kinds.append("backprop")
            panels["backprop"] = self.backprop_tab(widgets)
        if any(v != "backprop" for v in S.views):
            self.kinds.append("views")
            panels["views"] = self.views_tab(widgets)
        if len(self.kinds) == 1:
            self.tabs = None
            self.controls = core
        else:
            self.tabs = widgets.Tab(children=[panels[k] for k in self.kinds])
            for i, kind in enumerate(self.kinds):
                self.tabs.set_title(i, TITLES[kind])
            self.tabs.selected_index = self.kinds.index("backprop") if "backprop" in S.views else 0
            self.controls = self.tabs

        for s in self.sliders:
            s.observe(self.on_slider, names="value")
        self.eta.observe(self.on_eta, names="value")
        self.buttons["gradient step"].on_click(lambda _: self.steps(1))
        self.buttons["10 steps"].on_click(lambda _: self.steps(int(self.count.value)))
        self.buttons["back one step"].on_click(lambda _: self.step_back())
        self.buttons["reset"].on_click(lambda _: self.reset())

        # the picture first, then the status, the controls, and last the layout of the picture
        self.picture_box = self.out if self.use_output else self.image
        tail = [] if self.use_output else [self.out]
        self.style = widgets.HTML(control_css(), layout=widgets.Layout(display="none"))
        self.root = widgets.VBox([self.style, self.picture_box, self.status, self.controls, self.layout_row, *tail])
        self.root.layout.width = "100%"
        self.paint()
        self.arrange(S.layout.style)
        self.widgets = [self.root]
        self.update_buttons()
        self.say(self.stage_words() if "backprop" in S.views else "")

    # ---------------------------------------------------------------- building

    def paint(self):
        """The controls in the colours of the picture: the root box gets the class of the session's theme."""
        wanted = theme_class(self.session)
        for name in THEMES:
            if f"slopewalk-{name}" != wanted:
                self.root.remove_class(f"slopewalk-{name}")
        self.root.add_class(wanted)

    def arrange(self, style):
        """The picture takes the full width in the wide layout; in the standard one it keeps its own width."""
        self.picture_box.layout.max_width = "100%"
        self.picture_box.layout.width = "100%" if style == "wide" else ""

    def grouped_panel(self, widgets):
        """Up to 30 sliders: one column per unit with weights that move, headed by the unit and its inputs; the
        columns line up in a grid."""
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
        columns = f"repeat(auto-fill, minmax({UI['column']}, 1fr))"
        return widgets.GridBox(boxes, layout=widgets.Layout(grid_template_columns=columns, grid_gap=f"2px {UI['gap']}"))

    def dropdown_panel(self, widgets):
        """More than 30 sliders: choose one weight, then move it."""
        names = [self.session.names[k] for k in self.index]
        pick = widgets.Dropdown(
            options=[(nm, i) for i, nm in enumerate(names)],
            value=0,
            description="weight",
            style={"description_width": UI["label"]},
            layout=widgets.Layout(width=UI["short"]),
        )
        tip(pick, "choose the weight to move")
        holder = widgets.HBox([self.sliders[0]], layout=widgets.Layout(margin=f"0 0 0 {UI['gap']}"))
        pick.observe(lambda ch: setattr(holder, "children", [self.sliders[ch["new"]]]), names="value")
        return widgets.HBox([pick, holder])

    def backprop_tab(self, widgets):
        """Previous and next stage, and the case of the Worked example below the networks. The picture says which
        stage it shows (the Worked example's title), so there is no stage slider."""
        S = self.session
        self.bp_prev = tip(widgets.Button(description="previous stage"), "the stage before")
        self.bp_next = tip(
            widgets.Button(description="next stage", button_style="primary"), "the next stage, one layer back"
        )
        letters = [letter(c) for c in range(S.data.G)]
        self.bp_case = widgets.Dropdown(
            options=letters,
            value=letters[S.backprop_case],
            description="Worked example",
            style={"description_width": UI["setting_label"]},
            layout=widgets.Layout(width=UI["short"]),
        )
        tip(self.bp_case, "the case whose numbers the Worked example writes out (in a group: the one nearest its mean)")
        self.bp_prev.on_click(lambda _: self.to_stage(-1))
        self.bp_next.on_click(lambda _: self.to_stage(+1))
        self.bp_case.observe(self.on_backprop, names="value")
        self.bp_where = widgets.HTML(layout=widgets.Layout(margin=f"0 {UI['gap']}"))  # "stage 2 of 6"
        self.show_stage()
        row = widgets.HBox(
            [self.bp_prev, self.bp_where, self.bp_next, self.bp_case],
            layout=widgets.Layout(align_items="center", flex_flow="row wrap"),
        )
        return widgets.VBox([widgets.HTML(text_html(HINTS["backprop"])), row])

    def show_stage(self):
        """The stage between the stage buttons, quiet, so that the controls also say where the student is."""
        S = self.session
        self.bp_where.value = span(f"stage {S.backprop_step} of {len(S.backprop_stages) - 1}", "text", "quiet")

    def views_tab(self, widgets):
        """A checkbox per view with a line on what it shows, and the settings of the views."""
        S = self.session
        names = list(S.names)
        self.view_boxes = {
            v: tip(
                widgets.Checkbox(
                    value=v in S.views,
                    description=VIEW_LABELS[v],
                    indent=False,
                    layout=widgets.Layout(width="200px"),
                ),
                VIEW_HELP[v],
            )
            for v in VIEWS
            if v != "backprop"
        }

        def setting(label=UI["setting_label"], width=UI["long"]):  # each setting with its own layout
            return dict(style={"description_width": label}, layout=widgets.Layout(width=width))

        self.case_pick = tip(
            widgets.Dropdown(options=names, value=names[S.case_weight], description="case slopes of", **setting()),
            "the weight of the view 'each case's slope'",
        )
        a = S.slice_weights[0]
        b = S.slice_weights[1] if len(S.slice_weights) > 1 else None
        self.slice_pick = [
            tip(
                widgets.Dropdown(options=names, value=names[a], description="slice over", **setting(width=UI["short"])),
                "the first weight",
            ),
            tip(
                widgets.Dropdown(
                    options=["none"] + names,
                    value="none" if b is None else names[b],
                    description="and",
                    **setting(label=UI["label"], width="190px"),
                ),
                "the second weight, or none for a curve over one weight",
            ),
        ]
        self.beta = tip(
            widgets.FloatSlider(
                value=S.beta, min=0.0, max=0.99, step=0.01, description="beta", continuous_update=False, **setting()
            ),
            "momentum: the share of the last step that each step keeps",
        )
        self.ahead = tip(
            widgets.IntSlider(
                value=S.lookahead_steps,
                min=1,
                max=200,
                description="steps ahead",
                continuous_update=False,
                **setting(),
            ),
            "how many steps the momentum view looks ahead",
        )
        self.log_box = tip(
            widgets.Checkbox(value=S.log_scale, description="R on a log scale", indent=False),
            "draw R on a logarithmic axis",
        )
        for w in [*self.view_boxes.values(), self.case_pick, *self.slice_pick, self.beta, self.ahead, self.log_box]:
            w.observe(self.on_views, names="value")
        self.settings = {"cases": [self.case_pick], "slice": self.slice_pick, "momentum": [self.beta, self.ahead]}
        rows = [  # each view's checkbox, what it shows, and its settings in the same row
            widgets.HBox(
                [box, widgets.HTML(span(VIEW_HELP[v], "small", "quiet"), layout=widgets.Layout(width="460px"))]
                + self.settings.get(v, []),
                layout=widgets.Layout(align_items="center", flex_flow="row wrap"),
            )
            for v, box in self.view_boxes.items()
        ]
        self.update_settings()
        return widgets.VBox([widgets.HTML(text_html(HINTS["views"])), *rows, self.log_box])

    def update_settings(self):
        """The settings of a view can be changed while the view is on."""
        for v, controls in self.settings.items():
            for w in controls:
                w.disabled = not self.view_boxes[v].value

    # ---------------------------------------------------------------- state

    def say(self, text):
        """The message under the status; hidden while empty, so that it leaves no gap."""
        self.message.value = message_html(text) if text else ""
        self.message.layout.display = "" if text else "none"

    def stage_words(self):
        """The message for the current stage of the backward pass: where to look in the picture."""
        S = self.session
        kind = S.backprop_stages[S.backprop_step].kind
        several = " several" if S.task == "regression" and S.model.n_outputs > 1 else ""
        return LOOK.get(f"{kind} {S.task}{several}", LOOK.get(f"{kind} {S.task}", LOOK[kind]))

    def action_buttons(self):
        return [*self.buttons.values()] + ([self.bp_prev, self.bp_next] if "backprop" in self.kinds else [])

    def update_buttons(self):
        """Enables the buttons, except back one step with no step to undo and the stage buttons at the ends."""
        S = self.session
        for b in self.action_buttons():
            b.disabled = False
        self.buttons["back one step"].disabled = not S.undo
        if "backprop" in self.kinds:
            self.bp_prev.disabled = S.backprop_step == 0
            self.bp_next.disabled = S.backprop_step == len(S.backprop_stages) - 1

    @contextmanager
    def working(self):
        """Disables the buttons while an action runs and the picture is drawn again, so that clicks do not queue up;
        enables them afterwards, also when the action fails."""
        self.depth += 1
        if self.depth == 1:
            for b in self.action_buttons():
                b.disabled = True
        try:
            yield
        finally:
            self.depth -= 1
            if self.depth == 0:
                self.update_buttons()

    def values(self):
        """The full weight vector: the sliders for the weights that move, the network for the others."""
        w = self.session.weights()
        for k, s in zip(self.index, self.sliders):
            w[k] = s.value
        return w

    def on_slider(self, change=None):
        if self.busy:
            return
        before = self.last_R
        S = self.session
        if change is not None:
            S.last_moved = self.index[self.sliders.index(change["owner"])]
            if before is not None:
                self.action = (S.risk(self.values()) - before, f"{change['owner'].description} moved by hand")
        S.model.set_weights(self.values())
        self.redraw()
        if change is not None and before is not None:
            name, old, new = change["owner"].description, change["old"], change["new"]
            self.say(f"{name} changed from {old:.2f} to {new:.2f}: {change_words(before, self.last_R)}.")

    def on_count(self, *_):
        """The steps button, joined to the box with the number: "steps" (or "step" for 1), so that the two read as
        "10 steps" once."""
        n = int(self.count.value)
        button = self.buttons["10 steps"]
        button.description = "step" if n == 1 else "steps"
        tip(button, f"{n} gradient step{'' if n == 1 else 's'} in a row; the number is in the box to the left")

    def optimizer_row(self, widgets):
        """The rule of a step and its numbers: the optimizer, the weight decay and beta (shown when the call asked
        for an optimizer or a weight decay)."""
        S = self.session
        opt = S.optimizer
        self.opt_pick = tip(
            widgets.Dropdown(
                options=[("gradient descent", "gd"), ("momentum", "momentum"), ("Adam", "adam")],
                value=opt.kind,
                description="optimizer",
                style={"description_width": UI["setting_label"]},
                layout=widgets.Layout(width=UI["short"]),
            ),
            "the rule of a step: w <- w - eta x dR/dw; momentum keeps a share beta of the last step; Adam scales "
            "each weight's step by its recent slopes",
        )
        self.decay = tip(
            widgets.FloatSlider(
                value=opt.weight_decay,
                min=0.0,
                max=max(0.1, opt.weight_decay),
                step=0.001,
                description="weight decay",
                readout_format=".3f",
                continuous_update=False,
                style={"description_width": UI["setting_label"]},
                layout=widgets.Layout(width=UI["long"]),
            ),
            "lambda: lambda x w is added to each slope before the step; R itself stays the loss on the data",
        )
        self.opt_beta = tip(
            widgets.FloatSlider(
                value=opt.beta,
                min=0.0,
                max=0.99,
                step=0.01,
                description="beta",
                continuous_update=False,
                style={"description_width": UI["label"]},
                layout=widgets.Layout(width=UI["long"]),
            ),
            "momentum: the share of the last step each step keeps; Adam: its share of the last slopes",
        )
        for w in (self.opt_pick, self.decay, self.opt_beta):
            w.observe(self.on_optimizer, names="value")
        return widgets.HBox(
            [self.opt_pick, self.decay, self.opt_beta],
            layout=widgets.Layout(align_items="center", flex_flow="row wrap", margin="0 0 12px 0"),
        )

    def on_optimizer(self, *_):
        """A new rule or new numbers for the steps; a new rule forgets the state of the old one."""
        if self.busy:
            return
        S = self.session
        S.optimizer.set(kind=self.opt_pick.value, weight_decay=float(self.decay.value), beta=float(self.opt_beta.value))
        S.beta = float(self.opt_beta.value)
        if hasattr(self, "beta"):  # the momentum view shares beta
            self.busy = True
            try:
                self.beta.value = S.beta
            finally:
                self.busy = False
        self.redraw()
        name = {"gd": "gradient descent", "momentum": "momentum", "adam": "Adam"}[S.optimizer.kind]
        r = S.readouts(self.eta.value)
        self.say(
            f"Steps by {name}, weight decay {S.optimizer.weight_decay:g}: the next step should change R by about "
            f"{r['next_change']:+.4f}."
        )

    def on_eta(self, change):
        self.session.eta = float(change["new"])
        if not self.busy:
            self.redraw()
            eta = self.session.eta
            r = self.session.readouts(eta)
            self.say(f"eta is now {eta:g}: the next step should change R by about {r['next_change']:+.4f}.")

    def to_stage(self, change):
        """One stage on (+1) or back (-1), within the first and the last stage."""
        S = self.session
        S.backprop_step = max(0, min(len(S.backprop_stages) - 1, S.backprop_step + change))
        self.on_backprop()

    def on_backprop(self, *_):
        S = self.session
        S.backprop_case = ord(self.bp_case.value) - 65 if len(self.bp_case.value) == 1 else int(self.bp_case.value)
        self.show_stage()
        if not self.busy:
            self.redraw()
            self.say(self.stage_words())

    def on_layout(self, change):
        """Switches between the wide and the standard layout: a new figure."""
        S = self.session
        S.set_layout(change["new"])
        self.arrange(change["new"])
        self.picture = None
        self.redraw()
        where = "all networks in one row" if change["new"] == "wide" else "two networks per row, larger text"
        self.say(f"Layout {change['new']}: {where}.")

    def on_show(self, change):
        """A group's checkbox: its network panel comes or goes (a new figure); the last one cannot be taken away."""
        if self.busy:
            return
        chosen = [c for c, box in self.show_boxes.items() if box.value]
        if not chosen:
            self.busy = True
            try:
                change["owner"].value = True
            finally:
                self.busy = False
            self.say("One group stays shown.")
            return
        S = self.session
        S.set_shown(chosen)
        self.picture = None
        self.redraw()
        letters = ", ".join(letter(c) for c in chosen)
        self.say(f"Networks shown for {letters}" + (", drawn larger." if len(chosen) == 1 else "."))

    def on_theme(self, change):
        """Switches the colours of the picture and the controls: a new figure."""
        self.session.set_theme(change["new"])
        self.paint()
        self.picture = None
        self.redraw()
        where = "for a dark notebook theme" if change["new"] == "dark" else "for a light notebook theme"
        self.say(f"Colours {change['new']}: {where}.")

    def on_views(self, *_):
        if self.busy:
            return
        S = self.session
        names = list(S.names)
        keep = ("backprop",) if "backprop" in S.views else ()
        S.views = tuple(v for v in VIEWS if v in keep or (v in self.view_boxes and self.view_boxes[v].value))
        S.case_weight = names.index(self.case_pick.value)
        first = names.index(self.slice_pick[0].value)
        second = self.slice_pick[1].value
        S.slice_weights = [first] if second in ("none", self.slice_pick[0].value) else [first, names.index(second)]
        S.beta = float(self.beta.value)
        S.optimizer.set(beta=S.beta)
        if hasattr(self, "opt_beta") and self.opt_beta.value != S.beta:
            self.busy = True
            try:
                self.opt_beta.value = S.beta
            finally:
                self.busy = False
        S.lookahead_steps = int(self.ahead.value)
        S.log_scale = bool(self.log_box.value)
        self.picture = None  # the panels changed: a new figure
        self.redraw()
        self.update_settings()
        shown = [VIEW_LABELS[v] for v in S.views if v != "backprop"]
        self.say("Views: " + (", ".join(shown) if shown else "none") + ".")

    def note_action(self):
        """The last action, read from the history: k new steps, a step undone, or a fresh start."""
        S = self.session
        n = len(S.history)
        if n == 1 and not S.predictions:
            self.action = None
        elif n > self.known and self.last_R is not None:
            k = n - self.known
            predicted = sum(p.predicted - p.before for p in S.predictions[-k:])
            what = "one step" if k == 1 else f"{k} steps"
            self.action = (S.history[-1] - self.last_R, f"{what}, predicted {predicted:+.4f}")
        elif n < self.known and self.last_R is not None:
            self.action = (S.risk() - self.last_R, "back one step")
        self.known = n

    def sync(self):
        """After steps, back one step, reset or record, also in code: the status notes the action, the sliders show
        the network's weights, then one redraw."""
        self.note_action()
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
        with self.working():
            S.eta = float(self.eta.value)
            before = S.risk(self.values())
            S.step(m, S.eta, self.values())
            after = S.history[-1]
            if m == 1:
                p = S.predictions[-1]
                text = (
                    f"One gradient step with eta {S.eta:g}: {change_words(before, after)}. "
                    f"Predicted change {p.predicted - p.before:+.4f}, actual {p.actual - p.before:+.4f}."
                )
            else:
                text = f"{m} gradient steps with eta {S.eta:g}: {change_words(before, after)}."
            if after > before:
                text += " R went up: eta may be too large."
            self.say(text)

    def step_back(self):
        """Undoes the last gradient step; the session then updates the sliders."""
        S = self.session
        with self.working():
            before = S.risk(self.values())
            if S.step_back() is not None:
                self.say(f"Back one step: {change_words(before, S.risk())}.")

    def reset(self):
        """Back to the weights at the start, with no steps taken, and the sliders' starting ranges."""
        S = self.session
        with self.working():
            S.reset()
            self.busy = True
            try:
                for k, s in zip(self.index, self.sliders):
                    lo, hi = slider_range(S.w0[k])
                    s.min, s.max = lo, hi
            finally:
                self.busy = False
            self.say(f"Back to the start: the weights the cell started from, R = {S.history[0]:.4f}.")

    def redraw(self):
        S = self.session
        with self.working(), self.out:
            if self.picture is None:
                self.picture = Picture(S, dpi=self.dpi)
            w = self.values()
            self.picture.refresh(w)
            S.fig = self.picture.fig
            self.last_R = self.picture.frame.R
            png = self.picture.png()
            self.readout.value = readout_html(S.readouts(self.eta.value, w, frame=self.picture.frame), self.action)
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

# ------------------------ host.py ------------------------

# VS Code's own themes, as written in its settings
VSCODE_LIGHT = {
    "Default Light Modern",
    "Default Light+",
    "Visual Studio Light",
    "Default High Contrast Light",
    "Quiet Light",
    "Solarized Light",
}
VSCODE_DARK = {
    "Default Dark Modern",
    "Default Dark+",
    "Visual Studio Dark",
    "Default High Contrast",
    "Abyss",
    "Kimbie Dark",
    "Monokai",
    "Monokai Dimmed",
    "Red",
    "Solarized Dark",
    "Tomorrow Night Blue",
}


def auto_theme():
    """ "light" or "dark" for explore(theme="auto"): SLOPEWALK_THEME if set; light for still pictures (NB_STATIC=1) and
    in Colab; else the theme of VS Code or JupyterLab read from their settings, and light when that fails."""
    forced = os.environ.get("SLOPEWALK_THEME", "").strip().lower()
    if forced in ("light", "dark"):
        return forced
    if os.environ.get("NB_STATIC") == "1" or "google.colab" in sys.modules:
        return "light"
    try:
        found = vscode_theme() if os.environ.get("VSCODE_PID") else jupyterlab_theme()
    except Exception:  # a setting the kernel cannot read is not worth an error: the switch is there
        found = None
    return found or "light"


def read_jsonc(path):
    """A settings file in JSON with comments and trailing commas, as VS Code and JupyterLab write them, or None."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] == '"':  # a string, kept as it is
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(text[i])
            i += 1
    try:
        data = json.loads(re.sub(r",(\s*[}\]])", r"\1", "".join(out)))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def system_theme():
    """ "dark" or "light" from the system's appearance (macOS, Windows, GNOME), or None."""
    try:
        if sys.platform == "darwin":
            done = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True, timeout=2
            )
            return "dark" if done.stdout.strip().lower() == "dark" else "light"
        if sys.platform.startswith("win"):
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            )
            return "light" if winreg.QueryValueEx(key, "AppsUseLightTheme")[0] else "dark"
        done = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if done.returncode == 0:
            return "dark" if "dark" in done.stdout else "light"
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def vscode_folder():
    """VS Code's folder of user data (it holds User/settings.json): from the kernel's environment, which VS Code
    sets, or the usual place."""
    cache = os.environ.get("VSCODE_CODE_CACHE_PATH")  # .../Code/CachedData/<version>
    if cache and Path(cache).parent.name == "CachedData":
        return Path(cache).parent.parent
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Code"
    if sys.platform.startswith("win"):
        return Path(os.environ.get("APPDATA", Path.home())) / "Code"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "Code"


def vscode_theme():
    """The kind of theme VS Code shows. VS Code keeps the kind of the theme it last painted in
    User/globalStorage/storage.json (for its start-up screen), which follows extension themes and the system's
    appearance; without it, its settings: the system's appearance when VS Code follows it, else the name of its
    color theme (dark when none is set, VS Code's default)."""
    folder = vscode_folder()
    try:
        storage = json.loads((folder / "User" / "globalStorage" / "storage.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        storage = {}
    splash = storage.get("windowSplash") if isinstance(storage, dict) else None
    base = storage.get("theme") if isinstance(storage, dict) else None
    base = base or (splash.get("baseTheme") if isinstance(splash, dict) else None)
    if base in ("vs", "hc-light"):
        return "light"
    if base in ("vs-dark", "hc-black"):
        return "dark"
    settings = read_jsonc(folder / "User" / "settings.json") or {}
    if settings.get("window.autoDetectColorScheme"):
        return system_theme()
    return theme_kind(str(settings.get("workbench.colorTheme", "Default Dark Modern")))


def theme_kind(name):
    """ "light" or "dark" for the name of a VS Code color theme: VS Code's own themes, else a word in the name, or
    None."""
    if name in VSCODE_LIGHT:
        return "light"
    if name in VSCODE_DARK:
        return "dark"
    words = name.lower()
    if "light" in words:
        return "light"
    if any(w in words for w in ("dark", "night", "black", "monokai", "dracula")):
        return "dark"
    return None


def jupyterlab_theme():
    """The theme of JupyterLab or Notebook 7 on this machine, from its user settings, or None (the default, light)."""
    try:
        from jupyter_core.paths import jupyter_config_dir

        base = Path(jupyter_config_dir())
    except ImportError:  # jupyter_core comes with Jupyter; without it, its rule: JUPYTER_CONFIG_DIR, else ~/.jupyter
        base = Path(os.environ.get("JUPYTER_CONFIG_DIR") or Path.home() / ".jupyter")
    path = base / "lab" / "user-settings" / "@jupyterlab" / "apputils-extension" / "themes.jupyterlab-settings"
    settings = read_jsonc(path)
    if not settings:
        return None
    if settings.get("adaptive-theme"):
        return system_theme()
    words = str(settings.get("theme", "JupyterLab Light")).lower()
    if "light" in words:
        return "light"
    if "dark" in words or "night" in words:
        return "dark"
    return None

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


def auto_layout():
    """The layout of explore(layout="auto"): wide for the widgets in Jupyter or VS Code; standard in Colab (its
    notebooks have a fixed width), for still pictures (NB_STATIC=1) and without ipywidgets."""
    import importlib.util
    import sys

    if os.environ.get("NB_STATIC") == "1" or "google.colab" in sys.modules:
        return "standard"
    if importlib.util.find_spec("ipywidgets") is None:
        return "standard"
    return "wide"


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

    weights(), risk(), slopes(), predict(), step(k), step_back(), reset() work on the network itself; lookahead() and
    case_slopes() leave it as it is. save(path) writes the current picture and record(path) a GIF of a run.
    history holds R at the start and after every step; predictions one Prediction per step; path the weights after
    every step (a row of NaN marks a jump made with the sliders); w0 the weights at the start; active the indices of
    the weights that move; last_moved the weight a slider changed last (None after a step); controller the widgets
    or None.
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
        layout="standard",
        width=None,
        backprop_step=0,
        backprop_case=0,
        focus=None,
        X_test=None,
        y_test=None,
        optimizer=None,
        weight_decay=None,
        beta2=0.999,
    ):
        self.model = Model(net)
        M = self.model
        self.data = prepare(X, y, groups, n_inputs=M.n_inputs, dtype=M.dtype, task=task, n_outputs=M.n_outputs)
        self.task = self.model.task = self.data.task
        if (X_test is None) != (y_test is None):
            raise ValueError("slopewalk: give X_test and y_test together, or neither.")
        self.test = None  # test data: drawn hollow, R on it in the status and the R panel, never in the steps
        if X_test is not None:
            self.test = prepare(
                X_test, y_test, None, n_inputs=M.n_inputs, dtype=M.dtype, task=self.task, n_outputs=M.n_outputs
            )
        self.optimizer_asked = optimizer is not None or weight_decay is not None  # the widgets then show the row
        self.optimizer = Optimizer(optimizer or "gd", weight_decay or 0.0, check_beta(beta), beta2)
        self.layout = Layout(M.n_inputs, M.hidden, self.task, style=layout, width=width, outputs=M.n_outputs)
        self.base_theme, self.theme_name = THEMES["light"], "light"
        self.shown = list(range(self.data.G))
        self.set_theme(theme)
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
        n_stages = len(backprop_stages(self.model))
        if not (isinstance(backprop_step, (int, np.integer)) and 0 <= backprop_step < n_stages):
            raise ValueError(f"slopewalk: backprop_step must be a whole number from 0 to {n_stages - 1}.")
        self.backprop_step = int(backprop_step)
        self.backprop_case = self.group_index(backprop_case)
        if self.backprop_step > 0:
            self.views = as_views(self.views + ("backprop",))
        self.focus = focus is not None  # the call asked for one group: the widgets offer the others as checkboxes
        self.set_shown([self.group_index(focus)] if self.focus else range(self.data.G))
        self.w0 = self.model.get_weights()
        start = self.model.forward(self.data.X, self.data.y)
        self.loss0 = [self.data.stats(start.loss, c)[0] for c in range(self.data.G)]
        self.history = [self.risk(self.w0)]
        self.history_test = [self.risk_test(self.w0)]
        self.predictions = []
        self.path = [list(self.w0)]
        self.undo = []  # per step taken: the weights before it, the length of path before it, the optimizer's state
        self.last_moved = None
        self.cache = {}
        self.fig = None
        self.controller = None

    @property
    def names(self):
        return self.model.names

    @property
    def backprop_stages(self):
        """The stages of the backward pass: forward, the output's delta, then layer by layer back to the input."""
        return backprop_stages(self.model)

    def group_index(self, v):
        """The index of a group given by its letter ("A") or its index."""
        G = self.data.G
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool) and 0 <= v < G:
            return int(v)
        if isinstance(v, str) and len(v) == 1 and 0 <= ord(v.upper()) - 65 < G:
            return ord(v.upper()) - 65
        raise ValueError(f"slopewalk: {v!r} is not a group; the groups are A to {chr(64 + G)} (or 0 to {G - 1}).")

    def set_layout(self, style, width=None):
        """Switches the layout of the picture: "standard" or "wide" (the widgets also move the controls)."""
        M = self.model
        self.layout = Layout(M.n_inputs, M.hidden, self.task, style=style, width=width, outputs=M.n_outputs)
        self.layout.focus(len(self.shown) == 1)
        self.scale_theme()

    def set_theme(self, theme):
        """The colours of the picture and the controls: "light", "dark", "auto" (the editor's theme where the kernel
        can read it, see host.py), or a Theme (the widgets draw a new picture)."""
        if theme == "auto":
            theme = auto_theme()
        self.base_theme = as_theme(theme)
        self.theme_name = theme if isinstance(theme, str) else ("light" if theme is None else "custom")
        self.scale_theme()

    def set_shown(self, groups):
        """The groups drawn as network panels, in the order of the groups: all of them, or any non-empty subset by
        letter or index. One group alone is drawn larger. R, the table and the panels below stay over every point."""
        chosen = sorted({self.group_index(g) for g in groups})
        if not chosen:
            raise ValueError("slopewalk: at least one group must be shown.")
        self.shown = chosen
        self.layout.focus(len(chosen) == 1)
        self.scale_theme()

    def scale_theme(self):
        """The theme at the text size of the layout: larger in the wide layout, and for one panel shown alone."""
        self.theme = self.base_theme.scaled(self.layout.text_scale * self.layout.zoom)

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

    def risk_test(self, w=None):
        """R on the test data at the weights w (default: the current ones), or None without test data. Sets the
        network to w."""
        if self.test is None:
            if w is not None:
                self.model.set_weights(w)
            return None
        return self.model.risk(self.test.X, self.test.y, w)

    def next_step(self, eta=None, w=None, g=None):
        """The change of each weight that the next step would make, as {k: dw}, from w (default: the current weights)
        and the slopes g (computed when not given), without taking it."""
        eta = self.eta if eta is None else check_eta(eta)
        w = self.weights() if w is None else [float(v) for v in w]
        g = self.slopes(w) if g is None else g
        return self.optimizer.direction(g, w, self.active, eta)

    def predict(self, eta=None, w=None):
        """The first-order prediction of the change of R for one step from w: the slopes times the step, g . dw
        (-eta times the sum of the squared slopes of the weights that move, for plain gradient descent)."""
        w = self.weights() if w is None else [float(v) for v in w]
        g = self.slopes(w)
        dw = self.next_step(eta, w, g)
        return float(sum(g[k] * dw[k] for k in dw))

    def step(self, k=1, eta=None, w=None):
        """k gradient steps, w <- w - eta * dR/dw for the weights that move, from w (default: the current weights).
        Records R, the prediction and the weights of every step. Returns the new weights."""
        eta = self.eta if eta is None else check_eta(eta)
        w = [float(v) for v in (self.weights() if w is None else w)]
        mark = len(self.path)
        if any(abs(a - b) > 1e-6 * max(1.0, abs(b)) for a, b in zip(w, self.path[-1])):  # beyond float32 rounding
            self.path.append([math.nan] * len(w))  # the sliders moved the weights: a new segment of the path
            self.path.append(list(w))
        R = self.risk(w)
        for i in range(k):
            self.undo.append((list(w), mark if i == 0 else len(self.path), self.optimizer.copy()))
            g = self.slopes(w)
            dw = self.optimizer.step(g, w, self.active, eta)
            predicted = R + float(sum(g[j] * dw[j] for j in dw))
            for j, d in dw.items():
                w[j] += d
            R_new = self.risk(w)
            self.predictions.append(Prediction(R, predicted, R_new))
            self.history.append(R_new)
            self.history_test.append(self.risk_test(w))
            self.path.append(list(w))
            R = R_new
        self.last_moved = None  # a step moves every weight
        self.model.set_weights(w)
        self.changed()
        return w

    def step_back(self):
        """Undoes the last gradient step: the weights return to what they were just before it, so a slider move made
        after the step is undone as well, and the step leaves history, predictions and path. Returns the weights, or
        None when there is no step to undo."""
        if not self.undo:
            return None
        w, mark, state = self.undo.pop()
        self.optimizer = state
        del self.path[mark:]
        self.history.pop()
        self.history_test.pop()
        self.predictions.pop()
        self.last_moved = None
        self.model.set_weights(w)
        self.changed()
        return list(w)

    def reset(self):
        """Back to the weights at the start, with no steps taken. Returns the weights."""
        self.model.set_weights(self.w0)
        self.history = [self.risk(self.w0)]
        self.history_test = [self.risk_test(self.w0)]
        self.predictions = []
        self.path = [list(self.w0)]
        self.undo = []
        self.optimizer.reset()
        self.last_moved = None
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

    def readouts(self, eta=None, w=None, frame=None):
        """The numbers of the readout strip: R, the size of the gradient of the weights that move, the accuracy
        (classification), the predicted change of R for the next step, and the last step's predicted and actual
        change. frame: the numbers the picture computed for w, so that nothing is computed twice."""
        eta = self.eta if eta is None else check_eta(eta)
        w = self.weights() if w is None else list(w)
        if frame is not None and frame.eta == eta and list(frame.w) == [float(v) for v in w]:
            R, g, next_change, R_test = frame.R, frame.g, frame.next_change, frame.R_test
        else:
            R, g = self.risk(w), self.slopes(w)
            dw = self.next_step(eta, w, g)
            next_change, R_test = float(sum(g[k] * dw[k] for k in dw)), self.risk_test(w)
        g2 = float(sum(g[k] ** 2 for k in self.active))
        last = self.predictions[-1] if self.predictions else None
        classify = self.task != "regression"
        out = {
            "R": R,
            "grad_norm": math.sqrt(g2),
            "accuracy": self.model.accuracy(self.data.X, self.data.y, w) if classify else None,
            "eta": eta,
            "next_change": next_change,
            "last": None if last is None else (last.predicted - last.before, last.actual - last.before),
            "R_test": R_test,
            "accuracy_test": self.model.accuracy(self.test.X, self.test.y, w) if classify and self.test else None,
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

    def record_backprop(self, path, fps=1, dpi=80):
        """Writes every stage of the backward pass at the current weights to an animated GIF (or .mp4 with ffmpeg),
        one frame per stage. The weights do not change. Returns path."""
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
        views, step = self.views, self.backprop_step
        self.views = as_views(views + ("backprop",))
        try:
            picture = Picture(self, dpi=dpi)
            with writer.saving(picture.fig, path, dpi):
                for k in range(len(self.backprop_stages)):
                    self.backprop_step = k
                    picture.refresh()
                    writer.grab_frame()
        finally:
            self.views, self.backprop_step = views, step
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
    layout="auto",
    width=None,
    backprop_step=0,
    backprop_case=0,
    theme="auto",
    focus=None,
    X_test=None,
    y_test=None,
    optimizer=None,
    weight_decay=None,
    beta2=0.999,
):
    """Gradient descent by hand on `net`, with inputs X (n x d) and labels y.

    net: nn.Sequential of nn.Linear layers with nn.Tanh, nn.ReLU, nn.Sigmoid or nothing between them, ending in
    nn.Linear(..., 1) or nn.Linear(..., K). One output is the score a of class 1 (labels 0 or 1; R is the average log
    loss) or, with task="regression", the prediction (R is the mean squared error). K outputs, for K classes up to 4,
    are the scores of the classes (labels 0 to K-1; R is the average cross-entropy; task="multiclass"), or with
    task="regression" the predictions of K targets (y an n x K array; R is the mean squared error over all of them).
    groups: a number per point; one panel per group. None: one panel per point (up to 8 points), else one per class.
    eta: the step size of the gradient step buttons (a slider).
    params: the weights that move: a list of names such as ["w10", "w11"], "output" or "layer:k". The others are
    drawn lighter and stay fixed.
    views: extra panels, any of "cases" (each case's slope along one weight; case_slopes="w10" chooses it), "slice"
    (R over one or two weights; slice_weights chooses them), "line" (R along the next step), "hidden" (the cases at
    their hidden values) and "momentum" (the next `lookahead` steps of gradient descent and of momentum beta).
    log_scale: R on a log axis.
    layout: "wide" (network panels side by side and the lower panels in one row), "standard" (two network panels per
    row), or "auto" (the default): wide where the sliders run, standard in Colab, whose notebooks have a fixed width,
    and for still pictures. The controls are under the picture in both. width: the figure width in inches.
    backprop_step: with the view "backprop", the stage of the backward pass drawn (0: the forward pass; a step above
    0 switches the view on); backprop_case: the group whose case the Worked example follows ("A" or 0 is the first).
    theme: the colours of the picture and its controls: "light", "dark", or "auto" (the default), which follows the
    editor's theme where the kernel can read it (VS Code, and JupyterLab or Notebook 7, on the same machine) and is
    light elsewhere, Colab included; SLOPEWALK_THEME=dark (or light) in the environment chooses for every notebook.
    focus: one group ("A" or 0) drawn alone and larger, for a close look at one case; checkboxes under the controls
    add the other groups. R, the table and the panels below the networks stay over every point.
    X_test, y_test: test data, in the shapes of X and y. Its points are drawn as crosses, R on it joins the R panel
    and the status, and it takes no part in the steps.
    optimizer: the rule of a step: "gd" (the default, w <- w - eta * dR/dw), "momentum" (each step keeps the share
    beta of the last one) or "adam" (beta and beta2 are its shares of the last slopes and squared slopes).
    weight_decay: lambda, added to each slope as lambda * w before the step (as torch.optim's weight_decay). R stays
    the mean loss on the data, so the table and the backward pass show its slopes; the first-order prediction is the
    slopes times the step. Passing optimizer or weight_decay shows a row of controls for them.

    Draws one copy of the network per group (every value a mean ± standard deviation over the group), with each
    weight on its line, the units as z, activation and h, then the output and the loss with its change from the
    start. Below: the input plane, R over the steps taken with the first-order prediction of each step, and a table
    with each weight's value, the change of R for +0.1 and the slope dR/dw from R.backward().
    With ipywidgets: under the picture, the status (R and the change of R in the last action), then sliders for the
    weights, eta, `gradient step`, a box with a number of steps joined to the `steps` button that takes that many
    (10 at first), `back one step`, `reset`, and switches for the layout and the colours of the picture. The sliders
    change the weights of `net` itself. Without ipywidgets, or with NB_STATIC=1, a still picture of the start.
    Returns a Session: step(k), step_back(), reset(), save(path), record(path), weights(), history.
    """
    if layout == "auto":
        layout = auto_layout()
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
        layout=layout,
        width=width,
        backprop_step=backprop_step,
        backprop_case=backprop_case,
        focus=focus,
        X_test=X_test,
        y_test=y_test,
        optimizer=optimizer,
        weight_decay=weight_decay,
        beta2=beta2,
    )
    session.set_theme(theme)
    if os.environ.get("NB_STATIC") == "1":
        return still_picture(session, STATIC_MESSAGE)
    try:
        import ipywidgets  # noqa: F401
    except ImportError:
        return still_picture(session, NO_WIDGETS_MESSAGE)

    session.controller = Controller(session)
    return session.controller.show()

# ------------------------ guide.py ------------------------

FUNCTIONS = (explore, clusters, xor_centres, two_points, weight_table)
OPTIONS = (  # the arguments of explore, one line each
    ("net", "nn.Sequential of nn.Linear layers with nn.Tanh, nn.ReLU, nn.Sigmoid or nothing between them"),
    (
        "X, y",
        "the inputs (n x d), and the labels (0 or 1, or 0 to K-1) or the targets (task='regression': n numbers, "
        "or n x K for a network with K outputs)",
    ),
    ("groups", "a number per point, one panel per group; default: one per point up to 8 points, else per class"),
    ("eta", "the step size of a gradient step"),
    ("title", "a title above the picture"),
    ("task", "'binary', 'multiclass' or 'regression'; default: from the network's outputs and y"),
    ("params", "the weights that move: ['w10', 'w11'], 'output' or 'layer:k'; the others stay fixed"),
    ("views", "extra panels: 'backprop', 'cases', 'slice', 'line', 'hidden', 'momentum'"),
    ("case_slopes", "the weight of the 'cases' view, for example 'w10'"),
    ("slice_weights", "one or two weights for the 'slice' view"),
    ("beta, lookahead", "momentum, and the number of steps the 'momentum' view looks ahead"),
    ("log_scale", "R on a log axis"),
    ("layout, width", "'auto', 'wide' or 'standard'; the width of the figure in inches"),
    ("backprop_step", "the stage of the backward pass drawn (a step above 0 turns the view on)"),
    ("backprop_case", "the group whose case the Worked example follows ('A' or 0 is the first)"),
    ("theme", "'auto' (the editor's theme where the kernel can read it), 'light' or 'dark'"),
    ("focus", "one group ('A' or 0) drawn alone and larger; checkboxes under the controls add the others"),
    ("X_test, y_test", "test data in the shapes of X and y: drawn as crosses, R on it in the R panel and the status"),
    ("optimizer", "the rule of a step: 'gd' (the default), 'momentum' or 'adam'; passing it shows its controls"),
    ("weight_decay", "lambda: lambda * w is added to each slope before the step; R stays the loss on the data"),
    ("beta2", "Adam's share of the last squared slopes (beta is its share of the last slopes)"),
)
SESSION = (
    "step",
    "step_back",
    "reset",
    "weights",
    "risk",
    "slopes",
    "predict",
    "lookahead",
    "case_slopes",
    "save",
    "record",
    "record_backprop",
)


def first_line(obj):
    """The first sentence of an object's docstring."""
    doc = inspect.getdoc(obj) or ""
    return doc.split("\n\n")[0].replace("\n", " ").split(". ")[0].rstrip(".") + "."


def signature(name, obj, width=100):
    """name(arguments), wrapped under the opening parenthesis; a method without self."""
    sig = inspect.signature(obj)
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        sig = sig.replace(parameters=params[1:])
    text = f"{name}{sig}"
    return textwrap.fill(text, width=width, subsequent_indent=" " * (len(name) + 1), break_long_words=False)


def help(about=None):  # kept out of __all__, so that a star import leaves Python's own help alone
    """Prints the functions of slopewalk with their arguments, the options of explore and the methods of the session
    it returns. help("explore"), help("Session.step") or help(clusters) prints the full description of one
    of them."""
    if about is not None:
        print(describe(about))
        return
    lines = ["slopewalk: gradient descent by hand on small PyTorch networks.", ""]
    for f in FUNCTIONS:
        lines += [
            signature(f.__name__, f),
            textwrap.fill(first_line(f), 100, initial_indent="    ", subsequent_indent="    "),
        ]
    lines += ["", "Arguments of explore:"]
    lines += [
        textwrap.fill(text, 100, initial_indent=f"    {name:<17}", subsequent_indent=" " * 21) for name, text in OPTIONS
    ]
    lines += ["", "explore returns a Session; its methods:"]
    lines += [textwrap.indent(signature(m, getattr(Session, m), width=96), "    ") for m in SESSION]
    lines += ["    and history (R after each step), predictions, path (the weights after each step), w0 (the start)."]
    lines += ["", 'slopewalk.help("explore") or slopewalk.help("Session.step") prints the full description of one.']
    print("\n".join(lines))


def describe(about):
    """The signature and the whole docstring of a function or method, given by name or as the object."""
    if isinstance(about, str):
        names = {f.__name__: f for f in FUNCTIONS} | {f"Session.{m}": getattr(Session, m) for m in SESSION}
        names["Session"] = Session
        if about not in names:
            raise ValueError(f"slopewalk.help: {about!r} is not one of {', '.join(names)}.")
        name, obj = about, names[about]
    else:
        name, obj = getattr(about, "__qualname__", repr(about)), about
    return signature(name, obj) + "\n\n" + textwrap.indent(inspect.getdoc(obj) or "", "    ")

__all__ = [
    "explore",
    "clusters",
    "xor_centres",
    "two_points",
    "weight_table",
    "Session",
    "__version__",
]
