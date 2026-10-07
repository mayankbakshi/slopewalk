"""slopewalk 0.1.0.dev0: gradient descent by hand on small PyTorch networks, in one file.

Built by tools/make_single_file.py from the package at https://github.com/mayankbakshi/slopewalk; edit the package,
not this file. Same use as the package:

    net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    X, y, groups = clusters([(-1, 1), (1, 1), (-1, -1), (1, -1)], [1, 0, 0, 1])
    explore(net, X, y, groups)
"""

import os
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle

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

Weight = namedtuple("Weight", "name layer unit input value")
Values = namedtuple("Values", "hidden a p loss")  # hidden: [(z, h), ...] per hidden layer; a, p and loss per point


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
            "slopewalk: the network must end with nn.Linear(..., 1) (the score of class 1), without an activation."
        )
    return layers


class Model:
    """The parsed network.

    The weights are one list in the order of their names: layer by layer, unit by unit, each unit's incoming weights
    in the order of its inputs, then its bias. For nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)) this
    gives w1 to w3 (unit 1), w4 to w6, w7 to w9, w10 to w12 (output weights) and w13 (output bias).
    """

    def __init__(self, net):
        self.net = net
        self.layers = parse_layers(net)
        self.linears = [L.linear for L in self.layers]
        self.n_inputs = self.layers[0].in_features
        self.hidden = [L.out_features for L in self.layers[:-1]]
        self.n_weights = sum((L.in_features + 1) * L.out_features for L in self.layers)
        self.names = [f"w{k + 1}" for k in range(self.n_weights)]

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
            p = torch.sigmoid(a)
            loss = F.binary_cross_entropy_with_logits(a, y, reduction="none")
        return Values(hidden, a.numpy(), p.numpy(), loss.numpy())

    def risk(self, X, y, w=None):
        """R: the average log loss over all points, as F.binary_cross_entropy_with_logits."""
        if w is not None:
            self.set_weights(w)
        with torch.no_grad():
            return float(F.binary_cross_entropy_with_logits(self.net(X).reshape(-1), y))

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
        R = F.binary_cross_entropy_with_logits(self.net(X).reshape(-1), y)
        grads = torch.autograd.grad(R, params)
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


def weight_table(net):
    """One row per weight: name, layer (from 1), unit (from 1), input ("x1", "h2", "h1,2" or "bias") and value."""
    return Model(net).weight_table()

# ------------------------ data.py ------------------------

XOR_CENTRES = [(-1, 1), (1, 1), (-1, -1), (1, -1)]  # A, B, C, D: top left, top right, bottom left, bottom right
XOR_LABELS = [1, 0, 0, 1]


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


class Data:
    """The points of one explore call: X (n x d), y (n), groups (n), and which points belong to each group."""

    def __init__(self, X, y, groups):
        self.X, self.y, self.groups = X, y, groups
        self.n, self.d = X.shape
        self.gids = sorted(set(groups.tolist()))
        self.G = len(self.gids)
        self.gsel = [(groups == g).numpy() for g in self.gids]

    def stats(self, arr, c):
        """Mean and standard deviation over the points of group c of the rows of arr; the sd is 0 for a single point."""
        sel = arr[self.gsel[c]]
        return sel.mean(0), (sel.std(0) if sel.shape[0] > 1 else 0 * sel.mean(0))


def prepare(X, y, groups=None):
    """The tensors of an explore call. groups None: one group per point (up to 8 points), else one per class."""
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32).flatten()
    if X.dim() == 1:
        X = X[None, :]
    n = X.shape[0]
    if groups is None:
        groups = torch.arange(n) if n <= 8 else y.long()
    return Data(X, y, torch.as_tensor(groups).long())

# ------------------------ layout.py ------------------------

BW = 0.27  # half-width of a value box; lines end this far from the centre of the box
TABLE_ROWS = 15  # rows per column of the slope table


class Layout:
    """Positions for a network with n_inputs inputs and the hidden layers `hidden` (a list of unit counts)."""

    def __init__(self, n_inputs, hidden):
        self.n_inputs, self.hidden = n_inputs, hidden
        self.blocks, x = [], 1.75
        for _ in hidden:
            self.blocks.append((x, x + 0.75, x + 1.5))  # x of the z box, of the glyph and of the h box
            x += 2.75
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

    def figsize(self, n_panels):
        return (15, 2.9 * self.rows(n_panels) + 4.2)

    @staticmethod
    def table_fontsize(n_weights):
        return 9.5 if n_weights > 15 else 11

# ------------------------ draw.py ------------------------

class Theme:
    """Colours, fonts and sizes. Red is class 1 and blue class 0."""

    class_colour = {0: "blue", 1: "red"}
    class_tint = {0: "#E5ECFF", 1: "#FFE5E5"}
    mixed_colour, mixed_tint = "black", "#EEEEEE"
    positive, negative, negative_text, changed = "black", "tab:orange", "#B35900", "tab:blue"
    hidden = "green"  # the h boxes, the dot and the band on the glyphs, the history line
    arrow, curve, node_edge = "0.45", "0.35", "gray"
    value_box = dict(boxstyle="round,pad=0.25", fc="#F2F2F2", ec="0.7", lw=0.8)
    edge_label_size, glyph_name_size, value_size, node_size, title_size, suptitle_size = 8.5, 8, 9.5, 10, 11.5, 13
    plane_cmap, points_cmap = "RdBu_r", "bwr"


THEME = Theme()


def letter(c):
    """A, B, C, ... for the groups."""
    return chr(65 + c) if c < 26 else str(c)


def num(m, sd, sign=True):
    """A value, with its standard deviation on a second line when it is not zero."""
    s = f"{m:+.2f}" if sign else f"{m:.2f}"
    return s + (f"\n±{sd:.2f}" if np.any(sd > 0) else "")


def glyph(ax, xc, yc, act, m, sd, T):
    """The activation function as a small curve; a dot at m and, for groups, a band from m - sd to m + sd."""
    name, f, lim = act
    w_, h_ = 0.62, 0.5
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


def draw_case(ax, S, out, w, c, changed):
    """One copy of the network for group c: the lines with their weights, the units, the score, p and the loss."""
    M, D, L, T = S.model, S.data, S.layout, S.theme
    labs = D.y[D.gsel[c]]
    yi = int(round(float(labs.mean())))
    mixed = bool(labs.min() != labs.max())
    col, tint = (T.mixed_colour, T.mixed_tint) if mixed else (T.class_colour[yi], T.class_tint[yi])
    arrow = dict(arrowstyle="-|>", color=T.arrow, lw=1.0, shrinkA=0, shrinkB=0)
    show_names = L.show_names(len(w))
    src = [(0.12, yy) for yy in L.y_inputs]
    src_x, k = 0.0, 0
    for l, layer in enumerate(M.layers):
        last = l == len(M.layers) - 1
        tx = (L.x_a if last else L.blocks[l][0]) - BW
        tys = np.array([1.2]) if last else L.unit_ys(layer.out_features)
        bias = (src_x if l == 0 else src_x + 0.45, 2.8)
        ax.text(
            *bias,
            "1",
            fontsize=T.node_size,
            ha="center",
            va="center",
            zorder=4,
            bbox=dict(boxstyle="circle", fc="white", ec=T.node_edge),
        )
        sources = src + [(bias[0] + 0.1, bias[1] - 0.1)]
        ns, nt = len(sources), len(tys)
        for j, ty in enumerate(tys):
            for si, (sx, sy) in enumerate(sources):
                hot = k in changed
                t = L.label_fraction(si, j, ns, nt)
                ax.plot(
                    [sx, tx],
                    [sy, ty],
                    color=T.changed if hot else (T.positive if w[k] >= 0 else T.negative),
                    lw=min(0.6 + 1.3 * abs(w[k]), 7),
                    zorder=1,
                    solid_capstyle="round",
                )
                ax.text(
                    sx + t * (tx - sx),
                    sy + t * (ty - sy),
                    f"{M.names[k]} = {w[k]:.2f}" if show_names else f"{w[k]:.2f}",
                    fontsize=T.edge_label_size,
                    ha="center",
                    va="center",
                    zorder=3,
                    color=T.changed if hot else (T.positive if w[k] >= 0 else T.negative_text),
                    fontweight="bold" if hot else "normal",
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.9),
                )
                k += 1
        if not last:
            XZ, XG, XH = L.blocks[l]
            (zm, zs), (hm, hs) = D.stats(out.hidden[l][0], c), D.stats(out.hidden[l][1], c)
            for j, ty in enumerate(tys):
                nm = M.unit_name(l, j)
                ax.text(
                    XZ,
                    ty,
                    f"z{nm}\n" + num(zm[j], zs[j]),
                    fontsize=T.value_size,
                    ha="center",
                    va="center",
                    zorder=5,
                    bbox=T.value_box,
                )
                if layer.act:
                    ax.annotate("", xy=(XG - 0.31, ty), xytext=(XZ + BW, ty), arrowprops=arrow)
                    glyph(ax, XG, ty, layer.act, zm[j], zs[j], T)
                    ax.annotate("", xy=(XH - BW, ty), xytext=(XG + 0.31, ty), arrowprops=arrow)
                else:
                    ax.annotate("", xy=(XH - BW, ty), xytext=(XZ + BW, ty), arrowprops=arrow)
                ax.text(
                    XH,
                    ty,
                    f"h{nm}\n" + num(hm[j], hs[j]),
                    fontsize=T.value_size,
                    ha="center",
                    va="center",
                    zorder=5,
                    color=T.hidden,
                    bbox=T.value_box,
                )
            src = [(XH + BW, ty) for ty in tys]
            src_x = XH
    xm, xs = D.stats(D.X.numpy(), c)
    for i, yy in enumerate(L.y_inputs):
        ax.text(
            0,
            yy,
            f"x{i + 1}",
            fontsize=T.node_size,
            ha="center",
            va="center",
            zorder=4,
            color=col,
            bbox=dict(boxstyle="circle", fc=tint, ec=col),
        )
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
    (am, as_), (pm, ps), (lm, _) = D.stats(out.a, c), D.stats(out.p, c), D.stats(out.loss, c)
    ax.text(
        L.x_a, 1.2, "a\n" + num(am, as_), fontsize=T.node_size, ha="center", va="center", zorder=5, bbox=T.value_box
    )
    ax.annotate("", xy=(L.x_sigmoid - 0.31, 1.2), xytext=(L.x_a + BW, 1.2), arrowprops=arrow)
    glyph(ax, L.x_sigmoid, 1.2, SIGMOID, am, as_, T)
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
        f"{ytxt}\nloss {lm:.3f}\nchange\nfrom start\n{lm - S.loss0[c]:+.3f}",
        fontsize=T.node_size,
        ha="center",
        va="center",
        color=col,
        bbox=dict(boxstyle="round,pad=0.3", fc=tint, ec=col, lw=0.8),
    )
    where = "(" + ", ".join(f"{v:.1f}" for v in xm) + ")"
    what = "classes mixed" if mixed else f"class {yi}"
    npts = int(D.gsel[c].sum())
    ax.set_title(
        f"{letter(c)}: {'mean of ' + str(npts) + ' points ' if npts > 1 else ''}{where}, {what}:  loss {lm:.3f}",
        fontsize=T.title_size,
        color=col,
    )
    ax.set_xlim(*L.xlim)
    ax.set_ylim(*L.ylim)
    ax.axis("off")


def draw_input_plane(ax, S):
    """p for class 1 over the input plane, the boundary p = 0.5, the points and the group means (two inputs only)."""
    D, M, T = S.data, S.model, S.theme
    if D.d != 2:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{D.d} inputs: no picture of the input plane", ha="center", va="center")
        return
    X, y = D.X, D.y
    lo, hi = min(-2.0, float(X.min()) - 0.5), max(2.0, float(X.max()) + 0.5)
    u = np.linspace(lo, hi, 101)
    P1, P2 = np.meshgrid(u, u)
    with torch.no_grad():
        grid = torch.tensor(np.c_[P1.ravel(), P2.ravel()], dtype=torch.float32)
        P = torch.sigmoid(M.net(grid)).numpy().reshape(P1.shape)
    ax.contourf(P1, P2, P, levels=np.linspace(0, 1, 11), cmap=T.plane_cmap, alpha=0.5)
    if P.min() < 0.5 < P.max():
        ax.contour(P1, P2, P, levels=[0.5], colors="k")
    if D.n > D.G:
        ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap=T.points_cmap, s=4, vmin=0, vmax=1)
    for c in range(D.G):
        xm = X[D.gsel[c]].mean(0)
        yi = int(round(float(y[D.gsel[c]].mean())))
        ax.scatter(*xm.numpy(), s=160, color=T.class_colour[yi], edgecolor="k", zorder=3)
        ax.text(xm[0] + 0.15, xm[1] + 0.15, letter(c), fontsize=12)
    ax.set_title("p for class 1 and the boundary")
    ax.set_aspect("equal")


def draw_history(ax, history, R, T):
    """R over the gradient steps taken."""
    ax.plot(range(len(history)), history, "o-", color=T.hidden, ms=4)
    ax.set_xlabel("gradient steps taken")
    ax.set_ylabel("R")
    ax.set_ylim(0, max(1.1, max(history) * 1.05))
    ax.set_title(f"R = {R:.3f}")
    if len(history) < 12:
        ax.set_xticks(range(len(history)))


def draw_slope_table(ax, S, w):
    """For each weight: the change of R for +0.1, and the slope dR/dw from R.backward()."""
    ax.axis("off")
    R0, g, lines = S.risk(w), S.slopes(w), []
    for k in range(len(w)):
        wk = list(w)
        wk[k] += 0.1
        lines.append(f"{S.model.names[k]:>4}: {S.risk(wk) - R0:+.4f}   {g[k]:+.3f}")
    for ci, m0 in enumerate(range(0, len(lines), TABLE_ROWS)):
        head = "      +0.1 in w:   slope\n      change of R\n" if ci == 0 else "\n\n"
        ax.text(
            0.52 * ci,
            1.0,
            head + "\n".join(lines[m0 : m0 + TABLE_ROWS]),
            fontsize=S.layout.table_fontsize(len(w)),
            family="monospace",
            va="top",
            transform=ax.transAxes,
        )


def draw_figure(S, w, history):
    """The whole picture at w: one panel per group, then the input plane, R over the steps and the slope table."""
    M, D, L = S.model, S.data, S.layout
    changed = {k for k in range(len(w)) if abs(w[k] - S.w0[k]) > 1e-9}
    out = M.forward(D.X, D.y, w)
    per_row, rows = L.per_row(), L.rows(D.G)
    fig = plt.figure(figsize=L.figsize(D.G))
    gs = fig.add_gridspec(rows + 1, 3, height_ratios=[1] * rows + [1.45])
    for c in range(D.G):
        r, cc = divmod(c, per_row)
        ax = fig.add_subplot(gs[r, :] if per_row == 1 else gs[r, :].subgridspec(1, 2)[0, cc])
        draw_case(ax, S, out, w, c, changed)
    if S.title:
        fig.suptitle(S.title, fontsize=S.theme.suptitle_size)
    draw_input_plane(fig.add_subplot(gs[rows, 0]), S)
    draw_history(fig.add_subplot(gs[rows, 1]), history, S.risk(w), S.theme)
    draw_slope_table(fig.add_subplot(gs[rows, 2]), S, w)
    M.set_weights(w)
    fig.tight_layout()
    return fig

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

class Controller:
    """Builds the widgets for a session and keeps them, the network and the picture in step."""

    def __init__(self, session):
        import ipywidgets as widgets

        S = self.session = session
        self.sliders = [
            widgets.FloatSlider(
                value=v,
                min=-4,
                max=4,
                step=0.001,
                description=S.names[k],
                readout_format=".2f",
                continuous_update=False,
                layout=widgets.Layout(width="230px"),
            )
            for k, v in enumerate(S.w0)
        ]
        panel = self.grouped_panel(widgets) if len(self.sliders) <= 30 else self.dropdown_panel(widgets)
        self.eta = widgets.FloatSlider(value=S.eta, min=0.1, max=5, step=0.1, description="eta", readout_format=".1f")
        self.buttons = {name: widgets.Button(description=name) for name in ("gradient step", "10 steps", "reset")}
        self.out = widgets.Output()
        self.busy = False
        for s in self.sliders:
            s.observe(self.redraw, names="value")
        self.buttons["gradient step"].on_click(lambda _: self.steps(1))
        self.buttons["10 steps"].on_click(lambda _: self.steps(10))
        self.buttons["reset"].on_click(lambda _: self.reset())
        self.widgets = [panel, widgets.HBox([self.eta, *self.buttons.values()]), self.out]

    def grouped_panel(self, widgets):
        """Up to 30 weights: the sliders in one box per unit, with the names of the unit's inputs."""
        M, boxes, k = self.session.model, [], 0
        for layer, L in enumerate(M.layers):
            for j in range(L.out_features):
                head = widgets.HTML(
                    f"<b>{M.unit_title(layer, j)}</b>: weights for {', '.join(M.input_labels(layer))}, bias"
                )
                boxes.append(widgets.VBox([head] + self.sliders[k : k + L.in_features + 1]))
                k += L.in_features + 1
        return widgets.Box(boxes, layout=widgets.Layout(flex_flow="row wrap"))

    def dropdown_panel(self, widgets):
        """More than 30 weights: choose one weight, then move it."""
        pick = widgets.Dropdown(
            options=[(nm, i) for i, nm in enumerate(self.session.names)], value=0, description="weight"
        )
        holder = widgets.HBox([self.sliders[0]])
        pick.observe(lambda ch: setattr(holder, "children", [self.sliders[ch["new"]]]), names="value")
        return widgets.HBox([pick, holder])

    def values(self):
        return [s.value for s in self.sliders]

    def set_values(self, w):
        for s, v in zip(self.sliders, w):
            s.value = float(np.clip(v, s.min, s.max))

    def redraw(self, *_):
        if self.busy:
            return
        with self.out:
            self.out.clear_output(wait=True)
            self.session.draw(self.values())
            plt.show()

    def steps(self, m):
        """m gradient steps from the slider values with the current eta; the sliders then show the new weights."""
        self.busy = True
        w = self.session.step(m, self.eta.value, self.values())
        self.set_values(w)
        self.busy = False
        self.redraw()

    def reset(self):
        """Back to the weights at the start, with an empty history."""
        self.busy = True
        w = self.session.reset()
        for s, v in zip(self.sliders, w):
            s.value = v
        self.busy = False
        self.redraw()

    def show(self):
        from IPython.display import display

        display(*self.widgets)
        self.redraw()
        return self.session

# ------------------------ session.py ------------------------

class Session:
    """What explore returns.

    weights(), risk(), slopes(), step(k), reset() work on the network itself; save(path) writes the current picture.
    history holds R at the start and after every step; w0 the weights at the start; controller the widgets or None.
    """

    def __init__(self, net, X, y, groups=None, eta=1.0, title=None, theme=None):
        self.model = Model(net)
        self.data = prepare(X, y, groups)
        self.layout = Layout(self.model.n_inputs, self.model.hidden)
        self.theme = theme or THEME
        self.title, self.eta = title, eta
        self.w0 = self.model.get_weights()
        start = self.model.forward(self.data.X, self.data.y)
        self.loss0 = [self.data.stats(start.loss, c)[0] for c in range(self.data.G)]
        self.history = [self.risk(self.w0)]
        self.fig = None
        self.controller = None

    @property
    def names(self):
        return self.model.names

    def weights(self):
        return self.model.get_weights()

    def risk(self, w=None):
        """R at the weights w (default: the current ones). Sets the network to w."""
        return self.model.risk(self.data.X, self.data.y, w)

    def slopes(self, w=None):
        """dR/dw at the weights w (default: the current ones). Sets the network to w."""
        return self.model.slopes(self.data.X, self.data.y, w)

    def step(self, k=1, eta=None, w=None):
        """k gradient steps, w <- w - eta * dR/dw, from w (default: the current weights). Returns the new weights."""
        eta = self.eta if eta is None else eta
        w = list(self.weights() if w is None else w)
        for _ in range(k):
            g = self.slopes(w)
            w = [wk - eta * gk for wk, gk in zip(w, g)]
            self.history.append(self.risk(w))
        return w

    def reset(self):
        """Back to the weights at the start. Returns them."""
        self.model.set_weights(self.w0)
        self.history = [self.risk(self.w0)]
        return list(self.w0)

    def draw(self, w=None, history=None):
        """Draws the picture at w (default: the current weights) as a new figure and keeps it as self.fig."""
        w = self.weights() if w is None else w
        self.fig = draw_figure(self, w, self.history if history is None else history)
        return self.fig

    def save(self, path, dpi=120):
        """Saves the picture at the current weights to path (PNG, PDF or SVG by its extension)."""
        fig = draw_figure(self, self.weights(), self.history)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _ipython_display_(self):
        pass  # the picture and the controls are shown by explore itself

    def __repr__(self):
        return f"<slopewalk session: {len(self.names)} weights, R = {self.risk():.3f}>"


def explore(net, X, y, groups=None, eta=1.0, title=None):
    """Gradient descent by hand on `net`, with inputs X (n x d) and classes y (n, 0 or 1).

    net: nn.Sequential of nn.Linear layers with nn.Tanh, nn.ReLU, nn.Sigmoid or nothing between them, ending in
    nn.Linear(..., 1), whose output is the score a of class 1. R is the average log loss over all points.
    groups: a number per point; one panel per group. None: one panel per point (up to 8 points), else one per class.
    eta: the step size of the gradient step buttons (a slider).

    Draws one copy of the network per group (every value a mean ± standard deviation over the group), with each
    weight on its line, the units as z, activation and h, then a, the sigmoid, p and the loss with its change from the
    start. Below: the boundary p = 0.5 in the input plane (two inputs), R over the gradient steps taken, and for each
    weight the change of R for +0.1 and the slope dR/dw from R.backward().
    With ipywidgets: a slider per weight, eta, `gradient step`, `10 steps`, `reset`. The sliders change the weights of
    `net` itself. Without ipywidgets, or with NB_STATIC=1, a still picture of the start.
    Returns a Session: save(path) writes the current picture; step(k), reset(), weights(), history.
    """
    session = Session(net, X, y, groups, eta, title)
    if os.environ.get("NB_STATIC") == "1":
        return still_picture(session, STATIC_MESSAGE)
    try:
        import ipywidgets  # noqa: F401
    except ImportError:
        return still_picture(session, NO_WIDGETS_MESSAGE)

    session.controller = Controller(session)
    return session.controller.show()

__all__ = ["explore", "clusters", "xor_centres", "two_points", "weight_table", "Session", "__version__"]
