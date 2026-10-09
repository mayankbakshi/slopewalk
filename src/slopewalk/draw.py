"""Drawing with matplotlib: the network panels, the glyphs, the input plane, R over the steps, the slope table.

Every panel function takes (ax, S, F): the axes, the session and the frame of numbers for the weights drawn
(see picture.py), so that no panel computes R or the slopes again.
"""

import copy

import numpy as np
import torch
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties
from matplotlib.patches import BoxStyle, Rectangle

from .layout import BW, LABEL_TOLERANCE, TABLE_ROWS, place_labels
from .model import SIGMOID


class Theme:
    """Colours, fonts and sizes. Red is class 1 and blue class 0.

    Only the labels on the weight lines use edge_label_size, and only the weight lines use zorder 1.
    """

    class_colour = {0: "blue", 1: "#C62828"}  # a red dark enough for text: 5.6:1 on white, 4.7:1 on its tint
    class_tint = {0: "#E5ECFF", 1: "#FFE5E5"}
    mixed_colour, mixed_tint = "black", "#EEEEEE"
    neutral_colour, neutral_tint = "0.15", "#EEEEEE"  # regression panels
    positive, negative, negative_text = "black", "#E86A00", "#B35900"  # negative lines 3.2:1 on white
    line_width, line_slope, line_max = 1.0, 1.3, 7  # a weight's line: line_width + line_slope * |w|, at most line_max
    frozen_alpha, frozen_text, frozen_fill = 0.35, "0.55", "#F5F5F5"  # weights that do not move (params)
    hidden = "green"  # the h boxes, the dot and the band on the glyphs, the history line
    arrow, curve, node_edge = "0.45", "0.35", "gray"
    start_colour, prediction_colour = "0.35", "0.45"
    gd_colour, momentum_colour = "green", "#B9852B"
    value_box = dict(boxstyle="round,pad=0.25", fc="#F2F2F2", ec="0.7", lw=0.8)
    edge_label_size, glyph_name_size, value_size, node_size, title_size, suptitle_size = 8.5, 8, 9.5, 10, 11.5, 13
    note_size = 9.5  # notes inside panels, and the lines of the backward pass
    panel_title_size, legend_size, letter_size = 10.5, 8, 11  # the panels below the networks
    plane_cmap, value_cmap = "RdBu_r", "viridis"
    table_size, table_row_height = 10, 0.058  # row height as a fraction of the panel
    table_edge, table_head, table_highlight = "0.8", "#F2F2F2", "#FFF1D6"
    backprop_colour, backprop_tint, backprop_size = "#7B1FA2", "#F3E5F5", 9  # the backward pass
    # surfaces and neutral marks: every colour of the picture lives in this class, so that a theme can change them all
    paper, ink, marker_edge, boundary = "white", "black", "k", "k"
    note_text, derivation_text, derivation_muted, point_neutral = "0.25", "0.2", "0.5", "0.25"
    glyph_fill, glyph_edge, node_fill, label_backing = "white", "0.75", "white", "white"
    reference_line, zero_line, landscape_line, plane_surface, path_colour = "0.7", "0.8", "0.35", "0.5", "0.1"
    landscape_cmap = "Greys_r"
    network_sizes = ("edge_label_size", "glyph_name_size", "value_size", "node_size", "title_size", "backprop_size")

    def scaled(self, factor):
        """A copy whose text in the network panels is `factor` times larger; the panels below keep their sizes."""
        out = copy.copy(self)
        for name in self.network_sizes:
            setattr(out, name, round(getattr(self, name) * factor, 2))
        return out


THEME = Theme()


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
    """The labels on the lines: names and values for small networks, values only for larger ones, none when
    even the values cannot be placed without overlaps (the values stay on the sliders and in the table).
    spec: {k: (long text, short text, colour[, shorter text])} replaces the labels (only those lines get one), as in
    the backward pass. Returns the mode used: "names", "values" or "none"."""
    T = S.theme
    w = F.w
    renderer = get_renderer(ax.figure)
    obstacles = obstacles + box_obstacles(ax, renderer, T.edge_label_size)
    if spec is not None:
        edges = [e for e in edges if e[0] in spec]
        if not edges:
            return "none"
    segments = [(p0, p1) for _, p0, p1, _ in edges]
    defaults = [t0 for _, _, _, t0 in edges]
    chosen = None
    for variant, texts in label_variants(S, F, edges, spec):
        sizes = [text_size(ax, renderer, s, T.edge_label_size) for s in texts]
        places, _, worst_labels, worst_boxes = place_labels(segments, sizes, obstacles, defaults, pad=0.03)
        if max(worst_labels, worst_boxes) <= LABEL_TOLERANCE:
            chosen = variant, texts, places
            break
    if chosen is None:
        return "none"
    mode, labels, positions = chosen
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
    return mode


def draw_shared_note(ax, S):
    """Network panels after the first: their lines carry no numbers, because the weights are those of panel A."""
    T = S.theme
    ax.text(0.01, 0.02, "weights as in A", transform=ax.transAxes, fontsize=T.glyph_name_size, color=T.curve)


def draw_case(ax, S, F, c, styles=None):
    """One copy of the network for group c: the lines with their weights, the units, the output and the loss.

    The labels on the lines are drawn afterwards by draw_edge_labels, once the figure's layout is fixed.
    styles: {k: (colour or None, alpha)} overrides the lines' look (the backward pass). Returns (edges, obstacles).
    """
    M, D, L, T = S.model, S.data, S.layout, S.theme
    out, w = F.out, F.w
    col, tint, mixed, yi = group_style(S, c)
    arrow = dict(arrowstyle="-|>", color=T.arrow, lw=1.0, shrinkA=0, shrinkB=0)
    node = dict(fontsize=T.node_size, ha="center", va="center", zorder=4)
    value = dict(fontsize=T.value_size, ha="center", va="center", zorder=5, bbox=T.value_box)
    edges, obstacles = [], []  # edges: (k, source, target, preferred label fraction); obstacles: boxes to keep clear
    # in the backpropagation view the names of the values are written once, in panel A; the other panels show values
    tag = (lambda name: f"{name}\n") if c == 0 or F.stage is None else (lambda name: "")
    src = [(0.12, yy) for yy in L.y_inputs]
    src_x, k = 0.0, 0
    for l, layer in enumerate(M.layers):
        last = l == len(M.layers) - 1
        tx = (L.x_a if last else L.blocks[l][0]) - BW
        tys = np.array([1.2]) if last else L.unit_ys(layer.out_features)
        bias = (src_x if l == 0 else src_x + 0.45, 2.8)
        ax.text(*bias, "1", color=T.ink, bbox=dict(boxstyle="circle", fc=T.node_fill, ec=T.node_edge), **node)
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
                ax.text(XZ, ty, tag(f"z{nm}") + num(zm[j], zs[j]), **value)
                if layer.act:
                    ax.annotate("", xy=(XG - 0.31, ty), xytext=(XZ + BW, ty), arrowprops=arrow)
                    obstacles.append(glyph(ax, XG, ty, layer.act, zm[j], zs[j], T, height=glyph_height))
                    ax.annotate("", xy=(XH - BW, ty), xytext=(XG + 0.31, ty), arrowprops=arrow)
                else:
                    ax.annotate("", xy=(XH - BW, ty), xytext=(XZ + BW, ty), arrowprops=arrow)
                ax.text(XH, ty, tag(f"h{nm}") + num(hm[j], hs[j]), color=T.hidden, **value)
            src = [(XH + BW, ty) for ty in tys]
            src_x = XH
    xm, xs = D.stats(D.X.numpy(), c)
    for i, yy in enumerate(L.y_inputs):
        ax.text(0, yy, f"x{i + 1}", color=col, bbox=dict(boxstyle="circle", fc=tint, ec=col), **node)
        ax.text(
            L.x_input_value,
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
        box_a = dict(fontsize=T.node_size, ha="center", va="center", zorder=5, bbox=T.value_box)
        ax.text(L.x_a, 1.2, tag("ŷ") + num(am, as_), **box_a)
        ax.annotate("", xy=(L.x_loss - 0.42 * L.text_scale, 1.2), xytext=(L.x_a + BW, 1.2), arrowprops=arrow)
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
        box_a = dict(fontsize=T.node_size, ha="center", va="center", zorder=5, bbox=T.value_box)
        ax.text(L.x_a, 1.2, tag("a") + num(am, as_), **box_a)
        ax.annotate("", xy=(L.x_sigmoid - 0.31, 1.2), xytext=(L.x_a + BW, 1.2), arrowprops=arrow)
        obstacles.append(glyph(ax, L.x_sigmoid, 1.2, SIGMOID, am, as_, T))
        ax.annotate("", xy=(L.x_p - BW, 1.2), xytext=(L.x_sigmoid + 0.31, 1.2), arrowprops=arrow)
        ax.text(
            L.x_p,
            1.2,
            tag("p") + num(pm, ps, sign=False),
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
        colour, alpha = weight_colour(T, k, w), T.frozen_alpha if frozen else 1.0
        if styles is not None:
            colour = styles[k][0] or weight_colour(T, k, w)
            alpha = styles[k][1] * (T.frozen_alpha if frozen else 1.0)
        ax.plot(
            [sx, tx],
            [sy, ty],
            color=colour,
            lw=min(T.line_width + T.line_slope * abs(w[k]), T.line_max),
            alpha=alpha,
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
        ax.scatter(xs[c], ys[c], s=size, color=face, edgecolor=T.marker_edge, zorder=4)
        ax.text(xs[c], ys[c], f"  {letter(c)}", fontsize=T.letter_size, va="bottom", zorder=5, path_effects=halo(T))


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
            colours = [T.class_colour[int(v)] for v in y]
            ax.scatter(X[:, 0], y, c=colours, s=18, edgecolor=T.marker_edge, lw=0.3, zorder=3)
            ax.axhline(0.5, color=T.reference_line, lw=0.8, zorder=0)
            gy = [float(D.stats(y, c)[0]) for c in range(D.G)]
            ax.set_ylim(-0.08, 1.08)
            ax.set_ylabel("p")
            ax.set_title("p for class 1 against x1", fontsize=T.panel_title_size)
        else:
            ax.scatter(X[:, 0], y, s=18, color=T.point_neutral, zorder=3)
            gy = [float(D.stats(y, c)[0]) for c in range(D.G)]
            ax.set_ylabel("y")
            ax.set_title("prediction and the points", fontsize=T.panel_title_size)
        draw_group_marks(ax, S, [float(v[0]) for v in gx], gy, size=110)
        ax.set_xlabel("x1")
        if moved:
            ax.legend(fontsize=T.legend_size, loc="best")
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
            ax.contour(P1, P2, P, levels=[0.5], colors=T.boundary)
        if moved:
            a0 = start_values(S, "plane", grid).reshape(P1.shape)
            if a0.min() < 0 < a0.max():
                ax.contour(P1, P2, a0, levels=[0.0], colors=T.start_colour, linestyles="--", linewidths=1.0)
        if D.n > D.G:
            ax.scatter(X[:, 0], X[:, 1], c=[T.class_colour[int(v)] for v in y], s=4)
        for c in range(D.G):
            yi = group_style(S, c)[3]
            ax.scatter(*gx[c], s=160, color=T.class_colour[yi], edgecolor=T.marker_edge, zorder=3)
            ax.text(gx[c][0] + 0.15, gx[c][1] + 0.15, letter(c), fontsize=T.letter_size, path_effects=halo(T))
        ax.set_title("p for class 1 and the boundary", fontsize=T.panel_title_size)
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
            ax.text(gx[c][0] + 0.15, gx[c][1] + 0.15, letter(c), fontsize=T.letter_size, path_effects=halo(T))
        ax.set_title("prediction over the input plane; points coloured by y", fontsize=T.panel_title_size)
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
        ax.plot(ks, vs, "o", mfc=T.paper, mec=T.prediction_colour, ms=6, ls="none", label="first-order prediction")
        ax.legend(fontsize=T.legend_size, loc="lower left")
    ax.set_xlabel("gradient steps taken")
    ax.set_ylabel("R")
    values = list(history) + [v for _, v in predicted]
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


def width_share(ax):
    """The panel's share of the width of its row, from the width ratios of its grid: the same for every frame,
    before and after the layout."""
    spec = ax.get_subplotspec()
    if spec is None:
        return 1 / 3
    grid = spec.get_gridspec()
    ratios = list(grid.get_width_ratios() or [1] * grid.ncols)
    return sum(ratios[spec.colspan.start : spec.colspan.stop]) / sum(ratios)


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
    share = width_share(ax)
    short = n_blocks > 1 or ax.figure.get_figwidth() * share < 4.8  # narrow panels and two blocks: short headers
    fontsize = T.table_size if not short else max(T.table_size - 2, 8)
    headers = (
        ["w", "value", "R, +0.1", "slope"] if short else ["weight", "value", "change of R for +0.1", "slope dR/dw"]
    )
    ax.set_title("Change of R when one weight grows by 0.1,\nand the slope dR/dw", fontsize=T.panel_title_size)
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
            cell.set_facecolor(T.paper)
            text = cell.get_text()
            text.set_color(T.ink)
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
