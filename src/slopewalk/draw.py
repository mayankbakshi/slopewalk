"""Drawing with matplotlib: the network panels, the glyphs, the input plane, R over the steps, the slope table."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.patches import BoxStyle, Rectangle

from .layout import BW, LABEL_TOLERANCE, TABLE_ROWS, place_labels
from .model import SIGMOID


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


def glyph(ax, xc, yc, act, m, sd, T):
    """The activation function as a small curve; a dot at m and, for groups, a band from m - sd to m + sd.

    Returns the box (x0, y0, x1, y1) it occupies, including its name below.
    """
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
    return (xc - w_ / 2, yc - h_ / 2 - 0.16, xc + w_ / 2, yc + h_ / 2)


def data_per_pixel(ax):
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0, 0), (1, 1)])
    return x1 - x0, y1 - y0


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


def draw_edge_labels(ax, S, edges, obstacles, w, changed):
    """The labels on the lines: names and values for small networks, values only for larger ones, none when
    even the values cannot be placed without overlaps (the values stay on the sliders and in the table).
    Returns the mode used: "names", "values" or "none"."""
    M, L, T = S.model, S.layout, S.theme
    try:
        renderer = ax.figure.canvas.get_renderer()
    except AttributeError:
        renderer = None
    obstacles = obstacles + box_obstacles(ax, renderer, T.edge_label_size)
    segments = [(p0, p1) for _, p0, p1, _ in edges]
    defaults = [t0 for _, _, _, t0 in edges]
    modes = ["names", "values"] if L.show_names(len(w)) else ["values"]
    for mode in modes:
        labels = [f"{M.names[k]} = {w[k]:.2f}" if mode == "names" else f"{w[k]:.2f}" for k, _, _, _ in edges]
        sizes = [text_size(ax, renderer, s, T.edge_label_size) for s in labels]
        positions, _, worst_labels, worst_boxes = place_labels(segments, sizes, obstacles, defaults, pad=0.02)
        if max(worst_labels, worst_boxes) <= LABEL_TOLERANCE:
            break
    else:
        return "none"
    for (k, _, _, _), label, (cx, cy) in zip(edges, labels, positions):
        hot = k in changed
        ax.text(
            cx,
            cy,
            label,
            fontsize=T.edge_label_size,
            ha="center",
            va="center",
            zorder=3,
            color=T.changed if hot else (T.positive if w[k] >= 0 else T.negative_text),
            fontweight="bold" if hot else "normal",
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.9),
        )
    return mode


def draw_case(ax, S, out, w, c, changed):
    """One copy of the network for group c: the lines with their weights, the units, the score, p and the loss."""
    M, D, L, T = S.model, S.data, S.layout, S.theme
    labs = D.y[D.gsel[c]]
    yi = int(round(float(labs.mean())))
    mixed = bool(labs.min() != labs.max())
    col, tint = (T.mixed_colour, T.mixed_tint) if mixed else (T.class_colour[yi], T.class_tint[yi])
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
            (zm, zs), (hm, hs) = D.stats(out.hidden[l][0], c), D.stats(out.hidden[l][1], c)
            for j, ty in enumerate(tys):
                nm = M.unit_name(l, j)
                ax.text(XZ, ty, f"z{nm}\n" + num(zm[j], zs[j]), **value)
                if layer.act:
                    ax.annotate("", xy=(XG - 0.31, ty), xytext=(XZ + BW, ty), arrowprops=arrow)
                    obstacles.append(glyph(ax, XG, ty, layer.act, zm[j], zs[j], T))
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
    (am, as_), (pm, ps), (lm, _) = D.stats(out.a, c), D.stats(out.p, c), D.stats(out.loss, c)
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
    for k, (sx, sy), (tx, ty), _ in edges:
        hot = k in changed
        ax.plot(
            [sx, tx],
            [sy, ty],
            color=T.changed if hot else (T.positive if w[k] >= 0 else T.negative),
            lw=min(0.6 + 1.3 * abs(w[k]), 7),
            zorder=1,
            solid_capstyle="round",
        )
    return draw_edge_labels(ax, S, edges, obstacles, w, changed)


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
    """A table with one row per weight: its value, the change of R when it alone grows by 0.1, and the slope dR/dw.

    The row with the steepest slope is highlighted. Above 15 rows the table continues in a second block to the right.
    """
    ax.axis("off")
    T = S.theme
    R0, g = S.risk(w), S.slopes(w)
    deltas = []
    for k in range(len(w)):
        wk = list(w)
        wk[k] += 0.1
        deltas.append(S.risk(wk) - R0)
    rows = [[S.model.names[k], f"{w[k]:+.2f}", f"{deltas[k]:+.4f}", f"{g[k]:+.3f}"] for k in range(len(w))]
    steepest = int(np.argmax(np.abs(g)))
    blocks = [rows[i : i + TABLE_ROWS] for i in range(0, len(rows), TABLE_ROWS)]
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
            colWidths=[0.17, 0.21, 0.36, 0.26],
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
            else:
                if c > 0:
                    text.set_family("monospace")
                if b * TABLE_ROWS + r - 1 == steepest:
                    cell.set_facecolor(T.table_highlight)
                    text.set_fontweight("bold")


def draw_figure(S, w, history):
    """The whole picture at w: one panel per group, then the input plane, R over the steps and the slope table."""
    M, D, L = S.model, S.data, S.layout
    changed = {k for k in range(len(w)) if abs(w[k] - S.w0[k]) > 1e-9}
    out = M.forward(D.X, D.y, w)
    per_row, rows = L.per_row(), L.rows(D.G)
    with_sd = any(sel.sum() > 1 for sel in D.gsel)  # groups of points: the boxes show mean and standard deviation
    fig = plt.figure(figsize=L.figsize(D.G, with_sd))
    gs = fig.add_gridspec(rows + 1, 3, height_ratios=[L.row_height(with_sd)] * rows + [4.2])
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
