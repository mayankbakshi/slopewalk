"""Drawing with matplotlib: the network panels, the glyphs, the input plane, R over the steps, the slope table.

Every panel function takes (ax, S, F): the axes, the session and the frame of numbers for the weights drawn
(see picture.py), so that no panel computes R or the slopes again.
"""

import copy
from contextlib import contextmanager

import numpy as np
import torch
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import BoxStyle, Rectangle
from matplotlib.text import Text

from .layout import BOTTOM_HEIGHT, BW, LABEL_TOLERANCE, PANEL_GAP, TABLE_ROWS, TABLE_SHARE, place_labels
from .model import SIGMOID


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
    if test_handle(S) is not None:
        handles.append(test_handle(S))
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
    """Binary, two inputs: p for class 1 over the plane, the boundary p = 0.5 and the starting boundary (dashed).
    One input: p or the prediction against x1, with the points. Regression, two inputs: the prediction over the plane.
    """
    D, T = S.data, S.theme
    w = F.w
    moved = bool(F.changed)
    X, y = D.X.numpy(), D.y.numpy()
    gx = [D.stats(X, c)[0] for c in range(D.G)]
    if D.task == "multiclass" and D.d <= 2:
        draw_class_plane(ax, S, F, gx)
        S.model.set_weights(w)
        return
    if D.task == "regression" and (S.model.n_outputs > 1 or D.d > 2):
        if D.d == 1:
            draw_target_curves(ax, S, F)
        else:  # more inputs than a plane shows, or several targets over a plane
            draw_parity(ax, S, F)
        S.model.set_weights(w)
        return
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
        S.model.set_weights(w)
        return
    if D.d != 2:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{D.d} inputs: no picture of the input plane", ha="center", va="center", color=T.muted)
        return
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
    S.model.set_weights(w)


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
    if test_handle(S) is not None:
        handles.append(test_handle(S))
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
    if test_handle(S) is not None:
        handles.append(test_handle(S))
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
