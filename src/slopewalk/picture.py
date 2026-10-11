"""The picture of a session: one figure whose panel positions stay fixed, refreshed for new weights.

A Frame holds the numbers for one set of weights (computed once, read by every panel). A Picture builds the figure
for the session's panels, and refresh() clears each panel and draws it again; the layout is computed once. Still
pictures use a pyplot figure; the widgets and record() use an off-screen figure that pyplot never shows.
"""

import io
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .backprop import (
    backward_pass,
    draw_derivation,
    focus,
    label_spec,
    line_styles,
    overlay,
    shared_labels,
    stage_index,
)
from .draw import (
    draw_case,
    draw_edge_labels,
    draw_history,
    draw_input_plane,
    draw_shared_note,
    draw_slope_table,
    fade,
    style_panel,
)
from .layout import BOTTOM_HEIGHT, NET_GAP, PANEL_GAP
from .views import VIEW_DRAWERS, VIEWS, needs_3d

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
