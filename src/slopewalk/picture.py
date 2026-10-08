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

from .draw import draw_case, draw_edge_labels, draw_history, draw_input_plane, draw_slope_table
from .layout import BOTTOM_HEIGHT, BOTTOM_PER_ROW
from .views import VIEW_DRAWERS, VIEWS, needs_3d

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
