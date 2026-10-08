"""explore and the session behind it: the network, the points, the start weights, the steps taken, the picture."""

import math
import os
from collections import namedtuple

import numpy as np

from .data import prepare
from .draw import THEME
from .layout import Layout
from .model import Model
from .picture import Picture
from .static import NO_WIDGETS_MESSAGE, STATIC_MESSAGE, still_picture
from .views import VIEWS

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
        layout="standard",
        width=None,
    ):
        self.model = Model(net)
        self.data = prepare(X, y, groups, n_inputs=self.model.n_inputs, dtype=self.model.dtype, task=task)
        self.task = self.model.task = self.data.task
        self.layout = Layout(self.model.n_inputs, self.model.hidden, self.task, style=layout, width=width)
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
    layout="standard",
    width=None,
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
    layout: "standard", or "wide" for wide screens (network panels side by side, the lower panels in one row, and the
    controls beside the picture); width: the figure width in inches.

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
        layout=layout,
        width=width,
    )
    if os.environ.get("NB_STATIC") == "1":
        return still_picture(session, STATIC_MESSAGE)
    try:
        import ipywidgets  # noqa: F401
    except ImportError:
        return still_picture(session, NO_WIDGETS_MESSAGE)
    from .widgets import Controller

    session.controller = Controller(session)
    return session.controller.show()
