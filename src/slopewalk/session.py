"""explore and the session behind it: the network, the points, the start weights, the steps taken, the picture."""

import math
import os
from collections import namedtuple

import numpy as np

from .backprop import backprop_stages
from .data import prepare
from .draw import THEMES, as_theme
from .host import auto_theme
from .layout import Layout
from .model import Model
from .optim import Optimizer
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

    def readouts(self, eta=None, w=None):
        """The numbers of the readout strip: R, the size of the gradient of the weights that move, the accuracy
        (classification), the predicted change of R for the next step, and the last step's predicted and actual
        change."""
        eta = self.eta if eta is None else check_eta(eta)
        w = self.weights() if w is None else list(w)
        g = self.slopes(w)
        g2 = float(sum(g[k] ** 2 for k in self.active))
        dw = self.next_step(eta, w, g)
        last = self.predictions[-1] if self.predictions else None
        classify = self.task != "regression"
        out = {
            "R": self.risk(w),
            "grad_norm": math.sqrt(g2),
            "accuracy": self.model.accuracy(self.data.X, self.data.y, w) if classify else None,
            "eta": eta,
            "next_change": float(sum(g[k] * dw[k] for k in dw)),
            "last": None if last is None else (last.predicted - last.before, last.actual - last.before),
            "R_test": self.risk_test(w),
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
    from .widgets import Controller

    session.controller = Controller(session)
    return session.controller.show()
