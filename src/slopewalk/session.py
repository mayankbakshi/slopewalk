"""explore and the session behind it: the network, the points, the start weights, the history of R, the picture."""

import os

import matplotlib.pyplot as plt

from .data import prepare
from .draw import THEME, draw_figure
from .layout import Layout
from .model import Model
from .static import NO_WIDGETS_MESSAGE, STATIC_MESSAGE, still_picture


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
    from .widgets import Controller

    session.controller = Controller(session)
    return session.controller.show()
