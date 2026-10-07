"""The widgets, built without a front end: sliders and buttons change the network, the history and the picture."""

import copy

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn.functional as F
from helpers import make, naming_order

from slopewalk import explore, two_points, xor_centres
from slopewalk.widgets import Controller

ipywidgets = pytest.importorskip("ipywidgets")


@pytest.fixture
def live(monkeypatch):
    """explore with ipywidgets and no front end: display() does nothing, plt.show() does nothing."""
    import IPython.display

    monkeypatch.delenv("NB_STATIC", raising=False)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: None)
    yield
    plt.close("all")


def logistic_start():
    net = make("logistic")
    with torch.no_grad():
        net[0].weight[:] = torch.tensor([[0.5, 0.5]])
        net[0].bias[:] = 0.0
    X, y, _ = two_points()
    return net, X, y


def test_widgets_are_built(live):
    net, X, y = logistic_start()
    s = explore(net, X, y, eta=0.7)
    C = s.controller
    assert isinstance(C, Controller)
    assert [sl.description for sl in C.sliders] == ["w1", "w2", "w3"]
    assert [sl.value for sl in C.sliders] == [0.5, 0.5, 0.0]
    sl = C.sliders[0]
    assert (sl.min, sl.max, sl.step, sl.continuous_update, sl.readout_format) == (-4, 4, 0.001, False, ".2f")
    assert C.eta.value == 0.7 and (C.eta.min, C.eta.max, C.eta.step) == (0.1, 5, 0.1)
    assert list(C.buttons) == ["gradient step", "10 steps", "reset"]
    panel, controls, out = C.widgets
    assert isinstance(panel, ipywidgets.Box) and isinstance(out, ipywidgets.Output)
    assert list(controls.children) == [C.eta, *C.buttons.values()]
    assert s.fig is not None  # drawn once at the start


def test_grouped_panel_headers(live):
    X, y, groups = xor_centres()
    boxes = explore(make("tanh3"), X, y, groups).controller.widgets[0].children
    assert [b.children[0].value for b in boxes] == [
        "<b>unit 1</b>: weights for x1, x2, bias",
        "<b>unit 2</b>: weights for x1, x2, bias",
        "<b>unit 3</b>: weights for x1, x2, bias",
        "<b>output unit</b>: weights for h1, h2, h3, bias",
    ]
    assert [len(b.children) - 1 for b in boxes] == [3, 3, 3, 4]
    heads = [b.children[0].value for b in explore(make("two_hidden"), X, y, groups).controller.widgets[0].children]
    assert heads[0] == "<b>layer 1, unit 1</b>: weights for x1, x2, bias"
    assert heads[3] == "<b>layer 2, unit 1</b>: weights for h1,1, h1,2, h1,3, bias"
    assert heads[-1] == "<b>output unit</b>: weights for h2,1, h2,2, bias"


def test_dropdown_above_thirty_weights(live):
    X, y, groups = xor_centres()
    C = explore(make("wide"), X, y, groups).controller
    assert len(C.sliders) == 49
    pick, holder = C.widgets[0].children
    assert isinstance(pick, ipywidgets.Dropdown) and holder.children == (C.sliders[0],)
    assert pick.options[12] == ("w13", 12)
    pick.value = 5
    assert holder.children == (C.sliders[5],)


def test_slider_changes_the_network_and_redraws(live):
    net, X, y = logistic_start()
    s = explore(net, X, y)
    fig0 = s.fig
    s.controller.sliders[0].value = 0.6
    assert net[0].weight[0, 0].item() == pytest.approx(0.6)
    assert s.fig is not fig0
    assert len(s.history) == 1  # moving a slider is not a gradient step


def test_gradient_step_button(live):
    net, X, y = logistic_start()
    s = explore(net, X, y)
    g = s.slopes()
    s.controller.buttons["gradient step"].click()
    expected = [0.5 - g[0], 0.5 - g[1], 0.0 - g[2]]
    assert s.weights() == pytest.approx(expected, abs=1e-6)
    assert s.controller.values() == pytest.approx(expected, abs=1e-6)
    assert len(s.history) == 2 and s.history[1] < s.history[0]
    assert s.history[1] == pytest.approx(s.risk(expected), abs=1e-6)


def test_ten_steps_with_eta(live):
    X, y, groups = xor_centres()
    net = make("tanh3")
    ref = copy.deepcopy(net)
    s = explore(net, X, y, groups)
    s.controller.eta.value = 0.5
    s.controller.buttons["10 steps"].click()
    for _ in range(10):
        F.binary_cross_entropy_with_logits(ref(X).reshape(-1), y).backward()
        with torch.no_grad():
            for p in ref.parameters():
                p -= 0.5 * p.grad
                p.grad = None
    assert s.weights() == pytest.approx(naming_order(ref), abs=1e-5)
    assert len(s.history) == 11
    hist_ax = [ax for ax in s.fig.axes if ax.get_title().startswith("R = ")][0]
    assert len(hist_ax.lines[0].get_xdata()) == 11 and hist_ax.get_title() == f"R = {s.history[-1]:.3f}"


def test_reset_button(live):
    net, X, y = logistic_start()
    s = explore(net, X, y)
    s.controller.buttons["10 steps"].click()
    s.controller.sliders[2].value = 1.0
    s.controller.buttons["reset"].click()
    assert s.weights() == [0.5, 0.5, 0.0] and s.controller.values() == [0.5, 0.5, 0.0]
    assert s.history == [s.risk(s.w0)]


def test_step_beyond_the_slider_range_is_clipped(live):
    net, X, y = logistic_start()
    with torch.no_grad():
        net[0].weight[0, 0] = -3.99
    s = explore(net, X, y)
    s.controller.eta.value = 5.0
    assert -3.99 - 5.0 * s.slopes()[0] < -4  # this step would leave the range of the slider
    s.controller.buttons["gradient step"].click()
    assert s.controller.sliders[0].value == -4.0 and net[0].weight[0, 0].item() == -4.0
