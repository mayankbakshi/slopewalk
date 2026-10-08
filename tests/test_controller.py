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
    assert C.tabs is None  # no tabs unless the call asks for the backward pass or for extra views
    assert (
        "Try this" in C.controls.children[0].value and C.panel in C.controls.children and C.eta in C.controls.children
    )
    assert "Start here" in C.message.value and "next step, eta 0.7" in C.readout.value
    assert C.layout_pick in C.status.children
    assert isinstance(C.image, ipywidgets.Image) and C.image.value[:8] == b"\x89PNG\r\n\x1a\n"
    (root,) = C.widgets  # the default layout outside Colab: a band of controls above the full-width picture
    band, image, out = root.children
    assert list(band.children) == [C.controls, C.status] and image is C.image and out is C.out
    assert C.status.layout.width == "380px" and image.layout.width == "100%"
    assert s.fig is not None and s.fig is C.picture.fig  # drawn once at the start, off pyplot
    assert s.fig not in [plt.figure(n) for n in plt.get_fignums()]
    tooltip = getattr(C.buttons["gradient step"], "tooltip", None)
    assert tooltip and "eta" in tooltip


def test_standard_layout_puts_the_controls_above(live):
    net, X, y = logistic_start()
    C = explore(net, X, y, layout="standard").controller
    (root,) = C.widgets
    assert list(root.children) == [C.controls, C.status, C.image, C.out]


def test_layout_switch(live):
    s = explore(make("tanh3"), *xor_centres())
    C = s.controller
    assert C.layout_pick.value == "wide" and s.fig.get_size_inches()[0] == 27
    C.layout_pick.value = "standard"
    assert s.layout.style == "standard" and s.fig.get_size_inches()[0] == 15
    assert list(C.root.children)[:3] == [C.controls, C.status, C.image] and "Layout: standard" in C.message.value
    C.layout_pick.value = "wide"
    assert list(C.root.children)[0] is C.band and list(C.band.children) == [C.controls, C.status]
    assert s.fig.get_size_inches()[0] == 27


def test_messages_say_what_happened(live):
    net, X, y = logistic_start()
    s = explore(net, X, y)
    C = s.controller
    C.sliders[0].value = 0.6
    assert "You set w1 from 0.50 to 0.60: R went from 1.0032 to" in C.message.value
    C.buttons["gradient step"].click()
    p = s.predictions[-1]
    assert "One gradient step with eta 1" in C.message.value
    assert f"Predicted change {p.predicted - p.before:+.4f}, actual {p.actual - p.before:+.4f}" in C.message.value
    C.buttons["10 steps"].click()
    assert "10 gradient steps with eta 1" in C.message.value
    C.buttons["reset"].click()
    assert "Back to the start" in C.message.value and "R = 1.0032" in C.message.value
    C.eta.value = 5.0
    assert "eta is now 5" in C.message.value
    C.buttons["gradient step"].click()
    if s.history[-1] > s.history[0]:
        assert "eta may be too large" in C.message.value


def test_grouped_panel_headers(live):
    X, y, groups = xor_centres()
    boxes = explore(make("tanh3"), X, y, groups).controller.panel.children
    assert [b.children[0].value for b in boxes] == [
        "<b>unit 1</b>: weights for x1, x2, bias",
        "<b>unit 2</b>: weights for x1, x2, bias",
        "<b>unit 3</b>: weights for x1, x2, bias",
        "<b>output unit</b>: weights for h1, h2, h3, bias",
    ]
    assert [len(b.children) - 1 for b in boxes] == [3, 3, 3, 4]
    heads = [b.children[0].value for b in explore(make("two_hidden"), X, y, groups).controller.panel.children]
    assert heads[0] == "<b>layer 1, unit 1</b>: weights for x1, x2, bias"
    assert heads[3] == "<b>layer 2, unit 1</b>: weights for h1,1, h1,2, h1,3, bias"
    assert heads[-1] == "<b>output unit</b>: weights for h2,1, h2,2, bias"


def test_dropdown_above_thirty_weights(live):
    X, y, groups = xor_centres()
    C = explore(make("wide"), X, y, groups).controller
    assert len(C.sliders) == 49
    pick, holder = C.panel.children
    assert isinstance(pick, ipywidgets.Dropdown) and holder.children == (C.sliders[0],)
    assert pick.options[12] == ("w13", 12)
    pick.value = 5
    assert holder.children == (C.sliders[5],)


def test_slider_changes_the_network_and_redraws(live):
    net, X, y = logistic_start()
    s = explore(net, X, y)
    fig0, png0 = s.fig, s.controller.image.value
    s.controller.sliders[0].value = 0.6
    assert net[0].weight[0, 0].item() == pytest.approx(0.6)
    assert s.fig is fig0  # the same figure, drawn again
    assert s.controller.image.value != png0
    labels = [t.get_text() for ax in s.fig.axes for t in ax.texts if t.get_fontsize() == 8.5]
    assert "w1 = 0.60" in labels
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


def test_step_beyond_the_slider_range_widens_it(live):
    net, X, y = logistic_start()
    with torch.no_grad():
        net[0].weight[0, 0] = -3.99
    s = explore(net, X, y)
    s.controller.eta.value = 5.0
    target = -3.99 - 5.0 * s.slopes()[0]
    assert target < -4  # this step leaves the starting range of the slider
    s.controller.buttons["gradient step"].click()
    slider = s.controller.sliders[0]
    assert net[0].weight[0, 0].item() == pytest.approx(target, abs=1e-6)  # not clipped
    assert slider.value == pytest.approx(target, abs=1e-6) and slider.min <= target and slider.max == 4
    s.controller.buttons["reset"].click()
    assert (slider.min, slider.max) == (-4, 4)  # reset restores the starting range


def test_start_range_around_a_large_weight(live):
    net, X, y = logistic_start()
    with torch.no_grad():
        net[0].weight[0, 1] = 5.3
    s = explore(net, X, y)
    slider = s.controller.sliders[1]
    assert (slider.min, slider.max, slider.value) == (-4, 7, pytest.approx(5.3))


def test_subset_sliders_and_frozen_note(live):
    s = explore(make("tanh3"), *xor_centres(), params=["w10", "w11"])
    C = s.controller
    assert [sl.description for sl in C.sliders] == ["w10", "w11"]
    assert [b.children[0].value for b in C.panel.children] == ["<b>output unit</b>: weights for h1, h2"]
    assert "w1 to w9, w12 to w13" in C.frozen_note.value
    assert C.values() == s.weights() and len(C.values()) == 13


def test_steps_in_code_update_the_sliders(live):
    s = explore(make("tanh3"), *xor_centres(), eta=0.5)
    png = s.controller.image.value
    s.step(3)
    assert s.controller.values() == pytest.approx(s.weights(), abs=1e-9)
    assert s.controller.image.value != png
    s.controller.sliders[0].value += 0.2  # a slider move after the steps keeps them
    assert len(s.history) == 4 and s.weights()[1] == pytest.approx(s.controller.values()[1])


def test_readout_strip(live):
    s = explore(make("tanh3"), *xor_centres(), eta=0.5)
    C = s.controller
    assert "next step, eta 0.5: predicted change of R" in C.readout.value and "last step" not in C.readout.value
    C.buttons["gradient step"].click()
    p = s.predictions[-1]
    assert f"{p.predicted - p.before:+.4f}, {p.actual - p.before:+.4f}" in C.readout.value
    C.eta.value = 2.0
    assert s.eta == 2.0 and "next step, eta 2: predicted change" in C.readout.value


def test_tabs_appear_only_when_asked_for(live):
    X, y, groups = xor_centres()
    assert explore(make("tanh3"), X, y, groups).controller.tabs is None
    titles = lambda C: [C.tabs.get_title(i) for i in range(len(C.tabs.children))]  # noqa: E731
    C = explore(make("tanh3"), X, y, groups, views="backprop").controller
    assert titles(C) == ["1 · Weights and steps", "2 · Backprop"] and C.tabs.selected_index == 1
    C = explore(make("tanh3"), X, y, groups, views="slice").controller
    assert titles(C) == ["1 · Weights and steps", "2 · Views"] and C.tabs.selected_index == 0
    C = explore(make("tanh3"), X, y, groups, views=("backprop", "line"), case_slopes="w10").controller
    assert titles(C) == ["1 · Weights and steps", "2 · Backprop", "3 · Views"]
    assert C.view_boxes["line"].value and C.view_boxes["cases"].value and not C.view_boxes["slice"].value


def test_view_toggles_rebuild_the_picture(live):
    s = explore(make("tanh3"), *xor_centres(), views="hidden")
    C = s.controller
    n_axes = len(s.fig.axes)
    C.view_boxes["line"].value = True
    assert s.views == ("line", "hidden") and len(s.fig.axes) == n_axes + 1
    C.view_boxes["cases"].value = True
    C.case_pick.value = "w11"
    assert s.case_weight == 10 and any("along w11" in ax.get_title() for ax in s.fig.axes)
    C.slice_pick[1].value = "none"
    C.view_boxes["slice"].value = True
    assert s.slice_weights == [9] and any(ax.get_title().startswith("R over w10") for ax in s.fig.axes)
    C.log_box.value = True
    assert s.log_scale and [ax for ax in s.fig.axes if ax.get_title().startswith("R = ")][0].get_yscale() == "log"


def test_output_fallback(live, monkeypatch):
    import IPython.display

    shown = []
    monkeypatch.setenv("SLOPEWALK_DISPLAY", "output")
    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: shown.extend(a))
    monkeypatch.setattr(IPython.display, "clear_output", lambda *a, **k: None)
    s = explore(make("tanh3"), *xor_centres())
    assert s.controller.image not in s.controller.widgets
    assert any(isinstance(obj, IPython.display.Image) for obj in shown)


def test_wide_layout_puts_the_controls_in_a_band_above_the_picture(live):
    s = explore(make("tanh3"), *xor_centres(), layout="wide")
    C = s.controller
    (root,) = C.widgets
    band, image, out = root.children
    assert band is C.band and list(band.children) == [C.controls, C.status] and image is C.image
    assert image.value[:8] == b"\x89PNG\r\n\x1a\n"
